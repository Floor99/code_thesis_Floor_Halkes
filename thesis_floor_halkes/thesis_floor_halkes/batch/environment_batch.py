import torch
from torch_geometric.data import Batch, Data


class BatchedAmbulanceEnvDynamic:
    """
    Batched environment for ambulance routing with dynamic waiting times and light status.
    Processes a batch of graphs in parallel, preserving the core logic of AmbulanceEnvDynamic.

    API:
        obs_batch, current_nodes, valid_actions = env.reset()
        obs_batch, current_nodes, valid_actions, rewards, dones, infos = env.step(actions)

    Inputs:
      - batch: PyG Batch of Data objects, concatenating multiple graphs
      - start_nodes: LongTensor[B] of global start-node indices (use batch.ptr + per-graph local start)
      - end_nodes:   LongTensor[B] of global end-node indices
    """

    def __init__(
        self,
        batch: Batch,
        start_nodes: torch.LongTensor,
        end_nodes: torch.LongTensor,
        goal_bonus: float = 20.0,
        dead_end_penalty: float = 10.0,
        max_wait: float = 30.0,
        revisit_penalty: float = 1.0,
        step_penalty: float = 0.1,
        wait_weight: float = 1.0,
    ):
        # Store graph-batch and parameters
        self.batch = batch
        self.edge_index = batch.edge_index
        self.edge_attr = batch.edge_attr
        self.batch_vector = batch.batch  # maps each node to its graph in the batch
        self.ptr = batch.ptr  # index pointers for each graph
        self.num_graphs = batch.num_graphs
        assert start_nodes.numel() == self.num_graphs
        assert end_nodes.numel() == self.num_graphs
        self.start_nodes = start_nodes.to(batch.x.device)
        self.end_nodes = end_nodes.to(batch.x.device)
        self.device = batch.x.device

        # Static features
        #   traffic_lights: bool mask over all nodes
        self.traffic_lights = batch.x[:, 0].to(torch.bool)
        #   edge travel times: length / speed (converted to m/s)
        lengths = batch.edge_attr[:, 0]
        speeds_kmh = batch.edge_attr[:, 1]
        speeds_ms = speeds_kmh / 3.6
        self.edge_travel_times = lengths / speeds_ms

        # Adjacency: map each global node index -> list of (neighbor_idx, edge_idx)
        N_total = self.traffic_lights.size(0)
        self.adj = {i: [] for i in range(N_total)}
        for eidx, (u, v) in enumerate(batch.edge_index.t().tolist()):
            self.adj[u].append((v, eidx))

        # RL parameters
        self.goal_bonus = goal_bonus
        self.dead_end_penalty = dead_end_penalty
        self.max_wait = max_wait
        self.revisit_penalty = revisit_penalty
        self.step_penalty = step_penalty
        self.wait_weight = wait_weight

        # Initialize dynamic state
        self.reset()

    def reset(self):
        """
        Reset all B environments: set current_nodes to start, sample fresh dynamics.

        Returns:
            obs_batch: Data  (batched PyG Data)
            current_nodes: LongTensor[B]
            valid_actions: List[List[int]] of length B
        """
        # Per-graph current node pointers (global indices)
        self.current_nodes = self.start_nodes.clone()
        # Track which episodes are done
        self.done = torch.zeros(self.num_graphs, dtype=torch.bool, device=self.device)

        # Sample dynamic features across all nodes
        bits = torch.randint(
            0, 2, (self.traffic_lights.size(0),), dtype=torch.bool, device=self.device
        )
        self.light_status = bits & self.traffic_lights
        self.waiting_times = (
            torch.rand(self.traffic_lights.size(0), device=self.device) * self.max_wait
        )

        # Visited sets per graph (global indices)
        self.visited = [
            {int(self.current_nodes[i].item())} for i in range(self.num_graphs)
        ]

        return self._get_obs()

    def _get_obs(self):
        # Build the batched observation Data
        x = torch.stack(
            [
                self.traffic_lights.to(torch.float),
                self.light_status.to(torch.float),
                self.waiting_times,
            ],
            dim=1,
        )

        # new
        obs_batch = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            batch=self.batch_vector,  # ← this is the crucial bit
        )

        # Compute valid actions per graph
        valid_actions = []
        for i in range(self.num_graphs):
            cur = int(self.current_nodes[i].item())
            valid_actions.append([v for v, _ in self.adj[cur]])
        # Return new obs, current_nodes, and valid_actions mask
        return obs_batch, self.current_nodes.clone(), valid_actions

    def step(self, actions: torch.LongTensor):
        """
        Execute one step on all B graph-envs in parallel.

        Args:
            actions: LongTensor[B] of _global_ node-indices chosen for each graph
        Returns:
            obs_batch, current_nodes, valid_actions, rewards, dones, infos
        """
        actions = actions.to(self.device)
        rewards = torch.zeros(self.num_graphs, dtype=torch.float, device=self.device)
        infos = [{} for _ in range(self.num_graphs)]

        # Iterate per-graph to compute rewards and update state
        for i in range(self.num_graphs):
            if self.done[i]:
                continue
            action = int(actions[i].item())
            cur = int(self.current_nodes[i].item())
            # Validate action
            neighbors = [v for v, _ in self.adj[cur]]
            if action not in neighbors:
                raise ValueError(
                    f"Invalid action {action} in graph {i} from node {cur}."
                )
            # Travel time
            eidx = next(idx for (v, idx) in self.adj[cur] if v == action)
            travel_t = float(self.edge_travel_times[eidx].item())
            # Waiting time
            wait_t = 0.0
            if self.traffic_lights[action] and not self.light_status[action]:
                wait_t = float(self.waiting_times[action].item())
            # Reward calc
            r = -(travel_t + self.wait_weight * wait_t) / 10.0
            r -= self.step_penalty
            if action in self.visited[i]:
                r -= self.revisit_penalty
            # Update node & visited
            self.current_nodes[i] = action
            self.visited[i].add(action)

            # Re-sample dynamic features globally
            bits = torch.randint(
                0,
                2,
                (self.traffic_lights.size(0),),
                dtype=torch.bool,
                device=self.device,
            )
            self.light_status = bits & self.traffic_lights
            self.waiting_times = (
                torch.rand(self.traffic_lights.size(0), device=self.device)
                * self.max_wait
            )

            # Terminal checks
            if action == int(self.end_nodes[i].item()):
                self.done[i] = True
                r += self.goal_bonus
            else:
                # dead-end if all neighbors visited
                no_moves = all(v in self.visited[i] for v, _ in self.adj[action])
                if no_moves:
                    self.done[i] = True
                    r -= self.dead_end_penalty
            rewards[i] = r

        # Get next observation
        obs_batch, curr_nodes, valid_actions = self._get_obs()
        return obs_batch, curr_nodes, valid_actions, rewards, self.done.clone(), infos

    def get_num_graphs(self):
        return self.num_graphs

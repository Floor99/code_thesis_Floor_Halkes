import torch
from torch_geometric.loader import DataLoader
from graph_data_batch import RandomGraphDataset
from environment_batch import BatchedAmbulanceEnvDynamic
from agent_batch import AmbulanceAgent
from model_batch import PolicyNetworkGATDynamicAttention
from reinforce_baselines_batch import CriticBaseline, NoBaseline, ExponentialBaseline
import time


def train(
    dataset: RandomGraphDataset,
    hidden_size: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 10,
    epochs: int = 2,
    max_steps: int = 50,
    use_baseline: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate policy and agent
    sample_data = next(iter(loader))
    in_static = sample_data.x.size(1) - dataset.gen_kwargs.get("dyn_dim", 2)
    in_dyn = dataset.gen_kwargs.get("dyn_dim", 2)
    num_nodes = None  # dynamic per-graph, decoder handles masking

    model_kwargs = {
        "static_layers": 2,
        "static_heads": 4,
        "dyn_heads": 1,
        "dropout": 0.2,
        "edge_attr_dim": 2,
    }

    policy = PolicyNetworkGATDynamicAttention(
        in_static=in_static, in_dyn=in_dyn, hidden_size=hidden_size, **model_kwargs
    ).to(device)

    # baseline = CriticBaseline(policy, lr=lr, gamma=gamma) if use_baseline else NoBaseline()
    baseline = NoBaseline()
    agent = AmbulanceAgent(policy, baseline=baseline, lr=lr, gamma=gamma)

    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        total_actor = 0.0
        total_batches = 0

        for batch in loader:
            batch = batch.to(device)
            # print(f"batch {batch}")
            # Extract start/end per graph
            start_nodes = batch.start_node
            end_nodes = batch.end_node
            # print(f"batch 0 {batch[0]}")

            # Initialize batched environment
            env = BatchedAmbulanceEnvDynamic(
                batch=batch,
                start_nodes=start_nodes,
                end_nodes=end_nodes,
                # only graph‐generation args here:
                max_wait=graph_kwargs["max_wait"],
                # plus whatever reward‐shaping args you want:
                goal_bonus=20.0,
                dead_end_penalty=10.0,
                revisit_penalty=1.0,
                step_penalty=0.1,
                wait_weight=1.0,
            )

            # Reset env and agent caches
            obs, current_nodes, valid_actions = env.reset()
            policy.clear_static_cache()
            # agent.clear_static_cache()

            # Rollout
            for t in range(max_steps):
                actions, logp = agent.select_action(obs, current_nodes, valid_actions)
                obs, current_nodes, valid_actions, rewards, dones, _ = env.step(actions)
                agent.store_reward(rewards)
                if dones.all():
                    break

            # Update
            actor_loss, baseline_loss = agent.finish_episode()
            total_actor += actor_loss
            total_loss += actor_loss + baseline_loss
            total_batches += 1
            # print(f"batch done")

        avg_loss = total_loss / total_batches
        avg_actor = total_actor / total_batches
        print(
            f"Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f} | "
            f"Actor Loss: {avg_actor:.4f} | Time: {time.time() - epoch_start:.1f}s"
        )

    print("Training complete.")


if __name__ == "__main__":
    # Example usage
    graph_kwargs = {
        "min_nodes": 10,
        "max_nodes": 35,
        "min_prob": 0.1,
        "max_prob": 0.5,
        "max_wait": 30.0,
        "min_length": 100.0,
        "max_length": 1000.0,
        "min_speed": 30.0,
        "max_speed": 100.0,
    }
    dataset = RandomGraphDataset(50, **graph_kwargs)
    train(dataset)

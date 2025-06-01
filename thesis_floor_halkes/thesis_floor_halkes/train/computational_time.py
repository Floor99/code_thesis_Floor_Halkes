import datetime
import math
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
import mlflow
import os
from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_

from thesis_floor_halkes.environment.dynamic_ambulance import DynamicEnvironment
from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetterDataFrame

from thesis_floor_halkes.features.static.final_getter import StaticDataObjectSet
from thesis_floor_halkes.model.decoder import AttentionDecoder, FixedContext
from thesis_floor_halkes.model.encoders import StaticGATEncoder, DynamicGATEncoder
from thesis_floor_halkes.penalties.calculator import RewardModifierCalculator
from thesis_floor_halkes.penalties.revisit_node_penalty import (
    AggregatedStepPenalty,
    CloserToGoalBonus,
    DeadEndPenalty,
    GoalBonus,
    HigherSpeedBonus,
    NoSignalIntersectionPenalty,
    PenaltyPerStep,
    RevisitNodePenalty,
    WaitTimePenalty,
)
from thesis_floor_halkes.baselines.critic_network import CriticBaseline

from thesis_floor_halkes.agent.dynamic import DynamicAgent
from thesis_floor_halkes.utils.action_masking import masking_function_manager
from thesis_floor_halkes.utils.episode import finish_episode
from joblib.externals.loky.backend.context import get_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # For testing purposes, use CPU

print(f"Using device: {device}")


# torch.autograd.set_detect_anomaly(True)

from hydra.core.hydra_config import HydraConfig


# ==== Training Loop with MLFlow Tracking ====
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs("checkpoints", exist_ok=True)
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")

    train_base_dir = cfg.data.train_path
    val_base_dir = cfg.data.val_path
    test_base_dir = cfg.data.test_path

    train_set = StaticDataObjectSet(root=train_base_dir, processed_file_names=[cfg.data.data_file])
    # train_set = train_set[:2]

    val_set = StaticDataObjectSet(root=val_base_dir, processed_file_names=[cfg.data.data_file])
    # val_set = val_set[:2]

    test_set = StaticDataObjectSet(root=test_base_dir, processed_file_names=[cfg.data.data_file])
    # test_set = test_set[:2]
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        # num_workers = 2,
        # prefetch_factor=4,
        # multiprocessing_context=get_context('loky'),
        # persistent_workers=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        # num_workers = 2,
        # prefetch_factor=4,
        # multiprocessing_context=get_context('loky'),
        # persistent_workers=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        # num_workers = 2,
        # prefetch_factor=4,
        # multiprocessing_context=get_context('loky'),
        # persistent_workers=True
    )

    dynamic_node_idx = {
        "status": 0,
        "wait_time": 1,
        "current_node": 2,
        "visited_nodes": 3,
    }

    static_node_idx = {
        "lat": 0,
        "lon": 1,
        "has_light": 2,
        "dist_to_goal": 3,
    }

    static_edge_idx = {
        "length": 0,
        "speed": 1,
    }

    # ==== Reward Modifiers ====
    penalty_per_step = PenaltyPerStep(
        name="Penalty Per Step", penalty=cfg.reward_mod.penalty_per_step_value
    )
    goal_bonus = GoalBonus(name="Goal Bonus", bonus=cfg.reward_mod.goal_bonus_value)
    dead_end_penalty = DeadEndPenalty(
        name="Dead End Penalty", penalty=cfg.reward_mod.dead_end_penalty_value
    )
    waiting_time_penalty = WaitTimePenalty(name="Waiting Time Penalty")
    higher_speed_bonus = HigherSpeedBonus(
        name="Higher Speed Bonus", bonus=cfg.reward_mod.higher_speed_bonus_value
    )
    closer_to_goal_bonus = CloserToGoalBonus(
        name="Closer To Goal Bonus", bonus=cfg.reward_mod.closer_to_goal_bonus_value
    )
    no_signal_intersection_penalty = NoSignalIntersectionPenalty(
        name="No Signal Intersection Penalty",
        penalty=cfg.reward_mod.no_signal_intersection_penalty_value,
    )

    reward_modifier_calculator = RewardModifierCalculator(
        modifiers=[
            penalty_per_step,
            goal_bonus,
            waiting_time_penalty,
            dead_end_penalty,
            higher_speed_bonus,
            closer_to_goal_bonus,
            no_signal_intersection_penalty,
        ],
        weights=[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )

    action_masking_funcs = masking_function_manager(cfg)

    # ==== Environment and Agent ====
    env = DynamicEnvironment(
        static_dataset=train_set,
        dynamic_feature_getter=DynamicFeatureGetterDataFrame(),
        reward_modifier_calculator=reward_modifier_calculator,
        max_steps=cfg.training.max_steps,
        # start_timestamp="2024-01-31 08:30:00",
        dynamic_node_idx=dynamic_node_idx,
        static_node_idx=static_node_idx,
        static_edge_idx=static_edge_idx,
        action_mask_funcs=action_masking_funcs,
    )

    encoder_output_dim = cfg.stat_enc.out_size
    # static_encoder = StaticGATEncoder(
    #     in_channels=cfg.stat_enc.in_channels,
    #     hidden_size=cfg.stat_enc.hidden_size,
    #     edge_attr_dim=2,
    #     num_layers=cfg.stat_enc.num_layers,
    #     heads=cfg.stat_enc.num_heads,
    #     dropout=cfg.stat_enc.dropout,
    #     out_size=encoder_output_dim,
    # ).to(device)
    static_encoder = torch.load("saved_models/dynamic_ambulance_training/2025-05-31_00:57:27/optuna_trial_13/static_encoder.pth", weights_only=False).to(device)
    dynamic_encoder = torch.load("saved_models/dynamic_ambulance_training/2025-05-31_00:57:27/optuna_trial_13/dynamic_encoder.pth", weights_only=False).to(device)
    decoder = torch.load("saved_models/dynamic_ambulance_training/2025-05-31_00:57:27/optuna_trial_13/decoder.pth", weights_only=False).to(device)
    baseline = torch.load("saved_models/dynamic_ambulance_training/2025-05-31_00:57:27/optuna_trial_13/baseline.pth", weights_only=False).to(device)
    # dynamic_encoder = DynamicGATEncoder(
    #     in_channels=cfg.dyn_enc.in_channels,
    #     hidden_size=cfg.dyn_enc.hidden_size,
    #     num_layers=cfg.dyn_enc.num_layers,
    #     heads=cfg.dyn_enc.num_heads,
    #     dropout=cfg.dyn_enc.dropout,
    #     out_size=encoder_output_dim,
    # ).to(device)

    # decoder = AttentionDecoder(
    #     embed_dim=encoder_output_dim * 2, num_heads=cfg.decoder.num_heads
    # ).to(device)
    fixed_context = FixedContext(embed_dim=encoder_output_dim * 2).to(device)
    # baseline = CriticBaseline(
    #     encoder_output_dim * 2, hidden_dim=cfg.baseline.hidden_size, hidden_layers=cfg.baseline.num_layers
    # ).to(device)

    gamma = cfg.reinforce.discount_factor
    agent = DynamicAgent(
        static_encoder=static_encoder,
        dynamic_encoder=dynamic_encoder,
        decoder=decoder,
        fixed_context=fixed_context,
        baseline=baseline,
    )

    use_joint_optimization = True  # or False

    if use_joint_optimization and baseline is not None:
        # One optimizer for all trainable components
        optimizer = torch.optim.Adam(
            [
                {
                    "params": agent.static_encoder.parameters(),
                    "lr": cfg.decoder.learning_rate,
                },
                {
                    "params": agent.dynamic_encoder.parameters(),
                    "lr": cfg.decoder.learning_rate,
                },
                {
                    "params": agent.decoder.parameters(),
                    "lr": cfg.decoder.learning_rate},
                {
                    "params": agent.baseline.parameters(),
                    "lr": cfg.decoder.learning_rate,
                },
            ]
        )
    else:
        # Separate optimizers
        policy_optimizer = torch.optim.Adam(
            [
                {
                    "params": agent.static_encoder.parameters(),
                    "lr": cfg.stat_enc.learning_rate,
                },
                {
                    "params": agent.dynamic_encoder.parameters(),
                    "lr": cfg.dyn_enc.learning_rate,
                },
                {"params": agent.decoder.parameters(), "lr": cfg.decoder.learning_rate},
            ]
        )
        if baseline is not None:
            baseline_optimizer = torch.optim.Adam(
                agent.baseline.parameters(), lr=cfg.baseline.learning_rate
            )
    from hydra.types import RunMode

    is_sweep = HydraConfig.get().mode == RunMode.MULTIRUN
    job_num = HydraConfig.get().job.num if is_sweep else None
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
    with mlflow.start_run(
        run_name=f"optuna_trial_{job_num}" if is_sweep else "single_run",
        nested=bool(parent_run_id),
    ) as run:
        if parent_run_id:
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
        mlflow.set_tag("is_sweep", str(is_sweep))
        param_overrides = HydraConfig.get().overrides.task
        param_overrides = [param.split("=") for param in param_overrides]
        for k, v in param_overrides:
            if k.startswith("hydra."):
                continue
            if isinstance(v, str) and v.isdigit():
                v = int(v)
            elif isinstance(v, str) and v.replace(".", "", 1).isdigit():
                v = float(v)
            mlflow.log_param(k, v)

        best_val_epoch_score = -1
        early_stopping_counter = cfg.training.patience
        
        train_times = []
        val_times = []
        test_times = []
        
        for epoch in range(cfg.training.num_epochs):
            if early_stopping_counter <= 0:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. Best validation score: {best_val_epoch_score}. Best epoch: {epoch - cfg.training.patience + 1}"
                )
                break
            print(f"Job:{job_num} == Epoch {epoch + 1}/{cfg.training.num_epochs}")
            batch_infos = []
            agent.static_encoder.eval()
            agent.dynamic_encoder.eval()
            agent.decoder.eval()
            agent.baseline.eval() if baseline is not None else None

            train_start_time = datetime.datetime.now()
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                episode_infos = []
                skip_episode = False
                with torch.no_grad():
                    for episode in range(batch.num_graphs):
                        # print(
                        #     f"Episode {episode + 1}/{batch.num_graphs} in batch {batch_idx + 1}/{len(train_loader)}"
                        # )
                        static_data = batch.get_example(episode)
                        # print(f"Static data.x shape: {static_data.x.shape}")
                        static_data.start_node = int(static_data.start_node.item())
                        static_data.end_node = int(static_data.end_node.item())
                        env.static_data = static_data
                        state = env.reset()
                        agent.reset()

                        step_info = {}

                        for step in range(cfg.training.max_steps):
                            # print(f"Step {step + 1}/{cfg.training.max_steps}")

                            if len(env.states[-1].valid_actions) == 0:
                                print(
                                    f"Graph {episode} has no valid actions. Skipping episode."
                                )
                                skip_episode = True
                                break

                            next_state, terminated, truncated, step_info = execute_step(
                                env, agent, state, step_info
                            )
                            state = next_state
                            if terminated or truncated:
                                break

                        if skip_episode:
                            continue
            train_end_time = datetime.datetime.now()
            # train_time = (train_end_time - train_start_time).total_seconds()
            # train_times.append(train_time)

            agent.static_encoder.eval()
            agent.dynamic_encoder.eval()
            agent.decoder.eval()
            agent.baseline.eval() if baseline is not None else None
            batch_infos = []
            with torch.no_grad():
                val_start_time = datetime.datetime.now()
                for batch_idx, batch in enumerate(val_loader):
                    batch = batch.to(device)
                    episode_infos = []
                    skip_episode = False

                    for episode in range(batch.num_graphs):
                        # print(
                        #     f"Validation Episode {episode + 1}/{batch.num_graphs} in batch {batch_idx + 1}/{len(val_loader)}"
                        # )
                        static_data = batch.get_example(episode)
                        static_data.start_node = int(static_data.start_node.item())
                        static_data.end_node = int(static_data.end_node.item())
                        env.static_data = static_data
                        state = env.reset()
                        agent.reset()

                        step_info = {}

                        for step in range(cfg.training.max_steps):
                            # print(f"Step {step + 1}/{cfg.training.max_steps}")

                            if len(env.states[-1].valid_actions) == 0:
                                print(
                                    f"Graph {episode} has no valid actions. Skipping episode."
                                )
                                skip_episode = True
                                break

                            next_state, terminated, truncated, step_info = execute_step(
                                env, agent, state, step_info, greedy=True
                            )
                            state = next_state
                            if terminated or truncated:
                                break

                        if skip_episode:
                            continue
                val_end_time = datetime.datetime.now()

            
            agent.static_encoder.eval()
            agent.dynamic_encoder.eval()
            agent.decoder.eval()
            agent.baseline.eval() if baseline is not None else None
            batch_infos = []
            with torch.no_grad():
                test_start_time = datetime.datetime.now()
                for batch_idx, batch in enumerate(test_loader):
                    batch = batch.to(device)
                    episode_infos = []
                    skip_episode = False

                    for episode in range(batch.num_graphs):
                        # print(
                        #     f"Test Episode {episode + 1}/{batch.num_graphs} in batch {batch_idx + 1}/{len(test_loader)}"
                        # )
                        static_data = batch.get_example(episode)
                        static_data.start_node = int(static_data.start_node.item())
                        static_data.end_node = int(static_data.end_node.item())
                        env.static_data = static_data
                        state = env.reset()
                        agent.reset()

                        step_info = {}

                        for step in range(cfg.training.max_steps):
                            # print(f"Step {step + 1}/{cfg.training.max_steps}")

                            if len(env.states[-1].valid_actions) == 0:
                                print(
                                    f"Graph {episode} has no valid actions. Skipping episode."
                                )
                                skip_episode = True
                                break

                            next_state, terminated, truncated, step_info = execute_step(
                                env, agent, state, step_info, greedy=True
                            )
                            state = next_state
                            if terminated or truncated:
                                break

                        if skip_episode:
                            continue
                test_end_time = datetime.datetime.now()
        
        train_time = (train_end_time - train_start_time).total_seconds()
        val_time = (val_end_time - val_start_time).total_seconds()
        test_time = (test_end_time - test_start_time).total_seconds()
        
        print(f"Train time: {train_time} seconds")
        print(f"Validation time: {val_time} seconds")
        print(f"Test time: {test_time} seconds")
            
    return best_val_epoch_score


def locally_save_agent(agent, experiment_name: str, parent_run:str, child_run:str):
    # create directory if it doesn't exist
    os.makedirs(f"saved_models/{experiment_name}/{parent_run}/{child_run}", exist_ok=True)
    torch.save(agent.static_encoder, f"saved_models/{experiment_name}/{parent_run}/{child_run}/static_encoder.pth")
    torch.save(agent.dynamic_encoder, f"saved_models/{experiment_name}/{parent_run}/{child_run}/dynamic_encoder.pth")
    torch.save(agent.decoder, f"saved_models/{experiment_name}/{parent_run}/{child_run}/decoder.pth")
    torch.save(agent.baseline, f"saved_models/{experiment_name}/{parent_run}/{child_run}/baseline.pth")


def score_function(success_rates:list, penalized_travel_times:list, success_coeff:float, penalized_travel_time_coeff:float, min_time:float=0, max_time:float = 1000) -> float:
    #min max scale penalized_travel_time to [0, 1]
    penalized_travel_times = np.array(penalized_travel_times)
    penalized_travel_times = (penalized_travel_times - min_time) / (max_time - min_time)
    penalized_travel_times = np.clip(penalized_travel_times, 0, 1)
    
    # Calculate the score as a weighted sum of success rate and penalized travel time
    score = (
        success_coeff * np.mean(success_rates)
        - penalized_travel_time_coeff * np.mean(penalized_travel_times)
    )
    return score
    
def mlflow_log_epoch_metrics(
    epoch_info: dict,
    epoch_step_id: int,
    prefix: str = None,
    exclude_metrics: list = None,
):
    if exclude_metrics is not None:
        assert isinstance(exclude_metrics, list), "exclude_metrics should be a list"
        for metric in exclude_metrics:
            if metric in epoch_info:
                del epoch_info[metric]
            else:
                raise KeyError(f"Metric {metric} not found in epoch_info")
    if prefix is None:
        prefix = "EPOCH_"
    else:
        prefix = f"{prefix}_EPOCH_"

    del epoch_info["penalized_travel_time_for_each_batch"]
    epoch_info = {f"{prefix}{k}": v for k, v in epoch_info.items()}
    mlflow.log_metrics(
        epoch_info,
        step=epoch_step_id,
    )


def record_epoch_info(batch_infos: list, cfg: DictConfig):
    epoch_info = {}
    epoch_info["mean_policy_loss"] = torch.mean(
        torch.stack([batch_info["mean_policy_loss"] for batch_info in batch_infos])
    )
    epoch_info["mean_baseline_loss"] = torch.mean(
        torch.stack([batch_info["mean_baseline_loss"] for batch_info in batch_infos])
    )
    epoch_info["mean_entropy_loss"] = torch.mean(
        torch.stack([batch_info["mean_entropy_loss"] for batch_info in batch_infos])
    )
    epoch_info["mean_total_loss"] = torch.mean(
        torch.stack([batch_info["mean_total_loss"] for batch_info in batch_infos])
    )
    epoch_info["mean_reward"] = np.mean(
        [batch_info["mean_reward"] for batch_info in batch_infos]
    )
    epoch_info["success_rate"] = np.mean(
        [batch_info["success_rate"] for batch_info in batch_infos]
    )
    epoch_info["mean_travel_time"] = torch.mean(
        torch.stack([batch_info["mean_travel_time"] for batch_info in batch_infos])
    )
    epoch_info["penalized_travel_time_for_each_batch"] = [
        batch_info["mean_travel_time_with_penalty"]
        for batch_info in batch_infos
    ]
    epoch_info["scoring"] = score_function(
        success_rates=[batch_info["success_rate"] for batch_info in batch_infos],
        penalized_travel_times=epoch_info["penalized_travel_time_for_each_batch"],
        success_coeff=cfg.score.success_coeff,
        penalized_travel_time_coeff=cfg.score.travel_time_coeff,
        min_time=cfg.score.min_time,
        max_time=cfg.score.max_time,
    )
    return epoch_info


def execute_step(env, agent, state, step_info, greedy=False):
    action, log_prob, entropy = agent.select_action(state, greedy=greedy)
    baseline_value = agent.baseline(agent.final_embedding)
    next_state, reward, terminated, truncated, _ = env.step(action)

    step_info = record_step_info(
        step_info,
        env,
        next_state,
        action,
        log_prob,
        entropy,
        baseline_value,
        reward,
        terminated,
    )

    return next_state, terminated, truncated, step_info


def record_batch_info(episode_infos):
    batch_info = {}
    batch_info["mean_total_loss"] = torch.mean(
        torch.stack(
            [
                info["total_loss"]
                for info in episode_infos
                if info["total_loss"] is not None
            ]
        )
    )
    batch_info["mean_policy_loss"] = torch.mean(
        torch.stack(
            [
                info["policy_loss"]
                for info in episode_infos
                if info["policy_loss"] is not None
            ]
        )
    )
    batch_info["mean_baseline_loss"] = torch.mean(
        torch.stack(
            [
                info["baseline_loss"]
                for info in episode_infos
                if info["baseline_loss"] is not None
            ]
        )
    )
    batch_info["mean_entropy_loss"] = torch.mean(
        torch.stack(
            [
                info["entropy_loss"]
                for info in episode_infos
                if info["entropy_loss"] is not None
            ]
        )
    )
    batch_info["mean_reward"] = np.mean(
        [np.sum(info["rewards"]) for info in episode_infos]
    )
    batch_info["success_rate"] = np.mean(
        [1 if info["reached_goal"] else 0 for info in episode_infos]
    )
    batch_info["mean_travel_time"] = torch.mean(
        torch.stack(
            [
                torch.sum(torch.tensor(info["step_travel_time_route"]))
                for info in episode_infos
            ]
        )
    )
    
    # if not reached goal, travel time is 1000, else travel time
    batch_info["mean_travel_time_with_penalty"] = torch.mean(
        torch.stack(
            [
                torch.sum(torch.tensor(info["step_travel_time_route"]))
                if info["reached_goal"]
                else torch.tensor(1000.0)
                for info in episode_infos
            ]
        )
    ).clone().detach().cpu().item()
    
    return batch_info


def mlflow_log_batch_metrics(
    batch_info, batch_step_id, prefix=None, exclude_metrics: list = None
):
    if exclude_metrics is not None:
        assert isinstance(exclude_metrics, list), "exclude_metrics should be a list"
        for metric in exclude_metrics:
            if metric in batch_info:
                del batch_info[metric]
            else:
                raise KeyError(f"Metric {metric} not found in batch_info")

    if prefix is None:
        prefix = "BATCH_"
    else:
        prefix = f"{prefix}_BATCH_"

    batch_info = {
        "policy_loss": batch_info["mean_policy_loss"],#.clone().detach().cpu().item(),
        "baseline_loss": batch_info["mean_baseline_loss"],#.clone().detach().cpu().item(),
        "entropy_loss": batch_info["mean_entropy_loss"],#.clone().detach().cpu().item(),
        "total_loss": batch_info["mean_total_loss"],#.clone().detach().cpu().item(),
        "reward": batch_info["mean_reward"],
        "success_rate": batch_info["success_rate"],
    }
    batch_info = {f"{prefix}{k}": v for k, v in batch_info.items()}

    mlflow.log_metrics(
        batch_info,
        step=batch_step_id,
    )


def mlflow_log_episode_metrics(
    episode_info,
    epoch_id,
    batch_id,
    episode_id,
    graph_id,
    step_id,
    prefix=None,
    exclude_metrics: list = None,
):
    if prefix is None:
        prefix = "EPISODE_"
    else:
        prefix = f"{prefix}_EPISODE_"

    episode_metrics = {
        f"total_loss": episode_info["total_loss"].clone().detach().cpu().item(),
        f"policy_loss": episode_info["policy_loss"].clone().detach().cpu().item(),
        f"baseline_loss": episode_info["baseline_loss"].clone().detach().cpu().item(),
        f"entropy_loss": episode_info["entropy_loss"].clone().detach().cpu().item(),
        f"reached_goal": int(episode_info["reached_goal"]),
        f"num_steps": len(episode_info["rewards"]),
        f"reward": sum(episode_info["rewards"]),
    }
    if exclude_metrics is not None:
        assert isinstance(exclude_metrics, list), "exclude_metrics should be a list"
        remaining_metrics_keys = set(episode_metrics.keys()) - set(exclude_metrics)
        remaining_metrics = {k: episode_metrics[k] for k in remaining_metrics_keys}
        remaining_metrics = {f"{prefix}{k}": v for k, v in remaining_metrics.items()}
        mlflow.log_metrics(
            remaining_metrics,
            step=step_id,
        )
    else:
        # If no metrics are excluded, log all metrics with the prefix

        episode_metrics_with_prefix = {f"{prefix}{k}": v for k, v in episode_metrics.items()}

        mlflow.log_metrics(
            episode_metrics_with_prefix,
            step=step_id,
        )

    table = pd.DataFrame(episode_info[f"penalty_contributions"])
    table["policy_loss"] = episode_metrics[f"policy_loss"]
    table["baseline_loss"] = episode_metrics[f"baseline_loss"]
    table["entropy_loss"] = episode_metrics[f"entropy_loss"]
    table["total_loss"] = episode_metrics[f"total_loss"]
    table["total_reward"] = episode_metrics[f"reward"]
    table["reached_goal"] = episode_metrics[f"reached_goal"]
    table["num_steps"] = episode_metrics[f"num_steps"]
    table["route"] = episode_info[f"route"]
    table["success"] = 1 if episode_info[f"reached_goal"] else 0
    table["action_log_probs"] = [info.clone().detach().cpu().item() for info in episode_info["action_log_probs"]]
    table["entropies"] = [info.clone().detach().cpu().item() for info in episode_info["entropies"]]
    table["baseline_values"] = [info.clone().detach().cpu().item() for info in episode_info["baseline_values"]]
    table["rewards"] = episode_info["rewards"]
    table["advantages"] = [info.clone().detach().cpu().item() for info in episode_info["advantages"]]
    table["discounted_returns"] = [info.clone().detach().cpu().item() for info in episode_info["discounted_returns"]]
    table["step_travel_time_route"] = [info.clone().detach().cpu().item() for info in episode_info["step_travel_time_route"]]
    
    mlflow.log_table(
        data=table,
        artifact_file=f"{prefix}epoch_{epoch_id}/batch_{batch_id}/episode_{episode_id}_{graph_id}.json",
    )


def record_episode_info(
    step_info, total_loss, policy_loss, baseline_loss, entropy_loss, advantages, discounted_returns
):
    episode_info = {}
    episode_info["total_loss"] = total_loss
    episode_info["policy_loss"] = policy_loss
    episode_info["baseline_loss"] = baseline_loss
    episode_info["entropy_loss"] = entropy_loss
    episode_info["discounted_returns"]  = discounted_returns
    episode_info["advantages"] = advantages

    episode_info = episode_info | step_info
    return episode_info


def record_step_info(
    step_info, env, state, action, log_prob, entropy, baseline_value, reward, terminated
):
    if not step_info:
        step_info = {
            "rewards": [],
            "action_log_probs": [],
            "entropies": [],
            "baseline_values": [],
            "route": [],
            "states": [],
            "step_travel_time_route": [],
            "penalty_contributions": [],
        }
    step_info["rewards"].append(reward.clone().detach().cpu().item())
    step_info["action_log_probs"].append(log_prob)
    step_info["entropies"].append(entropy)
    step_info["baseline_values"].append(baseline_value)
    step_info["route"].append(action)
    step_info["states"].append(state)
    step_info["reached_goal"] = True if terminated else False
    step_info["step_travel_time_route"].append(env.step_travel_time_route[-1])
    step_info["penalty_contributions"].append(env.modifier_contributions)
    return step_info


if __name__ == "__main__":
    sweep_id = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    os.environ["SWEEP_ID"] = sweep_id

    # Always assume we're in single run; actual check is inside `main()`
    experiment = mlflow.set_experiment("computational_time_experiment")
    with mlflow.start_run(run_name=f"{sweep_id}") as parent_run:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run.info.run_id
        os.environ["MLFLOW_PARENT_RUN_NAME"] = parent_run.info.run_name
        os.environ["MLFLOW_PARENT_RUN_EXPERIMENT_NAME"] = experiment.name
        main()

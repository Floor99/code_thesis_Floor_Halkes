import mlflow


def log_episode_artifacts(episode_info, step_logs, epoch, batch_idx, graph_idx, fig):
    graph_folder = f"epoch_{epoch:03}/batch_{batch_idx:03}/graph_{graph_idx:02}"
    
    # Log main info and figure
    mlflow.log_dict(episode_info, f"{graph_folder}/info.json")
    mlflow.log_figure(fig, f"{graph_folder}/graph.png")

    # Log all step information in one file
    mlflow.log_dict(step_logs, f"{graph_folder}/steps.json")


def log_agent_checkpoint(agent, epoch):
    mlflow.pytorch.log_state_dict(
        {
            "static_encoder": agent.static_encoder.state_dict(),
            "dynamic_encoder": agent.dynamic_encoder.state_dict(),
            "decoder": agent.decoder.state_dict(),
            # "baseline": agent.baseline.state_dict(),
        },
        artifact_path=f"epoch_{epoch:03}/agent",
    )


def log_full_models(agent):
    mlflow.pytorch.log_model(agent.static_encoder, "static_encoder_model")
    mlflow.pytorch.log_model(agent.dynamic_encoder, "dynamic_encoder_model")
    mlflow.pytorch.log_model(agent.decoder, "decoder_model")
    # mlflow.pytorch.log_model(agent.baseline, "baseline_model")


def log_gradient_norms(module, module_name, step_id):
    for name, param in module.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            mlflow.log_metric(
                f"grad_norm/{module_name}/{name}", grad_norm, step=step_id
            )


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_nested(cfg, keys):
    for key in keys:
        cfg = cfg[key]
    return cfg


import glob
import os
import yaml
import mlflow

def log_latest_best_params_to_mlflow(multirun_dir="multirun", experiment_name="dynamic_ambulance_training"):
    """
    Finds the latest optimization_results.yaml in the multirun directory and logs best params to MLflow.
    """
    # Search for all optimization_results.yaml files in the multirun directory
    result_files = glob.glob(os.path.join(multirun_dir, "*", "*", "optimization_results.yaml"))
    if not result_files:
        print("No optimization_results.yaml files found.")
        return

    # Sort by modification time, newest last
    result_files.sort(key=os.path.getmtime)
    latest_result = result_files[-1]
    print(f"Using latest optimization_results.yaml: {latest_result}")

    with open(latest_result, "r") as f:
        results = yaml.safe_load(f)

    best_params = results.get("best_params", results)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="best_params"):
        mlflow.log_params(best_params)
        if "best_value" in results:
            mlflow.log_metric("best_value", results["best_value"])
        mlflow.set_tag("type", "best_params")
    print(f"Logged best parameters to MLflow from {latest_result}")
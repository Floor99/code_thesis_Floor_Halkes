import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # For testing purposes, use CPU



def finish_episode(
    rewards,
    action_log_probs,
    entropies,
    baseline_weight=0.5,
    baseline_values=None,
    gamma=0.99,
    entropy_coeff=0.01
):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, device=device)
    # if returns.numel() > 1:
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    if baseline_values is not None:
        baseline_values = torch.stack(baseline_values).to(device)

        advantages = (returns - baseline_values).detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        baseline_loss = F.mse_loss(baseline_values, returns)
    else:
        print("No baseline model provided. Using returns as advantages.")
        advantages = returns
        baseline_loss = 0

    log_probs_tensor = torch.stack(action_log_probs).to(device)
    policy_loss = -(log_probs_tensor * advantages)
    policy_loss = -(log_probs_tensor * advantages).mean()
    entropy_loss = torch.stack(entropies).mean()
    policy_loss = policy_loss - entropy_coeff * entropy_loss
    total_loss = policy_loss + (baseline_weight * baseline_loss)
    return total_loss, policy_loss, baseline_loss, entropy_loss, advantages, returns

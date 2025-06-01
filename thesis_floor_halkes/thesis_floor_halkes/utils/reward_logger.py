import matplotlib.pyplot as plt
import numpy as np


class RewardLogger:
    def __init__(self, smooth_window=50):
        self.total_rewards = []
        self.rewards_per_node = []
        # self.successes = []
        self.smooth_window = smooth_window

    def log(self, reward, num_nodes, success: bool = None):
        self.total_rewards.append(reward)
        self.rewards_per_node.append(reward / num_nodes)
        # self.successes.append(1 if success else 0)

    def plot(self):
        def smooth(x):
            return np.convolve(
                x, np.ones(self.smooth_window) / self.smooth_window, mode="valid"
            )

        plt.figure(figsize=(12, 6))

        # Raw reward
        plt.subplot(1, 2, 1)
        plt.plot(self.total_rewards, alpha=0.3, label="Total reward")
        plt.plot(smooth(self.total_rewards), label="Smoothed")
        plt.title("Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward (e.g. -cost)")
        plt.grid(True)
        plt.legend()

        # Reward per node
        plt.subplot(1, 2, 2)
        plt.plot(self.rewards_per_node, alpha=0.3, label="Reward / Node")
        plt.plot(smooth(self.rewards_per_node), label="Smoothed")
        plt.title("Reward per Node")
        plt.xlabel("Episode")
        plt.ylabel("Reward / Node")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("reward_plot.png")

    def summary(self):
        recent = slice(-self.smooth_window, None)
        # success_rate = np.mean(self.successes[recent]) * 100
        avg_reward = np.mean(self.total_rewards[recent])
        avg_per_node = np.mean(self.rewards_per_node[recent])

        print(f"Last {self.smooth_window} episodes:")
        print(f"  Avg total reward:     {avg_reward:.2f}")
        print(f"  Avg reward / node:    {avg_per_node:.4f}")
        # print(f"  Success rate:         {success_rate:.1f}%")

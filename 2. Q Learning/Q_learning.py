# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay,
                     alpha, gamma, q_table_save_path="q_table.npy", max_steps=500):

    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0

        for step in range(max_steps):  # <- Limit the number of steps
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)
            total_reward += reward

            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )

            state = next_state

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    np.save(q_table_save_path, q_table)
    print("Training finished. Q-table saved.")


# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(hell_state_coordinates=[(2, 1), (0, 4)],
                      goal_coordinates=(4, 4),
                      actions=["Right" , "Left", "Down", "Up"],
                      q_values_path="q_table.npy"):

    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
        # --------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask the goal state's Q-value for visualization:
            # ------------------------------------------------
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates] = True
            for h in hell_state_coordinates:
                if 0 <= h[0] < heatmap_data.shape[0] and 0 <= h[1] < heatmap_data.shape[1]:
                    mask[h] = True
                    ax.text(h[1] + 0.5, h[0] + 0.5, 'H', color='red',
                            ha='center', va='center', weight='bold', fontsize=14)

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})
            ax.invert_yaxis()

            # Denote Goal and Hell states:
            # ----------------------------
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[0][1] + 0.5, hell_state_coordinates[0][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[1][1] + 0.5, hell_state_coordinates[1][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")

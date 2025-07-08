# Imports:
# --------
from padm_env import create_env
from Q_learning import train_q_learning, visualize_q_table
from datetime import datetime
import numpy as np
import time

# User definitions:
# -----------------
train = False
test = True
visualize_results = True

"""
NOTE: Sometimes a fixed initializtion might push the agent to a local minimum.
In this case, it is better to use a random initialization.  
"""
random_initialization = True  # If True, the Q-table will be initialized randomly

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 1_000  # Number of episodes

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
q_table_filename = f"q_table_lr{learning_rate}_eps{epsilon}_gamma{gamma}_{timestamp}.npy"

env = create_env(
    goal_coordinates=(14, 14),
    random_initialization=False
)

goal_coordinates = tuple(env.goal_state)
hell_state_coordinates = [
    tuple(cell)
    for snake in env.obstacles
    for cell in snake
]

def test_agent(env, q_table_path, max_steps=500, delay=0.2):
    q_table = np.load(q_table_path)
    state, _ = env.reset()
    state = tuple(state)
    total_reward = 0

    for step in range(max_steps):
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(delay)

        total_reward += reward
        state = tuple(next_state)

        if done:
            print(f"Episode finished in {step+1} steps. Total reward: {total_reward}")
            break

    env.close()


# Execute:
if train:
    train_q_learning(
        env=env,
        no_episodes=no_episodes,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        alpha=learning_rate,
        gamma=gamma,
        q_table_save_path=f"./2. Q Learning/{q_table_filename}"
    )

if test:
    test_agent(env, q_table_path=f"./b-q_table_lr0.01_eps1.0_gamma0.99_20250704_024036.npy")

if visualize_results:
    visualize_q_table(
        hell_state_coordinates=hell_state_coordinates,
        goal_coordinates=goal_coordinates,
        q_values_path=f"./2. Q Learning/{q_table_filename}"
    )
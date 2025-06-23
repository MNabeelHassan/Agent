# Imports:
# --------
from padm_env import create_env
from Q_learning import train_q_learning, visualize_q_table

# User definitions:
# -----------------
train = True
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

env = create_env(
    goal_coordinates=(14, 14),
    random_initialization=True
)

goal_coordinates = tuple(env.goal_state)
hell_state_coordinates = [
    tuple(cell)
    for snake in env.obstacles
    for cell in snake
]

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
        q_table_save_path="q_table.npy"
    )

if visualize_results:
    visualize_q_table(
        hell_state_coordinates=hell_state_coordinates,
        goal_coordinates=goal_coordinates,
        q_values_path="q_table.npy"
    )
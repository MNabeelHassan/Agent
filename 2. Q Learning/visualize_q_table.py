from Q_learning import visualize_q_table

hell_state_coordinates = [  # Use same obstacle positions
    (2, 2), (2, 3), (3, 3), (3, 4),
    (6, 1), (6, 2), (7, 2), (7, 3),
    (10, 6), (10, 7), (11, 7), (11, 8),
    (4, 10), (5, 10), (5, 11), (6, 11),
    (12, 2), (12, 3), (13, 3), (13, 4),
    (1, 13), (2, 13), (2, 14), (3, 14),
    (8, 12), (8, 13), (9, 13), (9, 14),
    (5, 5), (5, 6), (6, 6), (6, 7),
]

goal_coordinates = (14, 14)

visualize_q_table(
    hell_state_coordinates=hell_state_coordinates,
    goal_coordinates=goal_coordinates,
    q_values_path="q_table.npy"  # your saved file
)

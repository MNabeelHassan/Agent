import  gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def draw_snake(ax, center, length=1.5, segments=100):
    x0, y0 = center

    # Snake path (spirally curve)
    t = np.linspace(0, 4 * np.pi, segments)
    x = x0 + 0.5 * np.cos(t) * np.sin(2 * t) * length
    y = y0 + 0.5 * np.sin(t) * np.cos(2 * t) * length

    # Draw snake body
    ax.plot(x, y, color='green', linewidth=2)

    # Draw head (larger circle at the end)
    ax.plot(x[-1], y[-1], 'o', color='darkgreen', markersize=8)

    # Draw tail (smaller circle at the start)
    ax.plot(x[0], y[0], 'o', color='lightgreen', markersize=4)

class PadmACustomEnv(gym.Env):
    def __init__(self, grid_size=15):
        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([1,1])
        
        self.obstacles = [
            [np.array([2, 2]), np.array([2, 3]), np.array([3, 3]), np.array([3,4])],   # Snake 1
            [np.array([10, 10]), np.array([10, 11]), np.array([11, 11]), np.array([11,12])]    # Snake 2
        ]      

        self.goal_state = np.array([0,14])
        self.action_space = gym.spaces.Discrete(4)
        self.observationobservation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(2,))
        self.fig, self.ax = plt.subplots()

    def reset(self):
        self.agent_state = np.array([1,1])
        return self.agent_state

    def step(self, action):
        if action == 0 and self.agent_state[1] < self.grid_size: # up
            self.agent_state[1] += 1
        
        elif action == 1 and self.agent_state[1] > 0: # down
            self.agent_state[1] -= 1

        elif action == 2 and self.agent_state[0] > 0: # left
            self.agent_state[0] -= 1

        elif action == 3 and self.agent_state[0] < self.grid_size: # right
            self.agent_state[0] += 1
        
        reward = 0

        # Check for obstacle collision using np.array_equal
        for snake in self.obstacles:
            for obs in snake:
                if np.array_equal(self.agent_state, obs):
                    print("⚠️ Snake bite! Back to start.")
                    self.agent_state[0] = 0
                    self.agent_state[1] = 0
                    break

        done = np.array_equal(self.agent_state,self.goal_state)


        if done:
            reward = 10
        else:
            pass

        info = {"distance to goal": self.goal_state - self.agent_state}

        return self.agent_state, reward, done, info

    def render(self):
        self.ax.clear()

        for snake in self.obstacles:
            xs = [pt[0] for pt in snake]
            ys = [pt[1] for pt in snake]
            self.ax.plot(xs, ys, color='green', linewidth=3, marker='o')

        self.ax.plot(self.agent_state[0], self.agent_state[1], "ro")
        self.ax.plot(self.goal_state[0], self.goal_state[1], "g+")
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect('equal')
        plt.pause(0.1)
        
    def close(self):
        plt.close()

if __name__ == "__main__":
    env = PadmACustomEnv()
    state = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        print(action)
        state, reward, done, info = env.step(action)
        env.render()
        print(f"State: {state}, Reward: {reward}, Action: {action}, Done:{done}, Info: {info}")
        if done:
            print("Bruhhhhhhhh!!!!!! I reached the destination. Stop pestering me.")
            break
    env.close()

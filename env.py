import  gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class PadmACustomEnv(gym.Env):
    def __init__(self, grid_size=15):
        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([1,1])
        
        self.obstacles = [
            np.array([2, 2]),
            np.array([3, 3])
        ]        

        self.goal_state = np.array([4,4])
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
        for obs in self.obstacles:
            if np.array_equal(self.agent_state, obs):
                print("⚠️ Obstacle hit! Teleporting...")
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

        # Draw obstacles (snake)
        for obs in self.obstacles:
            self.ax.plot(obs[0], obs[1], "ks", markersize=10)

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

# Ideas
## Snake and Ladder Game
## Here there is only snake, unless you have time for ladder
## So the idea is that snakes are the obstacles
## The agent goes through the grid and tries to reach the goal
## The snake will have a head and a tail
## Everytime the agent goes to the head of the snake, it will go down
## to the tail of snake and remember where the head was and try to avoid it 
## When trying to climb up the grid
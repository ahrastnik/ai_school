import numpy as np
import gym

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.reset()

    done = False

    while not done:
        action = 2
        new_state, reward, done, info = env.step(action)
        env.render()

    env.close()

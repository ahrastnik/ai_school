import numpy as np
import gym
import time


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    actions = env.action_space.n
    states = env.observation_space.n
    q_table = np.zeros([states, actions])

    EPISODES_NUM = 10000
    STEPS_MAX = 100

    LEARNING_RATE = 0.1
    DISCOUNT_RATE = 0.99

    exploration_rate = 1
    EXPLORATION_RATE_MAX = 1
    EXPLORATION_RATE_MIN = 0.01
    EXPLORATION_RATE_DECAY = 0.001

    for episode in range(EPISODES_NUM):
        state = env.reset()
        done = False
        episode_rewards = 0

        for step in range(STEPS_MAX):
            exploration_threshold = np.random.uniform(0, 1)
            if exploration_threshold > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)
            q_table[state, action] = q_table[state, action] * (
                1 - LEARNING_RATE
            ) + LEARNING_RATE * (reward + DISCOUNT_RATE * np.max(q_table[new_state, :]))

            state = new_state
            episode_rewards += reward

            if done:
                break

        exploration_rate = EXPLORATION_RATE_MIN + (
            EXPLORATION_RATE_MAX - EXPLORATION_RATE_MIN
        ) * np.exp(-EXPLORATION_RATE_DECAY * episode)

    print(q_table)

    for episode in range(3):
        state = env.reset()
        done = False
        time.sleep(1)

        for step in range(STEPS_MAX):
            env.render()
            time.sleep(0.3)

            action = np.argmax(q_table[state, :])
            new_state, reward, done, info = env.step(action)

            if done:
                env.render()
                if reward == 1:
                    print("Goal reached!")
                else:
                    print("Agent fell in hole!")
                time.sleep(3)
                break

            state = new_state

    env.close()

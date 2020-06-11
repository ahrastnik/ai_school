import numpy as np
import gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Create the learning environment
    env = gym.make("MountainCar-v0")
    # Create the Q-table
    actions = env.action_space.n
    states = [20] * len(env.observation_space.high)

    # Observation space [position, speed]
    observation_step_size = (
        env.observation_space.high - env.observation_space.low
    ) / states
    q_table = np.random.uniform(low=-2, high=0, size=(states + [actions]))

    def get_state_index(continues_state):
        index = (continues_state - env.observation_space.low) / observation_step_size
        return tuple(index.astype(np.int))

    EPISODES_NUM = 10000
    SHOW_EVERY = 1000

    LEARNING_RATE = 0.1
    DISCOUNT_RATE = 0.99

    EXPLORATION_RATE_MAX = 1
    EXPLORATION_RATE_MIN = 0.01
    EXPLORATION_RATE_DECAY = 0.001

    exploration_rate = 1
    episode_rewards = []
    episode_rewards_aggr = {"episode": [], "average": [], "min": [], "max": []}

    for episode in range(EPISODES_NUM):
        print(f"Episode: {episode}")
        if episode > 0 and episode % SHOW_EVERY == 0:
            render = True
        else:
            render = False

        state = env.reset()
        state_index = get_state_index(state)
        done = False
        episode_reward = 0

        while not done:
            exploration_threshold = np.random.uniform(0, 1)
            if exploration_threshold > exploration_rate:
                action = np.argmax(q_table[state_index])
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)
            episode_reward += reward
            new_state_index = get_state_index(new_state)

            if render:
                env.render()

            if not done:
                max_future_q = np.max(q_table[new_state_index])
                current_q = q_table[state_index + (action,)]
                new_q = current_q * (1 - LEARNING_RATE) + LEARNING_RATE * (
                    reward + DISCOUNT_RATE * max_future_q
                )
                q_table[state_index + (action,)] = new_q
            elif new_state[0] >= env.goal_position:
                print(f"Goal reached! Episode {episode}")
                q_table[state_index + (action,)] = 0

            state_index = new_state_index

        exploration_rate = EXPLORATION_RATE_MIN + (
            EXPLORATION_RATE_MAX - EXPLORATION_RATE_MIN
        ) * np.exp(-EXPLORATION_RATE_DECAY * episode)

        episode_rewards.append(episode_reward)

        if not episode % SHOW_EVERY:
            average_reward = sum(episode_rewards[-SHOW_EVERY:]) / len(
                episode_rewards[-SHOW_EVERY:]
            )
            episode_rewards_aggr["episode"].append(episode)
            episode_rewards_aggr["average"].append(average_reward)
            episode_rewards_aggr["min"].append(min(episode_rewards[-SHOW_EVERY:]))
            episode_rewards_aggr["max"].append(max(episode_rewards[-SHOW_EVERY:]))
            print(episode_rewards_aggr)

    env.close()

    plt.figure()
    plt.plot(
        episode_rewards_aggr["episode"],
        episode_rewards_aggr["average"],
        label="average",
    )
    plt.plot(episode_rewards_aggr["episode"], episode_rewards_aggr["min"], label="min")
    plt.plot(episode_rewards_aggr["episode"], episode_rewards_aggr["max"], label="max")
    plt.title("Learning metrics")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(loc=4)
    plt.show()

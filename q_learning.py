from enum import Enum
from abc import abstractmethod

import numpy as np
import gym
import matplotlib.pyplot as plt


class AgentTypes(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class RLAgent:
    def __init__(
        self,
        agent_type: AgentTypes,
        actions: int,
        states: np.array,
        state_space: np.array = None,
        learning_rate=0.1,
        discount_rate=0.99,
        exploration_rate_start=1.0,
        exploration_rate_end=0.01,
        exploration_rate_decay=0.001,
    ):
        # Basic agent classification
        self._agent_type = agent_type
        self._actions = actions
        self._states = states
        self._state_space = state_space
        self._state_space_step = None
        # Learning parameters
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate
        self._exploration_rate_start = exploration_rate_start
        self._exploration_rate_end = exploration_rate_end
        self._exploration_rate_decay = exploration_rate_decay
        self._exploration_rate = exploration_rate_start
        # Q-table
        self._q_table = np.random.uniform(low=-2, high=0, size=(states + [actions]))
        # Learning metrics
        self._episode_rewards = []
        self._episode_rewards_aggr = {
            "episode": [],
            "average": [],
            "min": [],
            "max": [],
        }
        # Characterize continuous state space
        if self._agent_type == AgentTypes.CONTINUOUS:
            if self._state_space is None:
                raise ValueError(
                    "State space must be a valid matrix containing max and min values for each variable!"
                )
            else:
                self._state_space_step = (
                    self._state_space[:, 1] - self._state_space[:, 0]
                ) / states

    def _get_discrete_state(self, continues_state):
        if self._agent_type == AgentTypes.DISCRETE:
            raise ValueError("Can't calculate state index in discrete state space!")

        index = (continues_state - self._state_space[:, 0]) / self._state_space_step
        return tuple(index.astype(np.int))

    def run(self, episodes: int, capture_every: int = 0, render_every: int = 0):
        for episode in range(episodes):
            if (render_every > 0) and (episode > 0) and (episode % render_every == 0):
                render = True
            else:
                render = False
            # Set the initial state or state index in case of a continuous agent
            init_state = self._environment_begin_callback()
            if self._agent_type == AgentTypes.CONTINUOUS:
                state = self._get_discrete_state(init_state)
            else:
                state = init_state

            done = False
            episode_reward = 0

            while not done:
                # Decide between exploration or exploitation
                exploration_threshold = np.random.uniform(0, 1)
                if exploration_threshold > self._exploration_rate:
                    action = np.argmax(self._q_table[state])
                else:
                    action = np.random.randint(0, self._actions)
                # Run the agent
                new_state, reward, done, info = self._step(action)
                self._step_callback(episode, action, new_state, reward, done, info)
                episode_reward += reward
                new_state_index = self._get_discrete_state(new_state)

                if render:
                    self._render()

                if not done:
                    # Update the Q-table and continue learning, if not finished
                    max_future_q = np.max(self._q_table[new_state_index])
                    current_q = self._q_table[state + (action,)]
                    new_q = current_q * (
                        1 - self._learning_rate
                    ) + self._learning_rate * (
                        reward + self._discount_rate * max_future_q
                    )
                    self._q_table[state + (action,)] = new_q

                # Update state
                if self._agent_type == AgentTypes.CONTINUOUS:
                    state = self._get_discrete_state(new_state)
                else:
                    state = new_state

            # Apply exploration decay
            self._exploration_rate = self._exploration_rate_end + (
                self._exploration_rate_start - self._exploration_rate_end
            ) * np.exp(-self._exploration_rate_decay * episode)

            self._environment_end_callback()

            # Capture learning metrics
            self._episode_rewards.append(episode_reward)

            if (capture_every > 0) and (not episode % capture_every):
                average_reward = sum(self._episode_rewards[-capture_every:]) / len(
                    self._episode_rewards[-capture_every:]
                )
                self._episode_rewards_aggr["episode"].append(episode)
                self._episode_rewards_aggr["average"].append(average_reward)
                self._episode_rewards_aggr["min"].append(
                    min(self._episode_rewards[-capture_every:])
                )
                self._episode_rewards_aggr["max"].append(
                    max(self._episode_rewards[-capture_every:])
                )

    @abstractmethod
    def _environment_begin_callback(self):
        pass

    def _environment_end_callback(self):
        pass

    @abstractmethod
    def _step(self, action, **kwargs):
        pass

    def _step_callback(self, episode, action, state, reward, done, info):
        pass

    def _render(self, **kwargs):
        pass

    def show_metrics(self):
        plt.figure()
        plt.plot(
            self._episode_rewards_aggr["episode"],
            self._episode_rewards_aggr["average"],
            label="average",
        )
        plt.plot(
            self._episode_rewards_aggr["episode"],
            self._episode_rewards_aggr["min"],
            label="min",
        )
        plt.plot(
            self._episode_rewards_aggr["episode"],
            self._episode_rewards_aggr["max"],
            label="max",
        )
        plt.title("Learning metrics")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend(loc=4)
        plt.show()


class GymAgent(RLAgent):
    def __init__(self, environment: gym.Env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env = environment

    def _environment_begin_callback(self):
        return self._env.reset()

    def _step(self, action, **kwargs):
        return self._env.step(action)

    def _render(self, **kwargs):
        self._env.render()

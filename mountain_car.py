import numpy as np
import gym

from q_learning import AgentTypes, GymAgent


class MountainCarAgent(GymAgent):
    def __init__(self, environment):
        env_actions = env.action_space.n
        env_states = [20] * len(env.observation_space.high)
        env_state_space = np.array(
            [env.observation_space.low, env.observation_space.high]
        ).T

        super().__init__(
            environment,
            AgentTypes.CONTINUOUS,
            env_actions,
            env_states,
            state_space=env_state_space,
        )

    def _step_callback(self, episode, action, state, reward, done, info):
        if state[0] >= env.goal_position:
            print(f"Goal reached! Episode {episode}")
            self._q_table[self._get_discrete_state(state) + (action,)] = 0


if __name__ == "__main__":
    # Create the learning environment
    env = gym.make("MountainCar-v0")
    agent = MountainCarAgent(env)
    agent.run(10000, capture_every=50, render_every=1000)
    env.close()
    agent.show_metrics()

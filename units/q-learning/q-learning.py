import numpy as np
import gymnasium
import random
import imageio
import os
import tqdm


def create_q_table(state_space: int, action_space: int):
    Qtable = np.array((state_space, action_space))
    return Qtable


def greedy_policy(q_table, state):
    # policies are the brain of the RL agent.
    # The tell us what action to take.

    # our Qtable is a matix of
    # row: state
    # column: action
    action = np.argmax(q_table[state])
    # return the index of the best action to take.
    return action


def epsilon_greedy_policy(q_table, state, epsilon):
    # This function is used to create the q_table
    # following the off-policy method (Different policies from train/inference)

    random_num = np.random.uniform(0, 1)

    # if random_num > episilon > exploitation
    if random_num > epsilon:
        action = greedy_policy(q_table)

    # otherwise exploration
    else:
        action = env.action_space.sample()

    return action


def training_loop(
    env,
    num_training_episode: int,
    learning_rate: float,
    min_epsilon: float,
    max_epsilon: float,
    decay_rate: float,
    max_steps: int,
    Qtable: np.array,
):
    for eps in range(num_training_episode):
        # Reduce epsilon ( we need less and less exploration )
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # reset env
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            # choose an action
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # Take Action at Observation Rt+1, st+1
            # Take the action (a) and observe the outcome state(s) and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s, a) := Q(s, a) +  lr [R(s,a) + gamma * max Q(s', a') - Q(s, a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )

            if terminated or truncated:
                break

            state = new_state

    return Qtable


if __name__ == "__main__":
    n_training_episodes = 10000  # Num of training episodes
    lr = 0.7  # learning rate
    n_evals = 100  # num of testing episodes

    env_id = "FrozenLake-v1"
    max_steps = 99  # max steps per episodes
    gamma = 0.95
    eval_seed = []

    # Exploration parameters
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 5e-4

    # RL World
    # Frozen lake has 4 Actions
    # 16 observable space
    frozen_lake = gymnasium.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
        success_rate=0.33,
        reward_schedule=(1, 0, 0),
        render_mode="rgb_array",
    )

    q_table_frozenlake = create_q_table(state_space=env.observation_space.n, action_space=env.action_space.n)

    trained_Q_table = training_loop(
        env=frozen_lake,
        num_training_episode=n_training_episodes,
        learning_rate=lr,
        min_epsilon=min_epsilon,
        max_epsilon=max_epsilon,
        decay_rate=decay_rate,
        max_steps=max_steps,
        Qtable=q_table_frozenlake,
    )

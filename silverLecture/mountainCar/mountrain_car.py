import numpy as np
import gym
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

class State:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

GAMMA = 0.9
ACTIONS_COUNT = 3
EPS_GREEDY = 0
EPISODES = 1000
TIMESTEPS = 200  # limit from gym
ALPHA = 0.5

def find_best_action_and_q(Q, state):
    best_action, best_q = None, None
    for action in range(ACTIONS_COUNT):
        cur_q = Q[(state, action)]
        if best_q is None or best_q < cur_q:
            best_action, best_q = action, cur_q
    return best_action, best_q


def eps_greedy_action(Q, state):
    best_action, best_q = find_best_action_and_q(Q, state)
    best_count = 0
    for action in range(ACTIONS_COUNT):
        if Q[(state, action)] == best_q:
            best_count += 1
    p = []
    for action in range(ACTIONS_COUNT):
        prob = EPS_GREEDY / ACTIONS_COUNT
        if Q[(state, action)] == best_q:
            prob += (1 - EPS_GREEDY) / best_count
        p.append(prob)
    return np.random.choice(ACTIONS_COUNT, 1, p=p)[0]


def observation_to_state(observation):
    return State(int(round(observation[0] / 0.1)),
                 int(round(observation[1] / 0.01)))


def evaluate():
    env = gym.make('MountainCar-v0')
    env.seed(0)
    np.random.seed(0)
    cumulative_completion = []
    completed = 0
    Q = defaultdict(lambda: 0.0)
    for episode in range(EPISODES):
        observation = env.reset()
        state = observation_to_state(observation)
        action = eps_greedy_action(Q, state)
        print("ep : ", episode)
        q_temp = []
        pos_temp = []
        veloc_temp = []
        for timestep in range(TIMESTEPS):
            # env.render()
            observation, reward, done, info = env.step(action)
            to_state = observation_to_state(observation)
            next_action = eps_greedy_action(Q, to_state)
            Q[(state, action)] += ALPHA * (reward +
                                           GAMMA * Q[(to_state, next_action)] -
                                           Q[(state, action)])
            q_temp.append(-np.max(Q[(state, action)]))
            pos_temp.append(state.position_tile * 0.1)
            veloc_temp.append(state.velocity_tile * 0.01)

            action, state = next_action, to_state
            if done:
                if timestep != TIMESTEPS - 1:
                    completed += 1
                cumulative_completion.append(completed)
                print("completed : ", completed)
                break

        # plot Q
        if episode == 500 :
            fig = plt.figure()
            ax = fig.gca(projection='3d')          
            surf = ax.plot_trisurf(pos_temp, veloc_temp, q_temp, cmap=plt.cm.viridis, linewidth=0.2)
            fig.colorbar( surf, shrink=0.5, aspect=10)
            ax.set_xlabel("position")
            ax.set_ylabel("velocity")
            ax.set_zlabel("q")
            plt.show()

    env.close()
    return cumulative_completion

# main
cumulative_completion = evaluate()

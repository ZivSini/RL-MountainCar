import gym
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random, choice
import json
from tempfile import TemporaryFile

outfile = TemporaryFile()

env = gym.make("MountainCar-v0")
env._max_episode_steps = 500

million = 1000000
number_of_samples = 100

sigma_v = 0.0004
sigma_p = 0.04
diag = np.array([[sigma_p, 0], [0, sigma_v]])
diag_inv = np.linalg.inv(diag)
number_of_actions = env.action_space.n
min_pos = env.min_position
max_pos = env.max_position
max_spd = env.max_speed
min_spd = -env.max_speed
takeSample = np.append(np.zeros(1), (np.linspace(0, million, number_of_samples)))
policy_value = []

# leftmost_pos_center = min_pos + abs(min_pos*0.2)
# rightmost_pos_center = max_pos - abs(max_pos)*0.2
# lowest_vel_center = min_spd + abs(min_spd)*0.2
# highest_vel_center = max_spd - abs(max_spd)*0.2
c_p = np.linspace(min_pos, max_pos, 6)
c_v = np.linspace(min_spd, max_spd, 8)

gaussian_mat = [(p, v) for p in c_p for v in c_v]
num_of_features = c_v.size * c_p.size


def epsilon_greedy(w, epsilon, theta):
    explore = (random() < epsilon)
    if explore:
        return np.random.randint(0, number_of_actions)
    else:
        action_value = [np.matmul(w[:, action], theta) for action in range(number_of_actions)]
        return np.argmax(action_value)


def get_theta(p, v):
    x_vector = np.zeros((num_of_features, 2))
    theta_mat = np.zeros(num_of_features)
    for i in range(num_of_features):
        x_vector[i] = np.abs(np.array([p, v]).transpose() - np.array(gaussian_mat[i]).transpose())
        theta_mat[i] = np.exp(-0.5 * (
            np.matmul(np.matmul(np.transpose(x_vector[i]), diag_inv), x_vector[i])))  #############################
    return theta_mat


def simulate(w, env):
    sum = 0.0
    for sim in range(100):
        (p, v) = env.reset()
        done = False
        while not done:
            (p, v), reward, done, info = env.step(epsilon_greedy(w, 0, get_theta(p, v)))
            sum += reward
    return sum / 100


def learn_sarsa():
    w = np.random.rand(num_of_features, number_of_actions)
    gamma = 1
    alpha = 0.02
    _lambda = 0.5
    next_sample = 0
    epsilon = 1
    episode = 1
    curr_reword = 0
    timeSteps = 0
    while timeSteps < million:
        if next_sample < len(takeSample) and timeSteps >= takeSample[next_sample]:
            policy_value.append(simulate(w, env))
            next_sample += 1
        E = np.zeros((num_of_features, number_of_actions))
        s_p, s_v = env.reset()
        F_s = get_theta(s_p, s_v)
        a = epsilon_greedy(w, epsilon, F_s)

        done = False
        while not done:
            E[:, a] += F_s
            (ns_p, ns_v), reward, done, info = env.step(a)
            delta = reward - np.matmul(w[:, a], F_s)
            F_s = get_theta(ns_p, ns_v)
            a = epsilon_greedy(w, epsilon, F_s)
            Qa = np.matmul(w[:, a], F_s)
            delta += gamma * Qa
            w += alpha * delta * E
            E *= gamma * _lambda

            timeSteps += 1
            curr_reword += reward
            timeSteps += 1
        if episode % 100 == 0 and episode > 0:
            print('episode ', episode, 'score ', curr_reword, 'epsilon %.3f' % epsilon)
            curr_reword = 0
        episode += 1
        epsilon *= 0.999
    policy_value.append(simulate(w, env))
    with open('aprox_val_fun.npy', 'wb') as f:
        np.save(f, w)


if __name__ == '__main__':
    # with open('aprox_val_fun.npy', 'rb') as f:
    #     w = np.load(f)
    #     (p, v) = env.reset()
    #     env.render()
    #     done = False
    #     while not done:
    #         (p, v), reward, done, info = env.step(epsilon_greedy(w, 0, get_theta(p, v)))
    #         env.render()
    #     env.reset()
    #     env.close()

    learn_sarsa()

    plt.plot(takeSample[:len(policy_value)], policy_value, linewidth=2)
    plt.xlabel(f"time steps")
    plt.ylabel("episode value".format(100))
    plt.show()

import gym
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random, choice


def epsilon_greedy(Q, epsilon, theta):
    explore = (random() < epsilon)
    if explore:
        return np.random.randint(0, len(Q[0]))
    else:
        action_value = np.zeros(3)
        for action in range(len(Q[0])):
            action_value[action] = np.matmul(Q[:, action], theta)
        return np.argmax(action_value)

    # values = Q[state, :]
    # max_value = max(values)
    # num_of_actions = len(values)
    # greedy_actions = [act for act in range(num_of_actions) if values[act] == max_value]


def get_theta(p, v, gaussian_mat, diag):
    x_vector = np.zeros((len(gaussian_mat), 2))
    theta_mat = np.zeros(len(gaussian_mat))
    for i in range(len(x_vector)):
        x_vector[i] = np.array([p, v]).transpose() - np.array(gaussian_mat[i]).transpose()
        theta_mat[i] = np.exp(
            -0.5 * (np.matmul(np.matmul(np.transpose(x_vector[i]), diag), x_vector[i])))
    return theta_mat


def simulate(Q, env):
    sum = 0.0
    for sim in range(100):
        current_state = env.reset()
        done = False
        while not done:
            values = Q[env.env.s, :]
            max_value = max(values)
            no_actions = len(values)
            greedy_actions1 = [a for a in range(no_actions) if values[a] == max_value]
            state, reward, done, info = env.step(greedy_actions1[0])
            sum += reward
    return sum / 100


def learn_sarsa():
    takeSample = np.linspace(1000, 1000000, 100)
    env = gym.make("MountainCar-v0")
    number_of_actions = env.action_space.n
    min_pos = env.min_position
    max_pos = env.max_position
    max_spd = env.max_speed
    min_spd = -env.max_speed
    c_p = np.linspace(min_pos, max_pos, 4)
    c_v = np.linspace(min_spd, max_spd, 8)
    gaussian_mat = []
    for p in c_p:
        for v in c_v:
            gaussian_mat.append((p, v))
    num_of_features = c_v.size * c_p.size
    # Q = np.zeros((num_of_features, number_of_actions))
    # Q.fill(np.random.rand())
    Q = np.random.rand(num_of_features, number_of_actions)
    next_x_vector = np.zeros((num_of_features, 2))
    curr_x_vector = np.zeros((num_of_features, 2))
    # next_x_vector = np.zeros((num_of_features, 2))
    sigma_v = 0.0004
    sigma_p = 0.04
    diag = [[sigma_p, 0], [0, sigma_v]]
    gamma = 1
    alpha = 0.02
    _lambda = 0.5
    next_theta_mat = np.zeros(num_of_features)
    curr_theta_mat = np.zeros(num_of_features)
    timeSteps = 0
    next_sample = 0
    policy_value = []
    epsilon = 1
    while timeSteps < 500000:
        curr_pos, curr_spd = env.reset()
        # for i in range(num_of_features):
        #     curr_x_vector[i] = np.array([curr_pos, curr_spd]).transpose() - np.array(gaussian_mat[i]).transpose()
        #     curr_theta_mat[i] = np.exp(-0.5 * (np.matmul(np.matmul(np.transpose(curr_x_vector[i]), diag), curr_x_vector[i])))
        curr_theta_mat = get_theta(curr_pos, curr_spd, gaussian_mat, diag)
        E = np.zeros((num_of_features, number_of_actions))
        curr_action = epsilon_greedy(Q, epsilon, curr_theta_mat)
        done = False
        # if next_sample < len(takeSample) and timeSteps > takeSample[next_sample]:
        #     policy_value.append(simulate(Q, env))
        #     next_sample += 1
        print(timeSteps)

        while not done:
            (next_pos, next_spd), reward, done, info = env.step(curr_action)
            next_action = epsilon_greedy(Q, epsilon, curr_theta_mat)
            # for i in range(num_of_features):
            #     next_x_vector[i] = np.array([curr_pos, curr_spd]).transpose() - np.array(gaussian_mat[i]).transpose()
            #     next_theta_mat[i] = np.exp(
            #         -0.5 * (np.matmul(np.matmul(np.transpose(next_x_vector[i]), diag), next_x_vector[i])))
            next_theta_mat = get_theta(curr_pos,curr_spd,gaussian_mat,diag)
            delta = reward + gamma * np.matmul(Q[:, next_action], next_theta_mat) - np.matmul(Q[:, curr_action],
                                                                                              curr_theta_mat)
            # delta = reward + gamma * Q[:, next_action] * next_x_vector.transpose() - Q[:, curr_action] * curr_x_vector
            E *= _lambda * gamma
            E[:, curr_action] += curr_theta_mat
            Q += alpha * delta * E  # maybe need to change + to -
            curr_pos = next_pos
            curr_spd = next_spd
            curr_action = next_action
            curr_theta_mat = next_theta_mat
            curr_x_vector = next_x_vector
            timeSteps += 1
        epsilon *= 0.9999

    (p, v) = env.reset()
    env.render()
    done = False
    while not done:
        (p, v), reward, done, info = env.step(epsilon_greedy(Q, 0, get_theta(p, v, gaussian_mat, diag)))
        env.render()
    env.reset()

    plt.plot(takeSample[:len(policy_value)], policy_value, linewidth=2)
    plt.xlabel(f"time steps")
    plt.ylabel("episode value".format(100))
    plt.show()




    #TODO fix the render problem
    #TODO write simulate function
    #TODO save the best learned Q in json






    # Q = np.zeros((number_of_states, number_of_actions))
    # epsilon = 1.0
    # gamma = 0.95
    # policy_value = []
    # counter = 0
    # next_sample = 0
    # while counter < 1000000:
    #     if next_sample < len(takeSample) and counter > takeSample[next_sample]:
    #         policy_value.append(simulate(Q, env))
    #         next_sample += 1
    #     E = np.zeros((number_of_states, number_of_actions))
    #     current_state = env.reset()
    #     current_action = epsilon_greedy(Q, epsilon, current_state)
    #     done = False
    #     step_number = 0
    #     while not done and step_number < 300:
    #         counter += 1
    #         next_state, reward, done, info = env.step(current_action)
    #         step_number += 1
    #         next_action = epsilon_greedy(Q, epsilon, next_state)
    #         delta = reward + gamma * Q[next_state, next_action] - Q[current_state, current_action]
    #         E[current_state, current_action] += 1
    #         Q = Q + alpha * delta * E
    #         E = _lambda * gamma * E
    #         current_state = next_state
    #         current_action = next_action
    #     epsilon = epsilon * 0.999
    # env.reset()
    # env.render()
    # done = False
    # while not done:
    #     values = Q[env.env.s, :]
    #     max_value = max(values)
    #     no_actions = len(values)
    #     greedy_actions1 = [a for a in range(no_actions) if values[a] == max_value]
    #     state, reward, done, info = env.step(greedy_actions1[0])
    #     env.render()
    # env.reset()
    #
    # plt.plot(takeSample[:len(policy_value)], policy_value, linewidth=2)
    # plt.xlabel(f"time steps\n   alpha= {alpha}  lambda= {_lambda} ")
    # plt.ylabel("episode value".format(100))
    # plt.show()


if __name__ == '__main__':
    learn_sarsa()
    # print("running...")
    # learn_sarsa(0.15, 0.7)
    # learn_sarsa(0.15, 0.3)
    # learn_sarsa(0.05, 0.7)
    # learn_sarsa(0.05, 0.3)

    # learn_sarsa(0.15, 0.5)
    #
    # learn_sarsa(0.1, 0.2)
    # learn_sarsa(0.2, 0.2)
    # learn_sarsa(0.1, 0.4)
    # learn_sarsa(0.2, 0.4)
    # #
    # learn_sarsa(0.2, 0.4)
    # learn_sarsa(0.2, 0.4)
    # learn_sarsa(0.2, 0.4)
    # learn_sarsa(0.2, 0.4)

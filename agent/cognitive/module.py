# coding: utf-8

import copy
import os

import brica1.gym
import numpy as np
import six.moves.cPickle as pickle
from chainer import cuda

from ml.cnn_feature_extractor import CnnFeatureExtractor
from ml.q_net import QNet


class VVCComponent(brica1.Component):
    image_feature_count = 1
    cnn_feature_extractor = 'alexnet_feature_extractor.pickle'
    model = 'bvlc_alexnet.caffemodel'
    model_type = 'alexnet'
    image_feature_dim = 256 * 6 * 6

    def __init__(self, use_gpu=True, n_output=10240, n_input=1):
        # image_feature_count = 1
        super(VVCComponent, self).__init__()

        self.use_gpu = use_gpu
        self.n_output = n_output
        self.n_input = n_input

        self.make_in_port('Isocortex#V1-Isocortex#VVC-Input', self.n_input)  # observation from environment
        self.make_out_port('Isocortex#VVC-UB-Output', self.n_output)  # feature vector
        self.make_out_port('Isocortex#VVC-BG-Output', self.n_output)  # feature vector

        # self.make_in_port('Isocortex.DVC-Isocortex.VVC-Input', 1) # this port is unused in this sample
        # self.make_out_port('Isocortex.VVC-Isocortex.ASC-Output', 1) # do not use in this sample

    def set_model(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def load_model(self, cnn_feature_extractor):
        if os.path.exists(cnn_feature_extractor):
            print("loading... " + cnn_feature_extractor),
            self.feature_extractor = pickle.load(open(cnn_feature_extractor))
            print("done")
        else:
            self.feature_extractor = CnnFeatureExtractor(self.use_gpu, self.model, self.model_type,
                                                         self.image_feature_dim)
            pickle.dump(self.feature_extractor, open(cnn_feature_extractor, 'w'))
            print("pickle.dump finished")

    def _observation_to_featurevec(self, observation):
        # TODO clean
        if self.image_feature_count == 1:
            return np.r_[self.feature_extractor.feature(observation["image"][0]),
                         observation["depth"][0]]
        elif self.image_feature_count == 4:
            return np.r_[self.feature_extractor.feature(observation["image"][0]),
                         self.feature_extractor.feature(observation["image"][1]),
                         self.feature_extractor.feature(observation["image"][2]),
                         self.feature_extractor.feature(observation["image"][3]),
                         observation["depth"][0],
                         observation["depth"][1],
                         observation["depth"][2],
                         observation["depth"][3]]
        else:
            print("not supported: number of camera")

    def fire(self):
        observation = self.get_in_port('Isocortex#V1-Isocortex#VVC-Input').buffer
        obs_array = self._observation_to_featurevec(observation)

        self.results['Isocortex#VVC-BG-Output'] = obs_array
        self.results['Isocortex#VVC-UB-Output'] = obs_array


class BGComponent(brica1.Component):
    actions = [0, 1, 2]
    policy_frozen = False
    epsilon_delta = 1.0 / 10 ** 4.4
    min_eps = 0.1

    def __init__(self, n_input=10240, n_output=1, use_gpu=True):
        super(BGComponent, self).__init__()
        self.use_gpu = use_gpu
        self.time = 0
        self.epsilon = 1.0
        self.input_dim = n_input
        self.q_net = QNet(use_gpu, self.actions, self.input_dim)

        # self.make_in_port('Isocortex.FL-BG-Input', 4) # this port is unused in this sample
        self.make_out_port('BG-Isocortex#FL-Output', 1)  # send action and reward to FL
        self.make_in_port('RB-BG-Input', 1)  # recieve reward from RB
        self.make_in_port('Isocortex#VVC-BG-Input', n_input)  # recieve state (feature vector) from VVC
        self.make_in_port('UB-BG-Input', 6)  # recieve replayed experience from UB
        # self.make_in_port('Isocortex.ASC-BG-Input', 10) # this port is unused in this sample
        # self.make_in_port('Isocortex.ODC-BG-Input', 10) # this port is unused in this sample
        # self.make_in_port('Isocortex.DVC-BG-Input', 10) # this port is unused in this sample

        self.results['BG-Isocortex#FL-Output'] = np.array([0])

        self.data_size = 10**5
        self.replay_size = 32
        self.hist_size = 1
        self.fl_exp = [False, 0, 0, 0, 0, False]
        self.vvc_exp = [0, 0]

    def start(self):
        self.state = np.zeros((self.q_net.hist_size, self.input_dim), dtype=np.uint8)
        self.features = self.get_in_port('Isocortex#VVC-BG-Input').buffer[0]
        self.state[0] = self.features

        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Generate an Action e-greedy
        action, q_now = self.q_net.e_greedy(state_, self.epsilon)
        return_action = action

        # Update for next step
        self.last_action = copy.deepcopy(return_action)
        self.last_state = self.state.copy()

        self.last_observation = self.features
        self.exp = [False, self.last_state, self .last_action, 0, self.state, False]
        self.fl_exp = [False, self.last_action, 0, False]
        self.vvc_exp = [self.last_state, self.state]
        return return_action

    def __step(self, features):
        if self.q_net.hist_size == 4:
            self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], features], dtype=np.uint8)
        elif self.q_net.hist_size == 2:
            self.state = np.asanyarray([self.state[1], features], dtype=np.uint8)
        elif self.q_net.hist_size == 1:
            self.state = np.asanyarray([features], dtype=np.uint8)
        else:
            print("self.DQN.hist_size err")

        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Exploration decays along the time sequence
        if self.policy_frozen is False:  # Learning ON/OFF
            if self.q_net.initial_exploration < self.time:
                self.epsilon -= self.epsilon_delta
                if self.epsilon < self.min_eps:
                    self.epsilon = self.min_eps
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print("Initial Exploration : %d/%d steps" % (self.time, self.q_net.initial_exploration)),
                eps = 1.0
        else:  # Evaluation
            print("Policy is Frozen")
            eps = 0.05

        # Generate an Action by e-greedy action selection
        action, q_now = self.q_net.e_greedy(state_, eps)

        return action, eps, q_now, features

    def __step_update(self, reward, action, eps, q_now, obs_array):
        if self.policy_frozen is False:  # Learning ON/OFF
            if self.exp[0]:
                self.q_net.optimizer.zero_grads()
                loss, _ = self.q_net.forward(self.exp[1], self.exp[2], self.exp[3], self.exp[4], self.exp[5])
                # loss, _ = self.q_net.forward(self.vvc_exp[0], self.fl_exp[1], self.fl_exp[2], self.vvc_exp[1],
                # self.fl_exp[3])
                loss.backward()
                self.q_net.optimizer.update()

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        # Simple text based visualization
        if self.use_gpu >= 0:
            q_max = np.max(q_now.get())
        else:
            q_max = np.max(q_now)

        print('Step:%d  Action:%d  Reward:%.1f  Epsilon:%.6f  Q_max:%3f' % (
            self.time, self.q_net.action_to_index(action), reward, eps, q_max))

        # Updates for next step
        self.last_observation = obs_array

        if self.policy_frozen is False:
            self.last_action = copy.deepcopy(action)
            self.last_state = self.state.copy()
            self.time += 1

    def agent_end(self, reward):  # Episode Terminated
        print('episode finished. Reward:%.1f / Epsilon:%.6f' % (reward, self.epsilon))

        # Learning Phase
        if self.policy_frozen is False:  # Learning ON/OFF
            self.exp = self.get_in_port('UB-BG-Input').buffer
            # self.fl_exp = self.get_in_port('Isocortex.FL-BG-Input').buffer
            # self.vvc_exp = self.get_in_port('Isocortex.VVC-BG-Input').buffer
            if self.exp[0]:
                self.q_net.optimizer.zero_grads()
                loss, _ = self.q_net.forward(self.exp[1], self.exp[2], self.exp[3], self.exp[4], self.exp[5])
                loss.backward()
                self.q_net.optimizer.update()

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        # Time count
        if self.policy_frozen is False:
            self.time += 1

        self.counter = 0

    def fire(self):

        # self.fl_exp = self.get_in_port('Isocortex.FL-BG-Input').buffer
        # self.vvc_exp = self.get_in_port('Isocortex.VVC-BG-Input').buffer
        reward = self.get_in_port('RB-BG-Input').buffer
        feature = self.get_in_port('Isocortex#VVC-BG-Input').buffer
        self.exp = self.get_in_port('UB-BG-Input').buffer

        action, eps, q_now, obs_array = self.__step(feature)

        self.__step_update(reward, action, eps, q_now, obs_array)
        # self.results['action'] = np.array([action])
        self.results['BG-Isocortex#FL-Output'] = np.array([action])


class UBComponent(brica1.Component):
    def __init__(self):
        super(UBComponent, self).__init__()

        self.use_gpu = 0
        self.data_size = 10**5
        self.replay_size = 32
        self.hist_size = 1
        # self.initial_exploration = 10
        self.initial_exploration = 10**3
        input_dim = 10240
        self.dim = input_dim

        self.d = [np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

        self.make_in_port('Isocortex#VVC-UB-Input', 10240)  # input: feature vector
        self.make_out_port('UB-BG-Output', 6)  # output: state_replay, state_dash_replay
        self.make_in_port('Isocortex#FL-UB-Input', 2)  # input: last_action, reward

        # self.make_in_port('Isocortex.DVC-UB-Input', 10) # this port is unused in this sample
        # self.make_in_port('Isocortex.ODC-UB-Input', 10) # this port is unused in this sample
        # self.make_out_port('UB-Isocortex.ASC-Output', 10) # this port is unused in this sample

        self.get_in_port('Isocortex#VVC-UB-Input').buffer = self.d[0][0]
        self.results['UB-BG-Output'] = False, 0, 0, 0, 0, 0
        # self.results['UB.UFL-Isocortex.FL-Output'] = False, 0, 0
        # self.results['UB.UVQ-Isocortex.VVC-Output'] = 0, 0
        self.last_state = self.d[0][0].copy()
        self.state = self.d[0][0].copy()
        self.time = 0

    def stock_experience(self, time, state, action, reward, state_dash, episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
        else:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
            self.d[3][data_index] = state_dash
        self.d[4][data_index] = episode_end_flag

    def experience_replay(self, time):
        replay_start = False
        if self.initial_exploration < time:
            replay_start = True
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            s_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.d[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.d[1][replay_index[i]]
                r_replay[i] = self.d[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.d[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.d[4][replay_index[i]]

            if self.use_gpu >= 0:
                s_replay = cuda.to_gpu(s_replay)
                s_dash_replay = cuda.to_gpu(s_dash_replay)

            return replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay

        else:
            return replay_start, 0, 0, 0, 0, False

    def end_episode(self):
        action, reward = self.get_in_port('Isocortex#FL-UB-Input').buffer
        # self.state = self.get_in_port('Isocortex.VVC-UB.UVQ-Input').buffer
        self.time += 1
        # self.stock_experience(self.time, self.second_last_state, action, reward, self.last_state, True)
        self.stock_experience(self.time, self.last_state, action, reward, self.state, True)
        replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay = \
            self.experience_replay(self.time)
        self.results['UB-BG-Output'] = [replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay]

    def fire(self):
        self.state = self.get_in_port('Isocortex#VVC-UB-Input').buffer
        action, reward = self.get_in_port('Isocortex#FL-UB-Input').buffer
        # self.stock_experience(self.time, self.second_last_state, action, reward, self.last_state, False)
        self.stock_experience(self.time, self.last_state, action, reward, self.state, False)
        replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay = \
            self.experience_replay(self.time)

        self.results['UB-BG-Output'] = [replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay]
        # self.second_last_state = self.last_state.copy()
        self.last_state = self.state.copy()
        self.time += 1


class FLComponent(brica1.Component):
    def __init__(self):
        super(FLComponent, self).__init__()
        # self.make_out_port('Isocortex.FL-BG-Output', 4) # this port is unused in this sample
        self.make_out_port('Isocortex#FL-MO-Output', 1)  # action
        self.make_out_port('Isocortex#FL-UB-Output', 2)  # action, reward　荒川さんのjsonにはないけど
        # self.make_out_port('Isocortex.FL-Isocortex.ASC-Output', 10) # this port is unused in this sample
        # self.make_out_port('Isocortex.FL-Isocortex.DVC-Output', 10) # this port is unused in this sample
        # self.make_in_port('Isocortex.ASC-Isocortex.FL-Input', 10)　# this port is unused in this sample
        self.make_in_port('BG-Isocortex#FL-Input', 1)  # action
        self.make_in_port('RB-Isocortex#FL-Input', 1)  # reward

        self.results['Isocortex#FL-UB-Output'] = [np.array([0]), 0]

        self.last_action = np.array([0])

    def fire(self):
        action = self.get_in_port('BG-Isocortex#FL-Input').buffer
        reward = self.get_in_port('RB-Isocortex#FL-Input').buffer
        self.results['Isocortex#FL-MO-Output'] = action
        self.results['Isocortex#FL-UB-Output'] = [self.last_action, reward]

        self.last_action = action

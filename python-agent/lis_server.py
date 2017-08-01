import os
import argparse
import cherrypy
import random
import cPickle as pickle

# Message Unpacking
import msgpack
import io
from PIL import Image
from PIL import ImageOps

import numpy as np

from cnn_feature_extractor import CnnFeatureExtractor
from cnn_dqn_agent import CnnDqnAgent

import brica1

def unpack(payload, depth_image_count=1, depth_image_dim=32*32):
    dat = msgpack.unpackb(payload)

    image = []
    for i in xrange(depth_image_count):
        image.append(Image.open(io.BytesIO(bytearray(dat['image'][i]))))

    depth = []
    for i in xrange(depth_image_count):
        d = (Image.open(io.BytesIO(bytearray(dat['depth'][i]))))
        depth.append(np.array(ImageOps.grayscale(d)).reshape(depth_image_dim))

    reward = dat['reward']
    observation = {"image": image, "depth": depth}

    return reward, observation

class RandomAgent:
    actions = [0, 1, 2]

    def agent_init(self, *args, **kwargs):
        pass

    def agent_start(self, observation):
        return random.choice(self.actions)

    def agent_end(self, reward):
        pass

    def agent_step(self, reward, observation):
        return random.choice(self.actions), None, None, None

    def agent_step_update(self, reward, action, eps, q_now, obs_array):
        pass

use_gpu = -1
cnn_feature_extractor = 'alexnet_feature_extractor.pickle'
model = 'bvlc_alexnet.caffemodel'
model_type = 'alexnet'
depth_image_dim = 32 * 32
depth_image_count = 1
image_feature_dim = 256 * 6 * 6

class CnnDqnComponent(brica1.Component):
    def __init__(self, **kwargs):
        super(CnnDqnComponent, self).__init__()

        self.agent = CnnDqnAgent()
        self.agent.agent_init(**kwargs)

        self.make_in_port('observation', 1)
        self.make_in_port('reward', 1)
        self.make_in_port('operation', 3)

        self.make_out_port('action', 1)

    def fire(self):
        observation = self.inputs['observation']
        reward = self.inputs['reward'][0]
        operation = self.inputs['operation']

        action = 0

        if operation[0] == 1:
            self.agent.agent_end(reward)

        if operation[1] == 1:
            action = self.agent.agent_start(observation)

        if operation[2] == 1:
            action, eps, q_now, obs_array = self.agent.agent_step(reward, observation)
            self.agent.agent_step_update(reward, action, eps, q_now, obs_array)

        self.results['action'] = np.array([action])

class Root(object):
    def __init__(self, **kwargs):
        if os.path.exists(cnn_feature_extractor):
            print("loading... " + cnn_feature_extractor),
            self.feature_extractor = pickle.load(open(cnn_feature_extractor))
            print("done")
        else:
            self.feature_extractor = CnnFeatureExtractor(use_gpu, model, model_type, image_feature_dim)
            pickle.dump(self.feature_extractor, open(cnn_feature_extractor, 'w'))
            print("pickle.dump finished")

        self.agents = {}
        self.schedulers = {}
        self.components = {}

    @cherrypy.expose()
    def flush(self, identifier):
        self.agents[identifier] = brica1.Agent()
        self.schedulers[identifier] = brica1.VirtualTimeScheduler(self.agents[identifier])
        self.components[identifier] = CnnDqnComponent(use_gpu=use_gpu, depth_image_dim=depth_image_dim * depth_image_count, feature_extractor=self.feature_extractor)
        self.agents[identifier].add_component('cnn_dqn_component', self.components[identifier])
        self.schedulers[identifier].update()

    @cherrypy.expose
    def create(self, identifier):
        body = cherrypy.request.body.read()
        reward, observation = unpack(body)

        if identifier not in self.agents:
            self.agents[identifier] = brica1.Agent()
            self.schedulers[identifier] = brica1.VirtualTimeScheduler(self.agents[identifier])
            self.components[identifier] = CnnDqnComponent(use_gpu=use_gpu, depth_image_dim=depth_image_dim * depth_image_count, feature_extractor=self.feature_extractor)
            self.agents[identifier].add_component('cnn_dqn_component', self.components[identifier])
            self.schedulers[identifier].update()

        self.components[identifier].get_in_port('observation').buffer = observation
        self.components[identifier].get_in_port('reward').buffer = np.array([reward])
        self.components[identifier].get_in_port('operation').buffer = np.array([0, 1, 0])

        self.schedulers[identifier].step()

        return str(self.components[identifier].get_out_port('action').buffer[0])

    @cherrypy.expose
    def step(self, identifier):
        body = cherrypy.request.body.read()
        reward, observation = unpack(body)

        if identifier not in self.agents:
            return str(-1)

        self.components[identifier].get_in_port('observation').buffer = observation
        self.components[identifier].get_in_port('reward').buffer = np.array([reward])
        self.components[identifier].get_in_port('operation').buffer = np.array([0, 0, 1])

        self.schedulers[identifier].step()

        return str(self.components[identifier].get_out_port('action').buffer[0])

    @cherrypy.expose
    def reset(self, identifier):
        body = cherrypy.request.body.read()
        reward, observation = unpack(body)

        if identifier not in self.agents:
            return str(-1)

        self.components[identifier].get_in_port('observation').buffer = observation
        self.components[identifier].get_in_port('reward').buffer = np.array([reward])
        self.components[identifier].get_in_port('operation').buffer = np.array([1, 1, 0])

        self.schedulers[identifier].step()

        return str(self.components[identifier].get_out_port('action').buffer[0])

def main(args):
    cherrypy.config.update({'server.socket_host': args.host, 'server.socket_port': args.port})
    cherrypy.quickstart(Root())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LIS Backend')
    parser.add_argument('--host', default='localhost', type=str, help='Server hostname')
    parser.add_argument('--port', default=8765, type=int, help='Server port number')
    args = parser.parse_args()

    main(args)

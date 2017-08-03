import argparse
import cPickle as pickle
import io
import os

import brica1
import cherrypy
import msgpack
import numpy as np
from PIL import Image
from PIL import ImageOps

from cognitive import interpreter
from ml.cnn_feature_extractor import CnnFeatureExtractor

from config import BRICA_CONFIG_FILE
from config.model import CNN_FEATURE_EXTRACTOR, CAFFE_MODEL, MODEL_TYPE


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


use_gpu = -1
depth_image_dim = 32 * 32
depth_image_count = 1
image_feature_dim = 256 * 6 * 6
image_feature_count = 1
feature_output_dim = (depth_image_dim * depth_image_count) + (image_feature_dim * image_feature_count)


class Root(object):
    def __init__(self, **kwargs):
        if os.path.exists(CNN_FEATURE_EXTRACTOR):
            print("loading... " + CNN_FEATURE_EXTRACTOR),
            self.feature_extractor = pickle.load(open(CNN_FEATURE_EXTRACTOR))
            print("done")
        else:
            self.feature_extractor = CnnFeatureExtractor(use_gpu, CAFFE_MODEL, MODEL_TYPE, image_feature_dim)
            pickle.dump(self.feature_extractor, open(CNN_FEATURE_EXTRACTOR, 'w'))
            print("pickle.dump finished")

        self.nb = interpreter.NetworkBuilder()
        f = open(BRICA_CONFIG_FILE)
        self.nb.load_file(f)
        self.agents = {}
        self.schedulers = {}
        self.v1_components = {}     # primary visual cortex
        self.vvc_components = {}    # visual what path
        self.bg_components = {}     # basal ganglia
        self.ub_components = {}     # Umataro box
        self.fl_components = {}     # frontal lobe
        self.mo_components = {}     # motor output
        self.rb_components = {}     # reward generator

    @cherrypy.expose()
    def flush(self, identifier):

        agent_builder = interpreter.AgentBuilder()
        self.agents[identifier] = agent_builder.create_agent(self.nb)
        modules = agent_builder.get_modules()
        self.schedulers[identifier] = brica1.VirtualTimeScheduler(self.agents[identifier])

        # set components
        self.v1_components[identifier] = modules['WBAH2017WBRA.Isocortex#V1'].get_component('WBAH2017WBRA.Isocortex#V1')
        self.vvc_components[identifier] = modules['WBAH2017WBRA.Isocortex#VVC'].get_component(
            'WBAH2017WBRA.Isocortex#VVC')
        self.bg_components[identifier] = modules['WBAH2017WBRA.BG'].get_component('WBAH2017WBRA.BG')
        self.ub_components[identifier] = modules['WBAH2017WBRA.UB'].get_component('WBAH2017WBRA.UB')
        self.fl_components[identifier] = modules['WBAH2017WBRA.Isocortex#FL'].get_component('WBAH2017WBRA.Isocortex#FL')
        self.mo_components[identifier] = modules['WBAH2017WBRA.MO'].get_component('WBAH2017WBRA.MO')
        self.rb_components[identifier] = modules['WBAH2017WBRA.RB'].get_component('WBAH2017WBRA.RB')

        # set feature_extractor
        self.vvc_components[identifier].set_model(self.feature_extractor)

        self.schedulers[identifier].update()

    @cherrypy.expose
    def create(self, identifier):
        body = cherrypy.request.body.read()
        reward, observation = unpack(body)

        agent_builder = interpreter.AgentBuilder()

        if identifier not in self.agents:

            # create agetns and schedulers
            self.agents[identifier] = agent_builder.create_agent(self.nb)
            modules = agent_builder.get_modules()
            self.schedulers[identifier] = brica1.VirtualTimeScheduler(self.agents[identifier])

            # set components
            self.v1_components[identifier] = modules['WBAH2017WBRA.Isocortex#V1'].get_component(
                'WBAH2017WBRA.Isocortex#V1')
            self.vvc_components[identifier] = modules['WBAH2017WBRA.Isocortex#VVC'].get_component(
                'WBAH2017WBRA.Isocortex#VVC')
            self.bg_components[identifier] = modules['WBAH2017WBRA.BG'].get_component('WBAH2017WBRA.BG')
            self.ub_components[identifier] = modules['WBAH2017WBRA.UB'].get_component('WBAH2017WBRA.UB')
            self.fl_components[identifier] = modules['WBAH2017WBRA.Isocortex#FL'].get_component(
                'WBAH2017WBRA.Isocortex#FL')
            self.mo_components[identifier] = modules['WBAH2017WBRA.MO'].get_component('WBAH2017WBRA.MO')
            self.rb_components[identifier] = modules['WBAH2017WBRA.RB'].get_component('WBAH2017WBRA.RB')

            # set feature_extractor
            self.vvc_components[identifier].set_model(self.feature_extractor)

            # set interval of each components
            self.vvc_components[identifier].interval = 1000
            self.bg_components[identifier].interval = 1000
            self.ub_components[identifier].interval = 1000
            self.mo_components[identifier].interval = 1000
            self.fl_components[identifier].interval = 1000

            # set offset
            self.vvc_components[identifier].offset = 0
            self.bg_components[identifier].offset = 1000
            self.fl_components[identifier].offset = 2000
            self.ub_components[identifier].offset = 3000
            self.mo_components[identifier].offset = 4000

            # set sleep
            self.vvc_components[identifier].sleep = 5000
            self.bg_components[identifier].sleep = 5000
            self.ub_components[identifier].sleep = 5000
            self.mo_components[identifier].sleep = 5000
            self.fl_components[identifier].sleep = 5000

            # self.agents[identifier].add_submodule()
            self.schedulers[identifier].update()

        # set observation in v1 for extracting feature vector using vcc
        self.v1_components[identifier].get_out_port('Isocortex#V1-Isocortex#VVC-Output').buffer = observation
        self.vvc_components[identifier].input(self.vvc_components[identifier].last_input_time)
        self.vvc_components[identifier].fire()
        self.vvc_components[identifier].output(self.vvc_components[identifier].last_output_time)
        features = self.vvc_components[identifier].get_out_port('Isocortex#VVC-BG-Output').buffer

        # agent start
        self.bg_components[identifier].get_in_port('Isocortex#VVC-BG-Input').buffer = features
        action = self.bg_components[identifier].start()
        self.schedulers[identifier].step()
        return str(action)

    @cherrypy.expose
    def step(self, identifier):
        body = cherrypy.request.body.read()
        reward, observation = unpack(body)

        if identifier not in self.agents:
            return str(-1)

        self.v1_components[identifier].get_out_port('Isocortex#V1-Isocortex#VVC-Output').buffer = observation
        self.rb_components[identifier].get_out_port('RB-Isocortex#FL-Output').buffer = np.array([reward])
        self.rb_components[identifier].get_out_port('RB-BG-Output').buffer = np.array([reward])
        self.schedulers[identifier].step()

        return str(self.mo_components[identifier].get_in_port('Isocortex#FL-MO-Input').buffer[0])

    @cherrypy.expose
    def reset(self, identifier):
        body = cherrypy.request.body.read()
        reward, observation = unpack(body)

        if identifier not in self.agents:
            return str(-1)

        current_time = self.schedulers[identifier].step()
        self.ub_components
        self.ub_components[identifier].end_episode()
        self.ub_components[identifier].output(current_time)
        self.bg_components[identifier].input(current_time)
        self.bg_components[identifier].agent_end(reward)

        return str(self.mo_components[identifier].get_in_port('Isocortex#FL-MO-Input').buffer[0])


def main(args):
    cherrypy.config.update({'server.socket_host': args.host, 'server.socket_port': args.port})
    cherrypy.quickstart(Root())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LIS Backend')
    parser.add_argument('--host', default='localhost', type=str, help='Server hostname')
    parser.add_argument('--port', default=8765, type=int, help='Server port number')
    args = parser.parse_args()

    main(args)

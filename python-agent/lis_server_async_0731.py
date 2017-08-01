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

from ml_module4 import BGComponent, VVCComponent, UmataroComponent, RBComponent, MOComponent, FLComponent
from cnn_feature_extractor import CnnFeatureExtractor

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

    identifier = str(dat['identifier'])
    reward = dat['reward']
    observation = {"image": image, "depth": depth}

    return identifier, reward, observation


use_gpu = 0
cnn_feature_extractor = 'alexnet_feature_extractor.pickle'
model = 'bvlc_alexnet.caffemodel'
model_type = 'alexnet'
image_feature_dim = 256 * 6 * 6
image_feature_count = 1
depth_image_dim = 32 * 32
depth_image_count = 1
feature_output_dim = (depth_image_dim * depth_image_count) + (image_feature_dim * image_feature_count)


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
        self.vvc_components = {}    # visual what path
        self.bg_components = {}     # basal ganglia
        self.ub_components = {}     # umataro box
        self.fl_components = {}     # frontal cortex
        self.mo_components = {}     # motor output
        self.rb_components = {}     # reward generator

    @cherrypy.expose
    def create(self):
        body = cherrypy.request.body.read()
        identifier, reward, observation = unpack(body)

        if identifier not in self.agents:
            self.agents[identifier] = brica1.Agent()
            self.schedulers[identifier] = brica1.VirtualTimeScheduler(self.agents[identifier])

            # set components
            self.vvc_components[identifier] = VVCComponent(n_input=1, n_output=feature_output_dim,
                                                            use_gpu=0, feature_extractor=self.feature_extractor)
            self.bg_components[identifier] = BGComponent(feature_output_dim, 1)
            self.ub_components[identifier] = UmataroComponent()
            self.fl_components[identifier] = FLComponent()
            self.mo_components[identifier] = MOComponent()
            self.rb_components[identifier] = RBComponent()

            # set interval of each components
            self.vvc_components[identifier].interval = 1000
            self.bg_components[identifier].interval = 1000
            self.ub_components[identifier].interval = 1000
            self.mo_components[identifier].interval = 1000
            self.fl_components[identifier].interval = 1000
            self.rb_components[identifier].interval = 1000

            # set offset
            self.vvc_components[identifier].offset = 0
            self.rb_components[identifier].offset = 1000
            self.bg_components[identifier].offset = 2000
            self.fl_components[identifier].offset = 3000
            self.ub_components[identifier].offset = 4000
            self.mo_components[identifier].offset = 5000

            # set sleep
            # self.vvc_components[identifier].sleep = 4500
            # self.bg_components[identifier].sleep = 4500
            # self.ub_components[identifier].sleep = 4500
            # self.mo_components[identifier].sleep = 4500
            # self.fl_components[identifier].sleep = 4500
            # self.rb_components[identifier].sleep = 4500

            self.vvc_components[identifier].sleep = 6000
            self.bg_components[identifier].sleep = 6000
            self.ub_components[identifier].sleep = 6000
            self.mo_components[identifier].sleep = 6000
            self.fl_components[identifier].sleep = 6000
            self.rb_components[identifier].sleep = 6000

            # connect port
            # from VVC to UB, BG(replay info), BG(state info)
            brica1.connect((self.vvc_components[identifier], 'Isocortex.VVC-UB-Output'),
                           (self.ub_components[identifier], 'Isocortex.VVC-UB-Input'))
            brica1.connect((self.vvc_components[identifier], 'Isocortex.VVC-BG-Output'),
                           (self.bg_components[identifier], 'Isocortex.VVC-BG-Input'))
            # brica1.connect((self.vvc_components[identifier], 'Isocortex.VVC-BG-Output'),
            #                (self.bg_components[identifier], 'Isocortex.VVC-BG-Input'))
            # from BG to FL
            brica1.connect((self.bg_components[identifier], 'BG-Isocortex.FL-Output'),
                           (self.fl_components[identifier], 'BG-Isocortex.FL-Input'))
            # from UB to BG
            brica1.connect((self.ub_components[identifier], 'UB-BG-Output'),
                           (self.bg_components[identifier], 'UB-BG-Input'))
            # from FL to BG, MO, UB
            # brica1.connect((self.fl_components[identifier], 'Isocortex.FL-BG-Output'),
            #                (self.bg_components[identifier], 'Isocortex.FL-BG-Input'))
            brica1.connect((self.fl_components[identifier], 'Isocortex.FL-MO-Output'),
                           (self.mo_components[identifier], 'Isocortex.FL-MO-Input'))
            brica1.connect((self.fl_components[identifier], 'Isocortex.FL-UB-Output'),
                           (self.ub_components[identifier], 'Isocortex.FL-UB-Input'))
            # from RB to FL, BG
            brica1.connect((self.rb_components[identifier], 'RB-Isocortex.FL-Output'),
                           (self.fl_components[identifier], 'RB-Isocortex.FL-Input'))
            brica1.connect((self.rb_components[identifier], 'RB-BG-Output'),
                           (self.bg_components[identifier], 'RB-BG-Input'))


            # add components
            self.agents[identifier].add_component('vvc_components', self.vvc_components[identifier])
            self.agents[identifier].add_component('bg_components', self.bg_components[identifier])
            self.agents[identifier].add_component('ub_components', self.ub_components[identifier])
            self.agents[identifier].add_component('fl_components', self.fl_components[identifier])
            self.agents[identifier].add_component('rb_components', self.rb_components[identifier])
            self.agents[identifier].add_component('mo_components', self.mo_components[identifier])

            self.schedulers[identifier].update()

        # set feature and agent start
        self.vvc_components[identifier].get_in_port('VVC-Env-Input').buffer = observation
        self.vvc_components[identifier].fire()
        self.vvc_components[identifier].output(self.vvc_components[identifier].last_output_time)
        features = self.vvc_components[identifier].get_out_port('Isocortex.VVC-BG-Output').buffer

        self.bg_components[identifier].get_in_port('Isocortex.VVC-BG-Input').buffer = features
        action = self.bg_components[identifier].start()
        self.schedulers[identifier].step()
        return str(action)

    @cherrypy.expose
    def step(self):
        body = cherrypy.request.body.read()
        identifier, reward, observation = unpack(body)

        if identifier not in self.agents:
            return str(-1)

        self.vvc_components[identifier].get_in_port('VVC-Env-Input').buffer = observation
        self.rb_components[identifier].get_in_port('RB-ENV-Input').buffer = np.array([reward])
        # print 'queue: ', self.schedulers[identifier].event_queue.queue
        self.schedulers[identifier].step()

        return str(self.mo_components[identifier].get_out_port('MO-ENV-Output').buffer[0])

    @cherrypy.expose
    def reset(self):
        body = cherrypy.request.body.read()
        identifier, reward, observation = unpack(body)

        if identifier not in self.agents:
            return str(-1)

        current_time = self.schedulers[identifier].step()
        self.ub_components
        self.ub_components[identifier].end_episode()
        self.ub_components[identifier].output(current_time)
        self.bg_components[identifier].input(current_time)
        self.bg_components[identifier].agent_end(reward)

        return str(self.mo_components[identifier].get_out_port('MO-ENV-Output').buffer[0])

def main(args):
    cherrypy.config.update({'server.socket_host': args.host, 'server.socket_port': args.port})
    cherrypy.quickstart(Root())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LIS Backend')
    parser.add_argument('--host', default='localhost', type=str, help='Server hostname')
    parser.add_argument('--port', default=8765, type=int, help='Server port number')
    args = parser.parse_args()

    main(args)

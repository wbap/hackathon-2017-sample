# -*- coding: utf-8 -*-

'''
SmoothGrad: https://arxiv.org/pdf/1706.03825.pdf
コードはQiitaに投稿された以下の記事を参考にしています
https://qiita.com/hogemon/items/cdff53174dd89034d18d
'''

import sys
sys.path.append('..')

from config.model import CAFFE_MODEL
from config.log import APP_KEY
import chainer
import chainer.functions as F
from chainer.variable import Variable
from chainer.links import VGG16Layers
from chainer.links import caffe
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

chainer.config.train=False
chainer.config.enable_backprop=True

# model = VGG16Layers()
model = caffe.CaffeFunction(CAFFE_MODEL)


image = Image.open("input_image.png")
image = image.resize((224, 224))

sample_size = 100
noise_level = 0.2  # 20%
sigma = noise_level * 255.0

grad_list = []

for _ in range(sample_size):
    x = np.asarray(image, dtype=np.float32)
    
    # RGB to BGR
    x = x[:, :, ::-1]
    
    # 平均をひく
    x -= np.array([103.99, 116.779, 123.68], dtype=np.float32)
    x = x.transpose((2, 0, 1))
    x = x[np.newaxis]
    
    # ノイズを追加
    x += sigma * np.random.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    x = Variable(np.asarray(x))

    # FPして最終層を取り出す
    # y = model(x, layers=['prob'])['prob']  # for VGG16
    y,  = model(inputs={'data':x}, outputs=['fc8'])  # for AlexNet

    # 予測が最大のラベルでBP
    t = np.zeros((x.data.shape[0]), dtype=np.int32)
    t[:] = np.argmax(y.data)
    t = Variable(np.asarray(t))
    loss = F.softmax_cross_entropy(y, t)
    loss.backward()

    # 勾配をリストに追加
    grad = np.copy(x.grad)
    grad_list.append(grad)

    # 勾配をクリア
    model.cleargrads()

G = np.array(grad_list)
M = np.mean(np.max(np.abs(G), axis=2), axis=0)
M = np.squeeze(M)

plt.imshow(M, "gray")
plt.savefig("output_saliency.png")
plt.show()
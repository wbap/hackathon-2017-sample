#!/bin/bash

echo "download caffemodel..."
dir=agent/model
if [ ! -d $dir ]; then
  mkdir $dir
fi
curl -o agent/model/bvlc_alexnet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel

curl -f -L -o agent/model/ilsvrc_2012_mean.npy https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/ilsvrc_2012_mean.npy

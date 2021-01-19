import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from onnx_tf.backend import prepare
import tensorflow as tf
from torch.autograd import Variable
import h5py
from tensorflow.python.keras import layers
from onnx2keras import onnx_to_keras

# class AlexNet(nn.Module):

#     def __init__(self, num_classes=10):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.classifier = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# # step 1, load pytorch model and export onnx during running.
# resume='checkpoints_100cifar_alexnetNirvana/bestepoch'


# model =AlexNet(100)
# model = torch.nn.DataParallel(model)
# criterion = nn.CrossEntropyLoss()

# print('==> Resuming from checkpoint..')
# assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
# checkpoint = os.path.dirname(resume)
# checkpoint = torch.load(resume)
# model.load_state_dict(checkpoint['state_dict'])

# dummy_input = Variable(torch.randn(1, 3, 32, 32)) # nchw
# # dummy_output = model.module(dummy_input)
# # print(dummy_output)

# onnx_filename = "model.onnx"
# torch.onnx.export(model.module, dummy_input, onnx_filename, output_names=['test_output'])

model_onnx = onnx.load('./model.onnx')

k_model = onnx_to_keras(model_onnx, input_names=['0'])

# Export model as .pb file
# tf_rep.export_graph('model_tf.h5')

# def load_pb(path_to_pb):
#     with tf.gfile.GFile(path_to_pb, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='')
#         return graph

# tf_graph = load_pb('model_tf.pb')
# sess = tf.Session(graph=tf_graph)


tf.keras.models.save_model(k_model, 'model_tf2.h5')
print('fin')
tf.keras.models.load_model('model_tf2.h5')
# # Show tensor names in graph
# for op in tf_graph.get_operations():
#   print(op.values())

# output_tensor = tf_graph.get_tensor_by_name('test_output:0')
# input_tensor = tf_graph.get_tensor_by_name('0:0')

# output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
# print(output)
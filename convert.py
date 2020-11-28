import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
import torchvision.datasets as datasets
from onnx_tf.backend import prepare
import tensorflow as tf

# step 1, load pytorch model and export onnx during running.
resume='checkpoints_100cifar_alexnetNirvana/bestepoch'
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

dataloader = datasets.CIFAR100

testset = dataloader(root='./data100', train=False, download=False, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)


model =AlexNet(100)
model = torch.nn.DataParallel(model).cuda()
criterion = nn.CrossEntropyLoss()

print('==> Resuming from checkpoint..')
assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
checkpoint = os.path.dirname(resume)
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint['state_dict'])

dummy_input = torch.from_numpy(testloader[0].reshape(1, -1)).float().to(device) # nchw
onnx_filename = "model.onnx"
torch.onnx.export(model, dummy_input,
                    onnx_filename,
                    verbose=True)

model_onnx = onnx.load('./models/model_simple.onnx')

tf_rep = prepare(model_onnx)

# Export model as .pb file
tf_rep.export_graph('model_simple.pb')
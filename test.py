import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from pytorch2keras.converter import pytorch_to_keras
import torchvision.models as models
import tensorflow as tf


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def convert(model_name):
    input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
    input_var = Variable(torch.FloatTensor(input_np))
    resume='whitebox_attack_target_models/checkpoints_100cifar_alexnet' + model_name + '/bestepoch'
    model =AlexNet(100)
    model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = os.path.dirname(resume)
    checkpoint = torch.load(resume, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])


    model.eval()
    k_model = pytorch_to_keras(model.module, input_var, [(3, 32, 32,)], verbose=True, change_ordering=True)
    k_model.save(model_name+'model')
    print('saved')
    tf.keras.models.load_model(model_name+'model')
    print('loaded')

    for i in range(3):
        input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)
        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(np.transpose(input_np, [0, 2, 3, 1]))
        error = np.max(pytorch_output - keras_output)
        print('error -- ', error)  # Around zero :)



if __name__ == '__main__':

    names = ['baseline', 'Nirvana_n10N5', 'Nirvana_n10N7', 'Nirvana_n20N5', 'Nirvana_n20N10', 'Nirvana_n30N10', 'Nirvana_n30N10k0.5']

    for name in names:
        convert(name)
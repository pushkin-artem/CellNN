from enum import Enum, auto

import numpy as np
import tensorflow as tf
from tensorflow import keras
from numpy import argmax as np_argmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
torch.cuda.is_available()

class CNNType(Enum):
    CNN = auto()
    CellNN = auto()

class ClassifierInterface():

    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

# CNNType = Enum('CNNType', 'CNN QNet')
class BaseNet(ClassifierInterface):

    def __init__(self, input_shape, output_shape):
        self._input_shape = input_shape
        self._output_shape = output_shape

        input_tensor, outputs = self._build_graph()
        self._create_model(input_tensor, outputs)

    def _create_model(self, input_tensor, outputs):
        self._model = keras.Model(inputs=input_tensor, outputs=outputs)
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def _build_graph(self):
        raise NotImplementedError("Model build function is not implemented")

    def predict(self, x):
        predictions = self._model.predict(x)
        return np_argmax(predictions, axis=-1)

    def fit(self, x, y=None, *, validation_data=None, batch_size=None, epochs=10):
        self._model.fit(x, y, validation_data=validation_data, batch_size=batch_size, epochs=epochs)


    def summary(self):
        self._model.summary()

    def evaluate(self, x, y=None):
        self._model.evaluate(x, y)


class CNN(BaseNet):

    def __init__(self, net_type, input_shape, classes, weights="imagenet"):
        self._net_type = net_type
        self._weights = weights
        super(CNN, self).__init__(input_shape, classes)
        # tf.compat.v1.reset_default_graph()

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor

        print(input_tensor)

        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)
        if len(self._input_shape) == 2 or len(self._input_shape) == 3 and self._input_shape[2] == 1:
            x = keras.layers.Lambda(lambda tens: tf.image.grayscale_to_rgb(tens))(x)

        x = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=3)(x)
        x = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=3)(x)
        x = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=3)(x)

        # add end node
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(100, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(self._output_shape, activation='softmax', name='predictions')(x)
        return input_tensor, x


class CeNNLayer(nn.Module):
    def __init__(self, InDepth=1, OutDepth=1, TimeStep=0.1, IterNum=100):
        super(CeNNLayer, self).__init__()
        self.rescale = nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1)
        self.A = nn.Conv2d(OutDepth, OutDepth, kernel_size=3, padding=1)
        self.B = nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1)
        self.Z = nn.Parameter(torch.randn(OutDepth))
        # self.Z =self.Zsingle.view(1,OutDepth,1,1).repeat(16,1,28,28)
        self.TimeStep = 0.1
        self.IterNum = 10

    def NonLin(self, x, alpha=0.01):
        y = torch.min(x, 1 + alpha * (x - 1))
        y = torch.max(y, -1 + alpha * (y + 1))
        return y

    def forward(self, x):
        InputCoupling = self.B(x)
        Zreshaped = self.Z.view(1, InputCoupling.shape[1], 1, 1).repeat(InputCoupling.shape[0], 1,
                                                                        InputCoupling.shape[2], InputCoupling.shape[3])
        InputCoupling = InputCoupling + Zreshaped
        x = self.rescale(x)
        for step in range(self.IterNum):
            Coupling = self.A(self.NonLin(x)) + InputCoupling
            x = x + self.TimeStep * (-x + Coupling)
        return self.NonLin(x)


class CellNN(nn.Module):

    def __init__(self):
        super(CellNN, self).__init__()
        self.Layer1 = CeNNLayer(1, 16)
        self.Layer2 = CeNNLayer(16, 32)
        self.Layer3 = CeNNLayer(32, 10)

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        return x


def SquaredDiff(NetOut, Labels):
    SquaredDiff = torch.mean(torch.square(NetOut - Labels))
    return SquaredDiff


def SofMaxLossArray(NetOut, Labels):
    preds = torch.mean(NetOut, [2, 3])
    preds = torch.softmax(preds, -1)
    loss = torch.log(torch.diag(preds[:, Labels]))
    loss = -torch.mean(loss)
    return loss


def train(model, epoch, trainx, trainy, testx, testy, opt):
    for i in range(trainy):
        model.train()
        data = trainx[i].cuda()
        label = trainy[i].cuda()
        opt.zero_grad()
        preds = model(data)

        # loss = SquaredDiff(preds,ImgLabels)
        loss = SofMaxLossArray(preds, label)

        loss.backward()
        opt.step()
        predind = torch.sum(preds, [2, 3])
        predind = predind.data.max(1)[1]
        acc = predind.eq(label.data).cpu().float().mean()

        if i % 100 == 0:

            print("Train Loss: " + str(loss.item()) + " Acc: " + str(acc.item()))

            # run independent test
            model.eval()  # set model in inference mode (need this because of dropout)
            test_loss = 0
            correct = 0
            SampleNum = 0
            for i in range(testy):
                datab = testx[i].cuda()
                label = testy[i].cuda()
                with torch.no_grad():
                    output = model(datab)
                    pred = torch.sum(output, [2, 3]).data.max(1)[1]
                    correct += pred.eq(label.data).cpu().sum()
                SampleNum += data.shape[0]
            accuracy = correct.item() / SampleNum
            print("Test Acc: " + str(accuracy))


if __name__ == '__main__':
    nn = CNN('a', (59, 100), 2)
    nn.summary()

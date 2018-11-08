import os
import time
import warnings
import mxnet as mx
import numpy as np
from mxnet import gluon, nd, autograd
from audiofolderdataset import AudioFolderDataset
from sklearn.preprocessing import LabelEncoder
from transforms import  Loader, MFCC


# Defining a neural network with number of labels
def get_net(num_labels = 10):
    
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(256, activation="relu")) # 1st layer (256 nodes)
        net.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer
    net.add(gluon.nn.Dense(num_labels))
    net.collect_params().initialize(mx.init.Xavier())
    return net


# Defining a function to evaluate accuracy
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for _, (data, label) in enumerate(data_iterator):        
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        predictions= predictions.reshape((-1,1))
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


def train(epochs = 30, batch_size=32):
    
    # Make a dataset from the local folder containing Audio data
    print("\nMaking an Audio Dataset...\n")
    tick = time.time()
    aud_dataset = AudioFolderDataset('./Train', has_csv=True, train_csv='./train.csv')
    tock = time.time()

    print("Loading the dataset took ",(tock-tick), " seconds.")
    print("\n=======================================\n")

    # Get the model to train
    net = get_net(len(aud_dataset.synsets))
    print("Model - Neural Network Generated!\n")
    print("=======================================\n")

    #Define the loss - Softmax CE Loss
    softmax_loss = gluon.loss.SoftmaxCELoss(from_logits=False, sparse_label=True)
    print("Loss function initialized!\n")
    print("=======================================\n")

    #Define the trainer with the optimizer
    trainer = gluon.Trainer(net.collect_params(), 'adadelta')
    print("Optimizer - Trainer function initialized!\n")
    print("=======================================\n")


    print("Loading the dataset to the Gluon's OOTB Dataloader...")

    #Getting the data loader out of the AudioDataset and passing the transform
    aud_transform = gluon.data.vision.transforms.Compose([Loader(), MFCC()])
    tick = time.time()

    audio_train_loader = gluon.data.DataLoader(aud_dataset.transform_first(aud_transform,lazy=False), 
                                                batch_size=32, shuffle=True)
    tock=time.time()
    print("Time taken to load data and apply transform here is ",(tock-tick)," seconds.")
    print("=======================================\n")


    print("Starting the training....\n")
    # Training loop
    tick = time.time()
    batch_size = batch_size
    num_examples = len(aud_dataset)
    
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(audio_train_loader):
            with autograd.record():
                output = net(data)
                loss = softmax_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += mx.nd.sum(loss).asscalar()
        
        if e%5==0:
            train_accuracy = evaluate_accuracy(audio_train_loader, net)
            print("Epoch %s. Loss: %s Train accuracy : %s " % (e, cumulative_loss/num_examples, train_accuracy))
            print("\n------------------------------\n")

    train_accuracy = evaluate_accuracy(audio_train_loader, net)
    tock = time.time()
    print("\nFinal training accuracy: ",train_accuracy)

    print("Training the sound classification for ",epochs," epochs, MLP model took ",(tock-tick)," seconds")
    print("====================== END ======================\n")


if __name__ == '__main__':
    train(epochs=30, batch_size=32)
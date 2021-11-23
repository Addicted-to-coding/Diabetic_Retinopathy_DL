from torch.utils.data import Dataset
from datagenerator import *
import pandas as pd
import numpy as np
import pickle
from torch.utils import data
import torch
import random
import logging
import matplotlib.pyplot as plt
from os import path
import torchvision.models as models
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--bsz', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--cutmix', type=bool, default=False)
parser.add_argument('--mixup', type=bool, default=False)


args = parser.parse_args()

log_file = f'log_{args.model}.txt'
model_file = f'{args.model}_{args.run}.pt'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file, 'a'))
print = logger.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.run)

class generate_model(torch.nn.Module):

    def __init__(self, base_model, hidden = 128, num_outs = 5):
        super(generate_model, self).__init__()

        # create a dummy input
        dummy_input = torch.rand(1, 3, 320, 320)
        out = base_model(dummy_input.to(device).float())
        input_size = out.shape[1]

        self.base_model = base_model
        self.fc = torch.nn.Sequential(
                                torch.nn.Linear(input_size, hidden), 
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden, num_outs)
                                )

    def forward(self, x):
        x = self.base_model(x)
        pred = self.fc(x)
        return pred

def xent():
    def loss(pred, y):
        if len(y.shape) == 1:
            lss = torch.nn.CrossEntropyLoss()
            ll = lss(pred, y)
        else:
            sm = torch.nn.Softmax(dim = -1)
            pred = sm(pred)
            ll = - torch.mean(torch.sum(y * torch.log(pred), 1))
        return ll
    return loss

def accuracy(model, loader):

    model.eval()
    total_correct = 0
    size = 0
    with torch.no_grad():
        for batch, data_batch in enumerate(loader):
            X, y = data_batch[0].to(device).float(), data_batch[1].to(device)
            predictions = model(X)
            predictions = torch.argmax(predictions, axis = -1)

            correct = (predictions == y).sum()
            total_correct += correct.item()
            size += y.shape[0]

    acc = 100 * total_correct / size
    return(acc)

def evaluate(model, objective, loader):

    model.eval()
    total_loss = 0
    size = 0
    
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(loader):

            X, y = data_batch[0].to(device).float(), data_batch[1].to(device)
            # X, y = map(lambda t: t.to(device).float(), (X, y))

            prediction = model(X)
            total_loss += objective(prediction, y) * X.shape[0]
            size += X.shape[0]

    total_loss = total_loss / size

    return total_loss

def train(model, objective, optimizer, train_loader, valid_loader, epochs = 1, save_interval = 1, patience = 3):
  
    model.train()

    val_loss = 1e7
    pat = patience

    for epoch in range(1, epochs + 1):
        train_loss = 0
        size = 0

        for batch_idx, data_batch in enumerate(train_loader):

            optimizer.zero_grad()

            train_X, train_y = data_batch[0].to(device).float(), data_batch[1].to(device)
            # train_X, train_y = map(lambda t: t.to(device).float(), (train_X, train_y))

            prediction = model(train_X)
            loss = objective(prediction, train_y)
            loss.backward()

            train_loss += loss.item() * train_X.shape[0]
            size += train_X.shape[0]

            optimizer.step()

        avg_loss = train_loss / size

        rt_val_loss = evaluate(model, objective, valid_loader)
        model.train()

        print(f'Epoch {epoch}: Training Loss : {avg_loss} | Validation loss : {rt_val_loss}')

        if rt_val_loss < val_loss:
            val_loss = rt_val_loss
            torch.save(model.state_dict(), model_file)
            pat = patience
        else:
            pat = pat - 1
            if pat == 0:
                print('Training Complete --> Exiting')
                break


train_X, train_y, valid_X, valid_y, test_X, test_y = get_dataset("/u/home/h/hbansal/M226/train.csv", 
                                                                 '/u/home/h/hbansal/M226/train_images_smol/')

bsz = args.bsz

train_dataset = DataGenerator(train_X, train_y, args.mixup, args.cutmix)
train_loader = data.DataLoader(train_dataset, batch_size= bsz, shuffle = True)

valid_dataset = DataGenerator(valid_X, valid_y, args.mixup, args.cutmix)
valid_loader = data.DataLoader(valid_dataset, batch_size = bsz, shuffle = True)

test_dataset = DataGenerator(test_X, test_y)
test_loader = data.DataLoader(test_dataset, batch_size = bsz, shuffle = False)

if 'resnet' in args.model: 
    basemodel = models.resnet18().to(device)
elif 'alexnet' in args.model:
    basemodel = models.alexnet().to(device)
elif 'vgg' in args.model:
    basemodel = models.vgg16().to(device)
elif 'densenet' in args.model:
    basemodel = models.densenet161().to(device)
else:
    print(f'{args.model} not found! Exiting!')
    sys.exit()

model = generate_model(base_model = basemodel).to(device)

lr = args.lr

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr)

criterion = xent()

if args.train:
    train(model, criterion, optimizer, train_loader, valid_loader, epochs = args.epochs)

model.load_state_dict(torch.load(model_file))
loss_ = evaluate(model, criterion, test_loader)
print(f'test loss is {loss_.item()}')

# train_accuracy = accuracy(model, train_loader)
# print(f'training accuracy: {train_accuracy}')

# valid_accuracy = accuracy(model, valid_loader)
# print(f'validing accuracy: {valid_accuracy}')

test_accuracy = accuracy(model, test_loader)
print(f'testing accuracy: {test_accuracy}')
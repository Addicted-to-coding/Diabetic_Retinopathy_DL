import torch.nn as nn
import torch
from torchvision import datasets, transforms, models
import logging

log_file = 'simclr_pretraining.txt'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file, 'a'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sm = nn.Softmax(dim = 1)

class SimCLR(nn.Module):

    def __init__(self, basemodel, hidden = 128):
        super(SimCLR, self).__init__()

        self.f = basemodel

        #temp
        temp = torch.rand(1, 3, 160, 160).to(device).float()
        out = self.f(temp)
        in_size = out.shape[1]

        self.h = nn.Sequential(
                    nn.Linear(in_size, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden // 2)  
                    )

        self.gaussian_blur = transforms.Compose([transforms.GaussianBlur(5)])
        self.color_jitter = transforms.Compose([transforms.ColorJitter()])
    
    def augmentation(self, x, n = 1):

        if n:
            return self.gaussian_blur(x)
        return self.color_jitter(x)

    def forward(self, x, pretraining = False):

        if pretraining:
            augmented_1 = self.augmentation(x, n = 1)
            augmented_2 = self.augmentation(x, n = 2)

            f1 = self.f(augmented_1)
            h1 = self.h(f1) 
            h1 = nn.functional.normalize(h1, dim = 1, p = 2)

            f2 = self.f(augmented_2)
            h2 = self.h(f2)
            h2 = nn.functional.normalize(h2, dim = 1, p = 2)

            return f1, f2, h1, h2
        
        else:

            return self.f(x)

def contrastive_loss(b1, b2, T = 1):

    # b1 is of the form (b, k)
    # b2 is of the form (b, k)
    # ith element of b1 and b2 are same

    similarity = torch.mm(b1, b2.t()) / T
    out = sm(similarity)
    out = -torch.log(out)
    loss = torch.mean(torch.diagonal(out))

    return loss

def evaluate_simclr(model, loader):

    model.eval()
    total_loss = 0
    size = 0
    
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(loader):
            X, y = data_batch[0].to(device).float(), data_batch[1].to(device)
            _,_, h1, h2 = model(X, True)
            total_loss += contrastive_loss(h1, h2) * X.shape[0]
            size += X.shape[0]

    total_loss = total_loss / size

    return total_loss

def train_simclr(model, optimizer, train_loader, valid_loader, model_file, epochs = 1, save_interval = 1, patience = 3):
  
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
            _, _, h1, h2 = model(train_X, True)
            loss = contrastive_loss(h1, h2)
            loss.backward()

            train_loss += loss.item() * train_X.shape[0]
            size += train_X.shape[0]

            optimizer.step()

        avg_loss = train_loss / size

        rt_val_loss = evaluate_simclr(model, valid_loader)
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


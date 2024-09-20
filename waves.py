'''
Code for predicting the magnitudes of waves.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset

import os

from models.s4.s4d import S4D
from tqdm.auto import tqdm
import scipy.io as mlio

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

batch_size = 64
epochs = 100

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Data
print('==> Preparing data..')

def get_data(bs_train,bs_test):
    
    class ncosine(torch.utils.data.Dataset):
        def __init__(self, L, N, seed=1):
            super(ncosine, self).__init__()
     
            dt = 0.01
            
            np.random.seed(1)
            f = np.linspace(1, 2**12, num=100)
            np.random.seed(seed)
            s = lambda x, f: np.cos(f * x)
            
            X = []
            y = []
            for i in range(N):
                for j in range(100):
                    noise = np.random.standard_normal(L) * 1
                    X.append(s(f[j], dt * np.arange(L)) + noise)
                    y.append(j)
                
            self.X = np.vstack(X)
            self.y = np.vstack(y)   
            
            print(self.X.shape)
            print(self.y.shape)
            
            self.len = self.X.shape[0]
    
    
        def __getitem__(self, index):
            return torch.from_numpy(self.X[index]).float(), torch.from_numpy(self.y[index]).long()
    
        def __len__(self):
            return self.len


class WaveDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,label):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        waves = mlio.loadmat('waves.mat')
        if label == 'train':
            self.data = np.array(waves['X'])
            self.label = np.array(waves['Y'][:,0:4])
        elif label == 'test':
            self.data = np.array(waves['X_test'])
            self.label = np.array(waves['Y_test'][:,0:4])
        else:
            self.data = np.array(waves['x_test2'])
            self.label = np.array(waves['y_test2'][:,0:4])
        self.ndata = self.label.shape[0]
            
    def __len__(self):
        return self.ndata

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx,:]), torch.tensor(self.label[idx,:])


# Dataloaders
trainloader = torch.utils.data.DataLoader(
    WaveDataset('train'), batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(
    WaveDataset('test'), batch_size=batch_size, shuffle=False, num_workers=0)

# Model
print('==> Building model..')

confignum = 4
# model = S4D(alpha = 1, beta = 0) # 1
# model = S4D(alpha = 100, beta = 1) # 2
# model = S4D(alpha = 0.01, beta = -1) # 3
model = S4D(alpha = 10, beta = 0.5) # 4

model = model.to('cuda')

dic = {}

dic['encoder'] = model.encoder.weight.data.cpu().detach().numpy()
dic['decoder'] = model.decoder.weight.data.cpu().detach().numpy()
dic['A'] = -np.exp(model.kernel.log_A_real.cpu().detach().numpy()) - 1j*model.kernel.A_imag.cpu().detach().numpy()
dic['C'] = torch.view_as_complex(model.kernel.C).cpu().detach().numpy()
dic['D'] = model.D.cpu().detach().numpy()
mlio.savemat('S4D_waves_init_' + str(confignum) + '.mat',dic)


criterion = nn.MSELoss()

all_parameters = list(model.parameters())
params = [p for p in all_parameters if not hasattr(p, "_optim")]
optimizer = optim.AdamW(params, lr=0.001, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

# Add parameters with special hyperparameters
hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
hps = [
    dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
]  # Unique dicts
for hp in hps:
    params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
    optimizer.add_param_group(
        {"params": params, **hp}
    )
    
keys = sorted(set([k for hp in hps for k in hp.keys()]))
for i, g in enumerate(optimizer.param_groups):
    group_hps = {k: g.get(k, None) for k in keys}
    print(' | '.join([
        f"Optimizer group {i}",
        f"{len(g['params'])} tensors",
    ] + [f"{k} {v}" for k, v in group_hps.items()]))

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train():
    model.train()
    train_loss = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.4f' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1))
        )


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        loss_acc = torch.zeros((4))
        for batch_idx, (inputs, targets) in pbar:
            inputs = inputs.to('cuda')
            outputs, _ = model(inputs)
            outputs = outputs.cpu()
            loss_acc += torch.mean(torch.abs(outputs-targets),0)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.4f' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1))
            )
    return eval_loss/(batch_idx+1), loss_acc / (batch_idx+1)

loss_epochs = []
loss_vec_epochs = []
if __name__ == '__main__':
    pbar = tqdm(range(start_epoch, epochs))
    for epoch in pbar:
        if epoch == 0:
            pbar.set_description('Epoch: %d' % (epoch))
        else:
            pbar.set_description('Epoch: %d | Loss: %1.4f' % (epoch, loss))
        train()
        loss, loss_vec = eval(epoch, testloader)
        scheduler.step()
        loss_epochs.append(loss)
        loss_vec_epochs.append(loss_vec)
        print(loss_vec)
        dic = {}
        dic['encoder'] = model.encoder.weight.data.cpu().detach().numpy()
        dic['decoder'] = model.decoder.weight.data.cpu().detach().numpy()
        dic['A'] = -np.exp(model.kernel.log_A_real.cpu().detach().numpy()) - 1j*model.kernel.A_imag.cpu().detach().numpy()
        dic['C'] = torch.view_as_complex(model.kernel.C).cpu().detach().numpy()
        dic['D'] = model.D.cpu().detach().numpy()
        mlio.savemat('S4D_waves_epoch' + str(epoch) + '_' + str(confignum) + '.mat',dic)
        print(-np.exp(model.kernel.log_A_real.cpu().detach().numpy()) - 1j*model.kernel.A_imag.cpu().detach().numpy())

print(loss_vec_epochs)
print(loss_epochs)
dic = {}
dic['loss'] = np.array(loss_epochs)
dic['loss_vec'] = np.array(loss_vec_epochs)
mlio.savemat('HOPE_waves_loss_' + str(confignum) + '.mat',dic)
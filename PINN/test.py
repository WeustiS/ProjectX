import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import util

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=1, hidden_size=36, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=36, hidden_size=1, batch_first=True)

    def forward(self, x):
        interm_out, _ = self.lstm1(x)
        interm_out = F.relu(interm_out)
        interm_out, _ = self.lstm2(interm_out)
        return interm_out

from torch.autograd import grad

L2 = nn.MSELoss()

class Aux_N(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        return self.layers(x)

class W_PDE(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (S, S_0, Tau)
        self.layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.h = Aux_N()
        self.g = Aux_N()
    
    # x: [batch, 3]
    def forward(self, x):
        return self.layers(x)
    
    # Enforces the PDE's structure
    # x: [batch, 3]
    # sigma: [batch]
    # r: [batch]
    def pde_structure(self, x, return_forward=True):
        x.requires_grad = True
        V = self.forward(x)
        
        Gradient, _ = second_order_partials(V, x)
        dXdXt =  Gradient[:,0]
        
        # Might break
        h = dXdXt - self.h(x) * (self.g(x) - x[:,0])
        
        if return_forward:
            return h, V
        else:
            return h

def second_order_partials(f, wrt, create_graph=True):
    gradient = grad(f, wrt, create_graph=create_graph, grad_outputs=torch.ones(f.shape).to(f.device))[0]
    hessian = grad(gradient, wrt, create_graph=create_graph, grad_outputs=torch.ones(gradient.shape).to(gradient.device))[0]
    
    return gradient, hessian

def set_seed(seed, device=None):
    import numpy as np
    import torch
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
data = np.load("data.npy")

SEQUENCE_LEN = 12

# Use wind speed, humidity, air pressure for baseline
speed_data = data[:, [0, 3, 4]]
# Normalize the data
speed_data = (speed_data - speed_data.mean(axis = 0)) / speed_data.std(axis = 0)
wind_speed_data = speed_data[:, 0]

# Get inputs and labels
X_data, Y_data = [], []
for i in range(0, len(speed_data) - 2*SEQUENCE_LEN - 1):
  X_data.append(speed_data[i:(i + SEQUENCE_LEN)])
  Y_data.append(speed_data[(i + SEQUENCE_LEN):(i + 2*SEQUENCE_LEN)])

X_data, Y_data = np.array(X_data), np.array(Y_data)
#Y_data = np.expand_dims(Y_data, axis = -1)

# Convert NumPy to Torch
X_data, Y_data = torch.from_numpy(X_data[:,:,0:1]), torch.from_numpy(Y_data[:,:,0:1])
    
# Shuffle
perm = np.random.permutation(len(X_data))
X_data = X_data[perm]
Y_data = Y_data[perm]

# Val split
xtrain = X_data[:-800].float()
ytrain = Y_data[:-800].float()
xval = X_data[-800:].float()
yval = Y_data[-800:].float()


# Setup parameters, model, and optimizer
seed = 0
epochs = 500
lr = 1e-3

criteria = nn.MSELoss()
#model = W_PDE()
model = Model()
model = model.cuda()
#optimizer = optim.LBFGS(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=lr)



# Trainin loop
print("Optimizing %d parameters on %s" % (util.count_parameters(model), 'cuda'))
time.sleep(0.5)
for epoch in range(epochs):
    running_loss = 0
    train_p_bar = tqdm(range(len(xtrain) // 128))
    for batch_idx in train_p_bar:
        xbatch = xtrain[batch_idx*128:(batch_idx+1)*128].cuda()
        ybatch = ytrain[batch_idx*128:(batch_idx+1)*128].cuda()
        
        #pde, y_hat = model.pde_structure(xbatch)
        #loss = criteria(pde, torch.zeros(pde.shape).to(pde.device)) + criteria(y_hat, ybatch)
        
        y_hat = model(xbatch)
        loss = criteria(y_hat, ybatch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_p_bar.set_description("Loss %.6f" % loss.item())

    running_loss /= (len(xtrain) // 128 + 1)
      
    running_val_loss = 0
    for batch_idx in range(len(xval) // 128):
        xbatch = xval[batch_idx:(batch_idx+1)*128].cuda()
        ybatch = yval[batch_idx:(batch_idx+1)*128].cuda()
        
        with torch.no_grad():
            y_hat = model(xbatch)
            loss = criteria(y_hat, ybatch)
        
        running_val_loss += loss.item()
    running_val_loss /= (len(xval) // 128 + 1)

    print("%d: Train: %.8f \t Val: %.8f" % (epoch + 1, running_loss, running_val_loss))
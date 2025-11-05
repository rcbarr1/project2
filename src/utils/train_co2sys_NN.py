"""
Created on Tue Nov 04 2025

TRAIN_CO2SYS_NN.py
build neural network to replace iterative solve for AT to add to reach preindustrial

@author: Reese C. Barrett
"""
#%%
import numpy as np
import xarray as xr
from src.utils import project2 as p2
import PyCO2SYS as pyco2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

output_path = '/Volumes/LaCie/outputs/'
data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/'
model_path = '/Users/Reese_1/Documents/Research Projects/project2/src/utils/' 

# import necessary model parameters
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].to_numpy()
model_depth = model_data['tz'].to_numpy()[:, 0, 0] # m below sea surface
model_lon = model_data['tlon'].to_numpy()[0, :, 0] # ºE
model_lat = model_data['tlat'].to_numpy()[0, 0, :] # ºN
ns = int(np.nansum(ocnmask[0,:,:])) # number of surface grid cells

mld = model_data.mld.values # [m]
grid_cell_depth = model_data['wz'].to_numpy() # depth of model layers (need bottom of grid cell, not middle) [m]
mldmask = (grid_cell_depth < mld[None, :, :]).astype(int) * ocnmask

#%% generate training data
# AT, DIC, T, S, pressure, Si, P at each grid cell are input variables
# output variables are amount of AT to add to that grid cell to reach preindustrial pH

# pull in data from a previous simulation
experiment_name = 'exp21_2025-10-29_20-08-08_t0_none_0.0'

ds = xr.open_mfdataset(
        output_path + experiment_name + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)

# import data from glodap
DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy') # dissolved inorganic carbon [µmol kg-1]
AT_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy')   # total alkalinity [µmol kg-1]
T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy')
S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy')
Si_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/silicate.npy') # silicate [µmol kg-1]
P_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/PO4.npy') # phosphate [µmol kg-1]

DIC = p2.flatten(DIC_3D, ocnmask)
AT = p2.flatten(AT_3D, ocnmask)
T = p2.flatten(T_3D, ocnmask)
S = p2.flatten(S_3D, ocnmask)
Si = p2.flatten(Si_3D, ocnmask)
P = p2.flatten(P_3D, ocnmask)

DIC_mld = p2.flatten(DIC_3D, mldmask)
AT_mld = p2.flatten(AT_3D, mldmask)
T_mld = p2.flatten(T_3D, mldmask)
S_mld = p2.flatten(S_3D, mldmask)
Si_mld = p2.flatten(Si_3D, mldmask)
P_mld = p2.flatten(P_3D, mldmask)

# create "pressure" array by broadcasting depth array
pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))
pressure = pressure_3D[ocnmask == 1].flatten(order='F')
pressure_mld = pressure_3D[mldmask == 1].flatten(order='F')

# calculate preindustrial pH for training data

# load in TRACE data
Canth_2015 = p2.loadmat(data_path + 'TRACEv1/trace_outputs_2015.mat')
Canth_2015 = Canth_2015['trace_outputs_2015']
Canth_2015 = Canth_2015.reshape(len(model_lon), len(model_lat), len(model_depth), order='F')
Canth_2015 = Canth_2015.transpose([2, 0, 1])

# calculate preindustrial pH by subtracting anthropogenic carbon
DIC_preind_3D = DIC_3D - Canth_2015
DIC_preind = p2.flatten(DIC_preind_3D, ocnmask)

# calculate preindustrial pH from DIC in 2015 minus Canth in 2015 AND TA in 2015 (assuming steady state)
# pyCO2SYS v2
co2sys_preind = pyco2.sys(dic=DIC_preind, alkalinity=AT, salinity=S, temperature=T,
                pressure=pressure, total_silicate=Si, total_phosphate=P)

pH_preind = co2sys_preind['pH']
pH_preind_3D = p2.make_3D(pH_preind, ocnmask)
pH_preind_mld = p2.flatten(pH_preind_3D, mldmask)

#%% format training data for input in NN

# need to get flatten AT and DIC from time step "t", match with T, S, Si, P, and pressure data (constant), and vertically stack
nt = 25 # number of time steps to include in training data
mldmask_flat = (mldmask == 1).flatten(order='F')

AT_flat = ds.delAT.isel(time=slice(0,nt)).values.reshape(nt,-1,order='F')
AT_flat = AT_flat[:, mldmask_flat] + AT_mld
AT_stacked = AT_flat.reshape(-1,1)

DIC_flat = ds.delDIC.isel(time=slice(0,nt)).values.reshape(nt,-1,order='F')
DIC_flat = DIC_flat[:, mldmask_flat] + DIC_mld
DIC_stacked = DIC_flat.reshape(-1,1)

T_stacked = np.expand_dims(np.tile(T_mld, nt),1)
S_stacked = np.expand_dims(np.tile(S_mld, nt),1)
Si_stacked = np.expand_dims(np.tile(Si_mld, nt),1)
P_stacked = np.expand_dims(np.tile(P_mld, nt),1)
pressure_stacked = np.expand_dims(np.tile(pressure_mld, nt),1)
pH_preind_stacked = np.expand_dims(np.tile(pH_preind_mld, nt),1)

X = np.hstack([AT_stacked, DIC_stacked, T_stacked, S_stacked, Si_stacked, P_stacked, pressure_stacked, pH_preind_stacked]) 

# this corresponds to the amount of AT added at time step "t+1"
AT_added_flat = ds.AT_added.isel(time=slice(1,nt+1)).values.reshape(nt,-1,order='F')
AT_added_flat = AT_added_flat[:, mldmask_flat]
AT_added_stacked = AT_added_flat.reshape(-1, 1)

y = AT_added_stacked
# %% preprocess data and split into training and testing

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=0)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
train_dataset = TensorDataset(X_train, y_train) 
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True) # batch to handle large amounts of data

# %% set up neural network model

class AT_added_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,128), # layer 1 shape
            nn.ReLU(),
            nn.Linear(128, 64), # layer 2 shape
            nn.ReLU(),
            nn.Linear(64, 32), # layer 3 shape
            nn.ReLU(),
            nn.Linear(32, 1) # layer 3 shape
        )

    def forward(self, x):
        return self.net(x)

model = AT_added_model()
criterion = nn.MSELoss() # mean squared error to evaluate model fit
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %% train meural network model (skip if uploading already-trained model)

for epoch in tqdm(range(500)):
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad() # reset gradients
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"epoch {epoch}: loss = {avg_loss:.4f}")

print('model training finished')

#%% save model for later use
# save model weights
torch.save(model.state_dict(), model_path + 'AT_added_model.pth')

# save scalers
joblib.dump(scaler_X, model_path + 'AT_added_model_scaler_X.save')
joblib.dump(scaler_y, model_path + 'AT_added_model_scaler_y.save')

#%% upload model for testing

# recreate model architecture
model = AT_added_model()
model.load_state_dict(torch.load(model_path + 'AT_added_model.pth'))
model.eval() 

# load scalers
scaler_X = joblib.load(model_path + 'AT_added_model_scaler_X.save')
scaler_y = joblib.load(model_path + 'AT_added_model_scaler_y.save')

# %% test model
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

model.eval()
all_preds = []
all_targets = []
val_loss = 0.0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        all_preds.append(y_pred)
        all_targets.append(y_batch)

        loss = criterion(y_pred, y_batch)
        val_loss += loss.item()

avg_val_loss = val_loss / len(test_loader)
print(f"validation Loss: {avg_val_loss:.4f}")

# concatenate all batches into full tensors
y_pred_full = torch.cat(all_preds, dim=0)
y_true_full = torch.cat(all_targets, dim=0)
y_pred = y_pred_full.numpy()
y_true = y_true_full.numpy()

# %% make figure comparing predicted and true
fig = plt.figure(figsize=(6,5), dpi=200)
ax = fig.gca()

y_pred_unscaled = np.squeeze(scaler_y.inverse_transform(y_pred))
y_true_unscaled = np.squeeze(scaler_y.inverse_transform(y_true))

ax.plot(np.arange(0,100), np.arange(0,100), ls=':', c='gray')

h = ax.hist2d(y_pred_unscaled,y_true_unscaled,bins=100,cmap='magma_r')
ax.set_xlabel('predicted AT to add (µmol kg-1)')
ax.set_ylabel('actual AT to add (µmol kg-1)')
ax.set_xlim([0,95])
ax.set_ylim([0,95])
c = fig.colorbar(h[3], ax=ax)
c.ax.set_ylabel('frequency')

# %% make figure comparing predicted and true (with predictor variable)
fig = plt.figure(figsize=(6,5), dpi=200)
ax = fig.gca()

y_pred_unscaled = np.squeeze(scaler_y.inverse_transform(y_pred))
y_true_unscaled = np.squeeze(scaler_y.inverse_transform(y_true))
X_test_unscaled = scaler_X.inverse_transform(X_test)

ax.hlines(0,0,3000,colors='gray',ls=':')

h = ax.hist2d(X_test_unscaled[:,7], y_pred_unscaled-y_true_unscaled,bins=100,cmap='magma_r') # pressure vs. difference (trend in depth?)
ax.set_xlabel('preindustrial pH')
ax.set_ylabel('predicted - actual AT to add (µmol kg-1)')
#ax.set_xlim([0,95])
#ax.set_ylim([0,95])
c = fig.colorbar(h[3], ax=ax)
c.ax.set_ylabel('frequency')

# %% make figure comparing predicted and true (by estimate)
fig = plt.figure(figsize=(6,5), dpi=200)
ax = fig.gca()

y_pred_unscaled = np.squeeze(scaler_y.inverse_transform(y_pred))
y_true_unscaled = np.squeeze(scaler_y.inverse_transform(y_true))
X_test_unscaled = scaler_X.inverse_transform(X_test)

ax.hlines(0,0,100,colors='gray',ls=':')

h = ax.hist2d(y_true_unscaled, y_pred_unscaled-y_true_unscaled, bins=100,cmap='magma_r') 
ax.set_xlabel('actual AT to add (µmol kg-1)')
ax.set_ylabel('predicted - actual AT to add (µmol kg-1)')
#ax.set_xlim([0,95])
#ax.set_ylim([0,95])
c = fig.colorbar(h[3], ax=ax)
c.ax.set_ylabel('frequency')

print('RMSE_normalized: ' + str(root_mean_squared_error(y_pred_unscaled, y_true_unscaled)/np.mean(y_pred_unscaled)))
# %%

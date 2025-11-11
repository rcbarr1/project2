import numpy as np
"""
Created on Mon Nov 10 2025

CO2SYS_INTERP.py
solve for AT to add with polynomial regression method 

@author: Reese C. Barrett
"""
#%%
import numpy as np
from src.utils import project2 as p2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
#%% create training data to sample
n_samples = 5000
rng = np.random.default_rng(0)

pH_preind_samp = np.random.uniform(7.43, 8.67, n_samples)
DIC_samp = np.random.uniform(1377, 2324, n_samples)
AT_samp = np.random.uniform(1444, 2641, n_samples)
T_samp = np.random.uniform(-2.2, 34, n_samples)
S_samp = np.random.uniform(15, 39.7, n_samples)
pressure_samp = np.random.uniform(0, 250, n_samples)
Si_samp = np.random.uniform(0, 92.3, n_samples)
P_samp = np.random.uniform(0, 2.4, n_samples)

X = np.column_stack([pH_preind_samp, DIC_samp, AT_samp, T_samp, S_samp, pressure_samp, Si_samp, P_samp])

#%% calculate AT to add at each combination of points
y = p2.calculate_AT_to_add(pH_preind_samp, DIC_samp, AT_samp, T_samp, S_samp, pressure_samp, Si_samp, P_samp, AT_mask=None, low=0, high=1000, tol=1e-6, maxiter=50)

#%% split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% make and compute regression model
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=3),
    Ridge(alpha=1e-3)
)

model.fit(X_train,y_train)

# %% test model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

rmse_train = root_mean_squared_error(y_train, y_pred_train)
rmse_test  = root_mean_squared_error(y_test, y_pred_test)
r2_train   = r2_score(y_train, y_pred_train)
r2_test    = r2_score(y_test, y_pred_test)

print(f"Train RMSE = {rmse_train:.4f}, R² = {r2_train:.4f}")
print(f"Test  RMSE = {rmse_test:.4f}, R² = {r2_test:.4f}")

# %% make figure comparing predicted and true
fig = plt.figure(figsize=(6,5), dpi=200)
ax = fig.gca()

ax.plot(np.arange(0,100), np.arange(0,100), ls=':', c='gray')

h = ax.hist2d(y_pred_test,y_test,bins=100,cmap='magma_r')
ax.set_xlabel('predicted AT to add (µmol kg-1)')
ax.set_ylabel('actual AT to add (µmol kg-1)')
ax.set_xlim([0,95])
ax.set_ylim([0,95])
c = fig.colorbar(h[3], ax=ax)
c.ax.set_ylabel('frequency')

# %% make figure comparing predicted and true (with predictor variable)
fig = plt.figure(figsize=(6,5), dpi=200)
ax = fig.gca()

ax.hlines(0,0,3000,colors='gray',ls=':')

h = ax.hist2d(X_test[:,1], y_pred_test-y_test,bins=100,cmap='magma_r') # pressure vs. difference (trend in depth?)
ax.set_xlabel('preindustrial pH')
ax.set_ylabel('predicted - actual AT to add (µmol kg-1)')
#ax.set_xlim([0,95])
#ax.set_ylim([0,95])
c = fig.colorbar(h[3], ax=ax)
c.ax.set_ylabel('frequency')

# %% make figure comparing predicted and true (by estimate)
fig = plt.figure(figsize=(6,5), dpi=200)
ax = fig.gca()

ax.hlines(0,0,100,colors='gray',ls=':')

h = ax.hist2d(y_test, y_pred_test-y_test, bins=100,cmap='magma_r') 
ax.set_xlabel('actual AT to add (µmol kg-1)')
ax.set_ylabel('predicted - actual AT to add (µmol kg-1)')
#ax.set_xlim([0,95])
#ax.set_ylim([0,95])
c = fig.colorbar(h[3], ax=ax)
c.ax.set_ylabel('frequency')

print('RMSE_normalized: ' + str(root_mean_squared_error(y_pred_unscaled, y_true_unscaled)/np.mean(y_pred_unscaled)))
# %%
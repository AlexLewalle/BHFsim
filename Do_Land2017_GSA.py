#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:49:58 2021

Modified from AL_20211123.py

@author: al12local
"""


import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
import torch
from GPErks.gp.data.dataset import Dataset
from GPErks.perks.gsa import SobolGSA

import Land2017_sim as M
import seaborn as sns

# set logger and enforce reproducibility
from GPErks.log.logger import get_logger
from GPErks.utils.random import set_seed
log = get_logger()
seed = 8
set_seed(seed)  # reproducible sampling

plt.ion()

Features = {}
WhichParams = 'AllParams'  # 'eta_l', 'eta_s', 'k_trpn', 'ku', 'kuw', 'kws') #
WhichFeatures = ['Fss', 'dFss', 'rFpeak', 'drFpeak', 'tdecay', 'dtdecay',   # <-- from quick stretch
                 'Fmax', 'nH', 'EC50',                                      # <-- from F-pCa at Lambda=1.0
                 'dFmaxdLambda', 'dEC50dLambda',                            # <-- from F-pCa at Lambda=1.1
                 'Fss_pas', 'dFss_pas', 'tdec_pas0', 'tdec_pas1',           # <-- from passive quick stretch
                 'dFCaiStep', 'tCaiStep']                                   # <-- from Cai step


#%%  Define reference model and variants

Model0 = M.Land2017()

print('Doing LH sampling')
PSet = M.MakeParamSetLH(Model0, 100,  WhichParams)
X = np.array([list(dict1.values()) for dict1 in PSet])
print('LH sampling completed')

#%% Do experiments


# for Feat1 in WhichFeatures:    Features[Feat1] = [None]*len(PSet)


# # %%% FpCa
# print('Doing FpCa')
# Features_FpCa = M.DoFpCa(PSet, Lambda0=1.0, ifPlot=True)
# Features = {**Features, **Features_FpCa}       # update Features dictionary with FpCa entries

# # %%% FpCa with different SL

# dLambda = 0.1
# print(f'/nDoing FpCa with a different SL={1+dLambda}')
# Features_1 = M.DoFpCa(PSet, Lambda0=1+dLambda, ifPlot=True)
# Features_FpCadiffSL = {'dFmaxdLambda': list((np.array(Features_1['Fmax'])-np.array(Features['Fmax']))/dLambda),
#                   'dEC50dLambda': list((np.array(Features_1['EC50'])-np.array(Features['EC50']))/dLambda)}
# Features = {**Features, **Features_FpCadiffSL}       # update Features dictionary with FpCa entries


# # import sys; sys.exit()

# # %%% Active quick strethc
# print('Doing QuickStretches')
# Features_QuickStretches = M.DoQuickStretches(PSet, ifPlot=True)
# Features = {**Features, **Features_QuickStretches}        # update Features dictionary with QuickStretch entries

# # %%% Passive quick stretch
# print('Doing passive QuickStretches')
# Features_QuickStretchesPassive = M.DoQuickStretches_passive(PSet)
# Features = {**Features, **Features_QuickStretchesPassive}        # update Features dictionary with QuickStretch entries


# # %%% Cai step
# print('Doing Cai steps')
# Features_CaiStep = M.DoCaiStep(PSet, Cai1=10**-7, Cai2=10**-4, L0=1.9, ifPlot=True)
# Features = {**Features, **Features_CaiStep}        # update Features dictionary with QuickStretch entries

# # %%% Save Features
# import pickle
# with open('Features_results.dat', 'wb') as file_features:
#     pickle.dump([Features_FpCa, Features_FpCadiffSL, Features_QuickStretches, Features_QuickStretchesPassive, Features_CaiStep], file_features)




#%% Create emulators

import pickle
with open('Features_results.dat', 'rb') as file_features:
    [Features_FpCa, Features_FpCadiffSL, Features_QuickStretches, Features_QuickStretchesPassive, Features_CaiStep] = pickle.load(file_features)
Features = {**Features_FpCa, **Features_FpCadiffSL, **Features_QuickStretches, **Features_QuickStretchesPassive, **Features_CaiStep}


# define experiment
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from torchmetrics import MeanSquaredError, R2Score
from GPErks.gp.experiment import GPExperiment
from GPErks.train.emulator import GPEmulator
# from GPErks.train.early_stop import GLEarlyStoppingCriterion


plt.style.use('seaborn')
numrows = floor(np.sqrt(len(WhichFeatures)))
numcols = ceil(len(WhichFeatures)/numrows)
figS1, axS1 = plt.subplots(numrows, numcols, figsize=([14,  9]), num='GSA')
figSummary, axSummary = plt.subplots(num='Summary - Cumulative sensitivities')
ParamSummary = np.array([0]*len(Model0.AllParams))

Emulators = {}
for iFeat, Feat1 in enumerate(WhichFeatures):
    print(f'Doing feature {Feat1}')
    dataset = Dataset(X_train=X,
                      y_train= Features[Feat1],
                      l_bounds=[Model0.ParRange[param1][0] for param1 in Model0.AllParams],
                      u_bounds=[Model0.ParRange[param1][1] for param1 in Model0.AllParams])

    likelihood = GaussianLikelihood()
    mean_function = LinearMean(input_size=dataset.input_size)
    kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))
    metrics = [MeanSquaredError(), R2Score()]
    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        seed=seed,  # reproducible training
        learn_noise=True
    )
    device = "cpu"

    Emulators[Feat1] = GPEmulator(experiment, device)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    # esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)

    print(f'Training emulator for {Feat1}')
    best_model, best_train_stats = Emulators[Feat1].train(optimizer) ;   print(f'   Emulator training completed for {Feat1}')

# Save Emulators     PICKLING ERROR ??????
# with open('Emulators_results.dat', 'wb') as file_emulators:
#     pickle.dump(Emulators, file_emulators)



#%% Do GSA

gsa = {}
for iFeat, Feat1 in enumerate(WhichFeatures):
    # Perform GSA on emulator
    print(f'Starting GSA for {Feat1}')
    dataset = Dataset(X_train=X,
                      y_train= Features[Feat1],
                      l_bounds=[Model0.ParRange[param1][0] for param1 in Model0.AllParams],
                      u_bounds=[Model0.ParRange[param1][1] for param1 in Model0.AllParams])
    gsa[Feat1] = SobolGSA(dataset, n=128, seed=seed)
    gsa[Feat1].estimate_Sobol_indices_with_emulator(Emulators[Feat1], n_draws=100); print(f'   GSA complete for {Feat1}')
    gsa[Feat1].summary()

    ParamSummary = ParamSummary + np.mean(gsa[Feat1].S1, axis=0)

    gsa[Feat1].ylabels = list(Model0.AllParams)
#    M.PlotS1(gsa, Feat1)
    ax1 = axS1.ravel()
    sns.boxplot(ax=ax1[iFeat], data=gsa[Feat1].S1)
    ax1[iFeat].set_title(Feat1)
    ax1[iFeat].set_xticklabels(gsa[Feat1].ylabels, rotation=90)

figS1.tight_layout()
axSummary.barh(Model0.AllParams, ParamSummary, align='center')



#%% Do PCA

fig_pca, ax_pca = plt.subplots(num='PCA')

from sklearn.decomposition import PCA
pca = PCA()
Y = np.array([Features[feat1] for feat1 in WhichFeatures])
Y1 = np.array([y1/np.std(y1) for y1 in Y])
pca.fit(Y1.T)
ax_pca.bar(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.show()
ax_pca.set_ylabel('explained variance ratio')
ax_pca.set_xlabel('principal components')
fig_pca.suptitle('Land2017')
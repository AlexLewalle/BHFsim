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

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from torchmetrics import MeanSquaredError, R2Score
from GPErks.gp.experiment import GPExperiment
from GPErks.train.emulator import GPEmulator
# from GPErks.train.early_stop import GLEarlyStoppingCriterion

import pickle


# FeatureData = {}
 

AllFeatures_QuickStretches = ['Fss', 'maxDF', 'dFss', 'rFpeak', 'drFpeak', 'tdecay', 'dtdecay'] # <-- from quick stretch
AllFeatures_FpCa = ['Fmax', 'nH', 'EC50']  # <-- from F-pCa at Lambda=1.0
AllFeatures_FpCadiffLambda = ['dFmaxdLambda', 'dEC50dLambda']  # <-- from F-pCa at Lambda=1.1
AllFeatures_PassiveQuickStretches = ['Fss_pas', 'dFss_pas', 'tdec_pas0', 'tdec_pas1']  # <-- from passive quick stretch
AllFeatures_CaiStep = ['dFCaiStep', 'tCaiStep']  # <-- from Cai step
AllFeatures = AllFeatures_QuickStretches + AllFeatures_FpCa + AllFeatures_FpCadiffLambda + AllFeatures_PassiveQuickStretches + AllFeatures_CaiStep

WhichFeatures = AllFeatures

#%%  Define reference model and variants

Model0 = M.Land2017()

# NumSamples = 100
# print(f'Doing LH sampling --- {NumSamples} samples')
# PSet = M.MakeParamSetLH(Model0, NumSamples,  WhichParams)
# X = np.array([list(dict1.values()) for dict1 in PSet])
# print('LH sampling completed')

#%% Do experiments


# for Feat1 in WhichFeatures:    Features[Feat1] = [None]*len(PSet)


# # %%% FpCa

# if set(WhichFeatures) & ( set(AllFeatures_FpCa) | set(AllFeatures_FpCadiffLambda) ):
#     print('Doing FpCa')
#     Features_FpCa = M.DoFpCa(PSet, Lambda0=1.0, ifPlot=True)
#     Features = {**Features, **Features_FpCa}       # update Features dictionary with FpCa entries

# # %%% FpCa with different SL

# if set(WhichFeatures) & set(AllFeatures_FpCadiffLambda) :
#     dLambda = 0.1
#     print(f'/nDoing FpCa with a different SL={1+dLambda}')
#     Features_1 = M.DoFpCa(PSet, Lambda0=1+dLambda, ifPlot=True)
#     Features_FpCadiffSL = {'dFmaxdLambda': list((np.array(Features_1['Fmax'])-np.array(Features['Fmax']))/dLambda),
#                       'dEC50dLambda': list((np.array(Features_1['EC50'])-np.array(Features['EC50']))/dLambda)}
#     Features = {**Features, **Features_FpCadiffSL}       # update Features dictionary with FpCa entries


# # %%% Active quick stretch

# if set(WhichFeatures) & set(AllFeatures_QuickStretches) :
#     print('Doing QuickStretches')
#     Features_QuickStretches = M.DoQuickStretches(PSet, ifPlot=True)
#     Features = {**Features, **Features_QuickStretches}        # update Features dictionary with QuickStretch entries

# # %%% Passive quick stretch

# if set(WhichFeatures) & set(AllFeatures_PassiveQuickStretches) :
#     print('Doing passive QuickStretches')
#     Features_QuickStretchesPassive = M.DoQuickStretches_passive(PSet)
#     Features = {**Features, **Features_QuickStretchesPassive}        # update Features dictionary with QuickStretch entries


# # %%% Cai step

# if set(WhichFeatures) & set(AllFeatures_CaiStep) :
#     print('Doing Cai steps')
#     Features_CaiStep = M.DoCaiStep(PSet, Cai1=10**-7, Cai2=10**-4, L0=1.9, ifPlot=True)
#     Features = {**Features, **Features_CaiStep}        # update Features dictionary with QuickStretch entries

# # %%% Save Features

# if WhichFeatures == AllFeatures:
#     import pickle
#     with open('Features_results.dat', 'wb') as file_features:
#         pickle.dump([Features_FpCa, Features_FpCadiffSL, Features_QuickStretches, Features_QuickStretchesPassive, Features_CaiStep], file_features)




#%% Create emulators

# if WhichFeatures == AllFeatures:
#     import pickle
#     with open('Features_results.dat', 'rb') as file_features:
#         [Features_FpCa, Features_FpCadiffSL, Features_QuickStretches, Features_QuickStretchesPassive, Features_CaiStep] = pickle.load(file_features)
#     Features = {**Features_FpCa, **Features_FpCadiffSL, **Features_QuickStretches, **Features_QuickStretchesPassive, **Features_CaiStep}





plt.style.use('seaborn')
# numrows = floor(np.sqrt(len(WhichFeatures)))
# numcols = ceil(len(WhichFeatures)/numrows)
# figS1, axS1 = plt.subplots(numrows, numcols, figsize=([14,  9]), num='GSA') ; 
# if len(WhichFeatures)==1: axS1 = np.array([axS1])
# figSummary, axSummary = plt.subplots(num='Summary - Cumulative sensitivities')
# ParamSummary = np.array([0]*len(Model0.AllParams))

# Emulators = {}
# for iFeat, Feat1 in enumerate(WhichFeatures):
#     print(f'Doing feature {Feat1}')
#     dataset = Dataset(X_train=X,
#                       y_train= Features[Feat1],
#                       l_bounds=[Model0.ParRange[param1][0] for param1 in Model0.AllParams],
#                       u_bounds=[Model0.ParRange[param1][1] for param1 in Model0.AllParams])

#     likelihood = GaussianLikelihood()
#     mean_function = LinearMean(input_size=dataset.input_size)
#     kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))
#     metrics = [MeanSquaredError(), R2Score()]
#     experiment = GPExperiment(
#         dataset,
#         likelihood,
#         mean_function,
#         kernel,
#         n_restarts=3,
#         metrics=metrics,
#         seed=seed,  # reproducible training
#         learn_noise=True
#     )
#     device = "cpu"

#     Emulators[Feat1] = GPEmulator(experiment, device)

#     optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
#     # esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)

#     print(f'Training emulator for {Feat1}')
#     best_model, best_train_stats = Emulators[Feat1].train(optimizer) ;   print(f'   Emulator training completed for {Feat1}')



#%% Do GSA

# gsa = {}
# for iFeat, Feat1 in enumerate(WhichFeatures):
#     # Perform GSA on emulator
#     print(f'Starting GSA for {Feat1}')
#     dataset = Dataset(X_train=X,
#                       y_train= Features[Feat1],
#                       l_bounds=[Model0.ParRange[param1][0] for param1 in Model0.AllParams],
#                       u_bounds=[Model0.ParRange[param1][1] for param1 in Model0.AllParams])
#     gsa[Feat1] = SobolGSA(dataset, n=128, seed=seed)
#     gsa[Feat1].estimate_Sobol_indices_with_emulator(Emulators[Feat1], n_draws=100); print(f'   GSA complete for {Feat1}')
#     gsa[Feat1].summary()

#     ParamSummary = ParamSummary + np.mean(gsa[Feat1].S1, axis=0)

#     gsa[Feat1].ylabels = list(Model0.AllParams)
# #    M.PlotS1(gsa, Feat1)
#     ax1 = axS1.ravel()
#     sns.boxplot(ax=ax1[iFeat], data=gsa[Feat1].S1)
#     ax1[iFeat].set_title(Feat1)
#     ax1[iFeat].set_xticklabels(gsa[Feat1].ylabels, rotation=90)

# figS1.tight_layout()
# axSummary.barh(Model0.AllParams, ParamSummary, align='center')



#%% Do PCA

# fig_pca, ax_pca = plt.subplots(num='PCA')

# from sklearn.decomposition import PCA
# pca = PCA()
# Y = np.array([Features[feat1] for feat1 in WhichFeatures])
# Y1 = np.array([y1/np.std(y1) for y1 in Y])
# pca.fit(Y1.T)
# ax_pca.bar(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
# plt.show()
# ax_pca.set_ylabel('explained variance ratio')
# ax_pca.set_xlabel('principal components')
# fig_pca.suptitle('Land2017')

#%% FUNCTIONS

def InitModels(NumSamples=100, *args):
    if len(args)==0 | ('AllParams' in args):
        WhichParams = 'AllParams'
    else:
        WhichParams = args
    print(f'Doing LH sampling --- {NumSamples} samples')
    PSet = M.MakeParamSetLH(Model0, NumSamples,  *WhichParams)
    X = np.array([list(dict1.values()) for dict1 in PSet])
    print('LH sampling completed')
    return PSet, X

def DoAllExperiments(PSet, ifSave=True):
    FeatureData = {}
    for Feat1 in WhichFeatures:    FeatureData[Feat1] = [None]*len(PSet)

    print('Doing FpCa')
    Features_FpCa = M.DoFpCa(PSet, Lambda0=1.0, ifPlot=True, ifSave=ifSave)
    FeatureData = {**FeatureData, **Features_FpCa}       # update Features dictionary with FpCa entries

    dLambda = 0.1
    print(f'/nDoing FpCa with a different SL={1+dLambda}')
    Features_1 = M.DoFpCa(PSet, Lambda0=1+dLambda, ifPlot=True, ifSave=ifSave)
    Features_FpCadiffSL = {'dFmaxdLambda': list((np.array(Features_1['Fmax'])-np.array(FeatureData['Fmax']))/dLambda),
                      'dEC50dLambda': list((np.array(Features_1['EC50'])-np.array(FeatureData['EC50']))/dLambda)}
    FeatureData = {**FeatureData, **Features_FpCadiffSL}       # update Features dictionary with FpCa entries

    print('Doing QuickStretches')
    Features_QuickStretches = M.DoQuickStretches(PSet, ifPlot=True, ifSave=ifSave)
    FeatureData = {**FeatureData, **Features_QuickStretches}        # update Features dictionary with QuickStretch entries

    print('Doing passive QuickStretches')
    Features_QuickStretchesPassive = M.DoQuickStretches_passive(PSet, ifSave=ifSave)
    FeatureData = {**FeatureData, **Features_QuickStretchesPassive}        # update Features dictionary with QuickStretch entries

    print('Doing Cai steps')
    Features_CaiStep = M.DoCaiStep(PSet, Cai1=10**-7, Cai2=10**-4, L0=1.9, ifPlot=True, ifSave=ifSave)
    FeatureData = {**FeatureData, **Features_CaiStep}        # update Features dictionary with QuickStretch entries

    if ifSave:
        import pickle
        with open('Features_results.dat', 'wb') as file_features:
            pickle.dump([Features_FpCa, Features_FpCadiffSL, Features_QuickStretches, Features_QuickStretchesPassive, Features_CaiStep], file_features)

    return FeatureData



def CreateEmulator(WhichFeature, X, FeatureData, ifLoad=True):
    # define experiment
    print(f'Emulating   {WhichFeature}')
    dataset = Dataset(X_train=X,
                      y_train= FeatureData[WhichFeature],
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
        learn_noise=False   # True  #   
    )
    device = "cpu"

    emulator = GPEmulator(experiment, device)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    # esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)

    print(f'Training emulator for {WhichFeature}')
    best_model, best_train_stats = emulator.train(optimizer) ;   print(f'   Emulator training completed for {WhichFeature}')

    return emulator


def DoAllEmulators(WhichFeatures, PSet, FeatureData):
    Emulators = {}
    for iFeat, Feat1 in enumerate(WhichFeatures):
        Emulators[Feat1] = CreateEmulator(Feat1, X, FeatureData)


# def DoGSA(WhichFeature, X, FeatureData, emulator = None, ifPlot=True, axgsa1=None):
#     print(f'Starting GSA for {WhichFeature}')
#     if emulator == None:
#         emulator = CreateEmulator(WhichFeature, X, FeatureData)
#     dataset = Dataset(X_train=X,
#                       y_train= FeatureData,
#                       l_bounds = [Model0.ParRange[param1][0] for param1 in Model0.AllParams],
#                       u_bounds = [Model0.ParRange[param1][1] for param1 in Model0.AllParams])
#     gsa1 = SobolGSA(dataset, n=128, seed=seed)
#     gsa1.estimate_Sobol_indices_with_emulator(emulator, n_draws=100); print(f'   GSA complete for {WhichFeature}')
#     gsa1.summary()
    
#     if ifPlot == True:
#         if axgsa1 == None:
#             figgsa1, axgsa1 = plt.subplots()        
#         sns.boxplot(ax=axgsa1, data=gsa1.S1)
#         axgsa1.set_title(WhichFeature)
#         axgsa1.set_xticklabels(Model0.AllParams, rotation=90)       
#     return gsa1

def DoPCA(FeatureData):
    fig_pca, ax_pca = plt.subplots(num='PCA')
    
    from sklearn.decomposition import PCA
    pca = PCA()
    Y = np.array([FeatureData[feat1] for feat1 in WhichFeatures])
    Y1 = np.array([y1/np.std(y1) for y1 in Y])
    pca.fit(Y1.T)
    ax_pca.bar(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.show()
    ax_pca.set_ylabel('explained variance ratio')
    ax_pca.set_xlabel('principal components')
    fig_pca.suptitle('Land2017')

    
        
def DoGSA(WhichFeatures, X, FeatureData, Emulators=None, ifPlot=True):
    GSA = {}
    if Emulators == None:
        Emulators = {}
        for Feat1 in WhichFeatures:
            Emulators = {**Emulators, **{Feat1: None} }
    numrows = floor(np.sqrt(len(WhichFeatures)))
    numcols = ceil(len(WhichFeatures)/numrows)
    figS1, axS1 = plt.subplots(numrows, numcols, figsize=([14,  9]), num='GSA') ; 
    if len(WhichFeatures)==1: axS1 = np.array([axS1])
    axS1 = np.ravel(axS1)
    figSummary, axSummary = plt.subplots(num='Summary - Cumulative sensitivities')
    ParamSummary = np.array([0]*len(Model0.AllParams))
    
    for iFeat, Feat1 in enumerate(WhichFeatures):
        print(f'Starting GSA for {Feat1}')
        if Emulators[Feat1] == None:
            Emulators[Feat1] = CreateEmulator(Feat1, X, FeatureData)
        dataset = Dataset(X_train=X,
                          y_train= FeatureData,
                          l_bounds = [Model0.ParRange[param1][0] for param1 in Model0.AllParams],
                          u_bounds = [Model0.ParRange[param1][1] for param1 in Model0.AllParams])
        gsa1 = SobolGSA(dataset, n=128, seed=seed)
        gsa1.estimate_Sobol_indices_with_emulator(Emulators[Feat1], n_draws=100); print(f'   GSA complete for {Feat1}')
        gsa1.summary()
        
        if ifPlot == True:
            sns.boxplot(ax=axS1[iFeat], data=gsa1.S1)
            axS1[iFeat].set_title(Feat1)
            axS1[iFeat].set_xticklabels(Model0.AllParams, rotation=90)       

        GSA[Feat1] = gsa1
        ParamSummary = ParamSummary + np.mean(gsa1.S1, axis=0)
        
 
    figS1.tight_layout()
    axSummary.barh(Model0.AllParams, ParamSummary, align='center')
 
    
#%% MAIN
 
if __name__ == '__main__':
    
    WhichParams = 'AllParams' 
    PSet, X = InitModels(100, WhichParams)
    
    
    # FeatureData = DoAllExperiments(PSet, ifSave=True)
    with open('Features_results.dat', 'rb') as file_features:
        [Features_FpCa, Features_FpCadiffSL, Features_QuickStretches, Features_QuickStretchesPassive, Features_CaiStep] = pickle.load(file_features)
    FeatureData = {**Features_FpCa, **Features_FpCadiffSL, **Features_QuickStretches, **Features_QuickStretchesPassive, **Features_CaiStep}
    
    # Emulators = DoAllEmulators(WhichFeatures, PSet, FeatureData)
    GSA = DoGSA(WhichFeatures, X, FeatureData)
    
    
    # DoGSA(['Fmax'], X, FeatureData)
    
    # with open('Features_FpCa.dat', 'rb') as file_features:
    #     [X, Features_FpCa] = pickle.load(file_features)
    
    
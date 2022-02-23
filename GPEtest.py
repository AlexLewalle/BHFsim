#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:57:17 2022

@author: al12
"""

import Do_Land2017_GSA as G
import Land2017_sim as M
import pickle

Model0 = M.Land2017()

# PSet, X = G.InitModels(100, 'AllParams')

# Features_FpCa = M.DoFpCa(PSet, ifPlot=True, ifSave=True)
with open('Features_FpCa.dat', 'rb') as file_features:
    [PSet, Features_FpCa] = pickle.load(file_features)

X = M.PSet2X(PSet)

emulator = G.CreateEmulator('Fmax', PSet, Features_FpCa)

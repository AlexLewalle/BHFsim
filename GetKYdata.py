#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:29:19 2022

@author: al12local
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

from collections import defaultdict

pCa_list = [4.8, 5.4, 5.5, 5.6, 5.8, 5.9, 6.0, 6.2, 6.4, 6.6, 9.0]
dFL_list = [ 1., 0., -1., 2., -2., 0.5, -0.5] #, 1.]
SL_list = [2.0, 2.3]
DataPath = '/home/al12local/Desktop/Pilot_stretches'

td = {}
for SL1 in SL_list:
    td[SL1] = {}
    for run1 in ['a', 'b', 'c']:
        td[SL1][run1]={}
        for pCa1 in pCa_list:
            td[SL1][run1][pCa1]={}
            for jdF, dFL1 in enumerate(dFL_list):
                td[SL1][run1][pCa1][dFL1]={}
                DataFile = f'{DataPath}/SL{SL1:.1f}/{run1}/KYStretch_pCa{pCa1:.1f}_step{dFL1:+.1f}' 
                if jdF==7:
                    DataFile += 'repeat'
                DataFile += '.mat'
                D = spio.loadmat(DataFile)['td']
                
                Values = [D[0][0][j] for j in range(len(D[0][0]))]
                Names = [D[0][0].dtype.descr[j][0]  for j in range(len(D[0][0]))]

                
                for j, Names1 in enumerate(Names):
                    td[SL1][run1][pCa1][dFL1][Names1] = np.ravel(Values[j])
                    if len(td[SL1][run1][pCa1][dFL1][Names1]) == 1:
                        td[SL1][run1][pCa1][dFL1][Names1] = td[SL1][run1][pCa1][dFL1][Names1][0]


import pickle
with open('/home/al12local/Dropbox/Data/Kentucky/KYStretchData2019.data', 'wb') as filep:
    pickle.dump(td, filep)

# print(Values)
# print(Names)

# keys = D['results'][0,0].dtype.descr

# for j in range(len(D)):
#     d = print(D[j])
#     print(d)
    
    
# time = D[0][-5]
# force = D[0][-4]
# sl = D[0][-3]
# fl = D[0][-2]
# intensity = D[0][-1]


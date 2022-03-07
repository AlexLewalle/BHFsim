#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:29:19 2022

@author: al12local
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

from os.path import expanduser
home = expanduser('~')

# import pickle

pCa_list = [4.8, 5.4, 5.5, 5.6, 5.8, 5.9, 6.0, 6.2, 6.4, 6.6, 9.0]
dFL_list = [ 1., 0., -1., 2., -2., 0.5, -0.5] #, 1.]


def TestFlSl(SL=2.0, Patient='BDC56', Prep='13Feb19a'):
    dFL_a = []
    dSLmeas_a = []
    dFLmeas_a = []
    fig, ax = plt.subplots(2,1, num=Patient+'/'+Prep)
    for pCa1 in pCa_list:
        for dFL1 in dFL_list:
            t, sl, fl, force = GetStretchData(SL, Patient, Prep, pCa1, dFL1, ifRepeat=False)
            idx_0 = [j for j,t1  in enumerate(t) if t1<0.05]
            idx_1 = [j for j,t1  in enumerate(t) if (t1>0.1 and t1<0.4)]
            
            dFL_a.append(dFL1)
            dSLmeas_a.append(np.mean(sl[idx_1])  -  np.mean(sl[idx_0]))
            dFLmeas_a.append(np.mean(fl[idx_1])  -  np.mean(fl[idx_0]))
            ax[0].plot(dFL_a, dSLmeas_a, 'bo'), ax[0].set_ylabel('dSL')
            ax[1].plot(dFL_a, dFLmeas_a, 'bo'), ax[1].set_ylabel('dFL'), ax[1].set_xlabel('nominal')
    fig.tight_layout()
    plt.show()
            

def GetStretchData(SL=2.0, Patient='BDC56', Prep='13Feb19a', pCa=4.8, Step=+1.0, ifRepeat=False):
    
    DataDir = f'{home}/Desktop/KYStretches/SL{SL:.1f}/{Patient}/{Prep}/length_control/'
    DataFile = f'td_pCa{pCa:.1f}_Step{Step:+.1f}'
    if ifRepeat & np.isclose(Step, +0.1, atol=0.00001):
        DataFile += 'repeat'
    DataFile += '.mat'   
    D = spio.loadmat(DataDir + DataFile)['td']

    Values = [D[0][0][j] for j in range(len(D[0][0]))]
    Names = [D[0][0].dtype.descr[j][0]  for j in range(len(D[0][0]))]
    
    td = {}
    for j, Names1 in enumerate(Names):
        td[Names1] = np.ravel(Values[j])
        if len(td[Names1]) == 1:
            td[Names1] = td[Names1][0]
            
    print(Names)            
    return td['time'], td['sl'], td['fl'], td['force']


if __name__=='__main__':
    # t, sl, fl, force = GetStretchData()
    # fig,  ax = plt.subplots(3,1)
    # ax[0].plot(t, sl); ax[0].set_ylabel('SL')
    # ax[1].plot(t, fl); ax[1].set_ylabel('FL')
    # ax[2].plot(t, force); ax[2].set_ylabel('Force')
    # fig.tight_layout()
    # plt.show()

    TestFlSl()

# def TdDotMat2py()

# SL_list = [2.0, 2.3]
# DataPath = '/home/al12local/Desktop/Pilot_stretches'

# td = {}
# for SL1 in SL_list:
#     td[SL1] = {}
#     for run1 in ['a', 'b', 'c']:
#         td[SL1][run1]={}
#         for pCa1 in pCa_list:
#             td[SL1][run1][pCa1]={}
#             for jdF, dFL1 in enumerate(dFL_list):
#                 td[SL1][run1][pCa1][dFL1]={}
#                 DataFile = f'{DataPath}/SL{SL1:.1f}/{run1}/KYStretch_pCa{pCa1:.1f}_step{dFL1:+.1f}' 
#                 if jdF==7:
#                     DataFile += 'repeat'
#                 DataFile += '.mat'
#                 D = spio.loadmat(DataFile)['td']
                
#                 Values = [D[0][0][j] for j in range(len(D[0][0]))]
#                 Names = [D[0][0].dtype.descr[j][0]  for j in range(len(D[0][0]))]

                
#                 for j, Names1 in enumerate(Names):
#                     td[SL1][run1][pCa1][dFL1][Names1] = np.ravel(Values[j])
#                     if len(td[SL1][run1][pCa1][dFL1][Names1]) == 1:
#                         td[SL1][run1][pCa1][dFL1][Names1] = td[SL1][run1][pCa1][dFL1][Names1][0]



# with open('~/Dropbox/Data/Kentucky/KYStretchData2019.data', 'wb') as filep:
#     pickle.dump(td, filep)

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


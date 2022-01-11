#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:51:47 2022

@author: al12local
"""
import numpy as np
import matplotlib.pyplot as plt
import Land2017_sim as M

Model1 = M.Land2017()

Lambda_array = np.linspace(0.5, 1.5,100)
h_array = [Model1.h(Lambda1) for Lambda1 in Lambda_array]
Ca50_array =[Model1.Ca50(Lambda1) for Lambda1 in Lambda_array]

fig1, ax1 = plt.subplots(ncols=2)
ax1[0].plot(Lambda_array, h_array)
ax1[0].set_xlabel('Lambda')
ax1[0].set_ylabel('h(Lambda)')

ax1[1].plot(Lambda_array, Ca50_array)
ax1[1].set_xlabel('Lambda')
ax1[1].set_ylabel('Ca50(Lambda)')

plt.show()

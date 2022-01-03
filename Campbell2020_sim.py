#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:42:23 2021

@author: al12local
"""

import numpy as np
from matplotlib import pyplot as plt

class Campbell2020:
    N0 =  6.9e16 #/m2   # number of myosin heads in a hypothetical cardiac half sarcomere - see eq. 8
    
    # Parameter values from Table S1 of Campbell2020
    kon = 5e8 # M-1s-1
    koff = 200 # s-1
    kcoop = 5 # dimensionless
    k1 = 2 # s-1
    k2 = 200 # s-1
    k3 = 100 # s-1 nm-1
    k40 = 200 # s-1
    k41 = 0.1 # s-1 nm-4
    kcb = 0.001 # pN nm-1
    xps = 5 # nm
    kfalloff = 0.0024 #nm-1
    sigma = 500 # N m-2
    L = 80 # nm
    Lslack = 900 # nm
    kforce = 1.74e-4 # N-1m2   (borrowed from Campbell2018 for the rat! Human value not specified in Campbell2020 Table S1!)
    
    kB = 1.38e-23 # J K-1
    T = 298 # K
    xmin = -10 # nm
    xmax = 10 # nm
    dx = 0.5 #nm

    lthin = 1120 # nm   from Campbell2009 eq3
    lthick = 815 # nm   from Campbell2009 eq3
    lbare = 80 # nm     from Campbell2009 eq3
    lambdafalloff = 0.005 # nm-1      from Campbell2009 eq3
    
    
    

    def Noverlap(self, xhs):
        """
        Proportion of crossbridges participating in the kinetic scheme in each half sarcomere.
        From Campbell2009   eq. 3
        xhs = half-sarcomere length
        """
        xoverlap = self.lthin + self.lthick -xhs
        xmaxoverlap = self.lthick - self.lbare
        if xhs > self.lthick + self.lthin:
            return 0
        elif xhs > self.lthin + self.lbare:
            return xoverlap/xmaxoverlap
        elif xhs > self.lthin:
            return 1
        elif xhs > self.lthin - self.lambdafalloff**-1: 
            return 1 - self.lambdafalloff*(self.lthin - xhs)
        else:
            return 0
        

    def dYdt(self, Y, t):
        Noff, Non, MOFF, MON, *MFG = Y
        
        xhs = 1000

        Nbound = sum(MFG)        
        
        Jon = self.kon * self.Cai * (self.Noverlap(xhs)(xhs)-Non) * (1 + self.kcoop*(Non/self.Noverlap(xhs)))           #eq S1
        Joff = self.koff(Non-Nbound)*(1+self.kcoop*(self.Noverlap(xhs)-Non)/self.Noverlap(xhs))               #eq S2
        J1 = self.k1 * (1+self.kforce*Ftotal)*MOFF                               #eq S3
        J2 = self.k2 * MON                                                           #eq S4
        def J3(self, x):
            return self.k3 * np.exp(-self.kcb*x^2/2/self.kB/self.T) * MON * (Non-Nbound)            #eq S5
        def J4(self, x):
            return (k40 + k41*(x-xps)^4) * MFG(x)                                #eq S6
    
    
        # State variable ODEs                   eq S7
        dNoffdt = -Jon + Joff
        dNondt = Jon - Joff
        dMOFFdt = -J1 + J2
        dMONdt = J1 - J2 + sum([J4(xi)-J3(xi) for xi in np.arange(xmin, xmax+dx, dx)])
        dMFGdt = [J3(xi)-J4(xi) for xi in np.arange(xmin, xmax+dx, dx)]      # <---- make sure this works!
        
        return (dNoffdt, dNondt, dMOFFdt, dMONdt, dMFGdt)


    def Factive(self, Y):
        Noff, Non, MOFF, MON, *MFG = Y
        return self.N0*self.kcb * sum([MFG])
    
    
    
if __name__ == "__main__":
    M1 = Campbell2020()
    x = np.linspace(700, 2100, 1000)
    y = np.array([None]*len(x))
    for i1, x1 in enumerate(x):
        y[i1] = M1.Noverlap(x1)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('xhs /nm')
    plt.ylabel('Noverlap')
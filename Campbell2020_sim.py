#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:42:23 2021

@author: al12local
"""

import numpy as np
from matplotlib import pyplot as plt

xmin = -10.; xmax = 10.; dx=0.5;
xMFG = np.arange(xmin, xmax+dx, dx)  # nm
xps = 5 # nm   power stroke

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
    L = 80 # nm            parameter in Fpassive; not the sarcomere length!
    Lslack = 900 # nm      half-sarcomere length for which passive force = 0
    kforce = 1.74e-4 # N-1m2   (borrowed from Campbell2018 for the rat! Human value not specified in Campbell2020 Table S1!)
    
    
    
    kB = 1.38e-23 # J K-1
    T = 298 # K

    lthin = 1120 # nm   from Campbell2009 eq3
    lthick = 815 # nm   from Campbell2009 eq3
    lbare = 80 # nm     from Campbell2009 eq3
    lambdafalloff = 0.005 # nm-1      from Campbell2009 eq3
    
    
    Lambda_ext = 0

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
        
        Lambda_ext = 1
        xhs = 1000 * Lambda_ext # nm

        Nbound = sum(MFG)        
        
        Jon = self.kon * self.Cai * (self.Noverlap(xhs)-Non) * (1 + self.kcoop*(Non/self.Noverlap(xhs)))           #eq S1
        Joff = self.koff(Non-Nbound)*(1+self.kcoop*(self.Noverlap(xhs)-Non)/self.Noverlap(xhs))               #eq S2
        J1 = self.k1 * (1+self.kforce*self.Ftotal(Y))*MOFF                               #eq S3
        J2 = self.k2 * MON                                                           #eq S4
        
        # J3 and J4 are arrays, for elements in xMFG
        J3 = self.k3 * np.exp(-self.kcb*xMFG^2/2/self.kB/self.T) * MON * (Non-Nbound)            #eq S5
        J4 = (self.k40 + self.k41*(xMFG-xps)^4) * MFG                                #eq S6
    
    
        # State variable ODEs                   eq S7
        dNoffdt = -Jon + Joff
        dNondt = Jon - Joff
        dMOFFdt = -J1 + J2
        dMONdt = J1 - J2 + sum(J4-J3)
        dMFGdt = J3-J4 
        
        return (dNoffdt, dNondt, dMOFFdt, dMONdt, dMFGdt)

    def Get_ss1(self, Y0=[1.,0.,0.,0.]+[0.]*len(xMFG) ):
        
        

    def Factive(self, Y):
        Noff, Non, MOFF, MON, *MFG = Y
        return self.N0*self.kcb * sum(np.array(MFG)*(xMFG + xps))
    
    def Fpassive(self, Y):
        Noff, Non, MOFF, MON, *MFG = Y
        return self.sigma * (np.exp((self.xhs-self.Lslack)/self.L)-1)     # eq. S9 of Campbell2020
    
    def Ftotal(self, Y):
        return self.Factive(Y) + self.Fpassive(Y)
    
    def QuickStretchResponse(self, dL, t):
        pass
    
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
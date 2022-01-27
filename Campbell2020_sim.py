#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:42:23 2021

@author: al12local
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint, ode




class Campbell2020:

    # %% Model parameters
    N0 =  6.9e16 #/m2   # number of myosin heads in a hypothetical cardiac half sarcomere - see eq. 8

    # Parameter values from Table S1 of Campbell2020
    kon = 5e8 *10**0 # M-1s-1
    koff = 200  # s-1
    kcoop = 5 # dimensionless
    k1 = 2# s-1
    k2 = 200 # s-1
    k3 = 100 *2# s-1 nm-1
    k40 = 200 # s-1
    k41 = 0.1 # s-1 nm-4
    kcb = 0.001 # pN nm-1
    xps = 5 # nm
    kfalloff = 0.0024 #nm-1
    sigma = 500 # N m-2
    L = 80 # nm            parameter in Fpassive; not the sarcomere length!
    Lslack = 900 # nm      half-sarcomere length for which passive force = 0
    kforce = 1.74e-4 # N-1m2   (borrowed from Campbell2018 for the rat! Human value not specified in Campbell2020 Table S1!)



    kB = 1.38e-23 * 10**30 # J K-1
    T = 273+37 # K

    lthin = 1120 # nm   from Campbell2009 eq3
    lthick = 815 # nm   from Campbell2009 eq3
    lbare = 80 # nm     from Campbell2009 eq3
    lambdafalloff = 0.005 # nm-1      from Campbell2009 eq3

    xmin = -20. + xps; xmax = 20.+xps; dx=0.5;
    xMFG = np.arange(xmin, xmax+dx, dx)  # nm

    Lambda_ext = 1

    SL0 = 2100 # nm  resting sarcomere length
    Cai = 10**-4

    # %%Functions
    def xhs(self):
        return (self.SL0 * self.Lambda_ext)/2

    def Noverlap(self):
        """
        Proportion of crossbridges participating in the kinetic scheme in each half sarcomere.
        From Campbell2009   eq. 3
        xhs = half-sarcomere length
        """
        xhs = self.xhs()
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

        Nbound = sum(MFG)

        # thin filaments
        Jon = self.kon * self.Cai * (self.Noverlap()-Non) * (1 + self.kcoop*(Non/self.Noverlap()))           #eq S1
        Joff = self.koff*(Non-Nbound)*(1+self.kcoop*(self.Noverlap()-Non)/self.Noverlap())               #eq S2

        # thick filaments
        J1 = self.k1 * (1+self.kforce*self.Ftotal(Y))*MOFF                               #eq S3
        J2 = self.k2 * MON                                                           #eq S4
        #  (J3 and J4 are arrays, for elements in xMFG)
        J3 = self.k3 * np.exp(-self.kcb*self.xMFG**2/2/self.kB/self.T) * MON * (Non-Nbound)            #eq S5
        J4 = (self.k40 + self.k41*(self.xMFG-self.xps)**4) * MFG                                #eq S6


        # State variable ODEs                   eq S7
        dNoffdt = -Jon + Joff
        dNondt = Jon - Joff
        dMOFFdt = -J1 + J2
        dMONdt = J1 - J2 + sum(J4-J3)
        dMFGdt = J3-J4

        return (dNoffdt, dNondt, dMOFFdt, dMONdt, *dMFGdt)

    def Get_ss(self, Y0=np.array([1.,0.,1.,0.]+[0.]*len(xMFG)) , ifPlot=False):
        tmax = 0.1
        t = np.linspace(0,tmax,100)
        Ysol = odeint(self.dYdt, Y0, t).transpose()
        Ta = self.Factive(Ysol)
        if ifPlot:
            fig_ss, ax_ss = plt.subplots(nrows=5, figsize=(7,10))
            ax_ss[0].plot(t, Ysol[0]); ax_ss[0].set_ylabel('Noff')
            ax_ss[1].plot(t, Ysol[1]); ax_ss[1].set_ylabel('Non')
            ax_ss[2].plot(t, Ysol[2]); ax_ss[2].set_ylabel('MOFF')
            ax_ss[3].plot(t, Ysol[3]); ax_ss[3].set_ylabel('MON')
            ax_ss[4].plot(t, Ysol[4:].transpose()); ax_ss[4].set_ylabel('MFG')
            # ax_ss.plot(self.Factive(Ysol))
            fig_ss.suptitle(f'pCa = {-np.log10(self.Cai)}')
            fig_ss.tight_layout()

            # fig_FG, ax_FG = plt.subplots(num='FG myosin ss')
            # ax_FG.plot(self.xMFG, Ysol[4:,-1])
            # plt.show()
        # print(f'Steady state: Noff = {Ysol[0,-1]}')
        # print(f'              Non  = {Ysol[1,-1]}')
        # print(f'              MOFF  = {Ysol[2,-1]}')
        # print(f'              MON  = {Ysol[3,-1]}')
        # print(self.Noverlap())
        return Ysol[:,-1]

    def Factive(self, Y):
        Noff, Non, MOFF, MON, *MFG = Y

        return self.N0*self.kcb * sum(  (np.array(MFG).transpose()*(self.xMFG + self.xps)).transpose() )

    def Fpassive(self, Y):
        Noff, Non, MOFF, MON, *MFG = Y
        xhs = (self.SL0 * self.Lambda_ext)/2
        return self.sigma * (np.exp((xhs-self.Lslack)/self.L)-1)     # eq. S9 of Campbell2020

    def Ftotal(self, Y):
        return self.Factive(Y) + self.Fpassive(Y)

    def QuickStretchResponse(self, dL, t):
        pass

def DoFpCa(PSet=[None], Lambda0 = 1., ifPlot=False):
    if ifPlot:
        figFpCa = plt.figure(num=f'F-pCa, Lambda0={Lambda0}', figsize=(7,7))
        ax_FpCa = figFpCa.add_subplot(2,1,1)
        ax_Fmax = figFpCa.add_subplot(2,3,4)
        ax_nH = figFpCa.add_subplot(2,3,5)
        ax_EC50 = figFpCa.add_subplot(2,3,6)

    Fmax_a = [None]*len(PSet)
    nH_a = [None]*len(PSet)
    EC50_a = [None]*len(PSet)

    Cai_array = 10**np.linspace(-10, -5, 10)
    for i1, PSet1 in enumerate(PSet):
        print(f'Doing FpCa (Lambda0={Lambda0})- PSet {i1}')
        Model = Campbell2020() #(PSet1)
        print(f'Noverlap = {Model.Noverlap()}')
        Model.Lambda_ext = Lambda0
        F_array = [None]*len(Cai_array)


        for iCai, Cai1 in enumerate(Cai_array):
            print(f'         iCai={iCai}, Cai={Cai1}')
            Model.Cai = Cai1
            Yss = Model.Get_ss(ifPlot=True)
            F_array[iCai] = Model.Factive(Yss)
        # plt.close()

        from scipy.optimize import curve_fit
        HillFn = lambda x, ymax, n, ca50 : ymax* x**n/(x**n + ca50**n)
        HillParams, cov = curve_fit(HillFn, Cai_array, F_array,
                                    p0=[F_array[-1], 1,
                                        [ca for ica, ca in enumerate(Cai_array) if F_array[ica]>F_array[-1]/2][0] ])   #10**-7])
        Fmax_a[i1] = HillParams[0]
        nH_a[i1] = HillParams[1]
        EC50_a[i1] = HillParams[2]

        if ifPlot:
            ax_FpCa.semilogx(Cai_array, F_array)
            ax_FpCa.set_xlabel('Cai (M)')
            ax_FpCa.set_ylabel('F')
            ax_FpCa.set_title(f'F-pCa, Lambda={Lambda0}')

    if ifPlot:
        bins = 10
        ax_Fmax.hist(Fmax_a, bins=bins, range=(0, max(Fmax_a)*1.1));
        ax_Fmax.set_xlabel('Fmax')
        ax_nH.hist(nH_a, bins=bins, range=(0, max(nH_a)*1.1)); ax_nH.set_xlabel('nH')
        ax_EC50.hist(EC50_a, bins=bins, range=(0, max(EC50_a)*1.1)); ax_EC50.set_xlabel('EC50')

    return {'Fmax':Fmax_a, 'nH':nH_a, 'EC50':EC50_a}





if __name__ == "__main__":
    M1 = Campbell2020()
    # Yss = M1.Get_ss(ifPlot=True)

    DoFpCa(ifPlot=True)
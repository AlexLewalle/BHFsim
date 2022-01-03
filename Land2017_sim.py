
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import scipy.optimize

import lhsmdu
lhsmdu.setRandomSeed(1)
import seaborn as sns



Features = {}    # This is where 'features' of the experimental results are stored, for use in GSA.
PSet = {}

class Land2017:

    AllParams = ('a', 'b', 'k', 'eta_l', 'eta_s', 'k_trpn', 'ntrpn', 'Ca50ref', 'ku', 'nTm', 'trpn50', 'kuw', 'kws', 'rw', 'rs', 'gs','gw', 'phi', 'Aeff','beta0', 'beta1', 'Tref', 'kue','keu','kforce')
    
    # Passive tension parameters
    a = 2.1e3  # Pa
    b = 9.1 # dimensionless
    k = 7 # dimensionless
    eta_l = 0.2 # s         # 200e3 # /s
    eta_s = 0.02 #          # 20e3 # /s

    # Active tension parameters
    k_trpn = 0.1e3 # /s
    ntrpn = 2 # dimensionless
    Ca50ref = 2.5e-6  # M
    ku = 1e3 # /s
    nTm = 2.2 # dimensionless
    trpn50 = 0.35 # dimensionless (CaTRPN in Eq.9 is dimensionless as it represents a proportion)
    kuw = 0.026e3 #0.026e3 # /s
    kws = 0.004e3 # /s
    rw = 0.5
    rs = 0.25
    gs = 0.0085e3 # /s (assume "/ms" was omitted in paper)
    gw = 0.615e3  # /s (assume "/ms" was omitted in paper)
    phi = 2.23
    Aeff = 25
    beta0 = 2.3
    beta1 = -2.4
    Tref = 40.5e3  # Pa
    
    kue = 0 # 200 # s-1    (<-- k2 in Campbell2020; rate constant from unattached "ON" state to "OFF" state.)
    keu = 2 # s-1      (<-- k1 in Campbell2020; rate constant from "OFF" state to unattacherd "ON" state.)
    kforce = 0 #1.74e-4 # Pa-1    # value missing in Campbell2020 !! This is the value from Campbell2018 for rat.

    Cai =  10**(-4) # M


    kwu = kuw *(1/rw -1) - kws 	# eq. 23
    ksu = kws*rw*(1/rs - 1)		# eq. 24
    kb = ku*trpn50**nTm /(1-rs-(1-rs)*rw)
    Aw = Aeff * rs/((1-rs)*rw + rs) 		# eq. 26
    As = Aw 		# eq. 26

    L0 = 1.9    
    dLambda_ext = 0.1
    Lambda_ext = 1
    dLambdadt_fun = None   # This gets specified by particular experiments

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : Dictionary specifying the multiplicatin factor for altering parameter values.
        **kwargs : Direct specification of parameter values.
        When both args and kwargs are specified, the parameter values are first set according to kwargs, and then modified according to args.
        
        """
        # if len(args)>0:
        #     print('args = ', args[0])
        # print('kwargs = ' ,  kwargs)
        
        # Initialise range dictionary
        self.ParRange = {}
        self.ParBounds = {}
        for param1 in self.AllParams:
            self.ParRange[param1] = (0.5, 2) #(0.2, 5)
            self.ParBounds[param1] = (self.ParRange[param1][0]*getattr(self, param1),
                                      self.ParRange[param1][1]*getattr(self, param1))
            if self.ParBounds[param1][0]>self.ParBounds[param1][1]:     # Ensure the lower bound is less than the upper bound.
                self.ParBounds[param1] = (self.ParBounds[param1][1], self.ParBounds[param1][0])
        
        for param, value in kwargs.items(): # Set specific parameter values explicitly, if provided as keyword arguments.
            setattr(self, param, value)
        if len(args)>0:
            for param, fac in args[0].items():
                setattr(self, param, getattr(self, param) * fac)
        

    def Ca50(self, Lambda):
        return self.Ca50ref * (1 + self.beta1*(min(Lambda, 1.2)-1))

    def h(self, Lambda):
        def hh(Lambda):
            return 1 + self.beta0*(Lambda + np.minimum(Lambda, 0.87) -1.87)
        return np.maximum(0, hh(np.minimum(Lambda,1.2)))

    def Ta(self, Y):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, E = Y.transpose()
        return self.h(Lambda)* self.Tref/self.rs * (S*(Zs+1) + W*Zw)
    
    def Ta_S(self, Y):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, E = Y.transpose()
        return self.h(Lambda)* self.Tref/self.rs * (S*(Zs+1) )
    
    def Ta_W(self, Y):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, E = Y.transpose()
        return self.h(Lambda)* self.Tref/self.rs * ( W*Zw)
    

    def F1(self, Y):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, E = Y.transpose()
        C = Lambda-1
        return self.a*(np.exp(self.b*C)-1) 

    def F2(self, Y):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, E = Y.transpose()
        return self.a*self.k*(Lambda-1 - Cd)
    
    def Ttotal(self, Y):
        return self.Ta(Y) + self.F1(Y) + self.F2(Y)

    def l_bounds(self, WhichParams):
        if WhichParams=='AllParams':
            WhichParams = self.AllParams 
        return [self.ParBounds[param1][0] for param1 in WhichParams]
    def u_bounds(self, WhichParams):
        if WhichParams=='AllParams':
            WhichParams = self.AllParams 
        return [self.ParBounds[param1][1] for param1 in WhichParams]
    

    # Analytical values for steady state
    def Get_ss(self): 
        Lambda_ss = self.Lambda_ext
        Cd_ss = Lambda_ss - 1        
        CaTRPN_ss = ((self.Cai/self.Ca50(Lambda_ss))**-self.ntrpn + 1)**-1
        U_ss = (1 \
                + self.kb/self.ku*CaTRPN_ss**-self.nTm \
                + self.kws*self.kuw/(self.ksu*(self.kwu+self.kws)) \
                + self.kuw/(self.kwu+self.kws) \
                + self.kue/self.keu \
                )**-1 
        W_ss = self.kuw/(self.kwu+self.kws)*U_ss
        S_ss = self.kws/self.ksu * W_ss
        # B_ss = U_ss/(1-self.rs-(1-self.rs)*self.rw)
        B_ss = self.kb/self.ku * CaTRPN_ss**-self.nTm * U_ss
        E_ss = self.kue/self.keu * U_ss
        Zw_ss = 0
        Zs_ss = 0
        Y_ss = np.array((CaTRPN_ss, B_ss, S_ss, W_ss , Zs_ss, Zw_ss, Lambda_ss, Cd_ss, E_ss))
        
        return Y_ss




    def dYdt(self, Y, t):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, E = Y
        U = 1-B-S-W  -E
        
        gwu = self.gw * abs(Zw)      # eq. 15
        if Zs+1 < 0:        # eq. 17
            gsu = self.gs*(-Zs-1)
        elif Zs+1 > 1:
            gsu = self.gs*Zs
        else:
            gsu = 0
        
        cw = self.phi * self.kuw * U/W
        cs = self.phi * self.kws * W/S
        dZwdt = self.Aw*self.dLambdadt_fun(t) - cw*Zw
        dZsdt = self.As*self.dLambdadt_fun(t) - cs*Zs
        dCaTRPNdt = self.k_trpn*((self.Cai/self.Ca50(Lambda))**self.ntrpn*(1-CaTRPN)-CaTRPN)     # eq. 9
        # kb = self.ku * self.trpn50**self.nTm/ (1 - self.rs - (1-self.rs)*self.rw)     # eq. 25
        dBdt = self.kb*CaTRPN**(-self.nTm/2)*U - self.ku*CaTRPN**(self.nTm/2)*B     # eq. 10
        dWdt = self.kuw*U -self.kwu*W - self.kws*W - gwu*W     # eq. 12
        dSdt = self.kws*W - self.ksu*S - gsu*S        # eq. 13

        dEdt = self.kue*U - self.keu*(1+self.kforce*self.Ttotal(Y))*E       # additional factor (1+kforce*Ftotal) to be included; but kforce is missing in Campbell 2020
        
        dLambdadt = self.dLambdadt_fun(t)    # Allow this function to be defined within particular experiments
        if Lambda-1-Cd > 0 :     # i.e., dCd/dt>0    (from eq. 5)
            dCddt = self.k/self.eta_l * (Lambda-1-Cd)     # eq. 5
        else:
            dCddt = self.k/self.eta_s * (Lambda-1-Cd)     # eq. 5
            
        return (dCaTRPNdt, dBdt, dSdt, dWdt, dZsdt, dZwdt, dLambdadt, dCddt, dEdt)    
    
    def dYdt_pas(self, Y, t):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, E = Y

        dZwdt = 0
        dZsdt = 0
        dCaTRPNdt = 0
        # kb = self.ku * self.trpn50**self.nTm/ (1 - self.rs - (1-self.rs)*self.rw)     # eq. 25
        dBdt = 0
        dWdt = 0
        dSdt = 0
        dEdt = 0
        
        dLambdadt = self.dLambdadt_fun(t)    # Allow this function to be defined within particular experiments
        if Lambda-1-Cd > 0 :     # i.e., dCd/dt>0    (from eq. 5)
            dCddt = self.k/self.eta_l * (Lambda-1-Cd)     # eq. 5
        else:
            dCddt = self.k/self.eta_s * (Lambda-1-Cd)     # eq. 5
            
        return (dCaTRPNdt, dBdt, dSdt, dWdt, dZsdt, dZwdt, dLambdadt, dCddt, dEdt)    

    def QuickStretchActiveResponse(self, dLambda, t):
        self.dLambdadt_fun = lambda t: 0
        CaTRPN_0, B_0, S_0, W_0 , Zs_0, Zw_0, Lambda_0, Cd_0, E_0 = self.Get_ss()
        Zs_0 = Zs_0 + self.As*dLambda         
        Zw_0 = Zw_0 + self.Aw*dLambda
        self.Lambda_ext = Lambda_0 + dLambda
        Y0 = [CaTRPN_0, B_0, S_0, W_0 , Zs_0, Zw_0, Lambda_0+dLambda, Cd_0, E_0]
        Ysol1 = odeint(self.dYdt, Y0, t)
        # Ysol1 = solve_ivp(self.dYdt, [t[0], t[-1]], Y0, t_eval=t)
        return Ysol1
    
    def QuickStretchPassiveResponse(self, dLambda, t):
        self.dLambdadt_fun = lambda t: 0
        CaTRPN_0 = 0
        B_0 = 1
        S_0 = 0
        W_0 = 0
        Zs_0 = 0
        Zw_0 = 0
        Lambda_0 = self.Lambda_ext
        Cd_0 = Lambda_0 - 1        
        Zs_0 = 0 
        Zw_0 = 0 
        E_0 = 0
        self.Lambda_ext = Lambda_0 + dLambda
        Y0 = [CaTRPN_0, B_0, S_0, W_0 , Zs_0, Zw_0, Lambda_0+dLambda, Cd_0, E_0]
        Ysol1 = odeint(self.dYdt_pas, Y0, t)
        # Ysol1 = solve_ivp(self.dYdt, [t[0], t[-1]], Y0, t_eval=t)
        return Ysol1

        

    
    def GetQuickStretchFeatures(self, F,t, F0):
        """
        Get scalar features of a single quick-stretch experiment
    
        Parameters
        ----------
        F : list of two quick-stretch responses, corresponding to stretch and release
        """
        
        frac_dec = 0.7
        Fdec0 = (F[0][0]-F[0][-1])*(1-frac_dec)  + F[0][-1]
        tdec0 = [t1 for it, t1 in enumerate(t) if F[0][it]>Fdec0][-1]
        Fdec1 = (F[1][0]-F[1][-1])*(1-frac_dec)  + F[1][-1]
        tdec1 = [t1 for it, t1 in enumerate(t) if F[1][it]<Fdec1][-1]
        
        return {'Fss' : F0[0],
                'dFss' : F0[1]/F0[0] ,
                'rFpeak': max(F[0])/F0[0], 
                'drFpeak': (max(F[0])-min(F[1])) / F0[0],
                'tdecay': tdec0,
                'dtdecay': tdec1/tdec0
                #'Fmin': F[0][jmin] if jmin>jmax else None, 
                #'Fmintime': t[jmin]  if jmin>jmax else None,
                }
        
        # return {'Fpeak': max(F[0]), 
        #         'Fmin': min(F[0]), 
        #         'Fmintime': t[np.argmin(F[0])] 
        #         }
        
    def GetQuickStretchPassiveFeatures(self, F, t, F0):
        exp_fun = lambda t, a, b, c : a*np.exp(-b*t) + c
        a0 = F[0][0]- F[0][-1]
        c0 = F[0][-1]
        b0 = 1/ ([t1 for i1, t1 in enumerate(t) 
                 if (F[0][i1]-F[0][-1])  /  (F[0][0]-F[0][-1])  < 1/2.718 ][0])
        Fit0, covFit = scipy.optimize.curve_fit(exp_fun, t, F[0], p0=[a0,b0,c0])
        
        a1 = F[1][0]- F[1][-1]
        b1 = 1
        c1 = F[1][-1]
        Fit1, covFit = scipy.optimize.curve_fit(exp_fun, t, F[1], p0=[a1,b1,c1])
        
        return {'Fss_pas': F[0][0],
                'dFss_pas': (F[1][0] - F[0][0])/F[0][0],
                'tdec_pas0': 1/Fit0[1],
                'tdec_pas1': 1/Fit1[1]}
        

def Sample(xmin=0, xmax=1, numsamples=1):
    """
    Homogeneous sampling between xmin and xmax.
    """
    return xmin + np.random.rand(numsamples) * (xmax-xmin)


def MakeParamSetLH(Model, numsamples, *args):
    """
    inputs:
        numsamples: number of parameter sets to generate
        args: equals either:
                - the names of the parameters to be modified. The other parameters remain fixed at their default values in the model.
                - the name "AllParams" makes all the parameters variable.
            The variation range of each parameter is specified below.
    outputs:
        Returns a list of dictionaries, with the key being the name of the parameter, and the corresponding value.
    """
    if numsamples == 0 :
        result = [{}]
        for p1 in Model.AllParams:
            result[0][p1] = 1
    else:
        if 'AllParams' in args:
            TargetSet = Model.AllParams
        else:
            TargetSet = args
        numparams = len(TargetSet)
        LHsamples = lhsmdu.sample(numparams, numsamples)
        
        result = [None]*numsamples
        for s1 in range(0,numsamples):        
            result[s1] = {}
            for ip1, p1 in enumerate(TargetSet):
                result[s1][p1] = Model.ParRange[p1][0] + (Model.ParRange[p1][1]-Model.ParRange[p1][0])*LHsamples[ip1, s1]  # Creates a dictionary entry for each modified parameter p1.
    return result


def PlotS1(gsa, Feature):
    plt.style.use("seaborn")
    figS1, axS1 = plt.subplots(1,2, figsize=(15,6))
    sns.boxplot(ax=axS1[0], data=gsa.S1)
    sns.boxplot(ax=axS1[1], data=gsa.ST)
    figS1.suptitle(Feature)
    axS1[0].set_xticklabels(gsa.ylabels, rotation=45); axS1[0].set_title('S1')
    axS1[1].set_xticklabels(gsa.ylabels, rotation=45); axS1[0].set_title('Stotal')
    

def DoQuickStretches(PSet, Cai=10**-4, L0=1.9, ifPlot = False):
    print(f'Quick stretches (L0={L0}, Cai={Cai})')
    text1=f'Quick stretches (L0={L0}, Cai={Cai})'
    if ifPlot:
        fig1, ax1 = plt.subplots(nrows=4, ncols=2, num=f'Quick stretches (L0={L0}, pCai={-np.log10(Cai)})', figsize=(21, 7))
        fig2, ax2 = plt.subplots(nrows=7, ncols=2, num=f'States (L0={L0}, pCai={-np.log10(Cai)})', figsize=(7,10))
    
    # Fbase_a = [None]*len(PSet)
    # Fpeak_a = [None]*len(PSet)
    # Fmin_a = [None]*len(PSet)
    # Fmintime_a = [None]*len(PSet)
    # DFssrel_a = [None]*len(PSet)
    Features_a = {}   # initialise features dictionary
    
    
    for iPSet, PSet1 in enumerate(PSet):
        Model = Land2017(PSet1)
        Model.Cai = Cai
        Model.L0 = L0
        dLambda = 0.1
        t = np.linspace(0, 1, 1000)
        Ysol = [None]*2; F0 = [None]*2; F0_S = [None]*2; F0_W = [None]*2; F0_pas = [None]*2; F = [None]*2; F_S = [None]*2; F_W = [None]*2; F_pas = [None]*2
        for i1, dLambda1 in enumerate((dLambda, -dLambda)):
            F0[i1] = Model.Ttotal(Model.Get_ss())
            F0_S[i1] = Model.Ta_S(Model.Get_ss())
            F0_W[i1] = Model.Ta_W(Model.Get_ss())
            F0_pas[i1] = Model.F1(Model.Get_ss()) + Model.F2(Model.Get_ss())
            Ysol[i1] = Model.QuickStretchActiveResponse(dLambda1, t)
            """
            Ysol is  a 2-by-1000-by-8 array containing the ODE solutions for the quick stretch and the quick contraction steps, as functions of time.
            State variables are :  CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, E
            """
            F[i1] = Model.Ttotal(Ysol[i1])
            F_S[i1] = Model.Ta_S(Ysol[i1])
            F_W[i1] = Model.Ta_W(Ysol[i1])
            F_pas[i1] = Model.F1(Ysol[i1]) + Model.F2(Ysol[i1])
            
            
            print(f'{iPSet} :  {Model.F1(Ysol[i1])[-1]} ,  {Model.F2(Ysol[i1])[-1]}')
            if ifPlot: 
                plt.figure(5); plt.plot(t, Ysol[i1][:,7], label=f'{iPSet}')
                plt.legend()

        
        features = Model.GetQuickStretchFeatures(F,t, F0)
        for feat1 in features.keys():
            if not feat1 in Features_a:
                Features_a[feat1] = [None]*len(PSet)
            Features_a[feat1][iPSet] = features[feat1]
        
        # Fbase_a[iPSet] = features['Fbase']
        # Fpeak_a[iPSet] = features['Fpeak']
        # Fmin_a[iPSet] = features['Fmin'] 
        # Fmintime_a[iPSet] = features['Fmintime']
        # DFssrel_a[iPSet] = features['DFssrel']
        
        if ifPlot:
            normF = 1 #F0[0]
            ax1[0,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0[0], F0[0]], F[0])/normF); ax1[0,0].set_ylabel('F_total'); 
            ax1[0,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0[1], F0[1]], F[1])/normF)
            ax1[1,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_S[0], F0_S[0]], F_S[0])/normF); ax1[1,0].set_ylabel('F_S')
            ax1[1,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_S[1], F0_S[1]], F_S[1])/normF)
            ax1[2,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_W[0], F0_W[0]], F_W[0])/normF); ax1[2,0].set_ylabel('F_W')
            ax1[2,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_W[1], F0_W[1]], F_W[1])/normF)
            ax1[3,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_pas[0], F0_pas[0]], F_pas[0])/normF); ax1[3,0].set_ylabel('F_passive')
            ax1[3,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_pas[1], F0_pas[1]], F_pas[1])/normF)
            fig1.suptitle(f'L0={Model.L0}, Cai={Model.Cai},    dLambda={dLambda}')
            
            # No need to plot CaTRPN (i.e. Ysol[0][:,0]) since no dynamics.
            ax2[0,0].plot(t, Ysol[0][:,1]); ax2[0,0].set_ylabel('B')
            ax2[0,1].plot(t, Ysol[1][:,1])        
            ax2[1,0].plot(t, Ysol[0][:,2]); ax2[1,0].set_ylabel('S')
            ax2[1,1].plot(t, Ysol[1][:,2])
            ax2[2,0].plot(t, Ysol[0][:,3]); ax2[2,0].set_ylabel('W')
            ax2[2,1].plot(t, Ysol[1][:,3])
            ax2[3,0].plot(t, np.ones(len(Ysol[0]))-(Ysol[0][:,1]+Ysol[0][:,2]+Ysol[0][:,3])); ax2[3,0].set_ylabel('U')
            ax2[3,1].plot(t, np.ones(len(Ysol[0]))-(Ysol[1][:,1]+Ysol[1][:,2]+Ysol[1][:,3]))        
            ax2[4,0].plot(t, Ysol[0][:,4]); ax2[4,0].set_ylabel('Zs')
            ax2[4,1].plot(t, Ysol[1][:,4])       
            ax2[5,0].plot(t, Ysol[0][:,5]); ax2[5,0].set_ylabel('Zw')
            ax2[5,1].plot(t, Ysol[1][:,5])       
            ax2[6,0].plot(t, Ysol[0][:,7]); ax2[6,0].set_ylabel('Cd')
            ax2[6,1].plot(t, Ysol[1][:,7])       
            
    if ifPlot: 
        for i2 in list(range(4)): # equalize axis ranges for stretch and release.
            ylim = (min( ax1[i2,0].get_ylim()[0], ax1[i2,1].get_ylim()[0]), 
                    max( ax1[i2,0].get_ylim()[1], ax1[i2,1].get_ylim()[1])) 
            ax1[i2,0].set_ylim(ylim); ax1[i2,1].set_ylim(ylim)
            # ax1[i2,0].spines['right'].set_visible(False)
            # ax1[i2,1].spines['left'].set_visible(False); ax1[i2,1].axes.yaxis.set_ticks([])
        for i2 in list(range(4)): # equalize axis ranges for stretch and release.
            ylim = (0, max( ax2[i2,0].get_ylim()[1], ax2[i2,1].get_ylim()[1]))
            ax2[i2,0].set_ylim((0,1.05)) #ylim)
            ax2[i2,1].set_ylim((0,1.05)) #ylim)
            ax2[i2,0].spines['right'].set_visible(False)
            ax2[i2,1].spines['left'].set_visible(False); ax2[i2,1].axes.yaxis.set_ticks([])
        for i2 in (4,5):
            ylim = (min( ax2[i2,0].get_ylim()[0], ax2[i2,1].get_ylim()[0]), 
                    max( ax2[i2,0].get_ylim()[1], ax2[i2,1].get_ylim()[1])) 
            ax2[i2,0].set_ylim(ylim); ax2[i2,1].set_ylim(ylim)
        fig1.suptitle(f'Quick stretches (L0={L0}, pCai={-np.log10(Cai)})')
        fig2.suptitle(f'States (L0={L0}, pCai={-np.log10(Cai)})')
    plt.show()
    return Features_a
    # return {'Fbase': Fbase_a,
    #         'Fpeak': Fpeak_a,
    #         'Fmin': Fmin_a, 
    #         'Fmintime': Fmintime_a,
    #         'DFssrel': DFssrel_a
    #         }

def DoQuickStretches_passive(PSet, L0=1.9, ifPlot = False):
    print(f'Passive Quick stretches (L0={L0})')
    text1=f'Passive Quick stretches (L0={L0})'
    if ifPlot:
        fig1, ax1 = plt.subplots(nrows=4, ncols=2, num=f'Passive quick stretches (L0={L0})', figsize=(21, 7))
        fig2, ax2 = plt.subplots(nrows=7, ncols=2, num=f'States (L0={L0})', figsize=(7,10))
    
    Features_a = {}   # initialise features dictionary
    
    
    for iPSet, PSet1 in enumerate(PSet):
        Model = Land2017(PSet1)
        Model.L0 = L0
        dLambda = 0.1
        t = np.linspace(0, 1, 1000)
        Ysol = [None]*2; F0 = [None]*2; F0_S = [None]*2; F0_W = [None]*2; F0_pas = [None]*2; F = [None]*2; F_S = [None]*2; F_W = [None]*2; F_pas = [None]*2
        for i1, dLambda1 in enumerate((dLambda, -dLambda)):
            F0_S[i1] = 0
            F0_W[i1] = 0
            F0_pas[i1] = Model.F1(Model.Get_ss()) + Model.F2(Model.Get_ss())
            F0[i1] = F0_pas[i1]
            Ysol[i1] = Model.QuickStretchPassiveResponse(dLambda1, t)
            """
            Ysol is  a 2-by-1000-by-8 array containing the ODE solutions for the quick stretch and the quick contraction steps, as functions of time.
            State variables are :  CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, E
            """
            F[i1] = Model.Ttotal(Ysol[i1])
            F_S[i1] = Model.Ta_S(Ysol[i1])
            F_W[i1] = Model.Ta_W(Ysol[i1])
            F_pas[i1] = Model.F1(Ysol[i1]) + Model.F2(Ysol[i1])
            
            if ifPlot:
                plt.figure(5); plt.plot(t, Ysol[i1][:,7], label=f'{iPSet}')
                plt.legend()

        
        features = Model.GetQuickStretchPassiveFeatures(F,t, F0)
        # exp_fun = lambda t, a, b, c : a*np.exp(-b*t) + c
        # plt.plot(t, exp_fun(t, features[0], features[1], features[2]), 'k--')
        for feat1 in features.keys():
            if not feat1 in Features_a:
                Features_a[feat1] = [None]*len(PSet)
            Features_a[feat1][iPSet] = features[feat1]
        
        # Fbase_a[iPSet] = features['Fbase']
        # Fpeak_a[iPSet] = features['Fpeak']
        # Fmin_a[iPSet] = features['Fmin'] 
        # Fmintime_a[iPSet] = features['Fmintime']
        # DFssrel_a[iPSet] = features['DFssrel']
        
        if ifPlot:
            normF = 1 #F0[0]
            ax1[0,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0[0], F0[0]], F[0])/normF); ax1[0,0].set_ylabel('F_total'); 
            ax1[0,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0[1], F0[1]], F[1])/normF)
            ax1[1,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_S[0], F0_S[0]], F_S[0])/normF); ax1[1,0].set_ylabel('F_S')
            ax1[1,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_S[1], F0_S[1]], F_S[1])/normF)
            ax1[2,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_W[0], F0_W[0]], F_W[0])/normF); ax1[2,0].set_ylabel('F_W')
            ax1[2,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_W[1], F0_W[1]], F_W[1])/normF)
            ax1[3,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_pas[0], F0_pas[0]], F_pas[0])/normF); ax1[3,0].set_ylabel('F_passive')
            ax1[3,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_pas[1], F0_pas[1]], F_pas[1])/normF)
            fig1.suptitle(f'L0={Model.L0}, dLambda={dLambda}')
            
            # No need to plot CaTRPN (i.e. Ysol[0][:,0]) since no dynamics.
            ax2[0,0].plot(t, Ysol[0][:,1]); ax2[0,0].set_ylabel('B')
            ax2[0,1].plot(t, Ysol[1][:,1])        
            ax2[1,0].plot(t, Ysol[0][:,2]); ax2[1,0].set_ylabel('S')
            ax2[1,1].plot(t, Ysol[1][:,2])
            ax2[2,0].plot(t, Ysol[0][:,3]); ax2[2,0].set_ylabel('W')
            ax2[2,1].plot(t, Ysol[1][:,3])
            ax2[3,0].plot(t, np.ones(len(Ysol[0]))-(Ysol[0][:,1]+Ysol[0][:,2]+Ysol[0][:,3])); ax2[3,0].set_ylabel('U')
            ax2[3,1].plot(t, np.ones(len(Ysol[0]))-(Ysol[1][:,1]+Ysol[1][:,2]+Ysol[1][:,3]))        
            ax2[4,0].plot(t, Ysol[0][:,4]); ax2[4,0].set_ylabel('Zs')
            ax2[4,1].plot(t, Ysol[1][:,4])       
            ax2[5,0].plot(t, Ysol[0][:,5]); ax2[5,0].set_ylabel('Zw')
            ax2[5,1].plot(t, Ysol[1][:,5])       
            ax2[6,0].plot(t, Ysol[0][:,7]); ax2[6,0].set_ylabel('Cd')
            ax2[6,1].plot(t, Ysol[1][:,7])       
            
    if ifPlot: 
        for i2 in list(range(4)): # equalize axis ranges for stretch and release.
            ylim = (min( ax1[i2,0].get_ylim()[0], ax1[i2,1].get_ylim()[0]), 
                    max( ax1[i2,0].get_ylim()[1], ax1[i2,1].get_ylim()[1])) 
            ax1[i2,0].set_ylim(ylim); ax1[i2,1].set_ylim(ylim)
            # ax1[i2,0].spines['right'].set_visible(False)
            # ax1[i2,1].spines['left'].set_visible(False); ax1[i2,1].axes.yaxis.set_ticks([])
        for i2 in list(range(4)): # equalize axis ranges for stretch and release.
            ylim = (0, max( ax2[i2,0].get_ylim()[1], ax2[i2,1].get_ylim()[1]))
            ax2[i2,0].set_ylim((0,1.05)) #ylim)
            ax2[i2,1].set_ylim((0,1.05)) #ylim)
            ax2[i2,0].spines['right'].set_visible(False)
            ax2[i2,1].spines['left'].set_visible(False); ax2[i2,1].axes.yaxis.set_ticks([])
        for i2 in (4,5):
            ylim = (min( ax2[i2,0].get_ylim()[0], ax2[i2,1].get_ylim()[0]), 
                    max( ax2[i2,0].get_ylim()[1], ax2[i2,1].get_ylim()[1])) 
            ax2[i2,0].set_ylim(ylim); ax2[i2,1].set_ylim(ylim)
        fig1.suptitle(f'Quick stretches (L0={L0})')
        fig2.suptitle(f'States (L0={L0})')
    plt.show()
    return Features_a
    # return {'Fbase': Fbase_a,
    #         'Fpeak': Fpeak_a,
    #         'Fmin': Fmin_a, 
    #         'Fmintime': Fmintime_a,
    #         'DFssrel': DFssrel_a
    #         }



def DoFpCa(PSet, ifPlot = False):
    if ifPlot:
        figFpCa = plt.figure(num='F-pCa', figsize=(7,7))
        ax_FpCa = figFpCa.add_subplot(2,1,1)
        ax_Fmax = figFpCa.add_subplot(2,3,4)
        ax_nH = figFpCa.add_subplot(2,3,5)
        ax_EC50 = figFpCa.add_subplot(2,3,6)

    Fmax_a = [None]*len(PSet)
    nH_a = [None]*len(PSet)
    EC50_a = [None]*len(PSet)

    # for i1 in range(0,len(PSet)):
    #     Model = Land2017(PSet[i1])
    #     Fmax, nH, EC50 = Model.Get_FpCa(Model.Lambda0, FpCaPlotAxis=ax_FpCa)
    #     Fmax_a[i1]=Fmax
    #     nH_a[i1]=nH
    #     EC50_a[i1]=EC50

    Cai_array = 10**np.linspace(-9, -4, 100)
    for i1, PSet1 in enumerate(PSet):
        Model = Land2017(PSet1)
        F_array = [None]*len(Cai_array)

        
        for iCai, Cai1 in enumerate(Cai_array):
            Model.Cai = Cai1            
            F_array[iCai] = Model.Ta(Model.Get_ss())
                
        from scipy.optimize import curve_fit
        HillFn = lambda x, ymax, n, ca50 : ymax* x**n/(x**n + ca50**n)
        HillParams, cov = curve_fit(HillFn, Cai_array, F_array, p0=[F_array[-1], 1, 10**-7])
        Fmax_a[i1] = HillParams[0]
        nH_a[i1] = HillParams[1]
        EC50_a[i1] = HillParams[2]
            
        if ifPlot:
            ax_FpCa.plot(Cai_array, F_array)
            ax_FpCa.set_xlabel('Cai (M)')
            ax_FpCa.set_ylabel('F')    
        
    if ifPlot:
        bins = 10
        ax_Fmax.hist(Fmax_a, bins=bins, range=(0, max(Fmax_a)*1.1)); 
        ax_Fmax.set_xlabel('Fmax')
        ax_nH.hist(nH_a, bins=bins, range=(0, max(nH_a)*1.1)); ax_nH.set_xlabel('nH')
        ax_EC50.hist(EC50_a, bins=bins, range=(0, max(EC50_a)*1.1)); ax_EC50.set_xlabel('EC50')
    
    return {'Fmax':Fmax_a, 'nH':nH_a, 'EC50':EC50_a}
        
            
def DoChirps(PSet, ifPlot = False):

    fmin = 0.2
    fmax = 10
    tmax = 30
    f_fun = lambda t : fmin + (fmax-fmin)*t/tmax 
    dfdt_fun = lambda t : (fmax-fmin)/tmax
    pointspercycle = 1000
    dLambda_amplitude = 0.1

    if ifPlot: figsol, axsol = plt.subplots(nrows=2, ncols=1, num='Chirp solutions')    
    fig, ax = plt.subplots(ncols=3, nrows=1, num = 'Chirp experiments', figsize = (15,7))
    
    t = [0]
    while t[-1] < tmax :
        t = t + [t[-1]+1/pointspercycle/f_fun(t[-1])]
    t = np.array(t)

    for i1, PSet1 in enumerate(PSet):
        print(f'Doing PSet {i1}')
        Model = Land2017(PSet[i1])
        Model.dLambdadt_fun = lambda t : \
            dLambda_amplitude * np.cos(2*np.pi*f_fun(t)*t) * 2*np.pi*( f_fun(t) + t*dfdt_fun(t) )   
    
        Y_ss0 = Model.Get_ss()
        Ysol = odeint(Model.dYdt, Y_ss0, t)
        Tasol = Model.Ta(Ysol)
        if ifPlot: axsol[0].plot(t, Tasol)
        
        dTa = np.diff(Tasol)
        dTa1 = dTa[1:]
        dTa0 = dTa[0:-1]
        jMaxTa = [i+1 for i, x in enumerate(np.sign(dTa0) > np.sign(dTa1)) if x]
        jMinTa = [i+1 for i, x in enumerate(np.sign(dTa0) < np.sign(dTa1)) if x]
        numOK = min(len(jMaxTa), len(jMinTa)) - 3
        jMaxTa = jMaxTa[-numOK:]; jMinTa =jMinTa[-numOK:]
        if ifPlot: axsol[1].plot(t[jMaxTa], [1+dLambda_amplitude]*len(jMaxTa), 'g^', t[jMinTa], [1-dLambda_amplitude]*len(jMaxTa), 'rv')

        AmpTa = (np.array(Tasol[jMaxTa]) - np.array(Tasol[jMinTa])) /2
        Stiffness = AmpTa / dLambda_amplitude
        DphaseTa = (2*np.pi*f_fun(t[jMinTa])*t[jMinTa])%(2*np.pi)
        DphaseTa = np.array([x if x>0 else x + np.pi for x in DphaseTa ])
        DphaseTa = 3/2*np.pi - DphaseTa
        
        # Stiffness, Dphase, tdyn = AnalyseDynamic(Model.dLambdadt_fun(t), Tasol, t, f_fun)
        ax[0].plot(f_fun(t[jMaxTa]), Stiffness)
        ax[1].plot(f_fun(t[jMaxTa]), DphaseTa)
        ax[2].plot(Stiffness*np.cos(DphaseTa), Stiffness*np.sin(DphaseTa))

    if ifPlot: axsol[1].plot(t, Ysol[:, 6])
    
def DoDynamic(PSet, ifPlot = False):

    fmin = 0.01
    fmax = 3000
    Numf = 100
    f_list = np.logspace(np.log10(fmin), np.log10(fmax), Numf) 
    
    # f_list = np.array([0.1])
    
    numcycles = 30
    pointspercycle = 1000
    dLambda_amplitude = 0.1
    

    if ifPlot: figsol, axsol = plt.subplots(nrows=2, ncols=1, num='Dynamic solutions')    
    fig, ax = plt.subplots(ncols=3, nrows=1, num = 'Dynamic experiments', figsize = (15,7))
    
    for i1, PSet1 in enumerate(PSet):
        print(f'Doing PSet {i1}')
        Model = Land2017(PSet[i1])

        Stiffness = [None]*len(f_list)
        DphaseTa = [None]*len(f_list)
    
        for ifreq, freq in enumerate(f_list):
            print(f'   Doing PSet {i1},  f{ifreq} = {freq}')            
            t = np.linspace(0, numcycles/freq, numcycles*pointspercycle)


            Model.dLambdadt_fun = lambda t : \
                dLambda_amplitude * np.cos(2*np.pi*freq*t) * 2*np.pi*freq
    
            Y_ss0 = Model.Get_ss()
            Ysol = odeint(Model.dYdt, Y_ss0, t)
            Tasol = Model.Ta(Ysol)
            
            from scipy.optimize import curve_fit
            def Sin_fun(x, *a):
                return a[0]*np.sin(2*np.pi*freq*x + a[1]) + a[2]
            SinFit, cov = curve_fit(Sin_fun, 
                                    t[-pointspercycle:], Tasol[-pointspercycle:],
                                    p0=(  (max(Tasol[-pointspercycle:])-min(Tasol[-pointspercycle:]))/2,
                                        0, Tasol[-1] ) )
            print(SinFit)
            # if ifPlot:        axsol[0].plot(t[-pointspercycle:]/t[-1], Sin_fun(t[-pointspercycle:], SinFit[0],SinFit[1],SinFit[2]), 'k--')
            Stiffness[ifreq] = SinFit[0] / dLambda_amplitude
            DphaseTa[ifreq] = SinFit[1]
                
            dTa = np.diff(Tasol)
            dTa1 = dTa[1:]
            dTa0 = dTa[0:-1]
            jMaxTa = [i+1 for i, x in enumerate(np.sign(dTa0) > np.sign(dTa1)) if x]
            jMinTa = [i+1 for i, x in enumerate(np.sign(dTa0) < np.sign(dTa1)) if x]
            
            if ifPlot: axsol[0].plot(t/t[-1], Tasol); \
                axsol[1].plot(t/t[-1], Ysol[:,6]); 
        
            # AmpTa = (Tasol[jMaxTa[-1]] - Tasol[jMinTa[-1]]) /2
            # Stiffness[ifreq] = AmpTa / dLambda_amplitude
            # DphaseTa[ifreq] = 3*np.pi/2 - (2*np.pi*freq)*t[jMinTa[-1]] % (2*np.pi)  
            
            
        # Stiffness, Dphase, tdyn = AnalyseDynamic(Model.dLambdadt_fun(t), Tasol, t, f_fun)
        ax[0].semilogx(f_list, Stiffness, '-')
        ax[1].semilogx(f_list, DphaseTa, '-')
        ax[2].plot(Stiffness*np.cos(DphaseTa), Stiffness*np.sin(DphaseTa))
        ax[2].set_aspect('equal', adjustable='box')

    
# def AnalyseDynamic (A,B, t, f):
#     dA = np.diff(A)
#     dA1 = dA[1:]
#     dA0 = dA[0:-1]
#     jMaxA = [i for i, x in enumerate(np.sign(dA0) > np.sign(dA1)) if x]
#     jMinA = [i for i, x in enumerate(np.sign(dA0) < np.sign(dA1)) if x]
#     dB = np.diff(B)
#     dB1 = dB[1:]
#     dB0 = dB[0:-1]
#     jMaxB = [i+1 for i, x in enumerate(np.sign(dB0) > np.sign(dB1)) if x]
#     jMinB = [i+1 for i, x in enumerate(np.sign(dB0) < np.sign(dB1)) if x]
#     jOK = min(len(jMaxA), len(jMinA), len(jMaxB), len(jMinB))
#     ; jMaxB = jMaxB[-jOK:0]; jMinB =jMinB[-jOK:0]; 

#     ampA = (A[jMaxA]-A[jMinA])/2
#     ampB = (B[jMaxB]-B[jMinB])/2
#     Modulus = ampB / ampA
#     Dphase = (t[jMaxB]-t[jMaxA]) * 2*np.pi*f( (t[jMaxB]+t[jMaxA])/2 )
    
#     return Modulus, Dphase, (t[jMaxB]+t[jMaxA])/2




if __name__ == '__main__':

    Model0 = Land2017()

    Nsamples = 10
    print('Doing LH sampling')
    PSet = MakeParamSetLH(Model0, Nsamples,  'AllParams') #'eta_l', 'eta_s', 'k_trpn', 'ku', 'kuw', 'kws') # 
    print('LH sampling completed')
    
    DoQuickStretches(PSet, Cai=10**-4, L0=1.9, ifPlot = True)    
    # DoQuickStretches_passive(PSet, L0=1.9, ifPlot = True)
    # DoFpCa(PSet, ifPlot = True)
    # DoChirps(PSet, ifPlot = True)
    # DoDynamic(PSet, ifPlot = True)

    plt.show()
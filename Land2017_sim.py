
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, ode
# from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import scipy.optimize
from scipy.optimize import curve_fit

import lhsmdu
lhsmdu.setRandomSeed(1)
import seaborn as sns

# def ode15s(f, Y0, t):
#     S = ode(f)
#     S.set_integrator('vode', method='bdf', order=15, nsteps=3000)
#     S.set_initial_value(Y0,t[0])
#     return S.integrate(t)

Features = {}    # This is where 'features' of the experimental results are stored, for use in GSA.
PSet = {}

ifkforce = True

class Land2017:

    AllParams = ('a', 
                 'b', 
                 'k', 
                 'eta_l', 
                 'eta_s', 
                 'k_trpn_on',
                 'k_trpn_off',
                 'ntrpn', 
                 'Ca50ref', 
                 'ku', 
                 'nTm', 
                 'trpn50', 
                 'kuw', 
                 'kws', 
                 'rw', 
                 'rs', 
                 'gs',
                 'gw', 
                 'phi', 
                 'Aeff',
                 'beta0', 
                 'beta1', 
                 'Tref', 
                 'k2',
                 'k1',
                 'kforce')

    # Passive tension parameters
    a = 2.1e3  # Pa
    b = 9.1 # dimensionless
    k = 7 # dimensionless
    eta_l =  200e-3 # s
    eta_s =  20e-3 # s

    # Active tension parameters
    k_trpn_on = 0.1e3 # /s
    k_trpn_off = 0.03e3 # /s
    ntrpn = 2 # dimensionless
    Ca50ref = 2.5e-6  # M
    ku = 1000  # /s
    nTm = 2.2 # dimensionless
    trpn50 = 0.35 # dimensionless (CaTRPN in Eq.9 is dimensionless as it represents a proportion)
    kuw = 0.026e3    # /s
    kws = 0.004e3 # /s
    rw = 0.5
    rs = 0.25
    gs = 0.0085e3 # /s (assume "/ms" was omitted in paper)
    gw = 0.615e3 # /s (assume "/ms" was omitted in paper)
    phi = 2.23
    Aeff = 25
    beta0 = 2.3
    beta1 = -2.4
    Tref = 40.5e3  # Pa

    k2 = 20 #200 # s-1    (<-- k2 in Campbell2020; rate constant from unattached "ON" state to "OFF" state.)
    k1 = 2 # s-1      (<-- k1 in Campbell2020; rate constant from "OFF" state to unattacherd "ON" state.)
    kforce = 1.74e-4 # Pa-1    # value missing in Campbell2020 !! This is the value from Campbell2018 for rat.

    Cai =  10**(-4) # M


    def kwu(self):
        return self.kuw *(1/self.rw -1) - self.kws 	# eq. 23
    def ksu(self):
        return self.kws*self.rw*(1/self.rs - 1)		# eq. 24
    def kb(self): 
        return self.ku*self.trpn50**self.nTm /(1-self.rs-(1-self.rs)*self.rw)

    L0 = 1.9
    dLambda_ext = 0.1       # To be specified by the experiment
    Lambda_ext = 1          # To be specified by the experiment
    dLambdadt_fun = lambda t: 0    # This gets specified by particular experiments

    # Aw = Aeff * rs/((1-rs)*rw + rs) 		# eq. 26
    # As = Aw
    def Aw(self):
        return self.Aeff * self.rs/((1-self.rs)*self.rw + self.rs) 		# eq. 26
    def As(self):
        return self.Aw()
    
    
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : Dictionary  (PSet) specifying the multiplicatin factor for altering parameter values.
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
            ParFac = 1.5
            self.ParRange[param1] = (1/ParFac, ParFac) #(0.2, 5)
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
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda)* self.Tref/self.rs * (S*(Zs+1) + W*Zw)

    def Ta_S(self, Y):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda)* self.Tref/self.rs * (S*(Zs+1) )

    def Ta_W(self, Y):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda)* self.Tref/self.rs * ( W*Zw)


    def F1(self, Y):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        C = Lambda-1
        return self.a*(np.exp(self.b*C)-1)

    def F2(self, Y):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
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




    # Analytical values for steady state   :  NO OFF STATE!!
    def Get_ss_analytic(self):
        Lambda_ss = self.Lambda_ext
        Cd_ss = Lambda_ss - 1
        CaTRPN_ss = ((self.Cai/self.Ca50(Lambda_ss))**-self.ntrpn + 1)**-1
        U_ss = (1 \
                + self.kb()/self.ku*CaTRPN_ss**-self.nTm \
                + self.kws*self.kuw/(self.ksu()*(self.kwu()+self.kws)) \
                + self.kuw/(self.kwu()+self.kws) \
                )**-1
        W_ss = self.kuw/(self.kwu()+self.kws)*U_ss
        S_ss = self.kws/self.ksu() * W_ss
        # B_ss = U_ss/(1-self.rs-(1-self.rs)*self.rw)
        B_ss = self.kb()/self.ku * CaTRPN_ss**-self.nTm * U_ss
        Zw_ss = 0
        Zs_ss = 0
        Y_ss0 = np.array((CaTRPN_ss, B_ss, S_ss, W_ss , Zs_ss, Zw_ss, Lambda_ss, Cd_ss))
        return Y_ss0

    def Get_ss(self):
        """
        Calculate steady-state probabilities for all the states, ASSUMING NO FORCE DEPENDENCE IN OFF->ON TRANSITION!
        """
        Lambda_ss = self.Lambda_ext
        Cd_ss = Lambda_ss - 1
        CaTRPN_ss = (self.k_trpn_off/self.k_trpn_on * (self.Cai/self.Ca50(Lambda_ss))**-self.ntrpn + 1)**-1

        Mat = np.zeros( (6,6) )
        #         U  B  S  W  BE  UE
        Mat[0] = [1, 1, 1, 1, 1,  1]
        Mat[1,0] = self.kb()*CaTRPN_ss**(-self.nTm/2)
        Mat[1,1] = -self.ku*CaTRPN_ss**(self.nTm/2) - self.k2
        Mat[1,4] = self.k1
        Mat[2,3] = self.kws
        Mat[2,2] = -self.ksu()
        Mat[3,0] = self.kuw
        Mat[3,3] = -self.kwu() - self.kws
        Mat[4,5] = self.kb()*CaTRPN_ss**(-self.nTm/2)
        Mat[4,4] = -self.ku*CaTRPN_ss**(self.nTm/2) - self.k1  #  <--- no force dependence in k1 here => approximation
        Mat[4,1] = self.k2
        Mat[5,5] = -self.kb()*CaTRPN_ss**(-self.nTm/2) - self.k1  #  <--- no force dependence in k1 here => approximation
        Mat[5,4] = self.ku*CaTRPN_ss**(self.nTm/2)
        Mat[5,0] = self.k2
        Zw_ss = 0
        Zs_ss = 0

        U_ss, B_ss, S_ss, W_ss, BE_ss, UE_ss =  np.dot(  np.linalg.inv(Mat),  np.array([1,0,0,0,0,0]).T  )
        Y_ss0 = np.array((CaTRPN_ss, B_ss, S_ss, W_ss , Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss))

        if ifkforce:
            # print('Include force-dependent OFF state k1.')
            dLambdadt_fun_STORE = self.dLambdadt_fun   # store dLambdadt_fun in case it has been set externally
            self.dLambdadt_fun = lambda t: 0.
            Ysolss = odeint(self.dYdt, Y_ss0, np.arange(0,2,0.1),
                            atol=1e-4)                                  # <------ TOLERANCE
            # plt.plot(self.Ta(Ysolss), '.-')
            self.dLambdadt_fun = dLambdadt_fun_STORE
            return Ysolss[-1]   # return the steady the last point of the solution
        else:
            return Y_ss0

    def gwu(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        return self.gw * abs(Zw)      # eq. 15
    
    def gsu(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        if Zs+1 < 0:        # eq. 17
            return self.gs*(-Zs-1)
        elif Zs+1 > 1:
            return self.gs*Zs
        else:
            return 0
        
    def cw(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        return self.phi * self.kuw * self.U(Y)/W
    
    def cs(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        return self.phi * self.kws * W/S

    def U(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        return 1-B-S-W  -BE-UE

    def dYdt(self, Y, t):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y

        
        dZwdt = self.Aw()*self.dLambdadt_fun(t) - self.cw(Y)*Zw
        dZsdt = self.As()*self.dLambdadt_fun(t) - self.cs(Y)*Zs
        dCaTRPNdt = self.k_trpn_on*(self.Cai/self.Ca50(Lambda))**self.ntrpn*(1-CaTRPN)-   self.k_trpn_off*CaTRPN     # eq. 9        # kb = self.ku * self.trpn50**self.nTm/ (1 - self.rs - (1-self.rs)*self.rw)     # eq. 25
        dBdt = self.kb()*CaTRPN**(-self.nTm/2)*self.U(Y)  \
            - self.ku*CaTRPN**(self.nTm/2)*B  \
            - self.k2*B   \
            + self.k1*(1+self.kforce*max((self.Ta(Y),0.))) * BE  # eq.10 in Land2017, amended to include myosin off state dynamics
        dWdt = self.kuw*self.U(Y) -self.kwu()*W - self.kws*W - self.gwu(Y)*W     # eq. 12
        dSdt = self.kws*W - self.ksu()*S - self.gsu(Y)*S        # eq. 13


        # New "myosin off" states
        dBEdt = self.kb()*CaTRPN**(-self.nTm/2)*UE  \
            - self.ku*CaTRPN**(self.nTm/2)*BE  \
            + self.k2*B \
            - self.k1*(1+self.kforce*self.Ttotal(Y)) * BE
        dUEdt = -self.kb()*CaTRPN**(-self.nTm/2)*UE  \
            + self.ku*CaTRPN**(self.nTm/2)*BE  \
            + self.k2*self.U(Y) \
            - self.k1*(1+self.kforce*self.Ttotal(Y)) * UE



        dLambdadt = self.dLambdadt_fun(t)    # Allow this function to be defined within particular experiments
        if Lambda-1-Cd > 0 :     # i.e., dCd/dt>0    (from eq. 5)
            dCddt = self.k/self.eta_l * (Lambda-1-Cd)     # eq. 5
        else:
            dCddt = self.k/self.eta_s * (Lambda-1-Cd)     # eq. 5
            

        return (dCaTRPNdt, dBdt, dSdt, dWdt, dZsdt, dZwdt, dLambdadt, dCddt, dBEdt, dUEdt)

    def dYdt_pas(self, Y, t):
        CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE = Y

        dZwdt = 0
        dZsdt = 0
        dCaTRPNdt = 0
        # kb = self.ku * self.trpn50**self.nTm/ (1 - self.rs - (1-self.rs)*self.rw)     # eq. 25
        dBdt = 0
        dWdt = 0
        dSdt = 0
        dBEdt = 0
        dUEdt = 0

        dLambdadt = self.dLambdadt_fun(t)    # Allow this function to be defined within particular experiments
        if Lambda-1-Cd > 0 :     # i.e., dCd/dt>0    (from eq. 5)
            dCddt = self.k/self.eta_l * (Lambda-1-Cd)     # eq. 5
        else:
            dCddt = self.k/self.eta_s * (Lambda-1-Cd)     # eq. 5

        return (dCaTRPNdt, dBdt, dSdt, dWdt, dZsdt, dZwdt, dLambdadt, dCddt, dBEdt, dUEdt)

    def CaiStepResponse(self, Cai1, Cai2, t):
        """
        Starting from the steady state, change Cai from Cai1 to Cai2. Lambda is fixed at
        Lambda_ext (specified previously).
        The initial state Y0 that is input into the ODE solver assumes that the Cai step occurs instantaneously.
        """
        self.dLambdadt_fun = lambda t: 0
        self.Cai = Cai1
        CaTRPN_0, B_0, S_0, W_0 , Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0 = self.Get_ss()
        Y0 = [CaTRPN_0, B_0, S_0, W_0 , Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0]
        self.Cai = Cai2
        Ysol1 = odeint(self.dYdt, Y0, t)
        # Ysol1 = solve_ivp(self.dYdt, [t[0], t[-1]], Y0, t_eval=t)
        return Ysol1

    def QuickStretchActiveResponse(self, dLambda, t):
        """
        Starting from the steady state, increase Lambda by dLambda. The initial Lambda is
        given by Lambda_ext (specified previously) **before** the stretch is performed.
        Then, this Lambda_ext is **updated**.
        The initial state Y0 that is input into the ODE solver assumes that the step change occurs
        instantaneously, so that only Lambda_0 (and Zs_0 and Zw_0, which are dependent on Lambda_0)
        are altered.
        """
        self.dLambdadt_fun = lambda t: 0
        CaTRPN_0, B_0, S_0, W_0 , Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0 = self.Get_ss()
        Zs_0 = Zs_0 + self.As()*dLambda
        Zw_0 = Zw_0 + self.Aw()*dLambda
        self.Lambda_ext = Lambda_0 + dLambda    # <----- update Lambda_ext
        Y0 = [CaTRPN_0, B_0, S_0, W_0 , Zs_0, Zw_0, Lambda_0+dLambda, Cd_0, BE_0, UE_0]
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
        BE_0 = 0
        UE_0 = 0
        self.Lambda_ext = Lambda_0 + dLambda
        Y0 = [CaTRPN_0, B_0, S_0, W_0 , Zs_0, Zw_0, Lambda_0+dLambda, Cd_0, BE_0, UE_0]
        Ysol1 = odeint(self.dYdt_pas, Y0, t)
        # Ysol1 = solve_ivp(self.dYdt, [t[0], t[-1]], Y0, t_eval=t)
        return Ysol1


    def GetCaiStepFeatures(self, F,t):
        """
        Get scalar features of a Cai step change experiment

        Parameters
        ----------
        F : Cai step response
        t : time
        """

        return {'dFCaiStep': np.abs(F[-1]-F[0]),
                'tCaiStep' : [t[i1] for i1, F1 in enumerate(F) if (F1-F[0])/(F[-1]-F[0])>0.5 ][0]
                }



    def GetQuickStretchFeatures(self, F,t, F0):
        """
        Get scalar features of a single quick-stretch experiment

        Parameters
        ----------
        F : list of two quick-stretch responses, corresponding to stretch and release
        t : time
        F0 : list of two steady state forces
        """

        # exp_fun = lambda t, a, b, c : a*np.exp(-b*t) + c

        # Fit0, covFit = scipy.optimize.curve_fit(exp_fun, t, F[0], p0=[F[0][0]-F[0][-1]])
        frac_dec = 0.7
        # Fdec0 = (F[0][0]-F[0][-1])*(1-frac_dec)  + F[0][-1]
        # tdec0 = [t1 for it, t1 in enumerate(t) if F[0][it]>Fdec0][-1]
        # plt.figure(10); plt.plot(t, F[0]); plt.show()
        Fdec0 = (F[0][0]-min(F[0]))*(1-frac_dec)  + min(F[0])
        tdec0 = [t1 for it, t1 in enumerate(t) if F[0][it]<Fdec0][0]   # ; print(f'tdec0 = {tdec0}')
        Fdec1 = (F[1][0]-F[1][-1])*(1-frac_dec)  + F[1][-1]
        tdec1 = [t1 for it, t1 in enumerate(t) if F[1][it]<Fdec1][-1]



        return {'Fss' : F0[0],
                'maxDF': F[0][0] - min(F[0]),
                'dFss' : F0[1] / F0[0] ,
                'rFpeak': max(F[0]) / F0[0],
                'drFpeak': (max(F[0])-min(F[1])) / F0[0],
                'tdecay': tdec0,
                'dtdecay': tdec1/tdec0
                #'Fmin': F[0][jmin] if jmin>jmax else None,
                #'Fmintime': t[jmin]  if jmin>jmax else None,
                }


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


    def SinResponse(self, freq, numcycles=4, pointspercycle=30, dLambda_amplitude=0.1, ifPlot=False):

        self.dLambdadt_fun = lambda t : \
            dLambda_amplitude * np.cos(2*np.pi*freq*t) * 2*np.pi*freq

        t = np.linspace(0, numcycles/freq, numcycles*pointspercycle)
        Y_ss0 = self.Get_ss()
        Ysol = odeint(self.dYdt, Y_ss0, t)
        Tasol = self.Ta(Ysol)

        def Sin_fun(t, *a):
            return a[0]*np.sin(2*np.pi*freq*(t + a[1])) + a[2]
        SinFit, cov = curve_fit(Sin_fun,
                                t[-pointspercycle:], Tasol[-pointspercycle:],
                                p0=(
                                    (max(Tasol[-pointspercycle:])-min(Tasol[-pointspercycle:]))/2,
                                    1/freq/4 - np.argmax(Tasol[-pointspercycle:])/pointspercycle/freq  ,
                                    np.mean(Tasol[-pointspercycle:]) ) )

        Stiffness = SinFit[0]/dLambda_amplitude
        DphaseTa = SinFit[1]*2*np.pi*freq

        if ifPlot:
            fig_sol, ax_sol = plt.subplots(nrows=2)
            ax_sol[0].plot(t, Tasol);
            ax_sol[0].plot(t, Sin_fun(t, *SinFit), 'k--'); ax_sol[0].set_ylabel('Ta')
            ax_sol[1].plot(t, Ysol[:,6]); ax_sol[1].set_ylabel('Lambda')
            fig_sol.suptitle(f'f = {freq}')
            
        return Tasol, Ysol, t, Stiffness, DphaseTa


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


def DoCaiStep(PSet, Cai1=10**-4, Cai2=10**-4, L0=1.9, ifPlot = False):
    print(f'Stepping Cai={Cai1} to {Cai2} (L0={L0})')
    text1=f'Stepping Cai={Cai1} to {Cai2} (L0={L0})'
    if ifPlot:
        fig1, ax1 = plt.subplots(nrows=3, num=f'Stepping pCai={-np.log10(Cai1)} to {-np.log10(Cai2)} (L0={L0})', figsize=(7, 7) )
        fig2, ax2 = plt.subplots(nrows=5, num=f'Cai step - States (L0={L0}, pCai={-np.log10(Cai1)} to {-np.log10(Cai2)})', figsize=(7,10))

    Features_a = {}   # initialise features dictionary
    for iPSet, PSet1 in enumerate(PSet):
        print(f'Doing Cai step - PSet {iPSet}')
        Model = Land2017(PSet1)
        Model.Cai = Cai1
        Model.L0 = L0
        t = np.linspace(0, 1, 1000)
        Ysol = None; F0 = [None]*2; F0_S = [None]*2; F0_W = [None]*2; F0_pas = [None]*2; F = [None]*2; F_S = [None]*2; F_W = [None]*2; F_pas = [None]*2

        F0 = Model.Ttotal(Model.Get_ss())
        F0_S = Model.Ta_S(Model.Get_ss())
        F0_W = Model.Ta_W(Model.Get_ss())
        F0_pas = Model.F1(Model.Get_ss()) + Model.F2(Model.Get_ss())
        Ysol = Model.CaiStepResponse(Cai1, Cai2, t)
        """
        Ysol is  a 2-by-1000-by-8 array containing the ODE solutions for the quick stretch and the quick contraction steps, as functions of time.
        State variables are :  CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, E
        """
        F = Model.Ttotal(Ysol)
        F_S = Model.Ta_S(Ysol)
        F_W = Model.Ta_W(Ysol)

        if ifPlot:
            normF = 1 #F0[0]
            ax1[0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0, F0], F)/normF); ax1[0].set_ylabel('F_total');
            ax1[1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_S, F0_S], F_S)/normF); ax1[1].set_ylabel('F_S')
            ax1[2].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_W, F0_W], F_W)/normF); ax1[2].set_ylabel('F_W')
            # ax1[3].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_pas, F0_pas], F_pas)/normF); ax1[3].set_ylabel('F_passive')

            # fig1.suptitle(f'L0={Model.L0}, Cai={Model.Cai}')

            ax2[0].plot(t, Ysol[:,0]); ax2[0].set_ylabel('CaTrpn')
            ax2[1].plot(t, Ysol[:,1]); ax2[1].set_ylabel('B')
            ax2[2].plot(t, Ysol[:,2]); ax2[2].set_ylabel('S')
            ax2[3].plot(t, Ysol[:,3]); ax2[3].set_ylabel('W')
            ax2[4].plot(t, np.ones(len(Ysol))-(Ysol[:,1]+Ysol[:,2]+Ysol[:,3])); ax2[4].set_ylabel('U')
            

        features = Model.GetCaiStepFeatures(F,t)
        for feat1 in features.keys():
            if not feat1 in Features_a:
                Features_a[feat1] = [None]*len(PSet)
            Features_a[feat1][iPSet] = features[feat1]

    return Features_a





def DoQuickStretches(PSet, Cai=10**-4, L0=1.9, ifPlot = False):
    print(f'Quick stretches (L0={L0}, Cai={Cai})')
    text1=f'Quick stretches (L0={L0}, Cai={Cai})'
    if ifPlot:
        fig1, ax1 = plt.subplots(nrows=4, ncols=2, num=f'Quick stretches (L0={L0}, pCai={-np.log10(Cai)})', figsize=(21, 7))
        fig2, ax2 = plt.subplots(nrows=9, ncols=2, num=f'Quick stretches - States (L0={L0}, pCai={-np.log10(Cai)})', figsize=(7,10))

    Features_a = {}   # initialise features dictionary


    for iPSet, PSet1 in enumerate(PSet):
        print(f'Doing Quick stretches - PSet {iPSet}')
        Model = Land2017(PSet1)
        Model.Cai = Cai
        Model.L0 = L0
        dLambda = 0.05
        t = np.linspace(0, 1, 1000)
        Ysol = [None]*2; F0 = [None]*2; F0_S = [None]*2; F0_W = [None]*2; F0_pas = [None]*2; F = [None]*2; F_S = [None]*2; F_W = [None]*2; F_pas = [None]*2
        for i1, dLambda1 in enumerate((dLambda, -dLambda)):
            Yss0 = Model.Get_ss()
            F0[i1] = Model.Ttotal(Yss0)
            F0_S[i1] = Model.Ta_S(Yss0)
            F0_W[i1] = Model.Ta_W(Yss0)
            F0_pas[i1] = Model.F1(Yss0) + Model.F2(Yss0)
            Ysol[i1] = Model.QuickStretchActiveResponse(dLambda1, t)
            """
            Ysol is  a 2-by-1000-by-8 array containing the ODE solutions for the quick stretch and the quick contraction steps, as functions of time.
            State variables are :  CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE
            """
            F[i1] = Model.Ttotal(Ysol[i1])
            F_S[i1] = Model.Ta_S(Ysol[i1])
            F_W[i1] = Model.Ta_W(Ysol[i1])
            F_pas[i1] = Model.F1(Ysol[i1]) + Model.F2(Ysol[i1])


        features = Model.GetQuickStretchFeatures(F,t, F0)
        for feat1 in features.keys():
            if not feat1 in Features_a:
                Features_a[feat1] = [None]*len(PSet)
            Features_a[feat1][iPSet] = features[feat1]

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
            ax2[0,0].plot(t, Ysol[0][:,0]); ax2[0,0].set_ylabel('CaTRPN')
            ax2[0,0].plot(t, Ysol[1][:,0]);
            ax2[1,0].plot(t, Ysol[0][:,1]); ax2[1,0].set_ylabel('B')
            ax2[1,1].plot(t, Ysol[1][:,1])
            ax2[2,0].plot(t, Ysol[0][:,2]); ax2[2,0].set_ylabel('S')
            ax2[2,1].plot(t, Ysol[1][:,2])
            ax2[3,0].plot(t, Ysol[0][:,3]); ax2[3,0].set_ylabel('W')
            ax2[3,1].plot(t, Ysol[1][:,3])
            ax2[4,0].plot(t, np.ones(len(Ysol[0]))-(Ysol[0][:,1]+Ysol[0][:,2]+Ysol[0][:,3]+Ysol[0][:,8]+Ysol[0][:,9])); ax2[4,0].set_ylabel('U')
            ax2[4,1].plot(t, np.ones(len(Ysol[0]))-(Ysol[1][:,1]+Ysol[1][:,2]+Ysol[1][:,3]))
            ax2[5,0].plot(t, Ysol[0][:,8]+Ysol[0][:,9]); ax2[5,0].set_ylabel('BE+UE')
            ax2[5,1].plot(t, Ysol[1][:,8]+Ysol[0][:,9])
            ax2[6,0].plot(t, Ysol[0][:,4]); ax2[6,0].set_ylabel('Zs')
            ax2[6,1].plot(t, Ysol[1][:,4])
            ax2[7,0].plot(t, Ysol[0][:,5]); ax2[7,0].set_ylabel('Zw')
            ax2[7,1].plot(t, Ysol[1][:,5])
            ax2[8,0].plot(t, Ysol[0][:,7]); ax2[8,0].set_ylabel('Cd')
            ax2[8,1].plot(t, Ysol[1][:,7])

    if ifPlot:
        # for i2 in list(range(5)): # equalize axis ranges for stretch and release.
        #     ylim = (min( ax1[i2,0].get_ylim()[0], ax1[i2,1].get_ylim()[0]),
        #             max( ax1[i2,0].get_ylim()[1], ax1[i2,1].get_ylim()[1]))
        #     ax1[i2,0].set_ylim(ylim); ax1[i2,1].set_ylim(ylim)
            # ax1[i2,0].spines['right'].set_visible(False)
            # ax1[i2,1].spines['left'].set_visible(False); ax1[i2,1].axes.yaxis.set_ticks([])
        for i2 in list(range(6)): # equalize axis ranges for stretch and release.
            ylim = (0, max( ax2[i2,0].get_ylim()[1], ax2[i2,1].get_ylim()[1]))
            ax2[i2,0].set_ylim((0,1.05)) #ylim)
            ax2[i2,1].set_ylim((0,1.05)) #ylim)
            ax2[i2,0].spines['right'].set_visible(False)
            ax2[i2,1].spines['left'].set_visible(False); ax2[i2,1].axes.yaxis.set_ticks([])
        for i2 in (6,7,8):
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
        print(f'Doing passive quick stretches - PSet {iPSet}')
        Model = Land2017(PSet1)
        Model.L0 = L0
        dLambda = 0.05
        t = np.linspace(0, 1, 1000)
        Ysol = [None]*2; F0 = [None]*2; F0_S = [None]*2; F0_W = [None]*2; F0_pas = [None]*2; F = [None]*2; F_S = [None]*2; F_W = [None]*2; F_pas = [None]*2
        for i1, dLambda1 in enumerate((dLambda, -dLambda)):
            Yss = Model.Get_ss()
            F0_S[i1] = 0
            F0_W[i1] = 0
            F0_pas[i1] = Model.F1(Yss) + Model.F2(Yss)
            F0[i1] = F0_pas[i1]
            Ysol[i1] = Model.QuickStretchPassiveResponse(dLambda1, t)
            """
            Ysol is  a 2-by-1000-by-8 array containing the ODE solutions for the quick stretch and the quick contraction steps, as functions of time.
            State variables are :  CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE
            """
            F[i1] = Model.Ttotal(Ysol[i1])
            F_S[i1] = Model.Ta_S(Ysol[i1])
            F_W[i1] = Model.Ta_W(Ysol[i1])
            F_pas[i1] = Model.F1(Ysol[i1]) + Model.F2(Ysol[i1])

            if ifPlot:
                plt.figure(5); plt.plot(t, Ysol[i1][:,7], '.-', label=f'{iPSet}')
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



def DoFpCa(PSet, Lambda0 = 1., ifPlot = False):
    # import multiprocessing
    # def Do_one_Ca(iCai, Cai1, PSet1, F_array):
    #     Model = Land2017(PSet1)
    #     Model.Lamda_ext  = Lambda0
    #     Model.Cai = Cai1
    #     F_array[iCai] = Model.Ta(Model.Get_ss())
        
    if ifPlot:
        figFpCa = plt.figure(num=f'F-pCa, Lambda0={Lambda0}', figsize=(7,7))
        ax_FpCa = figFpCa.add_subplot(2,1,1)
        ax_Fmax = figFpCa.add_subplot(2,3,4)
        ax_nH = figFpCa.add_subplot(2,3,5)
        ax_EC50 = figFpCa.add_subplot(2,3,6)

    Fmax_a = [None]*len(PSet)
    nH_a = [None]*len(PSet)
    EC50_a = [None]*len(PSet)

    Cai_array = 10**np.linspace(-7, -4, 100)
    for i1, PSet1 in enumerate(PSet):
        print(f'Doing FpCa (Lambda0={Lambda0})- PSet {i1}')

        Model = Land2017(PSet1)
        Model.Lambda_ext = Lambda0
        F_array = [None]*len(Cai_array)

        # process_list = []
        for iCai, Cai1 in enumerate(Cai_array):
            Model.Cai = Cai1
            F_array[iCai] = Model.Ta(Model.Get_ss())
        #     p = multiprocessing.Process(target=Do_one_Ca, args=[iCai, Cai1, PSet1, F_array])
        #     p.start()
        #     process_list.append(p)
        # for process in process_list:
        #     process.join()
            
        

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
            ax_FpCa.set_title(f'F-pCa, Lambda={Lambda0}, ifkforce={ifkforce}')

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


def DoDynamic(PSet, fmin=1, fmax=100, Numf=10, ifPlot = False):

    f_list = np.logspace(np.log10(fmin), np.log10(fmax), Numf)

    # if ifPlot: figsol, axsol = plt.subplots(nrows=2, ncols=1, num='Dynamic solutions')
    fig, ax = plt.subplots(ncols=3, nrows=1, num = 'Dynamic experiments', figsize = (15,7))

    for i1, PSet1 in enumerate(PSet):
        print(f'Doing PSet {i1}')
        Model = Land2017(PSet[i1], Cai=10**-4.8)

        Stiffness_f = [None]*len(f_list)
        DphaseTa_f = [None]*len(f_list)

        for ifreq, freq in enumerate(f_list):
            print(f'   Doing PSet {i1},  f{ifreq} = {freq}')

            Tasol, Ysol, t , Stiffness, DphaseTa = Model.SinResponse(freq, ifPlot=True)

            Stiffness_f[ifreq] = Stiffness
            DphaseTa_f[ifreq] = DphaseTa

            # if ifPlot:        axsol[0].plot(t[-pointspercycle:]/t[-1], Sin_fun(t[-pointspercycle:], SinFit[0],SinFit[1],SinFit[2]), 'k--')


            # if ifPlot:
            #     axsol[0].plot(t/t[-1], Tasol)
            #     axsol[1].plot(t/t[-1], Ysol[:,6])


        Stiffness_f = np.array(Stiffness_f)
        DphaseTa_f = np.array(DphaseTa_f)
        ax[0].semilogx(f_list, Stiffness_f, '-')
        ax[1].semilogx(f_list, DphaseTa_f, '-')
        ax[2].plot(Stiffness_f * np.cos(DphaseTa_f), Stiffness_f * np.sin(DphaseTa_f) )
        ax[2].set_aspect('equal', adjustable='box')





if __name__ == '__main__':

    Model0 = Land2017()

    Nsamples = 10  
    print('Doing LH sampling')
    PSet = MakeParamSetLH(Model0, Nsamples, 'kforce') # 'AllParams') #   'ku' )  #   'kuw' ) #   
    print('LH sampling completed')

    # for iPSet, PSet1 in enumerate(PSet):  Land2017(PSet1).Get_ss()  # Test settling of steady state when ifkforce=True

    Features_a = DoQuickStretches(PSet, Cai=10**-4, L0=1.9, ifPlot = True)
    # DoCaiStep(PSet, Cai1=10**-6, Cai2=10**-4, ifPlot=True)
    # DoQuickStretches_passive(PSet, L0=1.9, ifPlot = True)
    # D = DoFpCa(PSet, Lambda0=1.0, ifPlot = True)


    # DoChirps(PSet, ifPlot = True)
    # DoDynamic(PSet, fmin=1, fmax=100, Numf=10, ifPlot = True)

    plt.show()
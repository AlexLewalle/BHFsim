
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# from scipy.optimize import fsolve

# Passive tension parameters
a = 2.1e3 # Pa
b = 9.1
k = 7
eta_l = 200e-3 # /s      <-- CHECK
eta_s = 20e-3 # /s       <-- CHECK

# Active tension parameters
k_trpn = 0.1e3 # /s
ntrpn = 2
Ca50ref = 2.5e-6  # M
ku = 1e3 # /s
nTm = 2.2
trpn50 = 0.35
kuw = 0.026e3 # /s
kws = 0.004e3 # /s
rw = 0.5
rs = 0.25
gs = 0.0085#e3 * 1 # /s (assume "/ms" was omitted in paper)
gw = 0.615#e3
phi = 2.23
Aeff = 25
beta0 = 2.3
beta1 = -2.4
Tref = 40.5e3  # Pa


Cai =  10**(- 5) # M


kwu = kuw *(1/rw -1) - kws 	# eq. 23
ksu = kws*rw*(1/rs - 1)		# eq. 24
Aw = Aeff * rs/((1-rs)*rw + rs) 		# eq. 26
As = Aw 		# eq. 26

def Ca50(L):
    return Ca50ref * (1 + beta1*(min(L, 1.2)-1))




def h(L):
    def hh(L):
        return 1 + beta0*(L + np.minimum(L, 0.87) -1.87)
    return np.maximum(0, hh(np.minimum(L,1.2)))

def Ta(A, L):
    CaTRPN, B, S, W , Zs, Zw = A.transpose()
    return h(L)* Tref/rs * (S*(Zs+1) + W*Zw)

def F1(L):
    C = L-1
    return a*(np.exp(b*C)-1) 

def F2(L, t):
    C = L - 1
    if L>L0:
        eta = eta_l
    else:
        eta = eta_s
    return (C-(L0-1)) *a*k*np.exp(-k*t/eta)


# Analytical values for steady state
def Get_ss(L):
    CaTRPN_ss = ((Cai/Ca50(L))**-ntrpn + 1)**-1
    kb = ku*trpn50**nTm /(1-rs-(1-rs)*rw)
    U_ss = (1 + kb/ku*CaTRPN_ss**-nTm + kws*kuw/(ksu*(kwu+kws)) + kuw/(kwu+kws))**-1 
    W_ss = kuw/(kwu+kws)*U_ss
    S_ss = kws/ksu * W_ss
    B_ss = U_ss/(1-rs-(1-rs)*rw)
    Zw_ss = 0
    Zs_ss = 0
    A_ss = np.array((CaTRPN_ss, B_ss, S_ss, W_ss , Zs_ss, Zw_ss))
    return A_ss


def GetPhase(A,B, t, f):
    dA = np.diff(A)
    dA1 = dA[1:]
    dA0 = dA[0:-1]
    iMaxA = [i for i, x in enumerate(np.sign(dA0) > np.sign(dA1)) if x]
    iMinA = [i for i, x in enumerate(np.sign(dA0) < np.sign(dA1)) if x]
    dB = np.diff(B)
    dB1 = dB[1:]
    dB0 = dB[0:-1]
    iMaxB = [i for i, x in enumerate(np.sign(dB0) > np.sign(dB1)) if x]
    iMinB = [i for i, x in enumerate(np.sign(dB0) < np.sign(dB1)) if x]
    phasediff = (t[iMaxB[-1]]-t[iMaxA[-1]]) * 2*np.pi*f
    return (iMaxA, iMaxB, iMinA, iMinB, phasediff)


def QuickStretchResponse(L0, dL, t):
    dLdt = 0

    def dYdt(Y, t):
        CaTRPN, B, S, W , Zs, Zw = Y

        gwu = gw * abs(Zw)      # eq. 15
        if Zs+1 < 0:        # eq. 17
            gsu = gs*(-Zs-1)
        elif Zs+1 > 1:
            gsu = gs*Zs
        else:
            gsu = 0
        U = 1-B-S-W
        cw = phi * kuw * U/W
        cs = phi * kws * W/S
        dZwdt = Aw*dLdt - cw*Zw
        dZsdt = As*dLdt - cs*Zs
        dCaTRPNdt = k_trpn*((Cai/Ca50(+L0+dL))**ntrpn*(1-CaTRPN)-CaTRPN)     # eq. 9
        kb = ku * trpn50**nTm/ (1 - rs - (1-rs)*rw)     # eq. 25
        dBdt = kb*CaTRPN**(-nTm/2)*U - ku*CaTRPN**(nTm/2)*B     # eq. 10
        dWdt = kuw*U -kwu*W - kws*W - gwu*W     # eq. 12
        dSdt = kws*W - ksu*S - gsu*S        # eq. 13
        return (dCaTRPNdt, dBdt, dSdt, dWdt, dZsdt, dZwdt)

    CaTRPN_0, B_0, S_0, W_0 , Zs_0, Zw_0 = Get_ss(L0)
    Zs_0 = Zs_0 + As*dL
    Zw_0 = Zw_0 + Aw*dL
    Y0 = [CaTRPN_0, B_0, S_0, W_0 , Zs_0, Zw_0]
    Ysol = odeint(dYdt, Y0, t)
    return Ysol 


if __name__ == '__main__':

    fig1, ax1 = plt.subplots(nrows=1, ncols=2)

    L0 = 1.9
    dL = 0.001
    t = np.linspace(0, 1, 1000)
    Ysol = [QuickStretchResponse(L0, dL, t), 
            QuickStretchResponse(L0+dL, -dL, t)]

    F0 = [Ta(Get_ss(L0), L0),
        Ta(Get_ss(L0+dL), L0+dL)]
    F = [Ta(Ysol[0], L0+dL),
        Ta(Ysol[1], L0)]
    
    ax1[0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0[0], F0[0]], F[0]))
    ax1[1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0[1], F0[1]], F[1]))


    ylim = (ax1[1].get_ylim()[0], ax1[0].get_ylim()[1]); ax1[0].set_ylim(ylim); ax1[1].set_ylim(ylim)
    ax1[0].spines['right'].set_visible(False)
    ax1[1].spines['left'].set_visible(False); ax1[1].axes.yaxis.set_ticks([])

    plt.show()







    # plt.close('all')
    # fig1, (ax_L, ax_Ta) = plt.subplots(2,1, num='Ta')
    # fig2, (ax_Stiffness, ax_Nyq) = plt.subplots(1,2, num='Nyquist')


    # L0 = 1.0

    # f_list = 10**np.linspace(0,2,50)
    # Stiffness = np.array([])
    # DPhase = np.array([])

    # for f in f_list:
    #     print('Doing f = ', f)
    #     t = np.linspace(0, 5/f, 10001)
    #     dL = Lamplitude * np.sin(2*np.pi*f * t)
        
        
    #     A_ss0 = Get_ss(L0)
    #     CaTRPN_ss0, B_ss0, S_ss0, W_ss0 , Zs_ss0, Zw_ss0 = A_ss0
    #     Asol = odeint(dAdt, A_ss0, t)
    #     Tasol = Ta(Asol, L0)
    #     ax_L.plot(t, dL)
    #     ax_Ta.plot( t, Tasol)
        
    #     iMaxdL, iMaxTa, iMindL, iMinTa, phasediff= GetPhase(dL, Tasol, t, f)
    #     # ax_Ta.plot(t[iMaxTa], Tasol[iMaxTa], 'o')
    #     # ax_L.plot(t[iMaxdL], dL[iMaxdL], 'o')
    #     # print(phasediff)

    #     Stiffness = np.append(Stiffness, (Tasol[iMaxTa[-1]]-Tasol[iMinTa[-1]] ) / (dL[iMaxdL[-1]]-dL[iMindL[-1]] ))
    #     DPhase = np.append(DPhase, -phasediff)

    # ax_Stiffness.plot(f_list, Stiffness)
    # ax_Nyq.plot(Stiffness*np.cos(DPhase), Stiffness*np.sin(DPhase))
    # plt.show()
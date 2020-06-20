# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 10:07:30 2020

@author: Ming Cai
"""

import numpy as np
np.random.seed(2020)

restart = True     

# Test Model Fair Unfair Coin 

N = 3 # Fair Coin and Unfair Coin 
M = 2 # H=1, T=0 
Wk = np.arange(N) # hidden state set
Vk = np.arange(M) # observable state set

A = np.array([[0.7,0.2,0.1],[0.2,0.6,0.2],[0.3,0.1,0.6]]) #coin state transition matrix, row current, col next
B = np.array([[0.5, 0.5], [0.1, 0.9],[0.6, 0.4]]) #1st coin fair, 2nd coin unfair 
pi = np.array([0.3,0.4,0.3]) 

#Generate sequence for coin 
T = 10 
I = np.ndarray(T, dtype = int)
O = np.ndarray(T, dtype = int) 

def DGP(A, B, pi, T): # Data Generating Function
    Pi = np.zeros([T,N]) 
    Pi[0] = pi 
    for t in range(1,T): 
        Pi[t] = Pi[t-1]@A 
    for t in range(T): 
        I[t] = np.random.choice(Wk, p = Pi[t]) 
        O[t] = np.random.choice(Vk, p = B[I[t]]) 
    return I, O 

#Data Generation Process 
    
D = 1000 # sample size 
Obs = np.ndarray([D, T], dtype = int)
Hidden = np.ndarray([D, T], dtype = int)
for d in range(D): 
    Hidden[d], Obs[d] = DGP(A, B, pi, T) 

"""we draw indices instead of actual numbers""" 
"""it will reach a stationary distribution """ 

# Algorithm 

# Init Guesses 
A0 = np.array([[0.75, 0.2, 0.05],[0.15,0.8, 0.05],[0.1,0.15,0.75]],dtype = 'float64')
B0 = np.array([[0.5,0.5],[0.3,0.7],[0.6,0.4]],dtype = 'float64')
pi0 = np.array([0.6,0.3,0.1],dtype = 'float64') 

#Forward probability alpha recursion
def get_alpha(A, B, pi, O):
    alpha = np.ndarray([T,N],dtype = 'float64')
    alpha[0] = pi*B[:,O[0]] 
    for t in range(1, T): 
        alpha[t] = (alpha[t-1]@A)*B[:,O[t]] 
    return alpha

#Backward probability beta recursion

def get_beta(A, B, pi, O):
    beta = np.zeros([T,N], dtype = 'float64')
    beta[T-1] = np.ones(N, dtype = 'float64') 
    for t in range(1,T): 
        beta[T-1-t] = (beta[T-t]*B[:,O[T-t]])@A.transpose() 
    return beta 

# Joint probabilities 

def get_gamma(A,B,pi,O): 
    ag = get_alpha(A, B, pi, O) 
    bg = get_beta(A, B, pi, O) 
    abg = ag*bg 
    denom_abg = np.sum(abg, axis = 1) 
    gamma = np.ndarray([T,N], dtype = 'float64') 
    for i in range(N): 
        gamma[:,i] = abg[:,i]/denom_abg 
    return gamma 

def get_ksi(A, B, pi, O): 
    ak = get_alpha(A, B, pi, O) 
    bk = get_beta(A, B, pi, O) 
    ksi = np.zeros([T, N, N],dtype = 'float64') 
    numer_ksi = np.zeros([T, N, N],dtype = 'float64') 
    for t in range(T): 
        numer_ksi[t] = ak[t].reshape([N,1])*A*B[:,O[t]]*bk[t] 
        ksi[t] = numer_ksi[t]/np.sum(numer_ksi[t]) 
    return ksi 

"""
Test sequences 
I0, O0 = DGP(A, B, pi, T)
a0 = get_alpha(A0, B0, pi0, O0) 
b0 = get_beta(A0, B0, pi0, O0) 
gamma0 = get_gamma(A0, B0, pi0, O0) 
ksi0 = get_ksi(A0, B0, pi0, O0) 
"""
# Estimation

def HMEstimate(A, B, pi, Obs): 
    Gamma = np.ndarray([D, T, N])
    Ksi = np.ndarray([D, T, N, N])
    for d in range(D):
        Gamma[d] = get_gamma(A, B, pi, Obs[d]) 
        Ksi[d] = get_ksi(A, B, pi, Obs[d]) 
        
    pi1 = np.average(Gamma[:, 0], axis = 0) # axis = 0 == column sum

    temp0_a = np.sum(Ksi[:,0:T-1], axis = 1)
    num_a = np.sum(temp0_a, axis = 0) 
    temp1_a = np.sum(Gamma[:,0:T-1], axis = 1) 
    denom_a = np.sum(temp1_a, axis = 0) 
    A1 = num_a/(denom_a.reshape([N,1])) 
    
    B1 = np.zeros([N, M],dtype = 'float64') 
    temp_b = np.sum(Gamma, axis = 0) 
    denom_b = np.sum(temp_b, axis = 0)
    for k in range(M): 
        for j in range(N):
            Out = Gamma[:,:,j]*np.int64(Obs==Vk[k]) 
            B1[j, k] = np.sum(Out) 
    B1out = B1/denom_b.reshape([N,1])
    
    return A1, B1out, pi1

# Iteration 
num_iter = 100 

if restart == True:
    Atemp = A0
    Btemp = B0
    pitemp = pi0 
    Obstemp = Obs

for o in range(num_iter): 
    if np.mod(o+1, 10) == 0:
        print("Iter #%d "%(o+1))
    A1, B1, pi1 = HMEstimate(Atemp, Btemp, pitemp, Obstemp) 
    Atemp = A1 
    Btemp = B1 
    pitemp = pi1 
    A1 = 0
    B1 = 0
    pi1 = 0

Aout = Atemp 
Bout = Btemp 
piout = pitemp 

print(Aout) 
print(Bout) 
print(piout) 
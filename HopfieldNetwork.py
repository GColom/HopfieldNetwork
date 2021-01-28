#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 18:03:41 2020

@author: giulio
"""

import numpy as np

from matplotlib import pyplot as plt, animation
import multiprocessing as mp

#CONTROL PARAMETERS

#Side of the bit matrix
N = 20
#Number of spins
size = N ** 2
#Temperature
temp = 0.1
#Patterns choice, "random" or "test"
p_type = "random"
#Number of patterns
p = 10
#Corruption percentage of loaded pattern
corr = 0.4
#Dynamics, choose "Glauber" or "Metropolis"
dyn = "Glauber"

#Testing patterns
pattern1 = np.array([-1,1]*(size//2)) #Vertical stripes
pattern2 = np.array(([-1]*20 + [1]*20)*(N//2)) #Horizontal stripes
pattern3 = np.array([[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., 1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., 1., 1., 1., 1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]]).flatten()
#Letter A
#Patterns to be stored in the network
patterns = [pattern1, pattern2, pattern3] if p_type == "test" else [np.random.choice([-1,1], size = size) for i in range(p)]
nproc = 12
pool = mp.Pool(nproc)


def set_temperature(newtemp = 0):
    global temp
    temp = newtemp

def get_current_temperature():
    return temp

def generate_weights(patterns):
    return sum([np.outer(p, p) - np.identity(size) for p in patterns])/size

def generate_random_spins(N):
    return np.random.choice([-1,1], size = (N,))

def delta_H(i, J):
    return 2 * np.matmul(J, spins)[i] * spins[i]

def corrupt_pattern(pat, frac = 0.2):
    return np.array([item if np.random.random() > frac else -item for item in pat])

def parallel_metropolis(T):
    global spins
    def single_evolve(idx):
        global temp
        T = temp
        dE = delta_H(idx, J)
        if dE < 0 or np.random.random() < np.exp(-dE/T):
            spins[idx] *= -1
    
    indices = np.random.choice(a = range(size), size = nproc, replace = False)
    #print(indices)
    print("T = ", get_current_temperature(),", ", "\u03B1", " = ", len(patterns)/size, ", init pattern : " ,loaded, ", Nominal fraction of corrupted bits = ", corr)
    for spin in indices:    
        pool.apply_async(single_evolve(spin), indices)

def parallel_glauber(T):
    global spins
    def single_evolve(idx):
        global temp
        T = temp
        dE = delta_H(idx, J)
        if np.random.random() < (1 + np.exp(dE/T))**(-1):
            spins[idx] *= -1
    
    indices = np.random.choice(a = range(size), size = nproc, replace = False)
    #print(indices)
    print("T = ", get_current_temperature(),", ", "\u03B1", " = ", len(patterns)/size, ", init pattern : " ,loaded, ", Nominal fraction of corrupted bits = ", corr)
    for spin in indices:    
        pool.apply_async(single_evolve(spin), indices)

def ev_step(T):
    global spins
    idx = np.random.randint(0, size)
    dE = delta_H(idx, J)
    print("T = ", get_current_temperature(),", ", "\u03B1", " = ", len(patterns)/size, ", init pattern : " ,loaded, ", Nominal fraction of corrupted bits = ", corr)
    if dE < 0 or np.random.random() < np.exp(-dE/T):
        spins[idx] *= -1
        
def overlap(n):
    return np.dot(patterns[n], spins)/size




J = generate_weights(patterns)

loaded = np.random.choice(len(patterns))

spins = corrupt_pattern(patterns[loaded], corr)

init_overlaps = [overlap(n) for n in range(len(patterns))]

fig_overlap, ax_overlap = plt.subplots()
ax_overlap.set_title('Pattern Overlaps')
bars = ax_overlap.bar([i for i in range(len(patterns))], [overlap(i) for i in range(len(patterns))])
ax_overlap.set_ylim([-1,1])
ax_overlap.set_xlim([-0.5 ,len(patterns)])
ax_overlap.grid(True)

fig1, ax1 = plt.subplots()
im1 = plt.imshow(spins.reshape(N,N))



def update_overlap(dummy):
    global fig_overlap, ax_overlap
    for bar, ovlp in zip(bars, [overlap(i) for i in range(len(patterns))]):
        bar.set_height(ovlp)
    return [fig_overlap]

def update_network(t):
    global spins, temp
    if dyn == "Glauber":
        parallel_glauber(temp)
    elif dyn == "Metropolis":
        parallel_metropolis(temp)
    im1.set_array(spins.reshape(N,N))
    return[im1]

anim_spins = animation.FuncAnimation(fig1, update_network)
anim_overlap = animation.FuncAnimation(fig_overlap, update_overlap)
    
fig2, ax2 = plt.subplots()
weight_distribution = ax2.hist(J.flatten(), bins = np.linspace(-0.04,0.04, 18), density = True)
ax2.set_title('Couplings distribution')
ax2.set_xlim([-0.04, 0.04])
ax2.grid(True)
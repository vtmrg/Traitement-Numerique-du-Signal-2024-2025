# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:49:13 2025

@author: guyadern
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift

N=100; fo=50; Fe=1000; Te=1/Fe;
k=np.arange(0,N,1)
t=Te*k
x_k=np.sin(2*np.pi*fo*t);

#% on crée un axe des temps 'continu'
tc=Te*np.arange(0,N,Te)
xc=np.sin(2*np.pi*fo*tc)

# Une façon de visualiser le signal et le signal échantillonné en tenant compte du temps en secondes
plt.figure();
plt.stem(t,x_k,label='Signal échantilonné');
plt.plot(tc,xc,'r',label='Signal')
plt.xlabel('Temps (ms)');plt.ylabel('Signal')
plt.legend()

# Une seconde façon de visualiser le signal
plt.figure()
plt.scatter(t,x_k,c='r',label='Signal échantillonné')
plt.plot(tc,xc,label='Signal');
plt.xlabel('Temps (ms)');plt.ylabel('Signal')
plt.legend()
# Visualisation en fonction des échantillons 
plt.figure()
plt.stem(k,x_k,label='Signal échantilonné');

#%%
# On calule la TF
X_f=fft(x_k);
# axe des fréquences réduites 
x_f=np.arange(0,1,1/N);
# axe des fréquences
freq=x_f*Fe;

plt.figure()
plt.plot(freq,np.abs(X_f)/N);
plt.xlabel('Fréquence'); plt.ylabel('Module du spectre');

plt.figure()
plt.plot(x_f,np.abs(X_f)/N);
plt.xlabel('Fréquence réduite'); plt.ylabel('Module du spectre');

# si on souhaite visualiser la partie imaginaire
plt.figure()
plt.plot(freq,np.imag(X_f)/N)

#%% On va représenter la TF centrée sur 0
X_fc=fftshift(X_f)
x_fc=np.arange(-1/2,1/2,1/N)
freqc=x_fc*Fe;

plt.figure()
plt.plot(freqc,np.abs(X_fc)/N);
plt.xlabel('Fréquence'); plt.ylabel('Module du spectre');

plt.figure()
plt.plot(x_fc,np.abs(X_fc)/N);
plt.xlabel('Fréquence réduite'); plt.ylabel('Module du spectre');

#%% TFD inverse
x_k2=ifft(X_f);
plt.figure()
plt.plot(k,np.real(x_k2));
plt.xlabel('Echantillons'); plt.ylabel('Signal après TFD inverse');

plt.show()
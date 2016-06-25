#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from time import time
#import scipy as sp
from scipy.signal import butter, lfilter, freqz
#import librosa as rosa
from librosa import load
from librosa.output import write_wav as write
from threading import Thread
import matplotlib.pyplot as plt
################################################################################
#                           Functions                                          #
################################################################################

#   Low-Medium-High passes
#LOW PASS
def lPass(cutOf, sampleRate, order=5):
    nyq = 0.5 * sampleRate
    norm_cut = cutOf / nyq
    b, a = butter(order, norm_cut, btype='low', analog=False)
    return b, a
def lPass_filter(data, cutOf, sampleRate, order=5):
    b, a = lPass(cutOf, sampleRate, order=order)
    out = lfilter(b, a, data)
    return out
#HIGH PASS
def hPass(cutOf, sampleRate, order=5):
    nyq = 0.5 * sampleRate
    norm_cut = cutOf / nyq
    b, a = butter(order, norm_cut, btype='high', analog=False)
    return b, a
def hPass_filter(data, cutOf, sampleRate, order=5):
    b, a = hPass(cutOf, sampleRate, order=order)
    out = lfilter(b, a, data)
    return out
#BAND PASS
def bPass_filter(data, min_cutOf, max_cutOf, sampleRate, order=5):
    high_pass = hPass_filter(data, min_cutOf, sampleRate, order=order)
    out = lPass_filter(high_pass, max_cutOf, sampleRate, order=order)
    return out
#BEATROOT - Beat finder ~~~NOT WORKING~~~
def beatRoot(data,sampleRate):
    chunks = np.int(len(data) / sampleRate)
    #dat = np.zeros((chunks, sampleRate))
    dat = data[0*sr:(0+1)*sr]
    f = np.real(np.fft.fft(dat))
    for i in range(1,chunks):
        dat = data[i*sr:(i+1)*sr]
        f_plus = np.real(np.fft.fft(dat))
        for j in range(len(f)):
            if f_plus[j]>f[j]:
                f[j] += f_plus[j]
    return f[0:np.int(len(f)/2)]





################################################################################
#                               PRELIMINARY                                    #
################################################################################
#Direct fourier transphorm
def dft(S, sampleRate): # Signal,  sr
    #This function computes Fourier tranform and the frequencies
    tf = np.fft.fft(S)
    n = S.size
    dt = 1/sampleRate
    nu = np.fft.fftfreq(n, d=dt)
    #nu here is already normalized by n so w = nu*2pi
    #The with the original nu = wT/2pi so w = nu*2pi
    T = np.size(S)/sampleRate #Duration in seconds
    #temp = np.where(nu >= 0)
    #tf = tf[temp]
    #nu = nu[temp]
    return tf.real, nu*2*np.pi #tf\n ?

def specter_reader(tf, freq, bins=3):
    #This Function create a "partition" that can be read with other
    #programs or function.
    #It should return sevral vectors with a time dimension
    #For now it will read dft specter in
    #Lets cut the histogram in half
    half = np.int(freq.size/2)

    tf = tf[:half]



    freq = freq[:half]
    freq_max = freq.max()
    error = (freq[1:-1] - freq[:-2]).mean()

    partition = np.zeros(bins)

    for i in range(bins):
        #freq lowest in bin i
        a = np.exp(i/bins-1)*freq_max
        if i==0:
            a = 0
        #freq highest in bin i
        b = np.exp((i+1)/bins-1)*freq_max

        (fmin,) = np.where(np.fabs(a-freq) <= error/2)
        (fmax,) = np.where(np.fabs(b-freq) <= error/2)

        partition[i] = tf[fmin:fmax].mean()/(tf.max()+1)

    return partition

################################################################################
###############################1°) Partition Maker  ############################
################################################################################

def partition_maker(in_path, bins=3):
    #This is the main function 
    #it will use dft and specter reader
    t0 = time()
    print("Importing music ...")
    song, sr = load(path)

    print("Setting time ...")
    t = np.linspace(0,len(song)/sr, num=len(song))

    print("Creating spectrogram ...")
    im_per_sec = 24
    sampleL = np.int(sr/im_per_sec)
    numberSample = np.int(song.size*im_per_sec/sr)

    cuts = np.zeros((numberSample,sampleL))
    specter = np.zeros((numberSample,sampleL))
    partition = np.zeros((numberSample,bins))
    k_time = np.linspace(0,len(song)/sr, num=numberSample)
    
    print("Begening loop")


    for i in range(numberSample):
        cuts[i, :] = song[i * sampleL : (i+1) * sampleL]
        specter[i, :], freq =  dft(cuts[i, :], sr)
        partition[i, :] = specter_reader(specter[i, :], freq, bins=bins)
    print("Reshaping Partition ...")
    for i in range(3):
        partition[:, i] = np.fabs(partition[:, i])
        maximum = np.max(partition[:, i])
        partition[:, i] /= maximum 

    print(time() - t0, "sec of execution for partition_maker()")

    return partition, k_time


################################################################################
###############################2°) Display 0  ##################################
################################################################################

class Corde(Thread):
    def __init__(self, target, color):
        Thread.__init__(self)
        self.target = target
        self.color = color
    def run(self):
        self.target(self.color)


class Grid:

    def __init__(self, bins=3):
        self.dt = 0.5
        self.speed = 1
        self.x_resol = 128
        self.y_resol = 128
        self.dx = 1
        self.dy = 1
        self.colors = bins
        self.old = np.zeros((self.x_resol, self.y_resol, self.colors))
        self.current = np.zeros((self.x_resol, self.y_resol, self.colors))
        self.new = np.zeros((self.x_resol, self.y_resol, self.colors))

    def next_time(self):
        self.old = self.current
        self.current = self.new
        self.new =  np.zeros((self.x_resol, self.y_resol, self.colors))
    def wave2(self, color):
        """Quite Fast wave equation solver"""
        X = self.x_resol
        Y = self.y_resol
        for j in range(Y):
            self.new[1:X-1,j,color] = (2 * self.current[1:X-1,j,color] -
                    self.old[1:X-1,j,color]) + (self.speed*self.dt/self.dx)**2*(
                            self.current[2:,j,color] + 
                            self.current[0:X-2,j,color] -
                            self.current[1:X-1,j,color])
        for i in range(X):
            self.new[i,1:X-1,color] += (self.speed*self.dt/self.dy)**2*(
                    self.current[i,2:,color] +
                    self.current[i,0:Y-2,color] -
                    self.current[i,1:Y-1,color])
        self.current[0,:,:] = 0
        self.current[:,0,:] = 0


    def wave(self, color):
        """Very slow wave equation solver"""
        k = np.arange(self.x_resol - 2) +1
        l = np.arange(self.y_resol - 2) +1
        for i in k:
            for j in l:
                self.new[i,j,color] = ((2*self.current[i,j,color] -
                    -self.old[i,j,color]) +
                    self.speed**2 * self.dt**2 *(
                            (self.current[i+1,j,color] +
                                self.current[i-1,j,color] -
                                2*self.current[i,j,color])/self.dx**2 +
                            (self.current[i,j+1,color] +
                                self.current[i,j-1,color] -
                                2*self.current[i,j,color])/self.dy**2 ))
        self.current[0,:,:] = 0
        self.current[:,0,:] = 0

    def read(self, parti):
        '''Reads partition bins must be 3 !!'''
        filtre = np.array([0,1,2,3,4,5,4,3,2,1,0])/5
        fitre = np.outer(filtre,filtre)
        for i in range(3):
            if parti[i] > 0.15:
                self.current[self.x_resol/2-5:self.x_resol/2+6,
                    self.y_resol/2-5:self.y_resol/2+6, i] =parti[i]*filtre
                #self.current[self.x_resol/2, self.y_resol/2, i] = parti[i]
        

    
    def do_it(self, parti):
        self.read(parti)
        R = Corde(self.wave2, 0)
        B = Corde(self.wave2, 1)
        G = Corde(self.wave2, 2)
        R.start()
        B.start()
        G.start()
        R.join()
        B.join()
        G.join()
#        for c in range(self.colors):
        self.next_time()




def wave_2D(partition, k_time):
#will compute 2D wave equation
#bins can 3 (RGB) or 4 (RGBA)
    numberSample, bins = partition.shape
    if bins != 3:
        raise "Bins is not 3, not supported in this method !"
    imag = Grid(bins=bins)
    for i in range(100): #len(k_time)
        imag.do_it(partition[i, :])
        plt.imsave("out/" + '{0:0>4}'.format(i)+ ".png", imag.current)
    return 0




################################################################################
#                             Main                                             #
################################################################################

#   Path to music and output
path = "songs/Spear.mp3"
output = "out/output.wav"

#partitions = partition_maker(path)









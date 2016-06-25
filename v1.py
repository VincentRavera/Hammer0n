#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from time import time
#import scipy as sp
from scipy.signal import butter, lfilter, freqz
#import librosa as rosa
from librosa import load
from librosa.output import write_wav as write
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
#BEATROOT - Beat finder
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


###############################Main function####################################

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
    print("Begening loop")

    partition = np.zeros((numberSample,bins))

    for i in range(numberSample):
        cuts[i, :] = song[i * sampleL : (i+1) * sampleL]
        specter[i, :], freq =  dft(cuts[i, :], sr)
        partition[i, :] = specter_reader(specter[i,:], freq, bins=bins)

    print(time() - t0, "sec of execution for partition_maker()")

    return partition





################################################################################
#                             Main                                             #
################################################################################

#   Path to music and output
path = "songs/Spear.mp3"
output = "out/output.wav"

#partitions = partition_maker(path)









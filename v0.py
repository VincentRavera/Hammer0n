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

#Main function
def partition_maker(in_path):
    #This is the main function 
    #it will use dft and specter reader
    print("Importing music ...")
    song, sr = load(path)

    print("Setting time ...")
    t = np.linspace(0,len(song)/sr, num=len(song))

    print("Creating spectrogram ...")
    im_per_sec = 24
    sampleL = np.int(sr/im_per_sec)
    numberSample = np.int(y.size*im_per_sec/sr)

    cuts = np.zeros((numberSample,sampleL))
    specter = np.zeros((numberSample,sampleL))
    print("Begening loop")
    t0 = time()

    for i in range(numberSample):
        cuts[i, :] = y[i * sampleL : (i+1) * sampleL]
        specter[i, :], freq =  dft(cuts[i, :], sr)
    print("Loop was", time() - t0,"sec long" )

    return 0

def specter_reader(tf, freq):
    #This Function create a "partition" that can be read with other
    #programs or function.
    #It should return sevral vectors with a time dimension
    #For now it will read dft specter in
    return 0




################################################################################
#                             Main                                             #
################################################################################

#   Path to music and output
path = "songs/Spear.mp3"
output = "out/output.wav"


#   Load music
#let y be the amplitude of the signal
#let sr be the sample rate as y[0:sr] last 1s
print("Importing music ...")
y, sr = load(path)


#   Time
print("Setting time")
t = np.linspace(0,len(y)/sr, num=len(y))

# 10 seconds
#t = t[:10*sr]
#y = y[:10*sr]

im_per_sec = 15
sampleL = np.int(sr/im_per_sec)
numberSample = np.int(y.size*im_per_sec/sr)

cuts = np.zeros((numberSample,sampleL))
specter = np.zeros((numberSample,sampleL))

print("Begening loop")
t0 = time()

for i in range(numberSample):
    cuts[i, :] = y[i * sampleL : (i+1) * sampleL]
    specter[i, :], freq =  dft(cuts[i, :], sr)
    #plt.plot(freq, specter[i, :])
    #plt.show()
print("Loop was", time() - t0,"sec long" )
#   3bins freq





################################################################################
#                             Test                                             #
################################################################################
#   Low pass parameter
#order = 6
#cutOf_low = 100. #Hz
#cutOf_high = 7000. #Hz

#test 100s - 1m40s

#y = y[:sr*1]
#t = t[:sr*1]

#low = lPass_filter(y, cutOf_low, sr, order=order)
#high = hPass_filter(y, cutOf_high, sr, order=order)
# plots


#   Find beat and tempo
#print("Finding beats ...")
#tempo, beat_frames = rosa.beat.beat_track(y=y, sr=sr)

#   Separate Percussive and harmonics of tracks
#print("Separtion of harmonic and percussion ...")
#y_harmonic, y_percussive = rosa.effects.hpss(y)

#   Visual Display Conversion
#visual_cst = 441                   #let visual_cst be a natural divider of sr
#real_time = np.arange(len(y))/sr
#visual_time = real_time[0:-1:visual_cst]
#visual_y = y[:-1:visual_cst]
#plt.plot(visual_time, visual_y)


#   Save file as .wav
#output = "out/output.wav"
#rosa.output.write_wav(out, y, sr, norm=True)






import numpy as np
import math
from random import shuffle
from scipy.fftpack import fft, ifft, fftshift

########################################################################################################################
# Data Functions
########################################################################################################################
def toFidsAlt(specs, axis=1):
    '''
    Convert to Fids (time domain) with numpy
    :param specs: [numSamples, specPoints]
           axis: *provided in case specs axes are swapped*
    :return: fids: [numSamples, specPoints]
    '''
    return np.fft.fft(np.fft.fftshift(specs, axes=axis), axis=axis)

def toFids(specs, axis=1):
    '''
    Convert to Fids (time domain)
    :param specs: [numSamples, specPoints]
           axis: *provided in case specs axes are swapped*
    :return: fids: [numSamples, specPoints]
    '''
    return ifft(fftshift(specs, axes=axis), axis=axis)

def toSpecsAlt(fids, axis=1):
    '''
    Convert to Specs (frequency domain) with numpy
    :param fids: [numSamples, specPoints]
           axis: *provided in case fids axes are swapped*
    :return: specs: [numSamples, specPoints]
    '''
    return np.fft.fftshift(np.fft.ifft(fids, axis=axis), axes=axis)

def toSpecs(fids, axis=1):
    '''
    Convert to Specs (frequency domain)
    :param fids: [numSamples, specPoints]
           axis: *provided in case fids axes are swapped*
    :return: specs: [numSamples, specPoints]
    '''
    return fftshift(fft(fids, axis=axis), axes=axis)

def simScans(direc, numTrans):
    '''
    Load simulated GT Specs (frequency domain)
    :param direc: string (directory for ground truths)
           waterType: string
           numTrans: integer
    :return: fids: fids for one subspectrum [numSamples(=numGTs * numTrans), specPoints]
    '''
    gt = np.load(direc)
    fids = np.repeat(gt, numTrans, axis=0)
    return fids

def addComplexNoise(fids, noiseStd):
    '''
    Add complex noise to fids (distinct distributions to real and imaginary)
    :param fids: [numSamples, specPoints]
           noiseStd: integer (0: low noise (~SNR10), 1: med noise (~SNR5), 2: lots of noise (~SNR2.5))
    :return: fids: [numSamples, specPoints] **with amplitude noise**
    '''
    normNoise = [np.random.uniform(8, 9.5, size=1), np.random.uniform(4, 4.5, size=1), np.random.uniform(2, 2.5, size=1)]
    fids = (fids.real + np.random.normal(0, normNoise[noiseStd], size=(fids.shape[0], fids.shape[1]))) + \
           (fids.imag + np.random.normal(0, normNoise[noiseStd], size=(fids.shape[0], fids.shape[1]))) * 1j
    return fids

def addFShift(fids, time, shiftRange):
    '''
    Add frequency shifts to fids
    :param fids: [numSamples, specPoints]
           time: float array [specPoints]
           shiftRange: integer for range of frequency shifts +/ range
    :return: fids: [numSamples, specPoints] **with frequency shifts**
    '''
    time = np.repeat(time[np.newaxis,:], fids.shape[0], axis=0)
    Fnoise = np.random.uniform(low=-shiftRange, high=shiftRange, size=(fids.shape[0], 1)).repeat(fids.shape[1], axis=1)
    fids = fids * np.exp(-1j * time * Fnoise * 2 * math.pi)
    return fids, Fnoise[:, 0]

def addPShift(fids, shiftRange):
    '''
    Add phase shifts to fids
    :param fids: [numSamples, specPoints]
           shiftRange: integer for range of phase shifts +/- range
    :return: fids: [numSamples, specPoints] **with phase shifts**
    '''
    PNoise = np.random.uniform(low=-shiftRange, high=shiftRange, size=(fids.shape[0], 1)).repeat(fids.shape[1], axis=1)
    fids = fids * np.exp(-1j * PNoise * math.pi / 180)
    return fids, PNoise[:, 0]

def corrFShift(fids, time, Fshifts):
    '''
    Correct frequency shifts to fids
    :param fids: [numSamples, specPoints]
           time: float array [specPoints]
           Fshifts: float array with shifts (negate for corrections) [numSamples]
    :return: fids: [numSamples, specPoints]   **frequency corrected**
    '''
    Fnoise = np.repeat(Fshifts[:, np.newaxis], fids.shape[1], axis=1)
    time = np.repeat(time[np.newaxis,:], fids.shape[0], axis=0)
    fids = fids * np.exp(-1j * time * -Fnoise * 2 * math.pi)
    return fids

def corrPShift(fids, Pshifts):
    '''
    Correct phase shifts to fids
    :param fids: [numSamples, specPoints]
           Pshifts: float array with shifts (negate for corrections) [numSamples]
    :return: fids: [numSamples, specPoints]   **phase corrected**
    '''
    Pnoise = np.repeat(Pshifts[:, np.newaxis], fids.shape[1], axis=1)
    fids = fids * np.exp(-1j * -Pnoise * math.pi/180)
    return fids

def loadModelPreds(simDir, snr, water, model):
    phaseLbs = np.load(f"{simDir}Predictions/PredLabels_Sim{snr}_{water}_{model}_phaseModel.npy")
    freqLbs = np.load(f"{simDir}Predictions/PredLabels_Sim{snr}_{water}_{model}_freqModel.npy")
    return phaseLbs, freqLbs

def reformScans(specs, numScans=36):
    '''
    Recombine transients into their original scans
    :param specs: [numSamples, specPoints]
    :return: specs: [numScans, numSubSpec, numTransients, specPoints]
    '''
    On, Off = specs[:int(specs.shape[0]/2), :], specs[int(specs.shape[0]/2):, :]
    On = np.reshape(On, (numScans, 1, int(On.shape[0]/numScans), -1))
    Off = np.reshape(Off, (numScans, 1, int(Off.shape[0]/numScans), -1))
    return np.concatenate((Off, On), axis=1)

def meanSpec(specs):
    '''
    Calculate the mean spectrum per scan
    :param: specs: [numScans, numSubSpec, numTransients, specPoints]
    :return: specs: [numScans, specPoints]
    '''
    return (specs[:, 1, :, :] - specs[:, 0, :, :]).mean(axis=1)

def normSpecs(specs):
    '''
    Normalize specs using max value
    :param Specs: [numSamples, specPoints]
    :return: Specs: [numSamples, specPoints]  **normalized by maximum value per spectrum**
    '''
    return specs/np.percentile(np.abs(specs), 100, axis=1, keepdims=True)

def divideDev(On, Off, percent):
    '''
    Split development set into train and validation sets
    :param ON: [numSamples, specPoints]
           OFF: [numSamples, specPoints]
    :return: train: [numSamples*percent, specPoints]  **first in list**
             val: [numSamples*percent, specPoints] **last in list**
    '''
    if On.ndim==2:
        train = np.concatenate((On[:int(percent * On.shape[0]), :], Off[:int(percent * Off.shape[0]), :]), axis=0)
        val = np.concatenate((On[int(percent * On.shape[0]):, :], Off[int(percent * Off.shape[0]):, :]), axis=0)
    else:   #assumes ndim=1
        train = np.concatenate((On[:int(percent * On.shape[0])], Off[:int(percent * Off.shape[0])]), axis=0)[np.newaxis, :]
        val = np.concatenate((On[int(percent * On.shape[0]):], Off[int(percent * Off.shape[0]):]), axis=0)[np.newaxis, :]
    return train, val

def shuffleData(specs, specsFc, freqLabels, phaseLabels):
    '''
    Shuffle data order
    :param Specs: [numSamples, specPoints]
           SpecsFC: (frequency corrected) [numSamples, specPoints]
           FreqLabels: [numSubSpec, numSamples]
           PhaseLabels: [numSubSpec, numSamples]
    :return: Specs: [numSamples, specPoints]    **shuffled**
             SpecsFC: (frequency corrected) [numSamples, specPoints] **shuffled**
             FreqLabels: [numSubSpec, numSamples] **shuffled**
             PhaseLabels: [numSubSpec, numSamples] **shuffled**
    '''
    index_shuf = list(range(specs.shape[0]))
    shuffle(index_shuf)
    specs, specsFc = specs[index_shuf, :], specsFc[index_shuf, :]
    freqLabels, phaseLabels = freqLabels[:, index_shuf], phaseLabels[:, index_shuf]
    return specs, specsFc, freqLabels, phaseLabels

def getMag(specs):
    '''
    get magnitude of specs
    :param Specs: [numSamples, specPoints]
    :return: Specs: [numChanns(1), numSamples, specPoints]    **1 channel magnitude only**
    '''
    return np.abs(specs)[np.newaxis, :, :]

def getComp(specs):
    '''
    separate complex data into two channels (first real, second imaginary)
    :param Specs: [numSamples, specPoints]
    :return: Specs: [numChanns(2: 0 is real and 1 is imag), numSamples, specPoints] **2 channel complex only**
    '''
    twoChanComp = np.empty((2, specs.shape[0], specs.shape[1]))
    twoChanComp[0, :, :], twoChanComp[1, :, :] = specs.real, specs.imag
    return twoChanComp

def getReal(specs):
    '''
    keep only real values in first channel
    :param specs: [numSamples, specPoints]
    :return: specs: [numChanns(1), numSamples, specPoints]  **1 channel real only**
    '''
    return (getComp(specs)[0, :, :])[np.newaxis, :, :]

def window1024(specs, ppm):
    '''
    select window of 1024 points in SIMULATED specs
    :param specs: [numChanns, numSamples, specPoints**2048**]
           ppm: [specPoints]
    :return: specs: [numChanns, numSamples, specPoints**1024**]
    '''
    start, finish = np.where(ppm <= 0.01)[0][-1], np.where(ppm >= 7.83)[0][0]   # swap start and finish and [-1] and [0] if flipped ppm
    return specs[:, :, start:finish]

def window1024Trim(specs, ppm):
    '''
    select window of 1024 points in SIMULATED specs
    :param specs: [numChanns, numSamples, specPoints**2048**]
           ppm: [specPoints]
    :return: specs: [numChanns, numSamples, specPoints**1024**]
             ppm: [specPoints**1024**]
    '''
    start, finish = np.where(ppm <= 0.01)[0][-1], np.where(ppm >= 7.83)[0][0]
    return specs[:, start:finish], ppm[start:finish]

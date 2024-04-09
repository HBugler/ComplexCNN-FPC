import numpy as np
import math
from random import shuffle
import matlab.engine
import scipy.stats as stats
from metric_calculator import calculate_snr, calculate_NewLW, calculate_linewidth, calculate_ModelledLW
########################################################################################################################
# Data Functions
########################################################################################################################
def loadSimGTs(waterType):
    '''
    Load simulated GT fids

    :param waterType: string
    :return: ON_GTFinal: fids for 'ON' subspectrum [numGTs, specPoints]
             OFF_GTFinal: fids for 'OFF' subspectrum [numGTs, specPoints]
    '''

    simFolder = "C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/GTs/"

    if waterType== "Mix":
        ON_GT = np.load(f"{simFolder}Development/Water{waterType}/fidsON_{waterType}W_GABAPlus_DevSet.npy")
        OFF_GT = np.load(f"{simFolder}Development/Water{waterType}/fidsOFF_{waterType}W_GABAPlus_DevSet.npy")
    else:
        ON_GT = np.loadtxt(f"{simFolder}Development/Water{waterType}/fidsON_{waterType}W_GABAPlus_DevSet.csv", dtype=complex, delimiter=",")
        OFF_GT = np.loadtxt(f"{simFolder}Development/Water{waterType}/fidsOFF_{waterType}W_GABAPlus_DevSet.csv", dtype=complex, delimiter=",")
    ON_GTFinal = np.swapaxes(ON_GT, axis1=1, axis2=0)
    OFF_GTFinal = np.swapaxes(OFF_GT, axis1=1, axis2=0)
    return ON_GTFinal, OFF_GTFinal

def loadVivoGTs():
    '''
    Load simulated GT fids

    :param waterType: string
    :return: all_specs_ON: fids for 'ON' subspectrum [numScans*160Trans, specPoints]
             all_specs_OFF: fids for 'OFF' subspectrum [numScans*160Trans, specPoints]
    '''

    InVivoGTs_dir = "C:/Users/Hanna B/FID-A/"
    eng1 = matlab.engine.start_matlab()
    G5_fids_ON, G5_fids_OFF = np.empty((2048, 160 * 12), dtype=complex), np.empty((2048, 160 * 12), dtype=complex)
    G7_fids_ON, G7_fids_OFF = np.empty((2048, 160 * 12), dtype=complex), np.empty((2048, 160 * 12), dtype=complex)
    G8_fids_ON, G8_fids_OFF = np.empty((2048, 160 * 12), dtype=complex), np.empty((2048, 160 * 12), dtype=complex)
    indTrans = 160
    for i in range(1, 13):
        if i < 10:
            # (2048, 320)
            GABA_fids5 = np.array(eng1.load(f"{InVivoGTs_dir}G5_S0{i}_GABA_68.mat")['fids'])
            GABA_fids7 = np.array(eng1.load(f"{InVivoGTs_dir}G7_S0{i}_GABA_68.mat")['fids'])
            GABA_fids8 = np.array(eng1.load(f"{InVivoGTs_dir}G8_S0{i}_GABA_68.mat")['fids'])
        else:
            GABA_fids5 = np.array(eng1.load(f"{InVivoGTs_dir}G5_S{i}_GABA_68.mat")['fids'])
            GABA_fids7 = np.array(eng1.load(f"{InVivoGTs_dir}G7_S{i}_GABA_68.mat")['fids'])
            GABA_fids8 = np.array(eng1.load(f"{InVivoGTs_dir}G8_S{i}_GABA_68.mat")['fids'])

        G5_fids_ON[:, indTrans * (i - 1):indTrans * (i)] = GABA_fids5[:, :, 0]
        G7_fids_ON[:, indTrans * (i - 1):indTrans * (i)] = GABA_fids7[:, :, 0]
        G8_fids_ON[:, indTrans * (i - 1):indTrans * (i)] = GABA_fids8[:, :, 0]
        G5_fids_OFF[:, indTrans * (i - 1):indTrans * (i)] = GABA_fids5[:, :, 1]
        G7_fids_OFF[:, indTrans * (i - 1):indTrans * (i)] = GABA_fids7[:, :, 1]
        G8_fids_OFF[:, indTrans * (i - 1):indTrans * (i)] = GABA_fids8[:, :, 1]

    all_specs_ON = np.concatenate((toSpecs(G5_fids_ON, 0), toSpecs(G7_fids_ON, 0), toSpecs(G8_fids_ON, 0)), axis=1)
    all_specs_OFF = np.concatenate((toSpecs(G5_fids_OFF, 0), toSpecs(G7_fids_OFF, 0), toSpecs(G8_fids_OFF, 0)), axis=1)
    all_specs_ON = np.swapaxes(all_specs_ON, 0, 1)
    all_specs_OFF = np.swapaxes(all_specs_OFF, 0, 1)
    np.save("allSpecsInVivoON_NoOffsets.npy", all_specs_ON)
    np.save("allSpecsInVivoOFF_NoOffsets.npy", all_specs_OFF)

    return all_specs_ON, all_specs_OFF

def loadVivo(subDir, fileExt, modelName, type):
    '''
    Load predictions, ground truth labels and specs to correct (in vivo)

    :param subDir: string (referring to sub directory)
           fileExt: string (referring to file extension)
           modelName: string (referring to model which made the predictions (default to model[0] for branches not requiring model))
           type: string (referring to type of information/data importing (specs, predictions, ground truth labels))
    :return: freqNone: *data type dependent on param 'type' with no offsets
             phaseNone: *data type dependent on param 'type' with no offsets
             freqSmall: *data type dependent on param 'type' with Small offsets
             phaseSmall: *data type dependent on param 'type' with Small offsets
             freqMed: *data type dependent on param 'type' with Med offsets
             phaseMed: *data type dependent on param 'type' with Med offsets
             freqLarge: *data type dependent on param 'type' with Large offsets
             phaseLarge: *data type dependent on param 'type' with Large offsets
    '''

    if type=="specs":
        # freq -> ON, phase -> OFF
        # freqNone = np.load(f"{subDir}Corrupt/allSpecsInVivoON_NoOffsets.npy")
        # phaseNone = np.load(f"{subDir}Corrupt/allSpecsInVivoOFF_NoOffsets.npy")
        freqSmall = np.load(f"{subDir}Corrupt/allSpecsInVivoON_SmallOffsets.npy")
        phaseSmall = np.load(f"{subDir}Corrupt/allSpecsInVivoOFF_SmallOffsets.npy")
        freqMed = np.load(f"{subDir}Corrupt/allSpecsInVivoON_MediumOffsets.npy")
        phaseMed = np.load(f"{subDir}Corrupt/allSpecsInVivoOFF_MediumOffsets.npy")
        freqLarge = np.load(f"{subDir}Corrupt/allSpecsInVivoON_LargeOffsets.npy")
        phaseLarge = np.load(f"{subDir}Corrupt/allSpecsInVivoOFF_LargeOffsets.npy")
    elif type=="pred":
        # freqNone = np.load(f"{subDir}Predictions/PredLabels_freq_InVivo_{modelName}{fileExt}NONE.npy")
        # phaseNone = np.load(f"{subDir}Predictions/PredLabels_phase_InVivo_{modelName}{fileExt}NONE.npy")
        freqSmall = np.load(f"{subDir}Predictions/PredLabels_freq_InVivo_{modelName}{fileExt}Small.npy")
        phaseSmall = np.load(f"{subDir}Predictions/PredLabels_phase_InVivo_{modelName}{fileExt}Small.npy")
        freqMed = np.load(f"{subDir}Predictions/PredLabels_freq_InVivo_{modelName}{fileExt}Medium.npy")
        phaseMed = np.load(f"{subDir}Predictions/PredLabels_phase_InVivo_{modelName}{fileExt}Medium.npy")
        freqLarge = np.load(f"{subDir}Predictions/PredLabels_freq_InVivo_{modelName}{fileExt}Large.npy")
        phaseLarge = np.load(f"{subDir}Predictions/PredLabels_phase_InVivo_{modelName}{fileExt}Large.npy")
    else:
        # freqNone = np.load(f"{subDir}Corrupt/FnoiseInVivo_NoOffsets.npy")
        # phaseNone = np.load(f"{subDir}Corrupt/PnoiseInVivo_NoOffsets.npy")
        freqSmall = np.load(f"{subDir}Corrupt/FnoiseInVivo_SmallOffsets.npy")
        phaseSmall = np.load(f"{subDir}Corrupt/PnoiseInVivo_SmallOffsets.npy")
        freqMed = np.load(f"{subDir}Corrupt/FnoiseInVivo_MediumOffsets.npy")
        phaseMed = np.load(f"{subDir}Corrupt/PnoiseInVivo_MediumOffsets.npy")
        freqLarge = np.load(f"{subDir}Corrupt/FnoiseInVivo_LargeOffsets.npy")
        phaseLarge = np.load(f"{subDir}Corrupt/PnoiseInVivo_LargeOffsets.npy")
        # freqNone = np.concatenate((freqNone[1, :], freqNone[0, :]))
        # phaseNone = np.concatenate((phaseNone[1, :], phaseNone[0, :]))
        freqSmall = np.concatenate((freqSmall[1, :], freqSmall[0, :]))
        phaseSmall = np.concatenate((phaseSmall[1, :], phaseSmall[0, :]))
        freqMed = np.concatenate((freqMed[1, :], freqMed[0, :]))
        phaseMed = np.concatenate((phaseMed[1, :], phaseMed[0, :]))
        freqLarge = np.concatenate((freqLarge[1, :], freqLarge[0, :]))
        phaseLarge = np.concatenate((phaseLarge[1, :], phaseLarge[0, :]))
    freqNone = np.load(f"{subDir}Corrupt/FnoiseInVivo_NoOffsets.npy")
    phaseNone = np.load(f"{subDir}Corrupt/PnoiseInVivo_NoOffsets.npy")
    return freqNone, phaseNone, freqSmall, phaseSmall, freqMed, phaseMed, freqLarge, phaseLarge

def simScans(waterType, numTrans):
    '''
    Load simulated GT Specs (frequency domain)

    :param waterType: string
           numTrans: integer
    :return: ONfids: fids for 'ON' subspectrum [numSamples(=numGTs * numTrans), specPoints]
             OFFfids: fids for 'OFF' subspectrum [numSamples(=numGTs * numTrans), specPoints]
    '''

    ON_GT, OFF_GT = loadSimGTs(waterType)
    ONfids = np.repeat(ON_GT, numTrans, axis=0)
    OFFfids = np.repeat(OFF_GT, numTrans, axis=0)
    return ONfids, OFFfids

def toFids(specs, axis):
    '''
    Convert to Fids (time domain)

    :param specs: [numSamples, specPoints]
           axis: *provided in case specs axes are swapped*
    :return: fids: [numSamples, specPoints]
    '''

    return np.fft.fft(np.fft.fftshift(specs, axes=axis), axis=axis)

def toSpecs(fids, axis):
    '''
    Convert to Specs (frequency domain)

    :param fids: [numSamples, specPoints]
           axis: *provided in case fids axes are swapped*
    :return: specs: [numSamples, specPoints]
    '''

    return np.fft.fftshift(np.fft.ifft(fids, axis=axis), axes=axis)

def flipSpecs(specs):
    '''
    Flip spectrum array

    :param specs: [numSamples, specPoints]
    :return: specs: [numSamples, stnioPceps]
    '''

    return specs[:, ::-1]

def addComplexNoise(ONfids, OFFfids, noiseStd):
    '''
    Add complex noise to fids (distinct distributions to real and imaginary)

    :param ONfids: [numSamples, specPoints]
           OFFfids: [numSamples, specPoints]
           noiseStd: integer
    :return: ONfids: [numSamples, specPoints] **with amplitude noise**
             OFFfids: [numSamples, specPoints] **with amplitude noise**
    '''

    ONfids = (ONfids.real + np.random.normal(0, noiseStd, size=(ONfids.shape[0], ONfids.shape[1]))) + \
           (ONfids.imag + np.random.normal(0, noiseStd, size=(ONfids.shape[0], ONfids.shape[1]))) * 1j
    OFFfids = (OFFfids.real + np.random.normal(0, noiseStd, size=(OFFfids.shape[0], OFFfids.shape[1]))) + \
           (OFFfids.imag + np.random.normal(0, noiseStd, size=(OFFfids.shape[0], OFFfids.shape[1]))) * 1j
    return ONfids, OFFfids

def addFShift(ONfids, OFFfids, time, shiftRange):
    '''
    Add frequency shifts to fids

    :param ONfids: [numSamples, specPoints]
           OFFfids: [numSamples, specPoints]
           time: float array [numSamples, specPoints]
           shiftRange: integer for range of frequency shifts +/ range
    :return: ONfids: [numSamples, specPoints] **with frequency shifts**
             OFFfids: [numSamples, specPoints] **with frequency shifts**
    '''

    # Fnoise = np.random.uniform(low=-shiftRange, high=shiftRange, size=(2, ONfids.shape[0], 1)).repeat(ONfids.shape[1], axis=2)
    Fnoise1 = np.random.uniform(low=10, high=20, size=(2, int(ONfids.shape[0]/2), 1)).repeat(ONfids.shape[1], axis=2)
    Fnoise2 = np.random.uniform(low=-20, high=-10, size=(2, int(ONfids.shape[0]/2), 1)).repeat(ONfids.shape[1], axis=2)
    Fnoise = np.concatenate((Fnoise1, Fnoise2), axis=1)
    ONfids = ONfids * np.exp(-1j * time * Fnoise[1, :, :] * 2 * math.pi)
    OFFfids = OFFfids * np.exp(-1j * time * Fnoise[0, :, :] * 2 * math.pi)
    return ONfids, OFFfids, Fnoise[:, :, 0]

def addPShift(ONfids, OFFfids, shiftRange):
    '''
    Add phase shifts to fids

    :param ONfids: [numSamples, specPoints]
           OFFfids: [numSamples, specPoints]
           shiftRange: integer for range of phase shifts +/ range
    :return: ONfids: [numSamples, specPoints] **with phase shifts**
             OFFfids: [numSamples, specPoints] **with phase shifts**
    '''

    # PNoise = np.random.uniform(low=-shiftRange, high=shiftRange, size=(2, ONfids.shape[0], 1)).repeat(ONfids.shape[1], axis=2)
    PNoise1 = np.random.uniform(low=45, high=90, size=(2, int(ONfids.shape[0]/2), 1)).repeat(ONfids.shape[1], axis=2)
    PNoise2 = np.random.uniform(low=-90, high=-45, size=(2, int(ONfids.shape[0]/2), 1)).repeat(ONfids.shape[1], axis=2)
    PNoise = np.concatenate((PNoise1, PNoise2), axis=1)
    ONfids = ONfids * np.exp(-1j * PNoise[1, :, :] * math.pi / 180)
    OFFfids = OFFfids * np.exp(-1j * PNoise[0, :, :] * math.pi / 180)
    return ONfids, OFFfids, PNoise[:, :, 0]

def corrFShift(fids, time, Fnoise):
    '''
    Correct frequency shifts to fids

    :param fids: [numSamples, specPoints]   **subspectra concatenated on axis 0**
           time: float array [specPoints]
           Fnoise: float array with shifts (negate for corrections) [numSamples]
    :return: fids: [numSamples, specPoints]   **frequency corrected**
    '''

    Fnoise = np.repeat(Fnoise[:, np.newaxis], fids.shape[1], axis=1)
    time = np.repeat(time[np.newaxis,:], fids.shape[0], axis=0)
    fids = fids * np.exp(-1j * time * -Fnoise * 2 * math.pi)
    return fids

def corrPShift(fids, Pnoise):
    '''
    Correct phase shifts to fids

    :param fids: [numSamples, specPoints]   **subspectra concatenated on axis 0**
           Pnoise: float array with shifts (negate for corrections) [numSamples]
    :return: fids: [numSamples, specPoints]   **phase corrected**
    '''

    Pnoise = np.repeat(Pnoise[:, np.newaxis], fids.shape[1], axis=1)
    fids = fids * np.exp(-1j * -Pnoise * math.pi/180)
    return fids

def reformScans(specsSmall, specsMed, specsLarge):
    '''
    Recombine transients into their original scans

    :param specsSmall: [numSamples, specPoints]
           specsMed: [numSamples, specPoints]
           specsLarge: [numSamples, specPoints]
    :return: specsSmall: [numScans, numSubSpec, numTransients, specPoints]
           specsMed: [numScans, numSubSpec, numTransients, specPoints]
           specsLarge: [numScans, numSubSpec, numTransients, specPoints]
    '''

    specsSmall_scans, specsMed_scans, specsLarge_scans = np.empty(
        shape=(int(specsSmall.shape[0] / 320), 2, 160, 2048), dtype=complex), np.empty(
        shape=(int(specsMed.shape[0] / 320), 2, 160, 2048), dtype=complex), np.empty(
        shape=(int(specsLarge.shape[0] / 320), 2, 160, 2048), dtype=complex)
    half = int(specsSmall.shape[0]/2)
    for k in range(0, 36):
        specsSmall_scans[k, 1, :, :] = specsSmall[160*k : 160*(k+1), :]
        specsSmall_scans[k, 0, :, :] = specsSmall[half+(160*k) : half+(160*(k+1)), :]
        specsMed_scans[k, 1, :, :] = specsMed[160*k : 160*(k+1), :]
        specsMed_scans[k, 0, :, :] = specsMed[half+(160*k) : half+(160*(k+1)), :]
        specsLarge_scans[k, 1, :, :] = specsLarge[160*k : 160*(k+1), :]
        specsLarge_scans[k, 0, :, :] = specsLarge[half+(160*k) : half+(160*(k+1)), :]
    return specsSmall_scans, specsMed_scans, specsLarge_scans

def meanSpec(specsSmall, specsMed, specsLarge):
    '''
    Calculate the mean spectrum per scan

    :param: specsSmall: [numScans, numSubSpec, numTransients, specPoints]
            specsMed: [numScans, numSubSpec, numTransients, specPoints]
            specsLarge: [numScans, numSubSpec, numTransients, specPoints]
    :return: specsSmall: [numScans, specPoints]
             specsMed: [numScans, specPoints]
             specsLarge: [numScans, specPoints]
    '''

    smallScans = (specsSmall[:, 1, :, :] - specsSmall[:, 0, :, :]).mean(axis=1)
    medScans = (specsMed[:, 1, :, :] - specsMed[:, 0, :, :]).mean(axis=1)
    largeScans = (specsLarge[:, 1, :, :] - specsLarge[:, 0, :, :]).mean(axis=1)
    return smallScans, medScans, largeScans

def getMetricsSignificance(specsSmall1, specsSmall2, specsSmall3, ppm, setName, sizeName):
    '''
    Calculate SNR and LW and determine their statistical significance compared to other model metric outputs

    :param: specsSmall1: [numScans, specPoints]
            specsSmall2: [numScans, specPoints]
            specsSmall3: [numScans, specPoints]
            ppm: [specPoints]
            setName: string (model name)
            sizeName: string (offset size name)
    '''
    # find appropriate linewidth function
    allSnrSmall1, meanSnrSmall1, stdSnrSmall1 = calculate_snr(specsSmall1, ppm)
    allLwSmall1, meanLwSmall1, stdLwSmall1 = calculate_linewidth(specsSmall1, ppm)
    allSnrSmall2, meanSnrSmall2, stdSnrSmall2 = calculate_snr(specsSmall2, ppm)
    allLwSmall2, meanLwSmall2, stdLwSmall2 = calculate_linewidth(specsSmall2, ppm)
    allSnrSmall3, meanSnrSmall3, stdSnrSmall3 = calculate_snr(specsSmall3, ppm)
    allLwSmall3, meanLwSmall3, stdLwSmall3 = calculate_linewidth(specsSmall3, ppm)
    snrStat12, snrPvalue12 = stats.wilcoxon(allSnrSmall1, allSnrSmall2)
    snrStat13, snrPvalue13 = stats.wilcoxon(allSnrSmall1, allSnrSmall3)
    snrStat23, snrPvalue23 = stats.wilcoxon(allSnrSmall2, allSnrSmall3)
    lwStat12, lwPvalue12 = stats.wilcoxon(allLwSmall1, allLwSmall2)
    lwStat13, lwPvalue13 = stats.wilcoxon(allLwSmall1, allLwSmall3)
    lwStat23, lwPvalue23 = stats.wilcoxon(allLwSmall2, allLwSmall3)
    print(f'{setName[0]} {sizeName} SNR: {meanSnrSmall1} +/- {stdSnrSmall1}')
    print(f'{setName[0]} {sizeName} LW: {meanLwSmall1} +/- {stdLwSmall1}')
    print(f'{setName[1]} {sizeName} SNR: {meanSnrSmall2} +/- {stdSnrSmall2}')
    print(f'{setName[1]} {sizeName} LW: {meanLwSmall2} +/- {stdLwSmall2}')
    print(f'{setName[2]} {sizeName} SNR: {meanSnrSmall3} +/- {stdSnrSmall3}')
    print(f'{setName[2]} {sizeName} LW: {meanLwSmall3} +/- {stdLwSmall3}')
    print(f'SNR pvalue {sizeName} {setName[0]} compared to {setName[1]} {snrPvalue12}')
    print(f'SNR pvalue {sizeName} {setName[0]} compared to {setName[2]} {snrPvalue13}')
    print(f'SNR pvalue {sizeName} {setName[1]} compared to {setName[2]} {snrPvalue23}')
    print(f'LW pvalue {sizeName} {setName[0]} compared to {setName[1]} {lwPvalue12}')
    print(f'LW pvalue {sizeName} {setName[0]} compared to {setName[2]} {lwPvalue13}')
    print(f'LW pvalue {sizeName} {setName[1]} compared to {setName[2]} {lwPvalue23}')

    return meanSnrSmall1, stdSnrSmall1, meanLwSmall1, stdLwSmall1, meanSnrSmall2, stdSnrSmall2, \
           meanLwSmall2, stdLwSmall2, meanSnrSmall3, stdSnrSmall3, meanLwSmall3, stdLwSmall3

def saveSpecs(ONfids_devN, ONfids_devFC, OFFfids_devN, OFFfids_devFC, Fnoise_dev, Pnoise_dev, snrTypes, waterTypes, termi):
    '''
    Save Simulated Data

    :param ONfids_devN: [numSamples, specPoints]
           ONfids_devFC: [numSamples, specPoints]
           OFFfids_devN: [numSamples, specPoints]
           OFFfids_devFC: [numSamples, specPoints]
           Fnoise_dev: [numSubSpec (ON=1, OFF=0), numSamples]
           Pnoise_dev: [numSubSpec (ON=1, OFF=0), numSamples]
           snrTypes: string
           waterTypes: string
           termi: string
    '''

    np.save(f"TruePhaseLabels_Sim{snrTypes}_{waterTypes}_{termi}.npy", Pnoise_dev)
    np.save(f"TrueFreqLabels_Sim{snrTypes}_{waterTypes}_{termi}.npy", Fnoise_dev)
    np.save(f"ON_AllSpecs_Sim{snrTypes}_{waterTypes}_{termi}.npy", ONfids_devN)
    np.save(f"OFF_AllSpecs_Sim{snrTypes}_{waterTypes}_{termi}.npy", OFFfids_devN)
    np.save(f"ON_AllSpecsFC_Sim{snrTypes}_{waterTypes}_{termi}.npy", ONfids_devFC)
    np.save(f"OFF_AllSpecsFC_Sim{snrTypes}_{waterTypes}_{termi}.npy", OFFfids_devFC)

def saveVIVOSpecs(ON, ON_FC, OFF, OFF_FC, Fnoise, Pnoise, size):
    '''
    Save Simulated Data

    :param ON: [numSamples, specPoints]
           ON_FC: [numSamples, specPoints]
           OFF: [numSamples, specPoints]
           OFF_FC: [numSamples, specPoints]
           Fnoise: [numSubSpec (ON=1, OFF=0), numSamples]
           Pnoise: [numSubSpec (ON=1, OFF=0), numSamples]
           termi: string
    '''

    np.save(f"allSpecsInVivoON_{size}Offsets.npy", ON)
    np.save(f"allSpecsInVivoOFF_{size}Offsets.npy", OFF)
    np.save(f"allSpecsInVivoONFC_{size}Offsets.npy", ON_FC)
    np.save(f"allSpecsInVivoOFFFC_{size}Offsets.npy", OFF_FC)
    np.save(f"FnoiseInVivo_{size}Offsets.npy", Fnoise)
    np.save(f"PnoiseInVivo_{size}Offsets.npy", Pnoise)

########################################################################################################################
# Model Train/Test Functions
########################################################################################################################
def loadSimData(snrTypes, waterTypes, termi, dataTypes):
    '''
    Load simulated corrupt specs (frequency domain)

    :param snrTypes: string
           waterType: string
           termi: string
           dataTypes: string
    :return: ON: fids for 'ON' subspectrum [numSamples, specPoints]
             OFF: fids for 'OFF' subspectrum [numSamples, specPoints]
             ON_FC: frequency corrected fids for 'ON' subspectrum [numSamples, specPoints]
             OFF_FC: frequency corrected fids for 'OFF' subspectrum [numSamples, specPoints]
             FreqL: frequency shift GT labels [numSubSpec (ON=1, OFF=0), numSamples]
             PhaseL: phase shift GT labels [numSubSpec (ON=1, OFF=0), numSamples]
    '''

    dataDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/Corrupt/"
    ON = np.load(f"{dataDir}Sim{snrTypes}_{waterTypes}/ON_AllSpecs_Sim{snrTypes}_{waterTypes}_{dataTypes}{termi}.npy")
    OFF = np.load(f"{dataDir}Sim{snrTypes}_{waterTypes}/OFF_AllSpecs_Sim{snrTypes}_{waterTypes}_{dataTypes}{termi}.npy")
    ON_FC = np.load(f"{dataDir}Sim{snrTypes}_{waterTypes}/ON_AllSpecsFC_Sim{snrTypes}_{waterTypes}_{dataTypes}{termi}.npy")
    OFF_FC = np.load(f"{dataDir}Sim{snrTypes}_{waterTypes}/OFF_AllSpecsFC_Sim{snrTypes}_{waterTypes}_{dataTypes}{termi}.npy")
    FreqL = np.load(f"{dataDir}Sim{snrTypes}_{waterTypes}/TrueFreqLabels_Sim{snrTypes}_{waterTypes}_{dataTypes}{termi}.npy")
    PhaseL = np.load(f"{dataDir}Sim{snrTypes}_{waterTypes}/TruePhaseLabels_Sim{snrTypes}_{waterTypes}_{dataTypes}{termi}.npy")
    return ON, OFF, ON_FC, OFF_FC, FreqL, PhaseL

def loadVivoData(offsetSize):
    '''
    Load in vivo corrupt specs (frequency domain)

    :param offsetSize: string
    :return: ON: fids for 'ON' subspectrum [numSamples, specPoints]
             OFF: fids for 'OFF' subspectrum [numSamples, specPoints]
             FreqL: frequency shift GT labels [numSubSpec (ON=1, OFF=0), numSamples]
             PhaseL: phase shift GT labels [numSubSpec (ON=1, OFF=0), numSamples]
    '''

    dataDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/"
    ON = np.load(f"{dataDir}InVivo/Corrupt/allSpecsInVivoON_{offsetSize}Offsets.npy")[0, :, :]
    OFF = np.load(f"{dataDir}InVivo/Corrupt/allSpecsInVivoOFF_{offsetSize}Offsets.npy")[0, :, :]
    FreqL = np.load(f"{dataDir}InVivo/Corrupt/FnoiseInVivo_{offsetSize}Offsets.npy")
    PhaseL = np.load(f"{dataDir}InVivo/Corrupt/PnoiseInVivo_{offsetSize}Offsets.npy")
    return ON, OFF, FreqL, PhaseL

def normSpecs95(ONSpecs, OFFSpecs):
    '''
    Normalize specs using 90-95th percentile

    :param ONSpecs: [numSamples, specPoints]
           OFFSpecs: [numSamples, specPoints]
    :return: ONSpecs: [numSamples, specPoints]  **normalized by 95th percentile value per spectrum**
             OFFSpecs: [numSamples, specPoints] **normalized by 95th percentile value per spectrum**
    '''

    ONSpecs = ((ONSpecs) / ((np.percentile(np.abs(ONSpecs), 95, axis=1, keepdims=True))))
    OFFSpecs = ((OFFSpecs) / ((np.percentile(np.abs(OFFSpecs), 95, axis=1, keepdims=True))))
    return ONSpecs, OFFSpecs

def normSpecsMax(ONSpecs, OFFSpecs):
    '''
    Normalize specs using 90-95th percentile

    :param ONSpecs: [numSamples, specPoints]
           OFFSpecs: [numSamples, specPoints]
    :return: ONSpecs: [numSamples, specPoints]  **normalized by maximum value per spectrum**
             OFFSpecs: [numSamples, specPoints] **normalized by maximum value per spectrum**
    '''

    ONSpecs = ((ONSpecs) / ((np.percentile(np.abs(ONSpecs), 100, axis=1, keepdims=True))))
    OFFSpecs = ((OFFSpecs) / ((np.percentile(np.abs(OFFSpecs), 100, axis=1, keepdims=True))))
    return ONSpecs, OFFSpecs

def divideDev(ON, OFF, percent):
    '''
    Split development set into train and validation sets

    :param ON: [numSamples, specPoints]
           OFF: [numSamples, specPoints]
    :return: train: [numSamples*percent, specPoints]  **first in list**
             val: [numSamples*percent, specPoints] **last in list**
    '''

    if ON.ndim==2:
        train = np.concatenate((ON[:int(percent * ON.shape[0]), :], OFF[:int(percent * OFF.shape[0]), :]), axis=0)
        val = np.concatenate((ON[int(percent * ON.shape[0]):, :], OFF[int(percent * OFF.shape[0]):, :]), axis=0)
    else:   #assumes ndim=1
        train = np.concatenate((ON[:int(percent * ON.shape[0])], OFF[:int(percent * OFF.shape[0])]), axis=0)[np.newaxis, :]
        val = np.concatenate((ON[int(percent * ON.shape[0]):], OFF[int(percent * OFF.shape[0]):]), axis=0)[np.newaxis, :]
    return train, val

def shuffleData(Specs, SpecsFC, FreqLabels, PhaseLabels):
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

    index_shuf = list(range(Specs.shape[0]))
    shuffle(index_shuf)
    Specs, SpecsFC = Specs[index_shuf, :], SpecsFC[index_shuf, :]
    FreqLabels, PhaseLabels = FreqLabels[:, index_shuf], PhaseLabels[:, index_shuf]
    return Specs, SpecsFC, FreqLabels, PhaseLabels

def getMag(Specs):
    '''
    get magnitude of specs

    :param Specs: [numSamples, specPoints]
    :return: Specs: [numSamples, specPoints]    **magnitude only**
    '''

    return np.abs(Specs)[np.newaxis, :, :]

def get2Chann(Specs):
    '''
    separate complex data into two channels (first real, second imaginary)

    :param Specs: [numSamples, specPoints]
    :return: Specs: [numChanns(2), numSamples, specPoints]
    '''

    twoChanComp = np.empty((2, Specs.shape[0], Specs.shape[1]))
    twoChanComp[0, :, :], twoChanComp[1, :, :] = Specs.real, Specs.imag
    return twoChanComp

def get1Chann(Specs):
    '''
    keep only real values in first channel

    :param Specs: [numSamples, specPoints]
    :return: Specs: [numChanns(1), numSamples, specPoints]
    '''

    oneChanComp = (get2Chann(Specs)[0, :, :])[np.newaxis, :, :]
    return oneChanComp

def window1024(Specs, ppm):
    '''
    select window of 1024 points in SIMULATED specs

    :param Specs: [numChanns, numSamples, specPoints**2048**]
           ppm: [specPoints]
    :return: Specs: [numChanns, numSamples, specPoints**1024**]
    '''

    finish, start = np.where(ppm <= 0.01)[0][0], np.where(ppm >= 7.83)[0][-1]
    return Specs[:, :, start:finish]

def window1024V(Specs, ppm):
    '''
    select window of 1024 points in IN VIVO specs

    :param Specs: [numChanns, numSamples, specPoints**2048**]
           ppm: [specPoints]
    :return: Specs: [numChanns, numSamples, specPoints**1024**]
    '''
    
    start, finish = np.where(ppm <= 0.00)[0][-1], np.where(ppm >= 7.83)[0][0] - 1
    return Specs[:, :, start:finish]

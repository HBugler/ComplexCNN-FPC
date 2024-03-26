# FPC Winter 2024
import numpy as np
from FPC_Functions import simScans, toFids, toSpecs, addComplexNoise, addPShift, addFShift, saveSpecs

########################################################################################################################
# Set-up device, hyperparameters and additional variables
########################################################################################################################
snrTypes = ["10", "5", "2_5"]
waterTypes = ["Pos", "None", "Neg", "Mix"]

ppm = np.load("C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/GTs/ppm_Sim.npy")
time = np.load("C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/GTs/time_Sim.npy")[np.newaxis, :]
timeDev = np.repeat(time, 160*250, axis=0)
timeTest = np.repeat(time, 6*250, axis=0)

for snr in snrTypes:
    indS = snrTypes.index(snr)
    for water in waterTypes:
        indW = waterTypes.index(water)
        print(f'Preparing Data for {waterTypes[indW]} Water - SNR {snrTypes[indS]}')

        # create scans
        ONfids_devGT, OFFfids_devGT = simScans(waterTypes[indW], numTrans=160)
        ONfids_testGT, OFFfids_testGT = simScans(waterTypes[indW], numTrans=6)

        # add amplitude noise
        normNoise_dev = [np.random.uniform(2, 2.5, size=1), np.random.uniform(4, 4.5, size=1), np.random.uniform(9.5, 10, size=1)]
        normNoise_test = [np.random.uniform(2, 2.5, size=1), np.random.uniform(4, 4.5, size=1), np.random.uniform(9.5, 10, size=1)]

        ONfids_dev, OFFfids_dev = addComplexNoise(ONfids_devGT, OFFfids_devGT, normNoise_dev[indS])
        ONfids_test, OFFfids_test = addComplexNoise(ONfids_testGT, OFFfids_testGT, normNoise_test[indS])

        # add frequency and phase noise
        ONfids_devFC, OFFfids_devFC, Pnoise_dev = addPShift(ONfids_dev, OFFfids_dev, shiftRange=90)
        ONfids_devN, OFFfids_devN, Fnoise_dev = addFShift(ONfids_devFC, OFFfids_devFC, timeDev, shiftRange=20)

        ONfids_testFC, OFFfids_testFC, Pnoise_test = addPShift(ONfids_test, OFFfids_test, shiftRange=90)
        ONfids_testN, OFFfids_testN, Fnoise_test = addFShift(ONfids_testFC, OFFfids_testFC, timeTest, shiftRange=20)

        # convert from fids to spectrum and save
        saveSpecs(toSpecs(ONfids_devN), toSpecs(ONfids_devFC), toSpecs(OFFfids_devN), toSpecs(OFFfids_devFC), Fnoise_dev, Pnoise_dev, snrTypes[indS], waterTypes[indW], "DevFinal0322")
        saveSpecs(toSpecs(ONfids_testN), toSpecs(ONfids_testFC), toSpecs(OFFfids_testN), toSpecs(OFFfids_testFC), Fnoise_test, Pnoise_test, snrTypes[indS], waterTypes[indW], "TestFinal0322")
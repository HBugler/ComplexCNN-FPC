import numpy as np
import matplotlib.pyplot as plt
import os
from FPC_Functions import toSpecs, addFShift, addPShift, simScans, addComplexNoise, normSpecs, window1024Trim

########################################################################################################################
# Simulated Data
########################################################################################################################
# create folder for corrupt simulated data
simDir = "C:/Users/Desktop/Simulated/"
simCorrPath = os.path.join(simDir, "Corrupt")
os.mkdir(simCorrPath)

# import time and prepare loop
time = np.load(f"{simDir}GTs/time_Sim.npy")
snrTypes = ["2_5", "5", "10"]
waterTypes = ["Pos", "Mix", "None"]

for snr in snrTypes:
    indS = snrTypes.index(snr)
    for water in waterTypes:
        indW = waterTypes.index(water)
        print(f'Preparing Data for {waterTypes[indW]} Water - SNR {snrTypes[indS]}')

        # create scans from ground truth
        simFidsOffScansDev = simScans(f"{simDir}GTs/fidsOff{waterTypes[indW]}_GABAPlus_Dev.npy", numTrans=160)
        simFidsOnScansDev = simScans(f"{simDir}GTs/fidsOn{waterTypes[indW]}_GABAPlus_Dev.npy", numTrans=160)
        simFidsOffScansTest = simScans(f"{simDir}GTs/fidsOff{waterTypes[indW]}_GABAPlus_Test.npy", numTrans=6)
        simFidsOnScansTest = simScans(f"{simDir}GTs/fidsOn{waterTypes[indW]}_GABAPlus_Test.npy", numTrans=6)

        # add complex Gaussian white noise
        simFidsOffNoisyDev = addComplexNoise(simFidsOffScansDev, noiseStd=indS)
        simFidsOnNoisyDev = addComplexNoise(simFidsOnScansDev, noiseStd=indS)
        simFidsOffNoisyTest = addComplexNoise(simFidsOffScansTest, noiseStd=indS)
        simFidsOnNoisyTest = addComplexNoise(simFidsOnScansTest, noiseStd=indS)

        # add phase noise
        simFidsOffFcDev, PNoiseOffDev = addPShift(simFidsOffNoisyDev, shiftRange=90)
        simFidsOnFcDev, PNoiseOnDev = addPShift(simFidsOnNoisyDev, shiftRange=90)
        simFidsOffFcTest, PNoiseOffTest = addPShift(simFidsOffNoisyTest, shiftRange=90)
        simFidsOnFcTest, PNoiseOnTest = addPShift(simFidsOnNoisyTest, shiftRange=90)

        # add frequency noise
        simFidsOffDev, FNoiseOffDev = addFShift(simFidsOffFcDev, time, shiftRange=20)
        simFidsOnDev, FNoiseOnDev = addFShift(simFidsOnFcDev, time, shiftRange=20)
        simFidsOffTest, FNoiseOffTest = addFShift(simFidsOffFcTest, time, shiftRange=20)
        simFidsOnTest, FNoiseOnTest = addFShift(simFidsOnFcTest, time, shiftRange=20)

        # create noise and specs datasets
        PNoiseOnDev, PNoiseOffDev = PNoiseOnDev[np.newaxis, :], PNoiseOffDev[np.newaxis, :]
        FNoiseOnDev, FNoiseOffDev = FNoiseOnDev[np.newaxis, :], FNoiseOffDev[np.newaxis, :]
        PNoiseOnTest, PNoiseOffTest = PNoiseOnTest[np.newaxis, :], PNoiseOffTest[np.newaxis, :]
        FNoiseOnTest, FNoiseOffTest = FNoiseOnTest[np.newaxis, :], FNoiseOffTest[np.newaxis, :]
        PNoiseDev, FNoiseDev = np.concatenate((PNoiseOffDev, PNoiseOnDev), axis=0), np.concatenate((FNoiseOffDev, FNoiseOnDev), axis=0)
        PNoiseTest, FNoiseTest = np.concatenate((PNoiseOffTest, PNoiseOnTest), axis=0), np.concatenate((FNoiseOffTest, FNoiseOnTest), axis=0)

        # convert to specs
        simSpecsOffFcDev, simSpecsOnFcDev = toSpecs(simFidsOffFcDev), toSpecs(simFidsOnFcDev)
        simSpecsOffDev, simSpecsOnDev = toSpecs(simFidsOffDev), toSpecs(simFidsOnDev)
        simSpecsOffFcTest, simSpecsOnFcTest = toSpecs(simFidsOffFcTest), toSpecs(simFidsOnFcTest)
        simSpecsOffTest, simSpecsOnTest = toSpecs(simFidsOffTest), toSpecs(simFidsOnTest)

        # save sim data
        np.save(f"{simDir}Corrupt/simSpecsOffFc{waterTypes[indW]}{snrTypes[indS]}Dev.npy", simSpecsOffFcDev)
        np.save(f"{simDir}Corrupt/simSpecsOnFc{waterTypes[indW]}{snrTypes[indS]}Dev.npy", simSpecsOnFcDev)
        np.save(f"{simDir}Corrupt/simSpecsOff{waterTypes[indW]}{snrTypes[indS]}Dev.npy", simSpecsOffDev)
        np.save(f"{simDir}Corrupt/simSpecsOn{waterTypes[indW]}{snrTypes[indS]}Dev.npy", simSpecsOnDev)
        np.save(f"{simDir}Corrupt/simPNoise{waterTypes[indW]}{snrTypes[indS]}Dev.npy", PNoiseDev)
        np.save(f"{simDir}Corrupt/simFNoise{waterTypes[indW]}{snrTypes[indS]}Dev.npy", FNoiseDev)

        np.save(f"{simDir}Corrupt/simSpecsOffFc{waterTypes[indW]}{snrTypes[indS]}Test.npy", simSpecsOffFcTest)
        np.save(f"{simDir}Corrupt/simSpecsOnFc{waterTypes[indW]}{snrTypes[indS]}Test.npy", simSpecsOnFcTest)
        np.save(f"{simDir}Corrupt/simSpecsOff{waterTypes[indW]}{snrTypes[indS]}Test.npy", simSpecsOffTest)
        np.save(f"{simDir}Corrupt/simSpecsOn{waterTypes[indW]}{snrTypes[indS]}Test.npy", simSpecsOnTest)
        np.save(f"{simDir}Corrupt/simPNoise{waterTypes[indW]}{snrTypes[indS]}Test.npy", PNoiseTest)
        np.save(f"{simDir}Corrupt/simFNoise{waterTypes[indW]}{snrTypes[indS]}Test.npy", FNoiseTest)

########################################################################################################################
# In Vivo Data
########################################################################################################################
# create folder for corrupt in vivo data
vivoDir = "C:/Users/Desktop/InVivo/"
vivoCorrPath = os.path.join(vivoDir, "Corrupt")
os.mkdir(vivoCorrPath)

# import time and "ground truths"
timeV = np.load(f"{vivoDir}GTs/time_InVivo.npy")
vivoFidsOff = np.load(f"{vivoDir}GTs/allFidsInVivoOFF_NoOffsets.npy")
vivoFidsOn = np.load(f"{vivoDir}GTs/allFidsInVivoON_NoOffsets.npy")

# add phase shifts (creation frequency corrected fids)
vivoFidsOffFcSmall, PNoiseOffSmall = addPShift(np.copy(vivoFidsOff), 20)
vivoFidsOnFcSmall, PNoiseOnSmall = addPShift(np.copy(vivoFidsOn), 20)
vivoFidsOffFcMed, PNoiseOffMed = addPShift(np.copy(vivoFidsOff), 45)
vivoFidsOnFcMed, PNoiseOnMed = addPShift(np.copy(vivoFidsOn), 45)
vivoFidsOffFcLarge, PNoiseOffLarge = addPShift(np.copy(vivoFidsOff), 90)
vivoFidsOnFcLarge, PNoiseOnLarge = addPShift(np.copy(vivoFidsOn), 90)

# add frequency shifts
vivoFidsOffSmall, FNoiseOffSmall = addFShift(np.copy(vivoFidsOff), timeV, 5)
vivoFidsOnSmall, FNoiseOnSmall = addFShift(np.copy(vivoFidsOn), timeV, 5)
vivoFidsOffMed, FNoiseOffMed = addFShift(np.copy(vivoFidsOff), timeV, 10)
vivoFidsOnMed, FNoiseOnMed = addFShift(np.copy(vivoFidsOn), timeV, 10)
vivoFidsOffLarge, FNoiseOffLarge = addFShift(np.copy(vivoFidsOff), timeV, 20)
vivoFidsOnLarge, FNoiseOnLarge = addFShift(np.copy(vivoFidsOn), timeV, 20)

# save in vivo fids to use for correction later
np.save(f"{vivoDir}Corrupt/vivoFidsOffSmallOffsets.npy", vivoFidsOffSmall)
np.save(f"{vivoDir}Corrupt/vivoFidsOnSmallOffsets.npy", vivoFidsOnSmall)
np.save(f"{vivoDir}Corrupt/vivoFidsOffMedOffsets.npy", vivoFidsOffMed)
np.save(f"{vivoDir}Corrupt/vivoFidsOnMedOffsets.npy", vivoFidsOnMed)
np.save(f"{vivoDir}Corrupt/vivoFidsOffLargeOffsets.npy", vivoFidsOffLarge)
np.save(f"{vivoDir}Corrupt/vivoFidsOnLargeOffsets.npy", vivoFidsOnLarge)

# create noise and specs datasets (VERIFY WHAT AXIS TO CONCATENATE ON)
PNoiseOnSmall, PNoiseOffSmall = PNoiseOnSmall[np.newaxis, :], PNoiseOffSmall[np.newaxis, :]
FNoiseOnSmall, FNoiseOffSmall = FNoiseOnSmall[np.newaxis, :], FNoiseOffSmall[np.newaxis, :]
PNoiseOnMed, PNoiseOffMed = PNoiseOnMed[np.newaxis, :], PNoiseOffMed[np.newaxis, :]
FNoiseOnMed, FNoiseOffMed = FNoiseOnMed[np.newaxis, :], FNoiseOffMed[np.newaxis, :]
PNoiseOnLarge, PNoiseOffLarge = PNoiseOnLarge[np.newaxis, :], PNoiseOffLarge[np.newaxis, :]
FNoiseOnLarge, FNoiseOffLarge = FNoiseOnLarge[np.newaxis, :], FNoiseOffLarge[np.newaxis, :]

PNoiseSmall, FNoiseSmall = np.concatenate((PNoiseOnSmall, PNoiseOffSmall), axis=0), np.concatenate((FNoiseOnSmall, FNoiseOffSmall), axis=0)
PNoiseMed, FNoiseMed = np.concatenate((PNoiseOnMed, PNoiseOffMed), axis=0), np.concatenate((FNoiseOnMed, FNoiseOffMed), axis=0)
PNoiseLarge, FNoiseLarge = np.concatenate((PNoiseOnLarge, PNoiseOffLarge), axis=0), np.concatenate((FNoiseOnLarge, FNoiseOffLarge), axis=0)

# convert to specs
vivoSpecsOffSmall, vivoSpecsOnSmall = toSpecs(vivoFidsOffSmall), toSpecs(vivoFidsOnSmall)
vivoSpecsOffFcSmall, vivoSpecsOnFcSmall = toSpecs(vivoFidsOffFcSmall), toSpecs(vivoFidsOnFcSmall)
vivoSpecsOffMed, vivoSpecsOnMed = toSpecs(vivoFidsOffMed), toSpecs(vivoFidsOnMed)
vivoSpecsOffFcMed, vivoSpecsOnFcMed = toSpecs(vivoFidsOffFcMed), toSpecs(vivoFidsOnFcMed)
vivoSpecsOffLarge, vivoSpecsOnLarge = toSpecs(vivoFidsOffLarge), toSpecs(vivoFidsOnLarge)
vivoSpecsOffFcLarge, vivoSpecsOnFcLarge = toSpecs(vivoFidsOffFcLarge), toSpecs(vivoFidsOnFcLarge)

# save in vivo specs to provide to model
np.save(f"{vivoDir}Corrupt/vivoSpecsOffSmallOffsets.npy", vivoSpecsOffSmall)
np.save(f"{vivoDir}Corrupt/vivoSpecsOnSmallOffsets.npy", vivoSpecsOnSmall)
np.save(f"{vivoDir}Corrupt/vivoSpecsOffFcSmallOffsets.npy", vivoSpecsOffFcSmall)
np.save(f"{vivoDir}Corrupt/vivoSpecsOnFcSmallOffsets.npy", vivoSpecsOnFcSmall)
np.save(f"{vivoDir}Corrupt/vivoPNoiseSmallOffsets.npy", PNoiseSmall)
np.save(f"{vivoDir}Corrupt/vivoFNoiseSmallOffsets.npy", FNoiseSmall)

np.save(f"{vivoDir}Corrupt/vivoSpecsOffMedOffsets.npy", vivoSpecsOffMed)
np.save(f"{vivoDir}Corrupt/vivoSpecsOnMedOffsets.npy", vivoSpecsOnMed)
np.save(f"{vivoDir}Corrupt/vivoSpecsOffFcMedOffsets.npy", vivoSpecsOffFcMed)
np.save(f"{vivoDir}Corrupt/vivoSpecsOnFcMedOffsets.npy", vivoSpecsOnFcMed)
np.save(f"{vivoDir}Corrupt/vivoPNoiseMedOffsets.npy", PNoiseMed)
np.save(f"{vivoDir}Corrupt/vivoFNoiseMedOffsets.npy", FNoiseMed)

np.save(f"{vivoDir}Corrupt/vivoSpecsOffLargeOffsets.npy", vivoSpecsOffLarge)
np.save(f"{vivoDir}Corrupt/vivoSpecsOnLargeOffsets.npy", vivoSpecsOnLarge)
np.save(f"{vivoDir}Corrupt/vivoSpecsOffFcLargeOffsets.npy", vivoSpecsOffFcLarge)
np.save(f"{vivoDir}Corrupt/vivoSpecsOnFcLargeOffsets.npy", vivoSpecsOnFcLarge)
np.save(f"{vivoDir}Corrupt/vivoPNoiseLargeOffsets.npy", PNoiseLarge)
np.save(f"{vivoDir}Corrupt/vivoFNoiseLargeOffsets.npy", FNoiseLarge)

########################################################################################################################
# Compare Data
########################################################################################################################
ppmS = np.load(f"{simDir}GTs/ppm_Sim.npy")
ppmV = np.load(f"{vivoDir}GTs/ppm_InVivo.npy")

vivoSpecsOffSmall = np.load(f"{vivoDir}Corrupt/vivoSpecsOffSmallOffsets.npy")
vivoSpecsOnSmall = np.load(f"{vivoDir}Corrupt/vivoSpecsOnSmallOffsets.npy")
vivoSpecsOffFcSmall = np.load(f"{vivoDir}Corrupt/vivoSpecsOffFcSmallOffsets.npy")
vivoSpecsOnFcSmall = np.load(f"{vivoDir}Corrupt/vivoSpecsOnFcSmallOffsets.npy")

vivoSpecsOffMed = np.load(f"{vivoDir}Corrupt/vivoSpecsOffMedOffsets.npy")
vivoSpecsOnMed = np.load(f"{vivoDir}Corrupt/vivoSpecsOnMedOffsets.npy")
vivoSpecsOffFcMed = np.load(f"{vivoDir}Corrupt/vivoSpecsOffFcMedOffsets.npy")
vivoSpecsOnFcMed = np.load(f"{vivoDir}Corrupt/vivoSpecsOnFcMedOffsets.npy")

vivoSpecsOffLarge = np.load(f"{vivoDir}Corrupt/vivoSpecsOffLargeOffsets.npy")
vivoSpecsOnLarge = np.load(f"{vivoDir}Corrupt/vivoSpecsOnLargeOffsets.npy")
vivoSpecsOffFcLarge = np.load(f"{vivoDir}Corrupt/vivoSpecsOffFcLargeOffsets.npy")
vivoSpecsOnFcLarge = np.load(f"{vivoDir}Corrupt/vivoSpecsOnFcLargeOffsets.npy")

simSpecsOffDev = np.load(f"{simDir}Corrupt/simSpecsOffPos2_5Dev.npy")
simSpecsOnDev = np.load(f"{simDir}Corrupt/simSpecsOnMix5Dev.npy")
simSpecsOffTest = np.load(f"{simDir}Corrupt/simSpecsOffPos10Test.npy")
simSpecsOnTest = np.load(f"{simDir}Corrupt/simSpecsOnPos10Test.npy")

simSpecsOffTest1, ppmSTrim = window1024Trim(normSpecs(simSpecsOffTest), ppmS)
vivoSpecsOffSmall1, ppmVTrim = window1024Trim(normSpecs(vivoSpecsOffSmall), ppmV)

fig, axs = plt.subplots(2, 2)
fig.suptitle("Sample In Vivo (blue) and Simulated (red) Spectra")
axs[0,0].set_xlabel('ppm')
axs[0,1].set_xlabel('ppm')
axs[1,0].set_xlabel('ppm')
axs[1,1].set_xlabel('ppm')
axs[0,0].plot(ppmV, normSpecs(vivoSpecsOnLarge)[-1, :].real, 'blue')
axs[0,0].plot(ppmS, normSpecs(simSpecsOnDev)[-1, :].real, 'red')
axs[0,1].plot(ppmV, normSpecs(vivoSpecsOffLarge)[-1, :].real, 'blue')
axs[0,1].plot(ppmS, normSpecs(simSpecsOffDev)[-1, :].real, 'red')
axs[1,0].plot(ppmV, normSpecs(vivoSpecsOnSmall)[0, :].real, 'blue')
axs[1,0].plot(ppmS, normSpecs(simSpecsOnTest)[0, :].real, 'red')
axs[1,1].plot(vivoSpecsOffSmall1[0, :].real, 'blue')
axs[1,1].plot(simSpecsOffTest1[0, :].real, 'red')
axs[0,0].get_yaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[0,0].invert_xaxis()
axs[0,1].invert_xaxis()
axs[1,0].invert_xaxis()
axs[1,1].invert_xaxis()
plt.show()
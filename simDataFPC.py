import numpy as np
import matplotlib.pyplot as plt
import os
from FPC_Functions import toFids, toSpecs, addFShift, addPShift, simScans, addComplexNoise, normSpecs

########################################################################################################################
# Simulated Data
########################################################################################################################
# create folder for corrupt simulated data
simDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/Data/Simulated/"
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

        # create noise and specs datasets (VERIFY WHAT AXIS TO CONCATENATE ON)
        PNoiseDev, FNoiseDev = np.concatenate((PNoiseOnDev, PNoiseOffDev)), np.concatenate((FNoiseOnDev, FNoiseOffDev))
        PNoiseTest, FNoiseTest = np.concatenate((PNoiseOnTest, PNoiseOffTest)), np.concatenate((FNoiseOnTest, FNoiseOffTest))

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
vivoDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/"
vivoCorrPath = os.path.join(vivoDir, "Corrupt")
os.mkdir(vivoCorrPath)

# import time and "ground truths"
timeV = np.load(f"{vivoDir}GTs/time_InVivo.npy")
vivoSpecsOn = np.load(f"{vivoDir}GTs/allSpecsInVivoON_NoOffsets.npy")
vivoSpecsOff = np.load(f"{vivoDir}GTs/allSpecsInVivoOFF_NoOffsets.npy")

# convert to fids
vivoFidsOff = toFids(vivoSpecsOff)
vivoFidsOn = toFids(vivoSpecsOn)

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

# create noise and specs datasets (VERIFY WHAT AXIS TO CONCATENATE ON)
PNoiseSmall, FNoiseSmall = np.concatenate((PNoiseOnSmall, PNoiseOffSmall)), np.concatenate((FNoiseOnSmall, FNoiseOffSmall))
PNoiseMed, FNoiseMed = np.concatenate((PNoiseOnMed, PNoiseOffMed)), np.concatenate((FNoiseOnMed, FNoiseOffMed))
PNoiseLarge, FNoiseLarge = np.concatenate((PNoiseOnLarge, PNoiseOffLarge)), np.concatenate((FNoiseOnLarge, FNoiseOffLarge))

# convert to specs
vivoSpecsOffSmall, vivoSpecsOnSmall = toSpecs(vivoFidsOffSmall), toSpecs(vivoFidsOnSmall)
vivoSpecsOffFcSmall, vivoSpecsOnFcSmall = toSpecs(vivoFidsOffFcSmall), toSpecs(vivoFidsOnFcSmall)
vivoSpecsOffMed, vivoSpecsOnMed = toSpecs(vivoFidsOffMed), toSpecs(vivoFidsOnMed)
vivoSpecsOffFcMed, vivoSpecsOnFcMed = toSpecs(vivoFidsOffFcMed), toSpecs(vivoFidsOnFcMed)
vivoSpecsOffLarge, vivoSpecsOnLarge = toSpecs(vivoFidsOffLarge), toSpecs(vivoFidsOnLarge)
vivoSpecsOffFcLarge, vivoSpecsOnFcLarge = toSpecs(vivoFidsOffFcLarge), toSpecs(vivoFidsOnFcLarge)

# save in vivo data
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
axs[1,1].plot(ppmV, normSpecs(vivoSpecsOffSmall)[0, :].real, 'blue')
axs[1,1].plot(ppmS, normSpecs(simSpecsOffTest)[0, :].real, 'red')
axs[0,0].get_yaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[0,0].invert_xaxis()
axs[0,1].invert_xaxis()
axs[1,0].invert_xaxis()
axs[1,1].invert_xaxis()
plt.show()
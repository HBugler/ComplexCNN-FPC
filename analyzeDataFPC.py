import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon as wlcx
from sklearn.metrics import mean_absolute_error as MAE
from FPC_Functions import loadModelPreds, toFids, toSpecs, corrFShift, corrPShift, reformScans, meanSpec, normSpecs
from plottingFPC import previewData, plotAllScans, plotAllModels, plotQMetric, plotQualityMetrics
from metric_calculator import calculate_snr, calculate_linewidth
########################################################################################################################
# Simulated Data
########################################################################################################################
# create folder for corrupt simulated data
simDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/Data/Simulated/"

# import time and prepare loop
time = np.load(f"{simDir}GTs/time_Sim.npy")
snrTypes = ["2_5", "5", "10"]
waterTypes = ["Pos", "Mix", "None"]
netTypes = ["freq", "phase"]
simModels = ["compReal", "compComp", "realReal", "Ma_4Convs", "Tapper"]

for water in waterTypes:
    indW = waterTypes.index(water)
    for snr in snrTypes:
        indS = snrTypes.index(snr)

        # import true frequency and phase shifts for current dataset
        TrueLbsPhase = np.load(f"{simDir}Corrupt/simPNoise{waterTypes[indW]}{snrTypes[indS]}Test.npy")
        TruePhase = np.concatenate((TrueLbsPhase[1, :], TrueLbsPhase[0, :]))
        trueLbsFreq = np.load(f"{simDir}Corrupt/simFNoise{waterTypes[indW]}{snrTypes[indS]}Test.npy")
        TrueFreq = np.concatenate((trueLbsFreq[1, :], trueLbsFreq[0, :]))

        # import predicted frequency and phase shifts for all models
        m1PhaseLbs, m1FreqLbs = loadModelPreds(simDir, snrTypes[indS], waterTypes[indW], simModels[0])
        m2PhaseLbs, m2FreqLbs = loadModelPreds(simDir, snrTypes[indS], waterTypes[indW], simModels[1])
        m3PhaseLbs, m3FreqLbs = loadModelPreds(simDir, snrTypes[indS], waterTypes[indW], simModels[2])
        m4PhaseLbs, m4FreqLbs = loadModelPreds(simDir, snrTypes[indS], waterTypes[indW], simModels[3])
        m5PhaseLbs, m5FreqLbs = loadModelPreds(simDir, snrTypes[indS], waterTypes[indW], simModels[4])

        # calc MAE, std dev, and statistical significance
        print(f'SNR {snrTypes[indS]} {waterTypes[indW]} Water')
        print(f'MAEs')
        print(f'Freq {simModels[0]}: {MAE(TrueFreq, m1FreqLbs)} +/- {np.std(abs(TrueFreq - m1FreqLbs))}')
        print(f'Phase {simModels[0]}: {MAE(TruePhase, m1PhaseLbs)} +/- {np.std(abs(TruePhase - m1PhaseLbs))}')
        print(f'Freq {simModels[1]}: {MAE(TrueFreq, m2FreqLbs)} +/- {np.std(abs(TrueFreq - m2FreqLbs))}')
        print(f'Phase {simModels[1]}: {MAE(TruePhase, m2PhaseLbs)} +/- {np.std(abs(TruePhase - m2PhaseLbs))}')
        print(f'Freq {simModels[2]}: {MAE(TrueFreq, m3FreqLbs)} +/- {np.std(abs(TrueFreq - m3FreqLbs))}')
        print(f'Phase {simModels[2]}: {MAE(TruePhase, m3PhaseLbs)} +/- {np.std(abs(TruePhase - m3PhaseLbs))}')
        print(f'Freq {simModels[3]}: {MAE(TrueFreq, m4FreqLbs)} +/- {np.std(abs(TrueFreq - m4FreqLbs))}')
        print(f'Phase {simModels[3]}: {MAE(TruePhase, m4PhaseLbs)} +/- {np.std(abs(TruePhase - m4PhaseLbs))}')
        print(f'Freq {simModels[4]}: {MAE(TrueFreq, m5FreqLbs)} +/- {np.std(abs(TrueFreq - m5FreqLbs))}')
        print(f'Phase {simModels[4]}: {MAE(TruePhase, m5PhaseLbs)} +/- {np.std(abs(TruePhase - m5PhaseLbs))}')
        print()
        print(f'Significance Ablation')
        print(f'Freq {simModels[0]} and {simModels[1]}: {wlcx(m1FreqLbs, m2FreqLbs)[1]}')
        print(f'Freq {simModels[0]} and {simModels[2]}: {wlcx(m1FreqLbs, m3FreqLbs)[1]}')
        print(f'Freq {simModels[1]} and {simModels[2]}: {wlcx(m2FreqLbs, m3FreqLbs)[1]}')
        print(f'Phase {simModels[0]} and {simModels[1]}: {wlcx(m1PhaseLbs, m2PhaseLbs)[1]}')
        print(f'Phase {simModels[0]} and {simModels[2]}: {wlcx(m1PhaseLbs, m3PhaseLbs)[1]}')
        print(f'Phase {simModels[1]} and {simModels[2]}: {wlcx(m2PhaseLbs, m3PhaseLbs)[1]}')
        print(f'Significance Comparative')
        print(f'Freq {simModels[0]} and {simModels[3]}: {wlcx(m1FreqLbs, m4FreqLbs)[1]}')
        print(f'Freq {simModels[0]} and {simModels[4]}: {wlcx(m1FreqLbs, m5FreqLbs)[1]}')
        print(f'Freq {simModels[3]} and {simModels[4]}: {wlcx(m4FreqLbs, m5FreqLbs)[1]}')
        print(f'Phase {simModels[0]} and {simModels[3]}: {wlcx(m1PhaseLbs, m4PhaseLbs)[1]}')
        print(f'Phase {simModels[0]} and {simModels[4]}: {wlcx(m1PhaseLbs, m5PhaseLbs)[1]}')
        print(f'Phase {simModels[3]} and {simModels[4]}: {wlcx(m4PhaseLbs, m5PhaseLbs)[1]}')
        print()

########################################################################################################################
# In Vivo Data
########################################################################################################################
vivoModels = ["compComp", "Ma_4Convs", "Tapper"]
size=["Small", "Med", "Large", "None"]
vivoDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/Data/InVivo/"

# import time and ppm
ppmV = np.load(f"{vivoDir}GTs/ppm_InVivo.npy")
timeV = np.load(f"{vivoDir}GTs/time_InVivo.npy")

# load specs, ground truth labels, and prediction labels
print('Loading Data...')

vivoFidsONN = np.load(f"{vivoDir}GTs/allFidsInVivoON_NoOffsets.npy")
vivoFidsOFFN = np.load(f"{vivoDir}GTs/allFidsInVivoOFF_NoOffsets.npy")
vivoFidsONS = np.load(f"{vivoDir}Corrupt/vivoFidsOn{size[0]}Offsets.npy")
vivoFidsOFFS = np.load(f"{vivoDir}Corrupt/vivoFidsOff{size[0]}Offsets.npy")
vivoFidsONM = np.load(f"{vivoDir}Corrupt/vivoFidsOn{size[1]}Offsets.npy")
vivoFidsOFFM = np.load(f"{vivoDir}Corrupt/vivoFidsOff{size[1]}Offsets.npy")
vivoFidsONL = np.load(f"{vivoDir}Corrupt/vivoFidsOn{size[2]}Offsets.npy")
vivoFidsOFFL = np.load(f"{vivoDir}Corrupt/vivoFidsOff{size[2]}Offsets.npy")

m1FreqLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_freq_InVivo_{vivoModels[0]}_{size[3]}.npy")
m1PhaseLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_phase_InVivo_{vivoModels[0]}_{size[3]}.npy")
m1FreqLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_freq_InVivo_{vivoModels[0]}_{size[0]}.npy")
m1PhaseLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_phase_InVivo_{vivoModels[0]}_{size[0]}.npy")
m1FreqLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_freq_InVivo_{vivoModels[0]}_{size[1]}.npy")
m1PhaseLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_phase_InVivo_{vivoModels[0]}_{size[1]}.npy")
m1FreqLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_freq_InVivo_{vivoModels[0]}_{size[2]}.npy")
m1PhaseLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_phase_InVivo_{vivoModels[0]}_{size[2]}.npy")

m2FreqLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_freq_InVivo_{vivoModels[1]}_{size[3]}.npy")
m2PhaseLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_phase_InVivo_{vivoModels[1]}_{size[3]}.npy")
m2FreqLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_freq_InVivo_{vivoModels[1]}_{size[0]}.npy")
m2PhaseLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_phase_InVivo_{vivoModels[1]}_{size[0]}.npy")
m2FreqLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_freq_InVivo_{vivoModels[1]}_{size[1]}.npy")
m2PhaseLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_phase_InVivo_{vivoModels[1]}_{size[1]}.npy")
m2FreqLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_freq_InVivo_{vivoModels[1]}_{size[2]}.npy")
m2PhaseLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_phase_InVivo_{vivoModels[1]}_{size[2]}.npy")

m3FreqLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_freq_InVivo_{vivoModels[2]}_{size[3]}.npy")
m3PhaseLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_phase_InVivo_{vivoModels[2]}_{size[3]}.npy")
m3FreqLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_freq_InVivo_{vivoModels[2]}_{size[0]}.npy")
m3PhaseLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_phase_InVivo_{vivoModels[2]}_{size[0]}.npy")
m3FreqLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_freq_InVivo_{vivoModels[2]}_{size[1]}.npy")
m3PhaseLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_phase_InVivo_{vivoModels[2]}_{size[1]}.npy")
m3FreqLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_freq_InVivo_{vivoModels[2]}_{size[2]}.npy")
m3PhaseLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_phase_InVivo_{vivoModels[2]}_{size[2]}.npy")

###################################################################################################################
# Apply Corrections to Data
###################################################################################################################
print('Applying Corrections...')
# assign in vivo data to models
FidsN = np.concatenate((np.copy(vivoFidsONN), np.copy(vivoFidsOFFN)), axis=0)
FidsS = np.concatenate((np.copy(vivoFidsONS), np.copy(vivoFidsOFFS)), axis=0)
FidsM = np.concatenate((np.copy(vivoFidsONM), np.copy(vivoFidsOFFM)), axis=0)
FidsL = np.concatenate((np.copy(vivoFidsONL), np.copy(vivoFidsOFFL)), axis=0)
SpecsN, SpecsS, SpecsM, SpecsL = toSpecs(FidsN), toSpecs(FidsS), toSpecs(FidsM), toSpecs(FidsL)

m1FidsN = np.concatenate((np.copy(vivoFidsONN), np.copy(vivoFidsOFFN)), axis=0)
m1FidsS = np.concatenate((np.copy(vivoFidsONS), np.copy(vivoFidsOFFS)), axis=0)
m1FidsM = np.concatenate((np.copy(vivoFidsONM), np.copy(vivoFidsOFFM)), axis=0)
m1FidsL = np.concatenate((np.copy(vivoFidsONL), np.copy(vivoFidsOFFL)), axis=0)

m2FidsN = np.concatenate((np.copy(vivoFidsONN), np.copy(vivoFidsOFFN)), axis=0)
m2FidsS = np.concatenate((np.copy(vivoFidsONS), np.copy(vivoFidsOFFS)), axis=0)
m2FidsM = np.concatenate((np.copy(vivoFidsONM), np.copy(vivoFidsOFFM)), axis=0)
m2FidsL = np.concatenate((np.copy(vivoFidsONL), np.copy(vivoFidsOFFL)), axis=0)

m3FidsN = np.concatenate((np.copy(vivoFidsONN), np.copy(vivoFidsOFFN)), axis=0)
m3FidsS = np.concatenate((np.copy(vivoFidsONS), np.copy(vivoFidsOFFS)), axis=0)
m3FidsM = np.concatenate((np.copy(vivoFidsONM), np.copy(vivoFidsOFFM)), axis=0)
m3FidsL = np.concatenate((np.copy(vivoFidsONL), np.copy(vivoFidsOFFL)), axis=0)

# apply frequency and phase correction
m1FidsN = corrFShift(m1FidsN, timeV, m1FreqLbsN)
m1FidsN = corrPShift(m1FidsN, m1PhaseLbsN)
m1FidsS = corrFShift(m1FidsS, timeV, m1FreqLbsS)
m1FidsS = corrPShift(m1FidsS, m1PhaseLbsS)
m1FidsM = corrFShift(m1FidsM, timeV, m1FreqLbsM)
m1FidsM = corrPShift(m1FidsM, m1PhaseLbsM)
m1FidsL = corrFShift(m1FidsL, timeV, m1FreqLbsL)
m1FidsL = corrPShift(m1FidsL, m1PhaseLbsL)

m2FidsN = corrFShift(m2FidsN, timeV, m2FreqLbsN)
m2FidsN = corrPShift(m2FidsN, m2PhaseLbsN)
m2FidsS = corrFShift(m2FidsS, timeV, m2FreqLbsS)
m2FidsS = corrPShift(m2FidsS, m2PhaseLbsS)
m2FidsM = corrFShift(m2FidsM, timeV, m2FreqLbsM)
m2FidsM = corrPShift(m2FidsM, m2PhaseLbsM)
m2FidsL = corrFShift(m2FidsL, timeV, m2FreqLbsL)
m2FidsL = corrPShift(m2FidsL, m2PhaseLbsL)

m3FidsN = corrFShift(m3FidsN, timeV, m3FreqLbsN)
m3FidsN = corrPShift(m3FidsN, m3PhaseLbsN)
m3FidsS = corrFShift(m3FidsS, timeV, m3FreqLbsS)
m3FidsS = corrPShift(m3FidsS, m3PhaseLbsS)
m3FidsM = corrFShift(m3FidsM, timeV, m3FreqLbsM)
m3FidsM = corrPShift(m3FidsM, m3PhaseLbsM)
m3FidsL = corrFShift(m3FidsL, timeV, m3FreqLbsL)
m3FidsL = corrPShift(m3FidsL, m3PhaseLbsL)

# convert to frequency domain SPECS
m1SpecsN, m1SpecsS, m1SpecsM, m1SpecsL = toSpecs(m1FidsN), toSpecs(m1FidsS), toSpecs(m1FidsM), toSpecs(m1FidsL)
m2SpecsN, m2SpecsS, m2SpecsM, m2SpecsL = toSpecs(m2FidsN), toSpecs(m2FidsS), toSpecs(m2FidsM), toSpecs(m2FidsL)
m3SpecsN, m3SpecsS, m3SpecsM, m3SpecsL = toSpecs(m3FidsN), toSpecs(m3FidsS), toSpecs(m3FidsM), toSpecs(m3FidsL)

########################################################################################################################
# reform scans and calculate mean specs (ON=1, OFF=0)
########################################################################################################################
print('Reforming Scans...')
m1SpecsNScans, m1SpecsSScans, m1SpecsMScans, m1SpecsLScans = reformScans(m1SpecsN), reformScans(m1SpecsS), reformScans(m1SpecsM), reformScans(m1SpecsL)
m2SpecsNScans, m2SpecsSScans, m2SpecsMScans, m2SpecsLScans = reformScans(m2SpecsN), reformScans(m2SpecsS), reformScans(m2SpecsM), reformScans(m2SpecsL)
m3SpecsNScans, m3SpecsSScans, m3SpecsMScans, m3SpecsLScans = reformScans(m3SpecsN), reformScans(m3SpecsS), reformScans(m3SpecsM), reformScans(m3SpecsL)

m1SpecsNScansMean, m1SpecsSScansMean, m1SpecsMScansMean, m1SpecsLScansMean = meanSpec(m1SpecsNScans), meanSpec(m1SpecsSScans), meanSpec(m1SpecsMScans), meanSpec(m1SpecsLScans)
m2SpecsNScansMean, m2SpecsSScansMean, m2SpecsMScansMean, m2SpecsLScansMean = meanSpec(m2SpecsNScans), meanSpec(m2SpecsSScans), meanSpec(m2SpecsMScans), meanSpec(m2SpecsLScans)
m3SpecsNScansMean, m3SpecsSScansMean, m3SpecsMScansMean, m3SpecsLScansMean = meanSpec(m3SpecsNScans), meanSpec(m3SpecsSScans), meanSpec(m3SpecsMScans), meanSpec(m3SpecsLScans)

specsNScans, specsSScans, specsMScans, specsLScans = reformScans(SpecsN), reformScans(SpecsS), reformScans(SpecsM), reformScans(SpecsL)
specsNScansMean, specsSScansMean, specsMScansMean, specsLScansMean = meanSpec(specsNScans), meanSpec(specsSScans), meanSpec(specsMScans), meanSpec(specsLScans)

# previewData(ppmV, toSpecs(vivoFidsONN), toSpecs(vivoFidsOFFN), toSpecs(vivoFidsONS), toSpecs(vivoFidsOFFS),
#                 toSpecs(vivoFidsONM), toSpecs(vivoFidsOFFM), toSpecs(vivoFidsONL), toSpecs(vivoFidsOFFL),
#                 m1SpecsSScansMean, m2SpecsSScansMean, m3SpecsSScansMean,
#                 m1SpecsMScansMean, m2SpecsMScansMean, m3SpecsMScansMean,
#                 m1SpecsLScansMean, m2SpecsLScansMean, m3SpecsLScansMean,)

########################################################################################################################
# metric outputs
########################################################################################################################
print('Calculating Metrics...')
allSnrSmall1, meanSnrSmall1, stdSnrSmall1 = calculate_snr(m1SpecsSScansMean, ppmV)
allSnrSmall2, meanSnrSmall2, stdSnrSmall2 = calculate_snr(m2SpecsSScansMean, ppmV)
allSnrSmall3, meanSnrSmall3, stdSnrSmall3 = calculate_snr(m3SpecsSScansMean, ppmV)
allLwSmall1, meanLwSmall1, stdLwSmall1 = calculate_linewidth(m1SpecsSScansMean, ppmV)
allLwSmall2, meanLwSmall2, stdLwSmall2 = calculate_linewidth(m2SpecsSScansMean, ppmV)
allLwSmall3, meanLwSmall3, stdLwSmall3 = calculate_linewidth(m3SpecsSScansMean, ppmV)
print(f'{vivoModels[0]} {size[0]} SNR: {meanSnrSmall1} +/- {stdSnrSmall1}')
print(f'{vivoModels[0]} {size[0]} LW: {meanLwSmall1} +/- {stdLwSmall1}')
print(f'{vivoModels[1]} {size[0]} SNR: {meanSnrSmall2} +/- {stdSnrSmall2}')
print(f'{vivoModels[1]} {size[0]} LW: {meanLwSmall2} +/- {stdLwSmall2}')
print(f'{vivoModels[2]} {size[0]} SNR: {meanSnrSmall3} +/- {stdSnrSmall3}')
print(f'{vivoModels[2]} {size[0]} LW: {meanLwSmall3} +/- {stdLwSmall3}')
print(f'SNR {vivoModels[0]} and {vivoModels[1]}: {wlcx(allSnrSmall1, allSnrSmall2)[1]}')
print(f'SNR {vivoModels[0]} and {vivoModels[2]}: {wlcx(allSnrSmall1, allSnrSmall3)[1]}')
print(f'SNR {vivoModels[1]} and {vivoModels[2]}: {wlcx(allSnrSmall2, allSnrSmall3)[1]}')
print(f'LW {vivoModels[0]} and {vivoModels[1]}: {wlcx(allLwSmall1, allLwSmall2)[1]}')
print(f'LW {vivoModels[0]} and {vivoModels[2]}: {wlcx(allLwSmall1, allLwSmall3)[1]}')
print(f'LW {vivoModels[1]} and {vivoModels[2]}: {wlcx(allLwSmall2, allLwSmall3)[1]}')

allSnrMed1, meanSnrMed1, stdSnrMed1 = calculate_snr(m1SpecsMScansMean, ppmV)
allSnrMed2, meanSnrMed2, stdSnrMed2 = calculate_snr(m2SpecsMScansMean, ppmV)
allSnrMed3, meanSnrMed3, stdSnrMed3 = calculate_snr(m3SpecsMScansMean, ppmV)
allLwMed1, meanLwMed1, stdLwMed1 = calculate_linewidth(m1SpecsMScansMean, ppmV)
allLwMed2, meanLwMed2, stdLwMed2 = calculate_linewidth(m2SpecsMScansMean, ppmV)
allLwMed3, meanLwMed3, stdLwMed3 = calculate_linewidth(m3SpecsMScansMean, ppmV)
print(f'{vivoModels[0]} {size[1]} SNR: {meanSnrMed1} +/- {stdSnrMed1}')
print(f'{vivoModels[0]} {size[1]} LW: {meanLwMed1} +/- {stdLwMed1}')
print(f'{vivoModels[1]} {size[1]} SNR: {meanSnrMed2} +/- {stdSnrMed2}')
print(f'{vivoModels[1]} {size[1]} LW: {meanLwMed2} +/- {stdLwMed2}')
print(f'{vivoModels[2]} {size[1]} SNR: {meanSnrMed3} +/- {stdSnrMed3}')
print(f'{vivoModels[2]} {size[1]} LW: {meanLwMed3} +/- {stdLwMed3}')
print(f'SNR {vivoModels[0]} and {vivoModels[1]}: {wlcx(allSnrMed1, allSnrMed2)[1]}')
print(f'SNR {vivoModels[0]} and {vivoModels[2]}: {wlcx(allSnrMed1, allSnrMed3)[1]}')
print(f'SNR {vivoModels[1]} and {vivoModels[2]}: {wlcx(allSnrMed2, allSnrMed3)[1]}')
print(f'LW {vivoModels[0]} and {vivoModels[1]}: {wlcx(allLwMed1, allLwMed2)[1]}')
print(f'LW {vivoModels[0]} and {vivoModels[2]}: {wlcx(allLwMed1, allLwMed3)[1]}')
print(f'LW {vivoModels[1]} and {vivoModels[2]}: {wlcx(allLwMed2, allLwMed3)[1]}')

allSnrLrg1, meanSnrLrg1, stdSnrLrg1 = calculate_snr(m1SpecsLScansMean, ppmV)
allSnrLrg2, meanSnrLrg2, stdSnrLrg2 = calculate_snr(m2SpecsLScansMean, ppmV)
allSnrLrg3, meanSnrLrg3, stdSnrLrg3 = calculate_snr(m3SpecsLScansMean, ppmV)
allLwLrg1, meanLwLrg1, stdLwLrg1 = calculate_linewidth(m1SpecsLScansMean, ppmV)
allLwLrg2, meanLwLrg2, stdLwLrg2 = calculate_linewidth(m2SpecsLScansMean, ppmV)
allLwLrg3, meanLwLrg3, stdLwLrg3 = calculate_linewidth(m3SpecsLScansMean, ppmV)
print(f'{vivoModels[0]} {size[2]} SNR: {meanSnrLrg1} +/- {stdSnrLrg1}')
print(f'{vivoModels[0]} {size[2]} LW: {meanLwLrg1} +/- {stdLwLrg1}')
print(f'{vivoModels[1]} {size[2]} SNR: {meanSnrLrg2} +/- {stdSnrLrg2}')
print(f'{vivoModels[1]} {size[2]} LW: {meanLwLrg2} +/- {stdLwLrg2}')
print(f'{vivoModels[2]} {size[2]} SNR: {meanSnrLrg3} +/- {stdSnrLrg3}')
print(f'{vivoModels[2]} {size[2]} LW: {meanLwLrg3} +/- {stdLwLrg3}')
print(f'SNR {vivoModels[0]} and {vivoModels[1]}: {wlcx(allSnrLrg1, allSnrLrg2)[1]}')
print(f'SNR {vivoModels[0]} and {vivoModels[2]}: {wlcx(allSnrLrg1, allSnrLrg3)[1]}')
print(f'SNR {vivoModels[1]} and {vivoModels[2]}: {wlcx(allSnrLrg2, allSnrLrg3)[1]}')
print(f'LW {vivoModels[0]} and {vivoModels[1]}: {wlcx(allLwLrg1, allLwLrg2)[1]}')
print(f'LW {vivoModels[0]} and {vivoModels[2]}: {wlcx(allLwLrg1, allLwLrg3)[1]}')
print(f'LW {vivoModels[1]} and {vivoModels[2]}: {wlcx(allLwLrg2, allLwLrg3)[1]}')

allSnrNone1, meanSnrNone1, stdSnrNone1 = calculate_snr(m1SpecsNScansMean, ppmV)
allSnrNone2, meanSnrNone2, stdSnrNone2 = calculate_snr(m2SpecsNScansMean, ppmV)
allSnrNone3, meanSnrNone3, stdSnrNone3 = calculate_snr(m3SpecsNScansMean, ppmV)
allLwNone1, meanLwNone1, stdLwNone1 = calculate_linewidth(m1SpecsNScansMean, ppmV)
allLwNone2, meanLwNone2, stdLwNone2 = calculate_linewidth(m2SpecsNScansMean, ppmV)
allLwNone3, meanLwNone3, stdLwNone3 = calculate_linewidth(m3SpecsNScansMean, ppmV)
print(f'{vivoModels[0]} None SNR: {meanSnrNone1} +/- {stdSnrNone1}')
print(f'{vivoModels[0]} None LW: {meanLwNone1} +/- {stdLwNone1}')
print(f'{vivoModels[1]} None SNR: {meanSnrNone2} +/- {stdSnrNone2}')
print(f'{vivoModels[1]} None LW: {meanLwNone2} +/- {stdLwNone2}')
print(f'{vivoModels[2]} None SNR: {meanSnrNone3} +/- {stdSnrNone3}')
print(f'{vivoModels[2]} None LW: {meanLwNone3} +/- {stdLwNone3}')
print(f'SNR {vivoModels[0]} and {vivoModels[1]}: {wlcx(allSnrNone1, allSnrNone2)[1]}')
print(f'SNR {vivoModels[0]} and {vivoModels[2]}: {wlcx(allSnrNone1, allSnrNone3)[1]}')
print(f'SNR {vivoModels[1]} and {vivoModels[2]}: {wlcx(allSnrNone2, allSnrNone3)[1]}')
print(f'LW {vivoModels[0]} and {vivoModels[1]}: {wlcx(allLwNone1, allLwNone2)[1]}')
print(f'LW {vivoModels[0]} and {vivoModels[2]}: {wlcx(allLwNone1, allLwNone3)[1]}')
print(f'LW {vivoModels[1]} and {vivoModels[2]}: {wlcx(allLwNone2, allLwNone3)[1]}')

########################################################################################################################
# make figures
########################################################################################################################
plotAllScans(ppmV, m1SpecsNScansMean, m1SpecsSScansMean, m1SpecsMScansMean, m1SpecsLScansMean)

plotAllModels(["CC-CNN", "CNN", "MLP"], ppmV, m1SpecsNScansMean, m2SpecsNScansMean, m3SpecsNScansMean,
              m1SpecsSScansMean, m2SpecsSScansMean, m3SpecsSScansMean,
              m1SpecsMScansMean, m2SpecsMScansMean, m3SpecsMScansMean,
              m1SpecsLScansMean, m2SpecsLScansMean, m3SpecsLScansMean,
              specsNScansMean, specsSScansMean, specsMScansMean, specsLScansMean)

plotQMetric(["CC-CNN", "CNN", "MLP"], ppmV, m1SpecsNScansMean.real, m2SpecsNScansMean.real, m3SpecsNScansMean.real,
            m1SpecsSScansMean.real, m2SpecsSScansMean.real, m3SpecsSScansMean.real,
            m1SpecsMScansMean.real, m2SpecsMScansMean.real, m3SpecsMScansMean.real,
            m1SpecsLScansMean.real, m2SpecsLScansMean.real, m3SpecsLScansMean.real)

plotQualityMetrics(["CC-CNN", "CNN", "MLP"], ppmV,
             m1SpecsNScansMean, m2SpecsNScansMean, m3SpecsNScansMean,
             m1SpecsSScansMean, m2SpecsSScansMean, m3SpecsSScansMean,
             m1SpecsMScansMean, m2SpecsMScansMean, m3SpecsMScansMean,
             m1SpecsLScansMean, m2SpecsLScansMean, m3SpecsLScansMean)

########################################################################################################################
# ADDITIONAL CODE FOR NUMPY CONVERSION
########################################################################################################################
# vivoModels = ["compComp", "Ma_4Convs", "Tapper"]
# size=["Small", "Medium", "Large", "None"]       #med for new set
# vivoDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/NumpyData/InVivo/"
#
# # import time and ppm
# ppmV = np.load(f"{vivoDir}GTs/ppm_InVivo.npy")
# timeV = np.load(f"{vivoDir}GTs/time_InVivo.npy")
#
# # load specs, ground truth labels, and prediction labels
# print('Loading Data...')
# vivoSpecsONN = np.load(f"{vivoDir}GTs/allSpecsInVivoON_NoOffsets.npy")
# vivoSpecsOFFN = np.load(f"{vivoDir}GTs/allSpecsInVivoOFF_NoOffsets.npy")
# vivoSpecsONS = np.load(f"{vivoDir}Corrupt/allSpecsInVivoON_{size[0]}Offsets.npy")
# vivoSpecsOFFS = np.load(f"{vivoDir}Corrupt/allSpecsInVivoOFF_{size[0]}Offsets.npy")
# vivoSpecsONM = np.load(f"{vivoDir}Corrupt/allSpecsInVivoON_{size[1]}Offsets.npy")
# vivoSpecsOFFM = np.load(f"{vivoDir}Corrupt/allSpecsInVivoOFF_{size[1]}Offsets.npy")
# vivoSpecsONL = np.load(f"{vivoDir}Corrupt/allSpecsInVivoON_{size[2]}Offsets.npy")
# vivoSpecsOFFL = np.load(f"{vivoDir}Corrupt/allSpecsInVivoOFF_{size[2]}Offsets.npy")

# m1FreqLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_freq_InVivo_{vivoModels[0]}_{size[3]}.npy")
# m1PhaseLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_phase_InVivo_{vivoModels[0]}_{size[3]}.npy")
# m1FreqLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_freq_InVivo_{vivoModels[0]}_{size[0]}.npy")
# m1PhaseLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_phase_InVivo_{vivoModels[0]}_{size[0]}.npy")
# m1FreqLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_freq_InVivo_{vivoModels[0]}_{size[1]}.npy")
# m1PhaseLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_phase_InVivo_{vivoModels[0]}_{size[1]}.npy")
# m1FreqLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_freq_InVivo_{vivoModels[0]}_{size[2]}.npy")
# m1PhaseLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_phase_InVivo_{vivoModels[0]}_{size[2]}.npy")
#
# m2FreqLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_freq_InVivo_{vivoModels[1]}_{size[3]}.npy")
# m2PhaseLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_phase_InVivo_{vivoModels[1]}_{size[3]}.npy")
# m2FreqLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_freq_InVivo_{vivoModels[1]}_{size[0]}.npy")
# m2PhaseLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_phase_InVivo_{vivoModels[1]}_{size[0]}.npy")
# m2FreqLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_freq_InVivo_{vivoModels[1]}_{size[1]}.npy")
# m2PhaseLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_phase_InVivo_{vivoModels[1]}_{size[1]}.npy")
# m2FreqLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_freq_InVivo_{vivoModels[1]}_{size[2]}.npy")
# m2PhaseLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_phase_InVivo_{vivoModels[1]}_{size[2]}.npy")
#
# m3FreqLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_freq_InVivo_{vivoModels[2]}_{size[3]}.npy")
# m3PhaseLbsN = np.load(f"{vivoDir}Predictions/PredLabels_{size[3]}_phase_InVivo_{vivoModels[2]}_{size[3]}.npy")
# m3FreqLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_freq_InVivo_{vivoModels[2]}_{size[0]}.npy")
# m3PhaseLbsS = np.load(f"{vivoDir}Predictions/PredLabels_{size[0]}_phase_InVivo_{vivoModels[2]}_{size[0]}.npy")
# m3FreqLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_freq_InVivo_{vivoModels[2]}_{size[1]}.npy")
# m3PhaseLbsM = np.load(f"{vivoDir}Predictions/PredLabels_{size[1]}_phase_InVivo_{vivoModels[2]}_{size[1]}.npy")
# m3FreqLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_freq_InVivo_{vivoModels[2]}_{size[2]}.npy")
# m3PhaseLbsL = np.load(f"{vivoDir}Predictions/PredLabels_{size[2]}_phase_InVivo_{vivoModels[2]}_{size[2]}.npy")
#
#
# # assign in vivo data to models
# FidsN = np.concatenate((np.copy(toFidsAlt(vivoSpecsONN)), np.copy(toFidsAlt(vivoSpecsOFFN))), axis=0)
# FidsS = np.concatenate((np.copy(toFidsAlt(vivoSpecsONS)), np.copy(toFidsAlt(vivoSpecsOFFS))), axis=0)
# FidsM = np.concatenate((np.copy(toFidsAlt(vivoSpecsONM)), np.copy(toFidsAlt(vivoSpecsOFFM))), axis=0)
# FidsL = np.concatenate((np.copy(toFidsAlt(vivoSpecsONL)), np.copy(toFidsAlt(vivoSpecsOFFL))), axis=0)
# SpecsN, SpecsS, SpecsM, SpecsL = toSpecs(FidsN), toSpecs(FidsS), toSpecs(FidsM), toSpecs(FidsL)
#
# m1FidsN = np.concatenate((np.copy(toFidsAlt(vivoSpecsONN)), np.copy(toFidsAlt(vivoSpecsOFFN))), axis=0)
# m1FidsS = np.concatenate((np.copy(toFidsAlt(vivoSpecsONS)), np.copy(toFidsAlt(vivoSpecsOFFS))), axis=0)
# m1FidsM = np.concatenate((np.copy(toFidsAlt(vivoSpecsONM)), np.copy(toFidsAlt(vivoSpecsOFFM))), axis=0)
# m1FidsL = np.concatenate((np.copy(toFidsAlt(vivoSpecsONL)), np.copy(toFidsAlt(vivoSpecsOFFL))), axis=0)
#
# m2FidsN = np.concatenate((np.copy(toFidsAlt(vivoSpecsONN)), np.copy(toFidsAlt(vivoSpecsOFFN))), axis=0)
# m2FidsS = np.concatenate((np.copy(toFidsAlt(vivoSpecsONS)), np.copy(toFidsAlt(vivoSpecsOFFS))), axis=0)
# m2FidsM = np.concatenate((np.copy(toFidsAlt(vivoSpecsONM)), np.copy(toFidsAlt(vivoSpecsOFFM))), axis=0)
# m2FidsL = np.concatenate((np.copy(toFidsAlt(vivoSpecsONL)), np.copy(toFidsAlt(vivoSpecsOFFL))), axis=0)
#
# m3FidsN = np.concatenate((np.copy(toFidsAlt(vivoSpecsONN)), np.copy(toFidsAlt(vivoSpecsOFFN))), axis=0)
# m3FidsS = np.concatenate((np.copy(toFidsAlt(vivoSpecsONS)), np.copy(toFidsAlt(vivoSpecsOFFS))), axis=0)
# m3FidsM = np.concatenate((np.copy(toFidsAlt(vivoSpecsONM)), np.copy(toFidsAlt(vivoSpecsOFFM))), axis=0)
# m3FidsL = np.concatenate((np.copy(toFidsAlt(vivoSpecsONL)), np.copy(toFidsAlt(vivoSpecsOFFL))), axis=0)
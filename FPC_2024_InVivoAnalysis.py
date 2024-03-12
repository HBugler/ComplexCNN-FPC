import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error
from metric_calculator import calculate_snr, calculate_NewLW, calculate_linewidth, calculate_ModelledLW
from plottingFPC import plotFig3A, plotFig3B, qMetricPlot, plotFig4
import random

# ########################################################################################################################
# # Load Data
# ########################################################################################################################
size = ["Small", "Medium", "Large"]
net = ["Freq", "Phase"]
model = ["compReal", "Ma_4Convs", "Tapper"]
subDir = f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/"
fileExt = f"_lr25PercentMaxOnly_B64Adam_E200_"

# phase labels already have frequency corrections applied
time = np.load(f"{subDir}time_InVivo.npy").flatten()
ppm = np.load(f"{subDir}ppm_InVivo.npy").flatten()

TrueLabels_FreqSmall = np.load(f"{subDir}FnoiseInVivo_{size[0]}Offsets.npy")
TrueLabels_FreqMed = np.load(f"{subDir}FnoiseInVivo_{size[1]}Offsets.npy")
TrueLabels_FreqLarge = np.load(f"{subDir}FnoiseInVivo_{size[2]}Offsets.npy")
TrueLabels_PhaseSmall = np.load(f"{subDir}PnoiseInVivo_{size[0]}Offsets.npy")
TrueLabels_PhaseMed = np.load(f"{subDir}PnoiseInVivo_{size[1]}Offsets.npy")
TrueLabels_PhaseLarge = np.load(f"{subDir}PnoiseInVivo_{size[2]}Offsets.npy")

TrueLabels_FreqSmall = np.concatenate((TrueLabels_FreqSmall[1, :], TrueLabels_FreqSmall[0, :]))
TrueLabels_FreqMed = np.concatenate((TrueLabels_FreqMed[1, :], TrueLabels_FreqMed[0, :]))
TrueLabels_FreqLarge = np.concatenate((TrueLabels_FreqLarge[1, :], TrueLabels_FreqLarge[0, :]))
TrueLabels_PhaseSmall = np.concatenate((TrueLabels_PhaseSmall[1, :], TrueLabels_PhaseSmall[0, :]))
TrueLabels_PhaseMed = np.concatenate((TrueLabels_PhaseMed[1, :], TrueLabels_PhaseMed[0, :]))
TrueLabels_PhaseLarge = np.concatenate((TrueLabels_PhaseLarge[1, :], TrueLabels_PhaseLarge[0, :]))

specsNONEON = np.load(f"{subDir}raw/allSpecsInVivoON_NoOffsets.npy")
specsNONEOFF = np.load(f"{subDir}raw/allSpecsInVivoOFF_NoOffsets.npy")

specsSmallON = np.load(f"{subDir}allSpecsInVivoON_{size[0]}Offsets.npy")
specsSmallOFF = np.load(f"{subDir}allSpecsInVivoOFF_{size[0]}Offsets.npy")
specsMedON = np.load(f"{subDir}allSpecsInVivoON_{size[1]}Offsets.npy")
specsMedOFF = np.load(f"{subDir}allSpecsInVivoOFF_{size[1]}Offsets.npy")
specsLargeON = np.load(f"{subDir}allSpecsInVivoON_{size[2]}Offsets.npy")
specsLargeOFF = np.load(f"{subDir}allSpecsInVivoOFF_{size[2]}Offsets.npy")

compReal_freqLabelsSmall = np.load(f"{subDir}PredLabels_{net[0]}_InVivo_{model[0]}{fileExt}{size[0]}.npy")
compReal_phaseLabelsSmall = np.load(f"{subDir}PredLabels_{net[1]}_InVivo_{model[0]}{fileExt}{size[0]}.npy")
compReal_freqLabelsMed = np.load(f"{subDir}PredLabels_{net[0]}_InVivo_{model[0]}{fileExt}{size[1]}.npy")
compReal_phaseLabelsMed = np.load(f"{subDir}PredLabels_{net[1]}_InVivo_{model[0]}{fileExt}{size[1]}.npy")
compReal_freqLabelsLarge = np.load(f"{subDir}PredLabels_{net[0]}_InVivo_{model[0]}{fileExt}{size[2]}.npy")
compReal_phaseLabelsLarge = np.load(f"{subDir}PredLabels_{net[1]}_InVivo_{model[0]}{fileExt}{size[2]}.npy")

Ma_freqLabelsMed = np.load(f"{subDir}PredLabels_{net[0]}_InVivo_{model[1]}{fileExt}{size[1]}.npy")
Ma_phaseLabelsMed = np.load(f"{subDir}PredLabels_{net[1]}_InVivo_{model[1]}{fileExt}{size[1]}.npy")
Ma_freqLabelsSmall = np.load(f"{subDir}PredLabels_{net[0]}_InVivo_{model[1]}{fileExt}{size[0]}.npy")
Ma_phaseLabelsSmall = np.load(f"{subDir}PredLabels_{net[1]}_InVivo_{model[1]}{fileExt}{size[0]}.npy")
Ma_freqLabelsLarge = np.load(f"{subDir}PredLabels_{net[0]}_InVivo_{model[1]}{fileExt}{size[2]}.npy")
Ma_phaseLabelsLarge = np.load(f"{subDir}PredLabels_{net[1]}_InVivo_{model[1]}{fileExt}{size[2]}.npy")

Tapper_freqLabelsSmall = np.load(f"{subDir}PredLabels_{net[0]}_InVivo_{model[2]}{fileExt}{size[0]}.npy")
Tapper_phaseLabelsSmall = np.load(f"{subDir}PredLabels_{net[1]}_InVivo_{model[2]}{fileExt}{size[0]}.npy")
Tapper_freqLabelsMed = np.load(f"{subDir}PredLabels_{net[0]}_InVivo_{model[2]}{fileExt}{size[1]}.npy")
Tapper_phaseLabelsMed = np.load(f"{subDir}PredLabels_{net[1]}_InVivo_{model[2]}{fileExt}{size[1]}.npy")
Tapper_freqLabelsLarge = np.load(f"{subDir}PredLabels_{net[0]}_InVivo_{model[2]}{fileExt}{size[2]}.npy")
Tapper_phaseLabelsLarge = np.load(f"{subDir}PredLabels_{net[1]}_InVivo_{model[2]}{fileExt}{size[2]}.npy")

# # check if initial predictions make sense
# specsCorrON = np.fft.fft(np.fft.fftshift(np.copy(specsMedON), axes=2), axis=2)
# specsCorrOFF = np.fft.fft(np.fft.fftshift(np.copy(specsMedOFF), axes=2), axis=2)
# half = int(compReal_freqLabelsSmall.shape[0]/2)
# for i in range(0, half):
#     specsCorrON[0, i, :] = specsCorrON[0, i, :] * np.exp(-1j * time * -compReal_freqLabelsMed[i] * 2 * math.pi)*np.exp(-1j * -compReal_phaseLabelsMed[i] * math.pi / 180)
#     specsCorrOFF[0, i, :] = specsCorrOFF[0, i, :] * np.exp(-1j * time * -compReal_freqLabelsMed[half+i] * 2 * math.pi)*np.exp(-1j * -compReal_phaseLabelsMed[half+i] * math.pi / 180)
# specsCorrON = np.fft.fftshift(np.fft.ifft(specsCorrON, axis=2), axes=2)
# specsCorrOFF = np.fft.fftshift(np.fft.ifft(specsCorrOFF, axis=2), axes=2)
#
# # sanity check
# for i in range(0, 1):
#     randScan1, randScan2, randScan3 = random.randint(0, specsNONEON.shape[1]-10), random.randint(0, specsNONEON.shape[1]-10), random.randint(0, specsNONEON.shape[1]-10)
#     print(f'Scan number {randScan1}')
#     print(f'Blue (ON) FREQ True {TrueLabels_FreqMed[randScan1]} and Pred: {compReal_freqLabelsMed[randScan1]}')
#     print(f'Scan number {randScan2}')
#     print(f'Orange (ON) FREQ True {TrueLabels_FreqMed[randScan2]} and Pred: {compReal_freqLabelsMed[randScan2]}')
#     print(f'Scan number {randScan3}')
#     print(f'Green (ON) FREQ True {TrueLabels_FreqMed[randScan3]} and Pred: {compReal_freqLabelsMed[randScan3]}')
#
#     fig1, (ax1, ax2, ax3) = plt.subplots(3)
#     ax1.plot(ppm, specsNONEON[0, randScan1, :].real, 'black')
#     ax1.plot(ppm, specsMedON[0, randScan1, :].real, 'blue')
#     ax1.plot(ppm, specsCorrON[0, randScan1, :].real, 'red')
#     ax1.invert_xaxis()
#     ax2.plot(ppm, specsNONEOFF[0, randScan2, :].real, 'black')
#     ax2.plot(ppm, specsMedOFF[0, randScan2, :].real, 'orange')
#     ax2.plot(ppm, specsCorrOFF[0, randScan2, :].real, 'red')
#     ax2.invert_xaxis()
#     ax3.plot(ppm, specsNONEON[0, randScan3, :].real, 'black')
#     ax3.plot(ppm, specsMedON[0, randScan3, :].real, 'green')
#     ax3.plot(ppm, specsCorrON[0, randScan3, :].real, 'red')
#     ax3.invert_xaxis()
#     plt.show()
#
# scan = random.randint(1, int(specsNONEON.shape[1]/160))
# print(f'scan number is {scan}')
# scan1None = (specsNONEON[0, 160*(scan-1):160*scan, :].real - specsNONEOFF[0, 160*(scan-1):160*scan, :].real).mean(axis=0)
# scan1Med = (specsMedON[0, 160*(scan-1):160*scan, :].real - specsMedOFF[0, 160*(scan-1):160*scan :].real).mean(axis=0)
# scan1Corr = (specsCorrON[0, 160*(scan-1):160*scan, :].real - specsCorrOFF[0, 160*(scan-1):160*scan :].real).mean(axis=0)
#
# fig2, (ax1) = plt.subplots(1)
# ax1.plot(ppm, scan1None, 'black')
# ax1.plot(ppm, scan1Med, 'blue')
# ax1.plot(ppm, scan1Corr, 'red')
# ax1.invert_xaxis()
# plt.show()
# quit()

########################################################################################################################
# Apply Corrections to Data
########################################################################################################################
# convert to time domain FID
CR_SmallON = np.fft.fft(np.fft.fftshift(np.copy(specsSmallON), axes=2), axis=2)
CR_SmallOFF = np.fft.fft(np.fft.fftshift(np.copy(specsSmallOFF), axes=2), axis=2)
CR_MedON = np.fft.fft(np.fft.fftshift(np.copy(specsMedON), axes=2), axis=2)
CR_MedOFF = np.fft.fft(np.fft.fftshift(np.copy(specsMedOFF), axes=2), axis=2)
CR_LargeON = np.fft.fft(np.fft.fftshift(np.copy(specsLargeON), axes=2), axis=2)
CR_LargeOFF = np.fft.fft(np.fft.fftshift(np.copy(specsLargeOFF), axes=2), axis=2)

Ma_SmallON = np.fft.fft(np.fft.fftshift(np.copy(specsSmallON), axes=2), axis=2)
Ma_SmallOFF = np.fft.fft(np.fft.fftshift(np.copy(specsSmallOFF), axes=2), axis=2)
Ma_MedON = np.fft.fft(np.fft.fftshift(np.copy(specsMedON), axes=2), axis=2)
Ma_MedOFF = np.fft.fft(np.fft.fftshift(np.copy(specsMedOFF), axes=2), axis=2)
Ma_LargeON = np.fft.fft(np.fft.fftshift(np.copy(specsLargeON), axes=2), axis=2)
Ma_LargeOFF = np.fft.fft(np.fft.fftshift(np.copy(specsLargeOFF), axes=2), axis=2)

Tp_SmallON = np.fft.fft(np.fft.fftshift(np.copy(specsSmallON), axes=2), axis=2)
Tp_SmallOFF = np.fft.fft(np.fft.fftshift(np.copy(specsSmallOFF), axes=2), axis=2)
Tp_MedON = np.fft.fft(np.fft.fftshift(np.copy(specsMedON), axes=2), axis=2)
Tp_MedOFF = np.fft.fft(np.fft.fftshift(np.copy(specsMedOFF), axes=2), axis=2)
Tp_LargeON = np.fft.fft(np.fft.fftshift(np.copy(specsLargeON), axes=2), axis=2)
Tp_LargeOFF = np.fft.fft(np.fft.fftshift(np.copy(specsLargeOFF), axes=2), axis=2)

half = int(compReal_freqLabelsSmall.shape[0]/2)

for i in range(0, half):
    CR_SmallON[0, i, :] = CR_SmallON[0, i, :] * np.exp(-1j * time * -compReal_freqLabelsSmall[i] * 2 * math.pi) * np.exp(-1j * -compReal_phaseLabelsSmall[i] * math.pi / 180)
    CR_SmallOFF[0, i, :] = CR_SmallOFF[0, i, :] * np.exp(-1j * time * -compReal_freqLabelsSmall[half+i] * 2 * math.pi) * np.exp(-1j * -compReal_phaseLabelsSmall[half+i] * math.pi / 180)
    CR_MedON[0, i, :] = CR_MedON[0, i, :] * np.exp(-1j * time * -compReal_freqLabelsMed[i] * 2 * math.pi) * np.exp(-1j * -compReal_phaseLabelsMed[i] * math.pi / 180)
    CR_MedOFF[0, i, :] = CR_MedOFF[0, i, :] * np.exp(-1j * time * -compReal_freqLabelsMed[half+i] * 2 * math.pi) * np.exp(-1j * -compReal_phaseLabelsMed[half+i] * math.pi / 180)
    CR_LargeON[0, i, :] = CR_LargeON[0, i, :] * np.exp(-1j * time * -compReal_freqLabelsLarge[i] * 2 * math.pi) * np.exp(-1j * -compReal_phaseLabelsLarge[i] * math.pi / 180)
    CR_LargeOFF[0, i, :] = CR_LargeOFF[0, i, :] * np.exp(-1j * time * -compReal_freqLabelsLarge[half+i] * 2 * math.pi) * np.exp(-1j * -compReal_phaseLabelsLarge[half+i] * math.pi / 180)

    Ma_SmallON[0, i, :] = Ma_SmallON[0, i, :] * np.exp(-1j * time * -Ma_freqLabelsSmall[i] * 2 * math.pi) * np.exp(-1j * -Ma_phaseLabelsSmall[i] * math.pi / 180)
    Ma_SmallOFF[0, i, :] = Ma_SmallOFF[0, i, :] * np.exp(-1j * time * -Ma_freqLabelsSmall[half+i] * 2 * math.pi) * np.exp(-1j * -Ma_phaseLabelsSmall[half+i] * math.pi / 180)
    Ma_MedON[0, i, :] = Ma_MedON[0, i, :] * np.exp(-1j * time * -Ma_freqLabelsMed[i] * 2 * math.pi) * np.exp(-1j * -Ma_phaseLabelsMed[i] * math.pi / 180)
    Ma_MedOFF[0, i, :] = Ma_MedOFF[0, i, :] * np.exp(-1j * time * -Ma_freqLabelsMed[half+i] * 2 * math.pi) * np.exp(-1j * -Ma_phaseLabelsMed[half+i] * math.pi / 180)
    Ma_LargeON[0, i, :] = Ma_LargeON[0, i, :] * np.exp(-1j * time * -Ma_freqLabelsLarge[i] * 2 * math.pi) * np.exp(-1j * -Ma_phaseLabelsLarge[i] * math.pi / 180)
    Ma_LargeOFF[0, i, :] = Ma_LargeOFF[0, i, :] * np.exp(-1j * time * -Ma_freqLabelsLarge[half+i] * 2 * math.pi) * np.exp(-1j * -Ma_phaseLabelsLarge[half+i] * math.pi / 180)

    Tp_SmallON[0, i, :] = Tp_SmallON[0, i, :] * np.exp(-1j * time * -Tapper_freqLabelsSmall[i] * 2 * math.pi) * np.exp(-1j * -Tapper_phaseLabelsSmall[i] * math.pi / 180)
    Tp_SmallOFF[0, i, :] = Tp_SmallOFF[0, i, :] * np.exp(-1j * time * -Tapper_freqLabelsSmall[half+i] * 2 * math.pi) * np.exp(-1j * -Tapper_phaseLabelsSmall[half+i] * math.pi / 180)
    Tp_MedON[0, i, :] = Tp_MedON[0, i, :] * np.exp(-1j * time * -Tapper_freqLabelsMed[i] * 2 * math.pi) * np.exp(-1j * -Tapper_phaseLabelsMed[i] * math.pi / 180)
    Tp_MedOFF[0, i, :] = Tp_MedOFF[0, i, :] * np.exp(-1j * time * -Tapper_freqLabelsMed[half+i] * 2 * math.pi) * np.exp(-1j * -Tapper_phaseLabelsMed[half+i] * math.pi / 180)
    Tp_LargeON[0, i, :] = Tp_LargeON[0, i, :] * np.exp(-1j * time * -Tapper_freqLabelsLarge[i] * 2 * math.pi) * np.exp(-1j * -Tapper_phaseLabelsLarge[i] * math.pi / 180)
    Tp_LargeOFF[0, i, :] = Tp_LargeOFF[0, i, :] * np.exp(-1j * time * -Tapper_freqLabelsLarge[half+i] * 2 * math.pi) * np.exp(-1j * -Tapper_phaseLabelsLarge[half+i] * math.pi / 180)

CR_SmallON = np.fft.fftshift(np.fft.ifft(CR_SmallON, axis=2), axes=2)
CR_SmallOFF = np.fft.fftshift(np.fft.ifft(CR_SmallOFF, axis=2), axes=2)
CR_MedON = np.fft.fftshift(np.fft.ifft(CR_MedON, axis=2), axes=2)
CR_MedOFF = np.fft.fftshift(np.fft.ifft(CR_MedOFF, axis=2), axes=2)
CR_LargeON = np.fft.fftshift(np.fft.ifft(CR_LargeON, axis=2), axes=2)
CR_LargeOFF = np.fft.fftshift(np.fft.ifft(CR_LargeOFF, axis=2), axes=2)

Ma_SmallON = np.fft.fftshift(np.fft.ifft(Ma_SmallON, axis=2), axes=2)
Ma_SmallOFF = np.fft.fftshift(np.fft.ifft(Ma_SmallOFF, axis=2), axes=2)
Ma_MedON = np.fft.fftshift(np.fft.ifft(Ma_MedON, axis=2), axes=2)
Ma_MedOFF = np.fft.fftshift(np.fft.ifft(Ma_MedOFF, axis=2), axes=2)
Ma_LargeON = np.fft.fftshift(np.fft.ifft(Ma_LargeON, axis=2), axes=2)
Ma_LargeOFF = np.fft.fftshift(np.fft.ifft(Ma_LargeOFF, axis=2), axes=2)

Tp_SmallON = np.fft.fftshift(np.fft.ifft(Tp_SmallON, axis=2), axes=2)
Tp_SmallOFF = np.fft.fftshift(np.fft.ifft(Tp_SmallOFF, axis=2), axes=2)
Tp_MedON = np.fft.fftshift(np.fft.ifft(Tp_MedON, axis=2), axes=2)
Tp_MedOFF = np.fft.fftshift(np.fft.ifft(Tp_MedOFF, axis=2), axes=2)
Tp_LargeON = np.fft.fftshift(np.fft.ifft(Tp_LargeON, axis=2), axes=2)
Tp_LargeOFF = np.fft.fftshift(np.fft.ifft(Tp_LargeOFF, axis=2), axes=2)

########################################################################################################################
# reform scans and calculate mean specs (ON=1, OFF=0)
########################################################################################################################
UnCorr_specsSmall_scans, UnCorr_specsMed_scans, UnCorr_specsLarge_scans = np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex), np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex), np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex)
compReal_specsSmall_scans, Ma_specsSmall_scans, Tapper_specsSmall_scans = np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex), np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex), np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex)
compReal_specsMed_scans, Ma_specsMed_scans, Tapper_specsMed_scans = np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex), np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex), np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex)
compReal_specsLarge_scans, Ma_specsLarge_scans, Tapper_specsLarge_scans = np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex), np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex), np.empty(shape=(int(compReal_freqLabelsSmall.shape[0]/320), 2, 160, 2048), dtype=complex)

for k in range(0, int(UnCorr_specsSmall_scans.shape[0])):
    UnCorr_specsSmall_scans[k, 1, :, :] = specsSmallON[0, 160*k:160*(k+1), :]
    UnCorr_specsSmall_scans[k, 0, :, :] = specsSmallOFF[0, 160*k:160*(k+1), :]
    UnCorr_specsMed_scans[k, 1, :, :] = specsMedON[0, 160*k:160*(k+1), :]
    UnCorr_specsMed_scans[k, 0, :, :] = specsMedOFF[0, 160*k:160*(k+1), :]
    UnCorr_specsLarge_scans[k, 1, :, :] = specsLargeON[0, 160*k:160*(k+1), :]
    UnCorr_specsLarge_scans[k, 0, :, :] = specsLargeOFF[0, 160*k:160*(k+1), :]

    compReal_specsSmall_scans[k, 1, :, :] = CR_SmallON[0, 160*k:160*(k+1), :]
    compReal_specsSmall_scans[k, 0, :, :] = CR_SmallOFF[0, 160*k:160*(k+1), :]
    Ma_specsSmall_scans[k, 1, :, :] = Ma_SmallON[0, 160*k:160*(k+1), :]
    Ma_specsSmall_scans[k, 0, :, :] = Ma_SmallOFF[0, 160*k:160*(k+1), :]
    Tapper_specsSmall_scans[k, 1, :, :] = Tp_SmallON[0, 160*k:160*(k+1), :]
    Tapper_specsSmall_scans[k, 0, :, :] = Tp_SmallOFF[0, 160*k:160*(k+1), :]

    compReal_specsMed_scans[k, 1, :, :] = CR_MedON[0, 160*k:160*(k+1), :]
    compReal_specsMed_scans[k, 0, :, :] = CR_MedOFF[0, 160*k:160*(k+1), :]
    Ma_specsMed_scans[k, 1, :, :] = Ma_MedON[0, 160*k:160*(k+1), :]
    Ma_specsMed_scans[k, 0, :, :] = Ma_MedOFF[0, 160*k:160*(k+1), :]
    Tapper_specsMed_scans[k, 1, :, :] = Tp_MedON[0, 160*k:160*(k+1), :]
    Tapper_specsMed_scans[k, 0, :, :] = Tp_MedOFF[0, 160*k:160*(k+1), :]

    compReal_specsLarge_scans[k, 1, :, :] = CR_LargeON[0, 160*k:160*(k+1), :]
    compReal_specsLarge_scans[k, 0, :, :] = CR_LargeOFF[0, 160*k:160*(k+1), :]
    Ma_specsLarge_scans[k, 1, :, :] = Ma_LargeON[0, 160*k:160*(k+1), :]
    Ma_specsLarge_scans[k, 0, :, :] = Ma_LargeOFF[0, 160*k:160*(k+1), :]
    Tapper_specsLarge_scans[k, 1, :, :] = Tp_LargeON[0, 160*k:160*(k+1), :]
    Tapper_specsLarge_scans[k, 0, :, :] = Tp_LargeOFF[0, 160*k:160*(k+1), :]

# calculate mean specs and take true value only
UnCorr_specsSmall_meanScans = (UnCorr_specsSmall_scans[:, 1, :, :] - UnCorr_specsSmall_scans[:, 0, :, :]).mean(axis=1).real
UnCorr_specsMed_meanScans = (UnCorr_specsMed_scans[:, 1, :, :] - UnCorr_specsMed_scans[:, 0, :, :]).mean(axis=1).real
UnCorr_specsLarge_meanScans = (UnCorr_specsLarge_scans[:, 1, :, :] - UnCorr_specsLarge_scans[:, 0, :, :]).mean(axis=1).real

compReal_specsSmall_meanscans = (compReal_specsSmall_scans[:, 1, :, :] - compReal_specsSmall_scans[:, 0, :, :]).mean(axis=1).real
Ma_specsSmall_meanscans = (Ma_specsSmall_scans[:, 1, :, :] - Ma_specsSmall_scans[:, 0, :, :]).mean(axis=1).real
Tapper_specsSmall_meanscans = (Tapper_specsSmall_scans[:, 1, :, :] - Tapper_specsSmall_scans[:, 0, :, :]).mean(axis=1).real

compReal_specsMed_meanscans = (compReal_specsMed_scans[:, 1, :, :] - compReal_specsMed_scans[:, 0, :, :]).mean(axis=1).real
Ma_specsMed_meanscans = (Ma_specsMed_scans[:, 1, :, :] - Ma_specsMed_scans[:, 0, :, :]).mean(axis=1).real
Tapper_specsMed_meanscans = (Tapper_specsMed_scans[:, 1, :, :] - Tapper_specsMed_scans[:, 0, :, :]).mean(axis=1).real

compReal_specsLarge_meanscans = (compReal_specsLarge_scans[:, 1, :, :] - compReal_specsLarge_scans[:, 0, :, :]).mean(axis=1).real
Ma_specsLarge_meanscans = (Ma_specsLarge_scans[:, 1, :, :] - Ma_specsLarge_scans[:, 0, :, :]).mean(axis=1).real
Tapper_specsLarge_meanscans = (Tapper_specsLarge_scans[:, 1, :, :] - Tapper_specsLarge_scans[:, 0, :, :]).mean(axis=1).real

########################################################################################################################
# metric outputs
########################################################################################################################
# allSnr_compRealSmall, meanSnr_compRealSmall, stdSnr_compRealSmall = calculate_snr(compReal_specsSmall_meanscans, ppm)
# allLw_compRealSmall, meanLw_compRealSmall, stdLw_compRealSmall = calculate_ModelledLW(compReal_specsSmall_meanscans, ppm)
allSnr_MaSmall, meanSnr_MaSmall, stdSnr_MaSmall = calculate_snr(Ma_specsSmall_meanscans, ppm)
allLw_MaSmall, meanLw_MaSmall, stdLw_MaSmall = calculate_ModelledLW(Ma_specsSmall_meanscans, ppm)
allSnr_TapperSmall, meanSnr_TapperSmall, stdSnr_TapperSmall = calculate_snr(Tapper_specsSmall_meanscans, ppm)
allLw_TapperSmall, meanLw_TapperSmall, stdLw_TapperSmall = calculate_ModelledLW(Tapper_specsSmall_meanscans, ppm)

# print(f'compReal Small SNR: {meanSnr_compRealSmall} +/- {stdSnr_compRealSmall}')
# print(f'compReal Small LW: {meanLw_compRealSmall} +/- {stdLw_compRealSmall}')
print(f'Ma Small SNR: {meanSnr_MaSmall} +/- {stdSnr_MaSmall}')
print(f'Ma Small LW: {meanLw_MaSmall} +/- {stdLw_MaSmall}')
print(f'Tapper Small SNR: {meanSnr_TapperSmall} +/- {stdSnr_TapperSmall}')
print(f'Tapper Small LW: {meanLw_TapperSmall} +/- {stdLw_TapperSmall}')

allSnr_compRealMed, meanSnr_compRealMed, stdSnr_compRealMed = calculate_snr(compReal_specsMed_meanscans, ppm)
allLw_compRealMed, meanLw_compRealMed, stdLw_compRealMed = calculate_ModelledLW(compReal_specsMed_meanscans, ppm)
allSnr_MaMed, meanSnr_MaMed, stdSnr_MaMed = calculate_snr(Ma_specsMed_meanscans, ppm)
allLw_MaMed, meanLw_MaMed, stdLw_MaMed = calculate_ModelledLW(Ma_specsMed_meanscans, ppm)
allSnr_TapperMed, meanSnr_TapperMed, stdSnr_TapperMed = calculate_snr(Tapper_specsMed_meanscans, ppm)
allLw_TapperMed, meanLw_TapperMed, stdLw_TapperMed = calculate_ModelledLW(Tapper_specsMed_meanscans, ppm)

print(f'compReal Med SNR: {meanSnr_compRealMed} +/- {stdSnr_compRealMed}')
print(f'compReal Med LW: {meanLw_compRealMed} +/- {stdLw_compRealMed}')
print(f'Ma Med SNR: {meanSnr_MaMed} +/- {stdSnr_MaMed}')
print(f'Ma Med LW: {meanLw_MaMed} +/- {stdLw_MaMed}')
print(f'Tapper Med SNR: {meanSnr_TapperMed} +/- {stdSnr_TapperMed}')
print(f'Tapper Med LW: {meanLw_TapperMed} +/- {stdLw_TapperMed}')

allSnr_compRealLarge, meanSnr_compRealLarge, stdSnr_compRealLarge = calculate_snr(compReal_specsLarge_meanscans, ppm)
allLw_compRealLarge, meanLw_compRealLarge, stdLw_compRealLarge, = calculate_ModelledLW(compReal_specsLarge_meanscans, ppm)
allSnr_MaLarge, meanSnr_MaLarge, stdSnr_MaLarge = calculate_snr(Ma_specsLarge_meanscans, ppm)
allLw_MaLarge, meanLw_MaLarge, stdLw_MaLarge = calculate_ModelledLW(Ma_specsLarge_meanscans, ppm)
allSnr_TapperLarge, meanSnr_TapperLarge, stdSnr_TapperLarge = calculate_snr(Tapper_specsLarge_meanscans, ppm)
allLw_TapperLarge, meanLw_TapperLarge, stdLw_TapperLarge = calculate_ModelledLW(Tapper_specsLarge_meanscans, ppm)

print(f'compReal Large SNR: {meanSnr_compRealLarge} +/- {stdSnr_compRealLarge}')
print(f'compReal Large LW: {meanLw_compRealLarge} +/- {stdLw_compRealLarge}')
print(f'Ma Large SNR: {meanSnr_MaLarge} +/- {stdSnr_MaLarge}')
print(f'Ma Large LW: {meanLw_MaLarge} +/- {stdLw_MaLarge}')
print(f'Tapper Large SNR: {meanSnr_TapperLarge} +/- {stdSnr_TapperLarge}')
print(f'Tapper Large LW: {meanLw_TapperLarge} +/- {stdLw_TapperLarge}')

########################################################################################################################
# calculate significance
########################################################################################################################
# randScan = random.randint(0, 35)
#
# fig1, (ax1, ax2, ax3) = plt.subplots(3)
# ax1.plot(ppm, np.real(specsSmallON[0, :, randScan]), 'blue')
# ax2.plot(ppmSIM, np.real(alltemp[0, :, randScan]), 'red')
# ax3.plot(ppm, np.real(specsSmallON[0, :, randScan]), 'blue')
# ax3.plot(ppmSIM, np.real(alltemp[0, :, randScan]), 'red')
# ax1.plot(ppm, compReal_specsLarge_meanscans[randScan, :], 'blue')
# ax1.plot(ppm, Ma_specsLarge_meanscans[randScan, :], 'red')
# ax1.plot(ppm, Tapper_specsLarge_meanscans[randScan, :], 'green')
# ax1.invert_xaxis()
# ax2.invert_xaxis()
# ax3.invert_xaxis()
# plt.show()
#
#alltemp
statSmall_UsMa_snr, pvalueSmall_UsMa_snr = stats.wilcoxon(allSnr_compRealSmall, allSnr_MaSmall)
statSmall_UsTapper_snr, pvalueSmall_UsTapper_snr = stats.wilcoxon(allSnr_compRealSmall, allSnr_TapperSmall)
statSmall_MaTapper_snr, pvalueSmall_MaTapper_snr = stats.wilcoxon(allSnr_MaSmall, allSnr_TapperSmall)
statSmall_UsMa_lw, pvalueSmall_UsMa_lw = stats.wilcoxon(allLw_compRealSmall, allLw_MaSmall)
statSmall_UsTapper_lw, pvalueSmall_UsTapper_lw = stats.wilcoxon(allLw_compRealSmall, allLw_TapperSmall)
statSmall_MaTapper_lw, pvalueSmall_MaTapper_lw = stats.wilcoxon(allLw_MaSmall, allLw_TapperSmall)

print(f'pvalueSmall_UsMa_snr {pvalueSmall_UsMa_snr}')
print(f'pvalueSmall_UsTapper_snr {pvalueSmall_UsTapper_snr}')
print(f'pvalueSmall_MaTapper_snr {pvalueSmall_MaTapper_snr}')
print(f'pvalueSmall_UsMa_lw {pvalueSmall_UsMa_lw}')
print(f'pvalueSmall_UsTapper_lw {pvalueSmall_UsTapper_lw}')
print(f'pvalueSmall_MaTapper_lw {pvalueSmall_MaTapper_lw}')

statMed_UsMa_snr, pvalueMed_UsMa_snr = stats.wilcoxon(allSnr_compRealMed, allSnr_MaMed)
statMed_UsTapper_snr, pvalueMed_UsTapper_snr = stats.wilcoxon(allSnr_compRealMed, allSnr_TapperMed)
statMed_MaTapper_snr, pvalueMed_MaTapper_snr = stats.wilcoxon(allSnr_MaMed, allSnr_TapperMed)
statMed_UsMa_lw, pvalueMed_UsMa_lw = stats.wilcoxon(allLw_compRealMed, allLw_MaSmall)
statMed_UsTapper_lw, pvalueMed_UsTapper_lw = stats.wilcoxon(allLw_compRealMed, allLw_TapperMed)
statMed_MaTapper_lw, pvalueMed_MaTapper_lw = stats.wilcoxon(allLw_MaMed, allLw_TapperMed)

print(f'pvalueMed_UsMa_snr {pvalueMed_UsMa_snr}')
print(f'pvalueMed_UsTapper_snr {pvalueMed_UsTapper_snr}')
print(f'pvalueMed_MaTapper_snr {pvalueMed_MaTapper_snr}')
print(f'pvalueMed_UsMa_lw {pvalueMed_UsMa_lw}')
print(f'pvalueMed_UsTapper_lw {pvalueMed_UsTapper_lw}')
print(f'pvalueMed_MaTapper_lw {pvalueMed_MaTapper_lw}')

statLarge_UsMa_snr, pvalueLarge_UsMa_snr = stats.wilcoxon(allSnr_compRealLarge, allSnr_MaLarge)
statLarge_UsTapper_snr, pvalueLarge_UsTapper_snr = stats.wilcoxon(allSnr_compRealLarge, allSnr_TapperLarge)
statLarge_MaTapper_snr, pvalueLarge_MaTapper_snr = stats.wilcoxon(allSnr_MaLarge, allSnr_TapperLarge)
statLarge_UsMa_lw, pvalueLarge_UsMa_lw = stats.wilcoxon(allLw_compRealLarge, allLw_MaLarge)
statLarge_UsTapper_lw, pvalueLarge_UsTapper_lw = stats.wilcoxon(allLw_compRealLarge, allLw_TapperLarge)
statLarge_MaTapper_lw, pvalueLarge_MaTapper_lw = stats.wilcoxon(allLw_MaLarge, allLw_TapperLarge)

print(f'pvalueLarge_UsMa_snr {pvalueLarge_UsMa_snr}')
print(f'pvalueLarge_UsTapper_snr {pvalueLarge_UsTapper_snr}')
print(f'pvalueLarge_MaTapper_snr {pvalueLarge_MaTapper_snr}')
print(f'pvalueLarge_UsMa_lw {pvalueLarge_UsMa_lw}')
print(f'pvalueLarge_UsTapper_lw {pvalueLarge_UsTapper_lw}')
print(f'pvalueLarge_MaTapper_lw {pvalueLarge_MaTapper_lw}')

########################################################################################################################
# make figures
########################################################################################################################
# mean scans [numScans, numSpecPoints]
plotFig3A(ppm, compReal_specsSmall_meanscans, compReal_specsMed_meanscans, compReal_specsLarge_meanscans)

plotFig3B(ppm, compReal_specsSmall_meanscans, Ma_specsSmall_meanscans, Tapper_specsSmall_meanscans,
          compReal_specsMed_meanscans, Ma_specsMed_meanscans, Tapper_specsMed_meanscans,
          compReal_specsLarge_meanscans, Ma_specsLarge_meanscans, Tapper_specsLarge_meanscans,
          UnCorr_specsSmall_meanScans, UnCorr_specsMed_meanScans, UnCorr_specsLarge_meanScans)
#
# qMetricPlot(ppm, np.real(compReal_specsSmall_meanscans), np.real(Ma_specsSmall_meanscans), np.real(compReal_specsMed_meanscans),
#             np.real(Ma_specsMed_meanscans), np.real(compReal_specsLarge_meanscans), np.real(Ma_specsLarge_meanscans))
#
plotFig4(meanSnr_compRealSmall, meanSnr_compRealMed, meanSnr_compRealLarge,
         meanSnr_MaSmall, meanSnr_MaMed, meanSnr_MaLarge,
         meanSnr_TapperSmall, meanSnr_TapperMed, meanSnr_TapperLarge,
         meanLw_compRealSmall, meanLw_compRealMed, meanLw_compRealLarge,
         meanLw_MaSmall, meanLw_MaMed, meanLw_MaLarge,
         meanLw_TapperSmall, meanLw_TapperMed, meanLw_TapperLarge,
         stdLw_compRealSmall, stdLw_MaSmall, stdLw_TapperSmall,
         stdLw_compRealMed, stdLw_MaMed, stdLw_TapperMed,
         stdLw_compRealLarge, stdLw_MaLarge, stdLw_TapperLarge,
         stdSnr_compRealSmall, stdSnr_MaSmall, stdSnr_TapperSmall,
         stdSnr_compRealMed, stdSnr_MaMed, stdSnr_TapperMed,
         stdSnr_compRealLarge, stdSnr_MaLarge, stdSnr_TapperLarge)

# need to change unCorrs to no offsets
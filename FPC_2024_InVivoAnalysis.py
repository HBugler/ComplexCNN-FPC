import numpy as np
import matplotlib.pyplot as plt
from plottingFPC import plotFig3A, plotFig3B, qMetricPlot, plotFig4
from FPC_Functions import toFids, toSpecs, corrFShift, corrPShift, reformScans, meanSpec, getMetricsSignificance, loadVivo
########################################################################################################################
# Load Data
########################################################################################################################
model = ["compReal", "Ma_4Convs", "Tapper"]
subDir = f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/"
fileExt = f"_MaxBothIndv_"

# load specs, ground truth labels, and prediction labels
time = np.load(f"{subDir}GTs/time_InVivo.npy")
ppm = np.load(f"{subDir}GTs/ppm_InVivo.npy")

specsNONEON, specsNONEOFF, specsSmallON, specsSmallOFF, specsMedON, specsMedOFF, specsLargeON, specsLargeOFF = loadVivo(subDir, fileExt, model[0], 'specs')
TruefreqNone, TruephaseNone, TruefreqSmall, TruephaseSmall, TruefreqMed, TruephaseMed, TruefreqLarge, TruephaseLarge = loadVivo(subDir, fileExt, model[0], 'true')
CrfreqNone, CrphaseNone, CrfreqSmall, CrphaseSmall, CrfreqMed, CrphaseMed, CrfreqLarge, CrphaseLarge = loadVivo(subDir, fileExt, model[0], 'pred')
MafreqNone, MaphaseNone, MafreqSmall, MaphaseSmall, MafreqMed, MaphaseMed, MafreqLarge, MaphaseLarge = loadVivo(subDir, fileExt, model[1], 'pred')
TpfreqNone, TpphaseNone, TpfreqSmall, TpphaseSmall, TpfreqMed, TpphaseMed, TpfreqLarge, TpphaseLarge = loadVivo(subDir, fileExt, model[2], 'pred')

###################################################################################################################
# Apply Corrections to Data
###################################################################################################################
# convert to time domain FID
CR_SmallFids = toFids(np.concatenate((np.copy(specsSmallON), np.copy(specsSmallOFF)), axis=0))
CR_MedFids = toFids(np.concatenate((np.copy(specsMedON), np.copy(specsMedOFF)), axis=0))
CR_LargeFids = toFids(np.concatenate((np.copy(specsLargeON), np.copy(specsLargeOFF)), axis=0))

Ma_SmallFids = toFids(np.concatenate((np.copy(specsSmallON), np.copy(specsSmallOFF)), axis=0))
Ma_MedFids = toFids(np.concatenate((np.copy(specsMedON), np.copy(specsMedOFF)), axis=0))
Ma_LargeFids = toFids(np.concatenate((np.copy(specsLargeON), np.copy(specsLargeOFF)), axis=0))

Tp_SmallFids = toFids(np.concatenate((np.copy(specsSmallON), np.copy(specsSmallOFF)), axis=0))
Tp_MedFids = toFids(np.concatenate((np.copy(specsMedON), np.copy(specsMedOFF)), axis=0))
Tp_LargeFids = toFids(np.concatenate((np.copy(specsLargeON), np.copy(specsLargeOFF)), axis=0))

# apply frequency correction
CR_SmallFids = corrFShift(CR_SmallFids, time, CrfreqSmall)
CR_MedFids = corrFShift(CR_MedFids, time, CrfreqMed)
CR_LargeFids = corrFShift(CR_LargeFids, time, CrfreqLarge)

Ma_SmallFids = corrFShift(Ma_SmallFids, time, MafreqSmall)
Ma_MedFids = corrFShift(Ma_MedFids, time, MafreqMed)
Ma_LargeFids = corrFShift(Ma_LargeFids, time, MafreqLarge)

Tp_SmallFids = corrFShift(Tp_SmallFids, time, TpfreqSmall)
Tp_MedFids = corrFShift(Tp_MedFids, time, TpfreqMed)
Tp_LargeFids = corrFShift(Tp_LargeFids, time, TpfreqLarge)

# apply phase correction
CR_SmallFids = corrPShift(CR_SmallFids, CrphaseSmall)
CR_MedFids = corrPShift(CR_MedFids, CrphaseMed)
CR_LargeFids = corrPShift(CR_LargeFids, CrphaseLarge)

Ma_SmallFids = corrPShift(Ma_SmallFids, MaphaseSmall)
Ma_MedFids = corrPShift(Ma_MedFids, MaphaseMed)
Ma_LargeFids = corrPShift(Ma_LargeFids, MaphaseLarge)

Tp_SmallFids = corrPShift(Tp_SmallFids, TpphaseSmall)
Tp_MedFids = corrPShift(Tp_MedFids, TpphaseMed)
Tp_LargeFids = corrPShift(Tp_LargeFids, TpphaseLarge)

# convert to frequency domain SPECS
CR_SmallSpecs, CR_MedSpecs, CR_LargeSpecs = toSpecs(CR_SmallFids), toSpecs(CR_MedFids), toSpecs(CR_LargeFids)
Ma_SmallSpecs, Ma_MedSpecs, Ma_LargeSpecs = toSpecs(Ma_SmallFids), toSpecs(Ma_MedFids), toSpecs(Ma_LargeFids)
Tp_SmallSpecs, Tp_MedSpecs, Tp_LargeSpecs = toSpecs(Tp_SmallFids), toSpecs(Tp_MedFids), toSpecs(Tp_LargeFids)
ON = CR_SmallSpecs[:int(CR_SmallSpecs.shape[0]/2), :]
OFF = CR_SmallSpecs[int(CR_SmallSpecs.shape[0]/2):, :]

########################################################################################################################
# reform scans and calculate mean specs (ON=1, OFF=0)
########################################################################################################################
CrSmallScans, CrMedScans, CrLargeScans = reformScans(CR_SmallSpecs, CR_MedSpecs, CR_LargeSpecs)
MaSmallScans, MaMedScans, MaLargeScans = reformScans(Ma_SmallSpecs, Ma_MedSpecs, Ma_LargeSpecs)
TpSmallScans, TpMedScans, TpLargeScans = reformScans(Tp_SmallSpecs, Tp_MedSpecs, Tp_LargeSpecs)

CrSmallScansMean, CrMedScansMean, CrLargeScansMean = meanSpec(CrSmallScans, CrMedScans, CrLargeScans)
MaSmallScansMean, MaMedScansMean, MaLargeScansMean = meanSpec(MaSmallScans, MaMedScans, MaLargeScans)
TpSmallScansMean, TpMedScansMean, TpLargeScansMean = meanSpec(TpSmallScans, TpMedScans, TpLargeScans)

########################################################################################################################
# metric outputs
########################################################################################################################
setName = ["compReal", "Ma", "Tapper"]
getMetricsSignificance(CrSmallScansMean.real, MaSmallScansMean.real, TpSmallScansMean.real, ppm, setName, sizeName="Small")
getMetricsSignificance(CrMedScansMean.real, MaMedScansMean.real, TpMedScansMean.real, ppm, setName, sizeName="Medium")
getMetricsSignificance(CrLargeScansMean.real, MaLargeScansMean.real, TpLargeScansMean.real, ppm, setName, sizeName="Large")

for iii in range(0, 36, 3):
    fig1, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(ppm, (specsNONEON[160*iii:160*(iii+1), :]-specsNONEOFF[160*iii:160*(iii+1), :]).mean(axis=0).real-0.00001, 'black')
    ax1.plot(ppm, (specsSmallON[160*iii:160*(iii+1), :]-specsSmallOFF[160*iii:160*(iii+1), :]).mean(axis=0).real, 'purple')
    ax1.plot(ppm, CrSmallScansMean[iii, :].real+0.00001, 'orange')
    ax1.plot(ppm, MaSmallScansMean[iii, :].real+0.00002, 'blue')
    ax1.plot(ppm, TpSmallScansMean[iii, :].real+0.00003, 'green')
    ax1.set_xlim(0, 8)
    ax1.set_ylim(-0.00002, 0.00004)
    ax2.plot(ppm, (specsNONEON[160*(iii+1):160*(iii+2), :]-specsNONEOFF[160*(iii+1):160*(iii+2), :]).mean(axis=0).real-0.00001, 'black')
    ax2.plot(ppm, (specsSmallON[160*(iii+1):160*(iii+2), :]-specsSmallOFF[160*(iii+1):160*(iii+2), :]).mean(axis=0).real, 'purple')
    ax2.plot(ppm, CrSmallScansMean[iii+1, :].real+0.00001, 'orange')
    ax2.plot(ppm, MaSmallScansMean[iii+1, :].real+0.00002, 'blue')
    ax2.plot(ppm, TpSmallScansMean[iii+1, :].real+0.00003, 'green')
    ax2.set_xlim(0, 8)
    ax2.set_ylim(-0.00002, 0.00004)
    ax3.plot(ppm, (specsNONEON[160*(iii+2):160*(iii+3), :]-specsNONEOFF[160*(iii+2):160*(iii+3), :]).mean(axis=0).real-0.00001, 'black')
    ax3.plot(ppm, (specsSmallON[160*(iii+2):160*(iii+3), :]-specsSmallOFF[160*(iii+2):160*(iii+3), :]).mean(axis=0).real, 'purple')
    ax3.plot(ppm, CrSmallScansMean[iii+2, :].real+0.00001, 'orange')
    ax3.plot(ppm, MaSmallScansMean[iii+2, :].real+0.00002, 'blue')
    ax3.plot(ppm, TpSmallScansMean[iii+2, :].real+0.00003, 'green')
    ax3.set_xlim(0, 8)
    ax3.set_ylim(-0.00002, 0.00004)
    plt.show()

########################################################################################################################
# make figures
########################################################################################################################
# plotFig3A(ppm, CrSmallScansMean, CrMedScansMean, CrLargeScansMean)
#
# plotFig3B(ppm, CrSmallScansMean, MaSmallScansMean, TpSmallScansMean,
#           CrMedScansMean, MaMedScansMean, TpMedScansMean,
#           CrLargeScansMean, MaLargeScansMean, TpLargeScansMean,
#           UnCorr_specsSmall_meanScans, UnCorr_specsMed_meanScans, UnCorr_specsLarge_meanScans)
# #
# # qMetricPlot(ppm, np.real(compReal_specsSmall_meanscans), np.real(Ma_specsSmall_meanscans), np.real(compReal_specsMed_meanscans),
# #             np.real(Ma_specsMed_meanscans), np.real(compReal_specsLarge_meanscans), np.real(Ma_specsLarge_meanscans))
# #
# plotFig4(meanSnr_compRealSmall, meanSnr_compRealMed, meanSnr_compRealLarge,
#          meanSnr_MaSmall, meanSnr_MaMed, meanSnr_MaLarge,
#          meanSnr_TapperSmall, meanSnr_TapperMed, meanSnr_TapperLarge,
#          meanLw_compRealSmall, meanLw_compRealMed, meanLw_compRealLarge,
#          meanLw_MaSmall, meanLw_MaMed, meanLw_MaLarge,
#          meanLw_TapperSmall, meanLw_TapperMed, meanLw_TapperLarge,
#          stdLw_compRealSmall, stdLw_MaSmall, stdLw_TapperSmall,
#          stdLw_compRealMed, stdLw_MaMed, stdLw_TapperMed,
#          stdLw_compRealLarge, stdLw_MaLarge, stdLw_TapperLarge,
#          stdSnr_compRealSmall, stdSnr_MaSmall, stdSnr_TapperSmall,
#          stdSnr_compRealMed, stdSnr_MaMed, stdSnr_TapperMed,
#          stdSnr_compRealLarge, stdSnr_MaLarge, stdSnr_TapperLarge)
#
# # need to change unCorrs to no offsets
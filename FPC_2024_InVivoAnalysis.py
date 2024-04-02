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
vivoPPM = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/GTs/ppm_InVivo.npy")
vivoTIME = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/GTs/time_InVivo.npy")
size=["Small", "Medium", "Large"]
vivoSpecsONN = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/GTs/allSpecsInVivoON_NoOffsets.npy")
vivoSpecsOFFN = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/GTs/allSpecsInVivoOFF_NoOffsets.npy")
vivoSpecsONS = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Corrupt/allSpecsInVivoON_{size[0]}Offsets.npy")
vivoSpecsOFFS = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Corrupt/allSpecsInVivoOFF_{size[0]}Offsets.npy")
vivoSpecsONM = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Corrupt/allSpecsInVivoON_{size[1]}Offsets.npy")
vivoSpecsOFFM = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Corrupt/allSpecsInVivoOFF_{size[1]}Offsets.npy")
vivoSpecsONL = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Corrupt/allSpecsInVivoON_{size[2]}Offsets.npy")
vivoSpecsOFFL = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Corrupt/allSpecsInVivoOFF_{size[2]}Offsets.npy")

CRfreqLabelsS = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Small_freq_InVivo_compReal_Small.npy")
CRphaseLabelsS = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Small_phase_InVivo_compReal_Small.npy")
CRfreqLabelsM = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Medium_freq_InVivo_compReal_Medium.npy")
CRphaseLabelsM = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Medium_phase_InVivo_compReal_Medium.npy")
CRfreqLabelsL = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Large_freq_InVivo_compReal_Large.npy")
CRphaseLabelsL = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Large_phase_InVivo_compReal_Large.npy")
MafreqLabelsS = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Small_freq_InVivo_Ma_4Convs_Small.npy")
MaphaseLabelsS = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Small_phase_InVivo_Ma_4Convs_Small.npy")
MafreqLabelsM = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Medium_freq_InVivo_Ma_4Convs_Medium.npy")
MaphaseLabelsM = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Medium_phase_InVivo_Ma_4Convs_Medium.npy")
MafreqLabelsL = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Large_freq_InVivo_Ma_4Convs_Large.npy")
MaphaseLabelsL = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Large_phase_InVivo_Ma_4Convs_Large.npy")
TpfreqLabelsS = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Small_freq_InVivo_Tapper_Small.npy")
TpphaseLabelsS = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Small_phase_InVivo_Tapper_Small.npy")
TpfreqLabelsM = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Medium_freq_InVivo_Tapper_Medium.npy")
TpphaseLabelsM = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Medium_phase_InVivo_Tapper_Medium.npy")
TpfreqLabelsL = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Large_freq_InVivo_Tapper_Large.npy")
TpphaseLabelsL = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/Predictions/PredLabels_Large_phase_InVivo_Tapper_Large.npy")

###################################################################################################################
# Apply Corrections to Data
###################################################################################################################
# assign data
CRspecsS = np.concatenate((np.copy(vivoSpecsONS), np.copy(vivoSpecsOFFS)), axis=0)
CRspecsM = np.concatenate((np.copy(vivoSpecsONM), np.copy(vivoSpecsOFFM)), axis=0)
CRspecsL = np.concatenate((np.copy(vivoSpecsONL), np.copy(vivoSpecsOFFL)), axis=0)

MaspecsS = np.concatenate((np.copy(vivoSpecsONS), np.copy(vivoSpecsOFFS)), axis=0)
MaspecsM = np.concatenate((np.copy(vivoSpecsONM), np.copy(vivoSpecsOFFM)), axis=0)
MaspecsL = np.concatenate((np.copy(vivoSpecsONL), np.copy(vivoSpecsOFFL)), axis=0)

TpspecsS = np.concatenate((np.copy(vivoSpecsONS), np.copy(vivoSpecsOFFS)), axis=0)
TpspecsM = np.concatenate((np.copy(vivoSpecsONM), np.copy(vivoSpecsOFFM)), axis=0)
TpspecsL = np.concatenate((np.copy(vivoSpecsONL), np.copy(vivoSpecsOFFL)), axis=0)

# convert to time domain FID
CRfidsS, CRfidsM, CRfidsL = toFids(CRspecsS, 1), toFids(CRspecsM, 1), toFids(CRspecsL, 1)
MafidsS, MafidsM, MafidsL = toFids(MaspecsS, 1), toFids(MaspecsM, 1), toFids(MaspecsL, 1)
TpfidsS, TpfidsM, TpfidsL = toFids(TpspecsS, 1), toFids(TpspecsM, 1), toFids(TpspecsL, 1)

# apply frequency and phase correction
CRfidsS = corrFShift(CRfidsS, vivoTIME, CRfreqLabelsS)
CRfidsS = corrPShift(CRfidsS, CRphaseLabelsS)
CRfidsM = corrFShift(CRfidsM, vivoTIME, CRfreqLabelsM)
CRfidsM = corrPShift(CRfidsM, CRphaseLabelsM)
CRfidsL = corrFShift(CRfidsL, vivoTIME, CRfreqLabelsL)
CRfidsL = corrPShift(CRfidsL, CRphaseLabelsL)

MafidsS = corrFShift(MafidsS, vivoTIME, MafreqLabelsS)
MafidsS = corrPShift(MafidsS, MaphaseLabelsS)
MafidsM = corrFShift(MafidsM, vivoTIME, MafreqLabelsM)
MafidsM = corrPShift(MafidsM, MaphaseLabelsM)
MafidsL = corrFShift(MafidsL, vivoTIME, MafreqLabelsL)
MafidsL = corrPShift(MafidsL, MaphaseLabelsL)

TpfidsS = corrFShift(TpfidsS, vivoTIME, TpfreqLabelsS)
TpfidsS = corrPShift(TpfidsS, TpphaseLabelsS)
TpfidsM = corrFShift(TpfidsM, vivoTIME, TpfreqLabelsM)
TpfidsM = corrPShift(TpfidsM, TpphaseLabelsM)
TpfidsL = corrFShift(TpfidsL, vivoTIME, TpfreqLabelsL)
TpfidsL = corrPShift(TpfidsL, TpphaseLabelsL)

# convert to frequency domain SPECS
CRspecsSFinal, CRspecsMFinal, CRspecsLFinal = toSpecs(CRfidsS, 1), toSpecs(CRfidsM, 1), toSpecs(CRfidsL, 1)
MaspecsSFinal, MaspecsMFinal, MaspecsLFinal = toSpecs(MafidsS, 1), toSpecs(MafidsM, 1), toSpecs(MafidsL, 1)
TpspecsSFinal, TpspecsMFinal, TpspecsLFinal = toSpecs(TpfidsS, 1), toSpecs(TpfidsM, 1), toSpecs(TpfidsL, 1)

########################################################################################################################
# reform scans and calculate mean specs (ON=1, OFF=0)
########################################################################################################################
CRspecsSmall_scans, CRspecsMed_scans, CRspecsLarge_scans = reformScans(CRspecsSFinal, CRspecsMFinal, CRspecsLFinal)
CRspecsSmall_Mscans, CRspecsMed_Mscans, CRspecsLarge_Mscans = meanSpec(CRspecsSmall_scans, CRspecsMed_scans, CRspecsLarge_scans)

MaspecsSmall_scans, MaspecsMed_scans, MaspecsLarge_scans = reformScans(MaspecsSFinal, MaspecsMFinal, MaspecsLFinal)
MaspecsSmall_Mscans, MaspecsMed_Mscans, MaspecsLarge_Mscans = meanSpec(MaspecsSmall_scans, MaspecsMed_scans, MaspecsLarge_scans)

TpspecsSmall_scans, TpspecsMed_scans, TpspecsLarge_scans = reformScans(TpspecsSFinal, TpspecsMFinal, TpspecsLFinal)
TpspecsSmall_Mscans, TpspecsMed_Mscans, TpspecsLarge_Mscans = meanSpec(TpspecsSmall_scans, TpspecsMed_scans, TpspecsLarge_scans)

########################################################################################################################
# metric outputs
########################################################################################################################
setName = ["compReal", "Ma", "Tapper"]
getMetricsSignificance(CRspecsSmall_Mscans.real, MaspecsSmall_Mscans.real, TpspecsSmall_Mscans.real, vivoPPM, setName, sizeName="Small")
getMetricsSignificance(CRspecsMed_Mscans.real, MaspecsMed_Mscans.real, TpspecsMed_Mscans.real, vivoPPM, setName, sizeName="Medium")
getMetricsSignificance(CRspecsLarge_Mscans.real, MaspecsLarge_Mscans.real, TpspecsLarge_Mscans.real, vivoPPM, setName, sizeName="Large")

# preview data
for iii in range(0, 32):
    fig1, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.set_title("No Artificial Offsets")
    ax1.plot(vivoPPM, (vivoSpecsONN[160*(iii):160*(iii+1), :]-vivoSpecsOFFN[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'black')
    ax2.set_title("Small Offsets")
    ax2.plot(vivoPPM, (vivoSpecsONS[160*(iii):160*(iii+1), :]-vivoSpecsOFFS[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'orange')
    ax2.plot(vivoPPM, (vivoSpecsONS[160*(iii):160*(iii+1), :]-vivoSpecsOFFS[160*(iii):160*(iii+1), :]).mean(axis=0).real-0.01, 'blue')
    ax2.plot(vivoPPM, (vivoSpecsONS[160*(iii):160*(iii+1), :]-vivoSpecsOFFS[160*(iii):160*(iii+1), :]).mean(axis=0).real+0.01, 'green')
    ax3.set_title("Small Offsets compReal CORRECTED")
    ax3.plot(vivoPPM, CRspecsSmall_Mscans[iii, :].real, 'orange')
    ax3.plot(vivoPPM, MaspecsSmall_Mscans[iii, :].real-2000, 'blue')
    ax3.plot(vivoPPM, TpspecsSmall_Mscans[iii, :].real+2000, 'green')
    plt.show()

    fig2, (ax11, ax22, ax33) = plt.subplots(3)
    ax11.set_title("No Artificial Offsets")
    ax11.plot(vivoPPM, (vivoSpecsONN[160*(iii):160*(iii+1), :]-vivoSpecsOFFN[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'black')
    ax22.set_title("Medium Offsets")
    ax22.plot(vivoPPM, (vivoSpecsONM[160*(iii):160*(iii+1), :]-vivoSpecsOFFM[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'orange')
    ax22.plot(vivoPPM, (vivoSpecsONM[160*(iii):160*(iii+1), :]-vivoSpecsOFFM[160*(iii):160*(iii+1), :]).mean(axis=0).real-0.01, 'blue')
    ax22.plot(vivoPPM, (vivoSpecsONM[160*(iii):160*(iii+1), :]-vivoSpecsOFFM[160*(iii):160*(iii+1), :]).mean(axis=0).real+0.01, 'green')
    ax33.set_title("Medium Offsets compReal CORRECTED")
    ax33.plot(vivoPPM, CRspecsMed_Mscans[iii, :].real, 'orange')
    ax33.plot(vivoPPM, MaspecsMed_Mscans[iii, :].real-2000, 'blue')
    ax33.plot(vivoPPM, TpspecsMed_Mscans[iii, :].real+2000, 'green')
    plt.show()

    fig3, (ax111, ax222, ax333) = plt.subplots(3)
    ax111.set_title("No Artificial Offsets")
    ax111.plot(vivoPPM, (vivoSpecsONN[160*(iii):160*(iii+1), :]-vivoSpecsOFFN[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'black')
    ax222.set_title("Large Offsets")
    ax222.plot(vivoPPM, (vivoSpecsONL[160*(iii):160*(iii+1), :]-vivoSpecsOFFL[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'blue')
    ax333.set_title("Large Offsets compReal CORRECTED")
    ax333.plot(vivoPPM, CRspecsLarge_Mscans[iii, :].real, 'orange')
    ax333.plot(vivoPPM, MaspecsLarge_Mscans[iii, :].real-2000, 'blue')
    ax333.plot(vivoPPM, TpspecsLarge_Mscans[iii, :].real+2000, 'green')
    plt.show()

    fig4, (ax111, ax222, ax333) = plt.subplots(3)
    ax111.set_title("compReal Corrections")
    ax111.plot(vivoPPM, CRspecsSmall_Mscans[iii, :].real-2000, 'black')
    ax111.plot(vivoPPM, CRspecsMed_Mscans[iii, :].real, 'purple')
    ax111.plot(vivoPPM, CRspecsLarge_Mscans[iii, :].real+2000, 'red')
    ax222.set_title("Ma Corrections")
    ax222.plot(vivoPPM, MaspecsSmall_Mscans[iii, :].real-2000, 'black')
    ax222.plot(vivoPPM, MaspecsMed_Mscans[iii, :].real, 'purple')
    ax222.plot(vivoPPM, MaspecsLarge_Mscans[iii, :].real+2000, 'red')
    ax333.set_title("Tapper Corrections")
    ax333.plot(vivoPPM, TpspecsSmall_Mscans[iii, :].real-2000, 'black')
    ax333.plot(vivoPPM, TpspecsMed_Mscans[iii, :].real, 'purple')
    ax333.plot(vivoPPM, TpspecsLarge_Mscans[iii, :].real+2000, 'red')
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
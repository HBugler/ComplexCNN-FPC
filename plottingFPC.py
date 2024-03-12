import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error as mae
import math
import random

def corrData(spectra, time, freqLabels, phaseLabels, freqCorr=False, phaseCorr=False):
    corrSpecs = np.zeros(shape=spectra.shape)
    fids = np.fft.fft(np.fft.fftshift(spectra, axes=1), axis=1)

    if freqCorr==True:
        for k in range(0, freqLabels.shape[0]):
            fids[k, :] = fids[k, :] * np.exp(1j * time[:] * (-freqLabels[k]) * 2 * math.pi)

    if phaseCorr == True:
        for k in range(0, phaseLabels.shape[0]):
            fids[k, :] = fids[k, :] * np.exp(1j * (-phaseLabels[:, k]) * math.pi / 180)

    specs = np.fft.fftshift(np.fft.ifft(fids, axis=1), axes=1)
    corrSpecs[0, :, :], corrSpecs[1, :, :] = specs.real, specs.imag

    return corrSpecs


# Figure 3A and 3B: Spectra Reconstruction
def plotFig3A(ppm, smallOffsets_CV_CNN, mediumOffsets_CV_CNN, largeOffsets_CV_CNN): #requires corrected data

    # Figure 3A: Reconstructions Separated by Offsets
    fig1, (ax1, ax2, ax3) = plt.subplots(3)
    fig1.suptitle("CR-CNN Frequency and Phase Corrected In Vivo Spectra")

    ax1. set_title("Small Additional Offsets")
    for scanS in range(0, smallOffsets_CV_CNN.shape[0]):
        ax1.plot(ppm, smallOffsets_CV_CNN[scanS, :], alpha=0.25, color='cadetblue')
    ax1.set_xlabel('ppm')
    ax1.invert_xaxis()
    ax1.set_xlim(1, 5)

    ax2. set_title("Medium Additional Offsets")
    for scanM in range(0, mediumOffsets_CV_CNN.shape[0]):
        ax2.plot(ppm, mediumOffsets_CV_CNN[scanM, :], alpha=0.25, color='darkcyan')
    ax2.set_xlabel('ppm')
    ax2.invert_xaxis()
    ax2.set_xlim(1, 5)

    ax3. set_title("Large Additional Offsets")
    for scanL in range(0, largeOffsets_CV_CNN.shape[0]):
        ax3.plot(ppm, largeOffsets_CV_CNN[scanL, :], alpha=0.25, color='darkslategrey')
    ax3.set_xlabel('ppm')
    ax3.invert_xaxis()
    ax3.set_xlim(1, 5)

    plt.show()


def plotFig3B(ppm, offsetsCompRealS, offsetsMaS, offsetsCompTapperS,
              offsetsCompRealM, offsetsMaM, offsetsCompTapperM,
              offsetsCompRealL, offsetsMaL, offsetsCompTapperL,
              offsetsUncorrS, offsetsUncorrM, offsetsUncorrL):

    randS, randM, randL = random.randint(0, 29), random.randint(0, 29), random.randint(0, 29)

    for i in range(0, 29):
        randS, randM, randL = i, i, i
            # Figure 3B: Selected Reconstructions by Different Models
        fig1, (ax1, ax2, ax3) = plt.subplots(3)
        fig1.suptitle("Sample Mean Spectrum of Frequency and Phase Corrected In Vivo Scan")

        ax1.set_title("Small Additional Offsets")
        ax1.set_xlabel('ppm')
        ax1.invert_xaxis()
        ax1.set_xlim(1, 5)
        ax1.plot(ppm, offsetsUncorrS[randS, :], 'black', alpha=0.75, label='Uncorr', linewidth=3)
        ax1.plot(ppm, offsetsCompTapperS[randS, :] + 0.000005, 'green', alpha=0.75, label='MLP', linewidth=3)
        ax1.plot(ppm, offsetsMaS[randS, :] + 0.0000025, 'blue', alpha=0.75, label='CNN', linewidth=3)
        ax1.plot(ppm, offsetsCompRealS[randS, :], 'orange', alpha=0.75, label='CR-CNN', linewidth=3)

        ax2.set_title("Medium Additional Offsets")
        ax2.set_xlabel('ppm')
        ax2.invert_xaxis()
        ax2.set_xlim(1, 5)
        ax2.plot(ppm, offsetsUncorrM[randS, :], 'black', alpha=0.75, label='Uncorr', linewidth=3)
        ax2.plot(ppm, offsetsCompTapperM[randM, :] + 0.000005, 'green', alpha=0.75, label='MLP', linewidth=3)
        ax2.plot(ppm, offsetsMaM[randM, :] + 0.0000025, 'blue', alpha=0.75, label='CNN', linewidth=3)
        ax2.plot(ppm, offsetsCompRealM[randM, :], 'orange', alpha=0.75, label='CR-CNN', linewidth=3)

        ax3.set_title("Large Additional Offsets")
        ax3.set_xlabel('ppm')
        ax3.invert_xaxis()
        ax3.set_xlim(1, 5)
        ax3.plot(ppm, offsetsUncorrL[randS, :], 'black', alpha=0.75, label='Uncorr', linewidth=3)
        ax3.plot(ppm, offsetsCompTapperL[randL, :] + 0.000005, 'green', alpha=0.75, label='MLP', linewidth=3)
        ax3.plot(ppm, offsetsMaL[randL, :] + 0.0000025, 'blue', alpha=0.75, label='CNN', linewidth=3)
        ax3.plot(ppm, offsetsCompRealL[randL, :], 'orange', alpha=0.75, label='CR-CNN', linewidth=3)

        plt.legend()
        plt.show()


def qMetricPlot(ppm, smallOffsets_M1, mediumOffsets_M1, largeOffsets_M1, smallOffsets_M2, mediumOffsets_M2, largeOffsets_M2):
    ppm = np.ndarray.round(ppm, 2)
    ind_close, ind_far = np.amax(np.where(ppm == 3.15)), np.amin(np.where(ppm == 3.28))

    both = (np.concatenate((smallOffsets_M1, mediumOffsets_M1, largeOffsets_M1, smallOffsets_M2, mediumOffsets_M2, largeOffsets_M2)))
    both_RMax = np.max(both.real)
    both_RMin = np.min(both.real)
    smallOffsets_M1 = ((smallOffsets_M1.real - both_RMin) / (both_RMax - both_RMin))
    mediumOffsets_M1 = ((mediumOffsets_M1.real - both_RMin) / (both_RMax - both_RMin))
    largeOffsets_M1 = ((largeOffsets_M1.real - both_RMin) / (both_RMax - both_RMin))
    smallOffsets_M2 = ((smallOffsets_M2.real - both_RMin) / (both_RMax - both_RMin))
    mediumOffsets_M2 = ((mediumOffsets_M2.real - both_RMin) / (both_RMax - both_RMin))
    largeOffsets_M2 = ((largeOffsets_M2.real - both_RMin) / (both_RMax - both_RMin))

    smallOffsets_M1 = smallOffsets_M1[:, ind_close:ind_far]
    varSmall_M1 = np.var(smallOffsets_M1, axis=1)

    mediumOffsets_M1 = mediumOffsets_M1[:, ind_close:ind_far]
    varMed_M1 = np.var(mediumOffsets_M1, axis=1)

    largeOffsets_M1 = largeOffsets_M1[:, ind_close:ind_far]
    varLarge_M1 = np.var(largeOffsets_M1, axis=1)

    smallOffsets_M2 = smallOffsets_M2[:, ind_close:ind_far]
    varSmall_M2 = np.var(smallOffsets_M2, axis=1)

    mediumOffsets_M2 = mediumOffsets_M2[:, ind_close:ind_far]
    varMed_M2 = np.var(mediumOffsets_M2, axis=1)

    largeOffsets_M2 = largeOffsets_M2[:, ind_close:ind_far]
    varLarge_M2 = np.var(largeOffsets_M2, axis=1)

    qSmall, qMed, qLarge = np.zeros(varSmall_M1.shape[0]), np.zeros(varSmall_M1.shape[0]), np.zeros(varSmall_M1.shape[0])

    fig3, (ax1, ax2, ax3) = plt.subplots(3)
    fig3.suptitle("In Vivo Choline Artifact CV-CNN vs. CNN")

    ax1.set_title("Small Additional Offsets")
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Q Value')

    ax2.set_title("Medium Additional Offsets")
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Q Value')

    ax3.set_title("Large Additional Offsets")
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Q Value')

    for i in range(0, varSmall_M1.shape[0]):
        qSmall[i] = (1 - varSmall_M1[i]) / (varSmall_M1[i] + varSmall_M2[i])
        qMed[i] = (1 - varMed_M1[i]) / (varMed_M1[i] + varMed_M2[i])
        qLarge[i] = (1 - varLarge_M1[i]) / (varLarge_M1[i] + varLarge_M2[i])

        if (qSmall[i] >= 0.5):
            ax1.plot(i, qSmall[i], 'blue', label='CR-CNN > CNN')
        else:
            ax1.plot(i, qSmall[i], 'red', label='CR-CNN < CNN')

        if (qMed[i] >= 0.5):
            ax1.plot(i, qMed[i], 'blue', label='CR-CNN > CNN')
        else:
            ax1.plot(i, qMed[i], 'red', label='CR-CNN < CNN')

        if (qLarge[i] >= 0.5):
            ax1.plot(i, qLarge[i], 'blue', label='CR-CNN > CNN')
        else:
            ax1.plot(i, qLarge[i], 'red', label='CR-CNN < CNN')

    plt.legend()
    plt.show()


def plotFig4(smallSNR_M1, medSNR_M1, largeSNR_M1,
         smallSNR_M2, medSNR_M2, largeSNR_M2,
         smallSNR_M3, medSNR_M3, largeSNR_M3,
         smallLW_M1, medLW_M1, largeLW_M1,
         smallLW_M2, medLW_M2, largeLW_M2,
         smallLW_M3, medLW_M3, largeLW_M3,
             smallLW_Std_M1, smallLW_Std_M2, smallLW_Std_M3,
             medLW_Std_M1, medLW_Std_M2, medLW_Std_M3,
             largeLW_Std_M1, largeLW_Std_M2, largeLW_Std_M3,
             smallSNR_Std_M1, smallSNR_Std_M2, smallSNR_Std_M3,
             medSNR_Std_M1, medSNR_Std_M2, medSNR_Std_M3,
             largeSNR_Std_M1, largeSNR_Std_M2, largeSNR_Std_M3):

    models = ["CR-CNN", "CNN", "MLP"]

    fig1, axs = plt.subplots(4, 2)
    fig1.suptitle("In Vivo Quality Metrics Analysis")

    axs[0,0].set_title("GABA Linewidth - No Offsets")
    list00 = axs[0,0].bar(models, [noLW_M1, noLW_M2, noLW_M3])
    axs[0,0].errorbar(models, [noLW_M1, noLW_M2, noLW_M3], yerr=[noLW_Std_M1, noLW_Std_M2, noLW_Std_M3], fmt='o', color='black')
    list00[0].set_color('orange'), list00[1].set_color('blue'), list00[2].set_color('green')

    axs[1,0].set_title("GABA Linewidth - Small Offsets")
    list10 = axs[1,0].bar(models, [smallLW_M1, smallLW_M2, smallLW_M3])
    axs[1,0].errorbar(models, [smallLW_M1, smallLW_M2, smallLW_M3], yerr=[smallLW_Std_M1, smallLW_Std_M2, smallLW_Std_M3], fmt='o', color='black')
    list10[0].set_color('orange'), list10[1].set_color('blue'), list10[2].set_color('green')

    axs[2,0].set_title("GABA Linewidth - Medium Offsets")
    list20 = axs[2,0].bar(models, [medLW_M1, medLW_M2, medLW_M3])
    axs[2,0].errorbar(models, [medLW_M1, medLW_M2, medLW_M3], yerr=[medLW_Std_M1, medLW_Std_M2, medLW_Std_M3], fmt='o', color='black')
    list20[0].set_color('orange'), list20[1].set_color('blue'), list20[2].set_color('green')

    axs[3,0].set_title("GABA Linewidth - Large Offsets")
    list30 = axs[3,0].bar(models, [largeLW_M1, largeLW_M2, largeLW_M3])
    axs[3,0].errorbar(models, [largeLW_M1, largeLW_M2, largeLW_M3], yerr=[largeLW_Std_M1, largeLW_Std_M2, largeLW_Std_M3], fmt='o', color='black')
    list30[0].set_color('orange'), list30[1].set_color('blue'), list30[2].set_color('green')

    axs[0,1].set_title("GABA SNR - No Offsets")
    list01 =axs[0,1].bar(models, [noSNR_M1, noSNR_M2, noSNR_M3])
    axs[0,1].errorbar(models, [noSNR_M1, noSNR_M2, noSNR_M3], yerr=[noSNR_Std_M1, noSNR_Std_M2, noSNR_Std_M3], fmt='o', color='black')
    list01[0].set_color('orange'), list01[1].set_color('blue'), list01[2].set_color('green')

    axs[1,1].set_title("GABA SNR - Small Offsets")
    list11 =axs[1,1].bar(models, [smallSNR_M1, smallSNR_M2, smallSNR_M3])
    axs[1,1].errorbar(models, [smallSNR_M1, smallSNR_M2, smallSNR_M3], yerr=[smallSNR_Std_M1, smallSNR_Std_M2, smallSNR_Std_M3], fmt='o', color='black')
    list11[0].set_color('orange'), list11[1].set_color('blue'), list11[2].set_color('green')

    axs[2,1].set_title("GABA SNR - Medium Offsets")
    list21 =axs[2,1].bar(models, [medSNR_M1, medSNR_M2, medSNR_M3])
    axs[2,1].errorbar(models, [medSNR_M1, medSNR_M2, medSNR_M3], yerr=[medSNR_Std_M1, medSNR_Std_M2, medSNR_Std_M3], fmt='o', color='black')
    list21[0].set_color('orange'), list21[1].set_color('blue'), list21[2].set_color('green')

    axs[3,1].set_title("GABA SNR - Large Offsets")
    list31 =axs[3,1].bar(models, [largeSNR_M1, largeSNR_M2, largeSNR_M3])
    axs[3,1].errorbar(models, [largeSNR_M1, largeSNR_M2, largeSNR_M3], yerr=[largeSNR_Std_M1, largeSNR_Std_M2, largeSNR_Std_M3], fmt='o', color='black')
    list31[0].set_color('orange'), list31[1].set_color('blue'), list31[2].set_color('green')

    plt.show()


def significance():
    allnets, allmodels, allwaters, allsnrs = [], [], [], []
    allMAES1, allMAES2, allMAES3 = [], [], []
    allpvalueNorm1, allpvalueNorm2, allpvalueNorm3 = [], [], []
    allpvalueW1, allpvalueW2, allpvalueW3 = [], [], []

    # load data
    labelsLocation = "C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/"
    netType = ["freq", "phase"]
    snrType = ["10", "5", "2_5"]
    waterType = ["None", "Pos", "Neg", "Mix"]
    modelType = ["realReal", "compReal", "compComp"]

    for water in waterType:
        indW = waterType.index(water)
        for snr in snrType:
            indS = snrType.index(snr)
            for net in netType:
                indN = netType.index(net)
                print(f'for {waterType[indW]}, {snrType[indS]}, {netType[indN]}')

                model1 = np.load(
                    f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/PredLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[0]}.npy")
                model2 = np.load(
                    f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/PredLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[1]}.npy")
                model3 = np.load(
                    f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/PredLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[2]}.npy")
                trueVals1 = np.load(
                    f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/TrueLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[0]}.npy")
                trueVals2 = np.load(
                    f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/TrueLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[1]}.npy")
                trueVals3 = np.load(
                    f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/TrueLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[2]}.npy")

                allnets.append(netType[indN]), allsnrs.append(snrType[indS]), allwaters.append(waterType[indW])

                # sort data
                OG_Sort1, OG_Sort2, OG_Sort3 = np.copy(trueVals1), np.copy(trueVals2), np.copy(trueVals3)
                trueVals1.sort(), trueVals2.sort(), trueVals3.sort()

                if (trueVals3 == trueVals1).all() and (trueVals2 == trueVals1).all() and (trueVals2 == trueVals3).all():
                    order1, order2, order3 = np.argsort(OG_Sort1), np.argsort(OG_Sort2), np.argsort(OG_Sort3)
                    sort1 = order1[np.searchsorted(OG_Sort1[order1], trueVals1)]
                    sort2 = order2[np.searchsorted(OG_Sort2[order2], trueVals2)]
                    sort3 = order3[np.searchsorted(OG_Sort3[order3], trueVals3)]

                    model1 = model1[sort1]
                    model2 = model2[sort2]
                    model3 = model3[sort3]

                    # recalculate Abs Errors
                    error1, error2, error3 = np.zeros(model1.shape), np.zeros(model2.shape), np.zeros(model3.shape)

                    for i in range(model1.shape[0]):
                        error1[i] = abs(trueVals1[i] - model1[i])
                        error2[i] = abs(trueVals2[i] - model2[i])
                        error3[i] = abs(trueVals3[i] - model3[i])

                    MAE1 = mae(trueVals1, model1)
                    MAE2 = mae(trueVals2, model2)
                    MAE3 = mae(trueVals3, model3)
                    allMAES1.append(MAE1), allMAES2.append(MAE2), allMAES3.append(MAE3)

                    # check for normality
                    statNorm1, pvalueNorm1 = stats.normaltest(error1, axis=0)
                    statNorm2, pvalueNorm2 = stats.normaltest(error2, axis=0)
                    statNorm3, pvalueNorm3 = stats.normaltest(error3, axis=0)
                    allpvalueNorm1.append(pvalueNorm1), allpvalueNorm2.append(pvalueNorm2), allpvalueNorm3.append(
                        pvalueNorm3)

                    # perform Wilcoxon signed rank test
                    statW1, pvalueW1 = stats.wilcoxon(error1, error2)
                    statW2, pvalueW2 = stats.wilcoxon(error1, error3)
                    statW3, pvalueW3 = stats.wilcoxon(error2, error3)
                    allpvalueW1.append(pvalueW1), allpvalueW2.append(pvalueW2), allpvalueW3.append(pvalueW3)
                else:
                    print(f'GOT FALSE!')

    # Create dataframe with results
    significanceFrame = pd.DataFrame({
        'Net Type': allnets,
        'SNR Type': allsnrs,
        'Water Type': allwaters,
        # 'Model Type': allmodels,

        f'MAE model1 {modelType[0]}': allMAES1,
        f'MAE model2 {modelType[1]}': allMAES2,
        f'MAE model3 {modelType[2]}': allMAES3,

        'Normality Test pvalue model1': allpvalueNorm1,
        'Normality Test pvalue model2': allpvalueNorm2,
        'Normality Test pvalue model3': allpvalueNorm3,

        'Wilcoxon Test pvalue (model1 vs. model2)': allpvalueW1,
        'Wilcoxon Test pvalue (model1 vs. model3)': allpvalueW2,
        'Wilcoxon Test pvalue (model2 vs. model3)': allpvalueW3})

    significanceFrame.to_csv(f"AblationStudySignificance_FPC2024.csv", index=False)

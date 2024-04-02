import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error as mae
import math
import random
from FPC_Functions import getMetricsSignificance

def previewData(vivoPPM, vivoSpecsONN, vivoSpecsOFFN, vivoSpecsONS, vivoSpecsOFFS,
                vivoSpecsONM, vivoSpecsOFFM, vivoSpecsONL, vivoSpecsOFFL,
                CRspecsSmall_Mscans, MaspecsSmall_Mscans, TpspecsSmall_Mscans,
                CRspecsMed_Mscans, MaspecsMed_Mscans, TpspecsMed_Mscans,
                CRspecsLarge_Mscans, MaspecsLarge_Mscans, TpspecsLarge_Mscans): #requires corrected data

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


# Figure 3A and 3B: Spectra Reconstruction
def plotFig3A(ppm, smallOffsets_CV_CNN, mediumOffsets_CV_CNN, largeOffsets_CV_CNN): #requires corrected data

    # Figure 3A: Reconstructions Separated by Offsets
    fig1, (ax1, ax2, ax3) = plt.subplots(3)
    fig1.suptitle("CR-CNN Frequency and Phase Corrected In Vivo Spectra")

    ax1. set_title("Small Additional Offsets")
    for scanS in range(0, smallOffsets_CV_CNN.shape[0]):
        ax1.plot(ppm, smallOffsets_CV_CNN[scanS, :].real, alpha=0.25, color='cadetblue')
    ax1.set_xlabel('ppm')
    ax1.get_yaxis().set_visible(False)
    ax1.set_xlim(1.5, 5)
    ax1.invert_xaxis()

    ax2. set_title("Medium Additional Offsets")
    for scanM in range(0, mediumOffsets_CV_CNN.shape[0]):
        ax2.plot(ppm, mediumOffsets_CV_CNN[scanM, :].real, alpha=0.25, color='darkcyan')
    ax2.set_xlabel('ppm')
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlim(1.5, 5)
    ax2.invert_xaxis()

    ax3. set_title("Large Additional Offsets")
    for scanL in range(0, largeOffsets_CV_CNN.shape[0]):
        ax3.plot(ppm, largeOffsets_CV_CNN[scanL, :].real, alpha=0.25, color='darkslategrey')
    ax3.set_xlabel('ppm')
    ax3.get_yaxis().set_visible(False)
    ax3.set_xlim(1.5, 5)
    ax3.invert_xaxis()

    plt.show()


def plotFig3B(setName, ppm, offsetsCompRealS, offsetsMaS, offsetsCompTapperS,
              offsetsCompRealM, offsetsMaM, offsetsCompTapperM,
              offsetsCompRealL, offsetsMaL, offsetsCompTapperL,
              offsetsUncorrS, offsetsUncorrM, offsetsUncorrL):

    randScan = random.randint(0, 29)

    for i in range(0, 29):
        randS, randM, randL = i, i, i
            # Figure 3B: Selected Reconstructions by Different Models
        fig1, (ax1, ax2, ax3) = plt.subplots(3)
        fig1.suptitle("Sample In Vivo Scan Reconstruction after Frequency and Phase Correction")

        ax1.set_title("In Vivo Data with Small Artificial Offsets")
        ax1.set_xlabel('ppm')
        ax1.set_xlim(1.5, 5)
        ax1.plot(ppm, offsetsUncorrS[randS, :].real-2000, 'black', alpha=0.75, label='Uncorr', linewidth=3)
        ax1.plot(ppm, offsetsCompTapperS[randS, :].real+2000, 'green', alpha=0.75, label=setName[2], linewidth=3)
        ax1.plot(ppm, offsetsMaS[randS, :].real+4000, 'blue', alpha=0.75, label=setName[1], linewidth=3)
        ax1.plot(ppm, offsetsCompRealS[randS, :].real, 'orange', alpha=0.75, label=setName[0], linewidth=3)
        ax1.get_yaxis().set_visible(False)
        ax1.invert_xaxis()

        ax2.set_title("In Vivo Data with Medium Artificial Offsets")
        ax2.set_xlabel('ppm')
        ax2.set_xlim(1.5, 5)
        ax2.plot(ppm, offsetsUncorrM[randS, :].real-2000, 'black', alpha=0.75, label='Uncorr', linewidth=3)
        ax2.plot(ppm, offsetsCompTapperM[randM, :].real+2000, 'green', alpha=0.75, label=setName[2], linewidth=3)
        ax2.plot(ppm, offsetsMaM[randM, :].real+4000, 'blue', alpha=0.75, label=setName[1], linewidth=3)
        ax2.plot(ppm, offsetsCompRealM[randM, :].real, 'orange', alpha=0.75, label=setName[0], linewidth=3)
        ax2.get_yaxis().set_visible(False)
        ax2.invert_xaxis()

        ax3.set_title("In Vivo Data with Large Artificial Offsets")
        ax3.set_xlabel('ppm')
        ax3.set_xlim(1.5, 5)
        ax3.plot(ppm, offsetsUncorrL[randS, :].real-2000, 'black', alpha=0.75, label='Uncorr', linewidth=3)
        ax3.plot(ppm, offsetsCompTapperL[randL, :].real+2000, 'green', alpha=0.75, label=setName[2], linewidth=3)
        ax3.plot(ppm, offsetsMaL[randL, :].real+4000, 'blue', alpha=0.75, label=setName[1], linewidth=3)
        ax3.plot(ppm, offsetsCompRealL[randL, :].real, 'orange', alpha=0.75, label=setName[0], linewidth=3)
        ax3.get_yaxis().set_visible(False)
        ax3.invert_xaxis()

        plt.legend()
        plt.show()


def qMetricPlot(setName, ppm, smallOffsets_M1, mediumOffsets_M1, largeOffsets_M1,
                smallOffsets_M2, mediumOffsets_M2, largeOffsets_M2,
                smallOffsets_M3, mediumOffsets_M3, largeOffsets_M3):

    finish, start = np.where(ppm <= 3.15)[0][0], np.where(ppm >= 3.28)[0][-1]

    smallOffsets_M1 = smallOffsets_M1[:, start:finish]
    varSmall_M1 = np.var(smallOffsets_M1, axis=1)

    mediumOffsets_M1 = mediumOffsets_M1[:, start:finish]
    varMed_M1 = np.var(mediumOffsets_M1, axis=1)

    largeOffsets_M1 = largeOffsets_M1[:, start:finish]
    varLarge_M1 = np.var(largeOffsets_M1, axis=1)

    smallOffsets_M2 = smallOffsets_M2[:, start:finish]
    varSmall_M2 = np.var(smallOffsets_M2, axis=1)

    mediumOffsets_M2 = mediumOffsets_M2[:, start:finish]
    varMed_M2 = np.var(mediumOffsets_M2, axis=1)

    largeOffsets_M2 = largeOffsets_M2[:, start:finish]
    varLarge_M2 = np.var(largeOffsets_M2, axis=1)

    smallOffsets_M3 = smallOffsets_M3[:, start:finish]
    varSmall_M3 = np.var(smallOffsets_M3, axis=1)

    mediumOffsets_M3 = mediumOffsets_M3[:, start:finish]
    varMed_M3 = np.var(mediumOffsets_M3, axis=1)

    largeOffsets_M3 = largeOffsets_M3[:, start:finish]
    varLarge_M3 = np.var(largeOffsets_M3, axis=1)

    qSmall12, qMed12, qLarge12 = np.zeros(varSmall_M1.shape[0]), np.zeros(varSmall_M1.shape[0]), np.zeros(varSmall_M1.shape[0])
    qSmall13, qMed13, qLarge13 = np.zeros(varSmall_M1.shape[0]), np.zeros(varSmall_M1.shape[0]), np.zeros(varSmall_M1.shape[0])

    fig3, axs = plt.subplots(2, 3)
    fig3.suptitle("In Vivo Choline Artifact CV-CNN vs. CNN")

    axs[0,0].set_title("Small Additional Offsets")
    axs[0,0].set_xlabel('Scan')
    axs[0,0].set_ylabel('Q Value')
    axs[0,0].plot([-1, 37], [0.5, 0.5], "k--")
    axs[0,0].set_xlim(-1, 37)

    axs[0,1].set_title("Medium Additional Offsets")
    axs[0,1].set_xlabel('Scan')
    axs[0,1].set_ylabel('Q Value')
    axs[0,1].plot([-1, 37], [0.5, 0.5], "k--")
    axs[0,1].set_xlim(-1, 37)

    axs[0,2].set_title("Large Additional Offsets")
    axs[0,2].set_xlabel('Scan')
    axs[0,2].set_ylabel('Q Value')
    axs[0,2].plot([-1, 37], [0.5, 0.5], "k--")
    axs[0,2].set_xlim(-1, 37)

    axs[1,0].set_title("Small Additional Offsets")
    axs[1,0].set_xlabel('Scan')
    axs[1,0].set_ylabel('Q Value')
    axs[1,0].plot([-1, 37], [0.5, 0.5], "k--")
    axs[1,0].set_xlim(-1, 37)

    axs[1,1].set_title("Medium Additional Offsets")
    axs[1,1].set_xlabel('Scan')
    axs[1,1].set_ylabel('Q Value')
    axs[1,1].plot([-1, 37], [0.5, 0.5], "k--")
    axs[1,1].set_xlim(-1, 37)

    axs[1,2].set_title("Large Additional Offsets")
    axs[1,2].set_xlabel('Scan')
    axs[1,2].set_ylabel('Q Value')
    axs[1,2].plot([-1, 37], [0.5, 0.5], "k--")
    axs[1,2].set_xlim(-1, 37)

    legend_elements12 = [Line2D([0], [0], marker='o', color='w', label=f'{setName[0]} > {setName[1]}',
                              markerfacecolor='blue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label=f'{setName[0]} < {setName[1]}',
                              markerfacecolor='red', markersize=15)]
    legend_elements13 = [Line2D([0], [0], marker='o', color='w', label=f'{setName[0]} > {setName[2]}',
                              markerfacecolor='blue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label=f'{setName[0]} < {setName[2]}',
                              markerfacecolor='red', markersize=15)]
    axs[0,0].legend(handles=legend_elements12, loc='upper left')
    axs[0,1].legend(handles=legend_elements12, loc='upper left')
    axs[0,2].legend(handles=legend_elements12, loc='upper left')
    axs[1,0].legend(handles=legend_elements13, loc='upper left')
    axs[1,1].legend(handles=legend_elements13, loc='upper left')
    axs[1,2].legend(handles=legend_elements13, loc='upper left')
    
    for i in range(0, varSmall_M1.shape[0]):
        # shouldn't have to use abs()
        qSmall12[i] = np.abs((1 - varSmall_M1[i]) / (varSmall_M1[i] + varSmall_M2[i]))
        qMed12[i] = np.abs((1 - varMed_M1[i]) / (varMed_M1[i] + varMed_M2[i]))
        qLarge12[i] = np.abs((1 - varLarge_M1[i]) / (varLarge_M1[i] + varLarge_M2[i]))

        qSmall13[i] = np.abs((1 - varSmall_M1[i]) / (varSmall_M1[i] + varSmall_M3[i]))
        qMed13[i] = np.abs((1 - varMed_M1[i]) / (varMed_M1[i] + varMed_M3[i]))
        qLarge13[i] = np.abs((1 - varLarge_M1[i]) / (varLarge_M1[i] + varLarge_M3[i]))

        if (qSmall12[i] >= 0.5):
            axs[0,0].plot(i, qSmall12[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > CNN')
        else:
            axs[0,0].plot(i, qSmall12[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < CNN')

        if (qMed12[i] >= 0.5):
            axs[0,1].plot(i, qMed12[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > CNN')
        else:
            axs[0,1].plot(i, qMed12[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < CNN')

        if (qLarge12[i] >= 0.5):
            axs[0,2].plot(i, qLarge12[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > CNN')
        else:
            axs[0,2].plot(i, qLarge12[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < CNN')

        if (qSmall13[i] >= 0.5):
            axs[1,0].plot(i, qSmall13[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > MLP')
        else:
            axs[1,0].plot(i, qSmall13[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < MLP')

        if (qMed13[i] >= 0.5):
            axs[1,1].plot(i, qMed13[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > MLP')
        else:
            axs[1,1].plot(i, qMed13[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < MLP')

        if (qLarge13[i] >= 0.5):
            axs[1,2].plot(i, qLarge13[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > MLP')
        else:
            axs[1,2].plot(i, qLarge13[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < MLP')

    avQS12, stdQS12 = np.mean(qSmall12), np.std(qSmall12)
    avQM12, stdQM12 = np.mean(qMed12), np.std(qMed12)
    avQL12, stdQL12 = np.mean(qLarge12), np.std(qLarge12)
    axs[0,0].text(10, np.min(qSmall12), f"E[Q] = {round(avQS12, 2)} +/- {round(stdQS12, 2)}")
    axs[0,1].text(10, np.min(qMed12), f"E[Q] = {round(avQM12, 2)} +/- {round(stdQM12, 2)}")
    axs[0,2].text(10, np.min(qLarge12), f"E[Q] = {round(avQL12, 2)} +/- {round(stdQL12, 2)}")

    avQS13, stdQS13 = np.mean(qSmall13), np.std(qSmall13)
    avQM13, stdQM13 = np.mean(qMed13), np.std(qMed13)
    avQL13, stdQL13 = np.mean(qLarge13), np.std(qLarge13)
    axs[1,0].text(10, np.min(qSmall13), f"E[Q] = {round(avQS13, 2)} +/- {round(stdQS13, 2)}")
    axs[1,1].text(10, np.min(qMed13), f"E[Q] = {round(avQM13, 2)} +/- {round(stdQM13, 2)}")
    axs[1,2].text(10, np.min(qLarge13), f"E[Q] = {round(avQL13, 2)} +/- {round(stdQL13, 2)}")

    plt.show()


def plotFig4(setName, vivoPPM,
             CRspecsNone_Mscans, MaspecsNone_Mscans, TpspecsNonel_Mscans,
             CRspecsSmall_Mscans, MaspecsSmall_Mscans, TpspecsSmall_Mscans,
             CRspecsMed_Mscans, MaspecsMed_Mscans, TpspecsMed_Mscans,
             CRspecsLarge_Mscans, MaspecsLarge_Mscans, TpspecsLarge_Mscans):

    meanSnrN1, stdSnrN1, meanLwN1, stdLwN1, \
    meanSnrN2, stdSnrN2, meanLwN2, stdLwN2, \
    meanSnrN3, stdSnrN3, meanLwN3, stdLwN3 = getMetricsSignificance(CRspecsNone_Mscans.real, MaspecsNone_Mscans.real, TpspecsNonel_Mscans.real, vivoPPM, setName, sizeName="None")

    meanSnrS1, stdSnrS1, meanLwS1, stdLwS1, \
    meanSnrS2, stdSnrS2, meanLwS2, stdLwS2, \
    meanSnrS3, stdSnrS3, meanLwS3, stdLwS3  = getMetricsSignificance(CRspecsSmall_Mscans.real, MaspecsSmall_Mscans.real, TpspecsSmall_Mscans.real, vivoPPM, setName, sizeName="Small")

    meanSnrM1, stdSnrM1, meanLwM1, stdLwM1, \
    meanSnrM2, stdSnrM2, meanLwM2, stdLwM2, \
    meanSnrM3, stdSnrM3, meanLwM3, stdLwM3  = getMetricsSignificance(CRspecsMed_Mscans.real, MaspecsMed_Mscans.real, TpspecsMed_Mscans.real, vivoPPM, setName, sizeName="Medium")

    meanSnrL1, stdSnrL1, meanLwL1, stdLwL1, \
    meanSnrL2, stdSnrL2, meanLwL2, stdLwL2, \
    meanSnrL3, stdSnrL3, meanLwL3, stdLwL3 = getMetricsSignificance(CRspecsLarge_Mscans.real, MaspecsLarge_Mscans.real, TpspecsLarge_Mscans.real, vivoPPM, setName, sizeName="Large")

    fig1, axs = plt.subplots(2, 4)
    fig1.suptitle("In Vivo Quality Metrics Analysis")

    axs[0,0].set_title("GABA Linewidth - No Offsets")
    list00 = axs[0,0].bar(setName, [meanLwN1, meanLwN2, meanLwN3])
    axs[0,0].errorbar(setName, [meanLwN1, meanLwN2, meanLwN3], yerr=[stdLwN1, stdLwN2, stdLwN3], fmt='o', color='black')
    list00[0].set_color('orange'), list00[1].set_color('blue'), list00[2].set_color('green')
    axs[0,0].set_ylabel('Linewidth (ppm)')

    axs[0,1].set_title("GABA Linewidth - Small Offsets")
    list10 = axs[0,1].bar(setName, [meanLwS1, meanLwS2, meanLwS3])
    axs[0,1].errorbar(setName, [meanLwS1, meanLwS2, meanLwS3], yerr=[stdLwS1, stdLwS2, stdLwS3], fmt='o', color='black')
    list10[0].set_color('orange'), list10[1].set_color('blue'), list10[2].set_color('green')
    axs[0,1].set_ylabel('Linewidth (ppm)')

    axs[0,2].set_title("GABA Linewidth - Medium Offsets")
    list20 = axs[0,2].bar(setName, [meanLwM1, meanLwM2, meanLwM3])
    axs[0,2].errorbar(setName, [meanLwM1, meanLwM2, meanLwM3], yerr=[stdLwM1, stdLwM2, stdLwM3], fmt='o', color='black')
    list20[0].set_color('orange'), list20[1].set_color('blue'), list20[2].set_color('green')
    axs[0,2].set_ylabel('Linewidth (ppm)')

    axs[0,3].set_title("GABA Linewidth - Large Offsets")
    list30 = axs[0,3].bar(setName, [meanLwL1, meanLwL2, meanLwL3])
    axs[0,3].errorbar(setName, [meanLwL1, meanLwL2, meanLwL3], yerr=[stdLwL1, stdLwL2, stdLwL3], fmt='o', color='black')
    list30[0].set_color('orange'), list30[1].set_color('blue'), list30[2].set_color('green')
    axs[0,3].set_ylabel('Linewidth (ppm)')

    axs[1,0].set_title("GABA SNR - No Offsets")
    list01 =axs[1,0].bar(setName, [meanSnrN1, meanSnrN2, meanSnrN3])
    axs[1,0].errorbar(setName, [meanSnrN1, meanSnrN2, meanSnrN3], yerr=[stdSnrN1, stdSnrN2, stdSnrN3], fmt='o', color='black')
    list01[0].set_color('orange'), list01[1].set_color('blue'), list01[2].set_color('green')
    axs[1,0].set_ylabel('SNR (GABA Peak)')

    axs[1,1].set_title("GABA SNR - Small Offsets")
    list11 =axs[1,1].bar(setName, [meanSnrS1, meanSnrS2, meanSnrS3])
    axs[1,1].errorbar(setName, [meanSnrS1, meanSnrS2, meanSnrS3], yerr=[stdSnrS1, stdSnrS2, stdSnrS3], fmt='o', color='black')
    list11[0].set_color('orange'), list11[1].set_color('blue'), list11[2].set_color('green')
    axs[1,1].set_ylabel('SNR (GABA Peak)')

    axs[1,2].set_title("GABA SNR - Medium Offsets")
    list21 =axs[1,2].bar(setName, [meanSnrM1, meanSnrM2, meanSnrM3])
    axs[1,2].errorbar(setName, [meanSnrM1, meanSnrM2, meanSnrM3], yerr=[stdSnrM1, stdSnrM2, stdSnrM3], fmt='o', color='black')
    list21[0].set_color('orange'), list21[1].set_color('blue'), list21[2].set_color('green')
    axs[1,2].set_ylabel('SNR (GABA Peak)')

    axs[1,3].set_title("GABA SNR - Large Offsets")
    list31 =axs[1,3].bar(setName, [meanSnrL1, meanSnrL2, meanSnrL3])
    axs[1,3].errorbar(setName, [meanSnrL1, meanSnrL2, meanSnrL3], yerr=[stdSnrL1, stdSnrL2, stdSnrL3], fmt='o', color='black')
    list31[0].set_color('orange'), list31[1].set_color('blue'), list31[2].set_color('green')
    axs[1,3].set_ylabel('SNR (GABA Peak)')

    plt.show()

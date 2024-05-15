# Based on the publication
# "Frequency and phase correction of GABA-edited magnetic resonance spectroscopy using complex-valued convolutional neural networks"
# (doi: 10.1016/j.mri.2024.05.008) by Hanna Bugler, Rodrigo Berto, Roberto Souza and Ashley Harris (2024)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from metric_calculator import calculate_snr, calculate_linewidth
import random

def previewData(vivoPPM, vivoSpecsONN, vivoSpecsOFFN, vivoSpecsONS, vivoSpecsOFFS,
                vivoSpecsONM, vivoSpecsOFFM, vivoSpecsONL, vivoSpecsOFFL,
                m1_specsSmall_Mscans, m2_specsSmall_Mscans, m3_specsSmall_Mscans,
                m1_specsMed_Mscans, m2_specsMed_Mscans, m3_specsMed_Mscans,
                m1_specsLarge_Mscans, m2_specsLarge_Mscans, m3_specsLarge_Mscans): #requires corrected data

    for iii in range(0, 32):
        fig1, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.set_title("No Artificial Offsets")
        ax1.plot(vivoPPM, (vivoSpecsONN[160*(iii):160*(iii+1), :]-vivoSpecsOFFN[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'black')
        ax2.set_title("Small Offsets")
        ax2.plot(vivoPPM, (vivoSpecsONS[160*(iii):160*(iii+1), :]-vivoSpecsOFFS[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'orange')
        ax2.plot(vivoPPM, (vivoSpecsONS[160*(iii):160*(iii+1), :]-vivoSpecsOFFS[160*(iii):160*(iii+1), :]).mean(axis=0).real-0.01, 'blue')
        ax2.plot(vivoPPM, (vivoSpecsONS[160*(iii):160*(iii+1), :]-vivoSpecsOFFS[160*(iii):160*(iii+1), :]).mean(axis=0).real+0.01, 'green')
        ax3.set_title("Small Offsets compReal CORRECTED")
        ax3.plot(vivoPPM, m1_specsSmall_Mscans[iii, :].real, 'orange')
        ax3.plot(vivoPPM, m2_specsSmall_Mscans[iii, :].real-2000, 'blue')
        ax3.plot(vivoPPM, m3_specsSmall_Mscans[iii, :].real+2000, 'green')
        plt.show()

        fig2, (ax11, ax22, ax33) = plt.subplots(3)
        ax11.set_title("No Artificial Offsets")
        ax11.plot(vivoPPM, (vivoSpecsONN[160*(iii):160*(iii+1), :]-vivoSpecsOFFN[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'black')
        ax22.set_title("Medium Offsets")
        ax22.plot(vivoPPM, (vivoSpecsONM[160*(iii):160*(iii+1), :]-vivoSpecsOFFM[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'orange')
        ax22.plot(vivoPPM, (vivoSpecsONM[160*(iii):160*(iii+1), :]-vivoSpecsOFFM[160*(iii):160*(iii+1), :]).mean(axis=0).real-0.01, 'blue')
        ax22.plot(vivoPPM, (vivoSpecsONM[160*(iii):160*(iii+1), :]-vivoSpecsOFFM[160*(iii):160*(iii+1), :]).mean(axis=0).real+0.01, 'green')
        ax33.set_title("Medium Offsets compReal CORRECTED")
        ax33.plot(vivoPPM, m1_specsMed_Mscans[iii, :].real, 'orange')
        ax33.plot(vivoPPM, m2_specsMed_Mscans[iii, :].real-2000, 'blue')
        ax33.plot(vivoPPM, m3_specsMed_Mscans[iii, :].real+2000, 'green')
        plt.show()

        fig3, (ax111, ax222, ax333) = plt.subplots(3)
        ax111.set_title("No Artificial Offsets")
        ax111.plot(vivoPPM, (vivoSpecsONN[160*(iii):160*(iii+1), :]-vivoSpecsOFFN[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'black')
        ax222.set_title("Large Offsets")
        ax222.plot(vivoPPM, (vivoSpecsONL[160*(iii):160*(iii+1), :]-vivoSpecsOFFL[160*(iii):160*(iii+1), :]).mean(axis=0).real, 'blue')
        ax333.set_title("Large Offsets compReal CORRECTED")
        ax333.plot(vivoPPM, m1_specsLarge_Mscans[iii, :].real, 'orange')
        ax333.plot(vivoPPM, m2_specsLarge_Mscans[iii, :].real-2000, 'blue')
        ax333.plot(vivoPPM, m3_specsLarge_Mscans[iii, :].real+2000, 'green')
        plt.show()

        fig4, (ax111, ax222, ax333) = plt.subplots(3)
        ax111.set_title("compReal Corrections")
        ax111.plot(vivoPPM, m1_specsSmall_Mscans[iii, :].real-2000, 'black')
        ax111.plot(vivoPPM, m1_specsMed_Mscans[iii, :].real, 'purple')
        ax111.plot(vivoPPM, m1_specsLarge_Mscans[iii, :].real+2000, 'red')
        ax222.set_title("CNN Corrections")
        ax222.plot(vivoPPM, m2_specsSmall_Mscans[iii, :].real-2000, 'black')
        ax222.plot(vivoPPM, m2_specsMed_Mscans[iii, :].real, 'purple')
        ax222.plot(vivoPPM, m2_specsLarge_Mscans[iii, :].real+2000, 'red')
        ax333.set_title("MLP Corrections")
        ax333.plot(vivoPPM, m3_specsSmall_Mscans[iii, :].real-2000, 'black')
        ax333.plot(vivoPPM, m3_specsMed_Mscans[iii, :].real, 'purple')
        ax333.plot(vivoPPM, m3_specsLarge_Mscans[iii, :].real+2000, 'red')
        plt.show()


def plotAllScans(ppm, noOffsets, smallOffsets, mediumOffsets, largeOffsets): #requires corrected data

    # Figure 3A: Reconstructions Separated by Offsets
    fig1, ax = plt.subplots(4, 2, gridspec_kw={'width_ratios': [3, 1]})
    fig1.suptitle("CC-CNN Frequency and Phase Corrected In Vivo Spectra")

    ax[0,0]. set_title("No Additional Offsets")
    for scanN in range(0, noOffsets.shape[0]):
        ax[0,0].plot(ppm, noOffsets[scanN, :].real, alpha=0.25, color='steelblue')
    ax[0,0].set_xlabel('ppm')
    ax[0,0].set_ylim(-65000000, 45000000)
    ax[0,0].get_yaxis().set_visible(False)
    ax[0,0].set_xlim(1.5, 5)
    ax[0,0].invert_xaxis()

    ax[1,0]. set_title("Small Additional Offsets")
    for scanS in range(0, smallOffsets.shape[0]):
        ax[1,0].plot(ppm, smallOffsets[scanS, :].real, alpha=0.25, color='cadetblue')
    ax[1,0].set_xlabel('ppm')
    ax[1,0].set_ylim(-65000000, 45000000)
    ax[1,0].get_yaxis().set_visible(False)
    ax[1,0].set_xlim(1.5, 5)
    ax[1,0].invert_xaxis()

    ax[2,0]. set_title("Medium Additional Offsets")
    for scanM in range(0, mediumOffsets.shape[0]):
        ax[2,0].plot(ppm, mediumOffsets[scanM, :].real, alpha=0.25, color='darkcyan')
    ax[2,0].set_xlabel('ppm')
    ax[2,0].set_ylim(-65000000, 45000000)
    ax[2,0].get_yaxis().set_visible(False)
    ax[2,0].set_xlim(1.5, 5)
    ax[2,0].invert_xaxis()

    ax[3,0]. set_title("Large Additional Offsets")
    for scanL in range(0, largeOffsets.shape[0]):
        ax[3,0].plot(ppm, largeOffsets[scanL, :].real, alpha=0.25, color='darkslategrey')
    ax[3,0].set_xlabel('ppm')
    ax[3,0].set_ylim(-65000000, 45000000)
    ax[3,0].get_yaxis().set_visible(False)
    ax[3,0].set_xlim(1.5, 5)
    ax[3,0].invert_xaxis()

    ax[0, 1].set_title("No Additional Offsets - GABA Peak")
    for scanN in range(0, noOffsets.shape[0]):
        ax[0, 1].plot(ppm, noOffsets[scanN, :].real, alpha=0.25, color='steelblue')
    ax[0, 1].set_xlabel('ppm')
    ax[0, 1].set_ylim(-6000000, 12500000)
    ax[0, 1].get_yaxis().set_visible(False)
    ax[0, 1].set_xlim(2.5, 4)
    ax[0, 1].invert_xaxis()

    ax[1, 1].set_title("Small Additional Offsets - GABA Peak")
    for scanS in range(0, smallOffsets.shape[0]):
        ax[1, 1].plot(ppm, smallOffsets[scanS, :].real, alpha=0.25, color='cadetblue')
    ax[1, 1].set_xlabel('ppm')
    ax[1, 1].set_ylim(-6000000, 12500000)
    ax[1, 1].get_yaxis().set_visible(False)
    ax[1, 1].set_xlim(2.5, 4)
    ax[1, 1].invert_xaxis()

    ax[2, 1].set_title("Medium Additional Offsets - GABA Peak")
    for scanM in range(0, mediumOffsets.shape[0]):
        ax[2, 1].plot(ppm, mediumOffsets[scanM, :].real, alpha=0.25, color='darkcyan')
    ax[2, 1].set_xlabel('ppm')
    ax[2, 1].set_ylim(-6000000, 12500000)
    ax[2, 1].get_yaxis().set_visible(False)
    ax[2, 1].set_xlim(2.5, 4)
    ax[2, 1].invert_xaxis()

    ax[3, 1].set_title("Large Additional Offsets - GABA Peak")
    for scanL in range(0, largeOffsets.shape[0]):
        ax[3, 1].plot(ppm, largeOffsets[scanL, :].real, alpha=0.25, color='darkslategrey')
    ax[3, 1].set_xlabel('ppm')
    ax[3, 1].set_xlim(2.5, 4)
    ax[3, 1].set_ylim(-6000000, 12500000)
    ax[3, 1].get_yaxis().set_visible(False)
    ax[3, 1].invert_xaxis()
    plt.show()


def plotAllModels(setName, ppm, offsetsm1N, offsetsm2N, offsetsm3N,
              offsetsm1S, offsetsm2S, offsetsm3S,
              offsetsm1M, offsetsm2M, offsetsm3M,
              offsetsm1L, offsetsm2L, offsetsm3L,
              offsetsUncorrN, offsetsUncorrS, offsetsUncorrM, offsetsUncorrL):

    randScan = random.randint(0, 29)

    for i in range(0, 1):
        randS, randM, randL = i, i, i
            # Figure 3B: Selected Reconstructions by Different Models
        fig1, (ax0, ax1, ax2, ax3) = plt.subplots(4)
        fig1.suptitle("Sample In Vivo Scan Reconstruction after Frequency and Phase Correction")

        ax0.set_title("In Vivo Data with No Artificial Offsets")
        ax0.set_xlabel('ppm')
        ax0.set_xlim(1.5, 5)
        ax0.plot(ppm, offsetsUncorrN[randS, :].real-7500000, 'black', alpha=0.75, label='Uncorr', linewidth=3)
        ax0.plot(ppm, offsetsm3N[randS, :].real+7500000, 'green', alpha=0.75, label=setName[2], linewidth=3)
        ax0.plot(ppm, offsetsm2N[randS, :].real+15000000, 'blue', alpha=0.75, label=setName[1], linewidth=3)
        ax0.plot(ppm, offsetsm1N[randS, :].real, 'orange', alpha=0.75, label=setName[0], linewidth=3)
        ax0.get_yaxis().set_visible(False)
        ax0.invert_xaxis()

        ax1.set_title("In Vivo Data with Small Artificial Offsets")
        ax1.set_xlabel('ppm')
        ax1.set_xlim(1.5, 5)
        ax1.plot(ppm, offsetsUncorrS[randS, :].real-7500000, 'black', alpha=0.75, label='Uncorr', linewidth=3)
        ax1.plot(ppm, offsetsm3S[randS, :].real+7500000, 'green', alpha=0.75, label=setName[2], linewidth=3)
        ax1.plot(ppm, offsetsm2S[randS, :].real+15000000, 'blue', alpha=0.75, label=setName[1], linewidth=3)
        ax1.plot(ppm, offsetsm1S[randS, :].real, 'orange', alpha=0.75, label=setName[0], linewidth=3)
        ax1.get_yaxis().set_visible(False)
        ax1.invert_xaxis()

        ax2.set_title("In Vivo Data with Medium Artificial Offsets")
        ax2.set_xlabel('ppm')
        ax2.set_xlim(1.5, 5)
        ax2.plot(ppm, offsetsUncorrM[randS, :].real-7500000, 'black', alpha=0.75, label='Uncorr', linewidth=3)
        ax2.plot(ppm, offsetsm3M[randM, :].real+7500000, 'green', alpha=0.75, label=setName[2], linewidth=3)
        ax2.plot(ppm, offsetsm2M[randM, :].real+15000000, 'blue', alpha=0.75, label=setName[1], linewidth=3)
        ax2.plot(ppm, offsetsm1M[randM, :].real, 'orange', alpha=0.75, label=setName[0], linewidth=3)
        ax2.get_yaxis().set_visible(False)
        ax2.invert_xaxis()

        ax3.set_title("In Vivo Data with Large Artificial Offsets")
        ax3.set_xlabel('ppm')
        ax3.set_xlim(1.5, 5)
        ax3.plot(ppm, offsetsUncorrL[randS, :].real-7500000, 'black', alpha=0.75, label='Uncorr', linewidth=3)
        ax3.plot(ppm, offsetsm3L[randL, :].real+7500000, 'green', alpha=0.75, label=setName[2], linewidth=3)
        ax3.plot(ppm, offsetsm2L[randL, :].real+15000000, 'blue', alpha=0.75, label=setName[1], linewidth=3)
        ax3.plot(ppm, offsetsm1L[randL, :].real, 'orange', alpha=0.75, label=setName[0], linewidth=3)
        ax3.get_yaxis().set_visible(False)
        ax3.invert_xaxis()

        plt.legend()
        plt.show()


def plotQMetric(setName, ppm, noOffsets_M1, noOffsets_M2, noOffsets_M3,
                smallOffsets_M1, smallOffsets_M2, smallOffsets_M3,
                medOffsets_M1, medOffsets_M2, medOffsets_M3,
                largeOffsets_M1, largeOffsets_M2, largeOffsets_M3):

    # finish, start = np.where(ppm <= 3.15)[0][0], np.where(ppm >= 3.28)[0][-1] # inverted ppm
    start, finish = np.where(ppm <= 3.15)[0][-1], np.where(ppm >= 3.28)[0][0]

    noOffsets_M1, noOffsets_M2, noOffsets_M3 = (noOffsets_M1[:, start:finish]), (noOffsets_M2[:, start:finish]), (noOffsets_M3[:, start:finish])
    varNo_M1, varNo_M2, varNo_M3 = np.var(noOffsets_M1, axis=1), np.var(noOffsets_M2, axis=1), np.var(noOffsets_M3, axis=1)

    smallOffsets_M1, smallOffsets_M2, smallOffsets_M3 = (smallOffsets_M1[:, start:finish]), (smallOffsets_M2[:, start:finish]), (smallOffsets_M3[:, start:finish])
    varSmall_M1, varSmall_M2, varSmall_M3 = np.var(smallOffsets_M1, axis=1), np.var(smallOffsets_M2, axis=1), np.var(smallOffsets_M3, axis=1)

    medOffsets_M1, medOffsets_M2, medOffsets_M3 = (medOffsets_M1[:, start:finish]), (medOffsets_M2[:, start:finish]), (medOffsets_M3[:, start:finish])
    varMed_M1, varMed_M2, varMed_M3 = np.var(medOffsets_M1, axis=1), np.var(medOffsets_M2, axis=1), np.var(medOffsets_M3, axis=1)

    largeOffsets_M1, largeOffsets_M2, largeOffsets_M3 = (largeOffsets_M1[:, start:finish]), (largeOffsets_M2[:, start:finish]), (largeOffsets_M3[:, start:finish])
    varLarge_M1, varLarge_M2, varLarge_M3 = np.var(largeOffsets_M1, axis=1), np.var(largeOffsets_M2, axis=1), np.var(largeOffsets_M3, axis=1)

    qNo12, qNo13 = np.zeros(varSmall_M1.shape[0]), np.zeros(varSmall_M1.shape[0])
    qSmall12, qSmall13 = np.zeros(varSmall_M1.shape[0]), np.zeros(varSmall_M1.shape[0])
    qMed12, qMed13 = np.zeros(varSmall_M1.shape[0]), np.zeros(varSmall_M1.shape[0])
    qLarge12, qLarge13 = np.zeros(varSmall_M1.shape[0]), np.zeros(varSmall_M1.shape[0])

    fig3, axs = plt.subplots(2, 4)
    fig3.suptitle("In Vivo Choline Artifact CV-CNN vs. CNN")

    axs[0,0].set_title("No Additional Offsets")
    axs[0,0].set_xlabel('Scan')
    axs[0,0].set_ylabel('Q Value')
    axs[0,0].plot([-1, 37], [0.5, 0.5], "k--")
    axs[0,0].set_ylim(0.0, 1.0)
    axs[0,0].set_xlim(-1, 37)

    axs[0,1].set_title("Small Additional Offsets")
    axs[0,1].set_xlabel('Scan')
    axs[0,1].set_ylabel('Q Value')
    axs[0,1].plot([-1, 37], [0.5, 0.5], "k--")
    axs[0,1].set_ylim(0.0, 1.0)
    axs[0,1].set_xlim(-1, 37)

    axs[0,2].set_title("Medium Additional Offsets")
    axs[0,2].set_xlabel('Scan')
    axs[0,2].set_ylabel('Q Value')
    axs[0,2].plot([-1, 37], [0.5, 0.5], "k--")
    axs[0,2].set_ylim(0.0, 1.0)
    axs[0,2].set_xlim(-1, 37)

    axs[0,3].set_title("Large Additional Offsets")
    axs[0,3].set_xlabel('Scan')
    axs[0,3].set_ylabel('Q Value')
    axs[0,3].plot([-1, 37], [0.5, 0.5], "k--")
    axs[0,3].set_ylim(0.0, 1.0)
    axs[0,3].set_xlim(-1, 37)

    axs[1,0].set_title("No Additional Offsets")
    axs[1,0].set_xlabel('Scan')
    axs[1,0].set_ylabel('Q Value')
    axs[1,0].plot([-1, 37], [0.5, 0.5], "k--")
    axs[1,0].set_ylim(0.0, 1.0)
    axs[1,0].set_xlim(-1, 37)

    axs[1,1].set_title("Small Additional Offsets")
    axs[1,1].set_xlabel('Scan')
    axs[1,1].set_ylabel('Q Value')
    axs[1,1].plot([-1, 37], [0.5, 0.5], "k--")
    axs[1,1].set_ylim(0.0, 1.0)
    axs[1,1].set_xlim(-1, 37)

    axs[1,2].set_title("Medium Additional Offsets")
    axs[1,2].set_xlabel('Scan')
    axs[1,2].set_ylabel('Q Value')
    axs[1,2].plot([-1, 37], [0.5, 0.5], "k--")
    axs[1,2].set_ylim(0.0, 1.0)
    axs[1,2].set_xlim(-1, 37)

    axs[1,3].set_title("Large Additional Offsets")
    axs[1,3].set_xlabel('Scan')
    axs[1,3].set_ylabel('Q Value')
    axs[1,3].plot([-1, 37], [0.5, 0.5], "k--")
    axs[1,3].set_ylim(0.0, 1.0)
    axs[1,3].set_xlim(-1, 37)

    legend_elements12 = [Line2D([0], [0], marker='o', color='w', label=f'{setName[0]} > {setName[1]}', markerfacecolor='blue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label=f'{setName[0]} < {setName[1]}', markerfacecolor='red', markersize=15)]
    legend_elements13 = [Line2D([0], [0], marker='o', color='w', label=f'{setName[0]} > {setName[2]}', markerfacecolor='blue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label=f'{setName[0]} < {setName[2]}', markerfacecolor='red', markersize=15)]

    axs[0,0].legend(handles=legend_elements12, loc='lower right')
    axs[0,1].legend(handles=legend_elements12, loc='lower right')
    axs[0,2].legend(handles=legend_elements12, loc='lower right')
    axs[0,3].legend(handles=legend_elements12, loc='lower right')
    axs[1,0].legend(handles=legend_elements13, loc='lower right')
    axs[1,1].legend(handles=legend_elements13, loc='lower right')
    axs[1,2].legend(handles=legend_elements13, loc='lower right')
    axs[1,3].legend(handles=legend_elements13, loc='lower right')

    for i in range(0, varSmall_M1.shape[0]):
        qNo12[i] = 1 - (varNo_M1[i]) / (varNo_M1[i] + varNo_M2[i])
        qSmall12[i] = 1 - (varSmall_M1[i]) / (varSmall_M1[i] + varSmall_M2[i])
        qMed12[i] = 1 - (varMed_M1[i]) / (varMed_M1[i] + varMed_M2[i])
        qLarge12[i] = 1 - (varLarge_M1[i]) / (varLarge_M1[i] + varLarge_M2[i])

        qNo13[i] = 1 - (varNo_M1[i]) / (varNo_M1[i] + varNo_M3[i])
        qSmall13[i] = 1 - (varSmall_M1[i]) / (varSmall_M1[i] + varSmall_M3[i])
        qMed13[i] = 1 - (varMed_M1[i]) / (varMed_M1[i] + varMed_M3[i])
        qLarge13[i] = 1 - (varLarge_M1[i]) / (varLarge_M1[i] + varLarge_M3[i])

        if (qNo12[i] >= 0.5):
            axs[0, 0].plot(i, qNo12[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > CNN')
        else:
            axs[0, 0].plot(i, qNo12[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < CNN')
        if (qSmall12[i] >= 0.5):
            axs[0,1].plot(i, qSmall12[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > CNN')
        else:
            axs[0,1].plot(i, qSmall12[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < CNN')

        if (qMed12[i] >= 0.5):
            axs[0,2].plot(i, qMed12[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > CNN')
        else:
            axs[0,2].plot(i, qMed12[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < CNN')

        if (qLarge12[i] >= 0.5):
            axs[0,3].plot(i, qLarge12[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > CNN')
        else:
            axs[0,3].plot(i, qLarge12[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < CNN')

        if (qNo13[i] >= 0.5):
            axs[1, 0].plot(i, qNo13[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > CNN')
        else:
            axs[1, 0].plot(i, qNo13[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < CNN')
        if (qSmall13[i] >= 0.5):
            axs[1,1].plot(i, qSmall13[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > MLP')
        else:
            axs[1,1].plot(i, qSmall13[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < MLP')

        if (qMed13[i] >= 0.5):
            axs[1,2].plot(i, qMed13[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > MLP')
        else:
            axs[1,2].plot(i, qMed13[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < MLP')

        if (qLarge13[i] >= 0.5):
            axs[1,3].plot(i, qLarge13[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='CR-CNN > MLP')
        else:
            axs[1,3].plot(i, qLarge13[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", label='CR-CNN < MLP')

    avQN12, stdQN12 = np.mean(qNo12), np.std(qNo12)
    avQS12, stdQS12 = np.mean(qSmall12), np.std(qSmall12)
    avQM12, stdQM12 = np.mean(qMed12), np.std(qMed12)
    avQL12, stdQL12 = np.mean(qLarge12), np.std(qLarge12)
    axs[0,0].text(0, 0.1, f"E[Q] = {round(avQN12, 2)} +/- {round(stdQN12, 2)}")
    axs[0,1].text(0, 0.1, f"E[Q] = {round(avQS12, 2)} +/- {round(stdQS12, 2)}")
    axs[0,2].text(0, 0.1, f"E[Q] = {round(avQM12, 2)} +/- {round(stdQM12, 2)}")
    axs[0,3].text(0, 0.1, f"E[Q] = {round(avQL12, 2)} +/- {round(stdQL12, 2)}")

    avQN13, stdQN13 = np.mean(qNo13), np.std(qNo13)
    avQS13, stdQS13 = np.mean(qSmall13), np.std(qSmall13)
    avQM13, stdQM13 = np.mean(qMed13), np.std(qMed13)
    avQL13, stdQL13 = np.mean(qLarge13), np.std(qLarge13)
    axs[1,0].text(0, 0.1, f"E[Q] = {round(avQN13, 2)} +/- {round(stdQN13, 2)}")
    axs[1,1].text(0, 0.1, f"E[Q] = {round(avQS13, 2)} +/- {round(stdQS13, 2)}")
    axs[1,2].text(0, 0.1, f"E[Q] = {round(avQM13, 2)} +/- {round(stdQM13, 2)}")
    axs[1,3].text(0, 0.1, f"E[Q] = {round(avQL13, 2)} +/- {round(stdQL13, 2)}")

    plt.show()


def plotQualityMetrics(setName, vivoPPM,
             m1SpecsNone_Mscans, m2SspecsNone_Mscans, m3SpecsNonel_Mscans,
             m1SpecsSmall_Mscans, m2SpecsSmall_Mscans, m3SpecsSmall_Mscans,
             m1SpecsMed_Mscans, m2SpecsMed_Mscans, m3SpecsMed_Mscans,
             m1SpecsLarge_Mscans, m2SpecsLarge_Mscans, m3SpecsLarge_Mscans):

    meanSnrN1, stdSnrN1 = calculate_snr(m1SpecsNone_Mscans, vivoPPM)[1:3]
    meanSnrS1, stdSnrS1 = calculate_snr(m1SpecsSmall_Mscans, vivoPPM)[1:3]
    meanSnrM1, stdSnrM1 = calculate_snr(m1SpecsMed_Mscans, vivoPPM)[1:3]
    meanSnrL1, stdSnrL1 = calculate_snr(m1SpecsLarge_Mscans, vivoPPM)[1:3]
    meanLwN1, stdLwN1 = calculate_linewidth(m1SpecsNone_Mscans, vivoPPM)[1:3]
    meanLwS1, stdLwS1 = calculate_linewidth(m1SpecsSmall_Mscans, vivoPPM)[1:3]
    meanLwM1, stdLwM1 = calculate_linewidth(m1SpecsMed_Mscans, vivoPPM)[1:3]
    meanLwL1, stdLwL1 = calculate_linewidth(m1SpecsLarge_Mscans, vivoPPM)[1:3]

    meanSnrN2, stdSnrN2 = calculate_snr(m2SspecsNone_Mscans, vivoPPM)[1:3]
    meanSnrS2, stdSnrS2 = calculate_snr(m2SpecsSmall_Mscans, vivoPPM)[1:3]
    meanSnrM2, stdSnrM2 = calculate_snr(m2SpecsMed_Mscans, vivoPPM)[1:3]
    meanSnrL2, stdSnrL2 = calculate_snr(m2SpecsLarge_Mscans, vivoPPM)[1:3]
    meanLwN2, stdLwN2 = calculate_linewidth(m2SspecsNone_Mscans, vivoPPM)[1:3]
    meanLwS2, stdLwS2 = calculate_linewidth(m2SpecsSmall_Mscans, vivoPPM)[1:3]
    meanLwM2, stdLwM2 = calculate_linewidth(m2SpecsMed_Mscans, vivoPPM)[1:3]
    meanLwL2, stdLwL2 = calculate_linewidth(m2SpecsLarge_Mscans, vivoPPM)[1:3]

    meanSnrN3, stdSnrN3 = calculate_snr(m3SpecsNonel_Mscans, vivoPPM)[1:3]
    meanSnrS3, stdSnrS3 = calculate_snr(m3SpecsSmall_Mscans, vivoPPM)[1:3]
    meanSnrM3, stdSnrM3 = calculate_snr(m3SpecsMed_Mscans, vivoPPM)[1:3]
    meanSnrL3, stdSnrL3 = calculate_snr(m3SpecsLarge_Mscans, vivoPPM)[1:3]
    meanLwN3, stdLwN3 = calculate_linewidth(m3SpecsNonel_Mscans, vivoPPM)[1:3]
    meanLwS3, stdLwS3 = calculate_linewidth(m3SpecsSmall_Mscans, vivoPPM)[1:3]
    meanLwM3, stdLwM3 = calculate_linewidth(m3SpecsMed_Mscans, vivoPPM)[1:3]
    meanLwL3, stdLwL3 = calculate_linewidth(m3SpecsLarge_Mscans, vivoPPM)[1:3]

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

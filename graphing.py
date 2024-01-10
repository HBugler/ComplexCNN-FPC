import numpy as np
import matplotlib.pyplot as plt


# Figure 3A and 3B: Spectra Reconstruction
def plotFig3A(ppm, smallOffsets_CV_CNN, mediumOffsets_CV_CNN, largeOffsets_CV_CNN):
    # Figure 3A: Reconstructions Separated by Offsets
    fig1, (ax1, ax2, ax3) = plt.subplots(3)
    fig1.suptitle("CV-CNN Frequency and Phase Corrected In Vivo Spectra")

    ax1. set_title("Small Additional Offsets")
    ax1.set_xlabel('ppm')
    ax1.invert_xaxis()
    ax1.set_xlim(0, 5)
    ax1.set_ylim(np.min(smallOffsets_CV_CNN) - 0.1, np.max(smallOffsets_CV_CNN) + 0.1)
    for trans in range(0, smallOffsets_CV_CNN.shape[0]):
        ax1.plot(ppm, smallOffsets_CV_CNN, alpha=0.25, color='cadetblue')

    ax2. set_title("Medium Additional Offsets")
    ax2.set_xlabel('ppm')
    ax2.invert_xaxis()
    ax2.set_xlim(0, 5)
    ax2.set_ylim(np.min(mediumOffsets_CV_CNN) - 0.1, np.max(mediumOffsets_CV_CNN) + 0.1)
    for trans in range(0, mediumOffsets_CV_CNN.shape[0]):
        ax2.plot(ppm, mediumOffsets_CV_CNN, alpha=0.25, color='darkcyan')

    ax3. set_title("Large Additional Offsets")
    ax3.set_xlabel('ppm')
    ax3.invert_xaxis()
    ax3.set_xlim(0, 5)
    ax3.set_ylim(np.min(largeOffsets_CV_CNN) - 0.1, np.max(largeOffsets_CV_CNN) + 0.1)
    for trans in range(0, largeOffsets_CV_CNN.shape[0]):
        ax2.plot(ppm, largeOffsets_CV_CNN, alpha=0.25, color='darkslategrey')

    plt.show()

def plotFig3B(ppm, posWater_CV_CNN, negWater_CV_CNN, mixWater_CV_CNN,
              posWater_MLP, negWater_MLP, mixWater_MLP,
              posWater_CNN, negWater_CNN, mixWater_CNN,
              ):
    # Figure 3B: Selected Reconstructions by Different Models for Pos, Inv, Mix Water
    fig1, (ax1, ax2, ax3) = plt.subplots(3)
    fig1.suptitle("Mean Spectrum of Frequency and Phase Corrected Simulated Scan")

    ax1.set_title("Positive Water Residual Spectrum Reconstruction")
    ax1.set_xlabel('ppm')
    ax1.invert_xaxis()
    ax1.set_xlim(0, 5)
    ax1.set_ylim(np.min(posWater_CV_CNN) - 0.1, np.max(posWater_CV_CNN) + 0.1)
    ax1.plot(ppm, posWater_MLP.mean(axis=1).flatten(), 'green', alpha=0.75, label='MLP')
    ax1.plot(ppm, posWater_CNN.mean(axis=1).flatten(), 'blue', alpha=0.75, label='CNN')
    ax1.plot(ppm, posWater_CV_CNN.mean(axis=1).flatten(), 'orange', alpha=0.75, label='CV-CNN')

    ax2.set_title("Negative Water Residual Spectrum Reconstruction")
    ax2.set_xlabel('ppm')
    ax2.invert_xaxis()
    ax2.set_xlim(0, 5)
    ax2.set_ylim(np.min(negWater_CV_CNN) - 0.1, np.max(negWater_CV_CNN) + 0.1)
    ax2.plot(ppm, negWater_MLP.mean(axis=1).flatten(), 'green', alpha=0.75, label='MLP')
    ax2.plot(ppm, negWater_CNN.mean(axis=1).flatten(), 'blue', alpha=0.75, label='CNN')
    ax2.plot(ppm, negWater_CV_CNN.mean(axis=1).flatten(), 'orange', alpha=0.75, label='CV-CNN')

    ax3.set_title("Mixed Sign Water Residual Spectrum Reconstruction")
    ax3.set_xlabel('ppm')
    ax3.invert_xaxis()
    ax3.set_xlim(0, 5)
    ax3.set_ylim(np.min(mixWater_CV_CNN) - 0.1, np.max(mixWater_CV_CNN) + 0.1)
    ax3.plot(ppm, mixWater_MLP.mean(axis=1).flatten(), 'green', alpha=0.75, label='MLP')
    ax3.plot(ppm, mixWater_CNN.mean(axis=1).flatten(), 'blue', alpha=0.75, label='CNN')
    ax3.plot(ppm, mixWater_CV_CNN.mean(axis=1).flatten(), 'orange', alpha=0.75, label='CV-CNN')

    plt.legend()
    plt.show()

def qMetricPlot(ppm, smallOffsets_CV_CNN, mediumOffsets_CV_CNN, largeOffsets_CV_CNN, smallOffsets_CNN, mediumOffsets_CNN, largeOffsets_CNN):
    ind_close, ind_far = np.amax(np.where(ppm == 3.15)), np.amin(np.where(ppm == 3.28))
    smallOffsets_CV_CNN = smallOffsets_CV_CNN[0, :, ind_far:ind_close].flatten()
    varSmall_CV = np.var(smallOffsets_CV_CNN, axis=1)
    mediumOffsets_CV_CNN = mediumOffsets_CV_CNN[0, :, ind_far:ind_close].flatten()
    varMed_CV = np.var(mediumOffsets_CV_CNN, axis=1)
    largeOffsets_CV_CNN = largeOffsets_CV_CNN[0, :, ind_far:ind_close].flatten()
    varLarge_CV = np.var(largeOffsets_CV_CNN, axis=1)
    smallOffsets_CNN = smallOffsets_CNN[0, :, ind_far:ind_close].flatten()
    varSmall_CNN = np.var(smallOffsets_CNN, axis=1)
    mediumOffsets_CNN = mediumOffsets_CNN[0, :, ind_far:ind_close].flatten()
    varMed_CNN = np.var(mediumOffsets_CNN, axis=1)
    largeOffsets_CNN = largeOffsets_CNN[0, :, ind_far:ind_close].flatten()
    varLarge_CNN = np.var(largeOffsets_CNN, axis=1)

    qSmall, qMed, qLarge = np.zeros(varSmall_CV.shape[0]), np.zeros(varSmall_CV.shape[0]), np.zeros(varSmall_CV.shape[0])

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

    for i in range(0, varSmall_CV.shape[0]):
        qSmall[i] = (1 - varSmall_CV) / (varSmall_CV + varSmall_CNN)
        qMed[i] = (1 - varMed_CV) / (varMed_CV + varMed_CNN)
        qLarge[i] = (1 - varLarge_CV) / (varLarge_CV + varLarge_CNN)

        if (qSmall >= 0.5):
            ax1.plot(i, qSmall[i], 'blue', label='CV-CNN > CNN')
        else:
            ax1.plot(i, qSmall[i], 'red', label='CV-CNN < CNN')

        if (qMed >= 0.5):
            ax1.plot(i, qMed[i], 'blue', label='CV-CNN > CNN')
        else:
            ax1.plot(i, qMed[i], 'red', label='CV-CNN < CNN')

        if (qLarge >= 0.5):
            ax1.plot(i, qLarge[i], 'blue', label='CV-CNN > CNN')
        else:
            ax1.plot(i, qLarge[i], 'red', label='CV-CNN < CNN')

    plt.legend()
    plt.show()

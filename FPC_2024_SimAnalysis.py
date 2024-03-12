import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error

########################################################################################################################
# Load Data
########################################################################################################################
snrTypes = ["2_5", "5", "10"]
net = ["Freq", "Phase"]
dataTypes = ["Mix", "None", "Pos"]
modelTypes = ["compReal", "compComp", "realReal", "Ma_4Convs", "Tapper"]
subDir = f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/"

for water in dataTypes:
    indW = dataTypes.index(water)
    for snr in snrTypes:
        indS = snrTypes.index(snr)

        TrueLabels_Freq = np.load(f"{subDir}Corrupt/Sim{snrTypes[indS]}_{dataTypes[indW]}/TrueFreqLabels_Sim{snrTypes[indS]}_{dataTypes[indW]}_Test.npy")[:, 0, :]
        TrueLabels_Phase = np.load(f"{subDir}Corrupt/Sim{snrTypes[indS]}_{dataTypes[indW]}/TruePhaseLabels_Sim{snrTypes[indS]}_{dataTypes[indW]}_Test.npy")[:, 0, :]

        TrueLabels_Freq = np.concatenate((TrueLabels_Freq[1, :], TrueLabels_Freq[0, :]))
        TrueLabels_Phase = np.concatenate((TrueLabels_Phase[1, :], TrueLabels_Phase[0, :]))

        compReal_freqLabels = np.load(f"{subDir}Predictions/PredLabels_{net[0]}_Sim{snrTypes[indS]}_{dataTypes[indW]}Water_{modelTypes[0]}.npy")
        compReal_phaseLabels = np.load(f"{subDir}Predictions/PredLabels_{net[1]}_Sim{snrTypes[indS]}_{dataTypes[indW]}Water_{modelTypes[0]}.npy")
        compComp_freqLabels = np.load(f"{subDir}Predictions/PredLabels_{net[0]}_Sim{snrTypes[indS]}_{dataTypes[indW]}Water_{modelTypes[1]}.npy")
        compComp_phaseLabels = np.load(f"{subDir}Predictions/PredLabels_{net[1]}_Sim{snrTypes[indS]}_{dataTypes[indW]}Water_{modelTypes[1]}.npy")
        realReal_freqLabels = np.load(f"{subDir}Predictions/PredLabels_{net[0]}_Sim{snrTypes[indS]}_{dataTypes[indW]}Water_{modelTypes[2]}.npy")
        realReal_phaseLabels = np.load(f"{subDir}Predictions/PredLabels_{net[1]}_Sim{snrTypes[indS]}_{dataTypes[indW]}Water_{modelTypes[2]}.npy")
        Ma_freqLabels = np.load(f"{subDir}Predictions/PredLabels_{net[0]}_Sim{snrTypes[indS]}_{dataTypes[indW]}Water_{modelTypes[3]}.npy")
        Ma_phaseLabels = np.load(f"{subDir}Predictions/PredLabels_{net[1]}_Sim{snrTypes[indS]}_{dataTypes[indW]}Water_{modelTypes[3]}.npy")
        Tapper_freqLabels = np.load(f"{subDir}Predictions/PredLabels_{net[0]}_Sim{snrTypes[indS]}_{dataTypes[indW]}Water_{modelTypes[4]}.npy")
        Tapper_phaseLabels = np.load(f"{subDir}Predictions/PredLabels_{net[1]}_Sim{snrTypes[indS]}_{dataTypes[indW]}Water_{modelTypes[4]}.npy")

        print(f'SNR {snrTypes[indS]} {dataTypes[indW]} Water')
        print(f'MAEs')
        print(f'Freq {modelTypes[0]}: {mean_absolute_error(TrueLabels_Freq, compReal_freqLabels)} +/- {np.std(abs(TrueLabels_Freq - compReal_freqLabels))}')
        print(f'Phase {modelTypes[0]}: {mean_absolute_error(TrueLabels_Phase, compReal_phaseLabels)} +/- {np.std(abs(TrueLabels_Phase - compReal_phaseLabels))}')
        print(f'Freq {modelTypes[1]}: {mean_absolute_error(TrueLabels_Freq, compComp_freqLabels)} +/- {np.std(abs(TrueLabels_Freq - compComp_freqLabels))}')
        print(f'Phase {modelTypes[1]}: {mean_absolute_error(TrueLabels_Phase, compComp_phaseLabels)} +/- {np.std(abs(TrueLabels_Phase - compComp_phaseLabels))}')
        print(f'Freq {modelTypes[2]}: {mean_absolute_error(TrueLabels_Freq, realReal_freqLabels)} +/- {np.std(abs(TrueLabels_Freq - realReal_freqLabels))}')
        print(f'Phase {modelTypes[2]}: {mean_absolute_error(TrueLabels_Phase, realReal_phaseLabels)} +/- {np.std(abs(TrueLabels_Phase - realReal_phaseLabels))}')
        print(f'Freq {modelTypes[3]}: {mean_absolute_error(TrueLabels_Freq, Ma_freqLabels)} +/- {np.std(abs(TrueLabels_Freq - Ma_freqLabels))}')
        print(f'Phase {modelTypes[3]}: {mean_absolute_error(TrueLabels_Phase, Ma_phaseLabels)} +/- {np.std(abs(TrueLabels_Phase - Ma_phaseLabels))}')
        print(f'Freq {modelTypes[4]}: {mean_absolute_error(TrueLabels_Freq, Tapper_freqLabels)} +/- {np.std(abs(TrueLabels_Freq - Tapper_freqLabels))}')
        print(f'Phase {modelTypes[4]}: {mean_absolute_error(TrueLabels_Phase, Tapper_phaseLabels)} +/- {np.std(abs(TrueLabels_Phase - Tapper_phaseLabels))}')
        print()
        print(f'Significance Ablation')
        print(f'Freq {modelTypes[0]} and {modelTypes[1]}: {stats.wilcoxon(compReal_freqLabels, compComp_freqLabels)[1]}')
        print(f'Freq {modelTypes[0]} and {modelTypes[2]}: {stats.wilcoxon(compReal_freqLabels, realReal_freqLabels)[1]}')
        print(f'Freq {modelTypes[1]} and {modelTypes[2]}: {stats.wilcoxon(compComp_freqLabels, realReal_freqLabels)[1]}')
        print(f'Phase {modelTypes[0]} and {modelTypes[1]}: {stats.wilcoxon(compReal_phaseLabels, compComp_phaseLabels)[1]}')
        print(f'Phase {modelTypes[0]} and {modelTypes[2]}: {stats.wilcoxon(compReal_phaseLabels, realReal_phaseLabels)[1]}')
        print(f'Phase {modelTypes[1]} and {modelTypes[2]}: {stats.wilcoxon(compComp_phaseLabels, realReal_phaseLabels)[1]}')
        print(f'Significance Comparative')
        print(f'Freq {modelTypes[0]} and {modelTypes[3]}: {stats.wilcoxon(compReal_freqLabels, Ma_freqLabels)[1]}')
        print(f'Freq {modelTypes[0]} and {modelTypes[4]}: {stats.wilcoxon(compReal_freqLabels, Tapper_freqLabels)[1]}')
        print(f'Freq {modelTypes[3]} and {modelTypes[4]}: {stats.wilcoxon(Ma_freqLabels, Tapper_freqLabels)[1]}')
        print(f'Phase {modelTypes[0]} and {modelTypes[3]}: {stats.wilcoxon(compReal_phaseLabels, Ma_phaseLabels)[1]}')
        print(f'Phase {modelTypes[0]} and {modelTypes[4]}: {stats.wilcoxon(compReal_phaseLabels, Tapper_phaseLabels)[1]}')
        print(f'Phase {modelTypes[3]} and {modelTypes[4]}: {stats.wilcoxon(Ma_phaseLabels, Tapper_phaseLabels)[1]}')
        print()
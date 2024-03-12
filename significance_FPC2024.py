import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error as mae
from plottingFPC import plotFig3A, plotFig3B, qMetricPlot

allnets, allmodels, allwaters, allsnrs = [], [], [], []
allMAES1, allMAES2, allMAES3 = [], [], []
allpvalueNorm1, allpvalueNorm2, allpvalueNorm3 = [], [], []
allpvalueW1, allpvalueW2, allpvalueW3 = [], [], []

# load data
labelsLocation = "C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/"
netType = ["freq", "phase"]
snrType = ["10", "5", "2_5"]
waterType = ["None", "Pos", "Mix"]                  #["None", "Pos", "Neg", "Mix"]   #
modelType = ["realReal", "compReal", "compComp"]    #["compReal", "Ma_4Convs", "Tapper"]


for water in waterType:
    indW = waterType.index(water)
    for snr in snrType:
        indS = snrType.index(snr)
        for net in netType:
            indN = netType.index(net)
            print(f'for {waterType[indW]}, {snrType[indS]}, {netType[indN]}')

            model1 = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/PredLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[0]}.npy")
            model2 = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/PredLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[1]}.npy")
            model3 = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/PredLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[2]}.npy")
            trueVals1 = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/TrueLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[0]}.npy")
            trueVals2 = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/TrueLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[1]}.npy")
            trueVals3 = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Models and Predictions New Norm/TrueLabels_{netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[2]}.npy")

            allnets.append(netType[indN]), allsnrs.append(snrType[indS]), allwaters.append(waterType[indW])

            # sort data
            OG_Sort1, OG_Sort2, OG_Sort3 = np.copy(trueVals1), np.copy(trueVals2), np.copy(trueVals3)
            trueVals1.sort(), trueVals2.sort(), trueVals3.sort()

            if (trueVals3==trueVals1).all() and (trueVals2==trueVals1).all() and (trueVals2==trueVals3).all():
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
                # print(f'Mean Absolute error for {netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[0]} is {MAE1} +/- {error1.std()}')
                # print(f'Mean Absolute error for {netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[1]} is {MAE2} +/- {error2.std()}')
                # print(f'Mean Absolute error for {netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water_{modelType[2]} is {MAE3} +/- {error3.std()}')

                cohens_d_1to2 = (np.mean(error1) - np.mean(error2)) / (np.sqrt((np.std(error1) ** 2 + np.std(error2) ** 2) / 2))
                cohens_d_1to3 = (np.mean(error1) - np.mean(error3)) / (np.sqrt((np.std(error1) ** 2 + np.std(error3) ** 2) / 2))
                cohens_d_2to3 = (np.mean(error2) - np.mean(error3)) / (np.sqrt((np.std(error2) ** 2 + np.std(error3) ** 2) / 2))

                print(f"Cohen's D ({modelType[0]} compared to {modelType[1]}) for {netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water is {cohens_d_1to2}")
                print(f"Cohen's D ({modelType[0]} compared to {modelType[2]}) for {netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water is {cohens_d_1to3}")
                print(f"Cohen's D ({modelType[1]} compared to {modelType[2]}) for {netType[indN]}_Sim{snrType[indS]}_{waterType[indW]}Water is {cohens_d_2to3}")

                # check for normality
                statNorm1, pvalueNorm1 = stats.normaltest(error1, axis=0)
                statNorm2, pvalueNorm2 = stats.normaltest(error2, axis=0)
                statNorm3, pvalueNorm3 = stats.normaltest(error3, axis=0)
                allpvalueNorm1.append(pvalueNorm1), allpvalueNorm2.append(pvalueNorm2), allpvalueNorm3.append(pvalueNorm3)

                # perform Wilcoxon signed rank test
                statW1, pvalueW1 = stats.wilcoxon(error1, error2)
                statW2, pvalueW2 = stats.wilcoxon(error1, error3)
                statW3, pvalueW3 = stats.wilcoxon(error2, error3)
                allpvalueW1.append(pvalueW1), allpvalueW2.append(pvalueW2), allpvalueW3.append(pvalueW3)
            else:
                print(f'GOT FALSE!')

# Create dataframe with results
# significanceFrame = pd.DataFrame({
#     'Net Type': allnets,
#     'SNR Type': allsnrs,
#     'Water Type': allwaters,
#     # 'Model Type': allmodels,
#
#     f'MAE model1 {modelType[0]}': allMAES1,
#     f'MAE model2 {modelType[1]}': allMAES2,
#     f'MAE model3 {modelType[2]}': allMAES3,
#
#     f'Normality Test pvalue {modelType[0]}': allpvalueNorm1,
#     f'Normality Test pvalue {modelType[1]}': allpvalueNorm2,
#     f'Normality Test pvalue {modelType[2]}': allpvalueNorm3,
#
#     f'Wilcoxon Test pvalue ({modelType[0]} vs. {modelType[1]})': allpvalueW1,
#     f'Wilcoxon Test pvalue ({modelType[0]} vs. {modelType[2]})': allpvalueW2,
#     f'Wilcoxon Test pvalue ({modelType[1]} vs. {modelType[2]})': allpvalueW3})
#
# significanceFrame.to_csv(f"SimCompStudySignificance_FPC2024.csv", index=False)

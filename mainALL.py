# FPC Winter 2024
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from FPC_Functions import loadSimData, loadVivoData, normSpecs, divideDev, shuffleData, get2Chann, get1Chann
from FPC_Functions import getMag, window1024, window1024V, toFids, toSpecs, corrFShift, corrPShift
from Datasets import FPC_Dataset_Tapper, FPC_Dataset_1C, FPC_Dataset_2C
from AblationStudyModels import compIn_realConv, compIn_compConv, realIn_realConv
from TapperModel import tapperModel
from MaModel import maModel

# ########################################################################################################################
# # VARIABLE SET-UP
# ########################################################################################################################
# modelTypes = ["compReal"]
# waterTypes, snrTypes, netTypes = ["Mix"], ["2_5"], ["freq", "phase"]
# ppm = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/GTs/ppm_Sim.npy")
# t = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/GTs/time_Sim.npy")
#
# # hyperparameters
# batch_size, learn_r, nmb_epochs = 64, 0.001, 50
# loss_fn = nn.L1Loss()
# lr_scheduler_freq = 25
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# predTrainLabels, predValLabels = 0, 0
#
# #########################################################################################################################
# # DATA PROCESSING
# #########################################################################################################################
# for water in waterTypes:
#     indW = waterTypes.index(water)
#     for snr in snrTypes:
#         indS = snrTypes.index(snr)
#         # load data
#         on_dev, off_dev, on_devFC, off_devFC, freq_dev, phase_dev = loadSimData(snrTypes[indS], waterTypes[indW], termi="Final0322", dataTypes="Dev")
#         on_test, off_test, on_testFC, off_testFC, freq_test, phase_test = loadSimData(snrTypes[indS], waterTypes[indW], termi="Final0322", dataTypes="test")
#
#         # Normalization
#         on_all_specs_dev, off_all_specs_dev = normSpecs(on_dev, off_dev)
#         on_all_specs_devFC, off_all_specs_devFC = normSpecs(on_devFC, off_devFC)
#         on_all_specs_test, off_all_specs_test = normSpecs(on_test, off_test)
#         on_all_specs_testFC, off_all_specs_testFC = normSpecs(on_testFC, off_testFC)
#
#         # Separate development set into train/val
#         trainSpecs, valSpecs = divideDev(on_all_specs_dev, off_all_specs_dev, percent=0.8)
#         trainSpecsFC, valSpecsFC = divideDev(on_all_specs_devFC, off_all_specs_devFC, percent=0.8)
#         trainFreqLabels, valFreqLabels = divideDev(freq_dev[1, :], freq_dev[0, :], percent=0.8)
#         trainPhaseLabels, valPhaseLabels = divideDev(phase_dev[1, :], phase_dev[0, :], percent=0.8)
#
#         testSpecs = np.concatenate((on_all_specs_test, off_all_specs_test), axis=0)
#         testSpecsFC = np.concatenate((on_all_specs_testFC, off_all_specs_testFC), axis=0)
#         testFreqLabels = np.concatenate((freq_test[1, :], freq_test[0, :]), axis=0)[np.newaxis, :]
#         testPhaseLabels = np.concatenate((phase_test[1, :], phase_test[0, :]), axis=0)[np.newaxis, :]
#
#         # pre-shuffle
#         trainSpecs, trainSpecsFC, trainFreqLabels, trainPhaseLabels = shuffleData(trainSpecs, trainSpecsFC, trainFreqLabels, trainPhaseLabels)
#         valSpecs, valSpecsFC, valFreqLabels, valPhaseLabels = shuffleData(valSpecs, valSpecsFC, valFreqLabels, valPhaseLabels)
#
#         # get magnitude (freq models) or 1-channel (phase models) or 2-channel data (complex freq and phase models)
#         allSpecsTrain_2ChanComp, allSpecsTrain_Mag = get2Chann(trainSpecs), getMag(trainSpecs)
#         allSpecsVal_2ChanComp, allSpecsVal_Mag = get2Chann(valSpecs), getMag(valSpecs)
#         allSpecsTest_2ChanComp, allSpecsTest_Mag = get2Chann(testSpecs), getMag(testSpecs)
#
#         freqCorr_specsTrain, freqCorr_specsTrain_real = get2Chann(trainSpecsFC), get1Chann(trainSpecsFC)
#         freqCorr_specsVal, freqCorr_specsVal_real = get2Chann(valSpecsFC), get1Chann(valSpecsFC)
#         freqCorr_specsTest, freqCorr_specsTest_real = get2Chann(testSpecsFC), get1Chann(testSpecsFC)
#
#         # select 1024 window, convert to tensors and transfer operations to GPU
#         allSpecsTrain_2ChanCompTensor = torch.from_numpy(window1024(allSpecsTrain_2ChanComp, ppm)).float()
#         allSpecsTrain_MagTensor = torch.from_numpy(window1024(allSpecsTrain_Mag, ppm)).float()
#         freqCorrTrain_Tensor = torch.from_numpy(window1024(freqCorr_specsTrain, ppm)).float()
#         freqCorrTrain_RealTensor = torch.from_numpy(window1024(freqCorr_specsTrain_real, ppm)).float()
#         trainFreqLabelsTensor = torch.from_numpy(trainFreqLabels).float()
#         trainPhaseLabelsTensor = torch.from_numpy(trainPhaseLabels).float()
#
#         allSpecsVal_2ChanCompTensor = torch.from_numpy(window1024(allSpecsVal_2ChanComp, ppm)).float()
#         allSpecsVal_MagTensor = torch.from_numpy(window1024(allSpecsVal_Mag, ppm)).float()
#         freqCorrVal_Tensor = torch.from_numpy(window1024(freqCorr_specsVal, ppm)).float()
#         freqCorrVal_RealTensor = torch.from_numpy(window1024(freqCorr_specsVal_real, ppm)).float()
#         valFreqLabelsTensor = torch.from_numpy(valFreqLabels).float()
#         valPhaseLabelsTensor = torch.from_numpy(valPhaseLabels).float()
#
#         allSpecsTest_2ChanCompTensor = torch.from_numpy(window1024(allSpecsTest_2ChanComp, ppm)).float()
#         allSpecsTest_MagTensor = torch.from_numpy(window1024(allSpecsTest_Mag, ppm)).float()
#         freqCorr_specs_test_tensor = torch.from_numpy(window1024(freqCorr_specsTest, ppm)).float()
#         freqCorr_specs_test_real_tensor = torch.from_numpy(window1024(freqCorr_specsTest_real, ppm)).float()
#         testFreqLabelsTensor = torch.from_numpy(testFreqLabels).float()
#         testPhaseLabelsTensor = torch.from_numpy(testPhaseLabels).float()
#
# ########################################################################################################################
# # DATASETS AND MODEL SET-UP
# ########################################################################################################################
#         for model_name in modelTypes:
#             for net in netTypes:  # nets: frequency and phase
#                 run_name = f'{net}_Sim{snr}_{waterTypes[indW]}Water_{model_name}_FINAL'
#                 print(run_name)
#
#                 # select based on network type and assign data to dataloader
#                 if net == "freq":
#                     if (model_name == "Ma_4Convs") or (model_name == "realReal"):
#                         train_dataset = FPC_Dataset_1C(allSpecsTrain_MagTensor, trainFreqLabelsTensor)
#                         val_dataset = FPC_Dataset_1C(allSpecsVal_MagTensor, valFreqLabelsTensor)
#                         test_dataset = FPC_Dataset_1C(allSpecsTest_MagTensor, testFreqLabelsTensor)
#                     elif (model_name == "Tapper"):
#                         train_dataset = FPC_Dataset_Tapper(allSpecsTrain_MagTensor, trainFreqLabelsTensor)
#                         val_dataset = FPC_Dataset_Tapper(allSpecsVal_MagTensor, valFreqLabelsTensor)
#                         test_dataset = FPC_Dataset_Tapper(allSpecsTest_MagTensor, testFreqLabelsTensor)
#                     elif (model_name == "compReal") or ("compComp"):
#                         train_dataset = FPC_Dataset_2C(allSpecsTrain_2ChanCompTensor, trainFreqLabelsTensor)
#                         val_dataset = FPC_Dataset_2C(allSpecsVal_2ChanCompTensor, valFreqLabelsTensor)
#                         test_dataset = FPC_Dataset_2C(allSpecsTest_2ChanCompTensor, testFreqLabelsTensor)
#                     else:
#                         print("Frequency Model Not Found!")
#                         break
#
#                 elif net == "phase":
#                     if (model_name == "Ma_4Convs") or (model_name == "realReal"):
#                         train_dataset = FPC_Dataset_1C(freqCorrTrain_RealTensor, trainPhaseLabelsTensor)
#                         val_dataset = FPC_Dataset_1C(freqCorrVal_RealTensor, valPhaseLabelsTensor)
#                         test_dataset = FPC_Dataset_1C(freqCorr_specs_test_real_tensor, testPhaseLabelsTensor)
#                     elif (model_name == "Tapper"):
#                         train_dataset = FPC_Dataset_Tapper(freqCorrTrain_RealTensor, trainPhaseLabelsTensor)
#                         val_dataset = FPC_Dataset_Tapper(freqCorrVal_RealTensor, valPhaseLabelsTensor)
#                         test_dataset = FPC_Dataset_Tapper(freqCorr_specs_test_real_tensor, testPhaseLabelsTensor)
#                     elif (model_name == "compReal") or ("compComp"):
#                         train_dataset = FPC_Dataset_2C(freqCorrTrain_Tensor, trainPhaseLabelsTensor)
#                         val_dataset = FPC_Dataset_2C(freqCorrVal_Tensor, valPhaseLabelsTensor)
#                         test_dataset = FPC_Dataset_2C(freqCorr_specs_test_tensor, testPhaseLabelsTensor)
#                     else:
#                         print("Phase Model Not Found!")
#                         break
#
#                 train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
#                 val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
#                 test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#
#                 # load correct model
#                 if model_name == "Ma_4Convs":
#                     model = maModel().float()
#                 elif model_name == "Tapper":
#                     model = tapperModel().float()
#                 elif model_name == "compReal":
#                     model = compIn_realConv().float()
#                 elif model_name == "compComp":
#                     model = compIn_compConv().float()
#                 elif model_name == "realReal":
#                     model = realIn_realConv().float()
#                 else:
#                     print("Model Architecture Not Found!")
#                     break
#
# ########################################################################################################################
# # TRAINING LOOP
# ########################################################################################################################
#                 # initialize training loop parameters
#                 model.to(device)
#                 optimizer = optim.Adam(model.parameters(), lr=learn_r)
#                 lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda lr: 0.5)
#                 best_val_loss = float('inf')
#                 current_loss, prev_loss = 0.0, 0.0
#                 allTrainLosses, allValLosses = [], []
#
#                 # place model in train mode and move through epochs
#                 model.train()
#                 for epoch in range(nmb_epochs):
#                     loopTrain, loopVal = 0, 0
#                     epoch_lossesT, epoch_lossesV = [], []
#
#                     # for index, sample in enumerate(train_loader):
#                     for sample in (train_loader):
#                         # FORWARD (Model predictions and loss)
#                         Trainspecs, Trainlabels = sample
#                         Trainspecs, Trainlabels = Trainspecs.to(device), Trainlabels.to(device)
#
#                         optimizer.zero_grad()
#                         TrainPred = model(Trainspecs.float())
#                         TrainLoss = loss_fn(TrainPred, Trainlabels)
#
#                         # BACKWARD (Optimization) and UPDATE
#                         TrainLoss.backward()
#                         optimizer.step()
#                         epoch_lossesT.append(TrainLoss.item())
#
#                     if (epoch + 1) % lr_scheduler_freq == 0:
#                         lr_scheduler.step()
#
#                     # validate model in eval mode on validation data
#                     model.eval()
#                     with torch.no_grad():
#                         for sample in val_loader:
#                             ValSpecs, ValLabels = sample
#                             ValSpecs, ValLabels = ValSpecs.to(device), ValLabels.to(device)
#
#                             ValPred = model(ValSpecs.float())
#                             val_loss = loss_fn(ValPred, ValLabels)
#                             epoch_lossesV.append(val_loss.item())
#
#                         if val_loss < best_val_loss:
#                             best_weights = model.state_dict()
#                             best_val_loss = val_loss
#
#                     if (epoch ==0) or (epoch%10==0):
#                         print(f'Epoch {epoch + 1}/{nmb_epochs}, Training loss: {sum(epoch_lossesT) / len(epoch_lossesT)} and Validation loss: {sum(epoch_lossesV) / len(epoch_lossesV)}')
#                         print(f'Validation Loss Improvement of {prev_loss - current_loss}')
#
#                     allTrainLosses.append(sum(epoch_lossesT) / len(epoch_lossesT))
#                     allValLosses.append(sum(epoch_lossesV) / len(epoch_lossesV))
#                     prev_loss = current_loss
#
#                 print(f'Training Complete for {run_name}')
#                 print(f'Best Validation Loss was {best_val_loss}')
#                 print()
#
#                 torch.save(best_weights, f"{run_name}_SimForVivo.pt")
#
# #########################################################################################################################
# # TESTING LOOP
# #########################################################################################################################
#                 # Testing Loop parameter set-up
#                 epoch_lossesTest, predTestLabels = [], np.zeros((testPhaseLabels.shape[1]))
#                 lowestLoss, highestLoss, ind = 100000, 0, 0
#
#                 model.eval()
#                 with torch.no_grad():
#                     for sample in test_loader:
#                         # getting specs and labels from dataloader sample and passing them to GPU
#                         Testspecs, TestLabels = sample
#                         Testspecs, TestLabels = Testspecs.to(device), TestLabels.to(device)
#
#                         # passing specs to model
#                         predTest = model(Testspecs.float())
#
#                         # calculating loss (only based on artificially added offsets) and saving predictions
#                         test_loss = loss_fn(predTest, TestLabels)
#                         epoch_lossesTest.append(test_loss.item())
#                         predLabelsTest = predTest.cpu().flatten()
#
#                         if predLabelsTest.shape[0]==64:
#                             predTestLabels[ind:ind + batch_size] = predLabelsTest
#                         else:   #because test specs is not multiple of 64
#                             predTestLabels[ind:ind + 56] = predLabelsTest
#                         ind = ind + batch_size
#
#                         if test_loss > highestLoss:
#                             highestLoss = test_loss
#                         if test_loss < lowestLoss:
#                             lowestLoss = test_loss
#
#                     print(f'Testing Complete for {run_name}')
#                     print(f'Lowest test_loss {lowestLoss}, highest loss {highestLoss} and mean loss {sum(epoch_lossesTest) / len(epoch_lossesTest)}')
#                     print()
#
#                 # save labels
#                 np.save(f"PredLabels_{run_name}.npy", predTestLabels)



########################################################################################################################
########################################################################################################################
# IN VIVO TESTING
########################################################################################################################
########################################################################################################################
modelTypes = ["compReal"]
offsetSize, netTypes = ["Small", "Medium", "Large"], ["freq", "phase"]
t_inVivo = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/GTs/time_InVivo.npy")
ppmVivo = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/GTs/ppm_InVivo.npy")

# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
loss_fn = nn.L1Loss()
########################################################################################################################
# MODEL SELECTION
########################################################################################################################
# load correct model
for model_name in modelTypes:
    for net in netTypes:
        if model_name == "Ma_4Convs":
            model = maModel().float()
        elif model_name == "Tapper":
            model = tapperModel().float()
        elif model_name == "compReal":
            model = compIn_realConv().float()
        elif model_name == "compComp":
            model = compIn_compConv().float()
        else:
            print("Model test load not found!")
            break

        model.to(device)

#########################################################################################################################
# DATA PROCESSING
#########################################################################################################################
# Test based on offset size (small, medium, large)
        for size in offsetSize:
            indS = offsetSize.index(size)
            modelLoadName = f"{net}_Sim2_5_MixWater_{model_name}_Final_SimForVivo"
            run_name_test = f'{size}_{net}_InVivo_{model_name}'
            print(run_name_test)

            # load and norm data
            on_test, off_test, Freq_test, Phase_test = loadVivoData(offsetSize[indS])
            on_all_specs_test, off_all_specs_test = normSpecs(on_test, off_test)

            # Concatenate ON and OFF specs
            testSpecs = np.concatenate((on_all_specs_test, off_all_specs_test), axis=0)
            testFreqLabels = np.concatenate((Freq_test[1, :], Freq_test[0, :]))[np.newaxis, :]
            testPhaseLabels = np.concatenate((Phase_test[1, :], Phase_test[0, :]))[np.newaxis, :]

            # Magnitude of Data and 2-channel Complex Data
            allSpecsTest_2ChanComp, allSpecsTest_Mag = get2Chann(testSpecs), getMag(testSpecs)

            # select 1024 window, convert to tensors and transfer operations to GPU (may need to create deep copy)
            allSpecsTest_2ChanCompTensor = torch.from_numpy(window1024V(allSpecsTest_2ChanComp, ppmVivo)).float()
            allSpecsTest_MagTensor = torch.from_numpy(window1024V(allSpecsTest_Mag, ppmVivo)).float()
            testFreqLabelsTensor = torch.from_numpy(testFreqLabels).float()
            testPhaseLabelsTensor = torch.from_numpy(testPhaseLabels).float()

#########################################################################################################################
# DATASETS SET-UP
#########################################################################################################################
            # select network for testing
            if net == "freq":
                if model_name == "Ma_4Convs":
                    test_dataset = FPC_Dataset_1C(allSpecsTest_MagTensor, testFreqLabelsTensor)
                elif model_name == "Tapper":
                    test_dataset = FPC_Dataset_Tapper(allSpecsTest_MagTensor, testFreqLabelsTensor)
                elif (model_name == "compReal") or (model_name == "compComp"):
                    test_dataset = FPC_Dataset_2C(allSpecsTest_2ChanCompTensor, testFreqLabelsTensor)
                else:
                    print(f'Frequency Test Dataset not found!')

            elif net == "phase":
                # apply frequency correction
                predFLabels = np.load(f"PredLabels_{size}_freq_InVivo_{model_name}_{offsetSize[indS]}.npy")
                TestFids = toFids(testSpecs)
                TestFids = corrFShift(TestFids, t_inVivo, predFLabels)
                testSpecsFC = toSpecs(TestFids)

                fig1, ax1 = plt.subplots(1)
                ax1.plot(ppmVivo, testSpecs[4, :], 'blue')
                ax1.plot(ppmVivo, testSpecsFC[4, :], 'red')
                plt.show()

                # get 1 & 2 channel data and pass to GPU
                freqCorr_specsTest, freqCorr_specsTest_real = get2Chann(testSpecsFC), get1Chann(testSpecsFC)
                freqCorr_specs_test_tensor = torch.from_numpy(window1024V(freqCorr_specsTest, ppmVivo)).float()
                freqCorr_specs_test_real_tensor = torch.from_numpy(window1024V(freqCorr_specsTest_real, ppmVivo)).float()

                # select model and give data to dataloader
                if model_name == "Ma_4Convs":
                    test_dataset = FPC_Dataset_1C(freqCorr_specs_test_real_tensor,  testPhaseLabelsTensor)
                elif model_name == "Tapper":
                    test_dataset = FPC_Dataset_Tapper(freqCorr_specs_test_real_tensor, testPhaseLabelsTensor)
                elif (model_name == "compReal") or (model_name == "compComp"):
                    test_dataset = FPC_Dataset_2C(freqCorr_specs_test_tensor, testPhaseLabelsTensor)
                else:  # assumed to be CR-CNN
                    print(f'Phase Test Dataset not found!')

            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#########################################################################################################################
# IN VIVO TESTING LOOP
#########################################################################################################################
            # Testing Loop parameter set-up
            epoch_lossesTest, predTestLabels = [], np.zeros((testPhaseLabels.shape[1]))
            lowestLoss, highestLoss, indV = 100000, 0, 0
            model.load_state_dict(torch.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/{modelLoadName}.pt"))
            model.eval()

            with torch.no_grad():
                for sample in test_loader:
                    # getting specs and labels from dataloader sample and passing them to GPU
                    Testspecs, TestLabels = sample
                    Testspecs, TestLabels = Testspecs.to(device), TestLabels.to(device)

                    # passing specs to model
                    predTest = model(Testspecs.float())

                    # calculating loss (only based on artificially added offsets) and saving predictions
                    test_loss = loss_fn(predTest, TestLabels)
                    epoch_lossesTest.append(test_loss.item())
                    predTestLabels[indV:indV + batch_size] = predTest.cpu().flatten()
                    indV = indV + batch_size

                    if test_loss > highestLoss:
                        highestLoss = test_loss
                    if test_loss < lowestLoss:
                        lowestLoss = test_loss

                print(f'Testing Complete for {run_name_test}')
                print(f'Lowest test_loss {lowestLoss}, highest loss {highestLoss} and mean loss {sum(epoch_lossesTest) / len(epoch_lossesTest)}')
                print()

            # save labels
            np.save(f"PredLabels_{run_name_test}_{offsetSize[indS]}.npy", predTestLabels)
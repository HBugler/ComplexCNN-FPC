# FPC Winter 2024
from random import shuffle
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Datasets import FPC_Dataset_Ma, FPC_Dataset_Tapper, FPC_Dataset_compReal
from AblationStudyModels import compIn_realConv, compIn_compConv
from TapperModel import tapperModel
from MaModel import maModel

########################################################################################################################
# Set Up Loop
########################################################################################################################
# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelTypes = ["compComp"]  # currently missing ["Ma_2Convs", "realReal", "Tapper", "compComp", "Ma_4Convs", "compReal"]
netTypes = ["freq", "phase"]  # ["freq", "phase"]
offsetSize = ["Small", "Medium", "Large"]

batch_size, learn_r = 64, 0.0001 #used to be smaller
nmb_epochs = [200]  # 200, 30
loss_fn = nn.L1Loss()
lr_scheduler_freq = 25
predTrainLabels, predValLabels = 0, 0

#########################################################################################################################
# DEVSET SETUP
#########################################################################################################################
# load data
dataDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/"
ppm = np.loadtxt(f"{dataDir}Simulated/GTs/Development/ppm_2048.csv", delimiter=",")[:, 0].flatten()[::-1]
t = np.loadtxt(f"{dataDir}Simulated/GTs/Development/t_2048.csv", delimiter=",")[:, 0].flatten()

dataType = "Mix"
on_all_specs_dev = np.load(f"{dataDir}Simulated/Corrupt/Sim2_5_{dataType}/ON_AllSpecs_Sim2_5_{dataType}_Dev.npy")
off_all_specs_dev = np.load(f"{dataDir}Simulated/Corrupt/Sim2_5_{dataType}/OFF_AllSpecs_Sim2_5_{dataType}_Dev.npy")
phaseLabels_dev = np.load(f"{dataDir}Simulated/Corrupt/Sim2_5_{dataType}/TruePhaseLabels_Sim2_5_{dataType}_Dev.npy")[:, 0, :]          #(ON/OFF, 1, #trans)
freqLabels_dev = np.load(f"{dataDir}Simulated/Corrupt/Sim2_5_{dataType}/TrueFreqLabels_Sim2_5_{dataType}_Dev.npy")[:, 0, :]

# Spectra Normalization
# Percentile
on_all_specs_dev = (on_all_specs_dev - (np.percentile(np.abs(on_all_specs_dev), 5, axis=0, keepdims=True))) / ((np.percentile(np.abs(on_all_specs_dev), 95, axis=0, keepdims=True)) - (np.percentile(np.abs(on_all_specs_dev), 5, axis=0, keepdims=True)))
off_all_specs_dev = (off_all_specs_dev - (np.percentile(np.abs(off_all_specs_dev), 5, axis=0, keepdims=True))) / ((np.percentile(np.abs(off_all_specs_dev), 95, axis=0, keepdims=True)) - (np.percentile(np.abs(off_all_specs_dev), 5, axis=0, keepdims=True)))

# # Max-min
# on_all_specs_dev = ((on_all_specs_dev - np.min(np.abs(on_all_specs_dev), axis=0)) / (np.max(np.abs(on_all_specs_dev), axis=0) - np.min(np.abs(on_all_specs_dev), axis=0)))
# off_all_specs_dev = ((off_all_specs_dev - np.min(np.abs(off_all_specs_dev), axis=0)) / (np.max(np.abs(off_all_specs_dev), axis=0) - np.min(np.abs(off_all_specs_dev), axis=0)))

# on_all_specs_dev = on_all_specs_dev - np.median(np.abs(on_all_specs_dev), axis=0)
# off_all_specs_dev = off_all_specs_dev - np.median(np.abs(off_all_specs_dev), axis=0)

# Window 1024 Points Starting at 0 ppm
simStart, simFinish = np.where(ppm <= 0.00)[0][-1], np.where(ppm >= 7.83)[0][0]-1
t = t[simStart:simFinish]
ppm = ppm[simStart:simFinish]
on_all_specs_dev = on_all_specs_dev[simStart:simFinish, :]
off_all_specs_dev = off_all_specs_dev[simStart:simFinish, :]

# Separate train/val evenly
# reduce to a single dimension by removing np.newaxis
trainSpecs = np.swapaxes(np.concatenate((on_all_specs_dev[:, :int(0.8*on_all_specs_dev.shape[1])], off_all_specs_dev[:, :int(0.8*off_all_specs_dev.shape[1])]), axis=1), axis1=1, axis2=0)
trainFreqLabels = np.concatenate((freqLabels_dev[1, :int(0.8*on_all_specs_dev.shape[1])], freqLabels_dev[0, :int(0.8*off_all_specs_dev.shape[1])]), axis=0)[np.newaxis, :]
trainPhaseLabels = np.concatenate((phaseLabels_dev[1, :int(0.8*on_all_specs_dev.shape[1])], phaseLabels_dev[0, :int(0.8*off_all_specs_dev.shape[1])]), axis=0)[np.newaxis, :]

valSpecs = np.swapaxes(np.concatenate((on_all_specs_dev[:, int(0.8*on_all_specs_dev.shape[1]):], off_all_specs_dev[:, int(0.8*off_all_specs_dev.shape[1]):]), axis=1), axis1=1, axis2=0)
valFreqLabels = np.concatenate((freqLabels_dev[1, int(0.8*on_all_specs_dev.shape[1]):], freqLabels_dev[0, int(0.8*off_all_specs_dev.shape[1]):]), axis=0)[np.newaxis, :]
valPhaseLabels = np.concatenate((phaseLabels_dev[1, int(0.8*on_all_specs_dev.shape[1]):], phaseLabels_dev[0, int(0.8*off_all_specs_dev.shape[1]):]), axis=0)[np.newaxis, :]

# Manual Sorting of ON and OFF transients (ON=1 (Sim) and ON=1 (Vivo))
index_shufTrain = list(range(trainSpecs.shape[0]))
index_shufVal = list(range(valSpecs.shape[0]))
shuffle(index_shufTrain), shuffle(index_shufVal)
indOrderTrain, indOrderVal = 0, 0

for indRandTrain in index_shufTrain:
    trainSpecs[indOrderTrain, :] = trainSpecs[indRandTrain, :]
    trainFreqLabels[:, indOrderTrain] = trainFreqLabels[:, indRandTrain]
    trainPhaseLabels[:, indOrderTrain] = trainPhaseLabels[:, indRandTrain]
    indOrderTrain = indOrderTrain + 1

for indRandVal in index_shufVal:
    valSpecs[indOrderVal, :] = valSpecs[indRandVal, :]
    valFreqLabels[:, indOrderVal] = valFreqLabels[:, indRandVal]
    valPhaseLabels[:, indOrderVal] = valPhaseLabels[:, indRandVal]
    indOrderVal = indOrderVal + 1

# Magnitude of Data and 2-channel Complex Data
allSpecsTrain_2ChanComp = np.empty((2, trainSpecs.shape[0], trainSpecs.shape[1]))
allSpecsTrain_2ChanComp[0, :, :], allSpecsTrain_2ChanComp[1, :, :] = trainSpecs.real, trainSpecs.imag
allSpecsTrain_Mag = np.abs(trainSpecs)[np.newaxis, :, :]

allSpecsVal_2ChanComp = np.empty((2, valSpecs.shape[0], valSpecs.shape[1]))
allSpecsVal_2ChanComp[0, :, :], allSpecsVal_2ChanComp[1, :, :] = valSpecs.real, valSpecs.imag
allSpecsVal_Mag = np.abs(valSpecs)[np.newaxis, :, :]

# Convert to tensors and transfer operations to GPU
allSpecsTrain_2ChanCompTensor = torch.from_numpy(allSpecsTrain_2ChanComp).float()  # shape is (#channels, #samples, #spectralPoints)
allSpecsTrain_MagTensor = torch.from_numpy(allSpecsTrain_Mag).float()
trainFreqLabelsTensor = torch.from_numpy(trainFreqLabels).float()
trainPhaseLabelsTensor = torch.from_numpy(trainPhaseLabels).float()

allSpecsVal_2ChanCompTensor = torch.from_numpy(allSpecsVal_2ChanComp).float()
allSpecsVal_MagTensor = torch.from_numpy(allSpecsVal_Mag).float()
valFreqLabelsTensor = torch.from_numpy(valFreqLabels).float()
valPhaseLabelsTensor = torch.from_numpy(valPhaseLabels).float()

########################################################################################################################
# Set-up Datasets, Dataloaders and Transforms
########################################################################################################################
# set-up arrays for frequency corrected specs after frequency network before phase network
freqCorr_specsTrain = np.zeros(shape=(2, allSpecsTrain_2ChanComp.shape[1], 1024))
freqCorr_specsVal = np.zeros(shape=(2, allSpecsVal_2ChanComp.shape[1], 1024))

for epochNUMS in nmb_epochs:
    for model_name in modelTypes:                                           # models: MLP, CNN, CR-CNN
        for net in netTypes:                                                # nets: frequency and phase
            run_name = f'{net}_InVivo_{model_name}_lr25Percent_B64Adam_E200'
            print(run_name)
            predTrainLabels, predValLabels = 0, 0

            # select based on network type
            if net == "freq":
                # select based on model (real refers to magnitude in the case of frequency network), each network has its own dataloader
                if model_name == "Ma_4Convs":
                    train_dataset = FPC_Dataset_Ma(allSpecsTrain_MagTensor, trainFreqLabelsTensor)
                    val_dataset = FPC_Dataset_Ma(allSpecsVal_MagTensor, valFreqLabelsTensor)
                elif model_name == "Tapper":
                    train_dataset = FPC_Dataset_Tapper(allSpecsTrain_MagTensor, trainFreqLabelsTensor)
                    val_dataset = FPC_Dataset_Tapper(allSpecsVal_MagTensor, valFreqLabelsTensor)
                elif model_name == "compReal":
                    train_dataset = FPC_Dataset_compReal(allSpecsTrain_2ChanCompTensor, trainFreqLabelsTensor)
                    val_dataset = FPC_Dataset_compReal(allSpecsVal_2ChanCompTensor, valFreqLabelsTensor)
                elif model_name == "compComp":
                    print('freq train compComp')
                    train_dataset = FPC_Dataset_compReal(allSpecsTrain_2ChanCompTensor, trainFreqLabelsTensor)
                    val_dataset = FPC_Dataset_compReal(allSpecsVal_2ChanCompTensor, valFreqLabelsTensor)
                else:
                    print("Model not found!")
                    break


            elif net == "phase":
                # Correct based on frequency labels
                TrainFids = np.fft.fft(np.fft.fftshift(trainSpecs, axes=1), axis=1)
                ValFids = np.fft.fft(np.fft.fftshift(valSpecs, axes=1), axis=1)

                for k in range(0, trainFreqLabels.shape[1]):
                    TrainFids[k, :] = TrainFids[k, :] * np.exp(-1j * t[:] * (-trainFreqLabels[0, k]) * 2 * math.pi)

                for p in range(0, valFreqLabels.shape[1]):
                    ValFids[p, :] = ValFids[p, :] * np.exp(-1j * t[:] * (-valFreqLabels[0, p]) * 2 * math.pi)

                # Convert fids back to specs for phase network (get real value only for some models)
                TrainFids = np.fft.fftshift(np.fft.ifft(TrainFids, axis=1), axes=1)
                freqCorr_specsTrain[0, :, :], freqCorr_specsTrain[1, :, :] = TrainFids.real, TrainFids.imag
                freqCorr_specsTrain_real = (freqCorr_specsTrain[0, :, :])[np.newaxis, :, :]

                ValFids = np.fft.fftshift(np.fft.ifft(ValFids, axis=1), axes=1)
                freqCorr_specsVal[0, :, :], freqCorr_specsVal[1, :, :] = ValFids.real, ValFids.imag
                freqCorr_specsVal_real = (freqCorr_specsVal[0, :, :])[np.newaxis, :, :]

                # create tensor from data
                freqCorrTrain_Tensor = torch.from_numpy(freqCorr_specsTrain).float()
                freqCorrTrain_RealTensor = torch.from_numpy(freqCorr_specsTrain_real).float()
                freqCorrVal_Tensor = torch.from_numpy(freqCorr_specsVal).float()
                freqCorrVal_RealTensor = torch.from_numpy(freqCorr_specsVal_real).float()

                # load data into proper dataloader based on model
                if model_name == "Ma_4Convs":
                    train_dataset = FPC_Dataset_Ma(freqCorrTrain_RealTensor, trainPhaseLabelsTensor)
                    val_dataset = FPC_Dataset_Ma(freqCorrVal_RealTensor, valPhaseLabelsTensor)

                elif model_name == "Tapper":
                    train_dataset = FPC_Dataset_Tapper(freqCorrTrain_RealTensor, trainPhaseLabelsTensor)
                    val_dataset = FPC_Dataset_Tapper(freqCorrVal_RealTensor, valPhaseLabelsTensor)

                elif model_name=="compComp":
                    print('phase train compComp')
                    train_dataset = FPC_Dataset_compReal(freqCorrTrain_Tensor, trainPhaseLabelsTensor)
                    val_dataset = FPC_Dataset_compReal(freqCorrVal_Tensor, valPhaseLabelsTensor)
                else:
                    train_dataset = FPC_Dataset_compReal(freqCorrTrain_Tensor, trainPhaseLabelsTensor)
                    val_dataset = FPC_Dataset_compReal(freqCorrVal_Tensor, valPhaseLabelsTensor)

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)  #changed to false due to pre-sort
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

########################################################################################################################
# Model and Loss Function
########################################################################################################################
            # load appropriate model
            if model_name == "Ma_4Convs":
                model = maModel().float()
            elif model_name == "Tapper":
                model = tapperModel().float()
            elif model_name == "compReal":
                model = compIn_realConv().float()
            elif model_name=="compComp":
                print('model selection compComp')
                model = compIn_compConv().float()
            else:
                print("Model not found!")
                break

            # load model to GPU and set-up optimizer and scheduler
            model.to(device)
            print(f'model number of parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
            optimizer = optim.Adam(model.parameters(), lr=learn_r)
            lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda lr: 0.5)

########################################################################################################################
# Training
########################################################################################################################
            # initialize training loop parameters
            best_val_loss = float('inf')
            current_loss, prev_loss = 0.0, 0.0
            bestModelFound, count, early_stop = False, 0, False
            allTrainLosses = []
            allValLosses = []

            # place model in train mode and move through epochs
            model.train()
            for epoch in range(epochNUMS):
                loopTrain, loopVal = 0, 0
                epoch_lossesT, epoch_lossesV = [], []

                # for index, sample in enumerate(train_loader):
                for sample in (train_loader):

                    # FORWARD (Model predictions and loss)
                    Trainspecs, Trainlabels = sample
                    Trainspecs, Trainlabels = Trainspecs.to(device), Trainlabels.to(device)

                    optimizer.zero_grad()
                    TrainPred = model(Trainspecs.float())
                    TrainLoss = loss_fn(TrainPred, Trainlabels)

                    # BACKWARD (Optimization) and UPDATE
                    TrainLoss.backward()
                    optimizer.step()
                    epoch_lossesT.append(TrainLoss.item())

                if (epoch + 1) % lr_scheduler_freq == 0:
                    lr_scheduler.step()
                    print(f'LR halved')

                # validate model in eval mode on validation data
                model.eval()
                with torch.no_grad():
                    for sample in val_loader:
                        ValSpecs, ValLabels = sample
                        ValSpecs, ValLabels = ValSpecs.to(device), ValLabels.to(device)

                        ValPred = model(ValSpecs.float())
                        val_loss = loss_fn(ValPred, ValLabels)
                        epoch_lossesV.append(val_loss.item())

                    if val_loss < best_val_loss:
                        best_weights = model.state_dict()
                        best_val_loss = val_loss

                    # early stopping according to training loss to prevent overfitting
                    current_loss = sum(epoch_lossesT) / len(epoch_lossesT)
                    if (((epoch + 1) > 25) and ((prev_loss - current_loss) > -0.00000001) and (
                            (prev_loss - current_loss) < 0.00000001)):
                        count = count + 1
                        if count == 5:
                            early_stop = True
                            print('Early Stop Criteria Reached!')  # validation loss hasn't improved in 5 consecutive epochs
                            break
                    else:
                        count = 0



                print(f'Epoch {epoch + 1}/{epochNUMS}, Training loss: {sum(epoch_lossesT) / len(epoch_lossesT)} and Validation loss: {sum(epoch_lossesV) / len(epoch_lossesV)}')
                allTrainLosses.append(sum(epoch_lossesT) / len(epoch_lossesT))
                allValLosses.append(sum(epoch_lossesV) / len(epoch_lossesV))

                print(f'Validation Loss Improvement of {prev_loss - current_loss}')
                prev_loss = current_loss

            print(f'Training Complete for {run_name}, Early Stopping {early_stop} at epoch {epoch + 1}')
            print(f'Best Validation Loss was {best_val_loss}')
            print()

            torch.save(best_weights, f"{run_name}_SimForVivo.pt")

            # # sanity check (visualize data) - Train and Val Curves
            # fig1, ax1 = plt.subplots(1)
            # ax1.plot(allTrainLosses, 'blue')
            # ax1.plot(allValLosses, 'orange')
            # plt.show()

#########################################################################################################################
# TEST SETUP
#########################################################################################################################
    # Test based on offset size (small, medium, large)
            for size in offsetSize:
                indS = offsetSize.index(size)

                run_name_test = f'{size}_{net}_InVivo_{model_name}'
                print(run_name_test)

                # load data
                dataVDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/"
                t_inVivo = np.load(f"{dataVDir}InVivo/time_InVivo.npy").flatten()
                ppmVivo = np.load(f"{dataVDir}InVivo/ppm_InVivo.npy").flatten()

                on_all_specs_test = np.load(f"{dataVDir}InVivo/allSpecsInVivoON_{offsetSize[indS]}Offsets.npy")
                off_all_specs_test = np.load(f"{dataVDir}InVivo/allSpecsInVivoOFF_{offsetSize[indS]}Offsets.npy")
                trueFreqLabels_test = np.load(f"{dataVDir}InVivo/FnoiseInVivo_{offsetSize[indS]}Offsets.npy")
                truePhaseLabels_test = np.load(f"{dataVDir}InVivo/PnoiseInVivo_{offsetSize[indS]}Offsets.npy")

                # Spectra Normalization
                # Percentile
                on_all_specs_test = (on_all_specs_test - (np.percentile(np.abs(on_all_specs_test), 5, axis=2, keepdims=True))) / ((np.percentile(np.abs(on_all_specs_test), 95, axis=2, keepdims=True)) - (np.percentile(np.abs(on_all_specs_test), 5, axis=2, keepdims=True)))
                off_all_specs_test = (off_all_specs_test - (np.percentile(np.abs(off_all_specs_test), 5, axis=2, keepdims=True))) / ((np.percentile(np.abs(off_all_specs_test), 95, axis=2, keepdims=True)) - (np.percentile(np.abs(off_all_specs_test), 5, axis=2, keepdims=True)))

                # # Max-min
                # on_all_specs_test = ((on_all_specs_test - np.min(np.abs(on_all_specs_test), axis=2, keepdims=True)) / (np.max(np.abs(on_all_specs_test), axis=2, keepdims=True) - np.min(np.abs(on_all_specs_test), axis=2, keepdims=True)))
                # off_all_specs_test = ((off_all_specs_test - np.min(np.abs(off_all_specs_test), axis=2, keepdims=True)) / (np.max(np.abs(off_all_specs_test), axis=2, keepdims=True) - np.min(np.abs(off_all_specs_test), axis=2, keepdims=True)))

                # on_all_specs_test = on_all_specs_test - np.median(np.abs(on_all_specs_test), axis=1)
                # off_all_specs_test = off_all_specs_test - np.median(np.abs(off_all_specs_test), axis=1)


                # Window 1024 Points Starting at 0 ppm
                ivStart, ivFinish = np.where(ppmVivo <= 0.00)[0][-1], np.where(ppmVivo >= 7.83)[0][0]-1
                t_inVivo = t_inVivo[ivStart:ivFinish]
                ppmVivo = ppmVivo[ivStart:ivFinish]
                on_all_specs_test = on_all_specs_test[:, :, ivStart:ivFinish]
                off_all_specs_test = off_all_specs_test[:, :, ivStart:ivFinish]

                # Concatenate ON and OFF specs
                testSpecs = np.concatenate((on_all_specs_test, off_all_specs_test), axis=1)
                testFreqLabels = np.concatenate((trueFreqLabels_test[1, :], trueFreqLabels_test[0, :]))
                testPhaseLabels = np.concatenate((truePhaseLabels_test[1, :], truePhaseLabels_test[0, :]))

                # # Manual Sorting (ON=1 (Sim) and ON=1 (Vivo))
                # index_shufTest = list(range(testSpecs.shape[1]))
                # shuffle(index_shufTest)
                # indOrderTest = 0
                #
                # for indRandTest in index_shufTest:
                #     testSpecs[:, indOrderTest, :] = testSpecs[:, indRandTest, :]
                #     testFreqLabels[indOrderTest] = testFreqLabels[indRandTest]
                #     testPhaseLabels[indOrderTest] = testPhaseLabels[indRandTest]
                #     indOrderTest = indOrderTest + 1
                #
                # np.save(f'{run_name}_TrueFreqLabelsInVivo.npy', testFreqLabels)
                # np.save(f'{run_name}_TruePhaseLabelsInVivo.npy', testPhaseLabels)
                # np.save(f'{run_name}_SpecsInVivo.npy', testSpecs)
                #
                # # sanity check
                # for i in range(0, 1):
                #     randScan = random.randint(0, on_all_specs_test.shape[1])
                #     print(f'Scan number {randScan}')
                #     fig1, (ax1, ax2) = plt.subplots(2)
                #     ax1.plot(trainSpecs[randScan, :].real, 'blue')
                #     ax1.plot(valSpecs[randScan, :].real, 'orange')
                #     ax1.plot(on_all_specs_test[0, randScan, :].real, 'green')
                #     ax1.plot(off_all_specs_test[0, randScan, :].real, 'black')
                #     ax1.invert_xaxis()
                #     ax2.plot(ppm, trainSpecs[randScan, :].real, 'blue')
                #     ax2.plot(ppm, valSpecs[randScan, :].real, 'orange')
                #     ax2.plot(ppmVivo, on_all_specs_test[0, randScan, :].real, 'green')
                #     ax2.plot(ppmVivo, off_all_specs_test[0, randScan, :].real, 'black')
                #     ax2.invert_xaxis()
                #     plt.show()

                # remove to reduce to a single dimension (would need to change dataset too)
                testFreqLabels = testFreqLabels[np.newaxis, :]
                testPhaseLabels = testPhaseLabels[np.newaxis, :]

                # Magnitude of Data and 2-channel Complex Data
                allSpecsTest_2ChanComp = np.empty((2, testSpecs.shape[1], testSpecs.shape[2]))
                allSpecsTest_2ChanComp[0, :, :], allSpecsTest_2ChanComp[1, :, :] = testSpecs.real, testSpecs.imag
                allSpecsTest_Mag = np.abs(testSpecs)

                # Convert to tensors and transfer operations to GPU
                allSpecsTest_2ChanCompTensor = torch.from_numpy(allSpecsTest_2ChanComp).float()  # shape is (#channels, #samples, #spectralPoints)
                allSpecsTest_MagTensor = torch.from_numpy(allSpecsTest_Mag).float()
                testFreqLabelsTensor = torch.from_numpy(testFreqLabels).float()
                testPhaseLabelsTensor = torch.from_numpy(testPhaseLabels).float()

                # select network for testing
                if net=="freq":
                    # select model
                    if model_name == "Ma_4Convs":
                        test_dataset = FPC_Dataset_Ma(allSpecsTest_MagTensor, testFreqLabelsTensor)

                    elif model_name == "Tapper":
                        test_dataset = FPC_Dataset_Tapper(allSpecsTest_MagTensor, testFreqLabelsTensor)
                    elif model_name=="compComp":
                        print('test freq compComp')
                        test_dataset = FPC_Dataset_compReal(allSpecsTest_2ChanCompTensor, testFreqLabelsTensor)
                    else:   # assumed to be CR-CNN
                        test_dataset = FPC_Dataset_compReal(allSpecsTest_2ChanCompTensor, testFreqLabelsTensor)

                elif net == "phase":
                    # apply frequency correction
                    freqCorr_specsTest = np.zeros(shape=(2, allSpecsTest_2ChanComp.shape[1], 1024))
                    TestFids = np.fft.fft(np.fft.fftshift(testSpecs, axes=2), axis=2)

                    # perform frequency correction based on true labels (only artificially added offsets)
                    for y in range(0, testFreqLabels.shape[1]):
                        TestFids[:, y, :] = TestFids[:, y, :] * np.exp(-1j * t_inVivo[:] * (-testFreqLabels[:, y]) * 2 * math.pi)

                    # convert back to specs and select real values for certain models
                    TestFids = np.fft.fftshift(np.fft.ifft(TestFids, axis=2), axes=2)
                    freqCorr_specsTest[0, :, :], freqCorr_specsTest[1, :, :] = TestFids.real, TestFids.imag
                    freqCorr_specsTest_real = (freqCorr_specsTest[0, :, :])[np.newaxis, :, :]
                    freqCorr_specs_test_tensor = torch.from_numpy(freqCorr_specsTest).float()
                    freqCorr_specs_test_real_tensor = torch.from_numpy(freqCorr_specsTest_real).float()

                    # select model and give data to dataloader
                    if model_name == "Ma_4Convs":
                        test_dataset = FPC_Dataset_Ma(freqCorr_specs_test_real_tensor, testPhaseLabelsTensor)
                    elif model_name == "Tapper":
                        test_dataset = FPC_Dataset_Tapper(freqCorr_specs_test_real_tensor, testPhaseLabelsTensor)
                    elif model_name=="compComp":
                        print('test phase compComp')
                        test_dataset = FPC_Dataset_compReal(freqCorr_specs_test_tensor, testPhaseLabelsTensor)
                    else: # assumed to be CR-CNN
                        test_dataset = FPC_Dataset_compReal(freqCorr_specs_test_tensor, testPhaseLabelsTensor)

                test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#########################################################################################################################
# Test Loop
#########################################################################################################################
                # Testing Loop parameter set-up
                epoch_lossesTest, predTestLabels = [], np.zeros((testPhaseLabels.shape[1]))
                loop, ind5 = 0, 0
                lowestLoss, highestLoss = 100000, 0
                predLabels = np.zeros(testSpecs.shape[1])
                model.eval()

                with torch.no_grad():
                    for sample in test_loader:
                        # getting specs and labels from dataloader sample and passing them to GPU
                        Testspecs, TestLabels = sample
                        Testspecs, TestLabels = Testspecs.to(device), TestLabels.to(device)

                        # passing specs to model
                        predTest = model(Testspecs.float())

                        if ind5%60==0:
                            print(predTest)
                        ind5 = ind5+1

                        # calculating loss (only based on artificially added offsets) and saving predictions
                        test_loss = loss_fn(predTest, TestLabels)
                        epoch_lossesTest.append(test_loss.item())
                        predTestLabels[loop:loop + batch_size] = predTest.cpu().flatten()
                        loop = loop + batch_size

                        if test_loss>highestLoss:
                            highestLoss = test_loss
                        if test_loss<lowestLoss:
                            lowestLoss = test_loss

                    # print(f'Testing Complete for {run_name_test}')
                    print(f'Testing Complete for {run_name}')
                    print(f'Lowest test_loss {lowestLoss}, highest loss {highestLoss} and mean loss {sum(epoch_lossesTest)/len(epoch_lossesTest)}')
                    print()

                # save labels
                # np.save(f"PredLabels_{run_name_test}_{epochNUMS}NIGHT_Sim2_5.npy", predTestLabels)
                np.save(f"PredLabels_{run_name}_{offsetSize[indS]}.npy", predTestLabels)
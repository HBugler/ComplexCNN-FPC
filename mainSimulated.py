# FPC Winter 2024 (03_18_2024)
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Datasets import FPC_Dataset_Tapper, FPC_Dataset_1C, FPC_Dataset_2C
from AblationStudyModels import compIn_realConv, compIn_compConv, realIn_realConv
from TapperModel import tapperModel
from MaModel import maModel

########################################################################################################################
# Set Up Loop
########################################################################################################################
# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelTypes = ["compReal", "Ma_4Convs", "Tapper"]
# whole list of models ["Ma_2Convs", "realReal", "Tapper", "compComp", "Ma_4Convs", "compReal"]
netTypes = ["phase"]
dataTypes = ["Mix", "None", "Pos"]
snrTypes = ["2_5", "5", "10"]

batch_size, learn_r = 64, 0.001
nmb_epochs = 200
loss_fn = nn.L1Loss()
lr_scheduler_freq = 25
predTrainLabels, predValLabels = 0, 0

#########################################################################################################################
# DEVSET SETUP
#########################################################################################################################
# load data
dataDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/Corrupt/"

for water in dataTypes:
    indW = dataTypes.index(water)
    for snr in snrTypes:
        indS = snrTypes.index(snr)
        ppm = np.load("C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/GTs/ppm_Sim.npy")
        t = np.load("C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/GTs/time_Sim.npy")

        on_all_specs_dev = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/ON_AllSpecs_Sim{snrTypes[indS]}_{dataTypes[indW]}_DevFinal.npy")
        off_all_specs_dev = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/OFF_AllSpecs_Sim{snrTypes[indS]}_{dataTypes[indW]}_DevFinal.npy")

        phaseLabels_dev = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/TruePhaseLabels_Sim{snrTypes[indS]}_{dataTypes[indW]}_DevFinal.npy")
        freqLabels_dev = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/TrueFreqLabels_Sim{snrTypes[indS]}_{dataTypes[indW]}_DevFinal.npy")

        on_all_specs_test = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/ON_AllSpecs_Sim{snrTypes[indS]}_{dataTypes[indW]}_testFinal.npy")
        off_all_specs_test = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/OFF_AllSpecs_Sim{snrTypes[indS]}_{dataTypes[indW]}_testFinal.npy")

        freqLabels_test = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/TrueFreqLabels_Sim{snrTypes[indS]}_{dataTypes[indW]}_testFinal.npy")
        phaseLabels_test = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/TruePhaseLabels_Sim{snrTypes[indS]}_{dataTypes[indW]}_testFinal.npy")

        on_all_specs_devFC = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/ON_AllSpecsFC_Sim{snrTypes[indS]}_{dataTypes[indW]}_DevFinal.npy")
        off_all_specs_devFC = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/OFF_AllSpecsFC_Sim{snrTypes[indS]}_{dataTypes[indW]}_DevFinal.npy")
        on_all_specs_testFC = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/ON_AllSpecsFC_Sim{snrTypes[indS]}_{dataTypes[indW]}_testFinal.npy")
        off_all_specs_testFC = np.load(f"{dataDir}Sim{snrTypes[indS]}_{dataTypes[indW]}/OFF_AllSpecsFC_Sim{snrTypes[indS]}_{dataTypes[indW]}_testFinal.npy")

        # Normalization
        on_all_specs_dev = ((on_all_specs_dev) / ((np.percentile(np.abs(on_all_specs_dev), 95, axis=0, keepdims=True))))
        off_all_specs_dev = ((off_all_specs_dev) / ((np.percentile(np.abs(off_all_specs_dev), 95, axis=0, keepdims=True))))
        on_all_specs_devFC = ((on_all_specs_devFC) / ((np.percentile(np.abs(on_all_specs_devFC), 95, axis=0, keepdims=True))))
        off_all_specs_devFC = ((off_all_specs_devFC) / ((np.percentile(np.abs(off_all_specs_devFC), 95, axis=0, keepdims=True))))

        on_all_specs_test = ((on_all_specs_test) / ((np.percentile(np.abs(on_all_specs_test), 95, axis=0, keepdims=True))))
        off_all_specs_test = ((off_all_specs_test) / ((np.percentile(np.abs(off_all_specs_test), 95, axis=0, keepdims=True))))
        on_all_specs_testFC = ((on_all_specs_testFC) / ((np.percentile(np.abs(on_all_specs_testFC), 95, axis=0, keepdims=True))))
        off_all_specs_testFC = ((off_all_specs_testFC) / ((np.percentile(np.abs(off_all_specs_testFC), 95, axis=0, keepdims=True))))

        # Separate train/val evenly
        trainSpecs = np.swapaxes(np.concatenate((on_all_specs_dev[:, :int(0.8 * on_all_specs_dev.shape[1])],
                                                 off_all_specs_dev[:, :int(0.8 * off_all_specs_dev.shape[1])]), axis=1), axis1=1, axis2=0)
        trainFreqLabels = np.concatenate((freqLabels_dev[1, :int(0.8 * on_all_specs_dev.shape[1])],
                                          freqLabels_dev[0, :int(0.8 * off_all_specs_dev.shape[1])]), axis=0)[np.newaxis, :]
        trainPhaseLabels = np.concatenate((phaseLabels_dev[1, :int(0.8 * on_all_specs_dev.shape[1])],
                                          phaseLabels_dev[0, :int(0.8 * off_all_specs_dev.shape[1])]), axis=0)[np.newaxis, :]

        valSpecs = np.swapaxes(np.concatenate((on_all_specs_dev[:, int(0.8 * on_all_specs_dev.shape[1]):],
                                               off_all_specs_dev[:, int(0.8 * off_all_specs_dev.shape[1]):]), axis=1), axis1=1, axis2=0)
        valFreqLabels = np.concatenate((freqLabels_dev[1, int(0.8 * on_all_specs_dev.shape[1]):],
                                        freqLabels_dev[0, int(0.8 * off_all_specs_dev.shape[1]):]), axis=0)[np.newaxis, :]
        valPhaseLabels = np.concatenate((phaseLabels_dev[1, int(0.8 * on_all_specs_dev.shape[1]):],
                                         phaseLabels_dev[0, int(0.8 * off_all_specs_dev.shape[1]):]), axis=0)[np.newaxis, :]

        testSpecs = np.swapaxes(np.concatenate((on_all_specs_test, off_all_specs_test), axis=1), axis1=1, axis2=0)
        testFreqLabels = np.concatenate((freqLabels_test[1, :], freqLabels_test[0, :]), axis=0)[np.newaxis, :]
        testPhaseLabels = np.concatenate((phaseLabels_test[1, :], phaseLabels_test[0, :]), axis=0)[np.newaxis, :]

        trainSpecsFC = np.swapaxes(np.concatenate((on_all_specs_devFC[:, :int(0.8 * on_all_specs_devFC.shape[1])],
                                                   off_all_specs_devFC[:, :int(0.8 * off_all_specs_devFC.shape[1])]), axis=1), axis1=1, axis2=0)
        valSpecsFC = np.swapaxes(np.concatenate((on_all_specs_devFC[:, int(0.8 * on_all_specs_devFC.shape[1]):],
                                                 off_all_specs_devFC[:, int(0.8 * off_all_specs_devFC.shape[1]):]), axis=1), axis1=1, axis2=0)
        testSpecsFC = np.swapaxes(np.concatenate((on_all_specs_testFC, off_all_specs_testFC), axis=1), axis1=1, axis2=0)

        # Manual Sorting of ON and OFF transients (ON=1 (Sim) and ON=1 (Vivo))
        index_shufTrain = list(range(trainSpecs.shape[0]))
        index_shufVal = list(range(valSpecs.shape[0]))
        shuffle(index_shufTrain), shuffle(index_shufVal)
        indOrderTrain, indOrderVal = 0, 0

        for indRandTrain in index_shufTrain:
            trainSpecs[indOrderTrain, :] = trainSpecs[indRandTrain, :]
            trainFreqLabels[:, indOrderTrain] = trainFreqLabels[:, indRandTrain]
            trainPhaseLabels[:, indOrderTrain] = trainPhaseLabels[:, indRandTrain]
            trainSpecsFC[indOrderTrain, :] = trainSpecsFC[indRandTrain, :]
            indOrderTrain = indOrderTrain + 1

        for indRandVal in index_shufVal:
            valSpecs[indOrderVal, :] = valSpecs[indRandVal, :]
            valFreqLabels[:, indOrderVal] = valFreqLabels[:, indRandVal]
            valPhaseLabels[:, indOrderVal] = valPhaseLabels[:, indRandVal]
            valSpecsFC[indOrderVal, :] = valSpecsFC[indRandVal, :]
            indOrderVal = indOrderVal + 1

        # Magnitude of Data and 2-channel Complex Data
        allSpecsTrain_2ChanComp = np.empty((2, trainSpecs.shape[0], trainSpecs.shape[1]))
        allSpecsTrain_2ChanComp[0, :, :], allSpecsTrain_2ChanComp[1, :, :] = trainSpecs.real, trainSpecs.imag
        allSpecsTrain_Mag = np.abs(trainSpecs)[np.newaxis, :, :]

        allSpecsVal_2ChanComp = np.empty((2, valSpecs.shape[0], valSpecs.shape[1]))
        allSpecsVal_2ChanComp[0, :, :], allSpecsVal_2ChanComp[1, :, :] = valSpecs.real, valSpecs.imag
        allSpecsVal_Mag = np.abs(valSpecs)[np.newaxis, :, :]

        allSpecsTest_2ChanComp = np.empty((2, testSpecs.shape[0], testSpecs.shape[1]))
        allSpecsTest_2ChanComp[0, :, :], allSpecsTest_2ChanComp[1, :, :] = testSpecs.real, testSpecs.imag
        allSpecsTest_Mag = np.abs(testSpecs)[np.newaxis, :, :]

        # Window 1024 Points Starting at 0 ppm
        simStart, simFinish = np.where(ppm <= 0.00)[0][-1], np.where(ppm >= 7.83)[0][0] - 1

        # Convert to tensors and transfer operations to GPU
        allSpecsTrain_2ChanCompTensor = torch.from_numpy(allSpecsTrain_2ChanComp[:, :, simStart:simFinish]).float()
        allSpecsTrain_MagTensor = torch.from_numpy(allSpecsTrain_Mag[:, :, simStart:simFinish]).float()
        trainFreqLabelsTensor = torch.from_numpy(trainFreqLabels).float()
        trainPhaseLabelsTensor = torch.from_numpy(trainPhaseLabels).float()

        allSpecsVal_2ChanCompTensor = torch.from_numpy(allSpecsVal_2ChanComp[:, :, simStart:simFinish]).float()
        allSpecsVal_MagTensor = torch.from_numpy(allSpecsVal_Mag[:, :, simStart:simFinish]).float()
        valFreqLabelsTensor = torch.from_numpy(valFreqLabels).float()
        valPhaseLabelsTensor = torch.from_numpy(valPhaseLabels).float()

        allSpecsTest_2ChanCompTensor = torch.from_numpy(allSpecsTest_2ChanComp[:, :, simStart:simFinish]).float()
        allSpecsTest_MagTensor = torch.from_numpy(allSpecsTest_Mag[:, :, simStart:simFinish]).float()
        testFreqLabelsTensor = torch.from_numpy(testFreqLabels).float()
        testPhaseLabelsTensor = torch.from_numpy(testPhaseLabels).float()

        ########################################################################################################################
        # Set-up Datasets, Dataloaders and Transforms
        ########################################################################################################################
        # set-up arrays for frequency corrected specs after frequency network before phase network
        freqCorr_specsTrain = np.zeros(shape=(2, trainSpecsFC.shape[0], 2048))
        freqCorr_specsVal = np.zeros(shape=(2, valSpecsFC.shape[0], 2048))
        freqCorr_specsTest = np.zeros(shape=(2, testSpecsFC.shape[0], 2048))

        for model_name in modelTypes:
            for net in netTypes:  # nets: frequency and phase
                run_name = f'{net}_Sim{snr}_{dataTypes[indW]}Water_{model_name}_FINAL'
                print(run_name)
                predTrainLabels, predValLabels = 0, 0

                # select based on network type
                if net == "freq":
                    if (model_name == "Ma_4Convs") or (model_name == "realReal"):
                        train_dataset = FPC_Dataset_1C(allSpecsTrain_MagTensor,trainFreqLabelsTensor)
                        val_dataset = FPC_Dataset_1C(allSpecsVal_MagTensor, valFreqLabelsTensor)
                        test_dataset = FPC_Dataset_1C(allSpecsTest_MagTensor, testFreqLabelsTensor)
                    elif (model_name == "Tapper"):
                        train_dataset = FPC_Dataset_Tapper(allSpecsTrain_MagTensor, trainFreqLabelsTensor)
                        val_dataset = FPC_Dataset_Tapper(allSpecsVal_MagTensor, valFreqLabelsTensor)
                        test_dataset = FPC_Dataset_Tapper(allSpecsTest_MagTensor, testFreqLabelsTensor)
                    elif (model_name == "compReal") or ("compComp"):
                        train_dataset = FPC_Dataset_2C(allSpecsTrain_2ChanCompTensor, trainFreqLabelsTensor)
                        val_dataset = FPC_Dataset_2C(allSpecsVal_2ChanCompTensor, valFreqLabelsTensor)
                        test_dataset = FPC_Dataset_2C(allSpecsTest_2ChanCompTensor, testFreqLabelsTensor)
                    else:
                        print("Frequency Model Not Found!")
                        break

                elif net == "phase":
                    freqCorr_specsTrain[0, :, :], freqCorr_specsTrain[1, :, :] = trainSpecsFC.real, trainSpecsFC.imag
                    freqCorr_specsTrain_real = (freqCorr_specsTrain[0, :, :])[np.newaxis, :, :]

                    freqCorr_specsVal[0, :, :], freqCorr_specsVal[1, :, :] = valSpecsFC.real, valSpecsFC.imag
                    freqCorr_specsVal_real = (freqCorr_specsVal[0, :, :])[np.newaxis, :, :]

                    freqCorr_specsTest[0, :, :], freqCorr_specsTest[1, :, :] = testSpecsFC.real, testSpecsFC.imag
                    freqCorr_specsTest_real = (freqCorr_specsTest[0, :, :])[np.newaxis, :, :]

                    # create tensor from data
                    freqCorrTrain_Tensor = torch.from_numpy(freqCorr_specsTrain[:, :, simStart:simFinish]).float()
                    freqCorrTrain_RealTensor = torch.from_numpy(freqCorr_specsTrain_real[:, :, simStart:simFinish]).float()

                    freqCorrVal_Tensor = torch.from_numpy(freqCorr_specsVal[:, :, simStart:simFinish]).float()
                    freqCorrVal_RealTensor = torch.from_numpy(freqCorr_specsVal_real[:, :, simStart:simFinish]).float()

                    freqCorr_specs_test_tensor = torch.from_numpy(freqCorr_specsTest[:, :, simStart:simFinish]).float()
                    freqCorr_specs_test_real_tensor = torch.from_numpy(freqCorr_specsTest_real[:, :, simStart:simFinish]).float()

                    # load data into proper dataloader based on model
                    if (model_name == "Ma_4Convs") or (model_name == "realReal"):
                        train_dataset = FPC_Dataset_1C(freqCorrTrain_RealTensor, trainPhaseLabelsTensor)
                        val_dataset = FPC_Dataset_1C(freqCorrVal_RealTensor, valPhaseLabelsTensor)
                        test_dataset = FPC_Dataset_1C(freqCorr_specs_test_real_tensor, testPhaseLabelsTensor)
                    elif (model_name == "Tapper"):
                        train_dataset = FPC_Dataset_Tapper(freqCorrTrain_RealTensor, trainPhaseLabelsTensor)
                        val_dataset = FPC_Dataset_Tapper(freqCorrVal_RealTensor, valPhaseLabelsTensor)
                        test_dataset = FPC_Dataset_Tapper(freqCorr_specs_test_real_tensor, testPhaseLabelsTensor)
                    elif (model_name == "compReal") or ("compComp"):
                        train_dataset = FPC_Dataset_2C(freqCorrTrain_Tensor, trainPhaseLabelsTensor)
                        val_dataset = FPC_Dataset_2C(freqCorrVal_Tensor, valPhaseLabelsTensor)
                        test_dataset = FPC_Dataset_2C(freqCorr_specs_test_tensor, testPhaseLabelsTensor)
                    else:
                        print("Phase Model Not Found!")
                        break

                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
                val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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
                elif model_name == "compComp":
                    model = compIn_compConv().float()
                elif model_name == "realReal":
                    model = realIn_realConv().float()
                else:
                    print("Model Architecture Not Found!")
                    break

                # load model to GPU and set-up optimizer and scheduler
                model.to(device)
                # print(f'model number of parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
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
                for epoch in range(nmb_epochs):
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
                        current_loss = sum(epoch_lossesV) / len(epoch_lossesV)
                        if (((epoch + 1) > 25) and (
                                (prev_loss - current_loss) > -0.00000001) and (
                                (prev_loss - current_loss) < 0.00000001)):
                            count = count + 1
                            if count == 5:
                                early_stop = True
                                print(
                                    'Early Stop Criteria Reached!')  # validation loss hasn't improved in 5 consecutive epochs
                                break
                        else:
                            count = 0

                    # if (epoch ==0) or (epoch%50==0):
                    print(f'Epoch {epoch + 1}/{nmb_epochs}, Training loss: {sum(epoch_lossesT) / len(epoch_lossesT)} and Validation loss: {sum(epoch_lossesV) / len(epoch_lossesV)}')
                    print(f'Validation Loss Improvement of {prev_loss - current_loss}')

                    allTrainLosses.append(sum(epoch_lossesT) / len(epoch_lossesT))
                    allValLosses.append(sum(epoch_lossesV) / len(epoch_lossesV))
                    prev_loss = current_loss

                print(f'Training Complete for {run_name}, Early Stopping {early_stop} at epoch {epoch + 1}')
                print(f'Best Validation Loss was {best_val_loss}')
                print()

                torch.save(best_weights, f"{run_name}_SimForVivo.pt")

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

                        # calculating loss (only based on artificially added offsets) and saving predictions
                        test_loss = loss_fn(predTest, TestLabels)
                        epoch_lossesTest.append(test_loss.item())
                        predTestLabels[loop:loop + batch_size] = predTest.cpu().flatten()
                        loop = loop + batch_size

                        if test_loss > highestLoss:
                            highestLoss = test_loss
                        if test_loss < lowestLoss:
                            lowestLoss = test_loss

                    # print(f'Testing Complete for {run_name_test}')
                    print(f'Testing Complete for {run_name}')
                    print(f'Lowest test_loss {lowestLoss}, highest loss {highestLoss} and mean loss {sum(epoch_lossesTest) / len(epoch_lossesTest)}')
                    print()

                # save labels
                np.save(f"PredLabels_{run_name}.npy", predTestLabels)
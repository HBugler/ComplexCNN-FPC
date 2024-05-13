# FPC Winter 2024
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader

from FPC_Functions import normSpecs, divideDev, shuffleData, getMag, getReal, getComp, window1024, toFids, toSpecs, corrFShift
from Datasets import FPC_Dataset_MLP, FPC_Dataset_1C, FPC_Dataset_2C
from modelsFPC import CNN_Model, MLP_Model, compIn_compConv, compIn_realConv, realIn_realConv

########################################################################################################################
# SIMULATED TRAINING/TESTING USER ARGUMENTS
########################################################################################################################
simParser = argparse.ArgumentParser()
simParser.add_argument("nmb_epochs", type=int, default=200, nargs='?', help="Number of epochs for training")
simParser.add_argument("batch_size", type=int, default=64, nargs='?', help="Batch size")
simParser.add_argument("learn_r", type=float, default=0.001, nargs='?', help="Learning rate")
simParser.add_argument("lr_scheduler_freq", type=int, default=25, nargs='?', help="Number of epochs until next learning rate update")
simParser.add_argument("-addSimModel", '--simModelTypes', action='append', default=['MLP', 'CNN', 'compComp', 'compReal', 'realReal'], nargs='*', help='Model types: MLP, CNN, compComp, compReal, and realReal')
simParser.add_argument("-addWater", '--waterTypes', action='append', default=['Pos', 'Mix', 'None'], nargs='*', help='Water types: Pos, Mix, None')
simParser.add_argument("-addSNR", '--snrTypes', action='append', default=['10', '5', '2_5'], nargs='*', help='SNR types: 10, 5, 2_5')
simParser.add_argument("-addNet", '--netTypes', action='append', default=['freq', 'phase'], nargs='*', help='Net types: freq, phase')
simParser = simParser.parse_args()

########################################################################################################################
# VARIABLE SET-UP
########################################################################################################################
simDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/Data/Simulated/"
ppm = np.load(f"{simDir}GTs/ppm_Sim.npy")
t = np.load(f"{simDir}GTs/time_Sim.npy")

# hyperparameters
loss_fn = nn.L1Loss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predTrainLabels, predValLabels = 0, 0

#########################################################################################################################
# DATA PROCESSING
#########################################################################################################################
for water in simParser.waterTypes:
    indW = simParser.waterTypes.index(water)
    for snr in simParser.snrTypes:
        indS = simParser.snrTypes.index(snr)
        # load data
        onDev = np.load(f"{simDir}Corrupt/simSpecsOn{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Dev.npy")
        offDev = np.load(f"{simDir}Corrupt/simSpecsOff{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Dev.npy")
        onDevFc = np.load(f"{simDir}Corrupt/simSpecsOnFc{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Dev.npy")
        offDevFc = np.load(f"{simDir}Corrupt/simSpecsOffFc{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Dev.npy")
        freqDev = np.load(f"{simDir}Corrupt/simFNoise{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Dev.npy")
        phaseDev = np.load(f"{simDir}Corrupt/simPNoise{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Dev.npy")

        onTest = np.load(f"{simDir}Corrupt/simSpecsOn{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Test.npy")
        offTest = np.load(f"{simDir}Corrupt/simSpecsOff{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Test.npy")
        onTestFc = np.load(f"{simDir}Corrupt/simSpecsOnFc{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Test.npy")
        offTestFc = np.load(f"{simDir}Corrupt/simSpecsOffFc{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Test.npy")
        freqTest = np.load(f"{simDir}Corrupt/simFNoise{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Test.npy")
        phaseTest = np.load(f"{simDir}Corrupt/simPNoise{simParser.waterTypes[indW]}{simParser.snrTypes[indS]}Test.npy")

        # Normalization
        onSpecsDev, offSpecsDev = normSpecs(onDev), normSpecs(offDev)
        onSpecsDevFc, offSpecsDevFc = normSpecs(onDevFc), normSpecs(offDevFc)
        onSpecsTest, offSpecsTest = normSpecs(onTest), normSpecs(offTest)
        onSpecsTestFc, offSpecsTestFc = normSpecs(onTestFc), normSpecs(offTestFc)

        # Separate development set into train/val
        trainSpecs, valSpecs = divideDev(onSpecsDev, offSpecsDev, percent=0.8)
        trainSpecsFC, valSpecsFC = divideDev(onSpecsDevFc, offSpecsDevFc, percent=0.8)
        trainFreqLabels, valFreqLabels = divideDev(freqDev[1, :], freqDev[0, :], percent=0.8)
        trainPhaseLabels, valPhaseLabels = divideDev(phaseDev[1, :], phaseDev[0, :], percent=0.8)

        testSpecs = np.concatenate((onSpecsTest, offSpecsTest), axis=0)
        testSpecsFC = np.concatenate((onSpecsTestFc, offSpecsTestFc), axis=0)
        testFreqLabels = np.concatenate((freqTest[1, :], freqTest[0, :]), axis=0)[np.newaxis, :]
        testPhaseLabels = np.concatenate((phaseTest[1, :], phaseTest[0, :]), axis=0)[np.newaxis, :]

        # pre-shuffle
        trainSpecs, trainSpecsFC, trainFreqLabels, trainPhaseLabels = shuffleData(trainSpecs, trainSpecsFC, trainFreqLabels, trainPhaseLabels)
        valSpecs, valSpecsFC, valFreqLabels, valPhaseLabels = shuffleData(valSpecs, valSpecsFC, valFreqLabels, valPhaseLabels)

        # get magnitude (freq models) or 1-channel (phase models) or 2-channel data (complex freq and phase models)
        allSpecsTrain_2ChanComp, allSpecsTrain_Mag = getComp(trainSpecs), getMag(trainSpecs)
        allSpecsVal_2ChanComp, allSpecsVal_Mag = getComp(valSpecs), getMag(valSpecs)
        allSpecsTest_2ChanComp, allSpecsTest_Mag = getComp(testSpecs), getMag(testSpecs)

        freqCorr_specsTrain, freqCorr_specsTrain_real = getComp(trainSpecsFC), getReal(trainSpecsFC)
        freqCorr_specsVal, freqCorr_specsVal_real = getComp(valSpecsFC), getReal(valSpecsFC)
        freqCorr_specsTest, freqCorr_specsTest_real = getComp(testSpecsFC), getReal(testSpecsFC)

        # select 1024 window, convert to tensors and transfer operations to GPU
        allSpecsTrain_2ChanCompTensor = torch.from_numpy(window1024(allSpecsTrain_2ChanComp, ppm)).float()
        allSpecsTrain_MagTensor = torch.from_numpy(window1024(allSpecsTrain_Mag, ppm)).float()
        freqCorrTrain_Tensor = torch.from_numpy(window1024(freqCorr_specsTrain, ppm)).float()
        freqCorrTrain_RealTensor = torch.from_numpy(window1024(freqCorr_specsTrain_real, ppm)).float()
        trainFreqLabelsTensor = torch.from_numpy(trainFreqLabels).float()
        trainPhaseLabelsTensor = torch.from_numpy(trainPhaseLabels).float()

        allSpecsVal_2ChanCompTensor = torch.from_numpy(window1024(allSpecsVal_2ChanComp, ppm)).float()
        allSpecsVal_MagTensor = torch.from_numpy(window1024(allSpecsVal_Mag, ppm)).float()
        freqCorrVal_Tensor = torch.from_numpy(window1024(freqCorr_specsVal, ppm)).float()
        freqCorrVal_RealTensor = torch.from_numpy(window1024(freqCorr_specsVal_real, ppm)).float()
        valFreqLabelsTensor = torch.from_numpy(valFreqLabels).float()
        valPhaseLabelsTensor = torch.from_numpy(valPhaseLabels).float()

        allSpecsTest_2ChanCompTensor = torch.from_numpy(window1024(allSpecsTest_2ChanComp, ppm)).float()
        allSpecsTest_MagTensor = torch.from_numpy(window1024(allSpecsTest_Mag, ppm)).float()
        freqCorr_specs_test_tensor = torch.from_numpy(window1024(freqCorr_specsTest, ppm)).float()
        freqCorr_specs_test_real_tensor = torch.from_numpy(window1024(freqCorr_specsTest_real, ppm)).float()
        testFreqLabelsTensor = torch.from_numpy(testFreqLabels).float()
        testPhaseLabelsTensor = torch.from_numpy(testPhaseLabels).float()

########################################################################################################################
# DATASETS AND MODEL SET-UP
########################################################################################################################
        for model_name in simParser.simModelTypes:
            for net in simParser.netTypes:  # nets: frequency and phase
                run_name = f'Sim{snr}_{simParser.waterTypes[indW]}_{model_name}_{net}Model'
                print(run_name)

                # select based on network type and assign data to dataloader
                if net == "freq":
                    if (model_name == "CNN") or (model_name == "realReal"):
                        train_dataset = FPC_Dataset_1C(allSpecsTrain_MagTensor, trainFreqLabelsTensor)
                        val_dataset = FPC_Dataset_1C(allSpecsVal_MagTensor, valFreqLabelsTensor)
                        test_dataset = FPC_Dataset_1C(allSpecsTest_MagTensor, testFreqLabelsTensor)
                    elif (model_name == "MLP"):
                        train_dataset = FPC_Dataset_MLP(allSpecsTrain_MagTensor, trainFreqLabelsTensor)
                        val_dataset = FPC_Dataset_MLP(allSpecsVal_MagTensor, valFreqLabelsTensor)
                        test_dataset = FPC_Dataset_MLP(allSpecsTest_MagTensor, testFreqLabelsTensor)
                    elif (model_name == "compReal") or ("compComp"):
                        train_dataset = FPC_Dataset_2C(allSpecsTrain_2ChanCompTensor, trainFreqLabelsTensor)
                        val_dataset = FPC_Dataset_2C(allSpecsVal_2ChanCompTensor, valFreqLabelsTensor)
                        test_dataset = FPC_Dataset_2C(allSpecsTest_2ChanCompTensor, testFreqLabelsTensor)
                    else:
                        print("Frequency Model Not Found!")
                        break

                elif net == "phase":
                    if (model_name == "CNN") or (model_name == "realReal"):
                        train_dataset = FPC_Dataset_1C(freqCorrTrain_RealTensor, trainPhaseLabelsTensor)
                        val_dataset = FPC_Dataset_1C(freqCorrVal_RealTensor, valPhaseLabelsTensor)
                        test_dataset = FPC_Dataset_1C(freqCorr_specs_test_real_tensor, testPhaseLabelsTensor)
                    elif (model_name == "MLP"):
                        train_dataset = FPC_Dataset_MLP(freqCorrTrain_RealTensor, trainPhaseLabelsTensor)
                        val_dataset = FPC_Dataset_MLP(freqCorrVal_RealTensor, valPhaseLabelsTensor)
                        test_dataset = FPC_Dataset_MLP(freqCorr_specs_test_real_tensor, testPhaseLabelsTensor)
                    elif (model_name == "compReal") or ("compComp"):
                        train_dataset = FPC_Dataset_2C(freqCorrTrain_Tensor, trainPhaseLabelsTensor)
                        val_dataset = FPC_Dataset_2C(freqCorrVal_Tensor, valPhaseLabelsTensor)
                        test_dataset = FPC_Dataset_2C(freqCorr_specs_test_tensor, testPhaseLabelsTensor)
                    else:
                        print("Phase Model Not Found!")
                        break

                train_loader = DataLoader(dataset=train_dataset, batch_size=simParser.batch_size, shuffle=False)
                val_loader = DataLoader(dataset=val_dataset, batch_size=simParser.batch_size, shuffle=False)
                test_loader = DataLoader(dataset=test_dataset, batch_size=simParser.batch_size, shuffle=False)

                # load correct model
                if model_name == "CNN":
                    model = CNN_Model().float()
                elif model_name == "MLP":
                    model = MLP_Model().float()
                elif model_name == "compReal":
                    model = compIn_realConv().float()
                elif model_name == "compComp":
                    model = compIn_compConv().float()
                elif model_name == "realReal":
                    model = realIn_realConv().float()
                else:
                    print("Model Architecture Not Found!")
                    break

########################################################################################################################
# TRAINING LOOP
########################################################################################################################
                # initialize training loop parameters
                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=simParser.learn_r)
                lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda lr: 0.5)
                best_val_loss = float('inf')
                current_loss, prev_loss = 0.0, 0.0
                allTrainLosses, allValLosses = [], []

                # place model in train mode and move through epochs
                model.train()
                for epoch in range(simParser.nmb_epochs):
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

                    if (epoch + 1) % simParser.lr_scheduler_freq == 0:
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

                    if (epoch ==0) or (epoch%10==0):
                        print(f'Epoch {epoch + 1}/{simParser.nmb_epochs})')
                        print(f'Training loss: {sum(epoch_lossesT) / len(epoch_lossesT)} and Validation loss: {sum(epoch_lossesV) / len(epoch_lossesV)}')
                        print(f'Validation Loss Improvement of {prev_loss - current_loss}')

                    allTrainLosses.append(sum(epoch_lossesT) / len(epoch_lossesT))
                    allValLosses.append(sum(epoch_lossesV) / len(epoch_lossesV))
                    prev_loss = current_loss

                print(f'Training Complete for {run_name}')
                print(f'Best Validation Loss was {best_val_loss}')
                print()

                torch.save(best_weights, f"Sim{snr}_{simParser.waterTypes[indW]}_{model_name}_{net}Model.pt")

#########################################################################################################################
# TESTING LOOP
#########################################################################################################################
                # Testing Loop parameter set-up
                epoch_lossesTest, predTestLabels = [], np.zeros((testPhaseLabels.shape[1]))
                lowestLoss, highestLoss, ind = 100000, 0, 0

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
                        predLabelsTest = predTest.cpu().flatten()

                        if predLabelsTest.shape[0]==64:
                            predTestLabels[ind:ind + simParser.batch_size] = predLabelsTest
                        else:   #because test specs is not multiple of 64
                            predTestLabels[ind:ind + 56] = predLabelsTest
                        ind = ind + simParser.batch_size

                        if test_loss > highestLoss:
                            highestLoss = test_loss
                        if test_loss < lowestLoss:
                            lowestLoss = test_loss

                    print(f'Testing Complete for {run_name}')
                    print(f'Lowest test_loss {lowestLoss}, highest loss {highestLoss} and mean loss {sum(epoch_lossesTest) / len(epoch_lossesTest)}')
                    print()

                # save labels
                np.save(f"PredLabels_{run_name}.npy", predTestLabels)

########################################################################################################################
########################################################################################################################
# IN VIVO TESTING USER ARGUMENTS
########################################################################################################################
########################################################################################################################
vivoParser = argparse.ArgumentParser()
vivoParser.add_argument("batch_size", type=int, default=64, nargs='?', help="Batch size")
vivoParser.add_argument("-addVivoModel", '--vivoModelTypes', action='append', default=['compComp', 'MLP', 'CNN', 'compReal'], nargs='*', help='Model types: MLP, CNN, compComp')
vivoParser.add_argument("-addOffsetSize", '--offsetSize', action='append', default=['None', 'Small', 'Med', 'Large'], nargs='*', help='Offset sizes: None, Small, Med, Large')
vivoParser.add_argument("-addNet", '--netTypes', action='append', default=['freq', 'phase'], nargs='*', help='Net types: freq, phase')
vivoParser = vivoParser.parse_args()

t_inVivo = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/Data/InVivo/GTs/time_InVivo.npy")
ppmVivo = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/Data/InVivo/GTs/ppm_InVivo.npy")

# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.L1Loss()

########################################################################################################################
# MODEL SELECTION
########################################################################################################################
# load correct model
for model_name in vivoParser.vivoModelTypes:
    for net in vivoParser.netTypes:
        if model_name == "CNN":
            model = CNN_Model().float()
        elif model_name == "MLP":
            model = MLP_Model().float()
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
        for size in vivoParser.offsetSize:
            indS = vivoParser.offsetSize.index(size)
            modelLoadName = f"Sim2_5_Pos_{model_name}_{net}Model"
            run_name_test = f'{size}_{net}_InVivo_{model_name}'
            print(run_name_test)

            # load and norm data
            dataDir = "C:/Users/Hanna B/Desktop/FPCFinal2024/Data/"
            onTest = np.load(f"{dataDir}InVivo/Corrupt/vivoSpecsOn{vivoParser.offsetSize[indS]}Offsets.npy")
            offTest = np.load(f"{dataDir}InVivo/Corrupt/vivoSpecsOff{vivoParser.offsetSize[indS]}Offsets.npy")
            freqTest = np.load(f"{dataDir}InVivo/Corrupt/vivoFNoise{vivoParser.offsetSize[indS]}Offsets.npy")
            phaseTest = np.load(f"{dataDir}InVivo/Corrupt/vivoPNoise{vivoParser.offsetSize[indS]}Offsets.npy")

            onSpecsTest, offSpecsTest = normSpecs(onTest), normSpecs(offTest)

            # Concatenate ON and OFF specs
            testSpecs = np.concatenate((onSpecsTest, offSpecsTest), axis=0)
            testFreqLabels = np.concatenate((freqTest[1, :], freqTest[0, :]))[np.newaxis, :]
            testPhaseLabels = np.concatenate((phaseTest[1, :], phaseTest[0, :]))[np.newaxis, :]

            # Magnitude of Data and 2-channel Complex Data
            allSpecsTest_2ChanComp, allSpecsTest_Mag = getComp(testSpecs), getMag(testSpecs)

            # select 1024 window, convert to tensors and transfer operations to GPU (may need to create deep copy)
            allSpecsTest_2ChanCompTensor = torch.from_numpy(window1024(allSpecsTest_2ChanComp, ppmVivo)).float()
            allSpecsTest_MagTensor = torch.from_numpy(window1024(allSpecsTest_Mag, ppmVivo)).float()
            testFreqLabelsTensor = torch.from_numpy(testFreqLabels).float()
            testPhaseLabelsTensor = torch.from_numpy(testPhaseLabels).float()

#########################################################################################################################
# DATASETS SET-UP
#########################################################################################################################
            # select network for testing
            if net == "freq":
                if model_name == "CNN":
                    test_dataset = FPC_Dataset_1C(allSpecsTest_MagTensor, testFreqLabelsTensor)
                elif model_name == "MLP":
                    test_dataset = FPC_Dataset_MLP(allSpecsTest_MagTensor, testFreqLabelsTensor)
                elif (model_name == "compReal") or (model_name == "compComp"):
                    test_dataset = FPC_Dataset_2C(allSpecsTest_2ChanCompTensor, testFreqLabelsTensor)
                else:
                    print(f'Frequency Test Dataset not found!')

            elif net == "phase":
                # apply frequency correction
                predFLabels = np.load(f"PredLabels_{size}_freq_InVivo_{model_name}_{vivoParser.offsetSize[indS]}.npy")
                TestFids = toFids(testSpecs, 1)
                TestFids = corrFShift(TestFids, t_inVivo, predFLabels)
                testSpecsFC = toSpecs(TestFids, 1)

                fig1, ax1 = plt.subplots(1)
                for iii in range (0, 20):
                    ax1.plot(ppmVivo, testSpecsFC[iii, :])
                plt.show()

                # get 1 & 2 channel data and pass to GPU
                freqCorr_specsTest, freqCorr_specsTest_real = getComp(testSpecsFC), getReal(testSpecsFC)
                freqCorr_specs_test_tensor = torch.from_numpy(window1024(freqCorr_specsTest, ppmVivo)).float()
                freqCorr_specs_test_real_tensor = torch.from_numpy(window1024(freqCorr_specsTest_real, ppmVivo)).float()

                # select model and give data to dataloader
                if model_name == "CNN":
                    test_dataset = FPC_Dataset_1C(freqCorr_specs_test_real_tensor,  testPhaseLabelsTensor)
                elif model_name == "MLP":
                    test_dataset = FPC_Dataset_MLP(freqCorr_specs_test_real_tensor, testPhaseLabelsTensor)
                elif (model_name == "compReal") or (model_name == "compComp"):
                    test_dataset = FPC_Dataset_2C(freqCorr_specs_test_tensor, testPhaseLabelsTensor)
                else:  # assumed to be CR-CNN
                    print(f'Phase Test Dataset not found!')

            test_loader = DataLoader(dataset=test_dataset, batch_size=vivoParser.batch_size, shuffle=False)

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
                    predTestLabels[indV:indV + vivoParser.batch_size] = predTest.cpu().flatten()
                    indV = indV + vivoParser.batch_size

                    if test_loss > highestLoss:
                        highestLoss = test_loss
                    if test_loss < lowestLoss:
                        lowestLoss = test_loss

                print(f'Testing Complete for {run_name_test}')
                print(f'Lowest test_loss {lowestLoss}, highest loss {highestLoss} and mean loss {sum(epoch_lossesTest) / len(epoch_lossesTest)}')
                print()

            # save labels
            np.save(f"PredLabels_{run_name_test}_{vivoParser.offsetSize[indS]}.npy", predTestLabels)
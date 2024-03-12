# FPC Winter 2024
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transient_maker import TransientMaker
from Datasets import FPC_Dataset, FPC_Dataset_Tapper
from AblationStudyModels import realIn_realConv, compIn_realConv, compIn_compConv
from TapperModel import tapperModel
from MaModel import maModel, maModel2Convs

#################################
# prep in vivo data
#################################




########################################################################################################################
# Clean and Split Data
########################################################################################################################
# set-up hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# inputTypes, convTypes = ["comp", "real"], ["comp", "real"]  # order of inputs is reversed since we are not doing realIN_compCONV
modelTypes = ["compReal"]   # currently missing ["Ma_2Convs", "realReal", "Tapper", "compComp", "Ma_4Convs", "compReal"]
netTypes = ["phase"]    #["freq", "phase"]
snrTypes = ["2_5"]      #["10", "5", "2_5"]
dataName = "Sim"
resid = ["Mix"]  # ["Mix", "Neg", "None", "Pos"]
set = ["Dev", "Test"]

batch_size, learn_r = 16, 0.0001
nmb_epochs = 100 # 200
loss_fn = nn.L1Loss()
lr_scheduler_freq = 25

ppm = np.loadtxt("C:/Users/Hanna B/PycharmProjects/FPC_2024/Development/ppm_2048.csv", delimiter=",")[:, 0].flatten()
ppm = np.ndarray.round(ppm, 2)
ind_close, ind_far = np.amax(np.where(ppm == 0.00)), np.amin(np.where(ppm == 7.83))


for water in resid:
    indW = resid.index(water)

    for snr in snrTypes:
        ind = snrTypes.index(snr)
        print(f'SNR {snrTypes[ind]} and water resid {resid[indW]}')
        print(f'example file location:... /Sim{snrTypes[ind]}_{resid[indW]}/ON_FreqCorrectedSpecs_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[0]}')

        on_freqCorr_specs_dev = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/ON_FreqCorrectedSpecs_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[0]}.npy")
        off_freqCorr_specs_dev = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/OFF_FreqCorrectedSpecs_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[0]}.npy")
        on_all_specs_dev = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/ON_AllSpecs_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[0]}.npy")
        off_all_specs_dev = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/OFF_AllSpecs_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[0]}.npy")

        on_freqCorr_specs_test = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/ON_FreqCorrectedSpecs_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[1]}.npy")
        off_freqCorr_specs_test = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/OFF_FreqCorrectedSpecs_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[1]}.npy")
        on_all_specs_test = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/ON_AllSpecs_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[1]}.npy")
        off_all_specs_test = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/OFF_AllSpecs_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[1]}.npy")

        phaseLabels_dev = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/TruePhaseLabels_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[0]}.npy")
        phaseLabels_test = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/TruePhaseLabels_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[1]}.npy")
        freqLabels_dev = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/TrueFreqLabels_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[0]}.npy")
        freqLabels_test = np.load(f"C:/Users/Hanna B/PycharmProjects/FPC_2024/Sim{snrTypes[ind]}_{resid[indW]}/TrueFreqLabels_{dataName}{snrTypes[ind]}_{resid[indW]}_{set[1]}.npy")

        print(phaseLabels_dev.shape)
        # Select window of points
        on_freqCorr_specs_dev = on_freqCorr_specs_dev[:, ind_far:ind_close, :]
        off_freqCorr_specs_dev = off_freqCorr_specs_dev[:, ind_far:ind_close, :]
        on_all_specs_dev = on_all_specs_dev[:, ind_far:ind_close, :]
        off_all_specs_dev = off_all_specs_dev[:, ind_far:ind_close, :]  # shape is (subspectra (1), num of spectral points (1024), num transients(160*250))

        on_freqCorr_specs_test = on_freqCorr_specs_test[:, ind_far:ind_close, :]
        off_freqCorr_specs_test = off_freqCorr_specs_test[:, ind_far:ind_close, :]
        on_all_specs_test = on_all_specs_test[:, ind_far:ind_close, :]
        off_all_specs_test = off_all_specs_test[:, ind_far:ind_close, :]  # shape is (subspectra (1), num of spectral points (1024), num transients(160*250))

        # data split (80 training / 20 validation)
        on_freqCorr_specs_train = on_freqCorr_specs_dev[:, :, :int(on_freqCorr_specs_dev.shape[2] * 0.8)]
        on_freqCorr_specs_val = on_freqCorr_specs_dev[:, :, int(on_freqCorr_specs_dev.shape[2] * 0.8):]
        off_freqCorr_specs_train = off_freqCorr_specs_dev[:, :, :int(off_freqCorr_specs_dev.shape[2] * 0.8)]
        off_freqCorr_specs_val = off_freqCorr_specs_dev[:, :, int(off_freqCorr_specs_dev.shape[2] * 0.8):]

        on_all_specs_train = on_all_specs_dev[:, :, :int(on_all_specs_dev.shape[2] * 0.8)]
        on_all_specs_val = on_all_specs_dev[:, :, int(on_all_specs_dev.shape[2] * 0.8):]
        off_all_specs_train = off_all_specs_dev[:, :, :int(off_all_specs_dev.shape[2] * 0.8)]
        off_all_specs_val = off_all_specs_dev[:, :, int(off_all_specs_dev.shape[2] * 0.8):]

        on_phaseLabels_train = phaseLabels_dev[0, :, :int(phaseLabels_dev.shape[2] * 0.8)]
        on_phaseLabels_val = phaseLabels_dev[0, :, int(phaseLabels_dev.shape[2] * 0.8):]
        off_phaseLabels_train = phaseLabels_dev[1, :, :int(phaseLabels_dev.shape[2] * 0.8)]
        off_phaseLabels_val = phaseLabels_dev[1, :, int(phaseLabels_dev.shape[2] * 0.8):]
        on_freqLabels_train = freqLabels_dev[0, :, :int(freqLabels_dev.shape[2] * 0.8)]
        on_freqLabels_val = freqLabels_dev[0, :, int(freqLabels_dev.shape[2] * 0.8):]
        off_freqLabels_train = freqLabels_dev[1, :, :int(freqLabels_dev.shape[2] * 0.8)]
        off_freqLabels_val = freqLabels_dev[1, :, int(freqLabels_dev.shape[2] * 0.8):]

        on_phaseLabels_test = phaseLabels_test[0, :, :]
        off_phaseLabels_test = phaseLabels_test[1, :, :]
        on_freqLabels_test = freqLabels_test[0, :, :]
        off_freqLabels_test = freqLabels_test[1, :, :]

        # data normalization (currently normalized per scan) and reshaping
        both = (np.concatenate((on_all_specs_train, off_all_specs_train)))
        both_RM = np.mean(both.real)
        both_IM = np.mean(both.imag)
        both_RS = np.std(both.real)
        both_IS = np.std(both.imag)

        on_freqCorr_specs_train = ((on_freqCorr_specs_train.real - both_RM) / both_RS) + ((on_freqCorr_specs_train.imag - both_IM) / both_IS)*1j
        on_freqCorr_specs_val = ((on_freqCorr_specs_val.real - both_RM) / both_RS) + ((on_freqCorr_specs_val.imag - both_IM) / both_IS)*1j
        off_freqCorr_specs_train = ((off_freqCorr_specs_train.real - both_RM) / both_RS) + ((off_freqCorr_specs_train.imag - both_IM) / both_IS)*1j
        off_freqCorr_specs_val = ((off_freqCorr_specs_val.real - both_RM) / both_RS) + ((off_freqCorr_specs_val.imag - both_IM) / both_IS)*1j

        on_all_specs_train = ((on_all_specs_train.real - both_RM) / both_RS) + ((on_all_specs_train.imag - both_IM) / both_IS)*1j
        on_all_specs_val = ((on_all_specs_val.real - both_RM) / both_RS) + ((on_all_specs_val.imag - both_IM) / both_IS)*1j
        off_all_specs_train = ((off_all_specs_train.real - both_RM) / both_RS) + ((off_all_specs_train.imag - both_IM) / both_IS)*1j
        off_all_specs_val = ((off_all_specs_val.real - both_RM) / both_RS) + ((off_all_specs_val.imag - both_IM) / both_IS)*1j

        on_freqCorr_specs_test = ((on_freqCorr_specs_test.real - both_RM) / both_RS) + ((on_freqCorr_specs_test.imag - both_IM) / both_IS)*1j
        off_freqCorr_specs_test = ((off_freqCorr_specs_test.real - both_RM) / both_RS) + ((off_freqCorr_specs_test.imag - both_IM) / both_IS)*1j
        on_all_specs_test = ((on_all_specs_test.real - both_RM) / both_RS) + ((on_all_specs_test.imag - both_IM) / both_IS)*1j
        off_all_specs_test = ((off_all_specs_test.real - both_RM) / both_RS) + ((off_all_specs_test.imag - both_IM) / both_IS)*1j

        on_freqCorr_specs_train = np.einsum('kij->kji', on_freqCorr_specs_train)
        on_freqCorr_specs_val = np.einsum('kij->kji', on_freqCorr_specs_val)
        off_freqCorr_specs_train = np.einsum('kij->kji', off_freqCorr_specs_train)
        off_freqCorr_specs_val = np.einsum('kij->kji', off_freqCorr_specs_val)

        on_all_specs_train = np.einsum('kij->kji', on_all_specs_train)
        on_all_specs_val = np.einsum('kij->kji', on_all_specs_val)
        off_all_specs_train = np.einsum('kij->kji', off_all_specs_train)
        off_all_specs_val = np.einsum('kij->kji', off_all_specs_val)

        on_freqCorr_specs_test = np.einsum('kij->kji', on_freqCorr_specs_test)
        off_freqCorr_specs_test = np.einsum('kij->kji', off_freqCorr_specs_test)
        on_all_specs_test = np.einsum('kij->kji', on_all_specs_test)
        off_all_specs_test = np.einsum('kij->kji', off_all_specs_test)

        # Concatenate data and separate real and imaginary
        all_specs_train, all_specs_val = np.zeros(shape=(2, on_all_specs_train.shape[1] * 2, 1024)), np.zeros(shape=(2, on_all_specs_val.shape[1] * 2, 1024))
        freqCorr_specs_train, freqCorr_specs_val = np.zeros(shape=(2, on_freqCorr_specs_train.shape[1] * 2, 1024)), np.zeros(shape=(2, on_freqCorr_specs_val.shape[1] * 2, 1024))
        freqLabels_train, freqLabels_val = np.zeros(shape=(1, on_freqLabels_train.shape[1] * 2)), np.zeros(shape=(1, on_freqLabels_val.shape[1] * 2))
        phaseLabels_train, phaseLabels_val = np.zeros(shape=(1, on_phaseLabels_train.shape[1] * 2)), np.zeros(shape=(1, on_phaseLabels_val.shape[1] * 2))

        all_specs_train[0, :on_all_specs_train.shape[1], :] = np.squeeze(on_all_specs_train.real)
        all_specs_train[0, on_all_specs_train.shape[1]:, :] = np.squeeze(off_all_specs_train.real)
        all_specs_train[1, :on_all_specs_train.shape[1], :] = np.squeeze(on_all_specs_train.imag)
        all_specs_train[1, on_all_specs_train.shape[1]:, :] = np.squeeze(off_all_specs_train.imag)

        all_specs_val[0, :on_all_specs_val.shape[1], :] = np.squeeze(on_all_specs_val.real)
        all_specs_val[0, on_all_specs_val.shape[1]:, :] = np.squeeze(off_all_specs_val.real)
        all_specs_val[1, :on_all_specs_val.shape[1], :] = np.squeeze(on_all_specs_val.imag)
        all_specs_val[1, on_all_specs_val.shape[1]:, :] = np.squeeze(off_all_specs_val.imag)

        freqCorr_specs_train[0, :on_freqCorr_specs_train.shape[1], :] = np.squeeze(on_freqCorr_specs_train.real)
        freqCorr_specs_train[0, on_freqCorr_specs_train.shape[1]:, :] = np.squeeze(off_freqCorr_specs_train.real)
        freqCorr_specs_train[1, :on_freqCorr_specs_train.shape[1], :] = np.squeeze(on_freqCorr_specs_train.imag)
        freqCorr_specs_train[1, on_freqCorr_specs_train.shape[1]:, :] = np.squeeze(off_freqCorr_specs_train.imag)
        freqCorr_specs_val[0, :on_freqCorr_specs_val.shape[1], :] = np.squeeze(on_freqCorr_specs_val.real)
        freqCorr_specs_val[0, on_freqCorr_specs_val.shape[1]:, :] = np.squeeze(off_freqCorr_specs_val.real)
        freqCorr_specs_val[1, :on_freqCorr_specs_val.shape[1], :] = np.squeeze(on_freqCorr_specs_val.imag)
        freqCorr_specs_val[1, on_freqCorr_specs_val.shape[1]:, :] = np.squeeze(off_freqCorr_specs_val.imag)

        freqLabels_train[:, :on_freqLabels_train.shape[1]] = on_freqLabels_train
        freqLabels_train[:, on_freqLabels_train.shape[1]:] = off_freqLabels_train
        freqLabels_val[:, :on_freqLabels_val.shape[1]] = on_freqLabels_val
        freqLabels_val[:, on_freqLabels_val.shape[1]:] = off_freqLabels_val

        phaseLabels_train[:, :on_phaseLabels_train.shape[1]] = on_phaseLabels_train
        phaseLabels_train[:, on_phaseLabels_train.shape[1]:] = off_phaseLabels_train
        phaseLabels_val[:, :on_phaseLabels_val.shape[1]] = on_phaseLabels_val
        phaseLabels_val[:, on_phaseLabels_val.shape[1]:] = off_phaseLabels_val

        all_specs_test, freqCorr_specs_test = np.zeros(shape=(2, on_all_specs_test.shape[1] * 2, 1024)), np.zeros(shape=(2, on_freqCorr_specs_test.shape[1] * 2, 1024))
        freqLabels_test, phaseLabels_test = np.zeros(shape=(1, on_freqLabels_test.shape[1] * 2)), np.zeros(shape=(1, on_phaseLabels_test.shape[1] * 2))

        all_specs_test[0, :on_all_specs_test.shape[1], :] = np.squeeze(on_all_specs_test.real)
        all_specs_test[0, on_all_specs_test.shape[1]:, :] = np.squeeze(off_all_specs_test.real)
        all_specs_test[1, :on_all_specs_test.shape[1], :] = np.squeeze(on_all_specs_test.imag)
        all_specs_test[1, on_all_specs_test.shape[1]:, :] = np.squeeze(off_all_specs_test.imag)
        freqCorr_specs_test[0, :on_freqCorr_specs_test.shape[1], :] = np.squeeze(on_freqCorr_specs_test.real)
        freqCorr_specs_test[0, on_freqCorr_specs_test.shape[1]:, :] = np.squeeze(off_freqCorr_specs_test.real)
        freqCorr_specs_test[1, :on_freqCorr_specs_test.shape[1], :] = np.squeeze(on_freqCorr_specs_test.imag)
        freqCorr_specs_test[1, on_freqCorr_specs_test.shape[1]:, :] = np.squeeze(off_freqCorr_specs_test.imag)

        freqLabels_test[:, :on_freqLabels_test.shape[1]] = on_freqLabels_test
        freqLabels_test[:, on_freqLabels_test.shape[1]:] = off_freqLabels_test
        phaseLabels_test[:, :on_phaseLabels_test.shape[1]] = on_phaseLabels_test
        phaseLabels_test[:, on_phaseLabels_test.shape[1]:] = off_phaseLabels_test

        # Real Valued CNN
        # magnitude value calculation
        all_specs_train_real = np.sqrt((all_specs_train[0, :, :]) * (all_specs_train[0, :, :]) + (all_specs_train[1, :, :]) * (all_specs_train[1, :, :]))
        all_specs_train_real = all_specs_train_real[np.newaxis, :, :]
        all_specs_val_real = np.sqrt((all_specs_val[0, :, :]) * (all_specs_val[0, :, :]) + (all_specs_val[1, :, :]) * (all_specs_val[1, :, :]))
        all_specs_val_real = all_specs_val_real[np.newaxis, :, :]

        # real value (because phase network)
        freqCorr_specs_train_real = (freqCorr_specs_train[0, :, :])[np.newaxis, :, :]
        freqCorr_specs_val_real = (freqCorr_specs_val[0, :, :])[np.newaxis, :, :]

        all_specs_test_real = np.sqrt((all_specs_test[0, :, :]) * (all_specs_test[0, :, :]) + (all_specs_test[1, :, :]) * (all_specs_test[1, :, :]))
        all_specs_test_real = all_specs_test_real[np.newaxis, :, :]
        freqCorr_specs_test_real = (freqCorr_specs_test[0, :, :])[np.newaxis, :, :]

        # Convert to tensors and transfer operations to GPU
        all_specs_train_tensor = torch.from_numpy(all_specs_train).float()  # shape is (#channels, #samples, #spectralPoints)
        all_specs_val_tensor = torch.from_numpy(all_specs_val).float()
        all_specs_train_real_tensor = torch.from_numpy(all_specs_train_real).float()
        all_specs_val_real_tensor = torch.from_numpy(all_specs_val_real).float()
        freqCorr_specs_train_tensor = torch.from_numpy(freqCorr_specs_train).float()
        freqCorr_specs_val_tensor = torch.from_numpy(freqCorr_specs_val).float()
        freqCorr_specs_train_real_tensor = torch.from_numpy(freqCorr_specs_train_real).float()
        freqCorr_specs_val_real_tensor = torch.from_numpy(freqCorr_specs_val_real).float()

        freqLabels_train_tensor = torch.from_numpy(freqLabels_train).float()  # shape is (1, #samples)
        freqLabels_val_tensor = torch.from_numpy(freqLabels_val).float()
        phaseLabels_train_tensor = torch.from_numpy(phaseLabels_train).float()
        phaseLabels_val_tensor = torch.from_numpy(phaseLabels_val).float()

        all_specs_test_tensor = torch.from_numpy(all_specs_test).float()
        all_specs_test_real_tensor = torch.from_numpy(all_specs_test_real).float()
        freqCorr_specs_test_tensor = torch.from_numpy(freqCorr_specs_test).float()
        freqCorr_specs_test_real_tensor = torch.from_numpy(freqCorr_specs_test_real).float()
        freqLabels_test_tensor = torch.from_numpy(freqLabels_test).float()
        phaseLabels_test_tensor = torch.from_numpy(phaseLabels_test).float()

        # ########################################################################################################################
        # # Set-up Datasets, Dataloaders and Transforms
        # ########################################################################################################################
        for model_name in modelTypes:
            for net in netTypes:
                run_name = f'{net}_{dataName}{snr}_{resid[indW]}Water_{model_name}'
                print(run_name)

                # select based on network type
                if net == "freq":
                    if model_name == "Ma_4Convs" or model_name =="Ma_2Convs" or  model_name =="realReal":
                        train_dataset = FPC_Dataset(all_specs_train_real_tensor, freqLabels_train_tensor)
                        val_dataset = FPC_Dataset(all_specs_val_real_tensor, freqLabels_val_tensor)
                        test_dataset = FPC_Dataset(all_specs_test_real_tensor, freqLabels_test_tensor)
                        TestSpecsSaved = np.zeros((all_specs_test.shape[1], 1, all_specs_test.shape[2]))  # our models
                    elif model_name=="Tapper":
                        train_dataset = FPC_Dataset_Tapper(all_specs_train_real_tensor, freqLabels_train_tensor)
                        val_dataset = FPC_Dataset_Tapper(all_specs_val_real_tensor, freqLabels_val_tensor)
                        test_dataset = FPC_Dataset_Tapper(all_specs_test_real_tensor, freqLabels_test_tensor)
                        TestSpecsSaved = np.zeros((all_specs_test.shape[1], all_specs_test.shape[2]))  # tapper model
                    else:
                        train_dataset = FPC_Dataset(all_specs_train_tensor, freqLabels_train_tensor)
                        val_dataset = FPC_Dataset(all_specs_val_tensor, freqLabels_val_tensor)
                        test_dataset = FPC_Dataset(all_specs_test_tensor, freqLabels_test_tensor)
                        TestSpecsSaved = np.zeros((all_specs_test.shape[1], all_specs_test.shape[0], all_specs_test.shape[2]))  # our models

                else:
                    if model_name == "Ma_4Convs" or model_name =="Ma_2Convs" or  model_name =="realReal":
                        train_dataset = FPC_Dataset(freqCorr_specs_train_real_tensor, phaseLabels_train_tensor)
                        val_dataset = FPC_Dataset(freqCorr_specs_val_real_tensor, phaseLabels_val_tensor)
                        test_dataset = FPC_Dataset(freqCorr_specs_test_real_tensor, phaseLabels_test_tensor)
                        TestSpecsSaved = np.zeros((all_specs_test.shape[1], 1, all_specs_test.shape[2]))  # our models
                    elif model_name=="Tapper":
                        train_dataset = FPC_Dataset_Tapper(freqCorr_specs_train_real_tensor, phaseLabels_train_tensor)
                        val_dataset = FPC_Dataset_Tapper(freqCorr_specs_val_real_tensor, phaseLabels_val_tensor)
                        test_dataset = FPC_Dataset_Tapper(freqCorr_specs_test_real_tensor, phaseLabels_test_tensor)
                        TestSpecsSaved = np.zeros((all_specs_test.shape[1], all_specs_test.shape[2]))  # tapper model
                    else:
                        train_dataset = FPC_Dataset(freqCorr_specs_train_tensor, phaseLabels_train_tensor)
                        val_dataset = FPC_Dataset(freqCorr_specs_val_tensor, phaseLabels_val_tensor)
                        test_dataset = FPC_Dataset(freqCorr_specs_test_tensor, phaseLabels_test_tensor)
                        TestSpecsSaved = np.zeros((all_specs_test.shape[1], all_specs_test.shape[0], all_specs_test.shape[2]))  # our models

                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

                ########################################################################################################################
                # Model and Loss Function
                ########################################################################################################################
                if model_name == "Ma_2Convs":
                    model = maModel2Convs().float()
                elif model_name == "Ma_4Convs":
                    model = maModel().float()
                elif model_name == "Tapper":
                    model = tapperModel().float()
                elif model_name == "realReal":
                    model = realIn_realConv().float()
                elif model_name == "compComp":
                    model = compIn_compConv().float()
                elif model_name == "compReal":
                    model = compIn_realConv().float()
                else:
                    print("Model not found!")
                    break

                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=learn_r)
                lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda lr: 0.9)

        #         ########################################################################################################################
        #         # Training
        #         ########################################################################################################################
        #         best_val_loss = float('inf')
        #         current_loss = 0.0
        #         count = 0
        #         prev_loss = 0.0
        #         early_stop = False
        #         print(f'model number of parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        #         model.train()
        #
        #         for epoch in range(nmb_epochs):
        #             epoch_lossesT = []
        #             epoch_lossesV = []
        #
        #             for index, sample in enumerate(train_loader):
        #                 # FORWARD (Model predictions and loss)
        #                 specs, Trainlabels = sample
        #                 specs, Trainlabels = specs.to(device), Trainlabels.to(device)
        #
        #                 optimizer.zero_grad()
        #
        #                 TrainPred = model(specs.float())
        #                 TrainLoss = loss_fn(TrainPred, Trainlabels)
        #
        #                 # BACKWARD (Optimization) and UPDATE
        #                 TrainLoss.backward()
        #                 optimizer.step()
        #                 epoch_lossesT.append(TrainLoss.item())
        #
        #             model.eval()
        #             with torch.no_grad():
        #                 for sample in val_loader:
        #                     ValSpecs, ValLabels = sample
        #                     ValSpecs, ValLabels = ValSpecs.to(device), ValLabels.to(device)
        #
        #                     ValPred = model(ValSpecs.float())
        #                     val_loss = loss_fn(ValPred, ValLabels)
        #                     epoch_lossesV.append(val_loss.item())
        #
        #                     if (epoch + 1) % lr_scheduler_freq == 0:
        #                         lr_scheduler.step()
        #
        #                 if val_loss < best_val_loss:
        #                     best_weights = model.state_dict()
        #                     best_val_loss = val_loss
        #
        #                 current_loss = sum(epoch_lossesV) / len(epoch_lossesV)
        #                 if (((epoch+1) > 25) and ((prev_loss - current_loss) > -0.00000001) and ((prev_loss - current_loss) < 0.00000001)):
        #                     count = count + 1
        #                     if count == 5:
        #                         early_stop = True
        #                         print('Early Stop Criteria Reached!') # validation loss hasn't improved in 5 consecutive epochs
        #                         break
        #                 else:
        #                     count = 0
        #
        #             # Print results every epoch
        #             # print(f"Training: Predicted {TrainPred} and True {Trainlabels}")
        #             # print(f"Validation: Predicted {ValPred} and True {ValLabels}")
        #             print(f'Epoch {epoch + 1}/{nmb_epochs}, Training loss: {sum(epoch_lossesT) / len(epoch_lossesT)} and Validation loss: {sum(epoch_lossesV) / len(epoch_lossesV)}')
        #             print(f'Validation Loss Improvement of {prev_loss - current_loss}')
        #             prev_loss = current_loss
        #
        #         print(f'Training Complete for {run_name}, Early Stopping {early_stop} at epoch {epoch + 1}')
        #         print(f'Best Validation Loss was {best_val_loss}')
        #         print()
        #
        #         torch.save(best_weights, f"{run_name}.pt")
        #
                # Testing Loop
                epoch_lossesTest = []
                loop = 0
                model.load_state_dict(torch.load(f"{run_name}.pt"))
                model.eval()
                predLabels = np.zeros(phaseLabels_test.shape[1])
                TrueLabels = np.zeros(phaseLabels_test.shape[1])

                with torch.no_grad():
                    for sample in test_loader:
                        Testspecs, TestLabels = sample
                        Testspecs, TestLabels = Testspecs.to(device), TestLabels.to(device)

                        predTest = model(Testspecs.float())
                        test_loss = loss_fn(predTest, TestLabels)
                        epoch_lossesTest.append(test_loss.item())

                        if(Testspecs.cpu().shape[0]==8):
                            predLabels[loop:loop + 8] = predTest.cpu().flatten()
                            TrueLabels[loop:loop + 8] = TestLabels.cpu().flatten()
                            if model_name == "Tapper":
                                TestSpecsSaved[loop:loop + 8, :] = Testspecs.cpu()
                            else:
                                TestSpecsSaved[loop:loop + 8, :, :] = Testspecs.cpu()
                        else:
                            predLabels[loop:loop + batch_size] = predTest.cpu().flatten()
                            TrueLabels[loop:loop + batch_size] = TestLabels.cpu().flatten()
                            if model_name == "Tapper":
                                TestSpecsSaved[loop:loop + batch_size, :] = Testspecs.cpu()
                            else:
                                TestSpecsSaved[loop:loop + batch_size, :, :] = Testspecs.cpu()
                        loop = loop + batch_size

                    print(f'Testing loss: {sum(epoch_lossesTest) / len(epoch_lossesTest)}')
                    print(f'Testing Complete')
                    print()

                # save labels
                np.save(f"PredLabels_{run_name}.npy", predLabels)
                np.save(f"TrueLabels_{run_name}.npy", TrueLabels)
                np.save(f"TestSpecs_{run_name}.npy", TestSpecsSaved)
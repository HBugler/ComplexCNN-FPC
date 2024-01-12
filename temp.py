

# FPC Winter 2024
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transient_maker import TransientMaker
from Datasets import FPC_Dataset
from AblationStudyModels import realIn_realConv, compIn_realConv, compIn_compConv
from TapperModel import tapperModel
from MaModel import maModel

########################################################################################################################
# Set-up device, hyperparameters and additional variables
########################################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputTypes, convTypes = ["comp", "real"], ["real",
                                           "comp"]  # order of inputs is reversed since we are not doing realIN_compCONV
netTypes = ["freq", "phase"]
snrTypes = ["10", "5", "2_5"]
dataName = "Sim"
resid = "None"

batch_size, learn_r = 16, 0.001
nmb_epochs = 20  # 200
loss_fn = nn.L1Loss()
lr_scheduler_freq = 25

transient_count_dev = 160
transient_count_test = 6
total_gts = 2
num_spec_points = 2048

ppm_location = "C:/Users/Hanna B/PycharmProjects/FPC_2024/Development/ppm_2048.csv"
time_location = "C:/Users/Hanna B/PycharmProjects/FPC_2024/Development/t_2048.csv"

ON_fid_dev_gt_location = "C:/Users/Hanna B/PycharmProjects/FPC_2024/Test/noWater/fidsON_NoW_GABAPlus_DevSet.csv"
OFF_fid_dev_gt_location = "C:/Users/Hanna B/PycharmProjects/FPC_2024/Test/noWater/fidsOFF_NoW_GABAPlus_DevSet.csv"

ON_fid_test_gt_location = "C:/Users/Hanna B/PycharmProjects/FPC_2024/Test/noWater/fidsON_July2022.csv"
OFF_fid_test_gt_location = "C:/Users/Hanna B/PycharmProjects/FPC_2024/Test/noWater/fidsOFF_July2022.csv"

#############################
# Simulate Data
#############################
for snr in snrTypes:
    phaseLabels_dev, freqLabels_dev = np.zeros(shape=(2, 1, transient_count_dev * total_gts)), np.zeros(
        shape=(2, 1, transient_count_dev * total_gts))
    on_freqCorr_specs_dev, off_freqCorr_specs_dev = np.zeros(
        shape=(1, num_spec_points, transient_count_dev * total_gts), dtype=complex), np.zeros(
        shape=(1, num_spec_points, transient_count_dev * total_gts), dtype=complex)
    on_all_specs_dev, off_all_specs_dev = np.zeros(shape=(1, num_spec_points, transient_count_dev * total_gts),
                                                   dtype=complex), np.zeros(
        shape=(1, num_spec_points, transient_count_dev * total_gts), dtype=complex)

    phaseLabels_test, freqLabels_test = np.zeros(shape=(2, 1, transient_count_test * total_gts)), np.zeros(
        shape=(2, 1, transient_count_test * total_gts))
    on_freqCorr_specs_test, off_freqCorr_specs_test = np.zeros(
        shape=(1, num_spec_points, transient_count_test * total_gts), dtype=complex), np.zeros(
        shape=(1, num_spec_points, transient_count_test * total_gts), dtype=complex)
    on_all_specs_test, off_all_specs_test = np.zeros(shape=(1, num_spec_points, transient_count_test * total_gts),
                                                     dtype=complex), np.zeros(
        shape=(1, num_spec_points, transient_count_test * total_gts), dtype=complex)

    for groundTruthNumber in range(0, total_gts):
        # create transient objects
        noisyTransients_dev = TransientMaker(groundTruthNumber, ON_fid_dev_gt_location, OFF_fid_dev_gt_location,
                                             ppm_location, time_location, transient_count=transient_count_dev)
        noisyTransients_test = TransientMaker(groundTruthNumber, ON_fid_test_gt_location, OFF_fid_test_gt_location,
                                              ppm_location, time_location, transient_count=transient_count_test)

        # add amplitude noise, and frequency and phase shifts (order: SNR 10, SNR 5, SNR 2.5)
        normNoise_dev = [np.random.uniform(2, 2.5, size=1), np.random.uniform(4, 4.5, size=1),
                         np.random.uniform(9.5, 10, size=1)]
        normNoise_test = [np.random.uniform(2, 2.5, size=1), np.random.uniform(4, 4.5, size=1),
                          np.random.uniform(9.5, 10, size=1)]
        snrInd = snrTypes.index(snr)
        noisyTransients_dev.add_time_domain_noise(noise_level=normNoise_dev[snrInd])
        noisyTransients_test.add_time_domain_noise(noise_level=normNoise_test[snrInd])

        # obtain the phase labels and 'frequency corrected' spectra
        phaseLabels_dev[:, :, (groundTruthNumber * transient_count_dev):(
                    groundTruthNumber * transient_count_dev + transient_count_dev)] = noisyTransients_dev.add_phase_shift_random(
            phase_var=90)  # shape is (2, 1, 160)
        on_freqCorr_specs_dev[0, :,
        (groundTruthNumber * transient_count_dev):(groundTruthNumber * transient_count_dev + transient_count_dev)], \
        off_freqCorr_specs_dev[0, :, (groundTruthNumber * transient_count_dev):(
                    groundTruthNumber * transient_count_dev + transient_count_dev)] = noisyTransients_dev.get_specs()  # shape is (1, 2048, 160)

        phaseLabels_test[:, :, (groundTruthNumber * transient_count_test):(
                    groundTruthNumber * transient_count_test + transient_count_test)] = noisyTransients_test.add_phase_shift_random(
            phase_var=90)  # shape is (2, 1, 160)
        on_freqCorr_specs_test[0, :,
        (groundTruthNumber * transient_count_test):(groundTruthNumber * transient_count_test + transient_count_test)], \
        off_freqCorr_specs_test[0, :, (groundTruthNumber * transient_count_test):(
                    groundTruthNumber * transient_count_test + transient_count_test)] = noisyTransients_test.get_specs()  # shape is (1, 2048, 160)

        # obtain the frequency labels and the 'nod-corrected' spectra
        freqLabels_dev[:, :, (groundTruthNumber * transient_count_dev):(
                    groundTruthNumber * transient_count_dev + transient_count_dev)] = noisyTransients_dev.add_freq_shift_random(
            freq_var=20)
        on_all_specs_dev[0, :,
        (groundTruthNumber * transient_count_dev):(groundTruthNumber * transient_count_dev + transient_count_dev)], \
        off_all_specs_dev[0, :, (groundTruthNumber * transient_count_dev):(
                    groundTruthNumber * transient_count_dev + transient_count_dev)] = noisyTransients_dev.get_specs()

        freqLabels_test[:, :, (groundTruthNumber * transient_count_test):(
                    groundTruthNumber * transient_count_test + transient_count_test)] = noisyTransients_test.add_freq_shift_random(
            freq_var=20)
        on_all_specs_test[0, :,
        (groundTruthNumber * transient_count_test):(groundTruthNumber * transient_count_test + transient_count_test)], \
        off_all_specs_test[0, :, (groundTruthNumber * transient_count_test):(
                    groundTruthNumber * transient_count_test + transient_count_test)] = noisyTransients_test.get_specs()

        # snr = noisyTransients.get_SNR(specs=(on_all_specs_dev - off_all_specs_dev))    # sanity check ensuring SNR is set correctly

    np.save(f"TruePhaseLabels_{dataName}{snr}_{resid}_Dev.npy", phaseLabels_dev)
    np.save(f"TrueFreqLabels_{dataName}{snr}_{resid}_Dev.npy", freqLabels_dev)
    np.save(f"ON_FreqCorrectedSpecs_{dataName}{snr}_{resid}_Dev.npy", on_freqCorr_specs_dev)
    np.save(f"OFF_FreqCorrectedSpecs_{dataName}{snr}_{resid}_Dev.npy", off_freqCorr_specs_dev)
    np.save(f"ON_AllSpecs_{dataName}{snr}_{resid}_Dev.npy", on_all_specs_dev)
    np.save(f"OFF_AllSpecs_{dataName}{snr}_{resid}_Dev.npy", off_all_specs_dev)

    np.save(f"TruePhaseLabels_{dataName}{snr}_{resid}_Test.npy", phaseLabels_test)
    np.save(f"TrueFreqLabels_{dataName}{snr}_{resid}_Test.npy", freqLabels_test)
    np.save(f"ON_FreqCorrectedSpecs_{dataName}{snr}_{resid}_Test.npy", on_freqCorr_specs_test)
    np.save(f"OFF_FreqCorrectedSpecs_{dataName}{snr}_{resid}_Test.npy", off_freqCorr_specs_test)
    np.save(f"ON_AllSpecs_{dataName}{snr}_{resid}_Test.npy", on_all_specs_test)
    np.save(f"OFF_AllSpecs_{dataName}{snr}_{resid}_Test.npy", off_all_specs_test)

    ########################################################################################################################
    # Clean and Split Data
    ########################################################################################################################
    # Select window of points
    ppm = np.ndarray.round(noisyTransients_dev.ppm, 2)
    ind_close, ind_far = np.amax(np.where(ppm == 0.00)), np.amin(np.where(ppm == 7.83))
    ppm_1024 = ppm[ind_far:ind_close]

    on_freqCorr_specs_dev = on_freqCorr_specs_dev[:, ind_far:ind_close, :]
    off_freqCorr_specs_dev = off_freqCorr_specs_dev[:, ind_far:ind_close, :]
    on_all_specs_dev = on_all_specs_dev[:, ind_far:ind_close, :]
    off_all_specs_dev = off_all_specs_dev[:, ind_far:ind_close,
                        :]  # shape is (subspectra (1), num of spectral points (1024), num transients(160*250))

    on_freqCorr_specs_test = on_freqCorr_specs_test[:, ind_far:ind_close, :]
    off_freqCorr_specs_test = off_freqCorr_specs_test[:, ind_far:ind_close, :]
    on_all_specs_test = on_all_specs_test[:, ind_far:ind_close, :]
    off_all_specs_test = off_all_specs_test[:, ind_far:ind_close,
                         :]  # shape is (subspectra (1), num of spectral points (1024), num transients(160*250))

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
    off_phaseLabels_test = phaseLabels_dev[1, :, :]
    on_freqLabels_test = freqLabels_dev[0, :, :]
    off_freqLabels_test = freqLabels_dev[1, :, :]

    # data normalization (currently normalized per scan) and reshaping
    both = np.concatenate((on_all_specs_train, off_all_specs_train))
    mean_train, std_train = both.mean(), both.std()

    on_freqCorr_specs_train = (on_freqCorr_specs_train - mean_train) / std_train
    on_freqCorr_specs_val = (on_freqCorr_specs_val - mean_train) / std_train
    off_freqCorr_specs_train = (off_freqCorr_specs_train - mean_train) / std_train
    off_freqCorr_specs_val = (off_freqCorr_specs_val - mean_train) / std_train

    on_all_specs_train = (on_all_specs_train - mean_train) / std_train
    on_all_specs_val = (on_all_specs_val - mean_train) / std_train
    off_all_specs_train = (off_all_specs_train - mean_train) / std_train
    off_all_specs_val = (off_all_specs_val - mean_train) / std_train

    on_freqCorr_specs_train = np.einsum('kij->kji', on_freqCorr_specs_train)
    on_freqCorr_specs_val = np.einsum('kij->kji', on_freqCorr_specs_val)
    off_freqCorr_specs_train = np.einsum('kij->kji', off_freqCorr_specs_train)
    off_freqCorr_specs_val = np.einsum('kij->kji', off_freqCorr_specs_val)

    on_all_specs_train = np.einsum('kij->kji', on_all_specs_train)
    on_all_specs_val = np.einsum('kij->kji', on_all_specs_val)
    off_all_specs_train = np.einsum('kij->kji', off_all_specs_train)
    off_all_specs_val = np.einsum('kij->kji', off_all_specs_val)

    on_freqCorr_specs_test = (on_freqCorr_specs_test - mean_train) / std_train
    off_freqCorr_specs_test = (off_freqCorr_specs_test - mean_train) / std_train
    on_all_specs_test = (on_all_specs_test - mean_train) / std_train
    off_all_specs_test = (off_all_specs_test - mean_train) / std_train

    on_freqCorr_specs_test = np.einsum('kij->kji', on_freqCorr_specs_test)
    off_freqCorr_specs_test = np.einsum('kij->kji', off_freqCorr_specs_test)
    on_all_specs_test = np.einsum('kij->kji', on_all_specs_test)
    off_all_specs_test = np.einsum('kij->kji', off_all_specs_test)

    # Concatenate data and separate real and imaginary
    all_specs_train, all_specs_val = np.zeros(shape=(2, on_all_specs_train.shape[1] * 2, 1024)), np.zeros(
        shape=(2, on_all_specs_val.shape[1] * 2, 1024))
    freqCorr_specs_train, freqCorr_specs_val = np.zeros(
        shape=(2, on_freqCorr_specs_train.shape[1] * 2, 1024)), np.zeros(
        shape=(2, on_freqCorr_specs_val.shape[1] * 2, 1024))
    freqLabels_train, freqLabels_val = np.zeros(shape=(1, on_freqLabels_train.shape[1] * 2)), np.zeros(
        shape=(1, on_freqLabels_val.shape[1] * 2))
    phaseLabels_train, phaseLabels_val = np.zeros(shape=(1, on_phaseLabels_train.shape[1] * 2)), np.zeros(
        shape=(1, on_phaseLabels_val.shape[1] * 2))

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

    freqLabels_train[:, :on_freqLabels_train.shape[1]] = np.squeeze(on_freqLabels_train)
    freqLabels_train[:, on_freqLabels_train.shape[1]:] = off_freqLabels_train
    freqLabels_val[:, :on_freqLabels_val.shape[1]] = on_freqLabels_val
    freqLabels_val[:, on_freqLabels_val.shape[1]:] = off_freqLabels_val

    phaseLabels_train[:, :on_phaseLabels_train.shape[1]] = on_phaseLabels_train
    phaseLabels_train[:, on_phaseLabels_train.shape[1]:] = off_phaseLabels_train
    phaseLabels_val[:, :on_phaseLabels_val.shape[1]] = on_phaseLabels_val
    phaseLabels_val[:, on_phaseLabels_val.shape[1]:] = off_phaseLabels_val

    all_specs_test, freqCorr_specs_test = np.zeros(shape=(2, on_all_specs_test.shape[1] * 2, 1024)), np.zeros(
        shape=(2, on_freqCorr_specs_test.shape[1] * 2, 1024))
    freqLabels_test, phaseLabels_test = np.zeros(shape=(1, on_freqLabels_test.shape[1] * 2)), np.zeros(
        shape=(1, on_phaseLabels_test.shape[1] * 2))

    all_specs_test[0, :on_all_specs_test.shape[1], :] = np.squeeze(on_all_specs_test.real)
    all_specs_test[0, on_all_specs_test.shape[1]:, :] = np.squeeze(off_all_specs_test.real)
    all_specs_test[1, :on_all_specs_test.shape[1], :] = np.squeeze(on_all_specs_test.imag)
    all_specs_test[1, on_all_specs_test.shape[1]:, :] = np.squeeze(off_all_specs_test.imag)
    freqCorr_specs_test[0, :on_freqCorr_specs_test.shape[1], :] = np.squeeze(on_freqCorr_specs_test.real)
    freqCorr_specs_test[0, on_freqCorr_specs_test.shape[1]:, :] = np.squeeze(off_freqCorr_specs_test.real)
    freqCorr_specs_test[1, :on_freqCorr_specs_test.shape[1], :] = np.squeeze(on_freqCorr_specs_test.imag)
    freqCorr_specs_test[1, on_freqCorr_specs_test.shape[1]:, :] = np.squeeze(off_freqCorr_specs_test.imag)
    freqLabels_test[:, :on_freqLabels_test.shape[1]] = np.squeeze(on_freqLabels_test)
    freqLabels_test[:, on_freqLabels_test.shape[1]:] = np.squeeze(off_freqLabels_test)
    phaseLabels_test[:, :on_phaseLabels_test.shape[1]] = np.squeeze(on_phaseLabels_test)
    phaseLabels_test[:, on_phaseLabels_test.shape[1]:] = np.squeeze(off_phaseLabels_test)

    # Real Valued CNN
    # magnitude value calculation
    all_specs_train_real = np.sqrt(
        (all_specs_train[0, :, :]) * (all_specs_train[0, :, :]) + (all_specs_train[1, :, :]) * (
        all_specs_train[1, :, :]))
    all_specs_train_real = all_specs_train_real[np.newaxis, :, :]
    all_specs_val_real = np.sqrt(
        (all_specs_val[0, :, :]) * (all_specs_val[0, :, :]) + (all_specs_val[1, :, :]) * (all_specs_val[1, :, :]))
    all_specs_val_real = all_specs_val_real[np.newaxis, :, :]

    # real value (because phase network)
    freqCorr_specs_train_real = (freqCorr_specs_train[0, :, :])[np.newaxis, :, :]
    freqCorr_specs_val_real = (freqCorr_specs_val[0, :, :])[np.newaxis, :, :]

    all_specs_test_real = np.sqrt(
        (all_specs_test[0, :, :]) * (all_specs_test[0, :, :]) + (all_specs_test[1, :, :]) * (all_specs_test[1, :, :]))
    all_specs_test_real = all_specs_test_real[np.newaxis, :, :]
    freqCorr_specs_test_real = (freqCorr_specs_test[0, :, :])[np.newaxis, :, :]

    # Convert to tensors and transfer operations to GPU
    all_specs_train_tensor = torch.from_numpy(
        all_specs_train).float()  # shape is (#channels, #samples, #spectralPoints)
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

    ########################################################################################################################
    # Set-up Datasets, Dataloaders and Transforms
    ########################################################################################################################
    for input in inputTypes:
        for conv in convTypes:
            for net in netTypes:
                run_name = f'{net}_{dataName}{snr}_{input}{conv}'
                print(run_name)

                # select based on network type
                if net == "freq":
                    if input == "real":
                        train_dataset = FPC_Dataset(all_specs_train_real_tensor, freqLabels_train_tensor)
                        val_dataset = FPC_Dataset(all_specs_val_real_tensor, freqLabels_val_tensor)
                        test_dataset = FPC_Dataset(all_specs_test_real_tensor, freqLabels_test_tensor)
                    else:
                        train_dataset = FPC_Dataset(all_specs_train_tensor, freqLabels_train_tensor)
                        val_dataset = FPC_Dataset(all_specs_val_tensor, freqLabels_val_tensor)
                        test_dataset = FPC_Dataset(all_specs_test_tensor, freqLabels_test_tensor)

                else:
                    if input == "real":
                        train_dataset = FPC_Dataset(freqCorr_specs_train_real_tensor, freqLabels_train_tensor)
                        val_dataset = FPC_Dataset(freqCorr_specs_val_real_tensor, freqLabels_val_tensor)
                        test_dataset = FPC_Dataset(freqCorr_specs_test_real_tensor, freqLabels_test_tensor)
                    else:
                        train_dataset = FPC_Dataset(freqCorr_specs_train_tensor, phaseLabels_train_tensor)
                        val_dataset = FPC_Dataset(freqCorr_specs_val_tensor, phaseLabels_val_tensor)
                        test_dataset = FPC_Dataset(freqCorr_specs_test_tensor, phaseLabels_test_tensor)

                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

                ########################################################################################################################
                # Model and Loss Function
                ########################################################################################################################
                # select based on input and convolution type
                if (input == "real" and conv == "real"):
                    model = realIn_realConv().float()
                elif (input == "comp" and conv == "real"):
                    model = compIn_realConv().float()
                else:
                    model = compIn_compConv().float()

                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=learn_r)
                lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda lr: 0.5)

                ########################################################################################################################
                # Training
                ########################################################################################################################
                best_val_loss = float('inf')
                current_loss = 0.0
                print(f'model number of parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
                model.train()

                for epoch in range(nmb_epochs):
                    epoch_lossesT = []
                    epoch_lossesV = []

                    for index, sample in enumerate(train_loader):
                        # FORWARD (Model predictions and loss)
                        specs, Trainlabels = sample
                        specs, Trainlabels = specs.to(device), Trainlabels.to(device)

                        optimizer.zero_grad()

                        TrainPred = model(specs.float())
                        TrainLoss = loss_fn(TrainPred, Trainlabels)

                        # BACKWARD (Optimization) and UPDATE
                        TrainLoss.backward()
                        optimizer.step()
                        epoch_lossesT.append(TrainLoss.item())

                    model.eval()
                    with torch.no_grad():
                        for sample in val_loader:
                            ValSpecs, ValLabels = sample
                            ValSpecs, ValLabels = ValSpecs.to(device), ValLabels.to(device)

                            ValPred = model(ValSpecs.float())
                            val_loss = loss_fn(ValPred, ValLabels)
                            epoch_lossesV.append(val_loss.item())

                            if (epoch + 1) % lr_scheduler_freq == 0:
                                lr_scheduler.step()

                        if val_loss < best_val_loss:
                            best_weights = model.state_dict()
                            best_val_loss = val_loss

                    # Print results every epoch
                    print(f"Training: Predicted {TrainPred} and True {Trainlabels}")
                    print(f"Validation: Predicted {ValPred} and True {ValLabels}")
                    print(
                        f'Epoch {epoch + 1}/{nmb_epochs}, Training loss: {sum(epoch_lossesT) / len(epoch_lossesT)} and Validation loss: {sum(epoch_lossesV) / len(epoch_lossesV)}')

                print(f'Training Complete')
                print(f'Best Validation Loss was {best_val_loss}')
                print()

                torch.save(best_weights, f"{run_name}.pt")

                # Testing Loop
                epoch_lossesTest = []
                loop = 0
                model.eval()
                predLabels = np.zeros(phaseLabels_test.shape[1])

                with torch.no_grad():
                    for sample in test_loader:
                        Testspecs, TestLabels = sample
                        Testspecs, TestLabels = Testspecs.to(device), TestLabels.to(device)

                        predTest = model(Testspecs.float())
                        test_loss = loss_fn(predTest, TestLabels)
                        epoch_lossesTest.append(test_loss.item())
                        predLabels[loop:loop + batch_size] = predTest.cpu().flatten()
                        loop = loop + batch_size

                print(f"Testing: Predicted {predTest} and True {TestLabels}")
                print(f'Testing loss: {sum(epoch_lossesTest) / len(epoch_lossesTest)}')
                print(f'Testing Complete')
                print()

                # save labels
                np.save(f"PredLabels_{run_name}.npy", predLabels)

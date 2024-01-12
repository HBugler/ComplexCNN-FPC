# FPC Winter 2024
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transient_maker import TransientMaker
from Datasets import FPC_Dataset
from AblationStudyModels import realIn_realConv, compIn_realConv, compIn_compConv

########################################################################################################################
# Set-up Device and Hyperparameters
########################################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputType = "R"              #"C"
waterType = "Pos"            #"Neg"  "Mix"
convType = "R"               #"C"
netType = "Freq"             #"Phase"
snrType = "10"               #"5"    "2.5"
dataType = "Sim"
run_name = f"{netType}_{dataType}{waterType}{snrType}_{inputType}{convType}"

batch_size, learn_r = 16, 0.0001
nmb_epochs = 10    #200

########################################################################################################################
# Simulate Data (train, val and test)
########################################################################################################################
transient_count = 160
total_gts = 250
num_spec_points = 2048
phaseLabels_dev, freqLabels_dev = np.zeros(shape=(2, 1, transient_count*total_gts)), np.zeros(shape=(2, 1, transient_count*total_gts))
on_freqCorr_specs_dev, off_freqCorr_specs_dev = np.zeros(shape=(1, num_spec_points, transient_count*total_gts), dtype=complex), np.zeros(shape=(1, num_spec_points, transient_count*total_gts), dtype=complex)
on_all_specs_dev, off_all_specs_dev = np.zeros(shape=(1, num_spec_points, transient_count*total_gts), dtype=complex), np.zeros(shape=(1, num_spec_points, transient_count*total_gts), dtype=complex)

ppm_location ="C:/Users/Hanna B/PycharmProjects/FPC_2024/Development/ppm_2048.csv"
time_location = "C:/Users/Hanna B/PycharmProjects/FPC_2024/Development/t_2048.csv"

#############################
# Simulate Train and Val Data
#############################
# data locations
ON_fid_dev_gt_location ="C:/Users/Hanna B/PycharmProjects/FPC_2024/Test/noWater/fidsON_July2022.csv"
OFF_fid_dev_gt_location ="C:/Users/Hanna B/PycharmProjects/FPC_2024/Test/noWater/fidsOFF_July2022.csv"

for groundTruthNumber in range(0, total_gts):
    # create transient object
    noisyTransients = TransientMaker(groundTruthNumber, ON_fid_dev_gt_location, OFF_fid_dev_gt_location, ppm_location, time_location, transient_count=transient_count)

    # add amplitude noise, and frequency and phase shifts
    normNoise = np.random.uniform(2, 2.5, size=1)   #SNR 10
    # normNoise = np.random.uniform(4, 4.5, size=1)   #SNR 5
    # normNoise = np.random.uniform(9.5, 10, size=1)   #SNR 2.5
    noisyTransients.add_time_domain_noise(noise_level=normNoise)

    # obtain the phase labels and 'frequency corrected' spectra
    phaseLabels_dev[:, :, (groundTruthNumber*transient_count):(groundTruthNumber*transient_count+transient_count)] = noisyTransients.add_phase_shift_random(phase_var=90)            # shape is (2, 1, 160)
    on_freqCorr_specs_dev[0, :, (groundTruthNumber*transient_count):(groundTruthNumber*transient_count+transient_count)],\
    off_freqCorr_specs_dev[0, :, (groundTruthNumber*transient_count):(groundTruthNumber*transient_count+transient_count)] = noisyTransients.get_specs()       # shape is (1, 2048, 160)

    # obtain the frequency labels and the 'nod-corrected' spectra
    freqLabels_dev[:, :, (groundTruthNumber*transient_count):(groundTruthNumber*transient_count+transient_count)] = noisyTransients.add_freq_shift_random(freq_var=20)
    on_all_specs_dev[0, :, (groundTruthNumber*transient_count):(groundTruthNumber*transient_count+transient_count)], \
    off_all_specs_dev[0, :, (groundTruthNumber*transient_count):(groundTruthNumber*transient_count+transient_count)] = noisyTransients.get_specs()

    # snr = noisyTransients.get_SNR(specs=(on_all_specs_dev - off_all_specs_dev))    # sanity check ensuring SNR is set correctly

    np.save(f"TruePhaseLabels_EXAMPLE_Dev.npy", phaseLabels_dev)
    np.save(f"TrueFreqLabels_EXAMPLE_Dev.npy", freqLabels_dev)
    np.save(f"ON_FreqCorrectedSpecs_EXAMPLE_Dev.npy", on_freqCorr_specs_dev)
    np.save(f"OFF_FreqCorrectedSpecs_EXAMPLE_Dev.npy", off_freqCorr_specs_dev)
    np.save(f"ON_AllSpecs_EXAMPLE_Dev.npy", on_all_specs_dev)
    np.save(f"OFF_AllSpecs_EXAMPLE_Dev.npy", off_all_specs_dev)
########################################################################################################################
# Clean and Split Data
########################################################################################################################
# Select window of points
ppm = np.ndarray.round(noisyTransients.ppm, 2)
ind_close, ind_far = np.amax(np.where(ppm == 0.00)), np.amin(np.where(ppm == 7.83))
ppm_1024 = ppm[ind_far:ind_close]
on_freqCorr_specs_dev = on_freqCorr_specs_dev[:, ind_far:ind_close, :]
off_freqCorr_specs_dev = off_freqCorr_specs_dev[:, ind_far:ind_close, :]
on_all_specs_dev = on_all_specs_dev[:, ind_far:ind_close, :]
off_all_specs_dev = off_all_specs_dev[:, ind_far:ind_close, :]          # shape is (subspectra (1), num of spectral points (1024), num transients(160*250))


# data split (80 training / 20 validation)
on_freqCorr_specs_train = on_freqCorr_specs_dev[:, :, :int(on_freqCorr_specs_dev.shape[2]*0.8)]
on_freqCorr_specs_val = on_freqCorr_specs_dev[:, :, int(on_freqCorr_specs_dev.shape[2]*0.8):]
off_freqCorr_specs_train = off_freqCorr_specs_dev[:, :, :int(off_freqCorr_specs_dev.shape[2]*0.8)]
off_freqCorr_specs_val = off_freqCorr_specs_dev[:, :, int(off_freqCorr_specs_dev.shape[2]*0.8):]

on_all_specs_train = on_all_specs_dev[:, :, :int(on_all_specs_dev.shape[2]*0.8)]
on_all_specs_val = on_all_specs_dev[:, :, int(on_all_specs_dev.shape[2]*0.8):]
off_all_specs_train = off_all_specs_dev[:, :, :int(off_all_specs_dev.shape[2]*0.8)]
off_all_specs_val = off_all_specs_dev[:, :, int(off_all_specs_dev.shape[2]*0.8):]

on_phaseLabels_train = phaseLabels_dev[0, :, :int(phaseLabels_dev.shape[2]*0.8)]
on_phaseLabels_val = phaseLabels_dev[0, :, int(phaseLabels_dev.shape[2]*0.8):]
off_phaseLabels_train = phaseLabels_dev[1, :, :int(phaseLabels_dev.shape[2]*0.8)]
off_phaseLabels_val = phaseLabels_dev[1, :, int(phaseLabels_dev.shape[2]*0.8):]
on_freqLabels_train = freqLabels_dev[0, :, :int(freqLabels_dev.shape[2]*0.8)]
on_freqLabels_val = freqLabels_dev[0, :, int(freqLabels_dev.shape[2]*0.8):]
off_freqLabels_train = freqLabels_dev[1, :, :int(freqLabels_dev.shape[2]*0.8)]
off_freqLabels_val = freqLabels_dev[1, :, int(freqLabels_dev.shape[2]*0.8):]


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


# Concatenate data and separate real and imaginary
all_specs_train, all_specs_val = np.zeros(shape=(2, on_all_specs_train.shape[1]*2, 1024)), np.zeros(shape=(2, on_all_specs_val.shape[1]*2, 1024))
freqCorr_specs_train, freqCorr_specs_val = np.zeros(shape=(2, on_freqCorr_specs_train.shape[1]*2, 1024)), np.zeros(shape=(2, on_freqCorr_specs_val.shape[1]*2, 1024))
freqLabels_train, freqLabels_val = np.zeros(shape=(1, on_freqLabels_train.shape[1]*2)), np.zeros(shape=(1, on_freqLabels_val.shape[1]*2))
phaseLabels_train, phaseLabels_val = np.zeros(shape=(1, on_phaseLabels_train.shape[1]*2)), np.zeros(shape=(1, on_phaseLabels_val.shape[1]*2))

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

# # Real Valued CNN
# # magnitude value calculation
all_specs_train = np.sqrt((all_specs_train[0, :, :])*(all_specs_train[0, :, :]) + (all_specs_train[1, :, :])*(all_specs_train[1, :, :]))
all_specs_train = all_specs_train[np.newaxis, :, :]
all_specs_val = np.sqrt((all_specs_val[0, :, :])*(all_specs_val[0, :, :]) + (all_specs_val[1, :, :])*(all_specs_val[1, :, :]))
all_specs_val = all_specs_val[np.newaxis, :, :]

# real value
freqCorr_specs_train = (freqCorr_specs_train[0, :, :])[np.newaxis, :, :]
freqCorr_specs_val = (freqCorr_specs_val[0, :, :])[np.newaxis, :, :]


# Convert to tensors and transfer operations to GPU
all_specs_train_tensor = torch.from_numpy(all_specs_train).float()      # shape is (#channels, #samples, #spectralPoints)
all_specs_val_tensor = torch.from_numpy(all_specs_val).float()
freqCorr_specs_train_tensor = torch.from_numpy(freqCorr_specs_train).float()
freqCorr_specs_val_tensor = torch.from_numpy(freqCorr_specs_val).float()

freqLabels_train_tensor = torch.from_numpy(freqLabels_train).float()    # shape is (1, #samples)
freqLabels_val_tensor = torch.from_numpy(freqLabels_val).float()
phaseLabels_train_tensor = torch.from_numpy(phaseLabels_train).float()
phaseLabels_val_tensor = torch.from_numpy(phaseLabels_val).float()

########################################################################################################################
# Set-up Datasets, Dataloaders and Transforms
########################################################################################################################
freq_train_dataset = FPC_Dataset(all_specs_train_tensor, freqLabels_train_tensor)
freq_train_loader = DataLoader(dataset=freq_train_dataset, batch_size=batch_size, shuffle=True)

freq_val_dataset = FPC_Dataset(all_specs_val_tensor, freqLabels_val_tensor)
freq_val_loader = DataLoader(dataset=freq_val_dataset, batch_size=batch_size, shuffle=True)

########################################################################################################################
# Model and Loss Function
########################################################################################################################
model = realIn_realConv().float()
model.to(device)
loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learn_r)
lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda lr: 0.5)
lr_scheduler_freq = 25

########################################################################################################################
# Training
########################################################################################################################
current_loss = 0.0
model.train()
print(f'model number of parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

for epoch in range(nmb_epochs):
    epoch_lossesT = []
    epoch_lossesV = []

    for index, sample in enumerate(freq_train_loader):
        # FORWARD (Model predictions and loss)
        specs, labels = sample
        specs = specs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        TrainPred = model(specs.float())
        TrainLoss = loss_fn(TrainPred, labels)

        # BACKWARD (Optimization) and UPDATE
        TrainLoss.backward()
        optimizer.step()
        epoch_lossesT.append(TrainLoss.item())

    model.eval()
    with torch.no_grad():
        for sample in freq_val_loader:
            ValSpecs, ValLabels = sample
            ValSpecs = ValSpecs.to(device)
            ValLabels = ValLabels.to(device)

            ValPred = model(ValSpecs.float())
            val_loss = loss_fn(ValPred, ValLabels)
            epoch_lossesV.append(val_loss.item())

            if (epoch+1)%lr_scheduler_freq==0:
                lr_scheduler.step()



    # Print results every epoch
    print(f"Training: Predicted {TrainPred} and True {labels}")
    print(f"Validation: Predicted {ValPred} and True {ValLabels}")
    print(f'Epoch {epoch+1}/{nmb_epochs}, Training loss: {sum(epoch_lossesT)/len(epoch_lossesT)} and Validation loss: {sum(epoch_lossesV)/len(epoch_lossesV)}')

print(f'Training Complete')
print()

torch.save(model.state_dict(), f"{run_name}.pt")


# # Testing Loop
# epoch_lossesTest = []
# loop = 0
# model.eval()
#
# with torch.no_grad():
#     for sample in test_loader:
#         Testspecs, TestLabels = sample
#         predTest = model(Testspecs.float())
#         test_loss = loss_fn(predTest, TestLabels)
#         epoch_lossesTest.append(test_loss.item())
#         trueLabels[loop:loop+8] = TestLabels.cpu().flatten()
#         predLabels[loop:loop+8] = predTest.cpu().flatten()
#         loop = loop + 8
#
# print(f"Testing: Predicted {predTest} and True {TestLabels}")
# print(f'Testing loss: {sum(epoch_lossesTest) / len(epoch_lossesTest)}')
# print(f'Testing Complete')
# print()
#
# # save labels
# np.save(f"PredLabels_{run_name}.npy", predLabels)     #needs to be manually changed with model
# np.save(f"TrueLabels_{run_name}.npy", trueLabels)
# np.save(f"AllSpecs_{run_name}.npy", all_specs_test)
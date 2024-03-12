#######################################################################################################################
# To Investigate FPC Data
#######################################################################################################################
import numpy as np
import math
import matlab.engine
import random
import matplotlib.pyplot as plt

# Import necessary data and set-up variables
FPC_dir_gen = "C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/"
InVivoGTs_dir = "C:/Users/Hanna B/Desktop/Research/CodeAndData/ISBI Challenge/ISBI In Vivo Data/track_02_big_gaba_GANNET/"

ppm = np.load(f"{FPC_dir_gen}Data/InVivo/ppm_InVivo.npy").flatten()
time = np.load(f"{FPC_dir_gen}Data/InVivo/time_InVivo.npy").flatten()
eng1 = matlab.engine.start_matlab()

G5_fids_ON, G5_fids_OFF = np.empty((1, 160*12, 2048), dtype=complex), np.empty((1, 160*12, 2048), dtype=complex)
G7_fids_ON, G7_fids_OFF = np.empty((1, 160*12, 2048), dtype=complex), np.empty((1, 160*12, 2048), dtype=complex)
G8_fids_ON, G8_fids_OFF = np.empty((1, 160*12, 2048), dtype=complex), np.empty((1, 160*12, 2048), dtype=complex)

# to denote ON (value=1) and OFF (value=0) in order 1, 0, 1, 0....
interleave = np.squeeze(np.array(eng1.load(f"C:/Users/Hanna B/Desktop/Research/CodeAndData/ISBI Challenge/ISBI In Vivo Data/track_02_big_gaba_GANNET/G5_MP/S01_GABA_68_final.mat")['MRS_struct']['fids']['ON_OFF']))
interON, interOFF, indTrans = 0, 0, 0

# sort scans and subspectra
for i in range(1, 13):
    if i<10:
        # (2048, 320)
        GABA_fids5 = np.array(eng1.load(f"{InVivoGTs_dir}G5_MP/S0{i}_GABA_68_final.mat")['MRS_struct']['fids']['data'])
        GABA_fids7 = np.array(eng1.load(f"{InVivoGTs_dir}G7_MP/S0{i}_GABA_68_final.mat")['MRS_struct']['fids']['data'])
        GABA_fids8 = np.array(eng1.load(f"{InVivoGTs_dir}G8_MP/S0{i}_GABA_68_final.mat")['MRS_struct']['fids']['data'])
    else:
        GABA_fids5 = np.array(eng1.load(f"{InVivoGTs_dir}G5_MP/S{i}_GABA_68_final.mat")['MRS_struct']['fids']['data'])
        GABA_fids7 = np.array(eng1.load(f"{InVivoGTs_dir}G7_MP/S{i}_GABA_68_final.mat")['MRS_struct']['fids']['data'])
        GABA_fids8 = np.array(eng1.load(f"{InVivoGTs_dir}G8_MP/S{i}_GABA_68_final.mat")['MRS_struct']['fids']['data'])

    interON, interOFF = 0, 0
    for k in range(0, 320):
        if interleave[k]==1:
            G5_fids_ON[0, indTrans + interON, :] = GABA_fids5[:, k]
            G7_fids_ON[0, indTrans + interON, :] = GABA_fids7[:, k]
            G8_fids_ON[0, indTrans + interON, :] = GABA_fids8[:, k]
            interON = interON+1
        else:
            G5_fids_OFF[0, indTrans + interOFF, :] = GABA_fids5[:, k]
            G7_fids_OFF[0, indTrans + interOFF, :] = GABA_fids7[:, k]
            G8_fids_OFF[0, indTrans + interOFF, :] = GABA_fids8[:, k]
            interOFF = interOFF+1

    indTrans = indTrans + 160

specsON_G5 = np.fft.fftshift(np.fft.ifft(G5_fids_ON, axis=2), axes=2)
specsOFF_G5 = np.fft.fftshift(np.fft.ifft(G5_fids_OFF, axis=2), axes=2)
specsON_G7 = np.fft.fftshift(np.fft.ifft(G7_fids_ON, axis=2), axes=2)
specsOFF_G7 = np.fft.fftshift(np.fft.ifft(G7_fids_OFF, axis=2), axes=2)
specsON_G8 = np.fft.fftshift(np.fft.ifft(G8_fids_ON, axis=2), axes=2)
specsOFF_G8 = np.fft.fftshift(np.fft.ifft(G8_fids_OFF, axis=2), axes=2)

np.save("G5_specs_ON.npy", specsON_G5)
np.save("G5_specs_OFF.npy", specsOFF_G5)
np.save("G7_specs_ON.npy", specsON_G7)
np.save("G7_specs_OFF.npy", specsOFF_G7)
np.save("G8_specs_ON.npy", specsON_G8)
np.save("G8_specs_OFF.npy", specsOFF_G8)

# concatenate specs
all_specs_ON = np.concatenate((specsON_G5, specsON_G7, specsON_G8), axis=1)
all_specs_OFF = np.concatenate((specsOFF_G5, specsOFF_G7, specsOFF_G8), axis=1)
np.save("allSpecsInVivoON_NoOffsets.npy", all_specs_ON)
np.save("allSpecsInVivoOFF_NoOffsets.npy", all_specs_OFF)

# convert to fids
all_specs_OFF = np.fft.fft(np.fft.fftshift(all_specs_OFF, axes=2), axis=2)
all_specs_ON = np.fft.fft(np.fft.fftshift(all_specs_ON, axes=2), axis=2)

all_specs_ON_small = np.zeros(all_specs_ON.shape, dtype=complex)
all_specs_ON_med = np.zeros(all_specs_ON.shape, dtype=complex)
all_specs_ON_large = np.zeros(all_specs_ON.shape, dtype=complex)

all_specs_OFF_small = np.zeros(all_specs_ON.shape, dtype=complex)
all_specs_OFF_med = np.zeros(all_specs_ON.shape, dtype=complex)
all_specs_OFF_large = np.zeros(all_specs_ON.shape, dtype=complex)


# create noise (Dimension 0 (ON (1) / OFF (0))
FnoiseSmall = np.random.uniform(low=-5, high=5, size=(2, all_specs_ON.shape[1]))
PnoiseSmall = np.random.uniform(low=-20, high=20, size=(2, all_specs_ON.shape[1]))

FnoiseMed = np.random.uniform(low=-10, high=10, size=(2, all_specs_ON.shape[1]))
PnoiseMed = np.random.uniform(low=-40, high=40, size=(2, all_specs_ON.shape[1]))

FnoiseLarge = np.random.uniform(low=-20, high=20, size=(2, all_specs_ON.shape[1]))
PnoiseLarge = np.random.uniform(low=-90, high=90, size=(2, all_specs_ON.shape[1]))

# apply noise
for k in range(0, all_specs_ON.shape[1]):
    # small offsets
    all_specs_ON_small[0, k, :] = all_specs_ON[0, k, :] * np.exp(-1j * PnoiseSmall[1, k] * math.pi / 180)
    all_specs_ON_small[0, k, :] = all_specs_ON_small[0, k, :] * np.exp(-1j * time * FnoiseSmall[1, k] * 2 * math.pi)

    all_specs_OFF_small[0, k, :] = all_specs_OFF[0, k, :] * np.exp(-1j * PnoiseSmall[0, k] * math.pi / 180)
    all_specs_OFF_small[0, k, :] = all_specs_OFF_small[0, k, :] * np.exp(-1j * time * FnoiseSmall[0, k] * 2 * math.pi)

    # medium offsets
    all_specs_ON_med[0, k, :] = all_specs_ON[0, k, :] * np.exp(-1j * PnoiseMed[1, k] * math.pi / 180)
    all_specs_ON_med[0, k, :] = all_specs_ON_med[0, k, :] * np.exp(-1j * time * FnoiseMed[1, k] * 2 * math.pi)

    all_specs_OFF_med[0, k, :] = all_specs_OFF[0, k, :] * np.exp(-1j * PnoiseMed[0, k] * math.pi / 180)
    all_specs_OFF_med[0, k, :] = all_specs_OFF_med[0, k, :] * np.exp(-1j * time * FnoiseMed[0, k] * 2 * math.pi)

    # large offsets
    all_specs_ON_large[0, k, :] = all_specs_ON[0, k, :] * np.exp(-1j * PnoiseLarge[1, k] * math.pi / 180)
    all_specs_ON_large[0, k, :] = all_specs_ON_large[0, k, :] * np.exp(-1j * time * FnoiseLarge[1, k] * 2 * math.pi)

    all_specs_OFF_large[0, k, :] = all_specs_OFF[0, k, :] * np.exp(-1j * PnoiseLarge[0, k] * math.pi / 180)
    all_specs_OFF_large[0, k, :] = all_specs_OFF_large[0, k, :] * np.exp(-1j * time * FnoiseLarge[0, k] * 2 * math.pi)

# convert to specs
all_specs_ON_small = np.fft.fftshift(np.fft.ifft(all_specs_ON_small, axis=2), axes=2)
all_specs_OFF_small = np.fft.fftshift(np.fft.ifft(all_specs_OFF_small, axis=2), axes=2)

all_specs_ON_med = np.fft.fftshift(np.fft.ifft(all_specs_ON_med, axis=2), axes=2)
all_specs_OFF_med = np.fft.fftshift(np.fft.ifft(all_specs_OFF_med, axis=2), axes=2)

all_specs_ON_large = np.fft.fftshift(np.fft.ifft(all_specs_ON_large, axis=2), axes=2)
all_specs_OFF_large = np.fft.fftshift(np.fft.ifft(all_specs_OFF_large, axis=2), axes=2)

# save data (concatenate with ON first)
allSpecsSmall = np.empty((2, all_specs_ON_small.shape[0], all_specs_ON_small.shape[1]))
allSpecsSmall[0, :, :], allSpecsSmall[1, :, :] = all_specs_OFF_small, all_specs_ON_small
allSpecsMed = np.empty((2, all_specs_ON_med.shape[0], all_specs_ON_med.shape[1]))
allSpecsMed[0, :, :], allSpecsMed[1, :, :] = all_specs_OFF_med, all_specs_ON_med
allSpecsLarge = np.empty((2, all_specs_ON_large.shape[0], all_specs_ON_large.shape[1]))
allSpecsLarge[0, :, :], allSpecsLarge[1, :, :] = all_specs_OFF_large, all_specs_ON_large

np.save("allSpecsInVivo_SmallOffsets.npy", allSpecsSmall)
np.save("allSpecsInVivo_MediumOffsets.npy", allSpecsMed)
np.save("allSpecsInVivo_LargeOffsets.npy", allSpecsLarge)

np.save("FnoiseInVivo_SmallOffsets.npy", FnoiseSmall)
np.save("PnoiseInVivo_SmallOffsets.npy", PnoiseSmall)
np.save("FnoiseInVivo_MediumOffsets.npy", FnoiseMed)
np.save("PnoiseInVivo_MediumOffsets.npy", PnoiseMed)
np.save("FnoiseInVivo_LargeOffsets.npy", FnoiseLarge)
np.save("PnoiseInVivo_LargeOffsets.npy", PnoiseLarge)



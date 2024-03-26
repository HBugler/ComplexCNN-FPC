#######################################################################################################################
# To Investigate FPC Data
#######################################################################################################################
import numpy as np
from FPC_Functions import loadVivoGTs, toFids, toSpecs, addFShift, addPShift, saveVIVOSpecs
import matplotlib.pyplot as plt

# Import necessary data and set-up variables
InVivoGTs_dir = "C:/Users/Hanna B/Desktop/Research/CodeAndData/ISBI Challenge/ISBI In Vivo Data/track_02_big_gaba_GANNET/"
ppm = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/GTs/ppm_InVivo.npy")
time = np.load(f"C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/InVivo/GTs/time_InVivo.npy")

# concatenate specs
all_specs_ON,  all_specs_OFF = loadVivoGTs()

# convert to fids
all_fids_OFF = toFids(all_specs_OFF)
all_fids_ON = toFids(all_specs_ON)

# add frequency and phase noise
all_fids_ON_FC_small, all_fids_OFF_FC_small, PnoiseSmall = addPShift(all_fids_ON, all_fids_OFF, 20)
all_fids_ON_small, all_fids_OFF_small, FnoiseSmall = addFShift(all_fids_ON_FC_small, all_fids_OFF_FC_small, time, 5)
all_fids_ON_FC_med, all_fids_OFF_FC_med, PnoiseMed = addPShift(all_fids_ON, all_fids_OFF, 45)
all_fids_ON_med, all_fids_OFF_med, FnoiseMed = addFShift(all_fids_ON_FC_med, all_fids_OFF_FC_med, time, 10)
all_fids_ON_FC_large, all_fids_OFF_FC_large, PnoiseLarge = addPShift(all_fids_ON, all_fids_OFF, 90)
all_fids_ON_large, all_fids_OFF_large, FnoiseLarge = addFShift(all_fids_ON_FC_large, all_fids_OFF_FC_large, time, 20)

# convert to specs
all_specs_ON_small, all_specs_OFF_small = toSpecs(all_fids_ON_small), toSpecs(all_fids_OFF_small)
all_specs_ON_FC_small, all_specs_OFF_FC_small = toSpecs(all_fids_ON_FC_small), toSpecs(all_fids_OFF_FC_small)
all_specs_ON_med, all_specs_OFF_med = toSpecs(all_fids_ON_med), toSpecs(all_fids_OFF_med)
all_specs_ON_FC_med, all_specs_OFF_FC_med = toSpecs(all_fids_ON_FC_med), toSpecs(all_fids_OFF_FC_med)
all_specs_ON_large, all_specs_OFF_large = toSpecs(all_fids_ON_large), toSpecs(all_fids_OFF_large)
all_specs_ON_FC_large, all_specs_OFF_FC_large = toSpecs(all_fids_ON_FC_large), toSpecs(all_fids_OFF_FC_large)

# save data (concatenate with ON first)
saveVIVOSpecs(all_specs_ON_small, all_specs_ON_FC_small, all_specs_OFF_small, all_specs_OFF_FC_small, FnoiseSmall, PnoiseSmall, "Small")
saveVIVOSpecs(all_specs_ON_med, all_specs_ON_FC_med, all_specs_OFF_med, all_specs_OFF_FC_med, FnoiseMed, PnoiseMed, "Medium")
saveVIVOSpecs(all_specs_ON_large, all_specs_ON_FC_large, all_specs_OFF_large, all_specs_OFF_FC_large, FnoiseLarge, PnoiseLarge, "Large")
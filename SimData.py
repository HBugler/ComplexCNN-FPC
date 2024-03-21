# FPC Winter 2024
import numpy as np
import math

########################################################################################################################
# Set-up device, hyperparameters and additional variables
########################################################################################################################
snrTypes =["10", "5", "2_5"]
waterTypes = ["Mix", "Pos", "None", "Neg"]

transient_count_dev = 160
transient_count_test = 6
total_gts = 250
num_spec_points = 2048

simFolder= "C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/GTs/"
ppm = np.load(f"{simFolder}ppm_Sim.npy")
time = np.load(f"{simFolder}time_Sim.npy")[:, np.newaxis]
timeDev = np.repeat(time, transient_count_dev*total_gts, axis=1)
timeTest = np.repeat(time, transient_count_test*total_gts, axis=1)

for snr in snrTypes:
    indS = snrTypes.index(snr)
    for water in waterTypes:
        indW = waterTypes.index(water)
        print(f'Preparing Data for {waterTypes[indW]} Water - SNR {snrTypes[indS]}')

        # GTs from .npy
        ON_fid_dev_gt = np.load(f"{simFolder}Development/Water{waterTypes[indW]}/fidsON_{waterTypes[indW]}W_GABAPlus_DevSet.npy")[::-1, :]
        OFF_fid_dev_gt = np.load(f"{simFolder}Development/Water{waterTypes[indW]}/fidsOFF_{waterTypes[indW]}W_GABAPlus_DevSet.npy")[::-1, :]

        ON_fid_test_gt = np.load(f"{simFolder}Test/Water{waterTypes[indW]}/fidsON_{waterTypes[indW]}W_GABAPlus_TestSet.npy")[::-1, :]
        OFF_fid_test_gt = np.load(f"{simFolder}Test/Water{waterTypes[indW]}/fidsOFF_{waterTypes[indW]}W_GABAPlus_TestSet.npy")[::-1, :]

        # repeat GT * number of desired transients and add amplitude noise
        ONfids_dev = np.repeat(ON_fid_dev_gt, transient_count_dev, axis=1)
        OFFfids_dev = np.repeat(OFF_fid_dev_gt, transient_count_dev, axis=1)
        ONfids_test = np.repeat(ON_fid_test_gt, transient_count_test, axis=1)
        OFFfids_test = np.repeat(OFF_fid_test_gt, transient_count_test, axis=1)

        normNoise_dev = [np.random.uniform(2, 2.5, size=1), np.random.uniform(4, 4.5, size=1), np.random.uniform(9.5, 10, size=1)]
        normNoise_test = [np.random.uniform(2, 2.5, size=1), np.random.uniform(4, 4.5, size=1), np.random.uniform(9.5, 10, size=1)]

        ONfids_dev = ONfids_dev + np.random.normal(0, normNoise_dev[indS], size=(num_spec_points, ONfids_dev.shape[1]))
        OFFfids_dev = OFFfids_dev + np.random.normal(0, normNoise_dev[indS], size=(num_spec_points, OFFfids_dev.shape[1]))
        ONfids_test = ONfids_test + np.random.normal(0, normNoise_test[indS], size=(num_spec_points, ONfids_test.shape[1]))
        OFFfids_test = OFFfids_test + np.random.normal(0, normNoise_test[indS], size=(num_spec_points, OFFfids_test.shape[1]))

        # Add frequency and phase noise
        Pnoise_dev = np.random.uniform(low=-90, high=90, size=(2, 1, ONfids_dev.shape[1])).repeat(num_spec_points, axis=1)
        Fnoise_dev = np.random.uniform(low=-20, high=20, size=(2, 1, ONfids_dev.shape[1])).repeat(num_spec_points, axis=1)
        Pnoise_test = np.random.uniform(low=-90, high=90, size=(2, 1, ONfids_test.shape[1])).repeat(num_spec_points, axis=1)
        Fnoise_test = np.random.uniform(low=-20, high=20, size=(2, 1, ONfids_test.shape[1])).repeat(num_spec_points, axis=1)

        ONfids_dev = ONfids_dev * np.exp(-1j * Pnoise_dev[1, :, :] * math.pi / 180) * np.exp(-1j * timeDev * Fnoise_dev[1, :, :] * 2 * math.pi)
        OFFfids_dev = OFFfids_dev * np.exp(-1j * Pnoise_dev[0, :, :] * math.pi / 180) * np.exp(-1j * timeDev * Fnoise_dev[0, :, :] * 2 * math.pi)

        ONfids_test = ONfids_test * np.exp(-1j * Pnoise_test[1, :, :] * math.pi / 180) * np.exp(-1j * timeTest * Fnoise_test[1, :, :] * 2 * math.pi)
        OFFfids_test = OFFfids_test * np.exp(-1j * Pnoise_test[0, :, :] * math.pi / 180) * np.exp(-1j * timeTest * Fnoise_test[0, :, :] * 2 * math.pi)

        # Convert from FIDs to Spectrum
        ONfids_dev = np.fft.fftshift(np.fft.ifft(ONfids_dev, axis=0), axes=0)
        OFFfids_dev = np.fft.fftshift(np.fft.ifft(OFFfids_dev, axis=0), axes=0)

        ONfids_test = np.fft.fftshift(np.fft.ifft(ONfids_test, axis=0), axes=0)
        OFFfids_test = np.fft.fftshift(np.fft.ifft(OFFfids_test, axis=0), axes=0)

        # fig1, ax1 = plt.subplots(1)
        # ax1.plot(ppm, (ONfids_dev[:, :160] - OFFfids_dev[:, :160]).mean(axis=1).real)
        # plt.show()

        # save specs
        allSpecsDev = np.empty((2, ONfids_dev.shape[0], ONfids_dev.shape[1]), dtype=complex)
        allSpecsDev[0, :, :], allSpecsDev[1, :, :] = OFFfids_dev, ONfids_dev
        allSpecsTest = np.empty((2, ONfids_test.shape[0], ONfids_test.shape[1]), dtype=complex)
        allSpecsTest[0, :, :], allSpecsTest[1, :, :] = OFFfids_test, ONfids_test

        np.save(f"TruePhaseLabels_Sim{snrTypes[indS]}_{waterTypes[indW]}_Dev.npy", Pnoise_dev[:,0,:])
        np.save(f"TrueFreqLabels_Sim{snrTypes[indS]}_{waterTypes[indW]}_Dev.npy", Fnoise_dev[:,0,:])
        np.save(f"AllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_Dev.npy", allSpecsDev)

        np.save(f"TruePhaseLabels_Sim{snrTypes[indS]}_{waterTypes[indW]}_Test.npy", Pnoise_test[:,0,:])
        np.save(f"TrueFreqLabels_Sim{snrTypes[indS]}_{waterTypes[indW]}_Test.npy", Fnoise_test[:,0,:])
        np.save(f"AllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_Test.npy", allSpecsTest)








# FPC Winter 2024
import numpy as np
import math

########################################################################################################################
# Set-up device, hyperparameters and additional variables
########################################################################################################################
snrTypes =["2_5"]
waterTypes = ["Mix"] #["Pos", "None", "Neg"]
#do mix with .npy files

transient_count_dev = 160
transient_count_test = 6
total_gts = 250
num_spec_points = 2048

simFolder= "C:/Users/Hanna B/Desktop/FPCFinal2024/SpecsGeneration/Data/Simulated/GTs/"
ppm = np.load(f"{simFolder}ppm_Sim.npy")
time = np.load(f"{simFolder}time_Sim.npy")[:, np.newaxis]
timeDev = np.repeat(time, transient_count_dev*total_gts, axis=1)
timeTest = np.repeat(time, transient_count_test*total_gts, axis=1)

for snr in snrTypes:
    indS = snrTypes.index(snr)
    for water in waterTypes:
        indW = waterTypes.index(water)
        print(f'Preparing Data for {waterTypes[indW]} Water - SNR {snrTypes[indS]}')

        # # GTs from .npy
        # ON_fid_dev_gt = np.loadtxt(f"{simFolder}Development/Water{waterTypes[indW]}/fidsON_{waterTypes[indW]}W_GABAPlus_DevSet.csv", dtype=complex,delimiter=",")
        # OFF_fid_dev_gt = np.loadtxt(f"{simFolder}Development/Water{waterTypes[indW]}/fidsOFF_{waterTypes[indW]}W_GABAPlus_DevSet.csv", dtype=complex,delimiter=",")
        #
        # ON_fid_test_gt = np.loadtxt(f"{simFolder}Test/Water{waterTypes[indW]}/fidsON_{waterTypes[indW]}W_GABAPlus_TestSet.csv", dtype=complex,delimiter=",")
        # OFF_fid_test_gt = np.loadtxt(f"{simFolder}Test/Water{waterTypes[indW]}/fidsOFF_{waterTypes[indW]}W_GABAPlus_TestSet.csv", dtype=complex,delimiter=",")

        # GTs from .npy
        ON_fid_dev_gt = np.load(f"{simFolder}Development/Water{waterTypes[indW]}/fidsON_{waterTypes[indW]}W_GABAPlus_DevSet.npy")
        OFF_fid_dev_gt = np.load(f"{simFolder}Development/Water{waterTypes[indW]}/fidsOFF_{waterTypes[indW]}W_GABAPlus_DevSet.npy")

        ON_fid_test_gt = np.load(f"{simFolder}Test/Water{waterTypes[indW]}/fidsON_{waterTypes[indW]}W_GABAPlus_TestSet.npy")
        OFF_fid_test_gt = np.load(f"{simFolder}Test/Water{waterTypes[indW]}/fidsOFF_{waterTypes[indW]}W_GABAPlus_TestSet.npy")

        # repeat GT * number of desired transients and add amplitude noise
        ONfids_dev = np.repeat(ON_fid_dev_gt, transient_count_dev, axis=1)
        OFFfids_dev = np.repeat(OFF_fid_dev_gt, transient_count_dev, axis=1)
        ONfids_test = np.repeat(ON_fid_test_gt, transient_count_test, axis=1)
        OFFfids_test = np.repeat(OFF_fid_test_gt, transient_count_test, axis=1)

        normNoise_dev = [np.random.uniform(2, 2.5, size=1), np.random.uniform(4, 4.5, size=1), np.random.uniform(9.5, 10, size=1)]
        normNoise_test = [np.random.uniform(2, 2.5, size=1), np.random.uniform(4, 4.5, size=1), np.random.uniform(9.5, 10, size=1)]

        ONfids_dev = (ONfids_dev.real + np.random.normal(0, normNoise_dev[indS], size=(num_spec_points, ONfids_dev.shape[1])))+(ONfids_dev.imag + np.random.normal(0, normNoise_dev[indS], size=(num_spec_points, ONfids_dev.shape[1])))*1j
        OFFfids_dev = (OFFfids_dev.real + np.random.normal(0, normNoise_dev[indS], size=(num_spec_points, OFFfids_dev.shape[1])))+(OFFfids_dev.imag + np.random.normal(0, normNoise_dev[indS], size=(num_spec_points, ONfids_dev.shape[1])))*1j
        ONfids_test = (ONfids_test.real + np.random.normal(0, normNoise_test[indS], size=(num_spec_points, ONfids_test.shape[1])))+(ONfids_test.imag + np.random.normal(0, normNoise_dev[indS], size=(num_spec_points, ONfids_test.shape[1])))*1j
        OFFfids_test = (OFFfids_test.real + np.random.normal(0, normNoise_test[indS], size=(num_spec_points, OFFfids_test.shape[1])))+(OFFfids_test.imag + np.random.normal(0, normNoise_dev[indS], size=(num_spec_points, ONfids_test.shape[1])))*1j

        # Add frequency and phase noise
        Pnoise_dev = np.random.uniform(low=-90, high=90, size=(2, 1, ONfids_dev.shape[1])).repeat(num_spec_points, axis=1)
        Fnoise_dev = np.random.uniform(low=-20, high=20, size=(2, 1, ONfids_dev.shape[1])).repeat(num_spec_points, axis=1)
        Pnoise_test = np.random.uniform(low=-90, high=90, size=(2, 1, ONfids_test.shape[1])).repeat(num_spec_points, axis=1)
        Fnoise_test = np.random.uniform(low=-20, high=20, size=(2, 1, ONfids_test.shape[1])).repeat(num_spec_points, axis=1)

        ONfids_devFC = ONfids_dev * np.exp(-1j * Pnoise_dev[1, :, :] * math.pi / 180)
        ONfids_devNC = ONfids_devFC * np.exp(-1j * timeDev * Fnoise_dev[1, :, :] * 2 * math.pi)

        OFFfids_devFC = OFFfids_dev * np.exp(-1j * Pnoise_dev[0, :, :] * math.pi / 180)
        OFFfids_devNC = OFFfids_devFC * np.exp(-1j * timeDev * Fnoise_dev[0, :, :] * 2 * math.pi)

        ONfids_testFC = ONfids_test * np.exp(-1j * Pnoise_test[1, :, :] * math.pi / 180)
        ONfids_testNC = ONfids_testFC * np.exp(-1j * timeTest * Fnoise_test[1, :, :] * 2 * math.pi)

        OFFfids_testFC = OFFfids_test * np.exp(-1j * Pnoise_test[0, :, :] * math.pi / 180)
        OFFfids_testNC = OFFfids_testFC * np.exp(-1j * timeTest * Fnoise_test[0, :, :] * 2 * math.pi)

        # Convert from FIDs to Spectrum
        ONfidsNC_dev = np.fft.fftshift(np.fft.ifft(ONfids_devNC, axis=0), axes=0)
        OFFfidsNC_dev = np.fft.fftshift(np.fft.ifft(OFFfids_devNC, axis=0), axes=0)

        ONfidsNC_test = np.fft.fftshift(np.fft.ifft(ONfids_testNC, axis=0), axes=0)
        OFFfidsNC_test = np.fft.fftshift(np.fft.ifft(OFFfids_testNC, axis=0), axes=0)

        ONfidsFC_dev = np.fft.fftshift(np.fft.ifft(ONfids_devFC, axis=0), axes=0)
        OFFfidsFC_dev = np.fft.fftshift(np.fft.ifft(OFFfids_devFC, axis=0), axes=0)

        ONfidsFC_test = np.fft.fftshift(np.fft.ifft(ONfids_testFC, axis=0), axes=0)
        OFFfidsFC_test = np.fft.fftshift(np.fft.ifft(OFFfids_testFC, axis=0), axes=0)

        # fig1, ax1 = plt.subplots(1)
        # ax1.plot(ppm, (ONfids_dev[:, :160] - OFFfids_dev[:, :160]).mean(axis=1).real)
        # plt.show()
        print(ONfids_dev[:, 44])

        # save specs
        # allSpecsDev = np.empty((2, ONfids_dev.shape[0], ONfids_dev.shape[1]), dtype=complex)
        # allSpecsDev[0, :, :], allSpecsDev[1, :, :] = OFFfids_dev, ONfids_dev
        # allSpecsTest = np.empty((2, ONfids_test.shape[0], ONfids_test.shape[1]), dtype=complex)
        # allSpecsTest[0, :, :], allSpecsTest[1, :, :] = OFFfids_test, ONfids_test

        np.save(f"TruePhaseLabels_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Dev.npy", Pnoise_dev[:,0,:])
        np.save(f"TrueFreqLabels_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Dev.npy", Fnoise_dev[:,0,:])
        np.save(f"OFF_AllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Dev.npy", OFFfids_devNC[::-1, :])
        np.save(f"ON_AllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Dev.npy", ONfids_devNC[::-1, :])
        np.save(f"OFF_FreqCorrAllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Dev.npy", OFFfids_devFC[::-1, :])
        np.save(f"ON_FreqCorrAllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Dev.npy", ONfids_devFC[::-1, :])
        # np.save(f"AllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_Dev.npy", allSpecsDev)

        np.save(f"TruePhaseLabels_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Test.npy", Pnoise_test[:,0,:])
        np.save(f"TrueFreqLabels_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Test.npy", Fnoise_test[:,0,:])
        np.save(f"OFF_AllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Test.npy", OFFfids_testNC[::-1, :])
        np.save(f"ON_AllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Test.npy", ONfids_testNC[::-1, :])
        np.save(f"OFF_FreqCorrAllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Test.npy", OFFfids_testFC[::-1, :])
        np.save(f"ON_FreqCorrAllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_FNFC_Test.npy", ONfids_testFC[::-1, :])
        # np.save(f"AllSpecs_Sim{snrTypes[indS]}_{waterTypes[indW]}_Test.npy", allSpecsTest)

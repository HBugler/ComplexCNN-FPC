## This script calculates the metrics for the challenge
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks



def calculate_metrics(x,y,ppm):
    # function to calculate all metrics and return dict with results
    # Args:
    #       - x: testing spectra
    #       - y: reference spectra
    #       - ppm: 1D ppm array
    # Output:
    #       - dictionary with metric name and value
    #

    mse = calculate_mse(x,y,ppm)
    snr = calculate_snr(x,ppm)
    # linewidth = calculate_linewidth(x,ppm)
    linewidth = calculate_NewLW(x,ppm)
    shape_score = calculate_shape_score(x,y,ppm)

    output={
        "mse":mse,
        "snr":snr,
        "linewidth":linewidth,
        "shape_score":shape_score
    }

    return output


def calculate_mse(x,y,ppm):
    # Calculate the MSE of a region of interest based on reference
    # Args:
    #       - x: testing spectra
    #       - y: reference spectra
    #       - ppm: 1D ppm reference
    # Output:
    #       - mse
    #

    # selecting region of interest (2.5-4 ppm)
    #
    mses = []
    for i in range(0,ppm.shape[0]):
        i_ppm = ppm[i]
        max_ind = np.amax(np.where(i_ppm >= 2.5))
        min_ind = np.amin(np.where(i_ppm <= 4))

        x_crop = x[i,min_ind:max_ind]
        y_crop = y[i,min_ind:max_ind]

        x_crop_norm = (x_crop-x_crop.min())/(x_crop.max()-x_crop.min())
        y_crop_norm = (y_crop-y_crop.min())/(y_crop.max()-y_crop.min())

        mses.append(np.square(y_crop_norm-x_crop_norm).mean())

    # standard devs (added by HB)
    stds = np.std(mses)

    return sum(mses)/len(mses), stds

def calculate_snr(x,ppm):
    snrs = []

    # looping over scans
    for i in range(x.shape[0]):
        # selecting indexes of regions of interest
        i_ppm = np.ndarray.round(ppm, 2)
        gaba_max_ind, gaba_min_ind = np.amax(np.where(i_ppm >= 2.8)), np.amin(np.where(i_ppm <= 3.2))
        dt_max_ind, dt_min_ind = np.amax(np.where(i_ppm >= 9.8)), np.amin(np.where(i_ppm <= 10.8))

        # selecting scan and extracting region peak
        spec = x[i]
        max_peak = spec[gaba_min_ind:gaba_max_ind].max()

        # calculating fitted standard deviation of noise region
        dt = np.polyfit(i_ppm[dt_min_ind:dt_max_ind], spec[dt_min_ind:dt_max_ind], 2)
        sizeFreq = i_ppm[dt_min_ind:dt_max_ind].shape[0]
        stdev_Man = np.sqrt(np.sum(np.square(np.real(spec[dt_min_ind:dt_max_ind] - np.polyval(dt, i_ppm[dt_min_ind:dt_max_ind])))) / (sizeFreq - 1))

        # calculating snr as peak/(2*stds)
        snrs += [np.real(max_peak) / (2 * stdev_Man)]
    stds = np.std(snrs)
    # return average of snrs
    return snrs, sum(snrs) / len(snrs), stds


def calculate_linewidth(x,ppm):
    # Calculate the GABA SNR
    # Args:
    #       - x: testing spectra
    #       - ppm: 1D ppm reference
    # Output:
    #       - GABA SNR
    #

    linewidths = []

    for i in range(x.shape[0]):
        i_ppm = np.ndarray.round(ppm, 2)
        gaba_max_ind, gaba_min_ind = np.amax(np.where(i_ppm >= 2.8)), np.amin(np.where(i_ppm <= 3.2))

        spec = x[i, gaba_min_ind:gaba_max_ind]
        # print(spec.shape)
        ##normalizing spec
        spec = (spec - spec.min()) / (spec.max() - spec.min())

        max_peak = spec.max()
        ind_max_peak = np.argmax(spec)
        left_side = spec[:ind_max_peak]
        left_ind = np.amin(np.where(left_side > max_peak / 2)) + gaba_min_ind

        right_side = spec[ind_max_peak:]
        right_ind = np.amax(np.where(right_side > max_peak / 2)) + gaba_min_ind + ind_max_peak

        left_ppm = i_ppm[left_ind]
        right_ppm = i_ppm[right_ind]

        linewidths.append(left_ppm - right_ppm)
    stds = np.std(linewidths)

    return linewidths, sum(linewidths) / len(linewidths), stds


def calculate_shape_score(x,y,ppm):
    gaba_corrs=[]
    glx_corrs=[]


    for i in range(0,x.shape[0]):
        i_ppm = ppm[i]

        gaba_max_ind, gaba_min_ind = np.amax(np.where(i_ppm >= 2.8)), np.amin(np.where(i_ppm <= 3.2))
        glx_max_ind, glx_min_ind = np.amax(np.where(i_ppm >= 3.6)), np.amin(np.where(i_ppm <= 3.9))

        gaba_spec_x = x[i,gaba_min_ind:gaba_max_ind]
        gaba_spec_x = (gaba_spec_x-gaba_spec_x.min())/(gaba_spec_x.max()-gaba_spec_x.min())

        gaba_spec_y = y[i,gaba_min_ind:gaba_max_ind]
        gaba_spec_y = (gaba_spec_y-gaba_spec_y.min())/(gaba_spec_y.max()-gaba_spec_y.min())

        gaba_corrs.append(np.corrcoef(gaba_spec_x,gaba_spec_y)[0,1]*0.6)
        # gaba_corrs.append(np.corrcoef(gaba_spec_x,gaba_spec_y)[0,1])

        glx_spec_x = x[i,glx_min_ind:glx_max_ind]
        glx_spec_x = (glx_spec_x-glx_spec_x.min())/(glx_spec_x.max()-glx_spec_x.min())

        glx_spec_y = y[i,glx_min_ind:glx_max_ind]
        glx_spec_y = (glx_spec_y-glx_spec_y.min())/(glx_spec_y.max()-glx_spec_y.min())

        glx_corrs.append(np.corrcoef(glx_spec_x,glx_spec_y)[0,1]*0.4)
        # glx_corrs.append(np.corrcoef(glx_spec_x,glx_spec_y)[0,1])

    gaba_score = sum(gaba_corrs)/len(gaba_corrs)
    glx_score = sum(glx_corrs)/len(glx_corrs)

    # standard devs (added by HB)
    all_scores = gaba_corrs + glx_corrs
    stds = np.std(all_scores)

    # return gaba_score*0.6 + glx_score*0.4
    return gaba_score + glx_score, stds

def calculate_NewLW(x, ppm):        # HB's implementation of the Linewidth code (not from GitHub)
    linewidths=[]

    # for i in range(0, x.shape[0]):
    for i in range(0, x.shape[0]):
        i_ppm = np.ndarray.round(ppm, 2)
        failed_bound, inc = False, 0
        leftFlag, rightFlag = False, False

        while True: # increase bounds if needed
            metab_indClose, metab_indFar = np.amin(np.where(i_ppm >= 2.60-inc)), np.amax(np.where(i_ppm <= 3.40+inc)) # was 2.8 to 3.2
            mean_specs = np.real(x[i,:])
            ppm_temp = i_ppm

            MAXy = np.amax(mean_specs[metab_indClose:metab_indFar], axis=0)         # max y value
            MINy = np.amin(mean_specs[metab_indClose:metab_indFar], axis=0)
            Metab_x = mean_specs[metab_indClose:metab_indFar]                       # specs in region of interest
            ppm_temp1 = ppm_temp[metab_indClose:metab_indFar]                       # ppm in region of interest
            MetabMAX_x = np.array(np.argmax(Metab_x, axis=0))                       # index of max y value
            RPB_ppm, LPB_ppm = ppm_temp1[:MetabMAX_x], ppm_temp1[MetabMAX_x:]       # ppm for left and right of max y value
            RPB_x, LPB_x = Metab_x[:MetabMAX_x], Metab_x[MetabMAX_x:]               # specs for left and right of max y value
            if RPB_x.size and LPB_x.size:
                break
            else:
                inc=inc+0.1

        try:    # Find where curve is greater and lesser than FWHM
            LHM_Greater = np.where((LPB_x) > (((MAXy - MINy) / 2) + MINy))[0]
            RHM_Greater = np.where((RPB_x) > (((MAXy - MINy) / 2) + MINy))[0]
            LHM_Lesser = np.where((LPB_x) < (((MAXy - MINy) / 2) + MINy))[0]
            RHM_Lesser = np.where((RPB_x) < (((MAXy - MINy) / 2) + MINy))[0]

        except ValueError:
            failed_bound = True
            pass

        # Find most probable FWHM given options (make sure not just noisy point)
        for l in range(len(LPB_x), -1, -1):
            if (l in LHM_Greater) and ((l+1) in LHM_Lesser) and ((l+2) in LHM_Lesser) and ((l+3) in LHM_Lesser) and ((l+4) in LHM_Lesser) and ((l+5) in LHM_Lesser):    # and ((l+4) in LHM_Lesser) and ((l+5) in LHM_Lesser)
                left_ind = l
                leftFlag = True

        if (leftFlag==False):
            # print(f"scan {i} took left default value")
            left_ind = np.argmin(np.where((LPB_x) > (((MAXy - MINy) / 2) + MINy))[0])

        for r in range(0, len(RPB_x)):
            if (r in RHM_Greater) and ((r-1) in RHM_Lesser) and ((r-2) in RHM_Lesser) and ((r-3) in RHM_Lesser) and ((r-4) in RHM_Lesser) and ((r-5) in RHM_Lesser):    # and ((r-4) in RHM_Lesser) and ((r-5) in RHM_Lesser)
                right_ind = r
                rightFlag = True

        if (rightFlag==False):
            right_ind = np.argmin(np.where((RPB_x) > (((MAXy - MINy) / 2) + MINy))[0])


        if failed_bound==False:
            left_ppm = LPB_ppm[left_ind]
            right_ppm = RPB_ppm[right_ind]

            linewidths.append(left_ppm - right_ppm)

        # fig1, ax1 = plt.subplots(1)
        # ax1.plot(ppm_temp1, Metab_x)
        # ax1.plot(ppm_temp1[MetabMAX_x], Metab_x[MetabMAX_x], marker='*', color='red')
        # ax1.plot(ppm_temp1[MetabMAX_x], (((Metab_x[MetabMAX_x] - MINy) / 2) + MINy), marker='*', color='orange')
        # ax1.plot(RPB_ppm, RPB_x, color='green')
        # ax1.plot(LPB_ppm, LPB_x, color='blue')
        # ax1.plot(LPB_ppm[LHM_Greater], np.repeat((((Metab_x[MetabMAX_x] - MINy) / 2) + MINy), LHM_Greater.shape[0]),
        #          marker='o', color='black')
        # ax1.plot(RPB_ppm[RHM_Greater], np.repeat((((Metab_x[MetabMAX_x] - MINy) / 2) + MINy), RHM_Greater.shape[0]),
        #          marker='o', color='black')
        # ax1.plot(left_ppm, (((Metab_x[MetabMAX_x] - MINy) / 2) + MINy), marker='x', color='red')
        # ax1.plot(right_ppm, (((Metab_x[MetabMAX_x] - MINy) / 2) + MINy), marker='x', color='red')
        # ax1.invert_xaxis()
        # plt.show()

    # standard devs (added by HB)
    stds = np.std(linewidths)

    return linewidths, sum(linewidths)/len(linewidths), stds


def calculate_ModelledLW(x, ppm):        # HB's implementation of the Linewidth code (not from GitHub)
    linewidths=[]

    # for i in range(0, x.shape[0]):
    for i in range(0, x.shape[0]):
        i_ppm = np.ndarray.round(ppm, 2)
        mean_specs = np.real(x[i, :])
        loop, decE, decO = 0, 0, 0

        # large window to narrow down peak location
        while True:
            metab_indCloseGen, metab_indFarGen = np.amin(np.where(i_ppm >= 2.50+decE)), np.amax(np.where(i_ppm <= 3.50-decO)) # was 2.8 to 3.2
            Metab_xGen = mean_specs[metab_indCloseGen:metab_indFarGen]
            ppm_xGen = i_ppm[metab_indCloseGen:metab_indFarGen]

            # fit 4th degree to determine small window location
            poly4thGenCoeffs = np.polyfit(ppm_xGen, Metab_xGen, 4)
            poly4thGenFit = np.polyval(poly4thGenCoeffs, ppm_xGen)

            MinsGen = find_peaks(-poly4thGenFit)                             # maxima of the negative signal

            if MinsGen[0].shape[0]>1:
                MinGen_ppm = ppm_xGen[MinsGen[0][0]] + 0.1
                MaxGen_ppm = ppm_xGen[MinsGen[0][1]] - 0.1
                if MinGen_ppm > MaxGen_ppm:
                    temp = MaxGen_ppm
                    MaxGen_ppm = MinGen_ppm
                    MinGen_ppm = temp
                break

            if loop%2==0:
                decE = decE + 0.05
            else:
                decO = decO + 0.05

            loop = loop + 1

        # smaller window to model peak shape
        metab_indClose, metab_indFar = np.amin(np.where(i_ppm >= MinGen_ppm)), np.amax(np.where(i_ppm <= MaxGen_ppm)) # was 2.8 to 3.2
        Metab_x = mean_specs[metab_indClose:metab_indFar]                # specs in region of interest
        ppm_x = i_ppm[metab_indClose:metab_indFar]
        print(ppm_x.shape)
        print(Metab_x.shape)
        poly2ndCoeffs = np.polyfit(ppm_x, Metab_x, 2)
        poly2ndFit = np.polyval(poly2ndCoeffs, ppm_x)

        MAXy = np.amax(poly2ndFit, axis=0)         # max y value
        MINy = np.amin(poly2ndFit, axis=0)         # min y value
        MetabMAX_x = np.array(np.argmax(poly2ndFit, axis=0))                            # index of max y value
        Lower_ppm, higher_ppm = ppm_x[:MetabMAX_x], ppm_x[MetabMAX_x:]                  # ppm for left and right of max y value
        Lower_specs, higher_specs = poly2ndFit[:MetabMAX_x], poly2ndFit[MetabMAX_x:]    # specs for left and right of max y value

        LowerBound = np.where((Lower_specs) > (((MAXy + MINy) / 2)))[0]
        HigherBound = np.where((higher_specs) < (((MAXy + MINy) / 2)))[0]

        low_ppm = Lower_ppm[LowerBound[0]]
        high_ppm = higher_ppm[HigherBound[0]]
        linewidths.append(high_ppm - low_ppm)

        # fig1, ax1 = plt.subplots(1)
        # ax1.plot(ppm_xGen, Metab_xGen, 'black')
        # ax1.plot(ppm_xGen, poly4thGenFit, 'purple')
        # ax1.plot(ppm_x, poly2ndFit, 'green')
        # ax1.plot(low_ppm, (((poly2ndFit[MetabMAX_x] - MINy) / 2) + MINy), marker='x', color='red')
        # ax1.plot(high_ppm, (((poly2ndFit[MetabMAX_x] - MINy) / 2) + MINy), marker='x', color='red')
        # ax1.invert_xaxis()
        # plt.show()

    stds = np.std(linewidths)

    return linewidths, sum(linewidths)/len(linewidths), stds
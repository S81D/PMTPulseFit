# Author: Steven Doran
#
# This code gathers raw ADC traces and their corresponding hits information to collect a sample
# of SPE pulses. These SPE pulses are then fit with a 9 parameter function taken from Daya Bay
# to properly describe the PMT response seen in data. These fits are then used by the 
# PMTWaveformSim tool to reconstruct MC pulses using the WCSim true photon information.
#
####################################################################################################
# Contents of the plots + files produced for this code                                             #
# ----------------------------------------------------                                             #
#                                                                                                  #
# These two plot directories are within 'plot_dir' specified below                                 #
# 'avg_waveform/' = contains plots showing the average waveform vs all the rest for each PMT type  #
# 'fit_results/'  = contains plots showing the 9 parameter fit to the data                         #
#                                                                                                  #
# 'fit_csv/'      = contains fit parameters + uncertainties + chi_sq + matrix elements             #
####################################################################################################

print('\nLoading packages, offsets, and geometry...\n')
import os             
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
import pandas as pd
from scipy.optimize import curve_fit
from scipy.linalg import cholesky     # cholesky decomposition
import csv
from scipy.interpolate import interp1d
font = {'family' : 'serif', 'size' : 10 }
mpl.rc('font', **font)
mpl.rcParams['mathtext.fontset'] = 'cm' # Set the math font to Computer Modern
mpl.rcParams['legend.fontsize'] = 1

# fitting functions, taken from Daya Bay

# ---- Lognormal function (Main peak) ----
def lognorm(x, p0, p1, p2):
    return p0 * np.exp(-0.5 * (np.log(x / p1) / p2) ** 2)

# ---- Lognormal with reflections ----
def lognorm_with_reflections(x, p0, p1, p2, T1, T2, T3, r1, r2, r3, N):
    total = lognorm(x, p0, p1, p2)
    Ts = [T1, T2, T3]; Rs = [r1, r2, r3]
    for i in range(1, N + 1):  # add reflection peaks
        reflection = lognorm(x - (i * Ts[i-1]), p0, p1, p2)
        total += Rs[i-1] * reflection
    return total

# --------------------------------------------------------------- #

# you can populate these  with more than 1 file
file_names = ['laser_4695_gains_integration.ntuple.root']     # BeamClusterAnalysis ntuple
raw_file_name = ['Laser_data_GI_PrintADCDataWaves.root']      # name of the ADC waveform file

plot_dir = 'plots/'

# --------------------------------------------------------------- #

# (if not already present) make the needed plot + file directories
os.system('mkdir -p plots/')
os.system('mkdir -p fit_csv/')
os.system('mkdir -p plots/avg_waveform/')
os.system('mkdir -p plots/fit_results/')

# Since we are comparing the hit times to raw waveform data,
# we have to "undo" the timing offsets applied to the hit times.
# We also need the PMT types for fitting.

df1 = pd.read_csv('lib/TankPMTTimingOffsets.csv')
offset_dict = dict(zip(df1['Channel'], df1['offset_value(ns)']))
all_channels = list(range(332, 464))
offsets = [offset_dict.get(ch, 0) for ch in all_channels]

df2 = pd.read_csv('lib/FullTankPMTGeometry.csv')
pmt_type = []; chankey = []
for i in range(len(df2['PMT_type'])):
    pmt_type.append(df2['PMT_type'][i])
    chankey.append(df2['channel_num'][i])

# ----------------------------------------------------- #

# initial data arrays
hitID = []; hitPE = []; hitT = []; eventTimeTank = []

print('---------------------------------------------------------')
print('Extracting information from BeamClusterAnalysis ntuple...\n')
print('There is: ', len(file_names), ' file(s)\n')

counter = 0
for file_name in file_names:
    with uproot.open(file_name) as file:
        
        print('\nRun: ', file_name, '(', (counter), '/', len(file_names), ')')
        print('------------------------------------------------------------')
    
        Event = file["Event"]
        PFN = Event["partFileNumber"].array()
        ETT = Event["eventTimeTank"].array()
        HPE1 = Event['hitPE'].array()
        HT1 = Event['hitT'].array()
        HID1 = Event['hitDetID'].array()

        for i in trange(len(ETT)):

            if PFN[i] == 11 or PFN[i] == 12 or PFN[i] == 13:     # customize accordingly to reflect the Raw ProcessedData part files used

                hitID.append(HID1[i])
                hitPE.append(HPE1[i])
                hitT.append(HT1[i])
                eventTimeTank.append(ETT[i])
            
                counter += 1

                
print(counter, ' total AmBe events loaded from the BeamClusterAnalysis ntuples\n\n')

# we now have a list of hit charges, their event times, and their hit IDs
# let's find some single PE hits

# It was noticed that all of the "effective" SPE hits (after hit finding/thresholding)
# for certain PMTs are not centered at 1 for the laser data, despite the occupancy (~2%) suggesting
# PMTs are only seeing a single photon at a time. These SPE distributions vary on the PMT type.
# This is telling me the extracted SPE factors from the calibration campaign are not great for many of these PMTs.
# We can therefore model our MC waveform simulator to take true photon charges centered at 1, and construct
# data-like pulses that will integrate to the "effective" SPE charges we see in the data.
# We choose bounds for the hit charges that capture the gaussian SPE peaks seen in the laser data, for each PMT type.

event_times = []; hit_charges = []; hit_times = []; hit_ids = []; hit_pmt_types = []
for i in trange(len(hitPE), desc = 'Filtering SPE hits from events...'):
    for j in range(len(hitPE[i])):
        indy = hitID[i][j] - 332
        pmtT = pmt_type[indy]
        t1 = offsets[indy]
        if pmtT == 'Hamamatsu' or pmtT == 'Watchboy' or pmtT == 'LUX':
            if 2 > hitPE[i][j] > 0.5 and 456 > hitT[i][j] > 451:   # within laser window (no TOF correction for now)
                event_times.append(eventTimeTank[i])
                hit_charges.append(hitPE[i][j]); 
                hit_times.append(hitT[i][j] + t1);     # remove offsets as the Raw Waveforms will not include any timing offset
                hit_ids.append(hitID[i][j])
                hit_pmt_types.append(pmtT)
        if pmtT == 'Watchman' or pmtT == 'ETEL':
            if 2.25 > hitPE[i][j] > 0.5 and 456 > hitT[i][j] > 451:   # laser window
                t1 = offsets[indy]
                event_times.append(eventTimeTank[i])
                hit_charges.append(hitPE[i][j]); 
                hit_times.append(hitT[i][j] + t1); 
                hit_ids.append(hitID[i][j])
                hit_pmt_types.append(pmtT)
        
print('\nFound', len(event_times), ' suitable pulses\n\n')

# ----------------------------------------------------- #

print('------------------------------------------------------')
print('Extracting information from Raw Waveform root files...\n')

channel_id = [str(i) for i in range(332,463+1)]
datas = []; timez = []; chargez = []; idz = []

for file in range(len(raw_file_name)):
    
    print('\n######## ', raw_file_name[file], ' ##########\n')
    
    root = uproot.open(raw_file_name[file])

    for i in trange(len(channel_id)):         # loop through PMT root files (by channel)

        # grab names of the TGraphs
        try:
            keys = root[channel_id[i]].keys()
        except:
            print(str(channel_id[i]) + ' key not found!')
            continue

        for k in range(len(event_times)):    # loop over suitable pulses

            if hit_ids[k] == int(channel_id[i]):

                for j in range(len(keys)):   # loop over all TGraphs

                    timestamp = keys[j].split('_')[3].split('.')[0]

                    TS = np.abs(int(timestamp) - int(event_times[k]))

                    if TS < 100:    # event timestamps are within 100ns of trigger

                        # Grab the Graph x and y values
                        TGraph = root[channel_id[i]][keys[j]]
                        data = TGraph.values(axis = 'both')

                        datas.append(data)
                        timez.append(hit_times[k])
                        chargez.append(hit_charges[k])
                        idz.append(hit_ids[k])
                    

print('\nMatched waveforms found: ' + str(len(idz)), '\n\n')

# ----------------------------------------------------- #

PMTTYPES = ['Hamamatsu', 'Watchboy', 'LUX', 'ETEL', 'Watchman']

for PT in PMTTYPES:

    print('\n**********************************************************************')

    # ----------------------------------------------------- #
    # First part is plotting the average waveform vs the rest
    # ----------------------------------------------------- #

    all_pulses_Data = []

    # time grid to for the waveforms (needs to be in ns, WRT registered hit time of the pulse)
    time_grid = np.linspace(0, 120, 1000)

    for k in trange(len(idz), desc = 'extracting SPE pulses for ' + PT + '...'):
        index = k
        pulse = [[], []]
        baseline = np.average(datas[index][1])    # simple average of all the data points in the PMT waveform
        indy = idz[k] - 332

        if pmt_type[indy] == PT:

            for i in range(len(datas[index][0])):
                if timez[index]/2 - 15 < datas[index][0][i] < timez[index]/2 + 55:  # larger window, can adjust if needed
                    pulse[0].append((datas[index][0][i] - timez[index]/2 + 5)*2)    # convert to ns
                    pulse[1].append(datas[index][1][i] - baseline)                  # subtract off baseline value

            # some data pulses have extreme shapes, yet are still within the allowable charge range
            # these pulses may not actually make it into the clusters, and they only account for < 1% of the pulses
            # enable to exclude them (I left them in for the analysis)
            #if max(pulse[1]) > 70 or min(pulse[1]) < -10:
            #   continue

            # interpolate the waveform
            if len(pulse[0]) > 1:
                interp_Data = interp1d(pulse[0], pulse[1], bounds_error=False, fill_value=0)
                all_pulses_Data.append(interp_Data(time_grid))

    # calculate average waveform + errors
    avg_pulse_Data = np.mean(all_pulses_Data, axis=0) if all_pulses_Data else np.zeros_like(time_grid)
    std_errors = np.std(all_pulses_Data, axis=0) / np.sqrt(len(all_pulses_Data))

    # plot and save the average + all the waveforms 
    plt.figure(figsize=(5, 4))
    for i in range(len(all_pulses_Data)):
        if i == 0:
            plt.plot(time_grid, all_pulses_Data[i], color='grey', linewidth=0.2, label = 'Raw Waveforms')
        else:
            plt.plot(time_grid, all_pulses_Data[i], color='grey', linewidth=0.2)
    plt.text(0.75, 0.65, f'N = {len(all_pulses_Data)}', fontsize=10, transform=plt.gca().transAxes)
    plt.plot(time_grid, avg_pulse_Data, color = 'red', linewidth = 1.5, label = 'Average Waveform')
    plt.xlabel('time [ns]', fontsize = 12)
    plt.ylabel('ADC', fontsize = 12)
    plt.xlim([0,80]); plt.ylim([-10,50])
    plt.title('SPE waveforms | ' + PT + ' PMTs', fontsize = 12)
    plt.legend(fontsize = 12, frameon = False, loc = 'upper right')
    plt.savefig(plot_dir + '/avg_waveform/' + PT + ' averaged SPE pulses.png',
                dpi=300, bbox_inches='tight', pad_inches=.3, facecolor='w')
    plt.close()

    print('\n' + PT + ' average waveform plot saved: ' + plot_dir + '/avg_waveform/' + PT + ' averaged SPE pulses.png\n')


    # ----------------------------------------------------- #
    # Next, we do the fitting
    # ----------------------------------------------------- #

    print('\nFitting ' + PT + ' PMTs...\n')

    # 9 parameters to fit for 3 reflection peaks (initial guesses below)

    p0_guess = 18      # Main peak amplitude
    p1_guess = 62      # Main peak location [ns]
    p2_guess = 0.04    # Main peak shape
    T1_guess = 8       # Reflection #1 time spacing, relative to position of the main peak
    T2_guess = 8       # Reflection #2 time spacing, relative to 1st reflection peak
    T3_guess = 8       # Reflection #3 time spacing, relative to 2nd reflection peak
    r1_guess = 0.25    # Reflection #1 amplitude, with respect to main peak (reflection decay factor)
    r2_guess = 0.18    # Reflection #2 amplitude, with respect to main peak
    r3_guess = 0.14    # Reflection #3 amplitude, with respect to main peak
    N_reflections = 3  # Number of reflections (note you must add 2 additional parameters to the fitting for every new reflection you add)

    # parameter bounds - we know general location of the reflection peaks (~8ns) and we know the reflection amps will not be greater than the main peak
    param_bounds = (
        [10, 55, 0.02, 5, 5, 5, 0, 0, 0],    # Lower bounds (p0, p1, p2, T1, T2, T3, r1, r2, r3)
        [30, 65, 0.06, 11, 11, 11, 1, 1, 1]  # Upper bounds (p0, p1, p2, T1, T2, T3, r1, r2, r3)
    )

    # move all x points away from 0 to avoid NaNs for the lognorm fit. We will introduce a T0 offset in the PMTWaveformSim tool to account for this
    time_grid = time_grid + 50

    # truncate time_grid to only fit relevant bits of the waveform
    mask = time_grid < 95
    time_grid_truncated = time_grid[mask]
    avg_pulse_Data_truncated = avg_pulse_Data[mask]
    std_errors_truncated = std_errors[mask]

    # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Perform the fitting
    popt, pcov = curve_fit(
        lambda x, p0, p1, p2, T1, T2, T3, r1, r2, r3: lognorm_with_reflections(x, p0, p1, p2, T1, T2, T3, r1, r2, r3, N_reflections),
        time_grid_truncated, avg_pulse_Data_truncated,
        p0=[p0_guess, p1_guess, p2_guess, T1_guess, T2_guess, T3_guess, r1_guess, r2_guess, r3_guess],
        bounds=param_bounds, sigma=std_errors_truncated
    )

    # Extract fitted parameters
    p0_fit, p1_fit, p2_fit, T1_fit, T2_fit, T3_fit, r1_fit, r2_fit, r3_fit = popt
    print(f"Fitted main peak parameters: p0={p0_fit:.3f}, p1={p1_fit:.3f}, p2={p2_fit:.3f}")
    print(f"Fitted reflection spacings: T1={T1_fit:.3f}, T2={T2_fit:.3f}, T3={T3_fit:.3f}")
    print(f"Fitted reflection coefficients: r1={r1_fit:.3f}, r2={r2_fit:.3f}, r3={r3_fit:.3f}")
    # Print the covariance matrix
    print("\nCovariance Matrix:")
    print(pcov)

    # Cholesky decomposition
    chol_factor = cholesky(pcov, lower=True)   # we want lower triangular form
    print('\nCholesky Decomposition (lower triangular matrix):')
    print(chol_factor)
    print("\nCholesky Decomposition Matrix Elements (Lower Triangular):")
    for i in range(chol_factor.shape[0]):
        for j in range(i + 1):
            print(f"U[{i},{j}] = {chol_factor[i, j]:.6f}")

    # We can also calculate the standard errors (uncertainties) of the parameters:
    param_errors = np.sqrt(np.diag(pcov))  # Standard deviation (errors) for each parameter
    print("\nParameter uncertainties (standard errors):")
    print(param_errors)

    # Generate fitted curve
    fitted_curve = lognorm_with_reflections(time_grid, p0_fit, p1_fit, p2_fit, T1_fit, T2_fit, T3_fit, r1_fit, r2_fit, r3_fit, N_reflections)

    # Compute chi-squared of the fit
    residuals = avg_pulse_Data_truncated - lognorm_with_reflections(time_grid_truncated, *popt, N_reflections)
    chi_sq = np.sum((residuals / std_errors_truncated) ** 2)
    ndf = len(time_grid_truncated) - len(popt)
    chi_sq_ndf = chi_sq / ndf
    print('chisq / dof:', chi_sq_ndf)

    # Plot results (zoomed into the waveform area of interest - no overshoot fitting so no need to display that portion of the waveform)
    plt.figure(figsize=(6, 4))
    plt.scatter(time_grid, avg_pulse_Data, label="Data", color='black', s=4)    # data
    plt.plot(time_grid, lognorm(time_grid, p0_fit, p1_fit, p2_fit), label="Main peak fit", color="red")   # main peak
    colors = ['teal', 'dodgerblue', 'orange']
    Tfits = [T1_fit, T2_fit, T3_fit]; Rfits = [r1_fit, r2_fit, r3_fit]
    for i in range(1, N_reflections+1):  # Plot first 3 reflections separately
        reflection_curve = (Rfits[i-1]) * lognorm(time_grid - i * Tfits[i-1], p0_fit, p1_fit, p2_fit)  # r1, r2, r3 multiplied by lognorm
        plt.plot(time_grid, reflection_curve, label=f"Reflection {i}", linestyle="dashed", color = colors[i-1])
    plt.plot(time_grid, fitted_curve, label=f"Total fit (χ²/ndf = {round(chi_sq)}/{ndf})", color='lime', linewidth = 2)    # total fit
    plt.xlabel('nanoseconds', loc = 'right')
    plt.ylabel('ADC')
    plt.title('SPE waveform modeling | ' + PT + ' PMTs')
    plt.legend(fontsize = 11, frameon = False)
    plt.xlim([50,95])
    plt.savefig(plot_dir + '/fit_results/' + PT + ' 9 parameter fit to main peak and reflections.png',
                dpi=300, bbox_inches='tight', pad_inches=.3, facecolor='w')
    plt.close()

    print('\n' + PT + 'fit result saved: ' + plot_dir +'/fit_results/' + PT + ' 9 parameter fit to main peak and reflections.png\n')


    # -------------------------------------------- #
    # finally, we can export the fits to a .csv file
    
    header = ["p0", "p1", "p2", "T1", "T2", "T3", "r1", "r2", "r3"]
    header += [f"U{i}{j}" for i in range(chol_factor.shape[0]) for j in range(i + 1)]
    header += [f"uncertainty_{param}" for param in header[:9]]  # Param uncertainties
    header += [f"chisq_ndf"]

    row_data = list(popt)  # Fitted parameters
    row_data += [chol_factor[i, j] for i in range(chol_factor.shape[0]) for j in range(i + 1)]
    row_data += list(param_errors)  # Parameter uncertainties
    row_data += chi_sq_ndf   # chi_sq / dof

    csv_filename = "fit_csv/fit_results_" + PT + ".csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(row_data)
        
    print(f"\nFit results saved to {csv_filename}")

# ---------------------------------------------------------------------------------------------------------- #

# We can stich together each csv file to construct the properly need Lognorm file for the PMTWaveformSim tool

print('\n\n--------------------------------------------------------')
print('\nCombining fit files into a PMTWaveformLognormFit.csv...')

outputfilename = 'fit_csv/PMTWaveformLognormFit.csv'

csv_files = {
    'Hamamatsu': 'fit_csv/fit_results_Hamamatsu.csv',
    'Watchboy': 'fit_csv/fit_results_Watchboy.csv',
    'LUX': 'fit_csv/fit_results_LUX.csv',
    'ETEL': 'fit_csv/fit_results_ETEL.csv',
    'Watchman': 'fit_csv/fit_results_Watchman.csv'
}

vals = ['p0','p1','p2','T1','T2','T3','r1','r2','r3',
        'U00',
        'U10','U11',
        'U20','U21','U22',
        'U30','U31','U32','U33',
        'U40','U41','U42','U43','U44',
        'U50','U51','U52','U53','U54','U55',
        'U60','U61','U62','U63','U64','U65','U66',
        'U70','U71','U72','U73','U74','U75','U76','U77',
        'U80','U81','U82','U83','U84','U85','U86','U87','U88']

fit_data = {}
for pmt_type_, file in csv_files.items():
    df = pd.read_csv(file, usecols=vals)       # Load fit parameters for the current PMT type
    fit_data[pmt_type_] = df.iloc[0].tolist()  # Save the first row of each fit result as the list of parameters

output_data = []

for i in range(len(chankey)):
    pmt = pmt_type[i]
    channel = chankey[i]
    
    # Ensure the PMT type exists in the fit_data dictionary
    if pmt in fit_data:
        entry = [channel]
        entry.extend(fit_data[pmt])
        output_data.append(entry)
    else:
        print(f"Warning: No data found for PMT type {pmt}, channel {channel}. Skipping this entry.")

# Convert the output data to a DataFrame and write it to a CSV file
output_columns = ['PMT'] + vals
with open(outputfilename, 'w', newline='') as f:
    f.write(', '.join(output_columns) + '\n')  # Write header with spaces after commas
    for row in output_data:
        f.write(', '.join(map(str, row)) + '\n')  # Write data rows with spaces after commas


# ---------------------------------------------------------------------------------------------------------- #

print('\n-----------------------------------------------\n\nel fin\n')
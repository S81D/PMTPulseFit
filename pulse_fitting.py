# =============================================================================
# Author: Steven Doran
# Date: Spring 2025
#
# Description:
# This code gathers raw ADC traces and their corresponding hits information to collect a sample
# of SPE pulses. These SPE pulses are then fit with a 7 parameter function taken from Daya Bay
# to properly describe the PMT response seen in data. These fits are then used by the 
# PMTWaveformSim tool to reconstruct MC pulses using the WCSim true photon information.
#
# Output:
# 'plots/avg_waveform/'        = contains plots showing the average waveform vs all the rest for each PMT type
# 'plots/fit_results/'         = contains plots showing the 9 parameter fit to the data
# 'fit_csv/'                   = contains fit parameters + uncertainties + chi_sq + matrix elements
# 'PMTWaveformLognormFit.csv'  = contains a single file with all fit + uncertainties to be used by the PMTWaveformSim tool
# 'examples/'                  = contains some example outputs from the script (fit results, plots, raw waveforms w/ interpolated average, final PMTWaveformSim fit file)
#
# Input required: 
# - ROOT file(s) containing ADC traces (from the PrintADCTraces tool / toolchain: https://github.com/fengyvoid/ToolAnalysis/tree/FreezeForEBV2_v1.0/configfiles/PrintADCTraces)
# - Customize for your dataset: time / charge windows, filenames
#
# Notes:
# - Originally 3 reflections + Cholesky decomposition; reduced to 2 reflections with diagonal errors as some of the PMT traces did not include 3 reflections before the pulse end point,
#   and often times we got large matrix element errors, leading to "blow ups" and very large reflections when generating reconstructed pulses
# - Data files containing the ADC traces were too large for github, you can find them here:
#      * Laser: /pnfs/annie/persistent/users/doran/datasets/PMTPulseFit_ADCTraces/R4695_p0_p100_ADCTraces.root
#      * Beam : /pnfs/annie/persistent/users/doran/datasets/PMTPulseFit_ADCTraces/R5430_ADCTraces.root
# - It was noticed that all of the "effective" SPE hits (after hit finding/thresholding)
#   for certain PMTs are not centered at 1 for the laser data, despite the occupancy (~2%) suggesting
#   PMTs are only seeing a single photon at a time. These SPE distributions vary on the PMT type.
#   This is telling me the extracted SPE factors from the calibration campaign are not great for many of these PMTs.
#   We can therefore model our MC waveform simulator to take true photon charges centered at 1, and construct
#   data-like pulses that will integrate to the "effective" SPE charges we see in the data.
#   We choose bounds for the hit charges that capture the gaussian SPE peaks seen in the laser data, for each PMT type.
# - output from PrintADCTraces root file will have all ADC pulses baseline-subtracted and have the pulse start times start at 0ns
# - We don't expect fits (or plots) from inactive / dead PMT channels; during the script it will skip channels where no waveforms are found
#
# =============================================================================
#
#

print('\nLoading packages...\n')
import os             
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import csv
from collections import defaultdict
from scipy.interpolate import interp1d
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

font = {'family' : 'serif', 'size' : 10 }
mpl.rc('font', **font)
mpl.rcParams['mathtext.fontset'] = 'cm' # Set the math font to Computer Modern
mpl.rcParams['legend.fontsize'] = 1


# ----------------------------------------------------------------------------------------------------------
# Fitting Functions (from Daya Bay: https://www.sciencedirect.com/science/article/abs/pii/S0168900218304133)
# ----------------------------------------------------------------------------------------------------------


# ---- Lognormal function (Main peak) ----
def lognorm(x, p0, p1, p2):
    """
    p0 = main peak amplitude, p1 = main peak width (tau), p2 = peak shape (sigma)
    """
    return p0 * np.exp(-0.5 * (np.log(x / p1) / p2) ** 2)

# ---- Lognormal with reflections ----
def lognorm_with_reflections(x, p0, p1, p2, T1, T2, r1, r2, N):
    """
    2 reflections: r = reflection amplitudes wrt main peak, T = reflection spacing [ns]
    """
    total = lognorm(x, p0, p1, p2)
    Ts = [T1, T2]; Rs = [r1, r2]
    for i in range(1, N + 1):  # add reflection peaks
        reflection = lognorm(x - (i * Ts[i-1]), p0, p1, p2)
        total += Rs[i-1] * reflection
    return total

#
# --------------------------------------------------------------- #
# MODIFY
# --------------------------------------------------------------- #
#

trace_file_name = 'lib/R5430_ADCTraces.root'   # name of the ADC waveform file
which_mode = 'Beam'                            # 'Laser' or 'Beam' -> depending on the run, we have different time and charge windows
verbose = True                                 # detailed verbosity (True = more, False = minimal)
plot_raw = False                                # Save raw, extracted SPE waveforms with the average fit for each PMT channel (in 'plots/avg_waveform/')

# Define time and charge windows  #            (based off of select datasets, just comment out which one you won't use (beam or laser))
# - - - - - - - - - - - - - - - - #
charge_min = 0.75   # p.e.                     For some reason our "SPE" data is not centered at 1.0 --> probably due to bad Gains fits
charge_max = 1.75   # p.e.

# laser (comment if not using)
#time_min = 445    # ns                        TOF-corrected
#time_max = 450    # ns

# beam (comment if not using)
time_min = 300    # ns                         Within the beam spill --> we could add more selection criteria but its fine for now
time_max = 1700   # ns

#
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
#

if verbose:
    print('\n' + which_mode + ' data using file: ' + trace_file_name)
    print('---------------------------------------------------')
    print('charge and time windows:   [' + str(charge_min) + ',' + str(charge_max) + '] p.e.    [' + str(time_min) + ',' + str(time_max) + '] ns\n')

# (if not already present) make the needed plot + file directories
os.system('mkdir -p plots/')
os.system('mkdir -p fit_csv/')
os.system('mkdir -p plots/avg_waveform/')
os.system('mkdir -p plots/fit_results/')

# -----------------------
# Load ADC Traces
# -----------------------

pulse_data = []
channel_id = [str(i) for i in range(332,463+1)]

root = uproot.open(trace_file_name)
for i in trange(len(channel_id), desc = 'Loading ADC Traces from root file: ' + str(trace_file_name) + '...'):   # loop through PMT root files (by channel)

    # grab names of the TGraphs
    try:
        keys = root[channel_id[i]].keys()
    except:
        if verbose:
            print(str(channel_id[i]) + ' key not found! (probably a dead/inactive PMT)')
        continue
        
    for j in range(len(keys)):   # loop over all TGraphs

        # Grab the Graph x and y values (ns, ADC)
        TGraph = root[channel_id[i]][keys[j]]
        data = TGraph.values(axis = 'both')
        
        # get the title, deconstruct it
        title = TGraph.member("fTitle")
        if title is None:
            continue
        try:
            parts = title.split("_")
            hit_time = float(parts[-2])
            hit_charge = float(parts[-1])
        except (IndexError, ValueError):
            print(f"Could not parse title: {title}")
            continue

        # Save as dictionary
        pulse_data.append({
            "channel": int(channel_id[i]),
            "hit_time": hit_time,
            "hit_charge": hit_charge,
            "trace_x": data[0],    # x-axis (ns)
            "trace_y": data[1]     # y-axis (ADC)
        })

print('\n')


# -----------------------
# Extract SPE pulses
# -----------------------

# if laser data, we must correct for TOF from the source position
c = 299792458  # [m/s]
c = c/(4/3)    # refractive index of water

def TOF(hitX, hitY, hitZ):

    # calculate residuals
    d_pos = np.abs(np.sqrt( (hitZ)**2  + \
                                (hitX)**2  +  (hitY)**2 ))
    tri = d_pos/c
    t_res_i = tri*1e9    # in ns
    
    return t_res_i   # light distance from center in [ns]


# Build lookup map for detector geometry and TOF corrections
df2 = pd.read_csv('lib/FullTankPMTGeometry.csv')
channel_info_map = {
    row["channel_num"]: {
        "residual": TOF(row["x_pos"], row["y_pos"] + 0.1446, row["z_pos"] - 1.681),
        "type": row["PMT_type"]
    }
    for _, row in df2.iterrows()
}


# Filter SPE pulse data
filtered_pulses = []

for entry in pulse_data:
    chan = entry["channel"]
    hit_time = entry["hit_time"]
    charge = entry["hit_charge"]

    # Check if channel info exists
    if chan not in channel_info_map:
        continue

    # Grab TOF correction and PMT type
    info = channel_info_map[chan]
    corrected_time = hit_time - info["residual"]

    # Apply selection
    if (charge_min <= charge <= charge_max) and (time_min <= corrected_time <= time_max):
        new_entry = entry.copy()
        new_entry["corrected_time"] = corrected_time
        new_entry["pmt_type"] = info["type"]
        filtered_pulses.append(new_entry)

print('\nSPE pulses extracted\n')

if verbose:
    hit_times_dummy = [entry["corrected_time"] for entry in filtered_pulses]
    hit_charges_dummy = [entry["hit_charge"] for entry in filtered_pulses]

    plt.figure(figsize=(4, 3))
    plt.hist(hit_times_dummy, bins = 30, color = 'navy')
    plt.xlabel('hit times [ns]')
    plt.title('SPE pulse times')
    plt.show()

    plt.figure(figsize=(4, 3))
    plt.hist(hit_charges_dummy, bins = 30, color = 'tomato')
    plt.xlabel('hit charges [p.e.]')
    plt.title('SPE pulse charges')
    plt.show()


# -----------------------------------------------------------------------------------------------------
# Take extracted SPE pulses, clean them up, and average them to get an average SPE waveform per channel
# -----------------------------------------------------------------------------------------------------

# Create dictionary: channel → list of (time, adc) tuples
channel_traces = defaultdict(list)

for entry in filtered_pulses:
    chan = entry["channel"]
    time = entry["trace_x"]
    adc = entry["trace_y"]
    channel_traces[chan].append((time, adc))



# the pulses will all have different "pulse end points"
# we will build an "average" pulse for each PMT with a certain pulse end point determined from the distribution
# because we are interested in just 2 reflections, we select the "1sigma_range" (but you can change this if you want to)

pulse_end_stats = {}

for chan, pulses in channel_traces.items():
    stop_times = [time[-1] for time, _ in pulses]  # last time point of each pulse

    if not stop_times:
        continue

    stop_times = np.sort(stop_times)
    n = len(stop_times)

    median = np.median(stop_times)
    
    # Get central percentiles
    p16, p84 = np.percentile(stop_times, [16, 84])      # 1σ (68%)
    p2_5, p97_5 = np.percentile(stop_times, [2.5, 97.5]) # 2σ (95%)
    p0_15, p99_85 = np.percentile(stop_times, [0.15, 99.85])  # 3σ (99.7%)

    pulse_end_stats[chan] = {
        "median": median,
        "1sigma_range": (p16, p84),
        "2sigma_range": (p2_5, p97_5),
        "3sigma_range": (p0_15, p99_85)
    }
    
cutoff_pulse = {}
if verbose:
    print('Pulse end points\n--------------------------------------')
    print('Channel, Median, 1sigma upper, 2sigma upper, 3sigma upper')
for chan in sorted(pulse_end_stats.keys()):
    stats = pulse_end_stats[chan]
    if verbose:
        print(f"{chan}, {stats['median']:.1f}, {stats['1sigma_range'][1]:.1f}, {stats['2sigma_range'][1]:.1f}, {stats['3sigma_range'][1]:.1f}")
    
    cutoff_pulse[chan] = stats["1sigma_range"][1]      # MODIFY IF NEEDED

print('\n')


# create averaged, interpolated waveforms for each channel with errors

if plot_raw:    # if the user wants the plots, do that first

    for chan in trange(332, 463+1, desc = 'Plotting raw ADC traces + average, interpolated waveform for each PMT ID...'):
        if chan not in channel_traces:
            continue  # skip channels with no data
            
        interpolated_pulses = []
        valid_mask = []   # for computing dynamic mean and errors as you move down the interpolated waveform
            
        mark_pulse = 0
        if chan in cutoff_pulse:
            mark_pulse = cutoff_pulse[chan]
        time_grid = np.linspace(0, int(mark_pulse), 10000)  # time grid with maximum set to the 99th percentile
        
        if mark_pulse == 0:
            if verbose:
                print('\nNO PULSE END CUTOFF FOR TIME_GRID (CHANNEL ' + str(chan) + ')!\n')
            continue
            
        plt.figure(figsize=(6, 4))
        
        count = 0
        for time, adc in channel_traces[chan]:
            
            # pulse cleaning
            skip_event = False
            for j in range(len(time)):
                if time[j] < 10:            # if the pulse has a very early "pre-pulse" (hit finding found an odd shaped pulse)
                    if adc[j] > 5:
                        skip_event = True
                        break
            if skip_event == True:
                continue
            
            interp_func = interp1d(time, adc, kind='cubic', bounds_error=False, fill_value=0)
            interpolated_adc = interp_func(time_grid)
            interpolated_pulses.append(interpolated_adc)

            mask = (time_grid >= time[0]) & (time_grid <= time[-1])
            valid_mask.append(mask)

            if count == 0:
                plt.plot(time, adc, color = 'lightsteelblue', label = 'raw traces', linewidth = 0.5)
            else:
                plt.plot(time, adc, color = 'lightsteelblue', linewidth = 0.5)
            count += 1
            
        
        interpolated_pulses = np.array(interpolated_pulses)
        valid_mask = np.array(valid_mask)
        
        num_valid = np.sum(valid_mask, axis=0)
        avg_pulse = np.sum(interpolated_pulses * valid_mask, axis=0) / np.clip(num_valid, 1, None)

        # Compute std over valid entries only
        residuals = (interpolated_pulses - avg_pulse[None, :])**2
        residuals[~valid_mask] = 0  # ignore out-of-range points
        std_dev = np.sqrt(np.sum(residuals, axis=0) / np.clip(num_valid - 1, 1, None))
        std_error = std_dev / np.sqrt(np.clip(num_valid, 1, None))

        plt.plot(time_grid, avg_pulse, color='blue', label='average trace')
        plt.fill_between(time_grid, avg_pulse - std_error, avg_pulse + std_error,
                        color='blue', alpha=0.5)
        plt.xlabel('time [ns]', fontsize = 12)
        plt.ylabel('ADC', fontsize = 12)
        plt.xlim([0,int(cutoff_pulse[chan])]); plt.ylim([-10,40])
        plt.title('SPE waveforms from beam data | PMT ' + str(chan), fontsize = 12)
        plt.legend(fontsize = 10, frameon = False, loc = 'upper right')
        plt.savefig('plots/avg_waveform/PMT ' + str(chan) + ' average and raw trace _ SPE ' + which_mode + ' pulses.png',
                    dpi=300, bbox_inches='tight', pad_inches=.3, facecolor='w')
        plt.close()

    if verbose:
        print('\nRaw traces + average for each PMT ID saved to "plots/avg_waveform/"\n\n')
    

# Probably can compress this with the section above, but do the same thing without the plotting

# Dictionary to store: chan → (time_grid, avg_pulse, std_error)
channel_avg_pulses = {}

for chan in trange(332, 464, desc = 'Creating averaged, interpolated waveforms with errors for lognorm fitting...'):
    if chan not in channel_traces:
        continue

    if chan not in cutoff_pulse or cutoff_pulse[chan] == 0:
        continue  # Skip if no valid time bound

    mark_pulse = cutoff_pulse[chan]
    time_grid = np.arange(0, int(mark_pulse) + 2, 2)

    interpolated_pulses = []
    valid_mask = []

    for time, adc in channel_traces[chan]:
        
        skip_event = False
        for j in range(len(time)):
            if time[j] < 10:
                if adc[j] > 5:
                    skip_event = True
                    break
        if skip_event == True:
            continue

        interp_func = interp1d(time, adc, bounds_error=False, fill_value=0)
        interpolated_adc = interp_func(time_grid)
        interpolated_pulses.append(interpolated_adc)

        mask = (time_grid >= time[0]) & (time_grid <= time[-1])
        valid_mask.append(mask)

    if not interpolated_pulses:
        continue

    interpolated_pulses = np.array(interpolated_pulses)
    valid_mask = np.array(valid_mask)

    num_valid = np.sum(valid_mask, axis=0)
    avg_pulse = np.sum(interpolated_pulses * valid_mask, axis=0) / np.clip(num_valid, 1, None)

    residuals = (interpolated_pulses - avg_pulse[None, :])**2
    residuals[~valid_mask] = 0
    std_dev = np.sqrt(np.sum(residuals, axis=0) / np.clip(num_valid - 1, 1, None))
    std_error = std_dev / np.sqrt(np.clip(num_valid, 1, None))

    # Store everything for future fitting
    channel_avg_pulses[chan] = {
        "time": time_grid + 50,       # to avoid NaN errors with the lognorm fitting, shift the waveform out +50 ns
        "avg": avg_pulse,
        "err": std_error
    }


print("\nAll average traces with error stored.\n")


if verbose: 
    # plot some sample waveforms for the user
    print('\nPlotting an average waveform for you...')
    chan = 411
    pulse = channel_avg_pulses[chan]

    plt.figure(figsize=(6, 4))
    # pulse["time"], pulse["avg"], and pulse["err"] now ready for fitting
    plt.errorbar(pulse["time"], pulse["avg"], yerr=pulse["err"], linestyle='none', color = 'k',
                marker = 'o', markersize=5)
    plt.xlabel("Time [ns]")
    plt.ylabel("ADC")
    plt.title("PMT " + str(chan) + " averaged pulse with error bars")
    plt.show()

print('\n')


# -------------------------------------------------------------------
# Perform the 7-parameter, lognorm fitting of the interpolated pulses
# -------------------------------------------------------------------

# 7 parameters to fit for 2 reflection peaks (initial guesses below)

p0_guess = 18      # Main peak amplitude
p1_guess = 65      # Main peak location [ns]
p2_guess = 0.04    # Main peak shape
T1_guess = 8       # Reflection #1 time spacing, relative to position of the main peak
T2_guess = 8       # Reflection #2 time spacing, relative to 1st reflection peak
r1_guess = 0.25    # Reflection #1 amplitude, with respect to main peak (reflection decay factor)
r2_guess = 0.18    # Reflection #2 amplitude, with respect to main peak
N_reflections = 2  # Number of reflections (note you must add 2 additional parameters to the fitting for every new reflection you add)
                   # The functions are currently only set up for 2 reflections --> if you want more you will need to modify the script accordingly

# parameter bounds - we know general location of the reflection peaks (~8ns) and we know the reflection amps will not be greater than the main peak
param_bounds = (
    [5, 55, 0.02, 3, 3, 0, 0],    # Lower bounds (p0, p1, p2, T1, T2, r1, r2)
    [30, 75, 0.06, 10, 10, 1, 1]  # Upper bounds (p0, p1, p2, T1, T2, r1, r2)
)

# NOTE: Certain PMT IDs have *terrible* looking pulses (tiny pulse amplitudes, large relative reflections, etc...)
#       You may have to re-run the script or export to a jupyter notebook and do custom fitting (adjust parameter guesses and bounds) to get a good fit



wohoooo = [[], [], [], [], [], [], []]   # main amp, main positon, main shape, r1, r2, t1, t2

for chan in trange(332, 463+1, desc = 'Fitting PMT data...'):
    
    if verbose:
        print('\n\n######### PMT ' + str(chan) + ' #########')
    
    try:
    
        pulse = channel_avg_pulses[chan]
        time_grid = pulse["time"]  # This is your original 2ns-discretized time grid

        # Fit the reflections part of the model with bounds
        popt, pcov = curve_fit(
            lambda x, p0, p1, p2, T1, T2, r1, r2: lognorm_with_reflections(x, p0, p1, p2, T1, T2, r1, r2, N_reflections),
            pulse["time"], pulse["avg"],
            p0=[p0_guess, p1_guess, p2_guess, T1_guess, T2_guess, r1_guess, r2_guess],
            bounds=param_bounds, sigma=pulse["err"]
        )

        # Extract fitted parameters
        p0_fit, p1_fit, p2_fit, T1_fit, T2_fit, r1_fit, r2_fit = popt
        wohoooo[0].append(p0_fit); wohoooo[1].append(p1_fit); wohoooo[2].append(p2_fit)
        wohoooo[3].append(T1_fit); wohoooo[4].append(T2_fit)
        wohoooo[5].append(r1_fit); wohoooo[6].append(r2_fit)
        if verbose:
            print(f"Fitted main peak parameters: p0={p0_fit:.3f}, p1={p1_fit:.3f}, p2={p2_fit:.3f}")
            print(f"Fitted reflection spacings: T1={T1_fit:.3f}, T2={T2_fit:.3f}")
            print(f"Fitted reflection coefficients: r1={r1_fit:.3f}, r2={r2_fit:.3f}")

        # calculate the standard errors (uncertainties) of the parameters:
        param_errors = np.sqrt(np.diag(pcov))  # Standard deviation (errors) for each parameter
        if verbose:
            print("\nParameter uncertainties (standard errors):")
            print(param_errors)

        # Generate fitted curve
        high_res_time_grid = np.linspace(min(time_grid), max(time_grid), 5000)
        fitted_curve = lognorm_with_reflections(high_res_time_grid, p0_fit, p1_fit, p2_fit, T1_fit, T2_fit, r1_fit, r2_fit, N_reflections)

        # Compute chi-squared
        residuals = pulse["avg"] - lognorm_with_reflections(time_grid, *popt, N_reflections)
        chi_sq = np.sum((residuals / pulse["err"]) ** 2)
        ndf = len(pulse["time"]) - len(popt)  # Number of degrees of freedom
        chi_sq_ndf = chi_sq / ndf
        if verbose:
            print('chisq / dof:', chi_sq_ndf)
        
        
        # Plot fit results
        plt.figure(figsize=(6, 4))
        plt.scatter(pulse["time"], pulse["avg"], label="Data", color='black', s=4)    # data
        plt.plot(high_res_time_grid, lognorm(high_res_time_grid, p0_fit, p1_fit, p2_fit), label="Main peak fit", color="red")   # main peak
        colors = ['teal', 'dodgerblue', 'orange']
        Tfits = [T1_fit, T2_fit]; Rfits = [r1_fit, r2_fit]
        for i in range(1, N_reflections+1):  # Plot first 3 reflections separately
            reflection_curve = (Rfits[i-1]) * lognorm(high_res_time_grid - i * Tfits[i-1], p0_fit, p1_fit, p2_fit)  # r1, r2 multiplied by lognorm
            plt.plot(high_res_time_grid, reflection_curve, label=f"Reflection {i}", linestyle="dashed", color = colors[i-1])
        plt.plot(high_res_time_grid, fitted_curve, label=f"Total fit", color='lime', linewidth = 2)    # total fit
        plt.xlabel('nanoseconds', loc = 'right')
        plt.ylabel('ADC')
        plt.title('SPE waveform modeling | PMT ' + str(chan))
        plt.legend(fontsize = 11, frameon = False)
        plt.savefig('plots/fit_results/' + str(chan) + ' SPE waveform fit two reflections _ ' + which_mode + '.png',
                    dpi=300, bbox_inches='tight', pad_inches=.3, facecolor='w')
        plt.close()
        

        # ---------------------------------------- #
        # we can export the fits results to a .csv
        # Prepare header
        header = ["p0", "p1", "p2", "T1", "T2", "r1", "r2"]
        header += [f"uncertainty_{param}" for param in header[:7]]  # Param uncertainties

        # Prepare the row data
        row_data = list(popt)  # Fitted parameters
        row_data += list(param_errors)  # Parameter uncertainties

        # Save to CSV
        csv_filename = "fit_csv/fit_results_" + str(chan) + "_" + which_mode + ".csv"
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(row_data)
        if verbose:
            print(f"\nFit results saved to {csv_filename}") 
        
    except:

        if verbose:
            print('Fits failed for some reason, moving to the next one')


print('\nFit results saved to: "fit_csv/"')
print('Plots of the fits saved to: "plots/fit_results/"\n')


# ----------------------------------------------------------------------------------------------------------------- #

# Lastly, we can stich together each csv file to construct the properly need Lognorm file for the PMTWaveformSim tool

print('\n\n--------------------------------------------------------')
print('\nCombining fit files into a PMTWaveformLognormFit.csv...\n')

outputfilename = 'PMTWaveformLognormFit.csv'

chankey = df2['channel_num'].tolist()

vals = ['p0','p1','p2','T1','T2','r1','r2',
        'uncertainty_p0','uncertainty_p1','uncertainty_p2',
        'uncertainty_T1','uncertainty_T2','uncertainty_r1','uncertainty_r2']

output_data = []

for chan in chankey:

    filename = f'fit_csv/fit_results_{chan}_{which_mode}.csv'
    
    # Check if the file exists (some channels won't have fits)
    if os.path.exists(filename):
        df = pd.read_csv(filename, usecols=vals)
        
        # Use the first row of fit results
        entry = [chan]  # Start with the channel number
        entry.extend(df.iloc[0].tolist())
        output_data.append(entry)
    else:
        if verbose:
            print(f"Warning: No fit file found for channel {chan}. Skipping.")
        

# Convert the output data to a DataFrame and write it to a CSV file
output_columns = ['PMT'] + vals
with open(outputfilename, 'w', newline='') as f:
    f.write(', '.join(output_columns) + '\n')  # Write header
    for row in output_data:
        f.write(', '.join(map(str, row)) + '\n')  # Write data rows
        

print("\nData successfully written to '" + outputfilename + "'\n")

# ---------------------------------------------------------------------------------------------------------- #

print('\n-----------------------------------------------\n\nel fin\n')
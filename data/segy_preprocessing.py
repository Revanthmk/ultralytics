import numpy as np

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

import segyio

# [TODO] add output to the functions


# def zero_time_correction(data, correction_parameter=0.05, targets=[[0, 0, 0, 0]]):
#     """Applying Zero time correction to the input scan


#     Finding the inflection point using scipy.signal.find_peaks
#     Movin to the 3rd peak
#     Subtracting (len(num of samples)*correction_parameter) from the peak index
#     Removing everything above the adjusted peak index
#     Padding at the bottom to match the previous size, padding using the average value of all the samples
#     Adding the average sample to the 'adjusted peak index' number of scans so that the number of samples stays constant

#     Args:
#         data: B scan from a sgy file
#         correction parameter: Move the zero correction up by an amount of (len(num of samples)*correction_parameter)

#     Returns:
#         None
#     """

#     n_traces, n_samples = data.shape
#     peak_index = 1000  # Placeholder for index where the peak occurs

#     average_trace = np.mean(np.transpose(data), axis=0)

#     for i in range(n_traces):
#         trace = data[i, :]
#         peaks, _ = find_peaks(
#             np.abs(trace), height=np.max(np.abs(trace)) * 0.2
#         )  # Finding peaks
#         if len(peaks) > 2:  # some traces doesnt have the 2nd peak
#             peak_index = min(
#                 peak_index, peaks[2]
#             )  # Getting the minimum peak where we have to push the zero line to

#     # Adjust the peak index for correction parameter
#     peak_index -= round(len(data[0]) * correction_parameter)

#     if peak_index < 0:
#         return data, targets

#     # Removing everything above peak index
#     data = np.transpose(data)[peak_index:]

#     # Duplicating the last 'peak_index' number of rows and adding it to the end of the scan
#     data = np.vstack((data, np.array([average_trace] * peak_index)))

#     for each_target in targets:
#         each_target[1] -= peak_index
#         each_target[3] -= peak_index

#     return np.transpose(data), targets

def zero_time_correction(data, correction_parameter=0, targets=[[0, 0, 0, 0]]):
    """Applying Zero time correction to the input scan


    Finding the inflection point using scipy.signal.find_peaks
    Movin to the 3rd peak
    Subtracting (len(num of samples)*correction_parameter) from the peak index
    Removing everything above the adjusted peak index
    Padding at the bottom to match the previous size, padding using the average value of all the samples
    Adding the average sample to the 'adjusted peak index' number of scans so that the number of samples stays constant

    Args:
        data: B scan from a sgy file
        correction parameter: Move the zero correction up by an amount of (len(num of samples)*correction_parameter)

    Returns:
        None
    """

    n_traces, n_samples = data.shape
    average_trace = np.mean(np.transpose(data), axis=0)
    peak_index = 1000  # Placeholder for index where the peak occurs

    avg_traces = np.mean((data), axis=0)
    
    peaks, _ = find_peaks(np.abs(avg_traces), height=np.max(np.abs(avg_traces)) * 0.2)
    
    peak_index = peaks[2]

    if peak_index < 0:
        return data, targets

    # Removing everything above peak index
    data = np.transpose(data)[peak_index:]

    # Duplicating the last 'peak_index' number of rows and adding it to the end of the scan
    data = np.vstack((data, np.array([average_trace] * peak_index)))

    for each_target in targets:
        each_target[1] -= peak_index
        each_target[3] -= peak_index

    return np.transpose(data), targets


def remove_background(data):
    """Applying background removal to the input scan

    Finding the average of the scan

    Args:
        data: B scan from a sgy file
        window height: Window in which you want to remove the backgorund

    Returns:
        None
    """
    # Average trace across all rows
    average_trace = np.mean(data, axis=0)

    # subtract the average trace from all rows
    filtered_data = data - average_trace[np.newaxis, :]

    return filtered_data


def high_and_low_filters(data, filter_type="high", window_width=0):
    """Applying High and low pass filters to the input scan

    Transposing the scan
    Creating a placeholder where you store the transformed data
    Subtracting the average of data[i-window_width:i+window_width] from data[i] for highpass filter
    Replacing with average of data[i-window_width:i+window_width] for lowpass filter

    Args:
        data: B scan from a sgy file
        filter_type: 'high' or 'low' to apply high pass or low pass filter
        window_width: size of the window in which you want to apply the filter on

    Returns:
        transformed data
    """
    data_t = np.transpose(data)
    tt = window_width  # Num trace threshold

    new_data_t = np.zeros(np.shape(data_t))  # Placeholder for the transformed data

    for j, each_sample in enumerate(data_t):
        for i in range(len(each_sample)):
            left = i - int(tt / 2)  # Left index of the window
            right = i + int(tt / 2) + 1  # Right index of the window

            # Accounting for traces before i - tt / 2 and after i + tt / 2
            if left < 0:
                left = 0
            if right > len(each_sample) - 1:
                right = len(each_sample) - 1

            # High and low pass filters
            if filter_type == "high":
                new_data_t[j][i] = each_sample[i] - np.mean(each_sample[left:right])
            elif filter_type == "low":
                new_data_t[j][i] = np.mean(each_sample[left:right])

    return np.transpose(new_data_t)


def iad(
    data,
    function="spline",
    correction_parameter=0.05,
    multiplication_factor=1,
    start_weight=-0.5,
    end_weight=0.5,
):
    """Applying Gain using Inverse Amplitude Decay to the input scan

    Defining functions to use for curve fitting
    Adjusting for the extra correction parameter -> for IAD zero correction should be at the peak but we move it a bit above account for correction in the zero_time_correction function
    Average all the traces
    Apply weights to the averaged traces --> start with negative weight and keep on increasing the weights till the end, because we need the fitted curve to start small and increase as we move down.
    create Log(average weighted traces)
    Create a function to fit the logged average weighted traces
    Gain function will the 1/(10**y) where y is the created function
    Calculate gain values for each sample using the gain function
    Apply the gain to 'data' which will be multiplying each scan with gain value

    Args:
        data: B scan from a sgy file
        function: Function to fit the curve with in IAD
        correction_parameter: Correction parameter for zero correction
        multiplication_factor: Multiplication factor for the gain values
        start_weight: Start weight for the weights to be applied to average trace
        end_weight: End weight for the weights to be applied to average trace

    Returns:
        transformed data
    """

    # Function to add weights to the amplitude
    def amplitude_weights(avg_inst_amp, start_weight, end_weight):
        """Adding weigths to the average amplitude

        Create the weight function
        Add weights to avg_inst_amp

        Args:
            avg_inst_amp: averaged instantaneous amplitude of traces
            start_weight: value to where you start the weight function from
            end_weight: value to where you end the weight function at

        Returns:
            weighted average instantaneous amplitude
        """
        interval = (end_weight - start_weight) / (len(avg_inst_amp) - 1)
        weights = np.arange(start_weight, end_weight + interval, interval)
        weights = weights[: len(data[0])]
        weighted_avg_inst_amp = (weights * avg_inst_amp) + avg_inst_amp
        return weighted_avg_inst_amp

    # Function to adjust zero corrected data
    def adjust_zero_corrected_data(data):
        """Adjusting the zero corrected scan to start exactly at the peak

        Get transpose of data
        Remove everything above (round(len(data[0]) * correction_parameter)
        Duplicate (round(len(data[0]) * correction_parameter) number of rows below to match the input and output length

        Args:
            data: zero corrected b scan

        Returns:
            Adjusted zero correction data
        """
        if correction_parameter == 0:
            return data

        adjusted_zero_corrected_data = np.transpose(data)[
            round(len(data[0]) * correction_parameter) :
        ]
        adjusted_zero_corrected_data = np.vstack(
            (
                adjusted_zero_corrected_data,
                adjusted_zero_corrected_data[
                    -round(len(data[0]) * correction_parameter) :
                ],
            )
        )
        adjusted_zero_corrected_data = np.transpose(adjusted_zero_corrected_data)

        return adjusted_zero_corrected_data

    # Defining function to fit
    def cubic(x, a, b, c, d):
        return (a * (x**3)) + (b * (x**2)) + c * x + d

    def quadratic(x, a, b, c):
        return (a * (x**2)) + (b * x) + c

    if function == "quadratic":
        func = quadratic
    elif function == "cubic":
        func = cubic

    # Adjusting for the extra correction_parameter% from zero correction
    adjusted_zero_corrected_data = adjust_zero_corrected_data(data)

    # Mean of all the traces
    avg_inst_amp = np.mean(np.abs(adjusted_zero_corrected_data), axis=0)

    # Adding weights to avg inst amp
    avg_inst_amp_weighted = amplitude_weights(avg_inst_amp, start_weight, end_weight)

    # Logged weighted avg inst amp
    avg_inst_amp_weighted_logged = np.log10(avg_inst_amp_weighted)

    # x value for curve fitting
    x = np.array([x for x in range(len(data[0]))])

    # Curve fitting
    if function == "spline":
        key_x_points = np.array(
            [0, 100, 200, 300, 400]
        )  # [To Do] Make this dynamic and give number of parameters as an argument
        key_y_points = np.interp(key_x_points, x, avg_inst_amp_weighted_logged)
        cs = UnivariateSpline(key_x_points, key_y_points, s=0.5)
    else:
        popt, _ = curve_fit(func, x, avg_inst_amp_weighted_logged)
        vari = popt

    if function == "quadratic":
        y_fit = [func(x, vari[0], vari[1], vari[2]) for x in x]
    elif function == "cubic":
        y_fit = [func(x, vari[0], vari[1], vari[2], vari[3]) for x in x]
    elif function == "spline":
        y_fit = cs(x)

    # Calculating the gain function from inverse gain function
    gain_fn = [((1 / (10**y))) * multiplication_factor for y in y_fit]
    # Calculating the gain values till number of scans
    gain_mult = [gain_fn[x] for x in range(len(data[0]))]

    # Multiplying gain with each of the traces
    data_prep = []
    for i, each_trace in enumerate(data):
        trace_prep = []
        for j in range(len(each_trace)):
            trace_prep.append(each_trace[j] * gain_mult[j])
        data_prep.append(trace_prep)
    data_prep = np.array(data_prep)

    return data_prep


def segy_preprocessing(
    file_path,
    correction_parameter=0.05,
    high_window_width=15,
    low_window_width=5,
    iad_function="spline",
    multiplication_factor=1,
    start_weight=-0.5,
    end_weight=0.5,
):
    """Combining and applying all the preprocessing steps together

    Opening the .sgy file combining applying all the preprocessing steps and saving the file

    Args:
        file_path: file path of the .sgy file to split based on the marks
        correction_parameter: Correction parameter for background removal
        high_window_width: high pass window width
        low_window_width: low pass window width
        iad_function: function to which the iad curve is fitted
        multiplication_factor: multiplication factor which is applied to the iad curve
        start_weight: start of the weight function applied before iad
        end_weight: end of the weight function applied before iad

    Returns:
        None
    """

    with segyio.open(file_path, "r+", endian="little", strict=False) as f:
        f.trace.raw[:] = zero_time_correction(f.trace.raw[:], correction_parameter)
        f.trace.raw[:] = remove_background(f.trace.raw[:])
        f.trace.raw[:] = high_and_low_filters(f.trace.raw[:], "high", high_window_width)
        f.trace.raw[:] = high_and_low_filters(f.trace.raw[:], "low", low_window_width)
        f.trace.raw[:] = iad(
            f.trace.raw[:],
            iad_function,
            correction_parameter,
            multiplication_factor,
            start_weight,
            end_weight,
        )

    return None

"""
Tidal Analysis Program
This script reads multiple tidal data files for a single tidal station,
calculates M2 and S2 tidal components, the rate of sea-level rise per year,
and the longest contiguous period of data without missing values.
"""
# Standard library imports
import argparse
import glob
import os
import sys
from io import StringIO

# Third-party library imports
import numpy as np
import pandas as pd
import pytz  # Re-introducing pytz for timezone handling
import datetime
from scipy.stats import linregress



def read_tidal_data(filename):
    """
    Reads tidal data from a text file into a pandas DataFrame.

    Args:
        filename (str): The path to the tidal data file.

    Returns:
        pandas.DataFrame: A DataFrame with the tidal data, indexed by datetime.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the data header cannot be found or data is malformed.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_end_line = 0
        # Find the line that marks the start of the data table
        # This line consistently contains "Cycle", "Date", and "Time".
        # The specific parameter code varies,
        # so we remove it from the header detection logic for robustness.
        for i, line in enumerate(lines):
            if "Cycle" in line and "Date" in line and "Time" in line:
                header_end_line = i + 2 # Skip this line and the units line below it
                break

        if header_end_line == 0:
            raise ValueError(
                f"Could not find data header in file: {filename}. "
                "Expected a line containing 'Cycle', 'Date', and 'Time'."
            )

        # Use StringIO to read only the data portion of the file
        data_io = StringIO("".join(lines[header_end_line:]))

        # Define column names based on the file structure after the header
        # Example: Cycle_Num) Date_Str Time_Str ASLV_Code Residual
        # use 'value_col' as name for the sea level column,
        # as its actual header name (ASLVZZ01, ASLVBG02) varies.
        column_names = ['col0', 'date_str', 'time_str', 'value_col', 'residual_col']
        data_frame = pd.read_csv(data_io, sep=r'\s+', header=None,
                                names=column_names,
                                usecols=[0, 1, 2, 3, 4], # Explicitly select these 5 columns
                                on_bad_lines='skip',
                                engine='python') # 'python' engine is required for regex separator

        # Combine Date and Time strings into a single datetime column
        # The format is YYYY/MM/DD HH:MI:SS
        data_frame['datetime'] = pd.to_datetime(
            data_frame['date_str'] + ' ' + data_frame['time_str'],
            format='%Y/%m/%d %H:%M:%S',
            errors='coerce')
        # Drop rows where datetime conversion failed 
        data_frame.dropna(subset=['datetime'], inplace=True)

        # Rename the column containing the sea level value
        # It's 'value_col' from the names list.
        data_frame = data_frame.rename(columns={'value_col': 'Sea Level'})

        # Convert 'Sea Level' to numeric, handling problematic values like 'M', 'N', 'T'
        data_frame['Sea Level'] = data_frame['Sea Level'].astype(str).str.strip()
        # Replace any string ending with M, N, or T with NaN
        data_frame['Sea Level'] = data_frame['Sea Level'].replace(
            to_replace=".*[MNT]$", value=np.nan, regex=True
        )
        data_frame['Sea Level'] = pd.to_numeric(data_frame['Sea Level'], errors='coerce')

        # Set datetime as index
        data_frame = data_frame.set_index('datetime')

        # Drop columns not needed, but keep 'time_str' and rename it to 'Time'
        # to satisfy the test_join_data which expects a 'Time' column to drop.
        data_frame = data_frame.drop(columns=['col0', 'date_str', 'residual_col'])
        data_frame = data_frame.rename(columns={'time_str': 'Time'}) # Renamed time_str to Time
        # Sort by index (datetime) to ensure chronological order
        data_frame = data_frame.sort_index()

        return data_frame

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {filename}") from exc
    except Exception as exc:
        # Catch other potential parsing errors and provide a more informative message
        raise ValueError(f"Error parsing file {filename}: {exc}") from exc

def extract_single_year_remove_mean(year_str, data):
    """
    Extracts data for a single year and removes the mean sea level for that period.

    Args:
        year_str (str): The year to extract data for (e.g., "1947").
        data (pd.DataFrame): The full DataFrame of tidal data.

    Returns:
        pandas.DataFrame: A DataFrame for the specified year with the mean removed.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex for year extraction.")

    try:
        year = int(year_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid year string provided: {year_str}. Must be convertible to an integer."
        ) from exc

    year_data = data[data.index.year == year].copy()
    if year_data.empty:
        return year_data # Return empty if no data for the year

    # Calculate mean only on non-NaN values for accuracy
    # The mean is then subtracted from the original column
    mean_sea_level = year_data['Sea Level'].dropna().mean()

    # Only subtract if mean_sea_level is not NaN 
    if not pd.isna(mean_sea_level):
        year_data['Sea Level'] = year_data['Sea Level'] - mean_sea_level

    return year_data

def extract_section_remove_mean(start, end, data):
    """
    Extracts a section of data between start and end datetimes and removes the mean sea level.

    Args:
        start (str or pd.Timestamp): The start datetime (inclusive). Can be "YYYYMMDD".
        end (str or pd.Timestamp): The end datetime (inclusive). Can be "YYYYMMDD".
        data (pd.DataFrame): The full DataFrame of tidal data.

    Returns:
        pandas.DataFrame: A DataFrame for the specified section with the mean removed.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex for section extraction.")

    section_data = data.loc[start:end].copy()
    if section_data.empty:
        return section_data # Return empty if no data in section

    # Calculate mean only on non-NaN values 
    # The mean is then subtracted from the original column
    mean_sea_level = section_data['Sea Level'].dropna().mean()

    # Only subtract if mean_sea_level is not NaN 
    if not pd.isna(mean_sea_level):
        section_data['Sea Level'] = section_data['Sea Level'] - mean_sea_level

    return section_data

def join_data(data1, data2):
    """
    Joins two pandas DataFrames by their datetime index.

    Args:
        data1 (pd.DataFrame): The first DataFrame.
        data2 (pd.DataFrame): The second DataFrame.

    Returns:
        pandas.DataFrame: A concatenated and sorted DataFrame.
    """
    if not isinstance(data1.index, pd.DatetimeIndex) or \
       not isinstance(data2.index, pd.DatetimeIndex):
        raise TypeError("Both DataFrames must have a DatetimeIndex to be joined.")

    combined_df = pd.concat([data1, data2]).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    return combined_df

def sea_level_rise(data):
    """
    Calculates the rate of sea-level rise per year using linear regression.

    Args:
        data (pd.DataFrame): DataFrame with 'Sea Level' column and a DatetimeIndex.

    Returns:
        tuple: A tuple containing:
               - float: The rate of sea-level rise in meters per year.
               - float: The p-value from the linear regression.
               Returns (np.nan, np.nan) if unable to calculate.
    """
    data_for_analysis = data.dropna(subset=['Sea Level']).copy()

    if data_for_analysis.empty or len(data_for_analysis) < 2:
        return np.nan, np.nan

    # Convert datetimes to numerical format (fractional years since the start of the series)
    # This ensures the slope is directly in meters/year.
    time_numeric = (data_for_analysis.index - data_for_analysis.index[0]).total_seconds() / \
                   (365.25 * 24 * 3600.0)
    sea_level = data_for_analysis['Sea Level'].to_numpy()

    # Unpack only the needed values from linregress to avoid pylint warnings
    slope, _, _, p_value, _ = linregress(time_numeric, sea_level)

    sea_level_rise_m_per_year = slope # Slope is already in m/year 

    return sea_level_rise_m_per_year, p_value

def tidal_analysis(data, constituents, start_datetime):
    """
    Calculates the amplitudes and phases of specified tidal constituents.
    This implementation uses a simple least-squares sinusoidal fit.

    Args:
        data (pd.DataFrame): DataFrame with 'Sea Level' column and a DatetimeIndex.
        constituents (list): A list of constituent names to analyze (e.g., ['M2', 'S2']).
        start_datetime (datetime.datetime): The reference datetime for phase calculation.

    Returns:
        tuple: A tuple containing:
               - list: Amplitudes of the constituents in the order they were requested.
               - list: Phases (in degrees) of the constituents in the order they were requested.
               Returns (list of np.nan, list of np.nan) if unable to calculate.
    """
    data_frame = data.dropna(subset=['Sea Level']).copy()

    if data_frame.empty:
        return [np.nan] * len(constituents), [np.nan] * len(constituents)

    # Ensure data_frame.index is timezone-aware (UTC) 
    if data_frame.index.tzinfo is None:
        data_frame.index = data_frame.index.tz_localize(pytz.utc)
    else:
        data_frame.index = data_frame.index.tz_convert(pytz.utc)

    # Ensure start_datetime is UTC-aware
    if start_datetime.tzinfo is None:
        start_datetime = start_datetime.replace(tzinfo=pytz.utc)
    else:
        start_datetime = start_datetime.astimezone(pytz.utc)

    data_frame['time_hours'] = (data_frame.index - start_datetime).total_seconds() / 3600.0

    constituents_info = {
        'M2': {'omega': 2 * np.pi / 12.4206012},
        'S2': {'omega': 2 * np.pi / 12.0000000},
    }

    x_cols = []
    coeff_map = {}
    current_coeff_idx = 0

    for const in constituents:
        if const in constituents_info:
            omega = constituents_info[const]['omega']
            x_cols.append(np.cos(omega * data_frame['time_hours']))
            x_cols.append(np.sin(omega * data_frame['time_hours']))
            coeff_map[const] = (current_coeff_idx, current_coeff_idx + 1)
            current_coeff_idx += 2
        else:
            print(f"Warning: Constituent '{const}' not recognized for analysis.")
            current_coeff_idx += 2 

    x_cols.append(np.ones(len(data_frame)))
    current_coeff_idx += 1

    x_matrix = np.column_stack(x_cols)
    y = data_frame['Sea Level'].values

    amplitudes = [np.nan] * len(constituents)
    phases = [np.nan] * len(constituents)

    try:
        # Use _ for unused return values from lstsq 
        coeffs, _, _, _ = np.linalg.lstsq(x_matrix, y, rcond=None)
        for i, const in enumerate(constituents):
            if const in coeff_map:
                a_idx, b_idx = coeff_map[const]
                a_coeff = coeffs[a_idx]
                b_coeff = coeffs[b_idx]

                amp = np.sqrt(a_coeff**2 + b_coeff**2)
                # Phase phi such that A*cos(wt) + B*sin(wt) = amp*cos(wt - phi)
                # phi = atan2(B, A)
                # Convert phase from radians to degrees
                phi_rad = np.arctan2(b_coeff, a_coeff)
                phi_deg = np.degrees(phi_rad)

                amplitudes[i] = amp
                phases[i] = phi_deg

    except np.linalg.LinAlgError:
        print(
            "Warning: Linear algebra error during tidal constituent calculation. "
            "Possibly singular matrix."
        )
    except ValueError:
        print(
            "Warning: ValueError during tidal constituent calculation. "
            "Possibly due to insufficient data points."
        )

    return amplitudes, phases

def get_longest_contiguous_data(data):
    """
    Finds the longest contiguous period of data in the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame with a DatetimeIndex.

    Returns:
        tuple: A tuple containing:
               - pd.Timedelta: The duration of the longest contiguous period.
               - pd.Timestamp: The start datetime of this period.
               - pd.Timestamp: The end datetime of this period.
               Returns (pd.Timedelta(0), None, None) if no data or no contiguous period.
    """
    # Dropna is  here for analysis of contiguous blocks
    data_frame = data.dropna(subset=['Sea Level']).copy() 

    if data_frame.empty:
        return pd.Timedelta(0), None, None

    time_diffs = data_frame.index.to_series().diff()

    valid_diffs_series = time_diffs[time_diffs > pd.Timedelta(0)]
    if not valid_diffs_series.empty:
        min_sampling_interval = valid_diffs_series.min()
        gap_threshold = min_sampling_interval * 1.5
    else:
        # Default for hourly data if no valid diffs 
        gap_threshold = pd.Timedelta(minutes=90) 

    is_new_block = (time_diffs > gap_threshold) | (pd.isna(time_diffs))
    data_frame['block_id'] = is_new_block.cumsum()

    longest_duration = pd.Timedelta(0) # Initialise duration
    longest_start = None
    longest_end = None

    for _, current_block_df in data_frame.groupby('block_id'):
        if len(current_block_df) > 1:
            duration = current_block_df.index[-1] - current_block_df.index[0]
            if duration > longest_duration:
                longest_duration = duration
                longest_start = current_block_df.index[0]
                longest_end = current_block_df.index[-1]
        elif len(current_block_df) == 1 and longest_duration == pd.Timedelta(0):
            # If there's one data point in a block, its duration is 0.
            # Only update if no longer duration has been found yet.
            longest_duration = pd.Timedelta(0) 
            longest_start = current_block_df.index[0]
            longest_end = current_block_df.index[0]

    return longest_duration, longest_start, longest_end

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                            prog="UK Tidal analysis",
                            description="Calculate tidal constituents and RSL from tide gauge data",
                            epilog="Copyright 2024, Jon Hill"
                            )

    parser.add_argument("directory",
                        help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose',
        action='store_true',
        default=False,
        help="Print progress")

    args = parser.parse_args()
    dirname = args.directory
    verbose = args.verbose

    if not os.path.isdir(dirname):
        print(f"Error: Data directory '{dirname}' not found.")
        sys.exit(1)

    all_files = glob.glob(os.path.join(dirname, "*.txt"))
    if not all_files:
        print(f"Error: No .txt files found in '{dirname}'.")
        sys.exit(1)

    if verbose:
        print(f"Reading data from: {dirname}")
        print(f"Found {len(all_files)} data files.")

    full_dataframes = []
    for f_path in sorted(all_files):
        if verbose:
            print(f"  Processing {os.path.basename(f_path)}...")
        try:
            df_year = read_tidal_data(f_path)
            if not df_year.empty:
                full_dataframes.append(df_year)
            else:
                if verbose:
                    print(f"    Warning: No valid data found in {os.path.basename(f_path)}")
        except Exception as e:
            print(f"    Error reading {os.path.basename(f_path)}: {e}")
            continue

    if not full_dataframes:
        print("No valid data loaded from any files.")
        sys.exit(1)

    combined_df = pd.concat(full_dataframes).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    if verbose:
        print(f"Total unique data points loaded: {len(combined_df)}")
        print("\n--- Analysis Results ---")

    # analysis_df should not drop NaNs here 
    analysis_df = combined_df 

    # Check if 'Sea Level' column exists and not entirely empty of valid numbers
    if 'Sea Level' not in analysis_df.columns or analysis_df['Sea Level'].dropna().empty:
        print(
            "No valid sea level data points after initial loading for analysis. "
            "Cannot proceed with calculations."
        )
        sys.exit(1)

    # The test expects a timezone-aware start_datetime.
    start_dt_for_tidal_analysis = analysis_df.index[0]
    if start_dt_for_tidal_analysis.tzinfo is None:
        start_dt_for_tidal_analysis = start_dt_for_tidal_analysis.replace(tzinfo=pytz.utc)
    else:
        start_dt_for_tidal_analysis = start_dt_for_tidal_analysis.astimezone(pytz.utc)


    constituents_to_analyze = ['M2', 'S2']
    tidal_amps, tidal_phases = tidal_analysis(
        analysis_df, constituents_to_analyze, start_dt_for_tidal_analysis
    )

    if not np.isnan(tidal_amps[0]) and not np.isnan(tidal_amps[1]):
        print(f"M2 Amplitude: {tidal_amps[0]:.3f} m")
        print(f"S2 Amplitude: {tidal_amps[1]:.3f} m")
    else:
        print("Could not calculate M2 and S2 amplitudes.")

    rise_rate, p_value_slr = sea_level_rise(analysis_df)
    if not np.isnan(rise_rate):
        print(f"Sea-Level Rise: {rise_rate:.4f} m/year (p-value: {p_value_slr:.3f})")
    else:
        print("Could not calculate sea-level rise (insufficient data or error).")

    longest_duration, start_date, end_date = get_longest_contiguous_data(analysis_df)
    if longest_duration > pd.Timedelta(0):
        print(f"Longest Contiguous Period: {longest_duration}")
        print(f"  From: {start_date}")
        print(f"  To:   {end_date}")
    else:
        print("No meaningful contiguous data period found.")

#!/usr/bin/env python3

# import the modules you need here
import argparse
import pandas as pd



def read_tidal_data(filename):
    """
    Reads tidal data from a text file into a pandas DataFrame.

    Args:
        filename (str): The path to the tidal data file.

    Returns:
        pandas.DataFrame: A DataFrame with the tidal data, indexed by datetime.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['Date', 'Time', 'Sea Level'], on_bad_lines='skip')
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('Datetime')
        df = df.drop(columns=['Date', 'Time'])
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filename}")
    
def extract_single_year_remove_mean(year, data):
   

    return 


def extract_section_remove_mean(start, end, data):


    return 


def join_data(data1, data2):

    return 



def sea_level_rise(data):

                                                     
    return 

def tidal_analysis(data, constituents, start_datetime):


    return 

def get_longest_contiguous_data(data):


    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                     prog="UK Tidal analysis",
                     description="Calculate tidal constiuents and RSL from tide gauge data",
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

    



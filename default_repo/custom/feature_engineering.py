import matplotlib.pyplot as plt
# import googlemaps
import os
import requests
import numpy as np
from datetime import timedelta
from scipy.stats import skew, kurtosis, entropy
import pandas as pd
import numpy as np
import random 
import math
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
from scipy import stats
import random
from tqdm import tqdm
import warnings
import json
import pywt

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@custom
def final_dataset(data_dict):

    warnings.filterwarnings("ignore")

    final_df = pd.DataFrame(columns = ['ID','mean','std','max','min','p2p','rms','lat','long','size','SKEW','KURT','ENERGY','ENTROPY','CREST',
                                        'MEAN_APPROXIMATION','STD_APPROXIMATION','MEAN_DETAIL','STD_DETAIL','ENERGY_APPROXIMATION','ENERGY_DETAIL',
                                        'ENTROPY_APPROXIMATION','ENTROPY_DETAIL','MAX_APPROXIMATION','MAX_DETAIL','MIN_APPROXIMATION','MIN_DETAIL','VARIANCE_APPROXIMATION','VARIANCE_DETAIL','label'])

    feature_dict = {}

    for key in data_dict.keys():

        key_df = data_dict[key]
        date_list_str, daily_consumption_list_total, daily_labels, daily_cons_past, mean_daily_labels = create_lists(key_df)

        mean_value = np.mean(daily_cons_past)
        std_value = np.std(daily_cons_past)
        max_value = np.max(daily_cons_past)
        min_value = np.min(daily_cons_past)
        p2p_value = np.ptp(daily_cons_past)
        rms_value = np.sqrt(np.mean(np.square(daily_cons_past)))
        
        skewn = skew(daily_cons_past, bias=True)
        kurt = kurtosis(daily_cons_past, bias=True)
        energy = calculate_energy(daily_cons_past)
        entropy = calc_entropy(np.array(daily_cons_past))
        crest_f = np.amax(abs(np.array(daily_cons_past)))/rms_value
        
        dwt_feats, dwt_feats_names = dwt_features(daily_cons_past)
        
        latitude = key_df['LATITUDE'].iloc[0]
        longitude = key_df['LONGITUDE'].iloc[0]
        size = key_df['SQUARE_METERS'].iloc[0]
        
        features_list = [int(key),mean_value,std_value,max_value,min_value,p2p_value,rms_value,latitude,longitude,size,skewn,kurt,energy,entropy,crest_f] + dwt_feats + [mean_daily_labels]
        
        final_df.loc[key] = features_list

        if final_df.empty:
            raise ValueError("The final_df DataFrame is")

    return final_df


def create_lists(dataframe):
    # Convert 'INVOICE_DATE' to datetime format
    dataframe['INVOICE_DATE'] = pd.to_datetime(dataframe['INVOICE_DATE'])

    # Ensure 'ENERGY_CONSUMPTION' is float
    dataframe['ENERGY_CONSUMPTION'] = pd.to_numeric(dataframe['ENERGY_CONSUMPTION'], errors='coerce')

    # Handle missing or invalid energy consumption values
    if dataframe['ENERGY_CONSUMPTION'].isnull().any():
        print("Warning: NaN values in ENERGY_CONSUMPTION! Filling with 0.")
        dataframe['ENERGY_CONSUMPTION'].fillna(0, inplace=True)

    # Extract the minimum and maximum dates
    min_date = dataframe['INVOICE_DATE'].min()
    max_date = dataframe['INVOICE_DATE'].max()

    # Calculate the start date (min_date - 120 days)
    start_date = min_date - timedelta(days=120)

    # Generate a list of dates
    date_list = pd.date_range(start=start_date, end=max_date).tolist()
    date_list_str = [date.strftime('%Y-%m-%d') for date in date_list]

    # Sort the dataframe by 'INVOICE_DATE'
    dataframe = dataframe.sort_values(by='INVOICE_DATE')

    # Calculate 'DAYS_BETWEEN_DATES'
    dataframe['DAYS_BETWEEN_DATES'] = dataframe['INVOICE_DATE'].diff().dt.days

    # Calculate daily consumption safely
    dataframe['DAILY_CONSUMPTION'] = (
        dataframe['ENERGY_CONSUMPTION'] / dataframe['DAYS_BETWEEN_DATES']
    )
    
    # Handle the first invoice
    first_invoice = dataframe['DAYS_BETWEEN_DATES'].isna()
    dataframe.loc[first_invoice, 'DAILY_CONSUMPTION'] = (
        dataframe['ENERGY_CONSUMPTION'] / 120
    )

    # Drop the intermediate column
    dataframe.drop(columns='DAYS_BETWEEN_DATES', inplace=True)

    # Initialize daily consumption list
    daily_consumption_list_total = []
    daily_cons = dataframe['DAILY_CONSUMPTION'].iloc[0]
    cnt = 0

    # Populate daily consumption list
    for date in date_list_str:
        if date == dataframe['INVOICE_DATE'].iloc[-1].strftime('%Y-%m-%d'):
            daily_cons = dataframe['DAILY_CONSUMPTION'].iloc[cnt]
        elif date == dataframe['INVOICE_DATE'].iloc[cnt].strftime('%Y-%m-%d'):
            daily_cons = dataframe['DAILY_CONSUMPTION'].iloc[cnt + 1]
            cnt += 1
        daily_consumption_list_total.append(daily_cons)

    # Split the daily consumption list
    daily_cons_past = [
        value for value in daily_consumption_list_total if value != daily_consumption_list_total[-1]
    ]
    daily_labels = [
        value for value in daily_consumption_list_total if value == daily_consumption_list_total[-1]
    ]

    # Calculate the mean of daily labels
    mean_daily_labels = np.mean(daily_labels)

    return date_list_str, daily_consumption_list_total, daily_labels, daily_cons_past, mean_daily_labels

def calculate_center_of_energy(wave, energy):
    wave = np.array(wave)
    energy_center = 0
    if energy == 0:
        return 1
    for i in range(wave.shape[0]):
        if np.isnan(wave[i]):
            wave[i] = 0
            #print('Calculation of COE: Wave is nan')
        energy_center+= i* wave[i]**2
    #print(energy_center)
    energy_center/=energy
    return energy_center


def calculate_energy(wave):
    energy = 0
    wave = np.array(wave)
    for i in range(wave.shape[0]):
        if np.isnan(wave[i]):
            wave[i] = 0
        energy+= wave[i]**2
    return energy

def dwt_features(signal):
    """
    Compute extended features of a signal using wavelet transform.

    Parameters:
    signal (list of float): The input signal.

    Returns:
    dict: A dictionary containing various wavelet features.
    """

    # Convert the signal to a numpy array
    signal = np.array(signal)

    # Perform Discrete Wavelet Transform
    coeffs = pywt.dwt(signal, 'haar')

    # Decompose into approximation and detail coefficients
    cA, cD = coeffs

    # Function to calculate entropy
    def entropy(x):
        return -np.sum(x * np.log2(x + np.finfo(float).eps))

    # Compute various features
    wavelet_features = {
        'MEAN_APPROXIMATION': np.mean(cA),
        'STD_APPROXIMATION': np.std(cA),
        'MEAN_DETAIL': np.mean(cD),
        'STD_DETAIL': np.std(cD),
        'ENERGY_APPROXIMATION': np.sum(cA ** 2),
        'ENERGY_DETAIL': np.sum(cD ** 2),
        'ENTROPY_APPROXIMATION': entropy(cA),
        'ENTROPY_DETAIL': entropy(cD),
        'MAX_APPROXIMATION': np.max(cA),
        'MAX_DETAIL': np.max(cD),
        'MIN_APPROXIMATION': np.min(cA),
        'MIN_DETAIL': np.min(cD),
        'VARIANCE_APPROXIMATION': np.var(cA),
        'VARIANCE_DETAIL': np.var(cD)
    }
    
    wavelet_feats = list(wavelet_features.values())
    wavelet_feats_names = list(wavelet_features.keys())
    
    return wavelet_feats, wavelet_feats_names

def calc_entropy(x):
    return -np.sum(x * np.log2(x + np.finfo(float).eps))

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
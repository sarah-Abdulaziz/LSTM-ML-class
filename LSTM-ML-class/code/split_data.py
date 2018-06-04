# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:41:01 2018

@author: sara
"""
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
from datetime import datetime
from dateutil.parser import parse
import sys, getopt, time ,csv

from keras.models import Sequential
from keras.layers import Dense , LSTM, Activation
from sklearn.metrics import mean_squared_error
from statsmodels import api as sm
from math import sqrt
from sklearn.model_selection import train_test_split

total_spaces = {"NORREPORT": 65,
                   "SKOLEBAKKEN":512,
                   "SCANDCENTER": 1240,
                   "BRUUNS": 953,
                   "BUSGADEHUSET": 130,
                   "MAGASIN": 400,
                   "KALKVAERKSVEJ": 210,
                   "SALLING": 700,
                   }

##############################################
###########split data#########################
df=pd.read_csv(r"C:\Users\sara\Desktop\aarhus_parking.csv",header=0,usecols=['vehiclecount', 'updatetime'  , 'totalspaces','garagecode'])
BRUUNS=df[df["garagecode"] == "BRUUNS"]
BUSGADEHUSET=df[df["garagecode"] == "BUSGADEHUSET"]
KALKVAERKSVEJ=df[df["garagecode"] == "KALKVAERKSVEJ"]
MAGASIN=df[df["garagecode"] == "MAGASIN"]
NORREPORT=df[df["garagecode"] == "NORREPORT"]
SALLING=df[df["garagecode"] == "SALLING"]
SCANDCENTER=df[df["garagecode"] == "SCANDCENTER"]
SKOLEBAKKEN=df[df["garagecode"] == "SKOLEBAKKEN"]
output_path=r"C:\Users\sara\Desktop\ML-class\SKOLEBAKKEN.csv"
SKOLEBAKKEN.to_csv(output_path,index=False,header=False)

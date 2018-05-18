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


total_spaces = {"NORREPORT": 65,
                   "SKOLEBAKKEN":512,
                   "SCANDCENTER": 1240,
                   "BRUUNS": 953,
                   "BUSGADEHUSET": 130,
                   "MAGASIN": 400,
                   "KALKVAERKSVEJ": 210,
                   "SALLING": 700,
                   }


###############################################################################
#################get hourly data##############################################
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

def split_data(df):
    train, test = train_test_split(df, test_size=0.2)
    return train,test

def hour_to_timestamp(hour):
    timestamp = datetime.fromtimestamp(hour * 3600)
    return timestamp
def convert_to_hourly_price(df):
    hourly_price = defaultdict(lambda:0.0, {})
    previous_record = defaultdict( lambda:-1, {})
    for line in reversed(open(df).readlines()):
        parknig_time = line.rstrip().split(",")
        parknig_time[1]=datetime.strptime(parknig_time[1], '%Y-%m-%d %H:%M:%S')
        if len(parknig_time) < 2:
            continue
        new_vehiclecount = float(parknig_time[0])
        current_second = int(time.mktime(parse(str(parknig_time[1])).timetuple()))
        current_hour = (current_second // 3600)
        if (previous_record['seconds'] >= 0):
                if (previous_record['hour'] == current_hour):
                    print("1")
                    hourly_price[current_hour] += ((current_second - previous_record['seconds']) / 3600.0 * previous_record['vehiclecount'])
                elif (previous_record['hour'] < current_hour):
                    print("2")
                    processed_second = previous_record['seconds']
                    for h in range(previous_record['hour'], current_hour):
                         hourly_price[h] += (((h+1)*3600 - processed_second)/3600.0) * previous_record['vehiclecount']
                         processed_second = (h+1)*3600
                    hourly_price[current_hour] += ((current_second - processed_second)/3600.0 * previous_record['vehiclecount'])
                else:

                    raise Exception("invalid hour entered old:%d new:%d",(previous_record['hour'], current_hour))
        previous_record['seconds'] = current_second
        previous_record['hour'] = current_hour
        previous_record['vehiclecount'] = new_vehiclecount
    with open(output_file_path, 'w', newline='') as f:
         writer = csv.writer(f)
         for k in sorted(hourly_price):
             str(hour_to_timestamp(k))+str(hourly_price[k])
             out= hour_to_timestamp(k)
             out1=hourly_price[k]
             writer.writerow([out,int(out1)])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:52:12 2020

@author: soni6
invasive ventilators data
This doesn't count the number of ventilators available, just more requirement
"""


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("Hospitalization_all_locs.csv") 
data_ven=data[['location_name', 'date','InvVen_mean', 'InvVen_lower', 'InvVen_upper']]
us_states=["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"];
           
data_ven= data_ven[data_ven.location_name.isin(us_states)]
data_ven_date=data_ven.groupby(['date']).sum()

#we can consider 16500 ventilators in the federal stock
plt.figure(figsize=(14,10),dpi=80)
for state in us_states:
    inv_state=data_ven.loc[data_ven['location_name'] == state][['InvVen_mean']]
    plt.plot(data_ven.date.unique(),inv_state,label=state)
plt.legend(fontsize=7)
plt.xlabel('Time',fontsize=16)
plt.ylabel('Number of additional ventilators required',fontsize=16)
plt.title('Demand data: Jan 1-August 4,2020',fontsize=16)
plt.gca().axes.xaxis.set_ticks([])
plt.savefig('demand-data.png')
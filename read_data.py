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
from datetime import datetime
import numpy as np

class vent_data:
    def __init__(self):
        self.demand=None
        self.states=None;
        self.scenarios=None
        self.penalty=None
        self.inventory={}
        self.stages=None
        self.distance={}
        
        
    def read_inv_data(self,plot=False):
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
                   
        data_ven=data_ven[data_ven.location_name.isin(us_states)]
        #data_ven_date=data_ven.groupby(['date']).sum()
        
        #we can consider 16500 ventilators in the federal stock
        if plot==True:
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
        return data_ven
    
    def weekly_aggregation(self,data_ven):
        data_ven['date'] = data_ven['date'].astype('datetime64[ns]')    
        #convert daily data to weekly
        weekly_data = data_ven.groupby("location_name").resample('W-Wed', label='right', closed = 'right', on='date').mean().reset_index().sort_values(by='date')
        return weekly_data
    
    def filter_data(self,weekly_data):
        #filter out data for the necessary peaks i.e. March 18-June 17, 12 periods, 
        #we can solve in two batches maybe if 12 are too many staes
        #focus on high demand states
        highDemandStates=["New York","New Jersey","Connecticut","Massachusetts","Florida","Michigan","Georgia","Pennsylvania","Texas","Illinois","Missouri","Indiana","Kentucky","Louisiana"];
        weekly_data_sddp=weekly_data[weekly_data.location_name.isin(highDemandStates)]
        weekly_data_index =(weekly_data['date'] > '2020-3-17') & (weekly_data['date'] <= '2020-6-17')
        return weekly_data_sddp.loc[weekly_data_index]
                   
    def generate_demand_data(self,trimmed_data,numNodes):
        #create data for various nodes in each stage
        #demand for each state is normal distribution with std mean
        #demand=mean+u*(Max-Mean)-l*(Mean-Min) where u and l are Rvs between (0,1)
        
        demand={};
        numStages=len(trimmed_data.date.unique());
        weeks=list(trimmed_data.date.unique());
        weeks.sort()
        for t in range(1,numStages+1):
            stage_data=trimmed_data.loc[(trimmed_data['date']==weeks[t-1])]
            for n in range(1,numNodes+1):
                for s in list(trimmed_data.location_name.unique()):
                    state_data=stage_data.loc[(stage_data['location_name']==s)]
                    u=np.random.random(1)[0];
                    l=np.random.random(1)[0];
                    demand[(t,n,s)]=max(0,round(float(state_data.InvVen_mean)+u*(float(state_data.InvVen_upper)-float(state_data.InvVen_mean))-l*(float(state_data.InvVen_mean)-float(state_data.InvVen_lower)),1))
        return demand
    
    def data(self):
        data=self.read_inv_data()
        weekly_data=self.weekly_aggregation(data);
        trimmed_data=self.filter_data(weekly_data);
        numNodes=10;
        self.demand=self.generate_demand_data(trimmed_data,numNodes)
        self.states=list(trimmed_data.location_name.unique()) #only high demand states
        self.scenarios=[i for i in range(1,numNodes+1)]
        self.penalty=60000
        #we consider each state has 800 ventilators to start with
        for s in self.states:
            self.inventory[s]=800
        self.stages=[i for i in range(1,len(trimmed_data.date.unique())+1)]
        #generating random data for distance
        for s1 in self.states:
            for s2 in self.states:
                self.distance[(s1,s2)]=round(np.random.uniform(400,2000),2)
    def gen_data(self):
        self.data()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:30:08 2020

@author: soni6
"""

import read_data as data
from gurobupi import Model, GRB
import numpy as np

class mssp:
    def __init__(self,data):
        self.states=data.states; #list of string
        self.distance=data.distance; #dictionary, eg D={('Wisconsin,Newyork'):2300}
        self.inventory=data.inventory;#initial inventory, eg I={'Wisconsin':24}
        self.penalty=data.penalty; #penalty per unit of unmet demand
        self.demand=data.demand; #demand data for each stage, demand={('#stage(t),#node(scenario),state(name)'):450}
        self.stages=data.stages; #list of integers
        self.scenarios=data.scenarios; #dict with # scenarios in each stage, scenarios={stage:[1,2,3..]}
        #models
        self.fwd_model={}, # a model for each stage which keeps getting updated with new cuts
        self.bwd_model={} #a model for each stage which gets updated with new cuts, diff scenarios solved by updating RHS
        #define variables needed to add cuts
        self.theta={};
        self.u={};
        #storing solutions
        self.prevStageVent={0:data.inventory}
        for i in self.stages:
            self.prevStageVent[i]={};
        self.alpha={};
        
    def model(self):
        m=Model()
        '''don't add cuts in this function, just build initial model
        time/scenario index not needed
        change rhs of constraints (flow_balance and demand_balance) to update for respective stage and scenario while executing'''
        #objective sense
        m.modelSense=GRB.MINIMIZE;    
        #define DVs
        x=m.addVras(self.states,self.states,lb=0,obj=self.distance,name='vent-flow'); #stage
        self.u=m.addVars(self.states,lb=0,name='vent-avail'); #stage
        y=m.addVras(self.states,lb=0,name='vent-unused'); #recourse
        z=m.addVras(self.states,lb=0,obj=self.penalty,name='vent-shortage'); #recourse
        self.theta=m.addVars(obj=1,name='ctg-approx')
        #add constraints
        m.addConstrs((self.u[s]-x.sum('*',s)+x.sum(s,'*')==0 for s in self.states),name="flow-balance")
        m.addConstrs((z[s]+self.u[s]-y[s]==0 for s in self.states),name="demand-balance")
        m.addConstrs((x.sum(s,'*')-y[s]<=0 for s in self.states),name="max-flow") 
        return m
        
    def forward_pass(self,path,iterNum): 
        if iterNum==1:
            for t in self.stages:
                self.fwd_model[t]=model(); #initialisae fwd model for each stage
        #solve m in each stage using previous stage state variables and a sample path
        for t in self.stages:
            for s in self.states:
                #update RHS
                self.fwd_model[t].getConstrByName("flow-balance[{}]".format(s)).setAttr(GRB.Attr.RHS, self.prevStageVent[t-1][s])                
                self.fwd_model[t].getConstrByName("demand-balance[{}]".format(s)).setAttr(GRB.Attr.RHS, self.demand[(t,path[t],s)])                
                #add cuts
            #update the model
            self.fwd_model[t].update()
            #optimize
            self.fwd_model[t].optimize()
            #store solution vals
            for s in self.states:                
                self.prevStageVent[t][s]=self.fwd_model[t].getVarByName("vent-avail[{}]".format(s)).x

    def backward_pass(self,iterNum):
        if iterNum==1:
            for t in self.stages:
                self.bwd_model[t]=model();
        #solve m in each stage using previous stage state variables and a sample path
        for t in self.stages:
            for s in self.states:
                self.bwd_model[t].getConstrByName("flow-balance[{}]".format(s)).setAttr(GRB.Attr.RHS, self.prevStageVent[t][s])                
            #add cuts
            for j in self.scenarios[t]:
                for s in self.states:
                    #update RHS
                    self.bwd_model[t].getConstrByName("demand-balance[{}]".format(s)).setAttr(GRB.Attr.RHS, self.demand[(t,j,s)])               
                #add cuts
                #update the model
                self.bwd_model[t].update()
                #optimize
                self.bwd_model[t].optimize()
                #store dual vals used for gen cuts
                self.alpha[iterNum,j,t]=self.bwd_model.objVal-sum(self.bwd_model.getConstrByName("flow_balance[{}]".format(s)).pi*self.prevStageVent[t-1][s] for s in self.states)        
    
    def execute_sddp(self):
        #FORWARD
        #generate a sample path i.e. select a j for each stage
        samplePath={} #storing sample path
        for i in self.stages:
            samplePath[i]=np.random.randint(1,len(self.scenarios[i]))
        self.forward_pass(samplePath)
        #BACKWARD
        'Resume work on adding cuts and linking forawrd pass together'
        
        
        
        
        
if __name__=='__main__':
    model=mssp(data)
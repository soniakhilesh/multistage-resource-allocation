#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:30:08 2020

@author: soni6
"""

from read_data import vent_data
from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt
class mssp:
    def __init__(self,data,maxiter):
        self.maxiter=maxiter;
        self.states=data.states; #list of string
        self.distance=data.distance; #dictionary, eg D={('Wisconsin,Newyork'):2300}
        self.inventory=data.inventory;#initial inventory, eg I={'Wisconsin':24}
        self.penalty=data.penalty; #penalty per unit of unmet demand
        self.demand=data.demand; #demand data for each stage, demand={('#stage(t),#node(scenario),state(name)'):450}
        self.stages=data.stages; #list of integers
        self.scenarios=data.scenarios; #dict with # scenarios in each stage, scenarios={stage:[1,2,3..]}
        #models
        #self.fwd_model={}, # a model for each stage which keeps getting updated with new cuts
        #self.bwd_model={} #a model for each stage which gets updated with new cuts, diff scenarios solved by updating RHS
        #storing solutions
        self.prevStageVent={0:data.inventory}
        for i in self.stages:
            self.prevStageVent[i]={};
        self.alpha={};
        self.rho={};
        self.fwdObjVal={};
        self.costFwdPass={};
        self.paths={}; #storing paths generated randomly
        self.piObjVals={};
        self.lb={}; 
        self.ub={};
        self.PI=0; #average perfect information cost
        self.xSol={}; #sddp x solution
        self.uSol={}; #sddp u solution
        self.scenObjVal={}; #storing cost after fixing state vars from sddp solution
        self.evpi={}; #storing evpi for each scenario

        
        
    def model(self):
        m=Model()
        '''don't add cuts in this function, just build initial model
        time/scenario index not needed
        change rhs of constraints (flow_balance and demand_balance) to update for respective stage and scenario while executing'''
        #objective sense
        m.modelSense=GRB.MINIMIZE;    
        #define DVs
        x=m.addVars(self.states,self.states,lb=0,obj=self.distance,name='vent-flow'); #stage
        u=m.addVars(self.states,lb=0,name='vent-avail'); #stage
        y=m.addVars(self.states,lb=0,name='vent-unused'); #recourse
        z=m.addVars(self.states,lb=0,obj=self.penalty,name='vent-shortage'); #recourse
        m.addVar(obj=1,lb=0,name='theta')
        #add constraints
        m.addConstrs((u[s]-x.sum('*',s)+x.sum(s,'*')==0 for s in self.states),name="flow-balance")
        m.addConstrs((z[s]+u[s]-y[s]==0 for s in self.states),name="demand-balance")
        m.addConstrs((x.sum(s,'*')-y[s]<=0 for s in self.states),name="max-flow") 
        m.setParam(GRB.Param.LogToConsole,0)
        m.update()
        return m
        
    def forward_pass(self,path,iterNum): 
        if iterNum==1:
            self.fwd_model={};
            for t in self.stages:
                self.fwd_model[t]=self.model(); #initialisae fwd model for each stage
        #solve m in each stage using previous stage state variables and a sample path
        for t in self.stages:
            for s in self.states:
                #update RHS
                self.fwd_model[t].getConstrByName("flow-balance[{}]".format(s)).setAttr(GRB.Attr.RHS, self.prevStageVent[t-1][s])                
                self.fwd_model[t].getConstrByName("demand-balance[{}]".format(s)).setAttr(GRB.Attr.RHS, self.demand[(t,path[t],s)])                
            #add cuts, just add for current iteration, 
            if iterNum>1 and t!=self.stages[-1]:
                self.fwd_model[t].addConstr((len(self.scenarios[t])*self.fwd_model[t].getVarByName('theta')>=sum(self.alpha[iterNum-1,j,t+1] for j in self.scenarios[t])+
                sum(self.rho[iterNum-1,j,t+1,s]*self.fwd_model[t].getVarByName('vent-avail[{}]'.format(s)) for s in self.states for j in self.scenarios[t]))
                ,name="cut-{}".format(iterNum))
            #update the model
            self.fwd_model[t].update()
            #optimize
            self.fwd_model[t].optimize()
            #store current solutions
            for s1 in self.states:
                for s2 in self.states:
                    self.xSol[t,s1,s2]=round(self.fwd_model[t].getVarByName('vent-flow[{},{}]'.format(s1,s2)).x,2)
                self.uSol[t,s1]=round(self.fwd_model[t].getVarByName('vent-avail[{}]'.format(s1)).x,2)
            self.fwdObjVal[(iterNum,t)]=self.fwd_model[t].objVal;
            #store solution vals
            for s in self.states:                
                self.prevStageVent[t][s]=self.fwd_model[t].getVarByName("vent-avail[{}]".format(s)).x

    def backward_pass(self,iterNum):
        if iterNum==1:
            self.bwd_model={}
            for t in self.stages:
                self.bwd_model[t]=self.model();
        #solve m in each stage and for each scenario
        for t in self.stages:
            #in reverse order
            t=self.stages[-t];
            #update RHS
            for s in self.states:
                self.bwd_model[t].getConstrByName("flow-balance[{}]".format(s)).setAttr(GRB.Attr.RHS, self.prevStageVent[t-1][s])                
                self.bwd_model[t].update()
            #add cuts
            if t!=self.stages[-1]:
                self.bwd_model[t].addConstr((len(self.scenarios[t])*self.bwd_model[t].getVarByName('theta')>=sum(self.alpha[iterNum,j,t+1] for j in self.scenarios[t])+
                sum(self.rho[iterNum,j,t+1,s]*self.bwd_model[t].getVarByName('vent-avail[{}]'.format(s)) for s in self.states for j in self.scenarios[t]))
                ,name="cut-{}".format(iterNum))
                self.bwd_model[t].update()
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
                for s in self.states:
                    self.rho[iterNum,j,t,s]=self.bwd_model[t].getConstrByName("flow-balance[{}]".format(s)).pi
                #store alpha vals
                if t!=self.stages[0]:
                    self.alpha[iterNum,j,t]=self.bwd_model[t].objVal-sum(self.rho[iterNum,j,t,s]*self.prevStageVent[t-1][s] for s in self.states)        
    
    def execute_sddp(self,epsilon=0.01):
        convergence=False;
        iterNum=0;
        while convergence==False:
            iterNum+=1; 
            #FORWARD-generate a sample path i.e. select a j for each stage
            samplePath={} #storing sample path
            for i in self.stages:
                samplePath[i]=np.random.randint(1,len(self.scenarios[i]))
            self.paths[iterNum]=samplePath;
            self.forward_pass(samplePath,iterNum)
            #find bounds
            lb=self.fwdObjVal[iterNum,1];
            costfwd=0;
            for t in self.stages:
                costfwd+=self.fwdObjVal[iterNum,t]-self.fwd_model[t].getVarByName('theta').x;
            self.costFwdPass[iterNum]=costfwd;
            ub=(1/iterNum)*sum(self.costFwdPass[iteration] for iteration in range(1,iterNum+1))
            print(iterNum)
            print('Lower Bound {}'.format(lb),'Upper Bound {}'.format(ub)) 
            self.lb[iterNum]=lb;
            self.ub[iterNum]=ub;
            #test convergence criterion
            if iterNum!=1:
                if abs(ub-lb)<epsilon or iterNum==self.maxiter:   
                    print(ub/lb)
                    break;
            #BACKWARD
            self.backward_pass(iterNum)



    def perfect_info(self,path):
        piModel=Model()
        '''don't add cuts in this function, just build initial model
        time/scenario index not needed
        change rhs of constraints (flow_balance and demand_balance) to update for respective stage and scenario while executing'''
        #objective sense
        piModel.modelSense=GRB.MINIMIZE;    
        #define DVs
        x=piModel.addVars(self.stages,self.states,self.states,lb=0,obj=self.distance,name='vent-flow'); #stage
        u=piModel.addVars(self.stages,self.states,lb=0,name='vent-avail'); #stage
        y=piModel.addVars(self.stages,self.states,lb=0,name='vent-unused'); #recourse
        z=piModel.addVars(self.stages,self.states,lb=0,obj=self.penalty,name='vent-shortage'); #recourse
        #add constraints
        #flow:
        piModel.addConstrs((u[t+1,s]-x.sum(t+1,'*',s)+x.sum(t+1,s,'*')-u[t,s]==0 for s in self.states for t in self.stages[:-1]),name="flow-balance")
        #flow: inventory con
        piModel.addConstrs((u[1,s]==self.prevStageVent[0][s] for s in self.states),name="flow-balance-inv")
        #demand
        piModel.addConstrs((z[t,s]+u[t,s]-y[t,s]==self.demand[t,path[t],s] for s in self.states for t in self.stages),name="demand-balance")
        #max-flow
        piModel.addConstrs((x.sum(t,s,'*')-y[t,s]<=0 for s in self.states for t in self.stages),name="max-flow") 
        piModel.setParam(GRB.Param.LogToConsole,0)
        piModel.update()
        return piModel
        
    def solvePI(self):
        samplePath={} #storing sample path
        numIterPI=0
        while True:
            numIterPI+=1;
            samplePath=self.paths[numIterPI]
            piModel=self.perfect_info(samplePath);
            #optimize
            piModel.optimize()
            self.piObjVals[numIterPI]=piModel.objVal
            if numIterPI==self.maxiter:
                print('Mean PI value is ',np.mean(list(self.piObjVals.values())))
                self.PI=np.mean(list(self.piObjVals.values()))
                break
 
    def plotBoundsSDDP(self):
        plt.figure(figsize=(14,10),dpi=80)
        plt.plot(list(self.lb.keys()),list(self.lb.values()),label='Lower Bound');
        plt.plot(list(self.ub.keys()),list(self.ub.values()),label='Upper Bound');
        #plt.plot(list(self.ub.keys()),[self.PI for i in list(self.ub.keys())],label='Perfect Info');    
        plt.legend(fontsize=16)
        plt.xlabel('Iteration',fontsize=16)
        plt.ylabel('Objective Value',fontsize=16)
        plt.title('SDDP convergence',fontsize=16)
        plt.savefig('sddp-convergence.png')

    def cal_EVPI(self):
        #call model and fix first stage decisions
        evpiPercentage={}
        for scenNum in list(self.paths.keys()):
            path=self.paths[scenNum]
            scenModel=Model()
            #objective sense
            scenModel.modelSense=GRB.MINIMIZE;    
            #define DVs
            x=scenModel.addVars(self.stages,self.states,self.states,lb=0,obj=self.distance,name='vent-flow'); #stage
            u=scenModel.addVars(self.stages,self.states,lb=0,name='vent-avail'); #stage
            y=scenModel.addVars(self.stages,self.states,lb=0,name='vent-unused'); #recourse
            z=scenModel.addVars(self.stages,self.states,lb=0,obj=self.penalty,name='vent-shortage'); #recourse
            #add constraints
            #flow:
            #scenModel.addConstrs((u[t+1,s]-x.sum(t+1,'*',s)+x.sum(t+1,s,'*')-u[t,s]==0 for s in self.states for t in self.stages[:-1]),name="flow-balance")
            #flow: inventory con
            #scenModel.addConstrs((u[1,s]==self.prevStageVent[0][s] for s in self.states),name="flow-balance-inv")
            #demand
            scenModel.addConstrs((z[t,s]+u[t,s]-y[t,s]==self.demand[t,path[t],s] for s in self.states for t in self.stages),name="demand-balance")
            #max-flow
            scenModel.addConstrs((x.sum(t,s,'*')-y[t,s]<=0 for s in self.states for t in self.stages),name="max-flow") 
            scenModel.setParam(GRB.Param.LogToConsole,0)
            scenModel.update()
            'fix first stage decision vars'
            scenModel.addConstrs((x[t,s1,s2]==self.xSol[t,s1,s2] for t in self.stages for s1 in self.states for s2 in self.states),name='fix-x')
            scenModel.addConstrs((u[t,s]==self.uSol[t,s] for t in self.stages for s in self.states),name='fix-u')
            scenModel.update()
            scenModel.optimize()
            self.scenObjVal[scenNum]=scenModel.objVal
            self.evpi[scenNum]=self.scenObjVal[scenNum]-self.piObjVals[scenNum]
            evpiPercentage[scenNum]=100*self.evpi[scenNum]/self.scenObjVal[scenNum];
        print('Mean EVPI is {}'.format(np.mean(list(self.evpi.values()))))
        print('Mean EVPI Percentage is {}'.format(np.mean(list(evpiPercentage.values()))))
        
            
if __name__=='__main__':
    data=vent_data()
    data.gen_data()
    model=mssp(data,200);
    model.execute_sddp()
    model.solvePI()
    #model.plotBoundsSDDP()
    model.cal_EVPI()
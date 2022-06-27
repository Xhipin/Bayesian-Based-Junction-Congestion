import pyAgrum as gum
from datetime import datetime
import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt


class FinalBayesGum:
    
    def __init__(self, fileNamecsv: str, fileNamejoint: str):
        self._jointNames = []
        self._jointNum = []
        self._bayesNodes = []
        self._redTimes = []
        self._cycleTimes = []
        self._timeLag = 1
        self._sampleTime = 0
        self._fileNamecsv = fileNamecsv
        self._fileNamejoint = fileNamejoint
        self._datasetLocs =  os.listdir(self._fileNamecsv)
        self._bn = gum.BayesNet()
        self._data = []
        
    def readJoint(self):
        with open(
                self._fileNamejoint) \
            as nameFile:
                
            readList = csv.reader(nameFile, delimiter = ',')
            
            jCount = 0
            for joint in readList:
                if jCount == 0:
                    temp = ''.join(joint)
                    temp = temp[3:]
                    self._jointNames.append(temp)
                    jCount += 1
                    continue
                
                self._jointNames.append(''.join(joint))
                
        self._jointNames.sort()
        jCount -= 1
        jPrevCount = 0
        j = 0
        for direct in self._datasetLocs:
            if j < len(self._jointNames) and direct.find(self._jointNames[j]) == -1:
                j += 1
                self._jointNum.append(jCount - jPrevCount)
                jPrevCount = jCount
            
            jCount += 1
        
        self._jointNum.append(jCount - jPrevCount)
        
    def edgesSpecific(self):
        nodesPerTimeLag = len(self._bayesNodes)//(self._timeLag + 1)
        for t in range(self._timeLag + 1):
            # 309 - 310
            self._bn.addArc(3 + t * nodesPerTimeLag, 6 + t * nodesPerTimeLag)
            self._bn.addArc(2 + t * nodesPerTimeLag, 7 + t * nodesPerTimeLag)
            
            # 310 - 311
            self._bn.addArc(9 + t * nodesPerTimeLag, 12 + t * nodesPerTimeLag)
            self._bn.addArc(8 + t * nodesPerTimeLag, 13 + t * nodesPerTimeLag)
            
             # 311 - 622
            self._bn.addArc(15 + t * nodesPerTimeLag, 32 + t * nodesPerTimeLag)
            self._bn.addArc(14 + t * nodesPerTimeLag, 33 + t * nodesPerTimeLag)
            
             # 622 - 326
            self._bn.addArc(35 + t * nodesPerTimeLag, 24 + t * nodesPerTimeLag)
            self._bn.addArc(34 + t * nodesPerTimeLag, 25 + t * nodesPerTimeLag)
            
             # 326 - 325
            self._bn.addArc(31 + t * nodesPerTimeLag, 20 + t * nodesPerTimeLag)
            self._bn.addArc(30+ t * nodesPerTimeLag, 21 + t * nodesPerTimeLag)
        

        
    def constructBayesNet(self):
        self.constructBayesNodes()
        lbl = ["0", "1", "2", "3", "4", "5"]
        for name in self._bayesNodes:
            vx = gum.LabelizedVariable(name, 'a labelized variable', lbl)
            self._bn.add(vx)
        
        nodesPerTimeLag = len(self._bayesNodes)//(self._timeLag + 1)
        totalNum = 0
        # Create backbone network
        for i in range(self._timeLag + 1):
            for n in self._jointNum:
                
                for j in range(n):
                    for k in range(n):
                        fromInd = totalNum + 2*j
                        toInd = totalNum + 2*k + 1
                        
                        self._bn.addArc(fromInd, toInd)
                        
                totalNum += n*2
        
        # Create timelag edges 
        for i in range(nodesPerTimeLag):
            for j in range(self._timeLag):
                outInd = (j+1)*nodesPerTimeLag + i
                inInd = j*nodesPerTimeLag + i
                self._bn.addArc(outInd, inInd)
        self.edgesSpecific()
        print(self._bn)
        
    def readSampleTime(self):
        with open(
                self._fileNamecsv + '/' + self._datasetLocs[0]) \
            as nameFile:
                readList = csv.reader(nameFile, delimiter = ',')

                i = 0
                for row in readList:
                    i += 1
                    if i == 5:
                        sampInterval = row[0]
                        break
                       
        firstSampleMin = int(sampInterval[3:5])
        secondSampleMin = int(sampInterval[-2:])
        self._sampleTime = 60*(secondSampleMin - firstSampleMin)
    
    
    def findMaxTimeLag(self):
        self.readSampleTime()
        numDataset = len(self._datasetLocs)
        tempTimeLag = 1
        for i in range(numDataset):
            with open(
                self._fileNamecsv + '/' + self._datasetLocs[i]) \
            as nameFile:
                
                readList = csv.reader(nameFile, delimiter = ',')

                j = 0
                for row in readList:
                    if j == 1:
                        sampInterval = row[-2:]
                        break
                    j+=1
                    
            self._cycleTimes.append(int(sampInterval[0]))
            
            currRedTime = int(sampInterval[0])-int(sampInterval[1])
            self._redTimes.append(currRedTime)
            
            tempTimeLag = 1 + currRedTime // self._sampleTime
            if tempTimeLag > self._timeLag:
                self._timeLag = tempTimeLag
        
    
    def constructBayesNodes(self):
        
        for i in range(self._timeLag + 1):
            for j in range(len(self._jointNames)):
                for k in range(self._jointNum[j]):
                    for s in ["FROM_", "TO_"]:
                        tempStr = "JOINT_" + self._jointNames[j] + \
                            "_CURRENT_" + s + str(k+1) + "_TIMELAG_" + str(i)
                            
                        self._bayesNodes.append(tempStr)
    
    def discretizeData(self):
        
        intervals = list(range(0,300,50))
        for l in range(3):
            if l == 0:
                data = self._morningData
            elif l==1:
                data = self._noonData
            elif l == 2:
                data = self._nightData
                
            sizeData = data.shape
            auxData = data.astype(str)
        
            for j in range(sizeData[0]):
                for k in range(sizeData[1]):
                    for i in range(len(intervals)-1):
                        if data.iloc[j,k] < intervals[i+1] and data.iloc[j,k] >= intervals[i] : 
                            auxData.iloc[j,k] = str(i)
                            
                    if data.iloc[j,k]>=intervals[-1]:
                        auxData.iloc[j,k] = str(len(intervals)-1)
                    
                    if np.isnan(data.iloc[j,k]) :
                        auxData.iloc[j,k] = '?'
                

            if l == 0: 
                self._morningData = auxData
                
            elif l==1:
                self._noonData = auxData
                
            elif l == 2:
                self._nightData = auxData
            
        self._morningData.to_csv('dataset/refinedData/morningData.csv', encoding = 'utf-8', index = False)
        self._noonData.to_csv('dataset/refinedData/noonData.csv', encoding = 'utf-8', index = False)
        self._nightData.to_csv('dataset/refinedData/nightData.csv', encoding = 'utf-8', index = False)
                    
    def organizeData(self):
       
        sizeNodes = len(self._bayesNodes)
        dayTimes = []
        dayTimes.append( 12 - self._timeLag)
        dayTimes.append( 8 - self._timeLag)
        dayTimes.append( 16 - self._timeLag)
        
        self._morningData = pd.DataFrame(np.empty([dayTimes[0],sizeNodes]), columns = self._bayesNodes)
        self._noonData = pd.DataFrame(np.empty([dayTimes[1],sizeNodes]), columns = self._bayesNodes)
        self._nightData = pd.DataFrame(np.empty([dayTimes[2],sizeNodes]), columns = self._bayesNodes)
        
        sizeJoint = len(self._jointNum)
        nodesPerTimeLag = len(self._bayesNodes)//(self._timeLag + 1)
        
        nodeCount = -2
        for i in range(sizeJoint):
            jointData = self._data[i]
            currNum = self._jointNum[i]
            for j in range(currNum):
                nodeCount += 2
                for t in range(self._timeLag + 1):
                   
                    
                    for k in range(3):
                        dayTimeSub = jointData[k]
                        for l in range(dayTimes[k]):
                        
                            innerPoint = l - t + self._timeLag
                            lSub = dayTimeSub[innerPoint,j*currNum \
                                              :(j+1)*currNum]
                            lSub2 = dayTimeSub[innerPoint, j::currNum]
                            locName1 = self._bayesNodes[t*nodesPerTimeLag + \
                                                       nodeCount]
                            locName2 = self._bayesNodes[t*nodesPerTimeLag + \
                                                       nodeCount+1]
                            if k == 0:
                                
                                self._morningData.loc[l, locName1] = np.sum(lSub)
                                self._morningData.loc[l, locName2] = np.sum(lSub2)
                                
                            elif k == 1:
                                self._noonData.loc[l, locName1] = np.sum(lSub)
                                self._noonData.loc[l, locName2] = np.sum(lSub2)
                            elif k == 2:
                                self._nightData.loc[l, locName1] = np.sum(lSub)
                                self._nightData.loc[l, locName2] = np.sum(lSub2)
                                
        self.discretizeData()  
                    
    
    def readData(self):
        numDataset = len(self._datasetLocs)
        
        currJoint = 0
        auxData = [None]*3
        for i in range(numDataset):
            currDatasetName = self._fileNamecsv + '/' + self._datasetLocs[i]
            tempDf = pd.read_csv(currDatasetName)
            
            auxMorning =  tempDf.iloc[3:15, 1:1+self._jointNum[currJoint]].to_numpy()
            auxNoon = tempDf.iloc[16:24, 1:1+self._jointNum[currJoint]].to_numpy()
            auxNight = tempDf.iloc[25:, 1:1+self._jointNum[currJoint]].to_numpy()
            
            auxMorning = auxMorning.astype(float)
            auxNoon = auxNoon.astype(float)
            auxNight = auxNight.astype(float)
            
            if i == sum(self._jointNum[0:currJoint]):
                auxData[0] = auxMorning
                auxData[1] = auxNoon
                auxData[2] = auxNight
                
            else:
                auxData[0] = np.append(auxData[0],auxMorning, axis = 1)
                auxData[1] = np.append(auxData[1],auxNoon, axis = 1)
                auxData[2] = np.append(auxData[2],auxNight, axis = 1)
            
            
            if sum(self._jointNum[0:currJoint + 1]) == i + 1:
                currJoint += 1
                self._data.append(auxData)
                auxData = [None]*3
        
        self.organizeData()
          
    
    def estimateCPD(self):
        self.readData()
        src = self._bn
        learnerMorning = gum.BNLearner('dataset/refinedData/morningData.csv',src, ["?"])
        learnerNoon = gum.BNLearner('dataset/refinedData/noonData.csv',src, ["?"])
        learnerNight = gum.BNLearner('dataset/refinedData/nightData.csv',src, ["?"])
        
        epsl = 1e-3
        
        learnerMorning.setVerbosity(True)
        learnerNoon.setVerbosity(True)
        learnerNight.setVerbosity(True)
        
        learnerMorning.useEM(epsl)
        learnerMorning.useAprioriSmoothing()
        print(learnerMorning)
        self._bnMorning = learnerMorning.learnParameters(src.dag())
        
        learnerNoon.useEM(epsl)
        learnerNoon.useAprioriSmoothing()
        print(learnerNoon)
        self._bnNoon = learnerNoon.learnParameters(src.dag())
        
        learnerNight.useEM(epsl)
        learnerNight.useAprioriSmoothing()
        print(learnerNight)
        self._bnNight = learnerNight.learnParameters(src.dag())
        
        plt.figure()
        plt.plot(np.arange(1, 1 + learnerMorning.nbrIterations()), learnerMorning.history(), label = "Morning")
        plt.title("Error During EM Iterations")
        
        
        plt.plot(np.arange(1, 1 + learnerNoon.nbrIterations()), learnerNoon.history(), label = "Noon")

        plt.plot(np.arange(1, 1 + learnerNight.nbrIterations()), learnerNight.history(), label = "Night")
        plt.legend()
        plt.semilogy()
        plt.show()
    
    def getJointNames(self):
        return self._jointNames
    
    def settleInferencewithData(self, dayTime: str, evs: dict()):
     if dayTime == 'morning':
            src = self._bnMorning

     elif dayTime == 'noon':
            src = self._bnNoon

     elif dayTime == 'night':
            src = self._bnNight
        
     else:
            return
        
     evs = dict()
           
    
     sampler = gum.GibbsSampling(src)
     if len(evs) != 0:
       sampler.setEvidence(evs)
     sampler.setMaxTime(50)
     sampler.setEpsilon(1e-2)
     sampler.makeInference()
   
     return sampler
        
    
    def settleInference(self, dayTime: str, sampleNum: int):
         if dayTime == 'morning':
            src = self._bnMorning
            data = self._morningData
         elif dayTime == 'noon':
            src = self._bnNoon
            data = self._noonData
         elif dayTime == 'night':
            src = self._bnNight
            data = self._nightData
         else:
            return
        
         nodesPerTimeLag = len(self._bayesNodes)//(self._timeLag + 1)
         evs = dict()
         sh = data.shape
         sample = sampleNum
         for i in range(nodesPerTimeLag):
             nodeNum = self._timeLag * nodesPerTimeLag + i
             
             if sampleNum > (sh[0]-1):
                 nodeNum -= (sampleNum - sh[0] + 1) * nodesPerTimeLag
                 sample = sh[0] - 1
             
             if data.iloc[sample, nodeNum] !='?':
                 evs[self._bayesNodes[nodeNum]] = \
                 data.iloc[sample, nodeNum]
             
                
         
         sampler = gum.GibbsSampling(src)
         if len(evs) != 0:
            sampler.setEvidence(evs)
         sampler.setMaxTime(50)
         sampler.setEpsilon(1e-2)
         sampler.makeInference()
        
         return sampler
    
    def getInferenceSingle(self, dayTime: str, sampleNum: int, \
                    jointList: list[int, int, int], timeLag = 0):
        
       
        sampler = self.settleInference(dayTime, sampleNum)
        
        jointIdx = jointList[0]
        jointNum = jointList[1] - 1
        fromOrTo = jointList[2]
        
        nodesPerTimeLag = len(self._bayesNodes)//(self._timeLag + 1)
        
        nodeNum = timeLag * nodesPerTimeLag + fromOrTo + \
        2*(sum(self._jointNum[0:jointIdx]) +jointNum) 
        
        mAP = sampler.posterior(nodeNum).tolist()
            
        idx = mAP.index(max(mAP))
        lbl = ["none", "few", "normal", "mild", "congestion", "serious"]
    
        return lbl[idx]
    
    def getInferenceWhole(self, dayTime: str, sampleNum: int, \
                     timeLag = 0):
        
       
        sampler = self.settleInference(dayTime, sampleNum)
        nodesPerTimeLag = len(self._bayesNodes)//(self._timeLag + 1)
        lbl = ["none", "few", "normal", "mild", "congestion", "serious"]
        
        
        out = dict()
        for n in range(nodesPerTimeLag):
            nodeNum = n + timeLag * nodesPerTimeLag
            mAP = sampler.posterior(nodeNum).tolist()
            
            idx = mAP.index(max(mAP))
            out[self._bayesNodes[n]] = lbl[idx]
           
        return out
    
    def setTimeLag(self, timeLag):
        self._timeLag = timeLag
        
        
        
    def getInferencewithData(self, dayTime: str, evs = dict(), \
                     timeLag = 0):
        
        sampler = self.settleInferencewithData(dayTime, evs)
        nodesPerTimeLag = len(self._bayesNodes)//(self._timeLag + 1)
        lbl = ["none", "few", "normal", "mild", "congestion", "serious"]
        
        
        out = dict()
        for n in range(nodesPerTimeLag):
            nodeNum = n + timeLag * nodesPerTimeLag
            mAP = sampler.posterior(nodeNum).tolist()
            
            idx = mAP.index(max(mAP))
            out[self._bayesNodes[n]] = lbl[idx]
           
        return out
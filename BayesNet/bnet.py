from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization as EM
from datetime import datetime
import numpy as np
import pandas as pd
import csv
import os



class FinalBayes:
    
    def __init__(self, fileNamecsv, fileNamejoint):
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
        self._bn = BayesianNetwork()
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
    
    def constructBayesNet(self):
        self.constructBayesNodes()
        self._bn.add_nodes_from(self._bayesNodes)
        
        nodesPerTimeLag = len(self._bayesNodes)//(self._timeLag + 1)
        totalNum = 0
        # Create backbone network
        for i in range(self._timeLag + 1):
            for n in self._jointNum:
                
                for j in range(n):
                    for k in range(n):
                        fromInd = totalNum + 2*j
                        toInd = totalNum + 2*k + 1
                        
                        self._bn.add_edge(self._bayesNodes[fromInd], self._bayesNodes[toInd])
                        
                totalNum += n*2
        
        # Create timelag edges 
        for i in range(nodesPerTimeLag):
            for j in range(self._timeLag):
                outInd = (j+1)*nodesPerTimeLag + i
                inInd = j*nodesPerTimeLag + i
                self._bn.add_edge(self._bayesNodes[outInd], self._bayesNodes[inInd])
            
        
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
        self.findMaxTimeLag()
        
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
        
            for j in range(sizeData[0]):
                for k in range(sizeData[1]):
                    for i in range(len(intervals)-1):
                        if data.iloc[j,k] < intervals[i+1] and data.iloc[j,k] >= intervals[i] : 
                            data.iloc[j,k] = i
                            
                    if data.iloc[j,k]>=intervals[-1]:
                        data.iloc[j,k] = len(intervals)-1
            data = data.to_numpy()
                
            nanIdx = np.isnan(data)
            auxData = data
            auxData[~nanIdx] = data[~nanIdx].astype(np.int32)
            auxData [nanIdx] = np.nan
            if l == 0: 
                self._morningData = pd.DataFrame(auxData, columns = self._bayesNodes)
                
            elif l==1:
                self._noonData = pd.DataFrame(auxData, columns = self._bayesNodes)
                
            elif l == 2:
                self._nightData = pd.DataFrame(auxData, columns = self._bayesNodes)
        
                    
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
        
        estMorning = EM(self._bn, self._morningData)
        # estNoon = EM(self._bn, self._noonData)
        # estNight = EM(self._bn, self._nightData)
        
        self._CPDMorning = estMorning.get_parameters()
        # self._CPDNoon = estNoon.get_parameters()
        # self._CPDNight = estNight.get_parameters()
    


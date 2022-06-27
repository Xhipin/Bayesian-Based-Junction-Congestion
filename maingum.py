import BayesNet

filecsv = 'dataset\csv'
filejoint = 'dataset/joints/joint_names.csv'
bnw = BayesNet.FinalBayesGum(filecsv, filejoint)
bnw.readJoint()
bnw.findMaxTimeLag()
bnw.constructBayesNet()
bnw.estimateCPD()

# 8.45
infrnce8_45 = bnw.getInferenceWhole(dayTime = 'morning', sampleNum = 6)

# 13.15
infrnce13_45 = bnw.getInferenceWhole(dayTime = 'noon', sampleNum = 4)

# 17.45
infrnce17_45 = bnw.getInferenceWhole(dayTime = 'night', sampleNum = 6)

lbl = ["none", "few", "normal", "mild", "congestion", "serious"]
x = list(infrnce17_45.keys())
num17_45 = dict()


for key in x:
    num17_45[key] = str(lbl.index(infrnce17_45[key]))
    
# 18.00 
infrnce18_00 = bnw.getInferencewithData(dayTime = 'night', evs = num17_45)

x = list(infrnce18_00.keys())
num18_00 = dict()


for key in x:
    num18_00[key] = str(lbl.index(infrnce18_00[key]))

# 18.15
infrnce18_15 = bnw.getInferencewithData(dayTime = 'night', evs = num18_00)

# 18.00 max timeLag 2
bnw = BayesNet.FinalBayesGum(filecsv, filejoint)
bnw.readJoint()
bnw.setTimeLag(2)
bnw.constructBayesNet()
bnw.estimateCPD()
infrnce18_00 = bnw.getInferenceWhole(dayTime = 'night', sampleNum = 6)




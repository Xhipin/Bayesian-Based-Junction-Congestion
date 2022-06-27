import BayesNet

filecsv = 'D:\Google Drive\EE5420\Final\Code\EE5420 Final Take Home Exam\dataset\csv'
filejoint = 'D:\Google Drive\EE5420\Final\Code\EE5420 Final Take Home Exam\dataset/joints/joint_names.csv'
bnw = BayesNet.FinalBayes(filecsv, filejoint)
bnw.readJoint()
bnw.constructBayesNet()
bnw.estimateCPD()
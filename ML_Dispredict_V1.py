#!/usr/bin/env python
# coding: utf-8

# Author: Md Wasi Ul Kabir
# Date: April 1 2021
Description="Dispredict TrainCV"
TestScript=True  ## Testing script with 100 samples to make sure there is no bug.


CVTrainOrTest=False  # True For CV Result from Training set(Estimator lists) and False for testing independent test set.


validation=False

threshold=0.5

if CVTrainOrTest:
	N_jobs=7
else:
	N_jobs=-1
###############################################################################
#                              Importing Libraries                            #
###############################################################################

import time
import numpy as np
import pandas as pd
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import shutil
import joblib
import seaborn as sns
import os
import sys
from datetime import datetime
import socket
import xgboost as xgb
import pathlib 

from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV ,cross_val_predict
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek ,SMOTEENN 
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import RUSBoostClassifier


from mlxtend.classifier import StackingCVClassifier 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
import subprocess

###############################################################################
#                              Initialize Parameters                          #
###############################################################################
warnings.simplefilter('ignore')



os.system('clear') 
np.random.seed(100)
# Save Environment if it does not exists.
if not os.path.exists("../environment.yaml"):
	bashCommand='conda env  export --no-builds > environment.yaml'
	output = subprocess.check_output(['bash','-c', bashCommand])

#Assign Seed
start = time.time()
dtime= str(datetime.now().strftime('%d_%m_%Y_%H_%M')) 
Output_file=open("Output_"+ dtime +".txt","w")
Compare_file=open("Compare_"+ dtime +".txt","w")
pathlib.Path("./save_files").mkdir(parents=True, exist_ok=True)    


path = os.getcwd()
print(path,file=Output_file)
host_name = socket.gethostname() 
print("Hostname :  ",host_name,file=Output_file) 

print(Description,file=Output_file)
print(Description,file=Compare_file)
###############################################################################
#                                    Methods                                  #
###############################################################################

def Predict_Result(y, predicted,predicted_proba,DatatsetName=""):
   

	
	confusion = confusion_matrix(y, predicted)

	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]


	# Specificity
	SPE_cla = (TN/float(TN+FP))*100 

	# False Positive Rate
	FPR = (FP/float(TN+FP))

	#False Negative Rate (Miss Rate)
	FNR = (FN/float(FN+TP))

	#Balanced Accuracy
	ACC_Bal = 0.5*((TP/float(TP+FN))+(TN/float(TN+FP)))*100

	# compute MCC
	MCC_cla = matthews_corrcoef(y, predicted)
	F1_cla = f1_score(y, predicted)
	PREC_cla = precision_score(y, predicted)*100
	REC_cla = recall_score(y, predicted)*100
	Accuracy_cla = accuracy_score(y, predicted)*100


	ROC_auc_score= roc_auc_score(y, predicted_proba)*100
	PR_auc_score= average_precision_score(y, predicted_proba)*100
	sw = ((REC_cla/100)+(SPE_cla/100)-1)*100



	print(f"SN (%), {REC_cla:.2f}")
	print(f"SP (%), {SPE_cla:.2f}")
	print(f"Sw (%), {sw:.2f}")
	print(f"BACC (%), {ACC_Bal:.2f}")
	print(f"MCC, {MCC_cla:.3f}")
	print(f"ACC (%), {Accuracy_cla:.2f}") 
	print(f"FPR, {FPR:.3f}")
	print(f"FNR, {FNR:.3f}")
	print(f"PR (%), {PREC_cla:.2f}")
	print(f"F1-score, {F1_cla:.3f}") 
	print(f"ROCAUC (%), {ROC_auc_score:.2f}")
	print(f"PRAUC (%), {PR_auc_score:.2f}")



	print(f"SN (%), {REC_cla:.2f}",file=Output_file)
	print(f"SP (%), {SPE_cla:.2f}",file=Output_file)
	print(f"Sw (%), {sw:.2f}",file=Output_file)
	print(f"BACC (%), {ACC_Bal:.2f}",file=Output_file)
	print(f"MCC, {MCC_cla:.3f}",file=Output_file)
	print(f"ACC (%), {Accuracy_cla:.2f}",file=Output_file) 
	print(f"FPR, {FPR:.3f}",file=Output_file)
	print(f"FNR, {FNR:.3f}",file=Output_file)
	print(f"PR (%), {PREC_cla:.2f}",file=Output_file)
	print(f"F1-score, {F1_cla:.3f}",file=Output_file) 
	print(f"ROCAUC (%), {ROC_auc_score:.2f}",file=Output_file)
	print(f"PRAUC (%), {PR_auc_score:.2f}",file=Output_file)





 
	#Improvement in test set
	#
	if (DatatsetName=="Disprot228"):
			SROC_auc= 81.00
			SPR_auc= 72.20 
			SMCC= 50.00 		
			SSw= 45.20 	

			print("\n\n", DatatsetName)
			print("Imp(%) ROCAUC ,",     ((ROC_auc_score-SROC_auc)/SROC_auc)*100       ,file=Compare_file)
			print("Imp(%) PRAUC,",       ((PR_auc_score-SPR_auc)/SPR_auc)*100     ,file=Compare_file)
			print("Imp(%) MCC,",         ((MCC_cla-SMCC)/SMCC)*100                      ,file=Compare_file)
			print("Imp(%) Sw,",          (( (REC_cla+SPE_cla-1) -SSw)/SSw) *100         ,file=Compare_file)

			print("Imp(%) ROCAUC ,",     ((ROC_auc_score-SROC_auc)/SROC_auc)*100       )
			print("Imp(%) PRAUC,",       ((PR_auc_score-SPR_auc)/SPR_auc)*100      )
			print("Imp(%) MCC,",         ((MCC_cla-SMCC)/SMCC)*100                     )
			print("Imp(%) Sw,",          (( (REC_cla+SPE_cla-1) -SSw)/SSw) *100        )
			


def thr_to_mcc(thr, Y_test, predictions):
	 return round(matthews_corrcoef(Y_test, (predictions[:,1] >= thr).astype(bool)) ,3)

def thr_to_sw(thr, Y_test, predictions):
	
	predicted=(predictions[:,1] >= thr).astype(bool)
	confusion = confusion_matrix(Y_test, predicted)

	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]

	REC_cla = round(recall_score(Y_test, predicted), 3)
	SPE_cla =  round((TN/float(TN+FP)) , 3) 
	sw=(REC_cla+SPE_cla-1)

	return round(sw ,3)

def thr_to_accuracy(thr, Y_test, predictions):
		
	predicted=(predictions[:,1] >= thr).astype(bool)
	confusion = confusion_matrix(Y_test, predicted)

	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	 #Balanced Accuracy
	ACC_Bal = 0.5*((TP/float(TP+FN))+(TN/float(TN+FP)))

	return round(ACC_Bal,3)




###############################################################################
#                              Dataset Preprocess                             #
###############################################################################


# Read from the file - first feature directory and second number of processor
with open('input_path.txt') as f:
	zero_line = f.readline()
	first_line = f.readline()
	second_line = f.readline()
	third_line = f.readline()
	fourth_line = f.readline()
   
# Number_of_processor=int(zero_line)
# print("Number of Processor ", Number_of_processor)   
train_df1 = pd.read_csv(zero_line.strip()) #,header=None
print("\nTrain Feature File Location: \n",zero_line)
print("train_df1 Shape:" ,train_df1.shape)

print("\nTrain Feature File Location: \n",zero_line,file=Output_file)
print("train_df1 Shape:" ,train_df1.shape,file=Output_file)

print(train_df1.head(3))

Test1Dataset=first_line.strip()

test_df1 = pd.read_csv(second_line.strip()) #,header=None
print("Test 1 Feature File Location: \n",second_line)
print("test_df1 Shape:" ,test_df1.shape)

print("Test 1 Feature File Location: \n",second_line,file=Output_file)
print("test_df1 Shape:" ,test_df1.shape,file=Output_file)

print(test_df1.head(3))

# Test2Datset=fourth_line.strip()
# if(Test2Datset)
# print("Test 2 Feature File Location: \n",third_line)
# test_df2 = pd.read_csv(third_line.strip(), header=None, low_memory=False)
# print("test_df2 Shape:" ,test_df2.shape)
# print("Test 3 Feature File Location: \n",fourth_line)
# test_df3 = pd.read_csv(fourth_line.strip(), header=None, low_memory=False)
# print("test_df3 Shape:" ,test_df3.shape)

######################################################################################################
# /home/mkabir3/Research/AIRBP/exp_84_Second_Review_Avdesh/exp_6_Combined_features/NewATHFull.csv

# train_df2 = pd.read_csv("/home/mkabir3/Research/12_DNABinding_Restart/4_Windowing/PDNA_62/feature_file_11.csv", header=None, low_memory=False)
# print("train_df2 Shape:" ,train_df2.shape)

# frame_combined = pd.concat([train_df1, train_df2], ignore_index=True)
# train_df=frame_combined.copy()
# print("New Train Shape:" ,train_df.shape)



# pd.set_option('expand_frame_repr', False)
# Split train and target
train = train_df1.to_numpy()
np.random.shuffle(train)
np.random.shuffle(train)
np.random.shuffle(train)
# print(pd.DataFrame(train).head(5))
y = train[:,0]
X = train[:,1:]

if(TestScript):
	y = train[1:100,0]
	X = train[1:100,1:]

if(validation):
	X_train, X_val, y_train, y_val = train_test_split(
	   X, y, test_size=0.10,stratify=y, random_state=42)
else:   
	X_train=X
	y_train=y


# print(y_train.dtype)
# print(X_train.dtype)
# print(pd.DataFrame(y_train).head(5))
# print(pd.DataFrame(X_train).head(5))

# Split train and target
test1 = test_df1.to_numpy()
# print(pd.DataFrame(test1).head(5))
y_test1 = test1[:,0]
X_test1 = test1[:,1:]
# print(pd.DataFrame(y_test1).head(5))
# print(pd.DataFrame(X_test1).head(5))


# Split train and target
# test2 = test_df2.to_numpy()
# # print(test2)

# # print(pd.DataFrame(test2).head(5))
# y_test2 = test2[:,0]
# X_test2 = test2[:,1:]
# print(pd.DataFrame(y_test2).head(5))
# print(pd.DataFrame(X_test2).head(5))

# Split train and target
# test3 = test_df3.to_numpy()
# # print(pd.DataFrame(test3).head(5))
# y_test3 = test3[:,0].astype(np.int64)
# X_test3 = test3[:,1:].astype(np.float)
# # print(pd.DataFrame(y_test3).head(5))
# # print(pd.DataFrame(X_test3).head(5))



# print(np.any(np.isnan(X_test1)))
# print(np.any(np.isnan(X_test2)))
# print(np.argwhere(np.isnan(X_test2)))
# print(np.any(np.isnan(X_test3)))

print("Train counts of label '1': {}".format(sum(y_train==1)),file=Output_file)
print("Train counts of label '0': {}".format(sum(y_train==0)),file=Output_file)

print("Test1 counts of label '1': {}".format(sum(y_test1==1)),file=Output_file)
print("Test1 counts of label '0': {}".format(sum(y_test1==0)),file=Output_file)


sc = StandardScaler()







###############################################################################
#                         Classifiers Or Regressors                           #
###############################################################################


BLGBM =LGBMClassifier(n_estimators =1000, n_jobs=N_jobs)
# NOLGBM =LGBMClassifier(n_estimators =1000,  is_unbalance = True, n_jobs=N_jobs)
# param = {'colsample_bytree': 0.4, 'gamma': 0.3,"n_jobs": 6, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1.5, 'n_estimators': 10000, 'subsample': 0.95}
# OXGB = XGBClassifier(**param)

XGB = XGBClassifier(n_estimators =1000, n_jobs=N_jobs)

KNN = KNeighborsClassifier(n_neighbors=9,n_jobs=N_jobs)

LG = LogisticRegression(max_iter =1000,n_jobs=N_jobs)

SVC =SVC(gamma='auto',probability=True)

RAND = RandomForestClassifier(n_estimators=1000,n_jobs=N_jobs)

ETC = ExtraTreesClassifier(n_estimators=1000,n_jobs=N_jobs)

CAT = CatBoostClassifier(n_estimators=1000,verbose=False,thread_count=N_jobs)




# # Initializing the StackingCV classifier
STACKSetup=[make_pipeline(sc, BLGBM), make_pipeline(sc, KNN), make_pipeline(sc,ETC)]
STACK = StackingCVClassifier(classifiers =STACKSetup ,
                            shuffle = False,
                            use_probas = True,
                            random_state =42,
                            stratify =True,
                            use_features_in_secondary=True,
                            cv = 10,
                            verbose=0,
                            n_jobs=-1,
                            store_train_meta_features = True,
                            meta_classifier = BLGBM
)


estimators = {
# 'SLGBM': SLGBM,
    
# 'NOLGBM': NOLGBM, 

'BLGBM': BLGBM, 
# 'XGB': XGB, 
# 'KNN': KNN, 
# 'LG': LG, 
'RAND': RAND, 
# 'ETC': ETC, 
# 'CAT': CAT, 
# 'SVC': SVC, 
'STACK': STACK, 
}




# ##############################################################################
#                              Training CV Result                             #
# ##############################################################################
if(CVTrainOrTest):
	for estimator_name, estimator_object in estimators.items():
		print("Cross Validation")
		print(",\n\n",estimator_name)

		clf = make_pipeline(sc,estimator_object)
		# XX=clf.train_meta_features_ 
		# XX=np.concatenate((X_train, XX), axis=1)
		proba_predict_train = cross_val_predict(clf,X_train, y_train, cv=10,verbose=0, n_jobs=-1, method='predict_proba')
		# pd.DataFrame(proba_predict_train).to_csv("./save_files/proba_predict_train.csv")
		y_pred = (proba_predict_train[:,1] >= threshold).astype(int)
		print("\n##### CV Train #####\n"+estimator_name,file=Output_file)
		Predict_Result(y_train.astype(float), np.array(y_pred),proba_predict_train[:,1])



# ##############################################################################
#                              Training For Testing                           #
# ##############################################################################
if not CVTrainOrTest:
	for estimator_name, estimator_object in estimators.items():
		print("Train for Test")
		clf=make_pipeline(sc,estimator_object)

		clf.fit(X_train, y_train)

		filename = './save_files/finalized_model.sav'
		joblib.dump(clf, filename)

		# clf=joblib.load(filename)
		# print(clf.train_meta_features_)
		# pd.DataFrame(clf.train_meta_features_).to_csv("./save_files/train_meta_features.csv")






###############################################################################
#                              Validation Result                              #
###############################################################################
	if(validation):
		prob_predict0=clf.predict_proba(X_val)  
		pd.DataFrame(prob_predict0).to_csv("./save_files/prob_predict_val_1.csv")
		y_pred = (prob_predict0[:,1] >= threshold).astype(int)
		print("\n#####Validation")
		print("\n#####Validation",file=Output_file)
		Predict_Result(y_val.astype(float), np.array(y_pred),prob_predict0[:,1])


###############################################################################
#                              Test1 Result                                   #
###############################################################################
	if not CVTrainOrTest:
		prob_predict1=clf.predict_proba(X_test1)      
		pd.DataFrame(prob_predict1).to_csv("./save_files/prob_predict_test_1.csv")
		y_pred = (prob_predict1[:,1] >= threshold).astype(int)
		print("\n\n#####Test1",Test1Dataset)
		print("\n\n#####Test1",Test1Dataset,file=Output_file)
		print("\n\n#####Test1",Test1Dataset,file=Compare_file)
		Predict_Result(y_test1.astype(float), np.array(y_pred),prob_predict1[:,1],Test1Dataset)




# ###############################################################################
# #                              Test2 Result                                   #
# ###############################################################################
# prob_predict2=clf.predict_proba(X_test2)       
# predict=clf.predict(X_test2)         
# pd.DataFrame(prob_predict2).to_csv("./save_files/prob_predict_test_2.csv")
# y_pred = (prob_predict2[:,1] >= threshold).astype(int)
# print("\n#####Human",file=Output_file)
# ACC_Bal,REC_cla,SPE_cla,auc_score,MCC_cla =Predict_Result(y_test2.astype(float), np.array(y_pred),prob_predict2[:,1])
# ABACC=ABACC+ACC_Bal
# AMCC=AMCC+MCC_cla

###############################################################################
#                              Test3 Result                                   #
###############################################################################
# prob_predict3=clf.predict_proba(X_test3)       
# predict=clf.predict(X_test3)         
# # print(prob_predict)
# pd.DataFrame(prob_predict3).to_csv("./save_files/prob_predict_test_3.csv")
# y_pred = (prob_predict3[:,1] >= threshold).astype(int)
# print("\n#####SC",file=Output_file)
# ACC_Bal,REC_cla,SPE_cla,auc_score,MCC_cla =Predict_Result(y_test3.astype(float), np.array(y_pred),prob_predict3[:,1])
# ABACC=ABACC+ACC_Bal
# AMCC=AMCC+MCC_cla



# print("Average Balanced Accuracy:", ABACC/3)
# print("Average MCC:", AMCC/3)
# print("\n#####Average Balanced Accuracy:", ABACC/3,file=Output_file)
# print("Average MCC:", AMCC/3,file=Output_file)




# end = time.time()
# hours, rem = divmod(end-start, 3600)
# minutes, seconds = divmod(rem, 60)
# print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
# print("\n{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds),file=Output_file)
Output_file.close()
Compare_file.close()




# ###############################################################################
# #                             Threshold Optimization                          #
# ###############################################################################
if(validation):
	df=pd.DataFrame(columns=["threshold","mccval","swal","mcc1","sw1"])
	for i in np.arange(0.0, 1.0, 0.01):

		# trainmcc=thr_to_mcc(round(i,2), y_train, proba_predict_train)
		# trainbacc=thr_to_accuracy(round(i,2), y_train, proba_predict_train)
		# "trainmcc","trainbacc",

		# 'trainmcc':trainmcc, 'trainbacc':trainbacc,
		mcc0=thr_to_mcc(round(i,2), y_val, prob_predict0)
		bacc0=thr_to_sw(round(i,2), y_val, prob_predict0) 
		# "mccval","baccval",
		# 'mccval':mcc0, 'baccval':bacc0,
		mcc1=thr_to_mcc(round(i,2), y_test1, prob_predict1)
		bacc1=thr_to_sw(round(i,2), y_test1, prob_predict1)
		# mcc2=thr_to_mcc(round(i,2), y_test2, prob_predict2)
		# bacc2=thr_to_accuracy(round(i,2), y_test2, prob_predict2)    
		# mcc3=thr_to_mcc(round(i,2), y_test3, prob_predict3)
		# bacc3=thr_to_accuracy(round(i,2), y_test3, prob_predict3)
					  
		
		new_row = {'threshold':i, 'mccval':mcc0, 'swval':bacc0, 'mcc1':mcc1, 'sw1':bacc1  }
		#append row to the dataframe
		df = df.append(new_row, ignore_index=True)


	# df.sort_values('all_test_average',ascending=False)
	df.to_csv("./save_files/Threshold.csv")









'''
------------------------------------
------------------------------------
	pt_main_willow.py
------------------------------------
------------------------------------
'''

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

import scipy.io
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pdb

#User defined functions
from pt_get_data_willow import function_get_training_data_cc, input_cc, output_cc, function_normalise_data
from pt_get_data_willow import input_data, function_get_input_data, classifier_data
from pt_get_data_willow import function_reduce_dimension_of_data
from pt_train_cc_willow import train_pt_cc_input, train_pt_cc_output, function_train_pytorch_cc
print "*****************************************************************************************************************************************"

'''
------------------------------------
------------------------------------
Constants/Model Parameters
------------------------------------
------------------------------------
'''
DATASET_NAME = 'WILLOW'
EPOCHS = 50
EPOCHS_CC = 50
BATCH_SIZE = 128
BATCH_SIZE_CC = 128
TR_VA_SPLIT = np.array([0.7, 0.3])
MIN_NUMBER_OF_SAMPLES_ACROSS_CLASSES = 50
NOISE_FACTOR = 0.05
INCREASE_FACTOR_CAE = 100
DATA_SAVE_PATH = '/home/SharedData/omkar/study/phd-research/codes/tf-codes/data-willow/data_conf218_1/'
#DATA_SAVE_PATH = '../data-willow/data_var_compare/'
USE_ENCODER_FEATURES = 1
DO_PCA = 0
NUMBER_OF_SAMPLES_FOR_TRAINING_CODER = 10000
REDUCED_DIMENSION_VISUAL_FEATURE = 500 
dimension_hidden_layer1_coder = 50
min_num_samples_per_class = 70
	
'''
------------------------------------
------------------------------------
load input data
------------------------------------
------------------------------------
'''
obj_input_data = input_data()
obj_input_data.dataset_name = DATASET_NAME

obj_input_data = function_get_input_data(obj_input_data)
visual_features_dataset = obj_input_data.visual_features_dataset
train_class_labels = obj_input_data.train_class_labels
test_class_labels = obj_input_data.test_class_labels
dataset_labels = obj_input_data.dataset_labels
dataset_train_labels = obj_input_data.dataset_train_labels
dataset_test_labels = obj_input_data.dataset_test_labels
number_of_train_classes = np.size(train_class_labels)
number_of_test_classes = np.size(test_class_labels)
dimension_visual_data = visual_features_dataset.shape[1]
number_of_samples_dataset = visual_features_dataset.shape[0]
number_of_classes = train_class_labels.shape[0]
print "Dataset visual features shape is: %d X %d" % visual_features_dataset.shape
print "Dimension of visual data: %d" %dimension_visual_data
print "Number of dataset samples: %d" %number_of_samples_dataset
print "Number of classes: %d" %number_of_classes
print "Train classes are"
print(train_class_labels)
print "Test classes are"
print(test_class_labels)
print "Noise factor %f"%NOISE_FACTOR 
print "Data augmentation factor %d"%INCREASE_FACTOR_CAE 
print "Dimension Coder Hidden1 %d" %(dimension_hidden_layer1_coder)


'''
------------------------------------
------------------------------------
Prepare data split for classifier
------------------------------------
------------------------------------
'''
#Get mean feature vector for each class
mean_feature_mat = np.empty((0, dimension_visual_data), float)
number_of_samples_per_class_train = []
number_of_samples_per_class_test = []
number_of_samples_per_class_valid = []

obj_classifier = classifier_data()
cnt = 0
#NOTE: TO BE REMOVED        
visual_features_dataset_PCAed_unnorm = function_reduce_dimension_of_data(visual_features_dataset, visual_features_dataset, REDUCED_DIMENSION_VISUAL_FEATURE)
visual_features_dataset_PCAed = function_normalise_data(visual_features_dataset)
visual_features_dataset = visual_features_dataset_PCAed

cnt = 0
for classI in train_class_labels:
	print "Stacking tr/val/ts data for class %d"%classI 
	train_val_indices = np.flatnonzero(dataset_train_labels == classI)
	test_indices = np.flatnonzero(dataset_test_labels == classI)
	classI_train_val_features = visual_features_dataset[train_val_indices.astype(int), :]
	classI_test_features = visual_features_dataset[test_indices.astype(int), :]
	#mean_feature = classI_features.mean(0)
	#mean_feature_mat = np.append(mean_feature_mat, mean_feature.reshape(1, dimension_visual_data), axis = 0)	
	number_of_samples_per_class_train.append(int(TR_VA_SPLIT[0] * np.size(train_val_indices))) 
	number_of_samples_per_class_valid.append(int(TR_VA_SPLIT[1] * np.size(train_val_indices)))
	number_of_samples_per_class_test.append(np.size(test_indices)) 
	start_vl = number_of_samples_per_class_train[-1]
	end_vl = start_vl + number_of_samples_per_class_valid[-1]
	start_ts = 0
	end_ts = start_ts + number_of_samples_per_class_test[-1]
	if cnt == 0:
		cnt = 1	
		obj_classifier.train_data = classI_train_val_features[:number_of_samples_per_class_train[-1], :] 
		obj_classifier.valid_data = classI_train_val_features[start_vl:end_vl, :] 
		obj_classifier.test_data = classI_test_features[start_ts:end_ts, :] 
		obj_classifier.train_labels = np.full((1, number_of_samples_per_class_train[-1]), classI, dtype=int) 
		obj_classifier.valid_labels = np.full((1, number_of_samples_per_class_valid[-1]), classI, dtype=int) 
		obj_classifier.test_labels = np.full((1, number_of_samples_per_class_test[-1]), classI, dtype=int) 
		train_valid_indices_all_classes = train_val_indices
		test_indices_all_classes = test_indices
	else:	
		obj_classifier.train_data = np.vstack((obj_classifier.train_data, classI_train_val_features[:number_of_samples_per_class_train[-1], :])) 
		obj_classifier.valid_data = np.vstack((obj_classifier.valid_data, classI_train_val_features[start_vl:end_vl, :])) 
		obj_classifier.test_data = np.vstack((obj_classifier.test_data, classI_test_features[start_ts:end_ts, :])) 
		obj_classifier.train_labels = np.hstack((obj_classifier.train_labels, np.full((1, number_of_samples_per_class_train[-1]), classI, dtype=int))) 
		obj_classifier.valid_labels = np.hstack((obj_classifier.valid_labels, np.full((1, number_of_samples_per_class_valid[-1]), classI, dtype=int))) 
		obj_classifier.test_labels = np.hstack((obj_classifier.test_labels, np.full((1, number_of_samples_per_class_test[-1]), classI, dtype=int))) 

		train_valid_indices_all_classes = np.hstack((train_valid_indices_all_classes, train_val_indices))
		test_indices_all_classes = np.hstack((test_indices_all_classes, test_indices))
#PCA
if DO_PCA:
	print "Doing PCA on training-validation set and applying on test set."
	train_valid_data = np.vstack((obj_classifier.train_data, obj_classifier.valid_data))
	n_samples_before_pca = train_valid_data.shape[0]
        train_valid_test_data_PCAed = function_reduce_dimension_of_data(train_valid_data, obj_classifier.test_data, REDUCED_DIMENSION_VISUAL_FEATURE)
	start = 0
	end = obj_classifier.train_labels.shape[1] + obj_classifier.valid_labels.shape[1]
	train_valid_data = train_valid_test_data_PCAed[start:end,:]
	print "Doing normalisation for train-valid data"
	train_valid_data = function_normalise_data(train_valid_data)
	start = 0
	end = obj_classifier.train_labels.shape[1]
	obj_classifier.train_data = train_valid_data[start:end, :]
	start = end
	end = end + obj_classifier.valid_labels.shape[1]
	obj_classifier.valid_data = train_valid_data[start:end, :]

	strt = end
	end = start + obj_classifier.test_labels.shape[1]
	obj_classifier.test_data = train_valid_test_data_PCAed[start:end,:]
	print "Doing normalisation for test data"
	obj_classifier.test_data = function_normalise_data(obj_classifier.test_data)
else:
	print "*NOT* doing PCA...."
#visual_features_dataset_PCAed_shuff = np.vstack((obj_classifier.train_data, obj_classifier.valid_data))
#n_samples_after_pca = visual_features_dataset_PCAed_shuff.shape[0]
#visual_features_dataset_PCAed_shuff = np.vstack((visual_features_dataset_PCAed_shuff, obj_classifier.test_data))

'''
------------------------------------
------------------------------------
Save classifier data
------------------------------------
------------------------------------
'''
base_filename = DATA_SAVE_PATH + DATASET_NAME + '_' + str(dimension_hidden_layer1_coder) + '_' + str(REDUCED_DIMENSION_VISUAL_FEATURE) + '_E2E_'
exp_name = 'clfr_'
filename = base_filename + exp_name + 'train_labels'
scipy.io.savemat(filename, dict(train_labels = obj_classifier.train_labels))
print filename

filename = base_filename + exp_name + 'test_labels'
scipy.io.savemat(filename, dict(test_labels = obj_classifier.test_labels))
print filename

filename = base_filename + exp_name + 'valid_labels'
scipy.io.savemat(filename, dict(valid_labels = obj_classifier.valid_labels))
print filename

#CNN features for SVM experiment
exp_name = 'cnn_svm_' + str(obj_classifier.train_data.shape[1]) + '_dim_'
filename = base_filename + exp_name + 'train_data'
scipy.io.savemat(filename, dict(train_data = obj_classifier.train_data))
print filename

exp_name = 'cnn_svm_' + str(obj_classifier.test_data.shape[1]) + '_dim_'
filename = base_filename + exp_name + 'test_data'
scipy.io.savemat(filename, dict(test_data = obj_classifier.test_data))
print filename

exp_name = 'cnn_svm_' + str(obj_classifier.valid_data.shape[1]) + '_dim_'
filename = base_filename + exp_name + 'valid_data'
scipy.io.savemat(filename, dict(valid_data = obj_classifier.valid_data))
print filename
cnt = 0

'''
------------------------------------
------------------------------------
Training class-specific AE
------------------------------------
------------------------------------
'''

exp_name = 'aec_features_all_classes_'
filename = base_filename + exp_name + 'tr_' + str(number_of_classes) + '.mat'
if not os.path.isfile(filename):
	for classI in train_class_labels:
		print "**************************************"
		classJ = classI	
		#Get data for training AEC
		cc1_start = time.time()
		obj_input_cc = input_cc()
		obj_input_cc.classI = classI
		obj_input_cc.classJ = classJ
                obj_input_cc.number_of_classes = number_of_classes
		obj_input_cc.visual_features = visual_features_dataset_PCAed
		obj_input_cc.train_valid_split = TR_VA_SPLIT
		obj_input_cc.dataset_labels = dataset_labels
		obj_input_cc.dataset_train_labels = dataset_train_labels
		obj_input_cc.dataset_test_labels = dataset_test_labels
		obj_input_cc.min_num_samples_per_class = min_num_samples_per_class

		obj_cc1_train_valid_data = function_get_training_data_cc(obj_input_cc)
		cc1_input_train_perm = obj_cc1_train_valid_data.input_train_perm
		INCREASE_FACTOR_AEC = int(NUMBER_OF_SAMPLES_FOR_TRAINING_CODER / cc1_input_train_perm.shape[0])
		print "Increase factor for AEC is %d"%(INCREASE_FACTOR_AEC)	
		cc1_input_train_perm = np.tile(cc1_input_train_perm, (INCREASE_FACTOR_AEC, 1))
		cc1_input_train_perm = cc1_input_train_perm + NOISE_FACTOR * np.random.normal(0, 1, cc1_input_train_perm.shape)
		cc1_input_train_perm = function_normalise_data(cc1_input_train_perm)

		if classI == classJ:
			cc1_output_train_perm = cc1_input_train_perm
		else:	
			cc1_output_train_perm  = obj_cc1_train_valid_data.output_train_perm
			cc1_output_train_perm = np.tile(cc1_output_train_perm, (INCREASE_FACTOR_AEC, 1))
			cc1_output_train_perm = cc1_output_train_perm + NOISE_FACTOR * np.random.normal(0, 1, cc1_output_train_perm.shape)
			cc1_output_train_perm = function_normalise_data(cc1_output_train_perm)

		#Train Pytorch cc
		print "Training coder over %d samples"%(cc1_input_train_perm.shape[0])
		obj_train_tf_cc_input = train_pt_cc_input()
		obj_train_tf_cc_input.classI = classI
		obj_train_tf_cc_input.classJ = classJ
		obj_train_tf_cc_input.dataset_name = DATASET_NAME 
		obj_train_tf_cc_input.data_save_path = DATA_SAVE_PATH 
		obj_train_tf_cc_input.dim_feature = obj_input_cc.visual_features.shape[1]
		obj_train_tf_cc_input.cc1_input_train_perm = cc1_input_train_perm
		obj_train_tf_cc_input.cc1_output_train_perm = cc1_output_train_perm
		obj_train_tf_cc_input.cc1_input_valid_perm = function_normalise_data(obj_cc1_train_valid_data.input_valid_perm)
		obj_train_tf_cc_input.cc1_output_valid_perm = function_normalise_data(obj_cc1_train_valid_data.output_valid_perm)
		obj_train_tf_cc_input.obj_classifier = obj_classifier
		obj_train_tf_cc_input.dimension_hidden_layer1 = dimension_hidden_layer1_coder
		obj_train_tf_cc_input.EPOCHS_CC = EPOCHS
		obj_train_tf_cc_input.neg_samples_pool = obj_cc1_train_valid_data.neg_samples_pool
                obj_train_tf_cc_input.neg_samples_pool_labels = obj_cc1_train_valid_data.neg_samples_pool_labels
		
		cc_start = time.time()	
		obj_train_tf_cc_output = function_train_pytorch_cc(obj_train_tf_cc_input)
		cc_end = time.time() 
		print "Processing time for Aec %f"%((cc_end - cc_start)/60)
	  	
		#Concatenate features
		if cnt == 0:
			cnt  = 1
			if USE_ENCODER_FEATURES:
				print "Using encoded features"
				cross_features_train = function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)
				cross_features_valid = function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)
				cross_features_test = function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)
			else:
				print "Caution : Using decoded features....................."
				raise ValueError('Check for normalisation if needed')
				cross_features_train = obj_train_tf_cc_output.decoded_data_train_cc1
				cross_features_valid = obj_train_tf_cc_output.decoded_data_valid_cc1
				cross_features_test = obj_train_tf_cc_output.decoded_data_test_cc1
		else:
			if USE_ENCODER_FEATURES:
				print "Using encoded features"
				cross_features_train = np.hstack((cross_features_train, function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)))
				cross_features_valid = np.hstack((cross_features_valid, function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)))
				cross_features_test = np.hstack((cross_features_test, function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)))
			else:
				print "Using decoded features"
				raise ValueError('Check for normalisation if needed')
				cross_features_train = np.hstack((cross_features_train, obj_train_tf_cc_output.decoded_data_train_cc1))
				cross_features_valid = np.hstack((cross_features_valid, obj_train_tf_cc_output.decoded_data_valid_cc1))
				cross_features_test = np.hstack((cross_features_test, obj_train_tf_cc_output.decoded_data_test_cc1))

	#Saving aec features
	exp_name = 'aec_features_all_classes_'
	filename = base_filename + exp_name + 'tr_' + str(classI)		
	print"Saving aec features *train data* for classes 1 to %d ...%s"%(classI, filename)
	scipy.io.savemat(filename, dict(cross_feautures_tr = cross_features_train))
		
	filename = base_filename + exp_name + 'vl_' + str(classI)		
	print"Saving aec features *valid data* for classes 1 to %d ...%s"%(classI, filename)
	scipy.io.savemat(filename, dict(cross_feautures_val = cross_features_valid))

	filename = base_filename + exp_name + 'ts_' + str(classI)		
	print"Saving aec features *test data* for classes 1 to %d ...%s"%(classI, filename)
	scipy.io.savemat(filename, dict(cross_feautures_ts = cross_features_test))

else:
	print "AEC features already calculated. Skipping..."

'''
------------------------------------
------------------------------------
Train CE
------------------------------------
------------------------------------
'''
for classI in train_class_labels:
	cnt = 0 #NOTE: cnt is made zero in order to save cross features for each class in different file
	#check if cross-features already calculated
	exp_name = 'cec_features_class_' 
	filename = base_filename + exp_name + 'tr_' + str(classI) + '.mat'		
	if not os.path.isfile(filename):
		print "%s does not exist. Finding cross features for class %d"%(filename, classI)
		for classJ in train_class_labels:
			if (classI != classJ):
				cc1_start = time.time()
				obj_input_cc = input_cc()
				obj_input_cc.classI = classI
				obj_input_cc.classJ = classJ
                		obj_input_cc.number_of_classes = number_of_classes
				obj_input_cc.visual_features = visual_features_dataset_PCAed
				obj_input_cc.train_valid_split = TR_VA_SPLIT
				obj_input_cc.dataset_labels = dataset_labels
				obj_input_cc.dataset_train_labels = dataset_train_labels
				obj_input_cc.dataset_test_labels = dataset_test_labels
				obj_input_cc.min_num_samples_per_class = min_num_samples_per_class
			  
				obj_cc1_train_valid_data = function_get_training_data_cc(obj_input_cc)
				cc1_input_train_perm = obj_cc1_train_valid_data.input_train_perm
				INCREASE_FACTOR_CAE = int(NUMBER_OF_SAMPLES_FOR_TRAINING_CODER / cc1_input_train_perm.shape[0])
				print "Increase factor for CEC is %d"%(INCREASE_FACTOR_CAE)
				cc1_input_train_perm = np.tile(cc1_input_train_perm, (INCREASE_FACTOR_CAE, 1))
				cc1_input_train_perm = cc1_input_train_perm + NOISE_FACTOR * np.random.normal(0, 1, cc1_input_train_perm.shape)
				cc1_input_train_perm = function_normalise_data(cc1_input_train_perm)

				if classI == classJ:
					cc1_output_train_perm = cc1_input_train_perm
				else:	
					cc1_output_train_perm  = obj_cc1_train_valid_data.output_train_perm
					cc1_output_train_perm = np.tile(cc1_output_train_perm, (INCREASE_FACTOR_CAE, 1))
					cc1_output_train_perm = cc1_output_train_perm + NOISE_FACTOR * np.random.normal(0, 1, cc1_output_train_perm.shape)
					cc1_output_train_perm = function_normalise_data(cc1_output_train_perm)

				#Train pytorch cc
				print "Training cc over %d samples"%(cc1_input_train_perm.shape[0])
				obj_train_tf_cc_input = train_pt_cc_input()
				obj_train_tf_cc_input.classI = classI
				obj_train_tf_cc_input.classJ = classJ
				obj_train_tf_cc_input.dataset_name = DATASET_NAME 
				obj_train_tf_cc_input.data_save_path = DATA_SAVE_PATH 
				obj_train_tf_cc_input.dim_feature = visual_features_dataset.shape[1]
				obj_train_tf_cc_input.cc1_input_train_perm = cc1_input_train_perm
				obj_train_tf_cc_input.cc1_output_train_perm = cc1_output_train_perm
				obj_train_tf_cc_input.cc1_input_valid_perm = function_normalise_data(obj_cc1_train_valid_data.input_valid_perm)
				obj_train_tf_cc_input.cc1_output_valid_perm = function_normalise_data(obj_cc1_train_valid_data.output_valid_perm)
				obj_train_tf_cc_input.obj_classifier = obj_classifier
				obj_train_tf_cc_input.dimension_hidden_layer1 = dimension_hidden_layer1_coder
				obj_train_tf_cc_input.EPOCHS_CC = EPOCHS_CC
				cc_start = time.time()
				obj_train_tf_cc_output = function_train_pytorch_cc(obj_train_tf_cc_input)
				cc_end = time.time() 
				print "Processing time for Aec %f"%((cc_end - cc_start)/60)
				
				#Concatenate features
				if cnt == 0:
					cnt = 1
					if USE_ENCODER_FEATURES:
						print "Using encoded features"
						cross_features_train = function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)
						cross_features_valid = function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)
						cross_features_test = function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)
					else:
						print "Caution : Using decoded features....................."
						raise ValueError('Check for normalisation if needed')
						cross_features_train = obj_train_tf_cc_output.decoded_data_train_cc1
						cross_features_valid = obj_train_tf_cc_output.decoded_data_valid_cc1
						cross_features_test = obj_train_tf_cc_output.decoded_data_test_cc1
				else:
					if USE_ENCODER_FEATURES:
						print "Using encoded features"
						cross_features_train = np.hstack((cross_features_train, function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)))
						cross_features_valid = np.hstack((cross_features_valid, function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)))
						cross_features_test = np.hstack((cross_features_test, function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)))
					else:
						print "Caution : Using decoded features....................."
						raise ValueError('Check for normalisation if needed')
						cross_features_train = np.hstack((cross_features_train, obj_train_tf_cc_output.decoded_data_train_cc1))
						cross_features_valid = np.hstack((cross_features_valid, obj_train_tf_cc_output.decoded_data_valid_cc1))
						cross_features_test = np.hstack((cross_features_test, obj_train_tf_cc_output.decoded_data_test_cc1))
		if 1:
			#Saving cross features
			exp_name = 'cec_features_class_' 
			filename = base_filename + exp_name + 'tr_' + str(classI)		
			print"%s"%(filename)
			scipy.io.savemat(filename, dict(cross_feautures_tr = cross_features_train))
				
			filename = base_filename + exp_name + 'vl_' + str(classI)		
			print"%s"%(filename)
			scipy.io.savemat(filename, dict(cross_feautures_val = cross_features_valid))
		
			filename = base_filename + exp_name + 'ts_' + str(classI)		
			print"%s"%(filename)
			scipy.io.savemat(filename, dict(cross_feautures_ts = cross_features_test))
	else:
		print "Cross features %s exists. Skipping..."%(filename)			


'''
------------------------------------------------------------
			main_awa.py
------------------------------------------------------------

'''

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras.callbacks import TensorBoard

import pickle
import scipy.io
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pdb

import tensorflow as tf

#User defined functions
from get_data_awa import function_get_training_data_cc, input_cc, output_cc, function_normalise_data
from get_data_awa import input_data, function_get_input_data, classifier_data
from get_data_awa import function_reduce_dimension_of_data, get_cluster_centers
from get_data_awa import train_seen_to_unseen_regressor, combine_specific_nonspecific_attributes
from get_data_awa import train_visual_to_semantic_specific_regressor, visualize_data
from get_data_awa import train_visual_to_semantic_nonspecific_regressor
from train_cc_awa import train_tf_cc_input, train_tf_cc_output, function_train_tensorflow_cc, function_train_keras_cc
from train_cc_awa import classifier_output, function_train_classifier_for_cc

EPOCHS = 50
EPOCHS_CC = 50
BATCH_SIZE = 128
BATCH_SIZE_CC = 128
TR_VA_SPLIT = np.array([0.7, 0.3])
NOISE_FACTOR = 0.05
INCREASE_FACTOR_CAE = 100
dataset_list = ['sample_wt', 'AwA']
DATASET_INDEX = 1
#DATA_SAVE_PATH = '../data-awa/data1/'
DATA_SAVE_PATH = '/home/SharedData/omkar/study/phd-research/codes/tf-codes/data-zsl/data-awa/data4/'
IMAGE_DATA_SAVE_PATH = DATA_SAVE_PATH + 'plots/'
print DATA_SAVE_PATH
USE_ENCODER_FEATURES = 1
DO_PCA = 1
NUMBER_OF_SAMPLES_FOR_TRAINING_CODER = 20000
REDUCED_DIMENSION_VISUAL_FEATURE = 1000 
dimension_hidden_layer1_coder = 100
min_num_samples_per_class = 92#NOTE: for AwA class 12 (training class) has samples 92
	
#Load input data
obj_input_data = input_data()
obj_input_data.dataset_name = dataset_list[DATASET_INDEX]
obj_input_data = function_get_input_data(obj_input_data)
visual_features_dataset = obj_input_data.visual_features_dataset
class_names = obj_input_data.class_names
train_class_labels = obj_input_data.train_class_labels
test_class_labels = obj_input_data.test_class_labels
attributes = obj_input_data.attributes
dataset_train_labels = obj_input_data.dataset_train_labels
dataset_test_labels = obj_input_data.dataset_test_labels
visual_features_dataset = function_normalise_data(visual_features_dataset)
number_of_train_classes = np.size(train_class_labels)
number_of_test_classes = np.size(test_class_labels)
dimension_visual_data = visual_features_dataset.shape[1]
number_of_samples_dataset = visual_features_dataset.shape[0]
print "Dataset visual features shape is: %d X %d" % visual_features_dataset.shape
print "Dimension of visual data: %d" %dimension_visual_data
print "Number of dataset samples: %d" %number_of_samples_dataset
print "Train classes are"
print train_class_labels
print "Test classes are"
print test_class_labels
print "Noise factor %f"%NOISE_FACTOR 
print "Data augmentation factor %d"%INCREASE_FACTOR_CAE 
print "Dimension Coder Hidden1 %d"%dimension_hidden_layer1_coder


#Get mean feature vector for each class
mean_feature_train_mat = np.empty((0, dimension_visual_data), float)
mean_feature_test_mat = np.empty((0, dimension_visual_data), float)
number_of_samples_per_class_train = []
number_of_samples_per_class_test = []
number_of_samples_per_class_valid = []

attributes = function_normalise_data(attributes)
obj_classifier = classifier_data()
obj_classifier.prototypes = attributes
cnt = 0
if 0:
	#NOTE: TO BE REMOVED        
	visual_features_dataset_PCAed_unnorm = function_reduce_dimension_of_data(visual_features_dataset, visual_features_dataset, REDUCED_DIMENSION_VISUAL_FEATURE)
	visual_features_dataset_PCAed = function_normalise_data(visual_features_dataset_PCAed_unnorm)
	visual_features_dataset = visual_features_dataset_PCAed

cnt = 0
for classI in train_class_labels:
	print "Stacking train data for class %d"%classI 
	train_val_indices = np.flatnonzero(dataset_train_labels == classI)
	classI_train_val_features = visual_features_dataset[train_val_indices.astype(int), :]
	mean_feature_train = classI_train_val_features.mean(0)
	mean_feature_train_mat = np.append(mean_feature_train_mat, mean_feature_train.reshape(1, dimension_visual_data), axis = 0)	
	number_of_samples_per_class_train.append(int(TR_VA_SPLIT[0] * np.size(train_val_indices))) 
	number_of_samples_per_class_valid.append(int(TR_VA_SPLIT[1] * np.size(train_val_indices)))
	start_vl = number_of_samples_per_class_train[-1]
	end_vl = start_vl + number_of_samples_per_class_valid[-1]
	train_classI_samples_attributes = np.tile(attributes[classI - 1, :], (number_of_samples_per_class_train[-1], 1))
	valid_classI_samples_attributes = np.tile(attributes[classI - 1, :], (number_of_samples_per_class_valid[-1], 1))
	if cnt == 0:
		cnt = 1	
		obj_classifier.train_data = classI_train_val_features[:number_of_samples_per_class_train[-1], :] 
		obj_classifier.valid_data = classI_train_val_features[start_vl:end_vl, :] 
		obj_classifier.train_labels = np.full((1, number_of_samples_per_class_train[-1]), classI, dtype=int) 
		obj_classifier.valid_labels = np.full((1, number_of_samples_per_class_valid[-1]), classI, dtype=int) 
		obj_classifier.train_attributes = train_classI_samples_attributes
		obj_classifier.valid_attributes = valid_classI_samples_attributes
		train_valid_indices_all_classes = train_val_indices
	else:	
		obj_classifier.train_data = np.vstack((obj_classifier.train_data, classI_train_val_features[:number_of_samples_per_class_train[-1], :])) 
		obj_classifier.valid_data = np.vstack((obj_classifier.valid_data, classI_train_val_features[start_vl:end_vl, :])) 
		obj_classifier.train_labels = np.hstack((obj_classifier.train_labels, np.full((1, number_of_samples_per_class_train[-1]), classI, dtype=int))) 
		obj_classifier.valid_labels = np.hstack((obj_classifier.valid_labels, np.full((1, number_of_samples_per_class_valid[-1]), classI, dtype=int))) 
		train_valid_indices_all_classes = np.hstack((train_valid_indices_all_classes, train_val_indices))
		obj_classifier.train_attributes = np.vstack((obj_classifier.train_attributes, train_classI_samples_attributes))
		obj_classifier.valid_attributes = np.vstack((obj_classifier.valid_attributes, valid_classI_samples_attributes))
cnt = 0
for classI in test_class_labels:
	print "Stacking test data for class %d"%classI 
	test_indices = np.flatnonzero(dataset_test_labels == classI)
	classI_test_features = visual_features_dataset[test_indices.astype(int), :]
	mean_feature_test = classI_test_features.mean(0)
	mean_feature_test_mat = np.append(mean_feature_test_mat, mean_feature_test.reshape(1, dimension_visual_data), axis = 0)	
	number_of_samples_per_class_test.append(np.size(test_indices)) 
	start_ts = 0
	end_ts = start_ts + number_of_samples_per_class_test[-1]
	test_classI_samples_attributes = np.tile(attributes[classI - 1, :], (number_of_samples_per_class_test[-1], 1))
	if cnt == 0:
		cnt = 1	
		obj_classifier.test_data = classI_test_features[start_ts:end_ts, :] 
		obj_classifier.test_labels = np.full((1, number_of_samples_per_class_test[-1]), classI, dtype=int) 
		obj_classifier.test_attributes = test_classI_samples_attributes
		test_indices_all_classes = test_indices
	else:	
		obj_classifier.test_data = np.vstack((obj_classifier.test_data, classI_test_features[start_ts:end_ts, :])) 
		obj_classifier.test_labels = np.hstack((obj_classifier.test_labels, np.full((1, number_of_samples_per_class_test[-1]), classI, dtype=int))) 
		obj_classifier.test_attributes = np.vstack((obj_classifier.test_attributes, test_classI_samples_attributes))
		test_indices_all_classes = np.hstack((test_indices_all_classes, test_indices))

obj_classifier.train_labels = (obj_classifier.train_labels).flatten()
obj_classifier.valid_labels = (obj_classifier.valid_labels).flatten()
obj_classifier.test_labels = (obj_classifier.test_labels).flatten()

#PCA
if DO_PCA:
	print "Doing PCA on training-validation set and applying on test set."
	train_valid_data = np.vstack((obj_classifier.train_data, obj_classifier.valid_data))
	n_samples_before_pca = train_valid_data.shape[0]
        train_valid_test_data_PCAed = function_reduce_dimension_of_data(train_valid_data, obj_classifier.test_data, REDUCED_DIMENSION_VISUAL_FEATURE)
	start = 0
	end = obj_classifier.train_labels.shape[0] + obj_classifier.valid_labels.shape[0]
	train_valid_data = train_valid_test_data_PCAed[start:end,:]
	print "Doing normalisation for train-valid data"
	train_valid_data = function_normalise_data(train_valid_data)
	start = 0
	end = obj_classifier.train_labels.shape[0]
	obj_classifier.train_data = train_valid_data[start:end, :]
	start = end
	end = end + obj_classifier.valid_labels.shape[0]
	obj_classifier.valid_data = train_valid_data[start:end, :]

	strt = end
	end = start + obj_classifier.test_labels.shape[0]
	obj_classifier.test_data = train_valid_test_data_PCAed[start:end,:]
	print "Doing normalisation for test data"
	obj_classifier.test_data = function_normalise_data(obj_classifier.test_data)
	visual_features_dataset_PCAed = np.vstack((obj_classifier.train_data, obj_classifier.valid_data))
else:
	print "*NOT* doing PCA...."

'''
-------------------------------------------------------
			Data save
-------------------------------------------------------
'''

base_filename = DATA_SAVE_PATH + dataset_list[DATASET_INDEX] + '_' + str(dimension_hidden_layer1_coder) + '_' + str(REDUCED_DIMENSION_VISUAL_FEATURE) + '_'
filename = base_filename + 'train_labels'
scipy.io.savemat(filename, dict(train_labels = obj_classifier.train_labels))
print filename

filename = base_filename + 'test_labels'
scipy.io.savemat(filename, dict(test_labels = obj_classifier.test_labels))
print filename

filename = base_filename + 'valid_labels'
scipy.io.savemat(filename, dict(valid_labels = obj_classifier.valid_labels))
print filename

#CNN features for SVM experiment
exp_name = 'cnn_svm_' + str(obj_classifier.train_data.shape[1]) + '_dim_'
filename = base_filename + 'train_data'
scipy.io.savemat(filename, dict(train_data = obj_classifier.train_data))
print filename

exp_name = 'cnn_svm_' + str(obj_classifier.test_data.shape[1]) + '_dim_'
filename = base_filename + 'test_data'
scipy.io.savemat(filename, dict(test_data = obj_classifier.test_data))
print filename

exp_name = 'cnn_svm_' + str(obj_classifier.valid_data.shape[1]) + '_dim_'
filename = base_filename + 'valid_data'
scipy.io.savemat(filename, dict(valid_data = obj_classifier.valid_data))
print filename

exp_name = 'cnn_svm_' + str(attributes.shape[1]) + '_dim_'
filename = base_filename + exp_name + 'attributes'
scipy.io.savemat(filename, dict(attributes = attributes))
print filename

string_title = 'Visual Train Data PCAed'
file_title = 'vis_PCAed_data_train'
visualize_data(obj_classifier.train_data, obj_classifier.train_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
string_title = 'Visual Valid Data PCAed'
file_title = 'vis_PCAed_data_valid'
visualize_data(obj_classifier.valid_data, obj_classifier.valid_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
string_title = 'Visual Test Data PCAed'
file_title = 'vis_PCAed_data_test'
visualize_data(obj_classifier.test_data, obj_classifier.test_labels, test_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)

cc_start = time.time() 
cnt = 0

'''
-----------------------------------------------------------------
		Regression f: viusal --> semantic (non-specific)
-----------------------------------------------------------------
'''

obj_classifier.train_attributes_nsp, obj_classifier.valid_attributes_nsp, obj_classifier.test_attributes_nsp = train_visual_to_semantic_nonspecific_regressor(obj_classifier)

#Save semantic features
filename = base_filename + '_semantic_features_train'
scipy.io.savemat(filename, dict(train_attributes_nsp = obj_classifier.train_attributes_nsp))
print filename

filename = base_filename + '_semantic_features_valid'
scipy.io.savemat(filename, dict(valid_attributes_nsp = obj_classifier.valid_attributes_nsp))
print filename

filename = base_filename + '_semantic_features_test'
scipy.io.savemat(filename, dict(test_attributes_nsp = obj_classifier.test_attributes_nsp))
print filename

string_title = 'Non-specific mapping: Visual to semantic (Seen classes-train)'
file_title = 'nsp_vis_sem_train'
visualize_data(obj_classifier.train_attributes_nsp, obj_classifier.train_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
string_title = 'Non-specific mapping: Visual to semantic (Seen classes-valid)'
file_title = 'nsp_vis_sem_valid'
visualize_data(obj_classifier.valid_attributes_nsp, obj_classifier.valid_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
string_title = 'Non-specific mapping: Visual to semantic (UnSeen classes)'
file_title = 'nsp_vis_sem_test'
visualize_data(obj_classifier.test_attributes_nsp, obj_classifier.test_labels, test_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
#NOTE: Train attributes cross should be weighted summations of all cross-domain
#attributes and non-specific attributes. Folowing to be changed in future.
#obj_classifier.train_attributes_cross = obj_classifier.train_attributes_nsp
#obj_classifier.valid_attributes_cross = obj_classifier.valid_attributes_nsp
#obj_classifier.test_attributes_cross = obj_classifier.test_attributes_nsp

'''
-----------------------------------------------------------------
			Auto-encoders training
-----------------------------------------------------------------

'''
mean_labels = []
ae_features_train_means = np.empty((0, dimension_hidden_layer1_coder), float)
ae_features_valid_means = np.empty((0, dimension_hidden_layer1_coder), float)
ae_features_test_means = np.empty((0, dimension_hidden_layer1_coder), float)


for classI in train_class_labels:
	exp_name = 'aec_feature_class_' + str(classI)
	filename = base_filename + exp_name + 'test.mat'
	if not os.path.isfile(filename):
		print "**************************************"
		classJ = classI	
		cc1_start = time.time()
		obj_input_cc = input_cc()
		obj_input_cc.classI = classI
		obj_input_cc.classJ = classJ
		#NOTE: Change in the visual_features
		obj_input_cc.visual_features = np.vstack((obj_classifier.train_data, obj_classifier.valid_data))
		obj_input_cc.train_valid_split = TR_VA_SPLIT
		obj_input_cc.dataset_train_labels = np.hstack((obj_classifier.train_labels, obj_classifier.valid_labels))
		obj_input_cc.dataset_test_labels = obj_classifier.test_labels
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

		#Train tensorflow cc.....................................
		print "Training coder over %d samples"%(cc1_input_train_perm.shape[0])
		#pdb.set_trace()
		obj_train_tf_cc_input = train_tf_cc_input()
		obj_train_tf_cc_input.classI = classI
		obj_train_tf_cc_input.classJ = classJ
		obj_train_tf_cc_input.dataset_name = dataset_list[DATASET_INDEX] 
		obj_train_tf_cc_input.data_save_path = DATA_SAVE_PATH 
		obj_train_tf_cc_input.dim_feature = obj_input_cc.visual_features.shape[1]
		obj_train_tf_cc_input.cc1_input_train_perm = cc1_input_train_perm
		obj_train_tf_cc_input.cc1_output_train_perm = cc1_output_train_perm
		obj_train_tf_cc_input.cc1_input_valid_perm = function_normalise_data(obj_cc1_train_valid_data.input_valid_perm)
		obj_train_tf_cc_input.cc1_output_valid_perm = function_normalise_data(obj_cc1_train_valid_data.output_valid_perm)
		obj_train_tf_cc_input.obj_classifier = obj_classifier
		obj_train_tf_cc_input.dimension_hidden_layer1 = dimension_hidden_layer1_coder
		obj_train_tf_cc_input.EPOCHS_CC = EPOCHS
		obj_train_tf_cc_output = function_train_tensorflow_cc(obj_train_tf_cc_input)
		#obj_train_tf_cc_output = function_train_keras_cc(obj_train_tf_cc_input)
		
		#pdb.set_trace()			
		#COncatenate features
		if cnt == 0:
			cnt  = 1
			if USE_ENCODER_FEATURES:
				print "Using encoded features"
				#raise ValueError('Check for normalisation if needed')
				cross_features_train = function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)
				cross_features_valid = function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)
				cross_features_test = function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)
			else:
				print "...................Caution : Using decoded features....................."
				cross_features_train = obj_train_tf_cc_output.decoded_data_train_cc1
				cross_features_valid = obj_train_tf_cc_output.decoded_data_valid_cc1
				cross_features_test = obj_train_tf_cc_output.decoded_data_test_cc1
		else:
			if USE_ENCODER_FEATURES:
				print "Using encoded features"
				#raise ValueError('Check for normalisation if needed')
				cross_features_train = np.hstack((cross_features_train, function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)))
				cross_features_valid = np.hstack((cross_features_valid, function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)))
				cross_features_test = np.hstack((cross_features_test, function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)))
			else:
				print "....................Caution : Using decoded features....................."
				cross_features_train = np.hstack((cross_features_train, obj_train_tf_cc_output.decoded_data_train_cc1))
				cross_features_valid = np.hstack((cross_features_valid, obj_train_tf_cc_output.decoded_data_valid_cc1))
				cross_features_test = np.hstack((cross_features_test, obj_train_tf_cc_output.decoded_data_test_cc1))
		cc_end = time.time() 
		print "Processing time for Aec %f"%((cc_end - cc_start))
		
		ae_features_train = function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)
		ae_features_valid = function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)
		ae_features_test = function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)

		#Find cluster centers of AE features
	        ae_features_train_classI_mean, ae_features_valid_classI_mean = get_cluster_centers(ae_features_train, ae_features_valid, ae_features_test, obj_classifier, classI, -1)
		ae_features_train_means = np.append(ae_features_train_means, ae_features_train_classI_mean, axis=0)
		ae_features_valid_means = np.append(ae_features_valid_means, ae_features_valid_classI_mean, axis=0)
		mean_labels.append(classI)

		#Saving aec features
		exp_name = 'aec_feature_class_' + str(classI)
		filename = base_filename + exp_name + 'train' 	
		print"Saving %s"%(filename)
		scipy.io.savemat(filename, dict(ae_feautures_tr = ae_features_train))

		filename = base_filename + exp_name + 'valid'		
		print"Saving %s"%(filename)
		scipy.io.savemat(filename, dict(ae_feautures_val = ae_features_valid))

		filename = base_filename + exp_name + 'test'		
		print"Saving %s"%(filename)
		scipy.io.savemat(filename, dict(ae_feautures_ts = ae_features_test))
		
		string_title = 'AE: Visual to semantic (Seen classes-train)'
		file_title = 'ae_vis_to_sem_train_class_' + str(classI)
		visualize_data(ae_features_train, obj_classifier.train_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
			
		string_title = 'AE: Visual to semantic (Seen classes-valid)'
		file_title = 'ae_vis_to_sem_valid_class_' + str(classI)
		visualize_data(ae_features_valid, obj_classifier.valid_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
		
		string_title = 'AE: Visual to semantic (Seen classes-test)'
		file_title = 'ae_vis_to_sem_test_class_' + str(classI)
		visualize_data(ae_features_test, obj_classifier.test_labels, test_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
	else:
		print "%s exist. Skipping..."%filename

obj_classifier.ae_features_train_means = ae_features_train_means
obj_classifier.ae_features_valid_means = ae_features_valid_means
obj_classifier.ae_features_test_means = ae_features_test_means
'''
------------------------------------------------------------
			Training CE
------------------------------------------------------------
'''
mean_labels_cross = []
ce_features_train_means = np.empty((0, dimension_hidden_layer1_coder), float)
ce_features_valid_means = np.empty((0, dimension_hidden_layer1_coder), float)
ce_features_test_means = np.empty((0, dimension_hidden_layer1_coder), float)

for classI in train_class_labels:
	cnt = 0 #NOTE: cnt is made zero in order to save cross features for each class in different file
	#check if cross-features already calculated
	exp_name = 'cec_features_class_' 
	filename = base_filename + exp_name + 'tr_' + str(classI) + '.mat'		
	for classJ in train_class_labels:
		if (classI != classJ):
			exp_name = 'test_attributes_cross_domain_class_' + str(classJ) + '_' + str(train_class_labels[-1]) 
			filename = base_filename + exp_name + '.mat'
			if not os.path.isfile(filename):					
				print "**************************************"
				#Get data for training CEC.........................
				cc1_start = time.time()
				obj_input_cc = input_cc()
				obj_input_cc.classI = classI
				obj_input_cc.classJ = classJ
				obj_input_cc.visual_features = np.vstack((obj_classifier.train_data, obj_classifier.valid_data))
				obj_input_cc.train_valid_split = TR_VA_SPLIT
				obj_input_cc.dataset_train_labels = np.hstack((obj_classifier.train_labels, obj_classifier.valid_labels))
				obj_input_cc.dataset_test_labels = obj_classifier.test_labels
				obj_input_cc.min_num_samples_per_class = min_num_samples_per_class
			  
				obj_cc1_train_valid_data = function_get_training_data_cc(obj_input_cc)
				cc1_input_train_perm = obj_cc1_train_valid_data.input_train_perm
				INCREASE_FACTOR_CAE = int(NUMBER_OF_SAMPLES_FOR_TRAINING_CODER / cc1_input_train_perm.shape[0])
				print "Increase factor for CEC is %d"%(INCREASE_FACTOR_CAE)
				if INCREASE_FACTOR_CAE > 0: 
					cc1_input_train_perm = np.tile(cc1_input_train_perm, (INCREASE_FACTOR_CAE, 1))
				cc1_input_train_perm = cc1_input_train_perm + NOISE_FACTOR * np.random.normal(0, 1, cc1_input_train_perm.shape)
				cc1_input_train_perm = function_normalise_data(cc1_input_train_perm)

				if classI == classJ:
					cc1_output_train_perm = cc1_input_train_perm
				else:	
					cc1_output_train_perm  = obj_cc1_train_valid_data.output_train_perm
					if INCREASE_FACTOR_CAE > 0: 
						cc1_output_train_perm = np.tile(cc1_output_train_perm, (INCREASE_FACTOR_CAE, 1))
					cc1_output_train_perm = cc1_output_train_perm + NOISE_FACTOR * np.random.normal(0, 1, cc1_output_train_perm.shape)
					cc1_output_train_perm = function_normalise_data(cc1_output_train_perm)

				#Train tensorflow cc.....................................
				print "Training cc over %d samples"%(cc1_input_train_perm.shape[0])
				#pdb.set_trace()
				obj_train_tf_cc_input = train_tf_cc_input()
				obj_train_tf_cc_input.classI = classI
				obj_train_tf_cc_input.classJ = classJ
				obj_train_tf_cc_input.dataset_name = dataset_list[DATASET_INDEX] 
				obj_train_tf_cc_input.data_save_path = DATA_SAVE_PATH 
				obj_train_tf_cc_input.dim_feature = visual_features_dataset.shape[1]
				obj_train_tf_cc_input.cc1_input_train_perm = cc1_input_train_perm
				obj_train_tf_cc_input.cc1_output_train_perm = cc1_output_train_perm
				obj_train_tf_cc_input.cc1_input_valid_perm = function_normalise_data(obj_cc1_train_valid_data.input_valid_perm)
				obj_train_tf_cc_input.cc1_output_valid_perm = function_normalise_data(obj_cc1_train_valid_data.output_valid_perm)
				obj_train_tf_cc_input.obj_classifier = obj_classifier
				obj_train_tf_cc_input.dimension_hidden_layer1 = dimension_hidden_layer1_coder
				obj_train_tf_cc_input.EPOCHS_CC = EPOCHS_CC
				obj_train_tf_cc_output = function_train_tensorflow_cc(obj_train_tf_cc_input)
				#obj_train_tf_cc_output = function_train_keras_cc(obj_train_tf_cc_input)
				
				#pdb.set_trace()			
				#COncatenate features
				if cnt == 0:
					cnt = 1
					if USE_ENCODER_FEATURES:
						print "Using encoded features"
						cross_features_train = function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)
						cross_features_valid = function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)
						cross_features_test = function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)
					else:
						print "........................Caution : Using decoded features....................."
						cross_features_train = obj_train_tf_cc_output.decoded_data_train_cc1
						cross_features_valid = obj_train_tf_cc_output.decoded_data_valid_cc1
						cross_features_test = obj_train_tf_cc_output.decoded_data_test_cc1
				else:
					if USE_ENCODER_FEATURES:
						print "Using enco:ded features"
						cross_features_train = np.hstack((cross_features_train, function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)))
						cross_features_valid = np.hstack((cross_features_valid, function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)))
						cross_features_test = np.hstack((cross_features_test, function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)))
					else:
						print "........................Caution : Using decoded features....................."
						cross_features_train = np.hstack((cross_features_train, obj_train_tf_cc_output.decoded_data_train_cc1))
						cross_features_valid = np.hstack((cross_features_valid, obj_train_tf_cc_output.decoded_data_valid_cc1))
						cross_features_test = np.hstack((cross_features_test, obj_train_tf_cc_output.decoded_data_test_cc1))
						
				latent_features_train = function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)
				latent_features_valid = function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)
				latent_features_test = function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)
				
				ce_features_train_classIJ_mean, ce_features_valid_classIJ_mean = get_cluster_centers(latent_features_train, latent_features_valid, latent_features_test, obj_classifier, classI, classJ)
				ce_features_train_means = np.append(ce_features_train_means, ce_features_train_classIJ_mean, axis=0)
				ce_features_valid_means = np.append(ce_features_valid_means, ce_features_valid_classIJ_mean, axis=0)
				mean_labels_cross.append([classI, classJ])
				#Saving cross features
				exp_name = 'latent_cross_features_train_class_' + str(classI) + '_' + str(classJ) 
				filename = base_filename + exp_name 		
				print"Saving %s"%(filename)
				scipy.io.savemat(filename, dict(latent_feautures_tr = latent_features_train))
					
				exp_name = 'latent_cross_features_valid_class_' + str(classI) + '_' + str(classJ) 
				filename = base_filename + exp_name 		
				print"Saving %s"%(filename)
				scipy.io.savemat(filename, dict(latent_feautures_val = latent_features_valid))
			
				exp_name = 'latent_cross_features_test_class_' + str(classI) + '_' + str(classJ) 
				filename = base_filename + exp_name 		
				print"Saving %s"%(filename)
				scipy.io.savemat(filename, dict(latent_feautures_ts = latent_features_test))

				string_title = 'CE: Visual to semantic (Seen classes-train) Class ' + str(classI) + ' to Class ' + str(classJ)
				file_title = 'ce_vis_to_sem_train_class_' + str(classI) + '_to_' + str(classJ)
				visualize_data(latent_features_train, obj_classifier.train_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
				
				string_title = 'CE: Visual to semantic (Seen classes-valid) Class ' + str(classI) + ' to Class ' + str(classJ)
				file_title = 'ce_vis_to_sem_valid_class_' + str(classI) + '_to_' + str(classJ)
				visualize_data(latent_features_valid, obj_classifier.valid_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
				
				string_title = 'CE: Visual to semantic (Seen classes-test) Class ' + str(classI) + ' to Class ' + str(classJ)
				file_title = 'ce_vis_to_sem_test_class_' + str(classI) + '_to_' + str(classJ)
				visualize_data(latent_features_test, obj_classifier.test_labels, test_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
				
				for classK in train_class_labels:
					if classK != classJ:
						exp_name = 'test_attributes_cross_domain_class_' + str(classJ) + '_' + str(classK) 
						filename = base_filename + exp_name + '.mat' 		
						if not os.path.isfile(filename):
							train_attributes_cross, valid_attributes_cross, test_attributes_cross = train_visual_to_semantic_specific_regressor(latent_features_train, latent_features_valid, latent_features_test, obj_classifier, classJ, classK)
							
							#Saving cross features
							exp_name = 'train_attributes_cross_domain_class_' + str(classJ) + '_' + str(classK) 
							filename = base_filename + exp_name 		
							print"Saving %s"%(filename)
							scipy.io.savemat(filename, dict(train_attributes_cross = train_attributes_cross))
								
							exp_name = 'valid_attributes_cross_domain_class_' + str(classJ) + '_' + str(classK) 
							filename = base_filename + exp_name 		
							print"Saving %s"%(filename)
							scipy.io.savemat(filename, dict(valid_attributes_cross = valid_attributes_cross))
						
							exp_name = 'test_attributes_cross_domain_class_' + str(classJ) + '_' + str(classK) 
							filename = base_filename + exp_name 		
							print"Saving %s"%(filename)
							scipy.io.savemat(filename, dict(test_attributes_cross = test_attributes_cross))
					
							string_title = 'CE: Visual to semantic cross domain (Seen classes-train) Class ' + str(classJ) + ' to Class ' + str(classK)
							file_title = 'ce_vis_to_sem_cross_domain_train_class_' + str(classJ) + '_to_' + str(classK)
							visualize_data(train_attributes_cross, obj_classifier.train_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
							
							string_title = 'CE: Visual to semantic cross domain (Seen classes-valid) Class ' + str(classJ) + ' to Class ' + str(classK)
							file_title = 'ce_vis_to_sem_cross_domain_valid_class_' + str(classJ) + '_to_' + str(classK)
							visualize_data(valid_attributes_cross, obj_classifier.valid_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
							
							string_title = 'CE: Visual to semantic cross domain (Seen classes-test) Class ' + str(classJ) + ' to Class ' + str(classK)
							file_title = 'ce_vis_to_sem_cross_domain_test_class_' + str(classJ) + '_to_' + str(classK)
							visualize_data(test_attributes_cross, obj_classifier.test_labels, test_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
						else:
							print "%s already exists. Skipping..."%filename
			else:
				print "%s already exists. Skipping..."%filename
		cc_end = time.time() 
		print "Processing time for cc %f"%((cc_end - cc_start))
#**********************************	
print "Time for cc %f"%(cc_start - cc_end)	


obj_classifier.ce_features_train_means = ce_features_train_means
obj_classifier.ce_features_valid_means = ce_features_valid_means
obj_classifier.ce_features_test_means = ce_features_test_means
'''
-----------------------------------------------------------------
		Combine AE and CE attriutes
-----------------------------------------------------------------
'''
obj_classifier.train_attributes_cross = np.zeros((obj_classifier.train_attributes.shape), float)
obj_classifier.valid_attributes_cross = np.zeros((obj_classifier.valid_attributes.shape), float)
obj_classifier.test_attributes_cross = np.zeros((number_of_train_classes * obj_classifier.test_attributes.shape[0], obj_classifier.test_attributes.shape[1]), float)
combine_specific_nonspecific_attributes(obj_classifier, mean_labels, mean_labels_cross, base_filename)

string_title = 'Combined attributes (Seen classes-train)'
file_title = 'combined_attr_train'
visualize_data(obj_classifier.train_attributes_cross, obj_classifier.train_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)

string_title = 'Combined attributes (Seen classes-valid)'
file_title = 'combined_attr_valid'
visualize_data(obj_classifier.valid_attributes_cross, obj_classifier.valid_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)

end = 0
for k in train_class_labels:
	strt = end
	end = strt + obj_classifier.test_attributes.shape[0]	
	string_title = 'Combined attributes (Seen classes-test) using train Class ' + str(k)
	file_title = 'combined_attr_test_using_train_class_' + str(k)
	visualize_data(obj_classifier.test_attributes_cross, obj_classifier.test_labels, test_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
	
'''
-----------------------------------------------------------------
		Regression f: semantic(seen) --> semantic(unseen)
-----------------------------------------------------------------
'''
k = 0
for seen_class in train_class_labels:
	for unseen_class in test_class_labels:
		print "..........Mapping seen class(%d) to unseen class(%d).............."%(seen_class, unseen_class)	
		train_attributes_cross_mapped, valid_attributes_cross_mapped, test_attributes_cross_mapped = train_seen_to_unseen_regressor(obj_classifier, seen_class, unseen_class, k)
		
		#Save semantic cross mapped features
		filename = base_filename + 'train_attributes_cross_mapped_seen_class_' + str(seen_class) + '_to_unseen_class_' + str(unseen_class)
		scipy.io.savemat(filename, dict(train_attributes_cross_mapped = train_attributes_cross_mapped))
		print filename

		filename = base_filename + 'valid_attributes_cross_mapped_seen_class_' + str(seen_class) + '_to_unseen_class_' + str(unseen_class)
		scipy.io.savemat(filename, dict(valid_attributes_cross_mapped = valid_attributes_cross_mapped))
		print filename

		filename = base_filename + 'test_attributes_cross_mapped_seen_class_' + str(seen_class) + '_to_unseen_class_' + str(unseen_class)
		scipy.io.savemat(filename, dict(test_attributes_cross_mapped = test_attributes_cross_mapped))
		print filename
		
		string_title = 'Seen to unseen classes mapping train: Class ' + str(seen_class) + ' to Class ' + str(unseen_class)
		file_title = 'seen_to_unseen_train_class_' + str(seen_class) + '_to_class_' + str(unseen_class)
		visualize_data(train_attributes_cross_mapped, obj_classifier.train_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
		string_title = 'Seen to unseen classes mapping valid: Class ' + str(seen_class) + ' to Class ' + str(unseen_class)
		file_title = 'seen_to_unseen_valid_class_' + str(seen_class) + '_to_class_' + str(unseen_class)
		visualize_data(valid_attributes_cross_mapped, obj_classifier.valid_labels, train_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
		
		string_title = 'Seen to unseen classes mapping test: Class ' + str(seen_class) + ' to Class ' + str(unseen_class)
		file_title = 'seen_to_unseen_test_class_' + str(seen_class) + '_to_class_' + str(unseen_class)
		visualize_data(test_attributes_cross_mapped, obj_classifier.test_labels, test_class_labels, class_names, string_title, file_title, IMAGE_DATA_SAVE_PATH)
	k = k +1
 

obj_classifier.train_class_labels = train_class_labels
obj_classifier.test_class_labels = test_class_labels
obj_classifier.number_of_train_classes = number_of_train_classes
obj_classifier.number_of_test_classes = number_of_test_classes
obj_classifier.base_filename = base_filename
filename = base_filename + 'obj_classifier.p'
print filename
pickle.dump(obj_classifier, open(filename, "wb"))
pdb.set_trace()
'''
-----------------------------------------------------------------
		Classifier
-----------------------------------------------------------------
'''

distance_mat = np.zeros((obj_classifier.test_attributes_nsp.shape[0], obj_classifier.number_of_test_classes), float)
k = 0
#NOTE: These weights to be learned.
weights = np.ones((obj_classifier.number_of_train_classes, 1), float)
#weights = weights.flatten()
weights = weights/obj_classifier.number_of_train_classes
for unseen_class in obj_classifier.test_class_labels:
	sqrd_distances_sp_l = []#np.zeros(obj_classifier.test_attributes_nsp.shape[0])
	unseen_prototype = obj_classifier.prototypes[unseen_class - 1, :]
	unseen_proto_mat = np.tile(unseen_prototype, (obj_classifier.test_attributes_nsp.shape[0], 1))
	sqrd_distances_nsp = np.sum(np.square(unseen_proto_mat - obj_classifier.test_attributes_nsp), axis=1)
	
	for seen_class in obj_classifier.train_class_labels:
	        filename = base_filename + 'test_attributes_cross_mapped_seen_class_' + str(seen_class) + '_to_unseen_class_' + str(unseen_class)
		tmp = scipy.io.loadmat(filename)
                test_attributes_cross_mapped = tmp['test_attributes_cross_mapped']
		sqrd_distances_sp_l.append(np.sum(np.square(unseen_proto_mat - test_attributes_cross_mapped), axis=1))
	
	sqrd_distances_sp_arr = np.asarray(sqrd_distances_sp_l)
	#sqrd_distances_sp_arr = np.reshape(sqrd_distances_sp_arr, (obj_classifier.test_attributes_nsp.shape[0], number_of_train_classes))
	sqrd_distances_sp_arr = weights*sqrd_distances_sp_arr
	sqrd_distances_sp = np.sum(sqrd_distances_sp_arr, axis=0)
	distance_mat[:, k] = sqrd_distances_nsp + sqrd_distances_sp
	k = k + 1

indices_predicted_unseen_classes = np.argmin(distance_mat, axis = 1)
predicted_unseen_classes = obj_classifier.test_class_labels[indices_predicted_unseen_classes]
accuracy = accuracy_score(obj_classifier.test_labels, predicted_unseen_classes)
print "Dataset %s: Accuracy %f"%(obj_input_data.dataset_name, accuracy)	
	
		
		



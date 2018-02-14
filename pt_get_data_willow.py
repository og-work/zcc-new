
'''
------------------------------------------------------------------------
------------------------------------------------------------------------
			pt_get_data_willow.py
------------------------------------------------------------------------
------------------------------------------------------------------------
'''
import numpy as np
import pdb
import scipy.io

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt



class output_cc:
        input_train_perm = np.array([])
        output_train_perm = np.array([])
        input_valid_perm = np.array([])
        output_valid_perm = np.array([])
        input_test_perm = np.array([])

        input_train = np.array([])
        input_valid = np.array([])
        input_test = np.array([])
        output_valid = np.array([])

        indices_input_samples_train_perm = np.array([])
        indices_ouput_samples_train_perm = np.array([])
        indices_input_samples_test_perm = np.array([])

        indices_classI_samples_train = np.array([])
        indices_classI_samples_test = np.array([])
        indices_classJ_samples_test = np.array([])
        indices_classI_samples_valid = np.array([])

        neg_samples_pool = np.array([])
        neg_samples_pool_lables = np.array([])

        def function(self):
                print("This is output_cc class")

class input_cc:
        classI = []
        classJ = []
        train_valid_split = np.array([])
        visual_features = np.array([])
        dataset_labels = np.array([])
        dataset_train_labels = np.array([])
        dataset_test_labels = np.array([])
        min_num_samples_per_class = []
        number_of_classes = []
	train_class_labels = []

        def function(self):
                print("This is input_cc class")

class input_data:
	dataset_name = []
	system_type = []
	train_class_labels = np.array([])
	test_class_labels = np.array([])
	visual_features_dataset = np.array([])
	attributes = np.array([])
	dataset_labels = np.array([])
	dataset_train_labels = np.array([])
	dataset_test_labels = np.array([])
	train_sample_indices = np.array([])
	test_sample_indices = np.array([])
	def function(self):
		print("This is the input_data class")	

class classifier_data():
	train_data = []
	valid_data = []
	test_data = []
	train_labels = []	
	valid_labels = []	
	test_labels = []
	cross_features_train = np.array([])	
	cross_features_valid = np.array([])	
	cross_features_test = np.array([])	
	epochs = []
	number_of_train_classes = []
	dim_hidden_layer1 = []
		
	def function(self):
		print "This is a classifer data object..."

class train_pt_cc_input:
        cc1_input_train_perm = np.array([])
        cc1_output_train_perm = np.array([])
        cc1_input_valid_perm = np.array([])
        cc1_output_valid_perm = np.array([])
        obj_classifier_data = classifier_data()
        dimension_hidden_layer1 = []
        EPOCHS_CC = []
        input_class = []
        output_class = []
        dataset_name = []
        data_save_path = []
        dim_feature = []
        def function(self):
                print("This is train pytorch cc input class")

class train_pt_cc_output:
        decoded_data_train_cc1 = np.array([])
        encoded_data_train_cc1 = np.array([])
        decoded_data_valid_cc1 = np.array([])
        encoded_data_valid_cc1 = np.array([])
        decoded_data_test_cc1 = np.array([])
        encoded_data_test_cc1 = np.array([])
        obj_classifier = classifier_data()

        def function(self):
                print("This is train_pt_cc_output class")


def function_get_input_data(obj_input_data):

	BASE_PATH = "/nfs4/omkar/Documents/" 
	print(BASE_PATH)
	if obj_input_data.dataset_name == 'WILLOW':
		if 0:
			print "NO bounding box for willow..."
			path_CNN_features = BASE_PATH + "study/phd-research/data/dataset/stanford-action-40/features/willow_features"
			tmp = scipy.io.loadmat(path_CNN_features)
			visual_features_dataset = tmp_features['features']
		else:
			print "Using bounding box for willow..."
			path_CNN_features = BASE_PATH + "study/phd-research/data/datasets/WILLOW/willow_features_labels_bb"
			tmp = scipy.io.loadmat(path_CNN_features)
			visual_features_dataset = tmp['features']
			dataset_labels = np.array([])
			dataset_train_labels = tmp['dataset_train_labels']	
			dataset_test_labels = tmp['dataset_test_labels']	
			dataset_train_labels = dataset_train_labels.flatten()
			dataset_test_labels = dataset_test_labels.flatten()
			train_class_labels = np.arange(1, 8, 1)
			test_class_labels = train_class_labels
	else:
		raise ValueError('Unknown Dataset')

	obj_input_data.test_class_labels = test_class_labels 
	obj_input_data.train_class_labels = train_class_labels 
	#obj_input_data.attributes = attributes 
	obj_input_data.visual_features_dataset = visual_features_dataset
	obj_input_data.dataset_labels = dataset_labels
	obj_input_data.dataset_train_labels = dataset_train_labels
	obj_input_data.dataset_test_labels = dataset_test_labels
	obj_input_data.train_sample_indices = np.array([])
	obj_input_data.test_sample_indices = np.array([])

	return obj_input_data

def function_normalise_data(unnormalised_data):
	norm_type = 2
	if norm_type == 1:	
		print "Normalisation between 0 to 1 ..."
		raise ValueError('Need to normalise the decoded/encoded features while using')
		max_val_array = unnormalised_data.max(axis = 0)
		max_val_array[max_val_array == 0] = 1.
		max_val_mat = np.tile(max_val_array, (unnormalised_data.shape[0], 1))
		normalised_data = unnormalised_data/max_val_mat
		#Normalise entire data between [0,1]
		#if unnormalised_data.shape[0] != 0:
		#	if (unnormalised_data.max() != unnormalised_data.min()):
		#		normalised_data = np.divide((unnormalised_data - unnormalised_data.min()), (unnormalised_data.max() - unnormalised_data.min()))
		#	else:
		#		normalised_data = unnormalised_data * 0
		#else:
		#	normalised_data = []
		raise NameError("Something went wrong in normalisation...") 
	elif norm_type == 2:
		#print "Normalisation between -1 to 1 ..."

		if unnormalised_data.shape[0] != 0:
			if (unnormalised_data.max() != unnormalised_data.min()):
				normalised_data = np.divide((unnormalised_data - unnormalised_data.min()), (unnormalised_data.max() - unnormalised_data.min()))
			else:
				normalised_data = unnormalised_data * 0
			normalised_data = normalised_data * 2 - 1
		else:
			normalised_data = []
			raise NameError("Something went wrong in normalisation. Number of samples can not be zero") 
		
	else:
		print "No normalisation ..."
		raise ValueError('Need to normalise the decoded/encoded features while using')
		normalised_data = unnormalised_data
	return normalised_data



def function_reduce_dimension_of_data(source_data, target_data, REDUCED_DIMENSION): 
	print "Doing PCA to reduce dimension from %d to %d"%(source_data.shape[1], REDUCED_DIMENSION)
	#scale function scales the data to have zero mean and unit variance
	source_data_norm = scale(source_data)
	target_data_norm = scale(target_data)
	pca = PCA(n_components=REDUCED_DIMENSION)
	#pca.fit(visual_features_dataset_norm)
	source_data_PCAed = pca.fit_transform(source_data_norm)	
	#The amount of variance that each PC explains (lambda_i/sum_i(lambda_i))
	var= pca.explained_variance_ratio_
	#Cumulative Variance explains
	var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
	#pdb.set_trace()
	if 1:
		plt.figure(1)
		plt.plot(var1)
		plt.ylabel('var1')
		plt.grid()
		plt.figure(2)
		plt.plot(var)
		plt.ylabel('var')
		plt.grid()
		plt.figure(3)
		plt.plot(source_data_norm.mean(axis=0))
		plt.ylabel('source_data_norm.mean')
		plt.grid()
		plt.figure(4)
		plt.plot(source_data_norm.std(axis=0))
		plt.ylabel('source_data_norm.std')
		plt.grid()
		#plt.show()

	target_data_PCAed = pca.transform(target_data_norm)	
	visual_features_dataset_PCAed = np.vstack((source_data_PCAed, target_data_PCAed))

	if 0:
		plt.figure(5)
		plt.plot(visual_features_dataset.std(axis=0))
		plt.ylabel('visual_features_dataset_norm.std')
		plt.grid()
        #visual_features_dataset_norm = StandardScaler().fit_transform(visual_features_dataset_ori)
	#visual_features_dataset = PCA(n_components = REDUCED_DIMENSION_VISUAL_FEATURE).fit_transform(visual_features_dataset_norm)
	#visual_features_dataset = function_normalise_data(visual_features_dataset)
	
	return visual_features_dataset_PCAed

def function_get_training_data_cc(obj_input_cc):
	#pdb.set_trace()
	classI = obj_input_cc.classI
	classJ = obj_input_cc.classJ
	TR_TS_VA_SPLIT = obj_input_cc.train_valid_split
	dataset_train_labels = obj_input_cc.dataset_train_labels
	visual_features_dataset = obj_input_cc.visual_features
	number_of_classes = obj_input_cc.number_of_classes

	print "************* Class %d >>> Class %d *************************"%(classI, classJ)
	MIN_NUMBER_OF_SAMPLES_OF_CLASS = obj_input_cc.min_num_samples_per_class #51 # class27 51 samples for apy class 1-32: 30

	indices_classI_samples = np.flatnonzero(dataset_train_labels == classI)
	indices_classJ_samples = np.flatnonzero(dataset_train_labels == classJ)

	if classI == classJ:
		number_of_samples_classI_for_train = int(TR_TS_VA_SPLIT[0] * np.size(indices_classI_samples))
		number_of_samples_classJ_for_train = int(TR_TS_VA_SPLIT[0] * np.size(indices_classJ_samples))
	else: 	
		number_of_samples_classI_for_train = int(TR_TS_VA_SPLIT[0] * MIN_NUMBER_OF_SAMPLES_OF_CLASS)
		number_of_samples_classJ_for_train = int(TR_TS_VA_SPLIT[0] * MIN_NUMBER_OF_SAMPLES_OF_CLASS)
	
	indices_classI_samples_train = indices_classI_samples[:number_of_samples_classI_for_train]
	indices_classJ_samples_train = indices_classJ_samples[:number_of_samples_classJ_for_train]
	
	number_of_samples_classI_for_valid = int(TR_TS_VA_SPLIT[1] * np.size(indices_classI_samples))
	number_of_samples_classJ_for_valid = int(TR_TS_VA_SPLIT[1] * np.size(indices_classJ_samples))

	start_vl_classI = number_of_samples_classI_for_train
	end_vl_classI = start_vl_classI + number_of_samples_classI_for_valid

	start_vl_classJ = number_of_samples_classJ_for_train
	end_vl_classJ = start_vl_classJ + number_of_samples_classJ_for_valid
	
	indices_classI_samples_valid = indices_classI_samples[start_vl_classI:end_vl_classI]
	indices_classJ_samples_valid = indices_classJ_samples[start_vl_classJ:end_vl_classJ]

	print "Total class %d samples %d "%(classI, np.size(indices_classI_samples))
	print "data split tr %d valid %d" %(number_of_samples_classI_for_train, \
	    number_of_samples_classI_for_valid)
	print "Total class %d samples %d "%(classJ, np.size(indices_classJ_samples))
	print "data split tr %d valid %d" %(number_of_samples_classJ_for_train, \
	    number_of_samples_classJ_for_valid)

	#Prepare train data for cc
	indices_input_samples_train = np.array([])
	indices_output_samples_train = np.array([])
		
	if classI != classJ:	
		for index_classI_sample in indices_classI_samples_train:
			indices_classI_samples_array = np.empty(indices_classJ_samples_train.size)
			indices_classI_samples_array.fill(index_classI_sample)
			indices_input_samples_train = np.concatenate((indices_input_samples_train, indices_classI_samples_array), axis = 0)
			indices_output_samples_train = np.concatenate((indices_output_samples_train, indices_classJ_samples_train), axis = 0)

		#Prepare validation data for cc
		indices_input_samples_valid = np.array([])
		indices_output_samples_valid = np.array([])
	
		for index_classI_sample in indices_classI_samples_valid:
			indices_classI_samples_array = np.empty(indices_classJ_samples_valid.size)
			indices_classI_samples_array.fill(index_classI_sample)
			indices_input_samples_valid = np.concatenate((indices_input_samples_valid, indices_classI_samples_array), axis = 0)
			indices_output_samples_valid = np.concatenate((indices_output_samples_valid, indices_classJ_samples_valid), axis = 0)

	else:
		indices_input_samples_train = indices_classI_samples_train		
		indices_output_samples_train = indices_classJ_samples_train	
		indices_input_samples_valid = indices_classI_samples_valid		
		indices_output_samples_valid = indices_classJ_samples_valid	

	print "Number of samples for CC train %d, for validation %d" %(np.size(indices_input_samples_train), np.size(indices_input_samples_valid))			
	#unit test	
	if (np.size(indices_input_samples_train) != np.size(indices_output_samples_train)):
		raise NameError('Input and output data dimensions are not matching for CC')

	obj_output_cc = output_cc()
	#permuted data
	obj_output_cc.input_train_perm = visual_features_dataset[indices_input_samples_train.astype(int), :]
	obj_output_cc.output_train_perm = visual_features_dataset[indices_output_samples_train.astype(int), :]
	obj_output_cc.input_valid_perm = visual_features_dataset[indices_input_samples_valid.astype(int), :]
	obj_output_cc.output_valid_perm = visual_features_dataset[indices_output_samples_valid.astype(int), :]
	#permuted indices
	obj_output_cc.indices_input_samples_train_perm = indices_input_samples_train
	obj_output_cc.indices_ouput_samples_train_perm = indices_output_samples_train
	obj_output_cc.indices_input_samples_valid_perm = indices_input_samples_valid
	obj_output_cc.indices_output_samples_valid_perm = indices_output_samples_valid
	
	#Negative samples i.e. Different than classJ (output class) samples
        number_of_negative_samples_per_class = 50
        neg_samples_pool = []
        neg_samples_labels = []
        for neg_class in range(1, number_of_classes + 1):
                if neg_class != classJ:
                        neg_class_ind = np.flatnonzero(dataset_train_labels == neg_class)
                        neg_class_ind = neg_class_ind.astype(int)
                        ind_few_samples = neg_class_ind[:number_of_negative_samples_per_class]
                        ind_few_samples = ind_few_samples.flatten()
                        neg_samples_pool.append(visual_features_dataset[ind_few_samples, :])
                        neg_samples_labels.append([neg_class]*number_of_negative_samples_per_class)
        neg_samples_pool = np.asarray(neg_samples_pool)
        obj_output_cc.neg_samples_pool = neg_samples_pool.reshape(neg_samples_pool.shape[0] * neg_samples_pool.shape[1], neg_samples_pool.shape[2])
        obj_output_cc.neg_samples_pool_labels = neg_samples_labels
		
	return obj_output_cc

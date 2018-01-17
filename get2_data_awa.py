'''
	get2_data_awa.py
'''

import numpy as np
import pdb
import scipy.io
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.layers import Dropout


from sklearn.metrics.pairwise import euclidean_distances
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import time
import os

MIN_SAMPLES_FOR_TRAIN = 64 #70% of 92 samples. Min samples 92 for class 12 in AwA
MIN_SAMPLES_FOR_VALID = 27 #30% of 92 samples. Min samples 92 for class 12 in AwA
EPOCHS = 300

GPU_PERCENTAGE = 0.6
if 1:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config.gpu_options.per_process_gpu_memory_fraction = GPU_PERCENTAGE
    set_session(tf.Session(config=config))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
#tensorboard = TensorBoard(log_dir=".".format(time()))
#tensorboard = TensorBoard(log_dir=path_cross_features, histogram_freq=1, write_graph=True,write_images=True,write_grads=False)
callbacks_list = [reduce_lr]#, early_stopping]

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

	def function(self):
		print("This is input_cc class")

def function_get_training_data_cc(obj_input_cc):
	#pdb.set_trace()
	classI = obj_input_cc.classI
	classJ = obj_input_cc.classJ
	TR_TS_VA_SPLIT = obj_input_cc.train_valid_split
	dataset_train_labels = obj_input_cc.dataset_train_labels
	dataset_train_labels = dataset_train_labels.flatten()
	visual_features_dataset = obj_input_cc.visual_features

	print "************* Class %d >>> Class %d *************************"%(classI, classJ)
	MIN_NUMBER_OF_SAMPLES_OF_CLASS = obj_input_cc.min_num_samples_per_class 

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
	#pdb.set_trace()
	#permuted indices
	obj_output_cc.indices_input_samples_train_perm = indices_input_samples_train
	obj_output_cc.indices_ouput_samples_train_perm = indices_output_samples_train
	obj_output_cc.indices_input_samples_valid_perm = indices_input_samples_valid
	obj_output_cc.indices_output_samples_valid_perm = indices_output_samples_valid
			
	return obj_output_cc

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
	class_names = []
	def function(self):
		print("This is the input_data class")	

def function_normalise_data(unnormalised_data):
	#Norimalise along each dimension separately
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
		print "Normalisation between -1 to 1 ..."

		if unnormalised_data.shape[0] != 0:
			if (unnormalised_data.max() != unnormalised_data.min()):
				normalised_data = np.divide((unnormalised_data - unnormalised_data.min()), (unnormalised_data.max() - unnormalised_data.min()))
			else:
				normalised_data = unnormalised_data * 0
			normalised_data = normalised_data * 2 - 1
		else:
			normalised_data = []
			raise NameError("Something went wrong in normalisation...") 
		
	else:
		print "No normalisation ..."
		raise ValueError('Need to normalise the decoded/encoded features while using')
		normalised_data = unnormalised_data
	return normalised_data

def function_get_input_data(obj_input_data):

	BASE_PATH = "/nfs4/omkar/Documents/" 
	print(BASE_PATH)
	if obj_input_data.dataset_name == 'AwA':
		print "Loading data for %s"%(obj_input_data.dataset_name)
		path_features = BASE_PATH + "study/phd-research/data/zsl-data/Synthesized-classifier-cvpr2016/SynC/data/AwA_googlenet"
		data = scipy.io.loadmat(path_features)
		visual_features_dataset = data['X']
		path_info = BASE_PATH + "study/phd-research/data/zsl-data/Synthesized-classifier-cvpr2016/SynC/data/AWA_inform_release"
		tmp = scipy.io.loadmat(path_info)
		test_locations = tmp['te_loc']
		train_locations = tmp['tr_loc']
		train_labels = tmp['y_tr']
		test_labels = tmp['y_te']
		path_info_attr = BASE_PATH + "study/phd-research/data/zsl-data/Synthesized-classifier-cvpr2016/SynC/data/AwA_attr2_new"
		tmp_attr = scipy.io.loadmat(path_info_attr)
		
		attributes = tmp_attr['attr2']

		#Python indexing starts from 0
		train_locations = train_locations - 1
		test_locations = test_locations - 1
	
		number_of_samples_in_dataset = visual_features_dataset.shape[0]
                dataset_labels = np.array([])
                dataset_train_labels = -1 * np.ones(number_of_samples_in_dataset)
                dataset_test_labels = -1 * np.ones(number_of_samples_in_dataset)
                dataset_train_labels[train_locations] = train_labels
                dataset_test_labels[test_locations] = test_labels
		#Standard AwA seen-unseen split
		#train_class_labels = np.arange(0, 50, 1)
		#test_class_labels = np.array([6, 14, 15, 18, 24, 25, 34, 39, 42, 48])
		
		class_names = ['01(S): antelope', '02(S): grizzly+bear', '03(S): killer+whale', '04(S): beaver', 
				'05(S): dalmatian', '06(U): persian+cat', '07(S): horse', '08(S): german+shepherd',
				'09(S): blue+whale', '10(S): siamese+cat', '11(S): skunk', '12(S): mole', '13(S): tiger',
				'14(U): hippopotamus', '15(U): leopard', '16(S): moose', '17(S): spider+monkey',
				'18(U): humpback+whale', '19(S): elephant', '20(S): gorilla', '21(S): ox',
				'22(S): fox', '23(S): sheep', '24(U): seal', '25(U): chimpanzee', '26(S): hamster',
				'27(S): squirrel', '28(S): rhinoceros', '29(S): rabbit', '30(S): bat', 
				'31(S): giraffe', '32(S): wolf', '33(S): chihuahua', '34(U): rat', '35(S): weasel',
				'36(S): otter', '37(S): buffalo', '38(S): zebra', '39(U): giant+panda', '40(S): deer',
				'41(S): bobcat', '42(U): pig', '43(S): lion', '44(S): mouse', '45(S): polar+bear', '46(S): collie',
				'47(S): walrus', '48(U): raccoon', '49(S): cow', '50(S): dolphin']	
		#train_class_labels = np.array([0, 1, 2, 3, 10, 20, 21, 22, 27, 28, 29, 45, 46])
		#test_class_labels = np.array([6, 14, 15, 18, 24, 34])
		train_class_labels = np.array([0, 2, 28, 45])
		test_class_labels = np.array([42, 48])
		test_class_labels = test_class_labels - 1
		train_class_labels = np.delete(train_class_labels, test_class_labels)
		train_class_labels = train_class_labels + 1
		test_class_labels = test_class_labels + 1
	else:
		raise ValueError('Invalid dataset')
	
	#pdb.set_trace()	
	obj_input_data.class_names = class_names
	obj_input_data.test_class_labels = test_class_labels 
	obj_input_data.train_class_labels = train_class_labels 
	obj_input_data.attributes = attributes 
	obj_input_data.visual_features_dataset = visual_features_dataset
	obj_input_data.dataset_labels = np.array([])
	obj_input_data.dataset_train_labels = dataset_train_labels
	obj_input_data.dataset_test_labels = dataset_test_labels

	return obj_input_data

class classifier_data():
	base_filename = []
	number_of_train_classes = []
	number_of_test_classes = []
	test_class_labels = []
	train_class_labels = []
	train_data = []
	valid_data = []
	test_data = []
	train_labels = []	
	valid_labels = []	
	test_labels = []
	train_attributes = []
	valid_attributes = []
	test_attributes = []
	train_attributes_nsp = []
	valid_attributes_nsp = []
	test_attributes_nsp = []
	train_attributes_cross = []
	valid_attributes_cross = []
	test_attributes_cross = []
	cross_features_train = np.array([])	
	cross_features_valid = np.array([])	
	cross_features_test = np.array([])	
	ae_features_train_means = []
	ae_features_valid_means = []
	ae_features_test_means = []
	ce_features_train_means = []
	ce_features_valid_means = []
	ce_features_test_means = []
	epochs = []
	number_of_train_classes = []
	dim_hidden_layer1 = []
	prototypes = []
			
	def function(self):
		print "This is a classifer data object..."


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

def baseline_model_regression_seen_to_unseen(visual_dim, semantic_dim):
	DIM1 = int(visual_dim * 0.5)
	model = Sequential()
	model.add(Dense(DIM1, input_dim=visual_dim, kernel_initializer='normal', activation='tanh'))
	model.add(Dropout(0.4))
	
	if DIM1*0.5 > semantic_dim:
		DIM2 = int(DIM1 * 0.5)
		model.add(Dense(DIM2, input_dim=visual_dim, kernel_initializer='normal', activation='tanh'))
		model.add(Dropout(0.4))
		if DIM2*0.5 > semantic_dim:
			DIM3 = int(DIM2 * 0.5)
			model.add(Dense(DIM3, input_dim=visual_dim, kernel_initializer='normal', activation='tanh'))
			model.add(Dropout(0.4))

	
	model.add(Dense(semantic_dim, kernel_initializer='normal'))
	OPTIMIZER_TYPE = 'SGD'
	if OPTIMIZER_TYPE == 'SGD':
            print "Using SGD optimizer..."
            OPTIMIZER = SGD(lr=0.05, decay=1e-7, momentum = 0.09, nesterov=True)
        elif OPTIMIZER_TYPE == 'adam':
            print "Using adam optimizer..."
            OPTIMIZER = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-7)

	model.compile(loss='mean_squared_error', optimizer=OPTIMIZER)
	print model.summary()
	return model

def baseline_model_regression(visual_dim, semantic_dim):
	model = Sequential()
	DIM1 = int(visual_dim * 0.5)
	model.add(Dense(DIM1, input_dim=visual_dim, kernel_initializer='normal', activation='tanh'))
	model.add(Dropout(0.4))
	if DIM1 * 0.5 > semantic_dim:
		DIM2 = int(DIM1 * 0.5)
		model.add(Dense(DIM2, input_dim=visual_dim, kernel_initializer='normal', activation='tanh'))
		model.add(Dropout(0.4))
		if DIM2 * 0.5 > semantic_dim:
			DIM3 = int(DIM2 * 0.5)
			model.add(Dense(DIM3, input_dim=visual_dim, kernel_initializer='normal', activation='tanh'))
			model.add(Dropout(0.4))
	model.add(Dense(semantic_dim, kernel_initializer='normal'))
	OPTIMIZER_TYPE = 'adam'
	if OPTIMIZER_TYPE == 'SGD':
            print "Using SGD optimizer..."
            OPTIMIZER = SGD(lr=0.05, decay=1e-7, momentum = 0.09, nesterov=True)
        elif OPTIMIZER_TYPE == 'adam':
            print "Using adam optimizer..."
            OPTIMIZER = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-7)

	model.compile(loss='mean_squared_error', optimizer=OPTIMIZER)
	print model.summary()
	return model


'''
-----------------------------------------------------------------
                   Regression non-specific f: visual --> semantic
-----------------------------------------------------------------
'''
def train_visual_to_semantic_nonspecific_regressor(obj):
	USE_SVR = 0
	if USE_SVR:
		num_regressors = obj.train_attributes.shape[1]
		train_data_semantic = np.empty((0, obj.train_data.shape[0]), float)	
		valid_data_semantic = np.empty((0, obj.valid_data.shape[0]), float)	
		test_data_semantic = np.empty((0, obj.test_data.shape[0]), float)	

		for k in range(num_regressors):
			print "Training Visual to semantic mapping function for dimension %d"%k
			st = time.time()
			clf = SVR(C=1.0, epsilon=0.2)
			clf.fit(obj.valid_data, obj.valid_attributes[:, k]) 
			train_data_semantic = np.append(train_data_semantic, [clf.predict(obj.train_data)], axis=0) 
			valid_data_semantic = np.append(valid_data_semantic, [clf.predict(obj.valid_data)], axis=0) 
			test_data_semantic = np.append(test_data_semantic, [clf.predict(obj.test_data)], axis=0) 
		
		end = time.time()
		print "Regressor training time %f"%((end-st)/60.0)
		#pdb.set_trace()	
	else:
		# fix random seed for reproducibility
		seed = 7
		np.random.seed(seed)
		model = baseline_model_regression(obj.train_data.shape[1], obj.train_attributes.shape[1])
		model.fit(obj.train_data, obj.train_attributes, validation_split = 0.3, validation_data = (obj.valid_data, obj.valid_attributes), verbose=1, shuffle=True, epochs = EPOCHS, callbacks = callbacks_list)
		train_data_semantic = function_normalise_data(model.predict(obj.train_data))		
		valid_data_semantic = function_normalise_data(model.predict(obj.valid_data))		
		test_data_semantic = function_normalise_data(model.predict(obj.test_data))		

	return train_data_semantic, valid_data_semantic, test_data_semantic

'''
-----------------------------------------------------------------
       Prepare data for f: semantic --> semantic (seen to unseen)
-----------------------------------------------------------------
'''

def get_data_for_seen_to_unseen_regressor(features_train, features_valid, obj, seen_class, unseen_class):
	train_indices_seen_class_samples = np.flatnonzero(obj.train_labels == seen_class)
	valid_indices_seen_class_samples = np.flatnonzero(obj.valid_labels == seen_class)
	
	ip_train = features_train[train_indices_seen_class_samples, :]
	ip_valid = features_valid[valid_indices_seen_class_samples, :]
	#NOTE: test_attributes_cross CAN NOT be learned/obtained during training.
	op_train = obj.prototypes[unseen_class - 1, :]
	op_valid = np.tile(obj.prototypes[unseen_class - 1, :], (ip_valid.shape[0], 1))
	
	NOISE_FACTOR = 0.05	
	INCREASE_FACTOR = int(1000 / ip_train.shape[0])
	print "Increase factor for seen (class %d) to unseen (class %d) regressor  is %d"%(seen_class, unseen_class, INCREASE_FACTOR)
	regressor_input = np.tile(ip_train, (INCREASE_FACTOR, 1))
	regressor_input = regressor_input + NOISE_FACTOR * np.random.normal(0, 1, regressor_input.shape)
	regressor_input = function_normalise_data(regressor_input)
	regressor_output = np.tile(op_train, (regressor_input.shape[0], 1))
	return regressor_input, regressor_output, ip_valid, op_valid


'''
-----------------------------------------------------------------
                        Regression f (NEW) : semantic(s) --> semantic(u)
-----------------------------------------------------------------
'''

def train_seen_to_unseen_regressor_NEW(features_train, features_valid, features_test, obj, seen_class, unseen_class):
	# fix random seed for reproducibility
	
	seed = 7
	np.random.seed(seed)
	regressor_input, regressor_output, valid_input, valid_output = get_data_for_seen_to_unseen_regressor(features_train, features_valid, obj, seen_class, unseen_class)
	model = baseline_model_regression_seen_to_unseen(regressor_input.shape[1], regressor_output.shape[1])
	model.fit(regressor_input, regressor_output, validation_split = 0.3, validation_data = (valid_input, valid_output), verbose=1, shuffle=True, epochs = EPOCHS, batch_size = 32, callbacks = callbacks_list)
	train_attributes_cross_mapped = model.predict(features_train)		
	valid_attributes_cross_mapped = model.predict(features_valid)		
	test_attributes_cross_mapped = model.predict(features_test)		
	
	train_attributes_cross_mapped = function_normalise_data(train_attributes_cross_mapped)
	valid_attributes_cross_mapped = function_normalise_data(valid_attributes_cross_mapped)
	test_attributes_cross_mapped = function_normalise_data(test_attributes_cross_mapped)

	return train_attributes_cross_mapped, valid_attributes_cross_mapped, test_attributes_cross_mapped
'''
-----------------------------------------------------------------
                        Regression f: semantic(s) --> semantic(u)
-----------------------------------------------------------------
'''

def train_seen_to_unseen_regressor(features_train, features_valid, obj, seen_class, unseen_class, k):
	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)
	regressor_input, regressor_output, valid_input, valid_output = get_data_for_seen_to_unseen_regressor(features_train, features_valid, obj, seen_class, unseen_class)
	model = baseline_model_regression(regressor_input.shape[1], regressor_output.shape[1])
	model.fit(regressor_input, regressor_output, validation_split = 0.3, validation_data = (valid_input, valid_output), verbose=1, shuffle=True, epochs = EPOCHS, callbacks = callbacks_list)
	train_attributes_cross_mapped = model.predict(features_train)		
	valid_attributes_cross_mapped = model.predict(features_valid)		
	strt = k * obj.test_attributes.shape[0]
	end = strt + obj.test_attributes.shape[0]
	test_attributes_cross_mapped = model.predict(obj.test_attributes_cross[strt:end, :])		

	return train_attributes_cross_mapped, valid_attributes_cross_mapped, test_attributes_cross_mapped

'''
-----------------------------------------------------------------
              Prepare data for f: visual --> semantic
-----------------------------------------------------------------
'''
def get_data_for_visual_to_semantic_regressor(features_train, features_valid, obj, visual_class, semantic_class):
	
	train_indices_input_class_samples = np.flatnonzero(obj.train_labels == visual_class)
	valid_indices_input_class_samples = np.flatnonzero(obj.valid_labels == visual_class)
	
	ip_train = features_train[train_indices_input_class_samples, :]
	ip_valid = features_valid[valid_indices_input_class_samples, :]
	
	NOISE_FACTOR = 0.05
	INCREASE_FACTOR = int(1000 / ip_train.shape[0])
	print "Increase factor for visual (class %d) to semantic (class %d) regressor  is %d"%(visual_class, semantic_class, INCREASE_FACTOR)
	#NOTE: Permutation of data TO BE implemented
	
	regressor_input = np.tile(ip_train, (INCREASE_FACTOR, 1))
	regressor_input = regressor_input + NOISE_FACTOR * np.random.normal(0, 1, regressor_input.shape)
	regressor_input = function_normalise_data(regressor_input)
	regressor_output = np.tile(obj.prototypes[semantic_class - 1, :], (regressor_input.shape[0], 1))
	op_valid = np.tile(obj.prototypes[semantic_class - 1, :], (ip_valid.shape[0], 1))
	valid_input = function_normalise_data(ip_valid)
	valid_output = function_normalise_data(op_valid)
	
	return regressor_input, regressor_output, valid_input, valid_output

'''
-----------------------------------------------------------------
                        Regression f (cross): visual --> semantic
-----------------------------------------------------------------
'''

def train_visual_to_semantic_cross_regressor(features_train, features_valid, features_test, obj, visual_class, semantic_class):
	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)
	regressor_input, regressor_output, valid_input, valid_output = get_data_for_visual_to_semantic_regressor(features_train, features_valid, obj, visual_class, semantic_class)
	model = baseline_model_regression(regressor_input.shape[1], regressor_output.shape[1])
	model.fit(regressor_input, regressor_output, validation_split = 0.3, validation_data = (valid_input, valid_output), verbose=1, shuffle=True, epochs = EPOCHS, callbacks = callbacks_list)
	train_attributes_cross = model.predict(features_train)		
	valid_attributes_cross = model.predict(features_valid)		
	test_attributes_cross = model.predict(features_test)		
	train_attributes_cross = function_normalise_data(train_attributes_cross)
	valid_attributes_cross = function_normalise_data(valid_attributes_cross)
	test_attributes_cross = function_normalise_data(test_attributes_cross)
	return train_attributes_cross, valid_attributes_cross, test_attributes_cross


'''
-----------------------------------------------------------------
                        Get cluster centers
-----------------------------------------------------------------
'''
def get_cluster_centers(features_train, features_valid, features_test, obj, classI, classJ):
        #NOTE: classI or classJ to be reconsidered, in case of cross-features!
	train_indices = np.flatnonzero(obj.train_labels == classI)
        valid_indices = np.flatnonzero(obj.valid_labels == classI)
	
	classIJ_features_train = features_train[train_indices, :]
	classIJ_features_valid = features_valid[valid_indices, :]
	train_mean = np.mean(classIJ_features_train, axis=0)
	valid_mean = np.mean(classIJ_features_valid, axis=0)
	#test_mean = np.mean(classIJ_features_test, axis=0)
	train_mean = np.reshape(train_mean, (1, train_mean.shape[0]))
	valid_mean = np.reshape(valid_mean, (1, valid_mean.shape[0]))
	return train_mean, valid_mean

'''
-----------------------------------------------------------------
                        Combine cross attributes
-----------------------------------------------------------------
'''
def combine_specific_nonspecific_attributes(obj, cluster_labels, cluster_labels_cross, base_filename):
	weights_mat = []
	for classI in cluster_labels:
		weights = []
		for classJ in cluster_labels:
			if classI != classJ:
				ind_cross = cluster_labels_cross.index([classI, classJ])
				ind = cluster_labels.index(classI)
				weight = euclidean_distances(obj.ae_features_train_means[ind, :], obj.ce_features_train_means[ind_cross, :])
				weights.append(weight[0][0])
		weights_mat.append(weights)
		
	k = 0
	classI_weights = 0
	end = 0
	for classI in cluster_labels:
		weights = []
		classI_indices_train = np.flatnonzero(obj.train_labels == classI)			
		classI_indices_valid = np.flatnonzero(obj.valid_labels == classI)			
		#NOTE: For test classes, code MUST be changed.
		#All test samples are considered.
		classI_indices_test = np.arange(0, obj.test_labels.shape[0])			
		norm_factor = np.sum(np.exp(weights_mat[classI_weights]))
		weights = weights_mat[classI_weights]/norm_factor
		tr = np.zeros((np.size(classI_indices_train), obj.prototypes.shape[1]), float)
		vl = np.zeros((np.size(classI_indices_valid), obj.prototypes.shape[1]), float)
		ts = np.zeros((np.size(classI_indices_test), obj.prototypes.shape[1]), float)
		for classJ in cluster_labels:
			if classI != classJ:
				path = base_filename + 'train_attributes_cross_domain_class_' + str(classJ) + '_' + str(classI)
				tmp_train = scipy.io.loadmat(path)
				cross_attri_train_all = tmp_train['train_attributes_cross']
				cross_attri_train = cross_attri_train_all[classI_indices_train,:]
				path = base_filename + 'valid_attributes_cross_domain_class_' + str(classJ) + '_' + str(classI)
				tmp_valid = scipy.io.loadmat(path)
				cross_attri_valid_all = tmp_valid['valid_attributes_cross']
				cross_attri_valid = cross_attri_valid_all[classI_indices_valid,:]
				path = base_filename + 'test_attributes_cross_domain_class_' + str(classJ) + '_' + str(classI)
				tmp_test = scipy.io.loadmat(path)
				cross_attri_test_all = tmp_test['test_attributes_cross']
				cross_attri_test = cross_attri_test_all[classI_indices_test,:]
				tr = tr + cross_attri_train*weights[k]
				vl = vl + cross_attri_valid*weights[k]
				ts = ts + cross_attri_test*weights[k]
				k = k + 1
		obj.train_attributes_cross[classI_indices_train, :] = tr
		obj.valid_attributes_cross[classI_indices_valid, :] = vl
		strt = end
		end = strt + classI_indices_test.shape[0]
		obj.test_attributes_cross[strt:end, :] = ts
		k = 0
		classI_weights = classI_weights + 1

	#return train_attributes_sp, valid_attributes_sp, test_attributes_sp

def function_scatter_plot(data, labels, classI, classJ):
	io_names =  ['Input', 'Output']#, 'Decoded']
	num_data_points = input_data.shape[0]
	#Y = TSNE(n_components=2).fit_transform(ip_data_decoded_data)
	#ip_data_decoded_data = PCA(n_components=2).fit_transform(ip_data_decoded_data)
        #Y = tsne(ip_data_decoded_data, 2, 50, 20.0);
	Y = PCA(n_components=2).fit_transform(data)
	print Y.shape
	io_colors = ['red', 'blue']# 'green']#'magenta', 'cyan']
	io_clusters = np.array([200*classI, 200*classJ])#-200*classJ])
	marker_size = 70
	for data_name, color, cluster in zip(io_names, io_colors, io_clusters):
		indices = np.flatnonzero(labels == cluster)
		plt.scatter(Y[indices, 0], Y[indices, 1], label = data_name + ' (class ' + str(abs(cluster)/200) + ')', s = marker_size, c=color)
		
	title_string = dataset_name.upper() + ': Cross-Encoder: class ' + str(classI) + ' to class ' + str(classJ)
	plt.title(title_string)	
	plt.legend(loc='best', fontsize=10)
	plt.xticks([])
	plt.yticks([])
	#figurename = data_save_path + 'SCATTER_PCA_class_' + str(classI) + '_class_' + str(classJ)	
	#plt.savefig(figurename + '.eps', format='eps',dpi=1000, bbox_inches='tight')
	#plt.savefig(figurename + '.png', bbox_inches = 'tight')
	plt.show()
	pdb.set_trace()
	plt.close("all")
	return

def visualize_data(data, labels, classes, dataset_class_names, string_title, figure_title, data_save_path):
	Y = PCA(n_components=2).fit_transform((data))
	print Y.shape
	marker_size = 5
	colors = cm.rainbow(np.linspace(0, 1, classes.shape[0]))
	plt.close("all")
	classes = classes - 1
	class_names = [dataset_class_names[i] for i in classes]
	classes = classes + 1
	for classI, color, name in zip(classes, colors, class_names):
		indices = np.flatnonzero(labels == classI)
		plt.scatter(Y[indices, 0], Y[indices, 1], label = name, s = marker_size, c=color)
		
	plt.title(string_title)	
	plt.legend(loc='best', fontsize=10)
	plt.xticks([])
	plt.yticks([])
	figurename = data_save_path + '/' + figure_title	
	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	#plt.show()
	plt.savefig(figurename + '.eps', format='eps',dpi=1000, bbox_inches='tight')
	plt.savefig(figurename + '.png', bbox_inches = 'tight')
	plt.close("all")
	#pdb.set_trace()
	
def visualize_attributes_and_proto(data, labels, classes, dataset_class_names, prototypes, string_title, figure_title, data_save_path):
	#data_and_proto = np.vstack((data, prototypes))
	#data_and_proto = scale(data_and_proto)
        pca = PCA(n_components=2)
	Y = pca.fit_transform(data)
	PCAed_proto = pca.transform(prototypes)
	Y = np.vstack((Y, PCAed_proto))
	print Y.shape
	marker_size = 5
	colors = cm.rainbow(np.linspace(0, 1, classes.shape[0]))
	plt.close("all")
	classes = classes - 1
	class_names = [dataset_class_names[i] for i in classes]
	classes = classes + 1
	for classI, color, name in zip(classes, colors, class_names):
		indices = np.flatnonzero(labels == classI)
		plt.scatter(Y[indices, 0], Y[indices, 1], label = name, s = marker_size, c=color)
		
	marker_size = 70
	for classI, color, name in zip(classes, colors, class_names):
		indices = data.shape[0] + (classI - 1)
		plt.scatter(Y[indices, 0], Y[indices, 1], label = name + ' PROTOTYPE', s = marker_size, c=color, marker='D')
	
	plt.title(string_title)	
	plt.legend(loc='best', fontsize=10)
	plt.xticks([])
	plt.yticks([])
	figurename = data_save_path + '/' + figure_title	
	plt.savefig(figurename + '.eps', format='eps',dpi=1000, bbox_inches='tight')
	plt.savefig(figurename + '.png', bbox_inches = 'tight')
	#plt.show()
	plt.close("all")
	#pdb.set_trace()

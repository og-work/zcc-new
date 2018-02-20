'''
-----------------------------------------------
-----------------------------------------------
		pt_train_cc_willow.py
		model 1
-----------------------------------------------
-----------------------------------------------
'''

'''
TODO List:
-add L2 regularizer loss for weights: done
-biases:done(by default)
-adding noise to sample:done
-load and store and resume training:done
-test/evaluation:done

-14 Feb 2018/Wednesday
-weight initialisation : done to check on actual weights
-decoder weights check/proper training of decoder
-concat from decoder and then one fc layer before classifier
-random shuffling of input data: done
-iterated training for coders

15 Feb 2018/ Thursday
-adaptive alpha: done
-average pooling of concat features to give k diemnsion only
-drop out in classifier

16 Feb 2018/ Friday
-adaptive pooling of coders itself by adding one layer to give weights to hidden feature of each coder
-check individual coders performance
-train N coders at a time rather only one
-stop classifier loss gradients for other coders /random learning
'''

import os
import pdb
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from pt_get_data_willow import classifier_data, input_cc, train_pt_cc_input, function_normalise_data
import torch.nn.functional as F
#from torchviz import make_dot, make_dot_from_trace
import time
import os 
import scipy.io

'''
--------------------------------------
Constants/Macro
--------------------------------------
'''
CUDA_TYPE = torch.cuda.FloatTensor
NUMBER_OF_EPOCHS = 500
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
ALPHA = 0.2
NOISE_FACTOR = 0.05
NUMBER_OF_CLASSES = 7
MIN_NUMBER_OF_SAMPLES_OF_CLASS = 70
TR_TS_VA_SPLIT = [0.8, 0.2]
NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET = int(TR_TS_VA_SPLIT[0] * MIN_NUMBER_OF_SAMPLES_OF_CLASS)
NUMBER_OF_PERMUTED_SAMPLES = NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET * NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET
CC_DATA_SIZE = 10000
INCREASE_FACTOR = int(CC_DATA_SIZE / NUMBER_OF_PERMUTED_SAMPLES)
TRAIN_BATCH_SIZE = 2000
#NUMBER_OF_BATCHES = int(NUMBER_OF_PERMUTED_SAMPLES / TRAIN_BATCH_SIZE)
NUMBER_OF_TRAIN_BATCHES = int(CC_DATA_SIZE / TRAIN_BATCH_SIZE)
NUMBER_OF_TEST_SAMPLES_ALL_CLASSES = 633
TEST_BATCH_SIZE = 633
NUMBER_OF_TEST_BATCHES = int(np.ceil(NUMBER_OF_TEST_SAMPLES_ALL_CLASSES / TEST_BATCH_SIZE))
DATA_SAVE_PATH = '../../data-willow/model1/'

MODEL_SAVE_PATH = DATA_SAVE_PATH + 'e2e_model_13.pth'
MODEL_LOAD_PATH = DATA_SAVE_PATH + 'e2e_model_13.pth'
LIST_TRAIN_OPTIONS = ['TRAIN_FROM_SCRATCH', 'RETRAIN']
TRAIN_OPTION = LIST_TRAIN_OPTIONS[0]

INITIAL_WEIGHT_PATH = '/home/SharedData/omkar/study/phd-research/codes/tf-codes/data-willow/initial_weights/'

'''
model  1: 70:30 train:valid and then test with test acc 70%
Model  3: full train and test
Model  4: 80:20 train:valid and then test with test acc X%
Model  5: 80:20 train:valid and then test with test & valid acc 60%, train 100%, alpha = 0.2
Model  6: 80:20 train:valid, weight init, zero bias, unit norm Xavier initialisation, alpha = 0.2, train 98% valid 58%
Model  7: 80:20 train:valid, weight init, zero bias, unit norm Xavier initialisation, shuffling train, alpha = 0.2, train 98% valid 58%
Model  8: 80:20 train:valid, weight init, zero bias, unit norm Xavier initialisation, shuffling train, alpha = adaptive, train 100% valid/test 62%
Model  8: 80:20 train:valid, weight init, zero bias, unit norm Xavier initialisation, shuffling train, alpha = adaptive, train 100% valid/test 62%
Model  9: Same as model 8, train 100% valid/test 62%
Model 10: Weight decay 1e-5, weight init from conf1_18 weights
Model 11: weight init from conf1_18 weights, SGD with LR 1e^-2, train 100, valid 65, test 65
Model 12: weight init from conf1_18 weights, SGD with LR 1e^-4
'''

def dump_weights(model, epoch, batch, input_class, output_class, string):
	#Encoder of AE[0] weights
	filename = DATA_SAVE_PATH + 'weight_dumped/ae_0_fc1_' + string + '_ip_' + str(input_class) + '_op_' + str(output_class) + '_epoch_' + str(epoch) + '_batch_' + str(batch)
	ae_0_fc1 = (model.ae_bank[0].fc1.weight.data).cpu().numpy()
	scipy.io.savemat(filename, dict(ae_0_fc1 = ae_0_fc1))

	#Decoder of AE[0] weights
	filename = DATA_SAVE_PATH + 'weight_dumped/ae_0_fc2_' + string + '_ip_' + str(input_class) + '_op_' + str(output_class)+ '_epoch_'+ str(epoch) + '_batch_' + str(batch)
	ae_0_fc2 = (model.ae_bank[0].fc2.weight.data).cpu().numpy()
	scipy.io.savemat(filename, dict(ae_0_fc2 = ae_0_fc2))

	#Encoder of AE[1] weights
	filename = DATA_SAVE_PATH + 'weight_dumped/ae_1_fc1_' + string + '_ip_' + str(input_class) + '_op_' + str(output_class) + '_epoch_'+ str(epoch) + '_batch_' + str(batch)
	ae_1_fc1 = (model.ae_bank[1].fc1.weight.data).cpu().numpy()
	scipy.io.savemat(filename, dict(ae_1_fc1 = ae_1_fc1))

	#Decoder of AE[1] weights
	filename = DATA_SAVE_PATH + 'weight_dumped/ae_1_fc2_' + string + '_ip_' + str(input_class) + '_op_' + str(output_class) + '_epoch_'+ str(epoch) + '_batch_' + str(batch)
	ae_1_fc2 = (model.ae_bank[1].fc2.weight.data).cpu().numpy()
	scipy.io.savemat(filename, dict(ae_1_fc2 = ae_1_fc2))

	#Encoder of AE[10] weights
	filename = DATA_SAVE_PATH + 'weight_dumped/ae_10_fc1_' + string + '_ip_' + str(input_class) + '_op_' + str(output_class) + '_epoch_' + str(epoch) + '_batch_' + str(batch)
        ae_10_fc1 = (model.ae_bank[10].fc1.weight.data).cpu().numpy()
        scipy.io.savemat(filename, dict(ae_10_fc1 = ae_10_fc1))

	#Decoder of AE[10] weights
        filename = DATA_SAVE_PATH + 'weight_dumped/ae_10_fc2_' + string + '_ip_' + str(input_class) + '_op_' + str(output_class)+ '_epoch_'+ str(epoch) + '_batch_' + str(batch)
        ae_10_fc2 = (model.ae_bank[10].fc2.weight.data).cpu().numpy()
        scipy.io.savemat(filename, dict(ae_10_fc2 = ae_10_fc2))
	
	#classifier weights
	filename = DATA_SAVE_PATH + 'weight_dumped/clafr_fc1_' + string + '_ip_' + str(input_class) + '_op_' + str(output_class) + '_epoch_'+ str(epoch) + '_batch_' + str(batch)
	clafr_fc1 = (model.mlp_classifier.fc1.weight.data).cpu().numpy()
	scipy.io.savemat(filename, dict(clafr_fc1 = clafr_fc1))

	filename = DATA_SAVE_PATH + 'weight_dumped/clafr_fc3_' + string + '_ip_' + str(input_class) + '_op_' + str(output_class) + '_epoch_'+ str(epoch) + '_batch_' + str(batch)
	clafr_fc3 = (model.mlp_classifier.fc3.weight.data).cpu().numpy()
	scipy.io.savemat(filename, dict(clafr_fc3 = clafr_fc3))




def write_sample_weights():
	filename = INITIAL_WEIGHT_PATH + 'willow_500_50_cl1_' + str(1) + '_cl2_' + str(1) + '_Wh'
	Wh = np.ones((500, 50), dtype=float) * 1.1
        scipy.io.savemat(filename, dict(Wh = Wh))
	
	filename = INITIAL_WEIGHT_PATH + 'willow_500_50_cl1_' + str(1) + '_cl2_' + str(1) + '_Wo'
	Wo = np.ones((50, 500), dtype=float) * 9.9
        scipy.io.savemat(filename, dict(Wo = Wo))

	filename = INITIAL_WEIGHT_PATH + 'willow_500_50_cl1_' + str(1) + '_cl2_' + str(1) + '_bh'
	bh = np.ones((50), dtype=float) * 2.2
        scipy.io.savemat(filename, dict(bh = bh))

	filename = INITIAL_WEIGHT_PATH + 'willow_500_50_cl1_' + str(1) + '_cl2_' + str(1) + '_bo'
	bo = np.ones((500), dtype=float) * 4.4
        scipy.io.savemat(filename, dict(bo = bo))
'''
-------------------------------------------------
Save results
-------------------------------------------------
'''
def save_performance_para(total_loss, mse_loss, clafr_loss, acc, split):
	filename = DATA_SAVE_PATH + 'total_loss' + split  
        scipy.io.savemat(filename, dict(total_loss = np.asarray(total_loss)))
	filename = DATA_SAVE_PATH + 'mse_loss' + split
        scipy.io.savemat(filename, dict(mse_loss = np.asarray(mse_loss)))
	filename = DATA_SAVE_PATH + 'clafr_loss' + split
        scipy.io.savemat(filename, dict(clafr_loss = np.asarray(clafr_loss)))


'''
---------------------------------------------------------
Utility functions for pytorch and numpy
---------------------------------------------------------
'''
def get_pytorch_variable(ip_ndarray):
	pt_tensor = Variable(torch.from_numpy(ip_ndarray))
        return pt_tensor.type(torch.FloatTensor)

def get_pytorch_long_variable(ip_ndarray):
	pt_tensor = Variable(torch.from_numpy(ip_ndarray))
        return pt_tensor.type(torch.LongTensor)
	
def get_ndarray(ip_pt_tensor):
        return ip_pt_tensor.data.numpy()

def get_validation_data(data, labels, test_classes):
	valid_indices = []
	for this_class in test_classes:
		indices_class_samples = np.flatnonzero(labels == this_class)
		valid_indices.append(indices_class_samples[NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET:])
	valid_indices = np.concatenate(valid_indices).ravel()
	input_batch_labels = labels[valid_indices]
	input_data = data[valid_indices, :]
	
	return input_data, input_batch_labels


'''
---------------------------------------------------------
Data loader for testing
---------------------------------------------------------
'''
def get_testing_data(batch, data, labels, test_classes):
	input_data = np.empty((0, data.shape[1]), float)
	batch_labels = []	
	m = 0
	test_indices = []
	for this_class in test_classes:
		indices_class_samples = np.flatnonzero(labels == this_class)
		test_indices.append(indices_class_samples)
	test_indices = np.concatenate(test_indices).ravel()
	input_start_ind = batch * TEST_BATCH_SIZE 
	input_end_ind = input_start_ind + TEST_BATCH_SIZE
	input_batch_indices = test_indices[input_start_ind:input_end_ind]
	input_batch_labels = labels[input_batch_indices]
	input_data = data[input_batch_indices, :]
	input_batch_labels = np.asarray(input_batch_labels)
	input_batch_labels = input_batch_labels.flatten()
	
	return input_data, input_batch_labels

'''
---------------------------------------------------------
Shuffle training data during each epoch
---------------------------------------------------------
'''
def shuffle_training_data(dataset_train_labels, train_class_labels):
	#***NOTE***: Only training samples to be shuffled and the validation samples.
	shuffled_indices_list = []
	for this_class in train_class_labels:
	        indices_this_class_samples = np.flatnonzero(dataset_train_labels == this_class)
	        indices_this_class_samples = indices_this_class_samples[:NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET]
		np.random.shuffle(indices_this_class_samples)
		shuffled_indices_list.append(indices_this_class_samples)
	return shuffled_indices_list

'''
---------------------------------------------------------
Data loader for validation data
---------------------------------------------------------
'''
def get_validation_data_class_specific(input_class, output_class, data, labels):
	indices_input_class_samples = np.flatnonzero(labels == input_class)
	indices_input_class_samples = indices_input_class_samples[NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET:]
	indices_output_class_samples = np.flatnonzero(labels == output_class)
	indices_output_class_samples = indices_output_class_samples[NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET:]
	
	indices_input_class_samples = np.repeat(indices_input_class_samples, np.size(indices_input_class_samples))
	indices_output_class_samples = np.tile(indices_output_class_samples, np.size(indices_output_class_samples))
	
	start_ind = 0 
	end_ind = np.size(indices_input_class_samples)

	indices_input_class_samples = indices_input_class_samples.flatten()
	indices_output_class_samples = indices_output_class_samples.flatten()
	
	input_batch_indices = indices_input_class_samples[start_ind:end_ind] 
	output_batch_indices = indices_output_class_samples[start_ind:end_ind] 
	
	input_data = data[input_batch_indices, :]
	output_data = data[output_batch_indices, :]

	batch_labels = []	
	batch_labels.append([input_class]*input_data.shape[0])
	batch_labels = np.asarray(batch_labels)
	batch_labels = batch_labels.flatten()

	return input_data, output_data, batch_labels
'''
---------------------------------------------------------
Data loader for training
---------------------------------------------------------
'''

def get_training_data_class_specific(input_class, output_class, batch, data, labels, train_classes, shuffled_indices_list):
	#indices_input_class_samples1 = np.flatnonzero(labels == input_class)
	indices_input_class_samples = np.asarray(shuffled_indices_list[input_class - 1])
	#indices_input_class_samples1 = indices_input_class_samples1[:NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET]
	#indices_input_class_samples = indices_input_class_samples[:NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET]
	#indices_output_class_samples = np.flatnonzero(labels == output_class)
	indices_output_class_samples = np.asarray(shuffled_indices_list[output_class - 1])
	#indices_output_class_samples = indices_output_class_samples[:NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET]
	
	indices_input_class_samples = np.repeat(indices_input_class_samples, np.size(indices_input_class_samples))
	indices_output_class_samples = np.tile(indices_output_class_samples, np.size(indices_output_class_samples))
	
	start_ind = batch * TRAIN_BATCH_SIZE 
	end_ind = start_ind + TRAIN_BATCH_SIZE

	#Data augmentation
	indices_input_class_samples = np.tile(indices_input_class_samples, (1, INCREASE_FACTOR))
	indices_output_class_samples = np.tile(indices_output_class_samples, (1, INCREASE_FACTOR))

	indices_input_class_samples = indices_input_class_samples.flatten()
	indices_output_class_samples = indices_output_class_samples.flatten()
	
	input_batch_indices = indices_input_class_samples[start_ind:end_ind] 
	output_batch_indices = indices_output_class_samples[start_ind:end_ind] 
	
	input_data = data[input_batch_indices, :]
	output_data = data[output_batch_indices, :]

	#Add noise to input and output
	input_data = input_data + NOISE_FACTOR * np.random.normal(0, 1, input_data.shape)
	output_data = output_data + NOISE_FACTOR * np.random.normal(0, 1, output_data.shape)
	
	input_data = function_normalise_data(input_data)
	output_data = function_normalise_data(output_data)

	batch_labels = []	
	batch_labels.append([input_class]*input_data.shape[0])
	batch_labels = np.asarray(batch_labels)
	batch_labels = batch_labels.flatten()

	return input_data, output_data, batch_labels

'''
---------------------------------------------------------
Save model using criteria
---------------------------------------------------------
'''
def save_model(model):
	#Use criterion to save the model	
	torch.save(model.state_dict(), MODEL_SAVE_PATH)

'''
---------------------------------------------------------
Work around for using lists in pytorch
---------------------------------------------------------
'''

class ListModule(object):
    #Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class testNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, steps=1):
        super(testNet, self).__init__()
        self.steps = steps

    def forward(self, input, hidden):
        for inp, i2h, h2h in zip(input, self.i2h, self.h2h):
            hidden = F.tanh(i2h(inp) + h2h(hidden))
        return hidden


'''
-------------------------------------------------
initiale weights using pretrained coders
-------------------------------------------------
'''

def get_initial_weights(input_class, output_class):
	#input_class = 1
	#output_class = 1 #TODO: to be removed later
 	filename = INITIAL_WEIGHT_PATH + 'willow_500_50_cl1_' + str(input_class) + '_cl2_' + str(output_class) + '_Wh'
	t1 = scipy.io.loadmat(filename)
	encoder_initial_weights = t1['Wh']	
	encoder_initial_weights = encoder_initial_weights.transpose()
 	filename = INITIAL_WEIGHT_PATH + 'willow_500_50_cl1_' + str(input_class) + '_cl2_' + str(output_class) + '_Wo'
	t2 = scipy.io.loadmat(filename)
	decoder_initial_weights = t2['Wo']	
	decoder_initial_weights = decoder_initial_weights.transpose()
	'''
 	filename = INITIAL_WEIGHT_PATH + 'willow_500_50_cl1_' + str(input_class) + '_cl2_' + str(output_class) + '_bh'
	t3 = scipy.io.loadmat(filename)
	encoder_initial_bias = t3['bh']	
 	filename = INITIAL_WEIGHT_PATH + 'willow_500_50_cl1_' + str(input_class) + '_cl2_' + str(output_class) + '_bo'
	t4 = scipy.io.loadmat(filename)
	decoder_initial_bias = t4['bo']
	'''
	return torch.from_numpy(encoder_initial_weights).float(), torch.from_numpy(decoder_initial_weights).float()
'''
---------------------------------------------
MLP Classifier
---------------------------------------------
'''
class mlp_classifier(nn.Module):
    def __init__(self, clafr_input_feature_dim, number_of_classes):
        super(mlp_classifier, self).__init__()
	self.clafr_input_feature_dim = clafr_input_feature_dim
	self.number_of_classes = number_of_classes
        if 0:
		self.classifier = nn.Sequential(
		    nn.Linear(self.clafr_input_feature_dim, int(self.clafr_input_feature_dim*0.5)),
		    nn.ReLU(True),
		    nn.Linear(int(self.clafr_input_feature_dim*0.5), int(self.clafr_input_feature_dim*0.25)),
		    nn.ReLU(True),
		    nn.Linear(int(self.clafr_input_feature_dim*0.25), int(self.clafr_input_feature_dim*0.1)),
		    nn.ReLU(True),
		    nn.Linear(int(self.clafr_input_feature_dim*0.1), self.clafr_number_of_classes))

	else:
		self.fc1 = nn.Linear(self.clafr_input_feature_dim, int(self.clafr_input_feature_dim*0.5))
		self.fc2 = nn.Linear(int(self.clafr_input_feature_dim*0.5), int(self.clafr_input_feature_dim*0.25))
		self.fc3 = nn.Linear(int(self.clafr_input_feature_dim*0.25), int(self.clafr_input_feature_dim*0.1))
		self.fc4 = nn.Linear(int(self.clafr_input_feature_dim*0.1), int(self.number_of_classes))
			
    def forward(self, x):
	if 0:
		predicted_labels = self.classifier(x)
		return predicted_labels
	else:
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)

		#return F.log_softmax(x, dim=1)
		return x

'''
---------------------------------------------
MLP Autoencoder
---------------------------------------------
'''

class autoencoder(nn.Module):
	def __init__(self, input_output_dim, hidden_layer_dim, input_class, output_class):
		super(autoencoder, self).__init__()
		self.input_output_dim = input_output_dim
		self.hidden_layer_dim = hidden_layer_dim

		if 0:
			self.encoder = nn.Sequential(
			nn.Linear(input_output_dim, hidden_layer_dim),
			nn.Tanh())
			self.decoder = nn.Sequential(
			nn.Linear(hidden_layer_dim, input_output_dim),
			nn.Tanh())
		else:
			encoder_initial_weights, decoder_initial_weights = get_initial_weights(input_class, output_class)
	                self.fc1 = nn.Linear(self.input_output_dim, self.hidden_layer_dim)
	                self.fc2 = nn.Linear(self.hidden_layer_dim, self.input_output_dim)
			self.fc1.weight = torch.nn.Parameter(encoder_initial_weights)
			self.fc2.weight = torch.nn.Parameter(decoder_initial_weights)
			#init.xavier_normal(self.fc1.weight, gain=1)
			#init.xavier_normal(self.fc2.weight, gain=1)
			#self.fc1.bias.data.fill_(0)
			#self.fc2.bias.data.fill_(0)
				
	def forward(self, x):
		if 0:
			encoded_features = self.encoder(x)
			decoded_features = self.decoder(encoded_features)	
		else:
			encoded_features = F.tanh(self.fc1(x))
			decoded_features = F.tanh(self.fc2(encoded_features))
		return encoded_features, decoded_features

'''
---------------------------------------------
E2E Network
---------------------------------------------
'''
class E2E_NETWORK_TRAIN(nn.Module):
	def __init__(self, input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels):
		super(E2E_NETWORK_TRAIN, self).__init__()
		self.input_output_dim = input_output_dim
		self.hidden_layer_dim = hidden_layer_dim
		self.number_of_classes = number_of_classes
		self.train_class_labels = train_class_labels
		#Concat
		self.clafr_input_feature_dim = hidden_layer_dim * number_of_classes * number_of_classes
		#Avg pool
		#self.clafr_input_feature_dim = hidden_layer_dim
		self.mlp_classifier = mlp_classifier(self.clafr_input_feature_dim, self.number_of_classes)
		self.mlp_classifier.cuda()
                self.decoded_features_this_cc = Variable(torch.zeros(TRAIN_BATCH_SIZE, self.input_output_dim), requires_grad=True)

		#Work around for using list modules
		self.ae_bank = ListModule(self, 'ae_bank_')
		for input_class in self.train_class_labels:
			for output_class in self.train_class_labels:
				    self.ae_bank.append(autoencoder(self.input_output_dim, self.hidden_layer_dim, input_class, output_class))

	def forward(self, data, input_class, output_class):
		p = 0
		concat_list_clfr = []
		#torch_avg_pooled_feature = Variable(torch.zeros(data.shape[0], self.hidden_layer_dim).type(CUDA_TYPE), requires_grad=True)
		#avg_pooled_feature = np.zeros((data.shape[0], self.hidden_layer_dim), dtype=float)
		
		#Passing input class data to all cc to form final feature vector
		for classI in self.train_class_labels:
			for classJ in self.train_class_labels:
				encoded_features, decoded_features = self.ae_bank[p](data)
				#torch_avg_pooled_feature += (encoded_features)
				concat_list_clfr.append(encoded_features)
				#print p, input_class, output_class, classI, classJ
				if (classI == input_class) & (classJ == output_class):
					self.decoded_features_this_cc = decoded_features
				p = p + 1
				
		#Concat hidden features along columns for different cc
		torch_encoded_features_conc = torch.cat(concat_list_clfr, dim = 1)
		predicted_labels = self.mlp_classifier(torch_encoded_features_conc.cuda())
		
		#Avg pooling
		#torch_avg_pooled_feature = torch_avg_pooled_feature / p
		#torch_avg_pooled_feature = get_pytorch_variable(avg_pooled_feature) 
		#predicted_labels = self.mlp_classifier((torch_avg_pooled_feature))
		return predicted_labels, self.decoded_features_this_cc

class E2E_NETWORK_TEST(nn.Module):
	def __init__(self, input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels):
		super(E2E_NETWORK_TEST, self).__init__()
		self.input_output_dim = input_output_dim
		self.hidden_layer_dim = hidden_layer_dim
		self.number_of_classes = number_of_classes
		self.train_class_labels = train_class_labels
		self.input_feature_dim = hidden_layer_dim * number_of_classes * number_of_classes
		self.mlp_classifier = mlp_classifier(self.input_feature_dim, self.number_of_classes)
		self.mlp_classifier.cuda()
		
		#Work around for using list modules
		self.ae_bank = ListModule(self, 'ae_bank_')
		k = 0
		for input_class in self.train_class_labels:
			for output_class in self.train_class_labels:
				self.ae_bank.append(autoencoder(self.input_output_dim, self.hidden_layer_dim, input_class, output_class))

	def forward(self, data):
		p = 0
		concat_list_clfr = []
		#avg_pooled_feature = Variable(torch.zeros(data.shape[0], self.hidden_layer_dim), requires_grad=True)
		#avg_pooled_feature = avg_pooled_features.cuda()
		#Passing input class data to all cc to form final feature vector
		for classI in self.train_class_labels:
			for classJ in self.train_class_labels:
				clafr_encoded_features, decoded_features = self.ae_bank[p](data)
				#avg_pooled_feature += encoded_features
				p = p + 1
				concat_list_clfr.append(clafr_encoded_features)
				
		#Concat hidden features along columsn for different cc
		torch_encoded_features_conc = torch.cat(concat_list_clfr, dim = 1)
		predicted_labels = self.mlp_classifier(torch_encoded_features_conc.cuda())
		#Avg pooling 
		#predicted_labels = self.mlp_classifier((avg_pooled_feature/p).cuda())
		return predicted_labels

'''
---------------------------------------------
Loss functions
---------------------------------------------
'''
mse_loss_function = nn.MSELoss(size_average=True)
cross_entropy_loss_function = nn.CrossEntropyLoss()# the target label is not one-hotted

def get_classifier_loss(predicted_labels, gt_labels):
	return cross_entropy_loss_function(predicted_labels, gt_labels)
	#return F.nll_loss(predicted_labels, gt_labels)

def get_mse_loss(decoded_data, output_data):
	mse_loss = mse_loss_function(decoded_data, output_data)
	return mse_loss

'''
---------------------------------------------
train the model
---------------------------------------------
'''
def train(model, optimizer, obj_input, batch, epoch, shuffled_indices_list, is_validation, alpha):
	model.cuda()
	mse_loss_list = []
	running_mse_loss = 0
	running_clafr_loss = 0
	running_total_loss = 0
	running_acc = 0
	k = 0
	#print "#####################################################################"
	#print epoch, batch, 0, 0
	#print model.ae_bank[0].fc1.weight.data
	#print model.ae_bank[1].fc1.weight.data
	#print model.ae_bank[10].fc1.weight.data
	#print model.ae_bank[47].fc1.weight.data
	#print model.ae_bank[48].fc1.weight.data
	for input_class in obj_input.train_class_labels:
		for output_class in obj_input.train_class_labels:
			k = k + 1
			#print "Batch %d, class pair: (%d, %d)"%(batch, input_class, output_class)
			if not is_validation:
				model.train()
				input_data, output_data, gt_labels = get_training_data_class_specific(input_class, output_class, batch, obj_input.visual_features, obj_input.dataset_train_labels, obj_input.train_class_labels, shuffled_indices_list)			
			else:
				model.eval()
				input_data, output_data, gt_labels = get_validation_data_class_specific(input_class, output_class, obj_input.visual_features, obj_input.dataset_train_labels)			
	
			# ===================forward=====================
			input_data = get_pytorch_variable(input_data)
			output_data = get_pytorch_variable(output_data)
			predicted_labels, decoded_feature_conc = model(input_data.cuda(), input_class, output_class)
			#decoded_feature_conc = get_pytorch_variable(decoded_feature_conc)	

			# ===================loss========================
			mse_loss = get_mse_loss(decoded_feature_conc.cuda(), output_data.cuda())
			mse_loss_list.append((mse_loss.data.cpu()).numpy())
			gt_labels = gt_labels - 1
			classifier_loss = get_classifier_loss(predicted_labels, (get_pytorch_long_variable(gt_labels)).cuda())

			# ===================backward====================
			if not is_validation:
				total_loss = mse_loss + alpha * classifier_loss
				optimizer.zero_grad()
				classifier_loss.backward(retain_graph=True)
				mse_loss.backward()
				optimizer.step()
				#print "#####################################################################"
				#print epoch, batch, input_class, output_class
				#print model.ae_bank[0].fc1.weight.data
				#print model.ae_bank[1].fc1.weight.data
				#print model.ae_bank[10].fc1.weight.data
				#pdb.set_trace()
				#print model.ae_bank[47].fc1.weight.data
				#print model.ae_bank[48].fc1.weight.data
			
			# ===================logs=========================
			_, predictions = torch.max(predicted_labels.data, 1)
			corrects = torch.sum(predictions == ((torch.from_numpy(gt_labels)).cuda()))	
			total_samples = predictions.shape[0]
			acc = corrects*100.0/total_samples
			running_acc += acc
			running_total_loss += mse_loss.data[0] + alpha * classifier_loss.data[0]
			running_mse_loss += mse_loss.data[0]
			running_clafr_loss += classifier_loss.data[0]
			
	save_model(model)
	mse_loss_list = np.asarray(mse_loss_list)
	mse_loss_list = mse_loss_list.flatten()
	return running_acc/k, running_total_loss/k, running_mse_loss/k, running_clafr_loss/k, mse_loss_list


'''
---------------------------------------------
Test the model
---------------------------------------------
'''
def test(obj_input, batch):
	model = E2E_NETWORK_TEST(obj_input.visual_features.shape[1],  obj_input.dimension_hidden_layer, obj_input.number_of_classes, obj_input.train_class_labels)
	model.load_state_dict(torch.load(MODEL_SAVE_PATH))
	model.eval()
	model.cuda()
	
	input_data, gt_labels = get_testing_data(batch, obj_input.visual_features, obj_input.dataset_test_labels, obj_input.train_class_labels)
	print "Number of test samples: %d"%gt_labels.shape[0]

	# ===================forward=====================
	input_data = get_pytorch_variable(input_data)
	predicted_labels = model(input_data.cuda())
	gt_labels = gt_labels - 1

	# ===================loss========================
	_, predictions = torch.max(predicted_labels.data, 1)
	corrects = torch.sum(predictions.type(torch.FloatTensor) == (torch.from_numpy(gt_labels).type(torch.FloatTensor)))	
	total_samples = predictions.shape[0]
	test_acc = corrects*100.0/total_samples
	return test_acc

def get_model_for_training(TRAIN_OPTION, input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels):
	if TRAIN_OPTION == 'TRAIN_FROM_SCRATCH':
		model = E2E_NETWORK_TRAIN(input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels)
	elif TRAIN_OPTION == 'RETRAIN':
		model = E2E_NETWORK_TRAIN(input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels)
		model.load_state_dict(torch.load(MODEL_LOAD_PATH))
	#for name, param in e2e_model_train.named_parameters():
	#	if param.requires_grad:
	#       	print param.size() #param.data	
	#print model
	return model


'''
---------------------------------------------
main function for train/test the network
---------------------------------------------
'''
def train_pytorch_cc(obj_input):
	input_output_dim = obj_input.visual_features.shape[1]
	hidden_layer_dim = obj_input.dimension_hidden_layer
	number_of_classes = obj_input.number_of_classes
	train_class_labels = obj_input.train_class_labels
	#write_sample_weights()
	if 1:
		'''
			-------------------Training-------------------
		'''
		e2e_model_train = get_model_for_training(TRAIN_OPTION, input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels)
		optimizer = torch.optim.Adam(e2e_model_train.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
		#optimizer = torch.optim.SGD(e2e_model_train.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
		start_train = time.time()
		mse_losses_for_coders_train = np.zeros(NUMBER_OF_CLASSES*NUMBER_OF_CLASSES, )
		mse_losses_for_coders_valid = np.zeros(NUMBER_OF_CLASSES*NUMBER_OF_CLASSES, )
		alpha = ALPHA
		
		total_loss_epoch_train = []
		mse_loss_epoch_train = []
		clafr_loss_epoch_train = []
		acc_epoch_train = []
		total_loss_epoch_valid = []
		mse_loss_epoch_valid = []
		clafr_loss_epoch_valid = []
		acc_epoch_valid = []

		for epoch in range(NUMBER_OF_EPOCHS):
			running_acc_batch_train = 0
			running_total_loss_train = 0
 			running_clafr_loss_train = 0
			running_mse_loss_train = 0
 			running_clafr_loss_valid = 0
			running_acc_batch_valid = 0
			running_total_loss_valid = 0
			running_mse_loss_valid = 0
			shuffled_train_indices_list = shuffle_training_data(obj_input.dataset_train_labels, train_class_labels)
			for batch in range(NUMBER_OF_TRAIN_BATCHES):
				#dump_weights(e2e_model_train, epoch, batch, 99, 99, 'before_mse_backprop_cl')
				IS_VALIDATION = 0
				acc_batch_train, total_loss_batch_train, mse_loss_batch_train, clafr_loss_batch_train, mse_loss_array_train = train(e2e_model_train, optimizer, obj_input, batch, epoch, shuffled_train_indices_list, IS_VALIDATION, alpha)

				IS_VALIDATION = 1#TODO: temporary
				acc_batch_valid, total_loss_batch_valid, mse_loss_batch_valid, clafr_loss_batch_valid, mse_loss_array_valid = train(e2e_model_train, optimizer, obj_input, batch, epoch, shuffled_train_indices_list, IS_VALIDATION, alpha)

				mse_losses_for_coders_train = np.vstack((mse_losses_for_coders_train, mse_loss_array_train))
				scipy.io.savemat(DATA_SAVE_PATH + 'mse_losses_for_coders_train', dict(mse_losses_for_coders_train = mse_losses_for_coders_train))
				mse_losses_for_coders_valid = np.vstack((mse_losses_for_coders_valid, mse_loss_array_valid))
				scipy.io.savemat(DATA_SAVE_PATH + 'mse_losses_for_coders_valid', dict(mse_losses_for_coders_valid = mse_losses_for_coders_valid))
				
				running_acc_batch_train += acc_batch_train
				running_mse_loss_train += mse_loss_batch_train
				running_total_loss_train += total_loss_batch_train	
				running_clafr_loss_train += clafr_loss_batch_train

				running_acc_batch_valid += acc_batch_valid
				running_mse_loss_valid += mse_loss_batch_valid
				running_total_loss_valid += total_loss_batch_valid	
				running_clafr_loss_valid += clafr_loss_batch_valid

			total_loss_epoch_train.append(running_total_loss_train / NUMBER_OF_TRAIN_BATCHES)
			mse_loss_epoch_train.append(running_mse_loss_train / NUMBER_OF_TRAIN_BATCHES)
			clafr_loss_epoch_train.append(running_clafr_loss_train / NUMBER_OF_TRAIN_BATCHES)
			acc_epoch_train.append(running_acc_batch_train / NUMBER_OF_TRAIN_BATCHES)

			total_loss_epoch_valid.append(running_total_loss_valid / NUMBER_OF_TRAIN_BATCHES)
			mse_loss_epoch_valid.append(running_mse_loss_valid / NUMBER_OF_TRAIN_BATCHES)
			clafr_loss_epoch_valid.append(running_clafr_loss_valid / NUMBER_OF_TRAIN_BATCHES)
			acc_epoch_valid.append(running_acc_batch_valid / NUMBER_OF_TRAIN_BATCHES)
			
			print('epoch %3d/%3d, Total Loss tr: %4.4f, MSE tr: :%4.4f, Clafr loss tr: %4.4f, Acc tr: %4.4f, Total Loss vl: %4.4f, MSE vl: :%4.4f, Clafr loss vl: %4.4f, Acc vl: %4.4f, alpha %4.4f'%(epoch + 1, NUMBER_OF_EPOCHS, total_loss_epoch_train[-1], mse_loss_epoch_train[-1], clafr_loss_epoch_train[-1], acc_epoch_train[-1], total_loss_epoch_valid[-1], mse_loss_epoch_valid[-1], clafr_loss_epoch_valid[-1], acc_epoch_valid[-1], alpha))
	
			alpha = mse_loss_epoch_train[-1] / clafr_loss_epoch_train[-1]
	
		save_performance_para(total_loss_epoch_train, mse_loss_epoch_train, clafr_loss_epoch_train, acc_epoch_train, 'train')
		save_performance_para(total_loss_epoch_valid, mse_loss_epoch_valid, clafr_loss_epoch_valid, acc_epoch_valid, 'valid')
		end_train = time.time()
		train_time = end_train - start_train
		print "Training time %d hr %d min"%((train_time / 3600), (train_time % 3600) / 60)
	if 1:
	
		'''
			-------------------Testing-------------------
		'''
		
		running_acc_batch = 0
		for batch in range(NUMBER_OF_TEST_BATCHES) :
			acc_batch = test(obj_input, batch)
			running_acc_batch += acc_batch
			running_acc_batch = running_acc_batch / (batch + 1.0)
			print('Batch [%4d/%4d], Running TEST Accuracy:{:%4.4f}'\
			%(batch + 1, NUMBER_OF_TEST_BATCHES, running_acc_batch))

	pdb.set_trace()

'''
-----------------------------------------------
-----------------------------------------------
		pt_train_cc_willow.py
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
-weight initialisation
-decoder weights check/proper training of decoder
-concat from decoder and then one fc layer before classifier
-random shuffling of input data
-iterated training for coders


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
from torchviz import make_dot, make_dot_from_trace
import time
import os 
import scipy.io

'''
--------------------------------------
Constants/Macro
--------------------------------------
'''

NUMBER_OF_EPOCHS = 100
learning_rate = 1e-5
alpha = 0.2
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
MODEL_SAVE_PATH = './e2e_model_8.pth'
MODEL_LOAD_PATH = './e2e_model_8.pth'
INITIAL_WEIGHT_PATH = '/home/SharedData/omkar/study/phd-research/codes/tf-codes/data-willow/initial_weights/'

'''
model 1: 70:30 train:valid and then test with test acc 70%
Model 3: full train and test
Model 4: 80:20 train:valid and then test with test acc X%
Model 5: 80:20 train:valid and then test with test & valid acc 60%, train 100%, alpha = 0.2
Model 6: 80:20 train:valid, weight init, zero bias, unit norm Xavier initialisation, alpha = 0.2, train 98% valid 58%
Model 7: 80:20 train:valid, weight init, zero bias, unit norm Xavier initialisation, shuffling train, alpha = 0.2, train 98% valid 58%
Model 8: 
'''
LIST_TRAIN_OPTIONS = ['TRAIN_FROM_SCRATCH', 'RETRAIN']
TRAIN_OPTION = LIST_TRAIN_OPTIONS[0]

def write_sample_weights():
	filename = INITIAL_WEIGHT_PATH + 'WILLOW_500_50_cl1_' + str(1) + '_cl2_' + str(1) + '_Wh'
	Wh = np.ones((500, 50), dtype=float) * 1.1
        scipy.io.savemat(filename, dict(Wh = Wh))
	
	filename = INITIAL_WEIGHT_PATH + 'WILLOW_500_50_cl1_' + str(1) + '_cl2_' + str(1) + '_Wo'
	Wo = np.ones((50, 500), dtype=float) * 9.9
        scipy.io.savemat(filename, dict(Wo = Wo))

	filename = INITIAL_WEIGHT_PATH + 'WILLOW_500_50_cl1_' + str(1) + '_cl2_' + str(1) + '_bh'
	bh = np.ones((50), dtype=float) * 2.2
        scipy.io.savemat(filename, dict(bh = bh))

	filename = INITIAL_WEIGHT_PATH + 'WILLOW_500_50_cl1_' + str(1) + '_cl2_' + str(1) + '_bo'
	bo = np.ones((500), dtype=float) * 4.4
        scipy.io.savemat(filename, dict(bo = bo))
'''
-------------------------------------------------
User defined functions
-------------------------------------------------

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

def shuffle_training_data(dataset_train_labels, train_class_labels):
	#***NOTE***: Only training samples to be shuffled and the validation samples.
	shuffled_indices_list = []
	for this_class in train_class_labels:
	        indices_this_class_samples = np.flatnonzero(dataset_train_labels == this_class)
	        indices_this_class_samples = indices_this_class_samples[:NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET]
		np.random.shuffle(indices_this_class_samples)
		shuffled_indices_list.append(indices_this_class_samples)
	return shuffled_indices_list

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

def get_training_data_class_specific(input_class, output_class, batch, data, labels, train_classes, shuffled_indices_list):
	#pdb.set_trace()
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

def save_model(model):
	#Use criterion to save the model	
	torch.save(model.state_dict(), MODEL_SAVE_PATH)


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

def get_initial_weights(input_class, output_class):
	input_class = 1
	output_class = 1 #TODO: to be removed later
 	filename = INITIAL_WEIGHT_PATH + 'WILLOW_500_50_cl1_' + str(input_class) + '_cl2_' + str(output_class) + '_Wh'
	t1 = scipy.io.loadmat(filename)
	encoder_initial_weights = t1['Wh']	
	encoder_initial_weights = encoder_initial_weights.transpose()
 	filename = INITIAL_WEIGHT_PATH + 'WILLOW_500_50_cl1_' + str(input_class) + '_cl2_' + str(output_class) + '_Wo'
	t2 = scipy.io.loadmat(filename)
	decoder_initial_weights = t2['Wo']	
	decoder_initial_weights = decoder_initial_weights.transpose()
	'''
 	filename = INITIAL_WEIGHT_PATH + 'WILLOW_500_50_cl1_' + str(input_class) + '_cl2_' + str(output_class) + '_bh'
	t3 = scipy.io.loadmat(filename)
	encoder_initial_bias = t3['bh']	
 	filename = INITIAL_WEIGHT_PATH + 'WILLOW_500_50_cl1_' + str(input_class) + '_cl2_' + str(output_class) + '_bo'
	t4 = scipy.io.loadmat(filename)
	decoder_initial_bias = t4['bo']
	'''
	return torch.from_numpy(encoder_initial_weights).float(), torch.from_numpy(decoder_initial_weights).float()
'''
---------------------------------------------
---------------------------------------------
MLP Classifier
---------------------------------------------
---------------------------------------------
'''
class mlp_classifier(nn.Module):
    def __init__(self, input_feature_dim, number_of_classes):
        super(mlp_classifier, self).__init__()
	self.input_feature_dim = input_feature_dim
	self.number_of_classes = number_of_classes
        if 0:
		self.classifier = nn.Sequential(
		    nn.Linear(self.input_feature_dim, int(self.input_feature_dim*0.5)),
		    nn.ReLU(True),
		    nn.Linear(int(self.input_feature_dim*0.5), int(self.input_feature_dim*0.25)),
		    nn.ReLU(True),
		    nn.Linear(int(self.input_feature_dim*0.25), int(self.input_feature_dim*0.1)),
		    nn.ReLU(True),
		    nn.Linear(int(self.input_feature_dim*0.1), self.number_of_classes))

	else:
		self.fc1 = nn.Linear(self.input_feature_dim, int(self.input_feature_dim*0.5))
		self.fc2 = nn.Linear(int(self.input_feature_dim*0.5), int(self.input_feature_dim*0.25))
		self.fc3 = nn.Linear(int(self.input_feature_dim*0.25), int(self.input_feature_dim*0.1))
		self.fc4 = nn.Linear(int(self.input_feature_dim*0.1), int(self.number_of_classes))
			
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
---------------------------------------------
MLP Autoencoder
---------------------------------------------
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
			#encoder_initial_weights, decoder_initial_weights = get_initial_weights(input_class, output_class)
	                self.fc1 = nn.Linear(self.input_output_dim, self.hidden_layer_dim)
	                self.fc2 = nn.Linear(self.hidden_layer_dim, self.input_output_dim)
			#pdb.set_trace()
			#self.fc1.weight = torch.nn.Parameter(encoder_initial_weights)
			#self.fc2.weight = torch.nn.Parameter(decoder_initial_weights)
			init.xavier_normal(self.fc1.weight, gain=1)
			init.xavier_normal(self.fc2.weight, gain=1)
			self.fc1.bias.data.fill_(0)
			self.fc2.bias.data.fill_(0)
				
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
---------------------------------------------
E2E Network
---------------------------------------------
---------------------------------------------
'''
class E2E_NETWORK_TRAIN(nn.Module):
	def __init__(self, input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels):
		super(E2E_NETWORK_TRAIN, self).__init__()
		self.input_output_dim = input_output_dim
		self.hidden_layer_dim = hidden_layer_dim
		self.number_of_classes = number_of_classes
		self.train_class_labels = train_class_labels
		self.input_feature_dim = hidden_layer_dim * number_of_classes * number_of_classes
		self.mlp_classifier = mlp_classifier(self.input_feature_dim, self.number_of_classes)
		self.mlp_classifier.cuda()

		#Work around for using list modules
		self.ae_bank = ListModule(self, 'ae_bank_')
		for input_class in self.train_class_labels:
			for output_class in self.train_class_labels:
				    self.ae_bank.append(autoencoder(self.input_output_dim, self.hidden_layer_dim, input_class, output_class))

	def forward(self, data, input_class, output_class):
		p = 0
		concat_list_clfr = []
		decoded_features_this_cc = Variable(torch.zeros(data.shape[0], data.shape[1]), requires_grad=True)

		#Passing input class data to all cc to form final feature vector
		for classI in self.train_class_labels:
			for classJ in self.train_class_labels:
				clfr_encoded_features, clfr_decoded_features = self.ae_bank[p](data)
				p = p + 1
				concat_list_clfr.append(clfr_encoded_features)
				if classI == input_class & classJ == output_class:
					decoded_features_this_cc = clfr_decoded_features
				
		#Concat hidden features along columns for different cc
		torch_encoded_features_conc = torch.cat(concat_list_clfr, dim = 1)
		predicted_labels = self.mlp_classifier(torch_encoded_features_conc.cuda())
		return predicted_labels, decoded_features_this_cc

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
		#Passing input class data to all cc to form final feature vector
		for classI in self.train_class_labels:
			for classJ in self.train_class_labels:
				clfr_encoded_features, clfr_decoded_features = self.ae_bank[p](data)
				p = p + 1
				concat_list_clfr.append(clfr_encoded_features)
				
		#Concat hidden features along columsn for different cc
		torch_encoded_features_conc = torch.cat(concat_list_clfr, dim = 1)
		predicted_labels = self.mlp_classifier(torch_encoded_features_conc.cuda())
		return predicted_labels

'''
---------------------------------------------
---------------------------------------------
Loss functions
---------------------------------------------
---------------------------------------------
'''
mse_loss_function = nn.MSELoss(size_average=True)
cross_entropy_loss_function = nn.CrossEntropyLoss()# the target label is not one-hotted

def get_classifier_loss(predicted_labels, gt_labels):
	return cross_entropy_loss_function(predicted_labels, gt_labels)
	#return F.nll_loss(predicted_labels, gt_labels)

def get_mse_loss(input_class, output_class, decoded_data, output_data, train_class_labels):
	mse_loss = mse_loss_function(decoded_data, output_data)
	return mse_loss

'''
---------------------------------------------
---------------------------------------------
train the model
---------------------------------------------
---------------------------------------------
'''
def train(model, optimizer, obj_input, batch, epoch, shuffled_indices_list, is_validation):
	model.cuda()
	mse_loss_list = []
	running_mse_loss = 0
	running_clafr_loss = 0
	running_total_loss = 0
	running_acc = 0
	k = 0
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

			# ===================loss=====================
			mse_loss = get_mse_loss(input_class, output_class, decoded_feature_conc.cuda(), output_data.cuda(), obj_input.train_class_labels)
			mse_loss_list.append((mse_loss.data.cpu()).numpy())
			gt_labels = gt_labels - 1
			classifier_loss = get_classifier_loss(predicted_labels, (get_pytorch_long_variable(gt_labels)).cuda())

			# ===================backward====================
			if not is_validation:
				total_loss = (1.0 - alpha) * mse_loss + alpha * classifier_loss
				optimizer.zero_grad()
				total_loss.backward()
				optimizer.step()
			
			# ===================logs===================
			_, predictions = torch.max(predicted_labels.data, 1)
			corrects = torch.sum(predictions == ((torch.from_numpy(gt_labels)).cuda()))	
			total_samples = predictions.shape[0]
			acc = corrects*100.0/total_samples
			running_acc += acc
			running_total_loss += (1.0 - alpha) * mse_loss.data[0] + alpha * classifier_loss.data[0]
			running_mse_loss += mse_loss.data[0]
			running_clafr_loss += clafr_loss.data[0]
			save_model(model)
			
	mse_loss_list = np.asarray(mse_loss_list)
	mse_loss_list = mse_loss_list.flatten()
	return running_acc/k, running_total_loss/k, running_mse_loss/k, running_clafr_loss/k, mse_loss_list


'''
---------------------------------------------
---------------------------------------------
Test the model
---------------------------------------------
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
		model.load_state_dict(torch.load(MODEL_SAVE_PATH))
	#for name, param in e2e_model_train.named_parameters():
	#	if param.requires_grad:
	#       	print param.size() #param.data	
	#print model
	return model


'''
---------------------------------------------
---------------------------------------------
main function for train/test the network
---------------------------------------------
---------------------------------------------
'''
def train_pytorch_cc(obj_input):
	input_output_dim = obj_input.visual_features.shape[1]
	hidden_layer_dim = obj_input.dimension_hidden_layer
	number_of_classes = obj_input.number_of_classes
	train_class_labels = obj_input.train_class_labels
	write_sample_weights()
	if 1:
		'''
			-------------------Training-------------------
		'''
		e2e_model_train = get_model_for_training(TRAIN_OPTION, input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels)
		optimizer = torch.optim.Adam(e2e_model_train.parameters(), lr=learning_rate, weight_decay=1e-5)
	
		start_train = time.time()
		mse_losses_for_coders_train = np.zeros(NUMBER_OF_CLASSES*NUMBER_OF_CLASSES, )
		mse_losses_for_coders_valid = np.zeros(NUMBER_OF_CLASSES*NUMBER_OF_CLASSES, )

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
				IS_VALIDATION = 0
				acc_batch_train, total_loss_batch_train, mse_loss_batch_train, clafr_loss_batch_train, mse_loss_array_train = train(e2e_model_train, optimizer, obj_input, batch, epoch, shuffled_train_indices_list, IS_VALIDATION, alpha)

				IS_VALIDATION = 1
				acc_batch_valid, total_loss_batch_valid, mse_loss_batch_valid, clafr_loss_batch_valid, mse_loss_array_valid = train(e2e_model_train, optimizer, obj_input, batch, epoch, shuffled_train_indices_list, IS_VALIDATION, alpha)

				mse_losses_for_coders_valid = np.vstack((mse_losses_for_coders_valid, mse_loss_array_valid))
				scipy.io.savemat('./mse_losses_for_coders_valid', dict(mse_losses_for_coders_valid = mse_losses_for_coders_valid))
				mse_losses_for_coders_train = np.vstack((mse_losses_for_coders_train, mse_loss_array_train))
				scipy.io.savemat('./mse_losses_for_coders_train', dict(mse_losses_for_coders_train = mse_losses_for_coders_train))
				
				running_acc_batch_train += acc_batch_train
				running_mse_loss_train += mse_loss_batch_train
				running_total_loss_train += total_loss_batch_train	
				running_clafr_loss_train += clafr_loss_batch_train

				running_acc_batch_valid += acc_batch_valid
				running_mse_loss_valid += mse_loss_batch_valid
				running_total_loss_valid += total_loss_batch_valid	
				running_clafr_loss_valid += clafr_loss_batch_valid

			total_loss_epoch_train = running_total_loss_train / NUMBER_OF_TRAIN_BATCHES
			mse_loss_epoch_train = running_mse_loss_train / NUMBER_OF_TRAIN_BATCHES
			clafr_loss_epoch_train = running_clafr_loss_train / NUMBER_OF_TRAIN_BATCHES
			acc_epoch_train = running_acc_batch_train / NUMBER_OF_TRAIN_BATCHES

			total_loss_epoch_valid = running_total_loss_valid / NUMBER_OF_TRAIN_BATCHES
			mse_loss_epoch_valid = running_mse_loss_valid / NUMBER_OF_TRAIN_BATCHES
			clafr_loss_epoch_valid = running_clafr_loss_valid / NUMBER_OF_TRAIN_BATCHES
			acc_epoch_valid = running_acc_batch_valid / NUMBER_OF_TRAIN_BATCHES

			print('epoch %3d/%3d, Total Loss tr: %4.4f, MSE tr: :%4.4f, Clafr loss tr: %4.4f, Acc tr: %4.4f, Total Loss vl: %4.4f, MSE vl: :%4.4f, Clafr loss vl: %4.4f, Acc vl: %4.4f,'%(epoch + 1, NUMBER_OF_EPOCHS, total_loss_epoch_train, mse_loss_epoch_train, clafr_loss_epoch_train, acc_epoch_train, total_loss_epoch_valid, mse_loss_epoch_valid, clafr_loss_epoch_valid, acc_epoch_valid))
	
		end_train = time.time()
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

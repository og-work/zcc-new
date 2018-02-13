'''
-----------------------------------------------
-----------------------------------------------
		pt_train_cc_willow.py
-----------------------------------------------
-----------------------------------------------
'''

'''
TODO List:
-add L2 regularizer loss for weights
-biases
-adding noise to sample
-load and store and resume training
-test/evaluation

'''

import os
import pdb
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from pt_get_data_willow import classifier_data, input_cc, train_pt_cc_input
import torch.nn.functional as F
from torchviz import make_dot, make_dot_from_trace


num_epochs = 1
learning_rate = 0.001
alpha = 0.5
NUMBER_OF_CLASSES = 7
MIN_NUMBER_OF_SAMPLES_OF_CLASS = 70
TR_TS_VA_SPLIT = [0.7, 0.3]
NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET = int(TR_TS_VA_SPLIT[0] * MIN_NUMBER_OF_SAMPLES_OF_CLASS)
NUMBER_OF_PERMUTED_SAMPLES = NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET * NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET
TRAIN_BATCH_SIZE = 128
NUMBER_OF_BATCHES = int(NUMBER_OF_PERMUTED_SAMPLES / TRAIN_BATCH_SIZE)
NUMBER_OF_TEST_SAMPLES_ALL_CLASSES = 210
TEST_BATCH_SIZE = 64
NUMBER_OF_TEST_BATCHES = int(NUMBER_OF_TEST_SAMPLES_ALL_CLASSES / TEST_BATCH_SIZE)
MODEL_SAVE_PATH = './e2e_model_1.pth'

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

def get_training_data_class_specific(input_class, output_class, batch, data, labels, train_classes):
	indices_input_class_samples = np.flatnonzero(labels == input_class)
	indices_input_class_samples = indices_input_class_samples[:NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET]
	indices_output_class_samples = np.flatnonzero(labels == output_class)
	indices_output_class_samples = indices_output_class_samples[:NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET]
	indices_input_class_samples = np.repeat(indices_input_class_samples, np.size(indices_input_class_samples))
	indices_output_class_samples = np.tile(indices_output_class_samples, np.size(indices_output_class_samples))
	
	start_ind = batch * TRAIN_BATCH_SIZE 
	end_ind = start_ind + TRAIN_BATCH_SIZE

	input_batch_indices = indices_input_class_samples[start_ind:end_ind] 
	output_batch_indices = indices_output_class_samples[start_ind:end_ind] 
	
	input_data = data[input_batch_indices, :]
	output_data = data[output_batch_indices, :]
	batch_labels = []	
	batch_labels.append([input_class]*input_data.shape[0])
	batch_labels = np.asarray(batch_labels)
	batch_labels = batch_labels.flatten()

	return input_data, output_data, batch_labels

def get_training_data(batch, data, labels, train_classes):
	input_data = np.empty((0, data.shape[1]), float)
	output_data = np.empty((0, data.shape[1]), float)
	batch_labels = []	
	m = 0
	indices_all_class_samples_train = []
	for this_class in train_classes:
		indices_class_samples = np.flatnonzero(labels == this_class)
	        indices_all_class_samples_train.append(indices_class_samples[:NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET])

	for input_class in train_classes:
		k = 0
		input_start_ind = batch * TRAIN_BATCH_SIZE 
		#input_start_ind = batch * TRAIN_BATCH_SIZE + m * NUMBER_OF_PERMUTED_SAMPLES	
		input_end_ind = input_start_ind + TRAIN_BATCH_SIZE
		input_class_indices = np.asarray(indices_all_class_samples_train[m])
		input_class_indices = np.repeat(input_class_indices, np.size(input_class_indices))
		m = m + 1
		for output_class in train_classes:
			#print "Preparing data for class pair (%d >> %d)"%(input_class, output_class)
			#output_start_ind = batch * TRAIN_BATCH_SIZE + k * NUMBER_OF_PERMUTED_SAMPLES
			output_start_ind = batch * TRAIN_BATCH_SIZE
			output_end_ind = output_start_ind + TRAIN_BATCH_SIZE
			output_class_indices = np.asarray(indices_all_class_samples_train[k])
			output_class_indices = np.tile(output_class_indices, np.size(output_class_indices))
			input_batch_indices = input_class_indices[input_start_ind:input_end_ind] 
			output_batch_indices = output_class_indices[output_start_ind:output_end_ind] 
			input_data = np.append(input_data, data[input_batch_indices, :], axis=0)
			output_data = np.append(output_data, data[output_batch_indices, :], axis=0)
			k = k + 1
			batch_labels.append([input_class]*TRAIN_BATCH_SIZE)
	batch_labels = np.asarray(batch_labels)
	batch_labels = batch_labels.flatten()

	if input_data.shape[0] != batch_labels.shape[0]:
	    raise AssertionError("Dimension mismatch!")	
	return input_data, output_data, batch_labels

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
	def __init__(self, input_output_dim, hidden_layer_dim):
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
	                self.fc1 = nn.Linear(self.input_output_dim, self.hidden_layer_dim)
	                self.fc2 = nn.Linear(self.hidden_layer_dim, self.input_output_dim)
	
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
		for i in range(self.number_of_classes*self.number_of_classes):
		    self.ae_bank.append(autoencoder(self.input_output_dim, self.hidden_layer_dim))

	def forward(self, data, input_class, output_class):
		p = 0
		concat_list_clfr = []
		decoded_features_this_cc = Variable(torch.randn(data.shape[0], data.shape[1]), requires_grad=False)

		#Passing input class data to all cc to form final feature vector
		for classI in self.train_class_labels:
			for classJ in self.train_class_labels:
				clfr_encoded_features, clfr_decoded_features = self.ae_bank[p](data)
				p = p + 1
				concat_list_clfr.append(clfr_encoded_features)
				if classI == input_class & classJ == output_class:
					decoded_features_this_cc = clfr_decoded_features
				
		#Concat hidden features along columsn for different cc
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
		for i in range(self.number_of_classes*self.number_of_classes):
		    self.ae_bank.append(autoencoder(self.input_output_dim, self.hidden_layer_dim))

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
def train(model, optimizer, obj_input, batch):
	model.train()
	model.cuda()
	running_corrects = 0
	running_total_samples = 0
	input_data, output_data, gt_labels = get_training_data(batch, obj_input.visual_features, obj_input.dataset_train_labels, obj_input.train_class_labels)			

	for input_class in obj_input.train_class_labels:
		for output_class in obj_input.train_class_labels:
			#print "Batch %d, class pair: (%d, %d)"%(batch, input_class, output_class)
			input_data, output_data, gt_labels = get_training_data_class_specific(input_class, output_class, batch, obj_input.visual_features, obj_input.dataset_train_labels, obj_input.train_class_labels)			
		
			# ===================forward=====================
			input_data = get_pytorch_variable(input_data)
			output_data = get_pytorch_variable(output_data)
			predicted_labels, decoded_feature_conc = model(input_data.cuda(), input_class, output_class)
			#decoded_feature_conc = get_pytorch_variable(decoded_feature_conc)	

			# ===================loss=====================
			mse_loss_train = get_mse_loss(input_class, output_class, decoded_feature_conc.cuda(), output_data.cuda(), obj_input.train_class_labels)
			gt_labels = gt_labels - 1
			classifier_loss_train = get_classifier_loss(predicted_labels, (get_pytorch_long_variable(gt_labels)).cuda())

			# ===================backward====================
			total_train_loss = mse_loss_train + alpha * classifier_loss_train
			optimizer.zero_grad()
			total_train_loss.backward()
			optimizer.step()
	
			#for name, param in model.named_parameters():
			#for param in model.ae_bank[0].parameters():
			#	if param.requires_grad:
			#        	print param.size() #param.data	
			#pdb.set_trace()	
			# ===================logs===================
			_, predictions = torch.max(predicted_labels.data, 1)
			running_corrects += torch.sum(predictions == ((torch.from_numpy(gt_labels)).cuda()))	
			running_total_samples += predictions.shape[0]
			train_acc = running_corrects*100.0/running_total_samples
			torch.save(model.state_dict(), MODEL_SAVE_PATH)
			total_loss = mse_loss_train.data[0] + classifier_loss_train.data[0]
			mse_loss = mse_loss_train.data[0]
			clafr_loss = classifier_loss_train.data[0]
	return train_acc, total_loss, mse_loss, clafr_loss


'''
---------------------------------------------
---------------------------------------------
Test the model
---------------------------------------------
---------------------------------------------
'''
def test(is_validation, optimizer, obj_input, batch):
	model = E2E_NETWORK_TEST(obj_input.visual_features.shape[1],  obj_input.dimension_hidden_layer, obj_input.number_of_classes, obj_input.train_class_labels)
	model.load_state_dict(torch.load(MODEL_SAVE_PATH))
	model.eval()
	model.cuda()
	
	if is_validation:
		input_data, gt_labels = get_validation_data(obj_input.visual_features, obj_input.dataset_train_labels, obj_input.train_class_labels)
	else:
		input_data, gt_labels = get_testing_data(batch, obj_input.visual_features, obj_input.dataset_test_labels, obj_input.train_class_labels)

	# ===================forward=====================
	input_data = get_pytorch_variable(input_data)
	predicted_labels = model(input_data.cuda())
	gt_labels = gt_labels - 1

	# ===================loss/logs===================
	_, predictions = torch.max(predicted_labels.data, 1)
	corrects = torch.sum(predictions.type(torch.FloatTensor) == (torch.from_numpy(gt_labels).type(torch.FloatTensor)))	
	total_samples = predictions.shape[0]
	test_acc = corrects*100.0/total_samples
	return test_acc
	

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

	'''
		----------------------------------------------
		-------------------Training-------------------
		----------------------------------------------
	
	'''
	e2e_model_train = E2E_NETWORK_TRAIN(input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels)
	print e2e_model_train
	#e2e_model_train.load_state_dict(torch.load(MODEL_SAVE_PATH))
	optimizer = torch.optim.Adam(e2e_model_train.parameters(), lr=learning_rate, weight_decay=1e-5)
	#x = Variable(torch.randn(64*7*7,500), requires_grad=True)
        #y1, y2 = e2e_model_train(x.cuda())
	#make_dot(, params=dict(e2e_model_train.named_parameters()))

	#pdb.set_trace()
	#for p in e2e_model_train.ae_bank[0].parameters():
	#	if p.requires_grad:
	#		print p.size()
	
	IS_VALIDATION = 1
	acc_valid_batch = 0
	for epoch in range(num_epochs):
		running_acc_batch = 0
		for batch in range(NUMBER_OF_BATCHES):
			acc_train_batch, total_loss_batch, mse_loss_batch, clafr_loss_batch = train(e2e_model_train, optimizer, obj_input, batch)
			running_acc_batch += acc_train_batch
			acc_train = running_acc_batch / (batch + 1.0)
			acc_valid_batch = test(IS_VALIDATION, optimizer, obj_input, batch)
			print('Batch [%4d/%4d], epoch [%3d/%3d], Total Loss: %4.4f, MSE loss: :%4.4f, Classifier loss: %4.4f, Train Acc: %4.4f, Valid Accuracy: %4.4f '%(batch, NUMBER_OF_BATCHES, epoch + 1, num_epochs, total_loss_batch, mse_loss_batch, clafr_loss_batch, acc_train, acc_valid_batch))

	'''
		----------------------------------------------
		-------------------Testing-------------------
		----------------------------------------------
	'''
	
	
	IS_VALIDATION = 0
	running_acc_batch = 0
	for batch in range(NUMBER_OF_TEST_BATCHES):
		acc_batch = test(IS_VALIDATION, optimizer, obj_input, batch)
		running_acc_batch += acc_batch
		running_acc_batch = running_acc_batch / (batch + 1.0)
		print('Batch [%4d/%4d], TEST Accuracy:{:%4.4f}'\
		%(batch, NUMBER_OF_BATCHES, running_acc_batch))

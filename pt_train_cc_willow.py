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
from pt_get_data_willow import classifier_data, input_cc

num_epochs = 50
batch_size = 64
learning_rate = 1e-3
alpha = 0.5
NUMBER_OF_CLASSES = 7
MIN_NUMBER_OF_SAMPLES_OF_CLASS = 70
TR_TS_VA_SPLIT = [0.7, 0.3]
NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET = int(TR_TS_VA_SPLIT[0] * MIN_NUMBER_OF_SAMPLES_OF_CLASS)
NUMBER_OF_PERMUTED_SAMPLES = NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET * NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET
BATCH_SIZE = 4
NUMBER_OF_BATCHES = int(NUMBER_OF_PERMUTED_SAMPLES / BATCH_SIZE)


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

def get_pytorch_variable(ip_ndarray):
	pt_tensor = Variable(torch.from_numpy(ip_ndarray))
        return pt_tensor.type(torch.FloatTensor)

def get_pytorch_long_variable(ip_ndarray):
	pt_tensor = Variable(torch.from_numpy(ip_ndarray))
        return pt_tensor.type(torch.LongTensor)
	
def get_ndarray(ip_pt_tensor):
        return ip_pt_tensor.data.numpy()

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

def precompute_training_indices(labels, train_classes):
	indices_input_class_samples_train = []
	indices_output_class_samples_train = []

	for this_class in train_classes:
		indices_class_samples = np.flatnonzero(labels == this_class)
	        indices_class_samples_train = indices_class_samples[:NUMBER_OF_TRAIN_SAMPLES_IN_SUBSET]
	        indices_input_class_samples_train.append(np.repeat(indices_class_samples_train, np.size(indices_class_samples_train)))
	        indices_output_class_samples_train.append(np.tile(indices_class_samples_train, np.size(indices_class_samples_train)))
	
	indices_input_class_samples_train = np.asarray(indices_input_class_samples_train)
	indices_input_class_samples_train = indices_input_class_samples_train.flatten()
	indices_output_class_samples_train = np.asarray(indices_output_class_samples_train)
	indices_output_class_samples_train = indices_output_class_samples_train.flatten()
	return indices_input_class_samples_train, indices_output_class_samples_train


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
		input_start_ind = batch * BATCH_SIZE 
		#input_start_ind = batch * BATCH_SIZE + m * NUMBER_OF_PERMUTED_SAMPLES	
		input_end_ind = input_start_ind + BATCH_SIZE
		input_class_indices = np.asarray(indices_all_class_samples_train[m])
		input_class_indices = np.repeat(input_class_indices, np.size(input_class_indices))
		m = m + 1
		for output_class in train_classes:
			#print "Preparing data for class pair (%d >> %d)"%(input_class, output_class)
			#output_start_ind = batch * BATCH_SIZE + k * NUMBER_OF_PERMUTED_SAMPLES
			output_start_ind = batch * BATCH_SIZE
			output_end_ind = output_start_ind + BATCH_SIZE
			output_class_indices = np.asarray(indices_all_class_samples_train[k])
			output_class_indices = np.tile(output_class_indices, np.size(output_class_indices))
			input_batch_indices = input_class_indices[input_start_ind:input_end_ind] 
			output_batch_indices = output_class_indices[output_start_ind:output_end_ind] 
			input_data = np.append(input_data, data[input_batch_indices, :], axis=0)
			output_data = np.append(output_data, data[output_batch_indices, :], axis=0)
			k = k + 1
			batch_labels.append([input_class]*BATCH_SIZE)
	batch_labels = np.asarray(batch_labels)
	batch_labels = batch_labels.flatten()
	
	return input_data, output_data, batch_labels
 
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
        self.classifier = nn.Sequential(
            nn.Linear(self.input_feature_dim, int(self.input_feature_dim*0.5)),
            nn.ReLU(True),
            nn.Linear(int(self.input_feature_dim*0.5), int(self.input_feature_dim*0.25)),
            nn.ReLU(True),
            nn.Linear(int(self.input_feature_dim*0.25), int(self.input_feature_dim*0.1)),
	    nn.ReLU(True),
            nn.Linear(int(self.input_feature_dim*0.1), self.number_of_classes))

    def forward(self, data):
        predicted_labels = self.classifier(data)
        return predicted_labels

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
		self.encoder = nn.Sequential(
		nn.Linear(input_output_dim, hidden_layer_dim),
		nn.Tanh())
		self.decoder = nn.Sequential(
		nn.Linear(hidden_layer_dim, input_output_dim),
		nn.Tanh())

	def forward(self, x):
	#	pdb.set_trace()
		encoded_features = self.encoder(x)
		decoded_features = self.decoder(encoded_features)
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
		#self.mlp_classifier.cuda()
		self.ae_bank = []
		for i in range(self.number_of_classes*self.number_of_classes):
			self.ae_bank.append(autoencoder(self.input_output_dim, self.hidden_layer_dim))

	def forward(self, data):
		encoded_feature_conc = np.empty((0, self.hidden_layer_dim), dtype=float)		
		decoded_feature_conc = np.empty((0, self.input_output_dim), dtype=float)
		#Passing input class data to corresponding cc
		k = 0		
		for input_class in self.train_class_labels:
			for output_class in self.train_class_labels:
				start_ind = k * BATCH_SIZE
				end_ind = start_ind + BATCH_SIZE
				input_data_cc = data[start_ind:end_ind, :]
				encoded_features, decoded_features = self.ae_bank[k](get_pytorch_variable(input_data_cc))
				#encoded_feature_conc = np.append(encoded_feature_conc, encoded_features.data.cpu().numpy(), axis=0)
				decoded_feature_conc = np.append(decoded_feature_conc, decoded_features.data.cpu().numpy(), axis=0)
				k = k + 1

		#Passing input class data to all cc to form final feature vector
		k = 0		
		clfr_features = np.empty((0, self.hidden_layer_dim*NUMBER_OF_CLASSES*NUMBER_OF_CLASSES), dtype=float)		
		for input_class in self.train_class_labels:
			for output_class in self.train_class_labels:
				start_ind = k * BATCH_SIZE
				end_ind = start_ind + BATCH_SIZE
				input_data_cc = data[start_ind:end_ind, :]
				p = 0
				clfr_encoded_feature_conc = np.empty((input_data_cc.shape[0], 0), dtype=float)		
				for classI in self.train_class_labels:
					for classJ in self.train_class_labels:
						clfr_encoded_features, clfr_decoded_features = self.ae_bank[p](get_pytorch_variable(input_data_cc))
						#Concat hidden features along columns for all cc
						clfr_encoded_feature_conc = np.append(clfr_encoded_feature_conc, clfr_encoded_features.data.cpu().numpy(), axis=1)
						p = p + 1
				k = k + 1
				#Concat hidden features along rows for different batches
				clfr_features = np.append(clfr_features, clfr_encoded_feature_conc, axis=0)
	
		predicted_labels = self.mlp_classifier(get_pytorch_variable(clfr_features))
		return predicted_labels, decoded_feature_conc

'''
---------------------------------------------
---------------------------------------------
Loss functions
---------------------------------------------
---------------------------------------------
'''
mse_loss_function = nn.MSELoss()
cross_entropy_loss_function = nn.CrossEntropyLoss()# the target label is not one-hotted

def get_classifier_loss(predicted_labels, gt_labels):
	return cross_entropy_loss_function(predicted_labels, gt_labels)

def get_mse_loss(decoded_data, output_data, train_class_labels):
	k = 0
	mse_loss = 0
	for input_class in train_class_labels:
		for output_class in train_class_labels:
			start_ind = k * BATCH_SIZE
			end_ind = start_ind + BATCH_SIZE				
			this_cc_output = output_data[start_ind:end_ind, :] 
			this_cc_decoded_output = decoded_data[start_ind:end_ind, :]
			k = k + 1
			this_cc_mse_loss = mse_loss_function(this_cc_output, this_cc_decoded_output)
			mse_loss = mse_loss + this_cc_mse_loss

	return mse_loss

def train_pytorch_cc(obj_input):
	input_output_dim = obj_input.visual_features.shape[1]
	hidden_layer_dim = obj_input.dimension_hidden_layer
	number_of_classes = obj_input.number_of_classes
	train_class_labels = obj_input.train_class_labels
	#input_indices, output_indices = precompute_training_indices(obj_input.dataset_train_labels, train_class_labels)

	e2e_model_train = E2E_NETWORK_TRAIN(input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels)
	#e2e_model_train.cuda()
	optimizer = torch.optim.Adam(e2e_model_train.parameters(), lr=learning_rate, weight_decay=1e-5)
	for epoch in range(num_epochs):
		for batch in range(NUMBER_OF_BATCHES):
			input_data, output_data, gt_labels = get_training_data(batch, obj_input.visual_features, obj_input.dataset_train_labels, train_class_labels)			
			# ===================forward AE=====================
			predicted_labels, decoded_feature_conc = e2e_model_train(input_data)			
			mse_loss_train = get_mse_loss(get_pytorch_variable(decoded_feature_conc), get_pytorch_variable(output_data), train_class_labels)
			predicted_labels = predicted_labels - 1
			gt_labels = gt_labels - 1
			classifier_loss_train = get_classifier_loss(predicted_labels, get_pytorch_long_variable(gt_labels))


			# ===================backward====================
			total_train_loss = mse_loss_train + alpha * classifier_loss_train
			optimizer.zero_grad()
			total_train_loss.backward()
			optimizer.step()
		
			# ===================log========================
			print('Batch [%4d/%4d], epoch [%3d/%3d], Total Loss {:%4f}, MSE loss:{:%4f}, Classifier loss:{:%4f}'
			  %(batch, NUMBER_OF_BATCHES, epoch + 1, num_epochs, (mse_loss_train.data[0] + classifier_loss_train.data[0]), mse_loss_train.data[0], classifier_loss_train.data[0]))
			torch.save(e2e_model_train.state_dict(), './e2e_model.pth')
	
	


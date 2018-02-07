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
from pt_get_data_willow import classifier_data

num_epochs = 10
batch_size = 64
learning_rate = 1e-4
alpha = 0.5
NUMBER_OF_CLASSES = 7

class train_pt_cc_input:
        cc1_input_train_perm = np.array([])
        cc1_output_train_perm = np.array([])
        cc1_input_valid_perm = np.array([])
        cc1_output_valid_perm = np.array([])
        obj_classifier_data = classifier_data()
        dimension_hidden_layer1 = []
        EPOCHS_CC = []
        classI = []
        classJ = []
        dataset_name = []
        data_save_path = []
        dim_feature = []
        def function(self):
                print("This is train pytorch cc input class")

def get_pytorch_variable(ip_ndarray):
	pt_tensor = Variable(torch.from_numpy(ip_ndarray))
        return pt_tensor.type(torch.cuda.FloatTensor)
	
def get_ndarray(ip_pt_tensor):
	pt_tensor = Variable(torch.from_numpy(ip_ndarray))
        return pt_tensor.type(torch.cuda.FloatTensor)

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

class mlp_classifier(nn.Module):
    def __init__(self, data, number_of_classes):
        #super(autoencoder, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(data.shape[1], int(data.shape[1]*0.5)),
            nn.ReLu(True),
            nn.Linear(data.shape[1]*0.5, int(data.shape[1]*0.25)),
            nn.ReLu(True),
            nn.Linear(data.shape[1]*0.25, int(data.shape[1]*0.1)),
	    nn.ReLu(True),
            nn.Linear(data.shape[1]*0.1, number_of_classes)))

    def forward(self, data):
        predicted_labels = self.classifier(data)
        return predicted_labels


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
        encoded_features = self.encoder(x)
        decoded_features = self.decoder(encoded_features)
        return encoded_features, decoded_features

class e2e_network(nn.Module)
	def __init__(self, input_output_dim, hidden_layer_dim, number_of_classes, train_class_labels):
		self.input_output_dim = input_output_dim
		self.hidden_layer_dim = hidden_layer_dim
		self.number_of_classes = number_of_classes
		self.mlp_classifier = mlp_classifier(self.data, self.number_of_classes)
		self.ae_bank = []
		for i in range(self.number_of_classes*self.number_of_classes):
			self.ae_bank.append(autoencoder(self.input_output_dim, self.hidden_layer_dim))

	def forward(self, data, labels):
		for classI in train_class_labels:
			for classJ in train_class_labels:
				data_cc = get_data_for_cc(classI, classJ, data, labels)
						
	
		

mse_loss_function = nn.MSELoss()
cross_entropy_loss_function = nn.CrossEntropyLoss()# the target label is not one-hotted

def train_pytorch_cc(obj_cc_input):
        obj_cc_output = train_pt_cc_output()
	input_output_dim = (obj_cc_input.cc1_input_train_perm).shape[1]
	hidden_layer_dim = obj_cc_input.dimension_hidden_layer1
	ae_model = autoencoder(input_output_dim, hidden_layer_dim).cuda()
	optimizer = torch.optim.Adam(ae_model.parameters(), lr=learning_rate, weight_decay=1e-5)
	feature_train_conc = np.empty((0, hidden_layer_dim), dtype=float)
	feature_valid_conc = np.empty((0, hidden_layer_dim), dtype=float)
	
	total_mse_loss_train = 0
	total_mse_loss_valid = 0

	for epoch in range(num_epochs):
		for classI in train_class_labels:
			obj_train_cc = get_train_data()
			
			# ===================forward AE=====================
			encoded_train, decoded_train = ae_model(get_pytorch_variable(obj_train_cc.cc1_input_train_perm))
			encoded_valid, decoded_valid = ae_model(get_pytorch_variable(obj_train_cc.cc1_input_valid_perm))
			mse_loss_train = mse_loss_function(decoded_train, get_pytorch_variable(obj_train_cc.cc1_output_train_perm))
			mse_loss_valid = mse_loss_function(decoded_valid, get_pytorch_variable(obj_train_cc.cc1_output_valid_perm))
			# ========== concatenate features ===============
			feature_train_conc = np.append(feature_train_conc, encoded_train.data.cpu().numpy(), axis=0)
			feature_valid_conc = np.append(feature_valid_conc, encoded_valid.data.cpu().numpy(), axis=0)
			
			total_mse_loss_train = total_mse_loss_train + mse_loss_train	
			total_mse_loss_valid = total_mse_loss_valid + mse_loss_valid
	
		# ===================forward classifier=====================
		predicted_labels = classifier_model(feature_conc_train)
		classifier_loss = cross_entropy_loss_function(predicted_labels, labels_train)

		# ===================backward====================
		total_loss = total_mse_loss_train + alpha * classifier_loss
		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()
		
		# ===================log========================
		print('epoch [%3d/%3d], train loss:{:%4f} validation loss:{:%4f}'
		  %(epoch + 1, num_epochs, mse_loss_train.data[0], mse_loss_valid.data[0]))
		torch.save(model.state_dict(), './sim_autoencoder.pth')
	
	encoded_data_train, decoded_data_train = ae_model(get_pytorch_variable(obj_cc_input.obj_classifier.train_data))
	encoded_data_valid, decoded_data_valid = ae_model(get_pytorch_variable(obj_cc_input.obj_classifier.valid_data))
	encoded_data_test, decoded_data_test = ae_model(get_pytorch_variable(obj_cc_input.obj_classifier.test_data))
	
	obj_cc_output.encoded_data_train_cc1 = encoded_data_train.data.cpu().numpy()
	obj_cc_output.decoded_data_train_cc1 = decoded_data_train.data.cpu().numpy()
	obj_cc_output.encoded_data_valid_cc1 = encoded_data_valid.data.cpu().numpy()
	obj_cc_output.decoded_data_valid_cc1 = decoded_data_valid.data.cpu().numpy()
	obj_cc_output.encoded_data_test_cc1 = encoded_data_test.data.cpu().numpy()
	obj_cc_output.decoded_data_test_cc1 = decoded_data_test.data.cpu().numpy()

	#pdb.set_trace()
	return obj_cc_output


def function_train_pytorch_cc(obj_cc_input):
        obj_cc_output = train_pt_cc_output()
	input_output_dim = (obj_cc_input.cc1_input_train_perm).shape[1]
	hidden_layer_dim = obj_cc_input.dimension_hidden_layer1
	model = autoencoder(input_output_dim, hidden_layer_dim).cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

	for epoch in range(num_epochs):
		# ===================forward=====================
		encoded_train, decoded_train = model(get_pytorch_variable(obj_cc_input.cc1_input_train_perm))
		encoded_valid, decoded_valid = model(get_pytorch_variable(obj_cc_input.cc1_input_valid_perm))
		train_loss = criterion(decoded_train, get_pytorch_variable(obj_cc_input.cc1_output_train_perm))
		valid_loss = criterion(decoded_valid, get_pytorch_variable(obj_cc_input.cc1_output_valid_perm))
		# ===================backward====================
		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()
		# ===================log========================
		print('epoch [%3d/%3d], train loss:{:%4f} validation loss:{:%4f}'
		  %(epoch + 1, num_epochs, train_loss.data[0], valid_loss.data[0]))
		torch.save(model.state_dict(), './sim_autoencoder.pth')
	
	encoded_data_train, decoded_data_train = model(get_pytorch_variable(obj_cc_input.obj_classifier.train_data))
	encoded_data_valid, decoded_data_valid = model(get_pytorch_variable(obj_cc_input.obj_classifier.valid_data))
	encoded_data_test, decoded_data_test = model(get_pytorch_variable(obj_cc_input.obj_classifier.test_data))
	
	obj_cc_output.encoded_data_train_cc1 = encoded_data_train.data.cpu().numpy()
	obj_cc_output.decoded_data_train_cc1 = decoded_data_train.data.cpu().numpy()
	obj_cc_output.encoded_data_valid_cc1 = encoded_data_valid.data.cpu().numpy()
	obj_cc_output.decoded_data_valid_cc1 = decoded_data_valid.data.cpu().numpy()
	obj_cc_output.encoded_data_test_cc1 = encoded_data_test.data.cpu().numpy()
	obj_cc_output.decoded_data_test_cc1 = decoded_data_test.data.cpu().numpy()

	#pdb.set_trace()
	return obj_cc_output

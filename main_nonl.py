from sklearn.metrics import accuracy_score
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
import torch.nn.functional as F
import pdb
import time
import numpy as np
import scipy.io
import pt_utils_cub as utl
from pt_utils_cub import create_sample_wise_data
from sklearn import preprocessing

#create_sample_wise_data()

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
def get_samples(data, labels, classI, classJ):
	indices_classI = np.flatnonzero(labels == classI)
	indices_classJ = np.flatnonzero(labels == classJ)

	if len(indices_classI) > 1:
		sample_classI = torch.mean(data[indices_classI, :], 0, True)
	elif len(indices_classI) == 1:
		sample_classI = data[indices_classI, :]
	#else:
	#	sample_classI = 0*data[0, :]
	if len(indices_classJ) > 1:
		sample_classJ = torch.mean(data[indices_classJ, :], 0, True)
	elif len(indices_classJ) == 1:
		sample_classJ = data[indices_classJ, :]
	#else:	
	#	sample_classJ = 0*data[0, :]

	return sample_classI, sample_classJ

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------
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
			'''
                        self.fc1 = nn.Linear(self.input_output_dim, int(self.input_output_dim*0.5))
                        self.fc2 = nn.Linear(int(self.input_output_dim*0.5), self.hidden_layer_dim)
                        self.fc3 = nn.Linear(self.hidden_layer_dim, int(0.5*self.input_output_dim))
                        self.fc4 = nn.Linear(int(self.input_output_dim*0.5), self.input_output_dim)
                        init.xavier_normal(self.fc1.weight, gain=1)
                        init.xavier_normal(self.fc2.weight, gain=1)
                        init.xavier_normal(self.fc3.weight, gain=1)
                        init.xavier_normal(self.fc4.weight, gain=1)
                        self.fc1.bias.data.fill_(0)
                        self.fc2.bias.data.fill_(0)
                        self.fc3.bias.data.fill_(0)
                        self.fc4.bias.data.fill_(0)
			'''
	
                        self.fc1 = nn.Linear(self.input_output_dim, self.hidden_layer_dim)
                        self.fc2 = nn.Linear(self.hidden_layer_dim, self.input_output_dim)
                        init.xavier_normal(self.fc1.weight, gain=1)
                        init.xavier_normal(self.fc2.weight, gain=1)
                        self.fc1.bias.data.fill_(0)
                        self.fc2.bias.data.fill_(0)

        def forward(self, x):
                if 0:
                        encoded_features = self.encoder(x)
                        decoded_features = self.decoder(encoded_features)
                else:
			'''
                        z1 = F.relu(self.fc1(x))
                        encoded_features = F.relu(self.fc2(z1))
                        z2 = F.relu(self.fc3(encoded_features))
                        decoded_features = F.relu(self.fc4(z2))
			'''
			
                        encoded_features = F.leaky_relu(self.fc1(x))
                        decoded_features = F.leaky_relu(self.fc2(encoded_features))

                return encoded_features, decoded_features

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
class E2E_NETWORK_TRAIN(nn.Module):
        def __init__(self, visual_input_output_dim, semantic_input_output_dim, hidden_layer_dim, number_of_classes, number_of_seen_classes):
                super(E2E_NETWORK_TRAIN, self).__init__()
                self.visual_input_output_dim = visual_input_output_dim
                self.semantic_input_output_dim = semantic_input_output_dim
                self.hidden_layer_dim = hidden_layer_dim
		self.number_of_seen_classes = number_of_seen_classes
		self.number_of_classes = number_of_classes
                self.visual_ae = autoencoder(self.visual_input_output_dim, self.hidden_layer_dim)
                self.semantic_ae = autoencoder(self.semantic_input_output_dim, self.hidden_layer_dim)
                self.visual_ae.cuda()
                self.semantic_ae.cuda()
		self.visual_graph_mat = Variable(torch.zeros(self.number_of_seen_classes, self.number_of_seen_classes)).cuda()

        def forward(self, vdata, sdata, labels):
		visual_encoded_features, visual_decoded_features = self.visual_ae(vdata)               
		semantic_encoded_features, semantic_decoded_features = self.semantic_ae(sdata)
		classes = np.unique(utl.get_ndarray(labels.cpu()))
		labels = utl.get_ndarray(labels.cpu())
		labels = labels.astype(int)
		classes = classes.astype(int)
		#print self.visual_graph_mat.shape	
		for i in range(len(utl.SEEN_CLASSES)):
			for j in range(len((utl.SEEN_CLASSES))):
				classI = utl.SEEN_CLASSES[i]
				classJ = utl.SEEN_CLASSES[j]
				sample_classI, sample_classJ = get_samples(visual_encoded_features, labels, classI, classJ)
				#semantic_sample_classI, semantic_sample_classJ = get_samples(semantic_encoded_features, labels, classI, classJ )
				self.visual_graph_mat[i, j] = torch.exp(-1*torch.dist(sample_classI, sample_classJ))	
				#self.semantic_graph_mat[classI-1, classJ-1] = torch.exp(-1*torch.dist(semantic_sample_classI, semantic_sample_classI))		
				#print classI, classJ, self.visual_graph_mat[classI, classJ], self.semantic_graph_mat[classI, classJ]	
		return self.visual_graph_mat, visual_encoded_features, visual_decoded_features, semantic_encoded_features, semantic_decoded_features	               

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
class E2E_NETWORK_TEST(nn.Module):
        def __init__(self, visual_input_output_dim, semantic_input_output_dim, hidden_layer_dim, number_of_classes, number_of_seen_classes):
                super(E2E_NETWORK_TEST, self).__init__()
                self.visual_input_output_dim = visual_input_output_dim
                self.semantic_input_output_dim = semantic_input_output_dim
                self.hidden_layer_dim = hidden_layer_dim
		self.number_of_seen_classes = number_of_seen_classes
		self.number_of_classes = number_of_classes
                self.visual_ae = autoencoder(self.visual_input_output_dim, self.hidden_layer_dim)
                self.semantic_ae = autoencoder(self.semantic_input_output_dim, self.hidden_layer_dim)
                self.visual_ae.cuda()
                self.semantic_ae.cuda()
		self.visual_graph_mat = Variable(torch.zeros(self.number_of_classes, self.number_of_classes)).cuda()
		self.semantic_graph_mat = Variable(torch.zeros(self.number_of_classes, self.number_of_classes)).cuda()

        def forward(self, vdata, sdata, labels):
		visual_encoded_features, visual_decoded_features = self.visual_ae(vdata)               
		semantic_encoded_features, semantic_decoded_features = self.semantic_ae(sdata)
		return visual_encoded_features, semantic_encoded_features	               
'''
---------------------------------------------
Loss functions
---------------------------------------------
'''
mse_loss_function = nn.MSELoss(size_average=True)
cross_entropy_loss_function = nn.CrossEntropyLoss()# the target label is not one-hotted

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
def get_classifier_loss(predicted_labels, gt_labels):
        return cross_entropy_loss_function(predicted_labels, gt_labels)
        #return F.nll_loss(predicted_labels, gt_labels)

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
def get_mse_loss(data1, data2, custom_loss):
	if custom_loss == 'True':
        	mse_loss = torch.mean(torch.mul((data1 -  data2), (data1 - data2)))
	else:
	        mse_loss = mse_loss_function(data1, data2)

        return mse_loss.cuda()

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
def get_model_for_training(TRAIN_OPTION):
        if TRAIN_OPTION == 'TRAIN_FROM_SCRATCH':
		model = E2E_NETWORK_TRAIN(utl.VISUAL_FEATURE_DIM, utl.SEMANTIC_FEATURE_DIM, utl.HIDDEN_LAYER_DIM, utl.NUMBER_OF_CLASSES, utl.NUMBER_OF_SEEN_CLASSES)
        elif TRAIN_OPTION == 'RETRAIN':
		model = E2E_NETWORK_TRAIN(utl.VISUAL_FEATURE_DIM, utl.SEMANTIC_FEATURE_DIM, utl.HIDDEN_LAYER_DIM, utl.NUMBER_OF_CLASSES, utl.NUMBER_OF_SEEN_CLASSES)
                model.load_state_dict(torch.load(utl.MODEL_LOAD_PATH))
        return model

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
def get_accuracy(vdata, sdata, labels, classes):
	distance_mat = np.zeros((vdata.shape[0], classes.shape[0]))
        for i in range(vdata.shape[0]):
                tiled_vdata = np.tile(vdata[i, :], (classes.shape[0], 1))
                distances = np.sum(np.square(tiled_vdata-sdata), axis=1)
                distance_mat[i, :] = distances

        predicted_indices = np.argmin(distance_mat, axis = 1)
        predicted_classes = classes[predicted_indices]
        accuracy = accuracy_score(labels, predicted_classes)
	return accuracy*100.00

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
def get_semantic_graph(sdata_train_gt):
	semantic_graph_mat = Variable(torch.zeros(len(utl.SEEN_CLASSES), len(utl.SEEN_CLASSES)))
	for i in range(len(utl.SEEN_CLASSES)):
		for j in range(len(utl.SEEN_CLASSES)):
			semantic_graph_mat[i, j] = torch.exp(-1*torch.dist(sdata_train_gt[i, :], sdata_train_gt[j, :]))
	return semantic_graph_mat

#create_sample_wise_data()
model = get_model_for_training(utl.TRAIN_OPTION)
optimizer = torch.optim.Adam(model.parameters(), lr=utl.LEARNING_RATE, weight_decay=utl.WEIGHT_DECAY)

alpha = 1.0*1e-0
beta = 1*1e-0
gamma = 1*1e-0
theta = 1*1e-0
sdata_train_gt, sdata_train_labels = utl.get_semantic_prototypes_seen()
sdata_train_gt = Variable(torch.from_numpy(sdata_train_gt))
sdata_train_labels = Variable(torch.from_numpy(sdata_train_labels))
semantic_graph_mat = get_semantic_graph(sdata_train_gt)

'''
-----------------------------------------------------------------------
				Test
-----------------------------------------------------------------------
'''
def test(epoch):
	#print 'Accuracy on testing'
	test_model = E2E_NETWORK_TEST(utl.VISUAL_FEATURE_DIM, utl.SEMANTIC_FEATURE_DIM, utl.HIDDEN_LAYER_DIM, utl.NUMBER_OF_CLASSES, utl.NUMBER_OF_SEEN_CLASSES)
        test_model.load_state_dict(torch.load(utl.MODEL_LOAD_PATH))
        test_model.eval()
        test_model.cuda()
	vdata, sdata, sdata_gt, labels = utl.get_test_data()
	vdata = utl.get_pytorch_variable(vdata, 'False')
	sdata = utl.get_pytorch_variable(sdata, 'False')
	sdata_gt = utl.get_pytorch_variable(sdata_gt, 'False')
	labels = utl.get_pytorch_variable(labels, 'False')
	visual_encoded_features, semantic_encoded_features = test_model(vdata.cuda(), sdata_gt.cuda(), labels.cuda())
	
	#utl.visualise_features(utl.get_ndarray(visual_encoded_features.cpu()), (utl.get_ndarray(labels.cpu()).astype(int)))	
	accuracy = get_accuracy(utl.get_ndarray(visual_encoded_features.cpu()), utl.get_ndarray(semantic_encoded_features.cpu()), utl.get_ndarray(labels.cpu()), utl.UNSEEN_CLASSES)
	utl.visualize_attributes_and_proto(utl.get_ndarray(visual_encoded_features.cpu()), utl.get_ndarray(labels.cpu()), utl.UNSEEN_CLASSES, utl.UNSEEN_CLASS_NAMES, utl.get_ndarray(semantic_encoded_features.cpu()), str(epoch), str(epoch), utl.DATA_SAVE_PATH)
	
	return accuracy
'''
-----------------------------------------------------------------------
				Validate
-----------------------------------------------------------------------
'''
def validate():
	#print 'Accuracy on training'
	model = E2E_NETWORK_TEST(utl.VISUAL_FEATURE_DIM, utl.SEMANTIC_FEATURE_DIM, utl.HIDDEN_LAYER_DIM, utl.NUMBER_OF_CLASSES, utl.NUMBER_OF_SEEN_CLASSES)
        model.load_state_dict(torch.load(utl.MODEL_LOAD_PATH))
        model.eval()
        model.cuda()
	vdata, sdata, sdata_gt, labels = utl.get_validation_data()
	vdata = utl.get_pytorch_variable(vdata, 'False')
	sdata = utl.get_pytorch_variable(sdata, 'False')
	sdata_gt = utl.get_pytorch_variable(sdata_gt, 'False')
	labels = utl.get_pytorch_variable(labels, 'False')
	visual_encoded_features, semantic_encoded_features = model(vdata.cuda(), sdata_gt.cuda(), labels.cuda())
	
	#utl.visualise_features(utl.get_ndarray(visual_encoded_features.cpu()), (utl.get_ndarray(labels.cpu()).astype(int)))	
	#utl.visualize_attributes_and_proto(utl.get_ndarray(visual_encoded_features.cpu()), utl.get_ndarray(labels.cpu()), utl.UNSEEN_CLASSES, utl.UNSEEN_CLASS_NAMES, utl.get_ndarray(semantic_encoded_features.cpu()), 'plot', 'plot', utl.MODEL_SAVE_PATH)
	
	#utl.visualize_attributes_and_proto(utl.get_ndarray(visual_encoded_features.cpu()), utl.get_ndarray(labels.cpu()), utl.SEEN_CLASSES, utl.SEEN_CLASS_NAMES, utl.get_ndarray(semantic_encoded_features.cpu()), 'plot', 'plot', utl.MODEL_SAVE_PATH)
	accuracy = get_accuracy(utl.get_ndarray(visual_encoded_features.cpu()), utl.get_ndarray(semantic_encoded_features.cpu()), utl.get_ndarray(labels.cpu()), utl.SEEN_CLASSES)

	return accuracy

'''
-----------------------------------------------------------------------
				Train
-----------------------------------------------------------------------
'''
if 1:
	#awa_dataset = utl.AwADataset(utl.DATA_SAVE_PATH + utl.DATASET_NAME + '_samplewise_data/')
	#dataloader = DataLoader(awa_dataset, batch_size=512, shuffle=True, num_workers=4)

	for epoch in range(utl.EPOCHS):
		utl.shuffle_data()
		#for i_batch, sample_batched in enumerate(dataloader):
		test_accuracy = test(epoch)
		validation_accuracy = validate()
		for i_batch in range(utl.NUMBER_OF_TRAIN_BATCHES):
			
			vdata, sdata, labels = utl.get_data(i_batch)	
			#vdata = sample_batched[0]
			#sdata = sample_batched[1]
			#labels = sample_batched[2]
			sdata = preprocessing.normalize(sdata, norm='l2', axis=1)
			vdata = preprocessing.normalize(vdata, norm='l2', axis=1)
			#labels = torch.squeeze(labels)
			labels = Variable(torch.from_numpy(labels))
			vdata = Variable(torch.from_numpy(vdata), requires_grad=False)
			sdata = Variable(torch.from_numpy(sdata), requires_grad=False)
			vdata = vdata.type(torch.FloatTensor)
			sdata = sdata.type(torch.FloatTensor)
			labels = labels.type(torch.FloatTensor)
			visual_latent_graph, visual_encoded_features, visual_decoded_features, semantic_encoded_features, semantic_decoded_features = model(vdata.cuda(), sdata.cuda(), labels.cuda())
			graph_loss = get_mse_loss(visual_latent_graph.cuda(), semantic_graph_mat.cuda(), 'True')
			visual_ae_loss = get_mse_loss(visual_decoded_features, vdata.cuda(), 'False')
			semantic_ae_loss = get_mse_loss(semantic_decoded_features, sdata.cuda(), 'False')
			latent_loss = get_mse_loss(semantic_encoded_features, visual_encoded_features, 'True')
			loss = alpha*visual_ae_loss + beta * semantic_ae_loss + gamma * graph_loss + theta * latent_loss
			optimizer.zero_grad()

			print 'Epoch [%3d/%3d], Batch [%3d],  Visual AE loss %2.4f, Semantic AE loss %2.4f, Graph Loss %4.4f, Latent Loss %4.4f, Total loss %4.4f, Test Acc %2.2f, Validation Acc %2.2f'%(epoch + 1, utl.EPOCHS, i_batch + 1, alpha*visual_ae_loss.data[0], beta*semantic_ae_loss.data[0], gamma*graph_loss.data[0], theta*latent_loss, loss, test_accuracy, validation_accuracy)

			loss.backward(retain_graph=True)
			optimizer.step()
		utl.save_model(model)



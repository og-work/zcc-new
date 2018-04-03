from sklearn import preprocessing
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
import torch.nn.functional as F
#from torchviz import make_dot, make_dot_from_trace
import time
import os 
import scipy.io
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import Dataset, DataLoader

'''
--------------------------------------
Constants/Macro
--------------------------------------
'''
CUDA_TYPE = torch.cuda.FloatTensor
NUMBER_OF_EPOCHS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUMBER_OF_CLASSES = 50
NUMBER_OF_SEEN_CLASSES = 40
MIN_NUMBER_OF_SAMPLES_OF_CLASS = 93#class 12 of AWA
TR_TS_VA_SPLIT = [0.8, 0.2]

NUMBER_OF_TRAIN_SAMPLES = 93*40#24295#min samples 93 from class12 of AWA
PER_CLASS_SAMPLES = 20
TRAIN_BATCH_SIZE = NUMBER_OF_SEEN_CLASSES*PER_CLASS_SAMPLES
EPOCHS = 500
NUMBER_OF_TRAIN_BATCHES = int(NUMBER_OF_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
VISUAL_FEATURE_DIM = 1024
SEMANTIC_FEATURE_DIM = 85
HIDDEN_LAYER_DIM = 50

DATA_LOAD_PATH = '/nfs4/omkar/Documents/study/phd-research/data/datasets/AWA/semantic_ae_data/'
DATASET_NAME = 'AWA'
DATA_SAVE_PATH = '../../data-zsl/AWA/'
ITR = 1 #for saving two models with different names to avoid corruption
MODEL_SAVE_PATH = DATA_SAVE_PATH + 'e2e_AWA_model1_4_'
MODEL_LOAD_PATH = DATA_SAVE_PATH + 'e2e_AWA_model1_4_' + str(ITR) + '.pth'
LIST_TRAIN_OPTIONS = ['TRAIN_FROM_SCRATCH', 'RETRAIN']
TRAIN_OPTION = LIST_TRAIN_OPTIONS[0]

ALL_CLASSES = np.arange(0,NUMBER_OF_CLASSES, 1)
UNSEEN_CLASSES = np.array([25,39,15,6,42,14,18,48,34,24])
UNSEEN_CLASSES = UNSEEN_CLASSES - 1
SEEN_CLASSES = np.delete(ALL_CLASSES, UNSEEN_CLASSES)
SEEN_CLASSES = SEEN_CLASSES + 1
UNSEEN_CLASSES = UNSEEN_CLASSES + 1
UNSEEN_CLASS_NAMES = []
SEEN_CLASS_NAMES = []
for cl in UNSEEN_CLASSES:
	UNSEEN_CLASS_NAMES.append(str(cl))
for cl in SEEN_CLASSES:
	SEEN_CLASS_NAMES.append(str(cl))
#INITIAL_WEIGHT_PATH = '/home/SharedData/omkar/study/phd-research/codes/tf-codes/data-willow/initial_weights/'
'''
Model 4: similarity matrix using exp(-dist)
'''


'''
--------------------------------------------
Dataset loader
--------------------------------------------
'''
class AwADataset(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir

	def __len__(self):
		return NUMBER_OF_TRAIN_SAMPLES

	def __getitem__(self, idx):
		sample_path = self.root_dir + 'sample_train_valid_' + str(idx + 1)
		tmp = scipy.io.loadmat(sample_path)
		vdata = tmp['v_sample'].flatten()
		sdata = tmp['s_sammple'].flatten()
		label = tmp['label']

		return vdata, sdata, label


'''
---------------------------------------------------------
Utility functions for pytorch and numpy
---------------------------------------------------------
'''
def get_pytorch_variable(ip_ndarray, requires_grad):
	if requires_grad == 'False':
		pt_tensor = Variable(torch.from_numpy(ip_ndarray), requires_grad=False)
		
	pt_tensor = Variable(torch.from_numpy(ip_ndarray))
        return pt_tensor.type(torch.FloatTensor)

def get_pytorch_long_variable(ip_ndarray, requires_grad):
	if requires_grad == 'False':
		pt_tensor = Variable(torch.from_numpy(ip_ndarray), requires_grad=False)
	pt_tensor = Variable(torch.from_numpy(ip_ndarray))
        return pt_tensor.type(torch.LongTensor)
	
def get_ndarray(ip_pt_tensor):
        return ip_pt_tensor.data.numpy()


'''
---------------------------------------------------------
Save model using criteria
---------------------------------------------------------
'''
def save_model(model):
	#Use criterion to save the model	
	#Use criterion to save the model        
        global ITR
        if ITR == 1:
                ITR = 2
                torch.save(model.state_dict(), MODEL_SAVE_PATH + str(ITR) + '.pth')
        else:
                ITR = 1
                torch.save(model.state_dict(), MODEL_SAVE_PATH + str(ITR) + '.pth')

def create_sample_wise_data():
	temp = scipy.io.loadmat(DATA_LOAD_PATH + 'awa_demo_data')
	temp1 = scipy.io.loadmat(DATA_LOAD_PATH + 'awa_prototypes')
	vdata_train = temp['X_tr']
	vdata_test = temp['X_te']
	sdata_train = temp['S_tr']
	sdata_test = temp['S_te_gt']
	labels_train = temp1['tr_labels']
	labels_test = temp1['ts_labels']
	labels_train = labels_train.flatten()
	labels_test = labels_test.flatten()

        #pdb.set_trace()
        dir_name = DATA_SAVE_PATH + DATASET_NAME + '_samplewise_data'
        if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        num_samples_train_valid = vdata_train.shape[0]
        num_samples_test = vdata_test.shape[0]
        dim_vfeat = vdata_train.shape[1]
	for cl in SEEN_CLASSES:
	        indices = np.flatnonzero(labels_train == cl)
		class_dir = dir_name + '/class_' + str(cl) 
		shuffled_indices_train_cl = np.arange(1, len(indices) + 1)
		np.random.shuffle(shuffled_indices_train_cl)
		shuffled_indices_train_cl = shuffled_indices_train_cl.flatten()
		file_path = class_dir + '/shuffled_indices_train_class_' + str(cl) 
		scipy.io.savemat(file_path, dict(shuffled_indices_train = shuffled_indices_train_cl))
		if not os.path.exists(class_dir):
			os.makedirs(class_dir)
		#for i in range(len(indices)):
		#	sample_path = class_dir + '/sample_train_valid_' + str(i+1)
		#	scipy.io.savemat(sample_path, dict(v_sample = vdata_train[indices[i], :],
		#					   s_sammple = sdata_train[indices[i], :],	
		#					   label = labels_train[indices[i]]))
		#	print "Writing sample %s label %d"%(sample_path, labels_train[indices[i]])
			
	pdb.set_trace()
        for i in range(num_samples_test):
                sample_path = dir_name + '/sample_test_' + str(i + 1)
                scipy.io.savemat(sample_path, dict(v_sample = vdata_test[i, :],
						   label = labels_test[i]))
                print "Writing sample %s %d"%(sample_path, labels_test[i])
        
        return


def get_data(batch):
	vdata = np.empty((TRAIN_BATCH_SIZE, VISUAL_FEATURE_DIM), float)       
	sdata = np.empty((TRAIN_BATCH_SIZE, SEMANTIC_FEATURE_DIM), float)       
	labels = []       
	p = 0
	for cl in SEEN_CLASSES:
		sample_path = DATA_SAVE_PATH + DATASET_NAME + '_samplewise_data/class_' + str(cl) + '/shuffled_indices_train_class_' + str(cl)
		t = scipy.io.loadmat(sample_path)
		shuffled_indices_train = t['shuffled_indices_train']
		shuffled_indices_train = shuffled_indices_train.flatten()
		st = batch*PER_CLASS_SAMPLES
		end = batch*PER_CLASS_SAMPLES + PER_CLASS_SAMPLES
		array = np.arange(st,end)
		indices = shuffled_indices_train[array%len(shuffled_indices_train)]
		for k in indices:
	                sample_path = DATA_SAVE_PATH + DATASET_NAME + '_samplewise_data/class_' + str(cl) + '/sample_train_valid_' + str(k)
			tmp = scipy.io.loadmat(sample_path)
			vdata[p, :] = tmp['v_sample'].flatten()
			sdata[p, :] = tmp['s_sammple'].flatten()
			labels.append(tmp['label'])
			p = p + 1
	#vdata = preprocessing.normalize(vdata, norm='l2', axis=1)		
	labels = np.asarray(labels).flatten()
	sdata = preprocessing.normalize(sdata, norm='l2', axis=1)		
	idx = np.random.permutation(vdata.shape[0])
	shuffled_vdata = vdata[idx]
	shuffled_sdata = sdata[idx]
	shuffled_labels = labels[idx]
	return shuffled_vdata, shuffled_sdata, shuffled_labels	


def shuffle_data():
        dir_name = DATA_SAVE_PATH + DATASET_NAME + '_samplewise_data'
	for cl in SEEN_CLASSES:
		file_path = dir_name + '/class_' + str(cl) +  '/shuffled_indices_train_class_' + str(cl) 
		t = scipy.io.loadmat(file_path)
		shuffled_indices_train = t['shuffled_indices_train']
		np.random.shuffle(shuffled_indices_train)
		shuffled_indices_train = shuffled_indices_train.flatten()
		scipy.io.savemat(file_path, dict(shuffled_indices_train = shuffled_indices_train))

def get_validation_data():
	temp = scipy.io.loadmat(DATA_LOAD_PATH + 'awa_demo_data')
	temp1 = scipy.io.loadmat(DATA_LOAD_PATH + 'awa_prototypes')
	vdata_train = temp['X_tr']
	vdata_test = temp['X_te']
	sdata_train = temp['S_tr']
	sdata_gt = temp['S_te_gt']
	labels_train = temp1['tr_labels']
	labels_test = temp1['ts_labels']
	labels_train = labels_train.flatten()
	labels_test = labels_test.flatten()
	#vdata_train = preprocessing.normalize(vdata_train, norm='l2', axis=1)		
	sdata_gt = np.zeros((len(SEEN_CLASSES), sdata_train.shape[1]), float)
	k = 0
	for cl in SEEN_CLASSES:
                indices = np.flatnonzero(labels_train == cl)
		sdata_gt[k, :] = sdata_train[indices[0], :]
		k = k + 1
	sdata_gt = preprocessing.normalize(sdata_gt, norm='l2', axis=1)		
	return vdata_train, sdata_train, sdata_gt, labels_train

def get_test_data():
	temp = scipy.io.loadmat(DATA_LOAD_PATH + 'awa_demo_data')
	temp1 = scipy.io.loadmat(DATA_LOAD_PATH + 'awa_prototypes')
	vdata_train = temp['X_tr']
	vdata_test = temp['X_te']
	sdata_train = temp['S_tr']
	sdata_gt = temp['S_te_gt']
	labels_train = temp1['tr_labels']
	labels_test = temp1['ts_labels']
	labels_train = labels_train.flatten()
	labels_test = labels_test.flatten()
	sdata_test = np.zeros((len(labels_test), sdata_gt.shape[1]), float)
	#vdata_train = preprocessing.normalize(vdata_train, norm='l2', axis=1)		
	sdata_gt = preprocessing.normalize(sdata_gt, norm='l2', axis=1)		
	k = 0
	for cl in UNSEEN_CLASSES:
                indices = np.flatnonzero(labels_test == cl)
		sdata_test[indices, :] = sdata_gt[k, :]
		k = k + 1
	return vdata_test, sdata_test, sdata_gt, labels_test

def visualise_features(data, labels):
        pdb.set_trace()
	tsne = TSNE(n_components=2)
        Y = tsne.fit_transform(data)
        colors = cm.rainbow(np.linspace(0, 1, UNSEEN_CLASSES.shape[0]))
        for classI, color in zip(UNSEEN_CLASSES, colors):
                indices = np.flatnonzero(labels == classI)
                plt.scatter(Y[indices, 0], Y[indices, 1], c=color)
        plt.legend(loc='best', fontsize=10)
        plt.show()
        #plt.close("all")
	return	

def visualize_attributes_and_proto(data, labels, classes, dataset_class_names, prototypes, string_title, figure_title, data_save_path):
        #pdb.set_trace()
        #data_and_proto = np.vstack((data, prototypes))
        #data_and_proto = scale(data_and_proto)
        data_proto = np.vstack((data, prototypes))
        #pca = PCA(n_components=2)
        #Y = pca.fit_transform(data_proto)
        tsne = TSNE(n_components=2)
        Y = tsne.fit_transform(data_proto)
        #PCAed_proto = pca.transform(prototypes)
        #Y = np.vstack((Y, PCAed_proto))
        #print Y.shape
        marker_size = 15
        colors = cm.rainbow(np.linspace(0, 1, classes.shape[0]))
        plt.close("all")
        #classes = classes - 1
        #class_names = [dataset_class_names[i] for i in classes]
        #classes = classes + 1
        for classI, color, name in zip(classes, colors, dataset_class_names):
                indices = np.flatnonzero(labels == classI)
                plt.scatter(Y[indices, 0], Y[indices, 1], label = name, s = marker_size, c=color)

        marker_size = 70
	k = 0
        for classI, color, name in zip(classes, colors, dataset_class_names):
                indices = data.shape[0] + k#(classI - 1)
                plt.scatter(Y[indices, 0], Y[indices, 1], label = name + ' PROTOTYPE', s = marker_size, c=color, marker='D')
                k = k + 1

        plt.title(string_title)
        #plt.legend(loc='best', fontsize=10)
        plt.xticks([])
        plt.yticks([])
        figurename = data_save_path + '/plots/' + figure_title
        #plt.savefig(figurename + '.eps', format='eps',dpi=1000, bbox_inches='tight')
        plt.savefig(figurename + '.png', bbox_inches = 'tight')
        #plt.show()
        #pdb.set_trace()
        #plt.close("all")

def get_semantic_prototypes_seen():
	temp1 = scipy.io.loadmat(DATA_LOAD_PATH + 'seen_prototypes_and_labels')
	sdata_gt = temp1['sdata_train_gt']
	sdata_labels = temp1['seen_proto_labels']
	return sdata_gt, sdata_labels

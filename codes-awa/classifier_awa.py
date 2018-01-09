'''
-----------------------------------------------------------------
                classifier_awa.py
-----------------------------------------------------------------
'''

import numpy as np
import pdb
from sklearn.metrics import accuracy_score
import scipy.io
import pickle

DATA_SAVE_PATH = '/home/SharedData/omkar/study/phd-research/codes/tf-codes/data-zsl/data-awa/data4/'
base_filename = DATA_SAVE_PATH + 'AwA_100_1000_'
filename = base_filename + 'obj_classifier.p'
print "Loading data for classifier from %s"%filename
obj_classifier = pickle.load(open(filename, "rb"))



distance_mat = np.zeros((obj_classifier.test_attributes_nsp.shape[0], obj_classifier.number_of_test_classes), float)
k = 0
#NOTE: These weights to be learned.
weights = np.ones((obj_classifier.number_of_train_classes, 1), float)
#weights = weights.flatten()
weights = weights/obj_classifier.number_of_train_classes
weights[0] = 0.4
weights[1] = 0.6
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
        #pdb.set_trace()
	distance_mat[:, k] = 0*sqrd_distances_nsp + sqrd_distances_sp
        k = k + 1

indices_predicted_unseen_classes = np.argmin(distance_mat, axis = 1)
predicted_unseen_classes = obj_classifier.test_class_labels[indices_predicted_unseen_classes]
accuracy = accuracy_score(obj_classifier.test_labels, predicted_unseen_classes)
print "Dataset AwA: Accuracy %f"%(accuracy)



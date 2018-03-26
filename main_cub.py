import numpy as np
import scipy.io
import scipy.linalg
import pdb
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

DATA_PATH = '/nfs4/omkar/Documents/study/phd-research/data/datasets/AWA/semantic_ae_data/'
lambda1 = 1e7
lambda2 = 1e7


'''
--------------------------------------------------------
Normalise features
--------------------------------------------------------
'''

def normalise_features(feature):
	return preprocessing.normalize(feature, norm='l2', axis=0)
	
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
'''

def visualise_data(data, labels):
	print np.unique(labels)
	print data.shape
	tsne = TSNE(n_components=2)
	tsne_results = tsne.fit_transform(np.transpose(data))
	plt.figure()
	plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='hsv')
	plt.legend()
	plt.show()
	#plt.close_all()

'''
--------------------------------------------------------
Visual branch
--------------------------------------------------------
'''
def train_visual_branch(vdata_train, prototypes, labels_train, lambda1):
	A_relation_mat = np.matmul(prototypes.transpose(), prototypes)
	#pdb.set_trace()
	label_mat = np.zeros((number_of_classes, vdata_train.shape[1]))
	label_mat[labels_train - 1, np.arange(vdata_train.shape[1])] = 1
	t1 = np.matmul(A_relation_mat, label_mat)
	t2 = np.matmul(label_mat.transpose(), A_relation_mat.transpose())
	A = np.matmul(label_mat, label_mat.transpose())#np.dot(t1,t2)
	B = lambda1*np.matmul(vdata_train, vdata_train.transpose())
	C = np.matmul(label_mat, lambda1*vdata_train.transpose() + vdata_train.transpose())
	W1 = scipy.linalg.solve_sylvester(A, B, C)
 
	return W1

'''
--------------------------------------------------------
Semantic branch
--------------------------------------------------------
'''

def train_semantic_branch(prototypes, prototypes_labels, lambda2):	
	A_relation_mat = np.matmul(prototypes.transpose(), prototypes)
	#pdb.set_trace()
	label_mat = np.identity(number_of_classes)
	t1 = np.matmul(A_relation_mat, label_mat)
	t2 = np.matmul(label_mat.transpose(), A_relation_mat.transpose())
	A = label_mat#np.matmul(t1,t2)
	B = lambda2*np.matmul(prototypes, prototypes.transpose())
	C = (1 + lambda2)*np.matmul(t1, prototypes.transpose())
	W2 = scipy.linalg.solve_sylvester(A, B, C)
 
	return W2

def test_zsl(W1, W2, vdata_test, prototypes, prototypes_labels, labels_test, unseen_classes):
	vdata_projections = np.matmul(W1, vdata_test)
	sdata_projections = np.matmul(W2, prototypes)
	#visualise_data(vdata_projections, labels_test)
	distance_mat = np.zeros((unseen_classes.shape[0], vdata_test.shape[1]))
	unseen_proto_mat = sdata_projections[:, unseen_classes]
	for i in range(vdata_projections.shape[1]):
		for i in range(unseen_classes.shape[0]):
			tiled_vdata = np.tile(vdata_projections[:, i], (unseen_classes.shape[0],1))
			tiled_vdata = tiled_vdata.transpose()
			distances = np.sum(np.square(tiled_vdata-unseen_proto_mat), axis=0)		
			distance_mat[:, i] = distances

	predicted_indices = np.argmin(distance_mat, axis = 0)	
	predicted_classes = unseen_classes[predicted_indices]
        accuracy = accuracy_score(labels_test, predicted_classes)
	cnf_matrix = confusion_matrix(labels_test, predicted_classes)
	print cnf_matrix
	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=unseen_classes, normalize=True,
			      title='Normalized confusion matrix')
	plt.show()
	pdb.set_trace()
	return accuracy	

temp = scipy.io.loadmat(DATA_PATH + 'awa_demo_data')
temp1 = scipy.io.loadmat(DATA_PATH + 'awa_prototypes')
number_of_classes = 50
all_classes = np.arange(0,50,1)
unseen_classes = np.array([25,39,15,6,42,14,18,48,34,24])
unseen_classes = unseen_classes - 1
seen_classes = np.delete(all_classes, unseen_classes)
seen_classes = seen_classes + 1
unseen_classes = unseen_classes + 1
vdata_train = temp['X_tr']
vdata_test = temp['X_te']
sdata_train = temp['S_tr']
sdata_test = temp['S_te_gt']
labels_train = temp1['tr_labels']
labels_test = temp1['ts_labels']
labels_train = labels_train.flatten()
labels_test = labels_test.flatten()
vdata_train = vdata_train.transpose()
vdata_test = vdata_test.transpose()
sdata_train = sdata_train.transpose()
sdata_test = sdata_test.transpose()

#vdata_train = normalise_features(vdata_train)
print 'v train %d %d'%vdata_train.shape
print 'v test %d %d'%vdata_test.shape
print 's train %d %d'%sdata_train.shape
print 's test %d %d'%sdata_test.shape
prototypes = temp1['prototypes']
prototypes_labels = temp1['prototypes_labels']
prototypes_labels = prototypes_labels.flatten()
prototypes = (prototypes.transpose())
print 'prototypes %d %d'%prototypes.shape

lambda1_array = np.linspace(1e7, 1e8, 1e4)
lambda2_array = np.linspace(1e7, 1e8, 1e4)
for i in range(lambda1_array.shape[0]):
	for j in range(lambda2_array.shape[0]):
		lambda1 = lambda1_array[i]	
		lambda2 = lambda2_array[j]	
		#print 'lambda1 %f, lambda2 %f'%(lambda1, lambda2)
		W1 = train_visual_branch(vdata_train, prototypes, labels_train, lambda1)
		W2 = train_semantic_branch(prototypes, prototypes_labels, lambda2)
		#W1 = normalise_features(W1.transpose())
		#W2 = normalise_features(W2.transpose())
		#W1 = W1.transpose()
		#W2 = W2.transpose()
		#print W1
		#print W2
		accuracy = test_zsl(W1, W2, vdata_test, prototypes, prototypes_labels, labels_test, unseen_classes)
		print 'Accuracy:%f, lambda1 %f, lambda2 %f'%(100*accuracy, lambda1, lambda2)




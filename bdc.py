import numpy as np
import os
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import scipy.io
import pdb
from matplotlib import rc,rcParams

class_labels = np.arange(3,8)
print class_labels
#filepath = "data-willow/data2_hemachandra/willow_50_500_cec_features_all_classes_tr_"#7.mat"
filepath = "data-willow/data2_hemachandra/willow_50_500_cec_features_class_tr_"#7.mat"
encoder_norm_arr = []
decoder_norm_arr = []
tmp = scipy.io.loadmat("data-willow/data2_hemachandra/willow_50_500_feat_fusion_clsfr_train_labels")
train_labels = tmp['train_labels']

for classI in class_labels:
	for classJ in class_labels:
		if classI != classJ:
			print classI, classJ
			strng = str(classI)
			if os.path.isfile(filepath+strng+'.mat'):

				tmp = scipy.io.loadmat(filepath + strng)
				encoded_features = tmp['cross_feautures_tr']
				classI_indices = np.flatnonzero(train_labels == classI)
				same_class_samples = encoded_features[classI_indices, :]
				sc_neuron_var = np.var(same_class_samples, 0)
			
				random_classes = np.delete(class_labels, classI - 1)
				np.random.shuffle(np.asarray(random_classes))
				classK = random_classes[0]
				classJ_indices = np.flatnonzero(train_labels == classK)
				diff_class_samples = encoded_features[classJ_indices, :]
				dc_neuron_var = np.var(diff_class_samples, 0)

				rep_same_class_samples = np.repeat(same_class_samples, diff_class_samples.shape[0], axis=0)	
				rep1_same_class_samples = np.repeat(same_class_samples, same_class_samples.shape[0], axis=0)	
				rep2_same_class_samples = np.tile(same_class_samples, (same_class_samples.shape[0],1))	
				rep_diff_class_samples = np.tile(diff_class_samples, (same_class_samples.shape[0], 1))
	
				bdc_diff_class = abs(rep_same_class_samples - rep_diff_class_samples)
				bdc_same_class = abs(rep1_same_class_samples - rep2_same_class_samples)
				mean_bdc_same_class = np.mean(bdc_same_class, axis=0)
				mean_bdc_diff_class = np.mean(bdc_diff_class, axis=0)
				#pdb.set_trace()	
				
				plt.figure()
				plt.title('Bounded difference for neuron activation: CE: class ' + str(classI) + '(input class) to class ' + str(classJ) + ' (output class)', fontsize=20)
				plt.ylabel('Bounded difference ', fontsize = 20)
				plt.xlabel('Neurons', fontsize=20)
				ax = plt.gca()
				ax.yaxis.set_tick_params(labelsize=20)
				ax.xaxis.set_tick_params(labelsize=20)
				#plt.plot(dc_neuron_var, 'r', label = 'Input samples from class ' + str(classJ) + ' (non-input class)', linewidth=5)
				#plt.plot(sc_neuron_var, 'g', label = 'Input samples from class ' + str(classI) + ' (input class)', linewidth=5)
				plt.plot(bdc_diff_class[25, :], 'r', label = 'Bounded difference between a sample of class ' + str(classI) + ' and a sample of class ' + str(classK), linewidth=3)
				plt.plot(bdc_same_class[23, :], 'g', label = 'Bounded difference between two samples of the same class ' + str(classI) + ' (input class)', linewidth=3)
				plt.plot(mean_bdc_diff_class, 'm', label = 'Mean of bounded difference between samples of class ' + str(classI) + ' and samples of class ' + str(classK), linewidth=3)
				plt.plot(mean_bdc_same_class, 'c', label = 'Mean of bounded difference between samples of the same class ' + str(classI) + ' (input class)', linewidth=3)
				plt.legend(loc='best', prop={'size':20})
				plt.grid()
				plt.show()
				figurename = 'Bounded_diff_class_' + str(classI) + '_' + str(classJ) + 'mean'
				#plt.savefig(figurename + '.eps', format='eps',dpi=1000, bbox_inches='tight')
				

				plt.figure()
				plt.title('Bounded difference for neuron activation: CE: class ' + str(classI) + '(input class) to class ' + str(classJ) + ' (output class)', fontsize=25)
				plt.ylabel('Mean/variance of bounded difference ', fontsize = 25)
				plt.xlabel('Neurons', fontsize=30)
				ax = plt.gca()
				ax.yaxis.set_tick_params(labelsize=25)
				ax.xaxis.set_tick_params(labelsize=25)
				plt.plot(mean_bdc_diff_class, 'm', label = 'Mean of bounded difference between samples of class ' + str(classI) + ' and samples of class ' + str(classK), linewidth=3)
				plt.plot(dc_neuron_var, 'r', label = 'Variance of bounded difference between samples for class ' + str(classI) + ' and samples of class ' + str(classK), linewidth=3)
				plt.plot(mean_bdc_same_class, 'c', label = 'Mean of bounded difference between samples of the same class ' + str(classI) + ' (input class)', linewidth=3)
				plt.plot(sc_neuron_var, 'g', label = 'Variance of bounded difference between samples of the same class ' + str(classI) + ' (input class)', linewidth=5)
				plt.legend(loc='best', prop={'size':22})
				plt.grid()
				plt.show()
				figurename = 'Bounded_diff_class_' + str(classI) + '_' + str(classJ) + 'mean'
				#plt.savefig(figurename + '.eps', format='eps',dpi=1000, bbox_inches='tight')
				


plt.close_all()

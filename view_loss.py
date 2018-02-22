import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb

t = scipy.io.loadmat('../../data-willow/model1/mse_losses_for_coders_train')
losses = t['mse_losses_for_coders_train']
losses = losses.transpose()
avg = np.sum(losses, axis=0)
plt.figure()
#plt.title('Avg loss over all coders across batches')
#plt.plot(avg[1:]/49.)
#plt.grid()
#plt.show()

if 1:
	t1 = scipy.io.loadmat('../../data-willow/model1/total_losstrain')
	loss_train = t1['total_loss']
	plt.title('Total train loss')
	plt.plot((loss_train.flatten())[1:])
	plt.grid()
	plt.show()

	t2 = scipy.io.loadmat('../../data-willow/model1/total_lossvalid')
	loss_valid = t2['total_loss']
	plt.title('Total valid loss')
	plt.plot((loss_valid.flatten())[1:], 'r')
	plt.grid()
	plt.show()

	t3 = scipy.io.loadmat('../../data-willow/model1/mse_losstrain')
	mse_train = t3['mse_loss']
	plt.title('MSE train loss')
	plt.plot((mse_train.flatten())[1:], 'r')
	plt.grid()
	plt.show()

	t4 = scipy.io.loadmat('../../data-willow/model1/mse_lossvalid')
	mse_valid = t4['mse_loss']
	plt.title('MSE valid loss')
	plt.plot((mse_valid.flatten())[1:], 'r')
	plt.grid()
	plt.show()

	t5 = scipy.io.loadmat('../../data-willow/model1/clafr_losstrain')
	clafr_train = t5['clafr_loss']
	plt.title('Clafr train loss')
	plt.plot((clafr_train.flatten())[1:], 'r')
	plt.grid()
	plt.show()

	t6 = scipy.io.loadmat('../../data-willow/model1/clafr_lossvalid')
	clafr_valid = t6['clafr_loss']
	plt.title('Clafr valid loss')
	plt.plot((clafr_valid.flatten())[1:], 'r')
	plt.grid()
	plt.show()
	'''	
	t7 = scipy.io.loadmat('../../data-willow/model1/acc_train')
	acc_train = t7['acc']
	plt.title('Acc train')
	plt.plot((acc_train.flatten())[1:], 'r')
	plt.grid()
	plt.show()
	
	t8 = scipy.io.loadmat('../../data-willow/model1/acc_valid')
	acc_valid = t8['acc']
	plt.title('Acc valid')
	plt.plot((acc_valid.flatten())[1:], 'r')
	plt.grid()
	plt.show()
	'''

i = 0
for input_class in range(7):
	for output_class in range(7):
		plt.figure()
		plt.plot(losses[i,1:])
		i += 1
		plt.title('Loss for a coder ' + str(input_class + 1) + ' to ' + str(output_class + 1))
		plt.grid()
		plt.show()
		plt.close("all")


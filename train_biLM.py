import biLM
import train_set_biLM_utils
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

data_path = './text8'
savepath = './npy/'
tensorflow_saver_path = './saver'
top_voca = 10000
window_size = 10
embedding_size = 300
x_max = 100
lr = 0.05


def weighting_function(data, x_max):
	# data: [N, 1]

	# if x < x_max
	weighting = data.copy()
	weighting[data<x_max] = (data[data<x_max]/x_max)**(3/4)
	# else
	weighting[data>=x_max] = 1.0

	return weighting


def train(model, dataset, x_max, lr):
	batch_size = 256
	loss = 0

	np.random.shuffle(dataset)

	for i in tqdm(range( int(np.ceil(len(dataset)/batch_size)) ), ncols=50):
		batch = dataset[batch_size * i: batch_size * (i + 1)] # [batch_size, 3]

		i_word_idx = batch[:, 0:1] # [batch_size, 1]
		k_word_idx = batch[:, 1:2] # [batch_size, 1] 
		target = batch[:, 2:].astype(np.float32) # [batch_size, 1] # will be applied log in model
		weighting = weighting_function(target, x_max)

		train_loss, _ = sess.run([model.cost, model.minimize],
					{
						model.i_word_idx:i_word_idx, 
						model.k_word_idx:k_word_idx, 
						model.target:target, 
						model.weighting:weighting,
						model.lr:lr 
					}
				)
		loss += train_loss
		
	return loss/len(dataset)



def run(model, dataset, x_max, lr, restore=0):

	if not os.path.exists(tensorflow_saver_path):
		print("create save directory")
		os.makedirs(tensorflow_saver_path)


	for epoch in range(restore+1, 20000+1):
		train_loss = train(model, dataset, x_max, lr)

		print("epoch:", epoch, 'train_loss:', train_loss, '\n')

		if (epoch) % 10 == 0:
			model.saver.save(sess, tensorflow_saver_path+str(epoch)+".ckpt")
		


time_depth = 2
cell_num = 4 # 4096
voca_size = 10
embedding_size = 3 # 512 == projection size 
stack = 2 
lr = 0.1
word_embedding = None
embedding_mode = 'char' # 'word' or 'char'
pad_idx = 0
window_size = [2,3,4] # for charCNN
filters = [3,4,5] # for charCNN  np.sum(filters) = 2048

sess = tf.Session()
	def __init__(self, sess, time_depth, cell_num, voca_size, embedding_size, stack, lr, word_embedding=None, 
					embedding_mode='char', pad_idx=0, window_size=[2,3,4], filters=[3,4,5]):
model = biLM.biLM(
			sess = sess, 
			time_depth = time_depth, 
			cell_num = cell_num, 
			voca_size = voca_size, 
			embedding_size = embedding_size, 
			stack = stack, 
			lr = lr, 
			word_embedding = word_embedding,
			embedding_mode = embedding_mode, 
			pad_idx = pad_idx,
			window_size = window_size,
			filters = filters
		)

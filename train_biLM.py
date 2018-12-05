import biLM
import preprocess as pr
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

data_path = './text8'
data_savepath = './npy/'
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
		



train_data_path = './PTB_dataset/ptb.train.txt'
valid_data_path = './PTB_dataset/ptb.valid.txt'
test_data_path = './PTB_dataset/ptb.test.txt'

data_util = pr.preprocess()
# data_util.get_vocabulary(data_path, top_voca=None, char_voca=True, save_path=data_savepath)
char2idx = data_util.load_data(data_savepath+'char2idx.npy', data_structure='dictionary')
idx2char = data_util.load_data(data_savepath+'idx2char.npy', data_structure='dictionary')
word2idx = data_util.load_data(data_savepath+'word2idx.npy', data_structure='dictionary')
idx2word = data_util.load_data(data_savepath+'idx2word.npy', data_structure='dictionary')
'''
print('char2idx', len(char2idx))
print('idx2char', len(idx2char))
print('word2idx', len(word2idx))
print('idx2word', len(idx2word))
'''
for i in range(10):
	print(i, idx2word[i])

'''
data_util.make_char_idx_dataset_csv(train_data_path, voca_path=data_savepath, save_path=data_savepath+'train.csv', time_step=35, word_length=65)
data_util.make_char_idx_dataset_csv(valid_data_path, voca_path=data_savepath, save_path=data_savepath+'valid.csv', time_step=35, word_length=65)
data_util.make_char_idx_dataset_csv(test_data_path, voca_path=data_savepath, save_path=data_savepath+'test.csv', time_step=35, word_length=65)
# data: [N, (time_step+1) * (word_length)] 임. # [N, time_step+1, word_length] 로 reshape하고, data[:, :-1, :] 는 input, data[:, 1:, :] 는 target.
'''

data_util.maximum_word(train_data_path)
data_util.maximum_word(valid_data_path)
data_util.maximum_word(test_data_path)


#data_util.get_idx_dataset(data_path, char_voca=True, savepath=None)
#data_savepath





'''
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
'''

'''
import GloVe
import matrix_utils
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

from sklearn.manifold import TSNE #pip install scipy, scikit-learn
import matplotlib.pyplot as plt #pip install matplotlib

data_path = './text8/text8'
savepath = './npy/'
tensorflow_saver_path = './saver/'
drawpath = './image/'

top_voca = 50000
sub_voca_len = 10000
window_size = 10
embedding_size = 300
x_max = 100
lr = 0.05

def draw_most_word_pyplot(model, idx2word, most, picture_name):
	most_word = [idx2word[i] for i in range(most)]
	
	most_i_word_embedding = sess.run(model.i_word_embedding_table[:most])
	most_k_word_embedding = sess.run(model.k_word_embedding_table[:most]) 
	most_word_embedding = most_i_word_embedding + most_k_word_embedding
	
	plt.figure(figsize=(18,18))
	tsne = TSNE(perplexity = 30, n_components = 2, init='pca')
	low_dim_embed = tsne.fit_transform(most_word_embedding)

	for i, label in enumerate(most_word):
		x, y = low_dim_embed[i]
		plt.scatter(x,y)
		plt.annotate(label, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')

	plt.savefig(picture_name)
	plt.close()


def weighting_function(data, x_max):
	# data: [N, 1]

	# if x < x_max
	weighting = data.copy()
	weighting[data<x_max] = (data[data<x_max]/x_max)**(3/4)
	# else
	weighting[data>=x_max] = 1.0

	return weighting


def train(model, dataset, x_max, lr):
	batch_size = 512
	loss = 0

	np.random.shuffle(dataset)

	for i in tqdm(range( int(np.ceil(len(dataset)/batch_size)) ), ncols=50):
		batch = dataset[batch_size * i: batch_size * (i + 1)] # [batch_size, 3]
		batch = np.array(batch)

		i_word_idx = batch[:, 0:1].astype(np.int32) # [batch_size, 1]
		k_word_idx = batch[:, 1:2].astype(np.int32) # [batch_size, 1] 
		target = batch[:, 2:] # [batch_size, 1] # will be applied log in model
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



def run(model, dataset, x_max, lr, idx2word, restore=0):

	if not os.path.exists(tensorflow_saver_path):
		print("create save directory")
		os.makedirs(tensorflow_saver_path)

	if not os.path.exists(drawpath):
		print("create draw directory")
		os.makedirs(drawpath)

	for epoch in range(restore+1, 140+1):
		train_loss = train(model, dataset, x_max, lr)
		print("epoch:", epoch, 'train_loss:', train_loss, '\n')

		if (epoch) % 5 == 0:
			draw_most_word_pyplot(model, idx2word, most=500, picture_name=drawpath+str(epoch))
			model.saver.save(sess, tensorflow_saver_path+str(epoch)+".ckpt")
		


sess = tf.Session()

matrix_utils = matrix_utils.matrix_utils()
model = GloVe.GloVe(
			sess = sess, 
			voca_size = top_voca, 
			embedding_size = embedding_size
		)


# 이미 계산한 결과가 있으면 불러옴.
if os.path.exists(savepath+'total_data_set.npy') and os.path.exists(savepath+'word2idx.npy') and os.path.exists(savepath+'idx2word.npy'):
	#word2idx = matrix_utils.load_data(savepath+'word2idx.npy', data_structure ='dictionary')
	idx2word = matrix_utils.load_data(savepath+'idx2word.npy', data_structure ='dictionary')
	dataset = matrix_utils.load_data(savepath+'matrix.npy')

else:
	dataset = matrix_utils.get_large_voca_matrix(
				data_path=data_path, 
				top_voca=50000, 
				window_size=10, 
				sub_voca_len=5000*2,  
				savepath=savepath
			) # 10000 * 50000 을 5번 해서 데이터셋 추출. # 메모리 안터짐 (8기가 램)
	#dataset = matrix_utils.get_voca_matrix(data_path, top_voca=top_voca, window_size=window_size, savepath=savepath) #메모리 터짐. (8기가 램)
	idx2word = matrix_utils.load_data(savepath+'idx2word.npy', data_structure ='dictionary')


print('\ntop_voca', top_voca)
print('window_size', window_size)
print('embedding_size', embedding_size)
print('x_max', x_max)
print('lr', lr)
print('dataset', dataset.shape)	# 50000 word: [42.486.604, 3],   10000 word: [20.943.764, 3]

run(model, dataset, x_max=x_max, idx2word=idx2word, lr=lr) 



'''

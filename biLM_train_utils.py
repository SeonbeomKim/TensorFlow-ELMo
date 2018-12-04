# https://nlp.stanford.edu/pubs/glove.pdf
import numpy as np
import collections
import os

class biLM_train_utils:
	def __init__(self):
		pass

	def get_vocabulary(self, data_path, top_voca=50000, savepath=None):
		with open(data_path, 'r') as f:
			word = (f.readline().split())	#text8은 하나의 줄이며 단어마다 띄어쓰기로 구분.

		word2idx = {}
		idx2word = {}

		table = collections.Counter(word).most_common(top_voca-1) #빈도수 상위 x-1개 뽑음. 튜플형태로 정렬되어있음 [("단어", 빈도수),("단어",빈도수)] 
		for index, data in enumerate(table):
			word2idx[data[0]] = index
			idx2word[index] = data[0]

		if savepath is not None:
			if not os.path.exists(savepath):
				print("create save directory")
				os.makedirs(savepath)
			self.save_data(savepath+'word2idx.npy', word2idx)
			print("word2idx save", savepath+'word2idx.npy')
			self.save_data(savepath+'idx2word.npy', word2idx)
			print("idx2word save", savepath+'idx2word.npy')

		return word2idx, idx2word


	def make_dataset(self, data_path, sequence_length, savepath=None):
		if os.path.exists(savepath+'word2idx.npy') and os.path.exists(savepath+'idx2word.npy'):
			word2idx = self.load_data(savepath+'word2idx.npy', data_structure ='dictionary')
			idx2word = self.load_data(savepath+'idx2word.npy', data_structure ='dictionary')
		else:
			word2idx, idx2word = self.get_vocabulary(data_path, top_voca, savepath)

		with open(data_path, 'r') as f:
			word = (f.readline().split())	#text8은 하나의 줄이며 단어마다 띄어쓰기로 구분.

		

	def load_data(self, path, data_structure = None):
		if data_structure == 'dictionary': 
			data = np.load(path, encoding='bytes').item()
		else:
			data = np.load(path, encoding='bytes')
		return data


	def save_data(self, path, data):
		np.save(path, data)
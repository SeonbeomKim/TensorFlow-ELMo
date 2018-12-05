# https://nlp.stanford.edu/pubs/glove.pdf
import numpy as np
import collections
import os
import csv

class preprocess:
	def __init__(self):
		pass

	def get_vocabulary(self, data_path, top_voca=50000, char_voca=True, save_path=None):
		word_counter = collections.Counter({})
		if char_voca is True:
			char_counter = collections.Counter({})

		with open(data_path, 'r', newline='') as f:
			wr = csv.reader(f)
			for sentence in wr:
				sentence = sentence[0].split()
				word_counter += collections.Counter(sentence)

				if char_voca is True:
					for char in sentence:
						char_counter += collections.Counter(char)

		#빈도수 상위 top_voca개 뽑음. 튜플형태로 정렬되어있음 [("단어", 빈도수),("단어",빈도수)] 	
		word_counter = word_counter.most_common(top_voca) # => top_voca is None이면 전부 다.
		word2idx = {'</p>':0, '</g>':1, '</e>':2} # pad go eos
		idx2word = {0:'</p>', 1:'</g>', 2:'</e>'} # pad go eos

		for index, word in enumerate(word_counter):
			word2idx[word[0]] = index+3
			idx2word[index+3] = word[0]

		if char_voca is True:
			char_counter = char_counter.most_common(None)
			char2idx = {'</p>':0} # pad
			idx2char = {0:'</p>'} # pad
			
			for index, char in enumerate(char_counter):
				char2idx[char[0]] = index+1
				idx2char[index+1] = char[0]

		
		if save_path is not None:
			if not os.path.exists(save_path):
				print("create save directory")
				os.makedirs(save_path)
			
			self.save_data(save_path+'word2idx.npy', word2idx)
			print("word2idx save", save_path+'word2idx.npy', len(word2idx))
			self.save_data(save_path+'idx2word.npy', idx2word)
			print("idx2word save", save_path+'idx2word.npy', len(idx2word))

			if char_voca is True:
				self.save_data(save_path+'char2idx.npy', char2idx)
				print("char2idx save", save_path+'char2idx.npy', len(char2idx))
				self.save_data(save_path+'idx2char.npy', idx2char)
				print("idx2char save", save_path+'idx2char.npy', len(idx2char))				
			
		
		if char_voca is True:
			return word2idx, idx2word, char2idx, idx2char	

		return word2idx, idx2word
		

	def make_char_idx_dataset_csv(self, data_path, voca_path=None, save_path=None, time_step=35, word_length=65):
		if voca_path is not None:
			if os.path.exists(voca_path+'word2idx.npy') and os.path.exists(voca_path+'idx2word.npy') and os.path.exists(voca_path+'char2idx.npy') and os.path.exists(voca_path+'idx2char.npy'):
				char2idx = self.load_data(voca_path+'char2idx.npy', data_structure='dictionary')
				idx2char = self.load_data(voca_path+'idx2char.npy', data_structure='dictionary')
				word2idx = self.load_data(voca_path+'word2idx.npy', data_structure='dictionary')
				idx2word = self.load_data(voca_path+'idx2word.npy', data_structure='dictionary')				
			else:
				word2idx, idx2word, char2idx, idx2char = self.get_vocabulary(data_path, top_voca=None, char_voca=True, save_path=save_path)
		else:
			word2idx, idx2word, char2idx, idx2char = self.get_vocabulary(data_path, top_voca=None, char_voca=True, save_path=save_path)

		'''
		print('char2idx', len(char2idx))
		print('idx2char', len(idx2char))
		print('word2idx', len(word2idx))
		print('idx2word', len(idx2word))
		'''
		
		o = open(save_path, 'w', newline='')
		wt = csv.writer(o)

		with open(data_path, 'r', newline='') as f:
			wr = csv.reader(f)

			for sentence in wr:
				# append '</g>' and '</e>' to sentence and padding('</p>')
				sentence = ['</g>'] + sentence[0].split() + ['</e>']

				
				for start in range(len(sentence)):
					'''
					TODO
					# 포문돌면서 time_Step+1만큼 뽑고 그 길이가 time_step보다 크면 바로 반영, 작으면 패딩하고 포문 종료.
					'''



				if len(sentence) > time_step:
					sentence = sentence[:time_step+1] # target은 [1:],  input은 [0:-1] 로끊어써야하므로 1개 더 뽑음.
				else:
					sentence = np.pad(sentence, (0, time_step+1-len(sentence)), 'constant', constant_values='</p>') # time_step이 N이면 data는 N+1개 만들어야 함.
				

				# make target(word) idx
				target_list = []
				for word in sentence[1:]:
					if word in word2idx:
						target_list.append(word2idx[word])
					else:
						target_list.append(word2idx['<unk>']) # shape: [time_step]


				# make input(char) idx
				input_list = []	
				for word in sentence[:-1]:
					if word == '</g>' or word == '</e>' or word == '</p>' or word == '<unk>' or word == 'N':
						#print(word, char2idx['</p>'])
						char_list = np.repeat([char2idx['</p>']], word_length)
						input_list.append(char_list)
					else:
						char_list = []
						word2char = list(word)
						for char in word2char:
							char_list.append(char2idx[char])
						char_list = np.pad(char_list, (0, word_length-len(char_list)), 'constant', constant_values=char2idx['</p>'])
						input_list.append(char_list)
				
				# input_list: [time_step, word_length]
				input_list = np.reshape(input_list, [-1]) # input_list: [(time_step)*word_length]
				input_target_concat = np.concatenate((input_list, target_list),axis=0)
				
				wt.writerow(input_target_concat)	

		o.close()
		print('ok', save_path)


	def maximum_word(self, data_path):
		maximum = 0
		with open(data_path, 'r', newline='') as f:
			wr = csv.reader(f)

			for index, sentence in enumerate(wr):
				# append '</g>' and '</e>' to sentence and padding('</p>')
				sentence = ['</g>'] + sentence[0].split() + ['</e>']
				print(index, len(sentence))
				maximum = max(maximum, len(sentence))
			print(data_path, maximum)



	def save_data(self, path, data):
		np.save(path, data)


	def load_data(self, path, data_structure = None):
		if data_structure == 'dictionary': 
			data = np.load(path, encoding='bytes').item()
		else:
			data = np.load(path, encoding='bytes')
		return data

""" 백업.
	def make_char_idx_dataset_csv(self, data_path, voca_path=None, save_path=None, time_step=35, word_length=65):
		if voca_path is not None:
			if os.path.exists(voca_path+'word2idx.npy') and os.path.exists(voca_path+'idx2word.npy') and os.path.exists(voca_path+'char2idx.npy') and os.path.exists(voca_path+'idx2char.npy'):
				char2idx = self.load_data(voca_path+'char2idx.npy', data_structure='dictionary')
				idx2char = self.load_data(voca_path+'idx2char.npy', data_structure='dictionary')
				word2idx = self.load_data(voca_path+'word2idx.npy', data_structure='dictionary')
				idx2word = self.load_data(voca_path+'idx2word.npy', data_structure='dictionary')				
			else:
				word2idx, idx2word, char2idx, idx2char = self.get_vocabulary(data_path, top_voca=None, char_voca=True, save_path=save_path)
		else:
			word2idx, idx2word, char2idx, idx2char = self.get_vocabulary(data_path, top_voca=None, char_voca=True, save_path=save_path)

		'''
		print('char2idx', len(char2idx))
		print('idx2char', len(idx2char))
		print('word2idx', len(word2idx))
		print('idx2word', len(idx2word))
		'''
		
		o = open(save_path, 'w', newline='')
		wt = csv.writer(o)

		with open(data_path, 'r', newline='') as f:
			wr = csv.reader(f)

			for sentence in wr:
				# append '</g>' and '</e>' to sentence and padding('</p>')
				sentence = ['</g>'] + sentence[0].split() + ['</e>']
				if len(sentence) > time_step:
					sentence = sentence[:time_step+1] # target은 [1:],  input은 [0:-1] 로끊어써야하므로 1개 더 뽑음.
				else:
					sentence = np.pad(sentence, (0, time_step+1-len(sentence)), 'constant', constant_values='</p>') # time_step이 N이면 data는 N+1개 만들어야 함.
				

				# make target(word) idx
				target_list = []
				for word in sentence[1:]:
					if word in word2idx:
						target_list.append(word2idx[word])
					else:
						target_list.append(word2idx['<unk>']) # shape: [time_step]


				# make input(char) idx
				input_list = []	
				for word in sentence[:-1]:
					if word == '</g>' or word == '</e>' or word == '</p>' or word == '<unk>' or word == 'N':
						#print(word, char2idx['</p>'])
						char_list = np.repeat([char2idx['</p>']], word_length)
						input_list.append(char_list)
					else:
						char_list = []
						word2char = list(word)
						for char in word2char:
							char_list.append(char2idx[char])
						char_list = np.pad(char_list, (0, word_length-len(char_list)), 'constant', constant_values=char2idx['</p>'])
						input_list.append(char_list)
				
				# input_list: [time_step, word_length]
				input_list = np.reshape(input_list, [-1]) # input_list: [(time_step)*word_length]
				input_target_concat = np.concatenate((input_list, target_list),axis=0)
				
				wt.writerow(input_target_concat)	

		o.close()
		print('ok', save_path)


"""
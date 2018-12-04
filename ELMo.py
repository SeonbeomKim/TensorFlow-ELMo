import tensorflow as tf

tf.set_random_seed(777)

class ELMo:
	def __init__(self, sess, time_depth, cell_num, voca_size, embedding_size, stack, lr, word_embedding=None, 
					embedding_mode='char', pad_idx=0, window_size=[2,3,4], filters=[3,4,5]):
		self.sess = sess
		self.time_depth = time_depth
		self.cell_num = cell_num
		self.voca_size = voca_size # 'a' 't' 'g' 'c' 4개
		self.embedding_size = embedding_size
		self.stack = stack
		self.lr = lr
		self.word_embedding = word_embedding
		self.embedding_mode = embedding_mode
		self.pad_idx = pad_idx
		self.window_size = window_size
		self.filters = filters

		with tf.name_scope("placeholder"):
			if embedding_mode is 'char':
				self.data = tf.placeholder(tf.int32, [None, None, None], name="x") # [N, word, char]
			elif embedding_mode is 'word':
				self.data = tf.placeholder(tf.int32, [None, None], name="x") # [N, word]
			self.target = tf.placeholder(tf.float32, [None, None], name="target") 
			self.keep_prob = tf.placeholder(tf.float32, name="keep_prob") # for dropout

		with tf.name_scope("embedding"):		
			if self.word_embedding is None:
				self.embedding_table = self.make_embadding_table(pad_idx=self.pad_idx)
				self.word_embedding = self.embedding_func(mode=embedding_mode)

		#self.biLM_embedding = self.biLM(self.test, stack=self.stack) # [N, self.time_depth, self.embedding_size] * (self.stack+1)
		#self.elmo_embedding = self._ELMo(self.biLM_embedding) # [N, self.time_depth, self.embeding_size]

		'''
		with tf.name_scope('nucleotide_embedding'):
			embedding_table = tf.Variable(tf.random_normal([self.voca_size, self.embedding_size]))  # 'a' 't' 'g' 'c' 순서
			embedding = tf.nn.embedding_lookup(embedding_table, tf.cast(self.x_reshape[:, :, 0], tf.int32)) # [N, self.time_depth, self.embedding_size]
			embedding = tf.nn.dropout(embedding, keep_prob = self.keep_prob)
			
			self.data = tf.concat((embedding, self.x_reshape[:, :, 1:]), axis=-1) # [N, self.time_dpeth, self.embedding_size + self.feature_num-1]

		with tf.name_scope("bidirectional_lstm_with_layernorm_and_dropout"):
			# https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LayerNormBasicLSTMCell
			self.val, self.state = self.stacked_bidirectional_lstm(data=self.data, stack=self.stack, residual_connection=True)
 			# self.val: [N, self.time_depth, self.cell_num * 2]
 			# self.state: [N, self.cell_num * 2]


		with tf.name_scope("fullyconnected_layer"):
			layer1 = tf.layers.dense(self.state, units = self.cell_num*2, activation=None) # [N, self.cell_num*2]
			# Layer Norm
			layer1_ln = tf.contrib.layers.layer_norm(layer1, begin_norm_axis=-1)
			layer1_relu = tf.nn.relu(layer1_ln)
			# drop out
			layer1_dropout = tf.nn.dropout(layer1_relu, keep_prob = self.keep_prob)

			self.pred = tf.layers.dense(layer1_dropout, units = self.num_classes) # [N, self.num_classes]


		with tf.name_scope('train'): 
			# calc train_cost
			self.train_cost = tf.reduce_mean(
						tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.pred)
					) # softmax_cross_entropy_with_logits: [N] => reduce_mean: scalar
			optimizer = tf.train.AdamOptimizer(self.lr)
			self.minimize = optimizer.minimize(self.train_cost)


		with tf.name_scope('metric'):
			self.correct_check = tf.reduce_sum(tf.cast( tf.equal( tf.argmax(self.pred, 1), tf.argmax(self.target, 1) ), tf.int32 ))


		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)

		'''
		#self.ElMo_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'ELMo')
		#self.biLM_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'biLM')
		#self.ElMo_minimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list=self.ElMo_variables) 
		#self.biLM_minimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list=self.biLM_variables) 
		
		self.sess.run(tf.global_variables_initializer())


	def make_embadding_table(self, pad_idx):
		zero = tf.zeros([1, self.embedding_size], dtype=tf.float32) # for padding
		embedding_table = tf.Variable(tf.random_normal([self.voca_size-1, self.embedding_size])) 
		front, end = tf.split(embedding_table, [pad_idx, -1])
		embedding_table = tf.concat((front, zero, end), axis=0)
		return embedding_table


	def convolution(self, embedding, embedding_size, window_size, filters):
		convolved_features = []
		for i in range(len(window_size)):
			convolved = tf.layers.conv2d(
						inputs = embedding, 
						filters = filters[i], 
						kernel_size = [window_size[i], embedding_size], 
						strides=[1, 1], 
						padding='VALID', 
						activation=tf.nn.relu
					) # [N, ?, 1, filters]
			convolved_features.append(convolved) # [N, ?, 1, filters] 이 len(window_size) 만큼 존재.
		return convolved_features


	def max_pooling(self, convolved_features):
		pooled_features = []
		for convolved in convolved_features: # [N, ?, 1, self.filters]
			max_pool = tf.reduce_max(
						input_tensor = convolved,
						axis = 1,
						keep_dims = True
					) # [N, 1, 1, self.filters]
			pooled_features.append(max_pool) # [N, 1, 1, self.filters] 이 len(window_size) 만큼 존재.
		return pooled_features


	def charCNN(self, window_size, filters):
		len_word = tf.shape(self.data)[1] # word length

		embedding = tf.nn.embedding_lookup(self.embedding_table, self.data) # [N, word, char, self.embedding_size] 
		embedding = tf.reshape(embedding, [-1, tf.shape(embedding)[2], self.embedding_size]) # [N*word, char, self.embedding_size]
			# => convolution 적용하기 위해서 word는 batch화 시킴. 동일하게 적용되도록.
		embedding = tf.expand_dims(embedding, axis=-1) # [N*word, char, self.embedding, 1]
			# => convolution을 위해 channel 추가.

		convolved_embedding = self.convolution(embedding, self.embedding_size, window_size, filters)
			# => [N*word, ?, 1, filters] 이 len(window_size) 만큼 존재.
		max_pooled_embedding = self.max_pooling(convolved_features=convolved_embedding)
			# => [N*word, 1, 1, filters] 이 len(window_size) 만큼 존재. 
		embedding = tf.concat(max_pooled_embedding, axis=-1) # [N*word, 1, 1, sum(filters)]
			# => filter 기준으로 concat
		embedding = tf.reshape(embedding, [-1, len_word, np.sum(filters)]) # [N, word, sum(filters)]
		return embedding		


	def highway_network(self, embedding):
		# embedding: [N, word, sum(filters)]
		transform_gate = tf.layers.dense(embedding, units=np.sum(self.filters), activation=tf.nn.sigmoid) # [N, word, sum(filters)]
		carry_gate = 1-transform_gate # [N, word, sum(*filters)]
		block_state = tf.layers.dense(embedding, units=np.sum(self.filters), activation=tf.nn.relu)
		highway = transform_gate * block_state + carry_gate * embedding # [N, word, sum(filters)]
			# if transfor_gate is 1. then carry_gate is 0. so only use block_state
			# if transfor_gate is 0. then carry_gate is 1. so only use embedding
			# if transfor_gate is 0.@@. then carry_gate is 0.@@. so use sum of scaled block_state and embedding
		return highway


	def embedding_func(self, mode=None):
		if mode is 'word':
			embedding = tf.nn.embedding_lookup(self.embedding_table, self.data) # [N, word, self.embedding_size]
			return embedding

		elif mode is 'char':
			embedding = self.charCNN(window_size=self.window_size, filters=self.filters) # [N, word, filters*len(window_size)]
			embedding = self.highway_network(embedding=embedding) # [N, word, filters*len(window_size)]
			return embedding


		'''
		data = [[[3, 3, 3, 2, 3, 1, 3, 1, 3], [0, 3, 1, 2, 2, 0, 3, 2, 1], [3, 3, 3, 2, 3, 1, 3, 1, 3]],
	 		[[3, 3, 3, 2, 2, 2, 2, 3, 1], [0, 0, 2, 2, 2, 0, 3, 3, 0], [3, 3, 3, 2, 3, 1, 3, 1, 3]]]
	 	data를 self.data에 feed해서 self.charCNN 돌려보면 같은 char로 이뤄진 단어는 같은 embedding을 갖는것을 확인할 수 있음.
		'''
		# char cnn(kim yoon) => highway layers => linear projection down to a 512
		# or
		# word embedding
		# or
		# pre trained word embedding: glove or word2vec
		pass

	def biLM(self, data, stack):
		# data [N, self.time_depth, self.embedding_size(512)]
		# stack:2 
		# cell_num: 4096
		# hiddenlayer를 512차원으로 줄여줄 projection 적용하고, residual connection 연결한다.
 			# 이 512차원이 입력단어의 벡터가 됨.
 		# 양방향은 파라미터 전부 공유(softmax하는 layer도 포함.)

		with tf.variable_scope('biLM') as scope:

			concat_layer_val = [data] # x_data
			for i in range(stack):

				# https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LayerNormBasicLSTMCell
				cell = tf.contrib.rnn.LSTMCell(self.cell_num)
				
				if i == 0:
					fw_input = concat_layer_val[i]
					bw_input = tf.reverse(fw_input, axis=[1])
				else:
					fw_input = concat_layer_val[i]+concat_layer_val[i-1]
					bw_input = tf.reverse(fw_input, axis=[1])

				# fw_bw_val: shape: [N, self.time_depth, self.cell_num]
				fw_val, _ = tf.nn.dynamic_rnn(cell, fw_input, dtype=tf.float32, scope='stack_fw'+str(i))				
				bw_val, _ = tf.nn.dynamic_rnn(cell, bw_input, dtype=tf.float32, scope='stack_bw'+str(i))

				# linear projection, shape: [N, self.time_depth, self.embedding_size//2]
				fw_val = tf.layers.dense(fw_val, units=self.embedding_size//2, activation=None, name='linear'+str(i))
				bw_val = tf.layers.dense(bw_val, units=self.embedding_size//2, activation=None, name='linear'+str(i), reuse=True)
				
				# concat fw||bw
				reverse_bw_val = tf.reverse(bw_val, axis=[1]) # 처음에 뒤집어서 넣었으므로 다시 뒤집어줌.
				concat_val = tf.concat((fw_val, reverse_bw_val), axis=-1) # [N, self.time_depth, self.embedding_size]

				# save current layer state for residual connection and ELMo
				concat_layer_val.append(concat_val)

			return concat_layer_val
				
	def _ELMo(self, concat_layer_val):
		# concat_layer_val: [N, self.time_depth, self.embedding_size] * (self.stack+1)
		
		with tf.variable_scope('ELMo') as scope:
			s_task = tf.Variable(tf.constant(value=0.0, shape=[self.stack+1])) # [self.stack+1] include x_data
			s_task = tf.nn.softmax(s_task) # [self.stack+1]
			gamma_task = tf.Variable(tf.constant(value=1.0))

			softmax_norm = []
			for i in range(self.stack+1):
				# paper3.2: apply layer normalization to each biLM layer before weighting
				if i == 0: # biLM은 아니고 embedding이므로 LN 안씀. 
					softmax_norm.append(s_task[i] * concat_layer_val[i])
				else: 
					softmax_norm.append(s_task[i] * tf.contrib.layers.layer_norm(concat_layer_val[i], begin_norm_axis=2))
				#softmax_norm.append(s_task[i] * concat_layer_val[i])

			ELMo_vector = gamma_task * tf.reduce_sum(softmax_norm, axis=0) # [N, self.time_depth, self.embedding_size]

			return ELMo_vector # [N, self.time_depth, self.embedding_size]


sess = tf.Session()
time_depth = 2
cell_num = 4
voca_size = 10
embedding_size = 3
stack = 2
lr = 0.1
embedding_mode = 'char'
pad_idx = 0
		

import numpy as np
np.random.seed(777)

#data = np.random.randint(0,4, [2, 3, 9])

data = [[[3, 3, 3, 2, 3, 1, 3, 1, 3], [0, 3, 1, 2, 2, 0, 3, 2, 1], [3, 3, 3, 2, 3, 1, 3, 1, 3]],
		 [[3, 3, 3, 2, 2, 2, 2, 3, 1], [0, 0, 2, 2, 2, 0, 3, 3, 0], [3, 3, 3, 2, 3, 1, 3, 1, 3]]]
model = ELMo(sess, time_depth, cell_num, voca_size, embedding_size, stack, lr, embedding_mode=embedding_mode, pad_idx=pad_idx)
a = sess.run(model.word_embedding, {model.data:data})
print(data, '\n')
print(a, '\n',a.shape)
#for i in a:
#	print(i.shape)


'''
data = np.random.randn(2,time_depth,embedding_size)
zero = np.zeros((2,time_depth,embedding_size), dtype=np.float32)
print(data.shape)

bi, elmo = sess.run([model.biLM_embedding, model.elmo_embedding], {model.test:data})

for i in bi:
	print(i/3,'\n')
	zero += i/3
print('zero\n',zero, '\n')

print('elmo\n',elmo)
'''
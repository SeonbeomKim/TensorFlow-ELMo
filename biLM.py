import tensorflow as tf

tf.set_random_seed(777)

class biLM:
	def __init__(self, sess, time_depth, cell_num, voca_size, target_size, embedding_size, stack, word_embedding=None, 
					embedding_mode='char', pad_idx=0, window_size=[2,3,4], filters=[3,4,5]):
		self.sess = sess
		self.time_depth = time_depth
		self.cell_num = cell_num # 4096
		self.voca_size = voca_size # 'word': 단어 개수, 'char': char 개수
		self.target_size = target_size
		self.embedding_size = embedding_size # 512 == projection size 
		self.stack = stack # biLM stack size
		self.word_embedding = word_embedding # when use pre-trained word_embedding like GloVe, word2vec
		self.embedding_mode = embedding_mode # 'word' or 'char'
		self.pad_idx = pad_idx # 0
		self.window_size = window_size # for charCNN
		self.filters = filters # for charCNN  np.sum(filters) = 2048
		self.initializer = tf.initializers.random_uniform(-0.05, 0.05)

		# lstm_char_cnn 짯던 코드 이용하자.
		# cnn, highway, projection는 파라미터 공유, lstm은 공유하지않음. 
		# fw, bw 별로 softmax_cross_entropy_with_logits 구하고 두 loss의 평균을 minimize

		# data: [N*time_depth, word_length]
		'''
		input: s a b c /s    원래 word_length: 3,  token_Word_length: 5(s, /s 포함)
			
			forward_target: a b c /s
			forward_input:  s a b c

			backward_target: c b a s
			backward_input: /s c b a
		'''


		with tf.name_scope("placeholder"):
			'''
				input: s a b c /s    원래 word_length: 3,  token_Word_length: 5(s, /s 포함)
				
				forward_target: a b c /s
				forward_input:  s a b c

				backward_target: c b a s
				backward_input: /s c b a
			'''


			# data: [N*time_depth, word_length], 이 형식으로 넣어줘야 나중에 char embedding만 추출 가능(data: [N, word_length])
			self.data = tf.placeholder(tf.int32, [None, self.word_length], name="char_x") # char_idx   form: s a b c /s 
			self.target = tf.placeholder(tf.int32, [None, None], name="target") # word_idx  form: s a b c /s
			self.lr = tf.placeholder(tf.float32, name="lr") # lr
			self.keep_prob = tf.placeholder(tf.float32, name="keep_prob") # keep_prob
			self.sentence_length = tf.placeholder(tf.int32, name='sentence_length')

		with tf.name_scope('split_fw_bw'):
			reshape_data = tf.reshape(self.data, [-1, self.sentence_length, self.word_length]) # [N, self.sentence_length, self.word_length]
			fw_data = reshape_data[:, :-1, :] # s a b c /s => s a b c
			fw_data = tf.reshape(fw_data, [-1, self.word_length]) # [N*self.sentence_length-1, self.word_length]
			fw_target = self.target[:, 1:] # s a b c /s => a b c /s ,  [N, self.sentence_length-1]

			bw_data = reshape_data[:, 1:, :] # s a b c /s => a b c /s
			bw_data = tf.reverse(bw_data, axis=[1]) # a b c /s => /s c b a
			bw_data = tf.reshape(bw_data, [-1, self.word_length]) # [N*self.sentence_length-1, self.word_length]
			bw_target = self.target[:, :-1] # s a b c /s => s a b c
			bw_target = tf.reverse(bw_target, axis=[1]) # s a b c => c b a s ,  [N, self.sentence_length-1]


		with tf.name_scope("biLM"):
			self.embedding_table = self.make_embadding_table(pad_idx=self.pad_idx)

			# charCNN
			fw_char_embedding = self.charCNN(
					data=fw_data, 
					window_size=self.window_size, 
					filters=self.filters,
					name='fw_bw_share'
				) # [N*self.sentence_length-1, sum(filters)]
			bw_char_embedding = self.charCNN(
					data=bw_data, 
					window_size=self.window_size, 
					filters=self.filters,
					name='fw_bw_share'
				) # [N*self.sentence_length-1, sum(filters)]

			# highway layer
			for i in range(self.highway_stack):
				fw_char_embedding = self.highway_network(
						embedding=fw_char_embedding, 
						units=np.sum(filters),
						name=str(i)
					) # [N*self.sentence_length-1, sum(filters)]
				bw_char_embedding = self.highway_network(
						embedding=bw_char_embedding, 
						units=np.sum(filters),
						name=str(i)
					) # [N*self.sentence_length-1, sum(filters)]

			# projection layer
			fw_char_embedding = tf.layers.dense(
					fw_char_embedding, 
					units=self.embedding_size, 
					activation=None, 
					name='projection'
				) # [N*self.sentence_length-1, self.embedding_size]
			bw_char_embedding = tf.layers.dense(
					bw_char_embedding, 
					units=self.embedding_size, 
					activation=None, 
					name='projection', 
					reuse=True # reuse same name weight
				) # [N*self.sentence_length-1, self.embedding_size] 

			# LSTM layer
			self.fw_char_embedding = self.LSTM(
					tf.reshape(fw_char_embedding, [-1, self.sentence_length-1, self.embedding_size]), 
					stack=self.stack, 
					name='fw'
				) # [N, self.sentence_length-1, self.embedding_size] * (self.stack+1)
			self.bw_char_embedding = self.LSTM(
					tf.reshape(bw_char_embedding, [-1, self.sentence_length-1, self.embedding_size]),
					bw_char_embedding, 
					stack=self.stack, 
					name='bw'
				) # [N, self.sentence_length-1, self.embedding_size] * (self.stack+1)


		with tf.name_scope('prediction'):
			fw_pred = tf.layers.dense(
					self.fw_char_embedding[-1], 
					units=self.target_size, 
					activation=None, 
					name='softmax_layer'
				) # [N, self.sentence_length-1, self.target_size] 
			bw_pred = tf.layers.dense(
					self.bw_char_embedding[-1], 
					units=self.target_size, 
					activation=None, 
					name='softmax_layer', 
					reuse=True # reuse same name weight
				) # [N, self.sentence_length-1, self.target_size] 


		with tf.name_scope('train'): 
			fw_target_one_hot = tf.one_hot(
					fw_target, # [N, self.sentence_length-1]
					depth=self.target_size,
					on_value = 1., # tf.float32
					off_value = 0., # tf.float32
				) # [N, self.sentence_length-1, self.target_size] 
			bw_target_one_hot = tf.one_hot(
					bw_target, # [N, self.sentence_length-1]
					depth=self.target_size,
					on_value = 1., # tf.float32
					off_value = 0., # tf.float32
				) # [N, self.sentence_length-1, self.target_size] 
		
			# calc train_cost
			fw_cost = tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits(labels=fw_target_one_hot, logits=fw_pred)
				) # softmax_cross_entropy_with_logits: [N, self.sentence_length-1] => reduce_mean: scalar
			bw_cost = tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits(labels=bw_target_one_hot, logits=bw_pred)
				) # softmax_cross_entropy_with_logits: [N, self.sentence_length-1] => reduce_mean: scalar
			self.cost = (fw_cost + bw_cost)/2 # fw bw mean cost

			optimizer = tf.train.AdamOptimizer(self.lr)
			self.minimize = optimizer.minimize(self.cost)


		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)

		'''
		#self.ElMo_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'ELMo')
		#self.biLM_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'biLM')
		#self.ElMo_minimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list=self.ElMo_variables) 
		#self.biLM_minimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list=self.biLM_variables) 
		'''

		self.sess.run(tf.global_variables_initializer())


	def make_embadding_table(self, pad_idx):
		zero = tf.zeros([1, self.embedding_size], dtype=tf.float32) # for padding
		embedding_table = tf.get_variable(
				name='embedding_table', 
				shape=[self.voca_size-1, self.embedding_size],
				initializer=self.initializer
			)
		front, end = tf.split(embedding_table, [pad_idx, -1])
		embedding_table = tf.concat((front, zero, end), axis=0)
		return embedding_table


	def convolution(self, embedding, embedding_size, window_size, filters):
		# embedding: [N*time_depth, word_length, self.embedding_size, 1]
		convolved_features = []
		for i in range(len(window_size)):
			convolved = tf.layers.conv2d(
					inputs = embedding, 
					filters = filters[i], 
					kernel_size = [window_size[i], embedding_size], 
					strides=[1, 1], 
					padding='VALID', 
					activation=tf.nn.tanh,
					kernel_initializer=self.initializer,
					bias_initializer=self.initializer
				) # [N, ?, 1, filters]
			convolved_features.append(convolved) # [N*time_depth, ?, 1, filters] * len(window_size).
		return convolved_features


	def max_pooling(self, convolved_features):
		# convolved_features: [N*time_depth, ?, 1, filters] * len(window_size)
		pooled_features = []
		for convolved in convolved_features: # [N*time_depth, ?, 1, self.filters]
			max_pool = tf.reduce_max(
					input_tensor = convolved,
					axis = 1,
					keep_dims = True
				) # [N, 1, 1, self.filters]
			max_pool = tf.layers.flatten(max_pool) # [N*time_depth, self.filters[i]]
			pooled_features.append(max_pool) # [N*time_depth, self.filters[i]] * len(window_size).
		return pooled_features


	def charCNN(self, data, window_size, filters, name):
		with tf.variable_scope('charCNN_'+name, reuse=tf.AUTO_REUSE):
			embedding = tf.nn.embedding_lookup(self.embedding_table, self.data) # [N*time_depth, word_length, self.embedding_size] 
			embedding = tf.expand_dims(embedding, axis=-1) # [N*time_depth, word_length, self.embedding_size, 1]  => convolution을 위해 channel 추가.

			convolved_embedding = self.convolution(embedding, self.embedding_size, window_size, filters) # [N*time_depth, ?, 1, filters] * len(window_size).
			max_pooled_embedding = self.max_pooling(convolved_features=convolved_embedding) # [N*time_depth, self.filters[i]] * len(window_size).

			embedding = tf.concat(max_pooled_embedding, axis=-1) # [N*time_depth, sum(filters)], filter 기준으로 concat
			return embedding



	def highway_network(self, embedding, units, name):
		with tf.variable_scope('highway_'+name, reuse=tf.AUTO_REUSE):
			# embedding: [N*time_depth, sum(filters)]
			transform_gate = tf.layers.dense(
					embedding, 
					units=units, 
					activation=tf.nn.sigmoid, 
					kernel_initializer=self.initializer,
					bias_initializer=tf.constant_initializer(-2)
				) # [N*time_depth, sum(filters)]
			carry_gate = 1-transform_gate # [N*time_depth, sum(filters)]
			block_state = tf.layers.dense(
					embedding, 
					units=units, 
					activation=tf.nn.relu,
					kernel_initializer=self.initializer,
					bias_initializer=self.initializer
				) # [N*time_depth, sum(filters)]
			highway = transform_gate * block_state + carry_gate * embedding # [N*time_depth, sum(filters)]
				# if transfor_gate is 1. then carry_gate is 0. so only use block_state
				# if transfor_gate is 0. then carry_gate is 1. so only use embedding
				# if transfor_gate is 0.@@. then carry_gate is 0.@@. so use sum of scaled block_state and embedding
			return highway # [N*time_depth, sum(filters)]



	def LSTM(self, data, stack, name):
		# data: [N, self.sentence_length-1, self.embedding_size]
		# stack:2 
		# cell_num: 4096
		# hiddenlayer를 512차원으로 줄여줄 projection 적용하고, residual connection 연결한다.
 			# 이 512차원이 입력단어의 벡터가 됨.
 		# 양방향은 파라미터 전부 공유(softmax하는 layer도 포함.)

		with tf.variable_scope('lstm_'+name) as scope:
			layer_val = [data]

			for i in range(stack):
				cell = tf.contrib.rnn.LSTMCell(self.cell_num)
				
				# data: shape: [N, self.sentence_length-1, self.cell_num]
				data, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32, scope='stack_'+name+'_'+str(i))				

				# linear projection, shape: [N, self.sentence_length-1, self.embedding_Size]
				linear_data = tf.layers.dense(data, units=self.embedding_size, activation=None, name='linear_'+name+'_'+str(i))
			
				# save current layer state for residual connection and ELMo
				layer_val.append(linear_data)

				# update next layer input
				if i < stack-1:
					data = linear_data + data
					
			return layer_val
				

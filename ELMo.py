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

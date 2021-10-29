"""
auth:bzx
time:2020-2-11
NMT模型
"""
"""
注意，mask有两种，一个用于encoder，一个用于decoder
对于encoder的mask，我们需要获取输入数据的mask，[batch,lens,lens]，矩阵中为1的代表是padding，其余的为0
decoder时需要保证不能看到当前时刻i后的数据，需要新的padding
"""

"""
还差，mask，look_forward_mask，train,infer 四个函数
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import layers
from configuration import Configuration
import tensorflow.contrib as contrib
import math


class BaseModel:
    """
    基础的sewq2seq模型，在该模型中：
        1.只有encoder-decoder，
        2.decoder的初始状态为encoder的末状态
        3.训练时采用 teach forcing
    """

    def __init__(self, num_layers, source_vocab_size, target_vocab_size, max_length, embedding_dimension):
        # self.num_units = num_units
        self.num_layers = num_layers
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_length = max_length
        self.embedding_dimension = embedding_dimension

        self.sos_id = 1
        self.eos_id = 2

        self.source_input_ids = tf.placeholder(tf.int32, shape=(None, self.max_length), name="source_input_ids")
        self.source_padding_mask = tf.placeholder(tf.float32, shape=(None, max_length, max_length),
                                                  name="source_padding_mask")
        self.target_inputs_ids = tf.placeholder(tf.int32, shape=(None, self.max_length),
                                                name="target_inputs_ids")
        self.target_layer1_mask = tf.placeholder(tf.float32, shape=(None, max_length, max_length),
                                                 name="target_layer1_mask")
        self.target_layer2_mask = tf.placeholder(tf.float32, shape=(None, max_length, max_length),
                                                 name="target_layer2_mask")
        self.target_labels = tf.placeholder(tf.int32, shape=(None, max_length), name="target_labels")
        self.target_lengths = tf.placeholder(tf.int32, shape=(None,), name="target_lengths")

        self.infer_tgt_inputs = tf.placeholder(tf.int32, shape=(None, None), name="infer_tgt_inputs")
        self.infer_tgt_layer1_mask = tf.placeholder(tf.float32, shape=(None, None, None),
                                                    name="infer_tgt_layer1_mask")
        self.infer_tgt_layer2_mask = tf.placeholder(tf.float32, shape=(None, None, None), name="infer_tgt_layer2_mask")
        self.infer_tgt_shape = tf.shape(self.infer_tgt_inputs)

        self.source_embedding_matrix = tf.Variable(
            np.random.uniform(size=(self.source_vocab_size, self.embedding_dimension)), dtype=tf.float32,
            trainable=True)
        self.target_embedding_matrix = tf.Variable(
            np.random.uniform(size=(self.target_vocab_size, self.embedding_dimension)), dtype=tf.float32,
            trainable=True)
        self.source_emb = self.word_embedding(self.source_input_ids, self.source_embedding_matrix)
        self.target_emb = self.word_embedding(self.target_inputs_ids, self.target_embedding_matrix)

        self.batch_size = tf.shape(self.source_input_ids)[0]
        # self.dimension = 512

        self.source_position_emb = tf.Variable(self.position_embedding(self.max_length, self.embedding_dimension),
                                               dtype=tf.float32,
                                               trainable=False)
        self.target_position_emb = tf.Variable(
            self.position_embedding(self.max_length, self.embedding_dimension), dtype=tf.float32,
            trainable=False)

        self.encoder_emb = self.encoder_embedding()

        self.h = 8
        self.d_k = self.embedding_dimension / self.h
        self.feed_net_units = 2048
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.98, epsilon=1e-9)

    def word_embedding(self, input_data, embedding_matrix):
        emb = tf.nn.embedding_lookup(embedding_matrix, input_data)
        return emb

    def position_embedding(self, max_lens, dimension):
        position = tf.expand_dims(tf.range(start=0.0, limit=max_lens, delta=1.0), 1)
        divm = tf.exp(2 * tf.range(0.0, dimension, delta=1.0) * (-math.log(10000) / dimension))
        matrix = tf.matmul(position, tf.reshape(divm, (1, dimension)))
        pe = []
        for i in range(dimension):
            if i % 2 == 0:
                temp = tf.sin(matrix[:, i])
            else:
                temp = tf.cos(matrix[:, i])
            pe.append(temp)
        res = tf.expand_dims(tf.stack(pe, 1), 0)
        return res

    def scaled_dot_product_attention(self, Q, K, V, mask):
        with tf.variable_scope("self_attention"):
            # Q和K矩阵相乘
            mtl = tf.matmul(Q, tf.transpose(K, (0, 2, 1)))
            # 加上scale缩放
            scale = mtl / tf.sqrt(self.d_k)
            # 加上mask
            mask = mask * -1e9
            mask = tf.add(scale, mask)
            # softmax
            att = tf.nn.softmax(mask)
            # attention*V
            output = tf.matmul(att, V)
        return output, att

    def multi_head_attention(self, Q, K, V, dimision, h, mask):
        with tf.variable_scope("multi_head_attention"):
            assert dimision % h == 0
            d_k = dimision // h
            layer = [tf.layers.Dense(dimision) for _ in range(4)]
            # 在transformer中，一个multi_head 包括 8个 scaled Dot_product,也就是h
            multi_att = []
            multi_sacled_output = []
            for i in range(h):
                scale_Q = layer[0](Q)
                scale_K = layer[1](K)
                scale_V = layer[2](V[:, :, i * d_k:(i + 1) * d_k])
                temp_output, temp_att = self.scaled_dot_product_attention(scale_Q, scale_K, scale_V, mask)
                # 输出应该是[batch,max_lens,dimision/h]
                multi_att.append(temp_att)
                multi_sacled_output.append(temp_output)
            # 将输出concat起来
            out_att = tf.concat(multi_att, axis=-1)
            out_output = tf.concat(multi_sacled_output, axis=-1)
            # 将concat后的数据通过linear，最终输出
            output = layer[-1](out_output)
        return output, out_att

    # multi_head_attention的输出，需要经过一层add+norm层，然后再通过一层FFN和add_and_norm
    def add_and_norm(self, inputs, sub_inputs):
        """
        这一层的输入和上一层的输入，先加再norm
        :param inputs: 这一层的输入
        :param sub_inputs: 上一层的输入
        :return:
        """
        data = inputs + sub_inputs
        res = layers.layer_norm(data)
        return res

    def feed_forward(self, inputs, hidden_units):
        out1 = tf.layers.dense(inputs, hidden_units, activation=tf.nn.relu)
        out2 = tf.layers.dense(out1, self.embedding_dimension)
        return out2

    def lexical(self,encoder_emb,target_emb,lexical_mask):
        with tf.variable_scope("lexical",reuse=tf.AUTO_REUSE):
            out, att = self.multi_head_attention(target_emb, encoder_emb, encoder_emb, self.embedding_dimension, self.h,
                                                   mask=lexical_mask)
            temp = tf.tanh(self.add_and_norm(out, target_emb))
            ffn_2 = tf.layers.dense(inputs=temp,units=self.embedding_dimension,activation=tf.nn.tanh,use_bias=False)+temp
        return ffn_2

    def encoder_embedding(self):
        return self.source_emb + self.source_position_emb

    def decoder_embedding(self, input_ids):
        word_emb = self.word_embedding(input_ids, self.target_embedding_matrix)
        size = tf.shape(input_ids)
        pos_emb = self.target_position_emb[:, :size[1], :]
        return word_emb + pos_emb

    def encoder_cell(self, inputs, mask, keep_prob=None):
        out, att = self.multi_head_attention(inputs, inputs, inputs, self.embedding_dimension, self.h, mask=mask)
        temp = self.add_and_norm(out, inputs)
        feed_out = self.feed_forward(temp, self.feed_net_units)
        res = self.add_and_norm(feed_out, temp)
        return res

    def decoder_cell(self, encoder_output, inputs, first_layer_mask, second_layer_mask, keep_prob=None):
        out1, att1 = self.multi_head_attention(inputs, inputs, inputs, self.embedding_dimension, self.h,
                                               mask=first_layer_mask)
        temp1 = self.add_and_norm(out1, inputs)

        out2, att2 = self.multi_head_attention(temp1, encoder_output, encoder_output, self.embedding_dimension, self.h,
                                               mask=second_layer_mask)
        temp2 = self.add_and_norm(out2, temp1)

        feed_out = self.feed_forward(temp2, self.feed_net_units)
        temp3 = self.add_and_norm(feed_out, temp2)
        return temp3

    def encoder(self, encoder_input, layer_number, mask, keep_prob=None):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            temp_inputs = encoder_input
            for i in range(layer_number):
                temp = self.encoder_cell(inputs=temp_inputs, mask=mask, keep_prob=keep_prob)
                temp_inputs = temp
        return temp_inputs

    def decoder(self, encoder_outputs, inputs, layer_number, first_layer_mask, second_layer_mask, keep_prob=None):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            temp_inputs = inputs
            for i in range(layer_number):
                temp = self.decoder_cell(encoder_outputs, temp_inputs, first_layer_mask, second_layer_mask, keep_prob)
                temp_inputs = temp
        return temp_inputs

    def generate(self, decoder_out):
        with tf.variable_scope("generate", reuse=tf.AUTO_REUSE):
            # decoder的最后需要加上一层dense和softmax
            decoder_out = tf.layers.dense(decoder_out, self.target_vocab_size)
        return decoder_out

    def transformer_model_train(self, encoder_input, decoder_input, keep_prob=None):

        encoder_out = self.encoder(encoder_input,self.num_layers, self.source_padding_mask, keep_prob)
        decoder_out = self.decoder(encoder_out, decoder_input, self.num_layers,
                                   self.target_layer1_mask, self.target_layer2_mask, keep_prob)
        lexical_out = self.lexical(self.source_emb,decoder_input,lexical_mask=self.target_layer2_mask)
        generate_out = self.generate(decoder_out+lexical_out)
        return generate_out

    def loss_function(self, final_outputs):
        sequence_mask = tf.sequence_mask(lengths=self.target_lengths, maxlen=self.max_length, dtype=tf.float32)
        loss = tf.losses.sparse_softmax_cross_entropy(self.target_labels, final_outputs)
        loss = tf.reduce_mean(loss * sequence_mask)
        return loss

    def greedy_decoding(self, encoder_out):
        tgt_emb = self.decoder_embedding(self.infer_tgt_inputs)
        # target_padding_mask = tf.convert_to_tensor(padding_mask, dtype=tf.float32)
        decoder_out = self.decoder(encoder_out, tgt_emb, self.num_layers,
                                   self.infer_tgt_layer1_mask,
                                   self.infer_tgt_layer2_mask)
        lexical_out = self.lexical(self.source_emb, tgt_emb, lexical_mask=self.infer_tgt_layer2_mask)
        generate = self.generate(decoder_out+lexical_out)
        generate = tf.nn.softmax(generate)
        out_id = tf.arg_max(generate, dimension=-1)
        return out_id[:,-1]


if __name__ == "__main__":
    a = 1

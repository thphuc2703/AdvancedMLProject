#!/usr/local/bin/python
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import neural as neural
from MSINModule import MSINCell, MSIN, MSINStateTuple
import tensorflow.contrib.distributions as ds
from tensorflow.contrib.layers import batch_norm
from ConfigLoader import logger, ss_size, vocab_size, config_model, path_parser






class Model:
    def __init__(self, config_model, vocab_size, ss_size, logger, path_parser):
        logger.info('INIT: #stock: {0}, #vocab+1: {1}'.format(ss_size, vocab_size))
        
        # Model configuration
        self.mode = config_model['mode']
        self.opt = config_model['opt']
        self.lr = config_model['lr']
        self.decay_step = config_model['decay_step']
        self.decay_rate = config_model['decay_rate']
        self.momentum = config_model['momentum']
        self.kl_lambda_anneal_rate = config_model['kl_lambda_anneal_rate']
        self.kl_lambda_start_step = config_model['kl_lambda_start_step']
        self.use_constant_kl_lambda = config_model['use_constant_kl_lambda']
        self.constant_kl_lambda = config_model['constant_kl_lambda']

        self.daily_att = config_model['daily_att']
        self.alpha = config_model['alpha']
        self.clip = config_model['clip']
        self.n_epochs = config_model['n_epochs']
        self.batch_size_for_name = config_model['batch_size']

        self.max_n_days = config_model['max_n_days']
        self.max_n_msgs = config_model['max_n_msgs']
        self.max_n_words = config_model['max_n_words']

        self.weight_init = config_model['weight_init']
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.bias_initializer = tf.keras.initializers.Zeros()

        self.word_embed_type = config_model['word_embed_type']
        self.y_size = config_model['y_size']
        self.word_embed_size = config_model['word_embed_size']
        self.stock_embed_size = config_model['stock_embed_size']
        self.price_embed_size = config_model['price_embed_size']

        # More initialization...
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def _build_placeholders(self):
        with tf.name_scope('placeholder'):
            self.is_training_phase = tf.Variable(False, dtype=tf.bool, trainable=False, name='is_training_phase')
            self.batch_size = tf.Variable(0, dtype=tf.int32, trainable=False, name='batch_size')

            # Input tensors
            self.word_table_init = tf.Variable(
                tf.zeros([vocab_size, self.word_embed_size], dtype=tf.float32), trainable=False, name='word_table_init'
            )
            self.stock_ph = tf.keras.Input(shape=(), dtype=tf.int32, name="stock_placeholder")
            self.T_ph = tf.keras.Input(shape=(None,), dtype=tf.int32, name="T_placeholder")
            self.n_words_ph = tf.keras.Input(
                shape=(self.max_n_days, self.max_n_msgs), dtype=tf.int32, name="n_words_placeholder"
            )
            self.n_msgs_ph = tf.keras.Input(
                shape=(self.max_n_days,), dtype=tf.int32, name="n_msgs_placeholder"
            )
            self.y_ph = tf.keras.Input(
                shape=(self.max_n_days, self.y_size), dtype=tf.float32, name="y_placeholder"
            )
            self.mv_percent_ph = tf.keras.Input(
                shape=(self.max_n_days,), dtype=tf.int32, name="mv_percent_placeholder"
            )
            self.price_ph = tf.keras.Input(
                shape=(self.max_n_days, 3), dtype=tf.float32, name="price_placeholder"
            )
            self.word_ph = tf.keras.Input(
                shape=(self.max_n_days, self.max_n_msgs, self.max_n_words), dtype=tf.int32, name="word_placeholder"
            )
            self.ss_index_ph = tf.keras.Input(
                shape=(self.max_n_days, self.max_n_msgs), dtype=tf.int32, name="ss_index_placeholder"
            )

    def _build_embeds(self):
        with tf.name_scope('embeds'):
            # Use an Embedding layer instead of manually creating a lookup table
            self.word_embed_layer = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=self.word_embed_size,
                weights=[self.word_table_init.numpy()],  # Initialize with word_table_init
                trainable=False,
                name="word_embedding_layer"
            )
            self.word_embed = self.word_embed_layer(self.word_ph)
        

    def _create_msg_embed_layer_in(self):
        """
            acquire the inputs for MEL.

            Input:
                word_embed: batch_size * max_n_days * max_n_msgs * max_n_words * word_embed_size

            Output:
                mel_in: same as word_embed
        """
        with tf.name_scope('mel_in'):
            # Batch Normalization layer if required
            if self.use_in_bn:
                bn_layer = tf.keras.layers.BatchNormalization(name='bn-mel_inputs')
                self.word_embed = bn_layer(self.word_embed, training=self.is_training_phase)

            # Dropout layer for input
            dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_train_mel_in, name="mel_input_dropout")
            self.mel_in = dropout_layer(self.word_embed, training=self.is_training_phase)

    def _create_msg_embed_layer(self):
        """
            Input:
                mel_in: same as word_embed

            Output:
                msg_embed: batch_size * max_n_days * max_n_msgs * msg_embed_size
        """

        def _for_one_trading_day(daily_in, daily_ss_index_vec):
            """
            daily_in: max_n_msgs * max_n_words * word_embed_size
            """
            # Define bidirectional RNNs
            mel_cell_f = tf.keras.layers.LSTM(self.mel_h_size, return_sequences=True, return_state=False)
            mel_cell_b = tf.keras.layers.LSTM(self.mel_h_size, return_sequences=True, return_state=False)

            # Apply bidirectional RNN
            out_f, out_b = mel_cell_f(daily_in), mel_cell_b(daily_in)
            msg_embed = (out_f + out_b) / 2  # Average forward and backward outputs
            return msg_embed

        with tf.name_scope('mel'):
            msg_embed_list = []
            for i in range(self.max_n_days):
                day_embed = _for_one_trading_day(self.mel_in[:, i], self.ss_index_ph[:, i])
                msg_embed_list.append(day_embed)

            self.msg_embed = tf.stack(msg_embed_list, axis=1)  # Stack along days dimension
            self.msg_embed = tf.nn.dropout(self.msg_embed, rate=self.dropout_train_mel)

        def _for_one_sample(sample, sample_ss_index, sample_mask):
            return neural.iter(size=self.max_n_days, func=_for_one_trading_day,
                               iter_arg=sample, iter_arg2=sample_ss_index, iter_arg3=sample_mask)

        def _for_one_batch():
            return neural.iter(size=self.batch_size, func=_for_one_sample,
                               iter_arg=self.mel_in, iter_arg2=self.ss_index_ph, iter_arg3=self.n_words_ph)
    def _create_corpus_embed(self):
        """
            msg_embed: batch_size * max_n_days * max_n_msgs * msg_embed_size

            => corpus_embed: batch_size * max_n_days * corpus_embed_size
        """
        
        with tf.name_scope('corpus_embed'):
            # Dense layer for projecting u_t
            proj_layer = tf.keras.layers.Dense(self.msg_embed_size, activation='tanh', use_bias=False, name="proj_u")
            proj_u = proj_layer(self.msg_embed)

            # Attention weights
            w_u = tf.Variable(
                tf.random.normal([self.msg_embed_size, 1], stddev=0.1),
                name="w_u",
                trainable=True
            )
            u = tf.reduce_mean(tf.matmul(proj_u, w_u), axis=-1)

            # Apply masking
            mask_msgs = tf.sequence_mask(self.n_msgs_ph, maxlen=self.max_n_msgs, dtype=tf.float32)
            masked_u = tf.where(mask_msgs, u, tf.fill(tf.shape(u), -1e9))
            u = tf.nn.softmax(masked_u, axis=-1)

            # Weighted sum for corpus embedding
            u = tf.expand_dims(u, axis=-2)  # Shape: batch_size * max_n_days * 1 * max_n_msgs
            corpus_embed = tf.matmul(u, self.msg_embed)  # Weighted sum
            corpus_embed = tf.squeeze(corpus_embed, axis=-2)  # Remove singleton dimension

            # Apply dropout
            dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_train_ce)
            self.corpus_embed = dropout_layer(corpus_embed, training=self.is_training_phase)
            


    def _build_mie(self):
        """
            Create market information encoder.

            corpus_embed: batch_size * max_n_days * corpus_embed_size
            price: batch_size * max_n_days * 3
            => x: batch_size * max_n_days * x_size
        """
        with tf.name_scope('mie'):
            self.price = self.price_ph
            self.price_size = 3

            if self.variant_type == 'tech':
                self.x = self.price
                self.x_size = self.price_size
            else:
                self._create_msg_embed_layer_in()
                self._create_msg_embed_layer()
                initial_state = None

                # MSINCell updated with TensorFlow 2.x-compatible implementations
                cell = MSINCell(input_size=self.price_size, num_units=self.msin_h_size,
                                v_size=self.msg_embed_size, max_n_msgs=self.max_n_msgs)

                msin = MSIN()
                self.x, self.P, state = msin.dynamic_msin(
                    cell=cell,
                    inputs=self.price,
                    s_inputs=self.msg_embed,
                    sequence_length=self.T_ph,
                    initial_state=initial_state,
                    dtype=tf.float32
                )
                """
                self._create_msg_embed_layer_in()
                self._create_msg_embed_layer()
                    
                
                self._create_corpus_embed()
                if self.variant_type == 'fund':
                    self.x = self.corpus_embed
                    self.x_size = self.corpus_embed_size
                else:
                    self.x = tf.concat([self.corpus_embed, self.price], axis=2)
                    self.x_size = self.corpus_embed_size + self.price_size
                """







    def _create_vmd_with_h_rec(self):
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_h_rec'):
                x = tf.nn.dropout(self.x, rate=self.dropout_train_vmd_in)
                x = tf.transpose(x, [1, 0, 2])  # max_n_days * batch_size * x_size
                y_ = tf.transpose(self.y_ph, [1, 0, 2])  # max_n_days * batch_size * y_size

                self.mask_aux_trading_days = tf.sequence_mask(self.T_ph - 1, self.max_n_days, dtype=tf.bool)

                def _loop_body(t, ta_h_s, ta_z_prior, ta_z_post, ta_kl):
                    h_s_t_1 = ta_h_s.read(t - 1) if t > 0 else tf.zeros([self.batch_size, self.h_size])
                    z_t_1 = ta_z_post.read(t - 1) if t > 0 else tf.zeros([self.batch_size, self.z_size])

                    r = tf.keras.layers.Dense(self.h_size, activation='sigmoid')(tf.concat([x[t], h_s_t_1, z_t_1], axis=-1))
                    u = tf.keras.layers.Dense(self.h_size, activation='sigmoid')(tf.concat([x[t], h_s_t_1, z_t_1], axis=-1))

                    h_tilde = tf.keras.layers.Dense(self.h_size, activation='tanh')(tf.concat([x[t], r * h_s_t_1, z_t_1], axis=-1))
                    h_s_t = (1 - u) * h_s_t_1 + u * h_tilde

                    h_z_prior_t = tf.keras.layers.Dense(self.z_size, activation='tanh')(tf.concat([x[t], h_s_t], axis=-1))
                    z_prior_t = h_z_prior_t  # Normally, sampling logic would be here

                    h_z_post_t = tf.keras.layers.Dense(self.z_size, activation='tanh')(tf.concat([x[t], h_s_t, y_[t]], axis=-1))
                    z_post_t = h_z_post_t  # Normally, sampling logic would be here

                    ta_h_s = ta_h_s.write(t, h_s_t)
                    ta_z_prior = ta_z_prior.write(t, z_prior_t)
                    ta_z_post = ta_z_post.write(t, z_post_t)
                    ta_kl = ta_kl.write(t, tf.reduce_mean(z_post_t - z_prior_t))

                    return t + 1, ta_h_s, ta_z_prior, ta_z_post, ta_kl

                ta_h_s = tf.TensorArray(tf.float32, size=self.max_n_days)
                ta_z_prior = tf.TensorArray(tf.float32, size=self.max_n_days)
                ta_z_post = tf.TensorArray(tf.float32, size=self.max_n_days)
                ta_kl = tf.TensorArray(tf.float32, size=self.max_n_days)

                _, ta_h_s, ta_z_prior, ta_z_post, ta_kl = tf.while_loop(
                    lambda t, *args: t < self.max_n_days, _loop_body, [0, ta_h_s, ta_z_prior, ta_z_post, ta_kl]
                )

                self.h_s = tf.transpose(ta_h_s.stack(), [1, 0, 2])
                self.z_prior = tf.transpose(ta_z_prior.stack(), [1, 0, 2])
                self.z_post = tf.transpose(ta_z_post.stack(), [1, 0, 2])
                self.kl = tf.reduce_sum(tf.transpose(ta_kl.stack(), [1, 0, 2]), axis=2)

    def _create_vmd_with_zh_rec(self):
        """
        Create a variational movement decoder.

        x: batch_size * max_n_days * vmd_in_size
        => vmd_h: batch_size * max_n_days * vmd_h_size
        => z: batch_size * max_n_days * vmd_z_size
        => y: batch_size * max_n_days * 2
        """
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_zh_rec', reuse=tf.AUTO_REUSE):
                x = tf.nn.dropout(self.x, rate=self.dropout_train_vmd_in)

                lstm_cell = tf.keras.layers.LSTM(self.h_size, return_sequences=True, return_state=True)
                h_s, _ = lstm_cell(x)

                z = tf.keras.layers.Dense(self.z_size, activation='tanh')(h_s)
                y = tf.keras.layers.Dense(self.y_size, activation='softmax')(z)

                self.h_s = h_s
                self.z = z
                self.y = y


    def _create_discriminative_vmd(self):
        """
        Create a discriminative movement decoder.

        x: batch_size * max_n_days * vmd_in_size
        => vmd_h: batch_size * max_n_days * vmd_h_size
        => z: batch_size * max_n_days * vmd_z_size
        => y: batch_size * max_n_days * 2
        """
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_discriminative', reuse=tf.AUTO_REUSE):
                x = tf.nn.dropout(self.x, rate=self.dropout_train_vmd_in)

                lstm_cell = tf.keras.layers.LSTM(self.h_size, return_sequences=True, return_state=True)
                h_s, _ = lstm_cell(x)

                z = tf.keras.layers.Dense(self.z_size, activation='tanh')(h_s)
                y = tf.keras.layers.Dense(self.y_size, activation='softmax')(z)

                self.h_s = h_s
                self.z = z
                self.y = y

                # Extract g_T
                sample_index = tf.range(self.batch_size, dtype=tf.int32)
                indexed_T = tf.stack([sample_index, self.T_ph - 1], axis=1)
                self.g_T = tf.gather_nd(h_s, indexed_T)


    def _build_vmd(self):
        if self.variant_type == 'discriminative':
            self._create_discriminative_vmd()
        else:
            if self.vmd_rec == 'h':
                self._create_vmd_with_h_rec()
            else:
                self._create_vmd_with_zh_rec()

    def _build_temporal_att(self):
        """
        g: batch_size * max_n_days * g_size
        g_T: batch_size * g_size
        """
        with tf.name_scope('tda'):
            with tf.variable_scope('tda', reuse=tf.AUTO_REUSE):
                proj_i = tf.keras.layers.Dense(self.g_size, activation='tanh')(self.g)
                w_i = tf.Variable(tf.random.truncated_normal([self.g_size, 1], stddev=0.1), name='w_i')
                v_i = tf.reduce_sum(tf.matmul(proj_i, w_i), axis=-1)  # batch_size * max_n_days

                proj_d = tf.keras.layers.Dense(self.g_size, activation='tanh')(self.g)
                g_T_expanded = tf.expand_dims(self.g_T, axis=-1)  # batch_size * g_size * 1
                v_d = tf.reduce_sum(tf.matmul(proj_d, g_T_expanded), axis=-1)  # batch_size * max_n_days

                aux_score = tf.multiply(v_i, v_d, name='v_stared')
                ninf = tf.fill(tf.shape(aux_score), np.NINF)
                masked_aux_score = tf.where(self.mask_aux_trading_days, aux_score, ninf)
                v_stared = tf.nn.softmax(masked_aux_score)

                self.v_stared = tf.where(tf.math.is_nan(v_stared), tf.zeros_like(v_stared), v_stared)

                if self.daily_att == 'y':
                    context = tf.transpose(self.y, [0, 2, 1])  # batch_size * y_size * max_n_days
                else:
                    context = tf.transpose(self.g, [0, 2, 1])  # batch_size * g_size * max_n_days

                v_stared_expanded = tf.expand_dims(self.v_stared, -1)  # batch_size * max_n_days * 1
                att_c = tf.reduce_sum(context * v_stared_expanded, axis=-1)  # batch_size * g_size / y_size
                self.y_T = tf.keras.layers.Dense(self.y_size, activation='softmax')(
                    tf.concat([att_c, self.g_T], axis=-1)
                )


    def _create_generative_ata(self):
        """
        Calculate generative loss.

        g: batch_size * max_n_days * g_size
        y: batch_size * max_n_days * y_size
        kl_loss: batch_size * max_n_days
        => loss: batch_size
        """
        with tf.name_scope('ata'):
            with tf.variable_scope('ata', reuse=tf.AUTO_REUSE):
                v_aux = self.alpha * self.v_stared  # batch_size * max_n_days

                minor = 1e-7  # Small constant for numerical stability
                likelihood_aux = tf.reduce_sum(self.y_ph * tf.math.log(self.y + minor), axis=2)  # batch_size * max_n_days

                kl_lambda = self._kl_lambda()
                obj_aux = likelihood_aux - kl_lambda * self.kl  # batch_size * max_n_days

                # Special treatment for T
                self.y_T_ = tf.gather_nd(self.y_ph, self.indexed_T)  # batch_size * y_size
                likelihood_T = tf.reduce_sum(self.y_T_ * tf.math.log(self.y_T + minor), axis=1, keepdims=True)

                kl_T = tf.gather_nd(self.kl, self.indexed_T)  # batch_size * 1
                obj_T = likelihood_T - kl_lambda * kl_T

                # Total loss
                obj = obj_T + tf.reduce_sum(obj_aux * v_aux, axis=1, keepdims=True)  # batch_size * 1
                self.loss = tf.reduce_mean(-obj, axis=0)

        """
             calculate loss.

             g: batch_size * max_n_days * g_size
             y: batch_size * max_n_days * y_size
             kl_loss: batch_size * max_n_days
             => loss: batch_size
        """
        with tf.name_scope('ata'):
            with tf.variable_scope('ata'):
                v_aux = self.alpha * self.v_stared  # batch_size * max_n_days

                minor = 0.0 # 0.0, 1e-7*
                likelihood_aux = tf.reduce_sum(tf.multiply(self.y_ph, tf.log(self.y + minor)), axis=2)  # batch_size * max_n_days

                kl_lambda = self._kl_lambda()
                obj_aux = likelihood_aux - kl_lambda * self.kl  # batch_size * max_n_days

                # deal with T specially, likelihood_T: batch_size, 1
                self.y_T_ = tf.gather_nd(params=self.y_ph, indices=self.indexed_T)  # batch_size * y_size
                likelihood_T = tf.reduce_sum(tf.multiply(self.y_T_, tf.log(self.y_T + minor)), axis=1, keep_dims=True)

                kl_T = tf.reshape(tf.gather_nd(params=self.kl, indices=self.indexed_T), shape=[self.batch_size, 1])
                obj_T = likelihood_T - kl_lambda * kl_T

                obj = obj_T + tf.reduce_sum(tf.multiply(obj_aux, v_aux), axis=1, keep_dims=True)  # batch_size * 1
                self.loss = tf.reduce_mean(-obj, axis=[0, 1])
                '''
                msg_num = tf.to_float(tf.tile(tf.expand_dims(self.n_msgs_ph, axis=-1),[1,1,self.max_n_msgs]))
                new_P = tf.clip_by_value(self.P, 1e-30, 1)
                msg_num = tf.clip_by_value(tf.log(msg_num), 1e-30, 30)
                new_msg_num = tf.clip_by_value(1/msg_num, 1e-30, 1)
                
                P_obj = tf.reduce_sum(tf.multiply(self.P, tf.multiply( tf.log(new_P), new_msg_num )+1 ), axis=[1,2])
                
                nonzero = tf.to_float(tf.count_nonzero(self.n_msgs_ph,axis=-1))
                P_obj = tf.divide(P_obj, nonzero)

                self.loss = self.loss +  tf.reduce_mean(-P_obj)
                '''
        
                
    def _create_discriminative_ata(self):
        """
        Calculate discriminative loss.

        g: batch_size * max_n_days * g_size
        y: batch_size * max_n_days * y_size
        => loss: batch_size
        """
        with tf.name_scope('ata'):
            with tf.variable_scope('ata', reuse=tf.AUTO_REUSE):
                v_aux = self.alpha * self.v_stared  # batch_size * max_n_days

                minor = 1e-7  # Small constant for numerical stability
                likelihood_aux = tf.reduce_sum(self.y_ph * tf.math.log(self.y + minor), axis=2)  # batch_size * max_n_days

                # Special treatment for T
                self.y_T_ = tf.gather_nd(self.y_ph, self.indexed_T)  # batch_size * y_size
                likelihood_T = tf.reduce_sum(self.y_T_ * tf.math.log(self.y_T + minor), axis=1, keepdims=True)

                # Total loss
                obj = likelihood_T + tf.reduce_sum(likelihood_aux * v_aux, axis=1, keepdims=True)  # batch_size * 1

                # Incorporate additional penalty term based on `P`
                new_P = tf.clip_by_value(self.P, 1e-8, 1)
                P_obj = tf.reduce_sum(self.P * tf.math.log(new_P), axis=-1)
                
                self.loss = tf.reduce_mean(-obj, axis=[0, 1]) + tf.reduce_mean(-P_obj, axis=[0, 1])

        """
             calculate discriminative loss.

             g: batch_size * max_n_days * g_size
             y: batch_size * max_n_days * y_size
             => loss: batch_size
        """
        with tf.name_scope('ata'):
            with tf.variable_scope('ata'):
                v_aux = self.alpha * self.v_stared  # batch_size * max_n_days

                minor = 1e-7  # 0.0, 1e-7*
                likelihood_aux = tf.reduce_sum(tf.multiply(self.y_ph, tf.log(self.y + minor)), axis=2)  # batch_size * max_n_days

                # deal with T specially, likelihood_T: batch_size, 1
                self.y_T_ = tf.gather_nd(params=self.y_ph, indices=self.indexed_T)  # batch_size * y_size
                likelihood_T = tf.reduce_sum(tf.multiply(self.y_T_, tf.log(self.y_T + minor)), axis=1, keep_dims=True)

                obj = likelihood_T + tf.reduce_sum(tf.multiply(likelihood_aux, v_aux), axis=1, keep_dims=True)  # batch_size * 1
                new_P = tf.clip_by_value(self.P, 1e-8, 1)
                P_obj = tf.reduce_sum(tf.multiply(self.P, tf.log(new_P)), axis=-1)
                
                self.loss = tf.reduce_mean(-obj, axis=[0, 1])
                self.loss = self.loss +  tf.reduce_mean(-P_obj, axis=[0, 1])

    def _build_ata(self):
        if self.variant_type == 'discriminative':
            self._create_discriminative_ata()
        else:
            self._create_generative_ata()


    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            # Learning rate with decay
            if self.opt == 'sgd':
                decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=self.lr,
                    decay_steps=self.decay_step,
                    decay_rate=self.decay_rate
                )
                optimizer = tf.keras.optimizers.SGD(learning_rate=decayed_lr, momentum=self.momentum)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

            # Compute gradients and apply them
            gradients = tf.gradients(self.loss, self.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip)
            self.optimize = optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # Increment global step
            self.global_step.assign_add(1)

        with tf.name_scope('optimizer'):
            if self.opt == 'sgd':
                decayed_lr = tf.train.exponential_decay(learning_rate=self.lr, global_step=self.global_step,
                                                        decay_steps=self.decay_step, decay_rate=self.decay_rate)
                optimizer = tf.train.MomentumOptimizer(learning_rate=decayed_lr, momentum=self.momentum)
            else:
                optimizer = tf.train.AdamOptimizer(self.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip)
            self.optimize = optimizer.apply_gradients(zip(gradients, variables))
            self.global_step = tf.assign_add(self.global_step, 1)

    def assemble_graph(self):
        logger.info('Start graph assembling...')
        with tf.device('/device:GPU:0'):  # Ensure the model uses GPU
            self._build_placeholders()
            self._build_embeds()
            self._build_mie()
            self._build_vmd()
            self._build_temporal_att()
            self._build_ata()
            self._create_optimizer()

            

    def _kl_lambda(self):
        def _nonzero_kl_lambda():
            if self.use_constant_kl_lambda:
                return self.constant_kl_lambda
            else:
                return tf.minimum(self.kl_lambda_anneal_rate * global_step, 1.0)

        global_step = tf.cast(self.global_step, tf.float32)

        return tf.cond(global_step < self.kl_lambda_start_step,
                       lambda: 0.0,
                       _nonzero_kl_lambda)

        def _nonzero_kl_lambda():
            if self.use_constant_kl_lambda:
                return self.constant_kl_lambda
            else:
                return tf.minimum(self.kl_lambda_anneal_rate * global_step, 1.0)

        global_step = tf.cast(self.global_step, tf.float32)

        return tf.cond(global_step < self.kl_lambda_start_step, lambda: 0.0, _nonzero_kl_lambda)

    def _linear(self, args, output_size, activation=None, use_bias=True, use_bn=False):
        if not isinstance(args, (list, tuple)):
            args = [args]

        # Combine input arguments
        x = tf.concat(args, axis=-1)
        input_size = x.shape[-1]

        # Define weight and bias
        weight = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(input_size, output_size)),
                             name="weight")
        result = tf.matmul(x, weight)

        if use_bias:
            bias = tf.Variable(tf.zeros_initializer()(shape=(output_size,)), name="bias")
            result = tf.nn.bias_add(result, bias)

        # Apply batch normalization
        if use_bn:
            result = tf.keras.layers.BatchNormalization()(result)

        # Apply activation function
        if activation == 'tanh':
            result = tf.nn.tanh(result)
        elif activation == 'sigmoid':
            result = tf.nn.sigmoid(result)
        elif activation == 'relu':
            result = tf.nn.relu(result)
        elif activation == 'softmax':
            result = tf.nn.softmax(result)

        return result

        if type(args) not in (list, tuple):
            args = [args]

        shape = [a if a else -1 for a in args[0].get_shape().as_list()[:-1]]
        shape.append(output_size)

        sizes = [a.get_shape()[-1].value for a in args]
        total_arg_size = sum(sizes)
        scope = tf.get_variable_scope()
        x = args[0] if len(args) == 1 else tf.concat(args, -1)

        with tf.variable_scope(scope):
            weight = tf.get_variable('weight', [total_arg_size, output_size], dtype=tf.float32, initializer=self.initializer)
            res = tf.tensordot(x, weight, axes=1)
            if use_bias:
                bias = tf.get_variable('bias', [output_size], dtype=tf.float32, initializer=self.bias_initializer)
                res = tf.nn.bias_add(res, bias)

        res = tf.reshape(res, shape)

        if use_bn:
            res = batch_norm(res, center=True, scale=True, decay=0.99, updates_collections=None,
                             is_training=self.is_training_phase, scope=scope)

        if activation == 'tanh':
            res = tf.nn.tanh(res)
        elif activation == 'sigmoid':
            res = tf.nn.sigmoid(res)
        elif activation == 'relu':
            res = tf.nn.relu(res)
        elif activation == 'softmax':
            res = tf.nn.softmax(res)

        return res

    def _z(self, arg, is_prior):
        mean = self._linear(arg, self.z_size)
        log_var = self._linear(arg, self.z_size)

        stddev = tf.exp(0.5 * log_var)  # Variance to standard deviation
        epsilon = tf.random.normal(shape=(self.batch_size, self.z_size))

        z = mean if is_prior else mean + stddev * epsilon
        pdf_z = tfp.distributions.Normal(loc=mean, scale=stddev)

        return z, pdf_z

        mean = self._linear(arg, self.z_size)
        stddev = self._linear(arg, self.z_size)
        stddev = tf.sqrt(tf.exp(stddev))
        
        epsilon = tf.random_normal(shape=[self.batch_size, self.z_size])

        z = mean if is_prior else mean + tf.multiply(stddev, epsilon)
        pdf_z = ds.Normal(loc=mean, scale=stddev)

        return z, pdf_z

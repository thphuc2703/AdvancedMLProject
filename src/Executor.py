#!/usr/local/bin/python
import os
import tensorflow as tf
import metrics as metrics
import stat_logger as stat_logger
from DataPipe import DataPipe
from ConfigLoader import logger
import pdb


class Executor:
    def __init__(self, model, silence_step=200, skip_step=20):
        self.model = model
        self.silence_step = silence_step
        self.skip_step = skip_step
        self.pipe = DataPipe()

        # Replacing deprecated Saver with Checkpoint
        self.checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.model.tf_saver_path, max_to_keep=5
        )

        # TensorFlow 2.x GPU setup
        self.tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True

    def unit_test_train(self):
        with tf.compat.v1.Session(config=self.tf_config) as sess:
            word_table_init = self.pipe.init_word_table()
            feed_table_init = {self.model.word_table_init: word_table_init}
            sess.run(tf.compat.v1.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info("Word table init: done!")

            logger.info(f"Model: {self.model.model_name}, start a new session!")

            n_iter = sess.run(self.model.global_step)

            # forward
            train_batch_loss_list = []
            train_epoch_size = 0.0
            train_epoch_n_acc = 0.0
            train_batch_gen = self.pipe.batch_gen(phase="train")
            train_batch_dict = next(train_batch_gen)

            while n_iter < 100:
                feed_dict = {
                    self.model.is_training_phase: True,
                    self.model.batch_size: train_batch_dict["batch_size"],
                    self.model.stock_ph: train_batch_dict["stock_batch"],
                    self.model.T_ph: train_batch_dict["T_batch"],
                    self.model.n_words_ph: train_batch_dict["n_words_batch"],
                    self.model.n_msgs_ph: train_batch_dict["n_msgs_batch"],
                    self.model.y_ph: train_batch_dict["y_batch"],
                    self.model.price_ph: train_batch_dict["price_batch"],
                    self.model.mv_percent_ph: train_batch_dict["mv_percent_batch"],
                    self.model.word_ph: train_batch_dict["word_batch"],
                    self.model.ss_index_ph: train_batch_dict["ss_index_batch"],
                }

                ops = [self.model.y_T, self.model.y_T_, self.model.loss, self.model.optimize]
                pdb.set_trace()
                train_batch_y, train_batch_y_, train_batch_loss, _ = sess.run(ops, feed_dict)

                # Training batch stat
                train_epoch_size += float(train_batch_dict["batch_size"])
                train_batch_loss_list.append(train_batch_loss)
                train_batch_n_acc = sess.run(metrics.n_accurate(y=train_batch_y, y_=train_batch_y_))
                train_epoch_n_acc += float(train_batch_n_acc)

                stat_logger.print_batch_stat(
                    n_iter, train_batch_loss, train_batch_n_acc, train_batch_dict["batch_size"]
                )
                n_iter += 1

    def generation(self, sess, phase):
        generation_gen = self.pipe.batch_gen_by_stocks(phase)

        gen_loss_list = []
        gen_size, gen_n_acc = 0.0, 0.0
        y_list, y_list_ = [], []

        for gen_batch_dict in generation_gen:
            feed_dict = {
                self.model.is_training_phase: False,
                self.model.batch_size: gen_batch_dict["batch_size"],
                self.model.stock_ph: gen_batch_dict["stock_batch"],
                self.model.T_ph: gen_batch_dict["T_batch"],
                self.model.n_words_ph: gen_batch_dict["n_words_batch"],
                self.model.n_msgs_ph: gen_batch_dict["n_msgs_batch"],
                self.model.y_ph: gen_batch_dict["y_batch"],
                self.model.price_ph: gen_batch_dict["price_batch"],
                self.model.mv_percent_ph: gen_batch_dict["mv_percent_batch"],
                self.model.word_ph: gen_batch_dict["word_batch"],
                self.model.ss_index_ph: gen_batch_dict["ss_index_batch"],
                self.model.dropout_mel_in: 0.0,
                self.model.dropout_mel: 0.0,
                self.model.dropout_ce: 0.0,
                self.model.dropout_vmd_in: 0.0,
                self.model.dropout_vmd: 0.0,
            }

            gen_batch_y, gen_batch_y_, gen_batch_loss = sess.run(
                [self.model.y_T, self.model.y_T_, self.model.loss], feed_dict=feed_dict
            )

            # Gather results
            y_list.append(gen_batch_y)
            y_list_.append(gen_batch_y_)
            gen_loss_list.append(gen_batch_loss)

            gen_batch_n_acc = float(sess.run(metrics.n_accurate(y=gen_batch_y, y_=gen_batch_y_)))
            gen_n_acc += gen_batch_n_acc

            batch_size = float(gen_batch_dict["batch_size"])
            gen_size += batch_size

        results = metrics.eval_res(
            gen_n_acc, gen_size, gen_loss_list, y_list, y_list_, use_mcc=True
        )
        return results

    def train_and_dev(self):
        with tf.compat.v1.Session(config=self.tf_config) as sess:
            # Initialize variables
            writer = tf.summary.FileWriter(self.model.tf_graph_path, sess.graph)

            feed_table_init = {self.model.word_table_init: self.pipe.init_word_table()}
            sess.run(tf.compat.v1.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info("Word table init: done!")

            # Restore checkpoint if available
            checkpoint = self.checkpoint_manager.latest_checkpoint
            if checkpoint:
                logger.info(f"Restoring model from {checkpoint}")
                self.checkpoint.restore(checkpoint)
            else:
                logger.info(f"Starting a new session for model {self.model.model_name}")

            for epoch in range(self.model.n_epochs):
                logger.info(f"Epoch: {epoch + 1}/{self.model.n_epochs} start")

                train_batch_loss_list = []
                epoch_size, epoch_n_acc = 0.0, 0.0

                train_batch_gen = self.pipe.batch_gen(phase="train")

                for train_batch_dict in train_batch_gen:
                    feed_dict = {
                        self.model.is_training_phase: True,
                        self.model.batch_size: train_batch_dict["batch_size"],
                        self.model.stock_ph: train_batch_dict["stock_batch"],
                        self.model.T_ph: train_batch_dict["T_batch"],
                        self.model.n_words_ph: train_batch_dict["n_words_batch"],
                        self.model.n_msgs_ph: train_batch_dict["n_msgs_batch"],
                        self.model.y_ph: train_batch_dict["y_batch"],
                        self.model.price_ph: train_batch_dict["price_batch"],
                        self.model.mv_percent_ph: train_batch_dict["mv_percent_batch"],
                        self.model.word_ph: train_batch_dict["word_batch"],
                        self.model.ss_index_ph: train_batch_dict["ss_index_batch"],
                    }

                    ops = [
                        self.model.y_T,
                        self.model.y_T_,
                        self.model.loss,
                        self.model.optimize,
                        self.model.global_step,
                    ]
                    train_batch_y, train_batch_y_, train_batch_loss, _, n_iter = sess.run(
                        ops, feed_dict
                    )

                    # Training batch stats
                    epoch_size += float(train_batch_dict["batch_size"])
                    train_batch_loss_list.append(train_batch_loss)
                    train_batch_n_acc = sess.run(
                        metrics.n_accurate(y=train_batch_y, y_=train_batch_y_)
                    )
                    epoch_n_acc += float(train_batch_n_acc)

                    # Save model
                    if n_iter >= self.silence_step and n_iter % self.skip_step == 0:
                        self.checkpoint_manager.save()
                        stat_logger.print_batch_stat(
                            n_iter, train_batch_loss, train_batch_n_acc, train_batch_dict["batch_size"]
                        )

                # Epoch stats
                epoch_loss, epoch_acc = metrics.basic_train_stat(
                    train_batch_loss_list, epoch_n_acc, epoch_size
                )
                stat_logger.print_epoch_stat(epoch_loss=epoch_loss, epoch_acc=epoch_acc)

            writer.close()

    def restore_and_test(self):
        with tf.compat.v1.Session(config=self.tf_config) as sess:
            checkpoint = self.checkpoint_manager.latest_checkpoint
            if checkpoint:
                logger.info(f"Restoring model from {checkpoint}")
                self.checkpoint.restore(checkpoint)
            else:
                logger.info(f"Model {self.model.model_name} NOT found!")
                raise IOError

            results = self.generation(sess, phase="test")
            stat_logger.print_eval_res(results, use_mcc=True)

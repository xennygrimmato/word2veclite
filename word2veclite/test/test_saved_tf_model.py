import unittest
from unittest import TestCase
from word2veclite.word2veclite import Word2Vec
import numpy as np
import tensorflow as tf


class TestFunctions(TestCase):

    learning_rate = 0.1
    W1 = np.ones((7, 2))
    W2 = np.array([[0.2, 0.2, 0.3, 0.4, 0.5, 0.3, 0.2],
                   [0.3, 0., 0.1, 0., 1., 0.1, 0.5]])
    V, N = W1.shape
    context_words = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]])
    center_word = np.array([0., 0., 1., 0., 0., 0., 0.])

    # def test_saved_tf_model(self):
    #     skipgram = Word2Vec(learning_rate=self.learning_rate)
    #     W1_m, W2_m, loss_m = skipgram.skipgram(np.asmatrix(self.context_words),
    #                                            np.asmatrix(self.center_word),
    #                                            self.W1,
    #                                            self.W2,
    #                                            0.)
    #
    #     with tf.Session() as sess:
    #         saver = tf.train.import_meta_graph('./tmp_model.meta')
    #         graph = tf.get_default_graph()
    #         W1_tf = graph.get_tensor_by_name("W1_tf:0")
    #
    #     for i in range(self.V):
    #         for j in range(self.N):
    #             self.assertAlmostEqual(W1_m[i, j], W1_tf[i, j], places=5)
    #
    #     for i in range(self.N):
    #         for j in range(self.V):
    #             self.assertAlmostEqual(W2_m[i, j], W2_tf[i, j], places=5)
    #
    #     self.assertAlmostEqual(loss_m, float(loss_tf), places=5)

    def test_load_tf_model(self):
        W1_tf = tf.Variable(tf.zeros([7, 2]), dtype=tf.float32, name='W1_tf')
        W2_tf = tf.Variable(tf.zeros([2, 7]), dtype=tf.float32, name='W2_tf')
        loss_tf = tf.Variable(tf.zeros(shape=[1, ]), dtype=tf.float32, name='loss_tf')
        # grad_W1 = tf.Variable(dtype=tf.float32, name='grad_W1')
        # grad_W2 = tf.Variable(dtype=tf.float32, name='grad_W2')

        tmp_var = tf.Variable(tf.zeros([2, 3]), dtype=tf.float32, name='tmp_var')

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # saver = tf.train.import_meta_graph('./tmp_model.meta')
            saver.restore(sess, './tmp_model')
            # sess.run(tf.global_variables_initializer())
            print(sess.run(tmp_var))
            print(sess.run(W1_tf))
            print(sess.run(W2_tf))
            print(sess.run(loss_tf))
            # print(sess.run(grad_W1))
            # print(sess.run(grad_W2))


if __name__ == "__main__":
    unittest.main()

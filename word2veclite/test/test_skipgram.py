import os
import unittest
import numpy as np
import tensorflow as tf


class TestFunctions(unittest.TestCase):
    learning_rate = 0.1
    W1 = np.ones((7, 2))
    W2 = np.array([[0.2, 0.2, 0.3, 0.4, 0.5, 0.3, 0.2],
                   [0.3, 0., 0.1, 0., 1., 0.1, 0.5]])
    V, N = W1.shape # V = vocab size, N = dimension of word vectors
    context_words = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0]])
    center_word = np.array([0., 0., 1., 0., 0., 0., 0.])

    def test_skipgram(self):
        # with tf.name_scope("skipgram"):
        x = tf.placeholder(shape=[self.V, 1], dtype=tf.float32, name="x")
        W1_tf = tf.Variable(self.W1, dtype=tf.float32, name='W1_tf')
        W2_tf = tf.Variable(self.W2, dtype=tf.float32, name='W2_tf')
        h = tf.matmul(tf.transpose(W1_tf), x)
        u = tf.stack([tf.matmul(tf.transpose(W2_tf), h)
                      for i in range(len(self.context_words))
                      ])
        # TODO: Simplify loss_tf
        loss_tf = tf.Variable(np.zeros(shape=[1, ]), dtype=tf.float32, name='loss_tf')
        val = -tf.reduce_sum([u[i][int(np.where(c == 1)[0])]
                                   for i, c in zip(range(len(self.context_words)), self.context_words)],
                                   axis=0) + \
                tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(u), axis=1)), axis=0)
        tf.assign(loss_tf, val)

        # grad_W1 = tf.Variable(tf.Constant([1.0]), dtype=tf.float32, name='grad_W1')
        # grad_W2 = tf.Variable(tf.Constant([1.0]), dtype=tf.float32, name='grad_W2')
        # grad_w1_val, grad_w2_val = tf.gradients(loss_tf, [W1_tf, W2_tf])
        # tf.assign(grad_W1, grad_w1_val)
        # tf.assign(grad_W2, grad_w2_val)

        tmp_var = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='tmp_var')

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            try:
                saver = tf.train.import_meta_graph('./tmp_model.meta')
                saver.restore(sess, './tmp_model')
                sess.run(init)
                W1_tf, W2_tf, loss_tf, dW1_tf, dW2_tf = sess.run([W1_tf, W2_tf, loss_tf, grad_W1, grad_W2],
                                                                 feed_dict={x: self.center_word.reshape(self.V, 1)})
                W1_tf -= self.learning_rate * dW1_tf
                W2_tf -= self.learning_rate * dW2_tf
                saver.save(sess=sess, save_path='./tmp_model')
            except IOError:
                saver = tf.train.Saver()
                sess.run(init)
                saver.save(sess=sess, save_path='./tmp_model')


if __name__ == "__main__":
    unittest.main()

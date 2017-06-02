""" The mo frills implementation of word2vec skip-gram model using NCE loss. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from preTrain import process_data
import os
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 300  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000  # how many steps to skip before reporting the loss
LOG_DIR = 'tmp/demo'
CKPT_NAME = 'my-model.ckpt'
ECHO_TIME = 5

def word2vec(batch_gen,echo):
    """ Build the graph for word2vec model and train it """
    # Step 1: define the placeholders for input and output
    tf.reset_default_graph()
    center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
    target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

    # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
    # Step 2: define weights. In word2vec, it's actually the weights that we care about
    embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0),
                               name='embed_matrix')
    # Step 3: define the inference
    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

    # Step 4: construct variables for NCE loss
    nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                 stddev=1.0 / (EMBED_SIZE ** 0.5)),
                             name='nce_weight')
    nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

    # define loss function to be NCE loss function
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                         biases=nce_bias,
                                         labels=target_words,
                                         inputs=embed,
                                         num_sampled=NUM_SAMPLED,
                                         num_classes=VOCAB_SIZE), name='loss')

    # Step 5: define optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if echo != 1:
            saver.restore(sess, os.path.join(LOG_DIR, CKPT_NAME))
        else:
            sess.run(tf.global_variables_initializer())
        config = projector.ProjectorConfig()
        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        embedding = config.embeddings.add()
        embedding.tensor_name = embed_matrix.name
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
        total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
        for index in range(NUM_TRAIN_STEPS):
            try:
                centers, targets = next(batch_gen)
            except:
                saver.save(sess, os.path.join(LOG_DIR, CKPT_NAME))
                return 0

            loss_batch, _ = sess.run([loss, optimizer],
                                     feed_dict={center_words: centers, target_words: targets})
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0

        projector.visualize_embeddings(writer, config)
        writer.close()

def word2vec_look_up(word, dictionary): #get embeded word vector
    tf.reset_default_graph()
    embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0),
                               name='embed_matrix')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(LOG_DIR, CKPT_NAME))
        embed = tf.nn.embedding_lookup(embed_matrix, dictionary[word], name='embed')
        return embed
def main():
    for echo in range(1,ECHO_TIME):
        batch_gen, dictionary = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
        word2vec(batch_gen, echo)


if __name__ == '__main__':
    main()
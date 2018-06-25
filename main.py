import os
import pprint
import tensorflow as tf

from data import read_data
from model import MemN2N
from sklearn import model_selection
import numpy as np
pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 128, "internal state dimension [150]")
# flags.DEFINE_integer("lindim", 150, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 3, "number of hops [6]")
flags.DEFINE_integer("mem_size", 50,"memory size [100]")
flags.DEFINE_integer("sent_size", 8,"sentence size [20]")
flags.DEFINE_integer("batch_size", 32, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
# flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.1, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 40, "clip gradients to this norm [50]")
flags.DEFINE_string("data_dir", "data", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_string("data_name", "qa2", "data set name [ptb]/[qa]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")
flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
FLAGS = flags.FLAGS

def main(_):

    word2idx = {}

    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)

    train_data,test_data = read_data('%s/%s.' % (FLAGS.data_dir, FLAGS.data_name), word2idx,FLAGS)
    train_data,valid_data = model_selection.train_test_split(train_data, test_size=.1)


    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    for i in range(FLAGS.mem_size):
        word2idx['time{}'.format(i + 1)] = 'time{}'.format(i + 1)
    FLAGS.nwords = len(word2idx)
    print ('train data len:',len(train_data))
    print ('valid data len:', len(valid_data))
    print ('voca len:',len(word2idx))
    print ('story sample:', np.array(train_data[0][0]))
    # pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()

        if FLAGS.is_test:
            model.run(valid_data, test_data,idx2word,FLAGS)
        else:
            model.run(train_data, valid_data,idx2word,FLAGS)

if __name__ == '__main__':
    tf.app.run()

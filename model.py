import os
import math
import random
import numpy as np
import tensorflow as tf
from sklearn import metrics
# from past.builtins import xrange
import pdb


class MemN2N(object):
    def __init__(self, config, sess):
        self.nwords = config.nwords
        # self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.sent_size = config.sent_size
        self.mem_size = config.mem_size
        # self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        self.input = tf.placeholder(tf.int32, [None, self.sent_size], name="input")
        self.time = tf.placeholder(tf.int32, [None, self.mem_size, self.sent_size], name="time")
        self.target = tf.placeholder(tf.float32, [self.batch_size, self.nwords], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size, self.sent_size], name="context")

        self.hid = []

        self.share_list = []
        self.share_list.append([])

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step",trainable=False)

        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std),name='A')
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std),name='B')
        # self.C = tf.Variable(tf.random_normal([self.batch_size, self.mem_size, self.edim, 1], stddev=self.init_std),name='C')

        # Temporal Encoding
        self.T_A = tf.Variable(tf.random_normal([self.mem_size * self.sent_size, self.edim], stddev=self.init_std),name='T_A')
        # self.T_B = tf.Variable(tf.random_normal([self.mem_size * self.sent_size, self.edim], stddev=self.init_std),name='T_B')

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.nn.embedding_lookup(self.A, self.context)
        Ain_t = tf.nn.embedding_lookup(self.T_A, self.time)
        Ain = tf.add(Ain_c, Ain_t)

        # c_i = sum B_ij * u + T_B_i
        Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        Bin_t = tf.nn.embedding_lookup(self.T_A, self.time)
        Bin = tf.add(Bin_c, Bin_t)

        Qin = tf.nn.embedding_lookup(self.A, self.input)

        # pdb.set_trace()
        Qin = tf.expand_dims(Qin, 1)
        Qin= tf.tile(Qin,[1,self.mem_size,1,1])
        self.hid.append(Qin)

        # pdb.set_trace()

        # Bin = tf.reduce_sum(Bin, axis=2)
        # Ain = tf.reduce_sum(Ain, axis=2)  # for count the sents in memory

        # Ain_sents=tf.reduce_sum(Ain,axis=1) for #count the words in each sentences
        # pdb.set_trace()
        for h in xrange(self.nhop):
            self.hid3dim = self.hid[-1]  # tf.reshape(self.hid[-1], [-1, 1, self.edim])
            Aout = tf.matmul(self.hid3dim, Ain, adjoint_a=True)
            # Aout2dim = Aout  # tf.reshape(Aout, [-1, self.mem_size])
            # P = tf.nn.softmax(Aout2dim)

            # probs3dim = tf.transpose(self.hid3dim, [0,2,1,3])
            # Bin_=tf.transpose(Bin,[0,2,1,3])
            Bout = tf.matmul(Bin,Aout)
            # Bout2dim = Bout  # tf.reshape(Bout, [-1, self.edim])

            # A_B=tf.concat([Aout,Bout],axis=1)

            # Allout = tf.matmul(A_B, self.C)
            # Allout=tf.squeeze(Allout)

            # Dout = tf.add(Cout, Bout2dim)

            # self.share_list[0].append(Cout)

            # if self.lindim == self.edim:
            self.hid.append(Bout)
            # elif self.lindim == 0:
            #     self.hid.append(tf.nn.relu(Dout))
            # else:
            #     F = tf.slice(Dout, [0, 0, 0], [self.batch_size, self.sent_size, self.lindim])
            #     G = tf.slice(Dout, [0, 0, self.lindim], [self.batch_size, self.sent_size, self.edim - self.lindim])
            #     K = tf.nn.relu(G)
            #     self.hid.append(tf.concat(axis=2, values=[F, K]))

    def build_model(self):
        self.build_memory()
        # pdb.set_trace()
        out_hid = self.hid[-1]
        # out_hid = tf.reduce_sum(out_hid, axis=1) #need to be modified
        # self.hid2word = tf.Variable(tf.random_normal([self.batch_size,self.mem_size,self.sent_size, 1]))
        # out_hid = tf.matmul(out_hid, self.hid2word,transpose_a=True)
        # out_hid=tf.squeeze(out_hid)
        # self.hid2word2 = tf.Variable(tf.random_normal([self.batch_size, self.mem_size, 1]))
        # out_hid = tf.matmul(out_hid, self.hid2word2, transpose_a=True)

        with tf.variable_scope('cnn'):
            def _variable_on_cpu(name, shape, initializer):
                """Helper to create a Variable stored on CPU memory.
                Args:
                  name: name of the variable
                  shape: list of ints
                  initializer: initializer for Variable
                Returns:
                  Variable Tensor
                """
                with tf.device('/cpu:0'):
                    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
                return var

            def _variable_with_weight_decay(name, shape, stddev, wd):
                """Helper to create an initialized Variable with weight decay.
                Note that the Variable is initialized with a truncated normal distribution.
                A weight decay is added only if one is specified.
                Args:
                  name: name of the variable
                  shape: list of ints
                  stddev: standard deviation of a truncated Gaussian
                  wd: add L2Loss weight decay multiplied by this float. If None, weight
                      decay is not added for this Variable.
                Returns:
                  Variable Tensor
                """
                var = _variable_on_cpu(
                    name,
                    shape,
                    tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
                # if wd is not None:
                #     weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
                    # tf.add_to_collection('losses', weight_decay)
                return var



            context=out_hid
            # pdb.set_trace()
            # conv1
            with tf.variable_scope('conv1') as scope:
                kernel = _variable_with_weight_decay('weights',
                                                     shape=[2, 2, 128, 64],
                                                     stddev=5e-2,
                                                     wd=None)
                # pdb.set_trace()
                conv = tf.nn.conv2d(context, kernel, [1, 2, 2, 1], padding='SAME')
                biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
                pre_activation = tf.nn.bias_add(conv, biases)
                conv1 = tf.nn.relu(pre_activation, name=scope.name)

            # pool1
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 3, 2, 1],
                                   padding='SAME', name='pool1')
            # norm1
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm1')
            # conv2
            with tf.variable_scope('conv2') as scope:
                kernel = _variable_with_weight_decay('weights',
                                                     shape=[2, 2, 64, 20],
                                                     stddev=5e-2,
                                                     wd=None)
                conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='SAME')
                biases = _variable_on_cpu('biases', [20], tf.constant_initializer(0.1))
                pre_activation = tf.nn.bias_add(conv, biases)
                conv2 = tf.nn.relu(pre_activation, name=scope.name)


            # norm2
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
            # pool2
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                                   strides=[1, 1, 1, 1], padding='SAME', name='pool2')
            # pdb.set_trace()



        out_hid=tf.reshape(pool2,[self.batch_size,-1])
        v_d=int(out_hid.get_shape()[-1])
        self.W = tf.Variable(tf.random_normal([v_d, self.nwords], stddev=self.init_std),name='W')
        z = tf.matmul(out_hid, self.W)
        self.pred = z
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=self.target)

        self.lr = tf.Variable(self.current_lr,trainable=False)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        # params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W ]
        grads_and_vars = self.opt.compute_gradients(self.loss)
        # pdb.set_trace()
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                  for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def train(self, data,idx2word=None):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.sent_size], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size, self.sent_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords])  # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size, self.sent_size])

        # x.fill(self.init_hid)

        for t in xrange(self.mem_size):
            for s in xrange(self.sent_size):
                time[:, t, s].fill(t * self.sent_size + s)

        # pdb.set_trace()
        targets, predicts = [], []
        for idx in range(N):

            target.fill(0)
            for b in range(self.batch_size):
                m = random.randrange(0, len(data))
                context[b] = data[m][0]
                x[b] = data[m][1]
                target[b][data[m][2][0]] = 1
            # pdb.set_trace()
            _, loss, self.step, predict_ = self.sess.run([self.optim,
                                                          self.loss,
                                                          self.global_step, self.pred],
                                                         feed_dict={
                                                             self.input: x,
                                                             self.time: time,
                                                             self.target: target,
                                                             self.context: context})
            cost += np.sum(loss)
            predicts.extend(np.argmax(predict_, 1))
            targets.extend(np.argmax(target, 1))
        # if self.show: bar.finish()
        accuracy = metrics.accuracy_score(targets, predicts)
        return cost / N / self.batch_size, accuracy

    def test(self, data, label='Test'):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.sent_size], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size, self.sent_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords])  # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size, self.sent_size])

        # x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:, t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=N)

        m = 0
        targets, predicts = [], []
        for idx in xrange(N):

            target.fill(0)
            for b in xrange(self.batch_size):
                context[b] = data[m][0]
                x[b] = data[m][1]
                target[b][data[m][2][0]] = 1
                m += 1

                if m >= len(data):
                    m = self.mem_size

            loss, pred = self.sess.run([self.loss, self.pred], feed_dict={self.input: x,
                                                                          self.time: time,
                                                                          self.target: target,
                                                                          self.context: context})
            cost += np.sum(loss)
            predicts.extend(np.argmax(pred, 1))
            targets.extend(np.argmax(target, 1))

        # pdb.set_trace()
        accuracy = metrics.accuracy_score(targets, predicts)
        return cost / N / self.batch_size, accuracy

    def run(self, train_data, test_data,idx2word):
        if not self.is_test:
            for idx in xrange(self.nepoch):
                train_loss, train_acc = self.train(train_data,idx2word=idx2word)
                test_loss, test_acc = self.test(test_data, label='Validation')
                train_losses = np.sum(train_loss)
                test_losses = np.sum(test_loss)

                # Logging
                # self.log_aploss.append([train_losses, test_losses])
                # self.log_perp.pend([math.exp(train_losses), math.exp(test_losses)])

                state = {
                    'epoch': idx,
                    'lr': self.current_lr,
                    'train loss': train_losses,
                    'valid loss': test_losses,  # math.exp(test_loss),
                    'valid acc:': test_acc,
                    'train acc:': train_acc
                }
                print(state)

                # Learning rate annealing
                # if len(self.log_loss) > 5 and self.log_loss[idx][1] > self.log_loss[idx - 1][1] * 0.9999:
                #     self.current_lr = self.current_lr / 1.5
                #     self.lr.assign(self.current_lr).eval()
                # if self.current_lr < 1e-5: break

                if idx % 50 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step=self.step.astype(int))
        else:
            self.load()
            valid_loss, valid_acc = self.train(train_data)
            test_loss, test_acc = self.test(test_data, label='Validation')
            # valid_loss = np.sum(self.test(train_data, label='Validation'))
            # test_loss = np.sum(self.test(test_data, label='Test'))

            state = {
                'valid loss': valid_loss,
                'test loss': test_loss,
                'test accuracy': test_acc
            }
            print(state)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")

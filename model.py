import os
import math
import random
import numpy as np
import tensorflow as tf
from sklearn import metrics
# from past.builtins import xrange
import pdb

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope( name, "zero_nil_slot",[t]) as name:
        # pdb.set_trace()
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)


def position_encoding( sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    # ls = sentence_size + 1
    # le = embedding_size + 1
    # for i in range(1, le):
    #     for j in range(1, ls):
    #         encoding[i - 1, j - 1] = (i - (embedding_size + 1) / 2) * (j - (sentence_size + 1) / 2)
    # encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # # Make position encoding of time words identity to avoid modifying them
    # encoding[:, -1] = 1.0
    return np.transpose(encoding)

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
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim],stddev=0.1),name='A')
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=0.1),name='B')
        # self.C = tf.Variable(tf.random_normal([self.batch_size, self.mem_size, self.edim, 1], stddev=self.init_std),name='C')
        # self.C = position_encoding(self.mem_size * self.sent_size, self.edim)
        # Temporal Encoding
        self.T_A = tf.Variable(tf.random_normal([self.mem_size * self.sent_size, self.edim], stddev=0.1),name='T_A',trainable=False)
        self.T_B = tf.Variable(tf.random_normal([self.mem_size * self.sent_size, self.edim], stddev=0.1),name='T_B',trainable=False)
        # self.T_A = position_encoding(self.mem_size * self.sent_size,self.edim)
        # self.T_B = position_encoding(self.mem_size * self.sent_size, self.edim)
        # m_i = sum A_ij * x_ij + T_A_i
        # pdb.set_trace()
        self._nil_vars = set([self.A.name]+[self.B.name])#+[self.B.name])
        Ain_c = tf.nn.embedding_lookup(self.A, self.context)
        Ain_t = tf.nn.embedding_lookup(self.T_A, self.time)
        Ain = Ain_c * Ain_t
        Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
        Bin = Bin_c * Bin_t
        # c_i = sum B_ij * u + T_B_i
        # Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        # Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
        # Bin = Bin_c*Bin_t

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
        for hop in xrange(self.nhop):


            self.hid3dim = self.hid[-1]  # tf.reshape(self.hid[-1], [-1, 1, self.edim])
            Aout = tf.matmul(self.hid3dim, Ain, adjoint_a=True)
            Aout2dim = Aout  # tf.reshape(Aout, [-1, self.mem_size])
            P = tf.nn.softmax(Aout2dim)


            # probs3dim = tf.transpose(self.hid3dim, [0,2,1,3])
            # Bin_=tf.transpose(Bin,[0,2,1,3])
            Bout = tf.matmul(Bin,P)
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

    def _inference(self, stories, queries):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        with tf.variable_scope('inference'):

            self._init=tf.random_normal_initializer(stddev=0.1)
            # nil_word_slot = tf.zeros([1, self.edim])
            # A = tf.concat(axis=0, values=[ nil_word_slot, self._init([self.nwords-1, self.edim]) ])
            # C = tf.concat(axis=0, values=[ nil_word_slot, self._init([self.nwords-1, self.edim]) ])
            A = tf.random_normal([self.nwords, self.edim], stddev=0.1)
            C = tf.random_normal([self.nwords, self.edim], stddev=0.1)
            self.A_1 = tf.Variable(A, name="A")

            # Use A_1 for thee question embedding as per Adjacent Weight Sharing
            # self.A_1=tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std),name='A_1')
            self._encoding=tf.constant(position_encoding(self.sent_size,self.edim))
            # C=tf.random_normal([self.nwords, self.edim], stddev=self.init_std)
            self.C = []
            for hopn in range(self.nhop):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    self.C.append(tf.Variable(C, name="C"))

            q_emb = tf.nn.embedding_lookup(self.A_1, queries)
            u_0 = tf.reduce_sum(q_emb * self._encoding, 1)
            u = [u_0]
            self._nil_vars = set([self.A_1.name] + [x.name for x in self.C])
            for hopn in range(self.nhop):
                if hopn == 0:
                    m_emb_A = tf.nn.embedding_lookup(self.A_1, stories)
                    m_A = tf.reduce_sum(m_emb_A * self._encoding, 2)

                else:
                    with tf.variable_scope('hop_{}'.format(hopn - 1)):
                        m_emb_A = tf.nn.embedding_lookup(self.C[hopn - 1], stories)
                        m_A = tf.reduce_sum(m_emb_A * self._encoding, 2)

                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m_A * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                with tf.variable_scope('hop_{}'.format(hopn)):
                    m_emb_C = tf.nn.embedding_lookup(self.C[hopn], stories)
                m_C = tf.reduce_sum(m_emb_C * self._encoding, 2)

                c_temp = tf.transpose(m_C, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                # Dont use projection layer for adj weight sharing
                # u_k = tf.matmul(u[-1], self.H) + o_k

                u_k = u[-1] + o_k

                u.append(u_k)

            # Use last C for output (transposed)
            with tf.variable_scope('hop_{}'.format(self.nhop)):
                return tf.matmul(u_k, tf.transpose(self.C[-1], [1, 0]))

    def build_model(self):
        # logits=self._inference(self.context,self.input)
        self.build_memory()

        # pdb.set_trace()
        out_hid = self.hid[-1]
        # out_hid = tf.reduce_sum(out_hid, axis=1) #need to be modified
        # self.hid2word = tf.Variable(tf.random_normal([self.batch_size,self.mem_size,self.sent_size, 1]))
        # out_hid = tf.matmul(out_hid, self.hid2word,transpose_a=True)
        # out_hid=tf.squeeze(out_hid)
        # self.hid2word2 = tf.Variable(tf.random_normal([self.batch_size, self.mem_size, 1]))
        # out_hid = tf.matmul(out_hid, self.hid2word2, transpose_a=True)

        # with tf.variable_scope('cnn'):
        #     def _variable_on_cpu(name, shape, initializer):
        #         """Helper to create a Variable stored on CPU memory.
        #         Args:
        #           name: name of the variable
        #           shape: list of ints
        #           initializer: initializer for Variable
        #         Returns:
        #           Variable Tensor
        #         """
        #         with tf.device('/cpu:0'):
        #             var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        #         return var
        #
        #     def _variable_with_weight_decay(name, shape, stddev, wd):
        #         """Helper to create an initialized Variable with weight decay.
        #         Note that the Variable is initialized with a truncated normal distribution.
        #         A weight decay is added only if one is specified.
        #         Args:
        #           name: name of the variable
        #           shape: list of ints
        #           stddev: standard deviation of a truncated Gaussian
        #           wd: add L2Loss weight decay multiplied by this float. If None, weight
        #               decay is not added for this Variable.
        #         Returns:
        #           Variable Tensor
        #         """
        #         var = _variable_on_cpu(
        #             name,
        #             shape,
        #             tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
        #         # if wd is not None:
        #         #     weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        #             # tf.add_to_collection('losses', weight_decay)
        #         return var
        #
        #
        #
        #     context=out_hid
        #     # pdb.set_trace()
        #     # conv1
        #     with tf.variable_scope('conv1') as scope:
        #         kernel = _variable_with_weight_decay('weights',
        #                                              shape=[2, 2, 128, 64],
        #                                              stddev=5e-2,
        #                                              wd=None)
        #         # pdb.set_trace()
        #         conv = tf.nn.conv2d(context, kernel, [1, 2, 2, 1], padding='SAME')
        #         biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        #         pre_activation = tf.nn.bias_add(conv, biases)
        #         conv1 = tf.nn.relu(pre_activation, name=scope.name)
        #
        #     # pool1
        #     pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 3, 2, 1],
        #                            padding='SAME', name='pool1')
        #     # norm1
        #     norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                       name='norm1')
            # conv2
            # with tf.variable_scope('conv2') as scope:
            #     kernel = _variable_with_weight_decay('weights',
            #                                          shape=[2, 2, 64, 20],
            #                                          stddev=5e-2,
            #                                          wd=None)
            #     conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='SAME')
            #     biases = _variable_on_cpu('biases', [20], tf.constant_initializer(0.1))
            #     pre_activation = tf.nn.bias_add(conv, biases)
            #     conv2 = tf.nn.relu(pre_activation, name=scope.name)
            #
            #
            # # norm2
            # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            #                   name='norm2')
            # # pool2
            # pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
            #                        strides=[1, 1, 1, 1], padding='SAME', name='pool2')
            # # pdb.set_trace()



        out_hid=tf.reshape(out_hid,[self.batch_size,-1])
        v_d=int(out_hid.get_shape()[-1])
        self.W = tf.Variable(tf.random_normal([v_d, self.nwords], stddev=self.init_std),name='W')
        z = tf.matmul(out_hid, self.W)
        self.pred = z


        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=tf.cast(self.target, tf.float32))
        cross_entropy_sum = tf.reduce_sum(self.loss, name="cross_entropy_sum")
        self.lr = tf.Variable(self.current_lr,trainable=False)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        # params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W ]
        grads_and_vars = self.opt.compute_gradients(cross_entropy_sum)
        # pdb.set_trace()
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                  for gv in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in clipped_grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(nil_grads_and_vars)

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
            # pdb.set_trace()
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

    def run(self, train_data, test_data,idx2word,FLAGS):
        if not self.is_test:
            for idx in xrange(self.nepoch):
                train_loss, train_acc = self.train(train_data,idx2word=idx2word)
                test_loss, test_acc = self.test(test_data, label='Validation')
                # train_losses = np.sum(train_loss)
                # test_losses = np.sum(test_loss)

                # Logging
                # self.log_aploss.append([train_losses, test_losses])
                # self.log_perp.pend([math.exp(train_losses), math.exp(test_losses)])

                state = {
                    'epoch': idx,
                    'lr': self.current_lr,
                    # 'train loss': train_losses,
                    # 'valid loss': test_losses,  # math.exp(test_loss),
                    'valid acc:': test_acc,
                    'train acc:': train_acc
                }
                print(state)

                # Learning rate annealing
                if idx - 1 <= FLAGS.anneal_stop_epoch:
                    anneal = 2.0 ** ((idx - 1) // FLAGS.anneal_rate)
                else:
                    anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
                self.current_lr = FLAGS.init_lr / anneal
                self.lr.assign(self.current_lr).eval()

                # if idx > 25 and idx<100:
                #     current_lrself. = self.current_lr * 0.8
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

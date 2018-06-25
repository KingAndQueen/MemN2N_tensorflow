import os
from collections import Counter
import re
import pdb


def tokenize(sent, vocab=None):
    if vocab is None:
        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    else:
        words = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
        indexs = [vocab[word] for word in words]
        # pdb.set_trace()
        return indexs

def read_data(fname, word2idx,FLAGS):

    fname_train=fname+'train.txt'
    fname_test=fname+'test.txt'
    if os.path.isfile(fname_train) and os.path.isfile(fname_test):
        train=open(fname_train)
        lines_train = train.readlines()
        test=open(fname_test)
        lines_test=test.readlines()
    else:
        raise ("[!] Data %s not found" % fname)

    words = []
    for line in lines_train:
        words.extend(tokenize(line))  # .split())
    for line in lines_test:
        words.extend(tokenize(line))


    # count.extend(Counter(words).most_common())

    if len(word2idx) == 0:
        # word2idx['<eos>'] = 0
        word2idx['<pad>'] = 0
    for word in words:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    # pdb.set_trace()
    data_train=parse_data(lines_train,word2idx,FLAGS)
    data_test=parse_data(lines_test,word2idx,FLAGS)
    print("Read %s words from %s" % (len(data_train), fname_train))
    print("Read %s words from %s" % (len(data_test), fname_test))
    return data_train,data_test

def parse_data(lines,word2idx,FLAGS):
    data = list()
    for line in lines:

        nid, line = line.split(' ', 1)
        #	print nid #close vim then run or will error
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:  # question
            q,  ans, supporting = line.split('\t')
            q = tokenize(q, word2idx)
            pad_len=max(FLAGS.sent_size-len(q),0)
            for x in range(pad_len): q.append(word2idx['<pad>'])
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = tokenize(ans, word2idx)

            # substory = None
            # remove question marks
            if q[-1] == word2idx["?"]:
                q = q[:-1]

            substory = [x for x in story if x]
            if len(story) >= FLAGS.mem_size:
                substory=substory[-FLAGS.mem_size:]
            for i,substory_ in enumerate(substory):
                pad_len = max(FLAGS.sent_size - len(substory_), 0)
                for x in range(pad_len): substory_.append(word2idx['<pad>'])
                assert len(substory_)==FLAGS.sent_size
                # pdb.set_trace()
                substory_[-1] = len(word2idx) + FLAGS.mem_size + i - len(substory)

            pad_len=max(FLAGS.mem_size-len(substory),0)
            for x in range(pad_len):substory.append(FLAGS.sent_size*[word2idx['<pad>']])
            # pdb.set_trace()
            assert len(q)==FLAGS.sent_size
            assert len(substory)==FLAGS.mem_size
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line, word2idx)
            if sent[-1] == word2idx["."]:
                sent = sent[:-1]
            story.append(sent)
    # pdb.set_trace()

    return data

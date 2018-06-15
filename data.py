import os
from collections import Counter
import re
import pdb


def read_data(fname, count, word2idx,FLAGS):
    def tokenize(sent, vocab=None):
        if vocab is None:
            return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
        else:
            words = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
            indexs = [vocab[word] for word in words]
            # pdb.set_trace()
            return indexs

    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise ("[!] Data %s not found" % fname)

    words = []
    for line in lines:
        words.extend(tokenize(line))  # .split())

    if len(count) == 0:
        count.append(['<eos>', 0])
        count.append(['<pad>', 1])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0
        word2idx['<pad>'] = 1
    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    # pdb.set_trace()
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
            for substory_ in substory:
                pad_len = max(FLAGS.sent_size - len(substory_), 0)
                for x in range(pad_len): substory_.append(word2idx['<pad>'])
                assert len(substory_)==FLAGS.sent_size
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
    print("Read %s words from %s" % (len(data), fname))
    return data

import os
import codecs
from gensim.models import FastText


model = FastText(result, size=100, window=5, min_count=5, workers=4, sg=1)

#read file and extract text and make file for fasttext
#save as one document per line
#file path for train text

base_path = '/home/ailsb/PycharmProjects/test/data'
fns_train = [base_path + '/train/train.sent_data.txt', base_path + '/train/ratings_train.txt']


all_texts = []
all_targets = []

for num, fn in enumerate(fns_train):
    fn = os.path.join(os.path.dirname("__file__"), fn)
    with codecs.open(fn, 'r', encoding='utf-8') as f:
        data = [sent.rstrip('\n').split('\t') for sent in f.readlines()]
        all_texts += [x[1] for x in data]
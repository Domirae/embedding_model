import os
import codecs
from glove import Glove
from glove import Corpus
import operator
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def load_all_files_int(fns):
    all_texts = []
    all_targets = []

    for num, fn in enumerate(fns):
        fn = os.path.join(os.path.dirname("__file__"), fn)
        with codecs.open(fn, 'r', encoding='utf-8') as f:
            data = [sent.rstrip('\n').split('\t') for sent in f.readlines()]

        for x in data:
            if num == 0:
                if x[0] == 'NEG':
                    all_targets.append(0)
                else:
                    all_targets.append(1)
            else:
                if x[2] == '0':
                    all_targets.append(0)
                else:
                    all_targets.append(1)

        all_texts += [x[1] for x in data]

    return all_targets, all_texts


# read file and make corpus
def make_corpus(fns):
    print("[1.Read file and Making Corpus]")
    corpus = []

    for num, fn in enumerate(fns):
        fn = os.path.join(os.path.dirname("__file__"), fn)
        with codecs.open(fn, 'r', encoding='utf-8') as f:
            data = [sent.rstrip('\n').split('\t') for sent in f.readlines()]

        corpus += [x[1] for x in data]

    print("[finish]")
    return corpus

def read_corpus(data):

    delchars = [chr(c) for c in range(256)]
    delchars = [x for x in delchars if not x.isalnum()]
    delchars.remove(' ')
    delchars = ''.join(delchars)

    for text in data:
        yield text.lower().translate(str.maketrans('','',delchars)).split(' ')

# make dic
def make_dic(corpus):
    print("[2. Make Dict] make cooccurrence Matrix")
    vectorizer = CountVectorizer(min_df=10, ngram_range=(1, 1))
    X = vectorizer.fit_transform(corpus)
    Xc = X.T * X
    Xc.setdiag(0)
    result = Xc.toarray()
    dic = {}
    for idx1, word1 in enumerate(result):
        tmpdic = {}
        for idx2, word2 in enumerate(word1):
            if word2 > 0:
                tmpdic[idx2] = word2
        dic[idx1] = tmpdic


    vocab = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
    vocab = [word[0] for word in vocab]

    print("[finish]")
    return vocab,dic

if __name__ == '__main__':

    fns_train = ['/home/ailsb/PycharmProjects/test/data/train/train.sent_data.txt', '/home/ailsb/PycharmProjects/test/data/train/ratings_train.txt']
    fns_test = ['data/test/test.sent_data.txt', 'data/test/ratings_test.txt']

    train_targets, all_train_texts = load_all_files_int(fns_train)

    #make corpus data

    if os.path.isfile('./data/corpus.model'):
        corpus_model = Corpus.load('./data/corpus.model')
        print('Dict size: %s' % len(corpus_model.dictionary))
        print('Collocations: %s' % corpus_model.matrix.nnz)

    else:
        get_data = read_corpus
        corpus_model = Corpus()
        corpus_model.fit(get_data(all_train_texts), window=10)
        os.chdir("/home/ailsb/PycharmProjects/test/glove_dir")
        corpus_model.save('./data/corpus.model')

        print('Dict size: %s' % len(corpus_model.dictionary)    )
        print('Collocations: %s' % corpus_model.matrix.nnz)


    #training

    print(corpus_model.matrix)

    glove = Glove(no_components=300, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=150,no_threads=4, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    glove.save('./data/glove300.model')


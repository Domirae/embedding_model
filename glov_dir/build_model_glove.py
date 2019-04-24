import os
import codecs
from glove import Glove
import operator
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# read file and make corpus
def make_corpus(fns):
    print("[1.Read file and Making Corpus]")
    corpus = []

    for num, fn in enumerate(fns):
        fn = os.path.join(os.path.dirname("__file__"), fn)
        with codecs.open(fn, 'r', encoding='utf-8') as f:
            data = [sent.rstrip('\n').split('\t') for sent in f.readlines()]
        corpus += [x[1] for x in data]
    print(corpus)
    print("[finish]")
    return corpus

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

    #make glove data
    corpus = make_corpus(fns_train)
    vocab , dic =make_dic(corpus)


    # word2vec data
    # model = gensim.models.Word2Vec(size=300, sg=1, alpha=0.025, min_alpha=0.025, seed=1234, iter=150)
    # model.build_vocab(tokens)
    # model.train(tokens, epochs=model.iter, total_examples=model.corpus_count)

    model = Glove(dic,no_components=100, learning_rate=0.05)

    for epoch in range(150):
        err = model.train(batch_size=200, workers=4)
        print("epoch %d, error %.3f" % (epoch, err), flush=True)

    # 단어벡터 추출
    wordvectors = model.W

    os.chdir("/home/ailsb/PycharmProjects/test/glove_dir/data")
    model.save('glove_model2.bin')

    # 저장
    with open('glove', 'wb') as f:
        pickle.dump([vocab, wordvectors], f)


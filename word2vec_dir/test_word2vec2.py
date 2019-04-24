import os
from konlpy.tag import Okt
import tensorflow as tf
import numpy as np
import codecs
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

os.chdir("/home/ailsb/PycharmProjects/test/data")

def read_data(filenames):
    print("* Read data start *")
    datas = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[1:]  # header 제외 #
            datas.extend(data)

    print(len(datas))
    return datas

def tokenize(doc):
    okt = Okt()
    results = []
    for token in okt.pos(doc):
        if not token[1] in ["Josa", "Punctuation"]:
            results.append(token[0])
    return results

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

if __name__ =="__main__":

    data_path = 'npy/data.npy'
    # base_path = '/home/ailsb/PycharmProjects/test/data'
    #
    # fns_train = [base_path + '/train/train.sent_data.txt', base_path + '/train/ratings_train.txt']
    # fns_test = [base_path + '/test/test.sent_data.txt', base_path + '/test/ratings_test.txt']
    #
    # # 데이터 추출
    # a,all_train_texts = load_all_files_int(fns_train)
    # tokenizer_obj = Tokenizer()
    # tokenizer_obj.fit_on_texts(all_train_texts)
    # print(tokenizer_obj.)


    if os.path.isfile(data_path):
        tokens = np.load(data_path)
        print(len(tokens))

    else:
        datas = read_data(['train/train.sent_data.txt','train/ratings_train.txt'])

        tokens = []
        for row in datas:
            tokens.append(tokenize(row[1]))
        print(tokens)
        np.save('/home/ailsb/PycharmProjects/test/data/npy/data.npy',tokens)
        print('finish')


    model = gensim.models.Word2Vec(size=300, sg=1, alpha=0.025, min_alpha=0.025, seed=1234,iter=150)
    model.build_vocab(tokens)
    model.train(tokens,epochs=model.iter,total_examples=model.corpus_count)


    os.chdir("/home/ailsb/PycharmProjects/test/data/processed_file")
    model.save('Word2vec_review3.model')


    result = model.most_similar('재미/Noun', topn=10)  ## topn = len(model.wv.vocab)
    print(result)


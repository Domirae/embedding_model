import os
from konlpy.tag import Okt
import numpy as np
from gensim.models import FastText


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

if __name__ =="__main__":

    data_path = '/home/ailsb/PycharmProjects/test/data/npy/data4.npy'

    if os.path.isfile(data_path):
        tokens = np.load(data_path)
        print(len(tokens))

    else:
        datas = read_data(['/home/ailsb/PycharmProjects/test/data/train/train.sent_data.txt', '/home/ailsb/PycharmProjects/test/data/train/ratings_train.txt'])

        tokens = []
        for row in datas:
            tokens.append(tokenize(row[1]))
        print(tokens)
        np.save('/home/ailsb/PycharmProjects/test/data/npy/data.npy', tokens)
        print('finish')

    print('model train start')
    model = FastText(size=100, sg=1, alpha=0.025, min_alpha=0.025, seed=1234, iter=150)
    model.train(tokens, epochs=model.iter, total_examples=model.corpus_count)

    os.chdir("/home/ailsb/PycharmProjects/test/fastText_dir/data")
    model.save('fastText_model_okt_ver2.model')

    result = model.most_similar('재미', topn=10)  ## topn = len(model.wv.vocab)
    print(result)


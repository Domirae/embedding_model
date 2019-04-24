import os
import codecs
from konlpy.tag import Okt
from konlpy.tag import Komoran
import re

import json
from pprint import pprint
import nltk
import numpy as np

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


def normalize(data):
    return [sent.upper() for sent in data]

# 형태소 분석을 통해 품사를 태깅하는 작업
def tokenize_okt(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    okt = Okt()
    result = []
    for token in okt.pos(doc):
        if not token[1] in ["Josa", "Punctuation"]:
            result.append('/'.join(token))
    return result

#komoran 형태소 분석을 이용하여 토큰화
def tokenize_komoran(doc):

    komoran = Komoran()
    result = []

    for token in komoran.pos(doc):
        if not token[1] in ['SP','SF','SE','SO']:
            result.append('/'.join(token))

    return result

def data_processing(str, targets, texts):
    # 데이터 길이 출력
    print("[%s] target 데이터 길이:%d / text 데이터 길이:%d" % (str, len(targets), len(texts)))

    path = str + '_docs_ver2.json'

    if os.path.isfile(path):
        with open(path) as f:
            docs = json.load(f)
    else:
        print("---tokenizing start---")
        docs = [(tokenize_okt(text), targets[num]) for num, text in enumerate(texts)]
        print(docs)

        # JSON 파일로 저장
        with open(path, 'w', encoding="utf-8") as make_file:
            json.dump(docs, make_file, ensure_ascii=False, indent="\t")
    return docs

def selectWords(docs):
    print('-----[train selectWords result]-------')
    tokens = [t for d in docs for t in d[0]]
    nltk_texts = nltk.Text(tokens, name='NMSC')
    print("생성된 토큰 길이:%d / 중복제외 토큰: %d" % (len(tokens), len(set(nltk_texts.tokens))))
    # print("출현 빈도가 높은 상위 토큰 10개")
    # pprint(nltk_texts.vocab().most_common(10))
    select_words = [f[0] for f in nltk_texts.vocab().most_common(6500)]
    np.save("select_word.npy",select_words)
    print("-----[train selectWords finish]-------")
    return select_words


def term_frequency(doc, selected_words):
    return [doc.count(word) for word in selected_words]

#문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어 BOW(Bag of Words) 인코딩한 벡터를 만드는 역할
def countVectorization(docs, select_words):

    data_path='./data/npy/data_countVec.npy'
    label_path ='./data/npy/label_countVec.npy'

    if os.path.isfile(data_path) and os.path.isfile(label_path):
        print("hi")
        x = np.load(data_path)
        y = np.load(label_path)
    else:
        print("-----[start count Vectorization]-------")
        x_x = [term_frequency(d, select_words) for d, _ in docs]
        y_y = [c for _, c in docs]
        x = np.asarray(x_x).astype('float32')
        y = np.asarray(y_y).astype('float32')

        np.save(data_path,x)
        np.save(label_path,y)
        print("-----[finish count Vectorization]-------")
    return x, y

def processing():

    fns_train = ['data/train/train.sent_data.txt', 'data/train/ratings_train.txt']
    fns_test = ['data/test/test.sent_data.txt', 'data/test/ratings_test.txt']

    # 데이터 추출
    train_targets, all_train_texts = load_all_files_int(fns_train)
    test_targets, all_test_texts = load_all_files_int(fns_test)

    # 데이터 정규화
    train_texts = normalize(all_train_texts)
    test_texts = normalize(all_test_texts)
    # test(train_targets,train_texts)

    train_docs = data_processing('data/train', train_targets, train_texts)
    test_docs = data_processing('data/test', test_targets, test_texts)

    select_words = selectWords(train_docs)

    train_x, train_y = countVectorization(train_docs, select_words)
    test_x, test_y = countVectorization(test_docs, select_words)
    print(train_x[0])

    return train_x,train_y,test_x,test_y

if __name__ == '__main__':

    processing()


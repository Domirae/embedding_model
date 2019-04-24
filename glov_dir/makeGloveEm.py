import os
import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
import numpy as np
import json
import nltk
from konlpy.tag import Okt
from glove import Glove

class GloveEM():

    def __init__(self,maxlen,max_words,embedding_dim):

        self.maxlen = maxlen
        self.max_words = max_words
        self.embedding_dim = embedding_dim


    def load_data(self,fns):
        print("start load data")
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


    def data_processing(self, str, targets, texts):
        # 데이터 길이 출력
        print("[%s] target 데이터 길이:%d / text 데이터 길이:%d" % (str, len(targets), len(texts)))

        path = str + '_docs_ver2.json'

        if os.path.isfile(path):
            with open(path) as f:
                docs = json.load(f)
        else:
            print("---tokenizing start---")
            docs = [(self.tokenize_okt(text), targets[num]) for num, text in enumerate(texts)]
            print(docs)

            # JSON 파일로 저장
            with open(path, 'w', encoding="utf-8") as make_file:
                json.dump(docs, make_file, ensure_ascii=False, indent="\t")
        return docs


    # 형태소 분석을 통해 품사를 태깅하는 작업
    def tokenize_okt(self,doc):
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        okt = Okt()
        result = []
        for token in okt.pos(doc):
            if not token[1] in ["Josa", "Punctuation"]:
                result.append('/'.join(token))
        return result

    def selectWords(self,docs):
        print('-----[train selectWords result]-------')
        tokens = [t for d in docs for t in d[0]]
        nltk_texts = nltk.Text(tokens, name='NMSC')
        print("생성된 토큰 길이:%d / 중복제외 토큰: %d" % (len(tokens), len(set(nltk_texts.tokens))))
        # print("출현 빈도가 높은 상위 토큰 10개")
        # pprint(nltk_texts.vocab().most_common(10))
        select_words = [f[0] for f in nltk_texts.vocab().most_common(6500)]
        np.save("select_word.npy", select_words)
        print("-----[train selectWords finish]-------")
        return select_words

    def tokenize(self, targets, texts):
        print("start tokenize")

        self.tokenizer = Tokenizer(num_words = self.max_words)
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)

        data  = pad_sequences(sequences, maxlen = self.maxlen)
        word_index = self.tokenizer.word_index

        print('%s개의 토큰이 발견이 되었습니다.'%len(word_index))
        labels = np.asarray(targets)

        print('데이터 텐서의 크기:', data.shape)
        print('레이블 텐서의 크기:',labels.shape)

        return data,labels,self.tokenizer

    def Convert2Vec(self,token):  ## Convert corpus into vectors
        print("start convert2vec")
        os.chdir("/home/ailsb/PycharmProjects/test/glove_dir/data")

        glove = Glove.load('glove.model')

        embedding_index = {}
        for w in glove.dictionary.keys():
            embedding_index[w] = glove.word_vectors[glove.dictionary[w]]

        embedding_matrix = np.zeros((self.max_words, self.embedding_dim))

        for word,i in token.word_index.items():
            if i >= self.max_words:
                continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def load_model(self,model_name):

        os.chdir("/home/ailsb/PycharmProjects/test/glove_dir/data")
        model = Glove.load(model_name)
        return model







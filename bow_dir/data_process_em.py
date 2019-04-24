from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import codecs
from konlpy.tag import Okt
import json

import nltk

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

def data_processing(str, texts):
    # 데이터 길이 출력
    print("[%s] text 데이터 길이:%d" % (str,  len(texts)))

    path = str + '_docs_ver.json'

    if os.path.isfile(path):
        with open(path) as f:
            docs = json.load(f)
    else:
        print("---tokenizing start---")
        docs = [(tokenize_okt(text),0) for num, text in enumerate(texts)]
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
    print("-----[train selectWords finish]-------")
    return nltk_texts.tokens

def processing():
    os.chdir('/home/ailsb/PycharmProjects/test')
    fns_train = ['data/train/train.sent_data.txt', 'data/train/ratings_train.txt']
    fns_test = ['data/test/test.sent_data.txt', 'data/test/ratings_test.txt']

    # 데이터 추출
    train_targets, all_train_texts = load_all_files_int(fns_train)
    test_targets, all_test_texts = load_all_files_int(fns_test)

    tokenizer_obj = Tokenizer()
    total_reviews = all_train_texts + all_test_texts
    tokenizer_obj.fit_on_texts(total_reviews)

    total_reviews_docs = data_processing('data/total', total_reviews)
    select_words = selectWords(total_reviews_docs)

    # pad sequnces
    max_length = max([len(s.split()) for s in total_reviews])
    print(max_length)

    # vocabulady size
    vocab_size1 = len(set(select_words)) + 1
    vocab_size2 = len(tokenizer_obj.word_index) + 1

    x_train_tokens = tokenizer_obj.texts_to_sequences(all_train_texts)
    x_test_tokens = tokenizer_obj.texts_to_sequences(all_test_texts)

    x_train_pad = pad_sequences(x_train_tokens, maxlen=max_length, padding='post')
    x_test_pad = pad_sequences(x_test_tokens, maxlen=max_length, padding='post')

    return vocab_size2, max_length , x_train_pad,train_targets,x_test_pad,test_targets



if __name__ == '__main__':

    fns_train = ['data/train/train.sent_data.txt', 'data/train/ratings_train.txt']
    fns_test = ['data/test/test.sent_data.txt', 'data/test/ratings_test.txt']

    # 데이터 추출
    train_targets, all_train_texts = load_all_files_int(fns_train)
    test_targets, all_test_texts = load_all_files_int(fns_test)



    tokenizer_obj = Tokenizer()
    total_reviews = all_train_texts + all_test_texts
    tokenizer_obj.fit_on_texts(total_reviews)

    total_reviews_docs = data_processing('data/total', total_reviews)
    select_words = selectWords(total_reviews_docs)

#     pad sequnces
    max_length = max([len(s.split()) for s in total_reviews])

#vocabulady size
    vocab_size1 = len(set(select_words))+1
    print(vocab_size1)
    vocab_size2 = len(tokenizer_obj.word_index) + 1
    print(vocab_size2)

    x_train_tokens = tokenizer_obj.texts_to_sequences(all_train_texts)
    x_test_tokens = tokenizer_obj.texts_to_sequences(all_test_texts)

    print(x_train_tokens[0],train_targets[0])


    x_train_pad = pad_sequences(x_train_tokens, maxlen= max_length,padding='post')
    x_test_pad = pad_sequences (x_test_tokens, maxlen= max_length,padding='post')










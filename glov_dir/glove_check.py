import gensim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
import os
from glove import Glove

os.chdir("/home/ailsb/PycharmProjects/test/glove_dir/data")

print('Loading pre-trained GloVe model')
glove = Glove.load('glove.model')

embedding_index = {}

for w in glove.dictionary.keys():
     embedding_index[w] = glove.word_vectors[glove.dictionary[w]]
#
print('finish')
print(embedding_index['흠너무'])






model = gensim.models.word2vec.Word2Vec.load('/home/ailsb/PycharmProjects/test/data/processed_file/Word2vec_review3.model')
#
# embedding_index = {}
# print(model.wv.vocab.keys())
#
# print('Querying for %s' % "재미")
# results = glove.most_similar("별로", number=10)
#
# word = []
# data = []
#
# for result in results:
#
#     word.append(result[0])
#     data.append(result[1])
#
#
# font_name = font_manager.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf").get_name()
# rc('font', family=font_name)
#
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
#
# ypos = np.arange(len(word))
# rects = plt.barh(ypos,data, align='center', height=0.5)
# plt.yticks(ypos, word)
# for i, rect in enumerate(rects):
#     ax.text(0.95 * rect.get_width(), rect.get_y() + rect.get_height() / 2.0, str(data[i]) + '%', ha='right', va='center')
# plt.show()
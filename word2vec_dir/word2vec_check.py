from gensim.models import word2vec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc

model = word2vec.Word2Vec.load('/home/ailsb/PycharmProjects/test/data/processed_file/Word2vec_review3.model')
results = model.most_similar('재미있다', topn=10)

word = []
data = []

for result in results:

    word.append(result[0])
    data.append(result[1])


font_name = font_manager.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf").get_name()
rc('font', family=font_name)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

ypos = np.arange(len(word))
rects = plt.barh(ypos,data, align='center', height=0.5)
plt.yticks(ypos, word)
for i, rect in enumerate(rects):
    ax.text(0.95 * rect.get_width(), rect.get_y() + rect.get_height() / 2.0, str(data[i]) + '%', ha='right', va='center')
plt.show()
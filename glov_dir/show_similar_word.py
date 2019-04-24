import pprint
from glove import Glove
import os

os.chdir("/home/ailsb/PycharmProjects/test/glove_dir/data")

print('Loading pre-trained GloVe model')
glove = Glove.load('glove.model')
print('Querying for %s' % "재미")
pprint.pprint(glove.most_similar("영화", number=10))

from six.moves import urllib
import codecs
import os
from konlpy.tag import Twitter

corpus = 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt'
save_path = './ratings_train.txt'
if not os.path.isfile(save_path):
    urllib.request.urlretrieve(corpus, './ratings_train.txt')

def read_data(filename):
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]   # header 제외
    return data

def tokenize(doc):
    return ['/'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]

train_data = read_data(save_path)
train_docs = [row[1] for row in train_data]

tagger = Twitter()
sentences = [tokenize(d) for d in train_docs]
print(sentences[0])

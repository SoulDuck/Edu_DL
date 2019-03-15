import gensim
from gensim.models import Word2Vec
import os

print ("LOADING PRETRAINED WORD2VEC MODEL... ")
if not os.path.isdir("data/word2vecdb"):
    model = gensim.models.KeyedVectors.load_word2vec_format('/Users/nikhilbuduma/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    print(model)
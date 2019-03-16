import nltk
nltk.download('movie_reviews')

from nltk.corpus import movie_reviews
sentences = [list(s) for s in movie_reviews.sents()]
print(sentences[0])
from gensim.models.word2vec import Word2Vec


model = Word2Vec(sentences)
model.init_sims(replace=True)

# 두 단어에 대한 similarity
model.wv.similarity('actor', 'actress')

# 비슷한 단어를 나열하기
model.wv.most_similar("accident")

# positive 인수와 negative 인수 사요앟기
model.wv.most_similar(positive=['she', 'actor'], negative='actress', topn=1)


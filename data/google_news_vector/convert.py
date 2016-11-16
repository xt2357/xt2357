from gensim.models import word2vec
model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.save("google_news_vector.txt")
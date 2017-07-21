from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt

#fname = "/home/trideep/Downloads/embeddings/win5_500mil.txt"
fname = "/home/trideep/Downloads/GoogleNews-vectors-negative300.bin"
count_dict = {}
count_distance = []

model = Word2Vec.load_word2vec_format(fname, binary=True)

for word in model.vocab:
    count_dict[word] = model.vocab[word].count

count_dict= sorted(count_dict.items(), key=lambda x: x[1])

for word_count in count_dict:
    count = word_count[1]
    #distance = np.linalg.norm(model[word_count[0]] - np.zeros(300))
    distance = np.linalg.norm(model[word_count[0]])
    count_distance.append((word_count[0], count, distance))

print(count_distance[0:100])
print(count_distance[len(count_distance)-100: len(count_distance)-1])
# plotting the moving average of distance of words in increasing order of count
N = 20
cumsum, moving_aves = [0], []
for i, x in enumerate([x[2] for x in count_distance], 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        moving_aves.append(moving_ave)

plt.plot(moving_aves)
plt.ylabel('magnitude')
plt.xlabel('word index by sorted count')
plt.show()

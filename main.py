import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

ds=pd.read_csv('IMDB Dataset.csv')
print(ds.shape)

eg=ds['review'][50]

tokens = nltk.word_tokenize(eg)
print(tokens[:10])
tagged=nltk.pos_tag(tokens[:10])
print(tagged)
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


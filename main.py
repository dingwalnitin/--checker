

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import nltk


plt.style.use('ggplot')


ds = pd.read_csv('IMDB Dataset.csv')
print(ds.shape)
ds = ds.head(50000)
eg = ds['review'][50]

tokens = nltk.word_tokenize(eg)
print(tokens)
tagged = nltk.pos_tag(tokens[:10])
print(tagged)
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


sia = SentimentIntensityAnalyzer()

print(sia.polarity_scores(eg))

res = {}
for i, row in tqdm(ds.iterrows(), total=len(ds)):
    text = row['review']
    myid = row['s/no']
    res[myid] = sia.polarity_scores(text)


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 's/no'})
vaders = vaders.merge(ds, how='left', on='s/no')
ax = sns.barplot(data=vaders, x='sentiment', y='compound')
ax.set_title('title h bhai')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='sentiment', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='sentiment', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='sentiment', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

import os
import pandas as pd
import numpy as np
# string manipulation libs
import re
import string
import nltk
from nltk.corpus import stopwords
# sklearn libs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# viz libs
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text"""

    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    text = re.sub("ß", "ss", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords_eng_de]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text

#import
df = pd.read_excel ('input.xlsx')
df = pd.DataFrame(df, columns= ["corpus"])

#preprocessing
stopwords_eng_de = set(stopwords.words('english')) | set(stopwords.words('german'))
stopwords_eng_de.remove('die')

df['cleaned'] = df['corpus'].apply(lambda x: preprocess_text(x, remove_stopwords=True))

# init vectorizer
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(df["cleaned"])

# initialize kmeans with k centroids
k= 4
kmeans = KMeans(n_clusters=k, init='k-means++')
# fit the model
kmeans.fit(features)
# store cluster labels in a variable
df['cluster'] = kmeans.labels_
 

#Visualization
# initialize PCA with 2 components
pca = PCA(n_components=2)
# pass our features to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(features.toarray())
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]
df['x0'] = x0
df['x1'] = x1


# Silhouette Method to determine optimal k (centroid) value/amount between 2-20
from yellowbrick.cluster import KElbowVisualizer
visualizer = KElbowVisualizer(kmeans, k=(2,20), metric='silhouette', timings= True)
visualizer.fit(features)        # Fit data to visualizer
visualizer.show()        # Finalize and render figure


# TF-IDF + KMeans Book Title clustering visualization 
# set image size
plt.figure(figsize=(12, 7))
# set a title
plt.title("TF-IDF + KMeans Book Title clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="viridis")
plt.show()



# Output
print("Cluster centroids: \n")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(k):
    print("Cluster %d:" % i)
    for j in order_centroids[i, :10]: #print out 10 feature terms of each cluster
        print (' %s' % terms[j])
    print('------------')

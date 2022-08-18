# text_clustering
 
This text clustering pipeline can be used to import the 'input.xlsx' data.
The test data has a row named 'corpus' which contains book titles.
With the usage of TF-IDF Vectorizer and kMeans this code can perform basic text clustering to determine theme keywords from all the book titles.

This script was developed as a part of an coding challenge.

feature_extractor.py Functionality:
1. Import input.xlsx
2. Preprocess text
3. Perform TF-IDF Vectorization
4. Initialize centroids with kMeans on obtained features (k needs to be hardcoded)
5. Visualize optimal k value with yellowbrick KElbowVisualizer (Silhouette Method)
6. Visualize Clustering
7. Terminal Output of 10 keywords for each cluster centroid

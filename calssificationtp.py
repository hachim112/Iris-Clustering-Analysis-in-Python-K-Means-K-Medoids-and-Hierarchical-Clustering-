#TALEB ABDELMALEK 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, dendrogram

try:
    from sklearn_extra.cluster import KMedoids
    kmedoids_available = True
except ImportError:
    kmedoids_available = False

# Load the Iris dataset
data = load_iris()
features = data.data  # Predictor variables
true_labels = data.target  # Actual classes (0, 1, 2)

# Encode the labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(true_labels)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_predicted = kmeans.fit_predict(features)

# Confusion matrix and error rate for K-means
kmeans_confusion = confusion_matrix(encoded_labels, kmeans_predicted)
kmeans_error_rate = np.mean(encoded_labels != kmeans_predicted)

# Visualization of K-means clustering
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=kmeans_predicted, cmap='viridis', s=50)
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title('K-means Clustering')
plt.show()

# K-medoids clustering (if available)
if kmedoids_available:
    kmedoids = KMedoids(n_clusters=3, random_state=42)
    kmedoids_predicted = kmedoids.fit_predict(features)

    # Confusion matrix and error rate for K-medoids
    kmedoids_confusion = confusion_matrix(encoded_labels, kmedoids_predicted)
    kmedoids_error_rate = np.mean(encoded_labels != kmedoids_predicted)

    # Visualization of K-medoids clustering
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=kmedoids_predicted, cmap='viridis', s=50)
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_zlabel('Petal Length')
    plt.title('K-medoids Clustering')
    plt.show()

# Hierarchical clustering (Single linkage)
z_single = linkage(features, method='single')
plt.figure(figsize=(10, 6))
dendrogram(z_single, labels=true_labels, leaf_rotation=90)
plt.title('Dendrogram - Single Linkage')
plt.xlabel('Observations')
plt.ylabel('Distance')
plt.show()

# Hierarchical clustering (Complete linkage)
z_complete = linkage(features, method='complete')
plt.figure(figsize=(10, 6))
dendrogram(z_complete, labels=true_labels, leaf_rotation=90)
plt.title('Dendrogram - Complete Linkage')
plt.xlabel('Observations')
plt.ylabel('Distance')
plt.show()

# Summary of results
print(f"K-means Error Rate: {kmeans_error_rate * 100:.2f}%")
if kmedoids_available:
    print(f"K-medoids Error Rate: {kmedoids_error_rate * 100:.2f}%")

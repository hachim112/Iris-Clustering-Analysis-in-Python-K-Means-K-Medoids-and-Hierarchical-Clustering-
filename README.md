# Iris Clustering Analysis in Python (K-Means, K-Medoids and Hierarchical Clustering)

## Repository Name

iris-clustering-analysis-python

## Repository Description

This repository contains a complete Python implementation of several automatic (unsupervised) classification techniques applied to the famous Iris dataset. The project focuses on understanding, implementing, visualizing, and comparing different clustering algorithms, including partition-based methods and hierarchical methods.

The project is designed mainly for educational purposes and practical learning in machine learning and data mining courses.

Implemented algorithms:

* K-Means
* K-Medoids
* Hierarchical Clustering (Single Linkage)
* Hierarchical Clustering (Complete Linkage)

The project also includes:

* 3D visualization of clusters
* Dendrogram visualization
* Confusion matrices
* Error rate computation

---

## Project Objectives

The main objectives of this project are:

* Understand the concept of unsupervised learning
* Learn how clustering algorithms group data without class labels
* Apply multiple clustering methods on the same dataset
* Compare the behavior and performance of each method
* Visualize clustering results
* Interpret dendrograms
* Evaluate clustering quality

---

## Dataset Description

The Iris dataset is a classic dataset in machine learning.

Number of samples: 150

Classes:

* Setosa
* Versicolor
* Virginica

Features:

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

Each sample represents one flower described by four numeric measurements.

---

## Technologies and Libraries

* Python 3
* NumPy
* Matplotlib
* Scikit-learn
* SciPy
* scikit-learn-extra (optional)

---

## Installation

Install required packages using:

pip install numpy matplotlib scikit-learn scipy scikit-learn-extra

If scikit-learn-extra is not installed, the program will automatically skip K-Medoids.

---

## How to Run the Project

1. Clone the repository

2. Run the script:

python classificationtp.py

---

## General Concept: Unsupervised Clustering

In unsupervised learning, the algorithm does not know the true class labels. It tries to discover natural groupings in data based only on feature similarity.

Clustering aims to:

* Minimize distance between samples inside the same cluster
* Maximize distance between different clusters

---

## K-Means Clustering

### Theory

K-Means is a partition-based clustering algorithm.

It divides data into K clusters. Each cluster is represented by a centroid (mean of points).

The algorithm minimizes the total within-cluster squared distance.

Objective function:

Sum over all clusters of the squared distance between each point and its centroid.

### Algorithm Steps

1. Choose K initial centroids randomly
2. Assign each point to the nearest centroid
3. Recalculate centroids as the mean of assigned points
4. Repeat steps 2 and 3 until centroids stop changing

### Characteristics

Advantages:

* Simple
* Fast
* Works well for spherical clusters

Disadvantages:

* Sensitive to outliers
* Sensitive to initialization
* Requires K to be chosen manually

### Implementation

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_predicted = kmeans.fit_predict(features)

### Output

* Cluster labels
* Confusion matrix
* Error rate
* 3D scatter plot

---

## K-Medoids Clustering

### Theory

K-Medoids is similar to K-Means but uses real data points as cluster centers (medoids) instead of averages.

A medoid is the most centrally located point in a cluster.

The algorithm minimizes total distance between points and their medoid.

### Differences from K-Means

* Centers are actual samples
* More robust to noise
* Less affected by extreme values

### Characteristics

Advantages:

* Robust to outliers
* More stable clusters

Disadvantages:

* Slower than K-Means

### Implementation

kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids_predicted = kmedoids.fit_predict(features)

### Output

* Confusion matrix
* Error rate
* 3D visualization

---

## Hierarchical Clustering

Hierarchical clustering builds clusters step by step.

Two main approaches:

* Agglomerative (bottom-up)
* Divisive (top-down)

This project uses agglomerative clustering.

The result is visualized using a dendrogram (tree diagram).

---

## Single Linkage (Nearest Neighbor)

### Theory

Distance between two clusters = minimum distance between any pair of points.

### Properties

* Can create long chain-like clusters
* Sensitive to noise

### Implementation

z_single = linkage(features, method='single')

---

## Complete Linkage (Farthest Neighbor)

### Theory

Distance between two clusters = maximum distance between any pair of points.

### Properties

* Produces compact clusters
* Less chaining effect

### Implementation

z_complete = linkage(features, method='complete')

---

## Dendrogram Interpretation

* X-axis: data samples
* Y-axis: distance between merged clusters
* Height of merge indicates dissimilarity

Cutting the dendrogram at a horizontal level determines the number of clusters.

---

## Evaluation Methods

### Confusion Matrix

Shows relationship between true classes and predicted clusters.

confusion_matrix(encoded_labels, predicted_labels)

### Error Rate

Error Rate = Number of mismatched labels / Total samples

np.mean(encoded_labels != predicted_labels)

---

## Important Remark About Error Rate

Clustering algorithms assign arbitrary cluster numbers.

Example:
Cluster 0 may correspond to Virginica instead of Setosa.

This can cause artificially high error rate.

A mapping step between clusters and real classes is required for correct accuracy measurement.

---

## Visualizations

* 3D scatter plots showing cluster separation
* Dendrograms for hierarchical clustering

Visualization helps understand how data is grouped.

---

## Project Structure

iris-clustering-analysis-python/
│
├── classificationtp.py
├── README.md
├── images/
│   ├── kmeans.png
│   ├── kmedoids.png
│   ├── dendrogram_single.png
│   └── dendrogram_complete.png

---

## Author

Hachim Fernane
Master 1 Computer Science
University of Guelma

---

## License
FEEL FREE TO USE JUST PRAY FOR ME




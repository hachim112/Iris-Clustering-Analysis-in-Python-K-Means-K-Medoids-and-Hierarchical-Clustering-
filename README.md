# Iris Clustering Analysis Using Unsupervised Learning (Python)

---

## Overview

This repository contains a professional and well-structured implementation of several unsupervised clustering algorithms applied to the Iris dataset using Python. The project demonstrates partition-based and hierarchical clustering techniques, along with visualization and evaluation tools.

The project is suitable for academic use, machine learning courses, and practical experimentation.

---

## Implemented Methods


- K-Means
- K-Medoids
- Hierarchical Clustering (Single Linkage)
- Hierarchical Clustering (Complete Linkage)


---

## Objectives


- Understand unsupervised learning
- Apply clustering algorithms
- Visualize clusters
- Compare clustering methods
- Interpret dendrograms
- Evaluate clustering quality


---

## Dataset Description

```
Dataset: Iris
Samples: 150
Classes: 3 (Setosa, Versicolor, Virginica)
Features: 4
  x1: Sepal Length
  x2: Sepal Width
  x3: Petal Length
  x4: Petal Width
```

---

## Technologies

```
Python 3
NumPy
Matplotlib
Scikit-learn
SciPy
scikit-learn-extra (optional)
```

---

## Installation

```
pip install numpy matplotlib scikit-learn scipy scikit-learn-extra
```

---

## Run Project

```
python classificationtp.py
```

---

# Theory Background

---

## Unsupervised Learning

Unsupervised learning attempts to discover hidden patterns in data without labeled outputs.

Goal:

```
Minimize intra-cluster distance
Maximize inter-cluster distance
```

---

# K-Means Clustering

---

## Principle

K-Means partitions data into K clusters represented by centroids.

---

## Optimization Objective

```
Minimize J = Σ Σ || x_i - μ_k ||²
            k   i∈Ck
```

Where:

```
x_i : data point
μ_k : centroid of cluster k
Ck  : cluster k
```

---

## Algorithm Steps

```
1. Choose K initial centroids
2. Assign points to nearest centroid
3. Update centroids
4. Repeat until convergence
```

---

## Advantages

```
- Simple
- Fast
- Scalable
```

## Limitations

```
- Sensitive to outliers
- Sensitive to initialization
- Requires predefined K
```

---

## Implementation

```
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_predicted = kmeans.fit_predict(features)
```

---

# K-Medoids Clustering

---

## Principle

Cluster centers are real data points called medoids.

---

## Optimization Objective

```
Minimize J = Σ Σ d(x_i , m_k)
            k   i∈Ck
```

Where:

```
m_k : medoid of cluster k
d   : distance function
```

---

## Advantages

```
- Robust to noise
- Robust to outliers
```

## Limitations

```
- Slower than K-Means
```

---

## Implementation

```
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids_predicted = kmedoids.fit_predict(features)
```

---

# Hierarchical Clustering

---

## Principle

Agglomerative approach:

```
Each sample starts as its own cluster
Clusters are merged step by step
```

Result represented as dendrogram.

---

# Single Linkage

---

## Distance Definition

```
D(A,B) = min d(x , y)
         x∈A,y∈B
```

## Properties

```
- Chain effect
- Sensitive to noise
```

---

## Implementation

```
z_single = linkage(features, method='single')
```

---

# Complete Linkage

---

## Distance Definition

```
D(A,B) = max d(x , y)
         x∈A,y∈B
```

## Properties

```
- Compact clusters
- More stable
```

---

## Implementation

```
z_complete = linkage(features, method='complete')
```

---

# Dendrogram Interpretation

```
X-axis : Observations
Y-axis : Distance
Horizontal cut => number of clusters
```

---

# Evaluation Metrics

---

## Confusion Matrix

```
confusion_matrix(true_labels, predicted_labels)
```

---

## Error Rate

```
Error Rate = Number of wrong assignments / Total samples

Error Rate = mean(true != predicted)
```

---

# Important Remark

```
Cluster labels are arbitrary.
Direct comparison with true labels may be misleading.
Label mapping is required for true accuracy.
```

---

# Visualizations

```
- 3D cluster plots
- Dendrograms
```

---

# Project Structure

```
iris-clustering-analysis-python/
│
├── classificationtp.py
├── README.md
├── images/
│   ├── kmeans.png
│   ├── kmedoids.png
│   ├── dendrogram_single.png
│   └── dendrogram_complete.png
```

---

# Author
Hachim Fernane
---

# License

FEEL FREE TO USE JUST PRAY FOR ME



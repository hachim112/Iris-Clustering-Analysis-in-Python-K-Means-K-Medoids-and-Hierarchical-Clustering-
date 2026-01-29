# Iris Clustering Analysis Using Unsupervised Learning (Python)



## Overview

This repository contains a structured and professional implementation of unsupervised clustering techniques applied to the Iris dataset using Python. The project focuses on partition-based and hierarchical clustering methods, along with mathematical background, visual interpretation, and evaluation.

The current implementation generates:

* 3D visualization of K-Means clusters
* Dendrogram using Single Linkage
* Dendrogram using Complete Linkage

This repository is intended for academic use and practical experimentation in machine learning and data mining.

---

## Implemented Algorithms


- K-Means Clustering
- K-Medoids Clustering (optional)
- Hierarchical Clustering (Single Linkage)
- Hierarchical Clustering (Complete Linkage)


---

## Objectives


- Understand unsupervised learning
- Apply clustering algorithms on real data
- Visualize clusters in 3D
- Interpret dendrogram structures
- Evaluate clustering quality


---

## Dataset

```
Name: Iris Dataset
Samples: 150
Classes: 3 (Setosa, Versicolor, Virginica)
Features:
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

## Run the Program

```
python classificationtp.py
```

---

# Theoretical Background

---

## Unsupervised Clustering

Unsupervised learning groups data without using class labels.

Goal:


Minimize intra-cluster distance
Maximize inter-cluster distance


---

# K-Means Clustering

## Principle

K-Means partitions data into K clusters represented by centroids (means).

---

## Objective Function

```
J = Σ Σ || x_i - μ_k ||²
    k  i∈Ck
```

Where:

```
x_i : data point
μ_k : centroid of cluster k
Ck  : cluster k
```

---

## Algorithm Steps


1. Initialize K centroids
2. Assign each point to nearest centroid
3. Update centroids
4. Repeat until convergence


---

## Advantages


- Simple
- Fast


## Limitations

```
- Sensitive to outliers
- Sensitive to initialization
```

---

## Implementation

```
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_predicted = kmeans.fit_predict(features)
```

---

# K-Medoids Clustering

## Principle

Cluster centers are real samples called medoids.

---

## Objective Function

```
J = Σ Σ d(x_i , m_k)
    k  i∈Ck
```

Where:

```
m_k : medoid of cluster k
d   : distance metric
```

---

## Advantages

```
- Robust to outliers
```

## Limitation

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

## Principle

Agglomerative approach:

```
Each sample starts as one cluster
Closest clusters are merged iteratively
```

Result is represented using a dendrogram.

---

# Single Linkage

## Distance Definition

```
D(A,B) = min d(x,y)
         x∈A,y∈B
```

## Property

```
Chain effect
```

---

## Implementation

```
z_single = linkage(features, method='single')
```

---

# Complete Linkage

## Distance Definition

```
D(A,B) = max d(x,y)
         x∈A,y∈B
```

## Property

```
Compact clusters
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
Horizontal cut -> number of clusters
```

---

# Evaluation

## Confusion Matrix

```
confusion_matrix(true_labels, predicted_labels)
```

## Error Rate

```
Error Rate = mean(true_labels != predicted_labels)
```

---

# Important Note

```
Cluster labels are arbitrary.
Direct comparison may be misleading.
```

---

# Outputs Generated

```
- K-Means 3D Scatter Plot
- Dendrogram (Single Linkage)
- Dendrogram (Complete Linkage)
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
│   ├── dendrogram_single.png
│   └── dendrogram_complete.png
│
├── docs/
│   ├── TP3_Classification-Automatique.pdf
│   └── Rapport_TP3_Classification.pdf
```

---

# Author

```
Hachim Fernane

```

---

# License
FEEL FREE TO USE JUST PRAY FOR ME


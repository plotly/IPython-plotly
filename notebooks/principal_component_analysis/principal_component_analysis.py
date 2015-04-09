
# coding: utf-8

# ### About the Author

# Some of Sebastian Raschka's greatest passions are "Data Science" and machine learning. Sebastian enjoys everything that involves working with data: The discovery of interesting patterns and coming up with insightful conclusions using techniques from the fields of data mining and machine learning for predictive modeling.
# 
# Currently, Sebastian is sharpening his analytical skills as a PhD candidate at Michigan State University where he is working on a highly efficient virtual screening software for computer-aided drug-discovery and a novel approach to protein ligand docking (among other projects). Basically, it is about the screening of a database of millions of 3-dimensional structures of chemical compounds in order to identifiy the ones that could potentially bind to specific protein receptors in order to trigger a biological response.
# 
# You can follow Sebastian on Twitter ([@rasbt](https://twitter.com/rasbt)) or read more about his favorite projects on [his blog](http://sebastianraschka.com/articles.html).

# <br>
# <br>

# # Principal Component Analysis in 3 Simple Steps

# Principal Component Analysis (PCA) is a simple yet popular and useful linear transformation technique that is used in numerous applications, such as stock market predictions, the  analysis of gene expression data, and many more. In this tutorial, we will see that PCA is not just a "black box", and we are going to unravel its internals in 3 basic steps.

# <br>
# <br>

# <hr>

# ## Sections

# - [Introduction](#Introduction)
#     - [PCA Vs. LDA](#PCA-Vs.-LDA)
#     - [PCA and Dimensionality Reduction](#PCA-and-Dimensionality-Reduction)
#     - [A Summary of the PCA Approach](#A-Summary-of-the-PCA-Approach)
# - [Preparing the Iris Dataset](#Preparing-the-Iris-Dataset)
#     - [About Iris](#About-Iris)
#     - [Loading the Dataset](#Loading-the-Dataset)
#     - [Exploratory Visualization](#Exploratory-Visualization)
#     - [Standardizing](#Standardizing)
# - [1 - Eigendecomposition - Computing Eigenvectors and Eigenvalues](#1---Eigendecomposition---Computing-Eigenvectors-and-Eigenvalues)
#     - [Covariance Matrix](#Covariance-Matrix)
#     - [Correlation Matrix](#Correlation-Matrix)
#     - [Singular Vector Decomposition](#Singular-Vector-Decomposition)
# - [2 - Selecting Principal Components](#2---Selecting-Principal-Components)
#     - [Sorting Eigenpairs](#Sorting-Eigenpairs)
#     - [Explained Variance](#Explained-Variance)
#     - [Projection Matrix](#Projection-Matrix)
# - [3 - Projection Onto the New Feature Space](#3---Selecting-Principal-Components)
# - [Shortcut - PCA in scikit-learn](#Shortcut---PCA-in-scikit-learn)

# <br>
# <br>

# <hr>

# ## Introduction

# [[back to top](#Sections)]

# The sheer size of data in the modern age is not only a challenge for computer hardware but also a main bottleneck for the performance of many machine learning algorithms. The main goal of a PCA analysis is to identify patterns in data; PCA aims to detect the correlation between variables. If a strong correlation between variables exists, the attempt to reduce the dimensionality only makes sense. In a nutshell, this is what PCA is all about: Finding the directions of maximum variance in high-dimensional data and project it onto a smaller dimensional subspace while retaining most of the information.
# 
# <img src='http://sebastianraschka.com/Images/2015_pca_in_3_steps/pca_sketch.png' alt=''>

# <br>
# <br>

# ### PCA Vs. LDA

# [[back to top](#Sections)]

# Both Linear Discriminant Analysis (LDA) and PCA are linear transformation methods. PCA yields the directions (principal components) that maximize the variance of the data, whereas LDA also aims to find the directions that maximize the separation (or discrimination) between different classes, which can be useful in pattern classification problem (PCA "ignores" class labels).   
# ***In other words, PCA projects the entire dataset onto a different feature (sub)space, and LDA tries to determine a suitable feature (sub)space in order to distinguish between patterns that belong to different classes.***  

# <br>
# <br>

# ### PCA and Dimensionality Reduction

# [[back to top](#Sections)]

# Often, the desired goal is to reduce the dimensions of a $d$-dimensional dataset by projecting it onto a $(k)$-dimensional subspace (where $k\;<\;d$) in order to increase the computational efficiency while retaining most of the information. An important question is "what is the size of $k$ that represents the data 'well'?"
# 
# Later, we will compute eigenvectors (the principal components) of a dataset and collect them in a projection matrix. Each of those eigenvectors is associated with an eigenvalue which can be interpreted as the "length" or "magnitude" of the corresponding eigenvector. If some eigenvalues have a significantly larger magnitude than others that the reduction of the dataset via PCA onto a smaller dimensional subspace by dropping the "less informative" eigenpairs is reasonable.
# 

# <br>
# <br>

# ### A Summary of the PCA Approach

# [[back to top](#Sections)]

# -  Standardize the data.
# -  Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition.
# -  Sort eigenvalues in descending order and choose the $k$ eigenvectors that correspond to the $k$ largest eigenvalues where $k$ is the number of dimensions of the new feature subspace ($k \le d$)/.
# -  Construct the projection matrix $\mathbf{W}$ from the selected $k$ eigenvectors.
# -  Transform the original dataset $\mathbf{X}$ via $\mathbf{W}$ to obtain a $k$-dimensional feature subspace $\mathbf{Y}$.

# <br>
# <br>

# ## Preparing the Iris Dataset

# [[back to top](#Sections)]

# <br>
# <br>

# ### About Iris

# [[back to top](#Sections)]

# For the following tutorial, we will be working with the famous "Iris" dataset that has been deposited on the UCI machine learning repository   
# ([https://archive.ics.uci.edu/ml/datasets/Iris](https://archive.ics.uci.edu/ml/datasets/Iris)).
# 
# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset are:
# 
# 1. Iris-setosa (n=50)
# 2. Iris-versicolor (n=50)
# 3. Iris-virginica (n=50)
# 
# And the four features of in Iris dataset are:
# 
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 
# <img src="http://sebastianraschka.com/Images/2014_python_lda/iris_petal_sepal.png" alt="Iris" style="width: 200px;"/>

# <br>
# <br>

# ### Loading the Dataset

# [[back to top](#Sections)]

# In order to load the Iris data directly from the UCI repository, we are going to use the superb [pandas](http://pandas.pydata.org) library. If you haven't used pandas yet, I want encourage you to check out the [pandas tutorials](http://pandas.pydata.org/pandas-docs/stable/tutorials.html). If I had to name one Python library that makes working with data a wonderfully simple task, this would definitely be pandas!

# In[1]:

import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()


# In[2]:

# split data table into data X and class labels y

X = df.ix[:,0:4].values
y = df.ix[:,4].values


# Our iris dataset is now stored in form of a  $150 \times 4$ matrix where the columns are the different features, and every row represents a separate flower sample.
# Each sample row $\mathbf{x}$ can be pictured as a 4-dimensional vector   
# 
# 
# $\mathbf{x^T} = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{pmatrix} 
# = \begin{pmatrix} \text{sepal length} \\ \text{sepal width} \\\text{petal length} \\ \text{petal width} \end{pmatrix}$

# <br>
# <br>

# ### Exploratory Visualization

# [[back to top](#Sections)]

# To get a feeling for how the 3 different flower classes are distributes along the 4 different features, let us visualize them via histograms.

# In[3]:

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls


# In[4]:

# plotting histograms

traces = []

legend = {0:False, 1:False, 2:False, 3:True}

colors = {'Iris-setosa': 'rgb(31, 119, 180)', 
          'Iris-versicolor': 'rgb(255, 127, 14)', 
          'Iris-virginica': 'rgb(44, 160, 44)'}

for col in range(4):
    for key in colors:
        traces.append(Histogram(x=X[y==key, col], 
                        opacity=0.75,
                        xaxis='x%s' %(col+1),
                        marker=Marker(color=colors[key]),
                        name=key,
                        showlegend=legend[col]))

data = Data(traces)

layout = Layout(barmode='overlay',
                xaxis=XAxis(domain=[0, 0.25], title='sepal length (cm)'),
                xaxis2=XAxis(domain=[0.3, 0.5], title='sepal width (cm)'),
                xaxis3=XAxis(domain=[0.55, 0.75], title='petal length (cm)'),
                xaxis4=XAxis(domain=[0.8, 1], title='petal width (cm)'),
                yaxis=YAxis(title='count'),
                title='Distribution of the different Iris flower features')

fig = Figure(data=data, layout=layout)
py.iplot(fig)


# <br>
# <br>

# ### Standardizing

# [[back to top](#Sections)]

# Whether to standardize the data prior to a PCA on the covariance matrix depends on the measurement scales of the original features. Since PCA yields a feature subspace that maximizes the variance along the axes, it makes sense to standardize the data, especially, if it was measured on different scales. Although, all features in the Iris dataset were measured in centimeters, let us continue with the transformation of the data onto unit scale (mean=0 and variance=1), which is a requirement for the optimal performance of many machine learning algorithms.

# In[5]:

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


# <br>
# <br>

# ## 1 - Eigendecomposition - Computing Eigenvectors and Eigenvalues

# [[back to top](#Sections)]

# The eigenvectors and eigenvalues of a covariance (or correlation) matrix represent the "core" of a PCA: The eigenvectors (principal components) determine the directions of the new feature space, and the eigenvalues determine their magnitude. In other words, the eigenvalues explain the variance of the data along the new feature axes.

# <br>
# <br>

# ### Covariance Matrix

# [[back to top](#Sections)]

# The classic approach to PCA is to perform the eigendecomposition on the covariance matrix $\Sigma$, which is a $d \times d$ matrix where each element represents the covariance between two features. The covariance between two features is calculated as follows:
# 
# $\sigma_{jk} = \frac{1}{n-1}\sum_{i=1}^{N}\left(  x_{ij}-\bar{x}_j \right)  \left( x_{ik}-\bar{x}_k \right).$
# 
# We can summarize the calculation of the covariance matrix via the following matrix equation:   
# $\Sigma = \frac{1}{n-1} \left( (\mathbf{X} - \mathbf{\bar{x}})^T\;(\mathbf{X} - \mathbf{\bar{x}}) \right)$  
# where $\mathbf{\bar{x}}$ is the mean vector 
# $\mathbf{\bar{x}} = \sum\limits_{k=1}^n x_{i}.$  
# The mean vector is a $d$-dimensional vector where each value in this vector represents the sample mean of a feature column in the dataset.

# In[6]:

import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


# The more verbose way above was simply used for demonstration purposes, equivalently, we could have used the numpy `cov` function:

# In[7]:

print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))


# <br>
# <br>

# Next, we perform an eigendecomposition on the covariance matrix:

# In[8]:

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# <br>
# <br>

# ### Correlation Matrix

# [[back to top](#Sections)]

# Especially, in the field of "Finance," the correlation matrix typically used instead of the covariance matrix. However, the eigendecomposition of the covariance matrix (if the input data was standardized) yields the same results as a eigendecomposition on the correlation matrix, since the correlation matrix can be understood as the normalized covariance matrix.

# <br>
# <br>

# Eigendecomposition of the standardized data based on the correlation matrix:

# In[9]:

cor_mat1 = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# <br>
# <br>

# Eigendecomposition of the raw data based on the correlation matrix:

# In[10]:

cor_mat2 = np.corrcoef(X.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# <br>
# <br>

# We can clearly see that all three approaches yield the same eigenvectors and eigenvalue pairs:
#     
# - Eigendecomposition of the covariance matrix after standardizing the data.
# - Eigendecomposition of the correlation matrix.
# - Eigendecomposition of the correlation matrix after standardizing the data.

# <br>
# <br>

# ### Singular Vector Decomposition

# [[back to top](#Sections)]

# While the eigendecomposition of the covariance or correlation matrix may be more intuitiuve, most PCA implementations perform a Singular Vector Decomposition (SVD) to improve the computational efficiency. So, let us perform an SVD to  confirm that the result are indeed the same:

# In[11]:

u,s,v = np.linalg.svd(X_std.T)
u


# <br>
# <br>

# ## 2 - Selecting Principal Components

# [[back to top](#Sections)]

# <br>
# <br>

# ### Sorting Eigenpairs

# [[back to top](#Sections)]

# The typical goal of a PCA is to reduce the dimensionality of the original feature space by projecting it onto a smaller subspace, where the eigenvectors will form the axes. However, the eigenvectors only define the directions of the new axis, since they have all the same unit length 1, which can confirmed by the following two lines of code:

# In[12]:

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')


# <br>
# <br>

# In order to decide which eigenvector(s) can dropped without losing too much information
# for the construction of lower-dimensional subspace, we need to inspect the corresponding eigenvalues: The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data; those are the ones can be dropped.  
# In order to do so, the common approach is to rank the eigenvalues from highest to lowest in order choose the top $k$ eigenvectors.

# In[13]:

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# <br>
# <br>

# ### Explained Variance

# [[back to top](#Sections)]

# After sorting the eigenpairs, the next question is "how many principal components are we going to choose for our new feature subspace?" A useful measure is the so-called "explained variance," which can be calculated from the eigenvalues. The explained variance tells us how much information (variance) can be attributed to each of the principal components.

# In[14]:

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = Bar(
        x=['PC %s' %i for i in range(1,5)],
        y=var_exp,
        showlegend=False)

trace2 = Scatter(
        x=['PC %s' %i for i in range(1,5)], 
        y=cum_var_exp,
        name='cumulative explained variance')

data = Data([trace1, trace2])

layout=Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

fig = Figure(data=data, layout=layout)
py.iplot(fig)


# The plot above clearly shows that most of the variance (72.77% of the variance to be precise) can be explained by the first principal component alone. The second principal component still bears some information (23.03%) while the third and fourth principal components can safely be dropped without losing to much information. Together, the first two principal components contain 95.8% of the information.

# <br>
# <br>

# ### Projection Matrix

# [[back to top](#Sections)]

# It's about time to get to the really interesting part: The construction of the projection matrix that will be used to transform the Iris data onto the new feature subspace. Although, the name "projection matrix" has a nice ring to it, it is basically just a matrix of our concatenated top *k* eigenvectors.
# 
# Here, we are reducing the 4-dimensional feature space to a 2-dimensional feature subspace, by choosing the "top 2" eigenvectors with the highest eigenvalues to construct our $d \times k$-dimensional eigenvector matrix $\mathbf{W}$.

# In[15]:

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)


# <br>
# <br>

# ## 3 - Projection Onto the New Feature Space

# [[back to top](#Sections)]

# In this last step we will use the $4 \times 2$-dimensional projection matrix $\mathbf{W}$ to transform our samples onto the new subspace via the equation  
# $\mathbf{Y} = \mathbf{X} \times  \mathbf{W}$, where $\mathbf{Y}$ is a $150\times 2$ matrix of our transformed samples.

# In[16]:

Y = X_std.dot(matrix_w)


# In[17]:

traces = []

for name in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):

    trace = Scatter(
        x=Y[y==name,0],
        y=Y[y==name,1],
        mode='markers',
        name=name,
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
    traces.append(trace)


data = Data(traces)
layout = Layout(showlegend=True,
                scene=Scene(xaxis=XAxis(title='PC1'),
                yaxis=YAxis(title='PC2'),))

fig = Figure(data=data, layout=layout)
py.iplot(fig)


# <br>
# <br>
# <a name="mat_pca"></a>

# <br>
# <br>
# <a name="sklearn_pca"> </a>

# ## Shortcut - PCA in scikit-learn

# [[back to top](#Sections)]

# For educational purposes, we went a long way to apply the PCA to the Iris dataset. But luckily, there is already implementation in scikit-learn. 

# In[18]:

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)


# In[19]:

traces = []

for name in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):

    trace = Scatter(
        x=Y_sklearn[y==name,0],
        y=Y_sklearn[y==name,1],
        mode='markers',
        name=name,
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
    traces.append(trace)


data = Data(traces)
layout = Layout(xaxis=XAxis(title='PC1', showline=False),
                yaxis=YAxis(title='PC2', showline=False))
fig = Figure(data=data, layout=layout)
py.iplot(fig)


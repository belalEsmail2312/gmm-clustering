import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.interpolate import make_interp_spline
import timeit
from sklearn.decomposition import PCA

data = pd.read_csv("C:/Users/20109/Downloads/DataSet (1).csv")
mean = data.mean()
standDev = data.std()
normalizedF = (data - mean) / standDev


def initialize_parameters(data, K):
    N, D = data.shape
    np.random.seed(42)

    # Initialize means randomly
    means = data.sample(K, replace=False)

    # Initialize covariances as identity matrices
    covariances = [np.cov(data, rowvar=False) for _ in range(K)]

    # Initialize mixing coefficients uniformly
    weights = np.ones(K) / K

    means = means.values.tolist()

    return means, covariances, weights


def expectation(data, means, covariances, weights):
    N, K = data.shape[0], len(means)
    probabilities = np.zeros((N, K))

    for k in range(K):
        mean_k = means[k]
        covariance_k = covariances[k]
        weights_k = weights[k]

        probabilities[:, k] = weights_k * multivariate_normal.pdf(data, mean_k, covariance_k)

    # Normalize probabilities
    normalization_factor = np.sum(probabilities, axis=1, keepdims=True)
    probabilities /= normalization_factor

    return probabilities


def maximization(data, probabilities):
    N, D = data.shape
    K = probabilities.shape[1]

    # Update means
    means = np.dot(probabilities.T, data) / np.sum(probabilities, axis=0, keepdims=True).T

    # Update covariances
    covariances = [
        np.dot((data - means[k]).T, (data - means[k]) * probabilities[:, k][:, np.newaxis]) /
        np.sum(probabilities[:, k])
        for k in range(K)
    ]

    # Update weights
    weights = np.sum(probabilities, axis=0) / N

    return means, covariances, weights


def log_likelihood(data, means, covariances, weights):
    N, K = data.shape[0], len(means)
    likelihood = np.zeros(N)

    for k in range(K):
        likelihood += weights[k] * multivariate_normal.pdf(data, means[k], covariances[k])

    log_likelihood = np.sum(np.log(likelihood))

    return log_likelihood


def gmm(data, K, max_iterations=5, tolerance=1e-4):
    means, covariances, weights = initialize_parameters(data, K)

    for iteration in range(max_iterations):
        old_log_likelihood = log_likelihood(data, means, covariances, weights)
        probabilities = expectation(data, means, covariances, weights)
        means, covariances, weights = maximization(data, probabilities)
        new_log_likelihood = log_likelihood(data, means, covariances, weights)
        
        # Check for convergence
        if abs(new_log_likelihood - old_log_likelihood) < tolerance:
            break

    clusters = np.argmax(probabilities, axis=1)
    return clusters


# Define the range of sample sizes to test (from 5% to 100%)
sample_sizes = np.arange(0.05, 1.05, 0.05)

# Measure the execution time for each sample size for K=3
execution_times_k3 = []

for sample_size in sample_sizes:
    sample_data = normalizedF.sample(frac=sample_size, random_state=42)
    start_time = timeit.default_timer()
    clusters = gmm(sample_data, K=3)
    elapsed_time = timeit.default_timer() - start_time
    execution_times_k3.append(elapsed_time)

# Measure the execution time for each sample size for K=5
execution_times_k5 = []

for sample_size in sample_sizes:
    sample_data = normalizedF.sample(frac=sample_size, random_state=42)
    start_time = timeit.default_timer()
    clusters = gmm(sample_data, K=5)
    elapsed_time = timeit.default_timer() - start_time
    execution_times_k5.append(elapsed_time)

# Smooth the curves using interpolation
sample_sizes_smooth = np.linspace(sample_sizes.min(), sample_sizes.max(), 300)
execution_times_k3_smooth = make_interp_spline(sample_sizes, execution_times_k3)(sample_sizes_smooth)
execution_times_k5_smooth = make_interp_spline(sample_sizes, execution_times_k5)(sample_sizes_smooth)

# Plot the results for K=3 and K=5 with smooth curves
plt.plot(sample_sizes_smooth * 100, execution_times_k3_smooth, label='K=3')
plt.plot(sample_sizes_smooth * 100, execution_times_k5_smooth, label='K=5')
plt.title('Smoothed Execution Time vs Sample Size')
plt.xlabel('Sample Size (%)')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.show()






#-------------------------------------------------------------------------------------------------------------



np.random.seed(42)
# Instantiate PCA and specify the number of components (dimensions) you want
pca = PCA(n_components=2)
# Fit the model to the data and transform the data
reduced_data = pca.fit_transform(normalizedF)
reduced_data = pd.DataFrame(reduced_data)
clusters3 = gmm(reduced_data, K=3)
clusters5 = gmm(reduced_data, K=5)

total_samples = len(reduced_data)
percentage_cluster3 = (np.bincount(clusters3) / total_samples) * 100
percentage_cluster5 = (np.bincount(clusters5) / total_samples) * 100

# Print the percentage of each cluster for K=3
print("K =3")
for cluster, percentage in enumerate(percentage_cluster3):
    print(f'Cluster {cluster}: {percentage:.2f}%')
    
print("K =5")
for cluster, percentage in enumerate(percentage_cluster5):
 
    print(f'Cluster {cluster}: {percentage:.2f}%')
    
# Plot the clusters for K=3
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(reduced_data.iloc[:, 0], reduced_data.iloc[:, 1], c=clusters3, cmap='viridis')
plt.title('GMM Clustering (K=3)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot the clusters for K=5
plt.subplot(1, 2, 2)
plt.scatter(reduced_data.iloc[:, 0], reduced_data.iloc[:, 1], c=clusters5, cmap='viridis')
plt.title('GMM Clustering (K=5)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()

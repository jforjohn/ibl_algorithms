# pca_algorithm

The folder called clustering_algorithms comes with a:
    - requirements.txt for downloading possible dependencies (pip install -r requirements.txt)
    clustering.cfg configuration file in which you can define the specs of the algorithm you want to run

When you define what you want to run in the configuration file you just run the MainLauncher.py file.

NOTE: Don't worry about some Warnings that you may get in runtime.

Concerning the configuration file:
    - dataset: the name of the dataset without the \textit{.arff} extension, which is in the same directory as this file
    - algorithm: the algorithm or the algorithms you want to run separated by a dash (-) with no spaces e.g 1-2-3. Each algorithm corresponds to a number
        - Kmeans
        - Kmeans++
        - Kmedoids
        - Fuzzy C-means
        - Agglomerative complete linkage
        - Agglomerative average linkage
        - Agglomerative single linkage
    - k: the number of clusters
    - tol: the tolerance for the convergence
    - max_rep: the number of maximum repetitions
    - fuzzy_m: the degree in case of fuziness
    - kmeans_init_type: the type of initializing the centroids. The possible values are:
        - random: for getting random numbers following the uniform distribution
        - kmeans++: for applying KMeans++ algorithm for the initial centroids
    - run: the way you want to run the algorithms. The possible values are:
        - algorithms: for getting the indexes values for a specific k
        - silhouette: calculating the silhouette coefficient fir 15 different k and then it plots also the graph of best-k


Concerning the configuration file at the PCA section:
    - n_components: the number of PCA components/dimensions to keep
    - num_plot_features: the number of random features to plot

from mojo_kmeans import Matrix, Kmeans
from mojo_kmeans.utils import list_to_matrix
from time import now
from python import Python

def main():
    Python.add_to_path(".")
    py_kmeans = Python.import_module("python_kmeans")
    py_utils = Python.import_module("python_kmeans.utils")

    np = Python.import_module("numpy")
    sklearn_datasets = Python.import_module("sklearn.datasets")
    sklearn_cluster = Python.import_module("sklearn.cluster")
    
    onnxruntime = Python.import_module("onnxruntime")

    n_clusters = 6
    n_samples = 3000
    n_features = 500
    plot_result = True
    verbose = True

    X = sklearn_datasets.make_blobs(n_samples=n_samples, 
                                        cluster_std=4, 
                                        centers=n_clusters, 
                                        n_features=n_features, 
                                        return_centers=True,
                                        random_state=int(49))
    data = Matrix.from_numpy(X[0])

    # Common arguments:
    max_iterations = 100

    print("\n======== Mojo Kmeans ========")
    mojo_model = Kmeans(k=n_clusters)

    t = now()
    mojo_centroids = mojo_model.fit(data)
    t_mojo = Float64(now()-t)/1_000_000
    print('Mojo Kmeans complete (ms):',t_mojo)
    
    print("\n======== Python Kmeans ========")
    py_model = py_kmeans.Kmeans(k=n_clusters)

    t = now()
    py_centroids = py_model.fit(X[0])
    t_py = Float64(now()-t)/1_000_000
    print('Python Kmeans complete (ms):',t_py)

    print("\n======== SKLearn Kmeans ========")
    verbose_num = 1
    if not verbose:
        verbose_num = 0
    sklearn_model = sklearn_cluster.KMeans(n_clusters=n_clusters, 
                                            max_iter=max_iterations,
                                            verbose=verbose_num,
                                            tol=0)

    t = now()
    sklearn_centroids = sklearn_model.fit(X[0])
    t_sklearn = Float64(now()-t)/1_000_000
    print('Python Kmeans complete (ms):',t_sklearn)
    
    print("\n======== ONNX Kmeans ========")

    sess = onnxruntime.InferenceSession('./onnx_kmeans/kmeans.onnx')
    feed = Python.dict()
    feed['data'] = X[0]
    feed['n_clusters'] = np.array(n_clusters, np.int64)
    feed['max_iterations'] = np.array(max_iterations, np.int64)
    feed['tol']  = np.array(1e-4, np.float64)

    t = now()
    output_onnxruntime = sess.run(None, feed)
    t_onnx = Float64(now()-t)/1_000_000
    print('ONNX Kmeans complete (ms):',t_onnx)
    
    print()
    print("Config:")
    print("n_clusters =",n_clusters,"\nn_samples = ",n_samples,"\nn_features = ",n_features)

    print()
    print("Speedup Mojo vs. Python:",t_py/t_mojo)
    print("Speedup Mojo vs. SKLearn:",t_sklearn/t_mojo)
    print("Speedup Mojo vs. ONNX:",t_onnx/t_mojo)

    print()
    print("Comparing final inertia:")
    print("Mojo kmeans final inertia:", mojo_model.inertia)
    print("Python kmeans final inertia:", py_model.inertia)
    print("SKlearn kmeans final inertia:", sklearn_model.inertia_)
    print("ONNX kmeans final inertia:", output_onnxruntime[1])

    if plot_result:
        mojo_centroids_matrix = list_to_matrix[data.dtype](mojo_centroids).to_numpy()
        py_utils.plot_clusters(X[0], X[1], mojo_centroids_matrix, py_centroids, output_onnxruntime[0], X[2])
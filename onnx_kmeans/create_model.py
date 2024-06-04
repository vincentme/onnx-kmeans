from onnxscript import opset20 as op, script, DOUBLE, INT64, optimizer
import onnx, onnxsim
import numpy as np

@script()
def kmeans(data: DOUBLE['num_sample', 'num_feature'], n_clusters: INT64, max_iterations: INT64, tol: DOUBLE) -> tuple[DOUBLE['num_cluster', 'num_feature'], DOUBLE]:
    
    # kmeans++ initialization
    num_sample = op.Shape(data)[0]
    random_index = op.Cast((op.RandomUniform(dtype = onnx.TensorProto.DOUBLE, shape = [1])*op.Cast(num_sample, to = onnx.TensorProto.DOUBLE)), to = onnx.TensorProto.INT64)[0]
    random_sample = data[random_index]
    centroid_list = op.SequenceConstruct(random_sample)
    for i in range(n_clusters-1):
        centroid_array = op.Unsqueeze(op.ConcatFromSequence(centroid_list, axis = 0, new_axis=1), axes = [1]) # (num_centroid, 1,  num_feature)
        centroids_distances = op.ReduceSum((centroid_array - data)**2, axes = [2], keepdims = 0) # (num_centroid, num_sample)
        min_distances = op.ReduceMin(centroids_distances, axes=[0], keepdims=0)
        probs = min_distances/op.ReduceSum(min_distances)
        cumulative_probs = op.CumSum(probs, axis = 0)
        rand_prob = op.RandomUniform(dtype = onnx.TensorProto.DOUBLE, shape = [1])
        selected_index = op.ArgMin(op.Abs(cumulative_probs - rand_prob), keepdims = 0)
        centroid_list = op.SequenceInsert(centroid_list, data[selected_index])
    
    # kmeans iteration
    centroid_array = op.Unsqueeze(op.ConcatFromSequence(centroid_list, axis = 0, new_axis=1), axes = [1])
    previous_inertia = op.Constant(value = onnx.helper.make_tensor('zero', onnx.TensorProto.DOUBLE, (), [0.0]))
    inertia = op.Constant(value = onnx.helper.make_tensor('zero', onnx.TensorProto.DOUBLE, (), [0.0]))
    for i in range(max_iterations):
        centroids_distances = op.ReduceSum((centroid_array - data)**2, axes = [2], keepdims = 0) # (num_centroid, num_sample)
        inertia = op.ReduceSum(op.ReduceMin(centroids_distances, axes = [0]), keepdims=0) 
        labels = op.ArgMin(centroids_distances, axis = 0, keepdims = 0)
        
        for j in range(n_clusters):
            belong_to_cluster = op.Equal(labels, j)
            condition = op.Less(op.Constant(value = onnx.helper.make_tensor('zero', onnx.TensorProto.FLOAT16, (), [0.0])), op.ReduceSum(op.Cast(belong_to_cluster, to = onnx.TensorProto.FLOAT16), keepdims=0))
            if condition:
                centroid = op.ReduceMean(op.Compress(data, op.Equal(labels, j), axis=0), axes = [0], keepdims=0)
                centroid_list = op.SequenceErase(centroid_list, j)
                centroid_list = op.SequenceInsert(centroid_list, centroid, j)
        centroid_array = op.Unsqueeze(op.ConcatFromSequence(centroid_list, axis = 0, new_axis=1), axes = [1])
        
        converge = op.Less(op.Abs(inertia - previous_inertia), tol)
        previous_inertia = op.Identity(inertia)
        if converge:
            break
    
    centroid_array = op.ConcatFromSequence(centroid_list, axis = 0, new_axis=1)
    return centroid_array, inertia

    
if __name__ == '__main__':
    import sklearn.datasets as sklearn_datasets
    import time
    from onnxruntime import InferenceSession

    n_clusters = 6
    n_samples = 3000
    n_features = 100
    plot_result = True
    verbose = True

    X = sklearn_datasets.make_blobs(n_samples=n_samples, 
                                        cluster_std=5, 
                                        centers=n_clusters, 
                                        n_features=n_features, 
                                        return_centers=True,
                                        random_state=44)
    data = X[0]
    max_iterations = np.array(100, np.int64)
    tol = np.array(1e-4, np.float64)
    n_clusters = np.array(n_clusters, np.int64)
    
    centroid_array, inertia = kmeans(data, n_clusters, max_iterations, tol) # inference in onnxscript
    model = kmeans
    model_proto = model.to_model_proto()
    
    onnx.checker.check_model(model_proto)
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    model_proto = optimizer.optimize(model_proto)
    res = onnxsim.simplify(model_proto)
    model_proto = res[0]
    onnx.save(model_proto, 'kmeans.onnx')
    
    feeds = {'data':data, 'n_clusters':n_clusters, 'max_iterations':max_iterations, 'tol':tol}
    sess = InferenceSession(model_proto.SerializeToString())

    t = time.time()
    output_onnxruntime = sess.run(None, feeds)
    t_onnx = (time.time()-t)*1000
    print('Onnx Kmeans complete (ms):', t_onnx)

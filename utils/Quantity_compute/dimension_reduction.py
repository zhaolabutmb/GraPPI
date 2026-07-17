def dim_reduce_1d(method, arr, n_neighbors=15, random_state=42):
    """Dispatch to a 1D dimensionality reduction method.

    method: string, e.g. 'umap', 'pca', 'tsne', 'isomap'
    arr: 2D array-like, shape (n_samples, n_features)
    Returns a 1D numpy array of length n_samples.
    """
    import numpy as np
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0.0])

    m = method.lower()

    if m == 'umap':
        import umap
        reducer = umap.UMAP(
            n_components=1,
            n_neighbors=min(n_neighbors, n - 1),
            min_dist=0.0,
            metric='cosine',
            random_state=random_state,
            n_jobs=1,
        )
        return reducer.fit_transform(arr).squeeze()

    if m == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1, random_state=random_state)
        return pca.fit_transform(arr).squeeze()
    if m in ('tsne', 't-sne', 't_sne'):
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=1, metric='cosine', random_state=random_state, init='pca', learning_rate='auto')
        return tsne.fit_transform(arr).squeeze()
    if m == 'isomap':
        from sklearn.manifold import Isomap
        iso = Isomap(n_components=1, n_neighbors=min(n_neighbors, n - 1))
        return iso.fit_transform(arr).squeeze()
    raise ValueError(f"Unknown DR method: {method}. Supported: 'umap','pca','tsne','isomap'.")
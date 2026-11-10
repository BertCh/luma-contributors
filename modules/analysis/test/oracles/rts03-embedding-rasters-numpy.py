# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RTS-03 reference fixture: binary64 NumPy implementations of the embedding-raster operations.

The definitions are the AlphaEarth Satellite Embedding V1 dequantization
(sign(q) * (q / 127.5)^2, NoData -128), cosine / dot similarity, scikit-learn's
KNeighborsClassifier (uniform votes, ties to the smallest label), LogisticRegression
(multinomial softmax of X W^T + b), PCA (covariance with ddof = 1, eigh, svd_flip with
u_based_decision=False so the largest-magnitude loading of each component is positive) and the
AlphaEarth overview rule (average the decoded vectors, then renormalize to unit length).
scikit-learn itself is not installed here, so these are written directly from the documented
definitions with NumPy only; nothing was copied from scikit-learn.
Output: fixtures/rts03-embedding-rasters-numpy.json.
"""
import json
import pathlib
import platform

import numpy as np

OUTPUT = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'rts03-embedding-rasters-numpy.json'
NODATA = -128


def quantize(vectors):
    magnitude = np.rint(127.5 * np.sqrt(np.abs(vectors)))
    return (np.sign(vectors) * np.minimum(magnitude, 127)).astype(np.int64)


def dequantize(quantized):
    """Rows with any NoData component are invalid (NaN)."""
    q = quantized.astype(np.float64)
    decoded = np.sign(q) * (q / 127.5) ** 2
    decoded[(quantized == NODATA).any(axis=-1)] = np.nan
    return decoded


def unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def listify(array):
    """JSON-safe nested list with NaN as null."""
    array = np.asarray(array)
    if array.dtype.kind == 'f':
        return [None if not np.isfinite(v) else float(v) for v in array.ravel()]
    return [int(v) for v in array.ravel()]


def make_blobs(random, count, dimensions, centers, spread):
    labels = random.integers(0, centers.shape[0], size=count)
    vectors = unit(centers[labels] + spread * random.normal(size=(count, dimensions)))
    return vectors, labels


def cosine_matrix(left, right):
    return (left @ right.T) / (
        np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
    )


def knn(valid_vectors, training, labels, k, metric, class_count):
    if metric == 'euclidean':
        distance = np.linalg.norm(valid_vectors[:, None, :] - training[None, :, :], axis=2)
    else:
        distance = 1.0 - cosine_matrix(valid_vectors, training)
    predicted = np.zeros(valid_vectors.shape[0], dtype=np.int64)
    probabilities = np.zeros((valid_vectors.shape[0], class_count))
    smallest_gap = np.inf
    for row in range(valid_vectors.shape[0]):
        order = np.lexsort((np.arange(training.shape[0]), distance[row]))
        smallest_gap = min(smallest_gap, distance[row][order[k]] - distance[row][order[k - 1]])
        votes = np.bincount(labels[order[:k]], minlength=class_count)
        probabilities[row] = votes / k
        predicted[row] = int(np.argmax(votes))  # first maximum is the smallest label
    return predicted, probabilities, smallest_gap


def pca(vectors, component_count):
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    covariance = centered.T @ centered / (vectors.shape[0] - 1)
    values, basis = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1][:component_count]
    components = basis[:, order].T.copy()
    for row in range(components.shape[0]):
        if components[row, np.argmax(np.abs(components[row]))] < 0:
            components[row] *= -1
    return mean, covariance, values[order], components


def zone_means(vectors, tuples):
    zone_count = max(zone for zone, _, _ in tuples) + 1
    sums = np.zeros((zone_count, vectors.shape[1]))
    weights = np.zeros(zone_count)
    for zone, cell, fraction in tuples:
        if np.isfinite(vectors[cell]).all():
            sums[zone] += fraction * vectors[cell]
            weights[zone] += fraction
    means = np.full_like(sums, np.nan)
    lengths = np.full(zone_count, np.nan)
    for zone in range(zone_count):
        if weights[zone] > 0:
            mean = sums[zone] / weights[zone]
            lengths[zone] = np.linalg.norm(mean)
            means[zone] = mean / lengths[zone]
    return means, weights, lengths


def overview(decoded, width, height, scale_x, scale_y):
    out_width = -(-width // scale_x)
    out_height = -(-height // scale_y)
    dimensions = decoded.shape[1]
    vectors = np.full((out_width * out_height, dimensions), np.nan)
    counts = np.zeros(out_width * out_height, dtype=np.int64)
    lengths = np.full(out_width * out_height, np.nan)
    for out_row in range(out_height):
        for out_column in range(out_width):
            total = np.zeros(dimensions)
            for row in range(out_row * scale_y, min((out_row + 1) * scale_y, height)):
                for column in range(out_column * scale_x, min((out_column + 1) * scale_x, width)):
                    vector = decoded[row * width + column]
                    if np.isfinite(vector).all():
                        total += vector
                        counts[out_row * out_width + out_column] += 1
            index = out_row * out_width + out_column
            if counts[index] > 0:
                mean = total / counts[index]
                lengths[index] = np.linalg.norm(mean)
                vectors[index] = mean / lengths[index]
    return {
        'outWidth': out_width,
        'outHeight': out_height,
        'vectors': listify(vectors),
        'validCount': listify(counts),
        'meanLength': listify(lengths)
    }


def main():
    np.seterr(all='ignore')
    random = np.random.default_rng(20261109)
    dimensions = 8
    width, height = 16, 12
    pixel_count = width * height
    centers = unit(random.normal(size=(3, dimensions)))
    year_one, blob = make_blobs(random, pixel_count, dimensions, centers, 0.12)
    quantized = quantize(year_one)
    nodata_pixels = [0, 5, 17, 40, 41, 42, 99, 150, 191]
    quantized[nodata_pixels] = NODATA
    quantized[63, 3] = NODATA  # one NoData band is enough to invalidate a pixel
    decoded = dequantize(quantized)
    valid = np.isfinite(decoded).all(axis=1)

    # Year two: half of the pixels change direction.
    changed = random.random(pixel_count) < 0.5
    year_two = year_one.copy()
    year_two[changed] = unit(year_one[changed] + 0.6 * random.normal(size=(changed.sum(), dimensions)))
    quantized_two = quantize(year_two)
    quantized_two[[3, 63]] = NODATA
    decoded_two = dequantize(quantized_two)
    both = valid & np.isfinite(decoded_two).all(axis=1)
    dot = np.einsum('ij,ij->i', decoded, decoded_two)
    cosine = dot / (np.linalg.norm(decoded, axis=1) * np.linalg.norm(decoded_two, axis=1))
    cosine[~both] = np.nan
    angle = np.arccos(np.clip(cosine, -1, 1))
    magnitude = np.linalg.norm(decoded - decoded_two, axis=1)
    magnitude[~both] = np.nan

    # Similarity against four queries; the last one is long so cosine and dot differ.
    query_pixels = [20, 100, 170]
    queries = np.vstack([decoded[query_pixels], 3.0 * unit(random.normal(size=(1, dimensions)))])
    query_cosine = cosine_matrix(np.nan_to_num(decoded), queries)
    query_dot = np.nan_to_num(decoded) @ queries.T
    query_cosine[~valid] = np.nan
    query_dot[~valid] = np.nan
    best_query = np.where(valid, np.nanargmax(np.where(valid[:, None], query_cosine, -np.inf), axis=1), 0xFFFFFFFF)

    # Top-k nearest pixels by cosine for each query.
    nearest_k = 5
    nearest_ids, nearest_scores = [], []
    for query in range(queries.shape[0]):
        scores = np.where(valid, query_cosine[:, query], -np.inf)
        order = np.lexsort((np.arange(pixel_count), -scores))[:nearest_k]
        nearest_ids += [int(v) for v in order]
        nearest_scores += [float(scores[v]) for v in order]

    # kNN with a labelled set of 24 vectors per metric (retry until the kth gap is comfortable).
    class_count = 3
    valid_vectors = decoded[valid]
    knn_cases = {}
    for attempt in range(200):
        training, training_labels = make_blobs(random, 24, dimensions, centers, 0.2)
        gaps = []
        results = {}
        for metric in ('euclidean', 'cosine'):
            predicted, probabilities, gap = knn(valid_vectors, training, training_labels, 5, metric, class_count)
            results[metric] = (predicted, probabilities)
            gaps.append(gap)
        if min(gaps) > 1e-3:
            break
    else:
        raise RuntimeError('no kNN training set with a clear kth-neighbour gap')
    for metric, (predicted, probabilities) in results.items():
        full_labels = np.full(pixel_count, 0xFFFFFFFF, dtype=np.int64)
        full_labels[valid] = predicted
        full_probabilities = np.full((pixel_count, class_count), np.nan)
        full_probabilities[valid] = probabilities
        knn_cases[metric] = {
            'labels': listify(full_labels),
            'probabilities': listify(full_probabilities),
            'confidence': listify(np.where(valid, full_probabilities.max(axis=1, initial=-1), np.nan))
        }

    # Linear probe: multinomial logistic regression inference.
    weights = random.normal(size=(class_count, dimensions)) * 3
    bias = random.normal(size=class_count)
    logits = np.nan_to_num(decoded) @ weights.T + bias
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    ordered = np.sort(probabilities, axis=1)
    assert (ordered[valid, -1] - ordered[valid, -2] > 1e-3).all()
    probe_labels = np.where(valid, probabilities.argmax(axis=1), 0xFFFFFFFF)
    probabilities[~valid] = np.nan

    # PCA to three components with a +-2 sigma stretch to RGB.
    mean, covariance, eigenvalues, components = pca(valid_vectors, 3)
    projected = (np.nan_to_num(decoded) - mean) @ components.T
    projected[~valid] = np.nan
    low = -2 * np.sqrt(eigenvalues)
    high = 2 * np.sqrt(eigenvalues)
    rgb = np.clip((projected - low) / (high - low), 0, 1)

    # Zonal means: four rectangles, border pixels shared between neighbouring zones with weight 1/2.
    tuples = []
    for row in range(height):
        for column in range(width):
            cell = row * width + column
            zone = (0 if column < 8 else 1) + (0 if row < 6 else 2)
            tuples.append((zone, cell, 1.0))
            if column == 7:
                tuples[-1] = (zone, cell, 0.5)
                tuples.append((zone + 1, cell, 0.5))
    tuples = [t for t in tuples if t[1] % 11 != 0]  # a few cells outside every polygon
    tuple_means, tuple_weights, tuple_lengths = zone_means(decoded, tuples)
    raster_zones = [min(3, (cell % width) // 4) if cell % 7 else 0xFFFFFFFF for cell in range(pixel_count)]
    raster_tuples = [(z, cell, 1.0) for cell, z in enumerate(raster_zones) if z != 0xFFFFFFFF]
    raster_means, raster_weights, raster_lengths = zone_means(decoded, raster_tuples)

    # Overviews of a second, odd-sized raster with six bands (not a multiple of four).
    odd_width, odd_height, odd_dimensions = 9, 7, 6
    odd_vectors, _ = make_blobs(random, odd_width * odd_height, odd_dimensions, unit(random.normal(size=(2, odd_dimensions))), 0.3)
    odd_quantized = quantize(odd_vectors)
    odd_quantized[[4, 5, 20, 21, 22, 40]] = NODATA
    odd_decoded = dequantize(odd_quantized)
    overviews = {
        '2x2': overview(odd_decoded, odd_width, odd_height, 2, 2),
        '3x2': overview(odd_decoded, odd_width, odd_height, 3, 2)
    }

    fixture = {
        'versions': {'numpy': np.__version__, 'python': platform.python_version()},
        'note': 'binary64 NumPy implementations of the AlphaEarth, scikit-learn and PCA definitions; scikit-learn was not run',
        'width': width,
        'height': height,
        'dimensions': dimensions,
        'quantized': listify(quantized),
        'decoded': listify(decoded),
        'quantizedYearTwo': listify(quantized_two),
        'decodedYearTwo': listify(decoded_two),
        'valid': listify(valid.astype(int)),
        'blob': listify(blob),
        'blobMeans': listify(np.array([decoded[valid & (blob == b)].mean(axis=0) for b in range(3)])),
        'change': {'cosine': listify(cosine), 'angle': listify(angle), 'magnitude': listify(magnitude)},
        'queries': listify(queries),
        'queryCount': int(queries.shape[0]),
        'similarity': {
            'cosine': listify(query_cosine),
            'dot': listify(query_dot),
            'bestQuery': listify(best_query),
            'maximumCosine': listify(np.nanmax(np.where(valid[:, None], query_cosine, -np.inf), axis=1) * np.where(valid, 1, np.nan))
        },
        'nearest': {'k': nearest_k, 'ids': nearest_ids, 'scores': nearest_scores},
        'knn': {
            'k': 5,
            'classCount': class_count,
            'training': listify(training),
            'trainingLabels': listify(training_labels),
            'euclidean': knn_cases['euclidean'],
            'cosine': knn_cases['cosine']
        },
        'linearProbe': {
            'weights': listify(weights),
            'bias': listify(bias),
            'labels': listify(probe_labels),
            'probabilities': listify(probabilities)
        },
        'pca': {
            'mean': listify(mean),
            'covariance': listify(covariance),
            'eigenvalues': listify(eigenvalues),
            'components': listify(components),
            'projected': listify(projected),
            'stretch': listify(np.column_stack([low, high])),
            'rgb': listify(rgb),
            'validCount': int(valid.sum())
        },
        'zoneTuples': {
            'zoneIds': [t[0] for t in tuples],
            'cellIds': [t[1] for t in tuples],
            'fractions': [t[2] for t in tuples],
            'means': listify(tuple_means),
            'weights': listify(tuple_weights),
            'meanLengths': listify(tuple_lengths)
        },
        'zoneRaster': {
            'zones': raster_zones,
            'means': listify(raster_means),
            'weights': listify(raster_weights),
            'meanLengths': listify(raster_lengths)
        },
        'oddRaster': {
            'width': odd_width,
            'height': odd_height,
            'dimensions': odd_dimensions,
            'quantized': listify(odd_quantized),
            'decoded': listify(odd_decoded),
            'overviews': overviews
        }
    }
    OUTPUT.write_text(json.dumps(fixture, separators=(',', ':')))
    print(f'wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)')


if __name__ == '__main__':
    main()

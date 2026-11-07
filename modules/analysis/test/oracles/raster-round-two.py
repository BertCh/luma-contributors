# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct pinned-library raster acceptance fixtures; no production imports or shader replicas.
Run /tmp/luma-raster-round-two/bin/python raster-round-two.py after installing the versions below.
"""
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
from scipy.ndimage import fourier_shift
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.registration import phase_cross_correlation
from skimage import graph
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from shapely.geometry import box, mapping
from shapely.ops import unary_union

versions = {'numpy': '2.2.6', 'scipy': '1.15.3', 'scikit-image': '0.25.2', 'exactextract': '0.3.0', 'shapely': '2.1.2'}
for package, expected in versions.items():
    assert version(package) == expected

# graycomatrix computes the actual pairs. A sentinel category represents QA failures;
# removing its row/column retains only valid pairs before library normalization/properties.
width, height, radius, levels = 11, 9, 2, 8
image = np.fromfunction(lambda row, column: (3 * column + row * row + column * row) % levels, (height, width)).astype(np.uint8)
validity = np.ones_like(image)
validity[2, 5] = validity[4, 4] = validity[6, 7] = 0
texture_cases = []
for offset, symmetric in [([1, 0], False), ([1, 1], True), ([-1, 1], True)]:
    features, counts = [], []
    for row in range(height):
        for column in range(width):
            rows = slice(max(0, row-radius), min(height, row+radius+1))
            columns = slice(max(0, column-radius), min(width, column+radius+1))
            window = np.where(validity[rows, columns], image[rows, columns], levels).astype(np.uint8)
            histogram = graycomatrix(window, [np.hypot(*offset)], [np.arctan2(offset[1], offset[0])], levels=levels+1, symmetric=symmetric, normed=False)[:levels, :levels]
            counts.append(int(histogram.sum()))
            features.extend([float(graycoprops(histogram, name)[0, 0]) for name in ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']])
            features.append(float(entropy(histogram.flatten())))
    texture_cases.append(dict(offset=offset, symmetric=symmetric, features=features, counts=counts))

rng = np.random.default_rng(64983)
size = 32
reference = rng.normal(size=(size,size)).astype(np.float32)
# No Nyquist energy, so a real periodic subpixel translation remains a real signal.
spectrum = np.fft.fftn(reference); spectrum[size//2,:] = 0; spectrum[:,size//2] = 0
reference = np.fft.ifftn(spectrum).real.astype(np.float32)
registration = []
for shift in [(1.35, -2.2), (-3.1, 0.65)]:
    moving = np.fft.ifftn(fourier_shift(np.fft.fftn(reference), shift)).real.astype(np.float32)
    result, error, phase = phase_cross_correlation(reference, moving, upsample_factor=20, normalization='phase')
    registration.append(dict(values=moving.flatten().tolist(), shift=result[::-1].tolist()))

labels = np.repeat(np.arange(5)[None,:], 4, axis=0)
region_values = np.array([0, 0.6, 2, 2.7, 8], dtype=np.float32)
image_regions = np.repeat(region_values[labels][...,None], 3, axis=2)
region_graph = graph.rag_mean_color(image_regions, labels, connectivity=1, mode='distance')
# rag_mean_color repeats our scalar channel three times; normalize to scalar difference.
for first, second in region_graph.edges:
    region_graph[first][second]['weight'] /= np.sqrt(3)
adjacency = np.zeros((5,5), dtype=int)
for first, second in region_graph.edges: adjacency[first,second] = adjacency[second,first] = 4

def merge_regions(region_graph, source, destination):
    region_graph.nodes[destination]['total color'] += region_graph.nodes[source]['total color']
    region_graph.nodes[destination]['pixel count'] += region_graph.nodes[source]['pixel count']
    region_graph.nodes[destination]['mean color'] = region_graph.nodes[destination]['total color'] / region_graph.nodes[destination]['pixel count']
def weight_regions(region_graph, source, destination, neighbor):
    return {'weight': float(np.linalg.norm(region_graph.nodes[destination]['mean color'] - region_graph.nodes[neighbor]['mean color']) / np.sqrt(3))}
merged = graph.merge_hierarchical(labels, region_graph, thresh=0.8, rag_copy=True, in_place_merge=True, merge_func=merge_regions, weight_func=weight_regions)
canonical = [min(np.flatnonzero(merged[0] == merged[0, index])).item() for index in range(5)]

slices, period, minimum_segment = 48, 12, 6
time = np.arange(slices)
design = np.column_stack([np.ones(slices), time/slices, np.cos(2*np.pi*time/period), np.sin(2*np.pi*time/period)])
seasonal = []
for boundary, amplitude in [(23, 3.5), (None, 0), (35, -2)]:
    observations = design @ np.array([2, 1.7, 0.8, -0.4]) + rng.normal(0, 0.06, slices)
    if boundary is not None: observations[boundary:] += amplitude
    observations = observations.astype(np.float32).astype(float)
    mask = np.ones(slices, dtype=bool); mask[[3, 9, 28, 40]] = False
    matrix, response = design[mask], observations[mask]
    baseline = np.linalg.lstsq(matrix, response, rcond=None)[0]
    floor = max(1e-20, np.dot(response,response)*1e-12)
    baseline_score = len(response)*np.log(max(np.sum((response-matrix@baseline)**2),floor)/len(response)) + 4*np.log(len(response))
    chosen, gain, best = np.r_[baseline,0], 0., 0xffffffff
    for split in range(minimum_segment, slices-minimum_segment+1):
        if mask[:split].sum() < minimum_segment or mask[split:].sum() < minimum_segment: continue
        candidate = np.column_stack([matrix,(time>=split)[mask]])
        coefficients = np.linalg.lstsq(candidate, response, rcond=None)[0]
        score = len(response)*np.log(max(np.sum((response-candidate@coefficients)**2),floor)/len(response)) + 6*np.log(len(response))
        if baseline_score-score > gain: chosen, gain, best = coefficients, baseline_score-score, split
    seasonal.append(dict(values=observations.tolist(), validity=mask.astype(int).tolist(), coefficients=chosen.tolist(), improvement=gain, breakIndex=int(best)))

values = np.array([[1,3,9,100]], dtype=np.float32)
observation_weights = np.array([[2,4,8,0]], dtype=np.float32)
fractions = [1,.5,.25,1]
feature = {'type':'Feature','properties':{},'geometry':mapping(unary_union([box(index,0,index+fraction,1) for index,fraction in enumerate(fractions)]))}
weighted = exact_extract(NumPyRasterSource(values,0,0,4,1),[feature], ['weighted_mean','weighted_variance','weighted_sum'],weights=NumPyRasterSource(observation_weights,0,0,4,1))[0]['properties']
weights = np.array(fractions)*observation_weights[0]
weighted['quantiles'] = np.quantile(values[0],[0,.25,.5,.75,1],method='inverted_cdf',weights=weights).tolist()
weighted['sampleVariance'] = float(np.cov(values[0],aweights=weights,ddof=1))
scale_count = 16384
scale_index = np.arange(scale_count)
scale_values = ((scale_index % 61) / 7).astype(np.float32).astype(float)
scale_fractions = ((scale_index % 7 + 1) / 8).astype(np.float32)
scale_observation_weights = ((scale_index % 19 + 1) / 11).astype(np.float32)
scale_weights = (scale_fractions * scale_observation_weights).astype(float)
scale_mean = np.average(scale_values,weights=scale_weights)
scale_weight_sum = scale_weights.sum()
scale_squared_weight_sum = np.dot(scale_weights,scale_weights)
weighted_scale = dict(count=scale_count,probabilities=[.05,.5,.95],means=float(scale_mean),populationVariances=float(np.average((scale_values-scale_mean)**2,weights=scale_weights)),sampleVariances=float(np.cov(scale_values,aweights=scale_weights,ddof=1)),weightSums=float(scale_weight_sum),squaredWeightSums=float(scale_squared_weight_sum),effectiveSampleSizes=float(scale_weight_sum**2/scale_squared_weight_sum),quantiles=np.quantile(scale_values,[.05,.5,.95],method='inverted_cdf',weights=scale_weights).tolist())
fixture = dict(versions=versions, texture=dict(width=width,height=height,radius=radius,levels=levels,values=image.flatten().tolist(),validity=validity.flatten().tolist(),cases=texture_cases), registration=dict(size=size,reference=reference.flatten().tolist(),cases=registration), regions=dict(width=5,height=4,labels=labels.flatten().tolist(),values=region_values[labels].flatten().tolist(),adjacency=adjacency.flatten().tolist(),mapping=canonical), seasonal=dict(slices=slices,period=period,minimumSegmentLength=minimum_segment,cases=seasonal),weighted=weighted,weightedScale=weighted_scale)
Path(__file__).with_name('fixtures').joinpath('raster-round-two.json').write_text(json.dumps(fixture,indent=2)+'\n')
print(json.dumps({'versions':versions,'cases':{'texture':len(texture_cases),'registration':len(registration),'seasonal':len(seasonal),'regions':1,'weighted':1}},indent=2))

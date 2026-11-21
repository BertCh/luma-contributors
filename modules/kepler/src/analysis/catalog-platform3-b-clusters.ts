// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  GPUDBSCAN,
  getGPUDBSCANParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {
  addDroppedDiagnostic,
  collectAllPositions,
  getFeatureGeometryText,
  readNumericProperty,
  readPointSamples,
  type Sample
} from './catalog-platform3-b-common';
import {makeLocalFrame, type LocalFrame} from './catalog-local-frame';
import {buildAnalysisSet, type WeightsSettings} from './catalog-platform3-b-statistics';

const MAXIMUM_PAIR_WORK = 4e7;
const NOISE_LABEL = 0xffffffff;

function assertPairWork(count: number): void {
  // Pair work exceeds the CPU reference capacity; reduce the number of points or use the GPU engine.
  if (count * count > MAXIMUM_PAIR_WORK)
    throw new KeplerAllocationError('capacity', {message: 'Work capacity exceeded'});
}

function makePointText(frame: LocalFrame, x: number, y: number): string {
  return makeCatalogGeometryText('Point', frame.inverse(x, y));
}

// ---------------------------------------------------------------------------------------------
// DBSCAN (GPUDBSCAN)
// ---------------------------------------------------------------------------------------------

export type DbscanResult = {
  /** Compact cluster label per sample, -1 for noise. */
  labels: Int32Array;
  coreFlags: Uint8Array;
};

export const MAXIMUM_DBSCAN_CPU_POINTS = 20000;

/** Bucketed CPU DBSCAN has a point cap, independent of the matrix routes' pair-work cap. */
export function isDbscanCpuPointCountAdmitted(count: number): boolean {
  return Number.isInteger(count) && count >= 0 && count <= MAXIMUM_DBSCAN_CPU_POINTS;
}

/** Round the coordinate difference once, after compensating only measured translation errors. */
function getDbscanCoordinateDifference(
  first: number,
  second: number,
  firstError = 0,
  secondError = 0
): number {
  const difference = first - second;
  const virtualSecond = first - difference;
  const differenceError = first - (difference + virtualSecond) + (virtualSecond - second);
  return difference + (differenceError + (firstError - secondError));
}

/** Canonical bucketed CPU DBSCAN, at most MAXIMUM_DBSCAN_CPU_POINTS = 20000 total points. */
export function computeDbscanLabels(
  sample: Sample,
  epsilon: number,
  minimumSamples: number
): DbscanResult {
  const n = sample.count;
  validateRequirement(
    'density-clustering:',
    isDbscanCpuPointCountAdmitted(n),
    `points count must be at most ${MAXIMUM_DBSCAN_CPU_POINTS}; received ${n}; reduce or partition the input.`
  );
  const epsilonSquared = epsilon * epsilon;
  // Widen bucket lookup only; the radius check compensates measured translation errors below.
  let maximumTranslationError = 0;
  for (let index = 0; index < n; index++)
    maximumTranslationError = Math.max(
      maximumTranslationError,
      Math.abs(sample.translationErrorX?.[index] ?? 0),
      Math.abs(sample.translationErrorY?.[index] ?? 0)
    );
  const cellSize = epsilon + 2 * maximumTranslationError;
  const buckets = new Map<string, number[]>();
  for (let index = 0; index < n; index++) {
    const key = `${Math.floor(sample.x[index] / cellSize)}:${Math.floor(sample.y[index] / cellSize)}`;
    const bucket = buckets.get(key);
    if (bucket) bucket.push(index);
    else buckets.set(key, [index]);
  }
  // Query on demand: even a dense neighborhood uses linear memory at the full 20,000 point cap.
  function* neighbors(index: number): Generator<number> {
    const column = Math.floor(sample.x[index] / cellSize);
    const row = Math.floor(sample.y[index] / cellSize);
    const keys = new Set<string>();
    for (let columnOffset = -1; columnOffset <= 1; columnOffset++)
      for (let rowOffset = -1; rowOffset <= 1; rowOffset++)
        keys.add(`${column + columnOffset}:${row + rowOffset}`);
    for (const key of keys)
      for (const other of buckets.get(key) ?? []) {
        // Residuals are bounded by half an ulp of each translated coordinate, and zero when
        // translation was exact. No allowance is applied to an untranslated distance.
        const differenceX = getDbscanCoordinateDifference(
          sample.x[index],
          sample.x[other],
          sample.translationErrorX?.[index],
          sample.translationErrorX?.[other]
        );
        const differenceY = getDbscanCoordinateDifference(
          sample.y[index],
          sample.y[other],
          sample.translationErrorY?.[index],
          sample.translationErrorY?.[other]
        );
        if (differenceX ** 2 + differenceY ** 2 <= epsilonSquared) yield other;
      }
  }
  const coreFlags = new Uint8Array(n);
  for (let index = 0; index < n; index++) {
    let count = 0;
    for (const _other of neighbors(index)) {
      count++;
      if (count >= minimumSamples) {
        coreFlags[index] = 1;
        break;
      }
    }
  }
  const parent = Int32Array.from({length: n}, (_, index) => index);
  const find = (index: number): number => {
    while (parent[index] !== index) {
      parent[index] = parent[parent[index]];
      index = parent[index];
    }
    return index;
  };
  for (let index = 0; index < n; index++)
    if (coreFlags[index])
      for (const other of neighbors(index))
        if (coreFlags[other]) {
          const a = find(index);
          const b = find(other);
          if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
        }
  const roots = new Map<number, number>();
  for (let index = 0; index < n; index++)
    if (coreFlags[index]) {
      const root = find(index);
      if (!roots.has(root)) roots.set(root, -1);
    }
  [...roots.keys()].sort((a, b) => a - b).forEach((root, label) => roots.set(root, label));
  const labels = new Int32Array(n).fill(-1);
  for (let index = 0; index < n; index++) {
    if (coreFlags[index]) labels[index] = roots.get(find(index))!;
    else {
      let best = -1;
      let bestRoot = Infinity;
      for (const other of neighbors(index))
        if (coreFlags[other]) {
          const root = find(other);
          if (root < bestRoot) {
            bestRoot = root;
            best = roots.get(root)!;
          }
        }
      labels[index] = best;
    }
  }
  return {labels, coreFlags};
}

export async function computeDbscanLabelsGpu(
  context: CatalogGpuContext,
  sample: Sample,
  epsilon: number,
  minimumSamples: number
): Promise<DbscanResult> {
  const n = sample.count;
  const positions = new Float32Array(n * 2);
  let maxX = 0;
  let maxY = 0;
  for (let index = 0; index < n; index++) {
    positions[2 * index] = sample.x[index];
    positions[2 * index + 1] = sample.y[index];
    maxX = Math.max(maxX, sample.x[index]);
    maxY = Math.max(maxY, sample.y[index]);
  }
  const side = Math.max(1, Math.min(256, Math.ceil(Math.sqrt(n))));
  return withCatalogGpuSession(context.device, 'catalog-density-clustering', async session => {
    const labels = session.output('labels', 'uint32', n);
    const coreFlags = session.output('core-flags', 'uint32', n);
    session.graph.add(
      new GPUDBSCAN({
        id: 'catalog-density-clustering',
        positions: session.input('positions', positions, 'float32x2'),
        parameters: session.input(
          'parameters',
          getGPUDBSCANParameterValues({
            bounds: [-1, -1, maxX + 1, maxY + 1],
            epsilon,
            minimumSamples
          }),
          'float32'
        ),
        gridSize: [side, side],
        output: {labels: labels.view, coreFlags: coreFlags.view}
      })
    );
    session.run();
    const rawLabels = await session.readUint32(labels);
    const rawCore = await session.readUint32(coreFlags);
    return {
      labels: Int32Array.from(rawLabels, label => (label === NOISE_LABEL ? -1 : label)),
      coreFlags: Uint8Array.from(rawCore, flag => (flag ? 1 : 0))
    };
  });
}

export function makeClusterOutput(
  points: CatalogFeatureCollection,
  frame: LocalFrame,
  sample: Sample,
  labels: ArrayLike<number>,
  coreFlags: ArrayLike<number>,
  diagnostics: CatalogDiagnostic[],
  extra: {
    columns?: Record<string, ArrayLike<number>>;
    dispatchedContributors?: readonly string[];
    summary?: Record<string, string | number>;
  } = {}
): CatalogRunOutput {
  const extraColumns = extra.columns ?? {};
  const pointTable = makeCatalogTable(
    'points',
    'Clustered points',
    {
      point_index: 'int32',
      source_index: 'int32',
      source_id: 'string',
      cluster: 'int32',
      is_core: 'boolean',
      ...Object.fromEntries(Object.keys(extraColumns).map(name => [name, 'float64' as const])),
      geometry: 'string'
    },
    'geometry'
  );
  const clusterStats = new Map<number, {size: number; sumX: number; sumY: number}>();
  for (let index = 0; index < sample.count; index++) {
    appendCatalogRow(pointTable, {
      point_index: index,
      source_index: sample.sourceIndex[index],
      source_id: getFeatureIdentity(points.features[sample.sourceIndex[index]]),
      cluster: labels[index],
      is_core: coreFlags[index] === 1,
      ...Object.fromEntries(
        Object.entries(extraColumns).map(([name, column]) => [
          name,
          Number.isFinite(column[index]) ? column[index] : null
        ])
      ),
      geometry: makePointText(frame, sample.x[index], sample.y[index])
    });
    if (labels[index] >= 0) {
      const entry = clusterStats.get(labels[index]) ?? {size: 0, sumX: 0, sumY: 0};
      entry.size++;
      entry.sumX += sample.x[index];
      entry.sumY += sample.y[index];
      clusterStats.set(labels[index], entry);
    }
  }
  const clusterTable = makeCatalogTable(
    'clusters',
    'Clusters',
    {cluster: 'int32', size: 'int32', geometry: 'string'},
    'geometry'
  );
  for (const [cluster, entry] of [...clusterStats.entries()].sort((a, b) => a[0] - b[0]))
    appendCatalogRow(clusterTable, {
      cluster,
      size: entry.size,
      geometry: makePointText(frame, entry.sumX / entry.size, entry.sumY / entry.size)
    });
  let noiseCount = 0;
  for (let index = 0; index < sample.count; index++) if (labels[index] < 0) noiseCount++;
  return {
    tables: [pointTable, clusterTable],
    diagnostics,
    summary: {
      clusterCount: clusterTable.rowCount,
      noiseCount,
      pointCount: sample.count,
      ...(extra.summary ?? {})
    },
    ...(extra.dispatchedContributors ? {dispatchedContributors: extra.dispatchedContributors} : {})
  };
}

export function readClusterSamples(
  points: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
) {
  const frame = makeLocalFrame(coordinateSystem, collectAllPositions(points), {shift: 'minimum'});
  const sample = readPointSamples(points, '', frame);
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedDiagnostic(
    diagnostics,
    sample.droppedCount,
    'Features without a point position are skipped.'
  );
  // No usable point samples; nothing to cluster.
  if (!sample.count) throw new Error('No usable samples');
  return {frame, sample, diagnostics};
}

// ---------------------------------------------------------------------------------------------
// Distance helpers for the pairwise references
// ---------------------------------------------------------------------------------------------

export function getDistanceMatrix(sample: Sample): Float64Array {
  const n = sample.count;
  assertPairWork(n);
  const distances = new Float64Array(n * n);
  for (let first = 0; first < n; first++)
    for (let second = first + 1; second < n; second++) {
      const distance = Math.hypot(
        sample.x[first] - sample.x[second],
        sample.y[first] - sample.y[second]
      );
      distances[first * n + second] = distance;
      distances[second * n + first] = distance;
    }
  return distances;
}

/** Sorted distances to the other points of one row. */
export function getSortedNeighbors(distances: Float64Array, n: number, row: number) {
  const list: {distance: number; index: number}[] = [];
  for (let other = 0; other < n; other++)
    if (other !== row) list.push({distance: distances[row * n + other], index: other});
  return list.sort((a, b) => a.distance - b.distance || a.index - b.index);
}

// ---------------------------------------------------------------------------------------------
// OPTICS
// ---------------------------------------------------------------------------------------------

export function runOptics(
  points: CatalogFeatureCollection,
  parameters: {
    maximumDistance: number;
    minimumSamples: number;
    extractionDistance: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogRunOutput {
  const {frame, sample, diagnostics} = readClusterSamples(points, parameters.coordinateSystem);
  const n = sample.count;
  const distances = getDistanceMatrix(sample);
  const coreDistance = new Float64Array(n).fill(Infinity);
  for (let row = 0; row < n; row++) {
    // The point itself counts toward minimumSamples, as in the original definition.
    const within = getSortedNeighbors(distances, n, row).filter(
      entry => entry.distance <= parameters.maximumDistance
    );
    if (within.length + 1 >= parameters.minimumSamples)
      coreDistance[row] =
        parameters.minimumSamples <= 1 ? 0 : within[parameters.minimumSamples - 2].distance;
  }
  const reachability = new Float64Array(n).fill(Infinity);
  const processed = new Uint8Array(n);
  const order: number[] = [];
  for (let start = 0; start < n; start++) {
    if (processed[start]) continue;
    const seeds = new Set<number>([start]);
    reachability[start] = Infinity;
    while (seeds.size) {
      let current = -1;
      for (const candidate of seeds)
        if (
          current < 0 ||
          reachability[candidate] < reachability[current] ||
          (reachability[candidate] === reachability[current] && candidate < current)
        )
          current = candidate;
      seeds.delete(current);
      processed[current] = 1;
      order.push(current);
      if (!Number.isFinite(coreDistance[current])) continue;
      for (let other = 0; other < n; other++) {
        if (processed[other] || other === current) continue;
        const distance = distances[current * n + other];
        if (distance > parameters.maximumDistance) continue;
        const reach = Math.max(coreDistance[current], distance);
        if (reach < reachability[other] || !seeds.has(other)) {
          if (reach < reachability[other]) reachability[other] = reach;
          seeds.add(other);
        }
      }
    }
  }
  const labels = new Int32Array(n).fill(-1);
  let cluster = -1;
  const cut = parameters.extractionDistance;
  for (const point of order) {
    if (reachability[point] > cut) {
      if (coreDistance[point] <= cut) labels[point] = ++cluster;
    } else labels[point] = cluster;
  }
  const orderPosition = new Float64Array(n);
  order.forEach((point, position) => (orderPosition[point] = position));
  const result = makeClusterOutput(
    points,
    frame,
    sample,
    labels,
    Uint8Array.from(coreDistance, value => (Number.isFinite(value) ? 1 : 0)),
    diagnostics,
    {
      columns: {
        optics_order: orderPosition,
        reachability_distance: Float64Array.from(reachability),
        core_distance: Float64Array.from(coreDistance)
      }
    }
  );
  result.tables[0].id = 'points';
  return result;
}

// ---------------------------------------------------------------------------------------------
// Local outlier factor
// ---------------------------------------------------------------------------------------------

export function runLocalOutlierFactor(
  points: CatalogFeatureCollection,
  parameters: {neighborCount: number; threshold: number; coordinateSystem: CatalogCoordinateSystem},
  /** Scores computed elsewhere (the GPU engine) for the same samples; the CPU density formulas are skipped. */
  precomputed?: {
    read: ReturnType<typeof readClusterSamples>;
    kDistance: ArrayLike<number>;
    lof: ArrayLike<number>;
  }
): CatalogRunOutput {
  const {frame, sample, diagnostics} =
    precomputed?.read ?? readClusterSamples(points, parameters.coordinateSystem);
  const n = sample.count;
  // The neighbor count must be smaller than the number of points.
  if (n <= parameters.neighborCount) throw new Error('Too few features');
  const k = parameters.neighborCount;
  const distances = precomputed ? new Float64Array(0) : getDistanceMatrix(sample);
  const neighborLists = precomputed
    ? []
    : Array.from({length: n}, (_, row) => getSortedNeighbors(distances, n, row).slice(0, k));
  const kDistance = precomputed
    ? Array.from(precomputed.kDistance)
    : neighborLists.map(list => list[k - 1].distance);
  const localReachabilityDensity = (precomputed ? [] : neighborLists).map(list => {
    const total = list.reduce(
      (sum, entry) => sum + Math.max(kDistance[entry.index], entry.distance),
      0
    );
    return total === 0 ? Infinity : k / total;
  });
  const scores = new Float64Array(n);
  const flags = new Float64Array(n);
  let duplicateCount = 0;
  for (let row = 0; row < n; row++) {
    if (precomputed) {
      scores[row] = precomputed.lof[row];
      flags[row] = scores[row] > parameters.threshold ? 1 : 0;
      continue;
    }
    const own = localReachabilityDensity[row];
    const mean =
      neighborLists[row].reduce((sum, entry) => sum + localReachabilityDensity[entry.index], 0) / k;
    if (!Number.isFinite(own) || !Number.isFinite(mean)) {
      scores[row] = 1;
      duplicateCount++;
    } else scores[row] = mean / own;
    flags[row] = scores[row] > parameters.threshold ? 1 : 0;
  }
  if (duplicateCount)
    addCatalogDiagnostic(
      diagnostics,
      'coincident-points',
      'warning',
      'Points whose neighbors all coincide have infinite density; their factor is set to one.',
      duplicateCount
    );
  const table = makeCatalogTable(
    'outliers',
    'Local outlier factor',
    {
      point_index: 'int32',
      source_index: 'int32',
      source_id: 'string',
      lof: 'float64',
      is_outlier: 'boolean',
      k_distance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  for (let index = 0; index < n; index++)
    appendCatalogRow(table, {
      point_index: index,
      source_index: sample.sourceIndex[index],
      source_id: getFeatureIdentity(points.features[sample.sourceIndex[index]]),
      lof: scores[index],
      is_outlier: flags[index] === 1,
      k_distance: kDistance[index],
      geometry: makePointText(frame, sample.x[index], sample.y[index])
    });
  return {
    tables: [table],
    diagnostics,
    summary: {
      pointCount: n,
      outlierCount: flags.reduce((count, value) => count + value, 0)
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Mutual-reachability density clustering (flat cut of the HDBSCAN* hierarchy)
// ---------------------------------------------------------------------------------------------

export function runHDBSCANCut(
  points: CatalogFeatureCollection,
  parameters: {
    minimumSamples: number;
    minimumClusterSize: number;
    cutDistance: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogRunOutput {
  const {frame, sample, diagnostics} = readClusterSamples(points, parameters.coordinateSystem);
  addCatalogDiagnostic(
    diagnostics,
    'flat-density-cut',
    'info',
    'hierarchical-density-clustering: this result is a single cutDistance cut of the mutual-reachability tree, without HDBSCAN stability selection.'
  );
  const n = sample.count;
  // The core-distance neighbor count must be smaller than the number of points.
  if (n <= parameters.minimumSamples) throw new Error('Too few features');
  const distances = getDistanceMatrix(sample);
  const coreDistance = new Float64Array(n);
  for (let row = 0; row < n; row++)
    coreDistance[row] = getSortedNeighbors(distances, n, row)[
      parameters.minimumSamples - 1
    ].distance;
  const mutual = (a: number, b: number) =>
    Math.max(coreDistance[a], coreDistance[b], distances[a * n + b]);
  // Prim minimum spanning tree over mutual reachability.
  const inTree = new Uint8Array(n);
  const best = new Float64Array(n).fill(Infinity);
  const bestFrom = new Int32Array(n).fill(-1);
  best[0] = 0;
  const edges: {a: number; b: number; weight: number}[] = [];
  for (let step = 0; step < n; step++) {
    let next = -1;
    for (let index = 0; index < n; index++)
      if (!inTree[index] && (next < 0 || best[index] < best[next])) next = index;
    inTree[next] = 1;
    if (bestFrom[next] >= 0) edges.push({a: bestFrom[next], b: next, weight: best[next]});
    for (let index = 0; index < n; index++)
      if (!inTree[index]) {
        const weight = mutual(next, index);
        if (weight < best[index]) {
          best[index] = weight;
          bestFrom[index] = next;
        }
      }
  }
  const parent = Int32Array.from({length: n}, (_, index) => index);
  const find = (index: number): number => {
    while (parent[index] !== index) {
      parent[index] = parent[parent[index]];
      index = parent[index];
    }
    return index;
  };
  for (const edge of edges)
    if (edge.weight <= parameters.cutDistance) {
      const a = find(edge.a);
      const b = find(edge.b);
      if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
    }
  const sizes = new Map<number, number>();
  for (let index = 0; index < n; index++) sizes.set(find(index), (sizes.get(find(index)) ?? 0) + 1);
  const rootLabel = new Map<number, number>();
  [...sizes.keys()]
    .filter(root => sizes.get(root)! >= parameters.minimumClusterSize)
    .sort((a, b) => a - b)
    .forEach((root, label) => rootLabel.set(root, label));
  const labels = new Int32Array(n);
  for (let index = 0; index < n; index++) labels[index] = rootLabel.get(find(index)) ?? -1;
  addCatalogDiagnostic(
    diagnostics,
    'flat-cut',
    'info',
    'Clusters are the connected components of the mutual-reachability spanning tree at the cut distance; there is no stability-based cluster selection.'
  );
  return makeClusterOutput(points, frame, sample, labels, new Uint8Array(n), diagnostics, {
    columns: {core_distance: coreDistance}
  });
}

// ---------------------------------------------------------------------------------------------
// Similar locations
// ---------------------------------------------------------------------------------------------

export function runSimilarLocations(
  collection: CatalogFeatureCollection,
  parameters: {properties: string; referenceIndex: number; resultCount: number}
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const names = parameters.properties
    .split(',')
    .map(name => name.trim())
    .filter(Boolean);
  // At least one attribute is needed to compare locations.
  if (!names.length) throw new Error('Properties required');
  const rows: {index: number; values: number[]}[] = [];
  let droppedCount = 0;
  collection.features.forEach((feature, index) => {
    const values = names.map(name => readNumericProperty(feature, name));
    if (values.some(value => value === null)) droppedCount++;
    else rows.push({index, values: values as number[]});
  });
  addDroppedDiagnostic(
    diagnostics,
    droppedCount,
    'Features missing a numeric attribute are skipped.'
  );
  const reference = rows.find(row => row.index === parameters.referenceIndex);
  // The reference feature index must exist and have every attribute.
  if (!reference) throw new Error('Reference required');
  const means = names.map(
    (_, column) => rows.reduce((sum, row) => sum + row.values[column], 0) / rows.length
  );
  const deviations = names.map((_, column) =>
    Math.sqrt(
      rows.reduce((sum, row) => sum + (row.values[column] - means[column]) ** 2, 0) / rows.length
    )
  );
  const standardize = (row: {values: number[]}) =>
    row.values.map((value, column) =>
      deviations[column] ? (value - means[column]) / deviations[column] : 0
    );
  const target = standardize(reference);
  const scored = rows.map(row => ({
    row,
    distance: Math.sqrt(
      standardize(row).reduce((sum, value, column) => sum + (value - target[column]) ** 2, 0)
    )
  }));
  const ranked = scored
    .filter(entry => entry.row.index !== reference.index)
    .sort((a, b) => a.distance - b.distance || a.row.index - b.row.index);
  const rankOf = new Map<number, number>();
  ranked.forEach((entry, position) => rankOf.set(entry.row.index, position + 1));
  const table = makeCatalogTable(
    'similarity',
    'Similar locations',
    {
      source_index: 'int32',
      source_id: 'string',
      distance: 'float64',
      rank: 'int32',
      is_reference: 'boolean',
      is_similar: 'boolean',
      geometry: 'string'
    },
    'geometry'
  );
  for (const entry of scored) {
    const rank = rankOf.get(entry.row.index) ?? 0;
    appendCatalogRow(table, {
      source_index: entry.row.index,
      source_id: getFeatureIdentity(collection.features[entry.row.index]),
      distance: entry.distance,
      rank,
      is_reference: entry.row.index === reference.index,
      is_similar: rank >= 1 && rank <= parameters.resultCount,
      geometry: getFeatureGeometryText(collection.features[entry.row.index])
    });
  }
  addCatalogDiagnostic(
    diagnostics,
    'standardized-euclidean',
    'info',
    'Similarity is the Euclidean distance between attributes standardized to zero mean and unit variance; geography is ignored.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {featureCount: rows.length, referenceIndex: reference.index}
  };
}

// ---------------------------------------------------------------------------------------------
// Location allocation (p-median)
// ---------------------------------------------------------------------------------------------

/**
 * Demand weights must be non-negative: a negative weight would reward distance and invert the
 * p-median objective. Such features are skipped and added to `droppedCount` (once per feature).
 */
function removeNegativeWeights(sample: Sample): Sample {
  const kept: number[] = [];
  const droppedFeatures = new Set<number>();
  for (let index = 0; index < sample.count; index++)
    if (sample.value[index] >= 0) kept.push(index);
    else droppedFeatures.add(sample.sourceIndex[index]);
  if (!droppedFeatures.size) return sample;
  const pickFloat = (values: Float64Array) => Float64Array.from(kept, index => values[index]);
  const pickInteger = (values: Int32Array) => Int32Array.from(kept, index => values[index]);
  return {
    x: pickFloat(sample.x),
    y: pickFloat(sample.y),
    ...(sample.translationErrorX ? {translationErrorX: pickFloat(sample.translationErrorX)} : {}),
    ...(sample.translationErrorY ? {translationErrorY: pickFloat(sample.translationErrorY)} : {}),
    value: pickFloat(sample.value),
    sourceIndex: pickInteger(sample.sourceIndex),
    partIndex: pickInteger(sample.partIndex),
    count: kept.length,
    droppedCount: sample.droppedCount + droppedFeatures.size
  };
}

export function runLocationAllocation(
  demand: CatalogFeatureCollection,
  candidates: CatalogFeatureCollection,
  parameters: {
    facilityCount: number;
    weightProperty: string;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(
    parameters.coordinateSystem,
    [...collectAllPositions(demand), ...collectAllPositions(candidates)],
    {shift: 'minimum'}
  );
  const demandSample = removeNegativeWeights(
    readPointSamples(demand, parameters.weightProperty, frame)
  );
  const candidateSample = readPointSamples(candidates, '', frame);
  addDroppedDiagnostic(
    diagnostics,
    demandSample.droppedCount + candidateSample.droppedCount,
    'Features without a point position or with a missing, non-numeric or negative weight are skipped.'
  );
  const demandCount = demandSample.count;
  const candidateCount = candidateSample.count;
  // Both demand and candidate points are required, with at least as many candidates as facilities.
  if (!demandCount || candidateCount < parameters.facilityCount)
    throw new Error('Too few features');
  const p = parameters.facilityCount;
  // Demand-times-candidate-times-facility work per swap pass exceeds the CPU reference capacity.
  if (demandCount * candidateCount * p > 5e7)
    throw new KeplerAllocationError('capacity', {message: 'Work capacity exceeded'});
  const cost = new Float64Array(demandCount * candidateCount);
  for (let d = 0; d < demandCount; d++)
    for (let c = 0; c < candidateCount; c++)
      cost[d * candidateCount + c] =
        demandSample.value[d] *
        Math.hypot(
          demandSample.x[d] - candidateSample.x[c],
          demandSample.y[d] - candidateSample.y[c]
        );
  const totalCost = (selected: readonly number[]) => {
    let total = 0;
    for (let d = 0; d < demandCount; d++) {
      let best = Infinity;
      for (const c of selected) best = Math.min(best, cost[d * candidateCount + c]);
      total += best;
    }
    return total;
  };
  const selected: number[] = [];
  while (selected.length < p) {
    let bestCandidate = -1;
    let bestCost = Infinity;
    for (let c = 0; c < candidateCount; c++) {
      if (selected.includes(c)) continue;
      const value = totalCost([...selected, c]);
      if (value < bestCost - 1e-12) {
        bestCost = value;
        bestCandidate = c;
      }
    }
    selected.push(bestCandidate);
  }
  let current = totalCost(selected);
  for (let pass = 0; pass < 10; pass++) {
    let improved = false;
    for (let slot = 0; slot < p; slot++)
      for (let c = 0; c < candidateCount; c++) {
        if (selected.includes(c)) continue;
        const trial = selected.slice();
        trial[slot] = c;
        const value = totalCost(trial);
        if (value < current - 1e-9) {
          selected[slot] = c;
          current = value;
          improved = true;
        }
      }
    if (!improved) break;
  }
  selected.sort((a, b) => a - b);
  const facilityTable = makeCatalogTable(
    'facilities',
    'Selected facilities',
    {
      candidate_index: 'int32',
      source_index: 'int32',
      source_id: 'string',
      demand_count: 'int32',
      demand_weight: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const assignmentTable = makeCatalogTable('assignments', 'Demand assignments', {
    demand_index: 'int32',
    source_index: 'int32',
    source_id: 'string',
    facility_index: 'int32',
    facility_id: 'string',
    distance: 'float64'
  });
  const assignedCount = new Map<number, {count: number; weight: number}>();
  for (let d = 0; d < demandCount; d++) {
    let best = selected[0];
    for (const c of selected)
      if (cost[d * candidateCount + c] < cost[d * candidateCount + best]) best = c;
    const entry = assignedCount.get(best) ?? {count: 0, weight: 0};
    entry.count++;
    entry.weight += demandSample.value[d];
    assignedCount.set(best, entry);
    appendCatalogRow(assignmentTable, {
      demand_index: d,
      source_index: demandSample.sourceIndex[d],
      source_id: getFeatureIdentity(demand.features[demandSample.sourceIndex[d]]),
      facility_index: candidateSample.sourceIndex[best],
      facility_id: getFeatureIdentity(candidates.features[candidateSample.sourceIndex[best]]),
      distance: Math.hypot(
        demandSample.x[d] - candidateSample.x[best],
        demandSample.y[d] - candidateSample.y[best]
      )
    });
  }
  for (const c of selected) {
    const entry = assignedCount.get(c) ?? {count: 0, weight: 0};
    appendCatalogRow(facilityTable, {
      candidate_index: c,
      source_index: candidateSample.sourceIndex[c],
      source_id: getFeatureIdentity(candidates.features[candidateSample.sourceIndex[c]]),
      demand_count: entry.count,
      demand_weight: entry.weight,
      geometry: makePointText(frame, candidateSample.x[c], candidateSample.y[c])
    });
  }
  addCatalogDiagnostic(
    diagnostics,
    'straight-line-distance',
    'info',
    'Distances are straight-line; the heuristic is greedy add followed by single-swap vertex substitution and may stop at a local optimum.'
  );
  return {
    tables: [facilityTable, assignmentTable],
    diagnostics,
    summary: {facilityCount: p, demandCount, objective: current}
  };
}

// ---------------------------------------------------------------------------------------------
// SKATER regions
// ---------------------------------------------------------------------------------------------

export type SkaterInput = {
  diagnostics: CatalogDiagnostic[];
  names: string[];
  set: ReturnType<typeof buildAnalysisSet>;
  /** Analysis-set row of every kept feature. */
  rows: number[];
  attributes: number[][];
};

/** Reads the attribute table and the neighbor set of SKATER; shared by the CPU and GPU engines. */
export function readSkaterInput(
  collection: CatalogFeatureCollection,
  parameters: {properties: string} & Omit<WeightsSettings, 'rowStandardize'>
): SkaterInput {
  const diagnostics: CatalogDiagnostic[] = [];
  const names = parameters.properties
    .split(',')
    .map(name => name.trim())
    .filter(Boolean);
  // At least one attribute is needed to measure homogeneity.
  if (!names.length) throw new Error('Properties required');
  const set = buildAnalysisSet(collection, '', {...parameters, rowStandardize: false});
  const rows: number[] = [];
  const attributes: number[][] = [];
  set.sourceIndex.forEach((sourceIndex, row) => {
    const values = names.map(name => readNumericProperty(collection.features[sourceIndex], name));
    if (values.every(value => value !== null)) {
      rows.push(row);
      attributes.push(values as number[]);
    }
  });
  addDroppedDiagnostic(
    diagnostics,
    set.droppedCount + (set.sourceIndex.length - rows.length),
    'Features missing a numeric attribute or position are skipped.'
  );
  return {diagnostics, names, set, rows, attributes};
}

/** Region labels computed elsewhere (the GPU engine) for the rows of {@link readSkaterInput}. */
export type PrecomputedSkaterRegions = {
  input: SkaterInput;
  /** Region label per kept row (any integers; relabeled to 0..k-1 in first-seen order). */
  labels: ArrayLike<number>;
};

export function runSkaterRegions(
  collection: CatalogFeatureCollection,
  parameters: {
    properties: string;
    regionCount: number;
    minimumRegionSize: number;
  } & Omit<WeightsSettings, 'rowStandardize'>,
  precomputed?: PrecomputedSkaterRegions
): CatalogRunOutput {
  const {diagnostics, names, set, rows, attributes} =
    precomputed?.input ?? readSkaterInput(collection, parameters);
  const n = rows.length;
  if (precomputed)
    return makeSkaterOutput(
      collection,
      set,
      rows,
      precomputed.labels,
      parameters.regionCount,
      diagnostics,
      true
    );
  // Regionalization needs at least as many features as regions.
  if (n < parameters.regionCount) throw new Error('Too few features');
  const means = names.map((_, column) => attributes.reduce((sum, row) => sum + row[column], 0) / n);
  const deviations = names.map((_, column) =>
    Math.sqrt(attributes.reduce((sum, row) => sum + (row[column] - means[column]) ** 2, 0) / n)
  );
  const z = attributes.map(row =>
    row.map((value, column) =>
      deviations[column] ? (value - means[column]) / deviations[column] : 0
    )
  );
  const position = new Map<number, number>();
  rows.forEach((row, index) => position.set(row, index));
  const edges: {a: number; b: number; weight: number}[] = [];
  rows.forEach((row, a) =>
    set.weights.neighbors[row].forEach(other => {
      const b = position.get(other);
      if (b === undefined || b <= a) return;
      edges.push({
        a,
        b,
        weight: z[a].reduce((sum, value, column) => sum + (value - z[b][column]) ** 2, 0)
      });
    })
  );
  edges.sort((x, y) => x.weight - y.weight || x.a - y.a || x.b - y.b);
  const parent = Int32Array.from({length: n}, (_, index) => index);
  const find = (index: number): number => {
    while (parent[index] !== index) {
      parent[index] = parent[parent[index]];
      index = parent[index];
    }
    return index;
  };
  const treeAdjacency: number[][] = Array.from({length: n}, () => []);
  for (const edge of edges) {
    const a = find(edge.a);
    const b = find(edge.b);
    if (a === b) continue;
    parent[a] = b;
    treeAdjacency[edge.a].push(edge.b);
    treeAdjacency[edge.b].push(edge.a);
  }
  const componentCount = new Set(Array.from({length: n}, (_, index) => find(index))).size;
  if (componentCount > 1)
    addCatalogDiagnostic(
      diagnostics,
      'disconnected-neighborhood',
      'warning',
      'The neighborhood graph is disconnected; every connected component starts as its own region.',
      componentCount
    );
  const region = new Int32Array(n);
  const rootOf = new Map<number, number>();
  for (let index = 0; index < n; index++) {
    const root = find(index);
    if (!rootOf.has(root)) rootOf.set(root, rootOf.size);
    region[index] = rootOf.get(root)!;
  }
  let regionTotal = rootOf.size;
  const removed = new Set<string>();
  const edgeKey = (a: number, b: number) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  const getMembers = (id: number) => {
    const members: number[] = [];
    for (let index = 0; index < n; index++) if (region[index] === id) members.push(index);
    return members;
  };
  const getSsd = (count: number, squares: number, sums: number[]) =>
    squares - sums.reduce((total, value) => total + value * value, 0) / count;
  while (regionTotal < parameters.regionCount) {
    let bestGain = -Infinity;
    let bestCut: {child: number; parentNode: number; region: number} | null = null;
    for (let id = 0; id < regionTotal; id++) {
      const members = getMembers(id);
      if (members.length < 2 * parameters.minimumRegionSize) continue;
      const root = members[0];
      const order: number[] = [];
      const parentOf = new Map<number, number>([[root, -1]]);
      const stack = [root];
      while (stack.length) {
        const node = stack.pop()!;
        order.push(node);
        for (const other of treeAdjacency[node])
          if (!removed.has(edgeKey(node, other)) && !parentOf.has(other)) {
            parentOf.set(other, node);
            stack.push(other);
          }
      }
      const count = new Map<number, number>();
      const sums = new Map<number, number[]>();
      const squares = new Map<number, number>();
      for (const node of order) {
        count.set(node, 1);
        sums.set(node, z[node].slice());
        squares.set(
          node,
          z[node].reduce((total, value) => total + value * value, 0)
        );
      }
      for (let index = order.length - 1; index > 0; index--) {
        const node = order[index];
        const up = parentOf.get(node)!;
        count.set(up, count.get(up)! + count.get(node)!);
        squares.set(up, squares.get(up)! + squares.get(node)!);
        const target = sums.get(up)!;
        sums.get(node)!.forEach((value, column) => (target[column] += value));
      }
      const totalCount = count.get(root)!;
      const totalSquares = squares.get(root)!;
      const totalSums = sums.get(root)!;
      const wholeSsd = getSsd(totalCount, totalSquares, totalSums);
      for (let index = 1; index < order.length; index++) {
        const node = order[index];
        const leftCount = count.get(node)!;
        const rightCount = totalCount - leftCount;
        if (leftCount < parameters.minimumRegionSize || rightCount < parameters.minimumRegionSize)
          continue;
        const leftSums = sums.get(node)!;
        const rightSums = totalSums.map((value, column) => value - leftSums[column]);
        const gain =
          wholeSsd -
          getSsd(leftCount, squares.get(node)!, leftSums) -
          getSsd(rightCount, totalSquares - squares.get(node)!, rightSums);
        if (gain > bestGain + 1e-12) {
          bestGain = gain;
          bestCut = {child: node, parentNode: parentOf.get(node)!, region: id};
        }
      }
    }
    if (!bestCut) break;
    removed.add(edgeKey(bestCut.child, bestCut.parentNode));
    // Relabel the child's side with the new region id.
    const stack = [bestCut.child];
    const seen = new Set<number>([bestCut.child]);
    while (stack.length) {
      const node = stack.pop()!;
      region[node] = regionTotal;
      for (const other of treeAdjacency[node])
        if (!removed.has(edgeKey(node, other)) && !seen.has(other)) {
          seen.add(other);
          stack.push(other);
        }
    }
    regionTotal++;
  }
  return makeSkaterOutput(
    collection,
    set,
    rows,
    region,
    parameters.regionCount,
    diagnostics,
    false
  );
}

function makeSkaterOutput(
  collection: CatalogFeatureCollection,
  set: SkaterInput['set'],
  rows: readonly number[],
  labels: ArrayLike<number>,
  requestedRegionCount: number,
  diagnostics: CatalogDiagnostic[],
  relabelFirstSeen: boolean
): CatalogRunOutput {
  const n = rows.length;
  const relabeled = new Map<number, number>();
  const region = Int32Array.from({length: n}, (_, index) => {
    if (!relabelFirstSeen) return labels[index];
    if (!relabeled.has(labels[index])) relabeled.set(labels[index], relabeled.size);
    return relabeled.get(labels[index])!;
  });
  const regionTotal = new Set(region).size;
  if (regionTotal < requestedRegionCount)
    addCatalogDiagnostic(
      diagnostics,
      'fewer-regions',
      'warning',
      'The minimum region size prevented reaching the requested number of regions.'
    );
  const table = makeCatalogTable(
    'regions',
    'SKATER regions',
    {source_index: 'int32', source_id: 'string', region: 'int32', geometry: 'string'},
    'geometry'
  );
  for (let index = 0; index < n; index++) {
    const sourceIndex = set.sourceIndex[rows[index]];
    appendCatalogRow(table, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(collection.features[sourceIndex]),
      region: region[index],
      geometry: getFeatureGeometryText(collection.features[sourceIndex])
    });
  }
  return {tables: [table], diagnostics, summary: {featureCount: n, regionCount: regionTotal}};
}

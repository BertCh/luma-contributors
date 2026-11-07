// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUKnoxTest,
  GPULocalOutlierFactor,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  getGPULocalOutlierFactorParameterValues,
  getGPUSpaceTimeParameterValues,
  getGPUSpaceTimeRelativeTimes,
  type GPUSpatialWeights
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import type {CatalogGpuRunner} from './catalog-gpu-bindings';
import {getPointPositions, type CatalogCoordinateSystem} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  getDistanceMatrix,
  getSortedNeighbors,
  readClusterSamples,
  runLocalOutlierFactor
} from './catalog-platform3-b-clusters';
import {
  collectAllPositions,
  readNumericProperty,
  type Inputs,
  type Values
} from './catalog-platform3-b-common';
import {makeLocalFrame} from './catalog-local-frame';
import {runKnoxTest} from './catalog-platform3-b-statistics';

function toCsr(lists: readonly (readonly number[])[]): {
  offsets: Uint32Array;
  neighbors: Uint32Array;
} {
  const offsets = new Uint32Array(lists.length + 1);
  const neighbors: number[] = [];
  lists.forEach((list, row) => {
    neighbors.push(...[...list].sort((a, b) => a - b));
    offsets[row + 1] = neighbors.length;
  });
  return {offsets, neighbors: Uint32Array.from(neighbors)};
}

/** k-nearest-neighbor table with distances, the input of GPULocalOutlierFactor; the CPU picks the neighbors (exact ties). */
async function runGpuLocalOutlierFactor(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const parameters = {
    neighborCount: values['neighborCount'] as number,
    threshold: values['threshold'] as number,
    coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
  };
  const read = readClusterSamples(inputs['points'], parameters.coordinateSystem);
  const n = read.sample.count;
  const k = parameters.neighborCount;
  // The CPU reference raises the declared error for too few points.
  if (n <= k) throw new CatalogGpuUnsupportedError('Too few features');
  const distances = getDistanceMatrix(read.sample);
  const rows = Array.from({length: n}, (_, row) =>
    getSortedNeighbors(distances, n, row)
      .slice(0, k)
      .sort((a, b) => a.index - b.index)
  );
  const csr = toCsr(rows.map(list => list.map(entry => entry.index)));
  const slotDistances = Float32Array.from(rows.flatMap(list => list.map(entry => entry.distance)));
  // Coincident points make the CPU reference report a factor of one; the contributor uses a density floor.
  if (rows.some(list => list.some(entry => entry.distance === 0)))
    throw new CatalogGpuUnsupportedError('Coincident points');
  const result = await withCatalogGpuSession(context.device, 'catalog-lof', async session => {
    const neighbors: GPUSpatialWeights = {
      offsets: session.input('offsets', csr.offsets, 'uint32'),
      neighbors: session.input('neighbors', csr.neighbors, 'uint32'),
      weights: session.input('weights', new Float32Array(csr.neighbors.length).fill(1), 'float32'),
      distances: session.input('distances', slotDistances, 'float32')
    };
    const kDistance = session.output('k-distance', 'float32', n);
    const density = session.output('density', 'float32', n);
    const lof = session.output('lof', 'float32', n);
    session.graph.add(
      new GPULocalOutlierFactor({
        id: 'catalog-lof',
        neighbors,
        parameters: session.input(
          'parameters',
          getGPULocalOutlierFactorParameterValues({threshold: parameters.threshold}),
          'float32'
        ),
        output: {
          kDistance: kDistance.view,
          localReachabilityDensity: density.view,
          lof: lof.view
        }
      })
    );
    session.run();
    return {kDistance: await session.readFloat32(kDistance), lof: await session.readFloat32(lof)};
  });
  return {
    ...runLocalOutlierFactor(inputs['points'], parameters, {read, ...result}),
    dispatchedContributors: ['GPULocalOutlierFactor']
  };
}

/** Knox: the CPU selects the spatially close pairs (exact double distances); the GPU permutes the times. */
async function runGpuKnoxTest(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const parameters = {
    timeProperty: values['timeProperty'] as string,
    spatialThreshold: values['spatialThreshold'] as number,
    temporalThreshold: values['temporalThreshold'] as number,
    permutations: values['permutations'] as number,
    seed: values['seed'] as number,
    coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
  };
  const points = inputs['points'];
  const frame = makeLocalFrame(parameters.coordinateSystem, collectAllPositions(points), {
    shift: 'minimum'
  });
  const x: number[] = [];
  const y: number[] = [];
  const times: number[] = [];
  for (const feature of points.features) {
    const position = getPointPositions(feature.geometry)[0];
    const time = readNumericProperty(feature, parameters.timeProperty);
    if (!position || time === null) continue;
    const [px, py] = frame.forward(position);
    x.push(px);
    y.push(py);
    times.push(time);
  }
  const n = times.length;
  // The CPU reference raises the declared error for too few events.
  if (n < 3 || parameters.permutations < 1) throw new CatalogGpuUnsupportedError('Too few events');
  const lists: number[][] = Array.from({length: n}, () => []);
  for (let first = 0; first < n; first++)
    for (let second = first + 1; second < n; second++)
      if (Math.hypot(x[first] - x[second], y[first] - y[second]) <= parameters.spatialThreshold)
        lists[first].push(second);
  const csr = toCsr(lists);
  if (!csr.neighbors.length) throw new CatalogGpuUnsupportedError('No spatial pairs');
  const relative = getGPUSpaceTimeRelativeTimes(times);
  const summary = await withCatalogGpuSession(context.device, 'catalog-knox', async session => {
    const pairs: GPUSpatialWeights = {
      offsets: session.input('offsets', csr.offsets, 'uint32'),
      neighbors: session.input('neighbors', csr.neighbors, 'uint32'),
      weights: session.input('weights', new Float32Array(csr.neighbors.length).fill(1), 'float32')
    };
    const statistics = session.output('statistics', 'uint32', parameters.permutations + 1);
    const output = session.output('summary', 'float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
    session.graph.add(
      new GPUKnoxTest({
        id: 'catalog-knox',
        pairs,
        times: session.input('times', relative.times, 'float32'),
        parameters: session.input(
          'parameters',
          getGPUSpaceTimeParameterValues({
            seed: parameters.seed,
            permutationCount: parameters.permutations,
            timeThreshold: parameters.temporalThreshold
          }),
          'uint32'
        ),
        maximumPermutations: parameters.permutations,
        output: {statistics: statistics.view, summary: output.view}
      })
    );
    session.run();
    return session.readFloat32(output);
  });
  const run = runKnoxTest(points, parameters, {
    pValue: summary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater]
  });
  // float32 relative times can flip a pair at the temporal threshold; the counts must agree.
  if (run.tables[0].columns['joint_close_pairs'][0] !== summary[GPU_SPACE_TIME_SUMMARY.observed])
    throw new CatalogGpuUnsupportedError('Joint pair count differs in float32');
  return {...run, dispatchedContributors: ['GPUKnoxTest']};
}

/** PLATFORM5 GPU runners for point and event statistics that were CPU-reference only, keyed by binding id. */
export const PLATFORM5_POINT_STATISTICS_GPU_RUNNERS: Readonly<Record<string, CatalogGpuRunner>> = {
  'local-outlier-factor': {
    contributorNames: ['GPULocalOutlierFactor'],
    run: runGpuLocalOutlierFactor
  },
  'knox-test': {contributorNames: ['GPUKnoxTest'], run: runGpuKnoxTest}
};

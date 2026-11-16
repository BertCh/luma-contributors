// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import {getGPUHierarchicalDensityClusteringParameterValues} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUHierarchicalDensityClustering} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Runs one `GPUHierarchicalDensityClustering` graph and reads labels, core distances and the cluster count. */
export async function runHierarchicalDensityClustering(
  device: Device,
  positions: Float32Array,
  options: {
    minimumClusterSize: number;
    minimumSamples?: number;
    coreDistanceSearch?: 'bvh' | 'neighbor-search';
    timedEncodings?: number;
    positionUpdates?: readonly Float32Array[];
    statusPoison?: number;
    times?: Float32Array;
    timeScale?: number;
    timeScaleUpdates?: readonly number[];
    temporalContext?: GPUTemporalContext;
  }
): Promise<{
  labels: number[];
  coreDistances: number[];
  clusterCount: number;
  requiredClusterCount: number;
  positionUpdated: {
    labels: number[];
    coreDistances: number[];
    clusterCount: number;
    requiredClusterCount: number;
  }[];
  medianEncodingMilliseconds?: number;
  updated: {labels: number[]; coreDistances: number[]; clusterCount: number}[];
}> {
  const rows = positions.length / 2;
  const graph = new GPUCommandGraph(device, {id: 'hierarchical-density-clustering-test'});
  const positionsBuffer = createInputBuffer(device, positions);
  const labelsBuffer = createOutputBuffer(device, rows);
  const coreBuffer = createOutputBuffer(device, rows);
  const countBuffer = createOutputBuffer(device, 1);
  const requiredCountBuffer = createOutputBuffer(device, 1);
  const timesBuffer = options.times ? createInputBuffer(device, options.times) : undefined;
  const parametersBuffer = options.times
    ? createInputBuffer(
        device,
        getGPUHierarchicalDensityClusteringParameterValues({timeScale: options.timeScale ?? 1})
      )
    : undefined;
  graph.add(
    new GPUHierarchicalDensityClustering({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
      ...(timesBuffer && parametersBuffer
        ? {
            times: importGraphBuffer(graph, 'times', timesBuffer, 'float32', rows),
            temporalContext: options.temporalContext ?? {
              unit: 'second',
              epoch: 0n,
              encoding: 'float32-relative',
              axis: {kind: 'irregular'}
            },
            parameters: importGraphBuffer(graph, 'parameters', parametersBuffer, 'float32', 4)
          }
        : {}),
      minimumClusterSize: options.minimumClusterSize,
      minimumSamples: options.minimumSamples,
      coreDistanceSearch: options.coreDistanceSearch,
      output: {
        labels: importGraphBuffer(graph, 'labels', labelsBuffer, 'uint32', rows),
        coreDistances: importGraphBuffer(graph, 'core-distances', coreBuffer, 'float32', rows),
        status: {
          count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          requiredCount: importGraphBuffer(
            graph,
            'required-count',
            requiredCountBuffer,
            'uint32',
            1
          )
        }
      }
    })
  );
  const compiled = await graph.compileAsync();
  const poisonStatus = () => {
    countBuffer.write(new Uint32Array([options.statusPoison ?? 123]));
    requiredCountBuffer.write(new Uint32Array([options.statusPoison ?? 123]));
  };
  poisonStatus();
  submitGraph(device, compiled, undefined);
  let medianEncodingMilliseconds: number | undefined;
  if (options.timedEncodings) {
    await readUint32(countBuffer, 1);
    const times: number[] = [];
    for (let encoding = 0; encoding < options.timedEncodings; encoding++) {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      await readUint32(countBuffer, 1);
      times.push(performance.now() - start);
    }
    times.sort((left, right) => left - right);
    medianEncodingMilliseconds = times[Math.floor(times.length / 2)];
  }
  const result = {
    labels: await readUint32(labelsBuffer, rows),
    coreDistances: await readFloat32(coreBuffer, rows),
    clusterCount: (await readUint32(countBuffer, 1))[0],
    requiredClusterCount: (await readUint32(requiredCountBuffer, 1))[0],
    medianEncodingMilliseconds
  };
  const updated: {labels: number[]; coreDistances: number[]; clusterCount: number}[] = [];
  for (const timeScale of options.timeScaleUpdates ?? []) {
    parametersBuffer!.write(getGPUHierarchicalDensityClusteringParameterValues({timeScale}));
    submitGraph(device, compiled, undefined);
    updated.push({
      labels: await readUint32(labelsBuffer, rows),
      coreDistances: await readFloat32(coreBuffer, rows),
      clusterCount: (await readUint32(countBuffer, 1))[0]
    });
  }
  const positionUpdated = [];
  for (const update of options.positionUpdates ?? []) {
    positionsBuffer.write(update);
    poisonStatus();
    submitGraph(device, compiled, undefined);
    positionUpdated.push({
      labels: await readUint32(labelsBuffer, rows),
      coreDistances: await readFloat32(coreBuffer, rows),
      clusterCount: (await readUint32(countBuffer, 1))[0],
      requiredClusterCount: (await readUint32(requiredCountBuffer, 1))[0]
    });
  }
  compiled.destroy();
  for (const buffer of [
    positionsBuffer,
    labelsBuffer,
    coreBuffer,
    countBuffer,
    requiredCountBuffer
  ]) {
    buffer.destroy();
  }
  timesBuffer?.destroy();
  parametersBuffer?.destroy();
  return {...result, updated, positionUpdated};
}

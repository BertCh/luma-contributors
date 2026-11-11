// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUVecchiaLogLikelihood,
  getVecchiaLogLikelihoodResult
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';
import {GPU_VECCHIA_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

export type VecchiaLikelihoodScene = {
  /** f64 or f32 coordinates; uploaded as float32. */
  positions: ArrayLike<number>;
  /** Observation values. */
  values: ArrayLike<number>;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  neighborCount: number;
  /** Observation error variance per row, zero by default. */
  observationError?: ArrayLike<number>;
  /** Observation validity (0 excludes) per row, one by default. */
  validity?: ArrayLike<number>;
};

/** One compiled likelihood graph over a fixed scene; `evaluate` writes parameters and reads back. */
export async function createVecchiaLikelihoodHarness(scene: VecchiaLikelihoodScene) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = scene.offsets.length - 1;
  const graph = new GPUCommandGraph(device);
  const positions = createInputBuffer(
    device,
    Float32Array.from(scene.positions as ArrayLike<number>)
  );
  const observationRows = new Float32Array(rowCount * 4);
  for (let row = 0; row < rowCount; row++) {
    observationRows[row * 4] = scene.values[row];
    observationRows[row * 4 + 1] = scene.observationError?.[row] ?? 0;
    observationRows[row * 4 + 3] = scene.validity ? scene.validity[row] : 1;
  }
  const observations = createInputBuffer(device, observationRows);
  const offsets = createInputBuffer(device, scene.offsets);
  // A CSR with no edges is still a valid view: pad so the buffer is not empty.
  const neighbors = createInputBuffer(
    device,
    scene.neighbors.length ? scene.neighbors : new Uint32Array(1)
  );
  const parameters = createInputBuffer(device, new Float32Array(GPU_VECCHIA_PARAMETER_LENGTH));
  const conditionals = createOutputBuffer(device, rowCount * 4);
  const conditionalStatus = createOutputBuffer(device, rowCount);
  const siteTerms = createOutputBuffer(device, rowCount);
  const siteStatus = createOutputBuffer(device, rowCount);
  const logLikelihood = createOutputBuffer(device, 2);
  const status = createOutputBuffer(device, 3);
  const contributor = new GPUVecchiaLogLikelihood({
    positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', rowCount),
    observations: importGraphBuffer(graph, 'observations', observations, 'float32x4', rowCount),
    neighborhoodOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', rowCount + 1),
    neighborhoods: importGraphBuffer(
      graph,
      'neighbors',
      neighbors,
      'uint32',
      Math.max(scene.neighbors.length, 1)
    ),
    neighborCount: scene.neighborCount,
    parameters: importGraphBuffer(
      graph,
      'parameters',
      parameters,
      'float32',
      GPU_VECCHIA_PARAMETER_LENGTH
    ),
    output: {
      conditionals: importGraphBuffer(graph, 'conditionals', conditionals, 'float32x4', rowCount),
      conditionalStatus: importGraphBuffer(
        graph,
        'conditional-status',
        conditionalStatus,
        'uint32',
        rowCount
      ),
      siteTerms: importGraphBuffer(graph, 'site-terms', siteTerms, 'float32', rowCount),
      siteStatus: importGraphBuffer(graph, 'site-status', siteStatus, 'uint32', rowCount),
      logLikelihood: importGraphBuffer(graph, 'log-likelihood', logLikelihood, 'float32', 2),
      status: importGraphBuffer(graph, 'status', status, 'uint32', 3)
    }
  });
  let buildCount = 0;
  const getCommandNodes = contributor.getCommandNodes.bind(contributor);
  contributor.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof contributor.getCommandNodes;
  graph.add(contributor);
  const compiled = graph.compile();
  const buffers = [
    positions,
    observations,
    offsets,
    neighbors,
    parameters,
    conditionals,
    conditionalStatus,
    siteTerms,
    siteStatus,
    logLikelihood,
    status
  ];
  return {
    get buildCount() {
      return buildCount;
    },
    device,
    /** Writes the 8-element parameter block; evaluates; reads back; returns value and elapsed ms. */
    async evaluate(values: Float32Array) {
      parameters.write(values);
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      const [total, words] = await Promise.all([
        readFloat32(logLikelihood, 2),
        readUint32(status, 3)
      ]);
      const elapsedMilliseconds = performance.now() - start;
      return {
        ...getVecchiaLogLikelihoodResult(total, words),
        high: total[0],
        low: total[1],
        elapsedMilliseconds
      };
    },
    async readConditionals() {
      return {
        conditionals: await readFloat32(conditionals, rowCount * 4),
        conditionalStatus: await readUint32(conditionalStatus, rowCount),
        siteTerms: await readFloat32(siteTerms, rowCount),
        siteStatus: await readUint32(siteStatus, rowCount)
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** One compiled Fisher scoring graph over a fixed scene. */
export async function createVecchiaFisherHarness(scene: VecchiaLikelihoodScene) {
  const {GPUVecchiaFisherScoring, getVecchiaFisherEvaluation, GPU_VECCHIA_FISHER_CELL_COUNT} =
    await import(
      '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring'
    );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = scene.offsets.length - 1;
  const graph = new GPUCommandGraph(device);
  const observationRows = new Float32Array(rowCount * 4);
  for (let row = 0; row < rowCount; row++) {
    observationRows[row * 4] = scene.values[row];
    observationRows[row * 4 + 1] = scene.observationError?.[row] ?? 0;
    observationRows[row * 4 + 3] = scene.validity ? scene.validity[row] : 1;
  }
  const positions = createInputBuffer(
    device,
    Float32Array.from(scene.positions as ArrayLike<number>)
  );
  const observations = createInputBuffer(device, observationRows);
  const offsets = createInputBuffer(device, scene.offsets);
  const neighbors = createInputBuffer(
    device,
    scene.neighbors.length ? scene.neighbors : new Uint32Array(1)
  );
  const parameters = createInputBuffer(device, new Float32Array(GPU_VECCHIA_PARAMETER_LENGTH));
  const siteTerms = createOutputBuffer(device, rowCount * GPU_VECCHIA_FISHER_CELL_COUNT);
  const siteStatus = createOutputBuffer(device, rowCount);
  const totals = createOutputBuffer(device, GPU_VECCHIA_FISHER_CELL_COUNT * 2);
  const status = createOutputBuffer(device, 3);
  const contributor = new GPUVecchiaFisherScoring({
    positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', rowCount),
    observations: importGraphBuffer(graph, 'observations', observations, 'float32x4', rowCount),
    neighborhoodOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', rowCount + 1),
    neighborhoods: importGraphBuffer(
      graph,
      'neighbors',
      neighbors,
      'uint32',
      Math.max(scene.neighbors.length, 1)
    ),
    neighborCount: scene.neighborCount,
    parameters: importGraphBuffer(
      graph,
      'parameters',
      parameters,
      'float32',
      GPU_VECCHIA_PARAMETER_LENGTH
    ),
    output: {
      siteTerms: importGraphBuffer(
        graph,
        'site-terms',
        siteTerms,
        'float32',
        rowCount * GPU_VECCHIA_FISHER_CELL_COUNT
      ),
      siteStatus: importGraphBuffer(graph, 'site-status', siteStatus, 'uint32', rowCount),
      totals: importGraphBuffer(
        graph,
        'totals',
        totals,
        'float32',
        GPU_VECCHIA_FISHER_CELL_COUNT * 2
      ),
      status: importGraphBuffer(graph, 'status', status, 'uint32', 3)
    }
  });
  let buildCount = 0;
  const getCommandNodes = contributor.getCommandNodes.bind(contributor);
  contributor.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof contributor.getCommandNodes;
  graph.add(contributor);
  const compiled = graph.compile();
  const buffers = [
    positions,
    observations,
    offsets,
    neighbors,
    parameters,
    siteTerms,
    siteStatus,
    totals,
    status
  ];
  return {
    get buildCount() {
      return buildCount;
    },
    device,
    async evaluate(values: Float32Array) {
      parameters.write(values);
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      const [totalValues, words] = await Promise.all([
        readFloat32(totals, GPU_VECCHIA_FISHER_CELL_COUNT * 2),
        readUint32(status, 3)
      ]);
      const elapsedMilliseconds = performance.now() - start;
      return {...getVecchiaFisherEvaluation(totalValues, words), elapsedMilliseconds};
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** One compiled space-time Vecchia likelihood graph over a fixed scene. */
export async function createVecchiaSpaceTimeHarness(
  scene: VecchiaLikelihoodScene & {times: ArrayLike<number>}
) {
  const {GPUVecchiaSpaceTimeLogLikelihood, GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH} = await import(
    '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = scene.offsets.length - 1;
  const graph = new GPUCommandGraph(device);
  const observationRows = new Float32Array(rowCount * 4);
  for (let row = 0; row < rowCount; row++) {
    observationRows[row * 4] = scene.values[row];
    observationRows[row * 4 + 1] = scene.observationError?.[row] ?? 0;
    observationRows[row * 4 + 3] = scene.validity ? scene.validity[row] : 1;
  }
  const positions = createInputBuffer(
    device,
    Float32Array.from(scene.positions as ArrayLike<number>)
  );
  const times = createInputBuffer(device, Float32Array.from(scene.times as ArrayLike<number>));
  const observations = createInputBuffer(device, observationRows);
  const offsets = createInputBuffer(device, scene.offsets);
  const neighbors = createInputBuffer(
    device,
    scene.neighbors.length ? scene.neighbors : new Uint32Array(1)
  );
  const parameters = createInputBuffer(
    device,
    new Float32Array(GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH)
  );
  const conditionals = createOutputBuffer(device, rowCount * 4);
  const conditionalStatus = createOutputBuffer(device, rowCount);
  const siteTerms = createOutputBuffer(device, rowCount);
  const siteStatus = createOutputBuffer(device, rowCount);
  const logLikelihood = createOutputBuffer(device, 2);
  const status = createOutputBuffer(device, 3);
  const contributor = new GPUVecchiaSpaceTimeLogLikelihood({
    temporalContext: {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'irregular'}
    },
    positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', rowCount),
    times: importGraphBuffer(graph, 'times', times, 'float32', rowCount),
    observations: importGraphBuffer(graph, 'observations', observations, 'float32x4', rowCount),
    neighborhoodOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', rowCount + 1),
    neighborhoods: importGraphBuffer(
      graph,
      'neighbors',
      neighbors,
      'uint32',
      Math.max(scene.neighbors.length, 1)
    ),
    neighborCount: scene.neighborCount,
    parameters: importGraphBuffer(
      graph,
      'parameters',
      parameters,
      'float32',
      GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH
    ),
    output: {
      conditionals: importGraphBuffer(graph, 'conditionals', conditionals, 'float32x4', rowCount),
      conditionalStatus: importGraphBuffer(
        graph,
        'conditional-status',
        conditionalStatus,
        'uint32',
        rowCount
      ),
      siteTerms: importGraphBuffer(graph, 'site-terms', siteTerms, 'float32', rowCount),
      siteStatus: importGraphBuffer(graph, 'site-status', siteStatus, 'uint32', rowCount),
      logLikelihood: importGraphBuffer(graph, 'log-likelihood', logLikelihood, 'float32', 2),
      status: importGraphBuffer(graph, 'status', status, 'uint32', 3)
    }
  });
  let buildCount = 0;
  const getCommandNodes = contributor.getCommandNodes.bind(contributor);
  contributor.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof contributor.getCommandNodes;
  graph.add(contributor);
  const compiled = graph.compile();
  const buffers = [
    positions,
    times,
    observations,
    offsets,
    neighbors,
    parameters,
    conditionals,
    conditionalStatus,
    siteTerms,
    siteStatus,
    logLikelihood,
    status
  ];
  return {
    get buildCount() {
      return buildCount;
    },
    async evaluate(values: Float32Array) {
      parameters.write(values);
      submitGraph(device, compiled, undefined);
      const [total, words] = await Promise.all([
        readFloat32(logLikelihood, 2),
        readUint32(status, 3)
      ]);
      return getVecchiaLogLikelihoodResult(total, words);
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

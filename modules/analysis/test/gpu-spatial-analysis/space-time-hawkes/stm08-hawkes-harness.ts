// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpaceTimeHawkesParameterValues,
  GPUSpaceTimeHawkes,
  type GPUSpaceTimeHawkesSettings
} from '../../../src/gpu-spatial-analysis/space-time-hawkes/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {HawkesEvent, HawkesParameters, HawkesWindow} from './stm08-hawkes-reference';

/** Everything `GPUSpaceTimeHawkes` publishes, read back. */
export type HawkesGPUResult = {
  estimates: HawkesParameters;
  logLikelihood: number;
  backgroundProbability: number[];
  converged: number;
  iterationLimitReached: number;
  iterationCount: number;
  candidateOverflow: number;
  invalidCount: number;
  approximation: number;
};

/** Options of {@link runHawkesGPU}. */
export type HawkesGPUOptions = {
  maximumIterations?: number;
  maximumParents?: number;
  tolerance?: number;
  /** Slots of the packed parameter view to overwrite, to reach states the packer refuses. */
  parameterOverride?: Readonly<Record<number, number>>;
};

/** Runs `GPUSpaceTimeHawkes` once on `events` with every output present. */
export async function runHawkesGPU(
  device: Device,
  events: readonly HawkesEvent[],
  window: HawkesWindow,
  initial: HawkesParameters,
  options: HawkesGPUOptions = {}
): Promise<HawkesGPUResult> {
  const graph = new GPUCommandGraph(device, {id: 'space-time-hawkes'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const settings: GPUSpaceTimeHawkesSettings = {
    windowArea: window.area,
    windowDuration: window.duration,
    maximumLag: window.maximumLag,
    maximumDistance: window.maximumDistance,
    tolerance: options.tolerance,
    initialBackground: initial.background,
    initialBranchingRatio: initial.branchingRatio,
    initialDecayRate: initial.decayRate,
    initialSpatialStandardDeviation: initial.spatialStandardDeviation
  };
  const rowCount = events.length;
  const positions = Float32Array.from(events.flatMap(event => [event.x, event.y]));
  const times = Float32Array.from(events.map(event => event.time));
  const outputs = {
    estimates: output(4),
    logLikelihood: output(1),
    backgroundProbability: output(rowCount),
    converged: output(1),
    iterationLimitReached: output(1),
    iterationCount: output(1),
    candidateOverflow: output(1),
    invalidCount: output(1),
    approximation: output(1)
  };
  graph.add(
    new GPUSpaceTimeHawkes({
      positions: importGraphBuffer(graph, 'positions', input(positions), 'float32x2'),
      times: importGraphBuffer(graph, 'times', input(times), 'float32'),
      parameters: importGraphBuffer(
        graph,
        'parameters',
        input(overrideParameters(getGPUSpaceTimeHawkesParameterValues(settings), options)),
        'float32'
      ),
      maximumIterations: options.maximumIterations,
      maximumParents: options.maximumParents ?? window.maximumParents,
      output: {
        estimates: importGraphBuffer(graph, 'o-estimates', outputs.estimates, 'float32', 4),
        logLikelihood: importGraphBuffer(
          graph,
          'o-log-likelihood',
          outputs.logLikelihood,
          'float32',
          1
        ),
        backgroundProbability: importGraphBuffer(
          graph,
          'o-background',
          outputs.backgroundProbability,
          'float32',
          rowCount
        ),
        status: {
          converged: importGraphBuffer(graph, 'o-converged', outputs.converged, 'uint32', 1),
          iterationLimitReached: importGraphBuffer(
            graph,
            'o-limit',
            outputs.iterationLimitReached,
            'uint32',
            1
          ),
          iterationCount: importGraphBuffer(
            graph,
            'o-iterations',
            outputs.iterationCount,
            'uint32',
            1
          ),
          candidateOverflow: importGraphBuffer(
            graph,
            'o-overflow',
            outputs.candidateOverflow,
            'uint32',
            1
          ),
          invalidCount: importGraphBuffer(graph, 'o-invalid', outputs.invalidCount, 'uint32', 1),
          approximation: importGraphBuffer(
            graph,
            'o-approximation',
            outputs.approximation,
            'uint32',
            1
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const [background, branchingRatio, decayRate, spatialStandardDeviation] = await readFloat32(
      outputs.estimates,
      4
    );
    const [logLikelihood] = await readFloat32(outputs.logLikelihood, 1);
    return {
      estimates: {background, branchingRatio, decayRate, spatialStandardDeviation},
      logLikelihood,
      backgroundProbability: await readFloat32(outputs.backgroundProbability, rowCount),
      converged: (await readUint32(outputs.converged, 1))[0],
      iterationLimitReached: (await readUint32(outputs.iterationLimitReached, 1))[0],
      iterationCount: (await readUint32(outputs.iterationCount, 1))[0],
      candidateOverflow: (await readUint32(outputs.candidateOverflow, 1))[0],
      invalidCount: (await readUint32(outputs.invalidCount, 1))[0],
      approximation: (await readUint32(outputs.approximation, 1))[0]
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

function overrideParameters(values: Float32Array, options: HawkesGPUOptions): Float32Array {
  for (const [slot, value] of Object.entries(options.parameterOverride ?? {})) {
    values[Number(slot)] = value;
  }
  return values;
}

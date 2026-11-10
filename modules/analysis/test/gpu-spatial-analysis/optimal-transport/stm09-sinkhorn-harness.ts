// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSinkhornDivergenceParameterValues,
  GPUSinkhornDivergence,
  type GPUSinkhornDivergenceSettings
} from '../../../src/gpu-spatial-analysis/optimal-transport/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {SinkhornGridProblem} from './stm09-sinkhorn-reference';

/** Everything `GPUSinkhornDivergence` publishes, read back. */
export type SinkhornGPUResult = {
  divergence: number;
  /** `[OT(source, target), OT(source, source), OT(target, target)]`. */
  components: number[];
  marginalError: number;
  converged: number;
  iterationLimitReached: number;
  iterationCount: number;
  invalidCount: number;
  approximation: number;
};

/** Options of {@link runSinkhornGPU}. */
export type SinkhornGPUOptions = Partial<
  Pick<GPUSinkhornDivergenceSettings, 'tolerance' | 'epsilonStart' | 'scalingFactor'>
> & {
  maximumIterations?: number;
  blurRadius?: number;
  /** Slots of the packed parameter view to overwrite, to reach states the packer refuses. */
  parameterOverride?: Readonly<Record<number, number>>;
};

/** Runs `GPUSinkhornDivergence` once with every output present. */
export async function runSinkhornGPU(
  device: Device,
  problem: Pick<
    SinkhornGridProblem,
    'width' | 'height' | 'cellWidth' | 'cellHeight' | 'source' | 'target' | 'epsilon'
  >,
  options: SinkhornGPUOptions = {}
): Promise<SinkhornGPUResult> {
  const graph = new GPUCommandGraph(device, {id: 'sinkhorn-divergence'});
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
  const parameters = getGPUSinkhornDivergenceParameterValues({
    cellWidth: problem.cellWidth,
    cellHeight: problem.cellHeight,
    epsilon: problem.epsilon,
    tolerance: options.tolerance,
    epsilonStart: options.epsilonStart,
    scalingFactor: options.scalingFactor
  });
  for (const [slot, value] of Object.entries(options.parameterOverride ?? {})) {
    parameters[Number(slot)] = value;
  }
  const outputs = {
    divergence: output(1),
    components: output(3),
    marginalError: output(1),
    converged: output(1),
    iterationLimitReached: output(1),
    iterationCount: output(1),
    invalidCount: output(1),
    approximation: output(1)
  };
  graph.add(
    new GPUSinkhornDivergence({
      grid: makeGPURasterMetadata({width: problem.width, height: problem.height}),
      source: importGraphBuffer(
        graph,
        'source',
        input(Float32Array.from(problem.source)),
        'float32'
      ),
      target: importGraphBuffer(
        graph,
        'target',
        input(Float32Array.from(problem.target)),
        'float32'
      ),
      parameters: importGraphBuffer(graph, 'parameters', input(parameters), 'float32'),
      maximumIterations: options.maximumIterations,
      blurRadius: options.blurRadius,
      output: {
        divergence: importGraphBuffer(graph, 'o-divergence', outputs.divergence, 'float32', 1),
        components: importGraphBuffer(graph, 'o-components', outputs.components, 'float32', 3),
        marginalError: importGraphBuffer(graph, 'o-error', outputs.marginalError, 'float32', 1),
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
    return {
      divergence: (await readFloat32(outputs.divergence, 1))[0],
      components: await readFloat32(outputs.components, 3),
      marginalError: (await readFloat32(outputs.marginalError, 1))[0],
      converged: (await readUint32(outputs.converged, 1))[0],
      iterationLimitReached: (await readUint32(outputs.iterationLimitReached, 1))[0],
      iterationCount: (await readUint32(outputs.iterationCount, 1))[0],
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

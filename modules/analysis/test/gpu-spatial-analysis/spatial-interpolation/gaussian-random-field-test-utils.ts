// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {
  GPUGaussianRandomField,
  getGPUGaussianRandomFieldParameterValues,
  type GPUGaussianRandomFieldSettings
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';

export type FieldScenario = {
  name: string;
  method: string;
  dimensions?: number[];
  steps?: number[];
  gridOrigin?: readonly [number, number];
  positions?: number[][];
  covariance: number[];
  modeCount?: number;
  seed: number;
  realization: number;
  mean: number;
  conditioning?: {indices: number[]; values: number[]; mask?: number[]; neighborhoods: number[][]};
  expected?: number[];
  clippedEigenvalueMass?: number;
};

export function createFieldRun(device: Device, scenario: FieldScenario, compile = true) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends GPUVectorFormat>(
    format: Format,
    values: Float32Array | Uint32Array,
    length?: number
  ) => {
    const buffer = createInputBuffer(device, values.length ? values : new Float32Array(2));
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `${scenario.name}-input-${buffers.length}`,
      buffer,
      format,
      length
    );
  };
  const output = <Format extends GPUVectorFormat>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `${scenario.name}-output-${buffers.length}`,
        buffer,
        format,
        length
      )
    };
  };
  const grid: GPURasterMetadata | undefined = scenario.dimensions
    ? {
        width: scenario.dimensions[0]!,
        height: scenario.dimensions[1]!,
        affine: [
          scenario.steps![0]!,
          0,
          scenario.gridOrigin?.[0] ?? 0,
          0,
          scenario.steps![1]!,
          scenario.gridOrigin?.[1] ?? 0
        ],
        pixelInterpretation: 'point'
      }
    : undefined;
  const modelNames = ['spherical', 'exponential', 'gaussian', 'matern-3/2', 'matern-5/2'] as const;
  const settings: GPUGaussianRandomFieldSettings = {
    variogram: {
      model: modelNames[scenario.covariance[0]!]!,
      nugget: scenario.covariance[1]!,
      sill: scenario.covariance[2]!,
      range: scenario.covariance[3]!
    },
    seed: scenario.seed,
    realization: scenario.realization,
    mean: scenario.mean,
    grid
  };
  const parameterValues = getGPUGaussianRandomFieldParameterValues(settings);
  const parameters = input('float32', parameterValues);
  const parameterBuffer = buffers[buffers.length - 1]!;
  const rowCount = grid ? grid.width * grid.height : scenario.positions!.length;
  const values = output('float32', rowCount + 2);
  const validity = output('uint32', rowCount + 2);
  const approximation = output('uint32', 1);
  const mass = output('float32', 1);
  const conditioning = scenario.conditioning;
  const contributor = new GPUGaussianRandomField({
    ...(grid
      ? {grid}
      : {positions: input('float32x2', new Float32Array(scenario.positions!.flat()), rowCount)}),
    method: scenario.method as 'fft' | 'spectral',
    modeCount: scenario.modeCount,
    parameters,
    conditioning: conditioning
      ? {
          indices: input('uint32', new Uint32Array(conditioning.indices)),
          values: input('float32', new Float32Array(conditioning.values)),
          mask: conditioning.mask ? input('uint32', new Uint32Array(conditioning.mask)) : undefined,
          neighborhoods: input('uint32', new Uint32Array(conditioning.neighborhoods.flat())),
          neighborCount: conditioning.neighborhoods[0]!.length
        }
      : undefined,
    output: {
      values: values.view,
      validity: validity.view,
      status: {approximation: approximation.view},
      clippedEigenvalueMass: mass.view
    }
  });
  graph.add(contributor);
  const compiled = compile ? graph.compile() : undefined;
  return {
    graph,
    compiled,
    contributor,
    settings,
    parameterBuffer,
    async execute(realization?: number) {
      if (realization !== undefined)
        parameterBuffer.write(getGPUGaussianRandomFieldParameterValues({...settings, realization}));
      if (!compiled) throw new Error('createFieldRun requires compile for execution');
      submitGraph(device, compiled);
      return readFloat32(values.buffer);
    },
    async status() {
      return {
        validity: await readUint32(validity.buffer),
        approximation: (await readUint32(approximation.buffer))[0]!,
        mass: (await readFloat32(mass.buffer))[0]!
      };
    },
    destroy() {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPUTerrainHeightAboveDrainage} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import type {GPUTerrainFlowDirectionEncoding} from '../../../src/gpu-terrain/hydrology/terrain-flow-tracing';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';

/** Execute production final publication with controlled pointers, independent of solver scheduling. */
export function executeHANDPublication(input: {
  directionEncoding: GPUTerrainFlowDirectionEncoding;
  directions: number[];
  pointer: number[];
  streams: number[];
  includeDrainage?: boolean;
}) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const count = input.directions.length;
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length = count
  ) => createTransientView(graph, name, format, length);
  try {
    const nodes = new GPUTerrainHeightAboveDrainage({
      grid: {width: count, height: 1, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
      maximumIterations: 2,
      directionEncoding: input.directionEncoding,
      elevation: {
        id: 'dem',
        format: 'float32',
        storage: {kind: 'buffer', values: view('elevation', 'float32')}
      },
      flowDirections: view('directions', 'uint32'),
      streams: view('streams', 'uint32'),
      output: {
        heightAboveDrainage: view('height', 'float32'),
        drainageCells: input.includeDrainage === false ? undefined : view('drainage', 'uint32'),
        status: {converged: view('done', 'uint32', 1)}
      }
    }).getCommandNodes(graph);
    const pipeline = getRasterPipelines(nodes.filter(node => node.id.endsWith('-height')))[0];
    if (!pipeline) throw new Error('HAND publication kernel missing');
    const memory: Record<string, Float32Array | Uint32Array> = {
      flowDirections: Uint32Array.from(input.directions),
      streams: Uint32Array.from(input.streams),
      pointer: Uint32Array.from(input.pointer),
      elevation: Float32Array.from(count === 1 ? [5] : [5, 3]),
      heightOut: new Float32Array(count).fill(99),
      converged: Uint32Array.of(1),
      drainageOut: new Uint32Array(count).fill(99)
    };
    const bindings = Object.fromEntries(
      new WgslReflect(pipeline.source).storage.map(binding => {
        const values = memory[binding.name];
        if (!values) throw new Error(`Unseeded production binding ${binding.name}`);
        return [binding.binding, values];
      })
    );
    new WgslExec(new WgslParser().parse(pipeline.source)).dispatchWorkgroups(
      'main',
      [1, 1, 1],
      {0: bindings},
      {constants: {INVOCATION_COUNT: count, WGSL_KERNEL_DISPATCH_X: 1, WGSL_KERNEL_DISPATCH_Y: 1}}
    );
    return {
      drainage: Array.from(memory['drainageOut']),
      heights: Array.from(memory['heightOut']),
      converged: memory['converged'][0]
    };
  } finally {
    device.destroy();
  }
}

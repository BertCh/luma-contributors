// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {PAIR_LAG_INDEX_WGSL} from '../../../src/dataframe-analysis/pair-statistics/pair-lag-index-wgsl';
import {GPUCrossVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {GPUSpaceTimeVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-variogram';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const legacy of [true, false])
  for (const [lagCount, temporalLagCount] of [
    [29, 1],
    [1, 29]
  ]) {
    it(`B1 generated ${legacy ? 'legacy' : 'positive-lag'} classes correct both axes (${lagCount},${temporalLagCount})`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        name: string,
        format: Format,
        length: number
      ) => createTransientView(graph, name, format, length);
      const bins = legacy ? lagCount * temporalLagCount : (lagCount + 1) * (temporalLagCount + 1);
      const props = {
        positions: view('positions', 'float32x2', 3),
        values: view('values', 'float32', 3),
        times: view('times', 'float32', 3),
        parameters: view('parameters', 'float32', 8),
        gridSize: [3, 2] as const,
        lagCount,
        temporalLagCount,
        output: {
          semivariances: view('semivariances', 'float32', bins),
          pairCounts: view('counts', 'uint32', bins),
          status: view('status', 'uint32', bins)
        }
      };
      try {
        const contributor = legacy
          ? new GPUCrossVariogram(props)
          : new GPUSpaceTimeVariogram(props);
        const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
        const spatialName = legacy ? 'spatialBin' : 'spatialClass';
        const temporalName = legacy ? 'temporalBin' : 'temporalClass';
        const source = sources.find(kernel =>
          kernel.source.includes(`let ${spatialName} =`)
        )!.source;
        const spatialExpression = new RegExp(`let ${spatialName} = ([^;]+);`).exec(source)![1];
        const temporalExpression = new RegExp(`let ${temporalName} = ([^;]+);`).exec(source)![1];
        // Exercise the full production block so future helper dependencies are retained too.
        expect(source).toContain(PAIR_LAG_INDEX_WGSL);
        const helpers = PAIR_LAG_INDEX_WGSL;
        const constants = [
          ...source.matchAll(/const (?:LAG_COUNT|TEMPORAL_LAG_COUNT): u32 = \d+u;/g)
        ]
          .map(match => match[0])
          .join('\n');
        for (const [value, expected] of [
          [30, 15],
          [29.999998092651367, 14],
          [30.000001907348633, 15],
          [58, 28]
        ]) {
          const result = new Uint32Array(2);
          // Decimal literals keep the interpreter's bitcasts on f32 words; an integer literal
          // can retain its integer representation despite a local f32 annotation in this runner.
          new WgslExec(
            new WgslParser().parse(`
${helpers}
${constants}
struct Lattice { maximumDistance: f32 }
@group(0) @binding(0) var<storage,read_write> result: array<u32>;
@compute @workgroup_size(1) fn main() {
  let lattice = Lattice(58.0); let maximumDistance = 58.0; let maximumTime = 58.0;
  let distance: f32 = ${value.toFixed(20)}; let pairDistance = distance; let time = distance; let timeLag = distance;
  let deltaX = distance; let deltaY = 0.0;
  result[0] = ${spatialExpression}; result[1] = ${temporalExpression};
}`)
          ).dispatchWorkgroups('main', [1, 1, 1], {0: {0: result}});
          // width=58/29=2. Exact distance 30 starts bin 15; its predecessor is in bin 14.
          // Positive-lag layout reserves class zero, so its positive class is one higher.
          const shift = legacy ? 0 : 1;
          expect(Array.from(result), `lag ${value}`).toEqual([
            (lagCount === 29 ? expected : 0) + shift,
            (temporalLagCount === 29 ? expected : 0) + shift
          ]);
        }
      } finally {
        device.destroy();
      }
    });
  }

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPUSpatialScanStatistic} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {getGPUSpatialScanStatisticParameterValues} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('B2 whole-study circles have zero observed and replicate maxima and no live cluster', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `score-review-${serial++}`, format, length);
    const kernels = getGPUCommandNodeKernelSources(
      new GPUSpatialScanStatistic({
        positions: view('float32x2', 3),
        cases: view('uint32', 3),
        baseline: view('float32', 3),
        parameters: view('uint32', 8),
        timeBuckets: 1,
        maximumWindowZones: 3,
        maximumPermutations: 1,
        maximumClusters: 1,
        output: {
          statistics: view('float32', 2),
          summary: view('uint32', 4),
          clusterIndices: view('uint32', 4),
          clusterStatistics: view('float32', 8)
        }
      }).getCommandNodes(graph)
    );
    const totals = new Uint32Array(17);
    totals[0] = 64;
    totals[1] = new Uint32Array(Float32Array.of(3.6000001430511475).buffer)[0];
    const centerBest = new Uint32Array(30);
    for (let center = 0; center < 3; center++) centerBest[24 + center * 2] = 3;
    const memory: Record<string, Float32Array | Uint32Array> = {
      totals,
      centerBest,
      baseline: Float32Array.of(1.6, 0.4, 1.6),
      counts: Uint32Array.of(64, 0, 0, 0, 64, 0),
      parameters: getGPUSpatialScanStatisticParameterValues({
        seed: 0,
        permutationCount: 1,
        maximumPopulationFraction: 1
      }),
      neighbors: Uint32Array.of(0, 1, 2, 0, 1, 2, 0, 1, 2),
      // WgslExec eagerly evaluates ||'s unused final-circle operand; one padding word
      // permits that read. WGSL itself short-circuits before accessing beyond the nine keys.
      distances: new Float32Array(10),
      statistics: new Uint32Array(2),
      alive: new Uint32Array(3),
      members: new Uint32Array(3),
      state: new Uint32Array(8),
      roundDispatch: new Uint32Array(9),
      clusterIndices: new Uint32Array(4),
      clusterStatistics: new Float32Array(8)
    };
    for (const kernel of kernels.filter(kernel =>
      /-evaluate$|-batch-evaluate-0$|-select-init$/.test(kernel.id)
    )) {
      const bindings: Record<number, Record<number, Float32Array | Uint32Array>> = {0: {}};
      for (const binding of new WgslReflect(kernel.source).storage) {
        expect(memory[binding.name], binding.name).toBeDefined();
        bindings[binding.group][binding.binding] = memory[binding.name];
      }
      for (let center = 0; center < 3; center++) {
        // Execute each real per-center body independently; replace only the cooperative max
        // publication by its equivalent single-center atomic update, with no arithmetic edits.
        let source = kernel.source
          .replace(/^\s*_ = [^;]+;/gm, '')
          .replace(/@workgroup_size\(\d+\)/, '@workgroup_size(1)')
          .replace(
            /let index = workgroupIndex \* \d+u \+ localInvocationIndex;/,
            `let index = ${center}u;`
          );
        if (!kernel.id.endsWith('-select-init'))
          source = source
            .replace(
              /let center = \(group - localReplicate \* EVALUATE_GROUPS\) \* EVALUATE_SIZE \+ localInvocationIndex;/,
              `let center = ${center}u;`
            )
            .replace(
              /  \/\/ Workgroup max[\s\S]*$/,
              'if (best > 0.0) { atomicMax(&statistics[statisticsOffset + statisticRow], bitcast<u32>(best)); }\n}'
            );
        try {
          new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, bindings, {
            constants: kernel.constants
          });
        } catch (error) {
          throw new Error(`${kernel.id} center ${center}`, {cause: error});
        }
      }
    }
    // A complete circle contains every exposure and all 64 cases under either allocation.
    // Its conditional null expectation is exactly 64, so LLR=64 ln(64/64)=0.
    expect(Array.from(new Float32Array(memory['statistics'].buffer))).toEqual([0, 0]);
    expect(Array.from(memory['alive'])).toEqual([0, 0, 0]);
  } finally {
    device.destroy();
  }
});

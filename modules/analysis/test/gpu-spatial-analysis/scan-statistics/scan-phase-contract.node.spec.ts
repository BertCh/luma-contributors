// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {
  GPUCommandGraph,
  GraphDataView,
  createTransientView,
  type GPUCommandNode,
  type GraphBufferUse
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialScanStatistic} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function getPhaseContract(
  nodes: readonly GPUCommandNode[],
  kernels: ReturnType<typeof getGPUCommandNodeKernelSources>
) {
  const describeBuffer = (bound: GraphBufferUse['buffer']) => {
    const buffer = bound instanceof GraphDataView ? bound.buffer : bound;
    return {
      buffer: buffer.id,
      allocationBytes: buffer.byteLength,
      ...(bound instanceof GraphDataView
        ? {
            format: bound.format,
            byteOffset: bound.byteOffset,
            byteStride: bound.byteStride,
            length: bound.length,
            rowByteLength: bound.rowByteLength
          }
        : {})
    };
  };
  return {
    nodes: nodes.map(node => {
      const gate = node.condition;
      if (gate) expect(gate).toMatchObject({source: 'gpu', mode: 'indirect'});
      return {
        id: node.id,
        resources: node.resources?.map(resource => {
          if (!('buffer' in resource)) throw Error('unexpected non-buffer scan resource');
          return {usage: resource.usage, ...describeBuffer(resource.buffer)};
        }),
        gate:
          gate && gate.source === 'gpu' && gate.mode === 'indirect'
            ? {
                source: gate.source,
                mode: gate.mode,
                byteOffset: gate.byteOffset ?? 0,
                ...describeBuffer(gate.buffer)
              }
            : null
      };
    }),
    kernels
  };
}

for (const model of ['poisson', 'bernoulli', 'space-time-permutation'] as const)
  for (const parallelDraws of [false, true])
    it(`preserves phase order, shader text and dispatch constants: ${model}, parallel=${parallelDraws} (PTS-145)`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `phase-${serial++}`, format, length);
      try {
        const props = {
          id: 'phase-scan',
          model,
          parallelDraws,
          positions: view('float32x2', 3),
          cases: view('uint32', 6),
          ...(model === 'poisson'
            ? {baseline: view('float32', 6)}
            : model === 'bernoulli'
              ? {controls: view('uint32', 6)}
              : {}),
          parameters: view('uint32', 8),
          timeBuckets: 2,
          maximumWindowZones: 1,
          maximumPermutations: 2,
          maximumClusters: 2,
          output: {
            clusterIndices: view('uint32', 8),
            clusterStatistics: view('float32', 16),
            statistics: view('float32', 3),
            summary: view('uint32', 4)
          }
        } as const;
        const scan = new GPUSpatialScanStatistic(props);
        const nodes = scan.getCommandNodes(graph);
        // Reviewed phase contract, refreshed for B1/B2's exact fraction comparison and scoring.
        // Pins resources/gates as well as node order, complete WGSL and dispatch constants.
        const kernels = getGPUCommandNodeKernelSources(nodes);
        const contract = getPhaseContract(nodes, kernels);
        const signature = (candidate: readonly GPUCommandNode[]) =>
          JSON.stringify(getPhaseContract(candidate, kernels));
        const gated = nodes.find(node => node.condition)!;
        expect(gated).toBeDefined();
        expect(
          signature(
            nodes.map(node => (node === gated ? {...node, condition: undefined} : node))
          ) !== signature(nodes)
        ).toBe(true);
        const gate = gated.condition!;
        if (gate.source !== 'gpu' || gate.mode !== 'indirect')
          throw Error('expected indirect gate');
        expect(
          signature(
            nodes.map(node => {
              if (node !== gated) return node;
              // Indirect dispatch gates belong to compute nodes. Narrow the discriminant
              // before rebuilding the node, preserving the public command-node union.
              if (node.type !== 'compute') throw Error('expected gated compute node');
              return {...node, condition: {...gate, byteOffset: (gate.byteOffset ?? 0) + 12}};
            })
          ) !== signature(nodes)
        ).toBe(true);
        let swapped = false;
        const misbound = nodes.map(node => {
          if (swapped) return node;
          const resources = [...(node.resources ?? [])];
          for (let first = 0; first < resources.length; first++) {
            const left = resources[first];
            if (!('buffer' in left) || !(left.buffer instanceof GraphDataView)) continue;
            for (let second = first + 1; second < resources.length; second++) {
              const right = resources[second];
              if (!('buffer' in right) || !(right.buffer instanceof GraphDataView)) continue;
              if (
                left.buffer.format === right.buffer.format &&
                left.buffer.length === right.buffer.length &&
                left.buffer.buffer !== right.buffer.buffer
              ) {
                resources[first] = {...left, buffer: right.buffer};
                resources[second] = {...right, buffer: left.buffer};
                swapped = true;
                return {...node, resources};
              }
            }
          }
          return node;
        });
        expect(swapped).toBe(true);
        expect(signature(misbound) !== signature(nodes)).toBe(true);
        const digest = createHash('sha256').update(JSON.stringify(contract)).digest('hex');
        const expected = {
          poisson: [
            'c99974635c4a66aef8d91da0c2424e29bd637f0d79af246afcf20b3bec4fd9a7',
            'c99974635c4a66aef8d91da0c2424e29bd637f0d79af246afcf20b3bec4fd9a7'
          ],
          bernoulli: [
            'fd4e1f6c23b50f7483acd3f9e2a12e89f7a28741f079f6ee7b2773c6c5393931',
            '88e3a775bf829d37c3cde464bfb2b7c2f3ec618f30e965d72f93d8b2c1d077d5'
          ],
          'space-time-permutation': [
            '210c9327c614368f8260a7ceda5868f53e1366316f8b9fc42a43ef820615b8df',
            '593410642bae953f1a18e38bbb20f3e0962a02238a02dfbc96cbaf584d474bb2'
          ]
        };
        expect(digest).toBe(expected[model][Number(parallelDraws)]);
      } finally {
        device.destroy();
      }
    });

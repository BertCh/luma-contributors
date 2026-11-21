// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURedcapSingleLinkage,
  getGPURedcapSingleLinkageParameterValues
} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createRegionalizationAuditInput} from '../spatial-regionalization/regionalization-b-fixtures';
import {GPUSpatialWeightsMinimumSpanningTree} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {GPUSkaterRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {GPUSkaterRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation';
import {
  computeSkaterRegionsOnCPU,
  CPUSkaterRegionsOperation
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {REDCAP_AUDIT_CASES, createRedcapSSDAuditInput} from './redcap-audit-b-fixtures';

useCPUExecutorTestHooks();

for (const testCase of REDCAP_AUDIT_CASES) {
  it(`audits all single-linkage outputs against SciPy: k=${testCase.target}, tie=${testCase.tied}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      const input = createRegionalizationAuditInput();
      // An invalid island distinguishes invalidCount from adjacency loss while preserving the forest.
      input.values = testCase.tied ? new Float32Array(9).fill(64) : input.values;
      input.values[8] = NaN;
      let serial = 0;
      const upload = <Format extends 'uint32' | 'float32'>(
        values: Uint32Array | Float32Array,
        format: Format
      ) => {
        const view = createTransientView(graph, `input-${serial++}`, format, values.length);
        memory.write(view, values);
        return view;
      };
      const output = (name: string, length = 1) =>
        createTransientView(graph, name, 'uint32', length);
      const labels = output('labels', 9);
      const regionCount = output('count');
      const converged = output('converged');
      const iterationLimitReached = output('limit');
      const iterationCount = output('iterations');
      const invalidCount = output('invalid');
      const nodes = new GPURedcapSingleLinkage({
        weights: {
          offsets: upload(input.offsets, 'uint32'),
          neighbors: upload(input.neighbors, 'uint32'),
          weights: upload(new Float32Array(20).fill(1), 'float32')
        },
        values: upload(input.values, 'float32'),
        parameters: upload(
          getGPURedcapSingleLinkageParameterValues({regionCount: testCase.target}),
          'uint32'
        ),
        componentIterations: 8,
        standardize: false,
        output: {
          labels,
          regionCount,
          status: {converged, iterationLimitReached, iterationCount, invalidCount}
        }
      }).getCommandNodes(graph);
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(labels))).toEqual(testCase.labels);
      expect(Array.from(memory.readUint32(regionCount))).toEqual([testCase.count]);
      expect(Array.from(memory.readUint32(converged))).toEqual([1]);
      expect(Array.from(memory.readUint32(iterationLimitReached))).toEqual([0]);
      expect(Array.from(memory.readUint32(iterationCount))).toEqual([8]);
      expect(Array.from(memory.readUint32(invalidCount))).toEqual([1]);
    } finally {
      device.destroy();
    }
  });
}

it('separates the heaviest-edge cut from the exhaustive SSD cut in every existing SKATER form', async () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const input = createRedcapSSDAuditInput();
    const upload = <Format extends 'uint32' | 'float32'>(
      name: string,
      data: Uint32Array | Float32Array,
      format: Format
    ) => {
      const view = createTransientView(graph, name, format, data.length);
      memory.write(view, data);
      return view;
    };
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const weights = {
      offsets: upload('offsets', input.offsets, 'uint32'),
      neighbors: upload('neighbors', input.neighbors, 'uint32'),
      weights: upload('weights', input.weights, 'float32')
    };
    const values = upload('values', input.values, 'float32');
    const treeEdgeFlags = createTransientView(graph, 'tree-flags', 'uint32', 10);
    const componentLabels = createTransientView(graph, 'tree-labels', 'uint32', 8);
    const treeStatus = createTransientView(graph, 'tree-status', 'uint32', 2);
    const parameters = upload('cut-parameters', Uint32Array.of(4, 1), 'uint32');
    const createCutOutput = (prefix: string) => ({
      labels: createTransientView(graph, `${prefix}-labels`, 'uint32', 8),
      regionCount: scalar(`${prefix}-count`),
      cutEdges: createTransientView(graph, `${prefix}-edges`, 'uint32', 3),
      cutGains: createTransientView(graph, `${prefix}-gains`, 'float32', 3)
    });
    const direct = createCutOutput('direct');
    const lowered = createCutOutput('lowered');
    const operationProps = {
      id: 'ssd-operation',
      rowCount: 8,
      edgeCount: 10,
      maximumRegionCount: 4,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      values: 'values',
      treeEdgeFlags: 'treeEdgeFlags',
      componentLabels: 'componentLabels',
      treeStatus: 'treeStatus',
      parameters: 'parameters',
      output: {
        labels: 'labels',
        regionCount: 'regionCount',
        cutEdges: 'cutEdges',
        cutGains: 'cutGains'
      }
    };
    const views: Record<string, GraphDataView> = {
      ...weights,
      values,
      treeEdgeFlags,
      componentLabels,
      treeStatus,
      parameters,
      ...lowered
    };
    const operation = new GPUSkaterRegionsOperation(operationProps);
    const singleLabels = createTransientView(graph, 'single-labels', 'uint32', 8);
    const singleCount = scalar('single-count');
    const converged = scalar('converged');
    const iterationCount = scalar('iterations');
    const iterationLimitReached = scalar('iteration-limit');
    const invalidCount = scalar('invalid');
    const singleParameters = upload('single-parameters', Uint32Array.of(4), 'uint32');
    const nodes = [
      ...new GPUSpatialWeightsMinimumSpanningTree({
        weights,
        values,
        standardize: false,
        output: {treeEdgeFlags, componentLabels, status: treeStatus}
      }).getCommandNodes(graph),
      ...new GPUSkaterRegions({
        id: 'ssd-direct',
        weights,
        values,
        treeEdgeFlags,
        componentLabels,
        treeStatus,
        parameters,
        maximumRegionCount: 4,
        output: direct
      }).getCommandNodes(graph),
      ...operation
        .createContributor(
          <Format extends GPUGraphVectorFormat>(port: string) =>
            views[port] as GraphDataView<Format>
        )
        .getCommandNodes(graph),
      ...new GPURedcapSingleLinkage({
        weights,
        values,
        standardize: false,
        componentIterations: 8,
        parameters: singleParameters,
        output: {
          labels: singleLabels,
          regionCount: singleCount,
          status: {converged, iterationCount, iterationLimitReached, invalidCount}
        }
      }).getCommandNodes(graph)
    ];
    // NumPy exhaustively evaluates all five path cuts. Slot 4 minimizes SSD to
    // 5.853333263397228; slot 8 (heaviest edge) instead leaves SSD 13.259999456405648.
    // The total SSD reduction at slot 4 is 19.439999828338618. Islands add zero SSD.
    const expectCut = (actual: {
      labels: Uint32Array;
      regionCount: Uint32Array;
      cutEdges: Uint32Array;
      cutGains: Float32Array;
    }) => {
      expect(Array.from(actual.labels)).toEqual([0, 0, 0, 3, 3, 3, 6, 7]);
      expect(Array.from(actual.regionCount)).toEqual([4]);
      expect(Array.from(actual.cutEdges)).toEqual([4, 0xffffffff, 0xffffffff]);
      expect(Number.isFinite(actual.cutGains[0])).toBe(true);
      // Direct/subtracted f32 side means allow 2e-5 absolute gain error on this six-row path.
      expect(Math.abs(actual.cutGains[0] - 19.439999828338618)).toBeLessThan(2e-5);
      expect(Array.from(actual.cutGains.slice(1))).toEqual([0, 0]);
    };
    for (const target of [4, 1]) {
      memory.write(singleParameters, Uint32Array.of(target));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(treeEdgeFlags))).toEqual([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
      expect(Array.from(memory.readUint32(componentLabels))).toEqual([0, 0, 0, 0, 0, 0, 6, 7]);
      expect(Array.from(memory.readUint32(treeStatus))).toEqual([0, 0]);
      for (const output of [direct, lowered])
        expectCut({
          labels: memory.readUint32(output.labels),
          regionCount: memory.readUint32(output.regionCount),
          cutEdges: memory.readUint32(output.cutEdges),
          cutGains: memory.readFloat32(output.cutGains)
        });
      expect(Array.from(memory.readUint32(singleLabels))).toEqual(
        target === 4 ? [0, 0, 0, 0, 0, 5, 6, 7] : [0, 0, 0, 0, 0, 0, 6, 7]
      );
      expect(Array.from(memory.readUint32(singleCount))).toEqual([target === 4 ? 4 : 3]);
      expect(Array.from(memory.readUint32(converged))).toEqual([1]);
      expect(Array.from(memory.readUint32(iterationCount))).toEqual([8]);
      expect(Array.from(memory.readUint32(iterationLimitReached))).toEqual([0]);
      expect(Array.from(memory.readUint32(invalidCount))).toEqual([0]);
    }
    const cpuInput = {
      ...input,
      treeEdgeFlags: memory.readUint32(treeEdgeFlags),
      componentLabels: memory.readUint32(componentLabels),
      treeStatus: Uint32Array.of(0, 0),
      parameters: Uint32Array.of(4, 1)
    };
    expectCut(computeSkaterRegionsOnCPU(cpuInput, {maximumRegionCount: 4}));
    const cpuOperation = new CPUSkaterRegionsOperation(operationProps);
    const result = await cpuOperation.execute({
      inputs: Object.fromEntries(
        Object.entries(cpuInput).map(([name, data]) => [
          name,
          {
            format: data instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
            length: data.length,
            chunks: [{values: data, length: data.length}],
            pack: () => data
          }
        ])
      )
    });
    expectCut({
      labels: result['labels'] as Uint32Array,
      regionCount: result['regionCount'] as Uint32Array,
      cutEdges: result['cutEdges'] as Uint32Array,
      cutGains: result['cutGains'] as Float32Array
    });
  } finally {
    device.destroy();
  }
});

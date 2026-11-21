// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView, GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURedcapSingleLinkage,
  getGPURedcapSingleLinkageParameterValues
} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  AUDIT_OFFSETS,
  AUDIT_NEIGHBORS,
  createRegionalizationAuditInput
} from '../spatial-regionalization/regionalization-b-fixtures';
import {GPUSpatialWeightsMinimumSpanningTree} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {GPUSkaterRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {GPUSkaterRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation';
import {REDCAP_AUDIT_CASES, createRedcapSSDAuditInput} from './redcap-audit-b-fixtures';

for (const testCase of REDCAP_AUDIT_CASES) {
  it(`audits GPU single linkage against SciPy: k=${testCase.target}, tie=${testCase.tied}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const labels = rig.output('uint32', 9);
      const regionCount = rig.output('uint32', 1);
      const converged = rig.output('uint32', 1);
      const iterationLimitReached = rig.output('uint32', 1);
      const iterationCount = rig.output('uint32', 1);
      const invalidCount = rig.output('uint32', 1);
      // Shift and scale the non-tied fixture to exercise the shared standardizer. The island
      // stays invalid, so invalidCount=1; finite-column statistics exclude it. One-column
      // standardization preserves the edge order, hence these same SciPy component literals.
      const values = testCase.tied
        ? new Float32Array(9).fill(64)
        : createRegionalizationAuditInput(2 ** 64, 64).values;
      values[8] = NaN;
      rig.graph.add(
        new GPURedcapSingleLinkage({
          weights: rig.uploadWeights({
            offsets: AUDIT_OFFSETS,
            neighbors: AUDIT_NEIGHBORS,
            weights: new Array(20).fill(1),
            distances: []
          }),
          values: rig.input(values, 'float32', 9),
          parameters: rig.input(
            getGPURedcapSingleLinkageParameterValues({regionCount: testCase.target}),
            'uint32',
            1
          ),
          componentIterations: 8,
          output: {
            labels: labels.view,
            regionCount: regionCount.view,
            status: {
              converged: converged.view,
              iterationLimitReached: iterationLimitReached.view,
              iterationCount: iterationCount.view,
              invalidCount: invalidCount.view
            }
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(labels.buffer, 9)).toEqual(testCase.labels);
        expect(await readUint32(regionCount.buffer, 1)).toEqual([testCase.count]);
        expect(await readUint32(converged.buffer, 1)).toEqual([1]);
        expect(await readUint32(iterationLimitReached.buffer, 1)).toEqual([0]);
        expect(await readUint32(iterationCount.buffer, 1)).toEqual([8]);
        expect(await readUint32(invalidCount.buffer, 1)).toEqual([1]);
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  });
}

it('publishes the NumPy SSD cut and distinct heaviest-edge cut on the same forest', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const input = createRedcapSSDAuditInput();
    const weights = rig.uploadWeights({
      offsets: Array.from(input.offsets),
      neighbors: Array.from(input.neighbors),
      weights: Array.from(input.weights),
      distances: []
    });
    const values = rig.input(input.values, 'float32', 8);
    const treeEdgeFlags = rig.output('uint32', 10);
    const componentLabels = rig.output('uint32', 8);
    const treeStatus = rig.output('uint32', 2);
    const parameters = rig.input(Uint32Array.of(4, 1), 'uint32', 2);
    const createCutOutput = () => ({
      labels: rig.output('uint32', 8),
      regionCount: rig.output('uint32', 1),
      cutEdges: rig.output('uint32', 3),
      cutGains: rig.output('float32', 3)
    });
    const direct = createCutOutput();
    const lowered = createCutOutput();
    rig.graph.add(
      new GPUSpatialWeightsMinimumSpanningTree({
        weights,
        values,
        standardize: false,
        output: {
          treeEdgeFlags: treeEdgeFlags.view,
          componentLabels: componentLabels.view,
          status: treeStatus.view
        }
      })
    );
    rig.graph.add(
      new GPUSkaterRegions({
        id: 'ssd-direct',
        weights,
        values,
        treeEdgeFlags: treeEdgeFlags.view,
        componentLabels: componentLabels.view,
        treeStatus: treeStatus.view,
        parameters,
        maximumRegionCount: 4,
        output: {
          labels: direct.labels.view,
          regionCount: direct.regionCount.view,
          cutEdges: direct.cutEdges.view,
          cutGains: direct.cutGains.view
        }
      })
    );
    const views: Record<string, GraphDataView> = {
      offsets: weights.offsets,
      neighbors: weights.neighbors,
      weights: weights.weights,
      values,
      parameters,
      treeEdgeFlags: treeEdgeFlags.view,
      componentLabels: componentLabels.view,
      treeStatus: treeStatus.view,
      labels: lowered.labels.view,
      regionCount: lowered.regionCount.view,
      cutEdges: lowered.cutEdges.view,
      cutGains: lowered.cutGains.view
    };
    rig.graph.add(
      new GPUSkaterRegionsOperation({
        id: 'ssd-operation',
        rowCount: 8,
        edgeCount: 10,
        maximumRegionCount: 4,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        values: 'values',
        parameters: 'parameters',
        treeEdgeFlags: 'treeEdgeFlags',
        componentLabels: 'componentLabels',
        treeStatus: 'treeStatus',
        output: {
          labels: 'labels',
          regionCount: 'regionCount',
          cutEdges: 'cutEdges',
          cutGains: 'cutGains'
        }
      }).createContributor(
        <Format extends GPUGraphVectorFormat>(port: string) => views[port] as GraphDataView<Format>
      )
    );
    const singleLabels = rig.output('uint32', 8);
    const singleCount = rig.output('uint32', 1);
    const converged = rig.output('uint32', 1);
    const iterationCount = rig.output('uint32', 1);
    const iterationLimitReached = rig.output('uint32', 1);
    const invalidCount = rig.output('uint32', 1);
    const singleParameters = rig.input(Uint32Array.of(4), 'uint32', 1);
    rig.graph.add(
      new GPURedcapSingleLinkage({
        weights,
        values,
        parameters: singleParameters,
        standardize: false,
        componentIterations: 8,
        output: {
          labels: singleLabels.view,
          regionCount: singleCount.view,
          status: {
            converged: converged.view,
            iterationCount: iterationCount.view,
            iterationLimitReached: iterationLimitReached.view,
            invalidCount: invalidCount.view
          }
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const target of [4, 1]) {
        rig.bufferOf(singleParameters).write(Uint32Array.of(target));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(treeEdgeFlags.buffer, 10)).toEqual([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
        expect(await readUint32(componentLabels.buffer, 8)).toEqual([0, 0, 0, 0, 0, 0, 6, 7]);
        expect(await readUint32(treeStatus.buffer, 2)).toEqual([0, 0]);
        for (const output of [direct, lowered]) {
          expect(await readUint32(output.labels.buffer, 8)).toEqual([0, 0, 0, 3, 3, 3, 6, 7]);
          expect(await readUint32(output.regionCount.buffer, 1)).toEqual([4]);
          expect(await readUint32(output.cutEdges.buffer, 3)).toEqual([4, 0xffffffff, 0xffffffff]);
          const gains = await readFloat32(output.cutGains.buffer, 3);
          expect(Number.isFinite(gains[0])).toBe(true);
          // Exhaustive NumPy centered SSD over float32 inputs; allow 2e-5 for f32 side sums.
          expect(Math.abs(gains[0] - 19.439999828338618)).toBeLessThan(2e-5);
          expect(gains.slice(1)).toEqual([0, 0]);
        }
        expect(await readUint32(singleLabels.buffer, 8)).toEqual(
          target === 4 ? [0, 0, 0, 0, 0, 5, 6, 7] : [0, 0, 0, 0, 0, 0, 6, 7]
        );
        expect(await readUint32(singleCount.buffer, 1)).toEqual([target === 4 ? 4 : 3]);
        expect(await readUint32(converged.buffer, 1)).toEqual([1]);
        expect(await readUint32(iterationCount.buffer, 1)).toEqual([8]);
        expect(await readUint32(iterationLimitReached.buffer, 1)).toEqual([0]);
        expect(await readUint32(invalidCount.buffer, 1)).toEqual([0]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUSkaterRegionsParameterValues,
  GPU_SKATER_NO_CUT,
  GPU_SKATER_REGIONS_PARAMETER_LENGTH,
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {requireTestDevice} from '../../utils/require-test-device';

/** 3 x 3 rook lattice in CSR form. */
function createLattice() {
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < 9; row++) {
    const x = row % 3;
    for (const candidate of [row - 3, x > 0 ? row - 1 : -1, x < 2 ? row + 1 : -1, row + 3]) {
      if (candidate >= 0 && candidate < 9) neighbors.push(candidate);
    }
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors};
}

async function runPipeline(values: number[], connectStatus: boolean) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lattice = createLattice();
  const rig = new WeightsRig(device);
  const weights = rig.uploadWeights(
    {
      offsets: lattice.offsets,
      neighbors: lattice.neighbors,
      weights: lattice.neighbors.map(() => 1),
      distances: []
    },
    0
  );
  const flags = rig.output('uint32', lattice.neighbors.length);
  const components = rig.output('uint32', 9);
  const standardized = rig.output('float32', 9);
  const status = rig.output('uint32', 1);
  rig.graph.add(
    new GPUSpatialWeightsMinimumSpanningTree({
      weights,
      values: rig.input(new Float32Array(values), 'float32', 9),
      output: {
        treeEdgeFlags: flags.view,
        componentLabels: components.view,
        standardizedValues: standardized.view,
        status: status.view
      }
    })
  );
  const labels = rig.output('uint32', 9);
  const regionCount = rig.output('uint32', 1);
  const cutEdges = rig.output('uint32', 3);
  rig.graph.add(
    new GPUSkaterRegions({
      weights,
      treeEdgeFlags: flags.view,
      componentLabels: components.view,
      values: standardized.view,
      maximumRegionCount: 4,
      treeStatus: connectStatus ? status.view : undefined,
      parameters: rig.input(
        getGPUSkaterRegionsParameterValues({regionCount: 3}),
        'uint32',
        GPU_SKATER_REGIONS_PARAMETER_LENGTH
      ),
      output: {labels: labels.view, regionCount: regionCount.view, cutEdges: cutEdges.view}
    })
  );
  const compiled = rig.graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    regionCount: (await readUint32(regionCount.buffer, 1))[0],
    cutEdges: Array.from(await readUint32(cutEdges.buffer, 3)),
    invalidCount: (await readUint32(status.buffer, 1))[0]
  };
  compiled.destroy();
  rig.destroy();
  return result;
}

const FINITE = [0.3, 2.9, 1.1, 4.7, 0.5, 3.35, 7.2, 5.5, 0.9];

it('makes no cut when the tree reports non-finite attributes', async () => {
  const withNaN = [...FINITE];
  withNaN[4] = NaN;
  const result = await runPipeline(withNaN, true);
  expect(result.invalidCount).toBe(1);
  expect(result.regionCount).toBe(1);
  expect(result.cutEdges).toEqual([GPU_SKATER_NO_CUT, GPU_SKATER_NO_CUT, GPU_SKATER_NO_CUT]);
});

it('still cuts finite attributes down to the target region count', async () => {
  const result = await runPipeline(FINITE, true);
  expect(result.invalidCount).toBe(0);
  expect(result.regionCount).toBe(3);
  expect(result.cutEdges.filter(edge => edge !== GPU_SKATER_NO_CUT)).toHaveLength(2);
});

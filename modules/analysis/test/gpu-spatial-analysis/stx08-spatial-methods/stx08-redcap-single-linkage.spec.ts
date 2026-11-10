// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  getGPURedcapSingleLinkageParameterValues,
  GPURedcapSingleLinkage,
  GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/redcap-regions/index';
import fixture from '../../oracles/fixtures/stx08-spatial-methods.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

async function runRedcap(options: {
  offsets: number[];
  neighbors: number[];
  values: number[];
  columnCount: number;
  regionCount: number;
  componentIterations?: number;
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const rows = options.offsets.length - 1;
  const labels = rig.output('uint32', rows);
  const regionCount = rig.output('uint32', 1);
  const converged = rig.output('uint32', 1);
  const status = rig.output('uint32', 1);
  const iterationLimitReached = rig.output('uint32', 1);
  const contributor = new GPURedcapSingleLinkage({
    weights: rig.weights({
      offsets: options.offsets,
      neighbors: options.neighbors,
      weights: options.neighbors.map(() => 1)
    }),
    values: rig.input(new Float32Array(options.values), 'float32'),
    columnCount: options.columnCount,
    componentIterations: options.componentIterations,
    parameters: rig.input(
      getGPURedcapSingleLinkageParameterValues({regionCount: options.regionCount}),
      'uint32',
      GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH
    ),
    output: {labels, regionCount, status: {converged, invalidCount: status, iterationLimitReached}}
  });
  rig.run(contributor);
  const result = {
    labels: await rig.readUint(labels),
    regionCount: (await rig.readUint(regionCount))[0],
    converged: (await rig.readUint(converged))[0],
    invalidCount: (await rig.readUint(status))[0],
    iterationLimitReached: (await rig.readUint(iterationLimitReached))[0],
    estimated: contributor.estimatedCommandNodeCount
  };
  rig.destroy();
  return result;
}

describe('GPURedcapSingleLinkage', () => {
  for (const reference of fixture.redcap) {
    it(`equals the SciPy constrained single-linkage cut on ${reference.name}`, async () => {
      for (const outcome of reference.outcomes) {
        const result = await runRedcap({
          offsets: reference.offsets,
          neighbors: reference.neighbors,
          values: reference.values,
          columnCount: reference.columnCount,
          regionCount: outcome.regionCount
        });
        expect(result.labels, `regionCount ${outcome.regionCount}`).toEqual(outcome.labels);
        expect(result.regionCount).toBe(outcome.resultingRegionCount);
        expect(result.converged).toBe(1);
        expect(result.invalidCount).toBe(0);
      }
    });
  }

  it('refines the regions of k - 1 when it asks for k', async () => {
    const reference = fixture.redcap[0];
    const previous = reference.outcomes[3].labels;
    const next = reference.outcomes[4].labels;
    // Every region of the finer partition lies inside one coarser region.
    const parent = new Map<number, number>();
    next.forEach((label, row) => {
      expect(parent.get(label) ?? previous[row]).toBe(previous[row]);
      parent.set(label, previous[row]);
    });
  });

  it('labels a long path within the default iterations and reports non-convergence when starved', async () => {
    const rows = 300;
    const offsets = [0];
    const neighbors: number[] = [];
    for (let row = 0; row < rows; row++) {
      if (row > 0) neighbors.push(row - 1);
      if (row < rows - 1) neighbors.push(row + 1);
      offsets.push(neighbors.length);
    }
    // A staircase of distinct jumps: cutting the 2 heaviest edges gives 3 regions.
    const values = Array.from(
      {length: rows},
      (_, row) => row * 0.001 + (row >= 100 ? 5 : 0) + (row >= 200 ? 9 : 0)
    );
    const full = await runRedcap({offsets, neighbors, values, columnCount: 1, regionCount: 3});
    expect(full.converged).toBe(1);
    expect(full.regionCount).toBe(3);
    expect(full.labels.slice(0, 100).every(label => label === 0)).toBe(true);
    expect(full.labels.slice(100, 200).every(label => label === 100)).toBe(true);
    expect(full.labels.slice(200).every(label => label === 200)).toBe(true);
    const starved = await runRedcap({
      offsets,
      neighbors,
      values,
      columnCount: 1,
      regionCount: 1,
      componentIterations: 1
    });
    expect(starved.converged).toBe(0);
    expect(starved.iterationLimitReached).toBe(1);
    expect(full.iterationLimitReached).toBe(0);
  });

  it('keeps the command node count under its published estimate', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const reference = fixture.redcap[0];
    const contributor = new GPURedcapSingleLinkage({
      weights: rig.weights({
        offsets: reference.offsets,
        neighbors: reference.neighbors,
        weights: reference.neighbors.map(() => 1)
      }),
      values: rig.input(new Float32Array(reference.values), 'float32'),
      columnCount: reference.columnCount,
      parameters: rig.input(getGPURedcapSingleLinkageParameterValues({regionCount: 4}), 'uint32'),
      output: {labels: rig.output('uint32', reference.offsets.length - 1)}
    });
    const nodeCount = contributor.getCommandNodes(rig.graph).length;
    rig.destroy();
    expect(nodeCount).toBeGreaterThan(0);
    expect(nodeCount).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  });
});

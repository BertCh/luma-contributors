// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUArealInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation';
import {GPUPycnophylacticInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation/gpu-pycnophylactic';
import {getZoneAnchorNodes} from '../../../src/gpu-spatial-analysis/areal-interpolation/zone-anchor-nodes';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {createPaddedQueryFixture} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('retains disjoint neighbors and overflow packed within one alignment window', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createPaddedQueryFixture(device);
  const {graph, input, output} = fixture;
  const backing = output('shared', 'uint32', 5);
  const neighbors = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: backing.byteOffset,
    length: 4
  });
  const overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: backing.byteOffset + 16,
    length: 1
  });
  const offsets = output('offsets', 'uint32', 2);
  const weights = output('weights', 'float32', 4);
  graph.add(
    new GPUArealInterpolation({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      sourceZones: input('source', 'uint32', new Uint32Array([0, 0, 1, 1])),
      targetZones: input('destination', 'uint32', new Uint32Array(4)),
      sourceZoneCount: 2,
      targetZoneCount: 1,
      output: {weights: {offsets, neighbors, weights}, status: {overflow}}
    })
  );
  const compiled = graph.compile();
  try {
    const actual = await fixture.submitTwice(compiled, async () => {
      const shared = await fixture.readRows([backing]);
      return {
        offsets: await fixture.readRows([offsets]),
        neighbors: shared.slice(0, 4),
        weights: await fixture.readRows([weights]),
        overflow: shared.slice(4)
      };
    });
    expect(actual.offsets).toEqual([0, 2]);
    expect(actual.neighbors.slice(0, 2)).toEqual([0, 1]);
    expect(actual.weights.slice(0, 2)).toEqual([1, 1]);
    expect(actual.overflow).toEqual([0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('assigns competing anchors by ascending zone and leaves one cell per donor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createPaddedQueryFixture(device);
  const {graph, input} = fixture;
  const zones = input('zones', 'uint32', new Uint32Array([0, 0, 0, 1, 1, 0xffffffff]));
  const anchors = input('anchors', 'uint32', new Uint32Array([0, 3, 0, 0, 1, 2, 3, 4, 5, 5]));
  const anchored = getZoneAnchorNodes(graph, {
    id: 'anchors',
    operation: 'GPUArealInterpolation',
    zones,
    zoneCount: 10,
    anchorCells: anchors
  });
  graph.add(anchored.nodes);
  const readback = fixture.output('anchored-readback', 'uint32', 6);
  graph.addCopyPass({
    id: 'anchored-readback-copy',
    resources: [
      {buffer: anchored.zones, usage: 'copy-source'},
      {buffer: readback, usage: 'copy-destination'}
    ],
    compile: () => ({
      encode: ({commandEncoder, getBuffer}) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: getBuffer(anchored.zones),
          sourceOffset: anchored.zones.byteOffset,
          destinationBuffer: getBuffer(readback),
          destinationOffset: readback.byteOffset,
          size: 24
        })
    })
  });
  const compiled = graph.compile();
  try {
    const actual = await fixture.submitTwice(compiled, () => fixture.readRows([readback]));
    // Zone 2 wins cell 0; zone 4 takes cell 1; zone 5 cannot empty donor 0.
    // Zone 6 wins donor 1's quota; zone 8 wins the previously unassigned cell.
    expect(actual).toEqual([2, 4, 0, 6, 1, 8]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('preserves packed mass and status bytes written in separate pycnophylactic passes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createPaddedQueryFixture(device);
  const {graph, input, output} = fixture;
  const backing = output('mass-and-status', 'uint32', 5);
  const mass = graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 4,
    byteOffset: backing.byteOffset
  });
  const invalidCount = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: backing.byteOffset + 16
  });
  graph.add(
    new GPUPycnophylacticInterpolation({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      zones: input('zones', 'uint32', new Uint32Array(4)),
      totals: input('totals', 'float32', new Float32Array([8])),
      zoneCount: 1,
      maximumIterations: 1,
      output: {mass, status: {invalidCount}}
    })
  );
  const compiled = graph.compile();
  try {
    const actual = await fixture.submitTwice(compiled, () => fixture.readRows([backing]));
    expect(Array.from(new Float32Array(Uint32Array.from(actual.slice(0, 4)).buffer))).toEqual([
      2, 2, 2, 2
    ]);
    expect(actual[4]).toBe(0);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

for (const capacity of [1, 4]) {
  it(`reduces categorical areas with invalid categories, empty targets and capacity ${capacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createPaddedQueryFixture(device);
    const {graph, input, output} = fixture;
    const backing = output('neighbors-and-categories', 'uint32', capacity + 4);
    const neighbors = graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: capacity,
      byteOffset: backing.byteOffset
    });
    const categoryShares = graph.createDataView(backing.buffer, {
      format: 'float32',
      length: 4,
      byteOffset: backing.byteOffset + capacity * 4
    });
    const overflow = output('overflow', 'uint32', 1);
    graph.add(
      new GPUArealInterpolation({
        grid: makeGPURasterMetadata({width: 2, height: 2}),
        sourceZones: input('source', 'uint32', new Uint32Array([0, 0, 1, 1])),
        targetZones: input('destination', 'uint32', new Uint32Array(4)),
        cellWeights: input('cell-weights', 'float32', new Float32Array([1, 5, 2, 2])),
        sourceZoneCount: 2,
        targetZoneCount: 2,
        categories: {
          sourceCategories: input('source-categories', 'uint32', new Uint32Array([0, 0xffffffff])),
          categoryCount: 2
        },
        output: {
          weights: {
            offsets: output('offsets', 'uint32', 3),
            neighbors,
            weights: output('weights', 'float32', capacity)
          },
          categoryShares,
          status: {overflow}
        }
      })
    );
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, async () => ({
        packed: await fixture.readRows([backing]),
        overflow: await fixture.readRows([overflow])
      }));
      const shares = new Float32Array(Uint32Array.from(actual.packed.slice(capacity)).buffer);
      // Source 0 contributes 6 of 10 units. Invalid-category source 1 remains in the denominator.
      // Truncation retains only source 0; target 1 has no covered area.
      expect(shares[0]).toBeCloseTo(capacity === 1 ? 1 : 0.6, 6);
      expect(Array.from(shares.slice(1))).toEqual([0, 0, 0]);
      expect(actual.overflow).toEqual([capacity === 1 ? 1 : 0]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

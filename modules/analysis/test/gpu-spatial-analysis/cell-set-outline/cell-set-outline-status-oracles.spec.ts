// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUCellSetOutline,
  type GPUCellSetOutlineProps
} from '../../../src/gpu-spatial-analysis/cell-set-outline/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// One resolution-2 Quadbin square (tile 1,1; Morton prefix 0011) has four
// exposed edges, one closed ring, no open edge and no shared end vertex.
// Capacity 3 keeps three of the four edges; capacity 8 keeps all four.
// The ring outputs are exercised only with the complete four-edge outline.
for (const capacity of [3, 8]) {
  it(`outline and nested ring status overwrite poisoned words (capacity ${capacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const count = input('input-count', 'uint32', Uint32Array.of(1));
    const status = {
      count: output('count', 'uint32', 1),
      requiredCount: output('required', 'uint32', 1),
      overflow: output('overflow', 'uint32', 1)
    };
    const ringStatus = {
      count: output('ring-count', 'uint32', 1),
      overflow: output('ring-overflow', 'uint32', 1),
      openSegmentCount: output('open', 'uint32', 1),
      touchingSegmentCount: output('touching', 'uint32', 1)
    };
    graph.add(
      new GPUCellSetOutline({
        family: 'quadbin',
        cells: input('input-cells', 'uint32x2', Uint32Array.of(0xffffffff, 0x4823ffff)),
        count,
        output: {
          rows: output('rows', 'uint32', capacity),
          cells: output('cells', 'uint32x2', capacity),
          edgeIndices: output('edges', 'uint32', capacity),
          endpoints: output('endpoints', 'float32x4', capacity),
          status
        },
        rings:
          capacity === 8
            ? {
                output: {
                  ringOffsets: output('ring-offsets', 'uint32', 3),
                  positions: output('ring-positions', 'float32x2', 5),
                  status: ringStatus
                }
              }
            : undefined
      })
    );
    const compiled = graph.compile();
    try {
      for (const selected of [1, 0, 1]) {
        fixture.writeInput(count, Uint32Array.of(selected));
        const actual = await fixture.submitTwice(compiled, async () => ({
          outline: await fixture.readRows([status.count, status.requiredCount, status.overflow]),
          rings:
            capacity === 8
              ? await fixture.readRows([
                  ringStatus.count,
                  ringStatus.overflow,
                  ringStatus.openSegmentCount,
                  ringStatus.touchingSegmentCount
                ])
              : []
        }));
        expect(actual.outline).toEqual(
          selected === 0 ? [0, 0, 0] : capacity === 3 ? [3, 4, 1] : [4, 4, 0]
        );
        if (capacity === 8) expect(actual.rings).toEqual(selected ? [1, 0, 0, 0] : [0, 0, 0, 0]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

it('nested rings suppress an entire closed prefix after outline truncation and recover', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const count = input('input-count', 'uint32', Uint32Array.of(1));
  const status = {
    count: output('count', 'uint32', 1),
    requiredCount: output('required', 'uint32', 1),
    overflow: output('overflow', 'uint32', 1)
  };
  const ringOffsets = output('ring-offsets', 'uint32', 3);
  const ringStatus = {
    count: output('ring-count', 'uint32', 1),
    overflow: output('ring-overflow', 'uint32', 1),
    openSegmentCount: output('open', 'uint32', 1),
    touchingSegmentCount: output('touching', 'uint32', 1)
  };
  // Sorted resolution-2 tiles (1,1), (3,1): Morton prefixes 0011, 0111,
  // separated by tile (2,1). Eight exposed edges form two rings. Capacity four
  // retains exactly the first closed square; see independent Shapely oracle.
  const props: GPUCellSetOutlineProps = {
    family: 'quadbin',
    cells: input(
      'cells',
      'uint32x2',
      Uint32Array.of(0xffffffff, 0x4823ffff, 0xffffffff, 0x4827ffff)
    ),
    count,
    output: {
      rows: output('rows', 'uint32', 4),
      cells: output('out-cells', 'uint32x2', 4),
      edgeIndices: output('edges', 'uint32', 4),
      endpoints: output('endpoints', 'float32x4', 4),
      status
    },
    rings: {
      output: {
        ringOffsets,
        positions: output('ring-positions', 'float32x2', 10),
        status: ringStatus
      }
    }
  };
  const outline = new GPUCellSetOutline(props);
  // Replace caller maps before graph declaration. Every producer/consumer must
  // still target A above, not these equally sized B destinations. The Node
  // binding regression additionally mutates fields/status records in place.
  props.output = {
    rows: output('replacement-rows', 'uint32', 4),
    cells: output('replacement-cells', 'uint32x2', 4),
    edgeIndices: output('replacement-edges', 'uint32', 4),
    endpoints: output('replacement-endpoints', 'float32x4', 4),
    status: {
      count: output('replacement-count', 'uint32', 1),
      requiredCount: output('replacement-required', 'uint32', 1),
      overflow: output('replacement-overflow', 'uint32', 1)
    }
  };
  props.rings = {
    output: {
      ringOffsets: output('replacement-ring-offsets', 'uint32', 3),
      positions: output('replacement-ring-positions', 'float32x2', 10),
      status: {
        count: output('replacement-ring-count', 'uint32', 1),
        overflow: output('replacement-ring-overflow', 'uint32', 1),
        openSegmentCount: output('replacement-open', 'uint32', 1),
        touchingSegmentCount: output('replacement-touching', 'uint32', 1)
      }
    }
  };
  graph.add(outline);
  const compiled = graph.compile();
  try {
    for (const selected of [1, 2, 1]) {
      fixture.writeInput(count, Uint32Array.of(selected));
      const result = await fixture.submitTwice(compiled, async () => ({
        outline: await fixture.readRows([status.count, status.requiredCount, status.overflow]),
        rings: await fixture.readRows([
          ringStatus.count,
          ringStatus.overflow,
          ringStatus.openSegmentCount,
          ringStatus.touchingSegmentCount
        ]),
        offsets: await fixture.read(ringOffsets)
      }));
      expect(result.outline).toEqual(selected === 2 ? [4, 8, 1] : [4, 4, 0]);
      expect(result.rings).toEqual(selected === 2 ? [0, 0, 0, 0] : [1, 0, 0, 0]);
      expect(result.offsets).toEqual(selected === 2 ? [0, 0, 0] : [0, 5, 5]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

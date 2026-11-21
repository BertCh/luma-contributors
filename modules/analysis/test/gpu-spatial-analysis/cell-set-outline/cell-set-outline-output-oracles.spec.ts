// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const optional of [
  undefined,
  'ringGroups',
  'segmentRings',
  'segmentVertices',
  'segmentFlags',
  'requiredCount',
  'polygons',
  'sourceIds'
] as const) {
  it(`assembles the hand-enumerated isolated tile with only ${optional ?? 'required ring outputs'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const ringOffsets = output('ring-offsets', 'uint32', 2),
      positions = output('ring-positions', 'float32x2', 5);
    const count = output('count', 'uint32', 1),
      overflow = output('overflow', 'uint32', 1);
    const ringCount = output('ring-count', 'uint32', 1),
      ringOverflow = output('ring-overflow', 'uint32', 1);
    const selected =
      optional && optional !== 'polygons' && optional !== 'sourceIds'
        ? output(
            'selected',
            'uint32',
            ['segmentRings', 'segmentVertices', 'segmentFlags'].includes(optional) ? 4 : 1
          )
        : undefined;
    const polygons =
      optional === 'polygons' || optional === 'sourceIds'
        ? {
            kind: 'polygons' as const,
            featureOffsets: output('feature-offsets', 'uint32', 2),
            polygonOffsets: output('polygon-offsets', 'uint32', 2),
            ringOffsets: output('polygon-ring-offsets', 'uint32', 2),
            positions: output('polygon-positions', 'float32x2', 5),
            sourceIds: optional === 'sourceIds' ? output('source-ids', 'uint32', 1) : undefined
          }
        : undefined;
    graph.add(
      new GPUCellSetOutline({
        family: 'quadbin',
        // Independent Quadbin layout: zoom 2, Morton(1,1)=3, 48 footer bits.
        cells: input('cell', 'uint32x2', new Uint32Array([0xffffffff, 0x4823ffff])),
        groups: input('groups', 'uint32', new Uint32Array([37])),
        output: {
          rows: output('rows', 'uint32', 4),
          cells: output('cells', 'uint32x2', 4),
          edgeIndices: output('edges', 'uint32', 4),
          endpoints: output('endpoints', 'float32x4', 4),
          groups: output('out-groups', 'uint32', 4),
          status: {count, overflow}
        },
        rings: {
          output: {
            ringOffsets,
            positions,
            polygons,
            ...(optional &&
            ['ringGroups', 'segmentRings', 'segmentVertices', 'segmentFlags'].includes(optional)
              ? {[optional]: selected}
              : {}),
            status: {
              count: ringCount,
              overflow: ringOverflow,
              requiredCount: optional === 'requiredCount' ? selected : undefined
            }
          }
        }
      })
    );
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, async () => ({
        positions: await fixture.readRows([positions]),
        ringOffsets: await fixture.readRows([ringOffsets]),
        count: await fixture.readRows([count]),
        ringCount: await fixture.readRows([ringCount]),
        overflow: await fixture.readRows([ringOverflow]),
        selected: selected ? await fixture.readRows([selected]) : []
      }));
      expect(actual.ringOffsets).toEqual([0, 5]);
      expect(actual.count).toEqual([4]);
      expect(actual.ringCount).toEqual([1]);
      expect(actual.overflow).toEqual([0]);
      // Tile vertices NW,NE,SE,SW, closure. Latitude is atan(sinh(pi/2)) in degrees.
      const expected = [
        -90, 66.51326044311186, 0, 66.51326044311186, 0, 0, -90, 0, -90, 66.51326044311186
      ];
      actual.positions.forEach((value, index) => {
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value - expected[index])).toBeLessThan(2e-5);
      });
      expect(actual.selected).toEqual(
        optional === 'ringGroups'
          ? [37]
          : optional === 'segmentVertices'
            ? [0, 1, 2, 3]
            : optional === 'segmentRings' || optional === 'segmentFlags'
              ? [0, 0, 0, 0]
              : optional === 'requiredCount'
                ? [1]
                : []
      );
      if (polygons) {
        expect(await fixture.readRows([polygons.featureOffsets])).toEqual([0, 1]);
        expect(await fixture.readRows([polygons.polygonOffsets])).toEqual([0, 1]);
        expect(await fixture.readRows([polygons.ringOffsets])).toEqual([0, 4]);
        const coordinates = await fixture.readRows([polygons.positions]);
        coordinates.slice(0, 8).forEach((value, index) => {
          expect(Number.isFinite(value)).toBe(true);
          expect(Math.abs(value - expected[index])).toBeLessThan(2e-5);
        });
        if (polygons.sourceIds) expect(await fixture.readRows([polygons.sourceIds])).toEqual([37]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

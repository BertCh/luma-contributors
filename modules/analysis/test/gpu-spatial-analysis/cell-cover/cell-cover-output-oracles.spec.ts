// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const includeCore of [true, false]) {
  it(`writes the literal inset-rectangle core mask with core ${includeCore}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const cells = output('cells', 'uint32x2', 4);
    const featureIds = output('ids', 'uint32', 4);
    const core = includeCore ? output('core', 'uint32', 4) : undefined;
    const count = output('count', 'uint32', 1);
    const overflow = output('overflow', 'uint32', 1);
    const candidateOverflow = output('candidate-overflow', 'uint32', 1);
    graph.add(
      new GPUCellCover({
        family: 'quadbin',
        resolution: 2,
        containment: 'center',
        candidateCapacity: 32,
        candidateEnumeration: 'bounds',
        edgeSlabs: false,
        positions: input(
          'positions',
          'float32x2',
          new Float32Array([-100, -70, 60, -70, 60, 70, -100, 70])
        ),
        featureOffsets: input('features', 'uint32', new Uint32Array([0, 1])),
        polygonOffsets: input('polygons', 'uint32', new Uint32Array([0, 1])),
        ringOffsets: input('rings', 'uint32', new Uint32Array([0, 4])),
        output: {cells, featureIds, core, status: {count, overflow, candidateOverflow}}
      })
    );
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, async () => ({
        cells: await fixture.readRows([cells]),
        core: core ? await fixture.readRows([core]) : [],
        ids: await fixture.readRows([featureIds]),
        count: await fixture.readRows([count]),
        overflow: await fixture.readRows([overflow]),
        candidateOverflow: await fixture.readRows([candidateOverflow])
      }));
      // At zoom 2 the accepted tile centres are (x,y)=(1,1),(2,1),(1,2),(2,2).
      // Their boundaries are longitude -90/0/90 and latitude +-66.51326/0.
      // Only x=1 is wholly inside [-100,60]x[-70,70], including the safety margin.
      // Quadbin encodes Morton 3,6,9,12 above 48 footer bits, with header 0x482.
      const rows = Array.from({length: 4}, (_, row) => [
        actual.cells[row * 2],
        actual.cells[row * 2 + 1],
        ...(includeCore ? [actual.core[row]] : [])
      ]).sort((left, right) => left[1] - right[1]);
      expect(rows).toEqual([
        [0xffffffff, 0x4823ffff, ...(includeCore ? [1] : [])],
        [0xffffffff, 0x4826ffff, ...(includeCore ? [0] : [])],
        [0xffffffff, 0x4829ffff, ...(includeCore ? [1] : [])],
        [0xffffffff, 0x482cffff, ...(includeCore ? [0] : [])]
      ]);
      expect(actual.ids).toEqual([0, 0, 0, 0]);
      expect(actual.count).toEqual([4]);
      expect(actual.overflow).toEqual([0]);
      expect(actual.candidateOverflow).toEqual([0]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

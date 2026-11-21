// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCellMeasuresOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures-operation';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {H3_MEASURES_FIXTURES} from './h3-measures-fixtures';

for (const optional of ['perimeters', 'edgeLengths', 'pentagons', undefined] as const) {
  for (const includeAreas of optional ? [false, true] : [true]) {
    it(`writes Python H3 measures through operation lowering with ${optional ?? 'no extra port'}, areas ${includeAreas}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      // Python h3 4.5.0 fixture provenance is recorded in h3-measures-fixtures.ts.
      // A low-resolution hexagon, a distorted pentagon and a seven-edge face crossing
      // discriminate edge order, stride padding, perimeter truncation and pentagon flags.
      const references = ['8097fffffffffff', '81083ffffffffff', '81af7ffffffffff'].map(
        key => H3_MEASURES_FIXTURES.find(row => row[0] === key)!
      );
      const keys = references.map(row => BigInt(`0x${row[0]}`));
      const cells = fixture.input(
        'cells',
        'uint32x2',
        Uint32Array.from(keys.flatMap(key => [Number(key & 0xffffffffn), Number(key >> 32n)]))
      );
      const areas = includeAreas ? fixture.output('areas', 'float32', 3) : undefined;
      const selected = optional
        ? fixture.output(
            'selected',
            optional === 'pentagons' ? 'uint32' : 'float32',
            optional === 'edgeLengths' ? 30 : 3
          )
        : undefined;
      const operation = new GPUCellMeasuresOperation({
        family: 'h3',
        cells: 'cells',
        rowCount: 3,
        units: 'kilometers',
        maximumEdgeCount: 10,
        output: {
          ...(optional === 'perimeters'
            ? {perimeters: 'selected'}
            : optional === 'edgeLengths'
              ? {edgeLengths: 'selected'}
              : optional === 'pentagons'
                ? {pentagons: 'selected'}
                : {areas: 'areas'}),
          ...(includeAreas ? {areas: 'areas'} : {})
        }
      });
      // Use the operation's public deferred construction with poisoned caller-owned outputs;
      // the existing operation matrix additionally exercises GPUGraph's mixed executor.
      const views = {cells, areas, ...(optional ? {[optional]: selected} : {})};
      fixture.graph.add(
        operation.createContributor(port => views[port as keyof typeof views] as never)
      );
      const compiled = fixture.graph.compile();
      try {
        const actual = await fixture.submitTwice(compiled, async () => ({
          areas: areas ? await fixture.readRows([areas]) : [],
          selected: selected ? await fixture.readRows([selected]) : []
        }));
        const expected =
          optional === 'perimeters'
            ? references.map(row => row[2])
            : optional === 'pentagons'
              ? references.map(row => row[5])
              : optional === 'edgeLengths'
                ? references.flatMap(row => [...row[6], ...Array(10 - row[3]).fill(0)])
                : [];
        for (const [values, literals] of [
          [actual.areas, references.map(row => row[1])],
          [actual.selected, expected]
        ]) {
          values.forEach((value, index) => {
            // f32 boundary projections at resolution 1: 2e-4 relative covers their
            // documented fallback error, far below a missing edge or wrong normalization.
            expect(Number.isFinite(value)).toBe(true);
            if (literals[index] === 0 || (optional === 'pentagons' && values === actual.selected))
              expect(value).toBe(literals[index]);
            else expect(Math.abs(value / literals[index] - 1)).toBeLessThan(2e-4);
          });
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}

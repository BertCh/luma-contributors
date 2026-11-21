// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPointToCell} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell';
import {GPUPointToCellOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell-operation';
import {GPUHealpixPointToCellOperation} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell-operation';
import {
  GPUCellMeasuresOperation,
  type CellMeasuresOperationProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures-operation';
import {
  CPUCellMeasuresOperation,
  computeCellMeasuresOnCPU
} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation';
import {computePointToCellOnCPU} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation';
import {CELL_WRAP_LONGITUDE_WGSL} from '../../../src/gpu-spatial-analysis/cell-indexing/cell-longitude-policy';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('B1 retains rejection of unconditional extended precision on non-H3 indexing', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 1);
    const cells = createTransientView(graph, 'cells', 'uint32x2', 1);
    for (const family of ['quadbin', 'quadkey', 'geohash', 's2'] as const) {
      expect(
        () =>
          new GPUPointToCell({
            family,
            resolution: 5,
            precision: 'double-single',
            positions,
            output: {cells}
          })
      ).toThrow(/precision.*h3/);
    }
  } finally {
    device.destroy();
  }
});

for (const family of ['point', 'healpix'] as const) {
  it(`B2 ${family} rejects nonempty positions and masks before its zero-row branch`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const props = {
        positions: 'positions',
        mask: 'mask',
        rowCount: 0,
        output: {cells: 'cells', validity: 'validity'}
      };
      const createOperation = () =>
        family === 'point'
          ? new GPUPointToCellOperation({...props, family: 'quadbin', resolution: 5})
          : new GPUHealpixPointToCellOperation({...props, order: 1});
      for (const port of ['positions', 'mask'] as const) {
        await expect(
          runOperationGraph(device, {
            inputs: {
              positions: {
                format: 'float32x2',
                values: port === 'positions' ? Float32Array.of(22, 30) : new Float32Array()
              },
              mask: {
                format: 'uint32',
                values: port === 'mask' ? Uint32Array.of(1) : new Uint32Array()
              }
            },
            operations: [createOperation()],
            outputs: ['cells', 'validity']
          })
        ).rejects.toThrow(new RegExp(`${port}.*rowCount.*0.*1`));
      }
      expect(
        await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: new Float32Array()},
            mask: {format: 'uint32', values: new Uint32Array()}
          },
          operations: [createOperation()],
          outputs: ['cells', 'validity']
        })
      ).toEqual({cells: new Uint32Array(), validity: new Uint32Array()});
    } finally {
      device.destroy();
    }
  });
}

it('S1 named measure forms reject an empty terminal selection at construction', () => {
  // @ts-expect-error Empty terminal selection must fail before GPU allocation.
  const empty: CellMeasuresOperationProps['output'] = {};
  for (const Operation of [GPUCellMeasuresOperation, CPUCellMeasuresOperation]) {
    expect(() => new Operation({family: 'h3', cells: 'cells', rowCount: 1, output: empty})).toThrow(
      /output.*at least one/
    );
    for (const output of [
      {areas: 'areas'},
      {perimeters: 'perimeters'},
      {edgeLengths: 'edgeLengths'},
      {edgeCounts: 'edgeCounts'},
      {pentagons: 'pentagons'}
    ]) {
      expect(
        () => new Operation({family: 'h3', cells: 'cells', rowCount: 1, output})
      ).not.toThrow();
    }
  }
});

it('S2 pins independent fine-cell binary64 area for every precision spelling', () => {
  // h3-js cellArea('8ebedac6ecdb89f', 'm2') = 7.1095521951539355;
  // final float32 output is 7.109552383422852. This expectation never invokes our CPU twin.
  for (const precision of [undefined, 'float32', 'double-single'] as const) {
    const result = computeCellMeasuresOnCPU(Uint32Array.of(0x6ecdb89f, 0x08ebedac), {
      family: 'h3',
      precision
    });
    expect(Number.isFinite(result.areas[0])).toBe(true);
    expect(result.areas[0]).toBe(7.109552383422852);
  }
});

it('S3 preserves canonical negative subnormal longitude before floating arithmetic', () => {
  // Tile x=15 immediately west of zero vs x=16 at zero; y=16. Their Morton paths are 597/768.
  expect(
    computePointToCellOnCPU(Float32Array.of(-(2 ** -149), 0, 0, 0), {
      family: 'quadkey',
      resolution: 5,
      longitudePolicy: 'wrap'
    }).cells
  ).toEqual(Uint32Array.of(597, 0x14000000, 768, 0x14000000));
  const body = CELL_WRAP_LONGITUDE_WGSL.slice(
    CELL_WRAP_LONGITUDE_WGSL.indexOf('fn cellWrapLongitudeDegrees')
  );
  // A floating comparison can flush its operand; the original input must survive via integer bits.
  expect(body).toMatch(/bitcast<u32>\(longitude\)/);
  expect(body.indexOf('return longitude;')).toBeLessThan(
    body.indexOf('cellWrapReduceDegrees(longitude)')
  );
});

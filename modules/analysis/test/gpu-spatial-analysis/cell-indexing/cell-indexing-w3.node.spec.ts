// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WRAPPED_CELL_KEYS} from './cell-wrap-fixtures';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  GPUPointToCell,
  type GPUPointToCellProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell';
import {
  computePointToCellOnCPU,
  CPUPointToCellOperation
} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation';
import {
  GPUPointToCellOperation,
  type PointToCellOperationProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('NAM-06 selects extended H3 for shared precision and rejects retired spellings', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const precision of ['double-single', 'float32', 'extended', 'fp64', 'bogus']) {
      const graph = new GPUCommandGraph(device);
      const props = {
        family: 'h3',
        resolution: 5,
        precision,
        positions: createTransientView(graph, 'positions', 'float32x2', 1),
        output: {cells: createTransientView(graph, 'cells', 'uint32x2', 1)}
      } as GPUPointToCellProps;
      if (precision === 'bogus' || precision === 'extended' || precision === 'fp64') {
        expect(() => new GPUPointToCell(props)).toThrow(/precision.*float32.*double-single/);
      } else {
        const contributor = new GPUPointToCell(props);
        const nodes = contributor.getCommandNodes(graph);
        expect(nodes.find(node => node.type === 'compute')!.workload?.variant).toBe(
          precision === 'float32' ? 'h3' : 'h3-extended'
        );
        const sources = getGPUCommandNodeKernelSources(nodes);
        expect(sources[0]!.source.includes('fn cellIndexH3ExtendedFromLngLat')).toBe(
          precision !== 'float32'
        );
      }
    }
    const options = {
      family: 'quadbin',
      resolution: 5,
      precision: 'double-single'
    } as PointToCellOperationProps;
    expect(() => computePointToCellOnCPU(Float32Array.of(0, 0), options)).toThrow(/precision.*h3/);
    for (const Operation of [GPUPointToCellOperation, CPUPointToCellOperation]) {
      expect(
        () =>
          new Operation({...options, positions: 'positions', rowCount: 1, output: {cells: 'cells'}})
      ).toThrow(/precision.*h3/);
    }
  } finally {
    device.destroy();
  }
});

for (const family of ['quadbin', 'quadkey', 'geohash', 'h3', 's2'] as const) {
  it(`DGN-13 wraps equivalent continuous-branch longitudes for ${family}`, () => {
    // 210 - 360 = -150 exactly. Equal angles must identify the same cell at either latitude.
    for (const latitude of [0, 10]) {
      const options = {family, resolution: 5, longitudePolicy: 'wrap' as const};
      const result = computePointToCellOnCPU(
        Float32Array.of(210, latitude, -150, latitude),
        options
      );
      expect(Array.from(result.cells.slice(0, 2))).toEqual(Array.from(result.cells.slice(2)));
      expect(Array.from(result.validity)).toEqual([1, 1]);
      const expected = WRAPPED_CELL_KEYS[family][latitude === 0 ? 0 : 1];
      expect(BigInt(result.cells[0]) | (BigInt(result.cells[1]) << 32n)).toBe(expected);
      if (family === 'quadbin' || family === 'quadkey' || family === 'geohash') {
        const native = computePointToCellOnCPU(Float32Array.of(210, latitude, -150, latitude), {
          family,
          resolution: 5
        });
        expect(Array.from(native.cells.slice(0, 2))).not.toEqual(Array.from(native.cells.slice(2)));
      }
    }
  });
}

it('FRM-12 permits exact empty shapes and lowers empty named GPU forms without accessing output views', () => {
  const props: PointToCellOperationProps = {
    family: 'quadbin',
    resolution: 5,
    rowCount: 0,
    positions: 'positions',
    output: {cells: 'cells', validity: 'validity'}
  };
  for (const Operation of [GPUPointToCellOperation, CPUPointToCellOperation]) {
    const operation = new Operation(props);
    expect(operation.contract.outputs).toEqual({
      cells: expect.objectContaining({length: 0}),
      validity: expect.objectContaining({length: 0})
    });
  }
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 0);
    const primitive = new GPUPointToCellOperation(props).createContributor(port => {
      expect(port).toBe('positions'); // An empty route must validate input without requesting output views.
      return positions as never;
    });
    expect(
      (primitive as {getCommandNodes(graph: unknown): readonly unknown[]}).getCommandNodes(
        undefined
      )
    ).toEqual([]);
  } finally {
    device.destroy();
  }
  expect(computePointToCellOnCPU(new Float32Array(), props)).toEqual({
    cells: new Uint32Array(),
    validity: new Uint32Array()
  });
});

it('DGN-13 validates longitudePolicy and parses every wrapped GPU family', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const family of ['quadbin', 'quadkey', 'geohash', 'h3', 's2'] as const) {
      const graph = new GPUCommandGraph(device);
      const props: GPUPointToCellProps = {
        family,
        resolution: 5,
        longitudePolicy: 'wrap',
        positions: createTransientView(graph, 'positions', 'float32x2', 1),
        output: {cells: createTransientView(graph, 'cells', 'uint32x2', 1)}
      };
      const contributor = new GPUPointToCell(props);
      const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))[0]!.source;
      expect(source).toContain('longitude = cellWrapLongitudeDegrees(longitude)');
      expect(() => new WgslReflect(source)).not.toThrow();
      expect(() => new GPUPointToCell({...props, longitudePolicy: 'bogus' as 'wrap'})).toThrow(
        /longitudePolicy.*family-default.*wrap/
      );
      expect(() =>
        computePointToCellOnCPU(Float32Array.of(0, 0), {
          ...props,
          longitudePolicy: 'bogus' as 'wrap'
        })
      ).toThrow(/longitudePolicy/);
    }
  } finally {
    device.destroy();
  }
});

it('FRM-12 executes CPU empty named forms with exact requested outputs', async () => {
  const props: PointToCellOperationProps = {
    family: 'quadbin',
    resolution: 5,
    rowCount: 0,
    positions: 'positions',
    output: {cells: 'cells', validity: 'validity'}
  };
  const positions = {
    format: 'float32x2' as const,
    length: 0,
    chunks: [],
    pack: () => new Float32Array()
  };
  expect(await new CPUPointToCellOperation(props).execute({inputs: {positions}})).toEqual({
    cells: new Uint32Array(),
    validity: new Uint32Array()
  });
  expect(
    await new CPUPointToCellOperation({...props, output: {cells: 'cells'}}).execute({
      inputs: {positions}
    })
  ).toEqual({cells: new Uint32Array()});
});

it('FRM-12 runs a zero-row named GPU graph on a null device without commands', async () => {
  const {runOperationGraph} = await import('../../utils/mixed-execution-test-utils');
  const {GPUHealpixPointToCellOperation} = await import(
    '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell-operation'
  );
  const device = createNullWebGPUDevice();
  try {
    const props = {
      positions: 'positions',
      rowCount: 0,
      output: {cells: 'cells', validity: 'validity'}
    };
    for (const operation of [
      new GPUPointToCellOperation({...props, family: 'quadbin', resolution: 5}),
      new GPUHealpixPointToCellOperation({...props, order: 1})
    ]) {
      const rows = await runOperationGraph(device, {
        inputs: {positions: {format: 'float32x2', values: new Float32Array()}},
        operations: [operation],
        outputs: ['cells', 'validity']
      });
      expect(rows).toEqual({cells: new Uint32Array(), validity: new Uint32Array()});
    }
  } finally {
    device.destroy();
  }
});

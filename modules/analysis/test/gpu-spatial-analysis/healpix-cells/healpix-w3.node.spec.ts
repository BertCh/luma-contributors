// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type CPUHealpixPointToCellResult,
  type HealpixDGGSJSONFromReadbackProps,
  computeHealpixPointToCellOnCPU as computePublicHealpix,
  createHealpixDGGSJSONFromReadback
} from '../../../src/gpu-spatial-analysis/index';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUHealpixPointToCell} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {expect, expectTypeOf, it} from 'vitest';
import {
  computeHealpixPointToCellOnCPU,
  CPUHealpixPointToCellOperation
} from '../../../src/gpu-spatial-analysis/healpix-cells/cpu-healpix-point-to-cell-operation';
import {
  GPUHealpixPointToCellOperation,
  type HealpixPointToCellOperationProps
} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell-operation';
import * as interchange from '../../../src/gpu-spatial-analysis/index';

it('UCT-06 rejects RING on CPU, named forms and sparse JSON', () => {
  const scheme = 'ring' as 'nested';
  expect(() => computeHealpixPointToCellOnCPU(Float32Array.of(0, 90), {order: 1, scheme})).toThrow(
    /scheme.*nested.*uniq.*RING/
  );
  for (const Operation of [GPUHealpixPointToCellOperation, CPUHealpixPointToCellOperation]) {
    expect(
      () =>
        new Operation({
          scheme,
          order: 1,
          rowCount: 1,
          positions: 'positions',
          output: {cells: 'cells'}
        })
    ).toThrow(/scheme.*RING/);
  }
  expect(() =>
    interchange.createHealpixDGGSJSON({
      scheme,
      order: 1,
      cells: [3n],
      properties: {count: {values: [1]}}
    })
  ).toThrow(/scheme.*RING/);
});

it('EGR-03 accepts bigint typed arrays as explicit sparse cells', () => {
  const result = interchange.createHealpixDGGSJSON({
    order: 1,
    cells: BigUint64Array.of(0n, 3n),
    properties: {count: {values: [4, 7]}}
  });
  expect(result.values['count'][0].data).toEqual([4, null, null, 7]);
});

it('EGR-03 refuses overflow, checks count and joins packed high keys losslessly', () => {
  const createFromReadback = createHealpixDGGSJSONFromReadback;
  expectTypeOf(createFromReadback).parameter(0).toEqualTypeOf<HealpixDGGSJSONFromReadbackProps>();
  expect(createFromReadback).toBeTypeOf('function');
  const props = {
    order: 29,
    ancestorOrder: 29,
    ancestor: 0x123456789abcdefn,
    properties: {count: {values: [7, 999]}},
    output: {
      cells: Uint32Array.of(0x89abcdef, 0x01234567, 0, 0),
      status: {count: 1, requiredCount: 1, overflow: 0}
    }
  } satisfies HealpixDGGSJSONFromReadbackProps;
  expect(createFromReadback(props).values['count'][0].data).toEqual([7]);
  for (const cells of [
    props.output.cells.buffer,
    (() => {
      const bytes = new Uint8Array(props.output.cells.byteLength + 3);
      bytes.set(new Uint8Array(props.output.cells.buffer), 3);
      return bytes.subarray(3);
    })()
  ]) {
    expect(
      createFromReadback({
        ...props,
        output: {
          ...props.output,
          cells,
          status: {
            count: Uint32Array.of(1),
            requiredCount: Uint32Array.of(1),
            overflow: Uint32Array.of(0)
          }
        }
      }).values['count'][0].data
    ).toEqual([7]);
  }
  expect(() =>
    createFromReadback({...props, output: {...props.output, cells: new Uint8Array(3)}})
  ).toThrow(/whole/);
  expect(() =>
    createFromReadback({
      ...props,
      output: {...props.output, status: {...props.output.status, count: new Uint32Array()}}
    })
  ).toThrow(/at least 1/);
  expect(() =>
    createFromReadback({
      ...props,
      output: {...props.output, status: {...props.output.status, count: -1}}
    })
  ).toThrow(/count/);
  expect(() => createFromReadback({...props, wordOrder: 'bogus' as 'high-low'})).toThrow(
    /wordOrder/
  );
  expect(() => createFromReadback({...props, ancestor: 0n})).toThrow(/outside the ancestral/);
  expect(
    createFromReadback({
      ...props,
      wordOrder: 'high-low',
      output: {...props.output, cells: Uint32Array.of(0x01234567, 0x89abcdef)}
    }).values['count'][0].data
  ).toEqual([7]);
  expect(() =>
    createFromReadback({
      ...props,
      output: {...props.output, status: {count: 1, requiredCount: 2, overflow: 1}}
    })
  ).toThrow(/requiredCount.*2/);
  expect(() =>
    createFromReadback({
      ...props,
      output: {...props.output, status: {count: 3, requiredCount: 3, overflow: 0}}
    })
  ).toThrow(/count/);
  expect(() =>
    createFromReadback({...props, output: {...props.output, cells: Uint32Array.of(1)}})
  ).toThrow(/whole|words/);
  // At order 1, UNIQ marker is 4^(1+1)=16. Pixel 3 lives at the last descendant of face 0.
  for (const [scheme, key] of [
    ['nested', 3],
    ['uniq', 19]
  ] as const) {
    expect(
      createFromReadback({
        order: 1,
        scheme,
        properties: {count: {values: [7]}},
        output: {cells: Uint32Array.of(key, 0), status: {count: 1, requiredCount: 1, overflow: 0}}
      }).values['count'][0].data
    ).toEqual([null, null, null, 7]);
  }
});

it('FRM-12 exposes exact empty named shapes without a GPU dispatch', () => {
  const props: HealpixPointToCellOperationProps = {
    order: 1,
    rowCount: 0,
    positions: 'positions',
    output: {cells: 'cells', validity: 'validity'}
  };
  for (const Operation of [GPUHealpixPointToCellOperation, CPUHealpixPointToCellOperation]) {
    expect(new Operation(props).contract.outputs).toEqual({
      cells: expect.objectContaining({length: 0}),
      validity: expect.objectContaining({length: 0})
    });
  }
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 0);
    const primitive = new GPUHealpixPointToCellOperation(props).createContributor(port => {
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
  expect(computeHealpixPointToCellOnCPU(new Float32Array(), {order: 1})).toEqual({
    cells: new Uint32Array(),
    validity: new Uint32Array()
  });
});

it('TYP-16 gives the public pure function a named HEALPix result', () => {
  expectTypeOf<
    ReturnType<typeof computePublicHealpix>
  >().toEqualTypeOf<CPUHealpixPointToCellResult>();
  const result: CPUHealpixPointToCellResult = computePublicHealpix(Float32Array.of(0, 90), {
    order: 1
  });
  // Order-one north-face zero has both local bits set: nested key 3, with validity 1.
  expect(result).toEqual({cells: Uint32Array.of(3, 0), validity: Uint32Array.of(1)});
});

it('FRM-12 executes CPU HEALPix empty named forms with exact requested outputs', async () => {
  const props: HealpixPointToCellOperationProps = {
    order: 1,
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
  expect(await new CPUHealpixPointToCellOperation(props).execute({inputs: {positions}})).toEqual({
    cells: new Uint32Array(),
    validity: new Uint32Array()
  });
  expect(
    await new CPUHealpixPointToCellOperation({...props, output: {cells: 'cells'}}).execute({
      inputs: {positions}
    })
  ).toEqual({cells: new Uint32Array()});
});

it('UCT-06 validates direct HEALPix schemes and defaults only undefined', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      order: 1,
      positions: createTransientView(graph, 'positions', 'float32x2', 1),
      output: {cells: createTransientView(graph, 'cells', 'uint32x2', 1)}
    };
    for (const scheme of ['ring', 'RING', 'bogus', null] as unknown as 'nested'[]) {
      expect(() => new GPUHealpixPointToCell({...props, scheme})).toThrow(/scheme.*RING.*received/);
      expect(() =>
        computeHealpixPointToCellOnCPU(Float32Array.of(0, 90), {order: 1, scheme})
      ).toThrow(/scheme.*RING.*received/);
      expect(() =>
        interchange.createHealpixDGGSJSON({order: 1, scheme, cells: [3n], properties: {}})
      ).toThrow(/scheme.*RING.*received/);
    }
    for (const scheme of [undefined, 'nested', 'uniq'] as const) {
      expect(() => new GPUHealpixPointToCell({...props, scheme})).not.toThrow();
      const cells = computeHealpixPointToCellOnCPU(Float32Array.of(0, 90), {
        order: 1,
        scheme
      }).cells;
      expect(cells).toEqual(Uint32Array.of(scheme === 'uniq' ? 19 : 3, 0));
    }
  } finally {
    device.destroy();
  }
});

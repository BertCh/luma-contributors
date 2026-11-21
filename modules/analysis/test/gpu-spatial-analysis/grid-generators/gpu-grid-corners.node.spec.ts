// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUGridCellCount,
  getGPUGridCornerCount,
  getGPUGridVerticesPerCell,
  GPUGridGenerator,
  type GPUGridGeneratorProps
} from '../../../src/gpu-spatial-analysis/grid-generators/index';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('GPUGridGenerator validates corner output capacities, types, dependencies and graph ownership', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `corner-view-${serial++}`, format, length);
  const base: GPUGridGeneratorProps = {
    gridType: 'square',
    columns: 2,
    rows: 3,
    parameters: view('float32', 4),
    output: {corners: view('float32x2', 12)}
  };
  const extent = {
    kind: 'polygons' as const,
    positions: view('float32x2', 4),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2)
  };
  expect(new GPUGridGenerator(base).cornerCount).toBe(12);
  expect(new GPUGridGenerator(base).getCommandNodes(graph).map(node => node.id)).toEqual([
    'grid-generator-corners'
  ]);
  for (const gridType of ['triangle', 'point'] as const) {
    expect(() => new GPUGridGenerator({...base, gridType})).toThrow(/output.corners/);
  }
  expect(() => new GPUGridGenerator({...base, output: {corners: view('float32x2', 11)}})).toThrow(
    /output.corners/
  );
  expect(
    () => new GPUGridGenerator({...base, output: {corners: view('uint32', 12) as never}})
  ).toThrow();
  expect(() => new GPUGridGenerator({...base, extent})).toThrow(/intersection/);
  expect(
    () =>
      new GPUGridGenerator({
        ...base,
        output: {...base.output, cornerIntersects: view('uint32', 12)}
      })
  ).toThrow(/intersection/);
  expect(
    () =>
      new GPUGridGenerator({
        ...base,
        extent,
        output: {...base.output, cornerIntersects: view('uint32', 11)}
      })
  ).toThrow(/one row per/);
  expect(
    () =>
      new GPUGridGenerator({
        ...base,
        extent,
        output: {positions: view('float32x2', 24), cornerIntersects: view('uint32', 12)}
      })
  ).toThrow(/one row per/);
  expect(
    () =>
      new GPUGridGenerator({
        ...base,
        extent,
        output: {...base.output, intersects: view('uint32', 6)}
      })
  ).toThrow(/cell geometry/);
  expect(
    new GPUGridGenerator({
      ...base,
      extent,
      output: {...base.output, cornerIntersects: view('uint32', 12)}
    })
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual([
    'grid-generator-corners',
    'grid-generator-extent-bounds-initialize',
    'grid-generator-extent-bounds',
    'grid-generator-corner-intersects'
  ]);
  const otherGraph = new GPUCommandGraph(device);
  const foreignCorners = createTransientView(otherGraph, 'foreign-corners', 'float32x2', 12);
  expect(() =>
    new GPUGridGenerator({...base, output: {corners: foreignCorners}}).getCommandNodes(graph)
  ).toThrow();
  device.destroy();
});

it('GPUGridGenerator polygon and corner kernels parse for every grid type', () => {
  const device = createNullWebGPUDevice();
  for (const [gridType, hexOrientation] of [
    ['square', undefined],
    ['hex', undefined],
    ['hex', 'flat'],
    ['triangle', undefined]
  ] as const) {
    const graph = new GPUCommandGraph(device);
    const columns = 3;
    const rows = 2;
    const generator = new GPUGridGenerator({
      gridType,
      hexOrientation,
      columns,
      rows,
      parameters: createTransientView(graph, 'parameters', 'float32', 4),
      output: {
        positions: createTransientView(
          graph,
          'positions',
          'float32x2',
          getGPUGridCellCount(gridType, columns, rows) * getGPUGridVerticesPerCell(gridType)
        ),
        centers: createTransientView(
          graph,
          'centers',
          'float32x2',
          getGPUGridCellCount(gridType, columns, rows)
        ),
        ...(gridType === 'triangle'
          ? {}
          : {
              corners: createTransientView(
                graph,
                'corners',
                'float32x2',
                getGPUGridCornerCount(gridType, columns, rows)
              )
            })
      }
    });
    const kernels = getGPUCommandNodeKernelSources(generator.getCommandNodes(graph));
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels) {
      expect(() => new WgslReflect(kernel.source), `${gridType} ${kernel.id}`).not.toThrow();
    }
  }
  device.destroy();
});

// GEO-109: the kernels place cells with f32 lattice indices, exact only up to 2^24, so a one-row
// square grid of 2^24 + 2 columns used to repeat cell positions silently. Closed-form limits:
// square/point `column + 1 <= 2^24`; triangle `2 * columns + 2 <= 2^24`; pointy hex
// `2 * columns + 1 <= 2^24` and `3 * rows + 1 <= 2^24` (flat hex swaps the axes).
it('GPUGridGenerator rejects columns and rows whose lattice indices float32 cannot represent', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const create = (
    gridType: 'square' | 'point' | 'triangle' | 'hex',
    columns: number,
    rows: number,
    hexOrientation?: 'pointy' | 'flat'
  ) => {
    const cellCount = getGPUGridCellCount(gridType, columns, rows);
    const vertexCount = cellCount * getGPUGridVerticesPerCell(gridType);
    return new GPUGridGenerator({
      gridType,
      columns,
      rows,
      hexOrientation,
      parameters: createTransientView(graph, `limit-parameters-${serial++}`, 'float32', 4),
      output:
        gridType === 'point'
          ? {
              centers: createTransientView(
                graph,
                `limit-centers-${serial++}`,
                'float32x2',
                cellCount
              )
            }
          : {
              positions: createTransientView(
                graph,
                `limit-positions-${serial++}`,
                'float32x2',
                vertexCount
              )
            }
    });
  };
  const exact = 2 ** 24;
  const cases: [Parameters<typeof create>, 'columns' | 'rows', number][] = [
    [['point', exact / 2, 1], 'columns', exact / 2],
    [['square', exact, 1], 'columns', exact],
    [['square', 1, exact], 'rows', exact],
    [['triangle', exact / 2 - 1, 1], 'columns', exact / 2 - 1],
    [['triangle', 1, exact], 'rows', exact],
    [['hex', 8388607, 1], 'columns', 8388607],
    [['hex', 1, 5592405], 'rows', 5592405],
    [['hex', 5592405, 1, 'flat'], 'columns', 5592405],
    [['hex', 1, 8388607, 'flat'], 'rows', 8388607]
  ];
  for (const [[gridType, columns, rows, orientation], name, maximum] of cases) {
    const label = `${gridType} ${orientation ?? ''} ${name}`;
    expect(() => create(gridType, columns, rows, orientation), label).not.toThrow();
    const larger: [number, number] = name === 'columns' ? [columns + 1, rows] : [columns, rows + 1];
    expect(() => create(gridType, ...larger, orientation), label).toThrow(
      gridType === 'point'
        ? new RegExp(`${name}.*at most ${maximum}`)
        : `grid-generator ${name} must be an integer in [1, ${maximum}]`
    );
  }
  // The finding's reproduction: one row of 2^24 + 2 square cells.
  expect(() => create('square', exact + 2, 1)).toThrow(/columns must be an integer/);
});

it('admits exact half-cell center indices separately from square integer corners', () => {
  // Binary32 has 24 significand bits. Half-integers are exact only below 2^23;
  // round-to-even at columns 8388609 and 8388610 maps both unit centers to 8388610.
  expect([8388609, 8388610].map(column => Math.fround(Math.fround(column) + 0.5))).toEqual([
    8388610, 8388610
  ]);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => GPUGridGenerator.getMaximumAxisCellCounts({gridType: 'unknown' as never})).toThrow(
    /gridType.*square.*point.*received unknown/
  );
  expect(() =>
    GPUGridGenerator.getMaximumAxisCellCounts({gridType: 'hex', hexOrientation: 'unknown' as never})
  ).toThrow(/pointy.*flat.*received unknown/);
  expect(GPUGridGenerator.getMaximumAxisCellCounts({gridType: 'point'})).toEqual({
    columns: 8388608,
    rows: 8388608
  });
  expect(GPUGridGenerator.getMaximumAxisCellCounts({gridType: 'square', centers: true})).toEqual({
    columns: 8388608,
    rows: 8388608
  });
  expect(GPUGridGenerator.getMaximumAxisCellCounts({gridType: 'square'})).toEqual({
    columns: 16777216,
    rows: 16777216
  });
  const parameters = createTransientView(graph, 'parameters', 'float32', 4);
  const centers = createTransientView(graph, 'centers', 'float32x2', 1);
  try {
    for (const gridType of ['point', 'square'] as const) {
      for (const axis of ['columns', 'rows'] as const) {
        expect(
          () =>
            new GPUGridGenerator({
              gridType,
              columns: axis === 'columns' ? 8388609 : 1,
              rows: axis === 'rows' ? 8388609 : 1,
              parameters,
              output: {centers}
            })
        ).toThrow(new RegExp(`${axis}.*8388608.*received 8388609`));
      }
    }
  } finally {
    device.destroy();
  }
});

it('N2 sizing helpers reject unknown grid types and state their supported values', () => {
  expect(() => getGPUGridVerticesPerCell('hexagonal' as never)).toThrow(
    /square.*hex.*triangle.*point.*hexagonal/
  );
  expect(() => getGPUGridCellCount('hexagonal' as never, 2, 3)).toThrow(
    /square.*hex.*triangle.*point.*hexagonal/
  );
  expect(() => getGPUGridCornerCount('hexagonal' as never, 2, 3)).toThrow(/square.*hex.*hexagonal/);
  expect(() => getGPUGridCornerCount('point' as never, 2, 3)).toThrow(
    /square.*hex.*received point/
  );
  expect(getGPUGridCellCount('hex', 2, 3)).toBe(6);
  expect(getGPUGridCornerCount('hex', 2, 3)).toBe(22);
  expect(getGPUGridCellCount('triangle', 2, 3)).toBe(12);
  expect(getGPUGridCornerCount('square', 2, 3)).toBe(12);
});

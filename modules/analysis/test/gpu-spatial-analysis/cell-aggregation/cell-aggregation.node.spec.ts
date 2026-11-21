// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {cellToParent, getPentagons, getRes0Cells, latLngToCell, cellToChildren} from 'h3-js';
import {expect, it, vi} from 'vitest';
import {loadGuideFunction} from './family-guide-execution.test-utils';
import {
  GPUCellAggregation,
  GPUCellLevelSelection,
  GPUCellPyramid,
  GPUCellRollup,
  type GPUCellAggregationProps,
  type GPUCellTable
} from '../../../src/gpu-spatial-analysis/cell-aggregation/index';
import {getCellKeyLayout} from '../../../src/gpu-spatial-analysis/cell-aggregation/cell-keys';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  aggregateCellsOnCPU,
  getQuadbinTileX,
  getQuadbinTileY,
  getQuadbinTileYFloat64,
  getScaledValue,
  h3ToBigInt,
  quadbinCellToParent,
  quadbinCellToTile,
  quadbinIsValidCell,
  quadbinPointToCell,
  quadbinTileToCell
} from './cell-aggregation-oracle';
import {createPointPositions, createRandom, getEdgeCasePoints} from './cell-aggregation-points';

let serial = 0;

function createTable(graph: GPUCommandGraph, capacity: number, columns = true): GPUCellTable {
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    cells: view('uint32x2', capacity),
    counts: view('uint32', capacity),
    ...(columns
      ? {
          sums: view('uint32x2', capacity),
          sumValues: view('float32', capacity),
          minimums: view('float32', capacity),
          maximums: view('float32', capacity)
        }
      : {}),
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      candidateOverflow: view('uint32', 1)
    }
  };
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCellAggregationProps> = {}
): GPUCellAggregationProps {
  return {
    family: 'quadbin',
    resolution: 8,
    positions: createTransientView(graph, `positions-${serial++}`, 'float32x2', 10),
    values: createTransientView(graph, `values-${serial++}`, 'float32', 10),
    output: createTable(graph, 16),
    ...overrides
  };
}

it('quadbin oracle matches published quadbin-js values', () => {
  // quadbin-js README: tileToCell({x: 0, y: 0, z: 0}).
  expect(quadbinTileToCell(0, 0, 0)).toBe(5192650370358181887n);
  // quadbin-py README: point_to_cell(-3.7038, 40.4168, 4).
  expect(quadbinPointToCell(-3.7038, 40.4168, 4)).toBe(5207251884775047167n);
  expect(quadbinCellToTile(5207251884775047167n)).toEqual({x: 7, y: 6, z: 4});
  expect(quadbinCellToParent(5207251884775047167n, 0)).toBe(5192650370358181887n);
  expect(quadbinIsValidCell(5207251884775047167n)).toBe(true);
  expect(quadbinIsValidCell(5207251884775047167n - 1n)).toBe(false);
  // Antimeridian wraps to column 0; latitude clips to the Mercator limit.
  expect(getQuadbinTileX(180, 10)).toBe(0);
  expect(getQuadbinTileX(-180, 10)).toBe(0);
  expect(getQuadbinTileX(Math.fround(179.99999), 10)).toBe(1023);
  expect(getQuadbinTileX(-1e-40, 10)).toBe(511);
  expect(getQuadbinTileX(0, 10)).toBe(512);
  expect(getQuadbinTileX(-0, 10)).toBe(512);
  expect(getQuadbinTileY(90, 10)).toBe(0);
  expect(getQuadbinTileY(-90, 10)).toBe(1023);
});

it('quadbin f32 keys agree with the f64 formula except within one tile of an edge', () => {
  const report: Record<number, number> = {};
  const errors: Record<number, number> = {};
  for (const resolution of [1, 4, 8, 12, 16, 20, 24, 26]) {
    const positions = createPointPositions(resolution, 20000, resolution);
    let mismatches = 0;
    let maximumError = 0;
    for (let row = 0; row < positions.length / 2; row++) {
      const latitude = positions[2 * row + 1];
      const y32 = getQuadbinTileY(latitude, resolution);
      const y64 = getQuadbinTileYFloat64(latitude, resolution);
      if (y32 !== y64) {
        // Rows past 20000 are the deliberate tile-edge cases; rate only the random points.
        mismatches += row < 20000 ? 1 : 0;
        maximumError = Math.max(maximumError, Math.abs(y32 - y64));
      }
    }
    report[resolution] = mismatches / 20000;
    errors[resolution] = maximumError;
    // f32 keeps about 1e-7 of the Mercator row range (an f32 latitude near the pole is itself
    // only that precise), so only levels finer than 22 may miss by more than one row.
    expect(maximumError, `resolution ${resolution}`).toBeLessThanOrEqual(
      Math.max(1, Math.ceil(2 ** resolution * 1e-7))
    );
  }
  // Resolution 12 is ~10 m tiles; the f32 row matches the f64 formula almost everywhere there.
  expect(report[12]).toBeLessThan(0.002);
  expect(report[4]).toBeLessThan(0.0005);
  // eslint-disable-next-line no-console
  console.log('quadbin f32 row vs f64: mismatch rate', report, 'max rows', errors);
});

it('H3 parents are bit truncations (pentagons, poles, antimeridian)', () => {
  const random = createRandom(3);
  const cells: string[] = [...getRes0Cells(), ...getPentagons(5), ...getPentagons(15)];
  for (let index = 0; index < 400; index++) {
    cells.push(latLngToCell(random() * 180 - 90, random() * 360 - 180, Math.floor(random() * 16)));
  }
  for (const [longitude, latitude] of getEdgeCasePoints(10)) {
    if (Number.isFinite(longitude) && Math.abs(longitude) <= 180) {
      cells.push(latLngToCell(latitude, longitude, 12));
    }
  }
  cells.push(...cellToChildren(getPentagons(2)[0], 4));
  for (const cell of cells) {
    const key = h3ToBigInt(cell);
    const resolution = Number((key >> 52n) & 0xfn);
    for (let parent = 0; parent <= resolution; parent++) {
      const layout = getCellKeyLayout('h3', parent);
      const compact = (key >> BigInt(layout.lowBit)) & ((1n << BigInt(layout.width)) - 1n);
      const rebuilt =
        (BigInt(layout.headerHigh | (parent << 20)) << 32n) |
        (compact << BigInt(layout.lowBit)) |
        ((1n << BigInt(layout.lowBit)) - 1n);
      expect(rebuilt.toString(16)).toBe(cellToParent(cell, parent));
    }
  }
});

it('fixed-point scaling rounds half to even and saturates', () => {
  expect(getScaledValue(0.5, 1)).toBe(0n);
  expect(getScaledValue(1.5, 1)).toBe(2n);
  expect(getScaledValue(-2.5, 1)).toBe(-2n);
  expect(getScaledValue(-0.25, 4)).toBe(-1n);
  expect(getScaledValue(3e38, 65536)).toBe(BigInt(Math.fround(2 ** 62 - 2 ** 38)));
});

it('the cell aggregation oracle saturates fixed-point sums instead of wrapping (DGGS-102)', () => {
  // Independent expectation, BigInt arithmetic (numpy float32 for the products):
  //   fround(fround(7e13) * 65536) = 4587520127238406144, three rows = 13762560381715218432,
  //   above 2^63 - 1 = 9223372036854775807, so the sum saturates (the old oracle wrapped it to
  //   -4684183691994333184, a sumValue of -7.147e13).
  //   fround(1e14) * 65536 is above the largest scaled row fround(2^62 - 2^38) =
  //   4611685743549480960, so the row saturates there.
  const keys = [0, 0, 0, 1, 2, 2, 2, 3, 3].map(column => quadbinTileToCell(column, 0, 2));
  const values = Float32Array.of(7e13, 7e13, 7e13, 1e14, -7e13, -7e13, -7e13, 1.5, 2.5);
  const cells = aggregateCellsOnCPU({
    family: 'quadbin',
    resolution: 2,
    keys,
    values,
    sumScale: 65536
  });
  expect(cells.map(cell => [cell.count, cell.sum, cell.sumOverflow])).toEqual([
    [3, 9223372036854775807n, true],
    [1, 4611685743549480960n, true],
    [3, -9223372036854775807n, true],
    [2, 262144n, false]
  ]);
});

it('cell tables declare sum guards, saturation and sumOverflow on every path (DGGS-102, DGGS-107)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'cell-sum-overflow'});
  const withFlag = (table: GPUCellTable): GPUCellTable => ({
    ...table,
    status: {
      ...table.status,
      sumOverflow: createTransientView(graph, `flag-${serial++}`, 'uint32', 1)
    }
  });
  const source = withFlag(createTable(graph, 16));
  const contributors = [
    new GPUCellAggregation(createProps(graph, {output: source})),
    new GPUCellAggregation(
      createProps(graph, {output: withFlag(createTable(graph, 16)), scratchTileSize: 4})
    ),
    new GPUCellRollup({
      family: 'quadbin',
      sourceResolution: 8,
      resolution: 4,
      source,
      output: withFlag(createTable(graph, 8))
    })
  ];
  for (const [index, contributor] of contributors.entries()) {
    const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
    const ids = [...sources.keys()];
    const label = `contributor ${index}`;
    expect(
      ids.some(id => /saturate/.test(id)),
      label
    ).toBe(true);
    expect(
      ids.some(id => /sum-status/.test(id)),
      label
    ).toBe(true);
    const finish = [...sources.entries()].filter(
      ([id]) => /finish/.test(id) && /getScaledQuotient\(cellI64ToF32/.test(sources.get(id)!)
    );
    expect(finish.length, `${label} sumValues use the exponent-scaled quotient`).toBeGreaterThan(0);
    for (const [id, wgsl] of sources) {
      expect(/\/ SUM_SCALE/.test(wgsl), `${label} ${id} divides by SUM_SCALE`).toBe(false);
    }
  }
});

it('GPUCellAggregation validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const expectThrows = (overrides: Partial<GPUCellAggregationProps>, message: RegExp) =>
    expect(() => new GPUCellAggregation(createProps(graph, overrides))).toThrow(message);
  expectThrows({resolution: 27}, /resolution/);
  expectThrows({family: 'h3'}, /quadbin family only/);
  expectThrows(
    {cells: createTransientView(graph, 'cells-a', 'uint32x2', 10)},
    /exactly one of positions or cells/
  );
  expectThrows({values: undefined}, /needs values/);
  expectThrows({values: createTransientView(graph, 'values-b', 'float32', 9)}, /length/);
  expectThrows({sumScale: 0}, /sumScale/);
  expectThrows(
    {
      output: {
        ...createTable(graph, 16),
        counts: createTransientView(graph, 'counts-c', 'uint32', 15)
      }
    },
    /same length/
  );
  const positions = createTransientView(graph, 'positions-d', 'float32x2', 10);
  expectThrows(
    {
      positions,
      output: {
        ...createTable(graph, 16),
        status: {
          ...createTable(graph, 16).status,
          count: createTransientView(graph, 'x', 'uint32', 0)
        }
      }
    },
    /one uint32 row/
  );
  device.destroy();
});

it('GPUCellAggregation emits deterministic nodes with one or two key sorts', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const getIds = (resolution: number) => {
    const contributor = new GPUCellAggregation(
      createProps(graph, {id: `agg-${resolution}`, resolution})
    );
    const first = contributor.getCommandNodes(graph).map(node => node.id);
    return first;
  };
  const single = getIds(15);
  const double = getIds(16);
  expect(single.some(id => id.includes('-sort-high'))).toBe(false);
  expect(double.some(id => id.includes('-sort-high'))).toBe(true);
  expect(single[0]).toBe('agg-15-chunk-parameters-upload');
  expect(single.at(-1)).toBe('agg-15-candidate-status');
  device.destroy();
});

it('GPUCellRollup and GPUCellPyramid validate levels and columns', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const source = createTable(graph, 16, false);
  expect(
    () =>
      new GPUCellRollup({
        family: 'quadbin',
        sourceResolution: 6,
        resolution: 4,
        source,
        output: createTable(graph, 8)
      })
  ).toThrow(/source.sums/);
  expect(
    () =>
      new GPUCellRollup({
        family: 'h3',
        sourceResolution: 4,
        resolution: 6,
        source,
        output: createTable(graph, 8, false)
      })
  ).toThrow(/must not exceed/);
  const props = createProps(graph);
  const {output: _output, resolution: _resolution, ...rowProps} = props;
  expect(
    () =>
      new GPUCellPyramid({
        ...rowProps,
        resolutions: [8, 8],
        output: {levels: [createTable(graph, 16), createTable(graph, 16)]}
      })
  ).toThrow(/strictly decrease/);
  // Coarse levels with sums get transient sums on intermediate levels.
  const pyramid = new GPUCellPyramid({
    ...rowProps,
    resolutions: [8, 6, 2],
    output: {
      levels: [
        createTable(graph, 16, false),
        createTable(graph, 16, false),
        createTable(graph, 16)
      ],
      levelCounts: createTransientView(graph, 'level-counts', 'uint32', 3)
    }
  });
  const ids = pyramid.getCommandNodes(graph).map(node => node.id);
  expect(ids.filter(id => id.endsWith('-accumulate-sums'))).toEqual([
    'cell-pyramid-level-0-accumulate-sums',
    'cell-pyramid-level-1-accumulate-sums',
    'cell-pyramid-level-2-accumulate-sums'
  ]);
  expect(pyramid.levelFirstRows).toEqual([0, 0, 0]);
  expect(
    () =>
      new GPUCellLevelSelection({
        levelCounts: createTransientView(graph, 'counts-e', 'uint32', 3),
        activeLevel: createTransientView(graph, 'active-e', 'uint32', 1),
        levelFirstRows: [0, 1],
        output: {status: {count: createTransientView(graph, 'count-e', 'uint32', 1)}}
      })
  ).toThrow(/levelFirstRows/);
  device.destroy();
});

it('GPUCellPyramid level publishes share one pipeline for N and N + 1 levels (rule 12)', () => {
  const getPipelineCount = (levelCount: number, idPattern: RegExp) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const {output: _output, resolution: _resolution, ...rowProps} = createProps(graph);
    const resolutions = Array.from({length: levelCount}, (_, level) => 12 - level);
    const pyramid = new GPUCellPyramid({
      ...rowProps,
      resolutions,
      output: {
        levels: resolutions.map(() => createTable(graph, 16)),
        levelCounts: createTransientView(graph, `level-counts-${serial++}`, 'uint32', levelCount)
      }
    });
    const count = getGPUCommandNodePipelineKeys(pyramid.getCommandNodes(graph), {idPattern}).size;
    device.destroy();
    return count;
  };
  // Each level used to publish into its own row of the packed levelCounts view, baking a
  // different element offset into every level's publish kernel. (The roll-up key kernels still
  // bake each level's resolution.)
  const publish = /-level-\d+-publish$/;
  expect(getPipelineCount(5, publish)).toBe(getPipelineCount(4, publish));
  expect(getPipelineCount(4, publish)).toBeLessThanOrEqual(2);
  expect(getPipelineCount(5, /-level-counts$/)).toBe(1);
});

it('cell operations declare their coordinate space', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const planar = {coordinateSpace: 'planar', metric: 'native', units: 'native'} as const;
  const keyed = (overrides: Partial<GPUCellAggregationProps> = {}) => {
    const {positions: _positions, ...rest} = createProps(graph);
    return {
      ...rest,
      cells: createTransientView(graph, `cells-${serial++}`, 'uint32x2', 10),
      ...overrides
    };
  };
  // Points are longitude/latitude degrees: planar throws, the default is lon/lat.
  expect(new GPUCellAggregation(createProps(graph)).spatialContext).toMatchObject({
    coordinateSpace: 'longitude-latitude',
    metric: 'none'
  });
  expect(() => new GPUCellAggregation(createProps(graph, {spatialContext: planar}))).toThrow(
    /does not support planar coordinates/
  );
  expect(
    () =>
      new GPUCellAggregation(
        createProps(graph, {
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
        })
      )
  ).toThrow(/does not support the great-circle metric/);
  // Pre-keyed rows are discrete-grid; the family must match.
  expect(new GPUCellAggregation(keyed()).spatialContext).toMatchObject({
    coordinateSpace: 'discrete-grid',
    gridFamily: 'quadbin'
  });
  expect(() => new GPUCellAggregation(keyed({spatialContext: planar}))).toThrow(
    /does not support planar coordinates/
  );
  const grid = (gridFamily: string, gridResolution?: number) =>
    ({coordinateSpace: 'discrete-grid', metric: 'none', gridFamily, gridResolution}) as const;
  expect(() => new GPUCellAggregation(keyed({spatialContext: grid('h3')}))).toThrow(
    /gridFamily must be quadbin/
  );
  expect(() => new GPUCellAggregation(keyed({spatialContext: grid('Quadbin', 10)}))).not.toThrow();
  expect(() => new GPUCellAggregation(keyed({spatialContext: grid('quadbin', 6)}))).toThrow(
    /coarser than the output resolution 8/
  );
  const rollup = (spatialContext: typeof planar | ReturnType<typeof grid>) =>
    new GPUCellRollup({
      family: 'h3',
      sourceResolution: 6,
      resolution: 4,
      spatialContext,
      source: createTable(graph, 16),
      output: createTable(graph, 8)
    });
  expect(() => rollup(grid('h3', 6))).not.toThrow();
  expect(() => rollup(grid('quadbin', 6))).toThrow(/gridFamily must be h3/);
  expect(() => rollup(planar)).toThrow(/does not support planar coordinates/);
});

it('UCT-09 guide pyramid imports caller counts separately into both graphs', () => {
  const device = createNullWebGPUDevice();
  const imports = vi.spyOn(GPUCommandGraph.prototype, 'importBuffer');
  const allocations = vi.spyOn(device, 'createBuffer');
  const additions = vi.spyOn(GPUCommandGraph.prototype, 'add');
  const declare = loadGuideFunction<
    (device: Device) => {
      countsBuffer: Buffer;
      activeLevelBuffer: Buffer;
      positionsBuffer: Buffer;
    }
  >('operators-cells', 'declareCellPyramid');
  try {
    const result = declare(device);
    const counts = imports.mock.calls.flatMap(([descriptor, buffer], index) =>
      descriptor.id === 'level-counts' ? [{buffer, graph: imports.mock.contexts[index]}] : []
    );
    expect(counts).toHaveLength(2);
    expect(counts[0].buffer).toBe(counts[1].buffer);
    expect(counts[0].buffer).toBe(result.countsBuffer);
    expect(counts[0].graph).not.toBe(counts[1].graph);
    expect(
      allocations.mock.calls.some(
        ([props]) =>
          'data' in props &&
          props.data instanceof Float32Array &&
          Array.from(props.data).join(',') === '0,0,1,0,2,0'
      )
    ).toBe(true);
    const positionImport = imports.mock.calls.findIndex(
      ([descriptor]) => descriptor.id === 'positions'
    );
    expect(imports.mock.calls[positionImport][1]).toBe(result.positionsBuffer);
    const pyramid = additions.mock.calls
      .map(([producer]) => producer)
      .find(producer => producer instanceof GPUCellPyramid);
    if (!(pyramid instanceof GPUCellPyramid)) throw new Error('Guide did not add its pyramid');
    expect(pyramid.props.positions).toMatchObject({
      buffer: imports.mock.results[positionImport].value
    });
    result.countsBuffer.destroy();
    result.activeLevelBuffer.destroy();
    result.positionsBuffer.destroy();
  } finally {
    imports.mockRestore();
    allocations.mockRestore();
    additions.mockRestore();
  }
});

it('UCT-09 reproduces rejection of a view reused across graph ownership', () => {
  const device = createNullWebGPUDevice();
  const producer = new GPUCommandGraph(device);
  const selection = new GPUCommandGraph(device);
  expect(() =>
    selection.add(
      new GPUCellLevelSelection({
        levelCounts: createTransientView(producer, 'foreign-counts', 'uint32', 5),
        activeLevel: createTransientView(selection, 'active', 'uint32', 1),
        output: {status: {count: createTransientView(selection, 'count', 'uint32', 1)}}
      })
    )
  ).toThrow(/views must belong to the target graph/);
});

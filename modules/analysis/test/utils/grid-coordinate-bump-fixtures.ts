// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUGridIndex, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createEncounterScanNode} from '../../src/gpu-spatial-analysis/trajectory-encounters/encounter-kernels';
import {getGridCandidateNodes} from '../../src/gpu-spatial-analysis/region-statistics/region-statistics-grid';
import {GPUInverseDistanceWeighting} from '../../src/gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting';
import {getGPUInverseDistanceWeightingParameterValues} from '../../src/gpu-spatial-analysis/spatial-interpolation/spatial-interpolation-parameters';
import {createPointDensityGridKeysNode} from '../../src/gpu-spatial-analysis/point-density/point-density-kernels';
import {createFlowZoneNode} from '../../src/gpu-network/flow-aggregation/flow-aggregation-kernels';
import {makeExtentGrid} from '../gpu-spatial-analysis/spatial-interpolation/extent-grid';

export const GRID_COORDINATE_READERS = [
  'encounters',
  'regions',
  'interpolation',
  'density',
  'flow'
] as const;
export type GridCoordinateReader = (typeof GRID_COORDINATE_READERS)[number];

// Expected cells are fixed independently of the CPU reference below. Old index readers rescaled
// every zero-crossing domain; old density/flow readers never protected an overflowing width.
export const GRID_COORDINATE_CASES = [
  {
    name: 'zero-crossing edge',
    value: -177,
    minimum: -180,
    maximum: 180,
    size: 360,
    cell: 3,
    oldIndexCell: 2,
    oldBinningCell: 3
  },
  {
    name: 'positive domain edge',
    value: 103,
    minimum: 100,
    maximum: 460,
    size: 360,
    cell: 3,
    oldIndexCell: 3,
    oldBinningCell: 3
  },
  {
    name: 'interior exact edge',
    value: 0,
    minimum: -180,
    maximum: 180,
    size: 360,
    cell: 180,
    oldIndexCell: 180,
    oldBinningCell: 180
  },
  {
    name: 'inclusive maximum',
    value: 180,
    minimum: -180,
    maximum: 180,
    size: 360,
    cell: 359,
    oldIndexCell: 359,
    oldBinningCell: 359
  },
  {
    // The uploaded bounds are symmetric. Their f32 width overflows, but the exact fraction is 1/2.
    // Interior edges belong to the cell on their right, so floor((1/2) * 360) is 180.
    name: 'overflowing domain width',
    value: 0,
    minimum: -3e38,
    maximum: 3e38,
    size: 360,
    cell: 180,
    oldIndexCell: 180,
    oldBinningCell: 0
  },
  {
    name: 'upload rounds onto edge',
    value: -177.000001,
    minimum: -180,
    maximum: 180,
    size: 360,
    cell: 3,
    oldIndexCell: 2,
    oldBinningCell: 3
  }
] as const;
export type GridCoordinateCase = (typeof GRID_COORDINATE_CASES)[number];

const WIDE_GRID_EDGE_UNIT = 2 ** 122;
const WIDE_GRID_EDGE = 10 * WIDE_GRID_EDGE_UNIT;
const WIDE_GRID_EDGE_ULP = 2 ** 102;

export const WIDE_GRID_COORDINATE_CASES = [
  {
    name: 'asymmetric overflowing domain width',
    value: 0,
    minimum: -3e38,
    maximum: 2e38,
    size: 361,
    // Independent floor(361 * 3 / 5) = 216, away from a grid edge.
    cell: 216
  },
  // Exact rational coordinates from the uploaded float32 values are 13 - 1/5242880,
  // 13, and 13 + 1/5242880. The edge belongs to the cell on its right.
  ...[
    {
      name: 'wide grid one float32 below an edge',
      value: WIDE_GRID_EDGE - WIDE_GRID_EDGE_ULP,
      cell: 12
    },
    {name: 'wide grid exact interior edge', value: WIDE_GRID_EDGE, cell: 13},
    {
      name: 'wide grid one float32 above an edge',
      value: WIDE_GRID_EDGE + WIDE_GRID_EDGE_ULP,
      cell: 13
    }
  ].map(testCase => ({
    ...testCase,
    minimum: -55 * WIDE_GRID_EDGE_UNIT,
    maximum: 55 * WIDE_GRID_EDGE_UNIT,
    size: 22
  }))
] as const;

/** An uploaded float32 expressed exactly in units of the smallest positive subnormal. */
function getFloat32GridUnits(bits: number): bigint {
  const exponent = (bits >>> 23) & 0xff;
  const significand = (bits & 0x7fffff) | (exponent === 0 ? 0 : 0x800000);
  const magnitude = BigInt(significand) << BigInt(Math.max(exponent, 1) - 1);
  return bits >>> 31 ? -magnitude : magnitude;
}

/** Test-only CPU model of density and flow grid arithmetic on uploaded f32 inputs. */
export function getFloat32GridCoordinate(
  value: number,
  minimum: number,
  maximum: number,
  size: number
): number {
  value = Math.fround(value);
  minimum = Math.fround(minimum);
  maximum = Math.fround(maximum);
  if (maximum === minimum || value <= minimum) return 0;
  if (value >= maximum) return size - 1;
  const boundMagnitude = Math.max(Math.abs(minimum), Math.abs(maximum));
  if (boundMagnitude >= 2 ** 125 && Number.isFinite(boundMagnitude)) {
    const bits = new Uint32Array(Float32Array.of(value, minimum, maximum).buffer);
    const offset = getFloat32GridUnits(bits[0]) - getFloat32GridUnits(bits[1]);
    const width = getFloat32GridUnits(bits[2]) - getFloat32GridUnits(bits[1]);
    return Number((offset * BigInt(size)) / width);
  }
  const fraction = Math.fround(Math.fround(value - minimum) / Math.fround(maximum - minimum));
  return Math.min(Math.trunc(Math.fround(fraction * Math.fround(size))), size - 1);
}

/** Extract the public coordinate helper from a reader shader, including its overflow guard. */
export function getGridCoordinateFunction(source: string): string {
  const start = source.indexOf('fn getCoordinate(');
  if (start < 0) throw new Error('Reader shader has no getCoordinate');
  let depth = 0;
  for (let index = source.indexOf('{', start); index < source.length; index++) {
    if (source[index] === '{') depth++;
    if (source[index] === '}' && --depth === 0) return source.slice(start, index + 1);
  }
  throw new Error('Unterminated reader getCoordinate');
}

/** Real reader kernels plus a published index over the same uploaded point(s). No device acquisition. */
export function createGridCoordinateFixture(
  device: Device,
  reader: GridCoordinateReader,
  fixture: GridCoordinateCase
) {
  const graph = new GPUCommandGraph(device, {id: `grid-bump-${reader}`});
  const buffers: Record<string, Buffer> = {};
  function view<Format extends 'float32' | 'float32x2' | 'float32x3' | 'uint32' | 'uint32x2'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array
  ) {
    const buffer = device.createBuffer({
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers[name] = buffer;
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      values.length / getGPUVectorFormatInfo(format).components
    );
  }
  const bounds = [fixture.minimum, 0, fixture.maximum, 1] as const;
  const positions = view('positions', 'float32x2', Float32Array.of(fixture.value, 0.5));
  const encounter = reader === 'encounters';
  const encounterPositions = encounter
    ? view(
        'encounter-positions',
        'float32x3',
        Float32Array.of(fixture.value, 0.5, 0.5, fixture.value, 0.5, 0.5)
      )
    : undefined;
  const cellOffsets = view('cell-offsets', 'uint32', new Uint32Array(fixture.size + 1));
  const objectIds = view('object-ids', 'uint32', new Uint32Array(encounter ? 2 : 1));
  const index = new GPUGridIndex({
    id: 'published-builder',
    positions: encounterPositions ?? positions,
    gridSize: encounter ? [fixture.size, 1, 1] : [fixture.size, 1],
    bounds: encounter ? [fixture.minimum, 0, 0, fixture.maximum, 1, 1] : bounds,
    cellOffsets,
    objectIds,
    count: view('index-count', 'uint32', new Uint32Array(1)),
    overflow: view('index-overflow', 'uint32', new Uint32Array(1))
  });
  graph.add(index);
  const activeGridSize =
    reader === 'flow'
      ? view('active-grid-size', 'uint32', Uint32Array.of(fixture.size, 1))
      : undefined;
  let readerNodes: readonly GPUCommandNode[];
  if (reader === 'encounters') {
    readerNodes = [
      createEncounterScanNode(graph, {
        id: 'reader',
        shape: {
          trackCount: 2,
          bucketCount: 1,
          hitCapacity: 4,
          pairCapacity: 1,
          columns: fixture.size,
          rows: 1,
          bounds,
          cellSize: 1
        },
        positions: encounterPositions!,
        cellOffsets,
        objectIds,
        distance: view('distance', 'float32', Float32Array.of(0)),
        state: view('result', 'uint32', new Uint32Array(1)),
        hitPairs: view('hit-pairs', 'uint32x2', new Uint32Array(8)),
        hitBuckets: view('hit-buckets', 'uint32', new Uint32Array(4)),
        hitDistances: view('hit-distances', 'float32', new Float32Array(4))
      })
    ];
  } else if (reader === 'regions') {
    const candidates = getGridCandidateNodes(graph, {
      id: 'reader',
      positions,
      rowCount: 1,
      region: {
        kind: 'rectangle',
        bounds: view('query', 'float32', Float32Array.of(fixture.value, 0, fixture.value, 1))
      },
      spatialIndex: {kind: 'grid', index, candidateCapacity: 1},
      regionOverflow: view('region-overflow', 'uint32', new Uint32Array(1))
    });
    readerNodes = candidates.nodes;
    // Candidate masks are storage-only scratch: expose the result with a storage read/write pass.
    const result = view('result', 'uint32', new Uint32Array(1));
    readerNodes = [
      ...readerNodes,
      createWGSLKernelNode(graph, {
        id: 'read-candidate-mask',
        operation: 'GridCoordinateBump',
        invocationCount: 1,
        bindings: [
          {name: 'candidateMask', view: candidates.mask, type: 'u32', access: 'read'},
          {name: 'result', view: result, type: 'u32', access: 'read_write'}
        ],
        body: 'result[resultOffset] = candidateMask[candidateMaskOffset];'
      })
    ];
  } else if (reader === 'interpolation') {
    const center = Math.fround(fixture.value);
    const grid = makeExtentGrid(1, 1, [center - 0.5, 0, center + 0.5, 1]);
    const operation = new GPUInverseDistanceWeighting({
      id: 'reader',
      positions,
      values: view('values', 'float32', Float32Array.of(7)),
      grid,
      indexBounds: bounds,
      indexGridSize: [fixture.size, 1],
      parameters: view(
        'parameters',
        'float32',
        getGPUInverseDistanceWeightingParameterValues({searchRadius: 0}, grid)
      ),
      output: {
        values: view('result', 'float32', new Float32Array(1)),
        counts: view('neighbor-count', 'uint32', new Uint32Array(1))
      }
    });
    readerNodes = operation.getCommandNodes(graph);
  } else if (reader === 'density') {
    readerNodes = [
      createPointDensityGridKeysNode(graph, {
        id: 'reader',
        positions,
        bounds,
        gridSize: [fixture.size, 1],
        keyStart: 0,
        keys: view('result', 'uint32', new Uint32Array(1))
      })
    ];
  } else {
    readerNodes = [
      createFlowZoneNode(graph, {
        id: 'reader',
        positions,
        bounds,
        gridSize: [fixture.size + 1, 2],
        activeGridSize,
        kind: 'grid',
        rowBase: view('row-base', 'uint32', Uint32Array.of(0, 1, 0, 1, 1)),
        zones: view('result', 'uint32', new Uint32Array(1))
      })
    ];
  }
  graph.add(readerNodes);
  return {
    graph,
    readerNodes,
    buffers,
    view,
    positions,
    activeGridSize,
    destroy: () => {
      for (const buffer of Object.values(buffers)) buffer.destroy();
    }
  };
}

export const GRID_PROBE_COMPLETED = 0x51a7e001;

/** Probe the public helper with uploaded positions, literal bounds and flow's active dimensions. */
export function createGridCoordinateReaderProbe(
  fixture: ReturnType<typeof createGridCoordinateFixture>,
  testCase: GridCoordinateCase,
  readerSource: string
): GPUCommandNode {
  const output = fixture.view('coordinate', 'uint32', Uint32Array.of(0xffffffff, 0));
  return createWGSLKernelNode(fixture.graph, {
    id: 'probe-reader-coordinate',
    operation: 'GridCoordinateBump',
    invocationCount: 1,
    bindings: [
      {name: 'positions', view: fixture.positions, type: 'f32', access: 'read'},
      {name: 'coordinate', view: output, type: 'u32', access: 'read_write'},
      ...(fixture.activeGridSize
        ? [
            {
              name: 'activeGridSize',
              view: fixture.activeGridSize,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : [])
    ],
    declarations: `${getGridCoordinateFunction(readerSource)}
const DOMAIN_MINIMUM: f32 = ${getWGSLFloatLiteral(testCase.minimum)};
const DOMAIN_MAXIMUM: f32 = ${getWGSLFloatLiteral(testCase.maximum)};
const GRID_COLUMNS: u32 = ${testCase.size}u;`,
    body: `let columns = ${fixture.activeGridSize ? `clamp(activeGridSize[activeGridSizeOffset], 1u, ${testCase.size + 1}u)` : 'GRID_COLUMNS'};
  coordinate[coordinateOffset] = getCoordinate(positions[positionsOffset], DOMAIN_MINIMUM, DOMAIN_MAXIMUM, columns);
  coordinate[coordinateOffset + 1u] = ${GRID_PROBE_COMPLETED}u;`
  });
}

/** Keep the former runtime-bound probe as a separately stamped diagnostic, not a reader surrogate. */
export function createRuntimeGridCoordinateDiagnostic(
  fixture: ReturnType<typeof createGridCoordinateFixture>,
  testCase: GridCoordinateCase,
  readerSource: string
): GPUCommandNode {
  const parameters = fixture.view(
    'coordinate-parameters',
    'float32',
    Float32Array.of(testCase.value, testCase.minimum, testCase.maximum, testCase.size)
  );
  const output = fixture.view(
    'runtime-coordinate',
    'uint32',
    Uint32Array.of(0xffffffff, 0, 0, 0, 0)
  );
  return createWGSLKernelNode(fixture.graph, {
    id: 'diagnose-runtime-grid-coordinate',
    operation: 'GridCoordinateBump',
    invocationCount: 1,
    bindings: [
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'probeResults', view: output, type: 'u32', access: 'read_write'}
    ],
    declarations: getGridCoordinateFunction(readerSource),
    body: `let minimum = parameters[parametersOffset + 1u];
  let maximum = parameters[parametersOffset + 2u];
  let scale = max(abs(minimum), abs(maximum));
  probeResults[probeResultsOffset] = getCoordinate(parameters[parametersOffset], minimum, maximum, u32(parameters[parametersOffset + 3u]));
  probeResults[probeResultsOffset + 1u] = ${GRID_PROBE_COMPLETED}u;
  probeResults[probeResultsOffset + 2u] = bitcast<u32>(minimum / scale);
  probeResults[probeResultsOffset + 3u] = bitcast<u32>(maximum / scale);
  probeResults[probeResultsOffset + 4u] = bitcast<u32>(1.0 / scale);`
  });
}

/** Device agreement is the overflow correctness oracle; the CPU cell is recorded separately. */
export function getGridCoordinateAgreement(
  readerCell: number,
  builderCell: number,
  modelCell: number
) {
  const outcome =
    readerCell !== builderCell
      ? 'reader != builder'
      : builderCell !== modelCell
        ? 'reader == builder != model'
        : 'reader == builder == model';
  return {
    matchesBuilder: readerCell === builderCell,
    outcome,
    message: `${outcome}: reader=${readerCell}, builder=${builderCell}, model=${modelCell}`
  };
}

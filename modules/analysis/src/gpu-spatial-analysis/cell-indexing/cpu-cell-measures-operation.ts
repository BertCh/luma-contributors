// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// Boundary construction follows H3 faceijk.c (Apache-2.0), including substrate face crossings.

import {resolveCellPrecision} from './cell-precision';
import {snapshotCellConstruction} from './cell-construction-snapshot';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getGPUAuthalicSphereRadius} from '../../analysis-core/contracts/spatial-context';
import {getCPUOperationUint32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getCoveringCross,
  getCoveringDot,
  normalizeCoveringVector,
  type CPUCoveringVector
} from '../cell-topology/cpu-s2-covering-geometry';
import {getFaceBasisVectors} from './h3-index-extended-wgsl';
import {getRectangularCellMeasures} from './rectangular-cell-measures';
import {GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT} from './gpu-cell-measures';
import {
  GPUCellMeasuresOperation,
  type CellMeasuresOperationProps
} from './gpu-cell-measures-operation';

/** Options for packed H3, S2 and Quadbin cell measures. */
export type CPUCellMeasuresOptions = Pick<
  CellMeasuresOperationProps,
  'family' | 'wordOrder' | 'maximumEdgeCount' | 'units' | 'precision'
>;
/** Row-aligned measures; edgeLengths has maximumEdgeCount entries per cell. */
export type CPUCellMeasuresResult = {
  areas: Float32Array;
  perimeters: Float32Array;
  edgeLengths: Float32Array;
  edgeCounts: Uint32Array;
  pentagons: Uint32Array;
};

type Coordinates = readonly [number, number, number];
type FaceCoordinates = {face: number; coordinates: Coordinates; overage: number};
const PENTAGONS = new Set([4, 14, 24, 38, 49, 58, 63, 72, 83, 97, 107, 117]);
const CLOCKWISE_DIGITS = [0, 3, 6, 2, 5, 1, 4, 7];
const UNIT_COORDINATES: Coordinates[] = [
  [0, 0, 0],
  [0, 0, 1],
  [0, 1, 0],
  [0, 1, 1],
  [1, 0, 0],
  [1, 0, 1],
  [1, 1, 0]
];
const EVEN_VERTICES: Coordinates[] = [
  [2, 1, 0],
  [1, 2, 0],
  [0, 2, 1],
  [0, 1, 2],
  [1, 0, 2],
  [2, 0, 1]
];
const ODD_VERTICES: Coordinates[] = [
  [5, 4, 0],
  [1, 5, 0],
  [0, 5, 4],
  [0, 1, 5],
  [4, 0, 5],
  [5, 0, 1]
];

// Like parseDggsFaceTable, read the published table with an explicit shape check. These are
// H3 data tables, not implementations of luma.gl APIs.
function readIntegerTable(functionName: string, columns: number, count: number): number[][] {
  const start = dggs.source.indexOf(`fn ${functionName}`);
  const end = start < 0 ? -1 : dggs.source.indexOf(');', start);
  const rows =
    end < 0
      ? []
      : [...dggs.source.slice(start, end).matchAll(/vec[34][iu]\(([^)]*)\)/g)].map(match =>
          match[1].split(',').map(value => Number(value.trim().replace(/u$/, '')))
        );
  if (
    rows.length !== count ||
    rows.some(row => row.length !== columns || row.some(value => !Number.isInteger(value)))
  )
    throw new Error(
      `cell-measures dggs table ${functionName} must contain ${count} rows of ${columns} integers`
    );
  return rows;
}
const BASE_HOMES = readIntegerTable('dggs_h3_get_base_cell_home', 4, 122);
const NEIGHBOR_FACES = readIntegerTable('dggs_h3_get_face_neighbor', 3, 20);
const FACE_BASIS_VECTORS = getFaceBasisVectors();
const FACE_BASES = Array.from({length: 20}, (_, face) =>
  FACE_BASIS_VECTORS.slice(3 * face, 3 * face + 3).map(vector => vector as unknown as Coordinates)
);

/**
 * Decodes exact integer H3, S2 and Quadbin keys and measures their boundaries in float64, rounding only
 * the returned rows. Invalid keys produce zeros; edgeCounts is never truncated by the stride.
 *
 * What differs from the GPU contributor:
 * - Every precision option computes in binary64 here; 'double-single' selects hi/lo float32 arithmetic only on the GPU.
 * - All boundaries use binary64 face projection, including pentagons and face crossings at
 *   resolutions 11–15 where the GPU's float32 fallback is unreliable. Near-zero lengths and
 *   areas still have binary64 coordinate error; no float32 intermediate is emulated.
 * - S2 uses binary64 unit-vector differences rather than the GPU local-increment evaluation.
 * - Empty cells arrays are accepted by the pure function; graph operations require rowCount >= 1.
 * - Nothing differs in invalid-key handling, units, boundary order, padding or stride truncation.
 *
 * @throws Error naming cells if it does not contain packed uint32x2 rows.
 */
export function computeCellMeasuresOnCPU(
  cells: Uint32Array,
  options: CPUCellMeasuresOptions = {family: 'h3'}
): CPUCellMeasuresResult {
  validateOptions('cell-measures', options);
  if (cells.length % 2) throw new Error('cells must contain uint32x2 rows');
  const rowCount = cells.length / 2;
  const stride = options.maximumEdgeCount ?? (options.family === 'h3' ? 6 : 4);
  const result: CPUCellMeasuresResult = {
    areas: new Float32Array(rowCount),
    perimeters: new Float32Array(rowCount),
    edgeLengths: new Float32Array(rowCount * stride),
    edgeCounts: new Uint32Array(rowCount),
    pentagons: new Uint32Array(rowCount)
  };
  const radius = getGPUAuthalicSphereRadius({
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: options.units ?? 'meters'
  });
  for (let row = 0; row < rowCount; row++) {
    const high = cells[row * 2 + Number(options.wordOrder !== 'high-low')];
    const low = cells[row * 2 + Number(options.wordOrder === 'high-low')];
    const key = (BigInt(high) << 32n) | BigInt(low);
    if (options.family !== 'h3') {
      const measured = getRectangularCellMeasures(key, options.family);
      if (measured) {
        result.areas[row] = measured.area * radius * radius;
        result.perimeters[row] = measured.edges.reduce((sum, edge) => sum + edge, 0) * radius;
        result.edgeCounts[row] = 4;
        measured.edges.forEach((edge, index) => {
          result.edgeLengths[row * stride + index] = edge * radius;
        });
      }
      continue;
    }
    const resolution = (high >>> 20) & 15;
    const baseCell = (high >>> 13) & 127;
    const digits = Array.from({length: 15}, (_, index) =>
      Number((key >> BigInt(42 - 3 * index)) & 7n)
    );
    const leading = digits.slice(0, resolution).find(digit => digit !== 0) ?? 0;
    if (
      high >>> 24 !== 8 ||
      baseCell > 121 ||
      digits.some((digit, index) => (index < resolution ? digit === 7 : digit !== 7)) ||
      (PENTAGONS.has(baseCell) && leading === 1)
    )
      continue;
    const pentagon = PENTAGONS.has(baseCell) && leading === 0;
    const center = decodeCenter(baseCell, resolution, digits, leading);
    const boundary = getBoundary(center, resolution, pentagon);
    result.edgeCounts[row] = boundary.length;
    result.pentagons[row] = Number(pentagon);
    const centerPoint = projectPoint(toPlane(center.coordinates), center.face, resolution, false);
    let area = 0;
    let perimeter = 0;
    for (let edge = 0; edge < boundary.length; edge++) {
      const first = boundary[edge];
      const second = boundary[(edge + 1) % boundary.length];
      // Difference vectors prevent cancellation in the tiny triangle determinant.
      const determinant = Math.abs(
        getCoveringDot(
          centerPoint,
          getCoveringCross(subtract(first, centerPoint), subtract(second, centerPoint))
        )
      );
      area +=
        2 *
        Math.atan2(
          determinant,
          1 +
            getCoveringDot(first, second) +
            getCoveringDot(second, centerPoint) +
            getCoveringDot(centerPoint, first)
        );
      const length =
        Math.atan2(
          Math.hypot(...getCoveringCross(first, subtract(second, first))),
          getCoveringDot(first, second)
        ) * radius;
      perimeter += length;
      if (edge < stride) result.edgeLengths[row * stride + edge] = length;
    }
    result.areas[row] = area * radius * radius;
    result.perimeters[row] = perimeter;
  }
  return result;
}

/** CPU twin with the contributor's exact vector output layouts and optional ports. */
export class CPUCellMeasuresOperation extends GPUCPUOperation {
  constructor(props: CellMeasuresOperationProps) {
    props = snapshotCellConstruction(props, [
      'id',
      'family',
      'cells',
      'wordOrder',
      'maximumEdgeCount',
      'units',
      'precision',
      'rowCount',
      ['output', ['areas', 'perimeters', 'edgeLengths', 'edgeCounts', 'pentagons']]
    ]);
    const operation = new GPUCellMeasuresOperation(props);
    const {id} = operation;
    validateCapacity(id, 'rowCount', props.rowCount);
    validateOptions(id, props);
    validateRequirement(
      id,
      Object.values(props.output).some(value => value !== undefined),
      'output must contain at least one output'
    );
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const cells = getCPUOperationUint32Rows(id, inputs, 'cells', 'uint32x2');
        if (cells.length !== props.rowCount * 2)
          throw new Error(`${id} cells length must equal rowCount * 2`);
        const result = computeCellMeasuresOnCPU(cells, props);
        return Object.fromEntries(
          (Object.keys(operation.outputs) as (keyof CPUCellMeasuresResult)[]).map(port => [
            props.output[port]!,
            result[port]
          ])
        );
      }
    });
  }
}

function validateOptions(id: string, options: CPUCellMeasuresOptions): void {
  validateRequirement(
    id,
    ['h3', 's2', 'quadbin'].includes(options.family),
    "family must be 'h3', 's2' or 'quadbin'"
  );
  validateRequirement(
    id,
    options.wordOrder === undefined || ['little-endian', 'high-low'].includes(options.wordOrder),
    "wordOrder must be 'little-endian' or 'high-low'"
  );
  validateRequirement(
    id,
    options.units === undefined || ['meters', 'kilometers'].includes(options.units),
    "units must be 'meters' or 'kilometers'"
  );
  resolveCellPrecision(id, options.precision);
  validateCapacity(
    id,
    'maximumEdgeCount',
    options.maximumEdgeCount ?? (options.family === 'h3' ? 6 : 4),
    options.family === 'h3' ? 6 : 4,
    GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT
  );
}
function normalizeCoordinates(coordinates: Coordinates): Coordinates {
  const minimum = Math.min(...coordinates);
  return coordinates.map(value => value - minimum) as unknown as Coordinates;
}
function subtract(first: Coordinates, second: Coordinates): Coordinates {
  return first.map((value, axis) => value - second[axis]) as unknown as Coordinates;
}
function add(first: Coordinates, second: Coordinates): Coordinates {
  return first.map((value, axis) => value + second[axis]) as unknown as Coordinates;
}
function transform(coordinates: Coordinates, basis: Coordinates[]): Coordinates {
  return normalizeCoordinates(
    [0, 1, 2].map(axis =>
      coordinates.reduce((sum, value, index) => sum + value * basis[index][axis], 0)
    ) as unknown as Coordinates
  );
}
function rotateCoordinates(coordinates: Coordinates, clockwise = false): Coordinates {
  return transform(
    coordinates,
    clockwise
      ? [
          [1, 0, 1],
          [1, 1, 0],
          [0, 1, 1]
        ]
      : [
          [1, 1, 0],
          [0, 1, 1],
          [1, 0, 1]
        ]
  );
}
function descendCoordinates(
  coordinates: Coordinates,
  aperture: number,
  clockwise = false
): Coordinates {
  const scale = aperture === 3 ? 2 : 3;
  return transform(
    coordinates,
    clockwise
      ? [
          [scale, 1, 0],
          [0, scale, 1],
          [1, 0, scale]
        ]
      : [
          [scale, 0, 1],
          [1, scale, 0],
          [0, 1, scale]
        ]
  );
}
function getNeighbor(
  face: number,
  quadrant: number
): {face: number; translation: Coordinates; rotations: number} {
  const outer = face < 5 || face >= 15;
  return {
    face: NEIGHBOR_FACES[face][quadrant - 1],
    translation: quadrant === 3 ? [0, 2, 2] : (quadrant === 1) === outer ? [2, 0, 2] : [2, 2, 0],
    rotations: outer && quadrant !== 3 ? (quadrant === 1 ? 1 : 5) : 3
  };
}
function adjustOverage(
  input: FaceCoordinates,
  resolution: number,
  substrate: boolean,
  pentagonLeadingFour = false
): FaceCoordinates {
  let {face, coordinates} = input;
  const dimension = 2 * 7 ** (resolution / 2) * (substrate ? 3 : 1);
  const sum = coordinates.reduce((total, value) => total + value, 0);
  if (sum <= dimension) return {face, coordinates, overage: substrate && sum === dimension ? 1 : 0};
  const quadrant = coordinates[2] > 0 ? (coordinates[1] > 0 ? 3 : 2) : 1;
  if (quadrant === 2 && pentagonLeadingFour) {
    const origin: Coordinates = [dimension, 0, 0];
    coordinates = add(rotateCoordinates(subtract(coordinates, origin), true), origin);
  }
  const neighbor = getNeighbor(face, quadrant);
  face = neighbor.face;
  for (let rotation = 0; rotation < neighbor.rotations; rotation++)
    coordinates = rotateCoordinates(coordinates);
  coordinates = normalizeCoordinates(
    add(
      coordinates,
      neighbor.translation.map(value => (value * dimension) / 2) as unknown as Coordinates
    )
  );
  return {
    face,
    coordinates,
    overage:
      substrate && coordinates.reduce((total, value) => total + value, 0) === dimension ? 1 : 2
  };
}
function decodeCenter(
  baseCell: number,
  resolution: number,
  digits: number[],
  leading: number
): FaceCoordinates {
  const home = BASE_HOMES[baseCell];
  let center: FaceCoordinates = {
    face: home[0],
    coordinates: home.slice(1) as unknown as Coordinates,
    overage: 0
  };
  for (let level = 1; level <= resolution; level++) {
    const digit =
      PENTAGONS.has(baseCell) && leading === 5
        ? CLOCKWISE_DIGITS[digits[level - 1]]
        : digits[level - 1];
    center.coordinates = normalizeCoordinates(
      add(descendCoordinates(center.coordinates, 7, level % 2 === 0), UNIT_COORDINATES[digit])
    );
  }
  const original = center.coordinates;
  const adjustedResolution = resolution + (resolution % 2);
  if (resolution % 2) center.coordinates = descendCoordinates(center.coordinates, 7, true);
  let adjusted = adjustOverage(
    center,
    adjustedResolution,
    false,
    PENTAGONS.has(baseCell) && leading === 4
  );
  if (adjusted.overage) {
    center = adjusted;
    if (PENTAGONS.has(baseCell)) {
      do {
        adjusted = adjustOverage(center, adjustedResolution, false);
        center = adjusted;
      } while (adjusted.overage);
    }
    if (resolution % 2) {
      const first = center.coordinates[0] - center.coordinates[2];
      const second = center.coordinates[1] - center.coordinates[2];
      const divide = (value: number) => Math.trunc((value + (value >= 0 ? 3 : -3)) / 7);
      center.coordinates = normalizeCoordinates([
        divide(2 * first + second),
        divide(3 * second - first),
        0
      ]);
    }
  } else if (resolution % 2) center.coordinates = original;
  return center;
}
function toPlane(coordinates: Coordinates): [number, number] {
  const first = coordinates[0] - coordinates[2];
  const second = coordinates[1] - coordinates[2];
  return [first - second / 2, (second * Math.sqrt(3)) / 2];
}
function projectPoint(
  point: readonly [number, number],
  face: number,
  resolution: number,
  substrate: boolean
): CPUCoveringVector {
  let [horizontal, vertical] = point;
  if (!substrate && resolution % 2) {
    const angle = 0.3334731722518321;
    [horizontal, vertical] = [
      Math.cos(angle) * horizontal - Math.sin(angle) * vertical,
      Math.sin(angle) * horizontal + Math.cos(angle) * vertical
    ];
  }
  const scale = 0.381966011250105 / Math.sqrt(7) ** resolution / (substrate ? 3 : 1);
  return normalizeCoveringVector(
    FACE_BASES[face][0].map(
      (value, axis) =>
        value +
        scale * (horizontal * FACE_BASES[face][1][axis] + vertical * FACE_BASES[face][2][axis])
    ) as unknown as Coordinates
  );
}
function intersectFaceEdge(
  first: [number, number],
  second: [number, number],
  resolution: number,
  quadrant: number
): [number, number] {
  const dimension = 2 * 7 ** (resolution / 2);
  const vertices = [
    [3 * dimension, 0],
    [-1.5 * dimension, ((3 * Math.sqrt(3)) / 2) * dimension],
    [-1.5 * dimension, ((-3 * Math.sqrt(3)) / 2) * dimension]
  ];
  const edge = quadrant === 1 ? 0 : quadrant === 3 ? 1 : 2;
  const start = vertices[edge];
  const end = vertices[(edge + 1) % 3];
  const horizontal = second[0] - first[0];
  const vertical = second[1] - first[1];
  const edgeHorizontal = end[0] - start[0];
  const edgeVertical = end[1] - start[1];
  const fraction =
    (edgeHorizontal * (first[1] - start[1]) - edgeVertical * (first[0] - start[0])) /
    (-edgeHorizontal * vertical + horizontal * edgeVertical);
  return [first[0] + fraction * horizontal, first[1] + fraction * vertical];
}
function getBoundary(
  center: FaceCoordinates,
  resolution: number,
  pentagon: boolean
): CPUCoveringVector[] {
  const odd = resolution % 2 !== 0;
  const adjustedResolution = resolution + Number(odd);
  let coordinates = descendCoordinates(descendCoordinates(center.coordinates, 3), 3, true);
  if (odd) coordinates = descendCoordinates(coordinates, 7, true);
  const vertices = (odd ? ODD_VERTICES : EVEN_VERTICES)
    .slice(0, pentagon ? 5 : 6)
    .map(offset => normalizeCoordinates(add(coordinates, offset)));
  const boundary: CPUCoveringVector[] = [];
  let previous: FaceCoordinates | undefined;
  for (let step = 0; step <= vertices.length; step++) {
    const vertex = step % vertices.length;
    let current = adjustOverage(
      {face: center.face, coordinates: vertices[vertex], overage: 0},
      adjustedResolution,
      true
    );
    if (pentagon)
      while (current.overage === 2) current = adjustOverage(current, adjustedResolution, true);
    const direction = (from: number, to: number) =>
      [1, 2, 3].find(quadrant => getNeighbor(from, quadrant).face === to) ?? 0;
    if (odd && previous) {
      if (pentagon) {
        const currentDirection = direction(current.face, previous.face);
        const previousDirection = direction(previous.face, current.face);
        if (currentDirection && previousDirection) {
          const neighbor = getNeighbor(current.face, currentDirection);
          let transformed = current.coordinates;
          for (let rotation = 0; rotation < neighbor.rotations; rotation++)
            transformed = rotateCoordinates(transformed);
          transformed = normalizeCoordinates(
            add(
              transformed,
              neighbor.translation.map(
                value => value * 3 * 7 ** (adjustedResolution / 2)
              ) as unknown as Coordinates
            )
          );
          boundary.push(
            projectPoint(
              intersectFaceEdge(
                toPlane(previous.coordinates),
                toPlane(transformed),
                adjustedResolution,
                previousDirection
              ),
              previous.face,
              adjustedResolution,
              true
            )
          );
        }
      } else if (current.face !== previous.face && previous.overage !== 1) {
        const quadrant = direction(
          center.face,
          previous.face === center.face ? current.face : previous.face
        );
        const lastVertex = (vertex + 5) % 6;
        const onEdge = (point: Coordinates) =>
          point.reduce((sum, value) => sum + value, 0) === 6 * 7 ** (adjustedResolution / 2);
        if (quadrant && !onEdge(vertices[vertex]) && !onEdge(vertices[lastVertex]))
          boundary.push(
            projectPoint(
              intersectFaceEdge(
                toPlane(vertices[lastVertex]),
                toPlane(vertices[vertex]),
                adjustedResolution,
                quadrant
              ),
              center.face,
              adjustedResolution,
              true
            )
          );
      }
    }
    if (step < vertices.length)
      boundary.push(
        projectPoint(toPlane(current.coordinates), current.face, adjustedResolution, true)
      );
    previous = current;
  }
  return boundary;
}

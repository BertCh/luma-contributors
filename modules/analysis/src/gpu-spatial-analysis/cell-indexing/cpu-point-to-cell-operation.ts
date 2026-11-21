// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// H3 hex quantization and digit rotations derive from Uber's Apache-2.0 H3.

import {resolveCellPrecision} from './cell-precision';
import {resolveCellLongitudePolicy, wrapCellLongitudeDegrees} from './cell-longitude-policy';
import {snapshotCellConstruction} from './cell-construction-snapshot';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {getCellKeyLayout, QUADBIN_FIXED_POINT_CONSTANTS} from '../cell-aggregation/cell-keys';
import {GPU_CELL_INDEX_RESOLUTION_RANGES, type GPUCellIndexFamily} from './cell-index-families';
import {S2_POSITION_TO_IJ, S2_POSITION_TO_ORIENTATION} from './cell-index-wgsl';
import {getFaceBasisVectors} from './h3-index-extended-wgsl';
import {
  CELL_INDEX_H3_BASE_CELL_PACKED_TABLE,
  CELL_INDEX_H3_CW_OFFSET_PENTAGON_FACES
} from './h3-index-tables';
import {
  GPUPointToCellOperation,
  type PointToCellOperationProps
} from './gpu-point-to-cell-operation';

/** Options of the pure point-to-cell computation. */
export type CPUPointToCellOptions = Pick<
  PointToCellOperationProps,
  'family' | 'resolution' | 'precision' | 'longitudePolicy' | 'spatialContext'
>;

/** Little-endian key words and row-aligned validity (zero keys are invalid). */
export type CPUPointToCellResult = {cells: Uint32Array; validity: Uint32Array};

/**
 * Converts packed float32 longitude/latitude rows to the contributor's global grid keys.
 *
 * What differs from the GPU contributor:
 * - S2 uses float64 trigonometry and projection, so keys near cell edges can differ, especially
 *   at levels 26–30. Exact quadrant reduction preserves poles and equivalent ±180 longitudes.
 * - H3 uses binary64 projection for every precision option; it does not emulate float32 or
 *   double-float intermediates. Cell-edge decisions can differ from either GPU precision.
 * - Nothing differs for Quadbin, quadkey or geohash keys: their integer-defined cell methods
 *   are retained. Quadbin/quadkey fixed-point arithmetic uses exact BigInt intermediates.
 * - Empty arrays and zero-row named graph operations return empty cells and validity arrays.
 * - Non-finite rows produce zero keys. With family-default, H3 also returns zero for magnitudes at least 3e38,
 *   matching its kernel guard. Other finite S2/H3 longitudes are reduced modulo 360;
 *   S2 clamps latitude to ±90, H3 treats it as a periodic angle. Subnormals are retained.
 *
 * @throws Error naming positions or mask when packed array lengths are inconsistent.
 */
export function computePointToCellOnCPU(
  positions: Float32Array,
  options: CPUPointToCellOptions,
  mask?: Uint32Array
): CPUPointToCellResult {
  validateOptions('point-to-cell', options);
  if (positions.length % 2 !== 0) throw new Error('positions must hold float32x2 rows');
  const rowCount = positions.length / 2;
  if (mask !== undefined && mask.length !== rowCount)
    throw new Error('mask length must equal the positions row count');
  const cells = new Uint32Array(rowCount * 2);
  const validity = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    const longitude =
      options.longitudePolicy === 'wrap'
        ? wrapCellLongitudeDegrees(positions[row * 2])
        : positions[row * 2];
    const latitude = positions[row * 2 + 1];
    if ((mask && mask[row] === 0) || !Number.isFinite(longitude) || !Number.isFinite(latitude))
      continue;
    let key: bigint;
    switch (options.family) {
      case 'h3':
        key = getH3Key(longitude, latitude, options.resolution);
        break;
      case 's2':
        key = getS2Key(longitude, latitude, options.resolution);
        break;
      case 'geohash':
        key = getGeohashKey(longitude, latitude, options.resolution);
        break;
      default: {
        const resolution = options.resolution;
        const column = getTileColumn(longitude, resolution);
        const tileRow = getTileRow(latitude, resolution);
        let path = 0n;
        for (let bit = 0; bit < resolution; bit++) {
          path |= BigInt((column >>> bit) & 1) << BigInt(2 * bit);
          path |= BigInt((tileRow >>> bit) & 1) << BigInt(2 * bit + 1);
        }
        if (options.family === 'quadkey') key = (BigInt(resolution) << 58n) | path;
        else {
          const layout = getCellKeyLayout('quadbin', resolution);
          const shift = BigInt(layout.lowBit);
          key =
            (BigInt(layout.headerHigh) << 32n) |
            (BigInt(resolution) << 52n) |
            (path << shift) |
            ((1n << shift) - 1n);
        }
      }
    }
    cells[row * 2] = Number(key & 0xffffffffn);
    cells[row * 2 + 1] = Number(key >> 32n);
    validity[row] = Number(key !== 0n);
  }
  return {cells, validity};
}

/** CPU twin with the same vector ports and packed keys as GPUPointToCellOperation. */
export class CPUPointToCellOperation extends GPUCPUOperation {
  constructor(props: PointToCellOperationProps) {
    props = snapshotCellConstruction(props, [
      'id',
      'family',
      'resolution',
      'precision',
      'longitudePolicy',
      'spatialContext',
      'positions',
      'mask',
      'rowCount',
      ['output', ['cells', 'validity']]
    ]);
    const operation = new GPUPointToCellOperation(props);
    const {id} = operation;
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateOptions(id, props);
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
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must equal rowCount * 2`);
        const result = computePointToCellOnCPU(
          positions,
          props,
          props.mask === undefined ? undefined : getCPUOperationUint32Rows(id, inputs, 'mask')
        );
        return {
          [props.output.cells]: result.cells,
          ...(props.output.validity === undefined ? {} : {[props.output.validity]: result.validity})
        };
      }
    });
  }
}

function validateOptions(id: string, options: CPUPointToCellOptions): void {
  resolveCellPrecision(id, options.precision, options.family === 'h3');
  resolveCellLongitudePolicy(id, options.longitudePolicy);
  validateRequirement(
    id,
    Object.hasOwn(GPU_CELL_INDEX_RESOLUTION_RANGES, options.family),
    'family must be quadbin, h3, quadkey, geohash or s2'
  );
  const range = GPU_CELL_INDEX_RESOLUTION_RANGES[options.family as GPUCellIndexFamily];
  validateCapacity(id, 'resolution', options.resolution, range.minimum, range.maximum);
  resolveGPUSpatialContext(
    id,
    options.spatialContext ?? GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
    GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT
  );
}

function getTileColumn(longitude: number, resolution: number): number {
  if (resolution === 0) return 0;
  const clipped = Math.max(-180, Math.min(180, longitude));
  const offset = (Math.abs(clipped) * 2 ** resolution) / 360;
  // Split at the exact half tile count, preserving tiny negative offsets near longitude zero.
  const column = 2 ** (resolution - 1) + (clipped < 0 ? -Math.ceil(offset) : Math.floor(offset));
  return column % 2 ** resolution;
}

function getTileRow(latitude: number, resolution: number): number {
  if (resolution === 0) return 0;
  const half = 2 ** (resolution - 1);
  if (latitude === 0) return half;
  // Multiplication of a clipped float32 latitude by 2^25 is exact in float64. Taking its
  // floor gives the same Q.25 integer as quadbinGetLatitudeFixed's exponent/mantissa shifts.
  const latitudeFixed = BigInt(
    Math.floor(Math.min(Math.abs(latitude), QUADBIN_MAXIMUM_LATITUDE) * 2 ** 25)
  );
  const colatitude = BigInt(QUADBIN_FIXED_POINT_CONSTANTS.ninetyDegrees) - latitudeFixed;
  const angle =
    ((colatitude * BigInt(QUADBIN_FIXED_POINT_CONSTANTS.halfColatitudeScale)) >> 31n) &
    QUADBIN_FIXED_WORD_MASK;
  const squaredAngle = getQuadbinFixedHighProduct(angle, angle);
  const sineBracket = getQuadbinFixedTrigonometricBracket(
    squaredAngle,
    QUADBIN_FIXED_POINT_CONSTANTS.sinCoefficients
  );
  const cosineBracket = getQuadbinFixedTrigonometricBracket(
    squaredAngle,
    QUADBIN_FIXED_POINT_CONSTANTS.cosCoefficients
  );
  const sine =
    (angle -
      getQuadbinFixedHighProduct(angle, getQuadbinFixedHighProduct(squaredAngle, sineBracket))) &
    QUADBIN_FIXED_WORD_MASK;
  const cosine = -getQuadbinFixedHighProduct(squaredAngle, cosineBracket) & QUADBIN_FIXED_WORD_MASK;
  const sineLogarithm = getQuadbinFixedNegativeLogarithm(sine);
  const cosineLogarithm = getQuadbinFixedNegativeLogarithm(cosine);
  const ordinate = sineLogarithm > cosineLogarithm ? sineLogarithm - cosineLogarithm : 0n;
  const fraction = getQuadbinFixedHighProduct(
    (ordinate >> 2n) & QUADBIN_FIXED_WORD_MASK,
    BigInt(QUADBIN_FIXED_POINT_CONSTANTS.inverseTwoPi)
  );
  const fractionBits = BigInt(32 - resolution);
  const whole = Number(fraction >> fractionBits);
  const hasFraction = (fraction & ((1n << fractionBits) - 1n)) !== 0n;
  const row = latitude < 0 ? half + whole : half - Math.max(whole + Number(hasFraction), 1);
  return Math.max(0, Math.min(2 ** resolution - 1, row));
}

const QUADBIN_FIXED_WORD_MASK = 0xffffffffn;
const QUADBIN_MAXIMUM_LATITUDE = new Float32Array(
  Uint32Array.of(QUADBIN_FIXED_POINT_CONSTANTS.maximumLatitudeBits).buffer
)[0];

/** Exact upper word of a product of two unsigned fixed-point words. */
function getQuadbinFixedHighProduct(first: bigint, second: bigint): bigint {
  return (first * second) >> 32n;
}

function getQuadbinFixedTrigonometricBracket(
  squaredAngle: bigint,
  coefficients: readonly number[]
): bigint {
  let bracket = BigInt(coefficients[0]);
  for (let index = 1; index < coefficients.length; index++) {
    bracket =
      (BigInt(coefficients[index]) - getQuadbinFixedHighProduct(squaredAngle, bracket)) &
      QUADBIN_FIXED_WORD_MASK;
  }
  return bracket;
}

/** The kernel's normalized atanh series, returning -ln(value / 2^32) in Q.32. */
function getQuadbinFixedNegativeLogarithm(value: bigint): bigint {
  const leadingZeros = BigInt(Math.clz32(Number(value)));
  const normalized = (value << leadingZeros) & QUADBIN_FIXED_WORD_MASK;
  const difference = -normalized & QUADBIN_FIXED_WORD_MASK;
  const argument = (difference << 31n) / (0x80000000n + (normalized >> 1n));
  const squaredArgument = getQuadbinFixedHighProduct(argument, argument);
  const coefficients = QUADBIN_FIXED_POINT_CONSTANTS.atanhCoefficients;
  let bracket = BigInt(coefficients[0]);
  for (let index = 1; index < coefficients.length; index++) {
    bracket =
      (BigInt(coefficients[index]) + getQuadbinFixedHighProduct(squaredArgument, bracket)) &
      QUADBIN_FIXED_WORD_MASK;
  }
  const inverseHyperbolicTangent =
    (argument +
      getQuadbinFixedHighProduct(argument, getQuadbinFixedHighProduct(squaredArgument, bracket))) &
    QUADBIN_FIXED_WORD_MASK;
  return 2n * inverseHyperbolicTangent + leadingZeros * BigInt(QUADBIN_FIXED_POINT_CONSTANTS.ln2);
}

function getGeohashKey(longitude: number, latitude: number, length: number): bigint {
  const bounds = [
    [-180, 180],
    [-90, 90]
  ];
  const coordinates = [longitude, latitude];
  let key = 0n;
  for (let bit = 0; bit < length * 5; bit++) {
    const axis = bit % 2;
    const midpoint = (bounds[axis][0] + bounds[axis][1]) / 2;
    const upper = coordinates[axis] >= midpoint;
    bounds[axis][upper ? 0 : 1] = midpoint;
    key = (key << 1n) | BigInt(upper);
  }
  return (BigInt(length) << 60n) | key;
}

/** Exact quadrant reduction before float64 trig, including zeros at poles and half turns. */
function getSinCosDegrees(degrees: number): [number, number] {
  const wrapped = degrees % 360;
  const turns = wrapped / 90;
  const lower = Math.floor(turns);
  const fraction = turns - lower;
  const quadrant = lower + Number(fraction > 0.5 || (fraction === 0.5 && lower % 2 !== 0));
  const reduced = wrapped - quadrant * 90;
  const angle = (reduced * Math.PI) / 180;
  // Preserve exact symmetry at face ties rather than letting sin(pi/4) and cos(pi/4)
  // differ by one float64 ulp before the contributor's strict largest-component test.
  const sine = Math.abs(reduced) === 45 ? Math.sign(reduced) * Math.SQRT1_2 : Math.sin(angle);
  const cosine = Math.abs(reduced) === 45 ? Math.SQRT1_2 : Math.cos(angle);
  switch (((quadrant % 4) + 4) % 4) {
    case 1:
      return [cosine, -sine];
    case 2:
      return [-sine, -cosine];
    case 3:
      return [-cosine, sine];
    default:
      return [sine, cosine];
  }
}

function getS2Key(longitude: number, latitude: number, level: number): bigint {
  const [longitudeSine, longitudeCosine] = getSinCosDegrees(longitude);
  const [latitudeSine, latitudeCosine] = getSinCosDegrees(Math.max(-90, Math.min(90, latitude)));
  const point = [latitudeCosine * longitudeCosine, latitudeCosine * longitudeSine, latitudeSine];
  const magnitude = point.map(Math.abs);
  const axis =
    magnitude[0] > magnitude[1]
      ? magnitude[0] > magnitude[2]
        ? 0
        : 2
      : magnitude[1] > magnitude[2]
        ? 1
        : 2;
  const face = axis + (point[axis] < 0 ? 3 : 0);
  const [first, second, third] = point;
  const coordinates = [
    [second / first, third / first],
    [-first / second, third / second],
    [-first / third, -second / third],
    [third / first, second / first],
    [third / second, -first / second],
    [-second / third, -first / third]
  ][face].map(value => {
    const projected =
      value >= 0 ? 0.5 * Math.sqrt(1 + 3 * value) : 1 - 0.5 * Math.sqrt(1 - 3 * value);
    return Math.max(0, Math.min(2 ** 30 - 1, Math.floor(projected * 2 ** 30)));
  });
  let orientation = face & 1;
  let key = (BigInt(face) << 61n) | (1n << BigInt(60 - 2 * level));
  for (let step = 0; step < level; step++) {
    const bit = 29 - step;
    const coordinate = (((coordinates[0] >>> bit) & 1) << 1) | ((coordinates[1] >>> bit) & 1);
    const position = S2_POSITION_TO_IJ[orientation].indexOf(coordinate);
    orientation ^= S2_POSITION_TO_ORIENTATION[position];
    key |= BigInt(position) << BigInt(59 - 2 * step);
  }
  return key;
}

type CoordinateTriple = [number, number, number];
const H3_PENTAGONS = new Set([4, 14, 24, 38, 49, 58, 63, 72, 83, 97, 107, 117]);
const H3_ROTATE_COUNTERCLOCKWISE = [0, 5, 3, 1, 6, 4, 2, 7];
const H3_ROTATE_CLOCKWISE = [0, 3, 6, 2, 5, 1, 4, 7];
const H3_FACE_BASES = getH3FaceBases();

function getH3FaceBases(): number[][][] {
  const bases = getFaceBasisVectors();
  return Array.from({length: 20}, (_, face) => bases.slice(3 * face, 3 * face + 3));
}

function normalizeCoordinates(coordinates: CoordinateTriple): CoordinateTriple {
  const minimum = Math.min(...coordinates);
  return coordinates.map(value => value - minimum) as CoordinateTriple;
}

function getHexCoordinates(horizontal: number, vertical: number): CoordinateTriple {
  const second = Math.abs(vertical) / (Math.sqrt(3) / 2);
  const first = Math.abs(horizontal) + second / 2;
  const firstInteger = Math.trunc(first);
  const secondInteger = Math.trunc(second);
  const firstFraction = first - firstInteger;
  const secondFraction = second - secondInteger;
  let firstCoordinate: number;
  let secondCoordinate: number;
  if (firstFraction < 0.5) {
    if (firstFraction < 1 / 3) {
      firstCoordinate = firstInteger;
      secondCoordinate =
        secondFraction < (1 + firstFraction) / 2 ? secondInteger : secondInteger + 1;
    } else {
      secondCoordinate = secondFraction < 1 - firstFraction ? secondInteger : secondInteger + 1;
      firstCoordinate =
        1 - firstFraction <= secondFraction && secondFraction < 2 * firstFraction
          ? firstInteger + 1
          : firstInteger;
    }
  } else if (firstFraction < 2 / 3) {
    secondCoordinate = secondFraction < 1 - firstFraction ? secondInteger : secondInteger + 1;
    firstCoordinate =
      2 * firstFraction - 1 < secondFraction && secondFraction < 1 - firstFraction
        ? firstInteger
        : firstInteger + 1;
  } else {
    firstCoordinate = firstInteger + 1;
    secondCoordinate = secondFraction < firstFraction / 2 ? secondInteger : secondInteger + 1;
  }
  if (horizontal < 0) {
    const axis = Math.ceil(secondCoordinate / 2);
    firstCoordinate -= 2 * (firstCoordinate - axis) + (secondCoordinate % 2);
  }
  if (vertical < 0) {
    firstCoordinate -= Math.trunc((2 * secondCoordinate + 1) / 2);
    secondCoordinate = -secondCoordinate;
  }
  return normalizeCoordinates([firstCoordinate, secondCoordinate, 0]);
}

function getH3Key(longitude: number, latitude: number, resolution: number): bigint {
  if (Math.abs(longitude) >= 3e38 || Math.abs(latitude) >= 3e38) return 0n;
  const [longitudeSine, longitudeCosine] = getSinCosDegrees(longitude);
  const [latitudeSine, latitudeCosine] = getSinCosDegrees(latitude);
  const point = [latitudeCosine * longitudeCosine, latitudeCosine * longitudeSine, latitudeSine];
  const getDot = (vector: number[]) =>
    point.reduce((sum, value, axis) => sum + value * vector[axis], 0);
  let face = 0;
  let alignment = -2;
  for (let candidate = 0; candidate < 20; candidate++) {
    const candidateAlignment = getDot(H3_FACE_BASES[candidate][0]);
    if (candidateAlignment > alignment) {
      face = candidate;
      alignment = candidateAlignment;
    }
  }
  const scale = Math.sqrt(7) ** resolution / (alignment * 0.381966011250105);
  let horizontal = getDot(H3_FACE_BASES[face][1]) * scale;
  let vertical = getDot(H3_FACE_BASES[face][2]) * scale;
  if (resolution % 2 === 1) {
    const rotation = 0.3334731722518321;
    [horizontal, vertical] = [
      Math.cos(rotation) * horizontal + Math.sin(rotation) * vertical,
      -Math.sin(rotation) * horizontal + Math.cos(rotation) * vertical
    ];
  }
  let coordinates = getHexCoordinates(horizontal, vertical);
  let digits = new Array<number>(resolution + 1).fill(0);
  const divideSeven = (value: number) => Math.trunc((value + (value >= 0 ? 3 : -3)) / 7);
  for (let level = resolution; level > 0; level--) {
    const previous = coordinates;
    const first = coordinates[0] - coordinates[2];
    const second = coordinates[1] - coordinates[2];
    coordinates =
      level % 2 === 1
        ? normalizeCoordinates([
            divideSeven(3 * first - second),
            divideSeven(first + 2 * second),
            0
          ])
        : normalizeCoordinates([
            divideSeven(2 * first + second),
            divideSeven(3 * second - first),
            0
          ]);
    const [firstParent, secondParent, thirdParent] = coordinates;
    const center =
      level % 2 === 1
        ? [
            3 * firstParent + secondParent,
            3 * secondParent + thirdParent,
            firstParent + 3 * thirdParent
          ]
        : [
            3 * firstParent + thirdParent,
            firstParent + 3 * secondParent,
            secondParent + 3 * thirdParent
          ];
    const normalizedCenter = normalizeCoordinates(center as CoordinateTriple);
    const difference = normalizeCoordinates(
      previous.map((value, axis) => value - normalizedCenter[axis]) as CoordinateTriple
    );
    digits[level] = 4 * difference[0] + 2 * difference[1] + difference[2];
  }
  if (coordinates.some(value => value > 2)) return 0n;
  const packed =
    CELL_INDEX_H3_BASE_CELL_PACKED_TABLE[
      face * 27 + coordinates[0] * 9 + coordinates[1] * 3 + coordinates[2]
    ];
  if (packed === undefined || packed < 0) return 0n;
  const baseCell = packed & 255;
  const pentagon = H3_PENTAGONS.has(baseCell);
  const getLeadingDigit = () => digits.slice(1).find(value => value !== 0) ?? 0;
  const rotateDigits = (table: number[]) => {
    digits = digits.map((value, index) => (index === 0 ? 0 : table[value]));
  };
  if (pentagon && getLeadingDigit() === 1) {
    const clockwise = CELL_INDEX_H3_CW_OFFSET_PENTAGON_FACES.some(
      ([base, firstFace, secondFace]) =>
        base === baseCell && (face === firstFace || face === secondFace)
    );
    rotateDigits(clockwise ? H3_ROTATE_CLOCKWISE : H3_ROTATE_COUNTERCLOCKWISE);
  }
  for (let rotation = 0; rotation < packed >> 8; rotation++) {
    if (!pentagon) rotateDigits(H3_ROTATE_COUNTERCLOCKWISE);
    else {
      let found = false;
      for (let level = 1; level <= resolution; level++) {
        digits[level] = H3_ROTATE_COUNTERCLOCKWISE[digits[level]];
        if (!found && digits[level] !== 0) {
          found = true;
          if (getLeadingDigit() === 1) rotateDigits(H3_ROTATE_COUNTERCLOCKWISE);
        }
      }
    }
  }
  let key = (1n << 59n) | (BigInt(resolution) << 52n) | (BigInt(baseCell) << 45n);
  for (let level = 1; level <= 15; level++)
    key |= BigInt(level <= resolution ? digits[level] : 7) << BigInt(3 * (15 - level));
  return key;
}

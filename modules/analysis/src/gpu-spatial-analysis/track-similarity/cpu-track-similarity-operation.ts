// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {isPlanarTrackPointWithin} from './planar-track-metric';

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getCPUTrajectoryDistance,
  getCPUTrajectoryLongitudeDelta,
  roundTrajectoryEven
} from '../../analysis-core/operations/cpu-trajectory-distance';
import {GPU_TRACK_SIMILARITY_STATUS} from './gpu-track-similarity';
import {
  GPUTrackSimilarityOperation,
  snapshotTrackSimilarityOperationProps,
  type TrackSimilarityOperationProps
} from './gpu-track-similarity-operation';
import {validateMultipartTrackMeasures} from './multipart-track-layout';
import {validateCPULineFeatureLayout} from '../../analysis-core/contracts/line-feature-provenance';

/** Packed float32x2 positions, uint32 offsets and pair indices, as in GPUTrackSimilarity. */
export type CPUTrackSimilarityInput = {
  linesA: {positions: Float32Array; lineOffsets: Uint32Array; featureOffsets?: Uint32Array};
  linesB?: {positions: Float32Array; lineOffsets: Uint32Array; featureOffsets?: Uint32Array};
  pairA: Uint32Array;
  pairB: Uint32Array;
  activePairCount?: Uint32Array;
  frechetEpsilon?: Float32Array;
};

/** Per-pair columns. Unrequested floating columns are NaN and unrequested decisions are zero. */
export type CPUTrackSimilarityResult = {
  hausdorff: Float32Array;
  frechet: Float32Array;
  frechetWithin: Uint32Array;
  dynamicTimeWarping: Float32Array;
  editDistanceOnRealSequence: Float32Array;
  editDistanceWithRealPenalty: Float32Array;
  symmetricSegmentPathDistance: Float32Array;
  longestCommonSubsequence: Float32Array;
  maximumDistance: Float32Array;
  pairStatus: Uint32Array;
};

/** Numeric options have the contributor's meanings; an explicit shorter-track cap is honored. */
export type CPUTrackSimilarityOptions = Pick<
  TrackSimilarityOperationProps,
  | 'spatialContext'
  | 'densify'
  | 'maximumFrechetVertexCount'
  | 'dynamicTimeWarpingBandRadius'
  | 'editDistanceTolerance'
  | 'editDistanceGapPoint'
  | 'lcssTolerance'
  | 'lcssBandRadius'
> & {
  /** Defaults to every measure (frechetWithin only when frechetEpsilon is supplied). */
  measures?: readonly Exclude<keyof CPUTrackSimilarityResult, 'pairStatus'>[];
};

type TrackPoint = readonly [number, number];
type TrackPoints = TrackPoint[] & {partEnds?: Set<number>};

// JavaScript arrays and the contributor's densified row indices both have a uint32 length.
const MAXIMUM_DENSIFIED_VERTEX_COUNT = 0xffffffff;

function validateOptions(id: string, options: CPUTrackSimilarityOptions): void {
  const densify = options.densify ?? 0;
  validateRequirement(id, densify >= 0 && densify <= 1, 'densify must be in [0, 1]');
  validateRequirement(
    id,
    densify === 0 || Number.isSafeInteger(Math.round(1 / densify)),
    'densify must yield a safe integer subdivision count'
  );
  validateCapacity(
    id,
    'maximumFrechetVertexCount',
    options.maximumFrechetVertexCount ?? 256,
    1,
    Number.MAX_SAFE_INTEGER
  );
  for (const name of ['dynamicTimeWarpingBandRadius', 'lcssBandRadius'] as const) {
    if (options[name] !== undefined)
      validateCapacity(id, name, options[name], 0, Number.MAX_SAFE_INTEGER);
  }
  for (const name of ['editDistanceTolerance', 'lcssTolerance'] as const) {
    validateFiniteNumber(id, name, options[name] ?? 1, 0);
    validateRequirement(id, (options[name] ?? 1) >= 0, `${name} must be nonnegative`);
  }
  const gapPoint = options.editDistanceGapPoint ?? [0, 0];
  validateRequirement(id, gapPoint.length === 2, 'editDistanceGapPoint must have two coordinates');
  for (const coordinate of gapPoint) validateFiniteNumber(id, 'editDistanceGapPoint', coordinate);
  resolveGPUSpatialContext(id, options.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
}

function validateLines(name: string, lines: CPUTrackSimilarityInput['linesA']): void {
  validateCPULineFeatureLayout(
    'computeTrackSimilarityOnCPU',
    lines.positions,
    lines.lineOffsets,
    lines.featureOffsets,
    name,
    true
  );
}

function getTrackPoints(lines: CPUTrackSimilarityInput['linesA'], track: number): TrackPoints {
  const points: TrackPoints = [];
  const firstPart = lines.featureOffsets?.[track] ?? track;
  const endPart = lines.featureOffsets?.[track + 1] ?? track + 1;
  for (let part = firstPart; part < endPart; part++) {
    for (let row = lines.lineOffsets[part]; row < lines.lineOffsets[part + 1]; row++) {
      points.push([lines.positions[row * 2], lines.positions[row * 2 + 1]]);
    }
    if (lines.featureOffsets && points.length) {
      points.partEnds ??= new Set();
      points.partEnds.add(points.length - 1);
    }
  }
  return points;
}

function densifyTrack(points: TrackPoints, subdivisions: number, geographic: boolean): TrackPoints {
  if (subdivisions === 1 || points.length < 2) return points;
  const densePoints: TrackPoints = [];
  for (let segment = 0; segment + 1 < points.length; segment++) {
    const first = points[segment];
    if (points.partEnds?.has(segment)) {
      densePoints.push(first);
      densePoints.partEnds ??= new Set();
      densePoints.partEnds.add(densePoints.length - 1);
      continue;
    }
    const second = points[segment + 1];
    const longitudeDifference = second[0] - first[0];
    const longitudeDelta = geographic
      ? longitudeDifference - 360 * roundTrajectoryEven(longitudeDifference / 360)
      : longitudeDifference;
    for (let subdivision = 0; subdivision < subdivisions; subdivision++) {
      const fraction = subdivision / subdivisions;
      densePoints.push([
        first[0] + longitudeDelta * fraction,
        first[1] + (second[1] - first[1]) * fraction
      ]);
    }
  }
  densePoints.push(points[points.length - 1]);
  return densePoints;
}

/** GPU's planar projection, or geodesicSegmentFraction and geodesicSegmentDistance in float64. */
function computeTrackSegmentDistance(
  point: TrackPoint,
  start: TrackPoint,
  end: TrackPoint,
  radius: number | undefined
): number {
  let startLongitude = start[0] - point[0];
  const startLatitude = start[1] - point[1];
  const radians = Math.PI / 180;
  const cosine = radius === undefined ? 1 : Math.sin((90 - Math.abs(point[1])) * radians);
  if (radius !== undefined) {
    startLongitude = getCPUTrajectoryLongitudeDelta(point[0], start[0]);
  }
  const startEast = startLongitude * cosine;
  const longitudeDelta =
    radius === undefined ? end[0] - start[0] : getCPUTrajectoryLongitudeDelta(start[0], end[0]);
  const directionEast = longitudeDelta * cosine;
  const directionNorth = end[1] - start[1];
  const denominator = directionEast * directionEast + directionNorth * directionNorth;
  const fraction =
    denominator > 0 && (start[0] !== end[0] || start[1] !== end[1])
      ? Math.max(
          0,
          Math.min(1, -(startEast * directionEast + startLatitude * directionNorth) / denominator)
        )
      : 0;
  if (radius === undefined) {
    return Math.hypot(
      startLongitude + fraction * directionEast,
      startLatitude + fraction * directionNorth
    );
  }
  const longitudeOffset = startLongitude + fraction * longitudeDelta;
  const latitudeOffset = startLatitude + fraction * (end[1] - start[1]);
  const term =
    Math.sin(0.5 * latitudeOffset * radians) ** 2 +
    Math.sin((90 - Math.abs(point[1])) * radians) *
      Math.sin((90 - Math.abs(point[1] + latitudeOffset)) * radians) *
      Math.sin(0.5 * longitudeOffset * radians) ** 2;
  return Math.min(
    2 * radius * Math.asin(Math.sqrt(Math.max(0, Math.min(1, term)))),
    computeTrackPointDistance(point, start, radius),
    computeTrackPointDistance(point, end, radius)
  );
}

function computeTrackPointDistance(
  first: TrackPoint,
  second: TrackPoint,
  radius: number | undefined
): number {
  return getCPUTrajectoryDistance([first[0], first[1], second[0], second[1]], 0, 1, radius);
}

function computeNearestTrackDistance(
  point: TrackPoint,
  track: TrackPoints,
  radius: number | undefined
): number {
  if (track.length === 1) return computeTrackPointDistance(point, track[0], radius);
  let nearest = track.partEnds
    ? computeTrackPointDistance(point, track[track.length - 1], radius)
    : Infinity;
  for (let segment = 0; segment + 1 < track.length; segment++) {
    nearest = Math.min(
      nearest,
      computeTrackSegmentDistance(
        point,
        track[segment],
        track[track.partEnds?.has(segment) ? segment : segment + 1],
        radius
      )
    );
  }
  return nearest;
}

function computeTrackProgram(
  first: TrackPoint[],
  second: TrackPoint[],
  kind:
    | 'frechetWithin'
    | 'frechet'
    | 'dynamicTimeWarping'
    | 'editDistanceOnRealSequence'
    | 'editDistanceWithRealPenalty'
    | 'longestCommonSubsequence',
  radius: number | undefined,
  options: CPUTrackSimilarityOptions,
  epsilon = 0
): number {
  const longer = first.length >= second.length ? first : second;
  const shorter = first.length >= second.length ? second : first;
  const isEdit = kind === 'editDistanceOnRealSequence' || kind === 'editDistanceWithRealPenalty';
  const isDecision = kind === 'frechetWithin';
  const isSubsequence = kind === 'longestCommonSubsequence';
  const tolerance = Math.fround(
    isSubsequence ? (options.lcssTolerance ?? 1) : (options.editDistanceTolerance ?? 1)
  );
  const band = isSubsequence ? options.lcssBandRadius : options.dynamicTimeWarpingBandRadius;
  const gapPoint: TrackPoint = [
    Math.fround(options.editDistanceGapPoint?.[0] ?? 0),
    Math.fround(options.editDistanceGapPoint?.[1] ?? 0)
  ];
  const getGapCost = (point: TrackPoint) =>
    kind === 'editDistanceOnRealSequence' ? 1 : computeTrackPointDistance(point, gapPoint, radius);
  let previous = new Float64Array(shorter.length + 1).fill(isSubsequence || isEdit ? 0 : Infinity);
  previous[0] = 0;
  if (isEdit) {
    for (let column = 1; column <= shorter.length; column++)
      previous[column] = previous[column - 1] + getGapCost(shorter[column - 1]);
  }
  for (let row = 1; row <= longer.length; row++) {
    const current = new Float64Array(shorter.length + 1).fill(isSubsequence ? 0 : Infinity);
    if (isEdit) current[0] = previous[0] + getGapCost(longer[row - 1]);
    for (let column = 1; column <= shorter.length; column++) {
      const distance = computeTrackPointDistance(longer[row - 1], shorter[column - 1], radius);
      const insideBand =
        band === undefined ||
        (row - 1 + band >= column - 1 &&
          row - 1 <= column - 1 + longer.length - shorter.length + band);
      if (isSubsequence) {
        current[column] = Math.max(
          previous[column],
          current[column - 1],
          insideBand &&
            (radius === undefined
              ? isPlanarTrackPointWithin(longer[row - 1], shorter[column - 1], tolerance)
              : distance <= tolerance)
            ? previous[column - 1] + 1
            : 0
        );
      } else if (isEdit) {
        const substitution =
          kind === 'editDistanceOnRealSequence'
            ? Number(
                !(radius === undefined
                  ? isPlanarTrackPointWithin(longer[row - 1], shorter[column - 1], tolerance)
                  : distance <= tolerance)
              )
            : distance;
        current[column] = Math.min(
          previous[column - 1] + substitution,
          previous[column] + getGapCost(longer[row - 1]),
          current[column - 1] + getGapCost(shorter[column - 1])
        );
      } else if (isDecision) {
        const within =
          radius === undefined
            ? isPlanarTrackPointWithin(longer[row - 1], shorter[column - 1], epsilon)
            : distance <= epsilon;
        current[column] = Math.max(
          Number(!within),
          Math.min(previous[column - 1], previous[column], current[column - 1])
        );
      } else if (kind === 'frechet') {
        current[column] = Math.max(
          distance,
          Math.min(previous[column - 1], previous[column], current[column - 1])
        );
      } else if (insideBand) {
        current[column] =
          distance + Math.min(previous[column - 1], previous[column], current[column - 1]);
      }
    }
    previous = current;
  }
  return previous[shorter.length] / (isSubsequence ? shorter.length : 1);
}

/**
 * Computes the requested trajectory measures, with the contributor's densification, inclusive
 * matching thresholds, widened Sakoe-Chiba bands and original-vertex SSPD/maximum distance.
 *
 * What differs from the GPU contributor:
 * - Distance arithmetic and virtual densification use float64, with float32 result columns.
 *   Planar matching predicates agree on identical vertex inputs, but float64 CPU versus float32
 *   GPU virtual vertices can differ after densification and change Frechet, EDR or LCSS decisions.
 *   Geographic matching also retains distance-arithmetic differences. Compare noninteger
 *   distances with relative tolerance 3e-5.
 *   Numeric match tolerances and the ERP gap point are converted to float32 as GPU constants are.
 * - The shorter-track, subdivision, pair-dispatch and band-index GPU limits are dropped.
 *   An explicit maximumFrechetVertexCount bounds dynamic-program measures, setting
 *   frechetCapExceeded and leaving their outputs unavailable when exceeded.
 *   Measures that densify reject expanded tracks above 2^32 - 1 vertices before allocation.
 * - Dynamic programs compute the complete documented method without GPU strips or early exits.
 *   DTW costs are not saturated at the largest finite float32; output overflow becomes Infinity.
 * - Inconsistent arrays and decreasing/out-of-range lineOffsets throw an Error naming the array.
 *   Nonfinite coordinates or geographic latitudes outside [-90, 90] leave distances NaN and
 *   frechetWithin zero, with invalidGeometry evidence. Invalid epsilon affects only decision validity.
 *
 * @throws Error naming an argument when array layouts or options are inconsistent.
 */
export function computeTrackSimilarityOnCPU(
  input: CPUTrackSimilarityInput,
  options: CPUTrackSimilarityOptions = {}
): CPUTrackSimilarityResult {
  validateOptions('computeTrackSimilarityOnCPU', options);
  validateLines('linesA', input.linesA);
  if (input.linesB) validateLines('linesB', input.linesB);
  validateRequirement(
    'computeTrackSimilarityOnCPU',
    input.pairB.length === input.pairA.length,
    `pairB.length must equal pairA.length ${input.pairA.length}; received ${input.pairB.length}`
  );
  if (input.activePairCount?.length === 0)
    throw new Error('activePairCount must have at least one row');
  if (input.frechetEpsilon?.length === 0)
    throw new Error('frechetEpsilon must have at least one row');
  const count = input.pairA.length;
  const result: CPUTrackSimilarityResult = {
    hausdorff: new Float32Array(count).fill(NaN),
    frechet: new Float32Array(count).fill(NaN),
    frechetWithin: new Uint32Array(count),
    dynamicTimeWarping: new Float32Array(count).fill(NaN),
    editDistanceOnRealSequence: new Float32Array(count).fill(NaN),
    editDistanceWithRealPenalty: new Float32Array(count).fill(NaN),
    symmetricSegmentPathDistance: new Float32Array(count).fill(NaN),
    longestCommonSubsequence: new Float32Array(count).fill(NaN),
    maximumDistance: new Float32Array(count).fill(NaN),
    pairStatus: new Uint32Array(count)
  };
  const measures = new Set(
    options.measures ??
      Object.keys(result).filter(
        port =>
          port !== 'pairStatus' && (port !== 'frechetWithin' || input.frechetEpsilon !== undefined)
      )
  );
  validateMultipartTrackMeasures(
    'computeTrackSimilarityOnCPU',
    Boolean(input.linesA.featureOffsets || input.linesB?.featureOffsets),
    [...measures]
  );
  for (const measure of measures) {
    if (!(measure in result) || measure === 'pairStatus')
      throw new Error(`measures contains unknown measure ${measure}`);
  }
  if (measures.has('frechetWithin') && !input.frechetEpsilon)
    throw new Error('frechetEpsilon is required for frechetWithin');
  const context = resolveGPUSpatialContext('computeTrackSimilarityOnCPU', options.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
  const radius = getGPUGreatCircleRadius(context);
  const subdivisions = options.densify ? Math.round(1 / options.densify) : 1;
  const requiresDensification = [...measures].some(
    measure => measure !== 'maximumDistance' && measure !== 'symmetricSegmentPathDistance'
  );
  const linesB = input.linesB ?? input.linesA;
  for (let pair = 0; pair < count; pair++) {
    const firstTrack = input.pairA[pair];
    const secondTrack = input.pairB[pair];
    if (
      pair >= (input.activePairCount?.[0] ?? count) ||
      firstTrack + 1 >= (input.linesA.featureOffsets ?? input.linesA.lineOffsets).length ||
      secondTrack + 1 >= (linesB.featureOffsets ?? linesB.lineOffsets).length
    ) {
      result.pairStatus[pair] = GPU_TRACK_SIMILARITY_STATUS.invalidPair;
      continue;
    }
    if (
      measures.has('frechetWithin') &&
      (!Number.isFinite(input.frechetEpsilon![0]) || input.frechetEpsilon![0] < 0)
    ) {
      result.pairStatus[pair] |= GPU_TRACK_SIMILARITY_STATUS.invalidDecisionThreshold;
    }
    const first = getTrackPoints(input.linesA, firstTrack);
    const second = getTrackPoints(linesB, secondTrack);
    if (!first.length || !second.length) {
      result.pairStatus[pair] |= GPU_TRACK_SIMILARITY_STATUS.emptyTrack;
      continue;
    }
    if (
      ![first, second].every(track =>
        track.every(
          point =>
            point.every(Number.isFinite) && (radius === undefined || Math.abs(point[1]) <= 90)
        )
      )
    ) {
      result.pairStatus[pair] |= GPU_TRACK_SIMILARITY_STATUS.invalidGeometry;
      continue;
    }
    const capExceeded =
      options.maximumFrechetVertexCount !== undefined &&
      Math.min((first.length - 1) * subdivisions + 1, (second.length - 1) * subdivisions + 1) >
        options.maximumFrechetVertexCount &&
      [...measures].some(
        measure =>
          measure !== 'hausdorff' &&
          measure !== 'maximumDistance' &&
          measure !== 'symmetricSegmentPathDistance'
      );
    if (capExceeded) result.pairStatus[pair] |= GPU_TRACK_SIMILARITY_STATUS.frechetCapExceeded;
    const densifyPair = requiresDensification && (!capExceeded || measures.has('hausdorff'));
    if (densifyPair) {
      // Preflight both tracks before allocating either dense point array.
      for (const track of [first, second]) {
        validateCapacity(
          'computeTrackSimilarityOnCPU',
          'densifiedVertexCount',
          (track.length - 1) * subdivisions + 1,
          1,
          MAXIMUM_DENSIFIED_VERTEX_COUNT
        );
      }
    }
    const denseFirst = densifyPair
      ? densifyTrack(first, subdivisions, radius !== undefined)
      : first;
    const denseSecond = densifyPair
      ? densifyTrack(second, subdivisions, radius !== undefined)
      : second;
    if (measures.has('hausdorff')) {
      let largest = 0;
      for (const [points, target] of [
        [denseFirst, second],
        [denseSecond, first]
      ]) {
        for (const point of points)
          largest = Math.max(largest, computeNearestTrackDistance(point, target, radius));
      }
      result.hausdorff[pair] = largest;
    }
    if (measures.has('symmetricSegmentPathDistance')) {
      let means = 0;
      for (const [points, target] of [
        [first, second],
        [second, first]
      ]) {
        let sum = 0;
        for (const point of points) sum += computeNearestTrackDistance(point, target, radius);
        means += sum / points.length;
      }
      result.symmetricSegmentPathDistance[pair] = means / 2;
    }
    if (measures.has('maximumDistance')) {
      let largest = 0;
      for (const firstPoint of first) {
        for (const secondPoint of second)
          largest = Math.max(largest, computeTrackPointDistance(firstPoint, secondPoint, radius));
      }
      result.maximumDistance[pair] = largest;
    }
    for (const kind of [
      'frechet',
      'dynamicTimeWarping',
      'editDistanceOnRealSequence',
      'editDistanceWithRealPenalty',
      'longestCommonSubsequence'
    ] as const) {
      if (capExceeded) continue;
      if (!measures.has(kind) && !(kind === 'frechet' && measures.has('frechetWithin'))) continue;
      const value = computeTrackProgram(denseFirst, denseSecond, kind, radius, options);
      if (measures.has(kind)) result[kind][pair] = value;
      if (kind === 'frechet' && measures.has('frechetWithin')) {
        const epsilon = input.frechetEpsilon![0];
        result.frechetWithin[pair] = Number(
          Number.isFinite(epsilon) &&
            epsilon >= 0 &&
            computeTrackProgram(
              denseFirst,
              denseSecond,
              'frechetWithin',
              radius,
              options,
              epsilon
            ) === 0
        );
      }
    }
  }
  return result;
}

/** CPU twin, retaining uint32 output sizes and dropping GPU workgroup/dispatch limits. */
export class CPUTrackSimilarityOperation extends GPUCPUOperation {
  constructor(props: TrackSimilarityOperationProps) {
    props = snapshotTrackSimilarityOperationProps(props);
    const id = props.id ?? 'track-similarity';
    validateCapacity(id, 'pairCount', props.pairCount, 0);
    validateOptions(id, props);
    validateRequirement(
      id,
      Object.entries(props.output).some(
        ([port, value]) => port !== 'pairStatus' && value !== undefined
      ),
      'output must request at least one measure'
    );
    validateRequirement(
      id,
      (props.output.frechetWithin !== undefined) === (props.frechetEpsilon !== undefined),
      'output.frechetWithin and frechetEpsilon must be given together'
    );
    const operation = new GPUTrackSimilarityOperation(props);
    const measures = Object.keys(operation.outputs).filter(
      port => port !== 'pairStatus'
    ) as NonNullable<CPUTrackSimilarityOptions['measures']>;
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
        const pairA = getCPUOperationUint32Rows(id, inputs, 'pairA');
        if (pairA.length !== props.pairCount)
          throw new Error(`${id} pairA length must equal pairCount`);
        const result = computeTrackSimilarityOnCPU(
          {
            linesA: {
              positions: getCPUOperationFloat32Rows(id, inputs, 'positionsA', 'float32x2'),
              lineOffsets: getCPUOperationUint32Rows(id, inputs, 'offsetsA'),
              featureOffsets: props.linesA.featureOffsets
                ? getCPUOperationUint32Rows(id, inputs, 'featuresA')
                : undefined
            },
            linesB: props.linesB
              ? {
                  positions: getCPUOperationFloat32Rows(id, inputs, 'positionsB', 'float32x2'),
                  lineOffsets: getCPUOperationUint32Rows(id, inputs, 'offsetsB'),
                  featureOffsets: props.linesB.featureOffsets
                    ? getCPUOperationUint32Rows(id, inputs, 'featuresB')
                    : undefined
                }
              : undefined,
            pairA,
            pairB: getCPUOperationUint32Rows(id, inputs, 'pairB'),
            activePairCount:
              props.activePairCount === undefined
                ? undefined
                : getCPUOperationUint32Rows(id, inputs, 'activePairCount'),
            frechetEpsilon:
              props.frechetEpsilon === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'frechetEpsilon')
          },
          {...props, measures}
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTrackSimilarityResult]
          ])
        );
      }
    });
  }
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  isWithinValidationRadiusOnCPU,
  isWithinValidationNeighborOnCPU
} from './validation-buffer-membership-wgsl';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  snapshotSpatialCrossValidationProps,
  validateSpatialValidationDesignMode
} from './spatial-validation-design';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getPhilox4x32, PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUSpatialCrossValidationOperation,
  type SpatialCrossValidationOperationProps
} from './gpu-spatial-cross-validation-operation';

const NONE = 0xffffffff;
// The shader's finite sentinel is an output value in NNDM diagnostics, not Number.MAX_VALUE.
const NO_DISTANCE = Math.fround(3.402823e38);
/** Candidate count and selected index share float32 diagnostics; keep every index exact. */
export const CPU_SPATIAL_CROSS_VALIDATION_MAXIMUM_CANDIDATE_COUNT = 2 ** 24;
type NumericDesign<Design = NonNullable<SpatialCrossValidationOperationProps['design']>> =
  Design extends {mode: string}
    ? Omit<
        Design,
        'features' | 'predictionPositions' | 'predictionRowCount' | 'candidates' | 'diagnostics'
      >
    : never;

/** Packed rows for block folds or any of the four explicit fold designs. */
export type CPUSpatialCrossValidationInput = {
  positions: Float32Array;
  blocks: Uint32Array;
  mask?: Uint32Array;
  features?: Float32Array;
  predictionPositions?: Float32Array;
  candidates?: Uint32Array;
};
/** Numeric design properties. Candidate count still defines the requested restart/candidate set. */
export type CPUSpatialCrossValidationOptions = Pick<
  SpatialCrossValidationOperationProps,
  'id' | 'spatialContext' | 'foldCount' | 'seed' | 'bufferDistance'
> & {design?: NumericDesign};
/** Fold-major masks, per-fold [train, test, excluded] counts, and optional design state. */
export type CPUSpatialCrossValidationResult = {
  folds: Uint32Array;
  training: Uint32Array;
  testing: Uint32Array;
  counts: Uint32Array;
  status: Uint32Array;
  diagnostics?: Float32Array;
};

function validateOptions(options: CPUSpatialCrossValidationOptions, rowCount: number): void {
  const id = options.id ?? 'spatial-cross-validation';
  resolveGPUSpatialContext(id, options.spatialContext);
  validateCapacity(id, 'rowCount', rowCount, options.design ? 2 : 1);
  validateCapacity(id, 'foldCount', options.foldCount, 2);
  validateMaximumCount(id, 'rowCount times foldCount', rowCount * options.foldCount, NONE);
  validateMaximumCount(id, 'foldCount times three', options.foldCount * 3, NONE);
  validateCapacity(id, 'seed', options.seed ?? 0, 0);
  validateRequirement(
    id,
    Number.isFinite(options.bufferDistance ?? 0) && (options.bufferDistance ?? 0) >= 0,
    'bufferDistance must be finite and nonnegative'
  );
  const design = options.design;
  if (!design) return;
  validateSpatialValidationDesignMode(id, design.mode);
  if (design.mode === 'environmental') {
    validateCapacity(id, 'design.featureCount', design.featureCount);
    if (design.maximumIterations !== undefined)
      validateCapacity(id, 'design.maximumIterations', design.maximumIterations, 1);
    validateMaximumCount(id, 'rowCount times featureCount', rowCount * design.featureCount, NONE);
  }
  if (design.mode === 'nndm') {
    validateRequirement(
      id,
      options.foldCount === rowCount,
      'foldCount must equal rowCount for NNDM'
    );
    validateRequirement(
      id,
      Number.isFinite(design.maximumDistance) && design.maximumDistance >= 0,
      'design.maximumDistance must be finite and nonnegative'
    );
    validateRequirement(
      id,
      (design.minimumTrainingFraction ?? 0.05) > 0 && (design.minimumTrainingFraction ?? 0.05) < 1,
      'design.minimumTrainingFraction must be in (0, 1)'
    );
  }
  if (design.mode === 'knndm') {
    validateCapacity(
      id,
      'design.candidateCount',
      design.candidateCount ?? 8,
      1,
      CPU_SPATIAL_CROSS_VALIDATION_MAXIMUM_CANDIDATE_COUNT
    );
    validateRequirement(
      id,
      (design.maximumFoldFraction ?? 0.8) > 1 / options.foldCount &&
        (design.maximumFoldFraction ?? 0.8) < 1,
      'design.maximumFoldFraction must be in (1 / foldCount, 1)'
    );
    validateMaximumCount(
      id,
      'rowCount times candidateCount',
      rowCount * (design.candidateCount ?? 8),
      NONE
    );
  }
}

type ClusterResult = {
  assignments: Uint32Array;
  status: number;
  iterations: number;
  objective: number;
};

// Deterministic farthest-first initialization, followed by Lloyd steps until assignments stabilize.
function clusterFeatures(
  features: Float64Array,
  complete: readonly boolean[],
  featureCount: number,
  clusterCount: number,
  seed: number,
  standardize: boolean,
  maximumIterations = Infinity
): ClusterResult {
  const rowCount = complete.length;
  const result: ClusterResult = {
    assignments: new Uint32Array(rowCount).fill(NONE),
    status: 0,
    iterations: 0,
    objective: NaN
  };
  const members = complete.flatMap((valid, row) => (valid ? [row] : []));
  if (members.length < clusterCount) return {...result, status: 1};
  const scales = new Float64Array(featureCount).fill(1);
  if (standardize) {
    for (let column = 0; column < featureCount; column++) {
      let mean = 0;
      let squaredDeviations = 0;
      for (const [index, row] of members.entries()) {
        const difference = features[row * featureCount + column] - mean;
        mean += difference / (index + 1);
        squaredDeviations += difference * (features[row * featureCount + column] - mean);
      }
      scales[column] =
        squaredDeviations > 0 ? Math.sqrt(squaredDeviations / (members.length - 1)) : 1;
    }
  }
  let seedRow = seed % rowCount;
  while (!complete[seedRow]) seedRow = (seedRow + 1) % rowCount;
  const centers = new Float64Array(clusterCount * featureCount);
  const getSquaredDistance = (row: number, cluster: number): number => {
    let squared = 0;
    for (let column = 0; column < featureCount; column++)
      squared +=
        ((features[row * featureCount + column] - centers[cluster * featureCount + column]) /
          scales[column]) **
        2;
    return squared;
  };
  for (let cluster = 0; cluster < clusterCount; cluster++) {
    if (cluster > 0) {
      let farthest = -1;
      for (const row of members) {
        let nearest = Infinity;
        for (let previous = 0; previous < cluster; previous++)
          nearest = Math.min(nearest, getSquaredDistance(row, previous));
        if (nearest > farthest) {
          farthest = nearest;
          seedRow = row;
        }
      }
      if (!(farthest > 0)) return {...result, status: 2};
    }
    centers.set(
      features.subarray(seedRow * featureCount, (seedRow + 1) * featureCount),
      cluster * featureCount
    );
  }
  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    const counts = new Float64Array(clusterCount);
    const sums = new Float64Array(clusterCount * featureCount);
    let changed = false;
    let objective = 0;
    for (const row of members) {
      let nearest = Infinity;
      let selected = 0;
      for (let cluster = 0; cluster < clusterCount; cluster++) {
        const squared = getSquaredDistance(row, cluster);
        if (squared < nearest) {
          nearest = squared;
          selected = cluster;
        }
      }
      changed ||= result.assignments[row] !== selected;
      result.assignments[row] = selected;
      objective += nearest;
      counts[selected]++;
      for (let column = 0; column < featureCount; column++)
        sums[selected * featureCount + column] +=
          features[row * featureCount + column] - centers[selected * featureCount + column];
    }
    if (counts.some(count => count === 0))
      return {...result, assignments: new Uint32Array(rowCount).fill(NONE), status: 2};
    for (let cluster = 0; cluster < clusterCount; cluster++)
      for (let column = 0; column < featureCount; column++)
        centers[cluster * featureCount + column] +=
          sums[cluster * featureCount + column] / counts[cluster];
    result.iterations++;
    result.objective = objective;
    if (!changed) return result;
  }
  return {...result, assignments: new Uint32Array(rowCount).fill(NONE), status: 3};
}

/**
 * Computes seeded block folds or the groups, environmental, NNDM and kNNDM designs, then excludes
 * training rows at distance <= the buffer radius of a testing row. Only finite positions with a
 * nonzero mask are usable. Environmental rows additionally require complete finite features;
 * NNDM/kNNDM require every input and prediction row usable, otherwise design status is 1.
 * Diagnostics retain [status, iterations, objective, selected candidate] and NNDM radii. As on
 * GPU, groups and NNDM leave the objective/candidate NaN, environmental leaves candidate NaN,
 * and failed designs publish missing folds. Ties use the first row/cluster/candidate.
 *
 * What differs from the GPU contributor:
 * - Float64 distances, standardization and objectives round only when diagnostics are written.
 *   Near cluster ties, fold choices can differ; planar buffer membership uses inclusive Euclidean
 *   distance on stored coordinates; diagnostics use relative tolerance
 *   2e-5 (plus 2e-6 absolute for zero objectives).
 * - GPU-only row, fold, feature, prediction, cluster and candidate caps are removed; uint32
 *   output-length limits and a 2^24 candidate-count limit remain so float32 diagnostics preserve
 *   every selectable index exactly. Candidate count still defines the supplied/generated ladder.
 * - Environmental clustering honors explicit maximumIterations; exhausted budgets emit design
 *   status 3 and missing folds. Without a budget, CPU clustering retains convergence as its
 *   default, including generated kNNDM restarts; the GPU has a bounded default of 100 steps.
 * - Float64 accepts finite distances/features beyond the GPU's arithmetic range and removes
 *   the GPU's 1e18 buffer/maximum-distance bound. Non-finite rows keep the GPU's missing rules.
 * - An exhausted NNDM neighborhood uses the GPU's finite no-distance sentinel in diagnostics;
 *   internal comparisons use infinity rather than clamping finite distances to that sentinel.
 * - Failed NNDM designs return deterministic zero radii. The GPU returns before writing those
 *   fields, so a reused diagnostics view can retain radii from an earlier successful run.
 *
 * @throws If positions, blocks, mask, features, predictionPositions or candidates are inconsistent.
 */
export function computeSpatialCrossValidationOnCPU(
  input: CPUSpatialCrossValidationInput,
  options: CPUSpatialCrossValidationOptions
): CPUSpatialCrossValidationResult {
  options = snapshotSpatialCrossValidationProps(options);
  const {positions, blocks, mask, features, predictionPositions, candidates} = input;
  const rowCount = blocks.length;
  if (positions.length !== rowCount * 2)
    throw new Error('positions length must equal blocks length * 2');
  if (mask && mask.length !== rowCount) throw new Error('mask length must equal blocks length');
  validateOptions(options, rowCount);
  const {foldCount, design} = options;
  if (
    design?.mode === 'environmental' &&
    (!features || features.length !== rowCount * design.featureCount)
  )
    throw new Error('features length must equal blocks length * featureCount');
  if (
    (design?.mode === 'nndm' || design?.mode === 'knndm') &&
    (!predictionPositions || predictionPositions.length < 2 || predictionPositions.length % 2 !== 0)
  )
    throw new Error('predictionPositions must contain at least one xy pair');
  if (
    design?.mode === 'knndm' &&
    candidates &&
    candidates.length !== rowCount * (design.candidateCount ?? 8)
  )
    throw new Error('candidates length must equal blocks length * candidateCount');
  const usable = Array.from(
    {length: rowCount},
    (_, row) =>
      Number.isFinite(positions[row * 2]) &&
      Number.isFinite(positions[row * 2 + 1]) &&
      (mask?.[row] ?? 1) !== 0
  );
  const result: CPUSpatialCrossValidationResult = {
    folds: new Uint32Array(rowCount).fill(NONE),
    training: new Uint32Array(rowCount * foldCount),
    testing: new Uint32Array(rowCount * foldCount),
    counts: new Uint32Array(foldCount * 3),
    status: new Uint32Array(foldCount),
    diagnostics: design ? new Float32Array(4 + (design.mode === 'nndm' ? rowCount : 0)) : undefined
  };
  // Keep all design state in float64 until the very end; masks must not read rounded radii.
  const diagnostics = new Float64Array(4 + (design?.mode === 'nndm' ? rowCount : 0));
  const adaptiveNeighbors = new Uint32Array(design?.mode === 'nndm' ? rowCount : 0).fill(NONE);
  diagnostics[2] = diagnostics[3] = NaN;
  const getRowDistance = (first: number, second: number) =>
    Math.hypot(
      positions[first * 2] - positions[second * 2],
      positions[first * 2 + 1] - positions[second * 2 + 1]
    );
  if (!design || design.mode === 'groups') {
    const groupIds = [...new Set(blocks.filter((_, row) => usable[row]))];
    if (design) groupIds.sort((first, second) => first - second);
    else {
      const keys = new Map(
        groupIds.map(group => [
          group,
          getPhilox4x32(
            [group, 0, 0, 0],
            [options.seed ?? 0, PHILOX_PURPOSES.spatialCrossValidationBlockOrder.id]
          )[0]
        ])
      );
      groupIds.sort((first, second) => keys.get(first)! - keys.get(second)! || first - second);
    }
    const groupFolds = new Map(groupIds.map((group, rank) => [group, rank % foldCount]));
    for (let row = 0; row < rowCount; row++)
      if (usable[row]) result.folds[row] = groupFolds.get(blocks[row])!;
  } else if (design.mode === 'environmental') {
    const complete = usable.map(
      (valid, row) =>
        valid &&
        features!
          .subarray(row * design.featureCount, (row + 1) * design.featureCount)
          .every(Number.isFinite)
    );
    const clustered = clusterFeatures(
      Float64Array.from(features!),
      complete,
      design.featureCount,
      foldCount,
      options.seed ?? 0,
      design.standardize !== false,
      design.maximumIterations
    );
    result.folds.set(clustered.assignments);
    diagnostics.set([clustered.status, clustered.iterations, clustered.objective, NaN]);
  } else {
    const predictionCount = predictionPositions!.length / 2;
    if (usable.some(valid => !valid) || !predictionPositions!.every(Number.isFinite))
      diagnostics[0] = 1;
    else {
      const predictionDistances = Float64Array.from({length: predictionCount}, (_, prediction) => {
        let nearest = Infinity;
        for (let row = 0; row < rowCount; row++)
          nearest = Math.min(
            nearest,
            Math.hypot(
              predictionPositions![prediction * 2] - positions[row * 2],
              predictionPositions![prediction * 2 + 1] - positions[row * 2 + 1]
            )
          );
        return nearest;
      });
      if (design.mode === 'nndm') {
        const neighborhoods = Array.from({length: rowCount}, (_, row) =>
          Array.from({length: rowCount}, (_, other) => other)
            .filter(other => other !== row)
            .sort((first, second) => {
              const firstDistance = getRowDistance(row, first);
              const secondDistance = getRowDistance(row, second);
              if (firstDistance < secondDistance * (1 - 8 * Number.EPSILON)) return -1;
              if (firstDistance > secondDistance * (1 + 8 * Number.EPSILON)) return 1;
              const firstWithin = isWithinValidationNeighborOnCPU(
                positions[first * 2],
                positions[first * 2 + 1],
                positions[row * 2],
                positions[row * 2 + 1],
                positions[second * 2],
                positions[second * 2 + 1]
              );
              const secondWithin = isWithinValidationNeighborOnCPU(
                positions[second * 2],
                positions[second * 2 + 1],
                positions[row * 2],
                positions[row * 2 + 1],
                positions[first * 2],
                positions[first * 2 + 1]
              );
              return firstWithin === secondWithin ? first - second : firstWithin ? -1 : 1;
            })
        );
        const nearest = Float64Array.from(neighborhoods, (members, row) =>
          getRowDistance(row, members[0])
        );
        let radius = nearest.reduce((minimum, distance) => Math.min(minimum, distance), Infinity);
        let chosen = nearest.indexOf(radius);
        while (radius <= design.maximumDistance) {
          const trainingBelow = nearest.filter(distance => distance <= radius).length;
          const predictionBelow = predictionDistances.filter(distance => distance <= radius).length;
          const remove =
            (trainingBelow - 1) / rowCount >= predictionBelow / predictionCount &&
            neighborhoods[chosen].length / rowCount > (design.minimumTrainingFraction ?? 0.05);
          if (remove) {
            neighborhoods[chosen].shift();
            nearest[chosen] = neighborhoods[chosen].length
              ? getRowDistance(chosen, neighborhoods[chosen][0])
              : Infinity;
          } else if (trainingBelow === rowCount) break;
          let next = Infinity;
          for (let row = 0; row < rowCount; row++)
            if ((remove ? nearest[row] >= radius : nearest[row] > radius) && nearest[row] < next) {
              next = nearest[row];
              chosen = row;
            }
          radius = next;
          diagnostics[1]++;
        }
        for (let row = 0; row < rowCount; row++) {
          result.folds[row] = row;
          diagnostics[row + 4] = nearest[row];
          adaptiveNeighbors[row] = neighborhoods[row][0] ?? NONE;
        }
      } else {
        const candidateCount = design.candidateCount ?? 8;
        let candidateRows = candidates;
        if (!candidateRows) {
          candidateRows = new Uint32Array(rowCount * candidateCount);
          for (let candidate = 0; candidate < candidateCount; candidate++) {
            const clustered = clusterFeatures(
              Float64Array.from(positions),
              usable,
              2,
              foldCount,
              ((options.seed ?? 0) + Math.imul(candidate, 2654435761)) >>> 0,
              false
            );
            candidateRows.set(clustered.assignments, candidate * rowCount);
          }
        }
        let best = Infinity;
        let selected = NONE;
        for (let candidate = 0; candidate < candidateCount; candidate++) {
          const assignments = candidateRows.subarray(
            candidate * rowCount,
            (candidate + 1) * rowCount
          );
          const counts = new Uint32Array(foldCount);
          if (assignments.some(fold => fold >= foldCount)) continue;
          for (const fold of assignments) counts[fold]++;
          if (
            counts.some(
              count => count === 0 || count > (design.maximumFoldFraction ?? 0.8) * rowCount
            )
          )
            continue;
          const nearest = Float64Array.from({length: rowCount}, (_, row) => {
            let distance = Infinity;
            for (let other = 0; other < rowCount; other++)
              if (assignments[other] !== assignments[row])
                distance = Math.min(distance, getRowDistance(row, other));
            return distance;
          });
          const breaks = [...new Set([0, ...nearest, ...predictionDistances])].sort(
            (first, second) => first - second
          );
          let objective = 0;
          for (let index = 1; index < breaks.length; index++) {
            const previous = breaks[index - 1];
            const trainingBelow = nearest.filter(distance => distance <= previous).length;
            const predictionBelow = predictionDistances.filter(
              distance => distance <= previous
            ).length;
            objective +=
              Math.abs(trainingBelow / rowCount - predictionBelow / predictionCount) *
              (breaks[index] - previous);
          }
          if (objective < best) {
            best = objective;
            selected = candidate;
          }
        }
        diagnostics[1] = candidateCount;
        if (selected === NONE) diagnostics[0] = 2;
        else {
          diagnostics[2] = best;
          diagnostics[3] = selected;
          result.folds.set(candidateRows.subarray(selected * rowCount, (selected + 1) * rowCount));
        }
      }
    }
  }
  for (let fold = 0; fold < foldCount; fold++) {
    let validCount = 0;
    let trainCount = 0;
    let testCount = 0;
    for (let row = 0; row < rowCount; row++) {
      const assigned = result.folds[row];
      const test = assigned === fold;
      let train = assigned < foldCount && !test && diagnostics[0] === 0;
      if (assigned < foldCount) validCount++;
      if (train)
        for (let other = 0; other < rowCount; other++)
          if (result.folds[other] === fold) {
            if (
              isWithinValidationRadiusOnCPU(
                positions[row * 2],
                positions[row * 2 + 1],
                positions[other * 2],
                positions[other * 2 + 1],
                options.bufferDistance ?? 0
              ) ||
              (design?.mode === 'nndm' &&
                (adaptiveNeighbors[other] === NONE ||
                  isWithinValidationNeighborOnCPU(
                    positions[row * 2],
                    positions[row * 2 + 1],
                    positions[other * 2],
                    positions[other * 2 + 1],
                    positions[adaptiveNeighbors[other] * 2],
                    positions[adaptiveNeighbors[other] * 2 + 1]
                  )))
            ) {
              train = false;
              break;
            }
          }
      result.training[fold * rowCount + row] = Number(train);
      result.testing[fold * rowCount + row] = Number(test);
      trainCount += Number(train);
      testCount += Number(test);
    }
    result.counts.set([trainCount, testCount, validCount - trainCount - testCount], fold * 3);
    result.status[fold] = Number(trainCount === 0) | (testCount === 0 ? 2 : 0);
  }
  result.diagnostics?.set(diagnostics);
  if (design?.mode === 'nndm')
    for (let row = 0; row < rowCount; row++)
      if (diagnostics[row + 4] === Infinity) result.diagnostics![row + 4] = NO_DISTANCE;
  return result;
}

/**
 * CPU fold design. kNNDM accepts at most 16,777,216 candidates. Fold masks, fold counts,
 * environmental feature cells and kNNDM candidate cells each cap at 4,294,967,295 entries.
 * The explicit iteration budget still applies to environmental clustering.
 */
export class CPUSpatialCrossValidationOperation extends GPUCPUOperation {
  constructor(props: SpatialCrossValidationOperationProps) {
    props = snapshotSpatialCrossValidationProps(props);
    if (props.design)
      validateSpatialValidationDesignMode(
        props.id ?? 'spatial-cross-validation',
        props.design.mode
      );
    const id = props.id ?? 'spatial-cross-validation';
    validateOptions(props, props.rowCount);
    if (props.design?.mode === 'nndm' || props.design?.mode === 'knndm')
      validateCapacity(id, 'design.predictionRowCount', props.design.predictionRowCount);
    const operation = new GPUSpatialCrossValidationOperation(props);
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
        const design = props.design;
        const predictionPositions =
          design?.mode === 'nndm' || design?.mode === 'knndm'
            ? getCPUOperationFloat32Rows(id, inputs, 'predictionPositions', 'float32x2')
            : undefined;
        if (
          predictionPositions &&
          (design?.mode === 'nndm' || design?.mode === 'knndm') &&
          predictionPositions.length !== design.predictionRowCount * 2
        )
          throw new Error(`${id} predictionPositions length must equal predictionRowCount * 2`);
        const result = computeSpatialCrossValidationOnCPU(
          {
            positions,
            blocks: getCPUOperationUint32Rows(id, inputs, 'blocks'),
            mask:
              props.mask !== undefined ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined,
            features:
              design?.mode === 'environmental'
                ? getCPUOperationFloat32Rows(id, inputs, 'features')
                : undefined,
            predictionPositions,
            candidates:
              design?.mode === 'knndm' && design.candidates !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'candidates')
                : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUSpatialCrossValidationResult]!
          ])
        );
      }
    });
  }
}

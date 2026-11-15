// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  validateViewLength
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  createPointCloudGridIndex,
  validatePointCloudSearchGrid,
  type PointCloudGrid
} from './pointcloud-grid-index';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';
import {
  GPU_POINT_CLOUD_MAXIMUM_CHANGE_CELLS_PER_CORE,
  GPU_POINT_CLOUD_MAXIMUM_SAMPLING_COVARIANCE_POINTS,
  GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK
} from './pointcloud-limits';

export type GPUPointCloudChangeProps = {
  /** Per-frame support and uncertainty, packed by getGPUPointCloudChangeParameterValues. */
  parameters?: GraphDataView<'uint32'>;
  id?: string;
  reference: GraphDataView<'float32x3'>;
  comparison: GraphDataView<'float32x3'>;
  cores: GraphDataView<'float32x3'>;
  normals: GraphDataView<'float32x3'>;
  /** Explicit aligned-frame certification, or a column-major comparison-to-reference rigid transform (for example ICP output). */
  alignment: 'already-aligned' | {transform: GraphDataView<'float32'>};
  radius: number;
  halfDepth: number;
  minimumSupport?: number;
  /** Nonnegative registration uncertainty in coordinate units, added to the sampling standard error. */
  registrationError?: number;
  /** Optional aligned-frame translation covariance [xx,xy,xz,yy,yz,zz]; projected along each normal. */
  registrationCovariance?: readonly [number, number, number, number, number, number];
  /** Linear matches the M3C2 convention; quadrature combines independent Gaussian registration and sampling errors. */
  uncertaintyCombination?: 'linear' | 'quadrature';
  /** Two-sided normal quantile, default1.96 (95%); caller specifies their confidence convention. */
  confidenceMultiplier?: number;
  /** Optional known covariance of normal-direction measurement error: sigma² exp(-distance²/(2 length²)).
   * Uses all within-cylinder pairs exactly; variances are supplied separately from empirical roughness. */
  samplingCovariance?: {
    referenceVariance: number;
    comparisonVariance: number;
    correlationLength: number;
  };
  /** Signed comparison-minus-reference displacement, lower 95% bound, upper 95% bound. */
  changes: GraphDataView<'float32x3'>;
  /** Reference roughness, comparison roughness (sample standard deviations), 95% detection limit. */
  roughness: GraphDataView<'float32x3'>;
  /** Interleaved reference/comparison counts for each core. */
  counts: GraphDataView<'uint32'>;
  /**
   * Exact 3D index of `reference` and the aligned `comparison`, rebuilt each encoding. Each core
   * visits only the cells its cylinder can touch, so the cost follows the points near the cores
   * instead of every point. Required above `GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK`
   * core-point pairs. Points outside the half-open grid do not participate, so the grid must cover
   * both clouds (after alignment) for the result to equal the unindexed scan. Counts and
   * statistics equal the unindexed result up to float32 summation order.
   */
  grid?: PointCloudGrid;
  /**
   * Per-core status code (a value, not a bit set): `GPU_POINT_CLOUD_CHANGE_STATUS_VALID` (0) the
   * change and its bounds are published, `..._INVALID_CORE` (1) the core or its normal is not
   * finite or the normal is not near unit length, `..._INSUFFICIENT_SUPPORT` (2) fewer than
   * `minimumSupport` points of either epoch lie in the cylinder (counts are still published),
   * `..._NONFINITE_UNCERTAINTY` (3) a moment or the confidence interval overflowed float32.
   * Outputs other than `counts` are zero for any nonzero code. In grid mode the flag
   * `GPU_POINT_CLOUD_CHANGE_STATUS_GRID_INCOMPLETE` (0x100) is OR-ed onto the code when a cell
   * the cylinder needs lies outside the grid (not onto `..._INVALID_CORE`); mask it off to read
   * the code. A flagged `..._VALID` row is published but its counts may be incomplete.
   */
  status: GraphDataView<'uint32'>;
};

/**
 * `GPUPointCloudChange` status flag, OR-ed onto any code except `..._INVALID_CORE` in grid mode:
 * at least one grid cell the core's cylinder needs lies outside the grid, so points there were
 * dropped and the counts and statistics may be incomplete.
 */
export const GPU_POINT_CLOUD_CHANGE_STATUS_GRID_INCOMPLETE = 0x100;
/** `GPUPointCloudChange` status code: valid change and bounds. */
export const GPU_POINT_CLOUD_CHANGE_STATUS_VALID = 0;
/** `GPUPointCloudChange` status code: nonfinite core or unusable normal. */
export const GPU_POINT_CLOUD_CHANGE_STATUS_INVALID_CORE = 1;
/** `GPUPointCloudChange` status code: too few points of an epoch in the cylinder. */
export const GPU_POINT_CLOUD_CHANGE_STATUS_INSUFFICIENT_SUPPORT = 2;
/** `GPUPointCloudChange` status code: the uncertainty is not finite in float32. */
export const GPU_POINT_CLOUD_CHANGE_STATUS_NONFINITE_UNCERTAINTY = 3;

/** Twelve uint32 words: support followed by float32 bit patterns for uncertainty settings. */
export const GPU_POINT_CLOUD_CHANGE_PARAMETER_LENGTH = 12;

export type GPUPointCloudChangeSettings = Pick<
  GPUPointCloudChangeProps,
  | 'minimumSupport'
  | 'registrationError'
  | 'registrationCovariance'
  | 'confidenceMultiplier'
  | 'samplingCovariance'
>;

/**
 * Pack per-frame support and uncertainty for a caller-owned uint32 GPUParameterBuffer. The first
 * word is an exact uint32 minimumSupport; the remaining words hold float32 bit patterns. Covariance
 * model presence and the uncertainty combination are compile-time contributor settings.
 */
export function getGPUPointCloudChangeParameterValues(
  settings: GPUPointCloudChangeSettings = {}
): Uint32Array {
  validatePointCloudChangeSettings('gpu-pointcloud-change', settings);
  const words = new Uint32Array(GPU_POINT_CLOUD_CHANGE_PARAMETER_LENGTH);
  const values = new Float32Array(words.buffer);
  words[0] = settings.minimumSupport ?? 3;
  values[1] = (settings.registrationError ?? 0) ** 2;
  values[2] = settings.confidenceMultiplier ?? 1.96;
  values.set(settings.registrationCovariance ?? [0, 0, 0, 0, 0, 0], 3);
  values[9] = settings.samplingCovariance?.referenceVariance ?? 0;
  values[10] = settings.samplingCovariance?.comparisonVariance ?? 0;
  values[11] = 2 * (settings.samplingCovariance?.correlationLength ?? 1) ** 2;
  return words;
}

function validatePointCloudChangeSettings(
  identifier: string,
  props: GPUPointCloudChangeSettings
): void {
  validateCapacity(identifier, 'minimumSupport', props.minimumSupport ?? 3, 2);
  validateRequirement(
    identifier,
    (props.registrationError ?? 0) >= 0 &&
      Number.isFinite(Math.fround((props.registrationError ?? 0) ** 2)),
    'registrationError must be nonnegative with a finite float32 square'
  );
  validateRequirement(
    identifier,
    (props.confidenceMultiplier ?? 1.96) > 0 &&
      Number.isFinite(Math.fround(props.confidenceMultiplier ?? 1.96)),
    'confidenceMultiplier must be positive and finite in float32'
  );
  if (props.samplingCovariance) {
    const covariance = props.samplingCovariance;
    validateRequirement(
      identifier,
      [covariance.referenceVariance, covariance.comparisonVariance].every(
        value => value >= 0 && Number.isFinite(Math.fround(value))
      ),
      'samplingCovariance variances must be nonnegative and finite in float32'
    );
    validateRequirement(
      identifier,
      covariance.correlationLength > 0 &&
        Number.isFinite(Math.fround(2 * covariance.correlationLength ** 2)) &&
        Math.fround(covariance.correlationLength ** 2) > 0,
      'samplingCovariance.correlationLength must be positive with a finite nonzero float32 square'
    );
  }
  if (props.registrationCovariance) {
    validateRequirement(
      identifier,
      props.registrationCovariance.every(value => Number.isFinite(Math.fround(value))),
      'registrationCovariance must hold finite numbers'
    );
    // Normalize before testing principal minors so validity is independent of coordinate units.
    const scale = Math.max(...props.registrationCovariance.map(Math.abs));
    const [xx, xy, xz, yy, yz, zz] = props.registrationCovariance.map(value =>
      scale > 0 ? value / scale : 0
    );
    const tolerance = 32 * Number.EPSILON;
    validateRequirement(
      identifier,
      xx >= 0 &&
        yy >= 0 &&
        zz >= 0 &&
        xx * yy - xy * xy >= -tolerance &&
        xx * zz - xz * xz >= -tolerance &&
        yy * zz - yz * yz >= -tolerance,
      'registrationCovariance must be positive semidefinite'
    );
    validateRequirement(
      identifier,
      xx * yy * zz + 2 * xy * xz * yz - xx * yz * yz - yy * xz * xz - zz * xy * xy >= -tolerance,
      'registrationCovariance must be positive semidefinite'
    );
  }
}

/**
 * M3C2-style cylindrical mean differences with explicit orientation, alignment and support.
 *
 * Precision: all positions are float32. At projected magnitudes (~1e6 m) absolute coordinates are
 * quantized to ~0.06 m, so make every cloud, core and alignment transform relative to one shared
 * float64 origin on the CPU (`getPointCloudLocalOrigin`, `makePointCloudLocalPositions`) and read
 * ICP transforms in that same local frame.
 *
 * Work: without `grid` every core scans every reference and comparison point, so `cores *
 * (reference + comparison)` is limited to `GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK` and the
 * constructor asks for a `grid` above it. With `grid` each core visits only the cells its cylinder
 * can touch. `samplingCovariance` additionally visits in-cylinder pairs and limits each cloud to
 * `GPU_POINT_CLOUD_MAXIMUM_SAMPLING_COVARIANCE_POINTS` points.
 * Per-frame: input contents, alignment transform view, and parameters (support and uncertainty).
 * Compile-time: radius, halfDepth (bounded candidate loops), grid, covariance-model presence,
 * uncertaintyCombination, optional alignment view and view shapes. Without parameters, support
 * and uncertainty props retain their constructor-time behavior.
 */
export class GPUPointCloudChange implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUPointCloudChangeProps) {
    this.id = props.id ?? 'gpu-pointcloud-change';
    validateRequirement(
      this.id,
      props.alignment === 'already-aligned' || props.alignment.transform.length === 16,
      "alignment must be 'already-aligned' or hold a transform of 16 rows"
    );
    validateRequirement(
      this.id,
      props.radius > 0 &&
        Number.isFinite(Math.fround(props.radius ** 2)) &&
        Math.fround(props.radius ** 2) > 0,
      'radius must be positive with a finite nonzero float32 square'
    );
    validateRequirement(
      this.id,
      props.halfDepth > 0 && Number.isFinite(Math.fround(props.halfDepth)),
      'halfDepth must be positive and finite in float32'
    );
    validatePointCloudChangeSettings(this.id, props);
    if (props.parameters) {
      validateViewLength(
        this.id,
        'parameters',
        props.parameters,
        GPU_POINT_CLOUD_CHANGE_PARAMETER_LENGTH
      );
    }
    if (props.samplingCovariance) {
      // All-pairs sampling covariance is quadratic per core; bound each cloud so one thread stays short.
      validateMaximumCount(
        this.id,
        'samplingCovariance points per cloud',
        Math.max(props.reference.length, props.comparison.length),
        GPU_POINT_CLOUD_MAXIMUM_SAMPLING_COVARIANCE_POINTS
      );
    }
    if (props.grid) {
      validatePointCloudSearchGrid(this.id, 'grid', props.grid);
      // Cells one core visits: the cylinder box is at most sqrt(halfDepth^2 + radius^2) from the core.
      const cellsPerAxis =
        Math.ceil((2 * Math.hypot(props.halfDepth, props.radius)) / props.grid.cellSize) + 3;
      validateRequirement(
        this.id,
        cellsPerAxis ** 3 <= GPU_POINT_CLOUD_MAXIMUM_CHANGE_CELLS_PER_CORE,
        `each core would visit ${cellsPerAxis ** 3} grid cells, above the ${GPU_POINT_CLOUD_MAXIMUM_CHANGE_CELLS_PER_CORE} limit; enlarge the grid cellSize`
      );
    } else {
      // Exhaustive cylinder scan: pass a grid for larger inputs (contributor contract rule 3).
      const work = props.cores.length * (props.reference.length + props.comparison.length);
      validateRequirement(
        this.id,
        work <= GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK,
        `cores x (reference + comparison) is ${work}, above the ${GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK} limit of a scan without an index; pass grid`
      );
    }
    validateViewLength(this.id, 'normals', props.normals, props.cores.length);
    validateViewLength(this.id, 'changes', props.changes, props.cores.length);
    validateViewLength(this.id, 'roughness', props.roughness, props.cores.length);
    validateViewLength(this.id, 'status', props.status, props.cores.length);
    validateViewLength(this.id, 'counts', props.counts, props.cores.length * 2);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      reference,
      comparison,
      cores,
      normals,
      changes,
      roughness,
      counts,
      status,
      grid,
      parameters
    } = this.props;
    const nodes: GPUCommandNode<Parameters>[] = [];
    let alignedComparison = comparison;
    if (this.props.alignment !== 'already-aligned') {
      alignedComparison = createTransientView(
        graph,
        `${this.id}-aligned-comparison`,
        'float32x3',
        comparison.length
      );
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-align`,
          length: comparison.length,
          inputs: {comparison, transform: this.props.alignment.transform},
          outputs: {aligned: alignedComparison},
          body: `let rotation = mat3x3f(vec3f(readTransform(0u),readTransform(1u),readTransform(2u)),vec3f(readTransform(4u),readTransform(5u),readTransform(6u)),vec3f(readTransform(8u),readTransform(9u),readTransform(10u)));
let translation = vec3f(readTransform(12u),readTransform(13u),readTransform(14u));
writeAligned(index,rotation*readComparison(index)+translation);`
        })
      );
    }
    const halfDepth = Math.fround(this.props.halfDepth);
    const radiusSquared = Math.fround(this.props.radius ** 2);
    const minimumSupport = parameters ? 'readParameters(0u)' : `${this.props.minimumSupport ?? 3}u`;
    const samplingCovariance = this.props.samplingCovariance;
    const registrationCovarianceSource = parameters
      ? [3, 4, 5, 4, 6, 7, 5, 7, 8].map(row => `bitcast<f32>(readParameters(${row}u))`).join(',')
      : (() => {
          const [
            firstVariance,
            firstSecondCovariance,
            firstThirdCovariance,
            secondVariance,
            secondThirdCovariance,
            thirdVariance
          ] = this.props.registrationCovariance ?? [0, 0, 0, 0, 0, 0];
          return [
            firstVariance,
            firstSecondCovariance,
            firstThirdCovariance,
            firstSecondCovariance,
            secondVariance,
            secondThirdCovariance,
            firstThirdCovariance,
            secondThirdCovariance,
            thirdVariance
          ]
            .map(Math.fround)
            .join(',');
        })();
    const registrationVarianceSource = `let registrationCovariance = mat3x3f(${registrationCovarianceSource});
let registrationVariance = max(dot(normal,registrationCovariance*normal),0.0)+${parameters ? 'bitcast<f32>(readParameters(1u))' : Math.fround((this.props.registrationError ?? 0) ** 2)};`;

    // One kernel per cloud keeps each under the storage-binding limit of a shader stage (a grid adds
    // two bindings per cloud); the final kernel combines the two clouds' moments.
    const moments = {
      reference: createTransientView(
        graph,
        `${this.id}-reference-moments`,
        'float32',
        cores.length * 3
      ),
      comparison: createTransientView(
        graph,
        `${this.id}-comparison-moments`,
        'float32',
        cores.length * 3
      )
    };
    for (const [slot, name, positions] of [
      [0, 'reference', reference],
      [1, 'comparison', alignedComparison]
    ] as const) {
      // With a grid, the cloud is sorted by cell once and a core visits only the cells its
      // cylinder's bounding box touches; the per-point tests are the same as the full scan.
      const built = grid
        ? createPointCloudGridIndex(graph, `${this.id}-${name}-grid`, positions, grid)
        : undefined;
      if (built) nodes.push(...built.nodes);
      /** Runs `body` once per candidate point of this cloud, with the point row bound to `variable`. */
      const forEachCandidate = (variable: string, suffix: string, body: string): string => {
        if (!grid) {
          return `for (var ${variable} = 0u; ${variable} < ${positions.length}u; ${variable}++) {
${body}
}`;
        }
        return `for (var depth${suffix} = lower.z; depth${suffix} <= upper.z; depth${suffix}++) {
 for (var row${suffix} = lower.y; row${suffix} <= upper.y; row${suffix}++) {
  for (var column${suffix} = lower.x; column${suffix} <= upper.x; column${suffix}++) {
   let cell${suffix} = (u32(depth${suffix}) * ${grid.dimensions[1]}u + u32(row${suffix})) * ${grid.dimensions[0]}u + u32(column${suffix});
   for (var slot${suffix} = readOffsets(cell${suffix}); slot${suffix} < readOffsets(cell${suffix} + 1u); slot${suffix}++) {
    let ${variable} = readSortedIndices(slot${suffix});
${body}
   }
  }
 }
}`;
      };
      const gridBounds = grid
        ? `// Bounding box of the cylinder: half-depth along the normal plus radius across it, one cell of margin.
let halfExtent = vec3f(${halfDepth}) * abs(normal) + ${Math.fround(this.props.radius)} * sqrt(max(vec3f(1.0) - normal * normal, vec3f(0.0)));
let displacement = core - vec3f(${grid.origin.map(Math.fround).join(',')});
// Clamp in floating point before integer conversion so far-away cores cannot overflow cell coordinates.
let lower = vec3i(clamp(floor((displacement - halfExtent) / ${Math.fround(grid.cellSize)}) - vec3f(1.0), vec3f(0.0), vec3f(${grid.dimensions.map(value => `${value}.0`).join(',')})));
let upper = vec3i(clamp(floor((displacement + halfExtent) / ${Math.fround(grid.cellSize)}) + vec3f(1.0), vec3f(-1.0), vec3f(${grid.dimensions.map(value => `${value - 1}.0`).join(',')})));`
        : '';
      const inCylinder = (prefix: string) =>
        `let ${prefix}Point=readPoints(${prefix});let ${prefix}Delta=${prefix}Point-core;let ${prefix}Normal=dot(${prefix}Delta,normal);let ${prefix}Radial=${prefix}Delta-normal*${prefix}Normal;
 if(!(abs(${prefix}Normal)<=${halfDepth})||!(dot(${prefix}Radial,${prefix}Radial)<=${radiusSquared})){continue;}`;
      let correlation = '';
      if (samplingCovariance) {
        const kernel = `exp(-dot(separation,separation)/${parameters ? 'bitcast<f32>(readParameters(11u))' : Math.fround(2 * samplingCovariance.correlationLength ** 2)})`;
        // Unindexed: pairs (first, second > first) counted twice. Indexed: every ordered pair
        // (first, second != first) counted once, which is the same sum.
        const inner = grid
          ? forEachCandidate(
              'second',
              'Second',
              ` if(second==first){continue;}
 ${inCylinder('second')}
 let separation=firstPoint-secondPoint;correlationSum+=${kernel};`
            )
          : `for(var second=first+1u;second<${positions.length}u;second++){
  ${inCylinder('second')}
  let separation=firstPoint-secondPoint;correlationSum+=2.0*${kernel};
 }`;
        correlation = `var correlationSum=0.0;
${forEachCandidate(
  'first',
  'First',
  ` ${inCylinder('first')}
 correlationSum+=1.0;
 ${inner}`
)}
writeMoments(index*3u+2u,correlationSum);`;
      }
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-${name}`,
          length: cores.length,
          inputs: {
            points: positions,
            ...(parameters ? {parameters} : {}),
            cores,
            normals,
            ...(built ? {sortedIndices: built.sortedIndices, offsets: built.offsets} : {})
          },
          outputs: {counts, moments: moments[name]},
          body: `let core = readCores(index); let normal = safeNormal(readNormals(index));
writeCounts(index*2u+${slot}u,0u); writeMoments(index*3u,0.0); writeMoments(index*3u+1u,0.0); writeMoments(index*3u+2u,0.0);
if (!all(abs(core) <= vec3f(3.402823e38)) || dot(normal,normal) < 0.5) { return; }
${gridBounds}
var count = 0u; var mean = 0.0; var variance = 0.0;
${forEachCandidate(
  'point',
  '',
  `  let difference = readPoints(point) - core;
  let distance = dot(difference,normal);
  let radial = difference - normal * distance;
  if (!(abs(distance) <= ${halfDepth}) || !(dot(radial,radial) <= ${radiusSquared})) { continue; }
  count++; let delta = distance - mean; mean += delta / f32(count); variance += delta * (distance - mean);`
)}
writeCounts(index*2u+${slot}u,count); writeMoments(index*3u,mean); writeMoments(index*3u+1u,variance);
// The pair sum is wasted work below the support minimum; the final kernel then reports insufficient support.
if (count < ${minimumSupport}) { return; }
${correlation}`
        })
      );
    }
    const outsideBit = grid
      ? `// Cells the cylinder needs (without the search margin) that lie outside the grid.
let outsideHalfExtent = vec3f(${halfDepth}) * abs(normal) + ${Math.fround(this.props.radius)} * sqrt(max(vec3f(1.0) - normal * normal, vec3f(0.0)));
let outsideDisplacement = core - vec3f(${grid.origin.map(Math.fround).join(',')});
let outsideLower = floor((outsideDisplacement - outsideHalfExtent) / ${Math.fround(grid.cellSize)});
let outsideUpper = floor((outsideDisplacement + outsideHalfExtent) / ${Math.fround(grid.cellSize)});
let outsideBit = select(0u, ${GPU_POINT_CLOUD_CHANGE_STATUS_GRID_INCOMPLETE}u, any(outsideLower < vec3f(0.0)) || any(outsideUpper > vec3f(${grid.dimensions.map(value => `${value - 1}.0`).join(',')})));`
      : 'let outsideBit = 0u;';
    const coreSettings = parameters
      ? createTransientView(graph, `${this.id}-core-settings`, 'float32x3', cores.length)
      : undefined;
    if (coreSettings) {
      // Precompute projection and support context to keep the final stage at eight storage
      // bindings even with resident parameters (two moment inputs, counts, settings, parameters,
      // and three outputs). The settings view has the same shape as cores.
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-core-settings`,
          length: cores.length,
          inputs: {cores, normals, parameters: parameters!},
          outputs: {coreSettings},
          body: `let core = readCores(index); let normal = safeNormal(readNormals(index));
let valid = all(abs(core) <= vec3f(3.402823e38)) && dot(normal,normal) >= 0.5;
${outsideBit}
${registrationVarianceSource}
writeCoreSettings(index, vec3f(registrationVariance, f32(outsideBit), select(0.0, 1.0, valid)));`
        })
      );
    }
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: this.id,
        length: cores.length,
        inputs: {
          referenceMoments: moments.reference,
          comparisonMoments: moments.comparison,
          counts,
          ...(coreSettings ? {coreSettings} : {cores, normals}),
          ...(parameters ? {parameters} : {})
        },
        outputs: {changes, roughness, status},
        body: `${coreSettings ? 'let settings = readCoreSettings(index);' : 'let core = readCores(index); let normal = safeNormal(readNormals(index));'}
writeChanges(index,vec3f(0.0)); writeRoughness(index,vec3f(0.0));
if (${coreSettings ? 'settings.z == 0.0' : '!all(abs(core) <= vec3f(3.402823e38)) || dot(normal,normal) < 0.5'}) { writeStatus(index,${GPU_POINT_CLOUD_CHANGE_STATUS_INVALID_CORE}u); return; }
${coreSettings ? 'let outsideBit = u32(settings.y);' : outsideBit}
let referenceCount = readCounts(index*2u); let comparisonCount = readCounts(index*2u+1u);
if (referenceCount < ${minimumSupport} || comparisonCount < ${minimumSupport}) { writeStatus(index,${GPU_POINT_CLOUD_CHANGE_STATUS_INSUFFICIENT_SUPPORT}u | outsideBit); return; }
let referenceMean = readReferenceMoments(index*3u); let comparisonMean = readComparisonMoments(index*3u);
let referenceVarianceValue = max(readReferenceMoments(index*3u+1u) / f32(referenceCount - 1u),0.0);
let comparisonVarianceValue = max(readComparisonMoments(index*3u+1u) / f32(comparisonCount - 1u),0.0);
let samplingVariance=${samplingCovariance ? `${parameters ? 'bitcast<f32>(readParameters(9u))' : Math.fround(samplingCovariance.referenceVariance)}*readReferenceMoments(index*3u+2u)/(f32(referenceCount)*f32(referenceCount))+${parameters ? 'bitcast<f32>(readParameters(10u))' : Math.fround(samplingCovariance.comparisonVariance)}*readComparisonMoments(index*3u+2u)/(f32(comparisonCount)*f32(comparisonCount))` : 'referenceVarianceValue/f32(referenceCount)+comparisonVarianceValue/f32(comparisonCount)'};
${coreSettings ? 'let registrationVariance = settings.x;' : registrationVarianceSource}
let detection=${parameters ? 'bitcast<f32>(readParameters(2u))' : Math.fround(this.props.confidenceMultiplier ?? 1.96)}*${this.props.uncertaintyCombination === 'quadrature' ? 'sqrt(samplingVariance+registrationVariance)' : '(sqrt(samplingVariance)+sqrt(registrationVariance))'};
let difference = comparisonMean - referenceMean;
// A finite input can still overflow an accumulated moment or published confidence interval.
if (any((bitcast<vec4u>(vec4f(referenceVarianceValue,comparisonVarianceValue,samplingVariance,registrationVariance)) & vec4u(0x7f800000u)) == vec4u(0x7f800000u)) ||
    any((bitcast<vec4u>(vec4f(detection,difference,difference-detection,difference+detection)) & vec4u(0x7f800000u)) == vec4u(0x7f800000u))) {
  writeStatus(index,${GPU_POINT_CLOUD_CHANGE_STATUS_NONFINITE_UNCERTAINTY}u | outsideBit); return;
}
writeChanges(index,vec3f(difference,difference-detection,difference+detection));
writeRoughness(index,vec3f(sqrt(referenceVarianceValue),sqrt(comparisonVarianceValue),detection)); writeStatus(index,${GPU_POINT_CLOUD_CHANGE_STATUS_VALID}u | outsideBit);`
      })
    );
    return nodes;
  }
}

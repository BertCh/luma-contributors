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
  validateFiniteNumber,
  validateRequirement,
  validateViewLength
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  getPlaneMomentsSource,
  getPlaneSolveSource,
  POINT_CLOUD_PLANE_ICP_BLOCK_WORDS
} from './pointcloud-plane-icp';
import {
  createPointCloudGridIndex,
  validatePointCloudSearchGrid,
  type PointCloudGrid
} from './pointcloud-grid-index';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';
import {
  GPU_POINT_CLOUD_MAXIMUM_ICP_ITERATIONS,
  GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK
} from './pointcloud-limits';
import {
  getPointCloudAbsoluteTransform,
  getPointCloudLocalTransform
} from './pointcloud-local-origin';

export type GPUPointCloudICPProps = {
  id?: string;
  source: GraphDataView<'float32x3'>;
  target: GraphDataView<'float32x3'>;
  /**
   * Float64 origin that `source`, `target` and `targetGrid` were made relative to on the CPU (see
   * `makePointCloudLocalPositions`). Absolute float32 positions lose millimetres beyond ~1e5 m;
   * with an origin the GPU runs in the local frame, `initialTransform` stays absolute, and the
   * published `transform` is local: convert it with {@link GPUPointCloudICP.getAbsoluteTransform}.
   */
  origin?: readonly [number, number, number];
  /** Column-major rigid transform, source to target frame (absolute when `origin` is set). Defaults to identity. */
  initialTransform?: readonly number[];
  maximumDistance: number;
  /**
   * Exact static target index, rebuilt each encoding. Target points outside this extent are excluded explicitly.
   * Required above `GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK` source-target pairs: without it every
   * source point scans every target point and the constructor refuses larger inputs.
   */
  targetGrid?: PointCloudGrid;
  /**
   * Default point-to-point. Point-to-plane requires finite target normals and a full-rank
   * six-parameter system. Its Gauss-Newton moments are taken about the centroid of the matched
   * source points and the system is scaled to a unit diagonal, so the rank test does not depend on
   * the coordinate unit or on the distance from the origin.
   */
  metric?: 'point-to-point' | 'point-to-plane';
  targetNormals?: GraphDataView<'float32x3'>;
  /** Iteratively reweighted rigid least squares; scale is in coordinate units. Default is unweighted. */
  robust?: {method: 'huber' | 'tukey'; scale: number};
  iterations?: number;
  translationTolerance?: number;
  rotationTolerance?: number;
  /** Column-major 4x4 rigid transform. */
  transform: GraphDataView<'float32'>;
  /** One vec3 per iteration: pre-update inlier count, RMSE, translation step length. Frozen iterations repeat their residual. */
  history: GraphDataView<'float32x3'>;
  /** Final-transform correspondence flags in original source order. */
  inlierMask: GraphDataView<'uint32'>;
  /**
   * One row holding a status code (a value, not a bit set), published after the last iteration:
   * `GPU_POINT_CLOUD_ICP_STATUS_CONVERGED` (0) both steps fell below their tolerances,
   * `..._ITERATION_LIMIT` (1) the iteration budget ran out first (the transform is still the last
   * iterate), `..._INSUFFICIENT_MATCHES` (2) fewer than three inlier matches (six under point-to-plane),
   * `..._DEGENERATE` (3) the solution is not unique (the two largest eigenvalues of the Horn matrix
   * agree to 1e-6, or a pivot of the point-to-plane normal matrix, scaled to a unit diagonal, is at
   * or below 1e-5), `..._EIGENSOLVE_EXHAUSTED` (4) the point-to-point Jacobi eigensolve did not
   * converge within 48 rotations (one pivot each). After a code of 2, 3 or 4 the transform is the
   * last good iterate.
   */
  status: GraphDataView<'uint32'>;
};

/** `GPUPointCloudICP` status code: converged within both tolerances. */
export const GPU_POINT_CLOUD_ICP_STATUS_CONVERGED = 0;
/** `GPUPointCloudICP` status code: the iteration budget was exhausted. */
export const GPU_POINT_CLOUD_ICP_STATUS_ITERATION_LIMIT = 1;
/** `GPUPointCloudICP` status code: fewer than three inlier matches. */
export const GPU_POINT_CLOUD_ICP_STATUS_INSUFFICIENT_MATCHES = 2;
/** `GPUPointCloudICP` status code: degenerate (non-unique) rotation. */
export const GPU_POINT_CLOUD_ICP_STATUS_DEGENERATE = 3;
/** `GPUPointCloudICP` status code: the eigensolve did not converge. */
export const GPU_POINT_CLOUD_ICP_STATUS_EIGENSOLVE_EXHAUSTED = 4;

const IDENTITY_TRANSFORM = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Bounded GPU point-to-point ICP with distance rejection and Horn's rigid least-squares solve. */
export class GPUPointCloudICP implements GPUCommandNodeProducer {
  readonly id: string;
  readonly iterations: number;
  /** Initial transform in the local frame of `origin` (identity origin when omitted). */
  private readonly localInitialTransform: readonly number[];
  constructor(readonly props: GPUPointCloudICPProps) {
    this.id = props.id ?? 'gpu-pointcloud-icp';
    this.iterations = props.iterations ?? 20;
    validateCapacity(
      this.id,
      'iterations',
      this.iterations,
      1,
      GPU_POINT_CLOUD_MAXIMUM_ICP_ITERATIONS
    );
    validateFiniteNumber(this.id, 'maximumDistance', props.maximumDistance);
    validateRequirement(
      this.id,
      props.maximumDistance > 0 &&
        Math.fround(props.maximumDistance ** 2) > 0 &&
        Number.isFinite(Math.fround(props.maximumDistance ** 2)),
      'maximumDistance must be positive with a finite nonzero float32 square'
    );
    validateFiniteNumber(this.id, 'translationTolerance', props.translationTolerance ?? 1e-5, 0);
    validateFiniteNumber(this.id, 'rotationTolerance', props.rotationTolerance ?? 1e-5, 0);
    validateViewLength(this.id, 'transform', props.transform, 16);
    validateViewLength(this.id, 'history', props.history, this.iterations);
    validateViewLength(this.id, 'inlierMask', props.inlierMask, props.source.length);
    validateViewLength(this.id, 'status', props.status, 1);
    if (props.metric === 'point-to-plane') {
      validateRequirement(
        this.id,
        props.targetNormals?.length === props.target.length,
        'targetNormals must hold one row per target point for the point-to-plane metric'
      );
    }
    if (props.robust) {
      validateFiniteNumber(this.id, 'robust.scale', props.robust.scale);
      validateRequirement(
        this.id,
        Math.fround(props.robust.scale) > 0,
        'robust.scale must be positive'
      );
      if (props.robust.method === 'tukey') {
        // Tukey weights divide by the squared scale in float32.
        validateRequirement(
          this.id,
          Math.fround(props.robust.scale ** 2) > 0 &&
            Number.isFinite(Math.fround(props.robust.scale ** 2)),
          'robust.scale must have a finite nonzero float32 square for the tukey method'
        );
      }
    }
    const initial = props.initialTransform ?? IDENTITY_TRANSFORM;
    validateRequirement(
      this.id,
      initial.length === 16 && initial.every(value => Number.isFinite(Math.fround(value))),
      'initialTransform must hold 16 finite numbers'
    );
    validateRequirement(
      this.id,
      initial[3] === 0 && initial[7] === 0 && initial[11] === 0 && initial[15] === 1,
      'initialTransform must be a column-major rigid transform with bottom row 0, 0, 0, 1'
    );
    // Initial rotation must be orthonormal and proper, not a scaling/reflection matrix.
    for (let first = 0; first < 3; first++)
      for (let second = 0; second < 3; second++) {
        let product = 0;
        for (let axis = 0; axis < 3; axis++)
          product += initial[first * 4 + axis] * initial[second * 4 + axis];
        validateRequirement(
          this.id,
          Math.abs(product - Number(first === second)) < 1e-4,
          'initialTransform rotation must be orthonormal'
        );
      }
    const determinant =
      initial[0] * (initial[5] * initial[10] - initial[9] * initial[6]) -
      initial[4] * (initial[1] * initial[10] - initial[9] * initial[2]) +
      initial[8] * (initial[1] * initial[6] - initial[5] * initial[2]);
    validateRequirement(
      this.id,
      determinant > 0,
      'initialTransform rotation must not be a reflection'
    );
    if (props.origin) {
      validateRequirement(
        this.id,
        props.origin.every(Number.isFinite),
        'origin must hold finite numbers'
      );
    }
    this.localInitialTransform = props.origin
      ? getPointCloudLocalTransform(initial, props.origin)
      : initial;
    if (props.targetGrid) {
      validatePointCloudSearchGrid(this.id, 'targetGrid', props.targetGrid);
    } else {
      // Brute force is never the silent default at scale (contributor contract rule 3).
      validateRequirement(
        this.id,
        props.source.length * props.target.length <= GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK,
        `source x target is ${props.source.length * props.target.length}, above the ${GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK} limit of a search without an index; pass targetGrid`
      );
    }
  }
  /** Converts a published local-frame `transform` readback to the absolute frame of `origin`. */
  getAbsoluteTransform(localTransform: ArrayLike<number>): number[] {
    return this.props.origin
      ? getPointCloudAbsoluteTransform(localTransform, this.props.origin)
      : Array.from(localTransform);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {source, target, transform, history, inlierMask, status} = this.props;
    const states = [0, 1].map(value =>
      createTransientView(graph, `${this.id}-state-${value}`, 'float32', 17)
    );
    const matches = createTransientView(graph, `${this.id}-matches`, 'float32x3', source.length);
    const pointToPlane = this.props.metric === 'point-to-plane';
    const matchIds = pointToPlane
      ? createTransientView(graph, `${this.id}-match-ids`, 'uint32', source.length)
      : undefined;
    const weights = createTransientView(graph, `${this.id}-weights`, 'float32', source.length);
    const blockCount = Math.ceil(source.length / 256);
    const moments = createTransientView(
      graph,
      `${this.id}-moments`,
      'float32',
      blockCount * (pointToPlane ? POINT_CLOUD_PLANE_ICP_BLOCK_WORDS : 18)
    );
    const initial = this.localInitialTransform;
    const nodes = getPointCloudKernelNodes(graph, {
      id: `${this.id}-initialize`,
      length: 1,
      inputs: {},
      outputs: {state: states[0]},
      body: `${initial.map((value, index) => `writeState(${index}u, ${Math.fround(value)});`).join('\n')} writeState(16u, 1.0);`
    });
    const loadTransform = `let rotation = mat3x3f(vec3f(readState(0u),readState(1u),readState(2u)), vec3f(readState(4u),readState(5u),readState(6u)), vec3f(readState(8u),readState(9u),readState(10u)));
let translation = vec3f(readState(12u),readState(13u),readState(14u));`;
    const indexed = this.props.targetGrid
      ? createPointCloudGridIndex(graph, `${this.id}-target-index`, target, this.props.targetGrid)
      : undefined;
    if (indexed) nodes.push(...indexed.nodes);
    let candidateStart = `for (var candidate=0u;candidate<${target.length}u;candidate++){`;
    let candidateEnd = '}';
    if (indexed && this.props.targetGrid) {
      const grid = this.props.targetGrid;
      // Clamp in floating point before integer conversion; large radius/cell ratios and
      // queries outside the grid must not overflow signed cell coordinates.
      candidateStart = `let displacement=position-vec3f(${grid.origin.map(Math.fround).join(',')});
let lower=vec3i(clamp(floor((displacement-vec3f(${Math.fround(this.props.maximumDistance)}))/${Math.fround(grid.cellSize)})-vec3f(1.0),vec3f(0.0),vec3f(${grid.dimensions.join(',')})));
let upper=vec3i(clamp(floor((displacement+vec3f(${Math.fround(this.props.maximumDistance)}))/${Math.fround(grid.cellSize)})+vec3f(1.0),vec3f(-1.0),vec3f(${grid.dimensions.map(value => value - 1).join(',')})));
for(var depth=lower.z;depth<=upper.z;depth++){for(var row=lower.y;row<=upper.y;row++){for(var column=lower.x;column<=upper.x;column++){
let cell=(u32(depth)*${grid.dimensions[1]}u+u32(row))*${grid.dimensions[0]}u+u32(column);
for(var slot=readOffsets(cell);slot<readOffsets(cell+1u);slot++){let candidate=readSortedIndices(slot);`;
      candidateEnd = '}}}}';
    }
    const robust = this.props.robust;
    const weightExpression = !robust
      ? '1.0'
      : robust.method === 'huber'
        ? `min(1.0,${Math.fround(robust.scale)}/max(sqrt(distanceSquared),1e-30))`
        : `pow(max(0.0,1.0-distanceSquared/${Math.fround(robust.scale ** 2)}),2.0)`;
    const correspondence = (iteration: number) =>
      getPointCloudKernelNodes(graph, {
        id: `${this.id}-match-${iteration}`,
        length: source.length,
        inputs: {
          source,
          targetPositions: target,
          state: states[iteration % 2],
          ...(indexed ? {sortedIndices: indexed.sortedIndices, offsets: indexed.offsets} : {})
        },
        outputs: {matches, weights, ...(matchIds ? {matchIds} : {inlierMask})},
        body: `${loadTransform}
let position = rotation * readSource(index) + translation;
var nearest=vec3f(0.0);var nearestId=0xffffffffu;var distanceSquared:f32=${Math.fround(this.props.maximumDistance ** 2)};
if(all(abs(position)<=vec3f(3.402823e38))){
 ${candidateStart}
 let point=readTargetPositions(candidate);let difference=point-position;let distance=dot(difference,difference);
 if(distance<=distanceSquared&&(distance<distanceSquared||candidate<nearestId)){nearestId=candidate;nearest=point;distanceSquared=distance;}
 ${candidateEnd}
}
let weight=select(0.0,${weightExpression},nearestId!=0xffffffffu);
writeMatches(index,nearest);writeWeights(index,weight);${matchIds ? 'writeMatchIds(index,nearestId);' : 'writeInlierMask(index,u32(weight>0.0));'}`
      });
    for (let iteration = 0; iteration < this.iterations; iteration++) {
      nodes.push(...correspondence(iteration));
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-moments-${iteration}`,
          length: blockCount,
          inputs: {
            source,
            matches,
            weights,
            state: states[iteration % 2],
            ...(matchIds ? {matchIds, targetNormals: this.props.targetNormals!} : {})
          },
          outputs: {moments},
          body: pointToPlane
            ? getPlaneMomentsSource(loadTransform, source.length)
            : `${loadTransform}
var count=0u;var weightSum=0.0;var residualSum=0.0;var sourceMean=vec3f(0.0);var targetMean=vec3f(0.0);
var covariance=mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));
for(var point=index*256u;point<min((index+1u)*256u,${source.length}u);point++){
 let weight=readWeights(point);if(weight<=0.0){continue;}count++;
 let transformed=rotation*readSource(point)+translation;let matched=readMatches(point);
 let sourceDelta=transformed-sourceMean;weightSum+=weight;sourceMean+=sourceDelta*(weight/weightSum);targetMean+=(matched-targetMean)*(weight/weightSum);
 let targetDelta=matched-targetMean;
 covariance+=mat3x3f(sourceDelta*targetDelta.x,sourceDelta*targetDelta.y,sourceDelta*targetDelta.z)*weight;
 let difference=matched-transformed;residualSum+=weight*dot(difference,difference);
}
let base=index*18u;writeMoments(base,f32(count));writeMoments(base+1u,weightSum);writeMoments(base+2u,residualSum);
for(var axis=0u;axis<3u;axis++){writeMoments(base+3u+axis,sourceMean[axis]);writeMoments(base+6u+axis,targetMean[axis]);}
for(var column=0u;column<3u;column++){for(var row=0u;row<3u;row++){writeMoments(base+9u+column*3u+row,covariance[column][row]);}}`
        })
      );
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-solve-${iteration}`,
          length: 1,
          inputs: {moments, state: states[iteration % 2]},
          outputs: {next: states[(iteration + 1) % 2], history},
          body: pointToPlane
            ? getPlaneSolveSource(
                loadTransform,
                blockCount,
                iteration,
                this.props.translationTolerance ?? 1e-5,
                this.props.rotationTolerance ?? 1e-5
              )
            : `${loadTransform}
for (var entry = 0u; entry < 17u; entry++) { writeNext(entry, readState(entry)); }
var count=0u;var weightSum=0.0;var residualSum=0.0;var sourceMean=vec3f(0.0);var targetMean=vec3f(0.0);
var covariance=mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));
for(var batchIndex=0u;batchIndex<${blockCount}u;batchIndex++){
 let base=batchIndex*18u;let weight=readMoments(base+1u);if(weight<=0.0){continue;}
 count+=u32(readMoments(base));residualSum+=readMoments(base+2u);
 let firstMean=vec3f(readMoments(base+3u),readMoments(base+4u),readMoments(base+5u));
 let secondMean=vec3f(readMoments(base+6u),readMoments(base+7u),readMoments(base+8u));
 let firstDelta=firstMean-sourceMean;let secondDelta=secondMean-targetMean;
 let nextWeight=weightSum+weight;
 covariance+=mat3x3f(firstDelta*secondDelta.x,firstDelta*secondDelta.y,firstDelta*secondDelta.z)*(weightSum*weight/nextWeight);
 for(var column=0u;column<3u;column++){for(var row=0u;row<3u;row++){covariance[column][row]+=readMoments(base+9u+column*3u+row);}}
 sourceMean+=firstDelta*(weight/nextWeight);targetMean+=secondDelta*(weight/nextWeight);weightSum=nextWeight;
}
let residual=sqrt(residualSum/max(weightSum,1e-30));
writeHistory(${iteration}u,vec3f(f32(count),residual,0.0));
if(readState(16u)!=1.0){return;}if(count<3u||weightSum<=0.0){writeNext(16u,${GPU_POINT_CLOUD_ICP_STATUS_INSUFFICIENT_MATCHES}.0);return;}
covariance*=1.0/weightSum;
let trace = covariance[0][0] + covariance[1][1] + covariance[2][2];
let difference = vec3f(covariance[2][1]-covariance[1][2], covariance[0][2]-covariance[2][0], covariance[1][0]-covariance[0][1]);
var system = mat4x4f(vec4f(trace,difference),
 vec4f(difference.x, covariance[0][0]-covariance[1][1]-covariance[2][2], covariance[1][0]+covariance[0][1], covariance[2][0]+covariance[0][2]),
 vec4f(difference.y, covariance[1][0]+covariance[0][1], -covariance[0][0]+covariance[1][1]-covariance[2][2], covariance[2][1]+covariance[1][2]),
 vec4f(difference.z, covariance[2][0]+covariance[0][2], covariance[2][1]+covariance[1][2], -covariance[0][0]-covariance[1][1]+covariance[2][2]));
var vectors = mat4x4f(vec4f(1,0,0,0),vec4f(0,1,0,0),vec4f(0,0,1,0),vec4f(0,0,0,1));
var converged = false;
for (var jacobiRotation = 0u; jacobiRotation < 48u; jacobiRotation++) {
  var first = 0u; var second = 1u; var scale = 0.0;
  for (var column = 0u; column < 4u; column++) {
    scale = max(scale,abs(system[column][column]));
    for (var row = column + 1u; row < 4u; row++) { if (abs(system[column][row]) > abs(system[first][second])) { first = column; second = row; } }
  }
  if (abs(system[first][second]) <= max(scale * 1e-6,1e-30)) { converged = true; break; }
  let angle = 0.5 * atan2(2.0 * system[first][second], system[second][second] - system[first][first]);
  let cosine = cos(angle); let sine = sin(angle);
  var plane = mat4x4f(vec4f(1,0,0,0),vec4f(0,1,0,0),vec4f(0,0,1,0),vec4f(0,0,0,1));
  plane[first][first] = cosine; plane[second][second] = cosine; plane[first][second] = -sine; plane[second][first] = sine;
  system = transpose(plane) * system * plane; vectors = vectors * plane;
}
if (!converged) { writeNext(16u,${GPU_POINT_CLOUD_ICP_STATUS_EIGENSOLVE_EXHAUSTED}.0); return; }
var largest = 0u;
for (var axis = 1u; axis < 4u; axis++) { if (system[axis][axis] > system[largest][largest]) { largest = axis; } }
var runnerUp = -3.402823e38;
for (var axis = 0u; axis < 4u; axis++) { if (axis != largest) { runnerUp = max(runnerUp, system[axis][axis]); } }
if (!(system[largest][largest] - runnerUp > max(abs(system[largest][largest]) * 1e-6,1e-30))) { writeNext(16u,${GPU_POINT_CLOUD_ICP_STATUS_DEGENERATE}.0); return; }
let quaternion = normalize(vectors[largest]);
let scalar = quaternion.x; let first = quaternion.y; let second = quaternion.z; let third = quaternion.w;
let deltaRotation = mat3x3f(
 vec3f(1.0-2.0*(second*second+third*third), 2.0*(first*second+scalar*third), 2.0*(first*third-scalar*second)),
 vec3f(2.0*(first*second-scalar*third), 1.0-2.0*(first*first+third*third), 2.0*(second*third+scalar*first)),
 vec3f(2.0*(first*third+scalar*second), 2.0*(second*third-scalar*first), 1.0-2.0*(first*first+second*second)));
let deltaTranslation = targetMean - deltaRotation * sourceMean;
let updatedRotation = deltaRotation * rotation; let updatedTranslation = deltaRotation * translation + deltaTranslation;
for (var column = 0u; column < 3u; column++) { for (var row = 0u; row < 3u; row++) { writeNext(column*4u+row, updatedRotation[column][row]); } }
writeNext(12u,updatedTranslation.x); writeNext(13u,updatedTranslation.y); writeNext(14u,updatedTranslation.z);
let step = length(deltaTranslation);
let rotationStep = max(max(length(deltaRotation[0]-vec3f(1,0,0)),length(deltaRotation[1]-vec3f(0,1,0))),length(deltaRotation[2]-vec3f(0,0,1)));
if (step <= ${Math.fround(this.props.translationTolerance ?? 1e-5)} && rotationStep <= ${Math.fround(this.props.rotationTolerance ?? 1e-5)}) { writeNext(16u,${GPU_POINT_CLOUD_ICP_STATUS_CONVERGED}.0); }
writeHistory(${iteration}u,vec3f(f32(count),residual,step));`
        })
      );
    }
    nodes.push(...correspondence(this.iterations));
    if (matchIds)
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-final-plane-mask`,
          length: source.length,
          inputs: {weights, matchIds, targetNormals: this.props.targetNormals!},
          outputs: {inlierMask},
          body: `var valid=false;if(readWeights(index)>0.0){let normal=safeNormal(readTargetNormals(readMatchIds(index)));valid=dot(normal,normal)>0.5;}writeInlierMask(index,u32(valid));`
        })
      );
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-publish`,
        length: 1,
        inputs: {state: states[this.iterations % 2]},
        outputs: {transform, status},
        body: 'for (var entry = 0u; entry < 16u; entry++) { writeTransform(entry,readState(entry)); } writeStatus(0u,u32(readState(16u)));'
      })
    );
    return nodes;
  }
}

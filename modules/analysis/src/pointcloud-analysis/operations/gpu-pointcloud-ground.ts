// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  validateCommandNodeBudget,
  validateGraphPorts,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_POINT_CLOUD_MAXIMUM_GROUND_RADIUS_CELLS,
  GPU_POINT_CLOUD_MAXIMUM_GROUND_RADIUS_COUNT
} from './pointcloud-limits';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {validatePointCloudScratchBindings} from './pointcloud-kernel-nodes';
import {getPointCloudCellSource, validatePointCloudGrid} from './gpu-pointcloud-surfaces';
import type {PointCloudRasterGrid} from './gpu-pointcloud-surfaces';

import {
  planGraphChunkDispatch,
  validateGraphChunkDispatchBindings,
  validateGraphChunkBindingRanges,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getPointCloudChunkKernelNodes,
  getPointCloudRuntimeKernelNodes,
  validatePointCloudChunkViews,
  validatePointCloudDispatchParameterBindings
} from './pointcloud-chunk-kernel-nodes';

import {validatePointCloudGridParameters} from './pointcloud-grid-parameters';

// Internal raster scratch bit: a cell has no original support; never published directly.
const POINT_CLOUD_GROUND_UNSUPPORTED_CELL = 4;
// Bounded morphology and classification need at most nine uint32 dispatch words.
const POINT_CLOUD_GROUND_MAXIMUM_DISPATCH_WORD_COUNT = 9;

export type GPUPointCloudGroundSettings = {
  /** Nonnegative elevation threshold added at every stage, in z units. */
  initialThreshold: number;
  /** Nonnegative terrain slope per cell of radius increment. */
  slope: number;
  /** Nonnegative upper threshold, at least initialThreshold. */
  maximumThreshold: number;
};

/** One float32 threshold per compile-time radius. */
export function getGPUPointCloudGroundParameterLength(radii: readonly number[]): number {
  return radii.length;
}

/**
 * Pack stage thresholds for a caller-owned GPUParameterBuffer. Evaluate in float64 and round once,
 * preserving the constructor convention even at float32 classification boundaries.
 * @param settings Per-frame threshold settings.
 * @param grid The contributor's compile-time grid.
 * @param radii The contributor's compile-time radii, in cells.
 */
export function getGPUPointCloudGroundParameterValues(
  settings: GPUPointCloudGroundSettings,
  grid: PointCloudRasterGrid,
  radii: readonly number[]
): Float32Array {
  const identifier = 'gpu-pointcloud-ground';
  validateFiniteNumber(identifier, 'initialThreshold', settings.initialThreshold, 0);
  validateFiniteNumber(identifier, 'slope', settings.slope, 0);
  validateFiniteNumber(identifier, 'maximumThreshold', settings.maximumThreshold, 0);
  validateRequirement(
    identifier,
    settings.maximumThreshold >= settings.initialThreshold,
    'maximumThreshold must be at least initialThreshold'
  );
  return Float32Array.from(radii, (radius, index) =>
    Math.min(
      settings.maximumThreshold,
      settings.initialThreshold +
        settings.slope * grid.cellSize * (radius - (radii[index - 1] ?? 0))
    )
  );
}

export type GPUPointCloudGroundOutput = {
  groundMask: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Ground becomes LAS class 2; rejected existing class 2 becomes class 1; other classes survive. */
  outputClassifications: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /**
   * Per-point bit set, zero when clean. Bit 1 (`GPU_POINT_CLOUD_GROUND_STATUS_UNSUPPORTED`): the
   * point is outside the grid, not finite, or in a cell with no coverage. Bit 2
   * (`..._INCOMPLETE_WINDOW`): the final morphology window around the point's cell reached past the
   * covered cells. Bit 1 excludes bit 2. Bits above 2 are zero.
   */
  status: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
};

export type GPUPointCloudGroundProps = {
  /** Runtime origin/cellSize; supplied stage parameters still override threshold defaults. */
  gridParameters?: GraphDataView<'float32'>;
  /** Per-frame stage thresholds; overrides initialThreshold, slope and maximumThreshold. */
  parameters?: GraphDataView<'float32'>;
  id?: string;
  positions: GraphDataView<'float32x3'> | GraphVectorView<'float32x3'>;
  classifications: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  grid: PointCloudRasterGrid;
  /** Input minimum elevation per occupied cell, before classification. */
  minimumSurface: GraphDataView<'float32'>;
  coverage: GraphDataView<'uint32'>;
  /** Strictly increasing square-window radii in cells, each 1–32. */
  radii: readonly number[];
  /** Elevation threshold added at every stage, in the units of z. */
  initialThreshold: number;
  /**
   * Terrain slope (rise over run) per cell of radius increment. Stage `k` rejects a point more
   * than `min(maximumThreshold, initialThreshold + slope * cellSize * (radii[k] - radii[k - 1]))`
   * above the opening, with `radii[-1] = 0`.
   *
   * This is not equation 7 of Zhang et al. (2003), which uses `initialThreshold` alone while the
   * window is at most 3 cells wide and `initialThreshold + slope * (w[k] - w[k - 1]) * cellSize`
   * afterwards, with the full window width `w = 2 * radius + 1`. Two consequences: the first
   * stage here already adds `slope * cellSize * radii[0]`, and at later stages the same `slope` is
   * half as permissive, because a radius increment is half a window-width increment. Pass twice
   * the slope to match Zhang's later stages. With `radii: [1, 2]`, `cellSize: 1`, `slope: 0.1`
   * and `initialThreshold: 0.2` the thresholds here are 0.3 and 0.3; Zhang's are 0.2 and 0.4.
   */
  slope: number;
  /** Upper limit of every stage's threshold. */
  maximumThreshold: number;
  /** Caller-owned result views; their contents are written by the graph. */
  output: GPUPointCloudGroundOutput;
};

/** `GPUPointCloudGround` status bit 1: outside the grid, not finite, or an uncovered cell. */
export const GPU_POINT_CLOUD_GROUND_STATUS_UNSUPPORTED = 1;
/** `GPUPointCloudGround` status bit 2: the final morphology window has incomplete support. */
export const GPU_POINT_CLOUD_GROUND_STATUS_INCOMPLETE_WINDOW = 2;

/**
 * Bounded progressive morphological filter; a square-window PMF, not SMRF equivalence. The stage
 * threshold grows with the radius increment, not with the window-width increment of Zhang et al.
 * (2003): see the `slope` prop. Per-frame: input contents and parameters (stage thresholds).
 * Compile-time: grid, radii and view shapes. Without parameters, threshold props retain their
 * constructor-time behavior. Point rows and outputs accept identical aligned chunks; the raster stays packed.
 */
export class GPUPointCloudGround implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  constructor(readonly props: GPUPointCloudGroundProps) {
    props = snapshotConstructionFields(props, [
      'gridParameters',
      'parameters',
      'id',
      'positions',
      'classifications',
      'grid',
      'minimumSurface',
      'coverage',
      'radii',
      'initialThreshold',
      'slope',
      'maximumThreshold',
      'output'
    ]);
    this.id = props.id ?? 'gpu-pointcloud-ground';
    validateGraphPorts(this.id, props, {
      positions: {required: true, viewType: 'either', formats: ['float32x3']},
      classifications: {required: true, viewType: 'either', formats: ['uint32']},
      gridParameters: {required: false, viewType: 'data', formats: ['float32']},
      parameters: {required: false, viewType: 'data', formats: ['float32']},
      minimumSurface: {required: true, viewType: 'data', formats: ['float32']},
      coverage: {required: true, viewType: 'data', formats: ['uint32']},
      'output.groundMask': {required: true, viewType: 'either', formats: ['uint32']},
      'output.outputClassifications': {required: true, viewType: 'either', formats: ['uint32']},
      'output.status': {required: true, viewType: 'either', formats: ['uint32']}
    });
    validateRequirement(
      this.id,
      props.output !== undefined && props.output !== null,
      'output must be given'
    );
    const grid = snapshotConstructionFields(props.grid, [
      'origin',
      'cellSize',
      'width',
      'height',
      'rowDirection'
    ]);
    this.props = props = Object.freeze({
      ...props,
      grid: Object.freeze({
        ...grid,
        origin: Object.freeze([grid.origin[0], grid.origin[1]] as const)
      }),
      radii: Object.freeze([...props.radii]),
      output: snapshotConstructionFields(props.output, [
        'groundMask',
        'outputClassifications',
        'status'
      ])
    });
    validatePointCloudGrid(props.grid, this.id);
    validatePointCloudGridParameters(this.id, props.gridParameters);
    validateCapacity(
      this.id,
      'radii.length',
      props.radii.length,
      1,
      GPU_POINT_CLOUD_MAXIMUM_GROUND_RADIUS_COUNT
    );
    props.radii.forEach((radius, index) => {
      validateCapacity(
        this.id,
        `radii[${index}]`,
        radius,
        1,
        GPU_POINT_CLOUD_MAXIMUM_GROUND_RADIUS_CELLS
      );
      validateRequirement(
        this.id,
        !index || radius > props.radii[index - 1],
        'radii must be strictly increasing'
      );
    });
    validateFiniteNumber(this.id, 'initialThreshold', props.initialThreshold, 0);
    validateFiniteNumber(this.id, 'slope', props.slope, 0);
    validateFiniteNumber(this.id, 'maximumThreshold', props.maximumThreshold, 0);
    validateRequirement(
      this.id,
      props.maximumThreshold >= props.initialThreshold,
      'maximumThreshold must be at least initialThreshold'
    );
    if (props.parameters) {
      validateViewLength(
        this.id,
        'parameters',
        props.parameters,
        getGPUPointCloudGroundParameterLength(props.radii)
      );
    }
    validateViewLength(
      this.id,
      'minimumSurface',
      props.minimumSurface,
      props.grid.width * props.grid.height
    );
    validateViewLength(this.id, 'coverage', props.coverage, props.minimumSurface.length);
    validateViewLength(this.id, 'classifications', props.classifications, props.positions.length);
    validateViewLength(this.id, 'groundMask', props.output.groundMask, props.positions.length);
    validateViewLength(
      this.id,
      'outputClassifications',
      props.output.outputClassifications,
      props.positions.length
    );
    validateViewLength(this.id, 'status', props.output.status, props.positions.length);
    this.chunkPlan = planGraphChunkDispatch(this.id, {
      positions: props.positions,
      classifications: props.classifications,
      groundMask: props.output.groundMask,
      outputClassifications: props.output.outputClassifications,
      status: props.output.status
    });
    validatePointCloudChunkViews(
      this.id,
      this.chunkPlan,
      ['positions', 'classifications'],
      ['groundMask', 'outputClassifications', 'status'],
      {
        minimumSurface: props.minimumSurface,
        coverage: props.coverage,
        parameters: props.parameters,
        gridParameters: props.gridParameters
      },
      [
        {
          inputNames: ['positions'],
          outputNames: ['groundMask', 'status'],
          inputs: {parameters: props.parameters, gridParameters: props.gridParameters}
        },
        {inputNames: ['classifications', 'groundMask'], outputNames: ['outputClassifications']}
      ]
    );
    this.estimatedCommandNodeCount =
      4 * props.radii.length + (props.radii.length + 2) * (1 + this.chunkPlan.nonemptyChunkCount);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {grid, radii, parameters, gridParameters, minimumSurface, coverage} = this.props;
    if (!this.chunkPlan.nonemptyChunkCount) return [];
    validateGraphChunkDispatchBindings(graph, this.id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, this.id, {
      minimumSurface,
      coverage,
      parameters,
      gridParameters
    });
    validatePointCloudScratchBindings(graph, this.id, {
      'raster morphology scratch': minimumSurface.length * 4
    });
    validatePointCloudDispatchParameterBindings(
      graph,
      this.id,
      POINT_CLOUD_GROUND_MAXIMUM_DISPATCH_WORD_COUNT,
      1
    );
    const rowViews = this.chunkPlan.chunks[0].views;
    const erosion = createTransientView(
      graph,
      `${this.id}-erosion`,
      'float32',
      minimumSurface.length
    );
    const opening = createTransientView(
      graph,
      `${this.id}-opening`,
      'float32',
      minimumSurface.length
    );
    const erosionSupport = createTransientView(
      graph,
      `${this.id}-erosion-support`,
      'uint32',
      minimumSurface.length
    );
    const support = createTransientView(
      graph,
      `${this.id}-support`,
      'uint32',
      minimumSurface.length
    );
    const groundMask = rowViews['groundMask'] as GraphDataView<'uint32'>;
    const status = rowViews['status'] as GraphDataView<'uint32'>;
    const nodes = getPointCloudChunkKernelNodes(graph, {
      id: `${this.id}-initialize`,
      length: this.props.positions.length,
      plan: this.chunkPlan,
      inputs: {},
      outputs: {groundMask},
      body: 'writeGroundMask(index, 1u);'
    });
    for (const [iteration, radius] of radii.entries()) {
      for (const operation of ['erode', 'dilate'] as const)
        nodes.push(
          ...getPointCloudRuntimeKernelNodes(graph, {
            id: `${this.id}-${operation}-${iteration}`,
            length: minimumSurface.length,
            runtimeWords: [radius],
            inputs: {
              surface: operation === 'erode' ? minimumSurface : erosion,
              coverage,
              ...(operation === 'dilate' ? {partial: erosionSupport} : {})
            },
            outputs:
              operation === 'erode'
                ? {result: erosion, support: erosionSupport}
                : {result: opening, support},
            body: `let column = i32(index % ${grid.width}u); let row = i32(index / ${grid.width}u);
let radius = i32(readDispatchParameter(0u));
var value = ${operation === 'erode' ? '' : '-'}3.402823e38; var count = 0u; var complete = true;
for (var vertical = -radius; vertical <= radius; vertical++) {
  for (var horizontal = -radius; horizontal <= radius; horizontal++) {
    let otherColumn = column + horizontal; let otherRow = row + vertical;
    if (otherColumn < 0 || otherColumn >= ${grid.width} || otherRow < 0 || otherRow >= ${grid.height}) { continue; }
    let other = u32(otherRow) * ${grid.width}u + u32(otherColumn);
    if (readCoverage(other) == 0u) { continue; }
    ${operation === 'dilate' ? 'complete = complete && readPartial(other) == 0u;' : ''}
    value = ${operation === 'erode' ? 'min' : 'max'}(value, readSurface(other)); count++;
  }
}
let fullSupport = (2u * u32(radius) + 1u) * (2u * u32(radius) + 1u);
writeResult(index, select(0.0, value, count > 0u));
writeSupport(index, select(${GPU_POINT_CLOUD_GROUND_STATUS_INCOMPLETE_WINDOW}u, 0u, count == fullSupport && complete) | select(0u, ${POINT_CLOUD_GROUND_UNSUPPORTED_CELL}u, readCoverage(index) == 0u));`
          })
        );
      const threshold = Math.min(
        this.props.maximumThreshold,
        this.props.initialThreshold +
          this.props.slope * grid.cellSize * (radius - (radii[iteration - 1] ?? 0))
      );
      const thresholdWord = new Uint32Array(new Float32Array([threshold]).buffer)[0];
      nodes.push(
        ...getPointCloudChunkKernelNodes(graph, {
          id: `${this.id}-classify-${iteration}`,
          length: this.props.positions.length,
          plan: this.chunkPlan,
          runtimeWords: [
            iteration,
            thresholdWord,
            ...[this.props.initialThreshold, this.props.slope, this.props.maximumThreshold].map(
              value => new Uint32Array(new Float32Array([value]).buffer)[0]
            ),
            radius - (radii[iteration - 1] ?? 0)
          ],
          inputs: {
            positions: rowViews['positions'] as GraphDataView<'float32x3'>,
            opening,
            support,
            ...(parameters ? {parameters} : {}),
            ...(gridParameters ? {gridParameters} : {})
          },
          outputs: {groundMask, status},
          body: `let position = readPositions(index);
${getPointCloudCellSource(grid, 'position', Boolean(gridParameters))}
let valid = inside && all(abs(position) <= vec3f(3.402823e38)) && (readSupport(cellIndex) & ${POINT_CLOUD_GROUND_UNSUPPORTED_CELL}u) == 0u;
let ground = valid && readGroundMask(index) != 0u && position.z - readOpening(cellIndex) <= ${parameters ? 'readParameters(readDispatchParameter(0u))' : gridParameters ? 'min(bitcast<f32>(readDispatchParameter(4u)), bitcast<f32>(readDispatchParameter(2u)) + bitcast<f32>(readDispatchParameter(3u)) * readGridParameters(3u) * f32(readDispatchParameter(5u)))' : 'bitcast<f32>(readDispatchParameter(1u))'};
writeGroundMask(index, u32(ground));
writeStatus(index, select(${GPU_POINT_CLOUD_GROUND_STATUS_UNSUPPORTED}u, readSupport(cellIndex) & ${GPU_POINT_CLOUD_GROUND_STATUS_INCOMPLETE_WINDOW}u, valid));`
        })
      );
    }
    nodes.push(
      ...getPointCloudChunkKernelNodes(graph, {
        id: `${this.id}-classes`,
        length: this.props.positions.length,
        plan: this.chunkPlan,
        inputs: {
          classifications: rowViews['classifications'] as GraphDataView<'uint32'>,
          groundMask
        },
        outputs: {
          outputClassifications: rowViews['outputClassifications'] as GraphDataView<'uint32'>
        },
        body: `let original = readClassifications(index);
writeOutputClassifications(index, select(select(original, 1u, original == 2u), 2u, readGroundMask(index) != 0u));`
      })
    );
    return nodes;
  }
}

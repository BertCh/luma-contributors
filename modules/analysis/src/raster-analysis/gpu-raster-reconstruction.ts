// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGPUStatusPort} from '../analysis-core/contracts/status';
import {
  getGraphOutputViews,
  validateCommandNodeBudget,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../analysis-core/kernels/wgsl-common';
import {
  createRasterJacobiLoopNodes,
  getRasterJacobiLoopNodeCount,
  requireRasterLength,
  validateRasterExpansionGraph,
  validateRasterExpansionViews,
  type RasterIterationStatusOutputs
} from './raster-expansion-utils';

/** Default number of reconstruction rounds, each spreading the marker by one cell. */
const DEFAULT_MAXIMUM_ITERATIONS = 64;

export type GPURasterReconstructionProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Finite grayscale marker <= mask for dilation, >= mask for erosion; otherwise invalid. */
  marker: GraphDataView<'float32'>;
  mask: GraphDataView<'float32'>;
  method?: 'dilation' | 'erosion';
  /**
   * Neighbours a value spreads to in one round: `8` is the 3 x 3 square (edge and corner
   * neighbours), `4` the cross (edge neighbours only). Defaults to `8`, the default footprint of
   * `skimage.morphology.reconstruction` and of the published `GPURasterDilation` (`'square'`).
   * With `4`, a marker and a mask region that touch only at a corner stay unconnected. (The
   * default was `4` before 2026-11-10.)
   */
  connectivity?: 4 | 8;
  /** Rounds before the run stops unconverged. Default 64; at most 26,209 (the command node budget). */
  maximumIterations?: number;
  output: {
    values: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    /**
     * `converged` is 1 when a round changed nothing before the iteration limit. The optional
     * `iterationLimitReached` and `iterationCount` words are published when present.
     */
    status: RasterIterationStatusOutputs;
  };
};

/**
 * Geodesic reconstruction under an explicit mask, using synchronous bounded GPU rounds. One
 * round is a grey dilation (or erosion) over `connectivity` neighbours, 8 by default, clipped by
 * the mask; the fixed point is the reconstruction of `skimage.morphology.reconstruction`. Rounds
 * after convergence dispatch no work, and the current values are always published; check
 * `status.converged` to know whether they are final.
 */
export class GPURasterReconstruction implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterReconstructionProps;
  /** Command nodes `getCommandNodes` returns; checked against the node budget. */
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPURasterReconstructionProps) {
    this.id = props.id ?? 'raster-reconstruction';
    this.props = props;
    const {id} = this;
    const count = validateRasterExpansionViews(
      id,
      props.grid,
      [props.marker, props.mask],
      getGraphOutputViews(props.output)
    );
    requireRasterLength(props.marker, count, id, 'marker');
    requireRasterLength(props.mask, count, id, 'mask');
    requireRasterLength(props.output.values, count, id, 'output.values');
    requireRasterLength(props.output.validity, count, id, 'output.validity');
    requireRasterLength(props.output.status.converged, 1, id, 'output.status.converged');
    requireRasterLength(
      props.output.status.iterationLimitReached,
      1,
      id,
      'output.status.iterationLimitReached'
    );
    requireRasterLength(props.output.status.iterationCount, 1, id, 'output.status.iterationCount');
    validateGPUStatusPort(id, props.output.status);
    const maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    validateRequirement(
      id,
      Number.isInteger(maximumIterations) && maximumIterations > 0,
      'maximumIterations must be a positive integer'
    );
    // Initialization and publish, plus the gated rounds.
    this.estimatedCommandNodeCount = 2 + getRasterJacobiLoopNodeCount(maximumIterations + 1);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    const dilation = props.method !== 'erosion';
    validateRasterExpansionGraph(id, graph, [
      props.marker,
      props.mask,
      ...getGraphOutputViews(output)
    ]);
    const first = createTransientView(graph, `${id}-first`, 'float32', count);
    const second = createTransientView(graph, `${id}-second`, 'float32', count);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-init`,
        operation: 'GPURasterReconstruction',
        invocationCount: count,
        bindings: [
          {name: 'marker', view: props.marker, type: 'f32', access: 'read'},
          {name: 'mask', view: props.mask, type: 'f32', access: 'read'},
          {name: 'values', view: first, type: 'f32', access: 'read_write'}
        ],
        declarations: IS_FINITE_VALUE_WGSL,
        body: `let markerValue = marker[markerOffset + index];
  let maskValue = mask[maskOffset + index];
  let valid = isFiniteValue(markerValue) && isFiniteValue(maskValue)
    && markerValue ${dilation ? '<=' : '>='} maskValue;
  values[valuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), markerValue, valid);`
      })
    ];
    let previous = first;
    let next = second;
    // 8-connected (the 3 x 3 square) unless the cross is asked for.
    const connectivityWGSL =
      props.connectivity === 4 ? 'if (abs(rowOffset) + abs(columnOffset) > 1) { continue; }' : '';
    // One extra round tests whether the final iterate is a true fixed point.
    nodes.push(
      ...createRasterJacobiLoopNodes(graph, {
        id: `${id}-iterate`,
        operation: 'GPURasterReconstruction',
        invocationCount: count,
        roundCount: (props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS) + 1,
        status: output.status,
        createRound: (
          {nodeId, slotBinding, markChangedWGSL, condition, extraResources},
          status
        ) => {
          const round = createWGSLKernelNode(graph, {
            id: nodeId,
            operation: 'GPURasterReconstruction',
            variant: 'reconstruction-round',
            invocationCount: count,
            bindings: [
              {name: 'previous', view: previous, type: 'f32', access: 'read'},
              {name: 'mask', view: props.mask, type: 'f32', access: 'read'},
              {name: 'next', view: next, type: 'f32', access: 'read_write'},
              {name: 'status', view: status, type: 'atomic<u32>', access: 'read_write'},
              slotBinding
            ],
            declarations: `${IS_FINITE_VALUE_WGSL}
const WIDTH: u32 = ${grid.width}u;
const HEIGHT: u32 = ${grid.height}u;`,
            body: `let original = previous[previousOffset + index];
  var value = original;
  if (isFiniteValue(original)) {
    let column = i32(index % WIDTH);
    let row = i32(index / WIDTH);
    for (var rowOffset = -1; rowOffset <= 1; rowOffset++) {
      for (var columnOffset = -1; columnOffset <= 1; columnOffset++) {
        ${connectivityWGSL}
        let neighborColumn = column + columnOffset;
        let neighborRow = row + rowOffset;
        if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(WIDTH) || neighborRow >= i32(HEIGHT)) { continue; }
        let neighbor = previous[previousOffset + u32(neighborRow) * WIDTH + u32(neighborColumn)];
        if (isFiniteValue(neighbor)) { value = ${dilation ? 'max' : 'min'}(value, neighbor); }
      }
    }
    value = ${dilation ? 'min' : 'max'}(value, mask[maskOffset + index]);
  }
  next[nextOffset + index] = value;
  if (isFiniteValue(original) && value != original) { ${markChangedWGSL} }`,
            condition,
            extraResources
          });
          [previous, next] = [next, previous];
          return round;
        }
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPURasterReconstruction',
        invocationCount: count,
        bindings: [
          {name: 'previous', view: previous, type: 'f32', access: 'read'},
          {name: 'values', view: output.values, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_VALUE_WGSL,
        body: `// The current iterate is always published; status.converged says whether it is final.
  let value = previous[previousOffset + index];
  let valid = isFiniteValue(value);
  values[valuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), value, valid);
  validity[validityOffset + index] = u32(valid);`
      })
    );
    return nodes;
  }
}

export type GPURasterFillHolesProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GraphDataView<'uint32'>;
  /**
   * Connectivity of the background that is flooded from the boundary. Defaults to `4`, the rule
   * of `scipy.ndimage.binary_fill_holes`: background that reaches the boundary only through a
   * diagonal gap is a hole. With `8` a diagonal gap lets the flood through.
   */
  connectivity?: 4 | 8;
  /** Reconstruction rounds before the run stops unconverged. Default 64; at most 26,207 (the command node budget). */
  maximumIterations?: number;
  output: {values: GraphDataView<'uint32'>; status: RasterIterationStatusOutputs};
};
/**
 * Fills background components not connected to the boundary, using the selected background
 * connectivity. The current fill is always published. When `status.converged` is 0, background
 * the boundary flood has not reached yet is still reported as a hole.
 */
export class GPURasterFillHoles implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterFillHolesProps;
  /** Command nodes `getCommandNodes` returns; checked against the node budget. */
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPURasterFillHolesProps) {
    this.id = props.id ?? 'raster-fill-holes';
    this.props = props;
    const {id} = this;
    const count = validateRasterExpansionViews(
      id,
      props.grid,
      [props.input],
      getGraphOutputViews(props.output)
    );
    requireRasterLength(props.input, count, id, 'input');
    requireRasterLength(props.output.values, count, id, 'output.values');
    requireRasterLength(props.output.status.converged, 1, id, 'output.status.converged');
    requireRasterLength(
      props.output.status.iterationLimitReached,
      1,
      id,
      'output.status.iterationLimitReached'
    );
    requireRasterLength(props.output.status.iterationCount, 1, id, 'output.status.iterationCount');
    validateGPUStatusPort(id, props.output.status);
    const maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    validateRequirement(
      id,
      Number.isInteger(maximumIterations) && maximumIterations > 0,
      'maximumIterations must be a positive integer'
    );
    // Prepare and publish, plus the reconstruction.
    this.estimatedCommandNodeCount = 2 + getRasterJacobiLoopNodeCount(maximumIterations + 1) + 2;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const count = props.grid.width * props.grid.height;
    const marker = createTransientView(graph, `${id}-marker`, 'float32', count);
    const mask = createTransientView(graph, `${id}-mask`, 'float32', count);
    const reconstructed = createTransientView(graph, `${id}-reconstructed`, 'float32', count);
    const validity = createTransientView(graph, `${id}-validity`, 'uint32', count);
    validateRasterExpansionGraph(id, graph, [props.input, ...getGraphOutputViews(props.output)]);
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-prepare`,
        operation: 'GPURasterFillHoles',
        invocationCount: count,
        bindings: [
          {name: 'input', view: props.input, type: 'u32', access: 'read'},
          {name: 'marker', view: marker, type: 'f32', access: 'read_write'},
          {name: 'mask', view: mask, type: 'f32', access: 'read_write'}
        ],
        body: `let background=input[inputOffset+index]==0u;let column=index%${props.grid.width}u;let row=index/${props.grid.width}u;let boundary=column==0u||row==0u||column==${props.grid.width - 1}u||row==${props.grid.height - 1}u;mask[maskOffset+index]=f32(background);marker[markerOffset+index]=f32(background&&boundary);`
      }),
      ...new GPURasterReconstruction({
        id: `${id}-reconstruct`,
        grid: props.grid,
        marker,
        mask,
        // Hole filling keeps the 4-connected background of scipy.ndimage.binary_fill_holes.
        connectivity: props.connectivity ?? 4,
        maximumIterations: props.maximumIterations,
        output: {
          values: reconstructed,
          validity,
          status: props.output.status
        }
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPURasterFillHoles',
        invocationCount: count,
        bindings: [
          {name: 'reconstructed', view: reconstructed, type: 'f32', access: 'read'},
          {name: 'validity', view: validity, type: 'u32', access: 'read'},
          {name: 'values', view: props.output.values, type: 'u32', access: 'read_write'}
        ],
        body: 'values[valuesOffset+index]=select(0xfffffffeu,u32(reconstructed[reconstructedOffset+index]==0.0),validity[validityOffset+index]!=0u);'
      })
    ];
  }
}

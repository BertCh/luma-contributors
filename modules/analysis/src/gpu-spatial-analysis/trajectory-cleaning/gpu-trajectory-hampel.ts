// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createRowBoundsNodes} from '../trajectory-staypoints/movement-common';

const OPERATION = 'GPUTrajectoryHampel';

/** Largest accepted Hampel window size in rows. */
export const GPU_TRAJECTORY_HAMPEL_MAXIMUM_WINDOW_SIZE = 129;

/** Number of float32 elements in a Hampel parameter buffer. */
export const GPU_TRAJECTORY_HAMPEL_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUTrajectoryHampel}. */
export type GPUTrajectoryHampelSettings = {
  /** Number of scaled median absolute deviations above which a value is an outlier. */
  numberOfSigmas: number;
};

/**
 * Packs Hampel settings into the float32 layout read by {@link GPUTrajectoryHampel}.
 *
 * Layout: `[numberOfSigmas, 0, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If `numberOfSigmas` is negative or not finite.
 */
export function getGPUTrajectoryHampelParameterValues(
  settings: GPUTrajectoryHampelSettings,
  target: Float32Array = new Float32Array(GPU_TRAJECTORY_HAMPEL_PARAMETER_LENGTH)
): Float32Array {
  if (!Number.isFinite(settings.numberOfSigmas) || settings.numberOfSigmas < 0) {
    throw new Error('Hampel numberOfSigmas must be finite and non-negative');
  }
  target.set([settings.numberOfSigmas, 0, 0, 0]);
  return target;
}

/** Properties for {@link GPUTrajectoryHampel}. */
export type GPUTrajectoryHampelProps = {
  /** Prefix for node and transient IDs. Defaults to `'trajectory-hampel'`. */
  id?: string;
  /** One value per row, for example a speed or a coordinate. */
  values: GraphDataView<'float32'>;
  /** Track-to-row offsets with one terminal entry. Windows never cross a track. */
  trackOffsets: GraphDataView<'uint32'>;
  /** Full window size in rows (the `hampel` package's `window_size`). Compile-time, at most 129. */
  windowSize: number;
  /** Per-frame float32 view of {@link GPU_TRAJECTORY_HAMPEL_PARAMETER_LENGTH} elements. */
  parameters: GraphDataView<'float32'>;
  output: {
    /** Filtered values: the window median at an outlier, otherwise the input. */
    filtered: GraphDataView<'float32'>;
    /** Optional per-row `1`/`0` flag: the row was replaced. */
    flags?: GraphDataView<'uint32'>;
    /** Optional one-row count of replaced values. */
    outlierCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Hampel filter per track (the `hampel` package used by PTRAIL). A row is an outlier when
 * `|value - median| > numberOfSigmas * 1.4826 * MAD` over a window of `2 * floor(windowSize / 2) +
 * 1` rows centred on it, inside the track. Rows closer than half a window to a track end pass
 * through unchanged, as in the package. One thread per row sorts its window in registers.
 */
export class GPUTrajectoryHampel implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectoryHampelProps;
  private readonly halfWindow: number;

  constructor(props: GPUTrajectoryHampelProps) {
    this.id = props.id ?? 'trajectory-hampel';
    this.props = props;
    const id = this.id;
    const {values, trackOffsets, parameters, output, windowSize} = props;
    validatePackedView(values, ['float32'], `${id} values`);
    validatePackedView(trackOffsets, ['uint32'], `${id} trackOffsets`);
    validatePackedView(parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      trackOffsets.length >= 2,
      'trackOffsets must contain at least two rows'
    );
    validateRequirement(
      id,
      parameters.length >= GPU_TRAJECTORY_HAMPEL_PARAMETER_LENGTH,
      'parameters must hold 4 float32 values'
    );
    validateRequirement(
      id,
      Number.isInteger(windowSize) && windowSize >= 1,
      'windowSize must be a positive integer'
    );
    validateMaximumCount(id, 'windowSize', windowSize, GPU_TRAJECTORY_HAMPEL_MAXIMUM_WINDOW_SIZE);
    this.halfWindow = Math.floor(windowSize / 2);
    validatePackedView(output.filtered, ['float32'], `${id} output.filtered`);
    validateViewLength(id, 'output.filtered', output.filtered, values.length);
    if (output.flags) {
      validatePackedView(output.flags, ['uint32'], `${id} output.flags`);
      validateViewLength(id, 'output.flags', output.flags, values.length);
    }
    if (output.outlierCount) {
      validatePackedView(output.outlierCount, ['uint32'], `${id} output.outlierCount`);
      validateRequirement(
        id,
        output.outlierCount.length >= 1,
        'output.outlierCount must contain one uint32 row'
      );
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.filtered, output.flags, output.outlierCount],
      [values, trackOffsets, parameters]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, halfWindow} = this;
    const {values, trackOffsets, parameters, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      trackOffsets,
      parameters,
      output.filtered,
      output.flags,
      output.outlierCount
    ]);
    const rowCount = values.length;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'values',
      rowCount * 4,
      'split the tracks over several graphs'
    );
    const windowLength = 2 * halfWindow + 1;
    const rowStarts = createTransientView(graph, `${id}-row-starts`, 'uint32', rowCount);
    const rowEnds = createTransientView(graph, `${id}-row-ends`, 'uint32', rowCount);
    const rowTracks = createTransientView(graph, `${id}-row-tracks`, 'uint32', rowCount);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...createRowBoundsNodes<Parameters>(graph, {
        id: `${id}-row-bounds`,
        operation: OPERATION,
        trackOffsets,
        rowCount,
        rowStarts,
        rowEnds,
        rowTracks
      })
    ];
    if (output.outlierCount) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-count-reset`,
          operation: OPERATION,
          view: output.outlierCount,
          type: 'u32',
          value: '0u',
          componentCount: 1
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-filter`,
        operation: OPERATION,
        variant: `window-${windowLength}`,
        invocationCount: rowCount,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'rowStarts', view: rowStarts, type: 'u32', access: 'read'},
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'filtered', view: output.filtered, type: 'f32', access: 'read_write'},
          ...(output.flags
            ? [{name: 'flags', view: output.flags, type: 'u32', access: 'read_write'} as const]
            : []),
          ...(output.outlierCount
            ? [
                {
                  name: 'outlierCount',
                  view: output.outlierCount,
                  type: 'atomic<u32>',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        declarations: `const WINDOW_LENGTH: u32 = ${windowLength}u;
const HALF_WINDOW: u32 = ${halfWindow}u;
var<private> window: array<f32, WINDOW_LENGTH>;
fn sortWindow() {
  for (var i = 1u; i < WINDOW_LENGTH; i++) {
    let key = window[i];
    var j = i;
    while (j > 0u && window[j - 1u] > key) { window[j] = window[j - 1u]; j--; }
    window[j] = key;
  }
}
`,
        body: `let value = values[valuesOffset + index];
  var result = value;
  var isOutlier = false;
  let begin = rowStarts[rowStartsOffset + index];
  let end = rowEnds[rowEndsOffset + index];
  if (begin != 0xffffffffu && index >= begin + HALF_WINDOW && index + HALF_WINDOW < end) {
    for (var offset = 0u; offset < WINDOW_LENGTH; offset++) {
      window[offset] = values[valuesOffset + index + offset - HALF_WINDOW];
    }
    sortWindow();
    let median = window[HALF_WINDOW];
    for (var offset = 0u; offset < WINDOW_LENGTH; offset++) {
      window[offset] = abs(values[valuesOffset + index + offset - HALF_WINDOW] - median);
    }
    sortWindow();
    let deviation = window[HALF_WINDOW];
    let threshold = parameters[parametersOffset] * 1.4826 * deviation;
    if (abs(value - median) > threshold) { result = median; isOutlier = true; }
  }
  filtered[filteredOffset + index] = result;
  ${output.flags ? 'flags[flagsOffset + index] = select(0u, 1u, isOutlier);' : ''}
  ${output.outlierCount ? 'if (isOutlier) { atomicAdd(&outlierCount[outlierCountOffset], 1u); }' : ''}`
      })
    );
    return nodes;
  }
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {GPUCardinalityPort} from '../../analysis-core/contracts/index';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateCellViewBindingRanges, validateCellWritableViews} from './cell-validation';

/** Caller-owned outputs of {@link GPUCellLevelSelection}. */
export type GPUCellLevelSelectionOutput = {
  /** One-row status scalars; `count` receives the active level's occupied row count. */
  status: GPUCardinalityPort;
  /** Optional one-row scalar receiving the active level's first row (`levelFirstRows[level]`). */
  firstRow?: GraphDataView<'uint32'>;
  /**
   * Optional indirect draw arguments `[vertexCount, instanceCount, firstVertex, firstInstance]`.
   * Only `instanceCount` and `firstInstance` are written.
   */
  drawArguments?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUCellLevelSelection}. */
export type GPUCellLevelSelectionProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-level-selection'`. */
  id?: string;
  /** Per-level clamped counts, for example `GPUCellPyramidProps.levelCounts`. */
  levelCounts: GraphDataView<'uint32'>;
  /**
   * Per-frame active level index (one `uint32`), typically chosen from the zoom. Values past the
   * last level select the last (coarsest) level.
   */
  activeLevel: GraphDataView<'uint32'>;
  /** Row offset of each level's cells in a shared buffer. Defaults to zeros. */
  levelFirstRows?: readonly number[];
  output: GPUCellLevelSelectionOutput;
};

/**
 * Publishes the active pyramid level's count, first row and indirect draw arguments from a
 * per-frame level index, so a layer switches levels by rewriting one `uint32` without rebuilding
 * or re-running the pyramid. Put it in a small per-frame graph next to the draw.
 */
export class GPUCellLevelSelection implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellLevelSelectionProps;

  constructor(props: GPUCellLevelSelectionProps) {
    const fields = snapshotConstructionFields(props, [
      'id',
      'levelCounts',
      'activeLevel',
      'levelFirstRows',
      'output'
    ]);
    const output = snapshotConstructionFields(fields.output, [
      'status',
      'firstRow',
      'drawArguments'
    ]);
    const status = snapshotConstructionFields(output.status, ['count', 'requiredCount']);
    props = Object.freeze({
      ...fields,
      levelFirstRows: fields.levelFirstRows ? Object.freeze([...fields.levelFirstRows]) : undefined,
      output: Object.freeze({...output, status})
    });
    this.id = props.id ?? 'cell-level-selection';
    this.props = props;
    const id = this.id;
    validatePackedUint32View(props.levelCounts, `${id} levelCounts`);
    validatePackedUint32View(props.activeLevel, `${id} activeLevel`);
    validatePackedUint32View(props.output.status.count, `${id} output.status.count`);
    validateCapacity(id, 'output.status.count rows', props.output.status.count.length);
    if (props.output.status.requiredCount) {
      validatePackedUint32View(
        props.output.status.requiredCount,
        `${id} output.status.requiredCount`
      );
      validateCapacity(
        id,
        'output.status.requiredCount rows',
        props.output.status.requiredCount.length
      );
    }
    if (props.levelCounts.length < 1 || props.activeLevel.length < 1) {
      throw new Error(`${id} levelCounts and activeLevel must hold at least one row`);
    }
    const levelFirstRows = props.levelFirstRows ?? [];
    if (
      props.levelFirstRows &&
      (levelFirstRows.length !== props.levelCounts.length ||
        levelFirstRows.some(row => !Number.isInteger(row) || row < 0 || row > 0xffffffff))
    ) {
      throw new Error(`${id} levelFirstRows must hold one uint32 per level`);
    }
    if (props.output.firstRow) {
      validatePackedUint32View(props.output.firstRow, `${id} output.firstRow`);
      validateCapacity(id, 'output.firstRow rows', props.output.firstRow.length);
    }
    if (props.output.drawArguments) {
      validatePackedUint32View(props.output.drawArguments, `${id} output.drawArguments`);
      if (props.output.drawArguments.length < 4) {
        throw new Error(`${id} output.drawArguments must hold 4 uint32 values`);
      }
    }
    validateCellWritableViews(id, [
      ['output.status.count', props.output.status.count],
      ['output.status.requiredCount', props.output.status.requiredCount],
      ['output.firstRow', props.output.firstRow],
      ['output.drawArguments', props.output.drawArguments]
    ]);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.status.count,
        props.output.status.requiredCount,
        props.output.firstRow,
        props.output.drawArguments
      ],
      [props.levelCounts, props.activeLevel]
    );
  }

  /** Returns an optional first-row parameter upload and one selection node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateCellViewBindingRanges(graph, id, [
      props.levelCounts,
      props.activeLevel,
      output.status.count,
      output.status.requiredCount,
      output.firstRow,
      output.drawArguments
    ]);
    if (props.levelFirstRows) {
      validateCapacity(
        id,
        'levelFirstRows parameter byteLength',
        props.levelFirstRows.length * Uint32Array.BYTES_PER_ELEMENT,
        0,
        graph.device.limits.maxBufferSize
      );
    }
    // The first-row parameter binding is packed with the same word count as levelCounts,
    // whose actual aligned range was checked above. No resource exists before that preflight.
    const firstRows = props.levelFirstRows
      ? createSpatialStatisticsParameterNodes(
          graph,
          `${id}-first-rows`,
          Uint32Array.from(props.levelFirstRows)
        )
      : undefined;
    return [
      ...(firstRows?.nodes ?? []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select`,
        operation: 'GPUCellLevelSelection',
        bindings: [
          ...(firstRows
            ? [
                {
                  name: 'firstRows',
                  view: firstRows.parameters,
                  type: 'u32',
                  access: 'read'
                } as const
              ]
            : []),
          {
            name: 'levelCounts',
            view: props.levelCounts,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'activeLevel',
            view: props.activeLevel,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'countOut',
            view: output.status.count,
            type: 'u32',
            access: 'read_write'
          },
          ...(output.status.requiredCount
            ? [
                {
                  name: 'requiredCountOut',
                  view: output.status.requiredCount,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(output.firstRow
            ? [
                {
                  name: 'firstRowOut',
                  view: output.firstRow,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(output.drawArguments
            ? [
                {
                  name: 'drawArguments',
                  view: output.drawArguments,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: 1,
        body: `let levelCount = arrayLength(&levelCounts) - levelCountsOffset;
  let level = min(activeLevel[activeLevelOffset], levelCount - 1u);
  let count = levelCounts[levelCountsOffset + level];
  let firstRow = ${firstRows ? 'firstRows[firstRowsOffset + level]' : '0u'};
  countOut[countOutOffset] = count;
  ${output.status.requiredCount ? 'requiredCountOut[requiredCountOutOffset] = count;' : ''}
  ${output.firstRow ? 'firstRowOut[firstRowOutOffset] = firstRow;' : ''}
  ${
    output.drawArguments
      ? `drawArguments[drawArgumentsOffset + 1u] = count;
  drawArguments[drawArgumentsOffset + 3u] = firstRow;`
      : ''
  }`
      })
    ];
  }
}

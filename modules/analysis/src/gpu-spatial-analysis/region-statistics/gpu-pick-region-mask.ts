// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSpatialStatisticsWritableDispatches} from '../spatial-weights/spatial-statistics-writable-dispatches';
import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUStatusPort,
  type GPULonLatPolygonStatusPort,
  type GPUOverflowStatusPort
} from '../../analysis-core/contracts/index';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Caller-owned outputs of {@link GPUPickRegionMask}, both written on every submission. */
export type GPUPickRegionMaskOutput = {
  /** Packed mask with one row per source row, set to 1 for picked rows and 0 otherwise. */
  mask: GraphDataView<'uint32'>;
  /** One-row overflow flag, normalized to 0 or 1 from the pick result. */
  status: GPUOverflowStatusPort;
};

/** Validates pick storage before a composing contributor allocates graph resources. @internal */
export function validateRegionPicks(
  id: string,
  picks: GraphDataView<'uint32'>,
  batchIndex?: number
): void {
  validatePackedUint32View(picks, `${id} picks`);
  validateCapacity(id, 'picks.length', picks.length, 4, Number.MAX_SAFE_INTEGER);
  if (batchIndex !== undefined) validateCapacity(id, 'batchIndex', batchIndex, 0, 0x7fffffff);
}

/** Properties for {@link GPUPickRegionMask}. */
export type GPUPickRegionMaskProps = {
  /** ID prefix. Defaults to `'pick-region-mask'`. */
  id?: string;
  /**
   * Packed `GPUIndexPickingTarget.addRegionPass` result `[count, overflow, (objectIndex,
   * batchIndex)...]`. Its length must be at least 4.
   */
  picks: GraphDataView<'uint32'>;
  /** Compile-time batch filter in `[0, 2^31 - 1]`. When omitted, every batch is accepted. */
  batchIndex?: number;
  /** Source-aligned mask and pick overflow flag. */
  output: GPUPickRegionMaskOutput;
};

/**
 * Converts a capacity-bounded index-picking region result into a deduplicated source-aligned 0/1
 * mask. The application adds `GPUIndexPickingTarget.addRegionPass` before this contributor; the graph
 * orders the two through the hazard on `picks`.
 */
export class GPUPickRegionMask implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPickRegionMaskProps;

  constructor(props: GPUPickRegionMaskProps) {
    const construction = snapshotConstructionFields(props, ['id', 'picks', 'batchIndex', 'output']);
    const id = construction.id ?? 'pick-region-mask';
    const capturedOutput = snapshotConstructionFields(construction.output, ['mask', 'status']);
    const status: GPUOverflowStatusPort & GPULonLatPolygonStatusPort = capturedOutput.status;
    for (const name of ['antimeridianUncertainCount', 'poleUncertainCount'] as const) {
      validateRequirement(
        id,
        status[name] === undefined,
        `output.status.${name} must be omitted because the pick route has no geographic polygon diagnostics`
      );
    }
    props = Object.freeze({
      ...construction,
      output: Object.freeze({
        ...capturedOutput,
        status: snapshotConstructionFields(status, ['overflow'])
      })
    });
    this.id = id;
    this.props = props;
    const {picks, output} = props;
    validateRegionPicks(id, picks, props.batchIndex);
    validatePackedUint32View(output.mask, `${id} output.mask`);
    validateGPUStatusPort(`${id} output.status`, output.status);
    if (output.mask.length < 1 || !output.status.overflow) {
      throw new Error(`${id} output.mask and output.status.overflow are required`);
    }
    validateSpatialStatisticsWritableDispatches(
      id,
      {mask: output.mask, overflow: output.status.overflow},
      []
    );
    validateGraphOutputsDisjointFromInputs(id, [output.mask, output.status.overflow], [picks]);
  }

  /** Returns clear, scatter, and overflow nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateSpatialStatisticsBindings(graph, this.id, {
      picks: this.props.picks,
      mask: this.props.output.mask,
      overflow: this.props.output.status.overflow
    });
    const {id, props} = this;
    const {picks: result, batchIndex} = props;
    const outputMask = props.output.mask;
    const {overflow} = props.output.status;
    validateGraphViewsBelongToGraph(id, graph, [result, outputMask, overflow]);
    const pairCapacity = Math.floor((result.length - 2) / 2);
    return [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: 'GPUPickRegionMask',
        view: outputMask,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scatter`,
        operation: 'GPUPickRegionMask',
        variant: 'scatter',
        bindings: [
          {name: 'result', view: result, type: 'u32', access: 'read'},
          {name: 'outputMask', view: outputMask, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: `const ROW_COUNT: u32 = ${outputMask.length}u;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
        body: `if (index >= min(result[resultOffset], PAIR_CAPACITY)) { return; }
  let objectIndex = bitcast<i32>(result[resultOffset + 2u + index * 2u]);
  ${
    batchIndex === undefined
      ? ''
      : `let batchIndex = bitcast<i32>(result[resultOffset + 3u + index * 2u]);
  if (batchIndex != ${batchIndex}i) { return; }`
  }
  if (objectIndex < 0 || u32(objectIndex) >= ROW_COUNT) { return; }
  atomicStore(&outputMask[outputMaskOffset + u32(objectIndex)], 1u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-overflow`,
        operation: 'GPUPickRegionMask',
        variant: 'overflow',
        bindings: [
          {name: 'result', view: result, type: 'u32', access: 'read'},
          {name: 'overflow', view: overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: 'overflow[overflowOffset] = select(0u, 1u, result[resultOffset + 1u] != 0u);'
      })
    ];
  }
}

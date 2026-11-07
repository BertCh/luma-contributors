// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {GPUCommandGraph, GraphVectorView, type GraphDataView} from './gpu-command-graph';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBinding
} from './gpu-kernel-node';
import {GPUScan, type GPUScanInput, type GPUScanProps} from './gpu-scan';
import {
  getViewElementOffset,
  getGraphDataPrefix,
  validatePackedUint32View
} from './graph-data-view-utils';

/** Properties for materializing dense indices and a count from binary flags. */
export type GPUFlagOffsetsProps = {
  /** Prefix for generated graph node IDs. */
  id?: string;
  /** Packed zero-or-one flags. */
  flags: GPUScanInput;
  /** Exclusive dense index for every flag. */
  offsets: GPUScanInput;
  /** Single uint32 receiving the number of set flags. */
  count: GraphDataView<'uint32'>;
  /** Scan algorithm. Defaults to the `GPUScan` default (`multi-pass`); `decoupled-lookback` scans in one pass. */
  scanAlgorithm?: GPUScanProps['algorithm'];
};

/**
 * Converts binary flags into exclusive dense offsets and a GPU-resident count.
 *
 * Use this small primitive when a classifier has already identified selected slots and later graph
 * nodes need stable destinations or a valid-prefix length. It is also the reusable building block
 * for nullable values and each logical element depth in nested column layouts.
 */
export class GPUFlagOffsets {
  readonly id: string;
  readonly props: Readonly<GPUFlagOffsetsProps>;

  constructor(props: GPUFlagOffsetsProps) {
    this.id = props.id ?? 'gpu-flag-offsets';
    this.props = Object.freeze({...props, id: this.id});
    validateConfiguration(this.props);
  }

  /** Adds one exclusive scan, which also publishes the count, to a command graph. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    const {flags, count} = this.props;
    for (const view of [...getChunks(flags), ...getChunks(this.props.offsets), count]) {
      if (view.buffer.graph !== graph) {
        throw new Error(`${this.id} views must belong to the target graph`);
      }
    }
    const offsets = getGraphDataPrefix(graph, this.props.offsets, flags.length);
    if (flags.length > 0) {
      // The scan's final level writes its own total, which is exactly the number of set flags.
      const scanNodes = new GPUScan({
        id: `${this.id}-scan`,
        input: flags,
        output: offsets,
        mode: 'exclusive',
        total: count,
        algorithm: this.props.scanAlgorithm
      }).getCommandNodes(graph);
      for (const node of scanNodes) {
        // Recording graphs attribute the scan to this operator, which owns the published count.
        if (node.annotation) {
          node.annotation.op = 'GPUFlagOffsets';
          node.annotation.formula = 'gpu-core.flag-offsets';
        }
      }
      nodes.push(...scanNodes);
    } else {
      nodes.push(addZeroCountPass(graph, this.id, count));
    }

    return nodes;
  }
}

/** Publishes the zero count of an empty flag range. */
function addZeroCountPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  count: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  const bindings: GPUKernelBinding[] = [{name: 'count', view: count, access: 'write'}];
  const constants = {COUNT_OFFSET: getViewElementOffset(count)};
  return createGPUKernelNode(graph, {
    id: `${id}-count`,
    operation: 'GPUFlagOffsetsCount',
    moduleId: 'GPUFlagOffsets/count',
    formula: 'gpu-core.flag-offsets',
    source: /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(1) fn main() { count[COUNT_OFFSET] = 0u; }`,
    constants,
    bindings,
    workgroupSize: 1,
    dispatchLayout: {x: 1, y: 1, z: 1},
    invocationCount: 1
  });
}

function validateConfiguration(props: Readonly<GPUFlagOffsetsProps>): void {
  for (const view of getChunks(props.flags)) {
    validatePackedUint32View(view, `${props.id} flags`);
  }
  for (const view of getChunks(props.offsets)) {
    validatePackedUint32View(view, `${props.id} offsets`);
  }
  validatePackedUint32View(props.count, `${props.id} count`);
  if (props.offsets.length < props.flags.length) {
    throw new Error(`${props.id} offsets must contain at least flags.length rows`);
  }
  if (props.count.length < 1) {
    throw new Error(`${props.id} result views are too short`);
  }
}

function getChunks(input: GPUScanInput): readonly GraphDataView<'uint32'>[] {
  return input instanceof GraphVectorView ? input.data : [input];
}

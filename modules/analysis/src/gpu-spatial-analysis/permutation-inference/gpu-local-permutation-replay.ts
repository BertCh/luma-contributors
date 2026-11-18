// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateGraphChunkBindingRanges,
  validateDisjointWritableViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {getGPUSpatialWeightsChunks} from '../spatial-weights/gpu-chunked-spatial-weights';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getLocalPermutationKernelWGSL,
  getLocalPermutationReplayInputs,
  type GPULocalPermutationTest
} from './gpu-local-permutation-test';

/** Recording-only inputs for one row's GPU replay. */
export type LocalPermutationReplayProps = {
  /** Node prefix; defaults to `${test.id}-replay`. */
  id?: string;
  /** Whole-CSR test (one chunk covering all rows); lower it on the same graph before the replay. */
  test: GPULocalPermutationTest;
  /** One packed uint32 row index, rewritable each frame. */
  row: GraphDataView<'uint32'>;
  /** Caller-owned sinks. */
  output: {
    /** At least maximumPermutations floats; only the reported prefix is valid. */
    nullDistribution: GraphDataView<'float32'>;
    /**
     * At least three words: [raw observed f32, replay count f32, pinned row u32 bits].
     * Count is zero for an invalid/unavailable row; its distribution must not be displayed.
     * The third word is internal state, read as uint32 rather than float32.
     */
    replayInfo: GraphDataView<'float32'>;
  };
};

/**
 * Replays all P raw local terms for one pinned row with the test's identical Philox draws.
 *
 * Borrows the test's prepared views instead of rebuilding its compaction, moments or observed
 * statistic. Add the test first, then this producer, on a graph with `recording` enabled.
 * For post-run inspection, internal scenarios use `getLocalPermutationReplayInputs(test)` to list
 * the bindings and observed-value buffer IDs in `recording.retain`. Ordinary graph dependencies
 * already preserve these inputs until replay finishes; retain additionally permits later captures.
 * Sequential tests still replay all P draws: compare only the first permutationCounts[row] terms
 * when explaining their stopping decision. Folded tests need only one replay sweep.
 *
 * Captures must use provenance `gpu-replay` and meaning
 * `null distribution (raw local term) of row <r>`; views draw this provenance hatched.
 * This instrument is deliberately absent from every public barrel. @internal
 */
export class GPULocalPermutationReplay implements GPUCommandNodeProducer {
  /** Prefix for setup and draw nodes. */
  readonly id: string;
  /** Validated replay inputs and sinks. */
  readonly props: LocalPermutationReplayProps;

  constructor(props: LocalPermutationReplayProps) {
    const snapshot = snapshotConstructionFields(props, ['id', 'test', 'row', 'output']);
    props = Object.freeze({
      ...snapshot,
      output: snapshotConstructionFields(snapshot.output, ['nullDistribution', 'replayInfo'])
    });
    this.id = props.id ?? `${props.test.id}-replay`;
    this.props = props;
    const chunks = getGPUSpatialWeightsChunks(props.test.props.weights);
    validateRequirement(
      this.id,
      chunks.length === 1 &&
        chunks[0].rowStart === 0 &&
        chunks[0].rowCount === props.test.props.values.length,
      'test.weights must use whole-CSR weights (one chunk covering all test rows); multi-chunk replay is unsupported'
    );
    validatePackedUint32View(props.row, `${this.id} row`);
    validateCapacity(this.id, 'row length', props.row.length, 1, 1);
    validateSinglePackedView(
      props.output.nullDistribution,
      ['float32'],
      `${this.id} nullDistribution`
    );
    validateCapacity(
      this.id,
      'nullDistribution length',
      props.output.nullDistribution.length,
      props.test.props.maximumPermutations
    );
    validateSinglePackedView(props.output.replayInfo, ['float32'], `${this.id} replayInfo`);
    validateCapacity(this.id, 'replayInfo length', props.output.replayInfo.length, 3);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [props.row]);
    validateDisjointWritableViews(this.id, props.output);
  }

  /** Returns setup and a single-invocation replay, after the test's preparation nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const inputs = getLocalPermutationReplayInputs(props.test);
    validateRequirement(
      id,
      inputs !== undefined,
      'replay requires its test to have been lowered first (call the test getCommandNodes before the replay)'
    );
    const reads = [
      ...inputs.bindings.map(binding => binding.view),
      inputs.observedRaw,
      props.row,
      ...(inputs.negativeCount ? [inputs.negativeCount] : [])
    ];
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [...reads, ...Object.values(props.output)].map((view, index) => [`binding ${index}`, view])
      )
    );
    validateGraphViewsBelongToGraph(id, graph, [...reads, ...Object.values(props.output)]);
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), reads);
    const {declarations, body} = getLocalPermutationKernelWGSL(inputs.kernel, 'replay');
    const replayInfo = {
      name: 'replayInfo',
      view: props.output.replayInfo,
      type: 'u32' as const,
      access: 'read_write' as const
    };
    return [
      // Packing the pin and observed term leaves the draw kernel at eight storage bindings.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-setup`,
        operation: 'GPULocalPermutationReplay',
        variant: 'setup',
        bindings: [
          {name: 'row', view: props.row, type: 'u32', access: 'read'},
          {name: 'observedRaw', view: inputs.observedRaw, type: 'f32', access: 'read'},
          replayInfo,
          ...(inputs.negativeCount
            ? [
                {
                  name: 'negativeCount',
                  view: inputs.negativeCount,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        invocationCount: 1,
        // Negative data is rejected by the live test (no row is tested), so its replay is unavailable too.
        body: `let pinnedRow = row[rowOffset];
  replayInfo[replayInfoOffset] = 0x7fc00000u;
  replayInfo[replayInfoOffset + 1u] = 0u;
  replayInfo[replayInfoOffset + 2u] = pinnedRow;
  if (pinnedRow < ${props.test.props.values.length}u${inputs.negativeCount ? ' && negativeCount[negativeCountOffset] == 0u' : ''}) {
    replayInfo[replayInfoOffset] = bitcast<u32>(observedRaw[observedRawOffset + pinnedRow]);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-permute`,
        operation: 'GPULocalPermutationReplay',
        variant: 'replay',
        bindings: [
          ...inputs.bindings,
          {
            name: 'nullDistribution',
            view: props.output.nullDistribution,
            type: 'f32',
            access: 'read_write'
          },
          replayInfo
        ],
        invocationCount: 1,
        // The shared loop's index is the pinned row, never the replay dispatch's invocation zero.
        declarations: `${declarations}
fn replayRow(index: u32) {
  ${body}
}`,
        body: 'replayRow(replayInfo[replayInfoOffset + 2u]);'
      })
    ];
  }
}

/**
 * Creates a recording-only replay of one row's local permutation draws.
 * Requires whole-CSR weights: one chunk covering all rows of the test. Multiple row chunks
 * are unsupported and rejected at construction; lowering the test again cannot enable replay.
 * Add the test first on the same recording graph. The returned producer borrows its prepared
 * inputs; capture the output as `gpu-replay`, and suppress its distribution when replayInfo's
 * count is zero. Sequential tests replay the full budget; use the test's permutationCounts
 * to identify the prefix actually used for its stopping decision.
 */
export function createLocalPermutationReplay(
  props: LocalPermutationReplayProps
): GPUCommandNodeProducer {
  return new GPULocalPermutationReplay(props);
}

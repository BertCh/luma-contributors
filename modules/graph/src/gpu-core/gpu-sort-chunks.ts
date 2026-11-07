// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from './gpu-command-graph';
import type {GPUCommandNode} from './gpu-command-node';
import {GPUSort, getGPUSortCommandNodesWithDispatchLimit} from './gpu-sort';
import {alignGraphVectorViews} from './graph-vector-view-utils';
import {createTransientView, getViewElementOffset} from './graph-data-view-utils';
import {
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource
} from './gpu-dispatch-utils';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBinding
} from './gpu-kernel-node';

const MERGE_WORKGROUP_SIZE = 256;

type SortedRun = {keys: GraphDataView<'uint32'>; values: GraphDataView<'uint32'>};

/**
 * Stable global merge of independently sorted borrowed spans, without concatenation.
 *
 * Each span is sorted on its own, then adjacent runs are merged pairwise in a balanced tree: a
 * merge dispatch ranks every element of the earlier run among the later run with a binary search
 * for strictly preceding keys, ranks every element of the later run among the earlier run for
 * preceding-or-equal keys, and writes each element at `ownIndex + rank`. Ties therefore keep input
 * order. Merging `m` spans takes `m - 1` dispatches over `log2(m)` levels rather than all pairs.
 * @internal
 */
export function getChunkedSortNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: GPUSort,
  maximum: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const spans = alignGraphVectorViews(graph, [sort.keys, sort.values]);
  const outputs = alignGraphVectorViews(graph, [sort.outputKeys, sort.outputValues]);
  const nonEmptyOutputs = outputs.filter(([outputKeys]) => outputKeys.length > 0);
  // A single output chunk holding every row is written directly by the final sort or merge.
  const directOutput =
    nonEmptyOutputs.length === 1 && nonEmptyOutputs[0][0].length === sort.keys.length
      ? nonEmptyOutputs[0]
      : undefined;
  const nonEmptySpans = spans.filter(([keys]) => keys.length > 0);

  let runs: SortedRun[] = nonEmptySpans.map(([keys, values], index) => {
    const id = `${sort.id}-span-${index}`;
    const writesOutput = directOutput && nonEmptySpans.length === 1;
    const outputKeys = writesOutput
      ? directOutput[0]
      : createTransientView(graph, `${id}-keys`, 'uint32', keys.length);
    const outputValues = writesOutput
      ? directOutput[1]
      : createTransientView(graph, `${id}-values`, 'uint32', keys.length);
    nodes.push(
      ...getGPUSortCommandNodesWithDispatchLimit(
        new GPUSort({
          id,
          keys,
          values,
          outputKeys,
          outputValues,
          algorithm: sort.resolvedAlgorithm,
          direction: sort.direction,
          keyBits: sort.keyBits,
          digitBits: sort.digitBits,
          elementsPerThread: sort.elementsPerThread
        }),
        graph,
        maximum
      )
    );
    return {keys: outputKeys, values: outputValues};
  });

  // Only radix-style sorts order on a prefix of the key bits.
  const mask =
    sort.resolvedAlgorithm !== 'bitonic' && sort.keyBits < 32 ? 2 ** sort.keyBits - 1 : 0xffffffff;
  for (let level = 0; runs.length > 1; level++) {
    const nextRuns: SortedRun[] = [];
    for (let index = 0; index < runs.length; index += 2) {
      if (index + 1 >= runs.length) {
        nextRuns.push(runs[index]);
        continue;
      }
      const isFinalMerge = runs.length === 2;
      const length = runs[index].keys.length + runs[index + 1].keys.length;
      const id = `${sort.id}-merge-${level}-${index / 2}`;
      const merged: SortedRun =
        isFinalMerge && directOutput
          ? {keys: directOutput[0], values: directOutput[1]}
          : {
              keys: createTransientView(graph, `${id}-keys`, 'uint32', length),
              values: createTransientView(graph, `${id}-values`, 'uint32', length)
            };
      nodes.push(
        addMergePass(graph, {
          id,
          earlier: runs[index],
          later: runs[index + 1],
          merged,
          ascending: sort.direction === 'ascending',
          mask,
          maximum
        })
      );
      nextRuns.push(merged);
    }
    runs = nextRuns;
  }

  // Publish the merged run into the caller's output chunks unless it was written there directly.
  const [finalRun] = runs;
  if (finalRun && !directOutput) {
    let outputStart = 0;
    for (const [outputIndex, [outputKeys, outputValues]] of outputs.entries()) {
      if (outputKeys.length > 0) {
        nodes.push(
          addPublishPass(graph, {
            id: `${sort.id}-publish-${outputIndex}`,
            run: finalRun,
            outputKeys,
            outputValues,
            outputStart,
            maximum
          })
        );
      }
      outputStart += outputKeys.length;
    }
  }
  return nodes;
}

/** Merges two sorted runs into one run of their combined length. */
function addMergePass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    earlier: SortedRun;
    later: SortedRun;
    merged: SortedRun;
    ascending: boolean;
    mask: number;
    maximum: number;
  }
): GPUCommandNode<Parameters> {
  const {earlier, later, merged} = props;
  const length = earlier.keys.length + later.keys.length;
  const dispatchLayout = getBoundedDispatchLayout(
    'GPUSort merge',
    length,
    MERGE_WORKGROUP_SIZE,
    props.maximum
  );
  const bindings: GPUKernelBinding[] = [
    {name: 'earlierKeys', view: earlier.keys, access: 'read'},
    {name: 'earlierValues', view: earlier.values, access: 'read'},
    {name: 'laterKeys', view: later.keys, access: 'read'},
    {name: 'laterValues', view: later.values, access: 'read'},
    {name: 'mergedKeys', view: merged.keys, access: 'write'},
    {name: 'mergedValues', view: merged.values, access: 'write'}
  ];
  const constants = {
    EARLIER_LENGTH: earlier.keys.length,
    LATER_LENGTH: later.keys.length,
    KEY_MASK: props.mask,
    EARLIER_KEYS_OFFSET: getViewElementOffset(earlier.keys),
    EARLIER_VALUES_OFFSET: getViewElementOffset(earlier.values),
    LATER_KEYS_OFFSET: getViewElementOffset(later.keys),
    LATER_VALUES_OFFSET: getViewElementOffset(later.values),
    MERGED_KEYS_OFFSET: getViewElementOffset(merged.keys),
    MERGED_VALUES_OFFSET: getViewElementOffset(merged.values),
    ...getBoundedDispatchOverrides(dispatchLayout)
  };
  // 'before' is the comparison that places a key ahead of another in the requested order.
  const before = props.ascending ? '<' : '>';
  return createGPUKernelNode(graph, {
    id: props.id,
    operation: 'GPUSort',
    variant: 'chunk-merge',
    moduleId: 'GPUSort/chunk-merge',
    dedupName: 'chunk-merge',
    source: /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

// Counts later-run keys that come before 'key' (or equal to it when 'includeEqual').
fn countLaterBefore(key: u32, includeEqual: bool) -> u32 {
  var first = 0u;
  var last = LATER_LENGTH;
  loop {
    if (first == last) { break; }
    let middle = first + (last - first) / 2u;
    let other = laterKeys[LATER_KEYS_OFFSET + middle] & KEY_MASK;
    if (other ${before} key || (includeEqual && other == key)) { first = middle + 1u; } else { last = middle; }
  }
  return first;
}

fn countEarlierBefore(key: u32, includeEqual: bool) -> u32 {
  var first = 0u;
  var last = EARLIER_LENGTH;
  loop {
    if (first == last) { break; }
    let middle = first + (last - first) / 2u;
    let other = earlierKeys[EARLIER_KEYS_OFFSET + middle] & KEY_MASK;
    if (other ${before} key || (includeEqual && other == key)) { first = middle + 1u; } else { last = middle; }
  }
  return first;
}

@compute @workgroup_size(${MERGE_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexOverrideSource(MERGE_WORKGROUP_SIZE).body}
  if (index >= EARLIER_LENGTH + LATER_LENGTH) { return; }
  if (index < EARLIER_LENGTH) {
    // Equal keys in the later run follow this element, which came first in the input.
    let key = earlierKeys[EARLIER_KEYS_OFFSET + index];
    let position = index + countLaterBefore(key & KEY_MASK, false);
    mergedKeys[MERGED_KEYS_OFFSET + position] = key;
    mergedValues[MERGED_VALUES_OFFSET + position] = earlierValues[EARLIER_VALUES_OFFSET + index];
  } else {
    let laterIndex = index - EARLIER_LENGTH;
    let key = laterKeys[LATER_KEYS_OFFSET + laterIndex];
    let position = laterIndex + countEarlierBefore(key & KEY_MASK, true);
    mergedKeys[MERGED_KEYS_OFFSET + position] = key;
    mergedValues[MERGED_VALUES_OFFSET + position] = laterValues[LATER_VALUES_OFFSET + laterIndex];
  }
}`,
    constants,
    bindings,
    workgroupSize: MERGE_WORKGROUP_SIZE,
    dispatchLayout,
    invocationCount: length,
    workloadScope: 'recording'
  });
}

/** Copies the rows of a merged run that belong to one caller output chunk. */
function addPublishPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    run: SortedRun;
    outputKeys: GraphDataView<'uint32'>;
    outputValues: GraphDataView<'uint32'>;
    outputStart: number;
    maximum: number;
  }
): GPUCommandNode<Parameters> {
  const dispatchLayout = getBoundedDispatchLayout(
    'GPUSort merge',
    props.outputKeys.length,
    MERGE_WORKGROUP_SIZE,
    props.maximum
  );
  const bindings: GPUKernelBinding[] = [
    {name: 'keys', view: props.run.keys, access: 'read'},
    {name: 'values', view: props.run.values, access: 'read'},
    {name: 'outputKeys', view: props.outputKeys, access: 'write'},
    {name: 'outputValues', view: props.outputValues, access: 'write'}
  ];
  const constants = {
    ROW_COUNT: props.outputKeys.length,
    OUTPUT_START: props.outputStart,
    KEYS_OFFSET: getViewElementOffset(props.run.keys),
    VALUES_OFFSET: getViewElementOffset(props.run.values),
    OUTPUT_KEYS_OFFSET: getViewElementOffset(props.outputKeys),
    OUTPUT_VALUES_OFFSET: getViewElementOffset(props.outputValues),
    ...getBoundedDispatchOverrides(dispatchLayout)
  };
  return createGPUKernelNode(graph, {
    id: props.id,
    operation: 'GPUSort',
    variant: 'chunk-publish',
    moduleId: 'GPUSort/chunk-publish',
    dedupName: 'chunk-publish',
    source: /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

@compute @workgroup_size(${MERGE_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexOverrideSource(MERGE_WORKGROUP_SIZE).body}
  if (index >= ROW_COUNT) { return; }
  outputKeys[OUTPUT_KEYS_OFFSET + index] = keys[KEYS_OFFSET + OUTPUT_START + index];
  outputValues[OUTPUT_VALUES_OFFSET + index] = values[VALUES_OFFSET + OUTPUT_START + index];
}`,
    constants,
    bindings,
    workgroupSize: MERGE_WORKGROUP_SIZE,
    dispatchLayout,
    invocationCount: props.outputKeys.length,
    workloadScope: 'recording'
  });
}

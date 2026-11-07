// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {GPUCommandGraph, GraphVectorView, type GraphDataView} from './gpu-command-graph';
import {
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource,
  type GPUBoundedDispatchLayout
} from './gpu-dispatch-utils';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBinding
} from './gpu-kernel-node';
import {
  getGPUScanCommandNodesWithDispatchLimit,
  GPU_SCAN_SINGLE_WORKGROUP_LIMIT,
  GPUScan,
  type GPUScanProps
} from './gpu-scan';
import {getGPUShaderSubgroupStrategy} from './gpu-subgroup-utils';
import {
  createTransientVectorView,
  createTransientView,
  getViewElementOffset,
  validatePackedUint32View
} from './graph-data-view-utils';
import {alignGraphVectorViews} from './graph-vector-view-utils';

const COMPACTION_WORKGROUP_SIZE = 256;
/** Rows owned by one block-compaction thread. */
const BLOCK_ITEMS_PER_THREAD = 4;
/** Rows compacted by one block-compaction workgroup. */
const BLOCK_SIZE = COMPACTION_WORKGROUP_SIZE * BLOCK_ITEMS_PER_THREAD;
const MAXIMUM_SUBGROUP_COUNT = 64;

/** Packed uint32 graph data accepted by {@link GPUCompaction}. */
export type GPUCompactionInput = GraphDataView<'uint32'> | GraphVectorView<'uint32'>;

/** Properties for one graph-native stable compaction. */
export type GPUCompactionProps = {
  /** Prefix for generated graph node and transient resource IDs. */
  id?: string;
  /** Packed source values as one data view or an ordered vector. */
  input: GPUCompactionInput;
  /** Packed selection flags; zero rejects and any non-zero value accepts. */
  flags: GPUCompactionInput;
  /** Caller-owned destination with sufficient logical capacity. */
  output: GPUCompactionInput;
  /** Caller-owned view whose first row receives the accepted value count. */
  count: GraphDataView<'uint32'>;
  /**
   * Implementation. `block` (default) counts and ranks rows per 1,024-row block without a per-row
   * offset array, using subgroup prefix sums where available; it applies to single-view inputs
   * only, and vector inputs always use `scan`. `scan` scans flags into per-row offsets and
   * scatters, and honours `scanAlgorithm`.
   */
  strategy?: 'scan' | 'block';
  /** Scan algorithm used by the `scan` strategy and for vector inputs. Defaults to `GPUScan` default. */
  scanAlgorithm?: GPUScanProps['algorithm'];
};

/**
 * Stable graph-composed compaction of `uint32` values selected by flags.
 *
 * Single-view inputs use block compaction: one pass counts accepted rows per 1,024-row block, an
 * exclusive {@link GPUScan} turns block counts into block offsets and writes the total count, and a
 * scatter pass ranks rows inside each block (subgroup prefix sums on devices with subgroups, a
 * workgroup scan otherwise) and writes them in source order. Inputs of up to 1,024 rows compact in
 * one dispatch. No per-row offset array is ever materialized.
 *
 * Vector inputs are one logical sequence whose selected values fill the existing output topology;
 * they scan the flags into per-row offsets and scatter each input chunk through each output chunk.
 */
export class GPUCompaction {
  /** Prefix for generated graph node and transient resource IDs. */
  readonly id: string;
  /** Packed source values or ordered source vector. */
  readonly input: GPUCompactionInput;
  /** Packed selection flags aligned to the same logical rows as the input. */
  readonly flags: GPUCompactionInput;
  /** Caller-owned compacted destination with enough logical capacity. */
  readonly output: GPUCompactionInput;
  /** Caller-owned accepted-count destination. */
  readonly count: GraphDataView<'uint32'>;
  /** Requested implementation. */
  readonly strategy: 'scan' | 'block';
  /** Scan algorithm for the `scan` strategy. */
  readonly scanAlgorithm: GPUScanProps['algorithm'];

  /**
   * Creates and validates a stable compaction description.
   *
   * @throws If views are not packed `uint32` data, input/flag lengths differ, or an output lacks
   * capacity.
   */
  constructor(props: GPUCompactionProps) {
    this.id = props.id ?? 'gpu-compaction';
    this.input = props.input;
    this.flags = props.flags;
    this.output = props.output;
    this.count = props.count;
    this.strategy = props.strategy ?? 'block';
    this.scanAlgorithm = props.scanAlgorithm;
    validateCompactionInput(this.input, `${this.id} input`);
    validateCompactionInput(this.flags, `${this.id} flags`);
    validateCompactionInput(this.output, `${this.id} output`);
    validatePackedUint32View(this.count, `${this.id} count`);
    if (this.flags.length !== this.input.length) {
      throw new Error(`${this.id} flags.length must equal input.length`);
    }
    if (this.output.length < this.input.length) {
      throw new Error(`${this.id} output must contain at least input.length rows`);
    }
    if (this.count.length < 1) {
      throw new Error(`${this.id} count must contain one uint32 row`);
    }
  }

  /**
   * Adds the block or scan-and-scatter passes to the target graph.
   *
   * Empty input adds only a pass that writes a zero count. This method does not submit or read back
   * commands.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      ...getGPUCompactionCommandNodesWithDispatchLimit(
        this,
        graph,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );

    return nodes;
  }
}

/** Adds block or scan-and-scatter compaction passes with an explicit dispatch limit. @internal */
export function getGPUCompactionCommandNodesWithDispatchLimit<Parameters>(
  compaction: GPUCompaction,
  graph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (const view of [
    ...getCompactionChunks(compaction.input),
    ...getCompactionChunks(compaction.flags),
    ...getCompactionChunks(compaction.output),
    compaction.count
  ]) {
    if (view.buffer.graph !== graph) {
      throw new Error(`${compaction.id} views must belong to the target graph`);
    }
  }

  if (compaction.input.length === 0) {
    nodes.push(...addClearCountPass(graph, compaction.id, compaction.count));
    return nodes;
  }

  const {input, flags, output} = compaction;
  const isSingleView =
    !(input instanceof GraphVectorView) &&
    !(flags instanceof GraphVectorView) &&
    !(output instanceof GraphVectorView);
  if (isSingleView && compaction.strategy === 'block') {
    nodes.push(
      ...addBlockCompaction(
        graph,
        compaction.id,
        input,
        flags,
        output,
        compaction.count,
        maxComputeWorkgroupsPerDimension
      )
    );
    return nodes;
  }

  const offsets =
    compaction.flags instanceof GraphVectorView
      ? createTransientVectorView(graph, `${compaction.id}-offsets`, compaction.flags)
      : createTransientView(graph, `${compaction.id}-offsets`, 'uint32', compaction.flags.length);
  // Any nonzero flag selects its row, so the scan counts min(flag, 1) directly, and the scan's own
  // total is the accepted count.
  const scan = new GPUScan({
    id: `${compaction.id}-scan`,
    input: compaction.flags,
    output: offsets,
    countNonzeroInput: true,
    total: compaction.count,
    algorithm: compaction.scanAlgorithm
  });
  nodes.push(
    ...getGPUScanCommandNodesWithDispatchLimit(scan, graph, maxComputeWorkgroupsPerDimension)
  );
  nodes.push(
    ...addScatterPasses(
      graph,
      compaction.id,
      compaction.input,
      compaction.flags,
      offsets,
      compaction.output,
      maxComputeWorkgroupsPerDimension
    )
  );

  return nodes;
}

/** Writes the required zero count for an empty input. */
function addClearCountPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  count: GraphDataView<'uint32'>
): readonly GPUCommandNode<Parameters>[] {
  const bindings: GPUKernelBinding[] = [{name: 'outputCount', view: count, access: 'write'}];
  const constants = {COUNT_OFFSET: getViewElementOffset(count)};
  return [
    createGPUKernelNode(graph, {
      id: `${id}-clear-count`,
      operation: 'GPUCompaction',
      moduleId: 'GPUCompaction/clear-count',
      formula: 'gpu-core.compaction',
      source: /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(1) fn main() { outputCount[COUNT_OFFSET] = 0u; }`,
      constants,
      bindings,
      workgroupSize: 1,
      dispatchLayout: {x: 1, y: 1, z: 1},
      invocationCount: 1
    })
  ];
}

/**
 * Compacts single-view inputs without per-row offsets: count per block, scan block counts (whose
 * total is the accepted count), then rank and write inside each block.
 */
function addBlockCompaction<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: GraphDataView<'uint32'>,
  flags: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>,
  count: GraphDataView<'uint32'>,
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const length = input.length;
  const blockCount = Math.ceil(length / BLOCK_SIZE);
  const strategy = getGPUShaderSubgroupStrategy(graph.device, {requiresSubgroupId: true});
  const dispatchLayout = getBoundedDispatchLayout(
    'GPUCompaction',
    length,
    BLOCK_SIZE,
    maxComputeWorkgroupsPerDimension
  );

  let blockOffsets: GraphDataView<'uint32'> | undefined;
  if (blockCount > 1) {
    const blockCounts = createTransientView(graph, `${id}-block-counts`, 'uint32', blockCount);
    blockOffsets = createTransientView(graph, `${id}-block-offsets`, 'uint32', blockCount);
    nodes.push(
      addBlockCountPass(graph, {
        id,
        flags,
        blockCounts,
        length,
        blockCount,
        dispatchLayout,
        strategy
      })
    );
    nodes.push(
      ...getGPUScanCommandNodesWithDispatchLimit(
        new GPUScan({
          id: `${id}-scan`,
          input: blockCounts,
          output: blockOffsets,
          total: count,
          singleWorkgroupMaximumLength: GPU_SCAN_SINGLE_WORKGROUP_LIMIT
        }),
        graph,
        maxComputeWorkgroupsPerDimension
      )
    );
  }
  nodes.push(
    addBlockScatterPass(graph, {
      id,
      input,
      flags,
      blockOffsets,
      output,
      // A single block has no scan, so its scatter pass reports the count.
      count: blockCount === 1 ? count : undefined,
      length,
      blockCount,
      dispatchLayout,
      strategy
    })
  );
  return nodes;
}

/** WGSL that gives thread `lane` its four rows' linear block index guard and first row. */
function getBlockGuardSource(): string {
  return `let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  if (workgroupIndex >= BLOCK_COUNT) { return; }
  let firstRow = workgroupIndex * ${BLOCK_SIZE}u + lane * ${BLOCK_ITEMS_PER_THREAD}u;`;
}

/** Counts accepted rows in every block. */
function addBlockCountPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    flags: GraphDataView<'uint32'>;
    blockCounts: GraphDataView<'uint32'>;
    length: number;
    blockCount: number;
    dispatchLayout: GPUBoundedDispatchLayout;
    strategy: 'portable' | 'subgroups';
  }
) {
  const bindings: GPUKernelBinding[] = [
    {name: 'flags', view: props.flags, access: 'read'},
    {name: 'blockCounts', view: props.blockCounts, access: 'write'}
  ];
  const constants = {
    ELEMENT_COUNT: props.length,
    BLOCK_COUNT: props.blockCount,
    FLAGS_OFFSET: getViewElementOffset(props.flags),
    BLOCK_COUNTS_OFFSET: getViewElementOffset(props.blockCounts),
    ...getBoundedDispatchOverrides(props.dispatchLayout)
  };
  const declarations = `${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}`;
  const countRows = `var rowCount = 0u;
  for (var item = 0u; item < ${BLOCK_ITEMS_PER_THREAD}u; item++) {
    if (firstRow + item < ELEMENT_COUNT) {
      rowCount += min(flags[FLAGS_OFFSET + firstRow + item], 1u);
    }
  }`;
  const source =
    props.strategy === 'subgroups'
      ? /* wgsl */ `
enable subgroups;
requires subgroup_id;

${declarations}
var<workgroup> subgroupTotals: array<u32, ${MAXIMUM_SUBGROUP_COUNT}>;

@compute @workgroup_size(${COMPACTION_WORKGROUP_SIZE}) fn main(
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_id) subgroupId: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let lane = subgroupId * subgroupSize + subgroupInvocationId;
  ${getBlockGuardSource()}
  ${countRows}
  let subgroupTotal = subgroupAdd(rowCount);
  if (subgroupInvocationId == 0u) { subgroupTotals[subgroupId] = subgroupTotal; }
  workgroupBarrier();
  if (lane == 0u) {
    var blockTotal = 0u;
    for (var subgroupIndex = 0u; subgroupIndex < ${COMPACTION_WORKGROUP_SIZE}u / subgroupSize; subgroupIndex++) {
      blockTotal += subgroupTotals[subgroupIndex];
    }
    blockCounts[BLOCK_COUNTS_OFFSET + workgroupIndex] = blockTotal;
  }
}`
      : /* wgsl */ `
${declarations}
var<workgroup> scratch: array<u32, ${COMPACTION_WORKGROUP_SIZE}>;

@compute @workgroup_size(${COMPACTION_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBlockGuardSource()}
  ${countRows}
  scratch[lane] = rowCount;
  workgroupBarrier();
  for (var stride = ${COMPACTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { scratch[lane] = scratch[lane] + scratch[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u) { blockCounts[BLOCK_COUNTS_OFFSET + workgroupIndex] = scratch[0]; }
}`;
  return createGPUKernelNode(graph, {
    id: `${props.id}-block-count`,
    operation: 'GPUCompaction',
    variant: props.strategy,
    moduleId: `GPUCompaction/block-count/${props.strategy}`,
    formula: 'gpu-core.compaction',
    source,
    constants,
    bindings,
    workgroupSize: COMPACTION_WORKGROUP_SIZE,
    dispatchLayout: props.dispatchLayout,
    invocationCount: props.length
  });
}

/** Ranks accepted rows inside each block and writes them after the block's global offset. */
function addBlockScatterPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    input: GraphDataView<'uint32'>;
    flags: GraphDataView<'uint32'>;
    blockOffsets?: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
    count?: GraphDataView<'uint32'>;
    length: number;
    blockCount: number;
    dispatchLayout: GPUBoundedDispatchLayout;
    strategy: 'portable' | 'subgroups';
  }
) {
  const bindings: GPUKernelBinding[] = [
    {name: 'inputValues', view: props.input, access: 'read'},
    {name: 'flags', view: props.flags, access: 'read'}
  ];
  if (props.blockOffsets) {
    bindings.push({name: 'blockOffsets', view: props.blockOffsets, access: 'read'});
  }
  bindings.push({name: 'outputValues', view: props.output, access: 'write'});
  if (props.count) bindings.push({name: 'outputCount', view: props.count, access: 'write'});
  const constants: Record<string, number> = {
    ELEMENT_COUNT: props.length,
    BLOCK_COUNT: props.blockCount,
    INPUT_OFFSET: getViewElementOffset(props.input),
    FLAGS_OFFSET: getViewElementOffset(props.flags),
    OUTPUT_OFFSET: getViewElementOffset(props.output),
    ...(props.blockOffsets && {BLOCK_OFFSETS_OFFSET: getViewElementOffset(props.blockOffsets)}),
    ...(props.count && {COUNT_OFFSET: getViewElementOffset(props.count)}),
    ...getBoundedDispatchOverrides(props.dispatchLayout)
  };
  const declarations = `${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}`;
  const loadRows = `var rowFlags: array<u32, ${BLOCK_ITEMS_PER_THREAD}>;
  var rowValues: array<u32, ${BLOCK_ITEMS_PER_THREAD}>;
  var rowCount = 0u;
  for (var item = 0u; item < ${BLOCK_ITEMS_PER_THREAD}u; item++) {
    if (firstRow + item < ELEMENT_COUNT) {
      let flag = min(flags[FLAGS_OFFSET + firstRow + item], 1u);
      rowFlags[item] = flag;
      rowCount += flag;
      if (flag != 0u) { rowValues[item] = inputValues[INPUT_OFFSET + firstRow + item]; }
    }
  }`;
  const writeRows = `var outputIndex = ${props.blockOffsets ? 'blockOffsets[BLOCK_OFFSETS_OFFSET + workgroupIndex] + ' : ''}rowPrefix;
  for (var item = 0u; item < ${BLOCK_ITEMS_PER_THREAD}u; item++) {
    if (rowFlags[item] != 0u) {
      outputValues[OUTPUT_OFFSET + outputIndex] = rowValues[item];
      outputIndex += 1u;
    }
  }`;
  const source =
    props.strategy === 'subgroups'
      ? /* wgsl */ `
enable subgroups;
requires subgroup_id;

${declarations}
var<workgroup> subgroupTotals: array<u32, ${MAXIMUM_SUBGROUP_COUNT}>;

@compute @workgroup_size(${COMPACTION_WORKGROUP_SIZE}) fn main(
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_id) subgroupId: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let lane = subgroupId * subgroupSize + subgroupInvocationId;
  ${getBlockGuardSource()}
  ${loadRows}
  let subgroupPrefix = subgroupExclusiveAdd(rowCount);
  let subgroupTotal = subgroupAdd(rowCount);
  if (subgroupInvocationId == 0u) { subgroupTotals[subgroupId] = subgroupTotal; }
  workgroupBarrier();
  if (lane == 0u) {
    var runningTotal = 0u;
    for (var subgroupIndex = 0u; subgroupIndex < ${COMPACTION_WORKGROUP_SIZE}u / subgroupSize; subgroupIndex++) {
      let total = subgroupTotals[subgroupIndex];
      subgroupTotals[subgroupIndex] = runningTotal;
      runningTotal += total;
    }
    ${props.count ? 'outputCount[COUNT_OFFSET] = runningTotal;' : ''}
  }
  workgroupBarrier();
  let rowPrefix = subgroupTotals[subgroupId] + subgroupPrefix;
  ${writeRows}
}`
      : /* wgsl */ `
${declarations}
var<workgroup> scratch: array<u32, ${COMPACTION_WORKGROUP_SIZE}>;

@compute @workgroup_size(${COMPACTION_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBlockGuardSource()}
  ${loadRows}
  scratch[lane] = rowCount;
  workgroupBarrier();
  for (var stride = 1u; stride < ${COMPACTION_WORKGROUP_SIZE}u; stride = stride * 2u) {
    var addend = 0u;
    if (lane >= stride) { addend = scratch[lane - stride]; }
    workgroupBarrier();
    if (lane >= stride) { scratch[lane] = scratch[lane] + addend; }
    workgroupBarrier();
  }
  let rowPrefix = scratch[lane] - rowCount;
  ${props.count ? `if (lane == ${COMPACTION_WORKGROUP_SIZE - 1}u) { outputCount[COUNT_OFFSET] = scratch[lane]; }` : ''}
  ${writeRows}
}`;
  return createGPUKernelNode(graph, {
    id: `${props.id}-scatter`,
    operation: 'GPUCompaction',
    variant: props.strategy,
    moduleId: `GPUCompaction/block-scatter/${props.strategy}`,
    formula: 'gpu-core.compaction',
    source,
    constants,
    bindings,
    workgroupSize: COMPACTION_WORKGROUP_SIZE,
    dispatchLayout: props.dispatchLayout,
    invocationCount: props.length
  });
}

/** Routes every non-empty input chunk through each non-empty logical output range. */
function addScatterPasses<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: GPUCompactionInput,
  flags: GPUCompactionInput,
  offsets: GPUCompactionInput,
  output: GPUCompactionInput,
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const sourceSpans = alignGraphVectorViews(graph, [input, flags, offsets]);
  const inputChunks = sourceSpans.map(span => span[0]);
  const flagChunks = sourceSpans.map(span => span[1]);
  const offsetChunks = sourceSpans.map(span => span[2]);
  const outputChunks = getCompactionChunks(output);
  const inputChunkIndices = inputChunks
    .map((chunk, chunkIndex) => (chunk.length > 0 ? chunkIndex : -1))
    .filter(chunkIndex => chunkIndex >= 0);
  const isVector = sourceSpans.length > 1 || output instanceof GraphVectorView;

  let outputStart = 0;
  for (let outputChunkIndex = 0; outputChunkIndex < outputChunks.length; outputChunkIndex++) {
    const outputChunk = outputChunks[outputChunkIndex];
    const outputEnd = outputStart + outputChunk.length;
    if (outputChunk.length > 0) {
      for (const inputChunkIndex of inputChunkIndices) {
        nodes.push(
          addScatterPass(graph, {
            id: isVector
              ? `${id}-scatter-input-${inputChunkIndex}-output-${outputChunkIndex}`
              : `${id}-scatter`,
            input: inputChunks[inputChunkIndex],
            flags: flagChunks[inputChunkIndex],
            offsets: offsetChunks[inputChunkIndex],
            output: outputChunk,
            outputStart,
            outputEnd,
            dispatchLayout: getBoundedDispatchLayout(
              'GPUCompaction',
              inputChunks[inputChunkIndex].length,
              COMPACTION_WORKGROUP_SIZE,
              maxComputeWorkgroupsPerDimension
            )
          })
        );
      }
    }
    outputStart = outputEnd;
  }

  return nodes;
}

/** Scatters one input chunk into one logical output range. */
function addScatterPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    input: GraphDataView<'uint32'>;
    flags: GraphDataView<'uint32'>;
    offsets: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
    outputStart: number;
    outputEnd: number;
    dispatchLayout: GPUBoundedDispatchLayout;
  }
) {
  const bindings: GPUKernelBinding[] = [
    {name: 'inputValues', view: props.input, access: 'read'},
    {name: 'flags', view: props.flags, access: 'read'},
    {name: 'offsets', view: props.offsets, access: 'read'},
    {name: 'outputValues', view: props.output, access: 'write'}
  ];
  const constants = {
    ELEMENT_COUNT: props.input.length,
    INPUT_OFFSET: getViewElementOffset(props.input),
    FLAGS_OFFSET: getViewElementOffset(props.flags),
    OFFSETS_OFFSET: getViewElementOffset(props.offsets),
    OUTPUT_OFFSET: getViewElementOffset(props.output),
    OUTPUT_START: props.outputStart,
    OUTPUT_END: props.outputEnd,
    ...getBoundedDispatchOverrides(props.dispatchLayout)
  };
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

@compute @workgroup_size(${COMPACTION_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexOverrideSource(COMPACTION_WORKGROUP_SIZE).body}
  if (index >= ELEMENT_COUNT) { return; }
  let flag = min(flags[FLAGS_OFFSET + index], 1u);
  let outputIndex = offsets[OFFSETS_OFFSET + index];
  if (flag != 0u && outputIndex >= OUTPUT_START && outputIndex < OUTPUT_END) {
    outputValues[OUTPUT_OFFSET + outputIndex - OUTPUT_START] = inputValues[INPUT_OFFSET + index];
  }
}`;
  return createGPUKernelNode(graph, {
    id: props.id,
    operation: 'GPUCompaction',
    moduleId: 'GPUCompaction/scatter',
    formula: 'gpu-core.compaction',
    source,
    constants,
    bindings,
    workgroupSize: COMPACTION_WORKGROUP_SIZE,
    dispatchLayout: props.dispatchLayout,
    invocationCount: props.input.length
  });
}

/** Normalizes one compaction input or vector into its ordered atomic chunks. */
function getCompactionChunks(input: GPUCompactionInput): readonly GraphDataView<'uint32'>[] {
  return input instanceof GraphVectorView ? input.data : [input];
}

/** Validates every atomic chunk accepted by a compaction input. */
function validateCompactionInput(input: GPUCompactionInput, name: string): void {
  for (const chunk of getCompactionChunks(input)) {
    validatePackedUint32View(chunk, name);
  }
}

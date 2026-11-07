// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView, GraphVectorView} from './gpu-command-graph';
import {
  type GPUBoundedDispatchLayout,
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource,
  getBoundedInvocationIndexSource
} from './gpu-dispatch-utils';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBinding
} from './gpu-kernel-node';
import {
  createTransientView,
  getViewElementOffset,
  validatePackedUint32View
} from './graph-data-view-utils';
import {alignGraphVectorViews} from './graph-vector-view-utils';
import {getGPUShaderSubgroupStrategy} from './gpu-subgroup-utils';
import {addDecoupledLookbackScan} from './gpu-scan-lookback';

const SCAN_WORKGROUP_SIZE = 256;
const MAXIMUM_SCAN_SUBGROUP_COUNT = 64;
/**
 * Largest input a single workgroup can scan: 16 tiles of 256 rows. One dispatch then replaces the
 * three-dispatch block, summary, and offset hierarchy (for example radix histograms).
 * @internal
 */
export const GPU_SCAN_SINGLE_WORKGROUP_LIMIT = 4096;

/**
 * Scan implementation. `multi-pass` (default) is wait-free: block scans, summary scans, and offset
 * passes joined by dispatch boundaries. `decoupled-lookback` is a chained single-pass scan whose
 * workgroups hand prefixes to their successors through global memory; it falls back to recomputing
 * a predecessor's aggregate when that predecessor is slow, so it never requires inter-workgroup
 * forward progress.
 */
export type GPUScanAlgorithm = 'multi-pass' | 'decoupled-lookback';

/** Compute strategy selected for an unsegmented scan block. @internal */
export type GPUScanStrategy = 'portable' | 'subgroups';

/** Selects the subgroup path only when both device and WGSL language capabilities are present. */
export function getGPUScanStrategy(device: Device, segmented: boolean = false): GPUScanStrategy {
  return segmented ? 'portable' : getGPUShaderSubgroupStrategy(device, {requiresSubgroupId: true});
}

/** Packed uint32 graph data accepted by {@link GPUScan}. */
export type GPUScanInput = GraphDataView<'uint32'> | GraphVectorView<'uint32'>;

/** Properties for one graph-native prefix sum. */
export type GPUScanProps = {
  /** Prefix for generated graph node and transient resource IDs. */
  id?: string;
  /** One packed unsigned data view or an ordered vector of packed chunks. */
  input: GPUScanInput;
  /** Caller-owned destination with at least the input length; chunk boundaries may differ. */
  output: GPUScanInput;
  /** Prefix convention. Defaults to `exclusive`. */
  mode?: 'exclusive' | 'inclusive';
  /** Optional nonzero flags that start new segments. Row zero is always a segment start. */
  segmentFlags?: GPUScanInput;
  /**
   * Scans `min(input, 1)` instead of the raw values, so any nonzero row counts as one. Compaction
   * uses this to scan selection flags without a separate normalization pass.
   * @internal
   */
  countNonzeroInput?: boolean;
  /**
   * Single-row view that receives the total of all scanned rows (the inclusive sum of the last
   * row), written by the final scan level without an extra pass. Unsegmented scans only.
   * @internal
   */
  total?: GraphDataView<'uint32'>;
  /**
   * Implementation. Defaults to `'multi-pass'`. `'decoupled-lookback'` scans in one chained pass;
   * with `segmentFlags` it applies to a single chunk, and vector inputs with segment flags use
   * `'multi-pass'` because carrying segment heads across chunks needs the hierarchical carries.
   */
  algorithm?: GPUScanAlgorithm;
  /**
   * Largest unsegmented input, in rows, that a single workgroup scans in one dispatch instead of the
   * block, summary, and offset hierarchy (maximum 4096). Defaults to `0`, which keeps the
   * hierarchy; GPUSort and GPUCompaction enable it for the small scans they compose.
   */
  singleWorkgroupMaximumLength?: number;
  /**
   * Published-state polls the look-back algorithm makes on a predecessor before recomputing it.
   * Zero forces the fallback for every predecessor, which tests use to cover that path.
   * @internal
   */
  lookbackMaximumSpins?: number;
};

/**
 * Hierarchical prefix sum over packed `uint32` graph data.
 *
 * Each 256-thread block writes local prefixes and, when necessary, one block summary. Higher
 * levels scan those summaries before reverse-order offset passes add parent offsets back into every
 * lower level. Optional nonzero flags restart the prefix without treating vector chunks as segment
 * boundaries. Arithmetic wraps modulo 2^32.
 */
export class GPUScan {
  /** Prefix for generated graph node and transient resource IDs. */
  readonly id: string;
  /** Packed unsigned source data or ordered source vector. */
  readonly input: GPUScanInput;
  /** Caller-owned prefix destination with matching view kind. */
  readonly output: GPUScanInput;
  /** Whether output rows exclude or include their corresponding input rows. */
  readonly mode: 'exclusive' | 'inclusive';
  /** Optional nonzero flags that start new segments. */
  readonly segmentFlags?: GPUScanInput;
  /** Whether every nonzero input row is scanned as one. @internal */
  readonly countNonzeroInput: boolean;
  /** Optional single-row total destination. @internal */
  readonly total?: GraphDataView<'uint32'>;
  /** Requested implementation. */
  readonly algorithm: GPUScanAlgorithm;
  /** Look-back polling budget override. @internal */
  readonly lookbackMaximumSpins?: number;
  /** Largest input scanned by one workgroup; zero keeps the hierarchy. */
  readonly singleWorkgroupMaximumLength: number;

  /**
   * Creates and validates a scan description.
   *
   * @throws If either view is not packed `uint32` data or the output is shorter than the input.
   */
  constructor(props: GPUScanProps) {
    this.id = props.id ?? 'gpu-scan';
    this.input = props.input;
    this.output = props.output;
    this.mode = props.mode ?? 'exclusive';
    this.segmentFlags = props.segmentFlags;
    this.countNonzeroInput = props.countNonzeroInput ?? false;
    this.total = props.total;
    this.algorithm = props.algorithm ?? 'multi-pass';
    this.lookbackMaximumSpins = props.lookbackMaximumSpins;
    this.singleWorkgroupMaximumLength = Math.min(
      props.singleWorkgroupMaximumLength ?? 0,
      GPU_SCAN_SINGLE_WORKGROUP_LIMIT
    );
    if (this.total) validatePackedUint32View(this.total, `${this.id} total`);
    if (this.total && this.segmentFlags) {
      throw new Error(`${this.id} total is only defined for unsegmented scans`);
    }
    validateScanInput(this.input, `${this.id} input`);
    validateScanInput(this.output, `${this.id} output`);
    if (this.output.length < this.input.length) {
      throw new Error(`${this.id} output must contain at least input.length rows`);
    }
    if (this.output instanceof GraphVectorView && this.output.length !== this.input.length) {
      throw new Error(`${this.id} vector output length must equal input length`);
    }
    if (this.segmentFlags) {
      validateScanInput(this.segmentFlags, `${this.id} segmentFlags`);
      if (this.segmentFlags.length < this.input.length) {
        throw new Error(`${this.id} segmentFlags must contain at least input.length rows`);
      }
      if (
        this.segmentFlags instanceof GraphVectorView &&
        this.segmentFlags.length !== this.input.length
      ) {
        throw new Error(`${this.id} vector segmentFlags length must equal input length`);
      }
      const outputBuffers = new Set(getScanChunks(this.output).map(chunk => chunk.buffer));
      if (getScanChunks(this.segmentFlags).some(chunk => outputBuffers.has(chunk.buffer))) {
        throw new Error(`${this.id} segmentFlags and output must use separate buffers`);
      }
    }
  }

  /**
   * Adds local scan hierarchies, vector carry propagation, and graph-owned scratch.
   *
   * Empty inputs add no nodes. This method declares work only; it does not compile, encode, submit,
   * or read data back.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      ...getGPUScanCommandNodesWithDispatchLimit(
        this,
        graph,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );

    return nodes;
  }
}

/** Adds a scan using an explicit per-dimension dispatch limit. @internal */
export function getGPUScanCommandNodesWithDispatchLimit<Parameters>(
  scan: GPUScan,
  graph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  validateScanOwnership(graph, scan.input, scan.id);
  validateScanOwnership(graph, scan.output, scan.id);
  if (scan.segmentFlags) {
    validateScanOwnership(graph, scan.segmentFlags, scan.id);
  }
  const output = getScanPrefixView(graph, scan.output, scan.input.length);
  const segmentFlags = scan.segmentFlags
    ? getScanPrefixView(graph, scan.segmentFlags, scan.input.length)
    : undefined;
  const views = segmentFlags
    ? alignGraphVectorViews(graph, [scan.input, output, segmentFlags])
    : alignGraphVectorViews(graph, [scan.input, output]);
  nodes.push(
    ...addChunkedScan(
      graph,
      {
        id: scan.id,
        inputChunks: views.map(span => span[0]),
        outputChunks: views.map(span => span[1]),
        segmentFlagChunks: segmentFlags ? views.map(span => span[2]!) : undefined,
        isVector: views.length > 1,
        mode: scan.mode,
        countNonzeroInput: scan.countNonzeroInput,
        total: scan.total,
        algorithm: scan.algorithm,
        lookbackMaximumSpins: scan.lookbackMaximumSpins,
        singleWorkgroupMaximumLength: scan.singleWorkgroupMaximumLength
      },
      maxComputeWorkgroupsPerDimension
    )
  );

  return nodes;
}

function getScanPrefixView<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GPUScanInput,
  length: number
): GPUScanInput {
  if (view.length === length || view instanceof GraphVectorView) {
    return view;
  }
  return graph.createDataView(view.buffer, {
    format: view.format,
    length,
    byteOffset: view.byteOffset,
    byteStride: view.byteStride,
    rowByteLength: view.rowByteLength
  });
}

/** Normalizes atomic and vector inputs and adds the required local scans and vector carries. */
function addChunkedScan<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    inputChunks: readonly GraphDataView<'uint32'>[];
    outputChunks: readonly GraphDataView<'uint32'>[];
    segmentFlagChunks?: readonly GraphDataView<'uint32'>[];
    isVector: boolean;
    mode: 'exclusive' | 'inclusive';
    countNonzeroInput: boolean;
    total?: GraphDataView<'uint32'>;
    algorithm: GPUScanAlgorithm;
    lookbackMaximumSpins?: number;
    singleWorkgroupMaximumLength: number;
  },
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const inputChunks = props.inputChunks;
  const outputChunks = props.outputChunks;
  const segmentFlagChunks = props.segmentFlagChunks;
  const nonEmptyChunks = inputChunks
    .map((inputChunk, chunkIndex) => ({
      chunkIndex,
      input: inputChunk,
      output: outputChunks[chunkIndex],
      segmentFlags: segmentFlagChunks?.[chunkIndex]
    }))
    .filter(chunk => chunk.input.length > 0);
  if (nonEmptyChunks.length === 0) {
    if (props.total) nodes.push(...addZeroTotalPass(graph, props.id, props.total));
    return nodes;
  }
  if (nonEmptyChunks.length === 1) {
    const chunk = nonEmptyChunks[0];
    nodes.push(
      ...addScanLevels(graph, {
        id: props.isVector ? `${props.id}-chunk-${chunk.chunkIndex}` : props.id,
        input: chunk.input,
        output: chunk.output,
        mode: props.mode,
        segmentFlags: chunk.segmentFlags,
        countNonzeroInput: props.countNonzeroInput,
        finalSum: props.total,
        algorithm: props.algorithm,
        lookbackMaximumSpins: props.lookbackMaximumSpins,
        singleWorkgroupMaximumLength: props.singleWorkgroupMaximumLength,
        maxComputeWorkgroupsPerDimension
      })
    );
    return nodes;
  }

  const chunkTotals = createTransientView(
    graph,
    `${props.id}-chunk-totals`,
    'uint32',
    nonEmptyChunks.length
  );
  const chunkOffsets = createTransientView(
    graph,
    `${props.id}-chunk-offsets`,
    'uint32',
    nonEmptyChunks.length
  );
  const chunkSegmentFlags = segmentFlagChunks
    ? createTransientView(graph, `${props.id}-chunk-segment-flags`, 'uint32', nonEmptyChunks.length)
    : undefined;
  const chunkSegmentPrefixes = segmentFlagChunks
    ? nonEmptyChunks.map(chunk =>
        createTransientView(
          graph,
          `${props.id}-chunk-${chunk.chunkIndex}-segment-prefixes`,
          'uint32',
          chunk.input.length
        )
      )
    : undefined;
  for (const [partialIndex, chunk] of nonEmptyChunks.entries()) {
    nodes.push(
      ...addScanLevels(graph, {
        id: `${props.id}-chunk-${chunk.chunkIndex}`,
        input: chunk.input,
        output: chunk.output,
        mode: props.mode,
        segmentFlags: chunk.segmentFlags,
        countNonzeroInput: props.countNonzeroInput,
        outputSegmentPrefixes: chunkSegmentPrefixes?.[partialIndex],
        algorithm: props.algorithm,
        lookbackMaximumSpins: props.lookbackMaximumSpins,
        singleWorkgroupMaximumLength: props.singleWorkgroupMaximumLength,
        finalSum: createPackedSubview(graph, chunkTotals, partialIndex),
        finalSegmentFlag: chunkSegmentFlags
          ? createPackedSubview(graph, chunkSegmentFlags, partialIndex)
          : undefined,
        maxComputeWorkgroupsPerDimension
      })
    );
  }
  nodes.push(
    ...addScanLevels(graph, {
      id: `${props.id}-chunk-carries`,
      input: chunkTotals,
      output: chunkOffsets,
      mode: 'exclusive',
      segmentFlags: chunkSegmentFlags,
      segmentSummaryInput: Boolean(chunkSegmentFlags),
      finalSum: props.total,
      algorithm: props.algorithm,
      lookbackMaximumSpins: props.lookbackMaximumSpins,
      singleWorkgroupMaximumLength: props.singleWorkgroupMaximumLength,
      maxComputeWorkgroupsPerDimension
    })
  );
  for (const [partialIndex, chunk] of nonEmptyChunks.entries()) {
    nodes.push(
      ...addOffsetPass(graph, {
        id: `${props.id}-chunk-${chunk.chunkIndex}-add-carry`,
        output: chunk.output,
        offsets: chunkOffsets,
        length: chunk.output.length,
        offsetIndex: partialIndex,
        segmentPrefixes: chunkSegmentPrefixes?.[partialIndex],
        dispatchLayout: getGPUScanDispatchLayout(
          chunk.output.length,
          maxComputeWorkgroupsPerDimension
        )
      })
    );
  }

  return nodes;
}

/** Adds every hierarchical level required to scan one non-empty packed data view. */
function addScanLevels<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    input: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
    mode: 'exclusive' | 'inclusive';
    segmentFlags?: GraphDataView<'uint32'>;
    /** Applies to the first level only; upper levels scan already-summed counts. */
    countNonzeroInput?: boolean;
    outputSegmentPrefixes?: GraphDataView<'uint32'>;
    finalSum?: GraphDataView<'uint32'>;
    finalSegmentFlag?: GraphDataView<'uint32'>;
    /** Summary flags report a start anywhere in one lower-level block, not at its first row. */
    segmentSummaryInput?: boolean;
    algorithm?: GPUScanAlgorithm;
    lookbackMaximumSpins?: number;
    singleWorkgroupMaximumLength?: number;
    maxComputeWorkgroupsPerDimension: number;
  }
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (props.input.length === 0) {
    return nodes;
  }
  const isPlainScan =
    !props.segmentFlags &&
    !props.outputSegmentPrefixes &&
    !props.finalSegmentFlag &&
    !props.segmentSummaryInput;
  // A segmented look-back scan owns its whole chunk; carries between chunks stay multi-pass.
  const isSingleChunkSegmentedScan =
    Boolean(props.segmentFlags) &&
    !props.outputSegmentPrefixes &&
    !props.finalSegmentFlag &&
    !props.segmentSummaryInput;
  if ((isPlainScan || isSingleChunkSegmentedScan) && props.algorithm === 'decoupled-lookback') {
    return addDecoupledLookbackScan(graph, {
      id: `${props.id}-level-0`,
      input: props.input,
      output: props.output,
      mode: props.mode,
      countNonzeroInput: props.countNonzeroInput,
      segmentFlags: props.segmentFlags,
      finalSum: props.finalSum,
      maximumSpins: props.lookbackMaximumSpins,
      maxComputeWorkgroupsPerDimension: props.maxComputeWorkgroupsPerDimension
    });
  }
  if (
    isPlainScan &&
    props.input.length > SCAN_WORKGROUP_SIZE &&
    props.input.length <= (props.singleWorkgroupMaximumLength ?? 0)
  ) {
    return addSingleWorkgroupScanPass(graph, {
      id: `${props.id}-level-0-scan`,
      input: props.input,
      output: props.output,
      mode: props.mode,
      countNonzeroInput: props.countNonzeroInput,
      finalSum: props.finalSum,
      length: props.input.length
    });
  }

  const levels: Array<{
    output: GraphDataView<'uint32'>;
    length: number;
    blockOffsets?: GraphDataView<'uint32'>;
    segmentPrefixes?: GraphDataView<'uint32'>;
  }> = [];
  let levelInput = props.input;
  let levelOutput = props.output;
  let levelSegmentFlags = props.segmentFlags;
  let levelLength = props.input.length;
  let levelIndex = 0;

  while (true) {
    const blockCount = Math.ceil(levelLength / SCAN_WORKGROUP_SIZE);
    let blockSums: GraphDataView<'uint32'> | undefined;
    let blockSegmentFlags: GraphDataView<'uint32'> | undefined;
    if (blockCount > 1) {
      blockSums = createTransientView(
        graph,
        `${props.id}-level-${levelIndex}-block-sums`,
        'uint32',
        blockCount
      );
      if (levelSegmentFlags) {
        blockSegmentFlags = createTransientView(
          graph,
          `${props.id}-level-${levelIndex}-block-segment-flags`,
          'uint32',
          blockCount
        );
      }
    }
    const segmentPrefixes = levelSegmentFlags
      ? levelIndex === 0 && props.outputSegmentPrefixes
        ? props.outputSegmentPrefixes
        : blockCount > 1 || levelIndex > 0
          ? createTransientView(
              graph,
              `${props.id}-level-${levelIndex}-segment-prefixes`,
              'uint32',
              levelLength
            )
          : undefined
      : undefined;

    nodes.push(
      ...addBlockScanPass(graph, {
        id: `${props.id}-level-${levelIndex}-scan`,
        input: levelInput,
        output: levelOutput,
        mode: levelIndex === 0 ? props.mode : 'exclusive',
        countNonzeroInput: levelIndex === 0 && props.countNonzeroInput,
        segmentFlags: levelSegmentFlags,
        segmentSummaryInput:
          Boolean(levelSegmentFlags) && (levelIndex > 0 || props.segmentSummaryInput),
        segmentPrefixes,
        blockSums,
        blockSegmentFlags,
        finalSum: blockSums ? undefined : props.finalSum,
        finalSegmentFlag: blockSums ? undefined : props.finalSegmentFlag,
        length: levelLength,
        blockCount,
        dispatchLayout: getGPUScanDispatchLayout(
          levelLength,
          props.maxComputeWorkgroupsPerDimension
        )
      })
    );
    levels.push({output: levelOutput, length: levelLength, segmentPrefixes});

    if (!blockSums) {
      break;
    }
    const blockOffsets = createTransientView(
      graph,
      `${props.id}-level-${levelIndex}-block-offsets`,
      'uint32',
      blockCount
    );
    levels[levels.length - 1].blockOffsets = blockOffsets;
    levelInput = blockSums;
    levelOutput = blockOffsets;
    levelSegmentFlags = blockSegmentFlags;
    levelLength = blockCount;
    levelIndex++;
  }

  for (let index = levels.length - 2; index >= 0; index--) {
    const level = levels[index];
    const parentLevel = levels[index + 1];
    nodes.push(
      ...addOffsetPass(graph, {
        id: `${props.id}-level-${index}-add-offsets`,
        output: level.output,
        offsets: level.blockOffsets!,
        length: level.length,
        segmentPrefixes: level.segmentPrefixes,
        offsetSegmentPrefixes: parentLevel.segmentPrefixes,
        dispatchLayout: getGPUScanDispatchLayout(
          level.length,
          props.maxComputeWorkgroupsPerDimension
        )
      })
    );
  }

  if (graph.recording) {
    for (const node of nodes) {
      const iteration = Number(/^-level-(\d+)-/.exec(node.id.slice(props.id.length))?.[1]);
      if (node.annotation) {
        node.annotation.loop = {
          id: `${props.id}/levels`,
          iteration,
          phase: 'unconditional',
          maximumIterations: levels.length
        };
      }
    }
  }
  return nodes;
}

/** Writes the zero total of an empty scan. */
function addZeroTotalPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  total: GraphDataView<'uint32'>
): readonly GPUCommandNode<Parameters>[] {
  const bindings: GPUKernelBinding[] = [{name: 'totalValues', view: total, access: 'write'}];
  const constants = {TOTAL_OFFSET: getViewElementOffset(total)};
  return [
    createGPUKernelNode(graph, {
      id: `${id}-zero-total`,
      operation: 'GPUScan',
      moduleId: 'GPUScan/zero-total',
      formula: 'gpu-core.scan',
      source: /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(1) fn main() { totalValues[TOTAL_OFFSET] = 0u; }`,
      constants,
      bindings,
      workgroupSize: 1,
      dispatchLayout: {x: 1, y: 1, z: 1},
      invocationCount: 1
    })
  ];
}

/** Returns one packed row within a transient view without allocating another buffer. */
function createPackedSubview<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<'uint32'>,
  index: number
): GraphDataView<'uint32'> {
  return graph.createDataView(view.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: view.byteOffset + index * view.rowByteLength
  });
}

/** Validates every atomic chunk accepted by a scan input. */
function validateScanInput(input: GPUScanInput, name: string): void {
  const chunks = input instanceof GraphVectorView ? input.data : [input];
  for (const chunk of chunks) {
    validatePackedUint32View(chunk, name);
  }
}

/** Normalizes one data view or vector into its ordered atomic chunks. */
function getScanChunks(input: GPUScanInput): readonly GraphDataView<'uint32'>[] {
  return input instanceof GraphVectorView ? input.data : [input];
}

/** Verifies that every scan chunk belongs to the graph receiving the operation. */
function validateScanOwnership<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  input: GPUScanInput,
  id: string
): void {
  const chunks = input instanceof GraphVectorView ? input.data : [input];
  if (chunks.some(chunk => chunk.buffer.graph !== graph)) {
    throw new Error(`${id} views must belong to the target graph`);
  }
}

/** Adds one block-local scan level and optionally writes one summary pair per block. */
function addBlockScanPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    input: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
    mode: 'exclusive' | 'inclusive';
    segmentFlags?: GraphDataView<'uint32'>;
    countNonzeroInput?: boolean;
    /** Preserve carry into the current summary even when that summary contains a later start. */
    segmentSummaryInput?: boolean;
    segmentPrefixes?: GraphDataView<'uint32'>;
    blockSums?: GraphDataView<'uint32'>;
    blockSegmentFlags?: GraphDataView<'uint32'>;
    finalSum?: GraphDataView<'uint32'>;
    finalSegmentFlag?: GraphDataView<'uint32'>;
    length: number;
    blockCount: number;
    dispatchLayout: GPUScanDispatchLayout;
  }
): readonly GPUCommandNode<Parameters>[] {
  const sumOutput = props.blockSums ?? props.finalSum;
  const segmentFlagOutput = props.blockSegmentFlags ?? props.finalSegmentFlag;
  const bindings: GPUKernelBinding[] = [
    {name: 'inputValues', view: props.input, access: 'read'},
    {name: 'outputValues', view: props.output, access: 'write'}
  ];
  if (sumOutput) bindings.push({name: 'sumValues', view: sumOutput, access: 'write'});
  if (props.segmentFlags) {
    bindings.push({name: 'segmentFlags', view: props.segmentFlags, access: 'read'});
  }
  if (props.segmentPrefixes) {
    bindings.push({name: 'segmentPrefixes', view: props.segmentPrefixes, access: 'write'});
  }
  if (segmentFlagOutput) {
    bindings.push({name: 'summarySegmentFlags', view: segmentFlagOutput, access: 'write'});
  }
  // Sizes and offsets are pipeline constants, so every level and input size shares one pipeline.
  const constants: Record<string, number> = {
    ELEMENT_COUNT: props.length,
    BLOCK_COUNT: props.blockCount,
    INPUT_OFFSET: getViewElementOffset(props.input),
    OUTPUT_OFFSET: getViewElementOffset(props.output),
    ...(sumOutput && {SUM_OFFSET: getViewElementOffset(sumOutput)}),
    ...(props.segmentFlags && {SEGMENT_FLAGS_OFFSET: getViewElementOffset(props.segmentFlags)}),
    ...(props.segmentPrefixes && {
      SEGMENT_PREFIXES_OFFSET: getViewElementOffset(props.segmentPrefixes)
    }),
    ...(segmentFlagOutput && {
      SUMMARY_SEGMENT_FLAGS_OFFSET: getViewElementOffset(segmentFlagOutput)
    }),
    ...getBoundedDispatchOverrides(props.dispatchLayout)
  };
  const declarations = `${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}`;
  const sumWrite = props.blockSums
    ? 'sumValues[SUM_OFFSET + workgroupIndex] = scratch[255u];'
    : props.finalSum
      ? 'sumValues[SUM_OFFSET] = scratch[255u];'
      : '';
  const segmentFlagWrite = props.blockSegmentFlags
    ? 'summarySegmentFlags[SUMMARY_SEGMENT_FLAGS_OFFSET + workgroupIndex] = segmentScratch[255u];'
    : props.finalSegmentFlag
      ? 'summarySegmentFlags[SUMMARY_SEGMENT_FLAGS_OFFSET] = segmentScratch[255u];'
      : '';
  const inputRead = props.countNonzeroInput
    ? 'min(inputValues[INPUT_OFFSET + index], 1u)'
    : 'inputValues[INPUT_OFFSET + index]';
  const inputSegmentFlag = props.segmentFlags ? 'segmentFlags[SEGMENT_FLAGS_OFFSET + index]' : '0u';
  const scanUpdate = props.segmentFlags
    ? `if (lane >= stride) {
      scratch[lane] = select(addend + scratch[lane], scratch[lane], segmentScratch[lane] != 0u);
      segmentScratch[lane] = addendSegment | segmentScratch[lane];
    }`
    : `if (lane >= stride) {
      scratch[lane] = scratch[lane] + addend;
    }`;
  const outputValue = props.mode === 'inclusive' ? 'scratch[lane]' : 'scratch[lane] - inputValue';
  const segmentedOutput =
    props.segmentFlags && props.mode === 'exclusive'
      ? `var scannedOutput = 0u;
    if (lane > 0u) {
      scannedOutput = scratch[lane - 1u];
    }
    ${props.segmentSummaryInput ? '' : 'if (inputSegmentFlag != 0u) { scannedOutput = 0u; }'}
    outputValues[OUTPUT_OFFSET + index] = scannedOutput;`
      : `outputValues[OUTPUT_OFFSET + index] = ${outputValue};`;
  const segmentPrefixWrite = props.segmentPrefixes
    ? props.segmentSummaryInput
      ? `var segmentPrefix = 0u;
    if (lane > 0u) {
      segmentPrefix = segmentScratch[lane - 1u];
    }
    segmentPrefixes[SEGMENT_PREFIXES_OFFSET + index] = segmentPrefix;`
      : 'segmentPrefixes[SEGMENT_PREFIXES_OFFSET + index] = segmentScratch[lane];'
    : '';
  const portableSource = /* wgsl */ `
${declarations}
var<workgroup> scratch: array<u32, ${SCAN_WORKGROUP_SIZE}>;
${props.segmentFlags ? `var<workgroup> segmentScratch: array<u32, ${SCAN_WORKGROUP_SIZE}>;` : ''}

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexOverrideSource(SCAN_WORKGROUP_SIZE).body}
  if (workgroupIndex >= BLOCK_COUNT) { return; }
  let lane = localInvocationIndex;
  var inputValue = 0u;
  var inputSegmentFlag = 0u;
  if (index < ELEMENT_COUNT) {
    inputValue = ${inputRead};
    inputSegmentFlag = ${inputSegmentFlag};
  }
  scratch[lane] = inputValue;
  ${props.segmentFlags ? 'segmentScratch[lane] = inputSegmentFlag;' : ''}
  workgroupBarrier();

  for (var stride = 1u; stride < ${SCAN_WORKGROUP_SIZE}u; stride = stride * 2u) {
    var addend = 0u;
    ${props.segmentFlags ? 'var addendSegment = 0u;' : ''}
    if (lane >= stride) {
      addend = scratch[lane - stride];
      ${props.segmentFlags ? 'addendSegment = segmentScratch[lane - stride];' : ''}
    }
    workgroupBarrier();
    ${scanUpdate}
    workgroupBarrier();
  }

  if (lane == ${SCAN_WORKGROUP_SIZE - 1}u) {
    ${sumWrite}
    ${segmentFlagWrite}
  }
  if (index < ELEMENT_COUNT) {
    ${segmentedOutput}
    ${segmentPrefixWrite}
  }
}`;
  const strategy = getGPUScanStrategyForGraph(graph, Boolean(props.segmentFlags), props.length);
  const source =
    strategy === 'subgroups'
      ? getSubgroupBlockScanSource(props, declarations, sumOutput !== undefined)
      : portableSource;

  return [
    createGPUKernelNode(graph, {
      id: props.id,
      operation: 'GPUScan',
      variant: strategy,
      moduleId: `GPUScan/level-scan/${strategy}`,
      dedupName: 'level-scan',
      formula: 'gpu-core.scan',
      source,
      constants,
      bindings,
      workgroupSize: SCAN_WORKGROUP_SIZE,
      dispatchLayout: props.dispatchLayout,
      invocationCount: props.length,
      workload: {
        readByteLength: props.length * Uint32Array.BYTES_PER_ELEMENT,
        writeByteLength:
          props.length * Uint32Array.BYTES_PER_ELEMENT +
          props.blockCount * Uint32Array.BYTES_PER_ELEMENT
      }
    })
  ];
}

/**
 * Scans up to {@link GPU_SCAN_SINGLE_WORKGROUP_LIMIT} unsegmented rows with one workgroup, so a
 * small input (for example a radix histogram) needs one dispatch instead of block scan, summary
 * scan, and offset passes. The workgroup walks 256-row tiles in order, carrying the running sum.
 */
function addSingleWorkgroupScanPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    input: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
    mode: 'exclusive' | 'inclusive';
    countNonzeroInput?: boolean;
    finalSum?: GraphDataView<'uint32'>;
    length: number;
  }
): readonly GPUCommandNode<Parameters>[] {
  const bindings: GPUKernelBinding[] = [
    {name: 'inputValues', view: props.input, access: 'read'},
    {name: 'outputValues', view: props.output, access: 'write'}
  ];
  if (props.finalSum) bindings.push({name: 'sumValues', view: props.finalSum, access: 'write'});
  const constants: Record<string, number> = {
    ELEMENT_COUNT: props.length,
    INPUT_OFFSET: getViewElementOffset(props.input),
    OUTPUT_OFFSET: getViewElementOffset(props.output),
    ...(props.finalSum && {SUM_OFFSET: getViewElementOffset(props.finalSum)})
  };
  const declarations = `${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}`;
  const strategy = getGPUScanStrategyForGraph(graph, false, props.length);
  const inputRead = props.countNonzeroInput
    ? 'min(inputValues[INPUT_OFFSET + index], 1u)'
    : 'inputValues[INPUT_OFFSET + index]';
  const sumWrite = props.finalSum
    ? 'if (tile == tileCount - 1u) { sumValues[SUM_OFFSET] = tilePrefix; }'
    : '';
  const portableSource = /* wgsl */ `
${declarations}
var<workgroup> scratch: array<u32, ${SCAN_WORKGROUP_SIZE}>;
var<workgroup> carry: u32;

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) lane: u32
) {
  let tileCount = (ELEMENT_COUNT + ${SCAN_WORKGROUP_SIZE - 1}u) / ${SCAN_WORKGROUP_SIZE}u;
  if (lane == 0u) { carry = 0u; }
  for (var tile = 0u; tile < tileCount; tile++) {
    let index = tile * ${SCAN_WORKGROUP_SIZE}u + lane;
    var inputValue = 0u;
    if (index < ELEMENT_COUNT) { inputValue = ${inputRead}; }
    scratch[lane] = inputValue;
    workgroupBarrier();
    for (var stride = 1u; stride < ${SCAN_WORKGROUP_SIZE}u; stride = stride * 2u) {
      var addend = 0u;
      if (lane >= stride) { addend = scratch[lane - stride]; }
      workgroupBarrier();
      if (lane >= stride) { scratch[lane] = scratch[lane] + addend; }
      workgroupBarrier();
    }
    let base = carry;
    let tilePrefix = base + scratch[lane];
    if (index < ELEMENT_COUNT) {
      outputValues[OUTPUT_OFFSET + index] = ${props.mode === 'inclusive' ? 'tilePrefix' : 'tilePrefix - inputValue'};
    }
    workgroupBarrier();
    if (lane == ${SCAN_WORKGROUP_SIZE - 1}u) {
      carry = tilePrefix;
      ${sumWrite}
    }
  }
}`;
  const subgroupSource = /* wgsl */ `
enable subgroups;
requires subgroup_id;

${declarations}
var<workgroup> subgroupOffsets: array<u32, ${MAXIMUM_SCAN_SUBGROUP_COUNT}>;
var<workgroup> carry: u32;

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE}) fn main(
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_id) subgroupId: u32
) {
  let lane = subgroupId * subgroupSize + subgroupInvocationId;
  let subgroupCount = ${SCAN_WORKGROUP_SIZE}u / subgroupSize;
  let tileCount = (ELEMENT_COUNT + ${SCAN_WORKGROUP_SIZE - 1}u) / ${SCAN_WORKGROUP_SIZE}u;
  if (lane == 0u) { carry = 0u; }
  for (var tile = 0u; tile < tileCount; tile++) {
    let index = tile * ${SCAN_WORKGROUP_SIZE}u + lane;
    var inputValue = 0u;
    if (index < ELEMENT_COUNT) { inputValue = ${inputRead}; }
    let subgroupPrefix = subgroupInclusiveAdd(inputValue);
    if (subgroupInvocationId == subgroupSize - 1u) {
      subgroupOffsets[subgroupId] = subgroupPrefix;
    }
    workgroupBarrier();
    if (lane == 0u) {
      var runningOffset = carry;
      for (var subgroupIndex = 0u; subgroupIndex < subgroupCount; subgroupIndex++) {
        let subgroupSum = subgroupOffsets[subgroupIndex];
        subgroupOffsets[subgroupIndex] = runningOffset;
        runningOffset = runningOffset + subgroupSum;
      }
      carry = runningOffset;
    }
    workgroupBarrier();
    let tilePrefix = subgroupOffsets[subgroupId] + subgroupPrefix;
    if (index < ELEMENT_COUNT) {
      outputValues[OUTPUT_OFFSET + index] = ${props.mode === 'inclusive' ? 'tilePrefix' : 'tilePrefix - inputValue'};
    }
    ${sumWrite ? `if (lane == ${SCAN_WORKGROUP_SIZE - 1}u) { ${sumWrite} }` : ''}
    workgroupBarrier();
  }
}`;
  return [
    createGPUKernelNode(graph, {
      id: props.id,
      operation: 'GPUScan',
      variant: strategy,
      moduleId: `GPUScan/single-workgroup-scan/${strategy}`,
      dedupName: 'single-workgroup-scan',
      formula: 'gpu-core.scan',
      source: strategy === 'subgroups' ? subgroupSource : portableSource,
      constants,
      bindings,
      workgroupSize: SCAN_WORKGROUP_SIZE,
      dispatchLayout: {x: 1, y: 1, z: 1},
      invocationCount: props.length,
      workload: {
        readByteLength: props.length * Uint32Array.BYTES_PER_ELEMENT,
        writeByteLength: (props.length + 1) * Uint32Array.BYTES_PER_ELEMENT
      }
    })
  ];
}

/** Uses empirical adapter calibration when supplied, otherwise preserves capability selection. */
function getGPUScanStrategyForGraph<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  segmented: boolean,
  workloadSize: number
): GPUScanStrategy {
  const preferredStrategy = getGPUScanStrategy(graph.device, segmented);
  if (!graph.autotuner || segmented) return preferredStrategy;
  const alternateStrategy: GPUScanStrategy =
    preferredStrategy === 'subgroups' ? 'portable' : 'subgroups';
  return graph.autotuner.selectKernel({
    operation: 'GPUScan',
    workloadSize,
    candidates: [
      {id: preferredStrategy},
      {
        id: alternateStrategy,
        supported:
          alternateStrategy === 'portable' ||
          getGPUScanStrategy(graph.device, false) === 'subgroups'
      }
    ]
  }).variant as GPUScanStrategy;
}

/** Builds an ordered scan over virtual lanes derived from subgroup ID and invocation ID. */
function getSubgroupBlockScanSource(
  props: {
    mode: 'exclusive' | 'inclusive';
    countNonzeroInput?: boolean;
    blockSums?: GraphDataView<'uint32'>;
    finalSum?: GraphDataView<'uint32'>;
  },
  declarations: string,
  hasSumOutput: boolean
): string {
  const outputValue = props.mode === 'inclusive' ? 'blockPrefix' : 'blockPrefix - inputValue';
  const sumWrite = !hasSumOutput
    ? ''
    : props.blockSums
      ? 'sumValues[SUM_OFFSET + workgroupIndex] = blockPrefix;'
      : 'sumValues[SUM_OFFSET] = blockPrefix;';

  return /* wgsl */ `
enable subgroups;
requires subgroup_id;

${declarations}
var<workgroup> subgroupOffsets: array<u32, ${MAXIMUM_SCAN_SUBGROUP_COUNT}>;

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE}) fn main(
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_id) subgroupId: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let lane = subgroupId * subgroupSize + subgroupInvocationId;
  ${getBoundedInvocationIndexOverrideSource(SCAN_WORKGROUP_SIZE).body.replace('localInvocationIndex', 'lane')}
  if (workgroupIndex >= BLOCK_COUNT) { return; }
  var inputValue = 0u;
  if (index < ELEMENT_COUNT) {
    inputValue = ${props.countNonzeroInput ? 'min(inputValues[INPUT_OFFSET + index], 1u)' : 'inputValues[INPUT_OFFSET + index]'};
  }

  let subgroupPrefix = subgroupInclusiveAdd(inputValue);
  if (subgroupInvocationId == subgroupSize - 1u) {
    subgroupOffsets[subgroupId] = subgroupPrefix;
  }
  workgroupBarrier();

  let subgroupCount = ${SCAN_WORKGROUP_SIZE}u / subgroupSize;
  if (lane == 0u) {
    var runningOffset = 0u;
    for (var subgroupIndex = 0u; subgroupIndex < subgroupCount; subgroupIndex++) {
      let subgroupSum = subgroupOffsets[subgroupIndex];
      subgroupOffsets[subgroupIndex] = runningOffset;
      runningOffset = runningOffset + subgroupSum;
    }
  }
  workgroupBarrier();

  let blockPrefix = subgroupOffsets[subgroupId] + subgroupPrefix;
  if (lane == ${SCAN_WORKGROUP_SIZE - 1}u) {
    ${sumWrite}
  }
  if (index < ELEMENT_COUNT) {
    outputValues[OUTPUT_OFFSET + index] = ${outputValue};
  }
}`;
}

/** Adds a scanned block offset or one vector carry into an output view. */
function addOffsetPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    output: GraphDataView<'uint32'>;
    offsets: GraphDataView<'uint32'>;
    length: number;
    offsetIndex?: number;
    segmentPrefixes?: GraphDataView<'uint32'>;
    offsetSegmentPrefixes?: GraphDataView<'uint32'>;
    dispatchLayout: GPUScanDispatchLayout;
  }
): readonly GPUCommandNode<Parameters>[] {
  const bindings: GPUKernelBinding[] = [
    {name: 'outputValues', view: props.output, access: 'read_write'},
    {name: 'offsets', view: props.offsets, access: 'read'}
  ];
  if (props.segmentPrefixes) {
    bindings.push({
      name: 'segmentPrefixes',
      view: props.segmentPrefixes,
      access: props.offsetSegmentPrefixes ? 'read_write' : 'read'
    });
  }
  if (props.offsetSegmentPrefixes) {
    bindings.push({
      name: 'offsetSegmentPrefixes',
      view: props.offsetSegmentPrefixes,
      access: 'read'
    });
  }
  const constants: Record<string, number> = {
    ELEMENT_COUNT: props.length,
    OUTPUT_OFFSET: getViewElementOffset(props.output),
    OFFSETS_OFFSET: getViewElementOffset(props.offsets),
    ...(props.offsetIndex !== undefined && {OFFSET_INDEX: props.offsetIndex}),
    ...(props.segmentPrefixes && {
      SEGMENT_PREFIXES_OFFSET: getViewElementOffset(props.segmentPrefixes)
    }),
    ...(props.offsetSegmentPrefixes && {
      OFFSET_SEGMENT_PREFIXES_OFFSET: getViewElementOffset(props.offsetSegmentPrefixes)
    }),
    ...getBoundedDispatchOverrides(props.dispatchLayout)
  };
  const offsetIndex =
    props.offsetIndex === undefined ? `index / ${SCAN_WORKGROUP_SIZE}u` : 'OFFSET_INDEX';
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexOverrideSource(SCAN_WORKGROUP_SIZE).body}
  if (index < ELEMENT_COUNT) {
    ${props.offsetSegmentPrefixes ? `let offsetSegmentPrefix = offsetSegmentPrefixes[OFFSET_SEGMENT_PREFIXES_OFFSET + ${offsetIndex}];` : ''}
    ${
      props.segmentPrefixes
        ? `let segmentPrefix = segmentPrefixes[SEGMENT_PREFIXES_OFFSET + index];
    if (segmentPrefix == 0u) {`
        : ''
    }
      outputValues[OUTPUT_OFFSET + index] = outputValues[OUTPUT_OFFSET + index] + offsets[OFFSETS_OFFSET + ${offsetIndex}];
    ${props.segmentPrefixes ? '}' : ''}
    ${props.segmentPrefixes && props.offsetSegmentPrefixes ? 'segmentPrefixes[SEGMENT_PREFIXES_OFFSET + index] = segmentPrefix | offsetSegmentPrefix;' : ''}
  }
}`;
  return [
    createGPUKernelNode(graph, {
      id: props.id,
      operation: 'GPUScan',
      moduleId: 'GPUScan/add-offsets',
      dedupName: 'add-offsets',
      formula: 'gpu-core.scan',
      source,
      constants,
      bindings,
      workgroupSize: SCAN_WORKGROUP_SIZE,
      dispatchLayout: props.dispatchLayout,
      invocationCount: props.length,
      workload: {
        readByteLength: props.length * Uint32Array.BYTES_PER_ELEMENT * 2,
        writeByteLength: props.length * Uint32Array.BYTES_PER_ELEMENT
      }
    })
  ];
}

type GPUScanDispatchLayout = GPUBoundedDispatchLayout;

/** Plans a bounded three-dimensional dispatch for one scan pass. @internal */
export function getGPUScanDispatchLayout(
  elementCount: number,
  maxComputeWorkgroupsPerDimension: number
): GPUScanDispatchLayout {
  return getBoundedDispatchLayout(
    'GPUScan',
    elementCount,
    SCAN_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
}

/** Returns WGSL that maps a bounded 3D scan dispatch to one linear element index. @internal */
export function getGPUScanInvocationIndexSource(layout: GPUScanDispatchLayout): string {
  return getBoundedInvocationIndexSource(layout, SCAN_WORKGROUP_SIZE);
}

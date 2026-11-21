// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUKeyLookup} from '../../dataframe-analysis/key-join/gpu-key-join';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createRecipeKernelNode} from '../recipes/recipe-utils';

/** Per-target diagnostic codes; these are not row-selection flags. */
export const POPULATION_SUPPORT_ALIGNMENT_CODES = {
  MATCHED: 0,
  MISSING: 1,
  AMBIGUOUS: 2
} as const;

/** Decodes a per-target alignment diagnostic, preserving unknown codes explicitly. */
export function decodePopulationSupportAlignmentCode(
  code: number
): 'matched' | 'missing' | 'ambiguous' | 'unknown' {
  switch (code) {
    case POPULATION_SUPPORT_ALIGNMENT_CODES.MATCHED:
      return 'matched';
    case POPULATION_SUPPORT_ALIGNMENT_CODES.MISSING:
      return 'missing';
    case POPULATION_SUPPORT_ALIGNMENT_CODES.AMBIGUOUS:
      return 'ambiguous';
    default:
      return 'unknown';
  }
}

export type GPUPopulationSupportAlignmentOutput = {
  /** float32x2, target-aligned populations; unmatched or ambiguous rows contain NaN. */
  populations: GraphDataView<'float32x2'>;
  /** 0 matched, 1 no source, 2 duplicate source or target. Missing takes precedence. */
  alignmentCode: GraphDataView<'uint32'>;
  /** 1 for a unique source and target match, otherwise 0; usable as a downstream mask. */
  matchedMask: GraphDataView<'uint32'>;
  /** uint32x4: [complete, missingTargetRows, ambiguousTargetRows, excludedSourceRows]. */
  status: GraphDataView<'uint32x4'>;
};

/** Inputs and all four caller-owned outputs of {@link GPUPopulationSupportAlignment}. */
export type GPUPopulationSupportAlignmentProps = {
  id?: string;
  populations: GraphDataView<'float32x2'>;
  sourceIds: GraphDataView<'uint32'>;
  targetIds: GraphDataView<'uint32'>;
  output: GPUPopulationSupportAlignmentOutput;
};

/** Inputs and optional caller-owned outputs of the population support alignment recipe. */
export type GPUPopulationSupportAlignmentRecipeProps = Omit<
  GPUPopulationSupportAlignmentProps,
  'output'
> & {
  output?: Partial<GPUPopulationSupportAlignmentOutput>;
};

const POPULATION_SUPPORT_ALIGNMENT_OUTPUT_NAMES = [
  'populations',
  'alignmentCode',
  'matchedMask',
  'status'
] as const;

/** Validate supplied views without reserving graph resources; absent recipe outputs are allowed. @internal */
export function validatePopulationSupportAlignmentViews<Parameters>(
  id: string,
  props: GPUPopulationSupportAlignmentRecipeProps,
  graph?: GPUCommandGraph<Parameters>
): void {
  validateRequirement(id, props.populations.length > 0, 'populations must have at least 1 row');
  validateViewLength(id, 'sourceIds', props.sourceIds, props.populations.length);
  validateRequirement(id, props.targetIds.length > 0, 'targetIds must have at least 1 row');
  validateSinglePackedView(props.populations, ['float32x2'], `${id} populations`);
  validateSinglePackedView(props.sourceIds, ['uint32'], `${id} sourceIds`);
  validateSinglePackedView(props.targetIds, ['uint32'], `${id} targetIds`);
  for (const [name, format, length] of [
    ['populations', 'float32x2', props.targetIds.length],
    ['alignmentCode', 'uint32', props.targetIds.length],
    ['matchedMask', 'uint32', props.targetIds.length],
    ['status', 'uint32x4', 1]
  ] as const) {
    const view = props.output?.[name];
    if (!view) continue;
    validateSinglePackedView(view, [format], `${id} output.${name}`);
    validateViewLength(id, `output.${name}`, view, length);
  }
  const entries = POPULATION_SUPPORT_ALIGNMENT_OUTPUT_NAMES.flatMap<[string, GraphDataView]>(
    name => {
      const view = props.output?.[name];
      return view === undefined ? [] : [[name, view]];
    }
  );
  const outputs = entries.map(([, view]) => view);
  validateGraphOutputsDisjointFromInputs(id, outputs, [
    props.populations,
    props.sourceIds,
    props.targetIds
  ]);
  const outputNamesByBuffer = new Map<object, string>();
  for (const [name, view] of entries) {
    const buffer = view.buffer.defaultBuffer
      ? getCoreDefaultBuffer(view.buffer.defaultBuffer)
      : view.buffer;
    const previousName = outputNamesByBuffer.get(buffer);
    validateRequirement(
      id,
      previousName === undefined,
      `output.${previousName} and output.${name} must not share a buffer`
    );
    outputNamesByBuffer.set(buffer, name);
  }
  validateRequirement(
    id,
    [props.sourceIds, props.targetIds, ...outputs].every(
      view => view.buffer.graph === props.populations.buffer.graph
    ),
    'views must belong to one graph'
  );
  if (graph)
    validateGraphViewsBelongToGraph(id, graph, [
      props.populations,
      props.sourceIds,
      props.targetIds,
      ...outputs
    ]);
}

/**
 * Exact support-ID alignment using two sort/index joins and parallel status reductions.
 * Duplicate sources/targets invalidate the target row; sources absent from the target support
 * are counted separately. Population values are copied without validating their magnitudes.
 * Every uint32 ID, including 0xffffffff, is a valid identifier in this legacy contract.
 *
 * Per-frame: every input buffer's contents. Compile-time: view lengths and output bindings.
 * Inputs are single packed views. No coordinates or numeric parameters are read.
 */
export class GPUPopulationSupportAlignment implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPopulationSupportAlignmentProps;

  constructor(props: GPUPopulationSupportAlignmentProps) {
    validateRequirement(
      props.id ?? 'population-support-alignment',
      !('outputs' in props),
      'outputs was removed; use output'
    );
    const construction = snapshotConstructionFields(props, [
      'id',
      'populations',
      'sourceIds',
      'targetIds',
      'output'
    ]);
    const output =
      construction.output &&
      snapshotConstructionFields(construction.output, POPULATION_SUPPORT_ALIGNMENT_OUTPUT_NAMES);
    props = Object.freeze({...construction, output});
    this.id = props.id ?? 'population-support-alignment';
    this.props = props;
    const {id} = this;
    for (const name of POPULATION_SUPPORT_ALIGNMENT_OUTPUT_NAMES) {
      validateRequirement(id, Boolean(props.output?.[name]), `output.${name} is required`);
    }
    validatePopulationSupportAlignmentViews(id, props);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const sourceCount = props.sourceIds.length;
    const targetCount = props.targetIds.length;
    validateGraphViewsBelongToGraph(id, graph, [
      props.populations,
      props.sourceIds,
      props.targetIds,
      ...POPULATION_SUPPORT_ALIGNMENT_OUTPUT_NAMES.map(name => props.output[name])
    ]);
    // The key join uses uint32x2 sort keys and scratch pairs: eight bytes per row.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'source key scratch',
      sourceCount * 8,
      'reduce sourceIds rows'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'target key scratch',
      targetCount * 8,
      'reduce targetIds rows'
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const sourceKeys = createTransientView(graph, `${id}-source-keys`, 'uint32x2', sourceCount);
    const targetKeys = createTransientView(graph, `${id}-target-keys`, 'uint32x2', targetCount);
    for (const [name, input, output] of [
      ['sources', props.sourceIds, sourceKeys],
      ['targets', props.targetIds, targetKeys]
    ] as const) {
      nodes.push(
        createRecipeKernelNode(graph, {
          id: `${id}-${name}-keys`,
          operation: 'GPUPopulationSupportAlignment',
          invocationCount: input.length,
          bindings: [
            {name: 'input', view: input, type: 'u32', access: 'read'},
            {name: 'output', view: output, type: 'u32', access: 'read_write'}
          ],
          // A zero high word keeps even the legacy 0xffffffff ID distinct from a missing key.
          body: `output[outputOffset + 2u * index] = input[inputOffset + index];
output[outputOffset + 2u * index + 1u] = 0u;`
        })
      );
    }
    const sourceRows = createTransientView(graph, `${id}-source-rows`, 'uint32', targetCount);
    const sourceMatches = createTransientView(graph, `${id}-source-matches`, 'uint32', targetCount);
    const targetMatches = createTransientView(graph, `${id}-target-matches`, 'uint32', targetCount);
    const sourceMatched = createTransientView(graph, `${id}-source-matched`, 'uint32', sourceCount);
    nodes.push(
      ...new GPUKeyLookup({
        id: `${id}-sources`,
        leftKeys: targetKeys,
        rightKeys: sourceKeys,
        output: {rightRows: sourceRows, matchCounts: sourceMatches, rightMatched: sourceMatched}
      }).getCommandNodes(graph)
    );
    nodes.push(
      ...new GPUKeyLookup({
        id: `${id}-targets`,
        leftKeys: targetKeys,
        rightKeys: targetKeys,
        output: {matchCounts: targetMatches}
      }).getCommandNodes(graph)
    );
    const missingFlags = createTransientView(graph, `${id}-missing-flags`, 'uint32', targetCount);
    const duplicateFlags = createTransientView(
      graph,
      `${id}-duplicate-flags`,
      'uint32',
      targetCount
    );
    nodes.push(
      createRecipeKernelNode(graph, {
        id: `${id}-align`,
        operation: 'GPUPopulationSupportAlignment',
        invocationCount: targetCount,
        bindings: [
          {name: 'input', view: props.populations, type: 'f32', access: 'read'},
          {name: 'sourceRows', view: sourceRows, type: 'u32', access: 'read'},
          {name: 'sourceMatches', view: sourceMatches, type: 'u32', access: 'read'},
          {name: 'targetMatches', view: targetMatches, type: 'u32', access: 'read'},
          {name: 'output', view: props.output.populations, type: 'f32', access: 'read_write'},
          {
            name: 'alignmentCode',
            view: props.output.alignmentCode,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'matchedMask', view: props.output.matchedMask, type: 'u32', access: 'read_write'}
        ],
        body: `let matches = sourceMatches[sourceMatchesOffset + index];
let valid = matches == 1u && targetMatches[targetMatchesOffset + index] == 1u;
let code = select(select(${POPULATION_SUPPORT_ALIGNMENT_CODES.AMBIGUOUS}u, ${POPULATION_SUPPORT_ALIGNMENT_CODES.MISSING}u, matches == 0u), ${POPULATION_SUPPORT_ALIGNMENT_CODES.MATCHED}u, valid);
let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
var population = vec2<f32>(missing);
if (valid) {
  let source = sourceRows[sourceRowsOffset + index];
  population = vec2<f32>(input[inputOffset + 2u * source], input[inputOffset + 2u * source + 1u]);
}
output[outputOffset + 2u * index] = population.x;
output[outputOffset + 2u * index + 1u] = population.y;
alignmentCode[alignmentCodeOffset + index] = code;
matchedMask[matchedMaskOffset + index] = u32(valid);`
      }),
      createRecipeKernelNode(graph, {
        id: `${id}-alignment-flags`,
        operation: 'GPUPopulationSupportAlignment',
        invocationCount: targetCount,
        bindings: [
          {name: 'alignmentCode', view: props.output.alignmentCode, type: 'u32', access: 'read'},
          {name: 'missingFlags', view: missingFlags, type: 'u32', access: 'read_write'},
          {name: 'duplicateFlags', view: duplicateFlags, type: 'u32', access: 'read_write'}
        ],
        body: `let code = alignmentCode[alignmentCodeOffset + index];
missingFlags[missingFlagsOffset + index] = u32(code == ${POPULATION_SUPPORT_ALIGNMENT_CODES.MISSING}u);
duplicateFlags[duplicateFlagsOffset + index] = u32(code == ${POPULATION_SUPPORT_ALIGNMENT_CODES.AMBIGUOUS}u);`
      })
    );
    const excludedFlags = createTransientView(graph, `${id}-excluded-flags`, 'uint32', sourceCount);
    nodes.push(
      createRecipeKernelNode(graph, {
        id: `${id}-excluded`,
        operation: 'GPUPopulationSupportAlignment',
        invocationCount: sourceCount,
        bindings: [
          {name: 'matched', view: sourceMatched, type: 'u32', access: 'read'},
          {name: 'excluded', view: excludedFlags, type: 'u32', access: 'read_write'}
        ],
        body: 'excluded[excludedOffset + index] = u32(matched[matchedOffset + index] == 0u);'
      })
    );
    const totals = [missingFlags, duplicateFlags, excludedFlags].map((input, index) => {
      const output = createTransientView(graph, `${id}-total-${index}`, 'uint32', 1);
      nodes.push(
        ...new GPUReduction({
          id: `${id}-count-${index}`,
          input,
          output,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
      return output;
    });
    nodes.push(
      createRecipeKernelNode(graph, {
        id: `${id}-certificate`,
        operation: 'GPUPopulationSupportAlignment',
        invocationCount: 1,
        bindings: [
          {name: 'missingTotal', view: totals[0], type: 'u32', access: 'read'},
          {name: 'duplicateTotal', view: totals[1], type: 'u32', access: 'read'},
          {name: 'excludedTotal', view: totals[2], type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `let missing = missingTotal[missingTotalOffset];
let duplicates = duplicateTotal[duplicateTotalOffset];
let excluded = excludedTotal[excludedTotalOffset];
status[statusOffset] = u32(missing == 0u && duplicates == 0u && excluded == 0u);
status[statusOffset + 1u] = missing;
status[statusOffset + 2u] = duplicates;
status[statusOffset + 3u] = excluded;`
      })
    );
    return nodes;
  }
}

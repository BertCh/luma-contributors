// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {alignKeplerGraphVectorViews} from './luma-gaps/graph-vector-chunks';
import type {ResidentSuitabilityColumnFormat} from './resident-suitability-plan';

const WORKGROUP_SIZE = 256;
const FINITE_BITS_WGSL = /* wgsl */ `fn isFiniteBits(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}`;

type KernelBinding = {
  name: string;
  view: GraphDataView;
  type: 'f32' | 'i32' | 'u32' | 'atomic<u32>';
  access: 'read' | 'read_write';
};

/** Elementwise kernel over aligned spans; every binding is addressed with `index`. */
function createKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    invocationCount: number;
    bindings: readonly KernelBinding[];
    declarations?: string;
    body: string;
  }
): GPUCommandNode<Parameters> {
  const layout = getBoundedDispatchLayout(
    props.operation,
    Math.max(props.invocationCount, 1),
    WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const declarations = props.bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  // A phony use of each binding keeps it in the automatic pipeline layout.
  const references = props.bindings.map(binding => `_ = &${binding.name};`).join('\n  ');
  const source = /* wgsl */ `const LENGTH: u32 = ${props.invocationCount}u;
${declarations}
${props.declarations ?? ''}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${references}
  ${getBoundedInvocationIndexSource(layout, WORKGROUP_SIZE)}
  if (index >= LENGTH) { return; }
  ${props.body}
}`;
  let readByteLength = 0;
  let writeByteLength = 0;
  for (const binding of props.bindings) {
    const byteLength = binding.view.length * binding.view.rowByteLength;
    if (binding.access === 'read') readByteLength += byteLength;
    else writeByteLength += byteLength;
  }
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    workload: {
      operation: props.operation,
      commandCount: 1,
      maximumWorkgroupCount: layout.x * layout.y * layout.z,
      maximumInvocationCount: layout.x * layout.y * layout.z * WORKGROUP_SIZE,
      readByteLength,
      writeByteLength
    },
    resources: props.bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
    })),
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: props.bindings.map((binding, location) => ({
            name: binding.name,
            type: binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {};
          for (const binding of props.bindings)
            bindings[binding.name] = getViewBinding(binding.view, getBuffer);
          kernel.dispatch(computePass, {bindings, x: layout.x, y: layout.y, z: layout.z});
        },
        destroy: () => kernel.destroy()
      };
    }
  });
}

const SHADER_TYPES = {float32: 'f32', sint32: 'i32', uint32: 'u32'} as const;

/**
 * Converts one resident column, across all of its chunks, to Float32 values plus a defined-row
 * mask. A row is defined when its validity is set and (for Float32) the value is finite, matching
 * native `toNumber`. Undefined rows store 0.
 */
export function createSuitabilityPrepareNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    format: ResidentSuitabilityColumnFormat;
    source: GraphVectorView<ResidentSuitabilityColumnFormat>;
    validity?: GraphVectorView<'uint32'>;
    clean: GraphDataView<'float32'>;
    defined: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const type = SHADER_TYPES[props.format];
  const spans = props.validity
    ? alignKeplerGraphVectorViews(graph, [props.source, props.validity, props.clean, props.defined])
    : alignKeplerGraphVectorViews(graph, [props.source, props.clean, props.defined]);
  return spans.map((span, spanIndex) => {
    const [source, validity, clean, defined] =
      span.length === 4
        ? (span as unknown as GraphDataView[])
        : [span[0] as GraphDataView, undefined, span[1] as GraphDataView, span[2] as GraphDataView];
    return createKernelNode(graph, {
      id: `${props.id}-prepare-${spanIndex}`,
      operation: 'ResidentSuitabilityPrepare',
      invocationCount: clean.length,
      declarations: FINITE_BITS_WGSL,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        ...(validity
          ? [{name: 'validity', view: validity, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'clean', view: clean, type: 'f32', access: 'read_write'},
        {name: 'defined', view: defined, type: 'u32', access: 'read_write'}
      ],
      body: `let value = f32(source[sourceOffset + index]);
  let included = isFiniteBits(value)${validity ? ' && validity[validityOffset + index] != 0u' : ''};
  clean[cleanOffset + index] = select(0.0, value, included);
  defined[definedOffset + index] = select(0u, 1u, included);`
    });
  });
}

/**
 * Reclassifies one categorical column (dictionary codes across all of its chunks) to Float32
 * scores through a lookup table with one entry per dictionary code plus a trailing null entry.
 * NaN entries, null rows and codes beyond the table are undefined rows, which store 0.
 */
export function createSuitabilityReclassifyNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    source: GraphVectorView<'uint32'>;
    validity?: GraphVectorView<'uint32'>;
    lookup: GraphDataView<'float32'>;
    clean: GraphDataView<'float32'>;
    defined: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const spans = props.validity
    ? alignKeplerGraphVectorViews(graph, [props.source, props.validity, props.clean, props.defined])
    : alignKeplerGraphVectorViews(graph, [props.source, props.clean, props.defined]);
  return spans.map((span, spanIndex) => {
    const [source, validity, clean, defined] =
      span.length === 4
        ? (span as unknown as GraphDataView[])
        : [span[0] as GraphDataView, undefined, span[1] as GraphDataView, span[2] as GraphDataView];
    return createKernelNode(graph, {
      id: `${props.id}-reclassify-${spanIndex}`,
      operation: 'ResidentSuitabilityReclassify',
      invocationCount: clean.length,
      declarations: FINITE_BITS_WGSL,
      bindings: [
        {name: 'source', view: source, type: 'u32', access: 'read'},
        ...(validity
          ? [{name: 'validity', view: validity, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'lookup', view: props.lookup, type: 'f32', access: 'read'},
        {name: 'clean', view: clean, type: 'f32', access: 'read_write'},
        {name: 'defined', view: defined, type: 'u32', access: 'read_write'}
      ],
      body: `let slot = min(source[sourceOffset + index], ${props.lookup.length - 1}u);
  let value = lookup[lookupOffset + slot];
  let included = isFiniteBits(value)${validity ? ' && validity[validityOffset + index] != 0u' : ''};
  clean[cleanOffset + index] = select(0.0, value, included);
  defined[definedOffset + index] = select(0u, 1u, included);`
    });
  });
}

/** Writes a float as a WGSL f32 literal with enough digits to round-trip a Float32. */
function getFloatLiteral(value: number): string {
  return `${Math.fround(value).toExponential(9)}f`;
}

/**
 * Adds `weight * standardized` to the running score and counts defined columns per row. Undefined
 * rows contribute nothing; the final pass decides validity from the defined count. `first`
 * initializes the accumulators instead of reading them.
 */
export function createSuitabilityAccumulateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    first: boolean;
    weight: number;
    standardized: GraphDataView<'float32'>;
    defined: GraphDataView<'uint32'>;
    score: GraphDataView<'float32'>;
    definedCount: GraphDataView<'uint32'>;
    /** Running sum of the weights of the defined columns, kept only to renormalize partial rows. */
    weightSum?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createKernelNode(graph, {
    id: props.id,
    operation: 'ResidentSuitabilityAccumulate',
    invocationCount: props.score.length,
    bindings: [
      {name: 'standardized', view: props.standardized, type: 'f32', access: 'read'},
      {name: 'defined', view: props.defined, type: 'u32', access: 'read'},
      {name: 'score', view: props.score, type: 'f32', access: 'read_write'},
      {name: 'definedCount', view: props.definedCount, type: 'u32', access: 'read_write'},
      ...(props.weightSum
        ? [
            {
              name: 'weightSum',
              view: props.weightSum,
              type: 'f32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ],
    body: `let included = defined[definedOffset + index] != 0u;
  let term = select(0.0, ${getFloatLiteral(props.weight)} * standardized[standardizedOffset + index], included);
  score[scoreOffset + index] = ${props.first ? '' : 'score[scoreOffset + index] + '}term;
  definedCount[definedCountOffset + index] = ${props.first ? '' : 'definedCount[definedCountOffset + index] + '}defined[definedOffset + index];${
    props.weightSum
      ? `
  weightSum[weightSumOffset + index] = ${props.first ? '' : 'weightSum[weightSumOffset + index] + '}select(0.0, ${getFloatLiteral(props.weight)}, included);`
      : ''
  }`
  });
}

/**
 * Writes the score column and its validity into chunk-preserving outputs. A row is scored only
 * when every weighted column is defined and the sum is finite; other rows store 0. Rows defined
 * in some but not all weighted columns are counted into `mixedMissingFlag`, which the host reads
 * back. With `renormalize`, those rows are scored over their defined columns: the sum is scaled
 * by `totalWeight / weightSum` so the defined weights add up to the full weight total.
 */
export function createSuitabilityFinishNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    indicatorCount: number;
    score: GraphDataView<'float32'>;
    definedCount: GraphDataView<'uint32'>;
    scoreValues: GraphVectorView<'float32'>;
    scoreValidity: GraphVectorView<'uint32'>;
    mixedMissingFlag: GraphDataView<'uint32'>;
    /** Partial-row policy; `refuse` and `exclude` both null the score. Default `refuse`. */
    missingValues?: 'refuse' | 'exclude' | 'renormalize';
    /** Sum of all effective weights; required with `renormalize`. */
    totalWeight?: number;
    weightSum?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const renormalize = props.missingValues === 'renormalize' && props.weightSum !== undefined;
  return alignKeplerGraphVectorViews(graph, [
    ...(renormalize ? [props.weightSum!] : []),
    props.score,
    props.definedCount,
    props.scoreValues,
    props.scoreValidity
  ]).map((span, spanIndex) => {
    const [weightSum, score, definedCount, scoreValues, scoreValidity] = (
      renormalize ? span : [undefined, ...span]
    ) as [
      GraphDataView<'float32'> | undefined,
      GraphDataView<'float32'>,
      GraphDataView<'uint32'>,
      GraphDataView<'float32'>,
      GraphDataView<'uint32'>
    ];
    return createKernelNode(graph, {
      id: `${props.id}-finish-${spanIndex}`,
      operation: 'ResidentSuitabilityFinish',
      invocationCount: score.length,
      declarations: FINITE_BITS_WGSL,
      bindings: [
        {name: 'score', view: score, type: 'f32', access: 'read'},
        {name: 'definedCount', view: definedCount, type: 'u32', access: 'read'},
        {name: 'scoreValues', view: scoreValues, type: 'f32', access: 'read_write'},
        {name: 'scoreValidity', view: scoreValidity, type: 'u32', access: 'read_write'},
        {
          name: 'mixedMissingFlag',
          view: props.mixedMissingFlag,
          type: 'atomic<u32>',
          access: 'read_write'
        },
        ...(weightSum
          ? [{name: 'weightSum', view: weightSum, type: 'f32' as const, access: 'read' as const}]
          : [])
      ],
      body: `let count = definedCount[definedCountOffset + index];
  ${
    weightSum
      ? `let definedWeight = weightSum[weightSumOffset + index];
  let value = select(0.0, score[scoreOffset + index] * ${getFloatLiteral(props.totalWeight ?? 1)} / definedWeight, definedWeight > 0.0);
  let scored = count > 0u && definedWeight > 0.0 && isFiniteBits(value);`
      : `let value = score[scoreOffset + index];
  let scored = count == ${props.indicatorCount}u && isFiniteBits(value);`
  }
  scoreValues[scoreValuesOffset + index] = select(0.0, value, scored);
  scoreValidity[scoreValidityOffset + index] = select(0u, 1u, scored);
  if (count > 0u && count < ${props.indicatorCount}u) { atomicAdd(&mixedMissingFlag[mixedMissingFlagOffset], 1u); }`
    });
  });
}

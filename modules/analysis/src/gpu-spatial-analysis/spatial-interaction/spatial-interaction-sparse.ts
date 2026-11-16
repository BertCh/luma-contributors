// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelProps,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createSegmentSumNode,
  getSegmentSumDispatchShape,
  getSortKeyBits,
  SEGMENT_PARTIAL_SUMS_WGSL,
  SEGMENT_WORKGROUP_SIZE,
  getSegmentTreeSumWGSL
} from '../../analysis-core/kernels/sorted-segment-sums';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {getSortedSegmentOffsetNodes} from '../../analysis-core/kernels/sorted-segment-offsets';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {INTERACTION_BALANCING_WGSL} from './interaction-balancing-wgsl';
import type {GPUSpatialInteractionProps} from './gpu-spatial-interaction';

/** Stable pair-list IPF. Sort support once; reuse its permutation and offsets every round. @internal */
export function createSparseInteractionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUSpatialInteractionProps
): GPUCommandNode<Parameters>[] {
  const pairs = props.flows.length;
  const groups = Math.max(props.originCount, props.destinationCount);
  const nodes: GPUCommandNode<Parameters>[] = [];
  const transient = (name: string, length: number, format: 'uint32' | 'float32' = 'float32') =>
    createTransientView(graph, `${id}-${name}`, format, length);
  const binding = (
    name: string,
    view: GraphDataView,
    access: 'read' | 'read_write' = 'read',
    type: 'f32' | 'u32' = 'f32'
  ): WGSLKernelBinding => ({name, view, access, type});
  const unsigned = (name: string, view: GraphDataView, access: 'read' | 'read_write' = 'read') =>
    binding(name, view, access, 'u32');
  const control = transient('control', 4, 'uint32');
  // control: candidate index, initialization code, completed iterations, publish-best flag.
  const support = transient('support', pairs, 'uint32');
  const invalid = transient('invalid', pairs, 'uint32');
  const invalidTotal = transient('invalid-total', 1, 'uint32');
  const contributions = transient('contributions', pairs);
  const observed = transient('observed', 1);
  const predictions = transient('working-predictions', pairs);
  const exponents = transient('exponents', pairs);
  const invalidExponents = transient('invalid-exponents', pairs, 'uint32');
  const invalidExponentTotal = transient('invalid-exponent-total', 1, 'uint32');
  const minimumExponent = transient('minimum-exponent', 1);
  const maximumExponent = transient('maximum-exponent', 1);
  const sortedContributions = transient('sorted-contributions', pairs);
  const errors = transient('errors', groups);
  const error = transient('error', 1);
  const deviance = transient('deviance', 1);
  const errorOffsets = transient('error-offsets', 2, 'uint32');
  const pairOffsets = transient('pair-offsets', 2, 'uint32');
  const coefficientSource = `const COEFFICIENTS = array<f32, ${props.coefficients.length}>(${props.coefficients.map(getWGSLFloatLiteral).join(', ')});`;
  const stateSource = props.pairState ? 'pairState[pairStateOffset + index]' : '1u';
  const pairStateBinding = props.pairState ? [unsigned('pairState', props.pairState)] : [];
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-balance-dispatch`,
    operation: 'GPUSpatialInteraction',
    shapes: [
      [pairs],
      getSegmentSumDispatchShape(props.originCount),
      getSegmentSumDispatchShape(props.destinationCount),
      [groups],
      getSegmentSumDispatchShape(1)
    ]
  });
  const add = (options: WGSLKernelProps<Parameters>) => {
    nodes.push(createWGSLKernelNode(graph, options));
  };
  add({
    id: `${id}-support`,
    operation: 'GPUSpatialInteraction',
    variant: 'sparse-support',
    invocationCount: pairs,
    bindings: [
      unsigned('origins', props.origins),
      unsigned('destinations', props.destinations),
      binding('flows', props.flows),
      ...pairStateBinding,
      unsigned('support', support, 'read_write'),
      unsigned('invalid', invalid, 'read_write'),
      binding('contributions', contributions, 'read_write')
    ],
    declarations: IS_FINITE_BITS_WGSL,
    body: `let state = ${stateSource}; let flow = flows[flowsOffset + index];
let bad = state != 2u && (state > 2u || origins[originsOffset + index] >= ${props.originCount}u ||
  destinations[destinationsOffset + index] >= ${props.destinationCount}u || !isFiniteBits(flow) || flow < 0.0 || (state == 0u && flow != 0.0));
let included = state == 1u && !bad;
support[supportOffset + index] = select(0u, 1u, included);
invalid[invalidOffset + index] = select(0u, 1u, bad);
contributions[contributionsOffset + index] = select(0.0, flow, included);`
  });
  nodes.push(
    ...new GPUReduction({
      id: `${id}-invalid-reduce`,
      input: invalid,
      output: invalidTotal,
      operation: 'max'
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...new GPUReduction({
      id: `${id}-observed-reduce`,
      input: contributions,
      output: observed,
      operation: 'sum'
    }).getCommandNodes(graph)
  );

  const marginals = (
    [
      ['origin', props.originCount, props.origins, 1],
      ['destination', props.destinationCount, props.destinations, 2]
    ] as const
  ).map(([name, count, ids, gateIndex]) => {
    const keys = transient(`${name}-keys`, pairs, 'uint32');
    const indices = transient(`${name}-indices`, pairs, 'uint32');
    const sortedKeys = transient(`${name}-sorted-keys`, pairs, 'uint32');
    const sortedIndices = transient(`${name}-sorted-indices`, pairs, 'uint32');
    const offsets = transient(`${name}-offsets`, count + 1, 'uint32');
    const totals = transient(`${name}-totals`, count);
    const sums = transient(`${name}-sums`, count);
    add({
      id: `${id}-${name}-keys`,
      operation: 'GPUSpatialInteraction',
      variant: 'sparse-group-keys',
      invocationCount: pairs,
      bindings: [
        unsigned('ids', ids),
        unsigned('support', support),
        unsigned('keys', keys, 'read_write'),
        unsigned('indices', indices, 'read_write')
      ],
      body: `keys[keysOffset + index] = select(${count}u, ids[idsOffset + index], support[supportOffset + index] != 0u);
indices[indicesOffset + index] = index;`
    });
    nodes.push(
      ...new GPUSort({
        id: `${id}-${name}-sort`,
        keys: keys as GraphDataView<'uint32'>,
        values: indices as GraphDataView<'uint32'>,
        outputKeys: sortedKeys as GraphDataView<'uint32'>,
        outputValues: sortedIndices as GraphDataView<'uint32'>,
        keyBits: getSortKeyBits(count),
        algorithm: 'radix'
      }).getCommandNodes(graph)
    );
    nodes.push(
      ...getSortedSegmentOffsetNodes(graph, {
        id: `${id}-${name}`,
        operation: 'GPUSpatialInteraction',
        groupCount: count,
        sortedGroupKeys: sortedKeys as GraphDataView<'uint32'>,
        segmentOffsets: offsets as GraphDataView<'uint32'>
      })
    );
    return {name, count, ids, sortedIndices, offsets, totals, sums, gateIndex};
  });
  const reduceMarginal = (
    marginal: (typeof marginals)[number],
    source: GraphDataView,
    output: GraphDataView,
    suffix: string,
    gated: boolean
  ) => {
    add({
      id: `${id}-${suffix}-${marginal.name}-gather`,
      operation: 'GPUSpatialInteraction',
      variant: 'sparse-gather',
      invocationCount: pairs,
      ...(gated ? gate.getGate(0) : {}),
      bindings: [
        binding('source', source),
        unsigned('indices', marginal.sortedIndices),
        binding('sorted', sortedContributions, 'read_write')
      ],
      body: 'sorted[sortedOffset + index] = source[sourceOffset + indices[indicesOffset + index]];'
    });
    nodes.push(
      createInteractionSumNode(graph, {
        id: `${id}-${suffix}-${marginal.name}-reduce`,
        operation: 'GPUSpatialInteraction',
        segmentCount: marginal.count,
        input: sortedContributions as GraphDataView<'float32'>,
        segmentOffsets: marginal.offsets as GraphDataView<'uint32'>,
        output: output as GraphDataView<'float32'>,
        ...(gated ? gate.getGate(marginal.gateIndex) : {})
      })
    );
  };
  for (const marginal of marginals)
    reduceMarginal(marginal, contributions, marginal.totals, 'observed', false);
  add({
    id: `${id}-initialize-output`,
    operation: 'GPUSpatialInteraction',
    variant: 'sparse-initialize-output',
    invocationCount: pairs,
    bindings: [
      binding('output', props.output.predictions, 'read_write'),
      binding('summary', props.output.summary, 'read_write'),
      unsigned('status', props.output.status, 'read_write'),
      unsigned('control', control, 'read_write'),
      unsigned('pairOffsets', pairOffsets, 'read_write'),
      unsigned('errorOffsets', errorOffsets, 'read_write')
    ],
    declarations: GET_NAN_WGSL,
    body: `output[outputOffset + index] = getNaN();
if (index == 0u) {
  for (var slot = 0u; slot < 4u; slot++) { summary[summaryOffset + slot] = getNaN(); control[controlOffset + slot] = 0u; }
  status[statusOffset] = 3u;
  pairOffsets[pairOffsetsOffset] = 0u; pairOffsets[pairOffsetsOffset + 1u] = ${pairs}u;
  errorOffsets[errorOffsetsOffset] = 0u; errorOffsets[errorOffsetsOffset + 1u] = ${groups}u;
}`
  });
  for (let candidate = 0; candidate < props.coefficients.length; candidate++) {
    const prefix = `candidate-${candidate}`;
    add({
      id: `${id}-${prefix}-exponents`,
      operation: 'GPUSpatialInteraction',
      variant: 'sparse-exponents',
      invocationCount: pairs,
      bindings: [
        binding('covariate', props.covariate),
        unsigned('support', support),
        unsigned('control', control),
        binding('exponents', exponents, 'read_write'),
        unsigned('invalid', invalidExponents, 'read_write')
      ],
      declarations: `${coefficientSource}\n${IS_FINITE_BITS_WGSL}`,
      body: `var exponent = 0.0; var bad = false;
if (support[supportOffset + index] != 0u) { exponent = COEFFICIENTS[control[controlOffset]] * covariate[covariateOffset + index]; bad = !isFiniteBits(exponent); }
exponents[exponentsOffset + index] = select(exponent, 0.0, bad);
invalid[invalidOffset + index] = select(0u, 1u, bad);`
    });
    for (const [name, input, output, operation, mask] of [
      ['invalid-exponent', invalidExponents, invalidExponentTotal, 'max', undefined],
      ['minimum-exponent', exponents, minimumExponent, 'min', support],
      ['maximum-exponent', exponents, maximumExponent, 'max', support]
    ] as const)
      nodes.push(
        ...new GPUReduction({
          id: `${id}-${prefix}-${name}`,
          input,
          output,
          operation,
          mask: mask as GraphDataView<'uint32'> | undefined
        }).getCommandNodes(graph)
      );
    add({
      id: `${id}-${prefix}-initialize`,
      operation: 'GPUSpatialInteraction',
      variant: 'sparse-initialize-candidate',
      invocationCount: 1,
      bindings: [
        unsigned('invalid', invalidTotal),
        unsigned('invalidExponent', invalidExponentTotal),
        binding('observed', observed),
        binding('minimumExponent', minimumExponent),
        binding('maximumExponent', maximumExponent),
        unsigned('control', control, 'read_write'),
        binding('error', error, 'read_write'),
        gate.binding
      ],
      declarations: `${IS_FINITE_BITS_WGSL}\n${GET_NAN_WGSL}`,
      body: `var code = 0u;
if (invalid[invalidOffset] != 0u || invalidExponent[invalidExponentOffset] != 0u || !isFiniteBits(observed[observedOffset])) { code = 3u; }
else if (!(observed[observedOffset] > 0.0)) { code = 2u; }
else if (maximumExponent[maximumExponentOffset] - minimumExponent[minimumExponentOffset] > 80.0) { code = 4u; }
control[controlOffset + 1u] = code; control[controlOffset + 2u] = 0u; control[controlOffset + 3u] = 0u;
error[errorOffset] = select(getNaN(), 0.0, code == 0u);
${gate.armWGSL}
if (code != 0u) { ${gate.disarmWGSL} }`
    });
    add({
      id: `${id}-${prefix}-predictions`,
      operation: 'GPUSpatialInteraction',
      variant: 'sparse-initialize-predictions',
      invocationCount: pairs,
      bindings: [
        binding('exponents', exponents),
        binding('maximumExponent', maximumExponent),
        unsigned('control', control),
        ...pairStateBinding,
        binding('predictions', predictions, 'read_write')
      ],
      declarations: GET_NAN_WGSL,
      body: `var prediction = getNaN(); let state = ${stateSource};
if (control[controlOffset + 1u] == 0u) {
  if (state == 0u) { prediction = 0.0; }
  if (state == 1u) { prediction = exp(exponents[exponentsOffset + index] - maximumExponent[maximumExponentOffset]); }
}
predictions[predictionsOffset + index] = prediction;`
    });
    for (let iteration = 0; iteration < (props.maximumIterations ?? 256); iteration++) {
      const round = `${prefix}-round-${iteration}`;
      for (const marginal of marginals) {
        reduceMarginal(marginal, predictions, marginal.sums, `${round}-balance`, true);
        add({
          id: `${id}-${round}-${marginal.name}-scale`,
          operation: 'GPUSpatialInteraction',
          variant: 'sparse-scale',
          declarations: INTERACTION_BALANCING_WGSL,
          invocationCount: pairs,
          ...gate.getGate(0),
          bindings: [
            unsigned('ids', marginal.ids),
            unsigned('support', support),
            binding('totals', marginal.totals),
            binding('sums', marginal.sums),
            binding('predictions', predictions, 'read_write')
          ],
          body: `if (support[supportOffset + index] != 0u) {
  let group = ids[idsOffset + index]; let total = totals[totalsOffset + group]; let sum = sums[sumsOffset + group];
  predictions[predictionsOffset + index] = getBalancedPrediction(predictions[predictionsOffset + index], total, sum);
}`
        });
      }
      for (const marginal of marginals)
        reduceMarginal(marginal, predictions, marginal.sums, `${round}-fitted`, true);
      add({
        id: `${id}-${round}-errors`,
        operation: 'GPUSpatialInteraction',
        variant: 'sparse-marginal-errors',
        invocationCount: groups,
        ...gate.getGate(3),
        bindings: [
          binding('originTotals', marginals[0].totals),
          binding('originSums', marginals[0].sums),
          binding('destinationTotals', marginals[1].totals),
          binding('destinationSums', marginals[1].sums),
          binding('errors', errors, 'read_write')
        ],
        declarations: IS_FINITE_BITS_WGSL,
        body: `var error = 0.0;
if (index < ${props.originCount}u) {
  let total = originTotals[originTotalsOffset + index]; let sum = originSums[originSumsOffset + index];
  error = select(abs(sum - total) / max(1.0, total), bitcast<f32>(0x7f800000u | (index & 0u)), !isFiniteBits(sum));
}
if (index < ${props.destinationCount}u) {
  let total = destinationTotals[destinationTotalsOffset + index]; let sum = destinationSums[destinationSumsOffset + index];
  error = max(error, select(abs(sum - total) / max(1.0, total), bitcast<f32>(0x7f800000u | (index & 0u)), !isFiniteBits(sum)));
}
errors[errorsOffset + index] = error;`
      });
      nodes.push(
        createSegmentSumNode(graph, {
          id: `${id}-${round}-error-reduce`,
          operation: 'GPUSpatialInteraction',
          segmentCount: 1,
          input: errors as GraphDataView<'float32'>,
          segmentOffsets: errorOffsets as GraphDataView<'uint32'>,
          output: error as GraphDataView<'float32'>,
          reduction: 'max',
          ...gate.getGate(4)
        })
      );
      add({
        id: `${id}-${round}-check`,
        operation: 'GPUSpatialInteraction',
        variant: 'sparse-convergence',
        invocationCount: 1,
        bindings: [
          binding('error', error),
          unsigned('control', control, 'read_write'),
          gate.binding
        ],
        body: `if (control[controlOffset + 1u] != 0u) { return; }
control[controlOffset + 2u]++;
if (error[errorOffset] <= ${getWGSLFloatLiteral(props.tolerance ?? 1e-5)}) {
  control[controlOffset + 1u] = 5u; ${gate.disarmWGSL}
}`
      });
    }
    add({
      id: `${id}-${prefix}-deviance`,
      operation: 'GPUSpatialInteraction',
      variant: 'sparse-deviance',
      invocationCount: pairs,
      bindings: [
        unsigned('support', support),
        binding('flows', props.flows),
        binding('predictions', predictions),
        binding('contributions', contributions, 'read_write')
      ],
      body: `var term = 0.0;
if (support[supportOffset + index] != 0u) {
  let flow = flows[flowsOffset + index]; let prediction = predictions[predictionsOffset + index];
  term = prediction;
  if (flow > 0.0) { term = flow * (log(flow) - log(prediction)) - flow + prediction; }
}
contributions[contributionsOffset + index] = 2.0 * term;`
    });
    nodes.push(
      createInteractionSumNode(graph, {
        id: `${id}-${prefix}-deviance-reduce`,
        operation: 'GPUSpatialInteraction',
        segmentCount: 1,
        input: contributions as GraphDataView<'float32'>,
        segmentOffsets: pairOffsets as GraphDataView<'uint32'>,
        output: deviance as GraphDataView<'float32'>
      })
    );
    add({
      id: `${id}-${prefix}-select`,
      operation: 'GPUSpatialInteraction',
      variant: 'sparse-select',
      invocationCount: 1,
      bindings: [
        unsigned('control', control, 'read_write'),
        binding('deviance', deviance),
        binding('error', error),
        binding('diagnostics', props.output.candidates, 'read_write'),
        binding('summary', props.output.summary, 'read_write'),
        unsigned('status', props.output.status, 'read_write')
      ],
      declarations: `${coefficientSource}\n${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
      body: `let candidate = control[controlOffset]; let initialization = control[controlOffset + 1u];
var code = initialization;
if (initialization == 0u || initialization == 5u) { code = select(1u, 0u, initialization == 5u && isFiniteBits(deviance[devianceOffset])); }
let fitted = initialization == 0u || initialization == 5u;
let value = max(0.0, deviance[devianceOffset]);
diagnostics[diagnosticsOffset + 4u * candidate] = select(getNaN(), value, fitted);
diagnostics[diagnosticsOffset + 4u * candidate + 1u] = error[errorOffset];
diagnostics[diagnosticsOffset + 4u * candidate + 2u] = f32(control[controlOffset + 2u]);
diagnostics[diagnosticsOffset + 4u * candidate + 3u] = f32(code);
if (status[statusOffset] != 0u) {
  if (code == 3u) { status[statusOffset] = 2u; }
  else if (code == 2u && status[statusOffset] != 2u) { status[statusOffset] = 1u; }
}
if (code == 0u && (status[statusOffset] != 0u || value < summary[summaryOffset + 1u])) {
  control[controlOffset + 3u] = 1u; status[statusOffset] = 0u;
  summary[summaryOffset] = COEFFICIENTS[candidate]; summary[summaryOffset + 1u] = value;
  summary[summaryOffset + 2u] = error[errorOffset]; summary[summaryOffset + 3u] = f32(control[controlOffset + 2u]);
}`
    });
    add({
      id: `${id}-${prefix}-publish`,
      operation: 'GPUSpatialInteraction',
      variant: 'sparse-publish',
      invocationCount: pairs,
      bindings: [
        unsigned('control', control),
        binding('predictions', predictions),
        binding('output', props.output.predictions, 'read_write')
      ],
      body: 'if (control[controlOffset + 3u] != 0u) { output[outputOffset + index] = predictions[predictionsOffset + index]; }'
    });
    add({
      id: `${id}-${prefix}-advance`,
      operation: 'GPUSpatialInteraction',
      variant: 'sparse-advance',
      invocationCount: 1,
      bindings: [unsigned('control', control, 'read_write')],
      body: 'control[controlOffset]++;'
    });
  }
  if (props.output.originResiduals || props.output.destinationResiduals) {
    add({
      id: `${id}-residual-contributions`,
      operation: 'GPUSpatialInteraction',
      variant: 'sparse-residual-contributions',
      invocationCount: pairs,
      bindings: [
        unsigned('support', support),
        binding('flows', props.flows),
        binding('predictions', props.output.predictions),
        binding('contributions', contributions, 'read_write')
      ],
      body: 'contributions[contributionsOffset + index] = select(0.0, flows[flowsOffset + index] - predictions[predictionsOffset + index], support[supportOffset + index] != 0u);'
    });
    for (const [index, output] of [
      props.output.originResiduals,
      props.output.destinationResiduals
    ].entries()) {
      if (!output) continue;
      reduceMarginal(marginals[index], contributions, output, 'residual', false);
      add({
        id: `${id}-${marginals[index].name}-residual-status`,
        operation: 'GPUSpatialInteraction',
        variant: 'sparse-residual-status',
        invocationCount: output.length,
        bindings: [
          unsigned('status', props.output.status),
          binding('residuals', output, 'read_write')
        ],
        declarations: GET_NAN_WGSL,
        body: 'if (status[statusOffset] != 0u) { residuals[residualsOffset + index] = getNaN(); }'
      });
    }
  }
  return nodes;
}

/** Maximum whole workgroups whose linear invocation count fits uint32. */
const INTERACTION_MAXIMUM_SUM_WORKGROUPS = Math.floor(0xffffffff / SEGMENT_WORKGROUP_SIZE);

/**
 * The shared segment tree with compiler-resistant compensated lane totals. Large sparse OD
 * marginals can put tens of millions of small contributions in one segment: naive strided
 * lane sums drift beyond the prediction tolerance. Reuse the shared tree and dispatch shape;
 * only lane accumulation needs the analysis-domain correction.
 */
function createInteractionSumNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    segmentCount: number;
    input: GraphDataView<'float32'>;
    segmentOffsets: GraphDataView<'uint32'>;
    output: GraphDataView<'float32'>;
  } & Pick<WGSLKernelProps<Parameters>, 'condition' | 'extraResources'>
): GPUCommandNode<Parameters> {
  const [invocationCount, workgroupSize] = getSegmentSumDispatchShape(props.segmentCount);
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'sparse-compensated-sum',
    invocationCount,
    workgroupSize,
    guardIndex: false,
    condition: props.condition,
    extraResources: props.extraResources,
    bindings: [
      {name: 'contributions', view: props.input, type: 'f32', access: 'read'},
      {name: 'offsets', view: props.segmentOffsets, type: 'u32', access: 'read'},
      {name: 'totals', view: props.output, type: 'f32', access: 'read_write'}
    ],
    declarations: `${SEGMENT_PARTIAL_SUMS_WGSL}\n${ADDITION_ROUNDING_ERROR_WGSL}`,
    body: `let firstSegment = index / ${SEGMENT_WORKGROUP_SIZE}u;
let segmentStride = INVOCATION_COUNT / ${SEGMENT_WORKGROUP_SIZE}u;
for (var segmentPass = 0u; segmentPass < ${Math.ceil(props.segmentCount / INTERACTION_MAXIMUM_SUM_WORKGROUPS)}u; segmentPass++) {
  let segment = firstSegment + segmentPass * segmentStride;
  let included = index < INVOCATION_COUNT && segment < ${props.segmentCount}u;
  var partialSum = 0.0; var compensation = 0.0;
  if (included) {
    let begin = offsets[offsetsOffset + segment];
    let end = min(offsets[offsetsOffset + segment + 1u], ${props.input.length}u);
    for (var row = begin + localInvocationIndex; row < end; row += ${SEGMENT_WORKGROUP_SIZE}u) {
      let contribution = contributions[contributionsOffset + row];
      let next = partialSum + contribution;
      compensation += getAdditionRoundingError(partialSum, contribution, next);
      partialSum = next;
    }
  }
  partialSum += compensation;
  ${getSegmentTreeSumWGSL()}
  if (included && localInvocationIndex == 0u) { totals[totalsOffset + segment] = partialSums[0]; }
  workgroupBarrier();
}`
  });
}

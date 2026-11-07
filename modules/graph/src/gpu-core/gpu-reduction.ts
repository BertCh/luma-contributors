// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {type Device} from '@luma.gl/core';
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
  type GPUKernelBinding,
  type GPUKernelBindingList
} from './gpu-kernel-node';
import {
  createTransientView,
  getViewElementOffset,
  type GPUScalarFormat,
  validatePackedUint32View,
  validatePackedView
} from './graph-data-view-utils';
import {alignGraphVectorViews} from './graph-vector-view-utils';
import {getGPUShaderSubgroupStrategy} from './gpu-subgroup-utils';
import {DOUBLE_SINGLE_WGSL} from './gpu-double-single';

const REDUCTION_WORKGROUP_SIZE = 256;
/** Rows each thread accumulates serially before the tree stage, for large inputs. */
const REDUCTION_ROWS_PER_THREAD = 4;
/**
 * Inputs up to this many rows keep one row per thread (at most two levels), which preserves the
 * established node structure. Larger inputs would need three or more levels, so each thread
 * accumulates several strided rows to shrink the hierarchy.
 */
const REDUCTION_SINGLE_ROW_LENGTH = REDUCTION_WORKGROUP_SIZE * REDUCTION_WORKGROUP_SIZE;

/** Returns how many strided rows each thread accumulates when reducing `inputLength` rows. */
function getReductionRowsPerThread(inputLength: number): number {
  return inputLength > REDUCTION_SINGLE_ROW_LENGTH ? REDUCTION_ROWS_PER_THREAD : 1;
}
const SCALAR_FORMATS = ['uint32', 'sint32', 'float32'] as const;

/** Compute strategy selected for one reduction level. @internal */
export type GPUReductionStrategy = 'portable' | 'subgroups';

/** Selects subgroup collectives only when the device and WGSL language both expose them. */
export function getGPUReductionStrategy(device: Device): GPUReductionStrategy {
  return getGPUShaderSubgroupStrategy(device, {requiresSubgroupId: true});
}

/**
 * Operation performed by {@link GPUReduction}.
 *
 * `sum`, `min`, and `max` produce one row. `extent` produces two rows containing the minimum and
 * maximum respectively.
 */
export type GPUReductionOperation = 'sum' | 'min' | 'max' | 'extent';

/**
 * One packed scalar chunk or an ordered, fixed-width scalar vector reduced by
 * {@link GPUReduction}.
 */
export type GPUReductionInput<T extends GPUScalarFormat = GPUScalarFormat> =
  | GraphDataView<T>
  | GraphVectorView<T>;

/** Optional nonzero/zero row selection with the same logical length as reduction input. */
export type GPUReductionMask = GraphDataView<'uint32'> | GraphVectorView<'uint32'>;

/** Properties for a graph-native scalar reduction. */
export type GPUReductionProps<T extends GPUScalarFormat = GPUScalarFormat> = {
  /** Prefix for generated graph node and transient resource IDs. */
  id?: string;
  /** Packed scalar data view or ordered vector of packed scalar chunks. */
  input: GPUReductionInput<T>;
  /** Optional nonzero/zero selection with the same logical length as `input`; chunk boundaries may differ. */
  mask?: GPUReductionMask;
  /** Caller-owned result view: one row, or two rows for `extent`. */
  output: GraphDataView<T>;
  /** Aggregate to compute. */
  operation: GPUReductionOperation;
  /**
   * Compensated (double-single) float32 summation. Per-thread Neumaier accumulation and an
   * error-free pairwise tree keep the rounding error nearly independent of row count and order.
   * Valid only for float32 `sum`; always uses the shared-memory tree.
   */
  compensated?: boolean;
};

/**
 * Hierarchical reduction over packed 32-bit scalar graph data views and vectors.
 *
 * Each 256-thread workgroup reduces at most 256 input rows to one partial row. Additional levels
 * repeat that process until one row remains. Multi-chunk vectors first produce one partial per
 * non-empty chunk, then reduce those partials to one global result.
 *
 * @remarks Inputs and outputs are caller-owned. Hierarchical scratch is graph-owned. Adding the
 * reduction to a graph only declares nodes and resources; it does not compile, encode, submit,
 * map, or read data back.
 */
export class GPUReduction<T extends GPUScalarFormat = GPUScalarFormat> {
  /** Prefix for generated graph node and transient resource IDs. */
  readonly id: string;
  /** Scalar input chunk or vector. */
  readonly input: GPUReductionInput<T>;
  /** Optional source-aligned row selection. */
  readonly mask?: GPUReductionMask;
  /** Caller-owned result view. */
  readonly output: GraphDataView<T>;
  /** Aggregate computed by this reduction. */
  readonly operation: GPUReductionOperation;
  /** Whether float32 sums use compensated (double-single) accumulation. */
  readonly compensated: boolean;

  /**
   * Creates and validates a graph-native reduction description.
   *
   * @throws If views are not packed 32-bit scalar data, formats differ, the output has the wrong
   * length, or input and output use the same physical graph buffer.
   */
  constructor(props: GPUReductionProps<T>) {
    this.id = props.id ?? 'gpu-reduction';
    this.input = props.input;
    this.mask = props.mask;
    this.output = props.output;
    this.operation = props.operation;
    this.compensated = props.compensated ?? false;
    if (this.compensated && (props.operation !== 'sum' || props.input.format !== 'float32')) {
      throw new Error(`${this.id} compensated reduction requires a float32 sum`);
    }
    for (const [chunkIndex, input] of getReductionInputs(this.input).entries()) {
      const name = this.input instanceof GraphVectorView ? ` input chunk ${chunkIndex}` : ' input';
      validatePackedView(input, SCALAR_FORMATS, `${this.id}${name}`);
      if (input.format !== this.input.format) {
        throw new Error(`${this.id}${name} format must match the input format`);
      }
    }
    validatePackedView(this.output, SCALAR_FORMATS, `${this.id} output`);
    if (this.input.format !== this.output.format) {
      throw new Error(`${this.id} input and output formats must match`);
    }
    if (!['sum', 'min', 'max', 'extent'].includes(this.operation)) {
      throw new Error(`${this.id} operation must be sum, min, max, or extent`);
    }
    const outputLength = this.operation === 'extent' ? 2 : 1;
    if (this.output.length !== outputLength) {
      throw new Error(`${this.id} ${this.operation} output must contain ${outputLength} row(s)`);
    }
    if (getReductionInputs(this.input).some(input => input.buffer === this.output.buffer)) {
      throw new Error(`${this.id} inputs and output must use separate buffers`);
    }
    if (this.mask) {
      for (const [chunkIndex, mask] of getReductionMasks(this.mask).entries()) {
        validatePackedUint32View(mask, `${this.id} mask chunk ${chunkIndex}`);
        if (mask.buffer === this.output.buffer) {
          throw new Error(`${this.id} mask and output must use separate buffers`);
        }
      }
      if (this.input.length !== this.mask.length) {
        throw new Error(`${this.id} input and mask lengths must match`);
      }
    }
  }

  /**
   * Adds hierarchical reduction passes and graph-owned scratch to a command graph.
   *
   * Empty inputs add a one-thread clear pass. A single non-empty chunk is reduced directly.
   * Multiple non-empty chunks are reduced independently, then their partial rows are merged. A
   * final pass converts an invalid floating-point min/max/extent result to zero and writes the
   * caller-owned output.
   *
   * @param graph Mutable graph that owns all input/output handles and generated scratch.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    let inputs = getReductionInputs(this.input);
    let masks = this.mask ? getReductionMasks(this.mask) : undefined;
    if (
      inputs.some(input => input.buffer.graph !== graph) ||
      masks?.some(mask => mask.buffer.graph !== graph) ||
      this.output.buffer.graph !== graph
    ) {
      throw new Error(`${this.id} views must belong to the target graph`);
    }
    if (this.mask) {
      const spans = alignGraphVectorViews(graph, [this.input, this.mask]);
      inputs = spans.map(([input]) => input);
      masks = spans.map(([, mask]) => mask);
    }
    const nonEmptyInputs = inputs
      .map((input, chunkIndex) => ({input, mask: masks?.[chunkIndex]}))
      .filter(({input}) => input.length > 0);
    if (nonEmptyInputs.length === 0) {
      nodes.push(...addClearReductionPass(graph, this.id, this.output));
      return nodes;
    }

    const valuesPerRow = this.operation === 'extent' || this.compensated ? 2 : 1;
    const needsValidity =
      this.operation !== 'sum' && (this.input.format === 'float32' || Boolean(this.mask));
    let reductionResult: ReductionResult<T> & {nodes: readonly GPUCommandNode<Parameters>[]};

    if (nonEmptyInputs.length === 1) {
      reductionResult = addReductionLevels(graph, {
        id: this.id,
        format: this.input.format,
        operation: this.operation,
        inputValues: nonEmptyInputs[0].input,
        inputValidity: nonEmptyInputs[0].mask,
        inputLength: nonEmptyInputs[0].input.length,
        valuesPerRow,
        firstLevel: true,
        needsValidity,
        compensated: this.compensated
      });
      nodes.push(...reductionResult.nodes);
    } else {
      const partialValues = createTransientView(
        graph,
        `${this.id}-chunk-values`,
        this.input.format,
        nonEmptyInputs.length * valuesPerRow
      ) as GraphDataView<T>;
      const partialValidity = needsValidity
        ? createTransientView(graph, `${this.id}-chunk-validity`, 'uint32', nonEmptyInputs.length)
        : undefined;

      nonEmptyInputs.forEach(({input, mask}, partialIndex) => {
        nodes.push(
          ...addReductionLevels(graph, {
            id: `${this.id}-chunk-${partialIndex}`,
            format: this.input.format,
            operation: this.operation,
            inputValues: input,
            inputValidity: mask,
            inputLength: input.length,
            valuesPerRow,
            firstLevel: true,
            needsValidity,
            compensated: this.compensated,
            finalValues: createPackedSubview(
              graph,
              partialValues,
              partialIndex * valuesPerRow,
              valuesPerRow
            ),
            finalValidity: partialValidity
              ? createPackedSubview(graph, partialValidity, partialIndex, 1)
              : undefined
          }).nodes
        );
      });

      reductionResult = addReductionLevels(graph, {
        id: `${this.id}-merge`,
        format: this.input.format,
        operation: this.operation,
        inputValues: partialValues,
        inputValidity: partialValidity,
        inputLength: nonEmptyInputs.length,
        valuesPerRow,
        firstLevel: false,
        needsValidity,
        compensated: this.compensated
      });
      nodes.push(...reductionResult.nodes);
    }

    nodes.push(
      ...addFinalizeReductionPass(graph, {
        id: `${this.id}-finalize`,
        format: this.input.format,
        inputValues: reductionResult.values,
        inputValidity: reductionResult.validity,
        output: this.output,
        valuesPerRow,
        compensated: this.compensated
      })
    );

    return nodes;
  }
}

type ReductionResult<T extends GPUScalarFormat> = {
  /** One packed row for sum/min/max, or two packed components for extent. */
  values: GraphDataView<T>;
  /** One row indicating whether an eligible min/max/extent value was observed. */
  validity?: GraphDataView<'uint32'>;
};

type ReductionLevelsProps<T extends GPUScalarFormat> = {
  /** Prefix for generated level node and resource IDs. */
  id: string;
  /** Scalar format of all value views. */
  format: T;
  /** Aggregate computed at every level. */
  operation: GPUReductionOperation;
  /** Packed values consumed by the first generated level. */
  inputValues: GraphDataView<T>;
  /** Optional source selection or validity rows paired with already-reduced input rows. */
  inputValidity?: GraphDataView<'uint32'>;
  /** Number of logical rows consumed by the first generated level. */
  inputLength: number;
  /** Components in each partial row: two for extent, otherwise one. */
  valuesPerRow: number;
  /** Whether the first level reads raw scalar rows instead of partial rows. */
  firstLevel: boolean;
  /** Whether levels must preserve finite-value or source-selection validity. */
  needsValidity: boolean;
  /** Whether float32 sums carry a double-single (high, low) pair per row. */
  compensated: boolean;
  /** Optional destination for the last level's value row. */
  finalValues?: GraphDataView<T>;
  /** Optional destination for the last level's validity row. */
  finalValidity?: GraphDataView<'uint32'>;
};

/**
 * Adds a complete hierarchy that reduces `inputLength` rows to one partial row.
 *
 * Level `n` consumes the views produced by level `n - 1`. Each level emits
 * `ceil(currentLength / (256 * rowsPerThread))` rows, so the hierarchy always converges. Intermediate levels allocate
 * graph-owned views. When supplied, `finalValues` and `finalValidity` replace only the last
 * allocation; multi-chunk reduction uses that facility to write each chunk's partial directly into
 * its slot in the shared merge input.
 */
function addReductionLevels<Parameters, T extends GPUScalarFormat>(
  graph: GPUCommandGraph<Parameters>,
  props: ReductionLevelsProps<T>
): ReductionResult<T> & {nodes: readonly GPUCommandNode<Parameters>[]} {
  const nodes: GPUCommandNode<Parameters>[] = [];
  let currentValues = props.inputValues;
  let currentValidity = props.inputValidity;
  let currentLength = props.inputLength;
  let levelIndex = 0;

  // One workgroup produces one next-level row. The last level is therefore the first level whose
  // output contains exactly one row.
  while (currentLength > 1 || levelIndex === 0) {
    const nextLength = Math.ceil(
      currentLength / (REDUCTION_WORKGROUP_SIZE * getReductionRowsPerThread(currentLength))
    );
    const isLastLevel = nextLength === 1;
    const nextValues =
      isLastLevel && props.finalValues
        ? props.finalValues
        : (createTransientView(
            graph,
            `${props.id}-level-${levelIndex}-values`,
            props.format,
            nextLength * props.valuesPerRow
          ) as GraphDataView<T>);
    const nextValidity = props.needsValidity
      ? isLastLevel && props.finalValidity
        ? props.finalValidity
        : createTransientView(
            graph,
            `${props.id}-level-${levelIndex}-validity`,
            'uint32',
            nextLength
          )
      : undefined;
    nodes.push(
      ...addReductionLevelPass(graph, {
        id: `${props.id}-level-${levelIndex}`,
        format: props.format,
        operation: props.operation,
        inputValues: currentValues,
        inputValidity: currentValidity,
        outputValues: nextValues,
        outputValidity: nextValidity,
        inputLength: currentLength,
        valuesPerRow: props.valuesPerRow,
        firstLevel: props.firstLevel && levelIndex === 0,
        compensated: props.compensated
      })
    );
    currentValues = nextValues;
    currentValidity = nextValidity;
    currentLength = nextLength;
    levelIndex++;
  }

  if (graph.recording) {
    // Each level adds exactly one node, so node order is level order.
    nodes.forEach((node, iteration) => {
      if (node.annotation) {
        node.annotation.loop = {
          id: `${props.id}/levels`,
          iteration,
          phase: 'unconditional',
          maximumIterations: nodes.length
        };
      }
    });
  }

  return {values: currentValues, validity: currentValidity, nodes};
}

/** Creates a packed logical slice without allocating or changing hazard granularity. */
function createPackedSubview<T extends GPUScalarFormat, Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<T>,
  elementOffset: number,
  length: number
): GraphDataView<T> {
  return graph.createDataView(view.buffer, {
    format: view.format,
    length,
    byteOffset: view.byteOffset + elementOffset * view.rowByteLength
  });
}

/** Normalizes a single data view or vector view into its ordered chunk list. */
function getReductionInputs<T extends GPUScalarFormat>(
  input: GPUReductionInput<T>
): readonly GraphDataView<T>[] {
  return input instanceof GraphVectorView ? input.data : [input];
}

/** Normalizes a source-aligned selection into its ordered chunk list. */
function getReductionMasks(mask: GPUReductionMask): readonly GraphDataView<'uint32'>[] {
  return mask instanceof GraphVectorView ? mask.data : [mask];
}

/** Adds one 256-way reduction level from raw or already-reduced rows. */
function addReductionLevelPass<Parameters, T extends GPUScalarFormat>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    format: T;
    operation: GPUReductionOperation;
    inputValues: GraphDataView<T>;
    inputValidity?: GraphDataView<'uint32'>;
    outputValues: GraphDataView<T>;
    outputValidity?: GraphDataView<'uint32'>;
    inputLength: number;
    valuesPerRow: number;
    firstLevel: boolean;
    compensated: boolean;
  }
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const rowsPerThread = getReductionRowsPerThread(props.inputLength);
  const rowsPerWorkgroup = REDUCTION_WORKGROUP_SIZE * rowsPerThread;
  const dispatchLayout = getBoundedDispatchLayout(
    'GPUReduction',
    props.inputLength,
    rowsPerWorkgroup,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const shaderType = getShaderType(props.format);
  const zero = getZeroLiteral(props.format);
  const isSum = props.operation === 'sum';
  const isExtent = props.operation === 'extent';
  const hasSecondValue = isExtent || props.compensated;
  const finiteExpression =
    props.format === 'float32' ? 'value == value && abs(value) <= 3.402823466e+38' : 'true';
  // Reads one logical row into the named locals; the names let the same text fill the first
  // accumulator directly (one row per thread) or a per-row temporary (several rows per thread).
  const getReadRow = (names: {value: string; second: string; valid: string; index: string}) => {
    // `x == x` NaN tests are folded away by Tint and Metal, so renamed row temporaries rely on the
    // magnitude bound alone, which is false for NaN and infinities.
    const finite =
      names.value === 'value' || props.format !== 'float32'
        ? finiteExpression
        : `abs(${names.value}) <= 3.402823466e+38`;
    const normalizeMaskedFloatSum =
      props.firstLevel && props.inputValidity && props.format === 'float32' && isSum
        ? `\n    if (${names.valid} == 0u) { ${names.value} = 0.0; ${names.second} = 0.0; }`
        : '';
    const readRawValue = `${names.value} = inputValues[INPUT_OFFSET + ${names.index}];
    ${names.second} = ${props.compensated ? '0.0' : names.value};
    ${names.valid} = select(0u, 1u, ${finite});${normalizeMaskedFloatSum}`;
    return props.firstLevel
      ? props.inputValidity
        ? `if (inputValidity[VALIDITY_OFFSET + ${names.index}] != 0u) {
      ${readRawValue}
    }`
        : readRawValue
      : `${names.value} = inputValues[INPUT_OFFSET + ${names.index} * VALUES_PER_ROW];
    ${names.second} = inputValues[INPUT_OFFSET + ${names.index} * VALUES_PER_ROW + ${hasSecondValue ? '1u' : '0u'}];
    ${props.inputValidity ? `${names.valid} = inputValidity[VALIDITY_OFFSET + ${names.index}];` : `${names.valid} = 1u;`}`;
  };
  const readFirst = getReadRow({
    value: 'value',
    second: 'secondValue',
    valid: 'valid',
    index: 'index'
  });
  const combine = props.compensated
    ? `let combined = addDoubleSingle(
        vec2<f32>(firstScratch[lane], secondScratch[lane]),
        vec2<f32>(firstScratch[lane + stride], secondScratch[lane + stride])
      );
      firstScratch[lane] = combined.x;
      secondScratch[lane] = combined.y;`
    : isSum
      ? 'firstScratch[lane] = firstScratch[lane] + firstScratch[lane + stride];'
      : `let rightValid = validityScratch[lane + stride];
      if (rightValid != 0u) {
        if (validityScratch[lane] == 0u) {
          firstScratch[lane] = firstScratch[lane + stride];
          secondScratch[lane] = secondScratch[lane + stride];
        } else {
          firstScratch[lane] = ${props.operation === 'max' ? 'max' : 'min'}(firstScratch[lane], firstScratch[lane + stride]);
          secondScratch[lane] = ${isExtent ? 'max' : props.operation === 'max' ? 'max' : 'min'}(secondScratch[lane], secondScratch[lane + stride]);
        }
        validityScratch[lane] = 1u;
      }`;
  // Compensated sums keep a low part per lane, which subgroup collectives cannot carry.
  const strategy = props.compensated ? 'portable' : getGPUReductionStrategy(graph.device);
  const reductionSource =
    strategy === 'subgroups'
      ? getSubgroupReductionSource({
          format: props.format,
          operation: props.operation,
          combine
        })
      : getPortableReductionSource(combine);
  // Each thread folds `rowsPerThread` rows spaced one workgroup apart (coalesced) before the tree.
  const accumulate = props.compensated
    ? `let total = twoSum(value, rowValue);
      value = total.x;
      secondValue = secondValue + rowSecond + total.y;`
    : isSum
      ? 'value = value + rowValue;'
      : `if (rowValid != 0u) {
        if (valid == 0u) {
          value = rowValue;
          secondValue = rowSecond;
          valid = 1u;
        } else {
          value = ${props.operation === 'max' ? 'max' : 'min'}(value, rowValue);
          secondValue = ${isExtent || props.operation === 'max' ? 'max' : 'min'}(secondValue, rowSecond);
        }
      }`;
  const loadRows =
    rowsPerThread === 1
      ? `if (index < ELEMENT_COUNT) {
    ${readFirst}
  }`
      : `for (var rowStep = 0u; rowStep < ${rowsPerThread}u; rowStep++) {
    let rowIndex = index + rowStep * ${REDUCTION_WORKGROUP_SIZE}u;
    if (rowIndex < ELEMENT_COUNT) {
      var rowValue = ${zero};
      var rowSecond = ${zero};
      var rowValid = 0u;
      ${getReadRow({value: 'rowValue', second: 'rowSecond', valid: 'rowValid', index: 'rowIndex'})}
      ${accumulate}
    }
  }`;
  // Problem sizes and view offsets are pipeline constants, so one shader serves every input length
  // within a rows-per-thread regime, strategy, and operation.
  const bindings: GPUKernelBindingList = [
    {name: 'inputValues', view: props.inputValues, access: 'read'},
    props.inputValidity && {name: 'inputValidity', view: props.inputValidity, access: 'read'},
    {name: 'outputValues', view: props.outputValues, access: 'write'},
    props.outputValidity && {name: 'outputValidity', view: props.outputValidity, access: 'write'}
  ];
  const constants: Record<string, number> = {
    ELEMENT_COUNT: props.inputLength,
    VALUES_PER_ROW: props.valuesPerRow,
    WORKGROUP_COUNT: Math.ceil(props.inputLength / rowsPerWorkgroup),
    INPUT_OFFSET: getViewElementOffset(props.inputValues),
    OUTPUT_OFFSET: getViewElementOffset(props.outputValues),
    ...(props.inputValidity && {VALIDITY_OFFSET: getViewElementOffset(props.inputValidity)}),
    ...(props.outputValidity && {
      OUTPUT_VALIDITY_OFFSET: getViewElementOffset(props.outputValidity)
    }),
    ...getBoundedDispatchOverrides(dispatchLayout)
  };
  const invocationIndex = getBoundedInvocationIndexOverrideSource(rowsPerWorkgroup);
  const source = /* wgsl */ `
${strategy === 'subgroups' ? 'enable subgroups;\nrequires subgroup_id;' : ''}
${props.compensated ? DOUBLE_SINGLE_WGSL : ''}
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
var<workgroup> firstScratch: array<${shaderType}, ${REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> secondScratch: array<${shaderType}, ${REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> validityScratch: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;

@compute @workgroup_size(${REDUCTION_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  ${strategy === 'subgroups' ? '@builtin(subgroup_invocation_id) subgroupInvocationId: u32,\n  @builtin(subgroup_size) subgroupSize: u32,\n  @builtin(subgroup_id) subgroupId: u32,' : ''}
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${invocationIndex.body}
  if (workgroupIndex >= WORKGROUP_COUNT) {
    return;
  }
  ${props.compensated ? 'opaqueZero = workgroupId.x >> 31u;' : ''}
  let lane = localInvocationIndex;
  var value = ${zero};
  var secondValue = ${zero};
  var valid = 0u;
  ${loadRows}
  ${reductionSource}
  if (lane == 0u) {
    outputValues[OUTPUT_OFFSET + workgroupIndex * VALUES_PER_ROW] = firstScratch[0];
    ${hasSecondValue ? 'outputValues[OUTPUT_OFFSET + workgroupIndex * VALUES_PER_ROW + 1u] = secondScratch[0];' : ''}
    ${props.outputValidity ? 'outputValidity[OUTPUT_VALIDITY_OFFSET + workgroupIndex] = validityScratch[0];' : ''}
  }
}`;

  nodes.push(
    ...addKernelPass(graph, {
      id: props.id,
      source,
      bindings,
      constants,
      dispatchLayout,
      record: {
        moduleId: `GPUReduction/level/${strategy}`,
        workgroupSize: REDUCTION_WORKGROUP_SIZE,
        invocationCount: props.inputLength,
        dedupName: 'level',
        formula: 'gpu-core.reduction'
      }
    })
  );

  return nodes;
}

/** Emits the portable 256-lane shared-memory reduction. */
function getPortableReductionSource(combine: string): string {
  return `firstScratch[lane] = value;
  secondScratch[lane] = secondValue;
  validityScratch[lane] = valid;
  workgroupBarrier();
  var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u;
  loop {
    if (lane < stride) {
      ${combine}
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }`;
}

/** Emits one collective per subgroup followed by a short shared-memory merge of subgroup totals. */
function getSubgroupReductionSource(props: {
  format: GPUScalarFormat;
  operation: GPUReductionOperation;
  combine: string;
}): string {
  const firstCollective = getSubgroupCollective(
    props.format,
    props.operation === 'extent' ? 'min' : props.operation,
    'value',
    'valid'
  );
  const secondCollective = getSubgroupCollective(
    props.format,
    props.operation === 'extent' ? 'max' : props.operation,
    'secondValue',
    'valid'
  );
  return `let subgroupValid = subgroupMax(valid);
  let subgroupFirst = ${firstCollective};
  let subgroupSecond = ${secondCollective};
  if (subgroupInvocationId == 0u) {
    firstScratch[subgroupId] = subgroupFirst;
    secondScratch[subgroupId] = subgroupSecond;
    validityScratch[subgroupId] = subgroupValid;
  }
  workgroupBarrier();
  let subgroupCount = ${REDUCTION_WORKGROUP_SIZE}u / subgroupSize;
  if (subgroupCount > 1u) {
    var stride = subgroupCount / 2u;
    loop {
      if (lane < stride) {
        ${props.combine}
      }
      workgroupBarrier();
      if (stride == 1u) { break; }
      stride = stride / 2u;
    }
  }`;
}

/** Returns a type-correct subgroup collective with neutral values for invalid lanes. */
function getSubgroupCollective(
  format: GPUScalarFormat,
  operation: Exclude<GPUReductionOperation, 'extent'>,
  value: string,
  validity: string
): string {
  if (operation === 'sum') {
    return `subgroupAdd(${value})`;
  }
  const identity = getReductionIdentity(format, operation);
  const collective = operation === 'min' ? 'subgroupMin' : 'subgroupMax';
  return `${collective}(select(${identity}, ${value}, ${validity} != 0u))`;
}

/** Returns the neutral value used to exclude an invalid lane from min/max collectives. */
function getReductionIdentity(
  format: GPUScalarFormat,
  operation: Exclude<GPUReductionOperation, 'sum' | 'extent'>
): string {
  if (format === 'uint32') {
    return operation === 'min' ? '4294967295u' : '0u';
  }
  if (format === 'sint32') {
    return operation === 'min' ? '2147483647' : '(-2147483647 - 1)';
  }
  return operation === 'min' ? '3.402823466e+38' : '-3.402823466e+38';
}

/** Copies the single partial row to caller output and maps an invalid aggregate to zero. */
function addFinalizeReductionPass<Parameters, T extends GPUScalarFormat>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    format: T;
    inputValues: GraphDataView<T>;
    inputValidity?: GraphDataView<'uint32'>;
    output: GraphDataView<T>;
    valuesPerRow: number;
    compensated: boolean;
  }
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const zero = getZeroLiteral(props.format);
  // A compensated sum carries (high, low) per row and publishes their float32 total.
  const outputLines = Array.from(
    {length: props.compensated ? 1 : props.valuesPerRow},
    (_, index) => {
      const value = props.compensated
        ? '(inputValues[INPUT_OFFSET] + inputValues[INPUT_OFFSET + 1u])'
        : `inputValues[INPUT_OFFSET + ${index}u]`;
      return `outputValues[OUTPUT_OFFSET + ${index}u] = ${props.inputValidity ? `select(${zero}, ${value}, valid)` : value};`;
    }
  ).join('\n  ');
  const bindings: GPUKernelBindingList = [
    {name: 'inputValues', view: props.inputValues, access: 'read'},
    props.inputValidity && {name: 'inputValidity', view: props.inputValidity, access: 'read'},
    {name: 'outputValues', view: props.output, access: 'write'}
  ];
  const constants: Record<string, number> = {
    INPUT_OFFSET: getViewElementOffset(props.inputValues),
    OUTPUT_OFFSET: getViewElementOffset(props.output),
    ...(props.inputValidity && {VALIDITY_OFFSET: getViewElementOffset(props.inputValidity)})
  };
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(1) fn main() {
  ${props.inputValidity ? 'let valid = inputValidity[VALIDITY_OFFSET] != 0u;' : ''}
  ${outputLines}
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: props.id,
      source,
      bindings,
      constants,
      dispatchLayout: {x: 1, y: 1, z: 1},
      record: {
        moduleId: 'GPUReduction/finalize',
        workgroupSize: 1,
        invocationCount: 1,
        formula: 'gpu-core.reduction'
      }
    })
  );

  return nodes;
}

/** Writes the documented zero result for an input with no rows. */
function addClearReductionPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  output: GraphDataView<GPUScalarFormat>
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const zero = getZeroLiteral(output.format);
  const bindings: GPUKernelBinding[] = [{name: 'outputValues', view: output, access: 'write'}];
  const constants = {OUTPUT_OFFSET: getViewElementOffset(output), OUTPUT_LENGTH: output.length};
  nodes.push(
    ...addKernelPass(graph, {
      id: `${id}-clear`,
      source: `${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(1) fn main() {
  for (var index = 0u; index < OUTPUT_LENGTH; index++) { outputValues[OUTPUT_OFFSET + index] = ${zero}; }
}`,
      bindings,
      constants,
      dispatchLayout: {x: 1, y: 1, z: 1},
      record: {moduleId: 'GPUReduction/clear', workgroupSize: 1, invocationCount: 1}
    })
  );

  return nodes;
}

/** Adds one reduction kernel node; reduction kernels carry no workload estimate. */
function addKernelPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    source: string;
    bindings: GPUKernelBindingList;
    constants: Readonly<Record<string, number>>;
    dispatchLayout: GPUBoundedDispatchLayout;
    /** Kernel facts recorded on recording graphs only. */
    record: {
      moduleId: string;
      workgroupSize: number;
      invocationCount: number;
      dedupName?: string;
      formula?: string;
    };
  }
): readonly GPUCommandNode<Parameters>[] {
  return [
    createGPUKernelNode(graph, {
      id: props.id,
      operation: 'GPUReduction',
      moduleId: props.record.moduleId,
      formula: props.record.formula,
      dedupName: props.record.dedupName,
      source: props.source,
      constants: props.constants,
      bindings: props.bindings,
      workgroupSize: props.record.workgroupSize,
      dispatchLayout: props.dispatchLayout,
      invocationCount: props.record.invocationCount,
      workloadScope: 'never'
    })
  ];
}

/** Returns the WGSL scalar type corresponding to a supported GPU storage format. */
function getShaderType(format: GPUScalarFormat): 'u32' | 'i32' | 'f32' {
  return format === 'uint32' ? 'u32' : format === 'sint32' ? 'i32' : 'f32';
}

/** Returns a type-correct WGSL zero literal for a supported GPU storage format. */
function getZeroLiteral(format: GPUScalarFormat): string {
  return format === 'uint32' ? '0u' : format === 'sint32' ? '0' : '0.0';
}

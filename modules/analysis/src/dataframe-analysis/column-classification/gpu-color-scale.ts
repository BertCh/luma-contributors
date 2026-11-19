// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createClassificationKernelNode as createWGSLKernelNode} from './classification-kernel-node';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {COLUMN_ORDERED_KEY_WGSL} from './column-classification-shared';
import {COLOR_SCALE_WGSL} from './color-scale-wgsl';
import {GPU_COLOR_SCALE_PARAMETER_LENGTH} from './color-scale-parameters';

import {
  type ClassificationRows,
  type ClassificationRowView,
  validateClassificationScratch,
  validateClassificationBindings,
  validateClassificationRows,
  validateClassificationOutput,
  validateClassificationOutputs
} from './column-classification-rows';

const OPERATION = 'GPUColorScale';
/** Class counts use u32 atomics and must not wrap even when every row has one class. */
const MAXIMUM_COUNTED_ROW_COUNT = 2 ** 32 - 1;

/** Caller-owned outputs of {@link GPUColorScale}. At least one is required. */
export type GPUColorScaleOutput<
  Rows extends ClassificationRows<'float32' | 'uint32' | 'sint32'> =
    | GraphDataView<'float32'>
    | GraphDataView<'uint32'>
    | GraphDataView<'sint32'>
> = {
  /** Packed `rgba8` colour per row (`r | g << 8 | b << 16 | a << 24`). No-data rows get the no-data colour. */
  colors?: ClassificationRowView<'uint32', Rows>;
  /**
   * Palette class per row, `0xffffffff` for no-data rows. Classes are always clamped to
   * `[0, paletteCount - 1]`, so a class always indexes the palette.
   */
  classIndices?: ClassificationRowView<'uint32', Rows>;
  /** Rows per class, `maximumPaletteCount` rows. Rewritten every encoding; no-data rows are not counted. */
  classCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUColorScale}.
 *
 * Per-frame (no recompile): the contents of `parameters` (scale type, counts, interpolation,
 * clamp, no-data colour, log floor, exponent), `domain`, `palette`, `domainCount`, `mask` and
 * `values`. Topology (needs a new graph): view lengths, `maximumDomainCount`,
 * `maximumPaletteCount`, the `values` format, and which optional views and outputs are present, and the ordered row chunk lengths.
 */
export type GPUColorScaleProps<
  Rows extends ClassificationRows<'float32' | 'uint32' | 'sint32'> =
    | GraphDataView<'float32'>
    | GraphDataView<'uint32'>
    | GraphDataView<'sint32'>
> = {
  /** Prefix for generated node and transient IDs. Defaults to `'color-scale'`. */
  id?: string;
  /**
   * Packed or chunked values. `float32` serves every scale except `ordinal`; `uint32` category codes select
   * the `ordinal` scale at compile time (the `scale` parameter is then ignored).
   */
  values: Rows;
  /**
   * How an integer `values` view is read. `'ordinal'` (default) treats `uint32` codes as
   * categories for the `ordinal` scale, as above. `'numeric'` converts the integers to f32 on
   * read (exact up to 2^24 in magnitude) and applies the per-frame `scale` parameter as for a
   * float column, so a count column can be colored without a cast pass. A `sint32` view is
   * always numeric. Ignored for `float32`. Topology.
   */
  integerValues?: 'ordinal' | 'numeric';
  /** Optional row-aligned packed or chunked `uint32` mask; zero makes the row no-data. */
  mask?: ClassificationRowView<'uint32', Rows>;
  /**
   * Packed float32 domain, monotonic (continuous scales may descend), at least `maximumDomainCount` rows. Edges or stops, for
   * example the `breaks` of `GPUClassBreaks` or quantiles of `GPUColumnQuantiles`, used directly
   * with no readback. Entries may be `+-Infinity`; edges must not be NaN.
   */
  domain: GraphDataView<'float32'>;
  /**
   * Optional one-row `uint32` view holding a CLASS count `k` (for example the `classCount` output
   * of `GPUClassBreaks`). The active domain then has exactly `k + 1` edges, clamped to
   * `maximumDomainCount`; `k = 0` makes every row no-data. When absent, the `domainCount`
   * parameter is the number of active domain entries.
   */
  domainCount?: GraphDataView<'uint32'>;
  /** Packed `uint32` `rgba8` palette, at least `maximumPaletteCount` rows. */
  palette: GraphDataView<'uint32'>;
  /** Per-frame parameters written with `getGPUColorScaleParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Largest active domain length, compile-time. */
  maximumDomainCount: number;
  /** Largest active palette length, compile-time. */
  maximumPaletteCount: number;
  /** Caller-owned outputs. */
  output: GPUColorScaleOutput<Rows>;
};

/**
 * Applies a colour scale to a column on the GPU, with the scale type, domain, and palette all
 * per-frame data, so switching them never rebuilds or recompiles the graph.
 *
 * Semantics follow d3 scales. Continuous scales (`linear`, `sqrt`, `pow` with `exponent`, `log`,
 * `symlog`) transform the value and the domain ends, then normalise over `[domain[0],
 * domain[domainCount - 1]]` to `t`. When the domain has more than 2 entries and
 * `paletteCount == domainCount`, stop `i` maps to palette entry `i` and the mapping is piecewise
 * linear in the transformed space (a d3 multi-stop scale); otherwise the palette is a uniform
 * ramp. A degenerate domain (zero span) maps every valid row to `t = 0.5`. With `clamp` the
 * position is clamped to the ends; without it an out-of-domain row is no-data unless `extrapolate`
 * explicitly extends the end segments. Extrapolated channels saturate to bytes; class indices stay clamped. The class is
 * `min(floor(t * paletteCount), paletteCount - 1)` (piecewise: `min(floor(position),
 * paletteCount - 1)`, the stop segment). `step` interpolation outputs `palette[class]` bit exactly;
 * `linear` interpolation blends adjacent entries per channel in f32 and rounds half up to the
 * nearest byte (expect up to 1 per channel difference from a double precision reference).
 * `log` replaces values and domain entries `<= 0` with `logFloor`; a NaN floor makes them no-data.
 * Explicit `logMode: 'domain-sign'` instead uses log(x) on positive domains and -log(-x) on negative
 * domains. Domain stops must be finite, nonzero and same-sign; zero/wrong-sign values are no-data
 * except for a constant valid domain, which maps finite values to the midpoint. Native log and
 * extrapolation reject nonfinite source/normalized values. These controls retain the nine-word layout.
 *
 * `quantize` splits `[domain[0], domain[last]]` into `paletteCount` equal intervals; `clamp`
 * applies as above. `threshold` and `quantile` classify by the shared edge convention: the class
 * is the number of inner edges `domain[1 .. n - 2]` that are `<= v` (d3 `bisectRight`), found by
 * binary search on order-preserving integer keys, so class ids are exact; values outside the
 * ends fall into the first or last class regardless of `clamp`. Classes beyond the palette clamp
 * to the last entry. These class-based scales and `ordinal` ignore `interpolation`.
 * `ordinal` uses `palette[code]` for `code < paletteCount` and no-data (d3 `unknown`) otherwise.
 *
 * NaN and masked rows get the no-data colour and class `0xffffffff`, and are not counted.
 * Class counts use integer atomics and are exact; the colours are a pure function of the inputs
 * and so are run-to-run identical.
 *
 * Row inputs and outputs accept GraphVectorView with identical ordered chunk lengths.
 * Controls and class counts remain single packed views. Each nonempty chunk dispatches independently;
 * counts are reset once and accumulated across all chunks, bitwise equal to packed input.
 * With classCounts, at most 2^32 - 1 source rows are accepted to prevent counter overflow.
 */
export class GPUColorScale<
  Rows extends ClassificationRows<'float32' | 'uint32' | 'sint32'> =
    | GraphDataView<'float32'>
    | GraphDataView<'uint32'>
    | GraphDataView<'sint32'>
> implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUColorScaleProps<Rows>;
  /** Upper bound for chunk dispatches, including the single counts reset. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUColorScaleProps<Rows>) {
    this.id = props.id ?? 'color-scale';
    this.props = props;
    const id = this.id;
    const {output, maximumDomainCount, maximumPaletteCount} = props;
    validateCapacity(id, 'maximumDomainCount', maximumDomainCount);
    validateCapacity(id, 'maximumPaletteCount', maximumPaletteCount);
    if (!output.colors && !output.classIndices && !output.classCounts) {
      throw new Error(`${id} needs at least one output (colors, classIndices, or classCounts)`);
    }
    const rows = props.values.length;
    if (output.classCounts)
      validateMaximumCount(
        id,
        'values length with output.classCounts',
        rows,
        MAXIMUM_COUNTED_ROW_COUNT
      );
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    validateClassificationRows(id, 'values', props.values, ['float32', 'uint32', 'sint32']);
    if (
      props.integerValues !== undefined &&
      !['ordinal', 'numeric'].includes(props.integerValues)
    ) {
      throw new Error(`${id} integerValues must be 'ordinal' or 'numeric'`);
    }
    validateSinglePackedView(props.domain, ['float32'], `${id} domain`);
    validateSinglePackedView(props.palette, ['uint32'], `${id} palette`);
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.mask) {
      validateClassificationRows(id, 'mask', props.mask, ['uint32'], props.values);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    if (props.domain.length < maximumDomainCount) {
      throw new Error(`${id} domain must hold maximumDomainCount rows`);
    }
    if (props.palette.length < maximumPaletteCount) {
      throw new Error(`${id} palette must hold maximumPaletteCount rows`);
    }
    if (props.parameters.length < GPU_COLOR_SCALE_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_COLOR_SCALE_PARAMETER_LENGTH} float32 values`
      );
    }
    if (props.domainCount) {
      validateSinglePackedView(props.domainCount, ['uint32'], `${id} domainCount`);
      if (props.domainCount.length < 1) {
        throw new Error(`${id} domainCount must contain one uint32 row`);
      }
    }
    for (const name of ['colors', 'classIndices'] as const) {
      const view = output[name];
      if (view) {
        validateClassificationOutput(id, `output.${name}`, view, 'uint32', props.values);
        if (view.length < rows) {
          throw new Error(`${id} output.${name} must hold one row per value`);
        }
      }
    }
    if (output.classCounts) {
      validateSinglePackedView(output.classCounts, ['uint32'], `${id} output.classCounts`);
      if (output.classCounts.length < maximumPaletteCount) {
        throw new Error(`${id} output.classCounts must hold maximumPaletteCount rows`);
      }
    }
    const outputs = [output.colors, output.classIndices, output.classCounts];
    validateClassificationOutputs(id, outputs);
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      props.values,
      props.mask,
      props.domain,
      props.domainCount,
      props.palette,
      props.parameters
    ]);
    this.estimatedCommandNodeCount =
      getGraphViewChunks(props.values).filter(chunk => chunk.length > 0).length *
        (1 + Number(Boolean(output.colors || output.classCounts))) +
      Number(Boolean(output.classCounts));
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  /** Returns the optional init, classify, and colorize nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;

    const {output, maximumDomainCount, maximumPaletteCount} = props;
    validateClassificationBindings(id, graph, {
      values: props.values,
      mask: props.mask,
      domain: props.domain,
      domainCount: props.domainCount,
      palette: props.palette,
      parameters: props.parameters,
      'output.colors': output.colors,
      'output.classIndices': output.classIndices,
      'output.classCounts': output.classCounts
    });
    const maximumRows = getGraphViewChunks(props.values).reduce(
      (maximum, chunk) => Math.max(maximum, chunk.length),
      0
    );
    validateClassificationScratch(graph, id, {
      'class scratch': maximumRows,
      positions: maximumRows
    });
    const read = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'i32' | 'f32'
    ): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const declarations = `const MAXIMUM_DOMAIN_COUNT: u32 = ${maximumDomainCount}u;
const MAXIMUM_PALETTE_COUNT: u32 = ${maximumPaletteCount}u;
${COLUMN_ORDERED_KEY_WGSL}
${COLOR_SCALE_WGSL}`;
    const nodes: GPUCommandNode<Parameters>[] = [];

    if (output.classCounts) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-init`,
          operation: OPERATION,
          variant: 'init',
          bindings: [
            {
              name: 'counts',
              view: output.classCounts,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: maximumPaletteCount,
          body: 'counts[countsOffset + index] = 0u;'
        })
      );
    }

    const valueChunks = getGraphViewChunks(this.props.values);
    for (let chunkIndex = 0; chunkIndex < valueChunks.length; chunkIndex++) {
      if (valueChunks[chunkIndex].length === 0) continue;
      const props = {
        ...this.props,
        values: getGraphViewChunks(this.props.values)[chunkIndex],
        mask: this.props.mask ? getGraphViewChunks(this.props.mask)[chunkIndex] : undefined,
        output: {
          ...this.props.output,
          classIndices: this.props.output.classIndices
            ? getGraphViewChunks(this.props.output.classIndices)[chunkIndex]
            : undefined,
          colors: this.props.output.colors
            ? getGraphViewChunks(this.props.output.colors)[chunkIndex]
            : undefined
        }
      };
      const {output} = props;
      const chunkId = valueChunks.length === 1 ? id : `${id}-chunk-${chunkIndex}`;
      const rows = props.values.length;
      const isOrdinal = props.values.format === 'uint32' && props.integerValues !== 'numeric';
      const integerType =
        props.values.format === 'float32'
          ? undefined
          : props.values.format === 'uint32'
            ? 'u32'
            : 'i32';
      const classes =
        output.classIndices ?? createTransientView(graph, `${chunkId}-classes`, 'uint32', rows);
      const positions = createTransientView(graph, `${chunkId}-positions`, 'float32', rows);

      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${chunkId}-classify`,
          operation: OPERATION,
          variant: isOrdinal ? 'classify-ordinal' : integerType ? 'classify-integer' : 'classify',
          bindings: [
            read('values', props.values, integerType ?? 'f32'),
            ...(props.mask ? [read('rowMask', props.mask, 'u32')] : []),
            // The ordinal body never reads the domain; an unused declared binding breaks the inferred layout.
            ...(isOrdinal ? [] : [read('domain', props.domain, 'f32')]),
            ...(!isOrdinal && props.domainCount
              ? [read('domainCountIn', props.domainCount, 'u32')]
              : []),
            read('params', props.parameters, 'f32'),
            {name: 'classes', view: classes, type: 'u32', access: 'read_write'},
            {
              name: 'positions',
              view: positions,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: rows,
          declarations,
          body: `classes[classesOffset + index] = NO_CLASS;
  positions[positionsOffset + index] = 0.0;
  ${props.mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return;\n  }' : ''}
  let paletteCount = min(u32(max(params[paramsOffset + 2u], 0.0)), MAXIMUM_PALETTE_COUNT);
  if (paletteCount == 0u) {
    return;
  }
    ${isOrdinal ? ORDINAL_BODY : getFloatBody(props, integerType !== undefined)}`
        })
      );

      if (output.colors || output.classCounts) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${chunkId}-colorize`,
            operation: OPERATION,
            variant: 'colorize',
            bindings: [
              read('classes', classes, 'u32'),
              // The counts-only body reads neither the ramp positions nor the palette; an unused declared binding breaks the inferred layout.
              ...(output.colors
                ? [read('positions', positions, 'f32'), read('palette', props.palette, 'u32')]
                : []),
              read('params', props.parameters, 'f32'),
              ...(output.colors
                ? [
                    {
                      name: 'colors',
                      view: output.colors,
                      type: 'u32',
                      access: 'read_write'
                    } as WGSLKernelBinding
                  ]
                : []),
              ...(output.classCounts
                ? [
                    {
                      name: 'counts',
                      view: output.classCounts,
                      type: 'atomic<u32>',
                      access: 'read_write'
                    } as WGSLKernelBinding
                  ]
                : [])
            ],
            invocationCount: rows,
            declarations: `const IS_ORDINAL: bool = ${isOrdinal};
${declarations}`,
            body: `let classIndex = classes[classesOffset + index];
  let paletteCount = min(u32(max(params[paramsOffset + 2u], 0.0)), MAXIMUM_PALETTE_COUNT);
  let isValid = classIndex != NO_CLASS && classIndex < paletteCount;
  ${output.classCounts ? 'if (isValid) {\n    atomicAdd(&counts[countsOffset + classIndex], 1u);\n  }' : ''}
  ${
    output.colors
      ? `let noDataColor = u32(params[paramsOffset + 5u]) | (u32(params[paramsOffset + 6u]) << 16u);
  var color = noDataColor;
  if (isValid) {
    color = palette[paletteOffset + classIndex];
    let scale = u32(max(params[paramsOffset], 0.0));
    if (!IS_ORDINAL && (scale <= 4u || scale == 9u) && params[paramsOffset + 3u] == 1.0) {
      let position = positions[positionsOffset + index];
      let lower = u32(clamp(floor(position), 0.0, f32(max(paletteCount, 2u) - 2u)));
      let upper = min(lower + 1u, paletteCount - 1u);
      color = blendColors(
        palette[paletteOffset + lower],
        palette[paletteOffset + upper],
        position - f32(lower)
      );
    }
  }
  colors[colorsOffset + index] = color;`
      : ''
  }`
          })
        );
      }
    }
    return nodes;
  }
}

const ORDINAL_BODY = /* wgsl */ `let code = values[valuesOffset + index];
  if (code < paletteCount) {
    classes[classesOffset + index] = code;
  }`;

function getFloatBody(
  props: GPUColorScaleProps<ClassificationRows<'float32' | 'uint32' | 'sint32'>>,
  isInteger: boolean
): string {
  const domainCountSource = props.domainCount
    ? `let classTotal = domainCountIn[domainCountInOffset];
  domainCount = select(0u, min(classTotal, MAXIMUM_DOMAIN_COUNT) + 1u, classTotal > 0u);`
    : '';
  return `var domainCount = u32(max(params[paramsOffset + 1u], 0.0));
  ${domainCountSource}
  domainCount = min(domainCount, min(MAXIMUM_DOMAIN_COUNT, arrayLength(&domain) - domainOffset));
  if (domainCount == 0u) {
    return;
  }
  let value = ${isInteger ? 'f32(values[valuesOffset + index])' : 'values[valuesOffset + index]'};
  if (isNanBits(value)) {
    return;
  }
  let scale = u32(max(params[paramsOffset], 0.0));
  let isClamped = params[paramsOffset + 4u] == 1.0;
  let isExtrapolating = params[paramsOffset + 4u] == 2.0;
  let logFloor = params[paramsOffset + 7u];
  let exponent = params[paramsOffset + 8u];
  var classIndex = NO_CLASS;
  var rampPosition = 0.0;
  if (scale == 6u || scale == 7u) {
    // Threshold and quantile: count the inner edges domain[1 .. domainCount - 2] that are <= value.
    if (domainCount < 2u) {
      return;
    }
    let valueKey = getComparisonKey(value);
    var low = 1u;
    var high = domainCount - 1u;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (getComparisonKey(domain[domainOffset + middle]) <= valueKey) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    classIndex = min(low - 1u, paletteCount - 1u);
  } else if (scale == 5u) {
    if (domainCount < 2u) {
      return;
    }
    let lowEdge = domain[domainOffset];
    let span = domain[domainOffset + domainCount - 1u] - lowEdge;
    var t = select(0.5, (value - lowEdge) / span, span != 0.0);
    if (isNanBits(t) || (!isClamped && (t < 0.0 || t > 1.0))) {
      return;
    }
    t = clamp(t, 0.0, 1.0);
    classIndex = min(u32(floor(t * f32(paletteCount))), paletteCount - 1u);
  } else if (scale <= 4u || scale == 9u) {
    if (domainCount < 2u) {
      return;
    }
    let negativeLog = domain[domainOffset] < 0.0;
    let firstStop = transformColorValue(domain[domainOffset], scale, exponent, logFloor, negativeLog);
    let lastStop = transformColorValue(domain[domainOffset + domainCount - 1u], scale, exponent, logFloor, negativeLog);
    let transformed = transformColorValue(value, scale, exponent, logFloor, negativeLog);
    if (isNanBits(firstStop) || isNanBits(lastStop)) { return; }
    // Native finite constant domains normalize to the midpoint, including zero/wrong-sign log inputs.
    let t = normalizeColorValue(transformed, firstStop, lastStop);
    if (isNanBits(t) || (!isClamped && !isExtrapolating && (t < 0.0 || t > 1.0))) {
      return;
    }
    if ((isExtrapolating || scale == 9u) &&
        ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u ||
         (bitcast<u32>(t) & 0x7f800000u) == 0x7f800000u)) { return; }
    if (domainCount > 2u && paletteCount == domainCount) {
      if (!isExtrapolating && t <= 0.0) {
        rampPosition = 0.0;
      } else if (!isExtrapolating && t >= 1.0) {
        rampPosition = f32(paletteCount - 1u);
      } else {
        // Select the first/last segment outside the domain, including descending domains.
        var segment = 0u;
        let descending = lastStop < firstStop;
        for (var stopIndex = 1u; stopIndex + 1u < domainCount; stopIndex++) {
          let stop = transformColorValue(domain[domainOffset + stopIndex], scale, exponent, logFloor, negativeLog);
          if (select(stop <= transformed, stop >= transformed, descending)) {
            segment = stopIndex;
          }
        }
        let segmentStart = transformColorValue(domain[domainOffset + segment], scale, exponent, logFloor, negativeLog);
        let segmentEnd = transformColorValue(domain[domainOffset + segment + 1u], scale, exponent, logFloor, negativeLog);
        var fraction = normalizeColorValue(transformed, segmentStart, segmentEnd);
        if (isNanBits(fraction)) { return; }
        if (!isExtrapolating) {
          fraction = clamp(fraction, 0.0, 1.0);
          // Preserve the legacy duplicate-stop choice unless native log was explicitly requested.
          if (segmentStart == segmentEnd && scale != 9u) { fraction = 0.0; }
        }
        rampPosition = f32(segment) + fraction;
      }
      classIndex = u32(clamp(floor(rampPosition), 0.0, f32(paletteCount - 1u)));
    } else {
      let clampedT = clamp(t, 0.0, 1.0);
      rampPosition = select(clampedT, t, isExtrapolating) * f32(paletteCount - 1u);
      classIndex = min(u32(floor(clampedT * f32(paletteCount))), paletteCount - 1u);
    }
  } else {
    return;
  }
  if ((isExtrapolating || scale == 9u) &&
      (bitcast<u32>(rampPosition) & 0x7f800000u) == 0x7f800000u) { return; }
  classes[classesOffset + index] = classIndex;
  positions[positionsOffset + index] = rampPosition;`;
}

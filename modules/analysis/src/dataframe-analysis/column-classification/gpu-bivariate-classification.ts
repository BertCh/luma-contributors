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
import {GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH} from './bivariate-classification-parameters';

import {
  type ClassificationRows,
  type ClassificationRowView,
  validateClassificationScratch,
  validateClassificationBindings,
  validateClassificationRows,
  validateClassificationOutput,
  validateClassificationOutputs
} from './column-classification-rows';

const OPERATION = 'GPUBivariateClassification';
/** Class counts use u32 atomics and must not wrap even when every row has one class. */
const MAXIMUM_COUNTED_ROW_COUNT = 2 ** 32 - 1;

/** Largest per-axis class count. */
const MAXIMUM_AXIS_CLASS_COUNT = 16;

/** Caller-owned outputs of {@link GPUBivariateClassification}. At least one is required. */
export type GPUBivariateClassificationOutput<
  Rows extends ClassificationRows<'float32'> = GraphDataView<'float32'>
> = {
  /**
   * Joint class per row, `yClass * classCountX + xClass`, or `0xffffffff` for a masked row or a
   * row with a NaN X or Y value.
   */
  classIds?: ClassificationRowView<'uint32', Rows>;
  /** Packed `rgba8` colour per row (`r | g << 8 | b << 16 | a << 24`), with value-by-alpha applied. */
  colors?: ClassificationRowView<'uint32', Rows>;
  /** Rows per joint class, `maximumClassCount ** 2` rows, rewritten every encoding. */
  classCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUBivariateClassification}.
 *
 * Per-frame (no recompile): the contents of `parameters` (class counts, no-data colour,
 * value-by-alpha), `breaksX`, `breaksY`, `palette`, `alphaValues`, `mask` and the values.
 * Topology (needs a new graph): view lengths, `maximumClassCount`, and which optional views and
 * outputs are present.
 */
export type GPUBivariateClassificationProps<
  Rows extends ClassificationRows<'float32'> = GraphDataView<'float32'>
> = {
  /** Prefix for generated node and transient IDs. Defaults to `'bivariate-classification'`. */
  id?: string;
  /** Packed or chunked float32 X values. */
  valuesX: Rows;
  /** Float32 Y values with the same chunk boundaries as valuesX. */
  valuesY: ClassificationRowView<'float32', Rows>;
  /** Optional row-aligned packed or chunked `uint32` mask; zero makes the row no-data. */
  mask?: ClassificationRowView<'uint32', Rows>;
  /**
   * X edges, non-decreasing, at least `maximumClassCount + 1` rows; the first `classCountX + 1`
   * are active (for example the `breaks` of `GPUClassBreaks`). Edges must not be NaN.
   */
  breaksX: GraphDataView<'float32'>;
  /** Y edges, same layout as `breaksX`. */
  breaksY: GraphDataView<'float32'>;
  /**
   * Packed `rgba8` palette, row-major: `palette[yClass * classCountX + xClass]`, at least
   * `maximumClassCount ** 2` rows.
   */
  palette: GraphDataView<'uint32'>;
  /**
   * Optional row-aligned packed or chunked float32 value-by-alpha variable (for example an uncertainty). It is
   * read only when `parameters` enables value-by-alpha.
   */
  alphaValues?: ClassificationRowView<'float32', Rows>;
  /** Per-frame parameters written with `getGPUBivariateClassificationParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Largest class count per axis, `1..16`, compile-time. */
  maximumClassCount: number;
  /** Caller-owned outputs. */
  output: GPUBivariateClassificationOutput<Rows>;
};

/**
 * Classifies two columns into an `n x n` bivariate class grid and colours each row from a
 * bivariate palette, optionally fading by a third variable (value-by-alpha).
 *
 * Each axis uses the shared edge convention: with `k` classes the axis has `k + 1` edges and a
 * value's class is the number of inner edges `e[1 .. k - 1]` that are `<= v` (d3 `bisectRight`,
 * found by a linear-time binary search on order-preserving integer keys, so class ids are exact).
 * Values below the first or above the last edge clamp to the first or last class. The joint class
 * is `yClass * classCountX + xClass` and the colour is `palette[joint class]`.
 *
 * Value-by-alpha, when enabled and `alphaValues` is given: `factor = lerp(minimumAlpha, 1,
 * clamp((a - lo) / (hi - lo), 0, 1))` and the output alpha is `round(paletteAlpha * factor)`
 * (half up, f32; expect up to 1 difference from a double precision reference). A NaN alpha value
 * uses `minimumAlpha` (the row keeps its class colour, faded); `hi <= lo` uses factor 1.
 * Rows that are masked or have a NaN X or Y value get the no-data colour (unfaded), class
 * `0xffffffff`, and are not counted. Counts use integer atomics and are exact (at most 2^32 - 1 source rows with classCounts).
 *
 * Row inputs and outputs accept GraphVectorView with identical ordered chunk lengths.
 * Controls and class counts remain single packed views. Each nonempty chunk dispatches independently;
 * counts are reset once and accumulated across all chunks, bitwise equal to packed input.
 * With classCounts, at most 2^32 - 1 source rows are accepted to prevent counter overflow.
 */
export class GPUBivariateClassification<
  Rows extends ClassificationRows<'float32'> = GraphDataView<'float32'>
> implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUBivariateClassificationProps<Rows>;
  /** Upper bound for chunk dispatches, including the single counts reset. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUBivariateClassificationProps<Rows>) {
    this.id = props.id ?? 'bivariate-classification';
    this.props = props;
    const id = this.id;
    const {output, maximumClassCount} = props;
    validateCapacity(id, 'maximumClassCount', maximumClassCount, 1, MAXIMUM_AXIS_CLASS_COUNT);
    if (!output.classIds && !output.colors && !output.classCounts) {
      throw new Error(`${id} needs at least one output (classIds, colors, or classCounts)`);
    }
    const rows = props.valuesX.length;
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
    validateClassificationRows(id, 'valuesX', props.valuesX, ['float32']);
    validateClassificationRows(id, 'valuesY', props.valuesY, ['float32'], props.valuesX);
    validateSinglePackedView(props.breaksX, ['float32'], `${id} breaksX`);
    validateSinglePackedView(props.breaksY, ['float32'], `${id} breaksY`);
    validateSinglePackedView(props.palette, ['uint32'], `${id} palette`);
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.alphaValues) {
      validateClassificationRows(id, 'alphaValues', props.alphaValues, ['float32'], props.valuesX);
    }
    if (props.mask) {
      validateClassificationRows(id, 'mask', props.mask, ['uint32'], props.valuesX);
    }
    for (const [name, view] of [
      ['valuesY', props.valuesY],
      ['mask', props.mask],
      ['alphaValues', props.alphaValues]
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal valuesX length`);
      }
    }
    for (const [name, view] of [
      ['breaksX', props.breaksX],
      ['breaksY', props.breaksY]
    ] as const) {
      if (view.length < maximumClassCount + 1) {
        throw new Error(`${id} ${name} must hold maximumClassCount + 1 rows`);
      }
    }
    const jointCount = maximumClassCount * maximumClassCount;
    if (props.palette.length < jointCount) {
      throw new Error(`${id} palette must hold maximumClassCount * maximumClassCount rows`);
    }
    if (props.parameters.length < GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH} float32 values`
      );
    }
    for (const name of ['classIds', 'colors'] as const) {
      const view = output[name];
      if (view) {
        validateClassificationOutput(id, `output.${name}`, view, 'uint32', props.valuesX);
        if (view.length < rows) {
          throw new Error(`${id} output.${name} must hold one row per value`);
        }
      }
    }
    if (output.classCounts) {
      validateSinglePackedView(output.classCounts, ['uint32'], `${id} output.classCounts`);
      if (output.classCounts.length < jointCount) {
        throw new Error(`${id} output.classCounts must hold maximumClassCount ** 2 rows`);
      }
    }
    const outputs = [output.classIds, output.colors, output.classCounts];
    validateClassificationOutputs(id, outputs);
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      props.valuesX,
      props.valuesY,
      props.mask,
      props.breaksX,
      props.breaksY,
      props.palette,
      props.alphaValues,
      props.parameters
    ]);
    this.estimatedCommandNodeCount =
      getGraphViewChunks(props.valuesX).filter(chunk => chunk.length > 0).length *
        (1 + Number(Boolean(output.colors || output.classCounts))) +
      Number(Boolean(output.classCounts));
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  /** Returns the optional init, classify, and colorize nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;

    const {output, maximumClassCount} = props;
    validateClassificationBindings(id, graph, {
      valuesX: props.valuesX,
      valuesY: props.valuesY,
      mask: props.mask,
      breaksX: props.breaksX,
      breaksY: props.breaksY,
      palette: props.palette,
      alphaValues: props.alphaValues,
      parameters: props.parameters,
      'output.classIds': output.classIds,
      'output.colors': output.colors,
      'output.classCounts': output.classCounts
    });
    const maximumRows = getGraphViewChunks(props.valuesX).reduce(
      (maximum, chunk) => Math.max(maximum, chunk.length),
      0
    );
    validateClassificationScratch(graph, id, {'class scratch': maximumRows});
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const declarations = `const MAXIMUM_CLASS_COUNT: u32 = ${maximumClassCount}u;
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
          invocationCount: maximumClassCount * maximumClassCount,
          body: 'counts[countsOffset + index] = 0u;'
        })
      );
    }

    const valueChunks = getGraphViewChunks(this.props.valuesX);
    for (let chunkIndex = 0; chunkIndex < valueChunks.length; chunkIndex++) {
      if (valueChunks[chunkIndex].length === 0) continue;
      const props = {
        ...this.props,
        valuesX: getGraphViewChunks(this.props.valuesX)[chunkIndex],
        valuesY: getGraphViewChunks(this.props.valuesY)[chunkIndex],
        mask: this.props.mask ? getGraphViewChunks(this.props.mask)[chunkIndex] : undefined,
        alphaValues: this.props.alphaValues
          ? getGraphViewChunks(this.props.alphaValues)[chunkIndex]
          : undefined,
        output: {
          ...this.props.output,
          classIds: this.props.output.classIds
            ? getGraphViewChunks(this.props.output.classIds)[chunkIndex]
            : undefined,
          colors: this.props.output.colors
            ? getGraphViewChunks(this.props.output.colors)[chunkIndex]
            : undefined
        }
      };
      const {output} = props;
      const chunkId = valueChunks.length === 1 ? id : `${id}-chunk-${chunkIndex}`;
      const rows = props.valuesX.length;
      const classes =
        output.classIds ?? createTransientView(graph, `${chunkId}-classes`, 'uint32', rows);

      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${chunkId}-classify`,
          operation: OPERATION,
          variant: 'classify',
          bindings: [
            read('valuesX', props.valuesX, 'f32'),
            read('valuesY', props.valuesY, 'f32'),
            ...(props.mask ? [read('rowMask', props.mask, 'u32')] : []),
            read('breaksX', props.breaksX, 'f32'),
            read('breaksY', props.breaksY, 'f32'),
            read('params', props.parameters, 'f32'),
            {name: 'classes', view: classes, type: 'u32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `${declarations}
// Number of inner edges edges[1 .. classCount - 1] that are <= value, as a u32 key comparison.
fn getAxisClass(valueKey: u32, classCount: u32, isX: bool) -> u32 {
  var low = 1u;
  var high = classCount;
  while (low < high) {
    let middle = (low + high) / 2u;
    var edge = 0.0;
    if (isX) {
      edge = breaksX[breaksXOffset + middle];
    } else {
      edge = breaksY[breaksYOffset + middle];
    }
    if (getComparisonKey(edge) <= valueKey) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low - 1u;
}`,
          body: `classes[classesOffset + index] = NO_CLASS;
  ${props.mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return;\n  }' : ''}
  let valueX = valuesX[valuesXOffset + index];
  let valueY = valuesY[valuesYOffset + index];
  let classCountX = min(u32(max(params[paramsOffset], 0.0)), MAXIMUM_CLASS_COUNT);
  let classCountY = min(u32(max(params[paramsOffset + 1u], 0.0)), MAXIMUM_CLASS_COUNT);
  if (isNanBits(valueX) || isNanBits(valueY) || classCountX == 0u || classCountY == 0u) {
    return;
  }
  let classX = getAxisClass(getComparisonKey(valueX), classCountX, true);
  let classY = getAxisClass(getComparisonKey(valueY), classCountY, false);
  classes[classesOffset + index] = classY * classCountX + classX;`
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
              // The counts-only body reads neither the palette nor the parameters; an unused declared binding breaks the inferred layout.
              ...(output.colors
                ? [read('palette', props.palette, 'u32'), read('params', props.parameters, 'f32')]
                : []),
              ...(props.alphaValues && output.colors
                ? [read('alphaValues', props.alphaValues, 'f32')]
                : []),
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
            declarations,
            body: `let classId = classes[classesOffset + index];
  let isValid = classId != NO_CLASS;
  ${output.classCounts ? 'if (isValid) {\n    atomicAdd(&counts[countsOffset + classId], 1u);\n  }' : ''}
  ${
    output.colors
      ? `let noDataColor = u32(params[paramsOffset + 2u]) | (u32(params[paramsOffset + 3u]) << 16u);
  var color = noDataColor;
  if (isValid) {
    color = palette[paletteOffset + classId];
    ${
      props.alphaValues
        ? `if (params[paramsOffset + 4u] != 0.0) {
      let alphaLow = params[paramsOffset + 5u];
      let alphaHigh = params[paramsOffset + 6u];
      let minimumAlpha = clamp(params[paramsOffset + 7u], 0.0, 1.0);
      let alphaValue = alphaValues[alphaValuesOffset + index];
      var ramp = 1.0;
      if (alphaHigh > alphaLow) {
        ramp = clamp((alphaValue - alphaLow) / (alphaHigh - alphaLow), 0.0, 1.0);
      }
      if (isNanBits(alphaValue)) {
        ramp = 0.0;
      }
      let factor = minimumAlpha + (1.0 - minimumAlpha) * ramp;
      let alpha = packChannel(unpackChannel(color, 3u) * factor);
      color = (color & 0x00ffffffu) | (alpha << 24u);
    }`
        : ''
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

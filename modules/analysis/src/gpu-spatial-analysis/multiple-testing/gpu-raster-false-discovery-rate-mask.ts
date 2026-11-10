// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPURasterHistogram} from '@luma.gl/graph/gpu-raster';
import {
  captureGraphCommandNodes,
  validateCapacity,
  validateFiniteNumber,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUMultipleTestingMethod} from '../../analysis-core/contracts/index';
import {
  GPUMultipleTestingAdjustment,
  GPU_MULTIPLE_TESTING_MAXIMUM_ROWS
} from './gpu-multiple-testing-adjustment';

const OPERATION = 'GPURasterFalseDiscoveryRateMask';

/** Float32 elements of the per-frame parameters: the significance level `alpha`. */
export const GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH = 1;

/**
 * Histogram bins of the optional p-value histogram. The upstream histogram clears and bins in one
 * bounded dispatch, and a p-value histogram finer than this resolves nothing a viewer can read.
 */
export const GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_MAXIMUM_HISTOGRAM_BINS = 65536;

/** Methods that give a significance decision; `'none'` is the uncorrected p-value and is not offered. */
export type GPURasterFalseDiscoveryRateMethod = Exclude<GPUMultipleTestingMethod, 'none'>;

/** Per-frame settings of {@link GPURasterFalseDiscoveryRateMask}. */
export type GPURasterFalseDiscoveryRateMaskSettings = {
  /** Significance level (false discovery rate for the Benjamini procedures), in `(0, 1]`. */
  alpha: number;
};

/** Packs {@link GPURasterFalseDiscoveryRateMaskSettings}: `[alpha]`. */
export function getGPURasterFalseDiscoveryRateMaskParameterValues(
  settings: GPURasterFalseDiscoveryRateMaskSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH)
): Float32Array {
  validateFiniteNumber(
    'getGPURasterFalseDiscoveryRateMaskParameterValues',
    'alpha',
    settings.alpha,
    Number.MIN_VALUE,
    1
  );
  target[0] = settings.alpha;
  return target;
}

/** Caller-owned outputs of {@link GPURasterFalseDiscoveryRateMask}. */
export type GPURasterFalseDiscoveryRateMaskOutput = {
  /** `uint32` per cell: 1 where the adjusted p-value is at most `alpha`, else 0 (also 0 where untested). */
  significant: GraphDataView<'uint32'>;
  /** Optional per-cell adjusted p-value in `[0, 1]` (statsmodels `multipletests`), NaN where untested. */
  adjustedPValues?: GraphDataView<'float32'>;
  /** Optional one-row family size `m` used by the adjustment. */
  testedCount?: GraphDataView<'uint32'>;
  /** Optional one-row count of significant cells. */
  significantCount?: GraphDataView<'uint32'>;
  /**
   * Optional one-row largest raw p-value that is significant (the step-up threshold for the
   * Benjamini procedures); NaN when nothing is significant.
   */
  criticalPValue?: GraphDataView<'float32'>;
  /**
   * Optional counts of tested p-values over equal bins of `[0, 1]`, one `uint32` per bin. Its
   * length is the bin count. Reading it against the uniform line shows how much signal the
   * raster holds before any correction.
   */
  histogram?: GraphDataView<'uint32'>;
};

/** Properties of {@link GPURasterFalseDiscoveryRateMask}. */
export type GPURasterFalseDiscoveryRateMaskProps = {
  id?: string;
  /** Number of cells (rows) of the p-value raster, in row-major order. */
  cellCount: number;
  /** p-values, one per cell. Values outside `[0, 1]` and non-finite values are untested. */
  pValues: GraphDataView<'float32'>;
  /** Per-cell QA; zero makes the cell untested, so it never enters the family. */
  validity?: GraphDataView<'uint32'>;
  /** Defaults to `'benjamini-hochberg'`. Use `'benjamini-yekutieli'` for dependent local statistics. */
  method?: GPURasterFalseDiscoveryRateMethod;
  /** Pins the family size from below, see {@link GPUMultipleTestingAdjustment}. Defaults to 0. */
  minimumFamilySize?: number;
  /** Float32, see {@link getGPURasterFalseDiscoveryRateMaskParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPURasterFalseDiscoveryRateMaskOutput;
};

/**
 * Field significance of a p-value raster: the multiple-testing adjustment of every tested cell
 * (`GPUMultipleTestingAdjustment`: Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli) and a
 * significance mask `adjusted <= alpha`, which is statsmodels' `reject`. Run it on the p-value
 * outputs of Mann-Kendall, Welch or correlation operators so a trend map shows only cells that
 * survive the correction (Wilks 2016).
 *
 * The family is every tested cell, so a viewport mask applied here would change `m`; mask only when
 * drawing. Cells with zero `validity`, or with a p-value that is not finite or not in `[0, 1]`,
 * are untested. The optional histogram (`GPURasterHistogram` over `[0, 1]`) counts the tested
 * p-values only.
 *
 * float32 limit: adjusted p-values are float32, so a cell whose adjusted value lies within a few
 * ulp of `alpha` may fall on either side of statsmodels' float64 decision.
 *
 * Per-frame: `parameters` (alpha) and the contents of the views. Compile-time: `cellCount`,
 * `method`, `minimumFamilySize`, the histogram bin count and which optional views exist.
 */
export class GPURasterFalseDiscoveryRateMask implements GPUCommandNodeProducer {
  readonly id: string;
  readonly method: GPURasterFalseDiscoveryRateMethod;

  constructor(readonly props: GPURasterFalseDiscoveryRateMaskProps) {
    const id = props.id ?? 'raster-false-discovery-rate-mask';
    this.id = id;
    this.method = props.method ?? 'benjamini-hochberg';
    validateRequirement(
      id,
      (['bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'] as string[]).includes(
        this.method
      ),
      `method must be bonferroni, holm, benjamini-hochberg or benjamini-yekutieli, got ${String(this.method)}`
    );
    const {output} = props;
    validateCapacity(id, 'cellCount', props.cellCount, 1, GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1);
    validatePackedView(props.pValues, ['float32'], `${id} pValues`);
    validatePackedView(output.significant, ['uint32'], `${id} output.significant`);
    validateRequirement(
      id,
      props.pValues.length === props.cellCount && output.significant.length === props.cellCount,
      'pValues and output.significant must hold cellCount entries'
    );
    if (props.validity) {
      validatePackedView(props.validity, ['uint32'], `${id} validity`);
      validateRequirement(
        id,
        props.validity.length === props.cellCount,
        'validity must hold cellCount entries'
      );
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH} entry`
    );
    if (output.adjustedPValues) {
      validatePackedView(output.adjustedPValues, ['float32'], `${id} output.adjustedPValues`);
      validateRequirement(
        id,
        output.adjustedPValues.length === props.cellCount,
        'adjustedPValues must hold cellCount entries'
      );
    }
    for (const [name, view] of [
      ['testedCount', output.testedCount],
      ['significantCount', output.significantCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} output.${name}`);
        validateRequirement(id, view.length === 1, `${name} must hold one entry`);
      }
    }
    if (output.criticalPValue) {
      validatePackedView(output.criticalPValue, ['float32'], `${id} output.criticalPValue`);
      validateRequirement(
        id,
        output.criticalPValue.length === 1,
        'criticalPValue must hold one entry'
      );
    }
    if (output.histogram) {
      validatePackedView(output.histogram, ['uint32'], `${id} output.histogram`);
      validateCapacity(
        id,
        'output.histogram bin count',
        output.histogram.length,
        1,
        GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_MAXIMUM_HISTOGRAM_BINS
      );
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.significant,
        output.adjustedPValues,
        output.testedCount,
        output.significantCount,
        output.criticalPValue,
        output.histogram
      ],
      [props.pValues, props.validity, props.parameters]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output, cellCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.pValues,
      props.validity,
      props.parameters,
      output.significant,
      output.adjustedPValues,
      output.testedCount,
      output.significantCount,
      output.criticalPValue,
      output.histogram
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Untested cells become NaN, so the adjustment's family is exactly the valid cells with p in [0, 1].
    const testedPValues = createTransientView(graph, `${id}-tested-p`, 'float32', cellCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tested`,
        operation: OPERATION,
        variant: 'tested',
        bindings: [
          {name: 'pValues', view: props.pValues, type: 'f32', access: 'read'},
          ...(props.validity
            ? [{name: 'validity', view: props.validity, type: 'u32', access: 'read'} as const]
            : []),
          {name: 'tested', view: testedPValues, type: 'f32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: GET_NAN_WGSL,
        body: `let p = pValues[pValuesOffset + index];
let isTested = ${props.validity ? 'validity[validityOffset + index] != 0u && ' : ''}p >= 0.0 && p <= 1.0;
tested[testedOffset + index] = select(getNaN(), p + 0.0, isTested);`
      })
    );

    const adjustedPValues =
      output.adjustedPValues ??
      createTransientView(graph, `${id}-adjusted-p`, 'float32', cellCount);
    nodes.push(
      ...new GPUMultipleTestingAdjustment({
        id: `${id}-adjustment`,
        pValues: testedPValues,
        method: this.method,
        minimumFamilySize: props.minimumFamilySize,
        output: {adjustedPValues, testedCount: output.testedCount}
      }).getCommandNodes(graph)
    );

    const wantsSummary = Boolean(output.significantCount || output.criticalPValue);
    const summary = wantsSummary
      ? createTransientView(graph, `${id}-summary`, 'uint32', 2)
      : undefined;
    if (summary) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-summary-clear`,
          operation: OPERATION,
          view: summary,
          type: 'u32',
          value: '0u'
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-significant`,
        operation: OPERATION,
        variant: summary ? 'significant-summary' : 'significant',
        bindings: [
          {name: 'adjusted', view: adjustedPValues, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'significant', view: output.significant, type: 'u32', access: 'read_write'},
          ...(summary
            ? [
                {name: 'tested', view: testedPValues, type: 'f32', access: 'read'} as const,
                {name: 'summary', view: summary, type: 'atomic<u32>', access: 'read_write'} as const
              ]
            : [])
        ],
        invocationCount: cellCount,
        // NaN compares false, so an untested cell is never significant.
        body: `let isSignificant = adjusted[adjustedOffset + index] <= parameters[parametersOffset];
significant[significantOffset + index] = select(0u, 1u, isSignificant);
${
  summary
    ? `if (isSignificant) {
  atomicAdd(&summary[summaryOffset], 1u);
  // Non-negative f32 bit patterns order like u32.
  atomicMax(&summary[summaryOffset + 1u], bitcast<u32>(tested[testedOffset + index]));
}`
    : ''
}`
      })
    );

    if (summary) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-summary`,
          operation: OPERATION,
          variant: 'summary',
          bindings: [
            {name: 'summary', view: summary, type: 'u32', access: 'read'},
            ...(output.significantCount
              ? [
                  {
                    name: 'significantCount',
                    view: output.significantCount,
                    type: 'u32',
                    access: 'read_write'
                  } as const
                ]
              : []),
            ...(output.criticalPValue
              ? [
                  {
                    name: 'criticalPValue',
                    view: output.criticalPValue,
                    type: 'f32',
                    access: 'read_write'
                  } as const
                ]
              : [])
          ],
          invocationCount: 1,
          declarations: GET_NAN_WGSL,
          body: `${output.significantCount ? 'significantCount[significantCountOffset] = summary[summaryOffset];' : ''}
${
  output.criticalPValue
    ? `criticalPValue[criticalPValueOffset] = select(getNaN(), bitcast<f32>(summary[summaryOffset + 1u]), summary[summaryOffset] != 0u);`
    : ''
}`
        })
      );
    }

    if (output.histogram) {
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterHistogram({
            id: `${id}-histogram`,
            input: {
              id: `${id}-tested-p`,
              format: 'float32',
              storage: {kind: 'buffer', values: testedPValues}
            },
            output: output.histogram!,
            domain: [0, 1]
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }
}

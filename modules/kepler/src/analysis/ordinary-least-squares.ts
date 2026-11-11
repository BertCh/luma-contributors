// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addKeplerLinearKernel as addKernel} from './analysis-scaling';
import {
  createTransientView,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUOrdinaryLeastSquares,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  getGPUOrdinaryLeastSquaresParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  MAXIMUM_KEPLER_OLS_SOURCE_ROWS,
  parseKeplerOLSSettings,
  type KeplerOLSSettings
} from '../recipes/analysis-recipe';
import type {KeplerAnalysisProvenance} from './analysis-bindings';

export type KeplerOLSOutputs = {
  residuals?: GraphDataView<'float32'>;
  fitted?: GraphDataView<'float32'>;
  coefficients?: GraphDataView<'float32'>;
  standardErrors?: GraphDataView<'float32'>;
  tStatistics?: GraphDataView<'float32'>;
  summary?: GraphDataView<'float32'>;
  status?: {code?: GraphDataView<'uint32'>};
};

export type KeplerOLSProps = KeplerAnalysisProvenance & {
  id?: string;
  settings: KeplerOLSSettings;
  response: GraphDataView<'float32'>;
  /** Same ordered roles as settings.predictorColumns, in original source row order. */
  predictors: readonly GraphDataView<'float32'>[];
  /** Caller combines ALL input Arrow null masks and the declared population before binding. */
  mask?: GraphDataView<'uint32'>;
  outputs?: KeplerOLSOutputs;
};

/**
 * Adds a complete-case intercept model using the existing Float32 OLS contributor. Predictor
 * columns are explicitly interleaved on the GPU, one pass per predictor with three bindings.
 * Finite response/predictor checks intersect the caller's null/population mask. Selection refits
 * the model; fitted values and residuals remain NaN outside that fit population, not held-out
 * predictions. Original input columns are borrowed and their chunks are never modified.
 *
 * Read status.code before publication: 1 is singular/ill-conditioned, 2 is too few complete rows
 * (n <= predictorCount + 1). A precision warning is separate from failure. Constant predictors
 * remain singular with ridge; a constant response can have a valid fit but undefined R squared.
 * Standard errors use RSS/(n-p), likelihood uses RSS/n, and the intercept is never penalized.
 * Ridge standard errors and t statistics ignore shrinkage bias; the penalized result is described
 * without classical significance claims. No robust/spatial covariance or causal inference is added.
 *
 * Keep the returned parameter alive through GPU completion, then call destroyParameters().
 * Caller owns supplied outputs; omitted outputs and packing scratch belong to the compiled graph.
 */
export function addKeplerOLS<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerOLSProps
) {
  const settings = parseKeplerOLSSettings(props.settings);
  const id = props.id ?? 'kepler-ols';
  const rowCount = props.response.length;
  const predictorCount = settings.predictorColumns.length;
  const coefficientCount = predictorCount + 1;
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1 ||
    rowCount < 1 ||
    rowCount > MAXIMUM_KEPLER_OLS_SOURCE_ROWS ||
    props.predictors.length !== predictorCount ||
    (props.population !== 'all' && props.population !== 'selected') ||
    (props.population === 'selected'
      ? !props.mask ||
        !Number.isSafeInteger(props.selectionRevision) ||
        (props.selectionRevision ?? 0) < 1
      : props.selectionRevision !== undefined)
  )
    throw new Error('OLS requires a current bounded source and explicit population');
  const inputs = [props.response, ...props.predictors, ...(props.mask ? [props.mask] : [])];
  for (const view of inputs) {
    validatePackedView(view, view === props.mask ? ['uint32'] : ['float32'], id);
    if (view.length !== rowCount || view.buffer.graph !== graph)
      throw new Error('OLS inputs must match source rows and graph');
  }
  const provided = props.outputs ?? {};
  const specifications = [
    [provided.residuals, 'float32', rowCount],
    [provided.fitted, 'float32', rowCount],
    [provided.coefficients, 'float32', coefficientCount],
    [provided.standardErrors, 'float32', coefficientCount],
    [provided.tStatistics, 'float32', coefficientCount],
    [provided.summary, 'float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH],
    [provided.status?.code, 'uint32', 1]
  ] as const;
  const suppliedBuffers = new Set(inputs.map(view => view.buffer));
  for (const [view, format, length] of specifications) {
    if (!view) continue;
    validatePackedView(view, [format], id);
    if (view.length !== length || view.buffer.graph !== graph || suppliedBuffers.has(view.buffer))
      throw new Error('OLS outputs require distinct matching graph buffers');
    suppliedBuffers.add(view.buffer);
  }
  const residuals =
    provided.residuals ?? createTransientView(graph, `${id}-residuals-output`, 'float32', rowCount);
  const fitted =
    provided.fitted ?? createTransientView(graph, `${id}-fitted-output`, 'float32', rowCount);
  const coefficients =
    provided.coefficients ??
    createTransientView(graph, `${id}-coefficients-output`, 'float32', coefficientCount);
  const standardErrors =
    provided.standardErrors ??
    createTransientView(graph, `${id}-standard-errors-output`, 'float32', coefficientCount);
  const tStatistics =
    provided.tStatistics ??
    createTransientView(graph, `${id}-t-statistics-output`, 'float32', coefficientCount);
  const summary =
    provided.summary ??
    createTransientView(
      graph,
      `${id}-summary-output`,
      'float32',
      GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
    );
  const status = {
    code: provided.status?.code ?? createTransientView(graph, `${id}-status-output`, 'uint32', 1)
  };
  const effectiveMask = createTransientView(graph, `${id}-effective-mask`, 'uint32', rowCount);
  const design = createTransientView(graph, `${id}-design`, 'float32', rowCount * predictorCount);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: `${id}-parameters`,
    format: 'float32',
    length: 1,
    values: getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: settings.ridgeLambda})
  });
  try {
    const outputs = {residuals, fitted, coefficients, standardErrors, tStatistics, summary, status};
    const contributor = new GPUOrdinaryLeastSquares({
      id,
      response: props.response,
      predictors: design,
      predictorCount,
      mask: effectiveMask,
      parameters: parameters.importToGraph(graph),
      tileRowCount: 64,
      output: outputs
    });
    addKernel(
      graph,
      `${id}-response-validity`,
      rowCount,
      [
        {name: 'response', view: props.response},
        ...(props.mask ? [{name: 'sourceMask', view: props.mask}] : []),
        {name: 'effectiveMask', view: effectiveMask, write: true}
      ],
      `let valid = isFiniteValue(response[responseOffset + index])${props.mask ? ' && sourceMask[sourceMaskOffset + index] != 0u' : ''};
effectiveMask[effectiveMaskOffset + index] = select(0u, 1u, valid);`
    );
    for (const [column, predictor] of props.predictors.entries()) {
      addKernel(
        graph,
        `${id}-predictor-${column}`,
        rowCount,
        [
          {name: 'predictor', view: predictor},
          {name: 'design', view: design, write: true},
          {name: 'effectiveMask', view: effectiveMask, write: true}
        ],
        `let value = predictor[predictorOffset + index];
design[designOffset + index * ${predictorCount}u + ${column}u] = value;
effectiveMask[effectiveMaskOffset + index] &= select(0u, 1u, isFiniteValue(value));`
      );
    }
    graph.add(contributor);
    return {
      operation: 'ols' as const,
      source: Object.freeze({...props.source}),
      population: props.population,
      selectionRevision: props.selectionRevision,
      settings,
      numericalPolicy: 'float32' as const,
      inference:
        settings.ridgeLambda === 0
          ? ('classical-homoskedastic' as const)
          : ('penalized-descriptive' as const),
      outputs,
      status,
      effectiveMask,
      parameterByteLength: parameters.buffer.byteLength,
      destroyParameters() {
        parameters.destroy();
      }
    };
  } catch (error) {
    parameters.destroy();
    throw error;
  }
}

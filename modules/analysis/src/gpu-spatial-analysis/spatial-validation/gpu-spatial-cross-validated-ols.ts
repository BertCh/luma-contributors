// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getOrdinaryLeastSquaresAllocationLengths} from '../spatial-regression/gpu-ordinary-least-squares';

import {validateSpatialCrossValidationBindings} from './validation-binding-plans';

import {validateRegressionAllocations} from '../spatial-regression/spatial-likelihood-allocations';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRegressionOutputDispatches} from '../spatial-regression/regression-output-dispatches';
import {validateGraphOutputsDisjointFromInputs} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getSpatialDesignInputs} from './spatial-validation-design';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPUOrdinaryLeastSquares} from '../spatial-regression/gpu-ordinary-least-squares';
import {
  GPUSpatialCrossValidation,
  type GPUSpatialCrossValidationProps
} from './gpu-spatial-cross-validation';

export type GPUSpatialCrossValidatedOLSProps = {
  id?: string;
  folds: Omit<GPUSpatialCrossValidationProps, 'id'>;
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  response: GraphDataView<'float32'>;
  output: {
    /** Fold-major coefficients, intercept first, fitted exclusively to training rows. */
    coefficients: GraphDataView<'float32'>;
    /** Fold-major predictions for every row, including test rows; unknown when fit/predictors invalid. */
    predictions: GraphDataView<'float32'>;
    fitStatus: GraphDataView<'uint32'>;
    /** Per fold: scored rows, missing test rows, MSE, MAE, mean prediction-minus-observation, R². */
    metrics: GraphDataView<'float32'>;
    /** Per fold bitmask: 1 no scored rows, 2 missing test outcomes/predictions, 4 undefined R², 8 nonfinite metric arithmetic. */
    metricStatus: GraphDataView<'uint32'>;
  };
};

/**
 * Repeated resident OLS fits over spatially buffered block folds. Every OLS fit estimates its own
 * training-only centering, coefficients and diagnostics; held-out outcomes never enter a fit.
 * Held-out scores report missing coverage explicitly. This selected evaluation recipe does not
 * perform whole-data preprocessing or bandwidth/variogram selection. Those models require their
 * own nested training-only fit graph; supplying one whole-data fit is not cross-validation.
 */
export class GPUSpatialCrossValidatedOLS implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUSpatialCrossValidatedOLSProps) {
    this.id = props.id ?? 'spatial-cross-validated-ols';
    const rows = props.response.length;
    const folds = props.folds.foldCount;
    new GPUSpatialCrossValidation(props.folds);
    validateCapacity(this.id, 'predictorCount', props.predictorCount, 1, 15);
    if (props.folds.positions.length !== rows) throw new Error(`${this.id}: row mismatch`);
    for (const [view, format, length] of [
      [props.predictors, 'float32', rows * props.predictorCount],
      [props.response, 'float32', rows],
      [props.output.coefficients, 'float32', folds * (props.predictorCount + 1)],
      [props.output.predictions, 'float32', folds * rows],
      [props.output.fitStatus, 'uint32', folds],
      [props.output.metrics, 'float32', folds * 6],
      [props.output.metricStatus, 'uint32', folds]
    ] as const) {
      validateSinglePackedView(view, [format], this.id);
      if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
    }
    validateRegressionOutputDispatches(
      this.id,
      {
        ...props.output,
        ...Object.fromEntries(
          Object.entries(props.folds.output).map(([name, view]) => [`fold ${name}`, view])
        ),
        diagnostics: props.folds.design?.diagnostics
      },
      [
        ['coefficients', 'fitStatus', 'predictions'],
        {read: ['fold training'], write: ['coefficients', 'fitStatus', 'predictions']},
        {read: ['predictions', 'fold testing'], write: ['metrics', 'metricStatus']}
      ]
    );
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [
        ...Object.values(props.output),
        ...Object.values(props.folds.output),
        ...(props.folds.design?.diagnostics ? [props.folds.design.diagnostics] : [])
      ],
      [
        props.predictors,
        props.response,
        props.folds.positions,
        ...(!props.folds.design || props.folds.design.mode === 'groups'
          ? [props.folds.blocks]
          : []),
        ...(props.folds.mask ? [props.folds.mask] : []),
        ...getSpatialDesignInputs(props.folds.design)
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateSpatialCrossValidationBindings(graph, id, props.folds);
    validateGraphChunkBindingRanges(graph, id, {
      predictors: props.predictors,
      response: props.response,
      ...props.output
    });
    validateRegressionAllocations(graph.device, id, {
      ...getOrdinaryLeastSquaresAllocationLengths(props.response.length, props.predictorCount),
      'fold OLS output scratch': Math.max(props.predictorCount + 1, 22)
    });
    const rows = props.response.length;
    const columns = props.predictorCount;
    const folds = props.folds.foldCount;
    const slice = <Format extends 'float32' | 'uint32'>(
      view: GraphDataView<Format>,
      offset: number,
      length: number
    ) =>
      graph.createDataView(view.buffer, {
        format: view.format,
        length,
        byteOffset: view.byteOffset + offset * 4
      });
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      ...Object.values(props.output)
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUSpatialCrossValidation({...props.folds, id: `${id}-folds`}).getCommandNodes(graph)
    ];
    for (let fold = 0; fold < folds; fold++) {
      const coefficients = slice(props.output.coefficients, fold * (columns + 1), columns + 1);
      const status = slice(props.output.fitStatus, fold, 1);
      const mask = slice(props.folds.output.training, fold * rows, rows);
      nodes.push(
        ...new GPUOrdinaryLeastSquares({
          id: `${id}-fit-${fold}`,
          predictors: props.predictors,
          response: props.response,
          predictorCount: columns,
          mask,
          output: {
            coefficients,
            predictions: slice(props.output.predictions, fold * rows, rows),
            standardErrors: createTransientView(
              graph,
              `${id}-errors-${fold}`,
              'float32',
              columns + 1
            ),
            tStatistics: createTransientView(
              graph,
              `${id}-statistics-${fold}`,
              'float32',
              columns + 1
            ),
            summary: createTransientView(graph, `${id}-summary-${fold}`, 'float32', 22),
            status: {code: status}
          }
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-metrics`,
        operation: 'GPUSpatialCrossValidatedOLS',
        variant: 'held-out-metrics',
        invocationCount: folds,
        bindings: [
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'predictions', view: props.output.predictions, type: 'f32', access: 'read'},
          {name: 'testing', view: props.folds.output.testing, type: 'u32', access: 'read'},
          {name: 'metrics', view: props.output.metrics, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.metricStatus, type: 'u32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `var count=0u;var missing=0u;var squares=0.0;var absolute=0.0;var bias=0.0;var mean=0.0;var totalSquares=0.0;
for(var row=0u;row<${rows}u;row++){if(testing[testingOffset+index*${rows}u+row]==0u){continue;}let observed=response[responseOffset+row];let predicted=predictions[predictionsOffset+index*${rows}u+row];if(!isFiniteBits(observed)||!isFiniteBits(predicted)){missing++;continue;}count++;let error=predicted-observed;squares+=error*error;absolute+=abs(error);bias+=error;let difference=observed-mean;mean+=difference/f32(count);totalSquares+=difference*(observed-mean);}
let coefficientOfDetermination=1.0-squares/totalSquares;
let numericFailure=!isFiniteBits(squares)||!isFiniteBits(absolute)||!isFiniteBits(bias)||!isFiniteBits(mean)||!isFiniteBits(totalSquares)||(totalSquares>0.0&&!isFiniteBits(coefficientOfDetermination));
let base=metricsOffset+index*6u;metrics[base]=f32(count);metrics[base+1u]=f32(missing);metrics[base+2u]=select(getNaN(),squares/f32(count),count>0u&&!numericFailure);metrics[base+3u]=select(getNaN(),absolute/f32(count),count>0u&&!numericFailure);metrics[base+4u]=select(getNaN(),bias/f32(count),count>0u&&!numericFailure);metrics[base+5u]=select(getNaN(),coefficientOfDetermination,totalSquares>0.0&&!numericFailure);
status[statusOffset+index]=select(0u,1u,count==0u)|select(0u,2u,missing>0u)|select(0u,4u,!(totalSquares>0.0))|select(0u,8u,numericFailure);`
      })
    );
    return nodes;
  }
}

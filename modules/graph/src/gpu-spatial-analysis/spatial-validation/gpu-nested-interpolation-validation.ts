// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validatePackedView,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {validatePredictionViews} from '../../utils/gpu-prediction-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {
  GPULocalKriging,
  type GPUPredictionCovariance
} from '../spatial-interpolation/gpu-local-kriging';
import {GPULocalRBF, type GPULocalRBFProps} from '../spatial-interpolation/gpu-local-rbf';
import {GPUSpatialCrossValidation} from './gpu-spatial-cross-validation';

export type GPUNestedInterpolationCandidate =
  | {mode: 'ordinary-kriging'; covariance: GPUPredictionCovariance}
  | {
      mode: 'rbf';
      kernel?: GPULocalRBFProps['kernel'];
      degree?: GPULocalRBFProps['degree'];
      epsilon: number;
      smoothing?: number;
    };
export type GPUNestedInterpolationValidationProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  response: GraphDataView<'float32'>;
  /** Explicit outer fold masks, fold-major. Overlap is rejected by GPU status, never silently fitted. */
  training: GraphDataView<'uint32'>;
  testing: GraphDataView<'uint32'>;
  foldCount: number;
  /** Inner block membership; only outer-training rows enter inner assignment and scoring. */
  innerBlocks: GraphDataView<'uint32'>;
  innerFoldCount?: number;
  innerBufferDistance?: number;
  seed?: number;
  neighborCount?: number;
  /** Fixed hyperparameter ladder; model/range/sill/nugget or RBF inverse bandwidth is selected using inner held-out MSE. */
  candidates: readonly GPUNestedInterpolationCandidate[];
  output: {
    /** Per outer fold: chosen candidate, or 0xffffffff when no candidate covers every tested inner row. */
    selected: GraphDataView<'uint32'>;
    /** Outer-fold-major candidate scores [MSE, scored rows, missing predictions]. */
    scores: GraphDataView<'float32'>;
    /** Outer-fold-major predictions; only outer testing rows are published. */
    predictions: GraphDataView<'float32'>;
    /** Per fold [scored count, missing count, MSE, MAE, signed bias, R²]. */
    metrics: GraphDataView<'float32'>;
    /** Bit1 invalid overlapping outer masks, bit2 no selected candidate, bit4 missing heldout predictions, bit8 empty heldout set, bit16 nonfinite heldout metric arithmetic. */
    status: GraphDataView<'uint32'>;
  };
};

/** Entirely resident nested hyperparameter search. Outer test outcomes never affect inner fits or selection. */
export class GPUNestedInterpolationValidation implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUNestedInterpolationValidationProps) {
    this.id = props.id ?? 'nested-interpolation-validation';
    const rows = props.response.length;
    validateCapacity(this.id, 'rows', rows, 3, 256);
    validateCapacity(this.id, 'foldCount', props.foldCount, 2, 16);
    validateCapacity(this.id, 'innerFoldCount', props.innerFoldCount ?? 3, 2, 8);
    validateCapacity(this.id, 'candidates', props.candidates.length, 1, 16);
    validateCapacity(this.id, 'neighborCount', props.neighborCount ?? 16, 2, 32);
    for (const [view, format, length] of [
      [props.positions, 'float32x2', rows],
      [props.response, 'float32', rows],
      [props.training, 'uint32', rows * props.foldCount],
      [props.testing, 'uint32', rows * props.foldCount],
      [props.innerBlocks, 'uint32', rows],
      [props.output.selected, 'uint32', props.foldCount],
      [props.output.scores, 'float32', props.foldCount * props.candidates.length * 3],
      [props.output.predictions, 'float32', rows * props.foldCount],
      [props.output.metrics, 'float32', 6 * props.foldCount],
      [props.output.status, 'uint32', props.foldCount]
    ] as const) {
      validatePackedView(view, [format], this.id);
      if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
    }
    validatePredictionViews(
      this.id,
      [props.positions, props.response, props.training, props.testing, props.innerBlocks],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this,
      rows = props.response.length,
      innerCount = props.innerFoldCount ?? 3,
      candidates = props.candidates.length;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.response,
      props.training,
      props.testing,
      props.innerBlocks,
      ...Object.values(props.output)
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const slice = <Format extends 'float32' | 'uint32'>(
      view: GraphDataView<Format>,
      offset: number,
      length: number
    ) =>
      graph.createDataView(view.buffer, {
        format: view.format,
        byteOffset: view.byteOffset + 4 * offset,
        length
      });
    const binding = (name: string, view: GraphDataView, write = false) => ({
      name,
      view,
      type: view.format.startsWith('uint32') ? ('u32' as const) : ('f32' as const),
      access: write ? ('read_write' as const) : ('read' as const)
    });
    const declarations = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`;
    for (let fold = 0; fold < props.foldCount; fold++) {
      const prefix = `${id}-${fold}`;
      const outerTraining = slice(props.training, fold * rows, rows),
        outerTesting = slice(props.testing, fold * rows, rows),
        status = slice(props.output.status, fold, 1);
      const validTraining = createTransientView(graph, `${prefix}-valid-training`, 'uint32', rows);
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${prefix}-check-outer`,
          operation: 'GPUNestedInterpolationValidation',
          invocationCount: 1,
          bindings: [
            binding('training', outerTraining),
            binding('testing', outerTesting),
            binding('status', status, true)
          ],
          body: `var overlap=false;for(var row=0u;row<${rows}u;row++){overlap=overlap||(training[trainingOffset+row]!=0u&&testing[testingOffset+row]!=0u);}status[statusOffset]=u32(overlap);`
        })
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${prefix}-complete-training`,
          operation: 'GPUNestedInterpolationValidation',
          invocationCount: rows,
          declarations,
          bindings: [
            binding('training', outerTraining),
            binding('response', props.response),
            binding('status', status),
            binding('mask', validTraining, true)
          ],
          body: 'mask[maskOffset+index]=u32(training[trainingOffset+index]!=0u&&status[statusOffset]==0u&&isFiniteBits(response[responseOffset+index]));'
        })
      );
      const inner = {
        folds: createTransientView(graph, `${prefix}-folds`, 'uint32', rows),
        training: createTransientView(
          graph,
          `${prefix}-inner-training`,
          'uint32',
          rows * innerCount
        ),
        testing: createTransientView(graph, `${prefix}-inner-testing`, 'uint32', rows * innerCount),
        counts: createTransientView(graph, `${prefix}-counts`, 'uint32', 3 * innerCount),
        status: createTransientView(graph, `${prefix}-inner-status`, 'uint32', innerCount)
      };
      nodes.push(
        ...new GPUSpatialCrossValidation({
          id: `${prefix}-inner`,
          positions: props.positions,
          blocks: props.innerBlocks,
          mask: validTraining,
          foldCount: innerCount,
          seed: props.seed,
          bufferDistance: props.innerBufferDistance,
          output: inner
        }).getCommandNodes(graph)
      );
      const outerPredictions = createTransientView(
        graph,
        `${prefix}-outer-predictions`,
        'float32',
        candidates * rows
      );
      const innerPredictions = createTransientView(
        graph,
        `${prefix}-inner-predictions`,
        'float32',
        candidates * innerCount * rows
      );
      for (let candidateIndex = 0; candidateIndex < candidates; candidateIndex++) {
        const candidate = props.candidates[candidateIndex];
        for (let split = 0; split <= innerCount; split++) {
          const stage = `${prefix}-${candidateIndex}-${split}`;
          const mask =
            split === innerCount ? validTraining : slice(inner.training, split * rows, rows);
          const values = createTransientView(
            graph,
            `${stage}-input`,
            candidate.mode === 'ordinary-kriging' ? 'float32x4' : 'float32',
            rows
          );
          const result = createTransientView(graph, `${stage}-result`, 'float32x4', rows),
            resultStatus = createTransientView(graph, `${stage}-status`, 'uint32', rows);
          nodes.push(
            createWGSLKernelNode(graph, {
              id: `${stage}-mask`,
              operation: 'GPUNestedInterpolationValidation',
              invocationCount: rows,
              declarations,
              bindings: [
                binding('mask', mask),
                binding('response', props.response),
                binding('values', values, true)
              ],
              body:
                candidate.mode === 'ordinary-kriging'
                  ? 'values[valuesOffset+index*4u]=response[responseOffset+index];values[valuesOffset+index*4u+1u]=0.0;values[valuesOffset+index*4u+2u]=0.0;values[valuesOffset+index*4u+3u]=f32(mask[maskOffset+index]);'
                  : 'values[valuesOffset+index]=select(getNaN(),response[responseOffset+index],mask[maskOffset+index]!=0u);'
            })
          );
          if (candidate.mode === 'ordinary-kriging') {
            const observations = graph.createDataView(values.buffer, {
              format: 'float32x4',
              length: rows
            });
            nodes.push(
              ...new GPULocalKriging({
                id: `${stage}-fit`,
                positions: props.positions,
                observations,
                supports: props.positions,
                mode: 'ordinary',
                covariance: candidate.covariance,
                neighborCount: props.neighborCount ?? 16,
                output: {values: result, status: resultStatus}
              }).getCommandNodes(graph)
            );
          } else {
            const response = graph.createDataView(values.buffer, {format: 'float32', length: rows});
            nodes.push(
              ...new GPULocalRBF({
                id: `${stage}-fit`,
                positions: props.positions,
                values: response,
                queries: props.positions,
                kernel: candidate.kernel,
                degree: candidate.degree,
                epsilon: candidate.epsilon,
                smoothing: candidate.smoothing,
                neighborCount: props.neighborCount ?? 16,
                output: {values: result, status: resultStatus}
              }).getCommandNodes(graph)
            );
          }
          const destination = split === innerCount ? outerPredictions : innerPredictions;
          const offset =
            split === innerCount
              ? candidateIndex * rows
              : (candidateIndex * innerCount + split) * rows;
          nodes.push(
            createWGSLKernelNode(graph, {
              id: `${stage}-collect`,
              operation: 'GPUNestedInterpolationValidation',
              invocationCount: rows,
              declarations: GET_NAN_WGSL,
              bindings: [
                binding('result', result),
                binding('status', resultStatus),
                binding('predictions', destination, true)
              ],
              body: `predictions[predictionsOffset+${offset}u+index]=select(getNaN(),result[resultOffset+index*4u],status[statusOffset+index]==0u);`
            })
          );
        }
      }
      const scores = slice(props.output.scores, fold * candidates * 3, candidates * 3),
        selected = slice(props.output.selected, fold, 1),
        predictions = slice(props.output.predictions, fold * rows, rows),
        metrics = slice(props.output.metrics, fold * 6, 6);
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${prefix}-select`,
          operation: 'GPUNestedInterpolationValidation',
          invocationCount: 1,
          declarations,
          bindings: [
            binding('response', props.response),
            binding('testing', inner.testing),
            binding('predictions', innerPredictions),
            binding('scores', scores, true),
            binding('selected', selected, true),
            binding('status', status, true)
          ],
          body: `var best=3.402823e38;var chosen=0xffffffffu;for(var candidate=0u;candidate<${candidates}u;candidate++){var squares=0.0;var count=0u;var missing=0u;for(var split=0u;split<${innerCount}u;split++){for(var row=0u;row<${rows}u;row++){if(testing[testingOffset+split*${rows}u+row]==0u){continue;}let predicted=predictions[predictionsOffset+(candidate*${innerCount}u+split)*${rows}u+row];let observed=response[responseOffset+row];if(!isFiniteBits(predicted)||!isFiniteBits(observed)){missing++;continue;}let error=predicted-observed;squares+=error*error;count++;}}let score=select(getNaN(),squares/f32(count),count>0u&&missing==0u);scores[scoresOffset+candidate*3u]=score;scores[scoresOffset+candidate*3u+1u]=f32(count);scores[scoresOffset+candidate*3u+2u]=f32(missing);if(isFiniteBits(score)&&score<best){best=score;chosen=candidate;}}selected[selectedOffset]=chosen;if(chosen==0xffffffffu){status[statusOffset]|=2u;}`
        })
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${prefix}-publish`,
          operation: 'GPUNestedInterpolationValidation',
          invocationCount: rows,
          declarations: GET_NAN_WGSL,
          bindings: [
            binding('selected', selected),
            binding('testing', outerTesting),
            binding('candidates', outerPredictions),
            binding('predictions', predictions, true)
          ],
          body: `let chosen=selected[selectedOffset];var value=getNaN();if(chosen<${candidates}u&&testing[testingOffset+index]!=0u){value=candidates[candidatesOffset+chosen*${rows}u+index];}predictions[predictionsOffset+index]=value;`
        })
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${prefix}-metrics`,
          operation: 'GPUNestedInterpolationValidation',
          invocationCount: 1,
          declarations,
          bindings: [
            binding('response', props.response),
            binding('testing', outerTesting),
            binding('predictions', predictions),
            binding('metrics', metrics, true),
            binding('status', status, true)
          ],
          body: `var count=0u;var missing=0u;var squares=0.0;var absolute=0.0;var bias=0.0;var mean=0.0;var total=0.0;for(var row=0u;row<${rows}u;row++){if(testing[testingOffset+row]==0u){continue;}let predicted=predictions[predictionsOffset+row];let observed=response[responseOffset+row];if(!isFiniteBits(predicted)||!isFiniteBits(observed)){missing++;continue;}count++;let error=predicted-observed;squares+=error*error;absolute+=abs(error);bias+=error;let delta=observed-mean;mean+=delta/f32(count);total+=delta*(observed-mean);}let coefficientOfDetermination=1.0-squares/total;let numericFailure=!isFiniteBits(squares)||!isFiniteBits(absolute)||!isFiniteBits(bias)||!isFiniteBits(mean)||!isFiniteBits(total)||(total>0.0&&!isFiniteBits(coefficientOfDetermination));metrics[metricsOffset]=f32(count);metrics[metricsOffset+1u]=f32(missing);metrics[metricsOffset+2u]=select(getNaN(),squares/f32(count),count>0u&&!numericFailure);metrics[metricsOffset+3u]=select(getNaN(),absolute/f32(count),count>0u&&!numericFailure);metrics[metricsOffset+4u]=select(getNaN(),bias/f32(count),count>0u&&!numericFailure);metrics[metricsOffset+5u]=select(getNaN(),coefficientOfDetermination,total>0.0&&!numericFailure);status[statusOffset]|=select(0u,4u,missing>0u)|select(0u,8u,count+missing==0u)|select(0u,16u,numericFailure);`
        })
      );
    }
    return nodes;
  }
}

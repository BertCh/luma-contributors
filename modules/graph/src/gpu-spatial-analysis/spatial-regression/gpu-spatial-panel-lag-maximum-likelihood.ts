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
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL, IS_NAN_BITS_WGSL} from '../../utils/wgsl-common';
import {
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  validateGPUSpatialWeights,
  getGPUSpatialWeightsViews,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {GPU_UPPER_INCOMPLETE_GAMMA_WGSL} from '../distribution-dynamics/incomplete-gamma-wgsl';

export type GPUSpatialPanelLagMaximumLikelihoodProps = {
  id?: string;
  weights: GPUSpatialWeights;
  periodCount: number;
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  response: GraphDataView<'float32'>;
  effects?: 'random' | 'unit';
  maximumIterations?: number;
  maximumSearchIterations?: number;
  tolerance?: number;
  output: {
    /** Intercept, slopes, rho, phi. Unit effects absorb the intercept (NaN) and set phi=0. */
    coefficients: GraphDataView<'float32'>;
    /** Conditional expected-information covariance of intercept/slopes/rho, excluding phi and variance. */
    covariance: GraphDataView<'float32'>;
    /** sigma2, spreg-reported log likelihood, outer sweeps, phi change, rho bracket width, boundary bits (rho=1, phi=2). */
    summary: GraphDataView<'float32'>;
    /** 0 converged; 1 invalid/incomplete inputs; 2 singular; 3 iteration/search exhaustion. */
    status: GraphDataView<'uint32'>;
    /** Quasi-demeaned fitted response and innovations, matching spreg. */
    fitted: GraphDataView<'float32'>;
    residuals: GraphDataView<'float32'>;
  };
  /** Random-versus-unit comparison. Tests are [statistic, df, p]; df=predictorCount+2 follows spreg.k, rather than the smaller contrast rank. Status: 0 valid, 1 unsuccessful fit, 2 invalid covariance difference. */
  hausman?: {tests: GraphDataView<'float32'>; status: GraphDataView<'uint32'>};
};

/**
 * Balanced spatial lag ML, random or unit effects. Random mode alternates analytic phi
 * minimization with bounded exact-logdet rho profiling, matching spreg Panel_RE_Lag.
 * A dense entity determinant limits this reference-oriented mode to 32 entities, 32 periods,
 * 7 predictors. Complete panels and nonnegative row-standardized zero-diagonal weights are required.
 * No hidden regularization or CPU solve. Hausman uses spreg's RE-minus-FE conditional variance
 * difference; indefinite/singular or negative-statistic comparisons are explicitly invalid.
 */
export class GPUSpatialPanelLagMaximumLikelihood implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUSpatialPanelLagMaximumLikelihoodProps) {
    this.id = props.id ?? 'spatial-panel-lag-maximum-likelihood';
    const entities = validateGPUSpatialWeights(this.id, props.weights);
    const rows = entities * props.periodCount;
    const columns = props.predictorCount + 1;
    validateCapacity(this.id, 'entities', entities, 2, 32);
    validateCapacity(this.id, 'periodCount', props.periodCount, 2, 32);
    validateCapacity(this.id, 'predictorCount', props.predictorCount, 1, 7);
    validateCapacity(this.id, 'maximumIterations', props.maximumIterations ?? 64, 1, 128);
    validateCapacity(
      this.id,
      'maximumSearchIterations',
      props.maximumSearchIterations ?? 32,
      1,
      64
    );
    if (
      !['random', 'unit'].includes(props.effects ?? 'random') ||
      !(Number.isFinite(props.tolerance ?? 1e-5) && (props.tolerance ?? 1e-5) > 0) ||
      (props.hausman && props.effects === 'unit')
    )
      throw new Error(`${this.id}: invalid estimator settings`);
    for (const [view, format, length] of [
      [props.predictors, 'float32', rows * props.predictorCount],
      [props.response, 'float32', rows],
      [props.output.coefficients, 'float32', columns + 2],
      [props.output.covariance, 'float32', (columns + 1) ** 2],
      [props.output.summary, 'float32', 6],
      [props.output.status, 'uint32', 1],
      [props.output.fitted, 'float32', rows],
      [props.output.residuals, 'float32', rows],
      [props.hausman?.tests, 'float32', 3],
      [props.hausman?.status, 'uint32', 1]
    ] as const)
      if (view) {
        validatePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [...Object.values(props.output), ...Object.values(props.hausman ?? {})],
      [props.predictors, props.response, ...getGPUSpatialWeightsViews(props.weights)]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const entities = props.weights.offsets.length - 1,
      rows = props.response.length,
      columns = props.predictorCount + 1;
    const width = columns + 2;
    const covarianceOffset = width,
      summaryOffset = width + (columns + 1) ** 2,
      statusOffset = summaryOffset + 6,
      stride = statusOffset + 1;
    const data = createTransientView(graph, `${id}-packed-panel`, 'float32', rows * width * 2),
      dense = createTransientView(graph, `${id}-dense-weights`, 'float32', entities * entities),
      results = createTransientView(
        graph,
        `${id}-model-results`,
        'float32',
        stride * (props.hausman ? 2 : 1)
      ),
      spatialProduct = createTransientView(
        graph,
        `${id}-spatial-product`,
        'float32',
        entities * entities * (props.hausman ? 2 : 1)
      );
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      ...getGPUSpatialWeightsViews(props.weights),
      ...Object.values(props.output),
      ...Object.values(props.hausman ?? {})
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-dense-weights`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: 'validate-weights',
        invocationCount: entities,
        bindings: [
          {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: props.weights.weights, type: 'f32', access: 'read'},
          {name: 'dense', view: dense, type: 'f32', access: 'read_write'},
          ...(props.weights.status?.overflow
            ? [
                {
                  name: 'overflow',
                  view: props.weights.status.overflow,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `for (var column = 0u; column < ${entities}u; column++) { dense[denseOffset + index * ${entities}u + column] = 0.0; } var valid = ${props.weights.status?.overflow ? 'overflow[overflowOffset] == 0u' : 'true'}; var total = 0.0; let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u]; if (end < begin || end > ${props.weights.neighbors.length}u) { valid = false; } else { for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot]; if (neighbor >= ${entities}u || neighbor == index || !(weight >= 0.0) || !isFiniteBits(weight) || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { valid = false; } else { dense[denseOffset + index * ${entities}u + neighbor] = weight; total += weight; } } } if (!valid || abs(total-1.0) > 1.0e-5) { dense[denseOffset + index * ${entities}u] = getNaN(); }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-pack`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: 'pack-panel-means',
        invocationCount: rows,
        bindings: [
          {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'dense', view: dense, type: 'f32', access: 'read'},
          {name: 'data', view: data, type: 'f32', access: 'read_write'}
        ],
        body: `let entity = index % ${entities}u; let period = index / ${entities}u; let base = dataOffset + index * ${width * 2}u;
for (var column = 0u; column < ${columns}u; column++) { var value = 1.0; var mean = 1.0; if (column > 0u) { value = predictors[predictorsOffset + index * ${columns - 1}u + column - 1u]; mean = 0.0; for (var time = 0u; time < ${props.periodCount}u; time++) { mean += predictors[predictorsOffset + (time * ${entities}u + entity) * ${columns - 1}u + column - 1u] / ${props.periodCount}.0; } } data[base + column] = value; data[base + ${width}u + column] = mean; }
var meanResponse = 0.0; var lagResponse = 0.0; var meanLag = 0.0; for (var time = 0u; time < ${props.periodCount}u; time++) { meanResponse += response[responseOffset + time * ${entities}u + entity] / ${props.periodCount}.0; for (var neighbor = 0u; neighbor < ${entities}u; neighbor++) { meanLag += dense[denseOffset + entity * ${entities}u + neighbor] * response[responseOffset + time * ${entities}u + neighbor] / ${props.periodCount}.0; } } for (var neighbor = 0u; neighbor < ${entities}u; neighbor++) { lagResponse += dense[denseOffset + entity * ${entities}u + neighbor] * response[responseOffset + period * ${entities}u + neighbor]; }
data[base + ${columns}u] = response[responseOffset + index]; data[base + ${columns + 1}u] = lagResponse; data[base + ${width + columns}u] = meanResponse; data[base + ${width + columns + 1}u] = meanLag;`
      })
    ];
    const addFit = (unit: boolean, model: number) => {
      const predictors = unit ? columns - 1 : columns;
      const parameterCount = predictors + 1;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${unit ? 'unit' : 'random'}-fit`,
          operation: 'GPUSpatialPanelLagMaximumLikelihood',
          variant: unit ? 'unit-ml' : 'random-ml',
          invocationCount: 1,
          bindings: [
            {name: 'data', view: data, type: 'f32', access: 'read'},
            {name: 'dense', view: dense, type: 'f32', access: 'read'},
            {name: 'results', view: results, type: 'f32', access: 'read_write'},
            {name: 'spatialProduct', view: spatialProduct, type: 'f32', access: 'read_write'}
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${getCholeskyWGSL(predictors)}\n${getCholeskyWGSL(parameterCount)}\n${getInverseWGSL(entities)}
fn datum(row:u32,column:u32,phi:f32)->f32 {let base=dataOffset+row*${width * 2}u;return data[base+column]-(1.0-phi)*data[base+${width}u+column];}
fn objective(rho:f32,first:f32,cross:f32,last:f32)->f32 {
 let variance=(first-2.0*rho*cross+rho*rho*last)/${rows}.0; if (!(variance > 0.0)) {return 3.402823e38;}
 var matrix:array<f32,${entities * entities}>;for(var row=0u;row<${entities}u;row++){for(var column=0u;column<${entities}u;column++){matrix[row*${entities}u+column]=select(0.0,1.0,row==column)-rho*dense[denseOffset+row*${entities}u+column];}}
 var determinant=0.0;for(var pivot=0u;pivot<${entities}u;pivot++){var chosen=pivot;for(var row=pivot+1u;row<${entities}u;row++){if(abs(matrix[row*${entities}u+pivot])>abs(matrix[chosen*${entities}u+pivot])){chosen=row;}}
 if(!(abs(matrix[chosen*${entities}u+pivot])>1.0e-10)){return 3.402823e38;}if(chosen!=pivot){for(var column=pivot;column<${entities}u;column++){let saved=matrix[pivot*${entities}u+column];matrix[pivot*${entities}u+column]=matrix[chosen*${entities}u+column];matrix[chosen*${entities}u+column]=saved;}}
 let diagonal=matrix[pivot*${entities}u+pivot];determinant+=log(abs(diagonal));for(var row=pivot+1u;row<${entities}u;row++){let factor=matrix[row*${entities}u+pivot]/diagonal;for(var column=pivot+1u;column<${entities}u;column++){matrix[row*${entities}u+column]-=factor*matrix[pivot*${entities}u+column];}}}
 return ${getWGSLFloatLiteral(rows / 2)}*log(variance)-${props.periodCount}.0*determinant;
}`,
          body: `let base=resultsOffset+${model * stride}u;for(var slot=0u;slot<${stride}u;slot++){results[base+slot]=getNaN();}results[base+${statusOffset}u]=1.0;
for(var slot=0u;slot<${rows * width * 2}u;slot++){if(!isFiniteBits(data[dataOffset+slot])){return;}}for(var slot=0u;slot<${entities * entities}u;slot++){if(!isFiniteBits(dense[denseOffset+slot])){return;}}
var beta:array<f32,${predictors}>;var pooled:array<f32,${predictors * predictors}>;
for(var row=0u;row<${rows}u;row++){for(var column=0u;column<${predictors}u;column++){let value=datum(row,column+${unit ? 1 : 0}u,1.0);beta[column]+=value*datum(row,${columns}u,1.0);for(var other=0u;other<=column;other++){pooled[column*${predictors}u+other]+=value*datum(row,other+${unit ? 1 : 0}u,1.0);}}}
results[base+${statusOffset}u]=2.0;if(!choleskyFactor_${predictors}(&pooled)){return;}choleskySolve_${predictors}(&pooled,&beta);
var rho=0.1;var phi=${unit ? '0.0' : '0.1'};var state=3u;var completed=0u;var change=0.0;var bracketWidth=2.0;var first=0.0;var cross=0.0;var last=0.0;var gram:array<f32,${predictors * predictors}>;
for(var iteration=0u;iteration<${unit ? 1 : (props.maximumIterations ?? 64)}u;iteration++){
 ${unit ? '' : `let oldPhi=phi;var within=0.0;var between=0.0;for(var row=0u;row<${rows}u;row++){let rowBase=dataOffset+row*${width * 2}u;var residual=data[rowBase+${columns}u]-rho*data[rowBase+${columns + 1}u];var mean=data[rowBase+${width + columns}u]-rho*data[rowBase+${width + columns + 1}u];for(var column=0u;column<${predictors}u;column++){residual-=beta[column]*data[rowBase+column];mean-=beta[column]*data[rowBase+${width}u+column];}within+=(residual-mean)*(residual-mean);between+=mean*mean;}phi=1.0;if(between>0.0){phi=clamp(sqrt(within/(${props.periodCount - 1}.0*between)),1.0e-5,1.0);}change=abs(phi-oldPhi);`}
 var firstBeta:array<f32,${predictors}>;var lagBeta:array<f32,${predictors}>;for(var slot=0u;slot<${predictors * predictors}u;slot++){gram[slot]=0.0;}
 for(var row=0u;row<${rows}u;row++){for(var column=0u;column<${predictors}u;column++){let value=datum(row,column+${unit ? 1 : 0}u,phi);firstBeta[column]+=value*datum(row,${columns}u,phi);lagBeta[column]+=value*datum(row,${columns + 1}u,phi);for(var other=0u;other<=column;other++){gram[column*${predictors}u+other]+=value*datum(row,other+${unit ? 1 : 0}u,phi);}}}
 var factor=gram;if(!choleskyFactor_${predictors}(&factor)){state=2u;break;}choleskySolve_${predictors}(&factor,&firstBeta);choleskySolve_${predictors}(&factor,&lagBeta);first=0.0;cross=0.0;last=0.0;
 for(var row=0u;row<${rows}u;row++){var residual=datum(row,${columns}u,phi);var lagResidual=datum(row,${columns + 1}u,phi);for(var column=0u;column<${predictors}u;column++){let value=datum(row,column+${unit ? 1 : 0}u,phi);residual-=value*firstBeta[column];lagResidual-=value*lagBeta[column];}first+=residual*residual;cross+=residual*lagResidual;last+=lagResidual*lagResidual;}
 var lower=-0.999;var upper=0.999;var left=lower+0.38196601125*(upper-lower);var right=lower+0.61803398875*(upper-lower);var leftScore=objective(left,first,cross,last);var rightScore=objective(right,first,cross,last);
 for(var step=0u;step<${props.maximumSearchIterations ?? 32}u;step++){if(leftScore<=rightScore){upper=right;right=left;rightScore=leftScore;left=lower+0.38196601125*(upper-lower);leftScore=objective(left,first,cross,last);}else{lower=left;left=right;leftScore=rightScore;right=lower+0.61803398875*(upper-lower);rightScore=objective(right,first,cross,last);}}
 rho=select(right,left,leftScore<=rightScore);bracketWidth=upper-lower;for(var column=0u;column<${predictors}u;column++){beta[column]=firstBeta[column]-rho*lagBeta[column];}completed=iteration+1u;
 if(${unit ? 'true' : `change <= ${getWGSLFloatLiteral(props.tolerance ?? 1e-5)}`}){state=0u;break;}
}
if((state==0u||state==3u)&&bracketWidth>${getWGSLFloatLiteral(props.tolerance ?? 1e-5)}){state=3u;}
for(var column=0u;column<${predictors}u;column++){results[base+column+${unit ? 1 : 0}u]=beta[column];}results[base+${columns}u]=rho;results[base+${columns + 1}u]=phi;
let variance=(first-2.0*rho*cross+rho*rho*last)/${rows}.0;if(!(variance>0.0)||!isFiniteBits(variance)){state=2u;}results[base+${summaryOffset}u]=variance;results[base+${summaryOffset + 1}u]=-objective(rho,first,cross,last)-${getWGSLFloatLiteral(rows / 2)}*(1.0+1.8378770664093453)${unit ? `-${getWGSLFloatLiteral(rows / 2)}*log(${getWGSLFloatLiteral(rows)})` : ''};results[base+${summaryOffset + 2}u]=f32(completed);results[base+${summaryOffset + 3}u]=change;results[base+${summaryOffset + 4}u]=bracketWidth;results[base+${summaryOffset + 5}u]=select(0.0,1.0,abs(rho)>.9989)+select(0.0,2.0,(phi>=.99999||phi<=1.00001e-5) && ${unit ? 'false' : 'true'});results[base+${statusOffset}u]=f32(state);if(state!=0u||!(variance>0.0)){return;}
var spatialInverse:array<f32,${entities * entities}>;for(var row=0u;row<${entities}u;row++){for(var column=0u;column<${entities}u;column++){spatialInverse[row*${entities}u+column]=select(0.0,1.0,row==column)-rho*dense[denseOffset+row*${entities}u+column];}}
if(!inverse_${entities}(&spatialInverse)){results[base+${statusOffset}u]=2.0;return;}
for(var row=0u;row<${entities}u;row++){for(var column=0u;column<${entities}u;column++){spatialProduct[spatialProductOffset+${model * entities * entities}u+row*${entities}u+column]=0.0;for(var other=0u;other<${entities}u;other++){spatialProduct[spatialProductOffset+${model * entities * entities}u+row*${entities}u+column]+=dense[denseOffset+row*${entities}u+other]*spatialInverse[other*${entities}u+column];}}}
var traceSquare=0.0;var traceProduct=0.0;for(var row=0u;row<${entities}u;row++){for(var column=0u;column<${entities}u;column++){let value=spatialProduct[spatialProductOffset+${model * entities * entities}u+row*${entities}u+column];traceSquare+=value*spatialProduct[spatialProductOffset+${model * entities * entities}u+column*${entities}u+row];traceProduct+=value*value;}}
var information:array<f32,${parameterCount * parameterCount}>;for(var column=0u;column<${predictors}u;column++){for(var other=0u;other<=column;other++){information[column*${parameterCount}u+other]=gram[column*${predictors}u+other]/variance;}}
var predictionSquares=0.0;for(var row=0u;row<${rows}u;row++){let entity=row%${entities}u;let periodStart=row-entity;var lagPrediction=0.0;for(var neighbor=0u;neighbor<${entities}u;neighbor++){var prediction=0.0;for(var column=0u;column<${predictors}u;column++){prediction+=datum(periodStart+neighbor,column+${unit ? 1 : 0}u,phi)*beta[column];}lagPrediction+=spatialProduct[spatialProductOffset+${model * entities * entities}u+entity*${entities}u+neighbor]*prediction;}predictionSquares+=lagPrediction*lagPrediction;for(var column=0u;column<${predictors}u;column++){information[${predictors * parameterCount}u+column]+=datum(row,column+${unit ? 1 : 0}u,phi)*lagPrediction/variance;}}
information[${parameterCount * parameterCount - 1}u]=${props.periodCount}.0*(traceSquare+traceProduct)+predictionSquares/variance;if(!choleskyFactor_${parameterCount}(&information)){results[base+${statusOffset}u]=2.0;return;}
for(var column=0u;column<${parameterCount}u;column++){var inverseColumn:array<f32,${parameterCount}>;inverseColumn[column]=1.0;choleskySolve_${parameterCount}(&information,&inverseColumn);for(var row=0u;row<${parameterCount}u;row++){results[base+${covarianceOffset}u+(row+${unit ? 1 : 0}u)*${columns + 1}u+column+${unit ? 1 : 0}u]=inverseColumn[row];}}`
        })
      );
    };
    addFit(props.effects === 'unit', 0);
    if (props.hausman) addFit(true, 1);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: 'publish',
        invocationCount: 1,
        bindings: [
          {name: 'results', view: results, type: 'f32', access: 'read'},
          {
            name: 'coefficients',
            view: props.output.coefficients,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'covariance', view: props.output.covariance, type: 'f32', access: 'read_write'},
          {name: 'summary', view: props.output.summary, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `for(var slot=0u;slot<${width}u;slot++){coefficients[coefficientsOffset+slot]=results[resultsOffset+slot];}for(var slot=0u;slot<${(columns + 1) ** 2}u;slot++){covariance[covarianceOffset+slot]=results[resultsOffset+${covarianceOffset}u+slot];}for(var slot=0u;slot<6u;slot++){summary[summaryOffset+slot]=results[resultsOffset+${summaryOffset}u+slot];}status[statusOffset]=u32(results[resultsOffset+${statusOffset}u]);`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-residuals`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: 'quasi-demeaned-predictions',
        invocationCount: rows,
        bindings: [
          {name: 'data', view: data, type: 'f32', access: 'read'},
          {name: 'results', view: results, type: 'f32', access: 'read'},
          {name: 'fitted', view: props.output.fitted, type: 'f32', access: 'read_write'},
          {name: 'residuals', view: props.output.residuals, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: `fitted[fittedOffset+index]=getNaN();residuals[residualsOffset+index]=getNaN();let state=u32(results[resultsOffset+${statusOffset}u]);if(state!=0u&&state!=3u){return;}let base=dataOffset+index*${width * 2}u;let phi=results[resultsOffset+${columns + 1}u];var predicted=results[resultsOffset+${columns}u]*(data[base+${columns + 1}u]-(1.0-phi)*data[base+${width + columns + 1}u]);for(var column=${props.effects === 'unit' ? 1 : 0}u;column<${columns}u;column++){predicted+=results[resultsOffset+column]*(data[base+column]-(1.0-phi)*data[base+${width}u+column]);}fitted[fittedOffset+index]=predicted;residuals[residualsOffset+index]=data[base+${columns}u]-(1.0-phi)*data[base+${width + columns}u]-predicted;`
      })
    );
    if (props.hausman)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-hausman`,
          operation: 'GPUSpatialPanelLagMaximumLikelihood',
          variant: 'hausman',
          invocationCount: 1,
          bindings: [
            {name: 'results', view: results, type: 'f32', access: 'read'},
            {name: 'tests', view: props.hausman.tests, type: 'f32', access: 'read_write'},
            {name: 'status', view: props.hausman.status, type: 'u32', access: 'read_write'}
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${IS_NAN_BITS_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}\n${getInverseWGSL(columns)}\n${getCholeskyWGSL(columns)}`,
          body: `status[statusOffset]=1u;for(var slot=0u;slot<3u;slot++){tests[testsOffset+slot]=getNaN();}if(results[resultsOffset+${statusOffset}u]!=0.0||results[resultsOffset+${stride + statusOffset}u]!=0.0){return;}var difference:array<f32,${columns * columns}>;var contrast:array<f32,${columns}>;for(var row=0u;row<${columns}u;row++){contrast[row]=results[resultsOffset+${stride + 1}u+row]-results[resultsOffset+1u+row];for(var column=0u;column<${columns}u;column++){let offset=${covarianceOffset}u+(row+1u)*${columns + 1}u+column+1u;difference[row*${columns}u+column]=results[resultsOffset+offset]-results[resultsOffset+${stride}u+offset];}}status[statusOffset]=2u;var factor=difference;if(!choleskyFactor_${columns}(&factor)){return;}if(!inverse_${columns}(&difference)){return;}var statistic=0.0;for(var row=0u;row<${columns}u;row++){for(var column=0u;column<${columns}u;column++){statistic+=contrast[row]*difference[row*${columns}u+column]*contrast[column];}}tests[testsOffset]=statistic;tests[testsOffset+1u]=${columns + 1}.0;if(!(statistic>=0.0)||!isFiniteBits(statistic)){return;}tests[testsOffset+2u]=getChiSquareSurvivalValue(statistic,${columns + 1}.0);status[statusOffset]=0u;`
        })
      );
    return nodes;
  }
}
function getInverseWGSL(size: number): string {
  return `fn inverse_${size}(matrix:ptr<function,array<f32,${size * size}>>)->bool {var inverse:array<f32,${size * size}>;for(var row=0u;row<${size}u;row++){inverse[row*${size}u+row]=1.0;}for(var pivot=0u;pivot<${size}u;pivot++){var chosen=pivot;for(var row=pivot+1u;row<${size}u;row++){if(abs((*matrix)[row*${size}u+pivot])>abs((*matrix)[chosen*${size}u+pivot])){chosen=row;}}if(!(abs((*matrix)[chosen*${size}u+pivot])>1.0e-10)){return false;}if(chosen!=pivot){for(var column=0u;column<${size}u;column++){let first=(*matrix)[pivot*${size}u+column];(*matrix)[pivot*${size}u+column]=(*matrix)[chosen*${size}u+column];(*matrix)[chosen*${size}u+column]=first;let second=inverse[pivot*${size}u+column];inverse[pivot*${size}u+column]=inverse[chosen*${size}u+column];inverse[chosen*${size}u+column]=second;}}let diagonal=(*matrix)[pivot*${size}u+pivot];for(var column=0u;column<${size}u;column++){(*matrix)[pivot*${size}u+column]/=diagonal;inverse[pivot*${size}u+column]/=diagonal;}for(var row=0u;row<${size}u;row++){if(row==pivot){continue;}let factor=(*matrix)[row*${size}u+pivot];for(var column=0u;column<${size}u;column++){(*matrix)[row*${size}u+column]-=factor*(*matrix)[pivot*${size}u+column];inverse[row*${size}u+column]-=factor*inverse[pivot*${size}u+column];}}}(*matrix)=inverse;return true;}`;
}

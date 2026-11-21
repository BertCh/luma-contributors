import {
  getPanelPreciseDiagnosticNodes,
  PANEL_DIAGNOSTIC_SUM_COUNT
} from './panel-precise-diagnostics';
// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  ERFC_VALUE_WGSL,
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  validateGraphViewsBelongToGraph,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUOrdinaryLeastSquares} from './gpu-ordinary-least-squares';
import type {GPUSpatialPanelErrorMaximumLikelihood} from './gpu-spatial-panel-error-maximum-likelihood';
import {getRegressionWeightsLossNodes} from './regression-weights-completeness';
import {getPanelDiagnosticZeroFitNodes} from './panel-diagnostic-zero-fit-nodes';
import {getPanelWithinNodes, getPanelWithinScratchLengths} from './panel-within-nodes';
import {getRegressionTileLayout, getRowTileSumNodes} from './regression-row-tiles';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';

function binding(
  name: string,
  view: GraphDataView<'float32'> | GraphDataView<'uint32'>,
  access: 'read' | 'read_write' = 'read',
  type: 'f32' | 'u32' = 'f32'
): WGSLKernelBinding {
  return {name, view, access, type};
}
/**
 * Pooled complete-panel diagnostic definitions in spreg 1.8.3. The variance is the OLS summary
 * `RSS / (n - p)` (spreg `ols.sig2`), not the `RSS / n` of the cross-section LM tests.
 *
 * @internal
 */
export function getPanelDiagnosticNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUSpatialPanelErrorMaximumLikelihood,
  completeness?: GraphDataView<'uint32'>
): readonly GPUCommandNode<Parameters>[] {
  const {id, props} = contributor;
  const guard = completeness
    ? {nodes: [], loss: completeness}
    : getRegressionWeightsLossNodes(graph, `${id}-diagnostic`, props.weights);
  const diagnostics = props.diagnostics!;
  const rows = props.response.length;
  const entities = props.weights.offsets.length - 1;
  const columns = props.predictorCount + 1;
  const edges = props.weights.neighbors.length;
  validateGraphViewsBelongToGraph(id, graph, Object.values(diagnostics));
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'diagnostic lags',
    (rows * 4 + 1) * 4,
    'use fewer panel rows for pooled diagnostics'
  );
  const view = (name: string, length: number) =>
    createTransientView(graph, `${id}-diagnostic-${name}`, 'float32', length);
  const residuals = view('residuals', rows),
    fitted = view('fitted', rows),
    lags = view('lags', rows * 4 + 1),
    laggedFit = view('lagged-fit', rows),
    projectedResiduals = view('projected-residuals', rows),
    summary = view('summary', 22);
  const centeredPredictors = view('centered-predictors', props.predictors.length);
  const centeredResponse = view('centered-response', rows);
  const meanLength = getPanelWithinScratchLengths(1, rows, props.predictorCount).meanLength;
  const pooledCenters = view('pooled-centers', meanLength);
  const status = createTransientView(graph, `${id}-diagnostic-fit-status`, 'uint32', 1),
    projectionStatus = createTransientView(
      graph,
      `${id}-diagnostic-projection-status`,
      'uint32',
      1
    );
  const nodes: GPUCommandNode<Parameters>[] = [
    ...getPanelWithinNodes(graph, {
      id: `${id}-diagnostic-center`,
      entityCount: 1,
      periodCount: rows,
      predictorCount: props.predictorCount,
      predictors: props.predictors,
      response: props.response,
      output: {predictors: centeredPredictors, response: centeredResponse, means: pooledCenters}
    }),
    ...new GPUOrdinaryLeastSquares({
      id: `${id}-diagnostic-pooled`,
      predictors: centeredPredictors,
      predictorCount: props.predictorCount,
      response: centeredResponse,
      output: {
        coefficients: view('coefficients', columns),
        standardErrors: view('errors', columns),
        tStatistics: view('statistics', columns),
        summary,
        residuals,
        fitted,
        status: {code: status}
      }
    }).getCommandNodes(graph)
  ];
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-diagnostic-lags`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'pooled-diagnostic-lags',
      invocationCount: rows,
      bindings: [
        binding('offsets', props.weights.offsets, 'read', 'u32'),
        binding('neighbors', props.weights.neighbors, 'read', 'u32'),
        binding('weights', props.weights.weights),
        binding('pooledCenters', pooledCenters),
        binding('residuals', residuals),
        binding('fitted', fitted),
        binding('lags', lags, 'read_write'),
        ...(props.weights.status?.overflow
          ? [binding('overflow', props.weights.status.overflow, 'read', 'u32')]
          : [])
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${ROUNDED_DIFFERENCE_WGSL}\n${ADDITION_ROUNDING_ERROR_WGSL}`,
      body: `let entity=index%${entities}u;let periodStart=index-entity;let base=lagsOffset+index*4u;
if(index==0u){lags[lagsOffset+${rows * 4}u]=0.0;}
for(var component=0u;component<4u;component++){lags[base+component]=getNaN();}
${props.weights.status?.overflow ? 'if(overflow[overflowOffset]!=0u){return;}' : ''}
let begin=offsets[offsetsOffset+entity];let end=offsets[offsetsOffset+entity+1u];
let referenceBegin=offsets[offsetsOffset];let referenceEnd=offsets[offsetsOffset+1u];
if(end<begin || end>${edges}u || referenceEnd<referenceBegin || referenceEnd>${edges}u){return;}
var referenceSum=0.0;var referenceError=0.0;
for(var slot=referenceBegin;slot<referenceEnd;slot++){let weight=weights[weightsOffset+slot];let next=referenceSum+weight;referenceError+=getAdditionRoundingError(referenceSum,weight,next);referenceSum=next;}
var rowSum=0.0;var rowError=0.0;var lagResidual=0.0;var lagFitted=0.0;var trace=0.0;
for(var slot=begin;slot<end;slot++){
 let neighbor=neighbors[neighborsOffset+slot];let weight=weights[weightsOffset+slot];
 if(neighbor>=${entities}u || neighbor==entity || !isFiniteBits(weight) || (slot>begin && neighbors[neighborsOffset+slot-1u]>=neighbor)){return;}
 var reciprocal=0.0;for(var other=offsets[offsetsOffset+neighbor];other<min(offsets[offsetsOffset+neighbor+1u],${edges}u);other++){if(neighbors[neighborsOffset+other]==entity){reciprocal=weights[weightsOffset+other];}}
 trace+=weight*(weight+reciprocal);
 lagResidual+=weight*residuals[residualsOffset+periodStart+neighbor];lagFitted+=weight*fitted[fittedOffset+periodStart+neighbor];
 let next=rowSum+weight;rowError+=getAdditionRoundingError(rowSum,weight,next);rowSum=next;
}
let runtimeZero=bitcast<u32>(pooledCenters[pooledCentersOffset+${meanLength - 1}u]);
let rowDifference=getRoundedDifference(rowSum,referenceSum,runtimeZero)+getRoundedDifference(rowError,referenceError,runtimeZero);
// W fitted - mean(y)*(W1)[0]: only a constant is removed. Nonconstant W1 remains signal.
lagFitted+=pooledCenters[pooledCentersOffset]*rowDifference+pooledCenters[pooledCentersOffset+1u]*rowDifference;
lags[base]=0.0;lags[base+1u]=lagResidual;lags[base+2u]=lagFitted;lags[base+3u]=trace;`
    })
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-diagnostic-project-input`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'diagnostic-project-input',
      invocationCount: rows,
      bindings: [binding('lags', lags), binding('laggedFit', laggedFit, 'read_write')],
      declarations: ROUNDED_DIFFERENCE_WGSL,
      body: `laggedFit[laggedFitOffset+index]=getRoundedDifference(lags[lagsOffset+index*4u+2u],lags[lagsOffset+2u],bitcast<u32>(lags[lagsOffset+${rows * 4}u]));`
    })
  );
  nodes.push(
    ...new GPUOrdinaryLeastSquares({
      id: `${id}-diagnostic-projection`,
      predictors: centeredPredictors,
      predictorCount: props.predictorCount,
      response: laggedFit,
      output: {
        coefficients: view('projection-coefficients', columns),
        standardErrors: view('projection-errors', columns),
        tStatistics: view('projection-statistics', columns),
        summary: view('projection-summary', 22),
        residuals: projectedResiduals,
        status: {code: projectionStatus}
      }
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...getPanelPreciseDiagnosticNodes(graph, id, props, {
      lags,
      residuals,
      projected: projectedResiduals
    })
  );
  const layout = getRegressionTileLayout(rows);
  const sums = view('sums', PANEL_DIAGNOSTIC_SUM_COUNT);
  nodes.push(
    ...getRowTileSumNodes(graph, {
      id: `${id}-diagnostic-sums`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'pooled-panel-lm-sums',
      rowCount: rows,
      layout,
      sumCount: PANEL_DIAGNOSTIC_SUM_COUNT,
      partial: view('partials', layout.tileCount * PANEL_DIAGNOSTIC_SUM_COUNT),
      output: sums,
      bindings: [
        binding('residuals', residuals),
        binding('lags', lags),
        binding('projected', projectedResiduals)
      ],
      declarations: `${ROUNDED_DIFFERENCE_WGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 let base = lagsOffset + row * 4u;
 let residual = residuals[residualsOffset + row]; let projectedValue = projected[projectedOffset + row];
 (*terms)[0] += residual * lags[base + 1u];
 (*terms)[1] += residual * projectedValue;
 (*terms)[2] += lags[base + 3u];
 (*terms)[3] += projectedValue * projectedValue;
 let centeredLag = lags[base + 2u];
 (*terms)[4] += centeredLag * centeredLag;
 (*terms)[5] += centeredLag;
}`
    })
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-diagnostic-tests`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'pooled-panel-lm-tests',
      invocationCount: 1,
      bindings: [
        binding('sums', sums),
        binding('summary', summary),
        binding('fitStatus', status, 'read', 'u32'),
        binding('projectionStatus', projectionStatus, 'read', 'u32'),
        binding('tests', diagnostics.tests, 'read_write'),
        binding('status', diagnostics.status, 'read_write', 'u32')
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${ERFC_VALUE_WGSL}\n${ROUNDED_DIFFERENCE_WGSL}
fn upperChiSquareOne(statistic:f32)->f32 { return erfcValue(sqrt(max(0.0,statistic)*0.5)); }`,
      body: `status[statusOffset] = 1u;
${Array.from({length: 6}, (_, slot) => `tests[testsOffset + ${slot}u] = getNaN();`).join(' ')}
if (fitStatus[fitStatusOffset] != 0u || projectionStatus[projectionStatusOffset] != 0u || summary[summaryOffset] != ${rows}.0) { return; }
let residualLag = sums[sumsOffset]; let projectedScore = sums[sumsOffset + 1u];
let trace = sums[sumsOffset + 2u]; let projectedSquares = sums[sumsOffset + 3u];
let laggedFitSum = sums[sumsOffset + 5u];
let laggedFitSquares = sums[sumsOffset + 4u];
let variance = summary[summaryOffset + 3u]; let projectedInformation = projectedSquares / variance;
let information = projectedInformation + trace; status[statusOffset] = 2u;
// A residual norm at most 1e-6 of the projected input is numerical rank zero at f32 precision.
// In particular W1=1, Wx=-x must not turn QR/Cholesky roundoff into robust-test information.
if (!(variance > 0.0 && trace > 0.0 && projectedInformation > 0.0 && projectedSquares / laggedFitSquares > 1e-12) || !isFiniteBits(laggedFitSquares) || !isFiniteBits(laggedFitSum) || !isFiniteBits(projectedInformation) || !isFiniteBits(residualLag) || !isFiniteBits(projectedScore)) { return; }
// Normalize before squaring: response scale cancels here, before either square can overflow.
let errorScore = residualLag / variance;
let error = errorScore * errorScore / trace;
let lagContrast = projectedScore / variance;
let robustLag = lagContrast * lagContrast / projectedInformation;
let errorContrast = projectedInformation / information * (residualLag / variance) - trace / information * lagContrast;
let robustError = errorContrast * errorContrast / (trace * (projectedInformation/information));
if (!(error >= 0.0 && robustLag >= 0.0 && robustError >= 0.0) || !isFiniteBits(error) || !isFiniteBits(robustLag) || !isFiniteBits(robustError)) { return; }
let errorProbability=upperChiSquareOne(error);let lagProbability=upperChiSquareOne(robustLag);let robustProbability=upperChiSquareOne(robustError);
if (!isFiniteBits(errorProbability) || !isFiniteBits(lagProbability) || !isFiniteBits(robustProbability)) { return; }
tests[testsOffset] = error; tests[testsOffset + 1u] = errorProbability;
tests[testsOffset + 2u] = robustLag; tests[testsOffset + 3u] = lagProbability;
tests[testsOffset + 4u] = robustError; tests[testsOffset + 5u] = robustProbability;
status[statusOffset] = 0u;`
    })
  );
  nodes.push(
    ...getPanelDiagnosticZeroFitNodes(graph, {
      id,
      predictorCount: props.predictorCount,
      predictors: props.predictors,
      response: props.response,
      diagnostics
    })
  );
  // The fit consumes both overflow and candidateOverflow. Its dependent pooled tests must
  // honor the same flags, including when all retained CSR rows themselves look valid.
  for (const [index, flag] of (guard.loss ? [guard.loss] : []).entries()) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-diagnostic-incomplete-${index}`,
        operation: 'GPUSpatialPanelErrorMaximumLikelihood',
        variant: 'pooled-diagnostic-incomplete',
        invocationCount: 6,
        bindings: [
          binding('flag', flag, 'read', 'u32'),
          binding('tests', diagnostics.tests, 'read_write'),
          binding('status', diagnostics.status, 'read_write', 'u32')
        ],
        declarations: GET_NAN_WGSL,
        body: 'if (flag[flagOffset] != 0u) { if (index == 0u) { status[statusOffset] = 1u; } tests[testsOffset + index] = getNaN(); }'
      })
    );
  }
  return [...guard.nodes, ...nodes];
}

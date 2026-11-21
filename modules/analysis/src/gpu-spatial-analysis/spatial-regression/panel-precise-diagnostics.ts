// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fp64arithmetic} from '@luma.gl/shaders';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getRegressionTileLayout, getRowTileSumNodes} from './regression-row-tiles';
import {validateRegressionAllocations} from './spatial-likelihood-allocations';
import type {GPUSpatialPanelErrorMaximumLikelihoodProps} from './gpu-spatial-panel-error-maximum-likelihood';

/** Shared sum layout of the diagnostic writer and its preflight. */
export const PANEL_DIAGNOSTIC_SUM_COUNT = 6;

/** Double-single moments, binary reduction and two coefficient vectors; no row-sized DS matrix. */
export function getPanelPreciseDiagnosticAllocationLengths(rows: number, predictorCount: number) {
  const width = predictorCount + 1,
    terms = (width * (width + 1)) / 2 + width,
    tiles = getRegressionTileLayout(rows).tileCount;
  return {
    'diagnostic precise partials': tiles * terms * 2,
    'diagnostic precise reduction': Math.max(1, Math.ceil(tiles / 2)) * terms * 2,
    'diagnostic precise lag pairs': rows * 2,
    'diagnostic precise scales': predictorCount,
    'diagnostic precise scale partials': tiles * predictorCount,
    'diagnostic precise coefficients': (width + 1) * 2 + predictorCount
  };
}

/** Preserve the low part of W fitted and the projected residual until the final row publication. */
export function getPanelPreciseDiagnosticNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUSpatialPanelErrorMaximumLikelihoodProps,
  ports: {
    lags: GraphDataView<'float32'>;
    residuals: GraphDataView<'float32'>;
    projected: GraphDataView<'float32'>;
  }
) {
  const rows = props.response.length,
    columns = props.predictorCount,
    width = columns + 1,
    pack = (width * (width + 1)) / 2,
    terms = pack + width,
    layout = getRegressionTileLayout(rows),
    entities = props.weights.offsets.length - 1;
  validateRegressionAllocations(
    graph.device,
    id,
    getPanelPreciseDiagnosticAllocationLengths(rows, columns)
  );
  const view = (name: string, length: number) =>
    createTransientView(graph, `${id}-diagnostic-precise-${name}`, 'float32', length);
  const partial = view('partials', layout.tileCount * terms * 2),
    scratch = view('reduction', Math.max(1, Math.ceil(layout.tileCount / 2)) * terms * 2),
    pooled = view('pooled', (width + 1) * 2 + columns),
    projection = view('projection', (width + 1) * 2 + columns),
    lagPairs = view('lag-pairs', rows * 2),
    maximum = view('maximum', columns),
    scalePartials = view('scale-partials', layout.tileCount * columns);
  const nodes: ReturnType<typeof createWGSLKernelNode<Parameters>>[] = [];
  const binding = (
    name: string,
    value: GraphDataView,
    access: 'read' | 'read_write' = 'read',
    type: 'f32' | 'u32' = 'f32'
  ): WGSLKernelBinding => ({name, view: value, access, type});
  const kernel = (
    name: string,
    count: number,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = ''
  ) =>
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-diagnostic-precise-${name}`,
        operation: 'GPUSpatialPanelErrorMaximumLikelihood',
        invocationCount: count,
        bindings,
        body,
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${declarations}`,
        modules: [fp64arithmetic]
      })
    );
  const load = `fn readPair(slot:u32)->vec2f{return vec2f(totals[totalsOffset+slot*2u],totals[totalsOffset+slot*2u+1u]);}`;
  nodes.push(
    ...getRowTileSumNodes(graph, {
      id: `${id}-diagnostic-precise-scale`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'diagnostic-scales',
      rowCount: rows,
      layout,
      sumCount: 0,
      maximumCount: columns,
      partial: scalePartials,
      output: maximum,
      bindings: [binding('predictors', props.predictors)],
      declarations: `fn accumulateRow(row:u32,terms:ptr<function,array<f32,TERM_COUNT>>){for(var column=0u;column<${columns}u;column++){(*terms)[column]=max((*terms)[column],abs(predictors[predictorsOffset+row*${columns}u+column]));}}`
    })
  );
  const moments = (phase: 'pooled' | 'projection', coefficients: GraphDataView<'float32'>) => {
    kernel(
      `${phase}-tiles`,
      layout.tileCount,
      [
        binding('predictors', props.predictors),
        binding('maximum', maximum),
        binding('response', phase === 'pooled' ? props.response : lagPairs),
        binding('partial', partial, 'read_write')
      ],
      `
var totals:array<f32,${terms * 2}>;
for(var row=index*${layout.tileRowCount}u;row<min((index+1u)*${layout.tileRowCount}u,${rows}u);row++){
 let responseDeviation=${phase === 'pooled' ? 'sub_fp64(vec2f(response[responseOffset+row],0.0),vec2f(response[responseOffset],0.0))' : 'sub_fp64(vec2f(response[responseOffset+row*2u],response[responseOffset+row*2u+1u]),vec2f(response[responseOffset],response[responseOffset+1u]))'};
 for(var first=0u;first<${width}u;first++){
  for(var second=0u;second<=first;second++){let slot=first*(first+1u)/2u+second;let total=sum_fp64(vec2f(totals[slot*2u],totals[slot*2u+1u]),mul_fp64(readDesign(row,first),readDesign(row,second)));totals[slot*2u]=total.x;totals[slot*2u+1u]=total.y;}
  let slot=(${pack}u+first)*2u;let total=sum_fp64(vec2f(totals[slot],totals[slot+1u]),mul_fp64(readDesign(row,first),responseDeviation));totals[slot]=total.x;totals[slot+1u]=total.y;
 }
}
for(var slot=0u;slot<${terms}u;slot++){partial[partialOffset+(index*${terms}u+slot)*2u]=totals[slot*2u];partial[partialOffset+(index*${terms}u+slot)*2u+1u]=totals[slot*2u+1u];}`,
      `fn readDesign(row:u32,column:u32)->vec2f{if(column==0u){return vec2f(1.0,0.0);}let exponent=i32((bitcast<u32>(maximum[maximumOffset+column-1u])>>23u)&255u)-127;return sub_fp64(fp64_scale_fp64(vec2f(predictors[predictorsOffset+row*${columns}u+column-1u],0.0),-exponent),fp64_scale_fp64(vec2f(predictors[predictorsOffset+column-1u],0.0),-exponent));}`
    );
    let active = layout.tileCount,
      source = partial,
      target = scratch,
      pass = 0;
    while (active > 1) {
      const count = Math.ceil(active / 2);
      kernel(
        `${phase}-merge-${pass++}`,
        count * terms,
        [binding('source', source), binding('mergeOutput', target, 'read_write')],
        `let tile=index/${terms}u;let term=index%${terms}u;let first=(tile*2u*${terms}u+term)*2u;var value=vec2f(source[sourceOffset+first],source[sourceOffset+first+1u]);if(tile*2u+1u<${active}u){let second=first+${terms * 2}u;value=sum_fp64(value,vec2f(source[sourceOffset+second],source[sourceOffset+second+1u]));}mergeOutput[mergeOutputOffset+index*2u]=value.x;mergeOutput[mergeOutputOffset+index*2u+1u]=value.y;`
      );
      active = count;
      [source, target] = [target, source];
    }
    kernel(
      `${phase}-solve`,
      width,
      [
        binding('totals', source),
        binding('maximum', maximum),
        binding('coefficients', coefficients, 'read_write'),
        binding('reference', phase === 'pooled' ? props.response : lagPairs)
      ],
      `
var matrix:array<f32,${width * (width + 1) * 2}>;
for(var row=0u;row<${width}u;row++){for(var column=0u;column<${width}u;column++){let high=max(row,column);let low=min(row,column);writeMatrix(&matrix,row,column,readPair(high*(high+1u)/2u+low));}writeMatrix(&matrix,row,${width}u,readPair(${pack}u+row));}
for(var pivot=0u;pivot<${width}u;pivot++){
 var chosen=pivot;for(var row=pivot+1u;row<${width}u;row++){if(abs(readMatrix(&matrix,row,pivot).x)>abs(readMatrix(&matrix,chosen,pivot).x)){chosen=row;}}
 for(var column=0u;column<${width + 1}u;column++){let temporary=readMatrix(&matrix,pivot,column);let replacement=readMatrix(&matrix,chosen,column);writeMatrix(&matrix,pivot,column,replacement);writeMatrix(&matrix,chosen,column,temporary);}
 let divisor=readMatrix(&matrix,pivot,pivot);if(divisor.x==0.0){coefficients[coefficientsOffset+index*2u]=getNaN();coefficients[coefficientsOffset+index*2u+1u]=0.0;return;}
 for(var column=pivot+1u;column<${width + 1}u;column++){let value=div_fp64(readMatrix(&matrix,pivot,column),divisor);writeMatrix(&matrix,pivot,column,value);}writeMatrix(&matrix,pivot,pivot,vec2f(1.0,0.0));
 for(var row=0u;row<${width}u;row++){if(row==pivot){continue;}let multiplier=readMatrix(&matrix,row,pivot);for(var column=pivot+1u;column<${width + 1}u;column++){let value=sub_fp64(readMatrix(&matrix,row,column),mul_fp64(multiplier,readMatrix(&matrix,pivot,column)));writeMatrix(&matrix,row,column,value);}writeMatrix(&matrix,row,pivot,vec2f(0.0));}
}
if(index<${columns}u){coefficients[coefficientsOffset+${(width + 1) * 2}u+index]=f32(i32((bitcast<u32>(maximum[maximumOffset+index])>>23u)&255u)-127);}
let value=readMatrix(&matrix,index,${width}u);coefficients[coefficientsOffset+index*2u]=value.x;coefficients[coefficientsOffset+index*2u+1u]=value.y;
if(index==0u){let center=sum_fp64(div_fp64(readPair(${pack}u),vec2f(${rows}.0,0.0)),${phase === 'pooled' ? 'vec2f(reference[referenceOffset],0.0)' : 'vec2f(reference[referenceOffset],reference[referenceOffset+1u])'});coefficients[coefficientsOffset+${width * 2}u]=center.x;coefficients[coefficientsOffset+${width * 2 + 1}u]=center.y;}`,
      `${load}
fn readMatrix(matrix:ptr<function,array<f32,${width * (width + 1) * 2}>>,row:u32,column:u32)->vec2f{let slot=(row*${width + 1}u+column)*2u;return vec2f((*matrix)[slot],(*matrix)[slot+1u]);}
fn writeMatrix(matrix:ptr<function,array<f32,${width * (width + 1) * 2}>>,row:u32,column:u32,value:vec2f){let slot=(row*${width + 1}u+column)*2u;(*matrix)[slot]=value.x;(*matrix)[slot+1u]=value.y;}`
    );
  };
  moments('pooled', pooled);
  const fit = `fn pooledFit(row:u32)->vec2f{var value=sum_fp64(vec2f(response[responseOffset],0.0),vec2f(pooled[pooledOffset],pooled[pooledOffset+1u]));for(var column=0u;column<${columns}u;column++){let exponent=i32(pooled[pooledOffset+${(width + 1) * 2}u+column]);let coordinate=sub_fp64(fp64_scale_fp64(vec2f(predictors[predictorsOffset+row*${columns}u+column],0.0),-exponent),fp64_scale_fp64(vec2f(predictors[predictorsOffset+column],0.0),-exponent));value=sum_fp64(value,mul_fp64(coordinate,vec2f(pooled[pooledOffset+(column+1u)*2u],pooled[pooledOffset+(column+1u)*2u+1u])));}return value;}`;
  kernel(
    'lags',
    rows,
    [
      binding('offsets', props.weights.offsets, 'read', 'u32'),
      binding('neighbors', props.weights.neighbors, 'read', 'u32'),
      binding('weights', props.weights.weights),
      binding('predictors', props.predictors),
      binding('response', props.response),
      binding('pooled', pooled),
      binding('lags', ports.lags, 'read_write'),
      binding('pairs', lagPairs, 'read_write')
    ],
    `
let base=lagsOffset+index*4u;if(!isFiniteBits(lags[base])){pairs[pairsOffset+index*2u]=getNaN();pairs[pairsOffset+index*2u+1u]=0.0;return;}
let entity=index%${entities}u;let period=index-entity;var lagFit=vec2f(0.0);var lagResidual=vec2f(0.0);
for(var slot=offsets[offsetsOffset+entity];slot<offsets[offsetsOffset+entity+1u];slot++){let row=period+neighbors[neighborsOffset+slot];let weight=vec2f(weights[weightsOffset+slot],0.0);let fitted=pooledFit(row);lagFit=sum_fp64(lagFit,mul_fp64(weight,fitted));lagResidual=sum_fp64(lagResidual,mul_fp64(weight,sub_fp64(vec2f(response[responseOffset+row],0.0),fitted)));}
pairs[pairsOffset+index*2u]=lagFit.x;pairs[pairsOffset+index*2u+1u]=lagFit.y;lags[base+1u]=lagResidual.x+lagResidual.y;`,
    fit
  );
  moments('projection', projection);
  kernel(
    'publish',
    rows,
    [
      binding('predictors', props.predictors),
      binding('response', props.response),
      binding('pooled', pooled),
      binding('projection', projection),
      binding('pairs', lagPairs),
      binding('lags', ports.lags, 'read_write'),
      binding('residuals', ports.residuals, 'read_write'),
      binding('projected', ports.projected, 'read_write')
    ],
    `
let value=vec2f(pairs[pairsOffset+index*2u],pairs[pairsOffset+index*2u+1u]);var projectedValue=sub_fp64(sub_fp64(value,vec2f(pairs[pairsOffset],pairs[pairsOffset+1u])),vec2f(projection[projectionOffset],projection[projectionOffset+1u]));
for(var column=0u;column<${columns}u;column++){let exponent=i32(pooled[pooledOffset+${(width + 1) * 2}u+column]);let coordinate=sub_fp64(fp64_scale_fp64(vec2f(predictors[predictorsOffset+index*${columns}u+column],0.0),-exponent),fp64_scale_fp64(vec2f(predictors[predictorsOffset+column],0.0),-exponent));projectedValue=sub_fp64(projectedValue,mul_fp64(coordinate,vec2f(projection[projectionOffset+(column+1u)*2u],projection[projectionOffset+(column+1u)*2u+1u])));}
let centered=sub_fp64(value,vec2f(projection[projectionOffset+${width * 2}u],projection[projectionOffset+${width * 2 + 1}u]));let residual=sub_fp64(vec2f(response[responseOffset+index],0.0),pooledFit(index));
projected[projectedOffset+index]=projectedValue.x+projectedValue.y;residuals[residualsOffset+index]=residual.x+residual.y;lags[lagsOffset+index*4u+2u]=centered.x+centered.y;`,
    fit
  );
  return nodes;
}

/** Additional bounded dispatches in the precision-preserving diagnostic chain. */
export function getPanelPreciseDiagnosticNodeCount(rows: number): number {
  return 8 + 2 * Math.ceil(Math.log2(getRegressionTileLayout(rows).tileCount));
}

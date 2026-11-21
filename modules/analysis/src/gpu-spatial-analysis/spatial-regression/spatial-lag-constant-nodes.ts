// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getGPUSpatialWeightsTilePlan, wrapWeightsChunkRowBodyWGSL} from './chunked-weights-kernels';
import {getRegressionTileLayout, getRowTileSumNodes} from './regression-row-tiles';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUSpatialLagMaximumLikelihoodProps} from './gpu-spatial-lag-maximum-likelihood';

/** Exact row-product equality to row zero, with a conservative unsupported-span fallback. */
export function getSpatialLagConstantNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUSpatialLagMaximumLikelihoodProps,
  witness: GraphDataView<'float32'>,
  mismatch: GraphDataView<'float32'>,
  partial: GraphDataView<'float32'>
) {
  const rows = props.response.length,
    layout = getRegressionTileLayout(rows),
    plan = getGPUSpatialWeightsTilePlan(props.weights, layout.tileRowCount);
  const reference = plan.dispatches[0].weights;
  const nodes = plan.dispatches.map(dispatch =>
    createWGSLKernelNode(graph, {
      id: `${id}-constant-lag-witness${dispatch.idSuffix}`,
      operation: 'GPUSpatialLagMaximumLikelihood',
      invocationCount: dispatch.rowCount,
      bindings: [
        {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
        {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
        {name: 'referenceOffsets', view: reference.offsets, type: 'u32', access: 'read'},
        {name: 'referenceNeighbors', view: reference.neighbors, type: 'u32', access: 'read'},
        {name: 'referenceWeights', view: reference.weights, type: 'f32', access: 'read'},
        {name: 'response', view: props.response, type: 'f32', access: 'read'},
        {name: 'witness', view: witness, type: 'f32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}\n${EXACT_ORIENTATION_WGSL}\n${dispatch.constantsWGSL}
fn spatialLagAccumulatorBit(words:ptr<function,array<u32,8>>,bit:i32)->u32{if(bit<0i || bit>=256i){return 0u;}return ((*words)[u32(bit)/32u]>>(u32(bit)%32u))&1u;}
fn getSpatialLagRoundedConstant(words:ptr<function,array<u32,8>>,minimum:i32,negative:bool)->f32{
 var highest=-1i;for(var word=8u;word>0u;word--){if((*words)[word-1u]!=0u){highest=i32((word-1u)*32u+31u-countLeadingZeros((*words)[word-1u]));break;}}
 if(highest<0i){return 0.0;}var exponent=minimum+highest;
 var shift=highest-23i;if(exponent< -126i){shift= -149i-minimum;}
 var significand=0u;for(var bit=0u;bit<24u;bit++){significand|=spatialLagAccumulatorBit(words,shift+i32(bit))<<bit;}
 let guard=spatialLagAccumulatorBit(words,shift-1i);var sticky=false;
 for(var word=0u;word<8u;word++){let first=i32(word*32u);if(first+32i<=shift-1i){sticky=sticky||(*words)[word]!=0u;}else if(first<shift-1i){let mask=(1u<<u32(shift-1i-first))-1u;sticky=sticky||((*words)[word]&mask)!=0u;}}
 if(guard!=0u&&(sticky||(significand&1u)!=0u)){significand++;}
 let sign=select(0u,0x80000000u,negative);
 if(exponent< -126i){return bitcast<f32>(sign|significand);}
 if(significand>=0x1000000u){significand>>=1u;exponent++;}
 if(exponent>127i){return getNaN();}
 return bitcast<f32>(sign|(u32(exponent+127i)<<23u)|(significand&0x7fffffu));
}`,
      body: wrapWeightsChunkRowBodyWGSL(
        dispatch,
        `witness[witnessOffset+index]=getNaN();
let begin=offsets[offsetsOffset + index];let end=offsets[offsetsOffset + index + 1u];
let referenceBegin=referenceOffsets[referenceOffsetsOffset];let referenceEnd=referenceOffsets[referenceOffsetsOffset+1u];
if(end<begin || end>${dispatch.weights.weights.length}u || referenceEnd<referenceBegin || referenceEnd>${reference.weights.length}u){return;}
var minimum=10000i;var maximum=-10000i;
for(var side=0u;side<2u;side++){
 let first=select(begin,referenceBegin,side==1u);let last=select(end,referenceEnd,side==1u);
 for(var slot=first;slot<last;slot++){
  var neighbor=0u;var weight=0.0;
  if(side==0u){neighbor=neighbors[neighborsOffset+slot];weight=weights[weightsOffset+slot];}
  else{neighbor=referenceNeighbors[referenceNeighborsOffset+slot];weight=referenceWeights[referenceWeightsOffset+slot];}
  if(neighbor>=${rows}u){continue;}let value=response[responseOffset+neighbor];
  let firstBits=bitcast<u32>(weight)&0x7fffffffu;let secondBits=bitcast<u32>(value)&0x7fffffffu;
  if(firstBits>=0x7f800000u || secondBits>=0x7f800000u){return;}
  if(firstBits==0u || secondBits==0u){continue;}
  let firstExponent=select(i32(firstBits>>23u)-150,-149,(firstBits>>23u)==0u);
  let secondExponent=select(i32(secondBits>>23u)-150,-149,(secondBits>>23u)==0u);
  minimum=min(minimum,firstExponent+secondExponent);maximum=max(maximum,firstExponent+secondExponent);
 }
}
// Bound the exponent span with enough carry headroom for both CSR capacities.
if(maximum-minimum>${Math.min(180, 207 - Math.ceil(Math.log2(Math.max(1, dispatch.weights.weights.length + reference.weights.length))))}i){return;}
var positive:array<u32,8>;var negative:array<u32,8>;var referencePositive:array<u32,8>;var referenceNegative:array<u32,8>;
for(var side=0u;side<2u;side++){
 let first=select(begin,referenceBegin,side==1u);let last=select(end,referenceEnd,side==1u);
 for(var slot=first;slot<last;slot++){
  var neighbor=0u;var weight=0.0;
  if(side==0u){neighbor=neighbors[neighborsOffset+slot];weight=weights[weightsOffset+slot];}
  else{neighbor=referenceNeighbors[referenceNeighborsOffset+slot];weight=referenceWeights[referenceWeightsOffset+slot];}
  if(neighbor>=${rows}u){continue;}let value=response[responseOffset+neighbor];
  let firstBits=bitcast<u32>(weight);let secondBits=bitcast<u32>(value);
  let firstMagnitude=firstBits&0x7fffffffu;let secondMagnitude=secondBits&0x7fffffffu;
  if(firstMagnitude==0u || secondMagnitude==0u){continue;}
  let firstExponent=select(i32(firstMagnitude>>23u)-150,-149,(firstMagnitude>>23u)==0u);
  let secondExponent=select(i32(secondMagnitude>>23u)-150,-149,(secondMagnitude>>23u)==0u);
  let firstSignificand=(firstMagnitude&0x7fffffu)|select(0x800000u,0u,(firstMagnitude>>23u)==0u);
  let secondSignificand=(secondMagnitude&0x7fffffu)|select(0x800000u,0u,(secondMagnitude>>23u)==0u);
  let product=multiply24(firstSignificand,secondSignificand);let shift=u32(firstExponent+secondExponent-minimum);
  let originalSign=(firstBits^secondBits)>>31u;
  if(side==1u){if(originalSign==0u){accumulateShifted(&referencePositive,product.x,product.y,shift);}else{accumulateShifted(&referenceNegative,product.x,product.y,shift);}}
  let subtract=originalSign^side;
  if(subtract==0u){accumulateShifted(&positive,product.x,product.y,shift);}else{accumulateShifted(&negative,product.x,product.y,shift);}
 }
}
for(var word=0u;word<8u;word++){if(positive[word]!=negative[word]){return;}}
var negativeReference=false;
for(var word=8u;word>0u;word--){if(referencePositive[word-1u]!=referenceNegative[word-1u]){negativeReference=referencePositive[word-1u]<referenceNegative[word-1u];break;}}
var magnitude:array<u32,8>;var borrow=0u;
for(var word=0u;word<8u;word++){let first=select(referencePositive[word],referenceNegative[word],negativeReference);let second=select(referenceNegative[word],referencePositive[word],negativeReference);let partial=first-second;let next=partial-borrow;magnitude[word]=next;borrow=select(0u,1u,first<second)|select(0u,1u,partial<borrow);}
let value=getSpatialLagRoundedConstant(&magnitude,minimum,negativeReference);if(isFiniteBits(value)){witness[witnessOffset+index]=value;}`
      )
    })
  );
  return [
    ...nodes,
    ...getRowTileSumNodes(graph, {
      id: `${id}-constant-lag`,
      operation: 'GPUSpatialLagMaximumLikelihood',
      variant: 'constant-lag',
      rowCount: rows,
      layout,
      sumCount: 0,
      maximumCount: 1,
      partial,
      output: mismatch,
      bindings: [{name: 'witness', view: witness, type: 'f32', access: 'read'}],
      declarations: `${EXACT_ORIENTATION_WGSL}
fn accumulateRow(row:u32,terms:ptr<function,array<f32,TERM_COUNT>>){(*terms)[0]=max((*terms)[0],select(1.0,0.0,isFiniteBits(witness[witnessOffset+row])));}`
    })
  ];
}

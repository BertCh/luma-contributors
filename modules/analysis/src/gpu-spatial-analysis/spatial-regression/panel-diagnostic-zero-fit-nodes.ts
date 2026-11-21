// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getRegressionTileLayout} from './regression-row-tiles';

// Finite f32 products occupy fewer than 554 bits in units of 2^-298. At most 2^32
// rows need another 32 carry bits, so twenty u32 words retain every input product exactly.
const WORDS = 20,
  STRIDE = 2 * WORDS + 1;
const EXACT_DIAGNOSTIC_DOT_WGSL = `
fn addDiagnosticLimb(accumulator:ptr<function,array<u32,20>>,value:u32,shift:u32){
 let first=shift/32u;let bit=shift%32u;let low=value<<bit;var high=0u;
 if(bit>0u){high=value>>(32u-bit);}
 var carry=0u;
 for(var word=first;word<20u;word++){
  var addend=0u;if(word==first){addend=low;}else if(word==first+1u){addend=high;}else if(carry==0u){break;}
  let next=(*accumulator)[word]+addend;let firstCarry=select(0u,1u,next<addend);
  let total=next+carry;let secondCarry=select(0u,1u,total<next);
  (*accumulator)[word]=total;carry=firstCarry|secondCarry;
 }
}
fn addDiagnosticProduct(first:u32,second:u32,positive:ptr<function,array<u32,20>>,negative:ptr<function,array<u32,20>>)->bool{
 let firstMagnitude=first&0x7fffffffu;let secondMagnitude=second&0x7fffffffu;
 let firstEncoded=firstMagnitude>>23u;let secondEncoded=secondMagnitude>>23u;
 if(firstEncoded==255u || secondEncoded==255u){return false;}
 if(firstMagnitude==0u || secondMagnitude==0u){return true;}
 let firstSignificand=(firstMagnitude&0x7fffffu)|select(0u,0x800000u,firstEncoded!=0u);
 let secondSignificand=(secondMagnitude&0x7fffffu)|select(0u,0x800000u,secondEncoded!=0u);
 let firstExponent=select(-149i,i32(firstEncoded)-150i,firstEncoded!=0u);
 let secondExponent=select(-149i,i32(secondEncoded)-150i,secondEncoded!=0u);
 let shift=u32(firstExponent+secondExponent+298i);
 let low=(firstSignificand&4095u)*(secondSignificand&4095u);
 let middle=(firstSignificand>>12u)*(secondSignificand&4095u)+(firstSignificand&4095u)*(secondSignificand>>12u)+(low>>12u);
 let upper=(firstSignificand>>12u)*(secondSignificand>>12u)+(middle>>12u);
 let limbs=array<u32,4>(low&4095u,middle&4095u,upper&4095u,upper>>12u);
 for(var limb=0u;limb<4u;limb++){
  if(((first^second)>>31u)!=0u){addDiagnosticLimb(negative,limbs[limb],shift+12u*limb);}
  else{addDiagnosticLimb(positive,limbs[limb],shift+12u*limb);}
 }
 return true;
}`;

/** Exact original-input [1,X]'y=0 certificate. It refuses zero pooled-fit information
 * without a response-relative cutoff that would erase small identifiable slopes. @internal */
export function getPanelDiagnosticZeroFitNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    predictorCount: number;
    predictors: GraphDataView<'float32'>;
    response: GraphDataView<'float32'>;
    diagnostics: {tests: GraphDataView<'float32'>; status: GraphDataView<'uint32'>};
  }
): GPUCommandNode<Parameters>[] {
  const {id, predictorCount, predictors, response, diagnostics} = props;
  const columns = predictorCount + 1,
    rows = response.length;
  const {tileCount, tileRowCount} = getRegressionTileLayout(rows);
  const partialLength = columns * tileCount * STRIDE;
  for (const [name, length] of [
    ['exact diagnostic dot partials', partialLength],
    ['exact diagnostic dot flags', columns]
  ] as const)
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      length * 4,
      'reduce panel dimensions'
    );
  const partial = createTransientView(
    graph,
    `${id}-diagnostic-dot-partials`,
    'uint32',
    partialLength
  );
  const zero = createTransientView(graph, `${id}-diagnostic-dot-zero`, 'uint32', columns);
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-diagnostic-dot-tiles`,
      operation: 'PanelDiagnosticOrthogonality',
      invocationCount: columns * tileCount,
      bindings: [
        {name: 'predictors', view: predictors, type: 'u32', access: 'read'},
        {name: 'response', view: response, type: 'u32', access: 'read'},
        {name: 'partial', view: partial, type: 'u32', access: 'read_write'}
      ],
      declarations: EXACT_DIAGNOSTIC_DOT_WGSL,
      body: `let column=index/${tileCount}u;let tile=index%${tileCount}u;let base=partialOffset+index*${STRIDE}u;
var positive:array<u32,20>;var negative:array<u32,20>;var invalid=0u;
for(var row=tile*${tileRowCount}u;row<min((tile+1u)*${tileRowCount}u,${rows}u);row++){
 var value=0x3f800000u;if(column>0u){value=predictors[predictorsOffset+row*${predictorCount}u+column-1u];}
 if(!addDiagnosticProduct(value,response[responseOffset+row],&positive,&negative)){invalid=1u;}
}
for(var word=0u;word<20u;word++){partial[base+word]=positive[word];partial[base+20u+word]=negative[word];}
partial[base+40u]=invalid;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-diagnostic-dot-merge`,
      operation: 'PanelDiagnosticOrthogonality',
      invocationCount: columns,
      bindings: [
        {name: 'partial', view: partial, type: 'u32', access: 'read'},
        {name: 'zero', view: zero, type: 'u32', access: 'read_write'}
      ],
      body: `var positive:array<u32,20>;var negative:array<u32,20>;var valid=true;
for(var tile=0u;tile<${tileCount}u;tile++){
 let base=partialOffset+(index*${tileCount}u+tile)*${STRIDE}u;valid=valid && partial[base+40u]==0u;
 var positiveCarry=0u;var negativeCarry=0u;
 for(var word=0u;word<20u;word++){
  let first=positive[word]+partial[base+word];let firstCarry=select(0u,1u,first<positive[word]);let total=first+positiveCarry;
  positive[word]=total;positiveCarry=firstCarry|select(0u,1u,total<first);
  let second=negative[word]+partial[base+20u+word];let secondCarry=select(0u,1u,second<negative[word]);let other=second+negativeCarry;
  negative[word]=other;negativeCarry=secondCarry|select(0u,1u,other<second);
 }
}
for(var word=0u;word<20u;word++){valid=valid && positive[word]==negative[word];}
zero[zeroOffset+index]=select(0u,1u,valid);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-diagnostic-zero-fit`,
      operation: 'PanelDiagnosticOrthogonality',
      // contract-allow: single-invocation folds one certificate per design column and six outputs.
      invocationCount: 1,
      bindings: [
        {name: 'zero', view: zero, type: 'u32', access: 'read'},
        {name: 'tests', view: diagnostics.tests, type: 'f32', access: 'read_write'},
        {name: 'status', view: diagnostics.status, type: 'u32', access: 'read_write'}
      ],
      declarations: GET_NAN_WGSL,
      body: `if(status[statusOffset]==1u){return;}var exact=true;for(var column=0u;column<${columns}u;column++){exact=exact && zero[zeroOffset+column]!=0u;}
if(exact){status[statusOffset]=2u;for(var slot=0u;slot<6u;slot++){tests[testsOffset+slot]=getNaN();}}`
    })
  ];
}

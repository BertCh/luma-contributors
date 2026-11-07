// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {PHILOX_WGSL, PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

export type GPUSegregationDecompositionProps = {
  id?: string;
  /** Four common-support tables: baseline, composition-only, spatial-only, final.
   * Each vec2 contains focal and complement counts. Counterfactual construction is explicit caller policy. */
  populations: readonly [
    GraphDataView<'float32x2'>,
    GraphDataView<'float32x2'>,
    GraphDataView<'float32x2'>,
    GraphDataView<'float32x2'>
  ];
  /** Identifies the supplied common boundary/support and counterfactual construction, not an inference model. */
  supportId: string;
  counterfactualMethod: string;
  /** Paired within-support baseline/final label exchange, assuming exchangeability of the two labels. */
  randomization?: {replicates: number; seed: number; swaps?: GraphDataView<'uint32'>};
};

/** Two-factor Shapley decomposition of the dissimilarity-index change, without randomization inference. */
export function addSegregationDecompositionRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSegregationDecompositionProps
) {
  const id = props.id ?? 'segregation-decomposition';
  const count = props.populations[0].length;
  assert(count > 0 && props.populations.every(population => population.length === count));
  assert(props.supportId.length > 0 && props.counterfactualMethod.length > 0);
  const indices = getRecipeView(graph, `${id}-indices`, 'float32', 4);
  const validity = getRecipeView(graph, `${id}-validity`, 'uint32', 4);
  const decomposition = getRecipeView(graph, `${id}-decomposition`, 'float32x4', 1);
  for (let scenario = 0; scenario < 4; scenario++) {
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-index-${scenario}`,
        operation: id,
        invocationCount: 1,
        declarations:
          IS_FINITE_VALUE_WGSL +
          `
fn readPopulation(row:u32)->vec2<f32>{return vec2<f32>(population[populationOffset+row*2u+0u],population[populationOffset+row*2u+1u]);}
`,
        bindings: [
          {name: 'population', view: props.populations[scenario], type: 'f32', access: 'read'},
          {name: 'indices', view: indices, type: 'f32', access: 'read_write'},
          {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
        ],
        body: `var totals=vec2<f32>(0.0);var valid=true;
for(var unit=0u;unit<${count}u;unit++){let counts=readPopulation(unit);valid=valid&&isFiniteValue(counts.x)&&isFiniteValue(counts.y)&&all(counts>=vec2<f32>(0.0));totals+=counts;}
valid=valid&&all(totals>vec2<f32>(0.0))&&isFiniteValue(totals.x)&&isFiniteValue(totals.y);
var dissimilarity=0.0;for(var unit=0u;unit<${count}u;unit++){let shares=readPopulation(unit)/totals;dissimilarity+=0.5*abs(shares.x-shares.y);}
indices[indicesOffset+${scenario}u]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),dissimilarity,valid);validity[validityOffset+${scenario}u]=u32(valid);`
      })
    );
  }
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-shapley`,
      operation: id,
      invocationCount: 1,
      declarations: `
fn readOutput(row:u32)->vec4<f32>{return vec4<f32>(output[outputOffset+row*4u+0u],output[outputOffset+row*4u+1u],output[outputOffset+row*4u+2u],output[outputOffset+row*4u+3u]);}
fn writeOutput(row:u32,value:vec4<f32>){output[outputOffset+row*4u+0u]=value[0u];output[outputOffset+row*4u+1u]=value[1u];output[outputOffset+row*4u+2u]=value[2u];output[outputOffset+row*4u+3u]=value[3u];}
`,
      bindings: [
        {name: 'indices', view: indices, type: 'f32', access: 'read'},
        {name: 'output', view: decomposition, type: 'f32', access: 'read_write'}
      ],
      body: `let baseline=indices[indicesOffset];let compositionOnly=indices[indicesOffset+1u];let spatialOnly=indices[indicesOffset+2u];let finalValue=indices[indicesOffset+3u];
let composition=0.5*((compositionOnly-baseline)+(finalValue-spatialOnly));let spatial=0.5*((spatialOnly-baseline)+(finalValue-compositionOnly));let change=finalValue-baseline;
writeOutput(0u,vec4<f32>(change,composition,spatial,change-composition-spatial));`
    })
  );
  let randomization:
    | {distribution: GraphDataView<'float32'>; inference: GraphDataView<'float32x4'>}
    | undefined;
  if (props.randomization) {
    const options = props.randomization;
    assert(
      Number.isSafeInteger(options.replicates) &&
        options.replicates > 0 &&
        options.replicates <= 65536
    );
    assert(Number.isSafeInteger(options.seed) && options.seed >= 0 && options.seed <= 0xffffffff);
    assert(!options.swaps || options.swaps.length === count * options.replicates);
    const distribution = getRecipeView(graph, `${id}-randomization`, 'float32', options.replicates);
    const inference = getRecipeView(graph, `${id}-inference`, 'float32x4', 1);
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-paired-randomization`,
        operation: id,
        invocationCount: options.replicates,
        declarations:
          IS_FINITE_VALUE_WGSL +
          PHILOX_WGSL +
          `
fn swapAt(unit:u32,replicate:u32)->u32{return ${options.swaps ? 'swaps[swapsOffset+replicate*' + count + 'u+unit]' : 'philox4x32(vec4<u32>(unit,replicate,0u,0u),vec2<u32>(' + options.seed + 'u,' + PHILOX_PURPOSES.segregationPairedExchange.id + 'u)).x & 1u'};}
fn populationAt(unit:u32,scenario:u32,swap:u32)->vec2<f32>{let first=vec2<f32>(baseline[baselineOffset+2u*unit],baseline[baselineOffset+2u*unit+1u]);let second=vec2<f32>(finalPopulation[finalPopulationOffset+2u*unit],finalPopulation[finalPopulationOffset+2u*unit+1u]);return select(first,second,(swap!=0u)!=(scenario!=0u));}`,
        bindings: [
          {name: 'baseline', view: props.populations[0], type: 'f32', access: 'read'},
          {name: 'finalPopulation', view: props.populations[3], type: 'f32', access: 'read'},
          ...(options.swaps
            ? [{name: 'swaps', view: options.swaps, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'output', view: distribution, type: 'f32', access: 'read_write'}
        ],
        // One exchange draw per unit and pass serves both scenarios (two Philox blocks per unit
        // instead of five), with the same stream as before.
        body: `var values=vec2<f32>(0.0);var valid=true;var totals0=vec2<f32>(0.0);var totals1=vec2<f32>(0.0);
for(var unit=0u;unit<${count}u;unit++){let swap=swapAt(unit,index);valid=valid&&swap<=1u;let population0=populationAt(unit,0u,swap);let population1=populationAt(unit,1u,swap);valid=valid&&all(population0>=vec2<f32>(0.0))&&isFiniteValue(population0.x)&&isFiniteValue(population0.y)&&all(population1>=vec2<f32>(0.0))&&isFiniteValue(population1.x)&&isFiniteValue(population1.y);totals0+=population0;totals1+=population1;}
valid=valid&&all(totals0>vec2<f32>(0.0))&&isFiniteValue(totals0.x)&&isFiniteValue(totals0.y)&&all(totals1>vec2<f32>(0.0))&&isFiniteValue(totals1.x)&&isFiniteValue(totals1.y);
for(var unit=0u;unit<${count}u;unit++){let swap=swapAt(unit,index);let shares0=populationAt(unit,0u,swap)/totals0;let shares1=populationAt(unit,1u,swap)/totals1;values.x+=0.5*abs(shares0.x-shares0.y);values.y+=0.5*abs(shares1.x-shares1.y);}
output[outputOffset+index]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),values.y-values.x,valid);`
      })
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-randomization-inference`,
        operation: id,
        invocationCount: 1,
        declarations: IS_FINITE_VALUE_WGSL,
        bindings: [
          {name: 'indices', view: indices, type: 'f32', access: 'read'},
          {name: 'distribution', view: distribution, type: 'f32', access: 'read'},
          {name: 'output', view: inference, type: 'f32', access: 'read_write'}
        ],
        body: `let observed=indices[indicesOffset+3u]-indices[indicesOffset];var valid=0u;var extreme=0u;for(var replicate=0u;replicate<${options.replicates}u;replicate++){let value=distribution[distributionOffset+replicate];if(isFiniteValue(value)){valid++;extreme+=u32(abs(value)>=abs(observed));}}let complete=valid==${options.replicates}u&&isFiniteValue(observed);let missing=bitcast<f32>(0x7fc00000u | (index & 0u));output[outputOffset]=observed;output[outputOffset+1u]=select(missing,f32(extreme+1u)/f32(valid+1u),complete);output[outputOffset+2u]=f32(valid);output[outputOffset+3u]=f32(extreme);`
      })
    );
    randomization = {distribution, inference};
  }
  return {
    outputs: {indices, validity, decomposition},
    randomization,
    provenance: {
      supportId: props.supportId,
      counterfactualMethod: props.counterfactualMethod,
      populations: props.populations,
      statistic: 'dissimilarity' as const,
      decomposition: 'two-factor-shapley' as const,
      randomization: props.randomization
        ? {
            seed: props.randomization.seed,
            replicates: props.randomization.replicates,
            nullModel: 'paired-support-label-exchange',
            pValue: 'two-sided-plus-one',
            generator: 'Philox4x32-10'
          }
        : undefined
    }
  };
}

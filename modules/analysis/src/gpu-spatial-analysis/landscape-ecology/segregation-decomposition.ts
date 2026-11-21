// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {captureUrbanLandscapeDestinations} from './urban-landscape-destinations';
import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateUrbanLandscapeRecipeViews} from './urban-landscape-recipe-validation';
import {PHILOX_WGSL, PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Most randomization replicates: each replicate is a thread that sweeps every unit twice. */
export const SEGREGATION_DECOMPOSITION_MAXIMUM_REPLICATES = 65536;

/** Largest random seed: the Philox key word is a uint32. */
const SEGREGATION_DECOMPOSITION_MAXIMUM_SEED = 0xffffffff;

export type GPUSegregationDecompositionOutput = {
  /** `float32`, 4 rows: dissimilarity of baseline, composition-only, spatial-only and final. */
  indices: GraphDataView<'float32'>;
  /** `uint32`, 4 rows: one when the matching index is valid. */
  validity: GraphDataView<'uint32'>;
  /** `float32x4`, one row: change, composition term, spatial term, residual. */
  decomposition: GraphDataView<'float32x4'>;
  /** `float32`, one row per replicate, only with `randomization`: paired exchange change in D. */
  distribution: GraphDataView<'float32'>;
  /** `float32x4`, one row, only with `randomization`: observed change, p value, valid, extreme. */
  inference: GraphDataView<'float32x4'>;
};

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
  /**
   * Caller-owned result views. Any view that is absent is a graph-owned transient view, as before.
   * Formats and lengths are those of {@link GPUSegregationDecompositionOutput}.
   */
  output?: Partial<GPUSegregationDecompositionOutput>;
};

/** Two-factor Shapley decomposition of the dissimilarity-index change, without randomization inference. */
export function addSegregationDecompositionRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSegregationDecompositionProps
) {
  const id = props.id ?? 'segregation-decomposition';
  validateRequirement(id, !('outputs' in props), 'outputs was removed; use output');
  // Every rejection happens here, before the first graph.add (RCP-101).
  validateRequirement(
    id,
    Array.isArray(props.populations) && props.populations.length === 4,
    'populations must have 4 entries'
  );
  const count = props.populations[0].length;
  validateRequirement(id, count > 0, 'populations must have at least 1 row');
  for (const population of props.populations) {
    validateSinglePackedView(population, ['float32x2'], `${id} populations entry`);
    validateViewLength(id, 'populations entry', population, count);
  }
  validateRequirement(id, props.supportId.length > 0, 'supportId must not be empty');
  validateRequirement(
    id,
    props.counterfactualMethod.length > 0,
    'counterfactualMethod must not be empty'
  );
  const options = props.randomization;
  if (options) {
    validateCapacity(
      id,
      'randomization.replicates',
      options.replicates,
      1,
      SEGREGATION_DECOMPOSITION_MAXIMUM_REPLICATES
    );
    validateCapacity(
      id,
      'randomization.seed',
      options.seed,
      0,
      SEGREGATION_DECOMPOSITION_MAXIMUM_SEED
    );
    if (options.swaps) {
      validateSinglePackedView(options.swaps, ['uint32'], `${id} randomization.swaps`);
      validateViewLength(id, 'randomization.swaps', options.swaps, count * options.replicates);
    }
  }
  validateGraphViewsBelongToGraph(id, graph, [...props.populations, options?.swaps]);
  const output = captureUrbanLandscapeDestinations(id, {output: props.output}, [
    'indices',
    'validity',
    'decomposition',
    'distribution',
    'inference'
  ]);
  const inputs = {
    baseline: props.populations[0],
    compositionOnly: props.populations[1],
    spatialOnly: props.populations[2],
    final: props.populations[3],
    swaps: options?.swaps
  };
  validateUrbanLandscapeRecipeViews(
    graph,
    id,
    inputs,
    {
      indices: {view: output?.indices, format: 'float32', length: 4},
      validity: {view: output?.validity, format: 'uint32', length: 4},
      decomposition: {view: output?.decomposition, format: 'float32x4', length: 1},
      ...(options
        ? ({
            distribution: {
              view: output?.distribution,
              format: 'float32',
              length: options.replicates
            },
            inference: {view: output?.inference, format: 'float32x4', length: 1}
          } as const)
        : {})
    },
    [
      {reads: {}, writes: {indices: output?.indices, validity: output?.validity}},
      {reads: {indices: output?.indices}, writes: {decomposition: output?.decomposition}},
      ...(options
        ? [
            {
              reads: {baseline: inputs.baseline, final: inputs.final, swaps: inputs.swaps},
              writes: {distribution: output?.distribution}
            },
            {reads: {indices: output?.indices}, writes: {inference: output?.inference}}
          ]
        : [])
    ]
  );
  const indices = getRecipeView(graph, `${id}-indices`, 'float32', 4, output?.indices);
  const validity = getRecipeView(graph, `${id}-validity`, 'uint32', 4, output?.validity);
  const decomposition = getRecipeView(
    graph,
    `${id}-decomposition`,
    'float32x4',
    1,
    output?.decomposition
  );
  // Every index is built from reductions over the units: unit terms, totals, then the shares.
  for (let scenario = 0; scenario < 4; scenario++) {
    const prefix = `${id}-index-${scenario}`;
    const focal = createTransientView(graph, `${prefix}-focal`, 'float32', count);
    const complement = createTransientView(graph, `${prefix}-complement`, 'float32', count);
    const invalidUnits = createTransientView(graph, `${prefix}-invalid-units`, 'uint32', count);
    const dissimilarityTerms = createTransientView(graph, `${prefix}-terms`, 'float32', count);
    const focalTotal = createTransientView(graph, `${prefix}-focal-total`, 'float32', 1);
    const complementTotal = createTransientView(graph, `${prefix}-complement-total`, 'float32', 1);
    const invalidTotal = createTransientView(graph, `${prefix}-invalid-total`, 'uint32', 1);
    const dissimilarityTotal = createTransientView(graph, `${prefix}-total`, 'float32', 1);
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${prefix}-units`,
        operation: id,
        invocationCount: count,
        declarations: IS_FINITE_VALUE_WGSL,
        bindings: [
          {name: 'population', view: props.populations[scenario], type: 'f32', access: 'read'},
          {name: 'focal', view: focal, type: 'f32', access: 'read_write'},
          {name: 'complement', view: complement, type: 'f32', access: 'read_write'},
          {name: 'invalidUnits', view: invalidUnits, type: 'u32', access: 'read_write'}
        ],
        body: `let counts=vec2<f32>(population[populationOffset+index*2u],population[populationOffset+index*2u+1u]);
focal[focalOffset+index]=counts.x;
complement[complementOffset+index]=counts.y;
invalidUnits[invalidUnitsOffset+index]=u32(!(isFiniteValue(counts.x)&&isFiniteValue(counts.y)&&all(counts>=vec2<f32>(0.0))));`
      })
    );
    for (const [name, input, total] of [
      ['focal', focal, focalTotal],
      ['complement', complement, complementTotal]
    ] as const) {
      graph.add(
        new GPUReduction({
          id: `${prefix}-${name}-sum`,
          input,
          output: total,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    graph.add(
      new GPUReduction({
        id: `${prefix}-invalid-sum`,
        input: invalidUnits,
        output: invalidTotal,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${prefix}-shares`,
        operation: id,
        invocationCount: count,
        bindings: [
          {name: 'focal', view: focal, type: 'f32', access: 'read'},
          {name: 'complement', view: complement, type: 'f32', access: 'read'},
          {name: 'focalTotal', view: focalTotal, type: 'f32', access: 'read'},
          {name: 'complementTotal', view: complementTotal, type: 'f32', access: 'read'},
          {name: 'terms', view: dissimilarityTerms, type: 'f32', access: 'read_write'}
        ],
        body: `let shares=vec2<f32>(focal[focalOffset+index],complement[complementOffset+index])/vec2<f32>(focalTotal[focalTotalOffset],complementTotal[complementTotalOffset]);
terms[termsOffset+index]=0.5*abs(shares.x-shares.y);`
      })
    );
    graph.add(
      new GPUReduction({
        id: `${prefix}-dissimilarity-sum`,
        input: dissimilarityTerms,
        output: dissimilarityTotal,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: prefix,
        operation: id,
        invocationCount: 1,
        declarations: IS_FINITE_VALUE_WGSL,
        bindings: [
          {name: 'invalidTotal', view: invalidTotal, type: 'u32', access: 'read'},
          {name: 'focalTotal', view: focalTotal, type: 'f32', access: 'read'},
          {name: 'complementTotal', view: complementTotal, type: 'f32', access: 'read'},
          {name: 'dissimilarityTotal', view: dissimilarityTotal, type: 'f32', access: 'read'},
          {name: 'indices', view: indices, type: 'f32', access: 'read_write'},
          {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
        ],
        body: `let totals=vec2<f32>(focalTotal[focalTotalOffset],complementTotal[complementTotalOffset]);
let valid=invalidTotal[invalidTotalOffset]==0u&&all(totals>vec2<f32>(0.0))&&isFiniteValue(totals.x)&&isFiniteValue(totals.y);
indices[indicesOffset+${scenario}u]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),dissimilarityTotal[dissimilarityTotalOffset],valid);
validity[validityOffset+${scenario}u]=u32(valid);`
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
  if (options) {
    const distribution = getRecipeView(
      graph,
      `${id}-randomization`,
      'float32',
      options.replicates,
      output?.distribution
    );
    const inference = getRecipeView(graph, `${id}-inference`, 'float32x4', 1, output?.inference);
    const serialObserved = createTransientView(graph, `${id}-serial-observed`, 'float32', 1);
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-paired-randomization`,
        operation: id,
        invocationCount: options.replicates + 1,
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
          {name: 'output', view: distribution, type: 'f32', access: 'read_write'},
          {name: 'serialObserved', view: serialObserved, type: 'f32', access: 'read_write'}
        ],
        // One exchange draw per unit and pass serves both scenarios (two Philox blocks per unit
        // instead of five), with the same stream as before. The extra last thread runs the same
        // code without any exchange: the observed change summed in the replicates' serial order,
        // so an exchange that leaves the tables unchanged in exact arithmetic (no swap, every
        // swap, or swaps of units whose baseline and final rows are equal) ties it bit for bit
        // (URB-101). The tree-reduced indices differ from serial sums by a few float32 ulps.
        body: `var values=vec2<f32>(0.0);var valid=true;var totals0=vec2<f32>(0.0);var totals1=vec2<f32>(0.0);
let exchange=index<${options.replicates}u;let replicate=min(index,${options.replicates - 1}u);
for(var unit=0u;unit<${count}u;unit++){let swap=select(0u,swapAt(unit,replicate),exchange);valid=valid&&swap<=1u;let population0=populationAt(unit,0u,swap);let population1=populationAt(unit,1u,swap);valid=valid&&all(population0>=vec2<f32>(0.0))&&isFiniteValue(population0.x)&&isFiniteValue(population0.y)&&all(population1>=vec2<f32>(0.0))&&isFiniteValue(population1.x)&&isFiniteValue(population1.y);totals0+=population0;totals1+=population1;}
valid=valid&&all(totals0>vec2<f32>(0.0))&&isFiniteValue(totals0.x)&&isFiniteValue(totals0.y)&&all(totals1>vec2<f32>(0.0))&&isFiniteValue(totals1.x)&&isFiniteValue(totals1.y);
for(var unit=0u;unit<${count}u;unit++){let swap=select(0u,swapAt(unit,replicate),exchange);let shares0=populationAt(unit,0u,swap)/totals0;let shares1=populationAt(unit,1u,swap)/totals1;values.x+=0.5*abs(shares0.x-shares0.y);values.y+=0.5*abs(shares1.x-shares1.y);}
let value=select(bitcast<f32>(0x7fc00000u | (index & 0u)),values.y-values.x,valid);
if(exchange){output[outputOffset+index]=value;}else{serialObserved[serialObservedOffset]=value;}`
      })
    );
    // Valid and extreme replicates are counted by reductions over per-replicate flags.
    const validReplicates = createTransientView(
      graph,
      `${id}-valid-flags`,
      'uint32',
      options.replicates
    );
    const extremeReplicates = createTransientView(
      graph,
      `${id}-extreme-flags`,
      'uint32',
      options.replicates
    );
    const validTotal = createTransientView(graph, `${id}-valid-total`, 'uint32', 1);
    const extremeTotal = createTransientView(graph, `${id}-extreme-total`, 'uint32', 1);
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-randomization-flags`,
        operation: id,
        invocationCount: options.replicates,
        declarations: IS_FINITE_VALUE_WGSL,
        bindings: [
          {name: 'serialObserved', view: serialObserved, type: 'f32', access: 'read'},
          {name: 'distribution', view: distribution, type: 'f32', access: 'read'},
          {name: 'validFlags', view: validReplicates, type: 'u32', access: 'read_write'},
          {name: 'extremeFlags', view: extremeReplicates, type: 'u32', access: 'read_write'}
        ],
        body: `let observed=serialObserved[serialObservedOffset];
let value=distribution[distributionOffset+index];
let valid=isFiniteValue(value);
validFlags[validFlagsOffset+index]=u32(valid);
extremeFlags[extremeFlagsOffset+index]=u32(valid&&abs(value)>=abs(observed));`
      })
    );
    graph.add(
      new GPUReduction({
        id: `${id}-valid-replicates`,
        input: validReplicates,
        output: validTotal,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
    graph.add(
      new GPUReduction({
        id: `${id}-extreme-replicates`,
        input: extremeReplicates,
        output: extremeTotal,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-randomization-inference`,
        operation: id,
        invocationCount: 1,
        declarations: IS_FINITE_VALUE_WGSL,
        bindings: [
          {name: 'indices', view: indices, type: 'f32', access: 'read'},
          {name: 'validTotal', view: validTotal, type: 'u32', access: 'read'},
          {name: 'extremeTotal', view: extremeTotal, type: 'u32', access: 'read'},
          {name: 'output', view: inference, type: 'f32', access: 'read_write'}
        ],
        body: `let observed=indices[indicesOffset+3u]-indices[indicesOffset];
let valid=validTotal[validTotalOffset];
let extreme=extremeTotal[extremeTotalOffset];
let complete=valid==${options.replicates}u&&isFiniteValue(observed);
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
output[outputOffset]=observed;
output[outputOffset+1u]=select(missing,f32(extreme+1u)/f32(valid+1u),complete);
output[outputOffset+2u]=f32(valid);
output[outputOffset+3u]=f32(extreme);`
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

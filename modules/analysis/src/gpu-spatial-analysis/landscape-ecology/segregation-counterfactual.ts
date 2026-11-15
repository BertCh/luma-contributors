// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {RecipeBuilder} from '../recipes/recipe-utils';
import {
  GPUPopulationSupportAlignment,
  type GPUPopulationSupportAlignmentProps
} from './gpu-population-support-alignment';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  addSegregationDecompositionRecipe,
  type GPUSegregationDecompositionOutput
} from './segregation-decomposition';

/**
 * Most units: one thread insertion-sorts the units' shares and ranks every unit against all of
 * them, which is about this number squared steps (4 million at the limit), and the scratch holds
 * six floats per unit.
 */
export const SEGREGATION_COUNTERFACTUAL_MAXIMUM_UNIT_COUNT = 2048;

// The recipe never forwards `distribution` and `inference`, so they are not accepted as outputs.
export type GPUSegregationCounterfactualOutput = Omit<
  GPUSegregationDecompositionOutput,
  'distribution' | 'inference'
> & {
  /**
   * `float32x2`, one row per unit: the final composition on the baseline spatial structure (the
   * baseline units keep their ranks and totals and take the final table's composition
   * distribution; PySAL `counterfac_df1`).
   */
  compositionOnly: GraphDataView<'float32x2'>;
  /**
   * `float32x2`, one row per unit: the baseline composition on the final spatial structure (the
   * final units keep their ranks and totals and take the baseline table's composition
   * distribution; PySAL `counterfac_df2`).
   */
  spatialOnly: GraphDataView<'float32x2'>;
};

export type GPUSegregationCounterfactualProps = {
  id?: string;
  baseline: GraphDataView<'float32x2'>;
  final: GraphDataView<'float32x2'>;
  supportId: string;
  /** PySAL empirical-CDF counterfactual conventions, including average ties and linear quantiles. */
  method: 'composition' | 'dual-composition' | 'share';
  /**
   * Caller-owned result views. Any view that is absent is a graph-owned transient view, as before.
   * Formats and lengths are those of {@link GPUSegregationCounterfactualOutput}; the decomposition
   * views are passed to the decomposition this recipe adds.
   */
  output?: Partial<GPUSegregationCounterfactualOutput>;
};

/** Generates both replacement orders before the existing Shapley decomposition, without host readback. */
export function addSegregationCounterfactualRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSegregationCounterfactualProps
) {
  const id = props.id ?? 'segregation-counterfactual';
  const count = props.baseline.length;
  validateCapacity(id, 'baseline rows', count, 1, SEGREGATION_COUNTERFACTUAL_MAXIMUM_UNIT_COUNT);
  validateViewLength(id, 'final', props.final, count);
  validateRequirement(id, props.supportId.length > 0, 'supportId must not be empty');
  const output = props.output;
  const compositionOnly = getRecipeView(
    graph,
    `${id}-composition-only`,
    'float32x2',
    count,
    output?.compositionOnly
  );
  const spatialOnly = getRecipeView(
    graph,
    `${id}-spatial-only`,
    'float32x2',
    count,
    output?.spatialOnly
  );
  for (let scenario = 0; scenario < 2; scenario++) {
    const own = scenario === 0 ? props.baseline : props.final;
    const other = scenario === 0 ? props.final : props.baseline;
    const output = scenario === 0 ? compositionOnly : spatialOnly;
    const scratch = getRecipeView(graph, `${id}-scratch-${scenario}`, 'float32', 6 * count);
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-quantile-${scenario}`,
        operation: id,
        invocationCount: 1,
        workgroupSize: 2,
        // Serial by design: one thread sorts and ranks all units, bounded by
        // SEGREGATION_COUNTERFACTUAL_MAXIMUM_UNIT_COUNT. A GPUSort of the shares plus a rank
        // search per unit would remove it.
        declarations:
          IS_FINITE_VALUE_WGSL +
          `
fn readOwn(unit:u32)->vec2<f32>{return vec2<f32>(scratch[scratchOffset+2u*unit],scratch[scratchOffset+2u*unit+1u]);}
fn readOther(unit:u32,group:u32)->f32{return scratch[scratchOffset+${2 * count}u+2u*unit+group];}
fn writeOther(unit:u32,group:u32,value:f32){scratch[scratchOffset+${2 * count}u+2u*unit+group]=value;}
fn readMapped(unit:u32)->vec2<f32>{return vec2<f32>(scratch[scratchOffset+${4 * count}u+2u*unit],scratch[scratchOffset+${4 * count}u+2u*unit+1u]);}
`,
        bindings: [
          {name: 'own', view: own, type: 'f32', access: 'read'},
          {name: 'other', view: other, type: 'f32', access: 'read'},
          {name: 'output', view: output, type: 'f32', access: 'read_write'},
          {name: 'scratch', view: scratch, type: 'f32', access: 'read_write'}
        ],
        body: `var ownTotal=vec2<f32>(0.0);var otherTotal=vec2<f32>(0.0);var valid=true;
for(var unit=0u;unit<${count}u;unit++){let first=vec2<f32>(own[ownOffset+2u*unit],own[ownOffset+2u*unit+1u]);let second=vec2<f32>(other[otherOffset+2u*unit],other[otherOffset+2u*unit+1u]);valid=valid&&all(first>=vec2<f32>(0.0))&&all(second>=vec2<f32>(0.0))&&isFiniteValue(first.x)&&isFiniteValue(first.y)&&isFiniteValue(second.x)&&isFiniteValue(second.y);ownTotal+=first;otherTotal+=second;}
valid=valid&&all(ownTotal>vec2<f32>(0.0))&&all(otherTotal>vec2<f32>(0.0))&&isFiniteValue(ownTotal.x)&&isFiniteValue(ownTotal.y)&&isFiniteValue(otherTotal.x)&&isFiniteValue(otherTotal.y);
for(var unit=0u;unit<${count}u;unit++){let first=vec2<f32>(own[ownOffset+2u*unit],own[ownOffset+2u*unit+1u]);let second=vec2<f32>(other[otherOffset+2u*unit],other[otherOffset+2u*unit+1u]);
${props.method === 'share' ? 'let firstValue=first/ownTotal;let secondValue=second/otherTotal;' : 'let firstValue=select(vec2<f32>(0.0),first/(first.x+first.y),first.x+first.y>0.0);let secondValue=select(vec2<f32>(0.0),second/(second.x+second.y),second.x+second.y>0.0);'}
scratch[scratchOffset+2u*unit]=firstValue.x;scratch[scratchOffset+2u*unit+1u]=firstValue.y;writeOther(unit,0u,secondValue.x);writeOther(unit,1u,secondValue.y);
}
for(var group=0u;group<2u;group++){for(var unit=1u;unit<${count}u;unit++){let value=readOther(unit,group);var insertion=unit;loop{if(insertion==0u||readOther(insertion-1u,group)<=value){break;}writeOther(insertion,group,readOther(insertion-1u,group));insertion--;}writeOther(insertion,group,value);}}
var correction=vec2<f32>(0.0);
for(var unit=0u;unit<${count}u;unit++){for(var group=0u;group<2u;group++){var less=0.0;var equal=0.0;let value=readOwn(unit)[group];for(var candidate=0u;candidate<${count}u;candidate++){less+=f32(readOwn(candidate)[group]<value);equal+=f32(readOwn(candidate)[group]==value);}let rank=(less+0.5*(equal+1.0))/${count}.0;let position=rank*${count - 1}.0;let lower=u32(floor(position));let upper=min(lower+1u,${count - 1}u);scratch[scratchOffset+${4 * count}u+2u*unit+group]=mix(readOther(lower,group),readOther(upper,group),fract(position));}correction+=readMapped(unit);}
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
for(var unit=0u;unit<${count}u;unit++){let total=own[ownOffset+2u*unit]+own[ownOffset+2u*unit+1u];var result=readMapped(unit)*total;
${props.method === 'composition' ? 'result.y=total-result.x;' : props.method === 'share' ? 'result=readMapped(unit)/correction*ownTotal;' : ''}
output[outputOffset+2u*unit]=select(missing,result.x,valid);output[outputOffset+2u*unit+1u]=select(missing,result.y,valid);}`
      })
    );
  }
  const result = addSegregationDecompositionRecipe(graph, {
    id: `${id}-decomposition`,
    populations: [props.baseline, compositionOnly, spatialOnly, props.final],
    supportId: props.supportId,
    counterfactualMethod: props.method,
    output: {
      indices: output?.indices,
      validity: output?.validity,
      decomposition: output?.decomposition
    }
  });
  return {
    ...result,
    counterfactuals: {compositionOnly, spatialOnly},
    provenance: {
      ...result.provenance,
      quantile: 'linear' as const,
      ties: 'average-percentile' as const,
      maximumUnits: SEGREGATION_COUNTERFACTUAL_MAXIMUM_UNIT_COUNT
    }
  };
}

export type {
  GPUPopulationSupportAlignmentOutput,
  GPUPopulationSupportAlignmentProps
} from './gpu-population-support-alignment';

/** Exact ID alignment. Prefer the CPU operation when support data is already on the host. */
export function addPopulationSupportAlignmentRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPopulationSupportAlignmentProps
) {
  const id = props.id ?? 'population-support-alignment';
  // Preserve legacy validation order and caller-owned/transient output overrides.
  validateRequirement(id, props.populations.length > 0, 'populations must have at least 1 row');
  validateViewLength(id, 'sourceIds', props.sourceIds, props.populations.length);
  validateRequirement(id, props.targetIds.length > 0, 'targetIds must have at least 1 row');
  const populations = getRecipeView(
    graph,
    `${id}-populations`,
    'float32x2',
    props.targetIds.length,
    props.output?.populations
  );
  const validity = getRecipeView(
    graph,
    `${id}-validity`,
    'uint32',
    props.targetIds.length,
    props.output?.validity
  );
  const status = getRecipeView(graph, `${id}-status`, 'uint32x4', 1, props.output?.status);
  const builder = new RecipeBuilder(graph);
  builder.add(
    new GPUPopulationSupportAlignment({...props, id, output: {populations, validity, status}})
  );
  return {
    contributors: builder.contributors,
    outputs: {populations, validity},
    status,
    provenance: {
      sourceIds: props.sourceIds,
      targetIds: props.targetIds,
      alignment: 'exact-identifier-join' as const
    }
  };
}

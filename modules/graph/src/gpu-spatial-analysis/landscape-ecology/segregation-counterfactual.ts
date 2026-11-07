// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {addSegregationDecompositionRecipe} from './segregation-decomposition';

export type GPUSegregationCounterfactualProps = {
  id?: string;
  baseline: GraphDataView<'float32x2'>;
  final: GraphDataView<'float32x2'>;
  supportId: string;
  /** PySAL empirical-CDF counterfactual conventions, including average ties and linear quantiles. */
  method: 'composition' | 'dual-composition' | 'share';
};

/** Generates both replacement orders before the existing Shapley decomposition, without host readback. */
export function addSegregationCounterfactualRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSegregationCounterfactualProps
) {
  const id = props.id ?? 'segregation-counterfactual';
  const count = props.baseline.length;
  assert(count > 0 && count <= 2048 && props.final.length === count && props.supportId.length > 0);
  const compositionOnly = getRecipeView(graph, `${id}-composition-only`, 'float32x2', count);
  const spatialOnly = getRecipeView(graph, `${id}-spatial-only`, 'float32x2', count);
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
    counterfactualMethod: props.method
  });
  return {
    ...result,
    counterfactuals: {compositionOnly, spatialOnly},
    provenance: {
      ...result.provenance,
      quantile: 'linear' as const,
      ties: 'average-percentile' as const,
      maximumUnits: 2048
    }
  };
}

export type GPUPopulationSupportAlignmentProps = {
  id?: string;
  populations: GraphDataView<'float32x2'>;
  sourceIds: GraphDataView<'uint32'>;
  targetIds: GraphDataView<'uint32'>;
};
/** Explicit ID join: duplicate/missing source support is invalid, never silently aggregated or interpolated. */
export function addPopulationSupportAlignmentRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPopulationSupportAlignmentProps
) {
  const id = props.id ?? 'population-support-alignment';
  assert(
    props.populations.length > 0 &&
      props.sourceIds.length === props.populations.length &&
      props.targetIds.length > 0
  );
  const populations = getRecipeView(
    graph,
    `${id}-populations`,
    'float32x2',
    props.targetIds.length
  );
  const validity = getRecipeView(graph, `${id}-validity`, 'uint32', props.targetIds.length);
  const status = getRecipeView(graph, `${id}-status`, 'uint32x4', 1);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-align`,
      operation: id,
      invocationCount: props.targetIds.length,
      bindings: [
        {name: 'populations', view: props.populations, type: 'f32', access: 'read'},
        {name: 'sourceIds', view: props.sourceIds, type: 'u32', access: 'read'},
        {name: 'targetIds', view: props.targetIds, type: 'u32', access: 'read'},
        {name: 'output', view: populations, type: 'f32', access: 'read_write'},
        {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
      ],
      body: `let targetId=targetIds[targetIdsOffset+index];var matches=0u;var source=0u;var duplicate=false;for(var candidate=0u;candidate<${props.sourceIds.length}u;candidate++){if(sourceIds[sourceIdsOffset+candidate]==targetId){matches++;source=candidate;}}for(var candidate=0u;candidate<${props.targetIds.length}u;candidate++){duplicate=duplicate||(candidate!=index&&targetIds[targetIdsOffset+candidate]==targetId);}let valid=matches==1u&&!duplicate;let missing=bitcast<f32>(0x7fc00000u | (index & 0u));output[outputOffset+2u*index]=select(missing,populations[populationsOffset+2u*source],valid);output[outputOffset+2u*index+1u]=select(missing,populations[populationsOffset+2u*source+1u],valid);validity[validityOffset+index]=select(select(2u,1u,matches==0u),0u,valid);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-certificate`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'sourceIds', view: props.sourceIds, type: 'u32', access: 'read'},
        {name: 'targetIds', view: props.targetIds, type: 'u32', access: 'read'},
        {name: 'validity', view: validity, type: 'u32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read_write'}
      ],
      body: `var missing=0u;var duplicates=0u;var excluded=0u;for(var targetRow=0u;targetRow<${props.targetIds.length}u;targetRow++){missing+=u32(validity[validityOffset+targetRow]==1u);duplicates+=u32(validity[validityOffset+targetRow]==2u);}for(var source=0u;source<${props.sourceIds.length}u;source++){var found=false;for(var targetRow=0u;targetRow<${props.targetIds.length}u;targetRow++){found=found||sourceIds[sourceIdsOffset+source]==targetIds[targetIdsOffset+targetRow];}excluded+=u32(!found);}status[statusOffset]=u32(missing==0u&&duplicates==0u&&excluded==0u);status[statusOffset+1u]=missing;status[statusOffset+2u]=duplicates;status[statusOffset+3u]=excluded;`
    })
  );
  return {
    outputs: {populations, validity},
    status,
    provenance: {
      sourceIds: props.sourceIds,
      targetIds: props.targetIds,
      alignment: 'exact-identifier-join' as const
    }
  };
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
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
  /** `float32x2`, one row per unit: baseline composition on the final spatial structure. */
  compositionOnly: GraphDataView<'float32x2'>;
  /** `float32x2`, one row per unit: final composition on the baseline spatial structure. */
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

export type GPUPopulationSupportAlignmentOutput = {
  /** `float32x2`, one row per target ID: the matching source population, or NaN. */
  populations: GraphDataView<'float32x2'>;
  /** `uint32`, one row per target ID: 0 matched, 1 absent from the sources, 2 duplicated target. */
  validity: GraphDataView<'uint32'>;
  /** `uint32x4`, one row: `[aligned, missingCount, duplicateCount, excludedSourceCount]`. */
  status: GraphDataView<'uint32x4'>;
};

export type GPUPopulationSupportAlignmentProps = {
  id?: string;
  populations: GraphDataView<'float32x2'>;
  sourceIds: GraphDataView<'uint32'>;
  targetIds: GraphDataView<'uint32'>;
  /**
   * Caller-owned result views. Any view that is absent is a graph-owned transient view, as before.
   * Formats and lengths are those of {@link GPUPopulationSupportAlignmentOutput}.
   */
  output?: Partial<GPUPopulationSupportAlignmentOutput>;
};
/** Explicit ID join: duplicate/missing source support is invalid, never silently aggregated or interpolated. */
export function addPopulationSupportAlignmentRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPopulationSupportAlignmentProps
) {
  const id = props.id ?? 'population-support-alignment';
  validateRequirement(id, props.populations.length > 0, 'populations must have at least 1 row');
  validateViewLength(id, 'sourceIds', props.sourceIds, props.populations.length);
  validateRequirement(id, props.targetIds.length > 0, 'targetIds must have at least 1 row');
  const output = props.output;
  const populations = getRecipeView(
    graph,
    `${id}-populations`,
    'float32x2',
    props.targetIds.length,
    output?.populations
  );
  const validity = getRecipeView(
    graph,
    `${id}-validity`,
    'uint32',
    props.targetIds.length,
    output?.validity
  );
  const status = getRecipeView(graph, `${id}-status`, 'uint32x4', 1, output?.status);
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
  // The certificate counts per-row flags with reductions: missing and duplicated targets, and
  // sources that no target names (one search over the targets per source row).
  const missingFlags = createTransientView(
    graph,
    `${id}-missing-flags`,
    'uint32',
    props.targetIds.length
  );
  const duplicateFlags = createTransientView(
    graph,
    `${id}-duplicate-flags`,
    'uint32',
    props.targetIds.length
  );
  const excludedFlags = createTransientView(
    graph,
    `${id}-excluded-flags`,
    'uint32',
    props.sourceIds.length
  );
  const missingTotal = createTransientView(graph, `${id}-missing-total`, 'uint32', 1);
  const duplicateTotal = createTransientView(graph, `${id}-duplicate-total`, 'uint32', 1);
  const excludedTotal = createTransientView(graph, `${id}-excluded-total`, 'uint32', 1);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-target-flags`,
      operation: id,
      invocationCount: props.targetIds.length,
      bindings: [
        {name: 'validity', view: validity, type: 'u32', access: 'read'},
        {name: 'missingFlags', view: missingFlags, type: 'u32', access: 'read_write'},
        {name: 'duplicateFlags', view: duplicateFlags, type: 'u32', access: 'read_write'}
      ],
      body: `let code=validity[validityOffset+index];
missingFlags[missingFlagsOffset+index]=u32(code==1u);
duplicateFlags[duplicateFlagsOffset+index]=u32(code==2u);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-source-flags`,
      operation: id,
      invocationCount: props.sourceIds.length,
      bindings: [
        {name: 'sourceIds', view: props.sourceIds, type: 'u32', access: 'read'},
        {name: 'targetIds', view: props.targetIds, type: 'u32', access: 'read'},
        {name: 'excludedFlags', view: excludedFlags, type: 'u32', access: 'read_write'}
      ],
      body: `var found=false;
for(var targetRow=0u;targetRow<${props.targetIds.length}u;targetRow++){
  found=found||sourceIds[sourceIdsOffset+index]==targetIds[targetIdsOffset+targetRow];
}
excludedFlags[excludedFlagsOffset+index]=u32(!found);`
    })
  );
  for (const [name, input, total] of [
    ['missing', missingFlags, missingTotal],
    ['duplicate', duplicateFlags, duplicateTotal],
    ['excluded', excludedFlags, excludedTotal]
  ] as const) {
    graph.add(
      new GPUReduction({
        id: `${id}-${name}-count`,
        input,
        output: total,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
  }
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-certificate`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'missingTotal', view: missingTotal, type: 'u32', access: 'read'},
        {name: 'duplicateTotal', view: duplicateTotal, type: 'u32', access: 'read'},
        {name: 'excludedTotal', view: excludedTotal, type: 'u32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read_write'}
      ],
      body: `let missing=missingTotal[missingTotalOffset];
let duplicates=duplicateTotal[duplicateTotalOffset];
let excluded=excludedTotal[excludedTotalOffset];
status[statusOffset]=u32(missing==0u&&duplicates==0u&&excluded==0u);
status[statusOffset+1u]=missing;
status[statusOffset+2u]=duplicates;
status[statusOffset+3u]=excluded;`
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

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUReduction, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUPointPatternEnvelopeProps} from './gpu-point-pattern-envelope';
import {getPointPatternRejectionBudget} from './point-pattern-rejection-budget';

/** Exchangeable two-sided extreme midrank test; all ranks include the observed curve. */
export function createRankEnvelopeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPointPatternEnvelopeProps & {simulationCount: number},
  id: string
) {
  const count = props.observed.length;
  const curves = props.simulationCount + 1;
  const ranks = createTransientView(graph, `${id}-extreme-ranks`, 'float32', curves);
  const critical = createTransientView(graph, `${id}-critical-rank`, 'float32', 1);
  const candidates = createTransientView(graph, `${id}-rank-candidates`, 'float32', curves);
  const strictFlags = createTransientView(graph, `${id}-rank-strict-flags`, 'uint32', curves);
  const inclusiveFlags = createTransientView(graph, `${id}-rank-inclusive-flags`, 'uint32', curves);
  const invalidFlags = createTransientView(graph, `${id}-rank-invalid-flags`, 'uint32', curves);
  const strictCount = createTransientView(graph, `${id}-rank-strict-count`, 'uint32', 1);
  const inclusiveCount = createTransientView(graph, `${id}-rank-inclusive-count`, 'uint32', 1);
  const invalidCount = createTransientView(graph, `${id}-rank-invalid-count`, 'uint32', 1);
  const interval =
    props.output.pValueInterval ?? createTransientView(graph, `${id}-p-interval`, 'float32', 2);
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-extreme-ranks`,
      operation: 'GPUPointPatternEnvelope',
      variant: 'extreme-rank',
      invocationCount: curves,
      bindings: [
        {name: 'observed', view: props.observed, type: 'f32', access: 'read'},
        {name: 'simulations', view: props.simulations, type: 'f32', access: 'read'},
        {name: 'ranks', view: ranks, type: 'f32', access: 'read_write'}
      ],
      declarations: `${IS_FINITE_FLOAT_WGSL}
fn value(curve: u32, radius: u32) -> f32 {
  if (curve == ${props.simulationCount}u) { return observed[observedOffset + radius]; }
  return simulations[simulationsOffset + curve * ${count}u + radius];
}`,
      body: `var extreme = ${curves}.0; var valid = true;
for (var radius = 0u; radius < ${count}u; radius++) {
  let current = value(index, radius); valid = valid && isFiniteFloat(current);
  var less = 0u; var greater = 0u; var equal = 0u;
  for (var curve = 0u; curve < ${curves}u; curve++) {
    let other = value(curve,radius); valid = valid && isFiniteFloat(other);
    less += select(0u,1u,other < current); greater += select(0u,1u,other > current); equal += select(0u,1u,other == current);
  }
  extreme = min(extreme, f32(min(less,greater)) + 0.5 * f32(equal + 1u));
}
ranks[ranksOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), extreme, valid);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-rank-decisions`,
      operation: 'GPUPointPatternEnvelope',
      variant: 'extreme-rank-decisions',
      invocationCount: curves,
      bindings: [
        {name: 'ranks', view: ranks, type: 'f32', access: 'read'},
        {name: 'candidates', view: candidates, type: 'f32', access: 'read_write'},
        {name: 'strictFlags', view: strictFlags, type: 'u32', access: 'read_write'},
        {name: 'inclusiveFlags', view: inclusiveFlags, type: 'u32', access: 'read_write'},
        {name: 'invalidFlags', view: invalidFlags, type: 'u32', access: 'read_write'}
      ],
      declarations: IS_FINITE_FLOAT_WGSL,
      body: `let rank=ranks[ranksOffset+index];
let observedRank=ranks[ranksOffset+${props.simulationCount}u];
var rejected=0u;
for(var curve=0u;curve<${curves}u;curve++) {
 rejected+=select(0u,1u,ranks[ranksOffset+curve]<rank);
}
strictFlags[strictFlagsOffset+index]=select(0u,1u,rank<observedRank);
inclusiveFlags[inclusiveFlagsOffset+index]=select(0u,1u,rank<=observedRank);
invalidFlags[invalidFlagsOffset+index]=select(1u,0u,isFiniteFloat(rank));
candidates[candidatesOffset+index]=select(0.0,rank,isFiniteFloat(rank) && rejected<=${getPointPatternRejectionBudget(curves, props.significanceLevel ?? 0.05)}u);`
    }),
    ...new GPUReduction({
      id: `${id}-rank-strict-reduction`,
      input: strictFlags,
      output: strictCount,
      operation: 'sum'
    }).getCommandNodes(graph),
    ...new GPUReduction({
      id: `${id}-rank-inclusive-reduction`,
      input: inclusiveFlags,
      output: inclusiveCount,
      operation: 'sum'
    }).getCommandNodes(graph),
    ...new GPUReduction({
      id: `${id}-rank-invalid-reduction`,
      input: invalidFlags,
      output: invalidCount,
      operation: 'sum'
    }).getCommandNodes(graph),
    ...new GPUReduction({
      id: `${id}-rank-critical-reduction`,
      input: candidates,
      output: critical,
      operation: 'max'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-rank-test`,
      operation: 'GPUPointPatternEnvelope',
      variant: 'extreme-rank-global-test',
      invocationCount: 1,
      bindings: [
        {name: 'ranks', view: ranks, type: 'f32', access: 'read'},
        {name: 'strictCount', view: strictCount, type: 'u32', access: 'read'},
        {name: 'inclusiveCount', view: inclusiveCount, type: 'u32', access: 'read'},
        {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read'},
        {name: 'statistics', view: props.output.statistics, type: 'f32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
        {name: 'interval', view: interval, type: 'f32', access: 'read_write'}
      ],
      body: `let observedRank = ranks[ranksOffset + ${props.simulationCount}u];
let less=strictCount[strictCountOffset];
let inclusive=inclusiveCount[inclusiveCountOffset];
let valid=invalidCount[invalidCountOffset]==0u;
let invalid=bitcast<f32>(0x7fc00000u | (index & 0u));
statistics[statisticsOffset]=select(invalid,observedRank,valid);
statistics[statisticsOffset+1u]=select(invalid,f32(inclusive)/${curves}.0,valid);
statistics[statisticsOffset+2u]=1.0/${curves}.0; statistics[statisticsOffset+3u]=${props.simulationCount}.0;
interval[intervalOffset]=select(invalid,f32(less)/${curves}.0,valid);
interval[intervalOffset+1u]=select(invalid,f32(inclusive)/${curves}.0,valid);
status[statusOffset]=select(1u,0u,valid);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-rank-envelope`,
      operation: 'GPUPointPatternEnvelope',
      variant: 'global-rank-envelope',
      invocationCount: count,
      bindings: [
        {name: 'observed', view: props.observed, type: 'f32', access: 'read'},
        {name: 'simulations', view: props.simulations, type: 'f32', access: 'read'},
        {name: 'ranks', view: ranks, type: 'f32', access: 'read'},
        {name: 'critical', view: critical, type: 'f32', access: 'read'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
        {name: 'lower', view: props.output.lower, type: 'f32', access: 'read_write'},
        {name: 'upper', view: props.output.upper, type: 'f32', access: 'read_write'}
      ],
      declarations: `fn value(curve:u32,radius:u32)->f32 {if(curve==${props.simulationCount}u){return observed[observedOffset+radius];}return simulations[simulationsOffset+curve*${count}u+radius];}`,
      body: `var low=bitcast<f32>(0x7fc00000u | (index & 0u)); var high=low; var hasRetained=false;
if(status[statusOffset]==0u) {
 for(var curve=0u;curve<${curves}u;curve++) {
  if(ranks[ranksOffset+curve] >= critical[criticalOffset]) {
   let current=value(curve,index);
   if(!hasRetained) {low=current; high=current; hasRetained=true;}
   else {low=min(low,current); high=max(high,current);}
  }
 }
}
lower[lowerOffset+index]=low; upper[upperOffset+index]=high;`
    })
  ];
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUPointPatternEnvelopeProps} from './gpu-point-pattern-envelope';

/** Exchangeable two-sided extreme midrank test; all ranks include the observed curve. */
export function createRankEnvelopeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPointPatternEnvelopeProps,
  id: string
) {
  const count = props.observed.length;
  const curves = props.simulationCount + 1;
  const ranks = createTransientView(graph, `${id}-extreme-ranks`, 'float32', curves);
  const critical = createTransientView(graph, `${id}-critical-rank`, 'uint32', 1);
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
      id: `${id}-rank-test`,
      operation: 'GPUPointPatternEnvelope',
      variant: 'extreme-rank-global-test',
      invocationCount: 1,
      bindings: [
        {name: 'ranks', view: ranks, type: 'f32', access: 'read'},
        {name: 'statistics', view: props.output.statistics, type: 'f32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
        {name: 'interval', view: interval, type: 'f32', access: 'read_write'},
        {name: 'critical', view: critical, type: 'u32', access: 'read_write'}
      ],
      declarations: IS_FINITE_FLOAT_WGSL,
      body: `let observedRank = ranks[ranksOffset + ${props.simulationCount}u];
var less=0u; var inclusive=0u; var valid=true; var criticalRank=1u;
for (var curve=0u; curve<${curves}u; curve++) {
 let rank = ranks[ranksOffset+curve]; valid=valid && isFiniteFloat(rank);
 less += select(0u,1u,rank<observedRank); inclusive += select(0u,1u,rank<=observedRank);
}
for (var candidate=1u; candidate<=${Math.ceil(curves / 2)}u; candidate++) {
 var rejected=0u;
 for(var curve=0u; curve<${curves}u; curve++) { rejected += select(0u,1u,ranks[ranksOffset+curve]<f32(candidate)); }
 if (rejected<=${Math.floor((props.significanceLevel ?? 0.05) * curves)}u) {criticalRank=candidate;}
}
let invalid=bitcast<f32>(0x7fc00000u | (index & 0u));
statistics[statisticsOffset]=select(invalid,observedRank,valid);
statistics[statisticsOffset+1u]=select(invalid,f32(inclusive)/${curves}.0,valid);
statistics[statisticsOffset+2u]=1.0/${curves}.0; statistics[statisticsOffset+3u]=${props.simulationCount}.0;
interval[intervalOffset]=select(invalid,f32(less)/${curves}.0,valid);
interval[intervalOffset+1u]=select(invalid,f32(inclusive)/${curves}.0,valid);
critical[criticalOffset]=criticalRank; status[statusOffset]=select(1u,0u,valid);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-rank-envelope`,
      operation: 'GPUPointPatternEnvelope',
      variant: 'global-rank-envelope',
      invocationCount: count,
      bindings: [
        {name: 'observed', view: props.observed, type: 'f32', access: 'read'},
        {name: 'simulations', view: props.simulations, type: 'f32', access: 'read'},
        {name: 'critical', view: critical, type: 'u32', access: 'read'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
        {name: 'lower', view: props.output.lower, type: 'f32', access: 'read_write'},
        {name: 'upper', view: props.output.upper, type: 'f32', access: 'read_write'}
      ],
      declarations: `fn value(curve:u32,radius:u32)->f32 {if(curve==${props.simulationCount}u){return observed[observedOffset+radius];}return simulations[simulationsOffset+curve*${count}u+radius];}`,
      body: `var low=bitcast<f32>(0x7fc00000u | (index & 0u)); var high=low;
if(status[statusOffset]==0u) {
 for(var curve=0u;curve<${curves}u;curve++) {
  let current=value(curve,index); var rank=0u;
  for(var other=0u;other<${curves}u;other++) {let candidate=value(other,index); rank+=select(0u,1u,candidate<current || (candidate==current && other<curve));}
  if(rank+1u==critical[criticalOffset]) {low=current;}
  if(rank==${curves}u-critical[criticalOffset]) {high=current;}
 }
}
lower[lowerOffset+index]=low; upper[upperOffset+index]=high;`
    })
  ];
}

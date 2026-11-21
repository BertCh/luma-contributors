// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getRegressionTileLayout} from './regression-row-tiles';
import type {GPUSpatialPanelErrorMaximumLikelihoodProps} from './gpu-spatial-panel-error-maximum-likelihood';

const AFFINE_INPUT_BITS_WGSL = `
fn normalizePanelAffineBits(value: u32) -> u32 {
  return select(value, 0u, (value & 0x7fffffffu) == 0u);
}
fn isPanelAffineNormal(value: u32) -> bool {
  let magnitude = value & 0x7fffffffu;
  let exponent = magnitude >> 23u;
  return magnitude == 0u || (exponent > 0u && exponent < 255u);
}`;

// (b-a) cross (d-c), expanded into eight exact products. No rounded within means.
const PANEL_PARALLEL_WGSL = `
fn panelParallel(a:vec2f,b:vec2f,c:vec2f,d:vec2f)->bool {
  var left = array<f32,8>(b.x,b.x,a.x,a.x,b.y,b.y,a.y,a.y);
  var right = array<f32,8>(d.y,-c.y,-d.y,c.y,-d.x,c.x,d.x,-c.x);
  var minimum = 100000; var maximum = -100000;
  for(var term=0u;term<8u;term++) {
    let x=bitcast<u32>(left[term]) & 0x7fffffffu; let y=bitcast<u32>(right[term]) & 0x7fffffffu;
    if(x==0u || y==0u){continue;}
    let exponent=i32(x>>23u)+i32(y>>23u)-300;
    minimum=min(minimum,exponent); maximum=max(maximum,exponent);
  }
  if(minimum>maximum){return true;} if(maximum-minimum>200){return false;}
  var positive:array<u32,8>; var negative:array<u32,8>;
  for(var term=0u;term<8u;term++) {
    let x=bitcast<u32>(left[term]); let y=bitcast<u32>(right[term]);
    if((x & 0x7fffffffu)==0u || (y & 0x7fffffffu)==0u){continue;}
    let product=multiply24((x & 0x7fffffu)|0x800000u,(y & 0x7fffffu)|0x800000u);
    let shift=u32(i32((x & 0x7fffffffu)>>23u)+i32((y & 0x7fffffffu)>>23u)-300-minimum);
    if(((x^y)>>31u)!=0u){accumulateShifted(&negative,product.x,product.y,shift);}
    else {accumulateShifted(&positive,product.x,product.y,shift);}
  }
  for(var word=0u;word<8u;word++){if(positive[word]!=negative[word]){return false;}}
  return true;
}`;

/** Raw and effect-space certificates, ML rejection, optional diagnostics/residuals. @internal */
export const PANEL_AFFINE_FIT_MAXIMUM_NODE_COUNT = 15;

/** Bounded integer scratch; the caller preflights this before declaring any graph resources. @internal */
export function getPanelAffineFitAllocationLengths(
  rowCount: number,
  predictorCount: number
): Record<string, number> {
  const {tileCount} = getRegressionTileLayout(rowCount);
  return {
    'affine certificate partials': tileCount * predictorCount,
    'affine certificate anchors': predictorCount,
    'affine certificate columns': predictorCount,
    'affine certificate flags': 2
  };
}

/**
 * A sound original-input witness for y=a+b*x_j, including b=0. Such data have zero Gaussian
 * variance in every fixed-effects mode. Floating within/OLS fits can instead manufacture a
 * tiny positive variance. Never compare their variance with a threshold: prove raw collinearity.
 *
 * Inputs are read as u32 bits so a subnormal cannot silently become a proof of zero. The shared
 * integer orientation predicate certifies normal/zero coordinates whose product-exponent span
 * fits its accumulator; unsupported input or an uncertified orientation means NO witness.
 * A vertical line is not a witness: the selected anchor must differ in x, or y must be constant.
 * Other exact linear combinations retain the nested fit's handling; this witness is sufficient,
 * not necessary. CPU's original-data Gram certificate also covers multivariate combinations.
 * All row walks are parallel tiles (at most 4096 rows/tiles under the parent's 2^24-row cap).
 * @internal
 */
export function getPanelAffineFitNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUSpatialPanelErrorMaximumLikelihoodProps,
  effectCertificate = false
): GPUCommandNode<Parameters>[] {
  const rows = props.response.length;
  const {predictorCount, output, diagnostics} = props;
  const {tileCount, tileRowCount} = getRegressionTileLayout(rows);
  const partial = createTransientView(
    graph,
    `${id}-affine-partials`,
    'uint32',
    tileCount * predictorCount
  );
  const anchors = createTransientView(graph, `${id}-affine-anchors`, 'uint32', predictorCount);
  const columns = createTransientView(graph, `${id}-affine-columns`, 'uint32', predictorCount);
  // [raw affine witness, ML result rejected by this submission]. Reset both on every run.
  const flags = createTransientView(graph, `${id}-affine-flags`, 'uint32', 2);
  const entities = rows / props.periodCount;
  const baseRow = (row: string) =>
    props.effects === 'time' ? `(${row} / ${entities}u) * ${entities}u` : `${row} % ${entities}u`;
  const point = (row: string) =>
    `vec2f(bitcast<f32>(predictorBits[predictorBitsOffset + (${row}) * ${predictorCount}u + column]),bitcast<f32>(responseBits[responseBitsOffset + (${row})]))`;
  const inputBindings = [
    {name: 'predictorBits', view: props.predictors, type: 'u32', access: 'read'},
    {name: 'responseBits', view: props.response, type: 'u32', access: 'read'}
  ] as const;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-affine-anchor-tiles`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'affine-anchor-tiles',
      invocationCount: tileCount * predictorCount,
      workgroupSize: 64,
      bindings: [
        inputBindings[0],
        {name: 'partial', view: partial, type: 'u32', access: 'read_write'}
      ],
      declarations: AFFINE_INPUT_BITS_WGSL,
      body: effectCertificate
        ? `let column=index/${tileCount}u; let tile=index%${tileCount}u; var candidate=0xffffffffu;
for(var row=tile*${tileRowCount}u;row<min((tile+1u)*${tileRowCount}u,${rows}u);row++) {
if(normalizePanelAffineBits(predictorBits[predictorBitsOffset+row*${predictorCount}u+column])!=normalizePanelAffineBits(predictorBits[predictorBitsOffset+(${baseRow('row')})*${predictorCount}u+column])){candidate=min(candidate,row);}}
partial[partialOffset+index]=candidate;`
        : `let column = index / ${tileCount}u;
let tile = index % ${tileCount}u;
let first = normalizePanelAffineBits(predictorBits[predictorBitsOffset + column]);
var candidate = 0xffffffffu;
for (var row = tile * ${tileRowCount}u; row < min((tile + 1u) * ${tileRowCount}u, ${rows}u); row++) {
  let value = normalizePanelAffineBits(predictorBits[predictorBitsOffset + row * ${predictorCount}u + column]);
  if (isPanelAffineNormal(value) && value != first) { candidate = min(candidate, row); }
}
partial[partialOffset + index] = candidate;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-affine-anchor-merge`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'affine-anchor-merge',
      invocationCount: predictorCount,
      workgroupSize: 64,
      bindings: [
        {name: 'partial', view: partial, type: 'u32', access: 'read'},
        {name: 'anchors', view: anchors, type: 'u32', access: 'read_write'}
      ],
      body: `var candidate = 0xffffffffu;
for (var tile = 0u; tile < ${tileCount}u; tile++) { candidate = min(candidate, partial[partialOffset + index * ${tileCount}u + tile]); }
anchors[anchorsOffset + index] = candidate;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-affine-proof-tiles`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'affine-proof-tiles',
      invocationCount: tileCount * predictorCount,
      workgroupSize: 64,
      bindings: [
        ...inputBindings,
        {name: 'anchors', view: anchors, type: 'u32', access: 'read'},
        {name: 'partial', view: partial, type: 'u32', access: 'read_write'}
      ],
      declarations: `${AFFINE_INPUT_BITS_WGSL}\n${EXACT_ORIENTATION_WGSL}\n${PANEL_PARALLEL_WGSL}
fn supportedPanelRow(row:u32,column:u32)->bool {
return isPanelAffineNormal(predictorBits[predictorBitsOffset+row*${predictorCount}u+column]) && isPanelAffineNormal(responseBits[responseBitsOffset+row]);
}`,
      body: effectCertificate
        ? `let column=index/${tileCount}u; let tile=index%${tileCount}u; partial[partialOffset+index]=0u;
let anchor=anchors[anchorsOffset+column]; if(anchor>=${rows}u){return;}
if(!supportedPanelRow(anchor,column) || !supportedPanelRow(${baseRow('anchor')},column)){return;}
let first=${point(baseRow('anchor'))}; let second=${point('anchor')};
if(!isPanelAffineNormal(bitcast<u32>(first.x)) || !isPanelAffineNormal(bitcast<u32>(first.y)) || !isPanelAffineNormal(bitcast<u32>(second.x)) || !isPanelAffineNormal(bitcast<u32>(second.y))){return;}
for(var row=tile*${tileRowCount}u;row<min((tile+1u)*${tileRowCount}u,${rows}u);row++) {
if(!supportedPanelRow(row,column) || !supportedPanelRow(${baseRow('row')},column)){return;}
let base=${point(baseRow('row'))}; let current=${point('row')};
if(!isPanelAffineNormal(bitcast<u32>(base.x)) || !isPanelAffineNormal(bitcast<u32>(base.y)) || !isPanelAffineNormal(bitcast<u32>(current.x)) || !isPanelAffineNormal(bitcast<u32>(current.y)) || !panelParallel(first,second,base,current)){return;}}
partial[partialOffset+index]=1u;`
        : `let column = index / ${tileCount}u;
let tile = index % ${tileCount}u;
partial[partialOffset + index] = 0u;
let firstHorizontal = normalizePanelAffineBits(predictorBits[predictorBitsOffset + column]);
let firstVertical = normalizePanelAffineBits(responseBits[responseBitsOffset]);
if (!isPanelAffineNormal(firstHorizontal) || !isPanelAffineNormal(firstVertical)) { return; }
let first = vec2f(bitcast<f32>(firstHorizontal), bitcast<f32>(firstVertical));
let anchor = anchors[anchorsOffset + column];
var second = first;
if (anchor < ${rows}u) {
  let horizontal = normalizePanelAffineBits(predictorBits[predictorBitsOffset + anchor * ${predictorCount}u + column]);
  let vertical = normalizePanelAffineBits(responseBits[responseBitsOffset + anchor]);
  if (!isPanelAffineNormal(horizontal) || !isPanelAffineNormal(vertical)) { return; }
  second = vec2f(bitcast<f32>(horizontal), bitcast<f32>(vertical));
}
for (var row = tile * ${tileRowCount}u; row < min((tile + 1u) * ${tileRowCount}u, ${rows}u); row++) {
  let horizontal = normalizePanelAffineBits(predictorBits[predictorBitsOffset + row * ${predictorCount}u + column]);
  let vertical = normalizePanelAffineBits(responseBits[responseBitsOffset + row]);
  if (!isPanelAffineNormal(horizontal) || !isPanelAffineNormal(vertical)) { return; }
  if (anchor == 0xffffffffu) {
    if (horizontal != firstHorizontal || vertical != firstVertical) { return; }
  } else if (orientExact(first, second, vec2f(bitcast<f32>(horizontal), bitcast<f32>(vertical))) != 0) { return; }
}
partial[partialOffset + index] = 1u;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-affine-proof-merge`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'affine-proof-merge',
      invocationCount: predictorCount,
      workgroupSize: 64,
      bindings: [
        {name: 'partial', view: partial, type: 'u32', access: 'read'},
        {name: 'columns', view: columns, type: 'u32', access: 'read_write'}
      ],
      body: `var certified = 1u;
for (var tile = 0u; tile < ${tileCount}u; tile++) { certified &= partial[partialOffset + index * ${tileCount}u + tile]; }
columns[columnsOffset + index] = certified;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-affine-proof`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'affine-proof',
      // contract-allow: single-invocation folds at most 15 independently certified predictor columns.
      invocationCount: 1,
      bindings: [
        {name: 'columns', view: columns, type: 'u32', access: 'read'},
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
      ],
      body: `var certified = 0u;
for (var column = 0u; column < ${predictorCount}u; column++) { certified |= columns[columnsOffset + column]; }
flags[flagsOffset] = certified;
flags[flagsOffset + 1u] = 0u;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-affine-reject-fit`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'affine-reject-fit',
      // contract-allow: single-invocation clears at most 68 table words and the fixed 22-word summary.
      invocationCount: 1,
      bindings: [
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
        {name: 'code', view: output.status.code, type: 'u32', access: 'read_write'},
        {name: 'table', view: output.table, type: 'f32', access: 'read_write'},
        {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'},
        ...(output.status.converged
          ? [
              {
                name: 'converged',
                view: output.status.converged,
                type: 'u32',
                access: 'read_write'
              } as const
            ]
          : [])
      ],
      declarations: GET_NAN_WGSL,
      body: `if (flags[flagsOffset] == 0u || code[codeOffset] != 0u) { return; }
flags[flagsOffset + 1u] = 1u;
code[codeOffset] = 3u;
${output.status.converged ? 'converged[convergedOffset] = 0u;' : ''}
for (var slot = 0u; slot < ${(predictorCount + 2) * 4}u; slot++) { table[tableOffset + slot] = getNaN(); }
for (var slot = 0u; slot < 22u; slot++) { summary[summaryOffset + slot] = getNaN(); }
summary[summaryOffset] = ${rows}.0;`
    })
  ];
  if (diagnostics && !effectCertificate)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-affine-reject-diagnostics`,
        operation: 'GPUSpatialPanelErrorMaximumLikelihood',
        variant: 'affine-reject-diagnostics',
        invocationCount: 1,
        bindings: [
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'status', view: diagnostics.status, type: 'u32', access: 'read_write'},
          {name: 'tests', view: diagnostics.tests, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: `if (flags[flagsOffset] != 0u && status[statusOffset] == 0u) {
status[statusOffset] = 2u;
${Array.from({length: 6}, (_, slot) => `tests[testsOffset + ${slot}u] = getNaN();`).join('\n')}
}`
      })
    );
  if (output.residuals)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-affine-reject-residuals`,
        operation: 'GPUSpatialPanelErrorMaximumLikelihood',
        variant: 'affine-reject-residuals',
        invocationCount: rows,
        bindings: [
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'residuals', view: output.residuals, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: 'if (flags[flagsOffset + 1u] != 0u) { residuals[residualsOffset + index] = getNaN(); }'
      })
    );
  if (!effectCertificate)
    nodes.push(...getPanelAffineFitNodes(graph, `${id}-effects`, props, true));
  return nodes;
}

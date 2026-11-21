// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getRegressionTileLayout} from './regression-row-tiles';
import type {GPUSpatialPanelLagMaximumLikelihoodProps} from './gpu-spatial-panel-lag-maximum-likelihood';

/** Mantissa widths per independently generated proposal family. */
const CANDIDATE_COUNT = 24;
/** At most two direct and two proportional proposal families, plus the zero-rho witness. */
export const MAXIMUM_PANEL_LAG_POLE_CANDIDATE_COUNT = 97;

/** Exact original-input witness, including single-column rational proportionality. @internal */
export function hasPanelLagPoleOnCPU(
  input: {
    predictors: Float32Array;
    response: Float32Array;
    weights: {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array};
  },
  predictorCount: number,
  coefficients: ArrayLike<number>
): boolean {
  if (!input.predictors.every(Number.isFinite) || !input.response.every(Number.isFinite))
    return false;
  const storage = new DataView(new ArrayBuffer(4));
  const getBits = (value: number) => {
    storage.setFloat32(0, value);
    return storage.getUint32(0);
  };
  const getValue = (bits: number) => {
    storage.setUint32(0, bits);
    return storage.getFloat32(0);
  };
  const getDyadic = (value: number) => {
    const bits = getBits(value),
      magnitude = bits & 0x7fffffff,
      exponent = magnitude >>> 23;
    return {
      units: BigInt((magnitude & 0x7fffff) | (exponent ? 0x800000 : 0)) * (bits >>> 31 ? -1n : 1n),
      exponent: exponent ? exponent - 150 : -149
    };
  };
  let hasProportionalAt: ((rho: number) => boolean) | undefined;
  if (predictorCount === 1) {
    const entities = input.weights.offsets.length - 1;
    const difference = (values: Float32Array, row: number) => {
      const first = getDyadic(values[row]),
        second = getDyadic(values[row % entities]);
      const exponent = Math.min(first.exponent, second.exponent);
      return {
        units:
          (first.units << BigInt(first.exponent - exponent)) -
          (second.units << BigInt(second.exponent - exponent)),
        exponent
      };
    };
    let pivot: {units: bigint; exponent: number} | undefined;
    let pivotRow = input.response.length;
    for (let row = entities; row < input.response.length; row++) {
      const value = difference(input.predictors, row);
      if (value.units !== 0n) {
        pivot = value;
        pivotRow = row;
        break;
      }
    }
    const getSum = (terms: {units: bigint; exponent: number}[]) => {
      const present = terms.filter(value => value.units !== 0n);
      const exponent = present.length ? Math.min(...present.map(value => value.exponent)) : 0;
      return {
        units: present.reduce(
          (sum, value) => sum + (value.units << BigInt(value.exponent - exponent)),
          0n
        ),
        exponent
      };
    };
    const responseDifference = (row: number, rho: number) => {
      const terms = [difference(input.response, row)];
      const entity = row % entities,
        periodStart = row - entity;
      const parameter = getDyadic(rho);
      for (
        let slot = input.weights.offsets[entity];
        slot < input.weights.offsets[entity + 1];
        slot++
      ) {
        const neighbor = input.weights.neighbors[slot],
          weight = getDyadic(input.weights.weights[slot]);
        const temporal = difference(input.response, periodStart + neighbor);
        terms.push({
          units: -parameter.units * weight.units * temporal.units,
          exponent: parameter.exponent + weight.exponent + temporal.exponent
        });
      }
      return getSum(terms);
    };
    hasProportionalAt = rho => {
      if (!pivot || pivotRow === input.response.length) return false;
      const responsePivot = responseDifference(pivotRow, rho);
      let proportional = true;
      for (let row = entities; row < input.response.length && proportional; row++) {
        const response = responseDifference(row, rho),
          predictor = difference(input.predictors, row);
        const firstExponent = response.exponent + pivot.exponent;
        const secondExponent = predictor.exponent + responsePivot.exponent;
        const exponent = Math.min(firstExponent, secondExponent);
        proportional =
          (response.units * pivot.units) << BigInt(firstExponent - exponent) ===
          (predictor.units * responsePivot.units) << BigInt(secondExponent - exponent);
      }
      return proportional;
    };
    if (hasProportionalAt(0)) return true;
  }
  const raw = Array.from({length: predictorCount + 1}, (_, column) =>
    Math.fround(coefficients[column + 1])
  );
  if (!raw.every(Number.isFinite)) return false;
  const entityCount = input.weights.offsets.length - 1;
  for (let proposal = 0; proposal < CANDIDATE_COUNT * 2; proposal++) {
    const keep = (proposal % CANDIDATE_COUNT) + 1;
    const drop = 24 - keep;
    const candidate = raw.map(value => {
      const bits = getBits(value);
      let magnitude = bits & 0x7fffffff;
      if (proposal >= CANDIDATE_COUNT && Math.abs(value) < 2 ** -keep) magnitude = 0;
      else if (drop > 0) magnitude = (magnitude + 2 ** (drop - 1)) & ~(2 ** drop - 1);
      return getValue((magnitude | (bits & 0x80000000)) >>> 0);
    });
    if (!candidate.every(Number.isFinite) || Math.abs(candidate[predictorCount]) > 0.999) continue;
    if (hasProportionalAt?.(candidate[predictorCount])) return true;
    let certified = true;
    for (let row = entityCount; row < input.response.length && certified; row++) {
      const entity = row % entityCount,
        start = row - entity;
      const terms: {units: bigint; exponent: number}[] = [];
      const addTerm = (values: number[], subtract: boolean) => {
        const parts = values.map(getDyadic);
        terms.push({
          units: parts.reduce((product, part) => product * part.units, subtract ? -1n : 1n),
          exponent: parts.reduce((sum, part) => sum + part.exponent, 0)
        });
      };
      addTerm([input.response[row]], false);
      addTerm([input.response[entity]], true);
      for (let column = 0; column < predictorCount; column++) {
        addTerm([candidate[column], input.predictors[row * predictorCount + column]], true);
        addTerm([candidate[column], input.predictors[entity * predictorCount + column]], false);
      }
      for (
        let slot = input.weights.offsets[entity];
        slot < input.weights.offsets[entity + 1];
        slot++
      ) {
        const neighbor = input.weights.neighbors[slot],
          weight = input.weights.weights[slot];
        addTerm([candidate[predictorCount], weight, input.response[start + neighbor]], true);
        addTerm([candidate[predictorCount], weight, input.response[neighbor]], false);
      }
      const present = terms.filter(term => term.units !== 0n);
      const minimum = Math.min(...present.map(term => term.exponent));
      certified =
        present.reduce((sum, term) => sum + (term.units << BigInt(term.exponent - minimum)), 0n) ===
        0n;
    }
    if (certified) return true;
  }
  return false;
}

/** Scratch for sound, sufficient original-input zero-within-variance certificates. @internal */
export function getPanelLagPoleAllocationLengths(
  rows: number,
  predictorCount: number
): Record<string, number> {
  const {tileCount} = getRegressionTileLayout(rows);
  const count = predictorCount === 1 ? MAXIMUM_PANEL_LAG_POLE_CANDIDATE_COUNT : CANDIDATE_COUNT * 2;
  return {
    'pole candidates': count * (predictorCount + 2),
    'pole partials': count * tileCount,
    'pole witnesses': count + 1,
    ...(predictorCount === 1
      ? {
          'pole proportional pivots': tileCount + 1,
          'pole pivot reduction': tileCount > 1 ? Math.ceil(tileCount / 2) : 0
        }
      : {})
  };
}

// Every term has three normal/zero f32 factors. The exact product has at most 72 bits.
// A 168-bit exponent span plus 12 carry bits fits in the existing 256-bit accumulator:
// each row has at most 2*(1024 neighbors + 7 slopes + response) < 4096 terms.
const PANEL_LAG_POLE_WGSL = `${EXACT_ORIENTATION_WGSL}
fn includePoleTerm(first: u32, second: u32, third: u32, bounds: ptr<function, vec2i>) -> bool {
  let values = array<u32, 3>(first & 0x7fffffffu, second & 0x7fffffffu, third & 0x7fffffffu);
  var exponent = 0i;
  var zero = false;
  for (var item = 0u; item < 3u; item++) {
    let value = values[item]; let encoded = value >> 23u;
    if (encoded == 255u || (encoded == 0u && value != 0u)) { return false; }
    zero = zero || value == 0u; exponent += i32(encoded) - 150i;
  }
  if (!zero) { (*bounds).x = min((*bounds).x, exponent); (*bounds).y = max((*bounds).y, exponent); }
  return true;
}
fn accumulatePoleTerm(first: u32, second: u32, third: u32, subtract: bool, minimumExponent: i32,
  positive: ptr<function, array<u32, 8>>, negative: ptr<function, array<u32, 8>>) {
  let firstMagnitude = first & 0x7fffffffu; let secondMagnitude = second & 0x7fffffffu; let thirdMagnitude = third & 0x7fffffffu;
  if (firstMagnitude == 0u || secondMagnitude == 0u || thirdMagnitude == 0u) { return; }
  let pair = multiply24((firstMagnitude & 0x7fffffu) | 0x800000u, (secondMagnitude & 0x7fffffu) | 0x800000u);
  let significand = (thirdMagnitude & 0x7fffffu) | 0x800000u;
  let lower = multiply24(pair.x & 0xffffffu, significand);
  let upper = multiply24((pair.x >> 24u) | (pair.y << 8u), significand);
  let low = lower.x + (upper.x << 24u); let carry = select(0u, 1u, low < lower.x);
  let middlePartial = lower.y + (upper.x >> 8u);
  let middleNext = middlePartial + (upper.y << 24u);
  let middle = middleNext + carry;
  let high = (upper.y >> 8u) + select(0u, 1u, middleNext < middlePartial) + select(0u, 1u, middle < middleNext);
  let exponent = i32(firstMagnitude >> 23u) + i32(secondMagnitude >> 23u) + i32(thirdMagnitude >> 23u) - 450i;
  let shift = u32(exponent - minimumExponent);
  let isNegative = (((first ^ second ^ third) >> 31u) != 0u) != subtract;
  if (isNegative) { accumulateShifted(negative, low, middle, shift); accumulateShifted(negative, high, 0u, shift + 64u); }
  else { accumulateShifted(positive, low, middle, shift); accumulateShifted(positive, high, 0u, shift + 64u); }
}
fn includePoleQuadTerm(first:u32,second:u32,third:u32,fourth:u32,bounds:ptr<function,vec2i>)->bool {
 let values=array<u32,4>(first&0x7fffffffu,second&0x7fffffffu,third&0x7fffffffu,fourth&0x7fffffffu);
 var exponent=0i;var zero=false;
 for(var item=0u;item<4u;item++){let value=values[item];let encoded=value>>23u;
  if(encoded==255u || (encoded==0u && value!=0u)){return false;}
  zero=zero || value==0u;exponent+=i32(encoded)-150i;
 }
 if(!zero){(*bounds).x=min((*bounds).x,exponent);(*bounds).y=max((*bounds).y,exponent);}
 return true;
}
fn accumulatePoleQuadTerm(first:u32,second:u32,third:u32,fourth:u32,subtract:bool,minimumExponent:i32,
 positive:ptr<function,array<u32,8>>,negative:ptr<function,array<u32,8>>) {
 let values=array<u32,4>(first&0x7fffffffu,second&0x7fffffffu,third&0x7fffffffu,fourth&0x7fffffffu);
 var product:array<u32,8>;product[0]=1u;var exponent=0i;
 // Eight base-2^12 limbs hold the exact 96-bit four-significand product. Each carry
 // step sums at most two 12x12-bit products plus a carry, well below u32 overflow.
 for(var item=0u;item<4u;item++){
  let magnitude=values[item];if(magnitude==0u){return;}
  let significand=(magnitude&0x7fffffu)|0x800000u;
  let lower=significand&4095u;let upper=significand>>12u;
  var next:array<u32,8>;var carry=0u;
  for(var limb=0u;limb<8u;limb++){
   var total=product[limb]*lower+carry;
   if(limb>0u){total+=product[limb-1u]*upper;}
   next[limb]=total&4095u;carry=total>>12u;
  }
  product=next;exponent+=i32(magnitude>>23u)-150i;
 }
 let shift=u32(exponent-minimumExponent);
 let isNegative=(((first^second^third^fourth)>>31u)!=0u)!=subtract;
 for(var limb=0u;limb<8u;limb++){
  if(isNegative){accumulateShifted(negative,product[limb],0u,shift+12u*limb);}
  else{accumulateShifted(positive,product[limb],0u,shift+12u*limb);}
 }
}
`;

function getPanelLagProportionalProofWGSL(entities: number, capacity: number): string {
  const direct: [string, string, string, string, boolean][] = [
    ['rowResponse', 'pivotPredictor', 'one', 'one', false],
    ['rowResponse', 'pivotReferencePredictor', 'one', 'one', true],
    ['rowReferenceResponse', 'pivotPredictor', 'one', 'one', true],
    ['rowReferenceResponse', 'pivotReferencePredictor', 'one', 'one', false],
    ['rowPredictor', 'pivotResponse', 'one', 'one', true],
    ['rowPredictor', 'pivotReferenceResponse', 'one', 'one', false],
    ['rowReferencePredictor', 'pivotResponse', 'one', 'one', false],
    ['rowReferencePredictor', 'pivotReferenceResponse', 'one', 'one', true]
  ];
  const rowLag: typeof direct = [
    ['rho', 'weightBits', 'lagCurrent', 'pivotPredictor', true],
    ['rho', 'weightBits', 'lagCurrent', 'pivotReferencePredictor', false],
    ['rho', 'weightBits', 'lagReference', 'pivotPredictor', false],
    ['rho', 'weightBits', 'lagReference', 'pivotReferencePredictor', true]
  ];
  const pivotLag: typeof direct = [
    ['rho', 'weightBits', 'lagCurrent', 'rowPredictor', false],
    ['rho', 'weightBits', 'lagReference', 'rowPredictor', true],
    ['rho', 'weightBits', 'lagCurrent', 'rowReferencePredictor', true],
    ['rho', 'weightBits', 'lagReference', 'rowReferencePredictor', false]
  ];
  const terms = (values: typeof direct, proof: boolean) =>
    values
      .map(([first, second, third, fourth, subtract]) =>
        proof
          ? `accumulatePoleQuadTerm(${first},${second},${third},${fourth},${subtract},bounds.x,&positive,&negative);`
          : `if(!includePoleQuadTerm(${first},${second},${third},${fourth},&bounds)){return;}`
      )
      .join('\n');
  const lagTerms = (axis: 'row' | 'pivot', proof: boolean) => `{
let begin=offsets[offsetsOffset+${axis === 'row' ? 'entity' : 'pivotEntity'}];let end=offsets[offsetsOffset+${axis === 'row' ? 'entity' : 'pivotEntity'}+1u];
if(end<begin || end>${capacity}u){return;}
for(var slot=begin;slot<end;slot++){
 let neighbor=neighbors[neighborsOffset+slot];if(neighbor>=${entities}u){return;}
 let weightBits=weights[weightsOffset+slot];
 let lagCurrent=response[responseOffset+${axis === 'row' ? 'row-entity' : 'pivot-pivotEntity'}+neighbor];let lagReference=response[responseOffset+neighbor];
 ${terms(axis === 'row' ? rowLag : pivotLag, proof)}
}}`;
  // 96 product bits + 140 exponent bits + 14 carry bits (<=8200 terms) fit in 256 bits.
  return `
  var bounds=vec2i(10000i,-10000i);let one=0x3f800000u;
  ${terms(direct, false)}${lagTerms('row', false)}${lagTerms('pivot', false)}
  if(bounds.y-bounds.x>140i){return;}
  var positive:array<u32,8>;var negative:array<u32,8>;
  ${terms(direct, true)}${lagTerms('row', true)}${lagTerms('pivot', true)}
  for(var word=0u;word<8u;word++){if(positive[word]!=negative[word]){return;}}
`;
}

/**
 * Certify y_t-y_0 = sum beta_j*(x_tj-x_0j) + rho*sum w_is*(y_ts-y_0s)
 * directly on input bits. No floating residual, mean, lag or variance is used as proof.
 * One family rounds every fitted coordinate to 1..24 significant bits without zeroing it,
 * retaining the original float32 tuple. A second family additionally proposes small zeros.
 * A candidate is accepted
 * only by exact integer equality in every row. This is sufficient, not a symbolic rank test.
 * A single predictor also has exact cross-product certificates at each candidate rho,
 * checking y-rho*Wy against x without requiring a representable slope (including 1/3).
 * Four-factor products use a 140-bit exponent envelope; the original dyadic witnesses
 * retain their 168-bit envelope and are evaluated separately.
 * Unsupported subnormals/exponent spans produce no certificate, never a zero certificate.
 * Both random and unit Gaussian models are degenerate if within variance can vanish.
 * @internal
 */
export function getPanelLagPoleNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUSpatialPanelLagMaximumLikelihoodProps
): GPUCommandNode<Parameters>[] {
  const rows = props.response.length,
    entities = props.weights.offsets.length - 1,
    columns = props.predictorCount;
  const {tileCount, tileRowCount} = getRegressionTileLayout(rows);
  const candidateCount =
    columns === 1 ? MAXIMUM_PANEL_LAG_POLE_CANDIDATE_COUNT : CANDIDATE_COUNT * 2;
  const stride = columns + 2;
  const candidates = createTransientView(
    graph,
    `${id}-pole-candidates`,
    'uint32',
    candidateCount * stride
  );
  const partials = createTransientView(
    graph,
    `${id}-pole-partials`,
    'uint32',
    candidateCount * tileCount
  );
  const witnesses = createTransientView(
    graph,
    `${id}-pole-witnesses`,
    'uint32',
    candidateCount + 1
  );
  const proportionalPivots =
    columns === 1
      ? createTransientView(graph, `${id}-pole-proportional-pivots`, 'uint32', tileCount + 1)
      : undefined;
  const pivotScratch =
    columns === 1 && tileCount > 1
      ? createTransientView(graph, `${id}-pole-pivot-reduction`, 'uint32', Math.ceil(tileCount / 2))
      : undefined;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const kernel = (
    name: string,
    invocationCount: number,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = ''
  ) =>
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pole-${name}`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: `pole-${name}`,
        invocationCount,
        workgroupSize: 64,
        bindings,
        body,
        declarations
      })
    );
  if (proportionalPivots) {
    kernel(
      'proportional-pivots',
      tileCount,
      [
        {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
        {name: 'pivots', view: proportionalPivots, type: 'u32', access: 'read_write'}
      ],
      `var pivot=${rows}u;
for(var row=max(${entities}u,index*${tileRowCount}u);row<min((index+1u)*${tileRowCount}u,${rows}u);row++){
 if(predictors[predictorsOffset+row]!=predictors[predictorsOffset+row%${entities}u]){pivot=row;break;}
}pivots[pivotsOffset+index]=pivot;`
    );
    let activeCount = tileCount;
    let source = proportionalPivots;
    let target = pivotScratch ?? proportionalPivots;
    let pass = 0;
    do {
      const outputCount = Math.ceil(activeCount / 2);
      const final = outputCount === 1;
      const output = final ? proportionalPivots : target;
      const shared = source === output;
      kernel(
        `proportional-pivot-${pass}`,
        outputCount,
        shared
          ? [{name: 'pivots', view: source, type: 'u32', access: 'read_write'}]
          : [
              {name: 'pivots', view: source, type: 'u32', access: 'read'},
              {name: 'output', view: output, type: 'u32', access: 'read_write'}
            ],
        `let first=index*2u;
var pivot=pivots[pivotsOffset+first];
if(first+1u<${activeCount}u){pivot=min(pivot,pivots[pivotsOffset+first+1u]);}
${shared ? 'pivots[pivotsOffset' : 'output[outputOffset'}+${final ? `${tileCount}u` : 'index'}]=pivot;`
      );
      activeCount = outputCount;
      source = output;
      target = output === proportionalPivots ? pivotScratch! : proportionalPivots;
      pass++;
    } while (activeCount > 1);
  }
  kernel(
    'candidates',
    candidateCount,
    [
      {name: 'coefficients', view: props.output.coefficients, type: 'u32', access: 'read'},
      {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
      {name: 'candidates', view: candidates, type: 'u32', access: 'read_write'}
    ],
    `let base = candidatesOffset + index * ${stride}u;
candidates[base] = 0u;
if (status[statusOffset] == 1u || status[statusOffset] > 3u) { return; }
${columns === 1 ? `if(index==96u){candidates[base]=1u;return;}` : ''}
for (var column = 0u; column <= ${columns}u; column++) { let magnitude = coefficients[coefficientsOffset + column + 1u] & 0x7fffffffu; if (magnitude >= 0x7f800000u) { return; } }
let keep = index % 24u + 1u; let drop = 24u - keep;
for (var column = 0u; column <= ${columns}u; column++) {
  let original = coefficients[coefficientsOffset + column + 1u]; var magnitude = original & 0x7fffffffu;
  if (index % 48u >= 24u && bitcast<f32>(magnitude) < exp2(-f32(keep))) { magnitude = 0u; }
  else if (drop > 0u) { magnitude = (magnitude + (1u << (drop - 1u))) & ~((1u << drop) - 1u); }
  candidates[base + column + 1u] = magnitude | (original & 0x80000000u);
}
let rho = bitcast<f32>(candidates[base + ${columns + 1}u]);
candidates[base] = select(0u, 1u, abs(rho) < 0.999);`
  );
  const bindings: WGSLKernelBinding[] = [
    {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
    {name: 'weights', view: props.weights.weights, type: 'u32', access: 'read'},
    {name: 'predictors', view: props.predictors, type: 'u32', access: 'read'},
    {name: 'response', view: props.response, type: 'u32', access: 'read'},
    {name: 'candidates', view: candidates, type: 'u32', access: 'read'},
    {name: 'partials', view: partials, type: 'u32', access: 'read_write'}
  ];
  if (proportionalPivots)
    bindings.push({name: 'pivots', view: proportionalPivots, type: 'u32', access: 'read'});
  kernel(
    'proof-tiles',
    candidateCount * tileCount,
    bindings,
    `partials[partialsOffset + index] = 0u;
let candidate = index / ${tileCount}u; let tile = index % ${tileCount}u; let base = candidatesOffset + candidate * ${stride}u;
if (candidates[base] == 0u) { return; }
${
  proportionalPivots
    ? `if(candidate>=48u){
let rho=select(candidates[base+2u],0u,candidate==96u);
let pivot=pivots[pivotsOffset+${tileCount}u];if(pivot>=${rows}u){return;}let pivotEntity=pivot%${entities}u;
let pivotResponse=response[responseOffset+pivot];let pivotReferenceResponse=response[responseOffset+pivotEntity];
let pivotPredictor=predictors[predictorsOffset+pivot];let pivotReferencePredictor=predictors[predictorsOffset+pivotEntity];
for(var row=max(${entities}u,tile*${tileRowCount}u);row<min((tile+1u)*${tileRowCount}u,${rows}u);row++){
let entity=row%${entities}u;
let rowResponse=response[responseOffset+row];let rowReferenceResponse=response[responseOffset+entity];
let rowPredictor=predictors[predictorsOffset+row];let rowReferencePredictor=predictors[predictorsOffset+entity];
${getPanelLagProportionalProofWGSL(entities, props.weights.neighbors.length)}
}partials[partialsOffset+index]=1u;return;}`
    : ''
}
let rho = candidates[base + ${columns + 1}u]; let one = 0x3f800000u;
for (var row = tile * ${tileRowCount}u; row < min((tile + 1u) * ${tileRowCount}u, ${rows}u); row++) {
  let entity = row % ${entities}u; let periodStart = row - entity;
  let begin = offsets[offsetsOffset + entity]; let end = offsets[offsetsOffset + entity + 1u];
  if (end < begin || end > ${props.weights.neighbors.length}u) { return; }
  var bounds = vec2i(10000i, -10000i);
  if (!includePoleTerm(response[responseOffset + row], one, one, &bounds) || !includePoleTerm(response[responseOffset + entity], one, one, &bounds)) { return; }
  for (var column = 0u; column < ${columns}u; column++) {
    let coefficient = candidates[base + column + 1u];
    if (!includePoleTerm(coefficient, predictors[predictorsOffset + row * ${columns}u + column], one, &bounds) || !includePoleTerm(coefficient, predictors[predictorsOffset + entity * ${columns}u + column], one, &bounds)) { return; }
  }
  for (var slot = begin; slot < end; slot++) {
    let neighbor = neighbors[neighborsOffset + slot]; if (neighbor >= ${entities}u) { return; }
    if (!includePoleTerm(rho, weights[weightsOffset + slot], response[responseOffset + periodStart + neighbor], &bounds) || !includePoleTerm(rho, weights[weightsOffset + slot], response[responseOffset + neighbor], &bounds)) { return; }
  }
  if (bounds.y - bounds.x > 168i) { return; }
  var positive: array<u32, 8>; var negative: array<u32, 8>;
  accumulatePoleTerm(response[responseOffset + row], one, one, false, bounds.x, &positive, &negative);
  accumulatePoleTerm(response[responseOffset + entity], one, one, true, bounds.x, &positive, &negative);
  for (var column = 0u; column < ${columns}u; column++) {
    let coefficient = candidates[base + column + 1u];
    accumulatePoleTerm(coefficient, predictors[predictorsOffset + row * ${columns}u + column], one, true, bounds.x, &positive, &negative);
    accumulatePoleTerm(coefficient, predictors[predictorsOffset + entity * ${columns}u + column], one, false, bounds.x, &positive, &negative);
  }
  for (var slot = begin; slot < end; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    accumulatePoleTerm(rho, weights[weightsOffset + slot], response[responseOffset + periodStart + neighbor], true, bounds.x, &positive, &negative);
    accumulatePoleTerm(rho, weights[weightsOffset + slot], response[responseOffset + neighbor], false, bounds.x, &positive, &negative);
  }
  for (var word = 0u; word < 8u; word++) { if (positive[word] != negative[word]) { return; } }
}
partials[partialsOffset + index] = 1u;`,
    PANEL_LAG_POLE_WGSL
  );
  kernel(
    'merge',
    candidateCount,
    [
      {name: 'partials', view: partials, type: 'u32', access: 'read'},
      {name: 'witnesses', view: witnesses, type: 'u32', access: 'read_write'}
    ],
    `var certified = 1u;
for (var tile = 0u; tile < ${tileCount}u; tile++) { certified &= partials[partialsOffset + index * ${tileCount}u + tile]; }
witnesses[witnessesOffset + index] = certified;`
  );
  // contract-allow: single-invocation MAXIMUM_PANEL_LAG_POLE_CANDIDATE_COUNT = 97 bounds the independent candidate certificates.
  kernel(
    'fold',
    1,
    [{name: 'witnesses', view: witnesses, type: 'u32', access: 'read_write'}],
    `var certified = 0u;
for (var candidate = 0u; candidate < ${candidateCount}u; candidate++) { certified |= witnesses[witnessesOffset + candidate]; }
witnesses[witnessesOffset + ${candidateCount}u] = certified;`
  );
  for (const [name, view] of Object.entries(props.output)) {
    if (name === 'status') continue;
    kernel(
      `clear-${name}`,
      view.length,
      [
        {name: 'witnesses', view: witnesses, type: 'u32', access: 'read'},
        {name: 'output', view, type: 'f32', access: 'read_write'}
      ],
      `if (witnesses[witnessesOffset + ${candidateCount}u] != 0u) { output[outputOffset + index] = getNaN(); }`,
      GET_NAN_WGSL
    );
  }
  kernel(
    'status',
    1,
    [
      {name: 'witnesses', view: witnesses, type: 'u32', access: 'read'},
      {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
    ],
    `if (witnesses[witnessesOffset + ${candidateCount}u] != 0u) { status[statusOffset] = 2u; }`
  );
  if (props.hausman) {
    kernel(
      'hausman',
      3,
      [
        {name: 'witnesses', view: witnesses, type: 'u32', access: 'read'},
        {name: 'tests', view: props.hausman.tests, type: 'f32', access: 'read_write'}
      ],
      `if (witnesses[witnessesOffset + ${candidateCount}u] != 0u) { tests[testsOffset + index] = getNaN(); }`,
      GET_NAN_WGSL
    );
    kernel(
      'hausman-status',
      1,
      [
        {name: 'witnesses', view: witnesses, type: 'u32', access: 'read'},
        {name: 'status', view: props.hausman.status, type: 'u32', access: 'read_write'}
      ],
      `if (witnesses[witnessesOffset + ${candidateCount}u] != 0u) { status[statusOffset] = 1u; }`
    );
  }
  return nodes;
}

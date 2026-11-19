import{n as e}from"./gpu-network-cost-weights-D3rw4ZxN.js";import{n as t,t as n}from"./spatial-analysis-packed-kernel-CV3ZCYPQ.js";import{r}from"./kernel-C3tfQOn4.js";import{a as i}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as a,t as o}from"./catchment-accessibility-validation-BjMQQJgk.js";import{r as s,u as c}from"./wgsl-common-DfyXiYB0.js";import{t as l}from"./spatial-analysis-quotient-CbSG4rNM.js";import{E as u,T as d,x as f}from"./gpu-contributor-utils-BIwLQON5.js";import{c as p}from"./spatial-weights-CqWysstN.js";import{l as m}from"./spatial-context-DwZEd4EZ.js";import{t as h}from"./define-run-B-brkCe4.js";import{a as g,c as _,i as v,l as y,n as b,o as x,r as S,s as C}from"./access-Co69Wfld.js";import{p as w,v as T}from"./roads-DzovamHj.js";import{La as E}from"./entry-preview-D70TdSbK.js";var D=`GPUHuffTradeAreas`,O=4294967295;function k(e={}){let t=e.exponent??1;if(!Number.isFinite(Math.fround(t))||!(Math.fround(t)>0))throw Error(`GPUHuffTradeAreas exponent must be positive and finite`);return Float32Array.of(t)}var A=class{id;props;facilityCount;demandCount;constructor(e){let t=e.id??`huff-trade-areas`;this.id=t,this.props=e,this.demandCount=p(t,e.demandWeights,`demandWeights`),this.facilityCount=e.attractiveness.length,e.demandWeights?.spatialContext&&e.facilityWeights?.spatialContext&&m(t,e.demandWeights.spatialContext,e.facilityWeights.spatialContext);let{output:n}=e;u(e.attractiveness,[`float32`],`${t} attractiveness`),u(n.tradeArea,[`uint32`],`${t} output.tradeArea`);for(let[r,i]of[[`demand`,e.demand],[`parameters`,e.parameters],[`output.probabilities`,n.probabilities],[`output.tradeAreaProbability`,n.tradeAreaProbability],[`output.expectedDemand`,n.expectedDemand]])i&&u(i,[`float32`],`${t} ${r}`);if(n.tradeArea.length!==this.demandCount)throw Error(`${t} output.tradeArea length must equal the demand count`);if(n.tradeAreaProbability&&n.tradeAreaProbability.length!==this.demandCount)throw Error(`${t} output.tradeAreaProbability length must equal the demand count`);if(n.probabilities&&n.probabilities.length!==e.demandWeights.neighbors.length)throw Error(`${t} output.probabilities length must equal demandWeights.neighbors length`);if(e.parameters&&e.parameters.length<1)throw Error(`${t} parameters must hold 1 word`);if(n.validity&&(u(n.validity,[`uint32`],`${t} output.validity`),d(t,e.demand!==void 0,`output.validity needs demand`),d(t,e.demand?.length===this.demandCount&&n.validity.length===this.demandCount,`demand and output.validity must have one row per demand location`)),n.expectedDemand){if(!e.facilityWeights||!e.demand)throw Error(`${t} output.expectedDemand needs facilityWeights and demand`);if(n.expectedDemand.length!==this.facilityCount)throw Error(`${t} output.expectedDemand length must equal the facility count`);if(e.demand.length!==this.demandCount)throw Error(`${t} demand length must equal the demandWeights row count`);if(p(t,e.facilityWeights,`facilityWeights`)!==this.facilityCount)throw Error(`${t} facilityWeights row count must equal the facility count`)}let r=Object.fromEntries(M(e).map((e,t)=>[`input.${t}`,e])),i={offsets:e.demandWeights.offsets,neighbors:e.demandWeights.neighbors,weights:e.demandWeights.weights},o=e.output.expectedDemand?{offsets:e.facilityWeights?.offsets,neighbors:e.facilityWeights?.neighbors,weights:e.facilityWeights?.weights,demand:e.demand}:{};a(t,r,n,[{reads:i,writes:{tradeArea:n.tradeArea,tradeAreaProbability:n.tradeAreaProbability}},{reads:i,writes:{probabilities:n.probabilities}},{reads:o,writes:{expectedDemand:n.expectedDemand}},{reads:{demand:e.demand},writes:{validity:n.validity}}])}getCommandNodes(e){let{id:a,props:u,facilityCount:d,demandCount:p}=this,{attractiveness:m,demandWeights:h,parameters:g,output:_}=u;t(e,a,_),f(a,e,[...M(u),...j(_)]),i(e,a,{...Object.fromEntries(M(u).map((e,t)=>[`input.${t}`,e])),..._}),o(e.device,a,`normalization scratch`,p*4*4),o(e.device,a,`attraction scratch`,d*4);let v=r(e,`${a}-normalization`,`float32`,p*4),y=g?[{name:`parameters`,view:g,type:`f32`,access:`read`}]:[],b=r(e,`${a}-attraction`,`float32`,d),x=`const FACILITIES: u32 = ${d}u;
const DEMANDS: u32 = ${p}u;
${c}
${l}
// Mantissas are in [0.5, 1); their difference is exact. The atanh series computes the log
// ratio without rounding away near-equal values or depending on their common power-of-two scale.
fn getHuffLogRatio(value: f32, reference: f32) -> f32 {
  let first = frexp(value);
  let second = frexp(reference);
  var firstFraction = first.fract;
  var differencePower = first.exp - second.exp;
  // Put near-equal values on the same binary scale, including across a power-of-two boundary.
  // The remaining mantissa ratio stays in [0.5, 2], so |difference| <= 1/3.
  if (differencePower < 0 && firstFraction > second.fract) {
    firstFraction *= 0.5;
    differencePower += 1;
  } else if (differencePower > 0 && firstFraction < second.fract) {
    firstFraction *= 2.0;
    differencePower -= 1;
  }
  let difference = (firstFraction - second.fract) / (firstFraction + second.fract);
  let square = difference * difference;
  let series = 1.0 + square * (1.0 / 3.0 + square * (1.0 / 5.0 + square * (1.0 / 7.0 +
    square * (1.0 / 9.0 + square * (1.0 / 11.0 + square * (1.0 / 13.0 + square / 15.0))))));
  return 2.0 * difference * series + f32(differencePower) * 0.6931471805599453;
}
fn getHuffScore(value: f32, weight: f32, reference: f32, referenceWeight: f32, exponent: f32) -> f32 {
  if (exponent == 1.0) {
    // Bounded products preserve exact weighted ties at the default exponent, even if A*w
    // itself would overflow. Multiplying both attractiveness values by 2^k leaves this unchanged.
    let first = frexp(value);
    let second = frexp(weight);
    let firstReference = frexp(reference);
    let secondReference = frexp(referenceWeight);
    return getHuffLogRatio(first.fract * second.fract, firstReference.fract * secondReference.fract) +
      f32(first.exp + second.exp - firstReference.exp - secondReference.exp) * 0.6931471805599453;
  }
  return exponent * getHuffLogRatio(value, reference) + getHuffLogRatio(weight, referenceWeight);
}
fn getHuffShare(value: f32, weight: f32, reference: f32, referenceWeight: f32, maximum: f32, exponent: f32) -> f32 {
  if (!(value > 0.0) || !(weight > 0.0)) { return 0.0; }
  return exp(min(getHuffScore(value, weight, reference, referenceWeight, exponent) - maximum, 0.0));
}
fn getHuffContribution(value: f32, weight: f32, reference: f32, referenceWeight: f32, maximum: f32, total: f32, exponent: f32, demand: f32) -> f32 {
  if (!(value > 0.0) || !(weight > 0.0) || demand == 0.0) { return 0.0; }
  let score = min(getHuffScore(value, weight, reference, referenceWeight, exponent) - maximum, 0.0);
  // Below exp(-256), even the largest finite demand cannot produce a float32 subnormal.
  // This also keeps conversion to an integer exponent finite and in range.
  if (score < -256.0) { return 0.0; }
  let power = i32(floor(score * 1.4426950408889634));
  let fraction = exp(score - f32(power) * 0.6931471805599453);
  let demandParts = frexp(demand);
  return getScaledQuotient(demandParts.fract * fraction, total, 1.0, demandParts.exp + power);
}`,S=e=>`for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < FACILITIES) {
      let value = attraction[attractionOffset + partner];
      let weight = weights[weightsOffset + slot];
      if (value > 0.0 && weight > 0.0) {
      ${e}
      }
    }
  }`,C=[{name:`offsets`,view:h.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:h.neighbors,type:`u32`,access:`read`},{name:`weights`,view:h.weights,type:`f32`,access:`read`},{name:`attraction`,view:b,type:`f32`,access:`read`},...y],w=[n(e,{id:`${a}-attraction`,operation:D,variant:`attraction`,bindings:[{name:`attractiveness`,view:m,type:`f32`,access:`read`},{name:`attraction`,view:b,type:`f32`,access:`read_write`}],invocationCount:d,body:`attraction[attractionOffset + index] = attractiveness[attractivenessOffset + index];`}),n(e,{id:`${a}-trade-area`,operation:D,variant:`trade-area`,bindings:[...C,{name:`normalization`,view:v,type:`f32`,access:`read_write`},{name:`tradeArea`,view:_.tradeArea,type:`u32`,access:`read_write`},..._.tradeAreaProbability?[{name:`tradeProbability`,view:_.tradeAreaProbability,type:`f32`,access:`read_write`}]:[]],invocationCount:p,declarations:x,body:`let exponent = ${g?`parameters[parametersOffset]`:`1.0`};
  var reference = 0.0;
  var referenceWeight = 0.0;
  ${S(`reference = max(reference, value); referenceWeight = max(referenceWeight, weight);`)}
  var maximum = -3.4e38;
  ${S(`maximum = max(maximum, getHuffScore(value, weight, reference, referenceWeight, exponent));`)}
  var total = 0.0;
  var best = 0.0;
  var bestFacility = ${O}u;
  ${S(`let share = getHuffShare(value, weight, reference, referenceWeight, maximum, exponent);
      total += share;
      if (share > best || (share == best && share > 0.0 && partner < bestFacility)) {
        best = share;
        bestFacility = partner;
      }`)}
  normalization[normalizationOffset + 4u * index] = reference;
  normalization[normalizationOffset + 4u * index + 1u] = referenceWeight;
  normalization[normalizationOffset + 4u * index + 2u] = maximum;
  normalization[normalizationOffset + 4u * index + 3u] = total;
  tradeArea[tradeAreaOffset + index] = select(${O}u, bestFacility, total > 0.0);
  ${_.tradeAreaProbability?`tradeProbability[tradeProbabilityOffset + index] = select(0.0, getSpatialAnalysisQuotient(best,total), total > 0.0);`:``}`})];if(_.probabilities&&w.push(n(e,{id:`${a}-clear-probabilities`,operation:D,variant:`clear-probabilities`,bindings:[{name:`probabilities`,view:_.probabilities,type:`f32`,access:`read_write`}],invocationCount:_.probabilities.length,body:`probabilities[probabilitiesOffset + index] = 0.0;`}),n(e,{id:`${a}-probabilities`,operation:D,variant:`probabilities`,bindings:[...C,{name:`normalization`,view:v,type:`f32`,access:`read`},{name:`probabilities`,view:_.probabilities,type:`f32`,access:`read_write`}],invocationCount:p,declarations:x,body:`let reference = normalization[normalizationOffset + 4u * index];
  let referenceWeight = normalization[normalizationOffset + 4u * index + 1u];
  let maximum = normalization[normalizationOffset + 4u * index + 2u];
  let total = normalization[normalizationOffset + 4u * index + 3u];
  let exponent = ${g?`parameters[parametersOffset]`:`1.0`};
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    var probability = 0.0;
    if (partner < FACILITIES && total > 0.0) {
      probability = getSpatialAnalysisQuotient(getHuffShare(attraction[attractionOffset + partner], weights[weightsOffset + slot], reference, referenceWeight, maximum, exponent),total);
    }
    probabilities[probabilitiesOffset + slot] = probability;
  }`})),_.expectedDemand&&u.facilityWeights&&u.demand){let t=u.facilityWeights;w.push(n(e,{id:`${a}-expected-demand`,operation:D,variant:`expected-demand`,bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t.weights,type:`f32`,access:`read`},{name:`attraction`,view:b,type:`f32`,access:`read`},{name:`demand`,view:u.demand,type:`f32`,access:`read`},{name:`normalization`,view:v,type:`f32`,access:`read`},...y,{name:`expectedDemand`,view:_.expectedDemand,type:`f32`,access:`read_write`}],invocationCount:d,declarations:x,body:`let exponent = ${g?`parameters[parametersOffset]`:`1.0`};
  var captured = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < DEMANDS) {
      let reference = normalization[normalizationOffset + 4u * partner];
      let referenceWeight = normalization[normalizationOffset + 4u * partner + 1u];
      let maximum = normalization[normalizationOffset + 4u * partner + 2u];
      let total = normalization[normalizationOffset + 4u * partner + 3u];
      let partnerDemand = demand[demandOffset + partner];
      // A non-finite demand is missing and adds nothing, as in GPUCatchmentAccessibility.
      if (total > 0.0 && isFiniteValue(partnerDemand)) {
        captured += getHuffContribution(attraction[attractionOffset + index], weights[weightsOffset + slot], reference, referenceWeight, maximum, total, exponent, partnerDemand);
      }
    }
  }
  expectedDemand[expectedDemandOffset + index] = captured;`}))}_.validity&&u.demand&&w.push(n(e,{id:`${a}-validity`,operation:D,variant:`validity`,bindings:[{name:`demand`,view:u.demand,type:`f32`,access:`read`},{name:`validity`,view:_.validity,type:`u32`,access:`read_write`}],invocationCount:p,declarations:c,body:`validity[validityOffset + index] = u32(isFiniteValue(demand[demandOffset + index]));`}));let T=N(h),E=_.expectedDemand&&u.facilityWeights?N(u.facilityWeights):[];for(let[t,r,i,o,c]of[[`tradeArea`,_.tradeArea,T,`u32`,`${O}u`],[`probabilities`,_.probabilities,T,`f32`,`getNaN()`],[`tradeAreaProbability`,_.tradeAreaProbability,T,`f32`,`getNaN()`],[`expectedDemand`,_.expectedDemand,[...T,...E],`f32`,`getNaN()`]])r&&i.length!==0&&w.push(n(e,{id:`${a}-suppress-${t}`,operation:D,variant:`incomplete-input`,bindings:[...i.map((e,t)=>({name:`loss${t}`,view:e,type:`u32`,access:`read`})),{name:`result`,view:r,type:o,access:`read_write`}],invocationCount:r.length,declarations:o===`f32`?s:void 0,body:`if (${i.map((e,t)=>`loss${t}[loss${t}Offset] != 0u`).join(` || `)}) {
  result[resultOffset + index] = ${c};
}`}));return w.flat()}};function j(e){return[e.probabilities,e.tradeArea,e.tradeAreaProbability,e.expectedDemand,e.validity]}function M(e){return[e.attractiveness,e.output.expectedDemand||e.output.validity?e.demand:void 0,e.parameters,e.demandWeights.offsets,e.demandWeights.neighbors,e.demandWeights.weights,...N(e.demandWeights),...e.output.expectedDemand?[e.facilityWeights?.offsets,e.facilityWeights?.neighbors,e.facilityWeights?.weights,...e.facilityWeights?N(e.facilityWeights):[]]:[]]}function N(e){return[e.status?.overflow,e.status?.candidateOverflow].filter(e=>e!==void 0)}var P=h({parameters:E,caption:`Each demand location is colored by the facility with the highest Huff probability.`,createInputs:({random:e,world:t})=>{let n=v(e,t,6,360);return{...n,drag:g(n)}},build(t,n){let{network:r}=n,i=y(t,r),a=t.upload(`facility-nodes`,`uint32`,n.facilityNodes),o=t.upload(`facilities`,`float32x2`,n.facilityPositions),s=b(t,i,a,r.nodeCount),c=t.allocate(`decay-parameters`,`float32`,4),l=t.allocate(`exponent`,`float32`,1),u=n.facilityNodes.length,d=S(t,s,u,n.demandNodes,`columns`,c),f=t.allocate(`trade-area`,`uint32`,n.demandNodes.length);return t.graph.add(new A({attractiveness:t.upload(`attractiveness`,`float32`,n.supply).view,demandWeights:d.spatialWeights,parameters:l.view,output:{tradeArea:f.view}})),{outputs:{...T(t,r),tradeArea:f,demands:t.upload(`demands`,`float32x2`,n.demandPositions),facilities:o,facilityIds:t.upload(`facility-ids`,`uint32`,x(u))},update:t=>{a.write(n.facilityNodes),o.write(n.facilityPositions),i.costLimit.write(Float32Array.of(2.5)),c.write(e({costLimit:t.costLimit,decay:`exponential`,beta:t.beta})),l.write(k({exponent:t.exponent}))}}},interaction:{kind:`drag`,hint:`drag a facility`,rest:({inputs:e,toUnit:t})=>t([e.drag.defaultPositions[0],e.drag.defaultPositions[1]]),resolve(e,{inputs:t,values:n,peek:r,outputs:i}){let a=_(t,t.drag,e.x,e.y),o=[t.network.nodes[a.node*2],t.network.nodes[a.node*2+1]];return{rerun:a.changed,glyphs:[{shape:`crosshair`,at:o,size:14}],readout:`facility ${a.facility+1} · exponent ${n.exponent} · beta ${n.beta}`}}},draw:({outputs:e,theme:t})=>[w(e),{layer:`dots`,positions:e.demands,values:e.tradeArea,radius:8},...C(e.facilities,t,{ids:e.facilityIds,radius:9})]});export{P as default};
import{o as e}from"./raster-band-utils-BcTxc8_Y.js";import{c as t,d as n,f as r,h as i,m as a,p as o,s,u as c}from"./raster-algebra-input-5ZxymrFJ.js";import{n as l}from"./raster-storage-preflight-CIxFc-0g.js";import{r as u}from"./kernel-C3tfQOn4.js";import{o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{p as f,s as p}from"./gpu-contributor-utils-BIwLQON5.js";import{a as m,i as h,n as g,o as _,r as v,t as y}from"./raster-algebra-stack-input-DZ5O9caI.js";import{i as b,r as x,t as S}from"./wgsl-common-DfyXiYB0.js";import{a as C}from"./pointer-overlay-Qh8qflcA.js";import{t as w}from"./define-run-B-brkCe4.js";import{a as T}from"./inputs-CJKKIRKK.js";import{yi as E}from"./entry-preview-D70TdSbK.js";import{t as D}from"./bands-CVDIl0aE.js";var O=8,k=8;function A(e){return O+k*e}function j(t,n=new Float32Array(A(t.layers.length)),r={}){let i=A(t.layers.length);e(`getGPUWeightedOverlayParameterValues`,n,i),n.fill(0,0,i),n[0]=+!!t.normalizeWeights,n[1]=+(t.noDataPolicy===`ignore`);for(let[e,i]of t.layers.entries()){let t=i.inputMinimum??0,a=i.inputMaximum??1,o=i.breakCount??0;for(let[n,r]of[[`weight`,i.weight],[`inputMinimum`,t],[`inputMaximum`,a]])if(!Number.isFinite(r))throw Error(`getGPUWeightedOverlayParameterValues layer ${e} ${n} must be finite`);if(!Number.isSafeInteger(o)||o<0)throw Error(`getGPUWeightedOverlayParameterValues layer ${e} breakCount must be a non-negative integer`);if(i.mode===`table`&&r.hasTables===!1)throw Error(`getGPUWeightedOverlayParameterValues layer ${e} is a table layer but has no remap tables`);let s=O+e*k,c=Math.fround(a)===Math.fround(t);n[s]=i.weight,n[s+1]=+(i.mode===`table`),n[s+2]=t,n[s+3]=c?0:1/(Math.fround(a)-Math.fround(t)),n[s+4]=+!!i.invert,n[s+5]=o,n[s+6]=+(i.closed===`right`),n[s+7]=+!!c}return n}var M=`GPUWeightedOverlay`,N=256,P=3,F=class{id;estimatedCommandNodeCount;props;maximumBreakCount;constructor(e){this.id=e.id??`weighted-overlay`,this.props=e;let{id:t}=this,{output:o,layerCount:s,cellCount:c}=e;if(r(t,`layerCount`,s,1,16),r(t,`cellCount`,c,0,4294967295),s*c>4294967295)throw Error(`${t} layerCount * cellCount must fit in uint32`);if(a(t,e.noDataValue),_(t,e.stack,e.grid,s,c),this.estimatedCommandNodeCount=P+g(e.stack),f(t,this.estimatedCommandNodeCount),i(t,`mask`,e.mask,`uint32`,c),i(t,`parameters`,e.parameters,`float32`,A(s)),!!e.remapBreaks!=!!e.remapValues)throw Error(`${t} remapBreaks and remapValues must be provided together`);if(this.maximumBreakCount=e.remapBreaks?e.maximumBreakCount??0:0,e.remapBreaks&&(r(t,`maximumBreakCount`,this.maximumBreakCount,1,65536),i(t,`remapBreaks`,e.remapBreaks,`float32`,s*this.maximumBreakCount),i(t,`remapValues`,e.remapValues,`float32`,s*(this.maximumBreakCount+1))),!o?.score)throw Error(`${t} needs output.score`);i(t,`output.score`,o.score,`float32`,c),i(t,`output.validity`,o.validity,`uint32`,c),i(t,`output.scoreRange`,o.scoreRange,`float32`,2),n(t,{"output.score":o.score,"output.validity":o.validity,"output.scoreRange":o.scoreRange},{...y(e.stack),mask:e.mask,parameters:e.parameters,remapBreaks:e.remapBreaks,remapValues:e.remapValues})}getCommandNodes(e){let n=this.props;s(e,this.id,m(n.stack)?n.stack:[n.stack],n.grid),l(e,this.id,[n.mask,n.parameters,n.remapBreaks,n.remapValues,...p(n.output)],{"packed band stack":m(n.stack)?n.layerCount*n.cellCount*4:0,"range keys":n.output.scoreRange?8:0});let{id:r,props:i,maximumBreakCount:a}=this,{output:f,layerCount:g,cellCount:_}=i;o(r,e,[...h(i.stack),i.mask,i.parameters,i.remapBreaks,i.remapValues,f.score,f.validity,f.scoreRange]);let y=v(e,r,i.stack,i.grid,_),C=[...y.nodes],w=f.scoreRange?u(e,`${r}-range-keys`,`uint32`,2):void 0;w&&C.push(d(e,{id:`${r}-clear-range`,operation:M,variant:`clear-range`,bindings:[{name:`keys`,view:w,type:`u32`,access:`read_write`}],invocationCount:1,body:`keys[keysOffset] = 0xffffffffu;
  keys[keysOffset + 1u] = 0u;`}));let T=!!i.remapBreaks,E=[{name:`stack`,view:y.values,type:`f32`,access:`read`},...i.mask?[{name:`mask`,view:i.mask,type:`u32`,access:`read`}]:[],{name:`params`,view:i.parameters,type:`f32`,access:`read`}];return T&&E.push({name:`remapBreaks`,view:i.remapBreaks,type:`f32`,access:`read`},{name:`remapValues`,view:i.remapValues,type:`f32`,access:`read`}),E.push({name:`scoreOut`,view:f.score,type:`f32`,access:`read_write`}),f.validity&&E.push({name:`validityOut`,view:f.validity,type:`u32`,access:`read_write`}),w&&E.push({name:`rangeKeys`,view:w,type:`atomic<u32>`,access:`read_write`}),C.push(d(e,{id:`${r}-score`,operation:M,variant:`score`,bindings:E,invocationCount:_,guardIndex:!w,workgroupSize:w?N:void 0,declarations:`const LAYER_COUNT: u32 = ${g}u;
const CELL_COUNT: u32 = ${_}u;
const MAXIMUM_BREAK_COUNT: u32 = ${a}u;
${w?`var<workgroup> tileMinimum: array<u32, ${N}>;
var<workgroup> tileMaximum: array<u32, ${N}>;`:``}
${c(m(i.stack)?void 0:i.noDataValue)}
${T?t(`countBreaksBelow`,`remapBreaks`):``}
${b}`,body:`${w?`var minimumKey = 0xffffffffu;
  var maximumKey = 0u;
  if (index < CELL_COUNT) {`:``}
  let normalizeWeights = params[paramsOffset] != 0.0;
  let ignoreNoData = params[paramsOffset + 1u] != 0.0;
  var sum = 0.0;
  var weightSum = 0.0;
  var usedCount = 0u;
  var isDefined = true;
  for (var layer = 0u; layer < LAYER_COUNT; layer++) {
    let base = paramsOffset + 8u + layer * 8u;
    let weight = params[base];
    let value = stack[stackOffset + layer * CELL_COUNT + index];
    if (isNoDataValue(value)) {
      if (!ignoreNoData) {
        isDefined = false;
      }
      continue;
    }
    var remapped = 0.0;
    if (params[base + 1u] == 0.0) {
      let inputMinimum = params[base + 2u];
      if (params[base + 7u] != 0.0) {
        remapped = select(0.0, 1.0, value >= inputMinimum);
      } else {
        remapped = clamp((value - inputMinimum) * params[base + 3u], 0.0, 1.0);
      }
      if (params[base + 4u] != 0.0) {
        remapped = 1.0 - remapped;
      }
    } else {
      ${T?`let breakCountValue = params[base + 5u];
      let breakCount = min(u32(clamp(select(0.0, breakCountValue, isFiniteValue(breakCountValue)), 0.0, 4294967040.0)), MAXIMUM_BREAK_COUNT);
      let classIndex = countBreaksBelow(layer * MAXIMUM_BREAK_COUNT, breakCount, value, params[base + 6u] != 0.0);
      remapped = remapValues[remapValuesOffset + layer * (MAXIMUM_BREAK_COUNT + 1u) + classIndex];`:`remapped = getNaN();`}
    }
    if (isNaNValue(remapped)) {
      // Restricted class: the cell is excluded whatever the nodata policy.
      isDefined = false;
    }
    sum = sum + weight * remapped;
    weightSum = weightSum + abs(weight);
    usedCount = usedCount + 1u;
  }
  isDefined = isDefined && usedCount > 0u${i.mask?` && mask[maskOffset + index] != 0u`:``};
  var score = sum;
  if (normalizeWeights) {
    isDefined = isDefined && weightSum > 0.0;
    score = sum / weightSum;
  }
  score = select(getNaN(), score, isDefined && !isNaNValue(score));
  let isScoreDefined = !isNaNValue(score);
  scoreOut[scoreOutOffset + index] = score;
  ${f.validity?`validityOut[validityOutOffset + index] = select(0u, 1u, isScoreDefined);`:``}
  ${w?`if (isScoreDefined) {
    let key = getOrderedKey(score);
    minimumKey = key;
    maximumKey = key;
  }
  }
  // Combine the workgroup's keys in a tree; one pair of global atomics per workgroup, not per cell.
  tileMinimum[localInvocationIndex] = minimumKey;
  tileMaximum[localInvocationIndex] = maximumKey;
  workgroupBarrier();
  for (var stride = ${N/2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      tileMinimum[localInvocationIndex] = min(tileMinimum[localInvocationIndex], tileMinimum[localInvocationIndex + stride]);
      tileMaximum[localInvocationIndex] = max(tileMaximum[localInvocationIndex], tileMaximum[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && tileMinimum[0] <= tileMaximum[0]) {
    atomicMin(&rangeKeys[rangeKeysOffset], tileMinimum[0]);
    atomicMax(&rangeKeys[rangeKeysOffset + 1u], tileMaximum[0]);
  }`:``}`})),w&&f.scoreRange&&C.push(d(e,{id:`${r}-decode-range`,operation:M,variant:`decode-range`,bindings:[{name:`keys`,view:w,type:`u32`,access:`read`},{name:`rangeOut`,view:f.scoreRange,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${x}
${S}`,body:`let minimumKey = keys[keysOffset];
  let maximumKey = keys[keysOffset + 1u];
  let isEmpty = minimumKey > maximumKey;
  rangeOut[rangeOutOffset] = select(decodeOrderedKey(minimumKey), getNaN(), isEmpty);
  rangeOut[rangeOutOffset + 1u] = select(decodeOrderedKey(maximumKey), getNaN(), isEmpty);`})),C}},I=3;function L(e){let t=Float32Array.from(e).sort();return[t[Math.floor(t.length*.05)],t[Math.floor(t.length*.95)]]}var R=C(E,{kind:`pad`,hint:`weight NIR and SWIR`,x:`nirWeight`,y:`shortwaveWeight`}),z=w({parameters:E,caption:`Three layers in; the weighted suitability score out.`,createInputs:({world:e})=>{let t=T(e);return{...t,spreads:[L(t.green),L(t.shortwaveInfrared),L(t.nearInfrared)]}},build(e,t){let{spreads:n}=t,r=t.values.length,i=new Float32Array(r*I);i.set(t.green,0),i.set(t.shortwaveInfrared,r),i.set(t.nearInfrared,r*2);let a=e.upload(`stack`,`float32`,i),o=e.allocate(`parameters`,`float32`,A(I)),s=e.allocate(`score`,`float32`,r);return e.graph.add(new F({stack:a.view,layerCount:I,cellCount:r,parameters:o.view,output:{score:s.view}})),{outputs:{shortwave:e.upload(`shortwave`,`float32`,t.shortwaveInfrared),score:s},update:e=>o.write(j({layers:[{weight:e.greenWeight,inputMinimum:n[0][0],inputMaximum:n[0][1]},{weight:e.shortwaveWeight,inputMinimum:n[1][0],inputMaximum:n[1][1]},{weight:e.nirWeight,inputMinimum:n[2][0],inputMaximum:n[2][1]}],normalizeWeights:e.normalizeWeights===1}))}},interaction:{...R,resolve:(e,t)=>{let n=R.resolve(e,t),r={...t.values,...n.values};return{...n,glyphs:[{shape:`crosshair`,at:[e.x,e.y]}],readout:`NIR ${r.nirWeight.toFixed(2)} | SWIR ${r.shortwaveWeight.toFixed(2)}`}}},draw:({inputs:e,outputs:t})=>[D(e,t.shortwave,{context:!0}),D(e,t.score,{ramp:`sequential`,range:[-.5,1],smoothing:!0})]});export{z as default};
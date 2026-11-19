import{i as e,r as t,s as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{t as i}from"./gpu-uint32-gather-C7ioZBaz.js";import{i as a}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as o,o as s,t as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as l}from"./sorted-segment-sums-Dvmm6OdR.js";import{t as u}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{o as d,t as f}from"./fixed-point-accumulator-Ce6W3HVY.js";import{A as p,E as m,O as h,S as g,T as _,f as v,k as y,m as b,u as ee,v as te,w as x,x as ne,y as S}from"./gpu-contributor-utils-BIwLQON5.js";import{i as C,t as w}from"./wgsl-common-DfyXiYB0.js";import{t as T}from"./network-workgroup-counters-BG1Cyc9s.js";import{t as E}from"./define-run-B-brkCe4.js";import{n as D}from"./display-Bu20g6o5.js";import{_ as re,i as O,m as k,p as A,r as j,v as M}from"./roads-DzovamHj.js";import{s as N}from"./pointer-overlay-Qh8qflcA.js";import{ja as P}from"./entry-preview-D70TdSbK.js";var F=429496729;function ie(e){return Math.min(z,ee(e,8),Math.floor(e.limits.maxBufferSize/8))}var ae={liveNodeCount:0,overflowedNodeCount:1,countedEdgeCount:2,droppedEdgeCount:3,intraEdgeCount:4,interEdgeCount:5,groupCount:6,superedgeCount:7},I=4294967295,L=4294967294,R=4294967295,z=2147483647,B=`GPUNetworkCoarsening`,V=10,H=16,U=ae,W={positions:8,values:9,weights:10},oe=11,se=127;function G(e){let t=BigInt(0x3782dac000000000),n=e?`const SATURATION_LOW: u32 = ${t&4294967295n}u;
const SATURATION_HIGH: u32 = ${t>>32n}u;
fn quantize(value: f32, unitExponent: i32) -> vec2<u32> {
  let wide = quantizeFixedPoint(abs(value), unitExponent);
  var magnitude = wide.xy;
  let isInfinite = (bitcast<u32>(value) & 0x7fffffffu) == 0x7f800000u;
  if (isInfinite || wide.z != 0u || wide.y > SATURATION_HIGH ||
      (wide.y == SATURATION_HIGH && wide.x > SATURATION_LOW)) {
    magnitude = vec2<u32>(SATURATION_LOW, SATURATION_HIGH);
  }
  if (value < 0.0) {
    let low = ~magnitude.x + 1u;
    magnitude = vec2<u32>(low, ~magnitude.y + select(0u, 1u, low == 0u));
  }
  return magnitude;
}`:`fn quantize(value: f32, unitExponent: i32) -> vec2<u32> {
  return quantizeFixedPoint(value, unitExponent).xy;
}`;return`${f}
fn getSumUnitExponent(maximumExponentBits: u32, headroomBits: u32) -> i32 {
  return getFixedPointUnitExponent(maximumExponentBits, headroomBits) + 32;
}
${n}
fn dequantize(low: u32, high: u32, unitExponent: i32) -> f32 {
  let signWord = select(0u, 0xffffffffu, (high & 0x80000000u) != 0u);
  return getFixedPointFloat(vec3<u32>(low, high, signWord), unitExponent);
}
${C}${w}`}function K(e,t,n){return`{
    let addendOld = atomicAdd(&${e}[${t}], ${n}.x);
    let addendCarry = select(0u, 1u, addendOld + ${n}.x < addendOld);
    atomicAdd(&${e}[${t} + 1u], ${n}.y + addendCarry);
  }`}var ce=class{id;props;nodeCount;groupCapacity;edgeCapacity;constructor(t){this.id=t.id??`network-coarsening`,this.props=t;let{id:n}=this,{output:r}=t,i=p(n,t.topology);m(t.labels,[`uint32`],`${n} labels`),v(n,`topology.neighbors.length`,i.edgeCount,0,I),this.nodeCount=i.nodeCount;let{nodeCount:o}=this;if(v(n,`topology node count`,o,1,L),i.edgeCount<1)throw Error(`${n} topology.neighbors must contain at least one slot`);this.groupCapacity=t.groupCapacity;let s=this.groupCapacity;v(n,`groupCapacity`,s,1,F),t.fixedPointScale!==void 0&&x(n,`fixedPointScale`,t.fixedPointScale,1,2**24);let c=[[`nodeMask`,t.nodeMask,o,[`uint32`]],[`labels`,t.labels,o,[`uint32`]],[`nodeValues`,t.nodeValues,o,[`float32`]],[`edgeMask`,t.edgeMask,t.topology.neighbors.length,[`uint32`]],[`output.groupNodeCount`,r.groupNodeCount,s,[`uint32`]],[`output.groupIntraEdgeCount`,r.groupIntraEdgeCount,s,[`uint32`]],[`output.groupIntraWeight`,r.groupIntraWeight,s,[`float32`]],[`output.groupCentroid`,r.groupCentroid,s,[`float32x2`]],[`output.groupBounds`,r.groupBounds,s,[`float32x4`]],[`output.groupValueSum`,r.groupValueSum,s,[`float32`]]];for(let[e,t,r,i]of c)if(t&&(m(t,i,`${n} ${e}`),t.length!==r))throw Error(`${n} ${e} must contain exactly ${r} rows`);if((r.groupCentroid||r.groupBounds)&&!t.topology.nodePositions)throw Error(`${n} output.groupCentroid and output.groupBounds require topology.nodePositions`);if(r.groupValueSum&&!t.nodeValues)throw Error(`${n} output.groupValueSum requires nodeValues`);if(b(n,r.edges),this.edgeCapacity=r.edges.ids.length,this.edgeCapacity<1)throw Error(`${n} output.edges.ids must contain at least one row`);v(n,`output.edges.ids rows`,this.edgeCapacity,1,R);for(let[e,t,i]of[[`output.edgeTargetGroups`,r.edgeTargetGroups,[`uint32`]],[`output.edgeCounts`,r.edgeCounts,[`uint32`]],[`output.edgeWeights`,r.edgeWeights,[`float32`]]])if(t&&(m(t,i,`${n} ${e}`),t.length!==this.edgeCapacity))throw Error(`${n} ${e} must match output.edges.ids length`);if(t.output.summary&&(m(t.output.summary,[`uint32`],`${n} output.summary`),t.output.summary.length!==8))throw Error(`${n} output.summary must contain exactly 8 rows`);S(n,this.getOutputs(),this.getInputs());let l=this.getOutputs().filter(e=>!!e),u=new Set(this.getInputs().flatMap(e=>e?.buffer.defaultBuffer?[h(e.buffer.defaultBuffer)]:[]));for(let e of l)_(n,!e.buffer.defaultBuffer||!u.has(h(e.buffer.defaultBuffer)),`outputs must not share buffers with inputs`);for(let[t,r]of l.entries())for(let i of l.slice(t+1))_(n,r.buffer===i.buffer||!r.buffer.defaultBuffer||!i.buffer.defaultBuffer||h(r.buffer.defaultBuffer)!==h(i.buffer.defaultBuffer),`outputs sharing physical storage must use one graph buffer handle`),_(n,!e(r,i),`output writable byte ranges must not overlap`);let d=[[r.edges.ids,r.edgeTargetGroups,r.edgeCounts],[r.groupNodeCount,r.groupCentroid,r.groupBounds,r.groupValueSum],[r.groupIntraEdgeCount,r.groupIntraWeight],[r.edges.status.count,r.edges.status.overflow,r.edges.status.requiredCount,r.edges.status.candidateOverflow]];for(let e of d)a(n,Object.fromEntries(e.map((e,t)=>[`output[${t}]`,e])),{strictAlignedBindings:!0})}getInputs(){let{props:e}=this;return[...y(e.topology),e.nodeMask,e.edgeMask,e.labels,e.nodeValues]}getOutputs(){let{output:e}=this.props;return[e.groupNodeCount,e.groupIntraEdgeCount,e.groupIntraWeight,e.groupCentroid,e.groupBounds,e.groupValueSum,e.edges.ids,e.edges.status.count,e.edges.status.overflow,e.edges.status.candidateOverflow,e.edges.status.requiredCount,e.edgeTargetGroups,e.edgeCounts,e.edgeWeights,e.summary]}getCommandNodes(e){let{id:a,props:p,nodeCount:m,groupCapacity:h,edgeCapacity:_}=this;ne(a,e,[...this.getInputs(),...this.getOutputs()]),p.output.edgeWeights&&g(a,`output.edges.ids rows with output.edgeWeights`,_,ie(e.device));let v=p.topology.neighbors.length;for(let t of[...this.getInputs(),...this.getOutputs()])t&&(g(a,`packed view buffer byteLength`,t.buffer.byteLength,e.device.limits.maxBufferSize),te(e.device,a,`packed view`,n(t).size,`reduce the packed view length`));for(let[t,n]of[[`row scratch`,Math.max(m,v,1)*4],[`group accumulator`,h*V*4],[`intra accumulator`,p.output.groupIntraEdgeCount||p.output.groupIntraWeight?h*3*4:0],[`edge weight accumulator`,p.output.edgeWeights?_*8:0],[`statistics`,44]])g(a,`${t} byteLength`,n,e.device.limits.maxBufferSize),te(e.device,a,t,n,`reduce groupCapacity, topology size or output capacity`);let y=!!p.directed,b=p.fixedPointScale===void 0,ee=b?0:-Math.log2(p.fixedPointScale),x=b&&!!p.topology.weights,S=(e,t,n)=>b?`getSumUnitExponent(${e===void 0?`${se}u`:n===`atomic`?`atomicLoad(&stats[statsOffset + ${e}u])`:`stats[statsOffset + ${e}u]`}, ${d(t)}u)`:`${ee}`,C=e=>S(p.topology.weights?W.weights:void 0,p.topology.neighbors.length,e),w=h,E=!!p.topology.nodePositions,D=!!p.nodeValues,re=!!(p.output.groupIntraEdgeCount||p.output.groupIntraWeight),O=!!p.output.groupIntraWeight,k=[],A=(n,r)=>t(e,`${a}-${n}`,`uint32`,r),j=A(`effective-labels`,m),M=A(`group-acc`,h*V),N=re?A(`intra-acc`,h*3):void 0,P=A(`stats`,oe),F=A(`sort-keys`,v),ae=A(`sort-indices`,v),I=A(`target-keys`,v),L=A(`segment-flags`,v),R=A(`segment-scan`,v),z=A(`segment-total`,1),ce=A(`overflow-flag`,1),q=p.output.edgeWeights?A(`segment-weight-acc`,_*2):void 0,J=(t,n,r=`u32`,i=`0u`)=>{n&&k.push(c(e,{id:`${a}-${t}`,operation:B,view:n,type:r,value:i}))};if(J(`zero-stats`,P),J(`zero-group-counts`,p.output.groupNodeCount),J(`zero-intra`,N),J(`zero-edge-ids`,p.output.edges.ids),J(`zero-edge-targets`,p.output.edgeTargetGroups),J(`zero-edge-counts`,p.output.edgeCounts),J(`zero-edge-weight-acc`,q),J(`clear-sort-keys`,F,`u32`,`${w}u`),J(`clear-target-keys`,I,`u32`,`${w}u`),k.push(s(e,{id:`${a}-init-groups`,operation:B,variant:`init-groups`,bindings:[{name:`groupAcc`,view:M,type:`u32`,access:`read_write`}],invocationCount:h,body:`let base = index * ${V}u;
  for (var word = 0u; word < 6u; word++) {
    groupAcc[groupAccOffset + base + word] = 0u;
  }
  groupAcc[groupAccOffset + base + 6u] = 0xffffffffu;
  groupAcc[groupAccOffset + base + 7u] = 0xffffffffu;
  groupAcc[groupAccOffset + base + 8u] = 0u;
  groupAcc[groupAccOffset + base + 9u] = 0u;`})),b&&(E||D)){let t=[{name:`labels`,view:p.labels,type:`u32`,access:`read`}];p.nodeMask&&t.push({name:`vertexMask`,view:p.nodeMask,type:`u32`,access:`read`}),p.topology.nodePositions&&t.push({name:`positions`,view:p.topology.nodePositions,type:`f32`,access:`read`}),p.nodeValues&&t.push({name:`values`,view:p.nodeValues,type:`f32`,access:`read`}),t.push({name:`stats`,view:P,type:`atomic<u32>`,access:`read_write`});let n=T({counters:[{target:`stats`,word:W.positions,operation:`max`},{target:`stats`,word:W.values,operation:`max`}],rowCount:`NODE_COUNT`,rowSource:`${p.nodeMask?`if (vertexMask[vertexMaskOffset + index] == 0u) { return; }`:``}
  if (labels[labelsOffset + index] >= GROUP_CAPACITY) {
    return;
  }
  ${E?`maxCounter(0u, max(
    getFixedPointExponentBits(positions[positionsOffset + 2u * index]),
    getFixedPointExponentBits(positions[positionsOffset + 2u * index + 1u])));`:``}
  ${D?`maxCounter(1u, getFixedPointExponentBits(values[valuesOffset + index]));`:``}`});k.push(s(e,{id:`${a}-node-sum-scale`,operation:B,variant:`node-sum-scale`,bindings:t,invocationCount:m,workgroupSize:256,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${m}u;
const GROUP_CAPACITY: u32 = ${h}u;
${f}
${n.declarations}`,body:n.body}))}let Y=[{name:`labels`,view:p.labels,type:`u32`,access:`read`}];p.nodeMask&&Y.push({name:`vertexMask`,view:p.nodeMask,type:`u32`,access:`read`}),p.topology.nodePositions&&Y.push({name:`positions`,view:p.topology.nodePositions,type:`f32`,access:`read`}),p.nodeValues&&Y.push({name:`values`,view:p.nodeValues,type:`f32`,access:`read`}),Y.push({name:`effectiveLabels`,view:j,type:`u32`,access:`read_write`},{name:`groupCounts`,view:p.output.groupNodeCount,type:`atomic<u32>`,access:`read_write`},{name:`groupAcc`,view:M,type:`atomic<u32>`,access:`read_write`},{name:`stats`,view:P,type:`atomic<u32>`,access:`read_write`});let le=T({counters:[{target:`stats`,word:U.liveNodeCount},{target:`stats`,word:U.overflowedNodeCount},{target:`stats`,word:U.groupCount}],rowCount:`NODE_COUNT`,rowSource:`${p.nodeMask?`if (vertexMask[vertexMaskOffset + index] == 0u) {`:`if (false) {`}
    effectiveLabels[effectiveLabelsOffset + index] = DEAD;
    return;
  }
  addCounter(0u, 1u);
  let label = labels[labelsOffset + index];
  if (label >= GROUP_CAPACITY) {
    effectiveLabels[effectiveLabelsOffset + index] = OVERFLOWED;
    addCounter(1u, 1u);
    return;
  }
  effectiveLabels[effectiveLabelsOffset + index] = label;
  let previous = atomicAdd(&groupCounts[groupCountsOffset + label], 1u);
  if (previous == 0u) {
    addCounter(2u, 1u);
  }
  let base = label * ${V}u;
  ${E?`let position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let positionUnitExponent = ${S(W.positions,m,`atomic`)};
  let quantizedX = quantize(position.x, positionUnitExponent);
  let quantizedY = quantize(position.y, positionUnitExponent);
  ${K(`groupAcc`,`groupAccOffset + base`,`quantizedX`)}
  ${K(`groupAcc`,`groupAccOffset + base + 2u`,`quantizedY`)}
  atomicMin(&groupAcc[groupAccOffset + base + 6u], getOrderedKey(position.x));
  atomicMin(&groupAcc[groupAccOffset + base + 7u], getOrderedKey(position.y));
  atomicMax(&groupAcc[groupAccOffset + base + 8u], getOrderedKey(position.x));
  atomicMax(&groupAcc[groupAccOffset + base + 9u], getOrderedKey(position.y));`:``}
  ${D?`let quantizedValue = quantize(values[valuesOffset + index], ${S(W.values,m,`atomic`)});
  ${K(`groupAcc`,`groupAccOffset + base + 4u`,`quantizedValue`)}`:``}`});k.push(s(e,{id:`${a}-vertices`,operation:B,variant:`vertices`,bindings:Y,invocationCount:m,workgroupSize:256,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${m}u;
const GROUP_CAPACITY: u32 = ${h}u;
const DEAD: u32 = 0xffffffffu;
const OVERFLOWED: u32 = 0xfffffffeu;
${G(!b)}
${le.declarations}`,body:le.body}));let ue=O||!!p.output.edgeWeights;if(x&&ue&&p.topology.weights){let t=[{name:`offsets`,view:p.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:p.topology.neighbors,type:`u32`,access:`read`},{name:`effectiveLabels`,view:j,type:`u32`,access:`read`},{name:`weights`,view:p.topology.weights,type:`f32`,access:`read`}];p.edgeMask&&t.push({name:`edgeMask`,view:p.edgeMask,type:`u32`,access:`read`}),t.push({name:`stats`,view:P,type:`atomic<u32>`,access:`read_write`});let n=T({counters:[{target:`stats`,word:W.weights,operation:`max`}],rowCount:`NODE_COUNT`,rowSource:`let sourceLabel = effectiveLabels[effectiveLabelsOffset + index];
  if (sourceLabel >= GROUP_CAPACITY) {
    return;
  }
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  let rowBegin = min(offsets[offsetsOffset + index], rowEnd);
  var exponentBits = 0u;
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    let targetVertex = neighbors[neighborsOffset + slot];
    if (targetVertex >= NODE_COUNT) {
      continue;
    }
    var live = effectiveLabels[effectiveLabelsOffset + targetVertex] < GROUP_CAPACITY;
    ${p.edgeMask?`if (live) { live = edgeMask[edgeMaskOffset + slot] != 0u; }`:``}
    ${y?``:`live = live && index <= targetVertex;`}
    if (live) {
      exponentBits = max(exponentBits, getFixedPointExponentBits(weights[weightsOffset + slot]));
    }
  }
  maxCounter(0u, exponentBits);`});k.push(s(e,{id:`${a}-weight-sum-scale`,operation:B,variant:`weight-sum-scale`,bindings:t,invocationCount:m,workgroupSize:256,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${m}u;
const SLOT_COUNT: u32 = ${v}u;
const GROUP_CAPACITY: u32 = ${h}u;
${f}
${n.declarations}`,body:n.body}))}let de=p.topology.neighbors;p.edgeMask&&(de=A(`classified-neighbors`,v),k.push(s(e,{id:`${a}-mask-neighbors`,operation:B,invocationCount:v,bindings:[{name:`neighbors`,view:p.topology.neighbors,type:`u32`,access:`read`},{name:`edgeMask`,view:p.edgeMask,type:`u32`,access:`read`},{name:`classifiedNeighbors`,view:de,type:`u32`,access:`read_write`}],body:`classifiedNeighbors[classifiedNeighborsOffset + index] = select(0xffffffffu, neighbors[neighborsOffset + index], edgeMask[edgeMaskOffset + index] != 0u);`})));let X=[{name:`offsets`,view:p.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:de,type:`u32`,access:`read`},{name:`effectiveLabels`,view:j,type:`u32`,access:`read`}];p.topology.weights&&O&&X.push({name:`weights`,view:p.topology.weights,type:`f32`,access:`read`}),X.push({name:`sortKeys`,view:F,type:`u32`,access:`read_write`}),X.push({name:`targetKeys`,view:I,type:`u32`,access:`read_write`}),N&&X.push({name:`intraAcc`,view:N,type:`atomic<u32>`,access:`read_write`}),X.push({name:`stats`,view:P,type:`atomic<u32>`,access:`read_write`});let fe=T({counters:[{target:`stats`,word:U.countedEdgeCount},{target:`stats`,word:U.droppedEdgeCount},{target:`stats`,word:U.intraEdgeCount},{target:`stats`,word:U.interEdgeCount}],rowCount:`NODE_COUNT`,rowSource:`let sourceLabel = effectiveLabels[effectiveLabelsOffset + index];
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  let rowBegin = min(offsets[offsetsOffset + index], rowEnd);
  var counted = 0u;
  var dropped = 0u;
  var intra = 0u;
  var inter = 0u;
  var intraWeightLow = 0u;
  var intraWeightHigh = 0u;
  ${O?`let weightUnitExponent = ${C(`atomic`)};`:``}
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    var key = SENTINEL_KEY;
    var destinationKey = SENTINEL_KEY;
    let targetVertex = neighbors[neighborsOffset + slot];
    if (sourceLabel != DEAD && targetVertex < NODE_COUNT) {
      let targetLabel = effectiveLabels[effectiveLabelsOffset + targetVertex];
      var live = targetLabel != DEAD;
      ${y?``:`live = live && index <= targetVertex;`}
      if (live) {
        counted++;
        if (sourceLabel == OVERFLOWED || targetLabel == OVERFLOWED) {
          dropped++;
        } else if (sourceLabel == targetLabel) {
          intra++;
          ${O?`let quantizedWeight = quantize(${p.topology.weights?`weights[weightsOffset + slot]`:`1.0`}, weightUnitExponent);
          let previousLow = intraWeightLow;
          intraWeightLow += quantizedWeight.x;
          intraWeightHigh += quantizedWeight.y + select(0u, 1u, intraWeightLow < previousLow);`:``}
        } else {
          inter++;
          key = ${y?`sourceLabel`:`min(sourceLabel, targetLabel)`};
          destinationKey = ${y?`targetLabel`:`max(sourceLabel, targetLabel)`};
        }
      }
    }
    sortKeys[sortKeysOffset + slot] = key;
    targetKeys[targetKeysOffset + slot] = destinationKey;
  }
  ${N?`// Every intra-group slot of a row lands in the row's own group, so the row folds its intra
  // count and 64-bit fixed-point weight locally and issues one atomic set instead of one per slot
  // (integer addition is associative, so the sums are unchanged).
  if (intra > 0u) {
    let intraBase = sourceLabel * 3u;
    atomicAdd(&intraAcc[intraAccOffset + intraBase], intra);
    ${O?`let rowIntraWeight = vec2<u32>(intraWeightLow, intraWeightHigh);
    ${K(`intraAcc`,`intraAccOffset + intraBase + 1u`,`rowIntraWeight`)}`:``}
  }`:``}
  addCounter(0u, counted);
  addCounter(1u, dropped);
  addCounter(2u, intra);
  addCounter(3u, inter);`});k.push(s(e,{id:`${a}-slots`,operation:B,variant:`slots`,bindings:X,invocationCount:m,workgroupSize:256,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${m}u;
const SLOT_COUNT: u32 = ${v}u;
const GROUP_CAPACITY: u32 = ${h}u;
const SENTINEL_KEY: u32 = ${w}u;
const DEAD: u32 = 0xffffffffu;
const OVERFLOWED: u32 = 0xfffffffeu;
${G(!b)}
${fe.declarations}`,body:fe.body})),k.push(s(e,{id:`${a}-iota`,operation:B,variant:`iota`,bindings:[{name:`sortIndices`,view:ae,type:`u32`,access:`read_write`}],invocationCount:v,body:`sortIndices[sortIndicesOffset + index] = index;`}));let pe=u(e,{id:`${a}-sort`,operation:B,indices:ae,indicesAreIdentity:!0,packKeys:!1,keys:[{view:F,keyBits:l(h)},{view:I,keyBits:l(h)}]});k.push(...pe.nodes);let me=pe.sortedPrimaryKeys,he=pe.sortedIndices,Z=A(`sorted-targets`,v);k.push(...new i({id:`${a}-target-gather`,source:I,indices:he,output:Z}).getCommandNodes(e)),k.push(s(e,{id:`${a}-flags`,operation:B,variant:`flags`,bindings:[{name:`sortedKeys`,view:me,type:`u32`,access:`read`},{name:`sortedTargets`,view:Z,type:`u32`,access:`read`},{name:`flags`,view:L,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`const SENTINEL_KEY: u32 = ${w}u;`,body:`let key = sortedKeys[sortedKeysOffset + index];
  var isStart = key != SENTINEL_KEY;
  if (isStart && index > 0u) {
    isStart = sortedKeys[sortedKeysOffset + index - 1u] != key || sortedTargets[sortedTargetsOffset + index - 1u] != sortedTargets[sortedTargetsOffset + index];
  }
  flags[flagsOffset + index] = select(0u, 1u, isStart);`})),k.push(...new r({id:`${a}-segment-scan`,input:L,output:R,mode:`exclusive`}).getCommandNodes(e)),k.push(s(e,{id:`${a}-segment-total`,operation:B,variant:`segment-total`,bindings:[{name:`segmentScan`,view:R,type:`u32`,access:`read`},{name:`flags`,view:L,type:`u32`,access:`read`},{name:`stats`,view:P,type:`u32`,access:`read_write`},{name:`totalOut`,view:z,type:`u32`,access:`read_write`},{name:`overflowOut`,view:ce,type:`u32`,access:`read_write`}],invocationCount:1,body:`let total = segmentScan[segmentScanOffset + ${v-1}u] + flags[flagsOffset + ${v-1}u];
  totalOut[totalOutOffset] = total;
  stats[statsOffset + ${U.superedgeCount}u] = total;
  let labelOverflow = stats[statsOffset + ${U.overflowedNodeCount}u] > 0u ||
    stats[statsOffset + ${U.droppedEdgeCount}u] > 0u;
  overflowOut[overflowOutOffset] = select(0u, 1u, labelOverflow);`}));let ge=[{name:`sortedKeys`,view:me,type:`u32`,access:`read`},{name:`sortedTargets`,view:Z,type:`u32`,access:`read`},{name:`segmentScan`,view:R,type:`u32`,access:`read`},{name:`edgeIds`,view:p.output.edges.ids,type:`u32`,access:`read_write`},{name:`edgeTargets`,view:p.output.edgeTargetGroups,type:`u32`,access:`read_write`},{name:`edgeCounts`,view:p.output.edgeCounts,type:`atomic<u32>`,access:`read_write`}],_e=(e,t,n,r)=>`let first = index * ${H}u;
  let last = first + min(${H}u, SLOT_COUNT - first);
  var runSegment = 0xffffffffu;
  var runCount = 0u;
  ${e}
  for (var position = first; position < last; position++) {
    let key = sortedKeys[sortedKeysOffset + position];
    if (key == SENTINEL_KEY) {
      break;
    }
    var isStart = true;
    if (position > 0u) {
      isStart = sortedKeys[sortedKeysOffset + position - 1u] != key || sortedTargets[sortedTargetsOffset + position - 1u] != sortedTargets[sortedTargetsOffset + position];
    }
    let segment = segmentScan[segmentScanOffset + position] + select(0u, 1u, isStart) - 1u;
    if (segment != runSegment) {
      if (runCount > 0u) {
        ${n}
      }
      runSegment = segment;
      runCount = 0u;
      ${t}
    }
    if (segment >= EDGE_CAPACITY) {
      continue;
    }
    runCount++;
    ${r}
  }
  if (runCount > 0u) {
    ${n}
  }`,ve=`const SENTINEL_KEY: u32 = ${w}u;
const GROUP_CAPACITY: u32 = ${h}u;
const EDGE_CAPACITY: u32 = ${_}u;
const SLOT_COUNT: u32 = ${v}u;`;if(k.push(s(e,{id:`${a}-segments`,operation:B,variant:`segments`,bindings:ge,invocationCount:Math.ceil(v/H),declarations:ve,body:_e(``,``,`atomicAdd(&edgeCounts[edgeCountsOffset + runSegment], runCount);`,`if (isStart) {
      edgeIds[edgeIdsOffset + segment] = key;
      edgeTargets[edgeTargetsOffset + segment] = sortedTargets[sortedTargetsOffset + position];
    }`)})),q&&p.output.edgeWeights){let t=[{name:`sortedKeys`,view:me,type:`u32`,access:`read`},{name:`sortedTargets`,view:Z,type:`u32`,access:`read`},{name:`segmentScan`,view:R,type:`u32`,access:`read`},{name:`segmentWeightAcc`,view:q,type:`atomic<u32>`,access:`read_write`}];p.topology.weights&&t.push({name:`sortedIndices`,view:he,type:`u32`,access:`read`},{name:`weights`,view:p.topology.weights,type:`f32`,access:`read`});let n=x?[{name:`stats`,view:P,type:`u32`,access:`read`}]:[];k.push(s(e,{id:`${a}-segment-weights`,operation:B,variant:`segment-weights`,bindings:[...t,...n],invocationCount:Math.ceil(v/H),declarations:`${ve}
${G(!b)}`,body:`let weightUnitExponent = ${C(`plain`)};
  ${_e(`var runWeightLow = 0u;
  var runWeightHigh = 0u;`,`runWeightLow = 0u;
      runWeightHigh = 0u;`,`let runWeight = vec2<u32>(runWeightLow, runWeightHigh);
        ${K(`segmentWeightAcc`,`segmentWeightAccOffset + 2u * runSegment`,`runWeight`)}`,`let quantizedWeight = quantize(${p.topology.weights?`weights[weightsOffset + sortedIndices[sortedIndicesOffset + position]]`:`1.0`}, weightUnitExponent);
    let previousLow = runWeightLow;
    runWeightLow += quantizedWeight.x;
    runWeightHigh += quantizedWeight.y + select(0u, 1u, runWeightLow < previousLow);`)}`}),s(e,{id:`${a}-edge-weights`,operation:B,variant:`edge-weights`,bindings:[{name:`segmentWeightAcc`,view:q,type:`u32`,access:`read`},{name:`edgeWeights`,view:p.output.edgeWeights,type:`f32`,access:`read_write`},...n],invocationCount:_,declarations:G(!b),body:`edgeWeights[edgeWeightsOffset + index] = dequantize(
    segmentWeightAcc[segmentWeightAccOffset + 2u * index],
    segmentWeightAcc[segmentWeightAccOffset + 2u * index + 1u],
    ${C(`plain`)});`}))}let Q=[{name:`groupCounts`,view:p.output.groupNodeCount,type:`u32`,access:`read`},{name:`groupAcc`,view:M,type:`u32`,access:`read`}],$=[];if(p.output.groupCentroid&&(Q.push({name:`centroidOut`,view:p.output.groupCentroid,type:`f32`,access:`read_write`}),$.push(`let positionUnitExponent = ${S(W.positions,m,`plain`)};
  centroidOut[centroidOutOffset + 2u * index] = select(0.0, dequantize(groupAcc[groupAccOffset + base], groupAcc[groupAccOffset + base + 1u], positionUnitExponent) / f32(count), count > 0u);
  centroidOut[centroidOutOffset + 2u * index + 1u] = select(0.0, dequantize(groupAcc[groupAccOffset + base + 2u], groupAcc[groupAccOffset + base + 3u], positionUnitExponent) / f32(count), count > 0u);`)),p.output.groupBounds&&(Q.push({name:`boundsOut`,view:p.output.groupBounds,type:`f32`,access:`read_write`}),$.push(`boundsOut[boundsOutOffset + 4u * index] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 6u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 1u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 7u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 2u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 8u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 3u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 9u]), count > 0u);`)),p.output.groupValueSum&&(Q.push({name:`valueOut`,view:p.output.groupValueSum,type:`f32`,access:`read_write`}),$.push(`valueOut[valueOutOffset + index] = dequantize(groupAcc[groupAccOffset + base + 4u], groupAcc[groupAccOffset + base + 5u], ${S(W.values,m,`plain`)});`)),b&&(p.output.groupCentroid||p.output.groupValueSum)&&Q.push({name:`stats`,view:P,type:`u32`,access:`read`}),N){let t=[{name:`intraAcc`,view:N,type:`u32`,access:`read`}],n=[];p.output.groupIntraEdgeCount&&(t.push({name:`intraCountOut`,view:p.output.groupIntraEdgeCount,type:`u32`,access:`read_write`}),n.push(`intraCountOut[intraCountOutOffset + index] = intraAcc[intraAccOffset + 3u * index];`)),p.output.groupIntraWeight&&(t.push({name:`intraWeightOut`,view:p.output.groupIntraWeight,type:`f32`,access:`read_write`}),x&&t.push({name:`stats`,view:P,type:`u32`,access:`read`}),n.push(`intraWeightOut[intraWeightOutOffset + index] = dequantize(intraAcc[intraAccOffset + 3u * index + 1u], intraAcc[intraAccOffset + 3u * index + 2u], ${C(`plain`)});`)),k.push(s(e,{id:`${a}-intra-groups`,operation:B,variant:`intra-groups`,bindings:t,invocationCount:h,declarations:G(!b),body:n.join(`
  `)}))}return $.length>0&&k.push(s(e,{id:`${a}-groups`,operation:B,variant:`groups`,bindings:Q,invocationCount:h,declarations:G(!b),body:`let base = index * ${V}u;
  let count = groupCounts[groupCountsOffset + index];
  ${$.join(`
  `)}`})),k.push(o(e,{id:`${a}-publish`,operation:B,requiredCount:z,output:p.output.edges,candidateOverflowSources:p.output.edges.status.candidateOverflow?[ce]:[]})),p.output.summary&&k.push(s(e,{id:`${a}-summary`,operation:B,variant:`summary`,bindings:[{name:`stats`,view:P,type:`u32`,access:`read`},{name:`summary`,view:p.output.summary,type:`u32`,access:`read_write`}],invocationCount:8,body:`summary[summaryOffset + index] = stats[statsOffset + index];`})),k}},q=N(P,`cellSize`),J=128,Y=256,le=E({parameters:P,caption:`Group centroids and superedges after labelling nodes by spatial cell.`,createInputs:({random:e,world:t})=>({base:O(e,t,12),world:t,offset:[0,0]}),build(e,t,n){let{base:r,world:i}=t,a=j(r,n.closures),{topology:o}=re(e,a),s=e.upload(`node-positions`,`float32x2`,a.nodes),c=e.allocate(`labels`,`uint32`,a.nodeCount),l=e.allocate(`group-node-count`,`uint32`,J),u=e.allocate(`group-centroid`,`float32x2`,J),d=e.allocate(`edge-ids`,`uint32`,Y),f=e.allocate(`edge-count`,`uint32`,1),p=e.allocate(`edge-overflow`,`uint32`,1),m=e.allocate(`edge-targets`,`uint32`,Y),h=e.allocate(`edge-counts`,`uint32`,Y);return e.graph.add(new ce({topology:{...o,nodePositions:s.view},labels:c.view,groupCapacity:J,output:{groupNodeCount:l.view,groupCentroid:u.view,edges:{ids:d.view,status:{count:f.view,overflow:p.view}},edgeTargetGroups:m.view,edgeCounts:h.view}})),{outputs:{...M(e,a),labels:c,edgeCount:f,groupNodeCount:l,groupCentroid:u,superedges:D(e,{id:`superedges`,from:d,to:m,fallbackNode:127})},update:e=>{let n=Math.ceil(i.width/e.cellSize)+2;c.write(Uint32Array.from({length:a.nodeCount},(r,i)=>{let o=Math.floor((a.nodes[i*2]-t.offset[0])/e.cellSize),s=Math.floor((a.nodes[i*2+1]-t.offset[1])/e.cellSize);return Math.min(126,o+1+(s+1)*n)%127}))}}},rebuildOn:[`closures`],interaction:{kind:`window`,hint:`slide the grid`,rest:({world:e,toUnit:t})=>{let n=q.default;return t([n*Math.floor(e.width/2/n),n*Math.floor(.5/n)])},resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=r.cellSize,s=[e.x%o,e.y%o],c=s[0]!==t.offset[0]||s[1]!==t.offset[1];t.offset=s;let l=[];for(let e=s[0];e<=i.width;e+=o)l.push({shape:`line`,from:[e,0],to:[e,1],dashed:!0});for(let e=s[1];e<=1;e+=o)l.push({shape:`line`,from:[0,e],to:[i.width,e],dashed:!0});let u=a(n.groupNodeCount),d=a(n.edgeCount),f=0;if(u)for(let e of u)e>0&&f++;return{rerun:c,glyphs:l,readout:`cellSize ${o.toFixed(2)} · ${f} groups · ${d?d[0]:0} superedges`}}},draw:({outputs:e})=>[A(e),k(e,e.labels,{radius:1.8}),{layer:`segments`,nodes:e.groupCentroid,edges:e.superedges,radius:2.5,color:[.5,.8,1,.9]},{layer:`dots`,positions:e.groupCentroid,values:e.groupNodeCount,ramp:`sequential`,radius:9}]});export{le as default};
import{i as e,r as t,s as n}from"./kernel-C3tfQOn4.js";import{i as r,o as i,t as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,O as s,S as c,T as l,f as u,o as d,p as f,v as p,x as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{t as g}from"./gpu-neighbor-search-SzuWK-my.js";import{t as _}from"./neighbor-search-derived-parameters-DZFCGzl5.js";import{t as v}from"./gpu-compaction-B6IzcF0P.js";import{t as y}from"./statistics-kernel-node-lacKNERm.js";import{c as b}from"./wgsl-common-DfyXiYB0.js";import{t as x}from"./slot-grouping-CsHvMvbq.js";import{t as S}from"./spatial-weights-algebra-wgsl-S5xfkEuX.js";import{a as C,c as w,i as T,l as E,t as D}from"./spatial-weights-CqWysstN.js";import{t as O}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{t as k}from"./define-run-B-brkCe4.js";import{n as A,r as j}from"./inputs-CJKKIRKK.js";import{n as M,t as N}from"./pick-Bo3t3hjv.js";import{t as P}from"./kernel-DfdJX5rr.js";import{Vn as F}from"./entry-preview-D70TdSbK.js";import{C as I,h as L,n as R,o as z,p as B,t as V,y as H}from"./weights-BRH93xop.js";var U=`${b}
const NO_ATTACHMENT: u32 = 0xffffffffu;
fn isCloserDifference(difference: vec2<f32>, bestDifference: vec2<f32>, isFirst: bool) -> bool {
 let scale = max(max(abs(difference.x), abs(difference.y)), max(abs(bestDifference.x), abs(bestDifference.y)));
 var closer = isFirst;
 if (scale > 0.0) { let exponent = clamp(i32((bitcast<u32>(scale) >> 23u) & 255u) - 127, -120, 120); let normalizer = bitcast<f32>(u32(127 - exponent) << 23u); let normalized = difference * normalizer; let bestNormalized = bestDifference * normalizer; closer = closer || dot(normalized, normalized) < dot(bestNormalized, bestNormalized); }
 return closer;
}`,W=65536,G=16777215,K=256;function q(e,t=G,n=W){return Math.ceil(e/n)*Math.ceil(e/t)}function J(e,i){let{id:a,positions:o,unresolved:s,attachment:c,nearestDeclarations:l}=i,d=o.length,m=i.queryBatchSize??16777215,h=i.representativeBatchSize??65536;u(a,`queryBatchSize`,m,1,G),u(a,`representativeBatchSize`,h,1,W);let g=Math.min(d,m),_=Math.min(d,h),b=q(d,g,_);f(a,K+3*b);for(let[t,r]of[[`positions`,o],[`unresolved`,s],[`attachment`,c],[`uncertainCount`,i.uncertainCount]])r&&p(e.device,a,t,n(r).size,`reduce rows`);for(let[t,n]of[[`fallback rows`,d*4],[`fallback cursor`,20],[`fallback dispatch`,12]])p(e.device,a,t,n,`reduce rows`);let x=t(e,`${a}-fallback-row-ids`,`uint32`,d),S=t(e,`${a}-fallback-queries`,`uint32`,d),C=i.uncertainCount??t(e,`${a}-fallback-count`,`uint32`,1),w=t(e,`${a}-fallback-cursor`,`uint32`,5),T=(e,t,n=`read`,r=`u32`)=>({name:e,view:t,access:n,type:r}),E=[y(e,{id:`${a}-fallback-row-ids`,operation:`GPUIslandAttachment`,invocationCount:d,bindings:[T(`rowIds`,x,`read_write`)],body:`rowIds[rowIdsOffset + index] = index;`}),...new v({id:`${a}-fallback-compact`,input:x,flags:s,output:S,count:C}).getCommandNodes(e),y(e,{id:`${a}-fallback-initialize`,operation:`GPUIslandAttachment`,invocationCount:1,bindings:[T(`cursor`,w,`read_write`),T(`count`,C)],body:`cursor[cursorOffset] = 0u;
cursor[cursorOffset + 1u] = 0u;
cursor[cursorOffset + 2u] = min(count[countOffset], ${g}u);
cursor[cursorOffset + 3u] = ${_}u;
cursor[cursorOffset + 4u] = ${g}u;`})];for(let t=0;t<b;t++){let n=r(e,{id:`${a}-fallback-dispatch-${t}`,operation:`GPUIslandAttachment`,count:w,countRow:2,maximumItemCount:g,invocationsPerItem:256,workgroupSize:256});E.push(n.updateNode,y(e,{id:`${a}-resolve-unproven-${t}`,operation:`GPUIslandAttachment`,variant:`resolve-unproven`,invocationCount:g*256,workgroupSize:256,guardIndex:!1,condition:n.condition,bindings:[T(`positions`,o,`read`,`f32`),T(`queries`,S),T(`cursor`,w),T(`attachment`,c,`read_write`)],declarations:`${l}
const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> nearestRows: array<u32, 256>;
var<workgroup> nearestDifferences: array<vec2<f32>, 256>;`,body:`let querySlot = workgroupIndex;
let lane = localInvocationIndex;
let inRange = querySlot < cursor[cursorOffset + 2u];
let rowCount = (arrayLength(&positions) - positionsOffset) / 2u;
let representativeBase = cursor[cursorOffset];
let representativeCount = min(cursor[cursorOffset + 3u], rowCount - representativeBase);
var island = NO_ATTACHMENT;
var position = vec2<f32>(0.0);
if (inRange) {
 island = queries[queriesOffset + cursor[cursorOffset + 1u] + querySlot];
 position = vec2<f32>(positions[positionsOffset + 2u * island], positions[positionsOffset + 2u * island + 1u]);
}
var nearest = NO_ATTACHMENT;
var bestDifference = vec2<f32>(0.0);
if (inRange) {
 for (var slot = lane; slot < representativeCount; slot += WORKGROUP_SIZE) {
  let other = representativeBase + slot;
  let candidate = vec2<f32>(positions[positionsOffset + 2u * other], positions[positionsOffset + 2u * other + 1u]);
  if (other != island && isFiniteBits(candidate.x) && isFiniteBits(candidate.y)) {
   let difference = position * 0.5 - candidate * 0.5;
   if (isCloserDifference(difference, bestDifference, nearest == NO_ATTACHMENT)) {
    nearest = other; bestDifference = difference;
   }
  }
 }
}
nearestRows[lane] = nearest;
nearestDifferences[lane] = bestDifference;
workgroupBarrier();
for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
 if (lane < stride) {
  let other = nearestRows[lane + stride];
  let difference = nearestDifferences[lane + stride];
  nearest = nearestRows[lane];
  bestDifference = nearestDifferences[lane];
  if (other != NO_ATTACHMENT &&
      (isCloserDifference(difference, bestDifference, nearest == NO_ATTACHMENT) ||
       (other < nearest && !isCloserDifference(bestDifference, difference, false)))) {
   nearestRows[lane] = other;
   nearestDifferences[lane] = difference;
  }
 }
 workgroupBarrier();
}
if (inRange && lane == 0u) {
 nearest = nearestRows[0];
 bestDifference = nearestDifferences[0];
 let previous = attachment[attachmentOffset + island];
 if (previous != NO_ATTACHMENT) {
  let candidate = vec2<f32>(positions[positionsOffset + 2u * previous], positions[positionsOffset + 2u * previous + 1u]);
  let difference = position * 0.5 - candidate * 0.5;
  if (isCloserDifference(difference, bestDifference, nearest == NO_ATTACHMENT) ||
      (previous < nearest && !isCloserDifference(bestDifference, difference, false))) { nearest = previous; }
 }
 attachment[attachmentOffset + island] = nearest;
}`}),y(e,{id:`${a}-fallback-advance-${t}`,operation:`GPUIslandAttachment`,invocationCount:1,bindings:[T(`cursor`,w,`read_write`),T(`count`,C),T(`positions`,o,`read`,`f32`)],body:`let rowCount = (arrayLength(&positions) - positionsOffset) / 2u;
let nextRepresentative = cursor[cursorOffset] + min(cursor[cursorOffset + 3u], rowCount - cursor[cursorOffset]);
if (nextRepresentative == rowCount) {
 cursor[cursorOffset] = 0u;
 let queryBase = min(count[countOffset], cursor[cursorOffset + 1u] + cursor[cursorOffset + 4u]);
 cursor[cursorOffset + 1u] = queryBase;
 cursor[cursorOffset + 2u] = min(cursor[cursorOffset + 4u], count[countOffset] - queryBase);
} else { cursor[cursorOffset] = nextRepresentative; }`}))}return E}var Y=16,X=4294967294,Z=4294967295,Q=65535,$=4096,ee=1.00001,te=class{props;id;outputWeights;estimatedCommandNodeCount;constructor(t){this.props=t,this.id=t.id??`island-attachment`;let r=t.nearestSearch??`auto`;l(this.id,[`auto`,`scan`,`grid`].includes(r),`nearestSearch must be 'auto', 'scan' or 'grid'`);let i=w(this.id,t.weights);u(this.id,`rows`,i,1,X),r===`scan`&&c(this.id,`rows for nearestSearch scan`,i,Q);let a=r===`grid`||r===`auto`&&i>4096;if(a&&c(this.id,`candidate slots (rows * 16)`,i*Y,Z),c(this.id,`maximum output edge count`,t.weights.neighbors.length+i*(t.reciprocal===!1?1:2),Z),this.estimatedCommandNodeCount=$+(a?3*q(i):0),f(this.id,this.estimatedCommandNodeCount),E(this.id,t.output)!==i||t.positions.length!==i||t.output.attachment.length!==i||t.output.added.length!==t.output.neighbors.length)throw Error(`${this.id}: shape mismatch`);if(t.output.distances)throw Error(`${this.id}: distances output is not supported`);o(t.positions,[`float32x2`],`${this.id} positions`),o(t.output.attachment,[`uint32`],`${this.id} output.attachment`),o(t.output.added,[`uint32`],`${this.id} output.added`),h(this.id,[...C(t.output),...T(t.output),t.output.attachment,t.output.added,t.output.status.uncertainCount],[...C(t.weights),t.positions]);let d=[...C(t.output),...T(t.output),t.output.attachment,t.output.added,t.output.status.uncertainCount].filter(e=>!!e);t.output.status.uncertainCount&&(o(t.output.status.uncertainCount,[`uint32`],`${this.id} output.status.uncertainCount`),u(this.id,`output.status.uncertainCount length`,t.output.status.uncertainCount.length,1,1));for(let t=0;t<d.length;t++)for(let n=t+1;n<d.length;n++)l(this.id,!e(d[t],d[n]),`output writable byte ranges must not overlap`);let p=[[[t.output.offsets,!0],[t.output.status.overflow,!0],[t.output.status.requiredCount,!0]],...[t.output.neighbors,t.output.weights,t.output.added].map(e=>[[t.output.offsets,!1],[t.output.attachment,!1],[e,!0]])];for(let e of p)for(let t=0;t<e.length;t++)for(let r=t+1;r<e.length;r++){let[i,a]=e[t],[o,c]=e[r];if(!i||!o||!(a||c)||(i.buffer.defaultBuffer?s(i.buffer.defaultBuffer):i.buffer)!==(o.buffer.defaultBuffer?s(o.buffer.defaultBuffer):o.buffer))continue;l(this.id,a===c,`output read-only and writable bindings must use separate physical buffers`),l(this.id,i.buffer===o.buffer,`output views share a physical buffer: import it once and use offset views`);let u=n(i),d=n(o);l(this.id,u.offset+u.size<=d.offset||d.offset+d.size<=u.offset,`output aligned binding ranges used together must not overlap`)}this.outputWeights=D(t.output,{information:{rowCount:i,transform:`raw`,symmetric:t.reciprocal!==!1&&t.weights.information?.symmetric===!0||`unknown`,hasSelfWeights:!1,provenance:{source:`GPUIslandAttachment`,criterion:`nearest-original-island`}},status:t.output.status})}getCommandNodes(e){let{props:r,id:o}=this;O(e,o,r);let s=r.positions.length,l=r.output.neighbors.length,u=[[`weights.offsets`,r.weights.offsets],[`weights.neighbors`,r.weights.neighbors],[`weights.weights`,r.weights.weights],[`weights.distances`,r.weights.distances],[`positions`,r.positions],[`output.offsets`,r.output.offsets],[`output.neighbors`,r.output.neighbors],[`output.weights`,r.output.weights],[`output.distances`,r.output.distances],[`output.attachment`,r.output.attachment],[`output.added`,r.output.added],[`output.status.overflow`,r.output.status.overflow],[`output.status.requiredCount`,r.output.status.requiredCount],[`output.status.candidateOverflow`,r.output.status.candidateOverflow],[`output.status.uncertainCount`,r.output.status.uncertainCount]];for(let[t,r]of u)r&&p(e.device,o,t,n(r).size,`reduce rows or output capacity`);let f=r.nearestSearch??`auto`,h=f===`grid`||f===`auto`&&s>4096,v=[[`CSR counts and starts`,s*4],[`CSR scan total`,4]];if(r.reciprocal!==!1&&(v.push([`inverse offsets`,(s+1)*4],[`inverse sort rows`,s*4]),s>256&&v.push([`inverse radix histograms`,Math.ceil(s/256)*16*4])),h){let t=Math.max(1,Math.ceil(Math.sqrt(s/2)));c(o,`gridSize cells`,t*t,d(e.device)),v.push([`candidate lists and outputs`,s*Y*4],[`neighbor records`,s*3*4],[`neighbor grid positions`,s*2*4],[`neighbor grid offsets`,(t*t+1)*4],[`neighbor bounds partials`,Math.min(256,Math.ceil(s/4096))*16],[`fallback cursor`,20],[`fallback dispatch`,12],[`neighbor parameters`,48])}for(let[t,n]of v)p(e.device,o,t,n,`reduce rows`);let y=(e,t,n=`read`,r=`u32`)=>({name:e,view:t,access:n,type:r}),b=(t,n,r,a,s=``)=>i(e,{id:`${o}-${t}`,operation:`GPUIslandAttachment`,variant:t,invocationCount:n,bindings:r,declarations:s,body:a});m(o,e,[...C(r.weights),r.positions,...C(r.output),...T(r.output),r.output.attachment,r.output.added,r.output.status.uncertainCount]);let w=r.reciprocal!==!1,E=U,D=`for (var other = 0u; other < ${s}u; other++) { let candidate = vec2<f32>(positions[positionsOffset + 2u * other], positions[positionsOffset + 2u * other + 1u]);
 if (other != index && isFiniteBits(candidate.x) && isFiniteBits(candidate.y)) {
 let difference = position * 0.5 - candidate * 0.5;
 if (isCloserDifference(difference, bestDifference, nearest == NO_ATTACHMENT)) { bestDifference = difference; nearest = other; }
 }
}`,k=[],A=r.output.status.uncertainCount,j=()=>A?[a(e,{id:`${o}-uncertain-count-clear`,operation:`GPUIslandAttachment`,view:A,type:`u32`,value:`0u`,componentCount:1})]:[];if(!h)k.push(...j(),b(`nearest`,s,[y(`positions`,r.positions,`read`,`f32`),y(`offsets`,r.weights.offsets),y(`attachment`,r.output.attachment,`read_write`)],`var nearest = NO_ATTACHMENT; var bestDifference = vec2<f32>(0.0);
let position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
if (offsets[offsetsOffset + index] == offsets[offsetsOffset + index + 1u] && isFiniteBits(position.x) && isFiniteBits(position.y)) {
${D}
}
attachment[attachmentOffset + index] = nearest;`,E));else{let n=t(e,`${o}-island-flags`,`uint32`,s),i=t(e,`${o}-unresolved`,`uint32`,s),a=_(e,{id:`${o}-index-parameters`,operation:`GPUIslandAttachment`,positions:r.positions,radius:1/0}),c=t(e,`${o}-index-offsets`,`uint32`,s+1),l=t(e,`${o}-index-neighbors`,`uint32`,s*Y),u=t(e,`${o}-index-distances`,`float32`,s*Y);k.push(b(`island-flags`,s,[y(`offsets`,r.weights.offsets),y(`islandFlags`,n,`read_write`)],`islandFlags[islandFlagsOffset + index] = select(0u, 1u, offsets[offsetsOffset + index] == offsets[offsetsOffset + index + 1u]);`),...a.nodes,...new g({id:`${o}-index`,mode:`knn`,k:Y,positions:r.positions,queryPositions:r.positions,queryMask:n,parameters:a.parameters,gridSize:a.gridSize,output:{offsets:c,neighbors:l,weights:t(e,`${o}-index-weights`,`float32`,s*Y),distances:u,status:{overflow:t(e,`${o}-index-overflow`,`uint32`,1)}}}).getCommandNodes(e),b(`nearest`,s,[y(`positions`,r.positions,`read`,`f32`),y(`offsets`,r.weights.offsets),y(`candidateOffsets`,c),y(`candidateNeighbors`,l),y(`candidateDistances`,u,`read`,`f32`),y(`unresolved`,i,`read_write`),y(`attachment`,r.output.attachment,`read_write`)],`unresolved[unresolvedOffset + index] = 0u;
var nearest = NO_ATTACHMENT; var bestDifference = vec2<f32>(0.0);
let position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
if (offsets[offsetsOffset + index] == offsets[offsetsOffset + index + 1u] && isFiniteBits(position.x) && isFiniteBits(position.y)) {
 let begin = candidateOffsets[candidateOffsetsOffset + index];
 let found = candidateOffsets[candidateOffsetsOffset + index + 1u] - begin;
 var farthest = 0.0;
 for (var slot = 0u; slot < found; slot++) {
  let other = candidateNeighbors[candidateNeighborsOffset + begin + slot];
  farthest = max(farthest, candidateDistances[candidateDistancesOffset + begin + slot]);
  let candidate = vec2<f32>(positions[positionsOffset + 2u * other], positions[positionsOffset + 2u * other + 1u]);
  if (other != index && isFiniteBits(candidate.x) && isFiniteBits(candidate.y)) {
   let difference = position * 0.5 - candidate * 0.5;
   if (isCloserDifference(difference, bestDifference, nearest == NO_ATTACHMENT)) { bestDifference = difference; nearest = other; }
  }
 }
 // The list is complete when it holds fewer than the candidate count (an empty list means the grid
 // was unusable, for example overflowing bounds). Otherwise the answer is proven when the
 // farthest entry is clearly farther than the chosen one; otherwise use the cooperative pass.
 var proven = found > 0u && found < ${Y}u;
 if (found == ${Y}u && nearest != NO_ATTACHMENT) {
  let chosenDistance = 2.0 * length(bestDifference);
  proven = isFiniteBits(chosenDistance) && isFiniteBits(farthest) && farthest > chosenDistance * ${ee};
 }
 if (!proven) {
  unresolved[unresolvedOffset + index] = 1u;
  nearest = NO_ATTACHMENT;
 }
}
attachment[attachmentOffset + index] = nearest;`,E),...J(e,{id:o,positions:r.positions,unresolved:i,attachment:r.output.attachment,uncertainCount:A,nearestDeclarations:E}))}let M;if(w){let n=t(e,`${o}-inverse-offsets`,`uint32`,s+1),i=t(e,`${o}-inverse-keys`,`uint32`,s),a=t(e,`${o}-inverse-indices`,`uint32`,s);k.push(b(`inverse-keys`,s,[y(`attachment`,r.output.attachment),y(`slotKeys`,i,`read_write`),y(`slotIndices`,a,`read_write`)],`let attachedRow = attachment[attachmentOffset + index];
slotKeys[slotKeysOffset + index] = select(${s}u, attachedRow, attachedRow < ${s}u);
slotIndices[slotIndicesOffset + index] = index;`));let c=x(e,{id:`${o}-inverse`,operation:`GPUIslandAttachment`,slotKeys:i,slotIndices:a,columns:s,columnOffsets:n});k.push(...c.nodes),M={offsets:n,islands:c.sortedSlots}}let N=e=>`var originalNext = offsets[offsetsOffset + index];
let originalEnd = offsets[offsetsOffset + index + 1u];
var attached = attachment[attachmentOffset + index];
${w?`var inverseNext = inverseOffsets[inverseOffsetsOffset + index];
let inverseEnd = inverseOffsets[inverseOffsetsOffset + index + 1u];`:``}
loop {
 var originalId = NO_SLOT; if (originalNext < originalEnd) { originalId = neighbors[neighborsOffset + originalNext]; }
 var inverseId = NO_SLOT;${w?` if (inverseNext < inverseEnd) { inverseId = inverseIslands[inverseIslandsOffset + inverseNext]; }`:``}
 let other = min(min(originalId, attached), inverseId);
 if (other == NO_SLOT) { break; }
 var originalSlot = NO_SLOT;
 if (originalId == other) { originalSlot = originalNext; while (originalNext < originalEnd && neighbors[neighborsOffset + originalNext] == other) { originalNext++; } }
 if (attached == other) { attached = NO_SLOT; }
${w?` if (inverseId == other) { inverseNext++; }
`:``} if (other < ROWS && (originalSlot != NO_SLOT || other != index)) { ${e} }
}`;return k.push(...S(e,{id:`${o}-edges`,operation:`GPUIslandAttachment`,rows:s,targetOffsets:r.output.offsets,capacity:l,overflow:r.output.status.overflow,initializesOverflow:!0,totalNeighbors:r.output.status.requiredCount,bindings:[y(`offsets`,r.weights.offsets),y(`neighbors`,r.weights.neighbors),y(`attachment`,r.output.attachment),...M?[y(`inverseOffsets`,M.offsets),y(`inverseIslands`,M.islands)]:[]],visit:N,attributes:[{name:`neighbors`,output:r.output.neighbors,type:`u32`,value:`other`},{name:`weights`,output:r.output.weights,type:`f32`,value:`getOriginalWeight(originalSlot)`,bindings:[y(`originalWeights`,r.weights.weights,`read`,`f32`)],declarations:`fn getOriginalWeight(slot: u32) -> f32 {
 if (slot == NO_SLOT) { return 1.0; }
 return originalWeights[originalWeightsOffset + slot];
}`},{name:`added`,output:r.output.added,type:`u32`,value:`select(0u, 1u, originalSlot == NO_SLOT)`}]})),k}},ne=k({parameters:F,caption:`Isolated points are attached to their nearest neighbor.`,createInputs:({random:e,world:t})=>{let n=A(e,t,200,4,.18),r=H(n.positions,t);return{...n,positions:r,index:M(r),focus:{row:-1}}},build(e,t,n){let r=e.upload(`positions`,`float32x2`,t.positions),i=I(e,`band`,j(t.positions,n.radius)),a=z(e,`attached`,t.count,i.capacity+t.count*2),o=e.allocate(`attachment`,`uint32`,t.count),s=e.allocate(`added`,`uint32`,a.capacity);e.graph.add(new te({weights:i.input,positions:r.view,reciprocal:n.reciprocal===1,output:{...a.port,attachment:o.view,added:s.view}}));let c=V(e,`attached`,a),l=V(e,`band`,i),u=e.allocate(`gained-links`,`float32`,t.count);return P(e,{id:`gained-links`,invocationCount:t.count,bindings:[{name:`after`,buffer:c,access:`read`},{name:`before`,buffer:l,access:`read`},{name:`gained`,buffer:u,access:`read_write`}],body:`gained[gainedOffset + index] = after[afterOffset + index] - before[beforeOffset + index];`}),{outputs:{positions:r,attachment:o,cardinality:c,gained:u,bandLinks:R(e,`band`,i,r),addedLinks:R(e,`added`,a,r,{onlyWhere:s})}}},interaction:{kind:`probe`,hint:`hover a point`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=i(n.attachment);if(!a)return{};let o=-1,s=.3;for(let n=0;n<a.length;n++){if(a[n]===4294967295)continue;let r=t.index.distanceTo(n,e.x,e.y);r<s&&(o=n,s=r)}if(o<0&&(o=N(t.index,e.x,e.y,t.focus.row)),t.focus.row=o,o<0)return{};let c=[t.positions[o*2],t.positions[o*2+1]],l=[{shape:`ring`,at:c,radius:r.radius,dashed:!0}],u=a[o];if(u!==4294967295){let e=[t.positions[u*2],t.positions[u*2+1]];l.push({shape:`line`,from:c,to:e},{shape:`dot`,at:e});let n=Math.hypot(e[0]-c[0],e[1]-c[1]);return{focus:o,glyphs:l,readout:`island ${o} → point ${u} at ${n.toFixed(3)} (band ${r.radius})`}}let d=L(i(n.bandLinks),c);return l.push(...B(c,d,{dashed:!0})),{focus:o,glyphs:l,readout:`point ${o} · ${d.length} band neighbors, no island`}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.bandLinks},{layer:`lines`,segments:e.addedLinks,color:[1,.45,.2,.95],radius:1.8},{layer:`dots`,positions:e.positions,values:e.cardinality,radius:3.4,range:[0,8]},{layer:`dots`,positions:e.positions,values:e.gained,ramp:`sequential`,range:[0,4],fade:!0,sizeByValue:!0,radius:10}]});export{ne as default};
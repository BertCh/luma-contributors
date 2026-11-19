import{t as e}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as t,n}from"./weights-construction-snapshot-CikLt8kl.js";import{i as r,n as i,r as a}from"./spatial-weights-loss-u59PcHkg.js";import{l as o}from"./spatial-context-DwZEd4EZ.js";import{r as s}from"./kernel-C3tfQOn4.js";import{t as c}from"./gpu-sort-xRm8l5yn.js";import{c as l,o as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as d}from"./sorted-segment-sums-Dvmm6OdR.js";import{E as f,f as p,x as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{a as g,c as _,i as v,l as y,t as b}from"./spatial-weights-CqWysstN.js";import{n as x,r as S,t as C}from"./spatial-weights-algebra-wgsl-S5xfkEuX.js";function w(e,t,n){let{weights:r,output:i,cellCount:a,periodCount:o,layout:s}=n,c=[...n.lags].sort((e,t)=>t.lag-e.lag),u=c.some(e=>e.includeSelf),d=(e,t)=>s===`period-major`?`(${t}) * CELLS + ${e}`:`${e} * PERIODS + (${t})`,f=e=>c.map((t,n)=>{let r=`sourceOf${n}`,i=l(t.weight??1),a=t.includeSelf?`select(spatialWeight, 1.0, neighbor == cell)`:`spatialWeight`;return`if (${r} >= 0 && ${r} < i32(PERIODS)) {
        let id = ${d(`neighbor`,`u32(${r})`)};
        let weightValue = ${i} * ${a};
        ${e}
      }`}).join(`
      `),p=e=>c.map((t,n)=>{if(!t.includeSelf)return``;let r=`sourceOf${n}`;return`if (${r} >= 0 && ${r} < i32(PERIODS)) {
        let id = ${d(`cell`,`u32(${r})`)};
        let weightValue = ${l(t.weight??1)};
        ${e}
      }`}).join(`
      `),m=c.map((e,t)=>`let sourceOf${t}: i32 = i32(period) - (${e.lag});`).join(`
  `),h=e=>{let t=s===`period-major`?`let period = index / CELLS;
  let cell = index % CELLS;`:`let cell = index / PERIODS;
  let period = index % PERIODS;`;if(s===`cell-major`)return`${t}
  ${m}
  let spatialBegin = min(baseOffsets[baseOffsetsOffset + cell], BASE_CAPACITY);
  let spatialEnd = min(baseOffsets[baseOffsetsOffset + cell + 1u], BASE_CAPACITY);
  var selfPending = ${u?`true`:`false`};
  for (var slot = spatialBegin; slot < spatialEnd; slot++) {
    let neighbor = baseNeighbors[baseNeighborsOffset + slot];
    if (neighbor >= CELLS) {
      continue;
    }
    if (selfPending && neighbor >= cell) {
      selfPending = false;
      if (neighbor > cell) {
        ${p(e)}
      }
    }
    let spatialWeight = baseWeights[baseWeightsOffset + slot];
    ${f(e)}
  }
  if (selfPending) {
    ${p(e)}
  }`;let n=c.map((t,n)=>{let r=`sourceOf${n}`,i=l(t.weight??1),a=(t,n)=>`{
          let id = ${d(t,`u32(${r})`)};
          let weightValue = ${i} * ${n};
          ${e}
        }`;return`if (${r} >= 0 && ${r} < i32(PERIODS)) {
    var selfPending = ${t.includeSelf?`true`:`false`};
    for (var slot = spatialBegin; slot < spatialEnd; slot++) {
      let neighbor = baseNeighbors[baseNeighborsOffset + slot];
      if (neighbor >= CELLS) {
        continue;
      }
      if (selfPending && neighbor >= cell) {
        selfPending = false;
        ${a(`cell`,`1.0`)}
        if (neighbor == cell) {
          continue;
        }
      }
      ${a(`neighbor`,`baseWeights[baseWeightsOffset + slot]`)}
    }
    if (selfPending) {
      ${a(`cell`,`1.0`)}
    }
  }`}).join(`
  `);return`${t}
  ${m}
  let spatialBegin = min(baseOffsets[baseOffsetsOffset + cell], BASE_CAPACITY);
  let spatialEnd = min(baseOffsets[baseOffsetsOffset + cell + 1u], BASE_CAPACITY);
  ${n}`};return C(e,{...t,bindings:[{name:`baseOffsets`,view:r.offsets,type:`u32`,access:`read`},{name:`baseNeighbors`,view:r.neighbors,type:`u32`,access:`read`},{name:`baseWeights`,view:r.weights,type:`f32`,access:`read`}],declarations:`const CELLS: u32 = ${a}u;
const PERIODS: u32 = ${o}u;
const BASE_CAPACITY: u32 = ${r.neighbors.length}u;`,visit:h,attributes:[{name:`neighbors`,output:i.neighbors,type:`u32`,value:`id`},{name:`weights`,output:i.weights,type:`f32`,value:`weightValue`}]})}function T(e){return e.chunk?`(index - WEIGHTS_ROW_START)`:`index`}function E(e,t,n){let{operation:r,left:i,right:a,output:o}=n,s=[{name:`neighbors`,output:o.neighbors,type:`u32`,value:`id`},{name:`weights`,output:o.weights,type:`f32`,value:`combineWeight(leftSlot, rightSlot)`,bindings:[{name:`leftWeights`,view:i.weights,type:`f32`,access:`read`},{name:`rightWeights`,view:a.weights,type:`f32`,access:`read`}],declarations:`fn combineWeight(leftSlot: u32, rightSlot: u32) -> f32 {
  ${n.isBinaryRule?`return 1.0;`:``}
  if (rightSlot == NO_SLOT) {
    return leftWeights[leftWeightsOffset + leftSlot];
  }
  if (leftSlot == NO_SLOT) {
    return rightWeights[rightWeightsOffset + rightSlot];
  }
  let leftValue = leftWeights[leftWeightsOffset + leftSlot];
  let rightValue = rightWeights[rightWeightsOffset + rightSlot];
  return ${n.combineExpression};
}`}];if(o.distances){let e=a.distances&&r!==`intersection`&&r!==`difference`;s.push({name:`distances`,output:o.distances,type:`f32`,value:`pickDistance(leftSlot, rightSlot)`,bindings:[{name:`leftDistances`,view:i.distances,type:`f32`,access:`read`},...e?[{name:`rightDistances`,view:a.distances,type:`f32`,access:`read`}]:[]],declarations:`fn pickDistance(leftSlot: u32, rightSlot: u32) -> f32 {
  if (leftSlot != NO_SLOT) {
    return leftDistances[leftDistancesOffset + leftSlot];
  }
  ${e?`return rightDistances[rightDistancesOffset + rightSlot];`:`return 0.0;`}
}`})}let c=T(t);return C(e,{...t,bindings:[{name:`leftOffsets`,view:i.offsets,type:`u32`,access:`read`},{name:`leftNeighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`rightOffsets`,view:a.offsets,type:`u32`,access:`read`},{name:`rightNeighbors`,view:a.neighbors,type:`u32`,access:`read`}],visit:e=>S(r,e,c),attributes:s})}function D(e,t,n){let{weights:r,output:i,selfWeight:a}=n,o=typeof a==`number`?l(a):`selfWeights[selfWeightsOffset + index]`,s=[{name:`neighbors`,output:i.neighbors,type:`u32`,value:`id`},{name:`weights`,output:i.weights,type:`f32`,value:`select(sourceWeights[sourceWeightsOffset + sourceSlot], ${o}, id == index)`,bindings:[{name:`sourceWeights`,view:r.weights,type:`f32`,access:`read`},...typeof a==`number`?[]:[{name:`selfWeights`,view:a,type:`f32`,access:`read`}]]}];i.distances&&s.push({name:`distances`,output:i.distances,type:`f32`,value:`select(sourceDistances[sourceDistancesOffset + sourceSlot], 0.0, id == index)`,bindings:[{name:`sourceDistances`,view:r.distances,type:`f32`,access:`read`}]});let c=T(t);return C(e,{...t,bindings:[{name:`sourceOffsets`,view:r.offsets,type:`u32`,access:`read`},{name:`sourceNeighbors`,view:r.neighbors,type:`u32`,access:`read`}],visit:e=>`var inserted = false;
  for (var slot = sourceOffsets[sourceOffsetsOffset + ${c}]; slot < sourceOffsets[sourceOffsetsOffset + ${c} + 1u]; slot++) {
    let neighbor = sourceNeighbors[sourceNeighborsOffset + slot];
    if (!inserted && neighbor >= index) {
      inserted = true;
      {
        let id = index;
        let sourceSlot = slot;
        ${e}
      }
      if (neighbor == index) {
        continue;
      }
    }
    {
      let id = neighbor;
      let sourceSlot = slot;
      ${e}
    }
  }
  if (!inserted) {
    let id = index;
    let sourceSlot = 0u;
    ${e}
  }`,attributes:s})}function O(e,t,n){let{weights:r,output:i,mask:a}=n,o=[{name:`neighbors`,output:i.neighbors,type:`u32`,value:`id`},{name:`weights`,output:i.weights,type:`f32`,value:`sourceWeights[sourceWeightsOffset + slot]`,bindings:[{name:`sourceWeights`,view:r.weights,type:`f32`,access:`read`}]}];i.distances&&o.push({name:`distances`,output:i.distances,type:`f32`,value:`sourceDistances[sourceDistancesOffset + slot]`,bindings:[{name:`sourceDistances`,view:r.distances,type:`f32`,access:`read`}]});let s=T(t);return C(e,{...t,bindings:[{name:`sourceOffsets`,view:r.offsets,type:`u32`,access:`read`},{name:`sourceNeighbors`,view:r.neighbors,type:`u32`,access:`read`},{name:`mask`,view:a,type:`u32`,access:`read`}],visit:e=>`if (mask[maskOffset + index] != 0u) {
    for (var slot = sourceOffsets[sourceOffsetsOffset + ${s}]; slot < sourceOffsets[sourceOffsetsOffset + ${s} + 1u]; slot++) {
      let id = sourceNeighbors[sourceNeighborsOffset + slot];
      if (id < ROWS && mask[maskOffset + id] != 0u) {
        ${e}
      }
    }
  }`,attributes:o})}var k=`GPUSpatialWeightsAlgebra`,A={left:`leftValue`,right:`rightValue`,sum:`leftValue + rightValue`,min:`min(leftValue, rightValue)`,max:`max(leftValue, rightValue)`,product:`leftValue * rightValue`,binary:`1.0`};function j(...e){let t=e[0]??`unknown`;return(t===`raw`||t===`binary`)&&e.every(e=>e===t)?t:`unknown`}function M(...e){return e.every(e=>e.information?.symmetric===!0)?!0:`unknown`}function N(e,t){let n={source:k,criterion:e.operation};switch(e.operation){case`union`:case`intersection`:case`difference`:case`symmetricDifference`:{let r=e.operation===`difference`?[e.left]:[e.left,e.right];return{rowCount:t,transform:e.weightRule===`binary`?`binary`:e.weightRule===`sum`?j(...r.map(e=>e.information?.transform===`binary`?`unknown`:e.information?.transform)):j(...r.map(e=>e.information?.transform)),symmetric:M(e.left,e.right),hasSelfWeights:!1,provenance:n}}case`higherOrder`:return{rowCount:t,transform:`binary`,symmetric:M(e.weights),hasSelfWeights:!1,provenance:{...n,order:e.order}};case`selfWeight`:return{rowCount:t,transform:e.weights.information?.transform===`raw`?`raw`:`unknown`,symmetric:M(e.weights),hasSelfWeights:!0,provenance:n};case`spaceTime`:{let r=e.lags.every(e=>(e.weight??1)===1&&!e.includeSelf),i=e.weights.information?.transform;return{rowCount:t,transform:r&&(i===`raw`||i===`binary`)?i:`unknown`,symmetric:`unknown`,hasSelfWeights:e.weights.information?.hasSelfWeights,provenance:{...n,order:e.lags.length}}}case`subgraph`:return{rowCount:t,transform:j(e.weights.information?.transform),symmetric:M(e.weights),hasSelfWeights:e.weights.information?.hasSelfWeights,provenance:n};default:return{rowCount:t,transform:`binary`,symmetric:!0,hasSelfWeights:!1,provenance:n}}}var P=class{id;props;outputWeights;constructor(e){e=n(e);let i=e.id??`spatial-weights-algebra`;this.id=i,this.props=e;let a=y(i,e.output),c=this.getInputWeights(),l=r(i,c),u;for(let e of c)u&&e.spatialContext&&o(i,u,e.spatialContext),u??=e.spatialContext;let d=(e,t)=>{if(_(i,e,t)!==a)throw Error(`${i} ${t}.offsets length must equal output.offsets length`)};switch(e.operation){case`union`:case`intersection`:case`difference`:case`symmetricDifference`:if(d(e.left,`left`),d(e.right,`right`),e.weightRule&&!(e.weightRule in A))throw Error(`${i} unknown weightRule ${e.weightRule}`);if(e.output.distances){let t=e.operation===`union`||e.operation===`symmetricDifference`;if(!e.left.distances||t&&!e.right.distances)throw Error(`${i} output.distances needs operand distances`)}break;case`higherOrder`:d(e.weights,`weights`),p(i,`order`,e.order,1,32),e.workCapacity!==void 0&&p(i,`workCapacity`,e.workCapacity);break;case`selfWeight`:if(d(e.weights,`weights`),typeof e.selfWeight==`number`){if(!(Number.isFinite(e.selfWeight)&&e.selfWeight>=0))throw Error(`${i} selfWeight must be a non-negative finite number`)}else if(e.selfWeight.length!==a)throw Error(`${i} selfWeight view length must equal the row count`);break;case`subgraph`:if(d(e.weights,`weights`),f(e.mask,[`uint32`],`${i} mask`),e.mask.length!==a)throw Error(`${i} mask length must equal the row count`);break;case`spaceTime`:{let t=_(i,e.weights,`weights`);if(p(i,`periodCount`,e.periodCount),t*e.periodCount!==a)throw Error(`${i} output.offsets length must equal cellCount * periodCount + 1`);if(e.layout!==`period-major`&&e.layout!==`cell-major`)throw Error(`${i} layout must be 'period-major' or 'cell-major'`);if(p(i,`lags length`,e.lags.length,1,16),new Set(e.lags.map(e=>e.lag)).size!==e.lags.length)throw Error(`${i} lags must be distinct`);for(let t of e.lags){if(p(i,`lag`,t.lag,-2147483647,2147483647),!(t.weight===void 0||Number.isFinite(t.weight)&&t.weight>0))throw Error(`${i} lag weight must be positive and finite`);if(t.includeSelf&&t.lag===0)throw Error(`${i} includeSelf is not allowed at lag 0`)}break}case`block`:if(f(e.groupIds,[`uint32`],`${i} groupIds`),e.groupIds.length!==a)throw Error(`${i} groupIds length must equal the row count`);p(i,`groupCount`,e.groupCount);break;default:throw Error(`${i} unknown operation`)}if(e.output.distances&&(e.operation===`higherOrder`||e.operation===`block`||e.operation===`spaceTime`))throw Error(`${i} '${e.operation}' has no distances: omit output.distances`);if(e.output.distances&&(e.operation===`selfWeight`||e.operation===`subgraph`)&&!e.weights.distances)throw Error(`${i} output.distances requires weights.distances`);this.outputWeights=b(e.output,{spatialContext:u,information:N(e,a),status:{...e.output.status,requiredCount:e.operation===`higherOrder`&&e.order>1?void 0:e.output.status.requiredCount}});let m=[...g(e.output),...v(e.output)];h(i,m,[...c.flatMap(g),...l,e.operation===`subgraph`?e.mask:void 0,e.operation===`block`?e.groupIds:void 0,e.operation===`selfWeight`&&typeof e.selfWeight!=`number`?e.selfWeight:void 0]),(e.output.status.candidateOverflow||l.length||e.operation===`higherOrder`)&&(this.outputWeights={...this.outputWeights,status:{...this.outputWeights.status,candidateOverflow:e.output.status.candidateOverflow??s(e.output.offsets.buffer.graph,`${i}-${e.output.offsets.buffer.id}-${e.output.offsets.byteOffset}-candidate-overflow`,`uint32`,1)}}),this.outputWeights=t(this.outputWeights)}getInputWeights(){let{props:e}=this;switch(e.operation){case`union`:case`intersection`:case`difference`:case`symmetricDifference`:return[e.left,e.right];case`higherOrder`:case`selfWeight`:case`subgraph`:case`spaceTime`:return[e.weights];default:return[]}}getCommandNodes(t){e(t,this.id,this.props,{"row scratch":this.props.output.offsets.length*4,"work scratch":(this.props.operation===`higherOrder`?this.props.workCapacity??this.props.output.neighbors.length:this.props.output.neighbors.length)*4});let{id:n}=this,o={...this.props,output:{...this.props.output,status:{...this.props.output.status,candidateOverflow:this.outputWeights.status?.candidateOverflow}}};m(n,t,[...this.getInputWeights().flatMap(g),...r(n,this.getInputWeights()),...g(o.output),...v(o.output),o.operation===`subgraph`?o.mask:void 0,o.operation===`block`?o.groupIds:void 0,o.operation===`selfWeight`&&typeof o.selfWeight!=`number`?o.selfWeight:void 0]);let s=(()=>{switch(o.operation){case`higherOrder`:return this.getHigherOrderNodes(t,o);case`selfWeight`:return this.getSelfWeightNodes(t,o);case`subgraph`:return this.getSubgraphNodes(t,o);case`spaceTime`:return w(t,this.getStageBase(o.output),{...o,cellCount:o.weights.offsets.length-1});case`block`:return this.getBlockNodes(t,o);default:return this.getBinaryNodes(t,o)}})(),c=a(this.getInputWeights()),l=o.output.status.candidateOverflow;return[...l?i(t,`${n}-candidate-clear`,[],l):[],...s,...l?i(t,`${n}-inherited-candidate`,[...c.overflow,...c.candidateOverflow],l,!0):[]]}getBinaryNodes(e,t){let{left:n,right:r,output:i,operation:a}=t,o=t.weightRule??`left`;return E(e,this.getStageBase(i),{operation:a,left:n,right:r,output:i,isBinaryRule:o===`binary`,combineExpression:A[o]})}getStageBase(e){return{id:this.id,operation:k,rows:e.offsets.length-1,targetOffsets:e.offsets,capacity:e.neighbors.length,overflow:e.status.overflow,initializesOverflow:!0,totalNeighbors:e.status.requiredCount}}getHigherOrderNodes(e,t){let{id:n}=this,{weights:r,output:i,order:a}=t,o=t.cumulative??!1,c=i.offsets.length-1,l=t.workCapacity??i.neighbors.length,u=[],d=t=>({offsets:s(e,`${n}-${t}-offsets`,`uint32`,c+1),neighbors:s(e,`${n}-${t}-neighbors`,`uint32`,l)}),f={offsets:i.offsets,neighbors:i.neighbors},p=!0,m=(e,r,i)=>{let a={id:`${n}-${e}`,operation:k,rows:c,targetOffsets:r.offsets,capacity:r.neighbors.length,overflow:i?t.output.status.overflow:t.output.status.candidateOverflow,initializesOverflow:i?!0:p,totalNeighbors:i?t.output.status.requiredCount:void 0,attributes:[{name:`neighbors`,output:r.neighbors,type:`u32`,value:`id`}]};return p=!1,a},h=(e,t)=>({name:e,view:t,type:`u32`,access:`read`}),g=a===1?f:d(`exact-1`);u.push(...C(e,{...m(`exact-1`,g,a===1),bindings:[h(`baseOffsets`,r.offsets),h(`baseNeighbors`,r.neighbors)],visit:e=>`for (var slot = baseOffsets[baseOffsetsOffset + index]; slot < baseOffsets[baseOffsetsOffset + index + 1u]; slot++) {
    let id = baseNeighbors[baseNeighborsOffset + slot];
    if (id != index && id < ROWS) {
      ${e}
    }
  }`}));let _=g,v=g;for(let t=2;t<=a;t++){let n=t===a,i=n&&!o?f:d(`exact-${t}`);if(u.push(...C(e,{...m(`exact-${t}`,i,n&&!o),bindings:[h(`baseOffsets`,r.offsets),h(`baseNeighbors`,r.neighbors),h(`frontierOffsets`,_.offsets),h(`frontierNeighbors`,_.neighbors),h(`visitedOffsets`,v.offsets),h(`visitedNeighbors`,v.neighbors)],declarations:`${x(`baseContains`,`baseNeighbors`)}
${x(`visitedContains`,`visitedNeighbors`)}`,visit:e=>`let frontierBegin = frontierOffsets[frontierOffsetsOffset + index];
  let frontierEnd = frontierOffsets[frontierOffsetsOffset + index + 1u];
  let visitedBegin = visitedOffsets[visitedOffsetsOffset + index];
  let visitedEnd = visitedOffsets[visitedOffsetsOffset + index + 1u];
  for (var frontierSlot = frontierBegin; frontierSlot < frontierEnd; frontierSlot++) {
    let middle = frontierNeighbors[frontierNeighborsOffset + frontierSlot];
    if (middle >= ROWS) {
      continue;
    }
    for (var slot = baseOffsets[baseOffsetsOffset + middle]; slot < baseOffsets[baseOffsetsOffset + middle + 1u]; slot++) {
      let id = baseNeighbors[baseNeighborsOffset + slot];
      if (id == index || id >= ROWS || visitedContains(visitedBegin, visitedEnd, id)) {
        continue;
      }
      var seen = false;
      for (var earlier = frontierBegin; earlier < frontierSlot; earlier++) {
        let earlierMiddle = frontierNeighbors[frontierNeighborsOffset + earlier];
        if (earlierMiddle < ROWS && baseContains(baseOffsets[baseOffsetsOffset + earlierMiddle], baseOffsets[baseOffsetsOffset + earlierMiddle + 1u], id)) {
          seen = true;
          break;
        }
      }
      if (!seen) {
        ${e}
      }
    }
  }`,sortRows:!0})),n&&!o){_=i;break}let s=n?f:d(`visited-${t}`);u.push(...C(e,{...m(`visited-${t}`,s,n),bindings:[h(`leftOffsets`,v.offsets),h(`leftNeighbors`,v.neighbors),h(`rightOffsets`,i.offsets),h(`rightNeighbors`,i.neighbors)],visit:e=>S(`union`,e)})),_=i,v=s}return u.push(F(e,`${n}-weights`,i)),u}getSelfWeightNodes(e,t){return D(e,this.getStageBase(t.output),t)}getSubgraphNodes(e,t){return O(e,this.getStageBase(t.output),t)}getBlockNodes(e,t){let{id:n}=this,{output:r,groupIds:i,groupCount:a}=t,o=r.offsets.length-1,l=s(e,`${n}-sort-keys`,`uint32`,o),f=s(e,`${n}-sort-rows`,`uint32`,o),p=s(e,`${n}-sorted-keys`,`uint32`,o),m=s(e,`${n}-sorted-rows`,`uint32`,o);return[u(e,{id:`${n}-sort-prepare`,operation:k,variant:`sort-prepare`,bindings:[{name:`groupIds`,view:i,type:`u32`,access:`read`},{name:`sortKeys`,view:l,type:`u32`,access:`read_write`},{name:`sortRows`,view:f,type:`u32`,access:`read_write`}],invocationCount:o,declarations:`const GROUP_COUNT: u32 = ${a}u;`,body:`sortKeys[sortKeysOffset + index] = min(groupIds[groupIdsOffset + index], GROUP_COUNT);
  sortRows[sortRowsOffset + index] = index;`}),...new c({id:`${n}-sort`,keys:l,values:f,outputKeys:p,outputValues:m,keyBits:d(a)}).getCommandNodes(e),...C(e,{id:n,operation:k,rows:o,targetOffsets:r.offsets,capacity:r.neighbors.length,overflow:t.output.status.overflow,initializesOverflow:!0,totalNeighbors:t.output.status.requiredCount,bindings:[{name:`groupIds`,view:i,type:`u32`,access:`read`},{name:`sortedKeys`,view:p,type:`u32`,access:`read`},{name:`sortedRows`,view:m,type:`u32`,access:`read`}],declarations:`const GROUP_COUNT: u32 = ${a}u;
fn lowerBound(key: u32) -> u32 {
  var low = 0u;
  var high = ${o}u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,visit:e=>`let group = min(groupIds[groupIdsOffset + index], GROUP_COUNT);
  if (group < GROUP_COUNT) {
    let groupEnd = lowerBound(group + 1u);
    for (var position = lowerBound(group); position < groupEnd; position++) {
      let id = sortedRows[sortedRowsOffset + position];
      if (id != index) {
        ${e}
      }
    }
  }`,attributes:[{name:`neighbors`,output:r.neighbors,type:`u32`,value:`id`},{name:`weights`,output:r.weights,type:`f32`,value:`1.0`}]})]}};function F(e,t,n){return u(e,{id:t,operation:k,variant:`weights`,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`weights`,view:n.weights,type:`f32`,access:`read_write`}],invocationCount:n.offsets.length-1,body:`for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    weights[weightsOffset + slot] = 1.0;
  }`})}export{D as a,E as i,P as n,O as o,j as r,A as t};
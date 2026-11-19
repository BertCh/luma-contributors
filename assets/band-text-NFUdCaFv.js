import{o as e,t}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as n}from"./kernel-C3tfQOn4.js";import{A as r,E as i,f as a,g as o,i as s,k as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{t as u}from"./scaled-quotient-wgsl-UDyhQag6.js";import{t as d}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{t as f}from"./gpu-raster-buffer-to-texture-BDZQU4A5.js";import{o as p,s as m}from"./network-view-utils-B1U8oNKx.js";var h=2**-149,g=(2-2**-23)*2**127,_=class{id;props;nodeCount;resolution;weightScale;constructor(e){this.id=e.id??`adjacency-matrix`,this.props=e,m(this.id,e.output,[...v(e)<=e.topology.offsets.buffer.graph.device.limits.maxStorageBuffersPerShaderStage?[{outputs:{counts:e.output.counts,weightSums:e.output.weightSums}}]:[],{outputs:{maximumCount:e.output.maximumCount,maximumWeightSum:e.output.maximumWeightSum},inputs:{counts:e.output.maximumCount?e.output.counts:void 0,weightSums:e.output.maximumWeightSum?e.output.weightSums:void 0}}]);let{id:t}=this;if(this.nodeCount=r(t,e.topology).nodeCount,this.resolution=e.resolution,a(t,`resolution`,e.resolution,1,4096),this.nodeCount*e.resolution>4294967295)throw Error(`${t} node count times resolution must be below 2^32`);if(this.weightScale=e.weightScale??1024,!Number.isFinite(this.weightScale)||this.weightScale<=0)throw Error(`${t} weightScale must be positive and finite`);o(t,`weightScale`,this.weightScale,h,g);for(let[n,r,a]of[[`nodeMask`,e.nodeMask,this.nodeCount],[`order`,e.order,this.nodeCount],[`edgeMask`,e.edgeMask,e.topology.neighbors.length]])if(r&&(i(r,[`uint32`],`${t} ${n}`),r.length!==a))throw Error(`${t} ${n} must contain ${a} rows`);if(e.parameters&&(i(e.parameters,[`uint32`],`${t} parameters`),e.parameters.length<4))throw Error(`${t} parameters must contain at least four uint32 rows`);let{output:n}=e,s=e.resolution*e.resolution;for(let[e,r,a]of[[`counts`,n.counts,s],[`weightSums`,n.weightSums,s],[`maximumCount`,n.maximumCount,1],[`maximumWeightSum`,n.maximumWeightSum,1]])if(r&&(i(r,[`uint32`],`${t} output.${e}`),e===`maximumCount`||e===`maximumWeightSum`?r.length<1:r.length!==a))throw Error(`${t} output.${e} must contain ${a===1?`at least one row`:`exactly ${a} rows`}`);if((n.weightSums||n.maximumWeightSum||n.textureStatistic===`weightSum`)&&!e.topology.weights)throw Error(`${t} weight outputs require weights`);if(n.maximumWeightSum&&!n.weightSums)throw Error(`${t} output.maximumWeightSum requires output.weightSums`);if(n.textureStatistic===`weightSum`&&!n.weightSums)throw Error(`${t} textureStatistic 'weightSum' requires output.weightSums`);if(n.texture&&(n.texture.format!==`r32float`||n.texture.width!==e.resolution||n.texture.height!==e.resolution))throw Error(`${t} output.texture must be r32float with resolution x resolution extent`);l(t,[n.counts,n.weightSums,n.maximumCount,n.maximumWeightSum],[...c(e.topology),e.nodeMask,e.edgeMask,e.order,e.parameters])}getCommandNodes(r){let{id:i,props:a,nodeCount:o,resolution:l}=this,{output:d}=a,{offsets:m,neighbors:h,weights:g}=a.topology;if(p(i,r,[...c(a.topology),a.nodeMask,a.edgeMask,a.order,a.parameters,d.counts,d.weightSums,d.maximumCount,d.maximumWeightSum]),d.texture&&d.texture.texture.graph!==r)throw Error(`${i} views must belong to the target graph`);p(i,r,[],{textureConversion:l*l*4});let _=`GPUAdjacencyMatrix`,y=!a.directed&&!!a.mirrorSlots,b=[],x=(e,n)=>{n&&b.push(t(r,{id:`${i}-${e}`,operation:_,view:n,type:`u32`,value:`0u`}))};x(`zero-counts`,d.counts),x(`zero-weight-sums`,d.weightSums),x(`zero-max-count`,d.maximumCount),x(`zero-max-weight-sum`,d.maximumWeightSum);let S=a.parameters?[{name:`windowWords`,view:a.parameters,type:`u32`,access:`read`}]:[],C=a.parameters?`let rowStart = windowWords[windowWordsOffset];
  let rowEnd = windowWords[windowWordsOffset + 1u];
  let colStart = windowWords[windowWordsOffset + 2u];
  let colEnd = windowWords[windowWordsOffset + 3u];`:`let rowStart = 0u;
  let rowEnd = NODE_COUNT;
  let colStart = 0u;
  let colEnd = NODE_COUNT;`,w=(t,n)=>{let s=[{name:`offsets`,view:m,type:`u32`,access:`read`},{name:`neighbors`,view:h,type:`u32`,access:`read`},...S];a.order&&s.push({name:`order`,view:a.order,type:`u32`,access:`read`}),a.nodeMask&&s.push({name:`vertexMask`,view:a.nodeMask,type:`u32`,access:`read`}),a.edgeMask&&s.push({name:`edgeMask`,view:a.edgeMask,type:`u32`,access:`read`}),n.some(e=>e.kind===`weight`)&&s.push({name:`weights`,view:g,type:`f32`,access:`read`});for(let{kind:e,view:t}of n)s.push({name:`${e}Cells`,view:t,type:`atomic<u32>`,access:`read_write`});let c=e=>e===`count`?`1u`:`fixedWeight(weights[weightsOffset + slot])`;b.push(e(r,{id:`${i}-${t}`,operation:_,variant:t,bindings:s,invocationCount:o,declarations:`const NODE_COUNT: u32 = ${o}u;
const RESOLUTION: u32 = ${l}u;
const WEIGHT_SCALE: f32 = ${Math.fround(this.weightScale).toExponential(9)}f;
// Rounds the float32 product half up. The product feeds only floor() and comparisons, so no
// compiler can fuse it into an addition, and above 2^23 (where every float32 is an integer and
// adding 0.5 would round to the even neighbour) the product is returned as it is.
fn fixedWeight(weight: f32) -> u32 {
  let scaled = weight * WEIGHT_SCALE;
  if (!(scaled > 0.0)) {
    return 0u;
  }
  if (scaled >= 4294967040.0) {
    return 0xffffffffu;
  }
  let whole = floor(scaled);
  let roundsUp = whole < 8388608.0 && scaled >= whole + 0.5;
  return u32(whole) + select(0u, 1u, roundsUp);
}
fn getBin(position: u32, start: u32, end: u32) -> u32 {
  return (position - start) * RESOLUTION / (end - start);
}`,body:`${C}
  if (${a.nodeMask?`vertexMask[vertexMaskOffset + index] == 0u`:`false`}) {
    return;
  }
  let source = ${a.order?`order[orderOffset + index]`:`index`};
  // A row contributes only from its own row position, or, when mirrored, from its column
  // position too. Skipping every other row keeps a zoomed window O(window rows), not O(slots).
  if (!((source >= rowStart && source < rowEnd)${y?` || (source >= colStart && source < colEnd)`:``})) {
    return;
  }
  let rowBegin = offsets[offsetsOffset + index];
  let rowEnd2 = offsets[offsetsOffset + index + 1u];
  for (var slot = rowBegin; slot < rowEnd2; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= NODE_COUNT) {
      continue;
    }
    ${a.nodeMask?`if (vertexMask[vertexMaskOffset + neighbor] == 0u) { continue; }`:``}
    ${a.edgeMask?`if (edgeMask[edgeMaskOffset + slot] == 0u) { continue; }`:``}
    let targetPosition = ${a.order?`order[orderOffset + neighbor]`:`neighbor`};
    if (source >= rowStart && source < rowEnd && targetPosition >= colStart && targetPosition < colEnd) {
      let cell = getBin(source, rowStart, rowEnd) * RESOLUTION + getBin(targetPosition, colStart, colEnd);
      ${(e=>n.map(({kind:t})=>`atomicAdd(&${t}Cells[${t}CellsOffset + ${e}], ${c(t)});`).join(`
      `))(`cell`)}
    }
    ${y?`if (targetPosition != source && targetPosition >= rowStart && targetPosition < rowEnd && source >= colStart && source < colEnd) {
      let mirrorCell = getBin(targetPosition, rowStart, rowEnd) * RESOLUTION + getBin(source, colStart, colEnd);
      ${(e=>n.map(({kind:t})=>`atomicAdd(&${t}Cells[${t}CellsOffset + ${e}], ${c(t)});`).join(`
      `))(`mirrorCell`)}
    }`:``}
  }`}))},T=[{kind:`count`,view:d.counts}];d.weightSums&&T.push({kind:`weight`,view:d.weightSums});let E=v(a);if(T.length===1||E<=r.device.limits.maxStorageBuffersPerShaderStage?w(T.length===1?`bin-counts`:`bin`,T):(w(`bin-counts`,[T[0]]),w(`bin-weights`,[T[1]])),d.maximumCount||d.maximumWeightSum){let t=[],n=``,a=``,o=``,s=``;d.maximumCount&&(t.push({name:`counts`,view:d.counts,type:`u32`,access:`read`},{name:`maximumCountOut`,view:d.maximumCount,type:`atomic<u32>`,access:`read_write`}),s+=`var<workgroup> sharedCounts: array<u32, 256>;
`,n+=`sharedCounts[localInvocationIndex] = select(0u, counts[countsOffset + index], inRange);
  `,a+=`sharedCounts[localInvocationIndex] = max(sharedCounts[localInvocationIndex], sharedCounts[localInvocationIndex + stride]);
      `,o+=`atomicMax(&maximumCountOut[maximumCountOutOffset], sharedCounts[0]);
    `),d.maximumWeightSum&&(t.push({name:`sums`,view:d.weightSums,type:`u32`,access:`read`},{name:`maxSumOut`,view:d.maximumWeightSum,type:`atomic<u32>`,access:`read_write`}),s+=`var<workgroup> sharedSums: array<u32, 256>;
`,n+=`sharedSums[localInvocationIndex] = select(0u, sums[sumsOffset + index], inRange);
  `,a+=`sharedSums[localInvocationIndex] = max(sharedSums[localInvocationIndex], sharedSums[localInvocationIndex + stride]);
      `,o+=`atomicMax(&maxSumOut[maxSumOutOffset], sharedSums[0]);
    `),b.push(e(r,{id:`${i}-maxima`,operation:_,variant:`maxima`,bindings:t,invocationCount:l*l,guardIndex:!1,declarations:s,body:`let inRange = index < INVOCATION_COUNT;
  ${n}workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      ${a}}
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    ${o}}`}))}if(d.texture){let t=d.textureStatistic===`weightSum`?d.weightSums:d.counts,a=n(r,`${i}-texture-values`,`float32`,l*l),o=d.textureStatistic===`weightSum`?Math.fround(this.weightScale):1;b.push(e(r,{id:`${i}-texture-values`,operation:_,variant:`texture-values`,bindings:[{name:`source`,view:t,type:`u32`,access:`read`},{name:`valuesOut`,view:a,type:`f32`,access:`read_write`}],invocationCount:l*l,declarations:u,body:`valuesOut[valuesOutOffset + index] = getScaledQuotient(f32(source[sourceOffset + index]), ${o.toExponential(9)}f, 1.0, 0);`}));let c=d.texture;b.push(...s(r,()=>new f({id:`${i}-texture`,input:{id:`${i}-texture-band`,format:`float32`,storage:{kind:`buffer`,values:a}},output:c}).addToGraph(r)))}return b}};function v(e){return 3+Number(!!e.parameters)+Number(!!e.order)+Number(!!e.nodeMask)+Number(!!e.edgeMask)+2*Number(!!e.output.weightSums)}var y=class{id;props;constructor(e){if(this.id=e.id??`adjacency-matrix-order`,this.props=e,i(e.groups,[`uint32`],`${this.id} groups`),i(e.output.order,[`uint32`],`${this.id} output.order`),e.groups.length<1)throw Error(`${this.id} groups must contain at least one row`);for(let[t,n]of[[`output.order`,e.output.order],[`tieKeys`,e.tieKeys]])if(n&&(i(n,[`uint32`],`${this.id} ${t}`),n.length!==e.groups.length))throw Error(`${this.id} ${t} must contain one row per node`);l(this.id,[e.output.order],[e.groups,e.tieKeys])}getCommandNodes(t){let{id:r,props:i}=this;p(r,t,[i.groups,i.tieKeys,i.output.order]);let a=i.groups.length;p(r,t,[],{sortRows:a*4});let o=`GPUAdjacencyMatrixOrder`,s=e=>n(t,`${r}-${e}`,`uint32`,a),c=[],l=s(`identity`);c.push(e(t,{id:`${r}-identity`,operation:o,variant:`identity`,bindings:[{name:`identityOut`,view:l,type:`u32`,access:`read_write`}],invocationCount:a,body:`identityOut[identityOutOffset + index] = index;`}));let u=d(t,{id:`${r}-sort`,operation:o,indices:l,indicesAreIdentity:!0,keys:[{view:i.groups},...i.tieKeys?[{view:i.tieKeys}]:[]]});c.push(...u.nodes);let f=u.sortedIndices;return c.push(e(t,{id:`${r}-scatter`,operation:o,variant:`scatter`,bindings:[{name:`vertices`,view:f,type:`u32`,access:`read`},{name:`orderOut`,view:i.output.order,type:`u32`,access:`read_write`}],invocationCount:a,body:`orderOut[orderOutOffset + vertices[verticesOffset + index]] = index;`})),c}};function b(e,t,n){let r=Array(n).fill(0);for(let t of e.groups)r[t]++;let i=[],a=[],o=0;return r.forEach((r,s)=>{let c=o/e.nodeCount*t.width,l=(o+r)/e.nodeCount*t.width,u=t.height*(1-o/e.nodeCount),d=t.height*(1-(o+r)/e.nodeCount),f=[[c,u],[l,u],[l,d],[c,d],[c,u]];for(let e=0;e<4;e++)a.push({from:f[e],to:f[e+1],tone:`ink`,width:1.2,opacity:.6,dash:5});i.push({text:s===0?`band 1 (north)`:s===n-1?`band ${s+1} (south)`:`band ${s+1}`,at:[c,u],anchor:`top-left`,offset:[4,-3],role:`label`,group:`bands`}),o+=r}),i.push({text:`tract links, either direction (1 in 6 cross-band kept)`,at:[0,1],space:`canvas`,anchor:`bottom-left`,offset:[4,4],role:`tick`,priority:3}),{labels:i,rules:a}}function x(){return{labels:[{text:`tract links, either direction (1 in 6 cross-band kept)`,at:[0,1],space:`canvas`,anchor:`bottom-left`,offset:[4,4],role:`tick`,priority:3}]}}export{_ as i,x as n,y as r,b as t};
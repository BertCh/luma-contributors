import{t as e}from"./storage-backed-top-k-Dm9YLNx9.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-sort-xRm8l5yn.js";import{c as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as i}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a,d as o,t as s}from"./spatial-join-geometry-DYRzw652.js";import{c,p as l,v as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d,i as f,n as p,o as m,r as h,s as g}from"./nearest-wgsl-ClkerZHh.js";import{i as _,l as v,m as y,n as b,r as x,t as S,v as C}from"./spatial-join-passes-CNxNpnDa.js";var w=5,T=1,E=4096,D=8,O=1024;function k(e,t,n,r,i,a,o=!1){let s=Math.max(3+t*2,n?t*w:0),c=a??Math.max(1,Math.floor((i/4-3)/s));return E+(o?O:0)+D*(Math.ceil(e/c)+r)}function A(e){if(a(e))return e.featureOffsets.length-1;switch(e.kind){case`points`:return e.positions.length;case`segments`:return e.starts.length;case`lines`:return e.lineOffsets.length-1;case`polygons`:return e.featureOffsets.length-1}}function j(e){switch(e.kind){case`points`:return[e.positions,...a(e)?[e.featureOffsets]:[]];case`segments`:return[e.starts,e.ends];case`lines`:return[e.positions,e.lineOffsets,...a(e)?[e.featureOffsets]:[]];case`polygons`:return[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets]}}function M(e){return e.kind===`segments`?e.starts.length*2:e.positions.length}function N(e,t,n){let r=(t,n,r)=>({name:`${e}${t}`,view:n,type:r,access:`read`});switch(t.kind){case`points`:return[r(`Positions`,t.positions,`f32`),...a(t)?[r(`FeatureOffsets`,t.featureOffsets,`u32`)]:[]];case`segments`:return[r(`Starts`,t.starts,`f32`),r(`Ends`,t.ends,`f32`)];case`lines`:return[r(`Positions`,t.positions,`f32`),r(`LineOffsets`,t.lineOffsets,`u32`),...a(t)?[r(`FeatureOffsets`,t.featureOffsets,`u32`)]:[]];case`polygons`:return[r(`Positions`,t.positions,`f32`),r(`FeatureRings`,n,`u32`),r(`RingOffsets`,t.ringOffsets,`u32`)]}}function P(t,n,r,i,a){return e(t,{id:n,operation:r,variant:`polygon-rings`,bindings:[{name:`featureOffsets`,view:i.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:i.polygonOffsets,type:`u32`,access:`read`},{name:`featureRings`,view:a,type:`u32`,access:`read_write`}],invocationCount:a.length,declarations:`const POLYGON_COUNT: u32 = ${i.polygonOffsets.length-1}u;
const RING_COUNT: u32 = ${i.ringOffsets.length-1}u;`,body:`let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var ringStart = 0u;
  var ringEnd = 0u;
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let first = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let last = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (first <= last && last <= RING_COUNT) { ringStart = first; ringEnd = last; }
  }
  featureRings[featureRingsOffset + index * 2u] = ringStart;
  featureRings[featureRingsOffset + index * 2u + 1u] = ringEnd;`})}function ee(e,t,n,r,i,o,c){return r.kind===`lines`||a(r)?s(e,t,{operation:n,featureCount:i,geometry:r,minima:o,maxima:c}):_(e,{id:t,operation:n,featureCount:i,source:r,minima:o,maxima:c})}function te(e,n){let{id:r,operation:i,features:a,featureCount:o,prepared:s}=n;if(s){if(!s.isDeclaredIn(e))throw Error(`${r} prepared handle must be added to the graph before the join`);let{storage:t}=s;return{bvh:{nodeMinima:t.nodeMinima,nodeMaxima:t.nodeMaxima,leafIds:t.leafIds,overflow:t.overflow,internalNodeCount:s.leafCapacity-1,nodeCount:2*s.leafCapacity-1,levelCount:Math.log2(s.leafCapacity)+1},nodes:[]}}let c=t(e,`${r}-feature-minima`,`float32x2`,o),l=t(e,`${r}-feature-maxima`,`float32x2`,o),u=[];o>0&&u.push(ee(e,`${r}-bounds`,i,a,o,c,l));let d=C(e,r,i,c,l,n.leafCapacity,n.spatialSort,void 0,n.spatialSortCurve);return u.push(...d.nodes),{bvh:d.bvh,nodes:u}}function ne(e,t){let{id:n,capacity:r,queryCount:s,featureCount:d}=t;o(n,[t.writeFeatureRows?void 0:t.featureIds],[t.neighborIds,t.neighborRows,t.neighborCounts,t.neighborDistances,t.overflow]);let f=3+2*r,p=!!(t.neighborFootPoints||t.neighborQueryPoints||t.neighborSegmentIndices),m=Math.floor(e.device.limits.maxStorageBufferBindingSize/4),h=Math.max(f,p?r*w:0),g=t.scratchChunkQueryCount;if(g!==void 0&&(!Number.isSafeInteger(g)||g<T))throw Error(`${n} scratchChunkQueryCount must be an integer at least MINIMUM_SCRATCH_QUERY_COUNT (1)`);let _=Math.floor((m-3)/h),v=Math.min(Math.max(s,1),g??_,_),y=s>v;if(v<1)throw Error(`${n} needs more scratch per query than one binding holds; reduce k`);let b=v*f,x=b+1;u(e.device,n,`topk`,(v*f+3)*4,`reduce the query count or k`);let S=[...t.queries.kind===`points`?c(t.queries.points):j(t.queries.geometry),...j(t.features),t.featureIds,t.maxDistance,t.queryIds,t.onAttribute?.left,t.onAttribute?.right,t.neighborIds,t.neighborRows,t.neighborCounts,t.neighborDistances,t.neighborFootPoints,t.neighborQueryPoints,t.neighborSegmentIndices,t.overflow,...t.prepared?Object.values(t.prepared.storage):[]];i(e,n,Object.fromEntries(S.filter(e=>!!(e&&typeof e==`object`&&`rowByteLength`in e)).map((e,t)=>[`nearest view ${t}`,e])));let C=2*t.leafCapacity-1,E=!!(t.exclusive||t.onAttribute);for(let[i,o]of[[`nodeData`,C*5+2*s+(E?2*d:0)],[`BVH node bounds and children`,C*2],[`feature bounds and sorting scratch`,d*2],[`query sorting scratch and bounds`,t.querySort?s*2:0],[`extras`,p?v*r*w:0],[`feature rings`,t.features.kind===`polygons`?d*2:0],[`query rings`,t.queries.kind===`geometry`&&t.queries.geometry.kind===`polygons`?s*2:0],[`query metadata`,t.querySort&&s>=2&&t.queries.kind===`geometry`&&t.queries.geometry.kind===`polygons`?s*2+t.queries.geometry.ringOffsets.length:t.querySort&&s>=2&&t.queries.kind===`geometry`&&t.queries.geometry.kind===`lines`&&a(t.queries.geometry)?t.queries.geometry.featureOffsets.length+t.queries.geometry.lineOffsets.length:0]])u(e.device,n,i,o*4,i===`query metadata`?`reduce query topology rows or disable querySort`:`reduce feature count, query count or neighborCapacity`);return l(n,k(s,r,p,t.queries.kind===`points`?c(t.queries.points).length:1,e.device.limits.maxStorageBufferBindingSize,t.scratchChunkQueryCount,t.querySort)),{stride:f,needsFootPass:p,scratchQueryCount:v,isChunked:y,cursorWord:b,dispatchWord:x}}function re(r,i,a,o,s,c){let l=t(r,`${i}-query-order-keys`,`uint32`,s),u=t(r,`${i}-query-order-rows`,`uint32`,s),d=t(r,`${i}-query-order-chunks`,`uint32`,s),f=t(r,`${i}-query-order-sorted-keys`,`uint32`,s),p=t(r,`${i}-query-order-sorted-rows`,`uint32`,s),m=t(r,`${i}-query-order-group-keys`,`uint32`,s),h=t(r,`${i}-query-order-output-keys`,`uint32`,s),g=t(r,`${i}-query-order`,`uint32`,s),_=[],v=0;for(let[t,n]of o.entries())n.length&&_.push(e(r,{id:`${i}-query-keys-${t}`,operation:a,variant:`query-keys-hilbert`,firstRow:v,batchIndex:t,invocationCount:n.length,bindings:[{name:`points`,view:n,type:`f32`,access:`read`},{name:`nodeMinima`,view:c.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:c.nodeMaxima,type:`f32`,access:`read`},{name:`keys`,view:l,type:`u32`,access:`read_write`},{name:`rows`,view:u,type:`u32`,access:`read_write`},{name:`chunkIds`,view:d,type:`u32`,access:`read_write`}],declarations:`${x}${b}${S}`,body:`let row = batchFirstRow + index;
  rows[rowsOffset + row] = row;
  chunkIds[chunkIdsOffset + row] = batchIndex;
  let point = vec2f(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var key = 0xffffffffu;
  if (isFiniteValue(point.x) && isFiniteValue(point.y)) {
    let column = quantizeAxis(point.x, nodeMinima[nodeMinimaOffset], nodeMaxima[nodeMaximaOffset]);
    let rasterRow = quantizeAxis(point.y, nodeMinima[nodeMinimaOffset + 1u], nodeMaxima[nodeMaximaOffset + 1u]);
    key = getHilbertIndex(column, rasterRow);
  }
  keys[keysOffset + row] = key;`})),v+=n.length;return _.push(...new n({id:`${i}-query-hilbert-sort`,keys:l,values:u,outputKeys:f,outputValues:p,algorithm:`radix`}).getCommandNodes(r)),_.push(e(r,{id:`${i}-query-chunk-keys`,operation:a,variant:`query-chunk-keys`,invocationCount:s,bindings:[{name:`sortedRows`,view:p,type:`u32`,access:`read`},{name:`chunkIds`,view:d,type:`u32`,access:`read`},{name:`groupKeys`,view:m,type:`u32`,access:`read_write`}],body:`groupKeys[groupKeysOffset + index] = chunkIds[chunkIdsOffset + sortedRows[sortedRowsOffset + index]];`})),_.push(...new n({id:`${i}-query-chunk-sort`,keys:m,values:p,outputKeys:h,outputValues:g,algorithm:`radix`}).getCommandNodes(r)),{nodes:_,order:g}}function F(n,i){let{id:o,operation:s,features:l,featureCount:_,queryCount:b,k:x,capacity:S,sphereRadius:C}=i,{stride:T,needsFootPass:E,scratchQueryCount:D,isChunked:O,cursorWord:k,dispatchWord:A}=ne(n,i),j=C!==void 0,F=j?r(Math.fround(C)):``,I=[],{bvh:L,nodes:ie}=te(n,{id:o,operation:s,features:l,featureCount:_,leafCapacity:i.leafCapacity,spatialSort:i.spatialSort,spatialSortCurve:i.spatialSortCurve,prepared:i.prepared});I.push(...ie);let R=!!(i.exclusive||i.onAttribute),z=L.nodeCount*5,B=z+2*b,V=t(n,`${o}-node-data`,`uint32`,B+(R?2*_:0));if(I.push(e(n,{id:`${o}-node-data`,operation:s,variant:`node-data`,bindings:[{name:`nodeMinima`,view:L.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:L.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:L.leafIds,type:`u32`,access:`read`},{name:`nodeData`,view:V,type:`u32`,access:`read_write`}],invocationCount:L.nodeCount,declarations:`const INTERNAL_NODE_COUNT: u32 = ${L.internalNodeCount}u;
const NO_FEATURE: u32 = 0xffffffffu;`,body:`let base = nodeDataOffset + index * 5u;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(NO_FEATURE, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`})),R){let t=[{name:`nodeData`,view:V,type:`u32`,access:`read_write`}],r=[[`queryIds`,i.exclusive?i.queryIds:void 0],[`featureIds`,i.exclusive?i.featureIds:void 0],[`queryKeys`,i.onAttribute?.left],[`featureKeys`,i.onAttribute?.right]];for(let[e,n]of r)n&&t.push({name:e,view:n,type:`u32`,access:`read`});let a=(e,t)=>r.find(([t,n])=>t===e&&n)?`${e}[${e}Offset + index]`:t;I.push(e(n,{id:`${o}-filter`,operation:s,variant:`filter`,bindings:t,invocationCount:Math.max(b,_,1),declarations:`const QUERY_COUNT: u32 = ${b}u;
const FEATURE_COUNT: u32 = ${_}u;`,body:`if (index < QUERY_COUNT) {
    let base = nodeDataOffset + ${z}u + index * 2u;
    nodeData[base] = ${a(`queryIds`,`index`)};
    nodeData[base + 1u] = ${a(`queryKeys`,`0u`)};
  }
  if (index < FEATURE_COUNT) {
    let base = nodeDataOffset + ${B}u + index * 2u;
    nodeData[base] = ${a(`featureIds`,`index`)};
    nodeData[base + 1u] = ${a(`featureKeys`,`0u`)};
  }`}))}let H;l.kind===`polygons`&&i.prepared?H=i.prepared.storage.featureRings:l.kind===`polygons`&&(H=t(n,`${o}-feature-rings`,`uint32x2`,_),_>0&&I.push(P(n,`${o}-feature-rings`,s,l,H)));let U=i.queries.kind===`geometry`?i.queries.geometry:void 0,W;U?.kind===`polygons`&&(W=t(n,`${o}-query-rings`,`uint32x2`,b),b>0&&I.push(P(n,`${o}-query-rings`,s,U,W)));let G=t(n,`${o}-topk`,`uint32`,D*T+3),K=[{name:`topk`,view:G,type:`u32`,access:`read_write`},{name:`overflow`,view:i.overflow,type:`u32`,access:`read_write`},{name:`bvhOverflow`,view:L.overflow,type:`u32`,access:`read`}];i.maxDistance&&K.push({name:`maxDistance`,view:i.maxDistance,type:`f32`,access:`read`});let q=`// A NaN, negative or infinite maxDistance sets a reserved finite rank that matches nothing.
    ${i.maxDistance?`let limit = maxDistance[maxDistanceOffset];
    var bound = NEAREST_INVALID_DISTANCE;
    if (limit >= 0.0 && limit <= FLOAT32_MAXIMUM) { bound = ${j?`// Canonicalize negative zero before storing the central-angle rank.
    abs(limit / SPHERE_RADIUS)`:`encodeNearestDistance(limit)`}; }`:`let bound = NEAREST_NO_DISTANCE;`}
    topk[header] = 0u;
    topk[header + 1u] = 0u;
    topk[header + 2u] = bitcast<u32>(bound);`,J=`${h}
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const STRIDE: u32 = ${T}u;${j?`\nconst SPHERE_RADIUS: f32 = ${F};`:``}`;O?I.push(e(n,{id:`${o}-clear-overflow`,operation:s,variant:`clear-overflow`,bindings:K.filter(e=>e.name.endsWith(`verflow`)),invocationCount:1,body:`overflow[overflowOffset] = bvhOverflow[bvhOverflowOffset];`})):I.push(e(n,{id:`${o}-clear`,operation:s,variant:j?`clear-great-circle`:`clear`,bindings:K,invocationCount:Math.max(b,1),declarations:`${J}
const QUERY_COUNT: u32 = ${b}u;`,body:`if (index == 0u) { overflow[overflowOffset] = bvhOverflow[bvhOverflowOffset]; }
  if (index < QUERY_COUNT) {
    let header = topkOffset + index * STRIDE;
    ${q}
  }`}));let Y=[];if(i.queries.kind===`points`){let e=c(i.queries.points),t=i.querySort&&b>1?re(n,o,s,e,b,L):void 0;t&&I.push(...t.nodes);let r=0;for(let[a,s]of e.entries()){if(s.length>0){let e={kind:`points`,positions:s},c=t?n.createDataView(t.order.buffer,{format:`uint32`,byteOffset:r*4,length:s.length}):void 0;Y.push({chunkId:v(o,i.queries.points,a),side:{prefix:`q`,kind:`points`,vertexCount:s.length},bindings:N(`q`,e,void 0),count:s.length,firstRow:r,kind:`points`,order:c})}r+=s.length}}else if(b>0){let r=i.queries.geometry,c={prefix:`q`,kind:r.kind,vertexCount:M(r),partCount:a(r)?r.kind===`points`?r.positions.length:r.lineOffsets.length-1:void 0},l=N(`q`,r,W),d;if(i.querySort&&b>=2){if(r.kind===`polygons`){let i=b*2;u(n.device,o,`query metadata`,(i+r.ringOffsets.length)*4,`reduce the polygon query count or disable querySort`);let a=t(n,`${o}-query-metadata`,`uint32`,i+r.ringOffsets.length);I.push(e(n,{id:`${o}-query-metadata`,operation:s,variant:`query-metadata`,bindings:[{name:`featureRings`,view:W,type:`u32`,access:`read`},{name:`ringOffsets`,view:r.ringOffsets,type:`u32`,access:`read`},{name:`metadata`,view:a,type:`u32`,access:`read_write`}],invocationCount:a.length,body:`if (index < ${i}u) {
    metadata[metadataOffset + index] = featureRings[featureRingsOffset + index];
  } else {
    metadata[metadataOffset + index] = ringOffsets[ringOffsetsOffset + index - ${i}u];
  }`})),c.packedRingOffsetsBase=i,l=[{name:`qPositions`,view:r.positions,type:`f32`,access:`read`},{name:`qFeatureRings`,view:a,type:`u32`,access:`read`}]}else if(r.kind===`lines`&&a(r)){let i=r.featureOffsets.length;u(n.device,o,`multipart query metadata`,(i+r.lineOffsets.length)*4,`reduce query parts or disable querySort`);let a=t(n,`${o}-query-metadata`,`uint32`,i+r.lineOffsets.length);I.push(e(n,{id:`${o}-query-metadata`,operation:s,variant:`multipart-query-metadata`,bindings:[{name:`featureOffsets`,view:r.featureOffsets,type:`u32`,access:`read`},{name:`lineOffsets`,view:r.lineOffsets,type:`u32`,access:`read`},{name:`metadata`,view:a,type:`u32`,access:`read_write`}],invocationCount:a.length,body:`if (index < ${i}u) { metadata[metadataOffset + index] = featureOffsets[featureOffsetsOffset + index]; }
  else { metadata[metadataOffset + index] = lineOffsets[lineOffsetsOffset + index - ${i}u]; }`})),c.packedRingOffsetsBase=i,l=[{name:`qPositions`,view:r.positions,type:`f32`,access:`read`},{name:`qFeatureOffsets`,view:a,type:`u32`,access:`read`}]}let i=t(n,`${o}-query-minima`,`float32x2`,b),f=t(n,`${o}-query-maxima`,`float32x2`,b),p=t(n,`${o}-query-centers`,`float32x2`,b);I.push(ee(n,`${o}-query-bounds`,s,r,b,i,f),e(n,{id:`${o}-query-centers`,operation:s,variant:`query-centers`,bindings:[{name:`minima`,view:i,type:`f32`,access:`read`},{name:`maxima`,view:f,type:`f32`,access:`read`},{name:`centers`,view:p,type:`u32`,access:`read_write`}],invocationCount:b,body:`let minimum = vec2f(minima[minimaOffset + index * 2u], minima[minimaOffset + index * 2u + 1u]);
  let maximum = vec2f(maxima[maximaOffset + index * 2u], maxima[maximaOffset + index * 2u + 1u]);
  // Halve before adding to avoid overflow for large finite bounds. Empty bounds sort last.
  let center = minimum * 0.5 + maximum * 0.5;
  let valid = all(minimum <= maximum);
  centers[centersOffset + index * 2u] = select(0x7fc00000u, bitcast<u32>(center.x), valid);
  centers[centersOffset + index * 2u + 1u] = select(0x7fc00000u, bitcast<u32>(center.y), valid);`}));let m=y(n,{id:`${o}-order`,operation:s,points:p,bvh:L});I.push(...m.nodes),d=m.order}Y.push({chunkId:o,side:c,bindings:l,count:b,firstRow:0,kind:r.kind,order:d})}let X={prefix:`r`,kind:l.kind,vertexCount:M(l),partCount:a(l)?l.kind===`points`?l.positions.length:l.lineOffsets.length-1:void 0},ae=N(`r`,l,H),oe={name:`nodeData`,view:V,type:`u32`,access:`read`},Z=e=>({name:`topk`,view:G,type:`u32`,access:e}),se=L.levelCount+2;E&&u(n.device,o,`extras`,D*S*w*4,`reduce the query count or k, or omit the neighbor foot-point outputs`);let Q=E?t(n,`${o}-extras`,`uint32`,D*S*w):void 0,ce=(t,r,a,o)=>{let c=a?o:void 0,l=c?[{name:`queryOrder`,view:c,type:`u32`,access:`read`}]:[],u=[Z(`read`),{name:`neighborIds`,view:i.neighborIds,type:`u32`,access:`read_write`},{name:`neighborCounts`,view:i.neighborCounts,type:`u32`,access:`read_write`},{name:`overflow`,view:i.overflow,type:`atomic<u32>`,access:`read_write`}];i.neighborRows&&u.push({name:`neighborRows`,view:i.neighborRows,type:`u32`,access:`read_write`}),i.featureIds&&!i.writeFeatureRows&&u.push({name:`featureIds`,view:i.featureIds,type:`u32`,access:`read`}),i.neighborDistances&&u.push({name:`neighborDistances`,view:i.neighborDistances,type:`f32`,access:`read_write`});let d=a?`let query = index / CAPACITY;
  let slot = index % CAPACITY;
  let position = topk[topkOffset + CURSOR_WORD] + query;
  if (position >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = ${c?`queryOrder[queryOrderOffset + position] - CHUNK_FIRST_ROW`:`position`};
  let row = CHUNK_FIRST_ROW + localQuery;
  let outputIndex = row * CAPACITY + slot;`:`let query = index / CAPACITY;
  let slot = index % CAPACITY;
  let row = query;
  let outputIndex = index;`;if(I.push(e(n,{id:`${t}-columns`,operation:s,variant:`${j?`columns-great-circle`:`columns`}${c?`-ordered`:``}`,bindings:[...u,...l],invocationCount:r*S,declarations:`${h}
const CAPACITY: u32 = ${S}u;
const STRIDE: u32 = ${T}u;
const NO_FEATURE: u32 = 0xffffffffu;${j?`\nconst SPHERE_RADIUS: f32 = ${F};`:``}${a?`\nconst CURSOR_WORD: u32 = ${k}u;
`:``}`,body:`let CHUNK_FIRST_ROW = topk[topkOffset + ${A}u];
  let CHUNK_QUERY_COUNT = topk[topkOffset + ${A+1}u];
  ${d}
  let header = topkOffset + query * STRIDE;
  let count = topk[header];
  if (slot == 0u) {
    neighborCounts[neighborCountsOffset + row] = count;
    if (topk[header + 1u] != 0u) { atomicOr(&overflow[overflowOffset], 1u); }
  }
  var featureRow = NO_FEATURE;
  var featureId = NO_FEATURE;
  var distance = -1.0;
  if (slot < count) {
    featureRow = topk[header + 3u + slot * 2u];
    featureId = featureRow;
    ${i.featureIds&&!i.writeFeatureRows?`featureId = featureIds[featureIdsOffset + featureId];`:``}
    distance = ${j?`bitcast<f32>(topk[header + 4u + slot * 2u]) * SPHERE_RADIUS`:`decodeNearestDistance(bitcast<f32>(topk[header + 4u + slot * 2u]))`};
  }
  neighborIds[neighborIdsOffset + outputIndex] = featureId;
  ${i.neighborRows?`neighborRows[neighborRowsOffset + outputIndex] = featureRow;`:``}
  ${i.neighborDistances?`neighborDistances[neighborDistancesOffset + outputIndex] = distance;`:``}`})),Q){let o=[Z(`read`),{name:`extras`,view:Q,type:`u32`,access:`read`}];i.neighborFootPoints&&o.push({name:`footPoints`,view:i.neighborFootPoints,type:`u32`,access:`read_write`}),i.neighborQueryPoints&&o.push({name:`queryPoints`,view:i.neighborQueryPoints,type:`u32`,access:`read_write`}),i.neighborSegmentIndices&&o.push({name:`segmentIndices`,view:i.neighborSegmentIndices,type:`u32`,access:`read_write`}),I.push(e(n,{id:`${t}-foot-columns`,operation:s,variant:c?`foot-columns-ordered`:`foot-columns`,bindings:[...o,...l],invocationCount:r*S,declarations:`const CAPACITY: u32 = ${S}u;
const STRIDE: u32 = ${T}u;
const NO_SEGMENT: u32 = 0xffffffffu;${a?`\nconst CURSOR_WORD: u32 = ${k}u;
`:``}`,body:`let CHUNK_FIRST_ROW = topk[topkOffset + ${A}u];
  let CHUNK_QUERY_COUNT = topk[topkOffset + ${A+1}u];
  ${d}
  let valid = slot < topk[topkOffset + query * STRIDE];
  let source = extrasOffset + index * ${w}u;
  ${i.neighborFootPoints?`// Unused slots hold a quiet NaN so a missing neighbor is never mistaken for the origin.
  footPoints[footPointsOffset + outputIndex * 2u] = select(0x7fc00000u, extras[source], valid);
  footPoints[footPointsOffset + outputIndex * 2u + 1u] = select(0x7fc00000u, extras[source + 1u], valid);`:``}
  ${i.neighborQueryPoints?`queryPoints[queryPointsOffset + outputIndex * 2u] = select(0x7fc00000u, extras[source + 3u], valid);
  queryPoints[queryPointsOffset + outputIndex * 2u + 1u] = select(0x7fc00000u, extras[source + 4u], valid);`:``}
  ${i.neighborSegmentIndices?`segmentIndices[segmentIndicesOffset + outputIndex] = select(NO_SEGMENT, extras[source + 2u], valid);`:``}`}))}},le=(t,r,a,o)=>{let c=t.order?[{name:`queryOrder`,view:t.order,type:`u32`,access:`read`}]:[],u=g({ties:i.ties,k:x,capacity:S,featureCount:_,internalNodeCount:L.internalNodeCount,stackSize:se,dispatchWord:A,scratchChunk:o?{cursorWord:k}:void 0,queryOrder:!!t.order,queryKind:t.kind,greatCircle:j,filter:R?{exclusive:!!i.exclusive,onAttribute:!!i.onAttribute,queryBase:z,featureBase:B}:void 0}),h=`${j?f:p}${m(t.side)}${m(X)}${d(t.side,X)}`;I.push(e(n,{id:`${r}-traverse`,operation:s,variant:`traverse-${t.kind}-${l.kind}${j?`-great-circle`:``}${t.order?`-ordered`:``}`,bindings:[...t.bindings,oe,...ae,Z(`read_write`),...c],invocationCount:a,declarations:`${h}${u.declarations}`,body:u.body})),Q&&I.push(e(n,{id:`${r}-foot-points`,operation:s,variant:`foot-${t.kind}-${l.kind}${j?`-great-circle`:``}${o&&t.order?`-ordered`:``}`,bindings:[...t.bindings,...ae,Z(`read`),{name:`extras`,view:Q,type:`u32`,access:`read_write`},...o?c:[]],invocationCount:a,declarations:`${h}
const CAPACITY: u32 = ${S}u;
const STRIDE: u32 = ${T}u;${o?`\nconst CURSOR_WORD: u32 = ${k}u;
`:``}`,body:`let CHUNK_FIRST_ROW = topk[topkOffset + ${A}u];
  let CHUNK_QUERY_COUNT = topk[topkOffset + ${A+1}u];
  ${o?`let position = topk[topkOffset + CURSOR_WORD] + index;
  if (position >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = ${t.order?`queryOrder[queryOrderOffset + position] - CHUNK_FIRST_ROW`:`position`};
  let row = index;`:`if (index >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = index;
  let row = CHUNK_FIRST_ROW + localQuery;`}
  let header = topkOffset + row * STRIDE;
  let count = topk[header];
  for (var slot = 0u; slot < count; slot++) {
    let result = pairResult(localQuery, topk[header + 3u + slot * 2u]);
    let output = extrasOffset + (row * CAPACITY + slot) * ${w}u;
    extras[output] = bitcast<u32>(result.foot.x);
    extras[output + 1u] = bitcast<u32>(result.foot.y);
    extras[output + 2u] = result.segment;
    extras[output + 3u] = bitcast<u32>(result.queryFoot.x);
    extras[output + 4u] = bitcast<u32>(result.queryFoot.y);
  }`}))};I.push(e(n,{id:`${o}-initialize-dispatch`,operation:s,variant:`initialize-dispatch`,bindings:[Z(`read_write`)],invocationCount:1,body:`topk[topkOffset + ${A}u] = 0u;
  topk[topkOffset + ${A+1}u] = 0u;`}));let $=t=>{let r=i.queries.kind===`points`?t.bindings[0].view:i.queries.geometry.kind===`lines`?i.queries.geometry.lineOffsets:i.queries.geometry.featureOffsets,a=i.queries.kind===`points`?2:1,o=i.queries.kind===`points`?0:1;I.push(e(n,{id:`${t.chunkId}-dispatch`,operation:s,variant:`dispatch-parameters`,bindings:[Z(`read_write`),{name:`queryShape`,view:r,type:`u32`,access:`read`}],invocationCount:1,body:`topk[topkOffset + ${A}u] += topk[topkOffset + ${A+1}u];
  topk[topkOffset + ${A+1}u] = (arrayLength(&queryShape) - queryShapeOffset) / ${a}u - ${o}u;`}))};if(!O){for(let e of Y)$(e),le(e,e.chunkId,e.count);return b>0&&ce(o,b),{nodes:I,bvh:L}}for(let t of Y){$(t);let r=Math.ceil(t.count/D),i={firstRow:t.firstRow,queryCount:t.count};I.push(e(n,{id:`${t.chunkId}-reset-cursor`,operation:s,variant:`reset-cursor`,bindings:[Z(`read_write`)],invocationCount:1,declarations:`const CURSOR_WORD: u32 = ${k}u;`,body:`topk[topkOffset + CURSOR_WORD] = 0u;`}));for(let a=0;a<r;a++){let o=Math.min(D,t.count-a*D),c=`${t.chunkId}-scratch-${a}`;I.push(e(n,{id:`${c}-clear`,operation:s,variant:j?`clear-chunk-great-circle`:`clear-chunk`,bindings:K.filter(e=>e.name===`topk`||e.name===`maxDistance`),invocationCount:o,declarations:`${J}
const CURSOR_WORD: u32 = ${k}u;
`,body:`let CHUNK_QUERY_COUNT = topk[topkOffset + ${A+1}u];
  if (topk[topkOffset + CURSOR_WORD] + index < CHUNK_QUERY_COUNT) {
    let header = topkOffset + index * STRIDE;
    ${q}
  }`})),le(t,c,o,i),ce(c,o,i,t.order),a<r-1&&I.push(e(n,{id:`${c}-advance-cursor`,operation:s,variant:`advance-cursor`,bindings:[Z(`read_write`)],invocationCount:1,declarations:`const CURSOR_WORD: u32 = ${k}u;
const SCRATCH_QUERY_COUNT: u32 = ${D}u;`,body:`topk[topkOffset + CURSOR_WORD] = topk[topkOffset + CURSOR_WORD] + SCRATCH_QUERY_COUNT;`}))}}return{nodes:I,bvh:L}}export{F as a,j as i,k as n,ne as o,A as r,te as t};
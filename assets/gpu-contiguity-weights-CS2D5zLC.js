import{t as e}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as t,t as n}from"./weights-construction-snapshot-CikLt8kl.js";import{r}from"./scan-total-C4QddMmi.js";import{r as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./gpu-scan-B4NNTfF9.js";import{o}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as s}from"./sorted-segment-sums-Dvmm6OdR.js";import{E as c,S as l,T as u,f as d,g as f,x as p,y as m}from"./gpu-contributor-utils-BIwLQON5.js";import{i as h,n as g,r as _,t as v}from"./key-pair-grouping-DgpiCtBP.js";import{a as y,i as b,l as x,n as S,t as C}from"./spatial-weights-CqWysstN.js";import{l as w}from"./wgsl-common-DfyXiYB0.js";import{r as T}from"./coverage-topology-kernels-ltNrCaNR.js";var E=`GPUContiguityWeights`,D=`${v}u`,O=2**24,k=4,A=4,j=2,M=4,N=class{id;props;outputWeights;constructor(e){e=n(e);let r=e.id??`contiguity-weights`;if(this.id=r,this.props=e,e.criterion!==`queen`&&e.criterion!==`rook`)throw Error(`${r} criterion must be 'queen' or 'rook'`);let{polygons:a}=e;c(a.positions,[`float32x2`],`${r} polygons.positions`),d(r,`polygons.positions length`,a.positions.length,1,2**31-1);for(let e of[`ringOffsets`,`polygonOffsets`,`featureOffsets`])c(a[e],[`uint32`],`${r} polygons.${e}`),d(r,`polygons.${e} length`,a[e].length,2);if(e.snapTolerance!==void 0&&(!Number.isFinite(e.snapTolerance)||e.snapTolerance<0))throw Error(`${r} snapTolerance must be finite and >= 0`);e.snapTolerance!==void 0&&(f(r,`snapTolerance (float32 representability)`,Math.fround(e.snapTolerance),0),u(r,e.snapTolerance===0||Math.fround(e.snapTolerance)>0,`snapTolerance must be zero or a positive finite float32 value`)),e.coordinateExtent!==void 0&&(f(r,`coordinateExtent`,e.coordinateExtent,0),e.snapTolerance&&l(r,`coordinateExtent / snapTolerance`,e.coordinateExtent/e.snapTolerance,O));let o=x(r,e.output);if(o!==a.featureOffsets.length-1)throw Error(`${r} output rows must equal the feature count`);e.pairCapacity!==void 0&&d(r,`pairCapacity`,e.pairCapacity),this.outputWeights=C(e.output,{information:{rowCount:o,transform:`binary`,symmetric:!0,hasSelfWeights:!1,provenance:{source:E,criterion:e.criterion}},status:e.output.status}),m(r,[...y(e.output),...b(e.output)],[a.positions,a.ringOffsets,a.polygonOffsets,a.featureOffsets]),this.outputWeights={...this.outputWeights,status:{...e.output.status,candidateOverflow:e.output.status.candidateOverflow??i(e.output.offsets.buffer.graph,`${r}-${e.output.offsets.buffer.id}-${e.output.offsets.byteOffset}-candidate-overflow`,`uint32`,1)}},this.outputWeights=t(this.outputWeights)}getFeatureCount(){return this.props.polygons.featureOffsets.length-1}getPairCapacity(){return this.props.pairCapacity??Math.max(1,this.props.output.neighbors.length*4)}getCapacityPlan(e={}){let{props:t}=this,n=t.output.neighbors.length,r=S({rowCount:this.getFeatureCount(),capacity:n,requiredCount:e.requiredCount,distances:!!t.output.distances}),i=this.getPairCapacity(),a=t.pairCapacity===void 0?r.capacity*4:i,o=e.candidateOverflow?Math.max(a,i*2):a,s=t.polygons.positions.length,c=P(s,t.criterion),l=F(o);return{slots:r,pairs:{capacity:o,estimatedRequiredCount:o,limited:!1,estimatedOutputBytes:0,estimatedPeakTransientBytes:l*Uint32Array.BYTES_PER_ELEMENT},estimatedWorkItems:s+o+this.getFeatureCount()+1,estimatedOutputBytes:r.estimatedOutputBytes,estimatedPeakTransientBytes:(c+l)*Uint32Array.BYTES_PER_ELEMENT}}getCommandNodes(t){e(t,this.id,this.props,{"vertex scratch":this.props.polygons.positions.length*4,"pair scratch":this.getPairCapacity()*4});let{id:n,props:c}=this,{output:l}=c,{ringOffsets:u,polygonOffsets:d,featureOffsets:f}=c.polygons,m=c.polygons.positions,{overflow:_,requiredCount:v}=l.status,x=this.outputWeights.status?.candidateOverflow;p(n,t,[m,u,d,f,...y(l),...b(l)]);let S=m.length,C=this.getFeatureCount(),O=d.length-1,k=u.length-1,A=l.neighbors.length,j=this.getPairCapacity(),M=c.snapTolerance??0,N=i(t,`${n}-vertex-high`,`uint32`,S),P=i(t,`${n}-vertex-low`,`uint32`,S),F=i(t,`${n}-vertex-feature`,`uint32`,S),I=i(t,`${n}-vertex-next`,`uint32`,S),L=[o(t,{id:`${n}-vertex-topology`,operation:E,variant:`vertex-topology`,bindings:[{name:`positions`,view:m,type:`f32`,access:`read`},{name:`ringOffsets`,view:u,type:`u32`,access:`read`},{name:`polygonOffsets`,view:d,type:`u32`,access:`read`},{name:`featureOffsets`,view:f,type:`u32`,access:`read`},{name:`vertexHigh`,view:N,type:`u32`,access:`read_write`},{name:`vertexLow`,view:P,type:`u32`,access:`read_write`},{name:`vertexPolygon`,view:F,type:`u32`,access:`read_write`},{name:`vertexNext`,view:I,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`const RING_COUNT: u32 = ${k}u;
const POLYGON_COUNT: u32 = ${O}u;
const FEATURE_COUNT: u32 = ${C}u;

${w}
// The snap key shared with the coverage operators: getSnapOrigin() (the first ring-owned vertex,
// 0 when it is not finite) and quantise(value, origin). Keys are formed from the offset to
// the origin, so they depend on the extent of the data and not on its distance from the
// coordinate origin (an absolute key x / tolerance passes 2^31 for coordinates in the millions
// with a millimetre tolerance), and from 2^23 steps the integer quotient is used as it is,
// because adding 0.5 there rounds an odd step to the next even one.
${T(M,k)}`,body:`var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  var polygon = 0u;
  high = POLYGON_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= ring) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  var feature = 0u;
  high = FEATURE_COUNT;
  while (feature + 1u < high) {
    let middle = (feature + high) / 2u;
    if (featureOffsets[featureOffsetsOffset + middle] <= polygon) {
      feature = middle;
    } else {
      high = middle;
    }
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let valid = index >= ringBegin && index < ringEnd &&
    ring >= polygonOffsets[polygonOffsetsOffset] && ring < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT] &&
    ring < polygonOffsets[polygonOffsetsOffset + polygon + 1u] &&
    polygon >= featureOffsets[featureOffsetsOffset] && polygon < featureOffsets[featureOffsetsOffset + FEATURE_COUNT] &&
    polygon < featureOffsets[featureOffsetsOffset + feature + 1u] &&
    isFiniteFloat(x) && isFiniteFloat(y);
  let origin = getSnapOrigin();
  vertexHigh[vertexHighOffset + index] = select(${D}, quantise(x, origin.x), valid);
  vertexLow[vertexLowOffset + index] = select(${D}, quantise(y, origin.y), valid);
  vertexPolygon[vertexPolygonOffset + index] = select(${D}, feature, valid);
  vertexNext[vertexNextOffset + index] = select(index + 1u, ringBegin, index + 1u == ringEnd);`})],R=h(t,`${n}-vertex`,E,S,N,P),z=g(t,`${n}-vertex`,E,S,N,P,R.sortedItems);L.push(...R.nodes,...z.nodes);let B=F,V=R.sortedItems,H=z;if(c.criterion===`rook`){let e=i(t,`${n}-point-ids`,`uint32`,S),r=i(t,`${n}-edge-high`,`uint32`,S),a=i(t,`${n}-edge-low`,`uint32`,S);B=i(t,`${n}-edge-polygon`,`uint32`,S),L.push(o(t,{id:`${n}-point-ids`,operation:E,variant:`point-ids`,bindings:[{name:`sortedItems`,view:R.sortedItems,type:`u32`,access:`read`},{name:`groupIndex`,view:z.groupIndex,type:`u32`,access:`read`},{name:`vertexPolygon`,view:F,type:`u32`,access:`read`},{name:`pointIds`,view:e,type:`u32`,access:`read_write`}],invocationCount:S,body:`let vertex = sortedItems[sortedItemsOffset + index];
  pointIds[pointIdsOffset + vertex] = select(${D},
    groupIndex[groupIndexOffset + index] - 1u, vertexPolygon[vertexPolygonOffset + vertex] != ${D});`}),o(t,{id:`${n}-edge-keys`,operation:E,variant:`edge-keys`,bindings:[{name:`vertexPolygon`,view:F,type:`u32`,access:`read`},{name:`vertexNext`,view:I,type:`u32`,access:`read`},{name:`pointIds`,view:e,type:`u32`,access:`read`},{name:`edgeHigh`,view:r,type:`u32`,access:`read_write`},{name:`edgeLow`,view:a,type:`u32`,access:`read_write`},{name:`edgePolygon`,view:B,type:`u32`,access:`read_write`}],invocationCount:S,body:`let first = pointIds[pointIdsOffset + index];
  let second = pointIds[pointIdsOffset + vertexNext[vertexNextOffset + index]];
  let valid = first != ${D} && second != ${D} && first != second;
  edgeHigh[edgeHighOffset + index] = select(${D}, min(first, second), valid);
  edgeLow[edgeLowOffset + index] = select(${D}, max(first, second), valid);
  edgePolygon[edgePolygonOffset + index] = select(${D}, vertexPolygon[vertexPolygonOffset + index], valid);`}));let c=h(t,`${n}-edge`,E,S,r,a,{high:s(S),low:s(S)});H=g(t,`${n}-edge`,E,S,r,a,c.sortedItems),V=c.sortedItems,L.push(...c.nodes,...H.nodes)}let U=i(t,`${n}-pair-counts`,`uint32`,S),W=i(t,`${n}-pair-starts`,`uint32`,S),G=i(t,`${n}-pair-total`,`uint32`,1),K=i(t,`${n}-pair-high`,`uint32`,j),q=i(t,`${n}-pair-low`,`uint32`,j),J=[{name:`sortedItems`,view:V,type:`u32`,access:`read`},{name:`itemPolygon`,view:B,type:`u32`,access:`read`},{name:`groupIndex`,view:H.groupIndex,type:`u32`,access:`read`},{name:`groupStarts`,view:H.groupStarts,type:`u32`,access:`read`}],Y=e=>`let item = sortedItems[sortedItemsOffset + index];
  let polygon = itemPolygon[itemPolygonOffset + item];
  if (polygon != ${D}) {
    let groupId = groupIndex[groupIndexOffset + index] - 1u;
    let begin = groupStarts[groupStartsOffset + groupId];
    let end = groupStarts[groupStartsOffset + groupId + 1u];
    var firstPolygonOccurrence = index == begin;
    if (!firstPolygonOccurrence) {
      let previousItem = sortedItems[sortedItemsOffset + index - 1u];
      firstPolygonOccurrence = itemPolygon[itemPolygonOffset + previousItem] != polygon;
    }
    if (firstPolygonOccurrence) {
      var previousOtherPolygon = ${D};
      for (var other = begin; other < end; other++) {
        let otherPolygon = itemPolygon[itemPolygonOffset + sortedItems[sortedItemsOffset + other]];
        let firstOtherPolygonOccurrence = otherPolygon != previousOtherPolygon;
        previousOtherPolygon = otherPolygon;
        if (firstOtherPolygonOccurrence && otherPolygon != ${D} && otherPolygon != polygon) {
          ${e}
        }
      }
    }
  }`;L.push(o(t,{id:`${n}-pair-counts`,operation:E,variant:`pair-counts`,bindings:[...J,{name:`pairCounts`,view:U,type:`u32`,access:`read_write`}],invocationCount:S,body:`var count = 0u;
  ${Y(`count++;`)}
  pairCounts[pairCountsOffset + index] = count;`}),...new a({id:`${n}-pair-scan`,input:U,output:W,mode:`exclusive`}).getCommandNodes(t),...r(t,{id:`${n}-pair`,operation:E,counts:U,offsets:W,rowCount:S,total:G}),o(t,{id:`${n}-pair-fill`,operation:E,variant:`pair-fill`,bindings:[{name:`pairHigh`,view:K,type:`u32`,access:`read_write`},{name:`pairLow`,view:q,type:`u32`,access:`read_write`}],invocationCount:j,body:`pairHigh[pairHighOffset + index] = ${D};
  pairLow[pairLowOffset + index] = ${D};`}),o(t,{id:`${n}-pair-emit`,operation:E,variant:`pair-emit`,bindings:[...J,{name:`pairStarts`,view:W,type:`u32`,access:`read`},{name:`pairHigh`,view:K,type:`u32`,access:`read_write`},{name:`pairLow`,view:q,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`const PAIR_CAPACITY: u32 = ${j}u;`,body:`var next = pairStarts[pairStartsOffset + index];
  ${Y(`if (next < PAIR_CAPACITY) {
          pairHigh[pairHighOffset + next] = polygon;
          pairLow[pairLowOffset + next] = otherPolygon;
        }
        next++;`)}`}));let X=h(t,`${n}-pair`,E,j,K,q,{high:s(C),low:s(C)}),Z=i(t,`${n}-unique`,`uint32`,j),Q=i(t,`${n}-unique-rank`,`uint32`,j);return L.push(...X.nodes,o(t,{id:`${n}-unique-flags`,operation:E,variant:`unique-flags`,bindings:[{name:`sortedPairs`,view:X.sortedItems,type:`u32`,access:`read`},{name:`pairHigh`,view:K,type:`u32`,access:`read`},{name:`pairLow`,view:q,type:`u32`,access:`read`},{name:`isUnique`,view:Z,type:`u32`,access:`read_write`}],invocationCount:j,body:`let item = sortedPairs[sortedPairsOffset + index];
  let high = pairHigh[pairHighOffset + item];
  let low = pairLow[pairLowOffset + item];
  var first = high != ${D};
  if (first && index > 0u) {
    let previous = sortedPairs[sortedPairsOffset + index - 1u];
    first = high != pairHigh[pairHighOffset + previous] || low != pairLow[pairLowOffset + previous];
  }
  isUnique[isUniqueOffset + index] = select(0u, 1u, first);`}),...new a({id:`${n}-unique-scan`,input:Z,output:Q,mode:`exclusive`}).getCommandNodes(t),o(t,{id:`${n}-offsets`,operation:E,variant:`offsets`,bindings:[{name:`sortedPairs`,view:X.sortedItems,type:`u32`,access:`read`},{name:`pairHigh`,view:K,type:`u32`,access:`read`},{name:`isUnique`,view:Z,type:`u32`,access:`read`},{name:`uniqueRank`,view:Q,type:`u32`,access:`read`},{name:`offsets`,view:l.offsets,type:`u32`,access:`read_write`}],invocationCount:C+1,declarations:`const CAPACITY: u32 = ${A}u;
const PAIR_CAPACITY: u32 = ${j}u;`,body:`var low = 0u;
  var high = PAIR_CAPACITY;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pairHigh[pairHighOffset + sortedPairs[sortedPairsOffset + middle]] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  var start = 0u;
  if (low < PAIR_CAPACITY) {
    start = uniqueRank[uniqueRankOffset + low];
  } else {
    start = uniqueRank[uniqueRankOffset + PAIR_CAPACITY - 1u] + isUnique[isUniqueOffset + PAIR_CAPACITY - 1u];
  }
  offsets[offsetsOffset + index] = min(start, CAPACITY);`}),o(t,{id:`${n}-totals`,operation:E,variant:`totals`,bindings:[{name:`isUnique`,view:Z,type:`u32`,access:`read`},{name:`uniqueRank`,view:Q,type:`u32`,access:`read`},{name:`scanTotal`,view:G,type:`u32`,access:`read`},{name:`overflow`,view:_,type:`u32`,access:`read_write`},...v?[{name:`requiredCount`,view:v,type:`u32`,access:`read_write`}]:[],...x?[{name:`candidateOverflow`,view:x,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`const CAPACITY: u32 = ${A}u;
const PAIR_CAPACITY: u32 = ${j}u;
const VERTEX_COUNT: u32 = ${S}u;`,body:`let total = uniqueRank[uniqueRankOffset + PAIR_CAPACITY - 1u] + isUnique[isUniqueOffset + PAIR_CAPACITY - 1u];
  let pairTotal = scanTotal[scanTotalOffset];
  overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
  ${v?`requiredCount[requiredCountOffset] = total;`:``}
  ${x?`candidateOverflow[candidateOverflowOffset] = select(0u, 1u, pairTotal > PAIR_CAPACITY);`:``}`}),o(t,{id:`${n}-emit`,operation:E,variant:`emit`,bindings:[{name:`sortedPairs`,view:X.sortedItems,type:`u32`,access:`read`},{name:`pairLow`,view:q,type:`u32`,access:`read`},{name:`isUnique`,view:Z,type:`u32`,access:`read`},{name:`uniqueRank`,view:Q,type:`u32`,access:`read`},{name:`neighbors`,view:l.neighbors,type:`u32`,access:`read_write`},{name:`weights`,view:l.weights,type:`f32`,access:`read_write`}],invocationCount:j,declarations:`const CAPACITY: u32 = ${A}u;`,body:`let rank = uniqueRank[uniqueRankOffset + index];
  if (isUnique[isUniqueOffset + index] != 0u && rank < CAPACITY) {
    neighbors[neighborsOffset + rank] = pairLow[pairLowOffset + sortedPairs[sortedPairsOffset + index]];
    weights[weightsOffset + rank] = 1.0;
  }`})),L}};function P(e,t){let n=6*e+_(e),r=t===`rook`?A*e+n:0;return k*e+n+r+j*e}function F(e){return(M+6)*e+1}export{N as t};
import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-scan-B4NNTfF9.js";import{t as n}from"./gpu-sort-xRm8l5yn.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as i,S as a,f as o}from"./gpu-contributor-utils-BIwLQON5.js";import{i as s,r as c}from"./segment-table-BeVxjLqW.js";import{t as l}from"./gpu-segment-intersection-CGBfhMHr.js";import{t as u}from"./ports-xmdKgL07.js";import{_ as d,v as f}from"./spatial-context-DwZEd4EZ.js";import{a as p}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as m,t as h}from"./geometry-storage-validation-T89ZhmCj.js";import{n as g}from"./segment-storage-validation-BUxpxgMF.js";import{A as _}from"./pointer-overlay-Qh8qflcA.js";import{t as v}from"./define-run-B-brkCe4.js";import{n as y,t as b}from"./pick-Bo3t3hjv.js";import{l as x,o as S,s as C}from"./polygon-set-CHja50mr.js";import{o as w}from"./pointer-support-C2rPRVFO.js";import{a as T}from"./shape-segments-DMcyF4V-.js";import{vo as E}from"./entry-preview-D70TdSbK.js";function D(e){return e.kind===`points`?[e.positions]:e.kind===`lines`?[e.positions,e.lineOffsets]:[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets]}function O(e){return e.kind===`points`?e.positions.length:e.kind===`lines`?e.lineOffsets.length-1:e.featureOffsets.length-1}function k(e,t,n=!0){let r=n?[{name:`${e}Positions`,view:t.positions,type:`f32`,access:`read`}]:[];return t.kind===`lines`?r.push({name:`${e}LineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`}):t.kind===`polygons`&&r.push({name:`${e}FeatureOffsets`,view:t.featureOffsets,type:`u32`,access:`read`},{name:`${e}PolygonOffsets`,view:t.polygonOffsets,type:`u32`,access:`read`},{name:`${e}RingOffsets`,view:t.ringOffsets,type:`u32`,access:`read`}),r}function A(e,t){return`fn ${e}(count: u32, value: u32) -> u32 {
  if (count == 0u || ${t}[${t}Offset] > value) { return 0u; }
  var low = 0u;
  var high = count;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (${t}[${t}Offset + middle] <= value) { low = middle; } else { high = middle; }
  }
  return low;
}`}function j(e,t,n=!0){let r=t.positions.length,i=`const ${e}_FEATURE_COUNT: u32 = ${O(t)}u;
const ${e}_VERTEX_COUNT: u32 = ${r}u;
${n?`fn ${e}Vertex(vertex: u32) -> vec2f {
  return vec2f(${e}Positions[${e}PositionsOffset + vertex * 2u], ${e}Positions[${e}PositionsOffset + vertex * 2u + 1u]);
}`:``}`;return t.kind===`points`?`${i}
fn ${e}VertexStart(feature: u32) -> u32 { return min(feature, ${e}_VERTEX_COUNT); }
fn ${e}FeatureOfVertex(vertex: u32) -> u32 { return vertex; }`:t.kind===`lines`?`${i}
fn ${e}VertexStart(feature: u32) -> u32 {
  return min(${e}LineOffsets[${e}LineOffsetsOffset + feature], ${e}_VERTEX_COUNT);
}
${A(`${e}LastLine`,`${e}LineOffsets`)}
fn ${e}FeatureOfVertex(vertex: u32) -> u32 { return ${e}LastLine(${e}_FEATURE_COUNT, vertex); }`:`${i}
const ${e}_POLYGON_COUNT: u32 = ${t.polygonOffsets.length-1}u;
const ${e}_RING_COUNT: u32 = ${t.ringOffsets.length-1}u;
fn ${e}VertexStart(feature: u32) -> u32 {
  let polygon = ${e}FeatureOffsets[${e}FeatureOffsetsOffset + feature];
  let ring = ${e}PolygonOffsets[${e}PolygonOffsetsOffset + polygon];
  return min(${e}RingOffsets[${e}RingOffsetsOffset + ring], ${e}_VERTEX_COUNT);
}
${A(`${e}LastRing`,`${e}RingOffsets`)}
${A(`${e}LastPolygon`,`${e}PolygonOffsets`)}
${A(`${e}LastFeature`,`${e}FeatureOffsets`)}
fn ${e}FeatureOfVertex(vertex: u32) -> u32 {
  let ring = ${e}LastRing(${e}_RING_COUNT, vertex);
  let polygon = ${e}LastPolygon(${e}_POLYGON_COUNT, ring);
  return ${e}LastFeature(${e}_FEATURE_COUNT, polygon);
}`}var M=`GPUGeometryPredicates`,N=4294967295,P=class{id;props;spatialContext;featureCount;vertexCount;constructor(e){if(this.id=e.id??`geometry-predicates`,this.spatialContext=f(this.id,e.spatialContext),e.otherSpatialContext!==void 0&&!e.other)throw Error(`${this.id} otherSpatialContext needs other`);d(this.id,`other`,e.otherSpatialContext,this.spatialContext),e={...e,geometry:u(this.id,`geometry`,e.geometry),other:e.other&&u(this.id,`other`,e.other)},this.props=e;let{id:t}=this,{geometry:n}=e;if(this.featureCount=O(n),this.vertexCount=n.positions.length,this.validateGeometry(`geometry`,n),this.featureCount<1)throw Error(`${t} geometry must contain at least one feature`);if(this.vertexCount<1)throw Error(`${t} geometry must contain at least one vertex`);for(let[n,r]of[[`isSimple`,e.output.isSimple],[`isRing`,e.output.isRing],[`isClosed`,e.output.isClosed],[`isCcw`,e.output.isCcw],[`numPoints`,e.output.numPoints],[`numCoordinates`,e.output.numCoordinates],[`numInteriorRings`,e.output.numInteriorRings],[`numGeometries`,e.output.numGeometries],[`equalsExact`,e.output.equalsExact],[`equalsIdentical`,e.output.equalsIdentical]])if(r&&(i(r,[`uint32`],`${t} ${n}`),r.length!==this.featureCount))throw Error(`${t} ${n} length must equal the feature count`);if(e.output.isSimple||e.output.isRing){if(!e.output.status?.overflow||!e.intersectionCapacity)throw Error(`${t} isSimple and isRing require status.overflow and intersectionCapacity`);o(t,`intersectionCapacity`,e.intersectionCapacity)}for(let[n,r]of[[`overflow`,e.output.status?.overflow],[`intersectionCount`,e.output.status?.intersectionCount]])if(r&&(i(r,[`uint32`],`${t} ${n}`),r.length<1))throw Error(`${t} ${n} must contain one uint32 row`);if(e.output.equalsExact||e.output.equalsIdentical){let{other:r}=e;if(!r)throw Error(`${t} equalsExact and equalsIdentical require other`);if(this.validateGeometry(`other`,r),r.kind!==n.kind||O(r)!==this.featureCount)throw Error(`${t} other must have the kind and feature count of geometry`)}if(e.output.equalsExact){if(!e.parameters)throw Error(`${t} equalsExact requires parameters`);if(i(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 elements`)}if(e.output.uniquePositions||e.output.uniqueOffsets){if(!e.output.uniquePositions||!e.output.uniqueOffsets)throw Error(`${t} uniquePositions and uniqueOffsets must be given together`);if(i(e.output.uniquePositions,[`float32x2`],`${t} uniquePositions`),i(e.output.uniqueOffsets,[`uint32`],`${t} uniqueOffsets`),e.output.uniquePositions.length<this.vertexCount)throw Error(`${t} uniquePositions must hold at least the vertex count`);if(e.output.uniqueOffsets.length!==this.featureCount+1)throw Error(`${t} uniqueOffsets must have featureCount + 1 rows`)}}validateGeometry(e,t){let{id:n}=this;i(t.positions,[`float32x2`],`${n} ${e}.positions`);for(let r of D(t).slice(1))if(i(r,[`uint32`],`${n} ${e} offsets`),r.length<1)throw Error(`${n} ${e} offsets require a terminal entry`)}getViews(){let{props:e}=this;return[...D(e.geometry),...(e.output.equalsExact||e.output.equalsIdentical)&&e.other?D(e.other):[],e.output.isSimple,e.output.isRing,e.output.isClosed,e.output.isCcw,e.output.numPoints,e.output.numCoordinates,e.output.numInteriorRings,e.output.numGeometries,...e.geometry.kind!==`points`&&(e.output.isSimple||e.output.isRing)?[e.output.status?.overflow,e.output.status?.intersectionCount]:[],e.output.equalsExact?e.parameters:void 0,e.output.equalsExact,e.output.equalsIdentical,e.output.uniquePositions,e.output.uniqueOffsets]}getCommandNodes(t){let{id:n,props:i,featureCount:o}=this;p(t,n,Object.fromEntries(this.getViews().map((e,t)=>[String(t),e]))),(i.output.isSimple||i.output.isRing)&&a(n,`geometry.positions rows`,this.vertexCount,s(t.device)),(i.output.isSimple||i.output.isRing)&&i.geometry.kind!==`points`&&g(t,n,this.vertexCount,this.vertexCount,i.intersectionCapacity,i.leafCapacity),m(t,n,{featureColumns:o*(i.output.equalsExact||i.output.equalsIdentical?12:4),...i.output.uniquePositions?h(this.vertexCount):{}});let{geometry:l}=i,u=[],d=k(`g`,l),f=`${F}\n${j(`g`,l)}`,_=l.kind===`lines`,v=(e,i,a,s={})=>{u.push(r(t,{id:`${n}-${e}`,operation:M,variant:e,bindings:[...d,...s.bindings??[],{name:`output`,view:i,type:`u32`,access:`read_write`}],invocationCount:o,declarations:`${f}\n${s.declarations??``}`,body:a}))},y=`let start = gVertexStart(index);
  let end = gVertexStart(index + 1u);`,b=e=>`output[outputOffset + index] = ${e};`;if(i.output.isClosed&&v(`is-closed`,i.output.isClosed,_?`${y}\n  ${b(I)}`:b(`0u`)),i.output.numPoints&&v(`num-points`,i.output.numPoints,_?`${y}\n  ${b(`end - start`)}`:b(`0u`)),i.output.numCoordinates&&v(`num-coordinates`,i.output.numCoordinates,`${y}\n  ${b(`end - start`)}`),i.output.numInteriorRings&&v(`num-interior-rings`,i.output.numInteriorRings,l.kind===`polygons`?`let firstPolygon = gFeatureOffsets[gFeatureOffsetsOffset + index];
  let polygonCount = gFeatureOffsets[gFeatureOffsetsOffset + index + 1u] - firstPolygon;
  var holes = 0u;
  if (polygonCount == 1u) {
    let ringCount = gPolygonOffsets[gPolygonOffsetsOffset + firstPolygon + 1u] - gPolygonOffsets[gPolygonOffsetsOffset + firstPolygon];
    holes = select(0u, ringCount - 1u, ringCount > 0u);
  }
  ${b(`holes`)}`:b(`0u`)),i.output.numGeometries&&v(`num-geometries`,i.output.numGeometries,l.kind===`polygons`?b(`gFeatureOffsets[gFeatureOffsetsOffset + index + 1u] - gFeatureOffsets[gFeatureOffsetsOffset + index]`):b(`1u`)),i.output.isCcw&&v(`is-ccw`,i.output.isCcw,_?`${y}\n  ${b(`getLineIsCcw(start, end - start)`)}`:b(`0u`),{declarations:`${c}\n${R}`}),(i.output.isSimple||i.output.isRing)&&l.kind!==`points`){let r=i.output.isSimple??e(t,`${n}-simple`,`uint32`,o);u.push(...this.getSimplicityNodes(t,l,r)),i.output.isRing&&v(`is-ring`,i.output.isRing,_?`${y}\n  ${b(`select(0u, simple[simpleOffset + index], ${I} == 1u)`)}`:b(`0u`),{bindings:[{name:`simple`,view:r,type:`u32`,access:`read`}]})}else i.output.isSimple&&v(`is-simple`,i.output.isSimple,b(`1u`)),i.output.isRing&&v(`is-ring`,i.output.isRing,b(`0u`));return(i.output.equalsExact||i.output.equalsIdentical)&&u.push(...this.getPairNodes(t)),i.output.uniquePositions&&i.output.uniqueOffsets&&u.push(...this.getUniquePointNodes(t,d,f)),u}getSimplicityNodes(t,n,i){let{id:a,props:o,featureCount:s}=this,c=o.intersectionCapacity,u=[];u.push(r(t,{id:`${a}-simple-clear`,operation:M,variant:`simple-clear`,bindings:[{name:`output`,view:i,type:`u32`,access:`read_write`}],invocationCount:s,body:`output[outputOffset + index] = 1u;`}));let d=e(t,`${a}-pair-left`,`uint32`,c),f=e(t,`${a}-pair-right`,`uint32`,c),p=e(t,`${a}-pair-count`,`uint32`,1),m=e(t,`${a}-left-features`,`uint32`,c),h=e(t,`${a}-left-rings`,`uint32`,c),g=e(t,`${a}-right-rings`,`uint32`,c);return u.push(...new l({id:`${a}-intersection`,left:n,sameFeatureOnly:!0,leafCapacity:o.leafCapacity,output:{pairs:{leftRows:d,rightRows:f,status:{count:p,overflow:o.output.status?.overflow,requiredCount:o.output.status?.intersectionCount}},leftFeatures:m,leftRings:h,rightRings:g}}).getCommandNodes(t)),u.push(r(t,{id:`${a}-simple-fold`,operation:M,variant:`simple-fold`,bindings:[{name:`pairCount`,view:p,type:`u32`,access:`read`},{name:`leftFeatures`,view:m,type:`u32`,access:`read`},{name:`leftRings`,view:h,type:`u32`,access:`read`},{name:`rightRings`,view:g,type:`u32`,access:`read`},{name:`simple`,view:i,type:`atomic<u32>`,access:`read_write`}],invocationCount:c,declarations:`const SAME_RING_ONLY: bool = ${n.kind===`polygons`};`,body:`if (index >= pairCount[pairCountOffset]) { return; }
  if (SAME_RING_ONLY && leftRings[leftRingsOffset + index] != rightRings[rightRingsOffset + index]) { return; }
  atomicAnd(&simple[simpleOffset + leftFeatures[leftFeaturesOffset + index]], 0u);`})),u}getPairNodes(t){let{id:n,props:i,featureCount:a}=this,o=i.geometry,s=i.other,c=[],l=e(t,`${n}-pair-layout`,`uint32`,a*3),u=`${j(`a`,o,!1)}\n${j(`b`,s,!1)}`,d;d=o.kind===`polygons`?`let firstA = aFeatureOffsets[aFeatureOffsetsOffset + index];
  let lastA = aFeatureOffsets[aFeatureOffsetsOffset + index + 1u];
  let firstB = bFeatureOffsets[bFeatureOffsetsOffset + index];
  let lastB = bFeatureOffsets[bFeatureOffsetsOffset + index + 1u];
  var matches = (lastA - firstA) == (lastB - firstB);
  for (var k = 0u; matches && k < lastA - firstA; k++) {
    let ringsStartA = aPolygonOffsets[aPolygonOffsetsOffset + firstA + k];
    let ringsEndA = aPolygonOffsets[aPolygonOffsetsOffset + firstA + k + 1u];
    let ringsStartB = bPolygonOffsets[bPolygonOffsetsOffset + firstB + k];
    let ringsEndB = bPolygonOffsets[bPolygonOffsetsOffset + firstB + k + 1u];
    if (ringsEndA - ringsStartA != ringsEndB - ringsStartB) { matches = false; break; }
    for (var r = 0u; r < ringsEndA - ringsStartA; r++) {
      let countA = aRingOffsets[aRingOffsetsOffset + ringsStartA + r + 1u] - aRingOffsets[aRingOffsetsOffset + ringsStartA + r];
      let countB = bRingOffsets[bRingOffsetsOffset + ringsStartB + r + 1u] - bRingOffsets[bRingOffsetsOffset + ringsStartB + r];
      if (countA != countB) { matches = false; break; }
    }
  }`:`let matches = (aVertexStart(index + 1u) - aVertexStart(index)) == (bVertexStart(index + 1u) - bVertexStart(index));`,c.push(r(t,{id:`${n}-pair-structure`,operation:M,variant:`pair-structure`,bindings:[...k(`a`,o,!1),...k(`b`,s,!1),{name:`pairLayout`,view:l,type:`u32`,access:`read_write`}],invocationCount:a,declarations:`${u}\nconst STRUCTURE_MISMATCH: u32 = ${N}u;`,body:`${d}
  let startA = aVertexStart(index);
  let startB = bVertexStart(index);
  pairLayout[pairLayoutOffset + index * 3u] = startA;
  pairLayout[pairLayoutOffset + index * 3u + 1u] = startB;
  pairLayout[pairLayoutOffset + index * 3u + 2u] = select(STRUCTURE_MISMATCH, aVertexStart(index + 1u) - startA, matches);`}));let{parameters:f}=i,{equalsExact:p,equalsIdentical:m}=i.output,h=[{name:`aPositions`,view:o.positions,type:`f32`,access:`read`},{name:`bPositions`,view:s.positions,type:`f32`,access:`read`},{name:`pairLayout`,view:l,type:`u32`,access:`read`}];return p&&f&&h.push({name:`parameters`,view:f,type:`f32`,access:`read`},{name:`exactOutput`,view:p,type:`u32`,access:`read_write`}),m&&h.push({name:`identicalOutput`,view:m,type:`u32`,access:`read_write`}),c.push(r(t,{id:`${n}-pair-compare`,operation:M,variant:`pair-compare`,bindings:h,invocationCount:a,declarations:`${F}
const STRUCTURE_MISMATCH: u32 = ${N}u;
const WRITE_EXACT: bool = ${!!p};
const WRITE_IDENTICAL: bool = ${!!m};
fn readA(vertex: u32) -> vec2f { return vec2f(aPositions[aPositionsOffset + vertex * 2u], aPositions[aPositionsOffset + vertex * 2u + 1u]); }
fn readB(vertex: u32) -> vec2f { return vec2f(bPositions[bPositionsOffset + vertex * 2u], bPositions[bPositionsOffset + vertex * 2u + 1u]); }`,body:`let count = pairLayout[pairLayoutOffset + index * 3u + 2u];
  var exactMatch = count != STRUCTURE_MISMATCH;
  var identicalMatch = exactMatch;
  if (exactMatch) {
    let startA = pairLayout[pairLayoutOffset + index * 3u];
    let startB = pairLayout[pairLayoutOffset + index * 3u + 1u];
    ${p?`let tolerance = parameters[parametersOffset];`:``}
    for (var k = 0u; k < count; k++) {
      let a = readA(startA + k);
      let b = readB(startB + k);
      ${p?`if (exactMatch && !withinTolerance(a, b, tolerance)) { exactMatch = false; }`:``}
      ${m?`if (identicalMatch && !identicalPosition(a, b)) { identicalMatch = false; }`:``}
      if ((!WRITE_EXACT || !exactMatch) && (!WRITE_IDENTICAL || !identicalMatch)) { break; }
    }
  }
  ${p?`exactOutput[exactOutputOffset + index] = select(0u, 1u, exactMatch);`:``}
  ${m?`identicalOutput[identicalOutputOffset + index] = select(0u, 1u, identicalMatch);`:``}`})),c}getUniquePointNodes(i,a,o){let{id:s,props:c,featureCount:l,vertexCount:u}=this,d=c.output.uniquePositions,f=c.output.uniqueOffsets,p=[],m=t=>e(i,`${s}-${t}`,`uint32`,u),h=m(`keys-y`),g=m(`identity`),_=m(`sorted-y`),v=m(`order-y`),y=m(`keys-x`),b=m(`sorted-x`),x=m(`order-x`),S=m(`keys-feature`),C=m(`sorted-feature`),w=m(`order`),T=m(`first-flags`),E=m(`first-starts`),D=(e,t,n,c=u)=>{p.push(r(i,{id:`${s}-unique-${e}`,operation:M,variant:`unique-${e}`,bindings:[...a,...t],invocationCount:c,declarations:`${o}\n${L}`,body:n}))},O=(e,t)=>({name:e,view:t,type:`u32`,access:`read`}),k=(e,t)=>({name:e,view:t,type:`u32`,access:`read_write`}),A=(e,t,r,a,o,c=32)=>p.push(...new n({id:`${s}-unique-sort-${e}`,keys:t,values:r,outputKeys:a,outputValues:o,keyBits:c}).getCommandNodes(i));return D(`keys-y`,[k(`keysOut`,h),k(`idsOut`,g)],`keysOut[keysOutOffset + index] = getOrderKey(gVertex(index).y);
  idsOut[idsOutOffset + index] = index;`),A(`y`,h,g,_,v),D(`keys-x`,[O(`orderIn`,v),k(`keysOut`,y)],`keysOut[keysOutOffset + index] = getOrderKey(gVertex(orderIn[orderInOffset + index]).x);`),A(`x`,y,v,b,x),D(`keys-feature`,[O(`orderIn`,x),k(`keysOut`,S)],`keysOut[keysOutOffset + index] = min(gFeatureOfVertex(orderIn[orderInOffset + index]), g_FEATURE_COUNT);`),A(`feature`,S,x,C,w,Math.max(1,Math.ceil(Math.log2(l+1)))),D(`mark`,[O(`sortedFeature`,C),O(`order`,w),k(`flags`,T)],`let vertex = order[orderOffset + index];
  var first = index == 0u;
  if (!first) {
    let previous = order[orderOffset + index - 1u];
    first = sortedFeature[sortedFeatureOffset + index - 1u] != sortedFeature[sortedFeatureOffset + index]
      || !samePoint(gVertex(vertex), gVertex(previous));
  }
  flags[flagsOffset + vertex] = select(0u, 1u, first);`),p.push(...new t({id:`${s}-unique-scan`,input:T,output:E,mode:`exclusive`}).getCommandNodes(i)),D(`emit`,[O(`flags`,T),O(`starts`,E),{name:`uniquePositions`,view:d,type:`f32`,access:`read_write`}],`if (flags[flagsOffset + index] == 1u) {
    let point = gVertex(index);
    let slot = starts[startsOffset + index];
    uniquePositions[uniquePositionsOffset + slot * 2u] = point.x;
    uniquePositions[uniquePositionsOffset + slot * 2u + 1u] = point.y;
  }`),D(`offsets`,[O(`flags`,T),O(`starts`,E),k(`uniqueOffsets`,f)],`let vertex = gVertexStart(index);
  var prefix = 0u;
  if (vertex >= g_VERTEX_COUNT) {
    prefix = starts[startsOffset + g_VERTEX_COUNT - 1u] + flags[flagsOffset + g_VERTEX_COUNT - 1u];
  } else {
    prefix = starts[startsOffset + vertex];
  }
  uniqueOffsets[uniqueOffsetsOffset + index] = prefix;`,l+1),p}},F=`
fn isNaNBits(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn canonicalBits(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits, 0u, (bits & 0x7fffffffu) == 0u);
}
fn samePoint(first: vec2f, second: vec2f) -> bool {
  return canonicalBits(first.x) == canonicalBits(second.x) && canonicalBits(first.y) == canonicalBits(second.y);
}
fn identicalValue(first: f32, second: f32) -> bool {
  return canonicalBits(first) == canonicalBits(second) || (isNaNBits(first) && isNaNBits(second));
}
fn identicalPosition(first: vec2f, second: vec2f) -> bool {
  return identicalValue(first.x, second.x) && identicalValue(first.y, second.y);
}
// Distance at most tolerance, scaled so that tiny and huge offsets neither underflow nor overflow.
fn withinTolerance(first: vec2f, second: vec2f, tolerance: f32) -> bool {
  if (isNaNBits(first.x) || isNaNBits(first.y) || isNaNBits(second.x) || isNaNBits(second.y)) { return false; }
  let delta = first - second;
  let scale = max(abs(delta.x), abs(delta.y));
  if (scale == 0.0) { return true; }
  let scaled = delta / scale;
  return scale * sqrt(scaled.x * scaled.x + scaled.y * scaled.y) <= tolerance;
}
`,I=`select(0u, 1u, end >= start + 2u && samePoint(gVertex(start), gVertex(end - 1u)))`,L=`
fn getOrderKey(value: f32) -> u32 {
  var bits = bitcast<u32>(value);
  if ((bits & 0x7fffffffu) == 0u) { bits = 0u; }
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`,R=`
fn getLineIsCcw(start: u32, size: u32) -> u32 {
  if (size < 4u) { return 0u; }
  let pointCount = size - 1u;
  var upHigh = gVertex(start);
  var upLow = upHigh;
  var previousY = upHigh.y;
  var upHighIndex = 0u;
  for (var i = 1u; i <= pointCount; i++) {
    let point = gVertex(start + i);
    if (point.y > previousY && point.y >= upHigh.y) {
      upHigh = point;
      upHighIndex = i;
      upLow = gVertex(start + i - 1u);
    }
    previousY = point.y;
  }
  if (upHighIndex == 0u) { return 0u; }
  var downLowIndex = upHighIndex;
  for (var step = 0u; step < size; step++) {
    downLowIndex = (downLowIndex + 1u) % pointCount;
    if (!(downLowIndex != upHighIndex && gVertex(start + downLowIndex).y == upHigh.y)) { break; }
  }
  let downLow = gVertex(start + downLowIndex);
  let downHigh = gVertex(start + select(pointCount - 1u, downLowIndex - 1u, downLowIndex > 0u));
  if (samePoint(upHigh, downHigh)) {
    if (samePoint(upLow, upHigh) || samePoint(downLow, upHigh) || samePoint(upLow, downLow)) { return 0u; }
    return select(0u, 1u, orientSign(upLow, upHigh, downLow) == 1);
  }
  return select(0u, 1u, downHigh.x < upHigh.x);
}
`;function z(e,t){let n=C(e,t,{rows:3,columns:5,vertexCount:8,irregularity:.2}),r=[],i=[0];for(let e=0;e<n.count;e++){let t=n.ringOffsets[e],a=[];for(let e=0;e<8;e++)a.push([n.positions[(t+e)*2],n.positions[(t+e)*2+1]]);let o=e%4;o===1&&a.reverse(),o===2&&([a[2],a[3]]=[a[3],a[2]]),o!==3&&a.push(a[0]);for(let[e,t]of a)r.push(e,t);i.push(r.length/2)}let a=Float32Array.from({length:n.count*2},(e,t)=>{let n=0;for(let e=i[t>>1];e<i[(t>>1)+1];e++)n+=r[e*2+(t&1)];return n/(i[(t>>1)+1]-i[t>>1])});return{count:n.count,positions:Float32Array.from(r),lineOffsets:Uint32Array.from(i),lineCenters:a,index:y(a),focus:{line:-1}}}var B=v({parameters:E,caption:`Per-line flags: simple, ring, closed and counter-clockwise.`,createInputs:({seed:e,world:t})=>z(_(e),t),build(e,t){let n=e.upload(`positions`,`float32x2`,t.positions),r=e.upload(`line-offsets`,`uint32`,t.lineOffsets),i={isSimple:e.allocate(`is-simple`,`uint32`,t.count),isRing:e.allocate(`is-ring`,`uint32`,t.count),isClosed:e.allocate(`is-closed`,`uint32`,t.count),isCcw:e.allocate(`is-ccw`,`uint32`,t.count)};e.graph.add(new P({geometry:{kind:`lines`,positions:n.view,lineOffsets:r.view},intersectionCapacity:4096,output:{isSimple:i.isSimple.view,isRing:i.isRing.view,isClosed:i.isClosed.view,isCcw:i.isCcw.view,status:{overflow:e.allocate(`overflow`,`uint32`,1).view}}}));let a=e.upload(`line-of-vertex`,`uint32`,x(t.lineOffsets));return{update:()=>void 0,outputs:{columns:i,positions:n,edges:e.upload(`edges`,`uint32x2`,S(t.lineOffsets)),flags:[T(e,`simple`,i.isSimple,a),T(e,`ring`,i.isRing,a),T(e,`closed`,i.isClosed,a),T(e,`ccw`,i.isCcw,a)]}}},interaction:{kind:`probe`,hint:`hover a line`,rest:({inputs:e,toUnit:t})=>t([e.lineCenters[4],e.lineCenters[5]]),resolve(e,{inputs:t,outputs:n,peek:r}){let i=b(t.index,e.x,e.y,t.focus.line);if(t.focus.line=i,i<0)return{};let a=t.lineOffsets[i],o=t.lineOffsets[i+1],s=[[`isSimple`,r(n.columns.isSimple)],[`isRing`,r(n.columns.isRing)],[`isClosed`,r(n.columns.isClosed)],[`isCcw`,r(n.columns.isCcw)]],c=w(t.positions,a,o-a-1);return{focus:i,glyphs:[{shape:`polyline`,points:t.positions.subarray(a*2,o*2)},...s[0][1]?.[i]===0?c.map(e=>({shape:`dot`,at:e})):[]],readout:s.map(([e,t])=>`${e} ${t?t[i]===1:`...`}`).join(` · `)}}},draw:({outputs:e,values:t})=>[{layer:`segments`,nodes:e.positions,edges:e.edges,values:e.flags[t.output],radius:2.8}]});export{B as default};
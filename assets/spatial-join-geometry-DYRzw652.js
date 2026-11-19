import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";import{s as t}from"./kernel-C3tfQOn4.js";import{E as n,O as r,T as i,c as a,f as o,v as s}from"./gpu-contributor-utils-BIwLQON5.js";import{a as c}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as l}from"./ports-xmdKgL07.js";import{i as u,r as d}from"./spatial-join-passes-CNxNpnDa.js";var f=4294967295;function p(e){return e.buffer.defaultBuffer?r(e.buffer.defaultBuffer):e.buffer}function m(e,n,r,l){c(e,n,r);let u=new Map;for(let[s,c]of Object.entries(r))for(let r of c?a(c):[]){o(n,`${s} aligned storage words (SPATIAL_JOIN_MAXIMUM_STORAGE_WORD_COUNT)`,t(r).size/4,1,f);let a=p(r),c=u.get(a);i(n,!c||c===r.buffer,`${s} physical buffer must be imported once; create subviews of its existing graph handle`),u.set(a,r.buffer),o(n,`${s}.buffer.byteLength (device.limits.maxBufferSize)`,r.buffer.byteLength,0,e.device.limits.maxBufferSize)}for(let[t,r]of Object.entries(l)){let i=Math.max(r,4);o(n,`${t} storage words (SPATIAL_JOIN_MAXIMUM_STORAGE_WORD_COUNT)`,i/4,1,f),s(e.device,n,t,i,`reduce row counts or capacities`),o(n,`${t} byteLength (device.limits.maxBufferSize)`,i,0,e.device.limits.maxBufferSize)}}function h(e,t,n){let r=new Set(t.flatMap(e=>e?a(e).filter(e=>e.length>0).map(p):[]));for(let t of n)for(let n of t?a(t):[])n.length!==0&&i(e,!r.has(p(n)),`one physical buffer must not be bound read-only and writable in the same dispatch`)}var g=`GPUSpatialPredicateJoin`;function _(e){return(e.kind===`points`||e.kind===`lines`)&&`featureOffsets`in e&&e.featureOffsets!==void 0}function v(e,t,n){(n.kind===`points`||n.kind===`lines`)&&i(e,!(`featureOffsets`in n)||n.featureOffsets===void 0,`${t}.featureOffsets multipart ${n.kind} are unsupported; use explicit part rows with an owner map or a feature-level CPU join`)}function y(e){if(_(e))return e.featureOffsets.length-1;switch(e.kind){case`points`:return e.positions.length;case`lines`:return e.lineOffsets.length-1;case`polygons`:return e.featureOffsets.length-1}}function b(e){return e.kind===`lines`?[e.positions,e.lineOffsets,..._(e)?[e.featureOffsets]:[]]:e.kind===`polygons`?[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets]:[e.positions,..._(e)?[e.featureOffsets]:[]]}function x(e,t,r,i={}){i.allowMultipart||v(e,t,r);for(let i of[...b(r),r.sourceIds])i&&n(i,[i.format],`${e} ${t}`);if(l(`${e} ${t}`,r),!i.allowEmpty&&y(r)<1)throw Error(`${e} ${t} needs at least one feature`)}function S(e,t){for(let[r,i]of Object.entries(t))if(i&&(n(i,[`uint32`],`${e} output.status.${r}`),i.length<1))throw Error(`${e} output.status.${r} needs one row`)}function C(e,t){if(e.kind!==t.kind)return!1;let n=b(e),r=b(t);return n.length===r.length&&n.every((e,t)=>e===r[t])}function w(t,n,r){let{geometry:i}=r;return e(t,{id:n,operation:g,variant:`feature-rings`,bindings:[{name:`featureOffsets`,view:i.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:i.polygonOffsets,type:`u32`,access:`read`},{name:`featureRings`,view:r.featureRings,type:`u32`,access:`read_write`}],invocationCount:r.featureCount,declarations:`const POLYGON_COUNT: u32 = ${i.polygonOffsets.length-1}u;
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
  featureRings[featureRingsOffset + index * 2u + 1u] = ringEnd;`})}function T(t,n,r){let{geometry:i}=r;if(i.kind===`points`&&!_(i))return u(t,{id:n,operation:r.operation??g,featureCount:r.featureCount,source:{kind:`points`,positions:i.positions},minima:r.minima,maxima:r.maxima});let a=[{name:`positions`,view:i.positions,type:`f32`,access:`read`}],o;if(_(i)){a.push({name:`featureOffsets`,view:i.featureOffsets,type:`u32`,access:`read`});let e=i.kind===`points`?i.positions.length:i.lineOffsets.length-1;i.kind===`lines`&&a.push({name:`lineOffsets`,view:i.lineOffsets,type:`u32`,access:`read`}),o=`let partStart = featureOffsets[featureOffsetsOffset + index];
  let partEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var vertexStart = 0u;
  var vertexEnd = 0u;
  if (partStart < partEnd && partEnd <= ${e}u) {
    vertexStart = ${i.kind===`points`?`partStart`:`lineOffsets[lineOffsetsOffset + partStart]`};
    vertexEnd = min(${i.kind===`points`?`partEnd`:`lineOffsets[lineOffsetsOffset + partEnd]`}, VERTEX_COUNT);
  }`}else i.kind===`lines`?(a.push({name:`lineOffsets`,view:i.lineOffsets,type:`u32`,access:`read`}),o=`let vertexStart = lineOffsets[lineOffsetsOffset + index];
  let vertexEnd = min(lineOffsets[lineOffsetsOffset + index + 1u], VERTEX_COUNT);`):i.kind===`polygons`?(a.push({name:`featureRings`,view:r.featureRings,type:`u32`,access:`read`},{name:`ringOffsets`,view:i.ringOffsets,type:`u32`,access:`read`}),o=`let ringStart = featureRings[featureRingsOffset + index * 2u];
  let ringEnd = featureRings[featureRingsOffset + index * 2u + 1u];
  var vertexStart = 0u;
  var vertexEnd = 0u;
  if (ringStart < ringEnd) {
    vertexStart = ringOffsets[ringOffsetsOffset + ringStart];
    vertexEnd = min(ringOffsets[ringOffsetsOffset + ringEnd], VERTEX_COUNT);
  }`):o=`let vertexStart = index; let vertexEnd = index + 1u;`;return a.push({name:`featureMinima`,view:r.minima,type:`f32`,access:`read_write`},{name:`featureMaxima`,view:r.maxima,type:`f32`,access:`read_write`}),e(t,{id:n,operation:r.operation??g,variant:`bounds-${i.kind}`,bindings:a,invocationCount:r.featureCount,declarations:`${d}
const VERTEX_COUNT: u32 = ${i.positions.length}u;`,body:`${o}
  var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
    let position = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
    if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
      minimum = min(minimum, position);
      maximum = max(maximum, position);
    }
  }
  featureMinima[featureMinimaOffset + index * 2u] = minimum.x;
  featureMinima[featureMinimaOffset + index * 2u + 1u] = minimum.y;
  featureMaxima[featureMaximaOffset + index * 2u] = maximum.x;
  featureMaxima[featureMaximaOffset + index * 2u + 1u] = maximum.y;`})}export{_ as a,v as c,h as d,b as i,S as l,w as n,C as o,y as r,x as s,T as t,m as u};
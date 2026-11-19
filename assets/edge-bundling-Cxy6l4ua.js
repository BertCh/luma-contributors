import{o as e,s as t}from"./network-view-utils-B1U8oNKx.js";import{Q as n}from"./dist-BusamGlx.js";import{r}from"./kernel-C3tfQOn4.js";import{E as i,f as a,v as o}from"./gpu-contributor-utils-BIwLQON5.js";import{v as s}from"./spatial-context-DwZEd4EZ.js";import{t as c}from"./gpu-dispatch-utils-CbRzFAgT.js";import{c as l,o as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as d}from"./wgsl-common-DfyXiYB0.js";import{t as f}from"./define-run-B-brkCe4.js";import{r as p,t as m}from"./commute-flows-CmHJas7f.js";import{a as h}from"./pointer-overlay-Qh8qflcA.js";import{n as g}from"./graphs-Bws-bfBV.js";import{Ba as _}from"./entry-preview-D70TdSbK.js";var v=`GPUEdgeBundling`,y=-1e8,b=-1e7,ee=1e-9,x=1e-9;function S(e){return`
const EDGE_COUNT: u32 = ${e.edgeCount}u;
const VERTEX_COUNT: u32 = ${e.vertexCount}u;
const POINTS: u32 = ${e.pointsPerEdge}u;
const RESOLUTION: u32 = ${e.densityResolution}u;
const RESOLUTION_F: f32 = ${l(e.densityResolution)};
const FIXED_POINT_SCALE: f32 = ${l(2**e.fixedPointExponent)};
const FIXED_POINT_INVERSE: f32 = ${l(2**-e.fixedPointExponent)};
const MAXIMUM_ITERATIONS: u32 = ${e.iterationCount}u;
const BOX_SCALE: f32 = ${l(1+2*e.boxPadding)};
const DEAD_COORDINATE: f32 = ${l(y)};
const DEAD_LIMIT: f32 = ${l(b)};
const LENGTH_EPSILON: f32 = ${l(ee)};
const GRADIENT_MINIMUM: f32 = ${l(4)} * FIXED_POINT_INVERSE;
const RADIUS_EPSILON: f32 = ${l(x)};`}function C(e){let t;if(!e)t=`
fn readActiveIterations() -> u32 { return MAXIMUM_ITERATIONS; }
fn readInitialRadius() -> f32 { return 0.03; }
fn readLambda() -> f32 { return 0.85; }
fn readSmoothing() -> f32 { return 0.5; }
fn readStepScale() -> f32 { return 1.0; }`;else{let n=e.format===`float32`,r=e=>n?`parameters[parametersOffset + ${e}u]`:`bitcast<f32>(parameters[parametersOffset + ${e}u])`;t=`
fn readActiveIterations() -> u32 {
  ${n?`return u32(clamp(parameters[parametersOffset], 0.0, f32(MAXIMUM_ITERATIONS)));`:`return min(parameters[parametersOffset], MAXIMUM_ITERATIONS);`}
}
fn readInitialRadius() -> f32 { return max(${r(1)}, 0.0); }
fn readLambda() -> f32 { return clamp(${r(2)}, 0.5, 0.9); }
fn readSmoothing() -> f32 { return clamp(${r(3)}, 0.0, 1.0); }
fn readStepScale() -> f32 { return ${e.length>4?`max(${r(4)}, 0.0)`:`1.0`}; }`}return t}function w(){return`
fn getIterationRadius() -> f32 {
  let iteration = iterationInfo[iterationInfoOffset];
  var radius = readInitialRadius();
  let lambda = readLambda();
  for (var round = 0u; round < iteration; round++) {
    radius = radius * lambda;
  }
  return radius;
}`}function T(e,t){return u(e,{id:t.id,operation:v,variant:`iteration-info`,bindings:[{name:`iterationInfo`,view:t.iterationInfo,type:`u32`,access:`read_write`}],invocationCount:t.iterationCount,body:`iterationInfo[iterationInfoOffset + index * 64u] = index;`})}function E(e){return`
// Strict 3.0e38 bound (finite and not near overflow), unlike the shared isFiniteValue.
${d}
fn isFiniteValue(value: f32) -> bool { return isFiniteBits(value) && abs(value) < 3.0e38; }
fn getVertexPosition(vertex: u32) -> vec2<f32> {
  return vec2<f32>(
    vertexPositions[vertexPositionsOffset + vertex * 2u],
    vertexPositions[vertexPositionsOffset + vertex * 2u + 1u]
  );
}
fn isEdgeLive(edge: u32) -> bool {
  let source = edgeSources[edgeSourcesOffset + edge];
  let targetVertex = edgeTargets[edgeTargetsOffset + edge];
  if (source >= VERTEX_COUNT || targetVertex >= VERTEX_COUNT) {
    return false;
  }
  ${e?`if (edgeMask[edgeMaskOffset + edge] == 0u) {
    return false;
  }`:``}
  let a = getVertexPosition(source);
  let b = getVertexPosition(targetVertex);
  return isFiniteValue(a.x) && isFiniteValue(a.y) && isFiniteValue(b.x) && isFiniteValue(b.y);
}`}function D(e){return`
fn decodeKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
fn getBox() -> vec4<f32> {
  if (boxKeys[boxKeysOffset] == 0xffffffffu) {
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
  }
  let minX = decodeKey(boxKeys[boxKeysOffset]);
  let minY = decodeKey(boxKeys[boxKeysOffset + 1u]);
  let maxX = decodeKey(boxKeys[boxKeysOffset + 2u]);
  let maxY = decodeKey(boxKeys[boxKeysOffset + 3u]);
  ${e?`let xScale = max(cos(0.5 * (minY + maxY) * 0.017453292519943295), 0.01);`:`let xScale = 1.0;`}
  let extent = max((maxX - minX) * xScale, maxY - minY);
  let side = select(extent * BOX_SCALE, 1.0, !(extent > 0.0));
  return vec4<f32>(
    0.5 * (minX + maxX) * xScale - 0.5 * side,
    0.5 * (minY + maxY) - 0.5 * side,
    side,
    xScale
  );
}
fn toWork(workBox: vec4<f32>, position: vec2<f32>) -> vec2<f32> {
  return (vec2<f32>(position.x * workBox.w, position.y) - workBox.xy) / workBox.z;
}
fn fromWork(workBox: vec4<f32>, normalized: vec2<f32>) -> vec2<f32> {
  let scaled = workBox.xy + normalized * workBox.z;
  return vec2<f32>(scaled.x / workBox.w, scaled.y);
}`}function O(e){let t=[{name:`vertexPositions`,view:e.positions,type:`f32`,access:`read`},{name:`edgeSources`,view:e.sources,type:`u32`,access:`read`},{name:`edgeTargets`,view:e.targets,type:`u32`,access:`read`}];return e.mask&&t.push({name:`edgeMask`,view:e.mask,type:`u32`,access:`read`}),t}var k=256,A=64,j={clear:0,splat:1,update:2};function M(e,t,n){if(!e)return{};let r=e.view.byteOffset+4*(e.iteration*9+3*j[n]);return{condition:{id:`${t}-gate`,source:`gpu`,mode:`indirect`,buffer:e.view.buffer,byteOffset:r},extraResources:[{buffer:e.view,usage:`indirect`}]}}function N(e,t){let{constants:n}=t,r=e.device.limits.maxComputeWorkgroupsPerDimension,i={clear:c(v,n.densityResolution*n.densityResolution,256,r),splat:c(v,n.edgeCount*n.pointsPerEdge,256,r),update:c(v,n.edgeCount,A,r)},a=[`clear`,`splat`,`update`].map((e,t)=>`    gate[gateOffset + base + ${3*t}u] = select(0u, ${i[e].x}u, isActive);
    gate[gateOffset + base + ${3*t+1}u] = ${i[e].y}u;
    gate[gateOffset + base + ${3*t+2}u] = ${i[e].z}u;`).join(`
`);return u(e,{id:t.id,operation:v,variant:`gate`,bindings:[{name:`parameters`,view:t.parameters,type:t.parameters.format===`float32`?`f32`:`u32`,access:`read`},{name:`gate`,view:t.gate,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${S(n)}
${C(t.parameters)}`,body:`let activeIterations = readActiveIterations();
  var radius = readInitialRadius();
  let lambda = readLambda();
  for (var iteration = 0u; iteration < MAXIMUM_ITERATIONS; iteration++) {
    let isActive = iteration < activeIterations && radius > RADIUS_EPSILON;
    let base = iteration * 9u;
${a}
    radius = radius * lambda;
  }`})}function P(e,t){return u(e,{id:t.id,operation:v,variant:`box-reset`,bindings:[{name:`boxKeys`,view:t.boxKeys,type:`u32`,access:`read_write`}],invocationCount:4,body:`boxKeys[boxKeysOffset + index] = select(0u, 0xffffffffu, index < 2u);`})}function F(e,t){return u(e,{id:t.id,operation:v,variant:`box`,workgroupSize:k,bindings:[...O(t.inputs),{name:`boxKeys`,view:t.boxKeys,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.constants.edgeCount,guardIndex:!1,declarations:`${S(t.constants)}
${E(!!t.inputs.mask)}
fn encodeKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
var<workgroup> sharedBox: array<u32, ${4*k}>;`,body:`var lowX = 0xffffffffu;
  var lowY = 0xffffffffu;
  var highX = 0u;
  var highY = 0u;
  if (index < EDGE_COUNT && isEdgeLive(index)) {
    let a = getVertexPosition(edgeSources[edgeSourcesOffset + index]);
    let b = getVertexPosition(edgeTargets[edgeTargetsOffset + index]);
    let low = min(a, b);
    let high = max(a, b);
    lowX = encodeKey(low.x);
    lowY = encodeKey(low.y);
    highX = encodeKey(high.x);
    highY = encodeKey(high.y);
  }
  sharedBox[localInvocationIndex] = lowX;
  sharedBox[${k}u + localInvocationIndex] = lowY;
  sharedBox[${2*k}u + localInvocationIndex] = highX;
  sharedBox[${3*k}u + localInvocationIndex] = highY;
  workgroupBarrier();
  for (var stride = ${k/2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      let other = localInvocationIndex + stride;
      sharedBox[localInvocationIndex] = min(sharedBox[localInvocationIndex], sharedBox[other]);
      sharedBox[${k}u + localInvocationIndex] =
        min(sharedBox[${k}u + localInvocationIndex], sharedBox[${k}u + other]);
      sharedBox[${2*k}u + localInvocationIndex] =
        max(sharedBox[${2*k}u + localInvocationIndex], sharedBox[${2*k}u + other]);
      sharedBox[${3*k}u + localInvocationIndex] =
        max(sharedBox[${3*k}u + localInvocationIndex], sharedBox[${3*k}u + other]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    atomicMin(&boxKeys[boxKeysOffset], sharedBox[0]);
    atomicMin(&boxKeys[boxKeysOffset + 1u], sharedBox[${k}u]);
    atomicMax(&boxKeys[boxKeysOffset + 2u], sharedBox[${2*k}u]);
    atomicMax(&boxKeys[boxKeysOffset + 3u], sharedBox[${3*k}u]);
  }`})}function I(e,t){return u(e,{id:t.id,operation:v,variant:`initialize`,bindings:[...O(t.inputs),{name:`boxKeys`,view:t.boxKeys,type:`u32`,access:`read`},{name:`work`,view:t.work,type:`f32`,access:`read_write`}],invocationCount:t.constants.edgeCount*t.constants.pointsPerEdge,declarations:`${S(t.constants)}
${E(!!t.inputs.mask)}
${D(t.constants.geographic)}`,body:`let edge = index / POINTS;
  let point = index % POINTS;
  if (!isEdgeLive(edge)) {
    work[workOffset + index * 2u] = DEAD_COORDINATE;
    work[workOffset + index * 2u + 1u] = DEAD_COORDINATE;
    return;
  }
  let workBox = getBox();
  let a = toWork(workBox, getVertexPosition(edgeSources[edgeSourcesOffset + edge]));
  let b = toWork(workBox, getVertexPosition(edgeTargets[edgeTargetsOffset + edge]));
  let t = f32(point) / f32(POINTS - 1u);
  var placed = a + (b - a) * t;
  if (point == 0u) {
    placed = a;
  } else if (point == POINTS - 1u) {
    placed = b;
  }
  work[workOffset + index * 2u] = placed.x;
  work[workOffset + index * 2u + 1u] = placed.y;`})}function L(e,t){return u(e,{id:t.id,operation:v,variant:`fill`,bindings:[{name:`density`,view:t.density,type:`u32`,access:`read_write`}],invocationCount:t.density.length,body:`density[densityOffset + index] = 0u;`,...M(t.gate,t.id,`clear`)})}function R(e,t){let{constants:n}=t,r=[{name:`iterationInfo`,view:t.iterationInfo,type:`u32`,access:`read`},{name:`work`,view:t.work,type:`f32`,access:`read`},{name:`density`,view:t.density,type:`atomic<u32>`,access:`read_write`}];return t.parameters&&r.push({name:`parameters`,view:t.parameters,type:t.parameters.format===`float32`?`f32`:`u32`,access:`read`}),u(e,{id:t.id,operation:v,variant:`splat`,bindings:r,invocationCount:n.edgeCount*n.pointsPerEdge,declarations:`${S(n)}
${C(t.parameters)}
${w()}`,body:`let radius = getIterationRadius();
  let px = work[workOffset + index * 2u];
  let py = work[workOffset + index * 2u + 1u];
  if (px < DEAD_LIMIT) {
    return;
  }
  let gx = px * RESOLUTION_F;
  let gy = py * RESOLUTION_F;
  let radiusCells = radius * RESOLUTION_F;
  let reach = i32(max(1.0, ceil(radiusCells)));
  let inverseSquare = 1.0 / (radiusCells * radiusCells);
  let cx = i32(floor(gx));
  let cy = i32(floor(gy));
  let maximumCell = i32(RESOLUTION) - 1;
  let x0 = max(cx - reach, 0);
  let x1 = min(cx + reach, maximumCell);
  let y0 = max(cy - reach, 0);
  let y1 = min(cy + reach, maximumCell);
  for (var y = y0; y <= y1; y++) {
    let dy = f32(y) + 0.5 - gy;
    for (var x = x0; x <= x1; x++) {
      let dx = f32(x) + 0.5 - gx;
      let q = (dx * dx + dy * dy) * inverseSquare;
      if (q < 1.0) {
        let weight = u32((1.0 - q) * FIXED_POINT_SCALE + 0.5);
        if (weight > 0u) {
          atomicAdd(&density[densityOffset + u32(y) * RESOLUTION + u32(x)], weight);
        }
      }
    }
  }`,...M(t.gate,t.id,`splat`)})}function z(e,t){let{constants:n}=t,r=[{name:`iterationInfo`,view:t.iterationInfo,type:`u32`,access:`read`},{name:`work`,view:t.work,type:`f32`,access:`read_write`},{name:`density`,view:t.density,type:`u32`,access:`read`}];return t.parameters&&r.push({name:`parameters`,view:t.parameters,type:t.parameters.format===`float32`?`f32`:`u32`,access:`read`}),u(e,{id:t.id,operation:v,variant:`update`,bindings:r,invocationCount:n.edgeCount,workgroupSize:64,declarations:`${S(n)}
${C(t.parameters)}
${w()}
fn readDensity(x: i32, y: i32) -> f32 {
  return f32(density[densityOffset + u32(y) * RESOLUTION + u32(x)]) * FIXED_POINT_INVERSE;
}
/** Bilinear density at grid coordinates where integers are cell centers; edges clamp. */
fn sampleDensity(gx: f32, gy: f32) -> f32 {
  let maximumCell = f32(RESOLUTION) - 1.0;
  let cx = clamp(gx, 0.0, maximumCell);
  let cy = clamp(gy, 0.0, maximumCell);
  let x0 = i32(floor(cx));
  let y0 = i32(floor(cy));
  let x1 = min(x0 + 1, i32(RESOLUTION) - 1);
  let y1 = min(y0 + 1, i32(RESOLUTION) - 1);
  let fx = cx - f32(x0);
  let fy = cy - f32(y0);
  return readDensity(x0, y0) * (1.0 - fx) * (1.0 - fy) +
    readDensity(x1, y0) * fx * (1.0 - fy) +
    readDensity(x0, y1) * (1.0 - fx) * fy +
    readDensity(x1, y1) * fx * fy;
}`,body:`let radius = getIterationRadius();
  let base = index * POINTS;
  if (work[workOffset + base * 2u] < DEAD_LIMIT) {
    return;
  }
  var points: array<vec2<f32>, ${n.pointsPerEdge}>;
  var resampled: array<vec2<f32>, ${n.pointsPerEdge}>;
  for (var i = 0u; i < POINTS; i++) {
    points[i] = vec2<f32>(
      work[workOffset + (base + i) * 2u],
      work[workOffset + (base + i) * 2u + 1u]
    );
  }
  // Advect interior points one scaled bandwidth along the bilinearly sampled, normalized gradient.
  let advectionStep = radius * readStepScale();
  for (var i = 1u; i < POINTS - 1u; i++) {
    let gx = points[i].x * RESOLUTION_F - 0.5;
    let gy = points[i].y * RESOLUTION_F - 0.5;
    if (gx < 1.0 || gy < 1.0 || gx >= RESOLUTION_F - 2.0 || gy >= RESOLUTION_F - 2.0) {
      continue;
    }
    let gradientX = sampleDensity(gx + 1.0, gy) - sampleDensity(gx - 1.0, gy);
    let gradientY = sampleDensity(gx, gy + 1.0) - sampleDensity(gx, gy - 1.0);
    let gradientLength = sqrt(gradientX * gradientX + gradientY * gradientY);
    if (gradientLength < GRADIENT_MINIMUM) {
      continue;
    }
    points[i] = points[i] + vec2<f32>(gradientX / gradientLength, gradientY / gradientLength) * advectionStep;
  }
  // Resample to uniform arc length, keeping both endpoints.
  var total = 0.0;
  for (var i = 1u; i < POINTS; i++) {
    let delta = points[i] - points[i - 1u];
    total = total + sqrt(delta.x * delta.x + delta.y * delta.y);
  }
  var segment = 1u;
  var consumed = 0.0;
  var delta = points[1] - points[0];
  var segmentLength = sqrt(delta.x * delta.x + delta.y * delta.y);
  for (var i = 0u; i < POINTS; i++) {
    let arcTarget = total * f32(i) / f32(POINTS - 1u);
    while (segment < POINTS - 1u && consumed + segmentLength < arcTarget) {
      consumed = consumed + segmentLength;
      segment = segment + 1u;
      delta = points[segment] - points[segment - 1u];
      segmentLength = sqrt(delta.x * delta.x + delta.y * delta.y);
    }
    let fraction = select((arcTarget - consumed) / segmentLength, 0.0, segmentLength < LENGTH_EPSILON);
    let previous = points[segment - 1u];
    resampled[i] = previous + (points[segment] - previous) * fraction;
  }
  resampled[0] = points[0];
  resampled[POINTS - 1u] = points[POINTS - 1u];
  // One simultaneous Laplacian pass from the resampled snapshot.
  let smoothing = readSmoothing();
  for (var i = 0u; i < POINTS; i++) {
    var position = resampled[i];
    if (i > 0u && i < POINTS - 1u) {
      let midpoint = 0.5 * (resampled[i - 1u] + resampled[i + 1u]) - position;
      position = position + smoothing * midpoint;
    }
    work[workOffset + (base + i) * 2u] = position.x;
    work[workOffset + (base + i) * 2u + 1u] = position.y;
  }`,...M(t.gate,t.id,`update`)})}function B(e,t){return u(e,{id:t.id,operation:v,variant:`finalize`,bindings:[...O(t.inputs),{name:`boxKeys`,view:t.boxKeys,type:`u32`,access:`read`},{name:`work`,view:t.work,type:`f32`,access:`read`},{name:`paths`,view:t.paths,type:`f32`,access:`read_write`}],invocationCount:t.constants.edgeCount*t.constants.pointsPerEdge,declarations:`${S(t.constants)}
${E(!!t.inputs.mask)}
${D(t.constants.geographic)}`,body:`let edge = index / POINTS;
  let point = index % POINTS;
  let source = edgeSources[edgeSourcesOffset + edge];
  var outputPoint = vec2<f32>(0.0, 0.0);
  if (!isEdgeLive(edge)) {
    if (source < VERTEX_COUNT) {
      outputPoint = getVertexPosition(source);
    }
  } else if (point == 0u) {
    outputPoint = getVertexPosition(source);
  } else if (point == POINTS - 1u) {
    outputPoint = getVertexPosition(edgeTargets[edgeTargetsOffset + edge]);
  } else {
    let workBox = getBox();
    let normalized = vec2<f32>(work[workOffset + index * 2u], work[workOffset + index * 2u + 1u]);
    outputPoint = fromWork(workBox, normalized);
  }
  paths[pathsOffset + index * 2u] = outputPoint.x;
  paths[pathsOffset + index * 2u + 1u] = outputPoint.y;`})}function V(e,t){let n=[];return t.startIndices&&n.push({name:`startIndices`,view:t.startIndices,type:`u32`,access:`read_write`}),t.drawRecord&&n.push({name:`drawRecord`,view:t.drawRecord,type:`u32`,access:`read_write`}),u(e,{id:t.id,operation:v,variant:`indices`,bindings:n,invocationCount:Math.max(t.constants.edgeCount+1,4),declarations:S(t.constants),body:`${t.startIndices?`if (index <= EDGE_COUNT) {
    startIndices[startIndicesOffset + index] = index * POINTS;
  }`:``}
  ${t.drawRecord?`if (index < 4u) {
    drawRecord[drawRecordOffset + index] = select(select(0u, EDGE_COUNT, index == 1u), POINTS, index == 0u);
  }`:``}`})}var H=.05,U={pointsPerEdge:16,maximumIterations:15,densityResolution:256,kernelRadius:.03,lambda:.85,smoothing:.5,stepScale:1},W={coordinateSpaces:[`planar`,`longitude-latitude`]},G=16,K=8,q=8;function J(e){let t=G;for(;t>0&&e*2**t>4294967295;)t--;return t}function Y(e,t){let n=Float32Array.of(e.activeIterations??64,e.kernelRadius??U.kernelRadius,e.lambda??U.lambda,e.smoothing??U.smoothing,e.stepScale??U.stepScale);if(t===`float32`)return n;let r=new Uint32Array(n.buffer.slice(0));return r[0]=Math.max(0,Math.floor(e.activeIterations??64)),r}var X=class{id;props;pointsPerEdge;maximumIterations;densityResolution;edgeCount;fixedPointExponent;spatialContext;constructor(e){this.id=e.id??`edge-bundling`,this.props=e,t(this.id,e.output),this.pointsPerEdge=e.pointsPerEdge??U.pointsPerEdge,this.maximumIterations=e.maximumIterations??U.maximumIterations,this.densityResolution=e.densityResolution??U.densityResolution,this.edgeCount=e.sourceNodes.length;let{id:n,pointsPerEdge:r,maximumIterations:o,densityResolution:c,edgeCount:l}=this,{output:u}=e;this.spatialContext=s(n,e.spatialContext,W),i(e.nodePositions,[`float32x2`],`${n} nodePositions`),i(u.paths,[`float32x2`],`${n} output.paths`);for(let[t,r]of[[`sourceNodes`,e.sourceNodes],[`targetNodes`,e.targetNodes],[`edgeMask`,e.edgeMask],[`output.lineOffsets`,u.lineOffsets],[`output.drawRecord`,u.drawRecord]])r&&i(r,[`uint32`],`${n} ${t}`);if(e.parameters&&(i(e.parameters,[`uint32`,`float32`],`${n} parameters`),e.parameters.length<4))throw Error(`${n} parameters must contain at least 4 rows`);if(l<1)throw Error(`${n} must have at least one edge`);if(e.nodePositions.length<1)throw Error(`${n} nodePositions must contain at least one node`);if(e.targetNodes.length!==l)throw Error(`${n} targetNodes length must equal sourceNodes length`);if(e.edgeMask&&e.edgeMask.length!==l)throw Error(`${n} edgeMask length must equal the edge count`);if(a(n,`pointsPerEdge`,r,2,64),a(n,`maximumIterations`,o,1,64),a(n,`densityResolution`,c,q),u.paths.length!==l*r)throw Error(`${n} output.paths length must equal edgeCount * pointsPerEdge`);if(u.lineOffsets&&u.lineOffsets.length!==l+1)throw Error(`${n} output.lineOffsets length must equal edgeCount + 1`);if(u.drawRecord&&u.drawRecord.length!==4)throw Error(`${n} output.drawRecord must contain exactly four rows`);if(this.fixedPointExponent=J(l*r),this.fixedPointExponent<K)throw Error(`${n} edgeCount * pointsPerEdge is too large for 32-bit density`);let d=[u.paths,u.lineOffsets,u.drawRecord].filter(e=>e!==void 0).map(e=>e.buffer),f=[e.nodePositions,e.sourceNodes,e.targetNodes,e.edgeMask,e.parameters].filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(d).size!==d.length||d.some(e=>f.includes(e)))throw Error(`${n} outputs must use separate buffers from each other and from inputs`)}getCommandNodes(t){let{id:i,props:a,maximumIterations:s,pointsPerEdge:c,densityResolution:l,edgeCount:u}=this,{output:d}=a;if(e(i,t,[a.nodePositions,a.sourceNodes,a.targetNodes,a.edgeMask,a.parameters,d.paths,d.lineOffsets,d.drawRecord]),e(i,t,[],{work:u*c*8,density:l*l*4,iterationInfo:s*64*4,gate:s*9*4}),l*l*Uint32Array.BYTES_PER_ELEMENT>t.device.limits.maxStorageBufferBindingSize)throw Error(`${i} density grid exceeds maxStorageBufferBindingSize`);let f={edgeCount:u,vertexCount:a.nodePositions.length,pointsPerEdge:c,densityResolution:l,fixedPointExponent:this.fixedPointExponent,iterationCount:s,boxPadding:H,geographic:this.spatialContext.coordinateSpace===`longitude-latitude`},p={positions:a.nodePositions,sources:a.sourceNodes,targets:a.targetNodes,mask:a.edgeMask},m=r(t,`${i}-box`,`uint32`,4),h=r(t,`${i}-density`,`uint32`,l*l);o(t.device,i,`work`,u*c*8,`lower pointsPerEdge or split the edges`);let g=r(t,`${i}-work`,`float32x2`,u*c),_=r(t,`${i}-iteration-info`,`uint32`,s*64),v=a.parameters?r(t,`${i}-gate`,`uint32`,s*9,n.STORAGE|n.INDIRECT):void 0,y=[P(t,{id:`${i}-box-reset`,boxKeys:m}),F(t,{id:`${i}-box`,constants:f,inputs:p,boxKeys:m}),I(t,{id:`${i}-initialize`,constants:f,inputs:p,boxKeys:m,work:g})];y.push(T(t,{id:`${i}-iteration-info`,iterationInfo:_,iterationCount:s})),v&&a.parameters&&y.push(N(t,{id:`${i}-gate`,constants:f,parameters:a.parameters,gate:v}));for(let e=0;e<s;e++){let n={constants:f,iterationInfo:t.createDataView(_.buffer,{format:`uint32`,length:1,byteOffset:_.byteOffset+4*e*64}),parameters:a.parameters,work:g,density:h,gate:v&&{view:v,iteration:e}};y.push(L(t,{id:`${i}-clear-${e}`,density:h,gate:n.gate}),R(t,{id:`${i}-splat-${e}`,...n}),z(t,{id:`${i}-update-${e}`,...n}))}return y.push(B(t,{id:`${i}-finalize`,constants:f,inputs:p,boxKeys:m,work:g,paths:d.paths})),(d.lineOffsets||d.drawRecord)&&y.push(V(t,{id:`${i}-indices`,constants:f,startIndices:d.lineOffsets,drawRecord:d.drawRecord})),y}},Z=140,Q=16,$=15,te=f({parameters:_,caption:`The 140 largest commuter flows in the Chicago extract, bundled into shared corridors.`,createInputs:({world:e})=>{let t=g(p(e),e),n=m.slice(0,Z),r=Uint32Array.from(n,([e])=>e),i=Uint32Array.from(n,([,e])=>e),a=new Uint32Array(280);for(let e=0;e<Z;e++)a.set([r[e],i[e]],e*2);let o=new Uint32Array(4200),s=0;for(let e=0;e<Z;e++)for(let t=0;t<15;t++)o.set([e*Q+t,e*Q+t+1],s++*2);return{positions:t,sources:r,targets:i,pairs:a,pathPairs:o}},build(e,t){let n=e.upload(`nodes`,`float32x2`,t.positions),r=e.allocate(`paths`,`float32x2`,2240),i=e.allocate(`parameters`,`float32`,5);return e.graph.add(new X({nodePositions:n.view,sourceNodes:e.upload(`sources`,`uint32`,t.sources).view,targetNodes:e.upload(`targets`,`uint32`,t.targets).view,pointsPerEdge:Q,maximumIterations:$,parameters:i.view,output:{paths:r.view}})),{outputs:{nodes:n,straight:e.upload(`straight`,`uint32x2`,t.pairs),paths:r,pathPairs:e.upload(`path-pairs`,`uint32x2`,t.pathPairs)},update:e=>i.write(Y({activeIterations:e.activeIterations,kernelRadius:e.kernelRadius,smoothing:e.smoothing},`float32`))}},interaction:h(_,{kind:`time`,hint:`scrub the bundling`,x:`activeIterations`,y:`kernelRadius`,decorate:e=>({readout:`iteration ${e.activeIterations} / ${$} · kernelRadius ${e.kernelRadius.toFixed(3)}`})}),draw:({outputs:e})=>[{layer:`segments`,context:!0,nodes:e.nodes,edges:e.straight,radius:.8},{layer:`segments`,nodes:e.paths,edges:e.pathPairs,radius:1.2,color:[1,.78,.25,.55]}]});export{te as default};
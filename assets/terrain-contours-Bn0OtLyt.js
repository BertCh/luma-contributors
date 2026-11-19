import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{c as t,d as n,o as r,r as i,t as a,u as o}from"./kernel-C3tfQOn4.js";import{t as s}from"./gpu-scan-B4NNTfF9.js";import{d as c,f as l,i as u,n as d,o as f,r as p,s as m,u as h}from"./raster-utils-CztuVGLY.js";import{o as g}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as _,i as v,x as y}from"./gpu-contributor-utils-BIwLQON5.js";import{c as b,n as x,r as S,u as C}from"./raster-band-utils-BcTxc8_Y.js";import{t as w}from"./define-run-B-brkCe4.js";import{d as T,s as E}from"./pointer-overlay-Qh8qflcA.js";import{t as D}from"./pixel-to-world-cNchXLzW.js";import{a as O,i as k,l as A,m as j,u as M}from"./terrain-Cc9dW1uq.js";import{O as N}from"./entry-preview-D70TdSbK.js";var P=class{id;width;height;input;level;cases;segmentCounts;cellCount;constructor(e){this.id=e.id??`gpu-raster-contour-classifier`,this.width=e.width,this.height=e.height,this.input=e.input,this.level=e.level,this.cases=e.cases,this.segmentCounts=e.segmentCounts,this.cellCount=L(this),R(this.cases,this.cellCount,`${this.id} cases`),R(this.segmentCounts,this.cellCount,`${this.id} segmentCounts`),z(this.id,this.input.storage.values.buffer.graph,[this.cases,this.segmentCounts]),B(this.id,[this.cases,this.segmentCounts],[this.input.storage.values,...this.input.validity?[this.input.validity]:[],...U(this.level)?[U(this.level)]:[]])}addToGraph(e){let t=U(this.level);if(H(this.id,e,[this.input.storage.values,this.cases,this.segmentCounts,...this.input.validity?[this.input.validity]:[],...t?[t]:[]]),this.cellCount===0)return;let[n,r]=p(e.device,this.width-1,this.height-1,this.id),i=[{buffer:this.input.storage.values,usage:`storage-read`},{buffer:this.cases,usage:`storage-write`},{buffer:this.segmentCounts,usage:`storage-write`}];this.input.validity&&i.push({buffer:this.input.validity,usage:`storage-read`}),t&&i.push({buffer:t,usage:`storage-read`}),I(e,{id:this.id,resources:i,source:W(this,t),bindings:[{name:`sourceValues`,view:this.input.storage.values,writable:!1},{name:`cellCases`,view:this.cases,writable:!0},{name:`cellSegmentCounts`,view:this.segmentCounts,writable:!0},...this.input.validity?[{name:`sourceValidity`,view:this.input.validity,writable:!1}]:[],...t?[{name:`levelValues`,view:t,writable:!1}]:[]],dispatch:[n,r]})}},F=class{id;width;height;input;level;vertices;segmentCount;overflow;requiredSegmentCount;draw;drawCommandIndex;capacity;metadata;cellCount;constructor(e){if(this.id=e.id??`gpu-raster-contours`,this.width=e.width,this.height=e.height,this.input=e.input,this.level=e.level,this.vertices=e.vertices,this.segmentCount=e.segmentCount,this.overflow=e.overflow,this.requiredSegmentCount=e.requiredSegmentCount,this.draw=e.draw,this.drawCommandIndex=e.drawCommandIndex??0,this.metadata=e.metadata,this.cellCount=L(this),this.capacity=e.capacity??this.vertices.length/2,n(this.vertices,[`float32x2`],`${this.id} vertices`),this.vertices.length%2!=0)throw Error(`${this.id} vertices must contain an even number of positions`);if(!Number.isSafeInteger(this.capacity)||this.capacity<0||this.capacity>this.vertices.length/2)throw Error(`${this.id} capacity must fit in the available vertex pairs`);if(l(this.segmentCount,`uint32`,1,`${this.id} segmentCount`),l(this.overflow,`uint32`,1,`${this.id} overflow`),this.requiredSegmentCount&&l(this.requiredSegmentCount,`uint32`,1,`${this.id} requiredSegmentCount`),this.metadata&&(c(this.metadata,`${this.id} metadata`),this.metadata.width!==this.width||this.metadata.height!==this.height))throw Error(`${this.id} metadata dimensions must match the raster grid`);if(!Number.isSafeInteger(this.drawCommandIndex)||this.drawCommandIndex<0)throw Error(`${this.id} drawCommandIndex must be a non-negative integer`);if(!this.draw&&this.drawCommandIndex!==0)throw Error(`${this.id} drawCommandIndex requires a draw command`);this.draw&&V(this.id,this.draw,this.drawCommandIndex);let t=[this.vertices,this.segmentCount,this.overflow,...this.requiredSegmentCount?[this.requiredSegmentCount]:[],...this.draw?[this.draw.words]:[]];z(this.id,this.input.storage.values.buffer.graph,t),B(this.id,t,[this.input.storage.values,...this.input.validity?[this.input.validity]:[],...U(this.level)?[U(this.level)]:[]])}addToGraph(e){let t=U(this.level);if(H(this.id,e,[this.input.storage.values,this.vertices,this.segmentCount,this.overflow,...this.input.validity?[this.input.validity]:[],...t?[t]:[],...this.requiredSegmentCount?[this.requiredSegmentCount]:[],...this.draw?[this.draw.words]:[]]),this.cellCount>0&&(e.device.limits.maxComputeInvocationsPerWorkgroup<256||e.device.limits.maxComputeWorkgroupSizeX<256))throw Error(`${this.id} scan exceeds device workgroup limits`);let n=i(e,`${this.id}-cases`,`uint32`,this.cellCount),r=i(e,`${this.id}-segment-counts`,`uint32`,this.cellCount),a=i(e,`${this.id}-segment-offsets`,`uint32`,this.cellCount);new P({id:`${this.id}-classify`,width:this.width,height:this.height,input:this.input,level:this.level,cases:n,segmentCounts:r}).addToGraph(e),e.add(new s({id:`${this.id}-scan`,input:r,output:a,mode:`exclusive`})),this.cellCount>0&&this.capacity>0&&this.addScatterPass(e,n,a,t),this.addSummaryPass(e,r,a)}addScatterPass(e,t,n,r){let[i,a]=p(e.device,this.width-1,this.height-1,`${this.id}-scatter`),o=[{buffer:this.input.storage.values,usage:`storage-read`},{buffer:t,usage:`storage-read`},{buffer:n,usage:`storage-read`},{buffer:this.vertices,usage:`storage-write`}];r&&o.push({buffer:r,usage:`storage-read`}),I(e,{id:`${this.id}-scatter`,resources:o,source:G(this,t,n,r),bindings:[{name:`sourceValues`,view:this.input.storage.values,writable:!1},{name:`cellCases`,view:t,writable:!1},{name:`cellOffsets`,view:n,writable:!1},{name:`outputVertices`,view:this.vertices,writable:!0},...r?[{name:`levelValues`,view:r,writable:!1}]:[]],dispatch:[i,a]})}addSummaryPass(e,t,n){let r=this.cellCount>0,i=[...r?[{name:`cellSegmentCounts`,view:t,writable:!1}]:[],...r?[{name:`cellOffsets`,view:n,writable:!1}]:[],{name:`outputSegmentCount`,view:this.segmentCount,writable:!0},{name:`outputOverflow`,view:this.overflow,writable:!0},...this.requiredSegmentCount?[{name:`outputRequiredCount`,view:this.requiredSegmentCount,writable:!0}]:[],...this.draw?[{name:`drawCommandWords`,view:this.draw.words,writable:!0}]:[]];I(e,{id:`${this.id}-publish`,resources:i.map(({view:e,writable:t})=>({buffer:e,usage:t?`storage-write`:`storage-read`})),source:K(this,t,n,i),bindings:i,dispatch:[1,1]})}};function I(e,t){e.addComputePass({id:t.id,resources:t.resources,compile:({device:e})=>{let n=t.bindings.map((e,t)=>({name:e.name,type:e.writable?`storage`:`read-only-storage`,group:0,location:t})),i=new a(e,{id:t.id,source:t.source,shaderLayout:{bindings:n}});return{encode:({computePass:e,getBuffer:n})=>{let a={};for(let e of t.bindings)a[e.name]=r(e.view,n);let o=a;i.dispatch(e,{bindings:o,x:t.dispatch[0],y:t.dispatch[1]})},destroy:()=>i.destroy()}}})}function L(e){let t=e.id??`gpu-raster-contours`;if(!Number.isSafeInteger(e.width)||e.width<=0||!Number.isSafeInteger(e.height)||e.height<=0)throw Error(`${t} dimensions must be positive integers`);let n=e.width*e.height;if(!Number.isSafeInteger(n)||n>4294967295)throw Error(`${t} pixel count must fit in uint32`);let r=(e.width-1)*(e.height-1);if(r*2>4294967295)throw Error(`${t} maximum segment count must fit in uint32`);if(e.input.storage.kind!==`buffer`)throw Error(`${t} requires a buffer-backed raster band`);let i=h(e.input,e,`${t} input`);if(u(e.input.scale??1),u(e.input.offset??0),typeof e.level==`number`){if(!Number.isFinite(e.level))throw Error(`${t} contour level must be finite`);u(e.level)}else if(l(e.level,`float32`,1,`${t} level`),e.level.buffer.graph!==i)throw Error(`${t} contour level must belong to the same graph`);return r}function R(e,t,n){if(o(e,n),e.length!==t)throw Error(`${n} must contain exactly one value per raster cell`)}function z(e,t,n){if(n.some(e=>e.buffer.graph!==t))throw Error(`${e} resources must belong to the same graph`)}function B(e,t,n){let r=new Set(t.map(e=>e.buffer));if(r.size!==t.length||n.some(e=>r.has(e.buffer)))throw Error(`${e} input and output resources must use separate buffers`)}function V(e,t,n){if(t.type!==`draw`||t.recordByteLength!==16)throw Error(`${e} requires non-indexed two-vertex indirect draw records`);if(n>=t.capacity)throw Error(`${e} drawCommandIndex exceeds the indirect draw capacity`);if(o(t.words,`${e} indirect command words`),t.words.buffer!==t.buffer||t.words.length<t.capacity*4)throw Error(`${e} indirect command words must span every draw record`)}function H(e,t,n){for(let r of n){if(r.buffer.graph!==t)throw Error(`${e} resources must belong to the target graph`);d(t.device,r,`${e} ${r.buffer.id}`)}}function U(e){return typeof e==`number`?void 0:e}function W(e,n){let r=e.input.validity,i=r?4:3,a=r?`@group(0) @binding(3) var<storage, read> sourceValidity: array<u32>;
const VALIDITY_OFFSET: u32 = ${t(r)}u;`:``,o=n?`@group(0) @binding(${i}) var<storage, read> levelValues: array<f32>;
const LEVEL_OFFSET: u32 = ${t(n)}u;`:``,s=n?`levelValues[LEVEL_OFFSET]`:u(e.level),c=e.input.noDataValue!==void 0&&!Number.isNaN(e.input.noDataValue)?` && rawValue != ${f(e.input.noDataValue,e.input.format)}`:``,l=r?` && sourceValidity[VALIDITY_OFFSET + index] != 0u`:``,d=e.input.format===`float32`?` && isFiniteValue(rawValue)`:``;return`
const WIDTH: u32 = ${e.width}u;
const CELL_WIDTH: u32 = ${e.width-1}u;
const CELL_HEIGHT: u32 = ${e.height-1}u;
const SOURCE_OFFSET: u32 = ${t(e.input.storage.values)}u;
const CASE_OFFSET: u32 = ${t(e.cases)}u;
const COUNT_OFFSET: u32 = ${t(e.segmentCounts)}u;
@group(0) @binding(0) var<storage, read> sourceValues: array<${m(e.input.format)}>;
@group(0) @binding(1) var<storage, read_write> cellCases: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellSegmentCounts: array<u32>;
${a}
${o}

fn isFiniteValue(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

fn isValidSample(index: u32, rawValue: ${m(e.input.format)}, value: f32) -> bool {
  return isFiniteValue(value)${d}${c}${l};
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= CELL_WIDTH || globalId.y >= CELL_HEIGHT) { return; }
  let cellIndex = globalId.y * CELL_WIDTH + globalId.x;
  let topLeftIndex = globalId.y * WIDTH + globalId.x;
  let topRightIndex = topLeftIndex + 1u;
  let bottomLeftIndex = topLeftIndex + WIDTH;
  let bottomRightIndex = bottomLeftIndex + 1u;
  let topLeftRaw = sourceValues[SOURCE_OFFSET + topLeftIndex];
  let topRightRaw = sourceValues[SOURCE_OFFSET + topRightIndex];
  let bottomRightRaw = sourceValues[SOURCE_OFFSET + bottomRightIndex];
  let bottomLeftRaw = sourceValues[SOURCE_OFFSET + bottomLeftIndex];
  let topLeft = f32(topLeftRaw) * ${u(e.input.scale??1)} + ${u(e.input.offset??0)};
  let topRight = f32(topRightRaw) * ${u(e.input.scale??1)} + ${u(e.input.offset??0)};
  let bottomRight = f32(bottomRightRaw) * ${u(e.input.scale??1)} + ${u(e.input.offset??0)};
  let bottomLeft = f32(bottomLeftRaw) * ${u(e.input.scale??1)} + ${u(e.input.offset??0)};
  let contourLevel = ${s};
  if (!isFiniteValue(contourLevel) ||
      !isValidSample(topLeftIndex, topLeftRaw, topLeft) ||
      !isValidSample(topRightIndex, topRightRaw, topRight) ||
      !isValidSample(bottomRightIndex, bottomRightRaw, bottomRight) ||
      !isValidSample(bottomLeftIndex, bottomLeftRaw, bottomLeft)) {
    cellCases[CASE_OFFSET + cellIndex] = 0u;
    cellSegmentCounts[COUNT_OFFSET + cellIndex] = 0u;
    return;
  }

  let caseIndex = select(0u, 1u, topLeft >= contourLevel) |
    select(0u, 2u, topRight >= contourLevel) |
    select(0u, 4u, bottomRight >= contourLevel) |
    select(0u, 8u, bottomLeft >= contourLevel);
  var encodedCase = caseIndex;
  var segmentCount = select(1u, 0u, caseIndex == 0u || caseIndex == 15u);
  if (caseIndex == 5u || caseIndex == 10u) {
    let magnitude = max(1.0, max(abs(contourLevel),
      max(max(abs(topLeft), abs(topRight)), max(abs(bottomRight), abs(bottomLeft)))));
    let topLeftDelta = topLeft / magnitude - contourLevel / magnitude;
    let topRightDelta = topRight / magnitude - contourLevel / magnitude;
    let bottomRightDelta = bottomRight / magnitude - contourLevel / magnitude;
    let bottomLeftDelta = bottomLeft / magnitude - contourLevel / magnitude;
    let determinant = topLeftDelta * bottomRightDelta - topRightDelta * bottomLeftDelta;
    let topRightPair = determinant > 0.0 || (determinant == 0.0 && caseIndex == 5u);
    encodedCase = caseIndex | select(0u, 16u, topRightPair);
    segmentCount = 2u;
  }
  cellCases[CASE_OFFSET + cellIndex] = encodedCase;
  cellSegmentCounts[COUNT_OFFSET + cellIndex] = segmentCount;
}`}function G(e,n,r,i){let a=i?`@group(0) @binding(4) var<storage, read> levelValues: array<f32>;
const LEVEL_OFFSET: u32 = ${t(i)}u;`:``,o=i?`levelValues[LEVEL_OFFSET]`:u(e.level),s=e.metadata?.pixelInterpretation===`point`?`0.0`:`0.5`;return`
const WIDTH: u32 = ${e.width}u;
const CELL_WIDTH: u32 = ${e.width-1}u;
const CELL_HEIGHT: u32 = ${e.height-1}u;
const SEGMENT_CAPACITY: u32 = ${e.capacity}u;
const SOURCE_OFFSET: u32 = ${t(e.input.storage.values)}u;
const CASE_OFFSET: u32 = ${t(n)}u;
const CELL_OFFSET: u32 = ${t(r)}u;
const VERTEX_OFFSET: u32 = ${t(e.vertices)}u;
@group(0) @binding(0) var<storage, read> sourceValues: array<${m(e.input.format)}>;
@group(0) @binding(1) var<storage, read> cellCases: array<u32>;
@group(0) @binding(2) var<storage, read> cellOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputVertices: array<f32>;
${a}

fn isFiniteValue(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

fn getInterpolation(first: f32, second: f32, contourLevel: f32) -> f32 {
  let directNumerator = contourLevel - first;
  let directDenominator = second - first;
  if (isFiniteValue(directNumerator) && isFiniteValue(directDenominator) &&
      directDenominator != 0.0) {
    return clamp(directNumerator / directDenominator, 0.0, 1.0);
  }
  let magnitude = max(1.0, max(abs(contourLevel), max(abs(first), abs(second))));
  let numerator = contourLevel / magnitude - first / magnitude;
  let denominator = second / magnitude - first / magnitude;
  if (denominator == 0.0) { return 0.5; }
  return clamp(numerator / denominator, 0.0, 1.0);
}

fn getEdgePosition(
  edge: u32,
  topLeft: f32,
  topRight: f32,
  bottomRight: f32,
  bottomLeft: f32,
  contourLevel: f32,
  origin: vec2<f32>
) -> vec2<f32> {
  switch edge {
    case 0u: {
      return origin + vec2<f32>(getInterpolation(topLeft, topRight, contourLevel), 0.0);
    }
    case 1u: {
      return origin + vec2<f32>(1.0, getInterpolation(topRight, bottomRight, contourLevel));
    }
    case 2u: {
      return origin + vec2<f32>(getInterpolation(bottomLeft, bottomRight, contourLevel), 1.0);
    }
    default: {
      return origin + vec2<f32>(0.0, getInterpolation(topLeft, bottomLeft, contourLevel));
    }
  }
}

fn writeSegment(
  segmentIndex: u32,
  firstEdge: u32,
  secondEdge: u32,
  topLeft: f32,
  topRight: f32,
  bottomRight: f32,
  bottomLeft: f32,
  contourLevel: f32,
  origin: vec2<f32>
) {
  if (segmentIndex >= SEGMENT_CAPACITY) { return; }
  let first = getEdgePosition(
    firstEdge, topLeft, topRight, bottomRight, bottomLeft, contourLevel, origin
  );
  let second = getEdgePosition(
    secondEdge, topLeft, topRight, bottomRight, bottomLeft, contourLevel, origin
  );
  let outputIndex = VERTEX_OFFSET + segmentIndex * 4u;
  outputVertices[outputIndex] = first.x;
  outputVertices[outputIndex + 1u] = first.y;
  outputVertices[outputIndex + 2u] = second.x;
  outputVertices[outputIndex + 3u] = second.y;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= CELL_WIDTH || globalId.y >= CELL_HEIGHT) { return; }
  let cellIndex = globalId.y * CELL_WIDTH + globalId.x;
  let encodedCase = cellCases[CASE_OFFSET + cellIndex];
  let caseIndex = encodedCase & 15u;
  if (caseIndex == 0u || caseIndex == 15u) { return; }
  let segmentIndex = cellOffsets[CELL_OFFSET + cellIndex];
  if (segmentIndex >= SEGMENT_CAPACITY) { return; }
  let topLeftIndex = globalId.y * WIDTH + globalId.x;
  let topLeft = f32(sourceValues[SOURCE_OFFSET + topLeftIndex]) * ${u(e.input.scale??1)} + ${u(e.input.offset??0)};
  let topRight = f32(sourceValues[SOURCE_OFFSET + topLeftIndex + 1u]) * ${u(e.input.scale??1)} + ${u(e.input.offset??0)};
  let bottomLeft = f32(sourceValues[SOURCE_OFFSET + topLeftIndex + WIDTH]) * ${u(e.input.scale??1)} + ${u(e.input.offset??0)};
  let bottomRight = f32(sourceValues[SOURCE_OFFSET + topLeftIndex + WIDTH + 1u]) * ${u(e.input.scale??1)} + ${u(e.input.offset??0)};
  let contourLevel = ${o};
  let origin = vec2<f32>(f32(globalId.x) + ${s}, f32(globalId.y) + ${s});
  var firstEdge = 0u;
  var secondEdge = 0u;
  switch caseIndex {
    case 1u, 14u: { firstEdge = 3u; secondEdge = 0u; }
    case 2u, 13u: { firstEdge = 0u; secondEdge = 1u; }
    case 3u, 12u: { firstEdge = 3u; secondEdge = 1u; }
    case 4u, 11u: { firstEdge = 1u; secondEdge = 2u; }
    case 6u, 9u: { firstEdge = 0u; secondEdge = 2u; }
    case 7u, 8u: { firstEdge = 2u; secondEdge = 3u; }
    default: {
      let topRightPair = (encodedCase & 16u) != 0u;
      firstEdge = select(3u, 0u, topRightPair);
      secondEdge = select(0u, 1u, topRightPair);
      let thirdEdge = select(1u, 2u, topRightPair);
      let fourthEdge = select(2u, 3u, topRightPair);
      writeSegment(segmentIndex + 1u, thirdEdge, fourthEdge, topLeft, topRight,
        bottomRight, bottomLeft, contourLevel, origin);
    }
  }
  writeSegment(segmentIndex, firstEdge, secondEdge, topLeft, topRight,
    bottomRight, bottomLeft, contourLevel, origin);
}`}function K(e,n,r,i){let a=i.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.writable?`read_write`:`read`}> ${e.name}: array<u32>;`).join(`
`),o=e.cellCount>0?`cellOffsets[${t(r)}u + ${e.cellCount-1}u] + cellSegmentCounts[${t(n)}u + ${e.cellCount-1}u]`:`0u`,s=e.requiredSegmentCount?`outputRequiredCount[${t(e.requiredSegmentCount)}u] = requiredCount;`:``,c=``;if(e.draw){let n=t(e.draw.words)+e.drawCommandIndex*4;c=`drawCommandWords[${n}u] = 2u;
  drawCommandWords[${n+1}u] = clampedCount;
  drawCommandWords[${n+2}u] = 0u;
  drawCommandWords[${n+3}u] = 0u;`}return`
${a}

@compute @workgroup_size(1)
fn main() {
  let requiredCount = ${o};
  let clampedCount = min(requiredCount, ${e.capacity}u);
  outputSegmentCount[${t(e.segmentCount)}u] = clampedCount;
  outputOverflow[${t(e.overflow)}u] = select(0u, 1u, requiredCount > ${e.capacity}u);
  ${s}
  ${c}
}`}var q=class{id;props;requiredHalo=1;constructor(e){this.id=e.id??`terrain-contours`,this.props=e;let{id:t}=this;C(t,e.grid);let{overflow:n}=e.output.status;if(e.levels.length<1)throw Error(`${t} requires at least one contour level`);for(let[n,r]of e.levels.entries())Y(t,n,r);if(_(n,[`uint32`],`${t} output.status.overflow`),n.length<1)throw Error(`${t} output.status.overflow must contain one uint32 row`);if([...x(e.elevation),...e.levels.flatMap(e=>[e.vertices,e.status.count,e.status.requiredCount,e.status.overflow,typeof e.level==`number`?void 0:e.level])].filter(e=>e!==void 0).map(e=>e.buffer).includes(n.buffer))throw Error(`${t} outputs must not share buffers with each other or with inputs`)}getCommandNodes(t){e(t,this.id,this.props,{"contour scratch":4*this.props.grid.width*this.props.grid.height});let{id:n,props:r}=this,{width:a,height:o}=r.grid,{overflow:s}=r.output.status;b(n,t,r.elevation,[]),y(n,t,[s,...r.levels.flatMap(e=>[e.vertices,e.status.count,e.status.requiredCount,e.status.overflow,typeof e.level==`number`?void 0:e.level,e.draw?.words])]);let c=S(t,n,r.elevation,r.grid,!1),l=[...c.nodes];for(let[e,u]of r.levels.entries()){let d=u.draw!==void 0&&!J(u),f=u.status.overflow??i(t,`${n}-level-${e}-overflow`,`uint32`,1);l.push(...v(t,()=>new F({id:`${n}-level-${e}`,width:a,height:o,input:c.band,level:u.level,vertices:u.vertices,segmentCount:u.status.count,overflow:f,requiredSegmentCount:u.status.requiredCount,draw:d?void 0:u.draw,drawCommandIndex:d?void 0:u.drawCommandIndex,capacity:u.capacity,metadata:r.grid}).addToGraph(t))),d&&u.draw&&l.push(X(t,n,e,u,u.draw)),l.push(g(t,{id:`${n}-overflow-${e}`,operation:`GPUTerrainContours`,variant:`overflow`,bindings:[{name:`levelOverflow`,view:f,type:`u32`,access:`read`},{name:`outputOverflow`,view:s,type:`u32`,access:`read_write`}],invocationCount:1,body:`let previous = ${e===0?`0u`:`outputOverflow[outputOverflowOffset]`};
  outputOverflow[outputOverflowOffset] =
    select(0u, 1u, previous != 0u || levelOverflow[levelOverflowOffset] != 0u);`}))}return l}};function J(e){return(e.drawLayout??`instanced`)===`instanced`&&(e.verticesPerInstance??2)===2}function Y(e,t,n){let r=`${e} level ${t}`,{draw:i}=n;if(!i){if(n.drawCommandIndex!==void 0||n.drawLayout!==void 0||n.verticesPerInstance!==void 0)throw Error(`${r} drawCommandIndex, drawLayout and verticesPerInstance require draw`);return}let a=n.drawLayout??`instanced`;if(a!==`instanced`&&a!==`line-list`)throw Error(`${r} drawLayout must be 'instanced' or 'line-list'`);if(n.verticesPerInstance!==void 0){if(a!==`instanced`)throw Error(`${r} verticesPerInstance requires drawLayout 'instanced'`);if(!Number.isSafeInteger(n.verticesPerInstance)||n.verticesPerInstance<1)throw Error(`${r} verticesPerInstance must be a positive integer`)}let o=n.drawCommandIndex??0;if(!Number.isSafeInteger(o)||o<0)throw Error(`${r} drawCommandIndex must be a non-negative integer`);if(i.type!==`draw`||i.recordByteLength!==16)throw Error(`${r} draw must be a non-indexed indirect draw record view`);if(o>=i.capacity)throw Error(`${r} drawCommandIndex exceeds the indirect draw capacity`);if(_(i.words,[`uint32`],`${r} draw words`),i.words.buffer!==i.buffer||i.words.length<i.capacity*4)throw Error(`${r} draw words must span every draw record`)}function X(e,t,n,r,i){let a=r.capacity??Math.floor(r.vertices.length/2),o=(r.drawCommandIndex??0)*4,s=r.drawLayout===`line-list`;return g(e,{id:`${t}-level-${n}-draw`,operation:`GPUTerrainContours`,variant:`draw`,bindings:[{name:`segmentCount`,view:r.status.count,type:`u32`,access:`read`},{name:`drawWords`,view:i.words,type:`u32`,access:`read_write`}],invocationCount:1,body:`let count = min(segmentCount[segmentCountOffset], ${a}u);
  let base = drawWordsOffset + ${o}u;
  drawWords[base] = ${s?`2u * count`:`${r.verticesPerInstance??2}u`};
  drawWords[base + 1u] = ${s?`1u`:`count`};
  drawWords[base + 2u] = 0u;
  drawWords[base + 3u] = 0u;`})}var Z=10,Q=2400;function $(e,t){let n=E(N,`firstLevel`),r=((e-n.min)%t+t)%t;return T(n,n.min+r)}var ee=w({parameters:N,caption:`Elevation in; one contour line set per level out, written straight to GPU buffers.`,createInputs:({world:e})=>({...k(e),world:e}),build(e,t){let{grid:n,band:r,elevation:i}=j(e,t),a=e.allocate(`levels`,`float32`,Z),o=e.allocate(`overflow`,`uint32`,1),s=Array.from({length:Z},(t,n)=>{let r=e.allocate(`pixels-${n}`,`float32x4`,Q),i=e.allocate(`count-${n}`,`uint32`,1),o=e.allocate(`required-${n}`,`uint32`,1);return{pixels:r,count:i,props:{level:e.graph.createDataView(a.view.buffer,{format:`float32`,length:1,byteOffset:n*4}),vertices:e.graph.createDataView(r.view.buffer,{format:`float32x2`,length:Q*2}),status:{count:i.view,requiredCount:o.view}}}});return e.graph.add(new q({grid:n,elevation:r,levels:s.map(e=>e.props),output:{status:{overflow:o.view}}})),{outputs:{elevation:i,levels:s.map((n,r)=>({vertices:D(e,`vertices-${r}`,n.pixels,t,t.world),count:n.count}))},update:e=>a.write(Float32Array.from({length:Z},(t,n)=>e.firstLevel+n*e.interval))}},interaction:{kind:`level`,hint:`trace a contour`,rest:({inputs:e,world:t})=>{let n=E(N,`firstLevel`),r=E(N,`interval`).default,i=[e.columns>>1,e.rows>>1],a=1/0;for(let t=0;t<e.rows;t++)for(let o=0;o<e.columns;o++){let s=e.elevation[t*e.columns+o];if($(s,r)!==n.default)continue;let c=Math.hypot(o-e.columns/2,(t-e.rows/2)*1.5);c<a&&(a=c,i=[o,t])}let[o,s]=M(e,t,i[0],i[1]);return[o/t.width,1-s/t.height]},resolve(e,{inputs:t,values:n,world:r}){let[i,a]=A(t,r,e.x,e.y),o=t.elevation[a*t.columns+i],s=$(o,n.interval),c=Math.round((o-s)/n.interval);return{values:{firstLevel:s},focus:c,glyphs:[{shape:`dot`,at:M(t,r,i,a)}],readout:`level ${Math.round(s+c*n.interval)} m · interval ${n.interval} m`}}},draw:({inputs:e,outputs:t,interaction:n})=>[O(e,t.elevation),...t.levels.map((e,t)=>({layer:`lines`,segments:e.vertices,count:e.count,radius:t===n.focus?3:1.4,color:[1-t*.06,.8-t*.05,.35+t*.05,1]}))]});export{ee as default};
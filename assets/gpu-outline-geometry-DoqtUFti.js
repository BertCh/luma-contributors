import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-scan-B4NNTfF9.js";import{g as n,v as r,y as i}from"./spatial-context-DwZEd4EZ.js";import{E as a,S as o,T as s,f as c,g as l,p as u,v as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{c as p,o as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as h}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as g}from"./validate-outline-output-passes-DIcqH6Pw.js";var _=`GPUOutlineGeometry`,v=4294967295,y=2147483647,b=2**-126,x=34028234663852886e22,S=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),metricsByCoordinateSpace:Object.freeze({planar:Object.freeze([`native`]),"longitude-latitude":Object.freeze([`great-circle`])}),measure:`distance`});function C(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Outline geometry target must hold 4 elements`);if(!Number.isFinite(e.distance)||e.distance<0)throw Error(`Outline geometry distance must be a non-negative finite number`);return i(`GPUOutlineGeometry`,e.spatialContext,e.distance),l(_,`distance`,Math.fround(e.distance),0,x),t.set([e.distance,0,0,0]),t}function w(e){return 3*(e+2)}var T=class{id;props;spatialContext;radius;joinSegments;verticesPerInput;boundaryOffsets;estimatedCommandNodeCount;constructor(e){this.id=e.id??`outline-geometry`,this.props=e,this.estimatedCommandNodeCount=e.output.layout===`compact`?11:1,u(this.id,this.estimatedCommandNodeCount),this.spatialContext=r(this.id,e.spatialContext,S),this.radius=n(this.spatialContext),l(this.id,`spatialContext sphere radius`,Math.fround(this.radius),b,x),this.joinSegments=e.joinSegments??16,this.verticesPerInput=w(this.joinSegments);let{id:t}=this;if(this.boundaryOffsets=e.geometryType===`lines`?e.lineOffsets:e.geometryType===`polygons`?e.ringOffsets:void 0,![`points`,`lines`,`polygons`].includes(e.geometryType))throw Error(`${t} geometryType must be 'points', 'lines' or 'polygons'`);if(c(t,`joinSegments`,this.joinSegments,3),a(e.positions,[`float32x2`],`${t} positions`),e.positions.length<1)throw Error(`${t} needs at least one position`);if(e.geometryType!==`points`){let n=e.geometryType===`lines`?`lineOffsets`:`ringOffsets`;if(!this.boundaryOffsets)throw Error(`${t} ${n} is required`);if(a(this.boundaryOffsets,[`uint32`],`${t} ${n}`),this.boundaryOffsets.length<2)throw Error(`${t} ${n} needs two rows`);e.geometryType===`polygons`&&(a(e.featureOffsets,[`uint32`],`${t} featureOffsets`),a(e.polygonOffsets,[`uint32`],`${t} polygonOffsets`))}if(a(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 float32 values`);a(e.output.positions,[`float32x2`],`${t} output.positions`);let i=e.positions.length*this.verticesPerInput;o(t,`positions.length * verticesPerInput`,i,v);let d=e.output;if(d.layout===`compact`){s(t,d.positions.length>0&&d.positions.length%3==0,`output.positions.length must be positive and divisible by three`),a(d.vertexOffsets,[`uint32`],`${t} output.vertexOffsets`),s(t,d.vertexOffsets.length===e.positions.length+1,`output.vertexOffsets.length must equal positions.length + 1`),d.sourceRows&&(a(d.sourceRows,[`uint32`],`${t} output.sourceRows`),s(t,d.sourceRows.length===d.positions.length,`output.sourceRows.length must equal output.positions.length`));for(let[e,n]of Object.entries(d.status))n&&(a(n,[`uint32`],`${t} output.status.${e}`),c(t,`output.status.${e}.length`,n.length,1));g(t,{positions:d.positions,sourceRows:d.sourceRows,vertexOffsets:d.vertexOffsets,...d.status},[[`positions`,`sourceRows`],[`vertexOffsets`,`count`,`requiredCount`,`overflow`]])}else s(t,d.layout===void 0||d.layout===`fixed-stride`,`output.layout must be 'compact' or 'fixed-stride'`),s(t,d.positions.length===i,`output.positions must hold ${i} rows`);o(t,`output.positions.length`,d.positions.length,y),f(t,this.getOutputViews(),[e.positions,this.boundaryOffsets,e.parameters])}getCommandNodes(n){let{props:r,id:i}=this,a=r.output;if(h(n,i,{positions:r.positions,boundaryOffsets:this.boundaryOffsets,parameters:r.parameters,...Object.fromEntries(this.getOutputViews().map((e,t)=>[`output.${t}`,e]))}),a.layout!==`compact`)return[this.createGenerationNode(n)];let o=(r.positions.length+1)*4;d(n.device,i,`compact counts and scan scratch`,o,`reduce input vertex count`),c(i,`compact scratch byteLength`,o,1,n.device.limits.maxBufferSize);let s=e(n,`${i}-counts`,`uint32`,r.positions.length+1),l=e(n,`${i}-starts`,`uint32`,r.positions.length+1),u=[this.createGenerationNode(n,{counts:s}),...new t({id:`${i}-scan`,input:s,output:l}).getCommandNodes(n),this.createGenerationNode(n,{starts:l})];return u.push(m(n,{id:`${i}-publish`,operation:_,invocationCount:a.vertexOffsets.length,bindings:[{name:`starts`,view:l,type:`u32`,access:`read`},{name:`offsets`,view:a.vertexOffsets,type:`u32`,access:`read_write`},{name:`count`,view:a.status.count,type:`u32`,access:`read_write`},{name:`overflow`,view:a.status.overflow,type:`u32`,access:`read_write`},...a.status.requiredCount?[{name:`required`,view:a.status.requiredCount,type:`u32`,access:`read_write`}]:[]],body:`let requiredVertices = starts[startsOffset + ${r.positions.length}u];
offsets[offsetsOffset + index] = min(starts[startsOffset + index], ${a.positions.length}u);
if (index == 0u) {
 count[countOffset] = min(requiredVertices, ${a.positions.length}u);
 overflow[overflowOffset] = select(0u, 1u, requiredVertices > ${a.positions.length}u);
 ${a.status.requiredCount?`required[requiredOffset] = requiredVertices;`:``}
}`})),u}getOutputViews(){let{output:e}=this.props;return e.layout===`compact`?[e.positions,e.vertexOffsets,e.sourceRows,e.status.count,e.status.requiredCount,e.status.overflow]:[e.positions]}createGenerationNode(e,t={}){let{props:n,id:r,joinSegments:i}=this,a=n.output.layout===`compact`?n.output:void 0,{counts:o,starts:s}=t,c=n.geometryType!==`points`,l=[{name:`positions`,view:n.positions,type:`f32`,access:`read`}];c&&this.boundaryOffsets&&l.push({name:`lineOffsets`,view:this.boundaryOffsets,type:`u32`,access:`read`}),l.push({name:`parameters`,view:n.parameters,type:`f32`,access:`read`}),o?l.push({name:`counts`,view:o,type:`u32`,access:`read_write`}):l.push({name:`outputPositions`,view:n.output.positions,type:`f32`,access:`read_write`}),s&&l.push({name:`starts`,view:s,type:`u32`,access:`read`}),a?.sourceRows&&!o&&l.push({name:`sourceRows`,view:a.sourceRows,type:`u32`,access:`read_write`});let u=this.spatialContext.coordinateSpace===`longitude-latitude`;return m(e,{id:`${r}-${o?`count`:`generate`}`,operation:_,variant:n.geometryType,bindings:l,invocationCount:n.positions.length+ +!!o,declarations:`
const LINE_COUNT: u32 = ${this.boundaryOffsets?this.boundaryOffsets.length-1:0}u;
const ROW_COUNT: u32 = ${n.positions.length}u;
const JOIN_SEGMENTS: u32 = ${i}u;
const VERTICES_PER_INPUT: u32 = ${this.verticesPerInput}u;
const RADIUS: f32 = ${p(this.radius)};
const DEGREES_PER_RADIAN: f32 = 57.29577951308232;
const TAU: f32 = 6.283185307179586;

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

${o?``:`fn writeVertex(slot: u32, base: u32, point: vec2<f32>, owner: u32) {
  let destination = base + slot;
  if (destination >= ${n.output.positions.length}u) { return; }
  let coordinateOffset = outputPositionsOffset + 2u * destination;
  outputPositions[coordinateOffset] = point.x;
  outputPositions[coordinateOffset + 1u] = point.y;
  ${a?.sourceRows?`sourceRows[sourceRowsOffset + destination] = owner;`:``}
}`}

// Moves p by distance d along the metric direction (east, north).
fn offsetPoint(p: vec2<f32>, direction: vec2<f32>, d: f32) -> vec2<f32> {
  ${u?`let latitudeScale = d / RADIUS * DEGREES_PER_RADIAN;
  let longitudeScale = latitudeScale / max(cos(radians(p.y)), 1e-6);
  return p + vec2<f32>(direction.x * longitudeScale, direction.y * latitudeScale);`:`return p + direction * d;`}
}

// Unit metric direction of the segment a to b, or zero for a zero-length segment.
fn getDirection(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  ${u?`let middleLatitude = radians(0.5 * (a.y + b.y));
  let metric = vec2<f32>((b.x - a.x) * cos(middleLatitude), b.y - a.y);`:`let metric = b - a;`}
  let metricLength = length(metric);
  if (metricLength <= 0.0) {
    return vec2<f32>(0.0);
  }
  return metric / metricLength;
}
${c?E:``}`,body:`${o?`if (index == ROW_COUNT) { counts[countsOffset + index] = 0u; return; }`:``}
  let distance = max(parameters[parametersOffset], 0.0);
  let center = getPosition(index);
  var nextRow = ROW_COUNT;
  ${c?D(n.geometryType):``}
  var direction = vec2<f32>(0.0);
  if (nextRow < ROW_COUNT) { direction = getDirection(center, getPosition(nextRow)); }
  let hasQuad = dot(direction, direction) > 0.0;
  ${o?`counts[countsOffset + index] = select(0u, 3u * JOIN_SEGMENTS + select(0u, 6u, hasQuad), distance > 0.0);`:`
  ${a?`if (!(distance > 0.0)) { return; }`:``}
  let base = ${s?`starts[startsOffset + index]`:`index * VERTICES_PER_INPUT`};
  var rimA = offsetPoint(center, vec2<f32>(1.0, 0.0), distance);
  for (var corner = 0u; corner < JOIN_SEGMENTS; corner++) {
    let angle = TAU * f32(corner + 1u) / f32(JOIN_SEGMENTS);
    let rimB = offsetPoint(center, vec2<f32>(cos(angle), sin(angle)), distance);
    writeVertex(3u * corner, base, center, index);
    writeVertex(3u * corner + 1u, base, rimA, index);
    writeVertex(3u * corner + 2u, base, rimB, index);
    rimA = rimB;
  }
  var quad = array<vec2<f32>, 6>(center, center, center, center, center, center);
  if (hasQuad) {
    let next = getPosition(nextRow);
    let normal = vec2<f32>(-direction.y, direction.x);
    let aLeft = offsetPoint(center, normal, distance);
    let aRight = offsetPoint(center, -normal, distance);
    let bRight = offsetPoint(next, -normal, distance);
    let bLeft = offsetPoint(next, normal, distance);
    quad = array<vec2<f32>, 6>(aLeft, aRight, bRight, aLeft, bRight, bLeft);
  }
  ${a?`if (!hasQuad) { return; }`:``}
  for (var vertex = 0u; vertex < 6u; vertex++) {
    writeVertex(3u * JOIN_SEGMENTS + vertex, base, quad[vertex], index);
  }`}`})}},E=`
const NO_LINE: u32 = 0xffffffffu;

fn findLine(row: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || low > LINE_COUNT) {
    return NO_LINE;
  }
  return low - 1u;
}
`;function D(e){return`let line = findLine(index);
  if (line != NO_LINE) {
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT);
    if (index + 1u < lineEnd) {
      nextRow = index + 1u;
    }${e===`polygons`?` else if (lineEnd - lineStart >= 2u) {
      nextRow = lineStart;
    }`:``}
  }`}export{w as i,T as n,C as r,E as t};
import{s as e,v as t}from"./spatial-context-DwZEd4EZ.js";import{r as n,s as r,u as i}from"./wgsl-common-DfyXiYB0.js";import{E as a,f as o,p as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{t as l}from"./gpu-prediction-utils-DjZFSzGW.js";import{i as u,n as d,t as f}from"./graph-chunk-groups-BB8ACpQO.js";import{a as p,i as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as h}from"./define-run-B-brkCe4.js";import{Vs as g}from"./entry-preview-D70TdSbK.js";import{t as _}from"./channel-D8wtkSDN.js";import{t as v}from"./grid-CuWZ3T2e.js";import{a as y,i as b,o as x,t as S}from"./interpolation-CWOuK2ek.js";var C=Object.freeze({...e,measure:`distance`}),w=2,T=class{id;estimatedCommandNodeCount;chunkPlan;spatialContext;props;maximumVertexCount;maximumNeighborCount;constructor(e){if(this.props=e,this.id=e.id??`natural-neighbor`,this.spatialContext=t(this.id,e.spatialContext,C),this.maximumVertexCount=e.maximumVertexCount??64,this.maximumNeighborCount=e.maximumNeighborCount??32,o(this.id,`maximumNeighborCount`,this.maximumNeighborCount,3),o(this.id,`maximumVertexCount`,this.maximumVertexCount,3),this.maximumVertexCount>64||e.positions.length>16777216||e.output.weights&&e.output.weights.format!==(e.triangles?`float32x2`:`float32`)||!e.triangles&&e.positions.length>this.maximumNeighborCount||this.maximumNeighborCount>64||!!e.triangles!=!!e.queryTriangles||e.triangles&&(e.triangles.length%6!=0||e.queryTriangles.length!==e.queries.length)||e.positions.length!==e.values.length||e.output.values.length<e.queries.length||e.output.status.length<e.queries.length||e.output.weights&&e.output.weights.length<e.queries.length*(e.triangles?this.maximumNeighborCount:e.positions.length))throw Error(`${this.id} invalid capacities`);this.chunkPlan=d(this.id,{view:e.queries},{queries:{view:e.queries},queryTriangles:e.queryTriangles?{view:e.queryTriangles}:void 0,results:{view:e.output.values,allowPackedCapacity:!0},statuses:{view:e.output.status,allowPackedCapacity:!0},weights:e.output.weights?{view:e.output.weights,rowsPerQuery:e.triangles?this.maximumNeighborCount:e.positions.length,allowPackedCapacity:!0}:void 0});for(let[t,n]of Object.entries({positions:e.positions,sampleValues:e.values,triangles:e.triangles}))n&&a(n,[n.format],`${this.id} ${t}`);m(this.id,e.output),c(this.id,Object.values(e.output),[e.positions,e.values,e.triangles,e.queries,e.queryTriangles]),this.estimatedCommandNodeCount=this.chunkPlan.nonemptyChunkCount,s(this.id,this.estimatedCommandNodeCount)}getCommandNodes(e){let{props:t,id:n}=this;return p(e,n,{positions:t.positions,sampleValues:t.values,triangles:t.triangles,queries:t.queries,queryTriangles:t.queryTriangles,...t.output}),u(e,n,5+Number(!!t.triangles)*2+Number(!!t.output.weights),w),this.chunkPlan.chunks.filter(e=>e.length>0).map(r=>this.getQueryNode(e,{...t,queries:r.views.queries,queryTriangles:r.views.queryTriangles,output:{values:r.views.results,status:r.views.statuses,weights:r.views.weights}},this.chunkPlan.chunks.length===1?n:`${n}-chunk-${r.chunkIndex}`))}getQueryNode(e,t,a){let{id:o,maximumVertexCount:s}=this,c=t.triangles?this.maximumNeighborCount:t.positions.length,u=c,d=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`values`,view:t.values,type:`f32`,access:`read`},{name:`queries`,view:t.queries,type:`f32`,access:`read`},{name:`results`,view:t.output.values,type:`f32`,access:`read_write`},{name:`statuses`,view:t.output.status,type:`u32`,access:`read_write`}];t.triangles&&d.push({name:`triangles`,view:t.triangles,type:`u32`,access:`read`},{name:`queryTriangles`,view:t.queryTriangles,type:`u32`,access:`read`}),t.output.weights&&d.push({name:`weights`,view:t.output.weights,type:`f32`,access:`read_write`});let p=e=>t.triangles?`sampleIndices[${e}]`:e;return f(e,{id:a,pipelineId:o,operation:`GPUNaturalNeighbor`,variant:`sibson-area`,bindings:d,invocationCount:t.queries.length,workgroupSize:w,declarations:`${l({positions:2,queries:2},[`results`])}\n${n}\n${i}\n${r}
const SAMPLE_CAPACITY: u32 = ${u}u;
const VERTEX_CAPACITY: u32 = ${s}u;
${t.output.weights?`fn writeWeight(queryIndex: u32, slot: u32, sampleIndex: u32, weight: f32) { ${t.triangles?`weights[weightsOffset + queryIndex * SAMPLE_CAPACITY * 2u + slot * 2u] = f32(sampleIndex); weights[weightsOffset + queryIndex * SAMPLE_CAPACITY * 2u + slot * 2u + 1u] = weight;`:`weights[weightsOffset + queryIndex * SAMPLE_CAPACITY + slot] = weight;`} }`:``}
fn crossProduct(first: vec2<f32>, second: vec2<f32>) -> f32 { return first.x * second.y - first.y * second.x; }
var<private> polygon: array<vec2<f32>, ${s}>;
var<private> clipped: array<vec2<f32>, ${s}>;
var<private> polygonCount: u32;
fn clipPolygon(normal: vec2<f32>, limit: f32) -> bool {
  var count = 0u;
  for (var vertex = 0u; vertex < polygonCount; vertex++) {
    let start = polygon[vertex]; let finish = polygon[(vertex + 1u) % polygonCount];
    let startDistance = dot(normal, start) - limit; let finishDistance = dot(normal, finish) - limit;
    if (startDistance <= 0.0) { if (count >= VERTEX_CAPACITY) { return false; } clipped[count] = start; count++; }
    if ((startDistance < 0.0 && finishDistance > 0.0) || (startDistance > 0.0 && finishDistance < 0.0)) {
      if (count >= VERTEX_CAPACITY) { return false; }
      clipped[count] = start + (finish - start) * (startDistance / (startDistance - finishDistance)); count++;
    }
  }
  polygonCount = count; for (var vertex = 0u; vertex < count; vertex++) { polygon[vertex] = clipped[vertex]; }
  return true;
}
fn polygonArea() -> f32 {
  var area = 0.0; for (var vertex = 0u; vertex < polygonCount; vertex++) { area += crossProduct(polygon[vertex], polygon[(vertex + 1u) % polygonCount]); }
  return abs(area) * 0.5;
}`,body:`
write_results(index, vec4<f32>(getNaN(), 0.0, 0.0, 0.0)); statuses[statusesOffset + index] = 1u;
${t.output.weights?`for (var sampleIndex = 0u; sampleIndex < SAMPLE_CAPACITY; sampleIndex++) { weights[weightsOffset + index * SAMPLE_CAPACITY * ${t.triangles?2:1}u + sampleIndex * ${t.triangles?2:1}u] = ${t.triangles?`-1.0`:`0.0`}; ${t.triangles?`weights[weightsOffset + index * SAMPLE_CAPACITY * 2u + sampleIndex * 2u + 1u] = 0.0;`:``} }`:``}
let query = read_queries(index); if (!isFiniteValue(query.x) || !isFiniteValue(query.y)) { return; }
${t.triangles?`var sampleIndices: array<u32, ${Math.max(c,1)}>;`:``}
var sampleCount = ${t.triangles?0:u}u;
${t.triangles?`
let seedTriangle = queryTriangles[queryTrianglesOffset + index];
if (seedTriangle >= ${t.triangles.length/6}u) { return; }
var triangleQueue: array<u32, ${this.maximumNeighborCount*4}>;
triangleQueue[0] = seedTriangle; var queueCount = 1u; var queueCursor = 0u;
loop {
  if (queueCursor >= queueCount) { break; }
  let triangle = triangleQueue[queueCursor]; queueCursor++;
  let triangleStart = trianglesOffset + triangle * 6u;
  let firstIndex = triangles[triangleStart]; let secondIndex = triangles[triangleStart + 1u]; let thirdIndex = triangles[triangleStart + 2u];
  if (firstIndex >= ${t.positions.length}u || secondIndex >= ${t.positions.length}u || thirdIndex >= ${t.positions.length}u) { statuses[statusesOffset + index] = 2u; return; }
  var first = read_positions(firstIndex) - query; var second = read_positions(secondIndex) - query; var third = read_positions(thirdIndex) - query;
  let triangleScale = max(max(length(first), length(second)), length(third));
  if (!(triangleScale > 0.0) || !isFiniteValue(triangleScale)) { statuses[statusesOffset + index] = 2u; return; }
  first /= triangleScale; second /= triangleScale; third /= triangleScale;
  let orientation = crossProduct(second - first, third - first);
  if (abs(orientation) < 1e-9) { statuses[statusesOffset + index] = 2u; return; }
  if (triangle == seedTriangle) {
    let sideFirst = crossProduct(second - first, -first) * sign(orientation);
    let sideSecond = crossProduct(third - second, -second) * sign(orientation);
    let sideThird = crossProduct(first - third, -third) * sign(orientation);
    if (min(min(sideFirst, sideSecond), sideThird) < -1e-6) { return; }
  }
  let inCircle = (dot(first, first) * crossProduct(second, third) - dot(second, second) * crossProduct(first, third) + dot(third, third) * crossProduct(first, second)) * sign(orientation);
  if (inCircle < -1e-7) { continue; }
  for (var corner = 0u; corner < 3u; corner++) {
    let sampleIndex = triangles[triangleStart + corner]; var duplicate = false;
    for (var previous = 0u; previous < sampleCount; previous++) { duplicate = duplicate || sampleIndices[previous] == sampleIndex; }
    if (!duplicate) {
      if (sampleCount >= SAMPLE_CAPACITY) { statuses[statusesOffset + index] = 3u; return; }
      sampleIndices[sampleCount] = sampleIndex; sampleCount++;
    }
    let adjacent = triangles[triangleStart + 3u + corner];
    if (adjacent == 0xffffffffu) { continue; }
    if (adjacent >= ${t.triangles.length/6}u) { statuses[statusesOffset + index] = 2u; return; }
    var visited = false; for (var previous = 0u; previous < queueCount; previous++) { visited = visited || triangleQueue[previous] == adjacent; }
    if (!visited) {
      if (queueCount >= ${this.maximumNeighborCount*4}u) { statuses[statusesOffset + index] = 3u; return; }
      triangleQueue[queueCount] = adjacent; queueCount++;
    }
  }
}
`:``}
var points: array<vec2<f32>, ${Math.max(u,1)}>;
var scale = 0.0; var hitCount = 0u; var hitValue = 0.0;
for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
  let point = read_positions(${p(`sampleIndex`)}); let value = values[valuesOffset + ${p(`sampleIndex`)}];
  if (!isFiniteValue(point.x) || !isFiniteValue(point.y) || !isFiniteValue(value)) { return; }
  points[sampleIndex] = point - query; scale = max(scale, length(points[sampleIndex]));
  if (all(point == query)) { hitCount++; }
}
if (hitCount > 0u) {
  for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) { if (all(points[sampleIndex] == vec2<f32>(0.0))) { hitValue += values[valuesOffset + ${p(`sampleIndex`)}] / f32(hitCount); } }
  if (!isFiniteValue(hitValue)) { statuses[statusesOffset + index] = 2u; return; }
  write_results(index, vec4<f32>(hitValue, 0.0, 1.0, f32(hitCount))); statuses[statusesOffset + index] = 0u;
  ${t.output.weights?`for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) { if (all(points[sampleIndex] == vec2<f32>(0.0))) { writeWeight(index, sampleIndex, ${p(`sampleIndex`)}, 1.0 / f32(hitCount)); } }`:``}
  return;
}
if (sampleCount < 3u || !(scale > 0.0)) { statuses[statusesOffset + index] = 2u; return; }
for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) { points[sampleIndex] /= scale; }
var noncollinear = false; var edgeFirst = 0u; var edgeSecond = 0u; var edgeLength = 3.402823e38;
for (var first = 0u; first < sampleCount; first++) {
  for (var second = first + 1u; second < sampleCount; second++) {
    let direction = points[second] - points[first]; let magnitude = length(direction);
    if (magnitude <= 1e-7) { statuses[statusesOffset + index] = 2u; return; }
    var positive = false; var negative = false;
    for (var other = 0u; other < sampleCount; other++) {
      let side = crossProduct(direction, points[other] - points[first]); positive = positive || side > 1e-7; negative = negative || side < -1e-7;
    }
    noncollinear = noncollinear || positive || negative;
    if (positive && negative) { continue; }
    let side = crossProduct(direction, -points[first]);
    if ((positive && side < -1e-7) || (negative && side > 1e-7)) { return; }
    if (abs(side) <= 1e-7 && dot(-points[first], -points[second]) <= 0.0 && magnitude < edgeLength) { edgeFirst = first; edgeSecond = second; edgeLength = magnitude; }
  }
}
if (!noncollinear) { statuses[statusesOffset + index] = 2u; return; }
if (edgeLength < 3.402823e38) {
  let fraction = length(points[edgeFirst]) / edgeLength;
  write_results(index, vec4<f32>(mix(values[valuesOffset + ${p(`edgeFirst`)}], values[valuesOffset + ${p(`edgeSecond`)}], fraction), 0.0, 1.0, 2.0)); statuses[statusesOffset + index] = 0u;
  ${t.output.weights?`writeWeight(index, edgeFirst, ${p(`edgeFirst`)}, 1.0 - fraction); writeWeight(index, edgeSecond, ${p(`edgeSecond`)}, fraction);`:``}
  return;
}
var queryPolygon: array<vec2<f32>, ${s}>; var vertexCount = 0u;
for (var first = 0u; first < sampleCount; first++) {
  for (var second = first + 1u; second < sampleCount; second++) {
    let firstPoint = points[first]; let secondPoint = points[second]; let determinant = crossProduct(firstPoint, secondPoint);
    if (abs(determinant) <= 1e-9) { continue; }
    let firstLimit = dot(firstPoint, firstPoint) * 0.5; let secondLimit = dot(secondPoint, secondPoint) * 0.5;
    let vertex = vec2<f32>(firstLimit * secondPoint.y - secondLimit * firstPoint.y, firstPoint.x * secondLimit - secondPoint.x * firstLimit) / determinant;
    var feasible = true; for (var other = 0u; other < sampleCount; other++) { feasible = feasible && dot(points[other], vertex) <= dot(points[other], points[other]) * 0.5 + 1e-6; }
    if (!feasible) { continue; }
    var duplicate = false; for (var previous = 0u; previous < vertexCount; previous++) { duplicate = duplicate || distance(vertex, queryPolygon[previous]) <= 1e-6; }
    if (duplicate) { continue; }
    if (vertexCount >= VERTEX_CAPACITY) { statuses[statusesOffset + index] = 3u; return; }
    queryPolygon[vertexCount] = vertex; vertexCount++;
  }
}
if (vertexCount < 3u) { statuses[statusesOffset + index] = 2u; return; }
for (var vertex = 1u; vertex < vertexCount; vertex++) {
  // Vertices are offsets from the query, so x can be -0.0: guardedAtan2 (contract section 11).
  let point = queryPolygon[vertex]; let angle = guardedAtan2(point.y, point.x); var insertion = vertex;
  loop { if (insertion == 0u) { break; } let previous = queryPolygon[insertion - 1u]; if (guardedAtan2(previous.y, previous.x) <= angle) { break; } queryPolygon[insertion] = previous; insertion--; }
  queryPolygon[insertion] = point;
}
polygonCount = vertexCount; for (var vertex = 0u; vertex < vertexCount; vertex++) { polygon[vertex] = queryPolygon[vertex]; }
let queryArea = polygonArea(); if (!(queryArea > 1e-12) || !isFiniteValue(queryArea)) { statuses[statusesOffset + index] = 2u; return; }
var areas: array<f32, ${Math.max(u,1)}>; var areaSum = 0.0; var prediction = 0.0; var contributing = 0u;
for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
  polygonCount = vertexCount; for (var vertex = 0u; vertex < vertexCount; vertex++) { polygon[vertex] = queryPolygon[vertex]; }
  for (var other = 0u; other < sampleCount; other++) {
    if (other == sampleIndex) { continue; }
    if (!clipPolygon(points[other] - points[sampleIndex], (dot(points[other], points[other]) - dot(points[sampleIndex], points[sampleIndex])) * 0.5)) { statuses[statusesOffset + index] = 3u; return; }
    if (polygonCount < 3u) { break; }
  }
  let area = polygonArea(); areas[sampleIndex] = area; areaSum += area; prediction += area * values[valuesOffset + ${p(`sampleIndex`)}]; if (area > 0.0) { contributing++; }
}
if (!(areaSum > 0.0) || abs(areaSum - queryArea) > queryArea * 1e-3 || !isFiniteValue(prediction)) { statuses[statusesOffset + index] = 2u; return; }
write_results(index, vec4<f32>(prediction / areaSum, queryArea * scale * scale, 1.0, f32(contributing))); statuses[statusesOffset + index] = 0u;
${t.output.weights?`for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) { writeWeight(index, sampleIndex, ${p(`sampleIndex`)}, areas[sampleIndex] / areaSum); }`:``}`})}},E=h({parameters:g,caption:`Sibson weights from the Voronoi cells of the samples (dots); the surface stops at the hull of the samples (white). Move one sample.`,createInputs:({random:e,world:t})=>{let n=b(e,t,60),[r]=x(n.positions,t.width/2,.5,1);return D(n.positions,2,0,r),D(n.values,1,0,r),{grid:v(t,36),samples:n,carried:0,positions:Float32Array.from(n.positions)}},build(e,{grid:t,samples:n,positions:r},i){let a=t.columns*t.rows,o=i.sampleCount,s=e.allocate(`results`,`float32x4`,a),c=e.upload(`positions`,`float32x2`,r.slice(0,o*2));e.graph.add(new T({positions:c.view,values:e.upload(`values`,`float32`,n.values.slice(0,o)).view,queries:e.upload(`queries`,`float32x2`,t.centers).view,maximumVertexCount:i.maximumVertexCount,maximumNeighborCount:64,output:{values:s.view,status:e.allocate(`status`,`uint32`,a).view}}));let l=e.upload(`hull`,`float32x4`,S(r,o,64));return{outputs:{hull:l,prediction:_(e,s,0,`prediction`),positions:c,sampleValues:e.upload(`sample-values`,`float32`,n.values),sampleCount:o},update:()=>{c.write(r.slice(0,o*2)),l.write(S(r,o,64))}}},rebuildOn:[`maximumVertexCount`,`sampleCount`],interaction:{kind:`drag`,hint:`carry a sample`,rest:({inputs:e,toUnit:t})=>t([e.samples.positions[e.carried*2],e.samples.positions[e.carried*2+1]]),resolve(e,{inputs:t}){let n=Math.min(t.grid.window[2],Math.max(0,e.x)),r=Math.min(t.grid.window[3],Math.max(0,e.y));return t.positions[t.carried*2]=n,t.positions[t.carried*2+1]=r,{rerun:!0,glyphs:[{shape:`ring`,at:[n,r],radius:.035}],readout:`sample ${t.carried} = ${t.samples.values[t.carried].toFixed(2)} at ${n.toFixed(2)}, ${r.toFixed(2)}`}}},draw:({inputs:e,outputs:t})=>[{layer:`raster`,values:t.prediction,columns:e.grid.columns,rows:e.grid.rows,range:[0,1]},{layer:`lines`,segments:t.hull,radius:1.4,color:[1,1,1,.8]},...y(t.positions,t.sampleValues,t.sampleCount)]});function D(e,t,n,r){for(let i=0;i<t;i++){let a=e[n*t+i];e[n*t+i]=e[r*t+i],e[r*t+i]=a}}export{E as default};
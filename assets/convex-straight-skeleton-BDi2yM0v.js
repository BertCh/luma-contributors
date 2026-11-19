import{a as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t}from"./validate-outline-output-passes-DIcqH6Pw.js";import{r as n}from"./refinement-storage-validation-D9BhQGNW.js";import{v as r}from"./spatial-context-DwZEd4EZ.js";import{E as i,f as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{c as s,o as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as l}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{A as u}from"./pointer-overlay-Qh8qflcA.js";import{t as d}from"./define-run-B-brkCe4.js";import{s as f,v as p,y as m}from"./polygon-set-CHja50mr.js";import{a as h,c as g,d as _,f as v}from"./pointer-support-C2rPRVFO.js";import{s as y}from"./shape-segments-DMcyF4V-.js";import{wo as b}from"./entry-preview-D70TdSbK.js";var x=class{id;props;spatialContext;maximumVertices;maximumIterations;segmentCapacity;constructor(e){if(this.id=e.id??`convex-straight-skeleton`,this.props=e,n(this.id,`offsetDistance`,e.offsetDistance),n(this.id,`eventTolerance`,e.eventTolerance??1e-5,!0),this.spatialContext=r(this.id,e.spatialContext),this.maximumVertices=e.maximumVertices??32,this.maximumIterations=e.maximumIterations??this.maximumVertices,this.segmentCapacity=e.segmentCapacity??this.maximumVertices*2,a(this.id,`maximumVertices`,this.maximumVertices,3,64),a(this.id,`maximumIterations`,this.maximumIterations,0,128),a(this.id,`segmentCapacity`,this.segmentCapacity,1),!(Number.isFinite(e.offsetDistance)&&e.offsetDistance>=0&&Number.isFinite(e.eventTolerance??1e-5)&&(e.eventTolerance??1e-5)>0))throw Error(`Invalid wavefront parameters`);i(e.positions,[`float32x2`],this.id),i(e.ringOffsets,[`uint32`],this.id),i(e.output.segments,[`float32x4`],this.id),i(e.output.sourceEdges,[`uint32x2`],this.id),i(e.output.offsetPositions,[`float32x2`],this.id),i(e.output.status,[`uint32x4`],this.id);let s=e.ringOffsets.length-1;if(e.output.segments.length!==s*this.segmentCapacity||e.output.sourceEdges.length!==e.output.segments.length||e.output.offsetPositions.length!==s*this.maximumVertices||e.output.status.length!==s)throw Error(`Skeleton output capacity mismatch`);t(this.id,e.output,[Object.keys(e.output)]),o(this.id,Object.values(e.output),[e.positions,e.ringOffsets])}getCommandNodes(t){let{props:n,id:r,maximumVertices:i,segmentCapacity:a,maximumIterations:o}=this;return e(t,r,Object.fromEntries([n.positions,n.ringOffsets,...Object.values(n.output)].map((e,t)=>[`binding.${t}`,e]))),[c(t,{id:`${r}-events`,operation:`GPUConvexStraightSkeleton`,variant:`convex-wavefront`,invocationCount:n.ringOffsets.length-1,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`offsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`segments`,view:n.output.segments,type:`f32`,access:`read_write`},{name:`sourceEdges`,view:n.output.sourceEdges,type:`u32`,access:`read_write`},{name:`offsetPositions`,view:n.output.offsetPositions,type:`f32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`}],declarations:`const MAXIMUM_VERTICES: u32 = ${i}u;
const SEGMENT_CAPACITY: u32 = ${a}u;
const REQUESTED_DISTANCE: f32 = ${s(n.offsetDistance)};
const EVENT_TOLERANCE: f32 = ${s(n.eventTolerance??1e-5)};
${l(`point`,`positions`)}
fn cross(left: vec2f, right: vec2f) -> f32 { return left.x * right.y - left.y * right.x; }
fn intersection(left: vec3f, right: vec3f, time: f32) -> vec2f {
  let determinant = cross(left.xy, right.xy);
  return vec2f((left.z + time) * right.y - left.y * (right.z + time), left.x * (right.z + time) - (left.z + time) * right.x) / determinant;
}
fn writeArc(polygon: u32, slot: u32, start: vec2f, end: vec2f, left: u32, right: u32) {
  if (slot >= SEGMENT_CAPACITY) { return; }
  let destination = polygon * SEGMENT_CAPACITY + slot;
  segments[segmentsOffset + destination * 4u] = start.x; segments[segmentsOffset + destination * 4u + 1u] = start.y;
  segments[segmentsOffset + destination * 4u + 2u] = end.x; segments[segmentsOffset + destination * 4u + 3u] = end.y;
  sourceEdges[sourceEdgesOffset + destination * 2u] = left; sourceEdges[sourceEdgesOffset + destination * 2u + 1u] = right;
}`,body:`let first = min(offsets[offsetsOffset + index], ${n.positions.length}u);
let last = min(offsets[offsetsOffset + index + 1u], ${n.positions.length}u);
var count = last - first;
if (count > 1u && all(point(first) == point(last - 1u))) { count--; }
var flags = 0u; var required = 0u; var offsetCount = 0u; var offsetWritten = false;
if (count < 3u || count > MAXIMUM_VERTICES) { flags = 1u; }
var edges: array<vec3f, MAXIMUM_VERTICES>;
var edgeIds: array<u32, MAXIMUM_VERTICES>;
var births: array<vec2f, MAXIMUM_VERTICES>;
var orientation = 0.0;
if (flags == 0u) {
  orientation = sign(cross(point(first + 1u) - point(first), point(first + 2u) - point(first + 1u)));
  for (var vertex = 0u; vertex < count; vertex++) {
    let nextVertexIndex = (vertex + 1u) % count;
    let current = point(first + vertex); let next = point(first + nextVertexIndex);
    if (any((bitcast<vec2u>(current) & vec2u(0x7f800000u)) == vec2u(0x7f800000u))) { flags |= 8u; }
    let tangent = next - current;
    for (var otherVertexIndex = 0u; otherVertexIndex < count; otherVertexIndex++) {
      // The edge endpoints are on its own supporting line by definition. With
      // slanted float32 edges, recomputing that cross product can round slightly
      // negative and falsely reject an otherwise strictly convex polygon.
      if (otherVertexIndex == vertex || otherVertexIndex == nextVertexIndex) { continue; }
      if (cross(tangent, point(first + otherVertexIndex) - current) * orientation < 0.0) { flags |= 1u; }
    }
    let following = point(first + (vertex + 2u) % count) - next;
    if (cross(tangent, following) * orientation <= 0.0) { flags |= 1u; }
    let normal = orientation * vec2f(-tangent.y, tangent.x) / length(tangent);
    edges[vertex] = vec3f(normal, dot(normal, current)); edgeIds[vertex] = first + vertex; births[vertex] = current;
  }
}
var time = 0.0; var iterations = 0u;
loop {
  if (flags != 0u || count < 3u) { break; }
  var eventTimes: array<f32, MAXIMUM_VERTICES>;
  var eventPositions: array<vec2f, MAXIMUM_VERTICES>;
  var nextTime = 3.0e38;
  for (var vertex = 0u; vertex < count; vertex++) {
    let previous = (vertex + count - 1u) % count; let next = (vertex + 1u) % count;
    let currentPoint = intersection(edges[previous], edges[vertex], time);
    let nextPoint = intersection(edges[vertex], edges[next], time);
    let currentVelocity = intersection(vec3f(edges[previous].xy, 0.0), vec3f(edges[vertex].xy, 0.0), 1.0);
    let nextVelocity = intersection(vec3f(edges[vertex].xy, 0.0), vec3f(edges[next].xy, 0.0), 1.0);
    let tangent = orientation * vec2f(edges[vertex].y, -edges[vertex].x);
    let shrinkRate = dot(nextVelocity - currentVelocity, tangent);
    eventTimes[vertex] = 3.0e38;
    if (shrinkRate < 0.0) { eventTimes[vertex] = max(time, time - dot(nextPoint - currentPoint, tangent) / shrinkRate); }
    nextTime = min(nextTime, eventTimes[vertex]);
  }
  if (nextTime == 3.0e38) { flags |= 8u; break; }
  if (!offsetWritten && REQUESTED_DISTANCE < nextTime) {
    offsetCount = count; offsetWritten = true;
    for (var vertex = 0u; vertex < count; vertex++) {
      let current = intersection(edges[(vertex + count - 1u) % count], edges[vertex], REQUESTED_DISTANCE);
      let destination = (index * MAXIMUM_VERTICES + vertex) * 2u;
      offsetPositions[offsetPositionsOffset + destination] = current.x; offsetPositions[offsetPositionsOffset + destination + 1u] = current.y;
    }
  }
  if (iterations >= ${o}u) { flags |= 4u; break; }
  let epsilon = EVENT_TOLERANCE * max(1.0, abs(nextTime));
  var removed: array<u32, MAXIMUM_VERTICES>; var survivors = 0u;
  for (var vertex = 0u; vertex < count; vertex++) {
    removed[vertex] = select(0u, 1u, eventTimes[vertex] <= nextTime + epsilon);
    if (removed[vertex] == 0u) { survivors++; }
    eventPositions[vertex] = intersection(edges[(vertex + count - 1u) % count], edges[vertex], nextTime);
  }
  for (var vertex = 0u; vertex < count; vertex++) {
    let previous = (vertex + count - 1u) % count;
    if (removed[previous] != 0u || removed[vertex] != 0u || survivors < 3u) {
      if (distance(births[vertex], eventPositions[vertex]) > epsilon) {
        writeArc(index, required, births[vertex], eventPositions[vertex], edgeIds[previous], edgeIds[vertex]); required++;
      }
    }
  }
  if (survivors < 3u) {
    // Simultaneous opposing collapses leave a terminal ridge (e.g. a rectangle).
    var left = eventPositions[0]; var right = left; var longest = 0.0;
    for (var vertex = 0u; vertex < count; vertex++) {
      for (var other = vertex + 1u; other < count; other++) {
        let separation = distance(eventPositions[vertex], eventPositions[other]);
        if (separation > longest) { longest = separation; left = eventPositions[vertex]; right = eventPositions[other]; }
      }
    }
    if (longest > epsilon) { writeArc(index, required, left, right, 0xffffffffu, 0xffffffffu); required++; }
    count = 0u; time = nextTime; break;
  }
  var retained: array<u32, MAXIMUM_VERTICES>; var kept = 0u;
  for (var vertex = 0u; vertex < count; vertex++) { if (removed[vertex] == 0u) { retained[kept] = vertex; kept++; } }
  // Event positions are no longer needed after arc emission. Reuse them for the next
  // births before compacting edges in place; ascending retained IDs never read an overwritten edge.
  for (var vertex = 0u; vertex < survivors; vertex++) {
    let current = retained[vertex]; let previous = retained[(vertex + survivors - 1u) % survivors];
    if ((previous + 1u) % count == current) { eventPositions[vertex] = births[current]; }
    else { eventPositions[vertex] = intersection(edges[previous], edges[current], nextTime); }
  }
  for (var vertex = 0u; vertex < survivors; vertex++) {
    edges[vertex] = edges[retained[vertex]]; edgeIds[vertex] = edgeIds[retained[vertex]]; births[vertex] = eventPositions[vertex];
  }
  count = survivors; time = nextTime; iterations++;
}
if (required > SEGMENT_CAPACITY) { flags |= 2u; }
status[statusOffset + index * 4u] = min(required, SEGMENT_CAPACITY); status[statusOffset + index * 4u + 1u] = required;
status[statusOffset + index * 4u + 2u] = offsetCount; status[statusOffset + index * 4u + 3u] = flags;`})]}},S=6,C={convexShore:!0,vertexCount:S,rows:2,columns:4},w=[.01,.02,.035,.05,.07],T=.2,E=d({parameters:b,caption:`Skeleton arcs and the inward offset of six-vertex Great Lakes hulls.`,buildCache:5,createInputs:({seed:e,world:t})=>{let n=f(u(e),t,{...C,stretch:.2});return{seed:e,world:t,set:n,restPoint:h((e,t)=>{let r=v(n,e,t);return r?.isInside?r.depth:null},.035,t)}},build(e,t,n){let r=p(e,t.set),i=t.set.count,a=e.allocate(`segments`,`float32x4`,i*12),o=e.allocate(`offset-positions`,`float32x2`,i*S),s=e.allocate(`status`,`uint32x4`,i);e.graph.add(new x({positions:r.positions.view,ringOffsets:r.ringOffsets.view,maximumVertices:S,segmentCapacity:12,offsetDistance:n.offsetDistance,output:{segments:a.view,sourceEdges:e.allocate(`source-edges`,`uint32x2`,i*12).view,offsetPositions:o.view,status:s.view}}));let c=e.allocate(`visible-arcs`,`float32x4`,a.length);y(e,{id:`visible-arcs`,invocationCount:a.length,bindings:[{name:`segments`,buffer:a,access:`read`},{name:`status`,buffer:s,access:`read`},{name:`visible`,buffer:c,access:`read_write`}],body:`let count = status[statusOffset + (index / 12u) * 4u];
        for (var component = 0u; component < 4u; component++) {
          visible[visibleOffset + index * 4u + component] = select(-9.0, segments[segmentsOffset + index * 4u + component], index % 12u < count);
        }`});let l=e.allocate(`parked-offsets`,`float32x2`,o.length);y(e,{id:`parked-offsets`,invocationCount:o.length,bindings:[{name:`status`,buffer:s,access:`read`},{name:`positions`,buffer:o,access:`read`},{name:`parked`,buffer:l,access:`read_write`}],body:`let isWritten = index % ${S}u < status[statusOffset + (index / ${S}u) * 4u + 2u];
  parked[parkedOffset + index * 2u] = select(1000000.0, positions[positionsOffset + index * 2u], isWritten);
  parked[parkedOffset + index * 2u + 1u] = select(1000000.0, positions[positionsOffset + index * 2u + 1u], isWritten);`});let d=e.allocate(`offset-edges`,`uint32x2`,o.length);return y(e,{id:`offset-edges`,invocationCount:o.length,bindings:[{name:`status`,buffer:s,access:`read`},{name:`edges`,buffer:d,access:`read_write`}],body:`let row = index / ${S}u;
        let count = status[statusOffset + row * 4u + 2u];
        let corner = index % ${S}u;
        let base = row * ${S}u;
        edges[edgesOffset + index * 2u] = select(base, index, corner < count);
        edges[edgesOffset + index * 2u + 1u] = select(base, base + (corner + 1u) % max(1u, count), corner < count);`}),{outputs:{...r,offsetPositions:l,skeleton:c,offsetEdges:d},update:e=>m(r,f(u(t.seed),t.world,{...C,stretch:e.stretch}))}},rebuildOn:[`offsetDistance`],interaction:{kind:`reach`,hint:`go deeper inside`,rest:({inputs:e,toUnit:t})=>t(e.restPoint),resolve(e,{inputs:t,values:n}){let r=v(t.set,e.x,e.y),i=r?.isInside?_(w,r.depth):w[g(r?.depth??0,T,w.length)];return{settle:{offsetDistance:i},glyphs:r?[{shape:`line`,from:[e.x,e.y],to:r.point,dashed:!0},{shape:`dot`,at:r.point}]:[],readout:i===n.offsetDistance?`offsetDistance ${i.toFixed(3)}`:`offsetDistance ${n.offsetDistance.toFixed(3)} -> ${i.toFixed(3)}`}}},draw:({inputs:e,outputs:t,theme:n})=>[{layer:`lines`,context:!0,segments:t.outline,count:e.set.outlineCount},{layer:`lines`,segments:t.skeleton,color:[...n.ink,1],radius:1.6},{layer:`segments`,nodes:t.offsetPositions,edges:t.offsetEdges,color:[1,.6,.2,1],radius:2}]});export{E as default};
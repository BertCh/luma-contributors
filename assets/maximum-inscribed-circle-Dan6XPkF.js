import{a as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t}from"./validate-outline-output-passes-DIcqH6Pw.js";import{n,r}from"./refinement-storage-validation-D9BhQGNW.js";import{r as i}from"./kernel-C3tfQOn4.js";import{v as a}from"./spatial-context-DwZEd4EZ.js";import{E as o,f as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{c as l,o as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{o as d}from"./wgsl-common-DfyXiYB0.js";import{i as f}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{A as p}from"./pointer-overlay-Qh8qflcA.js";import{t as m}from"./define-run-B-brkCe4.js";import{n as h,s as g,v as _,y as v}from"./polygon-set-CHja50mr.js";import{l as y,r as b,t as x}from"./pointer-support-C2rPRVFO.js";import{r as S}from"./shape-segments-DMcyF4V-.js";import{lo as C}from"./entry-preview-D70TdSbK.js";var w=class{id;props;spatialContext;cellCapacity;maximumIterations;constructor(e){if(this.id=e.id??`maximum-inscribed-circle`,this.props=e,r(this.id,`tolerance`,e.tolerance,!0),this.spatialContext=a(this.id,e.spatialContext),this.cellCapacity=e.cellCapacity??256,this.maximumIterations=e.maximumIterations??1024,s(this.id,`cellCapacity`,this.cellCapacity,1,4096),s(this.id,`maximumIterations`,this.maximumIterations,0,65536),!(Number.isFinite(e.tolerance)&&e.tolerance>0))throw Error(`Invalid circle tolerance`);o(e.positions,[`float32x2`],this.id);for(let t of[e.ringOffsets,e.polygonOffsets,e.featureOffsets])o(t,[`uint32`],this.id),s(this.id,`offset length`,t.length,2);for(let[t,n]of[[`circles`,`float32x4`],[`witnesses`,`float32x2`],[`status`,`uint32`]])if(o(e.output[t],[n],this.id),e.output[t].length!==e.featureOffsets.length-1)throw Error(`Circle output length mismatch`);t(this.id,e.output,[Object.keys(e.output)]),c(this.id,Object.values(e.output),[e.positions,e.ringOffsets,e.polygonOffsets,e.featureOffsets])}getCommandNodes(t){let{props:r,id:a}=this;e(t,a,Object.fromEntries([r.positions,r.ringOffsets,r.polygonOffsets,r.featureOffsets,...Object.values(r.output)].map((e,t)=>[`binding.${t}`,e]))),n(t,a,{cells:(r.featureOffsets.length-1)*this.cellCapacity*16});let o=i(t,`${a}-cells`,`float32`,(r.featureOffsets.length-1)*this.cellCapacity*4);return[u(t,{id:`${a}-search`,operation:`GPUMaximumInscribedCircle`,variant:`branch-bound`,invocationCount:r.featureOffsets.length-1,bindings:[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:r.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:r.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:r.featureOffsets,type:`u32`,access:`read`},{name:`circles`,view:r.output.circles,type:`f32`,access:`read_write`},{name:`witnesses`,view:r.output.witnesses,type:`f32`,access:`read_write`},{name:`status`,view:r.output.status,type:`u32`,access:`read_write`},{name:`cells`,view:o,type:`f32`,access:`read_write`}],declarations:`${d}
fn loadCell(base: u32, cell: u32) -> vec4f {
  let address = cellsOffset + base + cell * 4u;
  return vec4f(cells[address], cells[address + 1u], cells[address + 2u], cells[address + 3u]);
}
fn storeCell(base: u32, cell: u32, value: vec4f) {
  let address = cellsOffset + base + cell * 4u;
  cells[address] = value.x; cells[address + 1u] = value.y;
  cells[address + 2u] = value.z; cells[address + 3u] = value.w;
}
const CELL_CAPACITY: u32 = ${this.cellCapacity}u;
const MAXIMUM_ITERATIONS: u32 = ${this.maximumIterations}u;
const TOLERANCE: f32 = ${l(r.tolerance)};
const VERTEX_COUNT: u32 = ${r.positions.length}u;
const RING_COUNT: u32 = ${r.ringOffsets.length-1}u;
const POLYGON_COUNT: u32 = ${r.polygonOffsets.length-1}u;
${f(`point`,`positions`)}
fn signedDistance(query: vec2f, firstRing: u32, lastRing: u32) -> vec3f {
  var inside = false;
  var distanceSquared = 3.4028234663852886e38;
  var witness = query;
  for (var ring = firstRing; ring < lastRing; ring++) {
    let firstVertex = min(ringOffsets[ringOffsetsOffset + ring], VERTEX_COUNT);
    let lastVertex = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (lastVertex <= firstVertex) { continue; }
    var previous = point(lastVertex - 1u);
    for (var vertex = firstVertex; vertex < lastVertex; vertex++) {
      let current = point(vertex);
      let edge = current - previous;
      let denominator = dot(edge, edge);
      if ((bitcast<u32>(denominator) & 0x7f800000u) == 0x7f800000u) { return vec3f(getQuietNaN()); }
      var fraction = 0.0;
      if (denominator > 0.0) { fraction = clamp(dot(query - previous, edge) / denominator, 0.0, 1.0); }
      let candidate = previous + fraction * edge;
      let candidateSquared = dot(query - candidate, query - candidate);
      if ((bitcast<u32>(candidateSquared) & 0x7f800000u) == 0x7f800000u) { return vec3f(getQuietNaN()); }
      if (candidateSquared < distanceSquared) { distanceSquared = candidateSquared; witness = candidate; }
      if ((current.y > query.y) != (previous.y > query.y)) {
        if (query.x < previous.x + (query.y - previous.y) * (current.x - previous.x) / (current.y - previous.y)) { inside = !inside; }
      }
      previous = current;
    }
  }
  return vec3f(witness, select(-sqrt(distanceSquared), sqrt(distanceSquared), inside));
}`,body:`let firstPolygon = min(featureOffsets[featureOffsetsOffset + index], POLYGON_COUNT);
let lastPolygon = min(featureOffsets[featureOffsetsOffset + index + 1u], POLYGON_COUNT);
let firstRing = min(polygonOffsets[polygonOffsetsOffset + firstPolygon], RING_COUNT);
let lastRing = min(polygonOffsets[polygonOffsetsOffset + lastPolygon], RING_COUNT);
var minimum = vec2f(3.0e38);
var maximum = vec2f(-3.0e38);
var vertexCount = 0u;
var invalid = false;
var bestPoint = vec2f(0.0);
for (var ring = firstRing; ring < lastRing; ring++) {
  for (var vertex = min(ringOffsets[ringOffsetsOffset + ring], VERTEX_COUNT); vertex < min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT); vertex++) {
    let current = point(vertex);
    if (any((bitcast<vec2u>(current) & vec2u(0x7f800000u)) == vec2u(0x7f800000u))) { invalid = true; }
    minimum = min(minimum, current); maximum = max(maximum, current);
    bestPoint = current; vertexCount++;
  }
}
let nan = getQuietNaN();
var bestRadius = 0.0;
var bestWitness = bestPoint;
var error = 0.0;
var resultStatus = 0u;
if (invalid || vertexCount == 0u) {
  bestPoint = vec2f(nan); bestRadius = nan; bestWitness = vec2f(nan); error = nan;
  resultStatus = select(2u, 3u, invalid);
} else {
  let cellBase = index * CELL_CAPACITY * 4u;
  let center = 0.5 * minimum + 0.5 * maximum;
  let halfSize = 0.5 * max(maximum.x - minimum.x, maximum.y - minimum.y);
  let seed = signedDistance(center, firstRing, lastRing);
  if (seed.z > bestRadius) { bestRadius = seed.z; bestPoint = center; bestWitness = seed.xy; }
  storeCell(cellBase, 0u, vec4f(center, halfSize, seed.z + 1.4142135623730951 * halfSize));
  var cellCount = 1u;
  var iterations = 0u;
  loop {
    if ((bitcast<u32>(seed.z) & 0x7f800000u) == 0x7f800000u) { resultStatus = 3u; break; }
    var winner = 0u;
    for (var cell = 1u; cell < cellCount; cell++) { if (loadCell(cellBase, cell).w > loadCell(cellBase, winner).w) { winner = cell; } }
    let parent = loadCell(cellBase, winner);
    error = max(0.0, parent.w - bestRadius);
    if (error <= TOLERANCE) { break; }
    if (iterations >= MAXIMUM_ITERATIONS || cellCount + 3u > CELL_CAPACITY || parent.z * 0.5 == parent.z) { resultStatus = 1u; break; }
    storeCell(cellBase, winner, loadCell(cellBase, cellCount - 1u)); cellCount--;
    let childHalfSize = parent.z * 0.5;
    for (var child = 0u; child < 4u; child++) {
      let childCenter = parent.xy + childHalfSize * vec2f(select(-1.0, 1.0, (child & 1u) != 0u), select(-1.0, 1.0, (child & 2u) != 0u));
      let distance = signedDistance(childCenter, firstRing, lastRing);
      if ((bitcast<u32>(distance.z) & 0x7f800000u) == 0x7f800000u) { resultStatus = 3u; break; }
      if (distance.z > bestRadius) { bestRadius = distance.z; bestPoint = childCenter; bestWitness = distance.xy; }
      storeCell(cellBase, cellCount, vec4f(childCenter, childHalfSize, distance.z + 1.4142135623730951 * childHalfSize)); cellCount++;
    }
    if (resultStatus == 3u) { break; }
    // Compact cells already dominated by the best lower bound, but retain one.
    var kept = 0u;
    for (var cell = 0u; cell < cellCount; cell++) {
      if (loadCell(cellBase, cell).w > bestRadius) { storeCell(cellBase, kept, loadCell(cellBase, cell)); kept++; }
    }
    if (kept == 0u) { error = 0.0; break; }
    cellCount = kept; iterations++;
  }
}
// Arithmetic overflow cannot provide a geometric bound or a usable witness.
if (resultStatus == 3u) { bestPoint = vec2f(nan); bestRadius = nan; bestWitness = vec2f(nan); error = nan; }
circles[circlesOffset + index * 4u] = bestPoint.x;
circles[circlesOffset + index * 4u + 1u] = bestPoint.y;
circles[circlesOffset + index * 4u + 2u] = bestRadius;
circles[circlesOffset + index * 4u + 3u] = error;
witnesses[witnessesOffset + index * 2u] = bestWitness.x;
witnesses[witnessesOffset + index * 2u + 1u] = bestWitness.y;
status[statusOffset + index] = resultStatus;`})]}},T={irregularity:.2,rows:2,columns:4},E=m({parameters:C,caption:`The biggest circle that still fits in each polygon, and the boundary point it touches.`,createInputs:({seed:e,world:t})=>{let n=g(p(e),t,{...T,bend:.6});return{seed:e,world:t,set:n,...b(n,2)}},build(e,t,n){let r=_(e,t.set),i=e.allocate(`circles`,`float32x4`,t.set.count),a=e.allocate(`witnesses`,`float32x2`,t.set.count),o=e.allocate(`status`,`uint32`,t.set.count);e.graph.add(new w({positions:r.positions.view,featureOffsets:r.featureOffsets.view,polygonOffsets:r.polygonOffsets.view,ringOffsets:r.ringOffsets.view,tolerance:n.tolerance,output:{circles:i.view,witnesses:a.view,status:o.view}}));let s=S(e,`circle`,i);return{outputs:{...r,witnesses:a,circles:i,circleSegments:s},update:e=>{let n=g(p(t.seed),t.world,{...T,bend:e.bend});h(n,t.drag),t.set=n,v(r,n)}}},interaction:{kind:`drag`,hint:`push the boundary`,rest:({inputs:e,toUnit:t})=>t(y(e)),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=x(t,e,1,.3),o=i(n.circles),s=t.drag?.feature??0,c=o?.[s*4+2];return{rerun:!0,glyphs:a.glyphs,readout:`radius ${c===void 0?`...`:c.toFixed(3)} · tolerance ${r.tolerance.toFixed(3)}`}}},rebuildOn:[`tolerance`],draw:({inputs:e,outputs:t})=>[{layer:`lines`,context:!0,segments:t.outline,count:e.set.outlineCount},{layer:`lines`,segments:t.circleSegments,radius:2.8},{layer:`dots`,positions:t.witnesses,radius:6,color:[1,.6,.2,1]}]});export{E as default};
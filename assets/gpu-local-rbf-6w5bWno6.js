import{n as e,r as t}from"./interpolation-view-validation-pYo9tqh3.js";import{l as n}from"./gpu-command-graph-BdzdmV6u.js";import{s as r,v as i}from"./spatial-context-DwZEd4EZ.js";import{c as a,o}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as s,u as c}from"./wgsl-common-DfyXiYB0.js";import{t as l}from"./scaled-quotient-wgsl-UDyhQag6.js";import{C as u,E as d,f,g as p,p as m}from"./gpu-contributor-utils-BIwLQON5.js";import{t as h}from"./gpu-prediction-utils-DjZFSzGW.js";import{a as g,i as _,r as v}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as y,t as b}from"./graph-chunk-groups-BB8ACpQO.js";import{t as x}from"./equilibrated-solve-wgsl-f1D0OhNH.js";var S=Object.freeze({...r,measure:`distance`}),C=5,w=32,T=class{id;spatialContext;props;neighborCount;degree;kernel;estimatedCommandNodeCount;chunkPlan;constructor(r){if(this.props=r,this.id=r.id??`local-rbf`,this.spatialContext=i(this.id,r.spatialContext,S),this.kernel=r.kernel??`thin-plate-spline`,this.degree=r.degree??(this.kernel===`quintic`?2:[`thin-plate-spline`,`cubic`].includes(this.kernel)?1:this.kernel===`linear`||this.kernel===`multiquadric`?0:-1),this.neighborCount=r.neighborCount??16,f(this.id,`neighborCount`,this.neighborCount,1,32),u(this.id,`epsilon`,r.epsilon??1),p(this.id,`smoothing`,r.smoothing??0,0),r.searchRadius!==void 0&&r.searchRadius!==1/0&&u(this.id,`searchRadius`,r.searchRadius),r.neighborhoods&&!r.neighborhoodOffsets&&f(this.id,`neighborhoodCapacity`,r.neighborhoodCapacity??this.neighborCount),![-1,0,1,2].includes(this.degree)||![`thin-plate-spline`,`cubic`,`gaussian`,`linear`,`quintic`,`multiquadric`,`inverse-multiquadric`,`inverse-quadratic`].includes(this.kernel)||[`thin-plate-spline`,`cubic`].includes(this.kernel)&&this.degree<1||[`linear`,`multiquadric`].includes(this.kernel)&&this.degree<0||this.kernel===`quintic`&&this.degree<2||!Number.isFinite(r.epsilon??1)||!((r.epsilon??1)>0)||!Number.isFinite(r.smoothing??0)||!((r.smoothing??0)>=0)||!(r.searchRadius===void 0||r.searchRadius>0))throw Error(`${this.id} invalid interpolation model`);if(r.positions.length!==r.values.length||r.sampleSmoothing&&r.sampleSmoothing.length!==r.positions.length||r.neighborhoodOffsets&&(!r.neighborhoods||r.neighborhoodOffsets.length!==r.queries.length+1)||r.neighborhoods&&!r.neighborhoodOffsets&&(!Number.isSafeInteger(r.neighborhoodCapacity??this.neighborCount)||(r.neighborhoodCapacity??this.neighborCount)<1||r.neighborhoods.length!==r.queries.length*(r.neighborhoodCapacity??this.neighborCount))||r.output.values.length<r.queries.length||r.output.status.length<r.queries.length)throw Error(`${this.id} invalid view lengths`);let a=[r.positions,r.values,r.queries,r.neighborhoods,r.neighborhoodOffsets,r.sampleSmoothing];for(let[e,t]of Object.entries({positions:r.positions,values:r.values,neighborhoods:r.neighborhoods,neighborhoodOffsets:r.neighborhoodOffsets,sampleSmoothing:r.sampleSmoothing}))t&&d(t,[t.format],`${this.id} ${e}`);if(r.queries instanceof n||r.output.values instanceof n||r.output.status instanceof n){this.chunkPlan=v(this.id,{queries:r.queries,results:r.output.values,statuses:r.output.status});for(let e of this.chunkPlan.chunks)d(e.views.queries,[`float32x2`],`${this.id} queries`),d(e.views.results,[`float32x4`],`${this.id} output.values`),d(e.views.statuses,[`uint32`],`${this.id} output.status`)}else t(this.id,a.filter(e=>e!==void 0),[r.output.values,r.output.status]);_(this.id,r.output),e(this.id,Object.values(r.output),a),this.estimatedCommandNodeCount=this.chunkPlan?this.chunkPlan.nonemptyChunkCount:1,m(this.id,this.estimatedCommandNodeCount)}getCommandNodes(e){let{id:t,props:n,chunkPlan:r}=this;return g(e,t,{positions:n.positions,values:n.values,queries:n.queries,results:n.output.values,statuses:n.output.status,neighborhoods:n.neighborhoods,neighborhoodOffsets:n.neighborhoodOffsets,sampleSmoothing:n.sampleSmoothing}),r?(y(e,t,C+Number(!!n.neighborhoods)+Number(!!n.neighborhoodOffsets)+Number(!!n.sampleSmoothing),w),r.chunks.filter(e=>e.length>0).map(i=>{let a=this.getKernelProps({...n,queries:i.views.queries,output:{values:i.views.results,status:i.views.statuses}},`queryIndex`);return b(e,{...a,id:`${t}-chunk-${i.chunkIndex}`,pipelineId:`${t}-chunk`,rowBase:i.rowBase,totalLength:r.totalLength,body:`let queryIndex = getChunkParameter(0u) + index;\n${a.body}`})})):[o(e,this.getKernelProps(n))]}getKernelProps(e,t=`index`){let{id:n,neighborCount:r,degree:i,kernel:o}=this,u=i<0?0:i===0?1:i===1?3:6,d=o===`thin-plate-spline`?`select(0.0, separation * separation * log(max(separation, 1e-30)), separation > 0.0)`:o===`cubic`?`separation * separation * separation`:o===`gaussian`?`exp(-separation * separation)`:o===`quintic`?`-pow(separation, 5.0)`:o===`multiquadric`?`-sqrt(1.0 + separation * separation)`:o===`inverse-multiquadric`?`inverseSqrt(1.0 + separation * separation)`:o===`inverse-quadratic`?`1.0 / (1.0 + separation * separation)`:`-separation`,f=o===`thin-plate-spline`?2:o===`cubic`?3:o===`quintic`?5:+(o===`linear`);return{id:n,operation:`GPULocalRBF`,variant:`${o}-${i}`,invocationCount:e.queries.length,workgroupSize:32,bindings:[{name:`positions`,view:e.positions,type:`f32`,access:`read`},{name:`values`,view:e.values,type:`f32`,access:`read`},{name:`queries`,view:e.queries,type:`f32`,access:`read`},{name:`results`,view:e.output.values,type:`f32`,access:`read_write`},{name:`statuses`,view:e.output.status,type:`u32`,access:`read_write`},...e.neighborhoodOffsets?[{name:`neighborhoodOffsets`,view:e.neighborhoodOffsets,type:`u32`,access:`read`}]:[],...e.neighborhoods?[{name:`neighborhoods`,view:e.neighborhoods,type:`u32`,access:`read`}]:[],...e.sampleSmoothing?[{name:`sampleSmoothing`,view:e.sampleSmoothing,type:`f32`,access:`read`}]:[]],declarations:`${h({positions:2,queries:2},[`results`])}\n${s}\n${c}\n${l}\n${x(r+u)}
var<private> radialScale: f32 = 1.0;
fn radial(first: vec2<f32>, second: vec2<f32>) -> f32 {
  let separation = distance(first, second) * ${a(e.epsilon??1)} * radialScale;
  return ${d};
}
fn scaleSmoothing(smoothing: f32, coordinateScale: f32) -> f32 {
  // Repeated divisions can reassociate into smoothing * pow(1 / scale, degree) on Metal.
  // Keep the power in integer exponents and a bounded [1/32, 1] mantissa instead.
  ${f>0?`let divisor = frexp(coordinateScale);
  return getScaledQuotient(smoothing, pow(divisor.fract, ${f}.0), 1.0, -${f} * divisor.exp);`:`return smoothing;`}
}
fn polynomial(position: vec2<f32>, column: u32) -> f32 {
  switch column { case 0u: { return 1.0; } case 1u: { return position.x; } case 2u: { return position.y; } case 3u: { return position.x * position.x; } case 4u: { return position.x * position.y; } default: { return position.y * position.y; } }
}`,body:`
write_results(index, vec4<f32>(getNaN(), 0.0, 0.0, 0.0));
statuses[statusesOffset + index] = 1u;
let query = read_queries(index);
if (!isFiniteValue(query.x) || !isFiniteValue(query.y)) { return; }
var neighbors: array<u32, ${r}>;
var distances: array<f32, ${r}>;
var retained = 0u;
${e.neighborhoodOffsets?`let candidateBegin = neighborhoodOffsets[neighborhoodOffsetsOffset + ${t}]; let candidateEnd = neighborhoodOffsets[neighborhoodOffsetsOffset + ${t} + 1u]; if (candidateBegin > candidateEnd || candidateEnd > arrayLength(&neighborhoods) - neighborhoodsOffset) { return; }`:``}
for (var candidate = ${e.neighborhoodOffsets?`candidateBegin`:`0u`}; candidate < ${e.neighborhoodOffsets?`candidateEnd`:e.neighborhoods?`${e.neighborhoodCapacity??r}u`:`${e.positions.length}u`}; candidate++) {
  let sample = ${e.neighborhoods?`neighborhoods[neighborhoodsOffset + ${e.neighborhoodOffsets?``:`${t} * ${e.neighborhoodCapacity??r}u + `}candidate]`:`candidate`};
  if (sample >= ${e.positions.length}u) { continue; }
  ${e.sampleSmoothing?`if (!isFiniteValue(sampleSmoothing[sampleSmoothingOffset + sample]) || sampleSmoothing[sampleSmoothingOffset + sample] < 0.0) { continue; }`:``}
  let position = read_positions(sample);
  if (!isFiniteValue(values[valuesOffset + sample]) || !isFiniteValue(position.x) || !isFiniteValue(position.y)) { continue; }
  let separation = distance(position, query);
  if (!(separation <= ${a(Number.isFinite(e.searchRadius)?e.searchRadius:3402823e32)})) { continue; }
  var insertion = retained;
  for (var neighbor = 0u; neighbor < retained; neighbor++) { if (separation < distances[neighbor] || (separation == distances[neighbor] && sample < neighbors[neighbor])) { insertion = neighbor; break; } }
  if (insertion >= ${r}u) { continue; }
  var shift = min(retained, ${r-1}u);
  loop { if (shift <= insertion) { break; } neighbors[shift] = neighbors[shift - 1u]; distances[shift] = distances[shift - 1u]; shift--; }
  neighbors[insertion] = sample; distances[insertion] = separation; retained = min(retained + 1u, ${r}u);
}
if (retained == 0u || retained < ${u}u) { return; }
let coordinateScale = max(distances[retained - 1u], 1e-20);
${f>0?`radialScale = 1.0 / coordinateScale;`:``}
for (var row = 0u; row < retained; row++) {
  let position = read_positions(neighbors[row]);
  for (var column = 0u; column < retained; column++) { matrix[row * SYSTEM_CAPACITY + column] = radial(position, read_positions(neighbors[column])); }
  matrix[row * SYSTEM_CAPACITY + row] += scaleSmoothing(${e.sampleSmoothing?`sampleSmoothing[sampleSmoothingOffset + neighbors[row]]`:a(e.smoothing??0)}, coordinateScale);
  right[row] = values[valuesOffset + neighbors[row]];
  for (var column = 0u; column < ${u}u; column++) {
    let value = polynomial((position - query) / coordinateScale, column);
    matrix[row * SYSTEM_CAPACITY + retained + column] = value;
    matrix[(retained + column) * SYSTEM_CAPACITY + row] = value;
  }
}
for (var row = 0u; row < ${u}u; row++) {
  right[retained + row] = 0.0;
  for (var column = 0u; column < ${u}u; column++) { matrix[(retained + row) * SYSTEM_CAPACITY + retained + column] = 0.0; }
}
statuses[statusesOffset + index] = 2u;
if (!solveSystem(retained + ${u}u)) { results[resultsOffset + index * 4u + 1u] = pivotRatio; return; }
var prediction = ${u?`solution[retained]`:`0.0`};
for (var row = 0u; row < retained; row++) { prediction += solution[row] * radial(query, read_positions(neighbors[row])); }
if (!isFiniteValue(prediction)) { return; }
write_results(index, vec4<f32>(prediction, pivotRatio, f32(retained), 0.0));
statuses[statusesOffset + index] = 0u;`}}};export{T as t};
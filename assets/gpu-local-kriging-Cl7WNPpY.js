import{n as e}from"./interpolation-view-validation-pYo9tqh3.js";import{r as t}from"./kernel-C3tfQOn4.js";import{s as n,v as r}from"./spatial-context-DwZEd4EZ.js";import{c as i,o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as o,u as s}from"./wgsl-common-DfyXiYB0.js";import{C as c,E as l,S as u,T as d,f,g as p,p as m,t as h,v as g,x as _}from"./gpu-contributor-utils-BIwLQON5.js";import{i as v,n as y,r as b}from"./gpu-neighbor-search-SzuWK-my.js";import{t as x}from"./neighbor-search-derived-parameters-DZFCGzl5.js";import{t as S}from"./gpu-prediction-utils-DjZFSzGW.js";import{i as C,n as w,r as T,t as E}from"./graph-chunk-groups-BB8ACpQO.js";import{a as D,i as O}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as k}from"./point-neighbor-grid-index-D0wS1y1I.js";import{t as A}from"./equilibrated-solve-wgsl-f1D0OhNH.js";import{i as j,n as M,s as N,t as P}from"./prediction-covariance-DfKlqgpX.js";import{Q as F}from"./dist-BusamGlx.js";function I(e,n,r){let i=t(e,`${n}-covariance`,`float32`,r.values.length,F.STORAGE|F.COPY_DST);return{parameters:i,nodes:[{id:`${n}-upload-covariance`,type:`copy`,resources:[{buffer:i,usage:`copy-destination`}],workload:{operation:`GPULocalKriging`,commandCount:1,writeByteLength:r.values.byteLength},compile:({device:e})=>{let t=new h(e,{id:`${n}-covariance-source`,format:`float32`,length:r.values.length,values:r.values});return r.buffers.add(t),{encode:({commandEncoder:e,getBuffer:n})=>{e.copyBufferToBuffer({sourceBuffer:t.buffer,destinationBuffer:n(i),size:i.length*Float32Array.BYTES_PER_ELEMENT})},destroy:()=>{r.buffers.delete(t),t.destroy()}}}}]}}var L=2**32,R=8,z=2048,B=32;function V(e){return Math.min(e+Math.max(e,R),256)}function H(e,t,n){return Math.max(0,Math.floor((e-t*8)/(n*4)))}var U=Object.freeze({...n,measure:`distance`}),W=class{id;estimatedCommandNodeCount;chunkPlan;props;neighborCount;supportCount;driftCount;queryCount;variableCount;suppliedDrift;covarianceStructureCount;spatialContext;sampleIndex;covarianceParameters;sampleIndexChunkQueryCount;constructor(t,n){this.sampleIndexChunkQueryCount=n,this.props=t,this.id=t.id??`local-kriging`,this.spatialContext=r(this.id,t.spatialContext,U),this.neighborCount=t.neighborCount??16,this.supportCount=t.supportCount??1,this.variableCount=t.mode===`co-kriging`?t.variableCount??2:1,this.suppliedDrift=t.mode===`external-drift`||t.mode===`universal`&&!!(t.sampleDrift||t.queryDrift),this.driftCount=this.suppliedDrift?t.driftCount??1:t.mode===`co-kriging`&&t.coKrigingMean===`ordinary`?this.variableCount:t.mode===`universal`?3:t.mode===`ordinary`?1:t.mode===`external-drift`?t.driftCount??1:0,this.queryCount=t.supports.length/this.supportCount,f(this.id,`neighborCount`,this.neighborCount,1,32),f(this.id,`supportCount`,this.supportCount,1,64),f(this.id,`queryCount`,this.queryCount,0),t.searchRadius!==void 0&&t.searchRadius!==1/0&&c(this.id,`searchRadius`,t.searchRadius);for(let[e,n]of(t.means??[]).entries())p(this.id,`means[${e}]`,n);if(t.neighborhoods&&!t.neighborhoodOffsets&&f(this.id,`neighborhoodCapacity`,t.neighborhoodCapacity??0),this.covarianceStructureCount=t.covarianceStructureCount??t.coregionalizationModels?.length??1,t.covariance)N(t.covariance);else if(!t.parameters)throw Error(`${this.id} needs covariance or parameters`);if(t.parameters&&(t.parameters.format!==`float32`||!Number.isInteger(this.covarianceStructureCount)||this.covarianceStructureCount<1||this.covarianceStructureCount>4||t.mode!==`co-kriging`&&this.covarianceStructureCount!==1||t.parameters.length<P(this.covarianceStructureCount,this.variableCount)))throw Error(`${this.id} invalid parameters`);if(this.driftCount<0||this.driftCount>8||!Number.isInteger(this.driftCount)||t.positions.length!==t.observations.length||t.output.values.length<this.queryCount||t.output.status.length<this.queryCount||t.output.weights&&t.output.weights.length<this.queryCount*this.neighborCount)throw Error(`${this.id} invalid capacities`);if(t.neighborhoodOffsets&&(!t.neighborhoods||t.neighborhoodOffsets.length!==this.queryCount+1)||t.neighborhoods&&!t.neighborhoodOffsets&&t.neighborhoods.length!==this.queryCount*(t.neighborhoodCapacity??0))throw Error(`${this.id} invalid neighborhoods`);if(![`simple`,`ordinary`,`universal`,`external-drift`,`co-kriging`].includes(t.mode)||!(t.searchRadius===void 0||t.searchRadius>0))throw Error(`${this.id} invalid mode or radius`);if(this.suppliedDrift&&(!t.sampleDrift||!t.queryDrift||t.sampleDrift.length!==t.positions.length*(t.sampleDrift.format===`float32`?this.driftCount:1)||t.queryDrift.length!==this.queryCount*(t.queryDrift.format===`float32`?this.driftCount:1)||this.driftCount>4&&(t.sampleDrift.format!==`float32`||t.queryDrift.format!==`float32`)||this.driftCount<1))throw Error(`${this.id} missing external drift`);if(t.mode===`co-kriging`&&!t.parameters&&!t.coregionalization&&!t.coregionalizationModels)throw Error(`${this.id} co-kriging requires PSD loading factors`);if(![...t.means??[0,0],...t.coregionalization??[1,0,0,1]].every(Number.isFinite))throw Error(`${this.id} nonfinite model`);let i=t.coregionalizationModels;if(this.variableCount<(t.mode===`co-kriging`?2:1)||this.variableCount>4||!Number.isInteger(this.variableCount)||!Number.isInteger(t.targetVariable??0)||(t.targetVariable??0)<0||(t.targetVariable??0)>=this.variableCount||t.means&&t.means.length<this.variableCount||t.coKrigingMean&&![`known`,`ordinary`].includes(t.coKrigingMean)||t.mode===`co-kriging`&&this.variableCount!==2&&!i&&!t.parameters||i&&(i.length<1||i.length>4))throw Error(`${this.id} invalid multivariate model`);for(let e of i??[])if(N(e.covariance),e.loadings.length!==this.variableCount**2||!e.loadings.every(Number.isFinite))throw Error(`${this.id} invalid PSD loading dimensions`);let a=t.emptyNeighborhood??`insufficient`;d(this.id,[`insufficient`,`prior`].includes(a),`emptyNeighborhood must be 'insufficient' or 'prior'`),d(this.id,a===`insufficient`||t.mode===`simple`,`emptyNeighborhood 'prior' needs mode 'simple' (only a known mean has a prior prediction)`);let o=t.sampleIndex??`auto`;d(this.id,[`auto`,`brute-force`,`grid`].includes(o),`sampleIndex must be 'auto', 'brute-force' or 'grid'`),d(this.id,!(t.neighborhoods&&o===`grid`),`sampleIndex 'grid' needs neighborhoods to be omitted`);let s=t.positions.length;if(t.neighborhoods)this.sampleIndex=`supplied`;else if(o===`brute-force`)u(this.id,`sampleIndex 'brute-force' work (samples times queries)`,s*this.queryCount,L),this.sampleIndex=`brute-force`;else{let e=s*this.queryCount,r=s<=4096&&e<=L,i=t.positions.buffer.graph.device.limits.maxStorageBufferBindingSize,a=s>0&&this.queryCount>0&&s*12<=i&&H(i,s,V(this.neighborCount))>0;this.sampleIndex=o===`grid`||!r&&a?`grid`:`brute-force`,d(this.id,this.sampleIndex!==`grid`||s>0&&this.queryCount>0,`sampleIndex 'grid' needs at least one sample and one query`),this.sampleIndex===`brute-force`&&o===`auto`&&u(this.id,`sampleIndex 'auto' scan work (samples times queries) where the grid does not fit`,e,L),n!==void 0&&f(this.id,`sampleIndexChunkQueryCount`,n,1)}this.chunkPlan=w(this.id,{view:t.supports,rowsPerQuery:this.supportCount},{supports:{view:t.supports,rowsPerQuery:this.supportCount},results:{view:t.output.values,allowPackedCapacity:!0},statuses:{view:t.output.status,allowPackedCapacity:!0},weights:t.output.weights?{view:t.output.weights,rowsPerQuery:this.neighborCount,allowPackedCapacity:!0}:void 0,queryDrift:this.suppliedDrift&&t.queryDrift?{view:t.queryDrift,rowsPerQuery:t.queryDrift.format===`float32`?this.driftCount:1}:void 0});let h={positions:t.positions,observations:t.observations,sampleDrift:t.sampleDrift,neighborhoods:t.neighborhoods,neighborhoodOffsets:t.neighborhoodOffsets,parameters:t.parameters};for(let[e,t]of Object.entries(h))t&&l(t,[t.format],`${this.id} ${e}`);O(this.id,t.output),e(this.id,Object.values(t.output),[...Object.values(h),t.supports,t.queryDrift]);let g=t.positions.buffer.graph,_=Math.max(1,Math.min(n??1/0,H(g.device.limits.maxStorageBufferBindingSize,s,V(this.neighborCount)))),v=this.sampleIndex===`grid`?this.chunkPlan.chunks.reduce((e,t)=>e+Math.ceil(t.length/_)*2,0):this.chunkPlan.nonemptyChunkCount;this.estimatedCommandNodeCount=z+v,m(this.id,this.estimatedCommandNodeCount),t.parameters||(this.covarianceParameters={values:M({covariance:t.covariance,...t.mode===`co-kriging`?{coregionalization:t.coregionalization,coregionalizationModels:t.coregionalizationModels,variableCount:this.variableCount}:{}}),buffers:new Set},d(this.id,this.covarianceParameters.values.length===P(this.covarianceStructureCount,this.variableCount),`covarianceStructureCount must match the initial covariance structures`))}setCovariance(e){let t=this.covarianceParameters;d(this.id,!!t,`setCovariance requires convenience covariance props`);let n=M(e);d(this.id,n.length===t.values.length&&(e.variableCount??(e.coregionalization||e.coregionalizationModels?2:1))===this.variableCount,`setCovariance must preserve covarianceStructureCount and variableCount`),t.values=n;for(let e of t.buffers)e.write(n)}getCommandNodes(e){let{props:n,id:r,driftCount:i}=this;D(e,r,{positions:n.positions,observations:n.observations,supports:n.supports,sampleDrift:n.sampleDrift,queryDrift:n.queryDrift,neighborhoods:n.neighborhoods,neighborhoodOffsets:n.neighborhoodOffsets,parameters:n.parameters,...n.output});let o=6+(this.suppliedDrift?2:0)+(n.neighborhoods&&this.sampleIndex!==`grid`?1:0)+ +!!n.neighborhoodOffsets+ +!!n.output.weights,s=!!this.covarianceParameters&&o>e.device.limits.maxStorageBuffersPerShaderStage;C(e,r,o-Number(s),B);let c=n.observations.length*4,l=n.parameters?.length??this.covarianceParameters.values.length,u=c+l;g(e.device,r,`covariance parameters`,l*4,`reduce covarianceStructureCount`),s&&g(e.device,r,`observations plus covariance`,u*4,`reduce the sample count`);let f=n.positions.length,p=V(this.neighborCount),m=H(e.device.limits.maxStorageBufferBindingSize,f,p),h=this.chunkPlan.chunks.reduce((e,t)=>Math.max(e,t.length),0),_=Math.min(h,this.sampleIndexChunkQueryCount??1/0,m);if(this.sampleIndex===`grid`){d(r,_>0,`sampleIndex 'grid' candidate binding must hold at least one query and the packed sample positions`);let t=Math.max(1,Math.ceil(Math.sqrt(f/2)));for(let[n,i]of Object.entries({gridPositions:f*8,sortedTargets:f*12,cellOffsets:(t*t+1)*4,candidateRows:_*p*4+f*8,gridParameters:48}))g(e.device,r,n,i,`reduce the packed sample count or query chunk size`)}let v=n.sampleDrift?.format===`float32`?i:4,y=`fn isUsableSample(sampleIndex: u32) -> bool {
  let observation = read_observations(sampleIndex);
  let position = read_positions(sampleIndex);
  if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0 || !isFiniteValue(position.x) || !isFiniteValue(position.y)) { return false; }
  ${n.mode===`co-kriging`?`if (!isFiniteValue(observation.z) || observation.z < 0.0 || observation.z >= ${this.variableCount}.0 || floor(observation.z) != observation.z) { return false; }`:``}
  ${this.suppliedDrift?`for (var column = 0u; column < ${i}u; column++) { if (!isFiniteValue(sampleDrift[sampleDriftOffset + sampleIndex * ${v}u + column])) { return false; } }`:``}
  return true;
}`,b=this.covarianceParameters?I(e,r,this.covarianceParameters):void 0,x=n.parameters??b.parameters,S=[...b?.nodes??[]],w;s&&(w=t(e,`${r}-observations-covariance`,`float32`,u),S.push(a(e,{id:`${r}-pack-observations-covariance`,operation:`GPULocalKriging`,variant:`pack-observations-covariance`,invocationCount:u,bindings:[{name:`observations`,view:n.observations,type:`f32`,access:`read`},{name:`parameters`,view:x,type:`f32`,access:`read`},{name:`packed`,view:w,type:`f32`,access:`read_write`}],body:`if (index < ${c}u) { packed[packedOffset + index] = observations[observationsOffset + index]; }
else { packed[packedOffset + index] = parameters[parametersOffset + index - ${c}u]; }`})));let E=this.sampleIndex===`grid`?this.getSampleIndexNodes(e,y,_):void 0;E&&S.push(...E.nodes);let O=0;for(let t of this.chunkPlan.chunks){let a=E?_:Math.max(t.length,1);for(let o=0;o<t.length;o+=a){let s=Math.min(a,t.length-o),c=(n,r)=>t.views[n]?T(e,t.views[n],o*r,s*r):void 0,l={...n,supports:c(`supports`,this.supportCount),queryDrift:c(`queryDrift`,n.queryDrift?.format===`float32`?i:1),output:{values:c(`results`,1),status:c(`statuses`,1),weights:c(`weights`,this.neighborCount)}},u=O===0?r:`${r}-chunk-${O}`;E&&S.push(this.getGridCandidateNode(e,l,E,`${u}-candidates`)),S.push(this.getKrigingKernelNode(e,y,x,w,E?{nodeId:u,candidateCapacity:p,candidates:E.candidates}:void 0,l,t.rowBase+o)),O++}}return S}getKrigingKernelNode(e,t,n,r,a,c,l=0){let{id:u,neighborCount:d,supportCount:f,driftCount:p}=this,m=c??this.props,h=m.supports.length/f,g=m.sampleDrift?.format===`float32`?p:4,v=m.queryDrift?.format===`float32`?p:4,y=a?a.candidates:m.neighborhoods,b=m.neighborhoodOffsets,x=a?a.candidateCapacity:m.neighborhoodCapacity,C=[...a?[]:[{name:`positions`,view:m.positions,type:`f32`,access:`read`}],{name:`observations`,view:r??m.observations,type:`f32`,access:`read`},{name:`supports`,view:m.supports,type:`f32`,access:`read`},{name:`results`,view:m.output.values,type:`f32`,access:`read_write`},{name:`statuses`,view:m.output.status,type:`u32`,access:`read_write`}];this.suppliedDrift&&C.push({name:`sampleDrift`,view:m.sampleDrift,type:`f32`,access:`read`},{name:`queryDrift`,view:m.queryDrift,type:`f32`,access:`read`}),b&&C.push({name:`neighborhoodOffsets`,view:b,type:`u32`,access:`read`}),y&&C.push({name:`neighborhoods`,view:y,type:`u32`,access:`read`}),r||C.push({name:`params`,view:n,type:`f32`,access:`read`}),m.output.weights&&C.push({name:`weights`,view:m.output.weights,type:`f32`,access:`read_write`}),_(u,e,C.map(e=>e.view));let w=m.mode===`co-kriging`?m.targetVariable??0:0,T=m.means??Array.from({length:this.variableCount},()=>0),D=m.mode===`co-kriging`&&!this.suppliedDrift&&p>0,O=D?`var driftColumns: array<u32, ${p}>;
var driftColumnCount = 0u;
for (var variable = 0u; variable < DRIFT_COUNT; variable++) {
  var present = variable == TARGET_VARIABLE;
  for (var row = 0u; row < retained; row++) {
    if (u32(read_observations(neighbors[row]).z) == variable) { present = true; break; }
  }
  if (present) { driftColumns[driftColumnCount] = variable; driftColumnCount++; }
}`:`let driftColumnCount = DRIFT_COUNT;`,k=this.suppliedDrift?`sampleDrift[sampleDriftOffset + neighbors[row] * ${g}u + column]`:m.mode===`co-kriging`?`select(0.0, 1.0, u32(observation.z) == ${D?`driftColumns[column]`:`column`})`:m.mode===`universal`?`select((read_positions(neighbors[row]) - center)[min(column - 1u, 1u)] / coordinateScale, 1.0, column == 0u)`:`1.0`,M=this.suppliedDrift?`queryDrift[queryDriftOffset + index * ${v}u + column]`:m.mode===`co-kriging`?`select(0.0, 1.0, TARGET_VARIABLE == ${D?`driftColumns[column]`:`column`})`:`select(0.0, 1.0, column == 0u)`,N=r?`(observationsOffset + ${m.observations.length*4}u)`:`paramsOffset`,P=r?`observations`:`params`,F=j(this.covarianceStructureCount,this.variableCount).replace(/\bparamsOffset\b/g,N).replace(/\bparams\b/g,P),I=i(Number.isFinite(m.searchRadius)?m.searchRadius:3402823e32),L=`if (sampleIndex < SAMPLE_COUNT && isUsableSample(sampleIndex)) {
  let position = read_positions(sampleIndex);
  let separation = distance(position, center);
  ${a?`farthestCandidate = max(farthestCandidate, separation);`:``}
  if (separation <= SEARCH_RADIUS) {
    var insertion = retained;
    for (var neighbor = 0u; neighbor < retained; neighbor++) { if (separation < distances[neighbor] || (separation == distances[neighbor] && sampleIndex < neighbors[neighbor])) { insertion = neighbor; break; } }
    if (insertion < NEIGHBOR_COUNT) {
      var shift = min(retained, NEIGHBOR_COUNT - 1u);
      loop { if (shift <= insertion) { break; } neighbors[shift] = neighbors[shift - 1u]; distances[shift] = distances[shift - 1u]; shift--; }
      neighbors[insertion] = sampleIndex; distances[insertion] = separation; retained = min(retained + 1u, NEIGHBOR_COUNT);
    }
  }
}`,R=a?`var farthestCandidate = -1.0;
var listed = 0u;
for (var candidate = 0u; candidate < CANDIDATE_COUNT; candidate++) {
  let sampleIndex = neighborhoods[neighborhoodsOffset + candidateQuery * CANDIDATE_COUNT + candidate];
  if (sampleIndex < SAMPLE_COUNT) { listed++; }
  ${L}
}
let boundary = select(SEARCH_RADIUS, distances[NEIGHBOR_COUNT - 1u], retained == NEIGHBOR_COUNT);
if (listed == CANDIDATE_COUNT && farthestCandidate <= boundary * ${i(1.000001)}) {
  retained = 0u;
  for (var sampleIndex = 0u; sampleIndex < SAMPLE_COUNT; sampleIndex++) {
    ${L}
  }
}`:`${b?`let candidateBegin = neighborhoodOffsets[neighborhoodOffsetsOffset + queryIndex];
let candidateEnd = neighborhoodOffsets[neighborhoodOffsetsOffset + queryIndex + 1u];
if (candidateBegin > candidateEnd || candidateEnd > arrayLength(&neighborhoods) - neighborhoodsOffset) { return; }`:``}
for (var candidate = ${b?`candidateBegin`:`0u`}; candidate < ${b?`candidateEnd`:y?`${x}u`:`SAMPLE_COUNT`}; candidate++) {
  let sampleIndex = ${y?`neighborhoods[neighborhoodsOffset + ${b?``:`queryIndex * ${x}u + `}candidate]`:`candidate`};
  ${L}
}`,z=S({...a?{}:{positions:2},observations:4,supports:m.supports.format===`float32x4`?4:2},[`results`]),V=a?`const CANDIDATE_COUNT: u32 = ${a.candidateCapacity}u;
fn read_positions(row: u32) -> vec2<f32> { return vec2<f32>(bitcast<f32>(neighborhoods[arrayLength(&neighborhoods) - SAMPLE_COUNT * 2u + row * 2u]), bitcast<f32>(neighborhoods[arrayLength(&neighborhoods) - SAMPLE_COUNT * 2u + row * 2u + 1u])); }`:``;return E(e,{id:a?a.nodeId:l===0?u:`${u}-chunk-${l}`,pipelineId:u,rowBase:l,totalLength:this.queryCount,operation:`GPULocalKriging`,variant:m.mode,bindings:C,invocationCount:h,workgroupSize:B,declarations:`${z}\n${V}\nconst SEARCH_RADIUS: f32 = ${I};\n${o}\n${s}\n${t}
${A(d+p)}\n${F}
const NEIGHBOR_COUNT: u32 = ${d}u;
const DRIFT_COUNT: u32 = ${p}u;
const SUPPORT_COUNT: u32 = ${f}u;
const SAMPLE_COUNT: u32 = ${m.positions.length}u;
const TARGET_VARIABLE: u32 = ${w}u;
const MEANS: array<f32, ${T.length}> = array<f32, ${T.length}>(${T.map(i).join(`, `)});
`,body:`let queryIndex = getChunkParameter(0u) + index;
${a?`let candidateQuery = index;`:``}
write_results(index, vec4<f32>(getNaN(), getNaN(), 0.0, 0.0));
statuses[statusesOffset + index] = 1u;
${m.output.weights?`for (var column = 0u; column < NEIGHBOR_COUNT; column++) { weights[weightsOffset + index * NEIGHBOR_COUNT + column] = 0.0; }`:``}
if (!loadCovarianceParameters()) { return; }
let covarianceScale = ${P}[${N}];
var center = vec2<f32>(0.0);
var supportWeightSum = 0.0;
var supportWeights: array<f32, ${f}>;
for (var support = 0u; support < SUPPORT_COUNT; support++) {
  let supportValue = read_supports(index * SUPPORT_COUNT + support);
  let point = supportValue.xy;
  let weight = ${m.supports.format===`float32x4`?`supportValue.z`:`1.0`};
  if (!isFiniteValue(weight) || weight < 0.0) { return; }
  supportWeights[support] = weight; supportWeightSum += weight;
  if (!isFiniteValue(point.x) || !isFiniteValue(point.y)) { return; }
  center += point * weight;
}
if (!(supportWeightSum > 0.0) || !isFiniteValue(supportWeightSum)) { return; }
center /= supportWeightSum;
for (var support = 0u; support < SUPPORT_COUNT; support++) { supportWeights[support] /= supportWeightSum; }
var neighbors: array<u32, ${d}>;
var distances: array<f32, ${d}>;
var retained = 0u;
${R}
${O}
${m.emptyNeighborhood===`prior`?`if (retained == 0u) {
  // Simple kriging of an empty neighbourhood: the known mean and the prior variance of the support.
  var priorVariance = 0.0;
  for (var first = 0u; first < SUPPORT_COUNT; first++) {
    for (var second = 0u; second < SUPPORT_COUNT; second++) {
      priorVariance += crossCovariance(read_supports(index * SUPPORT_COUNT + first).xy, read_supports(index * SUPPORT_COUNT + second).xy, TARGET_VARIABLE, TARGET_VARIABLE) * supportWeights[first] * supportWeights[second];
    }
  }
  if (isFiniteValue(priorVariance) && priorVariance >= 0.0) {
    write_results(index, vec4<f32>(MEANS[TARGET_VARIABLE], priorVariance, 1.0, 0.0));
    statuses[statusesOffset + index] = 0u;
  }
  return;
}`:``}
if (retained == 0u || retained < driftColumnCount) { return; }
let coordinateScale = max(distances[retained - 1u], 1e-20);
var crossValues: array<f32, ${d}>;
var queryDesign: array<f32, ${Math.max(p,1)}>;
for (var row = 0u; row < retained; row++) {
  let position = read_positions(neighbors[row]);
  let observation = read_observations(neighbors[row]);
  let variable = ${m.mode===`co-kriging`?`u32(observation.z)`:`0u`};
  for (var column = 0u; column < retained; column++) {
    let otherObservation = read_observations(neighbors[column]);
    matrix[row * SYSTEM_CAPACITY + column] = crossCovariance(position, read_positions(neighbors[column]), variable, ${m.mode===`co-kriging`?`u32(otherObservation.z)`:`0u`}) / covarianceScale;
  }
  matrix[row * SYSTEM_CAPACITY + row] += observation.y / covarianceScale;
  var crossValue = 0.0;
  for (var support = 0u; support < SUPPORT_COUNT; support++) { crossValue += crossCovariance(position, read_supports(index * SUPPORT_COUNT + support).xy, variable, TARGET_VARIABLE) * supportWeights[support]; }
  crossValues[row] = crossValue; right[row] = crossValue / covarianceScale;
  for (var column = 0u; column < driftColumnCount; column++) {
    let value = ${k};
    matrix[row * SYSTEM_CAPACITY + retained + column] = value;
    matrix[(retained + column) * SYSTEM_CAPACITY + row] = value;
  }
}
for (var column = 0u; column < driftColumnCount; column++) {
  let value = ${M};
  if (!isFiniteValue(value)) { return; }
  queryDesign[column] = value; right[retained + column] = value;
  for (var other = 0u; other < driftColumnCount; other++) { matrix[(retained + column) * SYSTEM_CAPACITY + retained + other] = 0.0; }
}
statuses[statusesOffset + index] = 2u;
if (!solveSystem(retained + driftColumnCount)) { results[resultsOffset + index * 4u + 2u] = pivotRatio; return; }
var prediction = ${p?`0.0`:`MEANS[TARGET_VARIABLE]`};
var variance = 0.0;
for (var first = 0u; first < SUPPORT_COUNT; first++) {
  for (var second = 0u; second < SUPPORT_COUNT; second++) {
    variance += crossCovariance(read_supports(index * SUPPORT_COUNT + first).xy, read_supports(index * SUPPORT_COUNT + second).xy, TARGET_VARIABLE, TARGET_VARIABLE) * supportWeights[first] * supportWeights[second];
  }
}
let varianceScale = max(variance, 1e-20);
for (var row = 0u; row < retained; row++) {
  let observation = read_observations(neighbors[row]);
  prediction += solution[row] * (observation.x ${p?``:`- MEANS[${m.mode===`co-kriging`?`u32(observation.z)`:`0u`}]`});
  variance -= solution[row] * crossValues[row];
}
for (var column = 0u; column < driftColumnCount; column++) { variance -= solution[retained + column] * queryDesign[column] * covarianceScale; }
if (!isFiniteValue(prediction) || !isFiniteValue(variance) || variance < -1e-4 * varianceScale) { return; }
write_results(index, vec4<f32>(prediction, max(variance, 0.0), pivotRatio, f32(retained)));
statuses[statusesOffset + index] = 0u;
${m.output.weights?`for (var row = 0u; row < retained; row++) { weights[weightsOffset + index * NEIGHBOR_COUNT + row] = solution[row]; }`:``}
`})}getSampleIndexNodes(e,n,r){let{props:i,id:c}=this,l=i.positions.length,u=V(this.neighborCount),d=x(e,{id:`${c}-index-parameters`,operation:`GPULocalKriging`,positions:i.positions,radius:1/0}),f=t(e,`${c}-index-unit-positions`,`float32x2`,l),p=v(`knn`,d.gridSize),m=a(e,{id:`${c}-index-prepare`,operation:`GPULocalKriging`,variant:`sample-index-prepare`,invocationCount:l,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`observations`,view:i.observations,type:`f32`,access:`read`},...this.suppliedDrift?[{name:`sampleDrift`,view:i.sampleDrift,type:`f32`,access:`read`}]:[],{name:`parameters`,view:d.parameters,type:`f32`,access:`read`},{name:`gridPositions`,view:f,type:`f32`,access:`read_write`}],declarations:`${S({positions:2,observations:4})}\n${o}\n${s}\n${p}\n${n}
fn getKrigingGridUnitPosition(lattice: Lattice, position: vec2<f32>) -> vec2<f32> {
  let width = max(lattice.maximumX - lattice.minimumX, lattice.cellWidth * f32(COLUMNS));
  let height = max(lattice.maximumY - lattice.minimumY, lattice.cellHeight * f32(ROWS));
  return (position - vec2<f32>(lattice.minimumX, lattice.minimumY)) / vec2<f32>(width, height);
}`,body:`let lattice = readLattice(); var unitPosition = vec2<f32>(getNaN());
if (lattice.valid && isUsableSample(index)) { unitPosition = getKrigingGridUnitPosition(lattice, read_positions(index)); }
gridPositions[gridPositionsOffset + index * 2u] = unitPosition.x;
gridPositions[gridPositionsOffset + index * 2u + 1u] = unitPosition.y;`}),h=k(e,{id:`${c}-index`,positions:f,gridSize:d.gridSize,rowCount:l}),g=t(e,`${c}-index-sorted-targets`,`uint32`,l*3),_=t(e,`${c}-index-rows`,`uint32`,r*u+l*2),y=a(e,{id:`${c}-index-pack-targets`,operation:`GPULocalKriging`,variant:`sample-index-pack-targets`,invocationCount:l,bindings:[{name:`positions`,view:i.positions,type:`u32`,access:`read`},{name:`rows`,view:h.sortedRows,type:`u32`,access:`read`},{name:`count`,view:h.gridCount,type:`u32`,access:`read`},{name:`sortedTargets`,view:g,type:`u32`,access:`read_write`},{name:`candidates`,view:_,type:`u32`,access:`read_write`}],body:`if (index < count[countOffset]) {
  let row = rows[rowsOffset + index];
  sortedTargets[sortedTargetsOffset + index * 3u] = positions[positionsOffset + row * 2u];
  sortedTargets[sortedTargetsOffset + index * 3u + 1u] = positions[positionsOffset + row * 2u + 1u];
  sortedTargets[sortedTargetsOffset + index * 3u + 2u] = row;
}
let positionsBase = arrayLength(&candidates) - ${l*2}u;
candidates[positionsBase + index * 2u] = positions[positionsOffset + index * 2u];
candidates[positionsBase + index * 2u + 1u] = positions[positionsOffset + index * 2u + 1u];`});return{nodes:[...d.nodes,m,...h.nodes,y],candidates:_,sortedTargets:g,cellOffsets:h.cellOffsets,parameters:d.parameters,gridSize:d.gridSize}}getGridCandidateNode(e,t,n,r){let i=V(this.neighborCount),a=v(`knn`,n.gridSize);return E(e,{id:r,pipelineId:`${this.id}-index-candidates`,operation:`GPULocalKriging`,variant:`sample-index-candidates`,invocationCount:t.supports.length/this.supportCount,workgroupSize:B,bindings:[{name:`supports`,view:t.supports,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`cellOffsets`,view:n.cellOffsets,type:`u32`,access:`read`},{name:`sortedTargets`,view:n.sortedTargets,type:`u32`,access:`read`},{name:`candidates`,view:n.candidates,type:`u32`,access:`read_write`}],declarations:`${S({supports:t.supports.format===`float32x4`?4:2})}\n${a}\n${s}\n${y(i)}`,body:`for (var slot = 0u; slot < K; slot++) { candidates[candidatesOffset + index * K + slot] = 0xffffffffu; }
var center = vec2<f32>(0.0); var weightSum = 0.0;
for (var support = 0u; support < ${this.supportCount}u; support++) {
  let value = read_supports(index * ${this.supportCount}u + support);
  let weight = ${t.supports.format===`float32x4`?`value.z`:`1.0`};
  if (!isFiniteValue(value.x) || !isFiniteValue(value.y) || !isFiniteValue(weight) || weight < 0.0) { return; }
  center += value.xy * weight; weightSum += weight;
}
if (!(weightSum > 0.0) || !isFiniteValue(weightSum)) { return; }
center /= weightSum;
let x = center.x; let y = center.y; let lattice = readLattice();
if (!lattice.valid || cellOffsets[cellOffsetsOffset + CELL_COUNT] == 0u) { return; }
var found = 0u; var bestDistances: array<f32, ${i}>; var bestIds: array<u32, ${i}>;
${b(`true`)}
for (var slot = 0u; slot < found; slot++) { candidates[candidatesOffset + index * K + slot] = bestIds[slot]; }`})}};export{W as t};
import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-reduction-tNgOVQ_C.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{t as r}from"./geodesic-wgsl-I0CYDKcz.js";import{a as i,g as a,v as o}from"./spatial-context-DwZEd4EZ.js";import{a as s,c,o as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{T as u,f as d,p as f,v as p,x as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{n as g}from"./temporal-context-xddVhAJD.js";import{t as _}from"./gpu-neighbor-search-SzuWK-my.js";import{t as v}from"./neighbor-search-derived-parameters-DZFCGzl5.js";import{l as y}from"./wgsl-common-DfyXiYB0.js";import{n as b,t as x}from"./point-process-utils-BVnRkMJj.js";import{t as S}from"./define-run-B-brkCe4.js";import{n as C}from"./inputs-CJKKIRKK.js";import{n as w,t as T}from"./pick-Bo3t3hjv.js";import{Et as E}from"./entry-preview-D70TdSbK.js";function D(t,n,r){let i=n.positions.length,a=n.output,o=e(t,`${r}-xi-areas`,`uint32`,i*4+6),c=a.clusterIntervals??e(t,`${r}-xi-intervals`,`uint32x2`,i),u=a.clusterSummary??e(t,`${r}-xi-summary`,`uint32`,2),d=s(t,{id:`${r}-xi-dispatch`,operation:`GPUOPTICS`,shapes:[[1]]}),f=[l(t,{id:`${r}-xi-initialize`,operation:`GPUOPTICS`,variant:`xi-initialize`,invocationCount:i,bindings:[{name:`ordering`,view:a.ordering,type:`u32`,access:`read`},{name:`status`,view:a.status,type:`u32`,access:`read`},{name:`labels`,view:a.labels,type:`u32`,access:`read_write`},{name:`areas`,view:o,type:`u32`,access:`read_write`},{name:`intervals`,view:c,type:`u32`,access:`read_write`},{name:`summary`,view:u,type:`u32`,access:`read_write`},d.binding],body:`labels[labelsOffset + index] = 0xffffffffu;
  if (index < status[statusOffset]) {
    areas[areasOffset + ${i*3}u + ordering[orderingOffset + index]] = index;
  }
  intervals[intervalsOffset + 2u * index] = 0xffffffffu;
  intervals[intervalsOffset + 2u * index + 1u] = 0xffffffffu;
  if (index == 0u) {
    for (var word = 0u; word < 6u; word++) { areas[areasOffset + ${i*4}u + word] = 0u; }
    areas[areasOffset + ${i*4+4}u] = status[statusOffset];
    summary[summaryOffset] = 0u; summary[summaryOffset + 1u] = 0u;
    ${d.armWGSL}
    if (status[statusOffset + 1u] == 0u || status[statusOffset] == 0u) { ${d.disarmWGSL} }
  }`})];for(let e=0;e<Math.ceil(i/64);e++)f.push(l(t,{id:`${r}-xi-${e}`,operation:`GPUOPTICS`,variant:`xi-extraction`,invocationCount:1,...d.getGate(0),bindings:[{name:`ordering`,view:a.ordering,type:`u32`,access:`read`},{name:`reachability`,view:a.reachability,type:`f32`,access:`read`},{name:`predecessors`,view:a.predecessors,type:`u32`,access:`read`},{name:`labels`,view:a.labels,type:`u32`,access:`read_write`},{name:`areas`,view:o,type:`u32`,access:`read_write`},{name:`intervals`,view:c,type:`u32`,access:`read_write`},{name:`summary`,view:u,type:`u32`,access:`read_write`}],declarations:`const VISITED_OFFSET = ${i*4+4}u;
const BATCH_START_OFFSET = ${i*4+5}u;
const INVERSE_ORDER_OFFSET = ${i*3}u;
const XI_COMPLEMENT = ${Math.fround(1-(n.xi??.05))}f;
fn reach(rank: u32) -> f32 {
  if (rank >= areas[areasOffset + VISITED_OFFSET]) { return bitcast<f32>(0x7f800000u | (rank & 0u)); }
  return reachability[reachabilityOffset + ordering[orderingOffset + rank]];
}
fn direction(rank: u32) -> vec4<bool> {
  let first = reach(rank); let second = reach(rank + 1u);
  // Both infinities and both zeros have no direction, matching undefined reference ratios.
  if (first == second) { return vec4<bool>(false); }
  return vec4<bool>(first <= XI_COMPLEMENT * second, first * XI_COMPLEMENT >= second, (first < second), (first > second));
}
fn extend(start: u32, downward: bool) -> u32 {
  var end = start; var nonSteep = 0u;
  for (var rank = start; rank < areas[areasOffset + VISITED_OFFSET]; rank++) {
    let flags = direction(rank);
    if (select(flags.x, flags.y, downward)) { nonSteep = 0u; end = rank; }
    else if (!select(flags.w, flags.z, downward)) {
      nonSteep++; if (nonSteep > ${n.minimumSamples}u) { break; }
    } else { break; }
  }
  return end;
}`,body:`let batchStart = areas[areasOffset + BATCH_START_OFFSET];
var areaCount = areas[areasOffset + ${i*4}u];
var next = areas[areasOffset + ${i*4+1}u];
var maximumBetween = bitcast<f32>(areas[areasOffset + ${i*4+2}u]);
var clusterCount = summary[summaryOffset];
var label = areas[areasOffset + ${i*4+3}u];
for (var steep = batchStart; steep < min(batchStart + 64u, areas[areasOffset + VISITED_OFFSET]); steep++) {
  if (steep < next) { continue; }
  let flags = direction(steep);
  if (!flags.x && !flags.y) { continue; }
  for (var rank = next; rank <= steep; rank++) { maximumBetween = max(maximumBetween, reach(rank)); }
  var retained = 0u;
  for (var area = 0u; area < areaCount; area++) {
    let offset = areasOffset + 3u * area;
    let start = areas[offset];
    if (bitcast<u32>(maximumBetween) != 0x7f800000u && maximumBetween <= reach(start) * XI_COMPLEMENT) {
      let destination = areasOffset + 3u * retained;
      areas[destination] = start; areas[destination + 1u] = areas[offset + 1u];
      areas[destination + 2u] = bitcast<u32>(max(bitcast<f32>(areas[offset + 2u]), maximumBetween)); retained++;
    }
  }
  areaCount = retained;
  let end = extend(steep, flags.y); next = end + 1u; maximumBetween = reach(next);
  if (flags.y) {
    let destination = areasOffset + 3u * areaCount;
    areas[destination] = steep; areas[destination + 1u] = end; areas[destination + 2u] = 0u; areaCount++;
    continue;
  }
  // Reverse the matching downward areas so inner intervals precede containing intervals.
  for (var reverse = areaCount; reverse > 0u; reverse--) {
    let offset = areasOffset + 3u * (reverse - 1u);
    var start = areas[offset]; var finish = end;
    let downwardEnd = areas[offset + 1u];
    if (reach(finish + 1u) * XI_COMPLEMENT < bitcast<f32>(areas[offset + 2u])) { continue; }
    let downwardMaximum = reach(start);
    if (downwardMaximum * XI_COMPLEMENT >= reach(finish + 1u)) {
      while (start < downwardEnd && reach(start + 1u) > reach(finish + 1u)) { start++; }
    } else if (reach(finish + 1u) * XI_COMPLEMENT >= downwardMaximum) {
      while (finish > steep && reach(finish - 1u) > downwardMaximum) { finish--; }
    }
    ${n.predecessorCorrection===!1?``:`while (start < finish) {
      if (reach(start) > reach(finish)) { break; }
      let predecessor = predecessors[predecessorsOffset + ordering[orderingOffset + finish]];
      if (predecessor != 0xffffffffu) {
        let predecessorRank = areas[areasOffset + INVERSE_ORDER_OFFSET + predecessor];
        if (start <= predecessorRank && predecessorRank < finish) { break; }
      }
      finish--;
    }
    if (start >= finish) { continue; }`}
    if (finish - start + 1u < ${n.minimumClusterSize??n.minimumSamples}u || start > downwardEnd || finish < steep) { continue; }
    if (clusterCount >= ${i}u) { summary[summaryOffset + 1u] = 1u; continue; }
    intervals[intervalsOffset + 2u * clusterCount] = start;
    intervals[intervalsOffset + 2u * clusterCount + 1u] = finish; clusterCount++;
    var occupied = false;
    for (var rank = start; rank <= finish; rank++) { occupied = occupied || labels[labelsOffset + ordering[orderingOffset + rank]] != 0xffffffffu; }
    if (!occupied) {
      for (var rank = start; rank <= finish; rank++) { labels[labelsOffset + ordering[orderingOffset + rank]] = label; }
      label++;
    }
  }
}
summary[summaryOffset] = clusterCount;
areas[areasOffset + ${i*4}u] = areaCount;
areas[areasOffset + ${i*4+1}u] = next;
areas[areasOffset + ${i*4+2}u] = bitcast<u32>(maximumBetween);
areas[areasOffset + ${i*4+3}u] = label;
areas[areasOffset + BATCH_START_OFFSET] = batchStart + 64u;
`}),l(t,{id:`${r}-xi-gate-${e}`,operation:`GPUOPTICS`,variant:`xi-gate`,invocationCount:1,bindings:[{name:`areas`,view:o,type:`u32`,access:`read`},d.binding],body:`if (areas[areasOffset + ${i*4+1}u] >= areas[areasOffset + ${i*4+4}u] ||
      areas[areasOffset + ${i*4+5}u] >= areas[areasOffset + ${i*4+4}u]) {
    ${d.disarmWGSL}
  }`}));return f.push(l(t,{id:`${r}-xi-finalize`,operation:`GPUOPTICS`,variant:`xi-finalize`,invocationCount:i,bindings:[{name:`labels`,view:a.labels,type:`u32`,access:`read_write`},{name:`summary`,view:u,type:`u32`,access:`read`}],body:`if (summary[summaryOffset + 1u] != 0u) { labels[labelsOffset + index] = 0xffffffffu; }`})),f}function O(n,r,i){let a=`${r.id??`optics`}-parallel`,o=r.positions.length,{output:c}=r,{distanceSource:u,points:d,offsets:f,neighbors:p,overflow:m,processed:h}=i,g=e(n,`${a}-keys`,`uint32`,o),_=e(n,`${a}-ties`,`uint32`,o),v=e(n,`${a}-minimum`,`uint32`,1),y=e(n,`${a}-focus`,`uint32`,1),b=new t({id:`${a}-keys`,input:g,output:v,operation:`min`}).getCommandNodes(n),x=new t({id:`${a}-ties`,input:_,output:y,operation:`min`}).getCommandNodes(n),S=[],C=o;do C=Math.ceil(C/256),S.push(C);while(C>1);S.push(1);let w=[...new Set([1,Math.ceil(o/256),...S])],T=s(n,{id:`${a}-dispatch`,operation:`GPUOPTICS`,shapes:w.map(e=>[e*256,256])}),E=e=>T.getGate(w.indexOf(e)),D=E(Math.ceil(o/256)),O=[l(n,{id:`${a}-initialize`,operation:`GPUOPTICS`,variant:`parallel-initialize`,invocationCount:o,bindings:[{name:`processed`,view:h,type:`u32`,access:`read`},{name:`overflow`,view:m,type:`u32`,access:`read`},{name:`keys`,view:g,type:`u32`,access:`read_write`},{name:`status`,view:c.status,type:`u32`,access:`read_write`},T.binding],body:`keys[keysOffset + index] = select(0x7f800000u, 0xffffffffu, processed[processedOffset + index] != 0u);
  if (index == 0u) {
    status[statusOffset + 1u] = u32(status[statusOffset + 2u] == ${o}u && overflow[overflowOffset] == 0u);
    ${T.armWGSL}
    if (overflow[overflowOffset] != 0u || status[statusOffset + 1u] != 0u) { ${T.disarmWGSL} }
  }`})],k=l(n,{id:`${a}-mark-ties`,operation:`GPUOPTICS`,variant:`parallel-ties`,invocationCount:o,workgroupSize:256,...D,bindings:[{name:`keys`,view:g,type:`u32`,access:`read`},{name:`minimum`,view:v,type:`u32`,access:`read`},{name:`ties`,view:_,type:`u32`,access:`read_write`}],body:`let key = keys[keysOffset + index];
  ties[tiesOffset + index] = select(0xffffffffu, index, key != 0xffffffffu && key == minimum[minimumOffset]);`}),A=l(n,{id:`${a}-commit`,operation:`GPUOPTICS`,variant:`parallel-commit`,invocationCount:1,bindings:[{name:`focus`,view:y,type:`u32`,access:`read`},{name:`keys`,view:g,type:`u32`,access:`read_write`},{name:`processed`,view:h,type:`u32`,access:`read_write`},{name:`ordering`,view:c.ordering,type:`u32`,access:`read_write`},{name:`status`,view:c.status,type:`u32`,access:`read_write`},T.binding],body:`if (dispatch[dispatchOffset] == 0u) { return; }
  let row = focus[focusOffset];
  if (row == 0xffffffffu) { ${T.disarmWGSL} return; }
  let visited = status[statusOffset];
  ordering[orderingOffset + visited] = row;
  keys[keysOffset + row] = 0xffffffffu;
  processed[processedOffset + row] = 1u;
  status[statusOffset] = visited + 1u;
  if (visited + 1u + status[statusOffset + 2u] == ${o}u) {
    status[statusOffset + 1u] = 1u;
    ${T.disarmWGSL}
  }`}),j=l(n,{id:`${a}-relax`,operation:`GPUOPTICS`,variant:`parallel-relax`,invocationCount:o,workgroupSize:256,...D,bindings:[{name:`points`,view:d,type:`f32`,access:`read`},{name:`offsets`,view:f,type:`u32`,access:`read`},{name:`neighbors`,view:p,type:`u32`,access:`read`},{name:`focus`,view:y,type:`u32`,access:`read`},{name:`keys`,view:g,type:`u32`,access:`read_write`},{name:`processed`,view:h,type:`u32`,access:`read`},{name:`reachability`,view:c.reachability,type:`f32`,access:`read_write`},{name:`predecessors`,view:c.predecessors,type:`u32`,access:`read_write`}],declarations:`${u}
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + 4u * row], points[pointsOffset + 4u * row + 1u]);
}`,body:`let selected = focus[focusOffset];
  let begin = offsets[offsetsOffset + selected];
  let end = offsets[offsetsOffset + selected + 1u];
  if (index >= end - begin) { return; }
  let row = neighbors[neighborsOffset + begin + index];
  if (processed[processedOffset + row] != 0u) { return; }
  let coreDistance = points[pointsOffset + 4u * selected + 3u];
  if (bitcast<u32>(coreDistance) == 0x7f800000u) { return; }
  ${r.times?`if (abs(points[pointsOffset + 4u * selected + 2u] - points[pointsOffset + 4u * row + 2u]) > ${Math.fround(r.maximumTimeDistance)}f) { return; }`:``}
  let pairDistance = getPointDistance(point(selected), point(row));
  if (pairDistance > ${Math.fround(r.maximumDistance)}f) { return; }
  let candidate = max(coreDistance, pairDistance);
  if (candidate < reachability[reachabilityOffset + row]) {
    reachability[reachabilityOffset + row] = candidate;
    predecessors[predecessorsOffset + row] = selected;
    keys[keysOffset + row] = bitcast<u32>(candidate);
  }`}),M=(e,t,n)=>{if(e.type!==`compute`)throw Error(`${a} GPUReduction nodes must be compute nodes`);let r=E(S[t]);return{...e,id:`${e.id}-${n}`,condition:r.condition,resources:[...e.resources??[],...r.extraResources]}};for(let e=0;e<(r.maximumIterations??o);e++)O.push(...b.map((t,n)=>M(t,n,e)),{...k,id:`${k.id}-${e}`},...x.map((t,n)=>M(t,n,e)),{...A,id:`${A.id}-${e}`},{...j,id:`${j.id}-${e}`});return O}var k=8,A=4096,j=2**20,M=65536,N=9*M+2*Math.ceil(M/64)+256,P=256,F=5;function I(e){let t=e,n=0;do t=Math.ceil(t/P),n++;while(t>1);return 2*n+F}var L=256,R=256,z=2**31-1,B=[`iterationCount`,`converged`,`iterationLimitReached`];function V(e){let{ports:t,...n}=e;return[...Object.values(n),...Object.values(t?.status??{})].filter(e=>e!==void 0)}var H=class{id;props;spatialContext;temporalContext;neighborIndex;neighborCapacity;estimatedCommandNodeCount;constructor(e){this.props=e,this.id=e.id??`optics`,this.spatialContext=o(this.id,e.spatialContext,{...i,measure:`distance`});let t=e.positions.length;e.times?(u(this.id,e.temporalContext!==void 0,`times need a temporalContext`),this.temporalContext=g(this.id,e.temporalContext,{encodings:[`float32-relative`]}),b(e.times,`float32`,t,`${this.id} times`),x(e.maximumTimeDistance??NaN,this.id,`maximumTimeDistance`)):u(this.id,e.temporalContext===void 0&&e.maximumTimeDistance===void 0,`temporalContext and maximumTimeDistance need times`);let n=e.neighborIndex??`auto`;u(this.id,[`auto`,`brute-force`,`grid`].includes(n),`neighborIndex must be 'auto', 'brute-force' or 'grid'`),this.neighborIndex=n===`grid`||n===`auto`&&t>4096?`grid`:`brute-force`,d(this.id,`positions rows`,t,1,this.neighborIndex===`grid`?j:A),u(this.id,[`sequential`,`parallel`].includes(e.orderingMethod??`sequential`),`orderingMethod must be 'sequential' or 'parallel'`),u(this.id,e.orderingMethod!==`parallel`||this.neighborIndex===`grid`,`orderingMethod 'parallel' needs neighborIndex 'grid'`),this.neighborCapacity=this.neighborIndex===`grid`?e.neighborCapacity??Math.max(1,t*Math.min(t-1,64)):0,this.neighborIndex===`grid`?d(this.id,`neighborCapacity`,this.neighborCapacity,1,z):u(this.id,e.neighborCapacity===void 0&&e.output.ports?.status.candidateOverflow===void 0,`neighborCapacity and output.ports.status.candidateOverflow need neighborIndex 'grid'`),d(this.id,`minimumSamples`,e.minimumSamples,1,L);let r=e.maximumIterations??t;d(this.id,`maximumIterations`,r,1,t),x(e.maximumDistance,this.id,`maximumDistance`),u(this.id,e.extractionMethod===void 0||[`dbscan`,`xi`].includes(e.extractionMethod),`extractionMethod must be 'dbscan' or 'xi'`),(e.extractionMethod??`dbscan`)===`dbscan`&&(x(e.extractionDistance??NaN,this.id,`extractionDistance`),u(this.id,e.extractionDistance<=e.maximumDistance,`extractionDistance must be at most maximumDistance`));let a=e.xi??.05;u(this.id,Number.isFinite(a)&&a>0&&a<1,`xi must be in (0, 1)`);let s=e.minimumClusterSize??e.minimumSamples;e.extractionMethod===`xi`&&(d(this.id,`xi positions rows`,t,1,M),d(this.id,`minimumClusterSize`,s,2,t));let c=e.frontierBatchSize??256;d(this.id,`frontierBatchSize`,c,1,R),this.estimatedCommandNodeCount=(e.orderingMethod===`parallel`?I(t)*r:Math.ceil(r/c)*(this.neighborIndex===`grid`?2:1))+(e.extractionMethod===`xi`?2*Math.ceil(t/64):0)+256,f(this.id,this.estimatedCommandNodeCount,e.orderingMethod===`parallel`?N:void 0),(e.output.clusterIntervals||e.output.clusterSummary)&&u(this.id,e.extractionMethod===`xi`,`clusterIntervals and clusterSummary need extractionMethod 'xi'`),e.output.clusterIntervals&&b(e.output.clusterIntervals,`uint32x2`,t,`${this.id} output.clusterIntervals`),e.output.clusterSummary&&b(e.output.clusterSummary,`uint32`,2,`${this.id} output.clusterSummary`),b(e.positions,`float32x2`,t,`${this.id} positions`);for(let n of[`ordering`,`predecessors`,`labels`])b(e.output[n],`uint32`,t,`${this.id} output.${n}`);for(let n of[`coreDistances`,`reachability`])b(e.output[n],`float32`,t,`${this.id} output.${n}`);b(e.output.status,`uint32`,3,`${this.id} output.status`);for(let[t,n]of Object.entries(e.output.ports?.status??{}))n&&b(n,`uint32`,1,`${this.id} output.ports.status.${t}`);h(this.id,V(e.output),[e.positions,e.times].filter(Boolean))}getCommandNodes(i){let{id:o,props:s}=this,{positions:u,output:d,minimumSamples:f}=s,h=u.length,{times:g}=s;m(o,i,[u,g,...V(d)]);for(let[e,t]of[[`row scratch`,h*4],[`neighbor offsets`,(h+1)*4],[`frontier points`,this.neighborIndex===`grid`?h*16:0],[`frontier scratch`,this.neighborIndex===`grid`?(h*2+4)*4:0],[`xi areas`,s.extractionMethod===`xi`?h*16+24:0],[`core neighbor scratch`,h*Math.max(0,f-1)*4],[`neighborCapacity`,this.neighborCapacity*4]])p(i.device,o,e,t,`lower the row count or capacity`);let b=g?[{name:`times`,view:g,type:`f32`,access:`read`}]:[],x=e(i,`${o}-processed`,`uint32`,h),S=e(i,`${o}-invalid-rows`,`uint32`,h),C=this.spatialContext.coordinateSpace===`longitude-latitude`,w=C?`${r}
fn getPointDistance(first: vec2<f32>, second: vec2<f32>) -> f32 { return geodesicHaversineDistance(first, second, ${c(a(this.spatialContext))}); }`:`fn getPointDistance(first: vec2<f32>, second: vec2<f32>) -> f32 { return distance(first, second); }`,T=`${y}
${w}
const ROWS = ${h}u;
const MAXIMUM_DISTANCE = ${Math.fround(s.maximumDistance)}f;
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn validPoint(row: u32) -> bool {
  let value = point(row);
  return isFiniteFloat(value.x) && isFiniteFloat(value.y)${C?` && abs(value.x) <= 180.0 && abs(value.y) <= 90.0`:``}${g?` && isFiniteFloat(times[timesOffset + row])`:``};
}
${g?`const MAXIMUM_TIME_DISTANCE = ${Math.fround(s.maximumTimeDistance)}f;
fn isRelated(first: u32, second: u32) -> bool {
  return abs(times[timesOffset + first] - times[timesOffset + second]) <= MAXIMUM_TIME_DISTANCE;
}`:``}`,E=f-1,O=[],A,j;if(E>0){let t=v(i,{id:`${o}-core-neighbors-setup`,operation:`GPUOPTICS`,positions:u,radius:s.maximumDistance,geographic:C,times:g,timeRadius:s.maximumTimeDistance}),n=h*E;A=e(i,`${o}-core-neighbor-offsets`,`uint32`,h+1),j=e(i,`${o}-core-neighbor-distances`,`float32`,n),O.push(...t.nodes,...new _({id:`${o}-core-neighbors`,mode:`knn`,k:E,spatialContext:this.spatialContext,positions:u,times:g,temporalContext:this.temporalContext,timeBucketCount:g?k:void 0,parameters:t.parameters,gridSize:t.gridSize,output:{offsets:A,neighbors:e(i,`${o}-core-neighbor-ids`,`uint32`,n),weights:e(i,`${o}-core-neighbor-weights`,`float32`,n),distances:j,status:{overflow:e(i,`${o}-core-neighbor-overflow`,`uint32`,1)}}}).getCommandNodes(i))}let M=[...O,l(i,{id:`${o}-core`,operation:`GPUOPTICS`,variant:`core`,invocationCount:h,bindings:[{name:`positions`,view:u,type:`f32`,access:`read`},...b,...A&&j?[{name:`neighborOffsets`,view:A,type:`u32`,access:`read`},{name:`neighborDistances`,view:j,type:`f32`,access:`read`}]:[],{name:`core`,view:d.coreDistances,type:`f32`,access:`read_write`}],declarations:T,body:`var coreDistance = bitcast<f32>(0x7f800000u | (index & 0u));
  if (validPoint(index)) {
    ${A?`let begin = neighborOffsets[neighborOffsetsOffset + index];
    let end = neighborOffsets[neighborOffsetsOffset + index + 1u];
    if (end - begin + 1u >= ${f}u) {
      coreDistance = 0.0;
      for (var slot = begin; slot < end; slot++) {
        coreDistance = max(coreDistance, neighborDistances[neighborDistancesOffset + slot]);
      }
    }`:`coreDistance = 0.0;`}
  }
  core[coreOffset + index] = coreDistance;`}),l(i,{id:`${o}-initialize`,operation:`GPUOPTICS`,variant:`frontier-initialize`,invocationCount:h,bindings:[{name:`positions`,view:u,type:`f32`,access:`read`},...b,{name:`ordering`,view:d.ordering,type:`u32`,access:`read_write`},{name:`reachability`,view:d.reachability,type:`f32`,access:`read_write`},{name:`predecessors`,view:d.predecessors,type:`u32`,access:`read_write`},{name:`processed`,view:x,type:`u32`,access:`read_write`},{name:`invalidRows`,view:S,type:`u32`,access:`read_write`},{name:`labels`,view:d.labels,type:`u32`,access:`read_write`}],declarations:T,body:`let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  let invalid = u32(!validPoint(index));
  invalidRows[invalidRowsOffset + index] = invalid;
  processed[processedOffset + index] = invalid;
  ordering[orderingOffset + index] = 0xffffffffu;
  predecessors[predecessorsOffset + index] = 0xffffffffu;
  reachability[reachabilityOffset + index] = infinity;
  labels[labelsOffset + index] = 0xffffffffu;`}),l(i,{id:`${o}-initialize-status`,operation:`GPUOPTICS`,variant:`frontier-initialize-status`,invocationCount:1,bindings:[{name:`status`,view:d.status,type:`u32`,access:`read_write`}],body:`status[statusOffset] = 0u;
  status[statusOffset + 1u] = 0u;`}),...new t({id:`${o}-invalid-count`,input:S,output:i.createDataView(d.status.buffer,{format:`uint32`,byteOffset:d.status.byteOffset+2*Uint32Array.BYTES_PER_ELEMENT,length:1}),operation:`sum`}).getCommandNodes(i)];this.neighborIndex===`grid`&&M.push(...this.getGridFrontierNodes(i,w,b,x));for(let e=0;this.neighborIndex===`brute-force`&&e<Math.ceil((s.maximumIterations??h)/(s.frontierBatchSize??256));e++)M.push(l(i,{id:`${o}-frontier-${e}`,operation:`GPUOPTICS`,variant:`frontier-batch`,invocationCount:256,workgroupSize:256,guardIndex:!1,bindings:[{name:`positions`,view:u,type:`f32`,access:`read`},...b,{name:`core`,view:d.coreDistances,type:`f32`,access:`read`},{name:`ordering`,view:d.ordering,type:`u32`,access:`read_write`},{name:`reachability`,view:d.reachability,type:`f32`,access:`read_write`},{name:`predecessors`,view:d.predecessors,type:`u32`,access:`read_write`},{name:`processed`,view:x,type:`u32`,access:`read_write`},{name:`status`,view:d.status,type:`u32`,access:`read_write`}],declarations:`${T}
var<workgroup> candidateRows: array<u32, 256>;
var<workgroup> candidateDistances: array<f32, 256>;
var<workgroup> visited: u32;`,body:`let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  if (index == 0u) { visited = status[statusOffset]; }
  workgroupBarrier();
  for (var iteration = 0u; iteration < ${Math.min(s.frontierBatchSize??256,(s.maximumIterations??h)-e*(s.frontierBatchSize??256))}u; iteration++) {
    var selected = 0xffffffffu; var best = infinity;
    for (var row = index; row < ROWS; row += 256u) {
      let candidate = reachability[reachabilityOffset + row];
      if (processed[processedOffset + row] == 0u && (candidate < best || (candidate == best && row < selected))) {
        selected = row; best = candidate;
      }
    }
    candidateRows[index] = selected; candidateDistances[index] = best;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
      if (index < stride) {
        let otherDistance = candidateDistances[index + stride]; let otherRow = candidateRows[index + stride];
        if (otherDistance < candidateDistances[index] || (otherDistance == candidateDistances[index] && otherRow < candidateRows[index])) {
          candidateDistances[index] = otherDistance; candidateRows[index] = otherRow;
        }
      }
      workgroupBarrier();
    }
    let focus = workgroupUniformLoad(&candidateRows[0]);
    if (focus == 0xffffffffu) { break; }
    if (index == 0u) { ordering[orderingOffset + visited] = focus; visited++; processed[processedOffset + focus] = 1u; }
    storageBarrier(); workgroupBarrier();
    let coreDistance = core[coreOffset + focus];
    if (isFiniteFloat(coreDistance)) {
      for (var row = index; row < ROWS; row += 256u) {
        if (processed[processedOffset + row] != 0u) { continue; }
        ${g?`if (!isRelated(focus, row)) { continue; }
        `:``}let pairDistance = getPointDistance(point(focus), point(row));
        if (pairDistance > MAXIMUM_DISTANCE) { continue; }
        let candidate = max(coreDistance, pairDistance);
        if (candidate < reachability[reachabilityOffset + row]) {
          reachability[reachabilityOffset + row] = candidate; predecessors[predecessorsOffset + row] = focus;
        }
      }
    }
    storageBarrier(); workgroupBarrier();
  }
  if (index == 0u) {
    status[statusOffset] = visited;
    status[statusOffset + 1u] = select(0u, 1u, visited + status[statusOffset + 2u] == ROWS);
  }`}));let N=B.filter(e=>d.ports?.status[e]);if(N.length&&M.push(l(i,{id:`${o}-iteration-status`,operation:`GPUOPTICS`,variant:`iteration-status`,invocationCount:1,bindings:[{name:`status`,view:d.status,type:`u32`,access:`read`},...d.ports?.status.candidateOverflow?[{name:`overflow`,view:d.ports.status.candidateOverflow,type:`u32`,access:`read`}]:[],...N.map(e=>({name:e,view:d.ports.status[e],type:`u32`,access:`read_write`}))],body:N.map(e=>`${e}[${e}Offset] = ${e===`iterationCount`?`status[statusOffset]`:e===`converged`?`status[statusOffset + 1u]`:`u32(status[statusOffset + 1u] == 0u && status[statusOffset] == ${s.maximumIterations??h}u${d.ports?.status.candidateOverflow?` && overflow[overflowOffset] == 0u`:``})`};`).join(`
`)})),s.extractionMethod===`xi`)return M.push(...D(i,s,o)),M;let P=e(i,`${o}-cluster-starts`,`uint32`,h),F=e(i,`${o}-cluster-counts`,`uint32`,h);return M.push(l(i,{id:`${o}-cluster-starts`,operation:`GPUOPTICS`,variant:`dbscan-cluster-starts`,invocationCount:h,bindings:[{name:`core`,view:d.coreDistances,type:`f32`,access:`read`},{name:`ordering`,view:d.ordering,type:`u32`,access:`read`},{name:`reachability`,view:d.reachability,type:`f32`,access:`read`},{name:`status`,view:d.status,type:`u32`,access:`read`},{name:`starts`,view:P,type:`u32`,access:`read_write`}],body:`var startsCluster = false;
  if (status[statusOffset + 1u] != 0u && index < status[statusOffset]) {
    let row = ordering[orderingOffset + index];
    startsCluster = reachability[reachabilityOffset + row] > ${Math.fround(s.extractionDistance)}f && core[coreOffset + row] <= ${Math.fround(s.extractionDistance)}f;
  }
  starts[startsOffset + index] = u32(startsCluster);`}),...new n({id:`${o}-cluster-scan`,input:P,output:F,mode:`inclusive`}).getCommandNodes(i),l(i,{id:`${o}-extract`,operation:`GPUOPTICS`,variant:`dbscan-extraction`,invocationCount:h,bindings:[{name:`starts`,view:P,type:`u32`,access:`read`},{name:`counts`,view:F,type:`u32`,access:`read`},{name:`ordering`,view:d.ordering,type:`u32`,access:`read`},{name:`reachability`,view:d.reachability,type:`f32`,access:`read`},{name:`status`,view:d.status,type:`u32`,access:`read`},{name:`labels`,view:d.labels,type:`u32`,access:`read_write`}],body:`if (status[statusOffset + 1u] == 0u || index >= status[statusOffset]) { return; }
  let row = ordering[orderingOffset + index];
  if (starts[startsOffset + index] != 0u || reachability[reachabilityOffset + row] <= ${Math.fround(s.extractionDistance)}f) {
    labels[labelsOffset + row] = counts[countsOffset + index] - 1u;
  }`})),M}getGridFrontierNodes(t,n,r,i){let{id:a,props:o,neighborCapacity:c}=this,{positions:u,output:d,times:f}=o,p=u.length,m=this.spatialContext.coordinateSpace===`longitude-latitude`,h=v(t,{id:`${a}-reach-neighbors-setup`,operation:`GPUOPTICS`,positions:u,radius:o.maximumDistance*1.0001,geographic:m,times:f,timeRadius:o.maximumTimeDistance===void 0?void 0:o.maximumTimeDistance*1.0001}),g=e(t,`${a}-reach-neighbor-offsets`,`uint32`,p+1),b=e(t,`${a}-reach-neighbor-ids`,`uint32`,c),x=d.ports?.status.candidateOverflow??e(t,`${a}-reach-neighbor-overflow`,`uint32`,1),S=e(t,`${a}-frontier-points`,`float32`,p*4),C=e(t,`${a}-frontier-scratch`,`uint32`,p*2+4),w=o.orderingMethod===`parallel`?void 0:s(t,{id:`${a}-frontier-dispatch`,operation:`GPUOPTICS`,shapes:[[256,256]]}),T=`${y}
${n}
const ROWS = ${p}u;
const MAXIMUM_DISTANCE = ${Math.fround(o.maximumDistance)}f;
const ROW_STATES = ${p}u;
const SEED_COUNT = ${p*2}u;
const CURSOR = ${p*2+1}u;
const VISITED = ${p*2+2}u;
const OVERFLOW = ${p*2+3}u;
// A row state is UNVISITED, VISITED, or the row's slot in the seed list.
const UNVISITED = 0xffffffffu;
const VISITED_ROW = 0xfffffffeu;
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + 4u * row], points[pointsOffset + 4u * row + 1u]);
}
${f?`const MAXIMUM_TIME_DISTANCE = ${Math.fround(o.maximumTimeDistance)}f;
fn isRelated(first: u32, second: u32) -> bool {
  return abs(points[pointsOffset + 4u * first + 2u] - points[pointsOffset + 4u * second + 2u]) <= MAXIMUM_TIME_DISTANCE;
}`:``}`,E=[...h.nodes,...new _({id:`${a}-reach-neighbors`,mode:`radius`,spatialContext:this.spatialContext,positions:u,times:f,temporalContext:this.temporalContext,timeBucketCount:f?k:void 0,parameters:h.parameters,gridSize:h.gridSize,output:{offsets:g,neighbors:b,weights:e(t,`${a}-reach-neighbor-weights`,`float32`,c),status:{overflow:x}}}).getCommandNodes(t),l(t,{id:`${a}-frontier-prepare`,operation:`GPUOPTICS`,variant:`grid-frontier-prepare`,invocationCount:p,bindings:[{name:`positions`,view:u,type:`f32`,access:`read`},...r,{name:`core`,view:d.coreDistances,type:`f32`,access:`read`},{name:`processed`,view:i,type:`u32`,access:`read`},{name:`overflow`,view:x,type:`u32`,access:`read`},{name:`points`,view:S,type:`f32`,access:`read_write`},{name:`scratch`,view:C,type:`u32`,access:`read_write`},...w?[w.binding]:[]],body:`points[pointsOffset + 4u * index] = positions[positionsOffset + 2u * index];
  points[pointsOffset + 4u * index + 1u] = positions[positionsOffset + 2u * index + 1u];
  points[pointsOffset + 4u * index + 2u] = ${f?`times[timesOffset + index]`:`0.0`};
  points[pointsOffset + 4u * index + 3u] = core[coreOffset + index];
  // Invalid rows were marked processed by the initialize pass.
  scratch[scratchOffset + ${p}u + index] = select(0xffffffffu, 0xfffffffeu, processed[processedOffset + index] != 0u);
  if (index == 0u) {
    scratch[scratchOffset + ${p*2}u] = 0u;
    scratch[scratchOffset + ${p*2+1}u] = 0u;
    scratch[scratchOffset + ${p*2+2}u] = 0u;
    scratch[scratchOffset + ${p*2+3}u] = overflow[overflowOffset];
    ${w?`${w.armWGSL}\n    if (overflow[overflowOffset] != 0u) { ${w.disarmWGSL} }`:``}
  }`})];if(o.orderingMethod===`parallel`)return E.push(...O(t,o,{distanceSource:n,points:S,offsets:g,neighbors:b,overflow:x,processed:i})),E;let D=o.frontierBatchSize??256,A=o.maximumIterations??p,j=[{name:`points`,view:S,type:`f32`,access:`read`},{name:`offsets`,view:g,type:`u32`,access:`read`},{name:`neighbors`,view:b,type:`u32`,access:`read`},{name:`ordering`,view:d.ordering,type:`u32`,access:`read_write`},{name:`reachability`,view:d.reachability,type:`f32`,access:`read_write`},{name:`predecessors`,view:d.predecessors,type:`u32`,access:`read_write`},{name:`scratch`,view:C,type:`u32`,access:`read_write`}];for(let e=0;e<Math.ceil(A/D);e++)E.push(l(t,{id:`${a}-frontier-${e}`,operation:`GPUOPTICS`,variant:`grid-frontier-batch`,invocationCount:256,workgroupSize:256,guardIndex:!1,...w.getGate(0),bindings:j,declarations:`${T}
var<workgroup> candidateRows: array<u32, 256>;
var<workgroup> candidateDistances: array<f32, 256>;
var<workgroup> visited: u32;
var<workgroup> seedCount: u32;
var<workgroup> cursor: u32;
var<workgroup> focusRow: u32;
var<workgroup> appendCount: atomic<u32>;
var<workgroup> overflowed: u32;`,body:`let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  if (index == 0u) {
    overflowed = scratch[scratchOffset + OVERFLOW];
    visited = scratch[scratchOffset + VISITED];
    seedCount = scratch[scratchOffset + SEED_COUNT];
    cursor = scratch[scratchOffset + CURSOR];
    atomicStore(&appendCount, 0u);
  }
  // A truncated candidate list would drop relaxations: leave the ordering incomplete.
  if (workgroupUniformLoad(&overflowed) != 0u) { return; }
  for (var iteration = 0u; iteration < ${Math.min(D,A-e*D)}u; iteration++) {
    let count = workgroupUniformLoad(&seedCount);
    if (count > 0u) {
      var selected = 0xffffffffu; var best = infinity;
      for (var slot = index; slot < count; slot += 256u) {
        let row = scratch[scratchOffset + slot];
        let candidate = reachability[reachabilityOffset + row];
        if (candidate < best || (candidate == best && row < selected)) { selected = row; best = candidate; }
      }
      candidateRows[index] = selected; candidateDistances[index] = best;
      workgroupBarrier();
      for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (index < stride) {
          let otherDistance = candidateDistances[index + stride]; let otherRow = candidateRows[index + stride];
          if (otherDistance < candidateDistances[index] || (otherDistance == candidateDistances[index] && otherRow < candidateRows[index])) {
            candidateDistances[index] = otherDistance; candidateRows[index] = otherRow;
          }
        }
        workgroupBarrier();
      }
      if (index == 0u) { focusRow = candidateRows[0]; }
    } else {
      // Every unvisited row has infinite reachability: the next root is the lowest unvisited row.
      if (index == 0u) { focusRow = 0xffffffffu; }
      loop {
        let base = workgroupUniformLoad(&cursor);
        if (base >= ROWS) { break; }
        let row = base + index;
        candidateRows[index] = select(0xffffffffu, row, row < ROWS && scratch[scratchOffset + ROW_STATES + min(row, ROWS - 1u)] == UNVISITED);
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride /= 2u) {
          if (index < stride) { candidateRows[index] = min(candidateRows[index], candidateRows[index + stride]); }
          workgroupBarrier();
        }
        if (index == 0u) {
          if (candidateRows[0] == 0xffffffffu) { cursor = base + 256u; } else { cursor = candidateRows[0]; focusRow = candidateRows[0]; }
        }
        if (workgroupUniformLoad(&focusRow) != 0xffffffffu) { break; }
      }
    }
    let focus = workgroupUniformLoad(&focusRow);
    if (focus == 0xffffffffu) { break; }
    if (index == 0u) {
      ordering[orderingOffset + visited] = focus; visited++;
      let slot = scratch[scratchOffset + ROW_STATES + focus];
      if (slot != UNVISITED) {
        // The focus is a seed: move the last seed into its slot.
        let last = scratch[scratchOffset + seedCount - 1u];
        scratch[scratchOffset + slot] = last; scratch[scratchOffset + ROW_STATES + last] = slot;
        seedCount--;
      }
      scratch[scratchOffset + ROW_STATES + focus] = VISITED_ROW;
    }
    storageBarrier(); workgroupBarrier();
    let coreDistance = points[pointsOffset + 4u * focus + 3u];
    let seedBase = workgroupUniformLoad(&seedCount);
    if (isFiniteFloat(coreDistance)) {
      let end = offsets[offsetsOffset + focus + 1u];
      for (var slot = offsets[offsetsOffset + focus] + index; slot < end; slot += 256u) {
        let row = neighbors[neighborsOffset + slot];
        let state = scratch[scratchOffset + ROW_STATES + row];
        if (state == VISITED_ROW) { continue; }
        ${f?`if (!isRelated(focus, row)) { continue; }
        `:``}let pairDistance = getPointDistance(point(focus), point(row));
        if (pairDistance > MAXIMUM_DISTANCE) { continue; }
        let candidate = max(coreDistance, pairDistance);
        if (candidate < reachability[reachabilityOffset + row]) {
          reachability[reachabilityOffset + row] = candidate; predecessors[predecessorsOffset + row] = focus;
          if (state == UNVISITED) {
            let seedSlot = seedBase + atomicAdd(&appendCount, 1u);
            scratch[scratchOffset + seedSlot] = row; scratch[scratchOffset + ROW_STATES + row] = seedSlot;
          }
        }
      }
    }
    storageBarrier(); workgroupBarrier();
    if (index == 0u) { seedCount += atomicExchange(&appendCount, 0u); }
    workgroupBarrier();
  }
  if (index == 0u) {
    scratch[scratchOffset + SEED_COUNT] = seedCount;
    scratch[scratchOffset + CURSOR] = cursor;
    scratch[scratchOffset + VISITED] = visited;
  }`}),l(t,{id:`${a}-frontier-gate-${e}`,operation:`GPUOPTICS`,variant:`grid-frontier-gate`,invocationCount:1,bindings:[{name:`scratch`,view:C,type:`u32`,access:`read`},{name:`status`,view:d.status,type:`u32`,access:`read`},w.binding],body:`if (scratch[scratchOffset + ${p*2}u] == 0u &&
      (scratch[scratchOffset + ${p*2+1}u] >= ${p}u ||
       scratch[scratchOffset + ${p*2+2}u] + status[statusOffset + 2u] == ${p}u)) {
    ${w.disarmWGSL}
  }`}));return E.push(l(t,{id:`${a}-frontier-status`,operation:`GPUOPTICS`,variant:`grid-frontier-status`,invocationCount:1,bindings:[{name:`scratch`,view:C,type:`u32`,access:`read`},{name:`status`,view:d.status,type:`u32`,access:`read_write`}],body:`let visited = scratch[scratchOffset + ${p*2+2}u];
  status[statusOffset] = visited;
  status[statusOffset + 1u] = select(0u, 1u, visited + status[statusOffset + 2u] == ${p}u);`})),E}},U=500,W=.3,G=[.04,.04,.96,.26],K=40,q=.22,J=S({parameters:E,caption:`OPTICS reachability plot and the clusters extracted with the xi method.`,createInputs:({random:e,world:t})=>{let n=C(e,{width:t.width,height:t.height-W},U,4,.12);for(let e=0;e<U;e++)n.positions[e*2+1]+=W;return{...n,index:w(n.positions),focus:{row:-1},lit:{key:``,labels:null}}},build(e,t,n){let r=e.allocate(`ordering`,`uint32`,t.count),i=e.allocate(`core-distances`,`float32`,t.count),a=e.allocate(`reachability`,`float32`,t.count),o=e.allocate(`predecessors`,`uint32`,t.count),s=e.allocate(`labels`,`uint32`,t.count),c=e.allocate(`status`,`uint32`,3);return e.graph.add(new H({positions:e.upload(`positions`,`float32x2`,t.positions).view,minimumSamples:n.minimumSamples,maximumDistance:n.maximumDistance,extractionMethod:`xi`,xi:n.xi,output:{ordering:r.view,coreDistances:i.view,reachability:a.view,predecessors:o.view,labels:s.view,status:c.view}})),{outputs:{labels:s,ordering:r,reachability:a,predecessors:o,plot:e.allocate(`plot`,`float32`,U),memberDots:e.allocate(`member-dots`,`float32x2`,U),clusterPlot:e.allocate(`cluster-plot`,`float32`,U),positions:e.upload(`points`,`float32x2`,t.positions)}}},interaction:{kind:`probe`,hint:`hover a point`,rest:[.6,.4],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=a(n.ordering),s=a(n.reachability),c=a(n.predecessors),l=a(n.labels),u=T(t.index,e.x,e.y,t.focus.row);if(l){let n=q;for(let r=0;r<U;r++){if(l[r]===4294967295)continue;let i=t.index.distanceTo(r,e.x,e.y);i<n&&(n=i,u=r)}}if(t.focus.row=u,!o||!s||!c||!l)return{};let d=new Float32Array(U),f=new Uint32Array(U);for(let e=0;e<U;e++){let t=o[e];f[t]=e;let n=s[t];d[e]=Number.isFinite(n)?Math.min(n,r.maximumDistance):r.maximumDistance}n.plot.write(d);let p=G[1]*i.height,m=[{shape:`line`,from:[G[0]*i.width,p],to:[G[2]*i.width,p]}],h=u<0||l[u]===4294967295?-1:l[u],g=`${h}`;if(t.lit.key!==g||t.lit.labels!==l){t.lit={key:g,labels:l};let e=new Float32Array(U*2).fill(-10),r=new Float32Array(U);for(let n=0;n<U;n++)h<0||l[n]!==h||(e.set([t.positions[n*2],t.positions[n*2+1]],n*2),r[f[n]]=d[f[n]]);n.memberDots.write(e),n.clusterPlot.write(r)}if(u<0)return{glyphs:m};let _=f[u],v=(G[0]+(_+.5)/U*(G[2]-G[0]))*i.width;m.push({shape:`line`,from:[v,p],to:[v,G[3]*i.height]});let y=[t.positions[u*2],t.positions[u*2+1]],b=u;for(let e=0;e<K;e++){let e=c[b];if(e===4294967295||e>=U)break;y.push(t.positions[e*2],t.positions[e*2+1]),b=e}y.length>2&&m.push({shape:`polyline`,points:y}),m.push({shape:`crosshair`,at:[y[0],y[1]],size:12});let x=s[u],S=l[u];return{glyphs:m,readout:`order ${_} · reachability ${Number.isFinite(x)?x.toFixed(3):`none`} · ${S===4294967295?`no cluster`:`cluster ${S}`}`}}},draw:({outputs:e,values:t,theme:n})=>[{layer:`dots`,context:!0,positions:e.positions,radius:2},{layer:`dots`,positions:e.memberDots,radius:10,color:[...n.ink,.9]},{layer:`dots`,positions:e.positions,values:e.labels,radius:4.5},{layer:`chart`,kind:`bars`,values:e.plot,range:[0,t.maximumDistance],rectangle:G,color:[...n.categories[0],.85]},{layer:`chart`,kind:`bars`,values:e.clusterPlot,range:[0,t.maximumDistance],rectangle:G,color:[...n.ink,.95]}]});export{J as default};
import{n as e}from"./gpu-terrain-flow-BhpjXagd.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-sort-xRm8l5yn.js";import{r}from"./raster-kernel-utils-DcIN7G5I.js";import{d as i,u as a}from"./raster-band-utils-BcTxc8_Y.js";import{D as o,S as s,T as c,f as l,v as u}from"./gpu-contributor-utils-BIwLQON5.js";import{u as d}from"./raster-grid-VB0AM_GY.js";import{c as f,o as p}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as m}from"./sorted-segment-offsets-ChmOtA9F.js";import{a as h}from"./sorted-segment-sums-Dvmm6OdR.js";import{a as g,o as _,t as v}from"./fixed-point-accumulator-Ce6W3HVY.js";import{a as y,c as b,l as x,n as S,o as C,r as w,t as T,u as E}from"./terrain-processing-utils-D-TSF_vl.js";import{i as D,r as O}from"./terrain-flow-tracing-BfyLX2kv.js";import{n as k,t as A}from"./terrain-resolution-status-DTEt89JU.js";import{t as j}from"./gpu-terrain-watersheds-DRm5bMzG.js";import{t as M}from"./define-run-B-brkCe4.js";import{o as N}from"./pick-Bo3t3hjv.js";import{ot as P}from"./entry-preview-D70TdSbK.js";import{d as F,l as I,o as L,r as R,s as z,u as B}from"./terrain-Bwc0vRX3.js";var V=4294967295;function H(e,t,n,r){for(let[i,a]of[[`basinCount x 9 sum words`,n*9],[`basinCount x histogramBins x 3 histogram words`,n*r*3],[`cell tiles x (histogramBins + 1)`,Math.ceil(t/256)*(r+1)]])s(e,i,a,V)}function U(e,t,n,r,i){for(let[a,o]of[[`basin metadata`,r*5],[`basin sum words`,r*9],[`histogram sum words`,r*i*3],[`profile state`,n*4]])u(e.device,t,a,o*4,`reduce grid size, basinCount or histogramBins`)}function W(e,n){let{id:r,cellCount:i,basinCount:a,histogramBins:o,isUniform:s,area:c,declarations:l,inputs:u}=n,d=`GPUTerrainCatchmentSummary`,f=Math.ceil(i/256),m=t(e,`${r}-basin-metadata`,`uint32`,a*5),h=t(e,`${r}-basin-sums`,`uint32`,a*9),y=t(e,`${r}-histogram-sums`,`uint32`,a*o*3),x=_(i),S=`${v}
fn getElevationKey(value: f32) -> u32 {
 let bits = bitcast<u32>(select(value, 0.0, value == 0.0));
 return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn getElevationFromKey(key: u32) -> f32 {
 return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}`,C=(e,t)=>({name:e,view:t,type:`atomic<u32>`,access:`read_write`});return[p(e,{id:`${r}-reduction-clear`,operation:d,invocationCount:Math.max(m.length,h.length,y.length),bindings:[b(`metadata`,m,`read_write`),b(`sums`,h,`read_write`),b(`histogram`,y,`read_write`)],body:`if (index < ${m.length}u) { metadata[metadataOffset + index] = select(0u, 0xffffffffu, index % 5u == 0u); }
if (index < ${h.length}u) { sums[sumsOffset + index] = 0u; }
if (index < ${y.length}u) { histogram[histogramOffset + index] = 0u; }`}),p(e,{id:`${r}-metadata-tiles`,operation:d,invocationCount:f,bindings:[...u,b(`order`,n.order),C(`metadata`,m)],declarations:`${l}\n${S}
fn publishMetadata(basin: u32, lower: u32, upper: u32, exponents: vec3<u32>) {
 if (basin >= ${a}u) { return; }
 let offset = metadataOffset + basin * 5u;
 atomicMin(&metadata[offset], lower); atomicMax(&metadata[offset + 1u], upper);
 atomicMax(&metadata[offset + 2u], exponents.x); atomicMax(&metadata[offset + 3u], exponents.y); atomicMax(&metadata[offset + 4u], exponents.z);
}`,body:`var basin = ${a}u; var lower = 0xffffffffu; var upper = 0u; var exponents = vec3<u32>(0u);
for (var local = 0u; local < 256u; local++) {
 let entry = index * 256u + local; if (entry >= ${i}u) { break; }
 let cell = order[orderOffset + entry]; let nextBasin = basins[basinsOffset + cell]; let value = elevation[elevationOffset + cell];
 if (nextBasin >= ${a}u || !isFiniteValue(value)) { break; }
 if (nextBasin != basin) { publishMetadata(basin, lower, upper, exponents); basin = nextBasin; lower = 0xffffffffu; upper = 0u; exponents = vec3<u32>(0u); }
 let key = getElevationKey(value); lower = min(lower, key); upper = max(upper, key);
 exponents = max(exponents, vec3<u32>(getFixedPointExponentBits(value), ${s?`0u`:`getFixedPointExponentBits(getGroundCellArea(cell))`}, getFixedPointExponentBits(getStreamEdgeLength(cell))));
}
publishMetadata(basin, lower, upper, exponents);`}),p(e,{id:`${r}-sum-tiles`,operation:d,invocationCount:f*(o+1),bindings:[...u,b(`order`,n.order),b(`metadata`,m),C(`sums`,h),C(`histogram`,y)],declarations:`${l}\n${S}
${g(`sums`)}
${g(`histogram`)}
fn publishSums(basin: u32, lane: u32, elevationSum: vec3<u32>, areaSum: vec3<u32>, lengthSum: vec3<u32>, belowCount: u32) {
 if (basin >= ${a}u) { return; }
 if (lane == 0u) { addFixedPointToSums(basin * 3u, elevationSum); addFixedPointToSums(basin * 3u + 1u, areaSum); addFixedPointToSums(basin * 3u + 2u, lengthSum); }
 else { ${s?`atomicAdd(&histogram[histogramOffset + (basin * ${o}u + lane - 1u) * 3u], belowCount);`:`addFixedPointToHistogram(basin * ${o}u + lane - 1u, areaSum);`} }
}`,body:`let tile = index / ${o+1}u; let lane = index % ${o+1}u;
var basin = ${a}u; var elevationSum = vec3<u32>(0u); var areaSum = vec3<u32>(0u); var lengthSum = vec3<u32>(0u); var belowCount = 0u;
for (var local = 0u; local < 256u; local++) {
 let entry = tile * 256u + local; if (entry >= ${i}u) { break; }
 let cell = order[orderOffset + entry]; let nextBasin = basins[basinsOffset + cell]; let value = elevation[elevationOffset + cell];
 if (nextBasin >= ${a}u || !isFiniteValue(value)) { break; }
 if (nextBasin != basin) { publishSums(basin, lane, elevationSum, areaSum, lengthSum, belowCount); basin = nextBasin; elevationSum = vec3<u32>(0u); areaSum = vec3<u32>(0u); lengthSum = vec3<u32>(0u); belowCount = 0u; }
 let offset = metadataOffset + basin * 5u;
 let areaUnit = getFixedPointUnitExponent(metadata[offset + 3u], ${x}u);
 if (lane == 0u) {
  elevationSum = addFixedPoint(elevationSum, quantizeFixedPoint(value, getFixedPointUnitExponent(metadata[offset + 2u], ${x}u)));
  ${s?``:`areaSum = addFixedPoint(areaSum, quantizeFixedPoint(getGroundCellArea(cell), areaUnit));`}
  lengthSum = addFixedPoint(lengthSum, quantizeFixedPoint(getStreamEdgeLength(cell), getFixedPointUnitExponent(metadata[offset + 4u], ${x}u)));
 } else {
  let lower = getElevationFromKey(metadata[offset]); let upper = getElevationFromKey(metadata[offset + 1u]);
  let threshold = lower + (upper - lower) * f32(lane) / ${o}.0;
  if (value <= threshold || lane == ${o}u) { ${s?`belowCount++;`:`areaSum = addFixedPoint(areaSum, quantizeFixedPoint(getGroundCellArea(cell), areaUnit));`} }
 }
}
publishSums(basin, lane, elevationSum, areaSum, lengthSum, belowCount);`}),p(e,{id:`${r}-statistics`,operation:d,invocationCount:a,bindings:[b(`metadata`,m),b(`sums`,h),b(`ranges`,n.ranges),b(`statistics`,n.statistics,`read_write`)],declarations:`${T}\n${S}
fn loadSum(slot: u32, exponent: u32) -> f32 {
 let offset = sumsOffset + slot * 3u;
 return getFixedPointFloat(vec3<u32>(sums[offset], sums[offset + 1u], sums[offset + 2u]), getFixedPointUnitExponent(exponent, ${x}u));
}`,body:`let sampleCount = ranges[rangesOffset + index + 1u] - ranges[rangesOffset + index];
let offset = metadataOffset + index * 5u;
let sum = loadSum(index * 3u, metadata[offset + 2u]);
let streamLength = loadSum(index * 3u + 2u, metadata[offset + 4u]);
let area = ${s?`f32(sampleCount) * ${c}`:`loadSum(index * 3u + 1u, metadata[offset + 3u])`};
let destination = statisticsOffset + index * 6u;
statistics[destination] = area; statistics[destination + 1u] = select(getNaN(), sum / f32(max(sampleCount, 1u)), sampleCount > 0u);
statistics[destination + 2u] = select(getNaN(), getElevationFromKey(metadata[offset]), sampleCount > 0u);
statistics[destination + 3u] = select(getNaN(), getElevationFromKey(metadata[offset + 1u]), sampleCount > 0u);
statistics[destination + 4u] = streamLength; statistics[destination + 5u] = select(getNaN(), streamLength / max(area, 1e-30), area > 0.0);`}),p(e,{id:`${r}-hypsometry`,operation:d,invocationCount:a*o,bindings:[b(`metadata`,m),b(`histogram`,y),b(`hypsometry`,n.hypsometry,`read_write`)],declarations:v,body:`let offset = histogramOffset + index * 3u;
hypsometry[hypsometryOffset + index] = ${s?`f32(histogram[offset]) * ${c}`:`getFixedPointFloat(vec3<u32>(histogram[offset], histogram[offset + 1u], histogram[offset + 2u]), getFixedPointUnitExponent(metadata[metadataOffset + (index / ${o}u) * 5u + 3u], ${x}u))`};`})]}function G(e){return[e.statistics,e.hypsometry,e.profiles,e.profileStatus,...k(e.status)]}var K=class{props;id;constructor(e){this.props=e,this.id=e.id??`terrain-catchment-summary`;let t=a(this.id,e.grid);d(this.id,e.cellSizeMode),D(this.id,e.directionEncoding),i(this.id,e.grid,e.cellSizeMode),l(this.id,`basinCount`,e.basinCount),l(this.id,`histogramBins`,e.histogramBins),l(this.id,`maximumSteps`,e.maximumSteps??t),H(this.id,t,e.basinCount,e.histogramBins),c(this.id,Math.abs(e.grid.affine[0]*e.grid.affine[4]-e.grid.affine[1]*e.grid.affine[3])>0,`grid affine must have a nonzero determinant`),o(this.id,`elevation`,e.elevation,t),o(this.id,`basinIds`,e.basinIds,t),o(this.id,`streams`,e.streams,t),o(this.id,`flowDirections`,e.flowDirections,t),E(this.id,e.validity,t),o(this.id,`output.statistics`,e.output.statistics,e.basinCount*6),o(this.id,`output.hypsometry`,e.output.hypsometry,e.basinCount*e.histogramBins),o(this.id,`output.profiles`,e.output.profiles,t*3),o(this.id,`output.profileStatus`,e.output.profileStatus,t),x(this.id,{...e.output.status}),w(this.id,[e.elevation,e.basinIds,e.streams,e.flowDirections,...e.validity?[e.validity]:[]],G(e.output),[{writes:[e.output.statistics]},{writes:[e.output.hypsometry]},{reads:[...e.validity?[]:[e.elevation],e.basinIds,e.streams,e.flowDirections],writes:[e.output.profiles,e.output.profileStatus]},...k(e.output.status).map(t=>({reads:t===e.output.status?.invalidCount?[e.output.profileStatus]:[],writes:[t]}))])}getCommandNodes(e){let{id:i,props:a}=this,o=a.elevation.length;S(i,e,[a.elevation,...a.validity?[a.validity]:[],a.basinIds,a.streams,a.flowDirections,...G(a.output)]),U(e,i,o,a.basinCount,a.histogramBins);let s=y(e,{id:i,operation:`GPUTerrainCatchmentSummary`,values:a.elevation,validity:a.validity}),c=s.values,l=(a.cellSizeMode??`uniform`)===`uniform`,u=f(Math.abs(a.grid.affine[0]*a.grid.affine[4]-a.grid.affine[1]*a.grid.affine[3])),d=`${O(a.grid.width,a.grid.height,a.directionEncoding)}\n${T}
${C(a.grid,a.cellSizeMode)}
fn edgeLength(cell: u32, receiver: u32) -> f32 { return getGroundEdgeLength(cell, receiver); }
fn isStreamWord(word: u32) -> bool { return word != 0u && word < UNRESOLVED_DIRECTION; }
fn getCellReceiver(cell: u32) -> u32 { return getFlowReceiver(cell, getFlowDirectionCode(directions[directionsOffset + cell])); }
// Length of the stream edge leaving 'cell', 0 when the cell or its receiver is not a stream cell.
fn getStreamEdgeLength(cell: u32) -> f32 {
 let receiver = getCellReceiver(cell);
 if (isStreamWord(streams[streamsOffset + cell]) && receiver != NO_CELL && isStreamWord(streams[streamsOffset + receiver]) && isFiniteValue(elevation[elevationOffset + receiver])) { return edgeLength(cell, receiver); }
 return 0.0;
}`,g=[b(`elevation`,c),b(`basins`,a.basinIds),b(`streams`,a.streams),b(`directions`,a.flowDirections)],_=t(e,`${i}-keys`,`uint32`,o),v=t(e,`${i}-cells`,`uint32`,o),x=t(e,`${i}-sorted-keys`,`uint32`,o),w=t(e,`${i}-order`,`uint32`,o),E=t(e,`${i}-ranges`,`uint32`,a.basinCount+1),D=[0,1].map(n=>t(e,`${i}-profiles-${n}`,`uint32`,o*4)),k=a.output.status?.iterationLimitReached?t(e,`${i}-profile-limit-flags`,`uint32`,o):void 0,j=t(e,`${i}-profile-limit-count`,`uint32`,1),M=[...s.nodes,p(e,{id:`${i}-basin-keys`,operation:`GPUTerrainCatchmentSummary`,invocationCount:o,declarations:T,bindings:[b(`elevation`,c),b(`basins`,a.basinIds),b(`keys`,_,`read_write`),b(`cells`,v,`read_write`)],body:`let basin = basins[basinsOffset + index]; keys[keysOffset + index] = select(${a.basinCount}u, basin, basin < ${a.basinCount}u && isFiniteValue(elevation[elevationOffset + index])); cells[cellsOffset + index] = index;`}),...new n({id:`${i}-basin-sort`,keys:_,values:v,outputKeys:x,outputValues:w,keyBits:h(a.basinCount)}).getCommandNodes(e),m(e,{id:`${i}-ranges`,operation:`GPUTerrainCatchmentSummary`,segmentCount:a.basinCount,sortedKeys:x,segmentOffsets:E}),...W(e,{id:i,cellCount:o,basinCount:a.basinCount,histogramBins:a.histogramBins,isUniform:l,area:u,declarations:d,inputs:g,order:w,ranges:E,statistics:a.output.statistics,hypsometry:a.output.hypsometry}),p(e,{id:`${i}-profile-seeds`,operation:`GPUTerrainCatchmentSummary`,invocationCount:o,declarations:d,bindings:[...g,b(`state`,D[0],`read_write`)],body:`var parent = NO_CELL; var distance = 0.0; var steps = 0u; var result = 2u;
let code = getFlowDirectionCode(directions[directionsOffset + index]);
if (isFiniteValue(elevation[elevationOffset + index]) && isStreamWord(streams[streamsOffset + index])) {
 if (code == 0u) { result = 1u; } else {
  let receiver = getFlowReceiver(index, code);
  if (receiver != NO_CELL && isFiniteValue(elevation[elevationOffset + receiver]) && streams[streamsOffset + receiver] < UNRESOLVED_DIRECTION) {
   if (streams[streamsOffset + receiver] == 0u) { result = 1u; }
   else { parent = receiver; distance = edgeLength(index, receiver); steps = 1u; }
  }
 }
}
state[stateOffset + index * 4u] = parent; state[stateOffset + index * 4u + 1u] = bitcast<u32>(distance);
state[stateOffset + index * 4u + 2u] = steps; state[stateOffset + index * 4u + 3u] = result;`})],N=Math.ceil(Math.log2(Math.max(o,1)));for(let t=0;t<N;t++)M.push(p(e,{id:`${i}-profile-jump-${t}`,operation:`GPUTerrainCatchmentSummary`,invocationCount:o,bindings:[b(`previous`,D[t%2]),b(`next`,D[(t+1)%2],`read_write`)],body:`let offset = previousOffset + index * 4u; let destination = nextOffset + index * 4u;
for (var component = 0u; component < 4u; component++) { next[destination + component] = previous[offset + component]; }
let parent = previous[offset];
if (parent != 0xffffffffu) {
 let ancestor = previousOffset + parent * 4u;
 next[destination] = previous[ancestor];
 next[destination + 1u] = bitcast<u32>(bitcast<f32>(previous[offset + 1u]) + bitcast<f32>(previous[ancestor + 1u]));
 next[destination + 2u] = min(${o}u, previous[offset + 2u] + previous[ancestor + 2u]);
 next[destination + 3u] = previous[ancestor + 3u];
}`}));return M.push(p(e,{id:`${i}-profiles`,operation:`GPUTerrainCatchmentSummary`,invocationCount:o,declarations:d,bindings:[...g,b(`state`,D[N%2]),b(`profiles`,a.output.profiles,`read_write`),b(`status`,a.output.profileStatus,`read_write`),...k?[b(`limitFlags`,k,`read_write`)]:[]],body:`let center = elevation[elevationOffset + index]; let offset = stateOffset + index * 4u;
let distance = bitcast<f32>(state[offset + 1u]); var gradient = 0.0;
var result = select(2u, 1u, state[offset] == NO_CELL && state[offset + 3u] == 1u && state[offset + 2u] < ${a.maximumSteps??o}u && isFiniteValue(distance));
if (!isFiniteValue(center) || basins[basinsOffset + index] >= ${a.basinCount}u || !isStreamWord(streams[streamsOffset + index])) { result = select(0u, 2u, streams[streamsOffset + index] >= UNRESOLVED_DIRECTION); }
let receiver = getCellReceiver(index);
if (receiver != NO_CELL && isStreamWord(streams[streamsOffset + receiver])) { gradient = (center - elevation[elevationOffset + receiver]) / edgeLength(index, receiver); }
status[statusOffset + index] = result;
${k?`let eligible = isFiniteValue(center) && basins[basinsOffset + index] < ${a.basinCount}u && isStreamWord(streams[streamsOffset + index]);
let exhausted = state[offset] != NO_CELL || (state[offset + 3u] == 1u && state[offset + 2u] >= ${a.maximumSteps??o}u);
limitFlags[limitFlagsOffset + index] = u32(result == 2u && eligible && exhausted);`:``}
profiles[profilesOffset + index * 3u] = select(getNaN(), distance, result == 1u);
profiles[profilesOffset + index * 3u + 1u] = center;
profiles[profilesOffset + index * 3u + 2u] = select(getNaN(), gradient, result == 1u);`})),k&&M.push(...r(e,{id:`${i}-profile-limit-count`,operation:`GPUTerrainCatchmentSummary`,values:k,count:o,condition:`value != 0u`,output:j})),M.push(...A(e,{id:i,operation:`GPUTerrainCatchmentSummary`,states:a.output.profileStatus,status:a.output.status,iterationLimit:{source:j,expression:`u32(source[sourceOffset] != 0u)`}})),M}},q=8,J=[.05,.06,.95,.5],Y=[[.95,.55,.2],[.3,.7,.95],[.55,.85,.4],[.9,.4,.65],[.75,.65,.95],[.95,.85,.3],[.3,.85,.75],[.85,.45,.4]],X=[.84,.28],Z=[[0,0,0,0],[1,1,1,.9]];function Q(e){let t=[...e].sort((e,t)=>e-t);return t[Math.floor(t.length/2)]??0}var $=M({parameters:P,caption:`Hypsometric histogram of every basin: how much of its area lies at each elevation.`,createInputs:e=>{let t=L(e),n=new Uint32Array(q);for(let e=0;e<q;e++)n[e]=z(t,.05+.125*e);return{...t,world:e.world,pourPoints:n,focus:{basin:-1,key:``}}},build(t,n,r){let i=R(t,n);i.writeStreamThreshold(25);let a=Math.round(r.histogramBins),o=t.allocate(`pour-points`,`uint32`,q),s=t.allocate(`basins`,`uint32`,n.cellCount);t.graph.add(new j({grid:n.grid,flowDirections:i.flowDirections.view,pourPoints:o.view,output:{labels:s.view}}));let c=t.allocate(`hypsometry`,`float32`,q*a);t.graph.add(new K({grid:n.grid,elevation:i.filledElevation.view,basinIds:s.view,basinCount:q,streams:i.streams.view,flowDirections:i.flowDirections.view,histogramBins:a,output:{statistics:t.allocate(`statistics`,`float32`,48).view,hypsometry:c.view,profiles:t.allocate(`profiles`,`float32`,n.cellCount*3).view,profileStatus:t.allocate(`profile-status`,`uint32`,n.cellCount).view}}));let l=t.allocate(`focus-mask`,`uint32`,n.cellCount),u=t.upload(`baseline`,`float32x4`,Float32Array.of(J[0]*n.world.width,J[1]*n.world.height,J[2]*n.world.width,J[1]*n.world.height));return{outputs:{filledElevation:i.filledElevation,basins:s,hypsometry:c,focusMask:l,baseline:u},update:t=>{let r=new Uint32Array(q).fill(e);r.set(n.pourPoints.subarray(0,Math.round(t.pourPointCount))),o.write(r)}}},rebuildOn:[`histogramBins`],async afterRun({outputs:e,values:t,upload:n}){let r=await e.hypsometry.buffer.readAsync(),i=new Float32Array(r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength)),a=Math.round(t.histogramBins),o=Math.round(t.pourPointCount),s=1;for(let e=0;e<o*a;e++)Number.isFinite(i[e])&&(s=Math.max(s,i[e]));return{groups:Array.from({length:o},(e,t)=>{let r=new Float32Array(o*a);for(let e=0;e<a;e++){let n=i[t*a+e];r[t*a+e]=Number.isFinite(n)?n:0}return n(`basin-${t}-bars`,`float32`,r)}),maximum:s}},interaction:{kind:`probe`,hint:`hover a basin`,rest:X,resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=a(n.basins);if(!o)return{};let s=Math.round(r.pourPointCount),c=N(J,i,e.x,e.y),l=Math.hypot(e.unitX-X[0],e.unitY-X[1])<.003,u=-1;if(l)u=-1;else if(c.isInside)u=Math.min(s-1,Math.floor(c.u*s));else{let n=o[I(t,i,e.x,e.y).index];n<s&&(u=n)}let d=`${u}:${s}:${o.length}`;if(d!==t.focus.key){t.focus={basin:u,key:d};let e=new Uint32Array(t.cellCount);for(let n=0;n<t.cellCount;n++)u>=0&&o[n]===u&&(e[n]=1);n.focusMask.write(e)}if(u<0)return{focus:-1,spotlight:null,readout:`${s} basins`};let f=[],p=0,m=[0,0];for(let e=0;e<t.cellCount;e++){if(o[e]!==u)continue;p++,f.push(t.elevation[e]);let n=B(t,t.world,e);m=[m[0]+n[0],m[1]+n[1]]}return{focus:u,spotlight:null,readout:`basin ${u+1} · ${p.toLocaleString(`en-US`)} cells · median elevation ${Q(f).toFixed(2)}`}}},draw:({inputs:e,outputs:t,extra:n,interaction:r})=>[F(e,t.filledElevation),{layer:`raster`,values:t.basins,columns:e.columns,rows:e.rows,categories:Y,alpha:r.focus===void 0||r.focus===-1?.8:.5},{layer:`raster`,values:t.focusMask,columns:e.columns,rows:e.rows,categories:Z,alpha:.55},{layer:`lines`,context:!0,segments:t.baseline,radius:1,color:[1,1,1,.5]},...n.groups.map((e,t)=>({layer:`chart`,kind:`bars`,values:e,range:[0,n.maximum*1.05],color:[...Y[t],r.focus===void 0||r.focus===-1||r.focus===t?1:.35],rectangle:J}))]});export{$ as default};
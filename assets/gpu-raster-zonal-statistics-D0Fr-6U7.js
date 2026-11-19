import{t as e}from"./common-CshkX-d_.js";import{r as t,s as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-group-aggregation-CEYb6GJw.js";import{o as i}from"./sorted-segment-sums-Dvmm6OdR.js";import{o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as s,S as c,T as l,f as u,s as d,u as f,x as p}from"./gpu-contributor-utils-BIwLQON5.js";import{a as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as h,r as g}from"./raster-fractional-tile-kernel-D2BjD3g1.js";import{t as _}from"./binary-search-wgsl-CMmaMYIt.js";import{c as v,h as y,n as b,r as x,u as S}from"./raster-band-utils-BcTxc8_Y.js";import{u as C}from"./wgsl-common-DfyXiYB0.js";import{t as w}from"./earth-DODowzP2.js";import{r as T}from"./spherical-cell-area-3w3VAb_m.js";function E(e,n){let{id:r,dictionary:i,invalidCount:s}=n,c=`GPURasterZonalStatistics`,l=i.length,u=t(e,`${r}-dictionary-invalid`,`uint32`,1),d=t(e,`${r}-dense-zones`,`uint32`,n.zones.length);return{zones:d,nodes:[o(e,{id:`${r}-dictionary-reset`,operation:c,view:u,type:`u32`,value:`0u`}),a(e,{id:`${r}-dictionary-validate`,operation:c,invocationCount:l,bindings:[{name:`dictionary`,view:i,type:`u32`,access:`read`},{name:`invalidDictionary`,view:u,type:`atomic<u32>`,access:`read_write`}],body:`if (index > 0u && dictionary[dictionaryOffset + index] <= dictionary[dictionaryOffset + index - 1u]) { atomicAdd(&invalidDictionary[invalidDictionaryOffset], 1u); }`}),a(e,{id:`${r}-dictionary-publish`,operation:c,invocationCount:l,bindings:[{name:`dictionary`,view:i,type:`u32`,access:`read`},{name:`invalidDictionary`,view:u,type:`u32`,access:`read`},{name:`sourceZoneIds`,view:n.sourceZoneIds,type:`u32`,access:`read_write`},{name:`denseZoneRows`,view:n.denseZoneRows,type:`u32`,access:`read_write`},{name:`invalidCount`,view:s,type:`u32`,access:`read_write`}],body:`sourceZoneIds[sourceZoneIdsOffset + index] = dictionary[dictionaryOffset + index];
  denseZoneRows[denseZoneRowsOffset + index] = select(index, 0xffffffffu, invalidDictionary[invalidDictionaryOffset] != 0u);
  if (index == 0u) { invalidCount[invalidCountOffset] = invalidDictionary[invalidDictionaryOffset]; }`}),a(e,{id:`${r}-dictionary-map`,operation:c,invocationCount:n.zones.length,bindings:[{name:`sourceZones`,view:n.zones,type:`u32`,access:`read`},{name:`dictionary`,view:i,type:`u32`,access:`read`},{name:`invalidDictionary`,view:u,type:`u32`,access:`read`},{name:`invalidCount`,view:s,type:`atomic<u32>`,access:`read_write`},{name:`denseZones`,view:d,type:`u32`,access:`read_write`}],declarations:_(`dictionary`),body:`let sourceZone = sourceZones[sourceZonesOffset + index];
  var denseZone = ${l}u;
  if (sourceZone != ${n.ignoredZone}u && invalidDictionary[invalidDictionaryOffset] == 0u) {
    let candidate = lowerBoundDictionary(0u, ${l}u, sourceZone);
    if (candidate < ${l}u && dictionary[dictionaryOffset + candidate] == sourceZone) { denseZone = candidate; }
    else { atomicAdd(&invalidCount[invalidCountOffset], 1u); }
  }
  denseZones[denseZonesOffset + index] = denseZone;`})]}}function D(e){return f(e,4)}function O(e){return Math.min(16777215,f(e,4,1))}var k=4294967295,A=4294967295,j=[[`cellCounts`,`count`],[`valueCounts`,`count`],[`sums`,`sum`],[`means`,`mean`],[`minimums`,`min`],[`maximums`,`max`]],M=[`weightSums`,`variances`,`standardDeviations`];function N(e,t,n){return`const ZONAL_GRID_WIDTH: u32 = ${e}u;
const ZONAL_GRID_HEIGHT: f32 = ${t}.0;
const ZONAL_PARAMETER_LAYOUT_VERSION: u32 = ${n}u;
var<private> extent: array<f32, 4>;
const extentOffset: u32 = 0u;
${T(w)}
fn getCellWeight(cell: u32) -> f32 {
  if (ZONAL_PARAMETER_LAYOUT_VERSION == 2u) {
    let row = f32(cell / ZONAL_GRID_WIDTH);
    let rowStep = settings[settingsOffset + 4u];
    let middlePoleDistance = settings[settingsOffset + 5u] - settings[settingsOffset + 6u] * (row + 0.5) * rowStep;
    let area = abs(2.0 * SPHERE_RADIUS * SPHERE_RADIUS * settings[settingsOffset] * DEGREES_TO_RADIANS *
      sin(middlePoleDistance * DEGREES_TO_RADIANS) * sin(0.5 * rowStep * DEGREES_TO_RADIANS));
    return select(0.0, area, isFiniteValue(area) && area > 0.0);
  }
  let northEdge = settings[settingsOffset + 2u];
  // Signed row height: negative when rows advance south, which makes getCellArea negative.
  let rowHeight = (settings[settingsOffset + 3u] - northEdge) / ZONAL_GRID_HEIGHT;
  extent = array<f32, 4>(0.0, northEdge, settings[settingsOffset], rowHeight);
  let area = abs(getCellArea(cell / ZONAL_GRID_WIDTH));
  return select(0.0, area, isFiniteValue(area) && area > 0.0);
}`}function P(e,t){let{output:n}=t,r=t.cellSizeMode===`geographic`||!!(n?.variances||n?.standardDeviations||n?.weightSums);return e===`sorted`&&(r||!!(n?.sums||n?.means))}var F=class e{id;props;sumOrder;constructor(e){this.id=e.id??`raster-zonal-statistics`,this.props=e;let{id:t}=this;if(this.sumOrder=e.sumOrder??`sorted`,this.sumOrder!==`atomic`&&this.sumOrder!==`sorted`)throw Error(`${t} sumOrder must be atomic or sorted`);let n=S(t,e.grid);if(s(e.zones,[`uint32`],`${t} zones`),e.zones.length!==n)throw Error(`${t} zones must contain one value per cell`);if(!Number.isSafeInteger(e.zoneCapacity)||e.zoneCapacity<1||e.zoneCapacity>4294967295)throw Error(`${t} zoneCapacity must be a positive integer that fits in uint32`);let r=e.ignoredZone??4294967295;if(!Number.isInteger(r)||r<0||r>4294967295)throw Error(`${t} ignoredZone must be a uint32 integer`);let i=e.cellSizeMode??`uniform`;if(l(t,e.parameterLayoutVersion===void 0||e.parameterLayoutVersion===1||e.parameterLayoutVersion===2,`parameterLayoutVersion must be 1 or 2`),i!==`uniform`&&i!==`geographic`)throw Error(`${t} cellSizeMode must be uniform or geographic`);if(i===`geographic`){if(l(t,e.grid.affine[1]===0&&e.grid.affine[3]===0,`geographic cell areas need an axis-aligned grid`),!e.parameters)throw Error(`${t} geographic cells need parameters`);y(t,e.parameters,e.parameterLayoutVersion===2?7:4)}else if(e.parameters)throw Error(`${t} parameters need cellSizeMode geographic`);let{output:a}=e;if(l(t,!!e.zoneDictionary==!!(a.sourceZoneIds||a.denseZoneRows||a.status?.invalidCount),`zoneDictionary must be present exactly when dictionary outputs are requested`),e.zoneDictionary){s(e.zoneDictionary,[`uint32`],`${t} zoneDictionary`),l(t,e.zoneDictionary.length===e.zoneCapacity,`zoneDictionary must have zoneCapacity entries`);for(let[n,r]of[[`sourceZoneIds`,a.sourceZoneIds],[`denseZoneRows`,a.denseZoneRows]])l(t,!!r,`output.${n} is required with zoneDictionary`),s(r,[`uint32`],`${t} output.${n}`),l(t,r.length===e.zoneCapacity,`output.${n} must have zoneCapacity entries`);l(t,!!a.status?.invalidCount,`output.status.invalidCount is required with zoneDictionary`),s(a.status.invalidCount,[`uint32`],`${t} output.status.invalidCount`),l(t,a.status.invalidCount.length===1,`output.status.invalidCount must have one entry`),c(t,`zones.length + zoneCapacity`,n+e.zoneCapacity,A)}if([...j.filter(([t])=>e.output[t]),...M.filter(t=>e.output[t])].length===0)throw Error(`${t} requires at least one output`);for(let[n,r]of j){let i=e.output[n];if(i&&(r===`count`?s(i,[`uint32`],`${t} output.${n}`):s(i,[`float32`],`${t} output.${n}`),i.length!==e.zoneCapacity))throw Error(`${t} output.${n} must contain zoneCapacity rows`)}for(let n of M){let r=e.output[n];if(r&&(s(r,[`float32`],`${t} output.${n}`),r.length!==e.zoneCapacity))throw Error(`${t} output.${n} must contain zoneCapacity rows`)}let o=a.status?.overflow;if(o&&(s(o,[`uint32`],`${t} output.status.overflow`),o.length!==1))throw Error(`${t} output.status.overflow must contain one uint32 row`);let u=e.cellSizeMode===`geographic`,d=!!(a.variances||a.standardDeviations),f=u||d||!!a.weightSums,p=P(this.sumOrder,e),m=!!a.valueCounts||p&&!!a.means||f&&(!!a.means||d),h=[a.cellCounts,m,a.minimums,a.maximums].filter(Boolean).length,_=e.zones.buffer.graph,v=h>0&&h*4*e.zoneCapacity<=_.device.limits.maxComputeWorkgroupStorageSize,x=e.values,C=x.storage.kind===`buffer`&&x.format===`float32`&&x.validity&&x.scale===void 0&&x.offset===void 0&&x.noDataValue===void 0?b(x):[],w=e.zoneDictionary?void 0:e.zones;g(t,a,[e.zones,e.zoneDictionary,e.parameters,...b(x)],[...v?[{writes:[a.cellCounts,a.valueCounts,a.minimums,a.maximums,o],reads:[w,...C]}]:[],...j.flatMap(([e])=>v&&e!==`sums`&&e!==`means`||(p||f)&&(e===`sums`||e===`means`)?[]:[{writes:[a[e]],reads:[w,...e===`cellCounts`?[]:C]}]),...p||f?[{writes:[a.sums],reads:p?[]:[w]},{writes:[a.weightSums],reads:p?[]:[w]},{writes:[a.means],reads:[a.sums,a.valueCounts,u?a.weightSums:void 0]},{writes:[a.variances,a.standardDeviations],reads:[a.valueCounts,u?a.weightSums:void 0]}]:[],{writes:[a.sourceZoneIds,a.denseZoneRows,a.status?.invalidCount],reads:[e.zoneDictionary]},{writes:[a.status?.invalidCount],reads:[e.zones,e.zoneDictionary]},...v?[]:[{writes:[o],reads:[w,e.parameters,...C]}]])}getCommandNodes(n){let{id:i,props:s}=this;if(this.validateBindings(n),s.zoneDictionary){let t=E(n,{id:i,zones:s.zones,dictionary:s.zoneDictionary,ignoredZone:s.ignoredZone??4294967295,sourceZoneIds:s.output.sourceZoneIds,denseZoneRows:s.output.denseZoneRows,invalidCount:s.output.status.invalidCount}),{sourceZoneIds:r,denseZoneRows:a,status:o,...c}=s.output;return[...t.nodes,...new e({...s,id:`${i}-dense`,zones:t.zones,zoneDictionary:void 0,ignoredZone:s.zoneCapacity,output:{...c,status:o?.overflow?{overflow:o.overflow}:void 0}}).getCommandNodes(n)]}let{zones:c,zoneCapacity:l,output:d}=s,f=d.status?.overflow,{width:m,height:h}=s.grid,g=s.ignoredZone??4294967295;v(i,n,s.values,[]),p(i,n,[c,f,s.parameters,...j.map(([e])=>d[e]),...M.map(e=>d[e])]),u(i,`zoneCapacity`,l,1,P(this.sumOrder,s)?O(n.device):D(n.device));let _=m*h,y=s.cellSizeMode===`geographic`,b=!!(d.variances||d.standardDeviations),S=y||b||!!d.weightSums,w=!!d.means||b,T=!!d.weightSums||y&&w,k=P(this.sumOrder,s),A=!!d.valueCounts||k&&!!d.means||S&&w,F=S||!!(d.valueCounts||d.sums||d.means||d.minimums||d.maximums),I=[d.cellCounts,A,d.minimums,d.maximums].filter(Boolean).length,R=I>0&&I*4*l<=n.device.limits.maxComputeWorkgroupStorageSize,z=!R&&!!d.cellCounts,ee=R?!k&&(S||!!(d.sums||d.means)):S||j.some(([e])=>e!==`cellCounts`&&d[e]),B=[],V,H;if(F){let e=x(n,`${i}-values`,s.values,s.grid,`reuse-canonical`);B.push(...e.nodes),V=e.band.storage.values,H=e.band.validity}f&&B.push(o(n,{id:`${i}-overflow-reset`,operation:`GPURasterZonalStatistics`,view:f,type:`u32`,value:`0u`}));let U=z?t(n,`${i}-zone-mask`,`uint32`,_):void 0,W=ee?t(n,`${i}-value-mask`,`uint32`,_):void 0,G=k?t(n,`${i}-sort-rows`,`uint32`,_):void 0,K=k||S?t(n,`${i}-contributions`,`float32`,_):void 0,q=S&&T?t(n,`${i}-weight-contributions`,`float32`,_):void 0,J=R?void 0:f,Y=(e,t)=>{let{needsValues:r}=t,i=[{name:`zones`,view:c,type:`u32`,access:`read`}];r&&i.push({name:`values`,view:V,type:`f32`,access:`read`},{name:`validity`,view:H,type:`u32`,access:`read`}),t.valueMask&&i.push({name:`valueMask`,view:t.valueMask,type:`u32`,access:`read_write`}),t.zoneMask&&i.push({name:`zoneMask`,view:t.zoneMask,type:`u32`,access:`read_write`}),t.sortRows&&i.push({name:`sortRows`,view:t.sortRows,type:`u32`,access:`read_write`}),t.contributions&&i.push({name:`contributions`,view:t.contributions,type:`f32`,access:`read_write`}),t.weightContributions&&i.push({name:`weightContributions`,view:t.weightContributions,type:`f32`,access:`read_write`}),y&&t.contributions&&i.push({name:`settings`,view:s.parameters,type:`f32`,access:`read`}),t.maskOverflow&&i.push({name:`overflowFlag`,view:t.maskOverflow,type:`atomic<u32>`,access:`read_write`}),B.push(a(n,{id:e,operation:`GPURasterZonalStatistics`,variant:`mask`,bindings:i,invocationCount:_,declarations:`const CAPACITY: u32 = ${l}u;
const IGNORED: u32 = ${g}u;
${C}
${y&&t.contributions?N(m,h,s.parameterLayoutVersion??1):``}`,body:`let zone = zones[zonesOffset + index];
  let inZone = zone < CAPACITY && zone != IGNORED;
  ${t.zoneMask?`zoneMask[zoneMaskOffset + index] = select(0u, 1u, inZone);`:``}
  ${r?`let value = values[valuesOffset + index];
  let isValid = inZone && validity[validityOffset + index] != 0u && isFiniteValue(value);`:``}
  ${t.valueMask?`valueMask[valueMaskOffset + index] = select(0u, 1u, isValid);`:``}
  ${t.sortRows?`sortRows[sortRowsOffset + index] = select(CAPACITY, zone, inZone);`:``}
  ${t.contributions?y?`let weight = getCellWeight(index);
  contributions[contributionsOffset + index] = select(0.0, weight * value, isValid);`:`contributions[contributionsOffset + index] = select(0.0, value, isValid);`:``}
  ${t.weightContributions?`weightContributions[weightContributionsOffset + index] = select(0.0, ${y?`weight`:`1.0`}, isValid);`:``}
  ${t.maskOverflow?`if (zone >= CAPACITY && zone != IGNORED) {
    atomicStore(&overflowFlag[overflowFlagOffset], 1u);
  }`:``}`}))},X={zoneMask:U,valueMask:W,sortRows:G,contributions:K,weightContributions:q,maskOverflow:J,needsValues:!!(W||G||K)};1+(X.needsValues?2:0)+[U,W,G,K,q,J].filter(Boolean).length+(y&&K?1:0)>8?(Y(`${i}-mask-zones`,{zoneMask:U,sortRows:G,maskOverflow:J,needsValues:!1}),Y(`${i}-mask`,{valueMask:W,contributions:K,weightContributions:q,needsValues:!0})):(U||W||G||K||J)&&Y(`${i}-mask`,X);let Z=d.cellCounts,Q=d.valueCounts??(A?t(n,`${i}-value-counts`,`uint32`,l):void 0),$={cellCounts:Z,valueCounts:Q};if(R)B.push(...L(n,{id:i,zones:c,values:V,validity:H,zoneCapacity:l,ignoredZone:g,cellCount:_,cellCounts:Z,valueCounts:Q,minimums:d.minimums,maximums:d.maximums,overflow:f}));else if(k||S)for(let e of[`cellCounts`,`valueCounts`])!d[e]&&$[e]&&B.push(...new r({id:`${i}-${e}`,keys:c,mask:e===`cellCounts`?U:W,output:$[e],operation:`count`}).getCommandNodes(n));for(let[e,t]of j){let a=d[e];if(!a||(k||S)&&(e===`sums`||e===`means`)||R&&e!==`sums`&&e!==`means`)continue;let o={id:`${i}-${e}`,keys:c,mask:e===`cellCounts`?U:W},s=t===`count`?new r({...o,output:a,operation:t}):new r({...o,values:V,output:a,operation:t});B.push(...s.getCommandNodes(n))}return(k||S)&&B.push(...this.getMomentNodes(n,{weighted:y,needsMeans:w,needsVariance:b,needsWeightSums:T,sortedSums:k,canonicalValues:V,canonicalValidity:H,valueMask:W,sortRows:G,contributions:K,weightContributions:q,valueCounts:Q})),B}validateBindings(e){let{id:t,props:r}=this,i=P(this.sumOrder,r);u(t,`zoneCapacity`,r.zoneCapacity,1,i?O(e.device):D(e.device));let a=[r.zones,r.zoneDictionary,r.parameters,...b(r.values),...d(r.output)];m(e,t,Object.fromEntries(a.map((e,t)=>[`view.${t}`,e]))),v(t,e,r.values,[]);for(let r of a)r&&h(e.device,t,r.buffer.id,n(r).size,`reduce the packed raster or zone capacity`,r.buffer.byteLength);for(let[n,a]of[[`per-cell planes and sort scratch`,r.grid.width*r.grid.height*4],[`zone scratch`,r.zoneCapacity*4],[`sorted segment offsets`,i?(r.zoneCapacity+1)*4:0],[`dictionary validation`,r.zoneDictionary?4:0]])h(e.device,t,n,a,`reduce grid dimensions or zoneCapacity`)}getMomentNodes(e,n){let{id:r,props:i}=this,{zones:o,zoneCapacity:s,output:c}=i,{width:l,height:u}=i.grid,d=i.ignoredZone??4294967295,{weighted:f,sortedSums:p}=n,m=[],h=c.sums??(n.needsMeans?t(e,`${r}-sorted-sums`,`float32`,s):void 0),g=c.weightSums??(n.needsWeightSums?t(e,`${r}-weight-sums`,`float32`,s):void 0),_=[...h?[{name:`sums`,contributions:n.contributions,output:h}]:[],...g?[{name:`weights`,contributions:n.weightContributions,output:g}]:[]];m.push(...this.getSumNodes(e,`${r}-sorted`,`${r}-`,p,n,_));let v=c.means??(n.needsVariance?t(e,`${r}-mean-values`,`float32`,s):void 0);if(v){let t=[{name:`sums`,view:h,type:`f32`,access:`read`},{name:`valueCounts`,view:n.valueCounts,type:`u32`,access:`read`}];f&&t.push({name:`weightSums`,view:g,type:`f32`,access:`read`}),t.push({name:`means`,view:v,type:`u32`,access:`read_write`}),m.push(a(e,{id:`${r}-means`,operation:`GPURasterZonalStatistics`,variant:f?`weighted-mean`:`sorted-mean`,bindings:t,invocationCount:s,body:f?`let count = valueCounts[valueCountsOffset + index];
  let weightSum = weightSums[weightSumsOffset + index];
  var mean = 0x7fc00000u;
  // A zone whose valid cells have no area has no weighted mean.
  if (count > 0u && weightSum > 0.0) {
    mean = bitcast<u32>(sums[sumsOffset + index] / weightSum);
  }
  means[meansOffset + index] = mean;`:`let count = valueCounts[valueCountsOffset + index];
  var mean = 0x7fc00000u;
  if (count > 0u) {
    mean = bitcast<u32>(sums[sumsOffset + index] / f32(count));
  }
  means[meansOffset + index] = mean;`}))}if(!n.needsVariance)return m;let y=l*u,b=t(e,`${r}-deviations`,`float32`,y),x=f?t(e,`${r}-weight-squares`,`float32`,y):void 0,S=[{name:`zones`,view:o,type:`u32`,access:`read`},{name:`values`,view:n.canonicalValues,type:`f32`,access:`read`},{name:`validity`,view:n.canonicalValidity,type:`u32`,access:`read`},{name:`means`,view:v,type:`f32`,access:`read`},{name:`deviations`,view:b,type:`f32`,access:`read_write`}];x&&S.push({name:`weightSquares`,view:x,type:`f32`,access:`read_write`},{name:`settings`,view:i.parameters,type:`f32`,access:`read`}),m.push(a(e,{id:`${r}-deviations`,operation:`GPURasterZonalStatistics`,variant:f?`weighted-deviations`:`deviations`,bindings:S,invocationCount:y,declarations:`const CAPACITY: u32 = ${s}u;
const IGNORED: u32 = ${d}u;
${C}
${f?N(l,u,i.parameterLayoutVersion??1):``}`,body:`let zone = zones[zonesOffset + index];
  let value = values[valuesOffset + index];
  let isValid = zone < CAPACITY && zone != IGNORED && validity[validityOffset + index] != 0u &&
    isFiniteValue(value);
  let weight = ${f?`getCellWeight(index)`:`1.0`};
  var deviation = 0.0;
  var weightSquare = 0.0;
  if (isValid && weight > 0.0) {
    let difference = value - means[meansOffset + zone];
    deviation = weight * difference * difference;
    weightSquare = weight * weight;
  }
  deviations[deviationsOffset + index] = deviation;
  ${x?`weightSquares[weightSquaresOffset + index] = weightSquare;`:``}`}));let w=t(e,`${r}-squared-deviation-sums`,`float32`,s),T=x?t(e,`${r}-weight-square-sums`,`float32`,s):void 0;m.push(...this.getSumNodes(e,`${r}-variance-sorted`,`${r}-variance-`,p,n,[{name:`deviations`,contributions:b,output:w},...x&&T?[{name:`weight-squares`,contributions:x,output:T}]:[]]));let E=[{name:`valueCounts`,view:n.valueCounts,type:`u32`,access:`read`},{name:`squaredDeviationSums`,view:w,type:`f32`,access:`read`}];T&&E.push({name:`weightSums`,view:g,type:`f32`,access:`read`},{name:`weightSquareSums`,view:T,type:`f32`,access:`read`});for(let e of[`variances`,`standardDeviations`])c[e]&&E.push({name:e,view:c[e],type:`u32`,access:`read_write`});return m.push(a(e,{id:`${r}-variances`,operation:`GPURasterZonalStatistics`,variant:f?`weighted-variances`:`variances`,bindings:E,invocationCount:s,body:`let valueCount = valueCounts[valueCountsOffset + index];
  var variance = 0x7fc00000u;
  var deviation = 0x7fc00000u;
  var denominator = 0.0;
  ${T?`let weightSum = weightSums[weightSumsOffset + index];
  if (weightSum != 0.0) {
    denominator = weightSum - weightSquareSums[weightSquareSumsOffset + index] / weightSum;
  }`:`denominator = f32(valueCount) - 1.0;`}
  if (valueCount >= 2u && denominator > 0.0) {
    let result = squaredDeviationSums[squaredDeviationSumsOffset + index] / denominator;
    variance = bitcast<u32>(result);
    deviation = bitcast<u32>(sqrt(result));
  }
  ${c.variances?`variances[variancesOffset + index] = variance;`:``}
  ${c.standardDeviations?`standardDeviations[standardDeviationsOffset + index] = deviation;`:``}`})),m}getSumNodes(e,t,n,a,o,s){return s.length===0?[]:a?i(e,{id:t,operation:`GPURasterZonalStatistics`,segmentCount:this.props.zoneCapacity,segmentKeys:o.sortRows,reductions:s}):s.flatMap(t=>new r({id:`${n}${t.name}`,keys:this.props.zones,mask:o.valueMask,values:t.contributions,output:t.output,operation:`sum`}).getCommandNodes(e))}},I=256;function L(t,n){let{id:r,zoneCapacity:i,cellCount:o}=n,s=[{name:`cellCounts`,view:n.cellCounts,initial:`0u`,merge:`atomicAdd`},{name:`valueCounts`,view:n.valueCounts,initial:`0u`,merge:`atomicAdd`},{name:`minimums`,view:n.minimums,initial:`0xffffffffu`,merge:`atomicMin`},{name:`maximums`,view:n.maximums,initial:`0u`,merge:`atomicMax`}].filter(e=>e.view),c=!!(n.valueCounts||n.minimums||n.maximums),l=e(Math.ceil(i/32),8,64),u=[];u.push(a(t,{id:`${r}-zone-reduce-init`,operation:`GPURasterZonalStatistics`,variant:`zone-reduce-init`,bindings:s.map(e=>({name:e.name,view:e.view,type:`u32`,access:`read_write`})),invocationCount:i,body:s.map(e=>`${e.name}[${e.name}Offset + index] = ${e.initial};`).join(`
  `)}));let d=[{name:`zones`,view:n.zones,type:`u32`,access:`read`}];c&&d.push({name:`values`,view:n.values,type:`f32`,access:`read`},{name:`validity`,view:n.validity,type:`u32`,access:`read`});for(let e of s)d.push({name:e.name,view:e.view,type:`atomic<u32>`,access:`read_write`});n.overflow&&d.push({name:`overflowFlag`,view:n.overflow,type:`atomic<u32>`,access:`read_write`});let f=s.map(e=>`var<workgroup> local_${e.name}: array<atomic<u32>, ${i}>;`).join(`
`),p=!!n.minimums,m=!!n.maximums;u.push(a(t,{id:`${r}-zone-reduce`,operation:`GPURasterZonalStatistics`,variant:`zone-reduce`,bindings:d,invocationCount:Math.ceil(o/l),guardIndex:!1,workgroupSize:I,declarations:`const CAPACITY: u32 = ${i}u;
const IGNORED: u32 = ${n.ignoredZone}u;
const CELL_COUNT: u32 = ${o}u;
const CELLS_PER_THREAD: u32 = ${l}u;
${f}
${C}
fn encodeOrderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`,body:`for (var slot = localInvocationIndex; slot < CAPACITY; slot += ${I}u) {
    ${s.map(e=>`atomicStore(&local_${e.name}[slot], ${e.initial});`).join(`
    `)}
  }
  workgroupBarrier();
  let tileStart = (index - localInvocationIndex) * CELLS_PER_THREAD;
  for (var step = 0u; step < CELLS_PER_THREAD; step++) {
    let cell = tileStart + step * ${I}u + localInvocationIndex;
    if (cell >= CELL_COUNT) {
      break;
    }
    let zone = zones[zonesOffset + cell];
    if (zone >= CAPACITY || zone == IGNORED) {
      ${n.overflow?`if (zone != IGNORED) {
        atomicStore(&overflowFlag[overflowFlagOffset], 1u);
      }`:``}
      continue;
    }
    ${n.cellCounts?`atomicAdd(&local_cellCounts[zone], 1u);`:``}
    ${c?`let value = values[valuesOffset + cell];
    if (validity[validityOffset + cell] != 0u && isFiniteValue(value)) {
      ${n.valueCounts?`atomicAdd(&local_valueCounts[zone], 1u);`:``}
      ${p||m?`let key = encodeOrderedFloat(value);`:``}
      ${p?`atomicMin(&local_minimums[zone], key);`:``}
      ${m?`atomicMax(&local_maximums[zone], key);`:``}
    }`:``}
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < CAPACITY; slot += ${I}u) {
    ${s.map(e=>`let partial_${e.name} = atomicLoad(&local_${e.name}[slot]);
    if (partial_${e.name} != ${e.initial}) {
      ${e.merge}(&${e.name}[${e.name}Offset + slot], partial_${e.name});
    }`).join(`
    `)}
  }`}));let h=s.filter(e=>e.name===`minimums`||e.name===`maximums`);return h.length>0&&u.push(a(t,{id:`${r}-zone-reduce-decode`,operation:`GPURasterZonalStatistics`,variant:`zone-reduce-decode`,bindings:h.map(e=>({name:e.name,view:e.view,type:`u32`,access:`read_write`})),invocationCount:i,declarations:`fn decodeOrderedKey(key: u32) -> u32 {
  return select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u);
}`,body:h.map(e=>`let key_${e.name} = ${e.name}[${e.name}Offset + index];
  ${e.name}[${e.name}Offset + index] = select(decodeOrderedKey(key_${e.name}), 0x7fc00000u, key_${e.name} == ${e.initial});`).join(`
  `)})),u}export{k as n,F as t};
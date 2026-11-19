import{n as e,t}from"./interpolation-view-validation-pYo9tqh3.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-gather-BJIYd708.js";import{t as i}from"./gpu-reduction-tNgOVQ_C.js";import{t as a}from"./gpu-scan-B4NNTfF9.js";import{t as o}from"./gpu-sort-xRm8l5yn.js";import{a as s,o as c,t as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as u,f as d,v as f}from"./gpu-contributor-utils-BIwLQON5.js";import{a as p}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as m,n as h,r as g}from"./sorted-segment-sums-Dvmm6OdR.js";import{a as _,d as v,v as y}from"./spatial-context-DwZEd4EZ.js";import{i as b,n as x}from"./spherical-cell-area-3w3VAb_m.js";import{u as S}from"./raster-band-utils-BcTxc8_Y.js";import{t as C}from"./zone-anchor-nodes-Dhba8wh9.js";import{t as w}from"./define-run-B-brkCe4.js";import{A as T}from"./pointer-overlay-Qh8qflcA.js";import{u as E}from"./inputs-CJKKIRKK.js";import{Ls as D}from"./entry-preview-D70TdSbK.js";import{t as O}from"./grid-CuWZ3T2e.js";import{n as k,r as A,t as j}from"./zones-ab5oHm8d.js";var M=`GPUPycnophylactic`,N=Object.freeze({..._,measure:`area`}),P=class{id;props;spatialContext;constructor(n){let r=n.id??`pycnophylactic`;this.id=r,this.props=n,this.spatialContext=y(r,n.spatialContext,N);let i=S(r,n.grid);if(d(r,`zoneCount`,n.zoneCount),d(r,`maximumIterations`,n.maximumIterations,0,512),n.kernel!==void 0&&n.kernel!==`rook`&&n.kernel!==`box`)throw Error(`${r} kernel must be 'rook' or 'box'`);let{output:a}=n;if(u(n.zones,[`uint32`],`${r} zones`),u(n.totals,[`float32`],`${r} totals`),u(a.mass,[`float32`],`${r} output.mass`),n.zones.length!==i)throw Error(`${r} zones must contain one row per grid cell`);if(a.mass.length!==i)throw Error(`${r} output.mass must contain one row per grid cell`);if(n.totals.length!==n.zoneCount)throw Error(`${r} totals must contain zoneCount rows`);if(this.spatialContext.coordinateSpace===`longitude-latitude`)b(r,n.parameters);else if(n.parameters)throw Error(`${r} parameters need a longitude-latitude spatialContext`);if(a.density&&(u(a.density,[`float32`],`${r} output.density`),a.density.length!==i))throw Error(`${r} output.density must contain one row per grid cell`);if(n.zoneAnchorCells&&(u(n.zoneAnchorCells,[`uint32`],`${r} zoneAnchorCells`),n.zoneAnchorCells.length!==n.zoneCount))throw Error(`${r} zoneAnchorCells must have zoneCount rows`);if(n.tolerance&&(u(n.tolerance,[`float32`],`${r} tolerance`),n.tolerance.length<1))throw Error(`${r} tolerance must hold one float32`);for(let[e,t]of[[`output.status.converged`,a.status?.converged],[`output.status.iterationCount`,a.status?.iterationCount],[`output.status.invalidCount`,a.status?.invalidCount]])if(t&&(u(t,[`uint32`],`${r} ${e}`),t.length<1))throw Error(`${r} ${e} must hold one uint32`);if((a.status?.converged||a.status?.iterationCount)&&!n.tolerance)throw Error(`${r} output.status.converged and output.status.iterationCount need a tolerance`);t(r,this.getOutputBindings(),[{write:{mass:a.mass,density:a.density}},{write:{invalidCount:a.status?.invalidCount}},{write:{converged:a.status?.converged,iterationCount:a.status?.iterationCount}}]),e(r,this.getOutputViews(),this.getInputViews())}getInputViews(){let{props:e}=this;return[e.zones,e.totals,e.parameters,e.zoneAnchorCells,e.tolerance]}getOutputViews(){return Object.values(this.getOutputBindings())}getOutputBindings(){let{output:e}=this.props;return{"output.mass":e.mass,"output.density":e.density,"output.status.converged":e.status?.converged,"output.status.iterationCount":e.status?.iterationCount,"output.status.invalidCount":e.status?.invalidCount}}getCommandNodes(e){let{id:t,props:u}=this,{width:d,height:_}=u.grid,{totals:y,zoneCount:b,maximumIterations:S,output:w}=u;p(e,t,{zones:u.zones,totals:u.totals,parameters:u.parameters,zoneAnchorCells:u.zoneAnchorCells,tolerance:u.tolerance,...this.getOutputBindings()});let T=d*_;for(let[n,r]of Object.entries({cellScratch:T,zoneScratch:b,segmentOffsets:b+1,convergenceFlags:u.tolerance?2:0,iterationDispatch:u.tolerance?6:0}))f(e.device,t,n,r*Uint32Array.BYTES_PER_ELEMENT,`reduce the grid size or zoneCount`);let E=u.kernel??`rook`,D=this.spatialContext.coordinateSpace===`longitude-latitude`,{tolerance:O}=u,k=[],A=u.zones;if(u.zoneAnchorCells){let n=C(e,{id:t,operation:M,zones:A,zoneCount:b,anchorCells:u.zoneAnchorCells});A=n.zones,k.push(...n.nodes)}let j=n(e,`${t}-zone-counts`,`uint32`,b),N=n(e,`${t}-sort-keys`,`uint32`,T),P=n(e,`${t}-sort-indices`,`uint32`,T),F=n(e,`${t}-sorted-keys`,`uint32`,T),I=n(e,`${t}-sorted-indices`,`uint32`,T),L=n(e,`${t}-segment-offsets`,`uint32`,b+1);k.push(l(e,{id:`${t}-counts-clear`,operation:M,view:j,type:`u32`,value:`0u`}),c(e,{id:`${t}-counts`,operation:M,variant:`counts`,bindings:[{name:`zones`,view:A,type:`u32`,access:`read`},{name:`counts`,view:j,type:`atomic<u32>`,access:`read_write`},{name:`sortKeys`,view:N,type:`u32`,access:`read_write`},{name:`sortIndices`,view:P,type:`u32`,access:`read_write`}],invocationCount:T,declarations:`const ZONE_COUNT: u32 = ${b}u;`,body:`let zone = zones[zonesOffset + index];
  if (zone < ZONE_COUNT) {
    atomicAdd(&counts[countsOffset + zone], 1u);
  }
  sortKeys[sortKeysOffset + index] = min(zone, ZONE_COUNT);
  sortIndices[sortIndicesOffset + index] = index;`}),...new o({id:`${t}-sort`,keys:N,values:P,outputKeys:F,outputValues:I,keyBits:m(b)}).getCommandNodes(e),...new a({id:`${t}-segment-scan`,input:j,output:L,mode:`exclusive`}).getCommandNodes(e),c(e,{id:`${t}-segment-total`,operation:M,variant:`segment-total`,bindings:[{name:`counts`,view:j,type:`u32`,access:`read`},{name:`segmentOffsets`,view:L,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const LAST_ZONE: u32 = ${b-1}u;`,body:`segmentOffsets[segmentOffsetsOffset + LAST_ZONE + 1u] =
    segmentOffsets[segmentOffsetsOffset + LAST_ZONE] + counts[countsOffset + LAST_ZONE];`}));let R=n(e,`${t}-scratch`,`float32`,T),z=n(e,`${t}-sorted-values`,`float32`,T),B=n(e,`${t}-zone-sums`,`float32`,b),V=n(e,`${t}-zone-areas`,`float32`,b),H=`const ZONE_COUNT: u32 = ${b}u;
const WIDTH: u32 = ${d}u;
const HEIGHT: u32 = ${_}u;
fn getTotal(zone: u32) -> f32 {
  let total = totals[totalsOffset + zone];
  // Negative, NaN and infinite totals count as 0.
  let isFinite = (bitcast<u32>(total) & 0x7f800000u) != 0x7f800000u;
  return select(0.0, total, isFinite && total > 0.0);
}`,U=(n,i,a,o=!1)=>[...new r({id:`${t}-${n}-gather`,source:i,indices:I,output:z}).getCommandNodes(e),h(e,{id:`${t}-${n}-sum`,operation:M,segmentCount:b,input:z,segmentOffsets:L,output:a,...o?ee:{}})],W=D?n(e,`${t}-cell-areas`,`float32`,T):void 0;W?k.push(c(e,{id:`${t}-cell-areas`,operation:M,variant:`cell-areas`,bindings:[{name:`zones`,view:A,type:`u32`,access:`read`},{name:`extent`,view:u.parameters,type:`f32`,access:`read`},{name:`cellAreas`,view:W,type:`f32`,access:`read_write`}],invocationCount:T,declarations:`const ZONE_COUNT: u32 = ${b}u;
const WIDTH: u32 = ${d}u;
const HEIGHT: u32 = ${_}u;
${x(v(this.spatialContext))}`,body:`var area = getPlacedCellArea(index / WIDTH);
  // Cells outside the zones and cells without a positive finite area carry no area.
  if (zones[zonesOffset + index] >= ZONE_COUNT || (bitcast<u32>(area) & 0x7f800000u) == 0x7f800000u || !(area > 0.0)) {
    area = 0.0;
  }
  cellAreas[cellAreasOffset + index] = area;`}),...U(`zone-areas`,W,V)):k.push(c(e,{id:`${t}-zone-areas`,operation:M,variant:`zone-areas-from-counts`,bindings:[{name:`counts`,view:j,type:`u32`,access:`read`},{name:`zoneAreas`,view:V,type:`f32`,access:`read_write`}],invocationCount:b,body:`zoneAreas[zoneAreasOffset + index] = f32(counts[countsOffset + index]);`}));let G=(r,i)=>{if(!W)return U(r,i,B,!0);let a=n(e,`${t}-${r}-mass`,`float32`,T);return[c(e,{id:`${t}-${r}-mass`,operation:M,variant:`mass`,...Q,bindings:[{name:`values`,view:i,type:`f32`,access:`read`},{name:`cellAreas`,view:W,type:`f32`,access:`read`},{name:`mass`,view:a,type:`f32`,access:`read_write`}],invocationCount:T,body:`mass[massOffset + index] = values[valuesOffset + index] * cellAreas[cellAreasOffset + index];`}),...U(r,a,B,!0)]};w.status?.invalidCount&&k.push(l(e,{id:`${t}-invalid-count-clear`,operation:M,view:w.status?.invalidCount,type:`u32`,value:`0u`,componentCount:1}),c(e,{id:`${t}-invalid-count`,operation:M,variant:`invalid-count`,bindings:[{name:`zoneAreas`,view:V,type:`f32`,access:`read`},{name:`invalidCount`,view:w.status?.invalidCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:b,body:`if (!(zoneAreas[zoneAreasOffset + index] > 0.0)) {
    atomicAdd(&invalidCount[invalidCountOffset], 1u);
  }`}));let K=D||!!w.density,q=K?n(e,`${t}-surface`,`float32`,T):w.mass,J=e=>(S-1-e)%2==0?q:R,Y=S===0?q:J(0)===q?R:q;k.push(c(e,{id:`${t}-initialize`,operation:M,variant:`initialize`,bindings:[{name:`zones`,view:A,type:`u32`,access:`read`},{name:`totals`,view:y,type:`f32`,access:`read`},{name:`zoneAreas`,view:V,type:`f32`,access:`read`},{name:`destination`,view:Y,type:`f32`,access:`read_write`}],invocationCount:T,declarations:H,body:`let zone = zones[zonesOffset + index];
  var value = 0.0;
  if (zone < ZONE_COUNT) {
    let area = zoneAreas[zoneAreasOffset + zone];
    value = select(0.0, getTotal(zone) / area, area > 0.0);
  }
  destination[destinationOffset + index] = value;`}));let X=O?n(e,`${t}-converged-flag`,`uint32`,2):void 0,Z=X?s(e,{id:`${t}-iteration-dispatch`,operation:M,shapes:[[T],g(b)]}):void 0,Q=Z?Z.getGate(0):{},ee=Z?Z.getGate(1):{};X&&Z&&k.push(c(e,{id:`${t}-converged-flag-clear`,operation:M,variant:`converged-flag-clear`,bindings:[{name:`flag`,view:X,type:`u32`,access:`read_write`},Z.binding],invocationCount:1,body:`flag[flagOffset] = 0u;
  flag[flagOffset + 1u] = 0u;
  ${Z.armWGSL}`}));let $=Y;for(let r=0;r<S;r++){let a=J(r),o=`iteration-${r}`;if(k.push(c(e,{id:`${t}-${o}-smooth`,operation:M,variant:`smooth-${E}`,bindings:[{name:`zones`,view:A,type:`u32`,access:`read`},{name:`totals`,view:y,type:`f32`,access:`read`},{name:`source`,view:$,type:`f32`,access:`read`},{name:`destination`,view:a,type:`f32`,access:`read_write`}],invocationCount:T,...Q,declarations:H,body:`let zone = zones[zonesOffset + index];
  var result = 0.0;
  if (zone < ZONE_COUNT) {
    let column = i32(index % WIDTH);
    let row = i32(index / WIDTH);
    var sum = 0.0;
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let isCenter = dx == 0 && dy == 0;
        let isDiagonal = dx != 0 && dy != 0;
        if ((isCenter && ${E===`rook`?`true`:`false`}) || (isDiagonal && ${E===`rook`?`true`:`false`})) {
          continue;
        }
        let x = column + dx;
        let y = row + dy;
        if (x < 0 || y < 0 || x >= i32(WIDTH) || y >= i32(HEIGHT)) {
          continue;
        }
        let neighbor = u32(y) * WIDTH + u32(x);
        if (zones[zonesOffset + neighbor] < ZONE_COUNT) {
          sum += source[sourceOffset + neighbor];
          count += 1.0;
        }
      }
    }
    result = select(source[sourceOffset + index], sum / count, count > 0.0);
  }
  destination[destinationOffset + index] = result;`}),...G(`${o}-restore`,a),c(e,{id:`${t}-${o}-restore-apply`,operation:M,variant:`restore-and-clamp`,bindings:[{name:`zones`,view:A,type:`u32`,access:`read`},{name:`totals`,view:y,type:`f32`,access:`read`},{name:`zoneAreas`,view:V,type:`f32`,access:`read`},{name:`zoneSums`,view:B,type:`f32`,access:`read`},{name:`destination`,view:a,type:`f32`,access:`read_write`}],invocationCount:T,...Q,declarations:H,body:`let zone = zones[zonesOffset + index];
  if (zone < ZONE_COUNT) {
    let area = zoneAreas[zoneAreasOffset + zone];
    // The shift is a density, so it restores the mass of the zone: sum(a * shift) = T - M.
    let shift = select(0.0, (getTotal(zone) - zoneSums[zoneSumsOffset + zone]) / area, area > 0.0);
    destination[destinationOffset + index] = max(destination[destinationOffset + index] + shift, 0.0);
  }`}),...G(`${o}-scale`,a),c(e,{id:`${t}-${o}-scale-apply`,operation:M,variant:`rescale`,bindings:[{name:`zones`,view:A,type:`u32`,access:`read`},{name:`totals`,view:y,type:`f32`,access:`read`},{name:`zoneAreas`,view:V,type:`f32`,access:`read`},{name:`zoneSums`,view:B,type:`f32`,access:`read`},{name:`destination`,view:a,type:`f32`,access:`read_write`}],invocationCount:T,...Q,declarations:H,body:`let zone = zones[zonesOffset + index];
  if (zone < ZONE_COUNT) {
    let total = getTotal(zone);
    let sum = zoneSums[zoneSumsOffset + zone];
    let area = zoneAreas[zoneAreasOffset + zone];
    // A zone whose clamped mass vanished is filled uniformly.
    destination[destinationOffset + index] = select(
      select(0.0, total / area, area > 0.0),
      destination[destinationOffset + index] * (total / sum),
      sum > 0.0
    );
  }`})),X&&O&&Z){let r=n(e,`${t}-${o}-change`,`float32`,T),s=n(e,`${t}-${o}-maximum`,`float32`,1);k.push(c(e,{id:`${t}-${o}-change`,operation:M,variant:`change`,...Q,bindings:[{name:`zones`,view:A,type:`u32`,access:`read`},{name:`totals`,view:y,type:`f32`,access:`read`},{name:`zoneAreas`,view:V,type:`f32`,access:`read`},{name:`source`,view:$,type:`f32`,access:`read`},{name:`destination`,view:a,type:`f32`,access:`read`},{name:`change`,view:r,type:`f32`,access:`read_write`}],invocationCount:T,declarations:H,body:`let zone = zones[zonesOffset + index];
  var value = 0.0;
  if (zone < ZONE_COUNT) {
    let area = zoneAreas[zoneAreasOffset + zone];
    // Change relative to the mean density of the zone; empty zones cannot change.
    let reference = select(0.0, getTotal(zone) / area, area > 0.0);
    if (reference > 0.0) {
      value = abs(destination[destinationOffset + index] - source[sourceOffset + index]) / reference;
    }
  }
  change[changeOffset + index] = value;`}),...new i({id:`${t}-${o}-maximum`,input:r,output:s,operation:`max`}).getCommandNodes(e),c(e,{id:`${t}-${o}-converged`,operation:M,variant:`converged-flag`,bindings:[{name:`maximumChange`,view:s,type:`f32`,access:`read`},{name:`tolerance`,view:O,type:`f32`,access:`read`},{name:`flag`,view:X,type:`u32`,access:`read_write`},Z.binding],invocationCount:1,body:`if (flag[flagOffset] != 0u) {
    return;
  }
  flag[flagOffset + 1u] = flag[flagOffset + 1u] + 1u;
  if (maximumChange[maximumChangeOffset] <= tolerance[toleranceOffset]) {
    flag[flagOffset] = 1u;
    ${Z.disarmWGSL}
  }`}))}$=a}return X&&S>0&&k.push(c(e,{id:`${t}-settle`,operation:M,variant:`settle`,bindings:[{name:`flag`,view:X,type:`u32`,access:`read`},{name:`scratch`,view:R,type:`f32`,access:`read`},{name:`surface`,view:q,type:`f32`,access:`read_write`}],invocationCount:T,declarations:`const ITERATIONS: u32 = ${S}u;`,body:`if ((ITERATIONS - flag[flagOffset + 1u]) % 2u == 1u) {
    surface[surfaceOffset + index] = scratch[scratchOffset + index];
  }`})),K&&k.push(c(e,{id:`${t}-convert`,operation:M,variant:`convert${W?`-area`:``}${w.density?`-density`:``}`,bindings:[{name:`surface`,view:q,type:`f32`,access:`read`},...W?[{name:`cellAreas`,view:W,type:`f32`,access:`read`}]:[],{name:`output`,view:w.mass,type:`f32`,access:`read_write`},...w.density?[{name:`density`,view:w.density,type:`f32`,access:`read_write`}]:[]],invocationCount:T,body:`let value = surface[surfaceOffset + index];
  output[outputOffset + index] = value${W?` * cellAreas[cellAreasOffset + index]`:``};
  ${w.density?`density[densityOffset + index] = value;`:``}`})),(w.status?.converged||w.status?.iterationCount)&&k.push(c(e,{id:`${t}-status`,operation:M,variant:`status`,bindings:[{name:`flag`,view:X,type:`u32`,access:`read`},...w.status?.converged?[{name:`converged`,view:w.status?.converged,type:`u32`,access:`read_write`}]:[],...w.status?.iterationCount?[{name:`iterationCount`,view:w.status?.iterationCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,body:`${w.status?.converged?`converged[convergedOffset] = flag[flagOffset];`:``}
  ${w.status?.iterationCount?`iterationCount[iterationCountOffset] = flag[flagOffset + 1u];`:``}`})),k}},F=w({parameters:D,caption:`Chicago tract populations redistributed to cells while preserving each tract total.`,createInputs:({world:e})=>({grid:O(e,40),world:e}),build(e,{grid:t,world:n},r){let i=E(T(11),n,Math.sqrt(r.zoneCount)),a=j(e,t,i),o=e.upload(`totals`,`float32`,i.population),s=A(t,i),c=0;i.population.forEach((e,t)=>{c=Math.max(c,e/Math.max(1,s[t]))});let l=e.allocate(`surface`,`float32`,t.columns*t.rows);return e.graph.add(new P({grid:t.grid,zones:a.view,zoneCount:i.count,totals:o.view,maximumIterations:r.maximumIterations,kernel:r.kernel===0?`rook`:`box`,output:{mass:l.view}})),{outputs:{surface:l,rangeMaximum:c,totals:k(e,a,o,`tract-totals`,e.upload(`tract-cells`,`float32`,s))}}},interaction:{kind:`swipe`,hint:`swipe totals and surface`,rest:[.5,.5],resolve:(e,{values:t})=>({curtain:e.x,glyphs:[{shape:`curtain`,x:e.x}],readout:`tract totals | after ${t.maximumIterations} iterations`})},draw:({inputs:e,outputs:t,interaction:n,world:r})=>{let i=n.curtain??r.width/2,a=[0,t.rangeMaximum];return[{layer:`raster`,values:t.totals,columns:e.grid.columns,rows:e.grid.rows,range:a,clip:{side:`left`,x:i}},{layer:`raster`,values:t.surface,columns:e.grid.columns,rows:e.grid.rows,range:a,smoothing:!0,clip:{side:`right`,x:i}}]}});export{F as default};
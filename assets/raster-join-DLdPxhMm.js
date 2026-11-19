import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{_ as t,n,v as r}from"./spatial-context-DwZEd4EZ.js";import{r as i}from"./kernel-C3tfQOn4.js";import{o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{o as s}from"./sorted-segment-sums-Dvmm6OdR.js";import{E as c,f as l,u,x as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{h as p,u as m}from"./raster-band-utils-BcTxc8_Y.js";import{n as h,t as g}from"./polygon-rasterization-parameters-B9SJE1ko.js";import{u as _}from"./wgsl-common-DfyXiYB0.js";import{t as v}from"./define-run-B-brkCe4.js";import{t as y}from"./workplace-rows-CYMTcZUq.js";import{t as b}from"./tract-zones-Ba7xrNFs.js";import{Ii as x}from"./entry-preview-D70TdSbK.js";import{a as S,n as C,o as w,s as T,u as E}from"./zones-Bbw4kokc.js";var D=`GPURasterJoin`;function O(e){return u(e,4)}function k(e){return Math.min(16777215,u(e,4,1))}var A=256,j=class{id;spatialContext;props;constructor(e){this.id=e.id??`raster-join`,this.spatialContext=r(this.id,e.spatialContext,{coordinateSpaces:[`planar`,`longitude-latitude`],metrics:[`none`,`native`],edgeModels:n}),t(this.id,`zones`,e.zonesSpatialContext,this.spatialContext,n),this.props=e;let{id:i}=this,{output:a}=e,o=m(i,e.grid);if(!Number.isSafeInteger(e.zoneCount)||e.zoneCount<1||e.zoneCount>=4294967295)throw Error(`${i} zoneCount must be a positive integer below the no-zone sentinel`);p(i,e.parameters,4),c(e.points,[`float32x2`],`${i} points`);let s=e.points.length;if(s<1)throw Error(`${i} points must not be empty`);if(e.values&&(c(e.values,[`float32`],`${i} values`),e.values.length!==s))throw Error(`${i} values must contain one row per point`);for(let[t,n]of[[`zones`,e.zones],[`boundary`,e.boundary]])if(n&&(c(n,[`uint32`],`${i} ${t}`),n.length!==o))throw Error(`${i} ${t} must contain one row per cell`);let l=Object.values(a).filter(Boolean);if(l.length===0)throw Error(`${i} requires at least one output`);if(a.sums&&!e.values)throw Error(`${i} sums require values`);if((a.boundaryCounts||a.unassignedBoundaryCount||a.pointBoundaryMask)&&!e.boundary)throw Error(`${i} boundary outputs require boundary`);for(let[t,n,r]of[[`counts`,a.counts,e.zoneCount],[`boundaryCounts`,a.boundaryCounts,e.zoneCount],[`unassignedBoundaryCount`,a.unassignedBoundaryCount,1],[`outsideCount`,a.outsideCount,1],[`pointZones`,a.pointZones,s],[`pointBoundaryMask`,a.pointBoundaryMask,s]])if(n&&(c(n,[`uint32`],`${i} output.${t}`),n.length!==r))throw Error(`${i} output.${t} must contain ${r} uint32 rows`);if(a.sums&&(c(a.sums,[`float32`],`${i} output.sums`),a.sums.length!==e.zoneCount))throw Error(`${i} output.sums must contain zoneCount rows`);f(i,l,[e.parameters,e.points,e.values,e.zones,e.boundary])}getCommandNodes(t){l(this.id,`zoneCount`,this.props.zoneCount,1,this.props.output.sums?k(t.device):O(t.device));let n=this.props;e(t,this.id,[n.parameters,n.points,n.output.sums?n.values:void 0,n.zones,n.boundary,...Object.values(n.output)],{"point scalar planes and sort scratch":n.points.length*4,"zone aggregation counts":n.zoneCount*4,"sorted zone offsets":n.output.sums?(n.zoneCount+1)*4:0,"boundary tallies":n.output.unassignedBoundaryCount||n.output.outsideCount?8:0});let{id:r,props:c}=this,{zoneCount:u,output:f,boundary:p}=c,{width:m,height:v}=c.grid;d(r,t,[c.parameters,c.points,c.values,c.zones,p,...Object.values(f)]);let y=c.points.length,b=[],x=`const WIDTH: u32 = ${m}u;
const HEIGHT: u32 = ${v}u;
const ZONE_COUNT: u32 = ${u}u;
const NO_ZONE: u32 = ${g}u;
// pointStatus values: inside a non-boundary cell, inside a boundary cell, outside the raster.
const STATUS_INTERIOR: u32 = 0u;
const STATUS_BOUNDARY: u32 = 1u;
const STATUS_OUTSIDE: u32 = 2u;`,S=f.pointZones??i(t,`${r}-point-zones`,`uint32`,y),C=i(t,`${r}-point-status`,`uint32`,y),w=[{name:`points`,view:c.points,type:`f32`,access:`read`},{name:`extent`,view:c.parameters,type:`f32`,access:`read`},{name:`zones`,view:c.zones,type:`u32`,access:`read`},{name:`pointZones`,view:S,type:`u32`,access:`read_write`},{name:`pointStatus`,view:C,type:`u32`,access:`read_write`}];p&&w.push({name:`boundary`,view:p,type:`u32`,access:`read`}),f.pointBoundaryMask&&w.push({name:`pointBoundaryMask`,view:f.pointBoundaryMask,type:`u32`,access:`read_write`}),b.push(a(t,{id:`${r}-lookup`,operation:D,variant:`lookup`,bindings:w,invocationCount:y,declarations:`${x}
${h}`,body:`let rasterExtent = getRasterExtent();
  let position = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var zone = NO_ZONE;
  var status = STATUS_OUTSIDE;
  if (rasterExtent.valid && isFiniteValue(position.x) && isFiniteValue(position.y)) {
    let cell = floor((position - rasterExtent.origin) / rasterExtent.cellSize);
    if (cell.x >= 0.0 && cell.y >= 0.0 && cell.x < f32(WIDTH) && cell.y < f32(HEIGHT)) {
      let cellIndex = getStoredRow(u32(cell.y), rasterExtent) * WIDTH + u32(cell.x);
      zone = zones[zonesOffset + cellIndex];
      if (zone >= ZONE_COUNT) {
        zone = NO_ZONE;
      }
      status = STATUS_INTERIOR;
      ${p?`if (boundary[boundaryOffset + cellIndex] != 0u) {
        status = STATUS_BOUNDARY;
      }`:``}
    }
  }
  pointZones[pointZonesOffset + index] = zone;
  pointStatus[pointStatusOffset + index] = status;
  ${f.pointBoundaryMask?`pointBoundaryMask[pointBoundaryMaskOffset + index] = select(0u, 1u, status == STATUS_BOUNDARY);`:``}`}));let T=f.counts,E=f.unassignedBoundaryCount||f.outsideCount?i(t,`${r}-tallies`,`uint32`,2):void 0,j=[{name:`pointZones`,view:S,type:`u32`,access:`read`},{name:`pointStatus`,view:C,type:`u32`,access:`read`}];for(let[e,n]of[[`counts`,T],[`boundaryCounts`,f.boundaryCounts],[`tallies`,E]])n&&(b.push(o(t,{id:`${r}-${e}-reset`,operation:D,view:n,type:`u32`,value:`0u`})),j.push({name:e,view:n,type:`atomic<u32>`,access:`read_write`}));if(j.length>2&&b.push(a(t,{id:`${r}-aggregate`,operation:D,variant:`aggregate`,bindings:j,invocationCount:y,workgroupSize:A,guardIndex:!1,declarations:`${x}
${E?`var<workgroup> workgroupTallies: array<atomic<u32>, 2>;`:``}`,body:`${E?`if (localInvocationIndex < 2u) { atomicStore(&workgroupTallies[localInvocationIndex], 0u); }
  workgroupBarrier();`:``}
  if (index < INVOCATION_COUNT) {
    let zone = pointZones[pointZonesOffset + index];
    let status = pointStatus[pointStatusOffset + index];
    if (zone != NO_ZONE) {
      ${T?`atomicAdd(&counts[countsOffset + zone], 1u);`:``}
      ${f.boundaryCounts?`if (status == STATUS_BOUNDARY) {
        atomicAdd(&boundaryCounts[boundaryCountsOffset + zone], 1u);
      }`:``}
    }
    ${E?`if (zone == NO_ZONE && status == STATUS_BOUNDARY) {
      atomicAdd(&workgroupTallies[0], 1u);
    }
    if (status == STATUS_OUTSIDE) {
      atomicAdd(&workgroupTallies[1], 1u);
    }`:``}
  }
  ${E?`workgroupBarrier();
  if (localInvocationIndex < 2u) {
    let tally = atomicLoad(&workgroupTallies[localInvocationIndex]);
    if (tally > 0u) { atomicAdd(&tallies[talliesOffset + localInvocationIndex], tally); }
  }`:``}`})),E){let e=[{name:`tallies`,view:E,type:`u32`,access:`read`}];f.unassignedBoundaryCount&&e.push({name:`unassignedOut`,view:f.unassignedBoundaryCount,type:`u32`,access:`read_write`}),f.outsideCount&&e.push({name:`outsideOut`,view:f.outsideCount,type:`u32`,access:`read_write`}),b.push(a(t,{id:`${r}-tallies-publish`,operation:D,variant:`publish`,bindings:e,invocationCount:1,body:`${f.unassignedBoundaryCount?`unassignedOut[unassignedOutOffset] = tallies[talliesOffset];`:``}
  ${f.outsideCount?`outsideOut[outsideOutOffset] = tallies[talliesOffset + 1u];`:``}`}))}if(f.sums){let e=i(t,`${r}-contributions`,`float32`,y);b.push(a(t,{id:`${r}-contributions`,operation:D,variant:`contributions`,bindings:[{name:`pointZones`,view:S,type:`u32`,access:`read`},{name:`values`,view:c.values,type:`f32`,access:`read`},{name:`contributions`,view:e,type:`f32`,access:`read_write`}],invocationCount:y,declarations:`const NO_ZONE: u32 = ${g}u;
${_}`,body:`let value = values[valuesOffset + index];
  let isContributing = pointZones[pointZonesOffset + index] != NO_ZONE && isFiniteValue(value);
  contributions[contributionsOffset + index] = select(0.0, value, isContributing);`})),b.push(...s(t,{id:`${r}-sorted`,operation:D,segmentCount:u,segmentKeys:S,reductions:[{name:`sums`,contributions:e,output:f.sums}]}))}return b}},M=v({parameters:x,rebuildOn:[`gridWidth`],caption:`Jobs by Chicago tract on a log scale. Teal dots mark workplace blocks in boundary cells.`,createInputs:({random:e,world:t})=>{let{tracts:n,positions:r,jobs:i,keys:a}=y(e,t);return{world:t,features:b(n),positions:r,jobs:i,keys:a}},build(e,t,n){let{features:r,positions:i,jobs:a}=t,o=C(e,t,n.gridWidth),s=e.upload(`points`,`float32x2`,i),c=e.upload(`point-values`,`float32`,a),l=r.featureCount,u=e.allocate(`sums`,`float32`,l),d=e.allocate(`boundary-mask`,`uint32`,a.length);return e.graph.add(new j({grid:o.grid,parameters:o.extent.view,points:s.view,values:c.view,zones:o.zones.view,boundary:o.boundary.view,zoneCount:l,output:{sums:u.view,pointBoundaryMask:d.view}})),{outputs:{sums:u,boundaryMask:d,pointPositions:s,hoverPoints:e.allocate(`hover-points`,`float32x2`,a.length),mesh:E(e,r)}}},interaction:{kind:`probe`,hint:`probe a tract`,rest:[.5,.5],resolve:(e,{inputs:t,outputs:n,peek:r})=>{let i=w(t.features,e.x,e.y);if(i<0)return{focus:0,readout:`outside the tracts`};let a=r(n.boundaryMask),o=r(n.sums),s=[],c=0;return t.keys.forEach((e,n)=>{e===i&&(s.push(t.positions[n*2],t.positions[n*2+1]),a?.[n]&&c++)}),n.hoverPoints.write(Float32Array.from({length:t.jobs.length*2},(e,t)=>s[t]??0)),{focus:s.length/2,glyphs:[{shape:`polyline`,closed:!0,points:T(t.features,i)}],readout:`tract ${t.features.geoids[i].slice(-6)} | ${o?Math.round(o[i]).toLocaleString():`-`} jobs | ${c} boundary pts`}}},draw:({inputs:e,outputs:t,interaction:n,theme:r})=>[S(e.features,t.mesh,t.sums,{ramp:`sequential`,logarithmic:!0,range:[0,19],alpha:.85}),{layer:`dots`,positions:t.pointPositions,values:t.boundaryMask,radius:2.3,ramp:`categorical`,categories:[r.muted,[0,.85,.8]]},{layer:`lines`,context:!0,segments:t.mesh.outline,color:[...r.ink,.7]},{layer:`dots`,positions:t.hoverPoints,count:typeof n.focus==`number`?n.focus:0,radius:4.5,color:[1,1,1,1]}]});export{M as default};
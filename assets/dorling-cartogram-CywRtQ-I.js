import{c as e,o as t,r as n,t as r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-reduction-tNgOVQ_C.js";import{t as a}from"./gpu-dispatch-utils-CbRzFAgT.js";import{s as o,v as s}from"./spatial-context-DwZEd4EZ.js";import{D as c,E as l,T as u,f as d,g as f,p,v as m,x as h,y as g}from"./gpu-contributor-utils-BIwLQON5.js";import{a as _,c as v,o as y}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as b}from"./gpu-command-graph-BdzdmV6u.js";import{a as x,s as S}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as C}from"./spatial-output-validation-nw5lOmVL.js";import{n as w}from"./point-neighbor-grid-index-D0wS1y1I.js";import{n as T,t as E}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{t as D}from"./define-run-B-brkCe4.js";import{u as O}from"./inputs-CJKKIRKK.js";import{o as k}from"./pointer-overlay-Qh8qflcA.js";import{t as A}from"./kernel-BXyKsp4B.js";import{ns as j}from"./entry-preview-D70TdSbK.js";var M=256;function N(e,t,n,r,i){let o=a(`Dorling keyed sort`,i,M,e.device.limits.maxComputeWorkgroupsPerDimension),s=n.map(e=>(u(t,e.type===`compute`&&!e.condition,`phase nodes must be unconditional compute nodes`),[(e.workload?.maximumWorkgroupCount??o.x*o.y*o.z)*2,2])),c=_(e,{id:`${t}-dispatch`,operation:`GPUDorlingCartogram`,shapes:s});return[y(e,{id:`${t}-phase-gate`,operation:`GPUDorlingCartogram`,variant:`phase-gate`,invocationCount:1,bindings:[{name:`status`,view:r,type:`u32`,access:`read`},c.binding],body:`if (status[statusOffset + 1u] == 0u) { ${c.armWGSL} }
else { ${c.disarmWGSL} }`}),...n.map((e,n)=>{u(t,e.type===`compute`,`phase nodes must be compute nodes`);let{condition:r,extraResources:i}=c.getGate(n);return{...e,condition:r,resources:[...e.resources??[],...i]}})]}var P=256;function F(n,i){let o=a(i.operation,i.invocationCount,P,n.device.limits.maxComputeWorkgroupsPerDimension),s=i.bindings,c=`${s.map((t,n)=>`
const ${t.name}Offset: u32 = ${e(t.view)}u;
@group(0) @binding(${n}) var<storage, ${t.access}> ${t.name}: array<${t.type}>;`).join(`
`)}
${i.declarations??``}
@compute @workgroup_size(${P})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u) {
  ${s.map(e=>`_ = &${e.name};`).join(`
  `)}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex > 0x00ffffffu) { return; }
  let index = workgroupIndex * ${P}u + localInvocationIndex;
  ${i.body}
}`;return b({id:i.id,resources:s.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:i.operation,variant:i.variant,commandCount:1,maximumWorkgroupCount:o.x*o.y*o.z,maximumInvocationCount:i.invocationCount},compile:({device:e})=>{let n=new r(e,{id:i.id,source:c,shaderLayout:{bindings:s.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:r})=>n.dispatch(e,{bindings:Object.fromEntries(s.map(e=>[e.name,t(e.view,r)])),x:o.x,y:o.y,z:o.z}),destroy:()=>n.destroy()}}})}var I=256,L=256,R=0x38d7ea4c68000,z=64,B=1024,V=4.76837158203125e-7,H=1073741822,U=65536,W=32768,G=class{id;props;spatialContext;maximumIterations;areaScale;overlapTolerance;neighborGrid;estimatedCommandNodeCount;constructor(e){this.props=e,this.id=e.id??`dorling-cartogram`,this.spatialContext=s(this.id,e.spatialContext,o),this.maximumIterations=e.maximumIterations??100,this.areaScale=e.areaScale??1,this.overlapTolerance=e.overlapTolerance??.001,u(this.id,e.neighborGrid===void 0||e.neighborGrid===`auto`||typeof e.neighborGrid==`boolean`,`neighborGrid must be true, false or 'auto'`),this.neighborGrid=e.neighborGrid===void 0||e.neighborGrid===`auto`?e.positions.length>=W:e.neighborGrid,d(this.id,`positions rows`,e.positions.length,0,this.neighborGrid?H:U),d(this.id,`maximumIterations`,this.maximumIterations,0,L);for(let[e,t]of[[`areaScale`,this.areaScale],[`overlapTolerance`,this.overlapTolerance]])f(this.id,e,t,0,R),u(this.id,t>=0&&t<=R&&(t===0||Math.fround(t)>0),`${e} must be zero or a positive float32 value at most ${R}`);f(this.id,`attraction`,e.attraction??.01,0,1),e.rowTileSize!==void 0&&d(this.id,`rowTileSize`,e.rowTileSize);let t=e.positions.length,n=[[`positions`,e.positions,`float32x2`,t],[`values`,e.values,`float32`,t],[`sourceIds`,e.sourceIds,`uint32`,t],[`output.positions`,e.output.positions,`float32x2`,t],[`output.radii`,e.output.radii,`float32`,t],[`output.sourceIds`,e.output.sourceIds,`uint32`,t],[`output.metrics`,e.output.metrics,`float32x2`,1],[`output.status`,e.output.status,`uint32`,4]];for(let[e,t,r,i]of n)t&&(l(t,[r],`${this.id} ${e}`),c(this.id,e,t,i));let r=Math.max(1,Math.ceil(Math.log(Math.max(t,1))/Math.log(I))),i=Math.ceil(Math.max(t,1)/(e.rowTileSize??Math.max(t,1))),a=4*(r+2)+1,m=this.neighborGrid?z+2*(r+2)+1:0;this.estimatedCommandNodeCount=t===0?1:3+(this.neighborGrid?r+2:0)+(this.maximumIterations+1)*(m+i*2+a+2)+this.maximumIterations,p(this.id,this.estimatedCommandNodeCount);let h=Object.values(e.output);g(this.id,h,[e.positions,e.values,e.sourceIds]),C(this.id,e.positions.length>0?e.output:{metrics:e.output.metrics,status:e.output.status},[...e.positions.length>0?[[e.output.positions,e.output.radii,e.output.sourceIds,e.output.status]]:[],[e.output.metrics,e.output.status],...e.positions.length>0&&this.maximumIterations>0?[{read:[e.output.status],write:[e.output.positions]}]:[]])}getCommandNodes(e){let{id:t,props:r,maximumIterations:o}=this,{output:s}=r,c=r.positions.length;x(e,t,{...c>0?{positions:r.positions,values:r.values,sourceIds:r.sourceIds,positionsOutput:s.positions,radii:s.radii,sourceIdsOutput:s.sourceIds}:{},metrics:s.metrics,status:s.status});let l=Math.max(1,Math.min(Math.ceil(Math.sqrt(c)),B)),u=l*l,d=Math.max(1,Math.ceil(Math.log(Math.max(c,1))/Math.log(I))),f=Math.ceil(Math.max(c,1)/(r.rowTileSize??Math.max(c,1))),p=(this.neighborGrid?z+2*(d+2)+1:0)+2*f+4*(d+2)+3;for(let[n,r]of[[`positions rows metric scratch`,c===0?0:(c+1)*16],[`position scratch`,c*8],[`reduction scratch`,Math.max(c*4,Math.ceil(c/I)*8)],[`reduced metrics`,16],[`tile parameters`,4],[`grid cell offsets`,this.neighborGrid?(u+1)*4:0],[`grid counts and keys`,this.neighborGrid?Math.max(c,u)*4:0],[`radix histogram and scan scratch`,this.neighborGrid?16*Math.ceil(c/I)*4:0],[`phase dispatch`,c===0?0:p*12]])m(e.device,t,n,r,`reduce positions rows or row tiles, or request a larger maxStorageBufferBindingSize`);if(h(t,e,[r.positions,r.values,r.sourceIds,...Object.values(s)]),c===0)return[y(e,{id:`${t}-empty`,operation:`GPUDorlingCartogram`,variant:`empty`,invocationCount:1,bindings:[{name:`metrics`,view:s.metrics,type:`f32`,access:`read_write`},{name:`status`,view:s.status,type:`u32`,access:`read_write`}],body:`metrics[metricsOffset] = 0.0; metrics[metricsOffset+1u] = 0.0; status[statusOffset] = 0u; status[statusOffset+1u] = 1u; status[statusOffset+2u] = 0u; status[statusOffset+3u] = 0u;`})];let g=n(e,`${t}-scratch`,`float32x2`,Math.max(c,1)),_=n(e,`${t}-row-metrics`,`float32x4`,c+1),b=this.neighborGrid,C=n(e,`${t}-reduction-input`,`float32`,c),D=e.createDataView(C.buffer,{format:`uint32`,length:c}),O=n(e,`${t}-reduced-metrics`,`float32`,4),k=(t,n,r=1)=>e.createDataView(t.buffer,{format:`float32`,length:r,byteOffset:t.byteOffset+n*4}),A=`const ROWS: u32 = ${c}u;
const TOLERANCE: f32 = ${v(this.overlapTolerance)};
const MAXIMUM_MAGNITUDE: f32 = ${v(R)};
const MAXIMUM_ITERATIONS: u32 = ${o}u;
${T}
fn valid(value: f32) -> bool { return isSpatialFinite(value); }
${E}
fn nan(row: u32) -> f32 { _ = row; return getSpatialNaN(); }`,j=S(e,`${t}-tile-parameters`,new Uint32Array([Math.min(c,r.rowTileSize??c)])),M=[...j.nodes];M.push(y(e,{id:`${t}-initialize`,operation:`GPUDorlingCartogram`,variant:`initialize`,invocationCount:Math.max(c,1),declarations:A,bindings:[{name:`initial`,view:r.positions,type:`f32`,access:`read`},{name:`values`,view:r.values,type:`f32`,access:`read`},...r.sourceIds?[{name:`ids`,view:r.sourceIds,type:`u32`,access:`read`}]:[],{name:`centers`,view:s.positions,type:`f32`,access:`read_write`},{name:`radii`,view:s.radii,type:`f32`,access:`read_write`},{name:`sourceIds`,view:s.sourceIds,type:`u32`,access:`read_write`},{name:`status`,view:s.status,type:`u32`,access:`read_write`}],body:`if (index == 0u) { status[statusOffset] = 0u; status[statusOffset+1u] = 0u; status[statusOffset+2u] = 0u; status[statusOffset+3u] = 0u; }
if (index >= ROWS) { return; }
let center = vec2f(initial[initialOffset+index*2u], initial[initialOffset+index*2u+1u]);
let value = values[valuesOffset+index];
let radius = sqrt(value) * ${v(Math.sqrt(this.areaScale/Math.PI))};
let included = valid(center.x) && valid(center.y) && all(abs(center) <= vec2f(MAXIMUM_MAGNITUDE)) && valid(value) && value >= 0.0 && valid(radius) && radius <= MAXIMUM_MAGNITUDE;
centers[centersOffset+index*2u] = select(nan(index), center.x, included);
centers[centersOffset+index*2u+1u] = select(nan(index), center.y, included);
radii[radiiOffset+index] = select(nan(index), radius, included);
sourceIds[sourceIdsOffset+index] = ${r.sourceIds?`ids[idsOffset+index]`:`index`};`})),M.push(y(e,{id:`${t}-reset-state`,operation:`GPUDorlingCartogram`,variant:`reset-state`,invocationCount:1,bindings:[{name:`rowMetrics`,view:_,type:`f32`,access:`read_write`}],declarations:A,body:`rowMetrics[rowMetricsOffset + ROWS * 4u + 1u] = bitcast<f32>(0u);
rowMetrics[rowMetricsOffset + ROWS * 4u + 2u] = bitcast<f32>(0u);`}));let P=t=>{let n=c,r=t.getCommandNodes(e);return r.map((t,i)=>{let o=i===r.length-1?{x:1,y:1,z:1}:a(`Dorling reduction`,n,I,e.device.limits.maxComputeWorkgroupsPerDimension);return n=Math.ceil(n/I),{...t,workload:{...t.workload,maximumWorkgroupCount:o.x*o.y*o.z}}})},L=t=>{let n=Math.min(c,r.rowTileSize??c),i=[{name:`rowMetrics`,view:_,type:`f32`,access:`read_write`},{name:`tileParameters`,view:j.parameters,type:`u32`,access:`read`}],a=[y(e,{id:`${t.id}-tile-reset`,operation:`GPUDorlingCartogram`,variant:`tile-reset`,invocationCount:1,declarations:A,bindings:i,body:`rowMetrics[rowMetricsOffset + ROWS * 4u + 1u] = bitcast<f32>(0u);
rowMetrics[rowMetricsOffset + ROWS * 4u + 3u] = bitcast<f32>(tileParameters[tileParametersOffset]);`})];for(let r=0,o=0;r<c;r+=n,o++)a.push(F(e,{...t,id:`${t.id}-tile-${o}`,invocationCount:Math.min(n,c-r),body:`{
let tileLength = bitcast<u32>(rowMetrics[rowMetricsOffset + ROWS * 4u + 3u]);
if (index >= tileLength) { return; }
let index = index + bitcast<u32>(rowMetrics[rowMetricsOffset + ROWS * 4u + 1u]);
${t.body}
}`}),y(e,{id:`${t.id}-tile-next-${o}`,operation:`GPUDorlingCartogram`,variant:`tile-next`,invocationCount:1,declarations:A,bindings:i,body:`let base = rowMetricsOffset + ROWS * 4u + 1u;
let nextBase = bitcast<u32>(rowMetrics[base]) + tileParameters[tileParametersOffset];
rowMetrics[base] = bitcast<f32>(nextBase);
rowMetrics[rowMetricsOffset + ROWS * 4u + 3u] = bitcast<f32>(min(tileParameters[tileParametersOffset], ROWS - min(ROWS, nextBase)));`}));return a},H=n=>{let r=[];for(let a=0;a<4;a++){let o=a===2;r.push(y(e,{id:`${t}-extract-metric-${n}-${a}`,operation:`GPUDorlingCartogram`,variant:`extract-metric`,invocationCount:c,bindings:[{name:`rowMetrics`,view:_,type:`f32`,access:`read`},{name:`values`,view:o?D:C,type:o?`u32`:`f32`,access:`read_write`}],body:`values[valuesOffset + index] = ${o?`u32`:`f32`}(rowMetrics[rowMetricsOffset + index * 4u + ${a}u]);`}));let s=k(O,a);r.push(...P(new i({id:`${t}-reduce-metric-${n}-${a}`,input:o?D:C,output:o?e.createDataView(s.buffer,{format:`uint32`,length:1,byteOffset:s.byteOffset}):s,operation:o?`sum`:`max`})))}return r.push(y(e,{id:`${t}-diagnose-${n}`,operation:`GPUDorlingCartogram`,variant:`diagnose`,invocationCount:1,declarations:A,bindings:[{name:`reduced`,view:O,type:`u32`,access:`read`},{name:`metrics`,view:s.metrics,type:`f32`,access:`read_write`},{name:`status`,view:s.status,type:`u32`,access:`read_write`},{name:`rowMetrics`,view:_,type:`f32`,access:`read_write`}],body:`let overlap = bitcast<f32>(reduced[reducedOffset]);
let displacement = bitcast<f32>(reduced[reducedOffset + 1u]);
let tolerance = max(TOLERANCE, bitcast<f32>(reduced[reducedOffset + 3u]) * ${V});
metrics[metricsOffset] = overlap; metrics[metricsOffset + 1u] = displacement;
let round = bitcast<u32>(rowMetrics[rowMetricsOffset + ROWS * 4u + 2u]);
status[statusOffset] = round;
rowMetrics[rowMetricsOffset + ROWS * 4u + 2u] = bitcast<f32>(round + 1u);
status[statusOffset + 1u] = select(0u, 1u, overlap <= tolerance);
status[statusOffset + 2u] = select(0u, 1u, overlap > tolerance && status[statusOffset] == MAXIMUM_ITERATIONS);
status[statusOffset + 3u] = reduced[reducedOffset + 2u];`})),r},U=b?n(e,`${t}-grid-bounds`,`float32`,4):void 0,W=b?n(e,`${t}-grid-keys`,`uint32`,c):void 0,G=b?n(e,`${t}-grid-source-rows`,`uint32`,c):void 0,K=`${A}
const GRID_SIDE: u32 = ${l}u;
fn getGridCell(value: f32, minimum: f32, maximum: f32, largestRadius: f32) -> u32 {
  // A radius-derived cell is enlarged only when the bounded grid cannot cover the extent.
  let width = max(largestRadius * 2.0, (maximum - minimum) / f32(GRID_SIDE));
  if (!(width > 0.0)) { return 0u; }
  return u32(clamp(floor((value - minimum) / width), 0.0, f32(GRID_SIDE - 1u)));
}`,q=[],J=e=>`let largestRadius = rowMetrics[rowMetricsOffset + ROWS * 4u];
  let reach = radius + largestRadius;
  let boundsMinimum = vec2f(gridBounds[gridBoundsOffset], gridBounds[gridBoundsOffset + 2u]);
  let boundsMaximum = vec2f(gridBounds[gridBoundsOffset + 1u], gridBounds[gridBoundsOffset + 3u]);
  let firstColumn = max(getGridCell(center.x - reach, boundsMinimum.x, boundsMaximum.x, largestRadius), 1u) - 1u;
  let lastColumn = min(getGridCell(center.x + reach, boundsMinimum.x, boundsMaximum.x, largestRadius) + 1u, GRID_SIDE - 1u);
  let firstRow = max(getGridCell(center.y - reach, boundsMinimum.y, boundsMaximum.y, largestRadius), 1u) - 1u;
  let lastRow = min(getGridCell(center.y + reach, boundsMinimum.y, boundsMaximum.y, largestRadius) + 1u, GRID_SIDE - 1u);
  for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
    for (var cellColumn = firstColumn; cellColumn <= lastColumn; cellColumn++) {
      let cell = cellRow * GRID_SIDE + cellColumn;
      let lastSlot = cellOffsets[cellOffsetsOffset + cell + 1u];
      for (var slot = cellOffsets[cellOffsetsOffset + cell]; slot < lastSlot; slot++) {
        let other = objectIds[objectIdsOffset + slot];
        ${e}
      }
    }
  }`,Y=(n,r)=>{for(let a=0;a<2;a++)M.push(y(e,{id:`${t}-extract-bounds-${r}-${a}`,operation:`GPUDorlingCartogram`,variant:`extract-bounds`,invocationCount:c,bindings:[{name:`centers`,view:n,type:`f32`,access:`read`},{name:`values`,view:C,type:`f32`,access:`read_write`}],body:`values[valuesOffset + index] = centers[centersOffset + index * 2u + ${a}u];`}),...P(new i({id:`${t}-bounds-${r}-${a}`,input:C,output:k(U,a*2,2),operation:`extent`})));M.push(y(e,{id:`${t}-grid-keys-${r}`,operation:`GPUDorlingCartogram`,variant:`grid-keys`,invocationCount:c,declarations:K,bindings:[{name:`centers`,view:n,type:`f32`,access:`read`},{name:`gridBounds`,view:U,type:`f32`,access:`read`},{name:`rowMetrics`,view:_,type:`f32`,access:`read`},{name:`keys`,view:W,type:`u32`,access:`read_write`},{name:`sourceRows`,view:G,type:`u32`,access:`read_write`}],body:`let center = vec2f(centers[centersOffset + index * 2u], centers[centersOffset + index * 2u + 1u]);
let largestRadius = rowMetrics[rowMetricsOffset + ROWS * 4u];
let column = getGridCell(center.x, gridBounds[gridBoundsOffset], gridBounds[gridBoundsOffset + 1u], largestRadius);
let row = getGridCell(center.y, gridBounds[gridBoundsOffset + 2u], gridBounds[gridBoundsOffset + 3u], largestRadius);
keys[keysOffset + index] = select(${u}u, row * GRID_SIDE + column, valid(center.x) && valid(center.y));
sourceRows[sourceRowsOffset + index] = index;`}));let a=w(e,{id:`${t}-grid-${r}`,rowCount:c,cellKeys:W,sourceRows:G,cellCount:u});M.push(...a.nodes),q=[{name:`cellOffsets`,view:a.cellOffsets,type:`u32`,access:`read`},{name:`objectIds`,view:a.sortedRows,type:`u32`,access:`read`},{name:`gridBounds`,view:U,type:`f32`,access:`read`}]},X=(e,n)=>{b&&Y(e,`diagnose-${n}`);let i=`if (index >= ROWS) { return; }
let radius = radii[radiiOffset+index]; var overlap = 0.0; var displacement = 0.0; var coordinate = 0.0;
if (valid(radius)) {
  let center = vec2f(centers[centersOffset+index*2u],centers[centersOffset+index*2u+1u]);
  displacement = distance(center,vec2f(initial[initialOffset+index*2u],initial[initialOffset+index*2u+1u]));
  coordinate = max(abs(center.x), abs(center.y));
  if (radius > 0.0) {
    ${b?J(`let otherRadius = radii[radiiOffset+other];
        if (other == index || !valid(otherRadius) || otherRadius == 0.0) { continue; }
        let otherCenter = vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
        overlap = max(overlap,radius+otherRadius-distance(center,otherCenter));`):`for (var other = index+1u; other < ROWS; other++) {
      let otherRadius = radii[radiiOffset+other];
      if (!valid(otherRadius) || otherRadius == 0.0) { continue; }
      let otherCenter = vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
      overlap = max(overlap,radius+otherRadius-distance(center,otherCenter));
    }`}
  }
}
rowMetrics[rowMetricsOffset+index*4u] = overlap;
rowMetrics[rowMetricsOffset+index*4u+1u] = displacement;
rowMetrics[rowMetricsOffset+index*4u+2u] = select(1.0,0.0,valid(radius));
rowMetrics[rowMetricsOffset+index*4u+3u] = coordinate;`;M.push(...L({id:`${t}-row-diagnose-${n}`,operation:`GPUDorlingCartogram`,variant:`row-diagnose`,invocationCount:Math.max(c,1),declarations:b?K:A,bindings:[{name:`centers`,view:e,type:`f32`,access:`read`},{name:`initial`,view:r.positions,type:`f32`,access:`read`},{name:`radii`,view:s.radii,type:`f32`,access:`read`},{name:`rowMetrics`,view:_,type:`f32`,access:`read_write`},...q],body:i})),M.push(...H(n))};b&&M.push(y(e,{id:`${t}-extract-radius`,operation:`GPUDorlingCartogram`,variant:`extract-radius`,invocationCount:c,declarations:A,bindings:[{name:`radii`,view:s.radii,type:`f32`,access:`read`},{name:`values`,view:C,type:`f32`,access:`read_write`}],body:`let radius = radii[radiiOffset + index];
values[valuesOffset + index] = select(0.0, radius, valid(radius));`}),...P(new i({id:`${t}-maximum-radius`,input:C,output:k(_,c*4),operation:`max`})));let Z=M.splice(0),Q=(n,r)=>{M.push(...N(e,`${t}-${r}`,n,s.status,c))};o===0&&(X(s.positions,0),Q(M.splice(0),`final`));for(let n=1;n<=o;n++){let i=s.positions,a=g,l=M.length;if(b&&Y(i,`relax-${n}`),M.push(...L({id:`${t}-relax-${n}`,operation:`GPUDorlingCartogram`,variant:`relax`,invocationCount:Math.max(c,1),declarations:b?K:A,bindings:[{name:`centers`,view:i,type:`f32`,access:`read`},{name:`destinations`,view:a,type:`f32`,access:`read_write`},{name:`radii`,view:s.radii,type:`f32`,access:`read`},...b?[]:[{name:`status`,view:s.status,type:`u32`,access:`read`}],{name:`initial`,view:r.positions,type:`f32`,access:`read`},{name:`rowMetrics`,view:_,type:`f32`,access:`read_write`},...q],body:`if (index >= ROWS) { return; }
let center = vec2f(centers[centersOffset+index*2u],centers[centersOffset+index*2u+1u]);
let radius = radii[radiiOffset+index]; var correction = vec2f(0.0); var neighbors = 0u;
var overlap = 0.0; var displacement = 0.0; var coordinate = 0.0;
let origin = vec2f(initial[initialOffset+index*2u],initial[initialOffset+index*2u+1u]);
if (valid(radius)) { displacement = distance(center, origin); coordinate = max(abs(center.x), abs(center.y)); }
if (valid(radius) && radius > 0.0) {
  ${b?J(`let otherRadius = radii[radiiOffset+other];
    if (other == index || !valid(otherRadius) || otherRadius <= 0.0) { continue; }
    let delta = center-vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
    let separation = length(delta); let penetration = radius+otherRadius-separation;
    if (penetration > 0.0) {
      overlap = max(overlap, penetration);
      var direction = vec2f(select(-1.0,1.0,index > other),0.0);
      if (separation > 0.0) { direction = delta/separation; }
      correction += direction*(penetration*0.5); neighbors++;
    }`):`for (var other = 0u; other < ROWS; other++) {
    let otherRadius = radii[radiiOffset+other];
    if (other == index || !valid(otherRadius) || otherRadius <= 0.0) { continue; }
    let delta = center-vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
    let separation = length(delta); let penetration = radius+otherRadius-separation;
    if (penetration > 0.0) {
      overlap = max(overlap, penetration);
      var direction = vec2f(select(-1.0,1.0,index > other),0.0);
      if (separation > 0.0) { direction = delta/separation; }
      correction += direction*(penetration*0.5); neighbors++;
    }
  }`}
}
rowMetrics[rowMetricsOffset+index*4u] = overlap;
rowMetrics[rowMetricsOffset+index*4u+1u] = displacement;
rowMetrics[rowMetricsOffset+index*4u+2u] = select(1.0,0.0,valid(radius));
rowMetrics[rowMetricsOffset+index*4u+3u] = coordinate;
var moved = center;
if (valid(radius) && radius > 0.0${b?``:` && status[statusOffset+1u] == 0u`}) {
  moved = center+correction/f32(max(neighbors,1u))+(origin-center)*${v(r.attraction??.01)};
}
destinations[destinationsOffset+index*2u] = moved.x; destinations[destinationsOffset+index*2u+1u] = moved.y;`})),M.push(...H(n-1),y(e,{id:`${t}-commit-${n}`,operation:`GPUDorlingCartogram`,variant:`commit`,invocationCount:Math.max(c*2,1),bindings:[{name:`source`,view:a,type:`f32`,access:`read`},{name:`destinations`,view:s.positions,type:`f32`,access:`read_write`},{name:`status`,view:s.status,type:`u32`,access:`read`}],body:`if (index < ${c*2}u && status[statusOffset+1u] == 0u) { destinations[destinationsOffset+index] = source[sourceOffset+index]; }`})),Q(M.splice(l),`round-${n}`),n===o){let e=M.length;X(s.positions,n),Q(M.splice(e),`final`)}}return[...Z,...M]}},K=28,q=[0,5,20,50,100],J=D({parameters:j,caption:`Centroids relaxed into circles sized by value; muted dots are where they started.`,createInputs:({random:e,world:t})=>{let n=O(e,t,6),r=Float32Array.from(n.values,e=>.012+.03*e);return{count:n.count,centers:n.centers,areas:r,outline:n.outline,outlineCount:n.outline.length/4}},build(e,t,n){let r=e.upload(`centroids`,`float32x2`,t.centers),i=e.upload(`areas`,`float32`,t.areas),a=e.allocate(`circle-centers`,`float32x2`,t.count),o=e.allocate(`circle-radii`,`float32`,t.count),s=e.allocate(`circle-sources`,`uint32`,t.count),c=e.allocate(`circle-metrics`,`float32x2`,1),l=e.allocate(`circle-status`,`uint32`,4);e.graph.add(new G({positions:r.view,values:i.view,maximumIterations:n.maximumIterations,areaScale:n.areaScale,attraction:n.attraction,output:{positions:a.view,radii:o.view,sourceIds:s.view,metrics:c.view,status:l.view}}));let u=e.allocate(`circle-nodes`,`float32x2`,t.count*K),d=e.allocate(`circle-node-values`,`float32`,t.count*K),f=e.upload(`circle-edges`,`uint32x2`,Uint32Array.from({length:t.count*K*2},(e,t)=>{let n=Math.floor(t/2),r=Math.floor(n/K),i=n%K;return r*K+(t%2==0?i:(i+1)%K)}));A(e,{id:`circle-pass`,invocationCount:t.count*K,bindings:[{name:`centers`,buffer:a,access:`read`},{name:`radii`,buffer:o,access:`read`},{name:`nodes`,buffer:u,access:`read_write`},{name:`nodeValues`,buffer:d,access:`read_write`}],body:`
  let circle = index / ${K}u;
  let angle = 6.2831853 * f32(index % ${K}u) / ${K}.0;
  let radius = radii[radiiOffset + circle];
  nodes[nodesOffset + index] = centers[centersOffset + circle] + radius * vec2f(cos(angle), sin(angle));
  nodeValues[nodeValuesOffset + index] = radius;`});let p=e.allocate(`links`,`float32x4`,t.count);return A(e,{id:`link-pass`,invocationCount:t.count,bindings:[{name:`start`,buffer:r,access:`read`},{name:`end`,buffer:a,access:`read`},{name:`links`,buffer:p,access:`read_write`}],body:`
  links[linksOffset + index] = vec4f(start[startOffset + index], end[endOffset + index]);`}),{outputs:{positions:r,nodes:u,nodeValues:d,edges:f,links:p,outline:e.upload(`outline`,`float32x4`,t.outline)}}},rebuildOn:[`maximumIterations`,`areaScale`,`attraction`],buildCache:5,interaction:{kind:`time`,hint:`scrub the relaxation`,rest:[.9,.5],resolve(e,{inputs:t}){let n=q[k(Math.floor(e.unitX*q.length),0,q.length-1)];return{settle:{maximumIterations:n},glyphs:[{shape:`curtain`,x:e.x}],readout:`maximumIterations ${n} · ${t.count} circles`}}},draw:({inputs:e,outputs:t})=>[{layer:`lines`,context:!0,segments:t.outline,count:e.outlineCount,radius:1,alpha:.5},{layer:`lines`,segments:t.links,color:[.8,.8,.85,.35],radius:1},{layer:`dots`,context:!0,positions:t.positions,radius:3},{layer:`segments`,nodes:t.nodes,edges:t.edges,values:t.nodeValues,radius:2.2}]});export{J as default};
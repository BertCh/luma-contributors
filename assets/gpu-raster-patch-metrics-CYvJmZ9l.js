import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./web-mercator-B0kaE2A1.js";import{t as r}from"./scaled-planar-distance-wgsl-BTfGg91B.js";import{c as i,o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as s,s as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{l as d,o as f,s as p,u as m}from"./raster-grid-VB0AM_GY.js";import{d as h,h as g}from"./raster-band-utils-BcTxc8_Y.js";import{r as _}from"./raster-cell-area-scale-DIJz5iDQ.js";import{a as v,i as y,n as b,o as x,r as S,t as C}from"./patch-labels-Bfw_yy99.js";var w=`GPURasterPatchMetrics`,T=`0xffffffffu`,E=class{id;props;patchCapacity;constructor(e){let t=e.id??`raster-patch-metrics`;this.id=t,this.props=e,x(t,e),e.parameters?(d(t,{grid:e.grid,cellSizeMode:e.cellSizeMode}),g(t,e.parameters,8)):(m(t,e.cellSizeMode),h(t,e.grid,e.cellSizeMode));let n=Object.entries(e.output).filter(([,e])=>e!==void 0);if(n.length===0)throw Error(`${t} output must provide at least one column`);this.patchCapacity=n[0][1].length;for(let[e,r]of n)if(e===`areas`||e===`perimeters`?s(r,[`float32`],`${t} output.${e}`):s(r,[`uint32`],`${t} output.${e}`),r.length!==this.patchCapacity)throw Error(`${t} output columns must have identical lengths`);if(this.patchCapacity<1)throw Error(`${t} output columns must hold at least one patch row`);u(t,n.map(([,e])=>e),[...S(e),...e.parameters?[e.parameters]:[]])}getCommandNodes(n){let r=this.props,s=r.grid.width*r.grid.height;e(n,this.id,[...S(r),r.output.areas||r.output.perimeters?r.parameters:void 0,...c(r.output)],{"patch keys":s*4,"patch scalar planes":this.patchCapacity*4,"patch guard state":r.status?.converged||r.status?.overflow?16:0});let{id:u,props:d,patchCapacity:f}=this,{output:p}=d,{width:m,height:h}=d.grid;l(u,n,[...S(d),...d.parameters?[d.parameters]:[],...Object.values(p).filter(e=>e!==void 0)]);let g=d.countRasterBorder??!0,_=(e,r)=>t(n,`${u}-${e}`,`uint32`,r),{node:x,keys:E}=C(n,u,w,d,f),D=[x],O=p.pixelCounts??_(`pixel-counts`,f),k=d.cellSizeMode??`uniform`,P=!!(p.areas||p.perimeters),F=k===`uniform`||!P?void 0:A(d,k),I=`const WIDTH: u32 = ${m}u;
const HEIGHT: u32 = ${h}u;`,L=P?`${I}\n${N(d)}`:I,R=F?`\n${F.wgsl}`:``,z=d.parameters?[{name:`georeference`,view:d.parameters,type:`f32`,access:`read`}]:[];if(D.push(o(n,{id:`${u}-counts-clear`,operation:w,view:O,type:`u32`,value:`0u`}),a(n,{id:`${u}-counts`,operation:w,variant:`counts`,bindings:[{name:`keys`,view:E,type:`u32`,access:`read`},{name:`pixelCounts`,view:O,type:`atomic<u32>`,access:`read_write`}],invocationCount:v(m,h),declarations:I,body:y(m,{onRun:`atomicAdd(&pixelCounts[pixelCountsOffset + runKey - 1u], runEnd - runStart);`})})),p.minColumns||p.minRows||p.maxColumns||p.maxRows){let e={minColumns:p.minColumns??_(`min-columns`,f),minRows:p.minRows??_(`min-rows`,f),maxColumns:p.maxColumns??_(`max-columns`,f),maxRows:p.maxRows??_(`max-rows`,f)};for(let t of[`minColumns`,`minRows`])D.push(o(n,{id:`${u}-${t}-clear`,operation:w,view:e[t],type:`u32`,value:T}));for(let t of[`maxColumns`,`maxRows`])D.push(o(n,{id:`${u}-${t}-clear`,operation:w,view:e[t],type:`u32`,value:`0u`}));D.push(a(n,{id:`${u}-extents`,operation:w,variant:`extents`,bindings:[{name:`keys`,view:E,type:`u32`,access:`read`},{name:`minColumns`,view:e.minColumns,type:`atomic<u32>`,access:`read_write`},{name:`minRows`,view:e.minRows,type:`atomic<u32>`,access:`read_write`},{name:`maxColumns`,view:e.maxColumns,type:`atomic<u32>`,access:`read_write`},{name:`maxRows`,view:e.maxRows,type:`atomic<u32>`,access:`read_write`}],invocationCount:v(m,h),declarations:I,body:y(m,{onRun:`atomicMin(&minColumns[minColumnsOffset + runKey - 1u], runStart);
        atomicMin(&minRows[minRowsOffset + runKey - 1u], row);
        atomicMax(&maxColumns[maxColumnsOffset + runKey - 1u], runEnd - 1u);
        atomicMax(&maxRows[maxRowsOffset + runKey - 1u], row);`})}),a(n,{id:`${u}-extents-empty`,operation:w,variant:`extents-empty`,bindings:[{name:`pixelCounts`,view:O,type:`u32`,access:`read`},{name:`minColumns`,view:e.minColumns,type:`u32`,access:`read_write`},{name:`minRows`,view:e.minRows,type:`u32`,access:`read_write`}],invocationCount:f,body:`if (pixelCounts[pixelCountsOffset + index] == 0u) {
    minColumns[minColumnsOffset + index] = 0u;
    minRows[minRowsOffset + index] = 0u;
  }`}))}if(p.perimeterFaceCounts||p.perimeters){let e=_(`faces-across-rows`,f),t=_(`faces-across-columns`,f),r=F&&p.perimeters?{low:_(`perimeter-low`,f),high:_(`perimeter-high`,f)}:void 0;if(r)for(let[e,t]of[[`perimeter-low`,r.low],[`perimeter-high`,r.high]])D.push(o(n,{id:`${u}-${e}-clear`,operation:w,view:t,type:`u32`,value:`0u`}));D.push(o(n,{id:`${u}-faces-rows-clear`,operation:w,view:e,type:`u32`,value:`0u`}),o(n,{id:`${u}-faces-columns-clear`,operation:w,view:t,type:`u32`,value:`0u`}),a(n,{id:`${u}-faces`,operation:w,variant:`faces`,bindings:[{name:`keys`,view:E,type:`u32`,access:`read`},{name:`acrossRows`,view:e,type:`atomic<u32>`,access:`read_write`},{name:`acrossColumns`,view:t,type:`atomic<u32>`,access:`read_write`},...r?[...z,{name:`perimeterLow`,view:r.low,type:`atomic<u32>`,access:`read_write`},{name:`perimeterHigh`,view:r.high,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:v(m,h),declarations:`${I}${r?`\n${N(d)}${R}`:``}
const COUNT_BORDER: bool = ${g};

// Whether the neighbor at (column + dx, row + dy) differs from \`key\`.
fn isExposed(column: u32, row: u32, dx: i32, dy: i32, key: u32) -> bool {
  let neighborColumn = i32(column) + dx;
  let neighborRow = i32(row) + dy;
  if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(WIDTH) || neighborRow >= i32(HEIGHT)) {
    return COUNT_BORDER;
  }
  return keys[keysOffset + u32(neighborRow) * WIDTH + u32(neighborColumn)] != key;
}`,body:y(m,{declareRunState:`var runRowFaces = 0u;
  var runColumnFaces = 0u;`,onRunStart:`runRowFaces = 0u;
      runColumnFaces = 0u;`,onPixel:`if (isExposed(column, row, 0, -1, key)) { runRowFaces++; }
      if (isExposed(column, row, 0, 1, key)) { runRowFaces++; }
      if (isExposed(column, row, -1, 0, key)) { runColumnFaces++; }
      if (isExposed(column, row, 1, 0, key)) { runColumnFaces++; }`,onRun:`if (runRowFaces > 0u) { atomicAdd(&acrossRows[acrossRowsOffset + runKey - 1u], runRowFaces); }
        if (runColumnFaces > 0u) { atomicAdd(&acrossColumns[acrossColumnsOffset + runKey - 1u], runColumnFaces); }${r?`
        let rowWeights = getRowWeights(row);
        if (runRowFaces > 0u) ${j(`perimeterLow`,`perimeterHigh`,`runRowFaces * rowWeights.y`)}
        if (runColumnFaces > 0u) ${j(`perimeterLow`,`perimeterHigh`,`runColumnFaces * rowWeights.z`)}`:``}`})}));let i=[{name:`acrossRows`,view:e,type:`u32`,access:`read`},{name:`acrossColumns`,view:t,type:`u32`,access:`read`}];r&&i.push({name:`perimeterLow`,view:r.low,type:`u32`,access:`read`},{name:`perimeterHigh`,view:r.high,type:`u32`,access:`read`}),p.perimeterFaceCounts&&i.push({name:`faceCounts`,view:p.perimeterFaceCounts,type:`u32`,access:`read_write`}),p.perimeters&&(i.push(...z),i.push({name:`perimeters`,view:p.perimeters,type:`f32`,access:`read_write`})),D.push(a(n,{id:`${u}-perimeters`,operation:w,variant:`perimeters`,bindings:i,invocationCount:f,declarations:p.perimeters?`${L}${F?R:``}`:``,body:`let rowFaces = acrossRows[acrossRowsOffset + index];
  let columnFaces = acrossColumns[acrossColumnsOffset + index];
  ${p.perimeterFaceCounts?`faceCounts[faceCountsOffset + index] = rowFaces + columnFaces;`:``}
  ${p.perimeters?r&&F?`perimeters[perimetersOffset + index] = ${M(`perimeterLow`,`perimeterHigh`,`index`)} * getPerimeterReference();`:`perimeters[perimetersOffset + index] = f32(rowFaces) * getColumnStep() + f32(columnFaces) * getRowStep();`:``}`}))}if(p.areas&&F){let e=_(`area-low`,f),t=_(`area-high`,f);for(let[r,i]of[[`area-low`,e],[`area-high`,t]])D.push(o(n,{id:`${u}-${r}-clear`,operation:w,view:i,type:`u32`,value:`0u`}));D.push(a(n,{id:`${u}-ground-areas`,operation:w,variant:`ground-areas`,bindings:[{name:`keys`,view:E,type:`u32`,access:`read`},{name:`areaLow`,view:e,type:`atomic<u32>`,access:`read_write`},{name:`areaHigh`,view:t,type:`atomic<u32>`,access:`read_write`},...z],invocationCount:v(m,h),declarations:`${L}${R}`,body:y(m,{onRun:j(`areaLow`,`areaHigh`,`(runEnd - runStart) * getRowWeights(row).x`)})}),a(n,{id:`${u}-areas`,operation:w,variant:`areas`,bindings:[{name:`areaLow`,view:e,type:`u32`,access:`read`},{name:`areaHigh`,view:t,type:`u32`,access:`read`},{name:`areas`,view:p.areas,type:`f32`,access:`read_write`},...z],invocationCount:f,declarations:L,body:`areas[areasOffset + index] = ${M(`areaLow`,`areaHigh`,`index`)} * ${i(F.areaReference)} * getPixelArea();`}))}else p.areas&&D.push(a(n,{id:`${u}-areas`,operation:w,variant:`areas`,bindings:[{name:`pixelCounts`,view:O,type:`u32`,access:`read`},{name:`areas`,view:p.areas,type:`f32`,access:`read_write`},...z],invocationCount:f,declarations:L,body:`areas[areasOffset + index] = f32(pixelCounts[pixelCountsOffset + index]) * getPixelArea();`}));return D.push(...b(n,u,w,d,Object.entries(p).filter(([,e])=>e!==void 0).map(([e,t])=>({view:t,type:e===`areas`||e===`perimeters`?`f32`:`u32`})))),D}},D=67108864,O=new Map;function k(e){let t=O.get(e);if(!t){let r=0,i=0,a=0;for(let t=0;t<=36e3;t++){let o=Math.min(90-t*.005,89.9999999);for(let t of[-1,1]){let s=o*t,c=e===`geographic`?s:n*Math.atanh(Math.sin(s*Math.PI/180)),[l,u]=f(e,c);r=Math.max(r,l),i=Math.max(i,u),a=Math.max(a,l*u)}}t=[r*1.001,i*1.001,a*1.001],O.set(e,t)}return t}function A(e,t){let[n,r,a]=k(t),o=a;return{wgsl:`const AREA_REFERENCE: f32 = ${i(o)};
fn getPerimeterReference() -> f32 {
  return max(${i(n)} * getColumnStep(), ${i(r)} * getRowStep());
}

// Ground weights of one row as 2^-26 fixed point fractions of the largest possible row value.
fn getRowWeights(row: u32) -> vec3<u32> {
  ${_(e.grid,e.parameters?`georeference`:void 0)}
  ${p(t)}
  let weights = min(
    vec3<f32>(
      groundScale.x * groundScale.y / AREA_REFERENCE,
      groundScale.x * getColumnStep() / getPerimeterReference(),
      groundScale.y * getRowStep() / getPerimeterReference()
    ),
    vec3<f32>(1.0)
  );
  return vec3<u32>(round(weights * ${i(D)}));
}`,areaReference:o}}function j(e,t,n){return`{
          let contribution = ${n};
          let previous = atomicAdd(&${e}[${e}Offset + runKey - 1u], contribution);
          if (previous + contribution < previous) { atomicAdd(&${t}[${t}Offset + runKey - 1u], 1u); }
        }`}function M(e,t,n){return`((f32(${t}[${t}Offset + ${n}]) * 4294967296.0 + f32(${e}[${e}Offset + ${n}])) / ${i(D)})`}function N(e){let[t,n,,a,o]=e.grid.affine,s=e=>`georeference[georeferenceOffset + ${e}u]`,c=(e,t)=>`convertPlanarDistanceValue(getPlanarDistanceVectorLength(getPlanarDistanceVector(vec2f(${s(e)}, ${s(t)}), vec2f(0.0))))`;return`${e.parameters?r:``}
fn getColumnStep() -> f32 {
  return ${e.parameters?c(2,4):i(Math.hypot(t,a))};
}
fn getRowStep() -> f32 {
  return ${e.parameters?c(3,5):i(Math.hypot(n,o))};
}
fn getPixelArea() -> f32 {
  return ${e.parameters?`abs(${s(6)})`:i(Math.abs(t*o-n*a))};
}`}export{E as t};
import{i as e,n as t,r as n}from"./gpu-terrain-line-of-sight-XlBQXjG1.js";import{t as r}from"./uint64-wgsl-DCuW6-R6.js";import{n as i,r as a}from"./terrain-binding-preflight-laSEObQE.js";import{n as o,t as s}from"./vertical-datum-BTN03tZP.js";import{r as c}from"./kernel-C3tfQOn4.js";import{o as l,t as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as d,x as f}from"./gpu-contributor-utils-BIwLQON5.js";import{c as p,d as m,h,l as g,n as _,r as v,u as y}from"./raster-band-utils-BcTxc8_Y.js";import{u as b}from"./raster-grid-VB0AM_GY.js";import{n as x,t as S}from"./terrain-sight-line-QKhP0RJG.js";import{t as C}from"./define-run-B-brkCe4.js";import{a as w,f as T,i as E,l as D,m as O,o as k,s as A,u as j}from"./terrain-Cc9dW1uq.js";import{J as M}from"./entry-preview-D70TdSbK.js";var N=`${r}
fn getViewshedBatchStart(batch: u32, batchSize: u32, pixelCount: u32) -> vec2<u32> {
  if (batch <= 0xffffffffu / batchSize) {
    let pair = batch * batchSize;
    return vec2<u32>(pair / pixelCount, pair % pixelCount);
  }
  let pair = divideUint64ByUint32(multiplyUint32Wide(batch, batchSize), pixelCount);
  return vec2<u32>(pair.quotient.y, pair.remainder);
}
`,P=2**28,F=64,I=class{id;props;pyramidLayout;observersPerDispatch;invocationsPerDispatch;constructor(t){this.id=t.id??`terrain-cumulative-viewshed`,this.props=t;let{id:n}=this;if(t.heightReference!==void 0&&t.heightReference!==`ground`&&t.heightReference!==`absolute`)throw Error(`${n} heightReference must be 'ground' or 'absolute'`);if(t.observerVerticalDatum!==void 0&&t.heightReference!==`absolute`)throw Error(`${n} observerVerticalDatum needs heightReference absolute`);s(n,t.observerVerticalDatum,o(t.elevation,t.elevationVerticalDatum));let{output:r}=t,i=y(n,t.grid);if(b(n,t.cellSizeMode),m(n,t.grid,t.cellSizeMode),d(t.observers,[`float32x2`],`${n} observers`),t.observerHeights&&(d(t.observerHeights,[`float32`],`${n} observerHeights`),t.observerHeights.length!==t.observers.length))throw Error(`${n} observerHeights must contain one value per observer`);if(d(r.visibleCount,[`uint32`],`${n} output.visibleCount`),r.visibleCount.length!==i)throw Error(`${n} output.visibleCount must contain one value per pixel`);if(r.marginalCount&&(d(r.marginalCount,[`uint32`],`${n} output.marginalCount`),r.marginalCount.length!==i))throw Error(`${n} output.marginalCount must contain one value per pixel`);if(h(n,t.parameters,12),t.observersPerDispatch!==void 0&&(!Number.isSafeInteger(t.observersPerDispatch)||t.observersPerDispatch<1))throw Error(`${n} observersPerDispatch must be a positive integer`);let a=Math.max(t.observers.length*i,1),c=Math.max(t.grid.width,t.grid.height),l=t.observersPerDispatch===void 0?Math.max(1,Math.floor(P/c)):t.observersPerDispatch*i;if(this.invocationsPerDispatch=Math.min(l,a),this.invocationsPerDispatch+i>4294967295)throw Error(`${n} observersPerDispatch is too large for this grid`);this.observersPerDispatch=this.invocationsPerDispatch/i,this.pyramidLayout=e(n,t.traversal,t.grid,t.pyramid),g(n,[r.visibleCount,r.marginalCount],[..._(t.elevation),t.parameters,t.observers,...t.observerHeights?[t.observerHeights]:[]])}getCommandNodes(e){a(e,this.id,this.props,{"canonical elevation":4*this.props.grid.width*this.props.grid.height,"pyramid fused parameters":i(this.props.pyramid?void 0:this.pyramidLayout),pyramid:this.pyramidLayout&&!this.props.pyramid?8*this.pyramidLayout.length:0,"observer records":12*this.props.observers.length,"batch table":256*Math.max(1,Math.ceil(this.props.observers.length*this.props.grid.width*this.props.grid.height/this.invocationsPerDispatch))});let{id:t,props:r,pyramidLayout:o}=this,{output:s}=r,{width:d,height:m}=r.grid,h=r.cellSizeMode??`uniform`,g=h===`uniform`?``:S(h),_=d*m;p(t,e,r.elevation,[]),f(t,e,[r.pyramid?.combined,r.parameters,r.observers,r.observerHeights,s.visibleCount,s.marginalCount]);let y=v(e,t,r.elevation,r.grid,!0),b=y.band.storage.values,C=y.band.validity,w=[...y.nodes],T=n(e,t,r.traversal,o,b,C,r.pyramid);w.push(...T.nodes);for(let[n,r]of[[`visible`,s.visibleCount],[`marginal`,s.marginalCount]])r&&w.push(u(e,{id:`${t}-${n}-count-zero`,operation:`GPUTerrainCumulativeViewshed`,view:r,type:`u32`,value:`0u`}));let E=r.observers.length,D=`GPUTerrainCumulativeViewshed`,O=c(e,`${t}-observer-record-data`,`float32`,3*E);w.push(l(e,{id:`${t}-observer-records`,operation:D,variant:`observer-records`,bindings:[{name:`observers`,view:r.observers,type:`f32`,access:`read`},...r.observerHeights?[{name:`observerHeights`,view:r.observerHeights,type:`f32`,access:`read`}]:[],{name:`settings`,view:r.parameters,type:`f32`,access:`read`},{name:`records`,view:O,type:`f32`,access:`read_write`}],invocationCount:E,body:`records[recordsOffset + 3u * index] = observers[observersOffset + 2u * index];
  records[recordsOffset + 3u * index + 1u] = observers[observersOffset + 2u * index + 1u];
  records[recordsOffset + 3u * index + 2u] = ${r.observerHeights?`observerHeights[observerHeightsOffset + index]`:`settings[settingsOffset]`};`}));let k=this.invocationsPerDispatch,A=Math.ceil(E*_/k),j=c(e,`${t}-batch-words`,`uint32`,Math.max(A,1)*F);w.push(l(e,{id:`${t}-batch-table`,operation:D,variant:`batch-table`,bindings:[{name:`batchTable`,view:j,type:`u32`,access:`read_write`}],invocationCount:A,declarations:N,body:`let firstPair = getViewshedBatchStart(index, ${k}u, ${_}u);
  batchTable[batchTableOffset + index * ${F}u] = firstPair.x;
  batchTable[batchTableOffset + index * ${F}u + 1u] = firstPair.y;`}));let M=[{name:`elevationValues`,view:b,type:`f32`,access:`read`},{name:`elevationValidity`,view:C,type:`u32`,access:`read`},...T.binding?[T.binding]:[],{name:`settings`,view:r.parameters,type:`f32`,access:`read`},{name:`records`,view:O,type:`f32`,access:`read`},{name:`visibleCount`,view:s.visibleCount,type:`atomic<u32>`,access:`read_write`},...s.marginalCount?[{name:`marginalCount`,view:s.marginalCount,type:`atomic<u32>`,access:`read_write`}]:[]],P=x({width:d,height:m,traversal:r.traversal??`march`,layout:o,clearance:!1});for(let n=0;n<A;n++)w.push(l(e,{id:`${t}-batch-${n}`,operation:D,variant:`${r.traversal===`pyramid`?`pyramid`:`march`}${h===`uniform`?``:`-${h}`}${r.heightReference===`absolute`?`-absolute`:``}`,bindings:[...M,{name:`batchStart`,view:e.createDataView(j.buffer,{format:`uint32`,length:2,byteOffset:j.byteOffset+n*F*4}),type:`u32`,access:`read`}],invocationCount:k,declarations:`${P}${g}
const OBSERVER_COUNT: u32 = ${E}u;
const PIXEL_COUNT: u32 = ${_}u;
fn isInsideGrid(position: vec2<f32>) -> bool {
  return isFiniteValue(position.x) && isFiniteValue(position.y) &&
    position.x >= 0.0 && position.y >= 0.0 &&
    position.x <= f32(WIDTH - 1u) && position.y <= f32(HEIGHT - 1u);
}`,body:`let pair = batchStart[batchStartOffset + 1u] + index;
  let observerIndex = batchStart[batchStartOffset] + pair / PIXEL_COUNT;
  if (observerIndex >= OBSERVER_COUNT) {
    return;
  }
  let pixel = pair % PIXEL_COUNT;
  let column = pixel % WIDTH;
  let row = pixel / WIDTH;
  let observer = vec2<f32>(
    records[recordsOffset + 3u * observerIndex],
    records[recordsOffset + 3u * observerIndex + 1u]);
  let observerHeight = records[recordsOffset + 3u * observerIndex + 2u];
  let targetHeight = settings[settingsOffset + 1u];
  let maxDistance = settings[settingsOffset + 2u];
  let cellSize = ${h===`uniform`?`vec2<f32>(settings[settingsOffset + 3u], settings[settingsOffset + 4u])`:`getRayGroundCell(vec2<f32>(settings[settingsOffset + 3u], settings[settingsOffset + 4u]), settings[settingsOffset + 10u], settings[settingsOffset + 11u], 0.5 * (observer.y + f32(row)) + 0.5)`};
  let curvature = settings[settingsOffset + 5u];
  let tolerance = vec4<f32>(
    settings[settingsOffset + 6u], settings[settingsOffset + 7u],
    settings[settingsOffset + 8u], settings[settingsOffset + 9u]);
  if (!isInsideGrid(observer) || !isValidPixel(column, row)) {
    return;
  }
${r.heightReference===`absolute`?``:`  let observerSample = sampleElevation(observer);
  if (observerSample.y == 0.0) {
    return;
  }
`}  let result = traceSightLine(
    observer, vec2<f32>(f32(column), f32(row)), ${r.heightReference===`absolute`?`observerHeight`:`observerSample.x + observerHeight`},
    getElevation(column, row) + targetHeight, cellSize, curvature, maxDistance, tolerance);
  if (result.code == VISIBLE) {
    atomicAdd(&visibleCount[visibleCountOffset + pixel], 1u);
  }${s.marginalCount?` else if (result.code == MARGINAL) {
    atomicAdd(&marginalCount[marginalCountOffset + pixel], 1u);
  }`:``}`}));return w}},L=[[0,0,0,0],[.99,.8,.3],[.96,.6,.2],[.9,.4,.2],[.8,.2,.3],[.6,.1,.4],[.4,.1,.5],[.25,.1,.55],[.15,.1,.6]],R=C({parameters:M,rebuildOn:[`observerCount`],caption:`Elevation and several towers in; how many towers see each cell out.`,createInputs:({world:e})=>{let t=E(e),n=A(t,8,9).map(e=>[e%t.columns,Math.floor(e/t.columns)]);return{...t,home:n.map(e=>[...e]),towers:n}},build(e,n,r){let{grid:i,band:a,elevation:o,cellCount:s}=O(e,n),c=Math.min(r.observerCount,n.towers.length),l=e.allocate(`observers`,`float32x2`,c),u=e.allocate(`towers`,`float32x2`,c),d=e.allocate(`parameters`,`float32`,12),f=e.allocate(`visible-count`,`uint32`,s);return e.graph.add(new I({grid:i,elevation:a,observers:l.view,parameters:d.view,output:{visibleCount:f.view}})),{outputs:{elevation:o,visibleCount:f,towers:u},update:e=>{let r=n.towers.slice(0,c);l.write(Float32Array.from(r.flat())),u.write(Float32Array.from(r.flatMap(([e,t])=>j(n,n.world,e,t)))),d.write(t({observerHeight:e.observerHeight,maximumDistance:e.maximumDistance},i))}}},interaction:{kind:`drag`,hint:`carry a tower`,rest:({inputs:e,world:t})=>{let[n,r]=j(e,t,e.home[0][0],e.home[0][1]);return[n/t.width,1-r/t.height]},resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){t.towers[0]=D(t,i,e.x,e.y);let[o,s]=t.towers[0],c=a(n.visibleCount),l=0;if(c){let e=0;for(let t of c)t>=2&&e++;l=Math.round(100*e/c.length)}let u=j(t,i,o,s);return{rerun:!0,glyphs:[{shape:`ring`,at:u,radius:r.maximumDistance*T(t,i),dashed:!0},{shape:`crosshair`,at:u}],readout:c?`tower 1 [${o}, ${s}] · seen by 2+ towers: ${l}%`:`tower 1 [${o}, ${s}] · sees ${r.maximumDistance} m`}}},draw:({inputs:e,outputs:t,values:n})=>[w(e,t.elevation),k(e,t.visibleCount,{ramp:`categorical`,categories:L,alpha:.85}),{layer:`dots`,positions:t.towers,radius:6,color:[1,1,1,1]}]});export{R as default};
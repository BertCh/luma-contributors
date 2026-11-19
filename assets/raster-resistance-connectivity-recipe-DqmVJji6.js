import{D as e,S as t,T as n,f as r,i,p as a}from"./gpu-contributor-utils-BIwLQON5.js";import{r as o,u as s}from"./wgsl-common-DfyXiYB0.js";import{f as c,s as l,u}from"./raster-grid-VB0AM_GY.js";import{d,h as f,o as p,r as m,s as h,u as ee}from"./raster-band-utils-BcTxc8_Y.js";import{d as g,o as _,r as v,t as te,u as y}from"./recipe-utils-CICB7qiS.js";import{a as b,i as x,n as S,o as C,r as w,t as ne}from"./sparse-resistance-connectivity-DN04AKGM.js";import{n as T,t as E}from"./gpu-raster-grid-C_izEAl0.js";import{t as D}from"./define-run-B-brkCe4.js";import{n as O,t as k}from"./pick-Bo3t3hjv.js";import{xr as A}from"./entry-preview-D70TdSbK.js";var j=`addRasterResistanceConnectivityRecipe`,re=x,M=S,N=0,P={cellSizeIndex:4,northEdgeIndex:6};function F(e){return e===4?2:4}function I(e,t=new Float32Array(8)){p(j,t,8);let[n,r,,i,a]=e.affine;return t[N]=n,t[1]=r,t[2]=i,t[3]=a,c(t,e,P),t}function L(e,t){return 16+C(e,t)}function R(e){let t=e=>`parameters[parametersOffset + ${e}u]`;return e===`uniform`?`fn getEdgeLength(row: u32, deltaColumn: f32, deltaRow: f32) -> f32 {
  return length(vec2<f32>(${t(N)} * deltaColumn + ${t(1)} * deltaRow, ${t(2)} * deltaColumn + ${t(3)} * deltaRow));
}`:`fn getEdgeLength(row: u32, deltaColumn: f32, deltaRow: f32) -> f32 {
  let rowCoordinate = f32(row) + 0.5 + 0.5 * deltaRow;
  let edge = mix(${t(P.northEdgeIndex)}, ${t(P.northEdgeIndex+1)}, rowCoordinate / f32(HEIGHT));
  ${l(e)}
  let ground = vec2<f32>(${t(P.cellSizeIndex)}, ${t(P.cellSizeIndex+1)}) * groundScale;
  return length(vec2<f32>(ground.x * abs(deltaColumn), ground.y * abs(deltaRow)));
}`}function z(c,l){let p=l.id??`raster-resistance-connectivity`,{grid:x,output:S}=l,C=l.neighborhood??8,T=l.cellSizeMode??`uniform`,E=l.maximumIterations??1024,D=l.connectivityIterations??128,O=ee(p,x);n(p,C===4||C===8,`neighborhood must be 4 or 8`),t(p,`grid cells (width * height)`,O,re);let k=F(C),A=O*k;t(p,`edge slots (grid cells * ${k} for neighborhood ${C})`,A,M),n(p,O>=2,`grid must have at least 2 cells`),r(p,`source`,l.source,0,O-1),r(p,`sink`,l.sink,0,O-1),n(p,l.source!==l.sink,`source and sink must be different cells`),r(p,`maximumIterations`,E,1,w),r(p,`connectivityIterations`,D,1,ne),n(p,l.tolerance===void 0||Number.isFinite(l.tolerance)&&l.tolerance>0,`tolerance must be a positive finite number`),u(p,T),d(p,x,T),h(l.resistance,x,`${p} resistance`),n(p,!l.resistance.validity||l.resistance.validity.length===O,`resistance.validity must hold one uint32 word per grid cell`),f(p,l.parameters,8),e(p,`output.potentials`,S.potentials,O),e(p,`output.vertexCurrents`,S.vertexCurrents,O),S.currents&&e(p,`output.currents`,S.currents,A);let N=L(E,D);a(p,N);let P=y(c,`${p}-edges`,`uint32x2`,A,S.edges),I=y(c,`${p}-conductances`,`float32`,A,S.conductances),z=e=>y(c,`${p}-${e}`,`uint32`,1,S.status?.[e]),B=z(`invalidCellCount`),V=z(`converged`),H=z(`iterationCount`),U=z(`iterationLimitReached`),W=z(`invalidEdgeCount`),G=[B,V,H,U,W];g(c,p,[l.parameters,...l.resistance.storage.kind===`buffer`?[l.resistance.storage.values]:[],l.resistance.validity],[S.potentials,S.vertexCurrents,S.diagnostics,S.solverStatus,S.currents,P,I,...G]);let K=new te(c),q=m(c,`${p}-resistance`,l.resistance,x,`reuse-canonical`);q.nodes.length>0&&K.addGlue(`resistance-band`,q.nodes);let J=q.band.storage.values,Y=q.band.validity,X=[{name:`resistance`,view:J,type:`f32`,access:`read`},{name:`validity`,view:Y,type:`u32`,access:`read`}],Z=`${s}
fn hasData(cell: u32) -> bool {
  return validity[validityOffset + cell] != 0u && isFiniteValue(resistance[resistanceOffset + cell]);
}`;K.addGlue(`edges`,_(c,{id:`${p}-edges`,operation:j,variant:`neighborhood-${C}-${T}`,invocationCount:A,declarations:`override WIDTH: u32;
override HEIGHT: u32;
override SLOTS_PER_CELL: u32;
${Z}
${o}
${R(T)}`,constants:{WIDTH:x.width,HEIGHT:x.height,SLOTS_PER_CELL:k},bindings:[...X,{name:`parameters`,view:l.parameters,type:`f32`,access:`read`},{name:`edges`,view:P,type:`u32`,access:`read_write`},{name:`conductances`,view:I,type:`f32`,access:`read_write`}],body:`let cell = index / SLOTS_PER_CELL;
  let direction = index % SLOTS_PER_CELL;
  let column = cell % WIDTH;
  let row = cell / WIDTH;
  // Directions: 0 east, 1 south, 2 south-east, 3 south-west.
  let deltaColumn = select(select(select(-1i, 1i, direction == 2u), 0i, direction == 1u), 1i, direction == 0u);
  let deltaRow = select(1i, 0i, direction == 0u);
  let otherColumn = i32(column) + deltaColumn;
  let otherRow = row + u32(deltaRow);
  let inside = otherColumn >= 0i && otherColumn < i32(WIDTH) && otherRow < HEIGHT;
  let other = select(cell, otherRow * WIDTH + u32(max(otherColumn, 0i)), inside);
  edges[edgesOffset + index * 2u] = cell;
  edges[edgesOffset + index * 2u + 1u] = other;
  var conductance = 0.0;
  if (inside && hasData(cell) && hasData(other)) {
    let first = resistance[resistanceOffset + cell];
    let second = resistance[resistanceOffset + other];
    conductance = getNaN();
    if (first > 0.0 && second > 0.0) {
      let value = 2.0 / (first + second) / getEdgeLength(row, f32(deltaColumn), f32(deltaRow));
      if (isFiniteValue(value)) { conductance = value; }
    }
  }
  conductances[conductancesOffset + index] = conductance;`})),v(K,{id:`${p}-invalid-cells`,operation:j,rowCount:O,bindings:[...X,{name:`edges`,view:P,type:`u32`,access:`read`},{name:`conductances`,view:I,type:`f32`,access:`read`}],declarations:`${Z}
const WIDTH: u32 = ${x.width}u;
const SLOTS_PER_CELL: u32 = ${k}u;
fn isOverflowSlot(slot: u32) -> bool {
  let bits = bitcast<u32>(conductances[conductancesOffset + slot]) & 0x7fffffffu;
  let first = edges[edgesOffset + slot * 2u];
  let second = edges[edgesOffset + slot * 2u + 1u];
  return bits > 0x7f800000u &&
    resistance[resistanceOffset + first] > 0.0 && resistance[resistanceOffset + second] > 0.0;
}
fn isInvalidCell(cell: u32) -> bool {
  if (!hasData(cell)) {
    return false;
  }
  if (!(resistance[resistanceOffset + cell] > 0.0)) {
    return true;
  }
  let column = cell % WIDTH;
  var invalid = false;
  for (var direction = 0u; direction < SLOTS_PER_CELL; direction++) {
    invalid = invalid || isOverflowSlot(cell * SLOTS_PER_CELL + direction);
  }
  if (column > 0u) {
    invalid = invalid || isOverflowSlot((cell - 1u) * SLOTS_PER_CELL);
  }
  if (cell >= WIDTH) {
    let north = cell - WIDTH;
    invalid = invalid || isOverflowSlot(north * SLOTS_PER_CELL + 1u);
    if (SLOTS_PER_CELL == 4u && column > 0u) {
      invalid = invalid || isOverflowSlot((north - 1u) * SLOTS_PER_CELL + 2u);
    }
    if (SLOTS_PER_CELL == 4u && column + 1u < WIDTH) {
      invalid = invalid || isOverflowSlot((north + 1u) * SLOTS_PER_CELL + 3u);
    }
  }
  return invalid;
}`,counts:[{name:`invalidCells`,output:B,value:`select(0u, 1u, isInvalidCell(index))`}]});let Q,ie=i(c,()=>{Q=b(c,{id:`${p}-solve`,edges:P,conductances:I,vertexCount:O,source:l.source,sink:l.sink,maximumIterations:E,tolerance:l.tolerance,connectivityIterations:D,output:{potentials:S.potentials,currents:S.currents,vertexCurrents:S.vertexCurrents,diagnostics:S.diagnostics,status:S.solverStatus}})});K.addGlue(`solve`,ie);let $=Q;return K.addGlue(`solve-status`,_(c,{id:`${p}-solve-status`,operation:j,invocationCount:1,bindings:[{name:`solverStatus`,view:S.solverStatus,type:`u32`,access:`read`},{name:`converged`,view:V,type:`u32`,access:`read_write`},{name:`iterationCount`,view:H,type:`u32`,access:`read_write`},{name:`limitReached`,view:U,type:`u32`,access:`read_write`},{name:`invalidEdges`,view:W,type:`u32`,access:`read_write`}],body:`let isConverged = solverStatus[solverStatusOffset];
  let iterations = solverStatus[solverStatusOffset + 1u];
  let ran = solverStatus[solverStatusOffset + 2u] != 0u && solverStatus[solverStatusOffset + 3u] == 0u;
  converged[convergedOffset] = isConverged;
  iterationCount[iterationCountOffset] = iterations;
  limitReached[limitReachedOffset] = select(0u, 1u, ran && isConverged == 0u && iterations >= ${E}u);
  invalidEdges[invalidEdgesOffset] = solverStatus[solverStatusOffset + 3u];`})),{contributors:K.contributors,outputs:{potentials:$.outputs.potentials,vertexCurrents:$.outputs.vertexCurrents,diagnostics:$.outputs.diagnostics,solverStatus:$.status,currents:$.outputs.currents},intermediates:{resistance:J,validity:Y,edges:P,conductances:I},status:{stages:[{stage:`resistance`,status:{invalidCount:B}},{stage:`solve`,status:{converged:V,iterationCount:H,iterationLimitReached:U,invalidCount:W}}]},estimatedCommandNodeCount:N,provenance:{...$.provenance,grid:x,neighborhood:C,cellSizeMode:T,conductance:`2 / (R(a) + R(b)) / edgeLength`}}}var B=32,V=[.12,.3,.5,.7,.88],H=.07,U=[.93,.5],W=.56,G=[.3,.42];function K(e,t){return 1.6+.9*(Math.sin(e*9.1+t*3.7)*Math.cos(t*7.3-e*2.1)+Math.sin(e*4.3-t*6.1))}var q=D({parameters:A,buildCache:10,caption:`Modeled scenario: a resistance raster with a road, one crossing and a lake of no-data cells; unit current from a habitat patch to the ground.`,createInputs:({world:e})=>{let t=Math.round(B*e.width),n=T({width:t,height:B,cellSize:1,origin:[0,B]}),r=1.1/t,i=(e,n)=>Math.min(31,Math.floor(n*B))*t+Math.min(t-1,Math.floor(e*t)),a=new Uint8Array(t*B),o=new Float32Array(t*B);for(let e=0;e<B;e++)for(let n=0;n<t;n++){let i=(n+.5)/t,s=(e+.5)/B,c=e*t+n;o[c]=Math.max(1,K(i,s));let l=Math.abs(i-(W+.05*(s-.5)))<r,u=s>G[0]&&s<G[1];a[c]=l&&!u?1:0,Math.hypot((i-.32)/.12,(s-.74)/.15/1)<1&&(o[c]=NaN)}let s=V.map(e=>i(H,e)),c=Float32Array.from(V.flatMap(t=>[H*e.width,1-t]));for(let e of s)o[e]=1;let l=i(U[0],U[1]);return o[l]=1,{columns:t,grid:n,world:e,base:o,isBarrier:a,patches:s,sink:l,patchPositions:c,patchIndex:O(c),chosen:A[0].default}},build(e,t,n){let{grid:r,columns:i}=t,a=i*B,o=e.allocate(`resistance`,`float32`,a),s=e.allocate(`raster-resistance-parameters`,`float32`,8),c=e.allocate(`potentials`,`float32`,a),l=e.allocate(`vertex-currents`,`float32`,a),u=e.allocate(`diagnostics`,`float32x4`,1),d=e.allocate(`solver-status`,`uint32x4`,1);return z(e.graph,{grid:r,resistance:E(o.view,r,{id:`resistance`}),parameters:s.view,neighborhood:n.neighborhood===0?4:8,source:t.patches[n.source],sink:t.sink,maximumIterations:1024,output:{potentials:c.view,vertexCurrents:l.view,diagnostics:u.view,solverStatus:d.view}}),{outputs:{resistance:o,vertexCurrents:l,diagnostics:u,solverStatus:d},update:e=>{s.write(I(r)),o.write(Float32Array.from(t.base,(n,r)=>t.isBarrier[r]?e.barrierResistance:n))}}},rebuildOn:[`source`,`neighborhood`],interaction:{kind:`place`,hint:`pick a source patch`,rest:({toUnit:e,inputs:t})=>{let n=A[0].default;return e([t.patchPositions[n*2],t.patchPositions[n*2+1]])},resolve(e,{inputs:t,outputs:n,values:r,peek:i,world:a}){t.chosen=k(t.patchIndex,e.x,e.y,t.chosen,.1);let o=t.chosen,s=[t.patchPositions[o*2],t.patchPositions[o*2+1]],c=i(n.diagnostics),l=i(n.solverStatus),u=o===r.source&&c&&l?l[0]===1?`effective resistance ${c[0].toPrecision(3)}`:`not converged`:`source patch ${o+1}`;return{settle:{source:o},glyphs:[{shape:`ring`,at:s,radius:.045},{shape:`dot`,at:[U[0]*a.width,1-U[1]]}],readout:u}}},draw:({inputs:e,outputs:t})=>[{layer:`raster`,context:!0,values:t.resistance,columns:e.columns,rows:B,ramp:`grey`,range:[1,4]},{layer:`raster`,values:t.vertexCurrents,columns:e.columns,rows:B,ramp:`sequential`,range:[0,.12],fade:!0}]});export{q as default};
import{n as e,t}from"./raster-storage-preflight-CIxFc-0g.js";import{T as n,g as r,p as i,s as a}from"./gpu-contributor-utils-BIwLQON5.js";import{t as o}from"./status-aOwAAXek.js";import{r as s,s as c,u as l}from"./raster-grid-VB0AM_GY.js";import{r as u}from"./kernel-C3tfQOn4.js";import{d,h as f,n as p,o as m,r as h}from"./raster-band-utils-BcTxc8_Y.js";import{o as g}from"./wgsl-kernel-nodes-DagJklH2.js";import{u as _}from"./wgsl-common-DfyXiYB0.js";import{a as v,i as y,n as b,r as x,t as S}from"./raster-expansion-utils-AZOLseCG.js";import{n as C,t as w}from"./gpu-raster-grid-C_izEAl0.js";import{t as T}from"./define-run-B-brkCe4.js";import{n as E,t as D}from"./elevation-DSYnxAKv.js";import{o as O,s as k}from"./pointer-overlay-Qh8qflcA.js";import{xi as A}from"./entry-preview-D70TdSbK.js";import{t as j}from"./bands-CVDIl0aE.js";import{a as M,i as N,r as P}from"./gestures-C1AoqRjN.js";var F=128;function I(e,t,n=new Float32Array(16)){let r=`getGPURasterWalkingCostParameterValues`;m(r,n,16),L(r,e);let[i,a,,o,c]=t.affine,{northEdge:l,southEdge:u}=s(t);return n.fill(0,0,16),n.set([i,a,o,c,l,u,e.frictionWeight??1],0),e.model?.kind===`grass`&&n.set([...e.model.coefficients,e.model.downhillSlopeThreshold],8),n}function L(e,t){r(e,`frictionWeight`,t.frictionWeight??1,0),t.model?.kind===`grass`&&(n(e,t.model.coefficients.length===4&&t.model.coefficients.every(Number.isFinite),`model.coefficients must be four finite numbers`),n(e,Number.isFinite(t.model.downhillSlopeThreshold)&&t.model.downhillSlopeThreshold<0,`model.downhillSlopeThreshold must be a negative finite number`))}var R=class{id;props;estimatedCommandNodeCount;constructor(e){this.id=e.id??`raster-walking-cost`,this.props=e;let{id:t}=this,r=v(t,e.grid,[...p(e.elevation),...p(e.friction),e.sources,e.parameters],a(e.output));x(e.sources,r,t,`sources`),x(e.output.costs,r,t,`output.costs`),x(e.output.validity,r,t,`output.validity`),x(e.output.status.converged,1,t,`output.status.converged`),x(e.output.status.iterationLimitReached,1,t,`output.status.iterationLimitReached`),x(e.output.status.iterationCount,1,t,`output.status.iterationCount`),x(e.output.status.invalidCount,1,t,`output.status.invalidCount`),o(t,e.output.status),l(t,e.cellSizeMode),d(t,e.grid,e.cellSizeMode),f(t,e.parameters,16);let s=e.maximumIterations??F;n(t,Number.isInteger(s)&&s>0,`maximumIterations must be a positive integer`),L(t,{model:e.model}),this.estimatedCommandNodeCount=18+ +!!e.output.status.invalidCount+b(s+1),i(t,this.estimatedCommandNodeCount)}getCommandNodes(n){let r=this.props,i=r.grid.width*r.grid.height;e(n,this.id,[...p(r.elevation),...p(r.friction),r.sources,r.parameters,...a(r.output)],{"walking surface":i*8,"canonical band and cost planes":i*4,"iteration state":16,"iteration gates":t((r.maximumIterations??F)+1)});let{id:o,props:s}=this,{grid:l,output:d}=s,f=l.width*l.height;y(o,n,[...p(s.elevation),...p(s.friction),s.sources,s.parameters,...a(d)]);let m=h(n,`${o}-elevation`,s.elevation,l,!0),v=h(n,`${o}-friction`,s.friction,l,!0),b=u(n,`${o}-surface`,`float32x2`,f),x=u(n,`${o}-first`,`float32`,f),C=u(n,`${o}-second`,`float32`,f),w=s.parameters,T=[...m.nodes,...v.nodes,g(n,{id:`${o}-prepare`,operation:`GPURasterWalkingCost`,invocationCount:f,bindings:[{name:`elevation`,view:m.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:m.band.validity,type:`u32`,access:`read`},{name:`friction`,view:v.band.storage.values,type:`f32`,access:`read`},{name:`frictionValidity`,view:v.band.validity,type:`u32`,access:`read`},{name:`sources`,view:s.sources,type:`u32`,access:`read`},{name:`surface`,view:b,type:`f32`,access:`read_write`},{name:`first`,view:x,type:`f32`,access:`read_write`}],body:`let valid = elevationValidity[elevationValidityOffset + index] != 0u
    && frictionValidity[frictionValidityOffset + index] != 0u
    && friction[frictionOffset + index] >= 0.0;
  // Invalid cells carry NaN elevation, so no neighbor relaxes through them.
  surface[surfaceOffset + index * 2u] = select(
    bitcast<f32>(0x7fc00000u | (index & 0u)),
    elevation[elevationOffset + index],
    valid
  );
  surface[surfaceOffset + index * 2u + 1u] = friction[frictionOffset + index];
  first[firstOffset + index] = select(
    bitcast<f32>(0x7f800000u | (index & 0u)),
    0.0,
    valid && sources[sourcesOffset + index] != 0u
  );`})],E=s.cellSizeMode??`uniform`,D=e=>`settings[settingsOffset + ${e}u]`,O=s.model?.kind===`grass`?`let energy = ${D(8)} * groundDistance
    + select(
      select(${D(10)} * rise, ${D(11)} * rise, slope < ${D(12)}),
      ${D(9)} * rise,
      rise > 0.0
    );`:`let energy = groundDistance * 0.6 * exp(3.5 * abs(slope + 0.05));`,k=s.model?.kind===`grass`?`fn parametersMatchModel() -> bool {
  return ${D(12)} < 0.0 && isFiniteValue(${D(8)}) && isFiniteValue(${D(9)}) &&
    isFiniteValue(${D(10)}) && isFiniteValue(${D(11)}) && isFiniteValue(${D(12)});
}`:`fn parametersMatchModel() -> bool {
  return ${D(8)} == 0.0 && ${D(9)} == 0.0 && ${D(10)} == 0.0 &&
    ${D(11)} == 0.0 && ${D(12)} == 0.0;
}`;d.status.invalidCount&&T.push(g(n,{id:`${o}-check-parameters`,operation:`GPURasterWalkingCost`,variant:`check-parameters`,invocationCount:1,bindings:[{name:`settings`,view:w,type:`f32`,access:`read`},{name:`invalidCount`,view:d.status.invalidCount,type:`u32`,access:`read_write`}],declarations:`${_}
${k}`,body:`invalidCount[invalidCountOffset] = select(1u, 0u, parametersMatchModel());`}));let A=E===`uniform`?``:`let rowFraction = (f32(row) + 0.5 + 0.5 * rowStep) / f32(HEIGHT);
  let edge = mix(${D(4)}, ${D(5)}, rowFraction);
  ${c(E)}
  worldStep = worldStep * groundScale;`,j=`${_}
const WIDTH: u32 = ${l.width}u;
const HEIGHT: u32 = ${l.height}u;
${k}

// Ground length of the step from a cell in this row to the neighbor at (columnOffset, rowOffset).
fn getStepDistance(row: i32, columnOffset: i32, rowOffset: i32) -> f32 {
  let columnStep = f32(columnOffset);
  let rowStep = f32(rowOffset);
  var worldStep = vec2<f32>(
    ${D(0)} * columnStep + ${D(1)} * rowStep,
    ${D(2)} * columnStep + ${D(3)} * rowStep
  );
  ${A}
  return length(worldStep);
}

// Cost of stepping from the neighbor to this cell: movement energy plus averaged friction.
fn getTransitionCost(
  elevation: f32,
  friction: f32,
  neighborElevation: f32,
  neighborFriction: f32,
  groundDistance: f32
) -> f32 {
  let rise = elevation - neighborElevation;
  let slope = rise / groundDistance;
  ${O}
  return energy + ${D(6)} * 0.5 * (friction + neighborFriction) * groundDistance;
}`,M=x,N=C;return T.push(...S(n,{id:`${o}-iterate`,operation:`GPURasterWalkingCost`,invocationCount:f,roundCount:(s.maximumIterations??F)+1,status:d.status,createRound:({nodeId:e,slotBinding:t,markChangedWGSL:r,condition:i,extraResources:a},o)=>{let s=g(n,{id:e,operation:`GPURasterWalkingCost`,variant:`relaxation-round`,invocationCount:f,bindings:[{name:`surface`,view:b,type:`f32`,access:`read`},{name:`previous`,view:M,type:`f32`,access:`read`},{name:`next`,view:N,type:`f32`,access:`read_write`},{name:`status`,view:o,type:`atomic<u32>`,access:`read_write`},{name:`settings`,view:w,type:`f32`,access:`read`},t],declarations:j,body:`let elevation = surface[surfaceOffset + index * 2u];
  let friction = surface[surfaceOffset + index * 2u + 1u];
  let original = previous[previousOffset + index];
  var best = original;
  if (isFiniteValue(elevation) && parametersMatchModel()) {
    let column = i32(index % WIDTH);
    let row = i32(index / WIDTH);
    for (var rowOffset = -1; rowOffset <= 1; rowOffset++) {
      for (var columnOffset = -1; columnOffset <= 1; columnOffset++) {
        if (columnOffset == 0 && rowOffset == 0) { continue; }
        let neighborColumn = column + columnOffset;
        let neighborRow = row + rowOffset;
        if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(WIDTH) || neighborRow >= i32(HEIGHT)) { continue; }
        let neighbor = u32(neighborRow) * WIDTH + u32(neighborColumn);
        let neighborElevation = surface[surfaceOffset + neighbor * 2u];
        if (!isFiniteValue(neighborElevation)) { continue; }
        let groundDistance = getStepDistance(row, columnOffset, rowOffset);
        if (groundDistance <= 0.0) { continue; }
        let cost = getTransitionCost(
          elevation,
          friction,
          neighborElevation,
          surface[surfaceOffset + neighbor * 2u + 1u],
          groundDistance
        );
        if (cost >= 0.0 && isFiniteValue(cost)) {
          best = min(best, previous[previousOffset + neighbor] + cost);
        }
      }
    }
  }
  next[nextOffset + index] = best;
  if (best < original) { ${r} }`,condition:i,extraResources:a});return[M,N]=[N,M],s}}),g(n,{id:`${o}-publish`,operation:`GPURasterWalkingCost`,invocationCount:f,bindings:[{name:`previous`,view:M,type:`f32`,access:`read`},{name:`surface`,view:b,type:`f32`,access:`read`},{name:`costs`,view:d.costs,type:`f32`,access:`read_write`},{name:`validity`,view:d.validity,type:`u32`,access:`read_write`}],declarations:_,body:`// The current iterate is always published; status.converged says whether it is final.
  let valid = isFiniteValue(surface[surfaceOffset + index * 2u]);
  costs[costsOffset + index] = select(
    bitcast<f32>(0x7fc00000u | (index & 0u)),
    previous[previousOffset + index],
    valid
  );
  validity[validityOffset + index] = u32(valid);`})),T}},z=k(A,`sourceX`);function B(e){return e===1?{kind:`grass`,coefficients:[.8,1.2,.5,2],downhillSlopeThreshold:-.1}:{kind:`tobler`}}var V=T({parameters:A,rebuildOn:[`model`,`maximumIterations`],caption:`Terrain, friction and a source in; the walking time to every cell out.`,createInputs:({world:e})=>{let t=Math.round(44*e.width/e.height),{elevation:n,cellSize:r}=E(t,44),i=n.map(e=>e<790?4:1);return{columns:t,rows:44,cellSize:r,elevation:n,values:D(n),friction:i,sourceRow:22}},build(e,t,n){let r=t.values.length,i=C({width:t.columns,height:t.rows,cellSize:t.cellSize}),a=e.upload(`elevation`,`float32`,t.elevation),o=e.upload(`friction`,`float32`,t.friction),s=e.allocate(`sources`,`uint32`,r),c=e.allocate(`costs`,`float32`,r),l=e.allocate(`validity`,`uint32`,r),u=e.allocate(`converged`,`uint32`,1),d=e.allocate(`parameters`,`float32`,16);return e.graph.add(new R({grid:i,elevation:w(a.view,i,{id:`elevation`}),friction:w(o.view,i,{id:`friction`}),sources:s.view,parameters:d.view,model:B(n.model),maximumIterations:n.maximumIterations,output:{costs:c.view,validity:l.view,status:{converged:u.view}}})),{outputs:{terrain:e.upload(`terrain`,`float32`,t.values),costs:c},update:e=>{d.write(I({frictionWeight:e.frictionWeight,model:B(e.model)},i));let n=new Uint32Array(r),a=Math.round(e.sourceX*(t.columns-1));n[t.sourceRow*t.columns+a]=1,s.write(n)}}},interaction:{kind:`place`,hint:`place the walker`,rest:({inputs:e})=>[(Math.round(z.default*(e.columns-1))+.5)/e.columns,(Math.floor(e.rows/2)+.5)/e.rows],resolve:(e,{inputs:t,outputs:n,world:r,peek:i})=>{let{column:a,row:o}=P(t,r,e);t.sourceRow=o;let s=i(n.costs),c=0;if(s)for(let e of s)Number.isFinite(e)&&(c=Math.max(c,e));return{values:{sourceX:O(a/(t.columns-1),z.min,z.max)},rerun:!0,glyphs:[N(t,r,a,o,1),{shape:`crosshair`,at:M(t,r,a,o)}],readout:`source [${a}, ${o}] | farthest ${(c/3600).toFixed(1)} h`}}},draw:({inputs:e,outputs:t})=>[j(e,t.terrain,{context:!0,range:[0,1]}),j(e,t.costs,{ramp:`sequential`,range:[0,28800],bands:12,alpha:.9})]});export{V as default};
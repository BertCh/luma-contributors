import{t as e}from"./gpu-histogram-fxACbUIT.js";import{s as t}from"./kernel-C3tfQOn4.js";import{c as n,d as r,f as i,h as a,n as o,o as s,r as c,u as l,v as u}from"./raster-band-utils-BcTxc8_Y.js";import{o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as f,f as p,g as m,p as h,x as g}from"./gpu-contributor-utils-BIwLQON5.js";import{u as _}from"./raster-grid-VB0AM_GY.js";import{n as v,r as y,t as b}from"./raster-grid-utils-D2opjNmg.js";import{c as x,d as S,f as C,i as w,l as T,m as E,p as D,u as O}from"./raster-relaxation-Di9wjtxp.js";import{o as k,u as A}from"./wgsl-common-DfyXiYB0.js";import{a as j}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as M,r as N}from"./raster-fractional-tile-kernel-D2BjD3g1.js";import{n as P,t as F}from"./gpu-raster-grid-C_izEAl0.js";import{i as I}from"./inputs-CJKKIRKK.js";import{c as L}from"./pick-Bo3t3hjv.js";var R=`GPUCostDistance`,z=512,B=64,V=96,H=256,U=4294967295,W=64,G=4294967295,K=4294967294;function q(e,t,n=new Float32Array(8)){return s(`getGPUCostDistanceParameterValues`,n,8),i(`getGPUCostDistanceParameterValues`,t),n.fill(0,0,8),y(n,t),n[4]=e.costLimit??1/0,n}var J=class{id;props;maximumIterations;maximumTieIterations;estimatedCommandNodeCount;constructor(e){this.id=e.id??`cost-distance`,this.props=e,this.maximumIterations=e.maximumIterations??(e.continuation?B:z);let{id:t}=this,{output:n}=e,s=l(t,e.grid);m(t,`friction.scale`,e.friction.scale??1),m(t,`friction.offset`,e.friction.offset??0),new u({id:`${t} friction`,metadata:e.grid,bands:[e.friction]}),_(t,e.cellSizeMode),r(t,e.grid,e.cellSizeMode),i(t,e.grid),E(t,`maximumIterations`,this.maximumIterations),this.maximumTieIterations=e.maximumTieIterations??this.maximumIterations,E(t,`maximumTieIterations`,this.maximumTieIterations),p(t,`sweepAfterIteration`,e.sweepAfterIteration??V,0,U),this.estimatedCommandNodeCount=W+this.maximumIterations+Math.ceil(this.maximumIterations/4)+(n.backLinks?this.maximumTieIterations+Math.ceil(this.maximumTieIterations/4):0),h(t,this.estimatedCommandNodeCount),a(t,e.parameters,8);for(let[r,i]of[[`sources`,e.sources],[`sourceCount`,e.sourceCount],[`sourceMask`,e.sourceMask],[`output.backLinks`,n.backLinks],[`output.settledAt`,n.settledAt],[`output.bands`,n.bands],[`output.bandCounts`,n.bandCounts],[`output.status.converged`,n.status?.converged],[`output.status.iterationCount`,n.status?.iterationCount],[`output.status.iterationLimitReached`,n.status?.iterationLimitReached]])i&&f(i,[`uint32`],`${t} ${r}`);for(let[r,i]of[[`sourceCosts`,e.sourceCosts],[`output.costs`,n.costs],[`costBreaks`,e.costBreaks],[`frictionParameters`,e.frictionParameters],[`continuation.values`,e.continuation?.values]])i&&f(i,[`float32`],`${t} ${r}`);if(e.continuation){if(f(e.continuation.state,[`uint32`],`${t} continuation.state`),n.backLinks)throw Error(`${t} continuation does not support output.backLinks`);if(e.continuation.values.length!==s)throw Error(`${t} continuation.values must contain one value per cell`);if(e.continuation.state.length!==S(e.grid))throw Error(`${t} continuation.state has the wrong length`)}let c=e.friction.storage;if(c.kind===`buffer`&&c.values.length!==s)throw Error(`${t} friction must contain one value per cell`);if(e.friction.validity&&e.friction.validity.length!==s)throw Error(`${t} friction validity must contain one value per cell`);if(!e.sources&&!e.sourceMask)throw Error(`${t} requires sources or sourceMask`);if((e.sourceCosts||e.sourceCount)&&!e.sources)throw Error(`${t} sourceCosts and sourceCount require sources`);if(e.sourceCosts&&e.sourceCosts.length!==e.sources?.length)throw Error(`${t} sourceCosts length must equal sources length`);if(e.sourceMask&&e.sourceMask.length!==s)throw Error(`${t} sourceMask must contain one value per cell`);for(let[e,r]of[[`output.costs`,n.costs],[`output.backLinks`,n.backLinks],[`output.settledAt`,n.settledAt],[`output.bands`,n.bands]])if(r&&r.length!==s)throw Error(`${t} ${e} must contain one value per cell`);for(let[r,i]of[[`sourceCount`,e.sourceCount],[`output.status.converged`,n.status?.converged],[`output.status.iterationCount`,n.status?.iterationCount],[`output.status.iterationLimitReached`,n.status?.iterationLimitReached]])if(i&&i.length!==1)throw Error(`${t} ${r} must contain exactly one row`);if(e.frictionParameters&&e.frictionParameters.length<2)throw Error(`${t} frictionParameters must contain at least 2 values`);if(!!(n.bands||n.bandCounts)!=!!e.costBreaks)throw Error(`${t} costBreaks is required exactly when output.bands or output.bandCounts is given`);if(e.costBreaks&&e.costBreaks.length<1)throw Error(`${t} costBreaks must contain at least one break`);if(n.bandCounts&&n.bandCounts.length!==e.costBreaks?.length)throw Error(`${t} output.bandCounts length must equal costBreaks length`);N(t,{...n,continuationState:e.continuation?.state,continuationValues:e.continuation?.values},[...o(e.friction),e.parameters,e.sources,e.sourceCosts,e.sourceCount,e.sourceMask,e.costBreaks,e.frictionParameters],[{writes:[e.continuation?.values??n.costs,e.continuation?.state,n.settledAt],reads:[e.parameters]},{writes:[e.continuation?.values??n.costs,e.continuation?.state],reads:[e.sources,e.sourceCosts,e.sourceCount,e.sourceMask,e.parameters]},{writes:[n.costs],reads:[e.continuation?.values]},{writes:[n.backLinks],reads:[n.costs,e.parameters]},{writes:[n.bands],reads:[n.costs,e.costBreaks]},{writes:[n.bandCounts],reads:[n.bands]},{writes:[n.bandCounts],reads:[e.continuation?.state]},{writes:[n.status?.converged,n.status?.iterationCount,n.status?.iterationLimitReached],reads:[e.continuation?.state]}])}getCommandNodes(r){let{id:i,props:a,maximumIterations:s}=this,{output:l}=a,{width:u,height:f}=a.grid,p=a.cellSizeMode??`uniform`;n(i,r,a.friction,[]),g(i,r,[a.parameters,a.frictionParameters,a.sources,a.sourceCosts,a.sourceCount,a.sourceMask,l.costs,l.backLinks,l.settledAt,a.costBreaks,l.bands,l.bandCounts,l.status?.converged,l.status?.iterationCount,l.status?.iterationLimitReached,a.continuation?.state,a.continuation?.values]);let m=u*f,h=[...o(a.friction),a.parameters,a.frictionParameters,a.sources,a.sourceCosts,a.sourceCount,a.sourceMask,a.costBreaks,l.costs,l.backLinks,l.settledAt,l.bands,l.bandCounts,l.status?.converged,l.status?.iterationCount,l.status?.iterationLimitReached,a.continuation?.state,a.continuation?.values];j(r,i,Object.fromEntries(h.map((e,t)=>[`view.${t}`,e])));for(let e of h)e&&M(r.device,i,e.buffer.id,t(e).size,`reduce the packed raster or source capacity`,e.buffer.byteLength);for(let[e,t]of[[`per-cell scratch`,m*4],[`tile stamps`,Math.ceil(u/16)*Math.ceil(f/16)*4],[`gate slots`,(Math.min(4,Math.max(s,l.backLinks?this.maximumTieIterations:0))+1)*H]])M(r.device,i,e,t,`reduce grid dimensions`);let _=c(r,`${i}-friction`,a.friction,a.grid,!0),y=b(r,R,i,`${i}-friction-auxiliary`,`float32`,m),x=[..._.nodes,d(r,{id:`${i}-friction-prepare`,operation:R,variant:`friction-prepare`,bindings:[{name:`frictionValues`,view:_.band.storage.values,type:`f32`,access:`read`},{name:`frictionValidity`,view:_.band.validity,type:`u32`,access:`read`},{name:`auxiliary`,view:y,type:`f32`,access:`read_write`},...a.frictionParameters?[{name:`frictionParameters`,view:a.frictionParameters,type:`f32`,access:`read`}]:[]],invocationCount:m,declarations:`
${A}
${k}`,body:`${a.frictionParameters?`let value = frictionValues[frictionValuesOffset + index] * frictionParameters[frictionParametersOffset] + frictionParameters[frictionParametersOffset + 1u];`:`let value = frictionValues[frictionValuesOffset + index];`}
  let passable = frictionValidity[frictionValidityOffset + index] != 0u && isFiniteValue(value) && value >= 0.0;
  auxiliary[auxiliaryOffset + index] = select(getQuietNaN(), value, passable);`})],{continuation:S}=a,C=S?.values??l.costs,E=Math.max(a.sources?.length??0,a.sourceMask?m:0),D={id:i,operation:R,width:u,height:f,cellSizeMode:p,maxIterations:s,continuation:S&&{state:S.state,restartShapes:[[m],[Math.max(E,1)]]}},N=a.sweepAfterIteration??V,{relaxation:P,resetNodes:F}=T(r,D,!1),I=P.continuation?.getRestartGate(0,`${i}-initialize`);x.push(...F,d(r,{id:`${i}-initialize`,operation:R,variant:`initialize`,bindings:[{name:`costs`,view:C,type:`u32`,access:`read_write`}],invocationCount:m,body:`costs[costsOffset + index] = 0x7f800000u;`,condition:I?.condition,extraResources:I?.extraResources}));let L=!1;((a.sources?.length??0)>0||a.sourceMask)&&(x.push(this.getSeedNode(r,y,P.tileStamps,p,`costs`,{costs:C,gate:P.continuation?.getRestartGate(1,`${i}-seed`)})),L=!0),x.push(...O(r,{...D,relaxation:P,sweepAfterIteration:N,values:C,settledAt:l.settledAt,settledAtResetGate:I,auxiliary:y,settings:a.parameters,additiveEdgeCosts:!0,declarations:`
${Y}
fn getRelaxationEdgeCost(neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  return getEdgeCost(direction, centerRow, neighborAuxiliary, centerAuxiliary);
}
fn getRelaxationLimit() -> f32 { return settings[settingsOffset + 4u]; }`}));let z;if(l.backLinks?z=this.getTieNodes(r,{nodes:x,auxiliary:y,cellSizeMode:p,hasSeedNode:L,sweepAfterIteration:N,costState:P.state}):(l.status?.converged||l.status?.iterationCount||l.status?.iterationLimitReached)&&x.push(w(r,{id:`${i}-finalize`,operation:R,state:P.state,converged:l.status?.converged,iterationCount:l.status?.iterationCount,iterationLimitReached:l.status?.iterationLimitReached})),l.backLinks&&z&&x.push(d(r,{id:`${i}-back-links`,operation:R,variant:`back-links`,bindings:[{name:`costs`,view:l.costs,type:`f32`,access:`read`},{name:`auxiliary`,view:y,type:`f32`,access:`read`},{name:`settings`,view:a.parameters,type:`f32`,access:`read`},{name:`levels`,view:z.levels,type:`f32`,access:`read`},{name:`masks`,view:z.masks,type:`f32`,access:`read`},{name:`costStatus`,view:P.state.status,type:`u32`,access:`read`},{name:`backLinks`,view:l.backLinks,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`${v({width:u,height:f,cellSizeMode:p})}
${Y}
${X}
const UNRESOLVED: u32 = ${K}u;`,body:`let centerAuxiliary = auxiliary[auxiliaryOffset + index];
  let centerCost = costs[costsOffset + index];
  var link = GRID_NONE;
  if (!isNaNValue(centerAuxiliary) && costStatus[costStatusOffset + 3u] == 0u) {
    // The cost relaxation stopped at maxIterations: any cost may still drop and an infinite
    // cost may still become finite, so no link of a passable cell is certified.
    link = UNRESOLVED;
  } else if (!isNaNValue(centerAuxiliary) && isFiniteValue(centerCost)) {
    let strictDirection = getStrictDirection(index, centerAuxiliary, centerCost);
    if (strictDirection != GRID_NONE) {
      link = getD8Code(strictDirection);
    } else {
      // Reached only across zero-cost edges (or a root source): link to the smallest direction
      // one tie level lower. Level 0 without a strict link is a root (link 0). A level the tie
      // relaxation never reached, or one with no neighbor one level lower (a budget cut it
      // short), is unresolved: link 0 would claim a source.
      let level = levels[levelsOffset + index];
      if (isFiniteValue(level) && level < 0.5) {
        link = 0u;
      } else {
        link = UNRESOLVED;
        if (isFiniteValue(level)) {
          let mask = u32(masks[masksOffset + index]);
          for (var direction = 0u; direction < 8u; direction++) {
            if (((mask >> direction) & 1u) == 0u) { continue; }
            let neighbor = getD8Neighbor(index, direction);
            if (neighbor != GRID_NONE && levels[levelsOffset + neighbor] == level - 1.0) {
              link = getD8Code(direction);
              break;
            }
          }
        }
      }
    }
  }
  backLinks[backLinksOffset + index] = link;`})),S&&x.push(d(r,{id:`${i}-publish`,operation:R,variant:`publish`,bindings:[{name:`working`,view:S.values,type:`u32`,access:`read`},{name:`costs`,view:l.costs,type:`u32`,access:`read_write`}],invocationCount:m,body:`costs[costsOffset + index] = working[workingOffset + index];`})),x.push(...this.getUnresolvedCostNodes(r,y,P.state)),a.costBreaks&&(l.bands||l.bandCounts)){let t=l.bands??b(r,R,i,`${i}-bands-scratch`,`uint32`,m);x.push(d(r,{id:`${i}-bands`,operation:R,variant:`bands`,bindings:[{name:`costs`,view:l.costs,type:`f32`,access:`read`},{name:`thresholds`,view:a.costBreaks,type:`f32`,access:`read`},{name:`bands`,view:t,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`const THRESHOLD_COUNT: u32 = ${a.costBreaks.length}u;`,body:`let cost = costs[costsOffset + index];
  var band = ${G}u;
  if ((bitcast<u32>(cost) & 0x7fffffffu) > 0x7f800000u) {
    // NaN: the cost relaxation did not converge, so the cell's band is not known yet.
    band = ${K}u;
  } else if (bitcast<u32>(cost) < 0x7f800000u) {
    var passedCount = 0u;
    for (var thresholdIndex = 0u; thresholdIndex < THRESHOLD_COUNT; thresholdIndex++) {
      if (thresholds[thresholdsOffset + thresholdIndex] < cost) { passedCount++; }
    }
    if (passedCount < THRESHOLD_COUNT) { band = passedCount; }
  }
  bands[bandsOffset + index] = band;`})),l.bandCounts&&(x.push(...new e({id:`${i}-band-counts`,input:t,output:l.bandCounts,domain:[0,a.costBreaks.length]}).getCommandNodes(r)),x.push(d(r,{id:`${i}-band-counts-unresolved`,operation:R,variant:`band-counts-unresolved`,bindings:[{name:`costStatus`,view:P.state.status,type:`u32`,access:`read`},{name:`bandCounts`,view:l.bandCounts,type:`u32`,access:`read_write`}],invocationCount:l.bandCounts.length,body:`if (costStatus[costStatusOffset + 3u] == 0u) {
    bandCounts[bandCountsOffset + index] = ${K}u;
  }`})))}return x}getUnresolvedCostNodes(e,t,n){let{id:r,props:i}=this,{output:a}=i,o=i.grid.width*i.grid.height,s=x(e,`${r}-unresolved`,R,o,r),{x:c,y:l,z:u}=s.layout,{condition:f,extraResources:p}=C(s,`${r}-unresolved-costs`);return[d(e,{id:`${r}-unresolved-arm`,operation:R,variant:`unresolved-arm`,bindings:[{name:`costStatus`,view:n.status,type:`u32`,access:`read`},{name:`dispatch`,view:s.dispatch,type:`u32`,access:`read_write`}],invocationCount:1,body:`let converged = costStatus[costStatusOffset + 3u] != 0u;
  dispatch[dispatchOffset] = select(${c}u, 0u, converged);
  dispatch[dispatchOffset + 1u] = ${l}u;
  dispatch[dispatchOffset + 2u] = ${u}u;`}),d(e,{id:`${r}-unresolved-costs`,operation:R,variant:`unresolved-costs`,bindings:[{name:`auxiliary`,view:t,type:`f32`,access:`read`},{name:`costs`,view:a.costs,type:`u32`,access:`read_write`}],invocationCount:s.invocationCount,workgroupSize:256,body:`if ((bitcast<u32>(auxiliary[auxiliaryOffset + index]) & 0x7fffffffu) <= 0x7f800000u) {
    costs[costsOffset + index] = 0x7fc00000u;
  }`,condition:f,extraResources:p})]}getTieNodes(e,t){let{id:n,props:r,maximumTieIterations:i}=this,{output:a}=r,{width:o,height:s}=r.grid,{nodes:c,auxiliary:l,cellSizeMode:u}=t,f=o*s,p=b(e,R,n,`${n}-tie-levels`,`float32`,f),m=b(e,R,n,`${n}-tie-masks`,`float32`,f);c.push(d(e,{id:`${n}-tie-prepare`,operation:R,variant:`tie-prepare`,bindings:[{name:`costs`,view:a.costs,type:`f32`,access:`read`},{name:`auxiliary`,view:l,type:`f32`,access:`read`},{name:`settings`,view:r.parameters,type:`f32`,access:`read`},{name:`levels`,view:p,type:`u32`,access:`read_write`},{name:`masks`,view:m,type:`f32`,access:`read_write`}],invocationCount:f,declarations:`${v({width:o,height:s,cellSizeMode:u})}
${Y}
${X}`,body:`let centerAuxiliary = auxiliary[auxiliaryOffset + index];
  let centerCost = costs[costsOffset + index];
  var level = 0x7f800000u;
  var mask = 0u;
  var maskValue = getQuietNaN();
  if (!isNaNValue(centerAuxiliary) && isFiniteValue(centerCost)) {
    if (getStrictDirection(index, centerAuxiliary, centerCost) != GRID_NONE) { level = 0u; }
    let centerRow = index / GRID_WIDTH;
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getD8Neighbor(index, direction);
      if (neighbor == GRID_NONE) { continue; }
      let neighborAuxiliary = auxiliary[auxiliaryOffset + neighbor];
      let neighborCost = costs[costsOffset + neighbor];
      if (isNaNValue(neighborAuxiliary) || !(neighborCost == centerCost)) { continue; }
      // A tight tie edge adds nothing to the cost, so the sum stays exactly the neighbor cost.
      if (neighborCost + getEdgeCost(direction, centerRow, neighborAuxiliary, centerAuxiliary) <= centerCost) {
        mask |= 1u << direction;
      }
    }
    maskValue = f32(mask);
  }
  levels[levelsOffset + index] = level;
  masks[masksOffset + index] = maskValue;`})),t.hasSeedNode&&c.push(this.getSeedNode(e,l,p,u,`roots`));let h={id:`${n}-tie`,operation:R,width:o,height:s,cellSizeMode:u,maxIterations:i},{relaxation:g,resetNodes:_}=T(e,h,!0);c.push(..._,...O(e,{...h,relaxation:g,sweepAfterIteration:t.sweepAfterIteration,values:p,auxiliary:m,settings:r.parameters,additiveEdgeCosts:!0,declarations:`
fn getRelaxationEdgeCost(neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  if (((u32(centerAuxiliary) >> direction) & 1u) == 0u) { return getInfinity(); }
  return 1.0;
}
fn getRelaxationLimit() -> f32 { return getInfinity(); }`}));let y=[{name:`costStatus`,view:t.costState.status,type:`u32`,access:`read`},{name:`tieStatus`,view:g.state.status,type:`u32`,access:`read`}];return a.status?.converged&&y.push({name:`converged`,view:a.status?.converged,type:`u32`,access:`read_write`}),a.status?.iterationCount&&y.push({name:`iterationCount`,view:a.status?.iterationCount,type:`u32`,access:`read_write`}),a.status?.iterationLimitReached&&y.push({name:`iterationLimitReached`,view:a.status.iterationLimitReached,type:`u32`,access:`read_write`}),(a.status?.converged||a.status?.iterationCount||a.status?.iterationLimitReached)&&c.push(d(e,{id:`${n}-finalize`,operation:R,variant:`tie-finalize`,bindings:y,invocationCount:1,body:`${a.status?.converged?`converged[convergedOffset] = costStatus[costStatusOffset + 3u] & tieStatus[tieStatusOffset + 3u];`:``}
  ${a.status?.iterationCount?`iterationCount[iterationCountOffset] = costStatus[costStatusOffset + 2u];`:``}
  ${a.status?.iterationLimitReached?`iterationLimitReached[iterationLimitReachedOffset] = 1u - (costStatus[costStatusOffset + 3u] & tieStatus[tieStatusOffset + 3u]);`:``}`})),{levels:p,masks:m}}getSeedNode(e,t,n,r,i=`costs`,a={}){let{id:o,props:s}=this,{output:c}=s,{width:l,height:u}=s.grid,f=s.sources?.length??0,p=l*u,m=[{name:`auxiliary`,view:t,type:`f32`,access:`read`},{name:`settings`,view:s.parameters,type:`f32`,access:`read`}];return s.sources&&m.push({name:`sources`,view:s.sources,type:`u32`,access:`read`}),s.sourceCosts&&m.push({name:`sourceCosts`,view:s.sourceCosts,type:`f32`,access:`read`}),s.sourceCount&&m.push({name:`sourceCount`,view:s.sourceCount,type:`u32`,access:`read`}),s.sourceMask&&m.push({name:`sourceMask`,view:s.sourceMask,type:`u32`,access:`read`}),m.push({name:`costBits`,view:a.costs??c.costs,type:`atomic<u32>`,access:`read_write`},i===`costs`?{name:`tileStamps`,view:n,type:`atomic<u32>`,access:`read_write`}:{name:`levelBits`,view:n,type:`atomic<u32>`,access:`read_write`}),d(e,{id:`${o}-${i===`costs`?`seed`:`tie-roots`}`,operation:R,variant:i===`costs`?`seed`:`tie-roots`,bindings:m,invocationCount:Math.max(f,s.sourceMask?p:0),declarations:`${v({width:l,height:u,cellSizeMode:r})}
${i===`costs`?D(l):``}
const CELL_COUNT: u32 = ${p}u;
const SOURCE_CAPACITY: u32 = ${f}u;

fn seedCell(cell: u32, cost: f32) {
  if (cell >= CELL_COUNT) { return; }
  if (!(cost >= 0.0) || cost > settings[settingsOffset + 4u]) { return; }
  let friction = auxiliary[auxiliaryOffset + cell];
  // Compare bits: NaN comparisons may be folded away by shader compilers.
  if ((bitcast<u32>(friction) & 0x7fffffffu) > 0x7f800000u) { return; }
  // Fold -0.0 so the u32 bit order of non-negative floats matches numeric order.
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= 0x7f800000u) { return; }
${i===`costs`?`atomicMin(&costBits[costBitsOffset + cell], bits);
  atomicMax(&tileStamps[tileStampsOffset + getRelaxationTile(cell)], 1u);`:`if (atomicLoad(&costBits[costBitsOffset + cell]) == bits) {
    atomicStore(&levelBits[levelBitsOffset + cell], 0u);
  }`}
}`,body:`${s.sources?`if (index < SOURCE_CAPACITY${s.sourceCount?` && index < sourceCount[sourceCountOffset]`:``}) {
    seedCell(sources[sourcesOffset + index], ${s.sourceCosts?`sourceCosts[sourceCostsOffset + index]`:`0.0`});
  }`:``}
  ${s.sourceMask?`if (index < CELL_COUNT && sourceMask[sourceMaskOffset + index] != 0u) { seedCell(index, 0.0); }`:``}`,condition:a.gate?.condition,extraResources:a.gate?.extraResources})}},Y=`
fn getEdgeCost(direction: u32, centerRow: u32, neighborFriction: f32, centerFriction: f32) -> f32 {
  return getD8Distance(direction, centerRow) * (0.5 * (neighborFriction + centerFriction));
}`,X=`
fn getStrictDirection(index: u32, centerAuxiliary: f32, centerCost: f32) -> u32 {
  var bestCandidate = getInfinity();
  var bestDirection = GRID_NONE;
  let centerRow = index / GRID_WIDTH;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE) { continue; }
    let neighborAuxiliary = auxiliary[auxiliaryOffset + neighbor];
    let neighborCost = costs[costsOffset + neighbor];
    if (isNaNValue(neighborAuxiliary) || !(neighborCost < centerCost)) { continue; }
    let candidate = neighborCost + getEdgeCost(direction, centerRow, neighborAuxiliary, centerAuxiliary);
    if (candidate < bestCandidate) {
      bestCandidate = candidate;
      bestDirection = direction;
    }
  }
  // 4 ulp of f32 (2^-21) relative: see the derivation in the TSDoc of STRICT_PREDECESSOR_WGSL.
  if (bestDirection != GRID_NONE && bestCandidate <= centerCost * (1.0 + 4.76837158e-7)) { return bestDirection; }
  return GRID_NONE;
}`;function Z(e,t=48){let{world:n}=e,r=I(n,t),{columns:i,cellSize:a}=r,o=i*t,s=new Float32Array(o),c=new Uint8Array(o),l=new Float32Array(o*2),u=Math.round(i*.55);for(let e=0;e<t;e++)for(let o=0;o<i;o++){let d=e*i+o,f=e/t;c[d]=+(Math.abs(o-u)<=1&&f>.04&&f<.96&&!(f>.2&&f<.3)&&!(f>.72&&f<.8)),s[d]=r.values[d]**2,l[d*2]=(o+.5)*a,l[d*2+1]=n.height-(e+.5)*a}let d=Uint32Array.from([.5,.2,.8,.35].slice(0,4).map((e,n)=>Math.floor(e*t)*i+Math.floor(i*(n===0?.12:.2))));return{columns:i,rows:t,cellCount:o,cellSize:a,grid:P({width:i,height:t,cellSize:a,origin:[0,t*a]}),getFriction:e=>Float32Array.from(s,(t,n)=>c[n]?NaN:1+e*t),sources:d,defaultSources:Uint32Array.from(d),isWallCell:c,bounds:[0,0,i*a,t*a],cellCenters:l}}function Q(e,t,n){return L(e.columns,e.rows,e.bounds,t,n)}function $(e,t){let{grid:n,cellCount:r}=t,i=e.upload(`friction`,`float32`,t.getFriction(7)),a=e.allocate(`friction-parameters`,`float32`,2),o=e.allocate(`parameters`,`float32`,8),s=e.upload(`sources`,`uint32`,t.sources),c=e.allocate(`source-count`,`uint32`,1),l=e.allocate(`costs`,`float32`,r),u=e.allocate(`back-links`,`uint32`,r);e.graph.add(new J({grid:n,friction:F(i.view,n,{id:`friction`}),frictionParameters:a.view,parameters:o.view,sources:s.view,sourceCount:c.view,output:{costs:l.view,backLinks:u.view,status:{converged:e.allocate(`converged`,`uint32`,1).view}}}));let d=e.allocate(`source-positions`,`float32x2`,4);return{friction:i,costs:l,backLinks:u,sources:d,writeSettings:e=>{s.write(t.sources),d.write(Float32Array.from({length:8},(e,n)=>{let r=t.sources[n>>1];return t.cellCenters[r*2+(n&1)]})),e.contrast!==void 0&&i.write(t.getFriction(e.contrast)),a.write(Float32Array.of(e.frictionScale??1,0)),o.write(q({costLimit:e.costLimit??1/0},n)),c.write(Uint32Array.of(Math.round(e.sourceCount??1)))}}}export{K as a,G as i,Z as n,Q as r,$ as t};
import{r as e}from"./kernel-C3tfQOn4.js";import{a as t,c as n,f as r,i,s as a,t as o}from"./raster-relaxation-Di9wjtxp.js";import{o as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{o as c,u as l}from"./wgsl-common-DfyXiYB0.js";import{a as u,i as d,o as f,r as p,t as m}from"./terrain-flow-tracing-BfyLX2kv.js";import{r as h}from"./gpu-terrain-flow-BhpjXagd.js";import{D as g,E as _,T as v,f as y,p as b,v as x,x as S,y as C}from"./gpu-contributor-utils-BIwLQON5.js";import{c as w,l as T,n as E,r as D,u as O}from"./raster-band-utils-BcTxc8_Y.js";import{t as k}from"./gpu-raster-grid-C_izEAl0.js";import{t as A}from"./define-run-B-brkCe4.js";import{o as j}from"./pointer-overlay-Qh8qflcA.js";import{et as M}from"./entry-preview-D70TdSbK.js";import{d as N,o as P,r as F}from"./terrain-Bwc0vRX3.js";var I=`GPUTerrainHeightAboveDrainage`;function L(o,u){let{id:d,grid:f,output:m,maximumIterations:g}=u,_=f.width*f.height,v=e(o,`${d}-current`,`uint32`,_*3),y=e(o,`${d}-next`,`uint32`,_*3),b=n(o,`${d}-average`,I,_),x=`${p(f.width,f.height)}
${c}
${l}
const STATE_WORDS: u32 = 3u;
const WAITING: u32 = 0u;
const RESOLVED: u32 = 1u;
const NO_DRAINAGE: u32 = 2u;
const UNKNOWN: u32 = 3u;
const MULTIPLE: u32 = ${B}u;
const UNRESOLVED: u32 = ${z}u;`,S={name:`elevation`,view:u.elevation,type:`f32`,access:`read`},C={name:`fractions`,view:u.fractions,type:`f32`,access:`read`},w={name:`current`,view:v,type:`u32`,access:`read`},T={name:`next`,view:y,type:`u32`,access:`read_write`},E=[s(o,{id:`${d}-average-init`,operation:I,invocationCount:_,declarations:x,bindings:[S,C,{name:`streams`,view:u.streams,type:`u32`,access:`read`},{...w,access:`read_write`}],body:`let base = currentOffset + index * STATE_WORDS;
var state = WAITING;
var drainage = UNRESOLVED;
var value = getQuietNaN();
let stream = streams[streamsOffset + index];
if (!isFiniteValue(elevation[elevationOffset + index])) {
  state = NO_DRAINAGE; drainage = NO_CELL;
} else if (stream == ${h}u) {
  state = UNKNOWN;
} else if (stream != 0u) {
  state = RESOLVED; drainage = index; value = 0.0;
} else {
  var valid = true;
  var total = 0.0;
  for (var direction = 0u; direction < 8u; direction++) {
    let fraction = fractions[fractionsOffset + index * 8u + direction];
    valid = valid && isFiniteValue(fraction) && fraction >= 0.0;
    if (fraction > 0.0 && getFlowReceiver(index, 1u << direction) == NO_CELL) { valid = false; }
    total += fraction;
  }
  if (!valid || (total != 0.0 && abs(total - 1.0) > 0.00001)) {
    state = UNKNOWN;
  } else if (total == 0.0) {
    state = NO_DRAINAGE; drainage = NO_CELL;
  }
}
current[base] = state;
current[base + 1u] = bitcast<u32>(value);
current[base + 2u] = drainage;`}),a(o,{id:`${d}-average-reset`,operation:I,state:b})];for(let e=0;e<g;e++){let n=`${d}-average-round-${e}`,i=r(b,n);E.push(s(o,{id:n,operation:I,variant:`fractional-hand`,invocationCount:_,declarations:x,...i,bindings:[w,T,S,C,{name:`status`,view:b.status,type:`atomic<u32>`,access:`read_write`}],body:`let base = currentOffset + index * STATE_WORDS;
let nextBase = nextOffset + index * STATE_WORDS;
var state = current[base];
var value = bitcast<f32>(current[base + 1u]);
var drainage = current[base + 2u];
if (state == WAITING) {
  var waiting = false;
  var missing = false;
  var unknown = false;
  var total = 0.0;
  var average = 0.0;
  var firstDrainage = NO_CELL;
  for (var direction = 0u; direction < 8u; direction++) {
    let fraction = fractions[fractionsOffset + index * 8u + direction];
    if (fraction <= 0.0) { continue; }
    let neighbor = getFlowReceiver(index, 1u << direction);
    let neighborBase = currentOffset + neighbor * STATE_WORDS;
    let neighborState = current[neighborBase];
    waiting = waiting || neighborState == WAITING;
    missing = missing || neighborState == NO_DRAINAGE;
    unknown = unknown || neighborState == UNKNOWN;
    if (neighborState != RESOLVED) { continue; }
    let drop = elevation[elevationOffset + index] - elevation[elevationOffset + neighbor];
    average += fraction * (drop + bitcast<f32>(current[neighborBase + 1u]));
    total += fraction;
    let neighborDrainage = current[neighborBase + 2u];
    if (firstDrainage == NO_CELL) { firstDrainage = neighborDrainage; }
    else if (firstDrainage != neighborDrainage) { firstDrainage = MULTIPLE; }
  }
  // Every positive branch must reach a stream, matching TauDEM's contamination check.
  if (waiting) { /* Downstream dependencies must finish before classifying this row. */ }
  else if (missing) { state = NO_DRAINAGE; drainage = NO_CELL; }
  else if (unknown) { state = UNKNOWN; }
  else {
    value = average / total;
    if (isFiniteValue(value)) { state = RESOLVED; drainage = firstDrainage; }
    else { state = NO_DRAINAGE; drainage = NO_CELL; value = getQuietNaN(); }
  }
  // A cycle must consume its budget even when no readiness word changes.
  atomicOr(&status[statusOffset], 1u);
}
next[nextBase] = state;
next[nextBase + 1u] = bitcast<u32>(value);
next[nextBase + 2u] = drainage;`}),s(o,{id:`${d}-average-copy-${e}`,operation:I,variant:`fractional-hand-copy`,invocationCount:_,...r(b,`${n}-copy`),bindings:[{...T,access:`read`},{...w,access:`read_write`}],body:`for (var word = 0u; word < 3u; word++) {
  current[currentOffset + index * 3u + word] = next[nextOffset + index * 3u + word];
}`}),t(o,{id:`${d}-average-gate-${e}`,operation:I,state:b,maxIterations:g}))}let D=[w];return m.heightAboveDrainage&&D.push({name:`height`,view:m.heightAboveDrainage,type:`f32`,access:`read_write`}),m.drainageCells&&D.push({name:`drainage`,view:m.drainageCells,type:`u32`,access:`read_write`}),E.push(s(o,{id:`${d}-average-publish`,operation:I,invocationCount:_,bindings:D,body:`let base = currentOffset + index * 3u;
${m.heightAboveDrainage?`height[heightOffset + index] = bitcast<f32>(current[base + 1u]);`:``}
${m.drainageCells?`drainage[drainageOffset + index] = current[base + 2u];`:``}`})),(m.status?.converged||m.status?.iterationLimitReached||m.status?.iterationCount)&&E.push(i(o,{id:`${d}-average-finalize`,operation:I,state:b,...m.status})),E}var R=`GPUTerrainHeightAboveDrainage`,z=4294967294,B=4294967293,V=class{id;props;estimatedCommandNodeCount;constructor(e){this.id=e.id??`terrain-height-above-drainage`,this.props=e;let{id:t}=this,{output:n}=e,r=O(t,e.grid),i=e.maximumIterations??(e.flowFractions?128:32);y(t,`maximumIterations`,i,1,o),this.estimatedCommandNodeCount=e.flowFractions?5+3*i:7+i+Math.ceil(i/4),b(t,this.estimatedCommandNodeCount),v(t,!!e.flowDirections!=!!e.flowFractions,`flowDirections and flowFractions require exactly one routing input`),v(t,!e.flowFractions||e.directionEncoding===void 0,`directionEncoding requires flowDirections`),d(t,e.directionEncoding),v(t,!!(n.heightAboveDrainage||n.drainageCells),`output requires at least one output view`),e.flowDirections&&u(t,`flowDirections`,e.flowDirections,`uint32`,r),e.flowFractions&&(v(t,e.flowFractions.format===`float32`,`flowFractions must use float32`),_(e.flowFractions,[`float32`],`${t} flowFractions`),g(t,`flowFractions`,e.flowFractions,r*8)),u(t,`streams`,e.streams,`uint32`,r),n.heightAboveDrainage&&u(t,`output.heightAboveDrainage`,n.heightAboveDrainage,`float32`,r),n.drainageCells&&u(t,`output.drainageCells`,n.drainageCells,`uint32`,r),v(t,e.elevation.storage.kind!==`buffer`||e.elevation.storage.values.length===r,`elevation must contain one value per cell`),f(t,n.status);let a=[n.heightAboveDrainage,n.drainageCells,n.status?.converged,n.status?.iterationLimitReached,n.status?.iterationCount];C(t,a,[...E(e.elevation),e.flowDirections,e.flowFractions,e.streams]),T(t,a,[])}getCommandNodes(t){let{id:n,props:r}=this,{output:i}=r,{width:a,height:o}=r.grid,u=a*o;x(t.device,n,`HAND scratch`,u*(r.flowFractions?12:4),`reduce the grid dimensions`),r.flowFractions&&x(t.device,n,`flowFractions`,u*32,`reduce the grid dimensions`),w(n,t,r.elevation,[]),S(n,t,[r.flowDirections,r.flowFractions,r.streams,i.heightAboveDrainage,i.drainageCells,i.status?.converged,i.status?.iterationLimitReached,i.status?.iterationCount]);let d=[],f;if(i.heightAboveDrainage||r.flowFractions){let e=D(t,n,r.elevation,r.grid,!0);d.push(...e.nodes),f=e.band.storage.values}if(r.flowFractions&&f)return[...d,...L(t,{id:n,grid:r.grid,elevation:f,fractions:r.flowFractions,streams:r.streams,output:i,maximumIterations:r.maximumIterations??128})];let g=r.flowDirections,_=e(t,`${n}-pointer`,`uint32`,u);d.push(...m(t,{id:n,contributorId:n,operation:R,width:a,height:o,maxIterations:r.maximumIterations??32,flowDirections:g,directionEncoding:r.directionEncoding,pointer:_,stops:{kind:`streams`,streams:r.streams},converged:i.status?.converged,iterationLimitReached:i.status?.iterationLimitReached,iterationCount:i.status?.iterationCount}));let v=[{name:`flowDirections`,view:g,type:`u32`,access:`read`},{name:`streams`,view:r.streams,type:`u32`,access:`read`},{name:`pointer`,view:_,type:`u32`,access:`read`}];return i.drainageCells&&v.push({name:`drainageOut`,view:i.drainageCells,type:`u32`,access:`read_write`}),f&&i.heightAboveDrainage&&v.push({name:`elevation`,view:f,type:`f32`,access:`read`},{name:`heightOut`,view:i.heightAboveDrainage,type:`f32`,access:`read_write`}),d.push(s(t,{id:`${n}-height`,operation:R,variant:`height-above-drainage`,bindings:v,invocationCount:u,declarations:`${p(a,o)}
${c}
${l}`,body:`let reached = pointer[pointerOffset + index];
  var drainage = NO_CELL;
  if (flowDirections[flowDirectionsOffset + index] != NO_CELL &&
      flowDirections[flowDirectionsOffset + reached] != NO_CELL) {
    if (pointer[pointerOffset + reached] != reached) {
      // Pointer jumping stopped early: 'reached' is an intermediate cell, not the first stream.
      drainage = ${z}u;
    } else if (streams[streamsOffset + reached] != 0u) {
      // An unresolved stream word stops the path too: the first stream may be that cell or lie
      // further down, so the drainage cell is unknown rather than wrong.
      drainage = select(reached, ${z}u,
        streams[streamsOffset + reached] == ${h}u);
    } else if (flowDirections[flowDirectionsOffset + reached] == UNRESOLVED_DIRECTION ||
        flowDirections[flowDirectionsOffset + index] == UNRESOLVED_DIRECTION) {
      // The path ends on a cell whose direction is unknown, not on an outlet or a sink.
      drainage = ${z}u;
    }
  }
  ${f?`var result = getQuietNaN();
  if (drainage < ${z}u) {
    let cellElevation = elevation[elevationOffset + index];
    let drainageElevation = elevation[elevationOffset + drainage];
    if (isFiniteValue(cellElevation) && isFiniteValue(drainageElevation)) {
      result = cellElevation - drainageElevation;
    }
  }
  heightOut[heightOutOffset + index] = result;`:``}
  ${i.drainageCells?`drainageOut[drainageOutOffset + index] = drainage;`:``}`})),d}},H=.06,U=.012,W=[[0,0,0,0],[.25,.55,1,.8]];function G(e){return j((1-e-.05)/.9,0,1)*H}var K=A({parameters:M,caption:`Height above the nearest stream: low cells near streams light up.`,createInputs:e=>({...P(e),stage:U}),build(e,t){let n=F(e,t),r=e.allocate(`hand`,`float32`,t.cellCount);e.graph.add(new V({grid:t.grid,elevation:k(n.filledElevation.view,t.grid,{id:`filled`}),flowDirections:n.flowDirections.view,streams:n.streams.view,output:{heightAboveDrainage:r.view}}));let i=e.allocate(`stage`,`float32`,1),a=e.allocate(`water`,`uint32`,t.cellCount);return e.graph.add({getCommandNodes:e=>[s(e,{id:`flood-stage`,operation:`flood-stage`,bindings:[{name:`hand`,view:r.view,type:`f32`,access:`read`},{name:`stage`,view:i.view,type:`f32`,access:`read`},{name:`water`,view:a.view,type:`u32`,access:`read_write`}],invocationCount:t.cellCount,body:`water[waterOffset + index] = select(0u, 1u, hand[handOffset + index] <= stage[stageOffset]);`})]}),{outputs:{filledElevation:n.filledElevation,heightAboveDrainage:r,water:a},update:e=>{n.writeStreamThreshold(e.streamThreshold),i.write(Float32Array.of(t.stage))}}},interaction:{kind:`level`,hint:`raise the flood`,rest:[.5,.77],resolve(e,{inputs:t,outputs:n,world:r,peek:i}){let a=Math.round(G(e.unitY)*1e3)/1e3,o=t.stage!==a;t.stage=a;let s=i(n.heightAboveDrainage),c=0;if(s)for(let e of s)e<=a&&c++;return{rerun:o,glyphs:[{shape:`line`,from:[0,e.y],to:[r.width,e.y],dashed:!0}],readout:`stage ${a.toFixed(3)}${s?` · flooded ${Math.round(100*c/s.length)}%`:``}`}}},draw:({inputs:e,outputs:t})=>[N(e,t.filledElevation),{layer:`raster`,values:t.heightAboveDrainage,columns:e.columns,rows:e.rows,ramp:`sequential`,range:[0,.06],alpha:.85,smoothing:!0},{layer:`raster`,values:t.water,columns:e.columns,rows:e.rows,categories:W}]});export{K as default};
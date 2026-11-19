import{n as e,r as t}from"./trajectory-binding-validation-C89oJIAZ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{a as r,m as i,v as a}from"./spatial-context-DwZEd4EZ.js";import{D as o,E as s,T as c,v as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{o as p}from"./trajectory-common-DFp8lX7h.js";import{d as m,f as h,u as g}from"./trajectory-metrics-kernels-Dj0TkZQS.js";import{o as _,r as v}from"./movement-common-nOvANN43.js";import{t as y}from"./define-run-B-brkCe4.js";import{o as b,s as x}from"./pointer-overlay-Qh8qflcA.js";import{i as S,l as C,o as w,r as T,s as E}from"./movement-DZVB3aIP.js";import{r as D}from"./units-Boef4ppn.js";import{f as O}from"./entry-preview-D70TdSbK.js";var k=`GPUTrajectoryOutlierFlags`;function A(e,t=new Float32Array(4)){let n=e.maximumAcceleration??0;for(let t of[e.maximumSpeed,n])if(!Number.isFinite(Math.fround(t))||t<0)throw Error(`Outlier limits must be finite and non-negative`);return t.set([e.maximumSpeed,n,0,0]),t}var j=class{id;props;spatialContext;constructor(e){this.id=e.id??`trajectory-outlier-flags`,this.props=e;let n=this.id;this.spatialContext=a(n,e.spatialContext,r);let{positions:i,timestamps:l,trackOffsets:u,parameters:f,output:m}=e;s(i,[`float32x2`],`${n} positions`),s(l,[`float32`,`uint32x2`],`${n} timestamps`),s(u,[`uint32`],`${n} trackOffsets`),s(f,[`float32`],`${n} parameters`),o(n,`timestamps`,l,i.length),p(n,l,e.temporalContext),c(n,u.length>=2,`trackOffsets must contain at least two rows`),c(n,f.length>=4,`parameters must hold 4 float32 values`),s(m.flags,[`uint32`],`${n} output.flags`),o(n,`output.flags`,m.flags,i.length),m.trackFlaggedCounts&&(s(m.trackFlaggedCounts,[`uint32`],`${n} output.trackFlaggedCounts`),o(n,`output.trackFlaggedCounts`,m.trackFlaggedCounts,u.length-1)),t(this.id,this.props.output),d(n,[m.trackFlaggedCounts],[m.flags]),d(n,[m.flags,m.trackFlaggedCounts],[i,l,u,f])}getCommandNodes(t){let{id:r,props:a}=this,{positions:o,timestamps:s,trackOffsets:c,parameters:d,output:p}=a;u(r,t,[o,s,c,d,p.flags,p.trackFlaggedCounts]);let y=o.length;l(t.device,r,`positions`,y*8,`split the tracks over several graphs`),e(t,this.id,this.props);let b=m(a),x=n(t,`${r}-row-starts`,`uint32`,y),S=n(t,`${r}-row-ends`,`uint32`,y),C=n(t,`${r}-row-tracks`,`uint32`,y),w=i(this.spatialContext),T=[...v(t,{id:`${r}-row-bounds`,operation:k,trackOffsets:c,rowCount:y,rowStarts:x,rowEnds:S,rowTracks:C}),f(t,{id:`${r}-flags`,operation:k,variant:b,invocationCount:y,bindings:[{name:`positions`,view:o,type:`f32`,access:`read`},...g(a),{name:`rowStarts`,view:x,type:`u32`,access:`read`},{name:`rowEnds`,view:S,type:`u32`,access:`read`},{name:`parameters`,view:d,type:`f32`,access:`read`},{name:`flags`,view:p.flags,type:`u32`,access:`read_write`}],declarations:`${h(b)}\n${_(w)}
fn stepSpeed(row: u32) -> f32 {
  let elapsed = timeDifferenceRows(row, row - 1u);
  return select(0.0, rowDistance(row, row - 1u) / elapsed, elapsed > 0.0);
}
fn isExcessiveStep(row: u32, limit: f32) -> bool {
  let elapsed = timeDifferenceRows(row, row - 1u);
  return limit > 0.0 && elapsed > 0.0 && stepSpeed(row) > limit;
}
// Signed acceleration of the step ending at row: positive when the step is faster than the one
// before it. 0 where no previous step exists in the track or no time has passed.
fn stepAcceleration(row: u32, begin: u32) -> f32 {
  if (row < begin + 2u) { return 0.0; }
  let elapsed = timeDifferenceRows(row, row - 1u);
  if (elapsed <= 0.0) { return 0.0; }
  return (stepSpeed(row) - stepSpeed(row - 1u)) / elapsed;
}
// A displaced fix has two fast steps of about equal speed, so the excessive accelerations are the
// rise into the row and the fall after the step out of it, two rows later. The acceleration between
// the two fast steps is near zero and is not tested. A side that the track end removes is waived.
fn isDisplacedByAcceleration(row: u32, begin: u32, last: u32, limit: f32) -> bool {
  if (limit <= 0.0) { return false; }
  let hasRise = row >= begin + 2u;
  let hasFall = row + 2u <= last;
  if (!hasRise && !hasFall) { return false; }
  if (hasRise) {
    if (!(stepAcceleration(row, begin) > limit)) { return false; }
  } else if (row > begin) {
    // Second row: its step in has no earlier step to rise from. A rise on its step out means the
    // step in was the slow one, which is a jump after this row, not a displaced fix.
    if (stepAcceleration(row + 1u, begin) > limit) { return false; }
  }
  if (hasFall) {
    if (!(stepAcceleration(row + 2u, begin) < -limit)) { return false; }
  } else if (row < last) {
    // Second-to-last row: a fall on its step out means the fix stayed where the step in took it.
    if (stepAcceleration(row + 1u, begin) < -limit) { return false; }
  }
  return true;
}`,body:`let begin = rowStarts[rowStartsOffset + index];
  let end = rowEnds[rowEndsOffset + index];
  var state = 0u;
  if (begin != 0xffffffffu && end - begin >= 2u) {
    let speedLimit = parameters[parametersOffset];
    let accelerationLimit = parameters[parametersOffset + 1u];
    let hasIncoming = index > begin;
    let hasOutgoing = index + 1u < end;
    let incoming = hasIncoming && isExcessiveStep(index, speedLimit);
    let outgoing = hasOutgoing && isExcessiveStep(index + 1u, speedLimit);
    var isOutlier = false;
    if (hasIncoming && hasOutgoing) { isOutlier = incoming && outgoing; }
    else if (hasOutgoing) { isOutlier = outgoing && !(index + 2u < end && isExcessiveStep(index + 2u, speedLimit)); }
    else if (hasIncoming) { isOutlier = incoming && !(index >= begin + 2u && isExcessiveStep(index - 1u, speedLimit)); }
    if (isOutlier) { state |= 1u; }
    if (isDisplacedByAcceleration(index, begin, end - 1u, accelerationLimit)) { state |= 2u; }
  }
  flags[flagsOffset + index] = state;`})];return p.trackFlaggedCounts&&T.push(f(t,{id:`${r}-counts`,operation:k,invocationCount:c.length-1,bindings:[{name:`trackOffsets`,view:c,type:`u32`,access:`read`},{name:`flags`,view:p.flags,type:`u32`,access:`read`},{name:`counts`,view:p.trackFlaggedCounts,type:`u32`,access:`read_write`}],body:`var total = 0u;
  for (var row = min(trackOffsets[trackOffsetsOffset + index], ${y}u); row < min(trackOffsets[trackOffsetsOffset + index + 1u], ${y}u); row++) {
    total += select(0u, 1u, flags[flagsOffset + row] != 0u);
  }
  counts[countsOffset + index] = total;`})),T}},M=10,N=10,P=8,F=y({parameters:O,caption:`Ten Bay vessel tracks with 10 synthetic GPS jumps (a fix moved 0.5 of the frame). Colored fixes are flagged as displaced: both steps around them are faster than maximumSpeed, or (maximumAcceleration above 0) the speed rises into the fix and falls out of it by more than that limit. Distances and speeds are approximate: the frame stretches x and y differently.`,createInputs:({world:e})=>{let t=T(e,M),n=Float32Array.from(t.positions),r=[];for(let e=0;e<t.trackCount;e++)for(let n=t.trackOffsets[e]+2;n<t.trackOffsets[e+1]-2;n++){let e=t.timestamps[n]-t.timestamps[n-1],i=t.timestamps[n+1]-t.timestamps[n];Math.max(e,i)<=P&&r.push(n)}let i=Array.from({length:N},(e,t)=>r[Math.floor((t+.5)*r.length/N)]),a=(r,i,a)=>{[e.width,1].forEach((e,o)=>{let s=t.positions[r*2+o],c=(o===0?Math.cos(i):Math.sin(i))*a;(s+c<0||s+c>e)&&(c=-c),n[r*2+o]=b(s+c,0,e)})},o=e=>{let r=w({...t,positions:n});return Math.min(r[e],r[e+1])},s=x(O,`maximumSpeed`).default;i.forEach((e,t)=>{let n=t*2.399963;if(t>0)return a(e,n,.5);for(let t=.1;t<=.6&&(a(e,n,t),!(o(e)>=s+.002));t+=.01);});let c={...t,positions:n},l=w(c),u=i[0];return{...c,segments:S(n,t.trackOffsets),speeds:l,restFix:u}},build(e,t){let n=e.upload(`positions`,`float32x2`,t.positions),r=e.allocate(`parameters`,`float32`,4),i=e.allocate(`flags`,`uint32`,t.rowCount);return e.graph.add(new j({positions:n.view,timestamps:e.upload(`timestamps`,`float32`,t.timestamps).view,trackOffsets:e.upload(`offsets`,`uint32`,t.trackOffsets).view,parameters:r.view,output:{flags:i.view}})),{outputs:{segments:e.upload(`segments`,`float32x4`,t.segments),positions:n,flags:i},update:e=>r.write(A({maximumSpeed:e.maximumSpeed,maximumAcceleration:e.maximumAcceleration}))}},async afterRun({outputs:e,inputs:t,upload:n}){let r=await C(e.flags),i=[],a=[];for(let e=0;e<r.length;e++)r[e]&&(i.push(t.positions[e*2],t.positions[e*2+1]),a.push(r[e]));return{flaggedCount:a.length,flagged:n(`flagged`,`float32x2`,Float32Array.from(a.length?i:[0,0])),kinds:n(`kinds`,`uint32`,Uint32Array.from(a.length?a:[0]))}},interaction:{kind:`level`,hint:`pick a fix`,rest:({toUnit:e,inputs:t})=>e([t.positions[t.restFix*2],t.positions[t.restFix*2+1]]),resolve(e,{inputs:t,values:n,peek:r,outputs:i}){let a=0,o=1/0;for(let n=0;n<t.rowCount;n++){let r=Math.hypot(t.positions[n*2]-e.x,t.positions[n*2+1]-e.y);r<o&&(a=n,o=r)}let s=E(t.trackOffsets,a),c=t.trackOffsets[s],l=t.trackOffsets[s+1]-1,u=a>c?t.speeds[a]:1/0,d=a<l?t.speeds[a+1]:1/0,f=x(O,`maximumSpeed`),p=b(Number((Math.floor(Math.min(u,d)/f.step)*f.step).toPrecision(12)),f.min,f.max),m=[t.positions[a*2],t.positions[a*2+1]],h=r(i.flags),g=0;for(let e=0;h&&e<h.length;e++)g+=+!!h[e];return{values:{maximumSpeed:p},glyphs:[...a>c?[{shape:`ring`,dashed:!0,at:[t.positions[a*2-2],t.positions[a*2-1]],radius:p*(t.timestamps[a]-t.timestamps[a-1])}]:[],...a<l?[{shape:`ring`,dashed:!0,at:[t.positions[a*2+2],t.positions[a*2+3]],radius:p*(t.timestamps[a+1]-t.timestamps[a])}]:[],...a>c?[{shape:`line`,from:[t.positions[a*2-2],t.positions[a*2-1]],to:m}]:[],...a<l?[{shape:`line`,from:m,to:[t.positions[a*2+2],t.positions[a*2+3]]}]:[],{shape:`crosshair`,at:m}],readout:`in ${Number.isFinite(u)?D(u,t.metersPerUnit):`-`} · out ${Number.isFinite(d)?D(d,t.metersPerUnit):`-`} · limit ${D(p,t.metersPerUnit)} · ${g} flagged`}}},draw:({inputs:e,outputs:t,extra:n})=>[{layer:`lines`,context:!0,segments:t.segments,count:e.segmentCount},{layer:`dots`,context:!0,positions:t.positions,radius:2.5},...n?[{layer:`dots`,positions:n.flagged,values:n.kinds,count:n.flaggedCount,radius:14,alpha:.85}]:[]]});export{F as default};
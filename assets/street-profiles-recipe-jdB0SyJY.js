import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-reduction-tNgOVQ_C.js";import{o as n,u as r}from"./recipe-utils-CICB7qiS.js";import{u as i}from"./wgsl-common-DfyXiYB0.js";import{T as a}from"./gpu-contributor-utils-BIwLQON5.js";import{c as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as s,i as c,n as l,r as u,t as d}from"./urban-ray-utils-BSeNtKp8.js";import{t as f}from"./define-run-B-brkCe4.js";import{n as p,t as m}from"./pick-Bo3t3hjv.js";import{n as h,t as g}from"./kernel-BXyKsp4B.js";import{a as _,r as v,t as y}from"./urban-D7A1ccno.js";import{Ao as b}from"./entry-preview-D70TdSbK.js";var x=1024,S=8,C=1e-5;function w(e,t,n,r){let i=`let blocked=getSegmentCandidate(origin,direction,${o(t)},other);
  if(blocked.y>0.0){nearest=min(nearest,blocked.x);if(other<row){nearestLowerRow=min(nearestLowerRow,blocked.x);}}`,a=`let reach=nearest+slack;
 return facadeDistance<=reach&&nearestLowerRow>reach;`;return r?`
fn isFacadeIntervalVisible(origin:vec2<f32>,direction:vec2<f32>,facadeDistance:f32,slack:f32,row:u32)->bool{
 var nearest=facadeDistance;var nearestLowerRow=3.0e38;
 var stack:array<u32,64>;var depth=1u;stack[0u]=0u;
 loop{
  if(depth==0u){break;}
  depth--;let node=stack[depth];
  // Only a hit within slack of the nearest one can change the result. Rounding slack on the
  // pruning distance keeps every such segment in the search.
  let pruning=nearest+slack;
  let limit=pruning+pruning*1.0e-5+${o(4*n)};
  if(getBoxEntry(node,origin,direction,limit)>=3.0e38){continue;}
  if(node<${r.internalNodeCount}u){stack[depth]=node*2u+2u;stack[depth+1u]=node*2u+1u;depth+=2u;continue;}
  let other=leafIds[leafIdsOffset+node-${r.internalNodeCount}u];
  if(other>=${e}u||other==row){continue;}
  ${i}
 }
 ${a}
}`:`
fn isFacadeIntervalVisible(origin:vec2<f32>,direction:vec2<f32>,facadeDistance:f32,slack:f32,row:u32)->bool{
 var nearest=facadeDistance;var nearestLowerRow=3.0e38;
 for(var other=0u;other<${e}u;other++){
  if(other==row){continue;}
  ${i}
 }
 ${a}
}`}function T(f,p){let m=p.id??`street-profiles`,h=p.positions.length,g=p.segments.length,_=p.buildingHeights.length,v=p.tolerance??1e-6;a(m,h>0,`positions must have at least 1 row`),a(m,p.normals.length===h,`normals must have one row per position`),a(m,g>0&&g<=16777216,`segments must have between 1 and ${d} rows`),a(m,p.segmentBuildingIds.length===g,`segmentBuildingIds must have one row per segment`),a(m,_>0,`buildingHeights must have at least 1 row`),a(m,Number.isFinite(p.maximumDistance)&&p.maximumDistance>0,`maximumDistance must be a positive finite number`),a(m,Number.isFinite(p.sampleSpacing)&&p.sampleSpacing>0,`sampleSpacing must be a positive finite number`),a(m,Number.isFinite(v)&&v>0,`tolerance must be a positive finite number`),p.streetSegment&&(a(m,p.streetSegment.length===1,`streetSegment must have 1 row`),a(m,g<=x,`segments must have at most ${x} rows when streetSegment is given`));let y=p.output,b=r(f,`${m}-profiles`,`float32x4`,h,y?.profiles),T=r(f,`${m}-buildings`,`uint32x2`,h,y?.buildings),E=r(f,`${m}-validity`,`uint32`,h,y?.validity),D=r(f,`${m}-canyon`,`float32x4`,h,y?.canyon),O=r(f,`${m}-frontage`,`float32x2`,_,y?.frontage),k=e(f,`${m}-non-finite-flags`,`uint32`,g),A=e(f,`${m}-bad-owner-flags`,`uint32`,g),j=e(f,`${m}-non-finite-count`,`uint32`,1),M=e(f,`${m}-bad-owner-count`,`uint32`,1);f.add(n(f,{id:`${m}-segment-flags`,operation:m,invocationCount:g,declarations:i,bindings:[{name:`segments`,view:p.segments,type:`f32`,access:`read`},{name:`owners`,view:p.segmentBuildingIds,type:`u32`,access:`read`},{name:`nonFiniteFlags`,view:k,type:`u32`,access:`read_write`},{name:`badOwnerFlags`,view:A,type:`u32`,access:`read_write`}],body:`let base=segmentsOffset+4u*index;
let finiteSegment=isFiniteValue(segments[base])&&isFiniteValue(segments[base+1u])&&isFiniteValue(segments[base+2u])&&isFiniteValue(segments[base+3u]);
nonFiniteFlags[nonFiniteFlagsOffset+index]=u32(!finiteSegment);
badOwnerFlags[badOwnerFlagsOffset+index]=u32(owners[ownersOffset+index]>=${_}u);`})),f.add(new t({id:`${m}-non-finite-segments`,input:k,output:j,operation:`sum`}).getCommandNodes(f)),f.add(new t({id:`${m}-bad-owner-segments`,input:A,output:M,operation:`sum`}).getCommandNodes(f));let N=s(p.obstacleIndex,g)===`bvh`?l(f,{id:`${m}-obstacles`,operation:m,segments:p.segments,tolerance:v}):void 0,P=i+c(g,v,N),F=N?u(N):[],I=e(f,`${m}-ray-hits`,`float32x4`,h);f.add(n(f,{id:`${m}-rays`,operation:m,invocationCount:h,declarations:P,bindings:[{name:`positions`,view:p.positions,type:`f32`,access:`read`},{name:`normals`,view:p.normals,type:`f32`,access:`read`},{name:`segments`,view:p.segments,type:`f32`,access:`read`},...F,{name:`rayHits`,view:I,type:`f32`,access:`read_write`},{name:`validity`,view:E,type:`u32`,access:`read_write`}],body:`let origin=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
let normal=vec2<f32>(normals[normalsOffset+2u*index],normals[normalsOffset+2u*index+1u]);
let inputValid=isFiniteValue(origin.x)&&isFiniteValue(origin.y)&&isFiniteValue(normal.x)&&isFiniteValue(normal.y)&&isFiniteValue(length(normal))&&length(normal)>0.0;
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
for(var side=0u;side<2u;side++){
  var hit=vec2<f32>(missing,-1.0);
  if(inputValid){hit=castRay(origin,normalize(normal)*select(1.0,-1.0,side==1u),${p.maximumDistance});}
  rayHits[rayHitsOffset+4u*index+side]=hit.x;
  rayHits[rayHitsOffset+4u*index+2u+side]=hit.y;
}
validity[validityOffset+index]=select(8u,0u,inputValid);`})),f.add(n(f,{id:`${m}-profile-flags`,operation:m,invocationCount:h,declarations:i,bindings:[{name:`nonFiniteCount`,view:j,type:`u32`,access:`read`},{name:`badOwnerCount`,view:M,type:`u32`,access:`read`},{name:`owners`,view:p.segmentBuildingIds,type:`u32`,access:`read`},{name:`heights`,view:p.buildingHeights,type:`f32`,access:`read`},{name:`rayHits`,view:I,type:`f32`,access:`read`},{name:`profiles`,view:b,type:`f32`,access:`read_write`},{name:`buildings`,view:T,type:`u32`,access:`read_write`},{name:`validity`,view:E,type:`u32`,access:`read_write`}],body:`let valid=validity[validityOffset+index]==0u&&nonFiniteCount[nonFiniteCountOffset]==0u&&badOwnerCount[badOwnerCountOffset]==0u;
var flags=0u;
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
for(var side=0u;side<2u;side++){
  var distance=missing;
  var height=missing;
  var owner=0xffffffffu;
  if(valid){
    distance=rayHits[rayHitsOffset+4u*index+side];
    let hitRow=rayHits[rayHitsOffset+4u*index+2u+side];
    if(hitRow>=0.0){
      owner=owners[ownersOffset+u32(hitRow)];
      height=heights[heightsOffset+owner];
      if(!isFiniteValue(height)||height<0.0){height=missing;}
      flags|=1u<<side;
    }
    if(distance<=${v}){flags|=4u;}
  }
  profiles[profilesOffset+4u*index+side]=distance;
  profiles[profilesOffset+4u*index+side+2u]=height;
  buildings[buildingsOffset+2u*index+side]=owner;
}
if(!valid){flags|=8u;}
validity[validityOffset+index]=flags;`})),f.add(n(f,{id:`${m}-canyon`,operation:m,invocationCount:h,declarations:i,bindings:[{name:`profiles`,view:b,type:`f32`,access:`read`},{name:`validity`,view:E,type:`u32`,access:`read`},{name:`output`,view:D,type:`f32`,access:`read_write`}],body:`let base=profilesOffset+4u*index;
let flags=validity[validityOffset+index];
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
let complete=flags==3u;
let width=profiles[base]+profiles[base+1u];
let meanHeight=0.5*(profiles[base+2u]+profiles[base+3u]);
output[outputOffset+4u*index]=select(missing,width,complete);
output[outputOffset+4u*index+1u]=select(missing,meanHeight,complete);
output[outputOffset+4u*index+2u]=select(missing,meanHeight/width,complete&&width>0.0&&isFiniteValue(meanHeight));
output[outputOffset+4u*index+3u]=select(missing,width,(flags&12u)==0u);`})),f.add(n(f,{id:`${m}-frontage`,operation:m,invocationCount:_,bindings:[{name:`buildings`,view:T,type:`u32`,access:`read`},{name:`validity`,view:E,type:`u32`,access:`read`},{name:`output`,view:O,type:`f32`,access:`read_write`}],body:`for(var side=0u;side<2u;side++){
  var length=0.0;
  for(var sample=0u;sample<${h}u;sample++){
    let flags=validity[validityOffset+sample];
    if((flags&12u)==0u&&(flags&(1u<<side))!=0u&&buildings[buildingsOffset+2u*sample+side]==index){length+=${p.sampleSpacing};}
  }
  output[outputOffset+2u*index+side]=length;
}`}));let L;if(p.streetSegment){let e=3*g+S,t=r(f,`${m}-facade-events`,`float32`,g*e),a=r(f,`${m}-facade-lengths`,`float32x2`,g,y?.facadeLengths),s=r(f,`${m}-facade-frontage`,`float32x2`,_,y?.facadeFrontage);f.add(n(f,{id:`${m}-facade-intervals`,operation:m,invocationCount:g,workgroupSize:2,declarations:`${i}${c(g,v,N,`local`)}
${w(g,p.maximumDistance,v,N)}
const EVENT_STRIDE:u32=${e}u;
fn setEvent(row:u32,slot:u32,value:f32){facadeEvents[facadeEventsOffset+row*EVENT_STRIDE+slot]=value;}
fn getEvent(row:u32,slot:u32)->f32{return facadeEvents[facadeEventsOffset+row*EVENT_STRIDE+slot];}`,bindings:[{name:`segments`,view:p.segments,type:`f32`,access:`read`},{name:`street`,view:p.streetSegment,type:`f32`,access:`read`},{name:`nonFiniteCount`,view:j,type:`u32`,access:`read`},...F,{name:`output`,view:a,type:`f32`,access:`read_write`},{name:`facadeEvents`,view:t,type:`f32`,access:`read_write`}],body:`urbanFrameOrigin=vec2<f32>(street[streetOffset],street[streetOffset+1u]);
let streetVector=vec2<f32>(street[streetOffset+2u],street[streetOffset+3u])-urbanFrameOrigin;
let streetLength=length(streetVector);
let axis=streetVector/streetLength;
let normal=vec2<f32>(-axis.y,axis.x);
// Every coordinate below is relative to the street start.
let segment=readSegment(index);
let start=segment.xy;
let end=segment.zw;
let displacement=end-start;
let facadeLength=length(displacement);
let projectedStart=dot(start,axis);
let projectedDelta=dot(displacement,axis);
let sideStart=dot(start,normal);
let sideDelta=dot(displacement,normal);
// A wall seen edge-on from the street has no visible length.
let edgeOn=abs(projectedDelta)<=${v}+${C}*facadeLength;
let slack=${v}+${C}*max(streetLength,${o(p.maximumDistance)});
setEvent(index,0u,0.0);
setEvent(index,1u,1.0);
var size=2u;
if(!edgeOn){
  setEvent(index,size,clamp(-projectedStart/projectedDelta,0.0,1.0));size++;
  setEvent(index,size,clamp((streetLength-projectedStart)/projectedDelta,0.0,1.0));size++;
}
if(abs(sideDelta)>${v}){
  for(var boundary=0u;boundary<3u;boundary++){
    setEvent(index,size,clamp((f32(i32(boundary)-1)*${p.maximumDistance}-sideStart)/sideDelta,0.0,1.0));size++;
  }
}
for(var other=0u;other<${g}u;other++){
  let obstacle=readSegment(other);
  if(!edgeOn){
    setEvent(index,size,clamp((dot(obstacle.xy,axis)-projectedStart)/projectedDelta,0.0,1.0));size++;
    setEvent(index,size,clamp((dot(obstacle.zw,axis)-projectedStart)/projectedDelta,0.0,1.0));size++;
  }
  let otherDelta=obstacle.zw-obstacle.xy;
  let denominator=displacement.x*otherDelta.y-displacement.y*otherDelta.x;
  if(abs(denominator)>${v}){
    let difference=obstacle.xy-start;
    let crossing=(difference.x*otherDelta.y-difference.y*otherDelta.x)/denominator;
    setEvent(index,size,clamp(crossing,0.0,1.0));size++;
  }
}
for(var event=1u;event<size;event++){
  let value=getEvent(index,event);
  var insertion=event;
  loop{
    if(insertion==0u||getEvent(index,insertion-1u)<=value){break;}
    setEvent(index,insertion,getEvent(index,insertion-1u));
    insertion--;
  }
  setEvent(index,insertion,value);
}
var total=vec2<f32>(0.0);
for(var event=1u;event<size;event++){
  let middle=0.5*(getEvent(index,event)+getEvent(index,event-1u));
  let longitudinal=projectedStart+middle*projectedDelta;
  let lateral=sideStart+middle*sideDelta;
  if(edgeOn||longitudinal<0.0||longitudinal>streetLength||abs(lateral)<=${v}||abs(lateral)>${p.maximumDistance}){continue;}
  // The sightline leaves the street axis at the foot of the perpendicular through the interval
  // midpoint, so it reaches this wall at abs(lateral). The lowest row within a slack of the
  // nearest hit owns the stretch; the slack scales with the corridor, and no two distances are
  // compared for equality.
  if(isFacadeIntervalVisible(longitudinal*axis,normal*sign(lateral),abs(lateral),slack,index)){
    let side=u32(lateral<0.0);
    total[side]+=(getEvent(index,event)-getEvent(index,event-1u))*facadeLength;
  }
}
let valid=isFiniteValue(streetLength)&&streetLength>${v}&&nonFiniteCount[nonFiniteCountOffset]==0u;
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
output[outputOffset+2u*index]=select(missing,total.x,valid);
output[outputOffset+2u*index+1u]=select(missing,total.y,valid);`})),f.add(n(f,{id:`${m}-facade-summary`,operation:m,invocationCount:_,bindings:[{name:`owners`,view:p.segmentBuildingIds,type:`u32`,access:`read`},{name:`lengths`,view:a,type:`f32`,access:`read`},{name:`output`,view:s,type:`f32`,access:`read_write`}],body:`var total=vec2<f32>(0.0);
for(var segment=0u;segment<${g}u;segment++){
  if(owners[ownersOffset+segment]==index){total+=vec2<f32>(lengths[lengthsOffset+2u*segment],lengths[lengthsOffset+2u*segment+1u]);}
}
output[outputOffset+2u*index]=total.x;
output[outputOffset+2u*index+1u]=total.y;`})),L={lengths:a,frontage:s}}return{outputs:{profiles:b,buildings:T,validity:E,canyon:D,frontage:O},facades:L,provenance:{positions:p.positions,normals:p.normals,segments:p.segments,segmentBuildingIds:p.segmentBuildingIds,maximumDistance:p.maximumDistance,sampleSpacing:p.sampleSpacing,tolerance:v,obstacleIndex:N?`bvh`:`brute-force`}}}var E=40,D=f({parameters:b,caption:`Street cross-sections: distance to the facade on each side, colored by height-to-width.`,createInputs:({world:e})=>{let t=y(e,24),n=_(t,E);return{layout:t,samples:n,normals:Float32Array.from({length:80},(e,t)=>t%2),index:p(n),focus:{row:-1}}},build(e,t,n){let r=v(t.layout,1,n.streetWidth),i=e.upload(`samples`,`float32x2`,t.samples),a=e.upload(`normals`,`float32x2`,t.normals),o=e.upload(`walls`,`float32x4`,r.segments),s=e.upload(`wall-owners`,`uint32`,r.segmentBuildingIds),c=e.upload(`heights`,`float32`,t.layout.heights),l=T(e.graph,{positions:i.view,normals:a.view,segments:o.view,segmentBuildingIds:s.view,buildingHeights:c.view,maximumDistance:n.maximumDistance,sampleSpacing:t.samples[2]-t.samples[0],tolerance:5e-4}),u=e.allocate(`profile-rays`,`float32x4`,80),d=e.allocate(`canyon-ratios`,`float32`,E);g(e,{id:`profile-publish-pass`,invocationCount:E,bindings:[{name:`positions`,buffer:i,access:`read`},{name:`normals`,buffer:a,access:`read`},{name:`profiles`,buffer:h(l.outputs.profiles),access:`read`},{name:`canyon`,buffer:h(l.outputs.canyon),access:`read`},{name:`validity`,buffer:h(l.outputs.validity),access:`read`},{name:`rays`,buffer:u,access:`read_write`},{name:`ratios`,buffer:d,access:`read_write`}],body:`
  let origin = positions[positionsOffset + index];
  let normal = normals[normalsOffset + index];
  let profile = profiles[profilesOffset + index];
  let isFinitePair = validity[validityOffset + index] == 3u;
  rays[raysOffset + 2u * index] = select(vec4f(-9.0), vec4f(origin, origin + normal * profile.x), isFinitePair);
  rays[raysOffset + 2u * index + 1u] = select(vec4f(-9.0), vec4f(origin, origin - normal * profile.y), isFinitePair);
  ratios[ratiosOffset + index] = select(1.0e30, canyon[canyonOffset + index].z, isFinitePair);`});let f=e.allocate(`wall-segments`,`float32x4`,r.segments.length/4);return{outputs:{positions:i,rays:u,ratios:d,wallSegments:f},update:e=>{let n=v(t.layout,1,e.streetWidth);o.write(n.segments),f.write(n.segments)}}},rebuildOn:[`maximumDistance`],interaction:{kind:`probe`,hint:`hover along the street`,rest:({inputs:e,toUnit:t})=>t([e.samples[40],e.samples[41]]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=m(t.index,e.x,t.samples[1],t.focus.row);if(t.focus.row=a,a<0)return{};let o=i(n.rays),s=i(n.ratios),c=[{shape:`ring`,at:[t.samples[a*2],t.samples[a*2+1]],radius:.03},{shape:`line`,from:[t.samples[a*2],t.samples[a*2+1]-r.maximumDistance],to:[t.samples[a*2],t.samples[a*2+1]+r.maximumDistance],dashed:!0}],l=`sample ${a} · no facade in reach`;if(o&&s&&o[a*8]>-8){let e=Math.hypot(o[a*8+2]-o[a*8],o[a*8+3]-o[a*8+1]),t=Math.hypot(o[a*8+6]-o[a*8+4],o[a*8+7]-o[a*8+5]);for(let e of[0,1]){let t=a*8+e*4;c.push({shape:`line`,from:[o[t],o[t+1]],to:[o[t+2],o[t+3]]},{shape:`dot`,at:[o[t+2],o[t+3]]})}l=`left ${e.toFixed(2)} · right ${t.toFixed(2)} · H/W ${s[a]>1e29?`-`:s[a].toFixed(1)}`}return{focus:a,glyphs:c,readout:l}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.wallSegments,radius:3},{layer:`lines`,segments:e.rays,color:[.38,.68,1,.3],radius:1},{layer:`dots`,positions:e.positions,values:e.ratios,radius:6.5,range:[0,2]}]});export{D as default};
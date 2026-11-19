import{a as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t}from"./validate-outline-output-passes-DIcqH6Pw.js";import{n,r}from"./refinement-storage-validation-D9BhQGNW.js";import{r as i}from"./kernel-C3tfQOn4.js";import{c as a,o}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as s,f as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{i as u}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{A as d}from"./pointer-overlay-Qh8qflcA.js";import{t as f}from"./define-run-B-brkCe4.js";import{s as p,v as m,y as h}from"./polygon-set-CHja50mr.js";import{a as g,c as _,d as v,f as y}from"./pointer-support-C2rPRVFO.js";import{m as b,s as x}from"./shape-segments-DMcyF4V-.js";import{Ja as S}from"./entry-preview-D70TdSbK.js";var C=class{id;props;maximumVertices;maximumEvents;constructor(e){if(this.id=e.id??`vector-straight-skeleton`,this.props=e,r(this.id,`offsetDistance`,e.offsetDistance),r(this.id,`eventTolerance`,e.eventTolerance??1e-5,!0),this.maximumVertices=e.maximumVertices??128,this.maximumEvents=e.maximumEvents??128,c(this.id,`maximumVertices`,this.maximumVertices,3,1024),c(this.id,`maximumEvents`,this.maximumEvents,0,4096),c(this.id,`segmentCapacity`,e.segmentCapacity,1),!(Number.isFinite(e.offsetDistance)&&e.offsetDistance>=0&&Number.isFinite(e.eventTolerance??1e-5)&&(e.eventTolerance??1e-5)>0))throw Error(`Invalid skeleton event parameters`);s(e.positions,[`float32x2`],this.id);for(let t of[e.ringOffsets,e.polygonOffsets,e.featureOffsets])s(t,[`uint32`],this.id);let n=e.polygonOffsets.length-1,i=this.maximumVertices+2*this.maximumEvents;for(let[t,r,a]of[[`segments`,`float32x4`,n*e.segmentCapacity],[`sourceEdges`,`uint32x2`,n*e.segmentCapacity],[`offsetPositions`,`float32x2`,n*i],[`offsetLinks`,`uint32x2`,n*i],[`status`,`uint32x4`,n]])if(s(e.output[t],[r],this.id),e.output[t].length!==a)throw Error(`Skeleton output size mismatch`);t(this.id,e.output,[Object.keys(e.output)]),l(this.id,Object.values(e.output),[e.positions,e.ringOffsets,e.polygonOffsets,e.featureOffsets])}getCommandNodes(t){let{props:r,id:s}=this,c=this.maximumVertices+2*this.maximumEvents,l=r.polygonOffsets.length-1;e(t,s,Object.fromEntries([r.positions,r.ringOffsets,r.polygonOffsets,...Object.values(r.output)].map((e,t)=>[`binding.${t}`,e]))),n(t,s,{wavefront:(r.polygonOffsets.length-1+(Math.min(r.positions.length,(r.polygonOffsets.length-1)*this.maximumVertices)+(r.polygonOffsets.length-1)*2*this.maximumEvents)*19)*4});let d=r.positions.length<l*this.maximumVertices,f=d?r.positions.length+l*2*this.maximumEvents:l*c,p=e=>d?`${e} + (rings[ringsOffset + polygons[polygonsOffset + ${e}]] + ${e} * ${2*this.maximumEvents}u) * 19u`:`${e} * ${1+c*19}u`,m=i(t,`${s}-wavefront`,`uint32`,l+f*19);return[o(t,{id:`${s}-events`,operation:`GPUVectorStraightSkeleton`,invocationCount:l,bindings:[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`rings`,view:r.ringOffsets,type:`u32`,access:`read`},{name:`polygons`,view:r.polygonOffsets,type:`u32`,access:`read`},{name:`workspace`,view:m,type:`u32`,access:`read_write`},{name:`segments`,view:r.output.segments,type:`f32`,access:`read_write`},{name:`sources`,view:r.output.sourceEdges,type:`u32`,access:`read_write`},{name:`offsets`,view:r.output.offsetPositions,type:`f32`,access:`read_write`},{name:`status`,view:r.output.status,type:`u32`,access:`read_write`}],declarations:`const SLOTS: u32 = ${c}u;
const TOLERANCE: f32 = ${a(r.eventTolerance??1e-5)};
fn readWord(base:u32, vertex:u32, field:u32)->u32 { return workspace[workspaceOffset+base+1u+vertex*19u+field]; }
fn put(base:u32, vertex:u32, field:u32, value:u32) { workspace[workspaceOffset+base+1u+vertex*19u+field]=value; }
fn scalar(base:u32, vertex:u32, field:u32)->f32 { return bitcast<f32>(readWord(base,vertex,field)); }
fn store(base:u32, vertex:u32, field:u32, value:f32) { put(base,vertex,field,bitcast<u32>(value)); }
${u(`point`,`positions`)}
fn cross(left:vec2f,right:vec2f)->f32 { return left.x*right.y-left.y*right.x; }
fn normal(base:u32,edge:u32)->vec2f { return vec2f(scalar(base,edge,11u),scalar(base,edge,12u)); }
fn at(base:u32,vertex:u32,time:f32)->vec2f { return vec2f(scalar(base,vertex,0u),scalar(base,vertex,1u))+time*vec2f(scalar(base,vertex,2u),scalar(base,vertex,3u)); }
fn initialize(base:u32,vertex:u32,left:u32,right:u32,previous:u32,next:u32,origin:vec2f,time:f32) {
  put(base,vertex,4u,left); put(base,vertex,5u,right); put(base,vertex,6u,previous); put(base,vertex,7u,next); put(base,vertex,8u,1u);
  put(base,vertex,15u,0xffffffffu); put(base,vertex,16u,0xffffffffu); put(base,vertex,17u,0u);
  store(base,vertex,9u,origin.x); store(base,vertex,10u,origin.y);
  let first=normal(base,left); let second=normal(base,right); let determinant=cross(first,second);
  var start=origin; var velocity=vec2f(0.0);
  if (abs(determinant)>1e-7) {
    let firstConstant=scalar(base,left,13u); let secondConstant=scalar(base,right,13u);
    start=vec2f(firstConstant*second.y-first.y*secondConstant,first.x*secondConstant-firstConstant*second.x)/determinant;
    velocity=vec2f(second.y-first.y,first.x-second.x)/determinant;
  }
  else if(dot(first,second)>0.999999) { velocity=first; start=origin-time*velocity; }
  store(base,vertex,0u,start.x); store(base,vertex,1u,start.y); store(base,vertex,2u,velocity.x); store(base,vertex,3u,velocity.y);
}
fn emit(polygon:u32,base:u32,vertex:u32,end:vec2f,required:u32)->u32 {
  let start=vec2f(scalar(base,vertex,9u),scalar(base,vertex,10u));
  if (distance(start,end)<=TOLERANCE) { return required; }
  if (required<${r.segmentCapacity}u) {
    let slot=polygon*${r.segmentCapacity}u+required;
    segments[segmentsOffset+slot*4u]=start.x; segments[segmentsOffset+slot*4u+1u]=start.y; segments[segmentsOffset+slot*4u+2u]=end.x; segments[segmentsOffset+slot*4u+3u]=end.y;
    sources[sourcesOffset+slot*2u]=readWord(base,readWord(base,vertex,4u),14u); sources[sourcesOffset+slot*2u+1u]=readWord(base,readWord(base,vertex,5u),14u);
  }
  return required+1u;
}
fn snapshot(polygon:u32,base:u32,count:u32,time:f32)->u32 {
  var activeCount=0u;
  for(var vertex=0u;vertex<count;vertex++) { if(readWord(base,vertex,8u)!=0u) {
    let location=at(base,vertex,time); let slot=polygon*SLOTS+vertex;
    offsets[offsetsOffset+slot*2u]=location.x; offsets[offsetsOffset+slot*2u+1u]=location.y;
    put(base,vertex,15u,readWord(base,vertex,6u)); put(base,vertex,16u,readWord(base,vertex,7u)); activeCount++;
  } }
  return activeCount;
}`,body:`let base=${p(`index`)};
var count=0u; var flags=0u; var required=0u;
let firstRing=polygons[polygonsOffset+index]; let lastRing=polygons[polygonsOffset+index+1u];
for(var ring=firstRing;ring<lastRing;ring++) {
  let first=rings[ringsOffset+ring]; var last=rings[ringsOffset+ring+1u];
  if(last>first && all(point(first)==point(last-1u))) { last--; }
  if(last-first<3u) { flags|=1u; continue; }
  var area=0.0; for(var vertex=first;vertex<last;vertex++) { let next=select(vertex+1u,first,vertex+1u==last); area+=cross(point(vertex),point(next)); }
  let reverse=select((area < 0.0), (area > 0.0), (ring > firstRing));
  let ringStart=count;
  for(var ordinal=0u;ordinal<last-first;ordinal++) {
    let vertex=first+select(ordinal,last-first-1u-ordinal,reverse);
    let before=first+(vertex-first+last-first-1u)%(last-first); let after=first+(vertex-first+1u)%(last-first);
    if(all(point(vertex)==point(before)) || all(point(vertex)==point(after))) { flags|=1u; continue; }
    if(count>=${this.maximumVertices}u) { flags|=1u; break; }
    let location=point(vertex); if(any((bitcast<vec2u>(location)&vec2u(0x7f800000u))==vec2u(0x7f800000u))) { flags|=1u; }
    store(base,count,9u,location.x); store(base,count,10u,location.y); put(base,count,14u,vertex); count++;
  }
  if(count-ringStart<3u) { flags|=1u; continue; }
  for(var vertex=ringStart;vertex<count;vertex++) {
    let next=select(vertex+1u,ringStart,vertex+1u==count); let previous=select(vertex-1u,count-1u,vertex==ringStart);
    let start=vec2f(scalar(base,vertex,9u),scalar(base,vertex,10u)); let finish=vec2f(scalar(base,next,9u),scalar(base,next,10u)); let edge=finish-start;
    let inward=vec2f(-edge.y,edge.x)/length(edge); store(base,vertex,11u,inward.x); store(base,vertex,12u,inward.y); store(base,vertex,13u,dot(inward,start)); put(base,vertex,6u,previous); put(base,vertex,7u,next);
  }
}
for(var vertex=0u;vertex<count;vertex++) { initialize(base,vertex,readWord(base,vertex,6u),vertex,readWord(base,vertex,6u),readWord(base,vertex,7u),vec2f(scalar(base,vertex,9u),scalar(base,vertex,10u)),0.0); }
var time=0.0; var offsetCount=0u; var captured=false; var eventCount=0u;
loop {
  if(flags!=0u) { break; }
  // Preserve ascending slot order while skipping every historical inactive vertex in searches.
  var firstActive=count;
  for(var remaining=count;remaining>0u;remaining--) { let vertex=remaining-1u; if(readWord(base,vertex,8u)!=0u) { put(base,vertex,18u,firstActive); firstActive=vertex; } }
  // A terminal two-vertex component is a ridge, including simultaneous opposing edge collapses.
  for(var vertex=firstActive;vertex<count;vertex=readWord(base,vertex,18u)) { if(readWord(base,vertex,8u)!=0u) {
    let next=readWord(base,vertex,7u);
    if(next==vertex) { put(base,vertex,8u,0u); }
    else if(readWord(base,next,7u)==vertex) {
      required=emit(index,base,vertex,vec2f(scalar(base,next,9u),scalar(base,next,10u)),required); put(base,vertex,8u,0u); put(base,next,8u,0u);
    }
  } }
  // Simultaneous parallel collisions leave zero-area wavefront components. Their
  // elementary undirected edges are terminal ridges; collinear vertices split them.
  let visitGeneration=eventCount+1u;
  for(var root=firstActive;root<count;root=readWord(base,root,18u)) { if(readWord(base,root,8u)==0u || readWord(base,root,17u)==visitGeneration) { continue; }
    var vertex=root; var twiceArea=0.0; var areaMagnitude=0.0; var perimeter=0.0; var traversed=0u;
    loop { let next=readWord(base,vertex,7u); let start=at(base,vertex,time); let finish=at(base,next,time); let areaTerm=cross(start,finish); twiceArea+=areaTerm; areaMagnitude+=abs(areaTerm); perimeter+=distance(start,finish); put(base,vertex,17u,visitGeneration); vertex=next; traversed++; if(vertex==root || traversed>count) { break; } }
    if(traversed>count) { flags|=8u; break; }
    if(abs(twiceArea)>TOLERANCE*(1.0+perimeter)) {
      // Every rotated sum uses the same terms. A conservative rounding envelope preserves
      // the original per-root checks when their order could change the tolerance decision.
      let roundingEnvelope=8.0*f32(traversed)*1.1920928955078125e-7*(areaMagnitude+TOLERANCE*(1.0+perimeter));
      if (!(abs(twiceArea)>TOLERANCE*(1.0+perimeter)+roundingEnvelope)) {
        vertex=root; loop { put(base,vertex,17u,0u); vertex=readWord(base,vertex,7u); if(vertex==root) { break; } }
      }
      continue;
    }
    vertex=root;
    loop { required=emit(index,base,vertex,at(base,vertex,time),required); vertex=readWord(base,vertex,7u); if(vertex==root) { break; } }
    vertex=root;
    loop {
      let next=readWord(base,vertex,7u); let start=at(base,vertex,time); let finish=at(base,next,time); let direction=finish-start; let lengthSquared=dot(direction,direction); var keep=lengthSquared>TOLERANCE*TOLERANCE;
      var other=root;
      loop {
        let otherNext=readWord(base,other,7u); let otherStart=at(base,other,time); let otherFinish=at(base,otherNext,time);
        let fraction=dot(otherStart-start,direction)/max(lengthSquared,1e-30);
        if(fraction>TOLERANCE && fraction<1.0-TOLERANCE && abs(cross(otherStart-start,direction))<=TOLERANCE*sqrt(lengthSquared)) { keep=false; }
        if(other<vertex && ((distance(start,otherStart)<=TOLERANCE && distance(finish,otherFinish)<=TOLERANCE) || (distance(start,otherFinish)<=TOLERANCE && distance(finish,otherStart)<=TOLERANCE))) { keep=false; }
        other=otherNext; if(other==root) { break; }
      }
      if(keep) { store(base,vertex,9u,start.x); store(base,vertex,10u,start.y); required=emit(index,base,vertex,finish,required); }
      vertex=next; if(vertex==root) { break; }
    }
    vertex=root; loop { put(base,vertex,8u,0u); vertex=readWord(base,vertex,7u); if(vertex==root) { break; } }
  }
  var activeCount=0u; for(var vertex=firstActive;vertex<count;vertex=readWord(base,vertex,18u)) { activeCount+=readWord(base,vertex,8u); }
  if(activeCount==0u) { break; }
  var eventTime=3.0e38; var eventVertex=0xffffffffu; var eventEdge=0xffffffffu; var split=false;
  for(var vertex=firstActive;vertex<count;vertex=readWord(base,vertex,18u)) { if(readWord(base,vertex,8u)==0u) { continue; }
    let next=readWord(base,vertex,7u); let delta=at(base,next,0.0)-at(base,vertex,0.0);
    let velocity=vec2f(scalar(base,next,2u)-scalar(base,vertex,2u),scalar(base,next,3u)-scalar(base,vertex,3u)); let denominator=dot(velocity,velocity);
    if(denominator>1e-12) {
      let candidate=-dot(delta,velocity)/denominator;
      if(candidate>=time-TOLERANCE && candidate<eventTime && length(delta+candidate*velocity)<=TOLERANCE*(1.0+length(at(base,vertex,candidate)))) { eventTime=max(time,candidate); eventVertex=vertex; eventEdge=next; split=false; }
    }
  }
  for(var vertex=firstActive;vertex<count;vertex=readWord(base,vertex,18u)) { if(readWord(base,vertex,8u)==0u || cross(normal(base,readWord(base,vertex,4u)),normal(base,readWord(base,vertex,5u)))>=-1e-7) { continue; }
    let origin=at(base,vertex,0.0); let velocity=vec2f(scalar(base,vertex,2u),scalar(base,vertex,3u));
    for(var edge=firstActive;edge<count;edge=readWord(base,edge,18u)) { if(readWord(base,edge,8u)==0u) { continue; } let next=readWord(base,edge,7u);
      if(edge==vertex || next==vertex) { continue; }
      let edgeId=readWord(base,edge,5u); let inward=normal(base,edgeId); let rate=dot(inward,velocity)-1.0;
      if(rate>=-1e-7) { continue; }
      let candidate=(scalar(base,edgeId,13u)-dot(inward,origin))/rate;
      if(candidate<time-TOLERANCE || candidate>=eventTime-TOLERANCE) { continue; }
      let location=origin+candidate*velocity; let start=at(base,edge,candidate); let finish=at(base,next,candidate); let direction=finish-start;
      let fraction=dot(location-start,direction)/dot(direction,direction);
      if(fraction>=-TOLERANCE && fraction<=1.0+TOLERANCE) { eventTime=max(time,candidate); eventVertex=vertex; eventEdge=edge; split=true; }
    }
  }
  if(!captured && ${a(r.offsetDistance)}<=eventTime) { offsetCount=snapshot(index,base,count,${a(r.offsetDistance)}); captured=true; }
  if(eventVertex==0xffffffffu) { flags|=8u; break; }
  if(eventCount>=${this.maximumEvents}u || count+select(1u,2u,split)>SLOTS) { flags|=4u; break; }
  let location=at(base,eventVertex,eventTime); let previous=readWord(base,eventVertex,6u); let next=readWord(base,eventVertex,7u);
  required=emit(index,base,eventVertex,location,required); put(base,eventVertex,8u,0u);
  if(split) {
    let edgeNext=readWord(base,eventEdge,7u); let edgeId=readWord(base,eventEdge,5u);
    initialize(base,count,readWord(base,eventVertex,4u),edgeId,previous,edgeNext,location,eventTime);
    initialize(base,count+1u,edgeId,readWord(base,eventVertex,5u),eventEdge,next,location,eventTime);
    put(base,previous,7u,count); put(base,edgeNext,6u,count); put(base,eventEdge,7u,count+1u); put(base,next,6u,count+1u); count+=2u;
  } else {
    let after=readWord(base,eventEdge,7u); required=emit(index,base,eventEdge,location,required); put(base,eventEdge,8u,0u);
    initialize(base,count,readWord(base,eventVertex,4u),readWord(base,eventEdge,5u),previous,after,location,eventTime);
    put(base,previous,7u,count); put(base,after,6u,count); count++;
  }
  time=eventTime; eventCount++;
}
workspace[workspaceOffset+base]=count;
if(required>${r.segmentCapacity}u) { flags|=2u; }
status[statusOffset+index*4u]=min(required,${r.segmentCapacity}u); status[statusOffset+index*4u+1u]=required; status[statusOffset+index*4u+2u]=offsetCount; status[statusOffset+index*4u+3u]=flags;`}),o(t,{id:`${s}-offset-links`,operation:`GPUVectorStraightSkeleton`,invocationCount:l*c,bindings:[{name:`workspace`,view:m,type:`u32`,access:`read`},{name:`rings`,view:r.ringOffsets,type:`u32`,access:`read`},{name:`polygons`,view:r.polygonOffsets,type:`u32`,access:`read`},{name:`links`,view:r.output.offsetLinks,type:`u32`,access:`read_write`}],body:`let polygon=index/${c}u; let vertex=index%${c}u;
let base=${p(`polygon`)};
var previous=0xffffffffu; var next=0xffffffffu;
if(vertex<workspace[workspaceOffset+base]) { previous=workspace[workspaceOffset+base+1u+vertex*19u+15u]; next=workspace[workspaceOffset+base+1u+vertex*19u+16u]; }
links[linksOffset+index*2u]=previous; links[linksOffset+index*2u+1u]=next;`})]}},w=16,T=32,E=64,D={vertexCount:12,rows:1,columns:3},O=[.008,.02,.032,.044,.06],k=.2,A=f({parameters:S,caption:`Skeleton arcs and the inward offset curve of star-shaped polygons.`,buildCache:5,createInputs:({seed:e,world:t})=>{let n=p(d(e),t,{...D,irregularity:.4});return{seed:e,world:t,set:n,restPoint:g((e,t)=>{let r=y(n,e,t);return r?.isInside?r.depth:null},.02,t)}},build(e,t,n){let r=m(e,t.set),i=t.set.count,a=e.allocate(`segments`,`float32x4`,i*E),o=e.allocate(`offset-positions`,`float32x2`,i*80),s=e.allocate(`offset-links`,`uint32x2`,i*80),c=e.allocate(`status`,`uint32x4`,i);e.graph.add(new C({positions:r.positions.view,featureOffsets:r.featureOffsets.view,polygonOffsets:r.polygonOffsets.view,ringOffsets:r.ringOffsets.view,maximumVertices:w,maximumEvents:T,segmentCapacity:E,offsetDistance:n.offsetDistance,output:{segments:a.view,sourceEdges:e.allocate(`source-edges`,`uint32x2`,i*E).view,offsetPositions:o.view,offsetLinks:s.view,status:c.view}}));let l=e.allocate(`offset-edges`,`uint32x2`,i*80),u=e.allocate(`parked-offsets`,`float32x2`,i*80);return x(e,{id:`offset-edges`,invocationCount:i*80,bindings:[{name:`links`,buffer:s,access:`read`},{name:`status`,buffer:c,access:`read`},{name:`positions`,buffer:o,access:`read`},{name:`edges`,buffer:l,access:`read_write`},{name:`parked`,buffer:u,access:`read_write`}],body:`let isWritten = index % 80u < status[statusOffset + (index / 80u) * 4u + 2u];
  let nextVertex = links[linksOffset + index * 2u + 1u];
  let hasNext = isWritten && nextVertex != 0xffffffffu;
  edges[edgesOffset + index * 2u] = index;
  edges[edgesOffset + index * 2u + 1u] = select(index, (index / 80u) * 80u + nextVertex, hasNext);
  parked[parkedOffset + index * 2u] = select(1000000.0, positions[positionsOffset + index * 2u], isWritten);
  parked[parkedOffset + index * 2u + 1u] = select(1000000.0, positions[positionsOffset + index * 2u + 1u], isWritten);`}),{outputs:{...r,offsetPositions:u,offsetEdges:l,skeleton:b(e,`skeleton`,a)},update:e=>h(r,p(d(t.seed),t.world,{...D,irregularity:e.irregularity}))}},rebuildOn:[`offsetDistance`],interaction:{kind:`reach`,hint:`go deeper inside`,rest:({inputs:e,toUnit:t})=>t(e.restPoint),resolve(e,{inputs:t,values:n}){let r=y(t.set,e.x,e.y),i=r?.isInside?v(O,r.depth):O[_(r?.depth??0,k,O.length)];return{settle:{offsetDistance:i},glyphs:r?[{shape:`line`,from:[e.x,e.y],to:r.point,dashed:!0},{shape:`dot`,at:r.point}]:[],readout:i===n.offsetDistance?`offsetDistance ${i.toFixed(3)}`:`offsetDistance ${n.offsetDistance.toFixed(3)} -> ${i.toFixed(3)}`}}},draw:({inputs:e,outputs:t,theme:n})=>[{layer:`lines`,context:!0,segments:t.outline,count:e.set.outlineCount},{layer:`lines`,segments:t.skeleton,color:[...n.ink,1],radius:1.6},{layer:`segments`,nodes:t.offsetPositions,edges:t.offsetEdges,color:[1,.6,.2,1],radius:2}]});export{A as default};
import{r as e,t}from"./gpu-offset-curve-Bj8gXMzb.js";import{a as n}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as r}from"./validate-outline-output-passes-DIcqH6Pw.js";import{n as i,r as a,t as o}from"./refinement-storage-validation-D9BhQGNW.js";import{r as s}from"./kernel-C3tfQOn4.js";import{t as c}from"./gpu-line-noding-vE-FDCfr.js";import{c as l,o as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as d,f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{i as m}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{t as h}from"./define-run-B-brkCe4.js";import{r as g}from"./pick-Bo3t3hjv.js";import{o as _}from"./hydrography-N8loF-Ds.js";import{a as v}from"./polygon-set-CHja50mr.js";import{a as y,d as b}from"./pointer-support-C2rPRVFO.js";import{s as x}from"./shape-segments-DMcyF4V-.js";import{io as S}from"./entry-preview-D70TdSbK.js";var C=4294967295,w=class{id;props;constructor(e){this.id=e.id??`offset-curve-components`,this.props=e,a(this.id,`distance magnitude`,Math.abs(e.distance)),a(this.id,`tolerance`,e.tolerance),f(this.id,`fragment metadata words`,e.segmentCapacity*(e.original.positions.length*6+1)*4,1,C),f(this.id,`cut parameter and coordinate words`,e.segmentCapacity*(e.original.positions.length*6+2)*3,1,C),f(this.id,`output component words`,e.output.components.length*4,0,C),f(this.id,`output position words`,e.output.positions.length*2,0,C);for(let t of[e.original,e.raw])d(t.positions,[`float32x2`],this.id),d(t.lineOffsets,[`uint32`],this.id);if(f(this.id,`segmentCapacity`,e.segmentCapacity,1),f(this.id,`intersectionCapacity`,e.intersectionCapacity,1),f(this.id,`rowsPerVertex`,e.rowsPerVertex??1,1),e.original.lineOffsets.length!==e.raw.lineOffsets.length||!Number.isFinite(e.distance)||!(Number.isFinite(e.tolerance)&&e.tolerance>=0))throw Error(`Invalid offset component contract`);if(d(e.output.positions,[`float32x2`],this.id),d(e.output.components,[`uint32x4`],this.id),d(e.output.sourceEdges,[`uint32`],this.id),d(e.output.status,[`uint32x4`],this.id),e.output.sourceEdges.length!==e.output.positions.length||e.output.status.length!==2)throw Error(`Invalid offset output size`);r(this.id,e.output,[Object.keys(e.output)]),p(this.id,Object.values(e.output),[e.original.positions,e.original.lineOffsets,e.raw.positions,e.raw.lineOffsets])}getCommandNodes(e){let{props:t,id:r}=this,a=t.segmentCapacity;n(e,r,Object.fromEntries([t.original.positions,t.original.lineOffsets,t.raw.positions,t.raw.lineOffsets,...Object.values(t.output)].map((e,t)=>[`binding.${t}`,e]))),o(e,r,t.raw.positions.length,t.raw.lineOffsets.length-1,t.intersectionCapacity);let d=t.original.positions.length*6+2;i(e,r,{fragmentsAndMetadata:a*(d-1)*16,cuts:a*d*12,nodedEndpoints:a*16});let f=(t,n,i)=>s(e,`${r}-${t}`,n,i),p={count:f(`count`,`uint32`,1),requiredCount:f(`required`,`uint32`,1),overflow:f(`overflow`,`uint32`,1),candidateOverflow:f(`candidate-overflow`,`uint32`,1)},h={endpoints:f(`endpoints`,`float32x4`,a),sourceFeatureIds:f(`source-lines`,`uint32`,a),sourceRingIds:f(`source-rings`,`uint32`,a),sourceEdgeIds:f(`source-edges`,`uint32`,a),sourceStartParameters:f(`starts`,`float32`,a),sourceEndParameters:f(`ends`,`float32`,a),status:p},g=t.original.positions.length*6+2,_=a*(g-1),v=f(`cuts`,`float32`,a*g*3),y=f(`fragments`,`float32x4`,_),b=s(e,`${r}-fragment-metadata`,`uint32`,_*4);return[...new c({id:`${r}-node`,lines:{kind:`lines`,...t.raw},intersectionCapacity:t.intersectionCapacity,output:h}).getCommandNodes(e),u(e,{id:`${r}-clip-distance-band`,operation:`GPUOffsetCurveComponents`,invocationCount:a,bindings:[{name:`positions`,view:t.original.positions,type:`f32`,access:`read`},{name:`offsets`,view:t.original.lineOffsets,type:`u32`,access:`read`},{name:`endpoints`,view:h.endpoints,type:`f32`,access:`read`},{name:`lines`,view:h.sourceFeatureIds,type:`u32`,access:`read`},{name:`edges`,view:h.sourceEdgeIds,type:`u32`,access:`read`},{name:`cuts`,view:v,type:`f32`,access:`read_write`},{name:`fragments`,view:y,type:`f32`,access:`read_write`},{name:`metadata`,view:b,type:`u32`,access:`read_write`}],declarations:`const CUTS:u32=${g}u;
const DISTANCE:f32=${l(Math.abs(t.distance))};
const TOLERANCE:f32=${l(t.tolerance)};
${m(`point`,`positions`)}
fn insertCut(base:u32,count:u32,value:f32,position:vec2f)->u32 {
 if(!(value>0.0 && value<1.0) || count>=CUTS) {return count;}
 var destination=count;
 loop {if(destination==0u || cuts[cutsOffset+(base+destination-1u)*3u]<=value) {break;}destination--;}
 if(destination>0u && cuts[cutsOffset+(base+destination-1u)*3u]==value) {return count;}
 for(var slot=count;slot>destination;slot--) {for(var component=0u;component<3u;component++) {cuts[cutsOffset+(base+slot)*3u+component]=cuts[cutsOffset+(base+slot-1u)*3u+component];}}
 cuts[cutsOffset+(base+destination)*3u]=value;cuts[cutsOffset+(base+destination)*3u+1u]=position.x;cuts[cutsOffset+(base+destination)*3u+2u]=position.y;return count+1u;
}
fn separation(query:vec2f,line:u32,owner:u32,outerJoin:bool)->f32 {
 var minimum=3.0e38;
 for(var vertex=offsets[offsetsOffset+line];vertex+1u<offsets[offsetsOffset+line+1u];vertex++) {
  if(outerJoin && (vertex==owner || vertex+1u==owner)) {continue;}
  let start=point(vertex);let direction=point(vertex+1u)-start;let denominator=dot(direction,direction);if(denominator==0.0) {continue;}
  let delta=query-start;let projection=dot(delta,direction)/denominator;
  var separation=abs(delta.x*direction.y-delta.y*direction.x)/sqrt(denominator);
  if(projection<=0.0) {separation=length(delta);}
  if(projection>=1.0) {separation=distance(query,point(vertex+1u));}
  minimum=min(minimum,separation);
 }
 return minimum;
}`,body:`let base=index*CUTS;let fragmentBase=index*(CUTS-1u);
for(var slot=0u;slot<CUTS-1u;slot++) {metadata[metadataOffset+(fragmentBase+slot)*4u+2u]=0u;}
let line=lines[linesOffset+index];if(line>=${t.original.lineOffsets.length-1}u) {return;}
let first=vec2f(endpoints[endpointsOffset+index*4u],endpoints[endpointsOffset+index*4u+1u]);let last=vec2f(endpoints[endpointsOffset+index*4u+2u],endpoints[endpointsOffset+index*4u+3u]);let direction=last-first;let squaredLength=dot(direction,direction);if(squaredLength==0.0) {return;}
let rawEdge=edges[edgesOffset+index];let owner=rawEdge/${t.rowsPerVertex??1}u;var outerJoin=false;
if(rawEdge%${t.rowsPerVertex??1}u+1u<${t.rowsPerVertex??1}u && owner>offsets[offsetsOffset+line] && owner+1u<offsets[offsetsOffset+line+1u]) {
 let incoming=point(owner)-point(owner-1u);let outgoing=point(owner+1u)-point(owner);
 outerJoin=(incoming.x*outgoing.y-incoming.y*outgoing.x)*${l(t.distance)}<0.0;
}
cuts[cutsOffset+base*3u]=0.0;cuts[cutsOffset+base*3u+1u]=first.x;cuts[cutsOffset+base*3u+2u]=first.y;
cuts[cutsOffset+(base+1u)*3u]=1.0;cuts[cutsOffset+(base+1u)*3u+1u]=last.x;cuts[cutsOffset+(base+1u)*3u+2u]=last.y;var count=2u;
for(var vertex=offsets[offsetsOffset+line];vertex+1u<offsets[offsetsOffset+line+1u];vertex++) {
 if(outerJoin && (vertex==owner || vertex+1u==owner)) {continue;}
 let start=point(vertex);let finish=point(vertex+1u);let edge=finish-start;let lengthSquared=dot(edge,edge);if(lengthSquared==0.0) {continue;}
 let normal=vec2f(-edge.y,edge.x)/sqrt(lengthSquared);let denominator=dot(direction,normal);
 if(abs(denominator)>1e-20) {for(var side=0u;side<2u;side++) {
  let bandDistance=select(-DISTANCE,DISTANCE,side==1u);
  let fraction=(bandDistance-dot(first-start,normal))/denominator;var position=first+fraction*direction;
  if(normal.x==0.0) {position.y=start.y+bandDistance*normal.y;}
  if(normal.y==0.0) {position.x=start.x+bandDistance*normal.x;}
  let projection=dot(position-start,edge)/lengthSquared;
  if(projection>=0.0 && projection<=1.0) {count=insertCut(base,count,fraction,position);}
 }}
 for(var endpoint=0u;endpoint<2u;endpoint++) {
  let center=select(start,finish,endpoint==1u);let delta=first-center;var segmentLength=sqrt(squaredLength);var unit=direction/segmentLength;
  if(direction.y==0.0) {segmentLength=abs(direction.x);unit=vec2f(sign(direction.x),0.0);}
  else if(direction.x==0.0) {segmentLength=abs(direction.y);unit=vec2f(0.0,sign(direction.y));}
  let along=dot(delta,unit);let across=delta.x*unit.y-delta.y*unit.x;
  let discriminant=(DISTANCE-abs(across))*(DISTANCE+abs(across));
  if(discriminant>=0.0) {
   let root=sqrt(discriminant);
   // Construct the geometric circle intersection directly. Reconstructing a tangent via a
   // rounded fraction can turn x=3 into 3.000000238 and prevent a reversed spur cancelling.
   let foot=center+across*vec2f(unit.y,-unit.x);
   count=insertCut(base,count,(-along-root)/segmentLength,foot-root*unit);
   count=insertCut(base,count,(-along+root)/segmentLength,foot+root*unit);
  }
 }
}
// Cut coordinates are stored once at insertion, and moved with their parameters while sorting.
let outward=vec2f(-direction.y,direction.x)/sqrt(squaredLength);
for(var slot=0u;slot+1u<count;slot++) {
 let startParameter=cuts[cutsOffset+(base+slot)*3u];let endParameter=cuts[cutsOffset+(base+slot+1u)*3u];if(endParameter<=startParameter) {continue;}
 let start=vec2f(cuts[cutsOffset+(base+slot)*3u+1u],cuts[cutsOffset+(base+slot)*3u+2u]);
 let finish=vec2f(cuts[cutsOffset+(base+slot+1u)*3u+1u],cuts[cutsOffset+(base+slot+1u)*3u+2u]);
 if(all(start==finish)) {continue;}
 let middle=first+0.5*(startParameter+endParameter)*direction;
 // Probe far enough to move in float32 along the normal (four relative ULPs), independently
 // of endpoint identity. A 1e-5-unit probe cannot move x at projected-coordinate magnitudes.
 let roundingStep=4.76837158203125e-7*max(abs(middle.x*outward.x),abs(middle.y*outward.y));
 let epsilon=max(roundingStep,max(1e-5,max(TOLERANCE*2.0,DISTANCE*1e-5)));
 if(separation(middle,line,owner,outerJoin)+TOLERANCE<DISTANCE) {continue;}
 // A probe longer than this interval cannot decide its local two-sided coverage. Retain the
 // already distance-band-selected interval rather than removing a tiny connector.
 if(distance(start,finish)>epsilon && max(separation(middle+epsilon*outward,line,owner,outerJoin),separation(middle-epsilon*outward,line,owner,outerJoin))<=DISTANCE+epsilon*0.05) {continue;}
 let fragment=fragmentBase+slot;
 fragments[fragmentsOffset+fragment*4u]=start.x;fragments[fragmentsOffset+fragment*4u+1u]=start.y;fragments[fragmentsOffset+fragment*4u+2u]=finish.x;fragments[fragmentsOffset+fragment*4u+3u]=finish.y;
 metadata[metadataOffset+fragment*4u]=line;metadata[metadataOffset+fragment*4u+1u]=owner;metadata[metadataOffset+fragment*4u+2u]=1u;
}`}),u(e,{id:`${r}-components`,operation:`GPUOffsetCurveComponents`,invocationCount:1,bindings:[{name:`fragments`,view:y,type:`f32`,access:`read`},{name:`metadata`,view:b,type:`u32`,access:`read_write`},{name:`positions`,view:t.output.positions,type:`f32`,access:`read_write`},{name:`components`,view:t.output.components,type:`u32`,access:`read_write`},{name:`sources`,view:t.output.sourceEdges,type:`u32`,access:`read_write`},{name:`status`,view:t.output.status,type:`u32`,access:`read_write`}],declarations:`const CAPACITY:u32=${_}u;
fn startPoint(fragment:u32)->vec2f {return vec2f(fragments[fragmentsOffset+fragment*4u],fragments[fragmentsOffset+fragment*4u+1u]);}
fn endPoint(fragment:u32)->vec2f {return vec2f(fragments[fragmentsOffset+fragment*4u+2u],fragments[fragmentsOffset+fragment*4u+3u]);}
// Shared noded/clipped endpoints establish topology; proximity must never bridge a clipped gap.
fn matches(left:vec2f,right:vec2f)->bool {return all(left==right);}
fn writeVertex(index:u32,value:vec2f,source:u32) {if(index<${t.output.positions.length}u) {positions[positionsOffset+index*2u]=value.x;positions[positionsOffset+index*2u+1u]=value.y;sources[sourcesOffset+index]=source;}}`,body:`var componentCount=0u;var vertexCount=0u;var writtenComponents=0u;var writtenVertices=0u;
for(var slot=0u;slot<${t.output.components.length*4}u;slot++) {components[componentsOffset+slot]=0u;}
// Opposite coincident fragments bound no offset component; cancel collapsed spurs.
for(var candidate=0u;candidate<CAPACITY;candidate++) {if(metadata[metadataOffset+candidate*4u+2u]!=1u) {continue;}
 for(var other=candidate+1u;other<CAPACITY;other++) {if(metadata[metadataOffset+other*4u+2u]!=1u || metadata[metadataOffset+other*4u]!=metadata[metadataOffset+candidate*4u]) {continue;}
  if(matches(startPoint(candidate),endPoint(other)) && matches(endPoint(candidate),startPoint(other))) {metadata[metadataOffset+candidate*4u+2u]=0u;metadata[metadataOffset+other*4u+2u]=0u;break;}
  if(matches(startPoint(candidate),startPoint(other)) && matches(endPoint(candidate),endPoint(other))) {metadata[metadataOffset+other*4u+2u]=0u;}
 }
}
loop {
 var seed=0xffffffffu;
 for(var candidate=0u;candidate<CAPACITY;candidate++) {if(metadata[metadataOffset+candidate*4u+2u]!=1u) {continue;}
  if(seed==0xffffffffu) {seed=candidate;}
  var incoming=false;for(var other=0u;other<CAPACITY;other++) {if(other!=candidate && metadata[metadataOffset+other*4u+2u]==1u && metadata[metadataOffset+other*4u]==metadata[metadataOffset+candidate*4u] && matches(endPoint(other),startPoint(candidate))) {incoming=true;break;}}
  if(!incoming) {seed=candidate;break;}
 }
 if(seed==0xffffffffu) {break;}
 let componentStart=vertexCount;let line=metadata[metadataOffset+seed*4u];var fragment=seed;
 writeVertex(vertexCount,startPoint(seed),metadata[metadataOffset+seed*4u+1u]);vertexCount++;
 loop {
  metadata[metadataOffset+fragment*4u+2u]=2u;let finish=endPoint(fragment);writeVertex(vertexCount,finish,metadata[metadataOffset+fragment*4u+1u]);vertexCount++;
  var next=0xffffffffu;for(var candidate=0u;candidate<CAPACITY;candidate++) {if(metadata[metadataOffset+candidate*4u+2u]==1u && metadata[metadataOffset+candidate*4u]==line && matches(finish,startPoint(candidate))) {next=candidate;break;}}
  if(next==0xffffffffu) {break;}fragment=next;
 }
 if(componentCount<${t.output.components.length}u && vertexCount<=${t.output.positions.length}u) {components[componentsOffset+componentCount*4u]=componentStart;components[componentsOffset+componentCount*4u+1u]=vertexCount-componentStart;components[componentsOffset+componentCount*4u+2u]=line;writtenComponents++;writtenVertices=vertexCount;}componentCount++;
}
let overflow=componentCount>${t.output.components.length}u || vertexCount>${t.output.positions.length}u;
status[statusOffset]=writtenComponents;status[statusOffset+1u]=writtenVertices;status[statusOffset+2u]=componentCount;status[statusOffset+3u]=vertexCount;status[statusOffset+4u]=select(0u,1u,overflow);status[statusOffset+6u]=0u;status[statusOffset+7u]=0u;`}),u(e,{id:`${r}-noding-status`,operation:`GPUOffsetCurveComponents`,invocationCount:1,bindings:[{name:`overflow`,view:p.overflow,type:`u32`,access:`read`},{name:`candidate`,view:p.candidateOverflow,type:`u32`,access:`read`},{name:`status`,view:t.output.status,type:`u32`,access:`read_write`}],body:`status[statusOffset+5u]=overflow[overflowOffset]|candidate[candidateOffset];`})]}},T=[`round`,`mitre`,`bevel`],E=8,D=1024,O=[.03,.06,.09,.12,.15],k=.5,A=128,j=h({parameters:S,caption:`The raw offset of a river reach, cleaned into valid components.`,buildCache:5,createInputs:({world:e})=>{let t=Float32Array.from(_(1,24,e).flat());return{positions:t,vertexCount:24,restPoint:y((e,n)=>g(t,e,n)?.distance??null,.09/k,e)}},build(n,{positions:r,vertexCount:i},a){let o=T[a.joinStyle],s=e(o,E,`buffer`),c=i*s,l=n.upload(`original`,`float32x2`,r),u=n.upload(`original-offsets`,`uint32`,Uint32Array.of(0,i)),d=n.allocate(`raw`,`float32x2`,c),f=n.upload(`raw-offsets`,`uint32`,Uint32Array.of(0,c)),p=n.upload(`parameters`,`float32`,Float32Array.of(a.distance,5,0,0));n.graph.add(new t({positions:l.view,lineOffsets:u.view,joinStyle:o,quadSegments:E,insideTurns:`buffer`,parameters:p.view,output:{positions:d.view}}));let m=n.allocate(`cleaned`,`float32x2`,D),h=n.allocate(`components`,`uint32x4`,A);n.graph.add(new w({original:{positions:l.view,lineOffsets:u.view},raw:{positions:d.view,lineOffsets:f.view},rowsPerVertex:s,distance:a.distance,tolerance:a.distance*.01,segmentCapacity:512,intersectionCapacity:256,output:{positions:m.view,sourceEdges:n.allocate(`source-edges`,`uint32`,D).view,components:h.view,status:n.allocate(`status`,`uint32x4`,2).view}}));let g=n.allocate(`component-edges`,`uint32x2`,D),_=n.allocate(`component-categories`,`uint32`,D);return x(n,{id:`component-edges`,invocationCount:D,bindings:[{name:`components`,buffer:h,access:`read`},{name:`edges`,buffer:g,access:`read_write`},{name:`categories`,buffer:_,access:`read_write`}],body:`var nextVertex = index;
  var category = 0xffffffffu;
  for (var component = 0u; component < ${A}u; component++) {
    let first = components[componentsOffset + component * 4u];
    let count = components[componentsOffset + component * 4u + 1u];
    if (count > 0u && index >= first && index < first + count) {
      category = component;
      if (index + 1u < first + count) { nextVertex = index + 1u; }
    }
  }
  edges[edgesOffset + index * 2u] = index;
  edges[edgesOffset + index * 2u + 1u] = nextVertex;
  categories[categoriesOffset + index] = category;`}),{outputs:{original:l,raw:d,cleaned:m,edges:g,categories:_,originalIds:n.upload(`original-ids`,`uint32`,v(i)),rawIds:n.upload(`raw-ids`,`uint32`,v(c))}}},rebuildOn:[`distance`,`joinStyle`],interaction:{kind:`reach`,hint:`move away from the river`,rest:({inputs:e,toUnit:t})=>t(e.restPoint),resolve(e,{inputs:t,values:n}){let r=g(t.positions,e.x,e.y),i=b(O,(r?.distance??0)*k);return{settle:{distance:i},glyphs:r?[{shape:`ring`,at:r.point,radius:i,dashed:!0},{shape:`line`,from:[e.x,e.y],to:r.point,dashed:!0}]:[],readout:i===n.distance?`distance ${i.toFixed(3)} · ${n.joinStyle===0?`round`:n.joinStyle===1?`mitre`:`bevel`} joins`:`distance ${n.distance.toFixed(3)} -> ${i.toFixed(3)}`}}},draw:({outputs:e,inputs:t,theme:n})=>[{layer:`paths`,context:!0,nodes:e.raw,ids:e.rawIds,count:e.rawIds.length,radius:1.2},{layer:`segments`,nodes:e.cleaned,edges:e.edges,values:e.categories,radius:2.8},{layer:`paths`,nodes:e.original,ids:e.originalIds,count:t.vertexCount,color:[...n.ink,1],radius:1.4}]});export{j as default};
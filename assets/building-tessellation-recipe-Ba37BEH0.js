import{t as e}from"./spatial-analysis-packed-kernel-CV3ZCYPQ.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-reduction-tNgOVQ_C.js";import{D as r,E as i,S as a,T as o,f as s,p as c,x as l}from"./gpu-contributor-utils-BIwLQON5.js";import{v as u}from"./spatial-context-DwZEd4EZ.js";import{c as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{u as f}from"./recipe-utils-CICB7qiS.js";import{a as p,i as m,n as h,o as g,s as _,t as v}from"./visibility-graph-point-grid-BR25lfUa.js";import{t as y}from"./gpu-group-aggregation-CEYb6GJw.js";import{t as b}from"./gpu-scan-B4NNTfF9.js";import{t as x}from"./gpu-sort-xRm8l5yn.js";import{n as S}from"./scan-total-C4QddMmi.js";import{t as C}from"./load-balanced-for-each-GB0Xy_f0.js";import{i as w,u as T}from"./wgsl-common-DfyXiYB0.js";import{t as E}from"./binary-search-wgsl-CMmaMYIt.js";import{n as D}from"./point-neighbor-grid-index-D0wS1y1I.js";import{n as O,r as k,t as A}from"./gpu-polygon-triangulation-CBZGZXVA.js";import{t as j}from"./gpu-polygon-overlay-Bu61B05s.js";import{t as M}from"./define-run-B-brkCe4.js";import{o as N}from"./pointer-overlay-Qh8qflcA.js";import{n as P,t as F}from"./pick-Bo3t3hjv.js";import{n as I}from"./kernel-BXyKsp4B.js";import{n as L}from"./polygons-D0hzhPvj.js";import{r as R,t as z}from"./urban-D7A1ccno.js";import{cs as B}from"./entry-preview-D70TdSbK.js";var V=16777216;function H(r,i){let{id:a,buildingCount:o,sampleCapacity:s}=i,c=i.segments.length,l=t(r,`${a}-segment-counts`,`uint32`,c),u=t(r,`${a}-segment-offsets`,`uint32`,c+1),d=t(r,`${a}-invalid-segments`,`uint32`,c),f=t(r,`${a}-invalid-count`,`uint32`,1),p=t(r,`${a}-sample-total`,`uint32`,1),m=t(r,`${a}-sample-data`,`uint32`,s*3),h=r.createDataView(m.buffer,{format:`float32x2`,length:s}),g=r.createDataView(m.buffer,{format:`uint32`,byteOffset:s*8,length:s}),_=t(r,`${a}-sample-rows`,`uint32`,s);r.add(e(r,{id:`${a}-count-segments`,operation:a,invocationCount:c,declarations:T,bindings:[{name:`segments`,view:i.segments,type:`f32`,access:`read`},{name:`owners`,view:i.segmentBuildingIds,type:`u32`,access:`read`},{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`counts`,view:l,type:`u32`,access:`read_write`},{name:`invalid`,view:d,type:`u32`,access:`read_write`}],body:`let base=segmentsOffset+4u*index;
let start=vec2<f32>(segments[base],segments[base+1u]);
let end=vec2<f32>(segments[base+2u],segments[base+3u]);
let span=distance(start,end);
let spacing=parameters[parametersOffset];
let required=ceil(span/spacing);
let valid=isFiniteValue(spacing)&&spacing>0.0&&isFiniteValue(start.x)&&isFiniteValue(start.y)&&isFiniteValue(end.x)&&isFiniteValue(end.y)&&span>0.0&&isFiniteValue(required)&&required<=${V}.0&&owners[ownersOffset+index]<${o}u;
counts[countsOffset+index]=select(0u,u32(clamp(required,0.0,${V}.0)),valid);
invalid[invalidOffset+index]=u32(!valid);`})),r.add(S(r,{id:`${a}-sample-scan`,operation:a,counts:l,offsets:u,rowCount:c,total:p})),r.add(new n({id:`${a}-invalid-total`,input:d,output:f,operation:`sum`}).getCommandNodes(r)),r.add(e(r,{id:`${a}-sample-status`,operation:a,invocationCount:1,bindings:[{name:`total`,view:p,type:`u32`,access:`read`},{name:`invalid`,view:f,type:`u32`,access:`read`},{name:`status`,view:i.samplingStatus,type:`u32`,access:`read_write`},{name:`offsets`,view:u,type:`u32`,access:`read_write`}],body:`let required=total[totalOffset];
offsets[offsetsOffset+${c}u]=required;
status[statusOffset]=min(required,${s}u);
status[statusOffset+1u]=required;
status[statusOffset+2u]=u32(required>${s}u);
status[statusOffset+3u]=invalid[invalidOffset];`})),r.add(e(r,{id:`${a}-clear-samples`,operation:a,invocationCount:s,bindings:[{name:`samples`,view:m,type:`u32`,access:`read_write`},{name:`rows`,view:_,type:`u32`,access:`read_write`}],body:`samples[samplesOffset+2u*index]=0x7fc00000u;
samples[samplesOffset+2u*index+1u]=0x7fc00000u;
samples[samplesOffset+${s*2}u+index]=0xffffffffu;
rows[rowsOffset+index]=index;`})),r.add(C(r,{id:`${a}-write-samples`,operation:a,offsets:u,itemCapacity:s,bindings:[{name:`segments`,view:i.segments,type:`f32`,access:`read`},{name:`buildingIds`,view:i.segmentBuildingIds,type:`u32`,access:`read`},{name:`counts`,view:l,type:`u32`,access:`read`},{name:`samples`,view:m,type:`u32`,access:`read_write`}],declarations:`fn visitLoadBalancedItem(row:u32,item:u32,lane:u32){
let base=segmentsOffset+4u*row;
let start=vec2<f32>(segments[base],segments[base+1u]);
let end=vec2<f32>(segments[base+2u],segments[base+3u]);
let part=item-loadBalancedOffsets[loadBalancedOffsetsOffset+row];
let position=mix(start,end,f32(part)/f32(counts[countsOffset+row]));
samples[samplesOffset+2u*item]=bitcast<u32>(position.x);samples[samplesOffset+2u*item+1u]=bitcast<u32>(position.y);
samples[samplesOffset+${s*2}u+item]=buildingIds[buildingIdsOffset+row];
}`}));let v=t(r,`${a}-building-counts`,`uint32`,o),w=t(r,`${a}-building-offsets`,`uint32`,o+1),E=t(r,`${a}-building-samples`,`uint32`,s),D=t(r,`${a}-sorted-owners`,`uint32`,s);r.add(new y({id:`${a}-count-buildings`,keys:g,output:v}).getCommandNodes(r)),r.add(new b({id:`${a}-building-scan`,input:v,output:w}).getCommandNodes(r)),r.add(new x({id:`${a}-building-sort`,keys:g,values:_,outputKeys:D,outputValues:E}).getCommandNodes(r));let O=t(r,`${a}-building-index`,`uint32`,4+o+1+s);return r.add(e(r,{id:`${a}-pack-buildings`,operation:a,invocationCount:Math.max(4,o+1,s),bindings:[{name:`counts`,view:v,type:`u32`,access:`read`},{name:`offsets`,view:w,type:`u32`,access:`read`},{name:`rows`,view:E,type:`u32`,access:`read`},{name:`output`,view:O,type:`u32`,access:`read_write`},{name:`status`,view:i.samplingStatus,type:`u32`,access:`read`}],body:`if(index<4u){output[outputOffset+index]=status[statusOffset+index];}
if(index<${o}u){output[outputOffset+4u+index]=offsets[offsetsOffset+index];}
if(index==${o}u){output[outputOffset+4u+index]=offsets[offsetsOffset+${o-1}u]+counts[countsOffset+${o-1}u];}
if(index<${s}u){output[outputOffset+${o+5}u+index]=rows[rowsOffset+index];}`})),{samples:h,sampleData:m,owners:g,buildingIndex:O}}var U=1024;function W(e){let t=Math.min(U,Math.max(1,Math.ceil(Math.sqrt(e)))),n=t*t;return{dimension:t,cellCount:n,rowsPerBin:Math.ceil(e/t),byteLength:(5*e+n+1)*4}}function G(n,r){let{id:i,positions:a,count:o}=r,s=a.length,{dimension:c,cellCount:l,rowsPerBin:u,byteLength:d}=W(s),f=t(n,`${i}-packed`,`uint32`,d/4),p=t(n,`${i}-keys`,`uint32`,s),m=t(n,`${i}-rows`,`uint32`,s);for(let r of[0,1]){let c=t(n,`${i}-sorted-keys-${r}`,`uint32`,s),l=t(n,`${i}-sorted-rows-${r}`,`uint32`,s);n.add(e(n,{id:`${i}-prepare-${r}`,operation:`BuildingTessellationIndex`,invocationCount:s,declarations:w+T,bindings:[{name:`positions`,view:a,type:`f32`,access:`read`},{name:`count`,view:o,type:`u32`,access:`read`},{name:`keys`,view:p,type:`u32`,access:`read_write`},{name:`rows`,view:m,type:`u32`,access:`read_write`}],body:`let point=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
let valid=index<count[countOffset]&&isFiniteValue(point.x)&&isFiniteValue(point.y);
keys[keysOffset+index]=select(0xffffffffu,getOrderedKey(point[${r}u]),valid);
rows[rowsOffset+index]=index;`})),n.add(new x({id:`${i}-sort-${r}`,keys:p,values:m,outputKeys:c,outputValues:l}).getCommandNodes(n)),n.add(e(n,{id:`${i}-pack-${r}`,operation:`BuildingTessellationIndex`,invocationCount:s,bindings:[{name:`keys`,view:c,type:`u32`,access:`read`},{name:`rows`,view:l,type:`u32`,access:`read`},{name:`pointIndex`,view:f,type:`u32`,access:`read_write`}],body:`pointIndex[pointIndexOffset+${2*r*s}u+index]=keys[keysOffset+index];
pointIndex[pointIndexOffset+${(2*r+1)*s}u+index]=rows[rowsOffset+index];`}))}let h=t(n,`${i}-bins`,`uint32x2`,s);n.add(e(n,{id:`${i}-ranks`,operation:`BuildingTessellationIndex`,invocationCount:s,bindings:[{name:`pointIndex`,view:f,type:`u32`,access:`read`},{name:`bins`,view:h,type:`u32`,access:`read_write`}],body:`let horizontalRow=pointIndex[pointIndexOffset+${s}u+index];
let verticalRow=pointIndex[pointIndexOffset+${3*s}u+index];
bins[binsOffset+2u*horizontalRow]=index/${u}u;
bins[binsOffset+2u*verticalRow+1u]=index/${u}u;`})),n.add(e(n,{id:`${i}-cells`,operation:`BuildingTessellationIndex`,invocationCount:s,bindings:[{name:`bins`,view:h,type:`u32`,access:`read`},{name:`count`,view:o,type:`u32`,access:`read`},{name:`keys`,view:p,type:`u32`,access:`read_write`},{name:`rows`,view:m,type:`u32`,access:`read_write`}],body:`keys[keysOffset+index]=select(${l}u,bins[binsOffset+2u*index]+${c}u*bins[binsOffset+2u*index+1u],index<count[countOffset]);
rows[rowsOffset+index]=index;`}));let g=D(n,{id:`${i}-grid`,rowCount:s,cellKeys:p,sourceRows:m,cellCount:l});return n.add(g.nodes),n.add(e(n,{id:`${i}-pack-grid`,operation:`BuildingTessellationIndex`,invocationCount:Math.max(s,l+1),bindings:[{name:`offsets`,view:g.cellOffsets,type:`u32`,access:`read`},{name:`rows`,view:g.sortedRows,type:`u32`,access:`read`},{name:`pointIndex`,view:f,type:`u32`,access:`read_write`}],body:`if(index<${l+1}u){pointIndex[pointIndexOffset+${4*s}u+index]=offsets[offsetsOffset+index];}
if(index<${s}u){pointIndex[pointIndexOffset+${4*s+l+1}u+index]=rows[rowsOffset+index];}`})),{data:f,rowCount:s,dimension:c,rowsPerBin:u}}function K(e){let{rowCount:t}=e;return w+E(`pointIndex`)+`
fn getCandidateRanges(center:vec2<f32>,radius:f32)->vec4<u32>{
 let lower=center-vec2<f32>(radius);let upper=center+vec2<f32>(radius);
 // Infinite endpoints are valid ordered search bounds; NaNs are excluded by the caller.
 let horizontalFirst=lowerBoundPointIndex(0u,${t}u,select(getOrderedKey(lower.x),0x7fffffffu,lower.x==0.0));
 let horizontalEnd=upperBoundPointIndex(0u,${t}u,select(getOrderedKey(upper.x),0x80000000u,upper.x==0.0));
 let verticalFirst=lowerBoundPointIndex(${2*t}u,${3*t}u,select(getOrderedKey(lower.y),0x7fffffffu,lower.y==0.0));
 let verticalEnd=upperBoundPointIndex(${2*t}u,${3*t}u,select(getOrderedKey(upper.y),0x80000000u,upper.y==0.0));
 return vec4<u32>(horizontalFirst,horizontalEnd,verticalFirst-${2*t}u,verticalEnd-${2*t}u);
}
fn getCandidateInterval(center:vec2<f32>,radius:f32)->vec3<u32>{
 let ranges=getCandidateRanges(center,radius);
 if(ranges.w-ranges.z<ranges.y-ranges.x){return vec3<u32>(ranges.z+${2*t}u,ranges.w+${2*t}u,${t}u);}
 return vec3<u32>(ranges.x,ranges.y,${t}u);
}
`}function q(e,t,n){let{dimension:r,rowsPerBin:i,rowCount:a}=e,o=4*a,s=o+r*r+1;return`let ranges=getCandidateRanges(center,searchRadius);
 if(ranges.x==ranges.y||ranges.z==ranges.w){return vec2<u32>(0u);}
 let lower=vec2<u32>(ranges.x,ranges.z)/${i}u;
 let upper=(vec2<u32>(ranges.y,ranges.w)-vec2<u32>(1u))/${i}u;
 for(var gridRow=lower.y;gridRow<=upper.y;gridRow++){
  for(var gridColumn=lower.x;gridColumn<=upper.x;gridColumn++){
   let cell=gridRow*${r}u+gridColumn;
   let first=pointIndex[pointIndexOffset+${o}u+cell];
   let end=pointIndex[pointIndexOffset+${o+1}u+cell];
   for(var slot=first;slot<end;slot++){
    ${n}
    let candidate=pointIndex[pointIndexOffset+${s}u+slot];
    ${t}
   }
  }
 }`}function J(n,r){let{id:i,positions:a,triangleCapacity:o}=r,s=t(n,`${i}-feature-offsets`,`uint32`,2),c=t(n,`${i}-polygon-offsets`,`uint32`,2),l=r.ringOffsets??t(n,`${i}-ring-offsets`,`uint32`,2),u=t(n,`${i}-indices`,`uint32`,3*o),d=t(n,`${i}-valid`,`uint32`,1),f=t(n,`${i}-triangles`,`float32x2`,1+3*o);return n.add(e(n,{id:`${i}-layout`,operation:`BuildingTessellationEnclosure`,invocationCount:1,bindings:[{name:`features`,view:s,type:`u32`,access:`read_write`},{name:`polygons`,view:c,type:`u32`,access:`read_write`},...r.ringOffsets?[]:[{name:`rings`,view:l,type:`u32`,access:`read_write`}]],body:`features[featuresOffset]=0u;features[featuresOffset+1u]=1u;
polygons[polygonsOffset]=0u;polygons[polygonsOffset+1u]=${l.length-1}u;
${r.ringOffsets?``:`rings[ringsOffset]=0u;rings[ringsOffset+1u]=${a.length}u;`}`})),n.add(new A({id:`${i}-triangulation`,positions:a,featureOffsets:s,polygonOffsets:c,ringOffsets:l,output:{indices:u,valid:d},useZOrderHash:!1,largePolygonMethod:`none`}).getCommandNodes(n)),n.add(e(n,{id:`${i}-pack`,operation:`BuildingTessellationEnclosure`,invocationCount:o,bindings:[{name:`positions`,view:a,type:`f32`,access:`read`},{name:`indices`,view:u,type:`u32`,access:`read`},{name:`validity`,view:d,type:`u32`,access:`read`},{name:`triangles`,view:f,type:`f32`,access:`read_write`}],body:`if(index==0u){triangles[trianglesOffset]=f32(validity[validityOffset]);triangles[trianglesOffset+1u]=0.0;}
for(var vertex=0u;vertex<3u;vertex++){
 let source=indices[indicesOffset+3u*index+vertex];
 let destination=trianglesOffset+2u*(1u+3u*index+vertex);
 triangles[destination]=positions[positionsOffset+2u*source];triangles[destination+1u]=positions[positionsOffset+2u*source+1u];
}`})),f}function Y(t,n){let{buildingCount:r,maximumSamplesPerBuilding:i,candidateCapacity:a,cellVertexCapacity:o}=n,s=n.simpleEnclosure?3:n.enclosure.length,c=i*n.enclosurePieceCount,l=s+a,u=n.sampleData.length/3;return e(t,{id:n.id,operation:`GPUBuildingTessellation`,variant:`local-sample-cell-union`,invocationCount:r,workgroupSize:4,bindings:[{name:`samples`,view:n.sampleData,type:`u32`,access:`read`},{name:`buildingIndex`,view:n.buildingIndex,type:`u32`,access:`read`},{name:`pointIndex`,view:n.pointIndex.data,type:`u32`,access:`read`},{name:`enclosure`,view:n.enclosure,type:`f32`,access:`read`},{name:`positions`,view:n.positions,type:`f32`,access:`read_write`},{name:`cells`,view:n.cells,type:`u32`,access:`read_write`},{name:`buildingStatus`,view:n.buildingStatus,type:`u32`,access:`read_write`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`}],declarations:T+K(n.pointIndex)+`
var<private> candidateRows:array<u32,${a}>;
fn readSample(row:u32)->vec2<f32>{return vec2<f32>(bitcast<f32>(samples[samplesOffset+2u*row]),bitcast<f32>(samples[samplesOffset+2u*row+1u]));}
fn readOwner(row:u32)->u32{return samples[samplesOffset+${2*u}u+row];}
fn readOwnSample(slot:u32)->u32{return buildingIndex[buildingIndexOffset+${r+5}u+slot];}
fn readEnclosure(row:u32)->vec2<f32>{return vec2<f32>(enclosure[enclosureOffset+2u*row],enclosure[enclosureOffset+2u*row+1u]);}
fn crossVectors(first:vec2<f32>,second:vec2<f32>)->f32{return first.x*second.y-first.y*second.x;}
// No constant-trip-count fill of candidateRows: all stores use a data-dependent append index.
fn gatherGridCandidates(center:vec2<f32>,searchRadius:f32)->vec2<u32>{
 var count=0u;var visited=0u;
 ${q(n.pointIndex,`
  visited++;
  let displacement=readSample(candidate)-center;
  if(any(abs(displacement)>vec2<f32>(searchRadius))){continue;}
  let candidateDistance=length(displacement);
  if(!isFiniteValue(candidateDistance)){return vec2<u32>(count,2u);}
  if(candidateDistance<=searchRadius){
   if(count>=${a}u){return vec2<u32>(count,1u);}
   candidateRows[count]=candidate;count++;
  }`,`if(visited>=${n.maximumGridVisits}u){return vec2<u32>(count,1u);}`)}
 return vec2<u32>(count,0u);
}
fn gatherCandidates(center:vec2<f32>,searchRadius:f32)->vec2<u32>{
 let gridResult=gatherGridCandidates(center,searchRadius);
 if(gridResult.y!=1u){return gridResult;}
 // Retry a dense/coarse grid query independently; neither interval depends on global extent.
 var count=0u;var visited=0u;
 let interval=getCandidateInterval(center,searchRadius);
 for(var slot=interval.x;slot<interval.y;slot++){
  if(visited>=${n.maximumGridVisits}u){return vec2<u32>(count,1u);}
  visited++;
  let candidate=pointIndex[pointIndexOffset+interval.z+slot];
  let displacement=readSample(candidate)-center;
  if(any(abs(displacement)>vec2<f32>(searchRadius))){continue;}
  let candidateDistance=length(displacement);
  if(!isFiniteValue(candidateDistance)){return vec2<u32>(count,2u);}
  if(candidateDistance<=searchRadius){
   if(count>=${a}u){return vec2<u32>(count,1u);}
   candidateRows[count]=candidate;count++;
  }
 }
 return vec2<u32>(count,0u);
}
`,body:`let first=buildingIndex[buildingIndexOffset+4u+index];
let end=buildingIndex[buildingIndexOffset+5u+index];
let ownCount=end-first;
let rowBase=buildingStatusOffset+4u*index;
buildingStatus[rowBase]=0u;buildingStatus[rowBase+1u]=0u;buildingStatus[rowBase+2u]=0u;buildingStatus[rowBase+3u]=0u;
for(var part=0u;part<${c}u;part++){
 let record=cellsOffset+4u*(index*${c}u+part);
 cells[record]=0u;cells[record+1u]=select(0u,0xffffffffu,part<ownCount*${n.enclosurePieceCount}u);cells[record+2u]=index;cells[record+3u]=0xffffffffu;
 if(part<ownCount*${n.enclosurePieceCount}u){cells[record+3u]=readOwnSample(first+part/${n.enclosurePieceCount}u);}
}
if(buildingIndex[buildingIndexOffset+3u]!=0u){buildingStatus[rowBase+3u]=1u;return;}
if(buildingIndex[buildingIndexOffset+2u]!=0u||ownCount>${i}u){buildingStatus[rowBase+2u]=1u;return;}
if(ownCount==0u){buildingStatus[rowBase+3u]=1u;return;}
let radius=parameters[parametersOffset+1u];let tolerance=parameters[parametersOffset+2u];
if(!isFiniteValue(radius)||radius<=0.0||!isFiniteValue(tolerance)||tolerance<=0.0){buildingStatus[rowBase+3u]=1u;return;}
var lower=vec2<f32>(3.402823e38);var upper=vec2<f32>(-3.402823e38);
for(var part=0u;part<ownCount;part++){let point=readSample(readOwnSample(first+part));lower=min(lower,point);upper=max(upper,point);}
let center=0.5*lower+0.5*upper;
// Slack covers float32 cell assignment, the radius predicate and the twice-radius certificate.
let slack=max(4.0*tolerance,1.0e-5*(max(abs(center.x),abs(center.y))+radius));
let searchRadius=radius+slack;
if(!isFiniteValue(searchRadius)||!isFiniteValue(center.x)||!isFiniteValue(center.y)){buildingStatus[rowBase+3u]=1u;return;}
for(var part=0u;part<ownCount;part++){
 let ownDistance=distance(readSample(readOwnSample(first+part)),center)+slack;
 if(!isFiniteValue(ownDistance)){buildingStatus[rowBase+3u]=1u;return;}
 if(ownDistance>=radius){buildingStatus[rowBase+2u]=1u;return;}
}
${n.simpleEnclosure?`if(enclosure[enclosureOffset]!=1.0){buildingStatus[rowBase+3u]=1u;return;}`:`// Convexity is certified against every edge, not only consecutive turns (a star ring can
// turn consistently). This is bounded by the named enclosure capacity, never by building count.
var enclosureValid=true;var enclosureArea=0.0;let enclosureAnchor=readEnclosure(0u)-center;
for(var vertex=0u;vertex<${s}u;vertex++){
 let point=readEnclosure(vertex)-center;let next=readEnclosure((vertex+1u)%${s}u)-center;
 enclosureValid=enclosureValid&&isFiniteValue(point.x)&&isFiniteValue(point.y);
 enclosureArea+=crossVectors(point-enclosureAnchor,next-enclosureAnchor);
 for(var otherVertex=0u;otherVertex<${s}u;otherVertex++){
  let other=readEnclosure(otherVertex)-center;let side=crossVectors(next-point,other-point);
  enclosureValid=enclosureValid&&isFiniteValue(side)&&side>=0.0;
  if(otherVertex!=vertex&&all(other==point)){enclosureValid=false;}
 }
}
if(!enclosureValid||!isFiniteValue(enclosureArea)||enclosureArea<=0.0){buildingStatus[rowBase+3u]=1u;return;}
`}
let gathered=gatherCandidates(center,searchRadius);
if(gathered.y!=0u){buildingStatus[rowBase+select(2u,3u,gathered.y==2u)]=1u;return;}
let exhaustive=gathered.x==buildingIndex[buildingIndexOffset];
var polygon:array<vec2<f32>,${l}>;
var scratch:array<vec2<f32>,${l}>;
var candidateOverflow=false;var overflow=false;var invalid=false;
// A building is one feature represented by the union of its own sample cells. Clipping against
// its other own sites partitions this union without changing its area or exterior boundary.
for(var part=0u;part<ownCount;part++){
 let sampleRow=readOwnSample(first+part);let site=readSample(sampleRow)-center;
 for(var enclosurePart=0u;enclosurePart<${n.enclosurePieceCount}u;enclosurePart++){
 let piece=part*${n.enclosurePieceCount}u+enclosurePart;
 var size=${s}u;
 for(var vertex=0u;vertex<${s}u;vertex++){
  polygon[vertex]=readEnclosure(${n.simpleEnclosure?`1u+3u*enclosurePart+vertex`:`vertex`})-center;
 }
 ${n.simpleEnclosure?`let triangleArea=crossVectors(polygon[1u]-polygon[0u],polygon[2u]-polygon[0u]);
 let firstVertex=readEnclosure(1u+3u*enclosurePart);
 let secondVertex=readEnclosure(2u+3u*enclosurePart);
 let thirdVertex=readEnclosure(3u+3u*enclosurePart);
 let padding=all(firstVertex==secondVertex)&&all(secondVertex==thirdVertex);
 // Padding is exactly repeated source vertices. A real triangle lost to translation/underflow
 // must invalidate the building instead of silently dropping area.
 if(padding){size=0u;}
 else if(!isFiniteValue(triangleArea)||triangleArea<=0.0){invalid=true;}`:``}
 for(var candidateIndex=0u;candidateIndex<gathered.x;candidateIndex++){
  if(size==0u||invalid){break;}
  let other=candidateRows[candidateIndex];if(other==sampleRow){continue;}
  let displacement=(readSample(other)-center)-site;let squared=dot(displacement,displacement);
  if(!isFiniteValue(squared)){invalid=true;break;}
  if(squared==0.0){
   if(readOwner(other)!=index){invalid=true;}
   else if(other<sampleRow){size=0u;}
   continue;
  }
  let threshold=0.5*squared;var outputSize=0u;
  for(var vertex=0u;vertex<size;vertex++){
   let start=polygon[vertex];let finish=polygon[(vertex+1u)%size];
   let startProducts=(start-site)*displacement;let endProducts=(finish-site)*displacement;
   let startSide=startProducts.x+startProducts.y-threshold;let endSide=endProducts.x+endProducts.y-threshold;
   if(!isFiniteValue(startProducts.x)||!isFiniteValue(startProducts.y)||!isFiniteValue(endProducts.x)||!isFiniteValue(endProducts.y)){invalid=true;break;}
   let inside=startSide<=0.0;let nextInside=endSide<=0.0;
   if(!isFiniteValue(startSide)||!isFiniteValue(endSide)){invalid=true;break;}
   if(inside){if(outputSize<${l}u){scratch[outputSize]=start;}outputSize++;}
   if(inside!=nextInside){
    // Normalize before subtracting: finite opposite sides can have an infinite difference.
    let sideScale=max(abs(startSide),abs(endSide));
    let scaledStart=startSide/sideScale;let scaledEnd=endSide/sideScale;
    let denominator=scaledStart-scaledEnd;let fraction=scaledStart/denominator;
    if(!isFiniteValue(sideScale)||sideScale==0.0||!isFiniteValue(denominator)||denominator==0.0||!isFiniteValue(fraction)||fraction<0.0||fraction>1.0){invalid=true;break;}
    let intersection=(1.0-fraction)*start+fraction*finish;
    invalid=invalid||!isFiniteValue(intersection.x)||!isFiniteValue(intersection.y);
    if(outputSize<${l}u){scratch[outputSize]=intersection;}outputSize++;
   }
  }
  if(outputSize>${l}u){invalid=true;break;}
  size=outputSize;for(var vertex=0u;vertex<size;vertex++){polygon[vertex]=scratch[vertex];}
 }
 if(size<3u){size=0u;}
 if(size>0u&&!invalid&&!exhaustive){
  var greatest=0.0;
  for(var vertex=0u;vertex<size;vertex++){greatest=max(greatest,distance(polygon[vertex],site));}
  // If an excluded site q could cut a point x, |q-site| <= 2|x-site|. Every excluded
  // site is beyond radius from center; this triangle bound therefore certifies completeness.
  let certificate=length(site)+2.0*greatest+slack;
  if(!isFiniteValue(greatest)||!isFiniteValue(certificate)){invalid=true;}
  else if(certificate>=radius){candidateOverflow=true;}
 }
 let record=cellsOffset+4u*(index*${c}u+piece);
 cells[record+1u]=select(size,0xffffffffu,invalid);
 if(!invalid&&size>${o}u){overflow=true;}
 if(!invalid&&size<=${o}u){
  cells[record]=size;
  for(var vertex=0u;vertex<size;vertex++){
   let destination=positionsOffset+2u*((index*${c}u+piece)*${o}u+vertex);
   let point=polygon[vertex]+center;invalid=invalid||!isFiniteValue(point.x)||!isFiniteValue(point.y);positions[destination]=point.x;positions[destination+1u]=point.y;
  }
 }
}
}
let valid=!candidateOverflow&&!overflow&&!invalid;
buildingStatus[rowBase]=u32(valid);buildingStatus[rowBase+1u]=u32(overflow&&!candidateOverflow&&!invalid);buildingStatus[rowBase+2u]=u32(candidateOverflow);buildingStatus[rowBase+3u]=u32(invalid);
if(!valid){for(var part=0u;part<${c}u;part++){
 let record=cellsOffset+4u*(index*${c}u+part);cells[record]=0u;
 if(part<ownCount*${n.enclosurePieceCount}u&&(candidateOverflow||invalid)){cells[record+1u]=0xffffffffu;}
}}`})}var X=1e6,Z=16777216,Q=8192,ee=2147483647;function te(p,m){let h=m.id??`building-tessellation`,v=m.maximumSamplesPerBuilding??8,y=m.candidateCapacity??128,b=m.cellVertexCapacity??12,x=m.maximumGridVisits??2048,S=u(h,m.spatialContext);s(h,`buildingCount`,m.buildingCount,1,X),s(h,`sampleCapacity`,m.sampleCapacity,1,Z),s(h,`maximumSamplesPerBuilding`,v,1,32),s(h,`candidateCapacity`,y,1,256),s(h,`cellVertexCapacity`,b,3,128),s(h,`maximumGridVisits`,x,1,Q),s(h,`enclosure.length`,m.enclosure.length,3,32),o(h,m.enclosureMode===void 0||m.enclosureMode===`convex`||m.enclosureMode===`simple`,`enclosureMode must be 'convex' or 'simple'`),o(h,!m.enclosureRingOffsets||m.enclosureMode===`simple`,`enclosureRingOffsets requires enclosureMode 'simple'`),m.enclosureRingOffsets&&(i(m.enclosureRingOffsets,[`uint32`],`${h} enclosureRingOffsets`),s(h,`enclosureRingOffsets.length`,m.enclosureRingOffsets.length,2,33));let C=m.enclosureMode===`simple`?O(m.enclosure.length,(m.enclosureRingOffsets?.length??2)-1,1)/3:1,w=v*C;for(let e of[`sampleSpacing`,`candidateRadius`,`vertexTolerance`])o(h,Number.isFinite(m[e])&&Math.fround(m[e])>0&&Number.isFinite(Math.fround(m[e])),`${e} must be a positive finite float32 number`);o(h,m.segments.length>0&&m.segmentBuildingIds.length===m.segments.length,`segments must have at least 1 row and segmentBuildingIds one row per segment`),i(m.segments,[`float32x4`],`${h} segments`),i(m.segmentBuildingIds,[`uint32`],`${h} segmentBuildingIds`),i(m.enclosure,[`float32x2`],`${h} enclosure`),m.parameters&&(i(m.parameters,[`float32`],`${h} parameters`),r(h,`parameters`,m.parameters,3)),l(h,p,[m.segments,m.segmentBuildingIds,m.enclosure,m.enclosureRingOffsets,m.parameters]),g(p,h,{segments:m.segments,segmentBuildingIds:m.segmentBuildingIds,enclosure:m.enclosure,enclosureRingOffsets:m.enclosureRingOffsets,parameters:m.parameters},m.output,[[`output.positions`,`output.cells`,`output.buildingStatus`,`parameters`,...m.enclosureMode===`simple`?[]:[`enclosure`]]]);let T=m.buildingCount*w,E=T*b;a(h,`buildingCount * piecesPerBuilding * cellVertexCapacity`,E,ee);for(let[e,t,n,a]of[[`positions`,`float32x2`,E,m.output?.positions],[`cells`,`uint32x4`,T,m.output?.cells],[`buildingStatus`,`uint32x4`,m.buildingCount,m.output?.buildingStatus],[`samplingStatus`,`uint32x4`,1,m.output?.samplingStatus],...Object.entries(m.output?.status??{}).map(([e,t])=>[e,`uint32`,1,t])])a&&(i(a,[t],`${h} output.${e}`),r(h,`output.${e}`,a,n));c(h,768);for(let[e,t,n]of[[`segments`,m.segments.length*16,`reduce segments length`],[`segment offsets`,(m.segments.length+1)*4,`reduce segments length`],[`sample coordinates and owners`,m.sampleCapacity*12,`reduce sampleCapacity`],[`sorted coordinate index`,W(m.sampleCapacity).byteLength,`reduce sampleCapacity`],[`building index`,(m.buildingCount+5+m.sampleCapacity)*4,`reduce buildingCount or sampleCapacity`],[`positions`,E*8,`reduce buildingCount, maximumSamplesPerBuilding or cellVertexCapacity`],[`cells`,T*16,`reduce buildingCount or maximumSamplesPerBuilding`],[`building status`,m.buildingCount*16,`reduce buildingCount`],[`enclosure triangulation scratch`,k(m.enclosure.length,(m.enclosureRingOffsets?.length??2)-1,1)*16,`reduce enclosure length or ring count`],[`enclosure triangles`,(1+3*C)*8,`reduce enclosure length or ring count`]])_(p.device,h,e,t,n);let D=m.output,A=f(p,`${h}-local-positions`,`float32x2`,E,D?.positions),j=f(p,`${h}-local-cells`,`uint32x4`,T,D?.cells),M=f(p,`${h}-local-building-status`,`uint32x4`,m.buildingCount,D?.buildingStatus),N=f(p,`${h}-local-sampling-status`,`uint32x4`,1,D?.samplingStatus),P=e=>f(p,`${h}-local-${e}`,`uint32`,1,D?.status?.[e]),F={count:P(`count`),requiredCount:P(`requiredCount`),overflow:P(`overflow`),candidateOverflow:P(`candidateOverflow`),invalidCount:P(`invalidCount`)},I=m.parameters??t(p,`${h}-local-parameters`,`float32`,3);m.parameters||p.add(e(p,{id:`${h}-local-default-parameters`,operation:h,invocationCount:1,bindings:[{name:`parameters`,view:I,type:`f32`,access:`read_write`}],body:`parameters[parametersOffset]=${d(m.sampleSpacing)};
parameters[parametersOffset+1u]=${d(m.candidateRadius)};
parameters[parametersOffset+2u]=${d(m.vertexTolerance)};`}));let L=H(p,{...m,parameters:I,id:`${h}-local`,samplingStatus:N}),R=G(p,{id:`${h}-local-index`,positions:L.samples,count:N}),z=m.enclosureMode===`simple`?J(p,{id:`${h}-local-enclosure`,positions:m.enclosure,ringOffsets:m.enclosureRingOffsets,triangleCapacity:C}):m.enclosure;p.add(Y(p,{id:`${h}-local-cells`,buildingCount:m.buildingCount,maximumSamplesPerBuilding:v,candidateCapacity:y,cellVertexCapacity:b,maximumGridVisits:x,parameters:I,enclosure:z,simpleEnclosure:m.enclosureMode===`simple`,enclosurePieceCount:C,pointIndex:R,sampleData:L.sampleData,buildingIndex:L.buildingIndex,positions:A,cells:j,buildingStatus:M}));let B=[`validity`,`overflow`,`candidateOverflow`,`invalidCount`].map(e=>t(p,`${h}-local-${e}-rows`,`uint32`,m.buildingCount));p.add(e(p,{id:`${h}-local-status-rows`,operation:h,invocationCount:m.buildingCount,bindings:[{name:`records`,view:M,type:`u32`,access:`read`},...B.map((e,t)=>({name:`flag${t}`,view:e,type:`u32`,access:`read_write`}))],body:B.map((e,t)=>`flag${t}[flag${t}Offset+index]=records[recordsOffset+4u*index+${t}u];`).join(`
`)}));let V=[F.count,F.overflow,F.candidateOverflow,F.invalidCount];return B.forEach((e,t)=>p.add(new n({id:`${h}-local-status-${t}`,input:e,output:V[t],operation:t===0||t===3?`sum`:`max`}).getCommandNodes(p))),p.add(e(p,{id:`${h}-local-required-count`,operation:h,invocationCount:1,bindings:[{name:`required`,view:F.requiredCount,type:`u32`,access:`read_write`}],body:`required[requiredOffset]=${m.buildingCount}u;`})),{outputs:{positions:A,cells:j,buildingStatus:M},status:{...F,sampling:N},intermediates:{...L,parameters:I},estimatedCommandNodeCount:768,provenance:{segments:m.segments,segmentBuildingIds:m.segmentBuildingIds,enclosure:m.enclosure,enclosureRingOffsets:m.enclosureRingOffsets,enclosurePieceCount:C,piecesPerBuilding:w,spatialContext:S,sampleSpacing:m.sampleSpacing,candidateRadius:m.candidateRadius,vertexTolerance:m.vertexTolerance,maximumSamplesPerBuilding:v,candidateCapacity:y,cellVertexCapacity:b,maximumGridVisits:x,mode:`local`,representation:`sample-cell-union`}}}var ne=16777216;function re(e,t){return t.mode===`local`||`candidateRadius`in t&&t.mode!==`reference`?te(e,t):ie(e,t)}function ie(r,a){let c=a.id??`building-tessellation`;i(a.segments,[`float32x4`],`${c} segments`),i(a.segmentBuildingIds,[`uint32`],`${c} segmentBuildingIds`),i(a.enclosure,[`float32x2`],`${c} enclosure`),a.enclosureRingOffsets&&i(a.enclosureRingOffsets,[`uint32`],`${c} enclosureRingOffsets`);let l=a.segments.length,u=a.enclosure.length,y=a.enclosureMode===`simple`,b=a.sampleCapacity,x=b+u;o(c,l>0&&a.segmentBuildingIds.length===l,`segments must have at least 1 row and segmentBuildingIds one row per segment`),s(c,`buildingCount`,a.buildingCount,1,64),o(c,u>=3&&u<=32,`enclosure must have between 3 and 32 rows`),s(c,`sampleCapacity`,b,1,128),o(c,Number.isFinite(Math.fround(a.sampleSpacing))&&Math.fround(a.sampleSpacing)>0,`sampleSpacing must be a positive finite float32 number`),o(c,Number.isFinite(Math.fround(a.vertexTolerance))&&Math.fround(a.vertexTolerance)>0,`vertexTolerance must be a positive finite float32 number`),o(c,Object.values(a.capacity).every(e=>Number.isSafeInteger(e)&&e>0),`capacity entries (intersections, nodedSegments, vertices, rings) must be positive integers`),o(c,!a.enclosureRingOffsets||y&&a.enclosureRingOffsets.length>=2,`enclosureRingOffsets needs enclosureMode 'simple' and at least 2 rows`),g(r,c,{segments:a.segments,segmentBuildingIds:a.segmentBuildingIds,enclosure:a.enclosure,enclosureRingOffsets:a.enclosureRingOffsets},a.output);let C=Math.max(b*(x+1),y?a.capacity.vertices+u+a.capacity.rings+(a.enclosureRingOffsets?.length??2)-1:0),w=C+a.capacity.intersections,E=Math.max(a.capacity.nodedSegments+C,2*w);for(let[e,t,n]of[[`point grid`,a.searchStructure===`grid`?m(b).byteLength:4,`reduce sampleCapacity`],[`segments`,l*16,`reduce segments length`],[`segmentBuildingIds`,l*4,`reduce segments length`],[`sample scratch`,b*8,`reduce sampleCapacity`],[`segment scratch`,l*4,`reduce segments length`],[`fragments and per-building packed positions`,b*x*8,`reduce sampleCapacity`],[`per-building offsets`,(b+1)*4,`reduce sampleCapacity`],[`complete`,a.buildingCount*4,`reduce buildingCount`],[`aligned building parameters`,a.buildingCount*256,`reduce buildingCount`],[`overlay hot-pixel scratch`,w*128,`reduce capacity.intersections or the overlay input`],[`overlay candidate scratch`,E*8,`reduce capacity.intersections or capacity.nodedSegments`],[`overlay boundary`,a.capacity.nodedSegments*16,`reduce capacity.nodedSegments`],[`overlay positions`,a.capacity.vertices*8,`reduce capacity.vertices`],[`overlay offsets`,(a.capacity.rings+1)*4,`reduce capacity.rings`]])_(r.device,c,e,t,n);let D=(e,t,n)=>f(r,`${c}-${e}`,t,n),O=a.output,k=f(r,`${c}-samples`,`float32x2`,b,O?.samples),A=f(r,`${c}-owners`,`uint32`,b,O?.owners),M=f(r,`${c}-sampling-status`,`uint32x4`,1,O?.samplingStatus),N=f(r,`${c}-fragments`,`float32x2`,b*x,O?.fragments),P=D(`enclosure-offsets`,`uint32`,2),F=D(`enclosure-indices`,`uint32`,2),I=D(`enclosure-polygons`,`uint32`,2);y&&r.add(e(r,{id:`${c}-enclosure-layout`,operation:c,invocationCount:1,bindings:[{name:`offsets`,view:P,type:`u32`,access:`read_write`},{name:`indices`,view:F,type:`u32`,access:`read_write`},{name:`polygons`,view:I,type:`u32`,access:`read_write`}],body:`offsets[offsetsOffset]=0u;offsets[offsetsOffset+1u]=${u}u;indices[indicesOffset]=0u;indices[indicesOffset+1u]=1u;polygons[polygonsOffset]=0u;polygons[polygonsOffset+1u]=${a.enclosureRingOffsets?a.enclosureRingOffsets.length-1:1}u;`}));let L=f(r,`${c}-fragment-counts`,`uint32`,b,O?.fragmentCounts),R=t(r,`${c}-sample-counts`,`uint32`,l),z=t(r,`${c}-sample-offsets`,`uint32`,l),B=t(r,`${c}-invalid-segments`,`uint32`,l),V=t(r,`${c}-required-samples`,`uint32`,1),H=t(r,`${c}-invalid-segment-count`,`uint32`,1);r.add(e(r,{id:`${c}-sample-counts`,operation:c,invocationCount:l,declarations:T,bindings:[{name:`segments`,view:a.segments,type:`f32`,access:`read`},{name:`buildingIds`,view:a.segmentBuildingIds,type:`u32`,access:`read`},{name:`counts`,view:R,type:`u32`,access:`read_write`},{name:`invalidFlags`,view:B,type:`u32`,access:`read_write`}],body:`let base=segmentsOffset+4u*index;
let start=vec2<f32>(segments[base],segments[base+1u]);
let end=vec2<f32>(segments[base+2u],segments[base+3u]);
let owner=buildingIds[buildingIdsOffset+index];
let length=distance(start,end);
var count=0u;
var invalid=!isFiniteValue(start.x)||!isFiniteValue(start.y)||!isFiniteValue(end.x)||!isFiniteValue(end.y)||!isFiniteValue(length)||owner>=${a.buildingCount}u||length==0.0;
if(!invalid){
  let countValue=ceil(length/${d(a.sampleSpacing)});
  if(!isFiniteValue(countValue)||countValue>${ne}.0){invalid=true;}else{count=u32(countValue);}
}
counts[countsOffset+index]=count;
invalidFlags[invalidFlagsOffset+index]=u32(invalid);`})),r.add(S(r,{id:`${c}-sample-total`,operation:c,counts:R,offsets:z,rowCount:l,total:V})),r.add(new n({id:`${c}-invalid-segment-total`,input:B,output:H,operation:`sum`}).getCommandNodes(r)),r.add(e(r,{id:`${c}-clear-samples`,operation:c,invocationCount:b,bindings:[{name:`samples`,view:k,type:`f32`,access:`read_write`},{name:`owners`,view:A,type:`u32`,access:`read_write`}],body:`owners[ownersOffset+index]=0xffffffffu;
samples[samplesOffset+2u*index]=0.0;
samples[samplesOffset+2u*index+1u]=0.0;`})),r.add(e(r,{id:`${c}-write-samples`,operation:c,invocationCount:l,bindings:[{name:`segments`,view:a.segments,type:`f32`,access:`read`},{name:`buildingIds`,view:a.segmentBuildingIds,type:`u32`,access:`read`},{name:`counts`,view:R,type:`u32`,access:`read`},{name:`offsets`,view:z,type:`u32`,access:`read`},{name:`samples`,view:k,type:`f32`,access:`read_write`},{name:`owners`,view:A,type:`u32`,access:`read_write`}],body:`let count=counts[countsOffset+index];
let first=offsets[offsetsOffset+index];
let base=segmentsOffset+4u*index;
let start=vec2<f32>(segments[base],segments[base+1u]);
let end=vec2<f32>(segments[base+2u],segments[base+3u]);
for(var part=0u;part<min(count,${b}u);part++){
  if(first>=${b}u||part>=${b}u-first){break;}
  let position=mix(start,end,f32(part)/f32(count));
  let sampleRow=first+part;
  samples[samplesOffset+2u*sampleRow]=position.x;
  samples[samplesOffset+2u*sampleRow+1u]=position.y;
  owners[ownersOffset+sampleRow]=buildingIds[buildingIdsOffset+index];
}`})),r.add(e(r,{id:`${c}-sampling-status`,operation:c,invocationCount:1,bindings:[{name:`required`,view:V,type:`u32`,access:`read`},{name:`invalid`,view:H,type:`u32`,access:`read`},{name:`status`,view:M,type:`u32`,access:`read_write`}],body:`let requiredTotal=required[requiredOffset];
status[statusOffset]=min(requiredTotal,${b}u);
status[statusOffset+1u]=requiredTotal;
status[statusOffset+2u]=u32(requiredTotal>${b}u);
status[statusOffset+3u]=invalid[invalidOffset];`}));let U=a.searchStructure===`grid`?v(r,{id:`${c}-sample-grid`,positions:k,count:M}):void 0,W=` if(other==index||size==0u){continue;}let displacement=readSample(other)-site;let squared=dot(displacement,displacement);
 if(squared==0.0){if(owners[ownersOffset+other]!=owners[ownersOffset+index]){valid=false;}else if(other<index){size=0u;}continue;}
 let threshold=0.5*squared;var outputSize=0u;
 for(var vertex=0u;vertex<size;vertex++){let start=polygon[vertex];let end=polygon[(vertex+1u)%size];let startProducts=(start-site)*displacement;let endProducts=(end-site)*displacement;
 let startSide=startProducts.x+startProducts.y-threshold;let endSide=endProducts.x+endProducts.y-threshold;
 if(!isFiniteValue(startProducts.x)||!isFiniteValue(startProducts.y)||!isFiniteValue(endProducts.x)||!isFiniteValue(endProducts.y)){valid=false;break;}let inside=startSide<=0.0;let nextInside=endSide<=0.0;
 if(inside){if(outputSize<${x}u){scratch[outputSize]=start;}outputSize++;}
 if(!isFiniteValue(startSide)||!isFiniteValue(endSide)||!isFiniteValue(squared)){valid=false;break;}
 if(inside!=nextInside){let sideScale=max(abs(startSide),abs(endSide));let scaledStart=startSide/sideScale;let denominator=scaledStart-endSide/sideScale;let fraction=scaledStart/denominator;
 let intersection=(1.0-fraction)*start+fraction*end;
 valid=valid&&isFiniteValue(fraction)&&isFiniteValue(denominator)&&denominator!=0.0&&isFiniteValue(intersection.x)&&isFiniteValue(intersection.y);
 if(outputSize<${x}u){scratch[outputSize]=intersection;}outputSize++;}
 }
 if(outputSize>${x}u){valid=false;break;}size=outputSize;for(var vertex=0u;vertex<size;vertex++){polygon[vertex]=scratch[vertex];}`;r.add(e(r,{id:`${c}-clip-cells`,operation:c,invocationCount:b,workgroupSize:2,declarations:T+(U?h(U):``)+`
fn readSample(row:u32)->vec2<f32>{return vec2<f32>(samples[samplesOffset+2u*row],samples[samplesOffset+2u*row+1u]);}
fn readEnclosure(row:u32)->vec2<f32>{return vec2<f32>(enclosure[enclosureOffset+2u*row],enclosure[enclosureOffset+2u*row+1u]);}
fn crossVectors(first:vec2<f32>,second:vec2<f32>)->f32{return first.x*second.y-first.y*second.x;}`,bindings:[...U?[{name:`pointGrid`,view:U.data,type:`u32`,access:`read`}]:[],{name:`samples`,view:k,type:`f32`,access:`read`},{name:`owners`,view:A,type:`u32`,access:`read`},{name:`status`,view:M,type:`u32`,access:`read`},{name:`enclosure`,view:a.enclosure,type:`f32`,access:`read`},{name:`fragments`,view:N,type:`f32`,access:`read_write`},{name:`counts`,view:L,type:`u32`,access:`read_write`}],body:`counts[countsOffset+index]=0u;if(index>=status[statusOffset]||status[statusOffset+2u]!=0u||status[statusOffset+3u]!=0u){return;}
var polygon:array<vec2<f32>,${x}>;var scratch:array<vec2<f32>,${x}>;var size=${u}u;let site=readSample(index);var valid=true;var area=0.0;
for(var vertex=0u;vertex<${u}u;vertex++){let position=readEnclosure(vertex);let next=readEnclosure((vertex+1u)%${u}u);let after=readEnclosure((vertex+2u)%${u}u);polygon[vertex]=position;valid=valid&&isFiniteValue(position.x)&&isFiniteValue(position.y)&&crossVectors(next-position,after-next)>=0.0;area+=crossVectors(position,next);}
valid=valid&&area>0.0;
${y?`valid=abs(area)>0.0;var lower=vec2<f32>(3.402823e38);var upper=vec2<f32>(-3.402823e38);
for(var vertex=0u;vertex<${u}u;vertex++){let point=readEnclosure(vertex);valid=valid&&isFiniteValue(point.x)&&isFiniteValue(point.y);lower=min(lower,point);upper=max(upper,point);}
polygon[0u]=lower;polygon[1u]=vec2<f32>(upper.x,lower.y);polygon[2u]=upper;polygon[3u]=vec2<f32>(lower.x,upper.y);size=4u;`:``}
${U?p(U,W):`for(var other=0u;other<status[statusOffset];other++){${W}}`}

if(!valid){counts[countsOffset+index]=0xffffffffu;return;}
if(size<3u){size=0u;}counts[countsOffset+index]=size;
for(var vertex=0u;vertex<size;vertex++){let destination=fragmentsOffset+2u*(index*${x}u+vertex);fragments[destination]=polygon[vertex].x;fragments[destination+1u]=polygon[vertex].y;}`}));let G=f(r,`${c}-complete`,`uint32`,a.buildingCount,O?.complete),K=D(`building-parameters`,`uint32`,a.buildingCount*64);r.add(e(r,{id:`${c}-building-parameters`,operation:c,invocationCount:a.buildingCount,bindings:[{name:`rows`,view:K,type:`u32`,access:`read_write`}],body:`rows[rowsOffset+64u*index]=index;`}));let q=[];for(let t=0;t<a.buildingCount;t++){let n=`building-${t}`,i=r.createDataView(K.buffer,{format:`uint32`,byteOffset:t*256,length:1}),o=D(`${n}-input-positions`,`float32x2`,b*x),s=D(`${n}-input-offsets`,`uint32`,b+1),l=D(`${n}-input-indices`,`uint32`,b+1);r.add(e(r,{id:`${c}-${n}-pack`,operation:c,invocationCount:1,bindings:[{name:`fragments`,view:N,type:`f32`,access:`read`},{name:`counts`,view:L,type:`u32`,access:`read`},{name:`owners`,view:A,type:`u32`,access:`read`},{name:`positions`,view:o,type:`f32`,access:`read_write`},{name:`offsets`,view:s,type:`u32`,access:`read_write`},{name:`indices`,view:l,type:`u32`,access:`read_write`},{name:`buildingParameter`,view:i,type:`u32`,access:`read`}],body:`var valid=true;for(var sampleRow=0u;sampleRow<${b}u;sampleRow++){valid=valid&&counts[countsOffset+sampleRow]!=0xffffffffu;}
var written=0u;offsets[offsetsOffset]=0u;indices[indicesOffset]=0u;
for(var sampleRow=0u;sampleRow<${b}u;sampleRow++){if(valid&&owners[ownersOffset+sampleRow]==buildingParameter[buildingParameterOffset]){for(var vertex=0u;vertex<counts[countsOffset+sampleRow];vertex++){let source=fragmentsOffset+2u*(sampleRow*${x}u+vertex);positions[positionsOffset+2u*written]=fragments[source];positions[positionsOffset+2u*written+1u]=fragments[source+1u];written++;}}offsets[offsetsOffset+sampleRow+1u]=written;indices[indicesOffset+sampleRow+1u]=sampleRow+1u;}`}));let u=e=>({count:D(`${n}-${e}-count`,`uint32`,1),overflow:D(`${n}-${e}-overflow`,`uint32`,1),candidateOverflow:D(`${n}-${e}-candidates`,`uint32`,1)}),d=e=>({...u(e),requiredRingCount:D(`${n}-${e}-required`,`uint32`,1)}),f=e=>({...u(e),requiredCount:D(`${n}-${e}-required`,`uint32`,1)}),p=e=>({geometry:{kind:`polygons`,positions:D(`${n}-${e}positions`,`float32x2`,a.capacity.vertices),featureOffsets:D(`${n}-${e}features`,`uint32`,a.capacity.rings+1),polygonOffsets:D(`${n}-${e}polygons`,`uint32`,a.capacity.rings+1),ringOffsets:D(`${n}-${e}rings`,`uint32`,a.capacity.rings+1)},status:d(`${e}geometry`),boundary:{endpoints:D(`${n}-${e}boundary`,`float32x4`,a.capacity.nodedSegments),operandIds:D(`${n}-${e}operands`,`uint32`,a.capacity.nodedSegments),sourceFeatureIds:D(`${n}-${e}boundary-sources`,`uint32`,a.capacity.nodedSegments),status:f(`${e}boundary`)}}),m=O?.overlays?.[t],h=!y&&m?m:p(``);r.add(new j({id:`${c}-${n}-dissolve`,left:{kind:`polygons`,positions:o,featureOffsets:l,polygonOffsets:l,ringOffsets:s},operation:`dissolve`,capacity:a.capacity,vertexTolerance:a.vertexTolerance,output:h}));let g=h;y&&(h=m??p(`clip-`),r.add(new j({id:`${c}-${n}-clip-enclosure`,left:g.geometry,right:{kind:`polygons`,positions:a.enclosure,featureOffsets:F,polygonOffsets:I,ringOffsets:a.enclosureRingOffsets??P},operation:`intersection`,capacity:a.capacity,vertexTolerance:a.vertexTolerance,output:h})));let _=h;q.push(_),r.add(e(r,{id:`${c}-${n}-evidence`,operation:c,invocationCount:1,bindings:[{name:`sampling`,view:M,type:`u32`,access:`read`},{name:`counts`,view:L,type:`u32`,access:`read`},{name:`overflow`,view:_.status.overflow,type:`u32`,access:`read`},{name:`candidateOverflow`,view:_.status.candidateOverflow,type:`u32`,access:`read`},{name:`dissolveOverflow`,view:g.status.overflow,type:`u32`,access:`read`},{name:`dissolveCandidates`,view:g.status.candidateOverflow,type:`u32`,access:`read`},{name:`complete`,view:G,type:`u32`,access:`read_write`},{name:`buildingParameter`,view:i,type:`u32`,access:`read`}],body:`var valid=dissolveOverflow[dissolveOverflowOffset]==0u&&dissolveCandidates[dissolveCandidatesOffset]==0u&&sampling[samplingOffset+2u]==0u&&sampling[samplingOffset+3u]==0u&&overflow[overflowOffset]==0u&&candidateOverflow[candidateOverflowOffset]==0u;for(var sampleRow=0u;sampleRow<${b}u;sampleRow++){valid=valid&&counts[countsOffset+sampleRow]!=0xffffffffu;}complete[completeOffset+buildingParameter[buildingParameterOffset]]=u32(valid);`}))}return{outputs:q,intermediates:{samples:k,owners:A,fragments:N,fragmentCounts:L},status:{sampling:M,complete:G},provenance:{segments:a.segments,segmentBuildingIds:a.segmentBuildingIds,enclosure:a.enclosure,sampleSpacing:a.sampleSpacing,vertexTolerance:a.vertexTolerance,enclosurePolicy:a.enclosureMode??`convex`,mode:`reference`,searchStructure:U?`grid`:`brute-force`}}}var $=[[.35,.66,1,1],[1,.7,.3,1],[1,.42,.42,1],[.31,.82,.77,1],[.55,.85,.35,1],[.78,.58,.93,1]],ae=M({parameters:B,caption:`Each building gets the territory closest to it; colored outlines are those territories.`,createInputs:({world:e})=>{let t=z(e,10,!0);return{layout:t,world:e,index:P(Float32Array.from(t.centers.flat())),carried:{building:-1,offset:[0,0]}}},build(e,t,n){let r=R(t.layout,n.fill),i=e.upload(`walls`,`float32x4`,r.segments),a=e.upload(`wall-owners`,`uint32`,r.segmentBuildingIds),o=t.layout.rings.flat(),s=Math.min(...o.map(e=>e[0]))-.07,c=Math.max(...o.map(e=>e[0]))+.07,l=Math.min(...o.map(e=>e[1]))-.07,u=Math.max(...o.map(e=>e[1]))+.07,d=e.upload(`enclosure`,`float32x2`,Float32Array.of(s,l,c,l,c,u,s,u)),f=re(e.graph,{segments:i.view,segmentBuildingIds:a.view,buildingCount:r.count,enclosure:d.view,sampleSpacing:n.sampleSpacing,sampleCapacity:128,vertexTolerance:.001,capacity:{intersections:4096,nodedSegments:4096,vertices:1024,rings:64}}).outputs.map((t,n)=>L(e,`territory-${n}`,{positions:I(t.geometry.positions),featureOffsets:I(t.geometry.featureOffsets),polygonOffsets:I(t.geometry.polygonOffsets),ringOffsets:I(t.geometry.ringOffsets)},I(t.status.count),{closed:!0})),p=e.allocate(`wall-segments`,`float32x4`,r.segments.length/4);return{outputs:{walls:p,territories:f},update:e=>{let n=R(t.layout,e.fill,0,t.carried);i.write(n.segments),p.write(n.segments)}}},rebuildOn:[`sampleSpacing`],interaction:{kind:`drag`,hint:`move a building`,rest:({inputs:e,toUnit:t})=>t(e.layout.centers[0]),resolve(e,{inputs:t}){let n=F(t.index,e.x,e.y,t.carried.building);if(n<0)return{};let[r,i]=t.layout.centers[n],a=e.x-r,o=e.y-i,s=N((.3-Math.hypot(a,o))/.22,0,1),c=[a*s,o*s],l=n!==t.carried.building||c[0]!==t.carried.offset[0]||c[1]!==t.carried.offset[1];return t.carried.building=n,t.carried.offset=c,{rerun:l,glyphs:[{shape:`polyline`,points:t.layout.rings[n].flatMap(([e,t])=>[e+c[0],t+c[1]]),closed:!0},{shape:`crosshair`,at:[e.x,e.y]}],readout:`building ${n} · territories re-form`}}},draw:({outputs:e})=>[...e.territories.map((e,t)=>({layer:`lines`,segments:e.segments,color:$[t%$.length],radius:2.6})),{layer:`lines`,context:!0,segments:e.walls,radius:3}]});export{ae as default};
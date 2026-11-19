import{r as e}from"./kernel-C3tfQOn4.js";import{c as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{u as r}from"./wgsl-common-DfyXiYB0.js";import{t as i}from"./segment-bvh-CDgckfAv.js";var a=16777216,o=2;function s(e,t){return e===`brute-force`||e===`bvh`?e:t>4096?`bvh`:`brute-force`}function c(a,s){let c=s.segments.length,l=e(a,`${s.id}-bvh-minima`,`float32x2`,c),u=e(a,`${s.id}-bvh-maxima`,`float32x2`,c);a.add(n(a,{id:`${s.id}-bvh-bounds`,operation:s.operation,invocationCount:c,declarations:r,bindings:[{name:`segments`,view:s.segments,type:`f32`,access:`read`},{name:`minima`,view:l,type:`f32`,access:`read_write`},{name:`maxima`,view:u,type:`f32`,access:`read_write`}],body:`let base=segmentsOffset+4u*index;let start=vec2<f32>(segments[base],segments[base+1u]);let end=vec2<f32>(segments[base+2u],segments[base+3u]);
let finiteSegment=isFiniteValue(start.x)&&isFiniteValue(start.y)&&isFiniteValue(end.x)&&isFiniteValue(end.y);
let largest=max(max(abs(start.x),abs(start.y)),max(abs(end.x),abs(end.y)));
let padding=${t(2*s.tolerance)}+1.0e-6*largest;
let lower=min(start,end)-vec2<f32>(padding);let upper=max(start,end)+vec2<f32>(padding);
minima[minimaOffset+2u*index]=select(1.0,lower.x,finiteSegment);minima[minimaOffset+2u*index+1u]=select(1.0,lower.y,finiteSegment);
maxima[maximaOffset+2u*index]=select(0.0,upper.x,finiteSegment);maxima[maximaOffset+2u*index+1u]=select(0.0,upper.y,finiteSegment);`}));let d=o;for(;d<c;)d*=2;let{bvh:f,nodes:p}=i(a,{id:s.id,operation:s.operation,minima:l,maxima:u,leafCapacity:d,spatialSort:!0});return a.add(p),f}function l(e){return[{name:`nodeMinima`,view:e.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:e.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:e.leafIds,type:`u32`,access:`read`}]}function u(e,n,r,i=`input`){let a=t(n),o=i===`local`?`var<private> urbanFrameOrigin:vec2<f32>;`:``,s=i===`local`?`-vec4<f32>(urbanFrameOrigin,urbanFrameOrigin)`:``,c=i===`local`?`-urbanFrameOrigin`:``,l=`
${o}
fn readSegment(row:u32)->vec4<f32>{let base=segmentsOffset+row*4u;return vec4<f32>(segments[base],segments[base+1u],segments[base+2u],segments[base+3u])${s};}
fn crossVectors(first:vec2<f32>,second:vec2<f32>)->f32{return first.x*second.y-first.y*second.x;}
// The offset from the segment start is formed first, so the result does not lose digits to the
// magnitude of the coordinates.
fn distanceToSegment(position:vec2<f32>,segment:vec4<f32>)->f32{
 let direction=segment.zw-segment.xy;let offset=position-segment.xy;let squared=dot(direction,direction);var fraction=0.0;
 if(squared>0.0){fraction=clamp(dot(offset,direction)/squared,0.0,1.0);}
 return length(offset-fraction*direction);
}
// Returns (candidate distance along the ray, 1.0 when the segment blocks the ray, else 0.0).
fn getSegmentCandidate(origin:vec2<f32>,direction:vec2<f32>,maximum:f32,segmentIndex:u32)->vec2<f32>{
 let segment=readSegment(segmentIndex);let edge=segment.zw-segment.xy;let displacement=segment.xy-origin;let denominator=crossVectors(direction,edge);var candidate=maximum;var intersects=false;
 if(distanceToSegment(origin,segment)<=${a}){candidate=0.0;intersects=true;}
 else if(abs(denominator)>${a}){
  let rayDistance=crossVectors(displacement,edge)/denominator;let fraction=crossVectors(displacement,direction)/denominator;
  if(rayDistance>=0.0&&fraction>=0.0&&fraction<=1.0){candidate=rayDistance;intersects=true;}
 }else if(abs(crossVectors(displacement,direction))<=${a}){
  let first=dot(displacement,direction);let second=dot(segment.zw-origin,direction);
  if(max(first,second)>=0.0){candidate=max(0.0,min(first,second));intersects=true;}
 }
 return vec2<f32>(candidate,f32(intersects));
}`;return r?`${l}
// Entry distance of the ray into a node box, or a huge value when the ray misses it or the box is empty.
fn getBoxEntry(node:u32,origin:vec2<f32>,direction:vec2<f32>,limit:f32)->f32{
 let component=node*2u;
 let boxLower=vec2<f32>(nodeMinima[nodeMinimaOffset+component],nodeMinima[nodeMinimaOffset+component+1u]);
 let boxUpper=vec2<f32>(nodeMaxima[nodeMaximaOffset+component],nodeMaxima[nodeMaximaOffset+component+1u]);
 if(!(boxLower.x<=boxUpper.x&&boxLower.y<=boxUpper.y)){return 3.0e38;}
 let lower=boxLower${c};let upper=boxUpper${c};
 var entry=0.0;var exit=limit;
 for(var axis=0u;axis<2u;axis++){
  if(abs(direction[axis])<1.0e-20){if(origin[axis]<lower[axis]||origin[axis]>upper[axis]){return 3.0e38;}}
  else{let first=(lower[axis]-origin[axis])/direction[axis];let second=(upper[axis]-origin[axis])/direction[axis];entry=max(entry,min(first,second));exit=min(exit,max(first,second));}
 }
 return select(3.0e38,entry,entry<=exit);
}
fn castRay(origin:vec2<f32>,direction:vec2<f32>,maximum:f32)->vec2<f32>{
 var nearest=maximum;var hit=-1.0;var stack:array<u32,64>;var depth=1u;stack[0u]=0u;
 loop{
  if(depth==0u){break;}
  depth--;let node=stack[depth];
  // Rounding slack on the pruning distance keeps tied and near-tied segments in the search.
  let limit=nearest+nearest*1.0e-5+${t(4*n)};
  if(getBoxEntry(node,origin,direction,limit)>=3.0e38){continue;}
  if(node<${r.internalNodeCount}u){stack[depth]=node*2u+2u;stack[depth+1u]=node*2u+1u;depth+=2u;continue;}
  let row=leafIds[leafIdsOffset+node-${r.internalNodeCount}u];
  if(row>=${e}u){continue;}
  let blocked=getSegmentCandidate(origin,direction,maximum,row);
  if(blocked.y>0.0&&blocked.x<=nearest&&(blocked.x<nearest||hit<0.0||f32(row)<hit)){nearest=blocked.x;hit=f32(row);}
 }
 return vec2<f32>(nearest,hit);
}`:`${l}
fn castRay(origin:vec2<f32>,direction:vec2<f32>,maximum:f32)->vec2<f32>{
 var nearest=maximum;var hit=-1.0;
 for(var segmentIndex=0u;segmentIndex<${e}u;segmentIndex++){
  let blocked=getSegmentCandidate(origin,direction,maximum,segmentIndex);
  if(blocked.y>0.0&&blocked.x<=nearest&&(blocked.x<nearest||hit<0.0)){nearest=blocked.x;hit=f32(segmentIndex);}
 }
 return vec2<f32>(nearest,hit);
}`}export{s as a,u as i,c as n,l as r,a as t};
import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-reduction-tNgOVQ_C.js";import{o as n,u as r}from"./recipe-utils-CICB7qiS.js";import{u as i}from"./wgsl-common-DfyXiYB0.js";import{i as a,r as o}from"./urban-ray-utils-BSeNtKp8.js";function s(s,c){let l=r(s,`${c.id}-viewpoint-classifications`,`uint32`,c.positions.length,c.output),u=`let inside=${c.region?.fillRule===`nonzero`?`winding!=0`:`(abs(winding)%2)==1`};
let allowed=${c.region?c.region.freeSpace===`inside`?`inside`:`!inside`:`true`};
classifications[classificationsOffset+index]=select(select(select(1u,0u,allowed),2u,boundary),3u,!valid);`,d=`let cross=crossVectors(segment.zw-segment.xy,point-segment.xy);
  if(segment.y<=point.y&&segment.w>point.y&&cross>0.0){winding++;}
  if(segment.y>point.y&&segment.w<=point.y&&cross<0.0){winding--;}`,f=c.obstacleBVH;if(f){let r=c.segments.length,p=e(s,`${c.id}-non-finite-segment-flags`,`uint32`,r),m=e(s,`${c.id}-non-finite-segment-count`,`uint32`,1);return s.add(n(s,{id:`${c.id}-non-finite-segment-flags`,operation:c.id,invocationCount:r,declarations:i,bindings:[{name:`segments`,view:c.segments,type:`f32`,access:`read`},{name:`flags`,view:p,type:`u32`,access:`read_write`}],body:`let base=segmentsOffset+4u*index;
flags[flagsOffset+index]=u32(!(isFiniteValue(segments[base])&&isFiniteValue(segments[base+1u])&&isFiniteValue(segments[base+2u])&&isFiniteValue(segments[base+3u])));`})),s.add(new t({id:`${c.id}-non-finite-segment-count`,input:p,output:m,operation:`sum`}).getCommandNodes(s)),s.add(n(s,{id:`${c.id}-classify-viewpoints`,operation:c.id,invocationCount:c.positions.length,declarations:i+a(r,c.tolerance,f),bindings:[{name:`positions`,view:c.positions,type:`f32`,access:`read`},{name:`segments`,view:c.segments,type:`f32`,access:`read`},...o(f),{name:`nonFiniteCount`,view:m,type:`u32`,access:`read`},{name:`classifications`,view:l,type:`u32`,access:`read_write`}],body:`let point=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
let valid=isFiniteValue(point.x)&&isFiniteValue(point.y)&&nonFiniteCount[nonFiniteCountOffset]==0u;
var boundary=false;
var winding=0;
var stack:array<u32,64>;var depth=1u;stack[0u]=0u;
loop{
  if(depth==0u){break;}
  depth--;let node=stack[depth];
  if(getBoxEntry(node,point,vec2<f32>(1.0,0.0),1.0e30)>=3.0e38){continue;}
  if(node<${f.internalNodeCount}u){stack[depth]=node*2u+2u;stack[depth+1u]=node*2u+1u;depth+=2u;continue;}
  let row=leafIds[leafIdsOffset+node-${f.internalNodeCount}u];
  if(row>=${r}u){continue;}
  let segment=readSegment(row);
  boundary=boundary||distanceToSegment(point,segment)<=${c.tolerance};
  ${d}
}
${u}`})),l}return s.add(n(s,{id:`${c.id}-classify-viewpoints`,operation:c.id,invocationCount:c.positions.length,declarations:i+a(c.segments.length,c.tolerance),bindings:[{name:`positions`,view:c.positions,type:`f32`,access:`read`},{name:`segments`,view:c.segments,type:`f32`,access:`read`},{name:`classifications`,view:l,type:`u32`,access:`read_write`}],body:`let point=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
var valid=isFiniteValue(point.x)&&isFiniteValue(point.y);
var boundary=false;
var winding=0;
for(var segmentIndex=0u;segmentIndex<${c.segments.length}u;segmentIndex++){
  let segment=readSegment(segmentIndex);
  valid=valid&&isFiniteValue(segment.x)&&isFiniteValue(segment.y)&&isFiniteValue(segment.z)&&isFiniteValue(segment.w);
  boundary=boundary||distanceToSegment(point,segment)<=${c.tolerance};
  ${d}
}
${u}`})),l}export{s as t};
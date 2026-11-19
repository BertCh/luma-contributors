import{t as e}from"./assert-unit-vector-BIr1r7vX.js";import{r as t}from"./kernel-C3tfQOn4.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as r,i}from"./graph-chunk-dispatch-zBt8k2-o.js";import{E as a,S as o,T as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{v as d}from"./spatial-context-DwZEd4EZ.js";import{s as f}from"./wgsl-common-DfyXiYB0.js";import{a as p,c as m,n as h,r as g,s as _,t as v}from"./spherical-limits-C7rAYBu5.js";import{t as y}from"./define-run-B-brkCe4.js";import{d as b,s as x}from"./pointer-overlay-Qh8qflcA.js";import{m as S,t as C}from"./kernels-IDOsx_br.js";import{i as w}from"./polygons--0zAAHu9.js";import{a as T,i as E,n as D,s as O}from"./spherical-surveys-Vp2aL65w.js";import{$c as k}from"./entry-preview-D70TdSbK.js";var A=class{props;id;spatialContext;referenceInside;constructor(t){this.props=t,this.id=t.id??`spherical-arc-overlay`,this.spatialContext=d(this.id,t.spatialContext??h,_),e(t.referencePoint,this.id,`referencePoint`),m(this.id,t.tolerance??1e-6,p,g);for(let e of[t.left,t.right])a(e.positions,[`float32x3`],this.id),a(e.ringOffsets,[`uint32`],this.id),s(this.id,e.ringOffsets.length>=1,`left and right ringOffsets must hold at least one entry`);o(this.id,`left plus right positions`,t.left.positions.length+t.right.positions.length,128),a(t.output.positions,[`float32x3`],this.id);for(let e of[t.output.ringOffsets,t.output.counts,t.output.status])a(e,[`uint32`],this.id);s(this.id,t.output.positions.length>=3&&t.output.positions.length<=1024,`output.positions must hold 3 to ${v} entries`),s(this.id,t.output.ringOffsets.length>=2,`output.ringOffsets must hold at least 2 entries`),s(this.id,t.output.counts.length===4&&t.output.status.length===1,`output.counts must hold 4 entries and output.status 1 entry`),u(this.id,Object.values(t.output),[t.left.positions,t.right.positions,t.left.ringOffsets,t.right.ringOffsets]);let n=t.left.referenceInside;i(this.id,t.output);let r=t.right.referenceInside;this.referenceInside=t.operation===`intersection`?n&&r:t.operation===`union`?n||r:t.operation===`difference`?n&&!r:n!==r}getCommandNodes(e){let{props:i,id:a}=this;r(e,a,{leftPositions:i.left.positions,rightPositions:i.right.positions,leftOffsets:i.left.ringOffsets,rightOffsets:i.right.ringOffsets,...i.output});for(let[t,n]of[[`packed ring offsets scratch`,4*(i.left.ringOffsets.length+i.right.ringOffsets.length)],[`arc segment scratch`,28*i.output.positions.length]])c(e.device,a,t,n,`reduce output capacity or ring count`);l(a,e,[i.left.positions,i.right.positions,i.left.ringOffsets,i.right.ringOffsets,...Object.values(i.output)]);let o=i.output.positions.length,s=t(e,`${a}-offsets`,`uint32`,i.left.ringOffsets.length+i.right.ringOffsets.length),u=t(e,`${a}-segments`,`float32`,o*7),d=[];for(let[t,r]of[[i.left,0],[i.right,i.left.ringOffsets.length]])d.push(n(e,{id:`${a}-pack-${r}`,operation:`GPUSphericalArcOverlay`,variant:`pack-offsets`,invocationCount:t.ringOffsets.length,bindings:[{name:`source`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`packed`,view:s,type:`u32`,access:`read_write`}],body:`packed[packedOffset+${r}u+index]=source[sourceOffset+index];`}));return[...d,n(e,{id:a,operation:`GPUSphericalArcOverlay`,variant:i.operation,invocationCount:1,bindings:[{name:`leftPositions`,view:i.left.positions,type:`f32`,access:`read`},{name:`rightPositions`,view:i.right.positions,type:`f32`,access:`read`},{name:`offsets`,view:s,type:`u32`,access:`read`},{name:`scratch`,view:u,type:`f32`,access:`read_write`},{name:`output`,view:i.output.positions,type:`f32`,access:`read_write`},{name:`rings`,view:i.output.ringOffsets,type:`u32`,access:`read_write`},{name:`counts`,view:i.output.counts,type:`u32`,access:`read_write`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`}],declarations:`const ARC_TOLERANCE:f32=${i.tolerance??1e-6};
${f}
const ARC_REFERENCE=vec3f(${i.referencePoint.join(`,`)});
fn arcStart(row:u32)->vec3f{let offset=scratchOffset+row*7u;return vec3f(scratch[offset],scratch[offset+1u],scratch[offset+2u]);}
fn arcEnd(row:u32)->vec3f{let offset=scratchOffset+row*7u+3u;return vec3f(scratch[offset],scratch[offset+1u],scratch[offset+2u]);}
fn arcStore(row:u32,start:vec3f,end:vec3f){let offset=scratchOffset+row*7u;scratch[offset]=start.x;scratch[offset+1u]=start.y;scratch[offset+2u]=start.z;scratch[offset+3u]=end.x;scratch[offset+4u]=end.y;scratch[offset+5u]=end.z;}
fn arcUsed(row:u32)->bool{return scratch[scratchOffset+row*7u+6u]!=0.0;}

fn arcLength(first:vec3f,second:vec3f)->f32{return guardedAtan2(length(cross(first,second)),dot(first,second));}
fn arcVertexCount(operand:u32)->u32{return select(${i.left.positions.length}u,${i.right.positions.length}u,operand==1u);}
fn arcRingCount(operand:u32)->u32{return select(${i.left.ringOffsets.length-1}u,${i.right.ringOffsets.length-1}u,operand==1u);}
fn arcOffset(operand:u32,row:u32)->u32{if(operand==0u){return offsets[offsetsOffset+row];}return offsets[offsetsOffset+${i.left.ringOffsets.length}u+row];}
fn arcPosition(operand:u32,row:u32)->vec3f{if(operand==0u){let offset=leftPositionsOffset+row*3u;return vec3f(leftPositions[offset],leftPositions[offset+1u],leftPositions[offset+2u]);}let offset=rightPositionsOffset+row*3u;return vec3f(rightPositions[offset],rightPositions[offset+1u],rightPositions[offset+2u]);}
fn arcNext(operand:u32,row:u32)->u32{for(var ring=0u;ring<arcRingCount(operand);ring++){let begin=arcOffset(operand,ring);let end=arcOffset(operand,ring+1u);if(row>=begin&&row<end){return select(row+1u,begin,row+1u==end);}}return row;}
fn arcOn(first:vec3f,second:vec3f,point:vec3f,margin:f32)->bool{let normal=normalize(cross(first,second));return abs(dot(normal,point))<=max(margin,ARC_TOLERANCE)&&dot(cross(first,point),normal)>=-margin&&dot(cross(point,second),normal)>=-margin;}
fn arcBoolean(left:bool,right:bool)->bool{return ${i.operation===`intersection`?`left&&right`:i.operation===`union`?`left||right`:i.operation===`difference`?`left&&!right`:`left!=right`};}
// Parity is anchored at a known point, so a ring may enclose more than a hemisphere.
// Alternate two-arc paths avoid rays through vertices without choosing longitude seams.
fn arcContains(operand:u32,point:vec3f)->vec2u{
  let referenceInside=select(${Number(i.left.referenceInside)}u,${Number(i.right.referenceInside)}u,operand==1u);
  for(var edge=0u;edge<arcVertexCount(operand);edge++){if(arcOn(arcPosition(operand,edge),arcPosition(operand,arcNext(operand,edge)),point,ARC_TOLERANCE)){return vec2u(0u,2u);}}
  let choices=array<vec3f,7>(ARC_REFERENCE,normalize(vec3f(1.0,0.371,0.127)),normalize(vec3f(-0.253,1.0,0.419)),normalize(vec3f(0.173,-0.391,1.0)),normalize(vec3f(-1.0,0.137,-0.293)),normalize(vec3f(0.317,-1.0,0.193)),normalize(vec3f(-0.227,0.433,-1.0)));
  for(var attempt=0u;attempt<7u;attempt++){
    var parity=referenceInside;var ambiguous=false;
    for(var part=0u;part<2u;part++){
      let first=select(ARC_REFERENCE,choices[attempt],part==1u);let second=select(choices[attempt],point,part==1u);let crossing=cross(first,second);
      if(length(crossing)<ARC_TOLERANCE){if(dot(first,second)<0.0){ambiguous=true;}continue;}
      let normal=normalize(crossing);
      for(var edge=0u;edge<arcVertexCount(operand);edge++){
        let third=arcPosition(operand,edge);let fourth=arcPosition(operand,arcNext(operand,edge));let firstSide=dot(normal,third);let secondSide=dot(normal,fourth);
        if((abs(firstSide)<=ARC_TOLERANCE&&arcOn(first,second,third,ARC_TOLERANCE))||(abs(secondSide)<=ARC_TOLERANCE&&arcOn(first,second,fourth,ARC_TOLERANCE))){ambiguous=true;}
        if(firstSide*secondSide<0.0){let intersection=normalize(cross(crossing,cross(third,fourth)));if((arcOn(first,second,intersection,-ARC_TOLERANCE)&&arcOn(third,fourth,intersection,-ARC_TOLERANCE))||(arcOn(first,second,-intersection,-ARC_TOLERANCE)&&arcOn(third,fourth,-intersection,-ARC_TOLERANCE))){parity^=1u;}}
      }
    }
    if(!ambiguous){return vec2u(parity,0u);}
  }
  return vec2u(0u,4u);
}
fn arcDistance(first:vec3f,second:vec3f,point:vec3f)->f32{let normal=normalize(cross(first,second));let projected=cross(normal,cross(point,normal));var distance=min(arcLength(first,point),arcLength(second,point));if(length(projected)>1e-10){let nearest=normalize(projected);if(arcOn(first,second,nearest,0.0)){distance=min(distance,arcLength(nearest,point));}}return distance;}`,body:`var failure=0u;var required=0u;for(var row=0u;row<${o}u;row++){scratch[scratchOffset+row*7u+6u]=0.0;}
for(var operand=0u;operand<2u;operand++){
  if(arcOffset(operand,0u)!=0u||arcOffset(operand,arcRingCount(operand))!=arcVertexCount(operand)){failure=1u;}
  for(var ring=0u;ring<arcRingCount(operand);ring++){let begin=arcOffset(operand,ring);let end=arcOffset(operand,ring+1u);if(begin>end||end>arcVertexCount(operand)||end-begin<3u){failure=1u;}}
  if(failure!=0u){break;}
  for(var edge=0u;edge<arcVertexCount(operand);edge++){
    let first=arcPosition(operand,edge);let second=arcPosition(operand,arcNext(operand,edge));let squared=dot(first,first);
    if((bitcast<u32>(squared)&0x7f800000u)==0x7f800000u||abs(squared-1.0)>1e-4||length(cross(first,second))<=ARC_TOLERANCE||arcOn(first,second,ARC_REFERENCE,ARC_TOLERANCE)){failure=1u;}
    for(var other=edge+1u;other<arcVertexCount(operand);other++){
      if(other==arcNext(operand,edge)||arcNext(operand,other)==edge){continue;}
      let third=arcPosition(operand,other);let fourth=arcPosition(operand,arcNext(operand,other));let crossing=cross(cross(first,second),cross(third,fourth));
      if(length(crossing)>ARC_TOLERANCE*ARC_TOLERANCE){let point=normalize(crossing);if((arcOn(first,second,point,ARC_TOLERANCE)&&arcOn(third,fourth,point,ARC_TOLERANCE))||(arcOn(first,second,-point,ARC_TOLERANCE)&&arcOn(third,fourth,-point,ARC_TOLERANCE))){failure=1u;}}
      else if(arcOn(first,second,third,ARC_TOLERANCE)||arcOn(first,second,fourth,ARC_TOLERANCE)){failure=1u;}
    }
  }
}
for(var operand=0u;operand<2u&&failure==0u;operand++){
  for(var edge=0u;edge<arcVertexCount(operand)&&failure==0u;edge++){
    let first=arcPosition(operand,edge);let second=arcPosition(operand,arcNext(operand,edge));var points:array<vec3f,130>;var angles:array<f32,130>;var pointCount=2u;points[0]=first;points[1]=second;angles[0]=0.0;angles[1]=arcLength(first,second);
    for(var other=0u;other<arcVertexCount(1u-operand);other++){
      let third=arcPosition(1u-operand,other);let fourth=arcPosition(1u-operand,arcNext(1u-operand,other));let crossing=cross(cross(first,second),cross(third,fourth));var candidates:array<vec3f,2>;
      if(length(crossing)>ARC_TOLERANCE*ARC_TOLERANCE){candidates[0]=normalize(crossing);candidates[1]=-candidates[0];}else{candidates[0]=third;candidates[1]=fourth;}
      for(var candidate=0u;candidate<2u;candidate++){let point=candidates[candidate];if(arcOn(first,second,point,ARC_TOLERANCE)&&arcOn(third,fourth,point,ARC_TOLERANCE)){
        let angle=arcLength(first,point);var duplicate=false;for(var previous=0u;previous<pointCount;previous++){duplicate=duplicate||abs(angles[previous]-angle)<=2.0*ARC_TOLERANCE;}
        if(!duplicate){if(pointCount>=130u){failure=2u;break;}var position=pointCount;loop{if(position==0u||angles[position-1u]<=angle){break;}angles[position]=angles[position-1u];points[position]=points[position-1u];position-=1u;}angles[position]=angle;points[position]=point;pointCount+=1u;}
      }}
    }
    for(var piece=0u;piece+1u<pointCount&&failure==0u;piece++){
      let start=points[piece];let end=points[piece+1u];if(arcLength(start,end)<=4.0*ARC_TOLERANCE){failure=4u;break;}
      let middle=normalize(start+end);let normal=normalize(cross(start,end));var separation=16.0*ARC_TOLERANCE;
      for(var source=0u;source<2u;source++){for(var other=0u;other<arcVertexCount(source);other++){let third=arcPosition(source,other);let fourth=arcPosition(source,arcNext(source,other));if(!arcOn(third,fourth,middle,ARC_TOLERANCE)){separation=min(separation,arcDistance(third,fourth,middle)*0.25);}}}
      if(separation<=2.0*ARC_TOLERANCE){failure=4u;break;}
      let leftPoint=normalize(middle+normal*separation);let rightPoint=normalize(middle-normal*separation);
      let firstLeft=arcContains(0u,leftPoint);let secondLeft=arcContains(1u,leftPoint);let firstRight=arcContains(0u,rightPoint);let secondRight=arcContains(1u,rightPoint);
      if((firstLeft.y|secondLeft.y|firstRight.y|secondRight.y)!=0u){failure=4u;break;}
      let leftInside=arcBoolean(firstLeft.x!=0u,secondLeft.x!=0u);let rightInside=arcBoolean(firstRight.x!=0u,secondRight.x!=0u);
      if(leftInside==rightInside){continue;}
      let startPoint=select(end,start,leftInside);let endPoint=select(start,end,leftInside);var duplicate=false;
      for(var previous=0u;previous<min(required,${o}u);previous++){duplicate=duplicate||(distance(arcStart(previous),startPoint)<=4.0*ARC_TOLERANCE&&distance(arcEnd(previous),endPoint)<=4.0*ARC_TOLERANCE);}
      if(!duplicate){if(required<${o}u){arcStore(required,startPoint,endPoint);}required+=1u;}
    }
  }
}
if(required>${o}u){failure|=2u;}
var vertexCount=0u;var ringCount=0u;
if(failure==0u){for(var seed=0u;seed<required&&failure==0u;seed++){
  if(arcUsed(seed)){continue;}if(ringCount>=${i.output.ringOffsets.length-1}u){failure=2u;break;}
  rings[ringsOffset+ringCount]=vertexCount;var current=seed;var closed=false;
  for(var step=0u;step<required;step++){
    if(arcUsed(current)){failure=4u;break;}scratch[scratchOffset+current*7u+6u]=1.0;let point=arcStart(current);output[outputOffset+vertexCount*3u]=point.x;output[outputOffset+vertexCount*3u+1u]=point.y;output[outputOffset+vertexCount*3u+2u]=point.z;vertexCount+=1u;
    let junction=arcEnd(current);let incoming=normalize(junction*dot(arcStart(current),junction)-arcStart(current));
    var next=0xffffffffu;var bestTurn=-4.0;
    for(var candidate=0u;candidate<required;candidate++){if((!arcUsed(candidate)||candidate==seed)&&distance(arcStart(candidate),junction)<=4.0*ARC_TOLERANCE){
      let outgoing=normalize(arcEnd(candidate)-junction*dot(junction,arcEnd(candidate)));let turn=guardedAtan2(dot(junction,cross(incoming,outgoing)),dot(incoming,outgoing));
      if(abs(turn-bestTurn)<=ARC_TOLERANCE){failure=4u;}if(turn>bestTurn){bestTurn=turn;next=candidate;}
    }}
    if(next==seed){closed=true;break;}
    if(next==0xffffffffu||failure!=0u){failure=4u;break;}current=next;
  }
  if(!closed){failure|=4u;}ringCount+=1u;rings[ringsOffset+ringCount]=vertexCount;
}}
if(failure!=0u){vertexCount=0u;ringCount=0u;}
for(var row=vertexCount;row<${o}u;row++){output[outputOffset+row*3u]=0.0;output[outputOffset+row*3u+1u]=0.0;output[outputOffset+row*3u+2u]=0.0;}
for(var ring=ringCount+1u;ring<${i.output.ringOffsets.length}u;ring++){rings[ringsOffset+ring]=vertexCount;}rings[ringsOffset]=0u;
counts[countsOffset]=vertexCount;counts[countsOffset+1u]=ringCount;counts[countsOffset+2u]=required;counts[countsOffset+3u]=${Number(this.referenceInside)}u;status[statusOffset]=failure;`})]}},j=[`intersection`,`union`,`difference`,`symmetric-difference`],M=128,N=x(k,`shift`),P=x(k,`shiftLatitude`),F=y({parameters:k,geographicWindow:D,caption:`Lake Superior’s mapped shoreline and a modeled survey corridor, combined along great-circle arcs.`,createInputs:()=>({shore:E(`Lake Superior`)}),build(e,{shore:t},n){let r=e.allocate(`right-positions`,`float32x3`,4),i=e.allocate(`result-positions`,`float32x3`,M),a=e.allocate(`result-rings`,`uint32`,17),o=e.allocate(`counts`,`uint32`,4),s=e.allocate(`status`,`uint32`,1);e.graph.add(new A({left:{positions:e.upload(`left-positions`,`float32x3`,w([t])).view,ringOffsets:e.upload(`left-rings`,`uint32`,Uint32Array.of(0,t.length)).view,referenceInside:!1},right:{positions:r.view,ringOffsets:e.upload(`right-rings`,`uint32`,Uint32Array.of(0,4)).view,referenceInside:!1},referencePoint:[0,0,1],operation:j[n.operation],output:{positions:i.view,ringOffsets:a.view,counts:o.view,status:s.view}}));let c=S(e,`result`,i),l=[e.upload(`left-outline`,`float32x4`,T([t])),e.allocate(`right-outline`,`float32x4`,4)];return{outputs:{operandOutlines:l,counts:o,result:C(e,`result`,{positions:c,ringOffsets:a,counts:o})},update:e=>{let t=O(e.shift,e.shiftLatitude);r.write(w([t])),l[1].write(T([t]))}}},rebuildOn:[`operation`],interaction:{kind:`drag`,hint:`drag the corridor`,rest:({toUnit:e})=>e([N.default,P.default]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=b(N,e.x),o=b(P,e.y),s=i(n.counts);return{values:{shift:a,shiftLatitude:o},glyphs:[{shape:`crosshair`,at:[a,o]}],readout:`${j[r.operation]} · ${s?`${s[1]} rings, ${s[0]} vertices`:`…`}`}}},draw:({outputs:e,theme:t})=>[{layer:`lines`,context:!0,segments:e.operandOutlines[0],radius:1.2,color:[.45,.65,1,.85]},{layer:`lines`,context:!0,segments:e.operandOutlines[1],radius:1.2,color:[1,.6,.35,.85]},{layer:`lines`,segments:e.result,radius:3,color:[...t.ink,1]}]});export{F as default};
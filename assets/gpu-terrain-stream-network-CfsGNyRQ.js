import{r as e,t}from"./terrain-binding-preflight-laSEObQE.js";import{T as n,f as r,p as i,s as a}from"./gpu-contributor-utils-BIwLQON5.js";import{r as o}from"./kernel-C3tfQOn4.js";import{t as s}from"./gpu-compaction-B6IzcF0P.js";import{t as c}from"./gpu-reduction-tNgOVQ_C.js";import{t as l}from"./gpu-scan-B4NNTfF9.js";import{u}from"./raster-grid-VB0AM_GY.js";import{c as d,o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{d as p}from"./raster-band-utils-BcTxc8_Y.js";import{i as m,r as h}from"./terrain-flow-tracing-BfyLX2kv.js";import{a as g,i as _,n as v,r as y,t as b}from"./raster-expansion-utils-AZOLseCG.js";import{o as x,s as S}from"./terrain-processing-utils-D-TSF_vl.js";var C=class{id;props;estimatedCommandNodeCount;localOrigin;constructor(e){this.id=e.id??`terrain-stream-network`,this.props=e,this.localOrigin=e.localOrigin??[0,0],n(this.id,this.localOrigin.length===2&&this.localOrigin.every(Number.isFinite),`localOrigin must be two finite numbers`),m(this.id,e.directionEncoding);let t=g(this.id,e.grid,[e.flowDirections,e.streams],a(e.output));y(e.flowDirections,t),y(e.streams,t),y(e.output.targetCells,e.output.sourceCells.length),y(e.output.lengths,e.output.sourceCells.length),y(e.output.vertexOffsets,e.output.sourceCells.length+1);for(let t of[e.output.reachCount,e.output.vertexCount,e.output.status.overflow,e.output.unresolvedCount])y(t,1);u(this.id,e.cellSizeMode),p(this.id,e.grid,e.cellSizeMode),r(this.id,`maximumSteps`,e.maximumSteps??t),this.estimatedCommandNodeCount=40+v(S(Math.min(e.maximumSteps??t,t))),i(this.id,this.estimatedCommandNodeCount)}getCommandNodes(n){e(n,this.id,this.props,{"pointer state":20*this.props.grid.width*this.props.grid.height,"gate table":t(S(Math.min(this.props.maximumSteps??this.props.grid.width*this.props.grid.height,this.props.grid.width*this.props.grid.height)))});let{id:r,props:i}=this,{grid:u,output:p}=i,m=u.width*u.height;_(r,n,[i.flowDirections,i.streams,...a(p)]);let g=`GPUTerrainStreamNetwork`,v=i.maximumSteps??m,y=S(Math.min(v,m)),C=[0,1].map(e=>o(n,`${r}-state-${e}`,`uint32`,m*5)),w=o(n,`${r}-converged`,`uint32`,1),T=o(n,`${r}-vertex-counts`,`uint32`,m),E=o(n,`${r}-unresolved-flags`,`uint32`,m),D=o(n,`${r}-reach-flags`,`uint32`,m),O=o(n,`${r}-cell-ids`,`uint32`,m),k=o(n,`${r}-compact-cells`,`uint32`,m),A=o(n,`${r}-reach-total`,`uint32`,1),j=o(n,`${r}-row-vertices`,`uint32`,m),M=o(n,`${r}-row-ends`,`uint32`,m),N=h(u.width,u.height,i.directionEncoding),P=`${N}
fn isStream(cell:u32)->bool {return streams[streamsOffset+cell]!=0u;}
fn incomingCount(cell:u32)->u32 {
 var incoming=0u; let column=i32(cell%GRID_WIDTH); let row=i32(cell/GRID_WIDTH);
 for(var rowOffset=-1;rowOffset<=1;rowOffset++){for(var columnOffset=-1;columnOffset<=1;columnOffset++){
 let otherColumn=column+columnOffset;let otherRow=row+rowOffset;
 if(otherColumn<0||otherRow<0||otherColumn>=i32(GRID_WIDTH)||otherRow>=i32(GRID_HEIGHT)||(columnOffset==0&&rowOffset==0)){continue;}
 let other=u32(otherRow)*GRID_WIDTH+u32(otherColumn);
 if(isStream(other)&&getFlowReceiver(other,getFlowDirectionCode(directions[directionsOffset+other]))==cell){incoming++;}
 }}return incoming;
}`,F=[{name:`directions`,view:i.flowDirections,type:`u32`,access:`read`},{name:`streams`,view:i.streams,type:`u32`,access:`read`}];return[f(n,{id:`${r}-seed`,operation:g,invocationCount:m,bindings:[...F,{name:`state`,view:C[0],type:`u32`,access:`read_write`}],declarations:P,body:`var nextCell=NO_CELL;var steps=1u;var outcome=0u;var reachNext=NO_CELL;var reachHops=0u;
if(isStream(index)){
 let stream=streams[streamsOffset+index];
 let code=getFlowDirectionCode(directions[directionsOffset+index]);
 let receiver=getFlowReceiver(index,code);
 let leaves=receiver==NO_CELL||!isStream(receiver);
 if(stream==NO_CELL||stream==UNRESOLVED_DIRECTION||(code!=0u&&receiver==NO_CELL)){outcome=2u;}
 else if(leaves){outcome=1u;}
 else{nextCell=receiver;}
 // The reach continues through the receiver only while it is a stream cell with exactly one donor.
 if(!leaves){reachHops=1u;if(incomingCount(receiver)==1u){reachNext=receiver;}}
}
let offset=stateOffset+index*5u;
state[offset]=nextCell;state[offset+1u]=steps;state[offset+2u]=outcome;state[offset+3u]=reachNext;state[offset+4u]=reachHops;`}),...b(n,{id:`${r}-jump`,operation:g,invocationCount:m,roundCount:y,status:{converged:w},createRound:({nodeId:e,iteration:t,slotBinding:r,markChangedWGSL:i,condition:a,extraResources:o},s)=>f(n,{id:e,operation:g,variant:`jump`,invocationCount:m,bindings:[{name:`previous`,view:C[t%2],type:`u32`,access:`read`},{name:`next`,view:C[(t+1)%2],type:`u32`,access:`read_write`},{name:`status`,view:s,type:`atomic<u32>`,access:`read_write`},r],declarations:`const NO_CELL: u32 = 0xffffffffu;`,condition:a,extraResources:o,body:`let offset=previousOffset+index*5u;let destination=nextOffset+index*5u;
for(var component=0u;component<5u;component++){next[destination+component]=previous[offset+component];}
let pathNext=previous[offset];
if(pathNext!=NO_CELL){
 let ancestor=previousOffset+pathNext*5u;
 next[destination]=previous[ancestor];
 next[destination+1u]=min(${m+1}u,previous[offset+1u]+previous[ancestor+1u]);
 next[destination+2u]=previous[ancestor+2u];
 ${i}
}
let reachNext=previous[offset+3u];
if(reachNext!=NO_CELL){
 let ancestor=previousOffset+reachNext*5u;
 next[destination+3u]=previous[ancestor+3u];
 next[destination+4u]=min(${m+1}u,previous[offset+4u]+previous[ancestor+4u]);
 ${i}
}`})}),f(n,{id:`${r}-reaches`,operation:g,invocationCount:m,bindings:[...F,{name:`state`,view:C[y%2],type:`u32`,access:`read`},{name:`vertexCounts`,view:T,type:`u32`,access:`read_write`},{name:`unresolvedFlags`,view:E,type:`u32`,access:`read_write`},{name:`reachFlags`,view:D,type:`u32`,access:`read_write`},{name:`cellIds`,view:O,type:`u32`,access:`read_write`}],declarations:P,body:`var vertices=0u;var unresolved=0u;
if(isStream(index)){
 let offset=stateOffset+index*5u;
 let complete=state[offset]==NO_CELL&&state[offset+2u]==1u&&state[offset+1u]<=${v}u;
 unresolved=u32(!complete);
 if(complete&&incomingCount(index)!=1u){
  vertices=1u+state[offset+4u];
  if(vertices<2u){vertices=0u;}
 }
}
vertexCounts[vertexCountsOffset+index]=vertices;unresolvedFlags[unresolvedFlagsOffset+index]=unresolved;
reachFlags[reachFlagsOffset+index]=u32(vertices>0u);cellIds[cellIdsOffset+index]=index;`}),...new c({id:`${r}-unresolved-sum`,input:E,output:p.unresolvedCount,operation:`sum`}).getCommandNodes(n),...new s({id:`${r}-compaction`,input:O,flags:D,output:k,count:A}).getCommandNodes(n),f(n,{id:`${r}-gather`,operation:g,invocationCount:m,bindings:[{name:`compactCells`,view:k,type:`u32`,access:`read`},{name:`reachTotal`,view:A,type:`u32`,access:`read`},{name:`vertexCounts`,view:T,type:`u32`,access:`read`},{name:`rowVertices`,view:j,type:`u32`,access:`read_write`}],body:`rowVertices[rowVerticesOffset+index]=select(0u,vertexCounts[vertexCountsOffset+compactCells[compactCellsOffset+index]],index<reachTotal[reachTotalOffset]);`}),...new l({id:`${r}-vertex-scan`,input:j,output:M,mode:`inclusive`}).getCommandNodes(n),f(n,{id:`${r}-publish-clear`,operation:g,invocationCount:1,bindings:[{name:`reachTotal`,view:A,type:`u32`,access:`read`},{name:`offsets`,view:p.vertexOffsets,type:`u32`,access:`read_write`},{name:`reachCount`,view:p.reachCount,type:`u32`,access:`read_write`},{name:`vertexCount`,view:p.vertexCount,type:`u32`,access:`read_write`},{name:`overflow`,view:p.status.overflow,type:`u32`,access:`read_write`}],body:`offsets[offsetsOffset]=0u;reachCount[reachCountOffset]=0u;vertexCount[vertexCountOffset]=0u;overflow[overflowOffset]=u32(reachTotal[reachTotalOffset]>0u);`}),f(n,{id:`${r}-publish`,operation:g,invocationCount:m,bindings:[{name:`compactCells`,view:k,type:`u32`,access:`read`},{name:`reachTotal`,view:A,type:`u32`,access:`read`},{name:`rowEnds`,view:M,type:`u32`,access:`read`},{name:`sources`,view:p.sourceCells,type:`u32`,access:`read_write`},{name:`offsets`,view:p.vertexOffsets,type:`u32`,access:`read_write`},{name:`reachCount`,view:p.reachCount,type:`u32`,access:`read_write`},{name:`vertexCount`,view:p.vertexCount,type:`u32`,access:`read_write`},{name:`overflow`,view:p.status.overflow,type:`u32`,access:`read_write`}],body:`let total=reachTotal[reachTotalOffset];
if(index>=total){return;}
let end=rowEnds[rowEndsOffset+index];
if(index>=${p.sourceCells.length}u||end>${p.positions.length}u){return;}
sources[sourcesOffset+index]=compactCells[compactCellsOffset+index];offsets[offsetsOffset+index+1u]=end;
var nextKept=false;
if(index+1u<total){nextKept=index+1u<${p.sourceCells.length}u&&rowEnds[rowEndsOffset+index+1u]<=${p.positions.length}u;}
if(!nextKept){reachCount[reachCountOffset]=index+1u;vertexCount[vertexCountOffset]=end;overflow[overflowOffset]=u32(index+1u<total);}`}),f(n,{id:`${r}-geometry`,operation:g,invocationCount:p.sourceCells.length,bindings:[{name:`directions`,view:i.flowDirections,type:`u32`,access:`read`},{name:`sources`,view:p.sourceCells,type:`u32`,access:`read`},{name:`offsets`,view:p.vertexOffsets,type:`u32`,access:`read`},{name:`reachCount`,view:p.reachCount,type:`u32`,access:`read`},{name:`positions`,view:p.positions,type:`f32`,access:`read_write`},{name:`targets`,view:p.targetCells,type:`u32`,access:`read_write`},{name:`lengths`,view:p.lengths,type:`f32`,access:`read_write`}],declarations:`${N}
${x(u,i.cellSizeMode)}
fn position(cell:u32)->vec2<f32>{let column=f32(cell%GRID_WIDTH)+${u.pixelInterpretation===`area`?`0.5`:`0.0`};let row=f32(cell/GRID_WIDTH)+${u.pixelInterpretation===`area`?`0.5`:`0.0`};return vec2<f32>(${d(u.affine[0])}*column+${d(u.affine[1])}*row+${d(u.affine[2]-this.localOrigin[0])},${d(u.affine[3])}*column+${d(u.affine[4])}*row+${d(u.affine[5]-this.localOrigin[1])});}`,body:`if(index>=reachCount[reachCountOffset]){return;}var current=sources[sourcesOffset+index];var total=0.0;var previousCell=current;
for(var vertex=offsets[offsetsOffset+index];vertex<offsets[offsetsOffset+index+1u];vertex++){
 let point=position(current);positions[positionsOffset+vertex*2u]=point.x;positions[positionsOffset+vertex*2u+1u]=point.y;
 // Measure the displacement between cells before adding the origin, which can round away small edges.
 total+=getGroundEdgeLength(previousCell,current);previousCell=current;
 if(vertex+1u<offsets[offsetsOffset+index+1u]){current=getFlowReceiver(current,getFlowDirectionCode(directions[directionsOffset+current]));}
}targets[targetsOffset+index]=current;lengths[lengthsOffset+index]=total;`})]}};export{C as t};
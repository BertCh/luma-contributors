import{a as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t}from"./validate-outline-output-passes-DIcqH6Pw.js";import{n}from"./refinement-storage-validation-D9BhQGNW.js";import{r}from"./kernel-C3tfQOn4.js";import{n as i,t as a}from"./scan-total-C4QddMmi.js";import{o}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as s}from"./gpu-scan-B4NNTfF9.js";import{i as c,n as l,t as u}from"./gpu-polygon-triangulation-CBZGZXVA.js";import{E as d,f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{t as m}from"./exact-orientation-wgsl-B_jI-HEc.js";import{i as h}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{A as g}from"./pointer-overlay-Qh8qflcA.js";import{t as _}from"./define-run-B-brkCe4.js";import{n as v,s as y,v as b,y as x}from"./polygon-set-CHja50mr.js";import{l as S,r as C,t as w}from"./pointer-support-C2rPRVFO.js";import{s as T}from"./shape-segments-DMcyF4V-.js";import{Ya as E}from"./entry-preview-D70TdSbK.js";function D(e,t){let{props:n,id:s,maximumVertices:c}=t,l=Math.max(1,n.positions.length+2*(n.ringOffsets.length-1)),u={positions:r(e,`${s}-seed-positions`,`float32x2`,l*128),pieces:r(e,`${s}-seed-pieces`,`uint32x4`,l),status:r(e,`${s}-seed-status`,`uint32x4`,1)},d=Math.min(l,O),f=r(e,`${s}-split-queue`,`float32`,33280*d),p=r(e,`${s}-leaf-counts`,`uint32`,l),m=r(e,`${s}-leaf-offsets`,`uint32`,l),h=r(e,`${s}-leaf-total`,`uint32`,1),g=r(e,`${s}-exhausted`,`uint32`,1),_=t=>o(e,{id:`${s}-rectilinear-${t}`,operation:`GPUSubdividePolygons`,variant:`rectilinear-queue-${t}`,invocationCount:d,bindings:[{name:`inputPositions`,view:u.positions,type:`f32`,access:`read`},{name:`inputPieces`,view:u.pieces,type:`u32`,access:`read`},{name:`inputStatus`,view:u.status,type:`u32`,access:`read`},{name:`queue`,view:f,type:`f32`,access:`read_write`},...t===`count`?[{name:`leafCounts`,view:p,type:`u32`,access:`read_write`},{name:`exhaustedFlag`,view:g,type:`atomic<u32>`,access:`read_write`}]:[{name:`leafOffsets`,view:m,type:`u32`,access:`read`},{name:`positions`,view:n.output.positions,type:`f32`,access:`read_write`},{name:`pieces`,view:n.output.pieces,type:`u32`,access:`read_write`}]],declarations:`${a}
const LIMIT:u32=${c}u;
const CAPACITY:u32=${n.output.pieces.length}u;
const THREADS:u32=${d}u;
const QUEUE_BASE:u32=33280u;
fn queuePoint(slot:u32,vertex:u32,thread:u32)->vec2f {let base=queueOffset+thread*QUEUE_BASE+slot*260u+vertex*2u;return vec2f(queue[base],queue[base+1u]);}
fn storePoint(slot:u32,vertex:u32,thread:u32,point:vec2f) {let base=queueOffset+thread*QUEUE_BASE+slot*260u+vertex*2u;queue[base]=point.x;queue[base+1u]=point.y;}`,body:k(t,n.maximumSplits??4096)});return[...new j({...n,id:`${s}-seeds`,method:`convex`,maximumVertices:128,output:u}).getCommandNodes(e),M(e,s,n.output.pieces),o(e,{id:`${s}-clear-counts`,operation:`GPUSubdividePolygons`,variant:`rectilinear-clear`,invocationCount:l,bindings:[{name:`leafCounts`,view:p,type:`u32`,access:`read_write`},{name:`exhaustedFlag`,view:g,type:`u32`,access:`read_write`}],body:`leafCounts[leafCountsOffset + index] = 0u; if (index == 0u) { exhaustedFlag[exhaustedFlagOffset] = 0u; }`}),_(`count`),...i(e,{id:`${s}-leaf-scan`,operation:`GPUSubdividePolygons`,counts:p,offsets:m,rowCount:l,total:h}),_(`write`),o(e,{id:`${s}-rectilinear-status`,operation:`GPUSubdividePolygons`,variant:`rectilinear-status`,invocationCount:1,bindings:[{name:`leafTotal`,view:h,type:`u32`,access:`read`},{name:`exhaustedFlag`,view:g,type:`u32`,access:`read`},{name:`inputStatus`,view:u.status,type:`u32`,access:`read`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`}],declarations:`const CAPACITY:u32=${n.output.pieces.length}u;`,body:`let required=leafTotal[leafTotalOffset];
status[statusOffset]=min(required,CAPACITY);status[statusOffset+1u]=required;
status[statusOffset+2u]=inputStatus[inputStatusOffset+2u]|select(0u,1u,required>CAPACITY);
status[statusOffset+3u]=inputStatus[inputStatusOffset+3u]|select(0u,0x80000000u,exhaustedFlag[exhaustedFlagOffset]!=0u);`})]}var O=256;function k(e,t){return`for(var seed=index;seed<inputStatus[inputStatusOffset];seed+=THREADS) {
 var leaves=0u;
 let initialCount=inputPieces[inputPiecesOffset+seed*4u];
 for(var vertex=0u;vertex<initialCount;vertex++) {let base=inputPositionsOffset+(seed*128u+vertex)*2u;storePoint(0u,vertex,index,vec2f(inputPositions[base],inputPositions[base+1u]));}
 queue[queueOffset+index*QUEUE_BASE+256u]=f32(initialCount);var pending=1u;var work=0u;var exhausted=false;
 loop {
  if(pending==0u) {break;}
  if(work>=${t}u) {exhausted=true;break;} work++;pending--;
  let count=u32(queue[queueOffset+index*QUEUE_BASE+pending*260u+256u]);
  var polygon:array<vec2f,128>;var lower=vec2f(3e38);var upper=vec2f(-3e38);
  for(var vertex=0u;vertex<count;vertex++) {let point=queuePoint(pending,vertex,index);polygon[vertex]=point;lower=min(lower,point);upper=max(upper,point);}
  if(count<=LIMIT) {
   ${e===`write`?`let required=getSaturatedScanSum(leafOffsets[leafOffsetsOffset+seed],leaves);
   if(required<CAPACITY) {
    pieces[piecesOffset+required*4u]=count;
    for(var column=1u;column<4u;column++) {pieces[piecesOffset+required*4u+column]=inputPieces[inputPiecesOffset+seed*4u+column];}
    for(var vertex=0u;vertex<count;vertex++) {let base=positionsOffset+(required*LIMIT+vertex)*2u;positions[base]=polygon[vertex].x;positions[base+1u]=polygon[vertex].y;}
   }`:``} leaves++;continue;
  }
  let axis=select(1u,0u,upper.x-lower.x>=upper.y-lower.y);let middle=lower[axis]+(upper[axis]-lower[axis])*0.5;
  if(!(middle>lower[axis] && middle<upper[axis]) || pending+2u>128u) {exhausted=true;break;}
  for(var half=0u;half<2u;half++) {
   var clipped:array<vec2f,130>;var clippedCount=0u;
   for(var vertex=0u;vertex<count;vertex++) {
    let first=polygon[vertex];let last=polygon[(vertex+1u)%count];let firstInside=select(first[axis]<=middle,first[axis]>=middle,half==1u);let lastInside=select(last[axis]<=middle,last[axis]>=middle,half==1u);
    if(firstInside) {clipped[clippedCount]=first;clippedCount++;}
    if(firstInside!=lastInside) {let fraction=(middle-first[axis])/(last[axis]-first[axis]);if(fraction>0.0 && fraction<1.0) {var crossing=first+fraction*(last-first);crossing[axis]=middle;clipped[clippedCount]=crossing;clippedCount++;}}
   }
   // Exact collinear vertices carry no corner information and need not consume the vertex limit.
   var compact:array<vec2f,130>;var compactCount=0u;
   for(var vertex=0u;vertex<clippedCount;vertex++) {
    let previous=clipped[(vertex+clippedCount-1u)%clippedCount];let current=clipped[vertex];let next=clipped[(vertex+1u)%clippedCount];let incoming=current-previous;let outgoing=next-current;
    if(all(current==previous) || incoming.x*outgoing.y-incoming.y*outgoing.x==0.0) {continue;}
    compact[compactCount]=current;compactCount++;
   }
   if(compactCount<3u) {continue;}
   if(compactCount>128u) {exhausted=true;break;}
   for(var vertex=0u;vertex<compactCount;vertex++) {storePoint(pending,vertex,index,compact[vertex]);}
   queue[queueOffset+index*QUEUE_BASE+pending*260u+256u]=f32(compactCount);pending++;
  }
 }
 ${e===`count`?`leafCounts[leafCountsOffset+seed]=leaves;if(exhausted) {atomicStore(&exhaustedFlag[exhaustedFlagOffset],1u);}`:``}
}`}function A(e,t){let n=Math.max(1,e+2*t);return{seedPositions:n*128*8,seedPieces:n*16,splitQueue:33280*Math.min(n,O)*4,leafCountsAndScan:n*4}}var j=class{id;props;maximumVertices;constructor(e){if(this.id=e.id??`subdivide-polygons`,this.props=e,this.maximumVertices=e.maximumVertices??32,f(this.id,`maximumVertices`,this.maximumVertices,e.method===`rectilinear`?4:3,128),f(this.id,`maximumSplits`,e.maximumSplits??4096,1,65536),f(this.id,`positions`,e.positions.length,1,2147483647),d(e.output.positions,[`float32x2`],this.id),d(e.output.pieces,[`uint32x4`],this.id),d(e.output.status,[`uint32x4`],this.id),e.output.positions.length!==e.output.pieces.length*this.maximumVertices||e.output.status.length!==1)throw Error(`Subdivision output capacity mismatch`);t(this.id,e.output,[[`positions`,`pieces`],[`status`]]),p(this.id,Object.values(e.output),[e.positions,e.ringOffsets,e.polygonOffsets,e.featureOffsets])}getCommandNodes(t){let{props:i,id:a,maximumVertices:d}=this;e(t,a,Object.fromEntries([i.positions,i.ringOffsets,i.polygonOffsets,i.featureOffsets,...Object.values(i.output)].map((e,t)=>[`binding.${t}`,e])));let f=i.polygonOffsets.length-1;if(c(t.device,a,i.positions.length,i.ringOffsets.length-1,f),n(t,a,{indices:Math.max(3,l(i.positions.length,i.ringOffsets.length-1,f))*4,polygonInfo:f*8,triangleOffsets:(f+1)*4,...i.method===`rectilinear`?A(i.positions.length,i.ringOffsets.length-1):{}}),i.method===`rectilinear`)return D(t,this);let p=i.polygonOffsets.length-1,g=Math.max(3,l(i.positions.length,i.ringOffsets.length-1,p)),_=r(t,`${a}-indices`,`uint32`,g),v=r(t,`${a}-valid`,`uint32`,p),y=r(t,`${a}-triangle-offsets`,`uint32`,p+1),b=r(t,`${a}-polygon-info`,`uint32`,Math.max(p*2,1)),x=Math.max(Math.floor(g/3),1),S=r(t,`${a}-consumed`,`uint32`,x),C=r(t,`${a}-piece-vertex-counts`,`uint32`,x),w=r(t,`${a}-piece-flags`,`uint32`,x),T=r(t,`${a}-piece-offsets`,`uint32`,x),E=e=>o(t,{id:`${a}-merge-${e}`,operation:`GPUSubdividePolygons`,variant:`convex-merge-${e}`,invocationCount:p,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`indices`,view:_,type:`u32`,access:`read`},{name:`consumed`,view:S,type:`u32`,access:`read_write`},{name:`info`,view:b,type:`u32`,access:`read`},{name:`triangleOffsets`,view:y,type:`u32`,access:`read`},...e===`count`?[{name:`pieceVertexCounts`,view:C,type:`u32`,access:`read_write`}]:[{name:`pieceOffsets`,view:T,type:`u32`,access:`read`},{name:`outPositions`,view:i.output.positions,type:`f32`,access:`read_write`},{name:`pieces`,view:i.output.pieces,type:`u32`,access:`read_write`}]],declarations:`${m}
const MAXIMUM_VERTICES: u32 = ${d}u;
const PIECE_CAPACITY: u32 = ${i.output.pieces.length}u;
${h(`point`,`positions`)}
fn triangle(row: u32) -> vec3u { return vec3u(indices[indicesOffset + row * 3u] & 0x7fffffffu, indices[indicesOffset + row * 3u + 1u], indices[indicesOffset + row * 3u + 2u]); }
fn turn(first: u32, second: u32, third: u32) -> i32 {
  return orientSign(point(first), point(second), point(third));
}`,body:`let polygon = index;
if (info[infoOffset + polygon * 2u] != 0u) { return; }
let first = triangleOffsets[triangleOffsetsOffset + polygon];
let last = triangleOffsets[triangleOffsetsOffset + polygon + 1u];
let feature = info[infoOffset + polygon * 2u + 1u];
for (var seed = first; seed < last; seed++) {
  if (consumed[consumedOffset + seed] != 0u) { continue; }
    let initial = triangle(seed); let orientation = turn(initial.x, initial.y, initial.z);
    if (orientation == 0) { continue; }
    var boundary: array<u32, MAXIMUM_VERTICES>;
    boundary[0] = initial.x; boundary[1] = initial.y; boundary[2] = initial.z;
    var vertexCount = 3u;
    consumed[consumedOffset + seed] = 1u;
    loop {
      if (vertexCount >= MAXIMUM_VERTICES) { break; }
      var merged = false;
      for (var candidate = first; candidate < last; candidate++) {
        if (consumed[consumedOffset + candidate] != 0u) { continue; }
        let current = triangle(candidate);
        if (turn(current.x, current.y, current.z) == 0) { continue; }
        for (var edge = 0u; edge < vertexCount; edge++) {
          for (var side = 0u; side < 3u; side++) {
            if (boundary[edge] != current[(side + 1u) % 3u] || boundary[(edge + 1u) % vertexCount] != current[side]) { continue; }
            let inserted = current[(side + 2u) % 3u];
            var proposed: array<u32, MAXIMUM_VERTICES>;
            for (var vertex = 0u; vertex <= vertexCount; vertex++) {
              if (vertex <= edge) { proposed[vertex] = boundary[vertex]; }
              else if (vertex == edge + 1u) { proposed[vertex] = inserted; }
              else { proposed[vertex] = boundary[vertex - 1u]; }
            }
            var convex = true;
            for (var vertex = 0u; vertex <= vertexCount; vertex++) {
              let direction = turn(proposed[vertex], proposed[(vertex + 1u) % (vertexCount + 1u)], proposed[(vertex + 2u) % (vertexCount + 1u)]);
              if (direction == 2 || direction * orientation < 0) { convex = false; }
              if (vertex != edge + 1u && proposed[vertex] == inserted) { convex = false; }
            }
            if (!convex) { continue; }
            for (var vertex = 0u; vertex <= vertexCount; vertex++) { boundary[vertex] = proposed[vertex]; }
            vertexCount++; consumed[consumedOffset + candidate] = 1u; merged = true; break;
          }
          if (merged) { break; }
        }
        if (merged) { break; }
      }
      if (!merged) { break; }
    }
  ${e===`count`?`pieceVertexCounts[pieceVertexCountsOffset + seed] = vertexCount;`:`let required = pieceOffsets[pieceOffsetsOffset + seed];
  if (required < PIECE_CAPACITY) {
    pieces[piecesOffset + required * 4u] = vertexCount;
    pieces[piecesOffset + required * 4u + 1u] = feature;
    pieces[piecesOffset + required * 4u + 2u] = polygon;
    pieces[piecesOffset + required * 4u + 3u] = seed;
    for (var vertex = 0u; vertex < vertexCount; vertex++) {
      let value = point(boundary[vertex]); let destination = (required * MAXIMUM_VERTICES + vertex) * 2u;
      outPositions[outPositionsOffset + destination] = value.x; outPositions[outPositionsOffset + destination + 1u] = value.y;
    }
  }`}
}`});return[...new u({...i,id:`${a}-triangulation`,output:{indices:_,valid:v,triangleOffsets:y}}).getCommandNodes(t),M(t,a,i.output.pieces),o(t,{id:`${a}-polygon-info`,operation:`GPUSubdividePolygons`,variant:`polygon-info`,invocationCount:p,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`indices`,view:_,type:`u32`,access:`read`},{name:`valid`,view:v,type:`u32`,access:`read`},{name:`triangleOffsets`,view:y,type:`u32`,access:`read`},{name:`featureOffsets`,view:i.featureOffsets,type:`u32`,access:`read`},{name:`info`,view:b,type:`u32`,access:`read_write`}],declarations:`${m}
${h(`point`,`positions`)}`,body:`var rejected = valid[validOffset + index] == 0u;
if (!rejected) {
  for (var candidate = triangleOffsets[triangleOffsetsOffset + index]; candidate < triangleOffsets[triangleOffsetsOffset + index + 1u]; candidate++) {
    let current = vec3u(indices[indicesOffset + candidate * 3u] & 0x7fffffffu, indices[indicesOffset + candidate * 3u + 1u], indices[indicesOffset + candidate * 3u + 2u]);
    if (orientSign(point(current.x), point(current.y), point(current.z)) == 2) { rejected = true; }
  }
}
// Largest feature whose first polygon is at or before this polygon (feature 0 is the default).
var low = 1u; var high = ${i.featureOffsets.length}u;
while (low < high) {
  let middle = (low + high) >> 1u;
  if (featureOffsets[featureOffsetsOffset + middle] <= index) { low = middle + 1u; } else { high = middle; }
}
info[infoOffset + index * 2u] = select(0u, 1u, rejected);
info[infoOffset + index * 2u + 1u] = low - 1u;`}),o(t,{id:`${a}-consumed-reset`,operation:`GPUSubdividePolygons`,variant:`convex-consumed-reset`,invocationCount:x,bindings:[{name:`indices`,view:_,type:`u32`,access:`read`},{name:`consumed`,view:S,type:`u32`,access:`read_write`},{name:`pieceVertexCounts`,view:C,type:`u32`,access:`read_write`}],body:`consumed[consumedOffset + index] = indices[indicesOffset + index * 3u] >> 31u;
pieceVertexCounts[pieceVertexCountsOffset + index] = 0u;`}),E(`count`),o(t,{id:`${a}-piece-flags`,operation:`GPUSubdividePolygons`,variant:`convex-piece-flags`,invocationCount:x,bindings:[{name:`pieceVertexCounts`,view:C,type:`u32`,access:`read`},{name:`pieceFlags`,view:w,type:`u32`,access:`read_write`}],body:`pieceFlags[pieceFlagsOffset + index] = select(0u, 1u, pieceVertexCounts[pieceVertexCountsOffset + index] != 0u);`}),...new s({id:`${a}-piece-scan`,input:w,output:T,mode:`exclusive`}).getCommandNodes(t),o(t,{id:`${a}-consumed-reset-write`,operation:`GPUSubdividePolygons`,variant:`convex-consumed-reset`,invocationCount:x,bindings:[{name:`indices`,view:_,type:`u32`,access:`read`},{name:`consumed`,view:S,type:`u32`,access:`read_write`},{name:`pieceVertexCounts`,view:C,type:`u32`,access:`read_write`}],body:`consumed[consumedOffset + index] = indices[indicesOffset + index * 3u] >> 31u;
pieceVertexCounts[pieceVertexCountsOffset + index] = 0u;`}),E(`write`),o(t,{id:`${a}-merge-status`,operation:`GPUSubdividePolygons`,variant:`convex-merge-status`,invocationCount:1,bindings:[{name:`pieceFlags`,view:w,type:`u32`,access:`read`},{name:`pieceOffsets`,view:T,type:`u32`,access:`read`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`}],declarations:`const PIECE_CAPACITY: u32 = ${i.output.pieces.length}u;`,body:`let required = pieceOffsets[pieceOffsetsOffset + ${x-1}u] + pieceFlags[pieceFlagsOffset + ${x-1}u];
status[statusOffset] = min(required, PIECE_CAPACITY); status[statusOffset + 1u] = required;
status[statusOffset + 2u] = select(0u, 1u, required > PIECE_CAPACITY); status[statusOffset + 3u] = 0u;`}),o(t,{id:`${a}-invalid-count`,operation:`GPUSubdividePolygons`,variant:`convex-invalid-count`,invocationCount:p,bindings:[{name:`info`,view:b,type:`u32`,access:`read`},{name:`status`,view:i.output.status,type:`atomic<u32>`,access:`read_write`}],body:`if (info[infoOffset + index * 2u] != 0u) { atomicAdd(&status[statusOffset + 3u], 1u); }`})]}};function M(e,t,n){return o(e,{id:`${t}-zero-pieces`,operation:`GPUSubdividePolygons`,variant:`zero-pieces`,invocationCount:n.length*4,bindings:[{name:`pieces`,view:n,type:`u32`,access:`read_write`}],body:`pieces[piecesOffset + index] = 0u;`})}var N=24,P={vertexCount:12,rows:2,columns:4},F=_({parameters:E,caption:`Each star split into convex pieces, one color per piece.`,createInputs:({seed:e,world:t})=>{let n=y(g(e),t,{...P,irregularity:.45});return{seed:e,world:t,set:n,...C(n,2)}},build(e,t,n){let r=b(e,t.set),i=n.maximumVertices,a=t.set.count*N,o=a*i,s=e.allocate(`piece-vertices`,`float32x2`,o),c=e.allocate(`pieces`,`uint32x4`,a);e.graph.add(new j({positions:r.positions.view,featureOffsets:r.featureOffsets.view,polygonOffsets:r.polygonOffsets.view,ringOffsets:r.ringOffsets.view,method:`convex`,maximumVertices:i,output:{positions:s.view,pieces:c.view,status:e.allocate(`status`,`uint32x4`,1).view}}));let l=e.allocate(`piece-edges`,`uint32x2`,o),u=e.allocate(`piece-categories`,`uint32`,o);return T(e,{id:`piece-rings`,invocationCount:o,bindings:[{name:`pieces`,buffer:c,access:`read`},{name:`edges`,buffer:l,access:`read_write`},{name:`categories`,buffer:u,access:`read_write`}],body:`let piece = index / ${i}u;
  let slot = index % ${i}u;
  let vertexCount = pieces[piecesOffset + piece * 4u];
  let isWritten = slot < vertexCount;
  edges[edgesOffset + index * 2u] = index;
  edges[edgesOffset + index * 2u + 1u] = select(index, piece * ${i}u + (slot + 1u) % max(vertexCount, 1u), isWritten);
  categories[categoriesOffset + index] = select(0xffffffffu, piece, isWritten);`}),{outputs:{...r,pieceVertices:s,edges:l,categories:u,pieces:c},update:e=>{let n=y(g(t.seed),t.world,{...P,irregularity:e.irregularity});v(n,t.drag),t.set=n,x(r,n)}}},interaction:{kind:`drag`,hint:`push a vertex inward`,rest:({inputs:e,toUnit:t})=>t(S(e)),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=w(t,e,0,.4),o=t.drag?.feature??0,s=i(n.pieces),c=0;if(s)for(let e=0;e<s.length;e+=4)s[e]>0&&s[e+1]===o&&c++;return{rerun:!0,glyphs:a.glyphs,readout:`maximumVertices ${r.maximumVertices} · ${s?c:`...`} pieces`}}},rebuildOn:[`maximumVertices`],draw:({inputs:e,outputs:t})=>[{layer:`lines`,context:!0,segments:t.outline,count:e.set.outlineCount},{layer:`segments`,nodes:t.pieceVertices,edges:t.edges,values:t.categories,radius:2.2}]});export{F as default};
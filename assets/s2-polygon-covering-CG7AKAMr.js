import{c as e}from"./hydrography-N8loF-Ds.js";import{t}from"./assert-unit-vector-BIr1r7vX.js";import{a as n,c as r,n as i,r as a,s as o}from"./spherical-limits-C7rAYBu5.js";import{t as s}from"./gpu-scan-B4NNTfF9.js";import{r as c}from"./kernel-C3tfQOn4.js";import{s as l}from"./cell-index-wgsl-Cw7syfdA.js";import{v as u}from"./spatial-context-DwZEd4EZ.js";import{o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as f,T as p,f as m,p as h,x as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{t as v}from"./s2-analysis-wgsl-CSV2LEdb.js";import{n as y,r as b,t as x}from"./s2-covering-support-BNzE_85J.js";import{i as S,r as C}from"./cell-dispatch-validation-PIkQyAK_.js";import{t as w}from"./define-run-B-brkCe4.js";import{t as T}from"./cells-Di9l-3nc.js";import{d as E}from"./kernels-IDOsx_br.js";import{n as D}from"./polygons--0zAAHu9.js";import{rl as O}from"./entry-preview-D70TdSbK.js";var k=4294967295,A=15,j=1048576,M=3*(A+1)+4,N=class{props;id;spatialContext;estimatedCommandNodeCount;constructor(e){this.props=e,this.id=e.id??`s2-polygon-covering`,this.spatialContext=u(this.id,e.spatialContext??i,o),e.hemisphereCenter&&t(e.hemisphereCenter,this.id,`hemisphereCenter`),f(e.positions,[`float32x3`],this.id),f(e.ringOffsets,[`uint32`],this.id),f(e.output.cells,[`uint32x2`],this.id),f(e.output.counts,[`uint32x2`],this.id),f(e.output.status,[`uint32`],this.id),m(this.id,`output.cells components`,e.output.cells.length*2,0,k),m(this.id,`positions components`,e.positions.length*3,0,k),m(this.id,`ringOffsets.length`,e.ringOffsets.length,2,k),e.output.visitedCount&&(f(e.output.visitedCount,[`uint32`],this.id),p(this.id,e.output.visitedCount.length===1,`output.visitedCount must hold one entry`)),m(this.id,`maximumLevel`,e.maximumLevel,0,A),m(this.id,`minimumLevel`,e.minimumLevel??0,0,e.maximumLevel),m(this.id,`capacity`,e.capacity),p(this.id,e.output.cells.length===e.capacity&&e.output.counts.length===1&&e.output.status.length===1,`output.cells must hold capacity entries, and output.counts and output.status one entry`),m(this.id,`maximumVisitedCells`,e.maximumVisitedCells??65536,6,j),r(this.id,e.tolerance??1e-6,n,a),p(this.id,e.refinement===void 0||e.refinement===`serial`||e.refinement===`level-synchronous`,`refinement must be 'serial' or 'level-synchronous'`),p(this.id,e.ringOffsets.length>=2,`ringOffsets must hold at least 2 entries (one ring)`),this.estimatedCommandNodeCount=e.refinement===`level-synchronous`?6+(e.maximumLevel+1)*(2+y(e.maximumVisitedCells??65536)):1,h(this.id,this.estimatedCommandNodeCount),S(this.id,e.output,e.refinement===`level-synchronous`?[{counts:e.output.counts,status:e.output.status,visitedCount:e.output.visitedCount}]:[e.output]),_(this.id,Object.values(e.output),[e.positions,e.ringOffsets])}getDeclarations(){let{props:e}=this;return`${l.source}\n${v}
const COVER_TOLERANCE:f32=${e.tolerance??1e-6};
fn coverPosition(row:u32)->vec3f {let offset=positionsOffset+row*3u;return vec3f(positions[offset],positions[offset+1u],positions[offset+2u]);}
fn coverOnArc(first:vec3f,second:vec3f,point:vec3f)->bool {let normal=normalize(cross(first,second));return abs(dot(normal,point))<=COVER_TOLERANCE&&dot(cross(first,point),normal)>=-COVER_TOLERANCE&&dot(cross(point,second),normal)>=-COVER_TOLERANCE;}
fn coverContains(point:vec3f)->vec2u {
  var inside=false;var boundary=false;
  for(var ring=0u;ring<${e.ringOffsets.length-1}u;ring++) {
    let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];var winding=0.0;
    for(var edge=begin;edge<end;edge++) {let first=coverPosition(edge);let second=coverPosition(select(edge+1u,begin,edge+1u==end));boundary=boundary||coverOnArc(first,second,point);let numerator=dot(point,cross(first,second));let denominator=dot(first,second)-dot(point,first)*dot(point,second);if(abs(numerator)+abs(denominator)>1e-12){winding+=s2AnalysisAtan2(numerator,denominator);}}
    if(ring==0u){inside=winding>3.14159265;}else{inside=inside&&winding<=3.14159265;}
  }
  return vec2u(select(0u,1u,${e.complement?`!inside`:`inside`}),select(0u,1u,boundary));
}
fn coverCellContains(normals:ptr<function,array<vec3f,4>>,point:vec3f)->bool {
  for(var edge=0u;edge<4u;edge++){if(dot((*normals)[edge],point)<-COVER_TOLERANCE){return false;}}
  return true;
}
fn coverCellOnArc(first:vec3f,second:vec3f,normal:vec3f,point:vec3f)->bool {
  return abs(dot(normal,point))<=COVER_TOLERANCE&&dot(cross(first,point),normal)>=-COVER_TOLERANCE&&dot(cross(point,second),normal)>=-COVER_TOLERANCE;
}
fn coverClassify(cell:vec2u)->vec2u {
  var intersects=false;var contained=true;var touched=false;
  var vertices:array<vec3f,4>;var normals:array<vec3f,4>;var crosses:array<vec3f,4>;
  for(var edge=0u;edge<4u;edge++){vertices[edge]=s2AnalysisVertex(cell,edge);}
  for(var edge=0u;edge<4u;edge++){crosses[edge]=cross(vertices[edge],vertices[(edge+1u)%4u]);normals[edge]=normalize(crosses[edge]);}
  // Polygon vertex containment does not depend on the current cell edge.
  for(var ring=0u;ring<${e.ringOffsets.length-1}u;ring++){
    let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];
    for(var vertex=begin;vertex<end;vertex++){if(coverCellContains(&normals,coverPosition(vertex))){touched=true;intersects=true;}}
  }
  for(var edge=0u;edge<4u;edge++) {
    let first=vertices[edge];let second=vertices[(edge+1u)%4u];let normal=normals[edge];
    let classification=coverContains(first);intersects=intersects||classification.x!=0u||classification.y!=0u;contained=contained&&classification.x!=0u&&classification.y==0u;
    for(var ring=0u;ring<${e.ringOffsets.length-1}u;ring++){
      let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];
      for(var vertex=begin;vertex<end;vertex++){
        let third=coverPosition(vertex);let fourth=coverPosition(select(vertex+1u,begin,vertex+1u==end));
        let crossing=cross(crosses[edge],cross(third,fourth));
        if(length(crossing)>COVER_TOLERANCE*COVER_TOLERANCE){
          let point=normalize(crossing);
          if((coverCellOnArc(first,second,normal,point)&&coverOnArc(third,fourth,point))||(coverCellOnArc(first,second,normal,-point)&&coverOnArc(third,fourth,-point))){touched=true;intersects=true;}
        }else if(coverCellOnArc(first,second,normal,third)||coverCellOnArc(first,second,normal,fourth)){touched=true;intersects=true;}
      }
    }
  }
  return vec2u(select(0u,1u,intersects),select(0u,1u,contained&&!touched));
}`}getCommandNodes(e){let{props:t,id:n}=this,r=t.refinement===`level-synchronous`,i=t.maximumVisitedCells??65536;return C(e,n,{positions:t.positions,ringOffsets:t.ringOffsets,...t.output},r?[[`frontier scratch`,i*8],[`count and offset scratch`,i*4],[`round totals scratch`,(t.maximumLevel+2)*4],[`control scratch`,8],[`ring center scratch`,(t.ringOffsets.length-1)*12]]:[]),r&&b(e,n,t.maximumLevel+1),g(n,e,[t.positions,t.ringOffsets,...Object.values(t.output)]),t.refinement===`level-synchronous`?this.getLevelSynchronousNodes(e):this.getSerialNodes(e)}getSerialNodes(e){let{props:t,id:n}=this;return[d(e,{id:n,operation:`GPUS2PolygonCovering`,variant:t.mode??`exterior`,invocationCount:1,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`offsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`cells`,view:t.output.cells,type:`u32`,access:`read_write`},{name:`counts`,view:t.output.counts,type:`u32`,access:`read_write`},{name:`status`,view:t.output.status,type:`u32`,access:`read_write`},...t.output.visitedCount?[{name:`visited`,view:t.output.visitedCount,type:`u32`,access:`read_write`}]:[]],declarations:this.getDeclarations(),body:`var failure=0u;var required=0u;var visitedCount=0u;
for(var ring=0u;ring<${t.ringOffsets.length-1}u;ring++){let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];if(begin>end||end>${t.positions.length}u||end-begin<3u){failure=1u;break;}
  var center=vec3f(0.0);for(var vertex=begin;vertex<end;vertex++){let point=coverPosition(vertex);let squared=dot(point,point);if((bitcast<u32>(squared)&0x7f800000u)==0x7f800000u||abs(squared-1.0)>1e-4){failure=1u;}center+=point;}
  ${t.hemisphereCenter?`center = vec3f(${t.hemisphereCenter.join(`,`)});`:``}
  if(length(center)<=COVER_TOLERANCE){failure=1u;break;}center=normalize(center);
  for(var vertex=begin;vertex<end&&failure==0u;vertex++){
    let next=select(vertex+1u,begin,vertex+1u==end);let first=coverPosition(vertex);let second=coverPosition(next);let normal=cross(first,second);
    if(dot(center,first)<=COVER_TOLERANCE||length(normal)<=COVER_TOLERANCE){failure=1u;break;}
    // A crossing or nonadjacent touch cannot define a simple ring or a guaranteed cover.
    for(var other=vertex+1u;other<end;other++){
      let otherNext=select(other+1u,begin,other+1u==end);if(other==next||otherNext==vertex){continue;}
      let third=coverPosition(other);let fourth=coverPosition(otherNext);let crossing=cross(normal,cross(third,fourth));
      if(length(crossing)>COVER_TOLERANCE*COVER_TOLERANCE){
        let intersection=normalize(crossing);
        if((coverOnArc(first,second,intersection)&&coverOnArc(third,fourth,intersection))||(coverOnArc(first,second,-intersection)&&coverOnArc(third,fourth,-intersection))){failure=1u;}
      }else if(coverOnArc(first,second,third)||coverOnArc(first,second,fourth)){failure=1u;}
    }
  }
}
var stack:array<vec2u,${M}>;var size=6u;
for(var face=0u;face<6u;face++){stack[5u-face]=vec2u((2u*face+1u)<<28u,0u);}
loop {if(size==0u||failure!=0u){break;}if(visitedCount>=${t.maximumVisitedCells??65536}u){failure|=4u;break;}size-=1u;let cell=stack[size];visitedCount+=1u;let classification=coverClassify(cell);if(classification.x==0u){continue;}let level=dggs_s2_get_level(cell);
  let emit=level>=${t.minimumLevel??0}u&&(classification.y!=0u||(${t.mode!==`interior`}&&level==${t.maximumLevel}u));
  if(emit){if(required<${t.capacity}u){cells[cellsOffset+required*2u]=cell.y;cells[cellsOffset+required*2u+1u]=cell.x;}required+=1u;}
  else if(level<${t.maximumLevel}u){let marker=dggs_u64_shift_left(vec2u(0u,1u),2u*(30u-level));let childMarker=dggs_u64_shift_right(marker,2u);let first=dggs_u64_add(dggs_u64_subtract(cell,marker),childMarker);let step=dggs_u64_shift_left(childMarker,1u);var child=first;for(var offset=0u;offset<4u;offset++){stack[size+3u-offset]=child;child=dggs_u64_add(child,step);}size+=4u;}
}
for(var slot=min(required,${t.capacity}u);slot<${t.capacity}u;slot++){cells[cellsOffset+slot*2u]=0u;cells[cellsOffset+slot*2u+1u]=0u;}
if(required>${t.capacity}u){failure|=2u;}counts[countsOffset]=min(required,${t.capacity}u);counts[countsOffset+1u]=required;status[statusOffset]=failure;
${t.output.visitedCount?`visited[visitedOffset]=visitedCount;`:``}`})]}getLevelSynchronousNodes(e){let{props:t,id:n}=this,r=`GPUS2PolygonCovering`,i=`${t.mode??`exterior`}-level-synchronous`,a=t.ringOffsets.length-1,o=t.positions.length,u=t.maximumLevel+1,f=t.maximumVisitedCells??65536,p=f,m=x(e,n,u),h=[c(e,`${n}-frontier-a`,`uint32x2`,f),c(e,`${n}-frontier-b`,`uint32x2`,f)],g=c(e,`${n}-round-counts`,`uint32`,f),_=c(e,`${n}-round-offsets`,`uint32`,f),v=c(e,`${n}-round-totals`,`uint32`,u+1),y=c(e,`${n}-control`,`uint32`,2),b=c(e,`${n}-ring-centers`,`float32x3`,a),S=this.getDeclarations(),C={name:`positions`,view:t.positions,type:`f32`,access:`read`},w={name:`offsets`,view:t.ringOffsets,type:`u32`,access:`read`},T=[...m.nodes,d(e,{id:`${n}-start`,operation:r,variant:i,invocationCount:1,bindings:[{name:`list`,view:h[0],type:`u32`,access:`read_write`},{name:`totals`,view:v,type:`u32`,access:`read_write`},{name:`control`,view:y,type:`u32`,access:`read_write`}],body:`for (var face = 0u; face < 6u; face++) {
  list[listOffset + face * 2u] = 0u; list[listOffset + face * 2u + 1u] = (2u * face + 1u) << 28u;
}
totals[totalsOffset] = 6u; control[controlOffset] = 0u; control[controlOffset + 1u] = 0u;`}),d(e,{id:`${n}-rings`,operation:r,variant:i,invocationCount:a,bindings:[C,w,{name:`centers`,view:b,type:`f32`,access:`read_write`},{name:`control`,view:y,type:`atomic<u32>`,access:`read_write`}],declarations:S,body:`let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
if (begin > end || end > ${o}u || end - begin < 3u) { atomicOr(&control[controlOffset], 1u); return; }
var center = vec3f(0.0); var invalid = false;
for (var vertex = begin; vertex < end; vertex++) {
  let point = coverPosition(vertex); let squared = dot(point, point);
  if ((bitcast<u32>(squared) & 0x7f800000u) == 0x7f800000u || abs(squared - 1.0) > 1e-4) { invalid = true; }
  center += point;
}
${t.hemisphereCenter?`center = vec3f(${t.hemisphereCenter.join(`,`)});`:``}
if (invalid || length(center) <= COVER_TOLERANCE) { atomicOr(&control[controlOffset], 1u); return; }
center = normalize(center);
centers[centersOffset + index * 3u] = center.x; centers[centersOffset + index * 3u + 1u] = center.y; centers[centersOffset + index * 3u + 2u] = center.z;`}),d(e,{id:`${n}-edges`,operation:r,variant:i,invocationCount:o,bindings:[C,w,{name:`centers`,view:b,type:`f32`,access:`read`},{name:`control`,view:y,type:`atomic<u32>`,access:`read_write`}],declarations:S,body:`if (atomicLoad(&control[controlOffset]) != 0u) { return; }
var ring = 0u; var begin = 0u; var end = 0u; var found = false;
for (var candidate = 0u; candidate < ${a}u; candidate++) {
  let first = offsets[offsetsOffset + candidate]; let last = offsets[offsetsOffset + candidate + 1u];
  if (index >= first && index < last) { ring = candidate; begin = first; end = last; found = true; break; }
}
if (!found) { return; }
let center = vec3f(centers[centersOffset + ring * 3u], centers[centersOffset + ring * 3u + 1u], centers[centersOffset + ring * 3u + 2u]);
let vertex = index; let next = select(vertex + 1u, begin, vertex + 1u == end);
let first = coverPosition(vertex); let second = coverPosition(next); let normal = cross(first, second);
if (dot(center, first) <= COVER_TOLERANCE || length(normal) <= COVER_TOLERANCE) { atomicOr(&control[controlOffset], 1u); return; }
for (var other = vertex + 1u; other < end; other++) {
  let otherNext = select(other + 1u, begin, other + 1u == end); if (other == next || otherNext == vertex) { continue; }
  let third = coverPosition(other); let fourth = coverPosition(otherNext); let crossing = cross(normal, cross(third, fourth));
  if (length(crossing) > COVER_TOLERANCE * COVER_TOLERANCE) {
    let intersection = normalize(crossing);
    if ((coverOnArc(first, second, intersection) && coverOnArc(third, fourth, intersection)) || (coverOnArc(first, second, -intersection) && coverOnArc(third, fourth, -intersection))) { atomicOr(&control[controlOffset], 1u); }
  } else if (coverOnArc(first, second, third) || coverOnArc(first, second, fourth)) { atomicOr(&control[controlOffset], 1u); }
}`})];for(let a=0;a<u;a++){let o=h[a%2],c=h[(a+1)%2];T.push(d(e,{id:`${n}-classify-${a}`,operation:r,variant:i,invocationCount:f,bindings:[{name:`roundParameters`,view:m.slots[a],type:`u32`,access:`read`},C,w,{name:`list`,view:o,type:`u32`,access:`read`},{name:`counts`,view:g,type:`u32`,access:`read_write`},{name:`control`,view:y,type:`atomic<u32>`,access:`read_write`},{name:`totals`,view:v,type:`u32`,access:`read`}],declarations:S,body:`let listCount = totals[totalsOffset + roundParameters[roundParametersOffset]];
var count = 0u;
if (index < listCount) {
  // A list longer than the frontier capacity cannot be written whole: that is exhaustion.
  if (listCount > ${f}u) { atomicOr(&control[controlOffset], 4u); }
  let cell = vec2u(list[listOffset + index * 2u + 1u], list[listOffset + index * 2u]);
  let level = dggs_s2_get_level(cell);
  if (level < roundParameters[roundParametersOffset]) { count = 1u; }
  else if (atomicLoad(&control[controlOffset]) == 0u) {
    let visit = atomicAdd(&control[controlOffset + 1u], 1u);
    if (visit >= ${p}u) { atomicOr(&control[controlOffset], 4u); }
    else {
      let classification = coverClassify(cell);
      if (classification.x != 0u) {
        let emit = level >= ${t.minimumLevel??0}u && (classification.y != 0u || (${t.mode!==`interior`} && level == ${t.maximumLevel}u));
        if (emit) { count = 1u; } else if (level < ${t.maximumLevel}u) { count = 4u; }
      }
    }
  }
}
counts[countsOffset + index] = count;`}),...new s({id:`${n}-scan-${a}`,input:g,output:_,mode:`exclusive`}).getCommandNodes(e),d(e,{id:`${n}-write-${a}`,operation:r,variant:i,invocationCount:f,bindings:[{name:`roundParameters`,view:m.slots[a],type:`u32`,access:`read`},{name:`source`,view:o,type:`u32`,access:`read`},{name:`destination`,view:c,type:`u32`,access:`read_write`},{name:`counts`,view:g,type:`u32`,access:`read`},{name:`offsets`,view:_,type:`u32`,access:`read`},{name:`totals`,view:v,type:`u32`,access:`read_write`}],declarations:l.source,body:`let count = counts[countsOffset + index]; let start = offsets[offsetsOffset + index];
// The scan total is the next round's list length; it is unclamped so overflow stays visible.
if (index == ${f-1}u) { totals[totalsOffset + roundParameters[roundParametersOffset] + 1u] = start + count; }
if (index < min(totals[totalsOffset + roundParameters[roundParametersOffset]], ${f}u) && count != 0u) {
  let cell = vec2u(source[sourceOffset + index * 2u + 1u], source[sourceOffset + index * 2u]);
  if (count == 1u) {
    if (start < ${f}u) { destination[destinationOffset + start * 2u] = cell.y; destination[destinationOffset + start * 2u + 1u] = cell.x; }
  } else {
    let level = dggs_s2_get_level(cell);
    let marker = dggs_u64_shift_left(vec2u(0u, 1u), 2u * (30u - level)); let childMarker = dggs_u64_shift_right(marker, 2u);
    var child = dggs_u64_add(dggs_u64_subtract(cell, marker), childMarker); let step = dggs_u64_shift_left(childMarker, 1u);
    for (var slot = 0u; slot < 4u; slot++) {
      if (start + slot < ${f}u) { destination[destinationOffset + (start + slot) * 2u] = child.y; destination[destinationOffset + (start + slot) * 2u + 1u] = child.x; }
      child = dggs_u64_add(child, step);
    }
  }
}`}))}let E=h[u%2];return T.push(d(e,{id:`${n}-cells`,operation:r,variant:i,invocationCount:t.capacity,bindings:[{name:`list`,view:E,type:`u32`,access:`read`},{name:`totals`,view:v,type:`u32`,access:`read`},{name:`control`,view:y,type:`u32`,access:`read`},{name:`cells`,view:t.output.cells,type:`u32`,access:`read_write`}],body:`var low = 0u; var high = 0u;
if ((control[controlOffset] & 5u) == 0u && index < min(totals[totalsOffset + ${u}u], ${f}u)) {
  low = list[listOffset + index * 2u]; high = list[listOffset + index * 2u + 1u];
}
cells[cellsOffset + index * 2u] = low; cells[cellsOffset + index * 2u + 1u] = high;`}),d(e,{id:`${n}-publish`,operation:r,variant:i,invocationCount:1,bindings:[{name:`totals`,view:v,type:`u32`,access:`read`},{name:`control`,view:y,type:`u32`,access:`read`},{name:`counts`,view:t.output.counts,type:`u32`,access:`read_write`},{name:`status`,view:t.output.status,type:`u32`,access:`read_write`},...t.output.visitedCount?[{name:`visited`,view:t.output.visitedCount,type:`u32`,access:`read_write`}]:[]],body:`var failure = control[controlOffset]; var written = 0u; var required = 0u;
if ((failure & 5u) == 0u) {
  required = totals[totalsOffset + ${u}u]; written = min(required, ${t.capacity}u);
  if (required > ${t.capacity}u) { failure |= 2u; }
}
counts[countsOffset] = written; counts[countsOffset + 1u] = required; status[statusOffset] = failure;
${t.output.visitedCount?`visited[visitedOffset] = min(control[controlOffset + 1u], ${p}u);`:``}`})),T}},P=1536,F=w({parameters:O,caption:`Adaptive S2 cover: coarse cells inside, fine cells on the boundary.`,createInputs:({world:t})=>({ring:e(3,t.width*.5,.5,.45,24),origin:[t.width*.5,.5],center:[t.width*.5,.5]}),build(e,t,n){let r=t.ring.length,i=e.allocate(`positions`,`float32x3`,r),a=e.allocate(`outline`,`float32x4`,r),o=e.upload(`ring-offsets`,`uint32`,Uint32Array.of(0,r)),s=e.allocate(`cells`,`uint32x2`,P),c=e.allocate(`counts`,`uint32x2`,1),l=e.allocate(`cover-status`,`uint32`,1);return e.graph.add(new N({positions:i.view,ringOffsets:o.view,minimumLevel:6,maximumLevel:n.maximumLevel,mode:n.mode===0?`exterior`:`interior`,capacity:P,output:{cells:s.view,counts:c.view,status:l.view}})),{outputs:{outline:a,counts:c,levels:E(e,`cover`,s),fill:T(e,`s2`,s)},update:e=>{let n=t.ring.map(([n,r])=>[t.center[0]+(n-t.origin[0])*e.size,t.center[1]+(r-t.origin[1])*e.size]);i.write(Float32Array.from(n.flatMap(([e,t])=>D(e,t)))),a.write(Float32Array.from(n.flatMap(([e,t],i)=>[e,t,...n[(i+1)%r]])))}}},rebuildOn:[`maximumLevel`,`mode`],interaction:{kind:`drag`,hint:`drag the polygon`,rest:({toUnit:e,inputs:t})=>e(t.origin),resolve(e,{inputs:t,outputs:n,peek:r}){t.center=[e.x,e.y];let i=r(n.counts),a=r(n.levels),o=`cover …`;if(i&&a){let e=30,t=0;for(let n=0;n<i[0];n++)e=Math.min(e,a[n]),t=Math.max(t,a[n]);o=`cover ${i[0]} cells · levels ${e} to ${t}`}return{rerun:!0,glyphs:[{shape:`crosshair`,at:t.center}],readout:o}}},draw:({outputs:e})=>[{layer:`polygons`,positions:e.fill.boundaries,indices:e.fill.indices,owners:e.fill.owners,values:e.levels,triangleCount:e.fill.triangleCount,ramp:`sequential`,range:[5,11],alpha:.75},{layer:`triangles`,positions:e.fill.boundaries,triangles:e.fill.wireframe,count:e.fill.wireframeCount,radius:1,color:[1,1,1,.5]},{layer:`lines`,context:!0,segments:e.outline,radius:1.8,color:[1,1,1,.95]}]});export{F as default};
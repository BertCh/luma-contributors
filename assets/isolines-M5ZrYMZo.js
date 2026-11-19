import{i as e,m as t,o as n,u as r}from"./raster-band-utils-BcTxc8_Y.js";import{n as i}from"./raster-storage-preflight-CIxFc-0g.js";import{r as a}from"./kernel-C3tfQOn4.js";import{t as o}from"./gpu-scan-B4NNTfF9.js";import{n as s,o as c,t as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{S as u,T as d,f,v as p,x as m}from"./gpu-contributor-utils-BIwLQON5.js";import{c as h,r as g,s as _}from"./raster-relaxation-Di9wjtxp.js";import{a as v,i as y,l as b,o as x,r as S,s as C,t as w}from"./raster-algebra-input-5ZxymrFJ.js";import{i as T,n as E,r as D,t as O}from"./isolines-utils-DRGA1fKt.js";import{t as k}from"./define-run-B-brkCe4.js";import{ft as A}from"./entry-preview-D70TdSbK.js";import{a as j,i as M,r as N,t as P}from"./iso-x4x8HW93.js";var F=2147483647,I=4294967295;function L(e,t,n){if(f(e,`grid.width`,t.width,2),f(e,`grid.height`,t.height,2),u(e,`grid samples`,t.width*t.height,F),!n)return;f(e,`tile.domainWidth`,n.domainWidth,2),f(e,`tile.domainHeight`,n.domainHeight,2),u(e,`tile domain samples`,n.domainWidth*n.domainHeight,F),f(e,`tile.columnOffset`,n.columnOffset,0,n.domainWidth-t.width),f(e,`tile.rowOffset`,n.rowOffset,0,n.domainHeight-t.height),d(e,n.ownedCells.length===4,`tile.ownedCells must contain four bounds`);let[r,i,a,o]=n.ownedCells;f(e,`tile.ownedCells.minColumn`,r,0,t.width-1),f(e,`tile.ownedCells.minRow`,i,0,t.height-1),f(e,`tile.ownedCells.maxColumn`,a,r,t.width-1),f(e,`tile.ownedCells.maxRow`,o,i,t.height-1)}function R(e,t,n,r){let[i,a,o,s]=n?.ownedCells??[0,0,t.width-1,t.height-1];u(e,`owned cells * levels.length * 2 required segments`,(o-i)*(s-a)*r*2,I)}var z=1073741824;function B(r,i,a=new Float32Array(8)){n(`getGPUIsolinesParameterValues`,a,8);let{width:o,height:s}=i,{levelCount:c,extent:l}=r;for(let[e,t]of[[`width`,o],[`height`,s]])if(!Number.isSafeInteger(t)||t<2)throw Error(`getGPUIsolinesParameterValues grid ${e} must be an integer of at least 2`);if(!Number.isSafeInteger(c)||c<0)throw Error(`Isolines levelCount must be a non-negative integer`);t(`Isolines`,l);let[u,d,f,p]=l,[m,h]=e(`Isolines`,r.localOrigin);a.fill(0,0,8),a[0]=c;let g=(f-u)/o,_=(p-d)/s,v=r.localOrigin?V(u,m,g,o):0,y=r.localOrigin?V(d,h,_,s):0;return a[1]=u-m+v*g,a[2]=d-h+y*_,a[3]=g,a[4]=_,a[5]=v,a[6]=y,a}function V(e,t,n,r){return Math.fround(Math.round(Math.max(0,Math.min(r-1,z,(t-e)/n))))}var H=`GPUIsolines`,U=class{id;props;requiredHalo=1;segmentCapacity;maximumLevelCount;constructor(e){this.id=e.id??`isolines`,this.props=e;let{id:t}=this,{output:n}=e,i=n?.polylines;r(t,e.grid);let{width:a,height:o}=e.grid;if(L(t,e.grid,e.tile),R(t,e.grid,e.tile,e.levels.length),e.noDataValue!==void 0&&!Number.isFinite(e.noDataValue))throw Error(`${t} noDataValue must be finite (NaN cells are always nodata)`);if(x(t,`values`,e.values,e.grid,a*o),T(t,`mask`,e.mask,`uint32`,a*o),T(t,`levels`,e.levels,`float32`,1),T(t,`parameters`,e.parameters,`float32`,8),!n?.segments)throw Error(`${t} needs output.segments`);this.maximumLevelCount=e.levels.length,this.segmentCapacity=n.segments.length;let s=this.segmentCapacity;if(s<1)throw Error(`${t} output.segments must hold at least one row`);if(T(t,`output.segments`,n.segments,`float32x4`,1),T(t,`output.segmentLevels`,n.segmentLevels,`uint32`,s),T(t,`output.segmentEdges`,n.segmentEdges,`uint32x2`,s),!n.status.count)throw Error(`${t} needs output.status.count`);if(!n.status?.overflow)throw Error(`${t} needs output.status.overflow`);for(let[e,r]of[[`status.count`,n.status.count],[`status.overflow`,n.status.overflow],[`status.requiredCount`,n.status.requiredCount]])T(t,`output.${e}`,r,`uint32`,1);if(i){T(t,`output.polylines.vertices`,i.vertices,`float32x2`,1),T(t,`output.polylines.polylineOffsets`,i.polylineOffsets,`uint32`,s+1),T(t,`output.polylines.polylineLevels`,i.polylineLevels,`uint32`,s),T(t,`output.polylines.polylineClosed`,i.polylineClosed,`uint32`,s);for(let e of[`polylineCount`,`vertexCount`]){if(!i[e])throw Error(`${t} needs output.polylines.${e}`);T(t,`output.polylines.${e}`,i[e],`uint32`,1)}if(!n.status.polylineOverflow)throw Error(`${t} needs output.status.polylineOverflow`);T(t,`output.status.polylineOverflow`,n.status.polylineOverflow,`uint32`,1)}D(t,b(n),{...w(e.values,`values`),mask:e.mask,levels:e.levels,parameters:e.parameters})}getOutputViews(){let{output:e}=this.props,{polylines:t,status:n}=e;return[e.segments,e.segmentLevels,e.segmentEdges,e.status.count,e.status.requiredCount,n.overflow,n.polylineOverflow,t?.vertices,t?.polylineOffsets,t?.polylineLevels,t?.polylineClosed,t?.polylineCount,t?.vertexCount].filter(e=>e!==void 0)}getCommandNodes(e){let t=this.props;C(e,this.id,[t.values],t.grid),i(e,this.id,[t.mask,t.levels,t.parameters,...this.getOutputViews()],{"cell counts and offsets":(t.grid.width-1)*(t.grid.height-1)*4,"segment edges":t.output.polylines&&!t.output.segmentEdges?this.segmentCapacity*8:0,"stitch states":t.output.polylines?this.segmentCapacity*16:0,"staged polyline vertices":(t.output.polylines?.vertices.length??0)*8,"stitch totals":t.output.polylines?8:4});let{id:n,props:r,segmentCapacity:l,maximumLevelCount:u}=this,{output:d}=r,{polylines:f}=d,{width:h,height:g}=r.grid;m(n,e,[...y(r.values),r.mask,r.levels,r.parameters,...this.getOutputViews()]);let _=(h-1)*(g-1);for(let[t,r]of[[`cell scratch`,_*4],[`segment scratch`,l*8],[`polyline scratch`,(f?.vertices.length??0)*8]])p(e.device,n,t,r,`use smaller tiles or capacities`);let b=S(e,`${n}-source`,r.values,r.grid),x=[...b.nodes],w=r.tile?.ownedCells??[0,0,h-1,g-1],T=!!r.mask,D=`const WIDTH: u32 = ${h}u;
const HEIGHT: u32 = ${g}u;
const MAXIMUM_LEVELS: u32 = ${u}u;
const CAPACITY: u32 = ${l}u;
${E(T,v(r.values)?void 0:r.noDataValue)}
fn ownsCell(column: u32, row: u32) -> bool {
  return column >= ${w[0]}u && column < ${w[2]}u &&
    row >= ${w[1]}u && row < ${w[3]}u;
}`,O=[{name:`params`,view:r.parameters,type:`f32`,access:`read`},{name:`levels`,view:r.levels,type:`f32`,access:`read`},{name:`raster`,view:b.values,type:`f32`,access:`read`},...r.mask?[{name:`validity`,view:r.mask,type:`u32`,access:`read`}]:[]],k=a(e,`${n}-cell-counts`,`uint32`,_),A=a(e,`${n}-cell-offsets`,`uint32`,_),j=a(e,`${n}-total`,`uint32`,1),M=!!f,N=d.segmentEdges??(M?a(e,`${n}-edges`,`uint32x2`,l):void 0);if(x.push(c(e,{id:`${n}-count`,operation:H,variant:`count`,bindings:[...O,{name:`counts`,view:k,type:`u32`,access:`read_write`}],invocationCount:_,declarations:D,body:`let cx = index % (WIDTH - 1u);
  let cy = index / (WIDTH - 1u);
  let corners = loadCorners(cx, cy);
  var cellTotal = 0u;
  if (ownsCell(cx, cy) && corners.valid) {
    let levelCount = min(u32(max(params[paramsOffset], 0.0)), MAXIMUM_LEVELS);
    for (var levelIndex = 0u; levelIndex < levelCount; levelIndex++) {
      let mask = getHighMask(corners.v, levels[levelsOffset + levelIndex]);
      cellTotal += countOneBits(getExitMask(mask));
    }
  }
  counts[countsOffset + index] = cellTotal;`}),...new o({id:`${n}-cell-scan`,input:k,output:A}).getCommandNodes(e),c(e,{id:`${n}-total`,operation:H,variant:`total`,bindings:[{name:`counts`,view:k,type:`u32`,access:`read`},{name:`offsets`,view:A,type:`u32`,access:`read`},{name:`totalOut`,view:j,type:`u32`,access:`read_write`}],invocationCount:1,body:`totalOut[totalOutOffset] =
    offsets[offsetsOffset + ${_-1}u] + counts[countsOffset + ${_-1}u];`})),x.push(c(e,{id:`${n}-scatter`,operation:H,variant:`scatter`,bindings:[...O,{name:`offsets`,view:A,type:`u32`,access:`read`},{name:`segments`,view:d.segments,type:`f32`,access:`read_write`},{name:`segmentLevels`,view:d.segmentLevels,type:`u32`,access:`read_write`},...N?[{name:`edges`,view:N,type:`u32`,access:`read_write`}]:[]],invocationCount:_,declarations:`${D}
${W(r.tile)}`,body:`let cx = index % (WIDTH - 1u);
  let cy = index / (WIDTH - 1u);
  let corners = loadCorners(cx, cy);
  if (!ownsCell(cx, cy) || !corners.valid) {
    return;
  }
  let v = corners.v;
  var position = offsets[offsetsOffset + index];
  let levelCount = min(u32(max(params[paramsOffset], 0.0)), MAXIMUM_LEVELS);
  for (var levelIndex = 0u; levelIndex < levelCount; levelIndex++) {
    let level = levels[levelsOffset + levelIndex];
    let mask = getHighMask(v, level);
    let exits = getExitMask(mask);
    if (exits == 0u) {
      continue;
    }
    let entries = getEntryMask(mask);
    // Saddles have two exits. High corners are joined when the bilinear saddle value is high.
    let isJoined = countOneBits(exits) < 2u || isSaddleJoined(v, level);
    for (var exitEdge = 0u; exitEdge < 4u; exitEdge++) {
      if (((exits >> exitEdge) & 1u) == 0u) {
        continue;
      }
      let entryEdge = getPairedEntry(exitEdge, entries, isJoined);
      // A corner exactly at the level gives a zero-length segment. It stays: it is the bridge
      // that links the neighbouring cells' chains in the edge-based stitching.
      if (position < CAPACITY) {
        let start = getEdgeCrossing(exitEdge, level, v, cx, cy);
        let end = getEdgeCrossing(entryEdge, level, v, cx, cy);
        let base = segmentsOffset + 4u * position;
        segments[base] = start.x;
        segments[base + 1u] = start.y;
        segments[base + 2u] = end.x;
        segments[base + 3u] = end.y;
        segmentLevels[segmentLevelsOffset + position] = levelIndex;
        ${N?`edges[edgesOffset + 2u * position] = getEdgeId(exitEdge, cx, cy);
        edges[edgesOffset + 2u * position + 1u] = getEdgeId(entryEdge, cx, cy);`:``}
      }
      position++;
    }
  }`}),s(e,{id:`${n}-publish`,operation:H,requiredCount:j,output:{ids:d.segmentLevels,status:{count:d.status.count,overflow:d.status.overflow,requiredCount:d.status.requiredCount}}})),f&&N&&x.push(...this.getStitchNodes(e,{counts:k,offsets:A,edges:N,polylines:f,common:G(h,g)})),r.tile&&d.segmentEdges){let{domainWidth:t,domainHeight:i,columnOffset:a,rowOffset:o}=r.tile;x.push(c(e,{id:`${n}-domain-edges`,operation:H,variant:`domain-edges`,bindings:[{name:`edges`,view:d.segmentEdges,type:`u32`,access:`read_write`},{name:`count`,view:d.status.count,type:`u32`,access:`read`}],invocationCount:l,body:`if (index >= count[countOffset]) { return; }
  for (var endpoint = 0u; endpoint < 2u; endpoint++) {
    let position = edgesOffset + 2u * index + endpoint;
    let edge = edges[position];
    var domainEdge = 0u;
    if (edge < ${g*(h-1)}u) {
      domainEdge = (edge / ${h-1}u + ${o}u) * ${t-1}u +
        edge % ${h-1}u + ${a}u;
    } else {
      let vertical = edge - ${g*(h-1)}u;
      domainEdge = ${i*(t-1)}u +
        (vertical / ${h}u + ${o}u) * ${t}u +
        vertical % ${h}u + ${a}u;
    }
    edges[position] = domainEdge;
  }`}))}return x}getStitchNodes(e,t){let{id:n,props:r,segmentCapacity:i}=this,{output:s}=r,{counts:u,offsets:d,edges:f,polylines:p}=t,m=[],v=(t,r=i)=>a(e,`${n}-${t}`,`uint32`,r),y=(r,a,o,s=``,l=i)=>m.push(c(e,{id:`${n}-${r}`,operation:H,variant:r,bindings:a,invocationCount:l,declarations:`const CAPACITY: u32 = ${i}u;
${t.common}
${s}`,body:o})),b=(e,t)=>({name:e,view:t,type:`u32`,access:`read`}),x=(e,t)=>({name:e,view:t,type:`u32`,access:`read_write`}),S=v(`active`,1);y(`stitch-active`,[b(`countIn`,s.status.count),b(`overflowIn`,s.status.overflow),x(`activeOut`,S)],`activeOut[activeOutOffset] =
    select(min(countIn[countInOffset], CAPACITY), 0u, overflowIn[overflowInOffset] != 0u);`,``,1);let C=v(`previous`);y(`links`,[b(`edges`,f),b(`levels`,s.segmentLevels),b(`offsets`,d),b(`counts`,u),b(`activeCount`,S),x(`previousOut`,C)],`var previousSegment = NONE;
  if (index < activeCount[activeCountOffset]) {
    let startEdge = edges[edgesOffset + 2u * index];
    let endEdge = edges[edgesOffset + 2u * index + 1u];
    let level = levels[levelsOffset + index];
    previousSegment = findLinkedSegment(getNeighbourCell(endEdge, startEdge), level, 1u, startEdge);
  }
  previousOut[previousOutOffset + index] = previousSegment;`,K());let w=2*Math.ceil((Math.ceil(Math.log2(i))+1)/2),T=[0,1].map(e=>v(`jump-${e}-state`,4*i));y(`jump-init`,[b(`previous`,C),x(`stateOut`,T[0])],`let hasPrevious = previous[previousOffset + index] != NONE;
  let base = stateOutOffset + 4u * index;
  stateOut[base] = select(index, previous[previousOffset + index], hasPrevious);
  stateOut[base + 1u] = select(0u, index + 1u, hasPrevious);
  stateOut[base + 2u] = 0u;
  stateOut[base + 3u] = select(0u, 1u, hasPrevious);`);let E=h(e,`${n}-jump`,H,i,n);m.push(_(e,{id:`${n}-jump-reset`,operation:H,state:E}),...g(e,{id:`${n}-jump`,operation:H,state:E,maxIterations:w,groupSize:2,createRound:({nodeId:t,iteration:n,slotBinding:r,roundSlotWGSL:i,condition:a,extraResources:o})=>c(e,{id:t,operation:H,variant:`jump`,bindings:[b(`stateIn`,T[n%2]),x(`stateOut`,T[(n+1)%2]),{name:`status`,view:E.status,type:`atomic<u32>`,access:`read_write`},r],invocationCount:E.invocationCount,body:`let ownBase = stateInOffset + 4u * index;
  let jump = stateIn[ownBase];
  let jumpBase = stateInOffset + 4u * jump;
  let ownKey = stateIn[ownBase + 1u];
  let jumpKey = stateIn[jumpBase + 1u];
  let ownWins = ownKey <= jumpKey;
  let outBase = stateOutOffset + 4u * index;
  stateOut[outBase] = stateIn[jumpBase];
  stateOut[outBase + 1u] = select(jumpKey, ownKey, ownWins);
  stateOut[outBase + 2u] = select(
    stateIn[jumpBase + 2u] + stateIn[ownBase + 3u],
    stateIn[ownBase + 2u],
    ownWins
  );
  stateOut[outBase + 3u] = stateIn[ownBase + 3u] + stateIn[jumpBase + 3u];
  let roundBit = 1u << ${i};
  if (!ownWins && (atomicLoad(&status[statusOffset]) & roundBit) == 0u) {
    atomicOr(&status[statusOffset], roundBit);
  }`,condition:a,extraResources:o})}));let D=T[0],O=v(`head-of`),k=v(`rank-of`);y(`resolve`,[b(`stateIn`,D),x(`headOut`,O),x(`rankOut`,k)],`let base = stateInOffset + 4u * index;
  let key = stateIn[base + 1u];
  let isRing = key != 0u;
  headOut[headOutOffset + index] = select(stateIn[base], key - 1u, isRing);
  rankOut[rankOutOffset + index] =
    stateIn[base + 2u] | select(0u, 0x80000000u, isRing);`);let A=v(`chain-length`),j=v(`head-kind`);m.push(l(e,{id:`${n}-chain-length-clear`,operation:H,view:A,type:`u32`,value:`0u`})),m.push(c(e,{id:`${n}-chain-length`,operation:H,variant:`chain-length`,bindings:[b(`headOf`,O),b(`rankOf`,k),b(`activeCount`,S),{name:`chainLength`,view:A,type:`atomic<u32>`,access:`read_write`},x(`headKindOut`,j)],invocationCount:i,body:`var kind = 0u;
  if (index < activeCount[activeCountOffset]) {
    let packed = rankOf[rankOfOffset + index];
    let rank = packed & 0x7fffffffu;
    atomicMax(&chainLength[chainLengthOffset + headOf[headOfOffset + index]], rank + 1u);
    kind = select(0u, select(1u, 2u, (packed >> 31u) != 0u), rank == 0u);
  }
  headKindOut[headKindOutOffset + index] = kind;`}));let M=v(`head-flag`),N=v(`head-vertex-count`),P=v(`head-index`),F=v(`vertex-offset`);y(`scan-input`,[b(`headKind`,j),b(`chainLength`,A),x(`flagOut`,M),x(`vertexCountOut`,N)],`let isHead = headKind[headKindOffset + index] != 0u;
  flagOut[flagOutOffset + index] = select(0u, 1u, isHead);
  vertexCountOut[vertexCountOutOffset + index] =
    select(0u, chainLength[chainLengthOffset + index] + 1u, isHead);`),m.push(...new o({id:`${n}-head-scan`,input:M,output:P}).getCommandNodes(e),...new o({id:`${n}-vertex-scan`,input:N,output:F}).getCommandNodes(e)),y(`polyline-records`,[b(`headKind`,j),b(`headIndex`,P),b(`vertexOffset`,F),b(`segmentLevels`,s.segmentLevels),x(`polylineOffsets`,p.polylineOffsets),x(`polylineLevels`,p.polylineLevels),x(`polylineClosed`,p.polylineClosed)],`let kind = headKind[headKindOffset + index];
  if (kind != 0u) {
    let polyline = headIndex[headIndexOffset + index];
    polylineOffsets[polylineOffsetsOffset + polyline] = vertexOffset[vertexOffsetOffset + index];
    polylineLevels[polylineLevelsOffset + polyline] = segmentLevels[segmentLevelsOffset + index];
    polylineClosed[polylineClosedOffset + polyline] = select(0u, 1u, kind == 2u);
  }`);let I=p.vertices.length,L=a(e,`${n}-staged-vertices`,`float32x2`,I);m.push(c(e,{id:`${n}-vertices`,operation:H,variant:`vertices`,bindings:[b(`headOf`,O),b(`rankOf`,k),b(`activeCount`,S),b(`chainLength`,A),b(`vertexOffset`,F),{name:`segments`,view:s.segments,type:`f32`,access:`read`},{name:`vertices`,view:L,type:`f32`,access:`read_write`}],invocationCount:i,declarations:`const VERTEX_CAPACITY: u32 = ${I}u;`,body:`if (index >= activeCount[activeCountOffset]) {
    return;
  }
  let head = headOf[headOfOffset + index];
  let packed = rankOf[rankOfOffset + index];
  let rank = packed & 0x7fffffffu;
  let length = chainLength[chainLengthOffset + head];
  let base = vertexOffset[vertexOffsetOffset + head];
  if (base + length + 1u > VERTEX_CAPACITY) {
    return;
  }
  let own = segmentsOffset + 4u * index;
  let slot = verticesOffset + 2u * (base + rank);
  vertices[slot] = segments[own];
  vertices[slot + 1u] = segments[own + 1u];
  if (rank + 1u == length) {
    // The last vertex of a ring repeats the first one; an open chain ends at its last p1.
    let closing = (packed >> 31u) != 0u;
    let source = select(own + 2u, segmentsOffset + 4u * head, closing);
    vertices[slot + 2u] = segments[source];
    vertices[slot + 3u] = segments[source + 1u];
  }`}));let R=v(`stitch-totals`,2);y(`stitch-totals`,[b(`headIndex`,P),b(`headFlag`,M),b(`vertexOffset`,F),b(`vertexCount`,N),x(`totalsOut`,R)],`totalsOut[totalsOutOffset] =
    headIndex[headIndexOffset + CAPACITY - 1u] + headFlag[headFlagOffset + CAPACITY - 1u];
  totalsOut[totalsOutOffset + 1u] =
    vertexOffset[vertexOffsetOffset + CAPACITY - 1u] +
    vertexCount[vertexCountOffset + CAPACITY - 1u];`,``,1),y(`stitch-publish`,[b(`totals`,R),b(`segmentOverflow`,s.status.overflow),x(`polylineOffsets`,p.polylineOffsets),x(`polylineCountOut`,p.polylineCount),x(`vertexCountOut`,p.vertexCount),x(`overflowOut`,s.status.polylineOverflow)],`let polylineTotal = totals[totalsOffset];
  let vertexTotal = totals[totalsOffset + 1u];
  polylineOffsets[polylineOffsetsOffset + polylineTotal] = vertexTotal;
  // Offsets are nondecreasing: keep the longest prefix of polylines that fits the vertex capacity.
  var low = 0u;
  var high = polylineTotal;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high + 1u) / 2u;
    if (polylineOffsets[polylineOffsetsOffset + middle] <= VERTEX_CAPACITY) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  polylineCountOut[polylineCountOutOffset] = low;
  vertexCountOut[vertexCountOutOffset] = polylineOffsets[polylineOffsetsOffset + low];
  overflowOut[overflowOutOffset] =
    select(0u, 1u, segmentOverflow[segmentOverflowOffset] != 0u || low < polylineTotal);`,`const VERTEX_CAPACITY: u32 = ${I}u;`,1);let z=v(`start-flag`,I),B=v(`keep-flag`,I),V=v(`keep-offset`,I),U=`const VERTEX_CAPACITY: u32 = ${I}u;`,W={name:`staged`,view:L,type:`f32`,access:`read`};return m.push(l(e,{id:`${n}-start-flag-clear`,operation:H,view:z,type:`u32`,value:`0u`})),y(`start-flags`,[b(`headKind`,j),b(`vertexOffset`,F),x(`startOut`,z)],`if (headKind[headKindOffset + index] != 0u) {
    let first = vertexOffset[vertexOffsetOffset + index];
    if (first < VERTEX_CAPACITY) {
      startOut[startOutOffset + first] = 1u;
    }
  }`,U),y(`keep-flags`,[W,b(`startFlag`,z),b(`vertexCount`,p.vertexCount),x(`keepOut`,B)],`var keep = 0u;
  if (index < vertexCount[vertexCountOffset]) {
    let current = staged[stagedOffset + 2u * index];
    let currentY = staged[stagedOffset + 2u * index + 1u];
    let isStart = startFlag[startFlagOffset + index] != 0u || index == 0u;
    // Bit-identical neighbours inside one polyline collapse; polyline starts always stay.
    let isDuplicate = !isStart &&
      staged[stagedOffset + 2u * index - 2u] == current &&
      staged[stagedOffset + 2u * index - 1u] == currentY;
    keep = select(1u, 0u, isDuplicate);
  }
  keepOut[keepOutOffset + index] = keep;`,U,I),m.push(...new o({id:`${n}-keep-scan`,input:B,output:V}).getCommandNodes(e)),y(`compact-vertices`,[W,b(`keepFlag`,B),b(`keepOffset`,V),{name:`vertices`,view:p.vertices,type:`f32`,access:`read_write`}],`if (keepFlag[keepFlagOffset + index] != 0u) {
    let destination = verticesOffset + 2u * keepOffset[keepOffsetOffset + index];
    vertices[destination] = staged[stagedOffset + 2u * index];
    vertices[destination + 1u] = staged[stagedOffset + 2u * index + 1u];
  }`,U,I),y(`compact-offsets`,[b(`keepFlag`,B),b(`keepOffset`,V),b(`polylineCount`,p.polylineCount),b(`vertexCount`,p.vertexCount),x(`polylineOffsets`,p.polylineOffsets)],`if (index <= polylineCount[polylineCountOffset]) {
    let old = polylineOffsets[polylineOffsetsOffset + index];
    let total = keepOffset[keepOffsetOffset + VERTEX_CAPACITY - 1u] +
      keepFlag[keepFlagOffset + VERTEX_CAPACITY - 1u];
    polylineOffsets[polylineOffsetsOffset + index] =
      select(keepOffset[keepOffsetOffset + old], total, old >= vertexCount[vertexCountOffset]);
  }`,U,i+1),y(`compact-totals`,[b(`keepFlag`,B),b(`keepOffset`,V),x(`vertexCountOut`,p.vertexCount)],`vertexCountOut[vertexCountOutOffset] =
    keepOffset[keepOffsetOffset + VERTEX_CAPACITY - 1u] +
    keepFlag[keepFlagOffset + VERTEX_CAPACITY - 1u];`,U,1),m}};function W(e){return`${O}

// The entry edge a segment leaving through exitEdge ends on, walking the cell boundary.
fn getPairedEntry(exitEdge: u32, entries: u32, isJoined: bool) -> u32 {
  for (var step = 1u; step < 4u; step++) {
    let candidate = select((exitEdge + 4u - step) % 4u, (exitEdge + step) % 4u, isJoined);
    if (((entries >> candidate) & 1u) != 0u) {
      return candidate;
    }
  }
  return exitEdge;
}

fn getEdgeCrossing(k: u32, level: f32, v: vec4<f32>, cx: u32, cy: u32) -> vec2<f32> {
  let anchorX = params[paramsOffset + 1u];
  let anchorY = params[paramsOffset + 2u];
  let cellWidth = params[paramsOffset + 3u];
  let cellHeight = params[paramsOffset + 4u];
  let columnAnchor = i32(params[paramsOffset + 5u]);
  let rowAnchor = i32(params[paramsOffset + 6u]);
  // Canonical orientation: horizontal edges run +x, vertical edges run +y.
  var a = v.x;
  var b = v.y;
  if (k == 1u) {
    a = v.y;
    b = v.z;
  } else if (k == 2u) {
    a = v.w;
    b = v.z;
  } else if (k == 3u) {
    a = v.x;
    b = v.w;
  }
  let t = (level - a) / (b - a);
  if (k == 0u || k == 2u) {
    let gridX = f32(i32(cx + ${e?.columnOffset??0}u) - columnAnchor) + t;
    let gridY = f32(i32(cy + ${e?.rowOffset??0}u + select(0u, 1u, k == 2u)) - rowAnchor);
    return vec2<f32>(anchorX + (gridX + 0.5) * cellWidth, anchorY + (gridY + 0.5) * cellHeight);
  }
  let gridX = f32(i32(cx + ${e?.columnOffset??0}u + select(0u, 1u, k == 1u)) - columnAnchor);
  let gridY = f32(i32(cy + ${e?.rowOffset??0}u) - rowAnchor) + t;
  return vec2<f32>(anchorX + (gridX + 0.5) * cellWidth, anchorY + (gridY + 0.5) * cellHeight);
}`}function G(e,t){return`
const WIDTH: u32 = ${e}u;
const HEIGHT: u32 = ${t}u;
const NONE: u32 = 0xffffffffu;

// The two cells that share edge e: x is below or left of it, y above or right of it.
fn getEdgeCells(e: u32) -> vec2<u32> {
  let cellColumns = WIDTH - 1u;
  let cellRows = HEIGHT - 1u;
  let horizontalCount = HEIGHT * cellColumns;
  var low = NONE;
  var high = NONE;
  if (e < horizontalCount) {
    let r = e / cellColumns;
    let c = e % cellColumns;
    if (r >= 1u) { low = (r - 1u) * cellColumns + c; }
    if (r < cellRows) { high = r * cellColumns + c; }
  } else {
    let i = e - horizontalCount;
    let r = i / WIDTH;
    let c = i % WIDTH;
    if (c >= 1u && r < cellRows) { low = r * cellColumns + c - 1u; }
    if (c < cellColumns && r < cellRows) { high = r * cellColumns + c; }
  }
  return vec2<u32>(low, high);
}

fn cellHasEdge(cell: u32, e: u32) -> bool {
  let cellColumns = WIDTH - 1u;
  let cx = cell % cellColumns;
  let cy = cell / cellColumns;
  let bottom = cy * cellColumns + cx;
  let left = HEIGHT * cellColumns + cy * WIDTH + cx;
  return e == bottom || e == bottom + cellColumns || e == left || e == left + 1u;
}

// Cell on the other side of edge \`across\`, given another edge of the segment's own cell.
fn getNeighbourCell(known: u32, across: u32) -> u32 {
  let cells = getEdgeCells(across);
  if (cells.x != NONE && cellHasEdge(cells.x, known)) {
    return cells.y;
  }
  return cells.x;
}`}function K(){return`
// Segment of \`level\` in \`cell\` whose start (field 0) or end (field 1) edge is \`edge\`.
fn findLinkedSegment(cell: u32, level: u32, field: u32, edge: u32) -> u32 {
  if (cell == NONE) {
    return NONE;
  }
  let first = offsets[offsetsOffset + cell];
  let segmentCount = counts[countsOffset + cell];
  for (var i = 0u; i < segmentCount; i++) {
    let position = first + i;
    if (position >= CAPACITY) {
      break;
    }
    if (levels[levelsOffset + position] == level &&
        edges[edgesOffset + 2u * position + field] == edge) {
      return position;
    }
  }
  return NONE;
}`}var q=2e4,J=k({parameters:A,caption:`Contour lines at fixed levels over a surface.`,createInputs:e=>({...P(e),cursorValue:null}),build(e,t){let n=e.upload(`values`,`float32`,t.valuesBottomUp),r=e.upload(`display`,`float32`,t.valuesTopDown),i=e.allocate(`levels`,`float32`,8),a=e.allocate(`parameters`,`float32`,12),o=e.allocate(`segments`,`float32x4`,q),s=e.allocate(`segment-count`,`uint32`,1);e.graph.add(new U({grid:t.grid,values:n.view,levels:i.view,parameters:a.view,output:{segments:o.view,segmentLevels:e.allocate(`segment-levels`,`uint32`,q).view,status:{count:s.view,overflow:e.allocate(`overflow`,`uint32`,1).view}}}));let c=e.allocate(`cursor-level`,`float32`,8),l=e.allocate(`cursor-parameters`,`float32`,12),u=e.allocate(`cursor-segments`,`float32x4`,q),d=e.allocate(`cursor-count`,`uint32`,1);return e.graph.add(new U({id:`cursor-isolines`,grid:t.grid,values:n.view,levels:c.view,parameters:l.view,output:{segments:u.view,segmentLevels:e.allocate(`cursor-segment-levels`,`uint32`,q).view,status:{count:d.view,overflow:e.allocate(`cursor-overflow`,`uint32`,1).view}}})),{outputs:{display:r,segments:o,segmentCount:s,cursorSegments:u,cursorCount:d},update:e=>{let n=t.cursorValue;i.write(n===null?N(e.interval,e.interval):M(n,e.interval)),a.write(B({levelCount:Math.round(e.levelCount),extent:t.extent},t.grid)),c.write(Float32Array.of(n??e.interval)),l.write(B({levelCount:1,extent:t.extent},t.grid))}}},interaction:{kind:`level`,hint:`pick a level`,rest:[.55,.45],resolve(e,{inputs:t,values:n}){let r=Math.round(j(t,e.x,e.y)*10)/10,i=t.cursorValue!==r;return t.cursorValue=r,{rerun:i,glyphs:[{shape:`crosshair`,at:[e.x,e.y]}],readout:`level ${r.toFixed(1)} · interval ${n.interval}`}}},draw:({inputs:e,outputs:t})=>[{layer:`raster`,context:!0,values:t.display,columns:e.columns,rows:e.rows,smoothing:!0},{layer:`lines`,segments:t.segments,count:t.segmentCount,radius:1.1,color:[1,1,1,.75]},{layer:`lines`,segments:t.cursorSegments,count:t.cursorCount,radius:2.4,color:[1,.78,.2,1]}]});export{J as default};
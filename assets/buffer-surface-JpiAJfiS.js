import{n as e,r as t,t as n}from"./gpu-offset-curve-Bj8gXMzb.js";import{t as r}from"./ports-xmdKgL07.js";import{a as i}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as a,n as o,r as s,t as c}from"./gpu-polygon-overlay-Bu61B05s.js";import{r as l,u}from"./kernel-C3tfQOn4.js";import{t as d}from"./gpu-scan-B4NNTfF9.js";import{o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as p,S as m,f as h,g,v as _,w as v,x as y}from"./gpu-contributor-utils-BIwLQON5.js";import{t as b}from"./gpu-geometry-orientation-BW3bBZIA.js";import{n as x}from"./gpu-snap-rounded-noding-BglCAlqG.js";import{s as S,v as C}from"./spatial-context-DwZEd4EZ.js";import{t as w}from"./define-run-B-brkCe4.js";import{d as T,s as E}from"./pointer-overlay-Qh8qflcA.js";import{f as D,h as O,l as k,s as A}from"./lines-kx5qHdti.js";import{a as j,c as M,i as N,l as P,n as F,r as I}from"./polygons-D0hzhPvj.js";import{us as L}from"./entry-preview-D70TdSbK.js";var R=Object.freeze({...S,measure:`distance`}),z=`GPUBufferSurface`,B=2**-126,V=2**127;function H(e){return x(e,0)}var U=class{id;props;spatialContext;joinStyle;capStyle;quadSegments;rowsPerVertex;constructor(e){this.id=e.id??`buffer-surface`,this.spatialContext=C(this.id,e.spatialContext,R),e={...e,geometry:r(this.id,`geometry`,e.geometry)},this.props=e,g(this.id,`vertexTolerance`,e.vertexTolerance,0,V),g(this.id,`vertexTolerance converted to float32`,Math.fround(e.vertexTolerance),B,V),o(this.id,e.output),this.joinStyle=e.joinStyle??`round`,this.capStyle=e.capStyle??`round`,this.quadSegments=e.quadSegments??8,this.rowsPerVertex=t(this.joinStyle,this.quadSegments,`buffer`);let{id:n}=this,{geometry:i}=e;if(p(i.positions,[`float32x2`],`${n} geometry.positions`),i.positions.length<1)throw Error(`${n} geometry.positions must not be empty`);if(i.kind===`lines`){if(u(i.lineOffsets,`${n} geometry.lineOffsets`),i.lineOffsets.length<2)throw Error(`${n} lineOffsets must contain at least two entries`)}else for(let[e,t]of[[`featureOffsets`,i.featureOffsets],[`polygonOffsets`,i.polygonOffsets],[`ringOffsets`,i.ringOffsets]])if(u(t,`${n} geometry.${e}`),t.length<2)throw Error(`${n} geometry.${e} must contain at least two entries`);if(p(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<4)throw Error(`${n} parameters must contain four float32 values`);if(![`round`,`mitre`,`bevel`].includes(this.joinStyle))throw Error(`${n} joinStyle must be round, mitre or bevel`);if(![`round`,`flat`,`square`].includes(this.capStyle))throw Error(`${n} capStyle must be round, flat or square`);h(n,`quadSegments`,this.quadSegments,1,256),this.props.leafCapacity!==void 0&&v(n,`leafCapacity`,this.props.leafCapacity)}getCommandNodes(e){let{geometry:t}=this.props;y(this.id,e,[t.positions,t.kind===`lines`?t.lineOffsets:t.featureOffsets,t.kind===`lines`?void 0:t.polygonOffsets,t.kind===`lines`?void 0:t.ringOffsets,this.props.parameters]);let n=this.capStyle===`round`?2*this.quadSegments-1:this.capStyle===`square`?2:0,{props:r,id:o}=this;i(e,o,Object.fromEntries([t.positions,r.parameters,...t.kind===`lines`?[t.lineOffsets]:[t.featureOffsets,t.polygonOffsets,t.ringOffsets],...s(r.output)].map((e,t)=>[`binding.${t}`,e])));let c=(t.kind===`lines`?t.lineOffsets:t.ringOffsets).length-1,l=t.kind===`lines`?c:c*3,u=t.kind===`lines`?2*this.rowsPerVertex*t.positions.length+2*n*c:(1+2*this.rowsPerVertex)*t.positions.length;for(let[n,r]of Object.entries({inputCompaction:t.positions.length*8,rowFlagsAndScan:t.positions.length*4,boundaryOffsets:(l+1)*4,sidePositions:this.rowsPerVertex*t.positions.length*8,rawPositions:u*8})){let t=Math.max(4,r);_(e.device,o,n,t,`split the input or reduce quadSegments`),h(o,`${n} byteLength`,t,4,e.device.limits.maxBufferSize)}return a(e,o,{expandedVertexCount:u+l,inputRingCount:l,intersections:r.capacity.intersections,nodedSegments:r.capacity.nodedSegments,boundaryCapacity:r.output.boundary.endpoints.length,outputRingCapacity:r.output.geometry.ringOffsets.length-1,outputVertexCapacity:r.output.geometry.positions.length,leafCapacity:r.leafCapacity}),m(this.id,`boundary rows`,t.kind===`lines`?2*this.rowsPerVertex*t.positions.length+(2*n+1)*(t.lineOffsets.length-1):(1+2*this.rowsPerVertex)*t.positions.length+3*(t.ringOffsets.length-1),H(e.device)-this.props.capacity.intersections),t.kind===`lines`?this._getLineNodes(e,t):this._getPolygonNodes(e,t)}_getPolygonNodes(e,t){let{id:r,props:i,rowsPerVertex:a}=this,o=this._getCompactionNodes(e,`ring`,t.positions,t.ringOffsets,!0),s=t.ringOffsets.length-1,c=t.polygonOffsets.length-1,u=t.positions.length,d=(1+2*a)*u,p=(t,n,i)=>l(e,`${r}-${t}`,n,i),m=p(`normalized`,`float32x2`,u),h=p(`orientation-parameters`,`float32`,4),g=p(`stroke-left-parameters`,`float32`,4),_=p(`stroke-right-parameters`,`float32`,4),v=p(`stroke-left`,`float32x2`,u*a),y=p(`stroke-right`,`float32x2`,u*a),x=p(`raw-positions`,`float32x2`,d),S=p(`raw-ring-offsets`,`uint32`,3*s+1),C=p(`raw-polygon-offsets`,`uint32`,c+1),w=`const RING_COUNT: u32 = ${s}u;
const POLYGON_COUNT: u32 = ${c}u;
const ROWS_PER_VERTEX: u32 = ${a}u;

fn findPolygonOfRing(ring: u32) -> u32 {
  var low = 0u;
  var high = POLYGON_COUNT;
  while (low + 1u < high) {
    let middle = (low + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= ring) { low = middle; } else { high = middle; }
  }
  return low;
}`,T=i=>new n({id:`${r}-stroke-${i}-offset`,spatialContext:this.spatialContext,positions:m,featureOffsets:t.featureOffsets,polygonOffsets:t.polygonOffsets,ringOffsets:o.offsets,geometryType:`polygons`,joinStyle:this.joinStyle,quadSegments:this.quadSegments,insideTurns:`buffer`,parameters:i===`left`?g:_,output:{positions:i===`left`?v:y}}).getCommandNodes(e),E=[...o.nodes,f(e,{id:`${r}-polygon-parameters`,operation:z,variant:`polygon-parameters`,bindings:[{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`orientation`,view:h,type:`f32`,access:`read_write`},{name:`left`,view:g,type:`f32`,access:`read_write`},{name:`right`,view:_,type:`f32`,access:`read_write`}],invocationCount:4,body:`orientation[orientationOffset + index] = 0.0;
  let distance = abs(parameters[parametersOffset]);
  let value = parameters[parametersOffset + index];
  left[leftOffset + index] = select(value, distance, index == 0u);
  right[rightOffset + index] = select(value, -distance, index == 0u);`}),...new b({id:`${r}-orient`,spatialContext:this.spatialContext,positions:o.positions,ringOffsets:o.offsets,polygonOffsets:t.polygonOffsets,featureOffsets:t.featureOffsets,mode:`orient-polygons`,parameters:h,output:{positions:m}}).getCommandNodes(e),...T(`left`),...T(`right`),f(e,{id:`${r}-stroke-collapse`,operation:z,variant:`stroke-collapse`,bindings:[{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`ringOffsets`,view:o.offsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:t.polygonOffsets,type:`u32`,access:`read`},{name:`normalized`,view:m,type:`f32`,access:`read`},{name:`leftPositions`,view:v,type:`f32`,access:`read_write`},{name:`rightPositions`,view:y,type:`f32`,access:`read_write`}],invocationCount:Math.max(s,1),declarations:`${w}
fn readNormalized(row: u32) -> vec2f {
  return vec2f(normalized[normalizedOffset + 2u * row], normalized[normalizedOffset + 2u * row + 1u]);
}`,body:`if (index >= RING_COUNT) { return; }
  let begin = ringOffsets[ringOffsetsOffset + index];
  let end = ringOffsets[ringOffsetsOffset + index + 1u];
  if (end < begin + 3u) { return; }
  let origin = readNormalized(begin);
  var low = vec2f(0.0);
  var high = vec2f(0.0);
  for (var row = begin; row < end; row++) {
    let current = readNormalized(row) - origin;
    low = min(low, current);
    high = max(high, current);
  }
  let extent = high - low;
  if (min(extent.x, extent.y) >= 2.0 * abs(parameters[parametersOffset])) { return; }
  // The interior side is left of a counter-clockwise ring and right of a clockwise one. The
  // rings were wound by the exact GPUGeometryOrientation above (shells counter-clockwise, holes
  // clockwise), so the winding is the ring's role. A second, f32 shoelace sum read a sliver whose
  // sum rounds to zero as clockwise and collapsed its outer offset ring.
  var isShell = true;
  if (index >= polygonOffsets[polygonOffsetsOffset] &&
      index < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
    isShell = polygonOffsets[polygonOffsetsOffset + findPolygonOfRing(index)] == index;
  }
  let rowBegin = begin * ROWS_PER_VERTEX;
  let rowEnd = end * ROWS_PER_VERTEX;
  if (isShell) {
    let point = vec2f(leftPositions[leftPositionsOffset + 2u * rowBegin], leftPositions[leftPositionsOffset + 2u * rowBegin + 1u]);
    for (var row = rowBegin; row < rowEnd; row++) {
      leftPositions[leftPositionsOffset + 2u * row] = point.x;
      leftPositions[leftPositionsOffset + 2u * row + 1u] = point.y;
    }
  } else {
    let point = vec2f(rightPositions[rightPositionsOffset + 2u * rowBegin], rightPositions[rightPositionsOffset + 2u * rowBegin + 1u]);
    for (var row = rowBegin; row < rowEnd; row++) {
      rightPositions[rightPositionsOffset + 2u * row] = point.x;
      rightPositions[rightPositionsOffset + 2u * row + 1u] = point.y;
    }
  }`}),f(e,{id:`${r}-raw-layout`,operation:z,variant:`raw-layout`,bindings:[{name:`ringOffsets`,view:o.offsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:t.polygonOffsets,type:`u32`,access:`read`},{name:`rawRingOffsets`,view:S,type:`u32`,access:`read_write`},{name:`rawPolygonOffsets`,view:C,type:`u32`,access:`read_write`}],invocationCount:Math.max(3*s,c)+1,declarations:w,body:`if (index <= POLYGON_COUNT) {
    rawPolygonOffsets[rawPolygonOffsetsOffset + index] = 3u * polygonOffsets[polygonOffsetsOffset + index];
  }
  if (index > 3u * RING_COUNT) { return; }
  var start = 0u;
  if (index == 3u * RING_COUNT) {
    start = (1u + 2u * ROWS_PER_VERTEX) * ringOffsets[ringOffsetsOffset + RING_COUNT];
  } else {
    let polygon = findPolygonOfRing(index / 3u);
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let lastRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    let firstVertex = ringOffsets[ringOffsetsOffset + firstRing];
    let lastVertex = ringOffsets[ringOffsetsOffset + lastRing];
    let local = index - 3u * firstRing;
    let polygonStart = (1u + 2u * ROWS_PER_VERTEX) * firstVertex;
    if (local < lastRing - firstRing) {
      start = polygonStart + ringOffsets[ringOffsetsOffset + firstRing + local] - firstVertex;
    } else {
      let band = local - (lastRing - firstRing);
      let ring = firstRing + band / 2u;
      let ringBegin = ringOffsets[ringOffsetsOffset + ring];
      let ringLength = ringOffsets[ringOffsetsOffset + ring + 1u] - ringBegin;
      start = polygonStart + (lastVertex - firstVertex) +
        2u * ROWS_PER_VERTEX * (ringBegin - firstVertex) + (band % 2u) * ROWS_PER_VERTEX * ringLength;
    }
  }
  rawRingOffsets[rawRingOffsetsOffset + index] = start;`}),f(e,{id:`${r}-raw-pieces`,operation:z,variant:`raw-pieces`,bindings:[{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`ringOffsets`,view:o.offsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:t.polygonOffsets,type:`u32`,access:`read`},{name:`rawRingOffsets`,view:S,type:`u32`,access:`read`},{name:`normalized`,view:m,type:`f32`,access:`read`},{name:`leftPositions`,view:v,type:`f32`,access:`read`},{name:`rightPositions`,view:y,type:`f32`,access:`read`},{name:`output`,view:x,type:`f32`,access:`read_write`}],invocationCount:d,declarations:w,body:`var value = vec2f(0.0);
  if (index < rawRingOffsets[rawRingOffsetsOffset + 3u * RING_COUNT]) {
    var rawRing = 0u;
    var high = 3u * RING_COUNT;
    while (rawRing + 1u < high) {
      let middle = (rawRing + high) / 2u;
      if (rawRingOffsets[rawRingOffsetsOffset + middle] <= index) { rawRing = middle; } else { high = middle; }
    }
    let polygon = findPolygonOfRing(rawRing / 3u);
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let ringsInPolygon = polygonOffsets[polygonOffsetsOffset + polygon + 1u] - firstRing;
    let local = rawRing - 3u * firstRing;
    let row = index - rawRingOffsets[rawRingOffsetsOffset + rawRing];
    if (local < ringsInPolygon) {
      // Reversed normalized ring: winding -1 inside the polygon.
      let ring = firstRing + local;
      let source = ringOffsets[ringOffsetsOffset + ring + 1u] - 1u - row;
      value = vec2f(normalized[normalizedOffset + 2u * source], normalized[normalizedOffset + 2u * source + 1u]);
    } else {
      // Stroke rings: the +|d| offset forward and the -|d| offset backward (stroke winding -1 for
      // a positive distance); a negative distance reads both the other way (+1).
      let band = local - ringsInPolygon;
      let ring = firstRing + band / 2u;
      let rowBegin = ringOffsets[ringOffsetsOffset + ring] * ROWS_PER_VERTEX;
      let rowCount = (ringOffsets[ringOffsetsOffset + ring + 1u] - ringOffsets[ringOffsetsOffset + ring]) * ROWS_PER_VERTEX;
      let isLeft = (band % 2u) == 0u;
      let forward = isLeft == (parameters[parametersOffset] >= 0.0);
      let source = rowBegin + select(rowCount - 1u - row, row, forward);
      if (isLeft) {
        value = vec2f(leftPositions[leftPositionsOffset + 2u * source], leftPositions[leftPositionsOffset + 2u * source + 1u]);
      } else {
        value = vec2f(rightPositions[rightPositionsOffset + 2u * source], rightPositions[rightPositionsOffset + 2u * source + 1u]);
      }
    }
  }
  output[outputOffset + 2u * index] = value.x;
  output[outputOffset + 2u * index + 1u] = value.y;`})];return E.push(...this._getDissolveNodes(e,{kind:`polygons`,positions:x,featureOffsets:t.featureOffsets,polygonOffsets:C,ringOffsets:S,sourceIds:t.sourceIds},`negative`)),E}_getLineNodes(e,t){let{id:r,props:i}=this,a=this._getCompactionNodes(e,`line`,t.positions,t.lineOffsets,!1),o=t.lineOffsets.length-1,s=t.positions.length*this.rowsPerVertex,c=this.capStyle===`round`?2*this.quadSegments-1:this.capStyle===`square`?2:0,u=s*2+o*c*2,d=l(e,`${r}-positive`,`float32`,4),p=l(e,`${r}-negative`,`float32`,4),m=l(e,`${r}-left`,`float32x2`,s),h=l(e,`${r}-right`,`float32x2`,s),g=l(e,`${r}-raw-positions`,`float32x2`,u),_=l(e,`${r}-raw-offsets`,`uint32`,o+1),v=l(e,`${r}-identity-offsets`,`uint32`,o+1),y=[...a.nodes,f(e,{id:`${r}-line-parameters`,operation:z,variant:`line-parameters`,bindings:[{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`positive`,view:d,type:`f32`,access:`read_write`},{name:`negative`,view:p,type:`f32`,access:`read_write`}],invocationCount:4,body:`let distance = max(parameters[parametersOffset], 0.0);
  let value = parameters[parametersOffset + index];
  positive[positiveOffset + index] = select(value, distance, index == 0u);
  negative[negativeOffset + index] = select(value, -distance, index == 0u);`}),...new n({id:`${r}-left-offset`,spatialContext:this.spatialContext,positions:a.positions,lineOffsets:a.offsets,geometryType:`lines`,joinStyle:this.joinStyle,quadSegments:this.quadSegments,insideTurns:`buffer`,parameters:d,output:{positions:m}}).getCommandNodes(e),...new n({id:`${r}-right-offset`,spatialContext:this.spatialContext,positions:a.positions,lineOffsets:a.offsets,geometryType:`lines`,joinStyle:this.joinStyle,quadSegments:this.quadSegments,insideTurns:`buffer`,parameters:p,output:{positions:h}}).getCommandNodes(e),f(e,{id:`${r}-line-topology`,operation:z,variant:`line-topology`,bindings:[{name:`lineOffsets`,view:a.offsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:_,type:`u32`,access:`read_write`},{name:`identityOffsets`,view:v,type:`u32`,access:`read_write`}],invocationCount:o+1,declarations:`const ROWS_PER_VERTEX: u32 = ${this.rowsPerVertex}u;
const CAP_ROWS: u32 = ${c}u;`,body:`ringOffsets[ringOffsetsOffset + index] = 2u * ROWS_PER_VERTEX * lineOffsets[lineOffsetsOffset + index] + 2u * CAP_ROWS * index;
  identityOffsets[identityOffsetsOffset + index] = index;`}),f(e,{id:`${r}-line-boundaries`,operation:z,variant:`line-${this.capStyle}`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:a.offsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:_,type:`u32`,access:`read`},{name:`parameters`,view:d,type:`f32`,access:`read`},{name:`leftPositions`,view:m,type:`f32`,access:`read`},{name:`rightPositions`,view:h,type:`f32`,access:`read`},{name:`output`,view:g,type:`f32`,access:`read_write`}],invocationCount:u,declarations:`const LINE_COUNT: u32 = ${o}u;
const ROWS_PER_VERTEX: u32 = ${this.rowsPerVertex}u;
const CAP_ROWS: u32 = ${c}u;
const PI: f32 = 3.141592653589793;

fn point(view: ptr<storage, array<f32>, read>, base: u32, row: u32) -> vec2f {
  return vec2f((*view)[base + 2u * row], (*view)[base + 2u * row + 1u]);
}

fn unitDirection(a: vec2f, b: vec2f) -> vec2f {
  let delta = b - a;
  return delta / max(length(delta), 1e-20);
}`,body:`var line = 0u;
  var high = LINE_COUNT;
  while (line + 1u < high) {
    let middle = (line + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) { line = middle; } else { high = middle; }
  }
  let firstVertex = lineOffsets[lineOffsetsOffset + line];
  let lastVertex = lineOffsets[lineOffsetsOffset + line + 1u];
  let vertexCount = lastVertex - firstVertex;
  let sideCount = vertexCount * ROWS_PER_VERTEX;
  let local = index - ringOffsets[ringOffsetsOffset + line];
  if (vertexCount < 2u) {
    output[outputOffset + index * 2u] = 0.0;
    output[outputOffset + index * 2u + 1u] = 0.0;
    return;
  }
  var value = vec2f(0.0);
  if (local < sideCount) {
    value = point(&leftPositions, leftPositionsOffset, firstVertex * ROWS_PER_VERTEX + local);
  } else if (local < sideCount + CAP_ROWS) {
    let cap = local - sideCount;
    let center = point(&positions, positionsOffset, lastVertex - 1u);
    let direction = unitDirection(point(&positions, positionsOffset, lastVertex - 2u), center);
    let normal = vec2f(-direction.y, direction.x);
    ${this._getCapWGSL(`end`,`cap`,`center`,`direction`,`normal`)}
  } else if (local < 2u * sideCount + CAP_ROWS) {
    let rightRow = local - sideCount - CAP_ROWS;
    value = point(&rightPositions, rightPositionsOffset, (lastVertex * ROWS_PER_VERTEX - 1u) - rightRow);
  } else {
    let cap = local - 2u * sideCount - CAP_ROWS;
    let center = point(&positions, positionsOffset, firstVertex);
    let direction = unitDirection(center, point(&positions, positionsOffset, firstVertex + 1u));
    let normal = vec2f(-direction.y, direction.x);
    ${this._getCapWGSL(`start`,`cap`,`center`,`direction`,`normal`)}
  }
  output[outputOffset + index * 2u] = value.x;
  output[outputOffset + index * 2u + 1u] = value.y;`})];return y.push(...this._getDissolveNodes(e,{kind:`polygons`,positions:g,featureOffsets:v,polygonOffsets:v,ringOffsets:_,sourceIds:t.sourceIds},`negative`)),y}_getCompactionNodes(e,t,n,r,i){let{id:a}=this,o=`${a}-${t}-compact`,s=l(e,`${o}-keep`,`uint32`,n.length),c=l(e,`${o}-kept`,`uint32`,n.length),u=l(e,`${o}-positions`,`float32x2`,n.length),p=l(e,`${o}-offsets`,`uint32`,r.length),m=r.length-1;return{nodes:[f(e,{id:`${o}-flags`,operation:z,variant:`${t}-compact-flags`,bindings:[{name:`positions`,view:n,type:`f32`,access:`read`},{name:`offsets`,view:r,type:`u32`,access:`read`},{name:`keep`,view:s,type:`u32`,access:`read_write`}],invocationCount:n.length,declarations:`const LINE_COUNT: u32 = ${m}u;
const DROP_CLOSING: bool = ${i};

fn isSame(first: u32, second: u32) -> bool {
  return positions[positionsOffset + 2u * first] == positions[positionsOffset + 2u * second] &&
    positions[positionsOffset + 2u * first + 1u] == positions[positionsOffset + 2u * second + 1u];
}`,body:`var line = 0u;
  var high = LINE_COUNT;
  while (line + 1u < high) {
    let middle = (line + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) { line = middle; } else { high = middle; }
  }
  let begin = offsets[offsetsOffset + line];
  let end = offsets[offsetsOffset + line + 1u];
  var flag = index < end;
  if (flag && index > begin && isSame(index - 1u, index)) { flag = false; }
  if (flag && DROP_CLOSING && index + 1u == end && index > begin && isSame(begin, index)) { flag = false; }
  keep[keepOffset + index] = select(0u, 1u, flag);`}),...new d({id:`${o}-scan`,input:s,output:c,mode:`inclusive`}).getCommandNodes(e),f(e,{id:`${o}-offsets`,operation:z,variant:`${t}-compact-offsets`,bindings:[{name:`offsets`,view:r,type:`u32`,access:`read`},{name:`kept`,view:c,type:`u32`,access:`read`},{name:`output`,view:p,type:`u32`,access:`read_write`}],invocationCount:r.length,body:`let begin = offsets[offsetsOffset + index];
  output[outputOffset + index] = select(kept[keptOffset + begin - 1u], 0u, begin == 0u);`}),f(e,{id:`${o}-scatter`,operation:z,variant:`${t}-compact-scatter`,bindings:[{name:`positions`,view:n,type:`f32`,access:`read`},{name:`keep`,view:s,type:`u32`,access:`read`},{name:`kept`,view:c,type:`u32`,access:`read`},{name:`output`,view:u,type:`f32`,access:`read_write`}],invocationCount:n.length,body:`if (keep[keepOffset + index] == 0u) { return; }
  let row = kept[keptOffset + index] - 1u;
  output[outputOffset + 2u * row] = positions[positionsOffset + 2u * index];
  output[outputOffset + 2u * row + 1u] = positions[positionsOffset + 2u * index + 1u];`})],positions:u,offsets:p}}_getCapWGSL(e,t,n,r,i){if(this.capStyle===`round`){let r=e===`end`?i:`-${i}`;return`let angle = -PI * f32(${t} + 1u) / f32(CAP_ROWS + 1u);
    let c = cos(angle); let s = sin(angle);
    let radial = vec2f(c * (${r}).x - s * (${r}).y, s * (${r}).x + c * (${r}).y);
    value = ${n} + radial * parameters[parametersOffset];`}return this.capStyle===`square`?`value = ${n} + (${e===`end`?r:`-${r}`} + ${e===`end`?`select(${i}, -${i}, ${t} == 1u)`:`select(-${i}, ${i}, ${t} == 1u)`}) * parameters[parametersOffset];`:`value = ${n};`}_getDissolveNodes(e,t,n){let{props:r,id:i}=this;return new c({id:`${i}-repair`,spatialContext:this.spatialContext,left:t,operation:`dissolve`,fillRule:n,capacity:r.capacity,vertexTolerance:r.vertexTolerance,output:r.output,spatialSort:r.spatialSort,leafCapacity:r.leafCapacity}).getCommandNodes(e)}},W=[`round`,`mitre`,`bevel`],G=w({parameters:L,caption:`Buffer polygons of a given distance around lines or polygons (input muted).`,createInputs:({random:e,world:t})=>({world:t,lines:A(t,2,16),blobs:N(e,t,{rows:1,columns:3,vertexCount:7,irregularity:.5,fill:.55})}),build(t,n,r){let i=r.geometry===0,a=I(t,`buffer`,{vertices:8192,rings:256,boundary:4096}),o=t.allocate(`buffer-parameters`,`float32`,4),s=t.upload(`line-positions`,`float32x2`,n.lines.positions),c=t.upload(`line-offsets`,`uint32`,n.lines.lineOffsets),l=P(t,`blobs`,n.blobs);t.graph.add(new U({geometry:i?{kind:`lines`,positions:s.view,lineOffsets:c.view}:M(l),parameters:o.view,joinStyle:W[r.joinStyle],capStyle:`round`,quadSegments:8,vertexTolerance:1e-4,capacity:{intersections:4096,nodedSegments:16384},output:a.output}));let u=F(t,`buffer`,a.polygons,a.count);return{outputs:{input:t.upload(`input`,`float32x4`,i?k(n.lines.positions,n.lines.lineOffsets,!1):j(n.blobs)),...u},update:t=>o.write(e({distance:t.distance}))}},rebuildOn:[`joinStyle`,`geometry`],interaction:{kind:`reach`,hint:`move off the line`,rest:({inputs:e,toUnit:t})=>{let n=E(L,`distance`).default;return t(O(e.lines,n,(t,n)=>D(e.lines,t,n)?.distance??0,e.world))},resolve(e,{inputs:t,values:n}){let r=n.geometry===0,i=D(r?t.lines:{positions:t.blobs.positions,lineOffsets:t.blobs.ringOffsets},e.x,e.y,!r);if(!i)return{};let a=T(E(L,`distance`),i.distance);return{values:{distance:a},glyphs:[{shape:`ring`,at:i.point,radius:a,dashed:!0},{shape:`line`,from:i.point,to:[e.x,e.y]},{shape:`dot`,at:i.point}],readout:`distance ${a.toFixed(3)} · ${W[n.joinStyle]} joins`}}},draw:({outputs:e})=>[{layer:`segments`,nodes:e.points,edges:e.edges,values:e.lineIds,radius:3},{layer:`lines`,context:!0,segments:e.input,radius:2.5,alpha:1}]});export{G as default};
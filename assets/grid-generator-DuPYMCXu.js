import{r as e}from"./kernel-C3tfQOn4.js";import{E as t,f as n,v as r,y as i}from"./gpu-contributor-utils-BIwLQON5.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{v as o}from"./spatial-context-DwZEd4EZ.js";import{t as s}from"./exact-orientation-wgsl-B_jI-HEc.js";import{c,i as l,t as u}from"./wgsl-common-DfyXiYB0.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{s as f}from"./point-observations-CeqP-hIo.js";import{t as p}from"./define-run-B-brkCe4.js";import{r as m}from"./lines-kx5qHdti.js";import{es as h}from"./entry-preview-D70TdSbK.js";var g=`GPUGridGenerator`;function _(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Grid generator target must hold 4 elements`);let{minX:n,minY:r,cellWidth:i,cellHeight:a}=e;for(let e of[n,r,i,a])if(!Number.isFinite(Math.fround(e)))throw Error(`Grid generator settings must be finite`);if(Math.fround(i)<=0||Math.fround(a)<=0)throw Error(`Grid generator cellWidth and cellHeight must be positive`);return t.set([n,r,i,a]),t}function v(e){switch(e){case`square`:return 4;case`hex`:return 6;case`triangle`:return 3;case`point`:return 0;default:throw Error(`Unknown grid type ${String(e)}`)}}function y(e,t,n){return t*n*(e===`triangle`?2:1)}function b(e,t,n){return e===`square`?(t+1)*(n+1):2*(t*n+t+n)}var x=class{id;props;spatialContext;cellCount;verticesPerCell;cornerCount;constructor(e){this.id=e.id??`grid-generator`,this.props=e;let{id:r}=this;if(this.spatialContext=o(r,e.spatialContext),this.verticesPerCell=v(e.gridType),n(r,`columns`,e.columns,1,2**53-1),n(r,`rows`,e.rows,1,2**53-1),this.cellCount=y(e.gridType,e.columns,e.rows),this.cornerCount=e.gridType===`square`||e.gridType===`hex`?b(e.gridType,e.columns,e.rows):0,this.cellCount*Math.max(this.verticesPerCell,1)>2147483647)throw Error(`${r} grid is too large`);if(t(e.parameters,[`float32`],`${r} parameters`),e.parameters.length<4)throw Error(`${r} parameters must hold 4 float32 values`);let{positions:a,centers:s,corners:c,cornerIntersects:l}=e.output;if(c&&(t(c,[`float32x2`],`${r} output.corners`),!this.cornerCount||this.cornerCount>2147483647||c.length!==this.cornerCount))throw Error(`${r} output.corners needs a correctly sized square or hex grid`);if(l&&(t(l,[`uint32`],`${r} output.cornerIntersects`),!c||l.length!==this.cornerCount))throw Error(`${r} output.cornerIntersects needs one row per output corner`);if(e.gridType===`point`){if(!s)throw Error(`${r} point grids need output.centers`);if(a)throw Error(`${r} point grids have no output.positions`)}else if(!a&&!c)throw Error(`${r} ${e.gridType} grids need output.positions`);if(a&&(t(a,[`float32x2`],`${r} output.positions`),a.length!==this.cellCount*this.verticesPerCell))throw Error(`${r} output.positions must hold ${this.cellCount*this.verticesPerCell} rows`);if(s&&(t(s,[`float32x2`],`${r} output.centers`),s.length!==this.cellCount))throw Error(`${r} output.centers must hold ${this.cellCount} rows`);let{intersects:u}=e.output;if(e.hexOrientation!==void 0&&e.hexOrientation!==`pointy`&&e.hexOrientation!==`flat`)throw Error(`${r} hexOrientation must be 'pointy' or 'flat'`);if(e.hexOrientation===`flat`&&e.gridType!==`hex`)throw Error(`${r} hexOrientation 'flat' needs gridType 'hex'`);let{extent:d}=e;if(!!(u||l)!=!!d)throw Error(`${r} extent needs an intersection output and vice versa`);if(d){if(u&&t(u,[`uint32`],`${r} output.intersects`),t(d.positions,[`float32x2`],`${r} extent.positions`),t(d.featureOffsets,[`uint32`],`${r} extent.featureOffsets`),t(d.polygonOffsets,[`uint32`],`${r} extent.polygonOffsets`),t(d.ringOffsets,[`uint32`],`${r} extent.ringOffsets`),u&&u.length!==this.cellCount)throw Error(`${r} output.intersects must hold ${this.cellCount} rows`);if(d.ringOffsets.length<2)throw Error(`${r} extent.ringOffsets needs at least one ring`);if(u&&(e.gridType===`point`?!s:!a))throw Error(`${r} output.intersects needs the cell geometry output`)}i(r,[a,s,c,u,l],[e.parameters,d?.positions,d?.featureOffsets,d?.polygonOffsets,d?.ringOffsets])}getCommandNodes(t){let{props:n,id:i}=this,{positions:o,centers:f,corners:p,cornerIntersects:m}=n.output,{intersects:h}=n.output,{extent:_}=n;d(t,i,{parameters:n.parameters,positions:o,centers:f,corners:p,cornerIntersects:m,intersects:h,extentPositions:h||m?_?.positions:void 0,extentRingOffsets:h||m?_?.ringOffsets:void 0}),_&&(h||m)&&r(t.device,i,`extent bounds scratch`,16,`use a device supporting four storage words`);let v=n.gridType===`hex`&&n.hexOrientation===`flat`,y=[{name:`parameters`,view:n.parameters,type:`f32`,access:`read`}];o&&y.push({name:`positionsOut`,view:o,type:`f32`,access:`read_write`}),f&&y.push({name:`centersOut`,view:f,type:`f32`,access:`read_write`});let b=[];(o||f)&&b.push(a(t,{id:`${i}-generate`,operation:g,variant:v?`hex-flat`:n.gridType,bindings:y,invocationCount:this.cellCount,declarations:`const COLUMNS: u32 = ${n.columns}u;
const SQRT3: f32 = 1.7320508;
const CELLS_PER_ROW: u32 = ${n.gridType===`triangle`?2*n.columns:n.columns}u;
${n.gridType===`hex`&&o?C(v):``}

${o?`fn writeVertex(cell: u32, slot: u32, p: vec2<f32>) {
  let base = positionsOutOffset + 2u * (cell * `+this.verticesPerCell+`u + slot);
  positionsOut[base] = p.x;
  positionsOut[base + 1u] = p.y;
}`:``}`,body:`let origin = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let w = parameters[parametersOffset + 2u];
  let h = parameters[parametersOffset + 3u];
  let row = index / CELLS_PER_ROW;
  let column = index % CELLS_PER_ROW;
  let shift = select(0.0, 0.5 * w, (row & 1u) == 1u);
  var center = vec2<f32>(0.0);
${T(n.gridType,!!o,v)}
  ${f?`centersOut[centersOutOffset + 2u * index] = center.x;
  centersOut[centersOutOffset + 2u * index + 1u] = center.y;`:``}`})),p&&b.push(a(t,{id:`${i}-corners`,operation:g,variant:v?`corners-hex-flat`:`corners-${n.gridType}`,bindings:[{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`cornersOut`,view:p,type:`f32`,access:`read_write`}],invocationCount:this.cornerCount,declarations:`const COLUMNS: u32 = ${v?n.rows:n.columns}u;
${n.gridType===`square`?``:C(v)}`,body:`let origin = vec2f(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let width = parameters[parametersOffset + 2u];
  let height = parameters[parametersOffset + 3u];
  ${S(n.gridType===`square`,v)}
  cornersOut[cornersOutOffset + 2u * index] = corner.x;
  cornersOut[cornersOutOffset + 2u * index + 1u] = corner.y;`}));let x;_&&(h||m)&&(x=e(t,`${i}-extent-bounds`,`uint32`,4),b.push(a(t,{id:`${i}-extent-bounds-initialize`,operation:g,variant:`extent-bounds-initialize`,bindings:[{name:`bounds`,view:x,type:`u32`,access:`read_write`}],invocationCount:1,body:`bounds[boundsOffset] = 0xffffffffu; bounds[boundsOffset + 1u] = 0xffffffffu; bounds[boundsOffset + 2u] = 0u; bounds[boundsOffset + 3u] = 0u;`}),a(t,{id:`${i}-extent-bounds`,operation:g,variant:`extent-bounds`,bindings:[{name:`extentPositions`,view:_.positions,type:`f32`,access:`read`},{name:`bounds`,view:x,type:`atomic<u32>`,access:`read_write`}],invocationCount:_.positions.length,declarations:`${c}
${l}`,body:`let point = vec2f(extentPositions[extentPositionsOffset + 2u * index], extentPositions[extentPositionsOffset + 2u * index + 1u]);
  if (isFiniteBits(point.x) && isFiniteBits(point.y)) {
    atomicMin(&bounds[boundsOffset], getOrderedKey(point.x));
    atomicMin(&bounds[boundsOffset + 1u], getOrderedKey(point.y));
    atomicMax(&bounds[boundsOffset + 2u], getOrderedKey(point.x));
    atomicMax(&bounds[boundsOffset + 3u], getOrderedKey(point.y));
  }`})));for(let[e,r,c,l,d]of[[h,n.gridType===`point`?f:o,this.cellCount,n.gridType===`point`,`intersects`],[m,p,this.cornerCount,!0,`corner-intersects`]])e&&_&&b.push(a(t,{id:`${i}-${d}`,operation:g,variant:`intersects`,bindings:[{name:`cellVertices`,view:r,type:`f32`,access:`read`},{name:`extentPositions`,view:_.positions,type:`f32`,access:`read`},{name:`extentRingOffsets`,view:_.ringOffsets,type:`u32`,access:`read`},{name:`intersectsOut`,view:e,type:`u32`,access:`read_write`},{name:`extentBounds`,view:x,type:`u32`,access:`read`}],invocationCount:c,declarations:`const VERTICES_PER_CELL: u32 = ${l?1:this.verticesPerCell}u;
const IS_POINT: bool = ${l};
const RING_COUNT: u32 = ${_.ringOffsets.length-1}u;
const EXTENT_VERTEX_COUNT: u32 = ${_.positions.length}u;
${s}
${u}
${w}`,body:`intersectsOut[intersectsOutOffset + index] = select(0u, 1u, cellIntersectsExtent(index));`}));return b}};function S(e,t){return e?`let corner = origin + vec2f(f32(index % (COLUMNS + 1u)) * width,
    f32(index / (COLUMNS + 1u)) * height);`:`var row = 0u;
  var column = 0u;
  var slot = index;
  let firstRowCount = 4u * COLUMNS + 2u;
  if (index < firstRowCount) {
    if (index >= 6u) {
      column = 1u + (index - 6u) / 4u;
      slot = (index - 6u) % 4u;
      slot = select(slot, slot + 2u, slot >= 2u);
    }
  } else {
    let rowCount = 2u * COLUMNS + 2u;
    row = 1u + (index - firstRowCount) / rowCount;
    let local = (index - firstRowCount) % rowCount;
    let firstCellCount = select(4u, 3u, (row & 1u) == 1u);
    slot = local;
    if (local >= firstCellCount) {
      column = 1u + (local - firstCellCount) / 2u;
      slot = (local - firstCellCount) % 2u;
    }
    // The right boundary of an odd row adds the bottom-right vertex of its final cell.
    if ((row & 1u) == 1u && local == rowCount - 1u) {
      column = COLUMNS - 1u;
      slot = 5u;
    }
  }
  let corner = hexLatticeVertex(origin, width, column, row, slot);`}function C(e){return`fn hexLatticeVertex(origin: vec2f, width: f32, latticeColumn: u32, latticeRow: u32, slot: u32) -> vec2f {
  let horizontal = array<i32, 6>(1, 0, -1, -1, 0, 1);
  let vertical = array<i32, 6>(1, 2, 1, -1, -2, -1);
  let lattice = vec2f(f32(2u * latticeColumn + (latticeRow & 1u)) + f32(horizontal[slot]) + 1.0,
    f32(3u * latticeRow) + f32(vertical[slot]) + 2.0);
  let displacement = lattice * vec2f(0.5 * width, 0.5 * width / 1.7320508);
  return origin + displacement${e?`.yx`:``};
}`}var w=`
fn cellVertex(cell: u32, slot: u32) -> vec2f {
  let base = cellVerticesOffset + 2u * (cell * VERTICES_PER_CELL + slot);
  return vec2f(cellVertices[base], cellVertices[base + 1u]);
}
fn extentVertex(vertex: u32) -> vec2f {
  let base = extentPositionsOffset + 2u * vertex;
  return vec2f(extentPositions[base], extentPositions[base + 1u]);
}
fn ringStart(ring: u32) -> u32 { return extentRingOffsets[extentRingOffsetsOffset + ring]; }
fn ringEnd(ring: u32) -> u32 {
  return min(extentRingOffsets[extentRingOffsetsOffset + ring + 1u], EXTENT_VERTEX_COUNT);
}
// Second vertex of the edge leaving 'vertex' in its ring (the ring closes implicitly).
fn nextVertex(vertex: u32, first: u32, last: u32) -> u32 {
  return select(vertex + 1u, first, vertex + 1u >= last);
}
fn insideBox(a: vec2f, b: vec2f, c: vec2f) -> bool {
  return c.x >= min(a.x, b.x) && c.x <= max(a.x, b.x) && c.y >= min(a.y, b.y) && c.y <= max(a.y, b.y);
}
// True when the closed segments p1p2 and q1q2 share a point.
fn segmentsTouch(p1: vec2f, p2: vec2f, q1: vec2f, q2: vec2f) -> bool {
  let o1 = orientSign(p1, p2, q1);
  let o2 = orientSign(p1, p2, q2);
  let o3 = orientSign(q1, q2, p1);
  let o4 = orientSign(q1, q2, p2);
  if (o1 == 2 || o2 == 2 || o3 == 2 || o4 == 2) { return false; }
  if (o1 != o2 && o3 != o4) { return true; }
  if (o1 == 0 && insideBox(p1, p2, q1)) { return true; }
  if (o2 == 0 && insideBox(p1, p2, q2)) { return true; }
  if (o3 == 0 && insideBox(q1, q2, p1)) { return true; }
  if (o4 == 0 && insideBox(q1, q2, p2)) { return true; }
  return false;
}
// Even-odd containment of 'point' in the extent rings; boundary counts as inside.
fn extentContains(point: vec2f) -> bool {
  var isInside = false;
  for (var ring = 0u; ring < RING_COUNT; ring++) {
    let first = ringStart(ring);
    let last = ringEnd(ring);
    for (var vertex = first; vertex < last; vertex++) {
      let a = extentVertex(vertex);
      let b = extentVertex(nextVertex(vertex, first, last));
      let side = orientSign(a, b, point);
      if (side == 2) { continue; }
      if (side == 0 && insideBox(a, b, point)) { return true; }
      if (a.y <= point.y && b.y > point.y && side > 0) { isInside = !isInside; }
      if (b.y <= point.y && a.y > point.y && side < 0) { isInside = !isInside; }
    }
  }
  return isInside;
}
// Convex counter-clockwise cell contains 'point' (boundary included).
fn cellContains(cell: u32, point: vec2f) -> bool {
  for (var slot = 0u; slot < VERTICES_PER_CELL; slot++) {
    let side = orientSign(cellVertex(cell, slot), cellVertex(cell, (slot + 1u) % VERTICES_PER_CELL), point);
    if (side < 0 || side == 2) { return false; }
  }
  return true;
}
// The extent bounding box rejects most cells of a view-filling grid before any edge is read. A
// bounds word that was never written decodes to NaN, which fails every comparison and rejects
// nothing.
fn missesExtentBounds(low: vec2f, high: vec2f) -> bool {
  let extentLow = vec2f(decodeOrderedKey(extentBounds[extentBoundsOffset]), decodeOrderedKey(extentBounds[extentBoundsOffset + 1u]));
  let extentHigh = vec2f(decodeOrderedKey(extentBounds[extentBoundsOffset + 2u]), decodeOrderedKey(extentBounds[extentBoundsOffset + 3u]));
  return high.x < extentLow.x || low.x > extentHigh.x || high.y < extentLow.y || low.y > extentHigh.y;
}
fn cellIntersectsExtent(cell: u32) -> bool {
  if (IS_POINT) {
    let point = cellVertex(cell, 0u);
    if (missesExtentBounds(point, point)) { return false; }
    return extentContains(point);
  }
  var low = cellVertex(cell, 0u);
  var high = low;
  for (var slot = 1u; slot < VERTICES_PER_CELL; slot++) {
    let corner = cellVertex(cell, slot);
    low = min(low, corner);
    high = max(high, corner);
  }
  if (missesExtentBounds(low, high)) { return false; }
  // Contact test: an extent edge whose box misses the cell box cannot touch the cell, so only the
  // few edges near the cell pay for exact predicates.
  for (var ring = 0u; ring < RING_COUNT; ring++) {
    let first = ringStart(ring);
    let last = ringEnd(ring);
    for (var vertex = first; vertex < last; vertex++) {
      let q1 = extentVertex(vertex);
      let q2 = extentVertex(nextVertex(vertex, first, last));
      if (max(q1.x, q2.x) < low.x || min(q1.x, q2.x) > high.x ||
          max(q1.y, q2.y) < low.y || min(q1.y, q2.y) > high.y) {
        continue;
      }
      for (var slot = 0u; slot < VERTICES_PER_CELL; slot++) {
        if (segmentsTouch(cellVertex(cell, slot), cellVertex(cell, (slot + 1u) % VERTICES_PER_CELL), q1, q2)) {
          return true;
        }
      }
    }
  }
  // No contact, so the cell outline lies wholly inside or wholly outside the extent (it never
  // crosses the boundary): one corner decides, instead of one containment scan per corner.
  if (extentContains(cellVertex(cell, 0u))) { return true; }
  // No contact: a ring is either wholly inside the cell or wholly outside it.
  for (var ring = 0u; ring < RING_COUNT; ring++) {
    if (ringEnd(ring) > ringStart(ring) && cellContains(cell, extentVertex(ringStart(ring)))) {
      return true;
    }
  }
  return false;
}`;function T(e,t,n){let r=e=>t?e:``;switch(e){case`square`:case`point`:{let t=`let lower = origin + vec2<f32>(f32(column) * w, f32(row) * h);
  center = lower + 0.5 * vec2<f32>(w, h);`;return e===`point`?t:`${t}
  ${r(`// Every vertex from the lattice expression (as the corner output), so neighbours share bits.
  let upper = origin + vec2<f32>(f32(column + 1u) * w, f32(row + 1u) * h);
  writeVertex(index, 0u, lower);
  writeVertex(index, 1u, vec2<f32>(upper.x, lower.y));
  writeVertex(index, 2u, upper);
  writeVertex(index, 3u, vec2<f32>(lower.x, upper.y));`)}`}case`hex`:return n?`let circumradius = w / SQRT3;
  center = origin + vec2<f32>(circumradius + f32(column) * 1.5 * circumradius,
    0.5 * w + f32(row) * w + select(0.0, 0.5 * w, (column & 1u) == 1u));
  ${r(`for (var corner = 0u; corner < 6u; corner++) {
    // Flat top: first corner at angle 0, counter-clockwise. Pointy-frame slot (7 - corner) % 6 is
    // that corner after the axis swap.
    writeVertex(index, corner, hexLatticeVertex(origin, w, row, column, (7u - corner) % 6u));
  }`)}`:`let circumradius = w / SQRT3;
  center = origin + vec2<f32>(0.5 * w + f32(column) * w + shift, circumradius + f32(row) * 1.5 * circumradius);
  ${r(`for (var corner = 0u; corner < 6u; corner++) {
    // Pointy top: first corner at 30 degrees, counter-clockwise.
    writeVertex(index, corner, hexLatticeVertex(origin, w, column, row, corner));
  }`)}`;case`triangle`:return`let pairIndex = column / 2u;
  let half = 0.5 * w;
  let step = 2u * pairIndex + (row & 1u);
  let y0 = origin.y + f32(row) * h;
  let y1 = origin.y + f32(row + 1u) * h;
  var a: vec2<f32>;
  var b: vec2<f32>;
  var c: vec2<f32>;
  if ((column & 1u) == 0u) {
    a = vec2<f32>(origin.x + f32(step) * half, y0);
    b = vec2<f32>(origin.x + f32(step + 2u) * half, y0);
    c = vec2<f32>(origin.x + f32(step + 1u) * half, y1);
  } else {
    a = vec2<f32>(origin.x + f32(step + 2u) * half, y0);
    b = vec2<f32>(origin.x + f32(step + 3u) * half, y1);
    c = vec2<f32>(origin.x + f32(step + 1u) * half, y1);
  }
  center = (a + b + c) / 3.0;
  ${r(`writeVertex(index, 0u, a);
  writeVertex(index, 1u, b);
  writeVertex(index, 2u, c);`)}`;default:throw Error(`Unknown grid type ${String(e)}`)}}var E=[`square`,`hex`,`triangle`],D=20,O=12,k=p({parameters:h,caption:`Sampling grid over Carolina rain gauges. Cell spacing changes while station locations stay fixed.`,createInputs:({random:e,world:t})=>{let{positions:n,rain:r}=f(e,t,180,!0),i=n.filter((e,t)=>t%2==0),a=n.filter((e,t)=>t%2==1);return{positions:n,rain:r,minX:Math.min(...i),minY:Math.min(...a),shift:[0,0]}},build(e,t,n){let r=E[n.gridType],i=y(r,D,O),a=v(r),o=e.allocate(`cell-positions`,`float32x2`,i*a),s=e.allocate(`cell-centers`,`float32x2`,i),c=e.allocate(`grid-parameters`,`float32`,4);e.graph.add(new x({gridType:r,columns:D,rows:O,hexOrientation:r===`hex`?n.hexOrientation===0?`pointy`:`flat`:void 0,parameters:c.view,output:{positions:o.view,centers:s.view}}));let l=e.upload(`ring-offsets`,`uint32`,Uint32Array.from({length:i+1},(e,t)=>t*a));return{outputs:{centers:s,...m(e,`cells`,o,l,{closed:!0}),stations:e.upload(`stations`,`float32x2`,t.positions),rainfall:e.upload(`rainfall-mm`,`float32`,t.rain)},update:e=>{let n=r===`hex`;c.write(_({minX:t.minX+t.shift[0],minY:t.minY+t.shift[1],cellWidth:e.cellWidth,cellHeight:n?e.cellWidth*.8660254:e.cellWidth}))}}},rebuildOn:[`gridType`,`hexOrientation`],interaction:{kind:`drag`,hint:`slide the grid origin`,rest:({inputs:e,toUnit:t})=>t([e.minX,e.minY]),resolve(e,{inputs:t,values:n}){let r=n.cellWidth,i=n.gridType===1?r*.8660254:r,a=(e,t)=>(e%t+t)%t;t.shift=[a(e.x-t.minX,r),a(e.y-t.minY,i)];let o=[t.minX+t.shift[0],t.minY+t.shift[1]];return{rerun:!0,glyphs:[{shape:`box`,minimum:o,maximum:[o[0]+r,o[1]+i]},{shape:`crosshair`,at:o}],readout:`origin ${o[0].toFixed(2)}, ${o[1].toFixed(2)} · cellWidth ${r.toFixed(3)}`}}},draw:({outputs:e})=>[{layer:`segments`,nodes:e.points,edges:e.edges,color:[.4,.53,.65,.65],radius:1.4},{layer:`dots`,positions:e.centers,color:[.5,.6,.7,.65],radius:2},{layer:`dots`,positions:e.stations,values:e.rainfall,context:!0,ramp:`sequential`,radius:5.5}]});export{k as default};
import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{s as r}from"./cell-index-wgsl-Cw7syfdA.js";import{n as i,o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,f as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{a as u,i as d,n as f,r as p}from"./cell-dispatch-validation-PIkQyAK_.js";import{n as m,r as h}from"./gpu-cell-geometry-CDEk1ld7.js";import{t as g}from"./s2-neighbor-wgsl-ByEyc4ST.js";import{t as _}from"./cell-keys-Dl1tIQ_Z.js";import{t as v}from"./h3-neighbor-wgsl-DdYw-iG5.js";import{i as y}from"./cell-topology-wgsl-gkK5LfSP.js";import{r as b}from"./spatial-context-DwZEd4EZ.js";import{t as x}from"./gpu-segment-ring-assembly-VExWprVe.js";var S=`GPUCellSetOutline`,C=4294967295,w={h3:10,quadbin:4,s2:4},T=class{id;props;edgeSlots;edgeModel;outputSpatialContext;interiorSide;ringAssembly;constructor(e){this.id=e.id??`cell-set-outline`,this.props=e;let n=this.id;if(![`h3`,`quadbin`,`s2`].includes(e.family))throw Error(`${n} family must be 'h3', 'quadbin' or 's2'`);this.edgeSlots=w[e.family],this.edgeModel=e.family===`quadbin`?`linear`:`great-circle`,this.outputSpatialContext={...b,edgeModel:this.edgeModel},this.interiorSide=e.family===`h3`?`left`:`right`;let{output:r}=e;o(e.cells,[`uint32x2`],`${n} cells`);let i=e.cells.length;if(i<1)throw Error(`${n} needs at least one cell row`);s(n,`cells.length times edgeSlots times endpoint components`,i*this.edgeSlots*4,1,C);for(let[i,a]of[[`count`,e.count],[`output.status.count`,r.status.count],[`output.status.overflow`,r.status.overflow],[`output.status.requiredCount`,r.status.requiredCount]])if(a&&(t(a,`${n} ${i}`),a.length<1))throw Error(`${n} ${i} must contain one uint32 row`);if(e.groups&&(t(e.groups,`${n} groups`),e.groups.length!==i))throw Error(`${n} groups length must equal the cell row count`);t(r.rows,`${n} output.rows`);let a=r.rows.length;if(a<1)throw Error(`${n} output capacity must be at least 1`);o(r.cells,[`uint32x2`],`${n} output.cells`),t(r.edgeIndices,`${n} output.edgeIndices`),o(r.endpoints,[`float32x4`],`${n} output.endpoints`),s(n,`output capacity times endpoint components`,a*4,1,C);for(let[e,t]of[[`cells`,r.cells.length],[`edgeIndices`,r.edgeIndices.length],[`endpoints`,r.endpoints.length]])if(t!==a)throw Error(`${n} output.${e} must have the same length as output.rows`);if(r.groups){if(!e.groups)throw Error(`${n} output.groups needs the groups input`);if(t(r.groups,`${n} output.groups`),r.groups.length!==a)throw Error(`${n} output.groups must have the same length as output.rows`)}d(n,{...f(r),...f(e.rings?.output??{},`rings`)},[{rows:r.rows,cells:r.cells,edgeIndices:r.edgeIndices,endpoints:r.endpoints},{...r.status}]),u(n,{rows:r.rows},{groups:r.groups}),l(n,E(r),[e.cells,e.count,e.groups]),this.ringAssembly=e.rings?new x({id:`${n}-rings`,endpoints:r.endpoints,count:r.status.count,groups:r.groups,vertexTolerance:e.rings.vertexTolerance??5e-5,normalizeWinding:e.rings.normalizeWinding,interiorSide:this.interiorSide,spatialContext:this.outputSpatialContext,output:e.rings.output}):void 0}getCommandNodes(t){let{id:o,props:s,edgeSlots:l}=this,{output:u,family:d}=s,b=[[`candidate endpoint scratch`,s.cells.length*l*16],[`outline scan scratch`,s.cells.length*4],[`outline total scratch`,4]];if(s.rings){let e=s.rings.output,t=e.ringOffsets.length-1;if(b.push([`ring segment sort scratch`,2**Math.ceil(Math.log2(u.rows.length))*4],[`ring counter scratch`,24],[`ring statistics scratch`,t*4]),e.ringShells||e.polygons){let e=2**Math.ceil(Math.log2(t));b.push([`ring shell records`,t*12],[`ring bounds`,t*16],[`ring shell BVH`,(2*e-1)*8])}}p(t,o,{cells:s.cells,count:s.count,groups:s.groups,...f(u,`output`),...f(s.rings?.output??{},`rings`)},b),c(o,t,[s.cells,s.count,s.groups,...E(u)]);let x=s.cells.length,C=x*l,w=u.rows.length,T=e(t,`${o}-masks`,`uint32`,x),j=e(t,`${o}-flags`,`uint32`,x),M=e(t,`${o}-accepted`,`uint32`,x),N=e(t,`${o}-candidate-endpoints`,`float32x4`,C),P=e(t,`${o}-total`,`uint32`,1),F=[{name:`cells`,view:s.cells,type:`u32`,access:`read`}];s.count&&F.push({name:`rowCount`,view:s.count,type:`u32`,access:`read`}),s.groups&&F.push({name:`groups`,view:s.groups,type:`u32`,access:`read`}),F.push({name:`masks`,view:T,type:`u32`,access:`read_write`},{name:`flags`,view:j,type:`u32`,access:`read_write`},{name:`candidateEndpoints`,view:N,type:`f32`,access:`read_write`});let I=d===`h3`,L=a(t,{id:`${o}-classify`,operation:S,variant:d,bindings:F,invocationCount:x,declarations:`const ROWS: u32 = ${x}u;
const EDGE_SLOTS: u32 = ${l}u;
const CELL_TOPOLOGY_K: u32 = 0u;
const CELL_TOPOLOGY_CAPACITY: u32 = 1u;
${r.source}
${_}
${I?v:``}
${I?h:``}
${y}
${D.replace(`GROUP_TEST`,s.groups?`return groups[groupsOffset + neighborRow] == groups[groupsOffset + row];`:`return true;`)}
${I?O:d===`s2`?`${m}\n${g}\n${A}`:k}`,body:`let validRows = ${s.count?`min(rowCount[rowCountOffset], ROWS)`:`ROWS`};
  masks[masksOffset + index] = 0u;
  if (index < validRows) {
    outlineClassifyRow(index, validRows);
  }
  flags[flagsOffset + index] = countOneBits(masks[masksOffset + index]);`}),R=[{name:`accepted`,view:M,type:`u32`,access:`read`},{name:`masks`,view:T,type:`u32`,access:`read`},{name:`candidateEndpoints`,view:N,type:`f32`,access:`read`},{name:`cells`,view:s.cells,type:`u32`,access:`read`},{name:`outRows`,view:u.rows,type:`u32`,access:`read_write`},{name:`outCells`,view:u.cells,type:`u32`,access:`read_write`},{name:`outEdges`,view:u.edgeIndices,type:`u32`,access:`read_write`},{name:`outEndpoints`,view:u.endpoints,type:`f32`,access:`read_write`}],z=a(t,{id:`${o}-write`,operation:S,variant:`write`,bindings:R,invocationCount:x,declarations:`const EDGE_SLOTS: u32 = ${l}u;
const OUTPUT_CAPACITY: u32 = ${w}u;`,body:`let mask = masks[masksOffset + index];
  if (mask == 0u) { return; }
  var outputSlot = 0u;
  if (index > 0u) { outputSlot = accepted[acceptedOffset + index - 1u]; }
  for (var edge = 0u; edge < EDGE_SLOTS; edge++) {
    if ((mask & (1u << edge)) == 0u) { continue; }
    if (outputSlot < OUTPUT_CAPACITY) {
      let candidate = index * EDGE_SLOTS + edge;
      outRows[outRowsOffset + outputSlot] = index;
      outEdges[outEdgesOffset + outputSlot] = edge;
      outCells[outCellsOffset + 2u * outputSlot] = cells[cellsOffset + 2u * index];
      outCells[outCellsOffset + 2u * outputSlot + 1u] = cells[cellsOffset + 2u * index + 1u];
      for (var component = 0u; component < 4u; component++) {
        outEndpoints[outEndpointsOffset + 4u * outputSlot + component] =
          candidateEndpoints[candidateEndpointsOffset + 4u * candidate + component];
      }
    }
    outputSlot++;
  }`}),B=u.groups&&s.groups?a(t,{id:`${o}-write-groups`,operation:S,variant:`write-groups`,bindings:[{name:`accepted`,view:M,type:`u32`,access:`read`},{name:`masks`,view:T,type:`u32`,access:`read`},{name:`groups`,view:s.groups,type:`u32`,access:`read`},{name:`outGroups`,view:u.groups,type:`u32`,access:`read_write`}],invocationCount:x,declarations:`const OUTPUT_CAPACITY: u32 = ${w}u;`,body:`let keptCount = countOneBits(masks[masksOffset + index]);
  if (keptCount == 0u) { return; }
  var outputSlot = 0u;
  if (index > 0u) { outputSlot = accepted[acceptedOffset + index - 1u]; }
  let group = groups[groupsOffset + index];
  for (var kept = 0u; kept < keptCount; kept++) {
    if (outputSlot + kept < OUTPUT_CAPACITY) {
      outGroups[outGroupsOffset + outputSlot + kept] = group;
    }
  }`}):null,V=a(t,{id:`${o}-total`,operation:S,variant:`total`,bindings:[{name:`accepted`,view:M,type:`u32`,access:`read`},{name:`total`,view:P,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const ROW_COUNT: u32 = ${x}u;`,body:`total[totalOffset] = accepted[acceptedOffset + ROW_COUNT - 1u];`}),H=this.ringAssembly?.getCommandNodes(t)??[];return[L,...new n({id:`${o}-scan`,input:j,output:M,mode:`inclusive`}).getCommandNodes(t),z,...B?[B]:[],V,i(t,{id:`${o}-publish`,operation:S,requiredCount:P,output:{ids:u.rows,status:{count:u.status.count,overflow:u.status.overflow,requiredCount:u.status.requiredCount}}}),...H]}};function E(e){return[e.rows,e.cells,e.edgeIndices,e.endpoints,e.groups,e.status.count,e.status.overflow,e.status.requiredCount]}var D=`
const OUTLINE_NO_ROW: u32 = 0xffffffffu;

fn outlineGetKey(row: u32) -> vec2u {
  return vec2u(cells[cellsOffset + 2u * row + 1u], cells[cellsOffset + 2u * row]);
}

fn outlineIsKeyLess(left: vec2u, right: vec2u) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}

/** Row of \`key\` in the sorted set, or OUTLINE_NO_ROW. */
fn outlineFindRow(key: vec2u, validRows: u32) -> u32 {
  if (key.x == 0u && key.y == 0u) {
    return OUTLINE_NO_ROW;
  }
  var low = 0u;
  var high = validRows;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (outlineIsKeyLess(outlineGetKey(middle), key)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < validRows && all(outlineGetKey(low) == key)) {
    return low;
  }
  return OUTLINE_NO_ROW;
}

/** Whether the neighbor key is in the set with the same group as \`row\`. */
fn outlineHasSameSideNeighbor(row: u32, neighbor: vec2u, validRows: u32) -> bool {
  let neighborRow = outlineFindRow(neighbor, validRows);
  if (neighborRow == OUTLINE_NO_ROW) {
    return false;
  }
  GROUP_TEST
}

fn outlineEmit(row: u32, edge: u32, a: vec2f, b: vec2f) {
  let slot = row * EDGE_SLOTS + edge;
  masks[masksOffset + row] = masks[masksOffset + row] | (1u << edge);
  candidateEndpoints[candidateEndpointsOffset + 4u * slot] = a.x;
  candidateEndpoints[candidateEndpointsOffset + 4u * slot + 1u] = a.y;
  candidateEndpoints[candidateEndpointsOffset + 4u * slot + 2u] = b.x;
  candidateEndpoints[candidateEndpointsOffset + 4u * slot + 3u] = b.y;
}
`,O=`
fn outlineClassifyRow(row: u32, validRows: u32) {
  let key = outlineGetKey(row);
  if (!dggs_h3_is_valid_cell_id(key)) {
    return;
  }
  let boundary = cellIndexH3GetBoundary(key);
  var neighbors: array<vec2u, 6>;
  var centers: array<vec2f, 6>;
  for (var direction = 1u; direction <= 6u; direction++) {
    let neighbor = cellTopologyH3Neighbor(key, direction);
    neighbors[direction - 1u] = neighbor;
    if (neighbor.x != 0u || neighbor.y != 0u) {
      centers[direction - 1u] = dggs_h3_get_center_lnglat(neighbor);
    }
  }
  for (var edge = 0u; edge < boundary.count; edge++) {
    let a = boundary.points[edge];
    let b = boundary.points[select(edge + 1u, 0u, edge + 1u == boundary.count)];
    // Wrap the second longitude next to the first so the midpoint is on the short arc.
    let wrappedB = vec2f(a.x + (b.x - a.x) - 360.0 * round((b.x - a.x) / 360.0), b.y);
    let midpoint = 0.5 * (a + wrappedB);
    let cosine = max(cos(radians(midpoint.y)), 0.01);
    var best = 1e30;
    var bestNeighbor = vec2u(0u);
    for (var slot = 0u; slot < 6u; slot++) {
      if (neighbors[slot].x == 0u && neighbors[slot].y == 0u) {
        continue;
      }
      let delta = centers[slot] - midpoint;
      let dx = (delta.x - 360.0 * round(delta.x / 360.0)) * cosine;
      let distance = dx * dx + delta.y * delta.y;
      if (distance < best) {
        best = distance;
        bestNeighbor = neighbors[slot];
      }
    }
    if (!outlineHasSameSideNeighbor(row, bestNeighbor, validRows)) {
      outlineEmit(row, edge, a, b);
    }
  }
}
`,k=`
fn outlineTileLatitude(row: u32, tileScale: f32) -> f32 {
  return dggs_web_mercator_tile_y_to_latitude(f32(row), tileScale);
}

fn outlineClassifyRow(row: u32, validRows: u32) {
  let key = outlineGetKey(row);
  if (!cellIsValidQuadbin(key)) {
    return;
  }
  let z = (key.x >> 20u) & 0x1fu;
  let tile = cellTopologyQuadbinGetTile(key);
  let tileCount = i32(1u << z);
  let tileScale = f32(1u << z);
  let west = f32(tile.x) / tileScale * 360.0 - 180.0;
  let east = f32(tile.x + 1u) / tileScale * 360.0 - 180.0;
  let north = outlineTileLatitude(tile.y, tileScale);
  let south = outlineTileLatitude(tile.y + 1u, tileScale);
  // Vertices NW, NE, SE, SW; edge e runs vertex e to vertex e + 1. Neighbors: N, E, S, W.
  var corners = array<vec2f, 4>(
    vec2f(west, north), vec2f(east, north), vec2f(east, south), vec2f(west, south)
  );
  var offsets = array<vec2i, 4>(vec2i(0, -1), vec2i(1, 0), vec2i(0, 1), vec2i(-1, 0));
  for (var edge = 0u; edge < 4u; edge++) {
    let neighborTile = vec2i(tile) + offsets[edge];
    var neighbor = vec2u(0u);
    if (neighborTile.y >= 0 && neighborTile.y < tileCount) {
      let x = u32(((neighborTile.x % tileCount) + tileCount) % tileCount);
      neighbor = cellTopologyQuadbinGetKey(x, u32(neighborTile.y), z);
    }
    if (!outlineHasSameSideNeighbor(row, neighbor, validRows)) {
      outlineEmit(row, edge, corners[edge], corners[(edge + 1u) % 4u]);
    }
  }
}
`,A=`
fn outlineClassifyRow(row: u32, validRows: u32) {
  let key = outlineGetKey(row);
  if (!dggs_s2_is_valid_cell_id(key)) { return; }
  // Match GPUCellGeometry's clockwise (00, 01, 11, 10) corner order.
  let neighborSlots = array<u32, 4>(0u, 3u, 1u, 2u);
  for (var edge = 0u; edge < 4u; edge++) {
    let neighbor = s2Neighbor(key, neighborSlots[edge]);
    if (!outlineHasSameSideNeighbor(row, neighbor, validRows)) {
      outlineEmit(row, edge, geometryGetVertex(key, edge), geometryGetVertex(key, (edge + 1u) % 4u));
    }
  }
}
`;export{T as t};
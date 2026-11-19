import{r as e}from"./gpu-command-graph-BdzdmV6u.js";import{t}from"./gpu-dispatch-utils-CbRzFAgT.js";import{o as n,t as r}from"./kernel-C3tfQOn4.js";import{a as i,i as a,n as o,o as s,r as c}from"./graph-chunk-dispatch-zBt8k2-o.js";import{E as l,S as u,T as d,c as f,f as p,p as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{s as g}from"./cell-index-wgsl-Cw7syfdA.js";import{t as _}from"./cell-keys-Dl1tIQ_Z.js";import{r as v}from"./cell-topology-wgsl-gkK5LfSP.js";import{t as y}from"./s2-neighbor-wgsl-ByEyc4ST.js";import{t as b}from"./h3-neighbor-wgsl-DdYw-iG5.js";import{t as x}from"./define-run-B-brkCe4.js";import{r as S,t as C}from"./point-observations-CeqP-hIo.js";import{r as w,t as T}from"./cells-Di9l-3nc.js";import{o as E}from"./pointer-CXys5Ipi.js";import{bl as D}from"./entry-preview-D70TdSbK.js";var O=4294967294,k=`
fn s2PathCenter(face: u32, coordinate: vec2u, size: u32) -> vec3i {
  return S2_FACE_NORMAL[face] * i32(size) + S2_FACE_COLUMN[face] * (i32(2u * coordinate.x + 1u) - i32(size)) + S2_FACE_ROW[face] * (i32(2u * coordinate.y + 1u) - i32(size));
}
fn s2PathBoundary(face: u32, coordinate: vec2u, direction: vec3i, size: u32) -> u32 {
  let columnSign = s2IntegerDot(direction, S2_FACE_COLUMN[face]);
  let axisSign = select(s2IntegerDot(direction, S2_FACE_ROW[face]), columnSign, columnSign != 0);
  let value = select(coordinate.y, coordinate.x, columnSign != 0);
  return select(value, size - 1u - value, axisSign > 0);
}
fn s2PathDistance(first: vec2u, second: vec2u) -> u32 {
  let firstFace = dggs_s2_get_face(first); let secondFace = dggs_s2_get_face(second);
  let firstCoordinate = dggs_s2_get_ij(first); let secondCoordinate = dggs_s2_get_ij(second);
  if (firstFace == secondFace) {
    let difference = abs(vec2i(firstCoordinate) - vec2i(secondCoordinate));
    return u32(difference.x) + u32(difference.y);
  }
  let size = 1u << dggs_s2_get_level(first);
  let firstNormal = S2_FACE_NORMAL[firstFace]; let secondNormal = S2_FACE_NORMAL[secondFace];
  let firstCenter = s2PathCenter(firstFace, firstCoordinate, size);
  let secondCenter = s2PathCenter(secondFace, secondCoordinate, size);
  if (s2IntegerDot(firstNormal, secondNormal) == 0) {
    let tangent = vec3i(1) - abs(firstNormal) - abs(secondNormal);
    let span = u32(abs(s2IntegerDot(firstCenter, tangent) - s2IntegerDot(secondCenter, tangent))) / 2u;
    return s2PathBoundary(firstFace, firstCoordinate, secondNormal, size) + s2PathBoundary(secondFace, secondCoordinate, firstNormal, size) + 1u + span;
  }
  var distance = 0xffffffffu;
  for (var side = 0u; side < 6u; side++) {
    let direction = S2_FACE_NORMAL[side];
    if (s2IntegerDot(firstNormal, direction) != 0) { continue; }
    let tangent = vec3i(1) - abs(firstNormal) - abs(direction);
    let span = u32(abs(s2IntegerDot(firstCenter, tangent) - s2IntegerDot(secondCenter, tangent))) / 2u;
    let candidateDistance = s2PathBoundary(firstFace, firstCoordinate, direction, size) + s2PathBoundary(secondFace, secondCoordinate, direction, size) + size + 1u + span;
    distance = min(distance, candidateDistance);
  }
  return distance;
}
`;function A(e,t){let{readCell:n,writeCell:r}=t,i=`${n(`origin`)}\n${n(`destination`)}
  var pathDistance = 0xffffffffu;
  var isValid = ${t.hasMask?`rowMask[rowMaskOffset + index] != 0u`:`true`};`,a=t.distances?`outDistances[outDistancesOffset + index] = pathDistance;`:``,o=t.counts?`outCounts[outCountsOffset + index] = select(pathDistance + 1u, 0u, pathDistance == 0xffffffffu);`:``;return e===`quadbin`?{declarations:`${g.source}\n${_}\n${v}`,body:`${i}
  isValid = isValid && cellIsValidQuadbin(origin) && cellIsValidQuadbin(destination) && ((origin.x >> 20u) & 31u) == ((destination.x >> 20u) & 31u);
  let level = (origin.x >> 20u) & 31u;
  let size = i32(1u << min(level, 26u));
  let first = cellTopologyQuadbinGetTile(origin);
  let last = cellTopologyQuadbinGetTile(destination);
  var difference = vec2i(last) - vec2i(first);
  if (abs(difference.x) > size / 2) { difference.x -= select(-size, size, difference.x > 0); }
  if (isValid) { pathDistance = u32(max(abs(difference.x), abs(difference.y))); }
  ${a}
  ${o}
  ${t.cells?`for (var slot = 0u; slot < ${t.maximumPathLength}u; slot++) {
    var result = vec2u(0u);
    if (pathDistance != 0xffffffffu && slot <= pathDistance) {
      let progress = min(vec2u(slot), vec2u(abs(difference)));
      let delta = vec2i(progress) * select(vec2i(-1), vec2i(1), difference >= vec2i(0));
      let tile = vec2i(first) + delta;
      result = cellTopologyQuadbinGetKey(u32((tile.x % size + size) % size), u32(tile.y), level);
    }
    ${r(`index * ${t.maximumPathLength}u + slot`,`result`)}
  }`:``}`}:{declarations:`${g.source}\n${y}\n${k}`,body:`${i}
  isValid = isValid && dggs_s2_is_valid_cell_id(origin) && dggs_s2_is_valid_cell_id(destination) && dggs_s2_get_level(origin) == dggs_s2_get_level(destination);
  if (isValid) {
    pathDistance = s2PathDistance(origin, destination);
    if (pathDistance > ${t.maximumSearchRadius}u) { pathDistance = 0xffffffffu; }
  }
  ${a}
  ${o}
  ${t.cells?`var current = origin;
  for (var slot = 0u; slot < ${t.maximumPathLength}u; slot++) {
    var result = vec2u(0u);
    if (pathDistance != 0xffffffffu && slot <= pathDistance) {
      result = current;
      if (slot < pathDistance) {
        for (var direction = 0u; direction < 4u; direction++) {
          let neighbor = s2Neighbor(current, direction);
          if (s2PathDistance(neighbor, destination) == pathDistance - slot - 1u) { current = neighbor; break; }
        }
      }
    }
    ${r(`index * ${t.maximumPathLength}u + slot`,`result`)}
  }`:``}`}}var j={origin:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,0,-1,5,-1,0,0,-1,-1,0,0,0,0,0,-1,1,0,0,-1,0,0,0,-1,0,-1,0],index:[0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,0,0,-1,1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,5,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]};function M(e,t){return`const ${e} = array<i32, ${t.length}>(${t.map(e=>`${e}`).join(`, `)});`}var N=`
${M(`CELL_TOPOLOGY_H3_ORIGIN_PENTAGON_ROTATIONS`,j.origin)}
${M(`CELL_TOPOLOGY_H3_INDEX_PENTAGON_ROTATIONS`,j.index)}

struct CellTopologyH3LocalIjk {
  ijk : vec3i,
  valid : u32,
};

/** Direction (0-6) from a base cell to an adjacent base cell, or -1. */
fn cellTopologyH3GetBaseCellDirection(fromBaseCell: u32, toBaseCell: u32) -> i32 {
  for (var direction = 0u; direction < 7u; direction++) {
    if ((CELL_TOPOLOGY_H3_BASE_CELL_NEIGHBORS[fromBaseCell * 7u + direction] & 0xffu) == toBaseCell) {
      return i32(direction);
    }
  }
  return -1;
}

/** H3 \`_h3RotatePent60cw\`: rotates again when the leading digit lands on the deleted K axis. */
fn cellTopologyH3RotatePentagon60Clockwise(cell: vec2u) -> vec2u {
  var result = cell;
  var foundFirstNonZeroDigit = false;
  let resolution = dggs_h3_get_resolution(cell);
  for (var level = 1u; level <= resolution; level++) {
    result = cellTopologyH3SetDigit(
      result,
      level,
      dggs_h3_rotate_digit_60_cw(dggs_h3_get_digit(result, level))
    );
    if (!foundFirstNonZeroDigit && dggs_h3_get_digit(result, level) != 0u) {
      foundFirstNonZeroDigit = true;
      if (dggs_h3_get_leading_non_zero_digit(result) == 1u) {
        result = cellTopologyH3Rotate60Clockwise(result);
      }
    }
  }
  return result;
}

/** IJK of a cell in its own base cell's coordinate system (base cell center at the origin). */
fn cellTopologyH3GetBaseCellIjk(cell: vec2u) -> vec3i {
  var ijk = vec3i(0);
  let resolution = dggs_h3_get_resolution(cell);
  for (var level = 1u; level <= resolution; level++) {
    if (dggs_h3_is_resolution_class_iii(level)) {
      ijk = dggs_h3_down_ap7(ijk);
    } else {
      ijk = dggs_h3_down_ap7r(ijk);
    }
    ijk = dggs_h3_neighbor(ijk, dggs_h3_get_digit(cell, level));
  }
  return ijk;
}

fn cellTopologyH3GetLocalIjk(origin: vec2u, cell: vec2u) -> CellTopologyH3LocalIjk {
  let invalidResult = CellTopologyH3LocalIjk(vec3i(0), 0u);
  if (!dggs_h3_is_valid_cell_id(origin) || !dggs_h3_is_valid_cell_id(cell)) {
    return invalidResult;
  }
  let resolution = dggs_h3_get_resolution(origin);
  if (dggs_h3_get_resolution(cell) != resolution) {
    return invalidResult;
  }
  let originBaseCell = dggs_h3_get_base_cell(origin);
  let cellBaseCell = dggs_h3_get_base_cell(cell);
  var direction = 0;
  if (originBaseCell != cellBaseCell) {
    direction = cellTopologyH3GetBaseCellDirection(originBaseCell, cellBaseCell);
    if (direction < 0) {
      return invalidResult;
    }
  }
  let originOnPentagon = dggs_h3_is_base_cell_pentagon(originBaseCell);
  let cellOnPentagon = dggs_h3_is_base_cell_pentagon(cellBaseCell);

  // Rotate the cell into the orientation of the origin base cell.
  var rotated = cell;
  if (direction != 0) {
    let rotations = CELL_TOPOLOGY_H3_BASE_CELL_NEIGHBORS[originBaseCell * 7u + u32(direction)] >> 8u;
    for (var rotation = 0u; rotation < rotations; rotation++) {
      if (cellOnPentagon) {
        rotated = cellTopologyH3RotatePentagon60Clockwise(rotated);
      } else {
        rotated = cellTopologyH3Rotate60Clockwise(rotated);
      }
    }
  }
  var ijk = cellTopologyH3GetBaseCellIjk(rotated);

  if (direction != 0) {
    var pentagonRotations = 0;
    var directionRotations = 0;
    if (originOnPentagon) {
      let leading = dggs_h3_get_leading_non_zero_digit(origin);
      let rotations = CELL_TOPOLOGY_H3_ORIGIN_PENTAGON_ROTATIONS[leading * 7u + u32(direction)];
      if (rotations < 0) {
        return invalidResult;
      }
      pentagonRotations = rotations;
      directionRotations = rotations;
    } else if (cellOnPentagon) {
      let leading = dggs_h3_get_leading_non_zero_digit(rotated);
      let rotations = CELL_TOPOLOGY_H3_INDEX_PENTAGON_ROTATIONS[u32(direction) * 7u + leading];
      if (rotations < 0) {
        return invalidResult;
      }
      pentagonRotations = rotations;
    }
    for (var rotation = 0; rotation < pentagonRotations; rotation++) {
      ijk = dggs_h3_ijk_rotate_60_ccw(ijk);
    }
    // The neighboring base cell center, scaled to the cell resolution.
    var offset = dggs_h3_get_unit_vector(u32(direction));
    for (var level = resolution; level >= 1u; level--) {
      if (dggs_h3_is_resolution_class_iii(level)) {
        offset = dggs_h3_down_ap7(offset);
      } else {
        offset = dggs_h3_down_ap7r(offset);
      }
    }
    for (var rotation = 0; rotation < directionRotations; rotation++) {
      offset = dggs_h3_ijk_rotate_60_ccw(offset);
    }
    ijk = dggs_h3_ijk_normalize(ijk + offset);
  } else if (originOnPentagon && cellOnPentagon) {
    let originLeading = dggs_h3_get_leading_non_zero_digit(origin);
    let cellLeading = dggs_h3_get_leading_non_zero_digit(rotated);
    let rotations = CELL_TOPOLOGY_H3_ORIGIN_PENTAGON_ROTATIONS[originLeading * 7u + cellLeading];
    if (rotations < 0) {
      return invalidResult;
    }
    for (var rotation = 0; rotation < rotations; rotation++) {
      ijk = dggs_h3_ijk_rotate_60_ccw(ijk);
    }
  }
  return CellTopologyH3LocalIjk(ijk, 1u);
}

/** H3 \`gridDistance\` between two local IJK coordinates of the same origin. */
fn cellTopologyH3GetGridDistance(a: vec3i, b: vec3i) -> u32 {
  let difference = dggs_h3_ijk_normalize(a - b);
  return u32(max(difference.x, max(difference.y, difference.z)));
}

/** Cube coordinates \`(-I, J, I - J)\` of a local IJK, as H3 \`ijkToCube\`. */
fn cellTopologyH3GetCube(ijk: vec3i) -> vec3i {
  let i = ijk.x - ijk.z;
  let j = ijk.y - ijk.z;
  return vec3i(-i, j, i - j);
}

/**
 * Rounds the interpolation \`from + (to - from) * step / distance\` of two cube coordinates to
 * the nearest cube coordinate, with H3 \`cubeRound\` (half away from zero, largest rounding error
 * recomputed) evaluated in exact integer arithmetic. \`quotient\` and \`remainder\` carry
 * \`floor((to - from) * step / distance)\` and its remainder, advanced by the caller one step at a
 * time so nothing overflows for long paths. Returns the local \`(i, j)\`.
 */
fn cellTopologyH3RoundCubeStep(
  start: vec3i,
  quotient: vec3i,
  remainder: vec3i,
  pathDistance: i32
) -> vec2i {
  let base = start + quotient;
  var rounded = vec3i(0);
  var roundingError = vec3i(0);
  for (var axis = 0; axis < 3; axis++) {
    let whole = base[axis];
    let part = remainder[axis];
    // The value is whole + part / pathDistance with part in [0, pathDistance).
    let roundsUp = select(2 * part > pathDistance, 2 * part >= pathDistance, whole >= 0);
    rounded[axis] = whole + select(0, 1, roundsUp);
    roundingError[axis] = select(2 * part, 2 * pathDistance - 2 * part, roundsUp);
  }
  if (roundingError.x > roundingError.y && roundingError.x > roundingError.z) {
    rounded.x = -rounded.y - rounded.z;
  } else if (roundingError.y > roundingError.z) {
    rounded.y = -rounded.x - rounded.z;
  } else {
    rounded.z = -rounded.x - rounded.y;
  }
  return vec2i(-rounded.x, rounded.y);
}
`,P=256,F=4294967295,I=16777216,L=`GPUCellGridPath`,R=class{id;props;maximumPathLength;estimatedCommandNodeCount;chunkPlan;recordChunks;constructor(e){this.id=e.id??`cell-grid-path`,this.props=e;let t=this.id;d(t,[`h3`,`quadbin`,`s2`].includes(e.family),`family must be 'h3', 'quadbin' or 's2'`),p(t,`maximumSearchRadius`,e.maximumSearchRadius??4294967294,0,O),d(t,e.wordOrder===void 0||[`little-endian`,`high-low`].includes(e.wordOrder),`wordOrder must be 'little-endian' or 'high-low'`);let{output:n}=e;d(t,!!(n.distances||n.cells),`output needs distances or cells`),d(t,!n.counts||!!n.cells,`output.counts needs output.cells`),this.maximumPathLength=n.cells?e.maximumPathLength:0,n.cells&&p(t,`maximumPathLength`,this.maximumPathLength,1,256),this.chunkPlan=c(t,{origins:e.origins,destinations:e.destinations,mask:e.mask,distances:n.distances,counts:n.counts}),p(t,`origins row count`,this.chunkPlan.totalLength);for(let e of this.chunkPlan.chunks)e.views.origins&&l(e.views.origins,[`uint32x2`],`${t} origins chunk ${e.chunkIndex}`),u(t,`origins chunk components`,e.length*2,F),e.views.destinations&&l(e.views.destinations,[`uint32x2`],`${t} destinations chunk ${e.chunkIndex}`),u(t,`destinations chunk components`,e.length*2,F),e.views.mask&&l(e.views.mask,[`uint32`],`${t} mask chunk ${e.chunkIndex}`),u(t,`mask chunk components`,e.length,F),e.views.distances&&l(e.views.distances,[`uint32`],`${t} distances chunk ${e.chunkIndex}`),u(t,`distances chunk components`,e.length,F),e.views.counts&&l(e.views.counts,[`uint32`],`${t} counts chunk ${e.chunkIndex}`),u(t,`counts chunk components`,e.length,F);let r={};if(r.pathCells=n.cells?f(n.cells):[],n.cells){d(t,n.cells.length===this.chunkPlan.totalLength*this.maximumPathLength,`output.cells length must equal input rows * record stride`),d(t,r.pathCells.length===this.chunkPlan.chunks.length,`output.cells must match input chunk topology`);for(let e of this.chunkPlan.chunks){let n=r.pathCells[e.chunkIndex];l(n,[`uint32x2`],`${t} pathCells chunk ${e.chunkIndex}`),d(t,n.length===e.length*this.maximumPathLength,`output.cells chunk ${e.chunkIndex} length must equal input chunk rows * record stride`),u(t,`pathCells chunk components`,n.length*2,F)}}this.recordChunks=r,a(t,e.output),this.estimatedCommandNodeCount=+(this.chunkPlan.chunks.length>0)+this.chunkPlan.nonemptyChunkCount,m(t,this.estimatedCommandNodeCount),h(t,[n.distances,n.cells,n.counts],[e.origins,e.destinations,e.mask])}getCommandNodes(a){let{id:c,props:l,maximumPathLength:d}=this,{output:f}=l;u(c,`storage bindings`,3+Number(!!l.mask)+Object.values(l.output).filter(Boolean).length,a.device.limits.maxStorageBuffersPerShaderStage),s(a,c,this.chunkPlan),i(a,c,{"output.cells":f.cells});let p=this.chunkPlan.chunks.map(e=>t(c,Math.max(e.length*1,1),P,a.device.limits.maxComputeWorkgroupsPerDimension)),{slots:m,nodes:h}=o(a,{id:`${c}-chunks`,plan:this.chunkPlan}),_,v=this.chunkPlan.chunks.filter(e=>e.length>0).map(t=>{let i=[{name:`chunkParameters`,view:m[t.chunkIndex],type:`u32`,access:`read`},{name:`origins`,view:t.views.origins,type:`u32`,access:`read`},{name:`destinations`,view:t.views.destinations,type:`u32`,access:`read`},...l.mask?[{name:`rowMask`,view:t.views.mask,type:`u32`,access:`read`}]:[],...f.distances?[{name:`outDistances`,view:t.views.distances,type:`u32`,access:`read_write`}]:[],...f.cells?[{name:`outCells`,view:this.recordChunks.pathCells[t.chunkIndex],type:`u32`,access:`read_write`}]:[],...f.counts?[{name:`outCounts`,view:t.views.counts,type:`u32`,access:`read_write`}]:[]],a=l.wordOrder===`high-low`,o=e=>`let ${e}Words = vec2u(${e}s[${e}sOffset + 2u * index], ${e}s[${e}sOffset + 2u * index + 1u]);
  let ${e} = ${a?`${e}Words`:`${e}Words.yx`};`,s=(e,t)=>`
    outCells[outCellsOffset + 2u * (${e})] = ${t}.${a?`x`:`y`};
    outCells[outCellsOffset + 2u * (${e}) + 1u] = ${t}.${a?`y`:`x`};`,u=f.cells?`
  var pathLength = 0u;
  var isPathValid = false;
  if (pathDistance != GRID_DISTANCE_UNDEFINED) {
    isPathValid = true;
    let distanceValue = i32(pathDistance);
    let startCube = cellTopologyH3GetCube(originLocal.ijk);
    let delta = cellTopologyH3GetCube(destinationLocal.ijk) - startCube;
    // floor((delta * step) / distance), advanced one step at a time without overflow.
    var quotientStep = vec3i(0);
    var remainderStep = vec3i(0);
    if (distanceValue > 0) {
      quotientStep = delta / vec3i(distanceValue);
      remainderStep = delta - quotientStep * distanceValue;
      for (var axis = 0; axis < 3; axis++) {
        if (remainderStep[axis] < 0) {
          remainderStep[axis] += distanceValue;
          quotientStep[axis] -= 1;
        }
      }
    }
    cellGridPath[0] = origin;
    pathLength = 1u;
    var quotient = vec3i(0);
    var remainder = vec3i(0);
    var previous = origin;
    let storedLength = min(pathDistance + 1u, PATH_STRIDE);
    for (var pathStep = 1u; pathStep < storedLength; pathStep++) {
      quotient += quotientStep;
      remainder += remainderStep;
      for (var axis = 0; axis < 3; axis++) {
        if (remainder[axis] >= distanceValue) {
          remainder[axis] -= distanceValue;
          quotient[axis] += 1;
        }
      }
      let roundedTarget = cellTopologyH3RoundCubeStep(startCube, quotient, remainder, distanceValue);
      var matchCount = 0u;
      var matchedNeighbor = vec2u(0u);
      for (var direction = 1u; direction <= 6u; direction++) {
        let neighbor = cellTopologyH3Neighbor(previous, direction);
        if (neighbor.x == 0u && neighbor.y == 0u) {
          continue;
        }
        let neighborLocal = cellTopologyH3GetLocalIjk(origin, neighbor);
        if (
          neighborLocal.valid != 0u &&
          neighborLocal.ijk.x - neighborLocal.ijk.z == roundedTarget.x &&
          neighborLocal.ijk.y - neighborLocal.ijk.z == roundedTarget.y
        ) {
          matchCount++;
          matchedNeighbor = neighbor;
        }
      }
      if (matchCount != 1u) {
        isPathValid = false;
        break;
      }
      cellGridPath[pathStep] = matchedNeighbor;
      previous = matchedNeighbor;
      pathLength = pathStep + 1u;
    }
  }
  for (var slot = 0u; slot < PATH_STRIDE; slot++) {
    let row = index * PATH_STRIDE + slot;
    if (isPathValid && slot < pathLength) {${s(`row`,`cellGridPath[slot]`)}
    } else {${s(`row`,`vec2u(0u)`)}
    }
  }
  ${f.counts?`outCounts[outCountsOffset + index] = select(0u, pathDistance + 1u, isPathValid);`:``}`:``,h=l.family===`h3`?void 0:A(l.family,{readCell:o,writeCell:s,maximumPathLength:d,maximumSearchRadius:l.maximumSearchRadius??4294967294,hasMask:!!l.mask,distances:!!f.distances,cells:!!f.cells,counts:!!f.counts}),v={id:`${c}-path`,operation:L,variant:`${l.family}-${f.cells?`path`:`distance`}`,bindings:i,invocationCount:t.length*1,declarations:h?.declarations??`const GRID_DISTANCE_UNDEFINED: u32 = 0xffffffffu;
const PATH_STRIDE: u32 = ${Math.max(d,1)}u;
var<private> cellGridPath: array<vec2u, PATH_STRIDE>;
${g.source}

fn cellMaskLow32(bits: u32) -> u32 {
  return select(0xffffffffu, (1u << bits) - 1u, bits < 32u);
}

fn cellShiftRight(value: vec2u, shift: u32) -> vec2u {
  if (shift == 0u) {
    return value;
  }
  if (shift >= 64u) {
    return vec2u(0u);
  }
  if (shift >= 32u) {
    return vec2u(0u, value.x >> (shift - 32u));
  }
  return vec2u(value.x >> shift, (value.y >> shift) | (value.x << (32u - shift)));
}

fn cellShiftLeft(value: vec2u, shift: u32) -> vec2u {
  if (shift == 0u) {
    return value;
  }
  if (shift >= 64u) {
    return vec2u(0u);
  }
  if (shift >= 32u) {
    return vec2u(value.y << (shift - 32u), 0u);
  }
  return vec2u((value.x << shift) | (value.y >> (32u - shift)), value.y << shift);
}

fn cellMaskLow(bits: u32) -> vec2u {
  if (bits >= 32u) {
    return vec2u(cellMaskLow32(bits - 32u), 0xffffffffu);
  }
  return vec2u(0u, cellMaskLow32(bits));
}

/** Path bits [lowBit, lowBit + width) of a canonical key, as an integer. */
fn cellGetCompactKey(key: vec2u, lowBit: u32, width: u32) -> vec2u {
  return cellShiftRight(key, lowBit) & cellMaskLow(width);
}

/** Rebuilds a canonical key from its compact key, resolution and family header. */
fn cellGetKey(compact: vec2u, headerHigh: u32, resolution: u32, lowBit: u32) -> vec2u {
  let header = vec2u(headerHigh | (resolution << 20u), 0u);
  return header | cellShiftLeft(compact, lowBit) | cellMaskLow(lowBit);
}

/** Quadbin validity, mirroring quadbin-py \`is_valid_cell\`. */
fn cellIsValidQuadbin(key: vec2u) -> bool {
  let resolution = (key.x >> 20u) & 0x1fu;
  if ((key.x & 0x80000000u) != 0u || (key.x & 0x40000000u) == 0u ||
      ((key.x >> 27u) & 7u) != 1u || resolution > 26u) {
    return false;
  }
  let unused = cellMaskLow(52u - 2u * resolution);
  return all((key & unused) == unused);
}

${b}
${N}`,body:h?.body??`${o(`origin`)}
  ${o(`destination`)}
  var isValid = true;
  ${l.mask?`isValid = rowMask[rowMaskOffset + index] != 0u;`:``}
  var pathDistance = GRID_DISTANCE_UNDEFINED;
  var originLocal = CellTopologyH3LocalIjk(vec3i(0), 0u);
  var destinationLocal = CellTopologyH3LocalIjk(vec3i(0), 0u);
  if (isValid) {
    originLocal = cellTopologyH3GetLocalIjk(origin, origin);
    destinationLocal = cellTopologyH3GetLocalIjk(origin, destination);
    if (originLocal.valid != 0u && destinationLocal.valid != 0u) {
      pathDistance = cellTopologyH3GetGridDistance(originLocal.ijk, destinationLocal.ijk);
    }
  }
  ${f.distances?`outDistances[outDistancesOffset + index] = pathDistance;`:``}
  ${u}`},y=v.bindings.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`),x={origins:2,destinations:2,rowMask:1,outDistances:1,outCells:d*2,outCounts:1},S=v.bindings.filter(e=>e.name!==`chunkParameters`).map(e=>`let ${e.name}Offset = arrayLength(&${e.name}) - rowCount * ${x[e.name]}u;`).join(`
  `),C=_??`${y}
${v.declarations??``}
@compute @workgroup_size(${P})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${I}u) { return; }
  let index = workgroupIndex * ${P}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount * 1u) { return; }
  ${S}
  ${v.body}
}`;_=C;let w=v.bindings,T=v.operation,E=p[t.chunkIndex];return e({id:`${v.id}-${t.chunkIndex}`,resources:v.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:v.operation,...v.variant?{variant:v.variant}:{},commandCount:1,maximumWorkgroupCount:E.x*E.y*E.z,maximumInvocationCount:t.length*1},compile:({device:e})=>{let t=new r(e,{id:T,source:C,shaderLayout:{bindings:w.map((e,t)=>({name:e.name,group:0,location:t,type:e.access===`read`?`read-only-storage`:`storage`}))}});return{encode:({computePass:e,getBuffer:r})=>t.dispatch(e,{bindings:Object.fromEntries(w.map(e=>[e.name,n(e.view,r)])),...E}),destroy:()=>t.destroy()}}})});return[...h,...v]}},z=5,B=Math.cos((C[1]+C[3])/2*Math.PI/180),V=x({parameters:D,geographicWindow:C,caption:`H3 paths from four Honshu epicenters to the epicenters around the pointer.`,createInputs:()=>({pointer:[...E(C)],origins:Float32Array.from(S([[140.8,36.45],[142.1,38.05],[141,37.35],[143.4,37.85]]).flat())}),build(e,t,n){let r=t.origins.length/2,i=e.upload(`origins`,`float32x2`,t.origins),a=e.allocate(`destinations`,`float32x2`,r),o=w(e,`h3`,z,i,`origin-keys`),s=w(e,`h3`,z,a,`destination-keys`),c=n.maximumPathLength,l=e.allocate(`path-cells`,`uint32x2`,r*c),u=e.allocate(`counts`,`uint32`,r);return e.graph.add(new R({family:`h3`,origins:o.view,destinations:s.view,maximumPathLength:c,output:{cells:l.view,counts:u.view}})),{outputs:{counts:u,origins:i,destinations:a,pairs:e.upload(`pairs`,`uint32`,Uint32Array.from({length:r*c},(e,t)=>Math.floor(t/c))),fill:T(e,`h3`,l)},update:e=>{let n=[];for(let i=0;i<r;i++){let a=(i+.5)/r*2*Math.PI;n.push([t.pointer[0]+Math.cos(a)*e.spread/B,t.pointer[1]+Math.sin(a)*e.spread])}a.write(Float32Array.from(S(n).flat()))}}},rebuildOn:[`maximumPathLength`],interaction:{kind:`place`,hint:`choose a destination`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){t.pointer=[e.x,e.y];let a=i(n.counts),o=a?[...a].filter(e=>e>0):[],s=o.length>0?Math.max(...o):0,c=o.filter(e=>e>r.maximumPathLength).length;return{rerun:!0,glyphs:[{shape:`crosshair`,at:t.pointer},{shape:`ring`,at:t.pointer,radius:r.spread,dashed:!0}],readout:c>0?`longest path ${s} cells · ${c} of ${o.length} drawn cut off at maximumPathLength ${r.maximumPathLength}`:`longest path ${s} cells · all ${o.length} drawn whole (maximumPathLength ${r.maximumPathLength})`}}},draw:({outputs:e})=>[{layer:`polygons`,positions:e.fill.boundaries,indices:e.fill.indices,owners:e.fill.owners,values:e.pairs,triangleCount:e.fill.triangleCount,alpha:.85},{layer:`dots`,positions:e.origins,radius:4,color:[1,1,1,1]},{layer:`dots`,positions:e.destinations,radius:4,color:[1,1,1,1]}]});export{V as default};
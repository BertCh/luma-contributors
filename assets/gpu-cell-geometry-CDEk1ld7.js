import{r as e}from"./spatial-context-DwZEd4EZ.js";import{r as t}from"./gpu-command-graph-BdzdmV6u.js";import{t as n}from"./gpu-dispatch-utils-CbRzFAgT.js";import{o as r,t as i}from"./kernel-C3tfQOn4.js";import{s as a}from"./cell-index-wgsl-Cw7syfdA.js";import{a as o,i as s,n as c,o as l,r as u}from"./graph-chunk-dispatch-zBt8k2-o.js";import{E as d,S as f,T as p,c as m,f as h,p as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import"./cell-keys-Dl1tIQ_Z.js";import{s as v}from"./wgsl-common-DfyXiYB0.js";var y=b();function b(e={unitType:`vec3f`,coordinateToPlane:`dggs_h3_ijk_to_hex2d`,source:x()}){return`
${v}
const CELL_INDEX_H3_BOUNDARY_MAX_VERTICES: u32 = 10u;
const CELL_INDEX_H3_BOUNDARY_NO_OVERAGE: u32 = 0u;
const CELL_INDEX_H3_BOUNDARY_FACE_EDGE: u32 = 1u;
const CELL_INDEX_H3_BOUNDARY_NEW_FACE: u32 = 2u;

struct CellIndexH3Boundary {
  count : u32,
  points : array<vec2f, 10>,
  units : array<${e.unitType}, 10>,
  // Offset of each original topological vertex, excluding inserted face-distortion vertices.
  vertexOffsets : array<u32, 6>,
};

// H3 _adjustOverageClassII, including the substrate-grid variant used for boundary vertices.
fn cellIndexH3BoundaryAdjustOverage(
  input: DggsH3FaceIJK,
  resolution: u32,
  pentagonLeading4: bool,
  substrate: bool
) -> DggsH3FaceIJKOverage {
  var faceIJK = input;
  let unscaledDimension = dggs_h3_get_max_dim_by_cii_resolution(resolution);
  var maximumDimension = unscaledDimension;
  if (substrate) {
    maximumDimension *= 3;
  }
  let dimension = faceIJK.coord.x + faceIJK.coord.y + faceIJK.coord.z;
  if (substrate && dimension == maximumDimension) {
    return DggsH3FaceIJKOverage(faceIJK, CELL_INDEX_H3_BOUNDARY_FACE_EDGE);
  }
  if (dimension <= maximumDimension) {
    return DggsH3FaceIJKOverage(faceIJK, CELL_INDEX_H3_BOUNDARY_NO_OVERAGE);
  }

  var quadrant = 1u;
  if (faceIJK.coord.z > 0) {
    quadrant = select(2u, 3u, faceIJK.coord.y > 0);
    if (quadrant == 2u && pentagonLeading4) {
      let origin = vec3i(maximumDimension, 0, 0);
      faceIJK.coord = dggs_h3_ijk_rotate_60_cw(faceIJK.coord - origin) + origin;
    }
  }
  let neighbor = dggs_h3_get_face_neighbor(faceIJK.face, quadrant);
  faceIJK.face = neighbor.face;
  for (var rotation = 0u; rotation < neighbor.valid; rotation++) {
    faceIJK.coord = dggs_h3_ijk_rotate_60_ccw(faceIJK.coord);
  }
  var unitScale = unscaledDimension / 2;
  if (substrate) {
    unitScale *= 3;
  }
  faceIJK.coord = dggs_h3_ijk_normalize(faceIJK.coord + neighbor.coord * unitScale);
  var overage = CELL_INDEX_H3_BOUNDARY_NEW_FACE;
  if (substrate && faceIJK.coord.x + faceIJK.coord.y + faceIJK.coord.z == maximumDimension) {
    overage = CELL_INDEX_H3_BOUNDARY_FACE_EDGE;
  }
  return DggsH3FaceIJKOverage(faceIJK, overage);
}

// Quadrant (1 = IJ, 2 = KI, 3 = JK) of the edge of face \`fromFace\` that borders \`toFace\`, or 0.
fn cellIndexH3BoundaryGetFaceDirection(fromFace: u32, toFace: u32) -> u32 {
  for (var quadrant = 1u; quadrant <= 3u; quadrant++) {
    if (dggs_h3_get_face_neighbor(fromFace, quadrant).face == toFace) {
      return quadrant;
    }
  }
  return 0u;
}

// True when a substrate coordinate lies exactly on an icosahedron face edge.
fn cellIndexH3BoundaryIsOnFaceEdge(coord: vec3i, adjustedResolution: u32) -> bool {
  return coord.x + coord.y + coord.z == 3 * dggs_h3_get_max_dim_by_cii_resolution(adjustedResolution);
}

${e.source}

fn cellIndexH3BoundaryPush(boundary: ptr<function, CellIndexH3Boundary>, unitVector: ${e.unitType}) {
  if ((*boundary).count < CELL_INDEX_H3_BOUNDARY_MAX_VERTICES) {
    (*boundary).points[(*boundary).count] = cellIndexH3BoundaryUnitVectorToLngLat(unitVector);
    (*boundary).units[(*boundary).count] = unitVector;
    (*boundary).count += 1u;
  }
}

// H3 _faceIjkToCellBoundary / _faceIjkPentToCellBoundary for the whole loop. \`center\` is the
// overage-adjusted face/IJK of the cell (dggs_h3_get_center_face_ijk).
fn cellIndexH3BoundaryFromFaceIJK(
  center: DggsH3FaceIJK,
  resolution: u32,
  isPentagon: bool
) -> CellIndexH3Boundary {
  var boundary: CellIndexH3Boundary;
  let isClassIII = dggs_h3_is_resolution_class_iii(resolution);
  let vertexTotal = select(6u, 5u, isPentagon);

  // _faceIjkToVerts: vertices on the aperture 33r (7r for Class III) substrate grid.
  var adjustedResolution = resolution;
  var centerCoord = dggs_h3_down_ap3r(dggs_h3_down_ap3(center.coord));
  if (isClassIII) {
    centerCoord = dggs_h3_down_ap7r(centerCoord);
    adjustedResolution += 1u;
  }
  var vertexCoords: array<vec3i, 6>;
  for (var vertex = 0u; vertex < vertexTotal; vertex++) {
    vertexCoords[vertex] = dggs_h3_ijk_normalize(
      centerCoord + dggs_h3_get_vertex_offset(resolution, vertex)
    );
  }

  var lastFace = 0xffffffffu;
  var lastOverage = CELL_INDEX_H3_BOUNDARY_NO_OVERAGE;
  var lastCoord = vec3i(0);
  // One extra iteration tests the last edge for a distortion vertex.
  for (var vert = 0u; vert <= vertexTotal; vert++) {
    let vertex = vert % vertexTotal;
    var adjusted = cellIndexH3BoundaryAdjustOverage(
      DggsH3FaceIJK(center.face, vertexCoords[vertex], 1u),
      adjustedResolution,
      false,
      true
    );
    if (isPentagon) {
      // _adjustPentVertOverage: repeat until the vertex settles on a face.
      for (var settlePass = 0u; settlePass < 4u && adjusted.overage == CELL_INDEX_H3_BOUNDARY_NEW_FACE; settlePass++) {
        adjusted = cellIndexH3BoundaryAdjustOverage(
          adjusted.faceIJK,
          adjustedResolution,
          false,
          true
        );
      }
    }
    let current = adjusted.faceIJK;

    if (isClassIII && vert > 0u) {
      if (isPentagon) {
        // All Class III pentagon edges cross an icosahedron edge: re-express the current vertex on
        // the previous vertex's face and intersect with the edge between the two faces.
        let currentToLast = cellIndexH3BoundaryGetFaceDirection(current.face, lastFace);
        let lastToCurrent = cellIndexH3BoundaryGetFaceDirection(lastFace, current.face);
        if (currentToLast != 0u && lastToCurrent != 0u) {
          let orient = dggs_h3_get_face_neighbor(current.face, currentToLast);
          var transformed = current.coord;
          for (var rotation = 0u; rotation < orient.valid; rotation++) {
            transformed = dggs_h3_ijk_rotate_60_ccw(transformed);
          }
          let unitScale = (dggs_h3_get_max_dim_by_cii_resolution(adjustedResolution) / 2) * 3;
          transformed = dggs_h3_ijk_normalize(transformed + orient.coord * unitScale);
          let point = cellIndexH3BoundaryIntersectFaceEdge(
            ${e.coordinateToPlane}(lastCoord),
            ${e.coordinateToPlane}(transformed),
            adjustedResolution,
            lastToCurrent
          );
          cellIndexH3BoundaryPush(
            &boundary,
            cellIndexH3BoundaryHex2dToUnitVector(point, lastFace, adjustedResolution)
          );
        }
      } else if (current.face != lastFace && lastOverage != CELL_INDEX_H3_BOUNDARY_FACE_EDGE) {
        // The crossed edge borders whichever of the two faces is not the center face.
        let neighborFace = select(lastFace, current.face, lastFace == center.face);
        let quadrant = cellIndexH3BoundaryGetFaceDirection(center.face, neighborFace);
        let lastVertex = (vertex + 5u) % 6u;
        // An intersection exactly at a hexagon vertex needs no extra vertex.
        let isAtVertex =
          cellIndexH3BoundaryIsOnFaceEdge(vertexCoords[vertex], adjustedResolution) ||
          cellIndexH3BoundaryIsOnFaceEdge(vertexCoords[lastVertex], adjustedResolution);
        if (quadrant != 0u && !isAtVertex) {
          let point = cellIndexH3BoundaryIntersectFaceEdge(
            ${e.coordinateToPlane}(vertexCoords[lastVertex]),
            ${e.coordinateToPlane}(vertexCoords[vertex]),
            adjustedResolution,
            quadrant
          );
          cellIndexH3BoundaryPush(
            &boundary,
            cellIndexH3BoundaryHex2dToUnitVector(point, center.face, adjustedResolution)
          );
        }
      }
    }

    if (vert < vertexTotal) {
      boundary.vertexOffsets[vertex] = boundary.count;
      cellIndexH3BoundaryPush(
        &boundary,
        cellIndexH3BoundaryHex2dToUnitVector(
          ${e.coordinateToPlane}(current.coord),
          current.face,
          adjustedResolution
        )
      );
    }
    lastFace = current.face;
    lastOverage = adjusted.overage;
    lastCoord = current.coord;
  }
  return boundary;
}

// All boundary vertices of an H3 cell; \`count\` is 0 for invalid cells.
fn cellIndexH3GetBoundary(cell: vec2u) -> CellIndexH3Boundary {
  var boundary: CellIndexH3Boundary;
  if (!dggs_h3_is_valid_cell_id(cell)) {
    return boundary;
  }
  let center = dggs_h3_get_center_face_ijk(cell);
  if (center.valid == 0u) {
    return boundary;
  }
  let isPentagon = dggs_h3_is_base_cell_pentagon(dggs_h3_get_base_cell(cell)) &&
    dggs_h3_get_leading_non_zero_digit(cell) == 0u;
  return cellIndexH3BoundaryFromFaceIJK(center, dggs_h3_get_resolution(cell), isPentagon);
}

fn cellIndexH3GetBoundaryVertexCount(cell: vec2u) -> u32 {
  return cellIndexH3GetBoundary(cell).count;
}

fn cellIndexH3GetBoundaryVertex(cell: vec2u, vertexIndex: u32) -> vec2f {
  let boundary = cellIndexH3GetBoundary(cell);
  if (vertexIndex >= boundary.count) {
    return vec2f(0.0);
  }
  return boundary.points[vertexIndex];
}
`}function x(){return`
// Intersection of the segment p0-p1 with the line through edge0-edge1 (H3 _v2dIntersect).
fn cellIndexH3BoundaryIntersect(p0: vec2f, p1: vec2f, edge0: vec2f, edge1: vec2f) -> vec2f {
  let s1 = p1 - p0;
  let s2 = edge1 - edge0;
  let t = (s2.x * (p0.y - edge0.y) - s2.y * (p0.x - edge0.x)) / (-s2.x * s1.y + s1.x * s2.y);
  return p0 + t * s1;
}

// Intersection of the hex2d segment p0-p1 with the face edge in \`quadrant\` of an icosahedron face
// at the given adjusted (Class II) resolution.
fn cellIndexH3BoundaryIntersectFaceEdge(
  p0: vec2f,
  p1: vec2f,
  adjustedResolution: u32,
  quadrant: u32
) -> vec2f {
  let maximumDimension = f32(dggs_h3_get_max_dim_by_cii_resolution(adjustedResolution));
  let v0 = vec2f(3.0 * maximumDimension, 0.0);
  let v1 = vec2f(-1.5 * maximumDimension, 3.0 * DGGS_H3_SQRT3_2 * maximumDimension);
  let v2 = vec2f(-1.5 * maximumDimension, -3.0 * DGGS_H3_SQRT3_2 * maximumDimension);
  if (quadrant == 1u) {
    return cellIndexH3BoundaryIntersect(p0, p1, v0, v1);
  }
  if (quadrant == 3u) {
    return cellIndexH3BoundaryIntersect(p0, p1, v1, v2);
  }
  return cellIndexH3BoundaryIntersect(p0, p1, v2, v0);
}

// Substrate hex2d point on \`face\` to a unit vector (H3 _hex2dToGeo with substrate = 1). Projects
// through the face's unit-vector basis instead of dggs_h3_hex2d_to_lnglat, whose sin/cos/atan2
// azimuth-distance chain loses about 1e-4 degrees in f32.
fn cellIndexH3BoundaryHex2dToUnitVector(point: vec2f, face: u32, adjustedResolution: u32) -> vec3f {
  var scale = DGGS_H3_RES0_U_GNOMONIC * DGGS_H3_ONETHIRD;
  for (var level = 0u; level < adjustedResolution; level++) {
    scale *= DGGS_H3_RSQRT7;
  }
  // \`adjustedResolution\` is always Class II here, so no extra aperture-7 factor applies.
  return normalize(
    dggs_h3_get_face_unit_vector_basis(face, 0u) +
    scale * (
      point.x * dggs_h3_get_face_unit_vector_basis(face, 1u) +
      point.y * dggs_h3_get_face_unit_vector_basis(face, 2u)
    )
  );
}

// atan2(y, -0.0) returns -sign(y) * pi / 2 on the test GPU (FX-17), so the longitude uses
// guardedAtan2; the second atan2 below has a nonnegative length as its x.
fn cellIndexH3BoundaryUnitVectorToLngLat(unitVector: vec3f) -> vec2f {
  return vec2f(
    guardedAtan2(unitVector.y, unitVector.x) * DGGS_RADIANS_TO_DEGREES,
    atan2(unitVector.z, length(unitVector.xy)) * DGGS_RADIANS_TO_DEGREES
  );
}

`}var S=`GPUCellGeometry`,C={quadbin:4,h3:6,quadkey:4,geohash:4,s2:4,a5:5},w=16,T=4294967295,E=256,D=16777216,O=class{id;props;maximumVertexCount;edgeModel;outputSpatialContext;estimatedCommandNodeCount;chunkPlan;boundaryChunks;constructor(t){this.id=t.id??`cell-geometry`,this.props=t;let n=this.id;p(n,Object.prototype.hasOwnProperty.call(C,t.family),`family must be one of ${Object.keys(C).join(`, `)}`),this.edgeModel=t.family===`h3`||t.family===`s2`||t.family===`a5`?`great-circle`:`linear`,this.outputSpatialContext={...e,edgeModel:this.edgeModel},p(n,t.wordOrder===void 0||[`little-endian`,`high-low`].includes(t.wordOrder),`wordOrder must be 'little-endian' or 'high-low'`);let r=C[t.family];this.maximumVertexCount=t.maximumVertexCount??r,h(n,`maximumVertexCount`,this.maximumVertexCount,r,w);let{output:i}=t;p(n,!i.vertexCounts||!!i.boundaries,`output.vertexCounts needs output.boundaries`),p(n,!!(i.centers||i.boundaries),`output needs centers or boundaries`),this.chunkPlan=u(n,{cells:t.cells,centers:i.centers,vertexCounts:i.vertexCounts,validity:i.validity}),h(n,`cells row count`,this.chunkPlan.totalLength);for(let e of this.chunkPlan.chunks){d(e.views.cells,[`uint32x2`],`${n} cells chunk ${e.chunkIndex}`);for(let[t,r]of[[`centers`,`float32x2`],[`vertexCounts`,`uint32`],[`validity`,`uint32`]]){let i=e.views[t];i&&d(i,[r],`${n} output.${t} chunk ${e.chunkIndex}`)}f(n,`cells chunk ${e.chunkIndex} components`,e.length*2,T)}if(this.boundaryChunks=i.boundaries?m(i.boundaries):[],i.boundaries){p(n,i.boundaries.length===this.chunkPlan.totalLength*this.maximumVertexCount,`output.boundaries length must equal cells rows * maximumVertexCount`),p(n,this.boundaryChunks.length===this.chunkPlan.chunks.length,`output.boundaries must match cells chunk topology`);for(let e of this.chunkPlan.chunks){let t=this.boundaryChunks[e.chunkIndex];d(t,[`float32x2`],`${n} output.boundaries chunk ${e.chunkIndex}`),p(n,t.length===e.length*this.maximumVertexCount,`output.boundaries chunk ${e.chunkIndex} length must equal cells chunk rows * maximumVertexCount (${e.length*this.maximumVertexCount})`),f(n,`output.boundaries chunk ${e.chunkIndex} components`,t.length*2,T)}}s(n,i),this.estimatedCommandNodeCount=1+this.chunkPlan.nonemptyChunkCount,g(n,this.estimatedCommandNodeCount),_(n,[i.centers,i.boundaries,i.vertexCounts,i.validity],[t.cells])}getCommandNodes(e){let{id:s,props:u,maximumVertexCount:d}=this,{family:p,output:m}=u;f(s,`storage bindings`,2+Object.values(m).filter(Boolean).length,e.device.limits.maxStorageBuffersPerShaderStage),l(e,s,this.chunkPlan),o(e,s,{boundaries:m.boundaries});let h=this.chunkPlan.chunks.map(t=>n(s,Math.max(t.length,1),E,e.device.limits.maxComputeWorkgroupsPerDimension)),{slots:g,nodes:_}=c(e,{id:`${s}-chunks`,plan:this.chunkPlan}),v,y=this.chunkPlan.chunks.filter(e=>e.length>0).map(e=>{let n=e.views,o=[{name:`chunkParameters`,view:g[e.chunkIndex],type:`u32`,access:`read`},{name:`cells`,view:n.cells,type:`u32`,access:`read`}];m.centers&&o.push({name:`centersOut`,view:n.centers,type:`f32`,access:`read_write`}),m.boundaries&&o.push({name:`boundariesOut`,view:this.boundaryChunks[e.chunkIndex],type:`f32`,access:`read_write`}),m.vertexCounts&&o.push({name:`vertexCountsOut`,view:n.vertexCounts,type:`u32`,access:`read_write`}),m.validity&&o.push({name:`validityOut`,view:n.validity,type:`u32`,access:`read_write`});let c=m.centers?`let center = select(vec2f(getNotANumber(index)), geometryGetCenter(key), isValid);
  centersOut[centersOutOffset + 2u * index] = center.x;
  centersOut[centersOutOffset + 2u * index + 1u] = center.y;`:``,l=m.boundaries?`var points: array<vec2f, 16>;
  var vertexCount = 0u;
  if (isValid) {
    vertexCount = geometryFillBoundary(key, &points);
  }
  vertexCount = min(vertexCount, MAXIMUM_VERTEX_COUNT);
  for (var vertex = 0u; vertex < MAXIMUM_VERTEX_COUNT; vertex++) {
    let point = points[vertex];
    let slot = (index * MAXIMUM_VERTEX_COUNT + vertex) * 2u;
    boundariesOut[boundariesOutOffset + slot] = point.x;
    boundariesOut[boundariesOutOffset + slot + 1u] = point.y;
  }
  ${m.vertexCounts?`vertexCountsOut[vertexCountsOutOffset + index] = vertexCount;`:``}`:``,f=o.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`),_=v??`${f}
const MAXIMUM_VERTEX_COUNT: u32 = ${d}u;
fn getNotANumber(runtimeZero: u32) -> f32 {
  return bitcast<f32>(0x7fc00000u | (runtimeZero & 0u));
}
${a.source}

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

${k}
${I[p]}
${p===`h3`?``:F}
@compute @workgroup_size(${E})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${D}u) { return; }
  let index = workgroupIndex * ${E}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  let cellsOffset = arrayLength(&cells) - rowCount * 2u;
  ${m.centers?`let centersOutOffset = arrayLength(&centersOut) - rowCount * 2u;`:``}
  ${m.boundaries?`let boundariesOutOffset = arrayLength(&boundariesOut) - rowCount * MAXIMUM_VERTEX_COUNT * 2u;`:``}
  ${m.vertexCounts?`let vertexCountsOutOffset = arrayLength(&vertexCountsOut) - rowCount;`:``}
  ${m.validity?`let validityOutOffset = arrayLength(&validityOut) - rowCount;`:``}
  let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${u.wordOrder===`high-low`?`words`:`words.yx`};
  let isValid = geometryIsValid(key);
  ${m.validity?`validityOut[validityOutOffset + index] = select(0u, 1u, isValid);`:``}
  ${c}
  ${l}
}`;v=_;let y=h[e.chunkIndex];return t({id:`${s}-geometry-${e.chunkIndex}`,resources:o.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:S,variant:p,commandCount:1,maximumWorkgroupCount:y.x*y.y*y.z,maximumInvocationCount:e.length},compile:({device:e})=>{let t=new i(e,{id:`${S}-${p}-geometry`,source:_,shaderLayout:{bindings:o.map((e,t)=>({name:e.name,group:0,location:t,type:e.access===`read`?`read-only-storage`:`storage`}))}});return{encode:({computePass:e,getBuffer:n})=>t.dispatch(e,{bindings:Object.fromEntries(o.map(e=>[e.name,r(e.view,n)])),...y}),destroy:()=>t.destroy()}}})});return[..._,...y]}},k=`
fn geometryGetTileBounds(tile: vec3u) -> vec4f {
  let tileScale = f32(1u << tile.z);
  let west = f32(tile.x) / tileScale * 360.0 - 180.0;
  let east = f32(tile.x + 1u) / tileScale * 360.0 - 180.0;
  let north = dggs_web_mercator_tile_y_to_latitude(f32(tile.y), tileScale);
  let south = dggs_web_mercator_tile_y_to_latitude(f32(tile.y + 1u), tileScale);
  return vec4f(west, south, east, north);
}

fn geometryGetTileCenter(tile: vec3u) -> vec2f {
  let tileScale = f32(1u << tile.z);
  return vec2f(
    (f32(tile.x) + 0.5) / tileScale * 360.0 - 180.0,
    dggs_web_mercator_tile_y_to_latitude(f32(tile.y) + 0.5, tileScale)
  );
}

fn geometryGetRectangleVertex(bounds: vec4f, vertex: u32) -> vec2f {
  return dggs_bounds_get_boundary_point(bounds, vertex);
}

/** Even bits of a 32-bit value packed into 16 bits (inverse of the Morton spread). */
fn geometryCompactEvenBits(value: u32) -> u32 {
  var x = value & 0x55555555u;
  x = (x | (x >> 1u)) & 0x33333333u;
  x = (x | (x >> 2u)) & 0x0f0f0f0fu;
  x = (x | (x >> 4u)) & 0x00ff00ffu;
  x = (x | (x >> 8u)) & 0x0000ffffu;
  return x;
}
`,A=`
fn geometryGetQuadbinTile(key: vec2u) -> vec3u {
  let resolution = (key.x >> 20u) & 0x1fu;
  let morton = cellGetCompactKey(key, 52u - 2u * resolution, 2u * resolution);
  let x = geometryCompactEvenBits(morton.y) | (geometryCompactEvenBits(morton.x) << 16u);
  let y = geometryCompactEvenBits(morton.y >> 1u) | (geometryCompactEvenBits(morton.x >> 1u) << 16u);
  return vec3u(x, y, resolution);
}
fn geometryIsValid(key: vec2u) -> bool { return cellIsValidQuadbin(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryGetTileCenter(geometryGetQuadbinTile(key)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryGetRectangleVertex(geometryGetTileBounds(geometryGetQuadbinTile(key)), vertex);
}
`,j=`
fn geometryIsValid(key: vec2u) -> bool {
  let length = key.x >> 26u;
  // Digits fill the low 2 * length bits; every bit between them and the length field is zero.
  let unused = cellMaskLow(58u) & ~cellMaskLow(2u * length);
  return length >= 1u && length <= DGGS_QUADKEY_MAX_LENGTH && all((key & unused) == vec2u(0u));
}
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryGetTileCenter(dggs_quadkey_get_tile(key)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryGetRectangleVertex(dggs_quadkey_get_bounds(key), vertex);
}
`,M=`
fn geometryIsValid(key: vec2u) -> bool {
  let length = key.x >> 28u;
  let unused = cellMaskLow(60u) & ~cellMaskLow(5u * length);
  return length >= 1u && length <= DGGS_GEOHASH_MAX_LENGTH && all((key & unused) == vec2u(0u));
}
fn geometryGetCenter(key: vec2u) -> vec2f {
  let bounds = dggs_geohash_get_bounds(key);
  return vec2f(0.5 * (bounds.x + bounds.z), 0.5 * (bounds.y + bounds.w));
}
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return dggs_geohash_get_boundary_point(key, vertex);
}
`,N=`
// dggs_s2_xyz_to_lnglat calls atan2 with a zero argument at the S2 face centers: atan2(0, 0) at the
// poles (faces 2 and 5) is undefined in WGSL, and on Metal atan2(1, 0) returned -pi/2 for the face
// 1 center (x = -u = -0). Exact zeros are therefore handled without atan2. Upstream
// dggs_s2_xyz_to_lnglat (a plain atan2) is not used: it changes the values on the axes.
fn geometryS2XyzToLngLat(xyz: vec3f) -> vec2f {
  let horizontal = length(xyz.xy);
  var longitude = 0.0;
  if (xyz.x == 0.0) {
    longitude = select(0.0, select(-90.0, 90.0, xyz.y > 0.0), xyz.y != 0.0);
  } else if (xyz.y == 0.0) {
    longitude = select(0.0, 180.0, xyz.x < 0.0);
  } else {
    longitude = atan2(xyz.y, xyz.x) * DGGS_RADIANS_TO_DEGREES;
  }
  return vec2f(longitude, atan2(xyz.z, horizontal) * DGGS_RADIANS_TO_DEGREES);
}
fn geometryS2GetPoint(key: vec2u, offset: vec2f) -> vec2f {
  let st = dggs_s2_ij_to_st(dggs_s2_get_ij(key), dggs_s2_get_level(key), offset);
  return geometryS2XyzToLngLat(dggs_s2_face_uv_to_xyz(dggs_s2_get_face(key), dggs_s2_st_to_uv(st)));
}
fn geometryIsValid(key: vec2u) -> bool { return dggs_s2_is_valid_cell_id(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryS2GetPoint(key, vec2f(0.5)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryS2GetPoint(key, dggs_s2_get_boundary_offset(vertex));
}
`,P=`
${y}
fn geometryIsValid(key: vec2u) -> bool { return dggs_h3_is_valid_cell_id(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return dggs_h3_get_center_lnglat(key); }
fn geometryFillBoundary(key: vec2u, points: ptr<function, array<vec2f, 16>>) -> u32 {
  let boundary = cellIndexH3GetBoundary(key);
  for (var vertex = 0u; vertex < boundary.count; vertex++) {
    (*points)[vertex] = boundary.points[vertex];
  }
  return boundary.count;
}
`,F=`
fn geometryFillBoundary(key: vec2u, points: ptr<function, array<vec2f, 16>>) -> u32 {
  let count = min(geometryGetVertexCount(key), 16u);
  for (var vertex = 0u; vertex < count; vertex++) {
    (*points)[vertex] = geometryGetVertex(key, vertex);
  }
  return count;
}
`,I={quadbin:A,quadkey:j,geohash:M,s2:N,h3:P,a5:`
fn geometryIsValid(key: vec2u) -> bool {
  return !dggs_u64_is_zero(key) && dggs_a5_deserialize(key).valid != 0u;
}
fn geometryGetCenter(key: vec2u) -> vec2f { return dggs_a5_get_center_lnglat(key); }
fn geometryGetVertexCount(key: vec2u) -> u32 {
  return dggs_a5_get_shape_vertex_count(dggs_a5_deserialize(key));
}
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return dggs_a5_get_boundary_point(key, vertex);
}
`};export{b as i,N as n,y as r,O as t};
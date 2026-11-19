import{r as e}from"./gpu-command-graph-BdzdmV6u.js";import{t}from"./gpu-dispatch-utils-CbRzFAgT.js";import{o as n,t as r}from"./kernel-C3tfQOn4.js";import{c as i,u as a}from"./model-CeCyfMN-.js";import{s as o}from"./cell-index-wgsl-Cw7syfdA.js";import{c as s,m as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as l,S as u,T as d,c as f,f as p,p as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{d as g,u as _}from"./spatial-context-DwZEd4EZ.js";import"./cell-keys-Dl1tIQ_Z.js";import{i as v,r as y}from"./gpu-cell-geometry-CDEk1ld7.js";import{t as b}from"./s2-cell-measures-wgsl-XtzhZtXJ.js";import{n as x,r as S,t as C}from"./s2-cell-measures-fp64-wgsl-QIyevJMB.js";import{n as w}from"./gpu-point-to-cell-CYNSElfx.js";import{a as T,i as E,n as D,o as O,r as k}from"./graph-chunk-dispatch-zBt8k2-o.js";var A=`

fn sphericalExcessTriangle(center: vec3f, first: vec3f, second: vec3f) -> f32 {
  let determinant = dot(center, cross(first, second));
  return 2.0 * atan2(determinant, 1.0 + dot(first, second) + dot(second, center) + dot(center, first));
}

const CELL_MEASURES_MAX_EDGES: u32 = 10u;

struct CellMeasures {
  count : u32,
  area : f32,
  perimeter : f32,
  edges : array<f32, 10>,
  isExact : u32,
};

// Hexagon center and per-vertex substrate offsets, or valid = 0 when the hexagon is not on one face.
struct CellMeasuresHexagon {
  valid : u32,
  scale : f32,
  center : vec2f,
  offsets : array<vec3i, 6>,
};

fn cellMeasuresGetHex2d(offset: vec3i) -> vec2f {
  let i = f32(offset.x - offset.z);
  let j = f32(offset.y - offset.z);
  return vec2f(i - 0.5 * j, j * DGGS_H3_SQRT3_2);
}

fn cellMeasuresGetHexagon(cell: vec2u, resolution: u32) -> CellMeasuresHexagon {
  var hexagon: CellMeasuresHexagon;
  hexagon.valid = 0u;
  let center = dggs_h3_get_center_face_ijk(cell);
  if (center.valid == 0u) {
    return hexagon;
  }
  var adjustedResolution = resolution;
  var centerCoord = dggs_h3_down_ap3r(dggs_h3_down_ap3(center.coord));
  if (dggs_h3_is_resolution_class_iii(resolution)) {
    centerCoord = dggs_h3_down_ap7r(centerCoord);
    adjustedResolution += 1u;
  }
  let maximumDimension = 3 * dggs_h3_get_max_dim_by_cii_resolution(adjustedResolution);
  for (var vertex = 0u; vertex < 6u; vertex++) {
    let offset = dggs_h3_get_vertex_offset(resolution, vertex);
    let coord = dggs_h3_ijk_normalize(centerCoord + offset);
    if (coord.x + coord.y + coord.z > maximumDimension) {
      return hexagon;
    }
    hexagon.offsets[vertex] = offset;
  }
  var scale = DGGS_H3_RES0_U_GNOMONIC * DGGS_H3_ONETHIRD;
  for (var level = 0u; level < adjustedResolution; level++) {
    scale *= DGGS_H3_RSQRT7;
  }
  hexagon.scale = scale;
  hexagon.center = scale * cellMeasuresGetHex2d(centerCoord);
  hexagon.valid = 1u;
  return hexagon;
}

// Length of the vector from the sphere center to a point of the gnomonic plane at distance 1.
fn cellMeasuresGetPlaneNorm(point: vec2f) -> f32 {
  return sqrt(1.0 + dot(point, point));
}

fn cellMeasuresGetExactHexagon(hexagon: CellMeasuresHexagon) -> CellMeasures {
  var measures: CellMeasures;
  measures.count = 6u;
  measures.isExact = 1u;
  let scale = hexagon.scale;
  let centerNorm = cellMeasuresGetPlaneNorm(hexagon.center);
  var points: array<vec2f, 6>;
  for (var vertex = 0u; vertex < 6u; vertex++) {
    points[vertex] = hexagon.center + scale * cellMeasuresGetHex2d(hexagon.offsets[vertex]);
  }
  var area = 0.0;
  var perimeter = 0.0;
  for (var vertex = 0u; vertex < 6u; vertex++) {
    let next = (vertex + 1u) % 6u;
    let offsetA = hexagon.offsets[vertex];
    let offsetB = hexagon.offsets[next];
    let pointA = points[vertex];
    let pointB = points[next];
    let latticeA = vec2f(f32(offsetA.x - offsetA.z), f32(offsetA.y - offsetA.z));
    let latticeB = vec2f(f32(offsetB.x - offsetB.z), f32(offsetB.y - offsetB.z));
    // Twice the planar area of the triangle (center, A, B): an exact lattice cross product.
    let determinant = abs(scale * scale * DGGS_H3_SQRT3_2 * (latticeA.x * latticeB.y - latticeA.y * latticeB.x));
    let normA = cellMeasuresGetPlaneNorm(pointA);
    let normB = cellMeasuresGetPlaneNorm(pointB);
    let dotAB = 1.0 + dot(pointA, pointB);
    let dotBC = 1.0 + dot(pointB, hexagon.center);
    let dotCA = 1.0 + dot(hexagon.center, pointA);
    let denominator = normA * normB * centerNorm + dotAB * centerNorm + dotBC * normA + dotCA * normB;
    area += 2.0 * atan2(determinant, denominator);

    // Great-circle arc A-B: |wA x wB| = sqrt(|d|^2 + (qA x d)^2) with d = qB - qA exact.
    let latticeDelta = latticeB - latticeA;
    let delta = scale * vec2f(latticeDelta.x - 0.5 * latticeDelta.y, latticeDelta.y * DGGS_H3_SQRT3_2);
    let crossProduct = pointA.x * delta.y - pointA.y * delta.x;
    let edge = atan2(sqrt(dot(delta, delta) + crossProduct * crossProduct), dotAB);
    measures.edges[vertex] = edge;
    perimeter += edge;
  }
  measures.area = area;
  measures.perimeter = perimeter;
  return measures;
}

// Unsigned spherical excess of the triangle (center, first, second) of unit vectors. The
// determinant dot(center, cross(first, second)) is taken as dot(center, cross(first - center,
// second - center)), which is the same number: the differences of nearby unit vectors are exact
// in f32, so the triple product no longer cancels three products of order 1 down to the triangle
// area. What remains is the rounding of the unit vectors themselves, about 6e-8 over the edge
// length in radians.
fn cellMeasuresGetTriangleExcess(center: vec3f, first: vec3f, second: vec3f) -> f32 {
  let determinant = abs(dot(center, cross(first - center, second - center)));
  return 2.0 * atan2(determinant, 1.0 + dot(first, second) + dot(second, center) + dot(center, first));
}

fn cellMeasuresGetBoundaryMeasures(cell: vec2u) -> CellMeasures {
  var measures: CellMeasures;
  measures.isExact = 0u;
  let boundary = cellIndexH3GetBoundary(cell);
  measures.count = boundary.count;
  if (boundary.count < 3u) {
    return measures;
  }
  let center = dggs_h3_get_center_unit_vector(cell);
  var area = 0.0;
  var perimeter = 0.0;
  for (var vertex = 0u; vertex < boundary.count; vertex++) {
    let a = boundary.units[vertex];
    let b = boundary.units[(vertex + 1u) % boundary.count];
    area += cellMeasuresGetTriangleExcess(center, a, b);
    let edge = atan2(length(cross(a, b)), dot(a, b));
    measures.edges[vertex] = edge;
    perimeter += edge;
  }
  measures.area = area;
  measures.perimeter = perimeter;
  return measures;
}

fn cellMeasuresIsPentagon(cell: vec2u) -> bool {
  return dggs_h3_is_valid_cell_id(cell) &&
    dggs_h3_is_base_cell_pentagon(dggs_h3_get_base_cell(cell)) &&
    dggs_h3_get_leading_non_zero_digit(cell) == 0u;
}

fn cellMeasuresGetH3(cell: vec2u) -> CellMeasures {
  var measures: CellMeasures;
  if (!dggs_h3_is_valid_cell_id(cell)) {
    return measures;
  }
  if (!cellMeasuresIsPentagon(cell)) {
    let hexagon = cellMeasuresGetHexagon(cell, dggs_h3_get_resolution(cell));
    if (hexagon.valid != 0u) {
      return cellMeasuresGetExactHexagon(hexagon);
    }
  }
  return cellMeasuresGetBoundaryMeasures(cell);
}
`,j=`
struct QuadbinCellMeasures {
  area: f32,
  perimeter: f32,
  edges: vec4f,
};

fn cellMeasuresGetQuadbin(cell: vec2u) -> QuadbinCellMeasures {
  let resolution = (cell.x >> 20u) & 31u;
  let morton = cellGetCompactKey(cell, 52u - 2u * resolution, 2u * resolution);
  var row = 0u;
  for (var bit = 0u; bit < resolution; bit++) {
    let digit = cellShiftRight(morton, 2u * bit + 1u).y & 1u;
    row = row | (digit << bit);
  }
  let step = 6.283185307179586 / f32(1u << resolution);
  let north = 3.141592653589793 - f32(row) * step;
  let south = north - step;
  let middle = north - 0.5 * step;
  // Half-angle and hyperbolic difference identities retain sub-ulp tile widths at zoom 26.
  // Builtin sinh loses its small argument to exp(x)-exp(-x) on some devices; use its series.
  var halfSine = 0.5 * step;
  var sine = step;
  if (step < 0.01) {
    halfSine *= 1.0 + step * step / 24.0;
    sine *= 1.0 + step * step / 6.0;
  } else {
    halfSine = sinh(0.5 * step);
    sine = sinh(step);
  }
  let northCosine = cosh(north);
  let southCosine = cosh(south);
  let meridian = 2.0 * atan2(halfSine, cosh(middle));
  let edges = vec4f(step / northCosine, meridian, step / southCosine, meridian);
  return QuadbinCellMeasures(step * sine / (northCosine * southCosine), edges.x + edges.y + edges.z + edges.w, edges);
}
`,M=`
const CELL_MEASURES_H3_FACE_BASES = array<vec2f, 180>(${w().flat().map(S).join(`,
`)});
const CELL_MEASURES_H3_SUBSTRATE_SCALES = array<vec2f, 9>(${Array.from({length:9},(e,t)=>S(.381966011250105/(3*7**t))).join(`,
`)});
const CELL_MEASURES_SQRT3_2 = ${S(Math.sqrt(3)/2)};

fn cellMeasuresH3CoordinateToPlane64(coordinates: vec3i) -> CellMeasuresPlane64 {
  let first = cellMeasuresInteger64(coordinates.x - coordinates.z);
  let second = cellMeasuresInteger64(coordinates.y - coordinates.z);
  return CellMeasuresPlane64(sub_fp64(first, mul_fp64(vec2f(0.5, 0.0), second)), mul_fp64(second, CELL_MEASURES_SQRT3_2));
}
fn cellIndexH3BoundaryIntersect(first: CellMeasuresPlane64, second: CellMeasuresPlane64, edgeFirst: CellMeasuresPlane64, edgeSecond: CellMeasuresPlane64) -> CellMeasuresPlane64 {
  let horizontal = sub_fp64(second.x, first.x);
  let vertical = sub_fp64(second.y, first.y);
  let edgeHorizontal = sub_fp64(edgeSecond.x, edgeFirst.x);
  let edgeVertical = sub_fp64(edgeSecond.y, edgeFirst.y);
  let numerator = sub_fp64(mul_fp64(edgeHorizontal, sub_fp64(first.y, edgeFirst.y)), mul_fp64(edgeVertical, sub_fp64(first.x, edgeFirst.x)));
  let denominator = sub_fp64(mul_fp64(horizontal, edgeVertical), mul_fp64(edgeHorizontal, vertical));
  let fraction = div_fp64(numerator, denominator);
  return CellMeasuresPlane64(sum_fp64(first.x, mul_fp64(fraction, horizontal)), sum_fp64(first.y, mul_fp64(fraction, vertical)));
}
fn cellIndexH3BoundaryIntersectFaceEdge(first: CellMeasuresPlane64, second: CellMeasuresPlane64, adjustedResolution: u32, quadrant: u32) -> CellMeasuresPlane64 {
  let dimension = cellMeasuresInteger64(dggs_h3_get_max_dim_by_cii_resolution(adjustedResolution));
  let horizontal = mul_fp64(vec2f(3.0, 0.0), dimension);
  let vertical = mul_fp64(horizontal, CELL_MEASURES_SQRT3_2);
  let firstCorner = CellMeasuresPlane64(horizontal, vec2f(0.0));
  let secondCorner = CellMeasuresPlane64(mul_fp64(vec2f(-0.5, 0.0), horizontal), vertical);
  let thirdCorner = CellMeasuresPlane64(secondCorner.x, -vertical);
  if (quadrant == 1u) { return cellIndexH3BoundaryIntersect(first, second, firstCorner, secondCorner); }
  if (quadrant == 3u) { return cellIndexH3BoundaryIntersect(first, second, secondCorner, thirdCorner); }
  return cellIndexH3BoundaryIntersect(first, second, thirdCorner, firstCorner);
}
fn cellMeasuresH3FaceBasis64(face: u32, basis: u32) -> CellMeasuresVector64 {
  let offset = face * 9u + basis * 3u;
  return CellMeasuresVector64(CELL_MEASURES_H3_FACE_BASES[offset], CELL_MEASURES_H3_FACE_BASES[offset + 1u], CELL_MEASURES_H3_FACE_BASES[offset + 2u]);
}
fn cellIndexH3BoundaryHex2dToUnitVector(point: CellMeasuresPlane64, face: u32, adjustedResolution: u32) -> CellMeasuresVector64 {
  let scale = CELL_MEASURES_H3_SUBSTRATE_SCALES[adjustedResolution / 2u];
  let horizontal = cellMeasuresVectorScale64(cellMeasuresH3FaceBasis64(face, 1u), mul_fp64(point.x, scale));
  let vertical = cellMeasuresVectorScale64(cellMeasuresH3FaceBasis64(face, 2u), mul_fp64(point.y, scale));
  return cellMeasuresVectorNormalize64(cellMeasuresVectorAdd64(cellMeasuresH3FaceBasis64(face, 0u), cellMeasuresVectorAdd64(horizontal, vertical)));
}
// Coordinates are optional products of the shared boundary walk; measures retain both words.
fn cellIndexH3BoundaryUnitVectorToLngLat(value: CellMeasuresVector64) -> vec2f {
  let unitVector = cellMeasuresVectorHigh64(value);
  return vec2f(guardedAtan2(unitVector.y, unitVector.x), atan2(unitVector.z, length(unitVector.xy))) * DGGS_RADIANS_TO_DEGREES;
}
`,N=`
${v({unitType:`CellMeasuresVector64`,coordinateToPlane:`cellMeasuresH3CoordinateToPlane64`,source:M})}

struct CellMeasures {
  count: u32,
  area: f32,
  perimeter: f32,
  edges: array<f32, 10>,
};
fn cellMeasuresIsPentagon(cell: vec2u) -> bool {
  return dggs_h3_is_valid_cell_id(cell) && dggs_h3_is_base_cell_pentagon(dggs_h3_get_base_cell(cell)) && dggs_h3_get_leading_non_zero_digit(cell) == 0u;
}
fn cellMeasuresGetH3(cell: vec2u) -> CellMeasures {
  var measures: CellMeasures;
  let boundary = cellIndexH3GetBoundary(cell);
  measures.count = boundary.count;
  if (boundary.count < 3u) { return measures; }
  let anchor = boundary.units[0];
  for (var vertex = 0u; vertex < boundary.count; vertex++) {
    let first = boundary.units[vertex];
    let second = boundary.units[(vertex + 1u) % boundary.count];
    let difference = cellMeasuresVectorSubtract64(second, first);
    let crossProduct = cellMeasuresVectorCross64(first, difference);
    let edge = atan2(cellMeasuresVectorLength64(crossProduct).x, cellMeasuresVectorDot64(first, second).x);
    measures.edges[vertex] = edge;
    measures.perimeter += edge;
    // A fan anchored on an actual boundary vertex needs no separately projected center.
    // Face-distortion vertices can make the boundary concave: retain signed triangles so
    // pieces outside the polygon cancel rather than adding their area twice.
    if (vertex > 0u && vertex + 1u < boundary.count) {
      let firstOffset = cellMeasuresVectorSubtract64(first, anchor);
      let secondOffset = cellMeasuresVectorSubtract64(second, anchor);
      let determinant = cellMeasuresVectorDot64(anchor, cellMeasuresVectorCross64(firstOffset, secondOffset));
      let denominator = sum_fp64(vec2f(1.0, 0.0), sum_fp64(cellMeasuresVectorDot64(anchor, first), sum_fp64(cellMeasuresVectorDot64(first, second), cellMeasuresVectorDot64(second, anchor))));
      measures.area += 2.0 * atan2(determinant.x, denominator.x);
    }
  }
  measures.area = abs(measures.area);
  return measures;
}
`,P=`
struct QuadbinCellMeasures { area: f32, perimeter: f32, edges: vec4f };
const CELL_MEASURES_PI = ${S(Math.PI)};
const CELL_MEASURES_TWO_PI = ${S(2*Math.PI)};

fn cellMeasuresCosh64(value: vec2f) -> vec2f {
  return mul_fp64(vec2f(0.5, 0.0), sum_fp64(exp_fp64(value), exp_fp64(-value)));
}
fn cellMeasuresSinh64(value: vec2f) -> vec2f {
  // Retain tiny zoom-26 widths without differencing exponentials close to one.
  if (value.x < 0.01) {
    let square = mul_fp64(value, value);
    let correction = sum_fp64(${S(1/6)}, mul_fp64(square, ${S(1/120)}));
    return mul_fp64(value, sum_fp64(vec2f(1.0, 0.0), mul_fp64(square, correction)));
  }
  return mul_fp64(vec2f(0.5, 0.0), sub_fp64(exp_fp64(value), exp_fp64(-value)));
}
fn cellMeasuresGetQuadbin(cell: vec2u) -> QuadbinCellMeasures {
  let resolution = (cell.x >> 20u) & 31u;
  let morton = cellGetCompactKey(cell, 52u - 2u * resolution, 2u * resolution);
  var row = 0u;
  for (var bit = 0u; bit < resolution; bit++) {
    row = row | ((cellShiftRight(morton, 2u * bit + 1u).y & 1u) << bit);
  }
  let step = mul_fp64(CELL_MEASURES_TWO_PI, vec2f(1.0 / f32(1u << resolution), 0.0));
  let north = sub_fp64(CELL_MEASURES_PI, mul_fp64(cellMeasuresInteger64(i32(row)), step));
  let south = sub_fp64(north, step);
  let halfStep = mul_fp64(vec2f(0.5, 0.0), step);
  let middle = sub_fp64(north, halfStep);
  let northCosine = cellMeasuresCosh64(north);
  let southCosine = cellMeasuresCosh64(south);
  let meridian = 2.0 * atan2(cellMeasuresSinh64(halfStep).x, cellMeasuresCosh64(middle).x);
  let edges = vec4f(div_fp64(step, northCosine).x, meridian, div_fp64(step, southCosine).x, meridian);
  let area = div_fp64(mul_fp64(step, cellMeasuresSinh64(step)), mul_fp64(northCosine, southCosine));
  return QuadbinCellMeasures(area.x, edges.x + edges.y + edges.z + edges.w, edges);
}
`,F=`GPUCellMeasures`,I=4294967295,L=256,R=16777216,z=class{id;props;maximumEdgeCount;units;measures;estimatedCommandNodeCount;chunkPlan;edgeLengthChunks;constructor(e){this.id=e.id??`cell-measures`,this.props=e;let t=this.id;d(t,[`h3`,`s2`,`quadbin`].includes(e.family),`family must be 'h3', 's2' or 'quadbin'`),d(t,e.wordOrder===void 0||[`little-endian`,`high-low`].includes(e.wordOrder),`wordOrder must be 'little-endian' or 'high-low'`),d(t,e.precision===void 0||[`float32`,`fp64`].includes(e.precision),`precision must be 'float32' or 'fp64'`),this.units=e.units??`meters`,d(t,[`meters`,`kilometers`].includes(this.units),`units must be 'meters' or 'kilometers'`),this.measures={lengthModel:e.family===`quadbin`?`sphere-parallel-meridian`:`great-circle`,areaModel:e.family===`quadbin`?`sphere-rectangle`:`sphere-polygon`,lengthUnits:this.units,areaUnits:_(this.units)};let n=e.family===`h3`?6:4;this.maximumEdgeCount=e.maximumEdgeCount??n,p(t,`maximumEdgeCount`,this.maximumEdgeCount,n,10);let{output:r}=e,i=[[`output.areas`,r.areas],[`output.perimeters`,r.perimeters],[`output.edgeLengths`,r.edgeLengths],[`output.edgeCounts`,r.edgeCounts],[`output.pentagons`,r.pentagons]];d(t,i.some(([,e])=>e),`output needs at least one output`),this.chunkPlan=k(t,{cells:e.cells,areas:r.areas,perimeters:r.perimeters,edgeCounts:r.edgeCounts,pentagons:r.pentagons}),p(t,`cells row count`,this.chunkPlan.totalLength);for(let e of this.chunkPlan.chunks){l(e.views.cells,[`uint32x2`],`${t} cells chunk ${e.chunkIndex}`),u(t,`cells chunk ${e.chunkIndex} components`,e.length*2,I);for(let[n,r]of[[`areas`,`float32`],[`perimeters`,`float32`],[`edgeCounts`,`uint32`],[`pentagons`,`uint32`]]){let i=e.views[n];i&&l(i,[r],`${t} output.${n} chunk ${e.chunkIndex}`)}}if(this.edgeLengthChunks=r.edgeLengths?f(r.edgeLengths):[],r.edgeLengths){d(t,r.edgeLengths.length===this.chunkPlan.totalLength*this.maximumEdgeCount,`output.edgeLengths length must equal cells rows * maximumEdgeCount`),d(t,this.edgeLengthChunks.length===this.chunkPlan.chunks.length,`output.edgeLengths must match cells chunk topology`);for(let e of this.chunkPlan.chunks){let n=this.edgeLengthChunks[e.chunkIndex];l(n,[`float32`],`${t} output.edgeLengths chunk ${e.chunkIndex}`),d(t,n.length===e.length*this.maximumEdgeCount,`output.edgeLengths chunk ${e.chunkIndex} length must equal cells chunk rows * maximumEdgeCount (${e.length*this.maximumEdgeCount})`),u(t,`output.edgeLengths chunk ${e.chunkIndex} components`,n.length,I)}}E(t,r),this.estimatedCommandNodeCount=1+this.chunkPlan.nonemptyChunkCount,m(t,this.estimatedCommandNodeCount),h(t,i.map(([,e])=>e),[e.cells])}getCommandNodes(l){let{id:d,props:f,maximumEdgeCount:p}=this,{output:m}=f;u(d,`storage bindings`,2+Object.values(m).filter(Boolean).length,l.device.limits.maxStorageBuffersPerShaderStage),O(l,d,this.chunkPlan),T(l,d,{edgeLengths:m.edgeLengths});let h=this.chunkPlan.chunks.map(e=>t(d,Math.max(e.length,1),L,l.device.limits.maxComputeWorkgroupsPerDimension)),_=!!(m.areas||m.perimeters||m.edgeLengths||m.edgeCounts),v=`let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${f.wordOrder===`high-low`?`words`:`words.yx`};
  ${m.pentagons?`pentagonsOut[pentagonsOutOffset + index] = select(0u, 1u, cellMeasuresIsPentagon(key));`:``}
  ${_?`let measures = cellMeasuresGetH3(key);
  ${m.areas?`areasOut[areasOutOffset + index] = measures.area * EARTH_RADIUS * EARTH_RADIUS;`:``}
  ${m.perimeters?`perimetersOut[perimetersOutOffset + index] = measures.perimeter * EARTH_RADIUS;`:``}
  ${m.edgeCounts?`edgeCountsOut[edgeCountsOutOffset + index] = measures.count;`:``}
  ${m.edgeLengths?`for (var edge = 0u; edge < MAXIMUM_EDGE_COUNT; edge++) {
    edgeLengthsOut[edgeLengthsOutOffset + index * MAXIMUM_EDGE_COUNT + edge] =
      select(0.0, measures.edges[edge] * EARTH_RADIUS, edge < measures.count);
  }`:``}`:``}`,S=`let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${f.wordOrder===`high-low`?`words`:`words.yx`};
  let valid = ${f.family===`s2`?`dggs_s2_is_valid_cell_id(key)`:`cellIsValidQuadbin(key)`};
  ${m.pentagons?`pentagonsOut[pentagonsOutOffset + index] = 0u;`:``}
  ${_?`var measures: ${f.family===`s2`?`S2CellMeasures`:`QuadbinCellMeasures`};
  if (valid) { measures = ${f.family===`s2`?`cellMeasuresGetS2`:`cellMeasuresGetQuadbin`}(key); }
  ${m.areas?`areasOut[areasOutOffset + index] = measures.area * EARTH_RADIUS * EARTH_RADIUS;`:``}
  ${m.perimeters?`perimetersOut[perimetersOutOffset + index] = measures.perimeter * EARTH_RADIUS;`:``}
  ${m.edgeCounts?`edgeCountsOut[edgeCountsOutOffset + index] = select(0u, 4u, valid);`:``}
  ${m.edgeLengths?`for (var edge = 0u; edge < MAXIMUM_EDGE_COUNT; edge++) {
    var edgeLength = 0.0;
    if (edge < 4u) { edgeLength = measures.edges[edge] * EARTH_RADIUS; }
    edgeLengthsOut[edgeLengthsOutOffset + index * MAXIMUM_EDGE_COUNT + edge] = edgeLength;
  }`:``}`:``}`,w=g({coordinateSpace:`longitude-latitude`,metric:`great-circle`,units:this.units}),E=f.precision===`fp64`,k=E?`${x}\n${f.family===`h3`?N:f.family===`s2`?C:P}`:f.family===`h3`?`${y}\n${A}`:f.family===`s2`?b:j,{slots:M,nodes:I}=D(l,{id:`${d}-chunks`,plan:this.chunkPlan}),z,B,V=this.chunkPlan.chunks.filter(e=>e.length>0).map(t=>{let u=t.views,g=[{name:`chunkParameters`,view:M[t.chunkIndex],type:`u32`,access:`read`},{name:`cells`,view:u.cells,type:`u32`,access:`read`}],_=(e,t,n)=>{t&&g.push({name:e,view:t,type:n,access:`read_write`})};_(`areasOut`,u.areas,`f32`),_(`perimetersOut`,u.perimeters,`f32`),_(`edgeLengthsOut`,this.edgeLengthChunks[t.chunkIndex],`f32`),_(`edgeCountsOut`,u.edgeCounts,`u32`),_(`pentagonsOut`,u.pentagons,`u32`);let y=g.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`),b=z??`${y}
const MAXIMUM_EDGE_COUNT: u32 = ${p}u;
const EARTH_RADIUS: f32 = ${s(w)};
${o.source}

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
@compute @workgroup_size(${L})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${R}u) { return; }
  let index = workgroupIndex * ${L}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  // Bound arrays expose prefix + logical components, including unaligned packed subviews.
  let cellsOffset = arrayLength(&cells) - rowCount * 2u;
  ${m.areas?`let areasOutOffset = arrayLength(&areasOut) - rowCount;`:``}
  ${m.perimeters?`let perimetersOutOffset = arrayLength(&perimetersOut) - rowCount;`:``}
  ${m.edgeLengths?`let edgeLengthsOutOffset = arrayLength(&edgeLengthsOut) - rowCount * MAXIMUM_EDGE_COUNT;`:``}
  ${m.edgeCounts?`let edgeCountsOutOffset = arrayLength(&edgeCountsOut) - rowCount;`:``}
  ${m.pentagons?`let pentagonsOutOffset = arrayLength(&pentagonsOut) - rowCount;`:``}
  ${f.family===`h3`?v:S}
}`;z=b;let x=B??(E?new a().assembleWGSLShader({platformInfo:i(l.device),source:b,modules:[c],defines:{LUMA_FP64_INTEGER_ARITHMETIC:!0},shaderStage:`compute`,computeEntryPoint:`main`,scanVertexAttributes:!1}).source:b);B=x;let C=h[t.chunkIndex];return e({id:`${d}-measures-${t.chunkIndex}`,resources:g.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:F,variant:`${f.family}${E?`-fp64`:``}`,commandCount:1,maximumWorkgroupCount:C.x*C.y*C.z,maximumInvocationCount:t.length},compile:({device:e})=>{let t=new r(e,{id:`${F}-${f.family}${E?`-fp64`:``}`,source:x,shaderLayout:{bindings:g.map((e,t)=>({name:e.name,group:0,location:t,type:e.access===`read`?`read-only-storage`:`storage`}))}});return{encode:({computePass:e,getBuffer:r})=>t.dispatch(e,{bindings:Object.fromEntries(g.map(e=>[e.name,n(e.view,r)])),...C}),destroy:()=>t.destroy()}}})});return[...I,...V]}};export{z as t};
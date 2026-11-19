import{t as e}from"./ports-xmdKgL07.js";import{t}from"./common-CshkX-d_.js";import{r as n,u as r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-scan-B4NNTfF9.js";import{t as a}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{c as o,o as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as c,S as l,f as u,u as d,w as f,x as ee}from"./gpu-contributor-utils-BIwLQON5.js";import{c as p,d as te,r as ne}from"./wgsl-common-DfyXiYB0.js";import{i as re,o as ie}from"./segment-table-BeVxjLqW.js";import{t as m}from"./gpu-segment-intersection-CGBfhMHr.js";import{t as ae}from"./bounds-reduction-6MyKJx-7.js";import{n as h}from"./topology-types-BLorEZd4.js";import{v as g}from"./spatial-context-DwZEd4EZ.js";var oe=2**22+1,se=`
fn latticeSignOf(value: i32) -> i32 {
  return select(select(0, -1, value < 0), 1, value > 0);
}

fn latticeBigSign(value: array<u32, 6>) -> i32 { return bitcast<i32>(value[0]); }

fn latticeBigZero() -> array<u32, 6> {
  return array<u32, 6>(0u, 0u, 0u, 0u, 0u, 0u);
}

// |value| < 2^31.
fn latticeBigFromI32(value: i32) -> array<u32, 6> {
  let magnitude = u32(abs(value));
  return array<u32, 6>(bitcast<u32>(latticeSignOf(value)), magnitude & 0xffffu, magnitude >> 16u, 0u, 0u, 0u);
}

fn latticeBigNegate(value: array<u32, 6>) -> array<u32, 6> {
  var result = value;
  result[0] = bitcast<u32>(-bitcast<i32>(value[0]));
  return result;
}

// Compares magnitudes (words 1 to 5): -1, 0 or 1.
fn latticeCompareMagnitude(first: array<u32, 6>, second: array<u32, 6>) -> i32 {
  // Parameters are copied to variables before indexing them with a loop index.
  var firstLimbs = first;
  var secondLimbs = second;
  for (var slot = 5u; slot > 0u; slot--) {
    if (firstLimbs[slot] != secondLimbs[slot]) { return select(-1, 1, firstLimbs[slot] > secondLimbs[slot]); }
  }
  return 0;
}

// Compares signed values: -1, 0 or 1.
fn latticeBigCompare(first: array<u32, 6>, second: array<u32, 6>) -> i32 {
  let firstSign = latticeBigSign(first);
  let secondSign = latticeBigSign(second);
  if (firstSign != secondSign) { return select(-1, 1, firstSign > secondSign); }
  if (firstSign == 0) { return 0; }
  let order = latticeCompareMagnitude(first, second);
  return select(-order, order, firstSign > 0);
}

fn latticeBigAdd(first: array<u32, 6>, second: array<u32, 6>) -> array<u32, 6> {
  let firstSign = latticeBigSign(first);
  let secondSign = latticeBigSign(second);
  if (firstSign == 0) { return second; }
  if (secondSign == 0) { return first; }
  var result = latticeBigZero();
  var firstLimbs = first;
  var secondLimbs = second;
  if (firstSign == secondSign) {
    var carry = 0u;
    for (var slot = 1u; slot < 6u; slot++) {
      let total = firstLimbs[slot] + secondLimbs[slot] + carry;
      result[slot] = total & 0xffffu;
      carry = total >> 16u;
    }
    result[0] = first[0];
    return result;
  }
  let order = latticeCompareMagnitude(first, second);
  if (order == 0) { return result; }
  var larger = firstLimbs;
  var smaller = secondLimbs;
  if (order < 0) {
    larger = secondLimbs;
    smaller = firstLimbs;
  }
  var borrow = 0u;
  for (var slot = 1u; slot < 6u; slot++) {
    let difference = larger[slot] + 0x10000u - smaller[slot] - borrow;
    result[slot] = difference & 0xffffu;
    borrow = 1u - (difference >> 16u);
  }
  result[0] = larger[0];
  return result;
}

// value * factor for |factor| < 2^31 (the product must stay below 2^80).
fn latticeBigMultiply(value: array<u32, 6>, factor: i32) -> array<u32, 6> {
  var result = latticeBigZero();
  if (latticeBigSign(value) == 0 || factor == 0) { return result; }
  let magnitude = u32(abs(factor));
  var factorLimbs = array<u32, 2>(magnitude & 0xffffu, magnitude >> 16u);
  var valueLimbs = value;
  for (var factorSlot = 0u; factorSlot < 2u; factorSlot++) {
    var carry = 0u;
    for (var slot = 1u; slot + factorSlot < 6u; slot++) {
      let total = result[slot + factorSlot] + valueLimbs[slot] * factorLimbs[factorSlot] + carry;
      result[slot + factorSlot] = total & 0xffffu;
      carry = total >> 16u;
    }
  }
  result[0] = bitcast<u32>(latticeBigSign(value) * latticeSignOf(factor));
  return result;
}

// Denominator D = cross(b - a, d - c) (words 0 to 5) and numerator N = cross(c - a, d - c) (words
// 6 to 11) of the crossing parameter of segments ab and cd. Coordinates are lattice integers
// (magnitude <= 2^22 + 1).
fn latticeCrossingTerms(a: vec2i, b: vec2i, c: vec2i, d: vec2i) -> array<u32, 12> {
  var left = array<i32, 4>(b.x - a.x, b.y - a.y, c.x - a.x, c.y - a.y);
  var right = array<i32, 4>(d.y - c.y, d.x - c.x, d.y - c.y, d.x - c.x);
  var denominator = latticeBigZero();
  var numerator = latticeBigZero();
  for (var term = 0u; term < 4u; term++) {
    var product = latticeBigMultiply(latticeBigFromI32(left[term]), right[term]);
    if ((term & 1u) == 1u) { product = latticeBigNegate(product); }
    if (term < 2u) { denominator = latticeBigAdd(denominator, product); } else { numerator = latticeBigAdd(numerator, product); }
  }
  var terms = array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  for (var slot = 0u; slot < 6u; slot++) {
    terms[slot] = denominator[slot];
    terms[slot + 6u] = numerator[slot];
  }
  return terms;
}

// The crossing point of ab and cd rounded half up to the lattice: per axis the largest integer k
// in the range of the segment with (2 (k - a) - 1) D <= 2 N (b - a), D > 0. Requires D != 0 and a
// crossing on ab (the exact point lies within the range of the segment).
fn latticeRoundedCrossing(a: vec2i, b: vec2i, terms: array<u32, 12>) -> vec2i {
  var denominator = latticeBigZero();
  var numerator = latticeBigZero();
  var termWords = terms;
  for (var slot = 0u; slot < 6u; slot++) {
    denominator[slot] = termWords[slot];
    numerator[slot] = termWords[slot + 6u];
  }
  if (latticeBigSign(denominator) < 0) {
    denominator = latticeBigNegate(denominator);
    numerator = latticeBigNegate(numerator);
  }
  var result = vec2i(0, 0);
  for (var axis = 0u; axis < 2u; axis++) {
    let origin = select(a.y, a.x, axis == 0u);
    let endCoordinate = select(b.y, b.x, axis == 0u);
    var low = min(origin, endCoordinate);
    var high = max(origin, endCoordinate);
    var right = latticeBigZero();
    var middle = low;
    // Step 0 forms 2 N (b - a); every later step tests one candidate. The at most 23 candidate
    // steps of a range below 2^23 fit the 24 steps.
    for (var step = 0u; step < 25u; step++) {
      var multiplicand = denominator;
      var factor = 0i;
      if (step == 0u) {
        multiplicand = numerator;
        factor = 2 * (endCoordinate - origin);
      } else {
        if (low >= high) { break; }
        middle = low + (high - low + 1) / 2;
        factor = 2 * (middle - origin) - 1;
      }
      let product = latticeBigMultiply(multiplicand, factor);
      if (step == 0u) {
        right = product;
      } else if (latticeBigCompare(product, right) <= 0) {
        low = middle;
      } else {
        high = middle - 1;
      }
    }
    if (axis == 0u) { result.x = low; } else { result.y = low; }
  }
  return result;
}

// Signs of cross(b - a, corner - a) for the four corners of the unit square around center, in the
// order bottom-left, top-left, top-right, bottom-right (coordinates doubled so corners are integers).
fn latticeCornerSigns(a: vec2i, b: vec2i, center: vec2i) -> vec4i {
  let edge = b - a;
  var signs = array<i32, 4>(0, 0, 0, 0);
  for (var corner = 0u; corner < 4u; corner++) {
    let cornerX = 2 * center.x + select(1, -1, corner < 2u);
    let cornerY = 2 * center.y + select(-1, 1, corner == 1u || corner == 2u);
    let relativeX = cornerX - 2 * a.x;
    let relativeY = cornerY - 2 * a.y;
    var difference = latticeBigZero();
    for (var term = 0u; term < 2u; term++) {
      var product = latticeBigMultiply(latticeBigFromI32(select(edge.x, edge.y, term == 1u)), select(relativeY, relativeX, term == 1u));
      if (term == 1u) { product = latticeBigNegate(product); }
      difference = latticeBigAdd(difference, product);
    }
    signs[corner] = latticeBigSign(difference);
  }
  return vec4i(signs[0], signs[1], signs[2], signs[3]);
}

// Whether segment ab meets the half-open pixel [center - 1/2, center + 1/2) in x and y, for a
// segment with lattice endpoints that meets the closed square. Only a segment that touches the
// closed square in exactly one top-left, top-right or bottom-right corner misses the half-open
// square (an endpoint is a lattice point, so the segment passes through such a corner and the
// square lies entirely on one side of its line).
fn latticeSegmentMeetsPixel(a: vec2i, b: vec2i, center: vec2i) -> bool {
  let signs = latticeCornerSigns(a, b, center);
  var values = array<i32, 4>(signs.x, signs.y, signs.z, signs.w);
  for (var corner = 1u; corner < 4u; corner++) {
    if (values[corner] != 0) { continue; }
    let cornerX = 2 * center.x + select(1, -1, corner < 2u);
    let cornerY = 2 * center.y + select(-1, 1, corner == 1u || corner == 2u);
    let onSegment = 2 * min(a.x, b.x) <= cornerX && cornerX <= 2 * max(a.x, b.x) &&
      2 * min(a.y, b.y) <= cornerY && cornerY <= 2 * max(a.y, b.y);
    if (!onSegment) { continue; }
    var hasPositive = false;
    var hasNegative = false;
    for (var other = 0u; other < 4u; other++) {
      if (other == corner) { continue; }
      hasPositive = hasPositive || values[other] > 0;
      hasNegative = hasNegative || values[other] < 0;
    }
    if (!(hasPositive && hasNegative)) { return false; }
  }
  return true;
}
`,_=`GPUSnapRoundedNoding`,v=23,y=2**v-1,b=8;function x(e){let n=Math.ceil(Math.log2(e));for(;2**n<e;)n++;for(;2**(n-1)>=e;)n--;return 2**t(n,-126,127)}function S(e,t){return d(e,128)-t}var C=class{id;props;spatialContext;gridCell;constructor(t){this.id=t.id??`snap-rounded-noding`,this.spatialContext=g(this.id,t.spatialContext),t={...t,lines:e(this.id,`lines`,t.lines)},this.props=t;let{id:n}=this,{lines:i,output:a}=t;if(i.kind!==`lines`)throw Error(`${n} lines must be linestring geometry`);if(c(i.positions,[`float32x2`],`${n} lines.positions`),r(i.lineOffsets,`${n} lines.lineOffsets`),i.lineOffsets.length<2||i.positions.length<1)throw Error(`${n} lines require at least one line and one vertex`);if(i.sourceIds&&r(i.sourceIds,`${n} lines.sourceIds`),!Number.isFinite(t.gridSize)||!(t.gridSize>0)||t.gridSize>2**127)throw Error(`${n} gridSize must be finite and positive`);this.gridCell=x(t.gridSize),u(n,`intersectionCapacity`,t.intersectionCapacity),t.leafCapacity!==void 0&&f(n,`leafCapacity`,t.leafCapacity);let o=a.endpoints.length;if(c(a.endpoints,[`float32x4`],`${n} output.endpoints`),c(a.latticeEndpoints,[`float32x4`],`${n} output.latticeEndpoints`),a.latticeEndpoints.length!==o)throw Error(`${n} output.latticeEndpoints length must equal output.endpoints length`);if(c(a.lattice,[`float32`],`${n} output.lattice`),a.lattice.length!==4)throw Error(`${n} output.lattice must contain four rows`);for(let[e,t]of[[`sourceFeatureIds`,a.sourceFeatureIds],[`sourceRingIds`,a.sourceRingIds],[`sourceEdgeIds`,a.sourceEdgeIds]])if(r(t,`${n} output.${e}`),t.length!==o)throw Error(`${n} output.${e} length must equal output.endpoints length`);for(let[e,t]of[[`sourceStartParameters`,a.sourceStartParameters],[`sourceEndParameters`,a.sourceEndParameters]])if(c(t,[`float32`],`${n} output.${e}`),t.length!==o)throw Error(`${n} output.${e} length must equal output.endpoints length`);for(let[e,t]of Object.entries(a.status))if(t&&(r(t,`${n} output.status.${e}`),t.length<1))throw Error(`${n} output.status.${e} must contain one row`);if(!a.status.requiredCount||!a.status.candidateOverflow)throw Error(`${n} output.status requires requiredCount and candidateOverflow`)}getCommandNodes(e){let{id:t,props:r}=this,{lines:c,output:u}=r;ee(t,e,[c.positions,c.lineOffsets,c.sourceIds,u.endpoints,u.latticeEndpoints,u.lattice,u.sourceFeatureIds,u.sourceRingIds,u.sourceEdgeIds,u.sourceStartParameters,u.sourceEndParameters,...Object.values(u.status)]);let d=c.positions.length,f=c.lineOffsets.length-1,g=u.endpoints.length,x=r.intersectionCapacity,S=d+x,C=Math.max(g+d,2*S),w=2*C;l(t,`lines.positions rows`,d,re(e.device)),l(t,`4 x (lines.positions rows + intersectionCapacity)`,4*S,re(e.device));let T=(r,i,a)=>n(e,`${t}-${r}`,i,a),E=[],ce={name:`positions`,view:c.positions,type:`f32`,access:`read`},le={name:`lineOffsets`,view:c.lineOffsets,type:`u32`,access:`read`},ue=`fn isLineRow(row: u32) -> bool {
  return row >= lineOffsets[lineOffsetsOffset] && row < min(lineOffsets[lineOffsetsOffset + ${f}u], ${d}u);
}`,D=T(`state`,`uint32`,4);E.push(s(e,{id:`${t}-clear-state`,operation:_,variant:`clear-state`,bindings:[{name:`state`,view:D,type:`u32`,access:`read_write`}],invocationCount:4,body:`state[stateOffset + index] = 0u;`}));let de=T(`bounds`,`float32`,4);E.push(...ae(e,{id:t,operation:_,variant:`bounds`,bindings:[ce,le],declarations:`${p}
${ue}
fn readItem(row: u32) -> vec4f {
  let point = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  let valid = isLineRow(row) && isFiniteBits(point.x) && isFiniteBits(point.y);
  return vec4f(point, select(0.0, 1.0, valid), 0.0);
}`,itemCount:d,output:de})),E.push(s(e,{id:`${t}-lattice`,operation:_,variant:`lattice`,bindings:[{name:`bounds`,view:de,type:`f32`,access:`read`},{name:`lattice`,view:u.lattice,type:`f32`,access:`read_write`},{name:`state`,view:D,type:`atomic<u32>`,access:`read_write`}],invocationCount:1,declarations:`${p}
const GRID_CELL: f32 = ${o(this.gridCell)};
// Smallest power of two at least value, for a finite value >= 0 (at least 2^-126).
fn ceilPowerOfTwo(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  let exponent = bits >> 23u;
  if (exponent == 0u) { return 1.17549435e-38; }
  let isExact = (bits & 0x7fffffu) == 0u;
  return bitcast<f32>(select(exponent + 1u, exponent, isExact) << 23u);
}`,body:`let minimum = vec2f(bounds[boundsOffset], bounds[boundsOffset + 1u]);
  let maximum = vec2f(bounds[boundsOffset + 2u], bounds[boundsOffset + 3u]);
  let extent = max(maximum.x - minimum.x, maximum.y - minimum.y);
  var usable = isFiniteBits(extent);
  let extentPower = ceilPowerOfTwo(select(0.0, extent, usable));
  usable = usable && isFiniteBits(extentPower);
  // The cell is also at least 2^-24 of the largest coordinate magnitude (one f32 step there, or
  // of the binade below when the magnitude is a power of two), so every lattice point of the data
  // is a multiple of the cell below 2^24 and therefore an f32 value. Sized from the extent alone,
  // a lattice far from zero is finer than f32: distinct lattice vertices were then written as one
  // world vertex and the output held zero-area rings.
  let magnitude = max(max(abs(minimum.x), abs(minimum.y)), max(abs(maximum.x), abs(maximum.y)));
  // Half the magnitude, so its power of two stays finite up to the largest f32.
  let halfMagnitudePower = ceilPowerOfTwo(select(0.0, 0.5 * magnitude, usable));
  // All operands are powers of two: so is the largest (the grid, 2^-22 of the extent, or 2^-24
  // of the magnitude, written as 2^-23 of its half).
  let cell = select(
    1.0,
    max(max(GRID_CELL, extentPower * 2.384185791015625e-7), halfMagnitudePower * 1.1920928955078125e-7),
    usable);
  let scaled = minimum / cell;
  usable = usable && isFiniteBits(scaled.x) && isFiniteBits(scaled.y);
  // The origin is a multiple of the cell, so the origin (and every lattice offset) is exact.
  let origin = select(vec2f(0.0), floor(scaled) * cell, usable);
  lattice[latticeOffset] = origin.x;
  lattice[latticeOffset + 1u] = origin.y;
  lattice[latticeOffset + 2u] = select(1.0, cell, usable);
  lattice[latticeOffset + 3u] = 0.0;
  if (!usable) { atomicOr(&state[stateOffset], 1u); }`}));let O=T(`lattice-positions`,`float32x2`,d);E.push(s(e,{id:`${t}-snap-vertices`,operation:_,variant:`snap-vertices`,bindings:[ce,le,{name:`lattice`,view:u.lattice,type:`f32`,access:`read`},{name:`latPos`,view:O,type:`f32`,access:`read_write`},{name:`state`,view:D,type:`atomic<u32>`,access:`read_write`}],invocationCount:d,declarations:`${p}
${ne}
${ue}
const MAXIMUM_COORDINATE: f32 = ${oe}.0;
// floor((value - origin) / cell + 1/2) without a rounding step: the scaled value, its floor and
// the origin multiple are exact, and only the fraction decides the half.
fn snapAxis(value: f32, originMultiple: f32, cell: f32) -> f32 {
  let scaled = value / cell;
  let base = floor(scaled);
  let fraction = scaled - base;
  return (base - originMultiple) + select(0.0, 1.0, fraction >= 0.5);
}`,body:`let cell = lattice[latticeOffset + 2u];
  let point = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  var snapped = vec2f(getNaN());
  // A padding row (outside every line) stays NaN: no segment uses it, and it is no invalid input.
  let lineRow = isLineRow(index);
  if (lineRow && isFiniteBits(point.x) && isFiniteBits(point.y)) {
    let x = snapAxis(point.x, lattice[latticeOffset] / cell, cell);
    let y = snapAxis(point.y, lattice[latticeOffset + 1u] / cell, cell);
    if (isFiniteBits(x) && isFiniteBits(y) && x >= 0.0 && y >= 0.0 && x <= MAXIMUM_COORDINATE && y <= MAXIMUM_COORDINATE) {
      snapped = vec2f(x, y);
    } else {
      atomicOr(&state[stateOffset], 1u);
    }
  } else if (lineRow) {
    atomicOr(&state[stateOffset], 1u);
  }
  latPos[latPosOffset + index * 2u] = snapped.x;
  latPos[latPosOffset + index * 2u + 1u] = snapped.y;`}));let k={kind:`lines`,positions:O,lineOffsets:c.lineOffsets},A=`const LINE_COUNT: u32 = ${f}u;
const VERTEX_COUNT: u32 = ${d}u;
const LINE_NONE: u32 = 0xffffffffu;
// The line holding the vertex, or LINE_NONE.
fn lineOfVertex(vertex: u32) -> u32 {
  if (lineOffsets[lineOffsetsOffset] > vertex) { return LINE_NONE; }
  var low = 0u;
  var high = LINE_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle; }
  }
  if (vertex >= min(lineOffsets[lineOffsetsOffset + low + 1u], VERTEX_COUNT)) { return LINE_NONE; }
  return low;
}`,j=`${te}
const LATTICE_VERTEX_COUNT: u32 = ${d}u;
// The vertex is clamped so that a read of the row after the last vertex stays in range even where a
// caller evaluates both sides of an && (the executed specs' interpreter does).
fn latticeFloat(vertex: u32) -> vec2f {
  let row = min(vertex, LATTICE_VERTEX_COUNT - 1u);
  return vec2f(latPos[latPosOffset + row * 2u], latPos[latPosOffset + row * 2u + 1u]);
}
fn latticePoint(vertex: u32) -> vec2i {
  let point = latticeFloat(vertex);
  return vec2i(i32(point.x), i32(point.y));
}
// Both ends finite and distinct: the segment starting at vertex survives snapping.
fn segmentUsable(vertex: u32) -> bool {
  let start = latticeFloat(vertex);
  let end = latticeFloat(vertex + 1u);
  return !isNanBits(start.x) && !isNanBits(start.y) && !isNanBits(end.x) && !isNanBits(end.y) &&
    (start.x != end.x || start.y != end.y);
}`,M=T(`keys-x`,`uint32`,S),N=T(`keys-y`,`uint32`,S);E.push(s(e,{id:`${t}-mark-vertices`,operation:_,variant:`mark-vertices`,bindings:[{name:`latPos`,view:O,type:`f32`,access:`read`},{name:`lineOffsets`,view:c.lineOffsets,type:`u32`,access:`read`},{name:`keysX`,view:M,type:`u32`,access:`read_write`},{name:`keysY`,view:N,type:`u32`,access:`read_write`},{name:`state`,view:D,type:`atomic<u32>`,access:`read_write`}],invocationCount:d,declarations:`${A}
${j}
const KEY_SENTINEL: u32 = ${y}u;`,body:`let line = lineOfVertex(index);
  var hot = false;
  if (line != LINE_NONE) {
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = min(lineOffsets[lineOffsetsOffset + line + 1u], VERTEX_COUNT);
    let hasNext = index + 1u < lineEnd && segmentUsable(index);
    let hasPrevious = index > lineStart && segmentUsable(index - 1u);
    hot = hasNext || hasPrevious;
    if (hasNext) { atomicAdd(&state[stateOffset + 1u], 1u); }
  }
  let point = latticePoint(index);
  keysX[keysXOffset + index] = select(KEY_SENTINEL, u32(point.x), hot);
  keysY[keysYOffset + index] = select(KEY_SENTINEL, u32(point.y), hot);`}));let P=T(`crossing-left`,`uint32`,x),F=T(`crossing-right`,`uint32`,x),I=T(`crossing-count`,`uint32`,1),L=T(`crossing-overflow`,`uint32`,1),R=T(`crossing-candidate-overflow`,`uint32`,1),fe=T(`crossing-required`,`uint32`,1),z=T(`crossing-kinds`,`uint32`,x),B=u.status.uncertainCount??T(`uncertain-count`,`uint32`,1);E.push(...new m({id:`${t}-crossings`,left:k,spatialSort:r.spatialSort,leafCapacity:r.leafCapacity,output:{pairs:{leftRows:P,rightRows:F,status:{count:I,overflow:L,candidateOverflow:R,requiredCount:fe}},kinds:z,status:{uncertainCount:B}}}).getCommandNodes(e));let V=T(`crossing-terms`,`uint32`,x*b),H=[{name:`pairLeft`,view:P,type:`u32`,access:`read`},{name:`pairRight`,view:F,type:`u32`,access:`read`},{name:`pairCount`,view:I,type:`u32`,access:`read`},{name:`kinds`,view:z,type:`u32`,access:`read`},{name:`latPos`,view:O,type:`f32`,access:`read`}],U=`${se}
${j.replace(/fn segmentUsable[\s\S]*$/,``)}
const KIND_PROPER: u32 = ${ie.proper}u;
fn isProperPair(row: u32) -> bool {
  return row < min(pairCount[pairCountOffset], ${x}u) && kinds[kindsOffset + row] == KIND_PROPER;
}`;E.push(s(e,{id:`${t}-crossing-terms`,operation:_,variant:`crossing-terms`,bindings:[...H,{name:`termWords`,view:V,type:`u32`,access:`read_write`}],invocationCount:x,declarations:U,body:`var found = array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  if (isProperPair(index)) {
    let left = pairLeft[pairLeftOffset + index];
    let right = pairRight[pairRightOffset + index];
    found = latticeCrossingTerms(latticePoint(left), latticePoint(left + 1u), latticePoint(right), latticePoint(right + 1u));
  }
  // Sign and three limbs (below 2^48) of the denominator, then of the numerator.
  let base = termWordsOffset + index * ${b}u;
  for (var slot = 0u; slot < 4u; slot++) {
    termWords[base + slot] = found[slot];
    termWords[base + 4u + slot] = found[6u + slot];
  }`}),s(e,{id:`${t}-round-crossings`,operation:_,variant:`round-crossings`,bindings:[...H,{name:`termWords`,view:V,type:`u32`,access:`read`},{name:`keysX`,view:M,type:`u32`,access:`read_write`},{name:`keysY`,view:N,type:`u32`,access:`read_write`}],invocationCount:x,declarations:`${U}
const KEY_SENTINEL: u32 = ${y}u;
fn readTerms(index: u32) -> array<u32, 12> {
  let base = termWordsOffset + index * ${b}u;
  var terms = array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  for (var slot = 0u; slot < 4u; slot++) {
    terms[slot] = termWords[base + slot];
    terms[6u + slot] = termWords[base + 4u + slot];
  }
  return terms;
}`,body:`var keyX = KEY_SENTINEL;
  var keyY = KEY_SENTINEL;
  if (isProperPair(index)) {
    let left = pairLeft[pairLeftOffset + index];
    let rounded = latticeRoundedCrossing(latticePoint(left), latticePoint(left + 1u), readTerms(index));
    keyX = u32(rounded.x);
    keyY = u32(rounded.y);
  }
  keysX[keysXOffset + ${d}u + index] = keyX;
  keysY[keysYOffset + ${d}u + index] = keyY;`}));let W=T(`row-ids`,`uint32`,S),pe=a(e,{id:`${t}-hot-sort`,algorithm:`radix`,operation:_,indices:W,indicesAreIdentity:!0,keys:[{view:M,keyBits:v},{view:N,keyBits:v}]}),me=pe.sortedIndices,G=T(`hot-flags`,`uint32`,S),K=T(`hot-scan`,`uint32`,S),q=T(`hot`,`uint32x2`,S);E.push(s(e,{id:`${t}-row-ids`,operation:_,variant:`row-ids`,bindings:[{name:`rowIds`,view:W,type:`u32`,access:`read_write`}],invocationCount:S,body:`rowIds[rowIdsOffset + index] = index;`}),...pe.nodes,s(e,{id:`${t}-unique-hot`,operation:_,variant:`unique-hot`,bindings:[{name:`hotOrder`,view:me,type:`u32`,access:`read`},{name:`keysX`,view:M,type:`u32`,access:`read`},{name:`keysY`,view:N,type:`u32`,access:`read`},{name:`hotFlags`,view:G,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`const KEY_SENTINEL: u32 = ${y}u;`,body:`let row = hotOrder[hotOrderOffset + index];
  var unique = keysX[keysXOffset + row] != KEY_SENTINEL;
  if (unique && index > 0u) {
    let previous = hotOrder[hotOrderOffset + index - 1u];
    unique = keysX[keysXOffset + previous] != keysX[keysXOffset + row] ||
      keysY[keysYOffset + previous] != keysY[keysYOffset + row];
  }
  hotFlags[hotFlagsOffset + index] = select(0u, 1u, unique);`}),...new i({id:`${t}-hot-scan`,input:G,output:K,mode:`inclusive`}).getCommandNodes(e),s(e,{id:`${t}-compact-hot`,operation:_,variant:`compact-hot`,bindings:[{name:`hotOrder`,view:me,type:`u32`,access:`read`},{name:`keysX`,view:M,type:`u32`,access:`read`},{name:`keysY`,view:N,type:`u32`,access:`read`},{name:`hotFlags`,view:G,type:`u32`,access:`read`},{name:`hotScan`,view:K,type:`u32`,access:`read`},{name:`hot`,view:q,type:`u32`,access:`read_write`}],invocationCount:S,body:`if (hotFlags[hotFlagsOffset + index] == 0u) { return; }
  let row = hotOrder[hotOrderOffset + index];
  let pixel = hotScan[hotScanOffset + index] - 1u;
  hot[hotOffset + pixel * 2u] = keysX[keysXOffset + row];
  hot[hotOffset + pixel * 2u + 1u] = keysY[keysYOffset + row];`}));let he=T(`diagonal-positions`,`float32x2`,4*S),ge=T(`diagonal-offsets`,`uint32`,2*S+1);E.push(s(e,{id:`${t}-build-diagonals`,operation:_,variant:`build-diagonals`,bindings:[{name:`hot`,view:q,type:`u32`,access:`read`},{name:`hotScan`,view:K,type:`u32`,access:`read`},{name:`diagonalPositions`,view:he,type:`f32`,access:`read_write`},{name:`diagonalOffsets`,view:ge,type:`u32`,access:`read_write`}],invocationCount:S,declarations:ne,body:`diagonalOffsets[diagonalOffsetsOffset + index * 2u] = index * 4u;
  diagonalOffsets[diagonalOffsetsOffset + index * 2u + 1u] = index * 4u + 2u;
  if (index + 1u == ${S}u) { diagonalOffsets[diagonalOffsetsOffset + index * 2u + 2u] = ${4*S}u; }
  let isPixel = index < hotScan[hotScanOffset + ${S-1}u];
  let nan = getNaN();
  let center = vec2f(f32(hot[hotOffset + index * 2u]), f32(hot[hotOffset + index * 2u + 1u]));
  // Diagonals bottom-left to top-right and top-left to bottom-right. Half-integers are exact.
  var corners = array<vec2f, 4>(
    center + vec2f(-0.5, -0.5), center + vec2f(0.5, 0.5),
    center + vec2f(-0.5, 0.5), center + vec2f(0.5, -0.5));
  for (var corner = 0u; corner < 4u; corner++) {
    let point = select(vec2f(nan), corners[corner], isPixel);
    diagonalPositions[diagonalPositionsOffset + (index * 4u + corner) * 2u] = point.x;
    diagonalPositions[diagonalPositionsOffset + (index * 4u + corner) * 2u + 1u] = point.y;
  }`}));let J=T(`candidate-left`,`uint32`,w),Y=T(`candidate-right`,`uint32`,w),_e=T(`candidate-count`,`uint32`,1),ve=T(`candidate-overflow`,`uint32`,1),ye=T(`incidence-candidate-overflow`,`uint32`,1),be=T(`candidate-required`,`uint32`,1);E.push(...new m({id:`${t}-incidence`,left:k,right:{kind:`lines`,positions:he,lineOffsets:ge},output:{pairs:{leftRows:J,rightRows:Y,status:{count:_e,overflow:ve,candidateOverflow:ye,requiredCount:be}}}}).getCommandNodes(e));let xe=T(`candidate-flags`,`uint32`,w),X=T(`candidate-scan`,`uint32`,w),Se=T(`incidence-segments`,`uint32`,C),Ce=T(`incidence-pixels`,`uint32`,C);E.push(s(e,{id:`${t}-filter-candidates`,operation:_,variant:`filter-candidates`,bindings:[{name:`pairLeft`,view:J,type:`u32`,access:`read`},{name:`pairRight`,view:Y,type:`u32`,access:`read`},{name:`pairCount`,view:_e,type:`u32`,access:`read`},{name:`latPos`,view:O,type:`f32`,access:`read`},{name:`hot`,view:q,type:`u32`,access:`read`},{name:`candidateFlags`,view:xe,type:`u32`,access:`read_write`}],invocationCount:w,declarations:`${se}
${j.replace(/fn segmentUsable[\s\S]*$/,``)}`,body:`var keep = false;
  if (index < min(pairCount[pairCountOffset], ${w}u)) {
    let segment = pairLeft[pairLeftOffset + index];
    let pixel = pairRight[pairRightOffset + index] >> 2u;
    var duplicate = false;
    if (index > 0u) {
      duplicate = pairLeft[pairLeftOffset + index - 1u] == segment &&
        (pairRight[pairRightOffset + index - 1u] >> 2u) == pixel;
    }
    if (!duplicate) {
      let center = vec2i(i32(hot[hotOffset + pixel * 2u]), i32(hot[hotOffset + pixel * 2u + 1u]));
      keep = latticeSegmentMeetsPixel(latticePoint(segment), latticePoint(segment + 1u), center);
    }
  }
  candidateFlags[candidateFlagsOffset + index] = select(0u, 1u, keep);`}),...new i({id:`${t}-candidate-scan`,input:xe,output:X,mode:`inclusive`}).getCommandNodes(e),s(e,{id:`${t}-scatter-incidences`,operation:_,variant:`scatter-incidences`,bindings:[{name:`pairLeft`,view:J,type:`u32`,access:`read`},{name:`pairRight`,view:Y,type:`u32`,access:`read`},{name:`candidateScan`,view:X,type:`u32`,access:`read`},{name:`incidenceSegments`,view:Se,type:`u32`,access:`read_write`},{name:`incidencePixels`,view:Ce,type:`u32`,access:`read_write`}],invocationCount:w,body:`let slot = candidateScan[candidateScanOffset + index];
  var before = 0u;
  if (index > 0u) { before = candidateScan[candidateScanOffset + index - 1u]; }
  if (slot == before || slot > ${C}u) { return; }
  incidenceSegments[incidenceSegmentsOffset + slot - 1u] = pairLeft[pairLeftOffset + index];
  incidencePixels[incidencePixelsOffset + slot - 1u] = pairRight[pairRightOffset + index] >> 2u;`}));let Z=T(`ordered-segments`,`uint32`,C),we=T(`ordered-pixels`,`uint32`,C),Te=`fn incidenceTotal() -> u32 { return min(candidateScan[candidateScanOffset + ${w-1}u], ${C}u); }`;E.push(s(e,{id:`${t}-order-incidences`,operation:_,variant:`order-incidences`,bindings:[{name:`candidateScan`,view:X,type:`u32`,access:`read`},{name:`incidenceSegments`,view:Se,type:`u32`,access:`read`},{name:`incidencePixels`,view:Ce,type:`u32`,access:`read`},{name:`hot`,view:q,type:`u32`,access:`read`},{name:`latPos`,view:O,type:`f32`,access:`read`},{name:`orderedSegments`,view:Z,type:`u32`,access:`read_write`},{name:`orderedPixels`,view:we,type:`u32`,access:`read_write`}],invocationCount:C,declarations:`${j.replace(/fn segmentUsable[\s\S]*$/,``)}
${Te}
// Key of an incidence row for the binary searches: the segment (mode 0, 1) or the pixel x (2, 3).
fn rowKey(row: u32, mode: u32) -> u32 {
  if (mode < 2u) { return incidenceSegments[incidenceSegmentsOffset + row]; }
  return hot[hotOffset + incidencePixels[incidencePixelsOffset + row] * 2u];
}
// First row in [first, last) whose key is at least searchKey (even mode) or above it (odd mode).
fn searchRows(first: u32, last: u32, mode: u32, searchKey: u32) -> u32 {
  var low = first;
  var high = last;
  while (low < high) {
    let middle = (low + high) / 2u;
    let key = rowKey(middle, mode);
    let before = select(key < searchKey, key <= searchKey, (mode & 1u) == 1u);
    if (before) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,body:`let total = incidenceTotal();
  if (index >= total) { return; }
  let segment = incidenceSegments[incidenceSegmentsOffset + index];
  let groupStart = searchRows(0u, total, 0u, segment);
  let groupEnd = searchRows(groupStart, total, 1u, segment);
  let x = rowKey(index, 2u);
  let runStart = searchRows(groupStart, groupEnd, 2u, x);
  let runEnd = searchRows(runStart, groupEnd, 3u, x);
  let start = latticePoint(segment);
  let end = latticePoint(segment + 1u);
  var before = groupEnd - runEnd;
  if (end.x >= start.x) { before = runStart - groupStart; }
  var offset = runEnd - 1u - index;
  if (end.y >= start.y) { offset = index - runStart; }
  let targetRow = groupStart + before + offset;
  orderedSegments[orderedSegmentsOffset + targetRow] = segment;
  orderedPixels[orderedPixelsOffset + targetRow] = incidencePixels[incidencePixelsOffset + index];`}));let Ee=T(`piece-flags`,`uint32`,C),De=T(`piece-scan`,`uint32`,C);E.push(s(e,{id:`${t}-flag-pieces`,operation:_,variant:`flag-pieces`,bindings:[{name:`candidateScan`,view:X,type:`u32`,access:`read`},{name:`orderedSegments`,view:Z,type:`u32`,access:`read`},{name:`pieceFlags`,view:Ee,type:`u32`,access:`read_write`}],invocationCount:C,declarations:Te,body:`let total = incidenceTotal();
  var isPiece = false;
  if (index + 1u < total) {
    isPiece = orderedSegments[orderedSegmentsOffset + index] == orderedSegments[orderedSegmentsOffset + index + 1u];
  }
  pieceFlags[pieceFlagsOffset + index] = select(0u, 1u, isPiece);`}),...new i({id:`${t}-piece-scan`,input:Ee,output:De,mode:`inclusive`}).getCommandNodes(e));let Q=`// Output row of piece i, or NONE when it is not a piece or beyond the capacity.
const OUTPUT_CAPACITY: u32 = ${g}u;
const ROW_NONE: u32 = 0xffffffffu;
fn pieceRow(index: u32) -> u32 {
  let scan = pieceScan[pieceScanOffset + index];
  var before = 0u;
  if (index > 0u) { before = pieceScan[pieceScanOffset + index - 1u]; }
  if (scan == before || scan > OUTPUT_CAPACITY) { return ROW_NONE; }
  return scan - 1u;
}`,$={name:`pieceScan`,view:De,type:`u32`,access:`read`},Oe=c.sourceIds?[{name:`inputSourceIds`,view:c.sourceIds,type:`u32`,access:`read`}]:[];return E.push(s(e,{id:`${t}-initialize`,operation:_,variant:`initialize`,bindings:[{name:`endpoints`,view:u.endpoints,type:`f32`,access:`read_write`},{name:`latticeEndpoints`,view:u.latticeEndpoints,type:`f32`,access:`read_write`},{name:`sourceFeatureIds`,view:u.sourceFeatureIds,type:`u32`,access:`read_write`},{name:`sourceRingIds`,view:u.sourceRingIds,type:`u32`,access:`read_write`},{name:`sourceEdgeIds`,view:u.sourceEdgeIds,type:`u32`,access:`read_write`},{name:`startParameters`,view:u.sourceStartParameters,type:`f32`,access:`read_write`},{name:`endParameters`,view:u.sourceEndParameters,type:`f32`,access:`read_write`}],invocationCount:g,body:`let endpoint = index * 4u;
  for (var component = 0u; component < 4u; component++) {
    endpoints[endpointsOffset + endpoint + component] = 0.0;
    latticeEndpoints[latticeEndpointsOffset + endpoint + component] = 0.0;
  }
  sourceFeatureIds[sourceFeatureIdsOffset + index] = ${h}u;
  sourceRingIds[sourceRingIdsOffset + index] = ${h}u;
  sourceEdgeIds[sourceEdgeIdsOffset + index] = ${h}u;
  startParameters[startParametersOffset + index] = 0.0;
  endParameters[endParametersOffset + index] = 0.0;`}),s(e,{id:`${t}-write-geometry`,operation:_,variant:`write-geometry`,bindings:[$,{name:`orderedPixels`,view:we,type:`u32`,access:`read`},{name:`hot`,view:q,type:`u32`,access:`read`},{name:`lattice`,view:u.lattice,type:`f32`,access:`read`},{name:`latticeEndpoints`,view:u.latticeEndpoints,type:`f32`,access:`read_write`},{name:`endpoints`,view:u.endpoints,type:`f32`,access:`read_write`}],invocationCount:C,declarations:Q,body:`let row = pieceRow(index);
  if (row == ROW_NONE) { return; }
  let first = orderedPixels[orderedPixelsOffset + index];
  let second = orderedPixels[orderedPixelsOffset + index + 1u];
  let firstCenter = vec2f(f32(hot[hotOffset + first * 2u]), f32(hot[hotOffset + first * 2u + 1u]));
  let secondCenter = vec2f(f32(hot[hotOffset + second * 2u]), f32(hot[hotOffset + second * 2u + 1u]));
  let origin = vec2f(lattice[latticeOffset], lattice[latticeOffset + 1u]);
  let cell = lattice[latticeOffset + 2u];
  // World position origin + k * cell: k * cell is exact (power-of-two scaling), the sum rounds once.
  let world = array<vec2f, 2>(origin + firstCenter * cell, origin + secondCenter * cell);
  let base = row * 4u;
  latticeEndpoints[latticeEndpointsOffset + base] = firstCenter.x;
  latticeEndpoints[latticeEndpointsOffset + base + 1u] = firstCenter.y;
  latticeEndpoints[latticeEndpointsOffset + base + 2u] = secondCenter.x;
  latticeEndpoints[latticeEndpointsOffset + base + 3u] = secondCenter.y;
  endpoints[endpointsOffset + base] = world[0].x;
  endpoints[endpointsOffset + base + 1u] = world[0].y;
  endpoints[endpointsOffset + base + 2u] = world[1].x;
  endpoints[endpointsOffset + base + 3u] = world[1].y;`}),s(e,{id:`${t}-write-provenance`,operation:_,variant:`write-provenance`,bindings:[$,{name:`orderedSegments`,view:Z,type:`u32`,access:`read`},{name:`lineOffsets`,view:c.lineOffsets,type:`u32`,access:`read`},...Oe,{name:`sourceFeatureIds`,view:u.sourceFeatureIds,type:`u32`,access:`read_write`},{name:`sourceRingIds`,view:u.sourceRingIds,type:`u32`,access:`read_write`},{name:`sourceEdgeIds`,view:u.sourceEdgeIds,type:`u32`,access:`read_write`}],invocationCount:C,declarations:`${Q}
${A}`,body:`let row = pieceRow(index);
  if (row == ROW_NONE) { return; }
  let segment = orderedSegments[orderedSegmentsOffset + index];
  let line = lineOfVertex(segment);
  sourceFeatureIds[sourceFeatureIdsOffset + row] = ${c.sourceIds?`inputSourceIds[inputSourceIdsOffset + line]`:`line`};
  sourceRingIds[sourceRingIdsOffset + row] = line;
  sourceEdgeIds[sourceEdgeIdsOffset + row] = segment;`}),s(e,{id:`${t}-write-parameters`,operation:_,variant:`write-parameters`,bindings:[$,{name:`latPos`,view:O,type:`f32`,access:`read`},{name:`latticeEndpoints`,view:u.latticeEndpoints,type:`f32`,access:`read`},{name:`sourceEdgeIds`,view:u.sourceEdgeIds,type:`u32`,access:`read`},{name:`startParameters`,view:u.sourceStartParameters,type:`f32`,access:`read_write`},{name:`endParameters`,view:u.sourceEndParameters,type:`f32`,access:`read_write`}],invocationCount:C,declarations:`${Q}
${j.replace(/fn segmentUsable[\s\S]*$/,``)}`,body:`let row = pieceRow(index);
  if (row == ROW_NONE) { return; }
  let segment = sourceEdgeIds[sourceEdgeIdsOffset + row];
  let start = latticeFloat(segment);
  let delta = latticeFloat(segment + 1u) - start;
  let lengthSquared = dot(delta, delta);
  let base = row * 4u;
  let first = vec2f(latticeEndpoints[latticeEndpointsOffset + base], latticeEndpoints[latticeEndpointsOffset + base + 1u]);
  let second = vec2f(latticeEndpoints[latticeEndpointsOffset + base + 2u], latticeEndpoints[latticeEndpointsOffset + base + 3u]);
  startParameters[startParametersOffset + row] = clamp(dot(first - start, delta) / lengthSquared, 0.0, 1.0);
  endParameters[endParametersOffset + row] = clamp(dot(second - start, delta) / lengthSquared, 0.0, 1.0);`})),E.push(s(e,{id:`${t}-status-candidates`,operation:_,variant:`status-candidates`,bindings:[{name:`crossingOverflow`,view:L,type:`u32`,access:`read`},{name:`incidenceOverflow`,view:ve,type:`u32`,access:`read`},{name:`crossingCandidateOverflow`,view:R,type:`u32`,access:`read`},{name:`incidenceCandidateOverflow`,view:ye,type:`u32`,access:`read`},{name:`candidateScan`,view:X,type:`u32`,access:`read`},{name:`state`,view:D,type:`u32`,access:`read`},{name:`uncertainCount`,view:B,type:`u32`,access:`read_write`},{name:`candidateOverflow`,view:u.status.candidateOverflow,type:`u32`,access:`read_write`}],invocationCount:1,body:`candidateOverflow[candidateOverflowOffset] =
    select(0u, 1u, crossingOverflow[crossingOverflowOffset] != 0u || incidenceOverflow[incidenceOverflowOffset] != 0u || crossingCandidateOverflow[crossingCandidateOverflowOffset] != 0u || incidenceCandidateOverflow[incidenceCandidateOverflowOffset] != 0u || candidateScan[candidateScanOffset + ${w-1}u] > ${C}u || state[stateOffset] != 0u || uncertainCount[uncertainCountOffset] != 0u);`}),s(e,{id:`${t}-status`,operation:_,variant:`status`,bindings:[{name:`candidateScan`,view:X,type:`u32`,access:`read`},{name:`state`,view:D,type:`u32`,access:`read`},{name:`count`,view:u.status.count,type:`u32`,access:`read_write`},{name:`requiredCount`,view:u.status.requiredCount,type:`u32`,access:`read_write`},{name:`overflow`,view:u.status.overflow,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const OUTPUT_CAPACITY: u32 = ${g}u;`,body:`let incidences = candidateScan[candidateScanOffset + ${w-1}u];
  let segments = min(state[stateOffset + 1u], incidences);
  let required = incidences - segments;
  requiredCount[requiredCountOffset] = required;
  count[countOffset] = min(required, OUTPUT_CAPACITY);
  overflow[overflowOffset] = select(0u, 1u, required > OUTPUT_CAPACITY);`})),E}};export{S as n,C as t};
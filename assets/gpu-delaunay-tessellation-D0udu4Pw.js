import{t as e}from"./common-CshkX-d_.js";import{v as t}from"./spatial-context-DwZEd4EZ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{E as i,O as a,T as o,f as s,p as c,v as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{t as d}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{a as f,o as p}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as m}from"./status-aOwAAXek.js";import{t as h}from"./exact-orientation-wgsl-B_jI-HEc.js";import{c as g}from"./wgsl-common-DfyXiYB0.js";import{a as _,i as v}from"./graph-chunk-dispatch-zBt8k2-o.js";function y(e,t){let n=Object.entries(t).filter(e=>!!e[1]);for(let t=0;t<n.length;t++){let[r,i]=n[t];if(i.length===0)continue;let s=i.buffer.defaultBuffer?a(i.buffer.defaultBuffer):i.buffer;for(let c=t+1;c<n.length;c++){let[t,l]=n[c];if(l.length===0)continue;let u=l.buffer.defaultBuffer?a(l.buffer.defaultBuffer):l.buffer;o(e,s!==u||i.byteOffset+i.length*i.rowByteLength<=l.byteOffset||l.byteOffset+l.length*l.rowByteLength<=i.byteOffset,`${r} and ${t} logical byte ranges must not overlap`)}}}var b=`GPUDelaunayTessellation`,x=1073741823,S=1431655765;function C(e,t,n=0){let r=2*t+Math.max(n,1)+7;return s(e,`scratch triangle count`,r,8,x),r}function w(e,t,n,r=0){let i=C(t,n,r);return l(e,t,`triangle topology scratch`,16*i,`reduce positions rows or groups.count`),i}function T(e,t){return{"output.triangles":e.triangles,...Object.fromEntries(Object.entries(e.status).map(([e,t])=>[`output.status.${e}`,t])),"output.groups.triangleCounts":t?e.groups?.triangleCounts:void 0,"output.groups.triangleOffsets":t?e.groups?.triangleOffsets:void 0,"output.groups.status.converged":t?e.groups?.status.converged:void 0}}var E=9,D=4096,O=512,k=37376,A=55,j=`const INVALID_SITE_KEY: u32 = 0xffffffffu;`,M=`
const INCIRCLE_FILTER_BOUND: f32 = 1.0e-6;
// 2^-30: nonzero coordinates below this could underflow the float filter.
const INCIRCLE_TINY_COORDINATE: f32 = 9.313225746154785e-10;
// 2^-90: below this permanent, underflow error could exceed the filter slack.
const INCIRCLE_MINIMUM_PERMANENT: f32 = 8.077935669463161e-28;
const INCIRCLE_MAXIMUM_SPAN: i32 = ${A};
// Workspace layout in 16-bit limbs: six 5-limb differences, two 11-limb scratch values, an
// 11-limb lift, an 11-limb minor, a 22-limb product, and 22-limb positive and negative sums.
const INCIRCLE_DIFFERENCES: u32 = 0u;
const INCIRCLE_SCRATCH_A: u32 = 30u;
const INCIRCLE_SCRATCH_B: u32 = 41u;
const INCIRCLE_LIFT: u32 = 52u;
const INCIRCLE_MINOR: u32 = 63u;
const INCIRCLE_PRODUCT: u32 = 74u;
const INCIRCLE_POSITIVE: u32 = 96u;
const INCIRCLE_NEGATIVE: u32 = 118u;
const INCIRCLE_WORKSPACE_LIMBS: u32 = 140u;

fn incircleCopy(work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, destination: u32, source: u32, limbs: u32) {
  for (var limb = 0u; limb < limbs; limb++) { (*work)[destination + limb] = (*work)[source + limb]; }
}

// destination += source over limbs; callers guarantee the sum fits.
fn incircleAdd(work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, destination: u32, source: u32, limbs: u32) {
  var carry = 0u;
  for (var limb = 0u; limb < limbs; limb++) {
    let sum = (*work)[destination + limb] + (*work)[source + limb] + carry;
    (*work)[destination + limb] = sum & 0xffffu;
    carry = sum >> 16u;
  }
}

// destination -= source over limbs; callers guarantee destination >= source.
fn incircleSubtract(work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, destination: u32, source: u32, limbs: u32) {
  var borrow = 0u;
  for (var limb = 0u; limb < limbs; limb++) {
    let difference = (*work)[destination + limb] + 0x10000u - (*work)[source + limb] - borrow;
    (*work)[destination + limb] = difference & 0xffffu;
    borrow = 1u - (difference >> 16u);
  }
}

fn incircleCompare(work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, firstOffset: u32, secondOffset: u32, limbs: u32) -> i32 {
  for (var limb = limbs; limb > 0u; limb--) {
    let left = (*work)[firstOffset + limb - 1u];
    let right = (*work)[secondOffset + limb - 1u];
    if (left != right) { return select(-1, 1, left > right); }
  }
  return 0;
}

// destination[0, leftLimbs + rightLimbs) = left * right; destination must not overlap either
// factor. Limbs stay below 2^16 after each row and the running carry is at most 2^16, so no
// u32 operation overflows.
fn incircleMultiply(
  work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>,
  destination: u32, leftOffset: u32, leftLimbs: u32, rightOffset: u32, rightLimbs: u32
) {
  for (var limb = 0u; limb < leftLimbs + rightLimbs; limb++) { (*work)[destination + limb] = 0u; }
  for (var i = 0u; i < leftLimbs; i++) {
    let leftLimb = (*work)[leftOffset + i];
    if (leftLimb != 0u) {
      var carry = 0u;
      for (var j = 0u; j < rightLimbs; j++) {
        let product = leftLimb * (*work)[rightOffset + j];
        let sum = (*work)[destination + i + j] + (product & 0xffffu) + carry;
        (*work)[destination + i + j] = sum & 0xffffu;
        carry = (sum >> 16u) + (product >> 16u);
      }
      (*work)[destination + i + rightLimbs] = carry;
    }
  }
}

// Writes |value| * 2^-minimumExponent as a 5-limb integer.
fn incircleLoadMagnitude(
  work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, destination: u32, valueBits: u32, alignmentExponent: i32
) {
  let absoluteBits = valueBits & 0x7fffffffu;
  let biasedExponent = absoluteBits >> 23u;
  let significand = select(0x800000u | (absoluteBits & 0x7fffffu), absoluteBits, biasedExponent == 0u);
  let shift = select(i32(biasedExponent) - 150, -149, biasedExponent == 0u) - alignmentExponent;
  for (var limb = 0u; limb < 5u; limb++) {
    let relative = 16 * i32(limb) - shift;
    var value = 0u;
    if (relative >= 0 && relative < 24) {
      value = significand >> u32(relative);
    } else if (relative < 0 && relative > -16) {
      value = significand << u32(-relative);
    }
    (*work)[destination + limb] = value & 0xffffu;
  }
}

// Writes |first - second| into difference slot and returns whether it is negative.
fn incircleDifference(
  work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, slot: u32, first: u32, second: u32, alignmentExponent: i32
) -> bool {
  let differenceOffset = INCIRCLE_DIFFERENCES + 5u * slot;
  incircleLoadMagnitude(work, INCIRCLE_SCRATCH_A, first, alignmentExponent);
  incircleLoadMagnitude(work, INCIRCLE_SCRATCH_B, second, alignmentExponent);
  let firstNegative = (first >> 31u) == 1u;
  let secondNegative = (second >> 31u) == 1u;
  if (firstNegative != secondNegative) {
    incircleCopy(work, differenceOffset, INCIRCLE_SCRATCH_A, 5u);
    incircleAdd(work, differenceOffset, INCIRCLE_SCRATCH_B, 5u);
    return firstNegative;
  }
  if (incircleCompare(work, INCIRCLE_SCRATCH_A, INCIRCLE_SCRATCH_B, 5u) >= 0) {
    incircleCopy(work, differenceOffset, INCIRCLE_SCRATCH_A, 5u);
    incircleSubtract(work, differenceOffset, INCIRCLE_SCRATCH_B, 5u);
    return firstNegative;
  }
  incircleCopy(work, differenceOffset, INCIRCLE_SCRATCH_B, 5u);
  incircleSubtract(work, differenceOffset, INCIRCLE_SCRATCH_A, 5u);
  return !firstNegative;
}

// Adds sign * lift(x, y) * (p * q - r * t) for difference slots to the exact sums.
fn incircleAccumulateRow(
  work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>,
  negatives: ptr<function, array<u32, 6>>,
  x: u32, y: u32, p: u32, q: u32, r: u32, t: u32, rowNegative: bool
) {
  let differences = INCIRCLE_DIFFERENCES;
  incircleMultiply(work, INCIRCLE_SCRATCH_A, differences + 5u * x, 5u, differences + 5u * x, 5u);
  incircleMultiply(work, INCIRCLE_SCRATCH_B, differences + 5u * y, 5u, differences + 5u * y, 5u);
  (*work)[INCIRCLE_SCRATCH_A + 10u] = 0u;
  (*work)[INCIRCLE_SCRATCH_B + 10u] = 0u;
  incircleCopy(work, INCIRCLE_LIFT, INCIRCLE_SCRATCH_A, 11u);
  incircleAdd(work, INCIRCLE_LIFT, INCIRCLE_SCRATCH_B, 11u);

  incircleMultiply(work, INCIRCLE_SCRATCH_A, differences + 5u * p, 5u, differences + 5u * q, 5u);
  incircleMultiply(work, INCIRCLE_SCRATCH_B, differences + 5u * r, 5u, differences + 5u * t, 5u);
  (*work)[INCIRCLE_SCRATCH_A + 10u] = 0u;
  (*work)[INCIRCLE_SCRATCH_B + 10u] = 0u;
  let firstNegative = ((*negatives)[p] ^ (*negatives)[q]) == 1u;
  let secondNegative = ((*negatives)[r] ^ (*negatives)[t]) == 1u;
  var minorNegative = firstNegative;
  if (firstNegative != secondNegative) {
    incircleCopy(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_A, 11u);
    incircleAdd(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_B, 11u);
  } else if (incircleCompare(work, INCIRCLE_SCRATCH_A, INCIRCLE_SCRATCH_B, 11u) >= 0) {
    incircleCopy(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_A, 11u);
    incircleSubtract(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_B, 11u);
  } else {
    incircleCopy(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_B, 11u);
    incircleSubtract(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_A, 11u);
    minorNegative = !firstNegative;
  }

  incircleMultiply(work, INCIRCLE_PRODUCT, INCIRCLE_LIFT, 11u, INCIRCLE_MINOR, 11u);
  if (minorNegative != rowNegative) {
    incircleAdd(work, INCIRCLE_NEGATIVE, INCIRCLE_PRODUCT, 22u);
  } else {
    incircleAdd(work, INCIRCLE_POSITIVE, INCIRCLE_PRODUCT, 22u);
  }
}

// Exact sign of the incircle determinant: -1, 0 or 1, or 2 when it cannot be evaluated.
fn incircleExact(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, point: vec2<f32>) -> i32 {
  var bits = array<u32, 8>(
    bitcast<u32>(a.x), bitcast<u32>(a.y), bitcast<u32>(b.x), bitcast<u32>(b.y),
    bitcast<u32>(c.x), bitcast<u32>(c.y), bitcast<u32>(point.x), bitcast<u32>(point.y));
  var minimumExponent: i32 = 100000;
  var maximumExponent: i32 = -100000;
  for (var coordinate = 0u; coordinate < 8u; coordinate++) {
    let magnitude = bits[coordinate] & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return 2; }
    if (magnitude != 0u) {
      let exponentBits = magnitude >> 23u;
      let exponent = select(i32(exponentBits) - 150, -149, exponentBits == 0u);
      minimumExponent = min(minimumExponent, exponent);
      maximumExponent = max(maximumExponent, exponent);
    }
  }
  if (minimumExponent > maximumExponent) { return 0; }
  // Aligned coordinates have at most 24 + 55 = 79 bits, so differences fit 5 limbs, lifts and
  // minors 11 limbs, and the three 322-bit terms 22 limbs.
  if (maximumExponent - minimumExponent > INCIRCLE_MAXIMUM_SPAN) { return 2; }
  var work: array<u32, INCIRCLE_WORKSPACE_LIMBS>;
  // Sign bits of the six differences, 1 when negative.
  var negatives: array<u32, 6>;
  for (var slot = 0u; slot < 6u; slot++) {
    let negative = incircleDifference(&work, slot, bits[slot], bits[6u + (slot & 1u)], minimumExponent);
    negatives[slot] = select(0u, 1u, negative);
  }
  // Slots: 0 adx, 1 ady, 2 bdx, 3 bdy, 4 cdx, 5 cdy, all relative to point.
  // alift * (bdx cdy - cdx bdy) - blift * (adx cdy - cdx ady) + clift * (adx bdy - bdx ady)
  incircleAccumulateRow(&work, &negatives, 0u, 1u, 2u, 5u, 4u, 3u, false);
  incircleAccumulateRow(&work, &negatives, 2u, 3u, 0u, 5u, 4u, 1u, true);
  incircleAccumulateRow(&work, &negatives, 4u, 5u, 0u, 3u, 2u, 1u, false);
  return incircleCompare(&work, INCIRCLE_POSITIVE, INCIRCLE_NEGATIVE, 22u);
}

// Sign of the incircle determinant for the counter-clockwise triangle abc: 1 when point is strictly
// inside the circumcircle, -1 strictly outside, 0 on it. Unrepresentable inputs (non-finite, or an
// exponent span over 55 bits) keep the float sign.
fn incircleSign(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, point: vec2<f32>) -> i32 {
  let av = a - point;
  let bv = b - point;
  let cv = c - point;
  let bcLeft = bv.x * cv.y;
  let bcRight = cv.x * bv.y;
  let caLeft = cv.x * av.y;
  let caRight = av.x * cv.y;
  let abLeft = av.x * bv.y;
  let abRight = bv.x * av.y;
  let aLift = av.x * av.x + av.y * av.y;
  let bLift = bv.x * bv.x + bv.y * bv.y;
  let cLift = cv.x * cv.x + cv.y * cv.y;
  let determinant = aLift * (bcLeft - bcRight) + bLift * (caLeft - caRight) + cLift * (abLeft - abRight);
  let permanent = (abs(bcLeft) + abs(bcRight)) * aLift + (abs(caLeft) + abs(caRight)) * bLift +
    (abs(abLeft) + abs(abRight)) * cLift;
  let first = vec4<f32>(a, b);
  let second = vec4<f32>(c, point);
  let tiny = any((abs(first) < vec4<f32>(INCIRCLE_TINY_COORDINATE)) & (first != vec4<f32>(0.0))) ||
    any((abs(second) < vec4<f32>(INCIRCLE_TINY_COORDINATE)) & (second != vec4<f32>(0.0)));
  if (!tiny && permanent >= INCIRCLE_MINIMUM_PERMANENT) {
    let bound = INCIRCLE_FILTER_BOUND * permanent;
    if (determinant > bound) { return 1; }
    if (-determinant > bound) { return -1; }
  }
  let exact = incircleExact(a, b, c, point);
  if (exact == 2) { return i32(sign(determinant)); }
  return exact;
}`,N=`
fn compareCoordinates(a: f32, b: f32) -> i32 {
  return select(select(0, -1, a < b), 1, a > b);
}
// Sign of orient(r, S_k, p): the tilted direction d_k decides, axis comparison first.
fn orientRealSuper(r: vec2<f32>, superIndex: u32, p: vec2<f32>) -> i32 {
  let dx = compareCoordinates(p.x, r.x);
  let dy = compareCoordinates(p.y, r.y);
  if (superIndex == 0u) { return select(dx, dy, dy != 0); }
  if (superIndex == 1u) { return select(dy, -dx, dx != 0); }
  return select(dx, -dy, dy != 0);
}
// Sign of orient(point(a), point(b), point(p)) for a real site p; a and b may be super vertices.
fn orientSymbolic(a: u32, b: u32, p: u32) -> i32 {
  let point = pointAt(p);
  if (a < POINT_COUNT && b < POINT_COUNT) { return orientSign(pointAt(a), pointAt(b), point); }
  if (a < POINT_COUNT) { return orientRealSuper(pointAt(a), b - POINT_COUNT, point); }
  if (b < POINT_COUNT) { return -orientRealSuper(pointAt(b), a - POINT_COUNT, point); }
  return select(-1, 1, b - POINT_COUNT == (a - POINT_COUNT + 1u) % 3u);
}`,P=`
// p is collinear with a and b: strictly inside the chord, exact comparisons only.
fn isInsideOpenSegment(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> bool {
  if (a.x != b.x) { return p.x > min(a.x, b.x) && p.x < max(a.x, b.x); }
  return p.y > min(a.y, b.y) && p.y < max(a.y, b.y);
}

// Sign of orient(x, y, z) when at least two of the three are super vertices, from their indices
// only: rotations of (S0, S1, S2) are counter-clockwise, and with one real vertex the two super
// vertices that follow it in cyclic order decide, as in orientSymbolic.
fn orientMostlySuper(x: u32, y: u32, z: u32) -> i32 {
  var rows = array<u32, 3>(x, y, z);
  var realSlot = 3u;
  for (var slot = 0u; slot < 3u; slot++) {
    if (rows[slot] < POINT_COUNT) { realSlot = slot; }
  }
  if (realSlot == 3u) {
    return select(-1, 1, rows[1] - POINT_COUNT == (rows[0] - POINT_COUNT + 1u) % 3u);
  }
  let first = rows[(realSlot + 1u) % 3u] - POINT_COUNT;
  let second = rows[(realSlot + 2u) % 3u] - POINT_COUNT;
  return select(-1, 1, second == (first + 1u) % 3u);
}

// D(x, y, z, p) for a counter-clockwise triangle x y z and a real site p: 1 inside, -1 outside.
// Never 0.
fn getInsideSign(x: u32, y: u32, z: u32, p: u32) -> i32 {
  var rows = array<u32, 4>(x, y, z, p);
  var farthestSlot = 3u;
  for (var slot = 0u; slot < 3u; slot++) {
    if (rows[slot] >= POINT_COUNT && (farthestSlot == 3u || rows[slot] > rows[farthestSlot])) {
      farthestSlot = slot;
    }
  }
  if (farthestSlot == 3u) {
    let incircle = incircleSign(pointAt(x), pointAt(y), pointAt(z), pointAt(p));
    if (incircle != 0) { return incircle; }
  }
  // One orientation decides the rest: the chord of a circle through a super vertex, or the
  // cofactor of the dominant site of a cocircular quadruple.
  var first = 0u;
  var second = 0u;
  var third = p;
  var cofactorSign = 1;
  if (farthestSlot != 3u) {
    first = rows[(farthestSlot + 1u) % 3u];
    second = rows[(farthestSlot + 2u) % 3u];
  } else {
    var dominant = 0u;
    for (var slot = 1u; slot < 4u; slot++) {
      if (rows[slot] < rows[dominant]) { dominant = slot; }
    }
    var others = array<u32, 3>(0u, 0u, 0u);
    var otherCount = 0u;
    for (var slot = 0u; slot < 4u; slot++) {
      if (slot != dominant) {
        others[otherCount] = rows[slot];
        otherCount++;
      }
    }
    first = others[0];
    second = others[1];
    third = others[2];
    cofactorSign = select(1, -1, (dominant & 1u) == 1u);
  }
  let side = orientSymbolic(first, second, third);
  if (farthestSlot != 3u) {
    if (side != 0) { return side; }
    return select(-1, 1, isInsideOpenSegment(pointAt(first), pointAt(second), pointAt(p)));
  }
  return cofactorSign * side;
}

// Whether edge a -> b of T1 = (a, b, c), shared with T2 = (b, a, d), must flip: D(a, b, c, d) > 0.
// The arguments are permuted so the query is a real site (an interior edge has one): D(a, b, c, d)
// = D(b, a, d, c) (two swaps), = -D(b, c, d, a) (a 4-cycle), = D(a, c, d, b) (a 3-cycle).
fn isFlipRequired(a: u32, b: u32, c: u32, d: u32) -> bool {
  var x = a;
  var y = b;
  var z = c;
  var p = d;
  var parity = 1;
  var knownCounterClockwise = true;
  if (d >= POINT_COUNT) {
    if (c < POINT_COUNT) {
      x = b; y = a; z = d; p = c;
    } else if (a < POINT_COUNT) {
      x = b; y = c; z = d; p = a; parity = -1; knownCounterClockwise = false;
    } else {
      x = a; y = c; z = d; p = b; knownCounterClockwise = false;
    }
  }
  // Both c and d are super vertices here, so x y z has at most one real vertex.
  if (!knownCounterClockwise && orientMostlySuper(x, y, z) < 0) {
    let swap = x;
    x = y;
    y = swap;
    parity = -parity;
  }
  return parity * getInsideSign(x, y, z, p) > 0;
}`,F=`
fn getInsertionPriority(row: u32) -> u32 {
  var hash = row;
  hash = hash ^ (hash >> 16u);
  hash = hash * 0x85ebca6bu;
  hash = hash ^ (hash >> 13u);
  hash = hash * 0xc2b2ae35u;
  hash = hash ^ (hash >> 16u);
  return hash;
}`,I=`
fn getFlipPriority(edgeId: u32) -> u32 {
  var hash = edgeId & 0x7fffffffu;
  hash = hash ^ (hash >> 15u);
  hash = (hash * 0x2c1b3c6du) & 0x7fffffffu;
  hash = hash ^ (hash >> 12u);
  hash = (hash * 0x297a2d39u) & 0x7fffffffu;
  hash = hash ^ (hash >> 15u);
  return hash;
}`;function L(e){if(!Number.isSafeInteger(e)||e<0)throw Error(`pointCount must be a non-negative safe integer`);return Math.max(0,2*e-5)}function R(t){if(t<1)return 0;let n=4*Math.ceil(Math.log2(t+2))+32+Math.ceil(.5*Math.sqrt(t));return e(2*t+32,n,D)}function z(e){let t=R(e);return e<=2048?Math.min(t,4*Math.ceil(Math.log2(e+2))+32+Math.ceil(.5*Math.sqrt(e))):t}var B=class{id;props;spatialContext;estimatedCommandNodeCount;constructor(e){this.id=e.id??`delaunay-tessellation`,this.props=e,this.spatialContext=t(this.id,e.spatialContext);let{id:n}=this,{output:r}=e;i(e.positions,[`float32x2`],`${n} positions`),i(r.triangles,[`uint32x3`],`${n} output.triangles`),m(`${n} output.status`,r.status),s(n,`output.triangles rows`,r.triangles.length,0,S),e.groups&&s(n,`groups.count`,e.groups.count),C(n,e.positions.length,e.groups?.count??0);let a=T(r,!!e.groups);y(n,a);let o=r.status;if(v(n,{invalidCount:o.invalidCount,duplicateCount:o.duplicateCount,requiredCount:o.requiredCount}),v(n,{invalidCount:o.invalidCount,converged:o.converged,iterationCount:o.iterationCount,iterationLimitReached:o.iterationLimitReached}),v(n,{requiredCount:o.requiredCount,count:o.count,overflow:o.overflow}),r.triangles.length>0&&v(n,{requiredCount:o.requiredCount,triangles:r.triangles}),e.groups){let e=r.groups;v(n,{counts:e?.triangleCounts,converged:e?.status.converged}),v(n,{counts:e?.triangleCounts,requiredCount:o.requiredCount}),v(n,{counts:e?.triangleCounts,offsets:e?.triangleOffsets}),v(n,{converged:e?.status.converged,invalidCount:o.invalidCount})}if(u(n,Object.values(a),[e.positions,e.groups?.ids]),!r.status.requiredCount)throw Error(`${n} output.status.requiredCount is required`);if(e.groups){let{ids:t,count:a}=e.groups;if(i(t,[`uint32`],`${n} groups.ids`),t.length!==e.positions.length||!Number.isInteger(a)||a<1)throw Error(`${n} groups.ids must have one row per position and count at least 1`);let o=r.groups;if(!o)throw Error(`${n} output.groups is required with groups`);for(let e of[o.triangleCounts,o.status.converged,o.triangleOffsets])if(e&&(i(e,[`uint32`],`${n} output.groups`),e.length!==a))throw Error(`${n} output.groups views must have one row per group`)}if(r.status.duplicateCount&&(i(r.status.duplicateCount,[`uint32`],`${n} output.status.duplicateCount`),r.status.duplicateCount.length<1))throw Error(`${n} output.status.duplicateCount must contain one row`);this.estimatedCommandNodeCount=E*z(e.positions.length)+O,c(n,this.estimatedCommandNodeCount,k)}getCommandNodes(e){let{id:t,props:r}=this,{output:i}=r,{status:a}=i;_(e,t,{positions:r.positions,"groups.ids":r.groups?.ids,...T(i,!!r.groups)});let o=r.positions.length,s=r.groups?.count??0,c=w(e.device,t,o,s),l=(r,i)=>n(e,`${t}-${r}`,`uint32`,Math.max(i,1)),u=l(`triangle-vertices`,4*c),d=l(`triangle-adjacency`,4*c),f=l(`replacement`,3*c),m=l(`children`,4*c),g=l(`votes`,c),v=l(`point-locations`,o),y=l(`duplicate-flags`,o),x=l(`seeds`,4*Math.max(s,1)),S=l(`site-group-keys`,s?o:0),C=l(`group-invalid`,s),E=l(`state`,8),D=a.invalidCount??l(`invalid-count`,1),O=a.duplicateCount??l(`duplicate-count`,1),k=H({pointCount:o,scratchTriangleCapacity:c}),A=`${k}
${U}
${h}
${N}`,j=`${A}
${M}
${P}
${I}`,F={name:`positions`,view:r.positions,type:`f32`,access:`read`},L=(e,t,n=`read`,r=`u32`)=>({name:e,view:t,type:r,access:n}),R=[p(e,{id:`${t}-initialize-status`,operation:b,variant:`initialize-status`,bindings:[L(`state`,E,`read_write`),L(`seeds`,x,`read_write`),L(`invalidCount`,D,`read_write`),L(`duplicateCount`,O,`read_write`),L(`requiredCount`,a.requiredCount,`read_write`)],invocationCount:1,body:`for (var word = 0u; word < 8u; word++) { state[stateOffset + word] = 0u; }
  ${s?``:`seeds[seedsOffset] = 0xffffffffu;
  seeds[seedsOffset + 1u] = 0xffffffffu;
  seeds[seedsOffset + 2u] = 0xffffffffu;
  seeds[seedsOffset + 3u] = 0u;`}
  invalidCount[invalidCountOffset] = 0u;
  duplicateCount[duplicateCountOffset] = 0u;
  requiredCount[requiredCountOffset] = 0u;`})];if(s){let n=i.groups;R.push(p(e,{id:`${t}-initialize-groups`,operation:b,variant:`initialize-groups`,bindings:[L(`seeds`,x,`read_write`),L(`groupInvalid`,C,`read_write`),L(`triangleCounts`,n.triangleCounts,`read_write`),L(`converged`,n.status.converged,`read_write`)],invocationCount:s,body:`seeds[seedsOffset + 4u * index] = 0xffffffffu;
  seeds[seedsOffset + 4u * index + 1u] = 0xffffffffu;
  seeds[seedsOffset + 4u * index + 2u] = 0xffffffffu;
  seeds[seedsOffset + 4u * index + 3u] = 0u;
  groupInvalid[groupInvalidOffset + index] = 0u;
  triangleCounts[triangleCountsOffset + index] = 0u;
  converged[convergedOffset + index] = 0u;`}))}return o>0&&(R.push(...this._getSiteClassificationNodes(e,{invalidCount:D,duplicateCount:O,duplicateFlags:y,seeds:x,siteGroupKeys:S,groupInvalid:C})),R.push(p(e,{id:`${t}-initialize-triangles`,operation:b,variant:`initialize-triangles`,bindings:[L(`triangleVertices`,u,`read_write`),L(`triangleAdjacency`,d,`read_write`),L(`votes`,g,`read_write`),L(`state`,E,`read_write`),L(`invalidCount`,D)],invocationCount:c,declarations:`const POINT_COUNT: u32 = ${o}u;
const NONE: u32 = 0xffffffffu;
const GROUP_COUNT: u32 = ${s}u;`,body:`let base = 4u * index;
  let triangulate = ${s?`index < GROUP_COUNT`:`index == 0u && invalidCount[invalidCountOffset] == 0u`};
  triangleVertices[triangleVerticesOffset + base] = select(NONE, POINT_COUNT, triangulate);
  triangleVertices[triangleVerticesOffset + base + 1u] = select(NONE, POINT_COUNT + 1u, triangulate);
  triangleVertices[triangleVerticesOffset + base + 2u] = select(NONE, POINT_COUNT + 2u, triangulate);
  triangleVertices[triangleVerticesOffset + base + 3u] = 0u;
  for (var word = 0u; word < 3u; word++) { triangleAdjacency[triangleAdjacencyOffset + base + word] = NONE; }
  triangleAdjacency[triangleAdjacencyOffset + base + 3u] = 0u;
  votes[votesOffset + index] = NONE;
  if (triangulate && index == 0u) {
    state[stateOffset] = ${s?`GROUP_COUNT`:`1u`};
    // Force one flip evaluation even when every site is a duplicate.
    state[stateOffset + 4u] = 1u;
  }`}),p(e,{id:`${t}-initialize-sites`,operation:b,variant:`initialize-sites`,bindings:[L(`duplicateFlags`,y),L(`invalidCount`,D),L(`pointLocations`,v,`read_write`),L(`state`,E,`read_write`,`atomic<u32>`),...s?[L(`siteGroupKeys`,S),L(`groupInvalid`,C)]:[]],invocationCount:o,declarations:`const NONE: u32 = 0xffffffffu;
const GROUP_COUNT: u32 = ${s}u;`,body:s?`let group = siteGroupKeys[siteGroupKeysOffset + index];
  let located = invalidCount[invalidCountOffset] == 0u && group < GROUP_COUNT &&
    groupInvalid[groupInvalidOffset + min(group, GROUP_COUNT - 1u)] == 0u &&
    duplicateFlags[duplicateFlagsOffset + index] == 0u;
  pointLocations[pointLocationsOffset + index] = select(NONE, (group << 2u) | 3u, located);
  if (located) { atomicAdd(&state[stateOffset + 2u], 1u); }`:`let located = invalidCount[invalidCountOffset] == 0u && duplicateFlags[duplicateFlagsOffset + index] == 0u;
  pointLocations[pointLocationsOffset + index] = select(NONE, 3u, located);
  if (located) { atomicAdd(&state[stateOffset + 2u], 1u); }`})),R.push(...this._getStepNodes(e,{pointCount:o,groupCount:s,siteGroupKeys:S,scratchTriangleCapacity:c,positionsBinding:F,binding:L,triangleVertices:u,triangleAdjacency:d,replacement:f,children:m,votes:g,pointLocations:v,seeds:x,state:E,orientationDeclarations:A,flipDeclarations:j,sharedDeclarations:k}))),R.push(...this._getOutputNodes(e,{pointCount:o,groupCount:s,siteGroupKeys:S,groupInvalid:C,scratchTriangleCapacity:c,triangleVertices:u,state:E,invalidCount:D})),R}_getSiteClassificationNodes(e,t){let{id:r,props:i}=this,a=i.positions.length;if(i.groups?.count??0)return this._getGroupSiteClassificationNodes(e,t);let o=n(e,`${r}-site-rows`,`uint32`,a),s=n(e,`${r}-site-x-keys`,`uint32`,a),c=n(e,`${r}-site-y-keys`,`uint32`,a),l=n(e,`${r}-exponent-range`,`uint32`,2),u=d(e,{id:`${r}-site-sort`,operation:b,indices:o,keys:[{view:s},{view:c}]});return[p(e,{id:`${r}-classify-sites`,operation:b,variant:`parallel-site-validation`,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`siteRows`,view:o,type:`u32`,access:`read_write`},{name:`siteXKeys`,view:s,type:`u32`,access:`read_write`},{name:`siteYKeys`,view:c,type:`u32`,access:`read_write`},{name:`invalidCount`,view:t.invalidCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:a,declarations:`${j}
${g}
// Float equality, not bit equality, decides duplicates: -0 and +0 share one key.
fn coordinateKey(value: f32) -> u32 {
  return select(bitcast<u32>(value), 0u, value == 0.0);
}`,body:`let base = positionsOffset + 2u * index;
  let point = vec2<f32>(positions[base], positions[base + 1u]);
  let valid = isFiniteBits(point.x) && isFiniteBits(point.y);
  if (!valid) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }
  siteRows[siteRowsOffset + index] = index;
  // Non-finite rows take a NaN bit pattern that no finite key equals, so they never pair.
  siteXKeys[siteXKeysOffset + index] = select(INVALID_SITE_KEY, coordinateKey(point.x), valid);
  siteYKeys[siteYKeysOffset + index] = select(INVALID_SITE_KEY, coordinateKey(point.y), valid);`}),p(e,{id:`${r}-exponent-range-initialize`,operation:b,variant:`exponent-range-initialize`,bindings:[{name:`range`,view:l,type:`u32`,access:`read_write`}],invocationCount:1,body:`range[rangeOffset] = 0xffffffffu; range[rangeOffset + 1u] = 0u;`}),p(e,{id:`${r}-exponent-range`,operation:b,variant:`exponent-range`,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`range`,view:l,type:`atomic<u32>`,access:`read_write`}],invocationCount:a*2,body:`let magnitude = bitcast<u32>(positions[positionsOffset + index]) & 0x7fffffffu;
if (magnitude != 0u && magnitude < 0x7f800000u) {
  let biased = max(magnitude >> 23u, 1u);
  atomicMin(&range[rangeOffset], biased); atomicMax(&range[rangeOffset + 1u], biased);
}`}),p(e,{id:`${r}-exponent-span-check`,operation:b,variant:`exponent-span-check`,bindings:[{name:`range`,view:l,type:`u32`,access:`read`},{name:`invalidCount`,view:t.invalidCount,type:`u32`,access:`read_write`}],invocationCount:1,body:`let minimum = range[rangeOffset]; let maximum = range[rangeOffset + 1u];
if (minimum != 0xffffffffu && maximum - minimum > ${A}u) { invalidCount[invalidCountOffset] += 1u; }`}),...u.nodes,p(e,{id:`${r}-flag-duplicate-sites`,operation:b,variant:`sorted-duplicate-sites`,bindings:[{name:`sortedRows`,view:u.sortedIndices,type:`u32`,access:`read`},{name:`sortedXKeys`,view:u.sortedPrimaryKeys,type:`u32`,access:`read`},{name:`siteYKeys`,view:c,type:`u32`,access:`read`},{name:`duplicateFlags`,view:t.duplicateFlags,type:`u32`,access:`read_write`},{name:`seeds`,view:t.seeds,type:`atomic<u32>`,access:`read_write`},{name:`duplicateCount`,view:t.duplicateCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:a,declarations:j,body:`let row = sortedRows[sortedRowsOffset + index];
  let xKey = sortedXKeys[sortedXKeysOffset + index];
  var duplicate = false;
  if (index > 0u && xKey != INVALID_SITE_KEY) {
    let previousRow = sortedRows[sortedRowsOffset + index - 1u];
    duplicate = sortedXKeys[sortedXKeysOffset + index - 1u] == xKey &&
      siteYKeys[siteYKeysOffset + previousRow] == siteYKeys[siteYKeysOffset + row];
  }
  duplicateFlags[duplicateFlagsOffset + row] = select(0u, 1u, duplicate);
  if (duplicate) { atomicAdd(&duplicateCount[duplicateCountOffset], 1u); }
  if (index == 0u) { atomicStore(&seeds[seedsOffset], row); }
  if (!duplicate && xKey != INVALID_SITE_KEY) { atomicMax(&seeds[seedsOffset + 3u], index); }`}),p(e,{id:`${r}-select-seed-edge`,operation:b,variant:`select-seed-edge`,bindings:[{name:`sortedRows`,view:u.sortedIndices,type:`u32`,access:`read`},{name:`seeds`,view:t.seeds,type:`u32`,access:`read_write`}],invocationCount:1,body:`seeds[seedsOffset + 1u] = sortedRows[sortedRowsOffset + seeds[seedsOffset + 3u]];`}),p(e,{id:`${r}-select-seed-triangle`,operation:b,variant:`select-seed-triangle`,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`seeds`,view:t.seeds,type:`atomic<u32>`,access:`read_write`},{name:`invalidCount`,view:t.invalidCount,type:`u32`,access:`read`}],invocationCount:a,declarations:`${U}
${h}`,body:`if (invalidCount[invalidCountOffset] != 0u) { return; }
  let first = atomicLoad(&seeds[seedsOffset]);
  let second = atomicLoad(&seeds[seedsOffset + 1u]);
  if (orientSign(pointAt(first), pointAt(second), pointAt(index)) != 0) {
    atomicMin(&seeds[seedsOffset + 2u], index);
  }`})]}_getGroupSiteClassificationNodes(t,r){let{id:i,props:a}=this,o=a.positions.length,{ids:s,count:c}=a.groups,l=n(t,`${i}-site-rows`,`uint32`,o),u=n(t,`${i}-site-x-keys`,`uint32`,o),f=n(t,`${i}-site-y-keys`,`uint32`,o),m=n(t,`${i}-exponent-range`,`uint32`,2*c),_=e(32-Math.clz32(c),1,32),v=d(t,{id:`${i}-site-sort`,operation:b,indices:l,keys:[{view:r.siteGroupKeys,keyBits:_},{view:u},{view:f}]}),y=`const GROUP_COUNT: u32 = ${c}u;
const NONE: u32 = 0xffffffffu;`;return[p(t,{id:`${i}-classify-sites`,operation:b,variant:`parallel-group-site-validation`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`groupIds`,view:s,type:`u32`,access:`read`},{name:`siteRows`,view:l,type:`u32`,access:`read_write`},{name:`siteXKeys`,view:u,type:`u32`,access:`read_write`},{name:`siteYKeys`,view:f,type:`u32`,access:`read_write`},{name:`siteGroupKeys`,view:r.siteGroupKeys,type:`u32`,access:`read_write`},{name:`groupInvalid`,view:r.groupInvalid,type:`atomic<u32>`,access:`read_write`}],invocationCount:o,declarations:`${j}
${g}
${y}
// Float equality, not bit equality, decides duplicates: -0 and +0 share one key.
fn coordinateKey(value: f32) -> u32 {
  return select(bitcast<u32>(value), 0u, value == 0.0);
}`,body:`let base = positionsOffset + 2u * index;
  let point = vec2<f32>(positions[base], positions[base + 1u]);
  let group = groupIds[groupIdsOffset + index];
  let member = group < GROUP_COUNT;
  let valid = isFiniteBits(point.x) && isFiniteBits(point.y);
  if (member && !valid) { atomicOr(&groupInvalid[groupInvalidOffset + group], 1u); }
  siteRows[siteRowsOffset + index] = index;
  siteGroupKeys[siteGroupKeysOffset + index] = select(GROUP_COUNT, group, member && valid);
  siteXKeys[siteXKeysOffset + index] = select(INVALID_SITE_KEY, coordinateKey(point.x), valid);
  siteYKeys[siteYKeysOffset + index] = select(INVALID_SITE_KEY, coordinateKey(point.y), valid);`}),p(t,{id:`${i}-exponent-range-initialize`,operation:b,variant:`group-exponent-range-initialize`,bindings:[{name:`range`,view:m,type:`u32`,access:`read_write`}],invocationCount:c,body:`range[rangeOffset + 2u * index] = 0xffffffffu; range[rangeOffset + 2u * index + 1u] = 0u;`}),p(t,{id:`${i}-exponent-range`,operation:b,variant:`group-exponent-range`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`siteGroupKeys`,view:r.siteGroupKeys,type:`u32`,access:`read`},{name:`range`,view:m,type:`atomic<u32>`,access:`read_write`}],invocationCount:o*2,declarations:y,body:`let group = siteGroupKeys[siteGroupKeysOffset + index / 2u];
if (group >= GROUP_COUNT) { return; }
let magnitude = bitcast<u32>(positions[positionsOffset + index]) & 0x7fffffffu;
if (magnitude != 0u && magnitude < 0x7f800000u) {
  let biased = max(magnitude >> 23u, 1u);
  atomicMin(&range[rangeOffset + 2u * group], biased); atomicMax(&range[rangeOffset + 2u * group + 1u], biased);
}`}),p(t,{id:`${i}-exponent-span-check`,operation:b,variant:`group-exponent-span-check`,bindings:[{name:`range`,view:m,type:`u32`,access:`read`},{name:`groupInvalid`,view:r.groupInvalid,type:`atomic<u32>`,access:`read_write`}],invocationCount:c,body:`let minimum = range[rangeOffset + 2u * index]; let maximum = range[rangeOffset + 2u * index + 1u];
if (minimum != 0xffffffffu && maximum - minimum > ${A}u) { atomicOr(&groupInvalid[groupInvalidOffset + index], 1u); }`}),...v.nodes,p(t,{id:`${i}-flag-duplicate-sites`,operation:b,variant:`sorted-group-duplicate-sites`,bindings:[{name:`sortedRows`,view:v.sortedIndices,type:`u32`,access:`read`},{name:`sortedGroupKeys`,view:v.sortedPrimaryKeys,type:`u32`,access:`read`},{name:`siteXKeys`,view:u,type:`u32`,access:`read`},{name:`siteYKeys`,view:f,type:`u32`,access:`read`},{name:`duplicateFlags`,view:r.duplicateFlags,type:`u32`,access:`read_write`},{name:`seeds`,view:r.seeds,type:`atomic<u32>`,access:`read_write`},{name:`duplicateCount`,view:r.duplicateCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:o,declarations:y,body:`let row = sortedRows[sortedRowsOffset + index];
  let group = sortedGroupKeys[sortedGroupKeysOffset + index];
  var duplicate = false;
  if (group < GROUP_COUNT) {
    let first = index == 0u || sortedGroupKeys[sortedGroupKeysOffset + index - 1u] != group;
    if (!first) {
      let previousRow = sortedRows[sortedRowsOffset + index - 1u];
      duplicate = siteXKeys[siteXKeysOffset + previousRow] == siteXKeys[siteXKeysOffset + row] &&
        siteYKeys[siteYKeysOffset + previousRow] == siteYKeys[siteYKeysOffset + row];
    }
    if (duplicate) { atomicAdd(&duplicateCount[duplicateCountOffset], 1u); }
    if (first) { atomicStore(&seeds[seedsOffset + 4u * group], row); }
    if (!duplicate) { atomicMax(&seeds[seedsOffset + 4u * group + 3u], index); }
  }
  duplicateFlags[duplicateFlagsOffset + row] = select(0u, 1u, duplicate);`}),p(t,{id:`${i}-select-seed-edge`,operation:b,variant:`select-group-seed-edge`,bindings:[{name:`sortedRows`,view:v.sortedIndices,type:`u32`,access:`read`},{name:`seeds`,view:r.seeds,type:`u32`,access:`read_write`}],invocationCount:c,declarations:y,body:`if (seeds[seedsOffset + 4u * index] == NONE) { return; }
  seeds[seedsOffset + 4u * index + 1u] = sortedRows[sortedRowsOffset + seeds[seedsOffset + 4u * index + 3u]];`}),p(t,{id:`${i}-select-seed-triangle`,operation:b,variant:`select-group-seed-triangle`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`seeds`,view:r.seeds,type:`atomic<u32>`,access:`read_write`},{name:`siteGroupKeys`,view:r.siteGroupKeys,type:`u32`,access:`read`},{name:`groupInvalid`,view:r.groupInvalid,type:`u32`,access:`read`}],invocationCount:o,declarations:`${U}
${h}
${y}`,body:`let group = siteGroupKeys[siteGroupKeysOffset + index];
  if (group >= GROUP_COUNT || groupInvalid[groupInvalidOffset + group] != 0u) { return; }
  let first = atomicLoad(&seeds[seedsOffset + 4u * group]);
  let second = atomicLoad(&seeds[seedsOffset + 4u * group + 1u]);
  if (first == NONE || second == NONE) { return; }
  if (orientSign(pointAt(first), pointAt(second), pointAt(index)) != 0) {
    atomicMin(&seeds[seedsOffset + 4u * group + 2u], index);
  }`})]}_getStepNodes(e,t){let{id:n}=this,{pointCount:r,scratchTriangleCapacity:i,positionsBinding:a,binding:o,triangleVertices:s,triangleAdjacency:c,replacement:l,children:u,votes:d,pointLocations:m,seeds:h,state:g}=t,_=f(e,{id:`${n}-insertion-dispatch`,operation:b,shapes:[[r],[i]]}),v=f(e,{id:`${n}-flip-dispatch`,operation:b,shapes:[[i],[r]]}),y=(n,r,a)=>p(e,{id:`${n}-repair-adjacency`,operation:b,variant:r===0?`insertion-adjacency-repair`:`flip-adjacency-repair`,...a,bindings:[o(`triangleVertices`,s),o(`triangleAdjacency`,c,`read_write`),o(`replacement`,l),o(`votes`,d,`read_write`),o(`state`,g)],invocationCount:i,declarations:`${t.sharedDeclarations}
const PHASE_OFFSET: u32 = ${r}u;`,body:J}),x=(n,i,c)=>p(e,{id:`${n}-relocate`,operation:b,variant:i===0?`insertion-relocation`:`flip-relocation`,...c,bindings:[a,o(`pointLocations`,m,`read_write`),o(`triangleVertices`,s),o(`children`,u),o(`state`,g,`read_write`,`atomic<u32>`)],invocationCount:r,declarations:`${t.orientationDeclarations}
const PHASE_OFFSET: u32 = ${i}u;`,body:Y}),S=[],C=z(r);for(let f=0;f<C;f++){let C=`${n}-step-${f}`,w=_.getGate(0),T=_.getGate(1),E=v.getGate(0),D=v.getGate(1);S.push(p(e,{id:`${C}-gate`,operation:b,variant:`step-gate`,bindings:[o(`state`,g,`read_write`),{..._.binding,name:`insertionDispatch`},{...v.binding,name:`flipDispatch`}],invocationCount:1,body:`state[stateOffset + 7u] = select(0u, 1u,
    state[stateOffset + 2u] > 0u || state[stateOffset + 4u] > 0u);
  let isActive = state[stateOffset + 1u] == 0u && state[stateOffset + 7u] != 0u;
  let isInserting = isActive && state[stateOffset + 2u] > 0u;
  state[stateOffset + 3u] = state[stateOffset + 3u] + 2u;
  if (isActive) {
    state[stateOffset + 4u] = 0u;
    state[stateOffset + 5u] = 0u;
    state[stateOffset + 6u] = state[stateOffset + 6u] + 1u;
  }
  if (isInserting) {
    ${V(_.armWGSL,`insertionDispatch`)}
  } else {
    ${V(_.disarmWGSL,`insertionDispatch`)}
  }
  if (isActive) {
    ${V(v.armWGSL,`flipDispatch`)}
  } else {
    ${V(v.disarmWGSL,`flipDispatch`)}
  }`}),p(e,{id:`${C}-insert-vote`,operation:b,variant:`insertion-vote`,...w,bindings:[o(`pointLocations`,m),o(`seeds`,h),o(`state`,g),o(`triangleAdjacency`,c),o(`votes`,d,`read_write`,`atomic<u32>`),...t.groupCount?[o(`siteGroupKeys`,t.siteGroupKeys)]:[]],invocationCount:r,declarations:`const NONE: u32 = 0xffffffffu;
const GROUP_COUNT: u32 = ${t.groupCount}u;
${F}`,body:`let seedStep = state[stateOffset + 6u];
  if (seedStep <= 3u) {
    ${t.groupCount?`let group = siteGroupKeys[siteGroupKeysOffset + index];
    if (group >= GROUP_COUNT || index != seeds[seedsOffset + 4u * group + seedStep - 1u]) { return; }`:`if (index != seeds[seedsOffset + seedStep - 1u]) { return; }`}
  }
  ${Z}`}),p(e,{id:`${C}-insert-split`,operation:b,variant:`insertion-split`,...w,bindings:[o(`pointLocations`,m,`read_write`),o(`triangleVertices`,s,`read_write`),o(`triangleAdjacency`,c,`read_write`),o(`replacement`,l,`read_write`),o(`children`,u,`read_write`),o(`votes`,d),o(`state`,g,`read_write`,`atomic<u32>`)],invocationCount:r,declarations:`${t.sharedDeclarations}
${W}
${F}`,body:G}),y(`${C}-insertion`,0,T),x(`${C}-insertion`,0,w),p(e,{id:`${C}-flip-evaluate`,operation:b,variant:`flip-evaluation`,...E,bindings:[a,o(`triangleVertices`,s),o(`triangleAdjacency`,c),o(`votes`,d,`read_write`,`atomic<u32>`),o(`state`,g,`read_write`,`atomic<u32>`)],invocationCount:i,declarations:t.flipDeclarations,body:X}),p(e,{id:`${C}-flip-apply`,operation:b,variant:`flip-application`,...E,bindings:[o(`triangleVertices`,s,`read_write`),o(`triangleAdjacency`,c,`read_write`),o(`replacement`,l,`read_write`),o(`children`,u,`read_write`),o(`votes`,d),o(`state`,g)],invocationCount:i,declarations:`${t.sharedDeclarations}
${W}
${I}`,body:K}),y(`${C}-flip`,1,E),x(`${C}-flip`,1,D))}if(C<R(r)){let i=`const ITERATION_LIMIT: u32 = ${R(r)}u;`,f=e=>e.replace(/state\[stateOffset(?: \+ \d+u)?\]/g,e=>`atomicLoad(&${e})`),h=e=>e.replace(/atomicMin\(&votes\[([^\]]+)\], ([^)]+)\)/g,`votes[$1] = min(votes[$1], $2)`);S.push(p(e,{id:`${n}-continue-insertion`,operation:b,variant:`serial-insertion-continuation`,bindings:[a,o(`triangleVertices`,s,`read_write`),o(`triangleAdjacency`,c,`read_write`),o(`replacement`,l,`read_write`),o(`children`,u,`read_write`),o(`votes`,d,`read_write`),o(`pointLocations`,m,`read_write`),o(`state`,g,`read_write`,`atomic<u32>`)],invocationCount:1,workgroupSize:32,declarations:`${t.orientationDeclarations}
${i}
const PHASE_OFFSET: u32 = 0u;
${W}
${F}
fn voteForInsertion(index: u32) { ${h(Z)} }
fn splitInsertion(index: u32) { ${G} }
fn repairInsertion(index: u32) { ${f(J)} }
fn relocateInsertion(index: u32) { ${Y} }`,body:`loop {
    if (atomicLoad(&state[stateOffset + 1u]) != 0u || atomicLoad(&state[stateOffset + 2u]) == 0u ||
        atomicLoad(&state[stateOffset + 6u]) >= ITERATION_LIMIT) { break; }
    atomicAdd(&state[stateOffset + 3u], 2u);
    atomicAdd(&state[stateOffset + 6u], 1u);
    // Force legalization even if the last parallel round found no illegal edge.
    atomicStore(&state[stateOffset + 4u], 1u);
    atomicStore(&state[stateOffset + 5u], 0u);
    for (var site = 0u; site < POINT_COUNT; site++) { voteForInsertion(site); }
    for (var site = 0u; site < POINT_COUNT; site++) { splitInsertion(site); }
    for (var triangle = 0u; triangle < atomicLoad(&state[stateOffset]); triangle++) { repairInsertion(triangle); }
    for (var site = 0u; site < POINT_COUNT; site++) { relocateInsertion(site); }
  }`}),p(e,{id:`${n}-continue-flips`,operation:b,variant:`serial-flip-continuation`,bindings:[a,o(`triangleVertices`,s,`read_write`),o(`triangleAdjacency`,c,`read_write`),o(`replacement`,l,`read_write`),o(`children`,u,`read_write`),o(`votes`,d,`read_write`),o(`state`,g,`read_write`)],invocationCount:1,workgroupSize:32,declarations:`${t.flipDeclarations}
${i}
const PHASE_OFFSET: u32 = 1u;
${W}
fn evaluateFlip(index: u32) {
${h(X).replace(/atomicLoad\(&([^)]*)\)/g,`$1`).replace(`atomicAdd(&state[stateOffset + 4u], 1u)`,`state[stateOffset + 4u] = state[stateOffset + 4u] + 1u`)}
}
fn applyFlip(index: u32) { ${K} }
fn repairFlip(index: u32) { ${J} }`,body:`loop {
    if (state[stateOffset + 1u] != 0u || state[stateOffset + 2u] != 0u ||
        state[stateOffset + 4u] == 0u || state[stateOffset + 6u] >= ITERATION_LIMIT) { break; }
    state[stateOffset + 3u] = state[stateOffset + 3u] + 2u;
    state[stateOffset + 6u] = state[stateOffset + 6u] + 1u;
    state[stateOffset + 4u] = 0u;
    for (var triangle = 0u; triangle < state[stateOffset]; triangle++) { evaluateFlip(triangle); }
    for (var triangle = 0u; triangle < state[stateOffset]; triangle++) { applyFlip(triangle); }
    for (var triangle = 0u; triangle < state[stateOffset]; triangle++) { repairFlip(triangle); }
  }
  state[stateOffset + 7u] = select(0u, 1u, state[stateOffset + 2u] > 0u || state[stateOffset + 4u] > 0u);`}))}return S}_getOutputNodes(t,i){let{id:a,props:o}=this,{output:s}=o,{status:c}=s,{pointCount:l,groupCount:u,scratchTriangleCapacity:f,triangleVertices:m,state:h,invalidCount:g}=i,_=n(t,`${a}-output-group-keys`,`uint32`,u?f:0),v=n(t,`${a}-output-rows`,`uint32`,f),y=n(t,`${a}-output-first-keys`,`uint32`,f),x=n(t,`${a}-output-second-keys`,`uint32`,f),S=d(t,{id:`${a}-output-sort`,operation:b,indices:v,keys:u?[{view:_,keyBits:e(32-Math.clz32(u),1,32)},{view:y},{view:x}]:[{view:y},{view:x}]}),C=`${H({pointCount:l,scratchTriangleCapacity:f})}
${q}
const GROUP_COUNT: u32 = ${u}u;`,w=[p(t,{id:`${a}-finalize`,operation:b,variant:`finalize`,bindings:[{name:`state`,view:h,type:`u32`,access:`read_write`},{name:`invalidCount`,view:g,type:`u32`,access:`read_write`},...[`converged`,`iterationCount`,`iterationLimitReached`].flatMap(e=>c[e]?[{name:e,view:c[e],type:`u32`,access:`read_write`}]:[])],invocationCount:1,body:`let exhausted = state[stateOffset + 1u] == 0u &&
    (state[stateOffset + 2u] > 0u || state[stateOffset + 4u] > 0u);
  invalidCount[invalidCountOffset] += select(state[stateOffset + 1u], 1u, exhausted);
  ${c.converged?`converged[convergedOffset] = select(0u, 1u, invalidCount[invalidCountOffset] == 0u);`:``}
  ${c.iterationCount?`iterationCount[iterationCountOffset] = state[stateOffset + 6u];`:``}
  ${c.iterationLimitReached?`iterationLimitReached[iterationLimitReachedOffset] = select(0u, 1u, exhausted);`:``}`}),...u?[p(t,{id:`${a}-classify-output`,operation:b,variant:`group-output-classification`,bindings:[{name:`triangleVertices`,view:m,type:`u32`,access:`read`},{name:`state`,view:h,type:`u32`,access:`read`},{name:`siteGroupKeys`,view:i.siteGroupKeys,type:`u32`,access:`read`},{name:`groupInvalid`,view:i.groupInvalid,type:`u32`,access:`read`},{name:`sortRows`,view:v,type:`u32`,access:`read_write`},{name:`firstKeys`,view:y,type:`u32`,access:`read_write`},{name:`secondKeys`,view:x,type:`u32`,access:`read_write`},{name:`triangleGroupKeys`,view:_,type:`u32`,access:`read_write`}],invocationCount:f,declarations:C,body:`sortRows[sortRowsOffset + index] = index;
  var accepted = state[stateOffset + 1u] == 0u &&
    index < min(state[stateOffset], SCRATCH_TRIANGLE_CAPACITY);
  var canonical = vec3<u32>(NONE, NONE, NONE);
  var group = GROUP_COUNT;
  if (accepted) {
    canonical = getCanonicalTriangle(index);
    accepted = canonical.x < POINT_COUNT && canonical.y < POINT_COUNT && canonical.z < POINT_COUNT;
  }
  if (accepted) {
    group = siteGroupKeys[siteGroupKeysOffset + canonical.x];
    accepted = group < GROUP_COUNT && groupInvalid[groupInvalidOffset + min(group, GROUP_COUNT - 1u)] == 0u;
  }
  triangleGroupKeys[triangleGroupKeysOffset + index] = select(GROUP_COUNT, group, accepted);
  firstKeys[firstKeysOffset + index] = select(NONE, canonical.x, accepted);
  secondKeys[secondKeysOffset + index] = select(NONE, canonical.y, accepted);`}),p(t,{id:`${a}-group-counts`,operation:b,variant:`group-triangle-counts`,bindings:[{name:`triangleGroupKeys`,view:_,type:`u32`,access:`read`},{name:`triangleCounts`,view:s.groups.triangleCounts,type:`atomic<u32>`,access:`read_write`},{name:`requiredCount`,view:c.requiredCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:f,declarations:`const GROUP_COUNT: u32 = ${u}u;`,body:`let group = triangleGroupKeys[triangleGroupKeysOffset + index];
  if (group < GROUP_COUNT) {
    atomicAdd(&triangleCounts[triangleCountsOffset + group], 1u);
    atomicAdd(&requiredCount[requiredCountOffset], 1u);
  }`})]:[p(t,{id:`${a}-classify-output`,operation:b,variant:`output-classification`,bindings:[{name:`triangleVertices`,view:m,type:`u32`,access:`read`},{name:`state`,view:h,type:`u32`,access:`read`},{name:`sortRows`,view:v,type:`u32`,access:`read_write`},{name:`firstKeys`,view:y,type:`u32`,access:`read_write`},{name:`secondKeys`,view:x,type:`u32`,access:`read_write`},{name:`requiredCount`,view:c.requiredCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:f,declarations:C,body:`sortRows[sortRowsOffset + index] = index;
  var accepted = state[stateOffset + 1u] == 0u &&
    index < min(state[stateOffset], SCRATCH_TRIANGLE_CAPACITY);
  var canonical = vec3<u32>(NONE, NONE, NONE);
  if (accepted) {
    canonical = getCanonicalTriangle(index);
    accepted = canonical.x < POINT_COUNT && canonical.y < POINT_COUNT && canonical.z < POINT_COUNT;
  }
  firstKeys[firstKeysOffset + index] = select(NONE, canonical.x, accepted);
  secondKeys[secondKeysOffset + index] = select(NONE, canonical.y, accepted);
  if (accepted) { atomicAdd(&requiredCount[requiredCountOffset], 1u); }`})],...S.nodes];if(s.triangles.length>0&&w.push(p(t,{id:`${a}-materialize`,operation:b,variant:`parallel-materialization`,bindings:[{name:`triangleVertices`,view:m,type:`u32`,access:`read`},{name:`sortedRows`,view:S.sortedIndices,type:`u32`,access:`read`},{name:`requiredCount`,view:c.requiredCount,type:`u32`,access:`read`},{name:`trianglesOut`,view:s.triangles,type:`u32`,access:`read_write`}],invocationCount:s.triangles.length,declarations:C,body:`if (index >= requiredCount[requiredCountOffset]) { return; }
  let canonical = getCanonicalTriangle(sortedRows[sortedRowsOffset + index]);
  let destinationBase = trianglesOutOffset + 3u * index;
  trianglesOut[destinationBase] = canonical.x;
  trianglesOut[destinationBase + 1u] = canonical.y;
  trianglesOut[destinationBase + 2u] = canonical.z;`})),u){let e=s.groups;e.triangleOffsets&&w.push(...new r({id:`${a}-group-scan`,input:e.triangleCounts,output:e.triangleOffsets,mode:`exclusive`}).getCommandNodes(t)),w.push(p(t,{id:`${a}-group-status`,operation:b,variant:`group-status`,bindings:[{name:`invalidCount`,view:g,type:`u32`,access:`read`},{name:`groupInvalid`,view:i.groupInvalid,type:`u32`,access:`read`},{name:`converged`,view:e.status.converged,type:`u32`,access:`read_write`}],invocationCount:u,body:`converged[convergedOffset + index] =
    select(0u, 1u, invalidCount[invalidCountOffset] == 0u && groupInvalid[groupInvalidOffset + index] == 0u);`}))}return w.push(p(t,{id:`${a}-publish-status`,operation:b,variant:`publish-status`,bindings:[{name:`requiredCountIn`,view:c.requiredCount,type:`u32`,access:`read`},{name:`countOut`,view:c.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:c.overflow,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const OUTPUT_CAPACITY: u32 = ${s.triangles.length}u;`,body:`let requiredCount = requiredCountIn[requiredCountInOffset];
  countOut[countOutOffset] = min(requiredCount, OUTPUT_CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, requiredCount > OUTPUT_CAPACITY);`})),w}};function V(e,t){return e.split(`dispatch[dispatchOffset`).join(`${t}[${t}Offset`)}function H(e){return`
const POINT_COUNT: u32 = ${e.pointCount}u;
const SCRATCH_TRIANGLE_CAPACITY: u32 = ${e.scratchTriangleCapacity}u;
const NONE: u32 = 0xffffffffu;
// Adjacency references are 3 * triangle + edge; bit 31 marks a reference written fresh by the
// operation of the current phase.
const FRESH: u32 = 0x80000000u;
fn getVertex(triangle: u32, vertex: u32) -> u32 {
  return triangleVertices[triangleVerticesOffset + 4u * triangle + vertex];
}
fn getStamp(triangle: u32) -> u32 {
  return triangleVertices[triangleVerticesOffset + 4u * triangle + 3u];
}`}var U=`
// Real sites only: rows at or above POINT_COUNT are symbolic super vertices with no coordinates.
fn pointAt(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,W=`
fn writeTriangle(triangle: u32, a: u32, b: u32, c: u32, phase: u32) {
  let base = triangleVerticesOffset + 4u * triangle;
  triangleVertices[base] = a;
  triangleVertices[base + 1u] = b;
  triangleVertices[base + 2u] = c;
  triangleVertices[base + 3u] = phase;
}
fn writeAdjacency(triangle: u32, edge0: u32, edge1: u32, edge2: u32) {
  let base = triangleAdjacencyOffset + 4u * triangle;
  triangleAdjacency[base] = edge0;
  triangleAdjacency[base + 1u] = edge1;
  triangleAdjacency[base + 2u] = edge2;
  triangleAdjacency[base + 3u] = 0u;
}
fn getAdjacency(triangle: u32, edge: u32) -> u32 {
  return triangleAdjacency[triangleAdjacencyOffset + 4u * triangle + edge % 3u];
}
// The new owner of old edge 'edge' of a consumed triangle, read by the adjacency repair.
fn writeReplacement(triangle: u32, edge: u32, reference: u32) {
  replacement[replacementOffset + 3u * triangle + edge % 3u] = reference;
}
// The triangles that now cover a consumed triangle, read by the relocation.
fn writeChildren(triangle: u32, first: u32, second: u32, third: u32) {
  let base = childrenOffset + 4u * triangle;
  children[base] = first;
  children[base + 1u] = second;
  children[base + 2u] = third;
}
fn getReference(triangle: u32, edge: u32) -> u32 {
  return 3u * triangle + edge;
}`,G=`let location = pointLocations[pointLocationsOffset + index];
  if (location == NONE) { return; }
  let triangle = location >> 2u;
  let code = location & 3u;
  let priority = getInsertionPriority(index);
  if (votes[votesOffset + triangle] != priority) { return; }
  var neighbor = NONE;
  var neighborEdge = 0u;
  if (code < 3u) {
    let reference = getAdjacency(triangle, code);
    neighbor = reference / 3u;
    neighborEdge = reference % 3u;
    if (votes[votesOffset + neighbor] != priority) { return; }
  }
  let first = atomicAdd(&state[stateOffset], 2u);
  if (first + 2u > SCRATCH_TRIANGLE_CAPACITY) {
    atomicStore(&state[stateOffset + 1u], 1u);
    return;
  }
  let second = first + 1u;
  let phase = atomicLoad(&state[stateOffset + 3u]);
  let site = index;
  if (code == 3u) {
    // (a, b, c) -> (a, b, q), (b, c, q), (c, a, q); edge 0 of each is old edge 0, 1, 2.
    let a = getVertex(triangle, 0u);
    let b = getVertex(triangle, 1u);
    let c = getVertex(triangle, 2u);
    let old0 = getAdjacency(triangle, 0u);
    let old1 = getAdjacency(triangle, 1u);
    let old2 = getAdjacency(triangle, 2u);
    writeTriangle(triangle, a, b, site, phase);
    writeTriangle(first, b, c, site, phase);
    writeTriangle(second, c, a, site, phase);
    writeAdjacency(triangle, old0, FRESH | getReference(first, 2u), FRESH | getReference(second, 1u));
    writeAdjacency(first, old1, FRESH | getReference(second, 2u), FRESH | getReference(triangle, 1u));
    writeAdjacency(second, old2, FRESH | getReference(triangle, 2u), FRESH | getReference(first, 1u));
    writeReplacement(triangle, 0u, getReference(triangle, 0u));
    writeReplacement(triangle, 1u, getReference(first, 0u));
    writeReplacement(triangle, 2u, getReference(second, 0u));
    writeChildren(triangle, triangle, first, second);
  } else {
    // Edge a -> b of (a, b, c), shared with (b, a, d): (a, q, c), (q, b, c), (b, q, d), (q, a, d).
    let a = getVertex(triangle, code);
    let b = getVertex(triangle, (code + 1u) % 3u);
    let c = getVertex(triangle, (code + 2u) % 3u);
    let d = getVertex(neighbor, (neighborEdge + 2u) % 3u);
    let nextOld = getAdjacency(triangle, code + 1u);
    let previousOld = getAdjacency(triangle, code + 2u);
    let neighborNextOld = getAdjacency(neighbor, neighborEdge + 1u);
    let neighborPreviousOld = getAdjacency(neighbor, neighborEdge + 2u);
    writeTriangle(triangle, a, site, c, phase);
    writeTriangle(first, site, b, c, phase);
    writeTriangle(neighbor, b, site, d, phase);
    writeTriangle(second, site, a, d, phase);
    writeAdjacency(triangle, FRESH | getReference(second, 0u), FRESH | getReference(first, 2u), previousOld);
    writeAdjacency(first, FRESH | getReference(neighbor, 0u), nextOld, FRESH | getReference(triangle, 1u));
    writeAdjacency(neighbor, FRESH | getReference(first, 0u), FRESH | getReference(second, 2u), neighborPreviousOld);
    writeAdjacency(second, FRESH | getReference(triangle, 0u), neighborNextOld, FRESH | getReference(neighbor, 1u));
    writeReplacement(triangle, code + 1u, getReference(first, 1u));
    writeReplacement(triangle, code + 2u, getReference(triangle, 2u));
    writeReplacement(neighbor, neighborEdge + 1u, getReference(second, 1u));
    writeReplacement(neighbor, neighborEdge + 2u, getReference(neighbor, 2u));
    writeChildren(triangle, triangle, first, NONE);
    writeChildren(neighbor, neighbor, second, NONE);
  }
  pointLocations[pointLocationsOffset + index] = NONE;
  atomicSub(&state[stateOffset + 2u], 1u);
  atomicAdd(&state[stateOffset + 5u], 1u);`,K=`if (index >= min(state[stateOffset], SCRATCH_TRIANGLE_CAPACITY)) { return; }
  // The vote is the priority of one owned edge of this triangle, or of a neighbor's edge (then no
  // owned edge matches and this thread returns before reading anything else).
  let key = votes[votesOffset + index];
  var edge = 3u;
  for (var candidate = 0u; candidate < 3u; candidate++) {
    if (getFlipPriority(3u * index + candidate) == key) { edge = candidate; }
  }
  if (edge == 3u) { return; }
  let reference = getAdjacency(index, edge);
  let neighbor = reference / 3u;
  let neighborEdge = reference % 3u;
  if (votes[votesOffset + neighbor] != key) { return; }
  let phase = state[stateOffset + 3u] + 1u;
  let a = getVertex(index, edge);
  let b = getVertex(index, (edge + 1u) % 3u);
  let c = getVertex(index, (edge + 2u) % 3u);
  let d = getVertex(neighbor, (neighborEdge + 2u) % 3u);
  let nextOld = getAdjacency(index, edge + 1u);
  let previousOld = getAdjacency(index, edge + 2u);
  let neighborNextOld = getAdjacency(neighbor, neighborEdge + 1u);
  let neighborPreviousOld = getAdjacency(neighbor, neighborEdge + 2u);
  writeTriangle(index, c, a, d, phase);
  writeTriangle(neighbor, d, b, c, phase);
  writeAdjacency(index, previousOld, neighborNextOld, FRESH | getReference(neighbor, 2u));
  writeAdjacency(neighbor, neighborPreviousOld, nextOld, FRESH | getReference(index, 2u));
  writeReplacement(index, edge + 2u, getReference(index, 0u));
  writeReplacement(neighbor, neighborEdge + 1u, getReference(index, 1u));
  writeReplacement(neighbor, neighborEdge + 2u, getReference(neighbor, 0u));
  writeReplacement(index, edge + 1u, getReference(neighbor, 1u));
  writeChildren(index, index, neighbor, NONE);
  writeChildren(neighbor, index, neighbor, NONE);`,q=`
fn getCanonicalTriangle(triangle: u32) -> vec3<u32> {
  let a = getVertex(triangle, 0u);
  let b = getVertex(triangle, 1u);
  let c = getVertex(triangle, 2u);
  if (b < a && b < c) { return vec3<u32>(b, c, a); }
  if (c < a && c < b) { return vec3<u32>(c, a, b); }
  return vec3<u32>(a, b, c);
}`,J=`if (index >= min(state[stateOffset], SCRATCH_TRIANGLE_CAPACITY)) { return; }
  let phase = state[stateOffset + 3u] + PHASE_OFFSET;
  votes[votesOffset + index] = NONE;
  for (var edge = 0u; edge < 3u; edge++) {
    let reference = triangleAdjacency[triangleAdjacencyOffset + 4u * index + edge];
    if (reference == NONE) { continue; }
    var resolved = reference;
    if ((reference & FRESH) != 0u) {
      resolved = reference & ~FRESH;
    } else if (getStamp(reference / 3u) == phase) {
      resolved = replacement[replacementOffset + reference];
    }
    triangleAdjacency[triangleAdjacencyOffset + 4u * index + edge] = resolved;
  }`,Y=`let location = pointLocations[pointLocationsOffset + index];
  if (location == NONE) { return; }
  let triangle = location >> 2u;
  let phase = atomicLoad(&state[stateOffset + 3u]) + PHASE_OFFSET;
  if (getStamp(triangle) != phase) { return; }
  var found = NONE;
  var foundCode = 3u;
  for (var slot = 0u; slot < 3u; slot++) {
    let candidate = children[childrenOffset + 4u * triangle + slot];
    if (candidate == NONE || found != NONE) { continue; }
    var outside = false;
    var zeroEdge = 3u;
    for (var edge = 0u; edge < 3u; edge++) {
      let side = orientSymbolic(getVertex(candidate, edge), getVertex(candidate, (edge + 1u) % 3u), index);
      if (side < 0) { outside = true; }
      if (side == 0) { zeroEdge = edge; }
    }
    if (!outside) {
      found = candidate;
      foundCode = zeroEdge;
    }
  }
  if (found == NONE) {
    atomicStore(&state[stateOffset + 1u], 1u);
    return;
  }
  pointLocations[pointLocationsOffset + index] = (found << 2u) | foundCode;`,X=`if (index >= min(atomicLoad(&state[stateOffset]), SCRATCH_TRIANGLE_CAPACITY)) { return; }
  for (var edge = 0u; edge < 3u; edge++) {
    let reference = triangleAdjacency[triangleAdjacencyOffset + 4u * index + edge];
    if (reference == NONE || reference / 3u <= index) { continue; }
    let neighbor = reference / 3u;
    let a = getVertex(index, edge);
    let b = getVertex(index, (edge + 1u) % 3u);
    let c = getVertex(index, (edge + 2u) % 3u);
    let d = getVertex(neighbor, (reference % 3u + 2u) % 3u);
    if (isFlipRequired(a, b, c, d)) {
      let key = getFlipPriority(3u * index + edge);
      atomicMin(&votes[votesOffset + index], key);
      atomicMin(&votes[votesOffset + neighbor], key);
      atomicAdd(&state[stateOffset + 4u], 1u);
    }
  }`,Z=`let location = pointLocations[pointLocationsOffset + index];
  if (location == NONE) { return; }
  let triangle = location >> 2u;
  let code = location & 3u;
  let priority = getInsertionPriority(index);
  atomicMin(&votes[votesOffset + triangle], priority);
  if (code < 3u) {
    let reference = triangleAdjacency[triangleAdjacencyOffset + 4u * triangle + code];
    atomicMin(&votes[votesOffset + reference / 3u], priority);
  }`;export{y as a,C as i,B as n,w as o,L as r,M as t};
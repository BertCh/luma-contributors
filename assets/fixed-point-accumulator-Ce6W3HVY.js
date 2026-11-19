import{l as e}from"./gpu-command-graph-BdzdmV6u.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";function n(e){if(!Number.isSafeInteger(e)||e<0||e>4294967295)throw Error(`fixed-point maximumTermCount must be an integer in [0, 2^32 - 1]`);return Math.max(1,e.toString(2).length)}var r=`
fn getFixedPointExponentBits(value: f32) -> u32 {
  let magnitudeBits = bitcast<u32>(value) & 0x7fffffffu;
  if (magnitudeBits == 0u || magnitudeBits >= 0x7f800000u) { return 0u; }
  return max(magnitudeBits >> 23u, 1u);
}
fn getFixedPointUnitExponent(maximumExponentBits: u32, headroomBits: u32) -> i32 {
  // A term with biased exponent E is below 2^(E - 126) = 2^(E - 127 + 1); 95 - h bits remain.
  return i32(max(maximumExponentBits, 1u)) - 127 + 1 + i32(headroomBits) - 95;
}
fn addFixedPoint(a: vec3<u32>, b: vec3<u32>) -> vec3<u32> {
  let low = a.x + b.x;
  let lowCarry = select(0u, 1u, low < a.x);
  let middlePartial = a.y + b.y;
  let middleCarry = select(0u, 1u, middlePartial < a.y);
  let middle = middlePartial + lowCarry;
  let middleRoundCarry = select(0u, 1u, middle < middlePartial);
  return vec3<u32>(low, middle, a.z + b.z + middleCarry + middleRoundCarry);
}
fn negateFixedPoint(value: vec3<u32>) -> vec3<u32> {
  return addFixedPoint(~value, vec3<u32>(1u, 0u, 0u));
}
fn shiftFixedPointLeft(value: u32, shift: u32) -> vec3<u32> {
  var words = array<u32, 4>(0u, 0u, 0u, 0u);
  let word = shift / 32u;
  let bit = shift % 32u;
  words[word] = value << bit;
  // (value >> 1) >> (31 - bit) is value >> (32 - bit) without a shift by 32 when bit is 0.
  words[word + 1u] = (value >> 1u) >> (31u - bit);
  return vec3<u32>(words[0], words[1], words[2]);
}
fn shiftFixedPointRight(value: vec3<u32>, shift: u32) -> u32 {
  var words = array<u32, 4>(value.x, value.y, value.z, 0u);
  let word = shift / 32u;
  let bit = shift % 32u;
  return (words[word] >> bit) | ((words[word + 1u] << 1u) << (31u - bit));
}
fn getFixedPointBit(value: vec3<u32>, position: u32) -> u32 {
  var words = array<u32, 3>(value.x, value.y, value.z);
  return (words[position / 32u] >> (position % 32u)) & 1u;
}
fn hasFixedPointBitsBelow(value: vec3<u32>, count: u32) -> bool {
  var words = array<u32, 3>(value.x, value.y, value.z);
  let word = count / 32u;
  var found = (words[word] & ((1u << (count % 32u)) - 1u)) != 0u;
  if (word > 0u) { found = found || words[0] != 0u; }
  if (word > 1u) { found = found || words[1] != 0u; }
  return found;
}
fn quantizeFixedPoint(value: f32, unitExponent: i32) -> vec3<u32> {
  let bits = bitcast<u32>(value);
  let exponentBits = (bits >> 23u) & 0xffu;
  if (exponentBits == 0xffu) { return vec3<u32>(0u); }
  var significand = bits & 0x7fffffu;
  var exponent = -149;
  if (exponentBits != 0u) {
    significand = significand | 0x800000u;
    exponent = i32(exponentBits) - 150;
  }
  // value = significand * 2^exponent; the term in units is significand * 2^shift.
  let shift = exponent - unitExponent;
  var magnitude = vec3<u32>(0u);
  if (shift >= 0) {
    // The scale bounds shift by 71 - h; the clamp only keeps a misused scale memory-safe.
    magnitude = shiftFixedPointLeft(significand, u32(min(shift, 71)));
  } else if (shift >= -24) {
    let right = u32(-shift);
    var quotient = significand >> right;
    let remainder = significand & ((1u << right) - 1u);
    let half = 1u << (right - 1u);
    if (remainder > half || (remainder == half && (quotient & 1u) != 0u)) {
      quotient = quotient + 1u;
    }
    magnitude = vec3<u32>(quotient, 0u, 0u);
  }
  return select(magnitude, negateFixedPoint(magnitude), (bits >> 31u) != 0u);
}
fn getFixedPointFloat(value: vec3<u32>, unitExponent: i32) -> f32 {
  let negative = (value.z >> 31u) != 0u;
  let magnitude = select(value, negateFixedPoint(value), negative);
  var leading = 0u;
  if (magnitude.z != 0u) {
    leading = 64u + firstLeadingBit(magnitude.z);
  } else if (magnitude.y != 0u) {
    leading = 32u + firstLeadingBit(magnitude.y);
  } else if (magnitude.x != 0u) {
    leading = firstLeadingBit(magnitude.x);
  } else {
    return 0.0;
  }
  var significand = magnitude.x;
  var exponent = unitExponent;
  if (leading >= 24u) {
    let shift = leading - 23u;
    significand = shiftFixedPointRight(magnitude, shift);
    let roundBit = getFixedPointBit(magnitude, shift - 1u);
    if (roundBit != 0u && (hasFixedPointBitsBelow(magnitude, shift - 1u) || (significand & 1u) != 0u)) {
      significand = significand + 1u;
    }
    exponent = unitExponent + i32(shift);
  }
  // significand <= 2^24, so f32(significand) is exact and the value is significand * 2^exponent.
  var infinityBits = 0x7f800000u;
  var result = bitcast<f32>(infinityBits);
  if (exponent < 104 || (exponent == 104 && significand < 0x1000000u)) {
    // Two steps keep the ldexp exponent in range for normal results below 2^-103.
    let leadingExponent = i32(firstLeadingBit(significand));
    result = ldexp(ldexp(f32(significand), -leadingExponent), exponent + leadingExponent);
  }
  return select(result, -result, negative);
}`;function i(e){return`
fn ${`addFixedPointTo${e[0].toUpperCase()}${e.slice(1)}`}(cell: u32, value: vec3<u32>) {
  let base = ${e}Offset + cell * 3u;
  var lowCarry = 0u;
  if (value.x != 0u) {
    let oldLow = atomicAdd(&${e}[base], value.x);
    lowCarry = select(0u, 1u, oldLow + value.x < oldLow);
  }
  let middleAddend = value.y + lowCarry;
  var middleCarry = select(0u, 1u, middleAddend < value.y);
  if (middleAddend != 0u) {
    let oldMiddle = atomicAdd(&${e}[base + 1u], middleAddend);
    middleCarry = middleCarry + select(0u, 1u, oldMiddle + middleAddend < oldMiddle);
  }
  let highAddend = value.z + middleCarry;
  if (highAddend != 0u) {
    atomicAdd(&${e}[base + 2u], highAddend);
  }
}`}function a(n,i){let a=[t(n,{id:`${i.id}-clear`,operation:i.operation,variant:`fixed-point-scale-clear`,bindings:[{name:`fixedPointScale`,view:i.scale,type:`u32`,access:`read_write`}],invocationCount:1,body:`fixedPointScale[fixedPointScaleOffset] = 0u;`})],o=i.values instanceof e?i.values.data:[i.values];for(let[e,s]of o.entries())s.length!==0&&a.push(t(n,{id:`${i.id}-${e}`,operation:i.operation,variant:`fixed-point-scale`,bindings:[{name:`fixedPointValues`,view:s,type:`f32`,access:`read`},{name:`fixedPointScale`,view:i.scale,type:`atomic<u32>`,access:`read_write`}],invocationCount:s.length,guardIndex:!1,declarations:`${r}
var<workgroup> workgroupExponentBits: atomic<u32>;`,body:`if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let exponentBits = getFixedPointExponentBits(fixedPointValues[fixedPointValuesOffset + index]);
    if (exponentBits != 0u) {
      atomicMax(&workgroupExponentBits, exponentBits);
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) {
      atomicMax(&fixedPointScale[fixedPointScaleOffset], workgroupMaximum);
    }
  }`}));return a}function o(e,n){return t(e,{id:n.id,operation:n.operation,variant:`fixed-point-clear`,bindings:[{name:`fixedPointWords`,view:n.words,type:`u32`,access:`read_write`}],invocationCount:n.words.length,workgroupSize:64,body:`fixedPointWords[fixedPointWordsOffset + index] = 0u;`})}function s(e,i){if(i.words.length<i.output.length*3)throw Error(`${i.operation}: fixed-point words must hold three words per output`);return t(e,{id:i.id,operation:i.operation,variant:`fixed-point-decode`,bindings:[{name:`fixedPointWords`,view:i.words,type:`u32`,access:`read`},{name:`fixedPointScale`,view:i.scale,type:`u32`,access:`read`},{name:`fixedPointOutput`,view:i.output,type:`f32`,access:`read_write`}],invocationCount:i.output.length,workgroupSize:64,declarations:`override FIXED_POINT_HEADROOM_BITS: u32;
${r}`,constants:{FIXED_POINT_HEADROOM_BITS:n(i.maximumTermCount)},body:`let base = fixedPointWordsOffset + index * 3u;
  let unitExponent = getFixedPointUnitExponent(
    fixedPointScale[fixedPointScaleOffset],
    FIXED_POINT_HEADROOM_BITS
  );
  fixedPointOutput[fixedPointOutputOffset + index] = getFixedPointFloat(
    vec3<u32>(fixedPointWords[base], fixedPointWords[base + 1u], fixedPointWords[base + 2u]),
    unitExponent
  );`})}export{i as a,a as i,o as n,n as o,s as r,r as t};
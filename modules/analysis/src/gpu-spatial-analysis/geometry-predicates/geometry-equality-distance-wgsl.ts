// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Inclusive Euclidean comparison of stored f32 operands. Squared differences are expanded
 * into seven integer products, so division, sqrt rounding and compiler reassociation cannot
 * decide a boundary. The 576-bit accumulators cover the full finite f32 exponent range,
 * including subnormal bit patterns, with room for carries from the seven signed terms.
 */
export const GEOMETRY_EQUALITY_DISTANCE_WGSL = `
fn equalityMultiply24(first: u32, second: u32) -> vec2u {
  let firstHigh = first >> 12u;
  let firstLow = first & 0xfffu;
  let secondHigh = second >> 12u;
  let secondLow = second & 0xfffu;
  let highProduct = firstHigh * secondHigh;
  let middle = firstHigh * secondLow + firstLow * secondHigh;
  let lowProduct = firstLow * secondLow;
  let partial = lowProduct + ((middle & 0xfffffu) << 12u);
  let carryOne = select(0u, 1u, partial < lowProduct);
  let low = partial + (highProduct << 24u);
  let carryTwo = select(0u, 1u, low < partial);
  return vec2u(low, (highProduct >> 8u) + (middle >> 20u) + carryOne + carryTwo);
}

// Adds (high * 2^32 + low) * 2^shift to a 576-bit little-endian accumulator. high < 2^16.
fn equalityAccumulateShifted(accumulator: ptr<function, array<u32, 18>>, low: u32, high: u32, shift: u32) {
  let firstWord = shift / 32u;
  let bit = shift % 32u;
  var word0 = low << bit;
  var word1 = high << bit;
  var word2 = 0u;
  if (bit > 0u) {
    word1 = word1 | (low >> (32u - bit));
    word2 = high >> (32u - bit);
  }
  var carry = 0u;
  for (var slot = firstWord; slot < 18u; slot++) {
    let relative = slot - firstWord;
    var addend = 0u;
    if (relative == 0u) { addend = word0; } else if (relative == 1u) { addend = word1; } else if (relative == 2u) { addend = word2; } else if (carry == 0u) { break; }
    let partial = (*accumulator)[slot] + addend;
    let carryOne = select(0u, 1u, partial < addend);
    let total = partial + carry;
    let carryTwo = select(0u, 1u, total < partial);
    (*accumulator)[slot] = total;
    carry = carryOne | carryTwo;
  }
}


fn equalityNormalOrZero(bits:u32)->bool {
 let magnitude=bits&0x7fffffffu;
 return magnitude==0u || (magnitude>=0x00800000u && magnitude<0x7f800000u);
}
fn equalityStableDelta(first:u32,second:u32,difference:u32)->bool {
 // A zero arithmetic difference is reliable only when the stored values really coincide.
 return first==second || ((first&0x7fffffffu)==0u && (second&0x7fffffffu)==0u) ||
   ((difference&0x7fffffffu)>=0x00800000u && (difference&0x7fffffffu)<0x7f800000u);
}
fn geometryWithinTolerance(first:vec2f,second:vec2f,tolerance:f32)->bool {
 let words=vec4u(bitcast<vec2u>(first),bitcast<vec2u>(second));
 if(any((words&vec4u(0x7f800000u))==vec4u(0x7f800000u))) {return false;}
 let toleranceBits=bitcast<u32>(tolerance);
 if((toleranceBits&0x7fffffffu)>0x7f800000u || ((toleranceBits&0x80000000u)!=0u && (toleranceBits&0x7fffffffu)!=0u)) {return false;}
 if(toleranceBits==0x7f800000u) {return true;}
 // Normal deltas whose reciprocal remains normal admit a broad float filter. Leave a generous
 // absolute margin in normalized units for reciprocal division and reassociation.
 // Zero, subnormal, very large and overflowing deltas reach the integer certificate.
 let delta=first-second;let scale=max(abs(delta.x),abs(delta.y));
 let scaleExponent=(bitcast<u32>(scale)>>23u)&255u;
 let safeOperands=equalityNormalOrZero(words.x) && equalityNormalOrZero(words.y) &&
   equalityNormalOrZero(words.z) && equalityNormalOrZero(words.w) && equalityNormalOrZero(toleranceBits);
 let safeDeltas=equalityStableDelta(words.x,words.z,bitcast<u32>(delta.x)) &&
   equalityStableDelta(words.y,words.w,bitcast<u32>(delta.y));
 if(safeOperands && safeDeltas && scaleExponent>0u && scaleExponent<253u) {
   let normalized=delta/scale;
   let length=sqrt(dot(normalized,normalized));let relativeTolerance=tolerance/scale;
   if(length<relativeTolerance-0.00001) {return true;}
   if(length>relativeTolerance+0.00001) {return false;}
 }
 // (ax-bx)^2 + (ay-by)^2 - tolerance^2. Cross products have an extra factor two.
 let firstBits=array<u32,7>(words.x,words.z,words.x,words.y,words.w,words.y,toleranceBits);
 let secondBits=array<u32,7>(words.x,words.z,words.z,words.y,words.w,words.w,toleranceBits);
 var positive:array<u32,18>;var negative:array<u32,18>;
 for(var term=0u;term<7u;term++) {
   let left=firstBits[term];let right=secondBits[term];
   let leftMagnitude=left&0x7fffffffu;let rightMagnitude=right&0x7fffffffu;
   if(leftMagnitude==0u || rightMagnitude==0u) {continue;}
   let leftExponent=leftMagnitude>>23u;let rightExponent=rightMagnitude>>23u;
   let leftSignificand=select((leftMagnitude&0x7fffffu)|0x800000u,leftMagnitude,leftExponent==0u);
   let rightSignificand=select((rightMagnitude&0x7fffffu)|0x800000u,rightMagnitude,rightExponent==0u);
   let product=equalityMultiply24(leftSignificand,rightSignificand);
   let cross=term==2u || term==5u;
   let exponent=select(i32(leftExponent)-150,-149,leftExponent==0u)+select(i32(rightExponent)-150,-149,rightExponent==0u)+select(0,1,cross);
   let isNegative=(((left^right)>>31u)!=0u) != (cross || term==6u);
   if(isNegative) {equalityAccumulateShifted(&negative,product.x,product.y,u32(exponent+298));}
   else {equalityAccumulateShifted(&positive,product.x,product.y,u32(exponent+298));}
 }
 for(var slot=18u;slot>0u;slot--) {
   if(positive[slot-1u]!=negative[slot-1u]) {return positive[slot-1u]<negative[slot-1u];}
 }
 return true;
}`;

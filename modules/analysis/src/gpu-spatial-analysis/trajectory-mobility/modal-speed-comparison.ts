// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
function getModalFloat32Integer(value: number): bigint {
  const bits = new Uint32Array(new Float32Array([value]).buffer)[0];
  const exponent = (bits >>> 23) & 255;
  const significand = (bits & 0x7fffff) | (exponent === 0 ? 0 : 0x800000);
  return BigInt(significand) << BigInt(Math.max(exponent - 1, 0));
}
/** Compare length/duration with an exclusive bound without dividing or rounding the quotient. @internal */
export function isModalSpeedBelowBound(length: number, duration: number, bound: number): boolean {
  if (bound <= 0) return false;
  if (bound === Infinity) return true;
  return (
    getModalFloat32Integer(length) << 149n <
    getModalFloat32Integer(duration) * getModalFloat32Integer(bound)
  );
}
/** Exact comparison of a binary32 length with a product of two positive binary32 values. @internal */
export const MODAL_SPEED_COMPARISON_WGSL = `
// Round every nonnegative path-sum addition to nearest-even using integer guard/round/sticky
// bits. Metal cannot reassociate this into a different reduction or keep an intermediate sum.
fn modalAddLength(first:f32,second:f32)->f32 {
  let firstBits=bitcast<u32>(first); let secondBits=bitcast<u32>(second);
  let larger=max(firstBits,secondBits); let smaller=min(firstBits,secondBits);
  if(smaller==0u || larger>=0x7f800000u){return bitcast<f32>(larger);}
  let largerBiased=larger>>23u; let smallerBiased=smaller>>23u;
  var exponent=max(largerBiased,1u);
  let gap=exponent-max(smallerBiased,1u);
  let largerSignificand=(larger&0x7fffffu)|select(0u,0x800000u,largerBiased!=0u);
  let smallerSignificand=(smaller&0x7fffffu)|select(0u,0x800000u,smallerBiased!=0u);
  let expanded=smallerSignificand<<3u;
  var aligned=expanded;
  if(gap>=28u){aligned=1u;} else if(gap>0u){aligned=(expanded>>gap)|select(0u,1u,(expanded&((1u<<gap)-1u))!=0u);}
  var summed=(largerSignificand<<3u)+aligned;
  if(summed>=0x8000000u){summed=(summed>>1u)|(summed&1u);exponent++;}
  var significand=summed>>3u;
  let remainder=summed&7u;
  if(remainder>4u || (remainder==4u && (significand&1u)!=0u)){significand++;}
  if(significand>=0x1000000u){significand=significand>>1u;exponent++;}
  if(exponent>=255u){return bitcast<f32>(exponent<<23u);}
  if(significand<0x800000u){exponent=0u;}
  return bitcast<f32>((exponent<<23u)|(significand&0x7fffffu));
}

fn modalProduct(first: u32, second: u32) -> vec2<u32> {
  let firstHigh=first>>12u; let firstLow=first&4095u;
  let secondHigh=second>>12u; let secondLow=second&4095u;
  let middle=firstHigh*secondLow+firstLow*secondHigh;
  let lower=firstLow*secondLow+((middle&4095u)<<12u);
  let upper=firstHigh*secondHigh+(middle>>12u)+(lower>>24u);
  return vec2<u32>((lower&0xffffffu)|(upper<<24u),upper>>8u);
}
fn modalSignificand(bits:u32)->u32 { return (bits&0x7fffffu)|select(0u,0x800000u,(bits&0x7f800000u)!=0u); }
fn modalExponent(bits:u32)->i32 { let biased=(bits>>23u)&255u; return select(-149,i32(biased)-150,biased!=0u); }
fn modalSpeedBelowBound(length:f32,duration:f32,bound:f32)->bool {
  if(bound<=0.0){return false;}
  let lengthBits=bitcast<u32>(length);
  let durationBits=bitcast<u32>(duration);
  let boundBits=bitcast<u32>(bound);
  let lengthSignificand=modalSignificand(lengthBits);
  if(lengthSignificand==0u){return true;}
  let product=modalProduct(modalSignificand(durationBits),modalSignificand(boundBits));
  let lengthTop=31u-countLeadingZeros(lengthSignificand);
  var productTop=31u-countLeadingZeros(product.x);
  if(product.y!=0u){productTop=63u-countLeadingZeros(product.y);}
  let lengthOrder=modalExponent(lengthBits)+i32(lengthTop);
  let productOrder=modalExponent(durationBits)+modalExponent(boundBits)+i32(productTop);
  if(lengthOrder!=productOrder){return lengthOrder<productOrder;}
  let normalizedLength=lengthSignificand<<(31u-lengthTop);
  var productHigh=0u;
  var productLow=0u;
  if(productTop>=32u){
    let shift=63u-productTop;
    productHigh=(product.y<<shift)|(product.x>>(32u-shift));
    productLow=product.x<<shift;
  } else {productHigh=product.x<<(31u-productTop);}
  return normalizedLength<productHigh || (normalizedLength==productHigh && productLow!=0u);
}`;

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';

/** Certify an endpoint's circle tangency before any normalized/reciprocal arithmetic. */
export const OFFSET_ENDPOINT_TANGENCY_WGSL = `${EXACT_ORIENTATION_WGSL}
fn endpointTangent(delta:vec2f,direction:vec2f,radius:f32)->bool {
 if(orientSign(vec2f(0.0),delta,vec2f(-direction.y,direction.x))!=0) {return false;}
 let values=array<u32,3>(bitcast<u32>(delta.x)&0x7fffffffu,bitcast<u32>(delta.y)&0x7fffffffu,bitcast<u32>(radius)&0x7fffffffu);
 var minimum=1000;var maximum=-1000;
 for(var term=0u;term<3u;term++) {
  let bits=values[term];if(bits==0u) {continue;}
  if(bits<0x00800000u || bits>=0x7f800000u) {return false;}
  let exponent=2*(i32(bits>>23u)-150);minimum=min(minimum,exponent);maximum=max(maximum,exponent);
 }
 if(maximum-minimum>200) {return false;}
 if(minimum>maximum) {return true;}
 var positive:array<u32,8>;var negative:array<u32,8>;
 for(var term=0u;term<3u;term++) {
  let bits=values[term];if(bits==0u) {continue;}
  let significand=(bits&0x7fffffu)|0x800000u;let product=multiply24(significand,significand);
  let shift=u32(2*(i32(bits>>23u)-150)-minimum);
  if(term<2u) {accumulateShifted(&positive,product.x,product.y,shift);} else {accumulateShifted(&negative,product.x,product.y,shift);}
 }
 for(var word=0u;word<8u;word++) {if(positive[word]!=negative[word]) {return false;}}
 return true;
}`;

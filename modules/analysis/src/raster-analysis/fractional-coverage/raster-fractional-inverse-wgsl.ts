// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Product differences for the packed affine inverse, including nearly parallel axes. @internal */
export const RASTER_FRACTIONAL_INVERSE_WGSL = /* wgsl */ `
// Exact 24-bit significand multiplication, split into 12-bit limbs. Integer cancellation
// survives Metal fast-math, unlike floating two-product/fma residual expressions.
struct FractionalProduct {low:u32,high:u32}
fn fractionalMultiply24(first:u32,second:u32)->FractionalProduct {
 let highProduct=(first>>12u)*(second>>12u);
 let middle=(first>>12u)*(second&0xfffu)+(first&0xfffu)*(second>>12u);
 let lowProduct=(first&0xfffu)*(second&0xfffu);
 let partial=lowProduct+((middle&0xfffffu)<<12u);
 let low=partial+(highProduct<<24u);
 return FractionalProduct(low,(highProduct>>8u)+(middle>>20u)+u32(partial<lowProduct)+u32(low<partial));
}
// Values retain a normal signed significand and a separate power of two. In particular,
// a subnormal determinant is never materialized before reciprocal-multiply division.
struct FractionalValue {significand:f32,exponent:i32}
fn fractionalValue(value:f32)->FractionalValue {
 let bits=bitcast<u32>(value); let magnitude=bits&0x7fffffffu;
 if(magnitude==0u||magnitude>=0x7f800000u) {return FractionalValue(value,0);}
 let exponentBits=magnitude>>23u;
 if(exponentBits!=0u) {
   return FractionalValue(bitcast<f32>((bits&0x807fffffu)|0x3f800000u),i32(exponentBits)-127);
 }
 let highest=31u-countLeadingZeros(magnitude);
 return FractionalValue(bitcast<f32>((bits&0x80000000u)|0x3f800000u|((magnitude<<(23u-highest))&0x7fffffu)),i32(highest)-149);
}
fn fractionalProduct(left:f32,right:f32)->FractionalValue {
 let first=fractionalValue(left); let second=fractionalValue(right);
 return FractionalValue(first.significand*second.significand,first.exponent+second.exponent);
}
fn fractionalAlign(significand:f32,shift:i32)->f32 {
 let bits=bitcast<u32>(significand);
 // Invalid per-frame axes must stay nonfinite rather than acquiring a finite exponent.
 if((bits&0x7fffffffu)>=0x7f800000u) {return significand;}
 // Smaller terms cannot change a normal f32 sum, and cannot nearly cancel it.
 if(shift < -125) {return 0.0;}
 return bitcast<f32>((bits&0x807fffffu)|(u32(i32((bits>>23u)&255u)+shift)<<23u));
}
fn fractionalOrdinaryDifference(firstLeft:f32,firstRight:f32,secondLeft:f32,secondRight:f32)->FractionalValue {
 let first=fractionalProduct(firstLeft,firstRight); let second=fractionalProduct(secondLeft,secondRight);
 if(first.significand==0.0) {return FractionalValue(-second.significand,second.exponent);}
 if(second.significand==0.0) {return first;}
 let exponent=max(first.exponent,second.exponent);
 let value=fractionalAlign(first.significand,first.exponent-exponent)-fractionalAlign(second.significand,second.exponent-exponent);
 return FractionalValue(value,exponent);
}
fn fractionalProductDifference(firstLeft:f32,firstRight:f32,secondLeft:f32,secondRight:f32)->FractionalValue {
 let ordinary=fractionalOrdinaryDifference(firstLeft,firstRight,secondLeft,secondRight);
 let firstLeftBits=bitcast<u32>(firstLeft); let firstRightBits=bitcast<u32>(firstRight);
 let secondLeftBits=bitcast<u32>(secondLeft); let secondRightBits=bitcast<u32>(secondRight);
 let firstLeftMagnitude=firstLeftBits&0x7fffffffu; let firstRightMagnitude=firstRightBits&0x7fffffffu;
 let secondLeftMagnitude=secondLeftBits&0x7fffffffu; let secondRightMagnitude=secondRightBits&0x7fffffffu;
 // Opposite signs cannot cancel. Zero/nonfinite/subnormal inputs use the ordinary path.
 if(min(min(firstLeftMagnitude,firstRightMagnitude),min(secondLeftMagnitude,secondRightMagnitude))<0x00800000u||
    max(max(firstLeftMagnitude,firstRightMagnitude),max(secondLeftMagnitude,secondRightMagnitude))>=0x7f800000u||
    ((firstLeftBits^firstRightBits^secondLeftBits^secondRightBits)&0x80000000u)!=0u) {return ordinary;}
 let firstExponent=i32(firstLeftMagnitude>>23u)+i32(firstRightMagnitude>>23u)-300;
 let secondExponent=i32(secondLeftMagnitude>>23u)+i32(secondRightMagnitude>>23u)-300;
 // Normal products have 46..48 significant bits. Only products within two binary
 // orders can nearly cancel; their exact aligned difference fits in 50 bits.
 if(abs(firstExponent-secondExponent)>2) {return ordinary;}
 var firstProduct=fractionalMultiply24((firstLeftMagnitude&0x7fffffu)|0x800000u,(firstRightMagnitude&0x7fffffu)|0x800000u);
 var secondProduct=fractionalMultiply24((secondLeftMagnitude&0x7fffffu)|0x800000u,(secondRightMagnitude&0x7fffffu)|0x800000u);
 let exponent=min(firstExponent,secondExponent);
 let firstShift=u32(firstExponent-exponent); let secondShift=u32(secondExponent-exponent);
 if(firstShift>0u) {firstProduct=FractionalProduct(firstProduct.low<<firstShift,(firstProduct.high<<firstShift)|(firstProduct.low>>(32u-firstShift)));}
 if(secondShift>0u) {secondProduct=FractionalProduct(secondProduct.low<<secondShift,(secondProduct.high<<secondShift)|(secondProduct.low>>(32u-secondShift)));}
 let negative=firstProduct.high<secondProduct.high||(firstProduct.high==secondProduct.high&&firstProduct.low<secondProduct.low);
 var larger=firstProduct; var smaller=secondProduct;
 if(negative) {larger=secondProduct;smaller=firstProduct;}
 let low=larger.low-smaller.low; let high=larger.high-smaller.high-u32(larger.low<smaller.low);
 if(low==0u&&high==0u) {return FractionalValue(0.0,0);}
 // Normalize after exact subtraction, retaining the exponent separately until division.
 let roundedBits=bitcast<u32>(f32(high)*4294967296.0+f32(low));
 let productNegative=((firstLeftBits^firstRightBits)&0x80000000u)!=0u;
 let signBit=select(0u,0x80000000u,negative!=productNegative);
 return FractionalValue(bitcast<f32>(signBit|0x3f800000u|(roundedBits&0x7fffffu)),i32(roundedBits>>23u)+exponent-127);
}
fn fractionalRestore(value:FractionalValue)->f32 {
 let parts=fractionalValue(value.significand);
 let bits=bitcast<u32>(parts.significand); let signBit=bits&0x80000000u;
 if((bits&0x7fffffffu)==0u||(bits&0x7fffffffu)>=0x7f800000u) {return parts.significand;}
 let resultExponent=parts.exponent+value.exponent+127;
 if(resultExponent>=255) {return bitcast<f32>(signBit|0x7f800000u);}
 if(resultExponent<=0) {
   if(resultExponent < -23) {return bitcast<f32>(signBit);}
   let shift=u32(1-resultExponent); let significand=(bits&0x7fffffu)|0x800000u;
   let tail=significand&((1u<<shift)-1u); let halfway=1u<<(shift-1u);
   let leading=significand>>shift;
   return bitcast<f32>(signBit|(leading+u32(tail>halfway||(tail==halfway&&(leading&1u)!=0u))));
 }
 return bitcast<f32>(signBit|(u32(resultExponent)<<23u)|(bits&0x7fffffu));
}
fn fractionalDivide(numerator:FractionalValue,denominator:FractionalValue)->f32 {
 // Divide normal significands first; only then restore the common power-of-two scale.
 return fractionalRestore(FractionalValue(numerator.significand/denominator.significand,numerator.exponent-denominator.exponent));
}
`;

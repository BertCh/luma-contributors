import{t as e}from"./scaled-quotient-wgsl-UDyhQag6.js";var t=`
${e}
fn getSpatialAnalysisQuotient(value:f32,denominator:f32)->f32 {
  if(isFiniteValue(value)&&isFiniteValue(denominator)&&denominator!=0.0){
    return getScaledQuotient(value,denominator,1.0,0);
  }
  return value/denominator;
}`;export{t};
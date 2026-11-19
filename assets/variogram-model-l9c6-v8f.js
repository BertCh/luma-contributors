var e=4.749031386012701,t=5.924462618951797,n=.5;function r(e,t){let n=.5,r=n;for(let t=3;t<=18;t++)n*=e/t,r+=n;return e*e*Math.exp(-e)*(r-t)}function i(i,a){switch(i){case`spherical`:return a>=1?1:1.5*a-.5*a**3;case`exponential`:return-Math.expm1(-3*a);case`gaussian`:return-Math.expm1(-3*a*a);case`matern-3/2`:{let t=e*a;return t<n?r(t,0):1-(1+t)*Math.exp(-t)}case`matern-5/2`:{let e=t*a;return e<n?r(e,1/3):1-(1+e+e*e/3)*Math.exp(-e)}default:throw Error(`Unknown variogram model ${String(i)}`)}}var a=`
// 1 - exp(-argument) for argument >= 0, by the series
// argument - argument^2/2 + argument^3/6 - argument^4/24 + argument^5/120 below 0.1.
fn getVariogramExponentialShape(argument: f32) -> f32 {
  if (argument < 0.1) {
    return argument * (1.0 - argument * (0.5 - argument * ((1.0 / 6.0) - argument * ((1.0 / 24.0) - argument * (1.0 / 120.0)))));
  }
  return 1.0 - exp(-argument);
}
fn getVariogramModelShape(modelCode: u32, scaled: f32) -> f32 {
  if (modelCode == 0u) {
    return select(1.5 * scaled - 0.5 * scaled * scaled * scaled, 1.0, scaled >= 1.0);
  }
  if (modelCode == 1u) { return getVariogramExponentialShape(3.0 * scaled); }
  if (modelCode == 2u) { return getVariogramExponentialShape(3.0 * scaled * scaled); }
  if (modelCode == 3u) {
    // 1 - (1 + a) exp(-a) = a^2/2 - a^3/3 + a^4/8 - a^5/30 + a^6/144 - ...
    let argument = ${e} * scaled;
    if (argument < 0.1) {
      return argument * argument * (0.5 - argument * ((1.0 / 3.0) - argument * (0.125 - argument * ((1.0 / 30.0) - argument * (1.0 / 144.0)))));
    }
    return 1.0 - (1.0 + argument) * exp(-argument);
  }
  // 1 - (1 + a + a^2/3) exp(-a) = a^2/6 - a^4/24 + a^5/45 - a^6/144 + ...
  let argument = ${t} * scaled;
  if (argument < 0.1) {
    return argument * argument * ((1.0 / 6.0) - argument * argument * ((1.0 / 24.0) - argument * ((1.0 / 45.0) - argument * (1.0 / 144.0))));
  }
  return 1.0 - (1.0 + argument + argument * argument / 3.0) * exp(-argument);
}
`;export{i,t as n,a as r,e as t};
import{n as e}from"./temporal-context-xddVhAJD.js";import{E as t,T as n,x as r}from"./gpu-contributor-utils-BIwLQON5.js";import{t as i}from"./time-words-BFl8KCxq.js";var a=31556952,o={second:1,millisecond:1e3,microsecond:1e6,nanosecond:1e9};function s(r,i,a){let o=e(r,i.temporalContext,{encodings:[`int64-words`],axisKinds:[`irregular`]});return t(i.acquisitionTimes,[`uint32x2`],`${r} dateAxis.acquisitionTimes`),n(r,i.acquisitionTimes.length>=a,`dateAxis.acquisitionTimes must hold one row per slice`),l(o.unit)}function c(e,t,n){r(e,t,[n.acquisitionTimes])}function l(e){return a*o[e]}function u(e,t){let n=l(t),r=e[0]+(e[e.length-1]-e[0])/2n;return Float32Array.from(e,e=>Number(e-r)/n)}function d(e){return{name:`acquisitionTimes`,view:e.acquisitionTimes,type:`u32`,access:`read`}}function f(e,t){return`${i}
const RASTER_AXIS_TICKS_PER_YEAR: f32 = ${e}.0;
fn rasterAxisWords(slice: u32) -> vec2<u32> {
  return vec2<u32>(acquisitionTimes[acquisitionTimesOffset + 2u * slice], acquisitionTimes[acquisitionTimesOffset + 2u * slice + 1u]);
}
fn rasterAxisMidpoint() -> vec2<u32> {
  let first = rasterAxisWords(0u);
  let difference = timeWordsSubtract(rasterAxisWords(${t-1}u), first);
  let low = first.x + ((difference.x >> 1u) | (difference.y << 31u));
  return vec2<u32>(low, first.y + (difference.y >> 1u) + select(0u, 1u, low < first.x));
}
fn rasterSliceYears(slice: u32) -> f32 {
  return timeWordsDifferenceToF32(rasterAxisWords(slice), rasterAxisMidpoint()) / RASTER_AXIS_TICKS_PER_YEAR;
}
fn rasterYearsBetween(later: u32, earlier: u32) -> f32 {
  return timeWordsDifferenceToF32(rasterAxisWords(later), rasterAxisWords(earlier)) / RASTER_AXIS_TICKS_PER_YEAR;
}
`}export{s as a,f as i,u as n,c as o,d as r,a as t};
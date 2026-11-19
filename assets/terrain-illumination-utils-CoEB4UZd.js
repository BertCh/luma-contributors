import{E as e}from"./gpu-contributor-utils-BIwLQON5.js";import{c as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as n,t as r}from"./raster-grid-VB0AM_GY.js";import{a as i,u as a}from"./wgsl-common-DfyXiYB0.js";var o=`
const PI: f32 = 3.141592653589793;
${r}
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
${a}
// getQuietNaN(seed) takes a runtime operand (unlike getNaN()) so the NaN is not constant-folded away.
${i}`;function s(e,t){if(!n(t).cellSize.every(e=>Number.isFinite(e)&&e>0))throw Error(`${e} grid cell size must be finite and positive`)}function c(t,n,r,i,a){if(r&&(e(r,[i],`${t} ${n}`),r.length!==a))throw Error(`${t} ${n} must contain ${a} ${i} values`)}function l(e,t,n,r){if(n>r)throw Error(`${e} ${t} needs ${n} bytes in one binding, above the device limit ${r}; use fewer directions or a smaller tile`)}var u=180/65534,d=Math.fround(65534/180);function f(e,t){if(![`float32`,`unorm16`].includes(t))throw Error(`${e} horizonFormat must be float32 or unorm16`)}function p(e,t,n){let r=t*n;return e===`unorm16`?Math.ceil(r/2):r}function m(e,t,n,r,i,a){c(e,t,n,r===`unorm16`?`uint32`:`float32`,p(r,i,a))}function h(e,n){return e===`float32`?`fn readHorizonAngle(element: u32) -> f32 {
  return ${n}[${n}Offset + element];
}`:`fn readHorizonAngle(element: u32) -> f32 {
  let word = ${n}[${n}Offset + (element >> 1u)];
  let code = (word >> ((element & 1u) * 16u)) & 0xffffu;
  if (code == 0u) { return getQuietNaN(element); }
  return f32(code - 1u) * ${t(u)} - 90.0;
}`}function g(e,n,r){return`const DIRECTION_COUNT: u32 = ${r}u;
const SECTOR_DEGREES: f32 = ${t(360/r)};
${h(e,n)}
fn getHorizonAtAzimuth(pixel: u32, azimuthDegrees: f32) -> f32 {
  let wrappedAzimuth = azimuthDegrees - 360.0 * floor(azimuthDegrees / 360.0);
  let sectorPosition = wrappedAzimuth / SECTOR_DEGREES;
  let lowerSector = min(u32(floor(sectorPosition)), DIRECTION_COUNT - 1u);
  let upperSector = (lowerSector + 1u) % DIRECTION_COUNT;
  let blend = clamp(sectorPosition - f32(lowerSector), 0.0, 1.0);
  return mix(
    readHorizonAngle(pixel * DIRECTION_COUNT + lowerSector),
    readHorizonAngle(pixel * DIRECTION_COUNT + upperSector),
    blend
  );
}
fn getSolarDiskVisibility(altitudeDegrees: f32, radiusDegrees: f32, horizonAngle: f32) -> f32 {
  var visibility = 0.0;
  if (altitudeDegrees + radiusDegrees > 0.0) {
    if (radiusDegrees > 0.0) {
      let u = clamp((altitudeDegrees - horizonAngle) / radiusDegrees, -1.0, 1.0);
      visibility = clamp(1.0 - (acos(u) - u * sqrt(max(1.0 - u * u, 0.0))) / PI, 0.0, 1.0);
    } else {
      visibility = select(0.0, 1.0, altitudeDegrees > horizonAngle);
    }
  }
  return visibility;
}`}export{g as a,l as c,p as i,s as l,o as n,f as o,h as r,m as s,d as t,c as u};
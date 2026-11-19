import{i as e,r as t}from"./earth-DODowzP2.js";import{t as n}from"./web-mercator-B0kaE2A1.js";import{T as r,f as i}from"./gpu-contributor-utils-BIwLQON5.js";import{c as a}from"./wgsl-kernel-nodes-DagJklH2.js";var o=t*(2-t),s=e*Math.PI/180,c=.017453292519943295,l=`const DEGREES_TO_RADIANS: f32 = ${c};`;function u(e){let[t,n,,r,i,a]=e.affine,o=e.pixelInterpretation===`point`?-.5:0;return{cellSize:[Math.hypot(t,r),Math.hypot(n,i)],northEdge:i*o+a,southEdge:i*(e.height+o)+a}}function d(e,t,n){let{cellSize:r,northEdge:i,southEdge:a}=u(t);e[n.cellSizeIndex]=r[0],e[n.cellSizeIndex+1]=r[1],e[n.northEdgeIndex]=i,e[n.northEdgeIndex+1]=a}function f(e,t){if(e===`uniform`)return[1,1];if(e===`web-mercator`){let e=t/n,r=Math.tanh(e),i=1-o*r*r,a=1/Math.sqrt(i),s=1/Math.cosh(e);return[a*s,a*(1-o)/i*s]}let r=t*Math.PI/180,i=Math.sin(r),a=1-o*i*i,c=s/Math.sqrt(a);return[c*Math.cos(r),c*(1-o)/a]}function p(e){let t=a(o),r=a(1-o);return e===`web-mercator`?`// WGS84 ground meters per EPSG:3857 meter at the row latitude atan(sinh(y / a)):
  // x = N cos(lat) / a, y = M cos(lat) / a, with cos(lat) = 1 / cosh(y / a).
  let mercatorY = edge * ${a(1/n)};
  let sinLatitude = tanh(mercatorY);
  let curvature = 1.0 - ${t} * sinLatitude * sinLatitude;
  let primeVertical = 1.0 / sqrt(curvature);
  let groundScale = vec2<f32>(primeVertical, primeVertical * ${r} / curvature) / cosh(mercatorY);`:`// WGS84 meters per degree at the row latitude: x = N cos(lat) PI / 180, y = M PI / 180.
  let latitude = edge * ${a(c)};
  let sinLatitude = sin(latitude);
  let curvature = 1.0 - ${t} * sinLatitude * sinLatitude;
  let primeVertical = ${a(s)} / sqrt(curvature);
  let groundScale = vec2<f32>(primeVertical * cos(latitude), primeVertical * ${r} / curvature);`}function m(e,t=`f32`){let n=e===`south`?-1:1;return`const ROW_NORTH_SIGN: ${t} = ${t===`f32`?`${n}.0`:n};`}function h(e,t){let n=t.settingsName??`settings`,r=`${n}Offset`,i=`vec2<f32>(${n}[${r} + ${t.cellSizeIndex}u], ${n}[${r} + ${t.cellSizeIndex+1}u])`;return e===`uniform`?`fn getGroundCellSize(row: u32) -> vec2<f32> {
  let cellSize = ${i};
  return cellSize;
}`:`fn getGroundCellSize(row: u32) -> vec2<f32> {
  let cellSize = ${i};
  let rowFraction = (f32(row) + 0.5) / f32(HEIGHT);
  let edge = ${`mix(${n}[${r} + ${t.northEdgeIndex}u], ${n}[${r} + ${t.northEdgeIndex+1}u], rowFraction)`};
  ${p(e)}
  return cellSize * groundScale;
}`}function g(e,t,n,r){let[i,a]=t.cellSize;if(e===`uniform`)return[i,a];let o=t.northEdge??0,[s,c]=f(e,o+((t.southEdge??0)-o)*(n/r));return[i*s,a*c]}var _=`
// Unit pixel step along a ground-space unit direction on cells of groundCell meters.
fn getPixelDirection(groundDirection: vec2<f32>, groundCell: vec2<f32>) -> vec2<f32> {
  if (groundCell.x == groundCell.y) { return groundDirection; }
  return normalize(groundDirection / groundCell);
}`;function v(e,t){if(![`uniform`,`web-mercator`,`geographic`].includes(t??`uniform`))throw Error(`${e} cellSizeMode must be uniform, web-mercator, or geographic`)}function y(e,t){if(![`south`,`north`].includes(t??`south`))throw Error(`${e} rowDirection must be south or north`)}function b(e,t){let{grid:n}=t;i(e,`grid.width`,n.width,1),i(e,`grid.height`,n.height,1),i(e,`grid cells`,n.width*n.height,1),r(e,n.affine.length===6&&n.affine.every(Number.isFinite),`grid.affine must contain six finite values`),r(e,[`area`,`point`].includes(n.pixelInterpretation),`grid.pixelInterpretation must be area or point`),v(e,t.cellSizeMode);let a=t.cellSizeMode??`uniform`,[o,s,c,l,d,f]=n.affine,p=o*d-s*l;r(e,Number.isFinite(p)&&p!==0,`grid.affine must be invertible (finite nonzero determinant)`),r(e,a===`uniform`||s===0&&l===0,`grid.affine must be axis-aligned for geographic or web-mercator cellSizeMode`);let m=t.localOrigin??[0,0];r(e,m.length===2&&m.every(Number.isFinite),`localOrigin must contain two finite coordinates`);let h=Float32Array.of(c-m[0],f-m[1],o,s,l,d,p,0),g=h[2]*h[5]-h[3]*h[4];return r(e,h.every(Number.isFinite)&&h[6]!==0&&g!==0,`grid.affine relative to localOrigin must be finite and invertible in float32`),{cellCount:n.width*n.height,cellSizeMode:a,localOrigin:m,georeference:u(n),parameterValues:h}}export{h as a,m as c,y as d,d as f,g as i,b as l,_ as n,f as o,u as r,p as s,l as t,v as u};
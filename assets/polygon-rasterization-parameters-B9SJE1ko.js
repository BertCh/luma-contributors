import{u as e}from"./wgsl-common-DfyXiYB0.js";import{o as t}from"./raster-band-utils-BcTxc8_Y.js";var n=4294967295;function r(e,n=new Float32Array(4)){t(`getGPUPolygonRasterizationParameterValues`,n,4);let[r,i,a,o,s,c]=e.affine;if(i!==0||o!==0)throw Error(`getGPUPolygonRasterizationParameterValues needs an axis-aligned grid`);let l=e.pixelInterpretation===`point`?-.5:0,u=a+r*l,d=c+s*l;return n[0]=u,n[1]=s<0?d+s*e.height:d,n[2]=r,n[3]=s,n}var i=`
${e}
struct RasterExtent { origin: vec2<f32>, cellSize: vec2<f32>, valid: bool, flipped: bool };
fn getRasterExtent() -> RasterExtent {
  let origin = vec2<f32>(extent[extentOffset], extent[extentOffset + 1u]);
  // A negative cell height means rows advance toward smaller y: the kernels count rows from the
  // smallest y and mirror the row index when they read or write the caller's rasters.
  let signedCellHeight = extent[extentOffset + 3u];
  let cellSize = vec2<f32>(extent[extentOffset + 2u], abs(signedCellHeight));
  let valid = isFiniteValue(origin.x) && isFiniteValue(origin.y) &&
    isFiniteValue(cellSize.x) && isFiniteValue(cellSize.y) && cellSize.x > 0.0 && cellSize.y > 0.0;
  return RasterExtent(origin, cellSize, valid, signedCellHeight < 0.0);
}
// Stored row of a row counted from the smallest y. Needs a HEIGHT constant.
fn getStoredRow(row: u32, rasterExtent: RasterExtent) -> u32 {
  return select(row, HEIGHT - 1u - row, rasterExtent.flipped);
}`;function a(e,t,n){return`
fn ${e}(value: u32) -> u32 {
  // Largest i in [0, rangeCount) with offsets[i] <= value.
  var low = 0u;
  var high = ${n}u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (${t}[${t}Offset + middle] <= value) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u) {
    return 0xffffffffu;
  }
  let range = low - 1u;
  if (value >= ${t}[${t}Offset + range + 1u]) {
    return 0xffffffffu;
  }
  return range;
}`}export{a as i,i as n,r,n as t};
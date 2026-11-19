import{l as e}from"./gpu-command-graph-BdzdmV6u.js";import{c as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{d as n}from"./kernel-C3tfQOn4.js";import{r,u as i}from"./wgsl-common-DfyXiYB0.js";import{i as a,o,u as s}from"./raster-band-utils-BcTxc8_Y.js";import{a as c}from"./raster-algebra-input-5ZxymrFJ.js";var l={nearest:0,bilinear:1,bicubic:2};function u(e,t,n,r,i){o(e,i,12),s(e,n);let{width:c,height:u}=n,[d,f,p,m,h,g]=n.affine;if(f!==0||m!==0)throw Error(`${e} needs an axis-aligned grid`);if(!(d>0)||!(Math.abs(h)>0))throw Error(`${e} grid cell size must be finite and positive`);let _=n.pixelInterpretation===`point`?-.5:0,v=p+d*_,y=g+h*_,b=v,x=v+d*c,S=y+h*u,C=Math.min(y,S),w=Math.max(y,S),T=t.method??`bilinear`;if(!(T in l))throw Error(`${e} method ${T} is not supported`);let E=t.noDataPolicy??`strict`;if(E!==`strict`&&E!==`renormalize`)throw Error(`${e} noDataPolicy ${E} is not supported`);let[D,O]=a(e,t.localOrigin);return i.fill(0,0,12),i[0]=b-D,i[1]=C-O,i[2]=x-D,i[3]=w-O,i[4]=Math.fround(d),i[5]=Math.fround(Math.abs(h)),i[6]=1/d,i[7]=1/h,i[8]=l[T],i[9]=+(E===`renormalize`),i[10]=r,i[11]=y-O,i}function d(e,t,n=new Float32Array(12)){return u(`getGPURasterSamplingParameterValues`,e,t,0,n)}function f(e,t,n=new Float32Array(12)){if(!(e.spacing>0)||!Number.isFinite(e.spacing))throw Error(`getGPURasterProfileParameterValues spacing must be finite and positive`);return u(`getGPURasterProfileParameterValues`,e,t,e.spacing,n)}function p(e){return c(e)?{"values.storage.values":e.storage.kind===`buffer`?e.storage.values:void 0,"values.validity":e.validity}:{values:e}}function m(t,r,i,a,o){if(!i)throw Error(`${t} needs ${r}`);if(i instanceof e)throw Error(`${t} ${r} must be a single packed view, not a chunked vector`);if(n(i,[a],`${t} ${r}`),i.length<o)throw Error(`${t} ${r} must hold at least ${o} rows`)}function h(e,t,n){let r=s(e,t);if(r>2147483647)throw Error(`${e} grid width * height must not exceed 2^31 - 1`);if(n!==void 0&&!Number.isFinite(n))throw Error(`${e} noDataValue must be finite (NaN cells are always nodata)`);return r}function g(e){let{width:n,height:a,noDataValue:o,hasValidity:s}=e;return`
const RASTER_WIDTH: i32 = ${n};
const RASTER_HEIGHT: i32 = ${a};
${r}
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
${i}

// Cell value with every nodata convention folded into NaN.
fn fetchCell(column: i32, row: i32) -> f32 {
  let cell = u32(row * RASTER_WIDTH + column);
  var value = raster[rasterOffset + cell];
  // An infinity is nodata like NaN (the published band rule): a weighted sum cannot interpolate it.
  if (!isFiniteValue(value)) { value = getNaN(); }
  ${s?`if (validity[validityOffset + cell] == 0u) { value = getNaN(); }`:``}
  ${o===void 0?``:`if (value == ${t(o)}) { value = getNaN(); }`}
  return value;
}
fn fetchClamped(column: i32, row: i32) -> f32 {
  return fetchCell(clamp(column, 0, RASTER_WIDTH - 1), clamp(row, 0, RASTER_HEIGHT - 1));
}

fn getCubicWeights(t: f32) -> vec4<f32> {
  return vec4<f32>(
    t * (-0.5 + t * (1.0 - 0.5 * t)),
    1.0 + t * t * (-2.5 + 1.5 * t),
    t * (0.5 + t * (2.0 - 1.5 * t)),
    t * t * (-0.5 + 0.5 * t)
  );
}

// A sample participates only when both of its axis weights are nonzero.
fn sampleBilinear(iu: i32, iv: i32, fx: f32, fy: f32, renormalize: bool) -> f32 {
  var wx = array<f32, 2>(1.0 - fx, fx);
  var wy = array<f32, 2>(1.0 - fy, fy);
  var sum = 0.0;
  var weightSum = 0.0;
  var hasNoData = false;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      if (wx[i] != 0.0 && wy[j] != 0.0) {
        let value = fetchClamped(iu + i, iv + j);
        let weight = wx[i] * wy[j];
        if (isNaNValue(value)) {
          hasNoData = true;
        } else {
          sum += weight * value;
          weightSum += weight;
        }
      }
    }
  }
  if (!hasNoData) {
    return sum;
  }
  if (renormalize && weightSum > 0.0) {
    return sum / weightSum;
  }
  return getNaN();
}

fn sampleBicubic(iu: i32, iv: i32, fx: f32, fy: f32, renormalize: bool) -> f32 {
  let cx = getCubicWeights(fx);
  let cy = getCubicWeights(fy);
  var wx = array<f32, 4>(cx.x, cx.y, cx.z, cx.w);
  var wy = array<f32, 4>(cy.x, cy.y, cy.z, cy.w);
  var cells = array<f32, 16>();
  var hasNoData = false;
  for (var j = 0; j < 4; j++) {
    for (var i = 0; i < 4; i++) {
      if (wx[i] != 0.0 && wy[j] != 0.0) {
        let value = fetchClamped(iu - 1 + i, iv - 1 + j);
        cells[j * 4 + i] = value;
        if (isNaNValue(value)) {
          hasNoData = true;
        }
      }
    }
  }
  if (hasNoData) {
    return sampleBilinear(iu, iv, fx, fy, renormalize);
  }
  var result = 0.0;
  for (var j = 0; j < 4; j++) {
    if (wy[j] != 0.0) {
      var row = 0.0;
      for (var i = 0; i < 4; i++) {
        if (wx[i] != 0.0) {
          row += wx[i] * cells[j * 4 + i];
        }
      }
      result += wy[j] * row;
    }
  }
  return result;
}

fn sampleRaster(x: f32, y: f32) -> f32 {
  if (isNaNValue(x) || isNaNValue(y)) {
    return getNaN();
  }
  let minX = params[paramsOffset];
  let minY = params[paramsOffset + 1u];
  if (x < minX || x > params[paramsOffset + 2u] || y < minY || y > params[paramsOffset + 3u]) {
    return getNaN();
  }
  let method = u32(params[paramsOffset + 8u]);
  let renormalize = params[paramsOffset + 9u] != 0.0;
  // The row coordinate counts from the outer edge of row 0 with a signed step, so it is correct
  // for rows that advance toward larger or smaller y.
  let tx = (x - minX) * params[paramsOffset + 6u];
  let ty = (y - params[paramsOffset + 11u]) * params[paramsOffset + 7u];
  if (method == 0u) {
    return fetchClamped(i32(floor(tx)), i32(floor(ty)));
  }
  let u = tx - 0.5;
  let v = ty - 0.5;
  let u0 = floor(u);
  let v0 = floor(v);
  if (method == 1u) {
    return sampleBilinear(i32(u0), i32(v0), u - u0, v - v0, renormalize);
  }
  return sampleBicubic(i32(u0), i32(v0), u - u0, v - v0, renormalize);
}`}export{f as a,m as i,g as n,d as o,h as r,p as t};
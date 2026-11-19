import{n as e,r as t}from"./terrain-binding-preflight-laSEObQE.js";import{c as n,o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{x as i}from"./gpu-contributor-utils-BIwLQON5.js";import{c as a,h as o,l as s,n as c,o as l,r as u,u as d}from"./raster-band-utils-BcTxc8_Y.js";import{d as f,r as p}from"./kernel-C3tfQOn4.js";import{n as m,t as h}from"./vertical-datum-BTN03tZP.js";import{a as g,i as _,n as v,r as y}from"./gpu-raster-extrema-pyramid-BYXR99nY.js";import{i as b,n as x,r as S}from"./earth-DODowzP2.js";import{a as C}from"./terrain-geomorphometry-utils-BhBw8dIY.js";import{t as w}from"./web-mercator-B0kaE2A1.js";import{r as T}from"./raster-grid-VB0AM_GY.js";import{u as E}from"./wgsl-common-DfyXiYB0.js";var D={planar:1,"web-mercator":2},O=x;function k(e,t,n,r,i){let a=t.projection??`planar`;if(a!==`planar`&&a!==`web-mercator`)throw Error(`${e} projection must be planar or web-mercator`);r[i]=D[a];let{cellSize:o,northEdge:s}=T(n);if(!o.every(e=>Number.isFinite(e)&&e>0))throw Error(`${e} grid cell size must be finite and positive`);if(a===`web-mercator`){let[t,i]=o;if(Math.abs(t-i)>1e-6*t||n.affine[4]>=0)throw Error(`${e} web-mercator grid needs square pixels with rows increasing south`);r[1]=2*Math.PI*w/t,r[2]=s/w}else r[1]=o[0],r[2]=o[1];r[0]=t.curvatureCoefficient??0,r[3]=t.maximumDistance??0,r[4]=0}function A(e,t,n=new Float32Array(8)){let r=`getGPUPointHorizonProfileParameterValues`;return l(r,n,8),n.fill(0,0,8),k(r,e,t,n,7),n}function j(e,t,n=new Float32Array(12)){let r=`getGPUPointHorizonVisibilityParameterValues`;l(r,n,12),n.fill(0,0,12),k(r,e,t,n,11),n[5]=e.toleranceDegrees??.02,n[6]=e.sigmaZ??5,n[7]=e.skylineToleranceDegrees??.05;let i=e.targetIgnoreDistance;if(i!==void 0&&!(Number.isFinite(i)&&i>=0))throw Error(`${r} targetIgnoreDistance must be finite and non-negative`);return n[8]=i??-1,n[9]=e.targetIgnoreFraction??0,n}var M=1<<24,N=-8;function P(e){let t=Math.floor(Math.log2(e));for(;2**(t+1)<=e;)t++;for(;2**t>e;)t--;return t}function F(e){let t=e.minimumDistance??20,{maximumDistance:n,cellSize:r}=e,i=e.stepFactor??35e-5,a=e.nearFactor??.01,o=e.cellSteps??.5,s=e.minimumStep??.25;for(let[e,c]of[[`minimumDistance`,t],[`maximumDistance`,n],[`cellSize`,r],[`stepFactor`,i],[`nearFactor`,a],[`cellSteps`,o],[`minimumStep`,s]])if(!(Number.isFinite(c)&&c>0))throw Error(`Point horizon ${e} must be finite and positive`);if(t<1)throw Error(`Point horizon minimumDistance must be at least 1 meter`);if(n<t)throw Error(`Point horizon maximumDistance must not be below minimumDistance`);let c=P(t),l=P(n),u=[],d=0;for(let e=c;e<=l;e++){let f=2**e,p=Math.max(i*f,Math.min(Math.max(a*f,s),r*o)),m=Math.min(Math.max(P(p),N),e);if(e-m>22)throw Error(`Point horizon distance lattice is too fine; increase stepFactor`);let h=2**m,g=f/h,_=e===c?Math.ceil((t-f)/h):0,v=e===l?Math.floor((n-f)/h)+1:g,y=Math.min(v,g);if(!(y<=_)&&(u.push({exponent:e,base:f,step:h,firstJ:_,endJ:y,firstIndex:d}),d+=y-_,d>M))throw Error(`Point horizon distance lattice has too many samples`)}if(d===0)throw Error(`Point horizon distance lattice is empty`);let f=e=>{if(!Number.isInteger(e)||e<0||e>=d)throw RangeError(`Point horizon lattice index ${e} out of range`);let t=u[0];for(let n of u)e>=n.firstIndex&&(t=n);return t.base+(e-t.firstIndex+t.firstJ)*t.step},p=e=>{let t=0,n=d;for(;t<n;){let r=t+n>>>1;f(r)>=e?n=r:t=r+1}return t};return{sampleCount:d,octaves:u,minimumDistance:t,maximumDistance:n,firstDistance:f(0),lastDistance:f(d-1),getDistance:f,getCeilIndex:p,getFloorIndex:e=>{let t=p(e);return t<d&&f(t)===e?t:t-1}}}function I(e,t={}){let n=t.maximumLatitude??80,r=t.segmentTolerance??2e-5;if(!(Number.isFinite(n)&&n>=0&&n<=90))throw Error(`Point horizon maximumLatitude must be in [0, 90]`);if(!(Number.isFinite(r)&&r>0))throw Error(`Point horizon segmentTolerance must be finite and positive`);let i=O,a=Math.max(.05,Math.tan(Math.min(80,n+3)*Math.PI/180))/i,o=Math.fround(e.maximumDistance),s=[Math.fround(e.minimumDistance)],c=s[0];for(;c<o;){let e=Math.max(200,Math.sqrt(8*r*c/a));c=Math.min(Math.fround(c+e),o),s.push(c)}return s.length===1&&s.push(s[0]+1),{distances:s,sines:s.map(e=>Math.fround(Math.sin(e/i))),oneMinusCosines:s.map(e=>Math.fround(2*Math.sin(e/(2*i))**2)),endIndices:s.slice(1).map((t,n)=>n===s.length-2?e.sampleCount:e.getCeilIndex(t))}}var L=4;function R(e,t,n,r){if(t.observerVerticalDatum!==void 0&&r===`ground`)throw Error(`${e} observerVerticalDatum needs heightReference absolute`);h(e,t.observerVerticalDatum,m(n,t.elevationVerticalDatum))}function z(e,t,n){let{width:r,height:i}=t;if(!Number.isSafeInteger(r)||!Number.isSafeInteger(i)||r<2||i<2)throw Error(`${e} dimensions must be integers of at least 2`);if(r*i>4294967295)throw Error(`${e} dimensions must be positive integers`);d(e,t);let a=n.projection??`planar`;if(a!==`planar`&&a!==`web-mercator`)throw Error(`${e} projection must be planar or web-mercator`);let o=C(e,t,n.rowDirection),s=n.heightReference??`ground`;if(s!==`ground`&&s!==`absolute`)throw Error(`${e} heightReference must be ground or absolute`);let c=n.traversal??`pyramid`;if(c!==`march`&&c!==`pyramid`)throw Error(`${e} traversal must be march or pyramid`);if(n.pyramid&&c!==`pyramid`)throw Error(`${e} pyramid requires the pyramid traversal`);let l=n.azimuthCount??720;if(!Number.isInteger(l)||l<1||l>M)throw Error(`${e} azimuthCount must be an integer in [1, 2^24]`);let u=n.firstAzimuth??0;if(!Number.isInteger(u)||u<0||u>=l)throw Error(`${e} firstAzimuth must be an integer in [0, azimuthCount)`);let f=n.azimuthSpan??l;if(!Number.isInteger(f)||f<1||f>l)throw Error(`${e} azimuthSpan must be an integer in [1, azimuthCount]`);let p;try{p=F(n)}catch(t){throw Error(`${e} ${t.message}`)}return{grid:t,width:r,height:i,projection:a,rowDirection:o,heightReference:s,traversal:c,azimuthCount:l,firstAzimuth:u,azimuthSpan:f,lattice:p,segments:a===`web-mercator`?I(p,n):null,layout:n.pyramid?g(e,n.pyramid,t):c===`pyramid`?y(t,{firstBlockSize:L,footprint:`bilinear`}):null,sharedPyramid:n.pyramid??null}}function B(e,t,n,r){f(n,[r],`${e} ${t}`)}var V=n,H=S*(2-S),U=`
const WGS84_ECCENTRICITY_SQUARED: f32 = ${V(H)};
const WGS84_POLAR_RATIO: f32 = ${V(1-H)};
const WGS84_SEMI_MAJOR_OVER_SPHERE: f32 = ${V(b/O)};
fn getGroundScaleAxes(eye: EyeState) -> vec2<f32> {
  let curvatureTerm = opaque(1.0 - opaque(WGS84_ECCENTRICITY_SQUARED * opaque(eye.sinP * eye.sinP)));
  let east = opaque(WGS84_SEMI_MAJOR_OVER_SPHERE / sqrt(curvatureTerm));
  return vec2<f32>(east, opaque(opaque(east * WGS84_POLAR_RATIO) / curvatureTerm));
}
fn getRayGroundScale(eye: EyeState, sinA: f32, cosA: f32) -> f32 {
  let axes = getGroundScaleAxes(eye);
  let east = opaque(axes.x * sinA);
  let north = opaque(axes.y * cosA);
  return sqrt(opaque(opaque(east * east) + opaque(north * north)));
}
fn getSphereAzimuth(eye: EyeState, sinT: f32, cosT: f32) -> vec2<f32> {
  let axes = getGroundScaleAxes(eye);
  let east = opaque(axes.y * sinT);
  let north = opaque(axes.x * cosT);
  let norm = sqrt(opaque(opaque(east * east) + opaque(north * north)));
  return vec2<f32>(opaque(east / norm), opaque(north / norm));
}
`,W=34028234663852886e22;function G(e){return`array<f32, ${e.length}>(${e.map(V).join(`, `)})`}function K(e){return`array<u32, ${e.length}>(${e.map(e=>`${e}u`).join(`, `)})`}function q(e,t){let n=t[t.length-1];for(let r=t.length-2;r>=0;r--)n=`pinMad(${e}, ${n}, ${t[r]})`;return n}var J=`
const PI: f32 = 3.14159265358979323846;
const HALF_PI: f32 = 1.57079632679489661923;
const QUARTER_PI: f32 = 0.78539816339744830962;
const DEGREES: f32 = 57.29577951308232;
const INV_TWO_PI: f32 = 0.15915494309189535;
const TWO_PI: f32 = 6.28318530717958647692;
const BIG: f32 = ${V(W)};
const EARTH_RADIUS: f32 = ${O};

// XOR with the always-zero settings word: the compiler cannot fold it, so Metal fast math can
// neither re-associate (v - hi) - lo nor fuse a product into the following add.
fn opaque(x: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(x) ^ bitcast<u32>(settings[settingsOffset + 4u]));
}

// a * b + c with the product and the sum both pinned (no fma contraction, no re-association).
fn pinMad(a: f32, b: f32, c: f32) -> f32 {
  return opaque(opaque(a * b) + c);
}

${E}
// NaN built from the runtime zero word: WGSL rejects a constant-evaluated NaN.
fn nanValue() -> f32 { return bitcast<f32>(0x7fc00000u | bitcast<u32>(settings[settingsOffset + 4u])); }

// Odd Taylor series of atan for |z| <= tan(pi/12) = 0.2679.
fn atanSeries(z: f32) -> f32 {
  let z2 = opaque(z * z);
  return opaque(z * ${q(`z2`,[`1.0`,`(-1.0 / 3.0)`,`(1.0 / 5.0)`,`(-1.0 / 7.0)`,`(1.0 / 9.0)`,`(-1.0 / 11.0)`,`(1.0 / 13.0)`,`(-1.0 / 15.0)`])});
}

// atan on [0, 1]: pi/6 + atan((z sqrt(3) - 1) / (z + sqrt(3))) above tan(pi/12).
fn atanUnit(a: f32) -> f32 {
  if (a > 0.2679491924311227) {
    let numerator = opaque(opaque(a * 1.7320508075688772) - 1.0);
    return opaque(0.5235987755982988 + atanSeries(numerator / (a + 1.7320508075688772)));
  }
  return atanSeries(a);
}

// atan to about 2 ULP plus one division, without the builtin.
fn atanAccurate(z: f32) -> f32 {
  let a = abs(z);
  var r: f32;
  if (a > 1.0) {
    r = opaque(HALF_PI - atanUnit(1.0 / a));
  } else {
    r = atanUnit(a);
  }
  return select(-r, r, z >= 0.0);
}

// atan2(y, x) for x > 0 with the series branch of mt-image's atanS, builtin for x <= 0.
fn atanS(y: f32, x: f32) -> f32 {
  if (x > 0.0) {
    return atanAccurate(y / x);
  }
  return atan2(y, x);
}

// atanh(z) for |z| < 0.2 (odd series to z^13), log form otherwise.
fn atanhS(z: f32) -> f32 {
  if (abs(z) < 0.2) {
    let z2 = opaque(z * z);
    return opaque(z * ${q(`z2`,[`1.0`,`(1.0 / 3.0)`,`(1.0 / 5.0)`,`(1.0 / 7.0)`,`(1.0 / 9.0)`,`(1.0 / 11.0)`,`(1.0 / 13.0)`])});
  }
  return 0.5 * log((1.0 + z) / (1.0 - z));
}

// sin and cos series for |x| <= pi / 2 (error below 1e-9).
fn sinSeries(x: f32) -> f32 {
  let x2 = opaque(x * x);
  return opaque(x * ${q(`x2`,[`1.0`,`(-1.0 / 6.0)`,`(1.0 / 120.0)`,`(-1.0 / 5040.0)`,`(1.0 / 362880.0)`,`(-1.0 / 39916800.0)`,`(1.0 / 6227020800.0)`,`(-1.0 / 1307674368000.0)`])});
}
fn cosSeries(x: f32) -> f32 {
  let x2 = opaque(x * x);
  return ${q(`x2`,[`1.0`,`(-1.0 / 2.0)`,`(1.0 / 24.0)`,`(-1.0 / 720.0)`,`(1.0 / 40320.0)`,`(-1.0 / 3628800.0)`,`(1.0 / 479001600.0)`,`(-1.0 / 87178291200.0)`])};
}

// Azimuth index i of N (angle i * 360 / N degrees clockwise from north) to (sin, cos) with no
// builtin trigonometry: 8 i = octant * N + r exactly in integers, the octant remainder is folded
// into [0, pi/4], series evaluate it, and a quadrant rotation finishes. (sin, cos) = (east, north).
fn azimuthSinCos(index: u32, count: u32) -> vec2<f32> {
  let eight = 8u * (index % count);
  let octant = eight / count;
  let remainder = eight - octant * count;
  let odd = (octant & 1u) == 1u;
  let folded = select(remainder, count - remainder, odd);
  let angle = opaque(opaque(f32(folded) / f32(count)) * QUARTER_PI);
  let s0 = sinSeries(angle);
  let c0 = cosSeries(angle);
  let s = select(s0, c0, odd);
  let c = select(c0, s0, odd);
  let quadrant = octant >> 1u;
  if (quadrant == 0u) { return vec2<f32>(s, c); }
  if (quadrant == 1u) { return vec2<f32>(c, -s); }
  if (quadrant == 2u) { return vec2<f32>(-s, -c); }
  return vec2<f32>(-c, s);
}

// Normalised Mercator offset (dx, dy) of the great-circle point at angular distance D (sinD, omc =
// 1 - cos D) along azimuth (sinA, cosA) from the eye (sinP1, cosP1, c2 = cosP1^2).
fn bp(sinD: f32, omc: f32, sinA: f32, cosA: f32, sinP1: f32, cosP1: f32, c2: f32) -> vec2<f32> {
  let ds = opaque(opaque(opaque(cosP1 * sinD) * cosA) - opaque(sinP1 * omc)); // sin(phi2) - sin(phi1)
  let den = opaque(c2 - opaque(sinP1 * ds)); // 1 - sin(phi1) sin(phi2)
  let dl = atanS(opaque(opaque(sinA * sinD) * cosP1), opaque(den - omc)); // cos D - sin(phi1) sin(phi2)
  let dy = atanhS(ds / den); // atanh(sin phi2) - atanh(sin phi1)
  return vec2<f32>(opaque(dl * INV_TWO_PI), opaque(-dy * INV_TWO_PI));
}

// sinh with a series below 0.5 (the exp form cancels there).
fn sinhS(x: f32) -> f32 {
  if (abs(x) < 0.5) {
    let x2 = opaque(x * x);
    return opaque(x * ${q(`x2`,[`1.0`,`(1.0 / 6.0)`,`(1.0 / 120.0)`,`(1.0 / 5040.0)`,`(1.0 / 362880.0)`,`(1.0 / 39916800.0)`,`(1.0 / 6227020800.0)`])});
  }
  return 0.5 * (exp(x) - exp(-x));
}
fn coshS(x: f32) -> f32 {
  let e = exp(abs(x));
  return 0.5 * (e + 1.0 / e);
}
// tanh with a series below 0.3, the exp form above.
fn tanhS(x: f32) -> f32 {
  let a = abs(x);
  var r: f32;
  if (a < 0.3) {
    let x2 = opaque(x * x);
    r = opaque(a * ${q(`x2`,[`1.0`,`(-1.0 / 3.0)`,`(2.0 / 15.0)`,`(-17.0 / 315.0)`,`(62.0 / 2835.0)`,`(-1382.0 / 155925.0)`])});
  } else {
    let e = exp(-2.0 * a);
    r = (1.0 - e) / (1.0 + e);
  }
  return select(-r, r, x >= 0.0);
}
// asin for z < 0.2 by series, builtin above.
fn asinS(z: f32) -> f32 {
  if (z < 0.2) {
    let z2 = opaque(z * z);
    return opaque(z * ${q(`z2`,[`1.0`,`(1.0 / 6.0)`,`(3.0 / 40.0)`,`(5.0 / 112.0)`,`(35.0 / 1152.0)`,`(63.0 / 2816.0)`,`(231.0 / 13312.0)`])});
  }
  return asin(min(z, 1.0));
}
`;function Y(e,t){let{lattice:n,segments:r,layout:i}=e,a=e.projection===`web-mercator`,o=e.traversal===`pyramid`,s=n.octaves,c=r?r.distances.length-1:1,l=r?r.endIndices:[n.sampleCount],u=e.heightReference===`ground`?`let ground = sampleGround(column, row);
  if (ground.y == 0.0 || !isFiniteValue(height)) {
    eye.valid = false;
    return eye;
  }
  eye.hi = ground.x;
  eye.lo = height;`:`if (!isFiniteValue(height)) {
    eye.valid = false;
    return eye;
  }
  eye.hi = height;
  eye.lo = 0.0;`,d=a?`let m1 = settings[settingsOffset + 2u] - TWO_PI * (row + 0.5) / settings[settingsOffset + 1u];
  eye.sinP = tanhS(m1);
  eye.cosP = 1.0 / coshS(m1);
  eye.m = m1;`:``,f=a?`let worldPixelSize = settings[settingsOffset + 1u];
  let c2 = opaque(eye.cosP * eye.cosP);
  var b0 = bp(SEGMENT_SINES[0], SEGMENT_OMC[0], sinA, cosA, eye.sinP, eye.cosP, c2);`:`let dirU = opaque(sinA / settings[settingsOffset + 1u]);
  let dirV = opaque(${e.rowDirection===`south`?`-`:``}cosA / settings[settingsOffset + 2u]);`,p=a?`let b1 = bp(SEGMENT_SINES[segment + 1u], SEGMENT_OMC[segment + 1u], sinA, cosA, eye.sinP, eye.cosP, c2);
    let dA = SEGMENT_DISTANCES[segment];
    let invLength = opaque(1.0 / (SEGMENT_DISTANCES[segment + 1u] - dA));
    let uA = eye.fx + opaque(b0.x * worldPixelSize);
    let vA = eye.fy + opaque(b0.y * worldPixelSize);
    let du = opaque(opaque((b1.x - b0.x) * worldPixelSize) * invLength);
    let dv = opaque(opaque((b1.y - b0.y) * worldPixelSize) * invLength);`:`let dA = 0.0;
    let uA = eye.fx;
    let vA = eye.fy;
    let du = dirU;
    let dv = dirV;`,m=a?`b0 = b1;`:``,h=o?`if (n >= noTest) {
        var skipTo = n;
        var skipped = false;
        for (var level = 0u; level < PYRAMID_LEVEL_COUNT; level++) {
          let cellIndex = pyramidLevelIndex(level, u32(p.xi), u32(p.yi));
          let hMax = pyramid[pyramidOffset + cellIndex];
          let empty = hMax <= PYRAMID_EMPTY_MAXIMUM;
          if (!empty && !has) { break; }
          let blockSize = i32(pyramidLevelBlockSize(level));
          let blockX = (p.xi / blockSize) * blockSize;
          let blockY = (p.yi / blockSize) * blockSize;
          // Analytic exit of the ray from the block (integer part exact).
          var exitX = BIG;
          if (du > 0.0) { exitX = (f32(blockX + blockSize - p.xi) - p.fx) / du; }
          else if (du < 0.0) { exitX = (f32(blockX - p.xi) - p.fx) / du; }
          var exitY = BIG;
          if (dv > 0.0) { exitY = (f32(blockY + blockSize - p.yi) - p.fy) / dv; }
          else if (dv < 0.0) { exitY = (f32(blockY - p.yi) - p.fy) / dv; }
          let exitDistance = d + min(exitX, exitY);
          var hMin = 0.0;
          if (!empty) {
            hMin = pyramid[pyramidOffset + PYRAMID_MINIMUM_OFFSET + cellIndex];
            // Cheap rejection with an upper estimate of the last skipped distance: any verified
            // last sample is below the exit distance, the segment end and the ray end, and the
            // bound only loosens as the last distance grows, so failing here loses nothing exact.
            let farthest = min(exitDistance, min(${a?`SEGMENT_DISTANCES[segment + 1u]`:`BIG`}, maximumDistance));
            if (skipBound(hMax, hMin, eye, d, farthest, curvature) > tBest) {
              if (level == 0u) { noTest = max(n + 1u, latCeilIndex(exitDistance)); }
              break;
            }
          }
          // Clamp to the last sample of the segment and of the ray, then verify and back off.
          var candidate = latFloorIndex(exitDistance);
          candidate = min(candidate, min(i32(segmentEnd) - 1, rayLast));
          var verified = false;
          var dEnd = d;
          for (var attempt = 0u; attempt < 4u; attempt++) {
            if (candidate < i32(n)) { break; }
            let candidateOctave = latOctave(u32(candidate));
            let candidateDistance = latDistance(
              candidateOctave, u32(candidate) - LAT_FIRST[candidateOctave] + LAT_J0[candidateOctave]);
            let end = getPosition(candidateDistance, dA, uA, vA, du, dv, eye);
            if (inGrid(end) && end.xi >= blockX && end.xi < blockX + blockSize &&
                end.yi >= blockY && end.yi < blockY + blockSize) {
              verified = true;
              dEnd = candidateDistance;
              break;
            }
            candidate = candidate - 1;
          }
          if (!verified) { break; }
          if (!empty && skipBound(hMax, hMin, eye, d, dEnd, curvature) > tBest) {
            if (level == 0u) { noTest = u32(candidate) + 1u; }
            break;
          }
          skipped = true;
          skipTo = max(skipTo, u32(candidate) + 1u);
        }
        if (skipped) {
          n = skipTo;
          if (n < LAT_COUNT) {
            octave = latOctave(n);
            j = n - LAT_FIRST[octave] + LAT_J0[octave];
          }
          continue;
        }
      }`:``;return`
const WIDTH: u32 = ${e.width}u;
const HEIGHT: u32 = ${e.height}u;
const WIDTH_I: i32 = ${e.width};
const HEIGHT_I: i32 = ${e.height};
const AZIMUTH_COUNT: u32 = ${e.azimuthCount}u;
const FIRST_AZIMUTH: u32 = ${e.firstAzimuth}u;
const AZIMUTH_SPAN: u32 = ${e.azimuthSpan}u;
const SKIP_EPSILON: f32 = 7.62939453125e-6;
${J}
// Exact power-of-two distance lattice: octave k holds 2^k + j * step_k, all exact in f32.
const LAT_OCTAVE_COUNT: u32 = ${s.length}u;
const LAT_K0: i32 = ${s[0].exponent};
const LAT_COUNT: u32 = ${n.sampleCount}u;
const LAT_FIRST_DISTANCE: f32 = ${V(n.firstDistance)};
const LAT_LAST_DISTANCE: f32 = ${V(n.lastDistance)};
const MAXIMUM_DISTANCE: f32 = ${V(n.maximumDistance)};
const LAT_BASE = ${G(s.map(e=>e.base))};
const LAT_STEP = ${G(s.map(e=>e.step))};
const LAT_INV_STEP = ${G(s.map(e=>1/e.step))};
const LAT_J0 = ${K(s.map(e=>e.firstJ))};
const LAT_J1 = ${K(s.map(e=>e.endJ))};
const LAT_FIRST = ${K(s.map(e=>e.firstIndex))};
const SEGMENT_COUNT: u32 = ${c}u;
const SEGMENT_END = ${K(l)};
${r?`const SEGMENT_DISTANCES = ${G(r.distances)};
const SEGMENT_SINES = ${G(r.sines)};
const SEGMENT_OMC = ${G(r.oneMinusCosines)};`:``}
${i?_(i,`pyramid`):``}

fn latDistance(octave: u32, j: u32) -> f32 {
  return LAT_BASE[octave] + f32(j) * LAT_STEP[octave];
}

fn latOctave(index: u32) -> u32 {
  var octave = 0u;
  for (var i = 1u; i < LAT_OCTAVE_COUNT; i++) {
    if (index >= LAT_FIRST[i]) { octave = i; }
  }
  return octave;
}

// Index of the first lattice sample with distance >= x (LAT_COUNT if none). The octave comes from
// the f32 exponent of x; x - 2^k and the division by a power of two are exact.
fn latCeilIndex(x: f32) -> u32 {
  if (!(x > LAT_FIRST_DISTANCE)) { return 0u; }
  if (x > LAT_LAST_DISTANCE) { return LAT_COUNT; }
  let exponent = i32((bitcast<u32>(x) >> 23u) & 0xffu) - 127;
  let octave = u32(exponent - LAT_K0);
  let steps = ceil((x - LAT_BASE[octave]) * LAT_INV_STEP[octave]);
  let j = min(max(u32(steps), LAT_J0[octave]), LAT_J1[octave]);
  return LAT_FIRST[octave] + j - LAT_J0[octave];
}

// Index of the last lattice sample with distance <= x (-1 if none).
fn latFloorIndex(x: f32) -> i32 {
  if (!(x >= LAT_FIRST_DISTANCE)) { return -1; }
  if (x >= LAT_LAST_DISTANCE) { return i32(LAT_COUNT) - 1; }
  let exponent = i32((bitcast<u32>(x) >> 23u) & 0xffu) - 127;
  let octave = u32(exponent - LAT_K0);
  let steps = floor((x - LAT_BASE[octave]) * LAT_INV_STEP[octave]);
  let j = min(u32(steps), LAT_J1[octave] - 1u);
  return i32(LAT_FIRST[octave] + j - LAT_J0[octave]);
}

// Bilinear height relative to the eye. Each corner is (v - hi) - lo with the subtraction pinned by
// opaque(): near the camera an f32 height formed after interpolation would carry the rounding of a
// 4-digit height, about 1e-3 degrees at 2 m. Products are pinned too so the result does not depend
// on the compiler's fused multiply-add choices (the pyramid bound relies on a stable value).
// Returns (height, 1), or (0, 0) when a corner is invalid. Canonical elevation is NaN exactly where a
// pixel is invalid, so one bit test per loaded corner replaces four validity loads.
fn bilinearRelative(cell: u32, fx: f32, fy: f32, hi: f32, lo: f32) -> vec2<f32> {
  let v00 = elevationValues[elevationValuesOffset + cell];
  let v10 = elevationValues[elevationValuesOffset + cell + 1u];
  let v01 = elevationValues[elevationValuesOffset + cell + WIDTH];
  let v11 = elevationValues[elevationValuesOffset + cell + WIDTH + 1u];
  if (!isFiniteValue(v00) || !isFiniteValue(v10) || !isFiniteValue(v01) || !isFiniteValue(v11)) {
    return vec2<f32>(0.0, 0.0);
  }
  let a0 = opaque(v00 - hi) - lo;
  let a1 = opaque(v10 - hi) - lo;
  let c0 = opaque(v01 - hi) - lo;
  let c1 = opaque(v11 - hi) - lo;
  let t1 = opaque(a0 - a1);
  let t2 = opaque(t1 - c0);
  let twist = t2 + c1;
  let rowSlope = (c0 - a0) + opaque(twist * fx);
  return vec2<f32>((a0 + opaque((a1 - a0) * fx)) + opaque(rowSlope * fy), 1.0);
}

// Bilinear height relative to (hi, lo) at a pixel-center position inside [0, W-1] x [0, H-1];
// .y is 0 when a corner is invalid.
fn sampleRelative(column: f32, row: f32, hi: f32, lo: f32) -> vec2<f32> {
  let x0 = min(i32(floor(column)), WIDTH_I - 2);
  let y0 = min(i32(floor(row)), HEIGHT_I - 2);
  let cell = u32(y0) * WIDTH + u32(x0);
  return bilinearRelative(cell, column - f32(x0), row - f32(y0), hi, lo);
}

fn sampleGround(column: f32, row: f32) -> vec2<f32> {
  return sampleRelative(column, row, 0.0, 0.0);
}

fn isInsideGrid(column: f32, row: f32) -> bool {
  return column >= 0.0 && row >= 0.0 && column <= f32(WIDTH - 1u) && row <= f32(HEIGHT - 1u);
}

struct EyeState {
  valid: bool,
  column: i32,
  row: i32,
  fx: f32,
  fy: f32,
  hi: f32,
  lo: f32,
  sinP: f32,
  cosP: f32,
  m: f32,
};

// Eye at (column, row) in pixel-center index space: integer pixel plus f32 fraction, and the eye
// height as an f32 (hi, lo) pair so (value - hi) - lo is exact to f32 near the eye.
// Settings packed for the other projection (tag 1 planar, 2 Web Mercator; 0 untagged) are rejected.
fn isProjectionTagValid() -> bool {
  let tag = settings[settingsOffset + ${t}u];
  return tag == 0.0 || tag == ${D[e.projection]}.0;
}

fn makeEye(column: f32, row: f32, height: f32) -> EyeState {
  var eye: EyeState;
  eye.valid = isInsideGrid(column, row) && isProjectionTagValid();
  if (!eye.valid) { return eye; }
  let floorColumn = floor(column);
  let floorRow = floor(row);
  eye.column = i32(floorColumn);
  eye.row = i32(floorRow);
  eye.fx = column - floorColumn;
  eye.fy = row - floorRow;
  ${u}
  ${d}
  return eye;
}
${a?U:``}
struct Position { xi: i32, yi: i32, fx: f32, fy: f32 };

// Position of the sample at distance d inside the segment starting at dA, relative to the eye's
// integer pixel; the same code serves the march, the skip verification and both kernels.
fn getPosition(d: f32, dA: f32, uA: f32, vA: f32, du: f32, dv: f32, eye: EyeState) -> Position {
  let f = d - dA;
  let uf = uA + opaque(f * du);
  let vf = vA + opaque(f * dv);
  let floorU = floor(uf);
  let floorV = floor(vf);
  return Position(eye.column + i32(floorU), eye.row + i32(floorV), uf - floorU, vf - floorV);
}

// Bilinear footprint of a sample must be inside the grid, otherwise the ray ends.
fn inGrid(p: Position) -> bool {
  return p.xi >= 0 && p.yi >= 0 && p.xi < WIDTH_I - 1 && p.yi < HEIGHT_I - 1;
}

// Upper bound of the tangent of every sample with distance in [d, dEnd] inside a cell of extrema
// (hMax, hMin), see the proof above. EPS pads every rounding, FMA fusion and re-association.
fn skipBound(hMax: f32, hMin: f32, eye: EyeState, d: f32, dEnd: f32, curvature: f32) -> f32 {
  let magnitude = max(abs(hMax - eye.hi), abs(hMin - eye.hi)) + abs(eye.lo);
  let numerator = (hMax - eye.hi) - eye.lo + SKIP_EPSILON * magnitude;
  let ratio = select(
    (numerator / dEnd) * (1.0 - SKIP_EPSILON),
    (numerator / d) * (1.0 + SKIP_EPSILON),
    numerator >= 0.0
  );
  let drop = min(d * curvature, dEnd * curvature);
  return ratio - drop + SKIP_EPSILON * abs(drop);
}

struct RayResult { has: bool, t: f32, d: f32, tQ: f32, evaluated: u32 };

// Marches one ray. Returns the maximum tangent tBest = hr / d - d c over valid samples and its
// distance, both in ground meters. tQ is the running maximum when the first sample at ground
// distance >= groundQ is reached (groundQ <= 0 or !useQ: -BIG). (sinA, cosA) is the azimuth on the
// sphere for Web Mercator, see getSphereAzimuth. evaluated counts the samples whose bilinear height was read (a debug measure of
// the pyramid's skipping).
fn marchRay(eye: EyeState, sinA: f32, cosA: f32, groundQ: f32, useQ: bool) -> RayResult {
  ${a?`// The ray is traced in mean-sphere meters and reported in WGS84 ground meters: with
  // d_ground = s d, the tangent h / d_ground - c d_ground is (h / d - (c s^2) d) / s, so the march
  // runs unchanged with the curvature c s^2 and limits divided by s, and s > 0 keeps the argmax.
  let groundScale = getRayGroundScale(eye, sinA, cosA);
  let curvature = opaque(opaque(settings[settingsOffset] * groundScale) * groundScale);
  let maximumFrame = settings[settingsOffset + 3u] / groundScale;
  let q = groundQ / groundScale;`:`let curvature = settings[settingsOffset];
  let maximumFrame = settings[settingsOffset + 3u];
  let q = groundQ;`}
  let maximumDistance = select(LAT_LAST_DISTANCE, min(maximumFrame, LAT_LAST_DISTANCE), maximumFrame > 0.0);
  let rayLast = latFloorIndex(maximumDistance);
  ${f}
  var has = false;
  var tBest = 0.0;
  var dBest = 0.0;
  var qDone = !useQ;
  var tQ = -BIG;
  var n = 0u;
  var octave = 0u;
  var j = LAT_J0[0];
  var ended = false;
  var noTest = 0u;
  var evaluated = 0u;
  for (var segment = 0u; segment < SEGMENT_COUNT; segment++) {
    if (ended || i32(n) > rayLast) { break; }
    let segmentEnd = SEGMENT_END[segment];
    ${p}
    loop {
      if (n >= segmentEnd || i32(n) > rayLast) { break; }
      let d = latDistance(octave, j);
      if (!qDone && d >= q) {
        tQ = select(-BIG, tBest, has);
        qDone = true;
      }
      let p = getPosition(d, dA, uA, vA, du, dv, eye);
      if (!inGrid(p)) {
        ended = true;
        break;
      }
      ${h}
      let cell = u32(p.yi) * WIDTH + u32(p.xi);
      evaluated = evaluated + 1u;
      let relativeSample = bilinearRelative(cell, p.fx, p.fy, eye.hi, eye.lo);
      if (relativeSample.y != 0.0) {
        let hr = relativeSample.x;
        let t = hr / d - opaque(d * curvature);
        if (!has || t > tBest) {
          has = true;
          tBest = t;
          dBest = d;
        }
      }
      n = n + 1u;
      j = j + 1u;
      if (j >= LAT_J1[octave]) {
        octave = octave + 1u;
        j = 0u;
      }
    }
    ${m}
  }
  if (!qDone) { tQ = select(-BIG, tBest, has); }
  ${a?`// Back to ground meters; the no-sample sentinel -BIG of tQ is kept exact.
  return RayResult(
    has, tBest / groundScale, dBest * groundScale, select(tQ / groundScale, -BIG, tQ == -BIG), evaluated);`:`return RayResult(has, tBest, dBest, tQ, evaluated);`}
}
`}var X=64;function Z(e,t){let n=Math.min(t.maximumChunkSize,Math.max(t.itemCount,1)),i=Math.max(Math.ceil(t.itemCount/n),1),a=p(e,`${t.id}-chunk-words`,`uint32`,i*X);return{chunkCount:i,chunkSize:n,writeNode:r(e,{id:`${t.id}-chunks`,operation:t.operation,variant:`chunk-starts`,bindings:[{name:`chunkWords`,view:a,type:`u32`,access:`read_write`}],invocationCount:i,body:`chunkWords[chunkWordsOffset + index * ${X}u] = index * ${n}u;`}),getChunkStart:t=>({name:`chunkStart`,view:e.createDataView(a.buffer,{format:`uint32`,length:1,byteOffset:a.byteOffset+t*X*4}),type:`u32`,access:`read`})}}function Q(e,t,n,r){let i=u(e,t,r,n.grid,!0),a=[...i.nodes],o=i.band.storage.values,s=i.band.validity,c=[{name:`elevationValues`,view:o,type:`f32`,access:`read`}];if(n.layout&&n.sharedPyramid)c.push({name:`pyramid`,view:n.sharedPyramid.combined,type:`f32`,access:`read`});else if(n.layout){let r=p(e,`${t}-pyramid`,`float32`,2*n.layout.length);a.push(...v(e,{id:`${t}-pyramid`,layout:n.layout,values:o,validity:s,combined:r})),c.push({name:`pyramid`,view:r,type:`f32`,access:`read`})}return{nodes:a,bindings:c}}var $=65536,ee=class{id;props;model;constructor(e){this.id=e.id??`point-horizon-profile`,this.props=e;let{id:t}=this;this.model=z(t,e.grid,e),R(t,e,e.elevation,this.model.heightReference);let{output:n}=e;if(!n.tangent&&!n.skylineAngle&&!n.distance)throw Error(`${t} requires at least one output`);if(o(t,e.parameters,8),B(t,`observers`,e.observers,`float32x4`),e.observers.length<1)throw Error(`${t} requires at least one observer`);let r=e.observers.length*this.model.azimuthSpan;for(let[e,i]of[[`tangent`,n.tangent],[`skylineAngle`,n.skylineAngle],[`distance`,n.distance]])if(i&&(B(t,`output.${e}`,i,`float32`),i.length!==r))throw Error(`${t} output.${e} must contain observers * azimuthSpan values`);let i=e.raysPerDispatch??65536;if(!Number.isInteger(i)||i<1)throw Error(`${t} raysPerDispatch must be a positive integer`);s(t,[n.tangent,n.skylineAngle,n.distance,n.samples],[...c(e.elevation),e.parameters,e.observers])}getCommandNodes(n){t(n,this.id,this.props,{"canonical elevation":4*this.props.grid.width*this.props.grid.height,"pyramid fused parameters":e(this.props.pyramid?void 0:this.model.layout),pyramid:this.model.layout&&!this.props.pyramid?8*this.model.layout.length:0,"chunk table":256*Math.max(1,Math.ceil(this.props.observers.length*this.model.azimuthSpan/(this.props.raysPerDispatch??65536)))});let{id:o,props:s,model:c}=this,{output:l}=s;a(o,n,s.elevation,[]),i(o,n,[s.pyramid?.combined,s.parameters,s.observers,l.tangent,l.skylineAngle,l.distance,l.samples]);let u=Q(n,o,c,s.elevation),d=[...u.nodes],f=[...u.bindings,{name:`settings`,view:s.parameters,type:`f32`,access:`read`},{name:`observers`,view:s.observers,type:`f32`,access:`read`}];l.tangent&&f.push({name:`tangentOutput`,view:l.tangent,type:`f32`,access:`read_write`});let p=!!(l.tangent&&l.skylineAngle);if(l.skylineAngle&&!p&&f.push({name:`skylineAngleOutput`,view:l.skylineAngle,type:`f32`,access:`read_write`}),l.distance&&f.push({name:`distanceOutput`,view:l.distance,type:`f32`,access:`read_write`}),l.samples&&f.push({name:`samplesOutput`,view:l.samples,type:`u32`,access:`read_write`}),f.length>7)throw Error(`${o} needs more than 8 storage bindings`);let m=Y(c,7),h=s.observers.length*c.azimuthSpan,g=s.raysPerDispatch??65536,_=e=>[l.tangent?`tangentOutput[tangentOutputOffset + ray] = ${e===`nan`?`nanValue()`:`select(-BIG, result.t, result.has)`};`:``,l.skylineAngle&&!p?`skylineAngleOutput[skylineAngleOutputOffset + ray] = ${e===`nan`?`nanValue()`:`select(-90.0, atanAccurate(result.t) * DEGREES, result.has)`};`:``,l.distance?`distanceOutput[distanceOutputOffset + ray] = ${e===`nan`?`nanValue()`:`select(0.0, result.d, result.has)`};`:``,l.samples?`samplesOutput[samplesOutputOffset + ray] = ${e===`nan`?`0u`:`result.evaluated`};`:``].join(`
  `),v=Z(n,{id:o,operation:`GPUPointHorizonProfile`,itemCount:h,maximumChunkSize:g});d.push(v.writeNode);for(let e=0;e<v.chunkCount;e++)d.push(r(n,{id:`${o}-march-${e}`,operation:`GPUPointHorizonProfile`,variant:c.traversal,bindings:[...f,v.getChunkStart(e)],invocationCount:v.chunkSize,workgroupSize:64,declarations:`const RAY_COUNT: u32 = ${h}u;
${m}`,body:`let ray = chunkStart[chunkStartOffset] + index;
  if (ray >= RAY_COUNT) { return; }
  let observerIndex = ray / AZIMUTH_SPAN;
  let azimuthOffset = ray - observerIndex * AZIMUTH_SPAN;
  let observerBase = observersOffset + observerIndex * 4u;
  let eye = makeEye(observers[observerBase], observers[observerBase + 1u], observers[observerBase + 2u]);
  if (!eye.valid) {
    ${_(`nan`)}
    return;
  }
  let bearing = azimuthSinCos(FIRST_AZIMUTH + azimuthOffset, AZIMUTH_COUNT);
  ${c.projection===`web-mercator`?`// Azimuths are ground bearings on the WGS84 ellipsoid; the ray is traced on the sphere.
  let direction = getSphereAzimuth(eye, bearing.x, bearing.y);`:`let direction = bearing;`}
  let result = marchRay(eye, direction.x, direction.y, 0.0, false);
  ${_(`result`)}`}));return l.tangent&&l.skylineAngle&&p&&d.push(r(n,{id:`${o}-skyline-angle`,operation:`GPUPointHorizonProfile`,variant:`skyline-angle`,bindings:[{name:`settings`,view:s.parameters,type:`f32`,access:`read`},{name:`tangent`,view:l.tangent,type:`f32`,access:`read`},{name:`skylineAngleOutput`,view:l.skylineAngle,type:`f32`,access:`read_write`}],invocationCount:h,declarations:`const RAY_COUNT: u32 = ${h}u;
${J}`,body:`if (index >= RAY_COUNT) { return; }
  let t = tangent[tangentOffset + index];
  var angle = t;
  if ((bitcast<u32>(t) & 0x7fffffffu) <= 0x7f800000u) {
    angle = select(-90.0, atanAccurate(t) * DEGREES, t != -BIG);
  }
  skylineAngleOutput[skylineAngleOutputOffset + index] = angle;`})),d}};export{Y as a,R as c,Q as i,A as l,$ as n,z as o,Z as r,B as s,ee as t,j as u};
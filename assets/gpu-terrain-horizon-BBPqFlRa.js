import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{t}from"./common-CshkX-d_.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-raster-buffer-to-texture-BDZQU4A5.js";import{c as i,o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{T as o,f as s,g as c,i as l,x as u}from"./gpu-contributor-utils-BIwLQON5.js";import{t as d}from"./construction-snapshot-DdkX9Pc0.js";import{c as f,d as p,h as m,l as h,n as g,o as _,p as v,r as ee,u as y}from"./raster-band-utils-BcTxc8_Y.js";import{c as b,i as x,l as S,n as C,o as w,r as te,s as T,t as E,u as D}from"./terrain-illumination-utils-CoEB4UZd.js";import{a as O,f as k,n as A,r as j,u as M}from"./raster-grid-VB0AM_GY.js";import{a as N}from"./terrain-geomorphometry-utils-BhBw8dIY.js";var P=65536,F=32767,I={cellSizeIndex:0,northEdgeIndex:4},L=64,R=2**24-1;function z(e,t,n){let r=t.sweepReferenceCellSize,i=t.sweepSourceGrid;o(e,r!==void 0&&i!==void 0,`sweepReferenceCellSize and sweepSourceGrid are required for sweep settings`),o(e,r.length===2,`sweepReferenceCellSize must contain exactly two positive finite ground cell sizes`);for(let[t,n]of r.entries())c(e,`sweepReferenceCellSize[${t}]`,n,Number.MIN_VALUE);let a=t.sweepOrigin??[0,0];o(e,a.length===2&&a.every(Number.isSafeInteger),`sweepOrigin must contain two safe integer source overview pixel coordinates`),s(e,`sweepSourceGrid.height`,i.height,1,R),s(e,`sweepSourceGrid.width`,i.width,1,2**53-1),o(e,a[0]>=0&&a[1]>=0&&a[0]+n.width<=i.width&&a[1]+n.height<=i.height,`sweepOrigin and window dimensions must lie inside sweepSourceGrid`);let l=j(i);return c(e,`sweepSourceGrid.cellSizeX`,l.cellSize[0],Number.MIN_VALUE),c(e,`sweepSourceGrid.cellSizeY`,l.cellSize[1],Number.MIN_VALUE),c(e,`sweepSourceGrid.northEdge`,l.northEdge),c(e,`sweepSourceGrid.southEdge`,l.southEdge),Float32Array.from([...r,a[0]%P,a[1]%P,...l.cellSize,l.northEdge,l.southEdge,i.height,a[1],0,0])}function B(e,t){return t===void 0?O(e,I):`${O(e,{cellSizeIndex:t,northEdgeIndex:t+2}).replaceAll(`getGroundCellSize`,`getSourceGroundCellSize`).replaceAll(`f32(HEIGHT)`,`settings[settingsOffset + ${t+4}u]`)}
fn getGroundCellSize(row: u32) -> vec2<f32> {
  return getSourceGroundCellSize(row + u32(settings[settingsOffset + ${t+5}u]));
}`}function ne(e,t,n=0){let r=((n%P*e+32768)%P+P)%P;return Math.floor((e*t+r)/P)}function V(e,t,n,r=[0,0]){let[i,a]=e;if(!Number.isFinite(i)||!Number.isFinite(a)||i===0&&a===0)throw Error(`Terrain sweep direction must be finite and non-zero`);if(!Number.isSafeInteger(t)||!Number.isSafeInteger(n)||t<1||n<1||t>32767||n>32767)throw Error(`Terrain sweep extents must be integers in [1, ${F}]`);let o=Math.abs(i)>=Math.abs(a),s=o?i:a,c=Math.round((o?a:i)/s*P)+0,l=o?t:n,u=o?n:t,d=ne(c,l-1,r[+!o]);return{xMajor:o,slopeFixed:c,travelSign:s>0?1:-1,firstLine:-Math.max(0,d)+0,lineCount:u+Math.abs(d),stepPixelLength:Math.fround(Math.hypot(1,c/P))}}var H=64,U=7;function re(e,t){let r=t.directions.length,i=new Uint32Array(r*U),o=new Float32Array(1),s=new Int32Array(1),c=e=>(o[0]=e,new Uint32Array(o.buffer)[0]),l=e=>(s[0]=e,new Uint32Array(s.buffer)[0]);for(let[e,n]of t.directions.entries()){let r=V(n,t.width,t.height);i.set([+!!r.xMajor,l(r.slopeFixed),l(r.travelSign),c(r.stepPixelLength),c(n[0]),c(n[1]),e],e*U)}let u=n(e,`${t.id}-sweep-sector-records`,`uint32`,Math.max(r,1)*H);return{writeNode:a(e,{id:`${t.id}-sweep-sectors`,operation:`GPUTerrainHorizon`,variant:`sweep-sector-table`,bindings:[{name:`records`,view:u,type:`u32`,access:`read_write`}],invocationCount:i.length,declarations:`const SECTOR_WORDS = array<u32, ${i.length}>(${Array.from(i,e=>`${e}u`).join(`, `)});`,body:`let sector = index / ${U}u;
  let word = index % ${U}u;
  records[recordsOffset + sector * ${H}u + word] = SECTOR_WORDS[index];`}),getRecord:t=>e.createDataView(u.buffer,{format:`uint32`,length:U,byteOffset:u.byteOffset+t*H*4})}}var W=`sectorRecord[sectorRecordOffset + 6u]`;function G(e,t){let{width:n,height:r,geometry:o}=t,s=o.kind===`table`?4:5;if(t.outputBindings.length>s)throw Error(`${t.id} sweep supports at most ${s} output bindings`);if(!Number.isFinite(t.maximumRadius)||t.maximumRadius<=0)throw Error(`${t.id} sweep maximumRadius must be positive`);let c=o.kind===`baked`?V(o.direction,n,r):void 0;if(n>32767||r>32767)throw Error(`${t.id} sweep extents must not exceed ${F}`);let l=t.cellSizeMode===`uniform`?`0u`:`HEIGHT / 2u`,u=o.kind===`settings`?o.index:0,d=o.kind===`settings`?o.groundDirectionIndex:0,f=`${o.kind===`table`?`var xMajor = sectorRecord[sectorRecordOffset] != 0u;
  var slopeFixed = bitcast<i32>(sectorRecord[sectorRecordOffset + 1u]);
  var travelSign = bitcast<i32>(sectorRecord[sectorRecordOffset + 2u]);
  var stepPixelLength = bitcast<f32>(sectorRecord[sectorRecordOffset + 3u]);
  let groundDirection = vec2<f32>(
    bitcast<f32>(sectorRecord[sectorRecordOffset + 4u]),
    bitcast<f32>(sectorRecord[sectorRecordOffset + 5u])
  );`:c?`var xMajor = ${c.xMajor?`true`:`false`};
  var slopeFixed: i32 = ${c.slopeFixed};
  var travelSign: i32 = ${c.travelSign};
  var stepPixelLength: f32 = ${i(c.stepPixelLength)};
  let groundDirection = GROUND_DIRECTION;`:`var xMajor = settings[settingsOffset + ${u}u] != 0.0;
  var slopeFixed = i32(settings[settingsOffset + ${u+1}u]);
  var travelSign = select(-1, 1, settings[settingsOffset + ${u+2}u] > 0.0);
  var stepPixelLength = settings[settingsOffset + ${u+3}u];
  ${o.kind===`settings`&&o.rowDirection===`north`?`var`:`let`} groundDirection = vec2<f32>(
    settings[settingsOffset + ${d}u],
    settings[settingsOffset + ${d+1}u]
  );${o.kind===`settings`&&o.rowDirection===`north`?`
  // The packed line assumes south-advancing rows; mirror it for north-advancing rows. Negating the
  // row component negates the slope, and the travel sign of y-major lines.
  groundDirection.y = -groundDirection.y;
  slopeFixed = -slopeFixed;
  if (!xMajor) { travelSign = -travelSign; }`:``}`}
  // The direction is a ground azimuth; on anisotropic cells the line follows the pixel direction
  // of the reference row instead (CPU twin: getTerrainSweepGroundLineGeometry).
  let referenceCell = ${t.referenceCellSizeIndex===void 0?`getGroundCellSize(${l})`:`vec2<f32>(settings[settingsOffset + ${t.referenceCellSizeIndex}u], settings[settingsOffset + ${t.referenceCellSizeIndex+1}u])`};
  if (referenceCell.x != referenceCell.y && all(referenceCell > vec2<f32>(0.0)) &&
      isFiniteValue(referenceCell.x + referenceCell.y) && any(groundDirection != vec2<f32>(0.0))) {
    let pixelDirection = groundDirection / referenceCell;
    xMajor = abs(pixelDirection.x) >= abs(pixelDirection.y);
    let majorComponent = select(pixelDirection.y, pixelDirection.x, xMajor);
    let minorComponent = select(pixelDirection.x, pixelDirection.y, xMajor);
    slopeFixed = i32(round((minorComponent / majorComponent) * ${P}.0));
    travelSign = select(-1, 1, majorComponent > 0.0);
    stepPixelLength = length(vec2<f32>(1.0, f32(slopeFixed) / ${P}.0));
  }`,p=[{name:`elevationValues`,view:t.elevationValues,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`hull`,view:t.hull,type:`u32`,access:`read_write`},...o.kind===`table`?[{name:`sectorRecord`,view:o.record,type:`u32`,access:`read`}]:[],...t.outputBindings];return a(e,{id:t.id,operation:`GPUTerrainHorizon`,variant:`sweep-${t.cellSizeMode}`,bindings:p,workgroupSize:L,invocationCount:n+r,declarations:`${ie(t)}
${o.kind===`baked`?`const GROUND_DIRECTION: vec2<f32> = vec2<f32>(${i(o.direction[0])}, ${i(o.direction[1])});`:``}`,body:`${f}
  let majorExtent = select(HEIGHT, WIDTH, xMajor);
  let minorExtent = select(WIDTH, HEIGHT, xMajor);
  let origin = ${t.originIndex===void 0?`0u`:`u32(settings[settingsOffset + ${t.originIndex}u + select(1u, 0u, xMajor)])`};
  // Unsigned arithmetic retains the exact low 16 product bits for either slope sign.
  sweepPhase = i32((bitcast<u32>(slopeFixed) * origin + 32768u) & 65535u);
  let lastOffset = getFixedOffset(slopeFixed, i32(majorExtent) - 1);
  let lineCount = i32(minorExtent) + abs(lastOffset);
  if (i32(index) >= lineCount) {
    return;
  }
  sweepXMajor = xMajor;
  sweepSlopeFixed = slopeFixed;
  sweepTravel = travelSign;
  sweepMajorExtent = i32(majorExtent);
  sweepMinorExtent = i32(minorExtent);
  sweepLine = i32(index) - max(0, lastOffset);
  sweepZ = f32(${t.zFactorSign}) * settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  sweepMaximumDistance = settings[settingsOffset + 6u];
  sweepStepPixelLength = stepPixelLength;
  let slope = f32(slopeFixed) / ${P}.0;
  let referenceStep = length(select(vec2<f32>(slope, 1.0), vec2<f32>(1.0, slope), xMajor) * referenceCell);
  sweepCurvature = curvature * referenceStep * referenceStep;
  var previous = NONE;
  for (var stepIndex = 0; stepIndex < sweepMajorExtent; stepIndex++) {
    let major = select(stepIndex, sweepMajorExtent - 1 - stepIndex, travelSign > 0);
    let minor = sweepLine + getFixedOffset(slopeFixed, major);
    if (minor < 0 || minor >= sweepMinorExtent) {
      continue;
    }
    let pixel = select(u32(major) * WIDTH + u32(minor), u32(minor) * WIDTH + u32(major), xMajor);
    let next = previous;
    previous = pixel;
    var firstAhead = NONE;
    if (next != NONE) {
      firstAhead = select(hull[hullOffset + next], next, isValidPixel(next));
    }
    var horizonAngle = getQuietNaN(pixel);
    if (!isValidPixel(pixel)) {
      hull[hullOffset + pixel] = firstAhead;
    } else {
      sweepElevation = elevationValues[elevationValuesOffset + pixel];
      sweepMajor = major;
      var tangent = firstAhead;
      if (tangent != NONE) {
        var chain = hull[hullOffset + tangent];
        while (chain != NONE && isGrowing(chain, tangent)) {
          tangent = chain;
          chain = hull[hullOffset + chain];
        }
      }
      hull[hullOffset + pixel] = tangent;
      let pixelStep = getStepLength(pixel / WIDTH, xMajor, slopeFixed);
      if (pixelStep > 0.0 && isFiniteValue(pixelStep)) {
        sweepPixelStep = pixelStep;
        var best = NONE;
        if (firstAhead != NONE && isInsideWindow(firstAhead)) {
          if (isInsideWindow(tangent)) {
            best = tangent;
          } else {
            best = queryWindow(firstAhead);
          }
        }
        if (best == NONE) {
          horizonAngle = 0.0;
        } else {
          let groundDistance = getDistanceIndex(best) * pixelStep;
          let rise = sweepZ * (elevationValues[elevationValuesOffset + best] - sweepElevation) -
            curvature * groundDistance * groundDistance;
          horizonAngle = atan(rise / groundDistance) * RADIANS_TO_DEGREES;
        }
      }
    }
    {
      ${t.outputWGSL}
    }
  }`})}function ie(e){return`const WIDTH: u32 = ${e.width}u;
const HEIGHT: u32 = ${e.height}u;
const NONE: u32 = 0xffffffffu;
const MAXIMUM_RADIUS: f32 = ${i(e.maximumRadius)};
${C}
${B(e.cellSizeMode,e.sourceGridIndex)}
var<private> sweepXMajor: bool;
var<private> sweepSlopeFixed: i32;
var<private> sweepTravel: i32;
var<private> sweepMajorExtent: i32;
var<private> sweepMinorExtent: i32;
var<private> sweepLine: i32;
var<private> sweepZ: f32;
var<private> sweepCurvature: f32;
var<private> sweepMaximumDistance: f32;
var<private> sweepStepPixelLength: f32;
var<private> sweepPixelStep: f32;
var<private> sweepElevation: f32;
var<private> sweepMajor: i32;
var<private> sweepPhase: i32;
fn getFixedOffset(slopeFixed: i32, major: i32) -> i32 {
  return (slopeFixed * major + sweepPhase) >> 16u;
}
// Ground length of one major step on the pixel row, using the row's cell size.
fn getStepLength(row: u32, xMajor: bool, slopeFixed: i32) -> f32 {
  let slope = f32(slopeFixed) / 65536.0;
  let cell = getGroundCellSize(row);
  return length(select(vec2<f32>(slope, 1.0), vec2<f32>(1.0, slope), xMajor) * cell);
}
// Canonical elevation is NaN exactly where a pixel is invalid.
fn isValidPixel(pixel: u32) -> bool {
  return isFiniteValue(elevationValues[elevationValuesOffset + pixel]);
}
fn getMajorIndex(pixel: u32) -> i32 {
  return i32(select(pixel / WIDTH, pixel % WIDTH, sweepXMajor));
}
fn getDistanceIndex(pixel: u32) -> f32 {
  return f32(abs(getMajorIndex(pixel) - sweepMajor));
}
// Hull-metric rise for the sample at distance index kf (curvature uses the reference step).
fn getRise(pixel: u32, kf: f32) -> f32 {
  return sweepZ * (elevationValues[elevationValuesOffset + pixel] - sweepElevation) -
    sweepCurvature * (kf * kf);
}
// True when sample r has a strictly larger tangent than sample c (cross-multiplied, exact ties).
fn isGrowing(r: u32, c: u32) -> bool {
  let kr = getDistanceIndex(r);
  let kc = getDistanceIndex(c);
  return getRise(r, kr) * kc > getRise(c, kc) * kr;
}
fn isInsideWindow(pixel: u32) -> bool {
  let kf = getDistanceIndex(pixel);
  return kf * sweepStepPixelLength <= MAXIMUM_RADIUS &&
    (sweepMaximumDistance <= 0.0 || kf * sweepPixelStep <= sweepMaximumDistance);
}
// First valid pixel strictly after c on the line, NONE at the end.
fn getValidAfter(c: u32) -> u32 {
  let major = getMajorIndex(c) + sweepTravel;
  if (major < 0 || major >= sweepMajorExtent) { return NONE; }
  let minor = sweepLine + getFixedOffset(sweepSlopeFixed, major);
  if (minor < 0 || minor >= sweepMinorExtent) { return NONE; }
  let pixel = select(u32(major) * WIDTH + u32(minor), u32(minor) * WIDTH + u32(major), sweepXMajor);
  return select(hull[hullOffset + pixel], pixel, isValidPixel(pixel));
}
// Exact maximum over the in-window samples at or after start (start is valid and inside).
fn queryWindow(start: u32) -> u32 {
  var best = NONE;
  var t = start;
  loop {
    if (t == NONE || !isInsideWindow(t)) { break; }
    var c = t;
    var r = hull[hullOffset + c];
    while (r != NONE && isInsideWindow(r) && isGrowing(r, c)) {
      c = r;
      r = hull[hullOffset + r];
    }
    if (best == NONE) {
      best = c;
    } else if (isGrowing(c, best)) {
      best = c;
    }
    if (r == NONE || isInsideWindow(r)) { break; }
    t = getValidAfter(c);
  }
  return best;
}
${e.outputDeclarations??``}`}var ae=2**53-1-1;function oe(e,t,n){let r=e.sweepReferenceCellSize!==void 0||e.sweepOrigin!==void 0||e.sweepSourceGrid!==void 0;o(`getGPUTerrainHorizonParameterValues`,!r||e.sweepReferenceCellSize!==void 0&&e.sweepSourceGrid!==void 0,`sweepReferenceCellSize and sweepSourceGrid are required for sweep settings`);let i=e.anisotropyAzimuthDegrees!==void 0||e.anisotropyLevel!==void 0||e.anisotropyMinimumWeight!==void 0,a=r||i||n!==void 0&&n.length>=12,s=n??new Float32Array(r?24:a?12:8);S(`getGPUTerrainHorizonParameterValues`,t),_(`getGPUTerrainHorizonParameterValues`,s,r?24:i?12:8);let c=e.anisotropyAzimuthDegrees??315,l=e.anisotropyLevel??4,u=e.anisotropyMinimumWeight??.4;if(a){if(!Number.isFinite(c))throw Error(`Terrain horizon anisotropy azimuth must be finite`);if(!Number.isSafeInteger(l)||l<0)throw Error(`Terrain horizon anisotropy level must be a non-negative integer`);if(!Number.isFinite(u)||u<0||u>1)throw Error(`Terrain horizon anisotropy minimum weight must be in [0, 1]`)}return s.set([0,0,e.zFactor??1,e.curvatureCoefficient??0,0,0,e.maximumDistance??0,0]),k(s,t,I),a&&s.set([c,l,u,0],8),e.sweepReferenceCellSize&&s.set(z(`getGPUTerrainHorizonParameterValues`,e,t),12),s}function se(e,t=1,n=e){K(`Terrain horizon`,e,t),o(`Terrain horizon`,Number.isFinite(n)&&n>=1,`maximumSampleDistance must be finite and >= 1`);let r=Math.min(e,n),i=[];for(let e=1;e<=r;)i.push(e),e=Math.max(e+1,e*t);return e<=n&&i[i.length-1]<e&&i.push(e),Float32Array.from(i)}function K(e,t,n){s(e,`maximumRadius`,t,1,2**53-1),o(e,Number.isFinite(n)&&n>=1,`stepGrowth must be finite and >= 1`)}function q(e,t,n=`south`){let r=2*Math.PI*e/t,i=n===`south`?-1:1,a=e=>Math.abs(e)<1e-9?0:Math.fround(e);return[a(Math.sin(r)),a(i*Math.cos(r))]}function J(e){let{sector:t,directionCount:n,mode:r}=e,a=e.horizonFormat??`float32`;if(r===`nadir`){if(e.horizon||e.sineSum||e.angleSum||e.anisotropicSum)throw Error(`Terrain horizon nadir output only accumulates nadirSum`)}else if(e.nadirSum)throw Error(`Terrain horizon zenith output cannot write nadirSum`);let o=[],s=[],c=``,l=e.sectorWGSL?`(${e.sectorWGSL})`:`${t}u`,u=(n,r)=>e.sectorWGSL?`${n} = select(${n} + ${r}, ${r}, ${l} == 0u);`:`${n} ${t===0?`=`:`+=`} ${r};`;if(e.horizon&&(a===`unorm16`?(o.push({name:`horizon`,view:e.horizon,type:`u32`,access:`read_write`}),c+=`fn encodeHorizonUnorm16(angle: f32) -> u32 {
  if (!isFiniteValue(angle)) { return 0u; }
  return 1u + u32(floor(clamp(angle, -90.0, 90.0) * ${i(E)} + 32767.0 + 0.5));
}
`,s.push(`let element = pixel * ${n}u + ${l};
  let wordIndex = horizonOffset + (element >> 1u);
  let shift = (element & 1u) * 16u;
  horizon[wordIndex] = (horizon[wordIndex] & ~(0xffffu << shift)) |
    (encodeHorizonUnorm16(horizonAngle) << shift);`)):(o.push({name:`horizon`,view:e.horizon,type:`f32`,access:`read_write`}),s.push(`horizon[horizonOffset + pixel * ${n}u + ${l}] = horizonAngle;`))),e.sineSum&&(o.push({name:`sineSum`,view:e.sineSum,type:`f32`,access:`read_write`}),s.push(u(`sineSum[sineSumOffset + pixel]`,`sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS)`))),e.angleSum&&(o.push({name:`angleSum`,view:e.angleSum,type:`f32`,access:`read_write`}),s.push(u(`angleSum[angleSumOffset + pixel]`,`horizonAngle`))),e.anisotropicSum&&(o.push({name:`anisotropicSum`,view:e.anisotropicSum,type:`f32`,access:`read_write`}),c+=Y(n),s.push(u(`anisotropicSum[anisotropicSumOffset + pixel]`,`getAnisotropicWeight(${l}) *
    sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS)`))),e.nadirSum&&(o.push({name:`nadirSum`,view:e.nadirSum,type:`f32`,access:`read_write`}),s.push(u(`nadirSum[nadirSumOffset + pixel]`,`horizonAngle`))),o.length===0)throw Error(`Terrain horizon sector output requires at least one output`);let d=o.length+(e.reservedBindingCount??3),f=e.maximumBindingCount??8;if(d>f)throw Error(`Terrain horizon sector output needs ${d} storage bindings, more than the limit of ${f}`);return{bindings:o,declarations:c,wgsl:`{\n  ${s.join(`
  `)}\n  }`}}function Y(e){return`fn getAnisotropicWeight(sector: u32) -> f32 {
  let mainDirection = settings[settingsOffset + 8u];
  let level = settings[settingsOffset + 9u];
  let minimumWeight = settings[settingsOffset + 10u];
  // Compass sector d is RVT direction (n - d) mod n; RVT angles run counterclockwise from north.
  let rvtDirection = f32((${e}u - sector) % ${e}u) * ${i(360/e)};
  let cosine = cos((rvtDirection - mainDirection) * 0.5 * DEGREES_TO_RADIANS);
  var power = 1.0;
  if (level > 0.0) {
    // pow is undefined for a zero or negative base; odd integer levels keep the sign like RVT.
    power = select(0.0, pow(abs(cosine), level), abs(cosine) > 0.0);
    if (cosine < 0.0 && fract(level * 0.5) != 0.0) { power = -power; }
  }
  return (1.0 - minimumWeight) * power + minimumWeight;
}
`}var ce=class{id;props;directionCount;stepDistances;horizonFormat;algorithm;parameterLayout;maximumRadius;rowDirection;constructor(e){e=d(e,[`id`,`grid`,`elevation`,`parameters`,`output`,`directionCount`,`maximumRadius`,`stepGrowth`,`algorithm`,`parameterLayout`,`maximumStepsPerDispatch`,`cellSizeMode`,`rowDirection`,`horizonFormat`]),this.id=e.id??`terrain-horizon`,this.props=e;let{id:t}=this,{output:n}=e,r=y(t,e.grid),i=e.directionCount??16;if(!Number.isSafeInteger(i)||i<4||i>64)throw Error(`${t} directionCount must be an integer in [4, 64]`);if(this.directionCount=i,K(t,e.maximumRadius,e.stepGrowth??1),this.maximumRadius=e.maximumRadius,this.parameterLayout=e.parameterLayout??`standard`,o(t,this.parameterLayout===`standard`||this.parameterLayout===`sweep`,`parameterLayout must be standard or sweep`),(e.algorithm??`march`)===`march`&&s(t,`maximumRadius`,this.maximumRadius,1,ae),!n.horizon&&!n.skyViewFactor&&!n.irradianceSkyViewFactor&&!n.positiveOpenness&&!n.negativeOpenness&&!n.anisotropicSkyViewFactor&&!n.outputValidity&&!n.skyViewFactorTexture)throw Error(`${t} requires at least one output`);if(w(t,e.horizonFormat??`float32`),this.horizonFormat=e.horizonFormat??`float32`,this.algorithm=e.algorithm??`march`,this.algorithm!==`march`&&this.algorithm!==`sweep`)throw Error(`${t} algorithm must be march or sweep`);if(this.algorithm===`sweep`){if((e.stepGrowth??1)!==1)throw Error(`${t} stepGrowth applies to the march algorithm only`);if(e.grid.width>32767||e.grid.height>32767)throw Error(`${t} sweep supports extents up to ${F}`)}if(this.stepDistances=this.algorithm===`march`?se(e.maximumRadius,e.stepGrowth??1,Math.max(1,Math.ceil(Math.hypot(e.grid.width-1,e.grid.height-1)))):new Float32Array,e.maximumStepsPerDispatch!==void 0&&!(Number.isFinite(e.maximumStepsPerDispatch)&&e.maximumStepsPerDispatch>0))throw Error(`${t} maximumStepsPerDispatch must be positive`);T(t,`output.horizon`,n.horizon,this.horizonFormat,r,i),D(t,`output.skyViewFactor`,n.skyViewFactor,`float32`,r),D(t,`output.irradianceSkyViewFactor`,n.irradianceSkyViewFactor,`float32`,r),D(t,`output.positiveOpenness`,n.positiveOpenness,`float32`,r),D(t,`output.negativeOpenness`,n.negativeOpenness,`float32`,r),D(t,`output.anisotropicSkyViewFactor`,n.anisotropicSkyViewFactor,`float32`,r),D(t,`output.outputValidity`,n.outputValidity,`uint32`,r),m(t,e.parameters,this.parameterLayout===`sweep`?24:n.anisotropicSkyViewFactor?12:8),v(t,`output.skyViewFactorTexture`,n.skyViewFactorTexture,[`r32float`,`rgba32float`],e.grid),M(t,e.cellSizeMode??`uniform`),p(t,e.grid,e.cellSizeMode),this.rowDirection=N(t,e.grid,e.rowDirection),h(t,[n.horizon,n.skyViewFactor,n.irradianceSkyViewFactor,n.positiveOpenness,n.negativeOpenness,n.anisotropicSkyViewFactor,n.outputValidity],[...g(e.elevation),e.parameters])}getHaloStage(){return o(this.id,(this.props.cellSizeMode??`uniform`)===`uniform`||this.parameterLayout===`sweep`,`latitude-dependent halo needs parameterLayout sweep with a shared sweepSourceGrid`),this.algorithm===`march`?{requiredHalo:this.maximumRadius+1}:(o(this.id,this.parameterLayout===`sweep`,`algorithm sweep needs parameterLayout sweep for a seamless halo bound`),{requiredHalo:this.maximumRadius})}getCommandNodes(t){e(t,this.id,this.props,{"canonical and hull planes":4*this.props.grid.width*this.props.grid.height,"horizon map":!this.props.output.horizon&&this.props.output.irradianceSkyViewFactor?4*this.props.grid.width*this.props.grid.height*this.directionCount:0,"sector table":this.algorithm===`sweep`||this.horizonFormat===`unorm16`&&this.directionCount%2==1?256*this.directionCount:256*Math.ceil(this.directionCount/Math.max(1,Math.min(this.directionCount,Math.floor((this.props.maximumStepsPerDispatch??$)/(this.props.grid.width*this.props.grid.height*this.stepDistances.length)))))});let{id:o,props:s,directionCount:c}=this,{output:d}=s,{width:p,height:m}=s.grid;f(o,t,s.elevation,[d.skyViewFactorTexture]),u(o,t,[s.parameters,d.horizon,d.skyViewFactor,d.irradianceSkyViewFactor,d.positiveOpenness,d.negativeOpenness,d.anisotropicSkyViewFactor,d.outputValidity]);let h=p*m,g=d.horizon?this.horizonFormat:`float32`,_=d.horizon??(d.irradianceSkyViewFactor?n(t,`${o}-horizon-map`,`float32`,x(`float32`,h,c)):void 0);_&&b(o,`horizon`,x(g,h,c)*4,t.device.limits.maxStorageBufferBindingSize);let v=ee(t,o,s.elevation,s.grid,!0),y=[...v.nodes],S=v.band.storage.values,w=v.band.validity,T=d.skyViewFactor??(d.skyViewFactorTexture?n(t,`${o}-sky-view-factor`,`float32`,h):void 0),E=!!(T||d.positiveOpenness),D=E?n(t,`${o}-sine-sum`,`float32`,h):void 0,O=E?n(t,`${o}-angle-sum`,`float32`,h):void 0,k=d.anisotropicSkyViewFactor?n(t,`${o}-anisotropic-sum`,`float32`,h):void 0,A=d.negativeOpenness?n(t,`${o}-nadir-sum`,`float32`,h):void 0,j=this.rowDirection,M=s.cellSizeMode??`uniform`,N=this.algorithm===`march`?Array.from(this.stepDistances,i).join(`, `):``,P=this.algorithm===`sweep`?n(t,`${o}-sweep-hull`,`uint32`,h):void 0,F=!P&&(this.horizonFormat!==`unorm16`||c%2!=1);F&&y.push(...ue(t,{id:o,width:p,height:m,directionCount:c,rowDirection:j,distances:N,stepCount:this.stepDistances.length,sourceGridIndex:this.parameterLayout===`sweep`?16:void 0,maximumStepsPerDispatch:s.maximumStepsPerDispatch??$,cellSizeMode:M,elevationValues:S,settings:s.parameters,horizonFormat:g,horizon:_,sineSum:D,angleSum:O,anisotropicSum:k,nadirSum:A}));let I=!F&&!P?Z(t,{id:o,directionCount:c,sectorsPerRange:1}):void 0;I&&y.push(I.writeNode);let L=P?re(t,{id:o,width:p,height:m,directions:Array.from({length:c},(e,t)=>q(t,c,j))}):void 0;L&&y.push(L.writeNode);for(let e=0;!F&&e<c;e++)P?y.push(...le(t,{id:`${o}-horizon-${e}`,width:p,height:m,sectorRecord:L.getRecord(e),directionCount:c,maximumRadius:this.maximumRadius,originIndex:this.parameterLayout===`sweep`?14:void 0,sourceGridIndex:this.parameterLayout===`sweep`?16:void 0,referenceCellSizeIndex:this.parameterLayout===`sweep`?12:void 0,cellSizeMode:M,elevationValues:S,settings:s.parameters,hull:P,horizonFormat:g,horizon:_,sineSum:D,angleSum:O,anisotropicSum:k,nadirSum:A})):y.push(de(t,{id:`${o}-horizon-${e}`,width:p,height:m,sectorRange:I.getRange(e),directionCount:c,rowDirection:j,distances:N,stepCount:this.stepDistances.length,sourceGridIndex:this.parameterLayout===`sweep`?16:void 0,cellSizeMode:M,elevationValues:S,settings:s.parameters,horizonFormat:g,horizon:_,sineSum:D,angleSum:O,anisotropicSum:k,nadirSum:A}));if(T||d.positiveOpenness||d.outputValidity){let e=[{name:`elevationValidity`,view:w,type:`u32`,access:`read`}];D&&O&&e.push({name:`sineSum`,view:D,type:`f32`,access:`read`},{name:`angleSum`,view:O,type:`f32`,access:`read`}),T&&e.push({name:`skyView`,view:T,type:`f32`,access:`read_write`}),d.positiveOpenness&&e.push({name:`openness`,view:d.positiveOpenness,type:`f32`,access:`read_write`}),d.outputValidity&&e.push({name:`validityValues`,view:d.outputValidity,type:`u32`,access:`read_write`}),y.push(a(t,{id:`${o}-sky-view`,operation:`GPUTerrainHorizon`,variant:`sky-view`,bindings:e,invocationCount:h,declarations:`const DIRECTION_COUNT: f32 = ${c}.0;
${C}`,body:`let isValid = elevationValidity[elevationValidityOffset + index] != 0u${D?` &&
    isFiniteValue(angleSum[angleSumOffset + index])`:``};
  let invalidValue = getQuietNaN(index);
  ${T?`skyView[skyViewOffset + index] = select(invalidValue,
    1.0 - sineSum[sineSumOffset + index] / DIRECTION_COUNT, isValid);`:``}
  ${d.positiveOpenness?`openness[opennessOffset + index] = select(invalidValue,
    90.0 - angleSum[angleSumOffset + index] / DIRECTION_COUNT, isValid);`:``}
  ${d.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`}))}if(d.irradianceSkyViewFactor&&_&&y.push(a(t,{id:`${o}-irradiance-sky-view`,operation:`GPUTerrainHorizon`,variant:`irradiance-sky-view-${g}`,bindings:[{name:`elevationValidity`,view:w,type:`u32`,access:`read`},{name:`horizon`,view:_,type:g===`unorm16`?`u32`:`f32`,access:`read`},{name:`irradianceSkyView`,view:d.irradianceSkyViewFactor,type:`f32`,access:`read_write`}],invocationCount:h,declarations:`const DIRECTION_COUNT: u32 = ${c}u;
${C}
${te(g,`horizon`)}`,body:`var sineSquaredSum = 0.0;
  for (var sector = 0u; sector < DIRECTION_COUNT; sector++) {
    let sine = sin(max(readHorizonAngle(index * DIRECTION_COUNT + sector), 0.0) * DEGREES_TO_RADIANS);
    sineSquaredSum += sine * sine;
  }
  let isValid = elevationValidity[elevationValidityOffset + index] != 0u &&
    isFiniteValue(sineSquaredSum);
  irradianceSkyView[irradianceSkyViewOffset + index] = select(getQuietNaN(index),
    1.0 - sineSquaredSum / f32(DIRECTION_COUNT), isValid);`})),k&&d.anisotropicSkyViewFactor||A&&d.negativeOpenness){let e=[{name:`elevationValidity`,view:w,type:`u32`,access:`read`}],n=`elevationValidity[elevationValidityOffset + index] != 0u`,r=``;k&&d.anisotropicSkyViewFactor&&(e.push({name:`settings`,view:s.parameters,type:`f32`,access:`read`},{name:`anisotropicSum`,view:k,type:`f32`,access:`read`},{name:`anisotropicSkyView`,view:d.anisotropicSkyViewFactor,type:`f32`,access:`read_write`}),n+=` &&
    isFiniteValue(anisotropicSum[anisotropicSumOffset + index])`,r+=`var weightSum = 0.0;
  for (var sector = 0u; sector < DIRECTION_COUNT_U32; sector++) {
    weightSum += getAnisotropicWeight(sector);
  }
  anisotropicSkyView[anisotropicSkyViewOffset + index] = select(invalidValue,
    1.0 - anisotropicSum[anisotropicSumOffset + index] / weightSum, isValid);
  `),A&&d.negativeOpenness&&(e.push({name:`nadirSum`,view:A,type:`f32`,access:`read`},{name:`negativeOpennessValues`,view:d.negativeOpenness,type:`f32`,access:`read_write`}),n+=` &&
    isFiniteValue(nadirSum[nadirSumOffset + index])`,r+=`negativeOpennessValues[negativeOpennessValuesOffset + index] = select(invalidValue,
    90.0 - nadirSum[nadirSumOffset + index] / DIRECTION_COUNT, isValid);
  `),y.push(a(t,{id:`${o}-extended`,operation:`GPUTerrainHorizon`,variant:`extended`,bindings:e,invocationCount:h,declarations:`const DIRECTION_COUNT: f32 = ${c}.0;
const DIRECTION_COUNT_U32: u32 = ${c}u;
${C}
${k?Y(c):``}`,body:`let isValid = ${n};
  let invalidValue = getQuietNaN(index);
  ${r}`}))}if(d.skyViewFactorTexture&&T){let e=d.skyViewFactorTexture;y.push(...l(t,()=>new r({id:`${o}-sky-view-texture`,input:{id:`${o}-sky-view-band`,format:`float32`,storage:{kind:`buffer`,values:T}},output:e,channel:0}).addToGraph(t)))}return y}};function le(e,t){let n={width:t.width,height:t.height,geometry:{kind:`table`,record:t.sectorRecord},maximumRadius:t.maximumRadius,originIndex:t.originIndex,sourceGridIndex:t.sourceGridIndex,referenceCellSizeIndex:t.referenceCellSizeIndex,cellSizeMode:t.cellSizeMode,elevationValues:t.elevationValues,settings:t.settings,hull:t.hull},r=[];if(t.horizon||t.sineSum||t.angleSum||t.anisotropicSum){let i=J({sector:0,sectorWGSL:W,directionCount:t.directionCount,horizonFormat:t.horizonFormat,horizon:t.horizon,sineSum:t.sineSum,angleSum:t.angleSum,anisotropicSum:t.anisotropicSum,mode:`zenith`,reservedBindingCount:4});r.push(G(e,{...n,id:t.id,zFactorSign:1,outputBindings:i.bindings,outputDeclarations:i.declarations,outputWGSL:i.wgsl}))}if(t.nadirSum){let i=J({sector:0,sectorWGSL:W,directionCount:t.directionCount,nadirSum:t.nadirSum,mode:`nadir`,reservedBindingCount:4});r.push(G(e,{...n,id:`${t.id}-nadir`,zFactorSign:-1,outputBindings:i.bindings,outputDeclarations:i.declarations,outputWGSL:i.wgsl}))}return r}var X=64;function Z(e,t){let{directionCount:r,sectorsPerRange:i}=t,o=Math.ceil(r/i),s=n(e,`${t.id}-sector-range-words`,`uint32`,o*X);return{rangeCount:o,writeNode:a(e,{id:`${t.id}-sector-ranges`,operation:`GPUTerrainHorizon`,variant:`sector-ranges`,bindings:[{name:`ranges`,view:s,type:`u32`,access:`read_write`}],invocationCount:o,body:`let first = index * ${i}u;
  ranges[rangesOffset + index * ${X}u] = first;
  ranges[rangesOffset + index * ${X}u + 1u] = min(${r}u, first + ${i}u);`}),getRange:t=>e.createDataView(s.buffer,{format:`uint32`,length:2,byteOffset:s.byteOffset+t*X*4})}}var Q=`// Canonical elevation is NaN exactly where a pixel is invalid, so the march kernels need no
// validity binding (its slot holds the sector range word).
fn isValidPixel(column: u32, row: u32) -> bool {
  return isFiniteValue(elevationValues[elevationValuesOffset + row * WIDTH + column]);
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear elevation relative to \`origin\` at an in-grid pixel-center position; .y is 0 when any
// corner is invalid. Corners are made origin-relative before interpolating, so the interpolation
// rounds at the scale of the local relief instead of the absolute elevation (at 4000 m the absolute
// form loses about 2.4e-4 m, 1.4e-3 degrees over one 10 m step).
fn sampleElevation(base: vec2<u32>, fraction: vec2<f32>, origin: f32) -> vec2<f32> {
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  // Canonical elevation is NaN exactly where a pixel is invalid: one bit test per corner replaces
  // four validity loads.
  let corner00 = getElevation(base.x, base.y);
  let corner10 = getElevation(next.x, base.y);
  let corner01 = getElevation(base.x, next.y);
  let corner11 = getElevation(next.x, next.y);
  if (!isFiniteValue(corner00) || !isFiniteValue(corner10) ||
      !isFiniteValue(corner01) || !isFiniteValue(corner11)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(corner00 - origin, corner10 - origin, fraction.x);
  let bottom = mix(corner01 - origin, corner11 - origin, fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}`,$=2**30;function ue(e,n){let{directionCount:r}=n,o=n.horizon||n.sineSum||n.angleSum||n.anisotropicSum?J({sector:0,sectorWGSL:`sector`,directionCount:r,horizonFormat:n.horizonFormat,horizon:n.horizon,sineSum:n.sineSum,angleSum:n.angleSum,anisotropicSum:n.anisotropicSum,mode:`zenith`,reservedBindingCount:3+ +!!n.nadirSum}):void 0,s=n.nadirSum?J({sector:0,sectorWGSL:`sector`,directionCount:r,nadirSum:n.nadirSum,mode:`nadir`,reservedBindingCount:3+(o?.bindings.length??0)}):void 0,c=[{name:`elevationValues`,view:n.elevationValues,type:`f32`,access:`read`},{name:`settings`,view:n.settings,type:`f32`,access:`read`},...o?.bindings??[],...s?.bindings??[]],l=!!s,u=n.width*n.height,d=t(Math.floor(n.maximumStepsPerDispatch/(u*n.stepCount)),1,r),f=Array.from({length:r},(e,t)=>{let[a,o]=q(t,r,n.rowDirection);return`vec2<f32>(${i(a)}, ${i(o)})`}).join(`, `),p=n.horizonFormat,m=[];n.horizon&&m.push(p===`unorm16`?`// Both sectors of a word belong to this pixel (even sector count), so no other thread shares it.
      let element = pixel * DIRECTION_COUNT + sector;
      let wordIndex = horizonOffset + (element >> 1u);
      let shift = (element & 1u) * 16u;
      horizon[wordIndex] = (horizon[wordIndex] & ~(0xffffu << shift)) |
        (encodeHorizonUnorm16(horizonAngle) << shift);`:`horizon[horizonOffset + pixel * DIRECTION_COUNT + sector] = horizonAngle;`),(n.sineSum||n.anisotropicSum)&&m.push(`let sineTerm = sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS);`,...n.sineSum?[`sineAccumulator = select(sineAccumulator + sineTerm, sineTerm, sector == 0u);`]:[],...n.anisotropicSum?[`let anisotropicTerm = getAnisotropicWeight(sector) * sineTerm;`,`anisotropicAccumulator = select(anisotropicAccumulator + anisotropicTerm, anisotropicTerm, sector == 0u);`]:[]),n.angleSum&&m.push(`angleAccumulator = select(angleAccumulator + horizonAngle, horizonAngle, sector == 0u);`);let h=[n.sineSum?`var sineAccumulator = select(0.0, sineSum[sineSumOffset + pixel], firstSector > 0u);`:``,n.angleSum?`var angleAccumulator = select(0.0, angleSum[angleSumOffset + pixel], firstSector > 0u);`:``,n.anisotropicSum?`var anisotropicAccumulator = select(0.0, anisotropicSum[anisotropicSumOffset + pixel], firstSector > 0u);`:``,n.nadirSum?`var nadirAccumulator = select(0.0, nadirSum[nadirSumOffset + pixel], firstSector > 0u);`:``].join(`
  `),g=[n.sineSum?`sineSum[sineSumOffset + pixel] = sineAccumulator;`:``,n.angleSum?`angleSum[angleSumOffset + pixel] = angleAccumulator;`:``,n.anisotropicSum?`anisotropicSum[anisotropicSumOffset + pixel] = anisotropicAccumulator;`:``,n.nadirSum?`nadirSum[nadirSumOffset + pixel] = nadirAccumulator;`:``].join(`
  `),_=Z(e,{id:n.id,directionCount:r,sectorsPerRange:d}),v=[_.writeNode];for(let t=0;t<_.rangeCount;t++)v.push(a(e,{id:`${n.id}-horizon-march-${t}`,operation:`GPUTerrainHorizon`,variant:`march-fused-${n.cellSizeMode}${l?`-nadir`:``}`,bindings:[...c,{name:`sectorRange`,view:_.getRange(t),type:`u32`,access:`read`}],invocationCount:u,declarations:`const WIDTH: u32 = ${n.width}u;
const HEIGHT: u32 = ${n.height}u;
const DIRECTION_COUNT: u32 = ${r}u;
const DIRECTIONS: array<vec2<f32>, ${r}> = array<vec2<f32>, ${r}>(${f});
const STEP_COUNT: u32 = ${n.stepCount}u;
const STEP_DISTANCES: array<f32, ${n.stepCount}> = array<f32, ${n.stepCount}>(${n.distances});
${C}
${o?.declarations??``}
${B(n.cellSizeMode,n.sourceGridIndex)}
${Q}
${A}`,body:`let pixel = index;
  let firstSector = sectorRange[sectorRangeOffset];
  let endSector = sectorRange[sectorRangeOffset + 1u];
  let column = index % WIDTH;
  let row = index / WIDTH;
  let zFactor = settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  let maximumDistance = settings[settingsOffset + 6u];
  let groundCell = getGroundCellSize(row);
  let centerValid = isValidPixel(column, row);
  let centerElevation = select(0.0, getElevation(column, row), centerValid);
  let limit = vec2<i32>(i32(WIDTH - 1u), i32(HEIGHT - 1u));
  let center = vec2<i32>(i32(column), i32(row));
  ${h}
  for (var sector = firstSector; sector < endSector; sector++) {
    // DIRECTIONS are ground azimuths; the pixel step follows them on anisotropic cells. The ray
    // keeps this row's cell size (see GPUTerrainHorizonAlgorithm for the along-ray error).
    let direction = getPixelDirection(DIRECTIONS[sector], groundCell);
    let groundStep = length(direction * groundCell);
    var horizonAngle = getQuietNaN(index);
    var nadirAngle = getQuietNaN(index);
    if (centerValid && groundStep > 0.0 && isFiniteValue(groundStep)) {
      var hasSample = false;
      var maximumTangent = 0.0;
      var maximumNadirTangent = 0.0;
      for (var stepIndex = 0u; stepIndex < STEP_COUNT; stepIndex++) {
        let stepDistance = STEP_DISTANCES[stepIndex];
        let displacement = direction * stepDistance;
        let integerDisplacement = floor(displacement);
        let fraction = displacement - integerDisplacement;
        let base = center + vec2<i32>(integerDisplacement);
        if (any(base < vec2<i32>(0)) || any(base > limit) ||
            (base.x == limit.x && fraction.x > 0.0) ||
            (base.y == limit.y && fraction.y > 0.0)) { break; }
        let groundDistance = stepDistance * groundStep;
        if (maximumDistance > 0.0 && groundDistance > maximumDistance) { break; }
        let elevationSample = sampleElevation(vec2<u32>(base), fraction, centerElevation);
        if (elevationSample.y == 0.0) { continue; }
        let rise = zFactor * elevationSample.x -
          curvature * groundDistance * groundDistance;
        let tangent = rise / groundDistance;
        if (!hasSample || tangent > maximumTangent) { maximumTangent = tangent; }
        ${l?`// Inverted terrain: the elevation difference flips sign, the curvature drop does not.
        let nadirRise = -zFactor * elevationSample.x -
          curvature * groundDistance * groundDistance;
        let nadirTangent = nadirRise / groundDistance;
        if (!hasSample || nadirTangent > maximumNadirTangent) { maximumNadirTangent = nadirTangent; }`:``}
        hasSample = true;
      }
      horizonAngle = select(0.0, atan(maximumTangent) * RADIANS_TO_DEGREES, hasSample);
      ${l?`nadirAngle = select(0.0, atan(maximumNadirTangent) * RADIANS_TO_DEGREES, hasSample);`:``}
    }
    ${m.join(`
    `)}
    ${l?`nadirAccumulator = select(nadirAccumulator + nadirAngle, nadirAngle, sector == 0u);`:``}
  }
  ${g}`}));return v}function de(e,t){let n=[{name:`elevationValues`,view:t.elevationValues,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`sectorRange`,view:t.sectorRange,type:`u32`,access:`read`}],r=t.horizon||t.sineSum||t.angleSum||t.anisotropicSum?J({sector:0,sectorWGSL:`sector`,directionCount:t.directionCount,horizonFormat:t.horizonFormat,horizon:t.horizon,sineSum:t.sineSum,angleSum:t.angleSum,anisotropicSum:t.anisotropicSum,mode:`zenith`,reservedBindingCount:3+ +!!t.nadirSum}):void 0,o=t.nadirSum?J({sector:0,sectorWGSL:`sector`,directionCount:t.directionCount,nadirSum:t.nadirSum,mode:`nadir`,reservedBindingCount:3+(r?.bindings.length??0)}):void 0;n.push(...r?.bindings??[],...o?.bindings??[]);let s=!!o,c=Array.from({length:t.directionCount},(e,n)=>{let[r,a]=q(n,t.directionCount,t.rowDirection);return`vec2<f32>(${i(r)}, ${i(a)})`}).join(`, `);return a(e,{id:t.id,operation:`GPUTerrainHorizon`,variant:`march-${t.cellSizeMode}${s?`-nadir`:``}`,bindings:n,invocationCount:t.width*t.height,declarations:`const WIDTH: u32 = ${t.width}u;
const HEIGHT: u32 = ${t.height}u;
const DIRECTION_COUNT: u32 = ${t.directionCount}u;
const DIRECTIONS: array<vec2<f32>, ${t.directionCount}> = array<vec2<f32>, ${t.directionCount}>(${c});
const STEP_COUNT: u32 = ${t.stepCount}u;
const STEP_DISTANCES: array<f32, ${t.stepCount}> = array<f32, ${t.stepCount}>(${t.distances});
${C}
${r?.declarations??``}
${B(t.cellSizeMode,t.sourceGridIndex)}
${Q}
${A}`,body:`let pixel = index;
  let sector = sectorRange[sectorRangeOffset];
  let column = index % WIDTH;
  let row = index / WIDTH;
  let zFactor = settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  let maximumDistance = settings[settingsOffset + 6u];
  let groundCell = getGroundCellSize(row);
  let direction = getPixelDirection(DIRECTIONS[sector], groundCell);
  let groundStep = length(direction * groundCell);
  var horizonAngle = getQuietNaN(index);
  var nadirAngle = getQuietNaN(index);
  if (isValidPixel(column, row) && groundStep > 0.0 && isFiniteValue(groundStep)) {
    let center = vec2<i32>(i32(column), i32(row));
    let centerElevation = getElevation(column, row);
    let limit = vec2<i32>(i32(WIDTH - 1u), i32(HEIGHT - 1u));
    var hasSample = false;
    var maximumTangent = 0.0;
    var maximumNadirTangent = 0.0;
    for (var stepIndex = 0u; stepIndex < STEP_COUNT; stepIndex++) {
      let stepDistance = STEP_DISTANCES[stepIndex];
      let displacement = direction * stepDistance;
      let integerDisplacement = floor(displacement);
      let fraction = displacement - integerDisplacement;
      let base = center + vec2<i32>(integerDisplacement);
      if (any(base < vec2<i32>(0)) || any(base > limit) ||
          (base.x == limit.x && fraction.x > 0.0) ||
          (base.y == limit.y && fraction.y > 0.0)) { break; }
      let groundDistance = stepDistance * groundStep;
      if (maximumDistance > 0.0 && groundDistance > maximumDistance) { break; }
      let elevationSample = sampleElevation(vec2<u32>(base), fraction, centerElevation);
      if (elevationSample.y == 0.0) { continue; }
      let rise = zFactor * elevationSample.x -
        curvature * groundDistance * groundDistance;
      let tangent = rise / groundDistance;
      if (!hasSample || tangent > maximumTangent) { maximumTangent = tangent; }
      ${s?`// Inverted terrain: the elevation difference flips sign, the curvature drop does not.
      let nadirRise = -zFactor * elevationSample.x -
        curvature * groundDistance * groundDistance;
      let nadirTangent = nadirRise / groundDistance;
      if (!hasSample || nadirTangent > maximumNadirTangent) { maximumNadirTangent = nadirTangent; }`:``}
      hasSample = true;
    }
    horizonAngle = select(0.0, atan(maximumTangent) * RADIANS_TO_DEGREES, hasSample);
    ${s?`nadirAngle = select(0.0, atan(maximumNadirTangent) * RADIANS_TO_DEGREES, hasSample);`:``}
  }
  ${r?.wgsl??``}
  ${o?`{\n  let horizonAngle = nadirAngle;\n  ${o.wgsl}\n  }`:``}`})}export{G as a,F as i,oe as n,V as o,I as r,z as s,ce as t};
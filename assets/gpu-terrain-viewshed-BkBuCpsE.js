import{n as e,r as t}from"./terrain-binding-preflight-laSEObQE.js";import{n,t as r}from"./vertical-datum-BTN03tZP.js";import{r as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./gpu-raster-buffer-to-texture-BDZQU4A5.js";import{o}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as s,i as c,x as l}from"./gpu-contributor-utils-BIwLQON5.js";import{a as u,n as d,r as f}from"./gpu-raster-extrema-pyramid-BYXR99nY.js";import{c as p,d as m,h,l as g,n as _,o as v,p as y,r as b,t as x,u as S}from"./raster-band-utils-BcTxc8_Y.js";import{r as C,u as w}from"./raster-grid-VB0AM_GY.js";import{n as T,t as E}from"./terrain-sight-line-QKhP0RJG.js";var D={hidden:0,visible:1,outOfRange:2,noData:3,marginal:4};function O(e,t,n=new Float32Array(10)){v(`getGPUTerrainViewshedParameterValues`,n,10);let{cellSize:r,northEdge:i,southEdge:a}=C(t);if(!r.every(e=>Number.isFinite(e)&&e>0))throw Error(`getGPUTerrainViewshedParameterValues grid cell size must be finite and positive`);return n.set([e.observer[0],e.observer[1],e.observerHeight??1.7,e.targetHeight??0,e.maximumDistance??0,r[0],r[1],e.curvatureCoefficient??0,i,a]),n}var k=class{id;props;pyramidLayout;constructor(e){this.id=e.id??`terrain-viewshed`,this.props=e;let{id:t}=this;if(e.heightReference!==void 0&&e.heightReference!==`ground`&&e.heightReference!==`absolute`)throw Error(`${t} heightReference must be 'ground' or 'absolute'`);if(e.observerVerticalDatum!==void 0&&e.heightReference!==`absolute`)throw Error(`${t} observerVerticalDatum needs heightReference absolute`);r(t,e.observerVerticalDatum,n(e.elevation,e.elevationVerticalDatum));let{output:i}=e,a=S(t,e.grid);if(w(t,e.cellSizeMode),m(t,e.grid,e.cellSizeMode),!i.visibility&&!i.visibilityTexture)throw Error(`${t} requires at least one output`);if(i.visibility&&(s(i.visibility,[`uint32`],`${t} output.visibility`),i.visibility.length!==a))throw Error(`${t} output.visibility must contain one value per pixel`);if(h(t,e.parameters,10),y(t,`output.visibilityTexture`,i.visibilityTexture,[`r32uint`,`rgba32uint`],e.grid),e.traversal!==void 0&&e.traversal!==`march`&&e.traversal!==`pyramid`)throw Error(`${t} traversal must be 'march' or 'pyramid'`);if(e.toleranceParameters&&h(t,e.toleranceParameters,4,`toleranceParameters`),e.pyramid&&e.traversal!==`pyramid`)throw Error(`${t} pyramid requires traversal 'pyramid'`);e.traversal===`pyramid`&&(this.pyramidLayout=e.pyramid?u(t,e.pyramid,e.grid):f(e.grid,{firstBlockSize:4,footprint:`bilinear`})),g(t,[i.visibility],[..._(e.elevation),e.parameters,...e.toleranceParameters?[e.toleranceParameters]:[]])}getCommandNodes(n){t(n,this.id,this.props,{"canonical elevation":4*this.props.grid.width*this.props.grid.height,"pyramid fused parameters":e(this.props.pyramid?void 0:this.pyramidLayout),pyramid:this.pyramidLayout&&!this.props.pyramid?8*this.pyramidLayout.length:0});let{id:r,props:s}=this,{output:u}=s,{width:d,height:f}=s.grid,m=s.cellSizeMode??`uniform`,h=m===`uniform`?``:E(m);p(r,n,s.elevation,[u.visibilityTexture]),l(r,n,[s.pyramid?.combined,s.parameters,u.visibility,...s.toleranceParameters?[s.toleranceParameters]:[]]);let g=d*f,_=b(n,r,s.elevation,s.grid,!0),v=[..._.nodes],y=u.visibility??i(n,`${r}-visibility`,`uint32`,g),S=_.band.validity;if(s.traversal===`pyramid`||s.toleranceParameters?v.push(...this._getSightLineNodes(n,_.band.storage.values,S,y)):v.push(o(n,{id:`${r}-viewshed`,operation:`GPUTerrainViewshed`,variant:`line-of-sight${m===`uniform`?``:`-${m}`}${s.heightReference===`absolute`?`-absolute`:``}`,bindings:[{name:`elevationValues`,view:_.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:S,type:`u32`,access:`read`},{name:`settings`,view:s.parameters,type:`f32`,access:`read`},{name:`visibilityValues`,view:y,type:`u32`,access:`read_write`}],invocationCount:g,declarations:`const WIDTH: u32 = ${d}u;
const HEIGHT: u32 = ${f}u;
const HIDDEN: u32 = ${D.hidden}u;
const VISIBLE: u32 = ${D.visible}u;
const OUT_OF_RANGE: u32 = ${D.outOfRange}u;
const NO_DATA: u32 = ${D.noData}u;
${x}${h}
fn isValidPixel(column: u32, row: u32) -> bool {
  return elevationValidity[elevationValidityOffset + row * WIDTH + column] != 0u;
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear elevation at a clamped pixel-center position; .y is 0 when any corner is invalid.
fn sampleElevation(position: vec2<f32>) -> vec2<f32> {
  let maximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let clamped = clamp(position, vec2<f32>(0.0), maximum);
  let base = vec2<u32>(floor(clamped));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = clamped - floor(clamped);
  // Canonical elevation is NaN exactly where a pixel is invalid, so one bit test per corner
  // replaces four validity loads; the interpolation below is unchanged.
  let corner00 = getElevation(base.x, base.y);
  let corner10 = getElevation(next.x, base.y);
  let corner01 = getElevation(base.x, next.y);
  let corner11 = getElevation(next.x, next.y);
  if (!isFiniteValue(corner00) || !isFiniteValue(corner10) ||
      !isFiniteValue(corner01) || !isFiniteValue(corner11)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(corner00, corner10, fraction.x);
  let bottom = mix(corner01, corner11, fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}`,body:`let column = index % WIDTH;
  let row = index / WIDTH;
  let observer = vec2<f32>(settings[settingsOffset], settings[settingsOffset + 1u]);
  let observerHeight = settings[settingsOffset + 2u];
  let targetHeight = settings[settingsOffset + 3u];
  let maxDistance = settings[settingsOffset + 4u];
  let cellSize = ${m===`uniform`?`vec2<f32>(settings[settingsOffset + 5u], settings[settingsOffset + 6u])`:`getRayGroundCell(vec2<f32>(settings[settingsOffset + 5u], settings[settingsOffset + 6u]), settings[settingsOffset + 8u], settings[settingsOffset + 9u], 0.5 * (observer.y + f32(row)) + 0.5)`};
  let curvature = settings[settingsOffset + 7u];
  let observerInside = isFiniteValue(observer.x) && isFiniteValue(observer.y) &&
    observer.x >= 0.0 && observer.y >= 0.0 &&
    observer.x <= f32(WIDTH - 1u) && observer.y <= f32(HEIGHT - 1u);
${s.heightReference===`absolute`?``:`  let observerSample = sampleElevation(select(vec2<f32>(0.0), observer, observerInside));
`}  if (!observerInside || ${s.heightReference===`absolute`?``:`observerSample.y == 0.0 || `}!isValidPixel(column, row)) {
    visibilityValues[visibilityValuesOffset + index] = NO_DATA;
    return;
  }
  let observerElevation = ${s.heightReference===`absolute`?`observerHeight`:`observerSample.x + observerHeight`};
  let delta = vec2<f32>(f32(column), f32(row)) - observer;
  let targetDistance = length(delta * cellSize);
  if (maxDistance > 0.0 && targetDistance > maxDistance) {
    visibilityValues[visibilityValuesOffset + index] = OUT_OF_RANGE;
    return;
  }
  if (targetDistance == 0.0) {
    visibilityValues[visibilityValuesOffset + index] = VISIBLE;
    return;
  }
  let targetElevation = getElevation(column, row) + targetHeight -
    curvature * targetDistance * targetDistance;
  let targetSlope = (targetElevation - observerElevation) / targetDistance;
  let stepCount = u32(ceil(max(abs(delta.x), abs(delta.y))));
  var visible = true;
  for (var stepIndex = 1u; stepIndex < stepCount; stepIndex++) {
    let fraction = f32(stepIndex) / f32(stepCount);
    let elevationSample = sampleElevation(observer + delta * fraction);
    // Invalid samples never block.
    if (elevationSample.y == 0.0) { continue; }
    let sampleDistance = targetDistance * fraction;
    let sampleSlope =
      (elevationSample.x - curvature * sampleDistance * sampleDistance - observerElevation) / sampleDistance;
    // A grazing (equal) sight line stays visible.
    if (sampleSlope > targetSlope) {
      visible = false;
      break;
    }
  }
  visibilityValues[visibilityValuesOffset + index] = select(HIDDEN, VISIBLE, visible);`})),u.visibilityTexture){let e=u.visibilityTexture;v.push(...c(n,()=>new a({id:`${r}-visibility-texture`,input:{id:`${r}-visibility-band`,format:`uint32`,storage:{kind:`buffer`,values:y}},output:e,channel:0}).addToGraph(n)))}return v}_getSightLineNodes(e,t,n,r){let{id:a,props:s,pyramidLayout:c}=this,{width:l,height:u}=s.grid,f=s.cellSizeMode??`uniform`,p=f===`uniform`?``:E(f),m=[],h=[{name:`elevationValues`,view:t,type:`f32`,access:`read`},{name:`elevationValidity`,view:n,type:`u32`,access:`read`}];if(c&&s.pyramid)h.push({name:`pyramid`,view:s.pyramid.combined,type:`f32`,access:`read`});else if(c){let r=i(e,`${a}-pyramid`,`float32`,2*c.length);m.push(...d(e,{id:`${a}-pyramid`,layout:c,values:t,validity:n,combined:r})),h.push({name:`pyramid`,view:r,type:`f32`,access:`read`})}return h.push({name:`settings`,view:s.parameters,type:`f32`,access:`read`}),s.toleranceParameters&&h.push({name:`toleranceBand`,view:s.toleranceParameters,type:`f32`,access:`read`}),h.push({name:`visibilityValues`,view:r,type:`u32`,access:`read_write`}),m.push(o(e,{id:`${a}-viewshed`,operation:`GPUTerrainViewshed`,variant:`${s.traversal===`pyramid`?`pyramid`:`tolerance`}${f===`uniform`?``:`-${f}`}${s.heightReference===`absolute`?`-absolute`:``}`,bindings:h,invocationCount:l*u,declarations:T({width:l,height:u,traversal:s.traversal??`march`,layout:c,clearance:!1})+p,body:`let column = index % WIDTH;
  let row = index / WIDTH;
  let observer = vec2<f32>(settings[settingsOffset], settings[settingsOffset + 1u]);
  let observerHeight = settings[settingsOffset + 2u];
  let targetHeight = settings[settingsOffset + 3u];
  let maxDistance = settings[settingsOffset + 4u];
  let cellSize = ${f===`uniform`?`vec2<f32>(settings[settingsOffset + 5u], settings[settingsOffset + 6u])`:`getRayGroundCell(vec2<f32>(settings[settingsOffset + 5u], settings[settingsOffset + 6u]), settings[settingsOffset + 8u], settings[settingsOffset + 9u], 0.5 * (observer.y + f32(row)) + 0.5)`};
  let curvature = settings[settingsOffset + 7u];
  ${s.toleranceParameters?`let toleranceValues = vec4<f32>(
    toleranceBand[toleranceBandOffset], toleranceBand[toleranceBandOffset + 1u],
    toleranceBand[toleranceBandOffset + 2u], toleranceBand[toleranceBandOffset + 3u]);`:`let toleranceValues = vec4<f32>(0.0);`}
  let observerInside = isFiniteValue(observer.x) && isFiniteValue(observer.y) &&
    observer.x >= 0.0 && observer.y >= 0.0 &&
    observer.x <= f32(WIDTH - 1u) && observer.y <= f32(HEIGHT - 1u);
${s.heightReference===`absolute`?``:`  let observerSample = sampleElevation(select(vec2<f32>(0.0), observer, observerInside));
`}  if (!observerInside || ${s.heightReference===`absolute`?``:`observerSample.y == 0.0 || `}!isValidPixel(column, row)) {
    visibilityValues[visibilityValuesOffset + index] = NO_DATA;
    return;
  }
  let result = traceSightLine(
    observer, vec2<f32>(f32(column), f32(row)), ${s.heightReference===`absolute`?`observerHeight`:`observerSample.x + observerHeight`},
    getElevation(column, row) + targetHeight, cellSize, curvature, maxDistance, toleranceValues);
  visibilityValues[visibilityValuesOffset + index] = result.code;`})),m}};export{D as n,O as r,k as t};
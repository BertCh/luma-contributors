import{a as e,c as t,i as n,o as r,r as i,s as a,u as o}from"./gpu-point-horizon-profile-D7TGBXeA.js";import{n as s,r as c}from"./terrain-binding-preflight-laSEObQE.js";import{o as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as u,x as d}from"./gpu-contributor-utils-BIwLQON5.js";import{n as f}from"./gpu-terrain-viewshed-BkBuCpsE.js";import{c as p,h as m,l as h,n as g}from"./raster-band-utils-BcTxc8_Y.js";import{t as _}from"./define-run-B-brkCe4.js";import{r as v}from"./text-DYSgVV-n.js";import{n as y,r as b}from"./pointer-BY8DXNWA.js";import{a as x,c as S,i as C,m as w,s as T,t as E,u as D}from"./terrain-Cc9dW1uq.js";import{z as O}from"./entry-preview-D70TdSbK.js";var k=class{id;props;model;constructor(e){this.id=e.id??`point-horizon-visibility`,this.props=e;let{id:n}=this,{output:i}=e;if(this.model=r(n,e.grid,{...e,azimuthCount:720,firstAzimuth:0,azimuthSpan:720}),t(n,e,e.elevation,this.model.heightReference),m(n,e.parameters,12),a(n,`observers`,e.observers,`float32x4`),a(n,`targets`,e.targets,`float32x4`),e.observers.length<1)throw Error(`${n} requires at least one observer`);if(u(i.visibility,[`uint32`],`${n} output.visibility`),i.visibility.length!==e.targets.length)throw Error(`${n} output.visibility must contain one value per target`);if(i.details&&(a(n,`output.details`,i.details,`float32x4`),i.details.length!==e.targets.length))throw Error(`${n} output.details must contain one row per target`);let o=e.targetsPerDispatch??65536;if(!Number.isInteger(o)||o<1)throw Error(`${n} targetsPerDispatch must be a positive integer`);h(n,[i.visibility,i.details],[...g(e.elevation),e.parameters,e.observers,e.targets])}getCommandNodes(t){c(t,this.id,this.props,{"canonical elevation":4*this.props.grid.width*this.props.grid.height,"pyramid fused parameters":s(this.props.pyramid?void 0:this.model.layout),pyramid:this.model.layout&&!this.props.pyramid?8*this.model.layout.length:0,"chunk table":256*Math.max(1,Math.ceil(this.props.targets.length/(this.props.targetsPerDispatch??65536)))});let{id:r,props:a,model:o}=this,{output:u}=a;p(r,t,a.elevation,[]),d(r,t,[a.pyramid?.combined,a.parameters,a.observers,a.targets,u.visibility,u.details]);let m=n(t,r,o,a.elevation),h=[...m.nodes],g=[...m.bindings,{name:`settings`,view:a.parameters,type:`f32`,access:`read`},{name:`observers`,view:a.observers,type:`f32`,access:`read`},{name:`targets`,view:a.targets,type:`f32`,access:`read`},{name:`visibilityOutput`,view:u.visibility,type:`u32`,access:`read_write`}];if(u.details&&g.push({name:`detailsOutput`,view:u.details,type:`f32`,access:`read_write`}),g.length>7)throw Error(`${r} needs more than 8 storage bindings`);let _=o.projection===`web-mercator`,v=_?`let worldPixelSize = settings[settingsOffset + 1u];
  let halfLambda = PI * deltaColumn / worldPixelSize;
  let halfM = -PI * deltaRow / worldPixelSize;
  let m2 = eye.m + 2.0 * halfM;
  let meanM = eye.m + halfM;
  let cosP2 = 1.0 / coshS(m2);
  let sinHalfLambda = sinAny(halfLambda);
  let cosHalfLambda = cosAny(halfLambda);
  let sinhHalfM = sinhS(halfM);
  let haversine = eye.cosP * cosP2 * (sinhHalfM * sinhHalfM + sinHalfLambda * sinHalfLambda);
  let sphereDistance = 2.0 * EARTH_RADIUS * asinS(sqrt(min(haversine, 1.0)));
  let sinDeltaPhi = 2.0 * coshS(meanM) * sinhHalfM * eye.cosP * cosP2;
  let north = sinDeltaPhi + eye.sinP * cosP2 * 2.0 * sinHalfLambda * sinHalfLambda;
  let east = 2.0 * sinHalfLambda * cosHalfLambda * cosP2;
  let rho = sqrt(north * north + east * east);
  let sinA = east / max(rho, 1.0e-30);
  let cosA = north / max(rho, 1.0e-30);
  // Ground meters per sphere meter along this ray: the WGS84 metric at the eye, the row scale
  // of the raster-grid contributors (see POINT_HORIZON_GROUND_SCALE_WGSL).
  let groundScale = getRayGroundScale(eye, sinA, cosA);
  let distance = sphereDistance * groundScale;
  // One Web Mercator pixel along the ray at the target's latitude, in ground meters.
  let targetPixelLength = 2.0 * PI * EARTH_RADIUS * cosP2 / worldPixelSize * groundScale;`:`let deltaX = deltaColumn * settings[settingsOffset + 1u];
  let deltaY = deltaRow * settings[settingsOffset + 2u];
  let distance = sqrt(deltaX * deltaX + deltaY * deltaY);
  let sinA = deltaX / max(distance, 1.0e-30);
  let cosA = ${o.rowDirection===`south`?`-`:``}deltaY / max(distance, 1.0e-30);
  // Ground length of one pixel step along the ray (the march's pixels-per-meter is (sinA / cellX, cosA / cellY)).
  let targetPixelLength = 1.0 / length(vec2<f32>(sinA / settings[settingsOffset + 1u], cosA / settings[settingsOffset + 2u]));`,y=e(o,11),{targets:b}=a,x=i(t,{id:r,operation:`GPUPointHorizonVisibility`,itemCount:b.length,maximumChunkSize:a.targetsPerDispatch??65536});h.push(x.writeNode);let S=u.details?`let detailsBase = detailsOutputOffset + (chunkStart[chunkStartOffset] + index) * 4u;
  detailsOutput[detailsBase] = result.details.x;
  detailsOutput[detailsBase + 1u] = result.details.y;
  detailsOutput[detailsBase + 2u] = result.details.z;
  detailsOutput[detailsBase + 3u] = result.details.w;`:``;for(let e=0;e<x.chunkCount;e++)h.push(l(t,{id:`${r}-classify-${e}`,operation:`GPUPointHorizonVisibility`,variant:o.traversal,bindings:[...g,x.getChunkStart(e)],invocationCount:x.chunkSize,workgroupSize:64,declarations:`const TARGET_COUNT: u32 = ${b.length}u;
const OBSERVER_COUNT: u32 = ${a.observers.length}u;
const HIDDEN: u32 = ${f.hidden}u;
const VISIBLE: u32 = ${f.visible}u;
const OUT_OF_RANGE: u32 = ${f.outOfRange}u;
const NO_DATA: u32 = ${f.noData}u;
const MARGINAL: u32 = ${f.marginal}u;
${y}
fn sinAny(x: f32) -> f32 { if (abs(x) > 1.6) { return sin(x); } return sinSeries(x); }
fn cosAny(x: f32) -> f32 { if (abs(x) > 1.6) { return cos(x); } return cosSeries(x); }

struct Classification { code: u32, details: vec4<f32> };

fn classifyTarget(index: u32) -> Classification {
  let notANumber = vec4<f32>(nanValue());
  let base = targetsOffset + (chunkStart[chunkStartOffset] + index) * 4u;
  let targetColumn = targets[base];
  let targetRow = targets[base + 1u];
  let targetHeight = targets[base + 2u];
  let observerValue = targets[base + 3u];
  if (!(observerValue >= 0.0 && observerValue < f32(OBSERVER_COUNT))) {
    return Classification(NO_DATA, notANumber);
  }
  let observerBase = observersOffset + u32(observerValue) * 4u;
  let observerColumn = observers[observerBase];
  let observerRow = observers[observerBase + 1u];
  let eye = makeEye(observerColumn, observerRow, observers[observerBase + 2u]);
  if (!eye.valid || !isInsideGrid(targetColumn, targetRow) || !isFiniteValue(targetHeight)) {
    return Classification(NO_DATA, notANumber);
  }
  let targetSample = sampleRelative(targetColumn, targetRow, eye.hi, eye.lo);
  if (targetSample.y == 0.0) {
    return Classification(NO_DATA, notANumber);
  }
  let relativeHeight = targetSample.x + targetHeight;
  let deltaColumn = targetColumn - observerColumn;
  let deltaRow = targetRow - observerRow;
  ${v}
  let maximumFrame = settings[settingsOffset + 3u];
  let maximumDistance = select(MAXIMUM_DISTANCE, min(maximumFrame, MAXIMUM_DISTANCE), maximumFrame > 0.0);
  if (distance > maximumDistance${_?` || sphereDistance > MAXIMUM_DISTANCE`:``}) {
    return Classification(OUT_OF_RANGE, notANumber);
  }
  if (!(distance > 0.0)) {
    return Classification(VISIBLE, notANumber);
  }
  let curvature = settings[settingsOffset];
  // A negative packed ignore distance is the derived default: one ground pixel at the target.
  let packedIgnoreDistance = settings[settingsOffset + 8u];
  let ignoreDistance = select(packedIgnoreDistance, targetPixelLength, packedIgnoreDistance < 0.0);
  let stop = distance - max(ignoreDistance, settings[settingsOffset + 9u] * distance);
  let ray = marchRay(eye, sinA, cosA, stop, stop > 0.0);
  let targetTangent = relativeHeight / distance - opaque(distance * curvature);
  let targetAngle = atanAccurate(targetTangent) * DEGREES;
  let occluderAngle = select(-90.0, atanAccurate(ray.tQ) * DEGREES, ray.tQ > -3.0e38);
  let skylineAngle = select(-90.0, atanAccurate(ray.t) * DEGREES, ray.has);
  let tolerance = settings[settingsOffset + 5u] + (settings[settingsOffset + 6u] / distance) * DEGREES;
  var code = VISIBLE;
  if (targetAngle < occluderAngle - tolerance) {
    code = HIDDEN;
  } else if (targetAngle - occluderAngle < tolerance) {
    // Not hidden, so the target is at or above the lower edge of the band: the band is
    // [occluder - tolerance, occluder + tolerance), with no gap at its lower edge.
    code = MARGINAL;
  }
  let onSkyline = code != HIDDEN && targetAngle >= skylineAngle - settings[settingsOffset + 7u];
  return Classification(code, vec4<f32>(targetAngle, occluderAngle, skylineAngle, select(0.0, 1.0, onSkyline)));
}`,body:`if (chunkStart[chunkStartOffset] + index >= TARGET_COUNT) { return; }
  let result = classifyTarget(index);
  visibilityOutput[visibilityOutputOffset + chunkStart[chunkStartOffset] + index] = result.code;
  ${S}`}));return h}},A=14,j=2e4,M=_({parameters:O,caption:`Elevation, an observer and summit targets in; visible or hidden per target out.`,createInputs:({world:e})=>{let t=C(e),n=T(t,A,7).map(e=>[e%t.columns,Math.floor(e/t.columns)]);return{...t,peaks:n}},build(e,t){let{grid:n,band:r,elevation:i}=w(e,t),a=t.peaks.length,s=e.allocate(`observers`,`float32x4`,1),c=e.allocate(`targets`,`float32x4`,a),l=e.allocate(`parameters`,`float32`,12),u=e.allocate(`visibility`,`uint32`,a),d=e.allocate(`details`,`float32x4`,a),f=e.upload(`peak-positions`,`float32x2`,Float32Array.from(t.peaks.flatMap(([e,n])=>D(t,t.world,e,n)))),p=e.allocate(`observer-dot`,`float32x2`,1);return e.graph.add(new k({grid:n,elevation:r,observers:s.view,targets:c.view,cellSize:t.cellSize,maximumDistance:j,parameters:l.view,output:{visibility:u.view,details:d.view}})),{outputs:{elevation:i,visibility:u,peakPositions:f,observerDot:p},update:e=>{let[r,i]=S(t,e.observerColumn,e.observerRow),a=D(t,t.world,r,i);s.write(Float32Array.of(r,i,e.observerHeight,0)),c.write(Float32Array.from(t.peaks.flatMap(([t,n])=>[t,n,e.targetHeight,0]))),p.write(Float32Array.from(a)),l.write(o({maximumDistance:j},n))}}},interaction:{kind:`place`,hint:`move the observer`,rest:y(O),resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=b(t,i,e),s=a(n.visibility),c=[{shape:`crosshair`,at:o.position}],l=0;return t.peaks.forEach(([e,n],r)=>{s&&s[r]===1&&(l++,c.push({shape:`line`,from:o.position,to:D(t,i,e,n)}))}),{values:{observerColumn:o.observerColumn,observerRow:o.observerRow},glyphs:c,readout:s?`observer [${o.column}, ${o.row}] · ${l} of ${t.peaks.length} summits visible`:`observer [${o.column}, ${o.row}] · ${r.observerHeight} m up`}}},draw:({inputs:e,outputs:t})=>[x(e,t.elevation),{layer:`dots`,positions:t.peakPositions,values:t.visibility,ramp:`categorical`,categories:E,radius:13},{layer:`dots`,positions:t.observerDot,radius:7,color:[1,1,1,1]},{layer:`text`,...v({corner:`bottom-left`,entries:[{label:`visible`,swatch:[.2,.85,.4]},{label:`hidden by terrain`,swatch:[.27,.1,.4]}]})}]});export{M as default};
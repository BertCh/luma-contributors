import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-raster-buffer-to-texture-BDZQU4A5.js";import{r}from"./gpu-terrain-derivatives-DmS0Xo0r.js";import{c as i,o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as o,x as s}from"./gpu-contributor-utils-BIwLQON5.js";import{s as c}from"./wgsl-common-DfyXiYB0.js";import{c as l,d as u,h as d,l as f,n as p,o as m,p as h,r as g,u as _}from"./raster-band-utils-BcTxc8_Y.js";import{n as v,u as y}from"./terrain-illumination-utils-CoEB4UZd.js";import{a as b,c as x,f as S,u as C}from"./raster-grid-VB0AM_GY.js";import{a as w,i as T,r as E}from"./terrain-geomorphometry-utils-BhBw8dIY.js";import{t as D}from"./define-run-B-brkCe4.js";import{i as O}from"./pointer-overlay-Qh8qflcA.js";import{t as k}from"./horizon-cisgaCRR.js";import{i as A,m as j}from"./terrain-Cc9dW1uq.js";import{t as M}from"./relief-CcqfaXl5.js";import{I as N}from"./entry-preview-D70TdSbK.js";var P=.72,F=65,I={cellSizeIndex:0,northEdgeIndex:3},L=20,R=44,z=76,B=[{azimuthDegrees:225,altitudeDegrees:30},{azimuthDegrees:270,altitudeDegrees:30},{azimuthDegrees:315,altitudeDegrees:30},{azimuthDegrees:360,altitudeDegrees:30}];function V(e,t,n=new Float32Array(80)){m(`getGPUReliefShadingParameterValues`,n,80);let r=e.lights??`mdow`,i=r===`mdow`?B:r;if(i.length<1||i.length>8)throw Error(`Relief shading needs 1 to 8 lights`);let a=e.elevationStops??[];if(a.length>8)throw Error(`Relief shading supports at most 8 elevation stops`);for(let e=1;e<a.length;e++)if(!(a[e].elevation>a[e-1].elevation))throw Error(`Relief shading elevation stops must be strictly ascending`);let o=e.lightWeighting??(r===`mdow`?`aspect`:`fixed`),s=e.warmColor??[1,.95,.85],c=e.coolColor??[.8,.85,1];n.fill(0,0,80),n.set([0,0,e.zFactor??1,0,0,i.length,o===`imhof-swing`?2:+(o===`aspect`),e.hillshadeStrength??1,e.skyViewStrength??1,e.textureShadeStrength??.5,e.exposure??1,e.tintStrength??.3,s[0],s[1],s[2],a.length,c[0],c[1],c[2],e.imhofSwingDegrees??(o===`imhof-swing`?F:0)]);for(let[e,t]of i.entries())n.set([t.azimuthDegrees,t.altitudeDegrees??45,t.weight??1],L+e*3);for(let[e,t]of a.entries())n.set([t.elevation,...t.color],R+e*4);return n.set([e.curvatureStrength??1,e.contrastLowElevation??0,e.contrastHighElevation??0,e.contrastStrength??0],z),S(n,t,I),n}var H=class{id;props;requiredHalo;rowDirection;constructor(e){this.id=e.id??`relief-shading`,this.props=e,this.requiredHalo=e.borderMode===`one-sided`?2:1;let{id:t}=this,{output:n}=e,r=_(t,e.grid);if(!n.hillshade&&!n.relief&&!n.color&&!n.outputValidity&&!n.hillshadeTexture&&!n.reliefTexture)throw Error(`${t} requires at least one output`);for(let[i,a]of[[`skyViewFactor`,e.skyViewFactor],[`textureShade`,e.textureShade],[`curvature`,e.curvature],[`output.hillshade`,n.hillshade],[`output.relief`,n.relief]])y(t,i,a,`float32`,r);y(t,`output.color`,n.color,`uint32`,r),y(t,`output.outputValidity`,n.outputValidity,`uint32`,r),d(t,e.parameters,80);for(let[r,i]of[[`output.hillshadeTexture`,n.hillshadeTexture],[`output.reliefTexture`,n.reliefTexture]])h(t,r,i,[`r32float`,`rgba32float`],e.grid);C(t,e.cellSizeMode??`uniform`),u(t,e.grid,e.cellSizeMode),this.rowDirection=w(t,e.grid,e.rowDirection),f(t,[n.hillshade,n.relief,n.color,n.outputValidity],[...p(e.elevation),e.parameters,e.skyViewFactor,e.textureShade,e.curvature])}getCommandNodes(i){let c=!!(this.props.output.relief||this.props.output.color||this.props.output.outputValidity||this.props.output.reliefTexture);e(i,this.id,{...this.props,skyViewFactor:c?this.props.skyViewFactor:void 0,textureShade:c?this.props.textureShade:void 0,curvature:c?this.props.curvature:void 0},{"gradient scratch":4*(this.props.grid.width+(this.props.borderMode===`one-sided`?2:0))*(this.props.grid.height+(this.props.borderMode===`one-sided`?2:0))});let{id:u,props:d}=this,{output:f}=d,{width:p,height:m}=d.grid;l(u,i,d.elevation,[f.hillshadeTexture,f.reliefTexture]),s(u,i,[d.parameters,d.skyViewFactor,d.textureShade,d.curvature,f.hillshade,f.relief,f.color,f.outputValidity]);let h=p*m,_=d.borderMode??`clamp`,y=_===`one-sided`,S=!!(f.relief||f.color||f.outputValidity||f.reliefTexture),C=g(i,u,d.elevation,d.grid,S||y),w=[...C.nodes],D=y?E(i,{id:u,operation:`GPUReliefShading`,width:p,height:m,band:C.band}):void 0;w.push(...D?.nodes??[]);let O=D?.width??p,k=D?.height??m,A=T(y),j=[`x`,`y`].map(e=>{let n=t(i,`${u}-gradient-${e}-values`,`float32`,O*k),a=t(i,`${u}-gradient-${e}-validity`,`uint32`,O*k);return w.push(...o(i,()=>new r({id:`${u}-gradient-${e}`,width:O,height:k,input:D?.band??C.band,output:n,outputValidity:a,operator:`sobel`,direction:e,borderMode:_===`one-sided`?`clamp`:_,scale:1}).addToGraph(i))),{values:n,validity:a}}),M=f.hillshade??t(i,`${u}-hillshade`,`float32`,h),N=d.cellSizeMode??`uniform`;w.push(a(i,{id:`${u}-hillshade`,operation:`GPUReliefShading`,variant:`hillshade-${N}`,bindings:[{name:`gradientX`,view:j[0].values,type:`f32`,access:`read`},{name:`gradientY`,view:j[1].values,type:`f32`,access:`read`},{name:`gradientValidity`,view:j[0].validity,type:`u32`,access:`read`},{name:`settings`,view:d.parameters,type:`f32`,access:`read`},{name:`hillshadeValues`,view:M,type:`f32`,access:`read_write`}],invocationCount:h,declarations:`const WIDTH: u32 = ${p}u;
const HEIGHT: u32 = ${m}u;
${x(this.rowDirection)}
const LIGHT_OFFSET: u32 = ${L}u;
const MAX_LIGHT_COUNT: u32 = 8u;
${v}
${b(N,I)}`,body:`let row = index / WIDTH;
  ${A}
  let rawX = gradientX[gradientXOffset + gradientIndex];
  let rawY = gradientY[gradientYOffset + gradientIndex];
  let groundCell = getGroundCellSize(row);
  let zFactor = settings[settingsOffset + 2u];
  // Sobel responds 8 to a unit-per-pixel ramp, so Horn's derivative is raw / (8 * cell).
  let eastGradient = zFactor * rawX / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor * rawY / (8.0 * groundCell.y);
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let isFlat = eastGradient == 0.0 && northGradient == 0.0;
  // Downslope aspect clockwise from north.
  let aspect = atan2(-eastGradient, -northGradient);
  let lightCount = min(u32(max(settings[settingsOffset + 5u], 0.0)), MAX_LIGHT_COUNT);
  let aspectWeighting = settings[settingsOffset + 6u] != 0.0;
  var weightedSum = 0.0;
  var weightSum = 0.0;
  for (var light = 0u; light < lightCount; light++) {
    let slot = settingsOffset + LIGHT_OFFSET + light * 3u;
    let azimuth = settings[slot] * DEGREES_TO_RADIANS;
    let altitude = settings[slot + 1u] * DEGREES_TO_RADIANS;
    var weight = settings[slot + 2u];
    if (aspectWeighting) {
      let angle = sin(aspect - azimuth);
      weight = select(angle * angle, 1.0, isFlat);
    }
    let direction = vec3<f32>(sin(azimuth) * cos(altitude), cos(azimuth) * cos(altitude), sin(altitude));
    weightedSum += weight * max(dot(normal, direction), 0.0);
    weightSum += weight;
  }
  let shade = select(0.0, weightedSum / weightSum, weightSum > 0.0);
  let isValid = gradientValidity[gradientValidityOffset + gradientIndex] != 0u &&
    isFiniteValue(rawX) && isFiniteValue(rawY) && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(shade);
  hillshadeValues[hillshadeValuesOffset + index] = select(getQuietNaN(index), shade, isValid);`})),d.imhofSwing&&w.push(U(i,{id:`${u}-hillshade-swing`,width:p,height:m,cellSizeMode:N,rowDirection:this.rowDirection,gradientIndexWGSL:A,gradientX:j[0].values,gradientY:j[1].values,gradientValidity:j[0].validity,settings:d.parameters,hillshade:M}));let P=f.relief??(f.reliefTexture?t(i,`${u}-relief`,`float32`,h):void 0),F=d.textureShade,R=3+[d.skyViewFactor,d.textureShade,d.curvature,f.relief??f.reliefTexture,f.color,f.outputValidity].filter(Boolean).length,z=S&&!!d.curvature&&R>8;z&&(F=t(i,`${u}-detail`,`float32`,h),w.push(W(i,{id:`${u}-detail`,pixelCount:h,settings:d.parameters,textureShade:d.textureShade,curvature:d.curvature,detail:F}))),S&&w.push(G(i,{id:`${u}-compose`,pixelCount:h,elevation:C.band.storage.values,hillshade:M,settings:d.parameters,skyViewFactor:d.skyViewFactor,textureShade:F,textureShadeFolded:z,curvature:z?void 0:d.curvature,relief:P,color:f.color,validity:f.outputValidity}));for(let[e,t,r]of[[`hillshade`,M,f.hillshadeTexture],[`relief`,P,f.reliefTexture]])r&&t&&w.push(...o(i,()=>new n({id:`${u}-${e}-texture`,input:{id:`${u}-${e}-band`,format:`float32`,storage:{kind:`buffer`,values:t}},output:r,channel:0}).addToGraph(i)));return w}};function U(e,t){return a(e,{id:t.id,operation:`GPUReliefShading`,variant:`hillshade-swing-${t.cellSizeMode}`,bindings:[{name:`gradientX`,view:t.gradientX,type:`f32`,access:`read`},{name:`gradientY`,view:t.gradientY,type:`f32`,access:`read`},{name:`gradientValidity`,view:t.gradientValidity,type:`u32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`hillshadeValues`,view:t.hillshade,type:`f32`,access:`read_write`}],invocationCount:t.width*t.height,declarations:`const WIDTH: u32 = ${t.width}u;
const HEIGHT: u32 = ${t.height}u;
${x(t.rowDirection)}
const LIGHT_OFFSET: u32 = ${L}u;
const MAX_LIGHT_COUNT: u32 = 8u;
${v}
${c}
${b(t.cellSizeMode,I)}`,body:`if (settings[settingsOffset + 6u] != 2.0) {
    return;
  }
  let row = index / WIDTH;
  ${t.gradientIndexWGSL}
  let rawX = gradientX[gradientXOffset + gradientIndex];
  let rawY = gradientY[gradientYOffset + gradientIndex];
  let groundCell = getGroundCellSize(row);
  let zFactor = settings[settingsOffset + 2u];
  let eastGradient = zFactor * rawX / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor * rawY / (8.0 * groundCell.y);
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  // -northGradient is -0.0 on a pure east or west slope with north row order, and both
  // arguments are zero on a flat cell, where guardedAtan2 returns 0 and the zero swing factor
  // keeps every light at its base azimuth (FX-17, terrain-surface-05).
  let aspect = guardedAtan2(-eastGradient, -northGradient);
  let lightCount = min(u32(max(settings[settingsOffset + 5u], 0.0)), MAX_LIGHT_COUNT);
  let swingRadians = settings[settingsOffset + 19u] * DEGREES_TO_RADIANS;
  let swingFactor = smoothstep(0.05, 0.4, length(normal.xy));
  var weightedSum = 0.0;
  var weightSum = 0.0;
  for (var light = 0u; light < lightCount; light++) {
    let slot = settingsOffset + LIGHT_OFFSET + light * 3u;
    let baseAzimuth = settings[slot] * DEGREES_TO_RADIANS;
    let azimuth = baseAzimuth + swingRadians * sin(aspect - baseAzimuth) * swingFactor;
    let altitude = settings[slot + 1u] * DEGREES_TO_RADIANS;
    let weight = settings[slot + 2u];
    let direction = vec3<f32>(sin(azimuth) * cos(altitude), cos(azimuth) * cos(altitude), sin(altitude));
    weightedSum += weight * max(dot(normal, direction), 0.0);
    weightSum += weight;
  }
  let shade = select(0.0, weightedSum / weightSum, weightSum > 0.0);
  let isValid = gradientValidity[gradientValidityOffset + gradientIndex] != 0u &&
    isFiniteValue(rawX) && isFiniteValue(rawY) && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(shade);
  hillshadeValues[hillshadeValuesOffset + index] = select(getQuietNaN(index), shade, isValid);`})}function W(e,t){let n=[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`curvatureValues`,view:t.curvature,type:`f32`,access:`read`}];return t.textureShade&&n.push({name:`textureShade`,view:t.textureShade,type:`f32`,access:`read`}),n.push({name:`detailValues`,view:t.detail,type:`f32`,access:`read_write`}),a(e,{id:t.id,operation:`GPUReliefShading`,variant:`detail`,bindings:n,invocationCount:t.pixelCount,declarations:v,body:`var detail = settings[settingsOffset + ${z}u] * curvatureValues[curvatureValuesOffset + index];
  ${t.textureShade?`detail += settings[settingsOffset + 9u] * textureShade[textureShadeOffset + index];`:``}
  detailValues[detailValuesOffset + index] = select(getQuietNaN(index), detail, isFiniteValue(detail));`})}function G(e,t){let n=[{name:`elevationValues`,view:t.elevation,type:`f32`,access:`read`},{name:`hillshadeValues`,view:t.hillshade,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`}];return t.skyViewFactor&&n.push({name:`skyView`,view:t.skyViewFactor,type:`f32`,access:`read`}),t.textureShade&&n.push({name:`textureShade`,view:t.textureShade,type:`f32`,access:`read`}),t.curvature&&n.push({name:`curvatureValues`,view:t.curvature,type:`f32`,access:`read`}),t.relief&&n.push({name:`reliefValues`,view:t.relief,type:`f32`,access:`read_write`}),t.color&&n.push({name:`colorValues`,view:t.color,type:`u32`,access:`read_write`}),t.validity&&n.push({name:`validityValues`,view:t.validity,type:`u32`,access:`read_write`}),a(e,{id:t.id,operation:`GPUReliefShading`,variant:`compose`,bindings:n,invocationCount:t.pixelCount,declarations:`const STOP_OFFSET: u32 = ${R}u;
const CONTRAST_PIVOT: f32 = ${i(P)};
const MAX_STOP_COUNT: u32 = 8u;
${v}
fn getSettingsColor(slot: u32) -> vec3<f32> {
  return vec3<f32>(settings[settingsOffset + slot], settings[settingsOffset + slot + 1u],
    settings[settingsOffset + slot + 2u]);
}
fn getElevationColor(elevation: f32) -> vec3<f32> {
  let stopCount = min(u32(max(settings[settingsOffset + 15u], 0.0)), MAX_STOP_COUNT);
  if (stopCount == 0u) { return vec3<f32>(0.85); }
  var color = getSettingsColor(STOP_OFFSET + 1u);
  if (elevation <= settings[settingsOffset + STOP_OFFSET]) { return color; }
  for (var stop = 1u; stop < stopCount; stop++) {
    let slot = STOP_OFFSET + stop * 4u;
    let lowerElevation = settings[settingsOffset + slot - 4u];
    let upperElevation = settings[settingsOffset + slot];
    color = getSettingsColor(slot + 1u);
    if (elevation < upperElevation) {
      let blend = (elevation - lowerElevation) / (upperElevation - lowerElevation);
      return mix(getSettingsColor(slot - 3u), color, blend);
    }
  }
  return color;
}`,body:`let hillshade = hillshadeValues[hillshadeValuesOffset + index];
  var shade = mix(1.0, hillshade, settings[settingsOffset + 7u]);
  var isValid = isFiniteValue(hillshade);
  ${t.skyViewFactor?`let skyViewFactor = skyView[skyViewOffset + index];
  isValid = isValid && isFiniteValue(skyViewFactor);
  shade *= mix(1.0, skyViewFactor, settings[settingsOffset + 8u]);`:``}
  ${t.textureShade?`let textureValue = textureShade[textureShadeOffset + index];
  isValid = isValid && isFiniteValue(textureValue);
  shade += ${t.textureShadeFolded?``:`settings[settingsOffset + 9u] * `}textureValue;`:``}
  ${t.curvature?`let curvatureValue = curvatureValues[curvatureValuesOffset + index];
  isValid = isValid && isFiniteValue(curvatureValue);
  shade += settings[settingsOffset + ${z}u] * curvatureValue;`:``}
  let contrastStrength = settings[settingsOffset + 79u];
  if (contrastStrength != 0.0) {
    let low = settings[settingsOffset + 77u];
    let high = settings[settingsOffset + 78u];
    let elevation = elevationValues[elevationValuesOffset + index];
    let ramp = select(select(0.0, 1.0, elevation >= low), smoothstep(low, high, elevation), high > low);
    shade = mix(CONTRAST_PIVOT, shade, mix(1.0 - contrastStrength, 1.0, ramp));
    isValid = isValid && isFiniteValue(shade);
  }
  let relief = clamp(shade * settings[settingsOffset + 10u], 0.0, 1.0);
  ${t.relief?`reliefValues[reliefValuesOffset + index] = select(getQuietNaN(index), relief, isValid);`:``}
  ${t.color?`let tintStrength = settings[settingsOffset + 11u];
  let lightness = clamp(2.0 * hillshade - 1.0, -1.0, 1.0);
  var tint = mix(vec3<f32>(1.0), getSettingsColor(12u), lightness * tintStrength);
  if (lightness < 0.0) {
    tint = mix(vec3<f32>(1.0), getSettingsColor(16u), -lightness * tintStrength);
  }
  let base = getElevationColor(elevationValues[elevationValuesOffset + index]);
  let color = clamp(base * relief * tint, vec3<f32>(0.0), vec3<f32>(1.0));
  colorValues[colorValuesOffset + index] = select(0u, pack4x8unorm(vec4<f32>(color, 1.0)), isValid);`:``}
  ${t.validity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`})}var K=D({parameters:N,caption:`Elevation and a sky view factor in; a lit relief out, darkened in hollows by the sky view factor.`,createInputs:({world:e})=>A(e),build(e,t){let n=j(e,t),{grid:r,band:i,elevation:a,cellCount:o}=n,{skyViewFactor:s}=k(e,n,{directionCount:16,maximumRadius:30}),c=e.allocate(`parameters`,`float32`,80),l=e.allocate(`relief`,`float32`,o),u=e.allocate(`color`,`uint32`,o);return e.graph.add(new H({grid:r,elevation:i,skyViewFactor:s.view,parameters:c.view,output:{relief:l.view,color:u.view}})),{outputs:{elevation:a,relief:l},update:e=>c.write(V({lights:[{azimuthDegrees:e.azimuthDegrees,altitudeDegrees:e.altitudeDegrees}],skyViewStrength:e.skyViewStrength,exposure:e.exposure,zFactor:e.zFactor},r))}},interaction:O(N,{hint:`aim the light`,angle:`azimuthDegrees`,distance:{parameter:`altitudeDegrees`,invert:!0},readout:e=>`light ${e.azimuthDegrees}° · ${e.altitudeDegrees}° up`}),draw:({inputs:e,outputs:t})=>[M(e,t.relief)]});export{K as default};
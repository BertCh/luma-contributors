import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-raster-buffer-to-texture-BDZQU4A5.js";import{_ as r,c as i,d as a,f as o,h as s,l as c,n as l,o as u,p as d,r as f,t as p,u as m}from"./raster-band-utils-BcTxc8_Y.js";import{i as h}from"./raster-utils-CztuVGLY.js";import{o as g}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as _,i as v,x as y}from"./gpu-contributor-utils-BIwLQON5.js";import{s as b}from"./wgsl-common-DfyXiYB0.js";import{a as x,c as S,f as C,t as w,u as T}from"./raster-grid-VB0AM_GY.js";import{a as E,i as D,r as O}from"./terrain-geomorphometry-utils-BhBw8dIY.js";var k=[-1,0,1,-2,0,2,-1,0,1],A=[-1,-2,-1,0,0,0,1,2,1],j=[-3,0,3,-10,0,10,-3,0,3],M=[-3,-10,-3,0,0,0,3,10,3],N=class extends r{operator;direction;scale;constructor(e){let t=e.id??`gpu-raster-gradient`;if(e.operator!==`sobel`&&e.operator!==`scharr`)throw Error(`${t} gradient operator must be sobel or scharr`);if(e.direction!==`x`&&e.direction!==`y`)throw Error(`${t} gradient direction must be x or y`);let n=F(e.scale,t),r=P(e.operator,e.direction);super({...e,id:t,radius:1,kernel:r.map(e=>e*n),noDataPolicy:`propagate`,normalize:!1}),this.operator=e.operator,this.direction=e.direction,this.scale=n}};function P(e,t){return e===`sobel`?t===`x`?k:A:t===`x`?j:M}function F(e,t){let n=e??1;if(!Number.isFinite(n)||n<=0)throw Error(`${t} derivative scale must be finite and positive`);if(h(n),Math.fround(n)===0)throw Error(`${t} derivative scale must fit in positive float32`);return n}var I={cellSizeIndex:0,northEdgeIndex:5};function L(e,t,n=new Float32Array(8)){return o(`getGPUTerrainDerivativesParameterValues`,t),u(`getGPUTerrainDerivativesParameterValues`,n,8),n[2]=e.zFactor??1,n[3]=e.azimuthDegrees??315,n[4]=e.altitudeDegrees??45,n[7]=0,C(n,t,I),n}var R=class{id;props;requiredHalo;rowDirection;constructor(e){this.id=e.id??`terrain-derivatives`,this.props=e,this.requiredHalo=(e.borderMode??`one-sided`)===`one-sided`?2:1;let{id:t}=this,{output:n}=e,r=m(t,e.grid);if(!n.slope&&!n.aspect&&!n.hillshade&&!n.outputValidity&&!n.hillshadeTexture)throw Error(`${t} requires at least one output`);for(let[e,i,a]of[[`slope`,n.slope,`float32`],[`aspect`,n.aspect,`float32`],[`hillshade`,n.hillshade,`float32`],[`outputValidity`,n.outputValidity,`uint32`]])if(i&&(_(i,[a],`${t} output.${e}`),i.length!==r))throw Error(`${t} output.${e} must contain one value per pixel`);if(s(t,e.parameters,8),d(t,`output.hillshadeTexture`,n.hillshadeTexture,[`r32float`,`rgba32float`],e.grid),T(t,e.cellSizeMode),a(t,e.grid,e.cellSizeMode),o(t,e.grid),![`degrees`,`percent`].includes(e.slopeUnits??`degrees`))throw Error(`${t} slopeUnits must be degrees or percent`);if(this.rowDirection=E(t,e.grid,e.rowDirection),![`one-sided`,`clamp`,`reflect`,`constant`,`nodata`].includes(e.borderMode??`one-sided`))throw Error(`${t} borderMode must be one-sided, clamp, reflect, constant, or nodata`);c(t,[n.slope,n.aspect,n.hillshade,n.outputValidity],[...l(e.elevation),e.parameters])}getCommandNodes(r){e(r,this.id,this.props,{"gradient scratch":4*(this.props.grid.width+((this.props.borderMode??`one-sided`)===`one-sided`?2:0))*(this.props.grid.height+((this.props.borderMode??`one-sided`)===`one-sided`?2:0))});let{id:a,props:o}=this,{output:s}=o,{width:c,height:l}=o.grid;i(a,r,o.elevation,[s.hillshadeTexture]),y(a,r,[o.parameters,s.slope,s.aspect,s.hillshade,s.outputValidity]);let u=c*l,d=o.borderMode??`one-sided`,p=d===`one-sided`,m=f(r,a,o.elevation,o.grid,p?`reuse-canonical`:!1),h=[...m.nodes],g=p?O(r,{id:a,operation:`GPUTerrainDerivatives`,width:c,height:l,band:m.band}):void 0;h.push(...g?.nodes??[]);let _=g?.width??c,b=g?.height??l,x=[`x`,`y`].map(e=>{let n=t(r,`${a}-gradient-${e}-values`,`float32`,_*b),i=t(r,`${a}-gradient-${e}-validity`,`uint32`,_*b);return h.push(...v(r,()=>new N({id:`${a}-gradient-${e}`,width:_,height:b,input:g?.band??m.band,output:n,outputValidity:i,operator:`sobel`,direction:e,borderMode:d===`one-sided`?`clamp`:d,scale:1}).addToGraph(r))),{values:n,validity:i}}),S=s.hillshade??(s.hillshadeTexture?t(r,`${a}-hillshade`,`float32`,u):void 0);if(h.push(z(r,{id:`${a}-shade`,width:c,height:l,gradientX:x[0].values,gradientY:x[1].values,gradientValidity:x[0].validity,settings:o.parameters,slope:s.slope,aspect:s.aspect,hillshade:S,validity:s.outputValidity,cellSizeMode:o.cellSizeMode??`uniform`,slopeUnits:o.slopeUnits??`degrees`,rowDirection:this.rowDirection,isPadded:p})),s.hillshadeTexture&&S){let e=s.hillshadeTexture;h.push(...v(r,()=>new n({id:`${a}-hillshade-texture`,input:{id:`${a}-hillshade-band`,format:`float32`,storage:{kind:`buffer`,values:S}},output:e,channel:0}).addToGraph(r)))}return h}};function z(e,t){let n=[{name:`gradientX`,view:t.gradientX,type:`f32`,access:`read`},{name:`gradientY`,view:t.gradientY,type:`f32`,access:`read`},{name:`gradientValidity`,view:t.gradientValidity,type:`u32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`}];return t.slope&&n.push({name:`slopeValues`,view:t.slope,type:`f32`,access:`read_write`}),t.aspect&&n.push({name:`aspectValues`,view:t.aspect,type:`f32`,access:`read_write`}),t.hillshade&&n.push({name:`hillshadeValues`,view:t.hillshade,type:`f32`,access:`read_write`}),t.validity&&n.push({name:`validityValues`,view:t.validity,type:`u32`,access:`read_write`}),g(e,{id:t.id,operation:`GPUTerrainDerivatives`,variant:`shade-${t.cellSizeMode}${t.isPadded?`-padded`:``}`,bindings:n,invocationCount:t.width*t.height,declarations:`const WIDTH: u32 = ${t.width}u;
const HEIGHT: u32 = ${t.height}u;
const PI: f32 = 3.141592653589793;
${w}
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
${S(t.rowDirection)}
${p}
${t.aspect?b:``}
${x(t.cellSizeMode,I)}`,body:`let row = index / WIDTH;
  ${D(t.isPadded)}
  let rawX = gradientX[gradientXOffset + gradientIndex];
  let rawY = gradientY[gradientYOffset + gradientIndex];
  let groundCell = getGroundCellSize(row);
  let zFactor = settings[settingsOffset + 2u];
  // Sobel responds 8 to a unit-per-pixel ramp, so Horn's derivative is raw / (8 * cell).
  let eastGradient = zFactor * rawX / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor * rawY / (8.0 * groundCell.y);
  let larger = max(abs(eastGradient), abs(northGradient));
  let smaller = min(abs(eastGradient), abs(northGradient));
  var gradientMagnitude = 0.0;
  if (larger > 0.0) {
    let ratio = smaller / larger;
    gradientMagnitude = larger * sqrt(1.0 + ratio * ratio);
  }
  let isValid = gradientValidity[gradientValidityOffset + gradientIndex] != 0u &&
    isFiniteValue(rawX) && isFiniteValue(rawY) && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(eastGradient) && isFiniteValue(northGradient) && isFiniteValue(gradientMagnitude);
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${t.slope?`let slope = ${t.slopeUnits===`degrees`?`atan(gradientMagnitude) * RADIANS_TO_DEGREES`:`gradientMagnitude * 100.0`};
  slopeValues[slopeValuesOffset + index] = select(invalidValue, slope, isValid);`:``}
  ${t.aspect?`var aspect = -1.0;
  if (gradientMagnitude > 0.0) {
    // -northGradient is -0.0 on a pure east or west slope with north row order (FX-17).
    aspect = guardedAtan2(-eastGradient, -northGradient) * RADIANS_TO_DEGREES;
    if (aspect < 0.0) { aspect += 360.0; }
    if (aspect >= 360.0) { aspect -= 360.0; }
  }
  aspectValues[aspectValuesOffset + index] = select(invalidValue, aspect, isValid);`:``}
  ${t.hillshade?`let azimuth = settings[settingsOffset + 3u] * DEGREES_TO_RADIANS;
  let altitude = settings[settingsOffset + 4u] * DEGREES_TO_RADIANS;
  let light = vec3<f32>(sin(azimuth) * cos(altitude), cos(azimuth) * cos(altitude), sin(altitude));
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let shade = max(dot(normal, light), 0.0);
  hillshadeValues[hillshadeValuesOffset + index] = select(invalidValue, shade, isValid);`:``}
  ${t.validity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`})}export{L as n,N as r,R as t};
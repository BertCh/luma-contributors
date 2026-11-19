import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{c as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,x as i}from"./gpu-contributor-utils-BIwLQON5.js";import{c as a,d as o,h as s,l as c,n as l,o as u,r as d,t as f,u as p}from"./raster-band-utils-BcTxc8_Y.js";import{a as m,n as h,o as g,t as _}from"./terrain-geomorphometry-utils-BhBw8dIY.js";import{a as v,u as y}from"./raster-grid-VB0AM_GY.js";import{t as b}from"./define-run-B-brkCe4.js";import{a as x,d as S,i as C,l as w,m as T,o as E,u as D}from"./terrain-Cc9dW1uq.js";import{E as O}from"./entry-preview-D70TdSbK.js";var k=4,A=1e-6,j=[.55,.45,0,0],M=5,N=[`profile`,`plan`,`tangential`,`mean`,`gaussian`,`minimal`,`maximal`,`unsphericity`,`difference`,`horizontal-excess`,`vertical-excess`,`accumulation`,`ring`,`rotor`,`laplacian`];function P(e,t,n=new Float32Array(12)){u(`getGPUTerrainCurvatureParameterValues`,n,12);let r=e.ringGains??j;if(r.length>k)throw Error(`Terrain curvature ringGains must hold at most 4 values`);n.fill(0,0,12),g(e,t,n),n[5]=e.flatGradient??A;for(let[e,t]of r.entries())n[6+e]=t;return n}var F=class{id;props;requiredHalo;rowDirection;constructor(e){this.id=e.id??`terrain-curvature`,this.props=e;let{id:t}=this,n=p(t,e.grid),i=e.method??`evans-young`;if(![`evans-young`,`zevenbergen-thorne`,`florinsky`].includes(i))throw Error(`${t} method must be evans-young, zevenbergen-thorne, or florinsky`);let{output:a}=e,u=Object.keys(a.curvatures??{});for(let e of u)if(!N.includes(e))throw Error(`${t} unknown curvature kind ${e}`);let d=u.map(e=>a.curvatures?.[e]).filter(e=>e!==void 0);if(d.length===0&&!a.ringCurvature&&!a.outputValidity)throw Error(`${t} requires at least one output`);for(let[e,i]of[...u.map(e=>[e,a.curvatures?.[e]]),[`ringCurvature`,a.ringCurvature]])if(i&&(r(i,[`float32`],`${t} output.${e}`),i.length!==n))throw Error(`${t} output.${e} must contain one value per pixel`);if(a.outputValidity&&(r(a.outputValidity,[`uint32`],`${t} output.outputValidity`),a.outputValidity.length!==n))throw Error(`${t} output.outputValidity must contain one value per pixel`);if(s(t,e.parameters,12),y(t,e.cellSizeMode),o(t,e.grid,e.cellSizeMode),this.rowDirection=m(t,e.grid,e.rowDirection),![`one-sided`,`clamp`,`nodata`].includes(e.borderMode??`one-sided`))throw Error(`${t} borderMode must be one-sided, clamp, or nodata`);if(![`none`,`pade-tanh`].includes(e.ringSquash??`none`))throw Error(`${t} ringSquash must be none or pade-tanh`);let f=e.ringRadii??[2,8];if(f.length<1||f.length>k||f.some((e,t)=>!Number.isInteger(e)||e<1||t>0&&e<=f[t-1]))throw Error(`${t} ringRadii must be 1 to 4 strictly ascending integers of at least 1`);c(t,[...d,a.ringCurvature,a.outputValidity],[...l(e.elevation),e.parameters]);let h=i===`florinsky`||(e.borderMode??`one-sided`)===`one-sided`?2:1;this.requiredHalo=a.ringCurvature?Math.max(h,f[f.length-1]):h}getCommandNodes(t){e(t,this.id,this.props,{"canonical elevation":4*this.props.grid.width*this.props.grid.height});let{id:n,props:r}=this,{output:o}=r;a(n,t,r.elevation,[]),i(n,t,[r.parameters,...N.map(e=>o.curvatures?.[e]),o.ringCurvature,o.outputValidity]);let s=d(t,n,r.elevation,r.grid,!0),c=[];for(let e of N){let t=o.curvatures?.[e];t&&c.push({kind:`curvature`,curvature:e,view:t})}o.ringCurvature&&c.push({kind:`ring`,view:o.ringCurvature}),o.outputValidity&&c.push({kind:`validity`,view:o.outputValidity});let l=c.some(e=>e.kind===`curvature`),u=!!o.ringCurvature,f=[...s.nodes];for(let e=0,i=0;e<c.length;e+=M,i++)f.push(z(t,{id:`${n}-curvature-${i}`,props:r,rowDirection:this.rowDirection,source:s,slots:c.slice(e,e+M),hasWindowOutput:l,hasRingOutput:u}));return f}},I={profile:`profile`,plan:`plan`,tangential:`tangential`,mean:`meanCurvature`,gaussian:`gaussian`,minimal:`minimal`,maximal:`maximal`,unsphericity:`unsphericity`,difference:`differenceCurvature`,"horizontal-excess":`horizontalExcess`,"vertical-excess":`verticalExcess`,accumulation:`accumulation`,ring:`ringProduct`,rotor:`rotor`,laplacian:`laplacian`};function L(e,t){let n=[],r=(e,r,i)=>{n.push(`let ${e} = readWindowDelta(column, row, ${r}, ${i*t}, centre);`)};if(e===`florinsky`){for(let e=0;e<25;e++)r(`z${e}`,e%5-2,2-Math.floor(e/5));return{samples:n.join(`
  `),partials:`let r = zFactor * (2.0 * (z0 + z4 + z5 + z9 + z10 + z14 + z15 + z19 + z20 + z24) - 2.0 * (z2 + z7 + z12 + z17 + z22) - z1 - z3 - z6 - z8 - z11 - z13 - z16 - z18 - z21 - z23) / (35.0 * wx * wx);
  let t = zFactor * (2.0 * (z0 + z1 + z2 + z3 + z4 + z20 + z21 + z22 + z23 + z24) - 2.0 * (z10 + z11 + z12 + z13 + z14) - z5 - z6 - z7 - z8 - z9 - z15 - z16 - z17 - z18 - z19) / (35.0 * wy * wy);
  let s = zFactor * (z8 + z16 - z6 - z18 + 4.0 * (z4 + z20 - z0 - z24) + 2.0 * (z3 + z9 + z15 + z21 - z1 - z5 - z19 - z23)) / (100.0 * wx * wy);
  let p = zFactor * (44.0 * (z3 + z23 - z1 - z21) + 31.0 * (z0 + z20 - z4 - z24 + 2.0 * (z8 + z18 - z6 - z16)) + 17.0 * (z14 - z10 + 4.0 * (z13 - z11)) + 5.0 * (z9 + z19 - z5 - z15)) / (420.0 * wx);
  let q = zFactor * (44.0 * (z5 + z9 - z15 - z19) + 31.0 * (z20 + z24 - z0 - z4 + 2.0 * (z6 + z8 - z16 - z18)) + 17.0 * (z2 - z22 + 4.0 * (z7 - z17)) + 5.0 * (z1 + z3 - z21 - z23)) / (420.0 * wy);`}}for(let[e,t,n]of[[1,-1,1],[2,0,1],[3,1,1],[4,-1,0],[6,1,0],[7,-1,-1],[8,0,-1],[9,1,-1]])r(`z${e}`,t,n);let i=`zFactor * (z3 + z7 - z1 - z9) / (4.0 * wx * wy)`;return e===`zevenbergen-thorne`?{samples:n.join(`
  `),partials:`let p = zFactor * (z6 - z4) / (2.0 * wx);
  let q = zFactor * (z2 - z8) / (2.0 * wy);
  let r = zFactor * (z4 + z6) / (wx * wx);
  let t = zFactor * (z2 + z8) / (wy * wy);
  let s = ${i};`}:{samples:n.join(`
  `),partials:`let p = zFactor * (z3 + z6 + z9 - z1 - z4 - z7) / (6.0 * wx);
  let q = zFactor * (z1 + z2 + z3 - z7 - z8 - z9) / (6.0 * wy);
  let r = zFactor * (z1 + z3 + z4 + z6 + z7 + z9 - 2.0 * (z2 + z8)) / (3.0 * wx * wx);
  let t = zFactor * (z1 + z2 + z3 + z7 + z8 + z9 - 2.0 * (z4 + z6)) / (3.0 * wy * wy);
  let s = ${i};`}}function R(e,n){let r=[],i=[];for(let[n,a]of e.entries()){let e=Math.max(1,Math.round(a/Math.SQRT2)),o=[[a,0],[-a,0],[0,a],[0,-a],[e,e],[e,-e],[-e,e],[-e,-e]].map(([e,t])=>`readRingDelta(column, row, ${e}, ${t}, centre)`).join(` + `);i.push(`let ringSum${n} = ${o};`),r.push(`settings[settingsOffset + ${6+n}u] * (-ringSum${n}) / (${t(8*a)} * cellMetres)`)}return`${i.join(`
  `)}
  var ringValue = zFactor * (${r.join(` + `)});
  ${n===`pade-tanh`?`let squashed = clamp(ringValue, -3.0, 3.0);
  ringValue = squashed * (27.0 + squashed * squashed) / (27.0 + 9.0 * squashed * squashed);`:``}`}function z(e,t){let{props:r,source:i,slots:a}=t,{width:o,height:s}=r.grid,c=r.method??`evans-young`,l=t.rowDirection===`south`?-1:1,u=r.borderMode??`one-sided`,d=a.some(e=>e.kind===`curvature`)||t.hasWindowOutput&&a.some(e=>e.kind===`validity`),p=a.some(e=>e.kind===`ring`)||t.hasRingOutput&&a.some(e=>e.kind===`validity`),m=i.band,g=[{name:`elevationValues`,view:m.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:m.validity,type:`u32`,access:`read`},{name:`settings`,view:r.parameters,type:`f32`,access:`read`}];for(let[e,t]of a.entries())g.push({name:`output${e}`,view:t.view,type:t.kind===`validity`?`u32`:`f32`,access:`read_write`});let y=L(c,l),b=r.ringRadii??[2,8],x=a.map((e,n)=>{let r=`output${n}`;return e.kind===`validity`?`${r}[${r}Offset + index] = select(0u, 1u, centreValid && ${[t.hasWindowOutput?`windowValid`:`true`,t.hasRingOutput?`ringValid`:`true`].join(` && `)});`:e.kind===`ring`?`${r}[${r}Offset + index] = select(invalidValue, ringValue, ringValid && centreValid && isFiniteValue(ringValue));`:`${r}[${r}Offset + index] = select(invalidValue, ${I[e.curvature]}, windowValid && centreValid);`}).join(`
  `),S=`let row = index / WIDTH;
  let column = index - row * WIDTH;
  let centre = elevationValues[elevationValuesOffset + index];
  let centreValid = elevationValidity[elevationValidityOffset + index] != 0u && isFiniteValue(centre);
  let groundCell = getGroundCellSize(row);
  let wx = groundCell.x;
  let wy = groundCell.y;
  let zFactor = settings[settingsOffset + 2u];
  let cellsValid = wx > 0.0 && wy > 0.0 && isFiniteValue(wx) && isFiniteValue(wy) && isFiniteValue(zFactor);
  let cellMetres = sqrt(wx * wy);
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  windowCentre = centre;
  windowValid = cellsValid;
  ringValid = cellsValid;
  ${d?`${y.samples}
  ${y.partials}
  let gradientSquared = p * p + q * q;
  let weight = 1.0 + gradientSquared;
  let weightRoot = sqrt(weight);
  let weightPower = weight * weightRoot;
  let flatLimit = settings[settingsOffset + 5u];
  let isFlat = gradientSquared <= flatLimit * flatLimit;
  let safeGradient = select(gradientSquared, 1.0, isFlat);
  let safeGradientPower = safeGradient * sqrt(safeGradient);
  let planNumerator = q * q * r - 2.0 * p * q * s + p * p * t;
  let profileNumerator = p * p * r + 2.0 * p * q * s + q * q * t;
  let tangential = select(-planNumerator / (safeGradient * weightRoot), 0.0, isFlat);
  let plan = select(-planNumerator / safeGradientPower, 0.0, isFlat);
  let profile = select(-profileNumerator / (safeGradient * weightPower), 0.0, isFlat);
  let meanCurvature = -((1.0 + q * q) * r - 2.0 * p * q * s + (1.0 + p * p) * t) / (2.0 * weightPower);
  let gaussian = (r * t - s * s) / (weight * weight);
  let unsphericity = sqrt(max(meanCurvature * meanCurvature - gaussian, 0.0));
  let minimal = meanCurvature - unsphericity;
  let maximal = meanCurvature + unsphericity;
  let differenceCurvature = (profile - tangential) * 0.5;
  let horizontalExcess = select(tangential - minimal, 0.0, isFlat);
  let verticalExcess = select(profile - minimal, 0.0, isFlat);
  let accumulation = tangential * profile;
  let ringProduct = horizontalExcess * verticalExcess;
  let rotor = select(((p * p - q * q) * s - p * q * (r - t)) / safeGradientPower, 0.0, isFlat);
  let laplacian = r + t;
  windowValid = windowValid && isFiniteValue(p) && isFiniteValue(q) && isFiniteValue(r) &&
    isFiniteValue(s) && isFiniteValue(t) && isFiniteValue(gradientSquared);`:``}
  ${p?R(b,r.ringSquash??`none`):``}
  ${x}`;return n(e,{id:t.id,operation:`GPUTerrainCurvature`,variant:`${c}-${r.cellSizeMode??`uniform`}-${u}`,bindings:g,invocationCount:o*s,declarations:`const WIDTH: u32 = ${o}u;
const HEIGHT: u32 = ${s}u;
const WIDTH_SIGNED: i32 = ${o};
const HEIGHT_SIGNED: i32 = ${s};
var<private> windowValid: bool;
var<private> ringValid: bool;
var<private> windowCentre: f32;
${f}
${v(r.cellSizeMode??`uniform`,_)}
${u===`one-sided`?`// In-grid elevation minus the window centre; .y is 1 when the sample is valid.
fn readGridSample(column: i32, row: i32) -> vec2<f32> {
  let sampleIndex = u32(row * WIDTH_SIGNED + column);
  let value = elevationValues[elevationValuesOffset + sampleIndex];
  let isValid = elevationValidity[elevationValidityOffset + sampleIndex] != 0u && isFiniteValue(value);
  return vec2<f32>(select(0.0, value - windowCentre, isValid), select(0.0, 1.0, isValid));
}
${h}`:``}
// Elevation of an offset sample minus the centre elevation; flags the window invalid on failure.
fn readWindowDelta(column: u32, row: u32, dx: i32, dy: i32, centre: f32) -> f32 {
  var sampleColumn = i32(column) + dx;
  var sampleRow = i32(row) + dy;
  ${u===`clamp`?`sampleColumn = clamp(sampleColumn, 0, WIDTH_SIGNED - 1);
  sampleRow = clamp(sampleRow, 0, HEIGHT_SIGNED - 1);`:`if (sampleColumn < 0 || sampleRow < 0 || sampleColumn >= WIDTH_SIGNED || sampleRow >= HEIGHT_SIGNED) {
    ${u===`one-sided`?`// One-sided border: extrapolate the centre-relative surface past the edge.
    let extended = readExtendedSample(sampleColumn, sampleRow);
    if (extended.y == 0.0) { windowValid = false; }
    return extended.x;`:`windowValid = false;
    return 0.0;`}
  }`}
  let sampleIndex = u32(sampleRow) * WIDTH + u32(sampleColumn);
  let value = elevationValues[elevationValuesOffset + sampleIndex];
  if (elevationValidity[elevationValidityOffset + sampleIndex] == 0u || !isFiniteValue(value)) {
    windowValid = false;
    return 0.0;
  }
  return value - centre;
}
// Ring samples never clamp: any sample outside the raster or invalid invalidates the ring.
fn readRingDelta(column: u32, row: u32, dx: i32, dy: i32, centre: f32) -> f32 {
  let sampleColumn = i32(column) + dx;
  let sampleRow = i32(row) + dy;
  if (sampleColumn < 0 || sampleRow < 0 || sampleColumn >= WIDTH_SIGNED || sampleRow >= HEIGHT_SIGNED) {
    ringValid = false;
    return 0.0;
  }
  let sampleIndex = u32(sampleRow) * WIDTH + u32(sampleColumn);
  let value = elevationValues[elevationValuesOffset + sampleIndex];
  if (elevationValidity[elevationValidityOffset + sampleIndex] == 0u || !isFiniteValue(value)) {
    ringValid = false;
    return 0.0;
  }
  return value - centre;
}`,body:S})}var B=[`evans-young`,`zevenbergen-thorne`,`florinsky`],V=[`profile`,`plan`,`mean`,`gaussian`],H=b({parameters:O,rebuildOn:[`method`,`curvature`],caption:`Elevation in; the surface curvature out: convex ridges against concave valleys.`,createInputs:({world:e})=>C(e),build(e,t,n){let{grid:r,band:i,elevation:a,cellCount:o}=T(e,t),s=e.allocate(`parameters`,`float32`,12),c=e.allocate(`curvature`,`float32`,o);return e.graph.add(new F({grid:r,elevation:i,parameters:s.view,method:B[n.method],output:{curvatures:{[V[n.curvature]]:c.view}}})),{outputs:{elevation:a,curvature:c},update:e=>s.write(P({zFactor:e.zFactor},r))}},interaction:{kind:`probe`,hint:`hover a cell`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let[o,s]=w(t,i,e.x,e.y),c=(e,n)=>t.elevation[Math.min(t.rows-1,Math.max(0,s+n))*t.columns+Math.min(t.columns-1,Math.max(0,o+e))],l=c(1,0)-c(-1,0),u=c(0,-1)-c(0,1),d=Math.hypot(l,u)||1,f=V[r.curvature]===`plan`,p=f?-u/d:l/d,m=f?l/d:u/d,h=D(t,i,o,s),g=2.5*S(t,i),_=a(n.curvature),v=_?_[s*t.columns+o]:NaN;return{glyphs:[{shape:`line`,from:[h[0]-p*g,h[1]-m*g],to:[h[0]+p*g,h[1]+m*g]},{shape:`dot`,at:h}],readout:Number.isFinite(v)?`${V[r.curvature]} ${v.toFixed(4)} /m · ${v>0?`convex`:`concave`}`:`${V[r.curvature]} curvature`}}},draw:({inputs:e,outputs:t})=>[x(e,t.elevation),E(e,t.curvature,{ramp:`diverging`,range:[-.003,.003],smoothing:!0})]});export{H as default};
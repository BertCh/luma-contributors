import{r as e}from"./kernel-C3tfQOn4.js";import{E as t,S as n,b as r,c as i,f as a,g as o,o as s,p as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d,c as f,o as p}from"./wgsl-kernel-nodes-DagJklH2.js";import{s as m,v as h}from"./spatial-context-DwZEd4EZ.js";import{t as g}from"./study-window-BK-B07m5.js";import{a as _,n as v,o as y}from"./study-window-wgsl-B3mOUanG.js";import{a as b,n as x,o as S,t as C}from"./fixed-point-accumulator-Ce6W3HVY.js";import{n as w,r as T,u as E}from"./wgsl-common-DfyXiYB0.js";import{u as D}from"./raster-band-utils-BcTxc8_Y.js";import{r as O}from"./spatial-interpolation-parameters-CfvG78hf.js";import{n as k,r as A}from"./gpu-inverse-distance-weighting-KIb4SzJG.js";import{t as j}from"./grid-cell-row-order-DUF05y4e.js";import{n as M}from"./point-neighbor-grid-index-D0wS1y1I.js";import{t as N}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as P}from"./exact-geometry-wgsl-y7TUnqOh.js";import{_ as F,a as I,b as L,f as R,g as z,n as B,o as V,r as H,t as ee,x as U,y as W}from"./density-scratch-D0AtmvQz.js";function te(e){return e===`epanechnikov`?192**(1/6):e===`quartic`?(2304/5)**(1/6):1}var ne=.32,re=2.5;function ie(e){if(e===`gaussian`){let e=f(1/(2*Math.PI));return`
fn getKernelValue(distanceSquared: f32, bandwidth: f32) -> f32 {
  let bandwidthSquared = bandwidth * bandwidth;
  let t = distanceSquared / bandwidthSquared;
  if (!(t < ${f(25)})) { return 0.0; }
  return ${e} * exp(-0.5 * t) / bandwidthSquared;
}`}let t=e===`epanechnikov`?1:2;return`
fn getKernelValue(distanceSquared: f32, bandwidth: f32) -> f32 {
  let bandwidthSquared = bandwidth * bandwidth;
  let t = distanceSquared / bandwidthSquared;
  if (!(t < 1.0)) { return 0.0; }
  let complement = 1.0 - t;
  return ${f((t+1)/Math.PI)} * ${t===1?`complement`:`complement * complement`} / bandwidthSquared;
}`}function G(e){if(e===`gaussian`)return`
${w}
fn getUpperTail(z: f32) -> f32 { return 0.5 * erfcValue(z * 0.70710678); }
fn getNormalMass(low: f32, high: f32) -> f32 {
  if (low > 0.0) { return getUpperTail(low) - getUpperTail(high); }
  if (high < 0.0) { return getUpperTail(-high) - getUpperTail(-low); }
  return 1.0 - getUpperTail(-low) - getUpperTail(high);
}
fn getWindowMass(point: vec2f, windowMin: vec2f, windowMax: vec2f, bandwidth: f32) -> f32 {
  let low = (windowMin - point) / bandwidth;
  let high = (windowMax - point) / bandwidth;
  return getNormalMass(low.x, high.x) * getNormalMass(low.y, high.y);
}`;let t=e===`epanechnikov`?1:2,n=f((t+1)/Math.PI),r=t===1?`height * (bulkEnd - bulkEnd3 / 3.0) - height3 * bulkEnd / 3.0`:`height * (bulkEnd - 2.0 * bulkEnd3 / 3.0 + bulkEnd5 / 5.0) - (2.0 * height3 / 3.0) * (bulkEnd - bulkEnd3 / 3.0) + height5 * bulkEnd / 5.0`,i=f(t===1?2/3:8/15);return`
fn getChordIntegral(x: f32) -> f32 {
  let clamped = clamp(x, 0.0, 1.0);
  let s = sqrt(max(1.0 - clamped * clamped, 0.0));
  let first = 0.5 * (clamped * s + asin(clamped));
  let third = clamped * s * s * s / 4.0 + 0.75 * first;
  ${t===1?`return third;`:`return clamped * s * s * s * s * s / 6.0 + (5.0 / 6.0) * third;`}
}
fn getQuadrantMass(a: f32, b: f32) -> f32 {
  let width = min(a, 1.0);
  let height = min(b, 1.0);
  if (!(width > 0.0) || !(height > 0.0)) { return 0.0; }
  let kink = sqrt(max(1.0 - height * height, 0.0));
  let bulkEnd = min(width, kink);
  let bulkEnd3 = bulkEnd * bulkEnd * bulkEnd;
  let bulkEnd5 = bulkEnd3 * bulkEnd * bulkEnd;
  let height3 = height * height * height;
  let height5 = height3 * height * height;
  var total = ${r};
  if (width > kink) {
    total += ${i} * (getChordIntegral(width) - getChordIntegral(kink));
  }
  return ${n} * total;
}
fn getSignedQuadrantMass(a: f32, b: f32) -> f32 {
  return sign(a) * sign(b) * getQuadrantMass(abs(a), abs(b));
}
fn getWindowMass(point: vec2f, windowMin: vec2f, windowMax: vec2f, bandwidth: f32) -> f32 {
  let low = (windowMin - point) / bandwidth;
  let high = (windowMax - point) / bandwidth;
  return getSignedQuadrantMass(high.x, high.y) - getSignedQuadrantMass(low.x, high.y) -
    getSignedQuadrantMass(high.x, low.y) + getSignedQuadrantMass(low.x, low.y);
}`}function ae(e,t,n=!1){return`var result = total;
  if (!valid) {
    result = getNaN();
  }
  ${e===`none`?``:`if (valid) {
    if (!(${n?`isInsideStudyWindow(center)`:`all(cellOffset >= windowMin) && all(cellOffset <= windowMax)`})) {
      result = getNaN();
    }${e===`uniform`?` else {
      let mass = ${n?`getStudyWindowKernelMass(center, bandwidth)`:`getWindowMass(cellOffset, windowMin, windowMax, bandwidth)`};
      result = select(getNaN(), total / mass, mass > 1e-6);
    }`:``}
  }`}
  ${t===`probability`?`if (valid) {
    let totalWeight = weightSum[weightSumOffset];
    result = select(getNaN(), result / totalWeight, totalWeight > 0.0);
  }`:``}
  valuesOut[valuesOutOffset + index] = result;`}function K(e,t){return`${N}${P}${_(y(e,t))}`}var q=`GPUKernelDensity`,oe=32,se=1e6,J=12,Y=2,ce=7,X={gather:0,bin:1,decode:2,blurX:3,blurY:4,finish:5},Z=64,le=32,ue=11184810;function de(e,t){let n=Math.max(e,t);for(;n>0;){let r=Q(e,t,n);if(r.paddedWidth*r.paddedHeight<=ue)break;n=Math.floor(n/2)}return n}function Q(e,t,n){let r=n+1;return{paddedWidth:e+2*r,paddedHeight:t+2*r,margin:r}}function fe(t,n){let{id:r,width:a,height:o,edgeCorrection:s,normalization:c,studyWindow:l}=n,u=n.positions.length,m=a*o,{paddedWidth:h,paddedHeight:g,margin:_}=Q(a,o,n.padding),y=h*g,w=n.autoGate?d(t,{id:`${r}-method-gate`,operation:q,shapes:[[m,Z],[u,Z],[y,Z],[g,le],[h,le],[m,Z]]}):void 0,T=e=>w?w.getGate(e):{},E=e(t,`${r}-blur-plan-words`,`float32`,J),D=e(t,`${r}-bin-scale-word`,`uint32`,1),O=e(t,`${r}-bin-words`,`uint32`,y*3),A=e(t,`${r}-bin-density`,`float32`,y),j=e(t,`${r}-bin-scratch`,`float32`,y),M=e=>({name:e,view:n.parameters,type:`f32`,access:`read`}),N=`const WIDTH: u32 = ${a}u;
const HEIGHT: u32 = ${o}u;
const PADDED_WIDTH: u32 = ${h}u;
const PADDED_HEIGHT: u32 = ${g}u;
const MARGIN: u32 = ${_}u;`,P=[p(t,{id:`${r}-blur-plan`,operation:q,variant:`blur-plan-${n.autoGate?`auto`:`binned`}`,bindings:[M(`params`),...n.bandwidthBindings,{name:`plan`,view:E,type:`f32`,access:`read_write`},...w?[w.binding]:[],...n.insufficientPadding?[{name:`insufficientPadding`,view:n.insufficientPadding,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`${N}
${k}
const EXPONENTIAL_SHARE: f32 = ${f(ne)};
const BOX_PASS_COUNT: f32 = ${f(3)};
const MINIMUM_REFRESH_INTERVAL: f32 = ${f(oe)};
const SUPPORT_FACTOR: f32 = ${f(5)};
const AUTO_SUPPORT_CELLS: f32 = ${f(32)};
const MARGIN_BANDWIDTHS: f32 = ${f(re)};
// Axis block: [box radius, box edge weight, box normalisation, refresh interval, exponential pole].
// Boxes are 2R + 1 cells plus fractional end taps of weight 'edge', so the variance of a pass
// is matched exactly: v = (R (R + 1) (2R + 1) / 3 + 2 edge (R + 1)^2) / (2R + 1 + 2 edge).
fn writeAxisPlan(base: u32, sigma: f32, lineLength: f32) {
  let varianceTotal = sigma * sigma;
  let varianceExponential = EXPONENTIAL_SHARE * varianceTotal;
  let varianceBox = (varianceTotal - varianceExponential) / BOX_PASS_COUNT;
  let radius = floor((sqrt(1.0 + 12.0 * varianceBox) - 1.0) * 0.5);
  let edge = clamp(
    (2.0 * radius + 1.0) * (varianceBox - radius * (radius + 1.0) / 3.0) /
      (2.0 * ((radius + 1.0) * (radius + 1.0) - varianceBox)),
    0.0,
    1.0
  );
  plan[planOffset + base] = radius;
  plan[planOffset + base + 1u] = edge;
  plan[planOffset + base + 2u] = 1.0 / (2.0 * radius + 1.0 + 2.0 * edge);
  plan[planOffset + base + 3u] = max(MINIMUM_REFRESH_INTERVAL, min(2.0 * radius + 3.0, lineLength));
  // Two-sided geometric with variance 2a / (1 - a)^2: the stable root of v a^2 - (2v + 2) a + v.
  plan[planOffset + base + 4u] = varianceExponential / ((varianceExponential + 1.0) + sqrt(2.0 * varianceExponential + 1.0));
}`,body:`let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let bandwidth = ${n.bandwidthExpression};
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellArea = abs(cellSize.x * cellSize.y);
  let sigmaCells = abs(vec2f(bandwidth) / cellSize);
  let valid = isFiniteValue(bandwidth) && bandwidth > 0.0 && isFiniteValue(cellSize.x) &&
    isFiniteValue(cellSize.y) && all(gridEnd != gridStart) && isFiniteValue(cellArea) &&
    cellArea > 0.0 && isFiniteValue(sigmaCells.x) && isFiniteValue(sigmaCells.y);
  let sigma = select(vec2f(0.0), min(sigmaCells, vec2f(${f(se)})), valid);
  plan[planOffset] = select(0.0, 1.0, valid);
  plan[planOffset + 1u] = select(0.0, 1.0 / cellArea, valid);
  writeAxisPlan(${Y}u, sigma.x, f32(PADDED_WIDTH));
  writeAxisPlan(${ce}u, sigma.y, f32(PADDED_HEIGHT));
  // The cascade drops mass that leaves the array, so it needs a margin of a few bandwidths.
  let sufficient = f32(MARGIN) >= MARGIN_BANDWIDTHS * max(sigma.x, sigma.y);${n.insufficientPadding?`
  insufficientPadding[insufficientPaddingOffset] = ${n.autoGate?`0u`:`select(0u, 1u, valid && !sufficient)`};`:``}${w?`
  // The exact gather visits (support / cell)^2 cells per point; the blur costs the same at any
  // bandwidth, so it wins once the support radius exceeds the threshold, provided the margin
  // suffices (otherwise the exact path stays correct).
  let useBinned = valid && sufficient && SUPPORT_FACTOR * max(sigma.x, sigma.y) > AUTO_SUPPORT_CELLS;
  ${w.armWGSL}
  if (useBinned) {
    dispatch[dispatchOffset + ${3*X.gather}u] = 0u;
  } else {
    ${[X.bin,X.decode,X.blurX,X.blurY,X.finish].map(e=>`dispatch[dispatchOffset + ${3*e}u] = 0u;`).join(`
    `)}
  }`:``}`})],F=i(n.positions),I=i(n.pointWeights),L=I.length===F.length,B=0,V=[...R(t,{id:`${r}-bin-scale`,operation:q,values:n.pointWeights,scale:D}),x(t,{id:`${r}-bin-clear`,operation:q,words:O}),...F.flatMap((e,n)=>{let i=B;return B+=e.length,e.length===0?[]:[z(t,{id:`${r}-bin${F.length===1?``:`-${n}`}`,operation:q,variant:`bin`,bindings:[...l?[v(l)]:[],{name:`positions`,view:e,type:`f32`,access:`read`},{name:`pointWeights`,view:I[L?n:0],type:`f32`,access:`read`},M(`params`),{name:`binScale`,view:D,type:`u32`,access:`read`},{name:`words`,view:O,type:`atomic<u32>`,access:`read_write`}],invocationCount:e.length,workgroupSize:Z,...T(X.bin),runtimeConstants:{ROW_START:L?0:i},declarations:`override FIXED_POINT_HEADROOM_BITS: u32;
${N}
${C}
${b(`words`)}${l?K(l):``}`,constants:{FIXED_POINT_HEADROOM_BITS:S(u)},body:`let weight = pointWeights[pointWeightsOffset + ROW_START + index];
  if (!(weight > 0.0)) {
    return;
  }
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  ${l?`if (!isInsideStudyWindow(position)) { return; }
  `:``}let coordinate = (position - gridStart) / cellSize + vec2f(f32(MARGIN) - 0.5);
  if (!(coordinate.x >= 0.0 && coordinate.y >= 0.0 &&
      coordinate.x < f32(PADDED_WIDTH - 1u) && coordinate.y < f32(PADDED_HEIGHT - 1u))) {
    return;
  }
  let base = floor(coordinate);
  let fraction = coordinate - base;
  let cell = u32(base.y) * PADDED_WIDTH + u32(base.x);
  let unitExponent = getFixedPointUnitExponent(binScale[binScaleOffset], FIXED_POINT_HEADROOM_BITS);
  addFixedPointToWords(cell, quantizeFixedPoint(weight * (1.0 - fraction.x) * (1.0 - fraction.y), unitExponent));
  addFixedPointToWords(cell + 1u, quantizeFixedPoint(weight * fraction.x * (1.0 - fraction.y), unitExponent));
  addFixedPointToWords(cell + PADDED_WIDTH, quantizeFixedPoint(weight * (1.0 - fraction.x) * fraction.y, unitExponent));
  addFixedPointToWords(cell + PADDED_WIDTH + 1u, quantizeFixedPoint(weight * fraction.x * fraction.y, unitExponent));`})]}),p(t,{id:`${r}-bin-decode`,operation:q,variant:`bin-decode`,bindings:[{name:`words`,view:O,type:`u32`,access:`read`},{name:`binScale`,view:D,type:`u32`,access:`read`},{name:`density`,view:A,type:`f32`,access:`read_write`}],invocationCount:y,workgroupSize:Z,...T(X.decode),declarations:`override FIXED_POINT_HEADROOM_BITS: u32;
${C}`,constants:{FIXED_POINT_HEADROOM_BITS:S(u)},body:`let base = wordsOffset + index * 3u;
  let unitExponent = getFixedPointUnitExponent(binScale[binScaleOffset], FIXED_POINT_HEADROOM_BITS);
  density[densityOffset + index] = getFixedPointFloat(
    vec3<u32>(words[base], words[base + 1u], words[base + 2u]),
    unitExponent
  );`})];for(let e of[`x`,`y`]){let n=e===`x`,i=n?g:h,a={LINE_LENGTH:n?h:g,LINE_STEP:n?1:h,LINE_STRIDE:n?h:1,PLAN_BASE:n?Y:ce},o=[[`box`,A,j],[`box`,j,A],[`box`,A,j],[`exponential`,j,A]];for(let[s,[c,l,u]]of o.entries())V.push(p(t,{id:`${r}-blur-${e}-${c}-${s}`,operation:q,variant:`blur-${c}`,bindings:[{name:`source`,view:l,type:`f32`,access:`read`},{name:`destination`,view:u,type:`f32`,access:`read_write`},{name:`plan`,view:E,type:`f32`,access:`read`}],invocationCount:i,workgroupSize:le,...T(n?X.blurX:X.blurY),declarations:`override LINE_LENGTH: u32;
override LINE_STEP: u32;
override LINE_STRIDE: u32;
override PLAN_BASE: u32;
${c===`box`?pe:``}`,constants:a,body:c===`box`?me:he}))}let H=[...l?[v(l)]:[],{name:`smoothed`,view:A,type:`f32`,access:`read`},M(`params`),{name:`plan`,view:E,type:`f32`,access:`read`},...n.bandwidthBindings,...c===`probability`&&n.weightSum?[{name:`weightSum`,view:n.weightSum,type:`f32`,access:`read`}]:[],{name:`valuesOut`,view:n.values,type:`f32`,access:`read_write`}];return V.push(p(t,{id:`${r}-finish`,operation:q,variant:`finish-${s}-${c}`,bindings:H,invocationCount:m,workgroupSize:Z,...T(X.finish),declarations:`${N}
${k}
${l?K(l,s===`uniform`?`gaussian`:void 0):s===`none`?``:G(`gaussian`)}`,body:`let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let bandwidth = ${n.bandwidthExpression};
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let valid = plan[planOffset] > 0.5 && isFiniteValue(center.x) && isFiniteValue(center.y);
  var total = 0.0;
  if (valid) {
    let paddedIndex = (rasterRow + MARGIN) * PADDED_WIDTH + column + MARGIN;
    total = smoothed[smoothedOffset + paddedIndex] * plan[planOffset + 1u];
  }
  ${ae(s,c,!!l)}`})),{planNodes:P,bodyNodes:V,exactGate:w?w.getGate(X.gather):void 0}}var pe=`
fn readTap(base: u32, position: i32) -> f32 {
  if (position < 0 || position >= i32(LINE_LENGTH)) { return 0.0; }
  return source[sourceOffset + base + u32(position) * LINE_STEP];
}`,me=`let base = index * LINE_STRIDE;
  let radius = i32(plan[planOffset + PLAN_BASE]);
  let edge = plan[planOffset + PLAN_BASE + 1u];
  let normalization = plan[planOffset + PLAN_BASE + 2u];
  let interval = i32(plan[planOffset + PLAN_BASE + 3u]);
  let lineLength = i32(LINE_LENGTH);
  var window = 0.0;
  for (var position = 0; position < lineLength; position++) {
    if (position % interval == 0) {
      var sum = 0.0;
      let last = min(position + radius, lineLength - 1);
      for (var tap = max(position - radius, 0); tap <= last; tap++) {
        sum += readTap(base, tap);
      }
      window = sum + edge * (readTap(base, position - radius - 1) + readTap(base, position + radius + 1));
    } else {
      window += (1.0 - edge) * (readTap(base, position + radius) - readTap(base, position - radius - 1)) +
        edge * (readTap(base, position + radius + 1) - readTap(base, position - radius - 2));
    }
    destination[destinationOffset + base + u32(position) * LINE_STEP] = max(window, 0.0) * normalization;
  }`,he=`let base = index * LINE_STRIDE;
  let pole = plan[planOffset + PLAN_BASE + 4u];
  var state = 0.0;
  for (var position = 0u; position < LINE_LENGTH; position++) {
    let slot = destinationOffset + base + position * LINE_STEP;
    state = (1.0 - pole) * source[sourceOffset + base + position * LINE_STEP] + pole * state;
    destination[slot] = state;
  }
  var next = state / (1.0 + pole);
  destination[destinationOffset + base + (LINE_LENGTH - 1u) * LINE_STEP] = next;
  for (var position = LINE_LENGTH - 1u; position > 0u; position--) {
    let slot = destinationOffset + base + (position - 1u) * LINE_STEP;
    next = (1.0 - pole) * destination[slot] + pole * next;
    destination[slot] = next;
  }`,ge=Object.freeze({...m,measure:`area`}),_e=20,$=`GPUKernelDensity`,ve=4294967295,ye=4294967295,be=512;function xe(e){return`const DOMAIN_MIN: vec2f = vec2f(${f(e[0])}, ${f(e[1])});
const DOMAIN_MAX: vec2f = vec2f(${f(e[2])}, ${f(e[3])});`}function Se(e,t,n=new Float32Array(9)){if(n.length<9)throw Error(`Kernel density parameter target must hold 9 elements`);if(!Number.isFinite(e.bandwidth)||e.bandwidth<=0)throw Error(`Kernel density bandwidth must be positive and finite`);let r=O(`getGPUKernelDensityParameterValues`,t),[i,a,o,s]=r;if(!r.every(Number.isFinite)||i===o||a===s)throw Error(`Kernel density grid must have a finite, nonzero cell size`);let c=e.window??[Math.min(i,o),Math.min(a,s),Math.max(i,o),Math.max(a,s)];if(c!==`study-window`&&(c.length!==4||!c.every(Number.isFinite)||c[0]>=c[2]||c[1]>=c[3]))throw Error(`Kernel density window must be finite [minX, minY, maxX, maxY]`);return n.set(r,0),n[4]=e.bandwidth,c===`study-window`?n.fill(NaN,5,9):n.set(c,5),n}var Ce=class{id;props;spatialContext;studyWindow;kernel;edgeCorrection;normalization;method;binnedPadding;estimatedCommandNodeCount;cellCount;constructor(e){this.id=e.id??`kernel-density`,this.props=e;let{id:s}=this;if(this.spatialContext=h(s,e.spatialContext,ge),this.studyWindow=g(s,e.studyWindow,this.spatialContext),this.kernel=e.kernel===`bisquare`?`quartic`:e.kernel??`gaussian`,this.edgeCorrection=e.edgeCorrection??`uniform`,this.studyWindow!==`rectangle`&&this.edgeCorrection===`none`)throw Error(`${s} studyWindow is ignored without edge correction; set edgeCorrection to 'uniform' or 'diggle'`);this.normalization=e.normalization??`intensity`,this.binnedPadding=e.binnedPadding??de(e.grid.width,e.grid.height);let l=e.method??`exact`;if(![`gaussian`,`quartic`,`epanechnikov`].includes(this.kernel))throw Error(`${s} kernel must be gaussian, quartic (alias bisquare), or epanechnikov`);if(![`none`,`uniform`,`diggle`].includes(this.edgeCorrection))throw Error(`${s} edgeCorrection must be none, uniform, or diggle`);if(![`intensity`,`probability`].includes(this.normalization))throw Error(`${s} normalization must be intensity or probability`);if(![`exact`,`binned`,`auto`].includes(l))throw Error(`${s} method must be exact, binned, or auto`);if(l===`binned`&&this.kernel!==`gaussian`)throw Error(`${s} method binned needs the gaussian kernel`);if(!Number.isSafeInteger(this.binnedPadding)||this.binnedPadding<0)throw Error(`${s} binnedPadding must be a nonnegative integer`);if(this.method=l===`auto`&&this.kernel!==`gaussian`?`exact`:l,n(s,`positions.length`,e.positions.length,ve),this.estimatedCommandNodeCount=this.method===`binned`?be+_e*i(e.positions).length:void 0,this.estimatedCommandNodeCount!==void 0&&c(s,this.estimatedCommandNodeCount),e.bandwidthRule!==void 0&&![`scott`,`silverman`].includes(e.bandwidthRule))throw Error(`${s} bandwidthRule must be scott or silverman`);if(e.output.bandwidth&&!e.bandwidthRule)throw Error(`${s} output.bandwidth needs bandwidthRule`);if(this.cellCount=D(s,e.grid),e.grid.affine[1]!==0||e.grid.affine[3]!==0)throw Error(`${s} needs an axis-aligned grid`);if(this.method!==`exact`){let{paddedWidth:t,paddedHeight:n}=Q(e.grid.width,e.grid.height,this.binnedPadding);if(t*n>ue)throw Error(`${s} binned grid of ${t} by ${n} cells is too large`)}for(let[t,n]of[[`indexGridSize[0]`,e.indexGridSize[0]],[`indexGridSize[1]`,e.indexGridSize[1]]])a(s,t,n,1,ye);for(let[t,n]of e.indexBounds.entries())o(s,`indexBounds[${t}]`,n);let[d,f,p,m]=e.indexBounds;if(e.indexBounds.length!==4||!e.indexBounds.every(Number.isFinite)||d>p||f>m)throw Error(`${s} indexBounds must be finite and ordered`);for(let n of i(e.positions))t(n,[`float32x2`],`${s} positions`);if(this.method!==`binned`&&(t(e.positions,[`float32x2`],`${s} positions`),e.weights&&t(e.weights,[`float32`],`${s} weights`)),e.positions.length<1)throw Error(`${s} needs at least one point`);if(e.weights){for(let n of i(e.weights))t(n,[`float32`],`${s} weights`);if(e.weights.length!==e.positions.length)throw Error(`${s} weights length must equal positions length`);r(s,`weights`,e.positions,e.weights)}if(t(e.parameters,[`float32`],`${s} parameters`),e.parameters.length<9)throw Error(`${s} parameters must hold 9 float32 values`);if(t(e.output.values,[`float32`],`${s} output.values`),e.output.values.length<this.cellCount)throw Error(`${s} output.values must hold grid.width * grid.height rows`);if(e.output.weightSum&&(t(e.output.weightSum,[`float32`],`${s} output.weightSum`),e.output.weightSum.length!==1))throw Error(`${s} output.weightSum must contain exactly one float32 row`);if(e.output.status?.invalidCount&&(t(e.output.status.invalidCount,[`uint32`],`${s} output.status.invalidCount`),e.output.status.invalidCount.length!==1))throw Error(`${s} output.status.invalidCount must contain exactly one uint32 row`);if(e.output.status?.insufficientPadding){if(this.method===`exact`)throw Error(`${s} output.status.insufficientPadding needs method binned or auto`);if(t(e.output.status.insufficientPadding,[`uint32`],`${s} output.status.insufficientPadding`),e.output.status.insufficientPadding.length!==1)throw Error(`${s} output.status.insufficientPadding must contain exactly one uint32 row`)}if(e.output.bandwidth&&(t(e.output.bandwidth,[`float32`],`${s} output.bandwidth`),e.output.bandwidth.length!==1))throw Error(`${s} output.bandwidth must contain exactly one float32 row`);U(s,{"output.values":e.output.values,"output.weightSum":e.output.weightSum,"output.status.invalidCount":e.output.status?.invalidCount,"output.status.insufficientPadding":e.output.status?.insufficientPadding,"output.bandwidth":e.output.bandwidth}),u(s,[e.output.values,e.output.weightSum,e.output.status?.invalidCount,e.output.status?.insufficientPadding,e.output.bandwidth],[e.positions,e.weights,e.parameters])}getBandwidthRuleNodes(t,n){let{id:r,props:a}=this,o=I(t,a.positions.length),s=[`weight`,`weight-square`,`x`,`y`].map(e=>B(t,`${r}-bandwidth-term-${e}`,a.positions,`float32`,!o)),[c,l,u,d]=s,f=s.map(e=>i(e)),[m,h,g,_,v,y]=[`weight`,`weight-square`,`x`,`y`,`spread-x`,`spread-y`].map(n=>e(t,`${r}-bandwidth-sum-${n}`,`float32`,1)),b=e(t,`${r}-bandwidth-means`,`float32`,2),x=i(a.positions),S=a.weights?i(a.weights):void 0,C=e=>[{name:`positions`,view:x[e],type:`f32`,access:`read`},...S?[{name:`inputWeights`,view:S[e],type:`f32`,access:`read`}]:[],{name:`params`,view:a.parameters,type:`f32`,access:`read`}],w=0,D=`let origin = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let rawWeight = ${a.weights?`inputWeights[inputWeightsOffset + index]`:`1.0`};
  let included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    isFiniteValue(rawWeight) && rawWeight >= 0.0;
  let weight = select(0.0, rawWeight, included);
  let relative = select(vec2f(0.0), position - origin, included);`,O=a.weights?`weighted`:`unweighted`,k=[...x.flatMap((e,n)=>{let i=w;return w+=e.length,e.length===0?[]:[z(t,{id:`${r}-bandwidth-terms${x.length===1?``:`-${n}`}`,operation:$,variant:`bandwidth-terms-${O}`,bindings:[...C(n),{name:`termWeight`,view:f[0][o?n:0],type:`f32`,access:`read_write`},{name:`termWeightSquare`,view:f[1][o?n:0],type:`f32`,access:`read_write`},{name:`termX`,view:f[2][o?n:0],type:`f32`,access:`read_write`},{name:`termY`,view:f[3][o?n:0],type:`f32`,access:`read_write`}],invocationCount:e.length,runtimeConstants:{ROW_START:o?0:i},declarations:E,body:`${D}
  termWeight[termWeightOffset + ROW_START + index] = weight;
  termWeightSquare[termWeightSquareOffset + ROW_START + index] = weight * weight;
  termX[termXOffset + ROW_START + index] = weight * relative.x;
  termY[termYOffset + ROW_START + index] = weight * relative.y;`})]})];for(let[e,n,i]of[[`weight`,c,m],[`weight-square`,l,h],[`x`,u,g],[`y`,d,_]])k.push(...H(t,`${r}-bandwidth-sum-${e}`,n,i));w=0,k.push(p(t,{id:`${r}-bandwidth-means`,operation:$,variant:`bandwidth-means`,bindings:[{name:`sumWeight`,view:m,type:`f32`,access:`read`},{name:`sumX`,view:g,type:`f32`,access:`read`},{name:`sumY`,view:_,type:`f32`,access:`read`},{name:`means`,view:b,type:`f32`,access:`read_write`}],invocationCount:1,body:`let total = sumWeight[sumWeightOffset];
  let valid = total > 0.0;
  means[meansOffset] = select(0.0, sumX[sumXOffset] / total, valid);
  means[meansOffset + 1u] = select(0.0, sumY[sumYOffset] / total, valid);`}),...x.flatMap((e,n)=>{let i=w;return w+=e.length,e.length===0?[]:[z(t,{id:`${r}-bandwidth-spread${x.length===1?``:`-${n}`}`,operation:$,variant:`bandwidth-spread-${O}`,bindings:[...C(n),{name:`means`,view:b,type:`f32`,access:`read`},{name:`termX`,view:f[2][o?n:0],type:`f32`,access:`read_write`},{name:`termY`,view:f[3][o?n:0],type:`f32`,access:`read_write`}],invocationCount:e.length,runtimeConstants:{ROW_START:o?0:i},declarations:E,body:`${D}
  let centered = relative - vec2f(means[meansOffset], means[meansOffset + 1u]);
  termX[termXOffset + ROW_START + index] = select(0.0, weight * centered.x * centered.x, included);
  termY[termYOffset + ROW_START + index] = select(0.0, weight * centered.y * centered.y, included);`})]}));for(let[e,n,i]of[[`spread-x`,u,v],[`spread-y`,d,y]])k.push(...H(t,`${r}-bandwidth-sum-${e}`,n,i));return k.push(p(t,{id:`${r}-bandwidth-rule`,operation:$,variant:`bandwidth-rule-${this.kernel}`,bindings:[{name:`sumWeight`,view:m,type:`f32`,access:`read`},{name:`sumWeightSquare`,view:h,type:`f32`,access:`read`},{name:`sumSpreadX`,view:v,type:`f32`,access:`read`},{name:`sumSpreadY`,view:y,type:`f32`,access:`read`},{name:`bandwidthState`,view:n,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${E}${T}`,body:`let total = sumWeight[sumWeightOffset];
  let totalSquare = sumWeightSquare[sumWeightSquareOffset];
  let effective = total * total / totalSquare;
  let correction = 1.0 - totalSquare / (total * total);
  let variance = 0.5 * (sumSpreadX[sumSpreadXOffset] + sumSpreadY[sumSpreadYOffset]) / (total * correction);
  let bandwidth = ${te(this.kernel)} * pow(effective, -1.0 / 6.0) * sqrt(variance);
  let valid = total > 0.0 && totalSquare > 0.0 && effective > 1.0 && correction > 0.0 &&
    isFiniteValue(bandwidth) && bandwidth > 0.0;
  bandwidthState[bandwidthStateOffset] = select(getNaN(), bandwidth, valid);`})),k}getIndexNodes(e){let{id:t,props:n}=this,r=n.positions.length,{indexGridSize:i,indexBounds:a}=n,o=M(e,{id:t,indexId:`${t}-index`,positions:n.positions,gridSize:i,bounds:a,rowCount:r}),{cellOffsets:s,sortedRows:c,gridCount:l}=o,u=[...o.nodes],d=j(e,{id:`${t}-cell-row-order`,operation:$,cellOffsets:s,objectIds:c,indexCount:l,rowCount:r});return u.push(...d.nodes),{nodes:u,cellOffsets:s,sortedIds:d.sortedRows}}getCommandNodes(t){let{id:r,props:o,kernel:c,edgeCorrection:u,normalization:d,method:f}=this,{output:m,grid:h,indexBounds:g}=o,_=m;if(n(r,`indexGridSize cells`,o.indexGridSize[0]*o.indexGridSize[1],s(t.device)),L(t,r,[[`positions`,o.positions],[`weights`,o.weights],[`parameters`,o.parameters],[`output.values`,_.values],[`output.weightSum`,_.weightSum],[`output.bandwidth`,_.bandwidth],[`output.status.invalidCount`,_.status?.invalidCount],[`output.status.insufficientPadding`,_.status?.insufficientPadding]],{skipEmptyChunks:!0}),f!==`binned`&&W(t,r,[[`index offset scratch`,(o.indexGridSize[0]*o.indexGridSize[1]+1)*4]]),V(t,r,o.positions,4),W(t,r,[[`grid scratch`,this.cellCount*4]]),a(r,`grid scratch byteLength`,this.cellCount*4,0,t.device.limits.maxBufferSize),f!==`exact`){W(t,r,[[`binned blur plan scratch`,48]]);let{paddedWidth:e,paddedHeight:n}=Q(h.width,h.height,this.binnedPadding);W(t,r,[[`binned fixed-point grid scratch`,e*n*4*3]]),a(r,`binned fixed-point grid scratch byteLength`,e*n*4*3,0,t.device.limits.maxBufferSize)}let y=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(t);y&&(L(t,r,[[`studyWindow.edges`,y.edges],[`studyWindow.summary`,y.summary],[`studyWindow.slabs.params`,y.slabs.params],[`studyWindow.slabs.slabOffsets`,y.slabs.slabOffsets],[`studyWindow.slabs.slabEntries`,y.slabs.slabEntries],[`studyWindow.sampleCells`,y.sampleCells],[`studyWindow.sampleCellCount`,y.sampleCellCount],[`studyWindow.packed`,y.packed]]),l(r,t,[y.edges,y.summary,y.slabs.params,y.slabs.slabOffsets,y.slabs.slabEntries,y.sampleCells,y.sampleCellCount,y.packed]));let b=f!==`binned`,{width:x,height:S}=h;l(r,t,[o.positions,o.weights,o.parameters,_.values,_.weightSum,_.status?.invalidCount,_.status?.insufficientPadding,_.bandwidth]);let C=F(t,r,$,{values:_.values,weightSum:_.weightSum,bandwidth:_.bandwidth,invalidCount:_.status?.invalidCount,insufficientPadding:_.status?.insufficientPadding},{values:this.cellCount});_={...m,...C.views,status:m.status?{invalidCount:C.views.invalidCount,insufficientPadding:C.views.insufficientPadding}:void 0};let w=[],T=b?this.getIndexNodes(t):void 0;T&&w.push(...T.nodes);let E=o.bandwidthRule?_.bandwidth??e(t,`${r}-bandwidth-state`,`float32`,1):void 0;E&&w.push(...this.getBandwidthRuleNodes(t,E));let D=E?[{name:`bandwidthState`,view:E,type:`f32`,access:`read`}]:[],O=E?`bandwidthState[bandwidthStateOffset]`:`params[paramsOffset + 4u]`,A=u===`diggle`&&(d===`probability`||!!_.weightSum),j=d===`probability`||!!_.weightSum,M=I(t,o.positions.length),N=B(t,`${r}-point-weights`,o.positions,`float32`,!M),P=A?B(t,`${r}-raw-weights`,o.positions,`float32`,!M):N,R=_.status?.invalidCount?B(t,`${r}-excluded`,o.positions,`uint32`,!M):void 0,U=i(N),te=i(P),ne=R?i(R):void 0,re=xe(g),ie=u!==`none`,q=i(o.positions),oe=o.weights?i(o.weights):void 0,se=0;for(let[e,n]of q.entries()){let i=se;if(se+=n.length,n.length===0)continue;let a=[...y?[v(y)]:[],{name:`positions`,view:n,type:`f32`,access:`read`},...o.weights?[{name:`inputWeights`,view:oe[e],type:`f32`,access:`read`}]:[],{name:`params`,view:o.parameters,type:`f32`,access:`read`},...D,{name:`pointWeights`,view:U[M?e:0],type:`f32`,access:`read_write`},...A?[{name:`rawWeights`,view:te[M?e:0],type:`f32`,access:`read_write`}]:[],...R?[{name:`excluded`,view:ne[M?e:0],type:`u32`,access:`read_write`}]:[]];w.push(z(t,{id:`${r}-point-weights${q.length===1?``:`-${e}`}`,operation:$,variant:`point-weights-${u}`,bindings:a,invocationCount:n.length,runtimeConstants:{ROW_START:M?0:i},declarations:`${re}
${k}
${y?K(y,u===`diggle`?c:void 0):u===`diggle`?G(c):``}`,body:`let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let bandwidth = ${O};
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let weight = ${o.weights?`inputWeights[inputWeightsOffset + index]`:`1.0`};
  var included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    all(position >= DOMAIN_MIN) && all(position <= DOMAIN_MAX) &&
    isFiniteValue(weight) && weight >= 0.0;
  let relative = position - gridStart;
  ${y?`included = included && isInsideStudyWindow(position);`:ie?`included = included && all(relative >= windowMin) && all(relative <= windowMax);`:``}
  var corrected = select(0.0, weight, included);
  ${P===N?``:`rawWeights[rawWeightsOffset + ROW_START + index] = corrected;`}
  ${u===`diggle`?`if (included) {
    let mass = ${y?`getStudyWindowKernelMass(position, bandwidth)`:`getWindowMass(relative, windowMin, windowMax, bandwidth)`};
    corrected = select(0.0, weight / mass, mass > 1e-6);
  }`:``}
  pointWeights[pointWeightsOffset + ROW_START + index] = corrected;
  ${R?`excluded[excludedOffset + ROW_START + index] = select(1u, 0u, included);`:``}`}))}let J=j?_.weightSum??e(t,`${r}-weight-sum-scratch`,`float32`,1):void 0;J&&w.push(...H(t,`${r}-weight-sum`,P,J)),R&&_.status?.invalidCount&&w.push(...ee(t,`${r}-invalid-count`,R,_.status.invalidCount));let Y=f===`exact`?void 0:fe(t,{id:r,positions:o.positions,parameters:o.parameters,pointWeights:N,studyWindow:y,bandwidthBindings:D,bandwidthExpression:O,weightSum:J,values:_.values,width:x,height:S,padding:this.binnedPadding,insufficientPadding:_.status?.insufficientPadding,edgeCorrection:u,normalization:d,autoGate:f===`auto`});if(Y&&w.push(...Y.planNodes),T){let n=y?e(t,`${r}-gathered`,`float32`,this.cellCount):void 0;w.push(this.getGatherNode(t,{exactIndex:T,gathered:n,values:_.values,pointWeights:U[0],bandwidthBindings:D,bandwidthExpression:O,weightSum:J,gate:Y?.exactGate})),y&&n&&w.push(p(t,{id:`${r}-polygon-finish`,operation:$,variant:`polygon-finish-${c}-${u}-${d}`,bindings:[{name:`gathered`,view:n,type:`f32`,access:`read`},{name:`params`,view:o.parameters,type:`f32`,access:`read`},...D,v(y),...d===`probability`&&J?[{name:`weightSum`,view:J,type:`f32`,access:`read`}]:[],{name:`valuesOut`,view:_.values,type:`f32`,access:`read_write`}],invocationCount:this.cellCount,workgroupSize:64,...Y?.exactGate,declarations:`${k}${K(y,u===`uniform`?c:void 0)}`,body:`let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let cellSize = (gridEnd - gridStart) / vec2f(${x}.0, ${S}.0);
  let center = gridStart + (vec2f(f32(index % ${x}u), f32(index / ${x}u)) + vec2f(0.5)) * cellSize;
  let bandwidth = ${O};
  let total = gathered[gatheredOffset + index];
  let valid = isFiniteValue(total) && isStudyWindowUsable();
  ${ae(u,d,!0)}`}))}return Y&&w.push(...Y.bodyNodes),w.push(...C.publishNodes),w}getGatherNode(e,t){let{id:n,props:r,cellCount:i,kernel:a}=this,o=t.gathered?`none`:this.edgeCorrection,s=t.gathered?`intensity`:this.normalization,{grid:c,indexGridSize:l,indexBounds:u}=r,{width:d,height:m}=c,{pointWeights:h,bandwidthBindings:g,bandwidthExpression:_,weightSum:v,gate:y}=t,{cellOffsets:b,sortedIds:x}=t.exactIndex,S=xe(u),C=[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`pointWeights`,view:h,type:`f32`,access:`read`},{name:`params`,view:r.parameters,type:`f32`,access:`read`},...g,{name:`cellOffsets`,view:b,type:`u32`,access:`read`},{name:`sortedIds`,view:x,type:`u32`,access:`read`},...s===`probability`&&v?[{name:`weightSum`,view:v,type:`f32`,access:`read`}]:[],{name:`valuesOut`,view:t.gathered??t.values,type:`f32`,access:`read_write`}],w=a===`gaussian`?5:1;return p(e,{id:`${n}-gather`,operation:$,variant:`gather-${a}-${o}-${s}`,bindings:C,invocationCount:i,workgroupSize:64,...y,declarations:`const WIDTH: u32 = ${d}u;
const HEIGHT: u32 = ${m}u;
const INDEX_WIDTH: u32 = ${l[0]}u;
const INDEX_HEIGHT: u32 = ${l[1]}u;
const SUPPORT_FACTOR: f32 = ${f(w)};
${S}
${k}
${ie(a)}
${o===`none`?``:G(a)}`,body:`let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let bandwidth = ${_};
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let radius = SUPPORT_FACTOR * bandwidth;
  let supportSquared = radius * radius;
  let queryMin = center - vec2f(radius);
  let queryMax = center + vec2f(radius);
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) &&
    isFiniteValue(bandwidth) && bandwidth > 0.0 && isFiniteValue(radius) &&
    all(gridEnd != gridStart);
  var total = 0.0;
  if (valid && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX)) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    // Widen by one index cell per side to absorb f32 rounding in the binning pass; the exact
    // distance test rejects the extra candidates.
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    ${A(`let pointWeight = pointWeights[pointWeightsOffset + sample];
        if (!(pointWeight > 0.0)) {
          continue;
        }
        let position = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]);
        // Distances come from local-origin offsets so large absolute coordinates cancel exactly.
        let delta = (position - gridStart) - cellOffset;
        let distanceSquared = dot(delta, delta);
        if (!(distanceSquared < supportSquared)) {
          continue;
        }
        total += pointWeight * getKernelValue(distanceSquared, bandwidth);`)}
  }
  ${ae(o,s)}`})}};export{G as i,Se as n,ie as r,Ce as t};
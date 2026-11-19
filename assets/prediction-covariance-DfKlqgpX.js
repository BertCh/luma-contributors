import{c as e}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as t,n,t as r}from"./variogram-model-l9c6-v8f.js";function i(e,t){return 3/e**(2*t)}function a(e,t,n){return(Math.exp(3/n)-1)/e**(2*t)}function o(e,n){return n<=0?0:e.nugget+e.sill*t(e.model,n/e.range)}function s(e,t,n){if(t<=0&&n<=0)return 0;switch(e.type){case`separable`:{let r=o(e.spatial,t),i=o(e.temporal,n);return e.sill*(r+i-r*i)}case`product-sum`:{let r=o(e.spatial,t),i=o(e.temporal,n),a=e.spatial.nugget+e.spatial.sill,s=e.temporal.nugget+e.temporal.sill;return(e.k*s+1)*r+(e.k*a+1)*i-e.k*r*i}case`metric`:return o(e.joint,Math.hypot(t,e.anisotropy*n));case`sum-metric`:return o(e.spatial,t)+o(e.temporal,n)+o(e.joint,Math.hypot(t,e.anisotropy*n));case`gneiting`:return e.nugget+e.sill*(1-c(e,t,n));default:throw Error(`Unknown space-time variogram model`)}}function c(e,t,n){let r=i(e.spatialRange,e.gamma),o=a(e.temporalRange,e.alpha,e.tau)*n**(2*e.alpha)+1;return o**-e.tau*Math.exp(-r*t**(2*e.gamma)*o**(-e.beta*e.gamma))}var l=.1;function u(e){switch(e.model){case`spherical`:return 0;case`exponential`:return 1;case`gaussian`:return 2;case`matern-3/2`:return 3;case`matern-5/2`:return 4;case`matern`:switch(e.smoothness){case .5:return 1;case 1.5:return 3;case 2.5:return 4;default:throw Error(`Invalid Matern smoothness`)}default:throw Error(`Invalid prediction covariance model`)}}function d(e){if(u(e),!(e.range>0)||!(e.sill>=0)||!((e.nugget??0)>=0)||!(e.sill+(e.nugget??0)>0)||![e.range,e.sill,e.nugget??0].every(Number.isFinite))throw Error(`Invalid prediction covariance`)}function f(t,n,r){return{statements:`let maternArgument = ${e(t)} * scaled;`,expression:`select(${r}, ${n}, maternArgument < ${e(l)})`}}function p(t){switch(t){case 0:return{statements:``,expression:`select(1.0 - 1.5 * scaled + 0.5 * scaled * scaled * scaled, 0.0, scaled >= 1.0)`};case 1:return{statements:``,expression:`exp(-3.0 * scaled)`};case 2:return{statements:``,expression:`exp(-3.0 * scaled * scaled)`};case 3:return f(r,`1.0 - maternArgument * maternArgument * (${e(1/2)} - maternArgument * (${e(1/3)} - maternArgument * (${e(1/8)} - maternArgument * (${e(1/30)} - maternArgument * ${e(1/144)}))))`,`(1.0 + maternArgument) * exp(-maternArgument)`);case 4:return f(n,`1.0 - maternArgument * maternArgument * (${e(1/6)} - maternArgument * maternArgument * (${e(1/24)} - maternArgument * (${e(1/45)} - maternArgument * ${e(1/144)})))`,`(1.0 + maternArgument + maternArgument * maternArgument * ${e(1/3)}) * exp(-maternArgument)`);default:throw Error(`Invalid prediction covariance model code`)}}function m(t){let{statements:n,expression:r}=p(u(t));return`fn covariance(first: vec2<f32>, second: vec2<f32>) -> f32 {
    let scaled = distance(first, second) / ${e(t.range)};
    ${n}
    return ${e(t.sill)} * (${r}) + select(0.0, ${e(t.nugget??0)}, all(first == second));
  }`}function h(e=1,t=1){return 4+4*e+(t>1?e*t**2:0)}function g(e,t){let n=e.coregionalizationModels??(e.coregionalization&&e.covariance?[{covariance:e.covariance,loadings:e.coregionalization}]:e.covariance?[{covariance:e.covariance,loadings:[1]}]:[]),r=!!(e.coregionalizationModels||e.coregionalization),i=e.variableCount??(r?2:1),a=h(n.length,i);if(n.length<1||n.length>4||!Number.isInteger(i)||i<(r?2:1)||i>(r?4:1))throw Error(`Invalid local kriging parameter settings`);let o=t??new Float32Array(a);if(o.length<a)throw Error(`Local kriging parameter target must hold ${a} elements`);o.fill(0);let s=0;n.forEach(({covariance:e,loadings:t},r)=>{if(d(e),t.length!==i**2||!t.every(Number.isFinite))throw Error(`Invalid local kriging loading dimensions`);let a=4+r*4;o[a]=u(e),o[a+1]=e.nugget??0,o[a+2]=e.sill,o[a+3]=e.range;let c=i>1?0:1;if(i>1){let e=4+4*n.length+r*i**2;for(let n=0;n<i**2;n++){let r=_(t,i,n);o[e+n]=r,c=Math.max(c,Math.abs(r))}}s=Math.max(s,(e.sill+(e.nugget??0))*c)});let c=Math.fround(s);return o[0]=c===0?1:c,o}function _(e,t,n){let r=Math.floor(n/t),i=n%t,a=0;for(let n=0;n<t;n++)a+=e[r*t+n]*e[i*t+n];return a}function v(){return`fn covarianceCorrelation(modelCode: u32, scaled: f32) -> f32 {
  switch modelCode {
${[0,1,2,3,4].map(e=>{let{statements:t,expression:n}=p(e);return`    ${e===4?`default`:`case ${e}u`} { ${t} return ${n}; }`}).join(`
`)}
  }
}`}function y(e,t){let n=e*t*t,r=t>1;return`var<private> covarianceStructures: array<vec4<f32>, ${e}>;
${r?`var<private> crossCoefficients: array<f32, ${n}>;`:``}
${v()}
fn structureCovariance(structure: u32, first: vec2<f32>, second: vec2<f32>) -> f32 {
  let values = covarianceStructures[structure];
  let scaled = distance(first, second) / values.w;
  return values.z * covarianceCorrelation(u32(values.x), scaled) + select(0.0, values.y, all(first == second));
}
fn loadCovarianceParameters() -> bool {
  if (!(params[paramsOffset] > 0.0)) { return false; }
  for (var structure = 0u; structure < ${e}u; structure++) {
    let base = paramsOffset + 4u + structure * 4u;
    let values = vec4<f32>(params[base], params[base + 1u], params[base + 2u], params[base + 3u]);
    if (!(values.x >= 0.0 && values.x <= 4.0) || !(values.y >= 0.0) || !(values.z >= 0.0) || !(values.w > 0.0)) { return false; }
    covarianceStructures[structure] = values;
  }
${r?`  for (var coefficient = 0u; coefficient < ${n}u; coefficient++) { crossCoefficients[coefficient] = params[paramsOffset + ${4+4*e}u + coefficient]; }`:``}
  return true;
}
fn crossCovariance(first: vec2<f32>, second: vec2<f32>, firstVariable: u32, secondVariable: u32) -> f32 {
  ${r?`var total = 0.0;\n  for (var structure = 0u; structure < ${e}u; structure++) { total += structureCovariance(structure, first, second) * crossCoefficients[structure * ${t*t}u + firstVariable * ${t}u + secondVariable]; }\n  return total;`:`return structureCovariance(0u, first, second);`}
}`}var b={separable:0,"product-sum":1,metric:2,"sum-metric":3,gneiting:4};function x(e){d({model:e.model,nugget:e.nugget,sill:e.sill,range:e.range})}function S(e){return e.nugget+e.sill}function C(e,t){if(!Number.isFinite(e)||!(e>0))throw Error(`Space-time covariance ${t} must be positive and finite`)}function w(e){switch(e.type){case`separable`:return e.sill;case`product-sum`:{let t=S(e.spatial),n=S(e.temporal);return e.k*t*n+t+n}case`metric`:return S(e.joint);case`sum-metric`:return S(e.spatial)+S(e.temporal)+S(e.joint);case`gneiting`:return e.nugget+e.sill;default:throw Error(`Unknown space-time covariance`)}}function T(e,t,n){e[t]=u({model:n.model,range:n.range,sill:n.sill}),e[t+1]=n.nugget,e[t+2]=n.sill,e[t+3]=n.range}function E(e,t=new Float32Array(28)){if(t.length<28)throw Error(`Space-time kriging parameter target must hold 28 elements`);let{covariance:n}=e;t.fill(0),t[1]=b[n.type];let r;switch(n.type){case`separable`:if(x(n.spatial),x(n.temporal),Math.abs(S(n.spatial)-1)>1e-6||Math.abs(S(n.temporal)-1)>1e-6)throw Error(`Separable space-time structures must be standardized to a total of 1`);C(n.sill,`sill`),t[3]=n.sill,T(t,4,n.spatial),T(t,8,n.temporal);break;case`product-sum`:if(x(n.spatial),x(n.temporal),!Number.isFinite(n.k)||!(n.k>=0))throw Error(`Space-time covariance k must be non-negative and finite`);t[3]=n.k,T(t,4,n.spatial),T(t,8,n.temporal);break;case`metric`:x(n.joint),C(n.anisotropy,`anisotropy`),r=n.anisotropy,T(t,12,n.joint);break;case`sum-metric`:x(n.spatial),x(n.temporal),x(n.joint),C(n.anisotropy,`anisotropy`),r=n.anisotropy,T(t,4,n.spatial),T(t,8,n.temporal),T(t,12,n.joint);break;case`gneiting`:if(C(n.sill,`sill`),C(n.spatialRange,`spatialRange`),C(n.temporalRange,`temporalRange`),!(n.nugget>=0)||!(n.alpha>0&&n.alpha<=1)||!(n.gamma>0&&n.gamma<=1)||!(n.beta>=0&&n.beta<=1)||!(n.tau>=n.beta)||!Number.isFinite(n.tau))throw Error(`Invalid Gneiting covariance parameters`);t[16]=n.sill,t[17]=n.nugget,t[18]=a(n.temporalRange,n.alpha,n.tau),t[19]=i(n.spatialRange,n.gamma),t[20]=n.alpha,t[21]=n.gamma,t[22]=n.beta,t[23]=n.tau;break;default:throw Error(`Unknown space-time covariance`)}let o=e.searchAnisotropy??r;if(o===void 0)throw Error(`searchAnisotropy is required for a model without an anisotropy`);C(o,`searchAnisotropy`);let s=w(n);return C(s,`variance`),t[0]=s,t[2]=r??0,t[24]=e.searchRadius??1/0,t[25]=e.maximumTimeLag??1/0,t[26]=o,t[27]=e.mean??0,t}function D(){return`var<private> stHeader: vec4<f32>;
var<private> stStructures: array<vec4<f32>, 3>;
var<private> stGneiting: array<f32, 8>;
${v()}
fn stRead(offset: u32) -> f32 { return params[paramsOffset + offset]; }
fn stStructureCovariance(slot: u32, lag: f32, coincident: bool) -> f32 {
  let values = stStructures[slot];
  return values.z * covarianceCorrelation(u32(values.x), lag / values.w) + select(0.0, values.y, coincident);
}
fn loadSpaceTimeCovariance() -> bool {
  stHeader = vec4<f32>(stRead(0u), stRead(1u), stRead(2u), stRead(3u));
  if (!(stHeader.x > 0.0) || !(stHeader.y >= 0.0 && stHeader.y <= 4.0)) { return false; }
  for (var slot = 0u; slot < 3u; slot++) {
    let base = 4u + slot * 4u;
    stStructures[slot] = vec4<f32>(stRead(base), stRead(base + 1u), stRead(base + 2u), stRead(base + 3u));
  }
  for (var entry = 0u; entry < 8u; entry++) { stGneiting[entry] = stRead(16u + entry); }
  return true;
}
fn spaceTimeCovariance(spatialLag: f32, timeLag: f32, samePlace: bool, sameTime: bool) -> f32 {
  let typeCode = u32(stHeader.y);
  if (typeCode == 2u || typeCode == 3u) {
    let metricLag = sqrt(spatialLag * spatialLag + stHeader.z * stHeader.z * timeLag * timeLag);
    let joint = stStructureCovariance(2u, metricLag, samePlace && sameTime);
    if (typeCode == 2u) { return joint; }
    return stStructureCovariance(0u, spatialLag, samePlace) + stStructureCovariance(1u, timeLag, sameTime) + joint;
  }
  if (typeCode == 4u) {
    let sill = stGneiting[0];
    let nugget = stGneiting[1];
    let a = stGneiting[2];
    let c = stGneiting[3];
    let psi = a * select(pow(timeLag, 2.0 * stGneiting[4]), 0.0, timeLag <= 0.0) + 1.0;
    let spatialPower = select(pow(spatialLag, 2.0 * stGneiting[5]), 0.0, spatialLag <= 0.0);
    return sill * pow(psi, -stGneiting[7]) * exp(-c * spatialPower * pow(psi, -stGneiting[6] * stGneiting[5])) + select(0.0, nugget, samePlace && sameTime);
  }
  let spatial = stStructureCovariance(0u, spatialLag, samePlace);
  let temporal = stStructureCovariance(1u, timeLag, sameTime);
  if (typeCode == 0u) { return stHeader.w * spatial * temporal; }
  return stHeader.w * spatial * temporal + spatial + temporal;
}`}export{D as a,s as c,y as i,g as n,m as o,E as r,d as s,h as t};
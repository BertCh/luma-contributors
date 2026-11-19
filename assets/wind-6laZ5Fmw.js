import{h as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as n}from"./kernel-C3tfQOn4.js";import{D as r,E as i,S as a,T as o,f as s,g as c,v as l,x as u}from"./gpu-contributor-utils-BIwLQON5.js";import{v as d}from"./spatial-context-DwZEd4EZ.js";import{r as f}from"./wgsl-common-DfyXiYB0.js";import{a as p,t as m}from"./projection-shader-BNpLa6h4.js";import{t as h}from"./web-mercator-B0kaE2A1.js";import{o as g,t as _}from"./geospatial-utils-DOOj6ORH.js";import{l as v}from"./raster-band-utils-BcTxc8_Y.js";import{t as y}from"./vector-field-sampling-KzXxZj8z.js";import{t as ee}from"./wind-CwPNbNMt.js";function b(e){let t=e.inverse?e.radius*Math.PI:Math.PI,n=e.inverse?t:Math.atan(Math.sinh(Math.PI));return[-t,-n,t,n]}function x(e,t){if(e.arithmetic!==`float32`||!(e.radius>0)||!Number.isFinite(Math.fround(e.radius*Math.PI))||Math.fround(e.radius)===0)throw Error(`Web Mercator requires explicit float32 arithmetic and a finite positive radius`);return`{
    if (compare_fp64(value.xy, PROGRAM_scalar(${t+2}u)) < 0 ||
        compare_fp64(value.zw, PROGRAM_scalar(${t+4}u)) < 0 ||
        compare_fp64(value.xy, PROGRAM_scalar(${t+6}u)) > 0 ||
        compare_fp64(value.zw, PROGRAM_scalar(${t+8}u)) > 0) { return invalid; }
    let radius = PROGRAM_scalar(${t}u).x;
    let coordinates = vec2f(value.x + value.y, value.z + value.w);
    let ordinate = ${e.inverse?`coordinates.y / radius`:`coordinates.y`};
    var transformed: f32;
    if (abs(ordinate) < 0.01) {
      // Odd fifth-order expansion avoids cancellation in implementations of asinh/sinh near zero.
      // The first omitted term is below Float32 rounding throughout this interval.
      let squared = ordinate * ordinate;
      transformed = ordinate * (1.0 + squared * (${e.inverse?`-`:``}1.0 / 6.0 + squared / 24.0));
    } else {
      transformed = ${e.inverse?`atan(sinh(ordinate))`:`sign(ordinate) * asinh(tan(abs(ordinate)))`};
    }
    let projected = ${e.inverse?`vec2f(coordinates.x / radius, transformed)`:`vec2f(coordinates.x * radius, transformed * radius)`};
    value = vec4f(projected.x, 0.0, projected.y, 0.0);
  }`}var S=[[-2,2/3,4/3,-82/45,32/45,4642/4725],[5/3,-16/15,-13/9,904/315,-1522/945],[-26/15,34/21,8/5,-12686/2835],[1237/630,-12/5,-24832/14175],[-734/315,109598/31185],[444337/155925]],C=[[2,-2/3,-2,116/45,26/45,-2854/675],[7/3,-8/5,-227/45,2704/315,2323/945],[56/15,-136/35,-1262/105,73814/2835],[4279/630,-332/35,-399572/14175],[4174/315,-144838/6237],[601676/22275]],te=[[1/2,-2/3,5/16,41/180,-127/288,7891/37800],[13/48,-3/5,557/1440,281/630,-1983433/1935360],[61/240,-103/140,15061/26880,167603/181440],[49561/161280,-179/168,6601661/7257600],[34729/80640,-3418889/1995840],[212378941/319334400]],w=[[-1/2,2/3,-37/96,1/360,81/512,-96199/604800],[-1/48,-1/15,437/1440,-46/105,1118711/3870720],[-17/480,37/840,209/4480,-5569/90720],[-4397/161280,11/504,830251/7257600],[-4583/161280,108847/3991680],[-20648693/638668800]],T=[-Math.PI/15,-85*Math.PI/180,Math.PI/15,85*Math.PI/180];function E(e){let{semiMajorAxis:t,semiMinorAxis:n,scaleFactor:r,latitudeOrigin:i}=e;if(e.arithmetic!==`float32`||![t,n,r,i].every(Number.isFinite)||!(t>0&&n>=.99*t&&n<=t&&r>0)||Math.abs(i)>T[3])throw Error(`unsupported Transverse Mercator parameters`);let a=(1-n/t)/(1+n/t),o=a*a,s=t/(1+a)*(1+o/4+o**2/64+o**3/256)*r;if(!Number.isFinite(Math.fround(s*Math.PI))||Math.fround(s)<2**-126)throw Error(`Transverse Mercator scale must fit the normal Float32 range`);let c=O(S,a),l=O(te,a);return[s,s*k(k(i,c),l),...e.inverse?O(C,a):c,...e.inverse?O(w,a):l]}function D(e){if(!e.inverse)return[...T];let[t,n]=E(e);return[-.3*t,-Math.PI/2*t-n,.3*t,Math.PI/2*t-n]}function O(e,t){return e.map((e,n)=>e.reduceRight((e,n)=>e*t+n,0)*t**(n+1))}function k(e,t){return e+t.reduce((t,n,r)=>t+n*Math.sin(2*(r+1)*e),0)}var A=`
fn PROGRAM_transverseSin(value: f32) -> f32 {
  if (abs(value) < 0.01) {
    let squared = value * value;
    return value * (1.0 + squared * (-1.0 / 6.0 + squared / 120.0));
  }
  return sin(value);
}
fn PROGRAM_transverseTan(value: f32) -> f32 {
  if (abs(value) < 0.01) {
    let squared = value * value;
    return value * (1.0 + squared * (1.0 / 3.0 + 2.0 * squared / 15.0));
  }
  return tan(value);
}
fn PROGRAM_transverseAtan2(numerator: f32, denominator: f32) -> f32 {
  if (denominator > 0.0 && abs(numerator) < 0.01 * denominator) {
    let ratio = numerator / denominator;
    let squared = ratio * ratio;
    return ratio * (1.0 + squared * (-1.0 / 3.0 + squared / 5.0));
  }
  return atan2(numerator, denominator);
}
fn PROGRAM_transverseSinh(value: f32) -> f32 {
  if (abs(value) < 0.01) {
    let squared = value * value;
    return value * (1.0 + squared * (1.0 / 6.0 + squared / 120.0));
  }
  return sinh(value);
}
fn PROGRAM_transverseAsinh(value: f32) -> f32 {
  if (abs(value) < 0.01) {
    let squared = value * value;
    return value * (1.0 + squared * (-1.0 / 6.0 + 3.0 * squared / 40.0));
  }
  return sign(value) * asinh(abs(value));
}`;function j(e,t){return`{
    if (compare_fp64(value.xy, PROGRAM_scalar(${t+28}u)) < 0 ||
        compare_fp64(value.zw, PROGRAM_scalar(${t+30}u)) < 0 ||
        compare_fp64(value.xy, PROGRAM_scalar(${t+32}u)) > 0 ||
        compare_fp64(value.zw, PROGRAM_scalar(${t+34}u)) > 0) { return invalid; }
    let scale = PROGRAM_scalar(${t}u).x;
    let origin = PROGRAM_scalar(${t+2}u).x;
    let coordinates = vec2f(value.x + value.y, value.z + value.w);
    ${e.inverse?`let sphericalNorthing = (coordinates.y + origin) / scale;
    let sphericalEasting = coordinates.x / scale;`:`var conformalLatitude = coordinates.y;
    for (var index = 0u; index < 6u; index++) {
      conformalLatitude += PROGRAM_scalar(${t+4}u + 2u * index).x * PROGRAM_transverseSin(2.0 * f32(index + 1u) * coordinates.y);
    }
    let tangent = PROGRAM_transverseTan(conformalLatitude);
    let cosine = cos(coordinates.x);
    let sphericalNorthing = PROGRAM_transverseAtan2(tangent, cosine);
    let sphericalEasting = PROGRAM_transverseAsinh(PROGRAM_transverseSin(coordinates.x) / sqrt(tangent * tangent + cosine * cosine));`}
    var northing = sphericalNorthing;
    var easting = sphericalEasting;
    for (var index = 0u; index < 6u; index++) {
      let harmonic = 2.0 * f32(index + 1u);
      let coefficient = PROGRAM_scalar(${t+16}u + 2u * index).x;
      northing += coefficient * PROGRAM_transverseSin(harmonic * sphericalNorthing) * cosh(harmonic * sphericalEasting);
      easting += coefficient * cos(harmonic * sphericalNorthing) * PROGRAM_transverseSinh(harmonic * sphericalEasting);
    }
    ${e.inverse?`let hyperbolic = PROGRAM_transverseSinh(easting);
    let cosine = cos(northing);
    let longitude = PROGRAM_transverseAtan2(hyperbolic, cosine);
    let conformalLatitude = PROGRAM_transverseAtan2(PROGRAM_transverseSin(northing), sqrt(hyperbolic * hyperbolic + cosine * cosine));
    var latitude = conformalLatitude;
    for (var index = 0u; index < 6u; index++) {
      latitude += PROGRAM_scalar(${t+4}u + 2u * index).x * PROGRAM_transverseSin(2.0 * f32(index + 1u) * conformalLatitude);
    }
    value = vec4f(longitude, 0.0, latitude, 0.0);
    if (compare_fp64(value.xy, PROGRAM_scalar(${t+36}u)) < 0 ||
        compare_fp64(value.zw, PROGRAM_scalar(${t+38}u)) < 0 ||
        compare_fp64(value.xy, PROGRAM_scalar(${t+40}u)) > 0 ||
        compare_fp64(value.zw, PROGRAM_scalar(${t+42}u)) > 0) { return invalid; }`:`value = vec4f(scale * easting, 0.0, scale * northing - origin, 0.0);`}
  }`}var M=[-Math.PI/2,-80*Math.PI/180,Math.PI/2,80*Math.PI/180],N=2**-20;function P(e){let{semiMajorAxis:t,semiMinorAxis:n,latitudeOrigin:r,firstStandardParallel:i,secondStandardParallel:a}=e,o=e.type===`lambert-conformal-conic`?e.scaleFactor:1;if(e.arithmetic!==`float32`||![t,n,r,i,a,o].every(Number.isFinite)||!(t>0&&n>=.99*t&&n<=t&&o>0)||[r,i,a].some(e=>Math.abs(e)>M[3])||i!==a&&Math.abs(i-a)<1e-7)throw Error(`unsupported conic parameters`);let s=1-(n/t)**2,c=Math.cos(i)/Math.sqrt(1-s*Math.sin(i)**2),l=Math.cos(a)/Math.sqrt(1-s*Math.sin(a)**2),u=i===a?Math.sin(i):e.type===`lambert-conformal-conic`?Math.log(c/l)/(I(i,s)-I(a,s)):(c**2-l**2)/(L(a,s)-L(i,s));if(!Number.isFinite(u)||Math.abs(u)<.05||Math.abs(u)>=1)throw Error(`conic exponent is outside the bounded native branch`);let d=e.type===`lambert-conformal-conic`?c/(u*Math.exp(u*I(i,s))):c**2+u*L(i,s),f=t=>e.type===`lambert-conformal-conic`?d*Math.exp(u*I(t,s)):Math.sqrt(d-u*L(t,s))/u,p=t*o,m=p*f(r),h=[f(M[1])**2,f(M[3])**2],g=Math.max(...h),_=[p,s,u,d,m,Math.min(...h)*(1+N),g*(1-N),Math.tan(Math.abs(u)*M[2])*(1-N)];if(Math.fround(p)<2**-126||![..._,Math.abs(m)+p*Math.sqrt(g)].every(e=>Number.isFinite(Math.fround(e))))throw Error(`conic parameters must fit the normal Float32 range`);return _}function F(e){if(!e.inverse)return[...M];let[t,,,,n,,r]=P(e),i=t*Math.sqrt(r);return[-i,n-i,i,n+i]}function I(e,t){let n=Math.sqrt(t);return-Math.asinh(Math.tan(e))+n*Math.atanh(n*Math.sin(e))}function L(e,t){let n=Math.sin(e),r=Math.sqrt(t);return r?(1-t)*(n/(1-t*n*n)+Math.atanh(r*n)/r):2*n}var ne=`
fn PROGRAM_conicSin(value: f32) -> f32 {
  if (abs(value) < 0.01) {
    let squared = value * value;
    return value * (1.0 + squared * (-1.0 / 6.0 + squared / 120.0));
  }
  return sin(value);
}
fn PROGRAM_conicAtan2(numerator: f32, denominator: f32) -> f32 {
  if (denominator > 0.0 && abs(numerator) < 0.01 * denominator) {
    let ratio = numerator / denominator;
    let squared = ratio * ratio;
    return ratio * (1.0 + squared * (-1.0 / 3.0 + squared / 5.0));
  }
  return atan2(numerator, denominator);
}
fn PROGRAM_conicAtanhRatio(squared: f32) -> f32 {
  return 1.0 + squared * (1.0 / 3.0 + squared * (1.0 / 5.0 + squared * (1.0 / 7.0 + squared / 9.0)));
}
fn PROGRAM_conicLogT(latitude: f32, eccentricitySquared: f32) -> f32 {
  var spherical: f32;
  if (abs(latitude) < 0.01) {
    let squared = latitude * latitude;
    spherical = latitude * (1.0 + squared * (1.0 / 6.0 + squared / 24.0));
  } else {
    spherical = sign(latitude) * asinh(tan(abs(latitude)));
  }
  let sine = PROGRAM_conicSin(latitude);
  return -spherical + eccentricitySquared * sine * PROGRAM_conicAtanhRatio(eccentricitySquared * sine * sine);
}
fn PROGRAM_conicQ(latitude: f32, eccentricitySquared: f32) -> f32 {
  let sine = PROGRAM_conicSin(latitude);
  let squared = eccentricitySquared * sine * sine;
  return (1.0 - eccentricitySquared) * sine * (1.0 / (1.0 - squared) + PROGRAM_conicAtanhRatio(squared));
}`;function re(e,t){let n=e.type===`lambert-conformal-conic`;return`{
    if (compare_fp64(value.xy, PROGRAM_scalar(${t+16}u)) < 0 ||
        compare_fp64(value.zw, PROGRAM_scalar(${t+18}u)) < 0 ||
        compare_fp64(value.xy, PROGRAM_scalar(${t+20}u)) > 0 ||
        compare_fp64(value.zw, PROGRAM_scalar(${t+22}u)) > 0) { return invalid; }
    let scale = PROGRAM_scalar(${t}u).x;
    let eccentricitySquared = PROGRAM_scalar(${t+2}u).x;
    let exponent = PROGRAM_scalar(${t+4}u).x;
    let constant = PROGRAM_scalar(${t+6}u).x;
    ${e.inverse?`
    let direction = sign(exponent);
    let eastingPair = mul_fp64(div_fp64(value.xy, PROGRAM_scalar(${t}u)), vec2f(direction, 0.0));
    let northingPair = mul_fp64(div_fp64(sub_fp64(PROGRAM_scalar(${t+8}u), value.zw), PROGRAM_scalar(${t}u)), vec2f(direction, 0.0));
    let squaredPair = sum_fp64(mul_fp64(eastingPair, eastingPair), mul_fp64(northingPair, northingPair));
    let absoluteEasting = select(eastingPair, -eastingPair, compare_fp64(eastingPair, vec2f(0.0)) < 0);
    if (compare_fp64(squaredPair, PROGRAM_scalar(${t+10}u)) < 0 ||
        compare_fp64(squaredPair, PROGRAM_scalar(${t+12}u)) > 0 ||
        compare_fp64(northingPair, vec2f(0.0)) <= 0 ||
        compare_fp64(absoluteEasting, mul_fp64(northingPair, PROGRAM_scalar(${t+14}u))) > 0) { return invalid; }
    let easting = eastingPair.x + eastingPair.y;
    let northing = northingPair.x + northingPair.y;
    let radius = direction * sqrt(easting * easting + northing * northing);
    let longitude = PROGRAM_conicAtan2(easting, northing) / exponent;
    let inverseValue = ${n?`log(radius / constant) / exponent`:`(constant - exponent * exponent * radius * radius) / exponent`};
    var latitude = ${n?`1.5707963267948966 - 2.0 * atan(exp(inverseValue))`:`asin(clamp(inverseValue * 0.5, -1.0, 1.0))`};
    for (var iteration = 0u; iteration < 8u; iteration++) {
      let sine = PROGRAM_conicSin(latitude);
      let denominator = 1.0 - eccentricitySquared * sine * sine;
      latitude += ${n?`(PROGRAM_conicLogT(latitude, eccentricitySquared) - inverseValue) * cos(latitude) * denominator / (1.0 - eccentricitySquared)`:`(inverseValue - PROGRAM_conicQ(latitude, eccentricitySquared)) * denominator * denominator / (2.0 * (1.0 - eccentricitySquared) * cos(latitude))`};
    }
    value = vec4f(longitude, 0.0, latitude, 0.0);
    if (compare_fp64(value.xy, PROGRAM_scalar(${t+24}u)) < 0 ||
        compare_fp64(value.zw, PROGRAM_scalar(${t+26}u)) < 0 ||
        compare_fp64(value.xy, PROGRAM_scalar(${t+28}u)) > 0 ||
        compare_fp64(value.zw, PROGRAM_scalar(${t+30}u)) > 0) { return invalid; }
    `:`
    let coordinates = vec2f(value.x + value.y, value.z + value.w);
    let radius = ${n?`scale * constant * exp(exponent * PROGRAM_conicLogT(coordinates.y, eccentricitySquared))`:`scale * sqrt(constant - exponent * PROGRAM_conicQ(coordinates.y, eccentricitySquared)) / exponent`};
    let angle = exponent * coordinates.x;
    let northing = sub_fp64(PROGRAM_scalar(${t+8}u), vec2f(radius * cos(angle), 0.0));
    value = vec4f(radius * PROGRAM_conicSin(angle), 0.0, northing);
    `}
  }`}var R=1024;function z(e){let[t,n]=e.interval,r=n-t,i=e.seamTolerance===void 0?r*2**-32:e.seamTolerance;if(![t,n,r,i].every(Number.isFinite)||r<2**-80||r>2**100||Math.max(Math.abs(t),Math.abs(n))>R*r||i<r*2**-32||i>=r/2)throw Error(`unsupported longitude interval or seam tolerance`);return{minimum:t,maximum:n,period:r,seamTolerance:i,inputBounds:[t-R*r,-Number.MAX_VALUE,n+R*r,Number.MAX_VALUE]}}function B(e){return`{
    let minimum = PROGRAM_scalar(${e}u);
    let period = PROGRAM_scalar(${e+2}u);
    let seamTolerance = PROGRAM_scalar(${e+4}u);
    if (compare_fp64(value.xy, PROGRAM_scalar(${e+6}u)) < 0 ||
        compare_fp64(value.xy, PROGRAM_scalar(${e+10}u)) > 0) { return invalid; }
    let relative = sub_fp64(value.xy, minimum);
    let quotient = div_fp64(relative, period);
    // The Float32 integer estimate may be one turn off near a boundary; correct in double-single.
    let turns = floor(quotient.x);
    var reduced = sub_fp64(relative, mul_fp64(vec2f(turns, 0.0), period));
    if (compare_fp64(reduced, vec2f(0.0)) < 0) { reduced = sum_fp64(reduced, period); }
    if (compare_fp64(reduced, period) >= 0) { reduced = sub_fp64(reduced, period); }
    if (compare_fp64(reduced, seamTolerance) <= 0 ||
        compare_fp64(sub_fp64(period, reduced), seamTolerance) <= 0) { return invalid; }
    value = vec4f(sum_fp64(minimum, reduced), value.zw);
  }`}function V(e,t=`float32x2`){let n=0,r=!1,i=e.operations.map((e,t)=>{let i=1,a=null,o=!0;switch(e.type){case`longitude-wrap`:a=Object.freeze(z(e).inputBounds),i=null,o=!1,n!==0&&(n=null);break;case`axis`:break;case`unit`:i=Math.abs(e.inverse?1/e.factor:e.factor);break;case`affine`:i=Math.max(...e.scale.map(t=>Math.abs(e.inverse?1/t:t)));break;case`web-mercator`:case`transverse-mercator`:case`lambert-conformal-conic`:case`albers-equal-area`:i=null,a=Object.freeze(e.type===`web-mercator`?b(e):e.type===`transverse-mercator`?D(e):F(e)),n!==0&&(n=null);break;case`adaptive`:i=null,a=Object.freeze([...e.plan.bounds]),o=!!e.inversePlan,n=r?null:e.plan.doubleSingleMaxError,r=!0}return i!==null&&n!==null&&(n*=i),n!==null&&!Number.isFinite(n)&&(n=null),Object.freeze({index:t,operation:e.type,arithmetic:e.type===`web-mercator`||e.type===`transverse-mercator`||e.type===`lambert-conformal-conic`||e.type===`albers-equal-area`?`float32`:`double-single`,inputDimensions:2,outputDimensions:2,inputBounds:a,invertible:o,...e.type===`adaptive`?{patchDomain:Object.freeze({kind:e.plan.strictDomains?`patch-union`:`rectangle`,patchCount:e.plan.patches.length,routing:e.plan.routingIndex?`indexed`:`scan`})}:{},...e.type===`longitude-wrap`?{longitudeWrap:Object.freeze({interval:Object.freeze([e.interval[0],e.interval[1]]),seamTolerance:z(e).seamTolerance,seamValidity:`invalid`})}:{},errorAmplification:i,approximationError:H(n,r)})});return Object.freeze({inputDimensions:2,outputDimensions:2,inputFormat:t,inputEncoding:t===`uint32x4`?`binary64`:t===`float32x4`?`double-single`:`float32`,arithmetic:i.some(e=>e.arithmetic===`float32`)?`mixed`:`double-single`,outputPrecision:e.precision,outputFrame:e.precision===`local-f32`?`origin-relative`:`absolute`,validity:`upstream-and-finite-and-stage-domain`,invertible:i.every(e=>e.invertible),approximationError:H(n,r),stages:Object.freeze(i)})}function H(e,t){return Object.freeze({kind:e===null?`unknown`:t?`sampled-estimate`:`none`,maximum:e,guaranteed:!1})}var U=class{precision;inputFormat;destinationOrigin;metadata;parameterWords;shaderSource;constructor(e,t=`float32x2`){this.precision=e.precision,this.inputFormat=t,this.destinationOrigin=[...e.destinationOrigin??[0,0]];let n=K(e,t);this.parameterWords=n.parameters,this.shaderSource=n.source,this.metadata=V(e,t)}packParameters(){return this.parameterWords.slice()}getShader(t={}){let n=t.namespace??`projection`,r=t.parameterOffset??0;if(!/^[A-Za-z][A-Za-z0-9_]*$/.test(n)||n.includes(`__`))throw Error(`projection namespace must be a simple identifier`);if(!Number.isSafeInteger(r)||r<0||r+this.parameterWords.length>4294967296)throw Error(`projection parameter offset must be a uint32 word offset`);let i=`projection_${n}`;return{source:this.shaderSource.replace(/PROGRAM|PARAMETER_OFFSET/g,e=>e===`PROGRAM`?i:`${r}u`),modules:[t.fp64Arithmetic===`classic`?e:_],defines:t.fp64Arithmetic===`classic`?g(`unknown`):{LUMA_FP64_INTEGER_ARITHMETIC:!0},entryPoint:`${i}_project`,bindingName:`${i}_parameters`,inputType:this.inputFormat===`uint32x4`?`vec4u`:this.inputFormat===`float32x4`?`vec4f`:`vec2f`,outputType:this.precision===`double-single`?`vec4f`:`vec2f`}}isCompatible(e){return this.shaderSource===e.shaderSource&&this.parameterWords.length===e.parameterWords.length}};function W(e,t={}){return new U(e,t.inputFormat)}function G(e,t={}){return{precision:e.precision,destinationOrigin:t.destinationOrigin??[0,0],operations:[...e.operations].reverse().map(e=>{switch(e.type){case`longitude-wrap`:throw Error(`longitude normalization discards turns and has no automatic inverse`);case`axis`:return{...e,order:[...e.order]};case`unit`:return{...e,inverse:!e.inverse};case`affine`:case`web-mercator`:case`transverse-mercator`:case`lambert-conformal-conic`:case`albers-equal-area`:return{...e,inverse:!e.inverse};case`adaptive`:if(!e.inversePlan)throw Error(`adaptive inversion requires an inverse projection plan`);return{type:`adaptive`,plan:e.inversePlan,inversePlan:e.plan}}})}}function K(e,t){if(![`local-f32`,`double-single`].includes(e.precision)||![`float32x2`,`float32x4`,`uint32x4`].includes(t))throw Error(`unsupported projection precision or input format`);let n=[],r=e=>{let t=Math.fround(e),r=Math.fround(e-t);if(!Number.isFinite(e)||!Number.isFinite(t)||e!==0&&t===0)throw Error(`projection parameters must fit the double-single exponent range`);n.push(...new Uint32Array(Float32Array.of(t,r).buffer))},i=e=>{for(let[t,r]of e.entries()){let e=Math.fround(r),i=Math.fround(r-e),a=e+i;(t<2&&a<r||t>=2&&a>r)&&(i=q(i,t<2)),n.push(...new Uint32Array(Float32Array.of(e,i).buffer))}};for(let t of e.destinationOrigin??[0,0])r(t);let a=[],o=[];for(let[s,c]of e.operations.entries()){let e=n.length,l=e=>`PROGRAM_scalar(${e}u)`;switch(c.type){case`longitude-wrap`:{let{minimum:t,period:n,seamTolerance:a,inputBounds:s}=z(c);for(let e of[t,n,a])r(e);i([s[0],0,s[2],0]),o.push(B(e));break}case`axis`:if(!(c.order[0]===0&&c.order[1]===1||c.order[0]===1&&c.order[1]===0))throw Error(`projection axes must be a permutation`);o.push(`value = ${c.order[0]===0?`value`:`value.zwxy`};`);break;case`unit`:case`affine`:{let t=c.type===`unit`?[c.factor,c.factor]:c.scale,n=c.type===`unit`?[0,0]:c.offset;if(t.some(e=>e===0))throw Error(`projection scale must be invertible`);for(let e of[...t,...n])r(e);let i=(t,n)=>c.inverse?`div_fp64(sub_fp64(${t}, ${l(e+4+n*2)}), ${l(e+n*2)})`:`sum_fp64(mul_fp64(${t}, ${l(e+n*2)}), ${l(e+4+n*2)})`;o.push(`value = vec4f(${i(`value.xy`,0)}, ${i(`value.zw`,1)});`);break}case`web-mercator`:o.push(x(c,e)),r(c.radius),i(b(c));break;case`transverse-mercator`:for(let e of E(c))r(e);i(D(c)),i(T),a.includes(`
fn PROGRAM_transverseSin(value: f32) -> f32 {
  if (abs(value) < 0.01) {
    let squared = value * value;
    return value * (1.0 + squared * (-1.0 / 6.0 + squared / 120.0));
  }
  return sin(value);
}
fn PROGRAM_transverseTan(value: f32) -> f32 {
  if (abs(value) < 0.01) {
    let squared = value * value;
    return value * (1.0 + squared * (1.0 / 3.0 + 2.0 * squared / 15.0));
  }
  return tan(value);
}
fn PROGRAM_transverseAtan2(numerator: f32, denominator: f32) -> f32 {
  if (denominator > 0.0 && abs(numerator) < 0.01 * denominator) {
    let ratio = numerator / denominator;
    let squared = ratio * ratio;
    return ratio * (1.0 + squared * (-1.0 / 3.0 + squared / 5.0));
  }
  return atan2(numerator, denominator);
}
fn PROGRAM_transverseSinh(value: f32) -> f32 {
  if (abs(value) < 0.01) {
    let squared = value * value;
    return value * (1.0 + squared * (1.0 / 6.0 + squared / 120.0));
  }
  return sinh(value);
}
fn PROGRAM_transverseAsinh(value: f32) -> f32 {
  if (abs(value) < 0.01) {
    let squared = value * value;
    return value * (1.0 + squared * (-1.0 / 6.0 + 3.0 * squared / 40.0));
  }
  return sign(value) * asinh(abs(value));
}`)||a.push(A),o.push(j(c,e));break;case`lambert-conformal-conic`:case`albers-equal-area`:for(let e of P(c))r(e);i(F(c)),i(M),a.includes(`
fn PROGRAM_conicSin(value: f32) -> f32 {
  if (abs(value) < 0.01) {
    let squared = value * value;
    return value * (1.0 + squared * (-1.0 / 6.0 + squared / 120.0));
  }
  return sin(value);
}
fn PROGRAM_conicAtan2(numerator: f32, denominator: f32) -> f32 {
  if (denominator > 0.0 && abs(numerator) < 0.01 * denominator) {
    let ratio = numerator / denominator;
    let squared = ratio * ratio;
    return ratio * (1.0 + squared * (-1.0 / 3.0 + squared / 5.0));
  }
  return atan2(numerator, denominator);
}
fn PROGRAM_conicAtanhRatio(squared: f32) -> f32 {
  return 1.0 + squared * (1.0 / 3.0 + squared * (1.0 / 5.0 + squared * (1.0 / 7.0 + squared / 9.0)));
}
fn PROGRAM_conicLogT(latitude: f32, eccentricitySquared: f32) -> f32 {
  var spherical: f32;
  if (abs(latitude) < 0.01) {
    let squared = latitude * latitude;
    spherical = latitude * (1.0 + squared * (1.0 / 6.0 + squared / 24.0));
  } else {
    spherical = sign(latitude) * asinh(tan(abs(latitude)));
  }
  let sine = PROGRAM_conicSin(latitude);
  return -spherical + eccentricitySquared * sine * PROGRAM_conicAtanhRatio(eccentricitySquared * sine * sine);
}
fn PROGRAM_conicQ(latitude: f32, eccentricitySquared: f32) -> f32 {
  let sine = PROGRAM_conicSin(latitude);
  let squared = eccentricitySquared * sine * sine;
  return (1.0 - eccentricitySquared) * sine * (1.0 / (1.0 - squared) + PROGRAM_conicAtanhRatio(squared));
}`)||a.push(ne),o.push(re(c,e));break;case`adaptive`:{if(c.plan.precision!==`double-single`||c.plan.patches.length===0)throw Error(`program adaptive stages require double-single projection plans`);for(let e of p(c.plan))n.push(e);let r=n.length;i(c.plan.bounds);let l=s===0&&t===`uint32x4`,u=m({precise:l,doubleSingle:!0,doubleSingleInput:!l,patchCount:c.plan.patches.length,routingNodeCount:c.plan.routingIndex?.length,strictDomains:c.plan.strictDomains,planOffset:e,inputBoundsOffset:r}),d=[...u.matchAll(/\b(?:fn|struct|const)\s+([A-Za-z_][A-Za-z0-9_]*)/g)].map(e=>e[1]),f=new Set(d);u=u.replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/g,e=>f.has(e)?`PROGRAM_stage${s}_${e}`:e),u=u.replace(`@group(0) @binding(auto) var<storage, read> projectionPlans: array<u32>;`,``),u=u.replace(/projectionPlans\[([^\]]+)\]/g,`PROGRAM_parameters[PARAMETER_OFFSET + $1]`),a.push(u);let h=l?`PROGRAM_stage${s}_makeRawPoint(position.x, position.y, position.z, position.w)`:`value`;o.push(`let stage${s} = PROGRAM_stage${s}_project(${h}, 0xffffffffu);
  if (stage${s}.valid == 0u) { return invalid; }
  value = stage${s}.position;`);break}default:throw Error(`unsupported projection operation`)}o.push(`if (!PROGRAM_finite(value)) { return invalid; }`)}let s=t===`uint32x4`?`vec4u`:t===`float32x4`?`vec4f`:`vec2f`,c=e.precision===`double-single`?`vec4f`:`vec2f`,l=t===`uint32x4`?`vec4f(sub_fp64u32_to_fp64(position.yx, vec2u(0u)), sub_fp64u32_to_fp64(position.wz, vec2u(0u)))`:t===`float32x4`?`vec4f(normalize_fp64(position.xy), normalize_fp64(position.zw))`:`vec4f(position.x, 0.0, position.y, 0.0)`,u=t===`uint32x4`&&e.operations[0]?.type===`adaptive`,d=n.length===4&&e.precision===`double-single`?`if (arrayLength(&PROGRAM_parameters) <= PARAMETER_OFFSET + 3u) { return invalid; }`:``;return{source:`
@group(0) @binding(auto) var<storage, read> PROGRAM_parameters: array<u32>;
struct PROGRAM_Result { position: ${c}, valid: u32 }
fn PROGRAM_scalar(offset: u32) -> vec2f {
  return vec2f(bitcast<f32>(PROGRAM_parameters[PARAMETER_OFFSET + offset]), bitcast<f32>(PROGRAM_parameters[PARAMETER_OFFSET + offset + 1u]));
}
fn PROGRAM_finite(value: vec4f) -> bool { return is_finite_fp64(value.xy) && is_finite_fp64(value.zw); }
${a.join(`
`)}
fn PROGRAM_project(position: ${s}, inputValidity: u32) -> PROGRAM_Result {
  let invalid = PROGRAM_Result(${c}(0.0), 0u);
  if (inputValidity == 0u) { return invalid; }
  ${d}
  var value = ${u?`vec4f(0.0)`:l};
  if (!PROGRAM_finite(value)) { return invalid; }
  ${o.join(`
  `)}
  ${e.precision===`double-single`?`return PROGRAM_Result(value, 1u);`:`let localX = sub_fp64(value.xy, PROGRAM_scalar(0u));
  let localY = sub_fp64(value.zw, PROGRAM_scalar(2u));
  let local = vec2f(localX.x + localX.y, localY.x + localY.y);
  if (!PROGRAM_finite(vec4f(local.x, 0.0, local.y, 0.0))) { return invalid; }
  return PROGRAM_Result(local, 1u);`}
}`,parameters:Uint32Array.from(n)}}function q(e,t){if(e===0)return t?2**-149:-(2**-149);let n=new Uint32Array(Float32Array.of(e).buffer);return n[0]+=e>0===t?1:-1,new Float32Array(n.buffer)[0]}var J={precision:`local-f32`,operations:[{type:`unit`,factor:Math.PI/180},{type:`web-mercator`,arithmetic:`float32`,radius:h}]};function Y(e,t){return d(e,t,{coordinateSpaces:[`planar`,`longitude-latitude`],metricsByCoordinateSpace:{planar:[`none`,`native`],"longitude-latitude":[`none`]}})}function X(r,i){let a=W(J),o=W(G(J)),s=a.packParameters(),c=o.packParameters(),l=new Uint32Array(s.length+c.length);l.set(s),l.set(c,s.length);let u=a.getShader({namespace:`flowForward`}),d=o.getShader({namespace:`flowInverse`,parameterOffset:s.length}),f=e=>e.source.replace(`@group(0) @binding(auto) var<storage, read> ${e.bindingName}: array<u32>;`,``).replaceAll(e.bindingName,`projectionParameters`).replaceAll(`projectionParameters[`,`projectionParameters[projectionParametersOffset + `),p=n(r,`${i}-projection-parameters`,`uint32`,l.length);return{binding:{name:`projectionParameters`,view:p,type:`u32`,access:`read`},modules:[e],defines:u.defines,declarations:`${f(u)}\n${f(d)}`,forward:u.entryPoint,inverse:d.entryPoint,nodes:[t(r,{id:`${i}-projection-parameters-initialize`,operation:`LagrangianGeographicProjection`,invocationCount:l.length,bindings:[{name:`projectionParameters`,view:p,type:`u32`,access:`read_write`}],declarations:`const WORDS = array<u32, ${l.length}>(${Array.from(l,e=>`${e}u`).join(`, `)});`,body:`projectionParameters[projectionParametersOffset + index] = WORDS[index];`})]}}var Z=`GPULagrangianFlowMap`,Q=65535,$=8192;function ie(e,t=new Float32Array(8)){if(t.length<8)throw Error(`${Z} parameter target must hold 8 elements`);let[n,r,i,a]=e.fieldExtent;for(let[e,t]of[[`cellWidth`,i],[`cellHeight`,a]])c(Z,e,t,Number.MIN_VALUE);for(let[t,i]of[[`originX`,n],[`originY`,r],[`startTime`,e.startTime],[`timeStep`,e.timeStep],[`sliceStartTime`,e.sliceStartTime],[`sliceEndTime`,e.sliceEndTime]])c(Z,t,i);return t.set([n,r,i,a,e.startTime,e.timeStep,e.sliceStartTime,e.sliceEndTime]),t}function ae(e,t,n){return y(e,t,n).replace(/\b(FIELD_WIDTH|FIELD_HEIGHT|isNanFloat|readFieldVelocity|isNanVelocity|sampleField)\b/g,`$1End`)}var oe=class{id;seedCount;props;constructor(e){let t=e.id??`lagrangian-flow-map`;Y(t,e.spatialContext);let{fieldWidth:n,fieldHeight:c,stepCount:l,output:u}=e;s(t,`fieldWidth`,n,1,Q),s(t,`fieldHeight`,c,1,Q),s(t,`stepCount`,l),a(t,`stepCount`,l,$),i(e.seeds,[`float32x2`],`${t} seeds`),i(e.velocitiesStart,[`float32x2`],`${t} velocitiesStart`),i(e.velocitiesEnd,[`float32x2`],`${t} velocitiesEnd`),r(t,`velocitiesStart`,e.velocitiesStart,n*c),r(t,`velocitiesEnd`,e.velocitiesEnd,n*c),i(e.parameters,[`float32`],`${t} parameters`),o(t,e.parameters.length>=8,`parameters must contain 8 float32 rows`),this.seedCount=e.seeds.length,s(t,`seeds`,this.seedCount),i(u.positions,[`float32x2`],`${t} output.positions`),r(t,`output.positions`,u.positions,this.seedCount),u.validity&&(i(u.validity,[`uint32`],`${t} output.validity`),r(t,`output.validity`,u.validity,this.seedCount)),e.regions&&(i(e.regions,[`uint32`],`${t} regions`),r(t,`regions`,e.regions,n*c)),o(t,!(u.regionClasses||u.residenceTimes)||e.regions!==void 0,`output.regionClasses and output.residenceTimes require regions`),u.regionClasses&&(i(u.regionClasses,[`uint32`],`${t} output.regionClasses`),r(t,`output.regionClasses`,u.regionClasses,2*this.seedCount)),u.residenceTimes&&(i(u.residenceTimes,[`float32`],`${t} output.residenceTimes`),r(t,`output.residenceTimes`,u.residenceTimes,this.seedCount)),v(t,[u.positions,u.regionClasses,u.residenceTimes,u.validity],[e.seeds,e.velocitiesStart,e.velocitiesEnd,e.parameters,e.regions]),this.id=t,this.props=e}getCommandNodes(e){let{id:r,props:i,seedCount:a}=this,{output:o,regions:s,fieldWidth:c,fieldHeight:d,stepCount:p}=i;u(r,e,[i.seeds,i.velocitiesStart,i.velocitiesEnd,i.parameters,s,o.positions,o.validity,o.regionClasses,o.residenceTimes]);let m=s!==void 0;l(e.device,r,`seeds`,a*8,`release fewer particles per graph`),l(e.device,r,`velocitiesStart/velocitiesEnd`,c*d*8,`use a smaller field`);let h=i.spatialContext?.coordinateSpace===`longitude-latitude`,g=!!(o.regionClasses||o.residenceTimes||o.validity);g&&l(e.device,r,`diagnostics`,a*16,`release fewer particles per graph`);let _=h?X(e,r):void 0,v=g?n(e,`${r}-diagnostics`,`uint32x4`,a):void 0;return[..._?.nodes??[],t(e,{id:`${r}-integrate`,operation:Z,variant:m?`regions`:`positions`,invocationCount:a,bindings:[{name:`seeds`,view:i.seeds,type:`f32`,access:`read`},{name:`velocitiesStart`,view:i.velocitiesStart,type:`f32`,access:`read`},{name:`velocitiesEnd`,view:i.velocitiesEnd,type:`f32`,access:`read`},{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`positions`,view:o.positions,type:`f32`,access:`read_write`},...m?[{name:`regions`,view:s,type:`u32`,access:`read`}]:[],...v?[{name:`diagnostics`,view:v,type:`u32`,access:`read_write`}]:[],..._?[_.binding]:[]],modules:_?.modules,defines:_?.defines,declarations:`
${_?.declarations??``}
${y(`velocitiesStart`,c,d)}
${ae(`velocitiesEnd`,c,d)}
${f}
const STEP_COUNT: u32 = ${p}u;
const NO_REGION: u32 = 0xffffffffu;

fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }

// Time-interpolated velocity: (u, v, valid).
fn sampleVelocity(position: vec2<f32>, extent: vec4<f32>, weight: f32) -> vec3<f32> {
  ${_?`let geographicPosition = ${_.inverse}(position, 1u);
  if (geographicPosition.valid == 0u) { return vec3<f32>(0.0); }
  let samplePosition = geographicPosition.position;`:`let samplePosition = position;`}
  let first = sampleField(samplePosition, extent);
  let second = sampleFieldEnd(samplePosition, extent);
  if (first.z == 0.0 || second.z == 0.0) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(mix(first.xy, second.xy, weight), 1.0);
}

fn isInsideField(position: vec2<f32>, extent: vec4<f32>) -> bool {
  ${_?`let geographicPosition = ${_.inverse}(position, 1u);
  if (geographicPosition.valid == 0u) { return false; }
  let samplePosition = geographicPosition.position;`:`let samplePosition = position;`}
  let local = (samplePosition - extent.xy) / extent.zw;
  return !isNanVelocity(local) && local.x >= 0.0 && local.y >= 0.0 &&
    local.x < f32(FIELD_WIDTH) && local.y < f32(FIELD_HEIGHT);
}

${m?`fn getRegion(position: vec2<f32>, extent: vec4<f32>) -> u32 {
  if (!isInsideField(position, extent)) {
    return NO_REGION;
  }
  ${_?`let samplePosition = ${_.inverse}(position, 1u).position;`:`let samplePosition = position;`}
  let local = (samplePosition - extent.xy) / extent.zw;
  let column = min(u32(local.x), FIELD_WIDTH - 1u);
  let row = min(u32(local.y), FIELD_HEIGHT - 1u);
  return regions[regionsOffset + row * FIELD_WIDTH + column];
}`:``}`,body:`
  let extent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let startTime = readParameter(4u);
  let timeStep = readParameter(5u);
  let sliceStart = readParameter(6u);
  let sliceEnd = readParameter(7u);
  let inverseSpan = select(0.0, 1.0 / (sliceEnd - sliceStart), sliceEnd != sliceStart);
  var position = vec2<f32>(seeds[seedsOffset + 2u * index], seeds[seedsOffset + 2u * index + 1u]);
  ${_?`let projectedSeed = ${_.forward}(position, 1u);
  position = projectedSeed.position;
  var lost = projectedSeed.valid == 0u || !isInsideField(position, extent);`:`var lost = !isInsideField(position, extent);`}
${m?`  let startRegion = select(getRegion(position, extent), NO_REGION, lost);
  var exited = startRegion == NO_REGION;
  var insideSteps = 0u;`:``}
  for (var step = 0u; step < STEP_COUNT; step = step + 1u) {
    if (lost) {
      break;
    }
${m?`    if (!exited) {
      if (getRegion(position, extent) == startRegion) {
        insideSteps = insideSteps + 1u;
      } else {
        exited = true;
      }
    }`:``}
    let time = startTime + f32(step) * timeStep;
    let weightStart = clamp((time - sliceStart) * inverseSpan, 0.0, 1.0);
    let weightMiddle = clamp((time + 0.5 * timeStep - sliceStart) * inverseSpan, 0.0, 1.0);
    let first = sampleVelocity(position, extent, weightStart);
    if (first.z == 0.0) {
      lost = true;
      break;
    }
    let second = sampleVelocity(position + 0.5 * timeStep * first.xy, extent, weightMiddle);
    if (second.z == 0.0) {
      lost = true;
      break;
    }
    position = position + timeStep * second.xy;
    if (!isInsideField(position, extent)) {
      lost = true;
    }
  }
  ${_?`let finalPosition = ${_.inverse}(position, select(1u, 0u, lost));
  lost = lost || finalPosition.valid == 0u;
  let endPosition = position;
  position = finalPosition.position;`:`let endPosition = position;`}
  if (lost) {
    position = vec2<f32>(getNaN());
  }
  positions[positionsOffset + 2u * index] = position.x;
  positions[positionsOffset + 2u * index + 1u] = position.y;
${v?`
  diagnostics[diagnosticsOffset + 4u * index] = ${m?`startRegion`:`NO_REGION`};
  diagnostics[diagnosticsOffset + 4u * index + 1u] = ${m?`select(getRegion(endPosition, extent), NO_REGION, lost)`:`NO_REGION`};
  diagnostics[diagnosticsOffset + 4u * index + 2u] = bitcast<u32>(${m?`f32(insideSteps) * abs(timeStep)`:`0.0`});
  diagnostics[diagnosticsOffset + 4u * index + 3u] = select(1u, 0u, lost);`:``}`}),...v?[t(e,{id:`${r}-publish-diagnostics`,operation:Z,invocationCount:a,bindings:[{name:`diagnostics`,view:v,type:`u32`,access:`read`},...o.regionClasses?[{name:`regionClasses`,view:o.regionClasses,type:`u32`,access:`read_write`}]:[],...o.residenceTimes?[{name:`residenceTimes`,view:o.residenceTimes,type:`f32`,access:`read_write`}]:[],...o.validity?[{name:`validity`,view:o.validity,type:`u32`,access:`read_write`}]:[]],body:`
${o.regionClasses?`regionClasses[regionClassesOffset + index] = diagnostics[diagnosticsOffset + 4u * index];
regionClasses[regionClassesOffset + ${a}u + index] = diagnostics[diagnosticsOffset + 4u * index + 1u];`:``}
${o.residenceTimes?`residenceTimes[residenceTimesOffset + index] = bitcast<f32>(diagnostics[diagnosticsOffset + 4u * index + 2u]);`:``}
${o.validity?`validity[validityOffset + index] = diagnostics[diagnosticsOffset + 4u * index + 3u];`:``}`})]:[]]}};function se(e,t=40){let n=Math.max(24,Math.round(t*e)),r=1/t,{east:i,north:a}=ee(n,t),o=new Float32Array(n*t*2),s=new Float32Array(n*t),c=1/Math.cos(28*Math.PI/180);for(let e=0;e<t;e++)for(let r=0;r<n;r++){let l=e*n+r,u=(t-e-1)*n+r;o[2*u]=i[l]/100*c,o[2*u+1]=a[l]/100,s[l]=Math.hypot(i[l],a[l])}return{columns:n,rows:t,cellSize:r,velocities:o,speeds:s}}export{Y as a,X as i,oe as n,ie as r,se as t};
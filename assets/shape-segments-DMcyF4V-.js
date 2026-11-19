import{c as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as n}from"./gpu-command-graph-BdzdmV6u.js";import{t as r}from"./gpu-dispatch-utils-CbRzFAgT.js";import{d as i,o as a,t as o}from"./kernel-C3tfQOn4.js";import{D as s,T as c,c as l,f as u,g as d,p as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{s as m}from"./wgsl-common-DfyXiYB0.js";import{t as h}from"./geodesic-wgsl-I0CYDKcz.js";import{a as g,n as _,r as v,t as y}from"./geodesic-kernels-DpMzdskM.js";import{g as b,p as x,v as S}from"./spatial-context-DwZEd4EZ.js";import{i as C,n as w,o as T,r as E}from"./graph-chunk-dispatch-zBt8k2-o.js";var D=`
${m}
const RHUMB_MAX_LATITUDE: f32 = 89.9999;

struct RhumbDestination {
  destination: vec2<f32>,
  finalBearing: f32,
}

fn rhumbClampLatitude(latitudeDegrees: f32) -> f32 {
  return clamp(latitudeDegrees, -RHUMB_MAX_LATITUDE, RHUMB_MAX_LATITUDE);
}

// cos of the mean latitude. Within one hemisphere the mean colatitude is formed from the exact
// colatitudes, so near a pole it keeps full relative precision (a + b would round to ~1e-5 degree).
fn rhumbMeanCosLatitude(a: f32, b: f32) -> f32 {
  if (a * b >= 0.0) {
    // colatitude(a) - (|b| - |a|) / 2, with the exact difference so nothing cancels near the pole.
    return sin(((90.0 - abs(a)) - 0.5 * (abs(b) - abs(a))) * GEODESIC_DEGREES_TO_RADIANS);
  }
  return cos(0.5 * (a + b) * GEODESIC_DEGREES_TO_RADIANS);
}

// Short edge: dPsi = dPhi / cos(mean latitude) to relative 1e-8, which keeps meter-scale edges
// exact where atanh of a ~1e-7 argument has no relative precision.
fn rhumbIsShortEdge(a: f32, b: f32) -> bool {
  let deltaPhi = (b - a) * GEODESIC_DEGREES_TO_RADIANS;
  return abs(deltaPhi) < 1e-4 * rhumbMeanCosLatitude(a, b);
}

// Stretched (isometric) latitude difference psi(b) - psi(a), with a and b in degrees.
fn rhumbStretchedDelta(a: f32, b: f32) -> f32 {
  if (rhumbIsShortEdge(a, b)) {
    return (b - a) * GEODESIC_DEGREES_TO_RADIANS / rhumbMeanCosLatitude(a, b);
  }
  let halfDelta = 0.5 * (b - a) * GEODESIC_DEGREES_TO_RADIANS;
  let sinHalfDelta = sin(halfDelta);
  let denominator = 2.0 * sinHalfDelta * sinHalfDelta +
    geodesicCosLatitude(a) * geodesicCosLatitude(b);
  // sin(b) - sin(a) = 2 cos(mean) sin(delta / 2), with the mean cosine from the exact colatitude.
  let sinDelta = 2.0 * rhumbMeanCosLatitude(a, b) * sinHalfDelta;
  let argument = clamp(sinDelta / denominator, -0.999999, 0.999999);
  // Native atanh can lower to log((1+x)/(1-x))/2, losing relative precision near zero.
  // For |x| < 1/8, the odd series through x^7 has relative remainder bounded by
  // x^8 / (9 * (1 - x^2)) < 6.8e-9, below one f32 rounding error.
  if (abs(argument) < 0.125) {
    let squaredArgument = argument * argument;
    return argument * (1.0 + squaredArgument * (1.0 / 3.0 +
      squaredArgument * (1.0 / 5.0 + squaredArgument / 7.0)));
  }
  return atanh(argument);
}

// dPhi / dPsi, with the east-west limit cos(latitude) when the latitudes (nearly) coincide.
fn rhumbStretchRatio(a: f32, b: f32) -> f32 {
  if (rhumbIsShortEdge(a, b)) {
    return rhumbMeanCosLatitude(a, b);
  }
  return (b - a) * GEODESIC_DEGREES_TO_RADIANS / rhumbStretchedDelta(a, b);
}

// Rhumb central angle (distance / radius).
fn rhumbAngle(origin: vec2<f32>, targetPosition: vec2<f32>) -> f32 {
  let a = rhumbClampLatitude(origin.y);
  let b = rhumbClampLatitude(targetPosition.y);
  let deltaPhi = (b - a) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicLongitudeDelta(origin.x, targetPosition.x) * GEODESIC_DEGREES_TO_RADIANS;
  let q = rhumbStretchRatio(a, b);
  return sqrt(deltaPhi * deltaPhi + q * q * deltaLambda * deltaLambda);
}

// Constant bearing in degrees clockwise from north, (-180, 180].
fn rhumbBearingDegrees(origin: vec2<f32>, targetPosition: vec2<f32>) -> f32 {
  let a = rhumbClampLatitude(origin.y);
  let b = rhumbClampLatitude(targetPosition.y);
  let deltaLambda = geodesicLongitudeDelta(origin.x, targetPosition.x) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaPsi = rhumbStretchedDelta(a, b);
  return guardedAtan2(deltaLambda, deltaPsi) * GEODESIC_RADIANS_TO_DEGREES;
}

// atanh with the odd series through x^7 below 1/8, where a native atanh that lowers to
// log((1 + x) / (1 - x)) / 2 loses relative precision. atanh(0) is exactly 0.
fn rhumbAtanh(argument: f32) -> f32 {
  if (abs(argument) < 0.125) {
    let squaredArgument = argument * argument;
    return argument * (1.0 + squaredArgument * (1.0 / 3.0 +
      squaredArgument * (1.0 / 5.0 + squaredArgument / 7.0)));
  }
  return atanh(argument);
}

// Isometric (Mercator) latitude psi = atanh(sin(latitude)) of a non-negative latitude given with
// its colatitude, both in degrees. Below 45 degrees from the sine; above, from the colatitude as
// -ln(tan(colatitude / 2)), because 1 - sin(latitude) has no f32 precision left near a pole
// while 90 - latitude is exact. Absolute error about 1e-7 times max(1, psi).
fn rhumbIsometricLatitude(absoluteLatitude: f32, colatitude: f32) -> f32 {
  if (absoluteLatitude < 45.0) {
    return rhumbAtanh(sin(absoluteLatitude * GEODESIC_DEGREES_TO_RADIANS));
  }
  return -log(tan(0.5 * colatitude * GEODESIC_DEGREES_TO_RADIANS));
}

// psi((a + b) / 2) - psi(a): the stretched-latitude difference from a to the mean latitude,
// with a and b in degrees. The mean latitude is never rounded to f32: its cosine and colatitude
// come from the exact colatitudes of a and b.
// - Short and moderate spans (atanh argument below 1/2) use the difference form of
//   rhumbStretchedDelta on a quarter of the latitude difference, which keeps relative precision
//   down to two latitudes one f32 step apart.
// - Long spans use psi(mean) - psi(a) from rhumbIsometricLatitude. The difference is then at least
//   atanh(1/2) = 0.55, so the absolute error of the two terms is negligible, and nothing is
//   clamped: an atanh argument within 1e-6 of 1 (a line that reaches latitude 89 and beyond) has
//   only a few f32 steps of 1 - x left.
fn rhumbStretchedHalfDelta(a: f32, b: f32) -> f32 {
  let isSameHemisphere = a * b >= 0.0;
  // cos of the latitude a quarter of the way from a to b, from the exact colatitude near a pole.
  var quarterCosine = cos((a + 0.25 * (b - a)) * GEODESIC_DEGREES_TO_RADIANS);
  if (isSameHemisphere) {
    quarterCosine = sin(((90.0 - abs(a)) - 0.25 * (abs(b) - abs(a))) * GEODESIC_DEGREES_TO_RADIANS);
  }
  let halfDeltaPhi = 0.5 * (b - a) * GEODESIC_DEGREES_TO_RADIANS;
  if (abs(halfDeltaPhi) < 1e-4 * quarterCosine) {
    return halfDeltaPhi / quarterCosine;
  }
  let sinQuarterDelta = sin(0.5 * halfDeltaPhi);
  let denominator = 2.0 * sinQuarterDelta * sinQuarterDelta +
    geodesicCosLatitude(a) * rhumbMeanCosLatitude(a, b);
  let argument = 2.0 * quarterCosine * sinQuarterDelta / denominator;
  if (abs(argument) < 0.5) {
    return rhumbAtanh(argument);
  }
  let meanLatitude = 0.5 * (a + b);
  var meanColatitude = 90.0 - abs(meanLatitude);
  var meanMagnitude = abs(meanLatitude);
  if (isSameHemisphere) {
    meanColatitude = (90.0 - abs(a)) - 0.5 * (abs(b) - abs(a));
    meanMagnitude = 90.0 - meanColatitude;
  }
  let meanIsometric = rhumbIsometricLatitude(meanMagnitude, meanColatitude);
  let originIsometric = rhumbIsometricLatitude(abs(a), 90.0 - abs(a));
  return select(meanIsometric, -meanIsometric, meanLatitude < 0.0) -
    select(originIsometric, -originIsometric, a < 0.0);
}

// Midpoint of the rhumb line by distance. Latitude is linear in distance along a loxodrome, so
// the midpoint has the mean latitude. Longitude is linear in the stretched (Mercator) latitude
// psi, not in distance: lambda = lambda1 + deltaLambda * (psi(mean) - psi1) / (psi2 - psi1). An
// east-west line (psi2 == psi1) has the plain mean longitude.
// The fraction is firstHalf / (firstHalf + secondHalf), with both halves from the same function:
// psi(mean) - psi1 and psi2 - psi(mean). The second half is taken as the first half of the
// mirrored line (-b to -a; psi is odd), so for a line that is symmetric across the equator the
// two calls have identical arguments, the halves are the same f32 and the fraction is exactly
// 1/2 whatever the GPU's sin, cos and atanh round to. (Dividing by rhumbStretchedDelta, whose
// atanh argument is clamped at 0.999999, put the midpoint of (0, -88) to (120, 88) at longitude
// 66.96.) The longitude holds the f32 spacing of the output up to the latitude clamp of 89.9999
// degrees; beyond it both latitudes are clamped first.
fn rhumbMidpoint(origin: vec2<f32>, targetPosition: vec2<f32>) -> vec2<f32> {
  let deltaLambda = geodesicLongitudeDelta(origin.x, targetPosition.x);
  let a = rhumbClampLatitude(origin.y);
  let b = rhumbClampLatitude(targetPosition.y);
  let firstHalf = rhumbStretchedHalfDelta(a, b);
  let secondHalf = rhumbStretchedHalfDelta(-b, -a);
  let stretchedDelta = firstHalf + secondHalf;
  var fraction = 0.5;
  if (stretchedDelta != 0.0) {
    fraction = clamp(firstHalf / stretchedDelta, 0.0, 1.0);
  }
  return vec2<f32>(
    origin.x + fraction * deltaLambda,
    origin.y + 0.5 * (targetPosition.y - origin.y)
  );
}

// Destination after travelling 'angle' radians along a constant bearing.

fn rhumbDestination(origin: vec2<f32>, bearingDegrees: f32, angle: f32) -> RhumbDestination {
  let theta = bearingDegrees * GEODESIC_DEGREES_TO_RADIANS;
  var latitude = origin.y + angle * cos(theta) * GEODESIC_RADIANS_TO_DEGREES;
  var bearing = geodesicWrapLongitudeDelta(bearingDegrees);
  if (latitude > 90.0) {
    latitude = 180.0 - latitude;
    bearing = geodesicWrapLongitudeDelta(180.0 - bearing);
  } else if (latitude < -90.0) {
    latitude = -180.0 - latitude;
    bearing = geodesicWrapLongitudeDelta(180.0 - bearing);
  }
  let a = rhumbClampLatitude(origin.y);
  let b = rhumbClampLatitude(latitude);
  let q = rhumbStretchRatio(a, b);
  let deltaLambda = angle * sin(theta) / q;
  return RhumbDestination(
    vec2<f32>(origin.x + deltaLambda * GEODESIC_RADIANS_TO_DEGREES, latitude),
    bearing
  );
}
`,O=`GPUGeodesicPairs`,k=5,A=256,j=16777216,ee=6,M=class{id;props;spatialContext;model;radius;ellipsoid;maximumIterations;estimatedCommandNodeCount;chunkPlan;constructor(e){this.id=e.id??`geodesic-pairs`,this.props=e,this.spatialContext=e.spatialContext;let t=F(this.id,this.spatialContext);this.model=t.model,this.radius=t.radius,this.ellipsoid=t.ellipsoid,this.maximumIterations=e.maximumIterations??16,I(this.id,this.model,this.radius,this.maximumIterations,this.ellipsoid);let{id:n}=this,{output:r}=e;for(let[t,r]of[[`origins`,e.origins],[`targets`,e.targets]])for(let e of l(r))i(e,[`float32x2`],`${n} ${t}`);let a=e.origins.length;u(n,`origins.length`,a),s(n,`targets`,e.targets,a),R(n,r,{distances:`float32`,initialBearings:`float32`,finalBearings:`float32`,midpoints:`float32x2`,"status.converged":`uint32`,"status.approximation":`uint32`,"status.iterationLimitReached":`uint32`},a),c(n,!r.status?.converged||this.model===`wgs84`,`output.status.converged requires model 'wgs84'`);let o=L(r);this.chunkPlan=E(n,{origins:e.origins,targets:e.targets,...o}),C(n,o),this.estimatedCommandNodeCount=1+this.chunkPlan.nonemptyChunkCount*Math.ceil(Object.values(o).filter(Boolean).length/k),f(n,this.estimatedCommandNodeCount),p(n,Object.values(o),[e.origins,e.targets])}getCommandNodes(i){let{props:s,id:c}=this,{output:u}=s;T(i,c,this.chunkPlan);let d=!(`buffer`in s.origins)||!(`buffer`in s.targets)||Object.values(L(u)).some(e=>e&&!(`buffer`in e)),f=[{name:`origins`,view:l(s.origins)[0],type:`f32`,access:`read`},{name:`targets`,view:l(s.targets)[0],type:`f32`,access:`read`}],p=[],m=this.model===`rhumb`,y=(e,t,n)=>{let r=e===`converged`||e===`approximation`||e===`iterationLimitReached`?u.status?.[e]:u[e];r&&(f.push({name:e,view:l(r)[0],type:t,access:`read_write`}),p.push(n))};y(`distances`,`f32`,`distances[distancesOffset + index] = distance;`),y(`initialBearings`,`f32`,`initialBearings[initialBearingsOffset + index] = initialBearing;`),y(`finalBearings`,`f32`,`finalBearings[finalBearingsOffset + index] = finalBearing;`),y(`midpoints`,`f32`,`let midpoint = ${m?`rhumbMidpoint(origin, targetPosition)`:`geodesicInterpolate(origin, targetPosition, geodesicCentralAngle(origin, targetPosition), 0.5)`};
  midpoints[midpointsOffset + 2u * index] = midpoint.x;
  midpoints[midpointsOffset + 2u * index + 1u] = midpoint.y;`),y(`converged`,`u32`,`converged[convergedOffset + index] = select(0u, 1u, isConverged);`),y(`approximation`,`u32`,`approximation[approximationOffset + index] = select(0u, 1u, isApproximation);`),y(`iterationLimitReached`,`u32`,`iterationLimitReached[iterationLimitReachedOffset + index] = select(0u, 1u, isIterationLimitReached);`);let b=this.model===`wgs84`,x=f.slice(2),S=d?k:ee,C=Math.ceil(x.length/S),E=this.chunkPlan.chunks.map(e=>r(c,e.length,A,i.device.limits.maxComputeWorkgroupsPerDimension)),M=d?w(i,{id:`${c}-chunks`,plan:this.chunkPlan}):void 0,N=[...M?.nodes??[]];for(let r=0;r<C;r++){let l=r*S,u={id:`${c}-pairs${r===0?``:`-${r}`}`,operation:O,variant:this.model,bindings:[...f.slice(0,2),...x.slice(l,l+S)],invocationCount:s.origins.length,declarations:`${h}
${b?g(this.maximumIterations,this.ellipsoid):``}
${v}
${m?D:``}
const RADIUS: f32 = ${e(this.radius)};`,body:`let origin = vec2<f32>(origins[originsOffset + 2u * index], origins[originsOffset + 2u * index + 1u]);
  let targetPosition = vec2<f32>(targets[targetsOffset + 2u * index], targets[targetsOffset + 2u * index + 1u]);
  var distance = ${m?`rhumbAngle(origin, targetPosition)`:`geodesicCentralAngle(origin, targetPosition)`} * RADIUS;
  var initialBearing = ${m?`rhumbBearingDegrees(origin, targetPosition)`:`geodesicInitialBearingDegrees(origin, targetPosition)`};
  var finalBearing = ${m?`initialBearing`:`sphereFinalBearingDegrees(origin, targetPosition)`};
  let antipodal = GEODESIC_PI - geodesicCentralAngle(origin, targetPosition) < ${e(_)};
  var isApproximation = ${m?`false`:`antipodal`};
  var isConverged = true;
  var isIterationLimitReached = false;
  ${b?`let inverse = vincentyInverse(origin, targetPosition);
  isConverged = inverse.converged && !antipodal;
  isApproximation = !isConverged;
  isIterationLimitReached = inverse.iterationLimitReached;
  if (isConverged) {
    distance = inverse.distance;
    initialBearing = inverse.initialBearing;
    finalBearing = inverse.finalBearing;
  }`:``}
  ${p.slice(l,l+S).join(`
  `)}`};if(!M){N.push(t(i,u));continue}let d=`
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
${u.bindings.map((e,t)=>`@group(0) @binding(${t+1}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`)}
${u.declarations}
@compute @workgroup_size(${A})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${u.bindings.map(e=>`_ = &${e.name};`).join(`
  `)}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${j}u) { return; }
  let index = workgroupIndex * ${A}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  ${u.bindings.map(e=>`let ${e.name}Offset = arrayLength(&${e.name}) - ${e.view.rowByteLength/4}u * rowCount;`).join(`
  `)}
  ${u.body}
}`;for(let e of this.chunkPlan.chunks){if(e.length===0)continue;let t=M.slots[e.chunkIndex],i=u.bindings.map(t=>({...t,view:e.views[t.name===`converged`||t.name===`approximation`||t.name===`iterationLimitReached`?`status.${t.name}`:t.name]}));N.push(n({id:`${c}-pairs-${r}-chunk-${e.chunkIndex}`,workload:{operation:O,variant:this.model,commandCount:1,maximumInvocationCount:e.length,maximumWorkgroupCount:E[e.chunkIndex].x*E[e.chunkIndex].y*E[e.chunkIndex].z},resources:[{buffer:t,usage:`storage-read`},...i.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`}))],compile:({device:n})=>{let s=new o(n,{id:`geodesic-pairs-chunk-${this.model}-${r}`,source:d,shaderLayout:{bindings:[{name:`chunkParameters`,group:0,location:0,type:`read-only-storage`},...i.map((e,t)=>({name:e.name,group:0,location:t+1,type:e.access===`read`?`read-only-storage`:`storage`}))]}});return{encode:({computePass:n,getBuffer:r})=>s.dispatch(n,{bindings:{chunkParameters:a(t,r),...Object.fromEntries(i.map(e=>[e.name,a(e.view,r)]))},...E[e.chunkIndex]}),destroy:()=>s.destroy()}}}))}}return N}},N=Object.freeze({measure:`distance`,coordinateSpaces:Object.freeze([`longitude-latitude`]),metricsByCoordinateSpace:Object.freeze({"longitude-latitude":Object.freeze([`great-circle`,`rhumb`,`ellipsoidal`])})}),P={"great-circle":`sphere`,rhumb:`rhumb`,ellipsoidal:`wgs84`};function F(e,t){let n=S(e,t,N),r=P[n.metric],i=x(n);return{model:r,radius:b(n),ellipsoid:n.units===`kilometers`?{...i,semiMajorAxis:i.semiMajorAxis/1e3}:i}}function I(e,t,n,r,i){if(c(e,t===`sphere`||t===`wgs84`||t===`rhumb`,`model must be 'sphere', 'wgs84' or 'rhumb'`),d(e,`radius`,n),c(e,Math.fround(n)>0,`radius must be positive in float32`),t===`wgs84`&&i){let{semiMajorAxis:t,flattening:n}=i,r=t*(1-n),a=(t*t-r*r)/(r*r);for(let[i,o]of Object.entries({semiMajorAxis:t,semiMinorAxis:r,flattening:n,secondEccentricitySquared:a}))d(e,`spatialContext.ellipsoid.${i}`,o);c(e,Math.fround(t)>0&&Math.fround(r)>0,`spatialContext.ellipsoid axes must be positive in float32`),c(e,Math.fround(n)<1,`spatialContext.ellipsoid.flattening must be less than 1 in float32`)}u(e,`maximumIterations`,r,1,200)}function L(e){let{status:t,...n}=e;return{...n,"status.converged":t?.converged,...t?.approximation?{"status.approximation":t.approximation}:{},...t?.iterationLimitReached?{"status.iterationLimitReached":t.iterationLimitReached}:{}}}function R(e,t,n,r){let a=0;for(let[o,u]of Object.entries(L(t))){if(!u)continue;let t=n[o];c(e,!!t,`output.${o} is not a known column`);for(let n of l(u))i(n,[t],`${e} output.${o}`);s(e,`output.${o}`,u,r),a++}c(e,a>0,`output requires at least one column`)}var z=`GPUGeodesicDestination`,B=4,V=256,H=16777216,U=class{id;props;spatialContext;model;radius;ellipsoid;maximumIterations;estimatedCommandNodeCount;chunkPlan;constructor(e){this.id=e.id??`geodesic-destination`,this.props=e,this.spatialContext=e.spatialContext;let t=F(this.id,this.spatialContext);this.model=t.model,this.radius=t.radius,this.ellipsoid=t.ellipsoid,this.maximumIterations=e.maximumIterations??16,I(this.id,this.model,this.radius,this.maximumIterations,this.ellipsoid);let{id:n}=this;for(let[t,r,a]of[[`origins`,e.origins,`float32x2`],[`bearings`,e.bearings,`float32`],[`distances`,e.distances,`float32`]])for(let e of l(r))i(e,[a],`${n} ${t}`);let r=e.origins.length;u(n,`origins.length`,r),s(n,`bearings`,e.bearings,r),s(n,`distances`,e.distances,r),R(n,e.output,{destinations:`float32x2`,finalBearings:`float32`,"status.converged":`uint32`,"status.iterationLimitReached":`uint32`,"status.approximation":`uint32`},r),c(n,!e.output.status?.converged||this.model===`wgs84`,`output.status.converged requires model 'wgs84'`);let a=L(e.output);this.chunkPlan=E(n,{origins:e.origins,bearings:e.bearings,distances:e.distances,...a}),C(n,a),this.estimatedCommandNodeCount=1+this.chunkPlan.nonemptyChunkCount*Math.ceil(Object.values(a).filter(Boolean).length/B),f(n,this.estimatedCommandNodeCount),p(n,Object.values(L(e.output)),[e.origins,e.bearings,e.distances])}getCommandNodes(i){let{props:s,id:c}=this,{output:u}=s;T(i,c,this.chunkPlan);let d=!(`buffer`in s.origins)||!(`buffer`in s.bearings)||!(`buffer`in s.distances)||Object.values(L(u)).some(e=>e&&!(`buffer`in e)),f=[{name:`origins`,view:l(s.origins)[0],type:`f32`,access:`read`},{name:`bearings`,view:l(s.bearings)[0],type:`f32`,access:`read`},{name:`distances`,view:l(s.distances)[0],type:`f32`,access:`read`}],p=[];u.destinations&&(f.push({name:`destinations`,view:l(u.destinations)[0],type:`f32`,access:`read_write`}),p.push(`destinations[destinationsOffset + 2u * index] = destination.x;
  destinations[destinationsOffset + 2u * index + 1u] = destination.y;`)),u.finalBearings&&(f.push({name:`finalBearings`,view:l(u.finalBearings)[0],type:`f32`,access:`read_write`}),p.push(`finalBearings[finalBearingsOffset + index] = finalBearing;`)),u.status?.converged&&(f.push({name:`converged`,view:l(u.status.converged)[0],type:`u32`,access:`read_write`}),p.push(`converged[convergedOffset + index] = select(0u, 1u, isConverged);`)),u.status?.iterationLimitReached&&(f.push({name:`iterationLimitReached`,view:l(u.status.iterationLimitReached)[0],type:`u32`,access:`read_write`}),p.push(`iterationLimitReached[iterationLimitReachedOffset + index] = select(1u, 0u, isConverged);`)),u.status?.approximation&&(f.push({name:`approximation`,view:l(u.status.approximation)[0],type:`u32`,access:`read_write`}),p.push(`approximation[approximationOffset + index] = select(0u, 1u, !isConverged || (distance != 0.0 && abs(geodesicCosLatitude(destination.y)) <= ${e(y)}));`));let m=this.model===`wgs84`,_=this.model===`rhumb`,b=`let origin = vec2<f32>(origins[originsOffset + 2u * index], origins[originsOffset + 2u * index + 1u]);
  let bearing = bearings[bearingsOffset + index];
  let distance = distances[distancesOffset + index];
  ${m?`let direct = vincentyDirect(origin, bearing, distance);
  let destination = direct.destination;
  let finalBearing = direct.finalBearing;
  let isConverged = direct.converged;`:_?`let rhumbResult = rhumbDestination(origin, bearing, distance / RADIUS);
  let destination = rhumbResult.destination;
  let finalBearing = rhumbResult.finalBearing;
  let isConverged = true;`:`let destination = geodesicDestination(origin, bearing, distance / RADIUS);
  let finalBearing = sphereDirectFinalBearingDegrees(origin, bearing, distance / RADIUS);
  let isConverged = true;`}
`,x={id:`${c}-destination`,operation:z,variant:this.model,bindings:f,invocationCount:s.origins.length,declarations:`${h}
${m?g(this.maximumIterations,this.ellipsoid):``}
${v}
${this.model===`rhumb`?D:``}
const RADIUS: f32 = ${e(this.radius)};`,body:`${b}\n  ${p.join(`
  `)}`};if(!d)return[t(i,x)];let S=this.chunkPlan.chunks.map(e=>r(c,e.length,V,i.device.limits.maxComputeWorkgroupsPerDimension)),{slots:C,nodes:E}=w(i,{id:`${c}-chunks`,plan:this.chunkPlan}),O=[...E],k=f.slice(3),A=Math.ceil(k.length/B);for(let e=0;e<A;e++){let t=e*B,r=[...f.slice(0,3),...k.slice(t,t+B)],i=`${b}\n  ${p.slice(t,t+B).join(`
  `)}`,s=`
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
${r.map((e,t)=>`@group(0) @binding(${t+1}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`)}
${x.declarations}
@compute @workgroup_size(${V})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${r.map(e=>`_ = &${e.name};`).join(`
  `)}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${H}u) { return; }
  let index = workgroupIndex * ${V}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  ${r.map(e=>`let ${e.name}Offset = arrayLength(&${e.name}) - ${e.view.rowByteLength/4}u * rowCount;`).join(`
  `)}
  ${i}
}`;for(let t of this.chunkPlan.chunks){if(t.length===0)continue;let i=C[t.chunkIndex],l=r.map(e=>({...e,view:t.views[e.name===`converged`||e.name===`approximation`||e.name===`iterationLimitReached`?`status.${e.name}`:e.name]}));O.push(n({id:`${c}-destination-${e}-chunk-${t.chunkIndex}`,workload:{operation:z,variant:this.model,commandCount:1,maximumInvocationCount:t.length,maximumWorkgroupCount:S[t.chunkIndex].x*S[t.chunkIndex].y*S[t.chunkIndex].z},resources:[{buffer:i,usage:`storage-read`},...l.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`}))],compile:({device:n})=>{let r=new o(n,{id:`geodesic-destination-chunk-${this.model}-${e}`,source:s,shaderLayout:{bindings:[{name:`chunkParameters`,group:0,location:0,type:`read-only-storage`},...l.map((e,t)=>({name:e.name,group:0,location:t+1,type:e.access===`read`?`read-only-storage`:`storage`}))]}});return{encode:({computePass:e,getBuffer:n})=>r.dispatch(e,{bindings:{chunkParameters:a(i,n),...Object.fromEntries(l.map(e=>[e.name,a(e.view,n)]))},...S[t.chunkIndex]}),destroy:()=>r.destroy()}}}))}}return O}},W=40;function G(e){return e.format.startsWith(`uint32`)?`u32`:e.format.startsWith(`sint32`)?`i32`:`f32`}function K(e,n){e.graph.add({getCommandNodes:e=>[t(e,{id:n.id,operation:`GeometryRunAdapter`,bindings:n.bindings.map(e=>({name:e.name,view:e.buffer.view,type:G(e.buffer),access:e.access})),invocationCount:n.invocationCount,body:n.body})]})}function q(e,t,n){let r=n.format===`float32x3`?3:4,i=e.allocate(`${t}-segments`,`float32x4`,n.length*W);return K(e,{id:`${t}-circle-segments`,invocationCount:n.length*W,bindings:[{name:`circles`,buffer:n,access:`read`},{name:`segments`,buffer:i,access:`read_write`}],body:`let circle = index / ${W}u;
  let step = index % ${W}u;
  let base = circlesOffset + circle * ${r}u;
  let center = vec2<f32>(circles[base], circles[base + 1u]);
  let radius = circles[base + 2u];
  let angle0 = f32(step) * ${Math.PI*2/W};
  let angle1 = f32(step + 1u) * ${Math.PI*2/W};
  let outBase = segmentsOffset + index * 4u;
  segments[outBase] = center.x + cos(angle0) * radius;
  segments[outBase + 1u] = center.y + sin(angle0) * radius;
  segments[outBase + 2u] = center.x + cos(angle1) * radius;
  segments[outBase + 3u] = center.y + sin(angle1) * radius;`}),i}function J(e,t,n){return Y(e,t,n,n.length)}function Y(e,t,n,r){let i=e.allocate(`${t}-segments`,`float32x4`,r*4);return K(e,{id:`${t}-box-segments`,invocationCount:r*4,bindings:[{name:`boxes`,buffer:n,access:`read`},{name:`segments`,buffer:i,access:`read_write`}],body:`let boxIndex = index / 4u;
  let side = index % 4u;
  let base = boxesOffset + boxIndex * 4u;
  let low = vec2<f32>(boxes[base], boxes[base + 1u]);
  let high = vec2<f32>(boxes[base + 2u], boxes[base + 3u]);
  var fromPoint = low;
  var toPoint = vec2<f32>(high.x, low.y);
  if (side == 1u) { fromPoint = toPoint; toPoint = high; }
  if (side == 2u) { fromPoint = high; toPoint = vec2<f32>(low.x, high.y); }
  if (side == 3u) { fromPoint = vec2<f32>(low.x, high.y); toPoint = low; }
  let outBase = segmentsOffset + index * 4u;
  segments[outBase] = fromPoint.x;
  segments[outBase + 1u] = fromPoint.y;
  segments[outBase + 2u] = toPoint.x;
  segments[outBase + 3u] = toPoint.y;`}),i}function X(e,t,n){let r=e.allocate(`${t}-segments`,`float32x4`,n.length);return K(e,{id:`${t}-quad-segments`,invocationCount:n.length,bindings:[{name:`corners`,buffer:n,access:`read`},{name:`segments`,buffer:r,access:`read_write`}],body:`let groupIndex = index / 4u;
  let side = index % 4u;
  let fromBase = cornersOffset + (groupIndex * 4u + side) * 2u;
  let toBase = cornersOffset + (groupIndex * 4u + (side + 1u) % 4u) * 2u;
  let outBase = segmentsOffset + index * 4u;
  segments[outBase] = corners[fromBase];
  segments[outBase + 1u] = corners[fromBase + 1u];
  segments[outBase + 2u] = corners[toBase];
  segments[outBase + 3u] = corners[toBase + 1u];`}),r}var Z=1e6;function te(e,t,n,r){let i=n.format===`float32x4`?2:1,a=n.length*i,o=e.allocate(`${t}-masked`,`float32x2`,a);return K(e,{id:`${t}-mask-points`,invocationCount:a,bindings:[{name:`points`,buffer:n,access:`read`},{name:`count`,buffer:r,access:`read`},{name:`masked`,buffer:o,access:`read_write`}],body:`let isLive = index / ${i}u < count[countOffset];
  masked[maskedOffset + index * 2u] = select(${Z}, points[pointsOffset + index * 2u], isLive);
  masked[maskedOffset + index * 2u + 1u] = select(${Z}, points[pointsOffset + index * 2u + 1u], isLive);`}),o}function ne(e,t,n,r,i){let a=e.allocate(`${t}-edges`,`uint32x2`,r);return K(e,{id:`${t}-output-edges`,invocationCount:r,bindings:[{name:`offsets`,buffer:n,access:`read`},{name:`edges`,buffer:a,access:`read_write`}],body:`var nextVertex = index;
  for (var line = 0u; line + 1u < ${n.length}u; line++) {
    let start = offsets[offsetsOffset + line];
    let end = offsets[offsetsOffset + line + 1u];
    if (index >= start && index < end) {
      if (index + 1u < end) { nextVertex = index + 1u; }
      ${i?`else { nextVertex = start; }`:``}
    }
  }
  edges[edgesOffset + index * 2u] = index;
  edges[edgesOffset + index * 2u + 1u] = nextVertex;`}),a}function re(e,t,n){let r=e.allocate(`${t}-written`,`float32x4`,n.length);return K(e,{id:`${t}-written-segments`,invocationCount:n.length,bindings:[{name:`segments`,buffer:n,access:`read`},{name:`written`,buffer:r,access:`read_write`}],body:`let base = segmentsOffset + index * 4u;
  let isWritten = segments[base] != 0.0 || segments[base + 1u] != 0.0 || segments[base + 2u] != 0.0 || segments[base + 3u] != 0.0;
  for (var component = 0u; component < 4u; component++) {
    written[writtenOffset + index * 4u + component] = select(${Z}, segments[base + component], isWritten);
  }`}),r}function ie(e,t,n,r){let i=e.allocate(`${t}-vertex-values`,`uint32`,r.length);return K(e,{id:`${t}-feature-values`,invocationCount:r.length,bindings:[{name:`featureValues`,buffer:n,access:`read`},{name:`featureOfVertex`,buffer:r,access:`read`},{name:`vertexValues`,buffer:i,access:`read_write`}],body:`vertexValues[vertexValuesOffset + index] = featureValues[featureValuesOffset + featureOfVertex[featureOfVertexOffset + index]];`}),i}function ae(e,t,n,r){let i=e.allocate(`${t}-world`,`float32x2`,n.length);return K(e,{id:`${t}-equirectangular`,invocationCount:n.length,bindings:[{name:`lonLat`,buffer:n,access:`read`},{name:`projected`,buffer:i,access:`read_write`}],body:`let lon = lonLat[lonLatOffset + index * 2u];
  let lat = lonLat[lonLatOffset + index * 2u + 1u];
  projected[projectedOffset + index * 2u] = ${r.width*.04} + (lon + 180.0) / 360.0 * ${r.width*.92};
  projected[projectedOffset + index * 2u + 1u] = 0.04 + (lat + 90.0) / 180.0 * 0.92;`}),i}function oe(e,t,n,r){let i=e.allocate(`${t}-pairs`,`float32x4`,n.length);return K(e,{id:`${t}-pair-segments`,invocationCount:n.length,bindings:[{name:`starts`,buffer:n,access:`read`},{name:`ends`,buffer:r,access:`read`},{name:`segments`,buffer:i,access:`read_write`}],body:`let base = segmentsOffset + index * 4u;
  segments[base] = starts[startsOffset + index * 2u];
  segments[base + 1u] = starts[startsOffset + index * 2u + 1u];
  segments[base + 2u] = ends[endsOffset + index * 2u];
  segments[base + 3u] = ends[endsOffset + index * 2u + 1u];`}),i}function se(e,t,n,r){let i=e.allocate(`${t}-ring-categories`,`uint32`,r);return K(e,{id:`${t}-ring-categories`,invocationCount:r,bindings:[{name:`offsets`,buffer:n,access:`read`},{name:`categories`,buffer:i,access:`read_write`}],body:`var category = 0xffffffffu;
  for (var ring = 0u; ring + 1u < ${n.length}u; ring++) {
    if (index >= offsets[offsetsOffset + ring] && index < offsets[offsetsOffset + ring + 1u]) { category = ring; }
  }
  categories[categoriesOffset + index] = category;`}),i}var Q=640;function $(e,t){let n=new Float32Array(Q*4).fill(Z),r=Math.floor(t.width/e),i=Math.floor(t.height/e),a=Math.max(1,Math.ceil((r+i+2)/Q)),o=0;for(let i=0;i<=r&&o<Q;i+=a)n.set([i*e,0,i*e,t.height],o++*4);for(let r=0;r<=i&&o<Q;r+=a)n.set([0,r*e,t.width,r*e],o++*4);return n}function ce(e,t,n,r){return e.upload(t,`float32x4`,$(n,r))}function le(e,t,n){let{origins:r,bearings:i,distances:a,rowCount:o,steps:s,world:c}=n,l=o*s,u=e.allocate(`${t}-sample-origins`,`float32x2`,l),d=e.allocate(`${t}-sample-bearings`,`float32`,l),f=e.allocate(`${t}-sample-distances`,`float32`,l);K(e,{id:`${t}-arc-samples`,invocationCount:l,bindings:[{name:`origins`,buffer:r,access:`read`},{name:`bearings`,buffer:i,access:`read`},{name:`distances`,buffer:a,access:`read`},{name:`sampleOrigins`,buffer:u,access:`read_write`},{name:`sampleBearings`,buffer:d,access:`read_write`},{name:`sampleDistances`,buffer:f,access:`read_write`}],body:`let row = index / ${s}u;
  let step = index % ${s}u;
  sampleOrigins[sampleOriginsOffset + index * 2u] = origins[originsOffset + row * 2u];
  sampleOrigins[sampleOriginsOffset + index * 2u + 1u] = origins[originsOffset + row * 2u + 1u];
  sampleBearings[sampleBearingsOffset + index] = bearings[bearingsOffset + row];
  sampleDistances[sampleDistancesOffset + index] = distances[distancesOffset + row] * f32(step + 1u) / ${s}.0;`});let p=e.allocate(`${t}-samples`,`float32x2`,l);e.graph.add(new U({id:`${t}-geodesic`,spatialContext:{coordinateSpace:`longitude-latitude`,metric:`ellipsoidal`,units:`meters`},origins:u.view,bearings:d.view,distances:f.view,output:{destinations:p.view,finalBearings:e.allocate(`${t}-sample-final-bearings`,`float32`,l).view}}));let m=e.allocate(`${t}-arc-segments`,`float32x4`,l);return K(e,{id:`${t}-arc-segments`,invocationCount:l,bindings:[{name:`origins`,buffer:r,access:`read`},{name:`samples`,buffer:p,access:`read`},{name:`segments`,buffer:m,access:`read_write`}],body:`let row = index / ${s}u;
  let step = index % ${s}u;
  var startPoint = vec2<f32>(origins[originsOffset + row * 2u], origins[originsOffset + row * 2u + 1u]);
  if (step > 0u) {
    startPoint = vec2<f32>(samples[samplesOffset + (index - 1u) * 2u], samples[samplesOffset + (index - 1u) * 2u + 1u]);
  }
  let endPoint = vec2<f32>(samples[samplesOffset + index * 2u], samples[samplesOffset + index * 2u + 1u]);
  let isWrapped = abs(startPoint.x - endPoint.x) > 180.0;
  let base = segmentsOffset + index * 4u;
  segments[base] = select(${c.width*.04} + (startPoint.x + 180.0) / 360.0 * ${c.width*.92}, ${Z}, isWrapped);
  segments[base + 1u] = select(0.04 + (startPoint.y + 90.0) / 180.0 * 0.92, ${Z}, isWrapped);
  segments[base + 2u] = select(${c.width*.04} + (endPoint.x + 180.0) / 360.0 * ${c.width*.92}, ${Z}, isWrapped);
  segments[base + 3u] = select(0.04 + (endPoint.y + 90.0) / 180.0 * 0.92, ${Z}, isWrapped);`}),m}function ue(e,t,n,r,i,a=1){let o=e.allocate(`${t}-arrows`,`float32x4`,i*3);return K(e,{id:`${t}-ring-arrows`,invocationCount:i,bindings:[{name:`positions`,buffer:n,access:`read`},{name:`offsets`,buffer:r,access:`read`},{name:`segments`,buffer:o,access:`read_write`}],body:`let first = offsets[offsetsOffset + index];
  let start = vec2<f32>(positions[positionsOffset + first * 2u], positions[positionsOffset + first * 2u + 1u]);
  let end = vec2<f32>(positions[positionsOffset + first * 2u + 2u], positions[positionsOffset + first * 2u + 3u]);
  let direction = normalize(end - start + vec2<f32>(1e-9, 0.0));
  let normal = vec2<f32>(-direction.y, direction.x);
  let middle = (start + end) * 0.5;
  let tip = middle + direction * ${(.024*a).toFixed(4)};
  let tail = middle - direction * ${(.024*a).toFixed(4)};
  let wingA = tip - direction * ${(.022*a).toFixed(4)} + normal * ${(.013*a).toFixed(4)};
  let wingB = tip - direction * ${(.022*a).toFixed(4)} - normal * ${(.013*a).toFixed(4)};
  let base = segmentsOffset + index * 12u;
  segments[base] = tail.x; segments[base + 1u] = tail.y; segments[base + 2u] = tip.x; segments[base + 3u] = tip.y;
  segments[base + 4u] = tip.x; segments[base + 5u] = tip.y; segments[base + 6u] = wingA.x; segments[base + 7u] = wingA.y;
  segments[base + 8u] = tip.x; segments[base + 9u] = tip.y; segments[base + 10u] = wingB.x; segments[base + 11u] = wingB.y;`}),o}export{U as _,ie as a,te as c,X as d,ue as f,$ as g,ce as h,ae as i,ne as l,re as m,Y as n,le as o,se as p,q as r,K as s,J as t,oe as u,M as v};
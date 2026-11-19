import{t as e}from"./spatial-output-validation-nw5lOmVL.js";import{a as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as n}from"./kernel-C3tfQOn4.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{v as i}from"./spatial-context-DwZEd4EZ.js";import{T as a,f as o,v as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{t as u}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as d}from"./exact-geometry-wgsl-y7TUnqOh.js";import{a as f,n as p,o as m}from"./study-window-wgsl-B3mOUanG.js";import{l as h}from"./wgsl-common-DfyXiYB0.js";import{n as g,t as _}from"./point-process-utils-BVnRkMJj.js";import{t as v}from"./gpu-study-window-preparation-CbhCN1Wi.js";import{t as y}from"./define-run-B-brkCe4.js";import{n as b}from"./inputs-CJKKIRKK.js";import{d as x,s as S}from"./pointer-overlay-Qh8qflcA.js";import{o as C}from"./interpolation-CWOuK2ek.js";import{Ys as w}from"./entry-preview-D70TdSbK.js";var T=`
fn overlapExactDeterminant(a: vec2f, b: vec2f, c: vec2f) -> f32 {
  var invalidBits = 0x7fc00000u;
  let invalid = bitcast<f32>(invalidBits);
  var xBits = array<u32, 6>(
    bitcast<u32>(a.x), bitcast<u32>(a.y) ^ 0x80000000u, bitcast<u32>(b.x),
    bitcast<u32>(b.y) ^ 0x80000000u, bitcast<u32>(c.x), bitcast<u32>(c.y) ^ 0x80000000u);
  var yBits = array<u32, 6>(
    bitcast<u32>(b.y), bitcast<u32>(b.x), bitcast<u32>(c.y),
    bitcast<u32>(c.x), bitcast<u32>(a.y), bitcast<u32>(a.x));
  var exponents: array<i32, 6>;
  var lows: array<u32, 6>;
  var highs: array<u32, 6>;
  var negatives: array<bool, 6>;
  var present: array<bool, 6>;
  var minimumExponent = 100000;
  var maximumExponent = -100000;
  for (var term = 0u; term < 6u; term++) {
    let xMagnitude = xBits[term] & 0x7fffffffu;
    let yMagnitude = yBits[term] & 0x7fffffffu;
    let xExponentBits = xMagnitude >> 23u;
    let yExponentBits = yMagnitude >> 23u;
    let xFraction = xMagnitude & 0x7fffffu;
    let yFraction = yMagnitude & 0x7fffffu;
    if (xMagnitude == 0u || yMagnitude == 0u) { continue; }
    let xSignificand = select(0x800000u | xFraction, xFraction, xExponentBits == 0u);
    let ySignificand = select(0x800000u | yFraction, yFraction, yExponentBits == 0u);
    let xExponent = select(i32(xExponentBits) - 150, -149, xExponentBits == 0u);
    let yExponent = select(i32(yExponentBits) - 150, -149, yExponentBits == 0u);
    let product = multiply24(xSignificand, ySignificand);
    present[term] = true;
    lows[term] = product.x;
    highs[term] = product.y;
    exponents[term] = xExponent + yExponent;
    negatives[term] = ((xBits[term] ^ yBits[term]) >> 31u) == 1u;
    minimumExponent = min(minimumExponent, exponents[term]);
    maximumExponent = max(maximumExponent, exponents[term]);
  }
  if (minimumExponent > maximumExponent) { return 0.0; }
  if (maximumExponent - minimumExponent > ORIENT_MAXIMUM_SPAN) { return invalid; }
  var positive: array<u32, 8>;
  var negative: array<u32, 8>;
  for (var term = 0u; term < 6u; term++) {
    if (!present[term]) { continue; }
    let shift = u32(exponents[term] - minimumExponent);
    if (negatives[term]) {
      accumulateShifted(&negative, lows[term], highs[term], shift);
    } else {
      accumulateShifted(&positive, lows[term], highs[term], shift);
    }
  }
  var negativeResult = false;
  for (var slot = 8u; slot > 0u; slot--) {
    let plus = positive[slot - 1u]; let minus = negative[slot - 1u];
    if (plus != minus) { negativeResult = plus < minus; break; }
  }
  var difference: array<u32, 8>;
  var borrow = 0u;
  var highest = 0u;
  var nonzero = false;
  for (var slot = 0u; slot < 8u; slot++) {
    let larger = select(positive[slot], negative[slot], negativeResult);
    let smaller = select(negative[slot], positive[slot], negativeResult);
    difference[slot] = larger - smaller - borrow;
    borrow = select(0u, 1u, larger < smaller || (larger == smaller && borrow != 0u));
    if (difference[slot] != 0u) { highest = slot; nonzero = true; }
  }
  if (!nonzero) { return 0.0; }
  let highestBit = 31u - countLeadingZeros(difference[highest]);
  let exponent = minimumExponent + i32(highest * 32u + highestBit);
  // Coordinates and constructed metric intersections must have normal finite determinants.
  if (exponent < -126 || exponent > 127) { return invalid; }
  var significand = 0u;
  if (highestBit >= 23u) { significand = difference[highest] >> (highestBit - 23u); }
  else {
    significand = difference[highest] << (23u - highestBit);
    if (highest > 0u) { significand |= difference[highest - 1u] >> (9u + highestBit); }
  }
  let signBit = select(0u, 0x80000000u, negativeResult);
  return bitcast<f32>(signBit | (u32(exponent + 127) << 23u) | (significand & 0x7fffffu));
}

fn overlapDeterminant(a: vec2f, b: vec2f, c: vec2f) -> f32 {
  let first = (a.x - c.x) * (b.y - c.y);
  let second = (a.y - c.y) * (b.x - c.x);
  let determinant = first - second;
  if (abs(determinant) > 3.0e-7 * (abs(first) + abs(second))) { return determinant; }
  return overlapExactDeterminant(a, b, c);
}
`;function E(e,t){return`
${T}
fn overlapTriangle(first: vec2f, second: vec2f, shift: vec2f, third: vec2f, fourth: vec2f) -> f32 {
  var polygon: array<vec2f, 8>;
  var clipped: array<vec2f, 8>;
  polygon[0] = vec2f(0.0); polygon[1] = first; polygon[2] = second;
  var count = 3u;
  let clip = array<vec2f, 3>(shift, shift + third, shift + fourth);
  for (var edge = 0u; edge < 3u; edge++) {
    let start = clip[edge]; let finish = clip[(edge + 1u) % 3u];
    var nextCount = 0u;
    for (var vertex = 0u; vertex < count; vertex++) {
      let previous = polygon[(vertex + count - 1u) % count];
      let current = polygon[vertex];
      let previousSign = orientSign(start, finish, previous);
      let currentSign = orientSign(start, finish, current);
      if (previousSign == 2i || currentSign == 2i) { return -1.0; }
      if ((previousSign < 0i) != (currentSign < 0i)) {
        if (nextCount >= 8u) { return -1.0; }
        var intersection = previous;
        if (currentSign == 0i) { intersection = current; }
        else if (previousSign != 0i) {
          let previousDistance = overlapDeterminant(start, finish, previous);
          let currentDistance = overlapDeterminant(start, finish, current);
          let denominator = previousDistance - currentDistance;
          if (denominator == 0.0 || !isFiniteFloat(denominator)) { return -1.0; }
          let fraction = clamp(previousDistance / denominator, 0.0, 1.0);
          intersection = previous + fraction * (current - previous);
        }
        clipped[nextCount] = intersection; nextCount++;
      }
      if (currentSign >= 0i) {
        if (nextCount >= 8u) { return -1.0; }
        clipped[nextCount] = current; nextCount++;
      }
    }
    count = nextCount;
    if (count < 3u) { return 0.0; }
    for (var vertex = 0u; vertex < count; vertex++) { polygon[vertex] = clipped[vertex]; }
  }
  var area = 0.0;
  for (var vertex = 1u; vertex + 1u < count; vertex++) {
    area += studyWindowCross(polygon[vertex] - polygon[0], polygon[vertex + 1u] - polygon[0]);
  }
  return abs(area) * 0.5;
}
fn polygonTranslationOverlap(shift: vec2f) -> f32 {
  let origin = vec2f(studyWindowSummaryValue(0u), studyWindowSummaryValue(1u));
  let extent = vec2f(studyWindowSummaryValue(2u), studyWindowSummaryValue(3u)) - origin;
  if (abs(shift.x) >= extent.x || abs(shift.y) >= extent.y) { return 0.0; }
  if (all(shift == vec2f(0.0))) { return studyWindowSummaryValue(4u); }
  // Streaming binary reduction: each occupied bin contains 2^level intersections.
  // Even at the maximum 2^20 work budget, summation depth stays logarithmic.
  var areaBins: array<f32, 21>;
  var magnitudeBins: array<f32, 21>;
  var visited = 0u;
  for (var firstEdge = 0u; firstEdge < ${e}u; firstEdge++) {
    let first = studyWindowEdge(firstEdge);
    if (!studyWindowFinitePoint(first.xy) || !studyWindowFinitePoint(first.zw)) { continue; }
    var firstStart = first.xy - origin; var firstEnd = first.zw - origin;
    let firstSign = orientSign(vec2f(0.0), firstStart, firstEnd);
    if (firstSign == 2i) { return -2.0; }
    if (firstSign == 0i) { continue; }
    if (firstSign < 0i) { let saved = firstStart; firstStart = firstEnd; firstEnd = saved; }
    for (var secondEdge = 0u; secondEdge < ${e}u; secondEdge++) {
      let second = studyWindowEdge(secondEdge);
      if (!studyWindowFinitePoint(second.xy) || !studyWindowFinitePoint(second.zw)) { continue; }
      var secondStart = second.xy - origin; var secondEnd = second.zw - origin;
      let secondSign = orientSign(vec2f(0.0), secondStart, secondEnd);
      if (secondSign == 2i) { return -2.0; }
      if (secondSign == 0i) { continue; }
      if (visited >= ${t}u) { return -1.0; } visited++;
      if (secondSign < 0i) { let saved = secondStart; secondStart = secondEnd; secondEnd = saved; }
      let intersection = overlapTriangle(firstStart, firstEnd, shift, secondStart, secondEnd);
      if (intersection < 0.0 || !isFiniteFloat(intersection)) { return -2.0; }
      var contribution = f32(firstSign * secondSign) * intersection;
      var absoluteContribution = intersection;
      var occupied = visited - 1u;
      var level = 0u;
      while ((occupied & 1u) != 0u) {
        contribution += areaBins[level]; absoluteContribution += magnitudeBins[level];
        occupied >>= 1u; level++;
      }
      areaBins[level] = contribution; magnitudeBins[level] = absoluteContribution;
    }
  }
  var area = 0.0; var magnitude = 0.0;
  for (var level = 0u; level < 21u; level++) {
    if ((visited & (1u << level)) != 0u) { area += areaBins[level]; magnitude += magnitudeBins[level]; }
  }
  // Refuse an unresolved cancellation/sliver instead of inventing a large reciprocal weight.
  if (!isFiniteFloat(area) || area <= max(magnitude, studyWindowSummaryValue(4u)) * 0.00000762939453125) { return -2.0; }
  return area;
}
`}function D(e,t,i,a,o){let s=t.positions.length,c=s*(s-1)/2,l=t.radii.length,g=n(e,`${i}-translation-pairs`,`float32`,Math.max(2,2*c)),_=n(e,`${i}-translation-partials`,`float32`,s*l*3),v=t.pairCorrelation,y=Math.fround(v?.bandwidth??1);return[r(e,{id:`${i}-translation-overlaps`,operation:`GPUInhomogeneousPointPattern`,variant:`translation-overlaps`,invocationCount:Math.max(1,c),bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`boundary`,view:a,type:`f32`,access:`read`},{name:`intensity`,view:t.output.intensity,type:`f32`,access:`read`},{name:`radii`,view:t.radii,type:`f32`,access:`read`},p(o),{name:`pairs`,view:g,type:`f32`,access:`read_write`}],declarations:`${h}\n${u}\n${d}\n${f(m(o))}\nfn point(row: u32) -> vec2f {
    return vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  }\n${E(o.edges.length/4,t.maximumOverlapEdgePairs??65536)}`,body:`if (index >= ${c}u) { return; }
  var second = u32(floor((1.0 + sqrt(1.0 + 8.0 * f32(index))) * 0.5));
  if (second * (second - 1u) / 2u > index) { second--; }
  if ((second + 1u) * second / 2u <= index) { second++; }
  let first = index - second * (second - 1u) / 2u;
  pairs[pairsOffset + 2u * index] = -1.0;
  pairs[pairsOffset + 2u * index + 1u] = 0.0;
  if (boundary[boundaryOffset + first] < 0.0 || boundary[boundaryOffset + second] < 0.0) { return; }
  let displacement = point(second) - point(first);
  let separation = length(displacement);
  pairs[pairsOffset + 2u * index] = separation;
  if (separation > radii[radiiOffset + ${l-1}u] + ${v?y:0}f) { return; }
  let firstIntensity = intensity[intensityOffset + first];
  let secondIntensity = intensity[intensityOffset + second];
  if (!(firstIntensity > 0.0 && secondIntensity > 0.0) || !isFiniteFloat(firstIntensity) || !isFiniteFloat(secondIntensity)) { return; }
  let overlap = polygonTranslationOverlap(displacement);
  var weight = -4.0;
  if (overlap == -1.0) { weight = -3.0; }
  if (overlap > 0.0) {
    weight = (1.0 / firstIntensity) / secondIntensity / overlap;
    if (!isFiniteFloat(weight)) { weight = -4.0; }
  }
  pairs[pairsOffset + 2u * index + 1u] = weight;`}),r(e,{id:`${i}-translation-partials`,operation:`GPUInhomogeneousPointPattern`,variant:v?`kernel-pcf-partials`:`translation-k-partials`,invocationCount:s*l,bindings:[{name:`boundary`,view:a,type:`f32`,access:`read`},{name:`intensity`,view:t.output.intensity,type:`f32`,access:`read`},{name:`radii`,view:t.radii,type:`f32`,access:`read`},{name:`pairs`,view:g,type:`f32`,access:`read`},{name:`partials`,view:_,type:`f32`,access:`read_write`}],declarations:`${h}
fn correlationKernel(offset: f32) -> f32 {
  let standardized = offset / ${y}f;
  return 0.75 / ${y}f * max(0.0, 1.0 - standardized * standardized);
}`,body:`let focus = index % ${s}u;
  let radius = radii[radiiOffset + index / ${s}u];
  var code = 0.0; var count = 0.0; var correlation = 0.0;
  let lambda = intensity[intensityOffset + focus];
  if (boundary[boundaryOffset + focus] < 0.0) { code = 5.0; }
  else if (!(lambda > 0.0) || !isFiniteFloat(lambda)) { code = 2.0; }
  else {
    for (var other = 0u; other < ${s}u; other++) {
      if (other == focus) { continue; }
      let second = max(focus, other); let first = min(focus, other);
      let pair = second * (second - 1u) / 2u + first;
      let separation = pairs[pairsOffset + 2u * pair];
      if (separation < 0.0 || separation > radius + ${v?y:0}f) { continue; }
      let weight = pairs[pairsOffset + 2u * pair + 1u];
      if (weight < 0.0) { code = max(code, -weight); continue; }
      if (separation <= radius) { count += weight; }
      ${v?`
      let divisor = ${v.divisor===`distance`?`separation`:`radius`};
      if (!(divisor > 0.0)) { code = max(code, 2.0); continue; }
      var kernel = correlationKernel(radius - separation);
      ${v.zeroCorrection===`reflection`?`kernel += correlationKernel(radius + separation);`:``}
      correlation += weight * kernel / (6.283185307179586 * divisor);`:``}
    }
  }
  partials[partialsOffset + index * 3u] = count;
  partials[partialsOffset + index * 3u + 1u] = correlation;
  partials[partialsOffset + index * 3u + 2u] = code;`}),r(e,{id:`${i}-translation-reduce`,operation:`GPUInhomogeneousPointPattern`,variant:v?`kernel-pcf-reduce`:`translation-k-reduce`,invocationCount:l*256,workgroupSize:256,guardIndex:!1,bindings:[{name:`partials`,view:_,type:`f32`,access:`read`},{name:`radii`,view:t.radii,type:`f32`,access:`read`},{name:`values`,view:t.output.k,type:`f32`,access:`read_write`},{name:`correlation`,view:t.output.pairCorrelation,type:`f32`,access:`read_write`},{name:`status`,view:t.output.status,type:`u32`,access:`read_write`}],declarations:`${h}
var<workgroup> partialCounts: array<f32, 256>;
var<workgroup> partialCorrelations: array<f32, 256>;
var<workgroup> partialCodes: array<u32, 256>;
var<workgroup> partialIncluded: array<u32, 256>;`,body:`if (workgroupIndex >= ${l}u) { return; }
  let radiusIndex = workgroupIndex;
  var count = 0.0; var pairCorrelation = 0.0; var code = 0u; var included = 0u;
  for (var focus = localInvocationIndex; focus < ${s}u; focus += 256u) {
    let base = partialsOffset + (radiusIndex * ${s}u + focus) * 3u;
    let partialCode = u32(partials[base + 2u]);
    if (partialCode == 5u) { continue; }
    included++;
    code = max(code, partialCode);
    count += partials[base]; pairCorrelation += partials[base + 1u];
  }
  partialCounts[localInvocationIndex] = count;
  partialCorrelations[localInvocationIndex] = pairCorrelation;
  partialCodes[localInvocationIndex] = code;
  partialIncluded[localInvocationIndex] = included;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (localInvocationIndex < stride) {
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
      partialCorrelations[localInvocationIndex] += partialCorrelations[localInvocationIndex + stride];
      partialCodes[localInvocationIndex] = max(partialCodes[localInvocationIndex], partialCodes[localInvocationIndex + stride]);
      partialIncluded[localInvocationIndex] += partialIncluded[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex != 0u) { return; }
  count = partialCounts[0]; pairCorrelation = partialCorrelations[0];
  code = partialCodes[0]; included = partialIncluded[0];
  let radius = radii[radiiOffset + radiusIndex];
  if (!(radius > 0.0) || !isFiniteFloat(radius) || included == 0u || !isFiniteFloat(count) || !isFiniteFloat(pairCorrelation)) { code = 2u; }
  for (var radiusRow = 0u; radiusRow < ${l}u; radiusRow++) {
    let candidate = radii[radiiOffset + radiusRow];
    if (!isFiniteFloat(candidate) || !(candidate > 0.0)) { code = 2u; }
    if (radiusRow > 0u && !(candidate > radii[radiiOffset + radiusRow - 1u])) { code = 2u; }
  }
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  values[valuesOffset + radiusIndex] = select(invalid, count, code == 0u);
  correlation[correlationOffset + radiusIndex] = select(invalid, pairCorrelation, code == 0u);
  status[statusOffset + radiusIndex] = code;`}),...v?[]:[r(e,{id:`${i}-translation-annular`,operation:`GPUInhomogeneousPointPattern`,variant:`translation-annular-pcf`,invocationCount:l,bindings:[{name:`radii`,view:t.radii,type:`f32`,access:`read`},{name:`values`,view:t.output.k,type:`f32`,access:`read`},{name:`correlation`,view:t.output.pairCorrelation,type:`f32`,access:`read_write`}],body:`var previous = 0.0; var previousRadius = 0.0;
  if (index > 0u) { previous = values[valuesOffset + index - 1u]; previousRadius = radii[radiiOffset + index - 1u]; }
  let radius = radii[radiiOffset + index];
  correlation[correlationOffset + index] = (values[valuesOffset + index] - previous) / (3.141592653589793 * (radius * radius - previousRadius * previousRadius));`})]]}var O=4096,k=256,A=1048576,j=class{id;props;spatialContext;constructor(t){let n=t.id??`inhomogeneous-point-pattern`;this.id=n,this.props=t,this.spatialContext=i(n,t.spatialContext);let r=t.positions.length,s=t.radii.length;o(n,`positions rows`,r,1,O),o(n,`radii rows`,s,1,k),a(n,t.studyWindow.kind===`polygon`&&t.studyWindow.spatialContext.coordinateSpace===`planar`,`studyWindow must be a prepared planar polygon window`),a(n,t.selfContribution===void 0||[`include`,`exclude`].includes(t.selfContribution),`selfContribution must be 'include' or 'exclude'`),a(n,t.edgeCorrection===void 0||[`border`,`translation`].includes(t.edgeCorrection),`edgeCorrection must be 'border' or 'translation'`),t.pairCorrelation&&(a(n,t.edgeCorrection===`translation`&&t.pairCorrelation.kernel===`epanechnikov`,`pairCorrelation needs edgeCorrection 'translation' and the 'epanechnikov' kernel`),_(t.pairCorrelation.bandwidth,n,`pairCorrelation.bandwidth`),a(n,t.pairCorrelation.divisor===void 0||[`radius`,`distance`].includes(t.pairCorrelation.divisor),`pairCorrelation.divisor must be 'radius' or 'distance'`),a(n,t.pairCorrelation.zeroCorrection===void 0||[`none`,`reflection`].includes(t.pairCorrelation.zeroCorrection),`pairCorrelation.zeroCorrection must be 'none' or 'reflection'`)),t.maximumPairCount!==void 0&&a(n,Number.isSafeInteger(t.maximumPairCount)&&t.maximumPairCount>=r*(r-1)/2,`maximumPairCount must be an integer of at least rows * (rows - 1) / 2`),t.maximumOverlapEdgePairs!==void 0&&o(n,`maximumOverlapEdgePairs`,t.maximumOverlapEdgePairs,1,A),t.suppliedIntensity||_(t.bandwidth??NaN,n,`bandwidth`),g(t.positions,`float32x2`,r,n),g(t.radii,`float32`,s,n),t.suppliedIntensity&&g(t.suppliedIntensity,`float32`,r,n),g(t.output.intensity,`float32`,r,n),g(t.output.k,`float32`,s,n),g(t.output.pairCorrelation,`float32`,s,n),g(t.output.status,`uint32`,s,n),e(this.id,t.output,[...t.edgeCorrection===`translation`?[[t.output.k,t.output.pairCorrelation,t.output.status]]:[{read:[t.output.intensity],write:[t.output.k,t.output.status]},{read:[t.output.k],write:[t.output.pairCorrelation]}]]),l(this.id,Object.values(t.output),[t.positions,t.radii,t.suppliedIntensity])}getCommandNodes(e){let{id:i,props:a}=this,o=a.positions.length,g=a.studyWindow.getViews(e);c(i,e,[a.positions,a.radii,a.suppliedIntensity,g.packed,...Object.values(a.output)]),l(i,Object.values(a.output),[g.packed]),t(e,i,{positions:a.positions,radii:a.radii,suppliedIntensity:a.suppliedIntensity,window:g.packed,...Object.fromEntries(Object.entries(a.output).map(([e,t])=>[`output.${e}`,t]))});for(let[t,n]of[[`boundary distances`,o*4],[`border terms`,a.edgeCorrection===`translation`?0:o*a.radii.length*4],[`translation pairs`,a.edgeCorrection===`translation`?Math.max(2,o*(o-1))*4:0],[`translation partials`,a.edgeCorrection===`translation`?o*a.radii.length*3*4:0]])s(e.device,i,t,n,`reduce positions or radii rows`);let _=n(e,`${i}-boundary`,`float32`,o),v=`${u}\n${d}\n${f(m(g,`epanechnikov`))}`,y=`fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,b=Math.fround(a.bandwidth??1),x=r(e,{id:`${i}-intensity`,operation:`GPUInhomogeneousPointPattern`,variant:a.suppliedIntensity?`supplied`:`estimated`,invocationCount:o,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},p(g),...a.suppliedIntensity?[{name:`supplied`,view:a.suppliedIntensity,type:`f32`,access:`read`}]:[],{name:`intensity`,view:a.output.intensity,type:`f32`,access:`read_write`},{name:`boundary`,view:_,type:`f32`,access:`read_write`}],declarations:`${h}\n${v}\n${y}`,body:`let focus = point(index);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  var estimate = invalid;
  var boundaryDistance = -1.0;
  if (isStudyWindowUsable() && isInsideStudyWindow(focus)) {
    boundaryDistance = getStudyWindowBoundaryDistance(focus, 3.402823e38);
    ${a.suppliedIntensity?`estimate = supplied[suppliedOffset + index];`:`
    var total = 0.0;
    for (var row = 0u; row < ${o}u; row++) {
      ${a.selfContribution===`include`?``:`if (row == index) { continue; }`}
      let source = point(row);
      if (!isInsideStudyWindow(source)) { continue; }
      let delta = (focus - source) / ${b}f;
      total += max(0.0, 1.0 - dot(delta, delta));
    }
    let mass = getStudyWindowKernelMass(focus, ${b}f);
    if (mass > 0.0) { estimate = total * (2.0 / 3.141592653589793) / ${b}f / ${b}f / mass; }`}
  }
  intensity[intensityOffset + index] = estimate;
  boundary[boundaryOffset + index] = boundaryDistance;`});if(a.edgeCorrection===`translation`)return[x,...D(e,a,i,_,g)];let S=a.radii.length,C=n(e,`${i}-border-k-values`,`float32`,S*o);return[x,r(e,{id:`${i}-border-k-terms`,operation:`GPUInhomogeneousPointPattern`,variant:`border-k-terms`,invocationCount:S*o,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`intensity`,view:a.output.intensity,type:`f32`,access:`read`},{name:`boundary`,view:_,type:`f32`,access:`read`},{name:`radii`,view:a.radii,type:`f32`,access:`read`},{name:`terms`,view:C,type:`f32`,access:`read_write`}],declarations:`${h}\n${y}`,body:`let radiusIndex = index / ${o}u;
  let focus = index % ${o}u;
  let radius = radii[radiiOffset + radiusIndex];
  let focusIntensity = intensity[intensityOffset + focus];
  var term = 0.0;
  if (boundary[boundaryOffset + focus] >= 0.0 && isFiniteFloat(focusIntensity) && focusIntensity > 0.0 && boundary[boundaryOffset + focus] > radius) {
    var neighbors = 0.0;
    for (var row = 0u; row < ${o}u; row++) {
      if (row == focus || boundary[boundaryOffset + row] < 0.0) { continue; }
      let sourceIntensity = intensity[intensityOffset + row];
      if (sourceIntensity > 0.0 && distance(point(focus), point(row)) <= radius) { neighbors += 1.0 / sourceIntensity; }
    }
    term = neighbors / focusIntensity;
  }
  terms[termsOffset + index] = term;`}),r(e,{id:`${i}-border-k`,operation:`GPUInhomogeneousPointPattern`,variant:`border-k`,invocationCount:S,bindings:[{name:`intensity`,view:a.output.intensity,type:`f32`,access:`read`},{name:`boundary`,view:_,type:`f32`,access:`read`},{name:`radii`,view:a.radii,type:`f32`,access:`read`},{name:`terms`,view:C,type:`f32`,access:`read`},{name:`result`,view:a.output.k,type:`f32`,access:`read_write`},{name:`status`,view:a.output.status,type:`u32`,access:`read_write`}],declarations:h,body:`let radius = radii[radiiOffset + index];
  var valid = isFiniteFloat(radius) && radius > 0.0;
  if (index > 0u) { valid = valid && radius > radii[radiiOffset + index - 1u]; }
  var denominator = 0.0;
  var numerator = 0.0;
  var included = 0u;
  for (var focus = 0u; focus < ${o}u; focus++) {
    if (boundary[boundaryOffset + focus] < 0.0) { continue; }
    included++;
    let focusIntensity = intensity[intensityOffset + focus];
    if (!isFiniteFloat(focusIntensity) || focusIntensity <= 0.0) { valid = false; continue; }
    if (boundary[boundaryOffset + focus] <= radius) { continue; }
    denominator += 1.0 / focusIntensity;
    numerator += terms[termsOffset + index * ${o}u + focus];
  }
  valid = valid && included > 0u && isFiniteFloat(numerator) && isFiniteFloat(denominator);
  var code = select(2u, select(1u, 0u, denominator > 0.0), valid);
  var value = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (code == 0u) { value = numerator / denominator; }
  result[resultOffset + index] = value;
  status[statusOffset + index] = code;`}),r(e,{id:`${i}-pair-correlation`,operation:`GPUInhomogeneousPointPattern`,variant:`annular-pcf`,invocationCount:a.radii.length,bindings:[{name:`radii`,view:a.radii,type:`f32`,access:`read`},{name:`values`,view:a.output.k,type:`f32`,access:`read`},{name:`result`,view:a.output.pairCorrelation,type:`f32`,access:`read_write`}],body:`var previousRadius = 0.0;
  var previousValue = 0.0;
  if (index > 0u) { previousRadius = radii[radiiOffset + index - 1u]; previousValue = values[valuesOffset + index - 1u]; }
  let radius = radii[radiiOffset + index];
  result[resultOffset + index] = (values[valuesOffset + index] - previousValue) / (3.141592653589793 * (radius * radius - previousRadius * previousRadius));`})]}},M=Float32Array.from({length:24},(e,t)=>.01+t*.01),N=[.04,.08,.4,.44],P=[.04,.1,.16,.24,.4],F=y({parameters:w,caption:`Event intensity (color) and the K function (inset) against complete spatial randomness.`,createInputs:({random:e,world:t})=>{let n=b(e,t,260,3,.25),[r]=C(n.positions,t.width*.7,.5,1);return{world:t,events:n,anchor:r}},build(e,{world:t,events:n},r){let i=new v({positions:e.upload(`window-positions`,`float32x2`,Float32Array.of(0,0,t.width,0,t.width,1,0,1,0,0)).view,ringOffsets:e.upload(`window-rings`,`uint32`,Uint32Array.of(0,5)).view,polygonOffsets:e.upload(`window-polygons`,`uint32`,Uint32Array.of(0,1)).view,featureOffsets:e.upload(`window-features`,`uint32`,Uint32Array.of(0,1)).view,samplingGrid:[32,32]});e.graph.add(i);let a=e.upload(`events`,`float32x2`,n.positions),o=e.allocate(`intensity`,`float32`,n.count),s=e.allocate(`k`,`float32`,M.length);e.graph.add(new j({positions:a.view,studyWindow:i,bandwidth:r.bandwidth,edgeCorrection:r.edgeCorrection===0?`border`:`translation`,selfContribution:r.selfContribution===0?`include`:`exclude`,radii:e.upload(`radii`,`float32`,M).view,output:{intensity:o.view,k:s.view,pairCorrelation:e.allocate(`pair-correlation`,`float32`,M.length).view,status:e.allocate(`status`,`uint32`,M.length).view}}));let c=Float32Array.from(M,e=>Math.PI*e*e);return{outputs:{positions:a,intensity:o,k:s,random:e.upload(`csr`,`float32`,c)}}},buildCache:5,rebuildOn:[`bandwidth`,`edgeCorrection`,`selfContribution`],interaction:{kind:`reach`,hint:`stretch the bandwidth`,rest:({inputs:e,toUnit:t})=>{let{positions:n}=e.events;return t([n[e.anchor*2]+.25,n[e.anchor*2+1]])},resolve(e,{inputs:t}){let{positions:n}=t.events,r=[n[t.anchor*2],n[t.anchor*2+1]],i=Math.hypot(e.x-r[0],e.y-r[1]),a=P[Math.min(P.length-1,Math.floor(i/.1))];return{settle:{bandwidth:x(S(w,`bandwidth`),a)},glyphs:[{shape:`ring`,at:r,radius:a},{shape:`dot`,at:r}],readout:`bandwidth ${a.toFixed(2)}`}}},draw:({outputs:e})=>[{layer:`dots`,positions:e.positions,values:e.intensity,radius:4},{layer:`chart`,kind:`line`,context:!0,values:e.random,range:[0,.2],rectangle:N,color:[.7,.7,.7,1]},{layer:`chart`,kind:`line`,values:e.k,range:[0,.2],rectangle:N,color:[1,.85,.3,1]}]});export{F as default};
import{a as e,i as t,n,r,t as i}from"./raster-time-series-chunks-b-CPab_8L5.js";import{T as a,f as o,g as s}from"./gpu-contributor-utils-BIwLQON5.js";import{t as c}from"./raster-kernel-utils-DcIN7G5I.js";import{a as l,i as u,o as d,r as f}from"./raster-irregular-date-axis-B315UxmE.js";var p=1e7;function m(e,t=new Float32Array(2)){return s(`getGPURasterWhittakerSmootherParameterValues`,`smoothingParameter`,e.smoothingParameter,0,p),t[0]=e.smoothingParameter,t[1]=e.minimumWeight??.01,t}var h=class{id;props;chunkPlan;estimatedCommandNodeCount;differenceOrder;reweightPassCount;reweightFunction;ticksPerYear;constructor(t){this.props=t;let r=t.id??`raster-whittaker-smoother`;this.id=r;let i=t.differenceOrder??2;this.differenceOrder=i;let s=t.reweightPassCount??3;this.reweightPassCount=s,this.reweightFunction=t.reweightFunction??`bisquare`,o(r,`cellCount`,t.cellCount),o(r,`sliceCount`,t.sliceCount,2,256),o(r,`differenceOrder`,i,1,3),a(r,t.sliceCount>i,`sliceCount must exceed differenceOrder ${i}`),a(r,Number.isInteger(s)&&s>=0&&s<=8,`reweightPassCount must be an integer in [0, 8]`),a(r,this.reweightFunction===`bisquare`||this.reweightFunction===`upper-envelope`,`reweightFunction must be 'bisquare' or 'upper-envelope'`);let c=t.cellCount*t.sliceCount;o(r,`cellCount times sliceCount`,c),a(r,[t.values,t.mask,t.qualityWeights].every(e=>!e||e.length>=c)&&t.output.smoothed.length>=c,`values, mask, qualityWeights and output.smoothed must hold cellCount times sliceCount entries`),a(r,t.output.status.length>=t.cellCount,`output.status must hold one entry per cell`),a(r,t.parameters.length>=2,`parameters must hold 2 entries`),this.ticksPerYear=t.dateAxis?l(r,t.dateAxis,t.sliceCount):0,this.chunkPlan=n(this.id,t.cellCount,{values:{view:t.values,entriesPerCell:t.sliceCount},mask:{view:t.mask,entriesPerCell:t.sliceCount},qualityWeights:{view:t.qualityWeights,entriesPerCell:t.sliceCount},"output.smoothed":{view:t.output.smoothed,entriesPerCell:t.sliceCount},"output.status":{view:t.output.status,entriesPerCell:1}}),this.estimatedCommandNodeCount=this.chunkPlan.estimatedCommandNodeCount,e(this.id,this.getInputs(),this.getOutputs(),this.chunkPlan)}getInputs(){let{values:e,mask:t,qualityWeights:n,parameters:r}=this.props;return[...this.props.dateAxis?[this.props.dateAxis.acquisitionTimes]:[],e,r,...t?[t]:[],...n?[n]:[]]}getOutputs(){return[this.props.output.smoothed,this.props.output.status]}getCommandNodes(e){let{props:n,id:a,differenceOrder:o,reweightPassCount:s,reweightFunction:l}=this,{sliceCount:p,values:m,mask:h,qualityWeights:g,parameters:_,dateAxis:v,output:y}=n;t(a,e,this.getInputs(),this.getOutputs()),v&&d(a,e,v);let b=o+1,x=[r(`values`,m),r(`parameters`,_),...h?[r(`mask`,h)]:[],...g?[r(`qualityWeights`,g)]:[]],S=v?`(rasterYearsBetween(row + offset + level, row + offset) / axisMeanSpacing)`:`f32(level)`;return i(e,this.chunkPlan,{id:a,operation:`GPURasterWhittakerSmoother`,invocationCount:n.cellCount,workgroupSize:8,bindings:[...x,...v?[f(v)]:[],r(`smoothed`,y.smoothed,!0),r(`status`,y.status,!0)],declarations:`${c}
${v?u(this.ticksPerYear,p):``}
fn baseWeight(cell: u32, slice: u32) -> f32 {
  let source = cell * ${p}u + slice;
  if (!finite(values[valuesOffset + source]) ${h?`|| mask[maskOffset + source] == 0u`:``}) { return 0.0; }
  return ${g?`clamp(qualityWeights[qualityWeightsOffset + source], 0.0, 1.0)`:`1.0`};
}
fn siftDown(heap: ptr<function, array<f32, ${p}>>, start: u32, size: u32) {
  var parent = start;
  loop {
    let left = 2u * parent + 1u;
    if (left >= size) { break; }
    var largest = left;
    if (left + 1u < size && (*heap)[left + 1u] > (*heap)[left]) { largest = left + 1u; }
    if ((*heap)[parent] >= (*heap)[largest]) { break; }
    let swapped = (*heap)[parent]; (*heap)[parent] = (*heap)[largest]; (*heap)[largest] = swapped;
    parent = largest;
  }
}
fn getMedian(heap: ptr<function, array<f32, ${p}>>, count: u32) -> f32 {
  for (var node = count / 2u; node > 0u; node--) { siftDown(heap, node - 1u, count); }
  for (var end = count; end > 1u; end--) {
    let swapped = (*heap)[0]; (*heap)[0] = (*heap)[end - 1u]; (*heap)[end - 1u] = swapped;
    siftDown(heap, 0u, end - 1u);
  }
  if (count % 2u == 1u) { return (*heap)[count / 2u]; }
  return 0.5 * ((*heap)[count / 2u - 1u] + (*heap)[count / 2u]);
}
// Rotates one row of the stacked least-squares matrix into the upper band R, where
// band[j * stride + k] holds R[j, j + k]. entries[k] is the value of the row in column row + k.
// Rows must arrive ordered by first column (weight row, then penalty row, for each slice), which
// keeps every rotation inside the band: a row is finished after at most order + 1 rotations.
fn insertRow(
  band: ptr<function, array<f32, ${p*b}>>,
  rightSide: ptr<function, array<f32, ${p}>>,
  row: u32,
  entries: array<f32, ${b}>,
  entryRightSide: f32
) {
  var work = entries;
  var workRightSide = entryRightSide;
  let last = min(row + ${o}u, ${p-1}u);
  for (var column = row; column <= last; column++) {
    let lead = work[0];
    let base = column * ${b}u;
    let diagonal = (*band)[base];
    if (lead != 0.0 && diagonal == 0.0) {
      for (var term = 0u; term <= ${o}u; term++) { (*band)[base + term] = work[term]; }
      (*rightSide)[column] = workRightSide;
      return;
    }
    var cosine = 1.0;
    var sine = 0.0;
    if (lead != 0.0) {
      let radius = sqrt(diagonal * diagonal + lead * lead);
      cosine = diagonal / radius;
      sine = lead / radius;
      (*band)[base] = radius;
    }
    for (var term = 1u; term <= ${o}u; term++) {
      let upper = (*band)[base + term];
      let lower = work[term];
      (*band)[base + term] = cosine * upper + sine * lower;
      work[term - 1u] = cosine * lower - sine * upper;
    }
    work[${o}u] = 0.0;
    let upperRightSide = (*rightSide)[column];
    (*rightSide)[column] = cosine * upperRightSide + sine * workRightSide;
    workRightSide = cosine * workRightSide - sine * upperRightSide;
  }
}`,body:`let smoothing = parameters[parametersOffset];
let minimumWeight = parameters[parametersOffset + 1u];
var band: array<f32, ${p*b}>;
var solution: array<f32, ${p}>;
var robust: array<f32, ${p}>;
var workspace: array<f32, ${p}>;
var support = 0u;
// The smoother is exact on constants (every difference of order one or more removes them), so the
// solve runs on values minus a per-cell reference and the reference is added back. Without this
// the float32 error grows with the magnitude of the data instead of its variation.
var reference = 0.0;
for (var slice = 0u; slice < ${p}u; slice++) {
  robust[slice] = 1.0;
  if (baseWeight(index, slice) > 0.0) {
    if (support == 0u) { reference = values[valuesOffset + index * ${p}u + slice]; }
    support++;
  }
}
if (support > 0u) {
  // Mean of the valid values, formed around the first one so the sum stays small.
  var offsetSum = 0.0;
  for (var slice = 0u; slice < ${p}u; slice++) {
    if (baseWeight(index, slice) > 0.0) { offsetSum += values[valuesOffset + index * ${p}u + slice] - reference; }
  }
  reference += offsetSum / f32(support);
}
let penaltyScale = sqrt(max(smoothing, 0.0));
${v?`let axisMeanSpacing = rasterYearsBetween(${p-1}u, 0u) / ${p-1}.0;`:``}
var code = 0u;
if (support < ${o+1}u) { code = 1u; }
if (code == 0u) {
  for (var iteration = 0u; iteration < ${s+1}u; iteration++) {
    for (var entry = 0u; entry < ${p*b}u; entry++) { band[entry] = 0.0; }
    for (var slice = 0u; slice < ${p}u; slice++) { solution[slice] = 0.0; }
    // Givens QR of the stacked matrix [sqrt(W); sqrt(lambda) D] instead of a factorisation of
    // W + lambda D'D: the condition number is the square root of the normal equations', which is
    // what float32 needs at large lambda.
    for (var row = 0u; row < ${p}u; row++) {
      let weight = baseWeight(index, row) * robust[row];
      if (weight > 0.0) {
        var weightRow: array<f32, ${b}>;
        weightRow[0] = sqrt(weight);
        insertRow(&band, &solution, row, weightRow, weightRow[0] * (values[valuesOffset + index * ${p}u + row] - reference));
      }
      if (row >= ${p-o}u) { continue; }
      var previous: array<array<f32, ${b}>, ${b}>;
      for (var offset = 0u; offset < ${b}u; offset++) { previous[offset][0] = 1.0; }
      for (var level = 1u; level <= ${o}u; level++) {
        var current: array<array<f32, ${b}>, ${b}>;
        for (var offset = 0u; offset <= ${o}u - level; offset++) {
          let span = ${S} / f32(level);
          for (var term = 0u; term <= level; term++) {
            var coefficient = 0.0;
            if (term > 0u) { coefficient += previous[offset + 1u][term - 1u]; }
            if (term < level) { coefficient -= previous[offset][term]; }
            current[offset][term] = coefficient / span;
          }
        }
        previous = current;
      }
      var penaltyRow = previous[0];
      for (var term = 0u; term <= ${o}u; term++) { penaltyRow[term] *= penaltyScale; }
      insertRow(&band, &solution, row, penaltyRow, 0.0);
    }
    // A diagonal of R that is zero or not finite means the system has no solution in float32
    // (for example lambda = 0 with a gap). The size of a diagonal against the largest one is not
    // tested: the penalty rows scale with sqrt(lambda), so that ratio is small for well-posed
    // series with few clustered observations, which this solve handles (see the class
    // documentation for the accuracy there).
    for (var row = 0u; row < ${p}u; row++) {
      let diagonal = abs(band[row * ${b}u]);
      if (!finite(diagonal) || !(diagonal > 0.0)) { code = 2u; }
    }
    if (code != 0u) { break; }
    // Back substitution on the upper band.
    for (var reverse = 0u; reverse < ${p}u; reverse++) {
      let row = ${p-1}u - reverse;
      var remainder = solution[row];
      for (var above = row + 1u; above <= min(row + ${o}u, ${p-1}u); above++) { remainder -= band[row * ${b}u + (above - row)] * solution[above]; }
      solution[row] = remainder / band[row * ${b}u];
      if (!finite(solution[row])) { code = 2u; }
    }
    if (code != 0u) { break; }
    if (iteration == ${s}u) { break; }
    var residualCount = 0u;
    var largestNegative = 0.0;
    for (var slice = 0u; slice < ${p}u; slice++) {
      if (baseWeight(index, slice) <= 0.0) { continue; }
      let residual = (values[valuesOffset + index * ${p}u + slice] - reference) - solution[slice];
      workspace[residualCount] = abs(residual); residualCount++;
      if (residual < 0.0) { largestNegative = max(largestNegative, -residual); }
    }
    let scale = 6.0 * getMedian(&workspace, residualCount);
    for (var slice = 0u; slice < ${p}u; slice++) {
      if (baseWeight(index, slice) <= 0.0) { continue; }
      let residual = (values[valuesOffset + index * ${p}u + slice] - reference) - solution[slice];
      ${l===`bisquare`?`var nextRobust = 1.0;
      if (scale > 0.0) { let ratio = abs(residual) / scale; nextRobust = select(minimumWeight, (1.0 - ratio * ratio) * (1.0 - ratio * ratio), ratio < 1.0); }`:`var nextRobust = 1.0;
      if (residual < 0.0 && largestNegative > 0.0) { nextRobust = 1.0 - abs(residual) / largestNegative; }`}
      robust[slice] = max(nextRobust, minimumWeight);
    }
  }
}
status[statusOffset + index] = code;
for (var slice = 0u; slice < ${p}u; slice++) {
  smoothed[smoothedOffset + index * ${p}u + slice] = select(missing(), solution[slice] + reference, code == 0u);
}`})}};export{m as n,h as t};
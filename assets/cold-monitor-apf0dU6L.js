import{a as e,i as t,n,t as r}from"./raster-time-series-chunks-FYvfnBw_.js";import{T as i,f as a,g as o,p as s}from"./gpu-contributor-utils-BIwLQON5.js";import{a as c,i as l,t as u}from"./raster-kernel-utils-DcIN7G5I.js";import{a as d,i as f,r as p}from"./raster-irregular-date-axis-B315UxmE.js";import{t as m}from"./raster-harmonic-trend-design-wgsl-DiyAnFyi.js";import{t as h}from"./define-run-B-brkCe4.js";import{c as g}from"./pointer-overlay-Qh8qflcA.js";import{a as _,b as v,c as y,d as b,g as x,m as S,n as C,o as w,p as T,r as E,t as D,u as O,v as k,x as A}from"./charts-Br0w6lsw.js";import{n as j,r as M,t as N}from"./probe-DLtMtDEc.js";import{r as P,t as F}from"./readback-C6WbEPkA.js";import{Yr as I}from"./entry-preview-D70TdSbK.js";var L=2048,R=16777216,z=4294967295;function B(e,t=new Float32Array(6+e.minimumRMSE.length)){let n=`getGPURasterCOLDMonitorParameterValues`;return o(n,`changeThreshold`,e.changeThreshold,0),o(n,`outlierThreshold`,e.outlierThreshold,0),t[0]=e.period??1,t[1]=e.changeThreshold,t[2]=e.outlierThreshold,t[3]=e.minimumMeanCosine??Math.SQRT1_2,t[4]=e.minimumSpanYears??1,t[5]=e.maximumInitializationGapYears??1,t.set(e.minimumRMSE,6),t}var V=class{props;id;estimatedCommandNodeCount;chunkProps;minimumObservations;consecutiveCount;maximumCoefficientCount;maximumSegmentCount;maximumFitCount;ticksPerYear;constructor(t){this.props=t;let r=t.id??`raster-cold-monitor`;this.id=r,this.minimumObservations=t.minimumObservations??12,this.consecutiveCount=t.consecutiveCount??6,this.maximumCoefficientCount=t.maximumCoefficientCount??8,this.maximumSegmentCount=t.maximumSegmentCount??8,this.maximumFitCount=t.maximumFitCount??96;let{sliceCount:o,bandCount:c}=t;a(r,`cellCount`,t.cellCount),a(r,`sliceCount`,o,1,256),a(r,`bandCount`,c,1,8),a(r,`minimumObservations`,this.minimumObservations,5,o),a(r,`consecutiveCount`,this.consecutiveCount,2,16),i(r,[4,6,8].includes(this.maximumCoefficientCount),`maximumCoefficientCount must be 4, 6 or 8`),a(r,`maximumSegmentCount`,this.maximumSegmentCount,1,32);let l=this.maximumCoefficientCount;a(r,`maximumFitCount`,this.maximumFitCount,1,Math.floor(R/(o*l*l*2*c)));let u=Math.max(o*l+l*l+2*l+8,o*l+3*o+Math.ceil(o/32)+2*c*8+2*c*this.consecutiveCount+64);i(r,u<=L,`per-thread working memory (sliceCount ${o}, ${l} coefficients, ${c} bands) must fit ${L} floats`);let f=t.cellCount*o,p=t.cellCount*this.maximumSegmentCount;a(r,`cellCount times sliceCount`,f),i(r,t.values.length>=f*c,`values must hold cellCount times sliceCount times bandCount entries`),i(r,!t.mask||t.mask.length>=f,`mask must hold cellCount times sliceCount entries`),i(r,t.parameters.length>=6+c,`parameters must hold ${6+c} entries`),i(r,t.output.segments.length>=p*4,`output.segments must hold 4 entries per segment`),i(r,t.output.coefficients.length>=p*c*8,`output.coefficients must hold 8 entries per segment and band`),i(r,t.output.segmentStatistics.length>=p*c*2,`output.segmentStatistics must hold two entries per segment and band`),i(r,t.output.cellSummary.length>=t.cellCount*3,`output.cellSummary must hold three entries per cell`),this.ticksPerYear=d(r,t.dateAxis,o),this.chunkProps=n(r,t,{values:t.sliceCount*t.bandCount,mask:t.sliceCount,"output.segments":this.maximumSegmentCount*4,"output.coefficients":this.maximumSegmentCount*t.bandCount*8,"output.segmentStatistics":this.maximumSegmentCount*t.bandCount*2,"output.cellSummary":3}),this.estimatedCommandNodeCount=this.chunkProps.length*1,s(r,this.estimatedCommandNodeCount),e(r,this.getInputs(),this.getOutputs())}getInputs(){let{values:e,mask:t,parameters:n}=this.props;return[e,n,...t?[t]:[],...this.props.dateAxis?[this.props.dateAxis.acquisitionTimes]:[]]}getOutputs(){let{segments:e,coefficients:t,segmentStatistics:n,cellSummary:r}=this.props.output;return[e,t,n,r]}getCommandNodes(e){return t(e,this.id,[...this.getInputs(),...this.getOutputs()]),this.chunkProps.flatMap((t,n)=>this.getChunkCommandNodes(e,{...t,id:`${this.id}-chunk-${n}`}))}getChunkCommandNodes(e,t){let n=t.id,{minimumObservations:i,consecutiveCount:a,maximumCoefficientCount:o,maximumSegmentCount:s,maximumFitCount:d}=this,{sliceCount:h,bandCount:g,values:_,mask:v,parameters:y,dateAxis:b,output:x}=t,S=Math.ceil(h/32),C=`${u}
${f(this.ticksPerYear,h)}
${m(h)}
${l(h,o,!0)}
var<private> cellIndex: u32;
var<private> observationSlices: array<u32, ${h}>;
var<private> observationTotal: u32;
var<private> includedMask: array<u32, ${S}>;
var<private> includedTotal: u32;
var<private> fitCoefficients: array<f32, ${g*8}>;
var<private> fitRMSE: array<f32, ${g}>;
var<private> fitsUsed: u32;
var<private> statusBits: u32;
fn observationValue(slice: u32, band: u32) -> f32 {
  return values[valuesOffset + (cellIndex * ${h}u + slice) * ${g}u + band];
}
fn harmonicPeriod() -> f32 { return parameters[parametersOffset]; }
fn minimumRMSEOf(band: u32) -> f32 { return parameters[parametersOffset + 6u + band]; }
fn clearIncluded() {
  for (var word = 0u; word < ${S}u; word++) { includedMask[word] = 0u; }
  includedTotal = 0u;
}
fn addIncluded(position: u32) {
  includedMask[position / 32u] |= 1u << (position % 32u);
  includedTotal++;
}
fn isIncluded(position: u32) -> bool {
  return (includedMask[position / 32u] & (1u << (position % 32u))) != 0u;
}
fn getCoefficientCountFor(count: u32) -> u32 {
  // Four coefficients below 18 observations, six below 24, eight above (Zhu et al. 2020).
  return min(${o}u, select(select(4u, 6u, count >= 18u), 8u, count >= 24u));
}
fn predictBand(slice: u32, band: u32) -> f32 {
  var prediction = 0.0;
  for (var column = 0u; column < 8u; column++) { prediction += fitCoefficients[band * 8u + column] * getHarmonicTrendColumn(slice, column, harmonicPeriod()); }
  return prediction;
}
// Refits every band on the included observations; false stops the cell (budget or rank).
fn fitIncluded(columnCount: u32) -> bool {
  if (fitsUsed >= ${d}u) { statusBits |= 8u; return false; }
  fitsUsed++;
  var refreshed: array<f32, ${g*8}>;
  var refreshedRMSE: array<f32, ${g}>;
  for (var band = 0u; band < ${g}u; band++) {
    var design: array<f32, ${h*o}>;
    var response: array<f32, ${h}>;
    for (var position = 0u; position < observationTotal; position++) {
      if (!isIncluded(position)) { continue; }
      let slice = observationSlices[position];
      response[slice] = observationValue(slice, band);
      for (var column = 0u; column < columnCount; column++) { design[column * ${h}u + slice] = getHarmonicTrendColumn(slice, column, harmonicPeriod()); }
    }
    let fit = leastSquares(design, response, columnCount);
    if (!fit.valid) { statusBits |= 2u; return false; }
    for (var column = 0u; column < columnCount; column++) { refreshed[band * 8u + column] = fit.coefficients[column]; }
    refreshedRMSE[band] = sqrt(fit.residual / f32(includedTotal - min(includedTotal, columnCount)));
  }
  fitCoefficients = refreshed;
  fitRMSE = refreshedRMSE;
  return true;
}
fn fillOutputRecord(segment: u32, startSlice: u32, endSlice: u32, breakSlice: u32, columnCount: u32, breakFound: bool, magnitudes: array<f32, ${g}>) {
  let base = (cellIndex * ${s}u + segment);
  segments[segmentsOffset + base * 4u] = startSlice;
  segments[segmentsOffset + base * 4u + 1u] = endSlice;
  segments[segmentsOffset + base * 4u + 2u] = breakSlice;
  segments[segmentsOffset + base * 4u + 3u] = columnCount;
  for (var band = 0u; band < ${g}u; band++) {
    let entry = base * ${g}u + band;
    segmentStatistics[segmentStatisticsOffset + entry * 2u] = fitRMSE[band];
    segmentStatistics[segmentStatisticsOffset + entry * 2u + 1u] = select(missing(), magnitudes[band], breakFound);
    for (var column = 0u; column < 8u; column++) { coefficients[coefficientsOffset + entry * 8u + column] = fitCoefficients[band * 8u + column]; }
  }
}
fn getMedian(sample: array<f32, ${a}>) -> f32 {
  var ordered = sample;
  for (var outer = 1u; outer < ${a}u; outer++) {
    let item = ordered[outer];
    var inner = outer;
    while (inner > 0u && ordered[inner - 1u] > item) { ordered[inner] = ordered[inner - 1u]; inner--; }
    ordered[inner] = item;
  }
  return ${a%2==1?`ordered[${(a-1)/2}u]`:`0.5 * (ordered[${a/2-1}u] + ordered[${a/2}u])`};
}`,w=`fillRasterSliceYears();
cellIndex = index;
let changeThreshold = parameters[parametersOffset + 1u];
let outlierThreshold = parameters[parametersOffset + 2u];
let minimumMeanCosine = parameters[parametersOffset + 3u];
let minimumSpanYears = parameters[parametersOffset + 4u];
let maximumInitializationGapYears = parameters[parametersOffset + 5u];
fitsUsed = 0u;
statusBits = 0u;
observationTotal = 0u;
for (var slice = 0u; slice < ${h}u; slice++) {
  var usable = ${v?`mask[maskOffset + index * ${h}u + slice] != 0u`:`true`};
  for (var band = 0u; band < ${g}u; band++) { usable = usable && finite(observationValue(slice, band)); }
  if (usable) { observationSlices[observationTotal] = slice; observationTotal++; }
}
var segment = 0u;
var cursor = 0u;
for (var unused = 0u; unused < ${s}u; unused++) {
  let base = (index * ${s}u + unused);
  for (var word = 0u; word < 4u; word++) { segments[segmentsOffset + base * 4u + word] = ${z}u; }
  for (var band = 0u; band < ${g}u; band++) {
    segmentStatistics[segmentStatisticsOffset + (base * ${g}u + band) * 2u] = missing();
    segmentStatistics[segmentStatisticsOffset + (base * ${g}u + band) * 2u + 1u] = missing();
    for (var column = 0u; column < 8u; column++) { coefficients[coefficientsOffset + (base * ${g}u + band) * 8u + column] = missing(); }
  }
}
loop {
  if (cursor >= observationTotal) { break; }
  if (segment == ${s}u) { statusBits |= 16u; break; }
  var start = cursor;
  var windowEnd = 0u;
  var initialized = false;
  loop {
    if (start + ${i}u > observationTotal || (statusBits & 8u) != 0u) { break; }
    var end = start + ${i}u;
    loop {
      if (end > observationTotal) { break; }
      if (sliceYears[observationSlices[end - 1u]] - sliceYears[observationSlices[start]] >= minimumSpanYears) { break; }
      end++;
    }
    if (end > observationTotal) { break; }
    var largestGap = 0.0;
    for (var position = start + 1u; position < end; position++) {
      largestGap = max(largestGap, sliceYears[observationSlices[position]] - sliceYears[observationSlices[position - 1u]]);
    }
    if (largestGap >= maximumInitializationGapYears) { start++; continue; }
    clearIncluded();
    for (var position = start; position < end; position++) { addIncluded(position); }
    if (!fitIncluded(4u)) { break; }
    let firstSlice = observationSlices[start];
    let lastSlice = observationSlices[end - 1u];
    let spanYears = sliceYears[lastSlice] - sliceYears[firstSlice];
    var score = 0.0;
    for (var band = 0u; band < ${g}u; band++) {
      let floorRMSE = max(fitRMSE[band], minimumRMSEOf(band));
      let firstResidual = abs(observationValue(firstSlice, band) - predictBand(firstSlice, band));
      let lastResidual = abs(observationValue(lastSlice, band) - predictBand(lastSlice, band));
      // The larger end residual, as pyxccd; the sum of the two rejects windows pyxccd accepts.
      let term = (abs(fitCoefficients[band * 8u + 1u]) * spanYears + max(firstResidual, lastResidual)) / floorRMSE;
      score += term * term;
    }
    if (score <= changeThreshold) { windowEnd = end; initialized = true; break; }
    start++;
  }
  if (!initialized) { break; }
  var columnCount = getCoefficientCountFor(includedTotal);
  if (!fitIncluded(columnCount)) { break; }
  var fittedCount = includedTotal;
  var position = windowEnd;
  var breakFound = false;
  var breakPosition = 0u;
  var breakMagnitudes: array<f32, ${g}>;
  loop {
    if (position + ${a}u > observationTotal) { break; }
    var normalized: array<f32, ${a*g}>;
    var residualSamples: array<f32, ${a*g}>;
    var allExceed = true;
    var firstScore = 0.0;
    for (var offset = 0u; offset < ${a}u; offset++) {
      let peekSlice = observationSlices[position + offset];
      var peekScore = 0.0;
      for (var band = 0u; band < ${g}u; band++) {
        let residual = observationValue(peekSlice, band) - predictBand(peekSlice, band);
        residualSamples[offset * ${g}u + band] = residual;
        let value = residual / max(fitRMSE[band], minimumRMSEOf(band));
        normalized[offset * ${g}u + band] = value;
        peekScore += value * value;
      }
      if (offset == 0u) { firstScore = peekScore; }
      allExceed = allExceed && peekScore > changeThreshold;
    }
    var cosineTotal = 0.0;
    for (var offset = 0u; offset + 1u < ${a}u; offset++) {
      var dotProduct = 0.0; var leftNorm = 0.0; var rightNorm = 0.0;
      for (var band = 0u; band < ${g}u; band++) {
        let left = normalized[offset * ${g}u + band];
        let right = normalized[(offset + 1u) * ${g}u + band];
        dotProduct += left * right; leftNorm += left * left; rightNorm += right * right;
      }
      cosineTotal += dotProduct / max(sqrt(leftNorm * rightNorm), 1e-12);
    }
    if (allExceed && cosineTotal / f32(${a-1}u) >= minimumMeanCosine) {
      breakFound = true;
      breakPosition = position;
      for (var band = 0u; band < ${g}u; band++) {
        var sample: array<f32, ${a}>;
        for (var offset = 0u; offset < ${a}u; offset++) { sample[offset] = residualSamples[offset * ${g}u + band]; }
        breakMagnitudes[band] = getMedian(sample);
      }
      break;
    }
    if (firstScore > outlierThreshold) { position++; continue; }
    addIncluded(position);
    position++;
    if (includedTotal * 100u >= fittedCount * 133u && includedTotal != fittedCount) {
      let refitColumnCount = getCoefficientCountFor(includedTotal);
      if (!fitIncluded(refitColumnCount)) { break; }
      columnCount = refitColumnCount;
      fittedCount = includedTotal;
    }
  }
  if (!breakFound) {
    for (var remaining = position; remaining < observationTotal; remaining++) { addIncluded(remaining); }
  }
  if (includedTotal != fittedCount) {
    let finalColumnCount = getCoefficientCountFor(includedTotal);
    if (fitIncluded(finalColumnCount)) { columnCount = finalColumnCount; }
    else if ((statusBits & 2u) != 0u) { break; }
  }
  var firstIncluded = 0u;
  var lastIncluded = 0u;
  var foundFirst = false;
  for (var scan = 0u; scan < observationTotal; scan++) {
    if (!isIncluded(scan)) { continue; }
    if (!foundFirst) { firstIncluded = scan; foundFirst = true; }
    lastIncluded = scan;
  }
  fillOutputRecord(segment, observationSlices[firstIncluded], observationSlices[lastIncluded], select(${z}u, observationSlices[breakPosition], breakFound), columnCount, breakFound, breakMagnitudes);
  segment++;
  if ((statusBits & 8u) != 0u || !breakFound) { break; }
  cursor = breakPosition;
}
if (segment == 0u) { statusBits |= 1u; }
cellSummary[cellSummaryOffset + index * 3u] = segment;
cellSummary[cellSummaryOffset + index * 3u + 1u] = statusBits;
cellSummary[cellSummaryOffset + index * 3u + 2u] = fitsUsed;`;return[r(e,{id:n,operation:`GPURasterCOLDMonitor`,invocationCount:t.cellCount,workgroupSize:8,bindings:[c(`values`,_),c(`parameters`,y),...v?[c(`mask`,v)]:[],p(b),c(`segments`,x.segments,!0),c(`coefficients`,x.coefficients,!0),c(`segmentStatistics`,x.segmentStatistics,!0),c(`cellSummary`,x.cellSummary,!0)],declarations:C,body:w})]}},H=96,U=62,W=4,G=6,K=8,q=120,[J]=E(1),Y=[-.2,1],X=h({parameters:I,rebuildOn:[`consecutiveCount`],buildCache:4,caption:`Synthetic series on measured Dixie NDVI levels and fire ratio, 96 irregular dates; one band.`,createInputs:({aspect:e,random:t})=>{let n=k(e,t,{sliceCount:H,yearCount:6,breakSlice:U});return{...n,observed:n.observe(.03,.05)}},build(e,t,n){let{grid:r,sliceCount:i}=t,a=e.upload(`series`,`float32`,t.observed),o=e.allocate(`parameters`,`float32`,7),s={segments:e.allocate(`segments`,`uint32`,r.cellCount*W*4),coefficients:e.allocate(`coefficients`,`float32`,r.cellCount*W*K),segmentStatistics:e.allocate(`segment-statistics`,`float32`,r.cellCount*W*2),cellSummary:e.allocate(`cell-summary`,`uint32`,r.cellCount*3)};return e.graph.add(new V({cellCount:r.cellCount,sliceCount:i,bandCount:1,values:a.view,parameters:o.view,dateAxis:A(e,t.times),consecutiveCount:Math.round(n.consecutiveCount),maximumCoefficientCount:G,maximumSegmentCount:W,output:{segments:s.segments.view,coefficients:s.coefficients.view,segmentStatistics:s.segmentStatistics.view,cellSummary:s.cellSummary.view}})),{outputs:{...s,blank:F(e,`blank`,r.cellCount),level:e.upload(`level`,`float32`,t.level),observations:D(e,`observations`,i),firstModel:D(e,`first-model`,240),laterModel:D(e,`later-model`,240)},update:e=>{t.observed=t.observe(e.cloudFraction,.05),a.write(t.observed),o.write(B({changeThreshold:e.changeThreshold,outlierThreshold:24,minimumRMSE:[.03]}))}}},async afterRun({outputs:e,inputs:t,upload:n}){let r=await P(e.cellSummary);return{map:n(`segment-class-map`,`float32`,Float32Array.from({length:t.grid.cellCount},(e,n)=>{if(!(t.level[n]>0)||r[n*3+1]!==0)return NaN;let i=r[n*3];return i<=1?NaN:i===2?1:2}))}},interaction:{kind:`probe`,hint:`probe a pixel`,rest:[.72,.5],resolve:(e,{inputs:t,outputs:n,values:r,world:i,peek:a})=>{let{grid:o,sliceCount:s}=t,c=M(o,i,e),l=t.observed.subarray(c.index*s,(c.index+1)*s),u=Array.from(t.years);x(n.observations,u,l);let d=a(n.segments),f=a(n.coefficients),p=a(n.cellSummary),m=[t.years[0],t.years[s-1]],h=[[[],[]],[[],[]]],_=[j(o,i,c),w(i,J),...S(i,J,m,b(t.times))],y=z,C=p?p[c.index*3]:0;if(d&&f)for(let e=0;e<Math.min(C,W);e++){let n=(c.index*W+e)*4,r=d[n],i=d[n+1],a=d[n+2];e===0&&(y=a);let o=(c.index*W+e)*K,l=h[e%2];for(let e=0;e<q;e++){let n=t.years[r]+(t.years[Math.min(i,s-1)]-t.years[r])*e/119,a=f[o]+f[o+1]*n;for(let e=1;e<=2;e++){let t=2*Math.PI*e*n;a+=f[o+2*e]*Math.cos(t)+f[o+2*e+1]*Math.sin(t)}l[0].push(n),l[1].push(a)}}x(n.firstModel,h[0][0],h[0][1]),x(n.laterModel,h[1][0],h[1][1]);let E=y!==4294967295&&y<s;return E&&_.push(T(i,J,m,t.years[y])),{spotlight:null,glyphs:_,readout:p?E?`break ${v(t.times[y])} · changeThreshold ${g(r.changeThreshold,1)} · ${N(o,c)}`:`no break · ${C} segment${C===1?``:`s`} · ${N(o,c)}`:`reading the segments`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>{let i=[e.years[0],e.years[e.sliceCount-1]];return[O(t.level,e.grid,e.grid.mapLeft,{context:!0,ramp:`sequential`,range:[0,1]}),O(n?.map??t.blank,e.grid,e.grid.mapLeft,{ramp:`categorical`,range:[0,7]}),_(t.observations,J,Y,i,y(r),3.5),_(t.firstModel,J,Y,i,C(r,0),1.8),_(t.laterModel,J,Y,i,C(r,1),1.8)]}});export{X as default};
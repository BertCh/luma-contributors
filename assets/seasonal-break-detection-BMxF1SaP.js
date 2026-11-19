import{a as e,i as t,n,r,t as i}from"./raster-time-series-chunks-b-CPab_8L5.js";import{c as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as o,t as s}from"./raster-kernel-utils-DcIN7G5I.js";import{a as c,i as l,o as u,r as d}from"./raster-irregular-date-axis-B315UxmE.js";import{T as f,f as p}from"./gpu-contributor-utils-BIwLQON5.js";import{t as m}from"./define-run-B-brkCe4.js";import{Oi as h}from"./entry-preview-D70TdSbK.js";import{i as g,r as _}from"./gestures-C1AoqRjN.js";import{t as v}from"./ndvi-37bSHUSy.js";function y(e,t,n,c,l=``){let u=n.sliceCount,f=n.monitoring?.historyLength??u,p=n.seasonality===`dummy`?n.period+2:5,m=p-1,h=n.seasonality===`dummy`?`return select(0.0, 1.0, slice % ${n.period}u == column - 1u);`:`let angle = 6.283185307179586 * ${n.dateAxis?`rasterSliceYears(slice)`:`f32(slice)`} / ${a(n.period)}; return select(sin(angle), cos(angle), column == 2u);`,g=n.dateAxis?`rasterSliceYears(slice)`:`f32(slice) / ${u}.0`;return i(e,c,{id:t,operation:`GPURasterSeasonalBreak`,variant:`extended-seasonal-decomposition-monitoring`,invocationCount:n.cellCount,workgroupSize:2,bindings:[r(`values`,n.values),...n.mask?[r(`mask`,n.mask)]:[],...n.dateAxis?[d(n.dateAxis)]:[],...Object.entries(n.output).filter(([,e])=>e!==void 0).map(([e,t])=>r(e,t,!0))],declarations:`${s}${o(f,p,!0)}${l}
fn isObserved(cell: u32, slice: u32) -> bool {
  let source = cell * ${u}u + slice;
  return finite(values[valuesOffset + source]) ${n.mask?`&& mask[maskOffset + source] != 0u`:``};
}
fn seasonalBasis(slice: u32, column: u32) -> f32 { ${h} }
struct ModelFit { coefficients: array<f32, ${p}>, residual: f32, valid: bool, converged: bool, iterations: u32 }
fn fitModel(design: array<f32, ${f*p}>, response: array<f32, ${f}>, columnCount: u32) -> ModelFit {
  var result: ModelFit;
  // Rank certification is independent of the bounded iterative trajectory.
  let certified = leastSquares(design, response, columnCount);
  if (!certified.valid) { return result; }
  ${n.fitMethod===`iterative`?`
  var residual = response;
  var norms: array<f32, ${p}>;
  for (var column = 0u; column < columnCount; column++) {
    for (var row = 0u; row < ${f}u; row++) { let value = design[column * ${f}u + row]; norms[column] += value * value; }
  }
  for (var iteration = 0u; iteration < ${n.maximumIterations??128}u; iteration++) {
    var maximumChange = 0.0; var maximumCoefficient = 0.0;
    for (var column = 0u; column < columnCount; column++) {
      var projection = 0.0;
      for (var row = 0u; row < ${f}u; row++) { projection += design[column * ${f}u + row] * residual[row]; }
      let change = projection / norms[column];
      result.coefficients[column] += change;
      if (!finite(result.coefficients[column])) { return result; }
      maximumChange = max(maximumChange, abs(change)); maximumCoefficient = max(maximumCoefficient, abs(result.coefficients[column]));
      for (var row = 0u; row < ${f}u; row++) { residual[row] -= design[column * ${f}u + row] * change; }
    }
    // Recompute each sweep to avoid accumulating incremental residual roundoff.
    for (var row = 0u; row < ${f}u; row++) {
      var prediction = 0.0;
      for (var column = 0u; column < columnCount; column++) { prediction += design[column * ${f}u + row] * result.coefficients[column]; }
      residual[row] = response[row] - prediction;
    }
    result.iterations = iteration + 1u;
    if (maximumChange <= ${a(n.tolerance??1e-5)} * (1.0 + maximumCoefficient)) { result.converged = true; break; }
  }
  for (var row = 0u; row < ${f}u; row++) { result.residual += residual[row] * residual[row]; }
  result.valid = finite(result.residual);
  `:`result.coefficients = certified.coefficients; result.residual = certified.residual; result.valid = true; result.converged = true; result.iterations = 1u;`}
  return result;
}`,body:`var training: array<u32, ${Math.ceil(f/32)}>;
var response: array<f32, ${f}>; var design: array<f32, ${f*p}>;
var rejected = false;
for (var slice = 0u; slice < ${u}u; slice++) {
  let source = index * ${u}u + slice; let value = values[valuesOffset + source];
  let observed = isObserved(index, slice);
  if (slice < ${f}u) {
    if (observed) { training[slice / 32u] |= 1u << (slice % 32u); } rejected = rejected || !observed;
    if (observed) { response[slice] = value; }
  }
}
${n.missingPolicy===`interpolate`?`
for (var slice = 0u; slice < ${f}u; slice++) {
  if (isObserved(index, slice)) { continue; }
  var left = slice; var right = slice;
  while (left > 0u && !isObserved(index, left)) { left--; }
  while (right + 1u < ${f}u && !isObserved(index, right)) { right++; }
  if (isObserved(index, left) && isObserved(index, right) && right > left) {
    response[slice] = mix(values[valuesOffset + index * ${u}u + left], values[valuesOffset + index * ${u}u + right], ${n.dateAxis?`rasterYearsBetween(slice, left) / rasterYearsBetween(right, left)`:`f32(slice - left) / f32(right - left)`}); training[slice / 32u] |= 1u << (slice % 32u);
  }
}`:``}
var count = 0u; var energy = 0.0;
for (var slice = 0u; slice < ${f}u; slice++) {
  if ((training[slice / 32u] & (1u << (slice % 32u))) == 0u) { continue; } count++; energy += response[slice] * response[slice];
  design[slice] = 1.0; design[${f}u + slice] = ${g};
  for (var column = 2u; column < ${m}u; column++) { design[column * ${f}u + slice] = seasonalBasis(slice, column); }
}
var chosen: array<f32, ${p}>; var boundary = 0xffffffffu; var gain = 0.0;
var code = 1u; var iterations = 0u; var converged = true; var selectedResidual = 0.0; var selectedColumns = ${m}u;
if (${n.missingPolicy===`reject`?`rejected`:`false`}) { code = 4u; }
else if (!finite(energy)) { code = 2u; }
else if (count >= ${Math.max(m+1,n.minimumSegmentLength*2)}u) {
  let baseline = fitModel(design, response, ${m}u); code = 2u;
  iterations = baseline.iterations; converged = baseline.converged;
  if (baseline.valid) {
    code = 0u; chosen = baseline.coefficients; selectedResidual = baseline.residual;
    let residualFloor = max(1e-20, energy * 1e-12);
    let baselineScore = f32(count) * log(max(baseline.residual, residualFloor) / f32(count)) + ${m}.0 * log(f32(count));
    for (var split = ${n.minimumSegmentLength}u; split <= ${f-n.minimumSegmentLength}u; split++) {
      var before = 0u; var after = 0u;
      for (var slice = 0u; slice < ${f}u; slice++) {
        design[${m*f}u + slice] = 0.0;
        if ((training[slice / 32u] & (1u << (slice % 32u))) != 0u) { if (slice < split) { before++; } else { after++; design[${m*f}u + slice] = 1.0; } }
      }
      if (before < ${n.minimumSegmentLength}u || after < ${n.minimumSegmentLength}u) { continue; }
      let candidate = fitModel(design, response, ${p}u);
      iterations = max(iterations, candidate.iterations);
      if (!candidate.valid) { continue; }
      converged = converged && candidate.converged;
      let candidateGain = baselineScore - (f32(count) * log(max(candidate.residual, residualFloor) / f32(count)) + ${m+2}.0 * log(f32(count)));
      if (candidateGain > max(gain, ${a(n.minimumBICGain??0)})) {
        chosen = candidate.coefficients; gain = candidateGain; boundary = split; selectedResidual = candidate.residual; selectedColumns = ${p}u;
      }
    }
    if (!converged) { code = 8u; }
  }
}
let fitted = code == 0u || code == 8u;
var residualScale = missing(); var alarm = -1.0; var maximumScore = 0.0; var consecutive = 0u; var previousSign = 0.0;
if (fitted) { residualScale = sqrt(max(selectedResidual, max(1e-20, energy * 1e-12)) / f32(max(1u, count - selectedColumns))); }
for (var column = 0u; column < ${p}u; column++) { coefficients[coefficientsOffset + index * ${p}u + column] = select(missing(), chosen[column], fitted); }
for (var slice = 0u; slice < ${u}u; slice++) {
  let destination = (index * ${u}u + slice) * 3u;
  var trend = missing(); var seasonal = missing(); var error = missing();
  if (fitted) {
    trend = chosen[0] + chosen[1] * ${g} + select(0.0, chosen[${m}], slice >= boundary);
    seasonal = 0.0; for (var column = 2u; column < ${m}u; column++) { seasonal += chosen[column] * seasonalBasis(slice, column); }
    if (isObserved(index, slice)) { error = values[valuesOffset + index * ${u}u + slice] - trend - seasonal; }
    ${n.monitoring?`
    if (slice >= ${f}u) {
      if (!isObserved(index, slice)) { consecutive = 0u; previousSign = 0.0; }
      else {
        let score = error / residualScale;
        if (!finite(score)) { code = 2u; }
        else {
          maximumScore = max(maximumScore, abs(score));
          if (abs(score) >= ${a(n.monitoring.threshold)}) {
            let direction = sign(score); consecutive = select(1u, consecutive + 1u, direction == previousSign); previousSign = direction;
            if (alarm < 0.0 && consecutive >= ${n.monitoring.minimumConsecutive}u) { alarm = f32(slice + 1u - ${n.monitoring.minimumConsecutive}u); }
          } else { consecutive = 0u; previousSign = 0.0; }
        }
      }
    }`:``}
  }
  components[componentsOffset + destination] = trend; components[componentsOffset + destination + 1u] = seasonal; components[componentsOffset + destination + 2u] = error;
}
breakIndex[breakIndexOffset + index] = boundary; improvement[improvementOffset + index] = select(missing(), gain, fitted); status[statusOffset + index] = code;
${n.output.diagnostics?`
let diagnosticOffset = diagnosticsOffset + index * 6u;
diagnostics[diagnosticOffset] = f32(iterations); diagnostics[diagnosticOffset + 1u] = select(0.0, 1.0, fitted && converged && code != 2u); diagnostics[diagnosticOffset + 2u] = f32(count);
diagnostics[diagnosticOffset + 3u] = residualScale; diagnostics[diagnosticOffset + 4u] = alarm; diagnostics[diagnosticOffset + 5u] = maximumScore;`:``}`})}var b=2048,x=12,S=256,C=16777216,w=class{id;props;chunkPlan;estimatedCommandNodeCount;ticksPerYear;constructor(t){this.props=t,this.id=t.id??`raster-seasonal-break`,this.ticksPerYear=t.dateAxis?c(this.id,t.dateAxis,t.sliceCount):0,f(this.id,!(t.dateAxis&&t.seasonality===`dummy`),`dateAxis requires harmonic seasonality; dummy seasons are defined on regular slots`);let r=t.monitoring?.historyLength??t.sliceCount,i=t.seasonality===`dummy`?t.period+2:5;f(this.id,r*i+i**2+2*i+8<=b,`per-thread QR working memory (history ${r}, ${i} coefficients) must fit ${b} floats; reduce sliceCount, monitoring.historyLength or period`),f(this.id,r*i+r+Math.ceil(r/32)+i+64<=b,`per-thread design and mask working memory (history ${r}, ${i} coefficients) must fit ${b} floats`),t.seasonality===`dummy`&&f(this.id,Number.isInteger(t.period)&&t.period<=x,`period must be an integer of at most ${x} for dummy seasonality`),t.monitoring&&(f(this.id,Number.isInteger(r)&&r>=6&&r<t.sliceCount,`monitoring.historyLength must be an integer in [6, sliceCount)`),f(this.id,Number.isFinite(Math.fround(t.monitoring.threshold))&&Math.fround(t.monitoring.threshold)>0,`monitoring.threshold must be a positive finite float32 value`),p(this.id,`monitoring.minimumConsecutive`,t.monitoring.minimumConsecutive,1,t.sliceCount-r)),p(this.id,`maximumIterations`,t.maximumIterations??128,1,S),f(this.id,Number.isFinite(Math.fround(t.tolerance??1e-5))&&Math.fround(t.tolerance??1e-5)>0,`tolerance must be a positive finite float32 value`),f(this.id,t.fitMethod!==`iterative`||r*i*(r-2*t.minimumSegmentLength+2)*(t.maximumIterations??128)<=C,`iterative fitting work (history times coefficients times candidate breaks times maximumIterations) must be at most ${C} per cell`),f(this.id,!(t.fitMethod===`iterative`||t.monitoring)||!!t.output.diagnostics,`output.diagnostics is required for iterative fitting and monitoring`),f(this.id,!t.output.diagnostics||t.output.diagnostics.length>=t.cellCount*6,`output.diagnostics must hold six entries per cell`),p(this.id,`cellCount`,t.cellCount),p(this.id,`sliceCount`,t.sliceCount,1,256),p(this.id,`cellCount times sliceCount times three`,t.cellCount*t.sliceCount*3),f(this.id,Number.isFinite(Math.fround(t.period))&&Math.fround(t.period)>(t.dateAxis?0:2),t.dateAxis?`period must be a positive finite number of fractional years with dateAxis`:`period must be a finite float32 value above 2`),p(this.id,`minimumSegmentLength`,t.minimumSegmentLength,3,Math.floor(r/2)),f(this.id,Number.isFinite(Math.fround(t.minimumBICGain??0))&&(t.minimumBICGain??0)>=0,`minimumBICGain must be a nonnegative finite float32 value`),f(this.id,t.values.length>=t.cellCount*t.sliceCount&&(!t.mask||t.mask.length>=t.cellCount*t.sliceCount),`values and mask must hold cellCount times sliceCount entries`),f(this.id,t.output.coefficients.length>=t.cellCount*i&&t.output.components.length>=t.cellCount*t.sliceCount*3,`output.coefficients must hold coefficientCount entries per cell and output.components three entries per observation`),f(this.id,[t.output.breakIndex,t.output.improvement,t.output.status].every(e=>e.length>=t.cellCount),`output.breakIndex, output.improvement and output.status must hold cellCount entries`),this.chunkPlan=n(this.id,t.cellCount,{values:{view:t.values,entriesPerCell:t.sliceCount},mask:{view:t.mask,entriesPerCell:t.sliceCount},coefficients:{view:t.output.coefficients,entriesPerCell:i},components:{view:t.output.components,entriesPerCell:t.sliceCount*3},breakIndex:{view:t.output.breakIndex,entriesPerCell:1},improvement:{view:t.output.improvement,entriesPerCell:1},status:{view:t.output.status,entriesPerCell:1},diagnostics:{view:t.output.diagnostics,entriesPerCell:6}}),this.estimatedCommandNodeCount=this.chunkPlan.estimatedCommandNodeCount,e(this.id,this.getInputs(),this.getOutputs(),this.chunkPlan)}getInputs(){return[this.props.values,...this.props.mask?[this.props.mask]:[],...this.props.dateAxis?[this.props.dateAxis.acquisitionTimes]:[]]}getOutputs(){return Object.values(this.props.output).filter(e=>e!==void 0)}getCommandNodes(e){let{props:n,id:c}=this,f=n.sliceCount;t(c,e,this.getInputs(),this.getOutputs()),n.dateAxis&&u(c,e,n.dateAxis);let p=n.dateAxis?`\n${l(this.ticksPerYear,f)}`:``;if(n.seasonality||n.fitMethod||n.missingPolicy||n.monitoring||n.output.diagnostics)return y(e,c,n,this.chunkPlan,p);let m=n.dateAxis?`rasterSliceYears(slice)`:`f32(slice) / ${f}.0`,h=n.dateAxis?`6.283185307179586 * rasterSliceYears(slice) / ${a(n.period)}`:`6.283185307179586 * f32(slice) / ${a(n.period)}`;return i(e,this.chunkPlan,{id:c,operation:`GPURasterSeasonalBreak`,invocationCount:n.cellCount,workgroupSize:32,bindings:[r(`values`,n.values),...n.mask?[r(`mask`,n.mask)]:[],...n.dateAxis?[d(n.dateAxis)]:[],...Object.entries(n.output).filter(([,e])=>e!==void 0).map(([e,t])=>r(e,t,!0))],declarations:s+o(f,5,!0)+p,body:`var design: array<f32, ${f*5}>; var response: array<f32, ${f}>; var valid: array<bool, ${f}>; var count = 0u; var energy = 0.0;
for (var slice = 0u; slice < ${f}u; slice++) {
  let source = index * ${f}u + slice; let value = values[valuesOffset + source];
  valid[slice] = finite(value) ${n.mask?`&& mask[maskOffset + source] != 0u`:``};
  if (!valid[slice]) { continue; }
  count++; response[slice] = value; energy += value * value;
  let angle = ${h};
  let row = array<f32, 4>(1.0, ${m}, cos(angle), sin(angle));
  for (var column = 0u; column < 4u; column++) { design[column * ${f}u + slice] = row[column]; }
}
var chosen: array<f32, 5>; var boundary = 0xffffffffu; var gain = 0.0; var code = 1u;
if (!finite(energy)) { code = 2u; }
else if (count >= ${Math.max(5,n.minimumSegmentLength*2)}u) {
  let baseline = leastSquares(design, response, 4u); code = 2u;
  if (baseline.valid) {
    code = 0u; for (var column = 0u; column < 4u; column++) { chosen[column] = baseline.coefficients[column]; }
    let residualFloor = max(1e-20, energy * 1e-12);
    let baselineScore = f32(count) * log(max(baseline.residual, residualFloor) / f32(count)) + 4.0 * log(f32(count));
    for (var split = ${n.minimumSegmentLength}u; split <= ${f-n.minimumSegmentLength}u; split++) {
      var before = 0u; var after = 0u;
      for (var slice = 0u; slice < ${f}u; slice++) { design[4u * ${f}u + slice] = 0.0; if (valid[slice]) { if (slice < split) { before++; } else { after++; design[4u * ${f}u + slice] = 1.0; } } }
      if (before < ${n.minimumSegmentLength}u || after < ${n.minimumSegmentLength}u) { continue; }
      let fit = leastSquares(design, response, 5u); if (!fit.valid) { continue; }
      let candidateGain = baselineScore - (f32(count) * log(max(fit.residual, residualFloor) / f32(count)) + 6.0 * log(f32(count)));
      if (candidateGain > max(gain, ${a(n.minimumBICGain??0)})) { gain = candidateGain; boundary = split; chosen = fit.coefficients; }
    }
  }
}
for (var column = 0u; column < 5u; column++) { coefficients[coefficientsOffset + index * 5u + column] = select(missing(), chosen[column], code == 0u); }
for (var slice = 0u; slice < ${f}u; slice++) {
  let destination = (index * ${f}u + slice) * 3u;
  var trend = missing(); var seasonal = missing(); var error = missing();
  if (code == 0u) { trend = chosen[0] + chosen[1] * ${m} + select(0.0, chosen[4], slice >= boundary); let angle = ${h}; seasonal = chosen[2] * cos(angle) + chosen[3] * sin(angle); if (valid[slice]) { error = response[slice] - trend - seasonal; } }
  components[componentsOffset + destination] = trend; components[componentsOffset + destination + 1u] = seasonal; components[componentsOffset + destination + 2u] = error;
}
breakIndex[breakIndexOffset + index] = boundary; improvement[improvementOffset + index] = select(missing(), gain, code == 0u); status[statusOffset + index] = code;`})}},T=36,E=[.04,.05,.42,.3],D=[-.4,1.3],O=20,k=m({parameters:h,rebuildOn:[`minimumBICGain`],caption:`A seasonal series per cell in; the strength of its level shift out.`,createInputs:({random:e,aspect:t})=>{let n=Math.round(24*t),r=v(n,24,[7,9]),i=n*24;return{columns:n,rows:24,cellCount:i,sliceCount:T,observed:r,noise:Float32Array.from({length:i*T},()=>e()-.5)}},build(e,t,n){let{cellCount:r,sliceCount:i}=t,a=e.allocate(`series`,`float32`,r*i),o={coefficients:e.allocate(`coefficients`,`float32`,r*5),components:e.allocate(`components`,`float32`,r*i*3),breakIndex:e.allocate(`break-index`,`uint32`,r),improvement:e.allocate(`improvement`,`float32`,r),status:e.allocate(`status`,`uint32`,r)};return e.graph.add(new w({cellCount:r,sliceCount:i,values:a.view,period:12,minimumSegmentLength:8,minimumBICGain:n.minimumBICGain,output:{coefficients:o.coefficients.view,components:o.components.view,breakIndex:o.breakIndex.view,improvement:o.improvement.view,status:o.status.view}})),{outputs:{improvement:o.improvement,breakIndex:o.breakIndex,inset:e.allocate(`inset`,`float32`,i)},update:e=>{let n=new Float32Array(r*i);for(let a=0;a<r;a++){let r=t.observed[a*2],o=t.observed[a*2+1],s=Number.isFinite(o)?o-r:0;for(let o=0;o<i;o++)n[a*i+o]=r+.2*Math.sin(o/12*Math.PI*2)+(o>=O?s*e.shift:0)+e.noise*t.noise[a*i+o]}a.write(n)}}},interaction:{kind:`probe`,hint:`probe a cell series`,rest:[.5,.5],resolve:(e,{inputs:t,outputs:n,values:r,world:i,peek:a})=>{let{column:o,row:s,index:c}=_(t,i,e),l=t.observed[c*2],u=t.observed[c*2+1],d=Number.isFinite(u)?u-l:0;n.inset.write(Float32Array.from({length:T},(e,n)=>l+.2*Math.sin(n/12*Math.PI*2)+(n>=O?d*r.shift:0)+r.noise*t.noise[c*T+n]));let f=a(n.breakIndex),p=a(n.improvement),m=p?p[c]:0,h=f&&m>0?f[c]:-1,v=e=>i.width*(E[0]+e/35*(E[2]-E[0]));return{glyphs:[g(t,i,o,s),{shape:`box`,minimum:[v(0),i.height*E[1]],maximum:[v(35),i.height*E[3]]},...h>=0?[{shape:`line`,from:[v(h),i.height*E[1]],to:[v(h),i.height*E[3]]}]:[]],readout:h>=0?`break at month ${h} | BIC gain ${m.toFixed(0)}`:`no break found`}}},draw:({inputs:e,outputs:t})=>[{layer:`raster`,values:t.improvement,columns:e.columns,rows:e.rows,ramp:`sequential`,range:[0,200],smoothing:!1},{layer:`chart`,kind:`line`,values:t.inset,range:D,rectangle:E,color:[1,1,1,.95]}]});export{k as default};
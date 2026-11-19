import{a as e,i as t,n,t as r}from"./raster-time-series-chunks-FYvfnBw_.js";import{T as i,f as a,g as o,p as s}from"./gpu-contributor-utils-BIwLQON5.js";import{a as c,i as l,t as u}from"./raster-kernel-utils-DcIN7G5I.js";import{a as d,i as f,r as p}from"./raster-irregular-date-axis-B315UxmE.js";import{t as m}from"./define-run-B-brkCe4.js";import{c as h}from"./pointer-overlay-Qh8qflcA.js";import{a as g,c as _,d as v,f as y,g as b,m as x,n as S,o as C,r as w,t as T,u as E,v as D,x as O}from"./charts-Br0w6lsw.js";import{n as k,r as A,t as j}from"./probe-DLtMtDEc.js";import{r as M,t as N}from"./readback-C6WbEPkA.js";import{Jr as P}from"./entry-preview-D70TdSbK.js";var F=2048;function I(e,t=new Float32Array(3)){return o(`getGPURasterHANTSParameterValues`,`period`,e.period,0),t[0]=e.period,t[1]=e.fitErrorTolerance,t[2]=e.degreeOfOverdetermination??1,t}var L=class{props;id;estimatedCommandNodeCount;chunkProps;rejectionPassCount;rejectionSide;ticksPerYear;constructor(t){this.props=t;let r=t.id??`raster-hants`;this.id=r;let o=t.rejectionPassCount??8;this.rejectionPassCount=o;let c=t.rejectionSide??`low`;this.rejectionSide=c,a(r,`cellCount`,t.cellCount),a(r,`sliceCount`,t.sliceCount,1,256),a(r,`harmonicCount`,t.harmonicCount,1);let l=2*t.harmonicCount+1;i(r,Number.isInteger(o)&&o>=0&&o<=16,`rejectionPassCount must be an integer in [0, 16]`),i(r,c===`low`||c===`high`||c===`both`,`rejectionSide must be 'low', 'high' or 'both'`),i(r,Math.max(t.sliceCount*l+l*l+2*l+8,t.sliceCount*l+t.sliceCount+Math.ceil(t.sliceCount/32)+l+64)<=F,`per-thread QR working memory (sliceCount ${t.sliceCount}, ${l} coefficients) must fit ${F} floats`);let u=t.cellCount*t.sliceCount;a(r,`cellCount times sliceCount`,u),i(r,[t.values,t.mask,t.output.smoothed].every(e=>!e||e.length>=u),`values, mask and output.smoothed must hold cellCount times sliceCount entries`),i(r,t.output.coefficients.length>=t.cellCount*l,`output.coefficients must hold ${l} entries per cell`),i(r,t.output.status.length>=t.cellCount,`output.status must hold one entry per cell`),i(r,!t.output.diagnostics||t.output.diagnostics.length>=t.cellCount*2,`output.diagnostics must hold two entries per cell`),i(r,t.parameters.length>=3,`parameters must hold 3 entries`),this.ticksPerYear=t.dateAxis?d(r,t.dateAxis,t.sliceCount):0,this.chunkProps=n(r,t,{values:t.sliceCount,mask:t.sliceCount,"output.smoothed":t.sliceCount,"output.coefficients":l,"output.status":1,"output.diagnostics":2}),this.estimatedCommandNodeCount=this.chunkProps.length*1,s(r,this.estimatedCommandNodeCount);let f=this.getInputs(),p=this.getOutputs();e(r,f,p,[{inputs:f,outputs:p}])}getInputs(){let{values:e,mask:t,parameters:n}=this.props;return[e,n,...t?[t]:[],...this.props.dateAxis?[this.props.dateAxis.acquisitionTimes]:[]]}getOutputs(){let{smoothed:e,coefficients:t,status:n,diagnostics:r}=this.props.output;return[e,t,n,...r?[r]:[]]}getCommandNodes(e){return t(e,this.id,[...this.getInputs(),...this.getOutputs()]),this.chunkProps.flatMap((t,n)=>this.getChunkCommandNodes(e,{...t,id:`${this.id}-chunk-${n}`}))}getChunkCommandNodes(e,t){let{rejectionPassCount:n,rejectionSide:i}=this,a=t.id,{sliceCount:o,harmonicCount:s,values:d,mask:m,parameters:h,dateAxis:g,output:_}=t,v=2*s+1,y=g?`rasterSliceYears(slice)`:`f32(slice) - ${(o-1)*.5}`,b=i===`both`?`abs(residual)`:i===`low`?`-residual`:`residual`;return[r(e,{id:a,operation:`GPURasterHANTS`,invocationCount:t.cellCount,workgroupSize:8,bindings:[c(`values`,d),c(`parameters`,h),...m?[c(`mask`,m)]:[],...g?[p(g)]:[],c(`smoothed`,_.smoothed,!0),c(`coefficients`,_.coefficients,!0),c(`status`,_.status,!0),..._.diagnostics?[c(`diagnostics`,_.diagnostics,!0)]:[]],declarations:`${u}
${g?f(this.ticksPerYear,o):``}
${l(o,v)}
fn getHarmonicColumn(slice: u32, column: u32, period: f32) -> f32 {
  if (column == 0u) { return 1.0; }
  let harmonic = (column + 1u) / 2u;
  let angle = 6.283185307179586 * f32(harmonic) * (${y}) / period;
  return select(sin(angle), cos(angle), column % 2u == 1u);
}`,body:`let period = parameters[parametersOffset];
let tolerance = parameters[parametersOffset + 1u];
let overdetermination = u32(max(parameters[parametersOffset + 2u], 0.0));
let minimumSupport = ${v}u + overdetermination;
var keptMask: array<u32, ${Math.ceil(o/32)}>;
var keptCount = 0u;
for (var slice = 0u; slice < ${o}u; slice++) {
  let source = index * ${o}u + slice;
  if (finite(values[valuesOffset + source]) ${m?`&& mask[maskOffset + source] != 0u`:``}) { keptMask[slice / 32u] |= 1u << (slice % 32u); keptCount++; }
}
var code = 0u;
var fitted: LeastSquaresResult;
var rejected = 0u;
var passesUsed = 0u;
if (keptCount < minimumSupport) {
  code = 1u;
} else {
  var finished = false;
  for (var iteration = 0u; iteration <= ${n}u; iteration++) {
    var design: array<f32, ${o*v}>;
    var response: array<f32, ${o}>;
    for (var slice = 0u; slice < ${o}u; slice++) {
      if ((keptMask[slice / 32u] & (1u << (slice % 32u))) == 0u) { continue; }
      response[slice] = values[valuesOffset + index * ${o}u + slice];
      for (var column = 0u; column < ${v}u; column++) { design[column * ${o}u + slice] = getHarmonicColumn(slice, column, period); }
    }
    fitted = leastSquares(design, response);
    passesUsed = iteration + 1u;
    if (!fitted.valid) { code = 2u; finished = true; break; }
    var worst = 0xffffffffu;
    var worstMagnitude = tolerance;
    for (var slice = 0u; slice < ${o}u; slice++) {
      if ((keptMask[slice / 32u] & (1u << (slice % 32u))) == 0u) { continue; }
      var prediction = 0.0;
      for (var column = 0u; column < ${v}u; column++) { prediction += fitted.coefficients[column] * getHarmonicColumn(slice, column, period); }
      let residual = values[valuesOffset + index * ${o}u + slice] - prediction;
      if (${b} > worstMagnitude) { worstMagnitude = ${b}; worst = slice; }
    }
    if (worst == 0xffffffffu || keptCount <= minimumSupport) { finished = true; break; }
    if (iteration == ${n}u) { break; }
    keptMask[worst / 32u] &= ~(1u << (worst % 32u));
    keptCount--;
    rejected++;
  }
  if (!finished) { code = 8u; }
}
let published = code == 0u || code == 8u;
for (var column = 0u; column < ${v}u; column++) {
  coefficients[coefficientsOffset + index * ${v}u + column] = select(missing(), fitted.coefficients[column], published);
}
for (var slice = 0u; slice < ${o}u; slice++) {
  var prediction = 0.0;
  if (published) {
    for (var column = 0u; column < ${v}u; column++) { prediction += fitted.coefficients[column] * getHarmonicColumn(slice, column, period); }
  }
  smoothed[smoothedOffset + index * ${o}u + slice] = select(missing(), prediction, published);
}
status[statusOffset + index] = code;
${_.diagnostics?`diagnostics[diagnosticsOffset + index * 2u] = rejected; diagnostics[diagnosticsOffset + index * 2u + 1u] = passesUsed;`:``}`})]}},R=64,z=.06,[B]=w(1),V=[-.2,1],H=m({parameters:P,rebuildOn:[`harmonicCount`,`rejectionPassCount`],buildCache:6,caption:`Synthetic seasonal cycle on measured Dixie NDVI levels with cloud dips and gaps; 64 irregular dates.`,createInputs:({aspect:e,random:t})=>{let n=D(e,t,{sliceCount:R,yearCount:3});return{...n,observed:n.observe(.12,z)}},build(e,t,n){let{grid:r,sliceCount:i}=t,a=Math.round(n.harmonicCount),o=e.upload(`series`,`float32`,t.observed),s=e.allocate(`parameters`,`float32`,3),c={smoothed:e.allocate(`smoothed`,`float32`,r.cellCount*i),coefficients:e.allocate(`coefficients`,`float32`,r.cellCount*(2*a+1)),status:e.allocate(`status`,`uint32`,r.cellCount),diagnostics:e.allocate(`diagnostics`,`uint32`,r.cellCount*2)};return e.graph.add(new L({cellCount:r.cellCount,sliceCount:i,values:o.view,parameters:s.view,harmonicCount:a,rejectionSide:`low`,rejectionPassCount:Math.round(n.rejectionPassCount),dateAxis:O(e,t.times),output:{smoothed:c.smoothed.view,coefficients:c.coefficients.view,status:c.status.view,diagnostics:c.diagnostics.view}})),{outputs:{...c,blank:N(e,`blank`,r.cellCount),level:e.upload(`level`,`float32`,t.level),kept:T(e,`kept`,i),rejected:T(e,`rejected`,i),curve:T(e,`curve`,i),lowerBand:T(e,`lower-band`,i),upperBand:T(e,`upper-band`,i)},update:e=>{t.observed=t.observe(e.cloudFraction,z),o.write(t.observed),s.write(I({period:1,fitErrorTolerance:e.fitErrorTolerance}))}}},async afterRun({outputs:e,inputs:t,upload:n}){let r=await M(e.diagnostics);return{map:n(`rejected-map`,`float32`,Float32Array.from({length:t.grid.cellCount},(e,n)=>t.level[n]>0?r[n*2]:NaN))}},interaction:{kind:`probe`,hint:`probe a pixel`,rest:[.72,.5],resolve:(e,{inputs:t,outputs:n,values:r,world:i,peek:a})=>{let{grid:o,sliceCount:s}=t,c=A(o,i,e),l=t.observed.subarray(c.index*s,(c.index+1)*s),u=a(n.smoothed),d=a(n.status),f=a(n.diagnostics),p=u?Array.from(u.subarray(c.index*s,(c.index+1)*s)):[],m=Array.from(t.years),g=r.fitErrorTolerance,_=f?f[c.index*2]:0,y=new Set(m.map((e,t)=>t).filter(e=>Number.isFinite(l[e]-p[e])).sort((e,t)=>l[e]-p[e]-(l[t]-p[t])).slice(0,_)),S=m.map((e,t)=>!y.has(t));b(n.kept,m,l.map((e,t)=>S[t]?e:NaN)),b(n.rejected,m,l.map((e,t)=>S[t]?NaN:e)),b(n.curve,m,p),b(n.lowerBand,m,p.map(e=>e-g)),b(n.upperBand,m,p.map(e=>e+g));let w=[t.years[0],t.years[s-1]];return{spotlight:null,glyphs:[k(o,i,c),C(i,B),...x(i,B,w,v(t.times))],readout:d&&f?d[c.index]===0||d[c.index]===8?`rejected ${f[c.index*2]} · passes ${f[c.index*2+1]} · tolerance ${h(g)} · ${j(o,c)}`:`too few valid observations`:`reading the fit`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>{let i=[e.years[0],e.years[e.sliceCount-1]];return[E(t.level,e.grid,e.grid.mapLeft,{context:!0,ramp:`sequential`,range:[0,1]}),E(n?.map??t.blank,e.grid,e.grid.mapLeft,{ramp:`sequential`,range:[0,14]}),g(t.lowerBand,B,V,i,y(r,.8),1.8),g(t.upperBand,B,V,i,y(r,.8),1.8),g(t.kept,B,V,i,_(r),4.5),g(t.rejected,B,V,i,S(r,1),4.5),g(t.curve,B,V,i,S(r,0),3)]}});export{H as default};
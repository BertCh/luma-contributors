import{a as e,i as t,n,t as r}from"./raster-time-series-chunks-FYvfnBw_.js";import{T as i,f as a,g as o,p as s}from"./gpu-contributor-utils-BIwLQON5.js";import{a as c,i as l,t as u}from"./raster-kernel-utils-DcIN7G5I.js";import{a as d,i as f,r as p}from"./raster-irregular-date-axis-B315UxmE.js";import{t as m}from"./raster-harmonic-trend-design-wgsl-DiyAnFyi.js";import{t as h}from"./define-run-B-brkCe4.js";import{c as g}from"./pointer-overlay-Qh8qflcA.js";import{_,a as v,b as y,c as b,d as x,f as S,g as C,h as w,i as T,m as E,n as D,o as O,p as k,r as A,t as j,u as M,x as N,y as P}from"./charts-Br0w6lsw.js";import{n as F,r as I,t as L}from"./probe-DLtMtDEc.js";import{n as R,r as z,t as B}from"./readback-C6WbEPkA.js";import{Xr as V}from"./entry-preview-D70TdSbK.js";var H=2048,U=4294967295;function W(e,t=new Float32Array(3)){return o(`getGPURasterBFASTMonitorParameterValues`,`criticalValue`,e.criticalValue,0),t[0]=e.period??1,t[1]=e.bandwidth??.25,t[2]=e.criticalValue,t}var G=class{props;id;estimatedCommandNodeCount;chunkProps;harmonicCount;includeTrend;columnCount;ticksPerYear;constructor(t){this.props=t;let r=t.id??`raster-bfast-monitor`;this.id=r,this.harmonicCount=t.harmonicCount??3,this.includeTrend=t.includeTrend??!0,this.columnCount=(this.includeTrend?2:1)+2*this.harmonicCount;let{sliceCount:o,historySliceCount:c}=t;a(r,`cellCount`,t.cellCount),a(r,`sliceCount`,o,2,256),a(r,`harmonicCount`,this.harmonicCount,1),a(r,`historySliceCount`,c,1,o-1);let l=this.columnCount;i(r,Math.max(c*l+l*l+2*l+8,c*l+3*o+l+64)<=H,`per-thread QR working memory (historySliceCount ${c}, ${l} coefficients) must fit ${H} floats`);let u=t.cellCount*o;a(r,`cellCount times sliceCount`,u),i(r,[t.values,t.mask].every(e=>!e||e.length>=u),`values and mask must hold cellCount times sliceCount entries`),i(r,t.output.coefficients.length>=t.cellCount*l,`output.coefficients must hold ${l} entries per cell`),i(r,t.output.scores.length>=t.cellCount*2,`output.scores must hold two entries per cell`),i(r,t.output.cellSummary.length>=t.cellCount*3,`output.cellSummary must hold three entries per cell`),i(r,!t.output.process||t.output.process.length>=t.cellCount*(o-c),`output.process must hold one entry per cell and monitoring slice`),i(r,t.parameters.length>=3,`parameters must hold 3 entries`),this.ticksPerYear=d(r,t.dateAxis,o),this.chunkProps=n(r,t,{values:t.sliceCount,mask:t.sliceCount,"output.coefficients":this.columnCount,"output.scores":2,"output.cellSummary":3,"output.process":t.sliceCount-t.historySliceCount}),this.estimatedCommandNodeCount=this.chunkProps.length*1,s(r,this.estimatedCommandNodeCount),e(r,this.getInputs(),this.getOutputs())}getInputs(){let{values:e,mask:t,parameters:n}=this.props;return[e,n,...t?[t]:[],...this.props.dateAxis?[this.props.dateAxis.acquisitionTimes]:[]]}getOutputs(){let{coefficients:e,scores:t,cellSummary:n,process:r}=this.props.output;return[e,t,n,...r?[r]:[]]}getCommandNodes(e){return t(e,this.id,[...this.getInputs(),...this.getOutputs()]),this.chunkProps.flatMap((t,n)=>this.getChunkCommandNodes(e,{...t,id:`${this.id}-chunk-${n}`}))}getChunkCommandNodes(e,t){let{columnCount:n,includeTrend:i}=this,a=t.id,{sliceCount:o,historySliceCount:s,values:d,mask:h,parameters:g,output:_}=t,v=o-s;return[r(e,{id:a,operation:`GPURasterBFASTMonitor`,invocationCount:t.cellCount,workgroupSize:8,bindings:[c(`values`,d),c(`parameters`,g),...h?[c(`mask`,h)]:[],p(t.dateAxis),c(`coefficients`,_.coefficients,!0),c(`scores`,_.scores,!0),c(`cellSummary`,_.cellSummary,!0),..._.process?[c(`process`,_.process,!0)]:[]],declarations:`${u}
${f(this.ticksPerYear,o)}
${m(o,i)}
${l(s,n)}
fn getLogPlus(value: f32) -> f32 { return select(1.0, log(value), value > 2.718281828459045); }
fn getSortedMedian(sample: ptr<function, array<f32, ${v}>>, count: u32) -> f32 {
  for (var outer = 1u; outer < count; outer++) {
    let item = (*sample)[outer];
    var inner = outer;
    while (inner > 0u && (*sample)[inner - 1u] > item) { (*sample)[inner] = (*sample)[inner - 1u]; inner--; }
    (*sample)[inner] = item;
  }
  if (count % 2u == 1u) { return (*sample)[count / 2u]; }
  return 0.5 * ((*sample)[count / 2u - 1u] + (*sample)[count / 2u]);
}`,body:`fillRasterSliceYears();
let period = parameters[parametersOffset];
let bandwidth = parameters[parametersOffset + 1u];
let criticalValue = parameters[parametersOffset + 2u];
var slicesUsed: array<u32, ${o}>;
var total = 0u;
var historyTotal = 0u;
var design: array<f32, ${s*n}>;
var response: array<f32, ${s}>;
for (var slice = 0u; slice < ${o}u; slice++) {
  let source = index * ${o}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) ${h?`|| mask[maskOffset + source] == 0u`:``}) { continue; }
  slicesUsed[total] = slice;
  total++;
  if (slice < ${s}u) {
    historyTotal++;
    response[slice] = value;
    for (var column = 0u; column < ${n}u; column++) { design[column * ${s}u + slice] = getHarmonicTrendColumn(slice, column, period); }
  }
}
var code = 1u;
var fit: LeastSquaresResult;
var scale = 0.0;
if (historyTotal > ${n}u) {
  fit = leastSquares(design, response);
  scale = sqrt(fit.residual / f32(historyTotal - ${n}u));
  code = select(2u, 0u, fit.valid && finite(scale) && scale > 0.0);
}
for (var column = 0u; column < ${n}u; column++) {
  coefficients[coefficientsOffset + index * ${n}u + column] = select(missing(), fit.coefficients[column], code == 0u);
}
${_.process?`for (var monitoringSlice = 0u; monitoringSlice < ${v}u; monitoringSlice++) { process[processOffset + index * ${v}u + monitoringSlice] = missing(); }`:``}
var breakAt = ${U}u;
var largestRatio = 0.0;
var monitoredMedian = missing();
if (code == 0u) {
  var residuals: array<f32, ${o}>;
  var monitored: array<f32, ${v}>;
  var monitoredTotal = 0u;
  for (var position = 0u; position < total; position++) {
    let slice = slicesUsed[position];
    var prediction = 0.0;
    for (var column = 0u; column < ${n}u; column++) { prediction += fit.coefficients[column] * getHarmonicTrendColumn(slice, column, period); }
    residuals[position] = values[valuesOffset + index * ${o}u + slice] - prediction;
    if (position >= historyTotal) { monitored[monitoredTotal] = residuals[position]; monitoredTotal++; }
  }
  let bandwidthCount = max(1u, u32(floor(bandwidth * f32(historyTotal))));
  let processScale = scale * sqrt(f32(historyTotal));
  for (var position = historyTotal; position < total; position++) {
    var windowTotal = 0.0;
    let windowStart = position + 1u - min(bandwidthCount, position + 1u);
    for (var entry = windowStart; entry <= position; entry++) { windowTotal += residuals[entry]; }
    let processValue = windowTotal / processScale;
    let boundary = criticalValue * sqrt(2.0 * getLogPlus(f32(position + 1u) / f32(historyTotal)));
    largestRatio = max(largestRatio, abs(processValue) / boundary);
    if (breakAt == ${U}u && abs(processValue) > boundary) { breakAt = slicesUsed[position]; }
    ${_.process?`process[processOffset + index * ${v}u + slicesUsed[position] - ${s}u] = processValue;`:``}
  }
  if (monitoredTotal > 0u) { monitoredMedian = getSortedMedian(&monitored, monitoredTotal); }
}
scores[scoresOffset + index * 2u] = select(missing(), monitoredMedian, code == 0u);
scores[scoresOffset + index * 2u + 1u] = select(missing(), largestRatio, code == 0u);
cellSummary[cellSummaryOffset + index * 3u] = breakAt;
cellSummary[cellSummaryOffset + index * 3u + 1u] = code;
cellSummary[cellSummaryOffset + index * 3u + 2u] = historyTotal;`})]}},K=1,q=3,[J,Y]=A(2),X=[-.2,1],Z=[-12,12],Q=h({parameters:V,rebuildOn:[`historySliceCount`],buildCache:4,caption:`Measured Dixie NDVI, 16 irregular Sentinel-2 scenes: break magnitude per pixel (grey: no break) from a history fit and a MOSUM test.`,createInputs:({aspect:e})=>_(e),build(e,t,n){let{grid:r,sliceCount:i}=t,a=Math.round(n.historySliceCount),o=i-a,s=e.upload(`series`,`float32`,t.observed),c=e.allocate(`parameters`,`float32`,3),l={coefficients:e.allocate(`coefficients`,`float32`,r.cellCount*q),scores:e.allocate(`scores`,`float32`,r.cellCount*2),cellSummary:e.allocate(`cell-summary`,`uint32`,r.cellCount*3),process:e.allocate(`process`,`float32`,r.cellCount*o)};return e.graph.add(new G({cellCount:r.cellCount,sliceCount:i,values:s.view,parameters:c.view,dateAxis:N(e,t.times),historySliceCount:a,harmonicCount:K,includeTrend:!1,output:{coefficients:l.coefficients.view,scores:l.scores.view,cellSummary:l.cellSummary.view,process:l.process.view}})),{outputs:{...l,series:s,blank:B(e,`blank`,r.cellCount),preFire:e.upload(`pre-fire`,`float32`,P(t,6)),historyDates:e.allocate(`history-dates`,`float32`,i),historyValues:e.allocate(`history-values`,`float32`,i),monitoringDates:e.allocate(`monitoring-dates`,`float32`,i),monitoringValues:e.allocate(`monitoring-values`,`float32`,i),curveDates:e.allocate(`curve-dates`,`float32`,i),curveValues:e.allocate(`curve-values`,`float32`,i),processDates:e.allocate(`process-dates`,`float32`,i),processValues:e.allocate(`process-values`,`float32`,i),upperBoundary:j(e,`upper-boundary`,i),lowerBoundary:j(e,`lower-boundary`,i)},update:e=>c.write(W({criticalValue:e.criticalValue,bandwidth:e.bandwidth}))}},async afterRun({outputs:e,inputs:t,upload:n}){let[r,i]=await Promise.all([z(e.cellSummary),R(e.scores)]),a=new Float32Array(t.grid.cellCount);for(let e=0;e<a.length;e++){let t=r[e*3];a[e]=r[e*3+1]===0&&t!==4294967295?i[e*2]:NaN}return{map:n(`magnitude-map`,`float32`,a)}},interaction:{kind:`probe`,hint:`probe a pixel`,rest:[.72,.5],resolve:(e,{inputs:t,outputs:n,values:r,world:i,peek:a})=>{let{grid:o,sliceCount:s}=t,c=Math.round(r.historySliceCount),l=I(o,i,e),u=a(n.cellSummary),d=a(n.coefficients),f=a(n.scores),p=a(n.process),m=x(t.times),h=[t.years[0],t.years[s-1]],_=t.observed.subarray(l.index*s,(l.index+1)*s),v=Array.from(t.years);w(n.historyDates,n.historyValues,v.slice(0,c),Array.from(_.subarray(0,c))),w(n.monitoringDates,n.monitoringValues,v.slice(c),Array.from(_.subarray(c)));let b=u?u[l.index*3+1]:1,S=v,T=v.map((e,t)=>{if(!d||b!==0||!Number.isFinite(_[t]))return NaN;let n=l.index*q,r=2*Math.PI*e;return d[n]+d[n+1]*Math.cos(r)+d[n+2]*Math.sin(r)});w(n.curveDates,n.curveValues,S,T);let D=s-c,A=[],j=[];for(let e=0;e<D;e++)A.push(v[c+e]),j.push(p?p[l.index*D+e]:NaN);w(n.processDates,n.processValues,A,j);let M=_.slice(0,c).filter(Number.isFinite).length,N=_.filter(Number.isFinite).length,P=e=>r.criticalValue*Math.sqrt(2*Math.max(1,Math.log(e/Math.max(M,1)))),R=P(Math.max(N,1)),z=[],B=[],V=M;for(let e=c;e<s;e++)Number.isFinite(_[e])&&(V++,z.push(v[e]),B.push(P(V)));C(n.upperBoundary,z,B),C(n.lowerBoundary,z,B.map(e=>-e));let H=u?u[l.index*3]:U,W=b===0&&H!==4294967295,G=f?f[l.index*2]:NaN,K=(v[c-1]+v[c])/2;return{spotlight:null,glyphs:[F(o,i,l),O(i,J),O(i,Y),...E(i,J,h,m),...E(i,Y,h,m),k(i,J,h,K,!0),k(i,Y,h,K,!0),...W?[k(i,J,h,v[H]),k(i,Y,h,v[H])]:[]],readout:u?b===0?W?`break ${y(t.times[H])} · magnitude ${g(G)} · ${L(o,l)}`:`no break · boundary ${g(R,1)} · ${L(o,l)}`:`no fit · ${M} history observations · ${L(o,l)}`:`reading the fit`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>{let i=[e.years[0],e.years[e.sliceCount-1]];return[M(t.preFire,e.grid,e.grid.mapLeft,{context:!0,ramp:`sequential`,range:[0,1]}),M(n?.map??t.blank,e.grid,e.grid.mapLeft,{ramp:`diverging`,range:[-.5,.5]}),T(t.historyValues,t.historyDates,J,X,i,b(r),4.5),T(t.monitoringValues,t.monitoringDates,J,X,i,D(r,1),4.5),T(t.curveValues,t.curveDates,J,X,i,D(r,0),3),v(t.upperBoundary,Y,Z,i,S(r,.9),3),v(t.lowerBoundary,Y,Z,i,S(r,.9),3),T(t.processValues,t.processDates,Y,Z,i,D(r,2),4.5)]}});export{Q as default};
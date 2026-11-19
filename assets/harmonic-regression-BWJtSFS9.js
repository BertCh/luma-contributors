import{Q as e}from"./dist-BusamGlx.js";import{r as t}from"./kernel-C3tfQOn4.js";import{a as n,i as r,n as i,r as a,t as o}from"./raster-time-series-chunks-FYvfnBw_.js";import{T as s,f as c,g as l,n as u,p as d,v as f}from"./gpu-contributor-utils-BIwLQON5.js";import{a as p,i as m,t as h}from"./raster-kernel-utils-DcIN7G5I.js";import{a as g,i as _,r as v}from"./raster-irregular-date-axis-B315UxmE.js";import{t as y}from"./raster-harmonic-trend-design-wgsl-DiyAnFyi.js";import{t as b}from"./define-run-B-brkCe4.js";import{c as x}from"./pointer-overlay-Qh8qflcA.js";import{a as S,c as C,d as w,g as T,m as E,n as D,o as O,r as k,s as A,t as j,u as M,v as N,x as P}from"./charts-Br0w6lsw.js";import{n as F,r as I,t as L}from"./probe-DLtMtDEc.js";import{qr as R}from"./entry-preview-D70TdSbK.js";var z=4096,B=u,V=2048;function H(e,t=new Float32Array(1)){return l(`getGPURasterHarmonicRegressionParameterValues`,`period`,e.period,0),t[0]=e.period,t}var U=class{props;id;estimatedCommandNodeCount;mode;chunkProps;ticksPerYear;constructor(e){this.props=e;let t=e.id??`raster-harmonic-regression`;this.id=t,this.mode=e.mode??`storage`,s(t,this.mode===`storage`||this.mode===`reference`,`mode must be 'storage' or 'reference'`),c(t,`cellCount`,e.cellCount),c(t,`sliceCount`,e.sliceCount,1,this.mode===`reference`?256:z),c(t,`harmonicCount`,e.harmonicCount,1);let r=2+2*e.harmonicCount;s(t,(this.mode===`reference`?Math.max(e.sliceCount*r+r*r+2*r+8,e.sliceCount*r+2*e.sliceCount+r+64):r*r+2*r+8)<=V,`per-thread QR working memory (sliceCount ${e.sliceCount}, ${r} coefficients) must fit ${V} floats`);let o=e.cellCount*e.sliceCount;c(t,`cellCount times sliceCount`,o),s(t,[e.values,e.mask,e.output.fitted].every(e=>!e||e.length>=o),`values, mask and output.fitted must hold cellCount times sliceCount entries`),s(t,e.output.coefficients.length>=e.cellCount*r,`output.coefficients must hold ${r} entries per cell`),s(t,!e.output.rmse||e.output.rmse.length>=e.cellCount,`output.rmse must hold one entry per cell`),s(t,e.output.summary.length>=e.cellCount*2,`output.summary must hold two entries per cell`),s(t,e.parameters.length>=1,`parameters must hold 1 entries`),this.ticksPerYear=g(t,e.dateAxis,e.sliceCount);let l={values:e.sliceCount,mask:e.sliceCount,"output.coefficients":r,"output.summary":2,"output.fitted":e.sliceCount,"output.rmse":1},u=i(t,e,l);if(this.chunkProps=this.mode===`storage`?a(u,l,Math.floor(B/(e.sliceCount*r*Float32Array.BYTES_PER_ELEMENT))):u,this.mode===`storage`)for(let n of this.chunkProps)c(t,`QR basis scratch entries`,n.cellCount*e.sliceCount*r);this.estimatedCommandNodeCount=this.chunkProps.length*(this.mode===`storage`&&e.output.fitted?2:1),d(t,this.estimatedCommandNodeCount);let f=this.getInputs(),p=this.getOutputs();n(t,f,p,[{inputs:f,outputs:this.mode===`reference`?p:p.filter(t=>t!==e.output.fitted)},...this.mode===`storage`?this.chunkProps.flatMap(e=>e.output.fitted?[{inputs:[e.parameters,e.dateAxis.acquisitionTimes,e.output.coefficients,e.output.summary],outputs:[e.output.fitted]}]:[]):[]])}getInputs(){let{values:e,mask:t,parameters:n}=this.props;return[e,n,...t?[t]:[],...this.props.dateAxis?[this.props.dateAxis.acquisitionTimes]:[]]}getOutputs(){let{coefficients:e,summary:t,fitted:n,rmse:r}=this.props.output;return[e,t,...n?[n]:[],...r?[r]:[]]}getCommandNodes(e){if(r(e,this.id,[...this.getInputs(),...this.getOutputs()]),this.mode===`storage`){c(this.id,`storage binding count`,6+ +!!this.props.mask+ +!!this.props.output.rmse,1,e.device.limits.maxStorageBuffersPerShaderStage);for(let t of this.chunkProps)f(e.device,this.id,`QR basis scratch`,t.cellCount*this.props.sliceCount*(2+2*this.props.harmonicCount)*Float32Array.BYTES_PER_ELEMENT,`reduce the cell count per chunk or sliceCount`)}return this.chunkProps.flatMap((t,n)=>this.getChunkCommandNodes(e,{...t,id:`${this.id}-chunk-${n}`}))}getChunkCommandNodes(e,t){if(this.mode===`storage`)return this.getStorageCommandNodes(e,t);let n=t.id,{sliceCount:r,harmonicCount:i,values:a,mask:s,parameters:c,dateAxis:l,output:u}=t,d=2+2*i;return[o(e,{id:n,operation:`GPURasterHarmonicRegression`,invocationCount:t.cellCount,workgroupSize:8,bindings:[p(`values`,a),p(`parameters`,c),...s?[p(`mask`,s)]:[],v(l),p(`coefficients`,u.coefficients,!0),p(`summary`,u.summary,!0),...u.fitted?[p(`fitted`,u.fitted,!0)]:[],...u.rmse?[p(`rmse`,u.rmse,!0)]:[]],declarations:`${h}
${_(this.ticksPerYear,r)}
${y(r)}
${m(r,d)}`,body:`fillRasterSliceYears();
let period = parameters[parametersOffset];
var design: array<f32, ${r*d}>;
var response: array<f32, ${r}>;
var count = 0u;
for (var slice = 0u; slice < ${r}u; slice++) {
  let source = index * ${r}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) ${s?`|| mask[maskOffset + source] == 0u`:``}) { continue; }
  response[slice] = value;
  count++;
  for (var column = 0u; column < ${d}u; column++) { design[column * ${r}u + slice] = getHarmonicTrendColumn(slice, column, period); }
}
var code = 1u;
var fit: LeastSquaresResult;
if (count > ${d}u) {
  fit = leastSquares(design, response);
  code = select(2u, 0u, fit.valid);
}
for (var column = 0u; column < ${d}u; column++) {
  coefficients[coefficientsOffset + index * ${d}u + column] = select(missing(), fit.coefficients[column], code == 0u);
}
summary[summaryOffset + index * 2u] = code;
summary[summaryOffset + index * 2u + 1u] = count;
${u.fitted?`for (var slice = 0u; slice < ${r}u; slice++) {
  var prediction = 0.0;
  for (var column = 0u; column < ${d}u; column++) { prediction += fit.coefficients[column] * getHarmonicTrendColumn(slice, column, period); }
  fitted[fittedOffset + index * ${r}u + slice] = select(missing(), prediction, code == 0u);
}`:``}
${u.rmse?`rmse[rmseOffset + index] = select(missing(), sqrt(fit.residual / f32(count - min(count, ${d}u))), code == 0u);`:``}`})]}getStorageCommandNodes(n,r){let{sliceCount:i,harmonicCount:a,values:s,mask:c,parameters:l,dateAxis:u,output:d}=r,f=r.id,m=2+2*a,g=t(n,`${f}-qr-basis`,`float32`,r.cellCount*i*m,e.STORAGE),b=`${h}
${_(this.ticksPerYear,i)}
${y(i).replace(/var<private> sliceYears:[\s\S]*?fn getHarmonicTrendColumn/,`fn getHarmonicTrendColumn`).replaceAll(`sliceYears[slice]`,`rasterSliceYears(slice)`)}`,x=[o(n,{id:f,operation:`GPURasterHarmonicRegression`,variant:`storage-qr`,invocationCount:r.cellCount,workgroupSize:8,bindings:[p(`values`,s),p(`parameters`,l),...c?[p(`mask`,c)]:[],v(u),p(`qrBasis`,g,!0),p(`coefficients`,d.coefficients,!0),p(`summary`,d.summary,!0),...d.rmse?[p(`rmse`,d.rmse,!0)]:[]],declarations:`${b}
fn isHarmonicObservationPresent(cell: u32, slice: u32) -> bool {
  let source = cell * ${i}u + slice;
  return finite(values[valuesOffset + source]) ${c?`&& mask[maskOffset + source] != 0u`:``};
}
fn getHarmonicResponse(cell: u32, slice: u32) -> f32 {
  if (!isHarmonicObservationPresent(cell, slice)) { return 0.0; }
  return values[valuesOffset + cell * ${i}u + slice];
}
fn getMaskedHarmonicColumn(cell: u32, slice: u32, column: u32, period: f32) -> f32 {
  if (!isHarmonicObservationPresent(cell, slice)) { return 0.0; }
  return getHarmonicTrendColumn(slice, column, period);
}
${W(i,m)}`,body:`let period = parameters[parametersOffset];
let basisBase = qrBasisOffset + index * ${i*m}u;
var count = 0u;
for (var slice = 0u; slice < ${i}u; slice++) {
  if (isHarmonicObservationPresent(index, slice)) { count++; }
  for (var column = 0u; column < ${m}u; column++) {
    qrBasis[basisBase + column * ${i}u + slice] = getMaskedHarmonicColumn(index, slice, column, period);
  }
}
var code = 1u;
var fit: LeastSquaresResult;
if (count > ${m}u) {
  fit = leastSquares(index, period);
  code = select(2u, 0u, fit.valid);
}
for (var column = 0u; column < ${m}u; column++) {
  coefficients[coefficientsOffset + index * ${m}u + column] = select(missing(), fit.coefficients[column], code == 0u);
}
summary[summaryOffset + index * 2u] = code;
summary[summaryOffset + index * 2u + 1u] = count;
${d.rmse?`rmse[rmseOffset + index] = select(missing(), sqrt(fit.residual / f32(count - min(count, ${m}u))), code == 0u);`:``}`})];return d.fitted&&x.push(o(n,{id:`${f}-fitted`,operation:`GPURasterHarmonicRegression`,variant:`fitted`,invocationCount:r.cellCount,workgroupSize:8,bindings:[p(`parameters`,l),v(u),p(`coefficients`,d.coefficients),p(`summary`,d.summary),p(`fitted`,d.fitted,!0)],declarations:b,body:`let period = parameters[parametersOffset];
let code = summary[summaryOffset + index * 2u];
for (var slice = 0u; slice < ${i}u; slice++) {
  var prediction = 0.0;
  for (var column = 0u; column < ${m}u; column++) {
    prediction += coefficients[coefficientsOffset + index * ${m}u + column] * getHarmonicTrendColumn(slice, column, period);
  }
  fitted[fittedOffset + index * ${i}u + slice] = select(missing(), prediction, code == 0u);
}`})),x}};function W(e,t){return m(e,t).replace(`fn leastSquares(design: array<f32, ${e*t}>, response: array<f32, ${e}>)`,`fn leastSquares(cell: u32, period: f32)`).replace(`var basis = design;`,`let basisBase = qrBasisOffset + cell * ${e*t}u;`).replace(/basis\[([^\]]+)\]/g,`qrBasis[basisBase + $1]`).replaceAll(`response[row]`,`getHarmonicResponse(cell, row)`).replaceAll(`design[column * ${e}u + row]`,`getMaskedHarmonicColumn(cell, row, column, period)`)}var G=64,K=.08,[q,J]=k(2),Y=[-.2,1],X=[-.3,.3],Z=b({parameters:R,rebuildOn:[`harmonicCount`],buildCache:3,caption:`Synthetic seasonal cycle on measured Dixie NDVI levels, 64 irregular dates with cloud dips and gaps.`,createInputs:({aspect:e,random:t})=>{let n=N(e,t,{sliceCount:G,yearCount:3});return{...n,observed:n.observe(.14,K)}},build(e,t,n){let{grid:r,sliceCount:i}=t,a=Math.round(n.harmonicCount),o=e.upload(`series`,`float32`,t.observed),s=e.allocate(`parameters`,`float32`,1),c={coefficients:e.allocate(`coefficients`,`float32`,r.cellCount*(2+2*a)),summary:e.allocate(`summary`,`uint32`,r.cellCount*2),fitted:e.allocate(`fitted`,`float32`,r.cellCount*i),rmse:e.allocate(`rmse`,`float32`,r.cellCount)};return e.graph.add(new U({cellCount:r.cellCount,sliceCount:i,values:o.view,parameters:s.view,harmonicCount:a,dateAxis:P(e,t.times),output:{coefficients:c.coefficients.view,summary:c.summary.view,fitted:c.fitted.view,rmse:c.rmse.view}})),{outputs:{...c,level:e.upload(`level`,`float32`,t.level),observations:j(e,`observations`,i),curve:j(e,`curve`,i),residuals:j(e,`residuals`,i)},update:e=>{t.observed=t.observe(e.cloudFraction,K),o.write(t.observed),s.write(H({period:e.period}))}}},interaction:{kind:`probe`,hint:`probe a pixel`,rest:[.72,.5],resolve:(e,{inputs:t,outputs:n,values:r,world:i,peek:a})=>{let{grid:o,sliceCount:s}=t,c=I(o,i,e),l=t.observed.subarray(c.index*s,(c.index+1)*s),u=a(n.fitted),d=a(n.rmse),f=a(n.summary),p=u?u.subarray(c.index*s,(c.index+1)*s):[],m=Array.from(t.years);T(n.observations,m,l),T(n.curve,m,p),T(n.residuals,m,m.map((e,t)=>l[t]-(p[t]??NaN)));let h=[t.years[0],t.years[s-1]],g=w(t.times),_=f&&f[c.index*2]===0;return{spotlight:null,glyphs:[F(o,i,c),O(i,q),O(i,J),A(i,J,X,0),...E(i,q,h,g),...E(i,J,h,g)],readout:f?_?`rmse ${x(d?.[c.index]??NaN,3)} · ${f[c.index*2+1]} valid · period ${x(r.period)} · ${L(o,c)}`:`too few valid observations`:`reading the fit`}}},draw:({inputs:e,outputs:t,theme:n})=>{let r=[e.years[0],e.years[e.sliceCount-1]];return[M(t.level,e.grid,e.grid.mapLeft,{context:!0,ramp:`sequential`,range:[0,1]}),M(t.rmse,e.grid,e.grid.mapLeft,{ramp:`sequential`,range:[0,.12]}),S(t.observations,q,Y,r,C(n),4),S(t.curve,q,Y,r,D(n,0),3),S(t.residuals,J,X,r,D(n,1),4)]}});export{Z as default};
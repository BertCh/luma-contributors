import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-reduction-tNgOVQ_C.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as r}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i,r as a}from"./weights-construction-snapshot-CikLt8kl.js";import{i as o,t as s}from"./spatial-weights-loss-u59PcHkg.js";import{t as c}from"./spatial-statistics-row-dispatch-DNaskhxv.js";import{t as l}from"./spatial-statistics-writable-dispatches-BC35SiUi.js";import{i as u,s as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as f}from"./spatial-statistics-counts-wgsl-B8N7XLDs.js";import{t as p}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{c as m,r as h}from"./wgsl-common-DfyXiYB0.js";import{t as g}from"./scaled-quotient-wgsl-UDyhQag6.js";import{D as _,E as v,f as y,v as b,x,y as S}from"./gpu-contributor-utils-BIwLQON5.js";import{a as C,i as w,s as T,t as E}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{t as D}from"./spatial-statistics-population-output-B7_Z5i4X.js";var O=4096,k=4294963199,A=2**24-1;function j(t,r){let i=Math.ceil(r.rows/O),a=e(t,`${r.id}-partials`,`float32`,i*r.columns*3);return[n(t,{id:`${r.id}-blocks`,operation:`GPUMultivariateLocalGeary`,variant:`moment-blocks`,invocationCount:i*r.columns,bindings:[...r.bindings,{name:`partials`,view:a,type:`f32`,access:`read_write`}],declarations:r.declarations,body:`let column = index / ${i}u; let blockIndex = index % ${i}u;
var count = 0u; var mean = 0.0; var squares = 0.0;
let begin = blockIndex * ${O}u;
for (var row = begin; row < min(begin + ${O}u, ${r.rows}u); row++) {
 if (!included(row)) { continue; }
 let value = values[valuesOffset + row * ${r.columns}u + column]; count++;
 let delta = value - mean; mean += delta / f32(count); squares += delta * (value - mean);
}
let destination = partialsOffset + index * 3u; partials[destination] = f32(count); partials[destination + 1u] = mean; partials[destination + 2u] = squares;`}),n(t,{id:`${r.id}-merge`,operation:`GPUMultivariateLocalGeary`,variant:`moment-merge`,invocationCount:r.columns,bindings:[{name:`partials`,view:a,type:`f32`,access:`read`},{name:`variance`,view:r.variance,type:`f32`,access:`read_write`},...r.count?[{name:`populationCount`,view:r.count,type:`u32`,access:`read_write`}]:[]],declarations:`${r.declarations.split(`fn included`)[0]}\n${f}\n${p}`,body:`var count = 0u; var mean = 0.0; var squares = 0.0; var correction = 0.0;
for (var blockIndex = 0u; blockIndex < ${i}u; blockIndex++) {
 let source = partialsOffset + (index * ${i}u + blockIndex) * 3u;
 let blockCount = u32(partials[source]); if (blockCount == 0u) { continue; }
 // The empty aggregate has no mean: avoid squaring a large first block mean.
 if (count == 0u) { count = blockCount; mean = partials[source + 1u]; squares = partials[source + 2u]; continue; }
 let combined = count + blockCount; let delta = partials[source + 1u] - mean;
 let ratio = divideSpatialCounts(blockCount, combined);
 let term = partials[source + 2u] + multiplyBySpatialCount(delta * delta * ratio, count);
 let next = squares + term; correction += getAdditionRoundingError(squares, term, next); squares = next;
 mean += delta * ratio; count = combined;
}
variance[varianceOffset + index] = select(getNaN(), divideBySpatialCount(squares + correction, count), count > 0u);
${r.count?`if (index == 0u) { populationCount[populationCountOffset] = count; }`:``}`})]}function M(e={},t=new Float32Array(1)){let n=`getGPULocalSpatialHeteroscedasticityParameterValues`;return y(n,`target.length`,t.length,1),y(n,`residualPower`,e.residualPower??2,1,2),t[0]=e.residualPower??2,t}function N(e){let t=a(e,[`id`,`weights`,`values`,`mask`,`output`,`residualPower`,`variableCount`,`mode`]);return Object.freeze({...t,weights:i(t.weights),output:a(t.output,[`statistic`,`status`,`normalizedResiduals`,`residualCount`,`includedCount`,`count`])})}var P=class{props;id;residualPowerValues;constructor(e){this.props=e,e=N(e),this.props=e,this.id=e.id??`local-spatial-heteroscedasticity`,I(this.id,e,1);let t=[e.output.statistic,e.output.status,e.output.normalizedResiduals,e.output.residualCount];l(this.id,{statistic:e.output.statistic,status:e.output.status,normalizedResiduals:e.output.normalizedResiduals,residualCount:e.output.residualCount},[[`statistic`,`status`]]),e.output.residualCount&&(v(e.output.residualCount,[`uint32`],`${this.id} output.residualCount`),y(this.id,`output.residualCount.length`,e.output.residualCount.length,1,1),S(this.id,[e.output.residualCount],R(e)));let n=e.residualPower??2;typeof n==`number`?(y(this.id,`residualPower`,n,1,2),this.residualPowerValues=M({residualPower:n})):(v(n,[`float32`],`${this.id} residualPower`),y(this.id,`residualPower.length`,n.length,1),S(this.id,t,[n])),e.output.normalizedResiduals&&(v(e.output.normalizedResiduals,[`float32`],`${this.id} normalizedResiduals`),_(this.id,`normalizedResiduals`,e.output.normalizedResiduals,L(e)),S(this.id,[e.output.normalizedResiduals],R(e)))}getCommandNodes(i){let{id:a}=this;r(i,a,this.props);let o=s(i,a,[this.props.weights]),l={...this.props,weights:{...this.props.weights,status:{overflow:o.view}}},u=L(l);r(i,a,l,{"row scratch":u*4,"moment partials":Math.ceil(u/O)*(`variableCount`in l?Number(l.variableCount):1)*12});let p=E(l.weights);b(i.device,a,`residual scratch`,u*Float32Array.BYTES_PER_ELEMENT,`reduce the row count`);let _=e(i,`${a}-finite-residuals`,`float32`,u),v=e(i,`${a}-residual-counts`,`uint32`,u),y=e(i,`${a}-maximum-residual`,`float32`,1),S=e(i,`${a}-normalized-sum`,`float32`,1),C=l.output.residualCount??e(i,`${a}-residual-count`,`uint32`,1),T=e(i,`${a}-residuals`,`float32`,u),D=e(i,`${a}-mean`,`float32`,1);x(a,i,[l.values,l.mask,...w(l.weights),l.weights.status?.overflow,l.output.statistic,l.output.status,l.output.normalizedResiduals,typeof l.residualPower==`object`?l.residualPower:void 0]);let k=d(i,`${a}-parameters`,this.residualPowerValues??l.residualPower),A=[...o.nodes,...k.nodes,...p.map(e=>c(i,{rowBase:e.rowStart,id:`${a}-residuals${e.idSuffix}`,operation:`GPULocalSpatialHeteroscedasticity`,variant:`residuals`,invocationCount:e.rowCount,bindings:[...B(l,e.weights),{name:`parameters`,view:k.parameters,type:`f32`,access:`read`},{name:`residuals`,view:T,type:`f32`,access:`read_write`}],declarations:V(l,u,1,e),body:`var result = getNaN();
if (included(index)) {
  var weighted = 0.0;
  var total = 0.0;
  ${H(`weighted += weight * values[valuesOffset + neighbor]; total += weight;`,e)}
  if (total > 0.0) {
    let deviation = abs(values[valuesOffset + index] - weighted / total);
    result = select(deviation * deviation, deviation, parameters[parametersOffset] == 1.0);
  }
}
residuals[residualsOffset + index] = result;`})),n(i,{id:`${a}-finite-residuals`,operation:`GPULocalSpatialHeteroscedasticity`,variant:`finite-residuals`,invocationCount:u,bindings:[{name:`residuals`,view:T,type:`f32`,access:`read`},{name:`finiteResiduals`,view:_,type:`f32`,access:`read_write`},{name:`residualCounts`,view:v,type:`u32`,access:`read_write`}],declarations:m,body:`let value = residuals[residualsOffset + index];
let finite = isFiniteBits(value);
finiteResiduals[finiteResidualsOffset + index] = select(0.0, value, finite);
residualCounts[residualCountsOffset + index] = u32(finite);`}),...new t({id:`${a}-maximum-residual`,input:_,output:y,operation:`max`}).getCommandNodes(i),...new t({id:`${a}-residual-count`,input:v,output:C,operation:`sum`}).getCommandNodes(i),n(i,{id:`${a}-scale-residuals`,operation:`GPULocalSpatialHeteroscedasticity`,variant:`scale-residuals`,invocationCount:u,bindings:[{name:`finiteResiduals`,view:_,type:`f32`,access:`read_write`},{name:`maximumResidual`,view:y,type:`f32`,access:`read`}],declarations:g,body:`let maximum = maximumResidual[maximumResidualOffset];
var scaled = 0.0;
if (maximum > 0.0) {
  // A direct division can flush the reciprocal of a large maximum to zero.
  scaled = getScaledQuotient(finiteResiduals[finiteResidualsOffset + index], maximum, 1.0, 0);
}
finiteResiduals[finiteResidualsOffset + index] = scaled;`}),...new t({id:`${a}-normalized-sum`,input:_,output:S,operation:`sum`}).getCommandNodes(i),n(i,{id:`${a}-mean`,operation:`GPULocalSpatialHeteroscedasticity`,variant:`mean`,invocationCount:1,bindings:[{name:`maximumResidual`,view:y,type:`f32`,access:`read`},{name:`normalizedSum`,view:S,type:`f32`,access:`read`},{name:`residualCount`,view:C,type:`u32`,access:`read`},{name:`mean`,view:D,type:`f32`,access:`read_write`}],declarations:`${h}
${f}
// 2^126 is finite; splitting larger exponents avoids an overflowing ldexp scale.
const MAXIMUM_MEAN_SCALE_EXPONENT: i32 = 126;`,body:`let count = residualCount[residualCountOffset];
let maximum = frexp(maximumResidual[maximumResidualOffset]);
// Keep the maximum's exponent out of float products, including compiler reassociation.
let fraction = maximum.fract * divideBySpatialCount(normalizedSum[normalizedSumOffset], count);
let firstExponent = min(maximum.exp, MAXIMUM_MEAN_SCALE_EXPONENT);
let average = ldexp(ldexp(fraction, firstExponent), maximum.exp - firstExponent);
mean[meanOffset] = select(getNaN(), average, count > 0u);`}),...p.map(e=>c(i,{rowBase:e.rowStart,id:`${a}-finish${e.idSuffix}`,operation:`GPULocalSpatialHeteroscedasticity`,variant:`finish`,invocationCount:e.rowCount,bindings:[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:e.weights.weights,type:`f32`,access:`read`},{name:`residuals`,view:T,type:`f32`,access:`read`},{name:`mean`,view:D,type:`f32`,access:`read`},{name:`result`,view:l.output.statistic,type:`f32`,access:`read_write`},{name:`status`,view:l.output.status,type:`u32`,access:`read_write`},...l.weights.status?.overflow?[{name:`overflow`,view:l.weights.status.overflow,type:`u32`,access:`read`}]:[]],declarations:`${h}\n${m}\n${g}\n`,body:`var resultValue = getNaN(); var state = 1u;
if (isFiniteBits(residuals[residualsOffset + index])) {
  var total = 0.0; var weighted = 0.0;
  let rowBegin = offsets[offsetsOffset + ${z(e)}];
  let rowEnd = offsets[offsetsOffset + ${z(e)} + 1u];
  for (var slot = rowBegin; slot < min(rowEnd, CAPACITY); slot++) {
    let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
    if (neighbor >= ${u}u || neighbor == index || !(weight > 0.0) || !isFiniteBits(weight)) { continue; }
    let residual = residuals[residualsOffset + neighbor];
    if (isFiniteBits(residual)) { weighted += weight * residual; total += weight; }
  }
  state = 2u;
  if (total > 0.0 && mean[meanOffset] > 0.0) {
    if (isFiniteBits(weighted) && isFiniteBits(total) && isFiniteBits(mean[meanOffset])) {
      resultValue = getScaledQuotient(weighted, total, mean[meanOffset], 0);
    } else {
      resultValue = weighted / total / mean[meanOffset];
    }
    state = 0u;
  }
}
${l.weights.status?.overflow?`if (overflow[overflowOffset] != 0u) { state = 3u; resultValue = getNaN(); }`:``}
if (!isFiniteBits(resultValue) && state == 0u) { state = 2u; }
result[resultOffset + index] = resultValue; status[statusOffset + index] = state;`}))];return l.output.normalizedResiduals&&A.push(n(i,{id:`${a}-normalized-residuals`,operation:`GPULocalSpatialHeteroscedasticity`,variant:`normalized-residuals`,invocationCount:u,bindings:[{name:`residuals`,view:T,type:`f32`,access:`read`},{name:`mean`,view:D,type:`f32`,access:`read`},{name:`normalized`,view:l.output.normalizedResiduals,type:`f32`,access:`read_write`}],declarations:`${h}\n${m}\n${g}`,body:`let average = mean[meanOffset];
let residual = residuals[residualsOffset + index];
var normalizedValue = getNaN();
if (average > 0.0 && isFiniteBits(average)) {
  normalizedValue = residual;
  if (isFiniteBits(residual)) {
    normalizedValue = getScaledQuotient(residual, average, 1.0, 0);
  }
}
normalized[normalizedOffset + index] = normalizedValue;`})),A}},F=class{props;id;populationCount;constructor(e){if(this.props=e,e=N(e),this.props=e,this.id=e.id??`multivariate-local-geary`,this.populationCount=D(this.id,e.output),I(this.id,e,e.variableCount),e.mode!==void 0&&e.mode!==`tiled`&&e.mode!==`reference`)throw Error(`${this.id} mode must be tiled or reference`);e.mode===`reference`&&y(this.id,`weights.rowCount (LOCAL_VARIABILITY_REFERENCE_MAXIMUM_ROWS)`,L(e),1,A),this.populationCount&&(v(this.populationCount,[`uint32`],`${this.id} output.${e.output.includedCount?`includedCount`:`count`}`),y(this.id,`output.${e.output.includedCount?`includedCount`:`count`}.length`,this.populationCount.length,1),S(this.id,[this.populationCount],R(e))),l(this.id,{statistic:e.output.statistic,status:e.output.status,count:this.populationCount},[[`statistic`,`status`]])}getCommandNodes(t){let{id:i}=this;r(t,i,this.props);let a=s(t,i,[this.props.weights]),o={...this.props,weights:{...this.props.weights,status:{overflow:a.view}}},l=L(o);r(t,i,o,{"row scratch":l*4,"moment partials":Math.ceil(l/O)*(`variableCount`in o?Number(o.variableCount):1)*12});let u=E(o.weights),d=e(t,`${i}-variance`,`float32`,o.variableCount),f=B(o,u[0].weights);x(i,t,[o.values,o.mask,...w(o.weights),o.weights.status?.overflow,o.output.statistic,o.output.status]);let p=V(o,l,o.variableCount,u[0]),m=f.filter(e=>[`values`,`mask`,`overflow`].includes(e.name)),_=[...a.nodes,...o.mode===`reference`?[n(t,{id:`${i}-moments`,operation:`GPUMultivariateLocalGeary`,variant:`moments`,invocationCount:o.variableCount,bindings:[...m,{name:`variance`,view:d,type:`f32`,access:`read_write`},...this.populationCount?[{name:`populationCount`,view:this.populationCount,type:`u32`,access:`read_write`}]:[]],declarations:p,body:`var count = 0.0; var mean = 0.0; var squared = 0.0;
for (var row = 0u; row < ${l}u; row++) {
  if (!included(row)) { continue; }
  let value = values[valuesOffset + row * ${o.variableCount}u + index];
  count += 1.0; let difference = value - mean; mean += difference / count; squared += difference * (value - mean);
}
variance[varianceOffset + index] = select(getNaN(), squared / count, count > 0.0);
${this.populationCount?`if (index == 0u) { populationCount[populationCountOffset] = u32(count); }`:``}`})]:j(t,{id:i,rows:l,columns:o.variableCount,bindings:m,declarations:p,variance:d,count:this.populationCount}),...u.map(e=>c(t,{rowBase:e.rowStart,id:`${i}-statistic${e.idSuffix}`,operation:`GPUMultivariateLocalGeary`,variant:`statistic`,invocationCount:e.rowCount,bindings:[...B(o,e.weights).filter(e=>e.name!==`overflow`),{name:`variance`,view:d,type:`f32`,access:`read`},{name:`result`,view:o.output.statistic,type:`f32`,access:`read_write`},{name:`status`,view:o.output.status,type:`u32`,access:`read_write`}],declarations:`${V({...o,weights:{...o.weights,status:void 0}},l,o.variableCount,e)}\n${g}`,body:`var state = 1u; var resultValue = getNaN();
if (included(index)) {
  var sum = 0.0; var total = 0.0; var defined = true;
  for (var column = 0u; column < ${o.variableCount}u; column++) { defined = defined && variance[varianceOffset + column] > 0.0 && isFiniteBits(variance[varianceOffset + column]); }
  if (defined) {
    ${H(`var distance = 0.0;
      for (var column = 0u; column < ${o.variableCount}u; column++) {
        let difference = values[valuesOffset + index * ${o.variableCount}u + column] - values[valuesOffset + neighbor * ${o.variableCount}u + column];
        // Normalize before squaring: the squared raw difference can overflow although the
        // variance and the dimensionless Geary statistic are finite. Exponent scaling also
        // prevents Metal's reciprocal lowering from flushing a large denominator to zero.
        let standardized = getScaledQuotient(difference, sqrt(variance[varianceOffset + column]), 1.0, 0);
        distance += standardized * standardized;
      }
      sum += weight * distance; total += weight;`,e)}
  }
  state = 2u;
  if (defined && total > 0.0) { resultValue = sum / total / ${o.variableCount}.0; state = select(2u, 0u, isFiniteBits(resultValue)); }
}
result[resultOffset + index] = resultValue; status[statusOffset + index] = state;`}))];return o.weights.status?.overflow&&_.push(n(t,{id:`${i}-upstream-status`,operation:`GPUMultivariateLocalGeary`,variant:`upstream-status`,invocationCount:l,bindings:[{name:`overflow`,view:o.weights.status.overflow,type:`u32`,access:`read`},{name:`result`,view:o.output.statistic,type:`f32`,access:`read_write`},{name:`status`,view:o.output.status,type:`u32`,access:`read_write`}],declarations:h,body:`if (overflow[overflowOffset] != 0u) { result[resultOffset + index] = getNaN(); status[statusOffset + index] = 3u; }`})),_}};function I(e,t,n){let r=T(e,t.weights);y(e,`weights.rowCount (LOCAL_VARIABILITY_MAXIMUM_ROWS)`,r,1,k),y(e,`variableCount (LOCAL_VARIABILITY_MAXIMUM_VARIABLE_COUNT)`,n,1,32),y(e,`values.length (uint32 row-major addressing)`,t.values.length,1,4294967295),u(e,{statistic:t.output.statistic,status:t.output.status});let i=[t.values,t.output.statistic,t.output.status,t.mask],a=[r*n,r,r,r];for(let t=0;t<i.length;t++){let n=i[t];if(n&&(v(n,[t<2?`float32`:`uint32`],e),n.length!==a[t]))throw Error(`${e}: invalid row count`)}S(e,[t.output.statistic,t.output.status],R(t))}function L(e){return C(e.weights)?e.weights.rowCount:e.weights.offsets.length-1}function R(e){return[e.values,e.mask,...o(e.id??`local-variability`,[e.weights]),...w(e.weights)]}function z(e){return`index - WEIGHTS_ROW_START`}function B(e,t){return[{name:`values`,view:e.values,type:`f32`,access:`read`},{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t.weights,type:`f32`,access:`read`},...e.mask?[{name:`mask`,view:e.mask,type:`u32`,access:`read`}]:[],...e.weights.status?.overflow?[{name:`overflow`,view:e.weights.status.overflow,type:`u32`,access:`read`}]:[]]}function V(e,t,n,r){return`${h}\n${m}
const ROWS: u32 = ${t}u;

fn included(row: u32) -> bool {
  if (row >= ROWS) { return false; }
  ${e.mask?`if (mask[maskOffset + row] == 0u) { return false; }`:``}
  ${e.weights.status?.overflow?`if (overflow[overflowOffset] != 0u) { return false; }`:``}
  for (var column = 0u; column < ${n}u; column++) {
    if (!isFiniteBits(values[valuesOffset + row * ${n}u + column])) { return false; }
  }
  return true;
}`}function H(e,t){let n=z(t);return`for (var slot = offsets[offsetsOffset + ${n}]; slot < min(offsets[offsetsOffset + ${n} + 1u], CAPACITY); slot++) {
    let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
    if (neighbor == index || !included(neighbor) || !(weight > 0.0) || !isFiniteBits(weight)) { continue; }
    ${e}
  }`}export{F as n,P as t};
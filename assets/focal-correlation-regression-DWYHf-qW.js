import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{r as t}from"./kernel-C3tfQOn4.js";import{c as n,l as r,n as i,r as a,s as o,u as s}from"./raster-band-utils-BcTxc8_Y.js";import{E as c,T as l,f as u,x as d}from"./gpu-contributor-utils-BIwLQON5.js";import{o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{u as p}from"./wgsl-common-DfyXiYB0.js";import{t as m}from"./define-run-B-brkCe4.js";import{i as h,l as g}from"./inputs-CJKKIRKK.js";import{c as _}from"./pointer-overlay-Qh8qflcA.js";import{$i as v}from"./entry-preview-D70TdSbK.js";import{i as y,r as b}from"./gestures-C1AoqRjN.js";import{i as x,n as S,t as C}from"./bands-CVDIl0aE.js";function w(e){return`
${p}
const WIDTH: i32 = ${e.grid.width};
const HEIGHT: i32 = ${e.grid.height};
const RADIUS: i32 = ${e.radius};
fn reflectCoordinate(value: i32, length: i32) -> i32 {
  if (length <= 1) { return 0; }
  let period = (length - 1) * 2;
  let reflected = ((value % period) + period) % period;
  return select(reflected, period - reflected, reflected >= length);
}
fn getSampleIndex(column: i32, row: i32) -> i32 {
  ${e.borderMode===`clamp`?`return clamp(row, 0, HEIGHT - 1) * WIDTH + clamp(column, 0, WIDTH - 1);`:e.borderMode===`reflect`?`return reflectCoordinate(row, HEIGHT) * WIDTH + reflectCoordinate(column, WIDTH);`:`if (column < 0 || column >= WIDTH || row < 0 || row >= HEIGHT) { return -1; }
  return row * WIDTH + column;`}
}
fn getPair(sampleIndex: i32) -> vec2<f32> {
  if (sampleIndex < 0) { return vec2<f32>(bitcast<f32>(0x7fc00000u | (u32(sampleIndex) & 0u))); }
  return vec2<f32>(predictor[predictorOffset + u32(sampleIndex)], response[responseOffset + u32(sampleIndex)]);
}
fn isPairValid(pair: vec2<f32>) -> bool {
  return isFiniteValue(pair.x) && isFiniteValue(pair.y);
}`}function T(e){return`
let column = i32(index % u32(WIDTH));
let row = i32(index / u32(WIDTH));
let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
var count = 0u;
var missingPair = ${e.propagateCenterNoData?`!isPairValid(getPair(i32(index)))`:`false`};
var anchor = vec2<f32>(0.0);
var totalDifference = vec2<f32>(0.0);
for (var rowOffset = -RADIUS; rowOffset <= RADIUS; rowOffset++) {
  for (var columnOffset = -RADIUS; columnOffset <= RADIUS; columnOffset++) {
    let pair = getPair(getSampleIndex(column + columnOffset, row + rowOffset));
    if (!isPairValid(pair)) { ${e.noDataPolicy===`propagate`?`missingPair = true;`:``} continue; }
    if (count == 0u) { anchor = pair; }
    totalDifference += pair - anchor;
    count++;
  }
}
pairedCount[pairedCountOffset + index] = count;
var code = select(0u, 1u, count < ${e.minimumCount??2}u) | select(0u, 2u, missingPair);
var covariance = invalid;
var correlation = invalid;
var slope = invalid;
var intercept = invalid;
var rSquared = invalid;
var residualSumSquares = invalid;
if (code == 0u) {
  let meanDifference = totalDifference / f32(count);
  var predictorSquares = 0.0;
  var responseSquares = 0.0;
  var crossProducts = 0.0;
  var centeredTotals = vec2<f32>(0.0);
  for (var rowOffset = -RADIUS; rowOffset <= RADIUS; rowOffset++) {
    for (var columnOffset = -RADIUS; columnOffset <= RADIUS; columnOffset++) {
      let pair = getPair(getSampleIndex(column + columnOffset, row + rowOffset));
      if (!isPairValid(pair)) { continue; }
      let centered = (pair - anchor) - meanDifference;
      centeredTotals += centered;
      predictorSquares += centered.x * centered.x;
      responseSquares += centered.y * centered.y;
      crossProducts += centered.x * centered.y;
    }
  }
  // A backend may reassociate (pair - anchor) - meanDifference into pair minus a
  // rounded large centroid. Correct its nonzero residual mean using only small values.
  let correctedPredictorSquares = predictorSquares - centeredTotals.x * (centeredTotals.x / f32(count));
  let correctedResponseSquares = responseSquares - centeredTotals.y * (centeredTotals.y / f32(count));
  let correctedCrossProducts = crossProducts - centeredTotals.x * (centeredTotals.y / f32(count));
  if (!isFiniteValue(predictorSquares) || !isFiniteValue(responseSquares) || !isFiniteValue(crossProducts) ||
      !isPairValid(centeredTotals) || !isFiniteValue(correctedPredictorSquares) ||
      !isFiniteValue(correctedResponseSquares) || !isFiniteValue(correctedCrossProducts)) {
    code |= 16u;
  } else {
    predictorSquares = max(0.0, correctedPredictorSquares);
    responseSquares = max(0.0, correctedResponseSquares);
    crossProducts = correctedCrossProducts;
    covariance = crossProducts / f32(count${e.covarianceNormalization===`population`?``:` - 1u`});
    if (predictorSquares == 0.0) { code |= 4u; }
    if (responseSquares == 0.0) { code |= 8u; }
    if (predictorSquares > 0.0) {
      slope = crossProducts / predictorSquares;
      intercept = (anchor.y - slope * anchor.x) + (meanDifference.y - slope * meanDifference.x);
      residualSumSquares = 0.0;
      var residualTotal = 0.0;
      for (var rowOffset = -RADIUS; rowOffset <= RADIUS; rowOffset++) {
        for (var columnOffset = -RADIUS; columnOffset <= RADIUS; columnOffset++) {
          let pair = getPair(getSampleIndex(column + columnOffset, row + rowOffset));
          if (!isPairValid(pair)) { continue; }
          let centered = (pair - anchor) - meanDifference;
          let residual = centered.y - slope * centered.x;
          residualTotal += residual;
          residualSumSquares += residual * residual;
        }
      }
      // The same centroid rounding adds a constant to every fitted residual.
      // Remove its squared mean without rebuilding any large original-coordinate mean.
      let correctedResidualSquares = residualSumSquares - residualTotal * (residualTotal / f32(count));
      if (!isFiniteValue(residualSumSquares) || !isFiniteValue(correctedResidualSquares)) {
        code |= 16u;
      } else {
        residualSumSquares = max(0.0, correctedResidualSquares);
      }
      if (responseSquares > 0.0) {
        // Divide successively to avoid overflowing the product of the two squared sums.
        correlation = clamp((crossProducts / sqrt(predictorSquares)) / sqrt(responseSquares), -1.0, 1.0);
        rSquared = clamp(1.0 - residualSumSquares / responseSquares, 0.0, 1.0);
      }
      if (!isFiniteValue(slope) || !isFiniteValue(intercept) || !isFiniteValue(residualSumSquares) ||
          (responseSquares > 0.0 && (!isFiniteValue(correlation) || !isFiniteValue(rSquared)))) {
        code |= 16u;
      }
    }
  }
}
if ((code & 16u) != 0u) {
  covariance = invalid; correlation = invalid; slope = invalid;
  intercept = invalid; rSquared = invalid; residualSumSquares = invalid;
}
let destination = metricsOffset + index * 6u;
metrics[destination] = covariance;
metrics[destination + 1u] = correlation;
metrics[destination + 2u] = slope;
metrics[destination + 3u] = intercept;
metrics[destination + 4u] = rSquared;
metrics[destination + 5u] = residualSumSquares;
status[statusOffset + index] = code;`}var E=6,D=[`covariance`,`correlation`,`slope`,`intercept`,`rSquared`,`residualSumSquares`],O=class{id;props;requiredHalo;constructor(e){this.id=e.id??`raster-focal-pairs`,this.props=e,this.requiredHalo=e.radius;let t=s(this.id,e.grid);o(e.predictor,e.grid,`${this.id} predictor`),o(e.response,e.grid,`${this.id} response`),u(this.id,`radius`,e.radius,0,8),u(this.id,`minimumCount`,e.minimumCount??2,2,289),l(this.id,[`nodata`,`clamp`,`reflect`].includes(e.borderMode??`nodata`),`borderMode must be 'nodata', 'clamp' or 'reflect'`),l(this.id,[`ignore`,`propagate`].includes(e.noDataPolicy??`ignore`),`noDataPolicy must be 'ignore' or 'propagate'`),l(this.id,[`sample`,`population`].includes(e.covarianceNormalization??`sample`),`covarianceNormalization must be 'sample' or 'population'`),l(this.id,e.propagateCenterNoData===void 0||typeof e.propagateCenterNoData==`boolean`,`propagateCenterNoData must be a boolean`),l(this.id,D.some(t=>e.output[t]),`output must request at least one float metric`),u(this.id,`grid cell count`,t,1,Math.floor(4294967295/E));for(let[n,r]of Object.entries(e.output))r&&(c(r,[n===`pairedCount`||n===`status`?`uint32`:`float32`],this.id),l(this.id,r.length>=t,`output.${n} must hold at least one entry per cell`));r(this.id,Object.values(e.output),[...i(e.predictor),...i(e.response)])}getCommandNodes(r){let o=this.props,s=o.grid.width*o.grid.height;e(r,this.id,[...i(o.predictor),...i(o.response),...Object.values(o.output)],{"canonical band planes":s*4,"focal metrics":s*D.length*4});let{id:c,props:l}=this;n(c,r,l.predictor,[]),n(c,r,l.response,[]),d(c,r,Object.values(l.output));let u=a(r,`${c}-predictor`,l.predictor,l.grid,!0),p=a(r,`${c}-response`,l.response,l.grid,!0),m=l.grid.width*l.grid.height,h=t(r,`${c}-metrics`,`float32`,m*D.length),g=f(r,{id:`${c}-moments`,operation:`GPURasterFocalPairs`,invocationCount:m,bindings:[{name:`predictor`,view:u.band.storage.values,type:`f32`,access:`read`},{name:`response`,view:p.band.storage.values,type:`f32`,access:`read`},{name:`metrics`,view:h,type:`f32`,access:`read_write`},{name:`pairedCount`,view:l.output.pairedCount,type:`u32`,access:`read_write`},{name:`status`,view:l.output.status,type:`u32`,access:`read_write`}],declarations:w(l),body:T(l)}),_=D.flatMap((e,t)=>l.output[e]?[{name:e,slot:t,view:l.output[e]}]:[]),v=f(r,{id:`${c}-outputs`,operation:`GPURasterFocalPairs`,invocationCount:m,bindings:[{name:`metrics`,view:h,type:`f32`,access:`read`},..._.map(({name:e,view:t})=>({name:e,view:t,type:`f32`,access:`read_write`}))],body:_.map(({name:e,slot:t})=>`${e}[${e}Offset + index] = metrics[metricsOffset + index * 6u + ${t}u];`).join(`
`)});return[...u.nodes,...p.nodes,g,v]}},k=[`slope`,`correlation`,`rSquared`,`intercept`],A=[.03,.05,.3,.36],j=289,M=m({parameters:v,rebuildOn:[`radius`,`minimumCount`],caption:`Predictor and response bands in; the local regression of one on the other out.`,createInputs:({random:e,world:t})=>{let n=h(t,56),r=g(e,2,2),i=Float32Array.from(n.values,()=>e()-.5),a=Float32Array.from(n.values,(e,t)=>{let i=Math.floor(t/n.columns),a=t%n.columns;return 2*r((a+.5)/n.columns,(i+.5)/n.rows)-1});return{...n,noise:i,slope:a,scatterX:[0,1],scatterY:[-1,1]}},build(e,t,n){let r=t.values.length,i=x(e,t,`predictor`,t.values),a=x(e,t,`response`,t.values),o={pairedCount:e.allocate(`paired-count`,`uint32`,r),status:e.allocate(`status`,`uint32`,r),correlation:e.allocate(`correlation`,`float32`,r),slope:e.allocate(`slope`,`float32`,r),intercept:e.allocate(`intercept`,`float32`,r),rSquared:e.allocate(`r-squared`,`float32`,r)};return e.graph.add(new O({grid:S(t),predictor:i.band,response:a.band,radius:n.radius,minimumCount:n.minimumCount,output:{pairedCount:o.pairedCount.view,status:o.status.view,correlation:o.correlation.view,slope:o.slope.view,intercept:o.intercept.view,rSquared:o.rSquared.view}})),{outputs:{...o,predictor:i.buffer,windowPredictor:e.allocate(`window-predictor`,`float32`,j),windowResponse:e.allocate(`window-response`,`float32`,j)},update:e=>a.buffer.write(Float32Array.from(t.values,(n,r)=>n*t.slope[r]+t.noise[r]*e.noise))}},interaction:{kind:`window`,hint:`move the regression window`,rest:[.5,.5],resolve:(e,{inputs:t,outputs:n,values:r,world:i,peek:a})=>{let{column:o,row:s,index:c}=b(t,i,e),l=new Float32Array(j).fill(NaN),u=new Float32Array(j).fill(NaN),d=0;for(let e=Math.max(0,s-r.radius);e<=Math.min(t.rows-1,s+r.radius);e++)for(let n=Math.max(0,o-r.radius);n<=Math.min(t.columns-1,o+r.radius);n++){let i=e*t.columns+n;l[d]=t.values[i],u[d]=t.values[i]*t.slope[i]+t.noise[i]*r.noise,d++}let f=e=>e.filter(Number.isFinite),p=e=>{let t=f(e),n=Math.min(...t),r=Math.max(...t),i=Math.max((r-n)*.15,.001);return[n-i,r+i]};t.scatterX=p(l),t.scatterY=p(u),n.windowPredictor.write(l),n.windowResponse.write(u);let m=a(n.slope),h=a(n.intercept),g=a(n.correlation),[v,x]=t.scatterX,[S,C]=t.scatterY,w=(e,t)=>[i.width*(A[0]+(e-v)/(x-v)*(A[2]-A[0])),i.height*(A[1]+(t-S)/(C-S)*(A[3]-A[1]))],T=e=>h&&m?h[c]+m[c]*e:0;return{glyphs:[y(t,i,o,s,r.radius),{shape:`box`,minimum:w(v,S),maximum:w(x,C)},...m&&h?[{shape:`line`,from:w(v,T(v)),to:w(x,T(x))}]:[]],readout:`slope ${m?_(m[c]):`-`} | r ${g?_(g[c]):`-`} | ${d} pairs`}}},draw:({inputs:e,outputs:t,values:n})=>{let r=k[Math.round(n.metric)];return[C(e,t.predictor,{context:!0}),C(e,t[r],{ramp:r===`slope`||r===`correlation`?`diverging`:`sequential`,range:r===`slope`?[-1.5,1.5]:r===`correlation`?[-1,1]:`auto`,smoothing:!0}),{layer:`chart`,kind:`scatter`,values:t.windowResponse,xValues:t.windowPredictor,xRange:e.scatterX,range:e.scatterY,color:[1,1,1,.9],radius:3,rectangle:A}]}});export{M as default};
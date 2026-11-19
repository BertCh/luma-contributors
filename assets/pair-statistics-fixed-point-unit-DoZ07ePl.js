import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as n}from"./column-quantiles-nodes-CFuZJihC.js";import{t as r}from"./pair-statistics-grid-L8bKgCyF.js";var i=2**-32,a=2**-12,o=1.3489795,s=1e-30;function c(c,l){let{id:u,operation:d,values:f,mask:p,statistics:m}=l,h=f.length,g=e(c,`${u}-unit-parameters`,`float32`,6),_=e(c,`${u}-unit-quantiles`,`float32`,2),v=e(c,`${u}-unit-valid-count`,`uint32`,1),y=e(c,`${u}-unit-variance`,`float32`,1);return{nodes:[t(c,{id:`${u}-unit-parameters`,operation:d,variant:`unit-parameters`,bindings:[{name:`quantileParameters`,view:g,type:`f32`,access:`read_write`}],invocationCount:1,body:`quantileParameters[quantileParametersOffset] = 3.0;
  quantileParameters[quantileParametersOffset + 1u] = 0.0;
  quantileParameters[quantileParametersOffset + 2u] = 1.0;
  quantileParameters[quantileParametersOffset + 3u] = 0.0;
  quantileParameters[quantileParametersOffset + 4u] = 0.25;
  quantileParameters[quantileParametersOffset + 5u] = 0.75;`}),...n(c,{id:`${u}-unit-quantile`,operation:d,values:f,mask:p,parameters:g,quantileCount:2,output:{quantiles:_,validCount:v}}),t(c,{id:`${u}-unit-variance`,operation:d,variant:`unit-variance`,bindings:[{name:`statistics`,view:m,type:`f32`,access:`read`},{name:`quantiles`,view:_,type:`f32`,access:`read`},{name:`unitVariance`,view:y,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${r}
const MOMENTS: u32 = ${h}u;`,body:`let variance = statistics[statisticsOffset + MOMENTS + 2u] / statistics[statisticsOffset + MOMENTS];
  let spread = (quantiles[quantilesOffset + 1u] - quantiles[quantilesOffset]) / ${o};
  let usable = isFiniteValue(spread) && spread > 0.0;
  let robustVariance = select(variance * ${a}, spread * spread, usable);
  let unit = clamp(robustVariance, variance * ${i}, variance);
  unitVariance[unitVarianceOffset] =
    select(0.0, unit, isFiniteValue(variance) && variance > ${s});`})],unitVariance:y}}export{c as t};
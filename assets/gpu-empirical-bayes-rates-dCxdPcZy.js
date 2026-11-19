import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-reduction-tNgOVQ_C.js";import{t as r}from"./statistics-kernel-node-lacKNERm.js";import{E as i,f as a,x as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{t as c}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as l}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as u}from"./status-aOwAAXek.js";import{n as d,t as f}from"./float32-values-60_XMjYS.js";import{t as p}from"./spatial-autocorrelation-kernels-CCrGBZuG.js";import{t as m}from"./spatial-statistics-population-output-B7_Z5i4X.js";var h=`GPUEmpiricalBayesRates`,g=256,_=4294967039,v={count:0,eventSum:1,populationSum:2,pooledRate:3,weightedRateVariance:4,priorVariance:5,clampedPriorVariance:6,length:8},y=class{id;props;populationCount;constructor(e){let n=e.id??`empirical-bayes-rates`;this.id=n,this.props=e,this.populationCount=m(n,e.output);let{output:r}=e,o=d(e.events,`${n} events`);if(a(n,`events rows`,o,1,_),o<1)throw Error(`${n} events must hold at least one row`);if(d(e.populations,`${n} populations`)!==o)throw Error(`${n} populations length must equal the events length`);if(e.mask&&(t(e.mask,`${n} mask`),e.mask.length!==o))throw Error(`${n} mask length must equal the events length`);for(let[e,t]of[[`standardizedRates`,r.standardizedRates],[`smoothedRates`,r.smoothedRates],[`rawRates`,r.rawRates],[`excessRisk`,r.excessRisk]])if(t&&(i(t,[`float32`],`${n} output.${e}`),t.length!==o))throw Error(`${n} output.${e} length must equal the events length`);if(r.summary&&(i(r.summary,[`float32`],`${n} output.summary`),r.summary.length<v.length))throw Error(`${n} output.summary is too short`);if(!r.standardizedRates&&!r.smoothedRates&&!r.rawRates&&!r.summary&&!r.excessRisk&&!this.populationCount&&!r.status?.invalidCount)throw Error(`${n} requests no output`);u(n,r.status??{});for(let[e,t]of[[r.includedCount?`includedCount`:`count`,this.populationCount],[`status.invalidCount`,r.status?.invalidCount]])t&&(i(t,[`uint32`],`${n} output.${e}`),a(n,`output.${e} length`,t.length,1,1));l(n,{...r,status:void 0,...r.status}),s(n,[r.standardizedRates,r.smoothedRates,r.rawRates,r.summary,r.excessRisk,this.populationCount,r.status?.invalidCount],[e.events,e.populations,e.mask])}getCommandNodes(t){let{id:i,props:a}=this;c(t,i,a,{"summary moments":v.length*4,"row conversion and invalid flags":a.events.length*4,"three-column partials":Math.ceil(a.events.length/g)*3*4});let{mask:s,output:l}=a;o(i,t,[a.events,a.populations,s,l.standardizedRates,l.smoothedRates,l.rawRates,l.excessRisk,l.summary]);let u=f(t,`${i}-events`,a.events),d=f(t,`${i}-populations`,a.populations),m=u.view,_=d.view,y=m.length,b=Math.ceil(y/g),x=e(t,`${i}-partials-a`,`float32`,3*b),S=e(t,`${i}-partials-b`,`float32`,b),C=e(t,`${i}-totals-a`,`float32`,3),w=l.summary??e(t,`${i}-summary`,`float32`,v.length),T=e(t,`${i}-count-partials`,`uint32`,b),E=this.populationCount??e(t,`${i}-count`,`uint32`,1),D=v,O=a.clampPriorVariance??!0,k=`const ROWS: u32 = ${y}u;
const LANES: u32 = ${g}u;
const GROUPS: u32 = ${b}u;
${p}
var<workgroup> scratch: array<f32, ${g}>;
// Fixed-order tree sum of one value per lane. Every lane of the workgroup must call it together.
fn reduceSum(value: f32, lane: u32) -> f32 {
  workgroupBarrier();
  scratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    workgroupBarrier();
  }
  return scratch[0];
}`,A=s?[{name:`mask`,view:s,type:`u32`,access:`read`}]:[],j=`${s?`mask[maskOffset + index] != 0u && `:``}isFiniteFloat(e) && isFiniteFloat(b) && b > 0.0`,M=`var e = 0.0;
  var b = 0.0;
  var included = false;
  if (index < ROWS) {
    e = events[eventsOffset + index];
    b = populations[populationsOffset + index];
    included = ${j};
  }`,N=[...u.nodes,...d.nodes,r(t,{id:`${i}-partials-a`,operation:h,variant:`partials-a`,bindings:[{name:`events`,view:m,type:`f32`,access:`read`},{name:`populations`,view:_,type:`f32`,access:`read`},...A,{name:`partialsA`,view:x,type:`f32`,access:`read_write`},{name:`countPartials`,view:T,type:`u32`,access:`read_write`}],invocationCount:b*g,workgroupSize:g,guardIndex:!1,declarations:k,body:`let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  ${M}
  let count = reduceSum(select(0.0, 1.0, included), lane);
  let eventSum = reduceSum(select(0.0, e, included), lane);
  let populationSum = reduceSum(select(0.0, b, included), lane);
  if (lane == 0u) {
    countPartials[countPartialsOffset + group] = u32(count);
    partialsA[partialsAOffset + group] = count;
    partialsA[partialsAOffset + GROUPS + group] = eventSum;
    partialsA[partialsAOffset + 2u * GROUPS + group] = populationSum;
  }`}),...new n({id:`${i}-exact-count`,input:T,output:E,operation:`sum`}).getCommandNodes(t),r(t,{id:`${i}-totals-a`,operation:h,variant:`totals-a`,bindings:[{name:`partialsA`,view:x,type:`f32`,access:`read`},{name:`totalsA`,view:C,type:`f32`,access:`read_write`}],invocationCount:3*g,workgroupSize:g,guardIndex:!1,declarations:k,body:`let lane = localInvocationIndex;
  let column = index / LANES;
  var sum = 0.0;
  for (var group = lane; group < GROUPS; group += LANES) {
    sum += partialsA[partialsAOffset + column * GROUPS + group];
  }
  let total = reduceSum(sum, lane);
  if (lane == 0u) {
    totalsA[totalsAOffset + column] = total;
  }`}),r(t,{id:`${i}-partials-b`,operation:h,variant:`partials-b`,bindings:[{name:`events`,view:m,type:`f32`,access:`read`},{name:`populations`,view:_,type:`f32`,access:`read`},...A,{name:`totalsA`,view:C,type:`f32`,access:`read`},{name:`partialsB`,view:S,type:`f32`,access:`read_write`}],invocationCount:b*g,workgroupSize:g,guardIndex:!1,declarations:k,body:`let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  ${M}
  let pooledRate = totalsA[totalsAOffset + 1u] / totalsA[totalsAOffset + 2u];
  let deviation = e / b - pooledRate;
  let total = reduceSum(select(0.0, b * deviation * deviation, included), lane);
  if (lane == 0u) {
    partialsB[partialsBOffset + group] = total;
  }`}),r(t,{id:`${i}-summary`,operation:h,variant:`summary`,bindings:[{name:`totalsA`,view:C,type:`f32`,access:`read`},{name:`exactCount`,view:E,type:`u32`,access:`read`},{name:`partialsB`,view:S,type:`f32`,access:`read`},{name:`summary`,view:w,type:`f32`,access:`read_write`}],invocationCount:g,workgroupSize:g,guardIndex:!1,declarations:k,body:`let lane = localInvocationIndex;
  var sum = 0.0;
  for (var group = lane; group < GROUPS; group += LANES) {
    sum += partialsB[partialsBOffset + group];
  }
  let deviationSum = reduceSum(sum, lane);
  if (lane == 0u) {
    let count = exactCount[exactCountOffset];
    let eventSum = totalsA[totalsAOffset + 1u];
    let populationSum = totalsA[totalsAOffset + 2u];
    let pooledRate = eventSum / populationSum;
    let weightedVariance = deviationSum / populationSum;
    summary[summaryOffset + ${D.count}u] = f32(count);
    summary[summaryOffset + ${D.eventSum}u] = eventSum;
    summary[summaryOffset + ${D.populationSum}u] = populationSum;
    summary[summaryOffset + ${D.pooledRate}u] = pooledRate;
    summary[summaryOffset + ${D.weightedRateVariance}u] = weightedVariance;
    let priorVariance = weightedVariance - pooledRate / (divideBySpatialCount(populationSum, count));
    summary[summaryOffset + ${D.priorVariance}u] = priorVariance;
    summary[summaryOffset + ${D.clampedPriorVariance}u] = max(priorVariance, 0.0);
    for (var field = ${D.clampedPriorVariance+1}u; field < ${D.length}u; field++) {
      summary[summaryOffset + field] = 0.0;
    }
  }`})];if(l.status?.invalidCount){let a=e(t,`${i}-invalid-flags`,`uint32`,y);N.push(r(t,{id:`${i}-invalid-flags`,operation:h,invocationCount:y,bindings:[{name:`events`,view:m,type:`f32`,access:`read`},{name:`populations`,view:_,type:`f32`,access:`read`},...A,{name:`invalidFlags`,view:a,type:`u32`,access:`read_write`}],declarations:k,body:`let e = events[eventsOffset + index]; let b = populations[populationsOffset + index];
  invalidFlags[invalidFlagsOffset + index] = u32(${s?`mask[maskOffset + index] != 0u && `:``}!(isFiniteFloat(e) && isFiniteFloat(b) && b > 0.0));`}),...new n({id:`${i}-invalid-count`,input:a,output:l.status.invalidCount,operation:`sum`}).getCommandNodes(t))}let P=[],F=[],I=[];for(let[e,t,n]of[[`standardizedRates`,l.standardizedRates,`standardized`],[`smoothedRates`,l.smoothedRates,`smoothed`],[`rawRates`,l.rawRates,`rate`],[`excessRisk`,l.excessRisk,`select(nan, e / (b * pooledRate), b * pooledRate > 0.0)`]])t&&(P.push({name:e,view:t,type:`f32`,access:`read_write`}),F.push(`${e}[${e}Offset + index] = ${n};`),I.push(`${e}[${e}Offset + index] = nan;`));return P.length===0||N.push(r(t,{id:`${i}-rates`,operation:h,variant:`rates`,bindings:[{name:`events`,view:m,type:`f32`,access:`read`},{name:`populations`,view:_,type:`f32`,access:`read`},...A,{name:`summary`,view:w,type:`f32`,access:`read`},...P],invocationCount:y,declarations:k,body:`let e = events[eventsOffset + index];
  let b = populations[populationsOffset + index];
  let nan = getQuietNaN(index);
  if (!(${j})) {
    ${I.join(`
    `)}
    return;
  }
  let pooledRate = summary[summaryOffset + ${D.pooledRate}u];
  let priorVariance = summary[summaryOffset + ${D.priorVariance}u];
  let rate = e / b;
  let noise = pooledRate / b;
  let rawVariance = priorVariance + noise;
  let variance = select(rawVariance, noise, rawVariance < 0.0);
  let standardized = (rate - pooledRate) / sqrt(variance);
  ${O?`let clampedVariance = summary[summaryOffset + ${D.clampedPriorVariance}u];
  let clampedDenominator = clampedVariance + noise;
  let weight = select(0.0, clampedVariance / clampedDenominator, clampedDenominator > 0.0);`:`let weight = priorVariance / rawVariance;`}
  let smoothed = weight * rate + (1.0 - weight) * pooledRate;
  ${F.join(`
  `)}`})),N}};export{y as t};
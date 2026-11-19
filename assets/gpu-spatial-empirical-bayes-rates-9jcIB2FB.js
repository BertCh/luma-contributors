import{u as e}from"./kernel-C3tfQOn4.js";import{E as t,x as n,y as r}from"./gpu-contributor-utils-BIwLQON5.js";import{n as i,t as a}from"./float32-values-60_XMjYS.js";import{t as o}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as s}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as c}from"./spatial-statistics-row-dispatch-DNaskhxv.js";import{t as l}from"./spatial-autocorrelation-kernels-CCrGBZuG.js";import{r as u}from"./spatial-weights-CqWysstN.js";import{i as d,s as f,t as p}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";var m=`GPUSpatialEmpiricalBayesRates`,h=class{id;props;constructor(n){let a=n.id??`spatial-empirical-bayes-rates`;this.id=a,this.props=n;let o=i(n.events,`${a} events`);if(o<1)throw Error(`${a} events must hold at least one row`);if(i(n.populations,`${a} populations`)!==o)throw Error(`${a} populations length must equal the events length`);if(f(a,n.weights)!==o)throw Error(`${a} weights row count must equal the events length`);if(n.mask&&(e(n.mask,`${a} mask`),n.mask.length!==o))throw Error(`${a} mask length must equal the events length`);s(a,n.output);let c=this._getOutputs();for(let[e,n]of c)if(t(n,[`float32`],`${a} output.${e}`),n.length!==o)throw Error(`${a} output.${e} length must equal the events length`);if(c.length===0)throw Error(`${a} requests no output`);r(a,c.map(([,e])=>e),[n.events,n.populations,n.mask,...d(n.weights)])}getCommandNodes(e){let{id:t,props:r}=this;o(e,t,r,{"row conversions":r.events.length*4});let{weights:i,mask:s}=r,f=this._getOutputs();n(t,e,[r.events,r.populations,s,...d(i),...f.map(([,e])=>e)]);let h=a(e,`${t}-events`,r.events),g=a(e,`${t}-populations`,r.populations),_=h.view,v=g.view,y=_.length,b=s?[{name:`mask`,view:s,type:`u32`,access:`read`}]:[],x=e=>`(${s?`mask[maskOffset + ${e}] != 0u && `:``}isFiniteFloat(events[eventsOffset + ${e}]) && isFiniteFloat(populations[populationsOffset + ${e}]) && populations[populationsOffset + ${e}] > 0.0)`,S=f.map(([e,t])=>({name:e,view:t,type:`f32`,access:`read_write`})),C={spatialRates:`spatialRate`,smoothedRates:`smoothed`,priorMeans:`priorMean`,priorVariances:`priorVariance`},w=(e,t)=>e.map(({name:e})=>`${e}[${e}Offset + index] = ${t(e)};`).join(`
  `),T=3-b.length,E=[];for(let e=0;e<S.length;e+=T)E.push(S.slice(e,e+T));E.length===0&&E.push([]);let D=p(i).flatMap(n=>{let r=e=>u(x(`neighbor`),e,`index`,`WEIGHTS_ROW_START`);return E.map((i,a)=>c(e,{rowBase:n.rowStart,id:`${E.length>1?`${t}-smooth-${a}`:`${t}-smooth`}${n.idSuffix}`,operation:m,variant:`smooth`,bindings:[{name:`offsets`,view:n.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:n.weights.weights,type:`f32`,access:`read`},{name:`events`,view:_,type:`f32`,access:`read`},{name:`populations`,view:v,type:`f32`,access:`read`},...b,...i],invocationCount:n.rowCount,declarations:`const ROWS: u32 = ${y}u;
${l}`,body:`let nan = getQuietNaN(index);
  if (!${x(`index`)}) {
    ${w(i,()=>`nan`)}
    return;
  }
  var eventSum = events[eventsOffset + index];
  var populationSum = populations[populationsOffset + index];
  var count = 1u;
  ${r(`eventSum += events[eventsOffset + neighbor];
      populationSum += populations[populationsOffset + neighbor];
      count++;`)}
  let priorMean = eventSum / populationSum;
  let ownDeviation = events[eventsOffset + index] / populations[populationsOffset + index] - priorMean;
  var deviationSum = populations[populationsOffset + index] * ownDeviation * ownDeviation;
  ${r(`let b = populations[populationsOffset + neighbor];
      let deviation = events[eventsOffset + neighbor] / b - priorMean;
      deviationSum += b * deviation * deviation;`)}
  let spatialRate = priorMean;
  let priorVariance = max(deviationSum / populationSum - priorMean / (divideBySpatialCount(populationSum, count)), 0.0);
  let ownPopulation = populations[populationsOffset + index];
  let ownRate = events[eventsOffset + index] / ownPopulation;
  let denominator = priorVariance + priorMean / ownPopulation;
  let weight = select(0.0, priorVariance / denominator, denominator > 0.0);
  let smoothed = weight * ownRate + (1.0 - weight) * priorMean;
  ${w(i,e=>C[e])}`}))});return[...h.nodes,...g.nodes,...D]}_getOutputs(){let{output:e}=this.props;return[[`spatialRates`,e.spatialRates],[`smoothedRates`,e.smoothedRates],[`priorMeans`,e.priorMeans],[`priorVariances`,e.priorVariances]].filter(e=>e[1]!==void 0)}};export{h as t};
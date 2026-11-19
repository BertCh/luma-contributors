import{n as e}from"./local-operations-parameters-C9XqscTT.js";import{d as t,f as n,h as r,m as i,p as a,s as o,u as s}from"./raster-algebra-input-5ZxymrFJ.js";import{n as c}from"./raster-storage-preflight-CIxFc-0g.js";import{o as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{p as u,s as d}from"./gpu-contributor-utils-BIwLQON5.js";import{a as f,i as p,n as m,o as h,r as g,t as _}from"./raster-algebra-stack-input-DZ5O9caI.js";import{t as v}from"./define-run-B-brkCe4.js";import{a as y,l as b}from"./inputs-CJKKIRKK.js";import{sa as x}from"./entry-preview-D70TdSbK.js";import{c as S}from"./pointer-overlay-Qh8qflcA.js";import{t as C}from"./bands-CVDIl0aE.js";import{i as w,r as T}from"./gestures-C1AoqRjN.js";var E=`GPURasterCellStatistics`,D=64,O=3,k=class{id;estimatedCommandNodeCount;props;constructor(e){this.id=e.id??`raster-cell-statistics`,this.props=e;let{id:a}=this,{layerCount:o,cellCount:s,output:c}=e;if(n(a,`layerCount`,o,1,D),n(a,`cellCount`,s,0,4294967295),o*s>4294967295)throw Error(`${a} layerCount * cellCount must fit in uint32`);i(a,e.noDataValue),h(a,e.stack,e.grid,o,s),this.estimatedCommandNodeCount=O+m(e.stack),u(a,this.estimatedCommandNodeCount),r(a,`mask`,e.mask,`uint32`,s),r(a,`parameters`,e.parameters,`float32`,4);let l=A(c);if(l.length===0)throw Error(`${a} needs at least one output`);for(let[e,t]of l)r(a,`output.${e}`,t,e===`counts`||e===`variety`?`uint32`:`float32`,s);t(a,Object.fromEntries(l.map(([e,t])=>[`output.${e}`,t])),{..._(e.stack),mask:e.mask,parameters:e.parameters})}getCommandNodes(e){let t=this.props;o(e,this.id,f(t.stack)?t.stack:[t.stack],t.grid),c(e,this.id,[t.mask,t.parameters,...d(t.output)],{"packed band stack":f(t.stack)?t.layerCount*t.cellCount*4:0});let{id:n,props:r}=this,{output:i,layerCount:u,cellCount:m}=r;a(n,e,[...p(r.stack),r.mask,r.parameters,...A(i).map(([,e])=>e)]);let h=g(e,n,r.stack,r.grid,m),_=[{name:`stack`,view:h.values,type:`f32`,access:`read`},...r.mask?[{name:`mask`,view:r.mask,type:`u32`,access:`read`}]:[],{name:`params`,view:r.parameters,type:`f32`,access:`read`}],v=(e,t,n=`f32`)=>t?[{name:e,view:t,type:n,access:`read_write`}]:[],y=`const LAYER_COUNT: u32 = ${u}u;
const CELL_COUNT: u32 = ${m}u;
${s(f(r.stack)?void 0:r.noDataValue)}
fn readLayer(layer: u32, cell: u32, value: ptr<function, f32>) -> bool {
  let sample = stack[stackOffset + layer * CELL_COUNT + cell];
  *value = sample;
  return !isNoDataValue(sample);
}`,b=`let propagateNoData = params[paramsOffset] != 0.0;
  let minimumCountValue = params[paramsOffset + 1u];
  let minimumCount = max(u32(clamp(select(1.0, minimumCountValue, isFiniteValue(minimumCountValue)), 0.0, 4294967040.0)), 1u);
  var count = 0u;
  let isKept = ${r.mask?`mask[maskOffset + index] != 0u`:`true`};`,x=`let isDefined = isKept && count >= minimumCount && (!propagateNoData || count == LAYER_COUNT);
  let nan = getNaN();`,S=e=>`for (var layer = 0u; layer < LAYER_COUNT; layer++) {
    var value = 0.0;
    if (!readLayer(layer, index, &value)) {
      continue;
    }
    ${e}
  }`,C=[...h.nodes];return(i.sum||i.mean||i.standardDeviation||i.counts)&&C.push(l(e,{id:`${n}-moments`,operation:E,variant:`moments`,bindings:[..._,...v(`sumOut`,i.sum),...v(`meanOut`,i.mean),...v(`deviationOut`,i.standardDeviation),...v(`countOut`,i.counts,`u32`)],invocationCount:m,declarations:y,body:`${b}
  var sum = 0.0;
  ${S(`sum = sum + value;
    count += 1u;`)}
  ${x}
  let mean = sum / f32(max(count, 1u));
  ${i.sum?`sumOut[sumOutOffset + index] = select(nan, sum, isDefined);`:``}
  ${i.mean?`meanOut[meanOutOffset + index] = select(nan, mean, isDefined);`:``}
  ${i.counts?`countOut[countOutOffset + index] = select(0u, count, isKept);`:``}
  ${i.standardDeviation?`var squaredDeviationSum = 0.0;
  ${S(`let deviation = value - mean;
    squaredDeviationSum = squaredDeviationSum + deviation * deviation;`)}
  deviationOut[deviationOutOffset + index] = select(nan, sqrt(squaredDeviationSum / f32(max(count, 1u))), isDefined);`:``}`})),(i.minimum||i.maximum||i.range)&&C.push(l(e,{id:`${n}-extremes`,operation:E,variant:`extremes`,bindings:[..._,...v(`minimumOut`,i.minimum),...v(`maximumOut`,i.maximum),...v(`rangeOut`,i.range)],invocationCount:m,declarations:y,body:`${b}
  var minimum = 0.0;
  var maximum = 0.0;
  ${S(`if (count == 0u) {
      minimum = value;
      maximum = value;
    } else {
      minimum = min(minimum, value);
      maximum = max(maximum, value);
    }
    count += 1u;`)}
  ${x}
  ${i.minimum?`minimumOut[minimumOutOffset + index] = select(nan, minimum, isDefined);`:``}
  ${i.maximum?`maximumOut[maximumOutOffset + index] = select(nan, maximum, isDefined);`:``}
  ${i.range?`rangeOut[rangeOutOffset + index] = select(nan, maximum - minimum, isDefined);`:``}`})),(i.majority||i.minority||i.variety)&&C.push(l(e,{id:`${n}-frequencies`,operation:E,variant:`frequencies`,bindings:[..._,...v(`majorityOut`,i.majority),...v(`minorityOut`,i.minority),...v(`varietyOut`,i.variety,`u32`)],invocationCount:m,workgroupSize:64,declarations:y,body:`${b}
  var gathered: array<f32, ${u}>;
  ${S(`gathered[count] = value;
    count += 1u;`)}
  ${x}
  var majority = 0.0;
  var majorityCount = 0u;
  var minority = 0.0;
  var minorityCount = 0xffffffffu;
  var variety = 0u;
  for (var position = 0u; position < count; position++) {
    let value = gathered[position];
    var frequency = 0u;
    var isFirst = true;
    for (var other = 0u; other < count; other++) {
      if (gathered[other] == value) {
        frequency += 1u;
        if (other < position) {
          isFirst = false;
        }
      }
    }
    if (!isFirst) {
      continue;
    }
    variety += 1u;
    if (frequency > majorityCount || (frequency == majorityCount && value < majority)) {
      majority = value;
      majorityCount = frequency;
    }
    if (frequency < minorityCount || (frequency == minorityCount && value < minority)) {
      minority = value;
      minorityCount = frequency;
    }
  }
  ${i.majority?`majorityOut[majorityOutOffset + index] = select(nan, majority, isDefined);`:``}
  ${i.minority?`minorityOut[minorityOutOffset + index] = select(nan, minority, isDefined);`:``}
  ${i.variety?`varietyOut[varietyOutOffset + index] = select(0u, variety, isDefined);`:``}`})),C}};function A(e){return[`minimum`,`maximum`,`range`,`sum`,`mean`,`standardDeviation`,`majority`,`minority`,`variety`,`counts`].flatMap(t=>{let n=e[t];return n?[[t,n]]:[]})}var j=[`mean`,`maximum`,`range`,`standardDeviation`,`variety`,`counts`],M=5,N={mean:[0,.25],maximum:[0,.5],range:[0,.3],standardDeviation:[0,.15],variety:[0,5],counts:[0,5]},P=v({parameters:x,caption:`A stack of five layers in; per-cell statistics of the valid layers out.`,createInputs:({random:e,world:t})=>{let n=y(t),r=[n.green,n.red,n.nearInfrared,n.shortwaveInfrared,n.shortwaveInfrared2],i=r.map(()=>{let t=b(e,3,4),r=new Float32Array(n.values.length);for(let e=0;e<r.length;e++){let i=Math.floor(e/n.columns),a=e%n.columns;r[e]=t((a+.5)/n.columns,(i+.5)/n.rows)}return r});return{...n,layers:r,clouds:i}},build(t,n){let r=n.values.length,i=t.allocate(`stack`,`float32`,r*M),a=t.allocate(`parameters`,`float32`,4),o={mean:t.allocate(`mean`,`float32`,r),maximum:t.allocate(`maximum`,`float32`,r),range:t.allocate(`range`,`float32`,r),standardDeviation:t.allocate(`standard-deviation`,`float32`,r),variety:t.allocate(`variety`,`uint32`,r),counts:t.allocate(`counts`,`uint32`,r)};t.graph.add(new k({stack:i.view,layerCount:M,cellCount:r,parameters:a.view,output:{mean:o.mean.view,maximum:o.maximum.view,range:o.range.view,standardDeviation:o.standardDeviation.view,variety:o.variety.view,counts:o.counts.view}}));let s=t.upload(`first-layer`,`float32`,n.layers[0]),c=t.allocate(`inset`,`float32`,M);return{outputs:{...o,first:s,inset:c},update:t=>{let o=new Float32Array(r*M),s=1-t.cloudCover;for(let e=0;e<M;e++)for(let t=0;t<r;t++)o[e*r+t]=n.clouds[e][t]>s?NaN:n.layers[e][t];i.write(o),a.write(e({minimumValidCount:t.minimumValidCount}))}}},interaction:{kind:`probe`,hint:`probe a cell`,rest:[.5,.5],resolve:(e,{inputs:t,outputs:n,values:r,world:i})=>{let{column:a,row:o,index:s}=T(t,i,e),c=1-r.cloudCover,l=t.layers.map((e,n)=>t.clouds[n][s]>c?NaN:e[s]),u=l.filter(e=>Number.isFinite(e));n.inset.write(Float32Array.from(l,e=>Number.isFinite(e)?e:0));let d=u.reduce((e,t)=>e+t,0)/(u.length||1),f=u.length>=r.minimumValidCount;return{glyphs:[w(t,i,a,o)],readout:`${u.length} of 5 valid | ${f?`mean ${S(d)}`:`nodata`}`}}},draw:({inputs:e,outputs:t,values:n})=>{let r=j[Math.round(n.statistic)];return[C(e,t.first,{context:!0}),C(e,t[r],{ramp:`sequential`,range:N[r],smoothing:r!==`variety`&&r!==`counts`}),{layer:`chart`,kind:`bars`,values:t.inset,range:[0,.6],rectangle:[.03,.04,.3,.3]}]}});export{P as default};
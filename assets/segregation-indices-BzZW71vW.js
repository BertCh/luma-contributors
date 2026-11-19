import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./statistics-kernel-node-lacKNERm.js";import{E as n,T as r,f as i,p as a,x as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{c,r as l}from"./spatial-weights-CqWysstN.js";import{r as u}from"./wgsl-common-DfyXiYB0.js";import{t as d}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as f,s as p}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as m}from"./define-run-B-brkCe4.js";import{m as h,u as g}from"./inputs-CJKKIRKK.js";import{a as _,i as v,r as y,t as b}from"./tract-rates-BlUiSczm.js";import{Tt as x}from"./entry-preview-D70TdSbK.js";function S(e){return{entropy:0,multiGroupDissimilarity:1,diversity:2,dissimilarity:3,isolation:3+e,atkinson:3+2*e,interaction:3+3*e,stride:3+3*e+e*e}}function C(e={},t=`segregation-parameters`){let n=Math.fround(e.selfWeight??1),i=Math.fround(e.atkinsonInequalityAversion??.5);return r(t,Number.isFinite(n)&&n>=0,`selfWeight must round to a finite nonnegative float32`),r(t,i>0&&i<1,`atkinsonInequalityAversion must round to a float32 in (0, 1)`),Float32Array.of(n,i)}var w=(e,t)=>`
const ENVIRONMENT_TOTALS: u32 = ${e*t}u;
const ENVIRONMENT_SCALE: u32 = ${e*t+e}u;
fn getEnvironmentTotal(unit: u32) -> f32 {
  return environment[environmentOffset + ENVIRONMENT_TOTALS + unit];
}
fn getScaleIndex() -> u32 {
  return u32(environment[environmentOffset + ENVIRONMENT_SCALE]);
}`,T=16,E=12,D=`GPUSegregation`,O=256,k=8,A=class{id;props;scaleCount;unitCount;parameterValues;estimatedCommandNodeCount;constructor(e){let t=e.id??`segregation`;this.id=t,this.props=e,this.parameterValues=C(e,t),e.parameters&&(n(e.parameters,[`float32`],`${t} parameters`),i(t,`parameters length`,e.parameters.length,2,2));let{groupCount:r}=e;i(t,`groupCount`,r,2,16);let o=e.scales??[null];if(o.length<1)throw Error(`${t} scales must not be empty`);this.scaleCount=o.length,this.estimatedCommandNodeCount=T+E*o.length,a(t,this.estimatedCommandNodeCount),n(e.groupCounts,[`float32`],`${t} groupCounts`);let l=e.groupCounts.length/r;if(!Number.isInteger(l)||l<1)throw Error(`${t} groupCounts length must be a positive multiple of groupCount`);this.unitCount=l,o.forEach((e,n)=>{if(e&&c(t,e,`scales[${n}]`)!==l)throw Error(`${t} scales[${n}] must have one row per unit`)});let u=e.atkinsonInequalityAversion??.5;if(!(u>0&&u<1))throw Error(`${t} atkinsonInequalityAversion must be in (0, 1)`);let d=e.selfWeight??1;if(!Number.isFinite(d)||d<0)throw Error(`${t} selfWeight must be finite and non-negative`);let p=S(r),{output:m}=e,h=[[`output.indices`,m.indices,this.scaleCount*p.stride],[`output.localEnvironment`,m.localEnvironment,this.scaleCount*l*r],[`output.localEntropy`,m.localEntropy,this.scaleCount*l],[`output.localDissimilarity`,m.localDissimilarity,this.scaleCount*l*r],[`output.localTheil`,m.localTheil,this.scaleCount*l]];for(let[e,r,i]of h)if(r&&(n(r,[`float32`],`${t} ${e}`),r.length!==i))throw Error(`${t} ${e} must contain ${i} rows`);s(t,this.getOutputViews(),this.getInputViews()),f(t,e.output)}getInputViews(){let e=[this.props.groupCounts,this.props.parameters];for(let t of this.props.scales??[])t&&e.push(t.offsets,t.neighbors,t.weights);return e}getOutputViews(){let{output:e}=this.props;return[e.indices,e.localEnvironment,e.localEntropy,e.localDissimilarity,e.localTheil]}getCommandNodes(n){let{id:r,props:i,unitCount:a}=this,s=2+2*i.groupCount+i.groupCount*i.groupCount;d(n,r,i,{"environment table":(a*(i.groupCount+1)+1)*4,"term partials":s*Math.ceil(a/O)*4,"scale sums":this.scaleCount*s*4,"scale totals":this.scaleCount*(i.groupCount+1)*4}),o(r,n,[...this.getInputViews(),...this.getOutputViews()]);let{groupCount:c,groupCounts:f,output:m}=i,{indices:h}=m,g={environment:m.localEnvironment,entropy:m.localEntropy,dissimilarity:m.localDissimilarity,theil:m.localTheil},_=i.scales??[null],v=S(c),y=2+2*c+c*c,b=p(n,`${r}-parameters`,i.parameters??this.parameterValues),x={name:`parameters`,view:b.parameters,type:`f32`,access:`read`},C=`SELF_WEIGHT = parameters[parametersOffset]; ATKINSON_B = parameters[parametersOffset + 1u];`,T=[...b.nodes],E=`${u}
const UNITS: u32 = ${a}u;
const K: u32 = ${c}u;
const TERM_COUNT: u32 = ${y}u;
var<private> ATKINSON_B: f32;
var<private> SELF_WEIGHT: f32;
fn countAt(unit: u32, group: u32) -> f32 {
  let value = groupCounts[groupCountsOffset + unit * K + group];
  // Negative, NaN and infinite counts are 0.
  let isFinite = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
  return select(0.0, value, isFinite && value > 0.0);
}
fn unitTotal(unit: u32) -> f32 {
  var total = 0.0;
  for (var group = 0u; group < K; group++) {
    total += countAt(unit, group);
  }
  return total;
}
fn entropyTerm(share: f32) -> f32 {
  return select(0.0, -share * log(share), share > 0.0);
}`,A=Math.ceil(a/O),j=(e,t)=>`${t}
const LANES: u32 = ${O}u;
const GROUPS: u32 = ${A}u;
const COLUMNS: u32 = ${e}u;
const COLUMN_BATCH: u32 = ${k}u;
var<private> columnValues: array<f32, ${e}>;
var<workgroup> batchScratch: array<f32, ${k*O}>;
fn reduceColumns(lane: u32, group: u32) {
  for (var first = 0u; first < COLUMNS; first += COLUMN_BATCH) {
    workgroupBarrier();
    for (var slot = 0u; slot < COLUMN_BATCH; slot++) {
      let column = first + slot;
      batchScratch[slot * LANES + lane] = select(0.0, columnValues[min(column, COLUMNS - 1u)], column < COLUMNS);
    }
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        for (var slot = 0u; slot < COLUMN_BATCH; slot++) {
          batchScratch[slot * LANES + lane] += batchScratch[slot * LANES + lane + stride];
        }
      }
      workgroupBarrier();
    }
    if (lane < COLUMN_BATCH && first + lane < COLUMNS) {
      partials[partialsOffset + (first + lane) * GROUPS + group] = batchScratch[lane * LANES];
    }
  }
}`,M=(e,i,o,s,l)=>t(n,{id:`${r}-${e}`,operation:D,variant:`column-totals`,bindings:[{name:`partials`,view:i,type:`f32`,access:`read`},{name:`output`,view:s,type:`f32`,access:`read_write`},...l?[{name:`collected`,view:l.view,type:`f32`,access:`read_write`},{name:`environment`,view:l.environment,type:`f32`,access:`read`}]:[]],invocationCount:o*O,workgroupSize:O,guardIndex:!1,declarations:`const LANES: u32 = ${O}u;
const GROUPS: u32 = ${A}u;
${l?`const COLLECTED_STRIDE: u32 = ${l.stride}u;
${w(a,c)}`:``}
var<workgroup> scratch: array<f32, ${O}>;`,body:`let column = index / LANES;
  let lane = localInvocationIndex;
  var sum = 0.0;
  for (var group = lane; group < GROUPS; group += LANES) {
    sum += partials[partialsOffset + column * GROUPS + group];
  }
  scratch[lane] = sum;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    output[outputOffset + column] = scratch[0];
    ${l?`collected[collectedOffset + getScaleIndex() * COLLECTED_STRIDE + column] = scratch[0];`:``}
  }`}),N=e(n,`${r}-all-totals`,`float32`,_.length*(c+1)),P=i.spatialForm===`smoothed-population`,F=(i,a,o)=>{let s=e(n,`${r}-${i}-partials`,`float32`,(c+1)*A),l=e(n,`${r}-${i}`,`float32`,c+1);return T.push(t(n,{id:`${r}-${i}-partials`,operation:D,variant:`total-partials`,bindings:[{name:`groupCounts`,view:a,type:`f32`,access:`read`},{name:`partials`,view:s,type:`f32`,access:`read_write`}],invocationCount:A*O,workgroupSize:O,guardIndex:!1,declarations:j(c+1,E),body:`let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  if (index < UNITS) {
    for (var column = 0u; column < K; column++) {
      columnValues[column] = countAt(index, column);
    }
    columnValues[K] = unitTotal(index);
  }
  reduceColumns(lane, group);`}),M(i,s,c+1,l,o?{view:N,stride:c+1,environment:o}:void 0)),l},I=(e,r,i)=>t(n,{id:`${e}-collect-totals`,operation:D,variant:`collect-totals`,bindings:[{name:`totals`,view:r,type:`f32`,access:`read`},{name:`allTotals`,view:N,type:`f32`,access:`read_write`},{name:`environment`,view:i,type:`f32`,access:`read`}],invocationCount:c+1,declarations:`const K: u32 = ${c}u;
${w(a,c)}`,body:`allTotals[allTotalsOffset + getScaleIndex() * (K + 1u) + index] = totals[totalsOffset + index];`}),L=P?void 0:F(`totals`,f),R=`${E}
${w(a,c)}
fn getShare(unit: u32, group: u32) -> f32 {
  let total = getEnvironmentTotal(unit);
  return select(0.0, environment[environmentOffset + unit * K + group] / total, total > 0.0);
}
fn getPopulationShare(group: u32) -> f32 {
  let total = totals[totalsOffset + K];
  return select(0.0, totals[totalsOffset + group] / total, total > 0.0);
}
fn getDiversity() -> f32 {
  var diversity = 0.0;
  for (var group = 0u; group < K; group++) {
    diversity += entropyTerm(getPopulationShare(group));
  }
  return diversity;
}`,z=e(n,`${r}-term-sums`,`float32`,_.length*y),B=e(n,`${r}-scale-cursor`,`uint32`,1);return T.push(t(n,{id:`${r}-scale-cursor-clear`,operation:D,variant:`scale-cursor-clear`,bindings:[{name:`scaleCursor`,view:B,type:`u32`,access:`read_write`}],invocationCount:1,body:`scaleCursor[scaleCursorOffset] = 0xffffffffu;`})),_.forEach((i,o)=>{let s=`${r}-scale-${o}`,u=e(n,`${s}-environment`,`float32`,a*c+a+1),d=e(n,`${s}-sums`,`float32`,y),p=e(n,`${s}-term-partials`,`float32`,y*A),m=P?u:f;T.push(t(n,{id:`${s}-environment`,operation:D,variant:i?`environment`:`environment-aspatial`,bindings:[x,{name:`groupCounts`,view:f,type:`f32`,access:`read`},...i?[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`weights`,view:i.weights,type:`f32`,access:`read`}]:[],{name:`environment`,view:u,type:`f32`,access:`read_write`},{name:`scaleCursor`,view:B,type:`u32`,access:`read_write`}],invocationCount:a,declarations:i?`${E}
const ROWS: u32 = ${a}u;
const CAPACITY: u32 = 0xffffffffu;`:E,body:`${C}
  if (index == 0u) {
    let scale = scaleCursor[scaleCursorOffset] + 1u;
    scaleCursor[scaleCursorOffset] = scale;
    environment[environmentOffset + UNITS * K + UNITS] = f32(scale);
  }
  ${i?`var total = 0.0;
  for (var group = 0u; group < K; group++) {
    var sum = SELF_WEIGHT * countAt(index, group);
    ${l(`true`,`sum += weight * countAt(neighbor, group);`)}
    environment[environmentOffset + index * K + group] = sum;
    total += sum;
  }
  environment[environmentOffset + UNITS * K + index] = total;`:`for (var group = 0u; group < K; group++) {
    environment[environmentOffset + index * K + group] = countAt(index, group);
  }
  environment[environmentOffset + UNITS * K + index] = unitTotal(index);`}`}));let h=L??F(`scale-${o}-totals`,m,u);if(T.push(...L?[I(s,h,u)]:[],t(n,{id:`${s}-terms`,operation:D,variant:`terms`,bindings:[x,{name:`groupCounts`,view:m,type:`f32`,access:`read`},{name:`environment`,view:u,type:`f32`,access:`read`},{name:`totals`,view:h,type:`f32`,access:`read`},{name:`partials`,view:p,type:`f32`,access:`read_write`}],invocationCount:A*O,workgroupSize:O,guardIndex:!1,declarations:j(y,R),body:`${C}
  let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  // A unit whose environment is empty has no composition: it contributes nothing to any sum.
  if (index < UNITS && getEnvironmentTotal(index) > 0.0) {
    let unitPopulation = unitTotal(index);
    var entropy = 0.0;
    var absolute = 0.0;
    for (var member = 0u; member < K; member++) {
      let share = getShare(index, member);
      let deviation = abs(share - getPopulationShare(member));
      entropy += entropyTerm(share);
      absolute += deviation;
      columnValues[2u + member] = unitPopulation * deviation;
      let base = max(1.0 - share, 0.0);
      let atkinson = select(0.0, pow(base, 1.0 - ATKINSON_B) * pow(share, ATKINSON_B), base > 0.0 && share > 0.0);
      columnValues[2u + K + member] = unitPopulation * atkinson;
      for (var other = 0u; other < K; other++) {
        columnValues[2u + 2u * K + member * K + other] = countAt(index, member) * getShare(index, other);
      }
    }
    columnValues[0] = unitPopulation * entropy;
    columnValues[1] = unitPopulation * absolute;
  }
  reduceColumns(lane, group);`}),M(`${s}-sums`,p,y,d,{view:z,stride:y,environment:u})),g.environment||g.entropy||g.dissimilarity||g.theil){let e=[[`localEnvironment`,g.environment],[`localEntropy`,g.entropy],[`localDissimilarity`,g.dissimilarity],[`localTheil`,g.theil]];T.push(t(n,{id:`${s}-local`,operation:D,variant:`local-${e.map(([,e])=>+!!e).join(``)}`,bindings:[{name:`groupCounts`,view:m,type:`f32`,access:`read`},{name:`environment`,view:u,type:`f32`,access:`read`},{name:`totals`,view:h,type:`f32`,access:`read`},...e.flatMap(([e,t])=>t?[{name:e,view:t,type:`f32`,access:`read_write`}]:[])],invocationCount:a,declarations:R,body:`let unitPopulation = unitTotal(index);
  let scaleIndex = getScaleIndex();
  // An empty environment has no composition, so every local value of the unit is NaN.
  let hasEnvironment = getEnvironmentTotal(index) > 0.0;
  let populationTotal = totals[totalsOffset + K];
  let diversity = getDiversity();
  var entropy = 0.0;
  for (var group = 0u; group < K; group++) {
    let share = getShare(index, group);
    entropy += entropyTerm(share);
    ${g.environment?`localEnvironment[localEnvironmentOffset + (scaleIndex * UNITS + index) * K + group] =
      select(getNaN(), share, hasEnvironment);`:``}
    ${g.dissimilarity?`let populationShare = getPopulationShare(group);
    let scale = 2.0 * populationTotal * populationShare * (1.0 - populationShare);
    localDissimilarity[localDissimilarityOffset + (scaleIndex * UNITS + index) * K + group] =
      select(getNaN(), unitPopulation * abs(share - populationShare) / scale, scale > 0.0 && hasEnvironment);`:``}
  }
  ${g.entropy?`localEntropy[localEntropyOffset + scaleIndex * UNITS + index] =
    select(getNaN(), entropy, hasEnvironment);`:``}
  ${g.theil?`localTheil[localTheilOffset + scaleIndex * UNITS + index] =
    select(getNaN(), unitPopulation * (diversity - entropy) / (diversity * populationTotal), diversity > 0.0 && populationTotal > 0.0 && hasEnvironment);`:``}`}))}}),T.push(t(n,{id:`${r}-finalize`,operation:D,variant:`finalize`,bindings:[x,{name:`allTotals`,view:N,type:`f32`,access:`read`},{name:`allSums`,view:z,type:`f32`,access:`read`},{name:`indices`,view:h,type:`f32`,access:`read_write`}],invocationCount:_.length,declarations:`${u}
const K: u32 = ${c}u;
const TERM_COUNT: u32 = ${y}u;
const STRIDE: u32 = ${v.stride}u;
var<private> ATKINSON_B: f32;
fn entropyTerm(share: f32) -> f32 {
  return select(0.0, -share * log(share), share > 0.0);
}`,body:`ATKINSON_B = parameters[parametersOffset + 1u];
  let populationTotal = allTotals[allTotalsOffset + index * (K + 1u) + K];
  let sums = index * TERM_COUNT;
  let base = index * STRIDE;
  var diversity = 0.0;
  var variance = 0.0;
  for (var group = 0u; group < K; group++) {
    let share = select(0.0, allTotals[allTotalsOffset + index * (K + 1u) + group] / populationTotal, populationTotal > 0.0);
    diversity += entropyTerm(share);
    variance += share * (1.0 - share);
  }
  let hasPopulation = populationTotal > 0.0;
  indices[indicesOffset + base + ${v.entropy}u] = select(
    getNaN(), 1.0 - allSums[allSumsOffset + sums] / (diversity * populationTotal), hasPopulation && diversity > 0.0);
  indices[indicesOffset + base + ${v.multiGroupDissimilarity}u] = select(
    getNaN(), allSums[allSumsOffset + sums + 1u] / (2.0 * populationTotal * variance), hasPopulation && variance > 0.0);
  indices[indicesOffset + base + ${v.diversity}u] = select(getNaN(), diversity, hasPopulation);
  for (var group = 0u; group < K; group++) {
    let groupTotal = allTotals[allTotalsOffset + index * (K + 1u) + group];
    let share = select(0.0, groupTotal / populationTotal, hasPopulation);
    let isMixed = hasPopulation && share > 0.0 && share < 1.0;
    let spread = 2.0 * populationTotal * share * (1.0 - share);
    indices[indicesOffset + base + ${v.dissimilarity}u + group] = select(
      getNaN(), allSums[allSumsOffset + sums + 2u + group] / spread, isMixed);
    let ratio = abs(allSums[allSumsOffset + sums + 2u + K + group] / max(share * populationTotal, 1e-30));
    let atkinson = 1.0 - share / (1.0 - share) * pow(ratio, 1.0 / (1.0 - ATKINSON_B));
    indices[indicesOffset + base + ${v.atkinson}u + group] = select(getNaN(), atkinson, isMixed);
    for (var other = 0u; other < K; other++) {
      let exposure = allSums[allSumsOffset + sums + 2u + 2u * K + group * K + other];
      let value = select(getNaN(), exposure / groupTotal, groupTotal > 0.0);
      indices[indicesOffset + base + ${v.interaction}u + group * K + other] = value;
      if (other == group) {
        indices[indicesOffset + base + ${v.isolation}u + group] = value;
      }
    }
  }`})),T}},j=m({parameters:x,caption:`Modeled group mixtures across Chicago tracts, with local entropy computed from the group counts.`,createInputs:({random:e,world:t})=>{let n=g(e,t);return{tracts:n,populations:n.population,focus:{row:-1}}},build(e,{tracts:t,populations:n},r){let i=r.groupCount,a=r.scale===1,o=e.allocate(`group-counts`,`float32`,t.count*i),s=e.allocate(`segregation-indices`,`float32`,3+3*i+i*i),c=e.allocate(`environment`,`float32`,1*t.count*i),l=e.allocate(`local-entropy`,`float32`,1*t.count);return e.graph.add(new A({groupCount:i,groupCounts:o.view,scales:[a?h(e,t.contiguity):null],output:{indices:s.view,localEnvironment:c.view,localEntropy:l.view}})),{outputs:{localEntropy:l,environment:c,globalIndices:s,groupCount:i,shares:e.allocate(`shares`,`float32`,i),..._(e,t)},update:e=>{let r=new Float32Array(t.count*i);for(let a=0;a<t.count;a++){let o=Array.from({length:i},(n,r)=>{let o=t.values[a]-(r+.5)/i;return Math.exp(-e.separation*14*i*o*o)}),s=o.reduce((e,t)=>e+t,0);o.forEach((e,t)=>{r[a*i+t]=Math.round(n[a]*e/s)})}o.write(r)}}},rebuildOn:[`groupCount`,`scale`],interaction:{kind:`probe`,hint:`hover a tract`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,peek:r}){let i=v(t.tracts,e,t.focus.row);t.focus.row=i;let a=r(n.environment),o=r(n.localEntropy)?.[i],s=r(n.globalIndices);return i<0||!a?{}:(n.shares.write(Float32Array.from({length:n.groupCount},(e,t)=>a[i*n.groupCount+t])),{glyphs:[b(t.tracts,i),...Array.from(y(t.tracts,i),e=>b(t.tracts,e))],readout:`entropy ${o===void 0?`…`:o.toFixed(2)} of ${Math.log(n.groupCount).toFixed(2)} · H ${s?s[0].toFixed(2):`…`} · D ${s?s[1].toFixed(2):`…`}`})}},draw:({inputs:e,outputs:t,values:n,theme:r})=>[{layer:`polygons`,positions:t.vertices,indices:t.meshIndices,owners:t.owners,values:t.localEntropy,triangleCount:e.tracts.triangleCount,range:[0,Math.log(n.groupCount)]},{layer:`lines`,context:!0,segments:t.outline,count:e.tracts.outlineCount},{layer:`chart`,kind:`bars`,values:t.shares,range:[0,1],rectangle:[.03,.04,.2,.22],color:[...r.categories[0],.95]}]});export{j as default};
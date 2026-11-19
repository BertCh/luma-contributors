import{i as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{r as n,u as r}from"./kernel-C3tfQOn4.js";import{t as i}from"./statistics-kernel-node-lacKNERm.js";import{E as a,f as o,x as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{n as l,t as u}from"./float32-values-60_XMjYS.js";import{r as d,u as f}from"./wgsl-common-DfyXiYB0.js";import{t as p}from"./define-run-B-brkCe4.js";import{u as m}from"./inputs-CJKKIRKK.js";import{a as h,i as g,n as _,t as v}from"./tract-rates-BlUiSczm.js";import{Wt as y}from"./entry-preview-D70TdSbK.js";var b=`GPUAssuncaoReisStandardization`,x=4096,S=256,C=2**24,w={count:0,mean:1,variance:2,priorVariance:3,population:4};function T(e,t){let{id:r,variant:a,rows:o,accumulators:s}=t,c=Math.ceil(o/x),l=e=>e===`f32`?`0.0`:`0u`,u=s.map(t=>({...t,view:n(e,`${r}-${a}-${t.name}-partials`,t.type===`f32`?`float32`:`uint32`,c)})),f=s.map(e=>`var<workgroup> shared_${e.name}: array<${e.type}, ${S}>;`).join(`
`),p=`workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      ${s.map(e=>`shared_${e.name}[localInvocationIndex] += shared_${e.name}[localInvocationIndex + stride];`).join(`
      `)}
    }
    workgroupBarrier();
  }`,m=e=>u.map(t=>({name:`${t.name}Partials`,view:t.view,type:t.type,access:e}));return[i(e,{id:`${r}-${a}-blocks`,operation:b,variant:`${a}-blocks`,bindings:[...t.rowBindings,...m(`read_write`)],workgroupSize:S,invocationCount:c*S,guardIndex:!1,declarations:`${t.rowDeclarations}
const ROW_COUNT: u32 = ${o}u;
const BLOCK_ROWS: u32 = ${x}u;
const WORKGROUP_SIZE: u32 = ${S}u;
${f}`,body:`let block = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  ${s.map(e=>`var local_${e.name} = ${l(e.type)};`).join(`
  `)}
  if (isInRange) {
    let end = min((block + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      let isValid = isRowValid(row);
      let isInvalid = isRowIncluded(row) && !isValid;
      ${s.map(e=>`if (${e.rows===`valid`?`isValid`:`isInvalid`}) { local_${e.name} += ${e.term}; }`).join(`
      `)}
    }
  }
  ${s.map(e=>`shared_${e.name}[localInvocationIndex] = local_${e.name};`).join(`
  `)}
  ${p}
  if (isInRange && localInvocationIndex == 0u) {
    ${s.map(e=>`${e.name}Partials[${e.name}PartialsOffset + block] = shared_${e.name}[0];`).join(`
    `)}
  }`}),i(e,{id:`${r}-${a}-reduce`,operation:b,variant:`${a}-reduce`,bindings:[...m(`read`),...t.finalizeBindings],workgroupSize:S,invocationCount:S,guardIndex:!1,declarations:`${d}
const BLOCK_COUNT: u32 = ${c}u;
const WORKGROUP_SIZE: u32 = ${S}u;
${f}`,body:`${s.map(e=>`var local_${e.name} = ${l(e.type)};`).join(`
  `)}
  for (var block = localInvocationIndex; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    ${s.map(e=>`local_${e.name} += ${e.name}Partials[${e.name}PartialsOffset + block];`).join(`
    `)}
  }
  ${s.map(e=>`shared_${e.name}[localInvocationIndex] = local_${e.name};`).join(`
  `)}
  ${p}
  if (localInvocationIndex == 0u) {
    ${s.map(e=>`let total_${e.name} = shared_${e.name}[0];`).join(`
    `)}
    ${t.finalize}
  }`})]}var E=class{id;props;constructor(t){let n=t.id??`assuncao-reis-standardization`;this.id=n,this.props=t;let{output:i}=t,s=l(t.events,`${n} events`);l(t.populations,`${n} populations`),a(i.standardizedRates,[`float32`],`${n} output.standardizedRates`),o(n,`events length`,s,0,C-1);for(let[e,r]of[[`populations`,t.populations],[`output.standardizedRates`,i.standardizedRates]])if(r.length!==s)throw Error(`${n} ${e} length must equal the events length`);if(t.mask&&(r(t.mask,`${n} mask`),t.mask.length!==s))throw Error(`${n} mask length must equal the events length`);i.statistics&&(a(i.statistics,[`float32`],`${n} output.statistics`),o(n,`output.statistics length`,i.statistics.length,5));let u=i.status?.invalidCount;u&&(r(u,`${n} output.status.invalidCount`),o(n,`output.status.invalidCount length`,u.length)),e(n,{standardizedRates:i.standardizedRates,statistics:i.statistics,invalidCount:u}),c(n,[i.standardizedRates,i.statistics,u],[t.events,t.populations,t.mask])}getCommandNodes(e){t(e,this.id,this.props,{"row scratch":this.props.events.length*4,"block sums":Math.ceil(this.props.events.length/x)*4});let{id:r,props:a}=this,{mask:o}=a,{standardizedRates:c}=a.output,l=a.output.status?.invalidCount;s(r,e,[a.events,a.populations,o,c,a.output.statistics,l]);let p=u(e,`${r}-events`,a.events),m=u(e,`${r}-populations`,a.populations),h=p.view,g=m.view,_=h.length,v=a.output.statistics??n(e,`${r}-statistics`,`float32`,5),y=[{name:`events`,view:h,type:`f32`,access:`read`},{name:`populations`,view:g,type:`f32`,access:`read`},...o?[{name:`mask`,view:o,type:`u32`,access:`read`}]:[]],S=`${f}
fn getSquare(value: f32) -> f32 { return value * value; }
fn isRowIncluded(row: u32) -> bool {
  return ${o?`mask[maskOffset + row] != 0u`:`true`};
}
fn isRowValid(row: u32) -> bool {
  let event = events[eventsOffset + row];
  let population = populations[populationsOffset + row];
  return isRowIncluded(row) && isFiniteValue(event) && isFiniteValue(population) &&
    event >= 0.0 && population > 0.0;
}`,C=e=>({name:`statistics`,view:v,type:`f32`,access:e}),E=w,D=e=>`statistics[statisticsOffset + ${e}u]`,O=`events[eventsOffset + row]`,k=`populations[populationsOffset + row]`,A=[...p.nodes,...m.nodes];return A.push(...T(e,{id:r,variant:`totals`,rows:_,accumulators:[{name:`events`,type:`f32`,term:O,rows:`valid`},{name:`population`,type:`f32`,term:k,rows:`valid`},{name:`count`,type:`u32`,term:`1u`,rows:`valid`},{name:`invalid`,type:`u32`,term:`1u`,rows:`invalid`}],rowBindings:y,rowDeclarations:S,finalizeBindings:[C(`read_write`),...l?[{name:`invalidCount`,view:l,type:`u32`,access:`read_write`}]:[]],finalize:`let count = f32(total_count);
    ${l?`invalidCount[invalidCountOffset] = total_invalid;`:``}
    ${D(E.count)} = count;
    ${D(E.population)} = total_population;
    if (count >= 1.0) {
      ${D(E.mean)} = total_events / total_population;
    } else {
      ${D(E.mean)} = getNaN();
      ${D(E.population)} = getNaN();
    }
    ${D(E.variance)} = getNaN();
    ${D(E.priorVariance)} = getNaN();`})),A.push(...T(e,{id:r,variant:`mean-correction`,rows:_,accumulators:[{name:`deviation`,type:`f32`,term:`${O} - ${D(E.mean)} * ${k}`,rows:`valid`}],rowBindings:[...y,C(`read`)],rowDeclarations:S,finalizeBindings:[C(`read_write`)],finalize:`if (${D(E.count)} >= 1.0) {
      ${D(E.mean)} = ${D(E.mean)} + total_deviation / ${D(E.population)};
    }`})),A.push(...T(e,{id:r,variant:`variance`,rows:_,accumulators:[{name:`square`,type:`f32`,term:`${k} * getSquare(${O} / ${k} - ${D(E.mean)})`,rows:`valid`}],rowBindings:[...y,C(`read`)],rowDeclarations:S,finalizeBindings:[C(`read_write`)],finalize:`if (${D(E.count)} >= 1.0) {
      let variance = total_square / ${D(E.population)};
      ${D(E.variance)} = variance;
      ${D(E.priorVariance)} = variance - ${D(E.mean)} / (${D(E.population)} / ${D(E.count)});
    }`})),A.push(i(e,{id:`${r}-standardize`,operation:b,variant:`standardize`,bindings:[...y,C(`read`),{name:`standardizedRates`,view:c,type:`f32`,access:`read_write`}],invocationCount:_,declarations:`${d}
${S}`,body:`if (!isRowValid(index)) {
    standardizedRates[standardizedRatesOffset + index] = getNaN();
    return;
  }
  let population = populations[populationsOffset + index];
  let rate = events[eventsOffset + index] / population;
  let mean = ${D(E.mean)};
  let poissonVariance = mean / population;
  let rawVariance = ${D(E.priorVariance)} + poissonVariance;
  let variance = select(rawVariance, poissonVariance, rawVariance < 0.0);
  standardizedRates[standardizedRatesOffset + index] =
    select(getNaN(), (rate - mean) / sqrt(variance), variance > 0.0);`})),A}},D=p({parameters:y,caption:`Standardized shares of residents age 65+ (Census counts with simulated sampling noise).`,createInputs:({random:e,world:t})=>({tracts:m(e,t),focus:{row:-1}}),build(e,{tracts:t}){let n=e.allocate(`events`,`float32`,t.count),r=e.allocate(`populations`,`float32`,t.count),i=e.allocate(`raw-rates`,`float32`,t.count),a=e.allocate(`standardized-rates`,`float32`,t.count);return e.graph.add(new E({events:n.view,populations:r.view,output:{standardizedRates:a.view}})),{outputs:{standardizedRates:a,rawRates:i,...h(e,t)},update:e=>{let a=_(t,e.riskContrast,e.populationScale);n.write(a.events),r.write(a.populations),i.write(Float32Array.from(a.events,(e,t)=>e/a.populations[t]))}}},interaction:{kind:`probe`,hint:`hover a tract`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=g(t.tracts,e,t.focus.row);if(t.focus.row=a,a<0)return{};let o=_(t.tracts,r.riskContrast,r.populationScale),s=i(n.standardizedRates)?.[a],c=o.events[a]/o.populations[a];return{glyphs:[v(t.tracts,a),{shape:`crosshair`,at:[t.tracts.centers[a*2],t.tracts.centers[a*2+1]],size:10}],readout:`${o.events[a]}/${o.populations[a]} · raw ${(c*100).toFixed(1)}% · z ${s===void 0?`…`:(s>=0?`+`:``)+s.toFixed(1)}`}}},draw:({inputs:e,outputs:t})=>[{layer:`polygons`,positions:t.vertices,indices:t.meshIndices,owners:t.owners,values:t.standardizedRates,triangleCount:e.tracts.triangleCount,ramp:`diverging`,range:[-3,3]},{layer:`lines`,context:!0,segments:t.outline,count:e.tracts.outlineCount}]});export{D as default};
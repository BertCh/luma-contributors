import{t as e}from"./spatial-output-validation-nw5lOmVL.js";import{a as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as n}from"./kernel-C3tfQOn4.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{T as i,f as a,v as o,x as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{l}from"./wgsl-common-DfyXiYB0.js";import{n as u}from"./point-process-utils-BVnRkMJj.js";import{t as d}from"./define-run-B-brkCe4.js";import{A as f}from"./pointer-overlay-Qh8qflcA.js";import{n as p}from"./inputs-CJKKIRKK.js";import{o as m,s as h}from"./pick-Bo3t3hjv.js";import{c as g,n as _,o as v}from"./text-DYSgVV-n.js";import{Rs as y}from"./entry-preview-D70TdSbK.js";function b(e,t,i){let a=t.observed.length,o=t.simulationCount+1,s=n(e,`${i}-extreme-ranks`,`float32`,o),c=n(e,`${i}-critical-rank`,`uint32`,1),u=t.output.pValueInterval??n(e,`${i}-p-interval`,`float32`,2);return[r(e,{id:`${i}-extreme-ranks`,operation:`GPUPointPatternEnvelope`,variant:`extreme-rank`,invocationCount:o,bindings:[{name:`observed`,view:t.observed,type:`f32`,access:`read`},{name:`simulations`,view:t.simulations,type:`f32`,access:`read`},{name:`ranks`,view:s,type:`f32`,access:`read_write`}],declarations:`${l}
fn value(curve: u32, radius: u32) -> f32 {
  if (curve == ${t.simulationCount}u) { return observed[observedOffset + radius]; }
  return simulations[simulationsOffset + curve * ${a}u + radius];
}`,body:`var extreme = ${o}.0; var valid = true;
for (var radius = 0u; radius < ${a}u; radius++) {
  let current = value(index, radius); valid = valid && isFiniteFloat(current);
  var less = 0u; var greater = 0u; var equal = 0u;
  for (var curve = 0u; curve < ${o}u; curve++) {
    let other = value(curve,radius); valid = valid && isFiniteFloat(other);
    less += select(0u,1u,other < current); greater += select(0u,1u,other > current); equal += select(0u,1u,other == current);
  }
  extreme = min(extreme, f32(min(less,greater)) + 0.5 * f32(equal + 1u));
}
ranks[ranksOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), extreme, valid);`}),r(e,{id:`${i}-rank-test`,operation:`GPUPointPatternEnvelope`,variant:`extreme-rank-global-test`,invocationCount:1,bindings:[{name:`ranks`,view:s,type:`f32`,access:`read`},{name:`statistics`,view:t.output.statistics,type:`f32`,access:`read_write`},{name:`status`,view:t.output.status,type:`u32`,access:`read_write`},{name:`interval`,view:u,type:`f32`,access:`read_write`},{name:`critical`,view:c,type:`u32`,access:`read_write`}],declarations:l,body:`let observedRank = ranks[ranksOffset + ${t.simulationCount}u];
var less=0u; var inclusive=0u; var valid=true; var criticalRank=1u;
for (var curve=0u; curve<${o}u; curve++) {
 let rank = ranks[ranksOffset+curve]; valid=valid && isFiniteFloat(rank);
 less += select(0u,1u,rank<observedRank); inclusive += select(0u,1u,rank<=observedRank);
}
for (var candidate=1u; candidate<=${Math.ceil(o/2)}u; candidate++) {
 var rejected=0u;
 for(var curve=0u; curve<${o}u; curve++) { rejected += select(0u,1u,ranks[ranksOffset+curve]<f32(candidate)); }
 if (rejected<=${Math.floor((t.significanceLevel??.05)*o)}u) {criticalRank=candidate;}
}
let invalid=bitcast<f32>(0x7fc00000u | (index & 0u));
statistics[statisticsOffset]=select(invalid,observedRank,valid);
statistics[statisticsOffset+1u]=select(invalid,f32(inclusive)/${o}.0,valid);
statistics[statisticsOffset+2u]=1.0/${o}.0; statistics[statisticsOffset+3u]=${t.simulationCount}.0;
interval[intervalOffset]=select(invalid,f32(less)/${o}.0,valid);
interval[intervalOffset+1u]=select(invalid,f32(inclusive)/${o}.0,valid);
critical[criticalOffset]=criticalRank; status[statusOffset]=select(1u,0u,valid);`}),r(e,{id:`${i}-rank-envelope`,operation:`GPUPointPatternEnvelope`,variant:`global-rank-envelope`,invocationCount:a,bindings:[{name:`observed`,view:t.observed,type:`f32`,access:`read`},{name:`simulations`,view:t.simulations,type:`f32`,access:`read`},{name:`critical`,view:c,type:`u32`,access:`read`},{name:`status`,view:t.output.status,type:`u32`,access:`read`},{name:`lower`,view:t.output.lower,type:`f32`,access:`read_write`},{name:`upper`,view:t.output.upper,type:`f32`,access:`read_write`}],declarations:`fn value(curve:u32,radius:u32)->f32 {if(curve==${t.simulationCount}u){return observed[observedOffset+radius];}return simulations[simulationsOffset+curve*${a}u+radius];}`,body:`var low=bitcast<f32>(0x7fc00000u | (index & 0u)); var high=low;
if(status[statusOffset]==0u) {
 for(var curve=0u;curve<${o}u;curve++) {
  let current=value(curve,index); var rank=0u;
  for(var other=0u;other<${o}u;other++) {let candidate=value(other,index); rank+=select(0u,1u,candidate<current || (candidate==current && other<curve));}
  if(rank+1u==critical[criticalOffset]) {low=current;}
  if(rank==${o}u-critical[criticalOffset]) {high=current;}
 }
}
lower[lowerOffset+index]=low; upper[upperOffset+index]=high;`})]}var x=4096,S=4096,C=16777216,w=(e,t)=>`
fn getOrderedFloatKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn selectOrderedValue(column: u32, order: u32) -> f32 {
  var answer = 0u;
  for (var bit = 32u; bit > 0u; bit--) {
    let trial = answer | (1u << (bit - 1u));
    var below = 0u;
    for (var other = 0u; other < ${t}u; other++) {
      below += select(0u, 1u, getOrderedFloatKey(simulations[simulationsOffset + other * ${e}u + column]) < trial);
    }
    if (below <= order) { answer = trial; }
  }
  return bitcast<f32>(select(~answer, answer ^ 0x80000000u, (answer & 0x80000000u) != 0u));
}
`,T=class{id;props;constructor(t){this.id=t.id??`point-pattern-envelope`,this.props=t;let n=t.observed.length,r=t.rank??1,o=this.id;a(o,`observed rows`,n,1,x),a(o,`simulationCount`,t.simulationCount,2,S),i(o,Number.isInteger(r)&&r>=1&&2*r<=t.simulationCount,`rank must be an integer with 1 <= rank and 2 * rank <= simulationCount`),i(o,t.nullModel.trim().length>0,`nullModel must be non-empty`),i(o,[`none`,`fixed-external`,`refit-each-replicate`].includes(t.fittedParameters),`fittedParameters must be 'none', 'fixed-external' or 'refit-each-replicate'`),i(o,t.statistic===void 0||[`maximum-deviation`,`integrated-squared-deviation`,`extreme-rank`].includes(t.statistic),`statistic must be 'maximum-deviation', 'integrated-squared-deviation' or 'extreme-rank'`);for(let e of[t.observed,t.reference,t.integrationWeights,t.output.lower,t.output.upper])u(e,`float32`,n,o);u(t.simulations,`float32`,n*t.simulationCount,o),u(t.output.statistics,`float32`,4,o),u(t.output.status,`uint32`,1,o),t.statistic===`extreme-rank`&&(i(o,(t.simulationCount+1)**2*n<=C,`(simulationCount + 1)^2 * observed rows must not exceed ${C}`),i(o,(t.significanceLevel??.05)>0&&(t.significanceLevel??.05)<1,`significanceLevel must lie in (0, 1)`)),t.output.pValueInterval&&(i(o,t.statistic===`extreme-rank`,`output.pValueInterval needs statistic 'extreme-rank'`),u(t.output.pValueInterval,`float32`,2,o)),e(this.id,t.output,[[t.output.statistics,t.output.status,t.output.pValueInterval],{read:[t.output.status],write:[t.output.lower,t.output.upper]}]),c(this.id,Object.values(t.output),[t.observed,t.simulations,t.reference,t.integrationWeights])}getCommandNodes(e){let{id:i,props:a}=this,c=a.observed.length,u=a.simulationCount,d=a.rank??1;if(s(i,e,[a.observed,a.simulations,a.reference,a.integrationWeights,...Object.values(a.output)]),t(e,i,{observed:a.observed,simulations:a.simulations,reference:a.reference,integrationWeights:a.integrationWeights,...Object.fromEntries(Object.entries(a.output).map(([e,t])=>[`output.${e}`,t]))}),o(e.device,i,`curve deviations/ranks`,(u+1)*4,`reduce simulationCount`),a.statistic===`extreme-rank`)return b(e,a,i);let f=n(e,`${i}-deviations`,`float32`,u+1);return[r(e,{id:`${i}-deviations`,operation:`GPUPointPatternEnvelope`,variant:a.statistic??`maximum-deviation`,invocationCount:u+1,bindings:[{name:`observed`,view:a.observed,type:`f32`,access:`read`},{name:`simulations`,view:a.simulations,type:`f32`,access:`read`},{name:`reference`,view:a.reference,type:`f32`,access:`read`},{name:`weights`,view:a.integrationWeights,type:`f32`,access:`read`},{name:`deviations`,view:f,type:`f32`,access:`read_write`}],declarations:l,body:`var statistic = 0.0;
  var valid = true;
  for (var radius = 0u; radius < ${c}u; radius++) {
    var value = observed[observedOffset + radius];
    if (index < ${u}u) { value = simulations[simulationsOffset + index * ${c}u + radius]; }
    let center = reference[referenceOffset + radius];
    let weight = weights[weightsOffset + radius];
    valid = valid && isFiniteFloat(value) && isFiniteFloat(center) && isFiniteFloat(weight) && weight > 0.0;
    let deviation = abs(value - center);
    ${a.statistic===`integrated-squared-deviation`?`statistic += weight * deviation * deviation;`:`statistic = max(statistic, deviation);`}
  }
  deviations[deviationsOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), statistic, valid && isFiniteFloat(statistic));`}),r(e,{id:`${i}-test`,operation:`GPUPointPatternEnvelope`,variant:`global-test`,invocationCount:1,bindings:[{name:`deviations`,view:f,type:`f32`,access:`read`},{name:`statistics`,view:a.output.statistics,type:`f32`,access:`read_write`},{name:`status`,view:a.output.status,type:`u32`,access:`read_write`}],declarations:l,body:`let observed = deviations[deviationsOffset + ${u}u];
  var valid = isFiniteFloat(observed);
  var exceedances = 0u;
  for (var replicate = 0u; replicate < ${u}u; replicate++) {
    let value = deviations[deviationsOffset + replicate];
    valid = valid && isFiniteFloat(value);
    exceedances += select(0u, 1u, value >= observed);
  }
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  statistics[statisticsOffset] = select(invalid, observed, valid);
  statistics[statisticsOffset + 1u] = select(invalid, f32(exceedances + 1u) / ${u+1}.0, valid);
  statistics[statisticsOffset + 2u] = 1.0 / ${u+1}.0;
  statistics[statisticsOffset + 3u] = ${u}.0;
  status[statusOffset] = select(1u, 0u, valid);`}),r(e,{id:`${i}-pointwise`,operation:`GPUPointPatternEnvelope`,variant:`pointwise`,invocationCount:c,bindings:[{name:`simulations`,view:a.simulations,type:`f32`,access:`read`},{name:`status`,view:a.output.status,type:`u32`,access:`read`},{name:`lower`,view:a.output.lower,type:`f32`,access:`read_write`},{name:`upper`,view:a.output.upper,type:`f32`,access:`read_write`}],declarations:w(c,u),body:`var lowerValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  var upperValue = lowerValue;
  if (status[statusOffset] == 0u) {
    lowerValue = selectOrderedValue(index, ${d-1}u);
    upperValue = selectOrderedValue(index, ${u-d}u);
  }
  lower[lowerOffset + index] = lowerValue;
  upper[upperOffset + index] = upperValue;`})]}};function E(e,t,n){let r=e.length/2,i=new Float32Array(t.length);for(let n=0;n<r;n++)for(let a=n+1;a<r;a++){let r=Math.hypot(e[n*2]-e[a*2],e[n*2+1]-e[a*2+1]);t.forEach((e,t)=>{r<=e&&(i[t]+=2)})}return Float32Array.from(i,(e,i)=>n*e/(r*(r-1))-Math.PI*t[i]**2)}var D=Float32Array.from({length:20},(e,t)=>.01+t*.01),O=.01,k=99,A=60,j=[1,3,5,7,10],M=[.07,.44,.95,.92],N=.3,P=[.07,.05],F=300,I=[-.04,.4],L=[`maximum-deviation`,`integrated-squared-deviation`,`extreme-rank`],R=d({parameters:y,caption:`Japanese earthquake epicenters against 99 simulated patterns of complete spatial randomness.`,createInputs:({world:e})=>{let t=f(5),n=new Float32Array(k*D.length);for(let r=0;r<k;r++){let i=Float32Array.from({length:120},(n,r)=>r%2==0?t()*e.width:t());n.set(E(i,D,e.width),r*D.length)}return{world:e,simulations:n}},build(e,{world:t,simulations:n},r){let i=f(6),a=t.width,o=()=>[i()*t.width,i()],s=p(i,t,A,1,0).positions;for(let e=0;e<A;e++)i()>r.clustering&&s.set(o(),e*2);let c=e.upload(`observed`,`float32`,E(s,D,a)),l=e.allocate(`lower`,`float32`,D.length),u=e.allocate(`upper`,`float32`,D.length);e.graph.add(new T({observed:c.view,simulations:e.upload(`simulations`,`float32`,n).view,reference:e.upload(`reference`,`float32`,new Float32Array(D.length)).view,integrationWeights:e.upload(`weights`,`float32`,new Float32Array(D.length).fill(.01)).view,simulationCount:k,rank:r.rank,statistic:L[r.statistic],nullModel:`complete spatial randomness`,fittedParameters:`none`,output:{lower:l.view,upper:u.view,statistics:e.allocate(`statistics`,`float32`,4).view,status:e.allocate(`status`,`uint32`,1).view,pValueInterval:r.statistic===2?e.allocate(`p-interval`,`float32`,2).view:void 0}}));let d=Float32Array.from(s,(e,n)=>n%2==0?P[0]*t.width+e*N:P[1]+e*N);return{outputs:{observed:c,lower:l,upper:u,pattern:e.upload(`pattern`,`float32x2`,d),band:e.allocate(`band`,`float32x4`,F),outside:e.allocate(`outside`,`float32x2`,D.length),zero:e.upload(`zero`,`float32`,new Float32Array(D.length))}}},buildCache:5,rebuildOn:[`rank`,`statistic`,`clustering`],interaction:{kind:`level`,hint:`read r, pick the rank`,rest:[.5,1-1.5/j.length],resolve(e,{outputs:t,values:n,world:r,peek:i}){let a=Math.min(D.length-1,Math.max(0,Math.floor(m(M,r,e.x,e.y).u*D.length))),o=j[Math.min(j.length-1,Math.floor((1-e.unitY)*j.length))],s=i(t.observed)?.[a],c=i(t.lower)?.[a],l=i(t.upper)?.[a],u=s===void 0||c===void 0||l===void 0?``:s>l?` above envelope`:s<c?` below envelope`:` inside envelope`,d=i(t.lower),f=i(t.upper),p=i(t.observed),g=0,_=0;if(d&&f&&p){let e=e=>(M[1]+(Math.min(I[1],Math.max(I[0],e))-I[0])/(I[1]-I[0])*(M[3]-M[1]))*r.height,n=new Float32Array(F*4);for(let t=0;t<F;t++){let i=t/299*(D.length-1),a=Math.floor(i),o=Math.min(D.length-1,a+1),s=i-a,c=h(M,r,(i+.5)/D.length);n.set([c,e(d[a]*(1-s)+d[o]*s),c,e(f[a]*(1-s)+f[o]*s)],t*4)}g=F;let i=new Float32Array(D.length*2);for(let t=0;t<D.length;t++)p[t]<=f[t]&&p[t]>=d[t]||i.set([h(M,r,(t+.5)/D.length),e(p[t])],_++*2);t.band.write(n),t.outside.write(i)}return{settle:{rank:o},focus:[g,_],glyphs:[{shape:`curtain`,x:h(M,r,(a+.5)/D.length)}],readout:`r ${D[a].toFixed(2)} · observed ${s?.toFixed(3)??`...`}${u} · rank ${o===n.rank?o:`${n.rank} → ${o}`}`}}},draw:({outputs:e,interaction:t,theme:n,world:r})=>{let[i,a]=Array.isArray(t.focus)?t.focus:[0,0],o=_({rectangle:v(r,M),x:{range:[D[0]-O/2,D[D.length-1]+O/2],ticks:[.05,.1,.15,.2],title:`r`},y:{range:I,ticks:[0,.1,.2,.3,.4],title:`K(r) − CSR`}}),s=.62*r.width;return[{layer:`text`,context:!0,...o.grid},{layer:`dots`,context:!0,positions:e.pattern,color:[...n.ink,.8],radius:4.5},...i>0?[{layer:`lines`,segments:e.band,count:i,radius:2.2,color:[.6,.75,.9,.4]}]:[],{layer:`chart`,kind:`line`,context:!0,values:e.zero,range:I,rectangle:M,color:[.5,.5,.5,.5]},{layer:`chart`,kind:`line`,values:e.lower,range:I,rectangle:M,radius:3.4,color:[.6,.75,.9,1]},{layer:`chart`,kind:`line`,values:e.upper,range:I,rectangle:M,radius:3.4,color:[.6,.75,.9,1]},{layer:`chart`,kind:`line`,values:e.observed,range:I,rectangle:M,radius:3.4,color:[1,.85,.3,1]},...a>0?[{layer:`dots`,positions:e.outside,count:a,color:[...n.categories[1],1],radius:5}]:[],{layer:`text`,...g(o,{labels:[{text:`observed pattern`,at:[P[0]*r.width,P[1]+N],anchor:`bottom-left`,offset:[0,3],role:`label`},{text:`observed K`,at:[s,.26],anchor:`left`,swatch:[1,.85,.3]},{text:`envelope of 99 random patterns`,at:[s,.18],anchor:`left`,swatch:[.6,.75,.9]},{text:`observed outside it`,at:[s,.1],anchor:`left`,swatch:1}]})}]}});export{R as default};
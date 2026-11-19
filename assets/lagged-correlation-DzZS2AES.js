import{a as e,i as t,n,r,t as i}from"./raster-time-series-chunks-b-CPab_8L5.js";import{T as a,f as o}from"./gpu-contributor-utils-BIwLQON5.js";import{t as s}from"./raster-kernel-utils-DcIN7G5I.js";import{t as c}from"./define-run-B-brkCe4.js";import{c as l}from"./pointer-overlay-Qh8qflcA.js";import{a as u,c as d,g as f,n as p,o as m,p as h,r as g,s as _,t as v,u as y,v as b}from"./charts-Br0w6lsw.js";import{l as x}from"./pick-Bo3t3hjv.js";import{n as S,r as C,t as w}from"./probe-DLtMtDEc.js";import{n as T,t as E}from"./readback-C6WbEPkA.js";import{Kr as D}from"./entry-preview-D70TdSbK.js";function O(e={},t=new Float32Array(1)){return t[0]=e.minimumPairCount??3,t}var k=class{id;props;chunkPlan;estimatedCommandNodeCount;constructor(t){this.props=t;let r=t.id??`raster-lagged-correlation`;this.id=r;let{cellCount:i,sliceCount:s,lags:c}=t;o(r,`cellCount`,i),o(r,`sliceCount`,s,2,256),o(r,`lags length`,c.length,1,64),a(r,c.every(e=>Number.isInteger(e)&&Math.abs(e)<s),`lags must be integers in (-sliceCount, sliceCount)`);let l=i*s;o(r,`cellCount times sliceCount`,l),a(r,[t.values,t.mask].every(e=>!e||e.length>=l),`values and mask must hold cellCount times sliceCount entries`),a(r,t.referenceIndex.length>=s,`referenceIndex must hold one entry per slice`);let u=i*c.length;a(r,[t.output.correlations,t.output.status,t.output.pairCount].every(e=>!e||e.length>=u),`output views must hold one entry per cell and lag`),a(r,t.parameters.length>=1,`parameters must hold 1 entries`),this.chunkPlan=n(this.id,t.cellCount,{values:{view:t.values,entriesPerCell:t.sliceCount},mask:{view:t.mask,entriesPerCell:t.sliceCount},"output.correlations":{view:t.output.correlations,entriesPerCell:t.lags.length},"output.status":{view:t.output.status,entriesPerCell:t.lags.length},"output.pairCount":{view:t.output.pairCount,entriesPerCell:t.lags.length}}),this.estimatedCommandNodeCount=this.chunkPlan.estimatedCommandNodeCount,e(this.id,this.getInputs(),this.getOutputs(),this.chunkPlan)}getInputs(){let{values:e,mask:t,referenceIndex:n,parameters:r}=this.props;return[e,n,r,...t?[t]:[]]}getOutputs(){let{correlations:e,status:t,pairCount:n}=this.props.output;return[e,t,...n?[n]:[]]}getCommandNodes(e){let{props:n,id:a}=this,{sliceCount:o,values:c,mask:l,referenceIndex:u,parameters:d,lags:f,output:p}=n;t(a,e,this.getInputs(),this.getOutputs());let m=f.length;return i(e,this.chunkPlan,{id:a,operation:`GPURasterLaggedCorrelation`,invocationCount:n.cellCount,workgroupSize:16,bindings:[r(`values`,c),r(`referenceIndex`,u),r(`parameters`,d),...l?[r(`mask`,l)]:[],r(`correlations`,p.correlations,!0),r(`status`,p.status,!0),...p.pairCount?[r(`pairCount`,p.pairCount,!0)]:[]],declarations:`${s}
const LAGS = array<i32, ${m}>(${f.map(e=>`${e}`).join(`, `)});`,body:`let minimumPairs = parameters[parametersOffset];
for (var lagIndex = 0u; lagIndex < ${m}u; lagIndex++) {
  let lag = LAGS[lagIndex];
  var count = 0u;
  var indexTotal = 0.0;
  var cellTotal = 0.0;
  for (var slice = 0i; slice < ${o}i; slice++) {
    let targetSlice = slice + lag;
    if (targetSlice < 0i || targetSlice >= ${o}i) { continue; }
    let indexValue = referenceIndex[referenceIndexOffset + u32(slice)];
    let source = index * ${o}u + u32(targetSlice);
    let cellValue = values[valuesOffset + source];
    if (!finite(indexValue) || !finite(cellValue) ${l?`|| mask[maskOffset + source] == 0u`:``}) { continue; }
    count++;
    indexTotal += indexValue;
    cellTotal += cellValue;
  }
  var code = 1u;
  var correlation = missing();
  if (f32(count) >= minimumPairs) {
    let indexMean = indexTotal / f32(count);
    let cellMean = cellTotal / f32(count);
    var crossTotal = 0.0;
    var indexSquares = 0.0;
    var cellSquares = 0.0;
    for (var slice = 0i; slice < ${o}i; slice++) {
      let targetSlice = slice + lag;
      if (targetSlice < 0i || targetSlice >= ${o}i) { continue; }
      let indexValue = referenceIndex[referenceIndexOffset + u32(slice)];
      let source = index * ${o}u + u32(targetSlice);
      let cellValue = values[valuesOffset + source];
      if (!finite(indexValue) || !finite(cellValue) ${l?`|| mask[maskOffset + source] == 0u`:``}) { continue; }
      let indexAnomaly = indexValue - indexMean;
      let cellAnomaly = cellValue - cellMean;
      crossTotal += indexAnomaly * cellAnomaly;
      indexSquares += indexAnomaly * indexAnomaly;
      cellSquares += cellAnomaly * cellAnomaly;
    }
    // Float32 means carry rounding, so variance below it counts as zero rather than as a correlation.
    if (indexSquares > f32(count) * indexMean * indexMean * 1e-12 && cellSquares > f32(count) * cellMean * cellMean * 1e-12) {
      code = 0u;
      correlation = clamp(crossTotal / sqrt(indexSquares * cellSquares), -1.0, 1.0);
    } else {
      code = 2u;
    }
  }
  correlations[correlationsOffset + index * ${m}u + lagIndex] = correlation;
  status[statusOffset + index * ${m}u + lagIndex] = code;
  ${p.pairCount?`pairCount[pairCountOffset + index * ${m}u + lagIndex] = count;`:``}
}`})}},A=64,j=[-4,-2,0,2,4,6,8],M=[.55,.5],[N,P]=g(2),F=[-.2,1],I=[-1,1],L=[0,63],R=[j[0]-1,j[j.length-1]+1];function z(e){return e.observed.slice(e.referenceCell*A,(e.referenceCell+1)*A)}var B=c({parameters:D,caption:`Synthetic green-up wave moving east on measured Dixie NDVI levels; reference is the picked pixel. Regular slots.`,createInputs:({aspect:e,random:t})=>{let n=b(e,t,{sliceCount:A,yearCount:4,jitter:0,phaseDrift:.35}),r=n.observe(.05,.1),i=Math.floor(M[0]*n.grid.columns),a=Math.floor(M[1]*n.grid.rows);return{...n,observed:r,referenceCell:a*n.grid.columns+i}},build(e,t){let{grid:n,sliceCount:r}=t,i=e.upload(`series`,`float32`,t.observed),a=e.upload(`reference`,`float32`,z(t)),o=e.allocate(`parameters`,`float32`,1),s={correlations:e.allocate(`correlations`,`float32`,n.cellCount*j.length),status:e.allocate(`status`,`uint32`,n.cellCount*j.length),pairCount:e.allocate(`pair-count`,`uint32`,n.cellCount*j.length)};return e.graph.add(new k({cellCount:n.cellCount,sliceCount:r,values:i.view,referenceIndex:a.view,parameters:o.view,lags:j,output:{correlations:s.correlations.view,status:s.status.view,pairCount:s.pairCount.view}})),{outputs:{...s,blank:E(e,`blank`,n.cellCount),level:e.upload(`level`,`float32`,t.level),cellSeries:v(e,`cell-series`,r),referenceSeries:v(e,`reference-series`,r),lagDots:v(e,`lag-dots`,j.length)},update:e=>{t.observed=t.observe(.05,e.gapFraction),i.write(t.observed),a.write(z(t)),o.write(O({minimumPairCount:e.minimumPairCount}))}}},async afterRun({outputs:e,inputs:t,values:n,upload:r}){let i=await T(e.correlations),a=Math.max(0,j.indexOf(n.lag));return{map:r(`correlation-map`,`float32`,Float32Array.from({length:t.grid.cellCount},(e,n)=>t.level[n]>0?i[n*j.length+a]:NaN))}},interaction:{kind:`place`,hint:`pick the reference pixel`,rest:M,resolve:(e,{inputs:t,outputs:n,values:r,world:i,peek:a})=>{let{grid:o}=t,s=C(o,i,e);t.referenceCell=s.index;let c=a(n.correlations),u=Math.max(0,j.indexOf(r.lag)),d=-1,p=-2;if(c)for(let e=0;e<o.cellCount;e++){let t=c[e*j.length+u];e!==s.index&&Number.isFinite(t)&&t>p&&(p=t,d=e)}let g=Array.from({length:A},(e,t)=>t);f(n.referenceSeries,g,z(t)),f(n.cellSeries,g,d>=0?t.observed.subarray(d*A,(d+1)*A):[]);let v=c&&d>=0?Array.from(c.subarray(d*j.length,(d+1)*j.length)):[];f(n.lagDots,j,v);let[y,b]=x(o.columns,o.rows,[0,0,i.width,i.height],s.column,s.row),T=d%o.columns,E=Math.floor(d/o.columns),[D,O]=d>=0?x(o.columns,o.rows,[0,0,i.width,i.height],T,E):[y,b];return{rerun:!0,spotlight:null,glyphs:[S(o,i,s),m(i,N),m(i,P),_(i,P,I,0),h(i,P,R,r.lag,!0),...d>=0?[{shape:`arrow`,from:[y,b],to:[D,O]}]:[]],readout:d>=0?`lag ${r.lag} · best r ${l(p)} · ${T-s.column} columns east · ${w(o,s)}`:`reading the correlations`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>[y(t.level,e.grid,e.grid.mapLeft,{context:!0,ramp:`sequential`,range:[0,1]}),y(n?.map??t.blank,e.grid,e.grid.mapLeft,{ramp:`diverging`,range:[-1,1]}),u(t.referenceSeries,N,F,L,p(r,1),3.5),u(t.cellSeries,N,F,L,d(r),4),u(t.lagDots,P,I,R,p(r,0),6)]});export{B as default};
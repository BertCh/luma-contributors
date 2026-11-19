import{r as e}from"./kernel-C3tfQOn4.js";import{o as t,t as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,T as i,f as a,x as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{r as c}from"./wgsl-common-DfyXiYB0.js";import{c as l,r as u}from"./spatial-weights-CqWysstN.js";import{t as d}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as f}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as p}from"./define-run-B-brkCe4.js";import{m,u as h}from"./inputs-CJKKIRKK.js";import{c as g,r as _,s as v}from"./probes-CVwGX4BJ.js";import{o as y}from"./tract-map-BETLV_b0.js";import{t as b}from"./readback-cfnHpL6G.js";import{Ml as x}from"./entry-preview-D70TdSbK.js";var S=`GPULocalCategoryDependence`,C={likeCount:0,neighborCount:1,proportion:2,expectedProportion:3,upperTailProbability:4,standardizedScore:5,length:6},w=class{id;props;rowCount;constructor(e){let t=e.id??`local-category-dependence`;this.id=t,this.props=e,this.rowCount=l(t,e.weights),a(t,`categoryCount`,e.categoryCount,1,32),r(e.categories,[`uint32`],`${t} categories`),i(t,e.categories.length===this.rowCount,`categories length must equal the weights row count`),r(e.output.statistics,[`float32`],`${t} output.statistics`),i(t,e.output.statistics.length>=this.rowCount*C.length,`output.statistics must hold rows * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length entries`),e.output.categoryRowCounts&&(r(e.output.categoryRowCounts,[`uint32`],`${t} output.categoryRowCounts`),i(t,e.output.categoryRowCounts.length>=e.categoryCount,`output.categoryRowCounts must hold categoryCount entries`)),f(t,e.output),s(t,[e.output.statistics,e.output.categoryRowCounts],[e.categories,e.weights.offsets,e.weights.neighbors,e.weights.weights])}getCommandNodes(r){let{id:i,props:a,rowCount:s}=this;d(r,i,a,{"category counters":a.categoryCount*4});let{weights:l,output:f,categoryCount:p}=a,m=C;o(i,r,[a.categories,l.offsets,l.neighbors,l.weights,f.statistics,f.categoryRowCounts]);let h=f.categoryRowCounts??e(r,`${i}-category-row-counts`,`uint32`,p);return[n(r,{id:`${i}-clear-rows`,operation:S,view:h,type:`u32`,value:`0u`,componentCount:p}),t(r,{id:`${i}-count-rows`,operation:S,variant:`count-rows`,bindings:[{name:`categories`,view:a.categories,type:`u32`,access:`read`},{name:`rowCounts`,view:h,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,declarations:`const CATEGORIES: u32 = ${p}u;`,body:`let own = categories[categoriesOffset + index];
  if (own < CATEGORIES) {
    atomicAdd(&rowCounts[rowCountsOffset + own], 1u);
  }`}),t(r,{id:`${i}-statistics`,operation:S,variant:`statistics`,bindings:[{name:`offsets`,view:l.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:l.neighbors,type:`u32`,access:`read`},{name:`weights`,view:l.weights,type:`f32`,access:`read`},{name:`categories`,view:a.categories,type:`u32`,access:`read`},{name:`rowCounts`,view:h,type:`u32`,access:`read`},{name:`statistics`,view:f.statistics,type:`f32`,access:`read_write`}],invocationCount:s,declarations:`const ROWS: u32 = ${s}u;
const CAPACITY: u32 = ${l.neighbors.length}u;
const CATEGORIES: u32 = ${p}u;
const COLUMNS: u32 = ${m.length}u;
${c}
// P(X >= like), X ~ Binomial(count, p), by the log-space pmf recurrence over k = 0..count.
fn getUpperTail(like: u32, count: u32, p: f32) -> f32 {
  if (like == 0u || p >= 1.0) {
    return 1.0;
  }
  let logP = log(p);
  let logQ = log(1.0 - p);
  var logChoose = 0.0;
  var tail = 0.0;
  for (var k = 0u; k <= count; k++) {
    if (k >= like) {
      tail += exp(logChoose + f32(k) * logP + f32(count - k) * logQ);
    }
    logChoose += log(f32(count - k) / f32(k + 1u));
  }
  return min(tail, 1.0);
}`,body:`let own = categories[categoriesOffset + index];
  let base = statisticsOffset + index * COLUMNS;
  let nan = getNaN();
  var likeCount = 0u;
  var neighborCount = 0u;
  if (own < CATEGORIES) {
    ${u(`categories[categoriesOffset + neighbor] < CATEGORIES`,`neighborCount++;
      likeCount += select(0u, 1u, categories[categoriesOffset + neighbor] == own);`)}
  }
  statistics[base + ${m.likeCount}u] = f32(likeCount);
  statistics[base + ${m.neighborCount}u] = f32(neighborCount);
  var total = 0u;
  for (var b = 0u; b < CATEGORIES; b++) {
    total += rowCounts[rowCountsOffset + b];
  }
  if (own >= CATEGORIES || neighborCount == 0u || total == 0u) {
    statistics[base + ${m.proportion}u] = nan;
    statistics[base + ${m.expectedProportion}u] = nan;
    statistics[base + ${m.upperTailProbability}u] = nan;
    statistics[base + ${m.standardizedScore}u] = nan;
    return;
  }
  let p = f32(rowCounts[rowCountsOffset + own]) / f32(total);
  let n = f32(neighborCount);
  let variance = n * p * (1.0 - p);
  statistics[base + ${m.proportion}u] = f32(likeCount) / n;
  statistics[base + ${m.expectedProportion}u] = p;
  statistics[base + ${m.upperTailProbability}u] = getUpperTail(likeCount, neighborCount, p);
  statistics[base + ${m.standardizedScore}u] = select(nan, (f32(likeCount) - n * p) / sqrt(variance), variance > 0.0);`})]}},T=C,E=p({parameters:x,caption:`Chicago tract classes (quantiles of poverty, unemployment or the 65+ share) and how unusual each neighbourhood is.`,createInputs:({random:e,world:t})=>({...h(e,t),focus:{tract:-1}}),build(e,t,n){let r=m(e,t.contiguity),i=e.allocate(`categories`,`uint32`,t.count),a=e.allocate(`statistics`,`float32`,t.count*T.length);return e.graph.add(new w({weights:r,categories:i.view,categoryCount:n.categoryCount,output:{statistics:a.view}})),{outputs:{categories:i,statistics:a,vertices:e.upload(`vertices`,`float32x2`,t.vertices),indices:e.upload(`indices`,`uint32`,t.triangleIndices),owners:e.upload(`owners`,`uint32`,t.triangleOwners),centers:e.upload(`centers`,`float32x2`,t.centers),outline:e.upload(`outline`,`float32x4`,t.outline)},update:e=>{let n=[t.povertyShare,t.unemploymentShare,Float32Array.from(t.age65,(e,n)=>e/t.population[n])][e.variable];i.write(y(n,e.categoryCount))}}},rebuildOn:[`categoryCount`],buildCache:3,async afterRun({outputs:e,upload:t}){let n=await b(e.statistics);return{score:t(`score`,`float32`,Float32Array.from({length:n.length/T.length},(e,t)=>{let r=n[t*T.length+T.upperTailProbability];return Number.isFinite(r)?-Math.log10(Math.max(r,1e-4)):0}))}},interaction:{kind:`probe`,hint:`hover a tract`,rest:[.3,.5],resolve(e,{inputs:t,outputs:n,peek:r}){let i=_(t,e.x,e.y);i>=0&&(t.focus.tract=i);let a=t.focus.tract;if(a<0)return{};let o=r(n.statistics),s=r(n.categories),c=[g(t,a)];if(s)for(let e of v(t,a))c.push(g(t,e,0,s[e]!==s[a]));if(!o)return{glyphs:c,spotlight:null};let l=a*T.length,u=o[l+T.likeCount],d=o[l+T.neighborCount],f=o[l+T.expectedProportion],p=o[l+T.upperTailProbability];return{glyphs:c,spotlight:null,readout:`like ${u}/${d} · expected share ${f.toFixed(2)} · p ${Number.isFinite(p)?p.toPrecision(2):`–`}`}}},draw:({inputs:e,outputs:t,extra:n})=>[{layer:`polygons`,positions:t.vertices,indices:t.indices,owners:t.owners,values:n.score,triangleCount:e.triangleCount,ramp:`sequential`,range:[0,3]},{layer:`lines`,context:!0,segments:t.outline,count:e.outlineCount},{layer:`dots`,positions:t.centers,values:t.categories,radius:4.5}]});export{E as default};
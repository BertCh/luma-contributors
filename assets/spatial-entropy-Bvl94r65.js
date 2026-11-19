import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as n,T as r,f as i,x as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{t as s}from"./spatial-statistics-counts-wgsl-B8N7XLDs.js";import{r as c}from"./wgsl-common-DfyXiYB0.js";import{t as l}from"./gpu-category-pair-counts-BamOyHbN.js";import{t as u}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{c as f}from"./spatial-weights-CqWysstN.js";import{t as p}from"./define-run-B-brkCe4.js";import{m}from"./inputs-CJKKIRKK.js";import{a as h}from"./pointer-overlay-Qh8qflcA.js";import{a as g,i as _,o as v,r as y,s as b,t as x}from"./tract-map-BETLV_b0.js";import{t as S}from"./readback-cfnHpL6G.js";import{El as C}from"./entry-preview-D70TdSbK.js";var w=`GPUSpatialEntropy`,T={shannon:0,originEntropy:1,destinationEntropy:2,cooccurrence:3,mutualInformation:4,relativeMutualInformation:5,linkCount:6,rowCount:7,length:8},E=class{id;props;constructor(e){let t=e.id??`spatial-entropy`;this.id=t,this.props=e,n(e.output.summary,[`float32`],`${t} output.summary`),r(t,e.output.summary.length>=T.length,`output.summary must hold ${T.length} words`),r(t,e.categoryCount<=32,`categoryCount must be at most 32`),i(t,`categoryCount`,e.categoryCount,1,32);let a=f(t,e.weights);n(e.categories,[`uint32`],`${t} categories`),r(t,e.categories.length===a,`categories length must equal the weights row count`),e.output.pairCounts&&(n(e.output.pairCounts,[`uint32`],`${t} output.pairCounts`),r(t,e.output.pairCounts.length>=e.categoryCount**2,`output.pairCounts must hold categoryCount squared entries`)),e.output.counts&&(n(e.output.counts,[`uint32`],`${t} output.counts`),i(t,`output.counts length`,e.output.counts.length,2,2)),d(t,e.output),o(t,Object.values(e.output),[e.categories,e.weights.offsets,e.weights.neighbors,e.weights.weights])}getCommandNodes(n){let{id:r,props:i}=this;u(n,r,i,{"pair count scratch":i.categoryCount**2*4,"category scratch":i.categoryCount*4});let{categoryCount:o,output:d}=i;a(r,n,[d.summary,d.pairCounts]);let f=d.pairCounts??e(n,`${r}-pair-counts`,`uint32`,o**2),p=e(n,`${r}-category-row-counts`,`uint32`,o);return[...new l({id:`${r}-pairs`,weights:i.weights,categories:i.categories,categoryCount:o,output:{pairCounts:f,categoryRowCounts:p}}).getCommandNodes(n),t(n,{id:`${r}-summary`,operation:w,variant:`summary`,bindings:[{name:`pairCounts`,view:f,type:`u32`,access:`read`},{name:`categoryRowCounts`,view:p,type:`u32`,access:`read`},{name:`summary`,view:d.summary,type:`f32`,access:`read_write`},...d.counts?[{name:`counts`,view:d.counts,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`const CATEGORIES: u32 = ${o}u;
${c}
${s}
fn getTerm(count: u32, total: u32) -> f32 {
  if (count == 0u) {
    return 0.0;
  }
  let share = divideSpatialCounts(count, total);
  return -share * log(share);
}`,body:`var rowTotal = 0u;
  var linkTotal = 0u;
  for (var a = 0u; a < CATEGORIES; a++) {
    rowTotal += categoryRowCounts[categoryRowCountsOffset + a];
    for (var b = 0u; b < CATEGORIES; b++) {
      linkTotal += pairCounts[pairCountsOffset + a * CATEGORIES + b];
    }
  }
  ${d.counts?`counts[countsOffset] = rowTotal; counts[countsOffset + 1u] = linkTotal;`:``}
  var shannon = 0.0;
  var origin = 0.0;
  var destination = 0.0;
  var joint = 0.0;
  for (var a = 0u; a < CATEGORIES; a++) {
    shannon += getTerm(categoryRowCounts[categoryRowCountsOffset + a], rowTotal);
    var originCount = 0u;
    var destinationCount = 0u;
    for (var b = 0u; b < CATEGORIES; b++) {
      originCount += pairCounts[pairCountsOffset + a * CATEGORIES + b];
      destinationCount += pairCounts[pairCountsOffset + b * CATEGORIES + a];
      joint += getTerm(pairCounts[pairCountsOffset + a * CATEGORIES + b], linkTotal);
    }
    origin += getTerm(originCount, linkTotal);
    destination += getTerm(destinationCount, linkTotal);
  }
  let nan = getNaN();
  let hasLinks = linkTotal > 0u;
  let mutual = select(nan, max(origin + destination - joint, 0.0), hasLinks);
  summary[summaryOffset + ${T.shannon}u] = select(nan, shannon, rowTotal > 0u);
  summary[summaryOffset + ${T.originEntropy}u] = select(nan, origin, hasLinks);
  summary[summaryOffset + ${T.destinationEntropy}u] = select(nan, destination, hasLinks);
  summary[summaryOffset + ${T.cooccurrence}u] = select(nan, joint, hasLinks);
  summary[summaryOffset + ${T.mutualInformation}u] = mutual;
  summary[summaryOffset + ${T.relativeMutualInformation}u] = select(nan, mutual / destination, hasLinks && destination > 0.0);
  summary[summaryOffset + ${T.linkCount}u] = f32(linkTotal);
  summary[summaryOffset + ${T.rowCount}u] = f32(rowTotal);`})]}},D=T,O=[D.shannon,D.originEntropy,D.destinationEntropy,D.cooccurrence,D.mutualInformation],k=p({parameters:C,caption:`Poverty classes of Chicago tracts. Bars (nats): Shannon, origin, destination, co-occurrence, mutual information.`,createInputs:({random:e,world:t})=>{let n=y(e,t),r=Array.from({length:n.count},(e,t)=>t);for(let t=r.length-1;t>0;t--){let n=Math.floor(e()*(t+1));[r[t],r[n]]=[r[n],r[t]]}return{...n,order:r}},build(e,t,n){let r=e.allocate(`categories`,`uint32`,t.count),i=e.allocate(`summary`,`float32`,D.length);return e.graph.add(new E({weights:m(e,t.contiguity),categories:r.view,categoryCount:n.categoryCount,output:{summary:i.view}})),{outputs:{categories:r,summary:i,map:b(e,t)},update:e=>{let n=v(t.povertyShare,e.categoryCount),i=Math.round(e.scrambled*t.count),a=t.order.slice(0,i),o=a.map(e=>n[e]);a.forEach((e,t)=>{n[e]=o[(t+1)%o.length]}),r.write(n)}}},rebuildOn:[`categoryCount`],buildCache:4,async afterRun({outputs:e,upload:t}){let n=await S(e.summary);return{bars:t(`bars`,`float32`,Float32Array.from(O,e=>Number.isFinite(n[e])?n[e]:0)),summary:n}},interaction:h(C,{kind:`time`,hint:`scramble the map`,x:`scrambled`,decorate:(e,t,n)=>{let r=n.peek(n.outputs.summary);return{readout:`scrambled ${Math.round(e.scrambled*100)}% · mutual information ${r?r[D.mutualInformation].toFixed(3):`…`} nats`}}}),draw:({inputs:e,outputs:t,extra:n})=>[g(t.map,e,t.categories),_(t.map,e),{layer:`chart`,kind:`bars`,values:n.bars,ramp:`sequential`,range:[0,Math.log(5)],rectangle:x}]});export{k as default};
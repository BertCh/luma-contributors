import{_ as e,a as t,c as n,d as r,f as i,g as a,h as o,i as s,l as c,m as l,n as u,o as d,p as f,r as p,s as m,t as h,u as ee,v as g}from"./space-time-kernels-a9nSiN0t.js";import{t as te}from"./spatial-weights-suppression-4G-LxXB_.js";import{t as _}from"./space-time-construction-Ck1ue1-P.js";import{a as v,i as y,n as ne,r as b}from"./trajectory-binding-validation-C89oJIAZ.js";import{t as x}from"./philox-random-DfVxWNPn.js";import{r as S}from"./kernel-C3tfQOn4.js";import{o as C}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as w,f as T,x as E,y as D}from"./gpu-contributor-utils-BIwLQON5.js";import{c as O}from"./spatial-weights-CqWysstN.js";import{t as k}from"./define-run-B-brkCe4.js";import{m as A}from"./inputs-CJKKIRKK.js";import{n as j,t as M}from"./permutation-chart-dNDRNqOt.js";import{c as N,h as P,l as F,n as I,u as L}from"./point-field-CLZ0m0cx.js";import{jt as R}from"./entry-preview-D70TdSbK.js";var z=`GPUMantelTest`,B=3,V=4096;function H(e,t){return t?`fn ${e}(value: f32) -> f32 { return pow(value + ${`(${t.constant.toPrecision(9)})`}, ${`(${t.power.toPrecision(9)})`}); }`:`fn ${e}(value: f32) -> f32 { return value; }`}var U=class{id;props;rows;constructor(e){e=_(e,[`id`,`pairs`,`times`,`parameters`,`maximumPermutations`,`stopAfterExceedances`,`distanceTransform`,`timeTransform`,`output`],`pairs`);let t=e.id??`mantel-test`;if(this.id=t,this.props=e,this.rows=O(t,e.pairs,`pairs`),!e.pairs.distances)throw Error(`${t} pairs.distances is required`);if(w(e.times,[`float32`],`${t} times`),w(e.parameters,[`uint32`],`${t} parameters`),w(e.output.statistics,[`float32`],`${t} output.statistics`),w(e.output.summary,[`float32`],`${t} output.summary`),this.rows<2)throw Error(`${t} needs at least two rows`);if(this.rows>p)throw Error(`${t} supports at most ${p} rows`);if(e.times.length!==this.rows)throw Error(`${t} times needs one row per pairs row`);if(e.parameters.length<4)throw Error(`${t} parameters needs 4 words`);if(T(t,`maximumPermutations`,e.maximumPermutations,1,u),e.output.statistics.length<e.maximumPermutations+1)throw Error(`${t} output.statistics needs maximumPermutations + 1 rows`);if(o(t,e.stopAfterExceedances,e.maximumPermutations,e.output.permutationCount),e.output.summary.length<12)throw Error(`${t} output.summary needs 12 rows`);if(b(this.id,this.props.output),v(this.id,[{summary:e.output.summary,permutationCount:e.output.permutationCount}]),D(t,[e.output.statistics,e.output.summary,e.output.permutationCount],[e.times,e.parameters,e.pairs.offsets,e.pairs.neighbors,e.pairs.weights,e.pairs.distances,e.pairs.status?.overflow,e.pairs.status?.candidateOverflow]),e.output.statistics.buffer===e.output.summary.buffer||e.output.permutationCount&&(e.output.permutationCount.buffer===e.output.statistics.buffer||e.output.permutationCount.buffer===e.output.summary.buffer))throw Error(`${t} outputs must use separate buffers`)}getInferenceMetadata(t){return e(`mantel`,t,this.props.maximumPermutations,this.props.stopAfterExceedances)}getCommandNodes(e){ne(e,this.id,this.props);let{id:a,props:o,rows:u}=this,{pairs:p,times:g,parameters:_,maximumPermutations:v}=o,{statistics:b,summary:w}=o.output,T=p.distances;E(a,e,[p.offsets,p.neighbors,p.weights,T,g,_,b,w,...o.output.permutationCount?[o.output.permutationCount]:[]]);let D=v+1,O=Math.floor(e.device.limits.maxStorageBufferBindingSize/4),[k,A]=d(u,Math.floor(O/(D*B+2))),j=n(k),M=o.stopAfterExceedances,N=M!==void 0,P=l(D),F=N?P:D;y(e,a,{"permutation partials":(F*k*B+k*2)*4,"pair statistic sums":Math.min(V,u)*8,"pair mask":p.neighbors.length*4,...N?{"stopping state":32,"stopping dispatch":24}:{}});let I=N?i(e,a):void 0,L=I?[{name:`state`,view:I,type:`u32`,access:`read`}]:[],R=N?`\nconst BATCH_SIZE: u32 = ${P}u;`:``,U=`${m(u,v,k,A)}
${H(`transformDistance`,o.distanceTransform)}
${H(`transformTime`,o.timeTransform)}`,W=Math.min(V,u),G=S(e,`${a}-pair-counts`,`uint32`,W),K=S(e,`${a}-pair-sums`,`float32`,W*2),q=S(e,`${a}-pair-count`,`uint32`,1),J=S(e,`${a}-pair-mask`,`uint32`,p.neighbors.length),Y=N?void 0:S(e,`${a}-shift`,`float32`,2),X=S(e,`${a}-partials`,`float32`,F*k*B+k*2),Z=[{name:`offsets`,view:p.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:p.neighbors,type:`u32`,access:`read`},{name:`pairMask`,view:J,type:`u32`,access:`read`},{name:`distances`,view:T,type:`f32`,access:`read`},{name:`times`,view:g,type:`f32`,access:`read`}],Q=t=>C(e,{id:t?.id??`${a}-partials`,operation:z,variant:`partials`,bindings:[...Z,{name:`parameters`,view:_,type:`u32`,access:`read`},...N?L:[{name:`shift`,view:Y,type:`f32`,access:`read`}],{name:`partials`,view:X,type:`f32`,access:`read_write`}],condition:t?.condition,extraResources:t?.extraResources,workgroupSize:64,invocationCount:F*j*64,guardIndex:!1,declarations:`${U}
${c(k,x.mantelTestShuffle.id)}
const SLOT_COUNT: u32 = ${F}u;${R}
const MOMENTS_BASE: u32 = partialsOffset + ${F*k*B}u;`,body:`${N?h:s}
  var sumX = 0.0;
  var sumXX = 0.0;
  var sumY = 0.0;
  var sumYY = 0.0;
  var sumXY = 0.0;
  if (runsSlot) {
    ${N?`let shiftDistance = bitcast<f32>(state[stateOffset + ${r.shiftDistance}u]);
    let shiftTime = bitcast<f32>(state[stateOffset + ${r.shiftTime}u]);`:`let shiftDistance = shift[shiftOffset];
    let shiftTime = shift[shiftOffset + 1u];`}
    for (var row = block; row < ROWS; row += BLOCKS) {
      let time = times[timesOffset + getPermutedRow(row, slot)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let other = neighbors[neighborsOffset + entry];
        if (pairMask[pairMaskOffset + entry] != 0u) {
          let x = transformDistance(distances[distancesOffset + entry]) - shiftDistance;
          let y = transformTime(abs(time - times[timesOffset + getPermutedRow(other, slot)])) - shiftTime;
          if (slot == 0u) {
            sumX += x;
            sumXX += x * x;
          }
          sumY += y;
          sumYY += y * y;
          sumXY += x * y;
        }
      }
    }
  }
  if (block < BLOCKS) {
    let base = partialsOffset + (block * SLOT_COUNT + ${N?`localSlot`:`slot`}) * ${B}u;
    partials[base] = sumY;
    partials[base + 1u] = sumYY;
    partials[base + 2u] = sumXY;
  }
  if (slot == 0u && block < BLOCKS) {
    partials[MOMENTS_BASE + 2u * block] = sumX;
    partials[MOMENTS_BASE + 2u * block + 1u] = sumXX;
  }`}),$=t=>C(e,{id:t?.id??`${a}-reduce`,operation:z,variant:`reduce`,bindings:[{name:`parameters`,view:_,type:`u32`,access:`read`},...L,{name:`partials`,view:X,type:`f32`,access:`read`},{name:`pairCount`,view:q,type:`u32`,access:`read`},{name:`statistics`,view:b,type:`f32`,access:`read_write`}],condition:t?.condition,extraResources:t?.extraResources,...N?{workgroupSize:64,invocationCount:P}:{invocationCount:b.length},declarations:`${U}
const SLOT_COUNT: u32 = ${F}u;${R}
const MOMENTS_BASE: u32 = partialsOffset + ${F*k*B}u;`,body:`${N?`let slot = state[stateOffset] * BATCH_SIZE + index;
  let localSlot = index;
  let runs = slot <= readPermutationCount();`:`let slot = index;
  let localSlot = index;
  let runs = index <= readPermutationCount();`}
  var result = 0.0;
  if (runs) {
    var sumX = 0.0;
    var sumXX = 0.0;
    var sumY = 0.0;
    var sumYY = 0.0;
    var sumXY = 0.0;
    for (var block = 0u; block < BLOCKS; block++) {
      let base = partialsOffset + (block * SLOT_COUNT + localSlot) * ${B}u;
      sumX += partials[MOMENTS_BASE + 2u * block];
      sumXX += partials[MOMENTS_BASE + 2u * block + 1u];
      sumY += partials[base];
      sumYY += partials[base + 1u];
      sumXY += partials[base + 2u];
    }
    let count = f32(pairCount[pairCountOffset]);
    let covariance = sumXY - sumX * sumY / count;
    let varianceX = sumXX - sumX * sumX / count;
    let varianceY = sumYY - sumY * sumY / count;
    if (count > 1.0 && varianceX > 0.0 && varianceY > 0.0) {
      result = clamp(covariance / sqrt(varianceX * varianceY), -1.0, 1.0);
    }
  }
  ${N?`if (runs) { statistics[statisticsOffset + slot] = result; }`:`statistics[statisticsOffset + slot] = result;`}`}),re=N&&I?f(e,{id:a,operation:z,stopAfterExceedances:M,slotCount:D,groupsPerSlot:j,workgroupSize:64,statisticType:`f32`,parameters:_,statistics:b,state:I,common:U,createPartialsNode:Q,createReduceNode:$}):[];return[t(e,{id:a,operation:z,rows:u,offsets:p.offsets,neighbors:p.neighbors,pairMask:J}),C(e,{id:`${a}-pair-stats`,operation:z,variant:`pair-stats`,bindings:[...Z,{name:`pairCounts`,view:G,type:`u32`,access:`read_write`},{name:`pairSums`,view:K,type:`f32`,access:`read_write`}],invocationCount:W,declarations:`const ROWS: u32 = ${u}u;
const PAIR_STAT_BLOCKS: u32 = ${W}u;
${H(`transformDistance`,o.distanceTransform)}
${H(`transformTime`,o.timeTransform)}`,body:`var count = 0u;
  var sumDistance = 0.0;
  var sumTime = 0.0;
  for (var row = index; row < ROWS; row += PAIR_STAT_BLOCKS) {
    for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
      let other = neighbors[neighborsOffset + entry];
      if (pairMask[pairMaskOffset + entry] != 0u) {
        count++;
        sumDistance += transformDistance(distances[distancesOffset + entry]);
        sumTime += transformTime(abs(times[timesOffset + row] - times[timesOffset + other]));
      }
    }
  }
  pairCounts[pairCountsOffset + index] = count;
  pairSums[pairSumsOffset + 2u * index] = sumDistance;
  pairSums[pairSumsOffset + 2u * index + 1u] = sumTime;`}),C(e,{id:`${a}-shift`,operation:z,variant:`shift`,bindings:[{name:`pairCounts`,view:G,type:`u32`,access:`read`},{name:`pairSums`,view:K,type:`f32`,access:`read`},{name:`pairCount`,view:q,type:`u32`,access:`read_write`},N&&I?{name:`state`,view:I,type:`u32`,access:`read_write`}:{name:`shift`,view:Y,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const BLOCKS: u32 = ${W}u;`,body:`var count = 0u;
  var sumDistance = 0.0;
  var sumTime = 0.0;
  for (var block = 0u; block < BLOCKS; block++) {
    count += pairCounts[pairCountsOffset + block];
    sumDistance += pairSums[pairSumsOffset + 2u * block];
    sumTime += pairSums[pairSumsOffset + 2u * block + 1u];
  }
  pairCount[pairCountOffset] = count;
  let divisor = max(f32(count), 1.0);
${N?`state[stateOffset + ${r.shiftDistance}u] = bitcast<u32>(sumDistance / divisor);
  state[stateOffset + ${r.shiftTime}u] = bitcast<u32>(sumTime / divisor);`:`shift[shiftOffset] = sumDistance / divisor;
  shift[shiftOffset + 1u] = sumTime / divisor;`}`}),...N?re:[Q(),$()],C(e,{id:`${a}-summary`,operation:z,variant:`summary`,bindings:[{name:`parameters`,view:_,type:`u32`,access:`read`},{name:`statistics`,view:b,type:`f32`,access:`read`},{name:`pairCount`,view:q,type:`u32`,access:`read`},{name:`summary`,view:w,type:`f32`,access:`read_write`},...L,...o.output.permutationCount?[{name:`permutationsUsed`,view:o.output.permutationCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:U,body:`let observedPairs = f32(pairCount[pairCountOffset]);
  let timeClosePairs = 0.0;
  let expected = 0.0;
  ${ee(`f32`,M)}`}),...te(e,a,z,o.pairs,{statistics:o.output.statistics,summary:o.output.summary,permutationCount:o.output.permutationCount},{summary:12,permutationCount:1,statistics:o.maximumPermutations+1})]}},W=499,G=24,K=[.08,.04,.92,.22],q=7,J=k({parameters:R,caption:`Space and time differences among Kīlauea earthquakes, with a shuffled reference.`,createInputs:({random:e,world:t})=>N(e,t),build(e,t,n){let r=I(t.positions,n.radius),i=e.allocate(`times`,`float32`,t.count),a=e.allocate(`parameters`,`uint32`,4),o=e.allocate(`statistics`,`float32`,500),s=e.allocate(`summary`,`float32`,12);return e.graph.add(new U({pairs:A(e,r.weights),times:i.view,parameters:a.view,maximumPermutations:W,output:{statistics:o.view,summary:s.view}})),{outputs:{times:i,statistics:o,summary:s,graph:r,bins:e.allocate(`bins`,`float32`,G),positions:e.upload(`positions`,`float32x2`,t.drawnPositions)},update:e=>{i.write(L(t,e.clustering)),a.write(g({seed:e.seed,permutationCount:e.permutationCount}))}}},rebuildOn:[`radius`],interaction:{kind:`probe`,hint:`hover an event`,rest:[.5,.4],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:o}){let s=o(n.summary),c=s?.[a.observed]??NaN,l=s?.[a.pseudoPGreater],{glyphs:u}=j({histogram:n.bins,samples:o(n.statistics),start:1,sampleCount:r.permutationCount,observed:c,rectangle:K,world:i,pseudoP:l}),d=`Mantel r ${Number.isFinite(c)?c.toFixed(2):`…`} · p ${l===void 0?`…`:l.toFixed(3)}`,f=P(t,e);if(f<0)return{glyphs:u,readout:d};let p=F(t.drawnPositions,n.graph.weights,f,L(t,r.clustering),q);return{glyphs:[...u,...p.glyphs],readout:`${p.neighborCount} neighbors within ${r.radius} km, mean Δ ${p.meanDifference.toFixed(1)} days · ${d}`}}},draw:({outputs:e,theme:t,world:n})=>[{layer:`dots`,positions:e.positions,values:e.times,range:[0,120],radius:4},{layer:`chart`,kind:`bars`,values:e.bins,range:[0,1],rectangle:K,color:[...t.categories[0],.9]},{layer:`text`,...M({histogram:e.bins,rectangle:K,world:n,statistic:`Mantel r`,decimals:2})}]});export{J as default};
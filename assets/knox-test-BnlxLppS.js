import{_ as e,a as t,c as n,f as r,g as i,h as a,i as o,l as s,m as c,n as l,o as u,p as d,r as f,s as p,t as m,u as h,v as g}from"./space-time-kernels-a9nSiN0t.js";import{t as _}from"./spatial-weights-suppression-4G-LxXB_.js";import{t as v}from"./space-time-construction-Ck1ue1-P.js";import{a as y,i as ee,n as b,r as x}from"./trajectory-binding-validation-C89oJIAZ.js";import{t as S}from"./philox-random-DfVxWNPn.js";import{r as C}from"./kernel-C3tfQOn4.js";import{t as w}from"./gpu-sort-xRm8l5yn.js";import{o as T}from"./wgsl-kernel-nodes-DagJklH2.js";import{E,f as D,x as te,y as O}from"./gpu-contributor-utils-BIwLQON5.js";import{c as k}from"./spatial-weights-CqWysstN.js";import{i as A,t as j}from"./wgsl-common-DfyXiYB0.js";import{t as M}from"./define-run-B-brkCe4.js";import{m as N}from"./inputs-CJKKIRKK.js";import{n as P,t as F}from"./permutation-chart-dNDRNqOt.js";import{c as I,h as L,l as R,n as z,u as B}from"./point-field-CLZ0m0cx.js";import{Lt as V}from"./entry-preview-D70TdSbK.js";var H=`GPUKnoxTest`,U=class{id;props;rows;constructor(e){e=v(e,[`id`,`pairs`,`times`,`parameters`,`maximumPermutations`,`stopAfterExceedances`,`output`],`pairs`);let t=e.id??`knox-test`;if(this.id=t,this.props=e,this.rows=k(t,e.pairs,`pairs`),E(e.times,[`float32`],`${t} times`),E(e.parameters,[`uint32`],`${t} parameters`),E(e.output.statistics,[`uint32`],`${t} output.statistics`),E(e.output.summary,[`float32`],`${t} output.summary`),this.rows<2)throw Error(`${t} needs at least two rows`);if(this.rows>f)throw Error(`${t} supports at most ${f} rows`);if(e.times.length!==this.rows)throw Error(`${t} times needs one row per pairs row`);if(e.parameters.length<4)throw Error(`${t} parameters needs 4 words`);if(D(t,`maximumPermutations`,e.maximumPermutations,1,l),e.output.statistics.length<e.maximumPermutations+1)throw Error(`${t} output.statistics needs maximumPermutations + 1 rows`);if(a(t,e.stopAfterExceedances,e.maximumPermutations,e.output.permutationCount),e.output.summary.length<12)throw Error(`${t} output.summary needs 12 rows`);if(x(this.id,this.props.output),y(this.id,[{summary:e.output.summary,permutationCount:e.output.permutationCount}]),O(t,[e.output.statistics,e.output.summary,e.output.permutationCount],[e.times,e.parameters,e.pairs.offsets,e.pairs.neighbors,e.pairs.weights,e.pairs.status?.overflow,e.pairs.status?.candidateOverflow]),e.output.statistics.buffer===e.output.summary.buffer||e.output.permutationCount&&(e.output.permutationCount.buffer===e.output.statistics.buffer||e.output.permutationCount.buffer===e.output.summary.buffer))throw Error(`${t} outputs must use separate buffers`)}getInferenceMetadata(t){return e(`knox`,t,this.props.maximumPermutations,this.props.stopAfterExceedances)}getCommandNodes(e){b(e,this.id,this.props);let{id:i,props:a,rows:l}=this,{pairs:f,times:g,parameters:v,maximumPermutations:y}=a,{statistics:x,summary:E}=a.output;te(i,e,[f.offsets,f.neighbors,f.weights,g,v,x,E,...a.output.permutationCount?[a.output.permutationCount]:[]]);let[D,O]=u(l),k=y+1,M=n(D),N=p(l,y,D,O),P=(t,n)=>C(e,`${i}-${t}`,`uint32`,n),F=a.stopAfterExceedances,I=F!==void 0,L=c(k),R=I?L:k;ee(e,i,{"permutation partials":R*D*4,"row scratch":l*4,"pair mask":f.neighbors.length*4,...I?{"stopping state":32,"stopping dispatch":24}:{}});let z=I?r(e,i):void 0,B=z?[{name:`state`,view:z,type:`u32`,access:`read`}]:[],V=I?`\nconst BATCH_SIZE: u32 = ${L}u;`:``,U=P(`sort-keys`,l),W=P(`sort-indices`,l),G=P(`sorted-keys`,l),K=P(`sorted-indices`,l),q=P(`time-close`,2),J=P(`partials`,R*D),Y=P(`pair-partials`,D),X=P(`pair-mask`,f.neighbors.length),Z=`
${A}${j}`,Q=t=>T(e,{id:t?.id??`${i}-partials`,operation:H,variant:`partials`,bindings:[{name:`offsets`,view:f.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:f.neighbors,type:`u32`,access:`read`},{name:`pairMask`,view:X,type:`u32`,access:`read`},{name:`times`,view:g,type:`f32`,access:`read`},{name:`parameters`,view:v,type:`u32`,access:`read`},...B,{name:`partials`,view:J,type:`u32`,access:`read_write`},{name:`pairPartials`,view:Y,type:`u32`,access:`read_write`}],condition:t?.condition,extraResources:t?.extraResources,workgroupSize:64,invocationCount:R*M*64,guardIndex:!1,declarations:`${N}
${s(D,S.knoxTestShuffle.id)}
const SLOT_COUNT: u32 = ${R}u;${V}`,body:`${I?m:o}
  var count = 0u;
  var pairCount = 0u;
  if (runsSlot) {
    let threshold = readTimeThreshold();
    for (var row = block; row < ROWS; row += BLOCKS) {
      let time = times[timesOffset + getPermutedRow(row, slot)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let other = neighbors[neighborsOffset + entry];
        if (pairMask[pairMaskOffset + entry] != 0u) {
          pairCount++;
          let otherTime = times[timesOffset + getPermutedRow(other, slot)];
          if (abs(time - otherTime) <= threshold) {
            count++;
          }
        }
      }
    }
  }
  if (block < BLOCKS) {
    partials[partialsOffset + block * SLOT_COUNT + ${I?`localSlot`:`slot`}] = count;
  }
  if (slot == 0u && block < BLOCKS) {
    pairPartials[pairPartialsOffset + block] = pairCount;
  }`}),$=t=>T(e,{id:t?.id??`${i}-reduce`,operation:H,variant:`reduce`,bindings:[{name:`parameters`,view:v,type:`u32`,access:`read`},...B,{name:`partials`,view:J,type:`u32`,access:`read`},{name:`statistics`,view:x,type:`u32`,access:`read_write`}],condition:t?.condition,extraResources:t?.extraResources,...I?{workgroupSize:64,invocationCount:L}:{invocationCount:x.length},declarations:`${N}
const SLOT_COUNT: u32 = ${R}u;${V}`,body:I?`let slot = state[stateOffset] * BATCH_SIZE + index;
  if (slot <= readPermutationCount()) {
    var total = 0u;
    for (var block = 0u; block < BLOCKS; block++) {
      total += partials[partialsOffset + block * SLOT_COUNT + index];
    }
    statistics[statisticsOffset + slot] = total;
  }`:`var total = 0u;
  if (index <= readPermutationCount()) {
    for (var block = 0u; block < BLOCKS; block++) {
      total += partials[partialsOffset + block * SLOT_COUNT + index];
    }
  }
  statistics[statisticsOffset + index] = total;`}),ne=I&&z?d(e,{id:i,operation:H,stopAfterExceedances:F,slotCount:k,groupsPerSlot:M,workgroupSize:64,statisticType:`u32`,parameters:v,statistics:x,state:z,common:N,createPartialsNode:Q,createReduceNode:$}):[];return[T(e,{id:`${i}-keys`,operation:H,variant:`keys`,bindings:[{name:`times`,view:g,type:`f32`,access:`read`},{name:`sortKeys`,view:U,type:`u32`,access:`read_write`},{name:`sortIndices`,view:W,type:`u32`,access:`read_write`}],invocationCount:l,declarations:Z,body:`sortKeys[sortKeysOffset + index] = getOrderedKey(times[timesOffset + index]);
  sortIndices[sortIndicesOffset + index] = index;`}),...new w({id:`${i}-sort`,keys:U,values:W,outputKeys:G,outputValues:K}).getCommandNodes(e),T(e,{id:`${i}-clear-close`,operation:H,variant:`clear-close`,bindings:[{name:`timeClose`,view:q,type:`u32`,access:`read_write`}],invocationCount:2,body:`timeClose[timeCloseOffset + index] = 0u;`}),T(e,{id:`${i}-count-close`,operation:H,variant:`count-close`,bindings:[{name:`sortedKeys`,view:G,type:`u32`,access:`read`},{name:`parameters`,view:v,type:`u32`,access:`read`},{name:`timeClose`,view:q,type:`atomic<u32>`,access:`read_write`}],invocationCount:l,workgroupSize:256,guardIndex:!1,declarations:`${Z}
const ROWS: u32 = ${l}u;
var<workgroup> groupLow: atomic<u32>;
var<workgroup> groupCarry: atomic<u32>;`,body:`if (index < ROWS) {
    let threshold = bitcast<f32>(parameters[parametersOffset + 3u]);
    let time = decodeOrderedKey(sortedKeys[sortedKeysOffset + index]);
    var low = index + 1u;
    var high = ROWS;
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (abs(decodeOrderedKey(sortedKeys[sortedKeysOffset + middle]) - time) <= threshold) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let close = low - (index + 1u);
    if (close > 0u) {
      let previous = atomicAdd(&groupLow, close);
      if (previous + close < previous) {
        atomicAdd(&groupCarry, 1u);
      }
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let groupTotal = atomicLoad(&groupLow);
    var carries = atomicLoad(&groupCarry);
    if (groupTotal > 0u) {
      let previous = atomicAdd(&timeClose[timeCloseOffset], groupTotal);
      if (previous + groupTotal < previous) {
        carries++;
      }
    }
    if (carries > 0u) {
      atomicAdd(&timeClose[timeCloseOffset + 1u], carries);
    }
  }`}),t(e,{id:i,operation:H,rows:l,offsets:f.offsets,neighbors:f.neighbors,pairMask:X}),...I?ne:[Q(),$()],T(e,{id:`${i}-summary`,operation:H,variant:`summary`,bindings:[{name:`parameters`,view:v,type:`u32`,access:`read`},{name:`statistics`,view:x,type:`u32`,access:`read`},{name:`pairPartials`,view:Y,type:`u32`,access:`read`},{name:`timeClose`,view:q,type:`u32`,access:`read`},{name:`summary`,view:E,type:`f32`,access:`read_write`},...B,...a.output.permutationCount?[{name:`permutationsUsed`,view:a.output.permutationCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:N,body:`var pairTotal = 0u;
  for (var block = 0u; block < BLOCKS; block++) {
    pairTotal += pairPartials[pairPartialsOffset + block];
  }
  let observedPairs = f32(pairTotal);
  let timeClosePairs = f32(timeClose[timeCloseOffset + 1u]) * 4294967296.0 + f32(timeClose[timeCloseOffset]);
  let allPairs = f32(ROWS) * f32(ROWS - 1u) * 0.5;
  let expected = observedPairs * timeClosePairs / allPairs;
  ${h(`u32`,F)}`}),..._(e,i,H,a.pairs,{statistics:a.output.statistics,summary:a.output.summary,permutationCount:a.output.permutationCount},{summary:12,permutationCount:1,statistics:a.maximumPermutations+1})]}},W=499,G=24,K=[.08,.04,.92,.22],q=M({parameters:V,caption:`Space–time proximity of Kīlauea earthquakes against permuted origin times.`,createInputs:({random:e,world:t})=>I(e,t),build(e,t,n){let r=z(t.positions,n.radius),i=e.allocate(`times`,`float32`,t.count),a=e.allocate(`parameters`,`uint32`,4),o=e.allocate(`statistics`,`uint32`,500),s=e.allocate(`summary`,`float32`,12);return e.graph.add(new U({pairs:N(e,r.weights),times:i.view,parameters:a.view,maximumPermutations:W,output:{statistics:o.view,summary:s.view}})),{outputs:{times:i,summary:s,statistics:o,graph:r,bins:e.allocate(`bins`,`float32`,G),positions:e.upload(`positions`,`float32x2`,t.drawnPositions)},update:e=>{i.write(B(t,e.clustering)),a.write(g({seed:1,permutationCount:e.permutationCount,timeThreshold:e.timeThreshold}))}}},rebuildOn:[`radius`],interaction:{kind:`probe`,hint:`hover an event`,rest:[.5,.4],resolve(e,{inputs:t,outputs:n,values:r,world:a,peek:o}){let s=o(n.summary),c=o(n.statistics),l=s?.[i.observed]??NaN,u=s?.[i.pseudoPGreater],{glyphs:d}=P({histogram:n.bins,samples:c,start:1,sampleCount:r.permutationCount,observed:l,rectangle:K,world:a,pseudoP:u}),f=`Knox ${Number.isFinite(l)?l:`…`} · p ${u===void 0?`…`:u.toFixed(2)}`,p=L(t,e);if(p<0)return{glyphs:d,readout:f};let m=B(t,r.clustering),h=R(t.drawnPositions,n.graph.weights,p,m,r.timeThreshold);return{glyphs:[...d,...h.glyphs],readout:`${h.closeCount} of ${h.neighborCount} within ${r.timeThreshold} d and ${r.radius} km · ${f}`}}},draw:({outputs:e,theme:t,world:n})=>[{layer:`dots`,positions:e.positions,values:e.times,range:[0,120],radius:4},{layer:`chart`,kind:`bars`,values:e.bins,range:[0,1],rectangle:K,color:[...t.categories[0],.9]},{layer:`text`,...F({histogram:e.bins,rectangle:K,world:n,statistic:`Knox count`,decimals:0})}]});export{q as default};
import{_ as e,c as t,f as n,g as r,h as i,i as a,l as o,m as s,n as c,o as l,p as u,r as d,s as f,t as p,u as m,v as h}from"./space-time-kernels-a9nSiN0t.js";import{t as g}from"./spatial-weights-suppression-4G-LxXB_.js";import{t as _}from"./space-time-construction-Ck1ue1-P.js";import{a as v,n as y,r as b}from"./trajectory-binding-validation-C89oJIAZ.js";import{t as x}from"./philox-random-DfVxWNPn.js";import{r as S}from"./kernel-C3tfQOn4.js";import{t as ee}from"./gpu-sort-xRm8l5yn.js";import{o as C}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as w,f as T,v as E,x as D,y as O}from"./gpu-contributor-utils-BIwLQON5.js";import{c as k}from"./spatial-weights-CqWysstN.js";import{i as A,n as j}from"./space-time-bands-Lbr2KxkP.js";import{i as M}from"./wgsl-common-DfyXiYB0.js";import{t as N}from"./define-run-B-brkCe4.js";import{m as P,o as F}from"./inputs-CJKKIRKK.js";import{t as I}from"./pick-Bo3t3hjv.js";import{c as L,n as R,o as z,s as B}from"./events-CU2v4wkX.js";import{t as V}from"./permutation-chart-H7P4Tmsc.js";import{n as H,t as U}from"./panel-BovyVgLI.js";import{dn as W}from"./entry-preview-D70TdSbK.js";var G=`GPUJacquezTest`,K=class{id;props;rows;temporalContext;constructor(e){e=_(e,[`id`,`neighbors`,`times`,`temporalContext`,`parameters`,`k`,`maximumPermutations`,`stopAfterExceedances`,`output`],`neighbors`);let t=e.id??`jacquez-test`;if(this.id=t,this.props=e,this.rows=k(t,e.neighbors,`neighbors`),w(e.times,[`float32`],`${t} times`),w(e.parameters,[`uint32`],`${t} parameters`),w(e.output.statistics,[`uint32`],`${t} output.statistics`),w(e.output.summary,[`float32`],`${t} output.summary`),this.temporalContext=A(t,e.temporalContext),T(t,`k`,e.k,1,256),this.rows<=e.k)throw Error(`${t} needs more than k rows`);if(this.rows>d)throw Error(`${t} supports at most ${d} rows`);if(e.times.length!==this.rows)throw Error(`${t} times needs one row per neighbors row`);if(e.parameters.length<4)throw Error(`${t} parameters needs 4 words`);if(T(t,`maximumPermutations`,e.maximumPermutations,1,c),e.output.statistics.length<e.maximumPermutations+1)throw Error(`${t} output.statistics needs maximumPermutations + 1 rows`);if(i(t,e.stopAfterExceedances,e.maximumPermutations,e.output.permutationCount),e.output.summary.length<12)throw Error(`${t} output.summary needs 12 rows`);if(b(this.id,this.props.output),v(this.id,[{summary:e.output.summary,permutationCount:e.output.permutationCount}]),O(t,[e.output.statistics,e.output.summary,e.output.permutationCount],[e.times,e.parameters,e.neighbors.offsets,e.neighbors.neighbors,e.neighbors.weights,e.neighbors.status?.overflow,e.neighbors.status?.candidateOverflow]),e.output.statistics.buffer===e.output.summary.buffer||e.output.permutationCount&&(e.output.permutationCount.buffer===e.output.statistics.buffer||e.output.permutationCount.buffer===e.output.summary.buffer))throw Error(`${t} outputs must use separate buffers`)}getInferenceMetadata(t){return{...e(`knox`,t,this.props.maximumPermutations,this.props.stopAfterExceedances),testedRowsDefinition:`all listed (event, spatial neighbor) entries, counted once each`}}getCommandNodes(e){y(e,this.id,this.props);let{id:r,props:i,rows:c}=this,{neighbors:d,times:h,parameters:_,maximumPermutations:v,k:b}=i,{statistics:w,summary:T}=i.output;D(r,e,[d.offsets,d.neighbors,d.weights,h,_,w,T,...i.output.permutationCount?[i.output.permutationCount]:[]]),E(e.device,r,`event-ranks`,c*8,`reduce the rows`),E(e.device,r,`entry-mask`,d.neighbors.length*4,`reduce the list capacity`);let[O,k]=l(c),A=v+1,N=t(O),P=f(c,v,O,k),F=(t,n)=>S(e,`${r}-${t}`,`uint32`,n),I=i.stopAfterExceedances,L=I!==void 0,R=s(A),z=L?R:A,B=L?n(e,r):void 0,V=B?[{name:`state`,view:B,type:`u32`,access:`read`}]:[],H=L?`\nconst BATCH_SIZE: u32 = ${R}u;`:``,U=F(`sort-keys`,c),W=F(`sort-indices`,c),K=F(`sorted-keys`,c),q=F(`sorted-indices`,c),J=F(`event-ranks`,c*2),Y=F(`entry-mask`,d.neighbors.length),X=F(`partials`,z*O),Z=F(`pair-partials`,O),Q=t=>C(e,{id:t?.id??`${r}-partials`,operation:G,variant:`partials`,bindings:[{name:`offsets`,view:d.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:d.neighbors,type:`u32`,access:`read`},{name:`entryMask`,view:Y,type:`u32`,access:`read`},{name:`eventRanks`,view:J,type:`u32`,access:`read`},{name:`parameters`,view:_,type:`u32`,access:`read`},...V,{name:`partials`,view:X,type:`u32`,access:`read_write`},{name:`pairPartials`,view:Z,type:`u32`,access:`read_write`}],condition:t?.condition,extraResources:t?.extraResources,workgroupSize:64,invocationCount:z*N*64,guardIndex:!1,declarations:`${P}
${o(O,x.knoxTestShuffle.id)}
const SLOT_COUNT: u32 = ${z}u;
const NEAREST: u32 = ${b}u;${H}`,body:`${L?p:a}
  var count = 0u;
  var entryCount = 0u;
  if (runsSlot) {
    for (var row = block; row < ROWS; row += BLOCKS) {
      let start = eventRanks[eventRanksOffset + getPermutedRow(row, slot) * 2u + 1u];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        if (entryMask[entryMaskOffset + entry] != 0u) {
          entryCount++;
          let otherRank = eventRanks[eventRanksOffset + getPermutedRow(neighbors[neighborsOffset + entry], slot) * 2u];
          if (otherRank >= start && otherRank <= start + NEAREST) {
            count++;
          }
        }
      }
    }
  }
  if (block < BLOCKS) {
    partials[partialsOffset + block * SLOT_COUNT + ${L?`localSlot`:`slot`}] = count;
  }
  if (slot == 0u && block < BLOCKS) {
    pairPartials[pairPartialsOffset + block] = entryCount;
  }`}),$=t=>C(e,{id:t?.id??`${r}-reduce`,operation:G,variant:`reduce`,bindings:[{name:`parameters`,view:_,type:`u32`,access:`read`},...V,{name:`partials`,view:X,type:`u32`,access:`read`},{name:`statistics`,view:w,type:`u32`,access:`read_write`}],condition:t?.condition,extraResources:t?.extraResources,...L?{workgroupSize:64,invocationCount:R}:{invocationCount:w.length},declarations:`${P}
const SLOT_COUNT: u32 = ${z}u;${H}`,body:L?`let slot = state[stateOffset] * BATCH_SIZE + index;
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
  statistics[statisticsOffset + index] = total;`}),te=L&&B?u(e,{id:r,operation:G,stopAfterExceedances:I,slotCount:A,groupsPerSlot:N,workgroupSize:64,statisticType:`u32`,parameters:_,statistics:w,state:B,common:P,createPartialsNode:Q,createReduceNode:$}):[];return[C(e,{id:`${r}-keys`,operation:G,variant:`keys`,bindings:[{name:`times`,view:h,type:`f32`,access:`read`},{name:`sortKeys`,view:U,type:`u32`,access:`read_write`},{name:`sortIndices`,view:W,type:`u32`,access:`read_write`}],invocationCount:c,declarations:M,body:`sortKeys[sortKeysOffset + index] = getOrderedKey(times[timesOffset + index]);
  sortIndices[sortIndicesOffset + index] = index;`}),...new ee({id:`${r}-sort`,keys:U,values:W,outputKeys:K,outputValues:q}).getCommandNodes(e),C(e,{id:`${r}-rank`,operation:G,variant:`rank`,bindings:[{name:`sortedIndices`,view:q,type:`u32`,access:`read`},{name:`eventRanks`,view:J,type:`u32`,access:`read_write`}],invocationCount:c,body:`eventRanks[eventRanksOffset + sortedIndices[sortedIndicesOffset + index] * 2u] = index;`}),C(e,{id:`${r}-window`,operation:G,variant:`window`,bindings:[{name:`times`,view:h,type:`f32`,access:`read`},{name:`sortedIndices`,view:q,type:`u32`,access:`read`},{name:`eventRanks`,view:J,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`const ROWS: u32 = ${c}u;
const NEAREST: u32 = ${b}u;`,body:`let time = times[timesOffset + index];
  var low = eventRanks[eventRanksOffset + index * 2u];
  var high = low;
  for (var step = 0u; step < NEAREST; step++) {
    var takeLow = false;
    if (low == 0u) {
      takeLow = false;
    } else if (high + 1u == ROWS) {
      takeLow = true;
    } else {
      let lowGap = time - times[timesOffset + sortedIndices[sortedIndicesOffset + low - 1u]];
      let highGap = times[timesOffset + sortedIndices[sortedIndicesOffset + high + 1u]] - time;
      takeLow = lowGap <= highGap;
    }
    if (takeLow) {
      low--;
    } else {
      high++;
    }
  }
  eventRanks[eventRanksOffset + index * 2u + 1u] = low;`}),j(e,{id:r,operation:G,rows:c,offsets:d.offsets,neighbors:d.neighbors,entryMask:Y}),...L?te:[Q(),$()],C(e,{id:`${r}-summary`,operation:G,variant:`summary`,bindings:[{name:`parameters`,view:_,type:`u32`,access:`read`},{name:`statistics`,view:w,type:`u32`,access:`read`},{name:`pairPartials`,view:Z,type:`u32`,access:`read`},{name:`summary`,view:T,type:`f32`,access:`read_write`},...V,...i.output.permutationCount?[{name:`permutationsUsed`,view:i.output.permutationCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:P,body:`var pairTotal = 0u;
  for (var block = 0u; block < BLOCKS; block++) {
    pairTotal += pairPartials[pairPartialsOffset + block];
  }
  let observedPairs = f32(pairTotal);
  let timeClosePairs = f32(ROWS) * ${b}.0;
  let expected = observedPairs * ${b}.0 / f32(ROWS - 1u);
  ${m(`u32`,I)}`}),...g(e,r,G,i.neighbors,{statistics:i.output.statistics,summary:i.output.summary,permutationCount:i.output.permutationCount},{summary:12,permutationCount:1,statistics:i.maximumPermutations+1})]}},q=499,J=28,Y=[.06,.04,.94,.22];function X(e,t,n){let r=Array.from({length:e.length},(e,t)=>t).filter(e=>e!==t);return r.sort((n,r)=>Math.abs(e[n]-e[t])-Math.abs(e[r]-e[t])||n-r),new Set(r.slice(0,n))}var Z=N({parameters:W,caption:`Kīlauea summit earthquakes, May–August 2018, colored by day; bars are J under permuted times, the solid tick the observed J.`,createInputs:({world:e})=>({...R([0,.28,e.width,e.height]),worldWidth:e.width}),build(e,t,n){let r=F(t.positions,n.k),i=e.allocate(`times`,`float32`,t.count),a=e.allocate(`parameters`,`uint32`,4),o=e.allocate(`statistics`,`uint32`,500),s=e.allocate(`summary`,`float32`,12);return e.graph.add(new K({neighbors:P(e,r),times:i.view,parameters:a.view,k:n.k,maximumPermutations:q,output:{statistics:o.view,summary:s.view}})),{outputs:{positions:e.upload(`positions`,`float32x2`,t.positions),times:i,statistics:o,summary:s,weights:r,panel:H(e,Y,{width:t.worldWidth,height:1}),bins:e.allocate(`bins`,`float32`,J)},update:e=>{i.write(L(t.days,e.observedShare)),a.write(h({seed:3,permutationCount:e.permutationCount}))}}},rebuildOn:[`k`],buildCache:5,interaction:{kind:`probe`,hint:`hover an event`,rest:[.48,.3],resolve(e,{inputs:t,outputs:n,values:i,world:a,peek:o}){let s=o(n.summary),c=s?.[r.observed]??NaN,l=V({histogram:n.bins,samples:o(n.statistics),start:1,sampleCount:i.permutationCount,observed:c,rectangle:Y,world:a}),u=`J ${z(c,0)} vs ${z(s?.[r.permutationMean],0)} · p ${z(s?.[r.pseudoPGreater],3)}`;if(e.y<.26)return{glyphs:l.glyphs,readout:u};let d=I(t.index,e.x,e.y,t.focus.row);if(t.focus.row=d,d<0)return{glyphs:l.glyphs,readout:u};let f=X(L(t.days,i.observedShare),d,i.k),p=[t.positions[d*2],t.positions[d*2+1]],m=[...l.glyphs,{shape:`crosshair`,at:p,size:12}],h=0;for(let e of B(n.weights,d)){let n=f.has(e);n&&h++;let r=[t.positions[e*2],t.positions[e*2+1]];m.push({shape:`line`,from:p,to:r,dashed:!n},{shape:`dot`,at:r})}return{focus:d,glyphs:m,readout:`${h} of k ${i.k} also nearest in time · ${u}`}}},draw:({outputs:e,theme:t})=>[{layer:`dots`,positions:e.positions,values:e.times,range:[0,120],radius:3.5},U(e.panel,t),{layer:`chart`,kind:`bars`,values:e.bins,range:[0,1],rectangle:Y,color:[...t.categories[0],.9]}]});export{Z as default};
import{n as e}from"./neighbor-search-parameters-I_7-Ugqy.js";import{_ as t,l as n,n as r,o as i,r as a,s as o,v as s}from"./space-time-kernels-a9nSiN0t.js";import{n as c,r as l}from"./trajectory-binding-validation-C89oJIAZ.js";import{t as u}from"./philox-random-DfVxWNPn.js";import{r as d}from"./kernel-C3tfQOn4.js";import{o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as p,f as m,v as h,x as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{c as v}from"./spatial-weights-CqWysstN.js";import{i as y,n as b}from"./space-time-bands-Lbr2KxkP.js";import{t as x}from"./gpu-neighbor-search-SzuWK-my.js";import{t as S}from"./define-run-B-brkCe4.js";import{a as C}from"./pointer-overlay-Qh8qflcA.js";import{a as w,c as T,n as E,t as D}from"./events-CU2v4wkX.js";import{t as O}from"./kernel-CrpKP5IU.js";import{un as k}from"./entry-preview-D70TdSbK.js";var A=`GPULocalKnox`,j={observed:0,permutationMean:1,greaterCount:2,lesserCount:3,pseudoPGreater:4,pseudoPLesser:5},M=class{id;props;rows;temporalContext;constructor(e){let t=e.id??`local-knox`;if(this.id=t,this.props=e,this.rows=v(t,e.pairs,`pairs`),p(e.times,[`float32`],`${t} times`),p(e.parameters,[`uint32`],`${t} parameters`),p(e.output.summary,[`float32`],`${t} output.summary`),this.temporalContext=y(t,e.temporalContext),this.rows<2)throw Error(`${t} needs at least two rows`);if(this.rows>a)throw Error(`${t} supports at most ${a} rows`);if(e.times.length!==this.rows)throw Error(`${t} times needs one row per pairs row`);if(e.parameters.length<4)throw Error(`${t} parameters needs 4 words`);if(m(t,`maximumPermutations`,e.maximumPermutations,1,r),e.output.summary.length<this.rows*6)throw Error(`${t} output.summary needs rows * 6 rows`);l(this.id,this.props.output),_(t,[e.output.summary],[e.times,e.parameters,e.pairs.offsets,e.pairs.neighbors,e.pairs.weights])}getInferenceMetadata(e){return{...t(`knox`,e,this.props.maximumPermutations),testedRowsDefinition:`every event, against its listed spatial neighbors`}}getCommandNodes(e){c(e,this.id,this.props);let{id:t,props:r,rows:a}=this,{pairs:s,times:l,parameters:p,maximumPermutations:m}=r,{summary:_}=r.output;g(t,e,[s.offsets,s.neighbors,s.weights,l,p,_]),h(e.device,t,`accumulators`,a*12,`reduce the rows`),h(e.device,t,`entry-mask`,s.neighbors.length*4,`reduce the pair capacity`);let[v,y]=i(a,64),x=o(a,m,v,y),S=d(e,`${t}-entry-mask`,`uint32`,s.neighbors.length),C=d(e,`${t}-observed`,`uint32`,a),w=d(e,`${t}-accumulators`,`uint32`,a*3);return[b(e,{id:t,operation:A,rows:a,offsets:s.offsets,neighbors:s.neighbors,entryMask:S}),f(e,{id:`${t}-observed`,operation:A,variant:`observed`,bindings:[{name:`offsets`,view:s.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read`},{name:`entryMask`,view:S,type:`u32`,access:`read`},{name:`times`,view:l,type:`f32`,access:`read`},{name:`parameters`,view:p,type:`u32`,access:`read`},{name:`observed`,view:C,type:`u32`,access:`read_write`},{name:`accumulators`,view:w,type:`u32`,access:`read_write`}],invocationCount:a,declarations:x,body:`let threshold = readTimeThreshold();
  let time = times[timesOffset + index];
  var count = 0u;
  for (var entry = offsets[offsetsOffset + index]; entry < offsets[offsetsOffset + index + 1u]; entry++) {
    if (entryMask[entryMaskOffset + entry] != 0u
      && abs(time - times[timesOffset + neighbors[neighborsOffset + entry]]) <= threshold) {
      count++;
    }
  }
  observed[observedOffset + index] = count;
  accumulators[accumulatorsOffset + index * 3u] = 0u;
  accumulators[accumulatorsOffset + index * 3u + 1u] = 0u;
  accumulators[accumulatorsOffset + index * 3u + 2u] = 0u;`}),f(e,{id:`${t}-permutations`,operation:A,variant:`permutations`,bindings:[{name:`offsets`,view:s.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read`},{name:`entryMask`,view:S,type:`u32`,access:`read`},{name:`times`,view:l,type:`f32`,access:`read`},{name:`parameters`,view:p,type:`u32`,access:`read`},{name:`observed`,view:C,type:`u32`,access:`read`},{name:`accumulators`,view:w,type:`atomic<u32>`,access:`read_write`}],workgroupSize:64,invocationCount:m*64,guardIndex:!1,declarations:`${x}
${n(v,u.knoxTestShuffle.id)}`,body:`let slot = index / SPACE_TIME_WORKGROUP_SIZE + 1u;
  let runsSlot = slot <= readPermutationCount();
  if (runsSlot) {
    loadSwapOrNotRounds(readSeedKey(), slot, ROWS, localInvocationIndex, SPACE_TIME_WORKGROUP_SIZE);
  }
  workgroupBarrier();
  if (runsSlot) {
    let threshold = readTimeThreshold();
    for (var row = localInvocationIndex; row < ROWS; row += SPACE_TIME_WORKGROUP_SIZE) {
      let time = times[timesOffset + row];
      let focalImage = getPermutedRow(row, slot);
      var count = 0u;
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        if (entryMask[entryMaskOffset + entry] != 0u) {
          var image = getPermutedRow(neighbors[neighborsOffset + entry], slot);
          if (image == row) {
            image = focalImage;
          }
          if (abs(time - times[timesOffset + image]) <= threshold) {
            count++;
          }
        }
      }
      let observedCount = observed[observedOffset + row];
      if (count >= observedCount) {
        atomicAdd(&accumulators[accumulatorsOffset + row * 3u], 1u);
      }
      if (count <= observedCount) {
        atomicAdd(&accumulators[accumulatorsOffset + row * 3u + 1u], 1u);
      }
      atomicAdd(&accumulators[accumulatorsOffset + row * 3u + 2u], count);
    }
  }`}),f(e,{id:`${t}-summary`,operation:A,variant:`summary`,bindings:[{name:`parameters`,view:p,type:`u32`,access:`read`},{name:`observed`,view:C,type:`u32`,access:`read`},{name:`accumulators`,view:w,type:`u32`,access:`read`},{name:`summary`,view:_,type:`f32`,access:`read_write`}],invocationCount:a,declarations:x,body:`let permutationCount = f32(readPermutationCount());
  let rowOffset = summaryOffset + index * 6u;
  let greater = accumulators[accumulatorsOffset + index * 3u];
  let lesser = accumulators[accumulatorsOffset + index * 3u + 1u];
  summary[rowOffset + ${j.observed}u] = f32(observed[observedOffset + index]);
  summary[rowOffset + ${j.permutationMean}u] = f32(accumulators[accumulatorsOffset + index * 3u + 2u]) / permutationCount;
  summary[rowOffset + ${j.greaterCount}u] = f32(greater);
  summary[rowOffset + ${j.lesserCount}u] = f32(lesser);
  summary[rowOffset + ${j.pseudoPGreater}u] = f32(greater + 1u) / (permutationCount + 1.0);
  summary[rowOffset + ${j.pseudoPLesser}u] = f32(lesser + 1u) / (permutationCount + 1.0);`})]}},N=199,P=.05,F=12,I=S({parameters:k,caption:`Kīlauea summit earthquakes, May–August 2018: colored by local Knox count where it beats 199 time permutations (p ≤ 0.05, uncorrected).`,createInputs:({world:e})=>({...E([0,0,e.width,e.height]),width:e.width}),build(t,n){let r=n.count,i=t.upload(`positions`,`float32x2`,n.positions),a=t.allocate(`search-parameters`,`float32`,12),o=D(t,`pairs`,r,r*r);t.graph.add(new x({mode:`radius`,positions:i.view,gridSize:[24,16],parameters:a.view,output:o.port}));let c=t.allocate(`parameters`,`uint32`,4),l=t.allocate(`times`,`float32`,r),u=t.allocate(`summary`,`float32`,r*6);t.graph.add(new M({pairs:o.input,times:l.view,parameters:c.view,maximumPermutations:N,output:{summary:u.view}}));let d=t.allocate(`significant`,`float32`,r);return O(t,{id:`local-knox-significant`,invocationCount:r,bindings:[{name:`summary`,buffer:u,access:`read`},{name:`significant`,buffer:d,access:`read_write`}],body:`let row = summaryOffset + index * 6u;
  let observed = summary[row + ${j.observed}u];
  let pValue = summary[row + ${j.pseudoPGreater}u];
  var notANumberBits = 0x7fc00000u;
  significant[significantOffset + index] = select(bitcast<f32>(notANumberBits), observed, observed > 0.0 && pValue <= ${P});`}),{outputs:{positions:i,significant:d,summary:u},update:t=>{l.write(T(n.days,t.observedShare)),a.write(e({bounds:[0,0,n.width,1],radius:t.radius/n.kilometersPerUnit})),c.write(s({seed:7,permutationCount:N,timeThreshold:t.timeThreshold}))}}},interaction:C(k,{kind:`pad`,hint:`time across, radius up`,x:`timeThreshold`,y:`radius`,decorate:(e,t,{inputs:n,outputs:r,peek:i})=>{let a=e.radius/n.kilometersPerUnit,o=n.index.nearest(t.x,t.y),s=T(n.days,e.observedShare),c=[],l=0,u=0;if(o>=0){let t=[n.positions[o*2],n.positions[o*2+1]];for(let r of w(n.positions,o,a)){let i=Math.abs(s[r]-s[o])<=e.timeThreshold;u++,i&&(l++,l>F||c.push({shape:`line`,from:t,to:[n.positions[r*2],n.positions[r*2+1]]}))}c.push({shape:`ring`,at:t,radius:a,dashed:!0})}let d=i(r.significant);return{glyphs:c,readout:`${(d?Array.from(d).filter(Number.isFinite).length:void 0)??`…`} events p ≤ ${P} · ${e.radius} km, ${e.timeThreshold} d · here ${l}/${u}`}}}),draw:({outputs:e})=>[{layer:`dots`,context:!0,positions:e.positions,radius:2.5,alpha:.4},{layer:`dots`,positions:e.positions,values:e.significant,range:[1,7],logarithmic:!0,radius:4.5}]});export{I as default};
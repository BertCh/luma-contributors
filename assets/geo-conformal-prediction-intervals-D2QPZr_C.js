import{a as e,c as t,d as n,f as r,l as i,n as a,o,s,t as c,u as l}from"./conformal-Cc4u0mjo.js";import{i as u}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as f}from"./kernel-C3tfQOn4.js";import{o as p}from"./wgsl-kernel-nodes-DagJklH2.js";import{D as m,E as h,x as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{t as v}from"./regression-output-dispatches-KKVTWkHP.js";import{c as y,r as b}from"./wgsl-common-DfyXiYB0.js";import{t as x}from"./define-run-B-brkCe4.js";import{c as S}from"./pointer-overlay-Qh8qflcA.js";import{n as C,t as w}from"./pick-Bo3t3hjv.js";import{a as T}from"./surface-CdnX4cOV.js";import{$t as E}from"./entry-preview-D70TdSbK.js";var D=2**-20,O=class{props;id;queryRows;constructor(e){this.props=e,this.id=e.id??`geo-conformal-prediction-intervals`;let{id:t}=this;this.queryRows=e.query.predictions.length,r(t,e.calibration,e.query,1,this.queryRows),h(e.neighbors.offsets,[`uint32`],`${t} neighbors.offsets`),m(t,`neighbors.offsets`,e.neighbors.offsets,this.queryRows+1),h(e.neighbors.indices,[`uint32`],`${t} neighbors.indices`),h(e.neighbors.weights,[`float32`],`${t} neighbors.weights`),m(t,`neighbors.weights`,e.neighbors.weights,e.neighbors.indices.length),o(t,e.parameters,e.output,this.queryRows),v(this.id,e.output,[[`intervals`,`status`]]),_(t,Object.values(e.output),[...l(e.calibration,e.query),e.neighbors.offsets,e.neighbors.indices,e.neighbors.weights,e.parameters])}getCommandNodes(t){let{id:n,props:r,queryRows:a}=this,{neighbors:o}=r;d(t,n,{...Object.fromEntries(l(r.calibration,r.query).map((e,t)=>[`input ${t}`,e])),neighborOffsets:o.offsets,neighborIndices:o.indices,neighborWeights:o.weights,parameters:r.parameters,...r.output}),u(t.device,n,{"conformal calibration scores and keys":r.calibration.observations.length,"conformal neighbor scores and weights":o.indices.length}),g(n,t,[...l(r.calibration,r.query),o.offsets,o.indices,o.weights,r.parameters,...Object.values(r.output)]);let c=r.calibration.observations.length,m=i(t,n,r.calibration,1),h=o.indices.length,_=f(t,`${n}-entry-scores`,`float32`,h),v=f(t,`${n}-entry-weights`,`float32`,h);return[...m.nodes,p(t,{id:`${n}-entries`,operation:`GPUGeoConformalPredictionIntervals`,variant:`neighbor-scores`,invocationCount:h,bindings:[{name:`scoreKeys`,view:m.scoreKeys,type:`u32`,access:`read`},{name:`foldKeys`,view:m.views.foldKeys,type:`u32`,access:`read`},{name:`neighborIndices`,view:o.indices,type:`u32`,access:`read`},{name:`neighborWeights`,view:o.weights,type:`f32`,access:`read`},{name:`entryScores`,view:_,type:`f32`,access:`read_write`},{name:`entryWeights`,view:v,type:`f32`,access:`read_write`}],declarations:y,body:`let row = neighborIndices[neighborIndicesOffset + index];
  let weight = neighborWeights[neighborWeightsOffset + index];
  let usable = row < ${c}u && foldKeys[foldKeysOffset + min(row, ${c-1}u)] == 0u && isFiniteBits(weight) && weight > 0.0;
  entryScores[entryScoresOffset + index] = select(0.0, bitcast<f32>(scoreKeys[scoreKeysOffset + min(row, ${c-1}u)]), usable);
  entryWeights[entryWeightsOffset + index] = select(0.0, weight, usable);`}),p(t,{id:`${n}-intervals`,operation:`GPUGeoConformalPredictionIntervals`,variant:`weighted-quantile`,invocationCount:a,bindings:[{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`entryScores`,view:_,type:`f32`,access:`read`},{name:`entryWeights`,view:v,type:`f32`,access:`read`},{name:`neighborOffsets`,view:o.offsets,type:`u32`,access:`read`},{name:`queryPredictions`,view:r.query.predictions,type:`f32`,access:`read`},...r.query.scales?[{name:`queryScales`,view:r.query.scales,type:`f32`,access:`read`}]:[],{name:`intervals`,view:r.output.intervals,type:`f32`,access:`read_write`},{name:`status`,view:r.output.status,type:`u32`,access:`read_write`}],declarations:`${y}
${b}
${e}
${s}`,body:`let miscoverage = parameters[parametersOffset];
  let selfWeight = parameters[parametersOffset + 1u];
  let scale = ${r.query.scales?`queryScales[queryScalesOffset + index]`:`1.0`};
  let prediction = queryPredictions[queryPredictionsOffset + index];
  let first = neighborOffsets[neighborOffsetsOffset + index];
  let available = neighborOffsets[neighborOffsetsOffset + index + 1u] - first;
  let used = min(available, 256u);
  var code = select(0u, 16u, available > used);
  var total = selfWeight;
  var usable = 0u;
  var equalWeights = isFiniteBits(selfWeight) && selfWeight > 0.0;
  for (var entry = 0u; entry < used; entry++) {
    let weight = entryWeights[entryWeightsOffset + first + entry];
    total += weight;
    usable += u32(weight > 0.0);
    equalWeights = equalWeights && (weight <= 0.0 || weight == selfWeight);
  }
  var lower = getNaN();
  var upper = getNaN();
  if (!(isFiniteBits(prediction) && isFiniteBits(scale) && scale > 0.0)) {
    code |= 4u;
  } else if (usable == 0u) {
    code |= 8u;
  } else {
    // The smallest score whose cumulative weight reaches the level; ties count together.
    var level = 0.0;
    var equalWeightRank = 0u;
    if (equalWeights) {
      let ranks = getConformalPredictionRanks(usable, miscoverage);
      equalWeightRank = ranks.y;
    } else {
      level = (1.0 - miscoverage) * total * (1.0 - ${D});
    }
    var best = getInfinity();
    for (var candidate = 0u; candidate < used; candidate++) {
      if (entryWeights[entryWeightsOffset + first + candidate] <= 0.0) { continue; }
      let candidateScore = entryScores[entryScoresOffset + first + candidate];
      if (candidateScore >= best) { continue; }
      var cumulative = 0.0;
      var cumulativeCount = 0u;
      for (var other = 0u; other < used; other++) {
        if (entryScores[entryScoresOffset + first + other] <= candidateScore) {
          let weight = entryWeights[entryWeightsOffset + first + other];
          cumulative += weight;
          cumulativeCount += u32(weight > 0.0);
        }
      }
      if (select(cumulative >= level, cumulativeCount >= equalWeightRank, equalWeights)) {
        best = candidateScore;
      }
    }
    let width = scale * best;
    lower = prediction - width;
    upper = prediction + width;
    if (best >= getInfinity()) {
      code |= 3u;
    }
  }
  intervals[intervalsOffset + index * 2u] = lower;
  intervals[intervalsOffset + index * 2u + 1u] = upper;
  status[statusOffset + index] = code;`})]}},k=64,A=[0,3.5];function j({random:e,world:t}){let n=c(e,t),{calibration:r,test:i}=n,a=new Uint32Array(i.count*k),o=new Float32Array(i.count*k);for(let e=0;e<i.count;e++){let t=i.positions[e*2],n=i.positions[e*2+1];Array.from({length:r.count},(e,i)=>({other:i,distance:Math.hypot(r.positions[i*2]-t,r.positions[i*2+1]-n)})).sort((e,t)=>e.distance-t.distance).slice(0,k).forEach((t,n)=>{a[e*k+n]=t.other,o[e*k+n]=t.distance})}let s=e=>[Math.floor(Math.min(...e)*4)/4,Math.ceil(Math.max(...e)*4)/4],l=C(i.positions),u=Float32Array.from(n.calibration.observations,(e,t)=>Math.abs(e-n.calibration.predictions[t])/n.calibration.scales[t]).sort(),d=l.nearest(n.mapWidth*.5,t.height*.5);return{...n,sortedScores:u,nearestRows:a,nearestDistances:o,index:l,restRow:d,focus:{row:-1,key:``},predictionRange:s(i.predictions),valueRange:s(i.observations)}}function M(e,t){return Math.exp(-.5*(e/t)**2)}function N(e,t,n){let{panel:r,valueRange:i,predictionRange:a}=e,o=Math.min(Math.max(n,i[0]),i[1]);return[r.x0+(t-a[0])/(a[1]-a[0])*(r.x1-r.x0),r.y0+(o-i[0])/(i[1]-i[0])*(r.y1-r.y0)]}function P(e,t,n){return N(e,e.test.predictions[t],n)}var F=x({parameters:E,caption:`Localized conformal intervals of a Helene rainfall forecast; hover a station to see its weighted neighbours.`,createInputs:j,build(e,t){let{calibration:r,test:i}=t,a=e.allocate(`parameters`,`float32`,4),o=e.allocate(`offsets`,`uint32`,i.count+1),s=e.allocate(`indices`,`uint32`,i.count*k),c=e.allocate(`weights`,`float32`,i.count*k),l=e.allocate(`intervals`,`float32`,i.count*2);return e.graph.add(new O({calibration:{observations:e.upload(`calibration-observations`,`float32`,r.observations).view,predictions:e.upload(`calibration-predictions`,`float32`,r.predictions).view,scales:e.upload(`calibration-scales`,`float32`,r.scales).view},query:{predictions:e.upload(`test-predictions`,`float32`,i.predictions).view,scales:e.upload(`test-scales`,`float32`,i.scales).view},neighbors:{offsets:o.view,indices:s.view,weights:c.view},parameters:a.view,output:{intervals:l.view,status:e.allocate(`status`,`uint32`,i.count).view}})),{outputs:{intervals:l,calibrationPositions:e.upload(`calibration-positions`,`float32x2`,r.positions),testPositions:e.upload(`test-positions`,`float32x2`,i.positions),focusPositions:e.allocate(`focus-positions`,`float32x2`,k),focusWeights:e.allocate(`focus-weights`,`float32`,k)},update:e=>{let r=e.neighborCount,l=Uint32Array.from({length:i.count+1},(e,t)=>t*r),u=new Uint32Array(i.count*k),d=new Float32Array(i.count*k);for(let n=0;n<i.count;n++)for(let i=0;i<r;i++)u[n*r+i]=t.nearestRows[n*k+i],d[n*r+i]=M(t.nearestDistances[n*k+i],e.bandwidth);o.write(l),s.write(u),c.write(d),a.write(n({miscoverage:e.miscoverage}))}}},async afterRun({inputs:e,outputs:t,upload:n}){let{test:r}=e,i=await t.intervals.buffer.readAsync(),o=new Float32Array(i.buffer,i.byteOffset,r.count*2).slice(),s=[],c=[],l=new Uint32Array(r.count);for(let t=0;t<r.count;t++){let n=o[t*2],i=o[t*2+1];c.push(...N(e,r.predictions[t],r.observations[t])),Number.isNaN(n)||Number.isNaN(i)||(l[t]=+(r.observations[t]<n||r.observations[t]>i),Number.isFinite(n)&&Number.isFinite(i)&&s.push(...P(e,t,n),...P(e,t,i)))}return{widths:n(`widths`,`float32`,a(o,r.count)),bars:n(`bars`,`float32x4`,Float32Array.from(s.length>0?s:[0,0,0,0])),barCount:s.length/4,observed:n(`observed`,`float32x2`,Float32Array.from(c)),missed:n(`missed`,`uint32`,l)}},interaction:{kind:`probe`,hint:`hover a station`,rest:({inputs:e,toUnit:t})=>t([e.test.positions[e.restRow*2],e.test.positions[e.restRow*2+1]]),resolve(e,{inputs:n,outputs:r,values:i,build:a,peek:o}){let s=w(n.index,e.x,e.y,n.focus.row);if(s<0)return{};let{test:c,calibration:l}=n,u=i.neighborCount,d=c.positions[s*2],f=c.positions[s*2+1],p=`${s}|${u}|${i.bandwidth}|${a}`,m=[{shape:`ring`,at:[d,f],radius:.02}],h=new Float32Array(128),g=new Float32Array(k),_=0;for(let e=0;e<u;e++){let t=n.nearestRows[s*k+e],r=M(n.nearestDistances[s*k+e],i.bandwidth),a=[l.positions[t*2],l.positions[t*2+1]];h.set(a,e*2),g[e]=r,_+=r,r>.02&&m.push({shape:`line`,from:[d,f],to:a})}n.focus.key!==p&&(r.focusPositions.write(h),r.focusWeights.write(g),n.focus.key=p),n.focus.row=s,m.push(T(n.panel));let v=o(r.intervals),y=`station ${s} · k ${u} · weight ${S(_,1)}`;if(v){let e=v[s*2],r=v[s*2+1];if(m.push({shape:`ring`,at:N(n,c.predictions[s],c.observations[s]),radius:.015}),Number.isFinite(e)&&Number.isFinite(r)){let a=n.sortedScores,o=t(a.length,i.miscoverage).upperRank,l=o<=a.length?c.scales[s]*a[o-1]:NaN;y=`±${S((r-e)/2,2)} here, ±${S(l,2)} global · station ${s}`,m.push({shape:`line`,from:P(n,s,e),to:P(n,s,r)},{shape:`dot`,at:P(n,s,e)},{shape:`dot`,at:P(n,s,r)})}else y=`unbounded: too few neighbours · k ${u}`}return{focus:s,glyphs:m,readout:y}}},draw:({outputs:e,extra:t,values:n,interaction:r})=>[{layer:`dots`,context:!0,positions:e.calibrationPositions,radius:2.5,alpha:.5},...t?[{layer:`dots`,positions:e.testPositions,values:t.widths,ramp:`sequential`,range:A,radius:5},{layer:`lines`,context:!0,segments:t.bars,count:t.barCount,radius:1},{layer:`dots`,positions:t.observed,values:t.missed,categories:[[.55,.65,.8],[.95,.5,.15]],radius:2.5},...typeof r?.focus==`number`&&r.focus>=0?[{layer:`dots`,positions:e.focusPositions,values:e.focusWeights,count:n.neighborCount,ramp:`sequential`,range:[0,1],radius:5,sizeByValue:!0}]:[]]:[]]});export{F as default};
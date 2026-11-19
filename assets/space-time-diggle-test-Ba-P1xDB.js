import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-scan-B4NNTfF9.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,T as i,f as a,n as o,p as s,r as c,u as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{s as f,v as p}from"./spatial-context-DwZEd4EZ.js";import{t as m}from"./philox-random-DfVxWNPn.js";import{t as h}from"./permutation-random-SiwCiDNI.js";import{t as g}from"./permutation-shuffle-zu7WXW69.js";import{n as _}from"./temporal-context-xddVhAJD.js";import{n as v,r as y,t as b}from"./pair-histogram-COfc4qPq.js";import{l as x,t as S}from"./pair-statistics-grid-L8bKgCyF.js";import{a as C,i as w,n as T,o as E,r as D,t as O}from"./space-time-k-function-nodes-RnGxQh-J.js";import{t as k}from"./define-run-B-brkCe4.js";import{c as A,n as j,o as M}from"./events-CU2v4wkX.js";import{t as N}from"./kernel-CrpKP5IU.js";import{t as P}from"./permutation-chart-H7P4Tmsc.js";import{i as F,n as I,r as L,t as R}from"./table-CZ7xiVf4.js";import{n as z,t as B}from"./panel-BovyVgLI.js";import{cn as V}from"./entry-preview-D70TdSbK.js";var H=`GPUSpaceTimeDiggleTest`,U={observed:0,permutationMean:1,permutationVariance:2,greaterCount:3,lesserCount:4,pseudoPGreater:5,pseudoPLesser:6,pseudoPTwoSided:7,zScore:8,usedClasses:9,includedCount:10},W=160,G=40;function K(e){if(!Number.isSafeInteger(e)||e<0)throw Error(`getGPUSpaceTimeDiggleRandomParameterValues seed must be a non-negative safe integer`);return Uint32Array.from([e%2**32,Math.floor(e/2**32)])}var q=class{id;props;spatialContext;temporalContext;edgeCorrection;batchSize;estimatedCommandNodeCount;constructor(e){this.id=e.id??`space-time-diggle-test`,this.props=e;let{id:t}=this,{output:n}=e;this.spatialContext=p(t,e.spatialContext,{...f,measure:`area`}),this.temporalContext=e.temporalContext&&_(t,e.temporalContext,{encodings:[`float32-relative`]}),this.edgeCorrection=e.edgeCorrection??`isotropic`,i(t,[`none`,`isotropic`].includes(this.edgeCorrection),`edgeCorrection must be 'none' or 'isotropic'`);let l=x(t,{positions:e.positions,parameters:e.parameters,parameterLength:11,gridSize:e.gridSize,mask:e.mask});r(e.times,[`float32`],`${t} times`),i(t,e.times.length===l,`times length must equal positions length`),r(e.randomParameters,[`uint32`],`${t} randomParameters`),i(t,e.randomParameters.length>=2,`randomParameters must hold 2 uint32 words`),a(t,`distanceClassCount`,e.distanceClassCount,1,64),a(t,`timeClassCount`,e.timeClassCount,1,64),a(t,`permutationCount`,e.permutationCount,1,999);let u=e.distanceClassCount*e.timeClassCount;i(t,u*2<=2048&&(u+e.distanceClassCount)*3<=2048,`distanceClassCount * timeClassCount is too large for the pair histogram (${u} classes)`),this.batchSize=Math.min(e.permutationCount,Math.floor(b/2/u),Math.floor(o/4/l)),a(t,`permutation batch size for this row count`,this.batchSize,1);let m=Math.ceil(e.permutationCount/this.batchSize);this.estimatedCommandNodeCount=W+m*G,s(t,this.estimatedCommandNodeCount,c);let h={d:u,statistics:e.permutationCount+1,summary:11,permutationMean:u,permutationStandardDeviation:u,k:u,spatialK:e.distanceClassCount,temporalK:e.timeClassCount,status:1},g=[];for(let[e,a]of Object.entries(n))a&&(r(a,[e===`status`?`uint32`:`float32`],`${t} output.${e}`),i(t,a.length>=h[e],`output.${e} must hold ${h[e]} rows`),g.push(a));i(t,new Set(g.map(e=>e.buffer)).size===g.length,`outputs must use separate buffers`),d(t,g,[e.positions,e.times,e.mask,e.parameters,e.randomParameters])}getCommandNodes(r){let{id:i,props:o}=this,{output:c,distanceClassCount:d,timeClassCount:f,permutationCount:p}=o,_=o.positions.length;D(i,r.device,_),u(i,r,[o.positions,o.times,o.mask,o.parameters,o.randomParameters,...Object.values(c)]);let b=d*f,x=o.gridSize[0]*o.gridSize[1],T=this.edgeCorrection===`isotropic`,k=Math.min(this.batchSize,Math.floor(l(r.device,4)/_));a(i,`permutation batch size for this row count`,k,1),s(i,W+Math.ceil(p/k)*G);let A=(t,n)=>e(r,`${i}-${t}`,`float32`,Math.max(n,1)),j=c.k??A(`k`,b),M=c.spatialK??A(`spatial-k`,d),N=c.temporalK??A(`temporal-k`,f),P=O(r,{id:i,operation:H,positions:o.positions,times:o.times,mask:o.mask,parameters:o.parameters,gridSize:o.gridSize,distanceClassCount:d,timeClassCount:f,edgeCorrection:this.edgeCorrection,tieRule:`splancs`,k:j,spatialK:M,temporalK:N,d:c.d,status:c.status}),F=[...P.nodes],{inputs:I}=P,L=`
const DISTANCE_CLASS_COUNT: u32 = ${d}u;
const TIME_CLASS_COUNT: u32 = ${f}u;
const BINS: u32 = ${b}u;
const PERMUTATION_COUNT: u32 = ${p}u;
const PERMUTATION_ROW_COUNT: u32 = ${_}u;
const WEIGHT_SCALE: f32 = ${E}.0;
const WEIGHT_CAP: f32 = 100.0;
const EDGE_ISOTROPIC: bool = ${T};
`,R=e(r,`${i}-indexed-rows`,`uint32`,_),z=e(r,`${i}-row-ranks`,`uint32`,_),B=e(r,`${i}-rank-rows`,`uint32`,_);F.push(n(r,{id:`${i}-indexed-rows`,operation:H,variant:`indexed-rows`,bindings:[{name:`cellKeys`,view:I.cellKeys,type:`u32`,access:`read`},{name:`indexedRows`,view:R,type:`u32`,access:`read_write`}],invocationCount:_,declarations:`const CELL_COUNT: u32 = ${x}u;`,body:`indexedRows[indexedRowsOffset + index] = select(0u, 1u, cellKeys[cellKeysOffset + index] < CELL_COUNT);`}),...new t({id:`${i}-row-rank-scan`,input:R,output:z,mode:`exclusive`}).getCommandNodes(r),n(r,{id:`${i}-rank-rows`,operation:H,variant:`rank-rows`,bindings:[{name:`indexedRows`,view:R,type:`u32`,access:`read`},{name:`rowRanks`,view:z,type:`u32`,access:`read`},{name:`rankRows`,view:B,type:`u32`,access:`read_write`}],invocationCount:_,body:`if (indexedRows[indexedRowsOffset + index] != 0u) {
    rankRows[rankRowsOffset + rowRanks[rowRanksOffset + index]] = index;
  }`}));let V=A(`simulated-k`,p*b),U=Math.ceil(_/64);for(let e=0;e<p;e+=k){let t=Math.min(k,p-e),a=`${i}-batch-${e}`,s=`${L}
const BATCH_START: u32 = ${e}u;
const BATCH_SIZE: u32 = ${t}u;`,c=A(`batch-${e}-times`,t*_);F.push(n(r,{id:`${a}-times`,operation:H,variant:`permuted-times`,bindings:[{name:`times`,view:o.times,type:`f32`,access:`read`},{name:`indexedRows`,view:R,type:`u32`,access:`read`},{name:`rowRanks`,view:z,type:`u32`,access:`read`},{name:`rankRows`,view:B,type:`u32`,access:`read`},{name:`cellOffsets`,view:I.cellOffsets,type:`u32`,access:`read`},{name:`randomParameters`,view:o.randomParameters,type:`u32`,access:`read`},{name:`permutedTimes`,view:c,type:`f32`,access:`read_write`}],invocationCount:t*U*64,workgroupSize:64,guardIndex:!1,declarations:`${h}
${g(m.spaceTimeDiggleShuffle.id)}
${s}
const CELL_COUNT: u32 = ${x}u;
const GROUPS_PER_PERMUTATION: u32 = ${U}u;
${S}`,body:`let workgroup = index / 64u;
  let localPermutation = workgroup / GROUPS_PER_PERMUTATION;
  let row = (workgroup - localPermutation * GROUPS_PER_PERMUTATION) * 64u + localInvocationIndex;
  let includedCount = cellOffsets[cellOffsetsOffset + CELL_COUNT];
  let running = localPermutation < BATCH_SIZE;
  if (running) {
    loadSwapOrNotRounds(vec2<u32>(randomParameters[randomParametersOffset], randomParameters[randomParametersOffset + 1u]),
      BATCH_START + localPermutation + 1u, includedCount, localInvocationIndex, 64u);
  }
  workgroupBarrier();
  if (running && row < PERMUTATION_ROW_COUNT) {
    var value = getQuietNaN(row);
    if (indexedRows[indexedRowsOffset + row] != 0u) {
      value = times[timesOffset + rankRows[rankRowsOffset + getSwapOrNotPermutationIndex(rowRanks[rowRanksOffset + row], includedCount)]];
    }
    permutedTimes[permutedTimesOffset + localPermutation * PERMUTATION_ROW_COUNT + row] = value;
  }`}));let l=v(r,{id:`${a}-histogram`,operation:H,positions:o.positions,parameters:o.parameters,gridSize:o.gridSize,sortedRows:I.sortedRows,cellOffsets:I.cellOffsets,slotCount:t*b,channelCount:2,pairOrder:`ordered`,extraBindings:[{name:`permutedTimes`,view:c,type:`f32`,access:`read`}],declarations:`${s}
${w}${C}`,focusPrologue:`let timeStart = readParameter(7u);
    let timeEnd = readParameter(8u);
    let maximumDistance = readParameter(5u);
    let maximumTimeLag = readParameter(6u);
    let focusUsable = isFiniteValue(maximumDistance) && maximumDistance > 0.0 &&
      isFiniteValue(maximumTimeLag) && maximumTimeLag > 0.0;
    let left = x - lattice.minimumX;
    let bottom = y - lattice.minimumY;
    let right = lattice.maximumX - x;
    let top = lattice.maximumY - y;`,pairAction:`if (!focusUsable) { continue; }
            // The class of splancs iplace; the last class is open above (sj .ge. smax is dropped).
            let distanceClass = getSpaceTimeSplancsClass(pairDistance, maximumDistance, DISTANCE_CLASS_COUNT);
            if (distanceClass >= DISTANCE_CLASS_COUNT) { continue; }
            let spatialWeight = select(1.0, getSpaceTimeSpatialWeight(left, bottom, right, top, pairDistance), EDGE_ISOTROPIC);
            for (var permutation = 0u; permutation < BATCH_SIZE; permutation++) {
              let timeBase = permutedTimesOffset + permutation * PERMUTATION_ROW_COUNT;
              let focusTime = permutedTimes[timeBase + focus];
              let timeLag = abs(permutedTimes[timeBase + neighbor] - focusTime);
              let lagClass = getSpaceTimeSplancsClass(timeLag, maximumTimeLag, TIME_CLASS_COUNT);
              if (lagClass >= TIME_CLASS_COUNT) { continue; }
              let temporalWeight = select(1.0, getSpaceTimeSplancsTemporalWeight(timeLag, min(focusTime - timeStart, timeEnd - focusTime)), EDGE_ISOTROPIC);
              accumulateScaled((permutation * DISTANCE_CLASS_COUNT + distanceClass) * TIME_CLASS_COUNT + lagClass, 0u, 1u, spatialWeight * temporalWeight * WEIGHT_SCALE);
            }`});F.push(...l.nodes),F.push(n(r,{id:`${a}-finish`,operation:H,variant:`batch-finish`,bindings:[{name:`accumulators`,view:l.accumulators,type:`u32`,access:`read`},{name:`cellOffsets`,view:I.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:o.parameters,type:`f32`,access:`read`},{name:`simulatedK`,view:V,type:`f32`,access:`read_write`}],invocationCount:t*b,declarations:`${S}
${s}
const CELL_COUNT: u32 = ${x}u;
const LARGE_UNIT: f32 = 4096.0;
${y(`accumulators`,2,``,t*b)}`,body:`let includedCount = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let area = (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]);
  let duration = parameters[parametersOffset + 8u] - parameters[parametersOffset + 7u];
  let defined = includedCount >= 2.0 && isFiniteValue(area) && area > 0.0 &&
    isFiniteValue(duration) && duration > 0.0 && !readPairOverflow();
  let localPermutation = index / BINS;
  let bin = index % BINS;
  let distanceIndex = bin / TIME_CLASS_COUNT;
  let lagIndex = bin % TIME_CLASS_COUNT;
  var sum = 0.0;
  for (var firstIndex = 0u; firstIndex <= distanceIndex; firstIndex++) {
    for (var secondIndex = 0u; secondIndex <= lagIndex; secondIndex++) {
      let slot = localPermutation * BINS + firstIndex * TIME_CLASS_COUNT + secondIndex;
      sum += readPairAccumulator(slot, 0u) + readPairAccumulator(slot, 1u) * LARGE_UNIT;
    }
  }
  simulatedK[simulatedKOffset + (BATCH_START + localPermutation) * BINS + bin] =
    select(getQuietNaN(index), area * duration / (includedCount * (includedCount - 1.0)) * sum / WEIGHT_SCALE, defined);`}))}let K=`${S}
${L}`;if(c.permutationMean||c.permutationStandardDeviation){let e=c.permutationMean??A(`permutation-mean`,b),t=c.permutationStandardDeviation??A(`permutation-standard-deviation`,b);F.push(n(r,{id:`${i}-permutation-moments`,operation:H,variant:`permutation-moments`,bindings:[{name:`simulatedK`,view:V,type:`f32`,access:`read`},{name:`spatialK`,view:M,type:`f32`,access:`read`},{name:`temporalK`,view:N,type:`f32`,access:`read`},{name:`permutationMean`,view:e,type:`f32`,access:`read_write`},{name:`permutationStandardDeviation`,view:t,type:`f32`,access:`read_write`}],invocationCount:b,declarations:K,body:`// Capacity-bounded loops over the permutations (at most 999).
  var sum = 0.0;
  for (var permutation = 0u; permutation < PERMUTATION_COUNT; permutation++) {
    sum += simulatedK[simulatedKOffset + permutation * BINS + index];
  }
  let mean = sum / f32(PERMUTATION_COUNT);
  var squares = 0.0;
  for (var permutation = 0u; permutation < PERMUTATION_COUNT; permutation++) {
    let deviation = simulatedK[simulatedKOffset + permutation * BINS + index] - mean;
    squares += deviation * deviation;
  }
  let product = spatialK[spatialKOffset + index / TIME_CLASS_COUNT] * temporalK[temporalKOffset + index % TIME_CLASS_COUNT];
  permutationMean[permutationMeanOffset + index] = mean - product;
  permutationStandardDeviation[permutationStandardDeviationOffset + index] =
    select(getQuietNaN(index), sqrt(squares / f32(PERMUTATION_COUNT - 1u)), PERMUTATION_COUNT > 1u);`}))}return F.push(n(r,{id:`${i}-statistics`,operation:H,variant:`statistics`,bindings:[{name:`d`,view:c.d,type:`f32`,access:`read`},{name:`simulatedK`,view:V,type:`f32`,access:`read`},{name:`spatialK`,view:M,type:`f32`,access:`read`},{name:`temporalK`,view:N,type:`f32`,access:`read`},{name:`statistics`,view:c.statistics,type:`f32`,access:`read_write`}],invocationCount:p+1,declarations:K,body:`// Row 0 is the observed pattern, row p the permutation p - 1.
  var total = 0.0;
  var finiteClasses = 0u;
  for (var bin = 0u; bin < BINS; bin++) {
    var value = d[dOffset + bin];
    if (index > 0u) {
      let product = spatialK[spatialKOffset + bin / TIME_CLASS_COUNT] * temporalK[temporalKOffset + bin % TIME_CLASS_COUNT];
      value = simulatedK[simulatedKOffset + (index - 1u) * BINS + bin] - product;
    }
    if (isFiniteValue(value)) {
      total += value;
      finiteClasses++;
    }
  }
  // The sum is over every class; one that is not finite makes the statistic undefined.
  statistics[statisticsOffset + index] = select(getQuietNaN(index), total, finiteClasses == BINS);`}),n(r,{id:`${i}-summary`,operation:H,variant:`summary`,bindings:[{name:`statistics`,view:c.statistics,type:`f32`,access:`read`},{name:`d`,view:c.d,type:`f32`,access:`read`},{name:`cellOffsets`,view:I.cellOffsets,type:`u32`,access:`read`},{name:`summary`,view:c.summary,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${K}
const CELL_COUNT: u32 = ${x}u;`,body:`let observed = statistics[statisticsOffset];
  let defined = isFiniteValue(observed);
  var greater = 0u;
  var lesser = 0u;
  var sum = 0.0;
  for (var permutation = 1u; permutation <= PERMUTATION_COUNT; permutation++) {
    let value = statistics[statisticsOffset + permutation];
    if (value >= observed) { greater++; }
    if (value <= observed) { lesser++; }
    sum += value;
  }
  let count = f32(PERMUTATION_COUNT);
  let mean = sum / count;
  var squares = 0.0;
  for (var permutation = 1u; permutation <= PERMUTATION_COUNT; permutation++) {
    let deviation = statistics[statisticsOffset + permutation] - mean;
    squares += deviation * deviation;
  }
  let variance = select(0.0, squares / (count - 1.0), PERMUTATION_COUNT > 1u);
  var usedClasses = 0u;
  for (var bin = 0u; bin < BINS; bin++) {
    if (isFiniteValue(d[dOffset + bin])) { usedClasses++; }
  }
  let nan = getQuietNaN(0u);
  let pGreater = f32(greater + 1u) / (count + 1.0);
  let pLesser = f32(lesser + 1u) / (count + 1.0);
  summary[summaryOffset] = observed;
  summary[summaryOffset + 1u] = select(nan, mean, defined);
  summary[summaryOffset + 2u] = select(nan, variance, defined);
  summary[summaryOffset + 3u] = select(nan, f32(greater), defined);
  summary[summaryOffset + 4u] = select(nan, f32(lesser), defined);
  summary[summaryOffset + 5u] = select(nan, pGreater, defined);
  summary[summaryOffset + 6u] = select(nan, pLesser, defined);
  summary[summaryOffset + 7u] = select(nan, min(1.0, 2.0 * min(pGreater, pLesser)), defined);
  summary[summaryOffset + 8u] = select(nan, (observed - mean) / sqrt(variance), defined && variance > 0.0);
  summary[summaryOffset + 9u] = f32(usedClasses);
  summary[summaryOffset + 10u] = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);`})),F}},J=6,Y=99,X=121,Z=160,Q=24,$=[.08,.04,.92,.22];function ee({world:e}){let t=j([0,.26,e.width*.6,e.height]),n=[1/0,1/0,-1/0,-1/0];for(let e=0;e<t.count;e++){let r=t.positions[e*2],i=t.positions[e*2+1];n=[Math.min(n[0],r),Math.min(n[1],i),Math.max(n[2],r),Math.max(n[3],i)]}return{...t,worldWidth:e.width,bounds:n,table:R(J,J,[e.width*.6,.3,e.width*.96,.94])}}var te=k({parameters:V,caption:`Kīlauea summit earthquakes, 2018: standardized D(u, v) against 99 time permutations; rows distance (near at top), columns time lag; bars are U, the sum of the unstandardized D over all classes, under the permutations, the solid tick the observed U.`,createInputs:ee,build(e,t){let n=e.allocate(`times`,`float32`,t.count),r=e.allocate(`parameters`,`float32`,11),i=e.allocate(`d`,`float32`,36),a=e.allocate(`permutation-mean`,`float32`,36),o=e.allocate(`permutation-deviation`,`float32`,36),s=e.allocate(`statistics`,`float32`,100),c=e.allocate(`summary`,`float32`,11);e.graph.add(new q({positions:e.upload(`positions-input`,`float32x2`,t.positions).view,times:n.view,parameters:r.view,randomParameters:e.upload(`random-parameters`,`uint32`,K(11)).view,gridSize:[16,16],distanceClassCount:J,timeClassCount:J,permutationCount:Y,output:{d:i.view,statistics:s.view,summary:c.view,permutationMean:a.view,permutationStandardDeviation:o.view}}));let l=e.allocate(`residual`,`float32`,36);N(e,{id:`diggle-residual`,invocationCount:36,bindings:[{name:`d`,buffer:i,access:`read`},{name:`mean`,buffer:a,access:`read`},{name:`deviation`,buffer:o,access:`read`},{name:`residual`,buffer:l,access:`read_write`}],body:`let spread = deviation[deviationOffset + index];
  var notANumberBits = 0x7fc00000u;
  let value = (d[dOffset + index] - mean[meanOffset + index]) / spread;
  residual[residualOffset + index] = select(bitcast<f32>(notANumberBits), value, spread > 0.0);`});let u=F(e,t.table);return{outputs:{positions:e.upload(`positions`,`float32x2`,t.positions),residual:l,d:i,statistics:s,panel:z(e,$,{width:t.worldWidth,height:1}),bins:e.allocate(`bins`,`float32`,Q),summary:c,tablePositions:u.positions,tableIndices:u.indices,tableOwners:u.owners},update:e=>{n.write(A(t.days,e.observedShare)),r.write(T({bounds:t.bounds,maximumDistance:e.maximumDistance/t.kilometersPerUnit,maximumTimeLag:e.maximumTimeLag,timeWindow:[0,X]}))}}},interaction:{kind:`probe`,hint:`hover a table cell`,rest:[.66,.2],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let{table:o}=t,s=L(o,e.x,e.y),c=[{shape:`box`,...I(o,s.row,s.column)}],l=A(t.days,r.observedShare),u=r.maximumDistance/J/t.kilometersPerUnit,d=r.maximumTimeLag/J,f=(e,t)=>Math.max(0,Math.ceil(e/t)-1),p=0;for(let e=0;e<t.count;e++)for(let n=e+1;n<t.count;n++)f(Math.hypot(t.positions[e*2]-t.positions[n*2],t.positions[e*2+1]-t.positions[n*2+1]),u)===s.row&&f(Math.abs(l[e]-l[n]),d)===s.column&&(p++,p>Z||c.push({shape:`line`,from:[t.positions[e*2],t.positions[e*2+1]],to:[t.positions[n*2],t.positions[n*2+1]]}));let m=a(n.summary),h=P({histogram:n.bins,samples:a(n.statistics),start:1,sampleCount:Y,observed:m?.[U.observed]??NaN,rectangle:$,world:i});c.push(...h.glyphs);let g=a(n.residual)?.[s.index],_=(s.row+1)*r.maximumDistance/J,v=(s.column+1)*r.maximumTimeLag/J;return{focus:s.index,glyphs:c,readout:`z ${M(g,1)} at ${_.toFixed(1)} km, ${v.toFixed(1)} d · ${p} pairs · U (all classes) p ${M(m?.[U.pseudoPGreater],2)}`}}},draw:({inputs:e,outputs:t,theme:n})=>[B(t.panel,n),{layer:`chart`,kind:`bars`,values:t.bins,range:[0,1],rectangle:$,color:[...n.categories[0],.9]},{layer:`dots`,context:!0,positions:t.positions,radius:2.5},{layer:`polygons`,positions:t.tablePositions,indices:t.tableIndices,owners:t.tableOwners,values:t.residual,triangleCount:e.table.triangleCount,ramp:`diverging`,range:[-4,4]}]});export{te as default};
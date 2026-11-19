import{i as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{u as t}from"./kernel-C3tfQOn4.js";import{l as n,o as r}from"./gpu-command-graph-BdzdmV6u.js";import{s as i}from"./wgsl-common-DfyXiYB0.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,c as s,f as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{i as d,n as f,r as p}from"./pair-histogram-COfc4qPq.js";import{c as m,i as h,l as g,t as _,u as v}from"./pair-statistics-grid-L8bKgCyF.js";import{t as y}from"./pair-statistics-fixed-point-unit-DoZ07ePl.js";import{t as b}from"./pair-statistics-parameters-BSUD2FEq.js";import{s as x,v as S}from"./spatial-context-DwZEd4EZ.js";import{t as C}from"./status-aOwAAXek.js";import{t as w}from"./study-window-BK-B07m5.js";import{t as T}from"./pair-statistics-preflight-CaTMfN03.js";function E(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPUVariogram target must hold 8 elements`);let n=e.azimuthOffset??0;if(b(`GPUVariogram`,e.bounds,e.maximumDistance),!Number.isFinite(n))throw Error(`GPUVariogram azimuthOffset must be finite`);return t.set([...e.bounds===`study-window`?[NaN,NaN,NaN,NaN]:e.bounds,e.maximumDistance,n,0,0]),t}var D=`GPUVariogram`,O=Object.freeze({...x,measure:`distance`}),k=16777216,A=16777216,j=class{id;props;spatialContext;studyWindow;directionCount;constructor(i){this.id=i.id??`variogram`,this.props=i;let a=this.id;this.spatialContext=S(a,i.spatialContext,O),this.studyWindow=w(a,i.studyWindow,this.spatialContext),g(a,{...i,parameterLength:8});let{output:l}=i,f=i.directionCount??1;this.directionCount=f,c(a,`lagCount`,i.lagCount),c(a,`directionCount`,f);let p=i.lagCount*f;if(p>512)throw Error(`${a} lagCount * directionCount must be at most 512`);d(a,p,l.robustSemivariances?5:4),l.status?.overflow&&C(a,{overflow:l.status?.overflow});for(let[e,t]of[[`output.semivariances`,l.semivariances],[`output.meanDistances`,l.meanDistances],[`output.robustSemivariances`,l.robustSemivariances]])if(t&&(o(t,[`float32`],`${a} ${e}`),t.length<p))throw Error(`${a} ${e} must hold lagCount * directionCount rows`);if(l.pairCounts&&(t(l.pairCounts,`${a} output.pairCounts`),l.pairCounts.length<p))throw Error(`${a} output.pairCounts must hold lagCount * directionCount rows`);if(l.statistics&&(o(l.statistics,[`float32`],`${a} output.statistics`),l.statistics.length<5))throw Error(`${a} output.statistics must hold 5 rows`);{let t={semivariances:l.semivariances,pairCounts:l.pairCounts,meanDistances:l.meanDistances,robustSemivariances:l.robustSemivariances,statistics:l.statistics,overflow:l.status?.overflow},i=[[l.semivariances,l.pairCounts,l.meanDistances,l.robustSemivariances]],o=Object.values(t).flatMap(e=>e?s(e):[]).map(e=>new r(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset,byteStride:4,rowByteLength:4})),c=o.reduce((e,t)=>e+t.length,0);e(a,{outputs:new n({id:`${a}-writable-ranges`,name:`outputs`,format:`uint32`,length:c,valueLength:c,stride:1,byteStride:4,rowByteLength:4,data:o})});for(let t of i)e(a,Object.fromEntries(t.map((e,t)=>[String(t),e])))}u(a,[l.semivariances,l.pairCounts,l.meanDistances,l.robustSemivariances,l.statistics,l.status?.overflow],[i.positions,i.values,i.parameters,i.mask])}getCommandNodes(e){let{id:t,props:n,directionCount:r}=this,{positions:o,parameters:s,gridSize:c,lagCount:u,output:d}=n;v(t,e.device,o.length),l(t,e,[o,n.values,s,n.mask,d.semivariances,d.pairCounts,d.meanDistances,d.robustSemivariances,d.statistics,d.status?.overflow]);let g=o.length,b=u*r,x=!!d.robustSemivariances,S=x?4:3,C=S+1,w=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(e);w&&l(t,e,[w.edges,w.summary,w.slabs.params,w.slabs.slabOffsets,w.slabs.slabEntries,w.sampleCells,w.sampleCellCount,w.packed]),T(e,t,{rows:g,gridSize:c,slots:b,channels:C,moments:!0,unitVariance:!0,views:{positions:o,values:n.values,mask:n.mask,parameters:s,window:w?.packed,semivariances:d.semivariances,pairCounts:d.pairCounts,meanDistances:d.meanDistances,robustSemivariances:d.robustSemivariances,statistics:d.statistics,overflow:d.status?.overflow}});let E=w?m(e,{id:t,operation:D,parameters:s,parameterLength:s.length,window:w}):void 0,O=E?.parameters??s,j=h(e,{id:t,operation:D,positions:o,parameters:O,gridSize:c,values:n.values,mask:n.mask,studyWindow:w}),M=j.statistics,N=[...E?.nodes??[],...j.nodes],P=y(e,{id:t,operation:D,values:n.values,mask:n.mask,statistics:M});N.push(...P.nodes);let F=P.unitVariance,I=`
const MOMENTS: u32 = ${g}u;
const LAG_COUNT: u32 = ${u}u;
const DIRECTION_COUNT: u32 = ${r}u;
const FIXED_POINT_SCALE: f32 = ${k}.0;
const SEMIVARIANCE_UNITS_PER_VARIANCE: f32 = ${A}.0;
const LARGE_TERM_UNIT: f32 = 4096.0;
const PI: f32 = 3.14159265358979;

struct VariogramScales {
  semivariance: f32,
  robust: f32,
  distance: f32
}

fn getVariogramScales() -> VariogramScales {
  let valueRange = statistics[statisticsOffset + MOMENTS + 4u] - statistics[statisticsOffset + MOMENTS + 3u];
  var scales: VariogramScales;
  // The Matheron unit comes from the typical term (the robust variance), not from the value range.
  let unitVariance = unitVarianceView[unitVarianceViewOffset];
  scales.semivariance = select(0.0, SEMIVARIANCE_UNITS_PER_VARIANCE / unitVariance, unitVariance > 0.0);
  scales.robust = select(0.0, FIXED_POINT_SCALE / sqrt(valueRange), valueRange > 0.0);
  scales.distance = FIXED_POINT_SCALE / parameters[parametersOffset + 4u];
  return scales;
}`,L=f(e,{id:`${t}-histogram`,operation:D,positions:o,parameters:O,gridSize:c,sortedRows:j.sortedRows,cellOffsets:j.cellOffsets,slotCount:b,channelCount:C,pairOrder:`unordered`,extraBindings:[{name:`statistics`,view:M,type:`f32`,access:`read`},{name:`unitVarianceView`,view:F,type:`f32`,access:`read`}],declarations:`${I}\n${i}`,focusPrologue:`let scales = getVariogramScales();
    let focusValue = statistics[statisticsOffset + focus];
    let azimuthOffset = readParameter(5u);`,pairAction:`let lag = min(u32(pairDistance / lattice.maximumDistance * f32(LAG_COUNT)), LAG_COUNT - 1u);
            var sector = 0u;
            // A coincident pair has no direction and atan2(0, 0) is indeterminate in WGSL: it
            // stays in sector 0. Guard either sign of zero deltaX before reducing modulo pi.
            if (DIRECTION_COUNT > 1u && (deltaX != 0.0 || deltaY != 0.0)) {
              let angle = guardedAtan2(deltaY, deltaX) - azimuthOffset;
              let reduced = angle - floor(angle / PI) * PI;
              sector = min(u32(max(reduced, 0.0) / PI * f32(DIRECTION_COUNT)), DIRECTION_COUNT - 1u);
            }
            let bin = sector * LAG_COUNT + lag;
            let difference = statistics[statisticsOffset + neighbor] - focusValue;
            accumulate(bin, 0u, 1u);
            accumulateScaled(bin, 1u, ${S}u, 0.5 * difference * difference * scales.semivariance);
            accumulate(bin, 2u, quantizePairAmount(pairDistance * scales.distance));
            ${x?`accumulate(bin, 3u, quantizePairAmount(sqrt(abs(difference)) * scales.robust));`:``}`});N.push(...L.nodes);let R=[{name:`accumulators`,view:L.accumulators,type:`u32`,access:`read`},{name:`statistics`,view:M,type:`f32`,access:`read`},{name:`unitVarianceView`,view:F,type:`f32`,access:`read`},{name:`parameters`,view:O,type:`f32`,access:`read`},{name:`semivariances`,view:d.semivariances,type:`f32`,access:`read_write`}];return d.pairCounts&&R.push({name:`pairCounts`,view:d.pairCounts,type:`u32`,access:`read_write`}),d.meanDistances&&R.push({name:`meanDistances`,view:d.meanDistances,type:`f32`,access:`read_write`}),d.robustSemivariances&&R.push({name:`robustSemivariances`,view:d.robustSemivariances,type:`f32`,access:`read_write`}),N.push(a(e,{id:`${t}-finish`,operation:D,variant:`finish`,bindings:R,invocationCount:b,declarations:`${_}
${I}
${p(`accumulators`,C,``,b)}`,body:`let scales = getVariogramScales();
  let pairCount = readPairAccumulator(index, 0u);
  let hasPairs = pairCount > 0.0;
  let nan = getQuietNaN(index);
  // The large channel holds terms in units of LARGE_TERM_UNIT fixed-point units.
  let semivarianceSum = readPairAccumulator(index, 1u) + readPairAccumulator(index, ${S}u) * LARGE_TERM_UNIT;
  // A zero variance means every difference is 0 (scale 0, sum 0).
  let semivariance = select(0.0, semivarianceSum / scales.semivariance / pairCount, scales.semivariance > 0.0);
  semivariances[semivariancesOffset + index] = select(nan, semivariance, hasPairs);
  ${d.pairCounts?`pairCounts[pairCountsOffset + index] = readPairCount(index, 0u);`:``}
  ${d.meanDistances?`meanDistances[meanDistancesOffset + index] =
    select(nan, readPairAccumulator(index, 2u) / scales.distance / pairCount, hasPairs);`:``}
  ${x?`let robustMean = select(0.0, readPairAccumulator(index, 3u) / scales.robust / pairCount, scales.robust > 0.0);
  let robustSquare = robustMean * robustMean;
  // Cressie and Hawkins (1980) bias correction, with the 1 / N^2 term that matters in sparse bins.
  let robustCorrection = 0.457 + 0.494 / pairCount + 0.045 / (pairCount * pairCount);
  let robustValue = robustSquare * robustSquare / (2.0 * robustCorrection);
  robustSemivariances[robustSemivariancesOffset + index] = select(nan, robustValue, hasPairs);`:``}`})),d.status?.overflow&&N.push(a(e,{id:`${t}-overflow`,operation:D,variant:`overflow`,bindings:[{name:`accumulators`,view:L.accumulators,type:`u32`,access:`read`},{name:`overflow`,view:d.status?.overflow,type:`u32`,access:`read_write`}],invocationCount:1,declarations:p(`accumulators`,C,``,b),body:`var flagged = readPairOverflow();
  for (var bin = 0u; bin < ${b}u; bin++) {
    flagged = flagged || readPairCount(bin, 0u) == 0xffffffffu;
  }
  overflow[overflowOffset] = select(0u, 1u, flagged);`})),d.statistics&&N.push(a(e,{id:`${t}-statistics`,operation:D,variant:`statistics`,bindings:[{name:`statistics`,view:M,type:`f32`,access:`read`},{name:`statisticsOut`,view:d.statistics,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const MOMENTS: u32 = ${g}u;`,body:`let count = statistics[statisticsOffset + MOMENTS];
  statisticsOut[statisticsOutOffset] = count;
  statisticsOut[statisticsOutOffset + 1u] = statistics[statisticsOffset + MOMENTS + 1u];
  statisticsOut[statisticsOutOffset + 2u] = statistics[statisticsOffset + MOMENTS + 2u] / count;
  statisticsOut[statisticsOutOffset + 3u] = statistics[statisticsOffset + MOMENTS + 3u];
  statisticsOut[statisticsOutOffset + 4u] = statistics[statisticsOffset + MOMENTS + 4u];`})),N}};export{E as n,j as t};
import{i as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as t}from"./wgsl-common-DfyXiYB0.js";import{r as n,u as r}from"./kernel-C3tfQOn4.js";import{l as i,o as a}from"./gpu-command-graph-BdzdmV6u.js";import{o}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as s,c,f as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{i as f,n as p,r as m}from"./pair-histogram-COfc4qPq.js";import{t as h}from"./pair-statistics-fixed-point-unit-DoZ07ePl.js";import{t as g}from"./pair-statistics-preflight-CaTMfN03.js";import{c as _,i as v,l as y,s as b,t as x,u as ee}from"./pair-statistics-grid-L8bKgCyF.js";import{t as S}from"./pair-statistics-parameters-BSUD2FEq.js";import{s as C,v as w}from"./spatial-context-DwZEd4EZ.js";import{t as T}from"./status-aOwAAXek.js";import{t as E}from"./study-window-BK-B07m5.js";import{t as D}from"./define-run-B-brkCe4.js";import{c as O,i as k,l as A,n as j,o as M,s as N,t as P}from"./distance-probe-BxIfRi4a.js";import{a as F,h as I}from"./helpers-DPAd8TjN.js";import{dc as L}from"./entry-preview-D70TdSbK.js";var R=4294967295;function z(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPUSpatialCorrelogram target must hold 8 elements`);S(`GPUSpatialCorrelogram`,e.bounds,e.maximumDistance);let n=e.varianceAssumption??`randomization`;if(n!==`normality`&&n!==`randomization`)throw Error(`GPUSpatialCorrelogram varianceAssumption must be normality or randomization`);return t.set([...e.bounds===`study-window`?[NaN,NaN,NaN,NaN]:e.bounds,e.maximumDistance,+(n===`randomization`),0,0]),t}var B=`GPUSpatialCorrelogram`,V=Object.freeze({...C,measure:`distance`}),H=64,U=16777216,W=8,G=1e-4,K=4,q=class{id;props;spatialContext;studyWindow;bandMode;constructor(t){this.id=t.id??`spatial-correlogram`,this.props=t;let n=this.id;this.spatialContext=w(n,t.spatialContext,V),this.studyWindow=E(n,t.studyWindow,this.spatialContext),y(n,{...t,parameterLength:8});let o=t.bandMode??`cumulative`;if(o!==`cumulative`&&o!==`annulus`)throw Error(`${n} bandMode must be cumulative or annulus`);this.bandMode=o;let{bandCount:u,output:p}=t;l(n,`bandCount`,u,1,H),f(n,u,W),p.status?.overflow&&T(n,{overflow:p.status?.overflow});for(let[e,t]of[[`output.moransI`,p.moransI],[`output.zScores`,p.zScores],[`output.pValues`,p.pValues],[`output.expectedI`,p.expectedI],[`output.varianceI`,p.varianceI]])if(t&&(s(t,[`float32`],`${n} ${e}`),t.length<u))throw Error(`${n} ${e} must hold bandCount rows`);if(p.pairCounts&&(r(p.pairCounts,`${n} output.pairCounts`),p.pairCounts.length<u))throw Error(`${n} output.pairCounts must hold bandCount rows`);if(p.peakBands&&(r(p.peakBands,`${n} output.peakBands`),p.peakBands.length<2))throw Error(`${n} output.peakBands must hold 2 rows`);if(p.statistics&&(s(p.statistics,[`float32`],`${n} output.statistics`),p.statistics.length<5))throw Error(`${n} output.statistics must hold 5 rows`);{let t={moransI:p.moransI,zScores:p.zScores,pValues:p.pValues,expectedI:p.expectedI,varianceI:p.varianceI,pairCounts:p.pairCounts,peakBands:p.peakBands,statistics:p.statistics,overflow:p.status?.overflow},r=[[p.moransI,p.zScores,p.pValues,p.expectedI,p.varianceI,p.pairCounts]],o=Object.values(t).flatMap(e=>e?c(e):[]).map(e=>new a(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset,byteStride:4,rowByteLength:4})),s=o.reduce((e,t)=>e+t.length,0);e(n,{outputs:new i({id:`${n}-writable-ranges`,name:`outputs`,format:`uint32`,length:s,valueLength:s,stride:1,byteStride:4,rowByteLength:4,data:o})});for(let t of r)e(n,Object.fromEntries(t.map((e,t)=>[String(t),e])))}d(n,[p.moransI,p.zScores,p.pValues,p.expectedI,p.varianceI,p.pairCounts,p.peakBands,p.statistics,p.status?.overflow],[t.positions,t.values,t.parameters,t.mask])}getCommandNodes(e){let{id:t,props:r,bandMode:i}=this,{positions:a,parameters:s,gridSize:c,bandCount:l,output:d}=r;ee(t,e.device,a.length),u(t,e,[a,r.values,s,r.mask,d.moransI,d.zScores,d.pValues,d.expectedI,d.varianceI,d.pairCounts,d.peakBands,d.statistics,d.status?.overflow]);let f=a.length,y=i===`cumulative`,S=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(e);S&&u(t,e,[S.edges,S.summary,S.slabs.params,S.slabs.slabOffsets,S.slabs.slabEntries,S.sampleCells,S.sampleCellCount,S.packed]),g(e,t,{views:{positions:a,values:r.values,mask:r.mask,parameters:s,moransI:d.moransI,zScores:d.zScores,pValues:d.pValues,expectedI:d.expectedI,varianceI:d.varianceI,pairCounts:d.pairCounts,peakBands:d.peakBands,statistics:d.statistics,overflow:d.status?.overflow,studyWindow:S?.packed},rows:f,gridSize:c,slots:l,channels:W,moments:!0,unitVariance:!0,additionalScratch:{bandResults:l*K}});let C=S?_(e,{id:t,operation:B,parameters:s,parameterLength:s.length,window:S}):void 0,w=C?.parameters??s,T=v(e,{id:t,operation:B,positions:a,parameters:w,gridSize:c,values:r.values,mask:r.mask,studyWindow:S}),E=T.statistics,D=[...C?.nodes??[],...T.nodes],O=h(e,{id:t,operation:B,values:r.values,mask:r.mask,statistics:E});D.push(...O.nodes);let k=O.unitVariance,A=`const MOMENTS: u32 = ${f}u;
const BAND_COUNT: u32 = ${l}u;
const PRODUCT_UNITS_PER_VARIANCE: f32 = ${U}.0;
const LARGE_TERM_UNIT: f32 = 4096.0;

// Fixed-point units per unit of product, from the typical magnitude (the variance), not the maximum.
fn getProductScale() -> f32 {
  let unitVariance = unitVarianceView[unitVarianceViewOffset];
  return select(0.0, PRODUCT_UNITS_PER_VARIANCE / unitVariance, unitVariance > 0.0);
}`,j=n(e,`${t}-fourth-terms`,`float32`,f),M=n(e,`${t}-fourth-total`,`float32`,1);D.push(o(e,{id:`${t}-fourth-terms`,operation:B,variant:`fourth-terms`,bindings:[{name:`statistics`,view:E,type:`f32`,access:`read`},{name:`fourthTerms`,view:j,type:`f32`,access:`read_write`}],invocationCount:f,declarations:x,body:`let centered = statistics[statisticsOffset + index];
  let square = centered * centered;
  fourthTerms[fourthTermsOffset + index] = select(0.0, square * square, isFiniteValue(centered));`}),...b(e,{id:`${t}-fourth-sum`,operation:B,input:j,output:M}));let N=p(e,{id:`${t}-histogram`,operation:B,positions:a,parameters:w,gridSize:c,sortedRows:T.sortedRows,cellOffsets:T.cellOffsets,slotCount:l,channelCount:W,pairOrder:`ordered`,extraBindings:[{name:`statistics`,view:E,type:`f32`,access:`read`},{name:`unitVarianceView`,view:k,type:`f32`,access:`read`}],declarations:A,focusPrologue:`let productScale = getProductScale();
    let focusValue = statistics[statisticsOffset + focus];
    var degree: array<u32, ${l}>;`,pairAction:`let band = min(u32(pairDistance / lattice.maximumDistance * f32(BAND_COUNT)), BAND_COUNT - 1u);
            degree[band] += 1u;
            let product = focusValue * statistics[statisticsOffset + neighbor];
            accumulate(band, 0u, 1u);
            let positive = product >= 0.0;
            accumulateScaled(band, select(2u, 1u, positive), select(7u, 6u, positive), abs(product) * productScale);`,focusEpilogue:`var runningDegree = 0u;
    for (var band = 0u; band < BAND_COUNT; band++) {
      ${y?`runningDegree += degree[band];`:`runningDegree = degree[band];`}
      let high = runningDegree >> 12u;
      let low = runningDegree & 0xfffu;
      accumulate(band, 3u, high * high);
      accumulate(band, 4u, 2u * high * low);
      accumulate(band, 5u, low * low);
    }`});D.push(...N.nodes);let P=n(e,`${t}-band-results`,`float32`,l*K),F=n(e,`${t}-band-counts`,`uint32`,l);D.push(o(e,{id:`${t}-finish`,operation:B,variant:y?`finish-cumulative`:`finish-annulus`,bindings:[{name:`accumulators`,view:N.accumulators,type:`u32`,access:`read`},{name:`statistics`,view:E,type:`f32`,access:`read`},{name:`unitVarianceView`,view:k,type:`f32`,access:`read`},{name:`parameters`,view:w,type:`f32`,access:`read`},{name:`fourthTotal`,view:M,type:`f32`,access:`read`},{name:`bandResults`,view:P,type:`f32`,access:`read_write`},{name:`bandCounts`,view:F,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`${x}
${A}
const CHANNEL_COUNT: u32 = ${W}u;
const VARIANCE_CANCELLATION: f32 = ${G};
const FIRST_BAND_OFFSET: u32 = ${+!y}u;

// Exact 64-bit sum of one channel over annuli [first, last], as (low, high) words.
fn sumChannel(first: u32, last: u32, channel: u32) -> vec2<u32> {
  var low = 0u;
  var high = 0u;
  for (var band = first; band <= last; band++) {
    let word = (band * CHANNEL_COUNT + channel) * 2u;
    let addend = accumulators[accumulatorsOffset + word];
    let previous = low;
    low += addend;
    high += accumulators[accumulatorsOffset + word + 1u] + select(0u, 1u, low < previous);
  }
  return vec2<u32>(low, high);
}

fn toFloat(sum: vec2<u32>) -> f32 {
  return f32(sum.y) * 4294967296.0 + f32(sum.x);
}`,body:`let firstBand = select(0u, index, FIRST_BAND_OFFSET == 1u);
  let count = f32(statistics[statisticsOffset + MOMENTS]);
  let sumOfSquares = statistics[statisticsOffset + MOMENTS + 2u];
  let productScale = getProductScale();
  let orderedPairs = sumChannel(firstBand, index, 0u);
  // Unordered pairs = ordered / 2, saturated to 32 bits.
  bandCounts[bandCountsOffset + index] =
    select(0xffffffffu, (orderedPairs.y << 31u) | (orderedPairs.x >> 1u), orderedPairs.y < 2u);
  let weightSum = toFloat(orderedPairs);
  // The large channels hold products in units of LARGE_TERM_UNIT fixed-point units.
  let positiveSum = toFloat(sumChannel(firstBand, index, 1u)) + toFloat(sumChannel(firstBand, index, 6u)) * LARGE_TERM_UNIT;
  let negativeSum = toFloat(sumChannel(firstBand, index, 2u)) + toFloat(sumChannel(firstBand, index, 7u)) * LARGE_TERM_UNIT;
  let productSum = select(0.0, (positiveSum - negativeSum) / productScale, productScale > 0.0);
  // Squared neighbor counts are per band (already cumulative when the bands are).
  let degreeSquares = toFloat(sumChannel(index, index, 3u)) * 16777216.0 +
    toFloat(sumChannel(index, index, 4u)) * 4096.0 + toFloat(sumChannel(index, index, 5u));
  let nan = getQuietNaN(index);
  var moransI = nan;
  var expected = nan;
  var variance = nan;
  var zScore = nan;
  let randomization = parameters[parametersOffset + 5u] != 0.0;
  if (count >= 2.0) {
    expected = -1.0 / (count - 1.0);
  }
  if (weightSum > 0.0 && count >= 3.0 && sumOfSquares > 0.0) {
    moransI = count / weightSum * productSum / sumOfSquares;
    let s1 = 2.0 * weightSum;
    let s2 = 4.0 * degreeSquares;
    let weightSquare = weightSum * weightSum;
    let countSquare = count * count;
    // Magnitude of the summed terms: a variance below VARIANCE_CANCELLATION of it is f32
    // cancellation noise (for example a band holding every pair, whose true variance is 0).
    var magnitude = 0.0;
    if (!randomization) {
      let firstTerm = countSquare / (countSquare - 1.0) * (s1 / weightSquare);
      let secondTerm = count / (countSquare - 1.0) * (s2 / weightSquare);
      let thirdTerm = 3.0 / (countSquare - 1.0);
      variance = firstTerm - secondTerm + thirdTerm - expected * expected;
      magnitude = firstTerm + secondTerm + thirdTerm + expected * expected;
    } else if (count >= 4.0) {
      let kurtosis = count * fourthTotal[fourthTotalOffset] / (sumOfSquares * sumOfSquares);
      let denominator = (count - 1.0) * (count - 2.0) * (count - 3.0);
      let firstTerm = count * ((countSquare - 3.0 * count + 3.0) * (s1 / weightSquare) -
        count * (s2 / weightSquare) + 3.0);
      let secondTerm = kurtosis * ((countSquare - count) * (s1 / weightSquare) -
        2.0 * count * (s2 / weightSquare) + 6.0);
      variance = (firstTerm - secondTerm) / denominator - expected * expected;
      magnitude = (abs(firstTerm) + abs(secondTerm)) / denominator + expected * expected;
    }
    if (isFiniteValue(variance) && variance > VARIANCE_CANCELLATION * magnitude) {
      zScore = (moransI - expected) / sqrt(variance);
    } else {
      variance = nan;
    }
  }
  let base = index * ${K}u;
  bandResults[bandResultsOffset + base] = moransI;
  bandResults[bandResultsOffset + base + 1u] = expected;
  bandResults[bandResultsOffset + base + 2u] = variance;
  bandResults[bandResultsOffset + base + 3u] = zScore;`}));let I=[{name:`bandResults`,view:P,type:`f32`,access:`read`},{name:`moransI`,view:d.moransI,type:`f32`,access:`read_write`}],L=[`moransI[moransIOffset + index] = bandResults[bandResultsOffset + base];`];for(let[e,t,n]of[[`expectedI`,d.expectedI,1],[`varianceI`,d.varianceI,2],[`zScores`,d.zScores,3]])t&&(I.push({name:e,view:t,type:`f32`,access:`read_write`}),L.push(`${e}[${e}Offset + index] = bandResults[bandResultsOffset + base + ${n}u];`));return d.pValues&&(I.push({name:`pValues`,view:d.pValues,type:`f32`,access:`read_write`}),L.push(`let zScore = bandResults[bandResultsOffset + base + 3u];
  pValues[pValuesOffset + index] = select(getQuietNaN(index), getTwoSidedPValue(zScore), isFiniteValue(zScore));`)),d.pairCounts&&(I.push({name:`bandCounts`,view:F,type:`u32`,access:`read`},{name:`pairCounts`,view:d.pairCounts,type:`u32`,access:`read_write`}),L.push(`pairCounts[pairCountsOffset + index] = bandCounts[bandCountsOffset + index];`)),D.push(o(e,{id:`${t}-publish`,operation:B,variant:`publish`,bindings:I,invocationCount:l,declarations:`${x}
${J}`,body:`let base = index * ${K}u;
  ${L.join(`
  `)}`})),d.status?.overflow&&D.push(o(e,{id:`${t}-overflow`,operation:B,variant:`overflow`,bindings:[{name:`accumulators`,view:N.accumulators,type:`u32`,access:`read`},{name:`bandCounts`,view:F,type:`u32`,access:`read`},{name:`overflow`,view:d.status?.overflow,type:`u32`,access:`read_write`}],invocationCount:1,declarations:m(`accumulators`,W,``,l),body:`var flagged = readPairOverflow();
  for (var band = 0u; band < ${l}u; band++) {
    flagged = flagged || bandCounts[bandCountsOffset + band] == 0xffffffffu;
  }
  ${y?`for (var channelIndex = 0u; channelIndex < 4u; channelIndex++) {
    let channel = select(channelIndex + 4u, channelIndex + 1u, channelIndex < 2u);
    var low = 0u;
    var high = 0u;
    for (var band = 0u; band < ${l}u; band++) {
      let word = (band * ${W}u + channel) * 2u;
      let previousLow = low;
      low += accumulators[accumulatorsOffset + word];
      let withLow = high + select(0u, 1u, low < previousLow);
      flagged = flagged || withLow < high;
      let nextHigh = withLow + accumulators[accumulatorsOffset + word + 1u];
      flagged = flagged || nextHigh < withLow;
      high = nextHigh;
    }
  }`:``}
  overflow[overflowOffset] = select(0u, 1u, flagged);`})),d.peakBands&&D.push(o(e,{id:`${t}-peaks`,operation:B,variant:`peaks`,bindings:[{name:`bandResults`,view:P,type:`f32`,access:`read`},{name:`peakBands`,view:d.peakBands,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${x}
const BAND_COUNT: u32 = ${l}u;
const NO_BAND: u32 = ${R}u;

fn readZ(band: u32) -> f32 {
  return bandResults[bandResultsOffset + band * ${K}u + 3u];
}`,body:`var firstPeak = NO_BAND;
  var maximumBand = NO_BAND;
  var maximumZ = 0.0;
  for (var band = 0u; band < BAND_COUNT; band++) {
    let z = readZ(band);
    if (!isFiniteValue(z)) {
      continue;
    }
    if (maximumBand == NO_BAND || z > maximumZ) {
      maximumBand = band;
      maximumZ = z;
    }
    if (firstPeak == NO_BAND && band > 0u && band + 1u < BAND_COUNT) {
      let previous = readZ(band - 1u);
      let next = readZ(band + 1u);
      if (isFiniteValue(previous) && isFiniteValue(next) && z > previous && z > next) {
        firstPeak = band;
      }
    }
  }
  peakBands[peakBandsOffset] = firstPeak;
  peakBands[peakBandsOffset + 1u] = maximumBand;`})),d.statistics&&D.push(o(e,{id:`${t}-statistics`,operation:B,variant:`statistics`,bindings:[{name:`statistics`,view:E,type:`f32`,access:`read`},{name:`fourthTotal`,view:M,type:`f32`,access:`read`},{name:`statisticsOut`,view:d.statistics,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const MOMENTS: u32 = ${f}u;`,body:`let count = statistics[statisticsOffset + MOMENTS];
  let sumOfSquares = statistics[statisticsOffset + MOMENTS + 2u];
  statisticsOut[statisticsOutOffset] = count;
  statisticsOut[statisticsOutOffset + 1u] = statistics[statisticsOffset + MOMENTS + 1u];
  statisticsOut[statisticsOutOffset + 2u] = sumOfSquares / count;
  statisticsOut[statisticsOutOffset + 3u] = sumOfSquares;
  statisticsOut[statisticsOutOffset + 4u] = count * fourthTotal[fourthTotalOffset] / (sumOfSquares * sumOfSquares);`})),D}},J=`
${t}
const ANALYTIC_P_VALUE_FLOOR: f32 = 1.17549435e-38;

fn getTwoSidedPValue(z: f32) -> f32 {
  return clamp(erfcValue(abs(z) * 0.70710678118654752), ANALYTIC_P_VALUE_FLOOR, 1.0);
}
`,Y=600,X=P,Z=[-.1,.4],Q=[-2,12],$=300,te=D({parameters:L,caption:`Moran’s I (bars) and z-score (line) of log flash energy by distance band.`,createInputs:({random:e,world:t})=>({...F(e,t.width*.5,Y),world:t,anchor:{row:0}}),build(e,t,n){let r=e.upload(`positions`,`float32x2`,t.positions),i=e.allocate(`values`,`float32`,Y),a=e.allocate(`parameters`,`float32`,8),o=e.allocate(`morans-i`,`float32`,n.bandCount),s=e.allocate(`z-scores`,`float32`,n.bandCount);return e.graph.add(new q({positions:r.view,values:i.view,parameters:a.view,gridSize:[32,32],bandCount:n.bandCount,bandMode:n.bandMode===0?`cumulative`:`annulus`,output:{moransI:o.view,zScores:s.view}})),{outputs:{positions:r,observed:i,moransI:o,zScores:s,highlight:e.allocate(`highlight`,`float32x2`,$),zeroRule:e.upload(`zero-rule`,`float32x4`,k(X,t.world,Z,[0])),significanceRule:e.upload(`significance-rule`,`float32x4`,k(X,t.world,Q,[1.96]))},update:e=>{i.write(I(t,e.noise)),a.write(z({bounds:[0,0,t.windowWidth,1],maximumDistance:e.maximumDistance}))}}},rebuildOn:[`bandCount`,`bandMode`],interaction:{kind:`probe`,hint:`hover a distance band`,rest:[N(.3),.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=r.bandCount,o=A(M(e.unitX),a),s=r.maximumDistance/a,c=(o+1)*s,l=r.bandMode===1?o*s:0,u=t.anchor.row,d=Math.abs(t.positions[u*2+1]-e.y)*1.15;for(let n=0;n<t.count;n++){let r=Math.abs(t.positions[n*2+1]-e.y);r<d&&(u=n,d=r)}t.anchor.row=u;let f=[t.positions[u*2],t.positions[u*2+1]],p=new Float32Array($*2).fill(-10),m=j(t.positions,[u],c,p,l);n.highlight.write(p);let h=i(n.moransI)?.[o],g=i(n.zScores)?.[o];return{spotlight:null,focus:m,glyphs:[O(X,t.world,o,a),{shape:`ring`,at:f,radius:c},...l>0?[{shape:`ring`,at:f,radius:l}]:[]],readout:`band ${l.toFixed(2)}-${c.toFixed(2)} · I ${h===void 0?`…`:h.toFixed(2)} · z ${g===void 0?`…`:g.toFixed(1)}`}}},draw:({outputs:e,interaction:t})=>[{layer:`dots`,context:!0,positions:e.positions,values:e.observed,range:[0,10],radius:2.8},{layer:`dots`,positions:e.highlight,count:Math.max(1,typeof t.focus==`number`?t.focus:0),radius:3.6,color:[1,.75,.2,1]},{layer:`lines`,context:!0,segments:e.zeroRule,count:1,color:[.3,.75,1,.7]},{layer:`lines`,context:!0,segments:e.significanceRule,count:1,color:[1,.6,.2,.7]},{layer:`chart`,kind:`bars`,values:e.moransI,range:Z,rectangle:X,color:[.3,.75,1,.8]},{layer:`chart`,kind:`line`,values:e.zScores,range:Q,rectangle:X,radius:2,color:[1,.6,.2,1]}]});export{te as default};
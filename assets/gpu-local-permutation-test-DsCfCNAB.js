import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-reduction-tNgOVQ_C.js";import{o as r,t as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,f as o,p as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{r as d,s as f}from"./spatial-weights-CqWysstN.js";import{t as p}from"./spatial-statistics-row-dispatch-DNaskhxv.js";import{i as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as h,t as g}from"./gpu-multiple-testing-adjustment-Bghjpx33.js";import{a as _,c as v,d as y,i as b,l as x,n as S,o as C,r as ee,s as w,t as T,u as E}from"./permutation-inference-kernels-COguC6vP.js";import{t as D}from"./permutation-random-SiwCiDNI.js";import{i as O,n as k,r as A,t as j}from"./permutation-alternative-B1Iqm_lI.js";import{t as M}from"./philox-random-DfVxWNPn.js";import{t as N}from"./construction-snapshot-DdkX9Pc0.js";import{i as P}from"./weights-construction-snapshot-CikLt8kl.js";import{i as te,n as F,r as ne}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";function I(e,t){let n=N(e,t),r=N(n.output,[`analysisCount`,`exceedances`,`pseudoPValues`,`observed`,`significant`,`classes`,`quadrants`,`adjustedPValues`,`permutationCounts`,`islandCount`,`status`]);return Object.freeze({...n,weights:P(n.weights),output:Object.freeze({...r,status:N(r.status,[`overflow`,`invalidCount`])})})}var L=`GPULocalPermutationTest`,R=256,z=M.localPermutationTest.id,B=4294967295,V={undefined:0,highHigh:1,lowLow:2,negativeAssociation:3,notSignificant:4};function H(e,t){let n={bivariate:!1,joinCount:!1,centerY:!0,centerX:!1};switch(e){case`localMoran`:return{...n,centerX:!0,drawnA:`dx`,term:`fx * lagA`};case`localGeary`:return{...n,centerX:!0,drawnA:`(fx - dx) * (fx - dx)`,term:`lagA`};case`localG`:return{...n,drawnA:`dx`,term:`lagA`};case`localGStar`:return{...n,drawnA:`dx`,term:`lagA + fx`};case`localBivariateMoran`:return{...n,centerX:!0,bivariate:!0,drawnA:`dy`,term:`fx * lagA`};case`localLeeL`:return{...n,centerX:!0,bivariate:!0,drawnA:`dx`,drawnB:`dy`,term:`lagA * lagB`};case`localJoinCount`:return{...n,joinCount:!0,drawnA:`isOne(dx)`,term:`isOne(fx) * lagA`};case`localBivariateJoinCount`:return t===`BJC`?{...n,joinCount:!0,bivariate:!0,centerY:!1,drawnA:`(1.0 - isOne(dx)) * isOne(dy)`,term:`isOne(fx) * (1.0 - isOne(fy)) * lagA`}:{...n,joinCount:!0,bivariate:!0,centerY:!1,drawnA:`isOne(dx) * isOne(dy)`,term:`isOne(fx) * isOne(fy) * lagA`};default:throw Error(`local-permutation-test statistic ${String(e)} is not supported`)}}function re(e,t=`test`){let{constantsWGSL:n,accessWGSL:r,maximumPermutations:i,maximumNeighbors:a,stopAfter:o,alternative:s,spec:c,neighborLoopWGSL:l,slotWeightWGSL:u}=e,d=t===`replay`,f=!d&&o!==void 0,p=!d&&s===`folded`;return{declarations:`${n}
${r}
${v(i)}
${D}
${f?`const STOP_AFTER_EXCEEDANCES: u32 = ${o}u;\n`:``}const RANDOM_TAG: u32 = ${z}u;

// Value at a virtual shuffle position: a recorded swap, or the position itself.
fn readSwap(
  swapPositions: ptr<function, array<u32, ${a}>>,
  swapValues: ptr<function, array<u32, ${a}>>,
  swapCount: u32,
  position: u32
) -> u32 {
  for (var entry = 0u; entry < swapCount; entry++) {
    if ((*swapPositions)[entry] == position) {
      return (*swapValues)[entry];
    }
  }
  return position;
}`,body:`let observed = ${d?`bitcast<f32>(replayInfo[replayInfoOffset])`:`observedRaw[observedRawOffset + index]`};
  // The permutation count is never clamped: an invalid one leaves every row untested.
  // Join counts test only rows with at least one join (esda sets the other p-values to NaN).
  if (!isFiniteFloat(observed) || !isPermutationCountValid()${c.joinCount?` || observed == 0.0`:``}) {
    ${d?``:`exceedances[exceedancesOffset + index] = NOT_TESTED;${f?`
    exceedances[exceedancesOffset + ROWS + index] = NOT_TESTED;`:``}`}
    return;
  }
  let position = rowPositions[rowPositionsOffset + index];
  let others = rowPositions[rowPositionsOffset + ROWS] - 1u;
  let fx = getX(position);
  let fy = getY(position);
  let permutations = readValidPermutationCount();${f?`
  var permutationsRun = permutations;`:``}
  let key = readSeedKey();
  ${d?``:`var greater = 0u;
  var lesser = 0u;
  var folded = 0u;
  var total = 0.0;
  var mean = 0.0;`}
  // The row's active neighbor weights in slot order, read once: a permutation only reassigns
  // values to slots, so the per-draw work is the random draw and one value gather, not four
  // CSR gathers (offsets, neighbor, neighbor position, weight) per slot per permutation.
  var slotWeights: array<f32, ${a}>;
  var neighborTotal = 0u;
  ${l(`if (neighborTotal < MAXIMUM_NEIGHBORS) {
      slotWeights[neighborTotal] = ${u};
      neighborTotal++;
    }`)}
  // Islands (kept) and rows that cannot draw k distinct others are not tested.
  if (neighborTotal == 0u || neighborTotal > others) {
    ${d?``:`exceedances[exceedancesOffset + index] = NOT_TESTED;${f?`
    exceedances[exceedancesOffset + ROWS + index] = NOT_TESTED;`:``}`}
    return;
  }
  ${d?``:`// 'folded' needs the simulated mean first: sweep 0 sums, sweep 1 counts (same deterministic stream).
  for (var sweep = 0u; sweep < ${p?2:1}u; sweep++) {`}
  for (var permutation = 0u; permutation < permutations; permutation++) {
    var stream = createPhiloxStream(key, index, permutation, RANDOM_TAG);
    var swapPositions: array<u32, ${a}>;
    var swapValues: array<u32, ${a}>;
    var swapCount = 0u;
    var lagA = 0.0;
    var lagB = 0.0;
    for (var drawn = 0u; drawn < neighborTotal; drawn++) {
      // Partial Fisher-Yates over the other analysis rows' virtual positions [0, others).
      let pick = drawn + nextPhiloxBelow(&stream, others - drawn);
      let chosen = readSwap(&swapPositions, &swapValues, swapCount, pick);
      if (pick != drawn) {
        let displaced = readSwap(&swapPositions, &swapValues, swapCount, drawn);
        var entry = 0u;
        loop {
          if (entry == swapCount || swapPositions[entry] == pick) {
            break;
          }
          entry++;
        }
        swapPositions[entry] = pick;
        swapValues[entry] = displaced;
        swapCount = max(swapCount, entry + 1u);
      }
      let drawnPosition = chosen + select(0u, 1u, chosen >= position);
      let dx = getX(drawnPosition);
      let dy = getY(drawnPosition);
      lagA += slotWeights[drawn] * getDrawnA(fx, fy, dx, dy);
      ${c.drawnB?`lagB += slotWeights[drawn] * getDrawnB(fx, fy, dx, dy);`:``}
    }
    let simulated = getLocalTerm(fx, fy, lagA, lagB);
    ${d?`nullDistribution[nullDistributionOffset + permutation] = simulated;`:p?`if (sweep == 0u) {
      total += simulated;
    } else {
      folded += select(0u, 1u, abs(simulated - mean) >= abs(observed - mean));
    }`:`greater += select(0u, 1u, simulated >= observed);
    lesser += select(0u, 1u, simulated <= observed);${f?`
    // Besag-Clifford: stop at the first permutation whose exceedance count reaches the target.
    if (${k(s)} >= STOP_AFTER_EXCEEDANCES) {
      permutationsRun = permutation + 1u;
      break;
    }`:``}`}
  }
  ${d?`replayInfo[replayInfoOffset + 1u] = bitcast<u32>(f32(permutations));`:`mean = total / f32(permutations);
  }
  exceedances[exceedancesOffset + index] = ${k(s)};${f?`
  exceedances[exceedancesOffset + ROWS + index] = permutationsRun;`:``}`}`}}var ie=new WeakMap,ae=class{id;props;estimatedCommandNodeCount;constructor(e){e=I(e,[`id`,`weights`,`values`,`secondValues`,`mask`,`statistic`,`joinCase`,`alternative`,`islands`,`parameters`,`maximumPermutations`,`stopAfterExceedances`,`maximumNeighbors`,`extendedNeighborScratch`,`rowTileSize`,`multipleTesting`,`output`]);let n=e.id??`local-permutation-test`;this.id=n,this.props=e;let r=E({...e,analysisCount:e.output.analysisCount,islandCount:e.output.islandCount,id:n,operation:L});if(![`localMoran`,`localGeary`,`localG`,`localGStar`,`localBivariateMoran`,`localLeeL`,`localJoinCount`,`localBivariateJoinCount`].includes(e.statistic))throw Error(`${n} unknown statistic ${e.statistic}`);if(e.statistic===`localBivariateJoinCount`){if(e.joinCase!==`BJC`&&e.joinCase!==`CLC`)throw Error(`${n} localBivariateJoinCount requires joinCase 'BJC' or 'CLC'`)}else if(e.joinCase!==void 0)throw Error(`${n} joinCase applies to localBivariateJoinCount only`);let i=H(e.statistic,e.joinCase).bivariate;if(i&&!e.secondValues)throw Error(`${n} ${e.statistic} requires secondValues`);if(!i&&e.secondValues)throw Error(`${n} ${e.statistic} does not take secondValues`);if(e.output.classes){if(e.statistic!==`localGeary`)throw Error(`${n} classes applies to localGeary only`);t(e.output.classes,`${n} classes`)}if(e.output.quadrants){if(e.statistic!==`localBivariateMoran`)throw Error(`${n} quadrants applies to localBivariateMoran only`);t(e.output.quadrants,`${n} quadrants`)}if(O(n,e.stopAfterExceedances,A(n,e.alternative),e.maximumPermutations),e.output.permutationCounts){if(e.stopAfterExceedances===void 0)throw Error(`${n} permutationCounts requires stopAfterExceedances`);t(e.output.permutationCounts,`${n} permutationCounts`)}f(n,e.islands),e.rowTileSize!==void 0&&o(n,`rowTileSize`,e.rowTileSize);let c=F(e.weights),l=c.reduce((t,n)=>t+Math.ceil(n.rowCount/Math.min(e.rowTileSize??n.rowCount,n.rowCount)),0),d=ne(e.weights).length;this.estimatedCommandNodeCount=R+3*c.length+l+2*d,s(n,this.estimatedCommandNodeCount),o(n,`maximumNeighbors`,e.maximumNeighbors??32,1,e.extendedNeighborScratch?512:64),t(e.output.exceedances,`${n} exceedances`),a(e.output.pseudoPValues,[`float32`],`${n} pseudoPValues`),e.output.observed&&a(e.output.observed,[`float32`],`${n} observed`),e.output.significant&&t(e.output.significant,`${n} significant`),e.output.adjustedPValues&&a(e.output.adjustedPValues,[`float32`],`${n} adjustedPValues`);for(let[t,i]of[[`exceedances`,e.output.exceedances],[`permutationCounts`,e.output.permutationCounts],[`pseudoPValues`,e.output.pseudoPValues],[`observed`,e.output.observed],[`significant`,e.output.significant],[`adjustedPValues`,e.output.adjustedPValues],[`classes`,e.output.classes],[`quadrants`,e.output.quadrants]])if(i&&i.length!==r)throw Error(`${n} ${t} length must equal the weights row count`);let p=h(n,e.multipleTesting),{significant:g,adjustedPValues:_,classes:v}=e.output;if(p!==`none`&&!g&&!_&&!v)throw Error(`${n} multipleTesting requires significant or adjustedPValues`);t(e.output.status.overflow,`${n} output.status.overflow`),o(n,`output.status.overflow length`,e.output.status.overflow.length),e.output.status.invalidCount&&(t(e.output.status.invalidCount,`${n} output.status.invalidCount`),o(n,`output.status.invalidCount length`,e.output.status.invalidCount.length)),m(n,{...e.output,status:void 0,...e.output.status}),u(n,[e.output.exceedances,e.output.analysisCount,e.output.permutationCounts,e.output.pseudoPValues,e.output.observed,e.output.significant,e.output.adjustedPValues,e.output.classes,e.output.quadrants,e.output.islandCount,e.output.status.overflow,e.output.status.invalidCount],[...te(e.weights),e.values,e.secondValues,e.mask,e.parameters,e.weights.status?.overflow])}getParameterValues(e){return y(e,{maximumPermutations:this.props.maximumPermutations})}getInferenceMetadata(e){this.getParameterValues(e);let t=f(this.id,this.props.islands)===`keep`,n=this.props.maximumNeighbors??32;return{alternative:A(this.id,this.props.alternative),permutationCount:e.permutationCount,seed:e.seed,multipleTesting:h(this.id,this.props.multipleTesting),significanceLevel:e.significanceLevel??.05,testedRowsDefinition:`rows with a nonzero mask and a finite value that have 1 to ${n} positive-weight neighbors among the analysis rows (islands, rows without such a neighbor, `+(t?`are kept in n but not tested)`:`are excluded from the analysis)`)}}getCommandNodes(t){let{id:n,props:a}=this,{weights:o,parameters:s,statistic:u,maximumPermutations:m}=a;l(n,t,[...te(o),o.status?.overflow,a.values,a.secondValues,a.mask,s,a.output.exceedances,a.output.analysisCount,a.output.permutationCounts,a.output.pseudoPValues,a.output.observed,a.output.significant,a.output.adjustedPValues,a.output.classes,a.output.quadrants,a.output.islandCount,a.output.status.overflow,a.output.status.invalidCount]),x(t,{...a,id:n,operation:L},[a.output.exceedances,a.output.analysisCount,a.output.permutationCounts,a.output.pseudoPValues,a.output.observed,a.output.significant,a.output.adjustedPValues,a.output.classes,a.output.quadrants,a.output.islandCount,...Object.values(a.output.status)]),(H(u,a.joinCase).bivariate||a.stopAfterExceedances!==void 0)&&c(t.device,n,`two-column local inference scratch`,a.values.length*8,`reduce rows or request larger device bindings`);let _=a.values.length,y=A(n,a.alternative),T=j(y),E=a.stopAfterExceedances,D=E!==void 0,O=f(n,a.islands)===`keep`,k=h(n,a.multipleTesting),M=a.maximumNeighbors??32,N=H(u,a.joinCase),P=N.bivariate,F=w(t,{...a,analysisCount:a.output.analysisCount,islandCount:a.output.islandCount,id:`${n}-inputs`,operation:L,centerX:N.centerX,centerY:N.centerY}),I=F.nodes,{rowPositions:R,compactX:z,compactY:V,totals:ae}=F,U=F.chunks,W=U.length===1&&U[0].rowCount===_,oe=F.weights,G=e(t,`${n}-observed-raw`,`float32`,_),K=P?e(t,`${n}-pairs`,`float32`,2*_):void 0,q=K?{name:`pairs`,view:K,type:`f32`,access:`read`}:{name:`compactX`,view:z,type:`f32`,access:`read`},se=e=>`const ROWS: u32 = ${_}u;
${e===void 0?``:`const CAPACITY: u32 = ${e}u;\n`}const INVALID_POSITION: u32 = ${ee}u;
const MAXIMUM_NEIGHBORS: u32 = ${M}u;
const NOT_TESTED: u32 = ${B}u;
${S}
// A binary join-count variable: exactly 1 is a join, anything else is 0.
fn isOne(value: f32) -> f32 {
  return select(0.0, 1.0, value == 1.0);
}
// The statistic up to a positive per-row constant; identical for observed and simulated terms.
fn getLocalTerm(fx: f32, fy: f32, lagA: f32, lagB: f32) -> f32 {
  return ${N.term};
}
// Per-draw terms of the weighted lags, from the focus (fx, fy) and the drawn row (dx, dy).
fn getDrawnA(fx: f32, fy: f32, dx: f32, dy: f32) -> f32 {
  return ${N.drawnA};
}
fn getDrawnB(fx: f32, fy: f32, dx: f32, dy: f32) -> f32 {
  return ${N.drawnB??`0.0`};
}`,ce=se(U[0].weights.neighbors.length),le=se(),ue=K?`fn getX(position: u32) -> f32 { return pairs[pairsOffset + 2u * position]; }
fn getY(position: u32) -> f32 { return pairs[pairsOffset + 2u * position + 1u]; }`:`fn getX(position: u32) -> f32 { return compactX[compactXOffset + position]; }
fn getY(position: u32) -> f32 { return 0.0 * f32(position); }`,J=N.joinCount?`1.0`:`weight`,Y=e=>t=>d(`rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION`,t,`index`,e),de={constantsWGSL:ce,accessWGSL:ue,maximumPermutations:m,maximumNeighbors:M,stopAfter:E,alternative:y,spec:N,neighborLoopWGSL:Y(),slotWeightWGSL:J};W&&ie.set(this,{kernel:de,bindings:[{name:`offsets`,view:U[0].weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:U[0].weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:oe,type:`f32`,access:`read`},{name:`rowPositions`,view:R,type:`u32`,access:`read`},q,{name:`parameters`,view:s,type:`u32`,access:`read`}],observedRaw:G});let fe=ne(o),X=D?e(t,`${n}-permute-results`,`uint32`,2*_):void 0,Z=D?a.output.permutationCounts??e(t,`${n}-permutation-counts`,`uint32`,_):void 0;I.push(i(t,{id:`${n}-overflow-clear`,operation:L,view:a.output.status.overflow,type:`u32`,value:`0u`,componentCount:1}));for(let[e,i]of fe.entries())I.push(r(t,{id:`${n}-weights-overflow${e===0?``:`-${e}`}`,operation:L,variant:`weights-overflow`,bindings:[{name:`weightsOverflow`,view:i,type:`u32`,access:`read`},{name:`overflow`,view:a.output.status.overflow,type:`atomic<u32>`,access:`read_write`}],invocationCount:1,body:`if (weightsOverflow[weightsOverflowOffset] != 0u) {
    atomicMax(&overflow[overflowOffset], 1u);
  }`}));K&&I.push(r(t,{id:`${n}-pairs`,operation:L,variant:`pairs`,bindings:[{name:`compactX`,view:z,type:`f32`,access:`read`},{name:`compactY`,view:V,type:`f32`,access:`read`},{name:`pairs`,view:K,type:`f32`,access:`read_write`}],invocationCount:_,body:`pairs[pairsOffset + 2u * index] = compactX[compactXOffset + index];
  pairs[pairsOffset + 2u * index + 1u] = compactY[compactYOffset + index];`})),I.push(r(t,{id:`${n}-count-validity`,operation:L,variant:`count-validity`,bindings:[{name:`parameters`,view:s,type:`u32`,access:`read`},{name:`overflow`,view:a.output.status.overflow,type:`atomic<u32>`,access:`read_write`}],invocationCount:1,declarations:v(m),body:`if (!isPermutationCountValid()) {
    atomicMax(&overflow[overflowOffset], 1u);
  }`}),...U.map((e,r)=>{let i=W?``:`-${r}`,o=le,s=Y(`WEIGHTS_ROW_START`),c=e=>e;return p(t,{rowBase:e.rowStart,id:`${n}-observed${i}`,operation:L,variant:`observed`,bindings:[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:e.effective,type:`f32`,access:`read`},{name:`rowPositions`,view:R,type:`u32`,access:`read`},q,{name:`observedRaw`,view:G,type:`f32`,access:`read_write`},{name:`overflow`,view:a.output.status.overflow,type:`atomic<u32>`,access:`read_write`},...a.output.quadrants?[{name:`quadrants`,view:a.output.quadrants,type:`u32`,access:`read_write`}]:[]],invocationCount:W?_:e.rowCount,declarations:`${o}
${ue}`,body:c(`var raw = getQuietNaN(index);
  var quadrant = 0u;
  let position = rowPositions[rowPositionsOffset + index];
  if (position != INVALID_POSITION) {
    let fx = getX(position);
    let fy = getY(position);
    var lagA = 0.0;
    var lagB = 0.0;
    var neighborCount = 0u;
    ${s(`let neighborPosition = rowPositions[rowPositionsOffset + neighbor];
    let dx = getX(neighborPosition);
    let dy = getY(neighborPosition);
    lagA += ${J} * getDrawnA(fx, fy, dx, dy);
    ${N.drawnB?`lagB += ${J} * getDrawnB(fx, fy, dx, dy);`:``}
    neighborCount++;`)}
    if (neighborCount > MAXIMUM_NEIGHBORS) {
      atomicMax(&overflow[overflowOffset], 1u);
    } else if (neighborCount > 0u || ${O}) {
      // A kept island has a zero lag: the formula value, never tested.
      raw = getLocalTerm(fx, fy, lagA, lagB);
      // esda Moran_Local_BV.q (PySAL scheme): 1 HH, 2 LH, 3 LL, 4 HL in (x_i, lag of y).
      quadrant = select(select(3u, 4u, fx > 0.0), select(2u, 1u, fx > 0.0), lagA > 0.0);
    }
  }
  ${a.output.quadrants?`quadrants[quadrantsOffset + index] = quadrant;`:``}
  observedRaw[observedRawOffset + index] = raw;`)})}));for(let[e,r]of U.entries()){let i=re({...de,constantsWGSL:le,neighborLoopWGSL:Y(`WEIGHTS_ROW_START`)}),o=r.rowCount,c=Math.max(1,Math.min(a.rowTileSize??o,o));for(let l=0;l<o;l+=c){let u=Math.min(c,o-l),d=r.rowStart+l,f=`${W?``:`-chunk${e}`}${c<o?`-${l}`:``}`;I.push(p(t,{rowBase:d,weightsRowStart:r.rowStart,id:`${n}-permute${f}`,operation:L,variant:`permute`,bindings:[{name:`offsets`,view:r.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:r.effective,type:`f32`,access:`read`},{name:`rowPositions`,view:R,type:`u32`,access:`read`},q,{name:`parameters`,view:s,type:`u32`,access:`read`},{name:`observedRaw`,view:G,type:`f32`,access:`read`},{name:`exceedances`,view:X??a.output.exceedances,type:`u32`,access:`read_write`}],invocationCount:u,...i}))}}X&&Z&&I.push(r(t,{id:`${n}-split-counts`,operation:L,variant:`split-counts`,bindings:[{name:`permuteResults`,view:X,type:`u32`,access:`read`},{name:`exceedances`,view:a.output.exceedances,type:`u32`,access:`read_write`},{name:`permutationCounts`,view:Z,type:`u32`,access:`read_write`}],invocationCount:_,declarations:`const ROWS: u32 = ${_}u;`,body:`exceedances[exceedancesOffset + index] = permuteResults[permuteResultsOffset + index];
  permutationCounts[permutationCountsOffset + index] = permuteResults[permuteResultsOffset + ROWS + index];`}));let Q=a.output.observed??(a.output.classes?e(t,`${n}-observed-geary`,`float32`,_):void 0);if(Q){let e=`rowPositions[rowPositionsOffset + ROWS]`,i=`totals[totalsOffset + ${b.sumSquares}u]`,a=`totals[totalsOffset + ${b.sumSquaresY}u]`,o={localMoran:`multiplyBySpatialCount(raw, max(${e}, 1u) - 1u) / ${i}`,localGeary:`multiplyBySpatialCount(raw, ${e}) / ${i}`,localG:`raw / (totals[totalsOffset + 1u] - compactX[compactXOffset + position])`,localGStar:`raw / totals[totalsOffset + 1u]`,localBivariateMoran:`multiplyBySpatialCount(raw, max(${e}, 1u) - 1u) / sqrt(${i} * ${a})`,localLeeL:`multiplyBySpatialCount(raw, ${e}) / sqrt(${i} * ${a})`,localJoinCount:`raw`,localBivariateJoinCount:`raw`}[u];I.push(r(t,{id:`${n}-observed-statistic`,operation:L,variant:`observed-statistic`,bindings:[{name:`observedRaw`,view:G,type:`f32`,access:`read`},{name:`rowPositions`,view:R,type:`u32`,access:`read`},{name:`compactX`,view:z,type:`f32`,access:`read`},{name:`totals`,view:ae,type:`f32`,access:`read`},{name:`observed`,view:Q,type:`f32`,access:`read_write`}],invocationCount:_,declarations:ce,body:`let raw = observedRaw[observedRawOffset + index];
  let position = min(rowPositions[rowPositionsOffset + index], ROWS - 1u);
  let scaled = ${o};
  // Canonical +0: a zero lag times a negative deviation is -0. Tested on the bits because a
  // floating-point compare may be folded by the shader compiler, which treats -0 as 0.
  let scaledBits = bitcast<u32>(scaled);
  let canonical = bitcast<f32>(select(scaledBits, 0u, (scaledBits & 0x7fffffffu) == 0u));
  observed[observedOffset + index] = select(getQuietNaN(index), canonical, isFiniteFloat(raw));`}))}if(u===`localG`||u===`localGStar`){let e=C(t,{id:n,operation:L,rows:_,compactX:z,rowPositions:R}),i=ie.get(this);i&&(i.negativeCount=e.negativeCount);let o=[{name:`negativeCount`,view:e.negativeCount,type:`u32`,access:`read`},{name:`exceedances`,view:a.output.exceedances,type:`u32`,access:`read_write`},{name:`overflow`,view:a.output.status.overflow,type:`atomic<u32>`,access:`read_write`}];Z&&o.push({name:`permutationCounts`,view:Z,type:`u32`,access:`read_write`}),a.output.observed&&o.push({name:`observed`,view:a.output.observed,type:`f32`,access:`read_write`}),a.output.status.invalidCount&&o.push({name:`invalidCount`,view:a.output.status.invalidCount,type:`u32`,access:`read_write`}),I.push(...e.nodes,r(t,{id:`${n}-negative-values`,operation:L,variant:`negative-values`,bindings:o,invocationCount:_,declarations:`const NOT_TESTED: u32 = ${B}u;
${S}`,body:`let invalid = negativeCount[negativeCountOffset] > 0u;
  if (invalid) {
    exceedances[exceedancesOffset + index] = NOT_TESTED;
    ${Z?`permutationCounts[permutationCountsOffset + index] = NOT_TESTED;`:``}
    ${a.output.observed?`observed[observedOffset + index] = getQuietNaN(index);`:``}
  }
  if (index == 0u) {
    ${a.output.status.invalidCount?`invalidCount[invalidCountOffset] = negativeCount[negativeCountOffset];`:``}
    if (invalid) {
      atomicMax(&overflow[overflowOffset], 1u);
    }
  }`}))}else a.output.status.invalidCount&&I.push(i(t,{id:`${n}-invalid-clear`,operation:L,view:a.output.status.invalidCount,type:`u32`,value:`0u`,componentCount:1}));let pe=k!==`none`||a.output.adjustedPValues!==void 0,$=[{name:`exceedances`,view:a.output.exceedances,type:`u32`,access:`read`},{name:`parameters`,view:s,type:`u32`,access:`read`},{name:`pseudoPValues`,view:a.output.pseudoPValues,type:`f32`,access:`read_write`}];Z&&$.push({name:`permutationCounts`,view:Z,type:`u32`,access:`read`});let me=D?`select(min(${T}u * (count + 1u), readValidPermutationCount() + 1u), min(${T}u * STOP_AFTER_EXCEEDANCES, permutationCounts[permutationCountsOffset + index]), count == STOP_AFTER_EXCEEDANCES)`:`min(${T}u * (count + 1u), readValidPermutationCount() + 1u)`,he=D?`select(readValidPermutationCount() + 1u, permutationCounts[permutationCountsOffset + index], count == STOP_AFTER_EXCEEDANCES)`:`readValidPermutationCount() + 1u`,ge=`significant[significantOffset + index] = select(0u, 1u, tested && f32(${me}) <= readSignificanceLevel() * f32(${he}));`;a.output.significant&&k===`none`&&$.push({name:`significant`,view:a.output.significant,type:`u32`,access:`read_write`}),I.push(r(t,{id:`${n}-classify`,operation:L,variant:`classify`,bindings:$,invocationCount:_,declarations:`${v(m)}
const NOT_TESTED: u32 = ${B}u;${D?`\nconst STOP_AFTER_EXCEEDANCES: u32 = ${E}u;`:``}`,body:`let count = exceedances[exceedancesOffset + index];
  let tested = count != NOT_TESTED;
  let pValue = f32(${me}) / f32(${he});
  pseudoPValues[pseudoPValuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), pValue, tested);
  ${a.output.significant&&k===`none`?ge:``}`}));let _e;if(pe){let i=a.output.adjustedPValues??e(t,`${n}-adjusted-p`,`float32`,_);_e=k===`none`?void 0:i,I.push(...new g({id:`${n}-multiple-testing`,pValues:a.output.pseudoPValues,method:k,output:{adjustedPValues:i}}).getCommandNodes(t)),a.output.significant&&I.push(r(t,{id:`${n}-significant`,operation:L,variant:`significant`,bindings:[{name:`adjustedPValues`,view:i,type:`f32`,access:`read`},{name:`parameters`,view:s,type:`u32`,access:`read`},{name:`significant`,view:a.output.significant,type:`u32`,access:`read_write`}],invocationCount:_,declarations:v(m),body:`let adjusted = adjustedPValues[adjustedPValuesOffset + index];
  significant[significantOffset + index] = select(0u, 1u, adjusted <= readSignificanceLevel());`}))}return a.output.classes&&Q&&I.push(...this.getGearyClassificationNodes(t,Q,_e,z,R,Z)),I}getGearyClassificationNodes(t,i,a,o,s,c){let{id:l,props:u}=this,d=u.values.length,f=u.output.classes,p=j(A(l,u.alternative)),m=e(t,`${l}-geary-mean-matrix`,`float32`,d),h=e(t,`${l}-geary-finite`,`uint32`,d),g=e(t,`${l}-geary-mean-sums`,`uint32`,2),y=t.createDataView(g.buffer,{format:`float32`,length:1}),b=t.createDataView(g.buffer,{format:`uint32`,length:1,byteOffset:4}),x=u.stopAfterExceedances,C=x===void 0?`min(${p}u * (count + 1u), readValidPermutationCount() + 1u)`:`select(min(${p}u * (count + 1u), readValidPermutationCount() + 1u), min(${p}u * ${x}u, permutationCounts[permutationCountsOffset + index]), count == ${x}u)`,w=x===void 0?`readValidPermutationCount() + 1u`:`select(readValidPermutationCount() + 1u, permutationCounts[permutationCountsOffset + index], count == ${x}u)`,E=a?`adjustedPValues[adjustedPValuesOffset + index] <= readSignificanceLevel()`:`count != NOT_TESTED && f32(${C}) <= readSignificanceLevel() * f32(${w})`;return[r(t,{id:`${l}-geary-mean-inputs`,operation:L,variant:`geary-mean-inputs`,bindings:[{name:`observedView`,view:i,type:`f32`,access:`read`},{name:`matrix`,view:m,type:`f32`,access:`read_write`},{name:`finiteFlags`,view:h,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`const ROWS: u32 = ${d}u;
${S}`,body:`let value = observedView[observedViewOffset + index];
  let finite = isFiniteFloat(value);
  matrix[matrixOffset + index] = select(0.0, value, finite);
  finiteFlags[finiteFlagsOffset + index] = u32(finite);`}),..._(t,{id:`${l}-geary-mean`,operation:L,matrix:m,rows:d,blockCount:Math.ceil(d/T),first:0,count:1,totals:y}),...new n({id:`${l}-geary-finite-count`,input:h,output:b,operation:`sum`}).getCommandNodes(t),r(t,{id:`${l}-geary-classes`,operation:L,variant:a?`geary-classes-adjusted`:`geary-classes`,bindings:[{name:`observedView`,view:i,type:`f32`,access:`read`},{name:`compactX`,view:o,type:`f32`,access:`read`},{name:`rowPositions`,view:s,type:`u32`,access:`read`},{name:`sums`,view:g,type:`u32`,access:`read`},{name:`parameters`,view:u.parameters,type:`u32`,access:`read`},a?{name:`adjustedPValues`,view:a,type:`f32`,access:`read`}:{name:`exceedances`,view:u.output.exceedances,type:`u32`,access:`read`},...!a&&c?[{name:`permutationCounts`,view:c,type:`u32`,access:`read`}]:[],{name:`classes`,view:f,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`${v(u.maximumPermutations)}
const INVALID_POSITION: u32 = ${ee}u;
const NOT_TESTED: u32 = ${B}u;
${S}`,body:`let value = observedView[observedViewOffset + index];
  let position = rowPositions[rowPositionsOffset + index];
  let mean = divideBySpatialCount(bitcast<f32>(sums[sumsOffset]), sums[sumsOffset + 1u]);
  ${a?``:`let count = exceedances[exceedancesOffset + index];`}
  var code = 0u;
  if (isFiniteFloat(value) && position != INVALID_POSITION && ${a?`isFiniteFloat(adjustedPValues[adjustedPValuesOffset + index]) && `:`count != NOT_TESTED && `}isFiniteFloat(mean)) {
    if (!(${E})) {
      code = ${V.notSignificant}u;
    } else if (value < mean) {
      let focus = compactX[compactXOffset + position];
      code = select(select(0u, ${V.lowLow}u, focus < 0.0), ${V.highHigh}u, focus > 0.0);
    } else if (value > mean) {
      code = ${V.negativeAssociation}u;
    }
  }
  classes[classesOffset + index] = code;`})]}};export{I as n,ae as t};
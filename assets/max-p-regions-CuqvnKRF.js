import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-scan-B4NNTfF9.js";import{a as n,o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as i,f as a,p as o,u as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as f}from"./load-balanced-for-each-GB0Xy_f0.js";import{t as p}from"./spatial-output-validation-nw5lOmVL.js";import{c as m}from"./spatial-weights-CqWysstN.js";import{l as ee}from"./wgsl-common-DfyXiYB0.js";import{n as te,t as ne}from"./philox-random-DfVxWNPn.js";import{i as re,t as ie}from"./gpu-azp-regions-DRYB6TTD.js";import{n as h,t as ae}from"./gpu-region-partition-evaluation-Cx3Lt0uB.js";import{a as g,i as _,n as oe,o as se,s as ce,t as le}from"./regionalization-inputs-CyTCPBba.js";import{r as v}from"./regionalization-command-budget-BFn_8OgY.js";import{t as ue}from"./define-run-B-brkCe4.js";import{A as y,a as b}from"./pointer-overlay-Qh8qflcA.js";import{m as x}from"./inputs-CJKKIRKK.js";import{Hs as S}from"./entry-preview-D70TdSbK.js";import{c as C}from"./geometry-Dz9W1tyx.js";import{c as w,s as T}from"./regions-CO_uxNyo.js";var E=4096;function de(e){return s(e,16)}var D=256,O=4,k=4,fe=3,A=2654435761,pe=`0.4`,me=`1.3`,he=1048576,j={invalid:0,selectedStart:1,round:2,expectedSize:3,moved:4,enclaveRound:5,startRecords:16},M=8,N={growing:0,unassigned:1,nextWave:2,waveFraction:3,cutKey:4,waveFlag:5,salt:6},P={growing:1,complete:2,failed:3,marked:4};function ge(){let e=1n<<32n,t=1n;for(let n=0;n<6;n++)t=t*(2n-BigInt(A)*t%e+e)%e;return Number(t)}function F(e,t){return{growthRounds:Math.min(t,2*e+2),enclaveRounds:Math.min(t,e+1)}}var I=1;function L(e={}){return _(`GPUMaxPRegions`,e.seed)}var R={regionCount:0,feasible:1,infeasibleRegionCount:2,invalidCount:3,selectedStart:4,improvementIterationCount:5,length:6},z=class{id;props;rowCount;columnCount;startCount;maximumGrowthSteps;maximumIterations;estimatedCommandNodeCount;constructor(e){this.id=e.id??`max-p-regions`,this.props=e,this.rowCount=m(this.id,e.weights),this.columnCount=e.columnCount??1,this.startCount=e.startCount??16,this.maximumGrowthSteps=e.maximumGrowthSteps??256,this.maximumIterations=e.maximumIterations??0;let{id:t,rowCount:n,columnCount:r,startCount:s,maximumIterations:c}=this,{output:l}=e;a(t,`columnCount`,r,1,8),a(t,`startCount`,s,1,64),a(t,`maximumGrowthSteps`,this.maximumGrowthSteps,1,E),a(t,`maximumIterations`,c,0,256);let{growthRounds:d,enclaveRounds:f}=F(n,this.maximumGrowthSteps);if(this.estimatedCommandNodeCount=v(n,d,f,c),o(t,this.estimatedCommandNodeCount),c>0&&n<2)throw Error(`${t} maximumIterations needs at least 2 rows`);if(g(t,e.parameters,I),i(e.values,[`float32`],`${t} values`),i(e.extensiveValues,[`float32`],`${t} extensiveValues`),i(e.threshold,[`float32`],`${t} threshold`),e.values.length!==n*r)throw Error(`${t} values length must equal rows * columnCount`);if(e.extensiveValues.length!==n)throw Error(`${t} extensiveValues length must equal the weights row count`);if(e.threshold.length<1)throw Error(`${t} threshold must hold one value`);if(i(l.labels,[`uint32`],`${t} output.labels`),i(l.status,[`uint32`],`${t} output.status`),i(l.objective,[`float32`],`${t} output.objective`),l.labels.length!==n)throw Error(`${t} output.labels length must equal the weights row count`);if(l.objective.length<1)throw Error(`${t} output.objective must hold one value`);if(l.status.length<R.length)throw Error(`${t} output.status must hold ${R.length} words`);u(t,[l.labels,l.objective,l.status],[e.weights.offsets,e.weights.neighbors,e.weights.weights,e.values,e.extensiveValues,e.threshold,e.parameters]),p(t,l,[[l.objective,l.status]])}getCommandNodes(i){let{id:a,props:o,rowCount:s,columnCount:u,startCount:p,maximumIterations:m}=this,{output:g}=o;l(a,i,[o.parameters,o.weights.offsets,o.weights.neighbors,o.weights.weights,o.values,o.extensiveValues,o.threshold,g.labels,g.objective,g.status]);let _=u+fe;d(i,a,{parameters:o.parameters,offsets:o.weights.offsets,neighbors:o.weights.neighbors,weights:o.weights.weights,values:o.values,extensiveValues:o.extensiveValues,threshold:o.threshold,...g}),se(a,o.weights,de(i.device)),ce(i,a,o.weights,s);for(let[e,t]of[[`standardized table`,o.standardize===!1?0:s*u*4],[`packed extensive values`,(s+1)*4],[`row state`,p*s*8],[`region phase`,p*s*4],[`region records`,p*s*_*4],[`region sums`,p*s*u*4],[`claims`,p*s*k*4],[`start state`,p*O*4],[`control state`,(j.startRecords+p*M)*4],[`region scan planes`,s*4],[`round dispatch`,36],[`evaluation summary`,m>0?0:h.length*4],[`evaluation label offsets`,m>0?0:(s+1)*4],[`evaluation row tile counts`,m>0?0:Math.ceil(s/D)*20],[`evaluation row tile sums`,m>0?0:Math.ceil(s/D)*u*4],[`evaluation label tile counts`,m>0?0:Math.ceil(s/D)*12],[`evaluation reduction carries`,m>0?0:f(s,s)*u*8]])c(i.device,a,e,t,`reduce startCount, rowCount or columnCount`);m>0&&re(i,a,s,u,s,!0);let{growthRounds:v,enclaveRounds:ue}=F(s,this.maximumGrowthSteps),y=o.standardize!==!1,b=y?e(i,`${a}-table`,`float32`,s*u):o.values,x=oe(i,{id:a,operation:`GPUMaxPRegions`,weights:o.weights,rowCount:s}),S=e(i,`${a}-extensive`,`float32`,s+1),C=p*s,w=e(i,`${a}-row-state`,`uint32`,2*C),T=e(i,`${a}-region-phase`,`uint32`,C),E=e(i,`${a}-region-data`,`float32`,C*_),I=e(i,`${a}-region-sums`,`float32`,C*u),L=e(i,`${a}-claims`,`uint32`,C*k),z=e(i,`${a}-start-state`,`uint32`,p*O),B=e(i,`${a}-state`,`uint32`,j.startRecords+p*M),V=e(i,`${a}-region-flags`,`uint32`,s),H=e(i,`${a}-region-ranks`,`uint32`,s),U=m>0,W=U?e(i,`${a}-best-labels`,`uint32`,s):g.labels,_e=U?e(i,`${a}-improvement-objective`,`float32`,1):null,ve=U?e(i,`${a}-improvement-status`,`uint32`,4):null,ye=U?null:e(i,`${a}-evaluation-summary`,`float32`,h.length),G=n(i,{id:`${a}-round-dispatch`,operation:`GPUMaxPRegions`,shapes:[[C],[C],[C]]}),K=`const ROW_COUNT: u32 = ${s}u;
const ROWS: u32 = ${s}u;
const COLUMN_COUNT: u32 = ${u}u;
const COLUMNS: u32 = ${u}u;
const REGION_STRIDE: u32 = ${_}u;
const REGION_HEADER: u32 = ${fe}u;
const START_COUNT: u32 = ${p}u;
const STARTS: u32 = ${p}u;
const NONE: u32 = 0xffffffffu;
const START_STATE_STRIDE: u32 = ${O}u;
const CLAIM_STRIDE: u32 = ${k}u;
const PHASE_GROWING: u32 = ${P.growing}u;
const PHASE_COMPLETE: u32 = ${P.complete}u;
const PHASE_FAILED: u32 = ${P.failed}u;
const PHASE_MARKED: u32 = ${P.marked}u;
const TIE_MULTIPLIER: u32 = ${A}u;
const TIE_INVERSE: u32 = ${ge()}u;
const STATE_INVALID: u32 = ${j.invalid}u;
const STATE_SELECTED: u32 = ${j.selectedStart}u;
const STATE_ROUND: u32 = ${j.round}u;
const STATE_EXPECTED_SIZE: u32 = ${j.expectedSize}u;
const STATE_MOVED: u32 = ${j.moved}u;
const STATE_ENCLAVE_ROUND: u32 = ${j.enclaveRound}u;
const STATE_STARTS: u32 = ${j.startRecords}u;
const START_RECORD: u32 = ${M}u;
const WORD_GROWING: u32 = ${N.growing}u;
const WORD_UNASSIGNED: u32 = ${N.unassigned}u;
const WORD_NEXT_WAVE: u32 = ${N.nextWave}u;
const WORD_WAVE_FRACTION: u32 = ${N.waveFraction}u;
const WORD_CUT_KEY: u32 = ${N.cutKey}u;
const WORD_WAVE_FLAG: u32 = ${N.waveFlag}u;
const WORD_SALT: u32 = ${N.salt}u;
${ee}
// Bijective tie order of rows and regions: unique per value, reshuffled by the start's salt.
fn getTie(value: u32, salt: u32) -> u32 { return (value * TIE_MULTIPLIER) ^ salt; }
fn getTieValue(tie: u32, salt: u32) -> u32 { return (tie ^ salt) * TIE_INVERSE; }`,q=`
fn getDistance(start: u32, row: u32, region: u32) -> f32 {
  let regionBase = regionDataOffset + (start * ROWS + region) * REGION_STRIDE;
  var distance = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    let mean = regionData[regionBase + REGION_HEADER + column];
    let difference = values[valuesOffset + row * COLUMNS + column] - mean;
    distance += difference * difference;
  }
  return distance;
}`,J=`let start = index / ROWS;
  let row = index - start * ROWS;`,Y=e=>G.getGate(e),be=(e,t,n)=>`${G.armWGSL}
  if (!(${e})) { dispatch[dispatchOffset] = 0u; }
  if (!(${t})) { dispatch[dispatchOffset + 3u] = 0u; }
  if (!(${n})) { dispatch[dispatchOffset + 6u] = 0u; }`,X=(...e)=>e,Z=()=>({name:`offsets`,view:x.weights.offsets,type:`u32`,access:`read`}),Q=()=>({name:`neighbors`,view:x.weights.neighbors,type:`u32`,access:`read`}),$=[...x.nodes,...y?[le(i,{id:`${a}-standardize`,operation:`GPUMaxPRegions`,values:o.values,scaled:b,rowCount:s,columnCount:u})]:[],r(i,{id:`${a}-initialize`,operation:`GPUMaxPRegions`,variant:`initialize`,invocationCount:1,bindings:[{name:`state`,view:B,type:`u32`,access:`read_write`},{name:`objective`,view:g.objective,type:`f32`,access:`read_write`},{name:`status`,view:g.status,type:`u32`,access:`read_write`}],body:`state[stateOffset] = 0u;
  state[stateOffset + 1u] = 0u;
  objective[objectiveOffset] = 0.0;
  for (var word = 0u; word < ${R.length}u; word++) { status[statusOffset + word] = 0u; }`}),r(i,{id:`${a}-pack-extensive`,operation:`GPUMaxPRegions`,variant:`pack-extensive`,invocationCount:s+1,bindings:[{name:`extensive`,view:o.extensiveValues,type:`f32`,access:`read`},{name:`threshold`,view:o.threshold,type:`f32`,access:`read`},{name:`packedValues`,view:S,type:`f32`,access:`read_write`},{name:`status`,view:g.status,type:`atomic<u32>`,access:`read_write`}],declarations:K,body:`var value = threshold[thresholdOffset];
  if (index < ROW_COUNT) { value = extensive[extensiveOffset + index]; }
  if (!isFiniteFloat(value) || value < 0.0) { atomicAdd(&status[statusOffset + 3u], 1u); }
  packedValues[packedValuesOffset + index] = value;`}),r(i,{id:`${a}-validate-rows`,operation:`GPUMaxPRegions`,variant:`parallel-validation`,invocationCount:s,bindings:[{name:`offsets`,view:o.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:o.weights.neighbors,type:`u32`,access:`read`},{name:`values`,view:o.values,type:`f32`,access:`read`},{name:`status`,view:g.status,type:`atomic<u32>`,access:`read_write`}],declarations:K,body:`var invalid = 0u;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    invalid += select(0u, 1u, !isFiniteFloat(values[valuesOffset + index * COLUMN_COUNT + column]));
  }
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    invalid += select(0u, 1u, neighbors[neighborsOffset + slot] >= ROW_COUNT);
  }
  if (invalid > 0u) { atomicAdd(&status[statusOffset + 3u], invalid); }`}),r(i,{id:`${a}-capture-invalid`,operation:`GPUMaxPRegions`,variant:`capture-invalid`,invocationCount:1,bindings:[{name:`status`,view:g.status,type:`u32`,access:`read`},{name:`state`,view:B,type:`u32`,access:`read_write`}],body:`state[stateOffset] = status[statusOffset + 3u];`}),r(i,{id:`${a}-initialize-control`,operation:`GPUMaxPRegions`,variant:`initialize-control`,invocationCount:D,workgroupSize:D,guardIndex:!1,bindings:[{name:`extensive`,view:S,type:`f32`,access:`read`},{name:`parameters`,view:o.parameters,type:`u32`,access:`read`},{name:`state`,view:B,type:`u32`,access:`read_write`}],declarations:`${K}
${te}
const PURPOSE: u32 = ${ne.maxPRegionsSeedOrder.id}u;
var<workgroup> partials: array<f32, ${D}>;`,body:`let lane = localInvocationIndex;
  var sum = 0.0;
  for (var row = lane; row < ROWS; row += ${D}u) {
    sum += extensive[extensiveOffset + row];
  }
  partials[lane] = sum;
  workgroupBarrier();
  for (var stride = ${D/2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) { partials[lane] = partials[lane] + partials[lane + stride]; }
    workgroupBarrier();
  }
  // Every lane holds the total, so every lane derives the same expected region size.
  let mean = partials[0] / f32(ROWS);
  let threshold = extensive[extensiveOffset + ROWS];
  var expectedSize = 1u;
  if (mean > 0.0) {
    expectedSize = u32(clamp(ceil(threshold / mean), 1.0, ${he}.0));
  }
  if (lane < STARTS) {
    let record = stateOffset + STATE_STARTS + lane * START_RECORD;
    state[record + WORD_GROWING] = 0u;
    state[record + WORD_UNASSIGNED] = ROWS;
    state[record + WORD_NEXT_WAVE] = 0u;
    state[record + WORD_WAVE_FRACTION] = bitcast<u32>(${pe} / f32(expectedSize));
    state[record + WORD_CUT_KEY] = 0u;
    state[record + WORD_WAVE_FLAG] = 0u;
    state[record + WORD_SALT] =
      philox4x32(vec4<u32>(NONE, lane, 0u, 0u), vec2<u32>(parameters[parametersOffset], PURPOSE)).y;
  }
  if (lane == 0u) {
    state[stateOffset + STATE_ROUND] = 0u;
    state[stateOffset + STATE_EXPECTED_SIZE] = expectedSize;
    state[stateOffset + STATE_MOVED] = 0u;
    state[stateOffset + STATE_ENCLAVE_ROUND] = 0u;
  }`}),r(i,{id:`${a}-initialize-starts`,operation:`GPUMaxPRegions`,variant:`initialize-starts`,invocationCount:C,bindings:[{name:`rowState`,view:w,type:`u32`,access:`read_write`},{name:`regionPhase`,view:T,type:`u32`,access:`read_write`},{name:`parameters`,view:o.parameters,type:`u32`,access:`read`}],declarations:`${K}
${te}
const PURPOSE: u32 = ${ne.maxPRegionsSeedOrder.id}u;`,body:`${J}
  rowState[rowStateOffset + index * 2u] = NONE;
  rowState[rowStateOffset + index * 2u + 1u] =
    philox4x32(vec4<u32>(row, start, 0u, 0u), vec2<u32>(parameters[parametersOffset], PURPOSE)).x;
  regionPhase[regionPhaseOffset + index] = 0u;`})];for(let e=0;e<v;e++){let t=`${a}-round-${e}`;$.push(r(i,{id:`${t}-control`,operation:`GPUMaxPRegions`,variant:`round-control`,invocationCount:1,bindings:[{name:`state`,view:B,type:`u32`,access:`read_write`},G.binding],declarations:K,body:`let round = state[stateOffset + STATE_ROUND];
  let period = (state[stateOffset + STATE_EXPECTED_SIZE] + 1u) / 2u;
  var anyActive = false;
  var anyWave = false;
  // Every start decides from its own counters, so its construction does not depend on the
  // other starts (a start that has settled never becomes active again).
  for (var start = 0u; start < STARTS; start++) {
    let record = stateOffset + STATE_STARTS + start * START_RECORD;
    let growing = state[record + WORD_GROWING];
    let unassigned = state[record + WORD_UNASSIGNED];
    let isActive = state[stateOffset + STATE_INVALID] == 0u && (growing > 0u || unassigned > 0u);
    let isWave = isActive && unassigned > 0u && (growing == 0u || round >= state[record + WORD_NEXT_WAVE]);
    if (isWave) {
      let fraction = bitcast<f32>(state[record + WORD_WAVE_FRACTION]);
      state[record + WORD_CUT_KEY] = select(u32(fraction * 4294967040.0), NONE, fraction >= 1.0);
      state[record + WORD_NEXT_WAVE] = round + period;
      state[record + WORD_WAVE_FRACTION] = bitcast<u32>(min(1.0, fraction * ${me}));
    }
    state[record + WORD_WAVE_FLAG] = select(0u, 1u, isWave);
    anyActive = anyActive || isActive;
    anyWave = anyWave || isWave;
  }
  ${be(`anyActive`,`anyWave`,`false`)}
  if (anyActive) { state[stateOffset + STATE_ROUND] = round + 1u; }`}),r(i,{id:`${t}-reset`,operation:`GPUMaxPRegions`,variant:`round-reset`,invocationCount:C,bindings:[{name:`claims`,view:L,type:`u32`,access:`read_write`}],declarations:K,body:`for (var word = 0u; word < CLAIM_STRIDE; word++) { claims[claimsOffset + index * CLAIM_STRIDE + word] = NONE; }`,...Y(0)}),r(i,{id:`${t}-seed-mark`,operation:`GPUMaxPRegions`,variant:`seed-mark`,invocationCount:C,bindings:X(Z(),Q(),{name:`rowState`,view:w,type:`u32`,access:`read`},{name:`regionPhase`,view:T,type:`u32`,access:`read_write`},{name:`state`,view:B,type:`u32`,access:`read`}),declarations:K,body:`${J}
  let stateBase = rowStateOffset + start * ROWS * 2u;
  let phaseBase = regionPhaseOffset + start * ROWS;
  let record = stateOffset + STATE_STARTS + start * START_RECORD;
  if (state[record + WORD_WAVE_FLAG] == 0u) { return; }
  let cutKey = state[record + WORD_CUT_KEY];
  let key = rowState[stateBase + row * 2u + 1u];
  if (rowState[stateBase + row * 2u] != NONE || key > cutKey) { return; }
  var blocked = false;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u] && !blocked; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor == row) { continue; }
    let neighborLabel = rowState[stateBase + neighbor * 2u];
    if (neighborLabel != NONE) {
      blocked = regionPhase[phaseBase + neighborLabel] == PHASE_GROWING;
    } else {
      let neighborKey = rowState[stateBase + neighbor * 2u + 1u];
      blocked = neighborKey <= cutKey && (neighborKey < key || (neighborKey == key && neighbor < row));
    }
  }
  if (!blocked) { regionPhase[phaseBase + row] = PHASE_MARKED; }`,...Y(1)}),r(i,{id:`${t}-seed-commit`,operation:`GPUMaxPRegions`,variant:`seed-commit`,invocationCount:C,bindings:X({name:`rowState`,view:w,type:`u32`,access:`read_write`},{name:`regionPhase`,view:T,type:`u32`,access:`read_write`},{name:`regionData`,view:E,type:`f32`,access:`read_write`},{name:`values`,view:b,type:`f32`,access:`read`},{name:`regionSums`,view:I,type:`f32`,access:`read_write`},{name:`extensive`,view:S,type:`f32`,access:`read`},{name:`state`,view:B,type:`atomic<u32>`,access:`read_write`}),declarations:K,body:`${J}
  if (regionPhase[regionPhaseOffset + index] != PHASE_MARKED) { return; }
  rowState[rowStateOffset + index * 2u] = row;
  let regionBase = regionDataOffset + index * REGION_STRIDE;
  let rowExtensive = extensive[extensiveOffset + row];
  regionData[regionBase] = 1.0;
  regionData[regionBase + 1u] = rowExtensive;
  regionData[regionBase + 2u] = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    let value = values[valuesOffset + row * COLUMNS + column];
    regionData[regionBase + REGION_HEADER + column] = value;
    regionSums[regionSumsOffset + index * COLUMNS + column] = value;
  }
  let record = stateOffset + STATE_STARTS + start * START_RECORD;
  atomicSub(&state[record + WORD_UNASSIGNED], 1u);
  if (rowExtensive >= extensive[extensiveOffset + ROWS]) {
    regionPhase[regionPhaseOffset + index] = PHASE_COMPLETE;
  } else {
    regionPhase[regionPhaseOffset + index] = PHASE_GROWING;
    atomicAdd(&state[record + WORD_GROWING], 1u);
  }`,...Y(1)}));for(let e of[1,2])$.push(r(i,{id:`${t}-propose-${e}`,operation:`GPUMaxPRegions`,variant:`propose-${e}`,invocationCount:C,bindings:X(Z(),Q(),{name:`rowState`,view:w,type:`u32`,access:`read`},{name:`regionPhase`,view:T,type:`u32`,access:`read`},{name:`regionData`,view:E,type:`f32`,access:`read`},{name:`values`,view:b,type:`f32`,access:`read`},{name:`claims`,view:L,type:`atomic<u32>`,access:`read_write`},{name:`state`,view:B,type:`u32`,access:`read`}),declarations:`${K}${q}`,body:`${J}
  let stateBase = rowStateOffset + start * ROWS * 2u;
  let phaseBase = regionPhaseOffset + start * ROWS;
  if (rowState[stateBase + row * 2u] != NONE) { return; }
  var priority = 0u;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    priority += select(0u, 1u, neighbor != row && rowState[stateBase + neighbor * 2u] == NONE);
  }
  priority = min(priority, 63u);
  let salt = state[stateOffset + STATE_STARTS + start * START_RECORD + WORD_SALT];
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let region = rowState[stateBase + neighbors[neighborsOffset + slot] * 2u];
    if (region == NONE || regionPhase[phaseBase + region] != PHASE_GROWING) { continue; }
    // Fewest unassigned neighbors first, then nearest to the region mean (positive float bits
    // order like the values; the low six bits are dropped to make room for the priority).
    let key = (priority << 26u) | (bitcast<u32>(getDistance(start, row, region)) >> 6u);
    let claimBase = claimsOffset + (start * ROWS + region) * CLAIM_STRIDE;
    ${e===1?`atomicMin(&claims[claimBase], key);`:`if (key == atomicLoad(&claims[claimBase])) { atomicMin(&claims[claimBase + 1u], getTie(row, salt)); }`}
  }`,...Y(0)}));for(let e of[1,2])$.push(r(i,{id:`${t}-claim-${e}`,operation:`GPUMaxPRegions`,variant:`claim-${e}`,invocationCount:C,bindings:X({name:`regionPhase`,view:T,type:`u32`,access:`read`},{name:`claims`,view:L,type:`atomic<u32>`,access:`read_write`},{name:`state`,view:B,type:`u32`,access:`read`}),declarations:K,body:`${J}
  if (regionPhase[regionPhaseOffset + index] != PHASE_GROWING) { return; }
  let ownBase = claimsOffset + index * CLAIM_STRIDE;
  let bestKey = atomicLoad(&claims[ownBase]);
  if (bestKey == NONE) { return; }
  let salt = state[stateOffset + STATE_STARTS + start * START_RECORD + WORD_SALT];
  let claimedRow = getTieValue(atomicLoad(&claims[ownBase + 1u]), salt);
  let targetBase = claimsOffset + (start * ROWS + claimedRow) * CLAIM_STRIDE;
  ${e===1?`atomicMin(&claims[targetBase + 2u], bestKey);`:`if (bestKey == atomicLoad(&claims[targetBase + 2u])) { atomicMin(&claims[targetBase + 3u], getTie(row, salt)); }`}`,...Y(0)}));$.push(r(i,{id:`${t}-commit`,operation:`GPUMaxPRegions`,variant:`commit`,invocationCount:C,bindings:X({name:`rowState`,view:w,type:`u32`,access:`read_write`},{name:`regionPhase`,view:T,type:`u32`,access:`read_write`},{name:`regionData`,view:E,type:`f32`,access:`read_write`},{name:`values`,view:b,type:`f32`,access:`read`},{name:`regionSums`,view:I,type:`f32`,access:`read_write`},{name:`extensive`,view:S,type:`f32`,access:`read`},{name:`claims`,view:L,type:`u32`,access:`read`},{name:`state`,view:B,type:`atomic<u32>`,access:`read_write`}),declarations:`${K}${q}`,body:`${J}
  if (regionPhase[regionPhaseOffset + index] != PHASE_GROWING) { return; }
  let ownBase = claimsOffset + index * CLAIM_STRIDE;
  if (claims[ownBase] == NONE) {
    // No frontier row is left and the threshold is not reached.
    regionPhase[regionPhaseOffset + index] = PHASE_FAILED;
    atomicSub(&state[stateOffset + STATE_STARTS + start * START_RECORD + WORD_GROWING], 1u);
    return;
  }
  let record = stateOffset + STATE_STARTS + start * START_RECORD;
  let salt = atomicLoad(&state[record + WORD_SALT]);
  let claimedRow = getTieValue(claims[ownBase + 1u], salt);
  if (claims[claimsOffset + (start * ROWS + claimedRow) * CLAIM_STRIDE + 3u] != getTie(row, salt)) { return; }
  let regionBase = regionDataOffset + index * REGION_STRIDE;
  let count = regionData[regionBase];
  // Incremental squared deviation about the mean: n / (n + 1) times the squared distance.
  regionData[regionBase + 2u] = regionData[regionBase + 2u] + (count / (count + 1.0)) * getDistance(start, claimedRow, row);
  regionData[regionBase] = count + 1.0;
  regionData[regionBase + 1u] = regionData[regionBase + 1u] + extensive[extensiveOffset + claimedRow];
  for (var column = 0u; column < COLUMNS; column++) {
    let sumIndex = regionSumsOffset + index * COLUMNS + column;
    let sum = regionSums[sumIndex] + values[valuesOffset + claimedRow * COLUMNS + column];
    regionSums[sumIndex] = sum;
    regionData[regionBase + REGION_HEADER + column] = sum / (count + 1.0);
  }
  rowState[rowStateOffset + (start * ROWS + claimedRow) * 2u] = row;
  atomicSub(&state[record + WORD_UNASSIGNED], 1u);
  if (regionData[regionBase + 1u] >= extensive[extensiveOffset + ROWS]) {
    regionPhase[regionPhaseOffset + index] = PHASE_COMPLETE;
    atomicSub(&state[record + WORD_GROWING], 1u);
  }`,...Y(0)}))}for(let e=0;e<ue;e++){let t=`${a}-enclave-${e}`;$.push(r(i,{id:`${t}-control`,operation:`GPUMaxPRegions`,variant:`enclave-control`,invocationCount:1,bindings:[{name:`state`,view:B,type:`u32`,access:`read_write`},G.binding],declarations:K,body:`let enclaveRound = state[stateOffset + STATE_ENCLAVE_ROUND];
  let isActive = state[stateOffset + STATE_INVALID] == 0u && (enclaveRound == 0u || state[stateOffset + STATE_MOVED] > 0u);
  ${be(`false`,`false`,`isActive`)}
  state[stateOffset + STATE_MOVED] = 0u;
  state[stateOffset + STATE_ENCLAVE_ROUND] = enclaveRound + 1u;`}),r(i,{id:`${t}-propose`,operation:`GPUMaxPRegions`,variant:`enclave-propose`,invocationCount:C,bindings:X(Z(),Q(),{name:`rowState`,view:w,type:`u32`,access:`read`},{name:`regionPhase`,view:T,type:`u32`,access:`read`},{name:`regionData`,view:E,type:`f32`,access:`read`},{name:`values`,view:b,type:`f32`,access:`read`},{name:`claims`,view:L,type:`u32`,access:`read_write`},{name:`state`,view:B,type:`u32`,access:`read`}),declarations:`${K}${q}`,body:`${J}
  let stateBase = rowStateOffset + start * ROWS * 2u;
  let phaseBase = regionPhaseOffset + start * ROWS;
  let label = rowState[stateBase + row * 2u];
  var bestRegion = NONE;
  if (label == NONE || regionPhase[phaseBase + label] != PHASE_COMPLETE) {
    let salt = state[stateOffset + STATE_STARTS + start * START_RECORD + WORD_SALT];
    var bestKey = NONE;
    var bestTie = NONE;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let region = rowState[stateBase + neighbors[neighborsOffset + slot] * 2u];
      if (region == NONE || regionPhase[phaseBase + region] != PHASE_COMPLETE) { continue; }
      let key = bitcast<u32>(getDistance(start, row, region));
      let tie = getTie(region, salt);
      if (key < bestKey || (key == bestKey && tie < bestTie)) {
        bestKey = key;
        bestTie = tie;
        bestRegion = region;
      }
    }
  }
  claims[claimsOffset + index * CLAIM_STRIDE + 3u] = bestRegion;`,...Y(2)}),r(i,{id:`${t}-commit`,operation:`GPUMaxPRegions`,variant:`enclave-commit`,invocationCount:C,bindings:X({name:`rowState`,view:w,type:`u32`,access:`read_write`},{name:`claims`,view:L,type:`u32`,access:`read`},{name:`state`,view:B,type:`atomic<u32>`,access:`read_write`}),declarations:K,body:`let proposal = claims[claimsOffset + index * CLAIM_STRIDE + 3u];
  if (proposal == NONE) { return; }
  rowState[rowStateOffset + index * 2u] = proposal;
  atomicAdd(&state[stateOffset + STATE_MOVED], 1u);`,...Y(2)}))}return $.push(r(i,{id:`${a}-finalize-starts`,operation:`GPUMaxPRegions`,variant:`finalize-starts`,invocationCount:p*D,workgroupSize:D,guardIndex:!1,bindings:[{name:`rowState`,view:w,type:`u32`,access:`read_write`},{name:`regionPhase`,view:T,type:`u32`,access:`read_write`},{name:`regionData`,view:E,type:`f32`,access:`read_write`},{name:`claims`,view:L,type:`u32`,access:`read_write`},{name:`startState`,view:z,type:`u32`,access:`read_write`}],declarations:`${K}
var<workgroup> sumScratch: array<f32, ${D}>;
var<workgroup> countScratch: array<u32, ${D}>;

// Sum of one value per lane over the workgroup in a fixed order, available to every lane.
fn reduceSum(lane: u32, value: f32) -> f32 {
  sumScratch[lane] = value;
  workgroupBarrier();
  for (var stride = ${D/2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) { sumScratch[lane] = sumScratch[lane] + sumScratch[lane + stride]; }
    workgroupBarrier();
  }
  let total = sumScratch[0];
  workgroupBarrier();
  return total;
}

fn reduceCount(lane: u32, value: u32) -> u32 {
  countScratch[lane] = value;
  workgroupBarrier();
  for (var stride = ${D/2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) { countScratch[lane] = countScratch[lane] + countScratch[lane + stride]; }
    workgroupBarrier();
  }
  let total = countScratch[0];
  workgroupBarrier();
  return total;
}`,body:`let lane = localInvocationIndex;
  // One workgroup per start, so \`start\` is uniform across the workgroup.
  let start = index / ${D}u;
  let stateBase = rowStateOffset + start * ROWS * 2u;
  let phaseBase = regionPhaseOffset + start * ROWS;
  let claimBase = claimsOffset + start * ROWS * CLAIM_STRIDE;
  for (var row = lane; row < ROWS; row += ${D}u) {
    claims[claimBase + row * CLAIM_STRIDE] = 0u;
  }
  storageBarrier();
  workgroupBarrier();
  for (var row = lane; row < ROWS; row += ${D}u) {
    var label = rowState[stateBase + row * 2u];
    if (label == NONE) {
      // Nobody reached this row: it is its own under-threshold region.
      label = row;
      rowState[stateBase + row * 2u] = row;
      regionPhase[phaseBase + row] = PHASE_FAILED;
      regionData[regionDataOffset + (start * ROWS + row) * REGION_STRIDE + 2u] = 0.0;
    }
    claims[claimBase + label * CLAIM_STRIDE] = 1u;
  }
  storageBarrier();
  workgroupBarrier();
  var squares = 0.0;
  var regions = 0u;
  var infeasible = 0u;
  for (var row = lane; row < ROWS; row += ${D}u) {
    if (claims[claimBase + row * CLAIM_STRIDE] == 1u) {
      regions += 1u;
      infeasible += select(1u, 0u, regionPhase[phaseBase + row] == PHASE_COMPLETE);
      squares += regionData[regionDataOffset + (start * ROWS + row) * REGION_STRIDE + 2u];
    }
  }
  let regionTotal = reduceCount(lane, regions);
  let infeasibleTotal = reduceCount(lane, infeasible);
  let squaresTotal = reduceSum(lane, squares);
  if (lane == 0u) {
    let base = startStateOffset + start * START_STATE_STRIDE;
    startState[base] = regionTotal;
    startState[base + 1u] = infeasibleTotal;
    startState[base + 2u] = bitcast<u32>(squaresTotal);
    startState[base + 3u] = 0u;
  }`}),r(i,{id:`${a}-select-start`,operation:`GPUMaxPRegions`,variant:`select-start`,invocationCount:1,bindings:[{name:`startState`,view:z,type:`u32`,access:`read`},{name:`state`,view:B,type:`u32`,access:`read_write`}],declarations:K,body:`var best = 0u;
  for (var start = 1u; start < START_COUNT; start++) {
    let base = startStateOffset + start * START_STATE_STRIDE;
    let bestBase = startStateOffset + best * START_STATE_STRIDE;
    let infeasible = startState[base + 1u];
    let bestInfeasible = startState[bestBase + 1u];
    let regionCount = startState[base];
    let bestRegionCount = startState[bestBase];
    let within = bitcast<f32>(startState[base + 2u]);
    let bestWithin = bitcast<f32>(startState[bestBase + 2u]);
    if (infeasible < bestInfeasible ||
        (infeasible == bestInfeasible && (regionCount > bestRegionCount ||
        (regionCount == bestRegionCount && within < bestWithin)))) {
      best = start;
    }
  }
  state[stateOffset + STATE_SELECTED] = best;`}),r(i,{id:`${a}-region-flags`,operation:`GPUMaxPRegions`,variant:`region-flags`,invocationCount:s,bindings:[{name:`claims`,view:L,type:`u32`,access:`read`},{name:`state`,view:B,type:`u32`,access:`read`},{name:`flags`,view:V,type:`u32`,access:`read_write`}],declarations:K,body:`flags[flagsOffset + index] = claims[claimsOffset + (state[stateOffset + STATE_SELECTED] * ROWS + index) * CLAIM_STRIDE];`}),...new t({id:`${a}-region-ranks`,input:V,output:H}).getCommandNodes(i),r(i,{id:`${a}-copy-best`,operation:`GPUMaxPRegions`,variant:`copy-best`,invocationCount:s,bindings:[{name:`rowState`,view:w,type:`u32`,access:`read`},{name:`ranks`,view:H,type:`u32`,access:`read`},{name:`state`,view:B,type:`u32`,access:`read`},{name:`labelsOut`,view:W,type:`u32`,access:`read_write`}],declarations:K,body:`labelsOut[labelsOutOffset + index] = ranks[ranksOffset + rowState[rowStateOffset + (state[stateOffset + STATE_SELECTED] * ROWS + index) * 2u]];`})),U?$.push(...new ie({id:`${a}-improvement`,weights:x.weights,values:b,columnCount:u,standardize:!1,extensiveValues:o.extensiveValues,extensiveThreshold:o.threshold,initialLabels:W,regionCapacity:s,maximumIterations:m,parameters:o.parameters,output:{labels:g.labels,objective:_e,status:ve}}).getCommandNodes(i)):$.push(...new ae({id:`${a}-evaluation`,values:b,columnCount:u,labels:W,labelCapacity:s,output:{summary:ye}}).getCommandNodes(i)),$.push(r(i,{id:`${a}-publish`,operation:`GPUMaxPRegions`,variant:`publish`,invocationCount:1,bindings:[{name:`startState`,view:z,type:`u32`,access:`read`},{name:`state`,view:B,type:`u32`,access:`read`},{name:`objective`,view:g.objective,type:`f32`,access:`read_write`},{name:`status`,view:g.status,type:`u32`,access:`read_write`},U?{name:`improvementObjective`,view:_e,type:`f32`,access:`read`}:{name:`evaluationSummary`,view:ye,type:`f32`,access:`read`},...U?[{name:`improvementStatus`,view:ve,type:`u32`,access:`read`}]:[]],declarations:K,body:`let best = state[stateOffset + STATE_SELECTED];
  let base = startStateOffset + best * START_STATE_STRIDE;
  var invalid = state[stateOffset + STATE_INVALID];
  ${U?`// Only an improvement-phase rejection (not already a max-p invalid input) adds to the count.
  invalid = max(invalid, improvementStatus[improvementStatusOffset + 3u]);
  let improvementIterationCount = improvementStatus[improvementStatusOffset];
  let withinSquares = improvementObjective[improvementObjectiveOffset];`:`let improvementIterationCount = 0u;
  let withinSquares = evaluationSummary[evaluationSummaryOffset + ${h.withinSumOfSquares}u];`}
  let infeasible = startState[base + 1u];
  objective[objectiveOffset] = select(withinSquares, 0.0, invalid > 0u);
  status[statusOffset] = select(startState[base], 0u, invalid > 0u);
  status[statusOffset + 1u] = select(0u, 1u, infeasible == 0u && invalid == 0u);
  status[statusOffset + 2u] = infeasible;
  status[statusOffset + 3u] = invalid;
  status[statusOffset + 4u] = best;
  status[statusOffset + ${R.improvementIterationCount}u] = improvementIterationCount;`})),$}},B=[.04,.03,.96,.2],V=4e4,H=ue({parameters:S,caption:`The most regions that each reach the population threshold; the bars are the regions' populations.`,createInputs:({world:e})=>T(y(7),e,8,2),build(e,t,n){let{tracts:r,attributes:i,columnCount:a,population:o}=t,s=e.allocate(`threshold`,`float32`,1),c=e.upload(`parameters`,`uint32`,L({seed:n.seed})),l=e.allocate(`labels`,`uint32`,r.count);return e.graph.add(new z({weights:x(e,r.contiguity),values:e.upload(`attributes`,`float32`,i).view,columnCount:a,standardize:!0,extensiveValues:e.upload(`population`,`float32`,o).view,threshold:s.view,startCount:n.startCount,parameters:c.view,output:{labels:l.view,objective:e.allocate(`objective`,`float32`,1).view,status:e.allocate(`status`,`uint32`,R.length).view}})),{outputs:{labels:l,...C(e,r)},update:e=>s.write(Float32Array.of(e.threshold))}},rebuildOn:[`startCount`,`seed`],async afterRun({outputs:e,inputs:t,upload:n}){let r=await e.labels.buffer.readAsync(),i=new Uint32Array(r.buffer,r.byteOffset,t.tracts.count),a=new Map;i.forEach((e,n)=>{e<t.tracts.count&&a.set(e,(a.get(e)??0)+t.population[n])});let o=new Float32Array(t.tracts.count);return[...a.values()].sort((e,t)=>t-e).forEach((e,t)=>o[t]=e),{bars:n(`region-population`,`float32`,o),regionCount:a.size}},interaction:b(S,{kind:`level`,hint:`set the threshold`,x:`threshold`,decorate:(e,t,{world:n})=>{let r=B[1]+e.threshold/V*(B[3]-B[1]);return{glyphs:[{shape:`line`,from:[B[0]*n.width,r],to:[B[2]*n.width,r]}],readout:`threshold ${e.threshold.toLocaleString(`en-US`)} residents per region`}}}),draw:({inputs:e,outputs:t,extra:n})=>[...w(e,t,t.labels),{layer:`chart`,kind:`bars`,values:n.bars,range:[0,V],rectangle:B,ramp:`sequential`}]});export{H as default};
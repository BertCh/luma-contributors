import{t as e}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as n}from"./kernel-C3tfQOn4.js";import{D as r,E as i,T as a,f as o,p as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as f}from"./statistics-kernel-node-lacKNERm.js";import{t as p}from"./wgsl-poisson-observed-term-DLwa4Kd3.js";import{a as m,g as h,v as g}from"./spatial-context-DwZEd4EZ.js";import{t as _}from"./philox-random-DfVxWNPn.js";import{t as v}from"./permutation-random-SiwCiDNI.js";import{t as y}from"./separation-wgsl-UgBSv4wS.js";import{n as ee,r as b,t as x}from"./scan-statistic-nearest-index-3IOGfIQN.js";import{t as S}from"./define-run-B-brkCe4.js";import{a as C}from"./pointer-overlay-Qh8qflcA.js";import{o as w,r as T}from"./events-CU2v4wkX.js";import{n as E}from"./time-window-ayU0wpq1.js";import{n as D,t as O}from"./panel-BovyVgLI.js";import{fn as k}from"./entry-preview-D70TdSbK.js";var te=`
// Uniform in the open interval (0, 1): the shared 23-bit form of PHILOX_WGSL, which cannot
// return 0 or 1. (The 24-bit form (word >> 8) + 0.5 rounds to exactly 1.0 for the largest word:
// 16777215.5 is not an f32. A draw of 1.0 gave the PTRS step a zero headroom to divide by.)
fn drawPhiloxUniform(stream: ptr<function, PhiloxStream>) -> f32 {
  return philoxOpenUnitFloat(nextPhiloxUint32(stream));
}
fn getLogFactorial(value: f32) -> f32 {
  // Stirling series of ln(value!): exact to f32 for value >= 10; smaller values are summed directly.
  if (value < 10.0) {
    var total = 0.0;
    var term = 2.0;
    loop {
      if (term > value) { break; }
      total += log(term);
      term += 1.0;
    }
    return total;
  }
  let shifted = value + 1.0;
  let inverse = 1.0 / shifted;
  return (shifted - 0.5) * log(shifted) - shifted + 0.9189385332046727 + inverse / 12.0 - inverse * inverse * inverse / 360.0;
}
fn drawPhiloxPoisson(stream: ptr<function, PhiloxStream>, mean: f32) -> u32 {
  if (!(mean > 0.0)) {
    return 0u;
  }
  if (mean < 30.0) {
    let limit = exp(-mean);
    var product = 1.0;
    var count = 0u;
    loop {
      product *= drawPhiloxUniform(stream);
      if (product <= limit) {
        break;
      }
      count++;
      if (count >= 1024u) {
        break;
      }
    }
    return count;
  }
  if (mean >= 1e7) {
    let radius = sqrt(-2.0 * log(drawPhiloxUniform(stream)));
    let angle = 6.283185307179586 * drawPhiloxUniform(stream);
    return u32(max(floor(mean + sqrt(mean) * radius * cos(angle) + 0.5), 0.0));
  }
  let logMean = log(mean);
  let scale = 0.931 + 2.53 * sqrt(mean);
  let offset = -0.059 + 0.02483 * scale;
  let inverseAlpha = 1.1239 + 1.1328 / (scale - 3.4);
  let squeeze = 0.9277 - 3.6224 / (scale - 2.0);
  for (var attempt = 0u; attempt < 64u; attempt++) {
    let offsetUniform = drawPhiloxUniform(stream) - 0.5;
    let acceptance = drawPhiloxUniform(stream);
    let headroom = 0.5 - abs(offsetUniform);
    let candidate = floor((2.0 * offset / headroom + scale) * offsetUniform + mean + 0.43);
    if (headroom >= 0.07 && acceptance <= squeeze) {
      return u32(candidate);
    }
    if (candidate < 0.0 || (headroom < 0.013 && acceptance > headroom)) {
      continue;
    }
    if (log(acceptance) + log(inverseAlpha) - log(offset / (headroom * headroom) + scale) <= -mean + candidate * logMean - getLogFactorial(candidate)) {
      return u32(candidate);
    }
  }
  return u32(mean + 0.5);
}`,A=`GPUExpectationBasedSpaceTimeScan`,j=Object.freeze({...m,measure:`distance`}),M=_.expectationBasedScan.id,N=256,P=2**25,F=4096,I=1024,L=4,R=99999,z={logLikelihoodRatio:0,observedCases:1,expectedCases:2,pValue:3},B={found:0,memberCount:1,windowLength:2,centerZone:3,replicateCount:4};function V(e){let{seed:t,replicateCount:n}=e;a(`expectation-based scan`,Number.isSafeInteger(t)&&t>=0,`seed must be a non-negative safe integer`),o(`expectation-based scan`,`replicateCount`,n,1,2**31-1);let r=new Uint32Array(4);return r[0]=t%2**32,r[1]=Math.floor(t/2**32),r[2]=n,r[3]=e.maximumWindowLength===void 0?0:Math.max(1,Math.floor(e.maximumWindowLength)),r}var H=class{id;props;spatialContext;zones;nearestZoneSearch;estimatedFullScanSeparationCount;batchReplicates;batchCount;estimatedCommandNodeCount;constructor(e,n){let c=e.id??`expectation-based-scan`;this.id=c,this.props=e,this.spatialContext=g(c,e.spatialContext,j);let{timeBuckets:l,maximumNeighbors:d,maximumWindowLength:f,maximumReplicates:p,parameters:m}=e,{output:h}=e;i(e.positions,[`float32x2`],`${c} positions`),this.zones=e.positions.length,o(c,`zones`,this.zones),o(c,`timeBuckets`,l),o(c,`maximumNeighbors`,d,1,Math.min(64,this.zones));let _=b(c,this.zones,d,e.nearestZoneSearch);this.nearestZoneSearch=_.nearestZoneSearch,this.estimatedFullScanSeparationCount=_.estimatedFullScanSeparationCount,o(c,`maximumWindowLength`,f,1,Math.min(32,l)),o(c,`maximumReplicates`,p,1,R);let v=this.zones*l;i(e.cases,[`uint32`],`${c} cases`),r(c,`cases`,e.cases,v),i(e.baseline,[`float32`],`${c} baseline`),r(c,`baseline`,e.baseline,v),i(m,[`uint32`],`${c} parameters`),a(c,m.length>=4,`parameters needs 4 rows`);let y=Math.floor(P/(this.zones*f))-1;a(c,y>=1,`zones * maximumWindowLength must be at most ${P/2}`),this.batchReplicates=Math.min(p,F,y,n??1/0),this.batchCount=Math.ceil(p/this.batchReplicates),a(c,this.batchCount<=I,`maximumReplicates needs ${this.batchCount} batches, more than ${I}`),this.estimatedCommandNodeCount=3*this.batchCount+7+(this.nearestZoneSearch===`indexed`?x:0),s(c,this.estimatedCommandNodeCount),i(h.clusterZones,[`uint32`],`${c} clusterZones`),r(c,`clusterZones`,h.clusterZones,d),i(h.clusterStatistics,[`float32`],`${c} clusterStatistics`),r(c,`clusterStatistics`,h.clusterStatistics,4),i(h.statistics,[`float32`],`${c} statistics`),r(c,`statistics`,h.statistics,p+1),i(h.summary,[`uint32`],`${c} summary`),r(c,`summary`,h.summary,5),h.zoneStatistics&&(i(h.zoneStatistics,[`float32`],`${c} zoneStatistics`),r(c,`zoneStatistics`,h.zoneStatistics,this.zones)),u(c,[h.clusterZones,h.clusterStatistics,h.statistics,h.summary,h.zoneStatistics],[e.positions,e.cases,e.baseline,m]),t(c,{clusterZones:e.output.clusterZones,clusterStatistics:e.output.clusterStatistics,statistics:e.output.statistics,summary:e.output.summary,zoneStatistics:e.output.zoneStatistics})}getInferenceMetadata(e){return V(e),{alternative:`greater`,permutationCount:Math.min(e.replicateCount,this.props.maximumReplicates),seed:e.seed,multipleTesting:`none`,testedRowsDefinition:`every center, every subset of the center and its nearest zones, and every window of the last buckets; p-values are Monte Carlo maxima over all of them; no adjustment for earlier analyses`}}getCommandNodes(t){let{id:r,props:i,zones:a,batchReplicates:o,batchCount:s}=this;e(t,r,i,{"neighbor tables":a*i.maximumNeighbors*4,"replicate table":(o+1)*a*i.maximumWindowLength*4,"center records":a*12});let{positions:u,cases:m,baseline:g,parameters:_,output:b}=i,{timeBuckets:x,maximumNeighbors:S,maximumWindowLength:C,maximumReplicates:w}=i,{clusterZones:T,clusterStatistics:E,statistics:D,summary:O,zoneStatistics:k}=b;l(r,t,[u,m,g,_,T,E,D,O,k]),c(t.device,r,`cases`,a*x*4,`reduce zones * timeBuckets`),c(t.device,r,`neighbor table`,a*S*4,`reduce zones * maximumNeighbors`);let j={context:this.spatialContext,geographic:this.spatialContext.coordinateSpace===`longitude-latitude`,sphereRadius:h(this.spatialContext)},P=a*C,F=o+1,I=(e,i)=>n(t,`${r}-${e}`,`uint32`,i),R=I(`neighbors`,a*S),z=n(t,`${r}-distances`,`float32`,a*S),B=I(`counts`,F*P),V=I(`center-best`,a*3),H=I(`state`,L),W=Math.ceil(a/N),G=o*P,K=o*W*N,q=d(t,{id:`${r}-batch-dispatch`,operation:A,shapes:[[G],[K,N]]}),J=`
const ZONES: u32 = ${a}u;
const BUCKETS: u32 = ${x}u;
const NEIGHBORS: u32 = ${S}u;
const WINDOW_CAPACITY: u32 = ${C}u;
const WINDOW_CELLS: u32 = ${P}u;
const MAXIMUM_REPLICATES: u32 = ${w}u;
const BATCH_REPLICATES: u32 = ${o}u;
const NO_CENTER: u32 = 0xffffffffu;
fn readReplicateCount() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_REPLICATES);
}
fn readWindowLimit() -> u32 {
  let requested = parameters[parametersOffset + 3u];
  return select(clamp(requested, 1u, WINDOW_CAPACITY), WINDOW_CAPACITY, requested == 0u);
}`,Y={name:`parameters`,view:_,type:`u32`,access:`read`},X=U(S),Z=(e,n)=>f(t,{id:e?`${r}-evaluate`:`${r}-batch-evaluate-${n}`,operation:A,variant:e?`evaluate`:`batch-evaluate`,...e?{}:q.getGate(1),bindings:[Y,{name:`neighbors`,view:R,type:`u32`,access:`read`},{name:`baseline`,view:g,type:`f32`,access:`read`},{name:`counts`,view:B,type:`u32`,access:`read`},{name:`statistics`,view:D,type:`atomic<u32>`,access:`read_write`},{name:`centerBest`,view:V,type:`u32`,access:e?`read_write`:`read`},...e?[]:[{name:`state`,view:H,type:`u32`,access:`read`}]],invocationCount:(e?1:o)*W*N,workgroupSize:N,guardIndex:!1,declarations:`${J}
${p}
${X}
const EVALUATE_GROUPS: u32 = ${W}u;
const EVALUATE_SIZE: u32 = ${N}u;
var<workgroup> maxima: array<f32, ${N}>;`,body:`let group = index / EVALUATE_SIZE;
  let localReplicate = group / EVALUATE_GROUPS;
  let center = (group - localReplicate * EVALUATE_GROUPS) * EVALUATE_SIZE + localInvocationIndex;
  ${e?`let slot = 0u;
  let statisticRow = 0u;
  let isLive = index < INVOCATION_COUNT;`:`let replicate = state[stateOffset + 3u] + localReplicate;
  let slot = localReplicate + 1u;
  let statisticRow = replicate + 1u;
  let isLive = index < INVOCATION_COUNT && replicate < readReplicateCount();`}
  var best = 0.0;
  if (isLive && center < ZONES) {
    let found = searchCenter(center, slot);
    best = found.score;
    ${e?`centerBest[centerBestOffset + center * 3u] = bitcast<u32>(best);
    centerBest[centerBestOffset + center * 3u + 1u] = found.windowLength;
    centerBest[centerBestOffset + center * 3u + 2u] = found.memberCount;`:``}
  }
  maxima[localInvocationIndex] = best;
  workgroupBarrier();
  for (var stride = EVALUATE_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      maxima[localInvocationIndex] = max(maxima[localInvocationIndex], maxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && isLive && maxima[0] > 0.0) {
    atomicMax(&statistics[statisticsOffset + statisticRow], bitcast<u32>(maxima[0]));
  }`}),Q=[...ee(t,{id:`${r}-nearest`,operation:A,positions:u,neighbors:R,distances:z,zones:a,listLength:S,separationWGSL:y(j),geographic:j.geographic,nearestZoneSearch:this.nearestZoneSearch}),f(t,{id:`${r}-init`,operation:A,variant:`init`,bindings:[{name:`cases`,view:m,type:`u32`,access:`read`},{name:`counts`,view:B,type:`u32`,access:`read_write`},{name:`statistics`,view:D,type:`u32`,access:`read_write`},{name:`state`,view:H,type:`u32`,access:`read_write`}],invocationCount:Math.max(P,D.length,L),declarations:`const WINDOW_CELLS: u32 = ${P}u;
const WINDOW_CAPACITY: u32 = ${C}u;
const BUCKETS: u32 = ${x}u;
const STATISTIC_ROWS: u32 = ${D.length}u;`,body:`if (index < WINDOW_CELLS) {
    let zone = index / WINDOW_CAPACITY;
    let windowBucket = index % WINDOW_CAPACITY;
    counts[countsOffset + index] = cases[casesOffset + zone * BUCKETS + BUCKETS - WINDOW_CAPACITY + windowBucket];
  }
  if (index < STATISTIC_ROWS) {
    statistics[statisticsOffset + index] = 0u;
  }
  if (index < ${L}u) {
    state[stateOffset + index] = select(0u, 0xffffffffu, index == 0u);
  }`}),Z(!0,0)];for(let e=0;e<s;e++)Q.push(f(t,{id:`${r}-batch-control-${e}`,operation:A,variant:`batch-control`,bindings:[Y,{name:`state`,view:H,type:`u32`,access:`read_write`},q.binding],invocationCount:1,declarations:J,body:`let base = state[stateOffset + 2u];
  state[stateOffset + 3u] = base;
  state[stateOffset + 2u] = base + BATCH_REPLICATES;
  if (base < readReplicateCount()) {
    ${q.armWGSL}
  } else {
    ${q.disarmWGSL}
  }`}),f(t,{id:`${r}-draw-${e}`,operation:A,variant:`draw`,...q.getGate(0),bindings:[Y,{name:`state`,view:H,type:`u32`,access:`read`},{name:`baseline`,view:g,type:`f32`,access:`read`},{name:`counts`,view:B,type:`u32`,access:`read_write`}],invocationCount:G,declarations:`${v}
${te}
${J}
const SCAN_STREAM: u32 = ${M}u;`,body:`let localReplicate = index / WINDOW_CELLS;
  let replicate = state[stateOffset + 3u] + localReplicate;
  if (replicate >= readReplicateCount()) {
    return;
  }
  let windowCell = index - localReplicate * WINDOW_CELLS;
  let zone = windowCell / WINDOW_CAPACITY;
  let windowBucket = windowCell % WINDOW_CAPACITY;
  let mean = baseline[baselineOffset + zone * BUCKETS + BUCKETS - WINDOW_CAPACITY + windowBucket];
  var stream = createPhiloxStream(vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]), replicate, windowCell, SCAN_STREAM);
  counts[countsOffset + (localReplicate + 1u) * WINDOW_CELLS + windowCell] = drawPhiloxPoisson(&stream, mean);`}),Z(!1,e));return Q.push(f(t,{id:`${r}-find-center`,operation:A,variant:`find-center`,bindings:[{name:`centerBest`,view:V,type:`u32`,access:`read`},{name:`statistics`,view:D,type:`u32`,access:`read`},{name:`state`,view:H,type:`atomic<u32>`,access:`read_write`}],invocationCount:a,body:`let bits = centerBest[centerBestOffset + index * 3u];
  if (bits != 0u && bits == statistics[statisticsOffset]) {
    atomicMin(&state[stateOffset], index);
  }`}),f(t,{id:`${r}-exceedance`,operation:A,variant:`exceedance`,bindings:[Y,{name:`statistics`,view:D,type:`f32`,access:`read`},{name:`state`,view:H,type:`atomic<u32>`,access:`read_write`}],invocationCount:w,declarations:J,body:`if (index < readReplicateCount() && statistics[statisticsOffset + index + 1u] >= statistics[statisticsOffset]) {
    atomicAdd(&state[stateOffset + 1u], 1u);
  }`}),f(t,{id:`${r}-cluster`,operation:A,variant:`cluster`,bindings:[Y,{name:`neighbors`,view:R,type:`u32`,access:`read`},{name:`baseline`,view:g,type:`f32`,access:`read`},{name:`counts`,view:B,type:`u32`,access:`read`},{name:`statistics`,view:D,type:`f32`,access:`read`},{name:`centerBest`,view:V,type:`u32`,access:`read`},{name:`state`,view:H,type:`u32`,access:`read`},{name:`clusterStatistics`,view:E,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${J}
${p}
${X}`,body:`let center = state[stateOffset];
  for (var row = 0u; row < 4u; row++) {
    clusterStatistics[clusterStatisticsOffset + row] = 0.0;
  }
  if (center != NO_CENTER) {
    let windowLength = centerBest[centerBestOffset + center * 3u + 1u];
    let memberCount = centerBest[centerBestOffset + center * 3u + 2u];
    let totals = getMemberTotals(center, 0u, windowLength, memberCount);
    clusterStatistics[clusterStatisticsOffset] = statistics[statisticsOffset];
    clusterStatistics[clusterStatisticsOffset + 1u] = totals.x;
    clusterStatistics[clusterStatisticsOffset + 2u] = totals.y;
    clusterStatistics[clusterStatisticsOffset + 3u] = f32(state[stateOffset + 1u] + 1u) / f32(readReplicateCount() + 1u);
  }`}),f(t,{id:`${r}-cluster-zones`,operation:A,variant:`cluster-zones`,bindings:[Y,{name:`neighbors`,view:R,type:`u32`,access:`read`},{name:`baseline`,view:g,type:`f32`,access:`read`},{name:`counts`,view:B,type:`u32`,access:`read`},{name:`centerBest`,view:V,type:`u32`,access:`read`},{name:`state`,view:H,type:`u32`,access:`read`},{name:`clusterZones`,view:T,type:`u32`,access:`read_write`},{name:`summary`,view:O,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${J}
${p}
${X}`,body:`let center = state[stateOffset];
  var windowLength = 0u;
  var memberCount = 0u;
  var ordered: array<u32, NEIGHBORS>;
  if (center != NO_CENTER) {
    windowLength = centerBest[centerBestOffset + center * 3u + 1u];
    memberCount = centerBest[centerBestOffset + center * 3u + 2u];
    ordered = getOrderedNeighbors(center, 0u, windowLength);
  }
  for (var row = 0u; row < NEIGHBORS; row++) {
    clusterZones[clusterZonesOffset + row] = select(0xffffffffu, neighbors[neighborsOffset + center * NEIGHBORS + ordered[row]], center != NO_CENTER && row < memberCount);
  }
  summary[summaryOffset] = select(0u, 1u, center != NO_CENTER);
  summary[summaryOffset + 1u] = memberCount;
  summary[summaryOffset + 2u] = windowLength;
  summary[summaryOffset + 3u] = select(0u, center, center != NO_CENTER);
  summary[summaryOffset + 4u] = readReplicateCount();`})),k&&Q.push(f(t,{id:`${r}-zone-statistics`,operation:A,variant:`zone-statistics`,bindings:[{name:`centerBest`,view:V,type:`u32`,access:`read`},{name:`zoneStatistics`,view:k,type:`u32`,access:`read_write`}],invocationCount:a,body:`zoneStatistics[zoneStatisticsOffset + index] = centerBest[centerBestOffset + index * 3u];`})),Q}};function U(e){let t=e>1?1<<Math.ceil(Math.log2(e)-1):0;return`
struct SearchResult {
  score: f32,
  windowLength: u32,
  memberCount: u32
}
struct NeighborhoodSums {
  caseSums: array<f32, ${e}>,
  baselineSums: array<f32, ${e}>
}
fn addWindowBucket(center: u32, slot: u32, windowLength: u32, sums: ptr<function, NeighborhoodSums>) {
  let bucketFromEnd = windowLength - 1u;
  let windowBucket = WINDOW_CAPACITY - 1u - bucketFromEnd;
  let bucket = BUCKETS - 1u - bucketFromEnd;
  for (var neighbor = 0u; neighbor < NEIGHBORS; neighbor++) {
    let zone = neighbors[neighborsOffset + center * NEIGHBORS + neighbor];
    (*sums).caseSums[neighbor] += f32(counts[countsOffset + slot * WINDOW_CELLS + zone * WINDOW_CAPACITY + windowBucket]);
    (*sums).baselineSums[neighbor] += baseline[baselineOffset + zone * BUCKETS + bucket];
  }
}
fn getOrder(sums: ptr<function, NeighborhoodSums>) -> array<u32, NEIGHBORS> {
  var keys: array<f32, NEIGHBORS>;
  var order: array<u32, NEIGHBORS>;
  for (var neighbor = 0u; neighbor < NEIGHBORS; neighbor++) {
    order[neighbor] = neighbor;
    keys[neighbor] = select(-1.0, (*sums).caseSums[neighbor] / (*sums).baselineSums[neighbor], (*sums).baselineSums[neighbor] > 0.0);
  }
  ${e>1?W(t):``}
  return order;
}
fn searchCenter(center: u32, slot: u32) -> SearchResult {
  var sums: NeighborhoodSums;
  var result = SearchResult(0.0, 0u, 0u);
  let windowLimit = readWindowLimit();
  for (var windowLength = 1u; windowLength <= windowLimit; windowLength++) {
    addWindowBucket(center, slot, windowLength, &sums);
    let order = getOrder(&sums);
    var caseTotal = 0.0;
    var baselineTotal = 0.0;
    for (var member = 0u; member < NEIGHBORS; member++) {
      let neighbor = order[member];
      if (!(sums.baselineSums[neighbor] > 0.0)) {
        break;
      }
      caseTotal += sums.caseSums[neighbor];
      baselineTotal += sums.baselineSums[neighbor];
      if (caseTotal > baselineTotal) {
        let score = getObservedTerm(caseTotal, baselineTotal);
        if (score > result.score) {
          result = SearchResult(score, windowLength, member + 1u);
        }
      }
    }
  }
  return result;
}
fn getOrderedNeighbors(center: u32, slot: u32, windowLength: u32) -> array<u32, NEIGHBORS> {
  var sums: NeighborhoodSums;
  for (var bucketLength = 1u; bucketLength <= windowLength; bucketLength++) {
    addWindowBucket(center, slot, bucketLength, &sums);
  }
  return getOrder(&sums);
}
fn getMemberTotals(center: u32, slot: u32, windowLength: u32, memberCount: u32) -> vec2<f32> {
  var sums: NeighborhoodSums;
  for (var bucketLength = 1u; bucketLength <= windowLength; bucketLength++) {
    addWindowBucket(center, slot, bucketLength, &sums);
  }
  var order = getOrder(&sums);
  var caseTotal = 0.0;
  var baselineTotal = 0.0;
  for (var member = 0u; member < memberCount; member++) {
    caseTotal += sums.caseSums[order[member]];
    baselineTotal += sums.baselineSums[order[member]];
  }
  return vec2<f32>(caseTotal, baselineTotal);
}`}function W(e){return`var increment = ${e}u;
  loop {
    var bound = ${e}u;
    var mergeBit = 0u;
    var gapWidth = increment;
    loop {
      for (var first = 0u; first + gapWidth < NEIGHBORS; first++) {
        if ((first & increment) == mergeBit) {
          let left = order[first];
          let right = order[first + gapWidth];
          if (keys[left] < keys[right] || (keys[left] == keys[right] && left > right)) {
            order[first] = right;
            order[first + gapWidth] = left;
          }
        }
      }
      if (bound == increment) {
        break;
      }
      gapWidth = bound - increment;
      bound = bound / 2u;
      mergeBit = increment;
    }
    increment = increment / 2u;
    if (increment == 0u) {
      break;
    }
  }`}var G=8,K=5,q=40,J=.5,Y=16,X=8,Z=9,Q=8,ne=199,re=99,ie=.25,$=[.06,.04,.94,.16];function ae({world:e}){let t=T([0,0,e.width,e.height],4096,0),n=Math.ceil(18/J),r=new Uint32Array(t.count),i=new Float32Array(q*n);for(let a=0;a<t.count;a++){let o=Math.min(7,Math.floor(t.positions[a*2]/e.width*G)),s=Math.min(4,Math.floor((1-t.positions[a*2+1])*K));r[a]=s*G+o;let c=Math.min(n-1,Math.floor(t.hours[a]/J));i[r[a]*n+c]++}let a=Float32Array.from({length:80},(t,n)=>{let r=n>>1;return n%2==0?(r%G+.5)/G*e.width:1-(Math.floor(r/G)+.5)/K}),o=new Float32Array(n);for(let e of t.hours)o[Math.min(n-1,Math.floor(e/J))]++;return{...t,world:e,binCount:n,zoneOf:r,counts:i,centers:a,binTotals:o}}var oe=e=>Math.round(e/J)-Y;function se(e,t){let n=oe(t),r=new Uint32Array(640),i=new Float32Array(640);for(let t=0;t<q;t++)for(let a=0;a<Y;a++){let o=n+a;r[t*Y+a]=e.counts[t*e.binCount+o];let s=0;for(let n=o-X;n<o;n++)n>=0&&(s+=e.counts[t*e.binCount+n]);i[t*Y+a]=Math.max(ie,s/X)}return{cases:r,baseline:i}}var ce=S({parameters:k,caption:`GOES lightning, May 21–22 2024: expectation-based scan of the latest half hours against each zone’s previous 4 h; boxes and bright flashes are the most likely cluster. 4,096 sampled GLM flashes; p is not adjusted for repeated looks.`,createInputs:ae,build(e,t){let n=e.allocate(`cases`,`uint32`,640),r=e.allocate(`baseline`,`float32`,640),i=e.allocate(`parameters`,`uint32`,4),a=e.allocate(`zone-statistics`,`float32`,q),o=e.allocate(`cluster-zones`,`uint32`,Z),s=e.allocate(`cluster-statistics`,`float32`,4),c=e.allocate(`summary`,`uint32`,5);e.graph.add(new H({positions:e.upload(`centers`,`float32x2`,t.centers).view,cases:n.view,baseline:r.view,timeBuckets:Y,maximumNeighbors:Z,maximumWindowLength:Q,maximumReplicates:ne,parameters:i.view,output:{clusterZones:o.view,clusterStatistics:s.view,statistics:e.allocate(`statistics`,`float32`,200).view,summary:c.view,zoneStatistics:a.view}}));let l=e.allocate(`window-flashes`,`float32`,t.count);return{outputs:{positions:e.upload(`positions`,`float32x2`,t.positions),windowFlashes:l,zoneStatistics:a,clusterZones:o,clusterStatistics:s,summary:c,panel:D(e,$,{width:t.world.width,height:1}),bins:e.upload(`bins`,`float32`,t.binTotals)},update:e=>{let a=se(t,e.windowEnd);n.write(a.cases),r.write(a.baseline),i.write(V({seed:2,replicateCount:re,maximumWindowLength:e.maximumWindowLength}))}}},interaction:C(k,{kind:`time`,hint:`move now across`,x:`windowEnd`,y:`maximumWindowLength`,decorate:(e,t,{inputs:n,outputs:r,peek:i})=>{let a=[],o=i(r.clusterZones),s=i(r.clusterStatistics),c=i(r.summary),l=n.world.width/G;for(let e of o??[]){if(e===4294967295||e>=q)continue;let t=e%G,n=Math.floor(e/G);a.push({shape:`box`,minimum:[t*l,1-(n+1)/K],maximum:[(t+1)*l,1-n/K]})}let u=c?.[B.windowLength],d=new Set(o??[]),f=e.windowEnd-(u??0)*J;r.windowFlashes.write(Float32Array.from(n.hours,(t,r)=>d.has(n.zoneOf[r])&&t>=f&&t<e.windowEnd?1:0));let p=e=>($[0]+e/18*($[2]-$[0]))*n.world.width;return a.push({shape:`line`,from:[p(e.windowEnd),$[1]],to:[p(e.windowEnd),$[3]+.02]}),u&&a.push({shape:`box`,minimum:[p(e.windowEnd-u*J),$[1]],maximum:[p(e.windowEnd),$[3]]}),{glyphs:a,readout:`now ${E(e.windowEnd)} · ${u===void 0?`…`:u*J} h window · ${w(s?.[z.observedCases],0)} vs ${w(s?.[z.expectedCases],0)} · p ${w(s?.[z.pValue],3)}`}}}),draw:({inputs:e,outputs:t,theme:n})=>[{layer:`raster`,values:t.zoneStatistics,columns:G,rows:K,range:[0,9],logarithmic:!0,alpha:.8},{layer:`dots`,positions:t.positions,values:t.windowFlashes,range:[0,1],ramp:`grey`,fade:!0,radius:2.5},O(t.panel,n),{layer:`chart`,kind:`bars`,values:t.bins,range:[0,Math.max(...e.binTotals)],rectangle:$,color:[...n.categories[0],.9]}]});export{ce as default};
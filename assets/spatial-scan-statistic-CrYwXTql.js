import{E as e,f as t,p as n,x as r,y as i}from"./gpu-contributor-utils-BIwLQON5.js";import{n as a,t as o}from"./scan-total-C4QddMmi.js";import{t as s}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as c}from"./graph-chunk-dispatch-zBt8k2-o.js";import{Q as l}from"./dist-BusamGlx.js";import{r as u}from"./kernel-C3tfQOn4.js";import{t as ee}from"./gpu-dispatch-utils-CbRzFAgT.js";import{a as d,s as te,t as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as p}from"./statistics-kernel-node-lacKNERm.js";import{a as m,g as ne,v as re}from"./spatial-context-DwZEd4EZ.js";import{t as ie}from"./separation-wgsl-UgBSv4wS.js";import{t as h}from"./permutation-random-SiwCiDNI.js";import{t as ae}from"./permutation-shuffle-zu7WXW69.js";import{t as g}from"./philox-random-DfVxWNPn.js";import{t as oe}from"./wgsl-poisson-observed-term-DLwa4Kd3.js";import{n as se,r as _,t as ce}from"./scan-statistic-nearest-index-3IOGfIQN.js";import{r as le}from"./wgsl-common-DfyXiYB0.js";import{t as ue}from"./define-run-B-brkCe4.js";import{t as de}from"./observed-positions-CQRS0vQQ.js";import{n as fe,t as v}from"./pick-Bo3t3hjv.js";import{vt as y}from"./entry-preview-D70TdSbK.js";var pe=2**24,me=99999,b={high:0,low:1,both:2};function x(e,t){if(!Number.isInteger(e)||e<1)throw Error(`scan statistic needs at least one zone`);let n=pe,r=e*e<=n?e:Math.max(1,Math.floor(n/e)-1),i=Math.min(t??r,e);if(!Number.isInteger(i)||i<1)throw Error(`scan statistic maximumWindowZones must be a positive integer`);let a=i<e?i+1:e;if(a*e>n)throw Error(`scan statistic zones * (maximumWindowZones + 1) must be <= ${n}`);return{usableZones:i,listLength:a}}var S={center:0,zoneCount:1,firstBucket:2,lastBucket:3},C={logLikelihoodRatio:0,observedCases:1,expectedCases:2,pValue:3,radius:4,observedOverExpected:5,truncatedByZoneCap:6},w={circle:0,nearest:1};function T(e){let{seed:n,permutationCount:r}=e;if(!Number.isSafeInteger(n)||n<0)throw Error(`scan statistic seed must be a non-negative safe integer`);t(`scan statistic`,`permutationCount`,r,1,2**31-1);let i=e.maximumPopulationFraction??.5;if(!(i>0&&i<=1))throw Error(`scan statistic maximumPopulationFraction must be in (0, 1]`);let a=new Uint32Array(8);return a[0]=n%2**32,a[1]=Math.floor(n/2**32),a[2]=r,a[3]=new Uint32Array(new Float32Array([i]).buffer)[0],a[4]=Math.max(1,Math.floor(e.maximumWindowZones??2**31)),a[5]=e.maximumTimeBuckets===void 0?0:Math.max(1,Math.floor(e.maximumTimeBuckets)),a[6]=w[e.windowShape??`circle`],a[7]=b[e.rateDirection??`high`],a}var he=`
struct ScanDrawInterval { first: u32, stop: u32 }
fn getScanDrawInterval(total: u32, blocks: u32, blockIndex: u32) -> ScanDrawInterval {
  let chunk = total / blocks + u32(total % blocks != 0u);
  var first = total;
  if (chunk > 0u && blockIndex <= total / chunk) { first = blockIndex * chunk; }
  return ScanDrawInterval(first, first + min(chunk, total - first));
}
`,ge=`
fn getScanBaselineExponent(bits: u32) -> i32 {
  let encoded = (bits >> 23u) & 255u;
  if (encoded != 0u) { return i32(encoded) - 127; }
  if (bits == 0u) { return 0; }
  return i32(31u - countLeadingZeros(bits)) - 149;
}
fn getScaledScanBaseline(bits: u32, maximumBits: u32) -> f32 {
  if (bits == 0u || maximumBits == 0u) { return 0.0; }
  var fraction = bits & 0x7fffffu;
  if ((bits & 0x7f800000u) == 0u) {
    let leading = 31u - countLeadingZeros(bits);
    fraction = (bits << (23u - leading)) & 0x7fffffu;
  }
  let significand = bitcast<f32>(fraction | 0x3f800000u);
  let shift = getScanBaselineExponent(bits) - getScanBaselineExponent(maximumBits);
  // Subnormal normalized contributions are below the working f32 range; avoid out-of-range ldexp.
  if (shift < -126) { return 0.0; }
  return ldexp(significand, shift);
}
`,E=`GPUSpatialScanStatistic`,_e=2048,D=Object.freeze({...m,measure:`distance`}),ve=256,ye=1024,be=64,O=256,k=g.spatialScanStatistic.id,A=8,j=256,xe=2**25,M=4096,Se=1024,Ce=7,we=4294967294,Te=class{id;props;spatialContext;zones;timeBuckets;model;usableZones;listLength;nearestZoneSearch;estimatedFullScanSeparationCount;estimatedCommandNodeCount;maximumClusters;batchReplicates;batchCount;parallelDraws;temporalMode;constructor(r,a){let o=r.id??`scan-statistic`;if(this.id=o,this.props=r,this.spatialContext=re(o,r.spatialContext,D),e(r.positions,[`float32x2`],`${o} positions`),e(r.cases,[`uint32`],`${o} cases`),this.model=r.model??`poisson`,![`poisson`,`bernoulli`,`space-time-permutation`].includes(this.model))throw Error(`${o} model must be 'poisson', 'bernoulli' or 'space-time-permutation'`);if(this.model===`poisson`){if(!r.baseline)throw Error(`${o} the poisson model needs a baseline`);e(r.baseline,[`float32`],`${o} baseline`)}if(this.model===`bernoulli`){if(!r.controls)throw Error(`${o} the bernoulli model needs controls`);e(r.controls,[`uint32`],`${o} controls`)}e(r.parameters,[`uint32`],`${o} parameters`);let{output:s}=r;e(s.clusterIndices,[`uint32`],`${o} output.clusterIndices`),e(s.clusterStatistics,[`float32`],`${o} output.clusterStatistics`),e(s.statistics,[`float32`],`${o} output.statistics`),e(s.summary,[`uint32`],`${o} output.summary`),this.zones=r.positions.length,this.timeBuckets=r.timeBuckets??1,this.maximumClusters=r.maximumClusters??8;let{maximumPermutations:l}=r;if(a!==void 0&&(!Number.isInteger(a)||a<1))throw Error(`${o} batchReplicates must be a positive integer`);if(this.zones<1)throw Error(`${o} needs at least one zone`);t(o,`timeBuckets`,this.timeBuckets,1,32);let u=x(this.zones,r.maximumWindowZones);this.usableZones=u.usableZones,this.listLength=u.listLength;let ee=_(o,this.zones,this.listLength,r.nearestZoneSearch);this.nearestZoneSearch=ee.nearestZoneSearch,this.estimatedFullScanSeparationCount=ee.estimatedFullScanSeparationCount,t(o,`maximumClusters`,this.maximumClusters,1,64),t(o,`maximumPermutations`,l,1,me);let d=this.zones*this.timeBuckets;if(r.cases.length!==d||this.model===`poisson`&&r.baseline.length!==d||this.model===`bernoulli`&&r.controls.length!==d)throw Error(`${o} cases, baseline and controls need zones * timeBuckets rows`);let te=Math.floor(xe/d)-1;if(te<1)throw Error(`${o} zones * timeBuckets must be <= ${xe/2}`);if(this.batchReplicates=Math.min(l,M,te,a??1/0),this.batchCount=Math.ceil(l/this.batchReplicates),this.parallelDraws=(r.parallelDraws??!1)&&this.model!==`poisson`,this.temporalMode=r.temporalMode??`retrospective`,this.temporalMode!==`retrospective`&&this.temporalMode!==`prospective`)throw Error(`${o} temporalMode must be 'retrospective' or 'prospective'`);if(this.batchCount>Se)throw Error(`${o} maximumPermutations needs ${this.batchCount} batches of ${this.batchReplicates} replicates, more than ${Se}`);if(this.estimatedCommandNodeCount=_e+(this.nearestZoneSearch===`indexed`?ce:1)+4*this.batchCount+4*this.maximumClusters,n(o,this.estimatedCommandNodeCount),r.parameters.length<8)throw Error(`${o} parameters needs 8 words`);if(s.clusterIndices.length<this.maximumClusters*4)throw Error(`${o} output.clusterIndices needs maximumClusters * 4 rows`);if(s.clusterStatistics.length<this.maximumClusters*8)throw Error(`${o} output.clusterStatistics needs maximumClusters * 8 rows`);if(s.statistics.length<l+1)throw Error(`${o} output.statistics needs maximumPermutations + 1 rows`);if(s.summary.length<4)throw Error(`${o} output.summary needs 4 rows`);if(s.zoneStatistics&&(e(s.zoneStatistics,[`float32`],`${o} output.zoneStatistics`),s.zoneStatistics.length<this.zones))throw Error(`${o} output.zoneStatistics needs one row per zone`);let f=[s.clusterIndices,s.clusterStatistics,s.statistics,s.summary,...s.zoneStatistics?[s.zoneStatistics]:[]];if(i(o,f,[r.positions,r.cases,...r.baseline?[r.baseline]:[],...r.controls?[r.controls]:[],r.parameters]),new Set(f.map(e=>e.buffer)).size!==f.length)throw Error(`${o} outputs must use separate buffers`);r.output.status&&(e(r.output.status.invalidCount,[`uint32`],`${o} output.status.invalidCount`),t(o,`output.status.invalidCount.length`,r.output.status.invalidCount.length,1),i(o,[r.output.status.invalidCount],[r.positions,r.cases,r.baseline,r.controls,r.parameters])),c(o,{clusterIndices:r.output.clusterIndices,clusterStatistics:r.output.clusterStatistics,statistics:r.output.statistics,summary:r.output.summary,zoneStatistics:r.output.zoneStatistics,invalidCount:r.output.status?.invalidCount})}getInferenceMetadata(e){T(e);let t=e.rateDirection??`high`;return{alternative:t===`high`?`greater`:t===`low`?`lesser`:`two-sided`,permutationCount:Math.min(e.permutationCount,this.props.maximumPermutations),seed:e.seed,multipleTesting:`none`,testedRowsDefinition:this.temporalMode===`prospective`?`every prospective window: a center zone, its nearest zones within maximumPopulationFraction of the population, and a run of consecutive time buckets ending at the last bucket, within maximumTimeBuckets; p-values are Monte Carlo maxima over all those windows; no adjustment for earlier analyses`:`every window: a center zone, its nearest zones within maximumPopulationFraction of the population, and a run of consecutive time buckets within maximumTimeBuckets; p-values are Monte Carlo maxima over all windows`}}getCommandNodes(e){let{id:t,props:n,zones:i,timeBuckets:c,listLength:m,usableZones:re,maximumClusters:g,model:_}=this;s(e,t,n,{"cell scratch":i*c*4,"neighbor scratch":i*m*4,"replicate table":(this.batchReplicates+1)*i*c*4,"center records":i*10*4});let ce=this.temporalMode===`prospective`,ue={context:this.spatialContext,geographic:this.spatialContext.coordinateSpace===`longitude-latitude`,sphereRadius:ne(this.spatialContext)},de=ie(ue),{positions:fe,cases:v,parameters:y,maximumPermutations:pe}=n,{clusterIndices:me,clusterStatistics:b,statistics:x,summary:S,zoneStatistics:C}=n.output;r(t,e,[fe,v,...n.baseline?[n.baseline]:[],...n.controls?[n.controls]:[],y,me,b,x,S,...C?[C]:[]]);let w=i*c,T=Math.min(ve,w),_e=Math.ceil(w/T),D=Math.min(ve,i),xe=Math.ceil(i/D),{batchReplicates:M,batchCount:Se}=this,Te=M+1,N=(n,r)=>u(e,`${t}-${n}`,`uint32`,r),P=(n,r)=>u(e,`${t}-${n}`,`float32`,r),F=N(`neighbors`,i*m),Ee=P(`distances`,i*m),I=N(`members`,i),L=P(`weights`,w),De=_===`poisson`?N(`baseline-maximum`,1):null,R=_===`bernoulli`?N(`population`,w):null,z=_===`space-time-permutation`?N(`zone-totals`,i):null,B=_===`space-time-permutation`?N(`bucket-totals`,c):null,Oe=this.parallelDraws&&_===`space-time-permutation`?N(`zone-cumulative`,i):null,ke=this.parallelDraws&&_===`space-time-permutation`?N(`bucket-cumulative`,c):null,Ae=P(`block-baseline`,T),je=N(`block-cases`,T),V=N(`totals`,Ce),H=N(`cumulative`,w),U=N(`counts`,Te*w),Me=Math.ceil(i/j),W=N(`center-best`,i*A+i*2),G=N(`alive`,i),Ne=N(`partials`,D*2),K=N(`state`,8),Pe=[m,i,D],Fe=Pe.map(te),Ie=Pe.map((t,n)=>ee(E,Math.max(t,1),Fe[n],e.device.limits.maxComputeWorkgroupsPerDimension)),q=u(e,`${t}-round-dispatch`,`uint32`,3*Ie.length,l.STORAGE|l.INDIRECT),Le=e=>({workgroupSize:Fe[e],condition:{id:`${t}-round-gate-${e}`,source:`gpu`,mode:`indirect`,buffer:q.buffer,byteOffset:q.byteOffset+e*12},extraResources:[{buffer:q,usage:`indirect`}]}),Re=Ie.map((e,t)=>`roundDispatch[roundDispatchOffset + ${3*t}u] = ${e.x}u;
    roundDispatch[roundDispatchOffset + ${3*t+1}u] = ${e.y}u;
    roundDispatch[roundDispatchOffset + ${3*t+2}u] = ${e.z}u;`).join(`
    `),ze=Ie.map((e,t)=>`roundDispatch[roundDispatchOffset + ${3*t}u] = 0u;`).join(` `),J=this.parallelDraws,Y=J?M*be*O:_===`poisson`?M*ye:M,X=d(e,{id:`${t}-batch-dispatch`,operation:E,shapes:[[M*w],J?[Y,O]:[Y],[M*Me*j,j]]}),Z=`${o}
const MAXIMUM_COUNT: u32 = ${we}u;

const ZONES: u32 = ${i}u;
const LIST: u32 = ${m}u;
const USABLE: u32 = ${re}u;
const BUCKETS: u32 = ${c}u;
const CELLS: u32 = ${w}u;
const BLOCKS: u32 = ${T}u;
const CELLS_PER_BLOCK: u32 = ${_e}u;
const ZONE_BLOCKS: u32 = ${D}u;
const ZONES_PER_BLOCK: u32 = ${xe}u;
const MAXIMUM_PERMUTATIONS: u32 = ${pe}u;
const BATCH_REPLICATES: u32 = ${M}u;
const CENTER_WORDS: u32 = ${A}u;
const WINDOW_INFO_START: u32 = ${i*A}u;
const NO_CENTER: u32 = 0xffffffffu;
// Weight of one cell in the integer cumulative table: its baseline share of 2^31, rounded to nearest.
fn getCellWeight(cellBaseline: f32, totalBaseline: f32) -> u32 {
  if (!(totalBaseline > 0.0) || !(cellBaseline > 0.0)) {
    return 0u;
  }
  return u32(floor(cellBaseline / totalBaseline * 2147483648.0 + 0.5));
}`,Q=`${Z}
fn readSeedKey() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}
fn readPermutations() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_PERMUTATIONS);
}
fn readMaximumFraction() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
fn readMaximumZones() -> u32 {
  return clamp(parameters[parametersOffset + 4u], 1u, USABLE);
}
// 0 selects the default, 50% of the study period.
fn readMaximumBuckets() -> u32 {
  let requested = parameters[parametersOffset + 5u];
  if (requested == 0u) {
    return max(1u, BUCKETS / 2u);
  }
  return clamp(requested, 1u, BUCKETS);
}
fn readWindowShape() -> u32 {
  return parameters[parametersOffset + 6u];
}
fn readRateDirection() -> u32 {
  return parameters[parametersOffset + 7u];
}`,Be=`${oe}
fn isDirected(observed: f32, expected: f32, direction: u32) -> bool {
  if (direction == 0u) {
    return observed > expected;
  }
  if (direction == 1u) {
    return observed < expected;
  }
  return observed != expected;
}
// Poisson divides to a bounded baseline fraction before multiplying by the case total.
fn getExpectedCases(windowWeight: f32, totalCases: f32, totalWeight: f32) -> f32 {
  return ${_===`poisson`?`(windowWeight / totalWeight) * totalCases`:`windowWeight * totalCases / totalWeight`};
}`,Ve=_===`bernoulli`?`${Be}
// Kulldorff (1997) Bernoulli LLR as the sum of o ln(o/e) over the 2x2 table of
// (cases, controls) x (inside, outside), with e = row total * column total / N.
fn getLogLikelihoodRatio(observed: f32, windowWeight: f32, totalCases: f32, totalWeight: f32, direction: u32) -> f32 {
  let outside = totalWeight - windowWeight;
  if (!(windowWeight > 0.0) || !(outside > 0.0)) {
    return 0.0;
  }
  let expectedInside = getExpectedCases(windowWeight, totalCases, totalWeight);
  if (!isDirected(observed, expectedInside, direction)) {
    return 0.0;
  }
  let totalControls = totalWeight - totalCases;
  let ratio = getObservedTerm(observed, expectedInside) +
    getObservedTerm(max(windowWeight - observed, 0.0), windowWeight * totalControls / totalWeight) +
    getObservedTerm(totalCases - observed, outside * totalCases / totalWeight) +
    getObservedTerm(max(totalControls - windowWeight + observed, 0.0), outside * totalControls / totalWeight);
  return max(ratio, 0.0);
}`:`${Be}
// Poisson LLR c ln(c/e) + (C - c) ln((C - c)/(C - e)); the space-time permutation model passes the
// window's e = (zone totals * bucket totals / C) as the window weight of a total weight C.
fn getLogLikelihoodRatio(observed: f32, windowWeight: f32, totalCases: f32, totalWeight: f32, direction: u32) -> f32 {
  let expected = getExpectedCases(windowWeight, totalCases, totalWeight);
  if (!(expected > 0.0) || !isDirected(observed, expected, direction)) {
    return 0.0;
  }
  // A window holding the whole expectation leaves no outside cell; its LLR is never positive.
  if (!(totalCases - expected > 0.0)) {
    return 0.0;
  }
  let ratio = getObservedTerm(observed, expected) +
    getObservedTerm(max(totalCases - observed, 0.0), totalCases - expected);
  return max(ratio, 0.0);
}`,He=e=>_===`bernoulli`?`population[populationOffset + ${e}]`:`getCellWeight(baseline[baselineOffset + ${e}], totalBaseline)`,Ue=R?{name:`population`,view:R,type:`u32`,access:`read`}:{name:`baseline`,view:L,type:`f32`,access:`read`},$=[...se(e,{id:`${t}-nearest`,operation:E,positions:fe,neighbors:F,distances:Ee,zones:i,listLength:m,separationWGSL:de,geographic:this.spatialContext.coordinateSpace===`longitude-latitude`,nearestZoneSearch:this.nearestZoneSearch})];if(_===`poisson`&&$.push(f(e,{id:`${t}-baseline-maximum-clear`,operation:E,view:De,type:`u32`,value:`0u`}),p(e,{id:`${t}-baseline-block-maximum`,operation:E,variant:`baseline-block-maximum`,bindings:[{name:`rawBaseline`,view:n.baseline,type:`u32`,access:`read`},{name:`baselineMaximum`,view:De,type:`atomic<u32>`,access:`read_write`}],invocationCount:T,declarations:Z,body:`var maximum = 0u;
  let first = index * CELLS_PER_BLOCK;
  for (var cell = first; cell < min(first + CELLS_PER_BLOCK, CELLS); cell++) {
    maximum = max(maximum, rawBaseline[rawBaselineOffset + cell] & 0x7fffffffu);
  }
  // Nonnegative float32 bits have the same order as uint32; at most 256 blocks contend.
  atomicMax(&baselineMaximum[baselineMaximumOffset], maximum);`}),p(e,{id:`${t}-baseline-normalize`,operation:E,variant:`baseline-normalize`,bindings:[{name:`rawBaseline`,view:n.baseline,type:`u32`,access:`read`},{name:`baselineMaximum`,view:De,type:`u32`,access:`read`},{name:`normalizedBaseline`,view:L,type:`f32`,access:`read_write`}],invocationCount:w,declarations:`${Z}\n${ge}`,body:`normalizedBaseline[normalizedBaselineOffset + index] = getScaledScanBaseline(
    rawBaseline[rawBaselineOffset + index] & 0x7fffffffu, baselineMaximum[baselineMaximumOffset]);`})),_===`bernoulli`&&$.push(p(e,{id:`${t}-population`,operation:E,variant:`population`,bindings:[{name:`cases`,view:v,type:`u32`,access:`read`},{name:`controls`,view:n.controls,type:`u32`,access:`read`},{name:`population`,view:R,type:`u32`,access:`read_write`},{name:`weights`,view:L,type:`f32`,access:`read_write`}],invocationCount:w,declarations:Z,body:`let cellPopulation = getSaturatedScanSum(cases[casesOffset + index], controls[controlsOffset + index]);
  population[populationOffset + index] = cellPopulation;
  weights[weightsOffset + index] = f32(cellPopulation);`})),_===`space-time-permutation`&&($.push(p(e,{id:`${t}-margins`,operation:E,variant:`margins`,bindings:[{name:`cases`,view:v,type:`u32`,access:`read`},{name:`zoneTotals`,view:z,type:`u32`,access:`read_write`},{name:`bucketTotals`,view:B,type:`u32`,access:`read_write`}],invocationCount:Math.max(i,c),declarations:Z,body:`if (index < ZONES) {
    var zoneSum = 0u;
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      zoneSum = getSaturatedScanSum(zoneSum, cases[casesOffset + index * BUCKETS + bucket]);
    }
    zoneTotals[zoneTotalsOffset + index] = zoneSum;
  }
  if (index < BUCKETS) {
    var bucketSum = 0u;
    for (var zone = 0u; zone < ZONES; zone++) {
      bucketSum = getSaturatedScanSum(bucketSum, cases[casesOffset + zone * BUCKETS + index]);
    }
    bucketTotals[bucketTotalsOffset + index] = bucketSum;
  }`}),p(e,{id:`${t}-weights`,operation:E,variant:`weights`,bindings:[{name:`zoneTotals`,view:z,type:`u32`,access:`read`},{name:`bucketTotals`,view:B,type:`u32`,access:`read`},{name:`weights`,view:L,type:`f32`,access:`read_write`}],invocationCount:w,declarations:Z,body:`let zone = index / BUCKETS;
  let bucket = index - zone * BUCKETS;
  var total = 0u;
  for (var other = 0u; other < BUCKETS; other++) {
    total = getSaturatedScanSum(total, bucketTotals[bucketTotalsOffset + other]);
  }
  weights[weightsOffset + index] = select(
    0.0,
    f32(zoneTotals[zoneTotalsOffset + zone]) * f32(bucketTotals[bucketTotalsOffset + bucket]) / f32(total),
    total > 0u
  );`})),Oe&&ke&&$.push(p(e,{id:`${t}-margin-cumulative`,operation:E,variant:`margin-cumulative`,bindings:[{name:`zoneTotals`,view:z,type:`u32`,access:`read`},{name:`bucketTotals`,view:B,type:`u32`,access:`read`},{name:`zoneCumulative`,view:Oe,type:`u32`,access:`read_write`},{name:`bucketCumulative`,view:ke,type:`u32`,access:`read_write`}],invocationCount:1,declarations:Z,body:`var zoneRunning = 0u;
  for (var zone = 0u; zone < ZONES; zone++) {
    zoneRunning = getSaturatedScanSum(zoneRunning, zoneTotals[zoneTotalsOffset + zone]);
    zoneCumulative[zoneCumulativeOffset + zone] = zoneRunning;
  }
  var bucketRunning = 0u;
  for (var bucket = 0u; bucket < BUCKETS; bucket++) {
    bucketRunning = getSaturatedScanSum(bucketRunning, bucketTotals[bucketTotalsOffset + bucket]);
    bucketCumulative[bucketCumulativeOffset + bucket] = bucketRunning;
  }`}))),$.push(p(e,{id:`${t}-block-totals`,operation:E,variant:`block-totals`,bindings:[{name:`baseline`,view:L,type:`f32`,access:`read`},{name:`cases`,view:v,type:`u32`,access:`read`},{name:`blockBaseline`,view:Ae,type:`f32`,access:`read_write`},{name:`blockCases`,view:je,type:`u32`,access:`read_write`}],invocationCount:T,declarations:Z,body:`var baselineSum = 0.0;
  var caseSum = 0u;
  let first = index * CELLS_PER_BLOCK;
  for (var cell = first; cell < min(first + CELLS_PER_BLOCK, CELLS); cell++) {
    baselineSum += baseline[baselineOffset + cell];
    caseSum = getSaturatedScanSum(caseSum, cases[casesOffset + cell]);
  }
  blockBaseline[blockBaselineOffset + index] = baselineSum;
  blockCases[blockCasesOffset + index] = caseSum;`}),p(e,{id:`${t}-totals`,operation:E,variant:`totals`,bindings:[{name:`blockBaseline`,view:Ae,type:`f32`,access:`read`},{name:`blockCases`,view:je,type:`u32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read_write`}],invocationCount:1,declarations:Z,body:`var baselineSum = 0.0;
  var caseSum = 0u;
  for (var blockIndex = 0u; blockIndex < BLOCKS; blockIndex++) {
    baselineSum += blockBaseline[blockBaselineOffset + blockIndex];
    caseSum = getSaturatedScanSum(caseSum, blockCases[blockCasesOffset + blockIndex]);
  }
  totals[totalsOffset + 5u] = u32(caseSum > MAXIMUM_COUNT);
  totals[totalsOffset + 6u] = caseSum;
  totals[totalsOffset] = select(caseSum, 0u, caseSum > MAXIMUM_COUNT);
  totals[totalsOffset + 1u] = bitcast<u32>(baselineSum);
  totals[totalsOffset + 2u] = 0u;
  totals[totalsOffset + 3u] = 0u;
  totals[totalsOffset + 4u] = 0u;`})),_!==`space-time-permutation`){let n=R;n||(n=N(`cell-weights`,w),$.push(p(e,{id:`${t}-cell-weights`,operation:E,variant:`cell-weights`,bindings:[Ue,{name:`totals`,view:V,type:`u32`,access:`read`},{name:`cellWeights`,view:n,type:`u32`,access:`read_write`}],invocationCount:w,declarations:Z,body:`let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  cellWeights[cellWeightsOffset + index] = ${He(`index`)};`}))),$.push(...a(e,{id:`${t}-cumulative`,operation:E,counts:n,offsets:H,total:V,totalRow:2,rowCount:w}),p(e,{id:`${t}-cumulative-inclusive`,operation:E,variant:`cumulative-inclusive`,invocationCount:w,bindings:[{name:`cumulative`,view:H,type:`u32`,access:`read_write`},{name:`cellWeights`,view:n,type:`u32`,access:`read`}],declarations:o,body:`cumulative[cumulativeOffset + index] = getSaturatedScanSum(cumulative[cumulativeOffset + index], cellWeights[cellWeightsOffset + index]);`}),p(e,{id:`${t}-weight-total`,operation:E,variant:`weight-total`,bindings:[{name:`cumulative`,view:H,type:`u32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read_write`}],invocationCount:1,declarations:Z,body:`let weightTotal = cumulative[cumulativeOffset + ${w-1}u];
  totals[totalsOffset + 2u] = weightTotal;
  if (weightTotal > MAXIMUM_COUNT) { totals[totalsOffset + 5u] = 1u; totals[totalsOffset] = 0u; }`}))}$.push(p(e,{id:`${t}-init-counts`,operation:E,variant:`init-counts`,bindings:[{name:`cases`,view:v,type:`u32`,access:`read`},{name:`counts`,view:U,type:`u32`,access:`read_write`},{name:`statistics`,view:x,type:`u32`,access:`read_write`},{name:`centerBest`,view:W,type:`u32`,access:`read_write`}],invocationCount:Math.max(w,x.length,i*A),declarations:`${Z}
const STATISTIC_WORDS: u32 = ${x.length}u;`,body:`// Clear observed records on every encoding, including invalid frames.
  if (index < ZONES * CENTER_WORDS) { centerBest[centerBestOffset + index] = 0u; }
  if (index < CELLS) {
    counts[countsOffset + index] = cases[casesOffset + index];
  }
  if (index < STATISTIC_WORDS) {
    statistics[statisticsOffset + index] = 0u;
  }`}));let We;We=_===`poisson`?n=>p(e,{id:`${t}-scatter-${n}`,...X.getGate(1),operation:E,variant:`scatter`,bindings:[{name:`parameters`,view:y,type:`u32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read`},{name:`cumulative`,view:H,type:`u32`,access:`read`},{name:`counts`,view:U,type:`atomic<u32>`,access:`read_write`}],invocationCount:Y,declarations:`${h}
${he}
${Q}
const CASE_BLOCKS: u32 = ${ye}u;
const SCAN_STREAM: u32 = ${k}u;`,body:`let localReplicate = index / CASE_BLOCKS;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  let blockIndex = index - localReplicate * CASE_BLOCKS;
  let totalCases = totals[totalsOffset];
  let totalWeight = totals[totalsOffset + 2u];
  if (replicate >= readPermutations() || totalWeight == 0u) {
    return;
  }
  let interval = getScanDrawInterval(totalCases, CASE_BLOCKS, blockIndex);
  let first = interval.first;
  let stop = interval.stop;
  var stream = createPhiloxStream(readSeedKey(), replicate, blockIndex, SCAN_STREAM);
  for (var draw = first; draw < stop; draw++) {
    // Multiply-high without rejection on purpose: the f32 weight total differs by an ulp between
    // devices, and a rejection test would turn that into a different stream. The bias is below one
    // part in the smallest cell weight (cells hold at least hundreds of units of 2^31).
    let drawTarget = getPhiloxMultiplyHigh(nextPhiloxUint32(&stream), totalWeight);
    var low = 0u;
    var high = CELLS;
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (cumulative[cumulativeOffset + middle] > drawTarget) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    atomicAdd(&counts[countsOffset + (localReplicate + 1u) * CELLS + low], 1u);
  }`}):J&&_===`bernoulli`?n=>p(e,{id:`${t}-draw-labels-parallel-${n}`,...X.getGate(1),operation:E,variant:`draw-labels-parallel`,bindings:[{name:`parameters`,view:y,type:`u32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read`},{name:`cumulative`,view:H,type:`u32`,access:`read`},{name:`counts`,view:U,type:`atomic<u32>`,access:`read_write`}],workgroupSize:O,invocationCount:Y,guardIndex:!1,declarations:`${h}
${ae(k,198)}
${he}
${Q}
const DRAW_BLOCKS: u32 = ${be}u;
const DRAW_SIZE: u32 = ${O}u;`,body:`let workgroup = index / DRAW_SIZE;
  let localReplicate = workgroup / DRAW_BLOCKS;
  let block = workgroup - localReplicate * DRAW_BLOCKS;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  let totalCases = totals[totalsOffset];
  let totalPopulation = totals[totalsOffset + 2u];
  let isLive = index < INVOCATION_COUNT && replicate < readPermutations() && totalPopulation > 0u;
  if (isLive) {
    loadSwapOrNotRounds(readSeedKey(), replicate, totalPopulation, localInvocationIndex, DRAW_SIZE);
  }
  workgroupBarrier();
  if (!isLive) {
    return;
  }
  let interval = getScanDrawInterval(totalCases, DRAW_BLOCKS, block);
  // Iterate relative positions: adding a lane or incrementing an absolute draw can wrap at uint32.
  for (var position = localInvocationIndex; position < interval.stop - interval.first; position += DRAW_SIZE) {
    let draw = interval.first + position;
    let individual = getSwapOrNotPermutationIndex(draw, totalPopulation);
    var low = 0u;
    var high = CELLS;
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (cumulative[cumulativeOffset + middle] > individual) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    atomicAdd(&counts[countsOffset + (localReplicate + 1u) * CELLS + low], 1u);
  }`}):J?n=>p(e,{id:`${t}-draw-times-parallel-${n}`,...X.getGate(1),operation:E,variant:`draw-times-parallel`,bindings:[{name:`parameters`,view:y,type:`u32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read`},{name:`zoneCumulative`,view:Oe,type:`u32`,access:`read`},{name:`bucketCumulative`,view:ke,type:`u32`,access:`read`},{name:`counts`,view:U,type:`atomic<u32>`,access:`read_write`}],workgroupSize:O,invocationCount:Y,guardIndex:!1,declarations:`${h}
${ae(k,198)}
${he}
${Q}
const DRAW_BLOCKS: u32 = ${be}u;
const DRAW_SIZE: u32 = ${O}u;`,body:`let workgroup = index / DRAW_SIZE;
  let localReplicate = workgroup / DRAW_BLOCKS;
  let block = workgroup - localReplicate * DRAW_BLOCKS;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  let totalCases = totals[totalsOffset];
  let isLive = index < INVOCATION_COUNT && replicate < readPermutations() && totalCases > 0u;
  if (isLive) {
    loadSwapOrNotRounds(readSeedKey(), replicate, totalCases, localInvocationIndex, DRAW_SIZE);
  }
  workgroupBarrier();
  if (!isLive) {
    return;
  }
  let interval = getScanDrawInterval(totalCases, DRAW_BLOCKS, block);
  // Iterate relative positions: adding a lane or incrementing an absolute draw can wrap at uint32.
  for (var position = localInvocationIndex; position < interval.stop - interval.first; position += DRAW_SIZE) {
    let draw = interval.first + position;
    var zoneLow = 0u;
    var zoneHigh = ZONES;
    while (zoneLow < zoneHigh) {
      let middle = zoneLow + (zoneHigh - zoneLow) / 2u;
      if (zoneCumulative[zoneCumulativeOffset + middle] > draw) {
        zoneHigh = middle;
      } else {
        zoneLow = middle + 1u;
      }
    }
    let rank = getSwapOrNotPermutationIndex(draw, totalCases);
    var bucketLow = 0u;
    var bucketHigh = BUCKETS;
    while (bucketLow < bucketHigh) {
      let middle = bucketLow + (bucketHigh - bucketLow) / 2u;
      if (bucketCumulative[bucketCumulativeOffset + middle] > rank) {
        bucketHigh = middle;
      } else {
        bucketLow = middle + 1u;
      }
    }
    atomicAdd(&counts[countsOffset + (localReplicate + 1u) * CELLS + zoneLow * BUCKETS + bucketLow], 1u);
  }`}):_===`bernoulli`?n=>p(e,{id:`${t}-draw-labels-${n}`,...X.getGate(1),operation:E,variant:`draw-labels`,bindings:[{name:`parameters`,view:y,type:`u32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read`},{name:`cumulative`,view:H,type:`u32`,access:`read`},{name:`population`,view:R,type:`u32`,access:`read`},{name:`counts`,view:U,type:`u32`,access:`read_write`}],invocationCount:Y,declarations:`${h}
${Q}
const SCAN_STREAM: u32 = ${k}u;`,body:`let localReplicate = index;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  let totalCases = totals[totalsOffset];
  let totalPopulation = totals[totalsOffset + 2u];
  if (replicate >= readPermutations() || totalPopulation == 0u) {
    return;
  }
  let complement = totalCases > totalPopulation - totalCases;
  let drawCount = select(totalCases, totalPopulation - totalCases, complement);
  var stream = createPhiloxStream(readSeedKey(), replicate, 0u, SCAN_STREAM);
  let row = countsOffset + (localReplicate + 1u) * CELLS;
  var drawn = 0u;
  while (drawn < drawCount) {
    let drawTarget = nextPhiloxBelow(&stream, totalPopulation);
    var low = 0u;
    var high = CELLS;
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (cumulative[cumulativeOffset + middle] > drawTarget) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    let cellPopulation = population[populationOffset + low];
    let unchosen = cellPopulation - counts[row + low];
    if (nextPhiloxBelow(&stream, max(cellPopulation, 1u)) < unchosen) {
      counts[row + low] += 1u;
      drawn++;
    }
  }
  if (complement) {
    for (var cell = 0u; cell < CELLS; cell++) {
      counts[row + cell] = population[populationOffset + cell] - counts[row + cell];
    }
  }`}):n=>p(e,{id:`${t}-draw-times-${n}`,...X.getGate(1),operation:E,variant:`draw-times`,bindings:[{name:`parameters`,view:y,type:`u32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read`},{name:`zoneTotals`,view:z,type:`u32`,access:`read`},{name:`bucketTotals`,view:B,type:`u32`,access:`read`},{name:`counts`,view:U,type:`u32`,access:`read_write`}],invocationCount:Y,declarations:`${h}
${Q}
const SCAN_STREAM: u32 = ${k}u;`,body:`let localReplicate = index;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  if (replicate >= readPermutations()) {
    return;
  }
  var remaining: array<u32, BUCKETS>;
  for (var bucket = 0u; bucket < BUCKETS; bucket++) {
    remaining[bucket] = bucketTotals[bucketTotalsOffset + bucket];
  }
  var left = totals[totalsOffset];
  var stream = createPhiloxStream(readSeedKey(), replicate, 0u, SCAN_STREAM);
  let row = countsOffset + (localReplicate + 1u) * CELLS;
  for (var zone = 0u; zone < ZONES; zone++) {
    let zoneCases = zoneTotals[zoneTotalsOffset + zone];
    for (var draw = 0u; draw < zoneCases; draw++) {
      let drawTarget = nextPhiloxBelow(&stream, left);
      var accumulated = 0u;
      var chosen = BUCKETS - 1u;
      for (var bucket = 0u; bucket < BUCKETS; bucket++) {
        accumulated += remaining[bucket];
        if (drawTarget < accumulated) {
          chosen = bucket;
          break;
        }
      }
      counts[row + zone * BUCKETS + chosen] += 1u;
      remaining[chosen] -= 1u;
      left -= 1u;
    }
  }`}),$.push(p(e,{id:`${t}-window-limit`,operation:E,variant:`window-limit`,bindings:[{name:`parameters`,view:y,type:`u32`,access:`read`},{name:`neighbors`,view:F,type:`u32`,access:`read`},{name:`baseline`,view:L,type:`f32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read`},{name:`centerBest`,view:W,type:`u32`,access:`read_write`}],invocationCount:i,declarations:Q,body:`let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  let limit = readMaximumFraction() * totalBaseline;
  let maximumZones = readMaximumZones();
  var populated = 0.0;
  var slots = 0u;
  var exceeded = false;
  for (var slot = 0u; slot < maximumZones; slot++) {
    let zone = neighbors[neighborsOffset + index * LIST + slot];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      populated += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    if (populated > limit) {
      exceeded = true;
      break;
    }
    slots = slot + 1u;
  }
  var truncated = 0u;
  if (!exceeded && USABLE < ZONES && slots == USABLE && maximumZones == USABLE) {
    let zone = neighbors[neighborsOffset + index * LIST + USABLE];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      populated += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    truncated = select(0u, 1u, populated <= limit);
  }
  centerBest[centerBestOffset + WINDOW_INFO_START + index * 2u] = slots;
  centerBest[centerBestOffset + WINDOW_INFO_START + index * 2u + 1u] = truncated;`}));let Ge=(n,r)=>p(e,{id:n?`${t}-evaluate`:`${t}-batch-evaluate-${r}`,operation:E,variant:n?`evaluate`:`batch-evaluate`,...n?{}:X.getGate(2),bindings:[{name:`parameters`,view:y,type:`u32`,access:`read`},{name:`neighbors`,view:F,type:`u32`,access:`read`},{name:`distances`,view:Ee,type:`f32`,access:`read`},{name:`counts`,view:U,type:`u32`,access:`read`},{name:`baseline`,view:L,type:`f32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read`},{name:`statistics`,view:x,type:`atomic<u32>`,access:`read_write`},{name:`centerBest`,view:W,type:`u32`,access:n?`read_write`:`read`}],invocationCount:(n?1:M)*Me*j,workgroupSize:j,guardIndex:!1,declarations:`${Q}
${Ve}
${n?de:``}
const EVALUATE_GROUPS: u32 = ${Me}u;
const EVALUATE_SIZE: u32 = ${j}u;
var<workgroup> maxima: array<f32, ${j}>;`,body:`let group = index / EVALUATE_SIZE;
  let localReplicate = group / EVALUATE_GROUPS;
  let center = (group - localReplicate * EVALUATE_GROUPS) * EVALUATE_SIZE + localInvocationIndex;
  ${n?`let slot = 0u;
  let statisticRow = 0u;
  let isLive = true;`:`let replicate = totals[totalsOffset + 4u] + localReplicate;
  let slot = localReplicate + 1u;
  let statisticRow = replicate + 1u;
  let isLive = replicate < readPermutations();`}
  var best = 0.0;
  if (isLive && center < ZONES && totals[totalsOffset + 5u] == 0u) {
  let totalCases = f32(totals[totalsOffset]);
  let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  let maximumBuckets = readMaximumBuckets();
  let direction = readRateDirection();
  let nearestShape = readWindowShape() == 1u;
  let windowSlots = centerBest[centerBestOffset + WINDOW_INFO_START + center * 2u];
  let slotStart = slot * CELLS;
  var caseSums: array<u32, BUCKETS>;
  var baselineSums: array<f32, BUCKETS>;
  var bestZones = 0u;
  var bestFirst = 0u;
  var bestLast = 0u;
  var bestCases = 0.0;
  var bestExpected = 0.0;
  for (var slotIndex = 0u; slotIndex < windowSlots; slotIndex++) {
    let zone = neighbors[neighborsOffset + center * LIST + slotIndex];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      caseSums[bucket] = getSaturatedScanSum(caseSums[bucket], counts[countsOffset + slotStart + zone * BUCKETS + bucket]);
      baselineSums[bucket] += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    // A circle never ends between two zones at the same distance.
    let separated = nearestShape || slotIndex + 1u >= ZONES ||
      distances[distancesOffset + center * LIST + slotIndex] < distances[distancesOffset + center * LIST + slotIndex + 1u];
    if (!separated) {
      continue;
    }
    for (var first = ${ce?`BUCKETS - min(maximumBuckets, BUCKETS)`:`0u`}; first < BUCKETS; first++) {
      var windowCases = 0u;
      var windowBaseline = 0.0;
      for (var last = first; last < min(first + maximumBuckets, BUCKETS); last++) {
        windowCases = getSaturatedScanSum(windowCases, caseSums[last]);
        windowBaseline += baselineSums[last];
        ${ce?`if (last != BUCKETS - 1u) { continue; }`:``}
        let ratio = getLogLikelihoodRatio(f32(windowCases), windowBaseline, totalCases, totalBaseline, direction);
        if (ratio > best) {
          best = ratio;
          bestZones = slotIndex + 1u;
          bestFirst = first;
          bestLast = last;
          bestCases = f32(windowCases);
          bestExpected = getExpectedCases(windowBaseline, totalCases, totalBaseline);
        }
      }
    }
  }
  ${n?`  if (slot == 0u) {
    let record = centerBestOffset + center * CENTER_WORDS;
    centerBest[record] = bitcast<u32>(best);
    centerBest[record + 1u] = bestZones;
    centerBest[record + 2u] = bestFirst;
    centerBest[record + 3u] = bestLast;
    centerBest[record + 4u] = bitcast<u32>(bestCases);
    centerBest[record + 5u] = bitcast<u32>(bestExpected);
    centerBest[record + 6u] = centerBest[centerBestOffset + WINDOW_INFO_START + center * 2u + 1u];
    // Squared radius in context units: decode the key of the farthest zone.
    centerBest[record + 7u] = select(
      0u,
      bitcast<u32>(getSquaredMeters(distances[distancesOffset + center * LIST + max(bestZones, 1u) - 1u])),
      bestZones > 0u
    );
  }
`:``}
  }
  // Workgroup max of the window maxima, then one atomicMax per workgroup: LLRs are
  // non-negative, so their f32 bits order as u32, and max is order independent (deterministic).
  maxima[localInvocationIndex] = best;
  workgroupBarrier();
  for (var stride = EVALUATE_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      maxima[localInvocationIndex] = max(maxima[localInvocationIndex], maxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && maxima[0] > 0.0) {
    atomicMax(&statistics[statisticsOffset + statisticRow], bitcast<u32>(maxima[0]));
  }`});$.push(Ge(!0,0));for(let n=0;n<Se;n++)$.push(p(e,{id:`${t}-batch-control-${n}`,operation:E,variant:`batch-control`,bindings:[{name:`parameters`,view:y,type:`u32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read_write`},X.binding],invocationCount:1,declarations:Q,body:`let base = totals[totalsOffset + 3u];
  totals[totalsOffset + 4u] = base;
  totals[totalsOffset + 3u] = base + BATCH_REPLICATES;
  if (base < readPermutations() && totals[totalsOffset + 5u] == 0u) {
    ${X.armWGSL}
  } else {
    ${X.disarmWGSL}
  }`}),p(e,{id:`${t}-batch-clear-${n}`,operation:E,variant:`batch-clear`,...X.getGate(0),bindings:[{name:`counts`,view:U,type:`u32`,access:`read_write`}],invocationCount:M*w,declarations:Z,body:`counts[countsOffset + CELLS + index] = 0u;`}),We(n),Ge(!1,n));$.push(p(e,{id:`${t}-select-init`,operation:E,variant:`select-init`,bindings:[{name:`centerBest`,view:W,type:`u32`,access:`read`},{name:`alive`,view:G,type:`u32`,access:`read_write`},{name:`members`,view:I,type:`u32`,access:`read_write`},{name:`state`,view:K,type:`u32`,access:`read_write`},{name:`clusterIndices`,view:me,type:`u32`,access:`read_write`},{name:`clusterStatistics`,view:b,type:`f32`,access:`read_write`},{name:`roundDispatch`,view:q,type:`u32`,access:`read_write`},...C?[{name:`zoneStatistics`,view:C,type:`f32`,access:`read_write`}]:[]],invocationCount:Math.max(i,g,8),declarations:`${Z}
const CLUSTERS: u32 = ${g}u;`,body:`if (index < ZONES) {
    let best = bitcast<f32>(centerBest[centerBestOffset + index * CENTER_WORDS]);
    alive[aliveOffset + index] = select(0u, 1u, best > 0.0);
    members[membersOffset + index] = 0u;
    ${C?`zoneStatistics[zoneStatisticsOffset + index] = best;`:``}
  }
  if (index < 8u) {
    state[stateOffset + index] = 0u;
  }
  if (index == 0u) {
    ${Re}
  }
  if (index < CLUSTERS) {
    for (var word = 0u; word < 4u; word++) {
      clusterIndices[clusterIndicesOffset + index * 4u + word] = 0u;
    }
    for (var word = 0u; word < 8u; word++) {
      clusterStatistics[clusterStatisticsOffset + index * 8u + word] = 0.0;
    }
  }`})),$.push(p(e,{id:`${t}-select-validity`,operation:E,variant:`select-validity`,invocationCount:i,bindings:[{name:`totals`,view:V,type:`u32`,access:`read`},{name:`alive`,view:G,type:`u32`,access:`read_write`},{name:`state`,view:K,type:`u32`,access:`read_write`},{name:`roundDispatch`,view:q,type:`u32`,access:`read_write`}],body:`if (totals[totalsOffset + 5u] != 0u) {
        alive[aliveOffset + index] = 0u;
        if (index == 0u) { state[stateOffset + 4u] = 1u; ${ze} }
      }`}));for(let n=0;n<g;n++)$.push(p(e,{id:`${t}-round-members-${n}`,operation:E,variant:`round-members`,...Le(0),bindings:[{name:`state`,view:K,type:`u32`,access:`read`},{name:`neighbors`,view:F,type:`u32`,access:`read`},{name:`members`,view:I,type:`u32`,access:`read_write`}],invocationCount:m,declarations:Z,body:`if (state[stateOffset + 3u] == 0u || index >= state[stateOffset + 2u]) {
    return;
  }
  let zone = neighbors[neighborsOffset + state[stateOffset + 1u] * LIST + index];
  members[membersOffset + zone] = 1u;`}),p(e,{id:`${t}-round-mask-${n}`,operation:E,variant:`round-mask`,...Le(1),bindings:[{name:`state`,view:K,type:`u32`,access:`read`},{name:`neighbors`,view:F,type:`u32`,access:`read`},{name:`centerBest`,view:W,type:`u32`,access:`read`},{name:`members`,view:I,type:`u32`,access:`read`},{name:`alive`,view:G,type:`u32`,access:`read_write`}],invocationCount:i,declarations:Z,body:`if (state[stateOffset + 3u] == 0u || alive[aliveOffset + index] == 0u) {
    return;
  }
  let ownZones = centerBest[centerBestOffset + index * CENTER_WORDS + 1u];
  for (var own = 0u; own < ownZones; own++) {
    if (members[membersOffset + neighbors[neighborsOffset + index * LIST + own]] == 1u) {
      alive[aliveOffset + index] = 0u;
      return;
    }
  }`}),p(e,{id:`${t}-round-partials-${n}`,operation:E,variant:`round-partials`,...Le(2),bindings:[{name:`centerBest`,view:W,type:`u32`,access:`read`},{name:`alive`,view:G,type:`u32`,access:`read`},{name:`partials`,view:Ne,type:`u32`,access:`read_write`}],invocationCount:D,declarations:Z,body:`var best = 0.0;
  var bestCenter = NO_CENTER;
  let first = index * ZONES_PER_BLOCK;
  for (var center = first; center < min(first + ZONES_PER_BLOCK, ZONES); center++) {
    let ratio = bitcast<f32>(centerBest[centerBestOffset + center * CENTER_WORDS]);
    if (alive[aliveOffset + center] == 1u && ratio > best) {
      best = ratio;
      bestCenter = center;
    }
  }
  partials[partialsOffset + index * 2u] = bitcast<u32>(best);
  partials[partialsOffset + index * 2u + 1u] = bestCenter;`}),p(e,{id:`${t}-round-select-${n}`,operation:E,variant:`round-select`,bindings:[{name:`parameters`,view:y,type:`u32`,access:`read`},{name:`partials`,view:Ne,type:`u32`,access:`read`},{name:`centerBest`,view:W,type:`u32`,access:`read`},{name:`roundDispatch`,view:q,type:`u32`,access:`read_write`},{name:`statistics`,view:x,type:`f32`,access:`read`},{name:`state`,view:K,type:`u32`,access:`read_write`},{name:`clusterIndices`,view:me,type:`u32`,access:`read_write`},{name:`clusterStatistics`,view:b,type:`f32`,access:`read_write`}],invocationCount:1,declarations:Q,body:`if (state[stateOffset + 4u] == 1u) {
    return;
  }
  var best = 0.0;
  var bestCenter = NO_CENTER;
  for (var blockIndex = 0u; blockIndex < ZONE_BLOCKS; blockIndex++) {
    let ratio = bitcast<f32>(partials[partialsOffset + blockIndex * 2u]);
    let center = partials[partialsOffset + blockIndex * 2u + 1u];
    if (center != NO_CENTER && ratio > best) {
      best = ratio;
      bestCenter = center;
    }
  }
  if (bestCenter == NO_CENTER) {
    state[stateOffset + 3u] = 0u;
    state[stateOffset + 4u] = 1u;
    ${ze}
    return;
  }
  let count = state[stateOffset];
  let record = centerBestOffset + bestCenter * CENTER_WORDS;
  let zoneCount = centerBest[record + 1u];
  let observed = bitcast<f32>(centerBest[record + 4u]);
  let expected = bitcast<f32>(centerBest[record + 5u]);
  var exceeding = 0u;
  let permutations = readPermutations();
  for (var replicate = 1u; replicate <= permutations; replicate++) {
    if (statistics[statisticsOffset + replicate] >= best) {
      exceeding++;
    }
  }
  clusterIndices[clusterIndicesOffset + count * 4u] = bestCenter;
  clusterIndices[clusterIndicesOffset + count * 4u + 1u] = zoneCount;
  clusterIndices[clusterIndicesOffset + count * 4u + 2u] = centerBest[record + 2u];
  clusterIndices[clusterIndicesOffset + count * 4u + 3u] = centerBest[record + 3u];
  let statisticStart = clusterStatisticsOffset + count * 8u;
  clusterStatistics[statisticStart] = best;
  clusterStatistics[statisticStart + 1u] = observed;
  clusterStatistics[statisticStart + 2u] = expected;
  clusterStatistics[statisticStart + 3u] = f32(exceeding + 1u) / f32(permutations + 1u);
  // Word 7 already holds squared distance in the context unit, not a separation key.
  clusterStatistics[statisticStart + 4u] = sqrt(bitcast<f32>(centerBest[record + 7u]));
  clusterStatistics[statisticStart + 5u] = observed / expected;
  clusterStatistics[statisticStart + 6u] = f32(centerBest[record + 6u]);
  state[stateOffset] = count + 1u;
  state[stateOffset + 1u] = bestCenter;
  state[stateOffset + 2u] = zoneCount;
  state[stateOffset + 3u] = 1u;`}));return $.push(p(e,{id:`${t}-summary`,operation:E,variant:`summary`,bindings:[{name:`parameters`,view:y,type:`u32`,access:`read`},{name:`state`,view:K,type:`u32`,access:`read`},{name:`totals`,view:V,type:`u32`,access:`read`},{name:`centerBest`,view:W,type:`u32`,access:`read`},{name:`summary`,view:S,type:`u32`,access:`read_write`}],invocationCount:1,declarations:Q,body:`summary[summaryOffset] = state[stateOffset];
  summary[summaryOffset + 1u] = totals[totalsOffset + 6u];
  summary[summaryOffset + 2u] = readPermutations();
  var overflow = 0u;
  for (var center = 0u; center < ZONES; center++) {
    overflow += centerBest[centerBestOffset + center * CENTER_WORDS + 6u];
  }
  summary[summaryOffset + 3u] = overflow;`})),$.push(p(e,{id:`${t}-invalid-count-status`,operation:E,variant:`invalid-count-status`,declarations:le,invocationCount:Math.max(x.length,b.length,C?.length??0),bindings:[{name:`totals`,view:V,type:`u32`,access:`read`},{name:`statistics`,view:x,type:`f32`,access:`read_write`},{name:`clusterStatistics`,view:b,type:`f32`,access:`read_write`},...C?[{name:`zoneStatistics`,view:C,type:`f32`,access:`read_write`}]:[],...n.output.status?[{name:`invalidCount`,view:n.output.status.invalidCount,type:`u32`,access:`read_write`}]:[]],body:`let invalid = totals[totalsOffset + 5u] != 0u;
${n.output.status?`if (index == 0u) { invalidCount[invalidCountOffset] = u32(invalid); }`:``}
if (invalid) {
 let missing = getNaN();
 if (index < ${x.length}u) { statistics[statisticsOffset + index] = missing; }
 if (index < ${b.length}u) { clusterStatistics[clusterStatisticsOffset + index] = missing; }
 ${C?`if (index < ${C.length}u) { zoneStatistics[zoneStatisticsOffset + index] = missing; }`:``}
}`})),$}},N=[[17001,-91.1877,39.9881,2148,34380],[17003,-89.3289,37.1804,121,2127],[17005,-89.4368,38.8884,417,9089],[17007,-88.8232,42.3236,1540,29579],[17009,-90.7506,39.962,166,2246],[17011,-89.5285,41.4046,979,16176],[17013,-90.6634,39.1684,108,1862],[17015,-89.9338,42.0684,371,6474],[17017,-90.2464,39.9707,475,6612],[17019,-88.1992,40.1396,6274,153990],[17021,-89.2788,39.5473,871,15673],[17023,-87.7871,39.3342,391,7702],[17025,-88.4887,38.7537,504,6670],[17027,-89.4244,38.6071,1165,18641],[17029,-88.2235,39.5195,1365,30204],[17031,-87.8167,41.84,161700,3241816],[17033,-87.7595,39.0027,492,8965],[17035,-88.2401,39.2744,335,5237],[17037,-88.7699,41.8933,3109,70859],[17039,-88.9097,40.1752,491,7854],[17041,-88.2185,39.7677,740,10744],[17043,-88.0859,41.8514,28918,522353],[17045,-87.7475,39.6808,432,8112],[17047,-88.0535,38.4161,128,2878],[17049,-88.5897,39.0594,1208,17970],[17051,-89.0245,39.0004,615,9934],[17053,-88.2236,40.5966,431,7009],[17055,-88.925,37.992,1185,18707],[17057,-90.2084,40.4728,814,15566],[17059,-88.2296,37.7643,97,2218],[17061,-90.3889,39.3591,296,5710],[17063,-88.4186,41.2856,1712,30875],[17065,-88.5388,38.0816,228,3995],[17067,-91.1633,40.4041,529,8181],[17069,-88.2689,37.5182,81,1610],[17071,-90.925,40.8189,164,2598],[17073,-90.1316,41.3528,1474,24143],[17075,-87.8248,40.747,793,12960],[17077,-89.3823,37.7859,1657,36579],[17079,-88.1544,39.0106,279,4386],[17081,-88.9241,38.2998,1244,17963],[17083,-90.3539,39.0886,504,11194],[17085,-90.2107,42.3655,463,8936],[17087,-88.8817,37.4593,321,5211],[17089,-88.4287,41.9383,16767,293706],[17091,-87.862,41.1378,3383,61523],[17093,-88.4285,41.591,4615,86156],[17095,-90.2132,40.9315,1431,24271],[17097,-88.0037,42.3225,19813,398813],[17099,-88.8862,41.3438,3353,56089],[17101,-87.7288,38.7194,338,6439],[17103,-89.2995,41.7456,1012,15682],[17105,-88.5577,40.8917,1188,18561],[17107,-89.3672,40.1241,746,15261],[17109,-90.6777,40.4567,735,18094],[17111,-88.4522,42.3244,9072,170305],[17113,-88.8464,40.4898,5083,118302],[17115,-88.9635,39.8606,3527,56336],[17117,-89.9243,39.2607,1172,22482],[17119,-89.9051,38.8299,7379,152458],[17121,-88.919,38.6497,1356,19074],[17123,-89.345,41.0329,349,5391],[17125,-89.9179,40.2397,276,6313],[17127,-88.7099,37.2207,314,6971],[17129,-89.8026,40.0274,342,6261],[17131,-90.743,41.2058,419,7718],[17133,-90.1759,38.2779,675,17900],[17135,-89.4779,39.2319,730,12626],[17137,-90.2024,39.7166,981,16705],[17139,-88.6209,39.6418,539,7696],[17141,-89.3223,42.0415,1557,26765],[17143,-89.7607,40.7878,6281,103501],[17145,-89.368,38.0849,553,9031],[17147,-88.5933,40.0134,467,8450],[17149,-90.8877,39.6227,503,7126],[17151,-88.5616,37.4136,81,1417],[17153,-89.1309,37.2263,148,2413],[17155,-89.2892,41.2021,133,2611],[17157,-89.8248,38.0509,836,13452],[17159,-88.0842,38.7125,477,7893],[17161,-90.5701,41.466,4740,77406],[17163,-89.9282,38.4711,7464,146650],[17165,-88.5412,37.7532,724,11808],[17167,-89.6617,39.7572,5841,110288],[17169,-90.611,40.1572,195,2940],[17171,-90.4781,39.6429,107,2440],[17173,-88.8061,39.3911,569,9768],[17175,-89.7972,41.0927,150,2487],[17177,-89.6623,42.3514,1148,21221],[17179,-89.5141,40.5075,3688,69508],[17181,-89.2555,37.4711,438,8115],[17183,-87.7328,40.1834,2546,36938],[17185,-87.8455,38.4412,223,5589],[17187,-90.6147,40.8492,583,9030],[17189,-89.4094,38.3533,359,6781],[17191,-88.4258,38.4295,477,7768],[17193,-88.1766,38.087,323,6650],[17195,-89.9156,41.7551,1655,27333],[17197,-87.9782,41.4449,20774,406133],[17199,-88.9299,37.7304,1931,35118],[17201,-89.1607,42.3364,9658,158310],[17203,-89.2091,40.7858,1178,20326]],P=1,F=4,Ee=[`both`,`high`,`low`],I=ue({parameters:y,caption:`Illinois county birth rates, colored by the strongest scan window centered on each county.`,createInputs:({world:e})=>{let t=de(N.map(([,e,t])=>[e,t]),e),n=Float32Array.from(N,e=>e[4]),r=Uint32Array.from(N,e=>e[3]),i=r.reduce((e,t)=>e+t,0)/n.reduce((e,t)=>e+t,0);return{zoneCount:N.length,positions:t,baseline:n,birthCounts:r,overallRate:i,index:fe(t),focus:{row:-1}}},build(e,t,n){let r=e.allocate(`cases`,`uint32`,t.zoneCount),i=e.upload(`baseline`,`float32`,t.baseline),a=e.allocate(`parameters`,`uint32`,8),o=e.allocate(`cluster-indices`,`uint32`,F*4),s=e.allocate(`cluster-statistics`,`float32`,F*8),c=e.allocate(`statistics`,`float32`,2),l=e.allocate(`summary`,`uint32`,4),u=e.allocate(`zone-statistics`,`float32`,t.zoneCount);return e.graph.add(new Te({positions:e.upload(`positions`,`float32x2`,t.positions).view,model:`poisson`,cases:r.view,baseline:i.view,maximumWindowZones:n.maximumWindowZones,maximumPermutations:P,maximumClusters:F,parameters:a.view,output:{clusterIndices:o.view,clusterStatistics:s.view,statistics:c.view,summary:l.view,zoneStatistics:u.view}})),{outputs:{zoneStatistics:u,clusterIndices:o,clusterStatistics:s,positions:e.upload(`zone-positions`,`float32x2`,t.positions)},update:e=>{r.write(Uint32Array.from(t.birthCounts,(n,r)=>{let i=t.overallRate*t.baseline[r];return Math.max(0,Math.round(i+e.contrast*(n-i)))})),a.write(T({seed:7,permutationCount:P,maximumPopulationFraction:e.maximumPopulationFraction,maximumWindowZones:n.maximumWindowZones,rateDirection:Ee[e.rateDirection]}))}}},rebuildOn:[`maximumWindowZones`],interaction:{kind:`probe`,hint:`hover a county`,rest:[.5,.45],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=v(t.index,e.x,e.y,t.focus.row);t.focus.row=a;let o=[],s=i(n.clusterIndices),c=i(n.clusterStatistics),l=``;if(s&&c){let e=s[S.center],n=s[S.zoneCount];if(e<t.zoneCount&&n>0){let r=t.index.within(t.positions[e*2],t.positions[e*2+1],1/0),i=0;for(let a of r.slice(0,n))i=Math.max(i,t.index.distanceTo(a,t.positions[e*2],t.positions[e*2+1])),o.push({shape:`dot`,at:[t.positions[a*2],t.positions[a*2+1]]});o.push({shape:`ring`,at:[t.positions[e*2],t.positions[e*2+1]],radius:i+.015,dashed:!0}),l=`cluster ${n} counties, LLR ${c[C.logLikelihoodRatio].toFixed(1)}`}}if(a<0)return{glyphs:o,readout:l};let u=i(n.zoneStatistics)?.[a],ee=t.overallRate*t.baseline[a];return o.push({shape:`crosshair`,at:[t.positions[a*2],t.positions[a*2+1]],size:12}),{glyphs:o,readout:`FIPS ${N[a][0]} · LLR ${u===void 0?`…`:u.toFixed(1)} · ${(t.birthCounts[a]/ee).toFixed(2)}x births · max ${r.maximumPopulationFraction*100}% pop`}}},draw:({outputs:e})=>[{layer:`dots`,context:!0,positions:e.positions,radius:9},{layer:`dots`,positions:e.positions,values:e.zoneStatistics,radius:13}]});export{I as default};
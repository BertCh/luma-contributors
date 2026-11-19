import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{c as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as r}from"./statistics-kernel-node-lacKNERm.js";import{E as i,T as a,f as o,g as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{n as u,r as d,t as f}from"./gpu-multiple-testing-adjustment-Bghjpx33.js";import{t as p}from"./spatial-autocorrelation-kernels-CCrGBZuG.js";import{i as m,s as h,t as g}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{t as _}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as v}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as y}from"./spatial-statistics-collective-node-BlCi_ESE.js";import{t as b}from"./spatial-statistics-counts-wgsl-B8N7XLDs.js";import{t as x}from"./spatial-statistics-population-output-B7_Z5i4X.js";function ee(e){let t=e*(e-1)/2;return`
fn getTheilSenSlope(series: ptr<function, array<f32, ${e}>>, count: u32) -> f32 {
  let pairCount = count * (count - 1u) / 2u;
  if (pairCount == 0u) {
    return 0.0;
  }
  let heapTarget = pairCount / 2u + 1u;
  var heap: array<f32, ${Math.floor(t/2)+1}>;
  var size = 0u;
  for (var first = 0u; first + 1u < count; first++) {
    for (var second = first + 1u; second < count; second++) {
      let slope = ((*series)[second] - (*series)[first]) / f32(second - first);
      if (size < heapTarget) {
        var child = size;
        heap[child] = slope;
        size++;
        while (child > 0u) {
          let parent = (child - 1u) / 2u;
          if (heap[parent] >= heap[child]) {
            break;
          }
          let swapped = heap[parent];
          heap[parent] = heap[child];
          heap[child] = swapped;
          child = parent;
        }
      } else if (slope < heap[0]) {
        heap[0] = slope;
        var parent = 0u;
        loop {
          let left = 2u * parent + 1u;
          if (left >= size) {
            break;
          }
          var larger = left;
          if (left + 1u < size && heap[left + 1u] > heap[left]) {
            larger = left + 1u;
          }
          if (heap[parent] >= heap[larger]) {
            break;
          }
          let swapped = heap[parent];
          heap[parent] = heap[larger];
          heap[larger] = swapped;
          parent = larger;
        }
      }
    }
  }
  var median = heap[0];
  if (pairCount % 2u == 0u && size >= 2u) {
    var secondLargest = heap[1];
    if (size > 2u) {
      secondLargest = max(secondLargest, heap[2]);
    }
    median = 0.5 * (median + secondLargest);
  }
  return median;
}`}var S={NO_PATTERN:0,NEW_HOT:1,CONSECUTIVE_HOT:2,INTENSIFYING_HOT:3,PERSISTENT_HOT:4,DIMINISHING_HOT:5,SPORADIC_HOT:6,OSCILLATING_HOT:7,HISTORICAL_HOT:8,NEW_COLD:9,CONSECUTIVE_COLD:10,INTENSIFYING_COLD:11,PERSISTENT_COLD:12,DIMINISHING_COLD:13,SPORADIC_COLD:14,OSCILLATING_COLD:15,HISTORICAL_COLD:16,UNDETERMINED:17},C={"0.9":1.6448536269514722,"0.95":1.959963984540054,"0.99":2.5758293035489004};function w(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPUEmergingHotSpots target is too short`);let{temporalWindow:n}=e,r=e.radius??0;if(!Number.isFinite(r)||r<0)throw Error(`GPUEmergingHotSpots radius must be a finite number >= 0`);if(!Number.isInteger(n)||n<0)throw Error(`GPUEmergingHotSpots temporalWindow must be an integer >= 0`);let i=e.confidenceLevel??.95;if(!(i>0&&i<1))throw Error(`GPUEmergingHotSpots confidenceLevel must be in (0, 1)`);let a=e.multipleTesting??`benjamini-hochberg`,o=e.criticalZ;if(o!==void 0){if(a!==`none`)throw Error(`GPUEmergingHotSpots criticalZ requires multipleTesting 'none'`);if(!Number.isFinite(o)||o<=0)throw Error(`GPUEmergingHotSpots criticalZ must be a finite number > 0`)}else if(o=C[String(i)],o===void 0){if(a===`none`)throw Error(`GPUEmergingHotSpots confidenceLevel must be 0.9, 0.95 or 0.99`);o=0}let s=e.minimumTrendSliceCount??10;if(!Number.isInteger(s)||s<0)throw Error(`GPUEmergingHotSpots minimumTrendSliceCount must be an integer >= 0`);let c=e.trendSignificanceLevel??1-i;if(!(c>0&&c<1))throw Error(`GPUEmergingHotSpots trendSignificanceLevel must be in (0, 1)`);let l=e.persistentFraction??.9;if(!(l>.5&&l<=1))throw Error(`GPUEmergingHotSpots persistentFraction must be in (0.5, 1]`);return t.set([r,n,o,c,l,s,1-i,+(e.sporadicRequiresFinalSlice===!1)]),t}var T=`GPUEmergingHotSpots`,E=4096,D=256,O=256,k=64,te=64,ne=1.959963984540054,A=2**31-1,j=2**24-1,re=6,M=class{id;props;populationCount;multipleTesting;finalSlice;trendAutocorrelationCorrection;constructor(e){this.id=e.id??`emerging-hot-spots`,this.props=e,this.populationCount=x(this.id,e.output);let n=this.id,{output:r}=e;if(this.multipleTesting=d(n,e.multipleTesting??`benjamini-hochberg`),this.finalSlice=e.finalSlice??`require`,this.trendAutocorrelationCorrection=e.trendAutocorrelationCorrection??`none`,this.finalSlice!==`require`&&this.finalSlice!==`last-valid`)throw Error(`${n} finalSlice must be 'require' or 'last-valid'`);a(n,[`none`,`hamed-rao`,`yue-wang`].includes(this.trendAutocorrelationCorrection),`trendAutocorrelationCorrection must be 'none', 'hamed-rao' or 'yue-wang'`),this.trendAutocorrelationCorrection!==`none`&&o(n,`sliceCount with trendAutocorrelationCorrection`,e.sliceCount,1,63),e.trendCorrectionMaximumLag!==void 0&&o(n,`trendCorrectionMaximumLag`,e.trendCorrectionMaximumLag,1),e.trendCorrectionCriticalZ!==void 0&&s(n,`trendCorrectionCriticalZ float32`,Math.fround(e.trendCorrectionCriticalZ),0);let{gridWidth:c,gridHeight:f,sliceCount:p}=e,g=c!==void 0||f!==void 0;if(g===(e.weights!==void 0))throw Error(`${n} needs exactly one of gridWidth and gridHeight, or weights`);let _=e.selfWeight??1;if(!Number.isFinite(Math.fround(_))||_<0)throw Error(`${n} selfWeight must round to a finite float32 >= 0`);g&&(o(n,`gridWidth`,c??NaN),o(n,`gridHeight`,f??NaN)),o(n,`sliceCount`,p,1,256);let y=e.weights?h(n,e.weights):c*f;if(y>j)throw Error(`${n} cellCount must be at most ${j}`);let b=y*p;if(b>A)throw Error(`${n} cellCount * sliceCount must be below 2^31`);if(o(n,`maximumRadius`,e.maximumRadius??4,0,32),i(e.values,[`float32`,`uint32`],`${n} values`),e.values.length!==b)throw Error(`${n} values length must equal cellCount * sliceCount`);if(i(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<8)throw Error(`${n} parameters is too short`);if(e.mask&&(t(e.mask,`${n} mask`),e.mask.length!==y))throw Error(`${n} mask length must equal the cell count`);if(i(r.giZScores,[`float32`],`${n} giZScores`),r.giZScores.length!==b)throw Error(`${n} giZScores length must equal values length`);for(let[e,t]of[[`pValues`,r.pValues],[`adjustedPValues`,r.adjustedPValues]])if(t&&(i(t,[`float32`],`${n} ${e}`),t.length!==b))throw Error(`${n} ${e} length must equal values length`);if((this.multipleTesting!==`none`||r.pValues||r.adjustedPValues)&&b>=u)throw Error(`${n} multiple testing bins must be less than ${u}`);if(r.lastValidSlice&&(t(r.lastValidSlice,`${n} lastValidSlice`),r.lastValidSlice.length!==y))throw Error(`${n} lastValidSlice length must equal the cell count`);i(r.trendZ,[`float32`],`${n} trendZ`),i(r.trendP,[`float32`],`${n} trendP`),i(r.trendS,[`sint32`],`${n} trendS`),t(r.category,`${n} category`),t(r.hotSliceCount,`${n} hotSliceCount`),t(r.coldSliceCount,`${n} coldSliceCount`);for(let[e,t]of[[`trendZ`,r.trendZ],[`trendP`,r.trendP],[`trendS`,r.trendS],[`category`,r.category],[`hotSliceCount`,r.hotSliceCount],[`coldSliceCount`,r.coldSliceCount]])if(t.length!==y)throw Error(`${n} ${e} length must equal the cell count`);if(r.globalStatistics&&(i(r.globalStatistics,[`float32`],`${n} globalStatistics`),r.globalStatistics.length<4))throw Error(`${n} globalStatistics is too short`);this.populationCount&&(t(this.populationCount,`${n} output.${r.includedCount?`includedCount`:`count`}`),o(n,`output.${r.includedCount?`includedCount`:`count`} length`,this.populationCount.length,1,1)),v(n,r),l(n,[r.giZScores,r.pValues,r.adjustedPValues,r.lastValidSlice,r.trendZ,r.trendP,r.trendS,r.category,r.hotSliceCount,r.coldSliceCount,r.globalStatistics,this.populationCount],[e.values,e.parameters,e.mask,...e.weights?m(e.weights):[]])}getParameterValues(e,t){return w(this._resolveSettings(e),t)}getInferenceMetadata(e){let t=this._resolveSettings(e);return w(t),{alternative:`two-sided`,permutationCount:0,multipleTesting:this.multipleTesting,significanceLevel:1-(t.confidenceLevel??.95),testedRowsDefinition:`valid space-time bins`}}_resolveSettings(e){if(e.multipleTesting!==void 0&&e.multipleTesting!==this.multipleTesting)throw Error(`${this.id} settings.multipleTesting must match the node`);return{...e,multipleTesting:this.multipleTesting}}getCommandNodes(t){let{id:i,props:a}=this;_(t,i,a,{"centered bins":a.values.length*4,moments:24});let{values:o,mask:s,parameters:l,weights:u,sliceCount:d,output:h}=a,{giZScores:v}=h,x=a.gridWidth??1,S=a.gridHeight??1,C=a.selfWeight??1;c(i,t,[o,s,l,...u?m(u):[],v,h.trendZ,h.trendP,h.pValues,h.adjustedPValues,h.lastValidSlice,h.trendS,h.category,h.hotSliceCount,h.coldSliceCount,h.globalStatistics]);let{finalSlice:w,multipleTesting:A}=this,j=A!==`none`,M=j||h.pValues!==void 0||h.adjustedPValues!==void 0,F=a.maximumRadius??4,I=u?g(u):[],L=u?I.reduce((e,t)=>e+t.rowCount,0):x*S,R=L*d;if(R>Math.floor(t.device.limits.maxStorageBufferBindingSize/4))throw Error(`${i} cube exceeds maxStorageBufferBindingSize`);let z=d>=te,B=Math.ceil(R/E),oe=o.format===`uint32`?`u32`:`f32`,V=e(t,`${i}-sum-partials`,`float32`,B),H=e(t,`${i}-count-partials`,`uint32`,B),U=e(t,`${i}-square-partials`,`float32`,B),se=e(t,`${i}-statistics`,`float32`,re),ce=e(t,`${i}-first-deviations`,`float32`,R),W=e(t,`${i}-deviation-partials`,`float32`,B),G=e(t,`${i}-residual-mean`,`float32`,1),K=`
const GRID_WIDTH: u32 = ${x}u;
const GRID_HEIGHT: u32 = ${S}u;
const CELL_COUNT: u32 = ${L}u;
const SLICE_COUNT: u32 = ${d}u;
const BIN_COUNT: u32 = ${R}u;
const BIN_BLOCK: u32 = ${E}u;
const MAXIMUM_RADIUS: f32 = ${F}.0;
const SELF_WEIGHT: f32 = ${n(C)};
${p}
`,q=`${K}
fn readValue(bin: u32) -> f32 {
  return f32(values[valuesOffset + bin]);
}

fn isBinValid(bin: u32, cell: u32) -> bool {
  ${s?`if (mask[maskOffset + cell] == 0u) { return false; }`:``}
  return isFiniteFloat(readValue(bin));
}
`,J=`${K}
fn readDeviation(bin: u32) -> f32 {
  return centeredValues[centeredValuesOffset + bin];
}

fn isBinValid(bin: u32, cell: u32) -> bool {
  return isFiniteFloat(readDeviation(bin));
}
`,le={name:`residualMean`,view:G,type:`f32`,access:`read`},Y=e=>({name:`centeredValues`,view:ce,type:`f32`,access:e}),ue={name:`values`,view:o,type:oe,access:`read`},de=s?[{name:`mask`,view:s,type:`u32`,access:`read`}]:[],X={name:`parameters`,view:l,type:`f32`,access:`read`},Z=e=>({name:`statistics`,view:se,type:`f32`,access:e}),Q=Math.max(1,Math.floor(O/d)),fe=e=>u?`var weighted = 0.0;
    var weightTotal = 0.0;
    var squareTotal = 0.0;
    let bin = cell * SLICE_COUNT + slice;
    // The focal cell has weight SELF_WEIGHT; its CSR row never lists the cell itself.
    if (SELF_WEIGHT != 0.0 && isBinValid(bin, cell)) {
      weighted += SELF_WEIGHT * readDeviation(bin);
      weightTotal += SELF_WEIGHT;
      squareTotal += SELF_WEIGHT * SELF_WEIGHT;
    }
    for (var slot = offsets[offsetsOffset + ${e}]; slot < offsets[offsetsOffset + ${e} + 1u]; slot++) {
      let neighborCell = neighbors[neighborsOffset + slot];
      if (neighborCell >= CELL_COUNT || neighborCell == cell) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      let neighborBin = neighborCell * SLICE_COUNT + slice;
      if (isBinValid(neighborBin, neighborCell)) {
        weighted += weight * readDeviation(neighborBin);
        weightTotal += weight;
        squareTotal += weight * weight;
      }
    }
    instantWeighted[local] = weighted;
    instantWeight[local] = weightTotal;
    instantSquare[local] = squareTotal;`:`var weighted = 0.0;
    var neighborCount = 0u;
    let column = i32(cell % GRID_WIDTH);
    let row = i32(cell / GRID_WIDTH);
    for (var deltaRow = -reach; deltaRow <= reach; deltaRow++) {
      let neighborRow = row + deltaRow;
      if (neighborRow < 0 || neighborRow >= i32(GRID_HEIGHT)) {
        continue;
      }
      for (var deltaColumn = -reach; deltaColumn <= reach; deltaColumn++) {
        let neighborColumn = column + deltaColumn;
        if (neighborColumn < 0 || neighborColumn >= i32(GRID_WIDTH) ||
            f32(deltaRow * deltaRow + deltaColumn * deltaColumn) > radiusSquared) {
          continue;
        }
        let neighborCell = u32(neighborRow) * GRID_WIDTH + u32(neighborColumn);
        let neighborBin = neighborCell * SLICE_COUNT + slice;
        if (isBinValid(neighborBin, neighborCell)) {
          weighted += readDeviation(neighborBin);
          neighborCount++;
        }
      }
    }
    instantWeighted[local] = weighted;
    instantWeight[local] = f32(neighborCount);
    instantSquare[local] = 0.0;`,pe=u?`var weightedSum = 0.0;
    var weightSum = 0.0;
    var squareSum = 0.0;
    for (var windowSlice = firstSlice; windowSlice <= slice; windowSlice++) {
      let slot = cellInGroup * SLICE_COUNT + windowSlice;
      weightedSum += instantWeighted[slot];
      weightSum += instantWeight[slot];
      squareSum += instantSquare[slot];
    }`:`var weightedSum = 0.0;
    var weightSum = 0.0;
    for (var windowSlice = firstSlice; windowSlice <= slice; windowSlice++) {
      let slot = cellInGroup * SLICE_COUNT + windowSlice;
      weightedSum += instantWeighted[slot];
      weightSum += instantWeight[slot];
    }
    let squareSum = weightSum;`,me=()=>`let chunkRowStart = getChunkParameter(36u);
  let chunkRowCount = getChunkParameter(37u);
  let group = index / ${O}u;
  let local = localInvocationIndex;
  let cellInGroup = local / SLICE_COUNT;
  let slice = local % SLICE_COUNT;
  let cell = group * CELLS_PER_GROUP + cellInGroup + chunkRowStart;
  let isActive = index < INVOCATION_COUNT && cellInGroup < CELLS_PER_GROUP && cell < chunkRowStart + chunkRowCount;
  let rawRadius = ${u?`0.0`:`parameters[parametersOffset]`};
  let rawWindow = parameters[parametersOffset + 1u];
  let count = (u32(statistics[statisticsOffset + 4u]) << 16u) | u32(statistics[statisticsOffset + 5u]);
  let variance = statistics[statisticsOffset + 2u];
  let deviation = statistics[statisticsOffset + 3u];
  let parametersValid = isFiniteFloat(rawRadius) && isFiniteFloat(rawWindow) && rawRadius >= 0.0 && rawWindow >= 0.0;
  let radius = min(rawRadius, MAXIMUM_RADIUS);
  let radiusSquared = radius * radius;
  let reach = i32(floor(radius));
  instantWeighted[local] = 0.0;
  instantWeight[local] = 0.0;
  instantSquare[local] = 0.0;
  if (isActive && parametersValid) {
    ${fe(`(cell - chunkRowStart)`)}
  }
  // Every invocation reaches the barrier.
  workgroupBarrier();
  if (isActive) {
    var zScore = getQuietNaN(index);
    if (parametersValid && isBinValid(cell * SLICE_COUNT + slice, cell)) {
      let windowSize = u32(min(floor(rawWindow), f32(SLICE_COUNT - 1u)));
      let firstSlice = slice - min(windowSize, slice);
      ${pe}
      ${u?`let spread = divideBySpatialCount(multiplyBySpatialCount(squareSum, count) - weightSum * weightSum, max(count, 1u) - 1u);`:`let spread = divideBySpatialCount(multiplyBySpatialCount(weightSum, count) - weightSum * weightSum, max(count, 1u) - 1u);`}
      if (count >= 2u && variance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
        // sum_j w_j (x_j - X): the stored deviations from the provisional mean, corrected once by
        // the mean of those deviations.
        let centeredSum = weightedSum - residualMean[residualMeanOffset] * weightSum;
        zScore = centeredSum / (deviation * sqrt(spread));
      }
    }
    giZScores[giZScoresOffset + cell * SLICE_COUNT + slice] = zScore;
  }`,he=h.pValues??(M?e(t,`${i}-p-values`,`float32`,R):void 0),$=h.adjustedPValues??(M?e(t,`${i}-adjusted-p-values`,`float32`,R):void 0),ge=()=>[r(t,{id:`${i}-p-values`,operation:T,variant:`two-sided-p-values`,bindings:[{name:`giZScores`,view:v,type:`f32`,access:`read`},{name:`pValues`,view:he,type:`f32`,access:`read_write`}],invocationCount:R,declarations:p,body:`let zScore = giZScores[giZScoresOffset + index];
  // erfc(|z| / sqrt(2)); NaN z (missing bin) stays NaN and is excluded from the family.
  pValues[pValuesOffset + index] = select(
    getQuietNaN(index),
    // The shared analytic p-value, floored at the smallest normal f32 instead of reporting exact
    // zeros that would tie in the correction.
    getTwoSidedPValue(zScore),
    isFiniteFloat(zScore)
  );`}),...new f({id:`${i}-adjustment`,pValues:he,method:A,output:{adjustedPValues:$}}).getCommandNodes(t)],_e=[];return _e.push(r(t,{id:`${i}-block-sums`,operation:T,variant:`block-sums`,bindings:[ue,...de,{name:`sumPartials`,view:V,type:`f32`,access:`read_write`},{name:`countPartials`,view:H,type:`u32`,access:`read_write`}],workgroupSize:D,invocationCount:B*D,guardIndex:!1,declarations:`${q}
var<workgroup> partialSums: array<f32, ${D}>;
var<workgroup> partialCounts: array<u32, ${D}>;`,body:`let blockIndex = index / ${D}u;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  var count = 0u;
  if (isInRange) {
    let begin = blockIndex * BIN_BLOCK;
    let end = min(begin + BIN_BLOCK, BIN_COUNT);
    for (var bin = begin + localInvocationIndex; bin < end; bin += ${D}u) {
      if (isBinValid(bin, bin / SLICE_COUNT)) {
        sum += readValue(bin);
        count++;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = ${D/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    sumPartials[sumPartialsOffset + blockIndex] = partialSums[0];
    countPartials[countPartialsOffset + blockIndex] = partialCounts[0];
  }`}),r(t,{id:`${i}-mean`,operation:T,variant:`mean`,bindings:[{name:`sumPartials`,view:V,type:`f32`,access:`read`},{name:`countPartials`,view:H,type:`u32`,access:`read`},Z(`read_write`)],workgroupSize:D,invocationCount:D,guardIndex:!1,declarations:`${b}
const BLOCK_COUNT: u32 = ${B}u;
var<workgroup> partialSums: array<f32, ${D}>;
var<workgroup> partialCounts: array<u32, ${D}>;`,body:`var sum = 0.0;
  var count = 0u;
  for (var blockIndex = localInvocationIndex; blockIndex < BLOCK_COUNT; blockIndex += ${D}u) {
    sum += sumPartials[sumPartialsOffset + blockIndex];
    count += countPartials[countPartialsOffset + blockIndex];
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = ${D/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    count = partialCounts[0];
    statistics[statisticsOffset] = f32(count);
    statistics[statisticsOffset + 4u] = f32(count >> 16u);
    statistics[statisticsOffset + 5u] = f32(count & 65535u);
    statistics[statisticsOffset + 1u] = select(0.0, divideBySpatialCount(partialSums[0], count), count > 0u);
    statistics[statisticsOffset + 2u] = 0.0;
    statistics[statisticsOffset + 3u] = 0.0;
  }`}),r(t,{id:`${i}-block-deviations`,operation:T,variant:`block-deviations`,bindings:[ue,...de,Z(`read`),Y(`read_write`),{name:`deviationPartials`,view:W,type:`f32`,access:`read_write`}],workgroupSize:D,invocationCount:B*D,guardIndex:!1,declarations:`${q}
var<workgroup> partialSums: array<f32, ${D}>;`,body:`let blockIndex = index / ${D}u;
  let isInRange = index < INVOCATION_COUNT;
  let provisionalMean = statistics[statisticsOffset + 1u];
  var sum = 0.0;
  if (isInRange) {
    let begin = blockIndex * BIN_BLOCK;
    let end = min(begin + BIN_BLOCK, BIN_COUNT);
    for (var bin = begin + localInvocationIndex; bin < end; bin += ${D}u) {
      let valid = isBinValid(bin, bin / SLICE_COUNT);
      let deviation = readValue(bin) - provisionalMean;
      centeredValues[centeredValuesOffset + bin] = select(getQuietNaN(bin), deviation, valid);
      sum += select(0.0, deviation, valid);
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = ${D/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    deviationPartials[deviationPartialsOffset + blockIndex] = partialSums[0];
  }`}),r(t,{id:`${i}-mean-correction`,operation:T,variant:`mean-correction`,bindings:[{name:`deviationPartials`,view:W,type:`f32`,access:`read`},Z(`read_write`),{name:`residualMean`,view:G,type:`f32`,access:`read_write`}],workgroupSize:D,invocationCount:D,guardIndex:!1,declarations:`${b}
const BLOCK_COUNT: u32 = ${B}u;
var<workgroup> partialSums: array<f32, ${D}>;`,body:`var sum = 0.0;
  for (var blockIndex = localInvocationIndex; blockIndex < BLOCK_COUNT; blockIndex += ${D}u) {
    sum += deviationPartials[deviationPartialsOffset + blockIndex];
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = ${D/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let count = (u32(statistics[statisticsOffset + 4u]) << 16u) | u32(statistics[statisticsOffset + 5u]);
    // Mean of the first-pass deviations: added to the reported mean and subtracted from every
    // stored deviation by the next kernel.
    let residual = select(0.0, divideBySpatialCount(partialSums[0], count), count > 0u);
    statistics[statisticsOffset + 1u] += residual;
    residualMean[residualMeanOffset] = residual;
  }`}),r(t,{id:`${i}-block-squares`,operation:T,variant:`block-squares`,bindings:[Y(`read`),le,{name:`squarePartials`,view:U,type:`f32`,access:`read_write`}],workgroupSize:D,invocationCount:B*D,guardIndex:!1,declarations:`${J}
var<workgroup> partialSums: array<f32, ${D}>;`,body:`let blockIndex = index / ${D}u;
  let isInRange = index < INVOCATION_COUNT;
  let residual = residualMean[residualMeanOffset];
  var sum = 0.0;
  if (isInRange) {
    let begin = blockIndex * BIN_BLOCK;
    let end = min(begin + BIN_BLOCK, BIN_COUNT);
    for (var bin = begin + localInvocationIndex; bin < end; bin += ${D}u) {
      if (isBinValid(bin, bin / SLICE_COUNT)) {
        let centered = readDeviation(bin) - residual;
        sum += centered * centered;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = ${D/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    squarePartials[squarePartialsOffset + blockIndex] = partialSums[0];
  }`}),r(t,{id:`${i}-deviation`,operation:T,variant:`deviation`,bindings:[{name:`squarePartials`,view:U,type:`f32`,access:`read`},Z(`read_write`)],workgroupSize:D,invocationCount:D,guardIndex:!1,declarations:`${b}
const BLOCK_COUNT: u32 = ${B}u;
var<workgroup> partialSums: array<f32, ${D}>;`,body:`var sum = 0.0;
  for (var blockIndex = localInvocationIndex; blockIndex < BLOCK_COUNT; blockIndex += ${D}u) {
    sum += squarePartials[squarePartialsOffset + blockIndex];
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = ${D/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let count = (u32(statistics[statisticsOffset + 4u]) << 16u) | u32(statistics[statisticsOffset + 5u]);
    let variance = select(0.0, divideBySpatialCount(partialSums[0], count), count > 0u);
    statistics[statisticsOffset + 2u] = variance;
    statistics[statisticsOffset + 3u] = sqrt(variance);
  }`}),...h.globalStatistics||this.populationCount?[r(t,{id:`${i}-publish-moments`,operation:T,invocationCount:h.globalStatistics?4:1,bindings:[Z(`read`),...h.globalStatistics?[{name:`globalStatistics`,view:h.globalStatistics,type:`f32`,access:`read_write`}]:[],...this.populationCount?[{name:`exactCount`,view:this.populationCount,type:`u32`,access:`read_write`}]:[]],body:`${h.globalStatistics?`globalStatistics[globalStatisticsOffset + index] = statistics[statisticsOffset + index];`:``}
          ${this.populationCount?`if (index == 0u) { exactCount[exactCountOffset] = (u32(statistics[statisticsOffset + 4u]) << 16u) | u32(statistics[statisticsOffset + 5u]); }`:``}`})]:[],...(u?I:[void 0]).map(e=>y(t,{id:`${i}-gi-star${e?.idSuffix??``}`,operation:T,variant:u?`gi-star-weights`:`gi-star`,bindings:[Y(`read`),le,X,...e?[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:e.weights.weights,type:`f32`,access:`read`}]:[],Z(`read`),{name:`giZScores`,view:v,type:`f32`,access:`read_write`}],workgroupSize:O,invocationCount:Math.ceil((e?.rowCount??L)/Q)*O,parameterValues:[e?.rowStart??0,e?.rowCount??L],declarations:`${J}
const CELLS_PER_GROUP: u32 = ${Q}u;
var<workgroup> instantWeighted: array<f32, ${O}>;
var<workgroup> instantWeight: array<f32, ${O}>;
var<workgroup> instantSquare: array<f32, ${O}>;`,body:me()})),...M?ge():[],r(t,{id:`${i}-slice-counts`,operation:T,variant:`slice-counts`,bindings:[X,{name:`giZScores`,view:v,type:`f32`,access:`read`},...j?[{name:`adjustedPValues`,view:$,type:`f32`,access:`read`}]:[],{name:`hotSliceCount`,view:h.hotSliceCount,type:`u32`,access:`read_write`},{name:`coldSliceCount`,view:h.coldSliceCount,type:`u32`,access:`read_write`},...h.lastValidSlice?[{name:`lastValidSlice`,view:h.lastValidSlice,type:`u32`,access:`read_write`}]:[]],invocationCount:L,declarations:`const SLICE_COUNT: u32 = ${d}u;
${p}
${N(j)}`,body:`let base = index * SLICE_COUNT;
  ${P({valueExpression:`giZScores[giZScoresOffset + base + slice]`,adjustedExpression:`adjustedPValues[adjustedPValuesOffset + base + slice]`,usesAdjustment:j,collectSeries:!1})}
  hotSliceCount[hotSliceCountOffset + index] = hot;
  coldSliceCount[coldSliceCountOffset + index] = cold;
  ${h.lastValidSlice?`lastValidSlice[lastValidSliceOffset + index] = lastValid;`:``}`}),r(t,{id:`${i}-mann-kendall`,operation:T,variant:z?`mann-kendall-cooperative`:`mann-kendall-classify`,bindings:[X,{name:`giZScores`,view:v,type:`f32`,access:`read`},...j?[{name:`adjustedPValues`,view:$,type:`f32`,access:`read`}]:[],{name:`trendZ`,view:h.trendZ,type:`f32`,access:`read_write`},{name:`trendP`,view:h.trendP,type:`f32`,access:`read_write`},{name:`trendS`,view:h.trendS,type:`i32`,access:`read_write`},{name:`category`,view:h.category,type:`u32`,access:`read_write`}],workgroupSize:z?k:void 0,invocationCount:z?L*k:L,guardIndex:!z,declarations:z?`const CELL_COUNT: u32 = ${L}u;
const SLICE_COUNT: u32 = ${d}u;
const MANN_KENDALL_LANES: u32 = ${k}u;
var<workgroup> partialStatistics: array<i32, ${k}>;
var<workgroup> partialTieTerms: array<u32, ${k}>;
var<workgroup> sharedSeries: array<f32, ${d}>;
var<workgroup> tieCandidates: array<u32, ${d}>;
var<workgroup> equalBeforeBits: array<atomic<u32>, ${Math.ceil(d/32)}>;
${p}
${N(j)}`:`const SLICE_COUNT: u32 = ${d}u;
${this.trendAutocorrelationCorrection===`none`?``:ee(d)}
${p}
${N(j)}`,body:z?`let cell = index / MANN_KENDALL_LANES;
  let lane = localInvocationIndex;
  let isActiveCell = cell < CELL_COUNT;
  let base = cell * SLICE_COUNT;
  for (var slice = lane; slice < SLICE_COUNT; slice += MANN_KENDALL_LANES) {
    var value = getQuietNaN(index);
    if (isActiveCell) {
      value = giZScores[giZScoresOffset + base + slice];
    }
    sharedSeries[slice] = value;
    tieCandidates[slice] = 0u;
  }
  for (var word = lane; word < ${Math.ceil(d/32)}u; word += MANN_KENDALL_LANES) {
    atomicStore(&equalBeforeBits[word], 0u);
  }
  workgroupBarrier();
  var laneStatistic = 0;
  for (var first = lane; first < SLICE_COUNT; first += MANN_KENDALL_LANES) {
    let firstValue = sharedSeries[first];
    if (isFiniteFloat(firstValue)) {
      var groupSize = 1u;
      for (var second = first + 1u; second < SLICE_COUNT; second++) {
        let secondValue = sharedSeries[second];
        if (isFiniteFloat(secondValue)) {
          laneStatistic += select(select(0, -1, secondValue < firstValue), 1, secondValue > firstValue);
          if (secondValue == firstValue) {
            groupSize++;
            atomicOr(&equalBeforeBits[second / 32u], 1u << (second % 32u));
          }
        }
      }
      if (groupSize > 1u) {
        tieCandidates[first] = groupSize * (groupSize - 1u) * (2u * groupSize + 5u);
      }
    }
  }
  partialStatistics[lane] = laneStatistic;
  workgroupBarrier();
  var laneTieTerm = 0u;
  for (var first = lane; first < SLICE_COUNT; first += MANN_KENDALL_LANES) {
    let hasEqualBefore = (atomicLoad(&equalBeforeBits[first / 32u]) & (1u << (first % 32u))) != 0u;
    if (!hasEqualBefore) {
      laneTieTerm += tieCandidates[first];
    }
  }
  partialTieTerms[lane] = laneTieTerm;
  workgroupBarrier();
  for (var stride = MANN_KENDALL_LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      partialStatistics[lane] += partialStatistics[lane + stride];
      partialTieTerms[lane] += partialTieTerms[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u && isActiveCell) {
  let statistic = partialStatistics[0];
  let tieTerm = partialTieTerms[0];
  ${P({valueExpression:`sharedSeries[slice]`,adjustedExpression:`adjustedPValues[adjustedPValuesOffset + base + slice]`,usesAdjustment:j,collectSeries:!1})}
  ${ie({cellName:`cell`,finalSlice:w})}
  }`:ae({sliceCount:d,usesAdjustment:j,finalSlice:w,trendCorrection:this.trendAutocorrelationCorrection,maximumLag:a.trendCorrectionMaximumLag??0,criticalZ:a.trendCorrectionCriticalZ??ne})})),_e}};function N(e){return e?`fn getSliceState(z: f32, adjustedP: f32, level: f32) -> u32 {
  if (adjustedP <= level) {
    if (z > 0.0) {
      return 1u;
    }
    if (z < 0.0) {
      return 2u;
    }
  }
  return 0u;
}`:`fn getSliceState(z: f32, criticalZ: f32) -> u32 {
  if (z >= criticalZ) {
    return 1u;
  }
  if (z <= -criticalZ) {
    return 2u;
  }
  return 0u;
}`}function P(e){let t=e.usesAdjustment?`getSliceState(value, ${e.adjustedExpression}, significanceLevel)`:`getSliceState(value, criticalZ)`;return`let criticalZ = parameters[parametersOffset + 2u];
  let significanceLevel = parameters[parametersOffset + 6u];
  var valid = 0u;
  var hot = 0u;
  var cold = 0u;
  var trailingHot = 0u;
  var trailingCold = 0u;
  var finalState = 0u;
  var lastValid = 0xffffffffu;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = ${e.valueExpression};
    if (isFiniteFloat(value)) {
      ${e.collectSeries?`series[valid] = value;`:``}
      valid++;
      lastValid = slice;
      let state = ${t};
      if (state == 1u) {
        hot++;
        trailingHot++;
        trailingCold = 0u;
        finalState = 1u;
      } else if (state == 2u) {
        cold++;
        trailingCold++;
        trailingHot = 0u;
        finalState = 2u;
      } else {
        trailingHot = 0u;
        trailingCold = 0u;
        finalState = 0u;
      }
    }
  }`}function ie(e){let{cellName:t}=e;return`let trendLevel = parameters[parametersOffset + 3u];
  let fraction = parameters[parametersOffset + 4u];
  let minimumTrendSliceCount = parameters[parametersOffset + 5u];
  let isLegacySporadic = parameters[parametersOffset + 7u] != 0.0;
  var numerator = 0u;
  if (valid >= 2u) {
    numerator = valid * (valid - 1u) * (2u * valid + 5u) - tieTerm;
  }
  var trendZScore = 0.0;
  var trendPValue = 1.0;
  if (numerator > 0u) {
    let deviation = sqrt(f32(numerator) / 18.0${e.applyVarianceCorrection?` * varianceCorrection`:``});
    if (statistic > 0) {
      trendZScore = f32(statistic - 1) / deviation;
    } else if (statistic < 0) {
      trendZScore = f32(statistic + 1) / deviation;
    }
    trendPValue = getTwoSidedPValue(trendZScore);
  }
  // Too few finite slices for a Mann-Kendall decision: no trend z or p, and no trend category.
  let isTrendKnown = f32(valid) >= minimumTrendSliceCount;
  if (!isTrendKnown) {
    trendZScore = getQuietNaN(${t});
    trendPValue = getQuietNaN(${t});
  }
  trendZ[trendZOffset + ${t}] = trendZScore;
  trendP[trendPOffset + ${t}] = trendPValue;
  trendS[trendSOffset + ${t}] = statistic;
  let threshold = fraction * f32(valid);
  let hotPersistent = f32(hot) >= threshold;
  let coldPersistent = f32(cold) >= threshold;
  let trendSignificant = trendPValue <= trendLevel;
  let trendUp = trendSignificant && trendZScore > 0.0;
  let trendDown = trendSignificant && trendZScore < 0.0;
  var result = 0u;
  if (valid > 0u) {
    if (finalState == 1u) {
      if (hot == 1u) {
        result = 1u;
      } else if (trailingHot >= 2u && trailingHot == hot && !hotPersistent) {
        result = 2u;
      } else if (hotPersistent) {
        result = select(17u, select(select(4u, 5u, trendDown), 3u, trendUp), isTrendKnown);
      } else {
        result = select(7u, 6u, cold == 0u);
      }
    } else if (finalState == 2u) {
      if (cold == 1u) {
        result = 9u;
      } else if (trailingCold >= 2u && trailingCold == cold && !coldPersistent) {
        result = 10u;
      } else if (coldPersistent) {
        result = select(17u, select(select(12u, 13u, trendUp), 11u, trendDown), isTrendKnown);
      } else {
        result = select(15u, 14u, hot == 0u);
      }
    } else if (hot > 0u && hotPersistent) {
      result = 8u;
    } else if (cold > 0u && coldPersistent) {
      result = 16u;
    } else if (isLegacySporadic && hot > 0u && cold == 0u) {
      result = 6u;
    } else if (isLegacySporadic && cold > 0u && hot == 0u) {
      result = 14u;
    }
    ${e.finalSlice===`require`?`// A missing final slice makes the pattern undecidable instead of silently shifting it.
    if (lastValid != SLICE_COUNT - 1u) {
      result = 17u;
    }`:``}
  }
  category[categoryOffset + ${t}] = result;`}function ae(e){let{sliceCount:t,trendCorrection:n}=e,r=n!==`none`;return`let base = index * SLICE_COUNT;
  var series: array<f32, SLICE_COUNT>;
  var hasEqualBefore: array<u32, ${Math.ceil(t/32)}>;
  ${P({valueExpression:`giZScores[giZScoresOffset + base + slice]`,adjustedExpression:`adjustedPValues[adjustedPValuesOffset + base + slice]`,usesAdjustment:e.usesAdjustment,collectSeries:!0})}
  var statistic = 0;
  var tieTerm = 0u;
  for (var first = 0u; first < valid; first++) {
    let firstValue = series[first];
    var groupSize = 1u;
    for (var second = first + 1u; second < valid; second++) {
      let secondValue = series[second];
      statistic += select(select(0, -1, secondValue < firstValue), 1, secondValue > firstValue);
      if (secondValue == firstValue) {
        groupSize++;
        hasEqualBefore[second / 32u] |= 1u << (second % 32u);
      }
    }
    let isFirstMember = (hasEqualBefore[first / 32u] & (1u << (first % 32u))) == 0u;
    if (isFirstMember && groupSize > 1u) {
      tieTerm += groupSize * (groupSize - 1u) * (2u * groupSize + 5u);
    }
  }
  ${r?F(t,n,e.maximumLag,e.criticalZ):``}
  ${ie({cellName:`index`,finalSlice:e.finalSlice,applyVarianceCorrection:r})}`}function F(e,t,r,i){let a=t===`hamed-rao`,o=a?`  var ranks: array<f32, ${e}>;
    for (var first = 0u; first < valid; first++) {
      var lower = 0u;
      var equal = 0u;
      for (var second = 0u; second < valid; second++) {
        if (detrended[second] < detrended[first]) {
          lower++;
        } else if (detrended[second] == detrended[first]) {
          equal++;
        }
      }
      ranks[first] = f32(lower) + 0.5 * f32(equal + 1u);
    }`:``,s=a?`ranks`:`detrended`,c=a?`  let lagFilter = ${n(i)} / sqrt(lengthValue);
      if (abs(correlation) > lagFilter) {
        let gap = lengthValue - f32(lag);
        lagSum += gap * (gap - 1.0) * (gap - 2.0) * correlation;
      }`:`  lagSum += (1.0 - f32(lag) / lengthValue) * correlation;`,l=a?`2.0 / (lengthValue * (lengthValue - 1.0) * (lengthValue - 2.0))`:`2.0`;return`var varianceCorrection = 1.0;
  if (valid >= 3u) {
    let lengthValue = f32(valid);
    let slope = getTheilSenSlope(&series, valid);
    var detrended: array<f32, ${e}>;
    for (var row = 0u; row < valid; row++) {
      detrended[row] = series[row] - slope * f32(row + 1u);
    }
    ${o}
    var seriesMean = 0.0;
    for (var row = 0u; row < valid; row++) {
      seriesMean += ${s}[row];
    }
    seriesMean /= lengthValue;
    var variance = 0.0;
    for (var row = 0u; row < valid; row++) {
      variance += (${s}[row] - seriesMean) * (${s}[row] - seriesMean);
    }
    if (variance > 0.0) {
      var lagSum = 0.0;
      let lastLag = ${r>0?`min(${r}u, valid - 1u)`:`valid - 1u`};
      for (var lag = 1u; lag <= lastLag; lag++) {
        var covariance = 0.0;
        for (var row = 0u; row + lag < valid; row++) {
          covariance += (${s}[row] - seriesMean) * (${s}[row + lag] - seriesMean);
        }
        let correlation = covariance / variance;
        ${c}
      }
      let factor = 1.0 + ${l} * lagSum;
      if (factor > 0.0) {
        varianceCorrection = factor;
      }
    }
  }`}export{S as n,w as r,M as t};
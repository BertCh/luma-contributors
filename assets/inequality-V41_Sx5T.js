import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-sort-xRm8l5yn.js";import{l as n,o as r}from"./gpu-command-graph-BdzdmV6u.js";import{t as i}from"./gpu-dispatch-utils-CbRzFAgT.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,S as s,T as c,c as l,f as u,p as d,u as f,v as p,y as m}from"./gpu-contributor-utils-BIwLQON5.js";import{a as h,i as g}from"./graph-chunk-dispatch-zBt8k2-o.js";import{c as _,i as v,r as y}from"./wgsl-common-DfyXiYB0.js";import{t as b}from"./scaled-quotient-wgsl-UDyhQag6.js";import{i as x,t as S}from"./graph-chunk-groups-BB8ACpQO.js";import{n as C}from"./sorted-segment-sums-Dvmm6OdR.js";import{n as w}from"./sorted-segment-offsets-ChmOtA9F.js";import{t as T}from"./define-run-B-brkCe4.js";import{t as E}from"./workplace-rows-CYMTcZUq.js";import{C as D,d as O}from"./helpers-DPAd8TjN.js";import{Pc as k}from"./entry-preview-D70TdSbK.js";function A(e){return f(e,44)}function j(e){return f(e,4)}function M(e={},t=new Float32Array(4)){if(t.length<4)throw Error(`target must hold 4 values`);let n=e.epsilon??1,r=e.palmaTopShare??.1,i=e.palmaBottomShare??.4;if(!Number.isFinite(n)||n<0||n>32)throw Error(`epsilon must be in [0, 32]`);for(let[e,t]of[[`palmaTopShare`,r],[`palmaBottomShare`,i]])if(!(t>0&&t<=1))throw Error(`${e} must be in (0, 1]`);return t[0]=n,t[1]=r,t[2]=i,t[3]=0,t}var N={MEAN:0,GINI:1,THEIL_T:2,THEIL_L:3,ATKINSON:4,HOOVER:5,PALMA:6,TOTAL_WEIGHT:7,TOTAL_INCOME:8,WEIGHTED_X_LOG_X:9},P=`
${b}
${y}

${_}

${v}

fn interpolateLorenz(cut: f32, previousP: f32, nextP: f32, previousL: f32, nextL: f32) -> f32 {
  if (cut >= nextP) {
    return nextL;
  }
  return previousL + getScaledQuotient(cut - previousP, nextP - previousP, 1.0, 0) * (nextL - previousL);
}
`;function F(e){return`let value = values[valuesOffset + index];
  let zone = zoneIds[zoneIdsOffset + index];
  var isValid = isFiniteBits(value) && value >= 0.0 && zone < ZONE_COUNT;
  ${e.hasMask?`isValid = isValid && rowMask[rowMaskOffset + index] != 0u;`:``}
  ${e.hasWeights?`let weight = weights[weightsOffset + index];
  isValid = isValid && isFiniteBits(weight) && weight > 0.0;`:``}
  // Valid keys are below 0xff800000, so invalid rows sort after every valid row; -0 sorts as +0.
  valueKeys[valueKeysOffset + index] = select(0xffffffffu, getOrderedKey(select(value, 0.0, value == 0.0)), isValid);
  rowIndices[rowIndicesOffset + index] = index;`}var I=`let row = sortedRows[sortedRowsOffset + index];
  let isValid = sortedValueKeys[sortedValueKeysOffset + index] != 0xffffffffu;
  zoneKeys[zoneKeysOffset + index] = select(ZONE_COUNT, zoneIds[zoneIdsOffset + row], isValid);`;function L(e){return`let zone = index;
  var searchLow = 0u;
  var searchHigh = ROW_COUNT;
  while (searchLow < searchHigh) {
    let middle = (searchLow + searchHigh) / 2u;
    if (sortedZones[sortedZonesOffset + middle] < zone) {
      searchLow = middle + 1u;
    } else {
      searchHigh = middle;
    }
  }
  let rangeStart = searchLow;
  searchHigh = ROW_COUNT;
  while (searchLow < searchHigh) {
    let middle = (searchLow + searchHigh) / 2u;
    if (sortedZones[sortedZonesOffset + middle] <= zone) {
      searchLow = middle + 1u;
    } else {
      searchHigh = middle;
    }
  }
  let rangeEnd = searchLow;
  let count = rangeEnd - rangeStart;
  counts[countsOffset + zone] = count;
  let nan = getNaN();
  let base = statsOffset + zone * STATS_STRIDE;
  let knotBase = lorenzOffset + zone * KNOT_COUNT;
  for (var slot = 0u; slot < STATS_STRIDE; slot++) {
    stats[base + slot] = nan;
  }
  for (var knot = 0u; knot < KNOT_COUNT; knot++) {
    lorenz[knotBase + knot] = nan;
  }
  stats[base + 7u] = 0.0;
  stats[base + 8u] = 0.0;
  stats[base + 9u] = 0.0;
  if (count == 0u) {
    return;
  }

  // Pass 1: totals in sorted order.
  var totalWeight = 0.0;
  var totalIncome = 0.0;
  var weightedXLogX = 0.0;
  var hasZero = false;
  for (var position = rangeStart; position < rangeEnd; position++) {
    let row = sortedRows[sortedRowsOffset + position];
    let x = values[valuesOffset + row];
    let w = ${e};
    totalWeight = totalWeight + w;
    totalIncome = totalIncome + w * x;
    if (x > 0.0) {
      weightedXLogX = weightedXLogX + w * x * log(x);
    } else {
      hasZero = true;
    }
  }
  let mean = getScaledQuotient(totalIncome, totalWeight, 1.0, 0);
  stats[base] = mean;
  stats[base + 7u] = totalWeight;
  stats[base + 8u] = totalIncome;
  stats[base + 9u] = weightedXLogX;
  if (!(totalIncome > 0.0)) {
    return;
  }

  // Pass 2: Lorenz walk and the mean-relative sums, in the same order.
  let epsilon = params[paramsOffset];
  let palmaTop = params[paramsOffset + 1u];
  let palmaBottom = params[paramsOffset + 2u];
  let power = 1.0 - epsilon;
  let usePower = epsilon != 1.0;
  let palmaBottomTarget = palmaBottom;
  let palmaTopTarget = 1.0 - palmaTop;
  var cumulativeWeight = 0.0;
  var cumulativeIncome = 0.0;
  var previousP = 0.0;
  var previousL = 0.0;
  var trapezoidArea = 0.0;
  var sumTheilT = 0.0;
  var sumLog = 0.0;
  var sumPower = 0.0;
  var sumAbsolute = 0.0;
  var nextKnot = 1u;
  var bottomL = nan;
  var topCutL = nan;
  var isBottomDone = false;
  var isTopDone = false;
  lorenz[knotBase] = 0.0;
  for (var position = rangeStart; position < rangeEnd; position++) {
    let row = sortedRows[sortedRowsOffset + position];
    let x = values[valuesOffset + row];
    let w = ${e};
    cumulativeWeight = cumulativeWeight + w;
    cumulativeIncome = cumulativeIncome + w * x;
    let nextP = getScaledQuotient(cumulativeWeight, totalWeight, 1.0, 0);
    let nextL = getScaledQuotient(cumulativeIncome, totalIncome, 1.0, 0);
    trapezoidArea = trapezoidArea + (nextP - previousP) * (previousL + nextL);
    sumAbsolute = sumAbsolute + w * abs(x - mean);
    if (x > 0.0) {
      let logRatio = log(getScaledQuotient(x, mean, 1.0, 0));
      sumTheilT = sumTheilT + w * (getScaledQuotient(x, mean, 1.0, 0)) * logRatio;
      sumLog = sumLog + w * logRatio;
      if (usePower) {
        sumPower = sumPower + w * exp(power * logRatio);
      }
    }
    while (nextKnot < KNOT_COUNT && f32(nextKnot) / f32(KNOT_COUNT - 1u) <= nextP) {
      lorenz[knotBase + nextKnot] = interpolateLorenz(f32(nextKnot) / f32(KNOT_COUNT - 1u), previousP, nextP, previousL, nextL);
      nextKnot = nextKnot + 1u;
    }
    if (!isBottomDone && palmaBottomTarget <= nextP) {
      bottomL = interpolateLorenz(palmaBottomTarget, previousP, nextP, previousL, nextL);
      isBottomDone = true;
    }
    if (!isTopDone && palmaTopTarget <= nextP) {
      topCutL = interpolateLorenz(palmaTopTarget, previousP, nextP, previousL, nextL);
      isTopDone = true;
    }
    previousP = nextP;
    previousL = nextL;
  }
  // Rounding can leave the last cumulative share a hair below a cut of exactly 1.
  while (nextKnot < KNOT_COUNT) {
    lorenz[knotBase + nextKnot] = previousL;
    nextKnot = nextKnot + 1u;
  }
  if (!isBottomDone) {
    bottomL = previousL;
  }
  if (!isTopDone) {
    topCutL = previousL;
  }

  stats[base + 1u] = 1.0 - trapezoidArea;
  stats[base + 2u] = getScaledQuotient(sumTheilT, totalWeight, 1.0, 0);
  stats[base + 3u] = select(-getScaledQuotient(sumLog, totalWeight, 1.0, 0), nan, hasZero);
  var atkinson = nan;
  if (!usePower) {
    if (!hasZero) {
      atkinson = 1.0 - exp(getScaledQuotient(sumLog, totalWeight, 1.0, 0));
    }
  } else if (power > 0.0 || !hasZero) {
    let powerMean = getScaledQuotient(sumPower, totalWeight, 1.0, 0);
    atkinson = 1.0 - pow(powerMean, 1.0 / power);
  }
  stats[base + 4u] = select(nan, atkinson, isFiniteBits(atkinson));
  stats[base + 5u] = getScaledQuotient(sumAbsolute, totalIncome, 2.0, 0);
  stats[base + 6u] = select(nan, getScaledQuotient(1.0 - topCutL, bottomL, 1.0, 0), bottomL > 0.0);`}var R=`
var<workgroup> laneFirst: array<f32, 256>;
var<workgroup> laneSecond: array<f32, 256>;
var<workgroup> laneThird: array<f32, 256>;
var<workgroup> laneIncluded: array<u32, 256>;`,ee=`let lane = localInvocationIndex;
  var laneWeight = 0.0;
  var laneIncome = 0.0;
  var laneXLogX = 0.0;
  var laneCount = 0u;
  for (var zone = lane; zone < ZONE_COUNT; zone += 256u) {
    let zoneCount = counts[countsOffset + zone];
    if (zoneCount == 0u) {
      continue;
    }
    let base = statsOffset + zone * STATS_STRIDE;
    laneWeight = laneWeight + stats[base + 7u];
    laneIncome = laneIncome + stats[base + 8u];
    laneXLogX = laneXLogX + stats[base + 9u];
    laneCount = laneCount + zoneCount;
  }
  laneFirst[lane] = laneWeight;
  laneSecond[lane] = laneIncome;
  laneThird[lane] = laneXLogX;
  laneIncluded[lane] = laneCount;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneFirst[lane] = laneFirst[lane] + laneFirst[lane + stride];
      laneSecond[lane] = laneSecond[lane] + laneSecond[lane + stride];
      laneThird[lane] = laneThird[lane] + laneThird[lane + stride];
      laneIncluded[lane] = laneIncluded[lane] + laneIncluded[lane + stride];
    }
    workgroupBarrier();
  }
  // Every lane reads the totals, then all lanes pass a barrier before the arrays are reused.
  let totalWeight = laneFirst[0];
  let totalIncome = laneSecond[0];
  let weightedXLogX = laneThird[0];
  let includedCount = laneIncluded[0];
  workgroupBarrier();
  // Lanes contribute nothing to pass B when the pooled income is not positive.
  let hasIncome = totalIncome > 0.0;
  var laneBetween = 0.0;
  var laneWithin = 0.0;
  for (var zone = lane; zone < ZONE_COUNT; zone += 256u) {
    if (!hasIncome || counts[countsOffset + zone] == 0u) {
      continue;
    }
    let base = statsOffset + zone * STATS_STRIDE;
    let zoneIncome = stats[base + 8u];
    if (!(zoneIncome > 0.0)) {
      continue;
    }
    let incomeShare = getScaledQuotient(zoneIncome, totalIncome, 1.0, 0);
    let populationShare = getScaledQuotient(stats[base + 7u], totalWeight, 1.0, 0);
    laneBetween = laneBetween + incomeShare * log(getScaledQuotient(incomeShare, populationShare, 1.0, 0));
    laneWithin = laneWithin + incomeShare * stats[base + 2u];
  }
  laneFirst[lane] = laneBetween;
  laneSecond[lane] = laneWithin;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneFirst[lane] = laneFirst[lane] + laneFirst[lane + stride];
      laneSecond[lane] = laneSecond[lane] + laneSecond[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let nan = getNaN();
    summary[summaryOffset + 4u] = f32(includedCount);
    summary[summaryOffset + 5u] = select(nan, getScaledQuotient(totalIncome, totalWeight, 1.0, 0), includedCount != 0u);
    summary[summaryOffset + 6u] = totalWeight;
    summary[summaryOffset + 7u] = totalIncome;
    if (!hasIncome) {
      summary[summaryOffset] = nan;
      summary[summaryOffset + 1u] = nan;
      summary[summaryOffset + 2u] = nan;
    } else {
      // The direct x*log(x) moment can overflow while normalized Theil terms remain finite.
      var pooledTheil = laneFirst[0] + laneSecond[0];
      if (isFiniteBits(weightedXLogX)) {
        pooledTheil = getScaledQuotient(weightedXLogX, totalIncome, 1.0, 0) - log(getScaledQuotient(totalIncome, totalWeight, 1.0, 0));
      }
      summary[summaryOffset] = pooledTheil;
      summary[summaryOffset + 1u] = laneFirst[0];
      summary[summaryOffset + 2u] = laneSecond[0];
    }
  }`;function te(e){return`let firstPosition = index * GINI_TILE_ROWS;
  let endPosition = min(firstPosition + GINI_TILE_ROWS, ROW_COUNT);
  var tileWeight = 0.0;
  var tileIncome = 0.0;
  for (var position = firstPosition; position < endPosition; position++) {
    if (sortedValueKeys[sortedValueKeysOffset + position] == 0xffffffffu) {
      break;
    }
    let row = sortedRows[sortedRowsOffset + position];
    let w = ${e};
    tileWeight = tileWeight + w;
    tileIncome = tileIncome + w * values[valuesOffset + row];
  }
  tileTotals[tileTotalsOffset + 2u * index] = tileWeight;
  tileTotals[tileTotalsOffset + 2u * index + 1u] = tileIncome;`}var ne=`
var<workgroup> scanWeight: array<f32, 256>;
var<workgroup> scanIncome: array<f32, 256>;`,re=`let lane = localInvocationIndex;
  const chunkSize: u32 = (GINI_TILE_COUNT + 255u) / 256u;
  let firstTile = min(lane * chunkSize, GINI_TILE_COUNT);
  let endTile = min(firstTile + chunkSize, GINI_TILE_COUNT);
  var chunkWeight = 0.0;
  var chunkIncome = 0.0;
  for (var tile = firstTile; tile < endTile; tile++) {
    chunkWeight = chunkWeight + tileTotals[tileTotalsOffset + 2u * tile];
    chunkIncome = chunkIncome + tileTotals[tileTotalsOffset + 2u * tile + 1u];
  }
  scanWeight[lane] = chunkWeight;
  scanIncome[lane] = chunkIncome;
  workgroupBarrier();
  for (var offset = 1u; offset < 256u; offset = offset << 1u) {
    var addWeight = 0.0;
    var addIncome = 0.0;
    if (lane >= offset) {
      addWeight = scanWeight[lane - offset];
      addIncome = scanIncome[lane - offset];
    }
    workgroupBarrier();
    scanWeight[lane] = scanWeight[lane] + addWeight;
    scanIncome[lane] = scanIncome[lane] + addIncome;
    workgroupBarrier();
  }
  // The scan is inclusive; this lane starts after the chunks before it.
  var cumulativeWeight = 0.0;
  var cumulativeIncome = 0.0;
  if (lane > 0u) {
    cumulativeWeight = scanWeight[lane - 1u];
    cumulativeIncome = scanIncome[lane - 1u];
  }
  for (var tile = firstTile; tile < endTile; tile++) {
    let base = tileTotalsOffset + 2u * tile;
    let tileWeight = tileTotals[base];
    let tileIncome = tileTotals[base + 1u];
    tileTotals[base] = cumulativeWeight;
    tileTotals[base + 1u] = cumulativeIncome;
    cumulativeWeight = cumulativeWeight + tileWeight;
    cumulativeIncome = cumulativeIncome + tileIncome;
  }
  if (lane == 255u) {
    pooledTotals[pooledTotalsOffset] = scanWeight[lane];
    pooledTotals[pooledTotalsOffset + 1u] = scanIncome[lane];
  }`;function z(e){return`let totalWeight = pooledTotals[pooledTotalsOffset];
  let totalIncome = pooledTotals[pooledTotalsOffset + 1u];
  var area = 0.0;
  if (totalIncome > 0.0) {
    let firstPosition = index * GINI_TILE_ROWS;
    let endPosition = min(firstPosition + GINI_TILE_ROWS, ROW_COUNT);
    var cumulativeWeight = tileTotals[tileTotalsOffset + 2u * index];
    var cumulativeIncome = tileTotals[tileTotalsOffset + 2u * index + 1u];
    var previousP = getScaledQuotient(cumulativeWeight, totalWeight, 1.0, 0);
    var previousL = getScaledQuotient(cumulativeIncome, totalIncome, 1.0, 0);
    for (var position = firstPosition; position < endPosition; position++) {
      if (sortedValueKeys[sortedValueKeysOffset + position] == 0xffffffffu) {
        break;
      }
      let row = sortedRows[sortedRowsOffset + position];
      let w = ${e};
      cumulativeWeight = cumulativeWeight + w;
      cumulativeIncome = cumulativeIncome + w * values[valuesOffset + row];
      let nextP = getScaledQuotient(cumulativeWeight, totalWeight, 1.0, 0);
      let nextL = getScaledQuotient(cumulativeIncome, totalIncome, 1.0, 0);
      area = area + (nextP - previousP) * (previousL + nextL);
      previousP = nextP;
      previousL = nextL;
    }
  }
  tileAreas[tileAreasOffset + index] = area;`}var B=`
var<workgroup> laneAreas: array<f32, 256>;`,V=`let lane = localInvocationIndex;
  var laneArea = 0.0;
  for (var tile = lane; tile < GINI_TILE_COUNT; tile += 256u) {
    laneArea = laneArea + tileAreas[tileAreasOffset + tile];
  }
  laneAreas[lane] = laneArea;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneAreas[lane] = laneAreas[lane] + laneAreas[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    if (!(pooledTotals[pooledTotalsOffset + 1u] > 0.0)) {
      summary[summaryOffset + 3u] = getNaN();
    } else {
      summary[summaryOffset + 3u] = 1.0 - laneAreas[0];
    }
  }`,H=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),U=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`});function W(t,n){let{id:r,sortedZones:i,sortedRows:a,values:o,weights:s,zoneCount:c}=n,l=a.length,u=[],d=(n,i)=>e(t,`${r}-${n}`,`float32`,i),f=(n,i)=>e(t,`${r}-${n}`,`uint32`,i),p=d(`weights`,l),m=d(`incomes`,l),h=Math.ceil(l/256),g={keys:i,length:l,weights:p,incomes:m,tailKeys:f(`tail-keys-0`,h),tailPairs:d(`tail-pairs-0`,h*2)},_=[g];u.push(S(t,{id:`${r}-rows`,operation:`GPUInequality`,variant:`lorenz-prefix-rows`,invocationCount:h,parameterValues:[l,c],bindings:[H(`zoneKeys`,i,`u32`),H(`sortedRows`,a,`u32`),H(`values`,o,`f32`),...s?[H(`weights`,s,`f32`)]:[],U(`prefixWeights`,p,`f32`),U(`prefixIncomes`,m,`f32`),U(`tailKeys`,g.tailKeys,`u32`),U(`tailPairs`,g.tailPairs,`f32`)],body:`let firstRow = index * 256u;
  let endRow = min(firstRow + 256u, getChunkParameter(36u));
  var previousZone = 0xffffffffu;
  var population = 0.0;
  var income = 0.0;
  for (var position = firstRow; position < endRow; position++) {
    let zone = zoneKeys[zoneKeysOffset + position];
    if (zone != previousZone) { population = 0.0; income = 0.0; }
    if (zone < getChunkParameter(37u)) {
      let row = sortedRows[sortedRowsOffset + position];
      let weight = ${s?`weights[weightsOffset + row]`:`1.0`};
      population += weight;
      income += weight * values[valuesOffset + row];
    }
    prefixWeights[prefixWeightsOffset + position] = population;
    prefixIncomes[prefixIncomesOffset + position] = income;
    previousZone = zone;
  }
  tailKeys[tailKeysOffset + index] = previousZone;
  tailPairs[tailPairsOffset + 2u * index] = population;
  tailPairs[tailPairsOffset + 2u * index + 1u] = income;`}));let v=g.tailKeys,y=g.tailPairs;do{let e=_.length,n=v.length,i=Math.ceil(n/256),a={keys:v,length:n,pairs:d(`prefix-pairs-${e}`,n*2),tailKeys:f(`tail-keys-${e}`,i),tailPairs:d(`tail-pairs-${e}`,i*2)};u.push(S(t,{id:`${r}-summaries-${e}`,operation:`GPUInequality`,variant:`lorenz-prefix-summaries`,invocationCount:i,parameterValues:[n],bindings:[H(`zoneKeys`,v,`u32`),H(`inputPairs`,y,`f32`),U(`prefixPairs`,a.pairs,`f32`),U(`tailKeys`,a.tailKeys,`u32`),U(`tailPairs`,a.tailPairs,`f32`)],body:`let firstRow = index * 256u;
  let endRow = min(firstRow + 256u, getChunkParameter(36u));
  var previousZone = 0xffffffffu;
  var population = 0.0;
  var income = 0.0;
  for (var position = firstRow; position < endRow; position++) {
    let zone = zoneKeys[zoneKeysOffset + position];
    if (zone != previousZone) { population = 0.0; income = 0.0; }
    population += inputPairs[inputPairsOffset + 2u * position];
    income += inputPairs[inputPairsOffset + 2u * position + 1u];
    prefixPairs[prefixPairsOffset + 2u * position] = population;
    prefixPairs[prefixPairsOffset + 2u * position + 1u] = income;
    previousZone = zone;
  }
  tailKeys[tailKeysOffset + index] = previousZone;
  tailPairs[tailPairsOffset + 2u * index] = population;
  tailPairs[tailPairsOffset + 2u * index + 1u] = income;`})),_.push(a),v=a.tailKeys,y=a.tailPairs}while(v.length>1);for(let e=_.length-1;e>=0;e--){let n=_[e],i=_[e+1],a=i?.keys??n.tailKeys,o=i?.pairs??n.tailPairs;u.push(S(t,{id:`${r}-carry-${e}`,operation:`GPUInequality`,variant:n.pairs?`lorenz-summary-carry`:`lorenz-row-carry`,invocationCount:n.length,bindings:[H(`zoneKeys`,n.keys,`u32`),H(`parentKeys`,a,`u32`),H(`parentPairs`,o,`f32`),...n.pairs?[U(`prefixPairs`,n.pairs,`f32`)]:[U(`prefixWeights`,n.weights,`f32`),U(`prefixIncomes`,n.incomes,`f32`)]],body:`let block = index / 256u;
  if (block > 0u && zoneKeys[zoneKeysOffset + index] == parentKeys[parentKeysOffset + block - 1u]) {
    let population = parentPairs[parentPairsOffset + 2u * (block - 1u)];
    let income = parentPairs[parentPairsOffset + 2u * (block - 1u) + 1u];
    ${n.pairs?`prefixPairs[prefixPairsOffset + 2u * index] += population;
    prefixPairs[prefixPairsOffset + 2u * index + 1u] += income;`:`prefixWeights[prefixWeightsOffset + index] += population;
    prefixIncomes[prefixIncomesOffset + index] += income;`}
  }`}))}return{weights:p,incomes:m,nodes:u}}var G=6,K=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),q=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`});function J(t,n){let{id:r,sortedZones:i,sortedRows:a,values:o,weights:s,zoneCount:c,parameters:l,stats:u,counts:d,lorenz:f,knotCount:p}=n,m=W(t,{id:`${r}-lorenz-prefix`,sortedZones:i,sortedRows:a,values:o,weights:s,zoneCount:c}),h=e(t,`${r}-zone-offsets`,`uint32`,c+1),g=e(t,`${r}-row-terms`,`float32`,o.length),_=e(t,`${r}-term-totals`,`float32`,c),v=`const STATS_STRIDE: u32 = 10u;\n${P}`,y=[...m.nodes,...w(t,{id:r,operation:`GPUInequality`,groupCount:c,sortedGroupKeys:i,segmentOffsets:h})];y.push(S(t,{id:`${r}-zone-stats`,operation:`GPUInequality`,variant:`zone-totals`,invocationCount:c,bindings:[K(`zoneOffsets`,h,`u32`),K(`sortedRows`,a,`u32`),K(`values`,o,`f32`),K(`prefixWeights`,m.weights,`f32`),K(`prefixIncomes`,m.incomes,`f32`),q(`stats`,u,`f32`),q(`counts`,d,`u32`)],declarations:v,body:`let begin = zoneOffsets[zoneOffsetsOffset + index];
  let end = zoneOffsets[zoneOffsetsOffset + index + 1u];
  let base = statsOffset + index * STATS_STRIDE;
  let nan = getNaN();
  for (var slot = 0u; slot < STATS_STRIDE; slot++) { stats[base + slot] = nan; }
  counts[countsOffset + index] = end - begin;
  stats[base + 7u] = 0.0; stats[base + 8u] = 0.0; stats[base + 9u] = 0.0;
  if (begin == end) { return; }
  let population = prefixWeights[prefixWeightsOffset + end - 1u];
  let income = prefixIncomes[prefixIncomesOffset + end - 1u];
  stats[base] = getScaledQuotient(income, population, 1.0, 0);
  stats[base + 7u] = population; stats[base + 8u] = income;
  let hasZero = values[valuesOffset + sortedRows[sortedRowsOffset + begin]] == 0.0;
  stats[base + 3u] = select(0.0, nan, hasZero);`}));for(let e=0;e<G;e++)y.push(S(t,{id:`${r}-terms-${e}`,operation:`GPUInequality`,variant:`inequality-row-term`,invocationCount:o.length,parameterValues:[e,c],bindings:[K(`sortedRows`,a,`u32`),K(`zoneKeys`,i,`u32`),K(`values`,o,`f32`),...s?[K(`weights`,s,`f32`)]:[],K(`stats`,u,`f32`),K(`prefixIncomes`,m.incomes,`f32`),K(`params`,l,`f32`),q(`terms`,g,`f32`)],declarations:v,body:`let zone = zoneKeys[zoneKeysOffset + index];
  terms[termsOffset + index] = 0.0;
  if (zone >= getChunkParameter(37u)) { return; }
  let row = sortedRows[sortedRowsOffset + index];
  let value = values[valuesOffset + row];
  let weight = ${s?`weights[weightsOffset + row]`:`1.0`};
  let base = statsOffset + zone * STATS_STRIDE;
  let population = stats[base + 7u]; let income = stats[base + 8u];
  let term = getChunkParameter(36u);
  var contribution = 0.0;
  if (term == 0u) {
    if (value > 0.0) { contribution = weight * value * log(value); }
  } else if (income > 0.0) {
    let ratio = getScaledQuotient(value, stats[base], 1.0, 0);
    if (term == 1u) {
      contribution = (getScaledQuotient(weight, population, 1.0, 0)) * (2.0 * (getScaledQuotient(prefixIncomes[prefixIncomesOffset + index], income, 1.0, 0)) - weight * getScaledQuotient(value, income, 1.0, 0));
    } else if (term == 5u) { contribution = weight * abs(value - stats[base]);
    } else if (value > 0.0) {
      let logRatio = log(ratio);
      if (term == 2u) { contribution = weight * ratio * logRatio; }
      if (term == 3u) { contribution = weight * logRatio; }
      if (term == 4u) { contribution = weight * exp((1.0 - params[paramsOffset]) * logRatio); }
    }
  }
  terms[termsOffset + index] = contribution;`})),y.push(C(t,{id:`${r}-term-sum-${e}`,operation:`GPUInequality`,input:g,segmentOffsets:h,segmentCount:c,output:_})),y.push(S(t,{id:`${r}-term-finish-${e}`,operation:`GPUInequality`,variant:`inequality-term-finish`,invocationCount:c,parameterValues:[e],bindings:[K(`totals`,_,`f32`),K(`counts`,d,`u32`),K(`params`,l,`f32`),q(`stats`,u,`f32`)],declarations:v,body:`if (counts[countsOffset + index] == 0u) { return; }
  let base = statsOffset + index * STATS_STRIDE;
  let term = getChunkParameter(36u);
  let total = totals[totalsOffset + index];
  if (term == 0u) { stats[base + 9u] = total; return; }
  let population = stats[base + 7u]; let income = stats[base + 8u];
  if (!(income > 0.0)) { return; }
  if (term == 1u) { stats[base + 1u] = 1.0 - total; }
  if (term == 2u) { stats[base + 2u] = getScaledQuotient(total, population, 1.0, 0); }
  if (term == 3u) { if (isFiniteBits(stats[base + 3u])) { stats[base + 3u] = -getScaledQuotient(total, population, 1.0, 0); } }
  if (term == 4u) {
    let power = 1.0 - params[paramsOffset]; var atkinson = getNaN();
    if (power == 0.0) { atkinson = 1.0 - exp(-stats[base + 3u]); }
    else if (power > 0.0 || isFiniteBits(stats[base + 3u])) { atkinson = 1.0 - pow(getScaledQuotient(total, population, 1.0, 0), 1.0 / power); }
    stats[base + 4u] = select(getNaN(), atkinson, isFiniteBits(atkinson));
  }
  if (term == 5u) { stats[base + 5u] = getScaledQuotient(total, income, 2.0, 0); }`}));return y.push(S(t,{id:`${r}-lorenz-cuts`,operation:`GPUInequality`,variant:`lorenz-cuts`,invocationCount:c*p,parameterValues:[p],bindings:[K(`zoneOffsets`,h,`u32`),K(`prefixWeights`,m.weights,`f32`),K(`prefixIncomes`,m.incomes,`f32`),K(`params`,l,`f32`),q(`stats`,u,`f32`),q(`lorenz`,f,`f32`)],declarations:`${v}
fn readLorenzCut(zone: u32, cut: f32) -> f32 {
  let begin = zoneOffsets[zoneOffsetsOffset + zone]; let end = zoneOffsets[zoneOffsetsOffset + zone + 1u];
  let population = stats[statsOffset + zone * STATS_STRIDE + 7u]; let income = stats[statsOffset + zone * STATS_STRIDE + 8u];
  if (begin == end || !(income > 0.0)) { return getNaN(); }
  if (cut <= 0.0) { return 0.0; } if (cut >= 1.0) { return 1.0; }
  let threshold = cut * population; var low = begin; var high = end;
  while (low < high) { let middle = low + (high - low) / 2u;
    if (prefixWeights[prefixWeightsOffset + middle] < threshold) { low = middle + 1u; } else { high = middle; }
  }
  let position = min(low, end - 1u); var previousPopulation = 0.0; var previousIncome = 0.0;
  if (position > begin) { previousPopulation = prefixWeights[prefixWeightsOffset + position - 1u]; previousIncome = prefixIncomes[prefixIncomesOffset + position - 1u]; }
  let nextPopulation = prefixWeights[prefixWeightsOffset + position]; let nextIncome = prefixIncomes[prefixIncomesOffset + position];
  let fraction = select(1.0, clamp(getScaledQuotient(threshold - previousPopulation, nextPopulation - previousPopulation, 1.0, 0), 0.0, 1.0), nextPopulation > previousPopulation);
  return getScaledQuotient(previousIncome + fraction * (nextIncome - previousIncome), income, 1.0, 0);
}`,body:`let knotCount = getChunkParameter(36u);
  let zone = index / knotCount; let knot = index % knotCount;
  lorenz[lorenzOffset + index] = readLorenzCut(zone, f32(knot) / f32(knotCount - 1u));
  if (knot == 0u) {
    let bottom = readLorenzCut(zone, params[paramsOffset + 2u]);
    let top = readLorenzCut(zone, 1.0 - params[paramsOffset + 1u]);
    stats[statsOffset + zone * STATS_STRIDE + 6u] = select(getNaN(), getScaledQuotient(1.0 - top, bottom, 1.0, 0), bottom > 0.0);
  }`})),y}var Y=`GPUInequality`,X=2147483648,Z=2,Q=1025,ie=class{id;props;rowCount;lorenzKnotCount;mode;estimatedCommandNodeCount;constructor(e){this.id=e.id??`inequality`,this.props=e;let t=this.id;this.mode=e.mode??`parallel`,c(t,this.mode===`parallel`||this.mode===`reference`,`mode must be parallel or reference`),this.estimatedCommandNodeCount=2112,d(t,this.estimatedCommandNodeCount);let{output:i,zoneCount:a}=e;if(u(t,`zoneCount`,a,1,4294967294),this.lorenzKnotCount=e.lorenzKnotCount??11,u(t,`lorenzKnotCount`,this.lorenzKnotCount,Z,Q),o(e.values,[`float32`],`${t} values`),o(e.zoneIds,[`uint32`],`${t} zoneIds`),this.rowCount=e.values.length,u(t,`values length`,this.rowCount,1,X),e.zoneIds.length!==this.rowCount)throw Error(`${t} zoneIds length must equal the row count`);if(e.weights&&(o(e.weights,[`float32`],`${t} weights`),e.weights.length!==this.rowCount))throw Error(`${t} weights length must equal the row count`);if(e.mask&&(o(e.mask,[`uint32`],`${t} mask`),e.mask.length!==this.rowCount))throw Error(`${t} mask length must equal the row count`);if(o(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 float32 values`);let s=[[`gini`,a,`float32`],[`theilT`,a,`float32`],[`theilL`,a,`float32`],[`atkinson`,a,`float32`],[`hoover`,a,`float32`],[`palma`,a,`float32`],[`mean`,a,`float32`],[`counts`,a,`uint32`],[`lorenzKnots`,a*this.lorenzKnotCount,`float32`],[`globalSummary`,8,`float32`]],f=0;for(let[e,n,r]of s){let a=i[e];if(a&&(f++,o(a,[r],`${t} output.${e}`),a.length<n))throw Error(`${t} output.${e} must hold at least ${n} values`)}if(f===0)throw Error(`${t} needs at least one output`);{let e=i,a=[[i.gini,i.theilT,i.theilL,i.atkinson],[i.hoover,i.palma,i.mean],...this.mode===`reference`?[[i.counts,i.lorenzKnots]]:[],...i.globalSummary?[[i.counts,i.globalSummary]]:[]],o=Object.values(e).flatMap(e=>e?l(e):[]).map(e=>new r(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset,byteStride:4,rowByteLength:4})),s=o.reduce((e,t)=>e+t.length,0);g(t,{outputs:new n({id:`${t}-writable-ranges`,name:`outputs`,format:`uint32`,length:s,valueLength:s,stride:1,byteStride:4,rowByteLength:4,data:o})});for(let e of a)g(t,Object.fromEntries(e.map((e,t)=>[String(t),e])))}m(t,s.map(([e])=>i[e]),[e.values,e.zoneIds,e.weights,e.mask,e.parameters])}getCommandNodes(n){let{props:r,id:o,rowCount:c,lorenzKnotCount:l}=this,{output:d,zoneCount:f,values:m,zoneIds:g,weights:_,mask:v,parameters:y}=r;u(o,`zoneCount`,f,1,A(n.device)),s(o,`zoneCount x lorenzKnotCount`,f*l,j(n.device)),h(n,o,{values:m,zoneIds:g,weights:_,mask:v,parameters:y,...d});let b=Math.ceil(c/256);for(let[e,t]of[[`row scratch`,c],[`stats scratch`,f*10],[`counts scratch`,f],[`lorenz scratch`,f*l],...d.globalSummary?[[`Gini tile scratch`,Math.max(2,2*b)]]:[]])s(o,`${e} byteLength`,t*4,n.device.limits.maxBufferSize),p(n.device,o,e,t*4,`reduce values length, zoneCount or lorenzKnotCount`);if(this.mode===`parallel`){x(n,o,8);for(let e of[c,f,f*l])i(o,e,256,n.device.limits.maxComputeWorkgroupsPerDimension);for(let[e,t]of[[`zone offset scratch`,f+1],[`Lorenz summary scratch`,2*Math.ceil(c/256)]])p(n.device,o,e,t*4,`reduce row count or zoneCount`),s(o,`${e} byteLength`,t*4,n.device.limits.maxBufferSize)}let S=(t,r,i)=>e(n,`${o}-${t}`,r,Math.max(i,1)),C=S(`value-keys`,`uint32`,c),w=S(`row-indices`,`uint32`,c),T=S(`sorted-value-keys`,`uint32`,c),E=S(`sorted-rows`,`uint32`,c),D=S(`zone-keys`,`uint32`,c),O=S(`sorted-zone-keys`,`uint32`,c),k=S(`zone-sorted-rows`,`uint32`,c),M=S(`stats`,`float32`,f*10),H=d.counts??S(`counts`,`uint32`,f),U=d.lorenzKnots??S(`lorenz`,`float32`,f*l),W=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),G=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`}),K=`const ROW_COUNT: u32 = ${c}u;
const ZONE_COUNT: u32 = ${f}u;
const KNOT_COUNT: u32 = ${l}u;
const STATS_STRIDE: u32 = 10u;
${P}`,q=_?`weights[weightsOffset + row]`:`1.0`,X=[];X.push(a(n,{id:`${o}-keys`,operation:Y,variant:`keys`,bindings:[W(`values`,m,`f32`),W(`zoneIds`,g,`u32`),...v?[W(`rowMask`,v,`u32`)]:[],..._?[W(`weights`,_,`f32`)]:[],G(`valueKeys`,C,`u32`),G(`rowIndices`,w,`u32`)],invocationCount:c,declarations:K,body:F({hasMask:!!v,hasWeights:!!_})})),X.push(...new t({id:`${o}-value-sort`,keys:C,values:w,outputKeys:T,outputValues:E,keyBits:32}).getCommandNodes(n)),X.push(a(n,{id:`${o}-zone-keys`,operation:Y,variant:`zone-keys`,bindings:[W(`sortedValueKeys`,T,`u32`),W(`sortedRows`,E,`u32`),W(`zoneIds`,g,`u32`),G(`zoneKeys`,D,`u32`)],invocationCount:c,declarations:K,body:I}));let Z=Math.max(1,Math.ceil(Math.log2(f+1)));X.push(...new t({id:`${o}-zone-sort`,keys:D,values:E,outputKeys:O,outputValues:k,algorithm:`radix`,keyBits:Math.min(32,Z)}).getCommandNodes(n)),this.mode===`parallel`?X.push(...J(n,{id:o,sortedZones:O,sortedRows:k,values:m,weights:_,zoneCount:f,parameters:y,stats:M,counts:H,lorenz:U,knotCount:l})):X.push(a(n,{id:`${o}-zone-stats`,operation:Y,variant:`zone-stats`,bindings:[W(`sortedZones`,O,`u32`),W(`sortedRows`,k,`u32`),W(`values`,m,`f32`),..._?[W(`weights`,_,`f32`)]:[],W(`params`,y,`f32`),G(`stats`,M,`f32`),G(`lorenz`,U,`f32`),G(`counts`,H,`u32`)],invocationCount:f,declarations:K,body:L(q)}));let Q=(e,t)=>{let r=t.filter(([,,e])=>e);r.length!==0&&X.push(a(n,{id:`${o}-${e}`,operation:Y,variant:e,bindings:[W(`stats`,M,`f32`),...r.map(([e,,t])=>G(e,t,`f32`))],invocationCount:f,declarations:K,body:r.map(([e,t])=>`${e}[${e}Offset + index] = stats[statsOffset + index * STATS_STRIDE + ${t}u];`).join(`
  `)}))};if(Q(`publish-indices`,[[`gini`,N.GINI,d.gini],[`theilT`,N.THEIL_T,d.theilT],[`theilL`,N.THEIL_L,d.theilL],[`atkinson`,N.ATKINSON,d.atkinson]]),Q(`publish-moments`,[[`hoover`,N.HOOVER,d.hoover],[`palma`,N.PALMA,d.palma],[`mean`,N.MEAN,d.mean]]),d.globalSummary){X.push(a(n,{id:`${o}-global-sums`,operation:Y,variant:`global-sums`,bindings:[W(`stats`,M,`f32`),G(`counts`,H,`u32`),G(`summary`,d.globalSummary,`f32`)],invocationCount:256,workgroupSize:256,guardIndex:!1,declarations:`${K}${R}`,body:ee}));let e=Math.ceil(c/256),t=S(`gini-tile-totals`,`float32`,2*e),r=S(`gini-tile-areas`,`float32`,e),i=S(`gini-pooled-totals`,`float32`,2),s=`${K}
const GINI_TILE_ROWS: u32 = 256u;
const GINI_TILE_COUNT: u32 = ${e}u;`,l=[W(`sortedValueKeys`,T,`u32`),W(`sortedRows`,E,`u32`),W(`values`,m,`f32`),..._?[W(`weights`,_,`f32`)]:[]];X.push(a(n,{id:`${o}-global-gini-tile-totals`,operation:Y,variant:`global-gini-tile-totals`,bindings:[...l,G(`tileTotals`,t,`f32`)],invocationCount:e,declarations:s,body:te(q)})),X.push(a(n,{id:`${o}-global-gini-tile-prefix`,operation:Y,variant:`global-gini-tile-prefix`,bindings:[G(`tileTotals`,t,`f32`),G(`pooledTotals`,i,`f32`)],invocationCount:256,workgroupSize:256,guardIndex:!1,declarations:`${s}${ne}`,body:re})),X.push(a(n,{id:`${o}-global-gini-tile-area`,operation:Y,variant:`global-gini-tile-area`,bindings:[...l,W(`tileTotals`,t,`f32`),W(`pooledTotals`,i,`f32`),G(`tileAreas`,r,`f32`)],invocationCount:e,declarations:s,body:z(q)})),X.push(a(n,{id:`${o}-global-gini`,operation:Y,variant:`global-gini`,bindings:[W(`tileAreas`,r,`f32`),W(`pooledTotals`,i,`f32`),G(`summary`,d.globalSummary,`f32`)],invocationCount:256,workgroupSize:256,guardIndex:!1,declarations:`${s}${B}`,body:V}))}return X}},ae=[[0,1],[0,2],[0,1],[0,1]],oe=[`Gini`,`Theil T`,`Atkinson`,`Hoover`],se=40;function ce({world:e,random:t}){let n=E(t,e);return{...n,world:e,probe:O(n.tracts,n.keys),focus:{tract:-1}}}function le(e,t,n){let r=t.map(t=>e[t]).sort((e,t)=>e-t),i=r.reduce((e,t)=>e+t,0)||1,a=Math.max(1,Math.ceil(r.length/se)),o=[n[0],n[1]],s=0;return r.forEach((e,t)=>{s+=e,((t+1)%a===0||t===r.length-1)&&o.push(n[0]+(t+1)/r.length*(n[2]-n[0]),n[1]+s/i*(n[3]-n[1]))}),o}var $=T({parameters:k,caption:`Concentration of jobs across observed workplace blocks in each Chicago tract.`,createInputs:ce,build(e,t){let{tracts:n}=t,r=e.upload(`jobs`,`float32`,t.jobs),i=e.allocate(`parameters`,`float32`,4),a=e.allocate(`gini`,`float32`,n.count),o=e.allocate(`theil-t`,`float32`,n.count),s=e.allocate(`atkinson`,`float32`,n.count),c=e.allocate(`hoover`,`float32`,n.count);return e.graph.add(new ie({values:r.view,zoneIds:e.upload(`zone-ids`,`uint32`,t.keys).view,zoneCount:n.count,parameters:i.view,output:{gini:a.view,theilT:o.view,atkinson:s.view,hoover:c.view}})),{outputs:{...D(e,n),gini:a,theilT:o,atkinson:s,hoover:c},update:e=>{i.write(M({epsilon:e.epsilon}))}}},rebuildOn:[],interaction:{kind:`probe`,hint:`hover a tract`,rest:({inputs:e})=>{let t=e.tracts.centers.subarray(e.probe.richest*2,e.probe.richest*2+2);return[t[0]/e.world.width,1-t[1]/e.world.height]},resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=t.probe.findTract(e.x,e.y);a>=0&&(t.focus.tract=a);let o=t.focus.tract;if(o<0)return{readout:`hover a tract`};let s=[.03*t.world.width,.04,.03*t.world.width+.24,.28],c=[{shape:`polyline`,points:t.probe.getRing(o),closed:!0},{shape:`box`,minimum:[s[0],s[1]],maximum:[s[2],s[3]]},{shape:`line`,from:[s[0],s[1]],to:[s[2],s[3]],dashed:!0},{shape:`polyline`,points:le(t.jobs,t.probe.members[o],s)}],l=e=>{let t=i(e)?.[o];return t===void 0?`…`:t.toFixed(2)},u=[n.gini,n.theilT,n.atkinson,n.hoover];return{glyphs:c,readout:`tract ${t.tracts.geoids[o].slice(-6)} · ${oe[r.output]} ${l(u[r.output])} · ${t.probe.members[o].length} blocks`}}},draw:({inputs:e,outputs:t,values:n})=>[{layer:`polygons`,positions:t.vertices,indices:t.indices,owners:t.owners,values:[t.gini,t.theilT,t.atkinson,t.hoover][n.output],triangleCount:e.tracts.triangleCount,range:ae[n.output]},{layer:`lines`,context:!0,segments:t.outline,count:e.tracts.outlineCount}]});export{$ as default};
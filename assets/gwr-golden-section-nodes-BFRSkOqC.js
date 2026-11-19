import{r as e}from"./kernel-C3tfQOn4.js";import{a as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{f as r,l as i,m as a,p as o,x as s}from"./local-weighted-least-squares-wgsl--tcHa6bd.js";import{c,r as l}from"./wgsl-common-DfyXiYB0.js";function u(e,t){return 1+(2*e+2)*(t+2)+e}function d(t,i){let{id:a,maximumIterations:o}=i,s=e(t,`${a}-interval`,`float32`,8),u=e(t,`${a}-result`,`float32`,7),d=e(t,`${a}-value`,`float32`,1),f=e(t,`${a}-score`,`float32`,4),p=(e,t)=>({name:e,view:t,type:`f32`,access:`read`}),m=(e,t)=>({name:e,view:t,type:`f32`,access:`read_write`}),h=(e,r,i)=>n(t,{id:`${a}-${e}`,operation:`GTWRBandwidthSearch`,variant:e.replace(/\d+/g,``),invocationCount:1,bindings:r,declarations:`${l}\n${c}`,body:i}),g=[h(`initialize`,[p(`bounds`,i.bounds),m(`state`,s),m(`result`,u),...i.parent?[p(`parent`,i.parent)]:[]],`
  let lower = bounds[boundsOffset + ${i.axis*2}u];
  let upper = bounds[boundsOffset + ${i.axis*2+1}u];
  let absoluteTolerance = bounds[boundsOffset + 4u];
  let relativeTolerance = bounds[boundsOffset + 5u];
  var valid = isFiniteBits(lower) && isFiniteBits(upper) && lower <= upper
    && lower ${i.axis===0?`> 0.0`:`>= 0.0`}
    && isFiniteBits(absoluteTolerance) && absoluteTolerance > 0.0
    && isFiniteBits(relativeTolerance) && relativeTolerance >= 0.0;
  ${i.integer?`valid = valid && lower == round(lower) && upper == round(upper);`:``}
  ${i.maximumValue===void 0?``:`valid = valid && upper <= ${i.maximumValue}.0;`}
  ${i.parent?`valid = valid && parent[parentOffset + 5u] > 0.5;`:``}
  state[stateOffset] = lower;
  state[stateOffset + 1u] = upper;
  state[stateOffset + 2u] = 0.0;
  state[stateOffset + 3u] = 0.0;
  state[stateOffset + 4u] = 0.0;
  state[stateOffset + 5u] = 0.0;
  state[stateOffset + 6u] = 0.0;
  state[stateOffset + 7u] = 0.0;
  for (var field = 0u; field < 7u; field++) { result[resultOffset + field] = 0.0; }
  result[resultOffset] = getNaN();
  result[resultOffset + 1u] = getNaN();
  result[resultOffset + 2u] = getNaN();
  result[resultOffset + 5u] = select(0.0, 1.0, valid);`)];for(let e=0;e<2+2*o;e++){let t=e<2,n=e%2;g.push(h(`prepare-${e}`,[m(`state`,s),p(`result`,u),m(`value`,d),m(`score`,f)],`
  if (result[resultOffset + 5u] < 0.5) { return; }
  let lower = state[stateOffset];
  let upper = state[stateOffset + 1u];
  var candidate = ${t?n===0?`lower`:`upper`:n===0?`lower + ${r} * (upper - lower)`:`upper - ${r} * (upper - lower)`};
  ${i.integer?`candidate = round(candidate);`:``}
  ${i.integer&&!t&&n===1?`// Rounded interior points must compare different counts while the bracket has room.
  // In a three-count bracket this visits the middle and upper counts; the lower was tested
  // when it became a bracket endpoint. The iteration can then certify all three counts.
  candidate = min(upper, max(candidate, state[stateOffset + 2u] + 1.0));`:``}
  value[valueOffset] = candidate;
  state[stateOffset + ${2+n}u] = candidate;
  for (var field = 0u; field < 4u; field++) { score[scoreOffset + field] = getNaN(); }`)),g.push(...i.evaluate(d,f,u,`${a}-evaluate-${e}`)),g.push(h(`collect-${e}`,[p(`value`,d),p(`score`,f),m(`state`,s),m(`result`,u)],`
  if (result[resultOffset + 5u] < 0.5) { return; }
  let objective = score[scoreOffset];
  let candidate = value[valueOffset];
  state[stateOffset + ${4+n}u] = objective;
  let best = result[resultOffset + 1u];
  if (isFiniteBits(objective) && (!isFiniteBits(best) || objective < best
      || (objective == best && candidate < result[resultOffset]))) {
    result[resultOffset] = candidate;
    result[resultOffset + 1u] = objective;
    result[resultOffset + 2u] = score[scoreOffset + 1u];
    state[stateOffset + 6u] = score[scoreOffset + 2u];
    state[stateOffset + 7u] = score[scoreOffset + 3u];
  }`)),!t&&n===1&&g.push(h(`iterate-${e}`,[p(`bounds`,i.bounds),m(`state`,s),m(`result`,u)],`
  if (result[resultOffset + 5u] < 0.5) { return; }
  let firstScore = state[stateOffset + 4u];
  let secondScore = state[stateOffset + 5u];
  let firstWins = isFiniteBits(firstScore) && (!isFiniteBits(secondScore) || firstScore <= secondScore);
  ${i.integer?`let resolvedIntegerInterval = state[stateOffset + 1u] - state[stateOffset] <= 2.0;`:``}
  if (firstWins) { state[stateOffset + 1u] = state[stateOffset + 3u]; }
  else { state[stateOffset] = state[stateOffset + 2u]; }
  let lower = state[stateOffset];
  let upper = state[stateOffset + 1u];
  let tolerance = max(bounds[boundsOffset + 4u], bounds[boundsOffset + 5u] * max(abs(lower), abs(upper)));
  let finished = ${i.integer?`resolvedIntegerInterval || `:``}upper - lower <= ${i.integer?`max(1.0, tolerance)`:`tolerance`};
  result[resultOffset + 4u] += 1.0;
  if (finished || result[resultOffset + 4u] >= ${o}.0) {
    result[resultOffset + 5u] = 0.0;
    result[resultOffset + 3u] = select(0.0, state[stateOffset + 6u], finished && isFiniteBits(result[resultOffset + 1u]));
    result[resultOffset + 6u] = max(state[stateOffset + 7u], select(0.0, 1.0, !finished));
  }`))}return{nodes:g,result:u}}var f=s;function p(e,s){let{id:c,operation:l,rowCount:u,tileCount:d,candidateWidth:p}=s,m=t(e,{id:`${c}-search-dispatch`,operation:l,shapes:s.dispatchShapes??[[u],[d*p],[p]]}),h=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),g=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`}),_=Object.entries(f).map(([e,t])=>`const STATE_${e}: u32 = ${t}u;`).join(`
`),v=`${s.common}
const MAXIMUM_ITERATIONS: u32 = ${s.maximumIterations}u;
const GOLDEN_DELTA: f32 = ${r};
const TOLERANCE_ABSOLUTE: f32 = ${a};
const TOLERANCE_RELATIVE: f32 = ${o};
${_}`,y=`${s.focalRowWGSL}
  let slot = candidateScratchOffset + focal * 2u;
  candidateScratch[slot] = SENTINEL;
  candidateScratch[slot + 1u] = 0.0;
  if (getIsAdaptive() || rowValid[rowValidOffset + focal] == 0u) {
    return;
  }
  let origin = getPosition(focal);
  var nearest = SENTINEL;
  var farthest = 0.0;
  ${s.hasGrid?`// Nearest strictly positive distance, as for the scan below: rows coincident with the focal
  // row (repeated measures per site) are skipped, not counted as the nearest neighbour.
  var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
  selectPositiveNeighborDistances(focal, 1u, &neighbors);
  if (neighbors[0] < SENTINEL) {
    nearest = sqrt(neighbors[0]);
  }
  // The farthest row is no farther than the farthest corner of the grid extent. This upper bound
  // is looser than the scan's exact farthest row, so the grid path brackets the search with a
  // wider upper bound than the scan path; the bracket is documented, not exact.
  let boundsBase = sortedIdsOffset + ROW_COUNT;
  let minimum = vec2f(bitcast<f32>(sortedIds[boundsBase]), bitcast<f32>(sortedIds[boundsBase + 1u]));
  let maximum = vec2f(bitcast<f32>(sortedIds[boundsBase + 2u]), bitcast<f32>(sortedIds[boundsBase + 3u]));
  farthest = max(
    max(getDistance(origin, minimum), getDistance(origin, maximum)),
    max(getDistance(origin, vec2f(minimum.x, maximum.y)), getDistance(origin, vec2f(maximum.x, minimum.y)))
  );`:`for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let distance = getDistance(origin, getPosition(row));
    if (distance > 0.0) {
      nearest = min(nearest, distance);
    }
    farthest = max(farthest, distance);
  }`}
  candidateScratch[slot] = nearest;
  candidateScratch[slot + 1u] = farthest;`,b=[n(e,{id:`${c}-search-bounds-rows`,operation:l,variant:`search-bounds-rows`,bindings:[...s.candidateBindings,g(`candidateScratch`,s.candidateScratch,`f32`)],invocationCount:u,declarations:s.hasGrid?`${s.candidateDeclarations}\n${i(`selectPositiveNeighborDistances`,!0)}`:s.candidateDeclarations,body:y}),n(e,{id:`${c}-search-bounds-tiles`,operation:l,variant:`search-bounds-tiles`,bindings:[h(`candidateScratch`,s.candidateScratch,`f32`),h(`rowValid`,s.rowValid,`u32`),g(`tileCandidates`,s.tileCandidates,`f32`)],invocationCount:d,declarations:v,body:`let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var nearest = SENTINEL;
  var farthest = 0.0;
  var included = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    included = included + 1.0;
    nearest = min(nearest, candidateScratch[candidateScratchOffset + row * 2u]);
    farthest = max(farthest, candidateScratch[candidateScratchOffset + row * 2u + 1u]);
  }
  let base = tileCandidatesOffset + index * 3u;
  tileCandidates[base] = nearest;
  tileCandidates[base + 1u] = farthest;
  tileCandidates[base + 2u] = included;`}),n(e,{id:`${c}-search-init`,operation:l,variant:`search-init`,bindings:[h(`params`,s.parameters,`f32`),g(`ladders`,s.ladders,`f32`),h(`tileCandidates`,s.tileCandidates,`f32`),g(`table`,s.candidateTable,`f32`),g(`state`,s.searchState,`f32`),m.binding,...s.runningState?[h(`runningState`,s.runningState,`u32`)]:[]],invocationCount:1,declarations:v,body:`let nan = getNaN();
  for (var row = 0u; row < TABLE_ROWS; row++) {
    for (var field = 0u; field < TABLE_STRIDE; field++) {
      table[tableOffset + row * TABLE_STRIDE + field] = nan;
    }
  }
  let isAdaptive = params[paramsOffset + 1u] > 0.5;
  let callerCount = u32(max(params[paramsOffset + 2u], 0.0));
  var nearest = SENTINEL;
  var farthest = 0.0;
  var included = 0.0;
  for (var tile = 0u; tile < ${s.callerBoundsOnly?`0u`:`TILE_COUNT`}; tile++) {
    let base = tileCandidatesOffset + tile * 3u;
    nearest = min(nearest, tileCandidates[base]);
    farthest = max(farthest, tileCandidates[base + 1u]);
    included = included + tileCandidates[base + 2u];
  }
  // mgwr _init_section: adaptive [40 + 2 P, n] (n capped at the neighbour list), fixed
  // [min_dist / 2, 2 max_dist]; a positive caller bound replaces its default.
  var lower = 0.0;
  var upper = 0.0;
  if (isAdaptive) {
    lower = f32(40u + 2u * P);
    upper = min(included, f32(MAXIMUM_NEIGHBORS));
  } else if (included > 0.0 && nearest < SENTINEL) {
    lower = 0.5 * nearest;
    upper = 2.0 * farthest;
  }
  let isSearch = callerCount >= 2u${s.runningState?` && runningState[runningStateOffset] == 3u`:``};
  if (isSearch) {
    let givenLower = params[paramsOffset + HEADER];
    let givenUpper = params[paramsOffset + HEADER + 1u];
    if (givenLower > 0.0 && isFiniteBits(givenLower)) {
      lower = givenLower;
    }
    if (givenUpper > 0.0 && isFiniteBits(givenUpper)) {
      upper = givenUpper;
    }
  }
  let low = min(lower, upper);
  let high = max(lower, upper);
  var first = low + GOLDEN_DELTA * (high - low);
  var second = high - GOLDEN_DELTA * (high - low);
  if (isAdaptive) {
    // numpy round: half to even, as WGSL round.
    first = round(first);
    second = round(second);
  }
  if (isSearch) {
    // Block 0 holds the caller header; it now carries the two interior points of iteration 1.
    ladders[laddersOffset + 3u] = ${s.probeUpperBound?`3.0`:`2.0`};
    ladders[laddersOffset + 1u + HEADER] = first;
    ladders[laddersOffset + 2u + HEADER] = second;
    ${s.probeUpperBound?`ladders[laddersOffset + 3u + HEADER] = high;`:``}
  }
  state[stateOffset + STATE_LOWER] = low;
  state[stateOffset + STATE_LOWER_INTERIOR] = first;
  state[stateOffset + STATE_UPPER] = high;
  state[stateOffset + STATE_UPPER_INTERIOR] = second;
  state[stateOffset + STATE_LOWER_INTERIOR_ROW] = 0.0;
  state[stateOffset + STATE_UPPER_INTERIOR_ROW] = 1.0;
  state[stateOffset + STATE_ITERATION_COUNT] = 0.0;
  state[stateOffset + STATE_ACTIVE] = select(0.0, 1.0, isSearch);
  // A single caller value is a fixed bandwidth: its evaluation is table row 0.
  state[stateOffset + STATE_OPTIMUM_ROW] = select(0.0, -1.0, isSearch);
  state[stateOffset + STATE_EVALUATION_COUNT] = select(1.0, ${s.probeUpperBound?`3.0`:`2.0`}, isSearch);
  state[stateOffset + STATE_CONVERGED] = 0.0;
  state[stateOffset + STATE_RESERVED] = 0.0;
  if (isSearch) {
    ${m.armWGSL}
  } else {
    ${m.disarmWGSL}
  }`})];return{setupNodes:s.callerBoundsOnly?b.slice(2):b,gate:m,createIterationNode:t=>n(e,{id:`${c}-search-${t}-iterate`,operation:l,variant:`search-iterate`,bindings:[g(`ladders`,s.ladders,`f32`),h(`table`,s.candidateTable,`f32`),g(`state`,s.searchState,`f32`),m.binding],invocationCount:1,declarations:`${v}
fn getRowScore(row: u32) -> f32 {
  return table[tableOffset + row * TABLE_STRIDE + 1u];
}
fn getRowSingular(row: u32) -> f32 {
  return table[tableOffset + row * TABLE_STRIDE + 2u];
}`,body:`if (state[stateOffset + STATE_ACTIVE] < 0.5) {
    return;
  }
  let iteration = u32(state[stateOffset + STATE_ITERATION_COUNT]) + 1u;
  let firstRow = u32(state[stateOffset + STATE_LOWER_INTERIOR_ROW]);
  let secondRow = u32(state[stateOffset + STATE_UPPER_INTERIOR_ROW]);
  let firstScore = getRowScore(firstRow);
  let secondScore = getRowScore(secondRow);
  let firstSingular = getRowSingular(firstRow);
  let secondSingular = getRowSingular(secondRow);
  let isFirstValid = isFiniteBits(firstScore);
  let isSecondValid = isFiniteBits(secondScore);
  // mgwr: score_b <= score_d keeps the lower interval. Fewest singular rows first; an invalid
  // score loses to a valid one; with two invalid scores the comparison is false, as for NaN.
  var firstWins = isFirstValid;
  if (isFirstValid && isSecondValid) {
    firstWins = firstSingular < secondSingular || (firstSingular == secondSingular && firstScore <= secondScore);
  }
  let isAdaptive = ladders[laddersOffset + 2u] > 0.5;
  var lower = state[stateOffset + STATE_LOWER];
  var first = state[stateOffset + STATE_LOWER_INTERIOR];
  var upper = state[stateOffset + STATE_UPPER];
  var second = state[stateOffset + STATE_UPPER_INTERIOR];
  let newRow = iteration + ${s.probeUpperBound?`2u`:`1u`};
  var optimumRow = secondRow;
  var nextFirstRow = firstRow;
  var nextSecondRow = secondRow;
  var newValue = 0.0;
  if (firstWins) {
    // c = d; d = b; b = a + delta |c - a|
    optimumRow = firstRow;
    upper = second;
    second = first;
    nextSecondRow = firstRow;
    first = lower + GOLDEN_DELTA * abs(upper - lower);
    if (isAdaptive) {
      first = round(first);
    }
    newValue = first;
    nextFirstRow = newRow;
  } else {
    // a = b; b = d; d = c - delta |c - a|
    lower = first;
    first = second;
    nextFirstRow = secondRow;
    second = upper - GOLDEN_DELTA * abs(upper - lower);
    if (isAdaptive) {
      second = round(second);
    }
    newValue = second;
    nextSecondRow = newRow;
  }
  ${s.probeUpperBound?`// The initial upper-bound evaluation is table row 2. A finite endpoint keeps the
  // bracket searchable even when both current interior points have invalid AICc.
  let endpointScore = getRowScore(2u);
  let endpointValid = isFiniteBits(endpointScore);
  let interiorScore = getRowScore(optimumRow);
  if (endpointValid && (!isFiniteBits(interiorScore) || endpointScore < interiorScore)) {
    optimumRow = 2u;
  }`:``}
  // mgwr loops while |score_b - score_d| > tol; the difference is of the pair just compared.
  var keepSearching = true;
  if (isFirstValid && isSecondValid && firstSingular == secondSingular) {
    let tolerance = max(TOLERANCE_ABSOLUTE, TOLERANCE_RELATIVE * max(abs(firstScore), abs(secondScore)));
    keepSearching = abs(firstScore - secondScore) > tolerance;
  } else if (!isFirstValid && !isSecondValid) {
    keepSearching = ${s.probeUpperBound?`endpointValid`:`false`};
  }
  state[stateOffset + STATE_LOWER] = lower;
  state[stateOffset + STATE_LOWER_INTERIOR] = first;
  state[stateOffset + STATE_UPPER] = upper;
  state[stateOffset + STATE_UPPER_INTERIOR] = second;
  state[stateOffset + STATE_LOWER_INTERIOR_ROW] = f32(nextFirstRow);
  state[stateOffset + STATE_UPPER_INTERIOR_ROW] = f32(nextSecondRow);
  state[stateOffset + STATE_ITERATION_COUNT] = f32(iteration);
  state[stateOffset + STATE_OPTIMUM_ROW] = f32(optimumRow);
  if (!keepSearching || iteration >= MAXIMUM_ITERATIONS) {
    // Converged or out of iterations: no new candidate, and the later passes launch nothing.
    state[stateOffset + STATE_ACTIVE] = 0.0;
    state[stateOffset + STATE_CONVERGED] = select(0.0, 1.0, !keepSearching);
    ${m.disarmWGSL}
    return;
  }
  // Schedule the one new candidate as block iteration of the ladders and advance the GPU cursor.
  ladders[laddersOffset] = f32(iteration);
  let blockBase = laddersOffset + 1u + iteration * PARAMETER_LENGTH;
  ladders[blockBase] = ladders[laddersOffset + 1u];
  ladders[blockBase + 1u] = ladders[laddersOffset + 2u];
  ladders[blockBase + 2u] = 1.0;
  ladders[blockBase + 3u] = ladders[laddersOffset + 4u];
  ladders[blockBase + HEADER] = newValue;
  state[stateOffset + STATE_EVALUATION_COUNT] = f32(newRow + 1u);`})}}export{p as n,u as r,d as t};
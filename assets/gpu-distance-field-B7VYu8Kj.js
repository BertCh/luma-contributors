import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-raster-buffer-to-texture-BDZQU4A5.js";import{E as r,i,s as a,x as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{h as c,i as l,o as u,u as d}from"./raster-band-utils-BcTxc8_Y.js";import{o as f,t as p}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as m}from"./raster-relaxation-Di9wjtxp.js";import{t as h}from"./raster-fractional-tile-kernel-D2BjD3g1.js";var g=4294967295,_=32768;function v(e,t,n=new Float32Array(8)){u(`getGPUDistanceFieldParameterValues`,n,8);let[r,i,a,o,s,c]=t.affine;if(i!==0||o!==0)throw Error(`getGPUDistanceFieldParameterValues needs an axis-aligned grid`);let d=Math.fround(Math.abs(r)),f=Math.fround(Math.abs(s));if(!(Number.isFinite(d)&&d>0&&Number.isFinite(f)&&f>0))throw Error(`getGPUDistanceFieldParameterValues cell size must be finite and positive`);let p=t.pixelInterpretation===`point`?-.5:0,[m,h]=l(`Distance field`,e.localOrigin);return n.fill(0,0,8),n[0]=a+r*p-m,n[1]=c+s*p-h,n[2]=d,n[3]=f,n[4]=1/Math.fround(r),n[5]=1/Math.fround(s),n[6]=e.maximumDistance??1/0,n}var y=`GPUDistanceField`,b=`
fn multiplyWide(left: u32, right: u32) -> vec2<u32> {
  let leftLow = left & 0xffffu;
  let leftHigh = left >> 16u;
  let rightLow = right & 0xffffu;
  let rightHigh = right >> 16u;
  let lowLow = leftLow * rightLow;
  let lowHigh = leftLow * rightHigh;
  let highLow = leftHigh * rightLow;
  let middle = (lowLow >> 16u) + (lowHigh & 0xffffu) + (highLow & 0xffffu);
  let low = (lowLow & 0xffffu) | (middle << 16u);
  let high = leftHigh * rightHigh + (lowHigh >> 16u) + (highLow >> 16u) + (middle >> 16u);
  return vec2<u32>(low, high);
}

/** Whether (candidate + ulp / 2)^2 > mantissa * 2^exponent, for a positive normal candidate. */
fn isUpperMidpointAbove(candidate: f32, mantissa: u32, exponent: i32) -> bool {
  let bits = bitcast<u32>(candidate);
  let candidateExponent = i32((bits >> 23u) & 0xffu);
  let odd = (((bits & 0x7fffffu) | 0x800000u) << 1u) | 1u;
  // Midpoint^2 = odd^2 * 2^(2 * candidateExponent - 302); compare odd^2 with mantissa * 2^shift.
  let square = multiplyWide(odd, odd);
  let shift = exponent - 2 * candidateExponent + 302;
  if (shift < 0) { return true; }
  if (shift >= 64) { return false; }
  var scaled: vec2<u32>;
  if (shift == 0) {
    scaled = vec2<u32>(mantissa, 0u);
  } else if (shift < 32) {
    scaled = vec2<u32>(mantissa << u32(shift), mantissa >> u32(32 - shift));
  } else {
    let highShift = u32(shift - 32);
    if (highShift > 0u && (mantissa >> (32u - highShift)) != 0u) { return false; }
    scaled = vec2<u32>(0u, mantissa << highShift);
  }
  return square.y > scaled.y || (square.y == scaled.y && square.x > scaled.x);
}

fn getRoundedSquareRoot(mantissa: u32, exponent: i32) -> f32 {
  if (mantissa == 0u) { return 0.0; }
  var root = sqrt(f32(mantissa) * exp2(f32(exponent)));
  for (var step = 0u; step < 8u; step++) {
    if (!isUpperMidpointAbove(root, mantissa, exponent)) {
      root = bitcast<f32>(bitcast<u32>(root) + 1u);
    } else if (isUpperMidpointAbove(bitcast<f32>(bitcast<u32>(root) - 1u), mantissa, exponent)) {
      root = bitcast<f32>(bitcast<u32>(root) - 1u);
    } else {
      break;
    }
  }
  return root;
}`,x=64;function S(e,t){return`
const GRID_WIDTH: u32 = ${e.width}u;
const GRID_HEIGHT: u32 = ${e.height}u;
const NONE: u32 = ${g}u;

fn getAbsoluteDifference(left: u32, right: u32) -> u32 {
  return select(right - left, left - right, left > right);
}
${t?`
fn isIsotropic() -> bool {
  return settings[settingsOffset + 2u] == settings[settingsOffset + 3u];
}

fn getAnisotropicKey(dx: u32, dy: u32) -> f32 {
  let groundX = settings[settingsOffset + 2u] * f32(dx);
  let groundY = settings[settingsOffset + 3u] * f32(dy);
  return groundX * groundX + groundY * groundY;
}

fn isCloser(dxA: u32, dyA: u32, idA: u32, dxB: u32, dyB: u32, idB: u32) -> bool {
  if (isIsotropic()) {
    let keyA = dxA * dxA + dyA * dyA;
    let keyB = dxB * dxB + dyB * dyB;
    if (keyA != keyB) { return keyA < keyB; }
  } else {
    let keyA = getAnisotropicKey(dxA, dyA);
    let keyB = getAnisotropicKey(dxB, dyB);
    if (keyA != keyB) { return keyA < keyB; }
  }
  return idA < idB;
}`:``}`}function C(e,t){let{id:n,grid:r}=t,i=r.width*r.height,a=[{name:`cellSeeds`,view:t.cellSeeds,type:`u32`,access:`read_write`}];t.seedMask&&a.unshift({name:`seedMask`,view:t.seedMask,type:`u32`,access:`read`});let o=[f(e,{id:`${n}-seed-clear`,operation:y,variant:`seed-clear`,bindings:a,invocationCount:i,body:t.seedMask?`cellSeeds[cellSeedsOffset + index] = seedMask[seedMaskOffset + index] - 1u;`:`cellSeeds[cellSeedsOffset + index] = ${g}u;`})],s=t.seedPositions;if(s&&s.length>0){let i=[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`positions`,view:s,type:`f32`,access:`read`}];t.seedIds&&i.push({name:`seedIds`,view:t.seedIds,type:`u32`,access:`read`}),t.seedCount&&i.push({name:`seedCount`,view:t.seedCount,type:`u32`,access:`read`}),i.push({name:`cellSeedBits`,view:t.cellSeeds,type:`atomic<u32>`,access:`read_write`}),o.push(f(e,{id:`${n}-seed-scatter`,operation:y,variant:`seed-scatter`,bindings:i,invocationCount:s.length,declarations:S(r,!1),body:`${t.seedCount?`if (index >= seedCount[seedCountOffset]) { return; }`:``}
  let seedId = ${t.seedIds?`seedIds[seedIdsOffset + index]`:`index`};
  if (seedId == NONE) { return; }
  let column = floor((positions[positionsOffset + index * 2u] - settings[settingsOffset]) * settings[settingsOffset + 4u]);
  let row = floor((positions[positionsOffset + index * 2u + 1u] - settings[settingsOffset + 1u]) * settings[settingsOffset + 5u]);
  // Negated comparisons also reject NaN and infinite coordinates.
  if (!(column >= 0.0 && column < f32(GRID_WIDTH) && row >= 0.0 && row < f32(GRID_HEIGHT))) { return; }
  atomicMin(&cellSeedBits[cellSeedBitsOffset + u32(row) * GRID_WIDTH + u32(column)], seedId);`}))}return o}function w(e){let t=Math.ceil(e.height/32),n=Math.ceil(e.width/32);return{columnBandCount:t,rowBandCount:n,mergeRoundCount:n>1?Math.ceil(Math.log2(n)):0}}function T(e,t){let{grid:n,scratch:r}=t,{columnBandCount:i}=w(n),a=`${S(n,!1)}
const BAND_SIZE: u32 = 32u;
const COLUMN_BAND_COUNT: u32 = ${i}u;`,o=i*n.width;return[f(e,{id:`${t.id}-column-bands`,operation:y,variant:`exact-column-bands`,bindings:[{name:`cellSeeds`,view:t.cellSeeds,type:`u32`,access:`read`},{name:`bandSeeds`,view:r.bandSeeds,type:`u32`,access:`read_write`}],invocationCount:o,declarations:a,body:`let column = index % GRID_WIDTH;
  let rowStart = (index / GRID_WIDTH) * BAND_SIZE;
  let rowEnd = min(rowStart + BAND_SIZE, GRID_HEIGHT);
  var topRow = NONE;
  var topId = NONE;
  var bottomRow = NONE;
  var bottomId = NONE;
  for (var row = rowStart; row < rowEnd; row++) {
    let seedId = cellSeeds[cellSeedsOffset + row * GRID_WIDTH + column];
    if (seedId != NONE) {
      if (topRow == NONE) {
        topRow = row;
        topId = seedId;
      }
      bottomRow = row;
      bottomId = seedId;
    }
  }
  let base = bandSeedsOffset + 4u * index;
  bandSeeds[base] = topRow;
  bandSeeds[base + 1u] = topId;
  bandSeeds[base + 2u] = bottomRow;
  bandSeeds[base + 3u] = bottomId;`}),f(e,{id:`${t.id}-column-carry`,operation:y,variant:`exact-column-carry`,bindings:[{name:`bandSeeds`,view:r.bandSeeds,type:`u32`,access:`read`},{name:`bandCarries`,view:r.bandCarries,type:`u32`,access:`read_write`}],invocationCount:n.width,workgroupSize:x,declarations:a,body:`let column = index;
  var lastRow = NONE;
  var lastId = NONE;
  for (var band = 0u; band < COLUMN_BAND_COUNT; band++) {
    let base = 4u * (band * GRID_WIDTH + column);
    bandCarries[bandCarriesOffset + base] = lastRow;
    bandCarries[bandCarriesOffset + base + 1u] = lastId;
    let bottomRow = bandSeeds[bandSeedsOffset + base + 2u];
    if (bottomRow != NONE) {
      lastRow = bottomRow;
      lastId = bandSeeds[bandSeedsOffset + base + 3u];
    }
  }
  lastRow = NONE;
  lastId = NONE;
  for (var step = 0u; step < COLUMN_BAND_COUNT; step++) {
    let base = 4u * ((COLUMN_BAND_COUNT - 1u - step) * GRID_WIDTH + column);
    bandCarries[bandCarriesOffset + base + 2u] = lastRow;
    bandCarries[bandCarriesOffset + base + 3u] = lastId;
    let topRow = bandSeeds[bandSeedsOffset + base];
    if (topRow != NONE) {
      lastRow = topRow;
      lastId = bandSeeds[bandSeedsOffset + base + 1u];
    }
  }`}),f(e,{id:`${t.id}-column-fill`,operation:y,variant:`exact-column-fill`,bindings:[{name:`cellSeeds`,view:t.cellSeeds,type:`u32`,access:`read`},{name:`bandCarries`,view:r.bandCarries,type:`u32`,access:`read`},{name:`columnRows`,view:t.columnRows,type:`u32`,access:`read_write`},{name:`columnIds`,view:t.columnIds,type:`u32`,access:`read_write`}],invocationCount:o,declarations:a,body:`let column = index % GRID_WIDTH;
  let rowStart = (index / GRID_WIDTH) * BAND_SIZE;
  let rowEnd = min(rowStart + BAND_SIZE, GRID_HEIGHT);
  let base = bandCarriesOffset + 4u * index;
  var lastRow = bandCarries[base];
  var lastId = bandCarries[base + 1u];
  for (var row = rowStart; row < rowEnd; row++) {
    let cell = row * GRID_WIDTH + column;
    let seedId = cellSeeds[cellSeedsOffset + cell];
    if (seedId != NONE) {
      lastRow = row;
      lastId = seedId;
    }
    columnRows[columnRowsOffset + cell] = lastRow;
    columnIds[columnIdsOffset + cell] = lastId;
  }
  lastRow = bandCarries[base + 2u];
  lastId = bandCarries[base + 3u];
  for (var row = rowEnd; row > rowStart; row--) {
    let cell = (row - 1u) * GRID_WIDTH + column;
    let seedId = cellSeeds[cellSeedsOffset + cell];
    if (seedId != NONE) {
      lastRow = row - 1u;
      lastId = seedId;
    }
    if (lastRow == NONE) { continue; }
    let forwardRow = columnRows[columnRowsOffset + cell];
    var useBackward = forwardRow == NONE;
    if (!useBackward) {
      let forwardDistance = row - 1u - forwardRow;
      let backwardDistance = lastRow - (row - 1u);
      useBackward = backwardDistance < forwardDistance ||
        (backwardDistance == forwardDistance && lastId < columnIds[columnIdsOffset + cell]);
    }
    if (useBackward) {
      columnRows[columnRowsOffset + cell] = lastRow;
      columnIds[columnIdsOffset + cell] = lastId;
    }
  }`})]}var E=`
/** Whether the candidate in column q beats the candidate in column v at cell x of this row. */
fn beatsAt(q: u32, qDy: u32, qId: u32, v: u32, vDy: u32, vId: u32, x: u32) -> bool {
  return isCloser(getAbsoluteDifference(x, q), qDy, qId, getAbsoluteDifference(x, v), vDy, vId);
}

fn getEntrySlot(column: u32, row: u32) -> u32 {
  return column * GRID_HEIGHT + row;
}

/**
 * Felzenszwalb-Huttenlocher insertion of the candidate in column q into the envelope whose tail is
 * 'tail': pops every tail entry q beats at its start (marking it off the envelope) and returns
 * vec2(new tail, start of q), start GRID_WIDTH when q never wins. The start is the first cell
 * after the tail's start where q wins, found by binary search with isCloser, so ties break exactly
 * on the smallest seed ID.
 */
fn insertEntry(q: u32, row: u32, tailIn: u32) -> vec2<u32> {
  let rowBase = row * GRID_WIDTH;
  let qDy = getAbsoluteDifference(columnRows[columnRowsOffset + rowBase + q], row);
  let qId = columnIds[columnIdsOffset + rowBase + q];
  var tail = tailIn;
  loop {
    if (tail == NONE) { return vec2<u32>(NONE, 0u); }
    let tailSlot = getEntrySlot(tail, row);
    let vStart = entryStarts[entryStartsOffset + tailSlot];
    let vDy = getAbsoluteDifference(columnRows[columnRowsOffset + rowBase + tail], row);
    let vId = columnIds[columnIdsOffset + rowBase + tail];
    if (beatsAt(q, qDy, qId, tail, vDy, vId, vStart)) {
      entryStarts[entryStartsOffset + tailSlot] = NONE;
      tail = entryPrevious[entryPreviousOffset + tailSlot];
      continue;
    }
    // First cell in (vStart, GRID_WIDTH) where q wins, or GRID_WIDTH when q never wins.
    var low = vStart + 1u;
    var high = GRID_WIDTH;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (beatsAt(q, qDy, qId, tail, vDy, vId, middle)) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    return vec2<u32>(tail, low);
  }
}`;function D(e,t){let{grid:n,scratch:r}=t,{rowBandCount:i,mergeRoundCount:a}=w(n),o=n.width*n.height,s=i*n.height,c=`${S(n,!0)}
const BAND_SIZE: u32 = 32u;
const ROW_BAND_COUNT: u32 = ${i}u;`,l=[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`columnRows`,view:t.columnRows,type:`u32`,access:`read`},{name:`columnIds`,view:t.columnIds,type:`u32`,access:`read`},{name:`entryNext`,view:r.entryNext,type:`u32`,access:`read_write`},{name:`entryPrevious`,view:r.entryPrevious,type:`u32`,access:`read_write`},{name:`entryStarts`,view:r.entryStarts,type:`u32`,access:`read_write`},{name:`blockEnds`,view:r.blockEnds,type:`u32`,access:`read_write`}],u=[f(e,{id:`${t.id}-row-bands`,operation:y,variant:`exact-row-bands`,bindings:l,invocationCount:s,declarations:`${c}\n${E}`,body:`let row = index % GRID_HEIGHT;
  let band = index / GRID_HEIGHT;
  let rowBase = row * GRID_WIDTH;
  let columnStart = band * BAND_SIZE;
  let columnEnd = min(columnStart + BAND_SIZE, GRID_WIDTH);
  var head = NONE;
  var tail = NONE;
  for (var q = columnStart; q < columnEnd; q++) {
    let slot = getEntrySlot(q, row);
    entryStarts[entryStartsOffset + slot] = NONE;
    if (columnRows[columnRowsOffset + rowBase + q] == NONE) { continue; }
    let inserted = insertEntry(q, row, tail);
    tail = inserted.x;
    if (inserted.y < GRID_WIDTH) {
      entryStarts[entryStartsOffset + slot] = inserted.y;
      entryPrevious[entryPreviousOffset + slot] = tail;
      entryNext[entryNextOffset + slot] = NONE;
      if (tail == NONE) {
        head = q;
      } else {
        entryNext[entryNextOffset + getEntrySlot(tail, row)] = q;
      }
      tail = q;
    }
  }
  blockEnds[blockEndsOffset + 2u * index] = head;
  blockEnds[blockEndsOffset + 2u * index + 1u] = tail;`})];if(a>0){let r=m(e,{id:`${t.id}-row-merge-rounds`,operation:y,groupSize:a});u.push(r.writeNode);for(let o=0;o<a;o++){let a=2**o,s=Math.floor((i+a-1)/(2*a))*n.height;u.push(h(e,{id:`${t.id}-row-merge-${o}`,operation:y,bindings:[...l,{name:`mergeRound`,view:r.getSlot(o),type:`u32`,access:`read`}],invocationCount:s,rowCount:`((ROW_BAND_COUNT + (1u << mergeRound[mergeRoundOffset]) - 1u) / (2u << mergeRound[mergeRoundOffset])) * GRID_HEIGHT`,declarations:`${c}\n${E}`,body:`let blockBands = 1u << mergeRound[mergeRoundOffset];
  let row = index % GRID_HEIGHT;
  let leftBand = (index / GRID_HEIGHT) * 2u * blockBands;
  let rightBand = leftBand + blockBands;
  if (rightBand >= ROW_BAND_COUNT) { return; }
  let leftEnds = blockEndsOffset + 2u * (leftBand * GRID_HEIGHT + row);
  let rightEnds = blockEndsOffset + 2u * (rightBand * GRID_HEIGHT + row);
  let leftHead = blockEnds[leftEnds];
  let rightTail = blockEnds[rightEnds + 1u];
  var tail = blockEnds[leftEnds + 1u];
  var current = blockEnds[rightEnds];
  var mergedHead = NONE;
  var reachedLocalState = false;
  loop {
    if (current == NONE) { break; }
    let slot = getEntrySlot(current, row);
    let localPrevious = entryPrevious[entryPreviousOffset + slot];
    let localStart = entryStarts[entryStartsOffset + slot];
    let localNext = entryNext[entryNextOffset + slot];
    let inserted = insertEntry(current, row, tail);
    tail = inserted.x;
    if (inserted.y >= GRID_WIDTH) {
      entryStarts[entryStartsOffset + slot] = NONE;
      current = localNext;
      continue;
    }
    entryStarts[entryStartsOffset + slot] = inserted.y;
    entryPrevious[entryPreviousOffset + slot] = tail;
    if (tail == NONE) {
      mergedHead = current;
    } else {
      entryNext[entryNextOffset + getEntrySlot(tail, row)] = current;
    }
    if (tail == localPrevious && inserted.y == localStart) {
      reachedLocalState = true;
      break;
    }
    tail = current;
    current = localNext;
  }
  // The left head survives unless the merge popped the whole left envelope.
  if (leftHead != NONE && entryStarts[entryStartsOffset + getEntrySlot(leftHead, row)] != NONE) {
    mergedHead = leftHead;
  }
  blockEnds[leftEnds] = mergedHead;
  blockEnds[leftEnds + 1u] = select(tail, rightTail, reachedLocalState);`}))}}let d=`const GRID_WIDTH: u32 = ${n.width}u;
const GRID_HEIGHT: u32 = ${n.height}u;
const NONE: u32 = ${g}u;
const BAND_SIZE: u32 = 32u;
const ROW_BAND_COUNT: u32 = ${i}u;`;return u.push(p(e,{id:`${t.id}-row-marker-clear`,operation:y,view:r.startMarkers,type:`u32`,value:`0u`}),f(e,{id:`${t.id}-row-start-markers`,operation:y,variant:`exact-row-start-markers`,bindings:[{name:`entryStarts`,view:r.entryStarts,type:`u32`,access:`read`},{name:`startMarkers`,view:r.startMarkers,type:`u32`,access:`read_write`}],invocationCount:o,declarations:d,body:`let start = entryStarts[entryStartsOffset + index];
  if (start == NONE) { return; }
  let column = index / GRID_HEIGHT;
  let row = index % GRID_HEIGHT;
  startMarkers[startMarkersOffset + row * GRID_WIDTH + start] = column + 1u;`}),f(e,{id:`${t.id}-row-marker-bands`,operation:y,variant:`exact-row-marker-bands`,bindings:[{name:`startMarkers`,view:r.startMarkers,type:`u32`,access:`read`},{name:`bandMarkers`,view:r.bandMarkers,type:`u32`,access:`read_write`}],invocationCount:s,declarations:d,body:`let row = index % GRID_HEIGHT;
  let columnStart = (index / GRID_HEIGHT) * BAND_SIZE;
  let columnEnd = min(columnStart + BAND_SIZE, GRID_WIDTH);
  var maximum = 0u;
  for (var x = columnStart; x < columnEnd; x++) {
    maximum = max(maximum, startMarkers[startMarkersOffset + row * GRID_WIDTH + x]);
  }
  bandMarkers[bandMarkersOffset + index] = maximum;`}),f(e,{id:`${t.id}-row-marker-carry`,operation:y,variant:`exact-row-marker-carry`,bindings:[{name:`bandMarkers`,view:r.bandMarkers,type:`u32`,access:`read_write`}],invocationCount:n.height,workgroupSize:x,declarations:d,body:`var running = 0u;
  for (var band = 0u; band < ROW_BAND_COUNT; band++) {
    let slot = bandMarkersOffset + band * GRID_HEIGHT + index;
    let bandMaximum = bandMarkers[slot];
    bandMarkers[slot] = running;
    running = max(running, bandMaximum);
  }`}),f(e,{id:`${t.id}-row-color`,operation:y,variant:`exact-row-color`,bindings:[{name:`columnRows`,view:t.columnRows,type:`u32`,access:`read`},{name:`columnIds`,view:t.columnIds,type:`u32`,access:`read`},{name:`startMarkers`,view:r.startMarkers,type:`u32`,access:`read`},{name:`bandMarkers`,view:r.bandMarkers,type:`u32`,access:`read`},{name:`nearestCells`,view:t.nearestCells,type:`u32`,access:`read_write`},{name:`nearestIds`,view:t.nearestIds,type:`u32`,access:`read_write`}],invocationCount:s,declarations:d,body:`let row = index % GRID_HEIGHT;
  let rowBase = row * GRID_WIDTH;
  let columnStart = (index / GRID_HEIGHT) * BAND_SIZE;
  let columnEnd = min(columnStart + BAND_SIZE, GRID_WIDTH);
  var running = bandMarkers[bandMarkersOffset + index];
  for (var x = columnStart; x < columnEnd; x++) {
    let cell = rowBase + x;
    running = max(running, startMarkers[startMarkersOffset + cell]);
    if (running == 0u) {
      nearestCells[nearestCellsOffset + cell] = NONE;
      nearestIds[nearestIdsOffset + cell] = NONE;
      continue;
    }
    let column = running - 1u;
    nearestCells[nearestCellsOffset + cell] = columnRows[columnRowsOffset + rowBase + column] * GRID_WIDTH + column;
    nearestIds[nearestIdsOffset + cell] = columnIds[columnIdsOffset + rowBase + column];
  }`})),u}function O(e,t){let n=[],r=1;for(;r<Math.max(e.width,e.height);)r*=2;for(let e=r/2;e>=1;e/=2)n.push(e);return t===2?n.push(2,1):t===1&&n.push(1),n}function k(e,n){let{id:r,grid:i,buffers:a}=n,o=i.width*i.height,s=[f(e,{id:`${r}-jump-flood-initialize`,operation:y,variant:`jump-flood-initialize`,bindings:[{name:`cellSeeds`,view:n.cellSeeds,type:`u32`,access:`read`},{name:`targetCells`,view:a[0].cells,type:`u32`,access:`read_write`},{name:`targetIds`,view:a[0].ids,type:`u32`,access:`read_write`}],invocationCount:o,body:`let seedId = cellSeeds[cellSeedsOffset + index];
  targetCells[targetCellsOffset + index] = select(${g}u, index, seedId != ${g}u);
  targetIds[targetIdsOffset + index] = seedId;`})],c=O(i,n.refinementIterationCount);if(c.length===0)return{nodes:s,result:a[0]};let l=t(e,`${r}-jump-flood-step-words`,`uint32`,c.length*64);s.push(f(e,{id:`${r}-jump-flood-steps`,operation:y,variant:`jump-flood-steps`,bindings:[{name:`stepWords`,view:l,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const STEPS = array<u32, ${c.length}>(${c.map(e=>`${e}u`).join(`, `)});`,body:`for (var stepIndex = 0u; stepIndex < ${c.length}u; stepIndex++) {
    stepWords[stepWordsOffset + stepIndex * 64u] = STEPS[stepIndex];
  }`}));let u=a[0];for(let[t]of c.entries()){let c=u===a[0]?a[1]:a[0];s.push(f(e,{id:`${r}-jump-flood-${t}`,operation:y,variant:`jump-flood-step`,bindings:[{name:`settings`,view:n.settings,type:`f32`,access:`read`},{name:`sourceCells`,view:u.cells,type:`u32`,access:`read`},{name:`sourceIds`,view:u.ids,type:`u32`,access:`read`},{name:`targetCells`,view:c.cells,type:`u32`,access:`read_write`},{name:`targetIds`,view:c.ids,type:`u32`,access:`read_write`},{name:`jumpStep`,view:e.createDataView(l.buffer,{format:`uint32`,length:1,byteOffset:l.byteOffset+t*64*4}),type:`u32`,access:`read`}],invocationCount:o,declarations:S(i,!0),body:`let step = i32(jumpStep[jumpStepOffset]);
  let x = index % GRID_WIDTH;
  let y = index / GRID_WIDTH;
  var bestCell = sourceCells[sourceCellsOffset + index];
  var bestId = sourceIds[sourceIdsOffset + index];
  for (var offsetY = -1; offsetY <= 1; offsetY++) {
    for (var offsetX = -1; offsetX <= 1; offsetX++) {
      let neighborX = i32(x) + offsetX * step;
      let neighborY = i32(y) + offsetY * step;
      if ((offsetX == 0 && offsetY == 0) || neighborX < 0 || neighborY < 0 ||
          neighborX >= i32(GRID_WIDTH) || neighborY >= i32(GRID_HEIGHT)) { continue; }
      let neighbor = u32(neighborY) * GRID_WIDTH + u32(neighborX);
      let candidateCell = sourceCells[sourceCellsOffset + neighbor];
      if (candidateCell == NONE) { continue; }
      let candidateId = sourceIds[sourceIdsOffset + neighbor];
      let candidateDx = getAbsoluteDifference(x, candidateCell % GRID_WIDTH);
      let candidateDy = getAbsoluteDifference(y, candidateCell / GRID_WIDTH);
      if (bestCell == NONE || isCloser(
        candidateDx, candidateDy, candidateId,
        getAbsoluteDifference(x, bestCell % GRID_WIDTH), getAbsoluteDifference(y, bestCell / GRID_WIDTH), bestId
      )) {
        bestCell = candidateCell;
        bestId = candidateId;
      }
    }
  }
  targetCells[targetCellsOffset + index] = bestCell;
  targetIds[targetIdsOffset + index] = bestId;`})),u=c}return{nodes:s,result:u}}function A(e,t){let{grid:n}=t,r=[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`nearestCells`,view:t.nearestCells,type:`u32`,access:`read`},{name:`nearestIds`,view:t.nearestIds,type:`u32`,access:`read`},{name:`distanceBits`,view:t.distances,type:`u32`,access:`read_write`}];return t.allocation&&r.push({name:`allocation`,view:t.allocation,type:`u32`,access:`read_write`}),t.nearestCellsOutput&&r.push({name:`nearestCellsOutput`,view:t.nearestCellsOutput,type:`u32`,access:`read_write`}),t.withinDistance&&r.push({name:`withinDistance`,view:t.withinDistance,type:`u32`,access:`read_write`}),f(e,{id:`${t.id}-finalize`,operation:y,variant:`finalize`,bindings:r,invocationCount:n.width*n.height,declarations:`${S(n,!0)}
${b}`,body:`var nearestCell = nearestCells[nearestCellsOffset + index];
  var seedId = nearestIds[nearestIdsOffset + index];
  var bits = 0x7f800000u;
  if (nearestCell != NONE) {
    let dx = getAbsoluteDifference(index % GRID_WIDTH, nearestCell % GRID_WIDTH);
    let dy = getAbsoluteDifference(index / GRID_WIDTH, nearestCell / GRID_WIDTH);
    var distance: f32;
    if (isIsotropic()) {
      // Exact integer key, so the only roundings are the square root and the scale.
      distance = settings[settingsOffset + 2u] * getRoundedSquareRoot(dx * dx + dy * dy, 0);
    } else {
      let key = getAnisotropicKey(dx, dy);
      let keyBits = bitcast<u32>(key);
      let keyExponent = (keyBits >> 23u) & 0xffu;
      if (keyExponent == 0u || keyExponent == 0xffu) {
        distance = sqrt(key);
      } else {
        distance = getRoundedSquareRoot((keyBits & 0x7fffffu) | 0x800000u, i32(keyExponent) - 150);
      }
    }
    if (distance > settings[settingsOffset + 6u]) {
      nearestCell = NONE;
      seedId = NONE;
    } else {
      bits = bitcast<u32>(distance);
    }
  }
  if (nearestCell == NONE) { seedId = NONE; }
  distanceBits[distanceBitsOffset + index] = bits;
  ${t.allocation?`allocation[allocationOffset + index] = seedId;`:``}
  ${t.nearestCellsOutput?`nearestCellsOutput[nearestCellsOutputOffset + index] = nearestCell;`:``}
  ${t.withinDistance?`withinDistance[withinDistanceOffset + index] = select(0u, 1u, nearestCell != NONE);`:``}`})}var j=class{id;props;mode;jumpFloodRefinementIterationCount;cellCount;constructor(e){this.id=e.id??`distance-field`,this.props=e,this.mode=e.mode??`exact`,this.jumpFloodRefinementIterationCount=e.jumpFloodRefinementIterationCount??1;let{id:t}=this,{output:n}=e;this.cellCount=d(t,e.grid);let{width:i,height:a}=e.grid;for(let[e,n]of[[`width`,i],[`height`,a]])if(n>32768)throw Error(`${t} grid ${e} must be an integer from 1 to ${_}`);if(![`exact`,`jump-flood`].includes(this.mode))throw Error(`${t} mode must be exact or jump-flood`);if(![0,1,2].includes(this.jumpFloodRefinementIterationCount))throw Error(`${t} jumpFloodRefinementIterationCount must be 0, 1, or 2`);if(c(t,e.parameters,8),!e.seedPositions&&!e.seedMask)throw Error(`${t} requires seedPositions or seedMask`);if(e.seedPositions&&r(e.seedPositions,[`float32x2`],`${t} seedPositions`),(e.seedIds||e.seedCount)&&!e.seedPositions)throw Error(`${t} seedIds and seedCount require seedPositions`);for(let[i,a]of[[`seedIds`,e.seedIds],[`seedCount`,e.seedCount],[`seedMask`,e.seedMask],[`output.allocation`,n.allocation],[`output.nearestCells`,n.nearestCells],[`output.withinDistance`,n.withinDistance]])a&&r(a,[`uint32`],`${t} ${i}`);if(r(n.distances,[`float32`],`${t} output.distances`),e.seedIds&&e.seedIds.length!==e.seedPositions?.length)throw Error(`${t} seedIds length must equal seedPositions length`);if(e.seedCount&&e.seedCount.length!==1)throw Error(`${t} seedCount must contain exactly one row`);for(let[r,i]of[[`seedMask`,e.seedMask],[`output.distances`,n.distances],[`output.allocation`,n.allocation],[`output.nearestCells`,n.nearestCells],[`output.withinDistance`,n.withinDistance]])if(i&&i.length!==this.cellCount)throw Error(`${t} ${r} must contain one row per cell`);if(n.texture&&(n.texture.format!==`r32float`||n.texture.width!==i||n.texture.height!==a))throw Error(`${t} output.texture must be r32float with a width x height extent`);let o=[n.distances,n.allocation,n.nearestCells,n.withinDistance];s(t,o,[e.parameters,e.seedPositions,e.seedIds,e.seedCount,e.seedMask]);let l=o.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(l).size!==l.length)throw Error(`${t} outputs must use separate buffers`)}getCommandNodes(r){let s=this.props,c=w(s.grid);e(r,this.id,[s.parameters,s.seedPositions,s.seedIds,s.seedCount,s.seedMask,...a(s.output)],{"cell planes":this.cellCount*4,"column band seeds and carries":this.mode===`exact`?16*c.columnBandCount*s.grid.width:0,"row block ends":this.mode===`exact`?8*c.rowBandCount*s.grid.height:0,"merge round slot table":this.mode===`exact`&&c.mergeRoundCount>0?(c.mergeRoundCount+1)*256:0,"jump flood step table":this.mode===`jump-flood`?O(s.grid,this.jumpFloodRefinementIterationCount).length*256:0});let{id:l,props:u,cellCount:d}=this,{output:f,parameters:p}=u,m={width:u.grid.width,height:u.grid.height};if(o(l,r,[p,u.seedPositions,u.seedIds,u.seedCount,u.seedMask,f.distances,f.allocation,f.nearestCells,f.withinDistance]),f.texture&&f.texture.texture.graph!==r)throw Error(`${l} views must belong to the target graph`);let h=e=>t(r,`${l}-${e}`,`uint32`,d),g=h(`cell-seeds`),_=C(r,{id:l,grid:m,settings:p,cellSeeds:g,seedMask:u.seedMask,seedPositions:u.seedPositions,seedIds:u.seedIds,seedCount:u.seedCount}),v;if(this.mode===`exact`){let e=h(`column-rows`),n=h(`column-ids`);v={cells:h(`nearest-cells`),ids:h(`nearest-ids`)};let i=w(m),a=(e,n)=>t(r,`${l}-${e}`,`uint32`,n);_.push(...T(r,{id:l,grid:m,cellSeeds:g,columnRows:e,columnIds:n,scratch:{bandSeeds:a(`column-band-seeds`,4*i.columnBandCount*m.width),bandCarries:a(`column-band-carries`,4*i.columnBandCount*m.width)}}),...D(r,{id:l,grid:m,settings:p,columnRows:e,columnIds:n,scratch:{entryNext:h(`entry-next`),entryPrevious:h(`entry-previous`),entryStarts:h(`entry-starts`),blockEnds:a(`block-ends`,2*i.rowBandCount*m.height),startMarkers:h(`start-markers`),bandMarkers:a(`band-markers`,i.rowBandCount*m.height)},nearestCells:v.cells,nearestIds:v.ids}))}else{let e=k(r,{id:l,grid:m,settings:p,cellSeeds:g,buffers:[{cells:h(`jump-flood-cells-0`),ids:h(`jump-flood-ids-0`)},{cells:h(`jump-flood-cells-1`),ids:h(`jump-flood-ids-1`)}],refinementIterationCount:this.jumpFloodRefinementIterationCount});_.push(...e.nodes),v=e.result}if(_.push(A(r,{id:l,grid:m,settings:p,nearestCells:v.cells,nearestIds:v.ids,distances:f.distances,allocation:f.allocation,nearestCellsOutput:f.nearestCells,withinDistance:f.withinDistance})),f.texture){let e=f.texture;_.push(...i(r,()=>new n({id:`${l}-texture`,input:{id:`${l}-texture-band`,format:`float32`,storage:{kind:`buffer`,values:f.distances}},output:e}).addToGraph(r)))}return _}};export{v as i,w as n,_ as r,j as t};
import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-sort-xRm8l5yn.js";import{S as n,T as r}from"./gpu-contributor-utils-BIwLQON5.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";var a=256;function o(e,t){let{positions:n,neighbors:r,distances:o,sortKeys:s,sortZones:c,zones:l,listLength:u,separationWGSL:d}=t;return i(e,{id:t.id,operation:t.operation,variant:`nearest`,bindings:[{name:`positions`,view:n,type:`f32`,access:`read`},{name:`neighbors`,view:r,type:`u32`,access:`read_write`},{name:`distances`,view:o,type:`u32`,access:`read_write`},{name:`sortKeys`,view:s,type:`u32`,access:`read_write`},{name:`sortZones`,view:c,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`${d}
const ZONES: u32 = ${l}u;
const LIST: u32 = ${u}u;
const RADIX_BINS: u32 = ${a}u;
// Separation keys are non-negative f32, so their bits order as u32 keys.
fn getZoneSeparationKey(zone: u32, centerX: f32, centerY: f32) -> u32 {
  let zoneX = positions[positionsOffset + zone * 2u];
  let zoneY = positions[positionsOffset + zone * 2u + 1u];
  return bitcast<u32>(getSeparationKey(vec2f(zoneX, zoneY), vec2f(centerX, centerY)));
}`,body:`var histogram: array<u32, RADIX_BINS>;
  let centerX = positions[positionsOffset + index * 2u];
  let centerY = positions[positionsOffset + index * 2u + 1u];
  let row = index * LIST;
  var threshold = 0xffffffffu;
  var equalCount = LIST;
  if (LIST < ZONES) {
    var prefix = 0u;
    var prefixMask = 0u;
    var rank = LIST - 1u;
    for (var level = 0u; level < 4u; level++) {
      let shift = 24u - 8u * level;
      for (var bin = 0u; bin < RADIX_BINS; bin++) {
        histogram[bin] = 0u;
      }
      for (var zone = 0u; zone < ZONES; zone++) {
        let key = getZoneSeparationKey(zone, centerX, centerY);
        if ((key & prefixMask) == prefix) {
          let bin = (key >> shift) & 255u;
          histogram[bin] = histogram[bin] + 1u;
        }
      }
      var digit = 0u;
      loop {
        if (digit == RADIX_BINS - 1u || rank < histogram[digit]) {
          break;
        }
        rank -= histogram[digit];
        digit++;
      }
      prefix = prefix | (digit << shift);
      prefixMask = prefixMask | (255u << shift);
    }
    threshold = prefix;
    equalCount = rank + 1u;
  }
  var filled = 0u;
  var equalTaken = 0u;
  for (var zone = 0u; zone < ZONES; zone++) {
    if (filled >= LIST) {
      break;
    }
    let key = getZoneSeparationKey(zone, centerX, centerY);
    var take = key < threshold;
    if (!take && key == threshold && equalTaken < equalCount) {
      take = true;
      equalTaken++;
    }
    if (take) {
      distances[distancesOffset + row + filled] = key;
      neighbors[neighborsOffset + row + filled] = zone;
      filled++;
    }
  }
  for (var shift = 0u; shift < 32u; shift += 16u) {
    // Digit shift: the table rows to the sort scratch.
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      histogram[bin] = 0u;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let bin = (distances[distancesOffset + row + slot] >> shift) & 255u;
      histogram[bin] = histogram[bin] + 1u;
    }
    var running = 0u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      let count = histogram[bin];
      histogram[bin] = running;
      running += count;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let key = distances[distancesOffset + row + slot];
      let bin = (key >> shift) & 255u;
      let destination = histogram[bin];
      histogram[bin] = destination + 1u;
      sortKeys[sortKeysOffset + row + destination] = key;
      sortZones[sortZonesOffset + row + destination] = neighbors[neighborsOffset + row + slot];
    }
    // Digit shift + 8: the sort scratch back to the table rows.
    let nextShift = shift + 8u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      histogram[bin] = 0u;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let bin = (sortKeys[sortKeysOffset + row + slot] >> nextShift) & 255u;
      histogram[bin] = histogram[bin] + 1u;
    }
    running = 0u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      let count = histogram[bin];
      histogram[bin] = running;
      running += count;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let key = sortKeys[sortKeysOffset + row + slot];
      let bin = (key >> nextShift) & 255u;
      let destination = histogram[bin];
      histogram[bin] = destination + 1u;
      distances[distancesOffset + row + destination] = key;
      neighbors[neighborsOffset + row + destination] = sortZones[sortZonesOffset + row + slot];
    }
  }`})}var s=2048;function c(e,t,i,a=`auto`){r(e,[`auto`,`indexed`,`full-scan`].includes(a),`nearestZoneSearch must be auto, indexed or full-scan`);let o=a===`auto`?i<t&&i<=65?`indexed`:`full-scan`:a;return o===`indexed`&&n(e,`nearestZoneSearch indexed listLength`,i,65),{nearestZoneSearch:o,estimatedFullScanSeparationCount:t*t*(i<t?5:1)}}function l(n,r){let{id:a,operation:s,positions:c,neighbors:l,distances:u,zones:d,listLength:f,separationWGSL:p,geographic:m}=r,h=(t,r=d)=>e(n,`${a}-${t}`,`uint32`,r);if(r.nearestZoneSearch===`full-scan`)return[o(n,{...r,sortKeys:h(`sort-keys`,d*f),sortZones:h(`sort-zones`,d*f)})];let g=h(`keys-x`),_=h(`keys-y`),v=h(`rows`),y=h(`sorted-keys-x`),b=h(`sorted-keys-y`),x=h(`sorted-rows-x`),S=h(`sorted-rows-y`),C=`// Float-flip orders finite coordinates numerically; the two zeros share one key.
fn getPositionKey(value: f32) -> u32 {
  var bits = bitcast<u32>(value);
  if (value == 0.0) { bits = 0u; }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn getPositionValue(key: u32) -> f32 {
  return bitcast<f32>(select(key ^ 0x80000000u, ~key, (key & 0x80000000u) == 0u));
}`;return[i(n,{id:`${a}-index-keys`,operation:s,variant:`nearest-index-keys`,invocationCount:d,bindings:[{name:`positions`,view:c,type:`f32`,access:`read`},{name:`keysX`,view:g,type:`u32`,access:`read_write`},{name:`keysY`,view:_,type:`u32`,access:`read_write`},{name:`rows`,view:v,type:`u32`,access:`read_write`}],declarations:C,body:`keysX[keysXOffset + index] = getPositionKey(positions[positionsOffset + index * 2u]);
  keysY[keysYOffset + index] = getPositionKey(positions[positionsOffset + index * 2u + 1u]);
  rows[rowsOffset + index] = index;`}),...new t({id:`${a}-index-x`,keys:g,values:v,outputKeys:y,outputValues:x,algorithm:`radix`}).getCommandNodes(n),...new t({id:`${a}-index-y`,keys:_,values:v,outputKeys:b,outputValues:S,algorithm:`radix`}).getCommandNodes(n),i(n,{id:a,operation:s,variant:`nearest-indexed`,invocationCount:d,bindings:[{name:`positions`,view:c,type:`f32`,access:`read`},{name:`sortedKeysX`,view:y,type:`u32`,access:`read`},{name:`sortedKeysY`,view:b,type:`u32`,access:`read`},{name:`sortedRowsX`,view:x,type:`u32`,access:`read`},{name:`sortedRowsY`,view:S,type:`u32`,access:`read`},{name:`neighbors`,view:l,type:`u32`,access:`read_write`},{name:`distances`,view:u,type:`u32`,access:`read_write`}],declarations:`${p}
${C}
const ZONES: u32 = ${d}u;
const LIST: u32 = ${f}u;
const GEOGRAPHIC: bool = ${m};
fn getSortedKey(rank: u32, axis: u32) -> u32 {
  if (axis == 0u) { return sortedKeysX[sortedKeysXOffset + rank]; }
  return sortedKeysY[sortedKeysYOffset + rank];
}
fn getSortedRow(rank: u32, axis: u32) -> u32 {
  if (axis == 0u) { return sortedRowsX[sortedRowsXOffset + rank]; }
  return sortedRowsY[sortedRowsYOffset + rank];
}
fn getZoneSeparationKey(zone: u32, centerX: f32, centerY: f32) -> u32 {
  let zoneX = positions[positionsOffset + zone * 2u];
  let zoneY = positions[positionsOffset + zone * 2u + 1u];
  return bitcast<u32>(getSeparationKey(vec2f(zoneX, zoneY), vec2f(centerX, centerY)));
}
// A same-longitude haversine drops a nonnegative term, so latitude is a lower bound even
// across the antimeridian and at a pole. Planar projection similarly drops one squared term.
fn getAxisBound(rank: u32, axis: u32, center: vec2f, canPrune: bool) -> u32 {
  if (!canPrune) { return 0u; }
  let coordinate = getPositionValue(getSortedKey(rank, axis));
  var projection = vec2f(coordinate, center.y);
  if (axis == 1u) { projection = vec2f(center.x, coordinate); }
  let bound = getSeparationKey(projection, center);
  // Retain a conservative margin for the GPU's trigonometric rounding near geographic bounds.
  return bitcast<u32>(select(bound, bound * 0.99, GEOGRAPHIC));
}`,body:`let row = index * LIST;
  let centerX = positions[positionsOffset + index * 2u];
  let centerY = positions[positionsOffset + index * 2u + 1u];
  let minimumY = getPositionValue(sortedKeysY[sortedKeysYOffset]);
  let maximumY = getPositionValue(sortedKeysY[sortedKeysYOffset + ZONES - 1u]);
  var axis = 1u;
  if (!GEOGRAPHIC) {
    let minimumX = getPositionValue(sortedKeysX[sortedKeysXOffset]);
    let maximumX = getPositionValue(sortedKeysX[sortedKeysXOffset + ZONES - 1u]);
    axis = select(1u, 0u, maximumX - minimumX >= maximumY - minimumY);
  }
  // Preserve the exhaustive result for out-of-range finite latitudes too: its trigonometric
  // expression need not have a nonnegative longitude term there, so disable pruning.
  let canPrune = !GEOGRAPHIC || (minimumY >= -90.0 && maximumY <= 90.0);
  let center = vec2f(centerX, centerY);
  let centerKey = getPositionKey(select(centerX, centerY, axis == 1u));
  var lower = 0u;
  var upper = ZONES;
  while (lower < upper) {
    let middle = lower + (upper - lower) / 2u;
    if (getSortedKey(middle, axis) < centerKey) { lower = middle + 1u; }
    else { upper = middle; }
  }
  var left = lower;
  var right = lower;
  var filled = 0u;
  var selectedKeys: array<u32, LIST>;
  var selectedRows: array<u32, LIST>;
  loop {
    if (left == 0u && right == ZONES) { break; }
    var leftBound = 0xffffffffu;
    var rightBound = 0xffffffffu;
    if (left > 0u) { leftBound = getAxisBound(left - 1u, axis, center, canPrune); }
    if (right < ZONES) { rightBound = getAxisBound(right, axis, center, canPrune); }
    if (filled == LIST) {
      let worst = selectedKeys[LIST - 1u];
      // Strict comparison retains every separation-key tie, including lower row IDs.
      if (leftBound > worst) { left = 0u; }
      if (rightBound > worst) { right = ZONES; }
    }
    if (left == 0u && right == ZONES) { break; }
    var rank = right;
    if (left > 0u && (right == ZONES || leftBound <= rightBound)) {
      left--;
      rank = left;
    } else { right++; }
    let zone = getSortedRow(rank, axis);
    let key = getZoneSeparationKey(zone, centerX, centerY);
    var destination = min(filled, LIST - 1u);
    if (filled == LIST && (key > selectedKeys[destination] ||
        (key == selectedKeys[destination] && zone >= selectedRows[destination]))) { continue; }
    while (destination > 0u) {
      let previous = destination - 1u;
      if (selectedKeys[previous] < key || (selectedKeys[previous] == key && selectedRows[previous] < zone)) { break; }
      selectedKeys[destination] = selectedKeys[previous];
      selectedRows[destination] = selectedRows[previous];
      destination--;
    }
    selectedKeys[destination] = key;
    selectedRows[destination] = zone;
    filled = min(filled + 1u, LIST);
  }
  for (var slot = 0u; slot < LIST; slot++) {
    neighbors[neighborsOffset + row + slot] = selectedRows[slot];
    distances[distancesOffset + row + slot] = selectedKeys[slot];
  }`})]}export{l as n,c as r,s as t};
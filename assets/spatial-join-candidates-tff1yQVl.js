import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as n,c as r,r as i}from"./spatial-join-passes-CNxNpnDa.js";import{t as a}from"./graph-chunk-groups-BB8ACpQO.js";import{n as o}from"./scan-total-C4QddMmi.js";import{p as s}from"./wgsl-common-DfyXiYB0.js";import{t as c}from"./geodesic-wgsl-I0CYDKcz.js";import{t as l}from"./scaled-quotient-wgsl-UDyhQag6.js";var u=`GPUSpatialJoinCandidates`;function d(e){let t=String(Math.fround(e));return/[.eE]/.test(t)?t:`${t}.0`}function f(f,p){let{id:m,leftCount:h,rightCount:g,candidateCapacity:_,tree:v,state:y,candidatePairs:b}=p;if(h<1)throw Error(`${m} requires at least one left feature`);let x=[],S=p.page?.counts??e(f,`${m}-left-counts`,`uint32`,h),C=p.page?.offsets??e(f,`${m}-left-offsets`,`uint32`,h),w=typeof p.margin==`number`?void 0:p.margin,{sphereRadius:T}=p,E=e=>[{name:`leftMinima`,view:p.leftMinima,type:`f32`,access:`read`},{name:`leftMaxima`,view:p.leftMaxima,type:`f32`,access:`read`},{name:`nodeMinima`,view:v.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:v.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:v.leafIds,type:`u32`,access:`read`},...w?[{name:`marginRow`,view:w,type:`f32`,access:`read`}]:[],...e],D=w?`fn getMargin() -> f32 {
  let margin = marginRow[marginRowOffset];
  return select(0.0, margin, margin >= 0.0 && margin <= 3.4028234e38);
}`:`fn getMargin() -> f32 { return ${d(Math.fround(p.margin))}; }`,O=`${i}
const INTERNAL_NODE_COUNT: u32 = ${v.internalNodeCount}u;
const RIGHT_COUNT: u32 = ${g}u;
const CANDIDATE_CAPACITY: u32 = ${_}u;
${D}
${s}
${T===void 0?``:`${c}
${l}
const SPHERE_RADIUS: f32 = ${d(T)};
const WRAP_SHIFT: vec2f = vec2f(360.0, 0.0);`}`,k=T===void 0?`let queryMinimum = boxMinimum - vec2f(margin);
    let queryMaximum = boxMaximum + vec2f(margin);`:`var queryMinimum = boxMinimum;
    var queryMaximum = boxMaximum;
    var wrapDown = false;
    var wrapUp = false;
    if (margin > 0.0) {
      let angle = getScaledQuotient(margin, SPHERE_RADIUS, 1.0, 0) * 1.001;
      let deltaLatitude = angle * GEODESIC_RADIANS_TO_DEGREES + 1.0e-5;
      queryMinimum.y = boxMinimum.y - deltaLatitude;
      queryMaximum.y = boxMaximum.y + deltaLatitude;
      let farLatitude = max(abs(queryMinimum.y), abs(queryMaximum.y));
      var ratio = 2.0;
      if (farLatitude < 90.0) { ratio = sin(0.5 * angle) / geodesicCosLatitude(farLatitude); }
      if (ratio < 1.0) {
        let deltaLongitude = 2.0 * asin(ratio) * GEODESIC_RADIANS_TO_DEGREES + 1.0e-5;
        queryMinimum.x = boxMinimum.x - deltaLongitude;
        queryMaximum.x = boxMaximum.x + deltaLongitude;
        wrapDown = queryMaximum.x > 180.0;
        wrapUp = queryMinimum.x < -180.0;
      } else {
        // The cap reaches a pole (or is wider than a hemisphere in longitude): every longitude.
        queryMinimum.x = -FLOAT32_MAXIMUM;
        queryMaximum.x = FLOAT32_MAXIMUM;
      }
    }`,A=T===void 0?`nodeOverlaps(node, queryMinimum, queryMaximum)`:`(nodeOverlaps(node, queryMinimum, queryMaximum) ||
        (wrapDown && nodeOverlaps(node, queryMinimum - WRAP_SHIFT, queryMaximum - WRAP_SHIFT)) ||
        (wrapUp && nodeOverlaps(node, queryMinimum + WRAP_SHIFT, queryMaximum + WRAP_SHIFT)))`,j=(e,t,n)=>`let boxMinimum = vec2f(leftMinima[leftMinimaOffset + index * 2u], leftMinima[leftMinimaOffset + index * 2u + 1u]);
  let boxMaximum = vec2f(leftMaxima[leftMaximaOffset + index * 2u], leftMaxima[leftMaximaOffset + index * 2u + 1u]);
  ${t}
  if (boxMinimum.x <= boxMaximum.x && boxMinimum.y <= boxMaximum.y) {
    let margin = getMargin();
    ${k}
    var node = 0u;
    loop {
      if (${A}) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let rightRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (rightRow < RIGHT_COUNT) {
          ${e}
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  ${n}`;(!p.page||p.page.initialize)&&(x.push(t(f,{id:`${m}-count-candidates`,operation:u,variant:`count-candidates`,bindings:E([{name:`leftCounts`,view:S,type:`u32`,access:`read_write`}]),invocationCount:h,declarations:O,body:j(`found = found + 1u;`,`var found = 0u;`,`leftCounts[leftCountsOffset + index] = found;`)})),x.push(...o(f,{id:`${m}-scan-candidates`,operation:u,counts:S,offsets:C,rowCount:h,total:y})));let M=p.page&&v.spatialSort?`
fn countRightsThrough(index: u32, maximumRight: u32) -> u32 {
${j(`if (rightRow <= maximumRight) { found++; }`,`var found = 0u;`,`return found;`)}
}
fn findFirstPageRight(index: u32, rank: u32) -> u32 {
  // The first retained candidate needs no rank selection, including every single-page query.
  if (rank == 0u) { return 0u; }
  var lower = 0u;
  var upper = RIGHT_COUNT - 1u;
  // At most 32 bisections of uint32 source IDs; no candidate-prefix allocation is needed.
  for (var step = 0u; step < 32u; step++) {
    if (lower >= upper) { break; }
    let middle = lower + (upper - lower) / 2u;
    if (countRightsThrough(index, middle) > rank) { upper = middle; }
    else { lower = middle + 1u; }
  }
  return lower;
}`:``,N=v.spatialSort?`
fn readCandidateRight(start: u32, slot: u32) -> u32 {
  return candidatePairs[candidatePairsOffset + (start + slot) * 2u + 1u];
}
fn writeCandidateRight(start: u32, slot: u32, value: u32) {
  candidatePairs[candidatePairsOffset + (start + slot) * 2u + 1u] = value;
}
fn siftCandidateDown(start: u32, size: u32, value: u32) {
  var parent = 0u;
  loop {
    var child = parent * 2u + 1u;
    if (child >= size) { break; }
    if (child + 1u < size && readCandidateRight(start, child + 1u) > readCandidateRight(start, child)) {
      child = child + 1u;
    }
    let childValue = readCandidateRight(start, child);
    if (childValue <= value) { break; }
    writeCandidateRight(start, parent, childValue);
    parent = child;
  }
  writeCandidateRight(start, parent, value);
}`:``,P=j(v.spatialSort?`${p.page?`if (rightRow >= minimumRight) {`:``}
          if (found < slotEnd - slotStart) {
            var child = found;
            found = found + 1u;
            loop {
              if (child == 0u) { break; }
              let parent = (child - 1u) / 2u;
              let parentValue = readCandidateRight(slotStart, parent);
              if (parentValue >= rightRow) { break; }
              writeCandidateRight(slotStart, child, parentValue);
              child = parent;
            }
            writeCandidateRight(slotStart, child, rightRow);
          } else if (rightRow < readCandidateRight(slotStart, 0u)) {
            siftCandidateDown(slotStart, found, rightRow);
          }
          ${p.page?`}`:``}`:`${p.page?`if (found >= skipped) {`:``}
          let slot = slotStart + found${p.page?` - skipped`:``};
          candidatePairs[candidatePairsOffset + slot * 2u] = index;
          candidatePairs[candidatePairsOffset + slot * 2u + 1u] = rightRow;
          if (slot + 1u >= slotEnd) { break; }
          ${p.page?`}`:``}
          found = found + 1u;`,p.page?`var found = 0u;
  let globalStart = leftOffsets[leftOffsetsOffset + index];
  var globalEnd = CANDIDATE_CAPACITY;
  if (index + 1u < ${h}u) { globalEnd = min(leftOffsets[leftOffsetsOffset + index + 1u], CANDIDATE_CAPACITY); }
  let pageFirst = getChunkParameter(36u);
  let pageEnd = pageFirst + getChunkParameter(37u);
  if (globalStart >= pageEnd || globalEnd <= pageFirst) { return; }
  let skipped = max(globalStart, pageFirst) - globalStart;
  let slotStart = max(globalStart, pageFirst) - pageFirst;
  let slotEnd = min(globalEnd, pageEnd) - pageFirst;
  ${v.spatialSort?`let minimumRight = findFirstPageRight(index, skipped);`:``}`:`var found = 0u;
  let slotStart = leftOffsets[leftOffsetsOffset + index];
  var slotEnd = CANDIDATE_CAPACITY;
  if (index + 1u < ${h}u) {
    slotEnd = min(leftOffsets[leftOffsetsOffset + index + 1u], CANDIDATE_CAPACITY);
  }
  if (slotStart >= slotEnd) { return; }`,v.spatialSort?`var remaining = found;
  loop {
    if (remaining <= 1u) { break; }
    remaining = remaining - 1u;
    let largest = readCandidateRight(slotStart, 0u);
    let replacement = readCandidateRight(slotStart, remaining);
    writeCandidateRight(slotStart, remaining, largest);
    siftCandidateDown(slotStart, remaining, replacement);
  }
  for (var slot = 0u; slot < found; slot = slot + 1u) {
    candidatePairs[candidatePairsOffset + (slotStart + slot) * 2u] = index;
  }`:``),F=p.page?.metadataRow===void 0?void 0:n(f,{id:m,offsets:C,total:y,firstSlot:p.page.firstSlot,length:p.page.length,candidateCapacity:_,queryFirst:0,queryLength:h,queryCount:h,metadataRow:p.page.metadataRow});return F&&x.push(F.updateNode),x.push((p.page?a:t)(f,{...p.page?{parameterValues:[p.page.firstSlot,p.page.length,p.page.metadataRow??0]}:{},...F?{condition:F.condition}:{},id:`${m}-write-candidates`,operation:u,variant:`write-candidates`,bindings:E([{name:`leftOffsets`,view:C,type:`u32`,access:`read`},{name:`candidatePairs`,view:b,type:`u32`,access:`read_write`}]),invocationCount:h,declarations:O+N+M+(F?`
fn writeCandidateQuery(index: u32) { ${P} }`:``),body:F?`let metadata = leftOffsetsOffset + getChunkParameter(38u);
  if (index >= leftOffsets[metadata + 1u]) { return; }
  writeCandidateQuery(leftOffsets[metadata] + index);`:P})),(!p.page||p.page.initialize)&&x.push(...r(f,{id:m,operation:u,counts:S,offsets:C,rowCount:h,candidateCapacity:_,total:y,candidateTruncated:p.candidateTruncated,candidateOverflow:p.candidateOverflow,bvhOverflow:p.bvhOverflow})),x}export{f as t};
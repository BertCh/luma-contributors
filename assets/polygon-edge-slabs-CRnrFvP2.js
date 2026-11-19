import{t as e}from"./common-CshkX-d_.js";import{u as t}from"./gpu-contributor-utils-BIwLQON5.js";import{r as n}from"./scan-total-C4QddMmi.js";import{r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-scan-B4NNTfF9.js";import{c as a}from"./wgsl-common-DfyXiYB0.js";import{o,t as s}from"./wgsl-kernel-nodes-DagJklH2.js";var c=4096,l=`
const EDGE_SLAB_NO_VERTEX: u32 = 0xffffffffu;
const EDGE_SLAB_PARAMS_STRIDE: u32 = 6u;
const EDGE_SLAB_COUNT_WORD: u32 = 0u;
const EDGE_SLAB_BASE_WORD: u32 = 1u;
const EDGE_SLAB_ORIGIN_WORD: u32 = 2u;
const EDGE_SLAB_SCALE_WORD: u32 = 3u;
const EDGE_SLAB_VERTEX_BEGIN_WORD: u32 = 4u;
const EDGE_SLAB_VERTEX_END_WORD: u32 = 5u;

// Monotone in y for a positive scale (subtract, multiply, floor, clamp). Do not rewrite it as an
// fma or in any other form: edges are listed by this exact expression.
fn edgeSlabOf(y: f32, slabCount: u32, origin: f32, scale: f32) -> u32 {
  if (slabCount <= 1u) { return 0u; }
  return min(u32(clamp(floor((y - origin) * scale), 0.0, f32(slabCount - 1u))), slabCount - 1u);
}
`;function u(e){return[{name:`edgeSlabParams`,view:e.params,type:`u32`,access:`read`},{name:`edgeSlabOffsets`,view:e.slabOffsets,type:`u32`,access:`read`},{name:`edgeSlabEntries`,view:e.slabEntries,type:`u32`,access:`read`}]}function d(e,t){if(!t)return e;let n={edgeSlabParams:t.params.array,edgeSlabParamsOffset:t.params.offset,edgeSlabOffsets:t.offsets.array,edgeSlabOffsetsOffset:t.offsets.offset,edgeSlabEntries:t.entries.array,edgeSlabEntriesOffset:t.entries.offset};return e.replace(/\bedgeSlab(?:Params|Offsets|Entries)(?:Offset)?\b/g,e=>n[e])}function f(e,t){return d(`${l}
const EDGE_SLAB_ENTRY_CAPACITY: u32 = ${e.entryCapacity}u;
const EDGE_SLAB_VERTEX_COUNT: u32 = ${e.vertexCount}u;

fn edgeSlabIsSlabbed(owner: u32) -> bool {
  let ownerBase = edgeSlabParamsOffset + owner * EDGE_SLAB_PARAMS_STRIDE;
  let slabCount = edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD];
  return slabCount != 0u &&
    edgeSlabOffsets[edgeSlabOffsetsOffset + edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD] + slabCount] <= EDGE_SLAB_ENTRY_CAPACITY;
}

fn edgeSlabOfHeight(owner: u32, y: f32) -> u32 {
  let ownerBase = edgeSlabParamsOffset + owner * EDGE_SLAB_PARAMS_STRIDE;
  return edgeSlabOf(
    y,
    edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD],
    bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_ORIGIN_WORD]),
    bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_SCALE_WORD])
  );
}
`,t)}function p(e){let[t,n]=e.indexNames??[`edgeStart`,`edgeEnd`],[r,i]=e.pointNames??[`a`,`b`],a=e.getVertex?`let ${r} = ${e.getVertex}(${t});
      let ${i} = ${e.getVertex}(${n});`:``;return d(`{
    let edgeSlabOwner = ${e.owner};
    let edgeSlabOwnerBase = edgeSlabParamsOffset + edgeSlabOwner * EDGE_SLAB_PARAMS_STRIDE;
    let edgeSlabIsWalked = !edgeSlabIsSlabbed(edgeSlabOwner);
    var edgeSlabFirst = 0u;
    var edgeSlabLast = 0u;
    if (!edgeSlabIsWalked) {
      edgeSlabFirst = edgeSlabOfHeight(edgeSlabOwner, min(${e.lowY}, ${e.highY}));
      edgeSlabLast = edgeSlabOfHeight(edgeSlabOwner, max(${e.lowY}, ${e.highY}));
    }
    let edgeSlabBase = edgeSlabParams[edgeSlabOwnerBase + EDGE_SLAB_BASE_WORD];
    for (var edgeSlabIndex = edgeSlabFirst; edgeSlabIndex <= edgeSlabLast; edgeSlabIndex++) {
      var edgeSlabBegin = edgeSlabParams[edgeSlabOwnerBase + EDGE_SLAB_VERTEX_BEGIN_WORD];
      var edgeSlabEnd = edgeSlabParams[edgeSlabOwnerBase + EDGE_SLAB_VERTEX_END_WORD];
      var edgeSlabShift = 0u;
      if (!edgeSlabIsWalked) {
        edgeSlabBegin = edgeSlabOffsets[edgeSlabOffsetsOffset + edgeSlabBase + edgeSlabIndex];
        edgeSlabEnd = edgeSlabOffsets[edgeSlabOffsetsOffset + edgeSlabBase + edgeSlabIndex + 1u];
        edgeSlabShift = EDGE_SLAB_VERTEX_COUNT;
      }
      for (var edgeSlabEntry = edgeSlabBegin; edgeSlabEntry < edgeSlabEnd; edgeSlabEntry++) {
        let ${t} = edgeSlabEntries[edgeSlabEntriesOffset + 2u * (edgeSlabShift + edgeSlabEntry)];
        if (${t} == EDGE_SLAB_NO_VERTEX) { continue; }
        let ${n} = edgeSlabEntries[edgeSlabEntriesOffset + 2u * (edgeSlabShift + edgeSlabEntry) + 1u];
        ${a}
        ${e.dedupe?`if (!edgeSlabIsWalked && edgeSlabIndex != max(edgeSlabFirst, edgeSlabOfHeight(edgeSlabOwner, ${e.dedupe.edgeLowY}))) { continue; }\n        `:``}${e.body}
      }
    }
  }`,e.storage)}function m(e){return e===`feature`?`let firstPolygon = featureOffsets[featureOffsetsOffset + index];
  let endPolygon = featureOffsets[featureOffsetsOffset + index + 1u];`:`let firstPolygon = index;
  let endPolygon = index + 1u;`}function h(){return`
fn edgeSlabGetVertex(vertex: u32) -> vec2f {
  let vertexBase = positionsOffset + 2u * vertex;
  return vec2f(positions[vertexBase], positions[vertexBase + 1u]);
}
`}function g(e){if(e.owner===`feature`&&!e.featureOffsets)throw Error(`${e.operation} polygon edge slabs with owner 'feature' need featureOffsets`);if(e.polygonOffsets.length<1||e.ringOffsets.length<1)throw Error(`${e.operation} polygon edge slabs need terminal polygon and ring offsets`);if(e.entryCapacity!==void 0&&!(e.entryCapacity>=1))throw Error(`${e.operation} polygon edge slab entryCapacity must be at least 1`)}function _(u,d){g(d);let{id:f,operation:p,owner:_,positions:v,polygonOffsets:y,ringOffsets:b}=d,x=d.featureOffsets,S=v.length,C=(_===`feature`?x.length:y.length)-1,w=Math.floor(S/4)+1,T=e(t(u.device,8,S),0,d.entryCapacity??Math.max(4*S,1024)),E=d.minimumVertexCount??8,D=b.length-1,O=y.length-1,k=r(u,`${f}-slab-params`,`uint32`,Math.max(C,1)*6),A=r(u,`${f}-slab-numbers`,`uint32`,C+1),j=r(u,`${f}-slab-starts`,`uint32`,C+1),M=r(u,`${f}-ring-owner`,`uint32`,Math.max(D,1)),N=r(u,`${f}-vertex-owner`,`uint32`,S),P=r(u,`${f}-slab-counters`,`uint32`,w+1),F=r(u,`${f}-slab-offsets`,`uint32`,w+1),I=r(u,`${f}-slab-entries`,`uint32x2`,S+T),L={owner:_,ownerCount:C,vertexCount:S,entryCapacity:T,slabCapacity:w,params:k,slabOffsets:F,slabEntries:I},R=[{name:`positions`,view:v,type:`f32`,access:`read`},..._===`feature`?[{name:`featureOffsets`,view:x,type:`u32`,access:`read`}]:[],{name:`polygonOffsets`,view:y,type:`u32`,access:`read`},{name:`ringOffsets`,view:b,type:`u32`,access:`read`}],z=o(u,{id:`${f}-slab-params`,operation:p,variant:`slab-params`,bindings:[...R,{name:`edgeSlabParams`,view:k,type:`u32`,access:`read_write`},{name:`slabNumbers`,view:A,type:`u32`,access:`read_write`},...d.gate?[{name:`gateCounts`,view:d.gate.counts,type:`u32`,access:`read`}]:[]],invocationCount:C+1,declarations:`${a}${l}${h()}
const OWNER_COUNT: u32 = ${C}u;
const MINIMUM_VERTEX_COUNT: u32 = ${E}u;
const MAXIMUM_SLABS: u32 = ${c}u;
const ENTRY_BUDGET_PER_VERTEX: f32 = ${Math.max(0,T/Math.max(S,1)).toFixed(8)};
fn edgeSlabIsFinite(p: vec2f) -> bool {
  return isFiniteBits(p.x) && isFiniteBits(p.y) && abs(p.x) < 1e30 && abs(p.y) < 1e30;
}`,body:`if (index == OWNER_COUNT) { slabNumbers[slabNumbersOffset + index] = 0u; return; }
  ${m(_)}
  var vertexBegin = 0u;
  var vertexEnd = 0u;
  var vertexCount = 0u;
  var isFirstRing = true;
  var hasNonFinite = false;
  var verticalTravel = 0.0;
  var minY = 1e30;
  var maxY = -1e30;
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      if (isFirstRing) {
        vertexBegin = firstVertex;
        isFirstRing = false;
      }
      vertexEnd = endVertex;
      if (endVertex > firstVertex) {
        vertexCount += endVertex - firstVertex;
      }
      var previousY = 0.0;
      if (endVertex > firstVertex) { previousY = edgeSlabGetVertex(endVertex - 1u).y; }
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let p = edgeSlabGetVertex(vertex);
        if (edgeSlabIsFinite(p)) {
          minY = min(minY, p.y);
          maxY = max(maxY, p.y);
          verticalTravel += abs(p.y - previousY);
        } else {
          hasNonFinite = true;
        }
        previousY = p.y;
      }
    }
  }
  ${d.gate?`let isGated = gateCounts[gateCountsOffset + index] >= ${d.gate.minimum}u;`:`let isGated = true;`}
  var slabCount = 0u;
  var slabScale = 0.0;
  if (isGated && !hasNonFinite && vertexCount >= MINIMUM_VERTEX_COUNT) {
    // An edge contributes at most 2 + abs(dy) * scale entries. Reserve two entries per
    // vertex, and use half the remaining budget for travel (rounding/accumulation headroom).
    // Vertex count alone chose 4096 slabs even for jagged rings, overflowing 4N entries and
    // turning the refine into a full indirect walk. All statistics here are GPU-resident.
    let travelBudget = max(0.0, ENTRY_BUDGET_PER_VERTEX - 2.0) * 0.5 * f32(vertexCount);
    var wanted = min(vertexCount / 4u, MAXIMUM_SLABS);
    if (verticalTravel > 0.0) {
      wanted = min(wanted, max(1u, u32(floor(travelBudget * ((maxY - minY) / verticalTravel)))));
    }
    let scale = f32(wanted) / (maxY - minY);
    if (scale > 0.0 && scale < 1e30) {
      slabCount = wanted;
      slabScale = scale;
    }
  }
  let ownerBase = edgeSlabParamsOffset + index * EDGE_SLAB_PARAMS_STRIDE;
  edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] = slabCount;
  edgeSlabParams[ownerBase + EDGE_SLAB_ORIGIN_WORD] = bitcast<u32>(minY);
  edgeSlabParams[ownerBase + EDGE_SLAB_SCALE_WORD] = bitcast<u32>(slabScale);
  edgeSlabParams[ownerBase + EDGE_SLAB_VERTEX_BEGIN_WORD] = vertexBegin;
  edgeSlabParams[ownerBase + EDGE_SLAB_VERTEX_END_WORD] = vertexEnd;
  slabNumbers[slabNumbersOffset + index] = slabCount;`}),B=o(u,{id:`${f}-slab-bases`,operation:p,variant:`slab-bases`,bindings:[{name:`edgeSlabParams`,view:k,type:`u32`,access:`read_write`},{name:`slabStarts`,view:j,type:`u32`,access:`read`}],invocationCount:Math.max(C,1),declarations:`${l}
const OWNER_COUNT: u32 = ${C}u;
const SLAB_CAPACITY: u32 = ${w}u;`,body:`if (index >= OWNER_COUNT) { return; }
  let ownerBase = edgeSlabParamsOffset + index * EDGE_SLAB_PARAMS_STRIDE;
  let base = slabStarts[slabStartsOffset + index];
  let slabCount = edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD];
  if (slabCount != 0u && base + slabCount > SLAB_CAPACITY) {
    edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] = 0u;
  }
  edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD] = base;`}),V=o(u,{id:`${f}-ring-owner`,operation:p,variant:`ring-owner`,bindings:[..._===`feature`?[{name:`featureOffsets`,view:x,type:`u32`,access:`read`}]:[],{name:`polygonOffsets`,view:y,type:`u32`,access:`read`},{name:`ringOwner`,view:M,type:`u32`,access:`read_write`}],invocationCount:Math.max(D,1),declarations:`const OWNER_COUNT: u32 = ${C}u;
const POLYGON_COUNT: u32 = ${O}u;
const RING_COUNT: u32 = ${D}u;
const NO_OWNER: u32 = 0xffffffffu;`,body:`if (index >= RING_COUNT) { return; }
  var polygon = 0u;
  var high = POLYGON_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= index) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  var ringOwnerValue = NO_OWNER;
  if (POLYGON_COUNT > 0u && index >= polygonOffsets[polygonOffsetsOffset + polygon] &&
      index < polygonOffsets[polygonOffsetsOffset + polygon + 1u]) {
    ${_===`feature`?`var low = 0u;
    high = OWNER_COUNT;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (featureOffsets[featureOffsetsOffset + middle] <= polygon) {
        low = middle;
      } else {
        high = middle;
      }
    }
    if (OWNER_COUNT > 0u && polygon >= featureOffsets[featureOffsetsOffset + low] &&
        polygon < featureOffsets[featureOffsetsOffset + low + 1u]) {
      ringOwnerValue = low;
    }`:`ringOwnerValue = polygon;`}
  }
  ringOwner[ringOwnerOffset + index] = ringOwnerValue;`}),H=e=>s(u,{id:`${f}-slab-clear-${e}`,operation:p,view:P,type:`u32`,value:`0u`}),U=`${l}${h()}
const RING_COUNT: u32 = ${D}u;
const ENTRY_CAPACITY: u32 = ${T}u;
const VERTEX_COUNT: u32 = ${S}u;

/** Slab range [first, last] of the edge a-b within the owner's lattice. */
fn edgeSlabGetEdgeRange(ownerBase: u32, a: vec2f, b: vec2f) -> vec2u {
  let slabCount = edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD];
  let origin = bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_ORIGIN_WORD]);
  let scale = bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_SCALE_WORD]);
  return vec2u(
    edgeSlabOf(min(a.y, b.y), slabCount, origin, scale),
    edgeSlabOf(max(a.y, b.y), slabCount, origin, scale)
  );
}`,W=o(u,{id:`${f}-slab-edges-count`,operation:p,variant:`slab-edges-count`,bindings:[{name:`positions`,view:v,type:`f32`,access:`read`},{name:`edgeSlabParams`,view:k,type:`u32`,access:`read`},{name:`ringOffsets`,view:b,type:`u32`,access:`read`},{name:`ringOwner`,view:M,type:`u32`,access:`read`},{name:`vertexOwner`,view:N,type:`u32`,access:`read_write`},{name:`edgeSlabEntries`,view:I,type:`u32`,access:`read_write`},{name:`slabCounters`,view:P,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.max(S,1),declarations:U,body:`if (index >= VERTEX_COUNT) { return; }
  var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  var ownerIndex = EDGE_SLAB_NO_VERTEX;
  var next = EDGE_SLAB_NO_VERTEX;
  if (RING_COUNT > 0u && index >= ringBegin && index < ringEnd) {
    ownerIndex = ringOwner[ringOwnerOffset + ring];
    if (ownerIndex != EDGE_SLAB_NO_VERTEX) {
      next = select(index + 1u, ringBegin, index + 1u == ringEnd);
    }
  }
  edgeSlabEntries[edgeSlabEntriesOffset + 2u * index] = select(EDGE_SLAB_NO_VERTEX, index, next != EDGE_SLAB_NO_VERTEX);
  edgeSlabEntries[edgeSlabEntriesOffset + 2u * index + 1u] = next;
  vertexOwner[vertexOwnerOffset + index] = select(EDGE_SLAB_NO_VERTEX, ownerIndex, next != EDGE_SLAB_NO_VERTEX);
  if (next == EDGE_SLAB_NO_VERTEX) { return; }
  let ownerBase = edgeSlabParamsOffset + ownerIndex * EDGE_SLAB_PARAMS_STRIDE;
  if (edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] == 0u) { return; }
  let edgeRange = edgeSlabGetEdgeRange(ownerBase, edgeSlabGetVertex(index), edgeSlabGetVertex(next));
  let base = edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD];
  for (var slab = edgeRange.x; slab <= edgeRange.y; slab++) {
    atomicAdd(&slabCounters[slabCountersOffset + base + slab], 1u);
  }`}),G=o(u,{id:`${f}-slab-edges-fill`,operation:p,variant:`slab-edges-fill`,bindings:[{name:`positions`,view:v,type:`f32`,access:`read`},{name:`edgeSlabParams`,view:k,type:`u32`,access:`read`},{name:`vertexOwner`,view:N,type:`u32`,access:`read`},{name:`edgeSlabOffsets`,view:F,type:`u32`,access:`read`},{name:`edgeSlabEntries`,view:I,type:`u32`,access:`read_write`},{name:`slabCounters`,view:P,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.max(S,1),declarations:U,body:`if (index >= VERTEX_COUNT) { return; }
  let ownerIndex = vertexOwner[vertexOwnerOffset + index];
  if (ownerIndex == EDGE_SLAB_NO_VERTEX) { return; }
  let ownerBase = edgeSlabParamsOffset + ownerIndex * EDGE_SLAB_PARAMS_STRIDE;
  if (edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] == 0u) { return; }
  let next = edgeSlabEntries[edgeSlabEntriesOffset + 2u * index + 1u];
  let edgeRange = edgeSlabGetEdgeRange(ownerBase, edgeSlabGetVertex(index), edgeSlabGetVertex(next));
  let base = edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD];
  for (var slab = edgeRange.x; slab <= edgeRange.y; slab++) {
    let slabIndex = base + slab;
    let position = edgeSlabOffsets[edgeSlabOffsetsOffset + slabIndex] +
      atomicAdd(&slabCounters[slabCountersOffset + slabIndex], 1u);
    if (position < ENTRY_CAPACITY) {
      edgeSlabEntries[edgeSlabEntriesOffset + 2u * (VERTEX_COUNT + position)] = index;
      edgeSlabEntries[edgeSlabEntriesOffset + 2u * (VERTEX_COUNT + position) + 1u] = next;
    }
  }`}),K=r(u,`${f}-slab-entry-total`,`uint32`,1);return{nodes:[z,...new i({id:`${f}-slab-number-scan`,input:A,output:j,mode:`exclusive`}).getCommandNodes(u),B,V,H(`count`),W,...new i({id:`${f}-slab-offset-scan`,input:P,output:F,mode:`exclusive`}).getCommandNodes(u),...n(u,{id:`${f}-slab-entry`,operation:p,counts:P,offsets:F,rowCount:w,total:K}),o(u,{id:`${f}-slab-entry-fallback`,operation:p,variant:`slab-entry-fallback`,bindings:[{name:`scanTotal`,view:K,type:`u32`,access:`read`},{name:`edgeSlabParams`,view:k,type:`u32`,access:`read_write`}],invocationCount:C,body:`if (scanTotal[scanTotalOffset] == 0xffffffffu) {
    edgeSlabParams[edgeSlabParamsOffset + index * 6u] = 0u;
  }`}),H(`cursor`),G],slabs:L}}export{f as i,p as n,_ as r,u as t};
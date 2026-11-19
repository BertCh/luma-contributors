import{i as e,n as t}from"./geometry-storage-validation-T89ZhmCj.js";import{r as n}from"./kernel-C3tfQOn4.js";import{n as r,t as i}from"./construction-snapshot-DdkX9Pc0.js";import{v as a}from"./spatial-context-DwZEd4EZ.js";import{E as o,T as s,f as c,g as l,p as u,t as d,v as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{c as m,o as h}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as g}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as _}from"./validate-outline-output-passes-DIcqH6Pw.js";import{Q as v}from"./dist-BusamGlx.js";import{i as y}from"./gpu-command-graph-BdzdmV6u.js";import{t as b}from"./gpu-grid-index-BoHdb71C.js";import{t as x}from"./scaled-quotient-wgsl-UDyhQag6.js";import{t as S}from"./bounds-reduction-6MyKJx-7.js";import{i as C}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{t as w}from"./define-run-B-brkCe4.js";import{r as T}from"./pick-Bo3t3hjv.js";import{Qa as E}from"./entry-preview-D70TdSbK.js";import{a as D,u as O}from"./polygon-set-CHja50mr.js";function k(e,t){let n=i(e,t),a=i(n.output,[`positions`,`sourceEdges`,`counts`,`validity`,`issueCode`,`candidateOverflow`,`status`]);return Object.freeze({...n,...n.spatialContext?{spatialContext:r(n.spatialContext)}:{},output:Object.freeze({...a,...a.status?{status:i(a.status,[`candidateOverflow`])}:{}})})}var A=`${x}
fn normalizeCoordinate(value: f32, minimum: f32, extent: f32) -> f32 {
  if (extent < 1.175494351e-38) { return 0.5; }
  // A conservative query expansion can overflow even though all caller coordinates are finite.
  // Never pass infinity into the finite-input exponent-scaled quotient helper.
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) { return select(0.0, 1.0, value > 0.0); }
  return clamp(getScaledQuotient(value * 0.5 - minimum, extent, 1.0, 0), 0.0, 1.0);
}
fn normalizePoint(value: vec2f) -> vec2f {
  let minimum = vec2f(bounds[boundsOffset], bounds[boundsOffset + 1u]) * 0.5;
  let maximum = vec2f(bounds[boundsOffset + 2u], bounds[boundsOffset + 3u]) * 0.5;
  let extent = maximum - minimum;
  return vec2f(normalizeCoordinate(value.x, minimum.x, extent.x), normalizeCoordinate(value.y, minimum.y, extent.y));
}
fn finitePoint(value: vec2f) -> bool { return all((bitcast<vec2u>(value) & vec2u(0x7f800000u)) != vec2u(0x7f800000u)); }`;function j(e,t){let{id:r,candidateCapacity:i,referencePositions:a}=t,o=n(e,`${r}-candidates`,`uint32`,3+t.batchLineCapacity*(i+2),v.STORAGE|v.COPY_DST),s=n(e,`${r}-bounds`,`float32`,4),c=n(e,`${r}-normalized`,`float32x2`,a.length),l=n(e,`${r}-cells`,`uint32`,256**2+1),u=n(e,`${r}-objects`,`uint32`,a.length),f=n(e,`${r}-grid-count`,`uint32`,1),p=n(e,`${r}-grid-overflow`,`uint32`,1),m=[h(e,{id:`${r}-reset-candidates`,operation:`GPUSegmentSnap`,variant:`reset-candidates`,invocationCount:1,bindings:[{name:`candidates`,view:o,type:`u32`,access:`read_write`}],body:`candidates[candidatesOffset] = 0u;`})];return m.push(...S(e,{id:r,operation:`GPUSegmentSnap`,variant:`reference-bounds`,itemCount:a.length,bindings:[{name:`references`,view:a,type:`f32`,access:`read`}],declarations:`fn readItem(row: u32) -> vec4f {
      let value = vec2f(references[referencesOffset + row * 2u], references[referencesOffset + row * 2u + 1u]);
      let finite = all((bitcast<vec2u>(value) & vec2u(0x7f800000u)) != vec2u(0x7f800000u));
      return vec4f(value, select(0.0, 1.0, finite), 0.0);
    }`,output:s})),m.push(h(e,{id:`${r}-normalize`,operation:`GPUSegmentSnap`,variant:`normalize-references`,invocationCount:a.length,bindings:[{name:`references`,view:a,type:`f32`,access:`read`},{name:`bounds`,view:s,type:`f32`,access:`read`},{name:`normalized`,view:c,type:`f32`,access:`read_write`},{name:`candidates`,view:o,type:`atomic<u32>`,access:`read_write`}],declarations:A,body:`let value = vec2f(references[referencesOffset + index * 2u], references[referencesOffset + index * 2u + 1u]);
      if (!finitePoint(value)) { atomicOr(&candidates[candidatesOffset], 1u); }
      let mapped = select(vec2f(-1.0), normalizePoint(value), finitePoint(value));
      normalized[normalizedOffset + index * 2u] = mapped.x;
      normalized[normalizedOffset + index * 2u + 1u] = mapped.y;`})),m.push(...new b({id:`${r}-grid`,positions:c,gridSize:[256,256],bounds:[0,0,1,1],cellOffsets:l,objectIds:u,count:f,overflow:p}).getCommandNodes(e)),{nodes:m,candidates:o,createBatchNodes:(n,c)=>{let f=`${r}-batch-${n}`,p=[y({id:`${f}-parameters`,resources:[{buffer:o,usage:`copy-destination`}],compile:({device:e})=>{let t=new d(e,{id:`${f}-defaults`,format:`uint32`,length:2,values:new Uint32Array([n,c])});return{encode:({commandEncoder:e,getBuffer:n})=>e.copyBufferToBuffer({sourceBuffer:t.buffer,sourceOffset:0,destinationBuffer:n(o),destinationOffset:o.byteOffset+4,size:8}),destroy:()=>t.destroy()}}})];return p.push(h(e,{id:`${f}-gather-candidates`,operation:`GPUSegmentSnap`,variant:`line-grid-candidates`,invocationCount:256,workgroupSize:256,guardIndex:!1,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`bounds`,view:s,type:`f32`,access:`read`},{name:`cellOffsets`,view:l,type:`u32`,access:`read`},{name:`objectIds`,view:u,type:`u32`,access:`read`},{name:`candidates`,view:o,type:`u32`,access:`read_write`}],declarations:`${A}
const CANDIDATE_CAPACITY: u32 = ${i}u;
const GRID_WIDTH: u32 = 256u;
const TOLERANCE: f32 = ${Math.fround(t.tolerance)};`,body:`if (index >= candidates[candidatesOffset + 2u]) { return; }
      let record = candidatesOffset + 3u + index * (CANDIDATE_CAPACITY + 2u);
      let rowCount = (arrayLength(&positions) - positionsOffset) / 2u;
      let first = min(lineOffsets[lineOffsetsOffset + candidates[candidatesOffset + 1u] + index], rowCount);
      let last = min(lineOffsets[lineOffsetsOffset + candidates[candidatesOffset + 1u] + index + 1u], rowCount);
      var minimum = vec2f(1.0); var maximum = vec2f(0.0);
      for (var row = first; row < last; row++) {
        let value = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
        // An invalid line cannot be snapped. Do not report candidate capacity loss for it.
        if (!finitePoint(value)) { candidates[record] = 0u; candidates[record + 1u] = 0u; return; }
        // Endpoints may move by tolerance before insertion, so the conservative radius is twice
        // tolerance. Add in half units to avoid overflowing the literal or multiplication.
        // Cover f32 rounding in snapping and convex interpolation too. Otherwise a reference
        // one ulp outside a very narrow line box can pass the exact kernel's rounded predicate.
        let roundingMargin = max(abs(value), vec2f(TOLERANCE)) * 9.5367431640625e-7 + vec2f(1.175494351e-38);
        let low = normalizePoint(value - vec2f(TOLERANCE) - vec2f(TOLERANCE) - roundingMargin);
        let high = normalizePoint(value + vec2f(TOLERANCE) + vec2f(TOLERANCE) + roundingMargin);
        minimum = min(minimum, low); maximum = max(maximum, high);
        if (!finitePoint(low) || !finitePoint(high)) { minimum = vec2f(0.0); maximum = vec2f(1.0); }
      }
      let lowerCell = vec2u(max(vec2f(0.0), floor(minimum * f32(GRID_WIDTH)) - vec2f(1.0)));
      let upperCell = vec2u(min(vec2f(f32(GRID_WIDTH - 1u)), floor(maximum * f32(GRID_WIDTH)) + vec2f(1.0)));
      var required = 0u;
      if (last > first) {
        for (var vertical = lowerCell.y; vertical <= upperCell.y; vertical++) {
          for (var horizontal = lowerCell.x; horizontal <= upperCell.x; horizontal++) {
            let cell = vertical * GRID_WIDTH + horizontal;
            let start = cellOffsets[cellOffsetsOffset + cell];
            let end = cellOffsets[cellOffsetsOffset + cell + 1u];
            let available = min(end - start, CANDIDATE_CAPACITY - min(required, CANDIDATE_CAPACITY));
            for (var slot = 0u; slot < available; slot++) {
              candidates[record + 2u + required + slot] = objectIds[objectIdsOffset + start + slot];
            }
            required += min(end - start, 0xffffffffu - required);
          }
        }
      }
      candidates[record] = required;
      candidates[record + 1u] = select(0u, 1u, required > CANDIDATE_CAPACITY);`})),p.push(h(e,{id:`${f}-deduplicate-candidates`,operation:`GPUSegmentSnap`,variant:`unique-line-candidates`,invocationCount:256,workgroupSize:256,guardIndex:!1,bindings:[{name:`references`,view:a,type:`f32`,access:`read`},{name:`candidates`,view:o,type:`u32`,access:`read_write`}],declarations:`const CANDIDATE_CAPACITY: u32 = ${i}u;
fn referencePoint(row: u32) -> vec2f { return vec2f(references[referencesOffset + row * 2u], references[referencesOffset + row * 2u + 1u]); }`,body:`if (index >= candidates[candidatesOffset + 2u]) { return; }
let record = candidatesOffset + 3u + index * (CANDIDATE_CAPACITY + 2u);
if (candidates[candidatesOffset] != 0u || candidates[record + 1u] != 0u) { return; }
let required = candidates[record];
var written = 0u;
for (var slot = 0u; slot < required; slot++) {
  let row = candidates[record + 2u + slot];
  let point = referencePoint(row);
  var duplicate = false;
  for (var other = 0u; other < required; other++) {
    let earlierRow = candidates[record + 2u + other];
    if (earlierRow < row && all(referencePoint(earlierRow) == point)) { duplicate = true; break; }
  }
  // Compaction only overwrites an earlier slot. Every coordinate's minimum source row remains
  // in the prefix or the unread suffix, so duplicate decisions retain the original row tie break.
  if (!duplicate) { candidates[record + 2u + written] = row; written++; }
}
candidates[record] = written;`})),p}}}var M=34028234663852886e22,N=4294967295,P=2147483647,F=4294967295,I=32,L=256,R=16777216;function z(e,t,n=1){return(e+n)*(t+1)**3<=R}function B(e,t,n,r,i,a){let o=t>0&&i.searchStructure!==`brute-force`&&n>(i.candidateCapacity??1024)&&(i.candidateOverflowPolicy===`incomplete`||!z(t,n,r));s(e,!o||a,`output must include validity, issueCode or status.candidateOverflow when grid candidate loss can suppress a line`)}var V=class{id;props;spatialContext;estimatedCommandNodeCount;outputViews;constructor(n){n=k(n,[`id`,`spatialContext`,`positions`,`lineOffsets`,`referencePositions`,`tolerance`,`maximumVerticesPerLine`,`searchStructure`,`candidateCapacity`,`candidateOverflowPolicy`,`output`]),this.id=n.id??`segment-snap`,this.props=n,this.outputViews={positions:n.output.positions,sourceEdges:n.output.sourceEdges,counts:n.output.counts,validity:n.output.validity,issueCode:n.output.issueCode,candidateOverflow:t(this.id,`output.status.candidateOverflow`,n.output.status?.candidateOverflow,n.output.candidateOverflow)},this.spatialContext=a(this.id,n.spatialContext),c(this.id,`maximumVerticesPerLine`,n.maximumVerticesPerLine,1,P),this.estimatedCommandNodeCount=n.searchStructure===`brute-force`?3:I+5*(n.lineOffsets.length-1)+1,s(this.id,n.searchStructure===void 0||n.searchStructure===`grid`||n.searchStructure===`brute-force`,`searchStructure must be grid or brute-force`),s(this.id,n.candidateOverflowPolicy===void 0||n.candidateOverflowPolicy===`bounded-brute-force`||n.candidateOverflowPolicy===`incomplete`,`candidateOverflowPolicy must be bounded-brute-force or incomplete`),c(this.id,`candidateCapacity`,n.candidateCapacity??1024,1,4294967290),n.searchStructure===`brute-force`&&c(this.id,`positions.length * (referencePositions.length + 1)`,n.positions.length*(n.referencePositions.length+1),0,N),n.searchStructure===`brute-force`&&s(this.id,z(n.positions.length,n.referencePositions.length,n.lineOffsets.length-1),`(positions.length + lineCount) * (referencePositions.length + 1)^3 must not exceed MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK (${R})`),l(this.id,`tolerance`,n.tolerance,0,M);for(let e of[n.positions,n.referencePositions,n.output.positions])o(e,[`float32x2`],this.id);for(let e of[n.lineOffsets,n.output.sourceEdges])o(e,[`uint32`],this.id);o(n.output.counts,[`uint32x4`],this.id);for(let[e,t]of[[`validity`,n.output.validity],[`issueCode`,n.output.issueCode]])t&&(o(t,[`uint32`],`${this.id} output.${e}`),t.length!==n.lineOffsets.length-1&&s(this.id,!1,`output.${e}.length must equal lineOffsets.length - 1`));c(this.id,`lineOffsets.length`,n.lineOffsets.length,2,1073741824),c(this.id,`positions.length`,n.positions.length,0,P),c(this.id,`referencePositions.length`,n.referencePositions.length,0,P);let r=n.lineOffsets.length-1;c(this.id,`lineCount * maximumVerticesPerLine`,r*n.maximumVerticesPerLine,1,P),(n.output.counts.length!==r||n.output.positions.length!==r*n.maximumVerticesPerLine||n.output.sourceEdges.length!==n.output.positions.length)&&s(this.id,!1,`output lengths must match lineCount and maximumVerticesPerLine`),this.outputViews.candidateOverflow&&(o(this.outputViews.candidateOverflow,[`uint32`],this.id),s(this.id,this.outputViews.candidateOverflow.length===r,`output.status.candidateOverflow.length must equal lineCount`)),B(this.id,n.positions.length,n.referencePositions.length,r,n,!!(this.outputViews.validity||this.outputViews.issueCode||this.outputViews.candidateOverflow)),_(this.id,this.outputViews,[[`positions`,`sourceEdges`,`counts`],[`counts`,`validity`,`issueCode`],[`candidateOverflow`]]);let i=[n.positions,n.lineOffsets,n.referencePositions],{candidateOverflow:u,...d}=this.outputViews;p(this.id,Object.values(d),i),e(this.id,i,[u])}getCommandNodes(e){let{props:t,id:r}=this;g(e,r,{positions:t.positions,lineOffsets:t.lineOffsets,referencePositions:t.referencePositions,...Object.fromEntries(Object.entries(this.outputViews).map(([e,t])=>[`output.${e}`,t]))});let i=t.candidateCapacity??1024,a=t.lineOffsets.length-1,o=t.searchStructure!==`brute-force`&&t.referencePositions.length>0,s=Math.min(F,Math.floor(Math.min(e.device.limits.maxStorageBufferBindingSize,e.device.limits.maxBufferSize)/4)),l=Math.max(1,Math.min(L,a,Math.floor((s-3)/(i+2)))),d=(3+l*(i+2))*4;if(u(r,o?I+5*Math.ceil(a/l)+1:3),o)for(let[n,i]of Object.entries({candidates:d,normalizedReferences:Math.max(1,t.referencePositions.length)*8,gridRowScratch:Math.max(1,t.referencePositions.length)*4,gridCellsAndScan:(256**2+1)*4,boundsPartials:Math.min(256,Math.max(1,Math.ceil(t.referencePositions.length/4096)))*16}))f(e.device,r,n,i,`reduce candidateCapacity or split the input`),c(r,`${n} byteLength`,i,1,e.device.limits.maxBufferSize);let p=this.outputViews.issueCode||this.outputViews.validity?n(e,`${r}-incomplete-lines`,`uint32`,a):void 0,_=o?j(e,{...t,id:r,candidateCapacity:i,batchLineCapacity:l}):void 0,v=_?.candidates,y=_?.nodes??[];for(let n=0;n<a;n+=o?l:a)_&&y.push(..._.createBatchNodes(n,Math.min(l,a-n))),y.push(h(e,{id:`${r}-insert-${n}`,operation:`GPUSegmentSnap`,variant:`sorted-insertions`,invocationCount:o?L:a,workgroupSize:o?L:void 0,guardIndex:!o,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`references`,view:t.referencePositions,type:`f32`,access:`read`},...v?[{name:`candidates`,view:v,type:`u32`,access:`read`}]:[],{name:`outPositions`,view:this.outputViews.positions,type:`f32`,access:`read_write`},{name:`sourceEdges`,view:this.outputViews.sourceEdges,type:`u32`,access:`read_write`},{name:`counts`,view:this.outputViews.counts,type:`u32`,access:`read_write`},...p?[{name:`incompleteLines`,view:p,type:`u32`,access:`read_write`}]:[]],declarations:`const CAPACITY: u32 = ${t.maximumVerticesPerLine}u;
const REFERENCE_COUNT: u32 = ${t.referencePositions.length}u;
const TOLERANCE: f32 = ${m(t.tolerance)};
${C(`point`,`positions`)}
${x}
const EXHAUSTIVE_ALLOWED: bool = ${t.candidateOverflowPolicy!==`incomplete`&&z(t.positions.length,t.referencePositions.length,t.lineOffsets.length-1)};
${v?`const CANDIDATE_CAPACITY: u32 = ${i}u;
fn requiresDuplicateSearch(line: u32) -> bool { return candidates[candidatesOffset + 4u + (line - candidates[candidatesOffset + 1u]) * (CANDIDATE_CAPACITY + 2u)] != 0u; }
fn candidateCount(line: u32) -> u32 { let base = candidatesOffset + 3u + (line - candidates[candidatesOffset + 1u]) * (CANDIDATE_CAPACITY + 2u); return select(candidates[base], REFERENCE_COUNT, candidates[base + 1u] != 0u); }
fn candidateRow(line: u32, slot: u32) -> u32 { let base = candidatesOffset + 3u + (line - candidates[candidatesOffset + 1u]) * (CANDIDATE_CAPACITY + 2u); if (candidates[base + 1u] != 0u) { return slot; } return candidates[base + 2u + slot]; }`:`fn requiresDuplicateSearch(line: u32) -> bool { return true; }
fn candidateCount(line: u32) -> u32 { return REFERENCE_COUNT; }
fn candidateRow(line: u32, slot: u32) -> u32 { return slot; }`}
fn scaledProductQuotient(value: f32, scale: f32, first: f32, second: f32) -> f32 {
  if (value == 0.0) { return value; }
  let scaling = frexp(scale);
  return getScaledQuotient(value * scaling.fract, first, second, scaling.exp);
}
fn reference(vertex: u32) -> vec2f { return vec2f(references[referencesOffset + vertex * 2u], references[referencesOffset + vertex * 2u + 1u]); }
fn finitePoint(value: vec2f) -> bool { return all((bitcast<vec2u>(value) & vec2u(0x7f800000u)) != vec2u(0x7f800000u)); }
fn pointDistance(first: vec2f, second: vec2f) -> f32 {
  let difference = first - second;
  let magnitude = max(abs(difference.x), abs(difference.y));
  if (magnitude == 0.0) { return 0.0; }
  // A nonfinite magnitude is larger than every finite tolerance. Avoid inf / inf below.
  if ((bitcast<u32>(magnitude) & 0x7f800000u) == 0x7f800000u) { return magnitude; }
  return magnitude * length(vec2f(getScaledQuotient(difference.x, magnitude, 1.0, 0), getScaledQuotient(difference.y, magnitude, 1.0, 0)));
}
fn snapPoint(line: u32, value: vec2f) -> vec2f {
  var result = value; var minimum = TOLERANCE; var found = false; var winner = 0xffffffffu;
  for (var slot = 0u; slot < candidateCount(line); slot++) {
    let candidate = candidateRow(line, slot);
    let distance = pointDistance(reference(candidate), value);
    if (distance <= TOLERANCE && (!found || distance < minimum || (distance == minimum && candidate < winner))) { result = reference(candidate); minimum = distance; found = true; winner = candidate; }
  }
  return result;
}
fn writeVertex(line: u32, slot: u32, value: vec2f, edge: u32) {
  if (slot >= CAPACITY) { return; }
  let destination = line * CAPACITY + slot;
  outPositions[outPositionsOffset + destination * 2u] = value.x;
  outPositions[outPositionsOffset + destination * 2u + 1u] = value.y;
  sourceEdges[sourceEdgesOffset + destination] = edge;
}`,body:`${v?`if (index >= candidates[candidatesOffset + 2u]) { return; }`:``}
let line = index ${v?`+ candidates[candidatesOffset + 1u]`:``};
let incomplete = ${v?`!EXHAUSTIVE_ALLOWED && requiresDuplicateSearch(line)`:`false`};
let first = min(lineOffsets[lineOffsetsOffset + line], ${t.positions.length}u);
let last = min(lineOffsets[lineOffsetsOffset + line + 1u], ${t.positions.length}u);
var required = 0u;
var invalid = false;
${v?`invalid = candidates[candidatesOffset] != 0u;`:`for (var candidate = 0u; candidate < REFERENCE_COUNT; candidate++) { invalid = invalid || !finitePoint(reference(candidate)); }`}
for (var vertex = first; vertex < last; vertex++) { invalid = invalid || !finitePoint(point(vertex)); }
if (!invalid && !incomplete && last > first) {
  var start = snapPoint(line, point(first));
  writeVertex(line, required, start, first); required += min(1u, 0xffffffffu - required);
  for (var vertex = first; vertex + 1u < last; vertex++) {
    let end = snapPoint(line, point(vertex + 1u));
    let halfEdge = end * 0.5 - start * 0.5;
    let magnitude = max(abs(halfEdge.x), abs(halfEdge.y));
    let divisor = max(magnitude, 1.175494351e-38);
    let edge = vec2f(getScaledQuotient(halfEdge.x, divisor, 1.0, 0), getScaledQuotient(halfEdge.y, divisor, 1.0, 0));
    let denominator = dot(edge, edge);
    var previousFraction = 0.0; var previousReference = 0xffffffffu;
    var previousPoint = start;
    if (denominator > 0.0) {
      for (var inserted = 0u; inserted < candidateCount(line); inserted++) {
        var winner = 0xffffffffu; var winnerFraction = 2.0;
        for (var slot = 0u; slot < candidateCount(line); slot++) {
          let candidate = candidateRow(line, slot);
          let current = reference(candidate);
          var duplicate = false;
          if (requiresDuplicateSearch(line)) {
            for (var earlierSlot = 0u; earlierSlot < candidateCount(line); earlierSlot++) {
              let earlier = candidateRow(line, earlierSlot);
              if (earlier < candidate && all(reference(earlier) == current)) { duplicate = true; break; }
            }
          }
          if (duplicate) { continue; }
          let halfRelative = current * 0.5 - start * 0.5;
          let relativeScale = max(max(abs(halfRelative.x), abs(halfRelative.y)), 1.175494351e-38);
          let relative = vec2f(getScaledQuotient(halfRelative.x, relativeScale, 1.0, 0), getScaledQuotient(halfRelative.y, relativeScale, 1.0, 0));
          let fraction = scaledProductQuotient(dot(relative, edge), relativeScale, divisor, denominator);
          // Perpendicular distance avoids reconstructing and rounding the projected coordinate.
          // The determinant is exactly zero for translated axis-collinear references at zero tolerance.
          let perpendicular = abs(relative.x * edge.y - relative.y * edge.x);
          let distance = scaledProductQuotient(perpendicular, relativeScale, sqrt(denominator) * 0.5, 1.0);
          if (fraction <= 0.0 || fraction >= 1.0 || distance > TOLERANCE) { continue; }
          if (fraction < previousFraction || (fraction == previousFraction && candidate <= previousReference)) { continue; }
          if (fraction < winnerFraction || (fraction == winnerFraction && candidate < winner)) { winner = candidate; winnerFraction = fraction; }
        }
        if (winner == 0xffffffffu) { break; }
        let current = reference(winner);
        if (any(current != previousPoint)) { writeVertex(line, required, current, vertex); required += min(1u, 0xffffffffu - required); previousPoint = current; }
        previousFraction = winnerFraction; previousReference = winner;
      }
    }
    if (any(end != previousPoint)) { writeVertex(line, required, end, vertex); required += min(1u, 0xffffffffu - required); }
    start = end;
  }
}
counts[countsOffset + line * 4u] = min(required, CAPACITY);
counts[countsOffset + line * 4u + 1u] = required;
counts[countsOffset + line * 4u + 2u] = select(0u, 1u, required > CAPACITY);
counts[countsOffset + line * 4u + 3u] = select(0u, 1u, invalid);
${p?`incompleteLines[incompleteLinesOffset + line] = select(0u, 1u, incomplete);`:``}`})),this.outputViews.candidateOverflow&&y.push(h(e,{id:`${r}-candidate-status-${n}`,operation:`GPUSegmentSnap`,variant:`candidate-status`,invocationCount:o?L:a,workgroupSize:o?L:void 0,guardIndex:!o,bindings:[...v?[{name:`candidates`,view:v,type:`u32`,access:`read`}]:[],{name:`candidateOverflow`,view:this.outputViews.candidateOverflow,type:`u32`,access:`read_write`}],body:v?`if (index >= candidates[candidatesOffset + 2u]) { return; }
candidateOverflow[candidateOverflowOffset + candidates[candidatesOffset + 1u] + index] = candidates[candidatesOffset + 4u + index * ${i+2}u];`:`candidateOverflow[candidateOverflowOffset + index] = 0u;`}));let{issueCode:b,validity:S}=this.outputViews;return(b||S)&&y.push(h(e,{id:`${r}-line-validity`,operation:`GPUSegmentSnap`,variant:`line-validity`,invocationCount:a,bindings:[{name:`counts`,view:this.outputViews.counts,type:`u32`,access:`read_write`},{name:`incompleteLines`,view:p,type:`u32`,access:`read`},...b?[{name:`issueCode`,view:b,type:`u32`,access:`read_write`}]:[],...S?[{name:`validity`,view:S,type:`u32`,access:`read_write`}]:[]],body:`let required = counts[countsOffset + 4u * index + 1u];
        let code = select(0u, 1u, required == 1u)
          | select(0u, 2u, counts[countsOffset + 4u * index + 3u] != 0u)
          | select(0u, 4u, counts[countsOffset + 4u * index + 2u] != 0u || incompleteLines[incompleteLinesOffset + index] != 0u);
        ${b?`issueCode[issueCodeOffset + index] = code;`:``}
        ${S?`validity[validityOffset + index] = select(0u, 1u, code == 0u);`:``}`})),y}},H=400,U=w({parameters:E,caption:`Reference vertices near the line are inserted into it.`,createInputs:({random:e,world:t})=>{let n=O(e,t,14),r=[];for(let t=0;t<40;t++){let t=Math.floor(e()*(n.count-1)),i=e(),a=n.positions[t*2]*(1-i)+n.positions[t*2+2]*i,o=n.positions[t*2+1]*(1-i)+n.positions[t*2+3]*i;r.push(a+(e()-.5)*.2,o+(e()-.5)*.2)}let i=Float32Array.from(r),a=0,o=1/0;for(let e=0;e<r.length/2;e++){let t=T(n.positions,r[e*2],r[e*2+1])?.distance??0,i=Math.abs(t-.8*E[0].default);i<o&&(o=i,a=e)}return{line:n,references:i,carriedIndex:a,carried:[i[a*2],i[a*2+1]]}},build(e,t,n){let{line:r,references:i}=t,a=e.upload(`positions`,`float32x2`,r.positions),o=e.upload(`line-offsets`,`uint32`,r.lineOffsets),s=e.allocate(`references`,`float32x2`,i.length/2),c=e.allocate(`carried`,`float32x2`,1),l=e.allocate(`is-inserted`,`uint32`,i.length/2),u=e.allocate(`snapped`,`float32x2`,H),d=e.allocate(`source-edges`,`uint32`,H),f=e.allocate(`counts`,`uint32x4`,1);return e.graph.add(new V({positions:a.view,lineOffsets:o.view,referencePositions:s.view,tolerance:n.tolerance,maximumVerticesPerLine:H,output:{positions:u.view,sourceEdges:d.view,counts:f.view}})),{outputs:{positions:a,referencePositions:s,isInserted:l,carriedDot:c,snapped:u,counts:f,ids:e.upload(`ids`,`uint32`,D(H))},update:e=>{let n=Float32Array.from(i);n.set(t.carried,t.carriedIndex*2),s.write(n),c.write(Float32Array.from(t.carried)),l.write(Uint32Array.from({length:n.length/2},(t,i)=>+((T(r.positions,n[i*2],n[i*2+1])?.distance??1/0)<=e.tolerance)))}}},interaction:{kind:`drag`,hint:`drag a reference vertex`,rest:({inputs:e,toUnit:t})=>t([e.references[e.carriedIndex*2],e.references[e.carriedIndex*2+1]]),resolve(e,{inputs:t,values:n,peek:r,outputs:i}){t.carried=[e.x,e.y];let a=T(t.line.positions,e.x,e.y),o=(a?.distance??1/0)<=n.tolerance,s=r(i.counts)?.[0];return{rerun:!0,glyphs:[{shape:`ring`,at:t.carried,radius:n.tolerance,dashed:!0},...a?[{shape:`line`,from:t.carried,to:a.point,dashed:!0}]:[],{shape:`crosshair`,at:t.carried}],readout:`tolerance ${n.tolerance.toFixed(3)} · ${o?`inserted`:`left out`}${s===void 0?``:` · ${s} vertices`}`}}},draw:({inputs:e,outputs:t})=>[{layer:`paths`,context:!0,nodes:t.positions,ids:t.ids,count:e.line.count},{layer:`paths`,nodes:t.snapped,ids:t.ids,count:t.counts,radius:2},{layer:`dots`,positions:t.referencePositions,values:t.isInserted,categories:[[.55,.6,.72],[1,.6,.2]],radius:5},{layer:`dots`,positions:t.carriedDot,radius:9,color:[1,.6,.2,1]}]});export{U as default};
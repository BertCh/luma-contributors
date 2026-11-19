import{n as e,r as t}from"./lon-lat-ring-wgsl-DpZBIfuE.js";import{a as n,d as r,i,l as a,n as o,o as s,r as c,s as l,t as u,u as d}from"./spatial-join-geometry-DYRzw652.js";import{c as f,i as p,o as m,t as h}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as g}from"./kernel-C3tfQOn4.js";import{t as _}from"./gpu-scan-B4NNTfF9.js";import{n as v,t as ee}from"./gpu-pairwise-point-in-polygon-DCUVp3i5.js";import{C as y,E as b,T as te,f as x,g as S,i as C,l as w,m as ne,w as T,x as E}from"./gpu-contributor-utils-BIwLQON5.js";import{c as D}from"./spatial-weights-CqWysstN.js";import{_ as O,g as re,n as ie,v as ae,y as k}from"./spatial-context-DwZEd4EZ.js";import{t as oe}from"./spatial-join-candidates-tff1yQVl.js";import{C as se,D as A,T as j,b as M,p as ce,r as N}from"./spatial-join-passes-CNxNpnDa.js";import{a as le,i as ue,n as P,o as de,r as fe,s as F,t as I}from"./spatial-predicate-wgsl-B9jvQ4Mb.js";import{n as pe}from"./ports-xmdKgL07.js";import{Q as me}from"./dist-BusamGlx.js";import{i as he,t as ge}from"./graph-chunk-groups-BB8ACpQO.js";import{t as _e}from"./scan-total-C4QddMmi.js";import{n as ve,t as L}from"./construction-snapshot-DdkX9Pc0.js";import{i as R,n as z,r as B,t as V}from"./spatial-join-construction-snapshot-B4G0PxaB.js";var H=2147483648,U=[`F`,`0`,`1`,`2`];function W(e){if(!/^[F012]{9}$/.test(e))throw Error(`DE-9IM matrix must be nine characters of F, 0, 1 or 2, got "${e}"`);let t=0;for(let n=0;n<9;n++)t|=U.indexOf(e[n])<<2*n;return t}function ye(e){return typeof e==`string`?[e]:e}function be(e){if(!/^[TF012*]{9}$/.test(e))throw Error(`DE-9IM pattern must be nine characters of T, F, 0, 1, 2 or *, got "${e}"`);let t={"*":15,T:14,F:1,0:2};return t[1]=4,t[2]=8,[...e].map(e=>t[e])}function G(e){return e.some(e=>{let t=be(e);return[0,1,3,4].every(e=>!!(t[e]&1))&&!!(t[8]&8)})}function K(e){return e===`points`?0:e===`lines`?1:2}function xe(e,t,n){switch(e){case`intersects`:return[`T********`,`*T*******`,`***T*****`,`****T****`];case`contains`:return[`T*****FF*`];case`within`:return[`T*F**F***`];case`covers`:return[`T*****FF*`,`*T****FF*`,`***T**FF*`,`****T*FF*`];case`coveredBy`:return[`T*F**F***`,`*TF**F***`,`**FT*F***`,`**F*TF***`];case`touches`:return t===0&&n===0?[]:[`FT*******`,`F**T*****`,`F***T****`];case`crosses`:return t<n?[`T*T******`]:t>n?[`T*****T**`]:t===1?[`0********`]:[];case`overlaps`:return t===n?t===1?[`1*T***T**`]:[`T*T***T**`]:[];case`equals`:return[`T*F**FFF*`];case`containsProperly`:return[`T**FF*FF*`]}}function q(e){let t=e.map(e=>{let t=be(e).map((e,t)=>e===15?void 0:`((${e}u >> ((m >> ${2*t}u) & 3u)) & 1u) != 0u`).filter(Boolean);return t.length?`(${t.join(` && `)})`:`true`});return`fn relateMatches(m: u32) -> bool { return ${t.length?t.join(` || `):`false`}; }`}function J(e){return`fn relateMatches(m: u32) -> bool {
  for (var slot = 0u; slot < ${e}u; slot++) {
    let low = patternWords[patternWordsOffset + slot * 2u];
    let high = patternWords[patternWordsOffset + slot * 2u + 1u];
    var matched = true;
    for (var cell = 0u; cell < 9u; cell++) {
      let mask = select((low >> (cell * 4u)) & 15u, high & 15u, cell == 8u);
      if (((mask >> ((m >> (cell * 2u)) & 3u)) & 1u) == 0u) { matched = false; break; }
    }
    if (matched) { return true; }
  }
  return false;
}`}function Se(e,t,n){return`${de(e,t,`dwithin`,n)}
var<private> relateLane: u32 = 0u;
var<private> relateStride: u32 = 1u;
var<workgroup> dwithinMatched: atomic<u32>;
// Squared gap between the bounding boxes of segments a-b and c-d (zero when they overlap).
fn boxGapSq(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  let gap = max(max(min(a, b) - max(c, d), min(c, d) - max(a, b)), vec2f(0.0));
  return dot(gap, gap);
}
fn dwithinWorkgroupPair(l: u32, r: u32, valid: bool, lane: u32) -> bool {
  relateLane = lane;
  relateStride = 64u;
  let limit = distanceLimit();
  let useLinearDistance = limit > PLANAR_MAXIMUM_SQUAREABLE_DISTANCE;
  let limitSq = distanceLimitSq();
  var reachSq = 0.0;
  if (!useLinearDistance) { reachSq = limitSq * 1.001 + 1e-30; }
  if (valid && limit >= 0.0 && !(leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0)) {
    // Bounding box of the right feature.
    var rightLow = vec2f(FLOAT32_MAXIMUM);
    var rightHigh = vec2f(-FLOAT32_MAXIMUM);
    ${I(t,`r`,`c`,`d`,`rightLow = min(rightLow, min(c, d));
    rightHigh = max(rightHigh, max(c, d));`)}
    var matched = false;
    ${I(e,`l`,`a`,`b`,`if (matched) { break; }
    ${t.kind===`polygons`?`if (rightLocate(a, r) != 0u) { matched = true; break; }`:``}
    if (!useLinearDistance && boxGapSq(a, b, rightLow, rightHigh) > reachSq) { continue; }
    ${I(t,`r`,`c`,`d`,`if (!useLinearDistance && boxGapSq(a, b, c, d) > reachSq) { continue; }
      if ((useLinearDistance && segmentDistance(a, b, c, d) <= limit) ||
          (!useLinearDistance && segmentDistanceSq(a, b, c, d) <= limitSq)) { matched = true; break; }`)}`,!0)}
    // Right components can also lie inside the left polygon without a boundary crossing.
    // Visit every component, with the same lane-strided distribution as the distance pass.
    ${e.kind===`polygons`?I(t,`r`,`c`,`d`,`if (matched) { break; }
    if (leftLocate(c, l) != 0u) { matched = true; break; }`,!0):``}
    if (matched) { atomicOr(&dwithinMatched, 1u); }
  }
  workgroupBarrier();
  return atomicLoad(&dwithinMatched) != 0u;
}`}var Y=`0u`,Ce=`1u`,X=`2u`;function we(e,t,n){let r=e.prefix,i=e.kind===`lines`?`let ${r}first = ${r}Vertex(${r}vs);
    let ${r}last = ${r}Vertex(${r}ve - 1u);
    let ${r}closed = ${r}first.x == ${r}last.x && ${r}first.y == ${r}last.y;`:``,a=e.kind===`points`?Y:e.kind===`polygons`?Ce:`select(0u, 1u, !${r}closed && ((v.x == ${r}first.x && v.y == ${r}first.y) || (v.x == ${r}last.x && v.y == ${r}last.y)))`;return`for (var ${r}r = ${r}RingStart(${t}); ${r}r < ${r}RingEnd(${t}); ${r}r++) {
    let ${r}vs = ${r}RingVertexStart(${r}r);
    let ${r}ve = ${r}RingVertexEnd(${r}r);
    if (${r}ve <= ${r}vs || ${r}ve - ${r}vs < ${P(e.kind)}u) { continue; }
    let ${r}n = ${r}ve - ${r}vs;
    ${i}
    for (var ${r}k = relateLane; ${r}k < ${r}n; ${r}k += relateStride) {
      let v = ${r}Vertex(${r}vs + ${r}k);
      let vc = ${a};
      ${n}
    }
  }`}function Te(e){if(e.kind!==`lines`)return`false`;let t=e.prefix,n=e=>`(onSegment(a, b, ${e}) && onSegment(c, d, ${e}))`,r=`${t}Vertex(${t}vs)`,i=`${t}Vertex(${t}ve - 1u)`;return`(!(${r}.x == ${i}.x && ${r}.y == ${i}.y) && (${n(r)} || ${n(i)}))`}function Ee(e,t,n,r,i,a,o,s){let c=I(e,t,n,r,o);if(!s||e.kind!==`polygons`)return c;let l=e.prefix;return`if (${l}SlabActive) {
    for (var ${l}r = ${l}RingStart(${t}); ${l}r < ${l}RingEnd(${t}); ${l}r++) {
      let ${l}vs = ${l}RingVertexStart(${l}r);
      let ${l}ve = ${l}RingVertexEnd(${l}r);
      if (${l}ve <= ${l}vs || ${l}ve - ${l}vs < ${P(e.kind)}u) { continue; }
      let ${n} = ${l}Vertex(${l}ve - 1u);
      let ${r} = ${l}Vertex(${l}vs);
      ${o}
    }
    let ${l}QueryFirst = ${l}SlabOf(${i});
    let ${l}QueryLast = ${l}SlabOf(${a});
    for (var ${l}Cursor = ${l}QueryFirst; ${l}Cursor <= ${l}QueryLast; ${l}Cursor++) {
      for (var ${l}Entry = ${l}SlabStarts[${l}Cursor]; ${l}Entry < ${l}SlabStarts[${l}Cursor + 1u]; ${l}Entry++) {
        let ${l}EntryEdge = ${l}SlabEntries[${l}Entry];
        let ${n} = ${l}Vertex(${l}EntryEdge);
        let ${r} = ${l}Vertex(${l}EntryEdge + 1u);
        if (max(${l}QueryFirst, ${l}SlabOf(min(${n}.y, ${r}.y))) != ${l}Cursor) { continue; }
        ${o}
      }
    }
  } else {
    ${c}
  }`}function De(e,t){let n=e.prefix,r=`fn ${n}RelateBox(f: u32) -> vec4f {
  var box = vec4f(FLOAT32_MAXIMUM, FLOAT32_MAXIMUM, -FLOAT32_MAXIMUM, -FLOAT32_MAXIMUM);
  ${I(e,`f`,`a`,`b`,`box = vec4f(min(box.xy, min(a, b)), max(box.zw, max(a, b)));`)}
  return box;
}
// Locate, short-cut for points outside the bounding box of the feature (the box is closed).
fn ${n}RelateLocate(q: vec2f, f: u32) -> u32 {
  if (q.x < ${n}Box.x || q.x > ${n}Box.z || q.y < ${n}Box.y || q.y > ${n}Box.w) { return 0u; }
  ${t&&e.kind===`polygons`?`if (${n}SlabActive) { return ${n}SlabLocate(q, f); }`:``}
  return ${n}Locate(q, f);
}
// False when the segment a-b is outside the bounding box of the feature: nothing there can touch it.
fn ${n}RelateNearEdge(a: vec2f, b: vec2f) -> bool {
  return !(max(a.x, b.x) < ${n}Box.x || min(a.x, b.x) > ${n}Box.z ||
    max(a.y, b.y) < ${n}Box.y || min(a.y, b.y) > ${n}Box.w);
}`;if(e.kind===`points`)return`${r}
fn ${n}RelateNextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 { return 1.0; }
fn ${n}RelateCollectBreaks(a: vec2f, b: vec2f, f: u32) -> u32 { return 0u; }
fn ${n}RelateBreakAfter(count: u32, t: f32, cursor: ptr<function, u32>) -> f32 { return 1.0; }
fn ${n}RelatePieceCode(a: vec2f, b: vec2f, f: u32, t0: f32, t1: f32) -> u32 { return 0u; }`;let i=e.kind===`polygons`?`return select(0u, 2u, ${n}RelatePieceInside(a, b, f, tm));`:`return 0u;`;return`${r}
${e.kind===`polygons`?fe(`${n}RelatePieceInside`,`${n}RelateLocate`,(n,r,i)=>Ee(e,`f`,`c`,`d`,n,r,i,t)):``}
// Next parameter after t in (0, 1] where segment a-b crosses or starts to overlap an edge.
fn ${n}RelateNextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 {
  var best = 1.0;
  if (!${n}RelateNearEdge(a, b)) { return best; }
  ${I(e,`f`,`c`,`d`,`if (dot(d - c, d - c) == 0.0 || boxesDisjoint(a, b, c, d)) { continue; }
    let breaks = edgeBreaks(a, b, c, d);
    if (breaks.x >= 1.0 && breaks.y > t && breaks.y < best) { best = breaks.y; }
    if (breaks.x >= 2.0 && breaks.z > t && breaks.z < best) { best = breaks.z; }`)}
  return best;
}
// Every break parameter of a-b against the edges of the feature, ascending, in one pass over those
// edges: the piece loop then reads them in order instead of rescanning the edges for each piece.
// Returns the break count, or RELATE_BREAK_OVERFLOW when more than RELATE_BREAK_CAPACITY exist (the
// caller then falls back to \`RelateNextBreak\`, which finds each break by its own scan).
var<private> ${n}Breaks: array<f32, RELATE_BREAK_CAPACITY>;
// Whether some edge of the feature may be collinear with a-b, so \`RelateCovers\` has to run. The
// collection scan answers it once per edge instead of once per piece.
var<private> ${n}CoverNeeded: bool = false;
fn ${n}RelateCollectBreaks(a: vec2f, b: vec2f, f: u32) -> u32 {
  var count = 0u;
  ${n}CoverNeeded = false;
  if (!${n}RelateNearEdge(a, b)) { return 0u; }
  ${Ee(e,`f`,`c`,`d`,`min(a.y, b.y)`,`max(a.y, b.y)`,`if (boxesDisjoint(a, b, c, d)) { continue; }
    if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) { ${n}CoverNeeded = true; }
    if (dot(d - c, d - c) == 0.0) { continue; }
    let breaks = edgeBreaks(a, b, c, d);
    let found = u32(breaks.x);
    for (var index = 0u; index < found; index++) {
      if (count == RELATE_BREAK_CAPACITY) { ${n}CoverNeeded = true; return RELATE_BREAK_OVERFLOW; }
      var position = count;
      let value = select(breaks.y, breaks.z, index == 1u);
      loop {
        if (position == 0u || ${n}Breaks[position - 1u] <= value) { break; }
        ${n}Breaks[position] = ${n}Breaks[position - 1u];
        position = position - 1u;
      }
      ${n}Breaks[position] = value;
      count = count + 1u;
    }`,t)}
  return count;
}
// First collected break after t, or 1.0. \`cursor\` only moves forward as t increases.
fn ${n}RelateBreakAfter(count: u32, t: f32, cursor: ptr<function, u32>) -> f32 {
  while (*cursor < count && ${n}Breaks[*cursor] <= t) { *cursor = *cursor + 1u; }
  return select(1.0, ${n}Breaks[*cursor], *cursor < count);
}
fn ${n}RelateCovers(a: vec2f, b: vec2f, f: u32, tm: f32) -> bool {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (!${n}RelateNearEdge(a, b)) { return false; }
  ${I(e,`f`,`c`,`d`,`if (boxesDisjoint(a, b, c, d)) { continue; }
    if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) {
      let tc = dot(c - a, ab) / lengthSq;
      let td = dot(d - a, ab) / lengthSq;
      if (min(tc, td) <= tm && tm <= max(tc, td)) { return true; }
    }`)}
  return false;
}
// 0: the piece [t0, t1] of a-b lies outside, 1: on the boundary or line, 2: strictly inside.
fn ${n}RelatePieceCode(a: vec2f, b: vec2f, f: u32, t0: f32, t1: f32) -> u32 {
  let tm = 0.5 * (t0 + t1);
  if (${n}CoverNeeded && ${n}RelateCovers(a, b, f, tm)) { return 1u; }
  ${i}
}`}function Oe(e,t,n,r){let i=n?`l`:`r`,a=n?`r`:`l`,o=(e,t,r)=>n?`raise(${e}, ${t}, ${r}u);`:`raise(${t}, ${e}, ${r}u);`,s=e.kind===`polygons`?Ce:Y,c=t.kind===`polygons`?Ce:Y,l=[];if(l.push(we(e,i,o(`vc`,`locationClass(${t.prefix}RelateLocate(v, ${a}))`,0))),e.kind!==`points`&&t.kind!==`points`&&l.push(I(e,i,`a`,`b`,`if (!${t.prefix}RelateNearEdge(a, b)) { continue; }
        ${Ee(t,a,`c`,`d`,`min(a.y, b.y)`,`max(a.y, b.y)`,`if (boxesDisjoint(a, b, c, d)) { continue; }
      let o1 = orient(a, b, c);
      let o2 = orient(a, b, d);
      let o3 = orient(c, d, a);
      let o4 = orient(c, d, b);
      if (((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0)) &&
          ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0)) &&
          !(${Te(e)} || ${Te(t)})) {
        ${o(s,c,0)}
      }
      if (o1 == 0.0 && o2 == 0.0) {
        let ab = b - a;
        let sc = dot(c - a, ab);
        let sd = dot(d - a, ab);
        let low = max(min(sc, sd), 0.0);
        let high = min(max(sc, sd), dot(ab, ab));
        if (high > low) { ${o(s,c,1)} }
      }`,r)}`,!0)),e.kind!==`points`){let r=e.kind===`polygons`&&t.kind===`polygons`?`if (code == 2u) {
        ${o(Y,Y,2)}
        ${o(X,Y,2)}
      }
      if (code == 0u) { ${o(Y,X,2)} }
      ${n?`if (code == 1u) {`:`if (false) {`}
        // Shared boundary piece: probe both sides along the normal, closer than any other edge.
        // Every shared segment lies on an edge of the left polygon, so the left pass sees all of them.
        let middle = a + ab * (0.5 * (t + next));
        let normal = vec2f(-ab.y, ab.x) / length(ab);
        let reach = relateProbeReach(middle, a, b, l, r);
        for (var side = 0u; side < 2u; side++) {
          let probe = middle + normal * select(reach, -reach, side == 1u);
          let inX = ${e.prefix}RelateLocate(probe, ${i}) == 1u;
          let inY = ${t.prefix}RelateLocate(probe, ${a}) == 1u;
          if (inX && inY) { ${o(Y,Y,2)} }
          else if (inX) { ${o(Y,X,2)} }
          else if (inY) { ${o(X,Y,2)} }
        }
      }`:``;l.push(I(e,i,`a`,`b`,`let ab = b - a;
    if (dot(ab, ab) == 0.0) { continue; }
    var t = 0.0;
    let breakCount = ${t.prefix}RelateCollectBreaks(a, b, ${a});
    var breakCursor = 0u;
    for (var guard = 0u; guard < ${t.prefix.toUpperCase()}_VERTEX_COUNT * 2u + 4u; guard++) {
      if (t >= 1.0) { break; }
      var next = 1.0;
      if (breakCount == RELATE_BREAK_OVERFLOW) {
        next = ${t.prefix}RelateNextBreak(a, b, ${a}, t);
      } else {
        next = ${t.prefix}RelateBreakAfter(breakCount, t, &breakCursor);
      }
      let code = ${t.prefix}RelatePieceCode(a, b, ${a}, t, next);
      if (code == 0u) { ${o(s,X,1)} }
      if (code == 2u) { ${o(s,Y,1)} }
      ${r}
      t = next;
    }`,!0))}return e.kind===`polygons`&&t.kind!==`polygons`&&l.push(o(Y,X,2)),l.join(`
  `)}var Z=32,ke=1280,Ae=48;function je(e,t){let n=e.prefix;return`for (var ${n}r = ${n}RingStart(f); ${n}r < ${n}RingEnd(f); ${n}r++) {
    let ${n}vs = ${n}RingVertexStart(${n}r);
    let ${n}ve = ${n}RingVertexEnd(${n}r);
    if (${n}ve <= ${n}vs || ${n}ve - ${n}vs < ${P(e.kind)}u) { continue; }
    for (var ${n}k = relateLane; ${n}k + 1u < ${n}ve - ${n}vs; ${n}k += relateStride) {
      let a = ${n}Vertex(${n}vs + ${n}k);
      let b = ${n}Vertex(${n}vs + ${n}k + 1u);
      let edge = ${n}vs + ${n}k;
      ${t}
    }
  }`}function Me(e){let t=e.prefix;return`
var<workgroup> ${t}SlabCounts: array<atomic<u32>, 33>;
var<workgroup> ${t}SlabFill: array<atomic<u32>, ${Z}>;
var<workgroup> ${t}SlabStarts: array<u32, 33>;
var<workgroup> ${t}SlabEntries: array<u32, ${ke}>;
var<workgroup> ${t}SlabFlag: u32;
var<private> ${t}SlabActive: bool = false;
var<private> ${t}SlabLow: f32 = 0.0;
var<private> ${t}SlabScale: f32 = 0.0;
fn ${t}SlabOf(y: f32) -> u32 {
  let t = (y - ${t}SlabLow) * ${t}SlabScale;
  return select(0u, u32(min(t, 31.0)), t > 0.0);
}
fn ${t}BuildSlabs(f: u32, enabled: bool) {
  ${t}SlabActive = false;
  if (relateLane <= ${Z}u) { atomicStore(&${t}SlabCounts[relateLane], 0u); }
  if (relateLane < ${Z}u) { atomicStore(&${t}SlabFill[relateLane], 0u); }
  workgroupBarrier();
  var usable = false;
  if (enabled) {
    let box = ${t}RelateBox(f);
    let range = box.w - box.y;
    var edges = 0u;
    for (var r = ${t}RingStart(f); r < ${t}RingEnd(f); r++) {
      let vs = ${t}RingVertexStart(r);
      let ve = ${t}RingVertexEnd(r);
      if (ve > vs && ve - vs >= ${P(e.kind)}u) { edges += ve - vs; }
    }
    ${t}SlabLow = box.y;
    ${t}SlabScale = ${Z}.0 / range;
    usable = edges >= ${Ae}u && box.y >= -FLOAT32_MAXIMUM && box.w <= FLOAT32_MAXIMUM &&
      range > 0.0 && ${t}SlabScale <= FLOAT32_MAXIMUM;
  }
  if (usable) {
    ${je(e,`let first = ${t}SlabOf(min(a.y, b.y));
      let last = ${t}SlabOf(max(a.y, b.y));
      for (var slab = first; slab <= last; slab++) { atomicAdd(&${t}SlabCounts[slab], 1u); }`)}
  }
  workgroupBarrier();
  if (relateLane == 0u) {
    var running = 0u;
    for (var slab = 0u; slab < ${Z}u; slab++) {
      ${t}SlabStarts[slab] = running;
      running += atomicLoad(&${t}SlabCounts[slab]);
    }
    ${t}SlabStarts[${Z}] = running;
    ${t}SlabFlag = select(0u, 1u, usable && running <= ${ke}u);
  }
  workgroupBarrier();
  ${t}SlabActive = ${t}SlabFlag != 0u;
  if (${t}SlabActive) {
    ${je(e,`let first = ${t}SlabOf(min(a.y, b.y));
      let last = ${t}SlabOf(max(a.y, b.y));
      for (var slab = first; slab <= last; slab++) {
        let position = atomicAdd(&${t}SlabFill[slab], 1u);
        ${t}SlabEntries[${t}SlabStarts[slab] + position] = edge;
      }`)}
  }
  workgroupBarrier();
}
// Same answer as ${t}Locate for a polygon: 2 on the boundary, 1 inside, 0 outside.
fn ${t}SlabLocate(q: vec2f, f: u32) -> u32 {
  var inside = false;
  for (var r = ${t}RingStart(f); r < ${t}RingEnd(f); r++) {
    let vs = ${t}RingVertexStart(r);
    let ve = ${t}RingVertexEnd(r);
    if (ve <= vs || ve - vs < ${P(e.kind)}u) { continue; }
    let a = ${t}Vertex(ve - 1u);
    let b = ${t}Vertex(vs);
    if (onSegment(a, b, q)) { return 2u; }
    if ((a.y > q.y) != (b.y > q.y)) {
      let o = orient(a, b, q);
      if ((b.y > a.y && o > 0.0) || (b.y < a.y && o < 0.0)) { inside = !inside; }
    }
  }
  let slab = ${t}SlabOf(q.y);
  for (var i = ${t}SlabStarts[slab]; i < ${t}SlabStarts[slab + 1u]; i++) {
    let edge = ${t}SlabEntries[i];
    let a = ${t}Vertex(edge);
    let b = ${t}Vertex(edge + 1u);
    if (onSegment(a, b, q)) { return 2u; }
    if ((a.y > q.y) != (b.y > q.y)) {
      let o = orient(a, b, q);
      if ((b.y > a.y && o > 0.0) || (b.y < a.y && o < 0.0)) { inside = !inside; }
    }
  }
  return select(0u, 1u, inside);
}`}function Ne(e,t,n=!1){let r=e.kind===`polygons`&&t.kind===`polygons`?`
// Squared distance from q to the bounding box of segment c-d, a lower bound of its distance.
fn boxDistanceSq(q: vec2f, c: vec2f, d: vec2f) -> f32 {
  let gap = max(max(min(c, d) - q, q - max(c, d)), vec2f(0.0));
  return dot(gap, gap);
}
// Distance to probe from a shared piece: half the distance to the nearest non-collinear edge.
fn relateProbeReach(middle: vec2f, a: vec2f, b: vec2f, l: u32, r: u32) -> f32 {
  // The result is capped at length(b - a), so edges farther than twice that cannot change it.
  var best = 4.0 * dot(b - a, b - a);
  ${I(e,`l`,`c`,`d`,`if (boxDistanceSq(middle, c, d) >= best) { continue; }
    if (!(orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0)) { best = min(best, pointSegmentDistanceSq(middle, c, d)); }`)}
  ${I(t,`r`,`c`,`d`,`if (boxDistanceSq(middle, c, d) >= best) { continue; }
    if (!(orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0)) { best = min(best, pointSegmentDistanceSq(middle, c, d)); }`)}
  return min(0.5 * sqrt(best), length(b - a));
}`:``;return`${le(!0)}
const RELATE_BREAK_CAPACITY: u32 = 24u;
const RELATE_BREAK_OVERFLOW: u32 = 0xffffffffu;
${ue(e)}
${ue(t)}
${De(e,n)}
${De(t,n)}
var<private> relateMatrix: u32 = 0u;
// Raises cell (a, b) of the matrix to at least dimension \`dimension\`.
fn raise(a: u32, b: u32, dimension: u32) {
  let shift = (a * 3u + b) * 2u;
  let current = (relateMatrix >> shift) & 3u;
  if (dimension + 1u > current) {
    relateMatrix = (relateMatrix & ~(3u << shift)) | ((dimension + 1u) << shift);
  }
}
// Maps an OGC point location (0 exterior, 1 interior, 2 boundary) to a matrix part (I=0, B=1, E=2).
fn locationClass(location: u32) -> u32 {
  return select(select(1u, 0u, location == 1u), 2u, location == 0u);
}${r}
${n?[e,t].filter(e=>e.kind===`polygons`).map(Me).join(`
`):``}
var<private> relateLane: u32 = 0u;
var<private> relateStride: u32 = 1u;
var<private> leftBox: vec4f;
var<private> rightBox: vec4f;
// One lane of the matrix of (l, r): this lane handles every relateStride-th edge of the outer loops.
fn pairRelateLane(l: u32, r: u32) -> u32 {
  relateMatrix = 0u;
  uncertainOrientation = false;
  if (leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0) { return 0u; }
  leftBox = leftRelateBox(l);
  rightBox = rightRelateBox(r);
  raise(${X}, ${X}, 2u);
  ${Oe(e,t,!0,n)}
  ${Oe(t,e,!1,n)}
  return relateMatrix | select(0u, ${H}u, uncertainOrientation);
}
// The whole matrix of (l, r) in one invocation.
fn pairRelate(l: u32, r: u32) -> u32 {
  relateLane = 0u;
  relateStride = 1u;
  return pairRelateLane(l, r);
}`}function Pe(e=[]){let t=e.filter(e=>e.kind===`polygons`).map(e=>`${e.prefix}BuildSlabs(${e.prefix===`left`?`l`:`r`}, usable);`).join(`
  `);return`
// Lane matrices merge with atomic ORs: each cell holds a thermometer code of its value (a value v
// sets the low v bits of a 3-bit field), so the OR of the lanes is the cell-wise maximum.
var<workgroup> relateMergedCells: atomic<u32>;
var<workgroup> relateMergedUncertain: atomic<u32>;
fn relateWorkgroupPair(l: u32, r: u32, valid: bool, lane: u32) -> u32 {
  relateLane = lane;
  relateStride = 256u;
  ${t?`// Per-pair y-slab indexes of the polygon sides (barriers: reached by every lane).
  let usable = valid && leftFirst(l).z != 0.0 && rightFirst(r).z != 0.0;
  ${t}`:``}
  if (valid) {
    let lanes = pairRelateLane(l, r);
    var thermometer = 0u;
    for (var cell = 0u; cell < 9u; cell++) {
      thermometer = thermometer | (((1u << ((lanes >> (cell * 2u)) & 3u)) - 1u) << (cell * 3u));
    }
    if (thermometer != 0u) { atomicOr(&relateMergedCells, thermometer); }
    if ((lanes & ${H}u) != 0u) { atomicOr(&relateMergedUncertain, 1u); }
  }
  workgroupBarrier();
  var merged = 0u;
  if (lane == 0u) {
    let cells = atomicLoad(&relateMergedCells);
    for (var cell = 0u; cell < 9u; cell++) {
      merged = merged | (countOneBits((cells >> (cell * 3u)) & 7u) << (cell * 2u));
    }
    merged = merged | select(0u, ${H}u, atomicLoad(&relateMergedUncertain) != 0u);
  }
  return merged;
}`}var Fe=16,Ie=4,Le=8,Re=16,ze=8,Be=24,Ve=24,He=20,Ue=4,We=256,Ge=8,Ke=12,qe=20,Je=8;function Ye(e){let t=et(e.leftCount,`leftCount`),n=et(e.rightCount,`rightCount`),r=e.leftKind??`lines`,i=e.rightKind??`lines`,a=$e(t,e.expectedCandidatesPerLeft??Math.min(n,Je),e.observedCandidateCount,`observedCandidateCount`),o=$e(t,e.expectedMatchesPerLeft??Math.min(n,1),e.observedRequiredCount,`observedRequiredCount`),s=r===`points`&&i===`polygons`||r===`polygons`&&i===`points`,c=Fe+(e.relate?Ie:0)+(s?32:0),l=Qe(a,e.candidateCapacity,0,c),u=l.capacity;if(e.candidateTileCapacity!==void 0){let t=et(e.candidateTileCapacity,`candidateTileCapacity`);if(t===0)throw Error(`candidateTileCapacity must be a positive safe integer`);u=Math.min(l.capacity,t),l.estimatedPeakTransientBytes=Q(u*c)}let d=Math.max(1,Math.ceil(l.capacity/Math.max(u,1))),f=Qe(o,e.pairCapacity,Le,0),p=e.leafCapacity??w(Math.max(n,1)),m=(r===`polygons`?t:0)*ze+(i===`polygons`?n:0)*ze,h=s?qe:0,g=(t+n)*Re+t*Ve+m+M(p)+Be+He+Ue+d*Ke*(1+Number(!!e.relate)+Number(s))+2*Xe(t)+d*Xe(u)+h;return{candidates:l,pairs:f,estimatedWorkItems:Q(t+n+a),estimatedOutputBytes:f.estimatedOutputBytes,estimatedPeakTransientBytes:Q(g+l.estimatedPeakTransientBytes)}}function Xe(e){let t=0;for(let n=e;n>We;)n=Math.ceil(n/We),t+=n*Ge;return t}function Ze(e,t){return Math.max(1,Ye({leftCount:e,rightCount:t}).candidates.capacity)}function Qe(e,t,n,r){let i=t===void 0?e:et(t,`capacity`);return{capacity:i,estimatedRequiredCount:e,limited:i<e,estimatedOutputBytes:Q(i*n),estimatedPeakTransientBytes:Q(i*r)}}function $e(e,t,n,r){if(n!==void 0)return et(n,r);if(!Number.isFinite(t)||t<0)throw Error(`expansionFactor must be a non-negative finite number`);return Q(Math.ceil(e*t))}function et(e,t){if(!Number.isSafeInteger(e)||e<0)throw Error(`${t} must be a non-negative safe integer`);return e}function Q(e){return Math.min(2**53-1,e)}var $=`GPUSpatialPredicateJoin`,tt=2147483646,nt=34028234663852886e22,rt=(262143|H)>>>0,it=8,at=Object.freeze({coordinateSpaces:[`planar`,`longitude-latitude`],edgeModels:ie,metrics:[`none`,`native`,`great-circle`]}),ot=Object.freeze({...at,measure:`distance`}),st=[`intersects`,`contains`,`within`,`dwithin`],ct=[`covers`,`coveredBy`,`touches`,`crosses`,`overlaps`,`equals`,`containsProperly`,`relate`],lt=1024;function ut(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function dt(e){return e.positions.length/Math.max(c(e),1)}function ft(e,t){return e.kind===`points`||t.kind===`points`?!1:dt(e)*dt(t)>=lt}var pt=class{id;props;outputWeights;views;scalarOutputs;leftCount;rightCount;candidateCapacity;candidatePages;estimatedCommandNodeCount;pairCapacity;leafCapacity;how;usesRelateEngine;usesWorkgroupDistance;spatialContext;sphereRadius;relatePatterns;patternView;distanceView;constructor(e){let t=L(e,[`id`,`left`,`right`,`predicate`,`engine`,`spatialContext`,`rightSpatialContext`,`distance`,`pattern`,`how`,`excludeSameRow`,`onAttribute`,`candidateCapacity`,`candidateTileCapacity`,`leafCapacity`,`prepared`,`output`]),i=L(t.output,[`candidateTruncated`,`unmatched`,`pairs`,`relate`,`weights`,`status`]),o=i.unmatched&&L(i.unmatched,[`ids`,`status`]),u=i.weights&&R(i.weights);e=Object.freeze({...t,left:V(t.left),right:V(t.right),...t.spatialContext?{spatialContext:ve(t.spatialContext)}:{},...t.rightSpatialContext?{rightSpatialContext:ve(t.rightSpatialContext)}:{},...t.onAttribute?{onAttribute:L(t.onAttribute,[`left`,`right`])}:{},...Array.isArray(t.pattern)?{pattern:Object.freeze([...t.pattern])}:{},output:Object.freeze({...i,...i.pairs?{pairs:z(i.pairs)}:{},...o?{unmatched:Object.freeze({...o,status:B(o.status)})}:{},...u?{weights:u}:{},...i.status?{status:B(i.status)}:{}})}),this.id=e.id??`spatial-predicate-join`,this.props=e;let{output:d,...f}=e,p={...f,...d,...d.status};this.views=p;let{id:m}=this,h=n(p.left)||n(p.right),_=[`intersects`,`within`,`coveredBy`].includes(p.predicate)&&p.relate===void 0&&p.engine!==`relate`;h&&te(m,_,`featureOffsets multipart joins support intersects, within and coveredBy with engine auto/fast and no relate output; the requested predicate or route is unsupported`),l(m,`left`,p.left,{allowMultipart:_}),l(m,`right`,p.right,{allowMultipart:_}),this.leftCount=c(p.left),this.rightCount=c(p.right);let{predicate:v}=p,ee=st.includes(v)||h&&v===`coveredBy`;if(this.distanceView=v===`dwithin`&&ut(p.distance)?p.distance:void 0,this.patternView=ut(p.pattern)?p.pattern:void 0,v===`dwithin`){if(this.distanceView){if(b(this.distanceView,[`float32`],`${m} distance`),this.distanceView.length<1)throw Error(`${m} distance must contain one float32 row`)}else te(m,typeof p.distance==`number`&&p.distance>=0,`distance must be a non-negative number`),S(m,`distance`,p.distance,0,nt)}else if(!ee&&!ct.includes(v))throw Error(`${m} unknown predicate ${String(v)}`);let C=ae(m,p.spatialContext,v===`dwithin`||p.weights?.distances?ot:at);if(this.spatialContext=C,O(m,`right`,p.rightSpatialContext,this.spatialContext,at.edgeModels),C.coordinateSpace===`longitude-latitude`){if(v===`dwithin`&&C.metric!==`great-circle`)throw Error(`${m} longitude-latitude dwithin needs metric 'great-circle'`);if(v!==`dwithin`&&C.metric!==`none`)throw Error(`${m} ${v} needs metric 'none' on longitude-latitude`)}if(v===`dwithin`&&typeof p.distance==`number`&&k(`GPUSpatialPredicateJoin`,this.spatialContext,p.distance),C.metric===`great-circle`){if(C.units===`kilometers`)throw Error(`${m} great-circle dwithin distances are in meters`);if(this.sphereRadius=re(C),y(m,`spatialContext.sphereRadius`,this.sphereRadius),typeof p.distance==`number`&&p.distance/this.sphereRadius>1.2566)throw Error(`${m} distance exceeds ${Math.floor(F*this.sphereRadius)} m`)}else this.sphereRadius=void 0;if(this.how=p.how??`inner`,this.how!==`inner`&&this.how!==`anti`)throw Error(`${m} unknown how ${String(this.how)}`);if(v===`relate`){if(p.pattern===void 0)throw Error(`${m} predicate 'relate' requires a pattern`)}else if(p.pattern!==void 0)throw Error(`${m} pattern requires predicate 'relate'`);if(p.relate&&v===`dwithin`)throw Error(`${m} relate output is not available for dwithin`);let E=p.engine??`auto`;if(E!==`auto`&&E!==`fast`&&E!==`relate`)throw Error(`${m} unknown engine ${String(E)}`);if(E===`fast`&&!ee)throw Error(`${m} engine 'fast' needs intersects, contains, within or dwithin`);if(this.sphereRadius!==void 0&&E===`relate`)throw Error(`${m} engine 'relate' is not available for great-circle dwithin`);let ie=!h&&(E===`relate`||E===`auto`&&ft(p.left,p.right));if(this.usesRelateEngine=!ee||p.relate!==void 0||v!==`dwithin`&&ie,this.usesWorkgroupDistance=v===`dwithin`&&ie&&this.sphereRadius===void 0,this.patternView){let e=this.patternView;if(b(e,[`uint32`],`${m} pattern`),e.length<2||e.length%2!=0)throw Error(`${m} pattern view must hold a positive multiple of 2 words`);this.relatePatterns=[]}else if(this.usesRelateEngine){let e=v===`relate`?[...ye(p.pattern)]:[...xe(v,K(p.left.kind),K(p.right.kind))];if(v===`relate`&&e.length===0)throw Error(`${m} pattern must not be empty`);if(G(e))throw Error(`${m} pattern admits disjoint pairs; use how: 'anti'`);this.relatePatterns=e}else this.relatePatterns=[];this.candidateCapacity=p.candidateCapacity??Ze(this.leftCount,this.rightCount),x(m,`candidateCapacity`,this.candidateCapacity,1,tt);let oe=v!==`dwithin`&&(p.left.kind===`points`&&p.right.kind===`polygons`||p.left.kind===`polygons`&&p.right.kind===`points`),M=this.usesWorkgroupDistance?64:this.usesRelateEngine&&!oe?256:1;if(this.leafCapacity=p.prepared?.leafCapacity??p.leafCapacity??w(Math.max(this.rightCount,1)),T(m,`leafCapacity`,this.leafCapacity),p.prepared&&!s(p.prepared.geometry,p.right))throw Error(`${m} prepared must index the same geometry views as right`);if(this.how===`anti`){if(!p.unmatched)throw Error(`${m} how 'anti' needs output.unmatched`);if(p.pairs||p.weights||p.relate)throw Error(`${m} how 'anti' writes only output.unmatched`);ne(m,p.unmatched),this.pairCapacity=0}else{if(p.unmatched)throw Error(`${m} output.unmatched needs how 'anti'`);if(!p.pairs&&!p.weights)throw Error(`${m} needs output.pairs or output.weights`);if(!p.pairs&&!p.overflow)throw Error(`${m} needs output.status.overflow without pairs`);if(p.pairs){let{pairs:e}=p;pe(`${m} output.pairs`,e);for(let t of[e.leftRows,e.rightRows,e.status.count,e.status.overflow,e.status.candidateOverflow,e.status.requiredCount])t&&b(t,[`uint32`],`${m} output.pairs`);if(e.leftRows.length<1)throw Error(`${m} output.pairs needs at least one slot`)}if(p.weights){if(D(m,p.weights)!==this.leftCount)throw Error(`${m} output.weights needs one row per left feature`);if(te(m,!p.weights.distances||C.metric!==`none`,`output.weights.distances requires spatialContext.metric 'native' or 'great-circle'`),p.pairs&&p.pairs.leftRows.length!==p.weights.neighbors.length)throw Error(`${m} output.weights capacity must equal pairs`)}if(this.pairCapacity=p.pairs?p.pairs.leftRows.length:p.weights.neighbors.length,p.relate){if(!p.pairs)throw Error(`${m} output.relate needs output.pairs`);if(b(p.relate,[`uint32`],`${m} output.relate`),p.relate.length!==this.pairCapacity)throw Error(`${m} output.relate needs one row per pair`)}}if(a(m,e.output.status??{}),p.candidateTruncated&&(b(p.candidateTruncated,[`uint32`],`${m} output.candidateTruncated`),p.candidateTruncated.length!==this.leftCount))throw Error(`${m} output.candidateTruncated needs one row per left feature`);if(p.onAttribute){let{left:e,right:t}=p.onAttribute;if(b(e,[`uint32`],`${m} onAttribute.left`),b(t,[`uint32`],`${m} onAttribute.right`),e.length!==this.leftCount||t.length!==this.rightCount)throw Error(`${m} onAttribute needs one key per feature`)}x(m,`left offset scratch words`,this.leftCount+5,1,j),this.candidatePages=A(p.left.positions.buffer.graph.device,m,this.candidateCapacity,e.candidateTileCapacity,oe?16:8,oe||this.distanceView?8:0,1,M),x(m,`candidate tile invocations`,this.candidatePages.capacity*M),this.estimatedCommandNodeCount=this.candidatePages.estimatedCommandNodeCount;let ce=this.getScalarOutputs().filter(([e,t])=>t!==void 0||!(!p.weights||e!==`weightsOverflow`&&e!==`weightsCandidateOverflow`)),N=[];for(let e=0;e<ce.length;e+=6)N.push(Object.fromEntries(ce.slice(e,e+6)));if(se(m,this.getInputViews(),this.getOutputViews(),[{leftRows:p.pairs?.leftRows,rightRows:p.pairs?.rightRows,relate:p.relate},{rightRows:p.weights?p.pairs?.rightRows:void 0,neighbors:p.weights?.neighbors,weights:p.weights?.weights},...N,...this.spatialContext.coordinateSpace===`longitude-latitude`?[{antimeridian:p.antimeridianUncertainCount,pole:p.poleUncertainCount}]:[]]),p.weights&&r(m,[p.pairs?.rightRows],[p.weights.neighbors,p.weights.weights]),p.prepared&&r(m,[(p.prepared.isDeclaredIn(p.left.positions.buffer.graph)?p.prepared.storage:p.prepared.props.storage)?.overflow],ce.map(([,e])=>e)),p.weights){let e=p.left.positions.buffer.graph,t=p.weights,n={...t.status?B(t.status):{},overflow:t.status?.overflow??g(e,`${m}-weights-overflow`,`uint32`,1),candidateOverflow:t.status?.candidateOverflow??g(e,`${m}-weights-candidate-overflow`,`uint32`,1)};this.outputWeights=Object.freeze({...t,status:Object.freeze(n)}),p.weights=this.outputWeights}let le=new Map(this.getScalarOutputs());this.scalarOutputs=ce.map(([e])=>[e,le.get(e)])}getScalarOutputs(){let{pairs:e,weights:t,unmatched:n}=this.views;return[[`overflow`,this.views.overflow],[`weightsCount`,t?.status?.count],[`weightsTotal`,t?.status?.requiredCount],[`weightsOverflow`,t?.status?.overflow],[`weightsCandidateOverflow`,t?.status?.candidateOverflow],[`weightsUncertainCount`,t?.status?.uncertainCount],[`pairsOverflow`,e?.status.overflow],[`pairsCandidateOverflow`,e?.status.candidateOverflow],[`pairsCount`,e?.status.count],[`pairsTotal`,e?.status.requiredCount],[`unmatchedOverflow`,n?.status.overflow],[`unmatchedCandidateOverflow`,n?.status.candidateOverflow],[`unmatchedCount`,n?.status.count],[`unmatchedTotal`,n?.status.requiredCount],[`candidateCount`,this.views.candidateCount],[`uncertainCount`,this.views.uncertainCount],[`candidateTruncatedCount`,this.views.candidateTruncatedCount]]}getInputViews(){return[...i(this.views.left),...i(this.views.right),...Object.values(this.views.prepared?.isDeclaredIn(this.views.left.positions.buffer.graph)?this.views.prepared.storage:this.views.prepared?.props.storage??{}),...this.distanceView?[this.distanceView]:[],...this.patternView?[this.patternView]:[],...this.views.onAttribute?[this.views.onAttribute.left,this.views.onAttribute.right]:[]]}getOutputViews(){let{pairs:e,weights:t,overflow:n,candidateCount:r,uncertainCount:i,unmatched:a,relate:o,candidateOverflow:s,candidateTruncatedCount:c,candidateTruncated:l}=this.views;return[e?.leftRows,e?.rightRows,e?.status.count,e?.status.overflow,e?.status.candidateOverflow,e?.status.requiredCount,o,t?.offsets,t?.neighbors,t?.weights,t?.distances,...Object.values(t?.status??{}),a?.ids,a?.status.count,a?.status.overflow,a?.status.candidateOverflow,a?.status.requiredCount,n,r,i,s,c,l,...this.spatialContext.coordinateSpace===`longitude-latitude`?[this.views.antimeridianUncertainCount,this.views.poleUncertainCount]:[]]}getCapacityPlan(e={}){let{views:t}=this,n=Ye({leftCount:this.leftCount,rightCount:this.rightCount,leftKind:t.left.kind,rightKind:t.right.kind,relate:this.usesRelateEngine,leafCapacity:this.leafCapacity,candidateTileCapacity:this.candidatePages.capacity,observedCandidateCount:e.candidateCount,observedRequiredCount:e.requiredCount});return t.weights?.distances&&(n.pairs.estimatedPeakTransientBytes+=n.pairs.capacity*2*Uint32Array.BYTES_PER_ELEMENT,n.estimatedPeakTransientBytes+=n.pairs.estimatedPeakTransientBytes),n}getCommandNodes(r){let{id:i,views:a,leftCount:s,rightCount:c,candidateCapacity:l,pairCapacity:y,how:b}=this,{left:x,right:S,predicate:w,pairs:ne,weights:T,prepared:D,unmatched:O,relate:re}=a;if(E(i,r,[...this.getInputViews(),...this.getOutputViews()]),D&&!D.isDeclaredIn(r))throw Error(`${i} requires prepared to be added to the graph first`);let ie=[...this.getInputViews(),...this.getOutputViews(),...Object.values(D?.storage??{})],ae=!n(x)&&!n(S)&&w!==`dwithin`&&(x.kind===`points`&&S.kind===`polygons`||x.kind===`polygons`&&S.kind===`points`),k=this.candidatePages.capacity,se={state:24,"candidate dispatch":12,"candidate pairs":(k+ +!!this.distanceView)*8,"candidate flags/offsets/matrices":(k+1)*4,"candidate probe counts/offsets/scan":(s+5)*4,"left feature bounds/rings":s*8,"right feature bounds/rings":D?0:c*8,"BVH node bounds/children":D?0:(2*this.leafCapacity-1)*8,"BVH leaf IDs":D?0:this.leafCapacity*4,"matched pair IDs":y*4,"distance pairs":T?.distances?y*8:0,"left match counts":T||b===`anti`?(s+1)*4:0,"anti flags/offsets":b===`anti`?s*4:0,"candidate truncated":b===`anti`||a.candidateTruncatedCount?s*4:0,"robust pair points":ae?(2*k+1)*8:0,"robust geometry offsets":ae?(2*k+2)*4:0};d(r,i,Object.fromEntries(ie.map((e,t)=>[`binding[${t}]`,e])),se),he(r,i,8);let A=[];if(this.spatialContext.coordinateSpace===`longitude-latitude`&&t(a)){for(let e of[`antimeridianUncertainCount`,`poleUncertainCount`]){let t=a[e];t&&A.push(h(r,{id:`${i}-clear-${e}`,operation:$,view:t,type:`u32`,value:`0u`,componentCount:1}))}for(let[t,n]of[x,S].filter(e=>e.kind===`polygons`).entries())A.push(e(r,{id:`${i}-lon-lat-polygon-check-${t}`,operation:$,positions:n.positions,ringOffsets:n.ringOffsets,status:{antimeridianUncertainCount:a.antimeridianUncertainCount,poleUncertainCount:a.poleUncertainCount}}))}let j=b===`anti`,M=[{name:`left`,geometry:x,count:s,isPrepared:!1},{name:`right`,geometry:S,count:c,isPrepared:!!D}].map(e=>{let t,n,a;return e.isPrepared?(t=D?.storage.featureRings,{...e,minima:n,maxima:a,featureRings:t}):(n=g(r,`${i}-${e.name}-minima`,`float32x2`,e.count),a=g(r,`${i}-${e.name}-maxima`,`float32x2`,e.count),e.geometry.kind===`polygons`&&(t=g(r,`${i}-${e.name}-feature-rings`,`uint32x2`,e.count),A.push(o(r,`${i}-${e.name}-rings`,{featureCount:e.count,geometry:e.geometry,featureRings:t}))),A.push(u(r,`${i}-${e.name}-bounds`,{featureCount:e.count,geometry:e.geometry,featureRings:t,minima:n,maxima:a})),{...e,minima:n,maxima:a,featureRings:t})}),[le,ue]=M,P,fe;if(D)P={nodeMinima:D.storage.nodeMinima,nodeMaxima:D.storage.nodeMaxima,leafIds:D.storage.leafIds,internalNodeCount:D.bvhInternalNodeCount,spatialSort:D.spatialSort},fe=D.storage.overflow;else{let{bvh:e,nodes:t}=ce(r,i,ue.minima,ue.maxima,this.leafCapacity);A.push(...t),P={nodeMinima:e.nodeMinima,nodeMaxima:e.nodeMaxima,leafIds:e.leafIds,internalNodeCount:e.internalNodeCount},fe=e.overflow}let F=g(r,`${i}-state`,`uint32`,6),I=a.candidateTruncated??(j||a.candidateTruncatedCount?g(r,`${i}-candidate-truncated`,`uint32`,s):void 0),pe=g(r,`${i}-page-counts`,`uint32`,s),ve=g(r,`${i}-page-offsets`,`uint32`,s+5,me.STORAGE|me.INDIRECT),L=g(r,`${i}-candidate-pairs`,`uint32x2`,k+ +!!this.distanceView),R=T?.distances?g(r,`${i}-distance-pairs`,`uint32x2`,y):void 0;R&&A.push(h(r,{id:`${i}-clear-distance-pairs`,operation:$,view:R,type:`u32`,value:`0xffffffffu`,componentCount:y*2}));let z=g(r,`${i}-flags`,`uint32`,k+1),B=j?void 0:g(r,`${i}-flag-offsets`,`uint32`,k),V=j?void 0:ne?.leftRows??g(r,`${i}-left-ids`,`uint32`,y),H=j?void 0:ne?.rightRows??g(r,`${i}-right-ids`,`uint32`,y),U=T||j?g(r,`${i}-left-match-counts`,`uint32`,s+1):void 0,W=this.usesRelateEngine?g(r,`${i}-matrices`,`uint32`,k):void 0,ye=2*k+1,be=x.kind===`polygons`?x:S.kind===`polygons`?S:null,G=ae?{pairPoints:g(r,`${i}-pair-points`,`float32x2`,ye),pairGeometryOffsets:g(r,`${i}-pair-geometry-offsets`,`uint32`,ye+1),pairClassifications:g(r,`${i}-pair-classifications`,`uint32`,ye)}:void 0;A.push(h(r,{id:`${i}-clear-state`,operation:$,view:F,type:`u32`,value:`0u`})),U&&A.push(h(r,{id:`${i}-clear-left-counts`,operation:$,view:U,type:`u32`,value:`0u`}));let K=a.excludeSameRow?` && left != right`:``,xe=this.sphereRadius===void 0?{kind:`planar`}:{kind:`great-circle`,radius:this.sphereRadius},q=M.map(e=>({prefix:e.name,kind:e.geometry.kind,vertexCount:e.geometry.positions.length,partCount:n(e.geometry)?e.geometry.kind===`points`?e.geometry.positions.length:e.geometry.lineOffsets.length-1:void 0})),J=[],Y=this.distanceView?{expression:`bitcast<f32>(candidatePairs[candidatePairsOffset + ${k*2}u])`}:a.distance??0;for(let e of M){let t=e.name;n(e.geometry)&&J.push({name:`${t}FeatureOffsets`,view:e.geometry.featureOffsets,type:`u32`,access:`read`}),J.push({name:`${t}Positions`,view:e.geometry.positions,type:`f32`,access:`read`}),e.geometry.kind===`lines`?J.push({name:`${t}RingOffsets`,view:e.geometry.lineOffsets,type:`u32`,access:`read`}):e.geometry.kind===`polygons`&&J.push({name:`${t}FeatureRings`,view:e.featureRings,type:`u32`,access:`read`},{name:`${t}RingOffsets`,view:e.geometry.ringOffsets,type:`u32`,access:`read`})}for(let e=0;e<this.candidatePages.pageCount;e++){let t=this.candidatePages.pageCount>1?`${this.id}-tile-${e}`:this.id,n=k,i=e*n,o=Math.min(n,this.candidateCapacity-i);{let e=[{name:`state`,view:F,type:`u32`,access:`read_write`},{name:`candidatePairs`,view:L,type:`u32`,access:`read_write`},{name:`flags`,view:z,type:`u32`,access:`read_write`}];this.distanceView&&e.push({name:`distanceRow`,view:this.distanceView,type:`f32`,access:`read`}),G&&e.push({name:`pairPoints`,view:G.pairPoints,type:`f32`,access:`read_write`},{name:`pairGeometryOffsets`,view:G.pairGeometryOffsets,type:`u32`,access:`read_write`});let i=[5,n,G?.pairGeometryOffsets.length??0];A.push(m(r,{id:`${t}-clear`,operation:$,variant:`clear`,bindings:e,invocationCount:Math.max(...i),declarations:N,body:`if (index == 0u) { flags[flagsOffset + ${n}u] = state[stateOffset + 2u]; }
  if (index < ${n}u) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
    flags[flagsOffset + index] = 0u;
  }
  ${this.distanceView?`if (index == 0u) { candidatePairs[candidatePairsOffset + ${n*2}u] = bitcast<u32>(distanceRow[distanceRowOffset]); }`:``}
  
  ${G?`if (index < ${G.pairPoints.length}u) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
    pairPoints[pairPointsOffset + index * 2u] = quietNaN;
    pairPoints[pairPointsOffset + index * 2u + 1u] = quietNaN;
  }
  if (index < ${G.pairGeometryOffsets.length}u) {
    pairGeometryOffsets[pairGeometryOffsetsOffset + index] =
      select(0u, ${be.polygonOffsets.length-1}u, index == ${G.pairGeometryOffsets.length-1}u);
  }`:``}`}))}A.push(...oe(r,{id:t,leftCount:s,rightCount:c,leftMinima:le.minima,leftMaxima:le.maxima,tree:P,candidateCapacity:this.candidateCapacity,page:{firstSlot:i,length:o,initialize:e===0,counts:pe,offsets:ve,metadataRow:s},margin:w===`dwithin`?this.distanceView??a.distance:0,sphereRadius:this.sphereRadius,state:F,candidatePairs:L,candidateTruncated:I,candidateOverflow:a.candidateOverflow,bvhOverflow:fe})),e===0&&a.candidateTruncatedCount&&I&&A.push(m(r,{id:`${t}-count-truncated`,operation:$,variant:`count-truncated`,bindings:[{name:`candidateTruncated`,view:I,type:`u32`,access:`read`},{name:`state`,view:F,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,body:`if (candidateTruncated[candidateTruncatedOffset + index] != 0u) {
    atomicAdd(&state[stateOffset + 4u], 1u);
  }`})),A.push(ge(r,{id:`${t}-page-count`,operation:$,invocationCount:1,parameterValues:[i,o],bindings:[{name:`state`,view:F,type:`u32`,access:`read_write`}],body:`let retained = min(state[stateOffset], ${this.candidateCapacity}u);
  let first = min(getChunkParameter(36u), retained);
  state[stateOffset + 5u] = min(retained - first, getChunkParameter(37u));`}));let l=p(r,{id:`${t}-candidate-dispatch`,operation:$,count:F,countRow:5,maximumItemCount:n});A.push(l.updateNode);let u=new Map,d=e=>{let i=u.get(e);return i||(i=p(r,{id:`${t}-candidate-dispatch-${e}`,operation:$,count:F,countRow:5,maximumItemCount:n,workgroupSize:e,invocationsPerItem:e}),u.set(e,i),A.push(i.updateNode)),i};if(a.onAttribute&&A.push(m(r,{id:`${t}-on-attribute`,operation:$,variant:`on-attribute`,bindings:[{name:`candidatePairs`,view:L,type:`u32`,access:`read_write`},{name:`leftKeys`,view:a.onAttribute.left,type:`u32`,access:`read`},{name:`rightKeys`,view:a.onAttribute.right,type:`u32`,access:`read`}],invocationCount:n,condition:l.condition,declarations:N,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  if (leftKeys[leftKeysOffset + left] != rightKeys[rightKeysOffset + right]) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
  }`})),G){let e=x.kind===`points`,i=e?x:S,a=e?S:x;A.push(m(r,{id:`${t}-expand`,operation:$,variant:`expand`,bindings:[{name:`candidatePairs`,view:L,type:`u32`,access:`read`},{name:`pointPositions`,view:i.positions,type:`f32`,access:`read`},{name:`featureOffsets`,view:a.featureOffsets,type:`u32`,access:`read`},{name:`pairPoints`,view:G.pairPoints,type:`f32`,access:`read_write`},{name:`pairGeometryOffsets`,view:G.pairGeometryOffsets,type:`u32`,access:`read_write`}],invocationCount:n,condition:l.condition,declarations:N,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let pointRow = ${e?`left`:`right`};
  let polygonRow = ${e?`right`:`left`};
  let row = index * 2u + 1u;
  pairPoints[pairPointsOffset + row * 2u] = pointPositions[pointPositionsOffset + pointRow * 2u];
  pairPoints[pairPointsOffset + row * 2u + 1u] = pointPositions[pointPositionsOffset + pointRow * 2u + 1u];
  pairGeometryOffsets[pairGeometryOffsetsOffset + row] = featureOffsets[featureOffsetsOffset + polygonRow];
  pairGeometryOffsets[pairGeometryOffsetsOffset + row + 1u] = featureOffsets[featureOffsetsOffset + polygonRow + 1u];`}));let o=p(r,{id:`${t}-robust-dispatch`,operation:$,count:F,countRow:5,maximumItemCount:n,invocationsPerItem:2,additionalInvocationCount:1});A.push(o.updateNode),A.push(...C(r,()=>new ee({id:`${t}-classify`,points:G.pairPoints,polygonPositions:a.positions,geometryOffsets:G.pairGeometryOffsets,polygonOffsets:a.polygonOffsets,ringOffsets:a.ringOffsets,output:G.pairClassifications}).addToGraph(r)).map(e=>(te(t,e.type===`compute`,`robust point-in-polygon classification must declare compute passes only`),{...e,condition:o.condition})))}if(W)this.addRelateNodes(r,A,{id:t,candidateCapacity:n,sides:M,sideSpecs:q,sideBindings:J,candidatePairs:L,matrices:W,flags:z,state:F,robust:G,sameRowTest:K,candidateCondition:l.condition,cooperativeCondition:G?void 0:d(256).condition});else if(G){let e=x.kind===`points`,i=w===`contains`&&e||w===`within`&&!e,a=w===`intersects`?`classification == INSIDE || classification == BOUNDARY`:`classification == INSIDE`;A.push(m(r,{id:`${t}-resolve`,operation:$,variant:`resolve`,bindings:[{name:`candidatePairs`,view:L,type:`u32`,access:`read`},{name:`pairClassifications`,view:G.pairClassifications,type:`u32`,access:`read`},{name:`flags`,view:z,type:`u32`,access:`read_write`},{name:`state`,view:F,type:`atomic<u32>`,access:`read_write`}],invocationCount:n,condition:l.condition,declarations:`${N}
const INSIDE: u32 = ${v.inside}u;
const BOUNDARY: u32 = ${v.boundary}u;
const UNCERTAIN: u32 = ${v.uncertain}u;`,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let classification = pairClassifications[pairClassificationsOffset + index * 2u + 1u];
    if (classification == UNCERTAIN) {
      atomicAdd(&state[stateOffset + 1u], 1u);
    } else {
      matched = ${i?`false`:`(${a})`}${K};
    }
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`})),A.push(m(r,{id:`${t}-fallback`,operation:$,variant:`fallback-${x.kind}-${S.kind}-${w}`,bindings:[...J,{name:`candidatePairs`,view:L,type:`u32`,access:`read`},{name:`pairClassifications`,view:G.pairClassifications,type:`u32`,access:`read`},{name:`flags`,view:z,type:`u32`,access:`read_write`}],invocationCount:n,condition:l.condition,declarations:`${de(q[0],q[1],w,0)}
const UNCERTAIN: u32 = ${v.uncertain}u;`,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  if (pairClassifications[pairClassificationsOffset + index * 2u + 1u] != UNCERTAIN) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  flags[flagsOffset + index] = select(0u, 1u, pairMatches(left, right)${K});`}))}else{let e=!this.usesWorkgroupDistance&&J.length+3<=it;A.push(m(r,{id:`${t}-exact`,operation:$,variant:`exact-${x.kind}-${S.kind}-${w}${this.sphereRadius===void 0?``:`-great-circle`}`,bindings:[...J,{name:`candidatePairs`,view:L,type:`u32`,access:`read`},{name:`flags`,view:z,type:`u32`,access:`read_write`},...e?[{name:`state`,view:F,type:`atomic<u32>`,access:`read_write`}]:[]],...this.usesWorkgroupDistance?{invocationCount:n*64,workgroupSize:64,guardIndex:!1,condition:d(64).condition,declarations:Se(q[0],q[1],Y),body:`let slot = index / 64u;
  let left = candidatePairs[candidatePairsOffset + slot * 2u];
  let right = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  let matched = dwithinWorkgroupPair(left, right, left != NO_FEATURE, localInvocationIndex);
  if (localInvocationIndex == 0u) {
    flags[flagsOffset + slot] = select(0u, 1u, matched${K});
  }`}:{invocationCount:n,condition:l.condition,declarations:de(q[0],q[1],w,Y,xe),body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  uncertainOrientation = false;
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    matched = pairMatches(left, right)${K};
    ${e?`if (uncertainOrientation) { atomicAdd(&state[stateOffset + 1u], 1u); }`:``}
  }
  flags[flagsOffset + index] = select(0u, 1u, matched)${e?``:` | select(0u, 2u, uncertainOrientation)`};`}})),!this.usesWorkgroupDistance&&!e&&A.push(m(r,{id:`${t}-count-uncertainty`,operation:$,variant:`candidate-uncertainty`,bindings:[{name:`flags`,view:z,type:`u32`,access:`read_write`},{name:`state`,view:F,type:`atomic<u32>`,access:`read_write`}],invocationCount:n,condition:l.condition,body:`let flag = flags[flagsOffset + index];
  if ((flag & 2u) != 0u) { atomicAdd(&state[stateOffset + 1u], 1u); }
  flags[flagsOffset + index] = flag & 1u;`}))}if(j&&O&&U)this.addAntiNodes(r,A,{id:t,candidateCapacity:n,countOnly:!0,candidatePairs:L,flags:z,leftMatchCounts:U,unmatched:O,state:F,candidateTruncated:I,candidateCondition:l.condition});else if(V&&H&&B){A.push(...new _({id:`${t}-scan-flags`,input:r.createDataView(z.buffer,{format:`uint32`,length:n}),output:B,mode:`exclusive`}).getCommandNodes(r));let e=[{name:`candidatePairs`,view:L,type:`u32`,access:`read`},{name:`flags`,view:z,type:`u32`,access:`read`},{name:`flagOffsets`,view:B,type:`u32`,access:`read`},{name:`leftRows`,view:V,type:`u32`,access:`read_write`},{name:`rightRows`,view:H,type:`u32`,access:`read_write`}];R&&e.push({name:`distancePairs`,view:R,type:`u32`,access:`read_write`}),re&&W&&e.push({name:`matrices`,view:W,type:`u32`,access:`read`},{name:`relateOut`,view:re,type:`u32`,access:`read_write`}),A.push(m(r,{id:`${t}-scatter`,operation:$,variant:re?`scatter-relate`:`scatter`,bindings:e,invocationCount:n,condition:l.condition,declarations:`const PAIR_CAPACITY: u32 = ${y}u;`,body:`if (flags[flagsOffset + index] != 0u) {
    let base = flags[flagsOffset + ${n}u];
    let localSlot = flagOffsets[flagOffsetsOffset + index];
    let slot = base + min(localSlot, 0xffffffffu - base);
    if (slot < PAIR_CAPACITY) {
      leftRows[leftRowsOffset + slot] = candidatePairs[candidatePairsOffset + index * 2u];
      rightRows[rightRowsOffset + slot] = candidatePairs[candidatePairsOffset + index * 2u + 1u];
      ${R?`distancePairs[distancePairsOffset + slot * 2u] = candidatePairs[candidatePairsOffset + index * 2u];
      distancePairs[distancePairsOffset + slot * 2u + 1u] = candidatePairs[candidatePairsOffset + index * 2u + 1u];`:``}
      ${re&&W?`relateOut[relateOutOffset + slot] = matrices[matricesOffset + index] & ${rt}u;`:``}
    }
  }`})),A.push(m(r,{id:`${t}-match-total`,operation:$,variant:`match-total`,bindings:[{name:`state`,view:F,type:`u32`,access:`read_write`},{name:`flags`,view:z,type:`u32`,access:`read`},{name:`flagOffsets`,view:B,type:`u32`,access:`read`}],invocationCount:1,declarations:`${_e}
const CANDIDATE_CAPACITY: u32 = ${n}u;`,body:`state[stateOffset + 2u] = getSaturatedScanSum(state[stateOffset + 2u],
    flagOffsets[flagOffsetsOffset + CANDIDATE_CAPACITY - 1u] +
    flags[flagsOffset + CANDIDATE_CAPACITY - 1u]);`}))}}j&&O&&U&&this.addAntiNodes(r,A,{id:i,candidateCapacity:k,finishOnly:!0,candidatePairs:L,flags:z,leftMatchCounts:U,unmatched:O,state:F,candidateTruncated:I,candidateCondition:void 0});{let e=this.scalarOutputs;for(let t=0;t===0||t<e.length;t+=6){let n=e.slice(t,t+6),a=[{name:`state`,view:F,type:`u32`,access:`read`},{name:`bvhOverflow`,view:fe,type:`u32`,access:`read`},...n.map(([e,t])=>({name:e,view:t,type:`u32`,access:`read_write`}))],o={overflow:`overflowValue`,weightsCount:`min(matchTotal, PAIR_CAPACITY)`,weightsTotal:`matchTotal`,weightsOverflow:`select(0u, 1u, pairIncomplete)`,weightsCandidateOverflow:`select(0u, 1u, candidateIncomplete)`,weightsUncertainCount:`state[stateOffset + 1u]`,pairsOverflow:`select(0u, 1u, pairIncomplete)`,pairsCandidateOverflow:`select(0u, 1u, candidateIncomplete)`,pairsCount:`min(matchTotal, PAIR_CAPACITY)`,pairsTotal:`matchTotal`,unmatchedOverflow:`select(0u, 1u, unmatchedIncomplete)`,unmatchedCandidateOverflow:`select(0u, 1u, candidateIncomplete)`,unmatchedCount:`min(unmatchedTotalValue, UNMATCHED_CAPACITY)`,unmatchedTotal:`unmatchedTotalValue`,candidateCount:`candidateTotal`,uncertainCount:`state[stateOffset + 1u]`,candidateTruncatedCount:`state[stateOffset + 4u]`},s=n.map(([e])=>`${e}[${e}Offset] = ${o[e]};`).join(`
  `);A.push(m(r,{id:t===0?`${i}-finalize`:`${i}-finalize-${t/6}`,operation:$,variant:`finalize`,bindings:a,invocationCount:1,declarations:`const CANDIDATE_CAPACITY: u32 = ${l}u;
const PAIR_CAPACITY: u32 = ${y}u;
const UNMATCHED_CAPACITY: u32 = ${O?.ids.length??0}u;`,body:`let candidateTotal = state[stateOffset];
  let matchTotal = ${j?`0u`:`state[stateOffset + 2u]`};
  let unmatchedTotalValue = ${j?`state[stateOffset + 3u]`:`0u`};
  let candidateIncomplete = bvhOverflow[bvhOverflowOffset] != 0u || candidateTotal > CANDIDATE_CAPACITY;
  let pairIncomplete = matchTotal > PAIR_CAPACITY;
  let unmatchedIncomplete = unmatchedTotalValue > UNMATCHED_CAPACITY;
  let overflowValue = select(0u, 1u, candidateIncomplete || pairIncomplete || unmatchedIncomplete);
  ${s}`}))}}if(T&&U&&V&&H){let e=p(r,{id:`${i}-pair-dispatch`,operation:$,count:F,countRow:2,maximumItemCount:y});A.push(e.updateNode),A.push(m(r,{id:`${i}-count-matches`,operation:$,variant:`count-matches`,bindings:[{name:`leftRows`,view:V,type:`u32`,access:`read`},{name:`state`,view:F,type:`u32`,access:`read`},{name:`leftMatchCounts`,view:U,type:`atomic<u32>`,access:`read_write`}],invocationCount:y,condition:e.condition,declarations:`const PAIR_CAPACITY: u32 = ${y}u;`,body:`if (index >= min(state[stateOffset + 2u], PAIR_CAPACITY)) { return; }
  atomicAdd(&leftMatchCounts[leftMatchCountsOffset + leftRows[leftRowsOffset + index]], 1u);`})),A.push(...new _({id:`${i}-scan-matches`,input:U,output:T.offsets,mode:`exclusive`}).getCommandNodes(r)),A.push(m(r,{id:`${i}-fill-weights`,operation:$,variant:`fill-weights`,bindings:[{name:`rightRows`,view:H,type:`u32`,access:`read`},{name:`state`,view:F,type:`u32`,access:`read`},{name:`neighbors`,view:T.neighbors,type:`u32`,access:`read_write`},{name:`weightValues`,view:T.weights,type:`f32`,access:`read_write`}],invocationCount:y,condition:e.condition,declarations:`const PAIR_CAPACITY: u32 = ${y}u;`,body:`if (index >= min(state[stateOffset + 2u], PAIR_CAPACITY)) { return; }
  neighbors[neighborsOffset + index] = rightRows[rightRowsOffset + index];
  weightValues[weightValuesOffset + index] = 1.0;`}))}return T?.distances&&R&&A.push(m(r,{id:`${i}-distances`,operation:$,variant:`distances`,bindings:[...J,{name:`distancePairs`,view:R,type:`u32`,access:`read`},{name:`distances`,view:T.distances,type:`f32`,access:`read_write`}],invocationCount:y,declarations:de(q[0],q[1],`dwithin`,0,xe),body:`let left = distancePairs[distancePairsOffset + index * 2u];
  let right = distancePairs[distancePairsOffset + index * 2u + 1u];
  if (left == NO_FEATURE || right == NO_FEATURE) {
    distances[distancesOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
    return;
  }
  let distanceValue = pairDistance(left, right);
  distances[distancesOffset + index] = ${this.sphereRadius===void 0?`distanceValue`:`distanceValue * ${f(this.sphereRadius)}`};`})),A}addRelateNodes(e,t,n){let{views:r}=this,{id:i,candidateCapacity:a}=n,{left:o,right:s}=r,{sideSpecs:c,sideBindings:l,candidatePairs:u,matrices:d,flags:f,state:p,robust:h}=n,g=`${H}u`,_=v;if(h){let[r,f,p]=o.kind===`points`?[`0FFFFF212`,`F0FFFF212`,`FF0FFF212`]:[`0F2FF1FF2`,`FF20F1FF2`,`FF2FF10F2`];t.push(m(e,{id:`${i}-resolve-relate`,operation:$,variant:`resolve-relate`,bindings:[{name:`candidatePairs`,view:u,type:`u32`,access:`read`},{name:`pairClassifications`,view:h.pairClassifications,type:`u32`,access:`read`},{name:`matrices`,view:d,type:`u32`,access:`read_write`}],invocationCount:a,condition:n.candidateCondition,declarations:`${N}
const INSIDE: u32 = ${_.inside}u;
const BOUNDARY: u32 = ${_.boundary}u;
const UNCERTAIN: u32 = ${_.uncertain}u;`,body:`var matrix = 0u;
  if (candidatePairs[candidatePairsOffset + index * 2u] != NO_FEATURE) {
    let classification = pairClassifications[pairClassificationsOffset + index * 2u + 1u];
    matrix = select(select(select(${W(p)}u, ${W(f)}u, classification == BOUNDARY), ${W(r)}u, classification == INSIDE), ${g}, classification == UNCERTAIN);
  }
  matrices[matricesOffset + index] = matrix;`})),t.push(m(e,{id:`${i}-fallback-relate`,operation:$,variant:`fallback-relate-${o.kind}-${s.kind}`,bindings:[...l,{name:`candidatePairs`,view:u,type:`u32`,access:`read`},{name:`pairClassifications`,view:h.pairClassifications,type:`u32`,access:`read`},{name:`matrices`,view:d,type:`u32`,access:`read_write`}],invocationCount:a,condition:n.candidateCondition,declarations:`${Ne(c[0],c[1])}
const UNCERTAIN: u32 = ${_.uncertain}u;`,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  if (pairClassifications[pairClassificationsOffset + index * 2u + 1u] != UNCERTAIN) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  matrices[matricesOffset + index] = pairRelate(left, right) | ${g};`}))}else t.push(m(e,{id:`${i}-relate`,operation:$,variant:`relate-${o.kind}-${s.kind}`,bindings:[...l,{name:`candidatePairs`,view:u,type:`u32`,access:`read`},{name:`matrices`,view:d,type:`u32`,access:`read_write`}],invocationCount:a*256,workgroupSize:256,guardIndex:!1,condition:n.cooperativeCondition,declarations:`${Ne(c[0],c[1],!0)}
${Pe(c)}`,body:`let slot = index / 256u;
  let left = candidatePairs[candidatePairsOffset + slot * 2u];
  let right = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  let matrix = relateWorkgroupPair(left, right, left != NO_FEATURE, localInvocationIndex);
  if (localInvocationIndex == 0u) {
    matrices[matricesOffset + slot] = matrix;
  }`}));t.push(m(e,{id:`${i}-mask`,operation:$,variant:`mask`,bindings:[{name:`candidatePairs`,view:u,type:`u32`,access:`read`},{name:`matrices`,view:d,type:`u32`,access:`read`},{name:`flags`,view:f,type:`u32`,access:`read_write`},{name:`state`,view:p,type:`atomic<u32>`,access:`read_write`},...this.patternView?[{name:`patternWords`,view:this.patternView,type:`u32`,access:`read`}]:[]],invocationCount:a,condition:n.candidateCondition,declarations:`${N}
${this.patternView?J(this.patternView.length/2):q(this.relatePatterns)}`,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let raw = matrices[matricesOffset + index];
    if ((raw & ${g}) != 0u) { atomicAdd(&state[stateOffset + 1u], 1u); }
    let matrix = raw & 0x3ffffu;
    // An all-F matrix marks an empty or invalid feature.
    matched = matrix != 0u && relateMatches(matrix)${n.sameRowTest};
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`}))}addAntiNodes(e,t,n){let{leftCount:r}=this,{id:i,candidateCapacity:a}=n,{candidatePairs:o,flags:s,leftMatchCounts:c,unmatched:l,state:u}=n;if(n.finishOnly||t.push(m(e,{id:`${i}-count-left-matches`,operation:$,variant:`count-left-matches`,bindings:[{name:`candidatePairs`,view:o,type:`u32`,access:`read`},{name:`flags`,view:s,type:`u32`,access:`read`},{name:`leftMatchCounts`,view:c,type:`atomic<u32>`,access:`read_write`}],invocationCount:a,condition:n.candidateCondition,body:`if (flags[flagsOffset + index] != 0u) {
    atomicAdd(&leftMatchCounts[leftMatchCountsOffset + candidatePairs[candidatePairsOffset + index * 2u]], 1u);
  }`})),n.countOnly)return;let d=g(e,`${i}-unmatched-flags`,`uint32`,r),f=g(e,`${i}-unmatched-offsets`,`uint32`,r);t.push(m(e,{id:`${i}-unmatched-flags`,operation:$,variant:`unmatched-flags`,bindings:[{name:`leftMatchCounts`,view:c,type:`u32`,access:`read`},...n.candidateTruncated?[{name:`candidateTruncated`,view:n.candidateTruncated,type:`u32`,access:`read`}]:[],{name:`unmatchedFlags`,view:d,type:`u32`,access:`read_write`}],invocationCount:r,body:`unmatchedFlags[unmatchedFlagsOffset + index] = select(0u, 1u, leftMatchCounts[leftMatchCountsOffset + index] == 0u${n.candidateTruncated?` && candidateTruncated[candidateTruncatedOffset + index] == 0u`:``});`})),t.push(...new _({id:`${i}-scan-unmatched`,input:d,output:f,mode:`exclusive`}).getCommandNodes(e)),t.push(m(e,{id:`${i}-scatter-unmatched`,operation:$,variant:`scatter-unmatched`,bindings:[{name:`unmatchedFlags`,view:d,type:`u32`,access:`read`},{name:`unmatchedOffsets`,view:f,type:`u32`,access:`read`},{name:`ids`,view:l.ids,type:`u32`,access:`read_write`},{name:`state`,view:u,type:`u32`,access:`read_write`}],invocationCount:r,declarations:`const UNMATCHED_CAPACITY: u32 = ${l.ids.length}u;`,body:`if (unmatchedFlags[unmatchedFlagsOffset + index] != 0u) {
    let slot = unmatchedOffsets[unmatchedOffsetsOffset + index];
    if (slot < UNMATCHED_CAPACITY) { ids[idsOffset + slot] = index; }
  }
  if (index == ${r-1}u) {
    state[stateOffset + 3u] = unmatchedOffsets[unmatchedOffsetsOffset + index] + unmatchedFlags[unmatchedFlagsOffset + index];
  }`}))}};export{pt as t};
import{v as e}from"./spatial-context-DwZEd4EZ.js";import{r as t}from"./kernel-C3tfQOn4.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,T as i,f as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{n as s,t as c}from"./philox-random-DfVxWNPn.js";import{n as l,t as u}from"./gpu-group-convex-hull-CpY8Xff2.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{A as f}from"./pointer-overlay-Qh8qflcA.js";import{t as p}from"./define-run-B-brkCe4.js";import{n as m,s as h,v as g,y as _}from"./polygon-set-CHja50mr.js";import{l as v,r as y,t as b}from"./pointer-support-C2rPRVFO.js";import{d as x,r as S}from"./shape-segments-DMcyF4V-.js";import{co as C}from"./entry-preview-D70TdSbK.js";var w=`GPUMinimumBounds`,T=2**24-1,E=2048;function D(e){return!e.labels}var O=class{id;props;spatialContext;groupCount;maximumHullVertices;constructor(t){this.id=t.id??`minimum-bounds`,this.props=t,this.spatialContext=e(this.id,t.spatialContext);let{id:n}=this,{output:s}=t;r(t.positions,[`float32x2`],`${n} positions`);let c=t.positions.length;if(c<1)throw Error(`${n} needs at least one row`);if(c>T)throw Error(`${n} supports fewer than 2^24 rows`);if(D(t)){for(let e of[`ringOffsets`,`polygonOffsets`,`featureOffsets`])if(r(t[e],[`uint32`],`${n} ${e}`),t[e].length<2)throw Error(`${n} ${e} must contain at least two rows`);this.groupCount=t.featureOffsets.length-1}else{if(r(t.labels,[`uint32`],`${n} labels`),t.labels.length!==c)throw Error(`${n} labels length must equal positions length`);a(n,`groupCount`,t.groupCount??0,1,T),this.groupCount=t.groupCount}this.maximumHullVertices=t.maximumHullVertices??E,a(n,`maximumHullVertices`,this.maximumHullVertices,3);let l=this.groupCount,u=[[`rectangleCorners`,s.rectangleCorners,[`float32x2`],l*4],[`rectangleSizes`,s.rectangleSizes,[`float32x4`],l],[`circles`,s.circles,[`float32x3`],l],[`longestLines`,s.longestLines,[`float32x4`],l],[`diameters`,s.diameters,[`float32`],l],[`bounds`,s.bounds,[`float32x4`],l],[`hullSizes`,s.hullSizes,[`uint32`],l],[`status.overflow`,s.status.overflow,[`uint32`],1],[`status.groupOverflow`,s.status.groupOverflow,[`uint32`],1]];for(let[e,t,a,o]of u)t&&(r(t,a,`${n} output.${e}`),i(n,t.length>=o,`output.${e} must hold at least ${o} rows`));if(!s.rectangleCorners&&!s.rectangleSizes&&!s.circles&&!s.longestLines&&!s.diameters&&!s.bounds)throw Error(`${n} needs at least one shape output`);for(let[e,t]of Object.entries(s.status))if(t)for(let[r,a]of u){if(!a||r===`status.${e}`)continue;let o=t.buffer===a.buffer&&t.byteOffset<a.byteOffset+a.length*a.rowByteLength&&a.byteOffset<t.byteOffset+t.length*t.rowByteLength;i(n,!o,`output.status.${e} must not overlap output.${r}`)}o(n,u.map(([,e])=>e),[t.positions,`ringOffsets`in t?t.ringOffsets:void 0,`polygonOffsets`in t?t.polygonOffsets:void 0,`featureOffsets`in t?t.featureOffsets:void 0,t.labels])}getCommandNodes(e){let{props:r,id:i,groupCount:a}=this,{output:o,positions:f}=r,{status:p,...m}=o,h=`ringOffsets`in r?r:void 0;d(e,i,{positions:f,labels:r.labels,ringOffsets:r.labels?void 0:h?.ringOffsets,polygonOffsets:r.labels?void 0:h?.polygonOffsets,featureOffsets:r.labels?void 0:h?.featureOffsets,...m,...Object.fromEntries(Object.entries(p).map(([e,t])=>[`status.${e}`,t]))}),l(e.device,i,f.length,a);let g=f.length,_=(n,r)=>t(e,`${i}-${n}`,`uint32`,r),v=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),y=(e,t,n=`f32`)=>({name:e,view:t,type:n,access:`read_write`}),b=(t,r,a,o,s=``)=>n(e,{id:`${i}-${t}`,operation:w,variant:t,bindings:a,invocationCount:r,declarations:s,body:o}),x=[],S=r.labels;if(!S){let{ringOffsets:n,polygonOffsets:a,featureOffsets:o}=r;S=t(e,`${i}-labels`,`uint32`,g);let s=(e,t,n,r)=>`if (${n} >= ${e}[${e}Offset] && ${n} < ${e}[${e}Offset + ${t}u]) {
    var ${r}Low = 0u;
    var ${r}High = ${t-1}u;
    while (${r}Low < ${r}High) {
      let middle = (${r}Low + ${r}High + 1u) >> 1u;
      if (${e}[${e}Offset + middle] <= ${n}) {
        ${r}Low = middle;
      } else {
        ${r}High = middle - 1u;
      }
    }
    ${r} = ${r}Low;`;x.push(b(`labels`,g,[v(`ringOffsets`,n,`u32`),v(`polygonOffsets`,a,`u32`),v(`featureOffsets`,o,`u32`),y(`labels`,S,`u32`)],`var ring = 0xffffffffu;
  var polygon = 0xffffffffu;
  var feature = 0xffffffffu;
  ${s(`ringOffsets`,n.length-1,`index`,`ring`)}
    ${s(`polygonOffsets`,a.length-1,`ring`,`polygon`)}
      ${s(`featureOffsets`,o.length-1,`polygon`,`feature`)}
      }
    }
  }
  labels[labelsOffset + index] = feature;`))}let C=_(`hull-indices`,g),T=t(e,`${i}-hull-positions`,`float32x2`,g),E=_(`hull-offsets`,a+1),D=_(`hull-counts`,a);x.push(...new u({id:`${i}-hull`,positions:f,labels:S,groupCount:a,maximumVerticesPerGroup:this.maximumHullVertices,vertexCapacity:g,output:{vertexIndices:C,vertexPositions:T,offsets:E,counts:D,sizes:o.hullSizes,status:{overflow:o.status.overflow,groupOverflow:o.status.groupOverflow}}}).getCommandNodes(e));let O=()=>[v(`offsets`,E,`u32`),v(`counts`,D,`u32`),v(`hull`,T,`f32`)],k=`let count = counts[countsOffset + index];
  let begin = offsets[offsetsOffset + index];
  let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var anchor = vec2<f32>(0.0, 0.0);
  if (count > 0u) {
    anchor = vec2<f32>(hull[hullOffset + begin * 2u], hull[hullOffset + begin * 2u + 1u]);
  }`,A=`const CALIPER_MINIMUM: u32 = 17u;
fn loadPoint(slot: u32, anchor: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(hull[hullOffset + slot * 2u], hull[hullOffset + slot * 2u + 1u]) - anchor;
}
// Rotating-caliper step: the hull is strictly convex, so a linear function along it rises to one
// maximum and falls again. Starting on the rising side, walk forward while one of the next two
// vertices is higher (two, to ride over float noise at the plateau of a perpendicular edge).
fn advanceMaximum(start: u32, begin: u32, count: u32, anchor: vec2<f32>, axis: vec2<f32>) -> u32 {
  var slot = start;
  var value = dot(loadPoint(begin + slot, anchor), axis);
  for (var step = 0u; step < count; step++) {
    var moved = false;
    for (var lookahead = 1u; lookahead <= 2u; lookahead++) {
      let candidate = (slot + lookahead) % count;
      let candidateValue = dot(loadPoint(begin + candidate, anchor), axis);
      if (candidateValue > value) {
        slot = candidate;
        value = candidateValue;
        moved = true;
        break;
      }
    }
    if (!moved) { break; }
  }
  return slot;
}
fn findMaximum(begin: u32, count: u32, anchor: vec2<f32>, axis: vec2<f32>) -> u32 {
  var best = 0u;
  var bestValue = dot(loadPoint(begin, anchor), axis);
  for (var slot = 1u; slot < count; slot++) {
    let value = dot(loadPoint(begin + slot, anchor), axis);
    if (value > bestValue) {
      best = slot;
      bestValue = value;
    }
  }
  return best;
}`;if(o.rectangleCorners||o.rectangleSizes){let e=O();o.rectangleCorners&&e.push(y(`corners`,o.rectangleCorners)),o.rectangleSizes&&e.push(y(`sizes`,o.rectangleSizes)),x.push(b(`rectangle`,a,e,`${k}
  var bestArea = 0.0;
  var bestEdge = 0xffffffffu;
  var bestDirection = vec2<f32>(1.0, 0.0);
  var bestRange = vec4<f32>(0.0);
  var rightSlot = 0u;
  var topSlot = 0u;
  var leftSlot = 0u;
  for (var edge = 0u; edge < count; edge++) {
    let from0 = loadPoint(begin + edge, anchor);
    let to0 = loadPoint(begin + (edge + 1u) % count, anchor);
    let delta = to0 - from0;
    let edgeLength = sqrt(dot(delta, delta));
    var direction = vec2<f32>(1.0, 0.0);
    if (edgeLength > 0.0) {
      direction = delta / edgeLength;
    }
    let normal = vec2<f32>(-direction.y, direction.x);
    var range = vec4<f32>(3.0e38, 3.0e38, -3.0e38, -3.0e38);
    if (count < CALIPER_MINIMUM) {
      for (var vertex = 0u; vertex < count; vertex++) {
        let point = loadPoint(begin + vertex, anchor);
        let along = dot(point, direction);
        let across = dot(point, normal);
        range = vec4<f32>(min(range.x, along), min(range.y, across), max(range.z, along), max(range.w, across));
      }
    } else {
      // Calipers: the edge's own side is the minimum across; the other three extremes turn
      // counter-clockwise with the edge, so each pointer only moves forward.
      if (edge == 0u) {
        rightSlot = findMaximum(begin, count, anchor, direction);
        topSlot = findMaximum(begin, count, anchor, normal);
        leftSlot = findMaximum(begin, count, anchor, -direction);
      } else {
        rightSlot = advanceMaximum(rightSlot, begin, count, anchor, direction);
        topSlot = advanceMaximum(topSlot, begin, count, anchor, normal);
        leftSlot = advanceMaximum(leftSlot, begin, count, anchor, -direction);
      }
      range = vec4<f32>(
        dot(loadPoint(begin + leftSlot, anchor), direction),
        min(dot(from0, normal), dot(to0, normal)),
        dot(loadPoint(begin + rightSlot, anchor), direction),
        dot(loadPoint(begin + topSlot, anchor), normal)
      );
    }
    let area = (range.z - range.x) * (range.w - range.y);
    if (bestEdge == 0xffffffffu || area < bestArea * (1.0 - 1.0e-6)) {
      bestArea = area;
      bestEdge = edge;
      bestDirection = direction;
      bestRange = range;
    }
  }
  let normal = vec2<f32>(-bestDirection.y, bestDirection.x);
  let width = bestRange.z - bestRange.x;
  let height = bestRange.w - bestRange.y;
  var angle = 0.0;
  if (bestDirection.x == 0.0) {
    angle = select(-1.5707964, 1.5707964, bestDirection.y > 0.0);
  } else {
    angle = atan2(bestDirection.y, bestDirection.x);
  }
  ${o.rectangleCorners?`for (var corner = 0u; corner < 4u; corner++) {
    let alongValue = select(bestRange.x, bestRange.z, corner == 1u || corner == 2u);
    let acrossValue = select(bestRange.y, bestRange.w, corner >= 2u);
    let point = anchor + bestDirection * alongValue + normal * acrossValue;
    corners[cornersOffset + (index * 4u + corner) * 2u] = select(nan, point.x, count > 0u);
    corners[cornersOffset + (index * 4u + corner) * 2u + 1u] = select(nan, point.y, count > 0u);
  }`:``}
  ${o.rectangleSizes?`let isValid = count > 0u;
  sizes[sizesOffset + index * 4u] = select(nan, width, isValid);
  sizes[sizesOffset + index * 4u + 1u] = select(nan, height, isValid);
  sizes[sizesOffset + index * 4u + 2u] = select(nan, angle, isValid);
  sizes[sizesOffset + index * 4u + 3u] = select(nan, width * height, isValid);`:``}`,A))}if(o.circles){let e=_(`circle-order`,g);x.push(b(`circle`,a,[...O(),y(`order`,e,`u32`),y(`circles`,o.circles)],`${k}
  if (count == 0u) {
    circles[circlesOffset + index * 3u] = nan;
    circles[circlesOffset + index * 3u + 1u] = nan;
    circles[circlesOffset + index * 3u + 2u] = nan;
    return;
  }
  for (var slot = 0u; slot < count; slot++) {
    order[orderOffset + begin + slot] = slot;
  }
  // Deterministic Fisher-Yates shuffle: one Philox block per (group, slot). The circle does not
  // depend on the order in exact arithmetic; the shuffle only bounds the expected running time.
  for (var slot = count - 1u; slot > 0u; slot--) {
    let pick = philox4x32(vec4<u32>(index, slot, 0u, 0u), vec2<u32>(0u, CIRCLE_SHUFFLE_PURPOSE)).x % (slot + 1u);
    let held = order[orderOffset + begin + slot];
    order[orderOffset + begin + slot] = order[orderOffset + begin + pick];
    order[orderOffset + begin + pick] = held;
  }
  var center = loadPoint(begin + order[orderOffset + begin], anchor);
  var radiusSquared = 0.0;
  for (var i = 1u; i < count; i++) {
    let pointI = loadPoint(begin + order[orderOffset + begin + i], anchor);
    if (isOutside(pointI, center, radiusSquared)) {
      center = pointI;
      radiusSquared = 0.0;
      for (var j = 0u; j < i; j++) {
        let pointJ = loadPoint(begin + order[orderOffset + begin + j], anchor);
        if (isOutside(pointJ, center, radiusSquared)) {
          center = 0.5 * (pointI + pointJ);
          radiusSquared = dot(pointI - center, pointI - center);
          for (var k = 0u; k < j; k++) {
            let pointK = loadPoint(begin + order[orderOffset + begin + k], anchor);
            if (isOutside(pointK, center, radiusSquared)) {
              center = getCircumcenter(pointI, pointJ, pointK);
              radiusSquared = max(
                dot(pointI - center, pointI - center),
                max(dot(pointJ - center, pointJ - center), dot(pointK - center, pointK - center))
              );
            }
          }
        }
      }
    }
  }
  // Recompute the radius from the hull so the circle covers every vertex.
  var coverSquared = 0.0;
  for (var slot = 0u; slot < count; slot++) {
    let delta = loadPoint(begin + slot, anchor) - center;
    coverSquared = max(coverSquared, dot(delta, delta));
  }
  circles[circlesOffset + index * 3u] = anchor.x + center.x;
  circles[circlesOffset + index * 3u + 1u] = anchor.y + center.y;
  circles[circlesOffset + index * 3u + 2u] = sqrt(coverSquared);`,`${A}
${s}
const CIRCLE_SHUFFLE_PURPOSE: u32 = ${c.minimumBoundsCircleShuffle.id}u;
fn isOutside(point: vec2<f32>, center: vec2<f32>, radiusSquared: f32) -> bool {
  let delta = point - center;
  return dot(delta, delta) > radiusSquared * (1.0 + 4.0e-6);
}
fn getDiametralCenter(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
  let ab = dot(a - b, a - b);
  let ac = dot(a - c, a - c);
  let bc = dot(b - c, b - c);
  if (ab >= ac && ab >= bc) {
    return 0.5 * (a + b);
  }
  if (ac >= bc) {
    return 0.5 * (a + c);
  }
  return 0.5 * (b + c);
}
fn getCircumcenter(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
  let ab = b - a;
  let ac = c - a;
  let determinant = 2.0 * (ab.x * ac.y - ab.y * ac.x);
  let lengths = sqrt(dot(ab, ab) * dot(ac, ac));
  if (abs(determinant) <= 2.0e-7 * lengths) {
    return getDiametralCenter(a, b, c);
  }
  let abSquared = dot(ab, ab);
  let acSquared = dot(ac, ac);
  return a + vec2<f32>(
    (ac.y * abSquared - ab.y * acSquared) / determinant,
    (ab.x * acSquared - ac.x * abSquared) / determinant
  );
}`))}if(o.longestLines||o.diameters){let e=O();o.longestLines&&e.push(y(`lines`,o.longestLines)),o.diameters&&e.push(y(`diameters`,o.diameters)),x.push(b(`longest-line`,a,e,`${k}
  bestSquared = -1.0;
  bestFrom = 0u;
  bestTo = 0u;
  if (count < CALIPER_MINIMUM) {
    for (var i = 0u; i < count; i++) {
      for (var j = i; j < count; j++) {
        considerPair(begin, anchor, i, j);
      }
    }
  } else {
    // Rotating calipers: a farthest pair has parallel supporting lines, so it is found among the
    // vertices opposite each edge. Pairs are compared exactly as the full scan does.
    var opposite = 0u;
    for (var edge = 0u; edge < count; edge++) {
      let next = (edge + 1u) % count;
      let edgeStart = loadPoint(begin + edge, anchor);
      let edgeVector = loadPoint(begin + next, anchor) - edgeStart;
      let normal = vec2<f32>(-edgeVector.y, edgeVector.x);
      if (edge == 0u) {
        opposite = findMaximum(begin, count, anchor, normal);
      } else {
        opposite = advanceMaximum(opposite, begin, count, anchor, normal);
      }
      for (var offset = count - 1u; offset <= count + 1u; offset++) {
        let candidate = (opposite + offset) % count;
        considerPair(begin, anchor, edge, candidate);
        considerPair(begin, anchor, next, candidate);
      }
    }
  }
  let isValid = count > 0u;
  ${o.longestLines?`let startPoint = loadPoint(begin + bestFrom, anchor) + anchor;
  let endPoint = loadPoint(begin + bestTo, anchor) + anchor;
  lines[linesOffset + index * 4u] = select(nan, startPoint.x, isValid);
  lines[linesOffset + index * 4u + 1u] = select(nan, startPoint.y, isValid);
  lines[linesOffset + index * 4u + 2u] = select(nan, endPoint.x, isValid);
  lines[linesOffset + index * 4u + 3u] = select(nan, endPoint.y, isValid);`:``}
  ${o.diameters?`diameters[diametersOffset + index] = select(nan, sqrt(max(bestSquared, 0.0)), isValid);`:``}`,`${A}
var<private> bestSquared: f32;
var<private> bestFrom: u32;
var<private> bestTo: u32;
// Keeps the farthest pair; equal distances go to the lowest (first, second) pair in hull order.
fn considerPair(begin: u32, anchor: vec2<f32>, one: u32, other: u32) {
  let first = min(one, other);
  let second = max(one, other);
  let delta = loadPoint(begin + second, anchor) - loadPoint(begin + first, anchor);
  let distanceSquared = dot(delta, delta);
  if (distanceSquared > bestSquared ||
      (distanceSquared == bestSquared && (first < bestFrom || (first == bestFrom && second < bestTo)))) {
    bestSquared = distanceSquared;
    bestFrom = first;
    bestTo = second;
  }
}`))}return o.bounds&&x.push(b(`bounds`,a,[...O(),y(`bounds`,o.bounds)],`${k}
  var low = vec2<f32>(3.0e38, 3.0e38);
  var high = vec2<f32>(-3.0e38, -3.0e38);
  for (var slot = 0u; slot < count; slot++) {
    let point = loadPoint(begin + slot, anchor);
    low = min(low, point);
    high = max(high, point);
  }
  let isValid = count > 0u;
  bounds[boundsOffset + index * 4u] = select(nan, anchor.x + low.x, isValid);
  bounds[boundsOffset + index * 4u + 1u] = select(nan, anchor.y + low.y, isValid);
  bounds[boundsOffset + index * 4u + 2u] = select(nan, anchor.x + high.x, isValid);
  bounds[boundsOffset + index * 4u + 3u] = select(nan, anchor.y + high.y, isValid);`,A)),x}},k={irregularity:.3,vertexCount:10,fill:.8,rows:2,columns:3},A=p({parameters:C,caption:`Rotated rectangle, bounding circle and longest line of each polygon.`,createInputs:({seed:e,world:t})=>{let n=h(f(e),t,{...k,stretch:.7});return{seed:e,world:t,set:n,...y(n,1)}},build(e,t,n){let r=g(e,t.set),i=t.set.count,a=e.allocate(`rectangle-corners`,`float32x2`,i*4),o=e.allocate(`circles`,`float32x3`,i),s=e.allocate(`longest-lines`,`float32x4`,i),c=e.allocate(`overflow`,`uint32`,1);return e.graph.add(new O({positions:r.positions.view,featureOffsets:r.featureOffsets.view,polygonOffsets:r.polygonOffsets.view,ringOffsets:r.ringOffsets.view,maximumHullVertices:n.maximumHullVertices,output:{rectangleCorners:a.view,circles:o.view,longestLines:s.view,status:{overflow:c.view}}})),{outputs:{...r,circles:o,longestLines:s,rectangleSegments:x(e,`rectangle`,a),circleSegments:S(e,`circle`,o)},update:e=>{let n=h(f(t.seed),t.world,{...k,stretch:e.stretch});m(n,t.drag),t.set=n,_(r,n)}}},interaction:{kind:`drag`,hint:`pull a vertex outward`,rest:({inputs:e,toUnit:t})=>t(v(e)),resolve(e,{inputs:t,outputs:n,peek:r}){let i=b(t,e,0,.35),a=t.drag?.feature??0,o=r(n.circles),s=r(n.longestLines),c=s?Math.hypot(s[a*4+2]-s[a*4],s[a*4+3]-s[a*4+1]):void 0;return{rerun:!0,glyphs:i.glyphs,readout:`circle r ${o?o[a*3+2].toFixed(3):`...`} · longest ${c===void 0?`...`:c.toFixed(3)}`}}},rebuildOn:[`maximumHullVertices`],draw:({inputs:e,outputs:t,theme:n})=>[{layer:`lines`,context:!0,segments:t.outline,count:e.set.outlineCount},{layer:`lines`,segments:t.rectangleSegments,color:[1,.6,.2,1]},{layer:`lines`,segments:t.circleSegments,color:[.4,.75,1,1]},{layer:`lines`,segments:t.longestLines,color:[...n.ink,1],radius:1.8}]});export{A as default};
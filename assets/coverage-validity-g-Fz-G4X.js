import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{o as n,t as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as i,S as a,l as o,w as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{a as u,i as d,r as f,t as p}from"./segment-table-BeVxjLqW.js";import{t as m}from"./segment-bvh-CDgckfAv.js";import{n as h}from"./coverage-topology-kernels-ltNrCaNR.js";import{p as g}from"./wgsl-common-DfyXiYB0.js";import{t as _}from"./exact-geometry-wgsl-y7TUnqOh.js";import{a as v,i as y}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as b,r as x}from"./geometry-storage-validation-T89ZhmCj.js";import{t as S}from"./segment-storage-validation-BUxpxgMF.js";import{n as C}from"./geographic-distribution-validation-BjMQQJgk.js";import{v as w}from"./spatial-context-DwZEd4EZ.js";import{d as T,s as E}from"./pointer-overlay-Qh8qflcA.js";import{t as D}from"./tract-coverage-BWGWYaF1.js";import{t as O}from"./define-run-B-brkCe4.js";import{f as k,h as A,r as j,t as M}from"./lines-kx5qHdti.js";import{l as N,o as P,s as F}from"./polygons-D0hzhPvj.js";import{r as I}from"./text-DYSgVV-n.js";import{rs as L}from"./entry-preview-D70TdSbK.js";function R(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Coverage validity target must hold 4 elements`);let{gapWidth:n}=e;if(!Number.isFinite(n)||n<0)throw Error(`Coverage validity gapWidth must be finite and non-negative`);return t.set([n,0,0,0]),t}var z=`GPUCoverageValidity`,B=1024,V={unmatched:1,crossing:2,gap:4,overlap:8,uncertain:16},H=class{id;props;leafCapacity;spatialSort;spatialContext;constructor(e){let n=e.id??`coverage-validity`;if(this.id=n,this.props=e,this.spatialContext=w(n,e.spatialContext),i(e.positions,[`float32x2`],`${n} positions`),e.positions.length<1)throw Error(`${n} positions must hold at least one vertex`);t(e.ringOffsets,`${n} ringOffsets`),t(e.polygonOffsets,`${n} polygonOffsets`),t(e.featureOffsets,`${n} featureOffsets`);for(let[t,r]of[[`ringOffsets`,e.ringOffsets],[`polygonOffsets`,e.polygonOffsets],[`featureOffsets`,e.featureOffsets]])if(r.length<2)throw Error(`${n} ${t} must hold at least two entries`);if(i(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<4)throw Error(`${n} parameters must hold 4 float32 values`);this.spatialSort=e.spatialSort??e.positions.length>=B,this.leafCapacity=e.leafCapacity??o(e.positions.length),s(n,`leafCapacity`,this.leafCapacity,e.positions.length);let{output:r}=e;if(t(r.segmentFlags,`${n} output.segmentFlags`),r.segmentFlags.length!==e.positions.length)throw Error(`${n} output.segmentFlags length must equal positions length`);if(r.featureInvalidCounts&&(t(r.featureInvalidCounts,`${n} output.featureInvalidCounts`),r.featureInvalidCounts.length!==e.featureOffsets.length-1))throw Error(`${n} output.featureInvalidCounts length must equal the feature count`);for(let[e,i]of[[`status.invalidCount`,r.status?.invalidCount],[`isValid`,r.isValid]])if(i&&(t(i,`${n} output.${e}`),i.length<1))throw Error(`${n} output.${e} must hold one uint32`);l(n,[r.segmentFlags,r.featureInvalidCounts,r.status?.invalidCount,r.isValid],[e.positions,e.ringOffsets,e.polygonOffsets,e.featureOffsets,e.parameters]),C(n,{},{segmentFlags:r.segmentFlags,featureInvalidCounts:r.featureInvalidCounts,invalidCount:r.status?.invalidCount,isValid:r.isValid},[]),y(n,{featureInvalidCounts:r.featureInvalidCounts,invalidCount:r.status?.invalidCount}),x(n,[r.segmentFlags],[r.featureInvalidCounts,r.status?.invalidCount]),x(n,[r.status?.invalidCount],[r.isValid])}getCommandNodes(t){let{id:i,props:o}=this,{positions:s,ringOffsets:l,polygonOffsets:y,featureOffsets:x,parameters:C,output:w}=o;v(t,i,{positions:s,ringOffsets:l,polygonOffsets:y,featureOffsets:x,parameters:C,segmentFlags:w.segmentFlags,featureInvalidCounts:w.featureInvalidCounts,invalidCount:w.status?.invalidCount,isValid:w.isValid}),b(t,i,{segmentTable:s.length*32,ringOrientation:(l.length-1)*4,featureCounts:(x.length-1)*4,scalarCount:4}),S(t,i,s.length,this.leafCapacity,this.spatialSort),c(i,t,[s,l,y,x,C,w.segmentFlags,w.featureInvalidCounts,w.status?.invalidCount,w.isValid]),a(i,`positions rows`,o.positions.length,d(t.device));let T=s.length,E=l.length-1,D=x.length-1,O=(n,r)=>e(t,`${i}-${n}`,`uint32`,r),k=O(`ring-flip`,E),A=O(`segment-table`,T*8),j=e(t,`${i}-bounds-minima`,`float32x2`,T),M=e(t,`${i}-bounds-maxima`,`float32x2`,T),N=[h(t,{id:`${i}-ring-orientation`,operation:z,positions:s,ringOffsets:l,polygonOffsets:y,ringFlip:k})];N.push(p(t,{id:`${i}-segment-table`,operation:z,geometry:{kind:`polygons`,positions:s,ringOffsets:l,polygonOffsets:y,featureOffsets:x},table:A,minima:j,maxima:M}));let{bvh:P,nodes:F}=m(t,{id:i,operation:z,minima:j,maxima:M,leafCapacity:this.leafCapacity,spatialSort:this.spatialSort});N.push(...F);let I=(e,t,n)=>`{
    let queryMinimum = ${e};
    let queryMaximum = ${t};
    var node = 0u;
    loop {
      if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let other = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (other < SEGMENT_COUNT && segmentValid(other) && segmentFeature(other) != polygon) {
          ${n}
        }
      }
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }`;if(N.push(n(t,{id:`${i}-probe`,operation:z,variant:`probe`,bindings:[{name:`segmentTable`,view:A,type:`u32`,access:`read`},{name:`nodeMinima`,view:P.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:P.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:P.leafIds,type:`u32`,access:`read`},{name:`ringFlip`,view:k,type:`u32`,access:`read`},{name:`parameters`,view:C,type:`f32`,access:`read`},{name:`segmentFlags`,view:w.segmentFlags,type:`u32`,access:`read_write`}],invocationCount:T,declarations:`${f}
${u(`segment`)}
const SEGMENT_COUNT: u32 = ${T}u;
const INTERNAL_NODE_COUNT: u32 = ${P.internalNodeCount}u;
const FLAG_UNMATCHED: u32 = ${V.unmatched}u;
const FLAG_CROSSING: u32 = ${V.crossing}u;
const FLAG_GAP: u32 = ${V.gap}u;
const FLAG_OVERLAP: u32 = ${V.overlap}u;
const FLAG_UNCERTAIN: u32 = ${V.uncertain}u;
const MATCHED_BIT: u32 = 0x80000000u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;

${g}
${_}

fn isFlipped(row: u32) -> bool {
  return ringFlip[ringFlipOffset + segmentRing(row)] != 0u;
}

fn samePoint(first: vec2f, second: vec2f) -> bool {
  return first.x == second.x && first.y == second.y;
}

// Length of the part of segment (start, end) that projects onto segment (otherStart, otherEnd), times
// the length of the latter, so that no division is needed and ties with gapWidth stay exact. Zero
// or negative when the one does not project onto the other at all.
fn projectedLengthScaled(start: vec2f, end: vec2f, otherStart: vec2f, otherEnd: vec2f) -> f32 {
  let direction = otherEnd - otherStart;
  let lengthSquared = dot(direction, direction);
  let first = dot(start - otherStart, direction);
  let second = dot(end - otherStart, direction);
  return min(max(first, second), lengthSquared) - max(min(first, second), 0.0);
}

// The point of segment (start, end) nearest to the projection of point on its line: GEOS
// LineSegment::project of one point, clamped to the segment. An end is returned as it is.
fn projectOntoSegment(point: vec2f, start: vec2f, end: vec2f) -> vec2f {
  let direction = end - start;
  let lengthSquared = dot(direction, direction);
  let along = dot(point - start, direction);
  if (along <= 0.0) { return start; }
  if (along >= lengthSquared) { return end; }
  return start + direction * (along / lengthSquared);
}

// The GEOS gap rule (coverage/InvalidSegmentDetector::isNearlyParallel): each segment is projected
// onto the other and clipped to it. The pair is a gap when both clipped projections are longer than
// gapWidth and their matching ends lie within gapWidth of each other. (a, b) is the tested segment
// and (c, d) the neighbor's, both in input order, as GEOS reads them. Everything is formed relative
// to a, so that the f32 step of large projected coordinates does not enter the distances.
fn isNearlyParallel(a: vec2f, b: vec2f, c: vec2f, d: vec2f, gapWidth: f32) -> bool {
  let origin = vec2f(0.0);
  let localB = b - a;
  let localC = c - a;
  let localD = d - a;
  if (projectedLengthScaled(localC, localD, origin, localB) <= gapWidth * length(localB) ||
      projectedLengthScaled(origin, localB, localC, localD) <= gapWidth * distance(localC, localD)) {
    return false;
  }
  // Projection of (c, d) on (a, b), and of (a, b) on (c, d).
  let onRowStart = projectOntoSegment(localC, origin, localB);
  let onRowEnd = projectOntoSegment(localD, origin, localB);
  var onOtherStart = projectOntoSegment(origin, localC, localD);
  var onOtherEnd = projectOntoSegment(localB, localC, localD);
  if (distance(onRowStart, onOtherEnd) < distance(onRowStart, onOtherStart)) {
    let swap = onOtherStart;
    onOtherStart = onOtherEnd;
    onOtherEnd = swap;
  }
  return distance(onRowStart, onOtherStart) <= gapWidth &&
    distance(onRowEnd, onOtherEnd) <= gapWidth;
}

// Flags of segment row against segment other (another polygon). Bit 31 reports a matched edge.
fn comparePair(row: u32, other: u32, gapWidth: f32) -> u32 {
  let a = segmentStart(row);
  let b = segmentEnd(row);
  let c = segmentStart(other);
  let d = segmentEnd(other);
  let sameDirection = samePoint(a, c) && samePoint(b, d);
  if (sameDirection || (samePoint(a, d) && samePoint(b, c))) {
    let effectivelySame = sameDirection != (isFlipped(row) != isFlipped(other));
    return select(MATCHED_BIT, FLAG_OVERLAP, effectivelySame);
  }
  let hit = classifySegments(a, b, c, d, false);
  if (hit.kind == KIND_UNCERTAIN) { return FLAG_UNCERTAIN; }
  if (hit.kind == KIND_PROPER) { return FLAG_CROSSING; }
  if (hit.kind == KIND_OVERLAP) { return FLAG_UNMATCHED; }
  if (hit.kind == KIND_TOUCH) {
    // A shared vertex is fine; an endpoint of either segment inside the other is a T-junction.
    let onSegmentRow = samePoint(hit.point, a) || samePoint(hit.point, b);
    let onSegmentOther = samePoint(hit.point, c) || samePoint(hit.point, d);
    if (!(onSegmentRow && onSegmentOther)) { return FLAG_UNMATCHED; }
  }
  if (gapWidth > 0.0 && isNearlyParallel(a, b, c, d, gapWidth)) { return FLAG_GAP; }
  return 0u;
}

// Contribution of segment other to the winding number of the probe (effective interior on the
// left). The probe is the symbolic point beside the exact midpoint of the tested segment (the
// shared side probe), so a neighbor's edge closer to the segment than one f32 step is still on
// the right side of it. An uncertain sign (non-finite input) counts as no crossing.
fn windingContribution(other: u32, probe: ExactProbe) -> i32 {
  var start = segmentStart(other);
  var end = segmentEnd(other);
  if (isFlipped(other)) {
    let swap = start;
    start = end;
    end = swap;
  }
  return exactProbeCrossing(probe, start, end).x;
}`,body:`var flags = 0u;
  if (segmentValid(index)) {
    let gapWidth = max(parameters[parametersOffset], 0.0);
    let polygon = segmentFeature(index);
    let a = segmentStart(index);
    let b = segmentEnd(index);
    var matched = false;
    ${I(`min(a, b) - vec2f(gapWidth)`,`max(a, b) + vec2f(gapWidth)`,`let result = comparePair(index, other, gapWidth);
          matched = matched || (result & MATCHED_BIT) != 0u;
          flags = flags | (result & ~MATCHED_BIT);`)}
    // A segment matched by an identical segment of another polygon is not tested against the rest,
    // as in GEOS; only a same-side duplicate (FLAG_OVERLAP) stays flagged.
    if (matched) {
      flags = flags & FLAG_OVERLAP;
    }
    if (flags == 0u && !matched) {
      // The rounded midpoint only selects BVH candidates: every edge the ray from the exact
      // midpoint can cross has that height and a larger x inside its bounds.
      let midpoint = a * 0.5 + b * 0.5;
      let probe = ExactProbe(midpoint, a, b, 1);
      var winding = 0;
      ${I(`vec2f(midpoint.x, midpoint.y)`,`vec2f(FLOAT32_MAXIMUM, midpoint.y)`,`winding = winding + windingContribution(other, probe);`)}
      if (winding > 0) { flags = FLAG_OVERLAP; }
    }
  }
  segmentFlags[segmentFlagsOffset + index] = flags;`})),w.featureInvalidCounts||w.status?.invalidCount||w.isValid){let e=w.featureInvalidCounts??O(`feature-invalid-counts`,D),a=w.status?.invalidCount??O(`invalid-count`,1);N.push(r(t,{id:`${i}-clear-polygon-counts`,operation:z,view:e,type:`u32`,value:`0u`}),r(t,{id:`${i}-clear-total`,operation:z,view:a,type:`u32`,value:`0u`}));let o=[{name:`segmentTable`,view:A,type:`u32`,access:`read`},{name:`segmentFlags`,view:w.segmentFlags,type:`u32`,access:`read`},{name:`polygonCounts`,view:e,type:`atomic<u32>`,access:`read_write`},{name:`total`,view:a,type:`atomic<u32>`,access:`read_write`}];N.push(n(t,{id:`${i}-count`,operation:z,variant:`count`,bindings:o,invocationCount:T,declarations:`const SEGMENT_NONE: u32 = 0xffffffffu;
${u(`segment`)}`,body:`if (segmentFlags[segmentFlagsOffset + index] != 0u && segmentValid(index)) {
    atomicAdd(&polygonCounts[polygonCountsOffset + segmentFeature(index)], 1u);
    atomicAdd(&total[totalOffset], 1u);
  }`})),w.isValid&&N.push(n(t,{id:`${i}-finalize`,operation:z,variant:`finalize`,bindings:[{name:`total`,view:a,type:`u32`,access:`read`},{name:`isValid`,view:w.isValid,type:`u32`,access:`read_write`}],invocationCount:1,body:`isValid[isValidOffset] = select(0u, 1u, total[totalOffset] == 0u);`}))}return N}};function U(e,t,n,r){return(e[n*2]-e[t*2])*(e[r*2+1]-e[t*2+1])-(e[n*2+1]-e[t*2+1])*(e[r*2]-e[t*2])}function W(e,t,n,r,i){return U(e,t,n,i)>=0&&U(e,n,r,i)>=0&&U(e,r,t,i)>=0}function G(e,t,n){let r=[];for(let e=t;e<n;e++)r.push(e);let i=0;for(let t=0;t<r.length;t++){let n=r[t],a=r[(t+1)%r.length];i+=e[n*2]*e[a*2+1]-e[a*2]*e[n*2+1]}let a=i>=0?1:-1,o=[],s=r.length*r.length;for(;r.length>3&&s-->0;){let t=!1;for(let n=0;n<r.length;n++){let i=r[(n+r.length-1)%r.length],s=r[n],c=r[(n+1)%r.length];if(!(U(e,i,s,c)*a<=0)&&!r.some(t=>t!==i&&t!==s&&t!==c&&(a>0?W(e,i,s,c,t):W(e,c,s,i,t)))){o.push(i,s,c),r.splice(n,1),t=!0;break}}if(!t)break}if(r.length===3)o.push(r[0],r[1],r[2]);else if(r.length>3)for(let e=1;e+1<r.length;e++)o.push(r[0],r[e],r[e+1]);return o}function K(e,t){let n=new Float32Array(t.length*2);return t.forEach((t,r)=>{n[r*2]=e[t*2],n[r*2+1]=e[t*2+1]}),n}var q=O({parameters:L,caption:`Coverage edges flagged as invalid (orange): overlaps and gaps narrower than gapWidth.`,createInputs:({random:e,world:t})=>{let n=D(e,t,4),r=new Map;[[.05,.5],[.2,1],[.35,2],[.5,4],[.65,3],[.8,-.8],[.95,1.5]].forEach(([e,t])=>r.set(Math.floor(n.featureCount*e),t));let i=e=>{let t=new Float32Array(n.positions);for(let i=0;i<n.featureCount;i++){if(!r.has(i))continue;let a=e*r.get(i),o=n.centers[i*2],s=n.centers[i*2+1],c=n.ringOffsets[i],l=n.ringOffsets[i+1],u=0;for(let e=c;e<l;e++)u=Math.max(u,Math.hypot(t[e*2]-o,t[e*2+1]-s));let d=1-a/(2*u);for(let e=c;e<l;e++)t[e*2]=o+(t[e*2]-o)*d,t[e*2+1]=s+(t[e*2+1]-s)*d}return{...n,positions:t}},a=[];for(let e=0;e+1<n.ringOffsets.length;e++)a.push(...G(n.positions,n.ringOffsets[e],n.ringOffsets[e+1]));return{getSet:i,world:t,triangleRows:a}},build(e,t,n){let r=t.getSet(n.gap),i=N(e,`coverage`,r),a=r.positions.length/2,o=e.allocate(`segment-flags`,`uint32`,a),s=e.allocate(`invalid-count`,`uint32`,1),c=e.allocate(`is-valid`,`uint32`,1),l=e.allocate(`gap-width`,`float32`,4);e.graph.add(new H({...F(i),parameters:l.view,output:{segmentFlags:o.view,status:{invalidCount:s.view},isValid:c.view}}));let u=j(e,`coverage`,i.positions,i.ringOffsets,{closed:!0,flags:o}),d=e.upload(`cell-fill`,`float32x2`,K(r.positions,t.triangleRows)),f=P(e,`cell-fill`,d.length);return{outputs:{...u,flagged:u.segments,fill:d,fillMesh:f},update:e=>{let n=t.getSet(e.gap);i.positions.write(n.positions),d.write(K(n.positions,t.triangleRows)),l.write(R({gapWidth:e.gapWidth}))}}},interaction:{kind:`reach`,hint:`move beside an edge`,rest:({inputs:e,values:t,toUnit:n})=>{let r=e.getSet(t.gap),i={positions:r.positions,lineOffsets:r.ringOffsets};return n(A(i,E(L,`gapWidth`).default/2,(e,t)=>k(i,e,t,!0)?.distance??0,e.world,.001))},resolve(e,{inputs:t,values:n}){let r=t.getSet(n.gap),i=k({positions:r.positions,lineOffsets:r.ringOffsets},e.x,e.y,!0);if(!i)return{};let a=T(E(L,`gapWidth`),i.distance*2);return{values:{gapWidth:a},glyphs:[{shape:`ring`,at:[e.x,e.y],radius:a/2},{shape:`dot`,at:i.point}],readout:`gapWidth ${a.toFixed(3)} · gap ${n.gap.toFixed(3)}`}}},draw:({outputs:e,theme:t})=>[{layer:`polygons`,context:!0,positions:e.fill,indices:e.fillMesh.indices,owners:e.fillMesh.owners,values:e.fillMesh.values,triangleCount:e.fillMesh.triangleCount,color:[...t.ink,1],alpha:.2},{layer:`segments`,context:!0,nodes:e.points,edges:e.edges,radius:1.5},{layer:`lines`,segments:e.flagged,color:M,radius:3.5},{layer:`text`,...I({corner:`bottom-left`,entries:[{label:`cell (gaps between cells show the background)`,swatch:t.muted},{label:`edge flagged by GPUCoverageValidity`,swatch:[M[0],M[1],M[2]]}]})}]});export{q as default};
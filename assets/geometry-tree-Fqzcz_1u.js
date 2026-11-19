import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as t}from"./kernel-C3tfQOn4.js";import{E as n,f as r,y as i}from"./gpu-contributor-utils-BIwLQON5.js";import{a}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as o}from"./exact-orientation-wgsl-B_jI-HEc.js";import{i as s}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{t as c}from"./validate-outline-output-passes-DIcqH6Pw.js";import{n as l}from"./refinement-storage-validation-D9BhQGNW.js";import{A as u}from"./pointer-overlay-Qh8qflcA.js";import{c as d,s as f}from"./hydrography-N8loF-Ds.js";var p=`fn normalizedPoint(vertex: u32) -> vec2f { return vec2f(normalized[normalizedOffset + vertex * 2u], normalized[normalizedOffset + vertex * 2u + 1u]); }
fn comparePoint(left: vec2f, right: vec2f) -> i32 {
  if (left.x != right.x) { return select(1, -1, left.x < right.x); }
  if (left.y != right.y) { return select(1, -1, left.y < right.y); }
  return 0;
}
// Iterative lexicographic tree comparison: no recursion, hashes or host traversal.
fn compareNodes(leftRoot: u32, rightRoot: u32) -> i32 {
  var leftStack: array<u32, 64>; var rightStack: array<u32, 64>; var childStack: array<u32, 64>;
  var depth = 0u; var left = leftRoot; var right = rightRoot;
  loop {
    let leftType = field(left, 0u); let rightType = field(right, 0u);
    if (leftType != rightType) { return select(1, -1, leftType < rightType); }
    let leftFirst = field(left, 1u); let rightFirst = field(right, 1u);
    let leftCount = field(left, 2u); let rightCount = field(right, 2u);
    if (isLeaf(leftType)) {
      if (leftCount != rightCount) { return select(1, -1, leftCount < rightCount); }
      for (var vertex = 0u; vertex < min(leftCount, rightCount); vertex++) {
        let comparison = comparePoint(normalizedPoint(leftFirst + vertex), normalizedPoint(rightFirst + vertex));
        if (comparison != 0) { return comparison; }
      }
      if (leftCount != rightCount) { return select(1, -1, leftCount < rightCount); }
    } else if (min(leftCount, rightCount) > 0u) {
      leftStack[depth] = left; rightStack[depth] = right; childStack[depth] = 1u; depth++;
      left = ordered[orderedOffset + leftFirst]; right = ordered[orderedOffset + rightFirst]; continue;
    } else if (leftCount != rightCount) { return select(1, -1, leftCount < rightCount); }
    var advanced = false;
    loop {
      if (depth == 0u) { return 0; }
      let parentLeft = leftStack[depth - 1u]; let parentRight = rightStack[depth - 1u];
      let nextChild = childStack[depth - 1u];
      let countLeft = field(parentLeft, 2u); let countRight = field(parentRight, 2u);
      if (nextChild < min(countLeft, countRight)) {
        childStack[depth - 1u]++; left = ordered[orderedOffset + field(parentLeft, 1u) + nextChild]; right = ordered[orderedOffset + field(parentRight, 1u) + nextChild]; advanced = true; break;
      }
      if (countLeft != countRight) { return select(1, -1, countLeft < countRight); }
      depth--;
    }
    if (!advanced) { return 0; }
  }
  return 0;
}`,m=64,h=class{id;props;constructor(e){if(this.id=e.id??`normalize-geometry`,this.props=e,r(this.id,`maximumNestingDepth`,e.maximumNestingDepth??m,1,m),n(e.positions,[`float32x2`],this.id),n(e.nodes,[`uint32x4`],this.id),n(e.children,[`uint32`],this.id),n(e.output.positions,[`float32x2`],this.id),n(e.output.children,[`uint32`],this.id),n(e.output.status,[`uint32`],this.id),e.positions.length!==e.output.positions.length||e.children.length!==e.output.children.length||e.nodes.length!==e.output.status.length)throw Error(`Canonical tree output size mismatch`);c(this.id,e.output,[[`positions`,`status`],[`children`,`status`]]),i(this.id,Object.values(e.output),[e.positions,e.nodes,e.children])}getCommandNodes(n){let{props:r,id:i}=this;a(n,i,Object.fromEntries([r.positions,r.nodes,r.children,...Object.values(r.output)].map((e,t)=>[`binding.${t}`,e]))),l(n,i,{nodeScratch:r.nodes.length*4,childScratch:r.children.length*4});let c=t(n,`${i}-hole-flags`,`uint32`,Math.max(r.nodes.length,1));return[e(n,{id:`${i}-copy-children`,operation:`GPUNormalizeGeometry`,variant:`copy-children`,invocationCount:r.children.length,bindings:[{name:`children`,view:r.children,type:`u32`,access:`read`},{name:`ordered`,view:r.output.children,type:`u32`,access:`read_write`}],body:`ordered[orderedOffset + index] = children[childrenOffset + index];`}),e(n,{id:`${i}-copy-positions`,operation:`GPUNormalizeGeometry`,variant:`copy-positions`,invocationCount:r.positions.length*2,bindings:[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`normalized`,view:r.output.positions,type:`f32`,access:`read_write`}],body:`normalized[normalizedOffset + index] = positions[positionsOffset + index];`}),e(n,{id:`${i}-hole-init`,operation:`GPUNormalizeGeometry`,variant:`hole-init`,invocationCount:r.nodes.length,bindings:[{name:`holeFlags`,view:c,type:`u32`,access:`read_write`}],body:`holeFlags[holeFlagsOffset + index] = 0u;`}),e(n,{id:`${i}-hole-mark`,operation:`GPUNormalizeGeometry`,variant:`hole-mark`,invocationCount:r.nodes.length,bindings:[{name:`nodes`,view:r.nodes,type:`u32`,access:`read`},{name:`children`,view:r.children,type:`u32`,access:`read`},{name:`holeFlags`,view:c,type:`atomic<u32>`,access:`read_write`}],body:`if (nodes[nodesOffset + index * 4u] != 5u) { return; }
let first = nodes[nodesOffset + index * 4u + 1u]; let count = nodes[nodesOffset + index * 4u + 2u];
let limit = min(count, ${r.children.length}u - min(first, ${r.children.length}u));
for (var child = 1u; child < limit; child++) {
  let holeNode = children[childrenOffset + first + child];
  if (holeNode < index) { atomicStore(&holeFlags[holeFlagsOffset + holeNode], 1u); }
}`}),e(n,{id:`${i}-leaves`,operation:`GPUNormalizeGeometry`,variant:`leaves`,invocationCount:r.nodes.length,bindings:[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`nodes`,view:r.nodes,type:`u32`,access:`read`},{name:`holeFlags`,view:c,type:`u32`,access:`read`},{name:`normalized`,view:r.output.positions,type:`f32`,access:`read_write`},{name:`status`,view:r.output.status,type:`u32`,access:`read_write`}],declarations:`${o}
fn field(node: u32, component: u32) -> u32 { return nodes[nodesOffset + node * 4u + component]; }
fn isLeaf(kind: u32) -> bool { return kind == 0u || kind == 2u || kind == 3u; }
${s(`point`,`positions`)}
fn comparePoint(left: vec2f, right: vec2f) -> i32 {
  if (left.x != right.x) { return select(1, -1, left.x < right.x); }
  if (left.y != right.y) { return select(1, -1, left.y < right.y); }
  return 0;
}`,body:`let node = index;
let kind = field(node, 0u); let first = field(node, 1u); let count = field(node, 2u);
if (kind > 7u) { status[statusOffset + node] = 2u; return; }
if (!isLeaf(kind)) { return; }
status[statusOffset + node] = 0u;
{
    if (first > ${r.positions.length}u || count > ${r.positions.length}u - first || (kind == 0u && count > 1u)) { status[statusOffset + node] = 2u; return; }
    for (var vertex = 0u; vertex < count; vertex++) { if (any((bitcast<vec2u>(point(first + vertex)) & vec2u(0x7f800000u)) == vec2u(0x7f800000u))) { status[statusOffset + node] = 1u; } }
    if (status[statusOffset + node] != 0u || count < 2u) { return; }
    let closed = all(point(first) == point(first + count - 1u));
    var rotation = 0u; var reverse = false;
    if (closed || kind == 3u) {
      let uniqueCount = count - select(0u, 1u, closed);
      for (var vertex = 0u; vertex < uniqueCount; vertex++) {
        if (comparePoint(point(first + vertex), point(first + rotation)) < 0) { rotation = vertex; }
      }
      // The extreme-vertex predicate avoids shoelace cancellation on translated rings.
      let center = point(first + rotation);
      var previous = center; var next = center;
      for (var step = 1u; step < uniqueCount; step++) {
        if (all(previous == center)) { previous = point(first + (rotation + uniqueCount - step) % uniqueCount); }
        if (all(next == center)) { next = point(first + (rotation + step) % uniqueCount); }
        if (any(previous != center) && any(next != center)) { break; }
      }
      var winding = 0;
      if (any(previous != center) && any(next != center)) { winding = orientSign(previous, center, next); }
      if (winding == 2) { status[statusOffset + node] = 2u; return; }
      // Holes are the non-first children of polygons, marked once by the hole-mark kernel.
      let hole = kind == 3u && holeFlags[holeFlagsOffset + node] != 0u;
      reverse = select(winding > 0, winding < 0, hole);
      for (var vertex = 0u; vertex < count; vertex++) {
        let offset = vertex % uniqueCount;
        let source = (rotation + select(offset, uniqueCount - offset, reverse)) % uniqueCount;
        let value = point(first + source); normalized[normalizedOffset + (first + vertex) * 2u] = value.x; normalized[normalizedOffset + (first + vertex) * 2u + 1u] = value.y;
      }
    } else {
      for (var vertex = 0u; vertex < count; vertex++) { let comparison = comparePoint(point(first + vertex), point(first + count - 1u - vertex)); if (comparison != 0) { reverse = comparison > 0; break; } }
      if (reverse) { for (var vertex = 0u; vertex < count; vertex++) { let value = point(first + count - 1u - vertex); normalized[normalizedOffset + (first + vertex) * 2u] = value.x; normalized[normalizedOffset + (first + vertex) * 2u + 1u] = value.y; } }
    }
}`}),...this._getCollectionNodes(n)]}_getCollectionNodes(n){let{props:r,id:i}=this,a=r.maximumNestingDepth??64,o=r.nodes.length,s=r.children.length,c=(e,r)=>t(n,`${i}-${e}`,`uint32`,Math.max(r,1)),l=c(`rounds`,o),u=c(`heights`,o),d=c(`scratch-children`,s),f=c(`slot-parents`,s),m=c(`round-state`,1),h=`fn field(node: u32, component: u32) -> u32 { return nodes[nodesOffset + node * 4u + component]; }
fn isLeaf(kind: u32) -> bool { return kind == 0u || kind == 2u || kind == 3u; }`,g=[e(n,{id:`${i}-round-init`,operation:`GPUNormalizeGeometry`,variant:`round-init`,invocationCount:o,bindings:[{name:`nodes`,view:r.nodes,type:`u32`,access:`read`},{name:`rounds`,view:l,type:`u32`,access:`read_write`},{name:`heights`,view:u,type:`u32`,access:`read_write`},{name:`roundState`,view:m,type:`u32`,access:`read_write`}],declarations:h,body:`if (index == 0u) { roundState[roundStateOffset] = 2u; }
let kind = field(index, 0u);
let finished = kind > 7u || isLeaf(kind);
rounds[roundsOffset + index] = select(0u, 1u, finished);
heights[heightsOffset + index] = select(0u, 1u, finished);`})];s>0&&g.push(e(n,{id:`${i}-slot-parent-init`,operation:`GPUNormalizeGeometry`,variant:`slot-parent-init`,invocationCount:s,bindings:[{name:`slotParents`,view:f,type:`u32`,access:`read_write`}],body:`slotParents[slotParentsOffset + index] = 0xffffffffu;`}),e(n,{id:`${i}-slot-parent-mark`,operation:`GPUNormalizeGeometry`,variant:`slot-parent-mark`,invocationCount:o,bindings:[{name:`nodes`,view:r.nodes,type:`u32`,access:`read`},{name:`slotParents`,view:f,type:`u32`,access:`read_write`}],declarations:h,body:`let kind = field(index, 0u);
if (kind > 7u || isLeaf(kind)) { return; }
let first = field(index, 1u); let count = field(index, 2u);
if (first > ${s}u || count > ${s}u - first) { return; }
for (var child = 0u; child < count; child++) { slotParents[slotParentsOffset + first + child] = index; }`}));let _=a+1;for(let t=2;t<=_;t++)g.push(e(n,{id:`${i}-round-${t}-nodes`,operation:`GPUNormalizeGeometry`,variant:`collection-round-nodes`,invocationCount:o,bindings:[{name:`nodes`,view:r.nodes,type:`u32`,access:`read`},{name:`ordered`,view:r.output.children,type:`u32`,access:`read_write`},{name:`scratch`,view:d,type:`u32`,access:`read`},{name:`rounds`,view:l,type:`atomic<u32>`,access:`read_write`},{name:`roundState`,view:m,type:`u32`,access:`read`},{name:`heights`,view:u,type:`u32`,access:`read_write`},{name:`status`,view:r.output.status,type:`u32`,access:`read_write`}],declarations:`${h}
const MAXIMUM_DEPTH: u32 = ${a}u;`,body:`let ROUND = roundState[roundStateOffset];
let kind = field(index, 0u); let first = field(index, 1u); let count = field(index, 2u);
if (kind > 7u || isLeaf(kind)) { return; }
let finishedRound = atomicLoad(&rounds[roundsOffset + index]);
if (finishedRound != 0u) {
  if (finishedRound == ROUND - 1u && status[statusOffset + index] == 0u) {
    for (var child = select(0u, 1u, kind == 5u); child < count; child++) {
      ordered[orderedOffset + first + child] = scratch[scratchOffset + first + child];
    }
  }
  return;
}
let valid = first <= ${s}u && count <= ${s}u - first;
var ready = true;
if (valid) {
  for (var child = 0u; child < count; child++) {
    let childNode = ordered[orderedOffset + first + child];
    if (childNode < index) {
      let childRound = atomicLoad(&rounds[roundsOffset + childNode]);
      if (childRound == 0u || childRound >= ROUND) { ready = false; }
    }
  }
}
if (!ready) { return; }
var failed = !valid;
var height = 1u;
if (valid) {
  for (var child = 0u; child < count; child++) {
    let childNode = ordered[orderedOffset + first + child];
    if (childNode >= index || status[statusOffset + childNode] != 0u) { failed = true; } else { height = max(height, heights[heightsOffset + childNode] + 1u); }
  }
}
if (height > MAXIMUM_DEPTH) { failed = true; }
status[statusOffset + index] = select(0u, 2u, failed);
heights[heightsOffset + index] = height;
atomicStore(&rounds[roundsOffset + index], ROUND);`})),s>0&&g.push(e(n,{id:`${i}-round-${t}-rank`,operation:`GPUNormalizeGeometry`,variant:`collection-round-rank`,invocationCount:s,bindings:[{name:`nodes`,view:r.nodes,type:`u32`,access:`read`},{name:`normalized`,view:r.output.positions,type:`f32`,access:`read`},{name:`ordered`,view:r.output.children,type:`u32`,access:`read`},{name:`scratch`,view:d,type:`u32`,access:`read_write`},{name:`slotParents`,view:f,type:`u32`,access:`read`},{name:`rounds`,view:l,type:`u32`,access:`read`},{name:`roundState`,view:m,type:`u32`,access:`read`},{name:`status`,view:r.output.status,type:`u32`,access:`read`}],declarations:`${h}
${p}`,body:`let ROUND = roundState[roundStateOffset];
let parent = slotParents[slotParentsOffset + index];
if (parent == 0xffffffffu || rounds[roundsOffset + parent] != ROUND || status[statusOffset + parent] != 0u) { return; }
let first = field(parent, 1u); let count = field(parent, 2u);
let start = select(0u, 1u, field(parent, 0u) == 5u);
let position = index - first;
if (position < start) { return; }
let selected = ordered[orderedOffset + index];
var rank = 0u;
for (var other = start; other < count; other++) {
  if (other == position) { continue; }
  let comparison = compareNodes(ordered[orderedOffset + first + other], selected);
  if (comparison > 0 || (comparison == 0 && other < position)) { rank++; }
}
scratch[scratchOffset + first + start + rank] = selected;`})),g.push(e(n,{id:`${i}-round-${t}-advance`,operation:`GPUNormalizeGeometry`,variant:`collection-round-advance`,invocationCount:1,bindings:[{name:`roundState`,view:m,type:`u32`,access:`read_write`}],body:`roundState[roundStateOffset] += 1u;`}));return g.push(e(n,{id:`${i}-collections-finish`,operation:`GPUNormalizeGeometry`,variant:`collection-finish`,invocationCount:o,bindings:[{name:`nodes`,view:r.nodes,type:`u32`,access:`read`},{name:`ordered`,view:r.output.children,type:`u32`,access:`read_write`},{name:`scratch`,view:d,type:`u32`,access:`read`},{name:`rounds`,view:l,type:`u32`,access:`read`},{name:`roundState`,view:m,type:`u32`,access:`read`},{name:`status`,view:r.output.status,type:`u32`,access:`read_write`}],declarations:h,body:`let kind = field(index, 0u); let first = field(index, 1u); let count = field(index, 2u);
if (kind > 7u || isLeaf(kind)) { return; }
let finishedRound = rounds[roundsOffset + index];
if (finishedRound == 0u) { status[statusOffset + index] = 2u; return; }
if (finishedRound + 1u == roundState[roundStateOffset] && status[statusOffset + index] == 0u) {
  for (var child = select(0u, 1u, kind == 5u); child < count; child++) {
    ordered[orderedOffset + first + child] = scratch[scratchOffset + first + child];
  }
}`})),g}},g=0,_=2,v=3,y=7,b=[v,_,g],x=4,S=3;function C(e,t){let n=u(7),r=[],i=[],a=[],o=[];for(let s=0;s<12;s++){let c=b[s%b.length],l=s%x,u=Math.floor(s/x),p=(l+.5)/x*e.width,m=(u+.5)/S*e.height,h=.11*(.8+.4*n()),y=r.length/2,C=[];if(c===v){C.push(...d(s,p,m,h*1.3,7));let e=(s+t.start)%7,n=C.map((t,n)=>C[(n+e)%7]);C.splice(0,C.length,...n,n[0])}else if(c===_){let e=f(s,8,{width:h*2.5,height:h*1.8});C.push(...e.map(([e,t])=>[p+e-h*1.25,m+t-h*.9]))}else C.push([p,m]);let w=(t.displacement??0)*s;for(let[e,t]of C)r.push(e+w,t),o.push(s);if(i.push(c,y,C.length,0),c!==g)for(let e=0;e+1<C.length;e++)a.push(y+e,y+e+1)}let s=Array.from({length:12},(e,t)=>t),c=u(t.shuffleSeed+11);for(let e=s.length-1;e>0;e--){let t=Math.floor(c()*(e+1));[s[e],s[t]]=[s[t],s[e]]}return i.push(y,0,12,0),{leafCount:12,positions:Float32Array.from(r),nodes:Uint32Array.from(i),children:Uint32Array.from(s),rootId:12,edges:Uint32Array.from(a),leafOfVertex:Uint32Array.from(o)}}export{h as n,C as t};
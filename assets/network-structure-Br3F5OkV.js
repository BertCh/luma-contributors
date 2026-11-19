import{r as e,s as t}from"./kernel-C3tfQOn4.js";import{A as n,E as r,S as i,T as a,f as o,k as s,p as c,v as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{a as f,o as p}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as h}from"./define-run-B-brkCe4.js";import{_ as g,a as _,f as v,h as y,i as b,m as x,p as S,r as C,v as w,y as T}from"./roads-DzovamHj.js";import{da as E}from"./entry-preview-D70TdSbK.js";var D=4096,O=4294967294,k=`GPUNetworkStronglyConnectedComponents`,A=256,j=4294967295,M=0,N=1,P=2,F=3,I=4,L=4,R=1,z=2,B=4,V=class{props;id;nodeCount;edgeCount;maximumIterations;estimatedCommandNodeCount;constructor(e){this.props=e,this.id=e.id??`network-strongly-connected-components`;let{id:t}=this,i=n(t,e.topology);this.nodeCount=i.nodeCount,this.edgeCount=i.edgeCount,this.maximumIterations=e.maximumIterations??256,o(t,`nodeCount`,this.nodeCount,1,O),o(t,`edgeCount`,this.edgeCount,0,O),o(t,`maximumIterations`,this.maximumIterations,1,D),this.estimatedCommandNodeCount=2+6*this.maximumIterations,c(t,this.estimatedCommandNodeCount);let l=[e.output.strongComponents,...Object.values(e.output.status)];for(let[n,i]of Object.entries({strongComponents:e.output.strongComponents,converged:e.output.status.converged,iterationCount:e.output.status.iterationCount,iterationLimitReached:e.output.status.iterationLimitReached})){r(i,[`uint32`],`${t} output.${n}`);let e=n===`strongComponents`?this.nodeCount:1;a(t,i.length===e,`output.${n} must hold exactly ${e} uint32 rows`)}a(t,new Set(l.map(e=>e.buffer)).size===l.length,`output strongComponents and status must use distinct buffers`),d(t,l,s(e.topology))}getCommandNodes(n){let{id:r,props:i,nodeCount:a,edgeCount:o}=this,c=i.output.strongComponents;u(r,n,[...s(i.topology),c,...Object.values(i.output.status)]);for(let[e,a]of Object.entries({offsets:i.topology.offsets,neighbors:i.topology.neighbors,strongComponents:c,...i.output.status}))l(n.device,r,`topology/output.${e}`,t(a).size,`reduce nodeCount or edgeCount to fit the device storage binding limit`);l(n.device,r,`color/flags scratch`,a*4,`reduce nodeCount to fit the device storage binding limit`);let d=[],m=e(n,`${r}-color`,`uint32`,a),h=e(n,`${r}-flags`,`uint32`,a),g=e(n,`${r}-control`,`uint32`,L),_=f(n,{id:`${r}-gate`,operation:k,shapes:[[a,A],[a,A],[a,A],[a,A]]}),v=_.armWGSL+`
`+[M,N,P,F].map(e=>`if (next != ${e}u) { dispatch[dispatchOffset + ${e*3}u] = 0u; }`).join(`
`),y=`const VERTEX_COUNT: u32 = ${a}u;
const CAPACITY: u32 = ${o}u;
const INVALID: u32 = ${j}u;
const FLAG_HAS_INPUT: u32 = ${R}u;
const FLAG_HAS_OUTPUT: u32 = ${z}u;
const FLAG_MARKED: u32 = ${B}u;`,b=[{name:`offsets`,view:i.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.topology.neighbors,type:`u32`,access:`read`}],x={name:`output`,view:c,type:`atomic<u32>`,access:`read_write`},S={name:`color`,view:m,type:`atomic<u32>`,access:`read_write`},C={name:`flags`,view:h,type:`atomic<u32>`,access:`read_write`},w={name:`control`,view:g,type:`atomic<u32>`,access:`read_write`},T=`let first = min(offsets[offsetsOffset + index], CAPACITY);
  let last = min(offsets[offsetsOffset + index + 1u], CAPACITY);`,E=`if (atomicLoad(&output[outputOffset + index]) != INVALID) { return; }`,D=(e,t,r,i,o)=>{let s=r===void 0?void 0:_.getGate(r);d.push(p(n,{id:e,operation:k,variant:t,invocationCount:a,workgroupSize:A,bindings:i,declarations:y,body:o,...s&&{condition:{...s.condition,id:`${e}-gate`},extraResources:s.extraResources}}))};D(`${r}-initialize`,`initialize`,void 0,[x,S,C,w,{name:`dispatch`,view:_.view,type:`u32`,access:`read_write`}],`atomicStore(&output[outputOffset + index], INVALID);
  atomicStore(&color[colorOffset + index], index);
  atomicStore(&flags[flagsOffset + index], 0u);
  if (index == 0u) {
    atomicStore(&control[controlOffset], ${M}u);
    atomicStore(&control[controlOffset + 1u], 0u);
    atomicStore(&control[controlOffset + 2u], 0u);
    atomicStore(&control[controlOffset + 3u], 0u);
    let next = ${M}u;
    ${v}
  }`);for(let e=0;e<this.maximumIterations;e++){let t=`${r}-iteration-${e}`;D(`${t}-trim-mark`,`trim-mark`,M,[...b,x,C],`${E}
  ${T}
  var hasOutput = false;
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == index) { continue; }
    if (atomicLoad(&output[outputOffset + neighbor]) != INVALID) { continue; }
    hasOutput = true;
    atomicOr(&flags[flagsOffset + neighbor], FLAG_HAS_INPUT);
  }
  if (hasOutput) { atomicOr(&flags[flagsOffset + index], FLAG_HAS_OUTPUT); }`),D(`${t}-trim-remove`,`trim-remove`,M,[x,C,w],`${E}
  let previous = atomicAnd(&flags[flagsOffset + index], ~(FLAG_HAS_INPUT | FLAG_HAS_OUTPUT));
  if ((previous & (FLAG_HAS_INPUT | FLAG_HAS_OUTPUT)) != (FLAG_HAS_INPUT | FLAG_HAS_OUTPUT)) {
    atomicStore(&output[outputOffset + index], index);
    atomicStore(&control[controlOffset + 1u], 1u);
  } else {
    atomicAdd(&control[controlOffset + 2u], 1u);
  }`),D(`${t}-color`,`color`,N,[...b,x,S,w],`${E}
  ${T}
  let vertexColor = atomicLoad(&color[colorOffset + index]);
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == index) { continue; }
    if (atomicLoad(&output[outputOffset + neighbor]) != INVALID) { continue; }
    if (atomicMin(&color[colorOffset + neighbor], vertexColor) > vertexColor) {
      atomicStore(&control[controlOffset + 1u], 1u);
    }
  }`),D(`${t}-back`,`back`,P,[...b,x,S,C,w],`${E}
  if ((atomicLoad(&flags[flagsOffset + index]) & FLAG_MARKED) != 0u) { return; }
  ${T}
  let vertexColor = atomicLoad(&color[colorOffset + index]);
  var reachesRoot = vertexColor == index;
  for (var slot = first; slot < last && !reachesRoot; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == index) { continue; }
    reachesRoot = (atomicLoad(&flags[flagsOffset + neighbor]) & FLAG_MARKED) != 0u &&
      atomicLoad(&color[colorOffset + neighbor]) == vertexColor;
  }
  if (reachesRoot) {
    atomicOr(&flags[flagsOffset + index], FLAG_MARKED);
    atomicStore(&control[controlOffset + 1u], 1u);
  }`),D(`${t}-finalize`,`finalize`,F,[x,S,C,w],`${E}
  if ((atomicLoad(&flags[flagsOffset + index]) & FLAG_MARKED) != 0u) {
    atomicStore(&output[outputOffset + index], atomicLoad(&color[colorOffset + index]));
    atomicStore(&flags[flagsOffset + index], 0u);
  } else {
    atomicStore(&color[colorOffset + index], index);
    atomicAdd(&control[controlOffset + 2u], 1u);
  }`),d.push(p(n,{id:`${t}-controller`,operation:k,variant:`controller`,invocationCount:1,workgroupSize:2,bindings:[w,_.binding],body:`let phase = atomicLoad(&control[controlOffset]);
  if (phase == ${I}u) { return; }
  atomicAdd(&control[controlOffset + 3u], 1u);
  let changed = atomicLoad(&control[controlOffset + 1u]);
  let remaining = atomicLoad(&control[controlOffset + 2u]);
  var next = phase;
  if (phase == ${M}u) {
    if (remaining == 0u) { next = ${I}u; } else if (changed == 0u) { next = ${N}u; }
  } else if (phase == ${N}u) {
    if (changed == 0u) { next = ${P}u; }
  } else if (phase == ${P}u) {
    if (changed == 0u) { next = ${F}u; }
  } else if (phase == ${F}u) {
    next = select(${M}u, ${I}u, remaining == 0u);
  }
  atomicStore(&control[controlOffset], next);
  atomicStore(&control[controlOffset + 1u], 0u);
  atomicStore(&control[controlOffset + 2u], 0u);
  ${v}`}))}return d.push(p(n,{id:`${r}-publish`,operation:k,variant:`publish`,invocationCount:a,workgroupSize:A,bindings:[x,w,{name:`converged`,view:i.output.status.converged,type:`u32`,access:`read_write`},{name:`iterationCount`,view:i.output.status.iterationCount,type:`u32`,access:`read_write`},{name:`iterationLimitReached`,view:i.output.status.iterationLimitReached,type:`u32`,access:`read_write`}],body:`let finished = atomicLoad(&control[controlOffset]) == ${I}u;
  if (!finished) { atomicStore(&output[outputOffset + index], ${j}u); }
  if (index == 0u) {
    converged[convergedOffset] = select(0u, 1u, finished);
    iterationCount[iterationCountOffset] = atomicLoad(&control[controlOffset + 3u]);
    iterationLimitReached[iterationLimitReachedOffset] = select(1u, 0u, finished);
  }`})),d}},H=4294967294;function U(e,t){let n=Math.max(1,Math.ceil(Math.log2(e))),r=Math.ceil(Math.log2(Math.max(t,2))),i=t+e*n,a=Math.ceil(Math.log2(Math.max(i,2))),o=n*(7+3*n),s=4*r,c=6*n,l=a*(7+3*a);return{treeLevels:n,edgeRounds:r,auxiliaryCount:i,componentRounds:a,estimatedCommandNodeCount:18+o+s+c+l}}function W(e,t,n,r,i){return new G(t,n,r,i).getCommandNodes(e)}var G=class{id;props;nodeCount;edgeCount;constructor(e,t,n,r){this.id=e,this.props=t,this.nodeCount=n,this.edgeCount=r}getCommandNodes(t){let{id:n,props:r,nodeCount:i,edgeCount:a}=this,{treeLevels:o,edgeRounds:s,auxiliaryCount:c,componentRounds:l}=U(i,a),u=[],d=(r,i)=>e(t,`${n}-${r}`,`uint32`,Math.max(i,1)),m=(e,t)=>({name:e,view:t,type:`u32`,access:`read`}),h=(e,t,n=!1)=>({name:e,view:t,type:n?`atomic<u32>`:`u32`,access:`read_write`}),g=`const VERTEX_COUNT: u32 = ${i}u;
const EDGE_COUNT: u32 = ${a}u;
const LEVEL_COUNT: u32 = ${o}u;
const NONE: u32 = 0xffffffffu;`,_=(e,r,i,a,o=``,s)=>{u.push(p(t,{id:`${n}-${e}`,operation:`GPUNetworkStructure`,variant:e.replace(/-\d+/g,``),invocationCount:r,bindings:i,declarations:`${g}\n${o}`,body:a,...s}))},v=(e,r,i,a,o,s)=>{let c=d(`${e}-pending`,1),l=f(t,{id:`${n}-${e}-dispatch`,operation:`GPUNetworkStructure`,shapes:[[r]]});_(`${e}-seed`,1,[h(`pending`,c),...s?[m(`outerDispatch`,s)]:[]],s?`pending[pendingOffset] = select(0u, 1u, outerDispatch[outerDispatchOffset] != 0u);`:`pending[pendingOffset] = 1u;`);for(let t=0;t<i;t++)_(`${e}-gate-${t}`,1,[h(`pending`,c),l.binding],`
if (pending[pendingOffset] != 0u) { ${l.armWGSL} }
else { ${l.disarmWGSL} }
pending[pendingOffset] = 0u;`),_(`${e}-jump-${t}`,r,[m(`parents`,a),h(`scratch`,o),h(`pending`,c,!0)],`
let parent = parents[parentsOffset + index];
let ancestor = parents[parentsOffset + parent];
scratch[scratchOffset + index] = ancestor;
if (ancestor != parent) { atomicOr(&pending[pendingOffset], 1u); }`,``,l.getGate(0)),_(`${e}-copy-${t}`,r,[m(`scratch`,o),h(`parents`,a)],`parents[parentsOffset + index] = scratch[scratchOffset + index];`,``,l.getGate(0))},y=d(`sources`,a*2),b=d(`tree`,a),x=d(`components`,i),S=d(`parents`,i),C=d(`vertex-scratch`,i),w=d(`choices`,i),T=r.reverseEdgeSlots,E=[m(`offsets`,r.topology.offsets),m(`neighbors`,r.topology.neighbors)];_(`initialize-slots`,a,[m(`neighbors`,r.topology.neighbors),h(`sources`,y),h(`tree`,b)],`
sources[sourcesOffset + index * 2u] = NONE;
sources[sourcesOffset + index * 2u + 1u] = neighbors[neighborsOffset + index];
tree[treeOffset + index] = 0u;`),_(`initialize-vertices`,i,[m(`offsets`,r.topology.offsets),h(`sources`,y),h(`components`,x)],`
components[componentsOffset + index] = index;
let first = min(offsets[offsetsOffset + index], EDGE_COUNT);
let last = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
for (var edge = first; edge < last; edge++) { sources[sourcesOffset + edge * 2u] = index; }`);let D=[m(`sources`,y),m(`reverse`,T)],O=`let source = sources[sourcesOffset + index * 2u];
let destination = sources[sourcesOffset + index * 2u + 1u];
let twin = reverse[reverseOffset + index];
if (source >= VERTEX_COUNT || destination >= VERTEX_COUNT || twin >= EDGE_COUNT || index > twin) { return; }`,k=d(`forest-pending`,1),A=f(t,{id:`${n}-forest-dispatch`,operation:`GPUNetworkStructure`,shapes:[[i],[a]]});_(`forest-seed`,1,[h(`pending`,k)],`pending[pendingOffset] = 1u;`);for(let e=0;e<o;e++)_(`forest-gate-${e}`,1,[h(`pending`,k),A.binding],`
if (pending[pendingOffset] != 0u) { ${A.armWGSL} }
else { ${A.disarmWGSL} }
pending[pendingOffset] = 0u;`),_(`forest-clear-${e}`,i,[m(`components`,x),h(`parents`,S),h(`choices`,w)],`
parents[parentsOffset + index] = components[componentsOffset + index];
choices[choicesOffset + index] = NONE;`,``,A.getGate(0)),_(`forest-choose-${e}`,a,[...D,m(`components`,x),h(`choices`,w,!0)],`
${O}
let first = components[componentsOffset + source];
let second = components[componentsOffset + destination];
if (first != second) {
  atomicMin(&choices[choicesOffset + first], index);
  atomicMin(&choices[choicesOffset + second], index);
}`,``,A.getGate(1)),_(`forest-hook-${e}`,a,[...D,m(`components`,x),m(`choices`,w),h(`tree`,b),h(`pending`,k,!0)],`
${O}
let first = components[componentsOffset + source];
let second = components[componentsOffset + destination];
if (first != second && (choices[choicesOffset + first] == index || choices[choicesOffset + second] == index)) {
  tree[treeOffset + index] = 1u;
  tree[treeOffset + twin] = 1u;
  atomicOr(&pending[pendingOffset], 1u);
}`,``,A.getGate(1)),_(`forest-link-${e}`,i,[m(`components`,x),m(`choices`,w),m(`sources`,y),h(`parents`,S)],`
let root = components[componentsOffset + index];
var parent = root;
if (root == index) {
  let edge = choices[choicesOffset + index];
  if (edge != NONE) {
    let first = components[componentsOffset + sources[sourcesOffset + edge * 2u]];
    let second = components[componentsOffset + sources[sourcesOffset + edge * 2u + 1u]];
    let neighbor = select(first, second, first == index);
    if (choices[choicesOffset + neighbor] != edge || index > neighbor) { parent = neighbor; }
  }
}
parents[parentsOffset + index] = parent;`,``,A.getGate(0)),v(`forest-compress-${e}`,i,o,S,C,A.view),_(`forest-publish-${e}`,i,[m(`parents`,S),h(`components`,x)],`components[componentsOffset + index] = parents[parentsOffset + index];`,``,A.getGate(0));let j=d(`successor`,a),M=d(`tour-minimum`,a),N=d(`edge-scratch`,a*2),P=d(`tour-distance`,a);_(`tour-clear`,a,[h(`successor`,j),h(`minimum`,M)],`
successor[successorOffset + index] = index;
minimum[minimumOffset + index] = index;`),_(`tour-link`,i,[m(`offsets`,r.topology.offsets),m(`tree`,b),m(`reverse`,T),h(`successor`,j)],`
var first = NONE; var previous = NONE;
let begin = min(offsets[offsetsOffset + index], EDGE_COUNT);
let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
for (var edge = begin; edge < end; edge++) {
  if (tree[treeOffset + edge] == 0u) { continue; }
  if (first == NONE) { first = edge; }
  if (previous != NONE) { successor[successorOffset + reverse[reverseOffset + previous]] = edge; }
  previous = edge;
}
if (previous != NONE) { successor[successorOffset + reverse[reverseOffset + previous]] = first; }`);for(let e=0;e<s;e++)_(`tour-minimum-${e}`,a,[m(`successor`,j),m(`minimum`,M),h(`scratch`,N)],`
let next = successor[successorOffset + index];
scratch[scratchOffset + index * 2u] = successor[successorOffset + next];
scratch[scratchOffset + index * 2u + 1u] = min(minimum[minimumOffset + index], minimum[minimumOffset + next]);`),_(`tour-minimum-copy-${e}`,a,[h(`successor`,j),h(`minimum`,M),m(`scratch`,N)],`
successor[successorOffset + index] = scratch[scratchOffset + index * 2u];
minimum[minimumOffset + index] = scratch[scratchOffset + index * 2u + 1u];`);_(`tour-relink-clear`,a,[h(`successor`,j)],`successor[successorOffset + index] = index;`),_(`tour-relink`,i,[m(`offsets`,r.topology.offsets),m(`tree`,b),m(`reverse`,T),h(`successor`,j)],`
var first = NONE; var previous = NONE;
let begin = min(offsets[offsetsOffset + index], EDGE_COUNT);
let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
for (var edge = begin; edge < end; edge++) {
  if (tree[treeOffset + edge] == 0u) { continue; }
  if (first == NONE) { first = edge; }
  if (previous != NONE) { successor[successorOffset + reverse[reverseOffset + previous]] = edge; }
  previous = edge;
}
if (previous != NONE) { successor[successorOffset + reverse[reverseOffset + previous]] = first; }`),_(`tour-cut`,a,[h(`successor`,j),m(`minimum`,M),h(`distance`,P)],`
if (minimum[minimumOffset + index] == index) { successor[successorOffset + index] = index; distance[distanceOffset + index] = 0u; }
else { distance[distanceOffset + index] = 1u; }`);for(let e=0;e<s;e++)_(`tour-rank-${e}`,a,[m(`successor`,j),m(`distance`,P),h(`scratch`,N)],`
let next = successor[successorOffset + index];
scratch[scratchOffset + index * 2u] = successor[successorOffset + next];
scratch[scratchOffset + index * 2u + 1u] = distance[distanceOffset + index] + distance[distanceOffset + next];`),_(`tour-rank-copy-${e}`,a,[h(`successor`,j),h(`distance`,P),m(`scratch`,N)],`
successor[successorOffset + index] = scratch[scratchOffset + index * 2u];
distance[distanceOffset + index] = scratch[scratchOffset + index * 2u + 1u];`);let F=d(`ancestors`,i*o),I=d(`parent-edges`,i),L=d(`depth`,i);_(`tree-clear`,i,[h(`ancestors`,F),h(`parentEdges`,I),h(`depth`,L)],`
ancestors[ancestorsOffset + index] = index;
parentEdges[parentEdgesOffset + index] = NONE;
depth[depthOffset + index] = 0u;`),_(`tree-orient`,a,[...D,m(`tree`,b),m(`distance`,P),h(`ancestors`,F),h(`parentEdges`,I),h(`depth`,L)],`
${O}
if (tree[treeOffset + index] == 0u) { return; }
var child = source; var parent = destination;
if (distance[distanceOffset + index] > distance[distanceOffset + twin]) { child = destination; parent = source; }
ancestors[ancestorsOffset + child] = parent;
parentEdges[parentEdgesOffset + child] = index;
depth[depthOffset + child] = 1u;`);let R=d(`level-state`,1);_(`tree-level-seed`,1,[h(`levelState`,R)],`levelState[levelStateOffset] = 1u;`);for(let e=1;e<o;e++)_(`tree-lift-${e}`,i,[h(`ancestors`,F),m(`levelState`,R)],`
let level = levelState[levelStateOffset];
let previous = (level - 1u) * VERTEX_COUNT;
let next = level * VERTEX_COUNT;
let parent = ancestors[ancestorsOffset + previous + index];
ancestors[ancestorsOffset + next + index] = ancestors[ancestorsOffset + previous + parent];`),_(`tree-level-advance-${e}`,1,[h(`levelState`,R)],`levelState[levelStateOffset] += 1u;`);_(`depth-seed`,i,[m(`ancestors`,F),h(`parents`,S)],`parents[parentsOffset + index] = ancestors[ancestorsOffset + index];`);let z=d(`depth-scratch`,i);for(let e=0;e<o;e++)_(`depth-rank-${e}`,i,[m(`parents`,S),m(`depth`,L),h(`scratch`,C),h(`depthScratch`,z)],`
let parent = parents[parentsOffset + index];
scratch[scratchOffset + index] = parents[parentsOffset + parent];
depthScratch[depthScratchOffset + index] = depth[depthOffset + index] + depth[depthOffset + parent];`),_(`depth-copy-${e}`,i,[h(`parents`,S),h(`depth`,L),m(`scratch`,C),m(`depthScratch`,z)],`
parents[parentsOffset + index] = scratch[scratchOffset + index];
depth[depthOffset + index] = depthScratch[depthScratchOffset + index];`);let B=d(`activeSegments`,i*o),V=d(`labels`,c),H=d(`auxiliary-parents`,c),W=d(`auxiliary-scratch`,c);_(`auxiliary-clear`,c,[h(`labels`,V)],`labels[labelsOffset + index] = index;`),_(`segments-clear`,i*o,[h(`activeSegments`,B)],`activeSegments[activeSegmentsOffset + index] = 0u;`),_(`segments-seed`,i,[m(`parentEdges`,I),h(`labels`,V)],`
let edge = parentEdges[parentEdgesOffset + index];
if (edge != NONE) { labels[labelsOffset + EDGE_COUNT + index] = edge; }`);let G=[...D,m(`tree`,b),m(`ancestors`,F),m(`depth`,L)],K=e=>`
${O}
if (source == destination || tree[treeOffset + index] != 0u) { return; }
var first = source; var second = destination;
if (depth[depthOffset + first] < depth[depthOffset + second]) { first = destination; second = source; }
let difference = depth[depthOffset + first] - depth[depthOffset + second];
for (var level = 0u; level < LEVEL_COUNT; level++) {
  if ((difference & (1u << level)) != 0u) {
    let segment = level * VERTEX_COUNT + first;
    ${e}
    first = ancestors[ancestorsOffset + segment];
  }
}
if (first == second) { return; }
for (var remaining = LEVEL_COUNT; remaining > 0u; remaining--) {
  let level = remaining - 1u;
  let firstAncestor = ancestors[ancestorsOffset + level * VERTEX_COUNT + first];
  let secondAncestor = ancestors[ancestorsOffset + level * VERTEX_COUNT + second];
  if (firstAncestor != secondAncestor) {
    { let segment = level * VERTEX_COUNT + first; ${e} }
    { let segment = level * VERTEX_COUNT + second; ${e} }
    first = firstAncestor; second = secondAncestor;
  }
}
{ let segment = first; ${e} }
{ let segment = second; ${e} }`;_(`cycles-activate`,a,[...G,h(`activeSegments`,B,!0)],K(`atomicStore(&activeSegments[activeSegmentsOffset + segment], 1u);`)),_(`segment-level-seed`,1,[h(`levelState`,R)],`levelState[levelStateOffset] = LEVEL_COUNT - 1u;`);for(let e=o-1;e>0;e--)_(`segments-activate-${e}`,i,[m(`ancestors`,F),h(`activeSegments`,B,!0),m(`levelState`,R)],`
let level = levelState[levelStateOffset];
let segment = level * VERTEX_COUNT + index;
if (atomicLoad(&activeSegments[activeSegmentsOffset + segment]) != 0u) {
  let half = (level - 1u) * VERTEX_COUNT;
  let middle = ancestors[ancestorsOffset + half + index];
  atomicStore(&activeSegments[activeSegmentsOffset + half + index], 1u);
  atomicStore(&activeSegments[activeSegmentsOffset + half + middle], 1u);
}`),_(`segment-level-retreat-${e}`,1,[h(`levelState`,R)],`levelState[levelStateOffset] -= 1u;`);let q=d(`auxiliary-pending`,1),J=f(t,{id:`${n}-auxiliary-dispatch`,operation:`GPUNetworkStructure`,shapes:[[c],[a],[i*o]]}),Y=(e,t)=>`{
let firstLabel = labels[labelsOffset + ${e}];
let secondLabel = labels[labelsOffset + ${t}];
if (firstLabel != secondLabel) {
  atomicMin(&choices[choicesOffset + firstLabel], secondLabel);
  atomicMin(&choices[choicesOffset + secondLabel], firstLabel);
  atomicOr(&pending[pendingOffset], 1u);
}}`;_(`auxiliary-seed`,1,[h(`pending`,q)],`pending[pendingOffset] = 1u;`);for(let e=0;e<l;e++){_(`auxiliary-gate-${e}`,1,[h(`pending`,q),J.binding],`
if (pending[pendingOffset] != 0u) { ${J.armWGSL} }
else { ${J.disarmWGSL} }
pending[pendingOffset] = 0u;`),_(`auxiliary-parents-${e}`,c,[h(`choices`,W)],`choices[choicesOffset + index] = NONE;`,``,J.getGate(0));let t=[m(`labels`,V),h(`choices`,W,!0),h(`pending`,q,!0)];_(`cycles-hook-${e}`,a,[...G,...t],K(Y(`index`,`EDGE_COUNT + segment`)),``,J.getGate(1)),_(`segments-hook-${e}`,i*o,[m(`activeSegments`,B),m(`ancestors`,F),m(`parentEdges`,I),...t],`
if (activeSegments[activeSegmentsOffset + index] == 0u) { return; }
let level = index / VERTEX_COUNT;
let vertex = index % VERTEX_COUNT;
if (level == 0u) {
  let edge = parentEdges[parentEdgesOffset + vertex];
  if (edge != NONE) { ${Y(`EDGE_COUNT + index`,`edge`)} }
} else {
  let half = (level - 1u) * VERTEX_COUNT;
  let middle = ancestors[ancestorsOffset + half + vertex];
  ${Y(`EDGE_COUNT + index`,`EDGE_COUNT + half + vertex`)}
  ${Y(`EDGE_COUNT + index`,`EDGE_COUNT + half + middle`)}
}`,``,J.getGate(2)),_(`auxiliary-link-${e}`,c,[m(`labels`,V),m(`choices`,W),h(`parents`,H)],`
let root = labels[labelsOffset + index];
var parent = root;
if (root == index) {
  let neighbor = choices[choicesOffset + index];
  if (neighbor != NONE && (choices[choicesOffset + neighbor] != index || index > neighbor)) { parent = neighbor; }
}
parents[parentsOffset + index] = parent;`,``,J.getGate(0)),v(`auxiliary-compress-${e}`,c,l,H,W,J.view),_(`auxiliary-publish-${e}`,c,[m(`parents`,H),h(`labels`,V)],`labels[labelsOffset + index] = parents[parentsOffset + index];`,``,J.getGate(0))}let X=r.output.biconnectedComponents??d(`blocks`,a),Z=d(`covered`,a);return _(`covered-clear`,a,[h(`covered`,Z)],`covered[coveredOffset + index] = 0u;`),_(`covered-tree`,i,[m(`parentEdges`,I),m(`activeSegments`,B),h(`covered`,Z),m(`reverse`,T)],`
let edge = parentEdges[parentEdgesOffset + index];
if (edge != NONE) {
  let value = activeSegments[activeSegmentsOffset + index];
  covered[coveredOffset + edge] = value;
  covered[coveredOffset + reverse[reverseOffset + edge]] = value;
}`),_(`publish-blocks`,a,[...D,m(`labels`,V),h(`blocks`,X),...r.output.bridges?[m(`tree`,b),m(`covered`,Z),h(`bridges`,r.output.bridges)]:[]],`
let source = sources[sourcesOffset + index * 2u];
let destination = sources[sourcesOffset + index * 2u + 1u];
let twin = reverse[reverseOffset + index];
let valid = source < VERTEX_COUNT && destination < VERTEX_COUNT && twin < EDGE_COUNT;
var label = NONE;
if (valid) { label = labels[labelsOffset + min(index, twin)]; }
blocks[blocksOffset + index] = label;
${r.output.bridges?`bridges[bridgesOffset + index] = select(0u, 1u, valid && tree[treeOffset + index] != 0u && covered[coveredOffset + index] == 0u);`:``}`),r.output.articulationPoints&&_(`publish-articulation`,i,[...E,m(`blocks`,X),h(`articulation`,r.output.articulationPoints)],`
var firstBlock = NONE; var articulationFlag = 0u;
let first = min(offsets[offsetsOffset + index], EDGE_COUNT);
let last = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
for (var edge = first; edge < last; edge++) {
  let blockLabel = blocks[blocksOffset + edge];
  if (neighbors[neighborsOffset + edge] == index || blockLabel == NONE) { continue; }
  if (firstBlock == NONE) { firstBlock = blockLabel; }
  else if (blockLabel != firstBlock) { articulationFlag = 1u; }
}
articulation[articulationOffset + index] = articulationFlag;`),u}},K=12,q=9,J=2147483646,Y=2147483646,X=1024,Z=65536,Q=class{props;id;nodeCount;edgeCount;mode;strongComponentsMode;strongComponentsMaximumIterations;estimatedCommandNodeCount;constructor(e){this.props=e,this.id=e.id??`network-structure`,this.mode=e.mode??`auto`,a(this.id,this.mode===`auto`||this.mode===`parallel`||this.mode===`exact`,`mode must be 'auto', 'parallel' or 'exact'`),this.strongComponentsMaximumIterations=e.strongComponentsMaximumIterations??256,o(this.id,`strongComponentsMaximumIterations`,this.strongComponentsMaximumIterations,1,D);let t=n(this.id,e.topology);this.nodeCount=t.nodeCount,this.edgeCount=t.edgeCount,this.strongComponentsMode=this.mode===`auto`?this.nodeCount<=1024&&this.edgeCount<=65536?`exact`:`parallel`:this.mode;let{output:l}=e,u=!!(l.bridges||l.articulationPoints||l.biconnectedComponents);o(this.id,`nodeCount`,this.nodeCount,1,this.mode===`exact`?X:J),o(this.id,`edgeCount`,this.edgeCount,0,this.mode===`exact`?Z:Y),a(this.id,Object.values(l).some(Boolean),`requires an output`);for(let[e,t]of Object.entries(l)){if(!t)continue;r(t,[`uint32`],`${this.id} ${e}`);let n=e===`strongComponents`||e===`articulationPoints`?this.nodeCount:this.edgeCount;a(this.id,t.length===n,`output.${e} must hold ${n} rows (one per ${n===this.nodeCount?`vertex`:`CSR slot`})`)}u&&(a(this.id,e.reverseEdgeSlots!==void 0,`reverseEdgeSlots is required for bridges, articulationPoints and biconnectedComponents`),r(e.reverseEdgeSlots,[`uint32`],`${this.id} reverseEdgeSlots`),a(this.id,e.reverseEdgeSlots.length===this.edgeCount,`reverseEdgeSlots must hold one row per CSR slot`));let f=e.strongComponentsStatus?Object.values(e.strongComponentsStatus):[];if(e.strongComponentsStatus){a(this.id,!!l.strongComponents,`strongComponentsStatus requires output.strongComponents`);for(let t of[`converged`,`iterationCount`,`iterationLimitReached`]){let n=e.strongComponentsStatus[t];r(n,[`uint32`],`${this.id} strongComponentsStatus.${t}`),a(this.id,n.length===1,`strongComponentsStatus.${t} must hold exactly one uint32 row`)}}let p=U(this.nodeCount,this.edgeCount);u&&this.mode!==`exact`&&i(this.id,`edgeCount + nodeCount * treeLevels (auxiliaryCount)`,p.auxiliaryCount,H),this.estimatedCommandNodeCount=(l.strongComponents?this.strongComponentsMode===`exact`?2+ +!!e.strongComponentsStatus:2+6*this.strongComponentsMaximumIterations:0)+(u?this.mode===`exact`?1:p.estimatedCommandNodeCount:0),c(this.id,this.estimatedCommandNodeCount),m(this.id,{...l,...e.strongComponentsStatus});let h=[...Object.values(e.output),...f].filter(e=>e!==void 0).map(e=>e.buffer);a(this.id,new Set(h).size===h.length,`outputs must use distinct buffers`),d(this.id,[...Object.values(l),...f],[...s(e.topology),e.reverseEdgeSlots])}getCommandNodes(n){let{id:r,props:a,nodeCount:o,edgeCount:c}=this,{output:d,topology:f}=a;u(r,n,[...s(f),a.reverseEdgeSlots,...Object.values(d),...Object.values(a.strongComponentsStatus??{})]);for(let[e,o]of Object.entries({offsets:f.offsets,neighbors:f.neighbors,weights:f.weights,nodePositions:f.nodePositions,reverseEdgeSlots:a.reverseEdgeSlots,...d,...a.strongComponentsStatus}))o&&(i(r,`${e} buffer byteLength (maxBufferSize)`,o.buffer.byteLength,n.device.limits.maxBufferSize),l(n.device,r,e,t(o).size,`reduce nodeCount or edgeCount to fit the device storage binding limit`));let m=!!(d.bridges||d.articulationPoints||d.biconnectedComponents),h={controlAndDispatch:(d.strongComponents&&this.strongComponentsMode===`parallel`?K:m&&this.mode!==`exact`?q:1)*4,vertices:o*4,...d.strongComponents&&this.strongComponentsMode===`exact`?{reachable:o*o*4}:{},...m&&this.mode===`exact`?{traversal:o*6*4,stack:Math.max(c,1)*4}:{},...m&&this.mode!==`exact`?{endpointsAndTour:Math.max(c*2,1)*4,auxiliary:U(o,c).auxiliaryCount*4}:{}};for(let[e,t]of Object.entries(h))i(r,`${e} scratch byteLength (maxBufferSize)`,t,n.device.limits.maxBufferSize),l(n.device,r,`${e} scratch`,t,`reduce nodeCount or edgeCount to fit the device storage binding limit`);let g=[];if(this.mode!==`exact`){if(d.strongComponents&&this.strongComponentsMode===`parallel`){let t=a.strongComponentsStatus??{converged:e(n,`${r}-strong-converged`,`uint32`,1),iterationCount:e(n,`${r}-strong-iteration-count`,`uint32`,1),iterationLimitReached:e(n,`${r}-strong-iteration-limit`,`uint32`,1)};g.push(...new V({id:`${r}-strong`,topology:f,maximumIterations:this.strongComponentsMaximumIterations,output:{strongComponents:d.strongComponents,status:t}}).getCommandNodes(n))}m&&g.push(...W(n,`${r}-resilience`,a,o,c))}let _=[{name:`offsets`,view:f.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:f.neighbors,type:`u32`,access:`read`}],v=`const NODE_COUNT: u32 = ${o}u; const EDGE_COUNT: u32 = ${c}u; const NONE: u32 = 0xffffffffu;`;if(d.strongComponents&&this.strongComponentsMode===`exact`){let t=e(n,`${r}-reachable`,`uint32`,o*o);g.push(p(n,{id:`${r}-reach`,operation:`GPUNetworkStructure`,variant:`reach`,invocationCount:o,bindings:[..._,{name:`reachable`,view:t,type:`u32`,access:`read_write`}],declarations:v,body:`var queue: array<u32, ${o}>;
  let base = reachableOffset + index * NODE_COUNT;
  for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { reachable[base + vertex] = 0u; }
  reachable[base + index] = 1u; queue[0] = index;
  var head = 0u; var tail = 1u;
  while (head < tail) {
    let vertex = queue[head]; head++;
    let first = min(offsets[offsetsOffset + vertex], EDGE_COUNT);
    let last = min(offsets[offsetsOffset + vertex + 1u], EDGE_COUNT);
    for (var edge = first; edge < last; edge++) {
      let destination = neighbors[neighborsOffset + edge];
      if (destination < NODE_COUNT && reachable[base + destination] == 0u) {
        reachable[base + destination] = 1u; queue[tail] = destination; tail++;
      }
    }
  }`}),p(n,{id:`${r}-strong-components`,operation:`GPUNetworkStructure`,variant:`strong-components`,invocationCount:o,bindings:[{name:`reachable`,view:t,type:`u32`,access:`read`},{name:`labels`,view:d.strongComponents,type:`u32`,access:`read_write`}],declarations:v,body:`var label = index;
  for (var vertex = 0u; vertex < index; vertex++) {
    if (reachable[reachableOffset + index * NODE_COUNT + vertex] != 0u && reachable[reachableOffset + vertex * NODE_COUNT + index] != 0u) { label = vertex; break; }
  }
  labels[labelsOffset + index] = label;`}))}if(this.mode===`exact`&&a.reverseEdgeSlots&&(d.bridges||d.articulationPoints||d.biconnectedComponents)){let t=d.bridges??e(n,`${r}-bridges`,`uint32`,Math.max(c,1)),i=d.articulationPoints??e(n,`${r}-articulation`,`uint32`,o),s=d.biconnectedComponents??e(n,`${r}-blocks`,`uint32`,Math.max(c,1)),l=e(n,`${r}-edge-stack`,`uint32`,Math.max(c,1)),u=e(n,`${r}-traversal`,`uint32`,o*6);g.push(p(n,{id:`${r}-resilience`,operation:`GPUNetworkStructure`,variant:`resilience`,invocationCount:1,workgroupSize:2,bindings:[..._,{name:`reverse`,view:a.reverseEdgeSlots,type:`u32`,access:`read`},{name:`bridges`,view:t,type:`u32`,access:`read_write`},{name:`articulation`,view:i,type:`u32`,access:`read_write`},{name:`blocks`,view:s,type:`u32`,access:`read_write`},{name:`stack`,view:l,type:`u32`,access:`read_write`},{name:`traversal`,view:u,type:`u32`,access:`read_write`}],declarations:v,body:`for (var vertex = 0u; vertex < NODE_COUNT; vertex++) {
    traversal[traversalOffset + 5u * NODE_COUNT + vertex] = 0u;
    traversal[traversalOffset + 0u * NODE_COUNT + vertex] = NONE; traversal[traversalOffset + 2u * NODE_COUNT + vertex] = NONE; traversal[traversalOffset + 3u * NODE_COUNT + vertex] = NONE;
    traversal[traversalOffset + 4u * NODE_COUNT + vertex] = min(offsets[offsetsOffset + vertex], EDGE_COUNT);
    articulation[articulationOffset + vertex] = 0u;
  }
  for (var edge = 0u; edge < EDGE_COUNT; edge++) { bridges[bridgesOffset + edge] = 0u; blocks[blocksOffset + edge] = NONE; }
  var clock = 0u; var stackCount = 0u;
  for (var root = 0u; root < NODE_COUNT; root++) {
    if (traversal[traversalOffset + 0u * NODE_COUNT + root] != NONE) { continue; }
    var vertex = root; traversal[traversalOffset + 0u * NODE_COUNT + vertex] = clock; traversal[traversalOffset + 1u * NODE_COUNT + vertex] = clock; clock++;
    loop {
      let last = min(offsets[offsetsOffset + vertex + 1u], EDGE_COUNT);
      if (traversal[traversalOffset + 4u * NODE_COUNT + vertex] < last) {
        let edge = traversal[traversalOffset + 4u * NODE_COUNT + vertex]; traversal[traversalOffset + 4u * NODE_COUNT + vertex]++;
        let destination = neighbors[neighborsOffset + edge];
        let twin = reverse[reverseOffset + edge];
        if (destination >= NODE_COUNT || twin >= EDGE_COUNT) { continue; }
        if (destination == vertex) { blocks[blocksOffset + edge] = min(edge, twin); continue; }
        if (traversal[traversalOffset + 3u * NODE_COUNT + vertex] != NONE && edge == reverse[reverseOffset + traversal[traversalOffset + 3u * NODE_COUNT + vertex]]) { continue; }
        if (traversal[traversalOffset + 0u * NODE_COUNT + destination] == NONE) {
          stack[stackOffset + stackCount] = edge; stackCount++;
          traversal[traversalOffset + 5u * NODE_COUNT + vertex]++; traversal[traversalOffset + 2u * NODE_COUNT + destination] = vertex; traversal[traversalOffset + 3u * NODE_COUNT + destination] = edge;
          vertex = destination; traversal[traversalOffset + 0u * NODE_COUNT + vertex] = clock; traversal[traversalOffset + 1u * NODE_COUNT + vertex] = clock; clock++;
        } else if (traversal[traversalOffset + 0u * NODE_COUNT + destination] < traversal[traversalOffset + 0u * NODE_COUNT + vertex]) {
          stack[stackOffset + stackCount] = edge; stackCount++;
          traversal[traversalOffset + 1u * NODE_COUNT + vertex] = min(traversal[traversalOffset + 1u * NODE_COUNT + vertex], traversal[traversalOffset + 0u * NODE_COUNT + destination]);
        }
      } else {
        if (traversal[traversalOffset + 2u * NODE_COUNT + vertex] == NONE) {
          articulation[articulationOffset + vertex] = select(0u, 1u, traversal[traversalOffset + 5u * NODE_COUNT + vertex] > 1u); break;
        }
        let ancestor = traversal[traversalOffset + 2u * NODE_COUNT + vertex]; let treeEdge = traversal[traversalOffset + 3u * NODE_COUNT + vertex];
        traversal[traversalOffset + 1u * NODE_COUNT + ancestor] = min(traversal[traversalOffset + 1u * NODE_COUNT + ancestor], traversal[traversalOffset + 1u * NODE_COUNT + vertex]);
        if (traversal[traversalOffset + 1u * NODE_COUNT + vertex] > traversal[traversalOffset + 0u * NODE_COUNT + ancestor]) {
          bridges[bridgesOffset + treeEdge] = 1u;
          bridges[bridgesOffset + reverse[reverseOffset + treeEdge]] = 1u;
        }
        if (traversal[traversalOffset + 1u * NODE_COUNT + vertex] >= traversal[traversalOffset + 0u * NODE_COUNT + ancestor]) {
          if (traversal[traversalOffset + 2u * NODE_COUNT + ancestor] != NONE) { articulation[articulationOffset + ancestor] = 1u; }
          let end = stackCount; var label = NONE;
          loop {
            if (stackCount == 0u) { break; }
            stackCount--; let popped = stack[stackOffset + stackCount];
            label = min(label, min(popped, reverse[reverseOffset + popped]));
            if (popped == treeEdge) { break; }
          }
          for (var position = stackCount; position < end; position++) {
            let popped = stack[stackOffset + position]; blocks[blocksOffset + popped] = label;
            blocks[blocksOffset + reverse[reverseOffset + popped]] = label;
          }
        }
        vertex = ancestor;
      }
    }
  }`}))}return a.strongComponentsStatus&&this.strongComponentsMode===`exact`&&g.push(p(n,{id:`${r}-exact-status`,operation:`GPUNetworkStructure`,variant:`exact-status`,invocationCount:1,bindings:Object.entries(a.strongComponentsStatus).map(([e,t])=>({name:e,view:t,type:`u32`,access:`read_write`})),body:`converged[convergedOffset] = 1u; iterationCount[iterationCountOffset] = 1u; iterationLimitReached[iterationLimitReachedOffset] = 0u;`})),g}},$=h({parameters:E,caption:`Bridges (roads whose loss disconnects), articulation points, strong components.`,createInputs:({random:e,world:t})=>({base:b(e,t),view:null}),build(e,t,n){let r=C(t.base,n.closures);t.view={network:r,slotSegments:_(r),cache:new Map};let{topology:i}=g(e,r),a=e.allocate(`bridges`,`uint32`,r.slotCount),o=e.allocate(`articulation-points`,`uint32`,r.nodeCount),s=e.allocate(`strong-components`,`uint32`,r.nodeCount);return e.graph.add(new Q({topology:i,reverseEdgeSlots:e.upload(`reverse-slots`,`uint32`,r.reverseSlots).view,output:{bridges:a.view,articulationPoints:o.view,strongComponents:s.view}})),{outputs:{...w(e,r),slots:T(e,r),bridges:a,articulationPoints:o,strongComponents:s,cutSide:e.allocate(`cut-side`,`uint32`,r.nodeCount)}}},interaction:{kind:`probe`,hint:`hover a bridge`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,peek:r}){let i=r(n.bridges),a=t.view;if(!i||!a)return{readout:`bridges`};let{network:o,slotSegments:s,cache:c}=a,l=-1,u=1/0;for(let t=0;t<o.slotCount;t++){if(i[t]!==1)continue;let n=s[t*4],r=s[t*4+1],a=s[t*4+2]-n,o=s[t*4+3]-r,c=a*a+o*o||1,d=Math.min(1,Math.max(0,((e.x-n)*a+(e.y-r)*o)/c)),f=Math.hypot(e.x-n-a*d,e.y-r-o*d);f<u&&(u=f,l=t)}if(l<0)return{readout:`no bridges`};let d=Math.min(l,o.reverseSlots[l]),f=c.get(d);if(!f){let e=(e,t)=>{let n=new Uint8Array(o.nodeCount),r=[e];n[e]=1;for(let e=0;e<r.length;e++){let t=r[e];for(let e=o.offsets[t];e<o.offsets[t+1];e++){if(e===d||e===o.reverseSlots[d])continue;let t=o.neighbors[e];n[t]||(n[t]=1,r.push(t))}}return{seen:n,size:r.length,blocked:t}},t=e(o.slotSources[d],d),n=e(o.neighbors[d],d),r=t.size<=n.size?t:n,i=new Uint32Array(o.nodeCount).fill(4294967295);for(let e=0;e<i.length;e++)r.seen[e]&&(i[e]=0);f={cut:i,size:r.size},c.set(d,f)}return n.cutSide.write(f.cut),{glyphs:[{shape:`line`,from:[s[l*4],s[l*4+1]],to:[s[l*4+2],s[l*4+3]]}],readout:`bridge ${d} · loses ${f.size} intersections if closed`}}},draw:({outputs:e,values:t,theme:n})=>[S(e),t.output===0?y(e.slots,e.bridges,{radius:2.4,...v(n)}):x(e,t.output===1?e.articulationPoints:e.strongComponents,{radius:2.4}),{layer:`dots`,positions:e.nodes,values:e.cutSide,ramp:`categorical`,categories:[[...n.categories[2],1]],radius:6}]});export{$ as default};
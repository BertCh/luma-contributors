import{a as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t}from"./spatial-output-validation-nw5lOmVL.js";import{i as n,r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-flag-offsets-uwVnNtbl.js";import{t as a}from"./gpu-scan-B4NNTfF9.js";import{t as o}from"./gpu-sort-xRm8l5yn.js";import{a as s,o as c,t as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as u,r as d,t as f}from"./parallel-merge-tree-DHFVp61C.js";import{o as p}from"./sorted-segment-sums-Dvmm6OdR.js";import{E as m,T as h,f as g,l as _,p as v,v as y,x as b,y as x}from"./gpu-contributor-utils-BIwLQON5.js";import{s as S,v as C}from"./spatial-context-DwZEd4EZ.js";import{n as w}from"./temporal-context-xddVhAJD.js";import{t as ee}from"./gpu-neighbor-search-SzuWK-my.js";import{t as te}from"./neighbor-search-derived-parameters-DZFCGzl5.js";import{v as ne}from"./spatial-join-passes-CNxNpnDa.js";import{t as T}from"./define-run-B-brkCe4.js";import{n as E}from"./inputs-CJKKIRKK.js";import{n as D,t as O}from"./pick-Bo3t3hjv.js";import{Rt as k}from"./entry-preview-D70TdSbK.js";var A=`GPUHierarchicalDensityClustering`;function j(e){return`const ROW_COUNT: u32 = ${e.rowCount}u;
const NODE_COUNT: u32 = ${2*e.rowCount}u;
const MINIMUM_SAMPLES: u32 = ${e.minimumSamples}u;
const MINIMUM_CLUSTER_SIZE: u32 = ${e.minimumClusterSize}u;
const CLUSTER_CAPACITY: u32 = ${e.clusterCapacity}u;
const LEAF_CAPACITY: u32 = ${e.leafCapacity}u;
const INTERNAL_NODE_COUNT: u32 = ${e.internalNodeCount}u;
const LEVEL_COUNT: u32 = ${e.levelCount}u;
const EDGE_WORDS: u32 = 3u;
const NODE_WORDS: u32 = 5u;
const DENDROGRAM_WORDS: u32 = 5u;
const CLUSTER_WORDS: u32 = 3u;
const SELECTION_WORDS: u32 = 4u;
const NONE: u32 = 0xffffffffu;
const FLOAT32_MAXIMUM: f32 = 3.402823e38;
fn isFiniteNumber(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}`}function M(e){return e?{condition:e.condition,extraResources:e.extraResources}:{}}var N=`
fn findRoot(start: u32) -> u32 {
  var current = start;
  loop {
    let parent = atomicLoad(&parents[parentsOffset + current]);
    if (parent == current) {
      return current;
    }
    let grandparent = atomicLoad(&parents[parentsOffset + parent]);
    if (grandparent != parent) {
      atomicStore(&parents[parentsOffset + current], grandparent);
    }
    current = parent;
  }
}`;function re(e,t){return c(e,{id:t.id,operation:A,variant:`init`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...t.timeData?[{name:`timeData`,view:t.timeData,type:`f32`,access:`read`}]:[],...t.neighborOffsets&&t.neighborDistances?[{name:`neighborOffsets`,view:t.neighborOffsets,type:`u32`,access:`read`},{name:`neighborDistances`,view:t.neighborDistances,type:`f32`,access:`read`}]:[],{name:`boxMinima`,view:t.boxMinima,type:`f32`,access:`read_write`},{name:`boxMaxima`,view:t.boxMaxima,type:`f32`,access:`read_write`},{name:`parents`,view:t.parents,type:`u32`,access:`read_write`},{name:`coreDistances`,view:t.coreDistances,type:`f32`,access:`read_write`},{name:`state`,view:t.state,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.shape.rowCount,declarations:j(t.shape),body:`let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let isValid = isFiniteNumber(x) && isFiniteNumber(y)${t.timeData?` && isFiniteNumber(timeData[timeDataOffset + index])`:``};
  parents[parentsOffset + index] = index;
  var core = bitcast<f32>(0x7f800000u | (index & 0u));
  if (isValid) {
    ${t.neighborOffsets?`let begin = neighborOffsets[neighborOffsetsOffset + index];
    let end = neighborOffsets[neighborOffsetsOffset + index + 1u];
    if (end - begin + 1u >= MINIMUM_SAMPLES) {
      core = 0.0;
      for (var slot = begin; slot < end; slot++) {
        core = max(core, neighborDistances[neighborDistancesOffset + slot]);
      }
    }`:`core = 0.0;`}
  }
  coreDistances[coreDistancesOffset + index] = core;
  boxMinima[boxMinimaOffset + index * 2u] = select(FLOAT32_MAXIMUM, x, isValid);
  boxMinima[boxMinimaOffset + index * 2u + 1u] = select(FLOAT32_MAXIMUM, y, isValid);
  boxMaxima[boxMaximaOffset + index * 2u] = select(-FLOAT32_MAXIMUM, x, isValid);
  boxMaxima[boxMaximaOffset + index * 2u + 1u] = select(-FLOAT32_MAXIMUM, y, isValid);
  if (isValid) { atomicAdd(&state[stateOffset], 1u); }`})}function ie(e,t){return c(e,{id:t.id,operation:A,variant:`gate`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},t.dispatchBinding],invocationCount:1,body:`if (state[stateOffset] > state[stateOffset + 1u] + 1u) {
    ${t.armWGSL}
  } else {
    ${t.disarmWGSL}
  }`})}function ae(e,t){return c(e,{id:t.id,operation:A,variant:`resolve`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...t.timeData?[{name:`timeData`,view:t.timeData,type:`f32`,access:`read`}]:[],{name:`parents`,view:t.parents,type:`atomic<u32>`,access:`read_write`},{name:`components`,view:t.components,type:`u32`,access:`read_write`},{name:`componentBest`,view:t.componentBest,type:`u32`,access:`read_write`}],invocationCount:t.shape.rowCount,declarations:`${j(t.shape)}
${N}`,body:`let isValid = isFiniteNumber(positions[positionsOffset + index * 2u]) &&
    isFiniteNumber(positions[positionsOffset + index * 2u + 1u])${t.timeData?` && isFiniteNumber(timeData[timeDataOffset + index])`:``};
  components[componentsOffset + index] = select(NONE, findRoot(index), isValid);
  for (var word = 0u; word < EDGE_WORDS; word++) {
    componentBest[componentBestOffset + index * EDGE_WORDS + word] = NONE;
  }`,...M(t.gate)})}function oe(e,t){let n=t.word===1?`true`:`atomicLoad(&componentBest[best + 1u]) == candidates[candidatesOffset + index * EDGE_WORDS + 1u]`;return c(e,{id:t.id,operation:A,variant:t.word===1?`tie-low`:`tie-high`,bindings:[{name:`components`,view:t.components,type:`u32`,access:`read`},{name:`candidates`,view:t.candidates,type:`u32`,access:`read`},{name:`componentBest`,view:t.componentBest,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.shape.rowCount,declarations:j(t.shape),body:`let component = components[componentsOffset + index];
  let weight = candidates[candidatesOffset + index * EDGE_WORDS];
  if (component == NONE || weight == NONE) { return; }
  let best = componentBestOffset + component * EDGE_WORDS;
  if (atomicLoad(&componentBest[best]) == weight && ${n}) {
    atomicMin(&componentBest[best + ${t.word}u], candidates[candidatesOffset + index * EDGE_WORDS + ${t.word}u]);
  }`,...M(t.gate)})}function se(e,t){return c(e,{id:t.id,operation:A,variant:`add`,bindings:[{name:`components`,view:t.components,type:`u32`,access:`read`},{name:`componentBest`,view:t.componentBest,type:`u32`,access:`read`},{name:`parents`,view:t.parents,type:`atomic<u32>`,access:`read_write`},{name:`edges`,view:t.edges,type:`u32`,access:`read_write`},{name:`state`,view:t.state,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.shape.rowCount,declarations:`${j(t.shape)}
${N}
fn unite(first: u32, second: u32) {
  var a = findRoot(first);
  var b = findRoot(second);
  loop {
    if (a == b) {
      return;
    }
    let low = min(a, b);
    let high = max(a, b);
    let result = atomicCompareExchangeWeak(&parents[parentsOffset + high], high, low);
    if (result.exchanged) {
      return;
    }
    a = findRoot(low);
    b = findRoot(high);
  }
}`,body:`if (components[componentsOffset + index] != index) { return; }
  let best = componentBestOffset + index * EDGE_WORDS;
  let weight = componentBest[best];
  if (weight == NONE) { return; }
  let low = componentBest[best + 1u];
  let high = componentBest[best + 2u];
  var other = components[componentsOffset + low];
  if (other == index) { other = components[componentsOffset + high]; }
  let otherBest = componentBestOffset + other * EDGE_WORDS;
  let isMutual = componentBest[otherBest] == weight && componentBest[otherBest + 1u] == low &&
    componentBest[otherBest + 2u] == high;
  if (isMutual && other < index) { return; }
  let slot = atomicAdd(&state[stateOffset + 1u], 1u);
  if (slot < ROW_COUNT) {
    edges[edgesOffset + slot * EDGE_WORDS] = low;
    edges[edgesOffset + slot * EDGE_WORDS + 1u] = high;
    edges[edgesOffset + slot * EDGE_WORDS + 2u] = weight;
  }
  unite(low, high);`,...M(t.gate)})}function ce(e,t){let n=[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`edges`,view:t.edges,type:`u32`,access:`read`}];return t.order&&n.push({name:`order`,view:t.order,type:`u32`,access:`read`}),t.identity&&n.push({name:`identity`,view:t.identity,type:`u32`,access:`read_write`}),n.push({name:`keys`,view:t.keys,type:`u32`,access:`read_write`}),c(e,{id:t.id,operation:A,variant:`edge-keys-${t.word}`,bindings:n,invocationCount:t.shape.rowCount,declarations:j(t.shape),body:`let edgeCount = min(state[stateOffset + 1u], ROW_COUNT);
  ${t.identity?`identity[identityOffset + index] = index;`:``}
  let slot = ${t.order?`order[orderOffset + index]`:`index`};
  keys[keysOffset + index] = select(NONE, edges[edgesOffset + slot * EDGE_WORDS + ${t.word}u], slot < edgeCount);`})}function le(e,t){return c(e,{id:t.id,operation:A,variant:`labels`,bindings:[{name:`lowClusters`,view:t.lowClusters,type:`u32`,access:`read`},{name:`selection`,view:t.selection,type:`u32`,access:`read`},{name:`labels`,view:t.labels,type:`u32`,access:`read_write`}],invocationCount:t.shape.rowCount,declarations:j(t.shape),body:`let cluster = lowClusters[lowClustersOffset + index];
  var label = NONE;
  if (cluster != NONE) {
    let representative = selection[selectionOffset + cluster * SELECTION_WORDS + 2u];
    if (representative != NONE) {
      label = selection[selectionOffset + representative * SELECTION_WORDS + 3u];
    }
  }
  labels[labelsOffset + index] = label;`})}var P=`GPUHierarchicalDensityClustering`;function F(e,t){return{name:e,view:t,type:`u32`,access:`read`}}function I(e,t,n=!1){return{name:e,view:t,type:n?`atomic<u32>`:`u32`,access:`read_write`}}function ue(e,t){let{nodes:n,appendJumpNodes:r}=f(e,t.id,P),i=t.rowCount*2;return r(`ancestors`,i,2,Math.ceil(Math.log2(i))+1,t.pointers,`
  for (var word = 0u; word < 2u; word++) {
    let next = table[tableOffset + index * 2u + word];
    var successor = next;
    if (next != NONE) { successor = table[tableOffset + next * 2u + word]; }
    scratch[scratchOffset + index * 2u + word] = successor;
    if (successor != NONE && table[tableOffset + successor * 2u + word] != successor) {
      atomicOr(&pending[pendingOffset], 1u);
    }
  }`),n}function de(e,t){let{nodes:n,createWordView:a,appendKernel:o,appendJumpNodes:c}=f(e,t.id,P),l=t.shape.clusterCapacity,u=Math.ceil(Math.log2(Math.max(l,2)))+1,d=a(`children`,l*2),p=a(`heavy`,l),m=a(`depth`,l*2),h=a(`cursor`,2),g=a(`functions`,l*3),_=r(e,`${t.id}-best`,`float32`,l);o(`init`,l,[I(`children`,d),I(`selection`,t.selection),I(`cursor`,h)],`
  children[childrenOffset + index * 2u] = NONE;
  children[childrenOffset + index * 2u + 1u] = 0u;
  selection[selectionOffset + index * 4u] = 0u;
  selection[selectionOffset + index * 4u + 1u] = 0u;
  selection[selectionOffset + index * 4u + 2u] = NONE;
  selection[selectionOffset + index * 4u + 3u] = NONE;
  if (index == 0u) { cursor[cursorOffset] = 0u; cursor[cursorOffset + 1u] = 0u; }`),o(`children`,l,[F(`clusterCount`,t.clusterCount),F(`clusterInfo`,t.clusterInfo),I(`children`,d,!0)],`
  if (index >= clusterCount[clusterCountOffset]) { return; }
  let parent = clusterInfo[clusterInfoOffset + index * 3u + 1u];
  if (parent != NONE) {
    atomicMin(&children[childrenOffset + parent * 2u], index);
    atomicMax(&children[childrenOffset + parent * 2u + 1u], index);
  }`),o(`heavy`,l,[F(`children`,d),F(`clusterInfo`,t.clusterInfo),I(`heavy`,p)],`
  let first = children[childrenOffset + index * 2u];
  var chosen = first;
  if (first != NONE) {
    let second = children[childrenOffset + index * 2u + 1u];
    if (clusterInfo[clusterInfoOffset + second * 3u + 2u] > clusterInfo[clusterInfoOffset + first * 3u + 2u]) {
      chosen = second;
    }
  }
  heavy[heavyOffset + index] = chosen;`),o(`depth-init`,l,[F(`clusterCount`,t.clusterCount),F(`clusterInfo`,t.clusterInfo),F(`heavy`,p),I(`depth`,m)],`
  var parent = NONE;
  var light = 0u;
  if (index < clusterCount[clusterCountOffset]) {
    parent = clusterInfo[clusterInfoOffset + index * 3u + 1u];
    if (parent != NONE) { light = select(1u, 0u, heavy[heavyOffset + parent] == index); }
  }
  depth[depthOffset + index * 2u] = parent;
  depth[depthOffset + index * 2u + 1u] = light;`),c(`depth`,l,2,u,m,`
  let base = tableOffset + index * 2u;
  var parent = table[base];
  var total = table[base + 1u];
  if (parent != NONE) {
    total += table[tableOffset + parent * 2u + 1u];
    parent = table[tableOffset + parent * 2u];
  }
  scratch[scratchOffset + index * 2u] = parent;
  scratch[scratchOffset + index * 2u + 1u] = total;
  if (parent != NONE) { atomicOr(&pending[pendingOffset], 1u); }`),o(`maximum-depth`,l,[F(`clusterCount`,t.clusterCount),F(`depth`,m),I(`cursor`,h,!0)],`
  if (index < clusterCount[clusterCountOffset]) {
    atomicMax(&cursor[cursorOffset], depth[depthOffset + index * 2u + 1u] + 1u);
  }`);let v=s(e,{id:`${t.id}-level-dispatch`,operation:P,shapes:[[l]]}),y=s(e,{id:`${t.id}-chain-dispatch`,operation:P,shapes:[[l]]}),b=a(`functions-scratch`,l*3);for(let e=0;e<=Math.floor(Math.log2(t.shape.rowCount));e++){o(`level-gate-${e}`,1,[I(`cursor`,h),v.binding],`
  if (cursor[cursorOffset] != 0u) { ${v.armWGSL} }
  else { ${v.disarmWGSL} }
  cursor[cursorOffset + 1u] = 0u;`),o(`functions-${e}`,l,[F(`clusterCount`,t.clusterCount),F(`children`,d),F(`heavy`,p),F(`depth`,m),{name:`stability`,view:t.stability,type:`f32`,access:`read`},{name:`best`,view:_,type:`f32`,access:`read`},I(`functions`,g),I(`cursor`,h,!0)],`
  if (index >= clusterCount[clusterCountOffset] || depth[depthOffset + index * 2u + 1u] + 1u != atomicLoad(&cursor[cursorOffset])) { return; }
  let next = heavy[heavyOffset + index];
  var addition = 0.0;
  if (next != NONE) {
    let first = children[childrenOffset + index * 2u];
    let second = children[childrenOffset + index * 2u + 1u];
    let light = select(first, second, next == first);
    addition = best[bestOffset + light];
    atomicOr(&cursor[cursorOffset + 1u], 1u);
  }
  functions[functionsOffset + index * 3u] = next;
  functions[functionsOffset + index * 3u + 1u] = bitcast<u32>(stability[stabilityOffset + index]);
  functions[functionsOffset + index * 3u + 2u] = bitcast<u32>(addition);`,``,v.getGate(0));for(let n=0;n<u;n++)o(`chain-gate-${e}-${n}`,1,[I(`cursor`,h),y.binding],`
  if (cursor[cursorOffset] != 0u && cursor[cursorOffset + 1u] != 0u) { ${y.armWGSL} }
  else { ${y.disarmWGSL} }
  cursor[cursorOffset + 1u] = 0u;`),o(`chain-jump-${e}-${n}`,l,[F(`clusterCount`,t.clusterCount),F(`depth`,m),F(`functions`,g),I(`scratch`,b),I(`cursor`,h,!0)],`
  if (index >= clusterCount[clusterCountOffset] || depth[depthOffset + index * 2u + 1u] + 1u != atomicLoad(&cursor[cursorOffset])) { return; }
  let base = functionsOffset + index * 3u;
  var next = functions[base];
  var floorValue = bitcast<f32>(functions[base + 1u]);
  var addition = bitcast<f32>(functions[base + 2u]);
  if (next != NONE) {
    let child = functionsOffset + next * 3u;
    floorValue = max(floorValue, addition + bitcast<f32>(functions[child + 1u]));
    addition += bitcast<f32>(functions[child + 2u]);
    next = functions[child];
  }
  scratch[scratchOffset + index * 3u] = next;
  scratch[scratchOffset + index * 3u + 1u] = bitcast<u32>(floorValue);
  scratch[scratchOffset + index * 3u + 2u] = bitcast<u32>(addition);
  if (next != NONE) { atomicOr(&cursor[cursorOffset + 1u], 1u); }`,``,y.getGate(0)),o(`chain-copy-${e}-${n}`,l,[F(`clusterCount`,t.clusterCount),F(`depth`,m),F(`cursor`,h),F(`scratch`,b),I(`functions`,g)],`
  if (index >= clusterCount[clusterCountOffset] || depth[depthOffset + index * 2u + 1u] + 1u != cursor[cursorOffset]) { return; }
  for (var word = 0u; word < 3u; word++) {
    functions[functionsOffset + index * 3u + word] = scratch[scratchOffset + index * 3u + word];
  }`,``,y.getGate(0));o(`best-${e}`,l,[F(`clusterCount`,t.clusterCount),F(`depth`,m),F(`cursor`,h),F(`functions`,g),{name:`best`,view:_,type:`f32`,access:`read_write`}],`
  if (index >= clusterCount[clusterCountOffset] || depth[depthOffset + index * 2u + 1u] + 1u != cursor[cursorOffset]) { return; }
  best[bestOffset + index] = bitcast<f32>(functions[functionsOffset + index * 3u + 1u]);`,``,v.getGate(0)),o(`advance-${e}`,1,[I(`cursor`,h)],`
  if (cursor[cursorOffset] != 0u) { cursor[cursorOffset] -= 1u; }`)}let x=a(`representatives`,l*2);o(`candidates`,l,[F(`clusterCount`,t.clusterCount),F(`clusterInfo`,t.clusterInfo),F(`children`,d),{name:`best`,view:_,type:`f32`,access:`read`},{name:`stability`,view:t.stability,type:`f32`,access:`read`},I(`representatives`,x)],`
  var parent = NONE;
  var representative = 0u;
  if (index < clusterCount[clusterCountOffset]) {
    parent = clusterInfo[clusterInfoOffset + index * 3u + 1u];
    let first = children[childrenOffset + index * 2u];
    var childBest = 0.0;
    if (first != NONE) { childBest = best[bestOffset + first] + best[bestOffset + children[childrenOffset + index * 2u + 1u]]; }
    // The root is excluded. Equality selects the parent, as in the serial EOM traversal.
    if (parent != NONE && !(childBest > stability[stabilityOffset + index])) { representative = index + 1u; }
  }
  representatives[representativesOffset + index * 2u] = parent;
  representatives[representativesOffset + index * 2u + 1u] = representative;`),c(`representatives`,l,2,u,x,`
  let base = tableOffset + index * 2u;
  var parent = table[base];
  var representative = table[base + 1u];
  if (parent != NONE) {
    representative = max(representative, table[tableOffset + parent * 2u + 1u]);
    parent = table[tableOffset + parent * 2u];
  }
  scratch[scratchOffset + index * 2u] = parent;
  scratch[scratchOffset + index * 2u + 1u] = representative;
  if (parent != NONE) { atomicOr(&pending[pendingOffset], 1u); }`);let S=a(`selected`,l),C=a(`offsets`,l);return o(`flags`,l,[F(`representatives`,x),I(`selected`,S),I(`selection`,t.selection)],`
  let representative = representatives[representativesOffset + index * 2u + 1u];
  selected[selectedOffset + index] = select(0u, 1u, representative == index + 1u);
  selection[selectionOffset + index * 4u + 1u] = selected[selectedOffset + index];
  selection[selectionOffset + index * 4u + 2u] = select(NONE, representative - 1u, representative != 0u);`),n.push(...new i({id:`${t.id}-offsets`,flags:S,offsets:C,count:t.selectedCount}).getCommandNodes(e)),o(`labels`,l,[F(`selected`,S),F(`offsets`,C),I(`selection`,t.selection)],`
  selection[selectionOffset + index * 4u + 3u] = select(NONE, offsets[offsetsOffset + index], selected[selectedOffset + index] != 0u);`),n}var L=`GPUHierarchicalDensityClustering`;function R(e){return`const ROW_COUNT: u32 = ${e.rowCount}u;
const NODE_COUNT: u32 = ${2*e.rowCount}u;
const MINIMUM_SAMPLES: u32 = ${e.minimumSamples}u;
const MINIMUM_CLUSTER_SIZE: u32 = ${e.minimumClusterSize}u;
const CLUSTER_CAPACITY: u32 = ${e.clusterCapacity}u;
const LEAF_CAPACITY: u32 = ${e.leafCapacity}u;
const INTERNAL_NODE_COUNT: u32 = ${e.internalNodeCount}u;
const LEVEL_COUNT: u32 = ${e.levelCount}u;
const EDGE_WORDS: u32 = 3u;
const NODE_WORDS: u32 = 4u;
const DENDROGRAM_WORDS: u32 = 3u;
const CLUSTER_WORDS: u32 = 3u;
const SELECTION_WORDS: u32 = 4u;
const NONE: u32 = 0xffffffffu;
const FLOAT32_MAXIMUM: f32 = 3.402823e38;
fn isFiniteNumber(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}`}function z(e){return e?{condition:e.condition,extraResources:e.extraResources}:{}}function fe(e,t){return c(e,{id:t.id,operation:L,variant:`node-data`,bindings:[{name:`nodeMinima`,view:t.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:t.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:t.leafIds,type:`u32`,access:`read`},{name:`nodeData`,view:t.nodeData,type:`u32`,access:`read_write`}],invocationCount:2*t.shape.leafCapacity-1,declarations:R(t.shape),body:`let base = nodeDataOffset + index * NODE_WORDS;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  var row = NONE;
  if (index >= INTERNAL_NODE_COUNT) {
    row = leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT];
    if (row >= ROW_COUNT ||
        nodeMinima[nodeMinimaOffset + index * 2u] > nodeMaxima[nodeMaximaOffset + index * 2u] ||
        nodeMinima[nodeMinimaOffset + index * 2u + 1u] > nodeMaxima[nodeMaximaOffset + index * 2u + 1u]) { row = NONE; }
  }
  if (index >= INTERNAL_NODE_COUNT) { nodeData[base + 2u] = row; }`})}function pe(e,t){return c(e,{id:t.id,operation:L,variant:`leaf-flags`,bindings:[{name:`nodeData`,view:t.nodeData,type:`u32`,access:`read`},{name:`components`,view:t.components,type:`u32`,access:`read`},{name:`changeFlags`,view:t.changeFlags,type:`u32`,access:`read_write`}],invocationCount:t.shape.leafCapacity,declarations:`${R(t.shape)}
fn getLeafComponent(slot: u32) -> u32 {
  let row = nodeData[nodeDataOffset + (INTERNAL_NODE_COUNT + slot) * NODE_WORDS + 2u];
  return select(components[componentsOffset + min(row, ROW_COUNT - 1u)], NONE, row == NONE);
}`,body:`var isChange = 0u;
  if (index > 0u && getLeafComponent(index) != getLeafComponent(index - 1u)) { isChange = 1u; }
  changeFlags[changeFlagsOffset + index] = isChange;`,...z(t.gate)})}function B(e,t){return`const STACK_SIZE: u32 = ${2*e.levelCount}u;
const BOUND_SLACK: f32 = 1.000001;
fn getPoint(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
}
fn getBoxDistance(node: u32, point: vec2<f32>) -> f32 {
  let base = nodeDataOffset + node * NODE_WORDS;
  if (node >= INTERNAL_NODE_COUNT && nodeData[base + 2u] == NONE) { return FLOAT32_MAXIMUM; }
  let minimum = vec2<f32>(bitcast<f32>(nodeData[base]), bitcast<f32>(nodeData[base + 1u]));
  var maximum = minimum;
  if (node < INTERNAL_NODE_COUNT) { maximum = vec2<f32>(bitcast<f32>(nodeData[base + 2u]), bitcast<f32>(nodeData[base + 3u])); }
  if (minimum.x > maximum.x || minimum.y > maximum.y) { return FLOAT32_MAXIMUM; }
  let gap = max(max(minimum - point, point - maximum), vec2<f32>(0.0));
  return sqrt(dot(gap, gap));
}

fn getPairDistance(first: u32, second: u32) -> f32 {
  let spatialDifference = getPoint(first) - getPoint(second);
  ${t?`let scale = timeData[timeDataOffset + ROW_COUNT];
  var timeDifference = 0.0;
  if (scale > 0.0) { timeDifference = (timeData[timeDataOffset + first] - timeData[timeDataOffset + second]) * scale; }
  return length(vec3<f32>(spatialDifference, timeDifference));`:`return length(spatialDifference);`}
}`}function me(e,t){return c(e,{id:t.id,operation:L,variant:`search`,bindings:[{name:`nodeData`,view:t.nodeData,type:`u32`,access:`read`},{name:`changeScan`,view:t.changeScan,type:`u32`,access:`read`},{name:`positions`,view:t.positions,type:`f32`,access:`read`},...t.timeData?[{name:`timeData`,view:t.timeData,type:`f32`,access:`read`}]:[],{name:`coreDistances`,view:t.coreDistances,type:`f32`,access:`read`},{name:`components`,view:t.components,type:`u32`,access:`read`},{name:`candidates`,view:t.candidates,type:`u32`,access:`read_write`},{name:`componentBest`,view:t.componentBest,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.shape.rowCount,declarations:`${R(t.shape)}
${B(t.shape,!!t.timeData)}
// True when every leaf under the node belongs to 'component' (the node covers a contiguous range
// of leaf slots in the flat complete tree).
fn isNodeInComponent(node: u32, component: u32) -> bool {
  let depth = firstLeadingBit(node + 1u);
  let span = 1u << (LEVEL_COUNT - 1u - depth);
  let first = (node + 1u - (1u << depth)) * span;
  let last = first + span - 1u;
  if (changeScan[changeScanOffset + last] != changeScan[changeScanOffset + first]) { return false; }
  let row = nodeData[nodeDataOffset + (INTERNAL_NODE_COUNT + first) * NODE_WORDS + 2u];
  return row != NONE && components[componentsOffset + row] == component;
}
// Lexicographic (weight bits, low row, high row) order of candidate edges; weights are
// non-negative, so their f32 bits order like the values.
fn isBetter(weight: u32, low: u32, high: u32, bestWeight: u32, bestLow: u32, bestHigh: u32) -> bool {
  if (weight != bestWeight) { return weight < bestWeight; }
  if (low != bestLow) { return low < bestLow; }
  return high < bestHigh;
}
// A degenerate node holds identical xy values. The stable Hilbert sort preserves ascending
// source rows inside its single key, so its first leaf is a conservative row tie bound.
// This avoids a quadratic equal-distance MST search for large duplicate populations.
fn cannotImproveCoincidentNode(node: u32, queryRow: u32, core: f32,
    bestWeight: u32, bestLow: u32, bestHigh: u32) -> bool {
  if (bestWeight == NONE || node >= INTERNAL_NODE_COUNT) { return false; }
  let base = nodeDataOffset + node * NODE_WORDS;
  if (nodeData[base] != nodeData[base + 2u] || nodeData[base + 1u] != nodeData[base + 3u]) { return false; }
  let depth = firstLeadingBit(node + 1u);
  let span = 1u << (LEVEL_COUNT - 1u - depth);
  let first = (node + 1u - (1u << depth)) * span;
  let lowestRow = nodeData[nodeDataOffset + (INTERNAL_NODE_COUNT + first) * NODE_WORDS + 2u];
  if (lowestRow == NONE) { return false; }
  if (bestWeight != 0u) {
    ${t.timeData?`if (timeData[timeDataOffset + ROW_COUNT] != 0.0) { return false; }`:``}
    // With no time contribution every point in this node has the same pair distance.
    if (bitcast<u32>(max(core, getPairDistance(queryRow, lowestRow))) != bestWeight) { return false; }
  }
  return !isBetter(bestWeight, min(queryRow, lowestRow), max(queryRow, lowestRow), bestWeight, bestLow, bestHigh);
}`,body:`let component = components[componentsOffset + index];
  var bestWeight = NONE;
  var bestLow = NONE;
  var bestHigh = NONE;
  if (component != NONE) {
    let point = getPoint(index);
    let core = coreDistances[coreDistancesOffset + index];
    var stackNode: array<u32, STACK_SIZE>;
    var stackLower: array<f32, STACK_SIZE>;
    var top = 0u;
    if (!isNodeInComponent(0u, component)) {
      stackNode[0] = 0u;
      stackLower[0] = max(core, getBoxDistance(0u, point));
      top = 1u;
    }
    loop {
      if (top == 0u) { break; }
      top = top - 1u;
      let node = stackNode[top];
      let bound = select(FLOAT32_MAXIMUM, bitcast<f32>(bestWeight) * BOUND_SLACK, bestWeight != NONE);
      if (stackLower[top] > bound ||
          cannotImproveCoincidentNode(node, index, core, bestWeight, bestLow, bestHigh)) { continue; }
      if (node < INTERNAL_NODE_COUNT) {
        let first = node * 2u + 1u;
        let second = first + 1u;
        let firstLower = max(core, getBoxDistance(first, point));
        let secondLower = max(core, getBoxDistance(second, point));
        let firstUsable = firstLower <= bound && getBoxDistance(first, point) < FLOAT32_MAXIMUM &&
          !isNodeInComponent(first, component);
        let secondUsable = secondLower <= bound && getBoxDistance(second, point) < FLOAT32_MAXIMUM &&
          !isNodeInComponent(second, component);
        if (firstUsable && secondUsable && top + 2u <= STACK_SIZE) {
          if (firstLower <= secondLower) {
            stackNode[top] = second; stackLower[top] = secondLower;
            stackNode[top + 1u] = first; stackLower[top + 1u] = firstLower;
          } else {
            stackNode[top] = first; stackLower[top] = firstLower;
            stackNode[top + 1u] = second; stackLower[top + 1u] = secondLower;
          }
          top = top + 2u;
        } else if (firstUsable && top + 1u <= STACK_SIZE) {
          stackNode[top] = first; stackLower[top] = firstLower;
          top = top + 1u;
        } else if (secondUsable && top + 1u <= STACK_SIZE) {
          stackNode[top] = second; stackLower[top] = secondLower;
          top = top + 1u;
        }
        continue;
      }
      let other = nodeData[nodeDataOffset + node * NODE_WORDS + 2u];
      if (other == NONE || other == index || components[componentsOffset + other] == component) { continue; }
      let otherCore = coreDistances[coreDistancesOffset + other];
      // The same expression from both ends gives the same f32 weight (a - b is -(b - a)).
      let weight = bitcast<u32>(max(max(core, otherCore), getPairDistance(index, other)));
      let low = min(index, other);
      let high = max(index, other);
      if (isBetter(weight, low, high, bestWeight, bestLow, bestHigh)) {
        bestWeight = weight;
        bestLow = low;
        bestHigh = high;
      }
    }
    if (bestWeight != NONE) {
      atomicMin(&componentBest[componentBestOffset + component * EDGE_WORDS], bestWeight);
    }
  }
  candidates[candidatesOffset + index * EDGE_WORDS] = bestWeight;
  candidates[candidatesOffset + index * EDGE_WORDS + 1u] = bestLow;
  candidates[candidatesOffset + index * EDGE_WORDS + 2u] = bestHigh;`,...z(t.gate)})}var V=`
fn getNodeParent(node: u32) -> u32 {
  return dendrogram[dendrogramOffset + node * DENDROGRAM_WORDS];
}
fn getNodeSize(node: u32) -> u32 {
  return dendrogram[dendrogramOffset + node * DENDROGRAM_WORDS + 1u];
}
// Merge density 1 / weight; a zero weight (duplicate points) gives the largest f32 instead of
// scikit-learn's infinity, so differences stay finite.
fn getNodeLambda(node: u32) -> f32 {
  let weight = bitcast<f32>(dendrogram[dendrogramOffset + node * DENDROGRAM_WORDS + 2u]);
  return select(FLOAT32_MAXIMUM, 1.0 / weight, weight > 0.0);
}
fn isBigNode(node: u32) -> bool {
  return node != NONE && getNodeSize(node) >= MINIMUM_CLUSTER_SIZE;
}
// A node of at least minimumClusterSize points whose parent is a true split (both children big),
// or the root, starts a condensed cluster.
`,H=`
fn isClusterStart(node: u32) -> bool {
  if (!isBigNode(node)) { return false; }
  let parent = getNodeParent(node);
  if (parent == NONE) { return true; }
  let base = dendrogramChildrenOffset + parent * 2u;
  return isBigNode(dendrogramChildren[base]) && isBigNode(dendrogramChildren[base + 1u]);
}`;function he(e,t){return c(e,{id:t.id,operation:L,variant:`pointer-init`,bindings:[{name:`dendrogram`,view:t.dendrogram,type:`u32`,access:`read`},{name:`dendrogramChildren`,view:t.dendrogramChildren,type:`u32`,access:`read`},{name:`pointers`,view:t.pointers,type:`u32`,access:`read_write`},{name:`startFlags`,view:t.startFlags,type:`u32`,access:`read_write`}],invocationCount:2*t.shape.rowCount,declarations:`${R(t.shape)}
${V}
${H}`,body:`let parent = getNodeParent(index);
  let isStart = isClusterStart(index);
  pointers[pointersOffset + index * 2u] = select(parent, index, isStart);
  pointers[pointersOffset + index * 2u + 1u] = select(parent, index, isBigNode(index));
  startFlags[startFlagsOffset + index] = select(0u, 1u, isStart);`})}function ge(e,t){return c(e,{id:t.id,operation:L,variant:`cluster-info`,bindings:[{name:`dendrogram`,view:t.dendrogram,type:`u32`,access:`read`},{name:`startFlags`,view:t.startFlags,type:`u32`,access:`read`},{name:`clusterOffsets`,view:t.clusterOffsets,type:`u32`,access:`read`},{name:`pointers`,view:t.pointers,type:`u32`,access:`read`},{name:`clusterInfo`,view:t.clusterInfo,type:`u32`,access:`read_write`}],invocationCount:2*t.shape.rowCount,declarations:`${R(t.shape)}
${V}`,body:`if (startFlags[startFlagsOffset + index] == 0u) { return; }
  let cluster = clusterOffsets[clusterOffsetsOffset + index];
  if (cluster >= CLUSTER_CAPACITY) { return; }
  let parent = getNodeParent(index);
  var birth = 0.0;
  var parentCluster = NONE;
  if (parent != NONE) {
    birth = getNodeLambda(parent);
    let parentStart = pointers[pointersOffset + parent * 2u];
    parentCluster = clusterOffsets[clusterOffsetsOffset + parentStart];
  }
  clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS] = bitcast<u32>(birth);
  clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS + 1u] = parentCluster;
  clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS + 2u] = getNodeSize(index);`})}function _e(e,t){return c(e,{id:t.id,operation:L,variant:`contributions`,bindings:[{name:`dendrogram`,view:t.dendrogram,type:`u32`,access:`read`},{name:`startFlags`,view:t.startFlags,type:`u32`,access:`read`},{name:`clusterOffsets`,view:t.clusterOffsets,type:`u32`,access:`read`},{name:`pointers`,view:t.pointers,type:`u32`,access:`read`},{name:`clusterInfo`,view:t.clusterInfo,type:`u32`,access:`read`},{name:`contributionKeys`,view:t.contributionKeys,type:`u32`,access:`read_write`},{name:`contributions`,view:t.contributions,type:`f32`,access:`read_write`},{name:`lowClusters`,view:t.lowClusters,type:`u32`,access:`read_write`}],invocationCount:2*t.shape.rowCount,declarations:`${R(t.shape)}
${V}
fn getBirth(cluster: u32) -> f32 {
  return bitcast<f32>(clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS]);
}`,body:`var key = CLUSTER_CAPACITY;
  var value = 0.0;
  if (index < ROW_COUNT) {
    var lowCluster = NONE;
    let fallNode = pointers[pointersOffset + index * 2u + 1u];
    if (fallNode != NONE) {
      let start = pointers[pointersOffset + fallNode * 2u];
      if (start != NONE) {
        let cluster = clusterOffsets[clusterOffsetsOffset + start];
        if (cluster < CLUSTER_CAPACITY) {
          lowCluster = cluster;
          key = cluster;
          value = getNodeLambda(fallNode) - getBirth(cluster);
        }
      }
    }
    lowClusters[lowClustersOffset + index] = lowCluster;
  } else if (startFlags[startFlagsOffset + index] != 0u) {
    let cluster = clusterOffsets[clusterOffsetsOffset + index];
    if (cluster < CLUSTER_CAPACITY) {
      let parentCluster = clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS + 1u];
      if (parentCluster != NONE) {
        key = parentCluster;
        value = f32(getNodeSize(index)) * (getBirth(cluster) - getBirth(parentCluster));
      }
    }
  }
  contributionKeys[contributionKeysOffset + index] = key;
  contributions[contributionsOffset + index] = value;`})}function ve(e,t){return c(e,{id:t.id,operation:L,variant:t.timeData?`space-time-core`:`bvh-core`,invocationCount:t.shape.rowCount,bindings:[{name:`nodeData`,view:t.nodeData,type:`u32`,access:`read`},{name:`positions`,view:t.positions,type:`f32`,access:`read`},...t.timeData?[{name:`timeData`,view:t.timeData,type:`f32`,access:`read`}]:[],{name:`coreDistances`,view:t.coreDistances,type:`f32`,access:`read_write`}],declarations:`${R(t.shape)}
${B(t.shape,!!t.timeData)}`,body:`let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  coreDistances[coreDistancesOffset + index] = infinity;
  let point = getPoint(index);
  if (!isFiniteNumber(point.x) || !isFiniteNumber(point.y)${t.timeData?` || !isFiniteNumber(timeData[timeDataOffset + index])`:``}) { return; }
  // Only initialized slots below count are ever read; insertion writes each slot first.
  var nearest: array<f32, MINIMUM_SAMPLES>;
  var count = 0u;
  var stack: array<u32, STACK_SIZE>;
  stack[0] = 0u;
  var top = 1u;
  loop {
    if (top == 0u) { break; }
    top--;
    let node = stack[top];
    var bound = infinity;
    if (count == MINIMUM_SAMPLES - 1u) { bound = ${t.timeData?`nearest[count - 1u]`:`sqrt(nearest[count - 1u])`} * BOUND_SLACK; }
    let lower = getBoxDistance(node, point);
    if (lower >= FLOAT32_MAXIMUM || lower > bound) { continue; }
    if (node < INTERNAL_NODE_COUNT) {
      let first = node * 2u + 1u;
      let second = first + 1u;
      // Complete binary tree: DFS needs at most LEVEL_COUNT pending nodes.
      if (getBoxDistance(first, point) <= getBoxDistance(second, point)) {
        stack[top] = second; stack[top + 1u] = first;
      } else {
        stack[top] = first; stack[top + 1u] = second;
      }
      top += 2u;
      continue;
    }
    let other = nodeData[nodeDataOffset + node * NODE_WORDS + 2u];
    if (other == NONE || other == index${t.timeData?` || !isFiniteNumber(timeData[timeDataOffset + other])`:``}) { continue; }
    ${t.timeData?`let candidate = getPairDistance(index, other);`:`// Match the packed reference's scalar squared-distance expression and round sqrt once.
    let deltaX = positions[positionsOffset + other * 2u] - point.x;
    let deltaY = positions[positionsOffset + other * 2u + 1u] - point.y;
    let candidate = deltaX * deltaX + deltaY * deltaY;`}
    var slot = min(count, MINIMUM_SAMPLES - 2u);
    if (count == MINIMUM_SAMPLES - 1u && candidate >= nearest[slot]) { continue; }
    loop {
      if (slot == 0u) { break; }
      if (nearest[slot - 1u] <= candidate) { break; }
      nearest[slot] = nearest[slot - 1u];
      slot--;
    }
    nearest[slot] = candidate;
    count = min(count + 1u, MINIMUM_SAMPLES - 1u);
    // Distances are nonnegative: a full list ending at zero is already the exact answer.
    if (count == MINIMUM_SAMPLES - 1u && nearest[count - 1u] == 0.0) { break; }
  }
  if (count == MINIMUM_SAMPLES - 1u) {
    coreDistances[coreDistancesOffset + index] = ${t.timeData?`nearest[count - 1u]`:`sqrt(nearest[count - 1u])`};
  }`})}var U=`GPUHierarchicalDensityClustering`;function W(e,t){return{name:e,view:t,type:`u32`,access:`read`}}function G(e,t,n=!1){return{name:e,view:t,type:n?`atomic<u32>`:`u32`,access:`read_write`}}function ye(e,t){let n=t.shape.rowCount,r=n*2,{nodes:i,createWordView:a,appendKernel:o,parents:c,children:l}=d(e,{id:t.id,operation:U,nodeCount:r,edgeCount:n,boundedSplicing:!0,spliceMaximumWork:t.spliceMaximumWork,initializeBindings:[G(`dendrogram`,t.dendrogram)],initializeBody:`let base = dendrogramOffset + index * 3u;
  dendrogram[base] = NONE;
  dendrogram[base + 1u] = 0u;
  dendrogram[base + 2u] = 0u;`,edgeBindings:[W(`state`,t.state),W(`edges`,t.edges),W(`edgeOrder`,t.edgeOrder),G(`dendrogram`,t.dendrogram)],edgeBody:`if (index >= min(state[stateOffset + 1u], ROW_COUNT - 1u)) { return; }
  let edge = edgesOffset + edgeOrder[edgeOrderOffset + index] * 3u;
  let node = ROW_COUNT + index;
  dendrogram[dendrogramOffset + node * 3u + 2u] = edges[edge + 2u];
  let firstEndpoint = edges[edge];
  let secondEndpoint = edges[edge + 1u];`,declarations:`const ROW_COUNT: u32 = ${n}u;`}),u=r*2,f=a(`tour-successors`,u),p=a(`tour-totals`,u),m=a(`tour-successor-scratch`,u),h=a(`tour-total-scratch`,u),g=a(`tour-pending`,1);o(`tour-init`,u,[W(`parents`,c),W(`children`,l),G(`successors`,f),G(`totals`,p)],`let node = index / 2u;
  let first = children[childrenOffset + node * 2u];
  var successor = NONE;
  var contribution = 0u;
  if (index % 2u == 0u) {
    successor = node * 2u + 1u;
    if (first != NONE) { successor = first * 2u; }
    else if (node < ROW_COUNT) { contribution = 1u; }
  } else {
    let parent = parents[parentsOffset + node];
    if (parent != NONE) {
      successor = parent * 2u + 1u;
      if (children[childrenOffset + parent * 2u] == node) {
        successor = children[childrenOffset + parent * 2u + 1u] * 2u;
      }
    }
  }
  successors[successorsOffset + index] = successor;
  totals[totalsOffset + index] = contribution;`,`const ROW_COUNT: u32 = ${n}u;`);let _=s(e,{id:`${t.id}-tour-dispatch`,operation:U,shapes:[[u]]});o(`tour-seed`,1,[G(`pending`,g)],`pending[pendingOffset] = 1u;`);for(let e=0;e<Math.ceil(Math.log2(u));e++)o(`tour-gate-${e}`,1,[G(`pending`,g),_.binding],`if (pending[pendingOffset] != 0u) { ${_.armWGSL} }
  else { ${_.disarmWGSL} }
  pending[pendingOffset] = 0u;`),o(`tour-jump-${e}`,u,[W(`successors`,f),W(`totals`,p),G(`successorScratch`,m),G(`totalScratch`,h),G(`pending`,g,!0)],`var successor = successors[successorsOffset + index];
  var total = totals[totalsOffset + index];
  if (successor != NONE) {
    total += totals[totalsOffset + successor];
    successor = successors[successorsOffset + successor];
  }
  successorScratch[successorScratchOffset + index] = successor;
  totalScratch[totalScratchOffset + index] = total;
  if (successor != NONE) { atomicOr(&pending[pendingOffset], 1u); }`,``,_.getGate(0)),o(`tour-copy-${e}`,u,[W(`successorScratch`,m),W(`totalScratch`,h),G(`successors`,f),G(`totals`,p)],`successors[successorsOffset + index] = successorScratch[successorScratchOffset + index];
  totals[totalsOffset + index] = totalScratch[totalScratchOffset + index];`,``,_.getGate(0));return o(`finish`,r,[W(`parents`,c),W(`totals`,p),G(`dendrogram`,t.dendrogram)],`let base = dendrogramOffset + index * 3u;
  dendrogram[base] = parents[parentsOffset + index];
  dendrogram[base + 1u] = totals[totalsOffset + index * 2u] - totals[totalsOffset + index * 2u + 1u];`),{nodes:i,children:l}}var be=Object.freeze({...S,measure:`distance`}),K=`GPUHierarchicalDensityClustering`,q=2048,J=2**22,Y=2**20,X=class{id;props;spatialContext;minimumSamples;roundCount;temporalContext;estimatedCommandNodeCount;constructor(e){this.id=e.id??`hierarchical-density-clustering`,this.props=e;let{id:r}=this;this.spatialContext=C(r,e.spatialContext,be),m(e.positions,[`float32x2`],`${r} positions`);let i=e.positions.length;e.times?(h(r,e.temporalContext!==void 0,`times require temporalContext`),h(r,e.parameters!==void 0,`times require parameters`),this.temporalContext=w(r,e.temporalContext,{encodings:[`float32-relative`]}),m(e.times,[`float32`],`${r} times`),h(r,e.times.length===i,`times length must equal positions length`),m(e.parameters,[`float32`],`${r} parameters`),h(r,e.parameters.length===4,`parameters length must equal 4`)):h(r,e.temporalContext===void 0&&e.parameters===void 0,`temporalContext and parameters require times`),g(r,`positions rows`,i,1,J),h(r,e.coreDistanceSearch===void 0||e.coreDistanceSearch===`bvh`||e.coreDistanceSearch===`neighbor-search`,`coreDistanceSearch must be 'bvh' or 'neighbor-search'`),e.coreDistanceSearch===`neighbor-search`&&(g(r,`positions rows for coreDistanceSearch neighbor-search`,i,1,Y),h(r,!e.times,`coreDistanceSearch neighbor-search requires positions without times`)),g(r,`minimumClusterSize`,e.minimumClusterSize,2),this.minimumSamples=e.minimumSamples??Math.min(e.minimumClusterSize,32),g(r,`minimumSamples`,this.minimumSamples,1,32);let{output:a}=e;if(m(a.labels,[`uint32`],`${r} output.labels`),a.labels.length!==i)throw Error(`${r} output.labels length must equal positions length`);if(a.coreDistances&&(m(a.coreDistances,[`float32`],`${r} output.coreDistances`),a.coreDistances.length!==i))throw Error(`${r} output.coreDistances length must equal positions length`);for(let[e,t]of[[`output.status.count`,a.status?.count],[`output.status.requiredCount`,a.status?.requiredCount]])t&&(m(t,[`uint32`],`${r} ${e}`),g(r,`${e} length`,t.length,1));t(r,{labels:a.labels,coreDistances:a.coreDistances,count:a.status?.count,requiredCount:a.status?.requiredCount},[{read:[a.status?.count],write:[a.status?.requiredCount]}]);let o=[a.labels,a.coreDistances,a.status?.count,a.status?.requiredCount].filter(e=>!!e);for(let e=0;e<o.length;e++)for(let t=e+1;t<o.length;t++)h(r,!n(o[e],o[t]),`output fields must have disjoint writable byte ranges`);this.roundCount=Math.ceil(Math.log2(Math.max(i,2)))+1;let s=Math.ceil(Math.log2(2*Math.max(i,2)))+1;this.estimatedCommandNodeCount=q+64*s+4*s*s+2*u(2*i)+1,v(r,this.estimatedCommandNodeCount),x(r,o,[e.positions,e.times,e.parameters])}getCommandNodes(t){let{id:n,props:l,minimumSamples:u}=this,{output:d}=l;b(n,t,[l.positions,l.times,l.parameters,d.labels,d.coreDistances,d.status?.count,d.status?.requiredCount]);let f=l.positions.length;e(t,n,{positions:l.positions,times:l.times,parameters:l.parameters,labels:d.labels,coreDistances:d.coreDistances,count:d.status?.count,requiredCount:d.status?.requiredCount});let m=_(f),h=Math.floor(2*f/l.minimumClusterSize)+1;for(let[e,r]of[[`BVH node data`,(2*m-1)*4*4],[`BVH boxes and children`,(2*m-1)*8],[`bounded reconstruction endpoint task planes`,f*2*4],[`MST edges and candidates`,f*3*4],[`dendrogram parent/size/weight`,2*f*3*4],[`tree children, pointers and Euler-tour planes`,f*4*4],[`condensed-tree selection and pointer scratch`,h*4*4],[`row scans, sorts and segment sums`,Math.max(2*f+1,m)*4],[`time data`,(f+1)*4],[`dispatch and state`,24]])y(t.device,n,e,r,`lower positions rows`);l.coreDistanceSearch===`neighbor-search`&&u>1&&(y(t.device,n,`neighbor parameter data`,48,`use a device with a larger storage binding limit`),y(t.device,n,`neighbor distances, IDs and weights`,f*(u-1)*4,`lower minimumSamples or positions rows`));let g=(e,i)=>r(t,`${n}-${e}`,`uint32`,Math.max(i,1)),v=(e,i)=>r(t,`${n}-${e}`,`float32`,Math.max(i,1)),x=[],S=u-1,C,w;if(S>0&&l.coreDistanceSearch===`neighbor-search`){let e=te(t,{id:`${n}-neighbors-setup`,operation:K,positions:l.positions,radius:1/0}),r=f*S;C=g(`neighbor-offsets`,f+1),w=v(`neighbor-distances`,r),x.push(...e.nodes,...new ee({id:`${n}-neighbors`,mode:`knn`,k:S,positions:l.positions,parameters:e.parameters,gridSize:e.gridSize,output:{offsets:C,neighbors:g(`neighbor-ids`,r),weights:v(`neighbor-weights`,r),distances:w,status:{overflow:g(`neighbor-overflow`,1)}}}).getCommandNodes(t))}let T=l.times?v(`time-data`,f+1):void 0;l.times&&l.parameters&&T&&x.push(c(t,{id:`${n}-time-data`,operation:K,variant:`time-data`,invocationCount:f,bindings:[{name:`times`,view:l.times,type:`f32`,access:`read`},{name:`parameters`,view:l.parameters,type:`f32`,access:`read`},{name:`timeData`,view:T,type:`f32`,access:`read_write`}],body:`let scale = parameters[parametersOffset];
  let validScale = scale >= 0.0 && (bitcast<u32>(scale) & 0x7f800000u) != 0x7f800000u;
  timeData[timeDataOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), times[timesOffset + index], validScale);
  if (index == 0u) { timeData[timeDataOffset + ${f}u] = scale; }`}));let E=g(`state`,2),D=r(t,`${n}-box-minima`,`float32x2`,f),O=r(t,`${n}-box-maxima`,`float32x2`,f),k=g(`parents`,f),A=d.coreDistances??v(`core-distances`,f);x.push(xe(t,`${n}-state-clear`,E),re(t,{id:`${n}-init`,shape:this.getShape(0,0,1),positions:l.positions,timeData:T,neighborOffsets:C,neighborDistances:w,boxMinima:D,boxMaxima:O,parents:k,coreDistances:A,state:E}));let{bvh:j,nodes:M}=ne(t,`${n}-points`,K,D,O,m,f>=2),N=this.getShape(m,j.internalNodeCount,j.levelCount),P=g(`node-data`,j.nodeCount*4);x.push(...M,fe(t,{id:`${n}-node-data`,shape:N,nodeMinima:j.nodeMinima,nodeMaxima:j.nodeMaxima,leafIds:j.leafIds,nodeData:P})),S>0&&l.coreDistanceSearch!==`neighbor-search`&&x.push(ve(t,{id:`${n}-${T?`space-time`:`bvh`}-core`,shape:N,nodeData:P,positions:l.positions,timeData:T,coreDistances:A}));let F=g(`components`,f),I=g(`component-best`,f*3),L=g(`candidates`,f*3),R=g(`change-flags`,m),z=g(`change-scan`,m),B=g(`edges`,f*3),V=s(t,{id:`${n}-round-gate`,operation:K,shapes:[[f],[m]]}),H=V.getGate(0),U=V.getGate(1);for(let e=0;e<this.roundCount;e++){let r=`-${e}`;x.push(ie(t,{id:`${n}-gate${r}`,state:E,dispatchBinding:V.binding,armWGSL:V.armWGSL,disarmWGSL:V.disarmWGSL}),ae(t,{id:`${n}-resolve${r}`,shape:N,positions:l.positions,timeData:T,parents:k,components:F,componentBest:I,gate:H}),pe(t,{id:`${n}-leaf-flags${r}`,shape:N,nodeData:P,components:F,changeFlags:R,gate:U}),...new a({id:`${n}-leaf-scan${r}`,input:R,output:z,mode:`inclusive`}).getCommandNodes(t),me(t,{id:`${n}-search${r}`,shape:N,nodeData:P,changeScan:z,positions:l.positions,timeData:T,coreDistances:A,components:F,candidates:L,componentBest:I,gate:H}),oe(t,{id:`${n}-tie-low${r}`,shape:N,word:1,components:F,candidates:L,componentBest:I,gate:H}),oe(t,{id:`${n}-tie-high${r}`,shape:N,word:2,components:F,candidates:L,componentBest:I,gate:H}),se(t,{id:`${n}-add${r}`,shape:N,components:F,componentBest:I,parents:k,edges:B,state:E,gate:H}))}let W=g(`edge-identity`,f),G;for(let e of[1,0,2]){let r=g(`edge-keys-${e}`,f),i=g(`edge-sorted-keys-${e}`,f),a=g(`edge-order-${e}`,f);x.push(ce(t,{id:`${n}-edge-keys-${e}`,shape:N,word:e,state:E,edges:B,order:G,identity:G?void 0:W,keys:r}),...new o({id:`${n}-edge-sort-${e}`,keys:r,values:G??W,outputKeys:i,outputValues:a}).getCommandNodes(t)),G=a}let be=G,q=2*f,J=g(`dendrogram`,q*3),Y=ye(t,{id:`${n}-dendrogram`,shape:N,state:E,edges:B,edgeOrder:be,dendrogram:J});x.push(...Y.nodes);let X=g(`pointers`,q*2),Z=g(`start-flags`,q),Q=g(`cluster-offsets`,q),Se=g(`cluster-count`,1);x.push(he(t,{id:`${n}-pointer-init`,shape:N,dendrogram:J,dendrogramChildren:Y.children,pointers:X,startFlags:Z})),x.push(...ue(t,{id:`${n}-condensing`,rowCount:f,pointers:X})),x.push(...new i({id:`${n}-cluster-offsets`,flags:Z,offsets:Q,count:Se}).getCommandNodes(t));let $=g(`cluster-info`,h*3),Ce=g(`contribution-keys`,q),we=v(`contributions`,q),Te=g(`low-clusters`,f),Ee=v(`stability`,h),De=g(`selection`,h*4),Oe=d.status?.count??g(`selected-count`,1);return x.push(ge(t,{id:`${n}-cluster-info`,shape:N,dendrogram:J,startFlags:Z,clusterOffsets:Q,pointers:X,clusterInfo:$}),_e(t,{id:`${n}-contributions`,shape:N,dendrogram:J,startFlags:Z,clusterOffsets:Q,pointers:X,clusterInfo:$,contributionKeys:Ce,contributions:we,lowClusters:Te}),...p(t,{id:`${n}-stability`,operation:K,segmentCount:h,segmentKeys:Ce,reductions:[{name:`stability`,contributions:we,output:Ee}]}),...de(t,{id:`${n}-selection`,shape:N,clusterCount:Se,stability:Ee,clusterInfo:$,selection:De,selectedCount:Oe}),le(t,{id:`${n}-labels`,shape:N,lowClusters:Te,selection:De,labels:d.labels})),d.status?.requiredCount&&x.push(c(t,{id:`${n}-required-count`,operation:K,variant:`required-count`,invocationCount:1,bindings:[{name:`selectedCount`,view:Oe,type:`u32`,access:`read`},{name:`requiredCount`,view:d.status.requiredCount,type:`u32`,access:`read_write`}],body:`requiredCount[requiredCountOffset] = selectedCount[selectedCountOffset];`})),x}getShape(e,t,n){let r=this.props.positions.length;return{rowCount:r,minimumSamples:this.minimumSamples,minimumClusterSize:this.props.minimumClusterSize,clusterCapacity:Math.floor(2*r/this.props.minimumClusterSize)+1,leafCapacity:e,internalNodeCount:t,levelCount:n}}};function xe(e,t,n){return l(e,{id:t,operation:K,view:n,type:`u32`,value:`0u`})}var Z=T({parameters:k,caption:`Stable density clusters; noise is muted.`,createInputs:({random:e,world:t})=>{let n=E(e,t,700,5,.12);return{...n,index:D(n.positions),focus:{row:-1}}},build(e,t,n){let r=e.allocate(`labels`,`uint32`,t.count),i=e.allocate(`core-distances`,`float32`,t.count),a=e.allocate(`members`,`uint32`,t.count);return e.graph.add(new X({positions:e.upload(`positions`,`float32x2`,t.positions).view,minimumClusterSize:n.minimumClusterSize,minimumSamples:n.minimumSamples,output:{labels:r.view,coreDistances:i.view}})),{outputs:{labels:r,coreDistances:i,members:a,positions:e.upload(`points`,`float32x2`,t.positions)}}},interaction:{kind:`probe`,hint:`hover a point`,rest:[.6,.45],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=O(t.index,e.x,e.y,t.focus.row);t.focus.row=a;let o=i(n.labels),s=i(n.coreDistances),c=new Uint32Array(t.count);if(a<0||!o||!s)return n.members.write(c),{};let l=o[a],u=l===4294967295,d=0;for(let e=0;e<t.count;e++)!u&&o[e]===l&&(c[e]=1,d++);n.members.write(c);let f=[t.positions[a*2],t.positions[a*2+1]];return{glyphs:[{shape:`ring`,at:f,radius:s[a],dashed:!0},{shape:`crosshair`,at:f,size:12}],readout:`${u?`noise`:`cluster ${l} · ${d} points`} · core distance ${s[a].toFixed(3)} (k ${r.minimumSamples})`}}},draw:({outputs:e})=>[{layer:`dots`,context:!0,positions:e.positions,radius:2},{layer:`dots`,positions:e.positions,values:e.labels,radius:4.5},{layer:`dots`,positions:e.positions,values:e.members,ramp:`sequential`,range:[0,1],fade:!0,alpha:.6,radius:7.5}]});export{Z as default};
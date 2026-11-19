import{u as e}from"./kernel-C3tfQOn4.js";import{E as t,l as n,y as r}from"./gpu-contributor-utils-BIwLQON5.js";import{d as i}from"./wgsl-common-DfyXiYB0.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as o}from"./segment-bvh-CDgckfAv.js";import{s,v as c}from"./spatial-context-DwZEd4EZ.js";import{a as l}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as u}from"./segment-storage-validation-BUxpxgMF.js";import{A as d,d as f,s as p}from"./pointer-overlay-Qh8qflcA.js";import{t as m}from"./define-run-B-brkCe4.js";import{n as h}from"./pick-Bo3t3hjv.js";import{s as g,t as _,v}from"./polygon-set-CHja50mr.js";import{a as y}from"./pointer-support-C2rPRVFO.js";import{s as b,u as x}from"./shape-segments-DMcyF4V-.js";import{qa as S}from"./entry-preview-D70TdSbK.js";var C=Object.freeze({...s,measure:`distance`}),w=64,T=`GPUVertexSnap`,E=4294967295;function D(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Vertex snap target must hold 4 elements`);if(!Number.isFinite(e.tolerance)||e.tolerance<0)throw Error(`Vertex snap tolerance must be a non-negative finite number`);return t.set([e.tolerance,0,0,0]),t}var O=class{id;props;spatialContext;constructor(n){this.id=n.id??`vertex-snap`,this.spatialContext=c(this.id,n.spatialContext,C),this.props=n;let{id:i}=this;if(t(n.positions,[`float32x2`],`${i} positions`),t(n.referencePositions,[`float32x2`],`${i} referencePositions`),n.positions.length<1)throw Error(`${i} needs at least one position`);if(!!n.featureOffsets!=!!n.referenceOffsets)throw Error(`${i} featureOffsets and referenceOffsets must be given together`);if(n.featureOffsets&&n.referenceOffsets){if(e(n.featureOffsets,`${i} featureOffsets`),e(n.referenceOffsets,`${i} referenceOffsets`),n.featureOffsets.length<2)throw Error(`${i} featureOffsets must contain at least two rows`);if(n.referenceOffsets.length!==n.featureOffsets.length)throw Error(`${i} referenceOffsets must have as many rows as featureOffsets`)}if(t(n.parameters,[`float32`],`${i} parameters`),n.parameters.length<4)throw Error(`${i} parameters must hold 4 float32 values`);if(t(n.output.positions,[`float32x2`],`${i} output.positions`),n.output.positions.length!==n.positions.length)throw Error(`${i} output.positions must hold ${n.positions.length} rows`);if(n.output.referenceRows&&(e(n.output.referenceRows,`${i} output.referenceRows`),n.output.referenceRows.length!==n.positions.length))throw Error(`${i} output.referenceRows must hold ${n.positions.length} rows`);r(i,[n.output.positions,n.output.referenceRows],[n.positions,n.referencePositions,n.featureOffsets,n.referenceOffsets,n.parameters])}getCommandNodes(e){let{props:t,id:r}=this;l(e,r,Object.fromEntries([t.positions,t.referencePositions,t.featureOffsets,t.referenceOffsets,t.parameters,t.output.positions,t.output.referenceRows].map((e,t)=>[String(t),e])));let s=!!(t.featureOffsets&&t.referenceOffsets);!s&&t.referencePositions.length>w&&u(e,r,t.referencePositions.length);let c=0,d=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`referencePositions`,view:t.referencePositions,type:`f32`,access:`read`}];t.featureOffsets&&t.referenceOffsets&&d.push({name:`featureOffsets`,view:t.featureOffsets,type:`u32`,access:`read`},{name:`referenceOffsets`,view:t.referenceOffsets,type:`u32`,access:`read`}),d.push({name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`outputPositions`,view:t.output.positions,type:`f32`,access:`read_write`}),t.output.referenceRows&&d.push({name:`outputReferenceRows`,view:t.output.referenceRows,type:`u32`,access:`read_write`});let f=!s&&t.referencePositions.length>w,p=[];if(f){let i=o(e,{id:r,operation:T,minima:t.referencePositions,maxima:t.referencePositions,leafCapacity:n(t.referencePositions.length),spatialSort:!0});p.push(...i.nodes),d.push({name:`nodeMinima`,view:i.bvh.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:i.bvh.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:i.bvh.leafIds,type:`u32`,access:`read`}),c=i.bvh.internalNodeCount}return p.push(a(e,{id:`${r}-snap`,operation:T,variant:s?`pairwise`:f?`global-bvh`:`global`,bindings:d,invocationCount:t.positions.length,declarations:`
const FEATURE_COUNT: u32 = ${t.featureOffsets?t.featureOffsets.length-1:0}u;
const REFERENCE_COUNT: u32 = ${t.referencePositions.length}u;
const NO_REFERENCE: u32 = 0xffffffffu;
${i}
${f?`const INTERNAL_NODE_COUNT: u32 = ${c}u;
fn boxDistanceSquared(node: u32, point: vec2f) -> f32 {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  let gap = max(max(minimum - point, point - maximum), vec2f(0.0));
  return dot(gap, gap);
}`:``}
${s?`
// Upper-bound binary search: the feature owning a position row, or FEATURE_COUNT when none.
fn findFeature(row: u32) -> u32 {
  var low = 0u;
  var high = FEATURE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (featureOffsets[featureOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || low > FEATURE_COUNT) {
    return FEATURE_COUNT;
  }
  return low - 1u;
}`:``}`,body:`let tolerance = max(parameters[parametersOffset], 0.0);
  let toleranceSquared = tolerance * tolerance;
  let vertex = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  var referenceStart = 0u;
  var referenceEnd = REFERENCE_COUNT;
  ${s?`let feature = findFeature(index);
  if (feature == FEATURE_COUNT) {
    referenceEnd = 0u;
  } else {
    referenceStart = referenceOffsets[referenceOffsetsOffset + feature];
    referenceEnd = min(referenceOffsets[referenceOffsetsOffset + feature + 1u], REFERENCE_COUNT);
  }`:``}
  var bestRow = NO_REFERENCE;
  var bestDistance = toleranceSquared;
  ${f?`// Branch and bound over the reference BVH. A box distance never exceeds the squared distance of
  // any vertex inside it, so pruning on '>' keeps equal-distance rows and the lowest row wins ties.
  if (!isNanBits(vertex.x) && !isNanBits(vertex.y)) {
    var stack: array<u32, 64>;
    var stackLower: array<f32, 64>;
    var depth = 1u;
    stack[0] = 0u;
    stackLower[0] = boxDistanceSquared(0u, vertex);
    loop {
      if (depth == 0u) { break; }
      depth = depth - 1u;
      if (stackLower[depth] > bestDistance) { continue; }
      let node = stack[depth];
      if (node < INTERNAL_NODE_COUNT) {
        let first = node * 2u + 1u;
        let second = first + 1u;
        let firstLower = boxDistanceSquared(first, vertex);
        let secondLower = boxDistanceSquared(second, vertex);
        // Push the farther child first so the nearer one is popped next.
        if (firstLower <= secondLower) {
          if (secondLower <= bestDistance) { stack[depth] = second; stackLower[depth] = secondLower; depth = depth + 1u; }
          if (firstLower <= bestDistance) { stack[depth] = first; stackLower[depth] = firstLower; depth = depth + 1u; }
        } else {
          if (firstLower <= bestDistance) { stack[depth] = first; stackLower[depth] = firstLower; depth = depth + 1u; }
          if (secondLower <= bestDistance) { stack[depth] = second; stackLower[depth] = secondLower; depth = depth + 1u; }
        }
        continue;
      }
      let row = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      if (row >= REFERENCE_COUNT) { continue; }
      let reference = vec2<f32>(referencePositions[referencePositionsOffset + 2u * row], referencePositions[referencePositionsOffset + 2u * row + 1u]);
      let delta = reference - vertex;
      let distanceSquared = dot(delta, delta);
      if (distanceSquared < bestDistance || (distanceSquared == bestDistance && bestRow != NO_REFERENCE && row < bestRow)) {
        bestDistance = distanceSquared;
        bestRow = row;
      }
    }
  }`:`for (var row = referenceStart; row < referenceEnd; row++) {
    let reference = vec2<f32>(referencePositions[referencePositionsOffset + 2u * row], referencePositions[referencePositionsOffset + 2u * row + 1u]);
    let delta = reference - vertex;
    let distanceSquared = dot(delta, delta);
    // Strictly closer than the tolerance and strictly better than the best so far: ties keep the lowest row.
    if (distanceSquared < bestDistance) {
      bestDistance = distanceSquared;
      bestRow = row;
    }
  }`}
  var snapped = vertex;
  if (bestRow != NO_REFERENCE) {
    snapped = vec2<f32>(referencePositions[referencePositionsOffset + 2u * bestRow], referencePositions[referencePositionsOffset + 2u * bestRow + 1u]);
  }
  outputPositions[outputPositionsOffset + 2u * index] = snapped.x;
  outputPositions[outputPositionsOffset + 2u * index + 1u] = snapped.y;
  ${t.output.referenceRows?`outputReferenceRows[outputReferenceRowsOffset + index] = bestRow;`:``}`})),p}},k=p(S,`tolerance`),A=.4,j=m({parameters:S,caption:`Noisy vertices pulled onto the reference vertices within the tolerance.`,createInputs:({seed:e,world:t})=>{let n=g(d(e),t,{rows:2,columns:4}),r=h(n.positions);return{seed:e,set:n,index:r,restPoint:y((e,t)=>r.distanceTo(r.nearest(e,t),e,t),k.default/A,t)}},build(e,t){let n=v(e,t.set),r=t.set.positions.length/2,i=e.allocate(`noisy`,`float32x2`,r),a=e.allocate(`parameters`,`float32`,4),o=e.allocate(`snapped`,`float32x2`,r),s=e.allocate(`reference-rows`,`uint32`,r);e.graph.add(new O({positions:i.view,referencePositions:n.positions.view,parameters:a.view,output:{positions:o.view,referenceRows:s.view}}));let c=e.allocate(`did-snap`,`uint32`,r);return b(e,{id:`snap-flags`,invocationCount:r,bindings:[{name:`rows`,buffer:s,access:`read`},{name:`flags`,buffer:c,access:`read_write`}],body:`flags[flagsOffset + index] = select(1u, 0u, rows[rowsOffset + index] == ${E}u);`}),{outputs:{...n,noisy:i,snapped:o,didSnap:c,links:x(e,`snap-links`,i,o)},update:e=>{i.write(_(t.set.positions,d(t.seed+1),e.noise*.07)),a.write(D({tolerance:e.tolerance}))}}},interaction:{kind:`reach`,hint:`stretch the snap radius`,rest:({inputs:e,toUnit:t})=>t(e.restPoint),resolve(e,{inputs:t,outputs:n,peek:r}){let i=t.index.nearest(e.x,e.y),a=[t.set.positions[i*2],t.set.positions[i*2+1]],o=f(k,t.index.distanceTo(i,e.x,e.y)*A),s=r(n.didSnap),c=0;if(s)for(let e of s)c+=e;return{values:{tolerance:o},glyphs:[{shape:`ring`,at:a,radius:o},{shape:`line`,from:a,to:[e.x,e.y],dashed:!0},{shape:`dot`,at:a}],readout:`tolerance ${o.toFixed(3)} · ${s?Math.round(100*c/s.length):`...`}% snapped`}}},draw:({outputs:e,theme:t})=>[{layer:`segments`,context:!0,nodes:e.noisy,edges:e.edges},{layer:`lines`,segments:e.links,radius:1,alpha:.5},{layer:`segments`,nodes:e.snapped,edges:e.edges,values:e.didSnap,radius:1.8},{layer:`dots`,positions:e.positions,radius:2.6,color:[...t.ink,1]}]});export{j as default};
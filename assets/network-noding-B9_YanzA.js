import{o as e,s as t}from"./network-view-utils-B1U8oNKx.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{t as i}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as s}from"./sorted-segment-offsets-ChmOtA9F.js";import{E as c,f as l}from"./gpu-contributor-utils-BIwLQON5.js";import{r as u,t as d}from"./gpu-line-split-BIIctsOh.js";import{n as f}from"./gpu-line-noding-vE-FDCfr.js";import{s as p,v as m}from"./spatial-context-DwZEd4EZ.js";import{t as h}from"./network-workgroup-counters-BG1Cyc9s.js";import{u as g}from"./wgsl-common-DfyXiYB0.js";import{t as _}from"./define-run-B-brkCe4.js";import{n as v}from"./display-Bu20g6o5.js";import{d as y,s as b}from"./pointer-overlay-Qh8qflcA.js";import{l as x}from"./roads-DzovamHj.js";import{_a as S}from"./entry-preview-D70TdSbK.js";import{n as C}from"./street-network-DNqX5OPC.js";import{t as w}from"./fit-local-map-Bh8eeDQt.js";var T=Object.freeze({...p,measure:`distance`}),E=`GPUNetworkNoding`;function D(e){return Math.floor(u(e)/4)}var O=`${g}
fn getOrderedKey(value: f32) -> u32 {
  let bits = bitcast<u32>(select(value, 0.0, value == 0.0));
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn getOrderedValue(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}`,k=class{id;props;lineSplit;nodeCapacity;edgeCapacity;spatialContext;constructor(e){this.id=e.id??`network-noding`,this.props=e;let{id:n}=this;this.spatialContext=m(n,e.spatialContext,T);let r=e.output;t(this.id,r,[{outputs:{edges:r.edges},inputs:{count:r.pieces.status.count,offsets:r.pieces.geometry.lineOffsets,positions:r.pieces.geometry.positions}},{outputs:{neighbors:r.topology.neighbors,weights:r.topology.weights,edgeIds:r.topology.edgeIds},inputs:{edges:r.edges}},{outputs:{count:r.nodes.status.count,requiredCount:r.nodes.status.requiredCount,overflow:r.status?.overflow},inputs:{splitOverflow:r.pieces.status.overflow}}]);let{nodes:i,edges:a,topology:o,pieces:s}=e.output;if(l(n,`intersectionCapacity`,e.intersectionCapacity),c(e.tolerance,[`float32`],`${n} tolerance`),e.tolerance.length<1)throw Error(`${n} tolerance must contain one float32 row`);c(i.positions,[`float32x2`],`${n} output.nodes.positions`);for(let[t,r]of[[`output.nodes.status.count`,i.status.count],[`output.nodes.status.requiredCount`,i.status.requiredCount],[`output.status.overflow`,e.output.status?.overflow],[`output.status.uncertainCount`,e.output.status?.uncertainCount]])if(r&&(c(r,[`uint32`],`${n} ${t}`),r.length<1))throw Error(`${n} ${t} must contain one uint32 row`);if(this.nodeCapacity=i.positions.length,this.edgeCapacity=s.sourceIds.length,this.nodeCapacity<1)throw Error(`${n} output.nodes.positions must be non-empty`);if(this.edgeCapacity<1)throw Error(`${n} output.pieces.sourceIds must be non-empty`);if(o.offsets.length!==this.nodeCapacity+1)throw Error(`${n} output.topology.offsets length must be output.nodes.positions.length + 1`);c(o.offsets,[`uint32`],`${n} output.topology.offsets`),c(o.neighbors,[`uint32`],`${n} output.topology.neighbors`),c(o.weights,[`float32`],`${n} output.topology.weights`);for(let[e,t]of[[`topology.neighbors`,o.neighbors],[`topology.weights`,o.weights],[`topology.edgeIds`,o.edgeIds]])if(t&&t.length!==this.edgeCapacity*2)throw Error(`${n} output.${e} length must be twice the edge capacity`);c(a.fromNodes,[`uint32`],`${n} output.edges.fromNodes`),c(a.toNodes,[`uint32`],`${n} output.edges.toNodes`),c(a.lengths,[`float32`],`${n} output.edges.lengths`);for(let[e,t]of[[`output.edges.fromNodes`,a.fromNodes],[`output.edges.toNodes`,a.toNodes],[`output.edges.lengths`,a.lengths]])if(t.length!==this.edgeCapacity)throw Error(`${n} ${e} length must equal output.pieces.sourceIds.length`);this.lineSplit=new d({id:`${n}-split`,lines:e.lines,intersectionCapacity:e.intersectionCapacity,spatialSort:e.spatialSort,leafCapacity:e.leafCapacity,output:{pieces:s,status:e.output.status?.uncertainCount?{uncertainCount:e.output.status.uncertainCount}:void 0}})}getCommandNodes(t){let{id:c,props:u,nodeCapacity:d,edgeCapacity:p}=this,{tolerance:m,output:g}=u,{pieces:_,nodes:v,edges:y,topology:b}=g;l(c,`intersectionCapacity`,u.intersectionCapacity,1,D(t.device)),f(c,t.device,u.lines.positions.length,Math.max(u.lines.lineOffsets.length-1,0),u.intersectionCapacity),e(c,t,[u.lines.positions,u.lines.lineOffsets,_.geometry.positions,_.geometry.lineOffsets,_.sourceIds,_.status.count,_.status.requiredCount,_.status.overflow,_.status.vertexCount,_.status.requiredVertexCount,_.vertexSourceEdges,g.status?.uncertainCount,m,v.positions,v.status.count,v.status.requiredCount,y.fromNodes,y.toNodes,y.lengths,b.offsets,b.neighbors,b.weights,b.edgeIds,u.output.status?.overflow]),e(c,t,[],{endpointPoints:p*2*8,nodeScratch:d*4,csrCounts:(d+1)*4});let x=[],S=p*2,C=(e,r,i)=>n(t,`${c}-${e}`,r,i),w=_.status.overflow;x.push(...this.lineSplit.getCommandNodes(t));let T=C(`keys-x`,`uint32`,S),k=C(`keys-y`,`uint32`,S),A=C(`endpoint-points`,`float32x2`,S),j=C(`endpoint-indices`,`uint32`,S),M=C(`snap-origin`,`uint32`,2),N=C(`snap-overflow`,`uint32`,1),P=h({counters:[{target:`snapOrigin`,word:0,operation:`min`},{target:`snapOrigin`,word:1,operation:`min`}],rowCount:`ENDPOINT_COUNT`,rowSource:`let edge = index / 2u;
  endpointIndices[endpointIndicesOffset + index] = index;
  var point = vec2f(0.0);
  if (edge < pieceCount[pieceCountOffset]) {
    var vertex = pieceOffsets[pieceOffsetsOffset + edge];
    if ((index & 1u) == 1u) { vertex = pieceOffsets[pieceOffsetsOffset + edge + 1u] - 1u; }
    point = vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
    if (isFiniteValue(point.x) && isFiniteValue(point.y)) {
      minCounter(0u, getOrderedKey(point.x));
      minCounter(1u, getOrderedKey(point.y));
    }
  }
  endpointPoints[endpointPointsOffset + index * 2u] = point.x;
  endpointPoints[endpointPointsOffset + index * 2u + 1u] = point.y;`});x.push(o(t,{id:`${c}-clear-snap-origin`,operation:E,view:M,type:`u32`,value:`0xffffffffu`}),o(t,{id:`${c}-clear-snap-overflow`,operation:E,view:N,type:`u32`,value:`0u`}),a(t,{id:`${c}-endpoints`,operation:E,variant:`endpoints`,bindings:[{name:`pieceCount`,view:_.status.count,type:`u32`,access:`read`},{name:`pieceOffsets`,view:_.geometry.lineOffsets,type:`u32`,access:`read`},{name:`piecePositions`,view:_.geometry.positions,type:`f32`,access:`read`},{name:`endpointPoints`,view:A,type:`f32`,access:`read_write`},{name:`endpointIndices`,view:j,type:`u32`,access:`read_write`},{name:`snapOrigin`,view:M,type:`atomic<u32>`,access:`read_write`}],invocationCount:S,workgroupSize:256,guardIndex:!1,declarations:`${O}
const ENDPOINT_COUNT: u32 = ${S}u;
${P.declarations}`,body:P.body}),a(t,{id:`${c}-endpoint-keys`,operation:E,variant:`endpoint-keys`,bindings:[{name:`pieceCount`,view:_.status.count,type:`u32`,access:`read`},{name:`endpointPoints`,view:A,type:`f32`,access:`read`},{name:`tolerance`,view:m,type:`f32`,access:`read`},{name:`snapOrigin`,view:M,type:`u32`,access:`read`},{name:`snapOverflow`,view:N,type:`atomic<u32>`,access:`read_write`},{name:`keysX`,view:T,type:`u32`,access:`read_write`},{name:`keysY`,view:k,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`${O}
// Integer tolerance cell of value counted from origin, or the bit pattern for tolerance 0
// (only bit-identical end points merge). Sets isOverflow past 2^24 cells, where the f32 quotient
// no longer resolves unit cells.
fn snapKey(value: f32, origin: f32, size: f32, isOverflow: ptr<function, bool>) -> u32 {
  if (!isFiniteValue(value)) {
    return 0xfffffffeu;
  }
  if (size > 0.0) {
    let cell = floor(max(value - origin, 0.0) / size);
    if (!(cell < 16777216.0)) {
      *isOverflow = true;
      return 16777215u;
    }
    return u32(cell);
  }
  return min(bitcast<u32>(select(value, 0.0, value == 0.0)), 0xfffffffeu);
}`,body:`var keyX = 0xffffffffu;
  var keyY = 0xffffffffu;
  if (index / 2u < pieceCount[pieceCountOffset]) {
    let point = vec2f(endpointPoints[endpointPointsOffset + index * 2u], endpointPoints[endpointPointsOffset + index * 2u + 1u]);
    let origin = vec2f(
      getOrderedValue(snapOrigin[snapOriginOffset]),
      getOrderedValue(snapOrigin[snapOriginOffset + 1u])
    );
    let size = tolerance[toleranceOffset];
    var isOverflow = false;
    keyX = snapKey(point.x, origin.x, size, &isOverflow);
    keyY = snapKey(point.y, origin.y, size, &isOverflow);
    if (isOverflow) {
      atomicStore(&snapOverflow[snapOverflowOffset], 1u);
    }
  }
  keysX[keysXOffset + index] = keyX;
  keysY[keysYOffset + index] = keyY;`}));let F=i(t,{id:`${c}-sort`,operation:E,indices:j,indicesAreIdentity:!0,keys:[{view:T},{view:k}]});x.push(...F.nodes);let I=F.sortedIndices,L=C(`head-flags`,`uint32`,S),R=C(`head-ranks`,`uint32`,S),z=C(`endpoint-nodes`,`uint32`,S);x.push(a(t,{id:`${c}-head-flags`,operation:E,variant:`head-flags`,bindings:[{name:`order`,view:I,type:`u32`,access:`read`},{name:`keysX`,view:T,type:`u32`,access:`read`},{name:`keysY`,view:k,type:`u32`,access:`read`},{name:`headFlags`,view:L,type:`u32`,access:`read_write`}],invocationCount:S,body:`let current = order[orderOffset + index];
  var flag = select(0u, 1u, keysX[keysXOffset + current] != 0xffffffffu);
  if (flag == 1u && index > 0u) {
    let previous = order[orderOffset + index - 1u];
    if (keysX[keysXOffset + previous] == keysX[keysXOffset + current] &&
        keysY[keysYOffset + previous] == keysY[keysYOffset + current]) { flag = 0u; }
  }
  headFlags[headFlagsOffset + index] = flag;`}),...new r({id:`${c}-head-scan`,input:L,output:R,mode:`exclusive`}).getCommandNodes(t),a(t,{id:`${c}-assign-nodes`,operation:E,variant:`assign-nodes`,bindings:[{name:`order`,view:I,type:`u32`,access:`read`},{name:`keysX`,view:T,type:`u32`,access:`read`},{name:`headFlags`,view:L,type:`u32`,access:`read`},{name:`headRanks`,view:R,type:`u32`,access:`read`},{name:`endpointPoints`,view:A,type:`f32`,access:`read`},{name:`endpointNodes`,view:z,type:`u32`,access:`read_write`},{name:`nodePositions`,view:v.positions,type:`f32`,access:`read_write`}],invocationCount:S,declarations:`const NODE_CAPACITY: u32 = ${d}u;`,body:`let endpoint = order[orderOffset + index];
  if (keysX[keysXOffset + endpoint] == 0xffffffffu) {
    endpointNodes[endpointNodesOffset + endpoint] = 0xffffffffu;
    return;
  }
  let group = headRanks[headRanksOffset + index] + headFlags[headFlagsOffset + index] - 1u;
  endpointNodes[endpointNodesOffset + endpoint] = group;
  if (headFlags[headFlagsOffset + index] == 1u && group < NODE_CAPACITY) {
    nodePositions[nodePositionsOffset + group * 2u] = endpointPoints[endpointPointsOffset + endpoint * 2u];
    nodePositions[nodePositionsOffset + group * 2u + 1u] = endpointPoints[endpointPointsOffset + endpoint * 2u + 1u];
  }`})),x.push(a(t,{id:`${c}-edges`,operation:E,variant:`edges`,bindings:[{name:`pieceCount`,view:_.status.count,type:`u32`,access:`read`},{name:`pieceOffsets`,view:_.geometry.lineOffsets,type:`u32`,access:`read`},{name:`piecePositions`,view:_.geometry.positions,type:`f32`,access:`read`},{name:`endpointNodes`,view:z,type:`u32`,access:`read`},{name:`fromNodes`,view:y.fromNodes,type:`u32`,access:`read_write`},{name:`toNodes`,view:y.toNodes,type:`u32`,access:`read_write`},{name:`lengths`,view:y.lengths,type:`f32`,access:`read_write`}],invocationCount:p,body:`var length = 0.0;
  var fromNode = 0xffffffffu;
  var toNode = 0xffffffffu;
  if (index < pieceCount[pieceCountOffset]) {
    fromNode = endpointNodes[endpointNodesOffset + index * 2u];
    toNode = endpointNodes[endpointNodesOffset + index * 2u + 1u];
    let first = pieceOffsets[pieceOffsetsOffset + index];
    let last = pieceOffsets[pieceOffsetsOffset + index + 1u];
    for (var vertex = first + 1u; vertex < last; vertex++) {
      let a = vec2f(piecePositions[piecePositionsOffset + (vertex - 1u) * 2u], piecePositions[piecePositionsOffset + (vertex - 1u) * 2u + 1u]);
      let b = vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
      length = length + distance(a, b);
    }
  }
  fromNodes[fromNodesOffset + index] = fromNode;
  toNodes[toNodesOffset + index] = toNode;
  lengths[lengthsOffset + index] = length;`}));let B=C(`csr-flags`,`uint32`,S),V=C(`csr-positions`,`uint32`,S),H=C(`sorted-starts`,`uint32`,S),U=C(`sorted-entries`,`uint32`,S);x.push(a(t,{id:`${c}-csr-flags`,operation:E,variant:`csr-flags`,bindings:[{name:`order`,view:I,type:`u32`,access:`read`},{name:`fromNodes`,view:y.fromNodes,type:`u32`,access:`read`},{name:`toNodes`,view:y.toNodes,type:`u32`,access:`read`},{name:`csrFlags`,view:B,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`const NODE_CAPACITY: u32 = ${d}u;`,body:`let endpoint = order[orderOffset + index];
  let edge = endpoint / 2u;
  let fromNode = fromNodes[fromNodesOffset + edge];
  let toNode = toNodes[toNodesOffset + edge];
  let isValid = fromNode < NODE_CAPACITY && toNode < NODE_CAPACITY &&
    !(fromNode == toNode && (endpoint & 1u) == 1u);
  csrFlags[csrFlagsOffset + index] = select(0u, 1u, isValid);`}),...new r({id:`${c}-csr-scan`,input:B,output:V,mode:`exclusive`}).getCommandNodes(t),a(t,{id:`${c}-csr-compact`,operation:E,variant:`csr-compact`,bindings:[{name:`order`,view:I,type:`u32`,access:`read`},{name:`fromNodes`,view:y.fromNodes,type:`u32`,access:`read`},{name:`toNodes`,view:y.toNodes,type:`u32`,access:`read`},{name:`csrFlags`,view:B,type:`u32`,access:`read`},{name:`csrPositions`,view:V,type:`u32`,access:`read`},{name:`sortedStarts`,view:H,type:`u32`,access:`read_write`},{name:`sortedEntries`,view:U,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`const NODE_CAPACITY: u32 = ${d}u;
const LAST: u32 = ${S-1}u;`,body:`let total = csrPositions[csrPositionsOffset + LAST] + csrFlags[csrFlagsOffset + LAST];
  if (csrFlags[csrFlagsOffset + index] != 0u) {
    let endpoint = order[orderOffset + index];
    let edge = endpoint / 2u;
    let position = csrPositions[csrPositionsOffset + index];
    sortedStarts[sortedStartsOffset + position] =
      select(fromNodes[fromNodesOffset + edge], toNodes[toNodesOffset + edge], (endpoint & 1u) == 1u);
    sortedEntries[sortedEntriesOffset + position] = endpoint;
  }
  if (index >= total) {
    sortedStarts[sortedStartsOffset + index] = NODE_CAPACITY;
    sortedEntries[sortedEntriesOffset + index] = 0xffffffffu;
  }`}),s(t,{id:`${c}-csr-offsets`,operation:E,variant:`csr-offsets`,segmentCount:d,sortedKeys:H,segmentOffsets:b.offsets}));let W=[{name:`sortedStarts`,view:H,type:`u32`,access:`read`},{name:`sortedEntries`,view:U,type:`u32`,access:`read`},{name:`fromNodes`,view:y.fromNodes,type:`u32`,access:`read`},{name:`toNodes`,view:y.toNodes,type:`u32`,access:`read`},{name:`lengths`,view:y.lengths,type:`f32`,access:`read`},{name:`neighbors`,view:b.neighbors,type:`u32`,access:`read_write`},{name:`weights`,view:b.weights,type:`f32`,access:`read_write`}];b.edgeIds&&W.push({name:`edgeIds`,view:b.edgeIds,type:`u32`,access:`read_write`}),x.push(a(t,{id:`${c}-csr-entries`,operation:E,variant:`csr-entries`,bindings:W,invocationCount:S,declarations:`const NODE_CAPACITY: u32 = ${d}u;`,body:`var neighbor = 0xffffffffu;
  var weight = bitcast<f32>(0x7fc00000u | (index & 0u));
  var edgeId = 0xffffffffu;
  if (sortedStarts[sortedStartsOffset + index] < NODE_CAPACITY) {
    let entry = sortedEntries[sortedEntriesOffset + index];
    let edge = entry / 2u;
    neighbor = select(toNodes[toNodesOffset + edge], fromNodes[fromNodesOffset + edge], (entry & 1u) == 1u);
    weight = lengths[lengthsOffset + edge];
    edgeId = edge;
  }
  neighbors[neighborsOffset + index] = neighbor;
  weights[weightsOffset + index] = weight;
  ${b.edgeIds?`edgeIds[edgeIdsOffset + index] = edgeId;`:``}`}));let G=[{name:`headFlags`,view:L,type:`u32`,access:`read`},{name:`headRanks`,view:R,type:`u32`,access:`read`},{name:`splitOverflow`,view:w,type:`u32`,access:`read`},{name:`snapOverflow`,view:N,type:`u32`,access:`read`},{name:`nodeCount`,view:v.status.count,type:`u32`,access:`read_write`}];return v.status.requiredCount&&G.push({name:`requiredCount`,view:v.status.requiredCount,type:`u32`,access:`read_write`}),u.output.status?.overflow&&G.push({name:`overflow`,view:u.output.status.overflow,type:`u32`,access:`read_write`}),x.push(a(t,{id:`${c}-scalars`,operation:E,variant:`scalars`,bindings:G,invocationCount:1,declarations:`const NODE_CAPACITY: u32 = ${d}u;
const LAST: u32 = ${S-1}u;`,body:`let total = headRanks[headRanksOffset + LAST] + headFlags[headFlagsOffset + LAST];
  nodeCount[nodeCountOffset] = min(total, NODE_CAPACITY);
  ${v.status.requiredCount?`requiredCount[requiredCountOffset] = total;`:``}
  ${u.output.status?.overflow?`overflow[overflowOffset] = select(0u, 1u, total > NODE_CAPACITY || splitOverflow[splitOverflowOffset] != 0u || snapOverflow[snapOverflowOffset] != 0u);`:``}`})),x}},A=b(S,`tolerance`),j=b(S,`lineCount`).default,M=512,N=512,P=4096,F=9,I=_({parameters:S,caption:`Lines cut at crossings: noded edges and the nodes they snap to.`,createInputs:({world:e})=>{let t=C.lines.flat(),n=w(t,e);return{vertices:Float32Array.from(t.flatMap(([e,t])=>n(e,t)))}},build(e,t,n){let r=Math.round(n.lineCount),i=r*F,a=e.allocate(`tolerance`,`float32`,1),o=e.allocate(`piece-offsets`,`uint32`,513),s=e.allocate(`piece-positions`,`float32x2`,P),c=e.allocate(`piece-sources`,`uint32`,N),l=e.allocate(`piece-count`,`uint32`,1),u=e.allocate(`piece-overflow`,`uint32`,1),d=e.allocate(`node-positions`,`float32x2`,M),f=e.allocate(`node-count`,`uint32`,1),p=e.allocate(`from-nodes`,`uint32`,N),m=e.allocate(`to-nodes`,`uint32`,N),h=e.allocate(`lengths`,`float32`,N),g=e.allocate(`offsets`,`uint32`,513),_=e.allocate(`neighbors`,`uint32`,N*2),y=e.allocate(`weights`,`float32`,N*2);e.graph.add(new k({lines:{kind:`lines`,positions:e.upload(`line-positions`,`float32x2`,t.vertices.slice(0,i*2)).view,lineOffsets:e.upload(`line-offsets`,`uint32`,Uint32Array.from({length:r+1},(e,t)=>t*F)).view},intersectionCapacity:1024,tolerance:a.view,output:{pieces:{geometry:{kind:`lines`,positions:s.view,lineOffsets:o.view},sourceIds:c.view,status:{count:l.view,overflow:u.view}},nodes:{positions:d.view,status:{count:f.view}},edges:{fromNodes:p.view,toNodes:m.view,lengths:h.view},topology:{offsets:g.view,neighbors:_.view,weights:y.view}}}));let b=new Float32Array(8*r*4),x=0;for(let e=0;e<r;e++)for(let n=0;n<8;n++){let r=(e*F+n)*2;b.set(t.vertices.subarray(r,r+4),x++*4)}return{outputs:{nodePositions:d,nodeCount:f,pieceCount:l,nodeIds:e.upload(`node-ids`,`uint32`,Uint32Array.from({length:M},(e,t)=>t)),edges:v(e,{id:`noded`,from:p,to:m,fallbackNode:511}),inputLines:e.upload(`input-lines`,`float32x4`,b)},update:e=>{a.write(Float32Array.of(e.tolerance)),d.write(new Float32Array(M*2).fill(-100))}}},rebuildOn:[`lineCount`],interaction:{kind:`reach`,hint:`reach to merge nodes`,rest:({inputs:e,world:t,toUnit:n})=>n(x(e.vertices.subarray(0,j*F*2),A.default,t)),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=Math.round(r.lineCount)*F,o=0,s=1/0;for(let n=0;n<a;n++){let r=Math.hypot(t.vertices[n*2]-e.x,t.vertices[n*2+1]-e.y);r<s&&(s=r,o=n)}let c=y(A,s),l=i(n.nodeCount),u=i(n.pieceCount);return{values:{tolerance:c},glyphs:[{shape:`ring`,at:[t.vertices[o*2],t.vertices[o*2+1]],radius:c,dashed:!0}],readout:`tolerance ${c.toFixed(2)} · ${l?l[0]:0} nodes · ${u?u[0]:0} pieces`}}},draw:({outputs:e,theme:t})=>[{layer:`lines`,context:!0,segments:e.inputLines,radius:1.2},{layer:`segments`,nodes:e.nodePositions,edges:e.edges,values:e.nodeIds,radius:3},{layer:`dots`,positions:e.nodePositions,radius:6,color:[...t.ink,1]}]});export{I as default};
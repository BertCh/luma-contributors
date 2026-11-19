import{r as e,s as t}from"./kernel-C3tfQOn4.js";import{A as n,D as r,E as i,f as a,k as o,v as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{o as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as d,i as f}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as p}from"./define-run-B-brkCe4.js";import{p as m}from"./inputs-CJKKIRKK.js";import{t as h}from"./pick-Bo3t3hjv.js";import{d as g,i as _,n as v,p as y,v as b}from"./roads-DzovamHj.js";import{u as x}from"./display-Bu20g6o5.js";import{Wa as S}from"./entry-preview-D70TdSbK.js";var C=65536,w=1e7,T=class{props;id;nodeCount;maximumExpansions;maximumFrontierSize;constructor(e){this.props=e,this.id=e.id??`alternative-routes`,this.nodeCount=n(this.id,e.topology,{weights:!0}).nodeCount,a(this.id,`nodeCount`,this.nodeCount,1,256),a(this.id,`routeCount`,e.routeCount,1,32),this.maximumFrontierSize=e.maximumFrontierSize??1024,a(this.id,`maximumFrontierSize`,this.maximumFrontierSize,1,C),this.maximumExpansions=e.maximumExpansions??1e5,a(this.id,`maximumExpansions`,this.maximumExpansions,1,w);for(let[t,n]of[[`sources`,e.sources],[`targets`,e.targets]])i(n,[`uint32`],`${this.id} ${t}`);a(this.id,`sources rows`,e.sources.length),r(this.id,`targets`,e.targets,e.sources.length);let t=e.sources.length*e.routeCount;for(let[n,r]of Object.entries({"output path rows":t*this.nodeCount,metadataWords:e.sources.length*(4+e.routeCount*2),frontierWords:e.sources.length*this.maximumFrontierSize*(2+2*this.nodeCount)}))a(this.id,n,r);for(let[n,a]of Object.entries(e.output)){i(a,[n===`costs`?`float32`:`uint32`],`${this.id} output.${n}`);let o=n===`vertices`||n===`edges`?t*this.nodeCount:n===`counts`||n===`complete`?e.sources.length:t;r(this.id,`output.${n}`,a,o)}f(this.id,e.output),l(this.id,Object.values(e.output),[...o(e.topology),e.sources,e.targets])}getCommandNodes(n){let{id:r,props:i,nodeCount:a}=this;c(r,n,[...o(i.topology),i.sources,i.targets,...Object.values(i.output)]);let l=i.sources.length*this.maximumFrontierSize*(2+2*a);d(n,r,Object.fromEntries([...o(i.topology),i.sources,i.targets,...Object.values(i.output)].map((e,t)=>[`view ${t}`,e])));for(let[e,a]of Object.entries({metadata:i.sources.length*(4+i.routeCount*2)*4,frontier:l*4,...Object.fromEntries([...o(i.topology),i.sources,i.targets,...Object.values(i.output)].filter(e=>e!==void 0).map((e,n)=>[`view ${n}`,t(e).size]))}))s(n.device,r,e,a,`reduce query count or maximumFrontierSize`);let f=e(n,`${r}-metadata`,`uint32`,i.sources.length*(4+i.routeCount*2)),p=e(n,`${r}-frontier`,`uint32`,l),m=4+i.routeCount*2,h=`const NODE_COUNT: u32 = ${a}u; const EDGE_COUNT: u32 = ${i.topology.neighbors.length}u; const ROUTE_COUNT: u32 = ${i.routeCount}u; const STRIDE: u32 = ${m}u; const NONE: u32 = 0xffffffffu;`;return[u(n,{id:`${r}-queries`,operation:`GPUAlternativeRoutes`,variant:`queries`,invocationCount:i.sources.length,bindings:[{name:`sources`,view:i.sources,type:`u32`,access:`read`},{name:`targets`,view:i.targets,type:`u32`,access:`read`},{name:`metadata`,view:f,type:`u32`,access:`read_write`}],declarations:h,body:`metadata[metadataOffset + index * STRIDE] = sources[sourcesOffset + index]; metadata[metadataOffset + index * STRIDE + 1u] = targets[targetsOffset + index];`}),u(n,{id:`${r}-search`,operation:`GPUAlternativeRoutes`,variant:`search`,invocationCount:i.sources.length,bindings:[{name:`offsets`,view:i.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.topology.neighbors,type:`u32`,access:`read`},{name:`weights`,view:i.topology.weights,type:`f32`,access:`read`},{name:`metadata`,view:f,type:`u32`,access:`read_write`},{name:`vertices`,view:i.output.vertices,type:`u32`,access:`read_write`},{name:`edges`,view:i.output.edges,type:`u32`,access:`read_write`},{name:`frontier`,view:p,type:`u32`,access:`read_write`}],declarations:h+`
const RECORD_SIZE: u32 = ${2+2*a}u;
const FRONTIER_SIZE: u32 = ${this.maximumFrontierSize}u;
fn precedes(first: u32, second: u32) -> bool {
  let firstCost = bitcast<f32>(frontier[first]);
  let secondCost = bitcast<f32>(frontier[second]);
  if (firstCost != secondCost) { return firstCost < secondCost; }
  let firstLength = frontier[first + 1u]; let secondLength = frontier[second + 1u];
  for (var slot = 0u; slot + 1u < min(firstLength, secondLength); slot++) {
    let firstEdge = frontier[first + 2u + NODE_COUNT + slot];
    let secondEdge = frontier[second + 2u + NODE_COUNT + slot];
    if (firstEdge != secondEdge) { return firstEdge < secondEdge; }
  }
  return firstLength < secondLength;
}
fn swapRecords(first: u32, second: u32) {
  for (var slot = 0u; slot < RECORD_SIZE; slot++) {
    let temporary = frontier[first + slot];
    frontier[first + slot] = frontier[second + slot]; frontier[second + slot] = temporary;
  }
}`,body:`let base = metadataOffset + index * STRIDE;
  let source = metadata[base]; let destination = metadata[base + 1u];
  let pathBase = index * ROUTE_COUNT * NODE_COUNT;
  for (var slot = 0u; slot < ROUTE_COUNT * NODE_COUNT; slot++) { vertices[verticesOffset + pathBase + slot] = NONE; edges[edgesOffset + pathBase + slot] = NONE; }
  for (var route = 0u; route < ROUTE_COUNT; route++) { metadata[base + 4u + route * 2u] = 0u; metadata[base + 5u + route * 2u] = 0x7f800000u; }
  metadata[base + 2u] = 0u; metadata[base + 3u] = 1u;
  if (source >= NODE_COUNT || destination >= NODE_COUNT) { return; }
  var path: array<u32, ${a}>; var pathEdges: array<u32, ${a}>;
  let frontierBase = frontierOffset + index * FRONTIER_SIZE * RECORD_SIZE;
  frontier[frontierBase] = 0u; frontier[frontierBase + 1u] = 1u;
  frontier[frontierBase + 2u] = source;
  var pending = 1u; var count = 0u; var expansions = 0u; var stopped = false;
  loop {
    if (pending == 0u || count == ROUTE_COUNT || stopped) { break; }
    let cost = bitcast<f32>(frontier[frontierBase]);
    let length = frontier[frontierBase + 1u];
    for (var slot = 0u; slot < length; slot++) {
      path[slot] = frontier[frontierBase + 2u + slot];
      if (slot + 1u < length) { pathEdges[slot] = frontier[frontierBase + 2u + NODE_COUNT + slot]; }
    }
    pending--;
    if (pending > 0u) {
      swapRecords(frontierBase, frontierBase + pending * RECORD_SIZE);
      var parent = 0u;
      loop {
        var child = parent * 2u + 1u;
        if (child >= pending) { break; }
        if (child + 1u < pending && precedes(frontierBase + (child + 1u) * RECORD_SIZE, frontierBase + child * RECORD_SIZE)) { child++; }
        if (!precedes(frontierBase + child * RECORD_SIZE, frontierBase + parent * RECORD_SIZE)) { break; }
        swapRecords(frontierBase + child * RECORD_SIZE, frontierBase + parent * RECORD_SIZE); parent = child;
      }
    }
    let vertex = path[length - 1u];
    if (vertex == destination) {
      metadata[base + 4u + count * 2u] = length;
      metadata[base + 5u + count * 2u] = bitcast<u32>(cost);
      for (var slot = 0u; slot < length; slot++) {
        vertices[verticesOffset + pathBase + count * NODE_COUNT + slot] = path[slot];
        if (slot + 1u < length) { edges[edgesOffset + pathBase + count * NODE_COUNT + slot] = pathEdges[slot]; }
      }
      count++; continue;
    }
    for (var edge = min(offsets[offsetsOffset + vertex], EDGE_COUNT); edge < min(offsets[offsetsOffset + vertex + 1u], EDGE_COUNT); edge++) {
      if (expansions >= ${this.maximumExpansions}u) { stopped = true; break; }
      expansions++;
      let neighbor = neighbors[neighborsOffset + edge]; let weight = weights[weightsOffset + edge];
      if (neighbor >= NODE_COUNT || !(weight >= 0.0) || (bitcast<u32>(weight) & 0x7f800000u) == 0x7f800000u) { continue; }
      var repeated = false;
      for (var slot = 0u; slot < length; slot++) { if (path[slot] == neighbor) { repeated = true; break; } }
      if (repeated || length == NODE_COUNT) { continue; }
      let nextCost = cost + weight;
      if ((bitcast<u32>(nextCost) & 0x7f800000u) == 0x7f800000u) { continue; }
      if (pending == FRONTIER_SIZE) { stopped = true; break; }
      let record = frontierBase + pending * RECORD_SIZE;
      frontier[record] = bitcast<u32>(nextCost); frontier[record + 1u] = length + 1u;
      for (var slot = 0u; slot < length; slot++) {
        frontier[record + 2u + slot] = path[slot];
        if (slot + 1u < length) { frontier[record + 2u + NODE_COUNT + slot] = pathEdges[slot]; }
      }
      frontier[record + 2u + length] = neighbor;
      frontier[record + 2u + NODE_COUNT + length - 1u] = edge;
      var child = pending; pending++;
      loop {
        if (child == 0u) { break; }
        let parent = (child - 1u) / 2u;
        if (!precedes(frontierBase + child * RECORD_SIZE, frontierBase + parent * RECORD_SIZE)) { break; }
        swapRecords(frontierBase + child * RECORD_SIZE, frontierBase + parent * RECORD_SIZE); child = parent;
      }
    }
  }
  if (stopped) { metadata[base + 3u] = 0u; }
  metadata[base + 2u] = count;`}),u(n,{id:`${r}-publish`,operation:`GPUAlternativeRoutes`,variant:`publish`,invocationCount:i.sources.length,bindings:[{name:`metadata`,view:f,type:`u32`,access:`read`},{name:`lengths`,view:i.output.lengths,type:`u32`,access:`read_write`},{name:`costs`,view:i.output.costs,type:`f32`,access:`read_write`},{name:`counts`,view:i.output.counts,type:`u32`,access:`read_write`},{name:`complete`,view:i.output.complete,type:`u32`,access:`read_write`}],declarations:h,body:`let base = metadataOffset + index * STRIDE;
  counts[countsOffset + index] = metadata[base + 2u]; complete[completeOffset + index] = metadata[base + 3u];
  for (var route = 0u; route < ROUTE_COUNT; route++) {
    lengths[lengthsOffset + index * ROUTE_COUNT + route] = metadata[base + 4u + route * 2u];
    costs[costsOffset + index * ROUTE_COUNT + route] = bitcast<f32>(metadata[base + 5u + route * 2u]);
  }`})]}},E=[[1,.78,.25,1],[.35,.8,1,1],[1,.4,.6,1],[.6,1,.5,1]],D=p({parameters:S,caption:`The k cheapest simple routes between two intersections, one color per route.`,createInputs:({random:e,world:t})=>{let n=_(e,t,7),r=n.findNearestNode(t.width*.7,.75);return{network:n,source:n.findNearestNode(t.width*.3,.25),defaultTarget:r,target:r,index:v(n),costs:new Float32Array(n.slotCount)}},build(e,t,n){let{network:r}=t,i=Math.round(n.routeCount),{topology:a,weights:o}=m(e,r),s=e.allocate(`vertices`,`uint32`,i*r.nodeCount),c=e.allocate(`edges`,`uint32`,i*r.nodeCount),l=e.allocate(`lengths`,`uint32`,i),u=e.allocate(`costs`,`float32`,i),d=e.allocate(`counts`,`uint32`,1),f=e.allocate(`complete`,`uint32`,1),p=e.allocate(`targets`,`uint32`,1),h=e.allocate(`target-position`,`float32x2`,1);e.graph.add(new T({topology:a,sources:e.upload(`sources`,`uint32`,Uint32Array.of(t.source)).view,targets:p.view,routeCount:i,maximumExpansions:1e6,maximumFrontierSize:8192,output:{vertices:s.view,edges:c.view,lengths:l.view,costs:u.view,counts:d.view,complete:f.view}}));let g=e.upload(`node-positions`,`float32`,r.nodes),_=Array.from({length:i},(t,n)=>x(e,{id:`route-${n}`,vertices:s,nodePositions:g,routeIndex:n,stride:r.nodeCount}));return{outputs:{...b(e,r),routes:_,costs:u,source:e.upload(`source-position`,`float32x2`,Float32Array.of(r.nodes[t.source*2],r.nodes[t.source*2+1])),target:h},update:e=>{p.write(Uint32Array.of(t.target)),h.write(Float32Array.of(r.nodes[t.target*2],r.nodes[t.target*2+1])),r.fillWeights(t.costs,e.congestion),o.write(t.costs)}}},rebuildOn:[`routeCount`],interaction:{kind:`place`,hint:`place the target`,rest:({inputs:e,toUnit:t})=>t([e.network.nodes[e.defaultTarget*2],e.network.nodes[e.defaultTarget*2+1]]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=h(t.index,e.x,e.y,t.target),o=a!==t.target;t.target=a;let s=i(n.costs),c=s?Array.from(s.subarray(0,Math.round(r.routeCount)),e=>Number.isFinite(e)&&e<0x56bc75e2d63100000?e.toFixed(2):`-`).join(` / `):``;return{rerun:o,glyphs:[{shape:`crosshair`,at:[t.network.nodes[a*2],t.network.nodes[a*2+1]],size:14}],readout:`routes ${Math.round(r.routeCount)} · costs ${c}`}}},draw:({inputs:e,outputs:t,theme:n})=>[y(t),...t.routes.map((t,n)=>({layer:`lines`,segments:t,count:e.network.nodeCount-1,radius:4.6-n*.9,color:E[n]})).reverse(),...g(t.source,n,{colorIndex:4}),...g(t.target,n,{colorIndex:2})]});export{D as default};
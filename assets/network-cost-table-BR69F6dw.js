import{o as e,s as t}from"./network-view-utils-B1U8oNKx.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./status-aOwAAXek.js";import{A as i,E as a,T as o,f as s,k as c,p as l,v as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{c as f}from"./wgsl-common-DfyXiYB0.js";import{o as p}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as m,t as h}from"./gpu-network-reachability-Coypydjh.js";import{t as g}from"./network-accessibility-lanes-D6-5poNo.js";import{t as _}from"./define-run-B-brkCe4.js";import{p as v,s as y}from"./inputs-CJKKIRKK.js";import{t as b}from"./pick-Bo3t3hjv.js";import{m as x,n as S,p as C,v as w}from"./roads-DzovamHj.js";import{ka as T}from"./entry-preview-D70TdSbK.js";var E=`GPUNetworkCostTable`,D=16,O=64,k=256,A=4294967295,j=4;function M(e,t){s(e,`maximumIterations`,t.maximumIterations??O,1,m),s(e,`localIterations`,t.localIterations??D,1,64)}var N=class{id;props;nodeCount;laneCount;batchCount;capacity;estimatedCommandNodeCount;constructor(e){let n=e.topology,c=e.output,u=c.status;e=Object.freeze({id:e.id,costLimit:e.costLimit,laneCount:e.laneCount,maximumIterations:e.maximumIterations,localIterations:e.localIterations,topology:Object.freeze({offsets:n.offsets,neighbors:n.neighbors,weights:n.weights,nodePositions:n.nodePositions}),output:Object.freeze({offsets:c.offsets,targets:c.targets,costs:c.costs,status:u?Object.freeze({complete:u.complete,overflow:u.overflow,requiredCount:u.requiredCount,converged:u.converged,iterationLimitReached:u.iterationLimitReached}):void 0})});let f=e.id??`network-cost-table`;M(f,e),this.id=f,this.props=e,t(this.id,e.output,[{outputs:{offsets:e.output.offsets}},{outputs:{targets:e.output.targets,costs:e.output.costs}},{outputs:{status:e.output.status}}]);let{output:p}=e,{nodeCount:m,edgeCount:h}=i(f,e.topology,{weights:!0});if(this.nodeCount=m,a(e.costLimit,[`float32`],`${f} costLimit`),e.costLimit.length!==1)throw Error(`${f} costLimit must contain exactly one row`);if(a(p.offsets,[`uint32`],`${f} output.offsets`),p.offsets.length!==m+1)throw Error(`${f} output.offsets must contain nodeCount + 1 rows`);if(a(p.targets,[`uint32`],`${f} output.targets`),a(p.costs,[`float32`],`${f} output.costs`),this.capacity=p.targets.length,s(f,`output.targets length`,this.capacity),p.costs.length!==this.capacity)throw Error(`${f} output.costs length must equal output.targets length`);r(f,{overflow:p.status?.overflow,requiredCount:p.status?.requiredCount});let _=p.status?.converged,v=p.status?.iterationLimitReached,y=p.status?.complete;for(let[e,t]of[[`overflow`,p.status?.overflow],[`requiredCount`,p.status?.requiredCount],[`converged`,_],[`iterationLimitReached`,v],[`complete`,y]])t&&(a(t,[`uint32`],`${f} output.status.${e}`),o(f,t.length===1,`output.status.${e} must contain exactly one row`));if(this.laneCount=e.laneCount??g({rowCount:m,nodeCount:m,edgeCount:h}),s(f,`laneCount`,this.laneCount,1,m),this.laneCount*m>=A)throw Error(`${f} laneCount * nodeCount must fit in uint32`);this.batchCount=Math.ceil(m/this.laneCount);let b=2+(e.maximumIterations??O)+Number(!!(_||v||y));this.estimatedCommandNodeCount=2+this.batchCount*(4+b),l(f,this.estimatedCommandNodeCount);let x=F(e);d(f,x,P(e))}getCommandNodes(t){let{id:r,props:i,nodeCount:a,laneCount:o,batchCount:s,capacity:c}=this,{output:l}=i,{offsets:d,neighbors:m,weights:g}=i.topology;e(r,t,[...P(i),...F(i)]);let _=o*a;u(t.device,r,`frontier queues`,2*_*4,`lower laneCount`);let v=l.status?.converged,y=l.status?.iterationLimitReached;e(r,t,[],{state:16,lanes:o*4,frontier:o*a*8});let b=n(t,`${r}-state`,`uint32`,j),x=n(t,`${r}-seeds`,`uint32`,o),S=n(t,`${r}-scratch`,`float32`,_),C=n(t,`${r}-lane-counts`,`uint32`,o),w=n(t,`${r}-lane-bases`,`uint32`,o),T=v||y||l.status?.complete?n(t,`${r}-batch-converged`,`uint32`,1):void 0,O=`const NODE_COUNT: u32 = ${a}u;
const LANE_COUNT: u32 = ${o}u;
const CAPACITY: u32 = ${c}u;
const NO_START: u32 = ${A}u;
// state[0]: entries written so far (unclamped, saturating); state[1]: first row of the next
// batch; state[2]: all batches converged so far.
const ENTRY_CURSOR: u32 = 0u;
const ROW_CURSOR: u32 = 1u;
const CONVERGED_FLAG: u32 = 2u;
${f}
fn addSaturated(left: u32, right: u32) -> u32 {
  return select(0xffffffffu, left + right, left <= 0xffffffffu - right);
}
fn isTableEntry(cost: f32) -> bool {
  // Unreached entries are +Infinity.
  return isFiniteBits(cost) && cost >= 0.0;
}`,M=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),N=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),I=[p(t,{id:`${r}-init`,operation:E,variant:`init`,bindings:[N(`state`,b)],invocationCount:1,body:`state[stateOffset + 0u] = 0u;
  state[stateOffset + 1u] = 0u;
  state[stateOffset + 2u] = 1u;
  state[stateOffset + 3u] = 0u;`})];for(let e=0;e<s;e++)I.push(p(t,{id:`${r}-batch-${e}-seeds`,operation:E,variant:`seeds`,bindings:[M(`state`,b),N(`seeds`,x)],invocationCount:o,declarations:O,body:`let row = state[stateOffset + ROW_CURSOR] + index;
  seeds[seedsOffset + index] = select(NO_START, index * NODE_COUNT + row, row < NODE_COUNT);`}),...new h({id:`${r}-batch-${e}`,topology:{offsets:d,neighbors:m,weights:g},laneCount:o,startNodes:x,costLimit:i.costLimit,maximumIterations:i.maximumIterations,localIterations:i.localIterations??D,output:{costs:S,status:{converged:T}}}).getCommandNodes(t),p(t,{id:`${r}-batch-${e}-count`,operation:E,variant:`count`,bindings:[M(`scratch`,S,`f32`),N(`laneCounts`,C)],workgroupSize:k,invocationCount:o*k,guardIndex:!1,declarations:`${O}
var<workgroup> partials: array<u32, ${k}>;`,body:`let lane = index / ${k}u;
  var count = 0u;
  for (var node = localInvocationIndex; node < NODE_COUNT; node += ${k}u) {
    if (isTableEntry(scratch[scratchOffset + lane * NODE_COUNT + node])) {
      count += 1u;
    }
  }
  partials[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = ${k/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    laneCounts[laneCountsOffset + lane] = partials[0];
  }`}),p(t,{id:`${r}-batch-${e}-scan`,operation:E,variant:`scan`,bindings:[N(`state`,b),M(`laneCounts`,C),N(`laneBases`,w),N(`offsets`,l.offsets),...T?[M(`batchConverged`,T)]:[]],workgroupSize:k,invocationCount:k,guardIndex:!1,declarations:`${O}
var<workgroup> prefix: array<u32, ${k}>;`,body:`let firstRow = state[stateOffset + ROW_CURSOR];
  var base = state[stateOffset + ENTRY_CURSOR];
  for (var chunk = 0u; chunk < LANE_COUNT; chunk += ${k}u) {
    let lane = chunk + localInvocationIndex;
    var count = 0u;
    if (lane < LANE_COUNT) {
      count = laneCounts[laneCountsOffset + lane];
    }
    prefix[localInvocationIndex] = count;
    workgroupBarrier();
    for (var stride = 1u; stride < ${k}u; stride = stride * 2u) {
      var addend = 0u;
      if (localInvocationIndex >= stride) {
        addend = prefix[localInvocationIndex - stride];
      }
      workgroupBarrier();
      prefix[localInvocationIndex] += addend;
      workgroupBarrier();
    }
    if (lane < LANE_COUNT) {
      let start = addSaturated(base, prefix[localInvocationIndex] - count);
      laneBases[laneBasesOffset + lane] = start;
      if (firstRow + lane < NODE_COUNT) {
        offsets[offsetsOffset + firstRow + lane] = min(start, CAPACITY);
      }
    }
    base = addSaturated(base, prefix[255u]);
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    state[stateOffset + ENTRY_CURSOR] = base;
    state[stateOffset + ROW_CURSOR] = firstRow + LANE_COUNT;
    // The last batch's write wins, and it holds the total.
    offsets[offsetsOffset + NODE_COUNT] = min(base, CAPACITY);
    ${T?`state[stateOffset + CONVERGED_FLAG] = min(state[stateOffset + CONVERGED_FLAG], batchConverged[batchConvergedOffset]);`:``}
  }`}),p(t,{id:`${r}-batch-${e}-fill`,operation:E,variant:`fill`,bindings:[M(`scratch`,S,`f32`),M(`laneBases`,w),N(`targets`,l.targets),N(`costs`,l.costs,`f32`)],workgroupSize:k,invocationCount:o*k,guardIndex:!1,declarations:`${O}
var<workgroup> prefix: array<u32, ${k}>;`,body:`let lane = index / ${k}u;
  var base = laneBases[laneBasesOffset + lane];
  for (var chunk = 0u; chunk < NODE_COUNT; chunk += ${k}u) {
    let node = chunk + localInvocationIndex;
    var cost = -1.0;
    if (node < NODE_COUNT) {
      cost = scratch[scratchOffset + lane * NODE_COUNT + node];
    }
    let isIncluded = isTableEntry(cost);
    prefix[localInvocationIndex] = select(0u, 1u, isIncluded);
    workgroupBarrier();
    for (var stride = 1u; stride < ${k}u; stride = stride * 2u) {
      var addend = 0u;
      if (localInvocationIndex >= stride) {
        addend = prefix[localInvocationIndex - stride];
      }
      workgroupBarrier();
      prefix[localInvocationIndex] += addend;
      workgroupBarrier();
    }
    let rank = prefix[localInvocationIndex] - 1u;
    // The base saturates, so test the slot without adding past the capacity.
    if (isIncluded && base < CAPACITY && rank < CAPACITY - base) {
      targets[targetsOffset + base + rank] = node;
      costs[costsOffset + base + rank] = cost;
    }
    base = addSaturated(base, prefix[255u]);
    workgroupBarrier();
  }`}));let L=[M(`state`,b)];return l.status?.overflow&&L.push(N(`overflow`,l.status.overflow)),l.status?.requiredCount&&L.push(N(`requiredCount`,l.status.requiredCount)),v&&L.push(N(`converged`,v)),y&&L.push(N(`iterationLimitReached`,y)),l.status?.complete&&L.push(N(`complete`,l.status.complete)),I.push(p(t,{id:`${r}-finalize`,operation:E,variant:`finalize`,bindings:L,invocationCount:1,declarations:O,body:`let total = state[stateOffset + ENTRY_CURSOR];
  ${l.status?.overflow?`overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);`:``}
  ${l.status?.requiredCount?`requiredCount[requiredCountOffset] = total;`:``}
  ${v?`converged[convergedOffset] = state[stateOffset + CONVERGED_FLAG];`:``}
  ${l.status?.complete?`complete[completeOffset] = select(0u, 1u, state[stateOffset + CONVERGED_FLAG] != 0u && total <= CAPACITY);`:``}
  ${y?`iterationLimitReached[iterationLimitReachedOffset] = 1u - state[stateOffset + CONVERGED_FLAG];`:``}`})),I}};function P(e){return[...c(e.topology),e.costLimit]}function F(e){let{output:t}=e;return[t.offsets,t.targets,t.costs,t.status?.overflow,t.status?.requiredCount,t.status?.converged,t.status?.iterationLimitReached,t.status?.complete]}var I=3e3,L=_({parameters:T,caption:`The cost table row of one origin, drawn as the cost to each intersection within the radius.`,createInputs:({random:e,world:t})=>{let n=y(e,t,9),r=n.findNearestNode(t.width/2,.5);return{network:n,defaultNode:r,node:r,index:S(n),costs:new Float32Array(n.slotCount)}},build(e,t){let{network:n}=t,{topology:r,weights:i}=v(e,n),a=e.allocate(`cost-limit`,`float32`,1),o=e.allocate(`offsets`,`uint32`,n.nodeCount+1),s=e.allocate(`targets`,`uint32`,I),c=e.allocate(`costs`,`float32`,I),l=e.allocate(`complete`,`uint32`,1);return e.graph.add(new N({topology:r,costLimit:a.view,maximumIterations:48,output:{offsets:o.view,targets:s.view,costs:c.view,status:{complete:l.view}}})),{outputs:{...w(e,n),offsets:o,targets:s,costs:c,complete:l,rowCosts:e.upload(`row-costs`,`float32`,new Float32Array(n.nodeCount).fill(NaN))},update:e=>{n.fillWeights(t.costs,e.congestion),i.write(t.costs),c.write(new Float32Array(I)),a.write(Float32Array.of(e.costLimit))}}},interaction:{kind:`probe`,hint:`hover an origin`,rest:({inputs:e,toUnit:t})=>t([e.network.nodes[e.defaultNode*2],e.network.nodes[e.defaultNode*2+1]]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){t.node=b(t.index,e.x,e.y,t.node);let a=t.node,o=i(n.offsets),s=i(n.targets),c=i(n.costs),l=(i(n.complete)?.[0]??0)!==0,u=new Float32Array(t.network.nodeCount).fill(NaN),d=0;if(l&&o&&s&&c)for(let e=o[a];e<o[a+1];e++)u[s[e]]=c[e],d++;return n.rowCosts.write(u),{glyphs:[{shape:`crosshair`,at:[t.network.nodes[a*2],t.network.nodes[a*2+1]],size:14},{shape:`ring`,at:[t.network.nodes[a*2],t.network.nodes[a*2+1]],radius:r.costLimit,dashed:!0}],readout:l&&o&&s&&c?`node ${a} · ${d} targets within ${r.costLimit.toFixed(2)}`:`node ${a} · incomplete table`}}},draw:({outputs:e,values:t})=>[C(e),x(e,e.rowCosts,{radius:2.4,range:[0,t.costLimit]})]});export{L as default};
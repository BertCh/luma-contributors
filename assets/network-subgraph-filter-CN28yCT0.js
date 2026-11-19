import{o as e,s as t}from"./network-view-utils-B1U8oNKx.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{n as i,o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{A as s,E as c,k as l,m as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{t as f}from"./time-words-BFl8KCxq.js";import{t as p}from"./network-workgroup-counters-BG1Cyc9s.js";import{t as m}from"./define-run-B-brkCe4.js";import{d as h,o as g,s as _}from"./pointer-overlay-Qh8qflcA.js";import{_ as v,f as y,h as b,i as x,p as S,r as C,v as w,y as T}from"./roads-DzovamHj.js";import{ua as E}from"./entry-preview-D70TdSbK.js";var D={liveNodeCount:0,liveSlotCount:1,selfLoopSlotCount:2,liveEdgeCount:3};function O(e){return(e.nodeColumnCount+e.edgeColumnCount+ +!!e.hasEdgeTimes)*4}function k(e,t={},n=new Float32Array(O(e))){if(n.length<O(e))throw Error(`GPUNetworkSubgraphFilter parameter target is too short`);n.fill(0,0,O(e));let r=[];for(let n=0;n<e.nodeColumnCount;n++)r.push(t.nodeRanges?.[n]);for(let n=0;n<e.edgeColumnCount;n++)r.push(t.edgeRanges?.[n]);e.hasEdgeTimes&&r.push(t.edgeTimeWindow);for(let[e,t]of r.entries())t&&(n[e*4]=t[0],n[e*4+1]=t[1],n[e*4+2]=1);return n}var A=8,j=6,M=`GPUNetworkSubgraphFilter`,N=class{id;props;nodeCount;slotCount;constructor(e){this.id=e.id??`network-subgraph-filter`,this.props=e;let n=e.output;t(this.id,n,[{outputs:{edgeMask:n.edgeMask},inputs:{nodeMask:n.nodeMask}},{outputs:{counts:n.counts},inputs:{nodeMask:n.nodeMask,edgeMask:n.edgeMask}},{outputs:{ids:n.liveNodes?.ids},inputs:{nodeMask:n.nodeMask}},{outputs:{ids:n.liveEdgeSlots?.ids},inputs:{edgeMask:n.edgeMask}},{outputs:{status:n.liveNodes?.status}},{outputs:{status:n.liveEdgeSlots?.status}},{outputs:{offsets:n.inducedTopology?.offsets,overflow:n.inducedTopology?.status.overflow},inputs:{edgeMask:n.edgeMask}},{outputs:{neighbors:n.inducedTopology?.neighbors,sourceSlots:n.inducedTopology?.sourceSlots},inputs:{edgeMask:n.edgeMask}}]);let{id:r}=this,i=s(r,e.topology);this.nodeCount=i.nodeCount,this.slotCount=i.edgeCount;let{nodeCount:a,slotCount:o}=this;if(o<1)throw Error(`${r} topology.neighbors must contain at least one row`);let l=e.nodeColumns??[],f=e.edgeColumns??[];if(l.length+f.length>64)throw Error(`${r} supports at most 64 columns`);let p=[[`nodeMask`,e.nodeMask,a,`node`],[`edgeMask`,e.edgeMask,o,`neighbor slot`],[`edgeTimes`,e.edgeTimes,o,`neighbor slot`],[`edgeTimeWords`,e.edgeTimeWords,o,`neighbor slot`],[`output.nodeMask`,e.output.nodeMask,a,`node`],[`output.edgeMask`,e.output.edgeMask,o,`neighbor slot`]];for(let[e,t]of l.entries())c(t,[`float32`],`${r} nodeColumns[${e}]`),p.push([`nodeColumns[${e}]`,t,a,`node`]);for(let[e,t]of f.entries())c(t,[`float32`],`${r} edgeColumns[${e}]`),p.push([`edgeColumns[${e}]`,t,o,`neighbor slot`]);for(let[e,t,n,i]of p)if(t&&(e===`edgeTimeWords`?c(t,[`uint32x2`],`${r} ${e}`):e===`edgeTimes`?c(t,[`float32`],`${r} ${e}`):!e.startsWith(`nodeColumns`)&&!e.startsWith(`edgeColumns`)&&c(t,[`uint32`],`${r} ${e}`),t.length!==n))throw Error(`${r} ${e} must contain one row per ${i}`);let m=O({nodeColumnCount:l.length,edgeColumnCount:f.length,hasEdgeTimes:!!e.edgeTimes});if(m>0){if(!e.parameters)throw Error(`${r} parameters are required with columns or edgeTimes`);if(c(e.parameters,[`float32`],`${r} parameters`),e.parameters.length<m)throw Error(`${r} parameters must contain at least ${m} float32 rows`)}if(e.edgeTimeWords){if(!e.uint32Parameters)throw Error(`${r} uint32Parameters are required with edgeTimeWords`);if(c(e.uint32Parameters,[`uint32`],`${r} uint32Parameters`),e.uint32Parameters.length<A)throw Error(`${r} uint32Parameters must contain at least 8 uint32 rows`)}let{output:h}=e;if(h.counts&&(c(h.counts,[`uint32`],`${r} output.counts`),h.counts.length<4))throw Error(`${r} output.counts must contain at least 4 uint32 rows`);for(let[e,t]of[[`liveNodes`,h.liveNodes],[`liveEdgeSlots`,h.liveEdgeSlots]])t&&u(`${r} output.${e}`,t);let g=h.inducedTopology;if(g){if(c(g.offsets,[`uint32`],`${r} output.inducedTopology.offsets`),c(g.neighbors,[`uint32`],`${r} output.inducedTopology.neighbors`),c(g.status.overflow,[`uint32`],`${r} output.inducedTopology.status.overflow`),g.offsets.length!==a+1)throw Error(`${r} output.inducedTopology.offsets must contain nodeCount + 1 rows`);if(g.status.overflow.length<1)throw Error(`${r} output.inducedTopology.status.overflow must contain one uint32 row`);if(g.sourceSlots&&(c(g.sourceSlots,[`uint32`],`${r} output.inducedTopology.sourceSlots`),g.sourceSlots.length!==g.neighbors.length))throw Error(`${r} output.inducedTopology.sourceSlots must match neighbors in length`)}d(r,this.getOutputViews(),this.getInputViews())}getInputViews(){let{props:e}=this;return[...l(e.topology),e.nodeMask,e.edgeMask,...e.nodeColumns??[],...e.edgeColumns??[],e.edgeTimes,e.edgeTimeWords,e.parameters,e.uint32Parameters]}getOutputViews(){let{output:e}=this.props;return[e.nodeMask,e.edgeMask,e.counts,e.liveNodes?.ids,e.liveNodes?.status.count,e.liveNodes?.status.overflow,e.liveNodes?.status.requiredCount,e.liveEdgeSlots?.ids,e.liveEdgeSlots?.status.count,e.liveEdgeSlots?.status.overflow,e.liveEdgeSlots?.status.requiredCount,e.inducedTopology?.offsets,e.inducedTopology?.neighbors,e.inducedTopology?.sourceSlots,e.inducedTopology?.status.overflow]}getCommandNodes(t){let{id:s,props:c,nodeCount:l,slotCount:u}=this;e(s,t,[...this.getInputViews(),...this.getOutputViews()]),e(s,t,[],{nodeScratch:l*4,slotScratch:u*4});let{output:d}=c,m=c.nodeColumns??[],h=c.edgeColumns??[],g=[],_=D,v=(e,n,r,i,o,c)=>g.push(a(t,{id:`${s}-${e}`,operation:M,variant:n,bindings:r,invocationCount:i,body:o,declarations:c})),y=(e,t,n,r,i)=>{for(let a=0;a<n.length;a+=j){let o=n.slice(a,a+j),s=[{name:`maskOut`,view:t,type:`u32`,access:`read_write`},{name:`parameters`,view:c.parameters,type:`f32`,access:`read`},...o.map((e,t)=>({name:`column${t}`,view:e,type:`f32`,access:`read`}))],l=o.map((e,t)=>`{
    let base = parametersOffset + ${(r+a+t)*4}u;
    if (parameters[base + 2u] != 0.0) {
      let value = column${t}[column${t}Offset + index];
      live = live && value >= parameters[base] && value < parameters[base + 1u];
    }
  }`).join(`
  `);v(`${e}-columns-${a/j}`,e,s,i,`var live = maskOut[maskOutOffset + index] != 0u;
  ${l}
  maskOut[maskOutOffset + index] = select(0u, 1u, live);`)}};v(`vertex-init`,`vertex-init`,[{name:`maskOut`,view:d.nodeMask,type:`u32`,access:`read_write`},...c.nodeMask?[{name:`maskIn`,view:c.nodeMask,type:`u32`,access:`read`}]:[]],l,c.nodeMask?`maskOut[maskOutOffset + index] = select(0u, 1u, maskIn[maskInOffset + index] != 0u);`:`maskOut[maskOutOffset + index] = 1u;`),y(`vertex`,d.nodeMask,m,0,l);let b=!c.directed&&c.pairUndirectedSlots!==!1,x=b?n(t,`${s}-own-edge-mask`,`uint32`,u):d.edgeMask;if(v(`edge-init`,`edge-init`,[{name:`maskOut`,view:x,type:`u32`,access:`read_write`},...c.edgeMask?[{name:`maskIn`,view:c.edgeMask,type:`u32`,access:`read`}]:[]],u,c.edgeMask?`maskOut[maskOutOffset + index] = select(0u, 1u, maskIn[maskInOffset + index] != 0u);`:`maskOut[maskOutOffset + index] = 1u;`),y(`edge`,x,h,m.length,u),c.edgeTimes){let e=(m.length+h.length)*4;v(`edge-times`,`edge-times`,[{name:`maskOut`,view:x,type:`u32`,access:`read_write`},{name:`parameters`,view:c.parameters,type:`f32`,access:`read`},{name:`times`,view:c.edgeTimes,type:`f32`,access:`read`}],u,`let base = parametersOffset + ${e}u;
  if (parameters[base + 2u] != 0.0) {
    let time = times[timesOffset + index];
    let live = time >= parameters[base] && time <= parameters[base + 1u];
    if (!live) {
      maskOut[maskOutOffset + index] = 0u;
    }
  }`)}c.edgeTimeWords&&v(`edge-time-words`,`edge-time-words`,[{name:`maskOut`,view:x,type:`u32`,access:`read_write`},{name:`timeWindow`,view:c.uint32Parameters,type:`u32`,access:`read`},{name:`timeWords`,view:c.edgeTimeWords,type:`u32`,access:`read`}],u,`let windowStart = vec2<u32>(timeWindow[timeWindowOffset], timeWindow[timeWindowOffset + 1u]);
  let windowEnd = vec2<u32>(timeWindow[timeWindowOffset + 2u], timeWindow[timeWindowOffset + 3u]);
  let startFraction = bitcast<f32>(timeWindow[timeWindowOffset + 4u]);
  let endFraction = bitcast<f32>(timeWindow[timeWindowOffset + 5u]);
  let time = vec2<u32>(
    timeWords[timeWordsOffset + 2u * index],
    timeWords[timeWordsOffset + 2u * index + 1u]
  );
  let live = isTimeWordsAtLeast(time, 0.0, windowStart, startFraction) &&
    isTimeWordsAtLeast(windowEnd, endFraction, time, 0.0);
  if (!live) {
    maskOut[maskOutOffset + index] = 0u;
  }`,f);let S=b?`var live = sourceLive && ownMask[ownMaskOffset + slot] != 0u && neighbor < NODE_COUNT;
    if (live && neighbor != row) {
      // The j-th slot row -> neighbor pairs with the j-th slot neighbor -> row. A missing
      // reverse slot leaves the slot's own result in force.
      var ordinal = 0u;
      for (var earlier = rowBegin; earlier < slot; earlier++) {
        ordinal += select(0u, 1u, neighbors[neighborsOffset + earlier] == neighbor);
      }
      let reverseEnd = min(offsets[offsetsOffset + neighbor + 1u], SLOT_COUNT);
      var seen = 0u;
      for (var reverse = offsets[offsetsOffset + neighbor]; reverse < reverseEnd; reverse++) {
        if (neighbors[neighborsOffset + reverse] == row) {
          if (seen == ordinal) {
            live = ownMask[ownMaskOffset + reverse] != 0u;
            break;
          }
          seen++;
        }
      }
    }`:`var live = sourceLive && edgeMask[edgeMaskOffset + slot] != 0u && neighbor < NODE_COUNT;`;if(v(`edge-endpoints`,`edge-endpoints`,[{name:`offsets`,view:c.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:c.topology.neighbors,type:`u32`,access:`read`},{name:`vertexMask`,view:d.nodeMask,type:`u32`,access:`read`},...b?[{name:`ownMask`,view:x,type:`u32`,access:`read`}]:[],{name:`edgeMask`,view:d.edgeMask,type:`u32`,access:`read_write`}],u,`let slot = index;
  // Owner row: the last row whose offset is at or before the slot.
  var low = 0u;
  var high = NODE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= slot) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let row = low;
  let rowBegin = offsets[offsetsOffset + row];
  let rowEnd = min(offsets[offsetsOffset + row + 1u], SLOT_COUNT);
  if (slot < rowBegin || slot >= rowEnd) {
    // A slot past offsets[NODE_COUNT] belongs to no row. The edge-init pass wrote 1 there (or the
    // caller's mask), so it must be cleared here or it reads as a live edge (review B-09).
    edgeMask[edgeMaskOffset + slot] = 0u;
    return;
  }
  let sourceLive = vertexMask[vertexMaskOffset + row] != 0u;
  let neighbor = neighbors[neighborsOffset + slot];
  ${S}
  if (live) {
    live = vertexMask[vertexMaskOffset + neighbor] != 0u;
  }
  edgeMask[edgeMaskOffset + slot] = select(0u, 1u, live);`,`const NODE_COUNT: u32 = ${l}u;
const SLOT_COUNT: u32 = ${u}u;`),c.dropIsolated){let e=n(t,`${s}-incident`,`uint32`,l);g.push(o(t,{id:`${s}-zero-incident`,operation:M,view:e,type:`u32`,value:`0u`})),v(`mark-incident`,`mark-incident`,[{name:`offsets`,view:c.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:c.topology.neighbors,type:`u32`,access:`read`},{name:`edgeMask`,view:d.edgeMask,type:`u32`,access:`read`},{name:`incident`,view:e,type:`atomic<u32>`,access:`read_write`}],l,`let rowBegin = offsets[offsetsOffset + index];
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    if (edgeMask[edgeMaskOffset + slot] != 0u) {
      atomicStore(&incident[incidentOffset + index], 1u);
      atomicStore(&incident[incidentOffset + neighbors[neighborsOffset + slot]], 1u);
    }
  }`,`const SLOT_COUNT: u32 = ${u}u;`),v(`drop-isolated`,`drop-isolated`,[{name:`vertexMask`,view:d.nodeMask,type:`u32`,access:`read_write`},{name:`incident`,view:e,type:`u32`,access:`read`}],l,`if (incident[incidentOffset + index] == 0u) {
    vertexMask[vertexMaskOffset + index] = 0u;
  }`)}if(d.counts){g.push(o(t,{id:`${s}-zero-counts`,operation:M,view:d.counts,type:`u32`,value:`0u`,componentCount:4}));let e=p({counters:[{target:`counts`,word:_.liveNodeCount},{target:`counts`,word:_.liveSlotCount},{target:`counts`,word:_.selfLoopSlotCount}],rowCount:`NODE_COUNT`,rowSource:`if (vertexMask[vertexMaskOffset + index] != 0u) {
    addCounter(0u, 1u);
  }
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  let rowBegin = min(offsets[offsetsOffset + index], rowEnd);
  var liveCount = 0u;
  var selfCount = 0u;
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    if (edgeMask[edgeMaskOffset + slot] != 0u) {
      liveCount++;
      if (neighbors[neighborsOffset + slot] == index) {
        selfCount++;
      }
    }
  }
  addCounter(1u, liveCount);
  addCounter(2u, selfCount);`});g.push(a(t,{id:`${s}-counts`,operation:M,variant:`counts`,bindings:[{name:`offsets`,view:c.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:c.topology.neighbors,type:`u32`,access:`read`},{name:`vertexMask`,view:d.nodeMask,type:`u32`,access:`read`},{name:`edgeMask`,view:d.edgeMask,type:`u32`,access:`read`},{name:`counts`,view:d.counts,type:`atomic<u32>`,access:`read_write`}],invocationCount:l,workgroupSize:256,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${l}u;
const SLOT_COUNT: u32 = ${u}u;
${e.declarations}`,body:e.body})),v(`counts-finish`,`counts-finish`,[{name:`counts`,view:d.counts,type:`u32`,access:`read_write`}],1,`let slots = counts[countsOffset + ${_.liveSlotCount}u];
  let selfLoops = counts[countsOffset + ${_.selfLoopSlotCount}u];
  counts[countsOffset + ${_.liveEdgeCount}u] =
    ${c.directed?`slots`:`(slots - selfLoops) / 2u + selfLoops`};`)}let C=(e,a,o,c=n(t,`${s}-${e}-offsets`,`uint32`,a.length))=>{let l=a.length,u=n(t,`${s}-${e}-total`,`uint32`,1);g.push(...new r({id:`${s}-${e}-scan`,input:a,output:c}).getCommandNodes(t)),v(`${e}-scatter`,`scatter`,[{name:`flags`,view:a,type:`u32`,access:`read`},{name:`prefix`,view:c,type:`u32`,access:`read`},{name:`idsOut`,view:o.ids,type:`u32`,access:`read_write`},{name:`totalOut`,view:u,type:`u32`,access:`read_write`}],l,`let live = flags[flagsOffset + index] != 0u;
  let position = prefix[prefixOffset + index];
  if (live && position < CAPACITY) {
    idsOut[idsOutOffset + position] = index;
  }
  if (index == ROW_COUNT - 1u) {
    totalOut[totalOutOffset] = position + select(0u, 1u, live);
  }`,`const CAPACITY: u32 = ${o.ids.length}u;
const ROW_COUNT: u32 = ${l}u;`),g.push(i(t,{id:`${s}-${e}-publish`,operation:M,requiredCount:u,output:o}))};d.liveNodes&&C(`live-vertices`,d.nodeMask,d.liveNodes);let w=d.liveEdgeSlots||d.inducedTopology?n(t,`${s}-live-slots-offsets`,`uint32`,u):void 0;d.liveEdgeSlots?C(`live-slots`,d.edgeMask,d.liveEdgeSlots,w):w&&g.push(...new r({id:`${s}-live-slots-scan`,input:d.edgeMask,output:w}).getCommandNodes(t));let T=d.inducedTopology;if(T&&w){let e=`fn getPrefix(slot: u32) -> u32 {
  if (slot >= SLOT_COUNT) {
    return edgeMask[edgeMaskOffset + SLOT_COUNT - 1u] + slotPrefix[slotPrefixOffset + SLOT_COUNT - 1u];
  }
  return slotPrefix[slotPrefixOffset + slot];
}`;v(`induced-offsets`,`induced-offsets`,[{name:`offsets`,view:c.topology.offsets,type:`u32`,access:`read`},{name:`edgeMask`,view:d.edgeMask,type:`u32`,access:`read`},{name:`slotPrefix`,view:w,type:`u32`,access:`read`},{name:`inducedOffsets`,view:T.offsets,type:`u32`,access:`read_write`},{name:`overflowOut`,view:T.status.overflow,type:`u32`,access:`read_write`}],l+1,`let base = getPrefix(offsets[offsetsOffset]);
  let value = getPrefix(offsets[offsetsOffset + index]) - base;
  inducedOffsets[inducedOffsetsOffset + index] = value;
  if (index == NODE_COUNT) {
    overflowOut[overflowOutOffset] = select(0u, 1u, value > CAPACITY);
  }`,`const NODE_COUNT: u32 = ${l}u;
const SLOT_COUNT: u32 = ${u}u;
const CAPACITY: u32 = ${T.neighbors.length}u;
${e}`);let n=[{name:`offsets`,view:c.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:c.topology.neighbors,type:`u32`,access:`read`},{name:`edgeMask`,view:d.edgeMask,type:`u32`,access:`read`},{name:`slotPrefix`,view:w,type:`u32`,access:`read`},{name:`inducedNeighbors`,view:T.neighbors,type:`u32`,access:`read_write`}];for(let[e,n]of[[`clear-induced-neighbors`,T.neighbors],[`clear-induced-slots`,T.sourceSlots]])n&&g.push(o(t,{id:`${s}-${e}`,operation:M,view:n,type:`u32`,value:`0xffffffffu`}));T.sourceSlots&&n.push({name:`inducedSlots`,view:T.sourceSlots,type:`u32`,access:`read_write`}),v(`induced-scatter`,`induced-scatter`,n,u,`let firstSlot = min(offsets[offsetsOffset], SLOT_COUNT);
  let endSlot = min(offsets[offsetsOffset + NODE_COUNT], SLOT_COUNT);
  if (index < firstSlot || index >= endSlot || edgeMask[edgeMaskOffset + index] == 0u) {
    return;
  }
  let position = slotPrefix[slotPrefixOffset + index] - getPrefix(firstSlot);
  if (position < CAPACITY) {
    inducedNeighbors[inducedNeighborsOffset + position] = neighbors[neighborsOffset + index];
    ${T.sourceSlots?`inducedSlots[inducedSlotsOffset + position] = index;`:``}
  }`,`const NODE_COUNT: u32 = ${l}u;
const SLOT_COUNT: u32 = ${u}u;
const CAPACITY: u32 = ${T.neighbors.length}u;
${e}`)}return g}},P=_(E,`minimumSpeed`),F=_(E,`maximumSpeed`),I=(P.default+F.default)/2,L=F.default-P.default,R={nodeColumnCount:0,edgeColumnCount:1,hasEdgeTimes:!1},z=m({parameters:E,caption:`Roads kept by the speed filter, colored; dropped roads are muted.`,createInputs:({random:e,world:t})=>{let n=C(x(e,t),0),r=Float32Array.from({length:n.slotCount},(e,t)=>Math.min(n.lengths[t],n.lengths[n.reverseSlots[t]])),i=Float32Array.from(r).sort();return{network:n,speeds:Float32Array.from({length:n.slotCount},(e,t)=>{let a=Math.min(t,n.reverseSlots[t]),o=i.indexOf(r[t])/Math.max(i.length-1,1),s=(Math.imul(a+1,2654435761)>>>0)/4294967296;return g(o*.85+s*.15,0,1)})}},build(e,{network:t,speeds:n}){let{topology:r}=v(e,t),i=e.allocate(`parameters`,`float32`,4),a=e.allocate(`node-mask`,`uint32`,t.nodeCount),o=e.allocate(`edge-mask`,`uint32`,t.slotCount);return e.graph.add(new N({topology:r,edgeColumns:[e.upload(`speed`,`float32`,n).view],parameters:i.view,dropIsolated:!0,output:{nodeMask:a.view,edgeMask:o.view}})),{outputs:{...w(e,t),slots:T(e,t),edgeMask:o,nodeMask:a},update:e=>i.write(k(R,{edgeRanges:[[e.minimumSpeed,e.maximumSpeed]]}))}},interaction:{kind:`window`,hint:`slide the speed window`,rest:[.05+.9*I,1-(.05+.9*((L-.1)/.9))],resolve(e,{outputs:t,world:n,peek:r}){let i=g((e.unitX-.05)/.9,0,1),a=.1+.9*g((1-e.unitY-.05)/.9,0,1),o=h(P,i-a/2),s=h(F,i+a/2),c=r(t.edgeMask),l=0;if(c)for(let e of c)l+=+(e===1);let u=n.width*.06,d=n.width*.88;return{values:{minimumSpeed:o,maximumSpeed:s},glyphs:[{shape:`line`,from:[u,.04],to:[u+d,.04]},{shape:`box`,minimum:[u+d*o,.02],maximum:[u+d*s,.06]}],readout:`speed ${o.toFixed(2)}-${s.toFixed(2)} · ${c?Math.round(100*l/c.length):0}% of roads kept`}}},draw:({outputs:e,theme:t})=>[S(e),b(e.slots,e.edgeMask,{radius:2.2,...y(t)})]});export{z as default};
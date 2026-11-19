import{n as e,o as t}from"./network-view-utils-B1U8oNKx.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{n as i,o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as s}from"./construction-snapshot-DdkX9Pc0.js";import{A as c,D as l,E as u,f as d,k as f,m as p}from"./gpu-contributor-utils-BIwLQON5.js";import{t as m}from"./gpu-network-reachability-Coypydjh.js";import{n as h,t as g}from"./pick-Bo3t3hjv.js";import{t as _}from"./pointer-driver-Bk4iFPMM.js";import{C as v,x as y}from"./index-DIfTuwM1.js";import{t as b}from"./street-network-DNqX5OPC.js";import{C as x,E as S,S as ee,T as te,_ as C,h as w,k as T,m as E,p as ne,w as D,y as re}from"./pointer-overlay-Qh8qflcA.js";var O=`GPUNetworkPathExtraction`,k=`
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;`;function A(e,t){let n=[{name:`status`,view:t.status,type:`u32`,access:`read_write`}];return t.converged&&n.push({name:`converged`,view:t.converged,type:`u32`,access:`read`}),a(e,{id:t.id,operation:O,variant:`status`,bindings:n,invocationCount:2,body:`let isConverged = ${t.converged?`converged[convergedOffset] != 0u`:`true`};
  status[statusOffset + index] = select(0u, select(0u, 1u, isConverged), index == 1u);`})}function j(e,t){let n=[{name:`predecessors`,view:t.predecessors,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`targets`,view:t.targets,type:`u32`,access:`read`},{name:`nodeLengths`,view:t.nodeLengths,type:`u32`,access:`read_write`},{name:`status`,view:t.status,type:`atomic<u32>`,access:`read_write`}];return typeof t.targetCount==`object`&&n.push({name:`targetCount`,view:t.targetCount,type:`u32`,access:`read`}),t.pathFound&&n.push({name:`pathFound`,view:t.pathFound,type:`u32`,access:`read_write`}),t.pathCosts&&n.push({name:`pathCosts`,view:t.pathCosts,type:`u32`,access:`read_write`}),a(e,{id:t.id,operation:O,variant:`measure`,bindings:n,invocationCount:t.targets.length,declarations:`${k}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const MAXIMUM_PATH_LENGTH: u32 = ${t.maxPathLength}u;`,body:`var length = 0u;
  var found = false;
  var costBits = INFINITY_BITS;
  let isActive = ${typeof t.targetCount==`number`?`index < ${t.targetCount}u`:t.targetCount?`index < targetCount[targetCountOffset]`:`true`};
  let targetNode = targets[targetsOffset + index];
  if (isActive && targetNode < NODE_COUNT) {
    let targetCost = costs[costsOffset + targetNode];
    if ((bitcast<u32>(targetCost) & 0x7fffffffu) < INFINITY_BITS) {
      var node = targetNode;
      var walked = 1u;
      // 0 walking, 1 root reached, 2 broken chain, 3 too long.
      var state = 0u;
      for (var step = 0u; step <= MAXIMUM_PATH_LENGTH; step++) {
        let predecessor = predecessors[predecessorsOffset + node];
        if (predecessor == NONE) {
          state = 1u;
          break;
        }
        if (predecessor >= NODE_COUNT) {
          state = 2u;
          break;
        }
        if (walked >= MAXIMUM_PATH_LENGTH) {
          state = 3u;
          break;
        }
        node = predecessor;
        walked++;
      }
      // A NONE predecessor is a source only when the search converged (status word 1).
      if (state == 1u && atomicLoad(&status[statusOffset + 1u]) != 0u) {
        found = true;
        length = walked;
        costBits = bitcast<u32>(targetCost);
      }
      if (state == 3u) {
        atomicOr(&status[statusOffset], 1u);
      }
    }
  }
  nodeLengths[nodeLengthsOffset + index] = length;
  ${t.pathFound?`pathFound[pathFoundOffset + index] = select(0u, 1u, found);`:``}
  ${t.pathCosts?`pathCosts[pathCostsOffset + index] = costBits;`:``}`})}function M(e,t){return a(e,{id:t.id,operation:O,variant:`edge-lengths`,bindings:[{name:`nodeLengths`,view:t.nodeLengths,type:`u32`,access:`read`},{name:`edgeLengths`,view:t.edgeLengths,type:`u32`,access:`read_write`}],invocationCount:t.nodeLengths.length,body:`let length = nodeLengths[nodeLengthsOffset + index];
  edgeLengths[edgeLengthsOffset + index] = select(0u, length - 1u, length > 0u);`})}function N(e,t){let n=[],r=[],i=t.targetCapacity;for(let[e,a]of[[`node`,t.nodes],[`edge`,t.edges]]){if(!a)continue;a.lengths&&a.starts&&n.push({name:`${e}Lengths`,view:a.lengths,type:`u32`,access:`read`},{name:`${e}Starts`,view:a.starts,type:`u32`,access:`read`}),n.push({name:`${e}Total`,view:a.total,type:`u32`,access:`read_write`}),a.pathOffsets&&n.push({name:`${e}PathOffsets`,view:a.pathOffsets,type:`u32`,access:`read_write`});let t=i>0?`${e}Starts[${e}StartsOffset + ${i-1}u] + ${e}Lengths[${e}LengthsOffset + ${i-1}u]`:`0u`;r.push(`if (index == ${i}u) {
    let total = ${t};
    ${e}Total[${e}TotalOffset] = total;
    ${a.pathOffsets?`${e}PathOffsets[${e}PathOffsetsOffset + ${i}u] = total;`:``}
  }
  ${a.pathOffsets&&i>0?`if (index < ${i}u) { ${e}PathOffsets[${e}PathOffsetsOffset + index] = ${e}Starts[${e}StartsOffset + index]; }`:``}`)}return a(e,{id:t.id,operation:O,variant:`totals`,bindings:n,invocationCount:i+1,body:r.join(`
  `)})}function P(e,t){let n=[{name:`predecessors`,view:t.predecessors,type:`u32`,access:`read`},{name:`targets`,view:t.targets,type:`u32`,access:`read`},{name:`nodeLengths`,view:t.nodeLengths,type:`u32`,access:`read`},{name:`nodeStarts`,view:t.nodeStarts,type:`u32`,access:`read`},{name:`ids`,view:t.ids,type:`u32`,access:`read_write`}];return t.nodeIds&&n.push({name:`nodeIds`,view:t.nodeIds,type:`u32`,access:`read`}),t.nodeRows&&n.push({name:`nodeRows`,view:t.nodeRows,type:`u32`,access:`read_write`}),a(e,{id:t.id,operation:O,variant:`write-nodes`,bindings:n,invocationCount:t.targets.length,declarations:`const CAPACITY: u32 = ${t.capacity}u;`,body:`let length = nodeLengths[nodeLengthsOffset + index];
  if (length == 0u) { return; }
  let start = nodeStarts[nodeStartsOffset + index];
  var node = targets[targetsOffset + index];
  for (var step = 0u; step < length; step++) {
    let position = start + length - 1u - step;
    if (position < CAPACITY) {
      ids[idsOffset + position] = ${t.nodeIds?`nodeIds[nodeIdsOffset + node]`:`node`};
      ${t.nodeRows?`nodeRows[nodeRowsOffset + position] = node;`:``}
    }
    node = predecessors[predecessorsOffset + node];
  }`})}function F(e,t){return a(e,{id:t.id,operation:O,variant:`map-edge-ids`,invocationCount:t.ids.length,bindings:[{name:`rows`,view:t.rows,type:`u32`,access:`read`},{name:`ids`,view:t.ids,type:`u32`,access:`read_write`},...t.edgeIds?[{name:`edgeIds`,view:t.edgeIds,type:`u32`,access:`read`}]:[]],body:`let row = rows[rowsOffset + index];
      var identity = row;
      ${t.edgeIds?`if (row != 0xffffffffu) { identity = edgeIds[edgeIdsOffset + row]; }`:``}
      ids[idsOffset + index] = identity;`})}function I(e,t){return a(e,{id:t.id,operation:O,variant:`write-links`,bindings:[{name:`predecessors`,view:t.predecessors,type:`u32`,access:`read`},{name:`targets`,view:t.targets,type:`u32`,access:`read`},{name:`nodeLengths`,view:t.nodeLengths,type:`u32`,access:`read`},{name:`edgeStarts`,view:t.edgeStarts,type:`u32`,access:`read`},{name:`linkTails`,view:t.linkTails,type:`u32`,access:`read_write`},{name:`linkHeads`,view:t.linkHeads,type:`u32`,access:`read_write`}],invocationCount:t.targets.length,declarations:`const CAPACITY: u32 = ${t.capacity}u;`,body:`let length = nodeLengths[nodeLengthsOffset + index];
  if (length < 2u) { return; }
  let start = edgeStarts[edgeStartsOffset + index];
  var head = targets[targetsOffset + index];
  for (var step = 0u; step + 1u < length; step++) {
    let tail = predecessors[predecessorsOffset + head];
    let position = start + length - 2u - step;
    if (position < CAPACITY) {
      linkTails[linkTailsOffset + position] = tail;
      linkHeads[linkHeadsOffset + position] = head;
    }
    head = tail;
  }`})}function L(e,t){let n=[{name:`linkTails`,view:t.linkTails,type:`u32`,access:`read`},{name:`linkHeads`,view:t.linkHeads,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t.weights,type:`f32`,access:`read`},{name:`ids`,view:t.ids,type:`u32`,access:`read_write`}];return t.edgeIds&&n.push({name:`edgeIds`,view:t.edgeIds,type:`u32`,access:`read`}),a(e,{id:t.id,operation:O,variant:`resolve-edges`,bindings:n,invocationCount:t.ids.length,declarations:`${k}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${t.edgeCount}u;`,body:`let tail = linkTails[linkTailsOffset + index];
  var result = NONE;
  if (tail < NODE_COUNT) {
    let head = linkHeads[linkHeadsOffset + index];
    let tailCost = costs[costsOffset + tail];
    let headCost = costs[costsOffset + head];
    let rangeStart = min(offsets[offsetsOffset + tail], EDGE_COUNT);
    let rangeEnd = min(offsets[offsetsOffset + tail + 1u], EDGE_COUNT);
    var exact = NONE;
    var fallback = NONE;
    for (var edge = rangeStart; edge < rangeEnd; edge++) {
      if (neighbors[neighborsOffset + edge] != head) { continue; }
      if (fallback == NONE) { fallback = edge; }
      let weight = weights[weightsOffset + edge];
      if (weight >= 0.0 && tailCost + weight == headCost) {
        exact = edge;
        break;
      }
    }
    result = select(exact, fallback, exact == NONE);
    ${t.edgeIds?`if (result != NONE) { result = edgeIds[edgeIdsOffset + result]; }`:``}
  }
  ids[idsOffset + index] = result;`})}var R=65536,z=1024,B=`GPUNetworkPathExtraction`;function V(e){let t=s(e,[`id`,`predecessors`,`costs`,`targets`,`targetCount`,`predecessorsComplete`,`maximumPathLength`,`nodeIds`,`output`,`edges`]),n=s(t.output,[`ids`,`nodeRows`,`status`,`lineOffsets`,`pathCosts`,`pathFound`]),r=t.edges&&s(t.edges,[`topology`,`edgeIds`,`output`]),i=r&&s(r.output,[`ids`,`edgeRows`,`status`,`lineOffsets`]),a=[`count`,`overflow`,`requiredCount`,`candidateOverflow`];return Object.freeze({...t,output:Object.freeze({...n,status:s(n.status,a)}),edges:r&&i?Object.freeze({...r,topology:s(r.topology,[`offsets`,`neighbors`,`weights`,`nodePositions`]),output:Object.freeze({...i,status:s(i.status,a)})}):void 0})}var ie=class{id;props;maximumPathLength;constructor(t){t=V(t),this.id=t.id??`network-path-extraction`,this.props=t,this.maximumPathLength=t.maximumPathLength??z;let{id:n}=this,{edges:r,output:i}=t;for(let[e,a]of[[`predecessors`,t.predecessors],[`targets`,t.targets],[`targetCount`,typeof t.targetCount==`object`?t.targetCount:void 0],[`predecessorsComplete`,t.predecessorsComplete],[`nodeIds`,t.nodeIds],[`output.lineOffsets`,i.lineOffsets],[`output.nodeRows`,i.nodeRows],[`output.pathFound`,i.pathFound],[`edges.edgeIds`,r?.edgeIds],[`edges.output.lineOffsets`,r?.output.lineOffsets],[`edges.output.edgeRows`,r?.output.edgeRows]])a&&u(a,[`uint32`],`${n} ${e}`);for(let[e,r]of[[`costs`,t.costs],[`output.pathCosts`,i.pathCosts]])r&&u(r,[`float32`],`${n} ${e}`);let a=t.predecessors.length,o=t.targets.length;if(a<1)throw Error(`${n} predecessors must contain at least one node`);if(t.costs.length!==a)throw Error(`${n} costs length must equal predecessors length`);if(t.nodeIds&&t.nodeIds.length!==a)throw Error(`${n} nodeIds length must equal the node count`);if(typeof t.targetCount==`number`)d(n,`targetCount`,t.targetCount,0,o);else if(t.targetCount&&t.targetCount.length!==1)throw Error(`${n} targetCount must contain exactly one row`);if(t.predecessorsComplete&&t.predecessorsComplete.length!==1)throw Error(`${n} predecessorsComplete must contain exactly one row`);for(let[e,t]of[[`output.pathCosts`,i.pathCosts],[`output.pathFound`,i.pathFound]])if(t&&t.length!==o)throw Error(`${n} ${e} length must equal targets length`);for(let[e,t]of[[`output.lineOffsets`,i.lineOffsets],[`edges.output.lineOffsets`,r?.output.lineOffsets]])if(t&&t.length!==o+1)throw Error(`${n} ${e} must contain one more row than targets`);if(r){let e=c(`${n} edges`,r.topology,{weights:!0});if(e.nodeCount!==a)throw Error(`${n} edges.topology must contain one node per predecessors row`);if(r.edgeIds&&r.edgeIds.length!==e.edgeCount)throw Error(`${n} edges.edgeIds length must equal topology.neighbors length`);p(`${n} edges`,r.output)}d(n,`maximumPathLength`,this.maximumPathLength,1,R),p(n,i),i.nodeRows&&l(n,`output.nodeRows`,i.nodeRows,i.ids.length),r?.output.edgeRows&&l(n,`edges.output.edgeRows`,r.output.edgeRows,r.output.ids.length);let s=[i.ids,i.nodeRows,i.status.count,i.status.overflow,i.status.candidateOverflow,i.status.requiredCount,i.lineOffsets,i.pathCosts,i.pathFound,r?.output.ids,r?.output.edgeRows,r?.output.status.count,r?.output.status.overflow,r?.output.status.candidateOverflow,r?.output.status.requiredCount,r?.output.lineOffsets].filter(e=>e!==void 0).map(e),m=[t.predecessors,t.costs,t.targets,typeof t.targetCount==`object`?t.targetCount:void 0,t.predecessorsComplete,t.nodeIds,...r?f(r.topology):[],r?.edgeIds].filter(e=>e!==void 0).map(e);if(new Set(s).size!==s.length||s.some(e=>m.includes(e)))throw Error(`${n} outputs must use separate buffers from each other and from inputs`)}getCommandNodes(e){let{id:a,props:s,maximumPathLength:c}=this,{edges:l,output:u}=s;t(a,e,H(s),{totals:4,status:s.targets.length>0?8:0,targetColumns:s.targets.length*4,linkColumns:l&&s.targets.length>0?l.output.ids.length*4:0});let d=s.predecessors.length,f=s.targets.length,p=n(e,`${a}-node-total`,`uint32`,1),m=l?n(e,`${a}-edge-total`,`uint32`,1):void 0,h=[];if(f===0)return h.push(N(e,{id:`${a}-totals`,targetCapacity:f,nodes:{total:p,pathOffsets:u.lineOffsets},edges:l&&m?{total:m,pathOffsets:l.output.lineOffsets}:void 0}),i(e,{id:`${a}-publish`,operation:B,requiredCount:p,output:u})),l&&m&&h.push(i(e,{id:`${a}-edge-publish`,operation:B,requiredCount:m,output:l.output})),h;let g=n(e,`${a}-status`,`uint32`,2),_=n(e,`${a}-node-lengths`,`uint32`,f),v=n(e,`${a}-node-starts`,`uint32`,f);h.push(A(e,{id:`${a}-clear`,status:g,converged:s.predecessorsComplete}),j(e,{id:`${a}-measure`,nodeCount:d,maxPathLength:c,predecessors:s.predecessors,costs:s.costs,targets:s.targets,targetCount:s.targetCount,nodeLengths:_,status:g,pathFound:u.pathFound,pathCosts:u.pathCosts}),...new r({id:`${a}-node-scan`,input:_,output:v}).getCommandNodes(e));let y,b;if(l&&(y=n(e,`${a}-edge-lengths`,`uint32`,f),b=n(e,`${a}-edge-starts`,`uint32`,f),h.push(M(e,{id:`${a}-edge-lengths`,nodeLengths:_,edgeLengths:y}),...new r({id:`${a}-edge-scan`,input:y,output:b}).getCommandNodes(e))),h.push(N(e,{id:`${a}-totals`,targetCapacity:f,nodes:{lengths:_,starts:v,total:p,pathOffsets:u.lineOffsets},edges:l&&m?{lengths:y,starts:b,total:m,pathOffsets:l.output.lineOffsets}:void 0})),u.ids.length>0&&h.push(P(e,{id:`${a}-write-nodes`,capacity:u.ids.length,predecessors:s.predecessors,targets:s.targets,nodeLengths:_,nodeStarts:v,nodeIds:s.nodeIds,ids:u.ids,nodeRows:u.nodeRows})),h.push(i(e,{id:`${a}-publish`,operation:B,requiredCount:p,output:u,candidateOverflowSources:u.status.candidateOverflow?[g]:[]})),l&&m&&b){let t=l.output.ids.length;if(t>0){let r=n(e,`${a}-link-tails`,`uint32`,t),i=n(e,`${a}-link-heads`,`uint32`,t);h.push(o(e,{id:`${a}-link-clear`,operation:B,view:r,type:`u32`,value:`0xffffffffu`}),I(e,{id:`${a}-write-links`,capacity:t,predecessors:s.predecessors,targets:s.targets,nodeLengths:_,edgeStarts:b,linkTails:r,linkHeads:i}),L(e,{id:`${a}-write-edges`,nodeCount:d,edgeCount:l.topology.neighbors.length,linkTails:r,linkHeads:i,costs:s.costs,offsets:l.topology.offsets,neighbors:l.topology.neighbors,weights:l.topology.weights,edgeIds:l.output.edgeRows?void 0:l.edgeIds,ids:l.output.edgeRows??l.output.ids})),l.output.edgeRows&&h.push(F(e,{id:`${a}-map-edge-ids`,rows:l.output.edgeRows,edgeIds:l.edgeIds,ids:l.output.ids}))}h.push(i(e,{id:`${a}-edge-publish`,operation:B,requiredCount:m,output:l.output,candidateOverflowSources:l.output.status.candidateOverflow?[g]:[]}))}return h}};function H(e){let{edges:t,output:n}=e;return[e.predecessors,e.costs,e.targets,typeof e.targetCount==`object`?e.targetCount:void 0,e.predecessorsComplete,e.nodeIds,n.ids,n.nodeRows,n.status.count,n.status.overflow,n.status.candidateOverflow,n.status.requiredCount,n.lineOffsets,n.pathCosts,n.pathFound,...t?f(t.topology):[],t?.edgeIds,t?.output.ids,t?.output.edgeRows,t?.output.status.count,t?.output.status.overflow,t?.output.status.candidateOverflow,t?.output.status.requiredCount,t?.output.lineOffsets]}var U=y,W=640,G=256,K=async e=>{let{device:t}=e;ee(t);let r=S(e.theme),i=v(U,e.parameters),a=e.width/e.height,o={worldWidth:a,worldHeight:1,width:e.width,height:e.height},s=b(a,W),{nodes:c,nodeCount:l,edgeCount:u,slotCount:d}=s,f=new Float32Array(d);s.fillWeights(f,i.congestion);let p=D(t,c),y=D(t,s.edgePairs),O=D(t,s.offsets),k=D(t,s.neighbors),A=D(t,f),j=D(t,Uint32Array.of(0)),M=D(t,Uint32Array.of(0)),N=x(t,l),P=x(t,l),F=x(t,1),I=x(t,G),L=x(t,1),R=x(t,1),z=D(t,new Float32Array(4)),B=x(t,2),V=await w.create(t,`demo-shortest-path`,e=>{let t=T(e,`costs`,N,`float32`,l),r=T(e,`predecessors`,P,`uint32`,l),i=T(e,`converged`,F,`uint32`,1);e.add(new m({topology:{offsets:T(e,`offsets`,O,`uint32`,l+1),neighbors:T(e,`neighbors`,k,`uint32`,d),weights:T(e,`weights`,A,`float32`,d)},startNodes:T(e,`start-nodes`,j,`uint32`,1),maximumIterations:24,localIterations:16,output:{costs:t,predecessors:r,status:{converged:i}}})),e.add(new ie({predecessors:r,costs:t,targets:T(e,`targets`,M,`uint32`,1),predecessorsComplete:i,maximumPathLength:G,output:{ids:n(e,`path-identities`,`uint32`,G),nodeRows:T(e,`path-rows`,I,`uint32`,G),status:{count:T(e,`path-count`,L,`uint32`,1),overflow:T(e,`path-overflow`,R,`uint32`,1)}}}))}),H=new re(t,p,y,N,u),K=new C(t,p,I,L,G),q=new ne(t,z,B,2),J=0,Y=s.findNearestNode(a*.85,.7),X=!0,Z=r.categories[1],Q=new E(()=>{X&&(X=!1,J=s.findNearestNode(a*(i.originPosition/100),.5),s.fillWeights(f,i.congestion),A.write(f),j.write(Uint32Array.of(J)),M.write(Uint32Array.of(Y)),z.write(Float32Array.of(c[J*2],c[J*2+1],c[Y*2],c[Y*2+1])),V.run());let e=r.muted;H.update(o,r,{range:[0,3],radius:1.2,alpha:.5,solid:[e[0],e[1],e[2],.3]}),K.update(o,r,{radius:3.2,solid:[Z[0],Z[1],Z[2],1]}),q.update(o,r,{mode:-1,radius:7,solid:[...r.ink,1],hillshade:1}),te(t,r,[H,K,q,...$.getLayers()])}),ae=h(c,l),$=_({device:t,view:o,theme:r,reducedMotion:e.reducedMotion,kind:`place`,hint:`move the destination`,rest:e=>e.toUnit([c[Y*2],c[Y*2+1]]),reportReadout:e.reportReadout,requestDraw:()=>Q.request(),resolve(e){let t=g(ae,e.x,e.y,Y);t!==Y&&(Y=t,X=!0,Q.request());let n=[c[Y*2],c[Y*2+1]];return{glyphs:[{shape:`line`,from:[e.x,e.y],to:n,dashed:!0},{shape:`crosshair`,at:n,size:12},{shape:`dot`,at:n}],readout:`destination node ${Y}`}}});return{gesture:$.gesture,render(e){$.advance(e),Q.flush()},resize(e,t){o.width=e,o.height=t,$.refresh(),Q.request()},setPointer:$.setPointer,resetPointer:$.resetPointer,setParameters(e){Object.assign(i,e),X=!0,Q.request()},destroy(){Q.cancel(),$.destroy(),H.destroy(),K.destroy(),q.destroy(),V.destroy();for(let e of[p,y,O,k,A,j,M,N,P,F,I,L,R,z,B])e.destroy()}}};export{K as default,U as parameters};
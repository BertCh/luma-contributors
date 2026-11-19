import{t as e}from"./kernel-BXyKsp4B.js";function t(t,n,r,i,a,o=i.length){let s=t.allocate(`${n}-segments`,`float32x4`,i.length),c=t.allocate(`${n}-nodes`,`float32x2`,i.length*2),l=t.allocate(`${n}-groups`,`uint32`,i.length*2),u=t.upload(`${n}-node-edges`,`uint32x2`,Uint32Array.from({length:i.length*2},(e,t)=>Math.floor(t/2)*2+t%2));return e(t,{id:`${n}-edge-pass`,invocationCount:i.length,bindings:[{name:`positions`,buffer:r,access:`read`},{name:`edges`,buffer:i,access:`read`},{name:`validity`,buffer:a,access:`read`},{name:`segments`,buffer:s,access:`read_write`},{name:`nodes`,buffer:c,access:`read_write`},{name:`groups`,buffer:l,access:`read_write`}],body:`
  var start = vec2f(-9.0);
  var end = vec2f(-9.0);
  if (validity[validityOffset + index] != 0u) {
    let edge = edges[edgesOffset + index];
    start = positions[positionsOffset + edge.x];
    end = positions[positionsOffset + edge.y];
  }
  segments[segmentsOffset + index] = vec4f(start, end);
  nodes[nodesOffset + index * 2u] = start;
  nodes[nodesOffset + index * 2u + 1u] = end;
  let group = index / ${Math.max(o,1)}u;
  groups[groupsOffset + index * 2u] = group;
  groups[groupsOffset + index * 2u + 1u] = group;`}),{segments:s,nodes:c,nodeEdges:u,groups:l}}export{t};
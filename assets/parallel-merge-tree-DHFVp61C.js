import{r as e}from"./kernel-C3tfQOn4.js";import{a as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";var r=2048;function i(e){return Math.ceil(e/r)}function a(e,t){return{name:e,view:t,type:`u32`,access:`read`}}function o(e,t,n=!1){return{name:e,view:t,type:n?`atomic<u32>`:`u32`,access:`read_write`}}function s(r,i,s){let c=[],l=(t,n)=>e(r,`${i}-${t}`,`uint32`,Math.max(n,1)),u=(e,t,a,o,l=``,u)=>{c.push(n(r,{id:`${i}-${e}`,operation:s,variant:e.replace(/-\d+/g,``),invocationCount:t,bindings:a,declarations:`const NONE: u32 = 0xffffffffu;\n${l}`,body:o,...u}))};return{nodes:c,createWordView:l,appendKernel:u,appendJumpNodes:(e,n,c,d,f,p,m=``,h=[])=>{let g=l(`${e}-scratch`,n*c),_=l(`${e}-pending`,1),v=t(r,{id:`${i}-${e}-dispatch`,operation:s,shapes:[[n]]});u(`${e}-seed`,1,[o(`pending`,_)],`pending[pendingOffset] = 1u;`);for(let t=0;t<d;t++)u(`${e}-gate-${t}`,1,[o(`pending`,_),v.binding],`
  if (pending[pendingOffset] != 0u) { ${v.armWGSL} }
  else { ${v.disarmWGSL} }
  pending[pendingOffset] = 0u;`),u(`${e}-jump-${t}`,n,[a(`table`,f),o(`scratch`,g),o(`pending`,_,!0),...h],p,m,v.getGate(0)),u(`${e}-copy-${t}`,n,[a(`scratch`,g),o(`table`,f)],`
  for (var word = 0u; word < ${c}u; word++) {
    table[tableOffset + index * ${c}u + word] = scratch[scratchOffset + index * ${c}u + word];
  }`,``,v.getGate(0))}}}function c(e,n){let c=s(e,n.id,n.operation),{createWordView:l,appendKernel:u}=c,d=l(`parents`,n.nodeCount),f=l(`children`,n.nodeCount*2);if(u(`init`,n.nodeCount,[o(`parents`,d),o(`children`,f),...n.initializeBindings??[]],`
  parents[parentsOffset + index] = NONE;
  children[childrenOffset + index * 2u] = NONE;
  children[childrenOffset + index * 2u + 1u] = 0u;
  ${n.initializeBody??``}`),n.boundedSplicing){let a=l(`splice-first-tasks`,n.edgeCount*2),s=l(`splice-second-tasks`,n.edgeCount*2),c=l(`splice-pending`,1),f=t(e,{id:`${n.id}-splice-dispatch`,operation:n.operation,shapes:[[n.edgeCount]]}),p=n.spliceMaximumWork;u(`splice-seed`,Math.max(n.edgeCount,1),[...n.edgeBindings,o(`firstTasks`,a),o(`secondTasks`,s),o(`pending`,c),...p?[o(`maximumWork`,p)]:[]],`if (index == 0u) {
 pending[pendingOffset] = ${n.edgeCount>0?`1u`:`0u`};
 ${p?`maximumWork[maximumWorkOffset] = 0u;`:``}
}
if (index >= ${n.edgeCount}u) { return; }
firstTasks[firstTasksOffset + index * 2u] = NONE;
firstTasks[firstTasksOffset + index * 2u + 1u] = NONE;
secondTasks[secondTasksOffset + index * 2u] = NONE;
secondTasks[secondTasksOffset + index * 2u + 1u] = NONE;
${n.edgeBody}
firstTasks[firstTasksOffset + index * 2u] = firstEndpoint;
firstTasks[firstTasksOffset + index * 2u + 1u] = node;
secondTasks[secondTasksOffset + index * 2u] = secondEndpoint;
secondTasks[secondTasksOffset + index * 2u + 1u] = node;`,n.declarations);for(let e=0;e<i(n.nodeCount);e++)u(`splice-gate-${e}`,1,[o(`pending`,c),f.binding],`if (pending[pendingOffset] != 0u) { ${f.armWGSL} }
else { ${f.disarmWGSL} }
pending[pendingOffset] = 0u;`),u(`splice-round-${e}`,n.edgeCount,[o(`parents`,d,!0),o(`firstTasks`,a),o(`secondTasks`,s),o(`pending`,c,!0),...p?[o(`maximumWork`,p,!0)]:[]],`var work = 0u;
for (var endpoint = 0u; endpoint < 2u; endpoint++) {
 var first = firstTasks[firstTasksOffset + index * 2u];
 var second = firstTasks[firstTasksOffset + index * 2u + 1u];
 if (endpoint == 1u) {
  first = secondTasks[secondTasksOffset + index * 2u];
  second = secondTasks[secondTasksOffset + index * 2u + 1u];
 }
 for (var attempt = 0u; attempt < ${r}u; attempt++) {
  if (first == NONE || first == second) { first = NONE; break; }
  let lower = min(first, second); let higher = max(first, second);
  let previous = atomicMin(&parents[parentsOffset + lower], higher);
  work++;
  if (previous == NONE || previous == higher) { first = NONE; break; }
  // Every existing parent is strictly larger than its child. Thus min(previous, higher)
  // is strictly larger than lower, regardless of the order in which lanes execute.
  first = previous; second = higher;
 }
 if (first == second) { first = NONE; }
 if (first != NONE) { atomicOr(&pending[pendingOffset], 1u); }
 if (endpoint == 0u) {
  firstTasks[firstTasksOffset + index * 2u] = first;
  firstTasks[firstTasksOffset + index * 2u + 1u] = second;
 } else {
  secondTasks[secondTasksOffset + index * 2u] = first;
  secondTasks[secondTasksOffset + index * 2u + 1u] = second;
 }
}
${p?`atomicMax(&maximumWork[maximumWorkOffset], work);`:``}`,``,f.getGate(0))}else u(`splice`,n.edgeCount,[...n.edgeBindings,o(`parents`,d,!0)],`
  ${n.edgeBody}
  for (var endpoint = 0u; endpoint < 2u; endpoint++) {
    var first = select(firstEndpoint, secondEndpoint, endpoint == 1u);
    var second = node;
    loop {
      if (first == second) { break; }
      let lower = min(first, second);
      let higher = max(first, second);
      let previous = atomicMin(&parents[parentsOffset + lower], higher);
      if (previous == NONE || previous == higher) { break; }
      first = previous;
      second = higher;
    }
  }`,n.declarations);return u(`children`,n.nodeCount,[a(`parents`,d),o(`children`,f,!0)],`
  let parent = parents[parentsOffset + index];
  if (parent != NONE) {
    atomicMin(&children[childrenOffset + parent * 2u], index);
    atomicMax(&children[childrenOffset + parent * 2u + 1u], index);
  }`),{...c,parents:d,children:f}}export{i as n,c as r,s as t};
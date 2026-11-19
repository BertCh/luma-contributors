import{r as e}from"./kernel-C3tfQOn4.js";import{o as t,t as n}from"./wgsl-kernel-nodes-DagJklH2.js";function r(r,i){let{id:a,operation:o,nodeCount:s,offsets:c,neighbors:l,labels:u,converged:d,iterations:f}=i,p=e(r,`${a}-changed`,`uint32`,1),m=[t(r,{id:`${a}-init`,operation:o,variant:`component-init`,bindings:[{name:`labels`,view:u,type:`u32`,access:`read_write`}],invocationCount:s,body:`labels[labelsOffset + index] = index;`})];for(let e=0;e<f;e++)m.push(n(r,{id:`${a}-clear-${e}`,operation:o,view:p,type:`u32`,value:`0u`}),t(r,{id:`${a}-step-${e}`,operation:o,variant:`component-step`,bindings:[{name:`offsets`,view:c,type:`u32`,access:`read`},{name:`neighbors`,view:l,type:`u32`,access:`read`},{name:`labels`,view:u,type:`atomic<u32>`,access:`read_write`},{name:`changed`,view:p,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,declarations:`const NODE_COUNT: u32 = ${s}u; const SLOT_CAPACITY: u32 = ${l.length}u;`,body:`let first = min(offsets[offsetsOffset + index], SLOT_CAPACITY);
  let last = min(offsets[offsetsOffset + index + 1u], SLOT_CAPACITY);
  let current = atomicLoad(&labels[labelsOffset + index]);
  var best = min(current, atomicLoad(&labels[labelsOffset + current]));
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < NODE_COUNT) {
      best = min(best, atomicLoad(&labels[labelsOffset + neighbor]));
    }
  }
  var didChange = atomicMin(&labels[labelsOffset + index], best) > best;
  didChange = (atomicMin(&labels[labelsOffset + current], best) > best) || didChange;
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < NODE_COUNT) {
      didChange = (atomicMin(&labels[labelsOffset + neighbor], best) > best) || didChange;
    }
  }
  if (didChange) {
    atomicStore(&changed[changedOffset], 1u);
  }`}));return d&&m.push(t(r,{id:`${a}-converged`,operation:o,variant:`component-converged`,bindings:[{name:`changed`,view:p,type:`u32`,access:`read`},{name:`converged`,view:d,type:`u32`,access:`read_write`}],invocationCount:1,body:`converged[convergedOffset] = select(1u, 0u, changed[changedOffset] != 0u);`})),m}export{r as t};
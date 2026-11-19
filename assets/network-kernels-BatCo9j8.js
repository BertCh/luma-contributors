import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";function t(t,n){let r=t.allocate(`budget`,`float32`,1);return t.graph.add(e(t.graph,{id:`prism-budget`,operation:`display-prism-budget`,invocationCount:1,bindings:[{name:`forward`,view:n.forward.view,type:`f32`,access:`read`},{name:`destination`,view:n.destination.view,type:`u32`,access:`read`},{name:`spare`,view:n.spare.view,type:`f32`,access:`read`},{name:`budget`,view:r.view,type:`f32`,access:`read_write`}],body:`
  budget[budgetOffset] = forward[forwardOffset + destination[destinationOffset]] * (1.0 + spare[spareOffset]);`})),r}function n(t,n){let r=t.allocate(`prism-tightness`,`float32`,n.slotCount*2);return t.graph.add(e(t.graph,{id:`prism-tightness`,operation:`display-prism-tightness`,invocationCount:n.slotCount,bindings:[{name:`inPrism`,view:n.inPrism.view,type:`u32`,access:`read`},{name:`slack`,view:n.slack.view,type:`f32`,access:`read`},{name:`budget`,view:n.budget.view,type:`f32`,access:`read`},{name:`spare`,view:n.spare.view,type:`f32`,access:`read`},{name:`doubled`,view:r.view,type:`f32`,access:`read_write`}],body:`
  let isInside = inPrism[inPrismOffset + index] == 1u;
  let room = max(budget[budgetOffset] * spare[spareOffset] / (1.0 + spare[spareOffset]), 1e-6);
  let tightness = 1.0 - slack[slackOffset + index] / room;
  let value = select(0.0, 0.15 + 0.85 * clamp(tightness, 0.0, 1.0), isInside);
  doubled[doubledOffset + index] = value;
  doubled[doubledOffset + ${n.slotCount}u + index] = value;`})),r}function r(t,n){let r=t.allocate(`${n.id}-column`,`float32`,n.vertexCount);return t.graph.add(e(t.graph,{id:`${n.id}-column`,operation:`display-table-column`,invocationCount:n.vertexCount,bindings:[{name:`table`,view:n.table.view,type:`f32`,access:`read`},{name:`selection`,view:n.selection.view,type:`u32`,access:`read`},{name:`column`,view:r.view,type:`f32`,access:`read_write`}],body:`
  let row = tableOffset + index * ${n.stride}u;
  let picked = selection[selectionOffset];
  column[columnOffset + index] = select(
    table[row + picked],
    table[row + ${n.stride-1}u] - table[row + 1u],
    picked == ${n.stride}u
  );`})),r}export{n,r,t};
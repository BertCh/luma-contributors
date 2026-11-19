import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as n}from"./gpu-contributor-utils-BIwLQON5.js";import{u as r}from"./raster-band-utils-BcTxc8_Y.js";import{c as i,f as a}from"./raster-relaxation-Di9wjtxp.js";var o=4294967295;function s(e,t){let i=r(e,t.grid);if(n(t.labels,[`uint32`],`${e} labels`),t.labels.length!==i)throw Error(`${e} labels length must equal grid width * height`);if(t.mask&&(n(t.mask,[`uint32`],`${e} mask`),t.mask.length!==i))throw Error(`${e} mask length must equal grid width * height`);for(let[r,i]of[[`status.converged`,t.status?.converged],[`status.componentCount`,t.status?.componentCount],[`status.overflow`,t.status?.overflow]])if(i&&(n(i,[`uint32`],`${e} ${r}`),i.length<1))throw Error(`${e} ${r} must hold one uint32`);return i}function c(e){return[e.labels,e.mask,e.status?.converged,e.status?.componentCount,e.status?.overflow].filter(e=>e!==void 0)}function l(n,r,i,a,o){let s=e(n,`${r}-keys`,`uint32`,a.labels.length),c=[{name:`labels`,view:a.labels,type:`u32`,access:`read`}],l=[];return a.mask&&(c.push({name:`labelValidity`,view:a.mask,type:`u32`,access:`read`}),l.push(`labelValidity[labelValidityOffset + index] != 0u`)),a.status?.converged&&(c.push({name:`converged`,view:a.status?.converged,type:`u32`,access:`read`}),l.push(`converged[convergedOffset] != 0u`)),a.status?.overflow&&(c.push({name:`overflow`,view:a.status?.overflow,type:`u32`,access:`read`}),l.push(`overflow[overflowOffset] == 0u`)),a.status?.componentCount&&(c.push({name:`componentCount`,view:a.status?.componentCount,type:`u32`,access:`read`}),l.push(`label <= componentCount[componentCountOffset]`)),c.push({name:`keys`,view:s,type:`u32`,access:`read_write`}),{node:t(n,{id:`${r}-keys`,operation:i,variant:`keys`,bindings:c,invocationCount:a.labels.length,body:`let label = labels[labelsOffset + index];
  let accepted = label != 0u && label <= ${o}u${l.map(e=>` && ${e}`).join(``)};
  keys[keysOffset + index] = select(0u, label, accepted);`}),keys:s}}function u(e,n,r,s,c){if(!s.status?.converged&&!s.status?.overflow||c.length===0)return[];let l=Math.max(...c.map(e=>e.view.length)),u=i(e,`${n}-unresolved`,r,l,n),{x:d,y:f,z:p}=u.layout,m=[],h=[];s.status?.converged&&(m.push({name:`converged`,view:s.status?.converged,type:`u32`,access:`read`}),h.push(`converged[convergedOffset] == 0u`)),s.status?.overflow&&(m.push({name:`overflow`,view:s.status?.overflow,type:`u32`,access:`read`}),h.push(`overflow[overflowOffset] != 0u`)),m.push({name:`dispatch`,view:u.dispatch,type:`u32`,access:`read_write`});let g=[t(e,{id:`${n}-unresolved-arm`,operation:r,variant:`unresolved-arm`,bindings:m,invocationCount:1,body:`let unfinished = ${h.join(` || `)};
  dispatch[dispatchOffset] = select(0u, ${d}u, unfinished);
  dispatch[dispatchOffset + 1u] = ${f}u;
  dispatch[dispatchOffset + 2u] = ${p}u;`})];for(let[i,s]of c.entries()){let{condition:c,extraResources:l}=a(u,`${n}-unresolved-${i}`);g.push(t(e,{id:`${n}-unresolved-${i}`,operation:r,variant:s.type===`f32`?`unresolved-float`:`unresolved-word`,bindings:[{name:`unresolvedTarget`,view:s.view,type:s.type,access:`read_write`}],invocationCount:u.invocationCount,workgroupSize:256,body:s.type===`f32`?`if (index < ${s.view.length}u) {
    var nanBits = 0x7fc00000u;
    unresolvedTarget[unresolvedTargetOffset + index] = bitcast<f32>(nanBits);
  }`:`if (index < ${s.view.length}u) {
    unresolvedTarget[unresolvedTargetOffset + index] = ${o}u;
  }`,condition:c,extraResources:l}))}return g}function d(e,t){return t*Math.ceil(e/16)}function f(e,t){return`let segmentsPerRow = ${Math.ceil(e/16)}u;
  let row = index / segmentsPerRow;
  let segmentStart = (index % segmentsPerRow) * 16u;
  let segmentEnd = min(segmentStart + 16u, WIDTH);
  let rowBase = row * WIDTH;
  var runKey = 0u;
  var runStart = segmentStart;
  ${t.declareRunState??``}
  // The extra iteration at segmentEnd sees a sentinel key that flushes the last run.
  for (var column = segmentStart; column <= segmentEnd; column++) {
    var key = 0xffffffffu;
    if (column < segmentEnd) {
      key = keys[keysOffset + rowBase + column];
    }
    if (key != runKey) {
      if (runKey != 0u) {
        let runEnd = column;
        ${t.onRun}
      }
      runKey = key;
      runStart = column;
      ${t.onRunStart??``}
    }
    ${t.onPixel?`if (column < segmentEnd && key != 0u) {\n      ${t.onPixel}\n    }`:``}
  }`}export{d as a,f as i,u as n,s as o,c as r,l as t};
import{$ as e,R as t}from"./dist-BusamGlx.js";import{t as n}from"./gpu-spatial-join-prepared-D289dmT8.js";import{c as r,l as i,o as a,s as o,t as s}from"./gpu-command-graph-BdzdmV6u.js";import{D as c,E as l,O as u,T as d,_ as f,i as p,t as m,x as h,y as g}from"./gpu-contributor-utils-BIwLQON5.js";import{r as _}from"./kernel-C3tfQOn4.js";import{t as v}from"./gpu-reduction-tNgOVQ_C.js";import{t as y}from"./gpu-scan-B4NNTfF9.js";import{o as b}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as x,t as S}from"./gpu-cell-topology-BbtEmE_O.js";import{n as C}from"./spatial-weights-CqWysstN.js";var w=new WeakMap;function T(e,t,a){if(w.has(e))return a(e,t);let c=new s(e.device,{id:e.id,autotuner:e.autotuner}),l=new Map,u=new Map,d=new Map,f=O(h),p=N(t,f.copy);function h(t){if(A(e,t),d.has(t))return d.get(t);if(t instanceof n){let e=new n({id:`${t.id}-recipe-preflight`,geometry:N(t.geometry,h),storage:N(t.storage,h),partitioning:t.partitioning&&N(t.partitioning,h),leafCapacity:t.leafCapacity,spatialSort:t.spatialSort,edgeSlabs:t.edgeSlabs});return d.set(t,e),j(c,e),c.add(e),e}let a=t instanceof m?t:t instanceof i?new i({...t,data:t.data.map(_)}):t instanceof r?c.createTextureView(g(t.texture),t):t instanceof o?g(t):_(t);return d.set(t,a),a}function g(e){let t=u.get(e);if(!t){let n={...e};t=e.transient?c.createTransientTexture(n):e.frameScoped?c.importFrameTexture(n):c.importTexture(n,e.defaultTexture),u.set(e,t)}return t}function _(e){let t=l.get(e.buffer);if(!t){let n={id:e.buffer.id,byteLength:e.buffer.byteLength,usage:e.buffer.usage};t=e.buffer.transient?c.createTransientBuffer(n):c.importBuffer(n,e.buffer.defaultBuffer),l.set(e.buffer,t)}return c.createDataView(t,{format:e.format,length:e.length,byteOffset:e.byteOffset,byteStride:e.byteStride,rowByteLength:e.rowByteLength})}let v={preparation:!0,writes:[],contributors:new Set,finalizers:[]};w.set(c,v);try{a(c,p);for(let e of v.finalizers)e()}finally{w.delete(c),f.detach();for(let e of[...v.contributors].reverse())e.destroy()}let y={preparation:!1,writes:[],contributors:new Set,finalizers:[]};w.set(e,y);try{let t=a(e,p);for(let e of y.finalizers)e();for(let e of y.writes)e();return t}finally{w.delete(e)}}function E(e){return w.has(e)}function D(e,t){let n=w.get(e);n?n.finalizers.push(t):t()}function O(e){let t=e;return{copy:e=>t(e),detach:()=>{t=k}}}function k(e){return e}function A(e,t){if(!(t instanceof m)){if(t instanceof n)d(`GPURecipePreparation`,t.isDeclaredIn(e),`prepared must be declared in the caller graph before adding a recipe`);else if(t instanceof a||t instanceof i)h(`GPURecipePreparation`,e,[t]);else{let n=t instanceof r?t.texture:t;d(`GPURecipePreparation`,n.graph===e,`textures must belong to the target graph`)}}}function j(e,t){let n=w.get(e);n?.preparation&&`destroy`in t&&typeof t.destroy==`function`&&n.contributors.add(t)}function M(e,t,n){let r=w.get(e);r?(f(t,n),r.preparation||r.writes.push(()=>t.write(n))):t.write(n)}function N(s,c,l=new Map){if(!s||typeof s!=`object`)return s;if(s instanceof a||s instanceof i||s instanceof o||s instanceof r||s instanceof m||s instanceof n)return c(s);if(l.has(s))return l.get(s);if(s instanceof e||s instanceof t||ArrayBuffer.isView(s))return s;let u=Array.isArray(s)?[]:Object.create(Object.getPrototypeOf(s)),d=new Map,f=new Map,p=new Map,h,g=new Proxy(u,{get(e,t){return d.has(t)||d.set(t,Reflect.get(s,t)),N(d.get(t),c,l)},has(e,t){return p.has(t)||p.set(t,Reflect.has(s,t)),p.get(t)},ownKeys(){return h??=Reflect.ownKeys(s),h},getOwnPropertyDescriptor(e,t){if(Array.isArray(u)&&t===`length`)return Reflect.getOwnPropertyDescriptor(u,t);f.has(t)||f.set(t,Reflect.getOwnPropertyDescriptor(s,t));let n=f.get(t);return n?{configurable:!0,enumerable:n.enumerable,writable:!0,value:void 0}:void 0},set:()=>!1});return l.set(s,g),g}var P=class{graph;deferGraphAddition;contributors=[];glueStages=[];constructor(e,t=E(e)){this.graph=e,this.deferGraphAddition=t}add(e){return j(this.graph,e),E(this.graph)?D(this.graph,()=>this.graph.add(e)):this.deferGraphAddition||this.graph.add(e),this.contributors.push(e),e}addGlue(e,t){let n=`type`in t?[t]:t;E(this.graph)?D(this.graph,()=>this.graph.add(n)):this.deferGraphAddition||this.graph.add(n),this.glueStages.push({stage:e,nodes:n}),this.contributors.push({getCommandNodes:()=>n})}addAdapter(e){this.add({getCommandNodes:t=>p(t,()=>e(t))})}addDeferredStages(){if(E(this.graph))return;d(`RecipeBuilder`,this.deferGraphAddition,`deferred graph addition must be enabled`);let e=this.contributors.flatMap(e=>[...e.getCommandNodes(this.graph)]);this.graph.add(e)}};function F(e,t){let{graph:n}=e;h(t.id,n,t.bindings.map(e=>e.view));for(let e of t.bindings)l(e.view,[e.view.format],t.id);g(t.id,t.bindings.filter(e=>e.access!==`read`).map(e=>e.view),t.bindings.filter(e=>e.access===`read`).map(e=>e.view));let r=t.counts.map(e=>_(n,`${t.id}-${e.name}-values`,`uint32`,t.rowCount));e.addGlue(t.id,[b(n,{id:t.id,operation:t.operation,variant:`count-rows`,bindings:[...t.bindings,...t.counts.map((e,t)=>({name:`${e.name}Values`,view:r[t],type:`u32`,access:`read_write`}))],invocationCount:t.rowCount,declarations:t.declarations,body:`${t.prelude??``}
${t.counts.map(e=>`${e.name}Values[${e.name}ValuesOffset + index] = ${e.value};`).join(`
`)}`})]),t.counts.forEach((n,i)=>{e.add(new v({id:`${t.id}-${n.name}-sum`,input:r[i],output:n.output,operation:`sum`}))})}function I(e,t,n,r,i){return i??_(e,t,n,r)}function L(e,t,n,r,i){let a=C({rowCount:n,capacity:r}).capacity;return{offsets:I(e,`${t}-offsets`,`uint32`,n+1,i?.offsets),neighbors:I(e,`${t}-neighbors`,`uint32`,a,i?.neighbors),weights:I(e,`${t}-weights`,`float32`,a,i?.weights),...i?.distances?{distances:i.distances}:{},status:{overflow:I(e,`${t}-overflow`,`uint32`,1,i?.status?.overflow),requiredCount:I(e,`${t}-required-count`,`uint32`,1,i?.status?.requiredCount),...i?.status?.candidateOverflow?{candidateOverflow:i.status.candidateOverflow}:{}}}}function R(e,t,n,r){let i=_(e,`${t}-offsets`,`uint32`,n+1),a=b(e,{id:`${t}-offsets`,operation:`GPURecipeIdentityOffsets`,bindings:[{name:`offsets`,view:i,type:`u32`,access:`read_write`}],invocationCount:n+1,body:`offsets[offsetsOffset + index] = index;`});return r?r.addGlue(`${t}-identity-offsets`,a):e.add(a),i}function z(e,t,n){if(!t)throw Error(`${e} ${n}`)}function B(e,t,n){return`fn ${e}(key: vec2u) -> u32 {
  let rowCount = ${n};
  var low = 0u;
  var high = rowCount;
  loop {
    if (low >= high) {
      break;
    }
    let middle = low + (high - low) / 2u;
    let middleLow = ${t}[${t}Offset + 2u * middle];
    let middleHigh = ${t}[${t}Offset + 2u * middle + 1u];
    if (middleHigh < key.x || (middleHigh == key.x && middleLow < key.y)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < rowCount && ${t}[${t}Offset + 2u * low] == key.y && ${t}[${t}Offset + 2u * low + 1u] == key.x) {
    return low;
  }
  return 0xffffffffu;
}`}function V(e,t,n,r){let{cells:i,weights:a}=r,o=i.length,s=a.neighbors.length,c=_(e,`${n}-disk-cells`,`uint32x2`,o*x(r.family,{type:`disk`,k:r.rings})),l=t.add(new S({id:`${n}-disk`,family:r.family,operation:{type:`disk`,k:r.rings},cells:i,mask:r.mask,output:{cells:c}})).stride,u=_(e,`${n}-slot-counts`,`uint32`,o),d=_(e,`${n}-slot-ends`,`uint32`,o),f=`const STRIDE: u32 = ${l}u;
const ROWS: u32 = ${o}u;
const CAPACITY: u32 = ${s}u;
const NOT_FOUND: u32 = 0xffffffffu;`,p=`${f}
${B(`findStudyCell`,`cells`,`countScalar[countScalarOffset]`)}`,m=`let entryLow = disk[diskOffset + 2u * (index * STRIDE + entry)];
    let entryHigh = disk[diskOffset + 2u * (index * STRIDE + entry) + 1u];
    if (entryLow == 0u && entryHigh == 0u) {
      continue;
    }
    let found = findStudyCell(vec2u(entryHigh, entryLow));
    if (found == NOT_FOUND || found == index) {
      continue;
    }`;return t.addGlue(`slot-counts`,b(e,{id:`${n}-slot-counts`,operation:`GPUCellDiskWeights`,variant:`count`,bindings:[{name:`disk`,view:c,type:`u32`,access:`read`},{name:`cells`,view:i,type:`u32`,access:`read`},{name:`countScalar`,view:r.count,type:`u32`,access:`read`},{name:`slotCounts`,view:u,type:`u32`,access:`read_write`}],invocationCount:o,declarations:p,body:`var count = 0u;
  for (var entry = 0u; entry < STRIDE; entry++) {
    ${m}
    count++;
  }
  slotCounts[slotCountsOffset + index] = count;`})),t.add(new y({id:`${n}-slot-scan`,input:u,output:d,mode:`inclusive`})),t.addGlue(`offsets`,b(e,{id:`${n}-offsets`,operation:`GPUCellDiskWeights`,variant:`offsets`,bindings:[{name:`slotEnds`,view:d,type:`u32`,access:`read`},{name:`rowOffsets`,view:a.offsets,type:`u32`,access:`read_write`},{name:`overflow`,view:r.overflow,type:`u32`,access:`read_write`},...r.requiredCount?[{name:`requiredCount`,view:r.requiredCount,type:`u32`,access:`read_write`}]:[]],invocationCount:o+1,declarations:f,body:`var rowOffset = 0u;
  if (index > 0u) {
    rowOffset = min(slotEnds[slotEndsOffset + index - 1u], CAPACITY);
  }
  rowOffsets[rowOffsetsOffset + index] = rowOffset;
  if (index == ROWS) {
    overflow[overflowOffset] = select(0u, 1u, slotEnds[slotEndsOffset + ROWS - 1u] > CAPACITY);
    ${r.requiredCount?`requiredCount[requiredCountOffset] = slotEnds[slotEndsOffset + ROWS - 1u];`:``}
  }`})),t.addGlue(`write`,b(e,{id:`${n}-write`,operation:`GPUCellDiskWeights`,variant:`write`,bindings:[{name:`disk`,view:c,type:`u32`,access:`read`},{name:`cells`,view:i,type:`u32`,access:`read`},{name:`countScalar`,view:r.count,type:`u32`,access:`read`},{name:`slotEnds`,view:d,type:`u32`,access:`read`},{name:`neighbors`,view:a.neighbors,type:`u32`,access:`read_write`},{name:`weightValues`,view:a.weights,type:`f32`,access:`read_write`}],invocationCount:o,declarations:p,body:`var start = 0u;
  if (index > 0u) {
    start = slotEnds[slotEndsOffset + index - 1u];
  }
  var written = 0u;
  for (var entry = 0u; entry < STRIDE; entry++) {
    ${m}
    let slot = start + written;
    if (slot < CAPACITY) {
      neighbors[neighborsOffset + slot] = found;
      weightValues[weightValuesOffset + slot] = 1.0;
    }
    written++;
  }
  // Ascending neighbor index within the row (insertion sort of the slots that fit).
  let stored = select(0u, min(written, CAPACITY - start), start < CAPACITY);
  for (var i = 1u; i < stored; i++) {
    let value = neighbors[neighborsOffset + start + i];
    var j = i;
    loop {
      if (j == 0u || neighbors[neighborsOffset + start + j - 1u] <= value) {
        break;
      }
      neighbors[neighborsOffset + start + j] = neighbors[neighborsOffset + start + j - 1u];
      j--;
    }
    neighbors[neighborsOffset + start + j] = value;
  }`})),l}function H(e,t,n,r,i){return i&&(l(i,[n],t),c(t,`provided output`,i,r),h(t,e,[i])),I(e,t,n,r,i)}function U(e,t){h(t.id,e,t.bindings.map(e=>e.view));for(let e of t.bindings)l(e.view,[e.view.format],t.id);return g(t.id,t.bindings.filter(e=>e.access!==`read`).map(e=>e.view),t.bindings.filter(e=>e.access===`read`).map(e=>e.view)),b(e,t)}function W(e,t,n,r){h(t,e,[...n,...r]),g(t,r,n);let i=r.filter(e=>e!==void 0).map(e=>e.buffer.defaultBuffer?u(e.buffer.defaultBuffer):e.buffer);d(t,new Set(i).size===i.length,`outputs must not share a buffer`)}export{R as a,B as c,W as d,T as f,z as i,I as l,V as n,U as o,M as p,F as r,L as s,P as t,H as u};
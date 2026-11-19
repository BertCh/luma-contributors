import{T as e,v as t}from"./gpu-contributor-utils-BIwLQON5.js";import{t as n}from"./common-CshkX-d_.js";import{r}from"./kernel-C3tfQOn4.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as a,t as o}from"./chunked-weights-kernels-BYYFtXmd.js";import{n as s,t as c}from"./philox-random-DfVxWNPn.js";import{t as l}from"./pairwise-sum-wgsl-iErcC5Or.js";var u=4096,d=16384,f=8388608,p=c.spatialWeightsTracesProbeCounter.id,m=c.spatialWeightsTracesProbeKey.id;function h(e,t,n,r,i,a,o){if(!Number.isInteger(t)||t<1||t>256)throw Error(`${e} maximumPower must be an integer in [1, 256]`);if(!/^[A-Za-z_][A-Za-z0-9_]*$/.test(n))throw Error(`${e} bindingName must be a WGSL identifier`);return`
fn ${e}(rho: f32) -> f32 {
  // The stored powers are those of W / s, so the series runs in t = rho * s.
  let t = rho * (${o});
  var accumulator = 0.0;
  for (var step = 0u; step < ${t}u; step++) {
    let q = ${t}u - step;
    accumulator = accumulator * t + ${`${n}[${n}Offset + q - 1u]${r?` / f32(q)`:``}`};
  }
  return ${a}t * accumulator${i};
}
`}function g(e,t,n=`1.0`,r=!1){let i=h(r?`getSpatialLogDeterminantTruncated`:`getSpatialLogDeterminant`,e,t,!0,``,`-`,n);return r?`${i}
fn getSpatialLogDeterminant(rho: f32) -> f32 {
  return getSpatialLogDeterminantTruncated(rho) + getSeriesTailLogDeterminant(rho * (${n}));
}
`:i}function _(e,t,n,r=`1.0`){return h(`getSpatialMultiplierTrace`,e,t,!1,` + f32(${n}u)`,``,r)}function v(e,t,n,r=`1.0`){return h(`getSpatialMultiplierTotal`,e,t,!1,` + f32(${n}u)`,``,r)}function y(e){return e?`${s}
fn getProbeValue(seed: u32, probe: u32, row: u32) -> f32 {
  let randomBlock = philox4x32(
    vec4<u32>(probe, row >> 2u, 0u, ${p}u),
    vec2<u32>(seed, ${m}u)
  );
  let word = randomBlock[row & 3u];
  return select(1.0, -1.0, (word & 1u) == 1u);
}`:`fn getProbeValue(seed: u32, probe: u32, row: u32) -> f32 {
  _ = seed;
  return select(0.0, 1.0, row == probe);
}`}var b=(e,t,n)=>({name:e,view:t,type:`f32`,access:n}),x=(e,t,n)=>({name:e,view:t,type:`u32`,access:n});function S(e,t,n){return(r,a,o,s,c,l={})=>i(e,{id:`${t}-${r}`,operation:n,variant:r.replace(/-\d+(-\d+)?$/,``),bindings:a,invocationCount:o,declarations:s,body:c,...l.constants?{constants:l.constants}:{},...l.workgroupSize?{workgroupSize:l.workgroupSize}:{}})}function C(e){return Math.ceil(e/64)*64}function w(t,n,r,i){return e(`spatial-maximum-likelihood-aligned-view`,r%64==0,`elementOffset must be a multiple of 64`),t.createDataView(n.buffer,{format:n.format,length:i,byteOffset:n.byteOffset+r*n.rowByteLength})}function T(e,t,n,i,a,o){let s=Math.ceil(i/a),c=C(a),u=r(e,`${t}-batch-information`,`uint32`,s*64);return{node:S(e,t,n)(`batch-information`,[x(`batchInformation`,u,`read_write`),...o?[x(`seedParameters`,o,`read`)]:[]],s,`const BATCH_PROBES: u32 = ${a}u;
const PROBE_TOTAL: u32 = ${i}u;
const SLOT_WORDS: u32 = 64u;`,`let start = index * BATCH_PROBES;
  batchInformation[batchInformationOffset + index * SLOT_WORDS] = start;
  batchInformation[batchInformationOffset + index * SLOT_WORDS + 1u] = min(BATCH_PROBES, PROBE_TOTAL - start);
  batchInformation[batchInformationOffset + index * SLOT_WORDS + 2u] = ${o?`seedParameters[seedParametersOffset]`:`0u`};`),batchProbeCount:a,batchCount:s,blockStride:c,getBatchInformation:t=>w(e,u,t*64,3),getBlock:(t,n)=>w(e,t,n*c,a),getSumWGSL:(e,t=`sum`,n=`0u`)=>l({count:`${i}u`,name:t,term:`${e}[${e}Offset + ${n} + (${t}Item / ${a}u) * ${c}u + ${t}Item % ${a}u]`}),getCenteredSquareSumWGSL:(e,t,n=`squareSum`,r=`0u`)=>l({count:`${i}u`,name:n,setup:`let ${n}Deviation = ${e}[${e}Offset + ${r} + (${n}Item / ${a}u) * ${c}u + ${n}Item % ${a}u] - (${t});`,term:`${n}Deviation * ${n}Deviation`})}}function E(e,t){let{id:i,rowCount:s,maximumPower:c,weights:u,tileRowCount:d,workspace:p}=t,m=o(u,d),h=m.tileCount,g=t.mode===`stochastic`,_=g?t.probeCount:s,v=n(Math.floor(f/s),1,_),C=S(e,i,t.operation),w=(t,n)=>r(e,`${i}-${t}`,`float32`,n),E=[w(`probe-a`,v*s),w(`probe-b`,v*s)],D=w(`probe-partial`,v*h),O=T(e,i,t.operation,_,v,t.seedParameters),k=w(`probe-dots`,O.batchCount*O.blockStride),A=`const ROWS: u32 = ${s}u;
const TILE_ROWS: u32 = ${d}u;
const TILE_COUNT: u32 = ${h}u;
const PARAMETER_SLOT: u32 = ${t.parameterSlot}u;`,j=e=>[x(`offsets`,e.offsets,`read`),x(`neighbors`,e.neighbors,`read`),b(`weights`,e.weights,`read`)],M=[O.node];for(let e=0;e<O.batchCount;e++){let t=x(`batchInformation`,O.getBatchInformation(e),`read`);for(let n=1;n<=c;n++){let r=n===c,i=n===1,o=E[n%2],s=E[(n-1)%2],l=e=>[...j(e),...i?[]:[b(`vectorIn`,o,`read`)],r?b(`partial`,D,`read_write`):b(`vectorOut`,s,`read_write`),b(`workspace`,p,`read`),t],u=i?`getProbeValue(probeSeed, globalProbe, neighbor)`:`vectorIn[vectorInOffset + base + neighbor]`;M.push(...m.dispatches.map(t=>C(`probe-${r?`final`:i?`first`:`step`}-${e}-${n}${t.idSuffix}`,l(t.weights),v*t.chunkTileCount,`${A}${t.tileConstantsWGSL?`\n${t.tileConstantsWGSL}`:``}
${y(g)}`,a(t,`let probe = index / TILE_COUNT;
  if (probe >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  let rho = workspace[workspaceOffset + PARAMETER_SLOT];
  let tile = index % TILE_COUNT;
  let globalProbe = batchInformation[batchInformationOffset] + probe;
  let probeSeed = batchInformation[batchInformationOffset + 2u];
  let base = probe * ROWS;
  let firstRow = tile * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var squares = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += weights[weightsOffset + slot] * ${u};
    }
${r?`    squares += sum * sum;`:`    vectorOut[vectorOutOffset + base + row] = getProbeValue(probeSeed, globalProbe, row) + rho * sum;`}
  }
${r?`  partial[partialOffset + probe * TILE_COUNT + tile] = squares;`:`  _ = squares;`}`),{workgroupSize:64})))}M.push(C(`probe-merge-${e}`,[b(`partial`,D,`read`),b(`dots`,O.getBlock(k,e),`read_write`),t],v,A,`if (index >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  ${l({count:`TILE_COUNT`,term:`partial[partialOffset + index * TILE_COUNT + sumItem]`})}
  dots[dotsOffset + index] = sum;`))}return{nodes:M,dots:k,dotsSumWGSL:O.getSumWGSL(`dots`),probeTotal:_}}function D(n,r,i){for(let[a,o]of Object.entries(i))t(n,r,a,Math.max(1,o)*4,`reduce rows, probes or powers`),e(r,Math.max(1,o)*4<=n.limits.maxBufferSize,`${a} allocation must fit maxBufferSize (${n.limits.maxBufferSize})`)}function O(e,t,n,r,i=1/0){let a=Math.min(n,i,Math.max(1,Math.floor(f/e))),o=Math.ceil(n/a),s=o*C(a);return{"probe vectors":a*e,"probe partials":a*t,"probe batch information":o*64,"probe dots":r*s}}function k(e){let{rowCount:t,edgeCount:n,predictorCount:r}=e,i=e.maximumPower??128,a=(e.traceMode??(t<=4096?`exact`:`stochastic`))===`exact`?t:e.traceProbeCount??64,o=e.tileRowCount??Math.max(64,Math.ceil(t/4096)),s=e.tileCount??Math.ceil(t/o),c=Math.max(32,Math.ceil(t/1024)),l=e.traceTileCount??Math.ceil(t/c),u=r+2,d=u*(u+1)/2,f=r*(r+1)/2+r,p=O(t,l,a,i),m=O(t,s,a,1);return{"likelihood workspace":22+4*r,"power traces/sums":i,"likelihood norm partials":s,"likelihood mean partials":s*(r+1),"likelihood OLS partials":s*f,"likelihood OLS totals":f,"likelihood Gram partials":3*s*d,"likelihood Gram totals":3*d,"likelihood residuals":t,"likelihood residual partials":s*4,"likelihood residual totals":4,"trace infinity norm":1,"trace norm partials":l,"trace scaled weights":n,"trace sums partials":i*C(l),"trace ones vectors":t,...Object.fromEntries(Object.entries(p).map(([e,t])=>[`trace ${e}`,t])),...Object.fromEntries(Object.entries(m).map(([e,t])=>[`information ${e}`,t]))}}function A(e,t,n,r=64){let i=n??(e<=4096?`exact`:`stochastic`),a=i===`exact`?e:r,o=Math.min(a,Math.max(1,Math.floor(f/e))),s=Math.ceil(a/o);return 128+2*t+s*((i===`stochastic`?3:2)*t+1)}export{_,S as a,w as c,C as d,d as f,v as g,g as h,D as i,b as l,f as m,A as n,T as o,u as p,O as r,E as s,k as t,x as u,y as v};
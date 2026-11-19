import{o as e}from"./raster-band-utils-BcTxc8_Y.js";import{n as t}from"./raster-storage-preflight-CIxFc-0g.js";import{r as n}from"./kernel-C3tfQOn4.js";import{o as r,t as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,f as o,s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{a as u,i as d,n as f,o as p,r as m,t as h}from"./patch-labels-Bfw_yy99.js";import{a as g}from"./pointer-overlay-Qh8qflcA.js";import{t as _}from"./define-run-B-brkCe4.js";import{i as v}from"./inputs-CJKKIRKK.js";import{Di as y}from"./entry-preview-D70TdSbK.js";import{n as b,t as x}from"./bands-CVDIl0aE.js";import{c as S,n as C,r as w,t as T}from"./gestures-C1AoqRjN.js";import{i as E,n as D,r as O,t as k}from"./labels-DBumpLdz.js";import{n as A,t as j}from"./patches-DgXB1bR3.js";function M(t,n=new Uint32Array(4)){e(`getGPURasterSieveParameterValues`,n,4);let{minimumPixels:r}=t;if(!Number.isInteger(r)||r<1||r>4294967295)throw Error(`getGPURasterSieveParameterValues minimumPixels must be an integer in [1, 2^32 - 1]`);return n.set([r,0,0,0]),n}var N=`GPURasterSieve`,P=`0xffffffffu`,F=256,I=class{id;props;mode;connectivity;constructor(e){let t=e.id??`raster-sieve`;this.id=t,this.props=e,this.mode=e.mode??`remove`,this.connectivity=e.connectivity??4;let n=p(t,e);if(this.mode!==`remove`&&this.mode!==`merge`)throw Error(`${t} mode must be 'remove' or 'merge'`);if(this.connectivity!==4&&this.connectivity!==8)throw Error(`${t} connectivity must be 4 or 8`);if(o(t,`patchCapacity`,e.patchCapacity,1,n),a(e.parameters,[`uint32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must contain at least 4 values`);let{output:r}=e;if(a(r.labels,[`uint32`],`${t} output.labels`),r.labels.length!==n)throw Error(`${t} output.labels length must equal grid width * height`);if(r.patchTargets&&(a(r.patchTargets,[`uint32`],`${t} output.patchTargets`),r.patchTargets.length!==e.patchCapacity))throw Error(`${t} output.patchTargets length must equal patchCapacity`);if(r.sievedCount&&(a(r.sievedCount,[`uint32`],`${t} output.sievedCount`),r.sievedCount.length<1))throw Error(`${t} output.sievedCount must hold one uint32`);l(t,[r.labels,r.patchTargets,r.sievedCount],[...m(e),e.parameters])}getCommandNodes(e){let a=this.props,o=a.grid.width*a.grid.height;t(e,this.id,[...m(a),a.parameters,...s(a.output)],{"patch keys":o*4,"patch scalar planes":a.patchCapacity*4,"patch guard state":a.status?.converged||a.status?.overflow?16:0});let{id:l,props:p,mode:g,connectivity:_}=this,{output:v,patchCapacity:y,parameters:b}=p,{width:x,height:S}=p.grid;c(l,e,[...m(p),b,v.labels,v.patchTargets,v.sievedCount]);let C=x*S,w=(t,r)=>n(e,`${l}-${t}`,`uint32`,r),{node:T,keys:E}=h(e,l,N,p,y),D=w(`pixel-counts`,y),O=v.patchTargets??w(`patch-targets`,y),k=`const WIDTH: u32 = ${x}u;
const HEIGHT: u32 = ${S}u;`,A=[T,i(e,{id:`${l}-counts-clear`,operation:N,view:D,type:`u32`,value:`0u`}),r(e,{id:`${l}-counts`,operation:N,variant:`counts`,bindings:[{name:`keys`,view:E,type:`u32`,access:`read`},{name:`pixelCounts`,view:D,type:`atomic<u32>`,access:`read_write`}],invocationCount:u(x,S),declarations:k,body:d(x,{onRun:`atomicAdd(&pixelCounts[pixelCountsOffset + runKey - 1u], runEnd - runStart);`})})],j,M;if(g===`merge`){j=w(`best-counts`,y),M=w(`merge-targets`,y);let t=e=>`let key = keys[keysOffset + index];
  if (key != 0u && pixelCounts[pixelCountsOffset + key - 1u] < parameters[parametersOffset]) {
    let column = index % WIDTH;
    let row = index / WIDTH;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        if ((dx == 0 && dy == 0) || (${_} == 4 && dx != 0 && dy != 0)) {
          continue;
        }
        let neighborColumn = i32(column) + dx;
        let neighborRow = i32(row) + dy;
        if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(WIDTH) || neighborRow >= i32(HEIGHT)) {
          continue;
        }
        let neighbor = keys[keysOffset + u32(neighborRow) * WIDTH + u32(neighborColumn)];
        if (neighbor == 0u || neighbor == key) {
          continue;
        }
        let neighborCount = pixelCounts[pixelCountsOffset + neighbor - 1u];
        if (neighborCount >= parameters[parametersOffset]) {
          ${e}
        }
      }
    }
  }`;A.push(i(e,{id:`${l}-best-counts-clear`,operation:N,view:j,type:`u32`,value:`0u`}),i(e,{id:`${l}-merge-targets-clear`,operation:N,view:M,type:`u32`,value:P}),r(e,{id:`${l}-best-counts`,operation:N,variant:`best-counts`,bindings:[{name:`keys`,view:E,type:`u32`,access:`read`},{name:`pixelCounts`,view:D,type:`u32`,access:`read`},{name:`parameters`,view:b,type:`u32`,access:`read`},{name:`bestCounts`,view:j,type:`atomic<u32>`,access:`read_write`}],invocationCount:C,declarations:k,body:t(`atomicMax(&bestCounts[bestCountsOffset + key - 1u], neighborCount);`)}),r(e,{id:`${l}-merge-targets`,operation:N,variant:`merge-targets`,bindings:[{name:`keys`,view:E,type:`u32`,access:`read`},{name:`pixelCounts`,view:D,type:`u32`,access:`read`},{name:`parameters`,view:b,type:`u32`,access:`read`},{name:`bestCounts`,view:j,type:`u32`,access:`read`},{name:`targets`,view:M,type:`atomic<u32>`,access:`read_write`}],invocationCount:C,declarations:k,body:t(`if (neighborCount == bestCounts[bestCountsOffset + key - 1u]) {
            atomicMin(&targets[targetsOffset + key - 1u], neighbor);
          }`)}))}let I=v.sievedCount;return A.push(...I?[i(e,{id:`${l}-sieved-clear`,operation:N,view:I,type:`u32`,value:`0u`,componentCount:1})]:[],r(e,{id:`${l}-patch-targets`,operation:N,variant:`patch-targets`,bindings:[{name:`pixelCounts`,view:D,type:`u32`,access:`read`},{name:`parameters`,view:b,type:`u32`,access:`read`},{name:`patchTargets`,view:O,type:`u32`,access:`read_write`},...j&&M?[{name:`bestCounts`,view:j,type:`u32`,access:`read`},{name:`targets`,view:M,type:`u32`,access:`read`}]:[],...I?[{name:`sievedCount`,view:I,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:y,...I?{guardIndex:!1,workgroupSize:F,declarations:`var<workgroup> workgroupSievedCount: atomic<u32>;`}:{},body:`${I?`if (localInvocationIndex == 0u) {
    atomicStore(&workgroupSievedCount, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {`:``}
  let count = pixelCounts[pixelCountsOffset + index];
  var finalLabel = 0u;
  if (count >= parameters[parametersOffset]) {
    finalLabel = index + 1u;
  } else if (count > 0u) {
    ${I?`atomicAdd(&workgroupSievedCount, 1u);`:``}
    ${g===`merge`?`if (bestCounts[bestCountsOffset + index] != 0u) {
      finalLabel = targets[targetsOffset + index];
    }`:``}
  }
  patchTargets[patchTargetsOffset + index] = finalLabel;${I?`
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupTotal = atomicLoad(&workgroupSievedCount);
    if (workgroupTotal > 0u) {
      atomicAdd(&sievedCount[sievedCountOffset], workgroupTotal);
    }
  }`:``}`}),r(e,{id:`${l}-relabel`,operation:N,variant:`relabel`,bindings:[{name:`keys`,view:E,type:`u32`,access:`read`},{name:`patchTargets`,view:O,type:`u32`,access:`read`},{name:`sievedLabels`,view:v.labels,type:`u32`,access:`read_write`}],invocationCount:C,body:`let key = keys[keysOffset + index];
  sievedLabels[sievedLabelsOffset + index] =
    select(0u, patchTargets[patchTargetsOffset + key - 1u], key != 0u);`})),A.push(...f(e,l,N,p,[{view:v.labels,type:`u32`},...v.patchTargets?[{view:v.patchTargets,type:`u32`}]:[],...I?[{view:I,type:`u32`}]:[]])),A}},L=90,R=g(y,{kind:`level`,hint:`raise the size threshold`,x:{parameter:`minimumPixels`,scale:`logarithmic`}}),z=_({parameters:y,rebuildOn:[`mode`,`connectivity`],caption:`Patch labels in (graph-coloured so neighbours differ); small patches removed or merged into neighbors out.`,createInputs:({random:e,world:t})=>{let n=v(t,54),r=j(e,n.columns,n.rows,L,n.values),i=A(r,L);return{...n,labels:i,sizes:D(i),colors:C(i,n.columns,n.rows),focus:{label:0}}},build(e,t,n){let r=t.values.length,i=e.upload(`labels`,`uint32`,t.labels),a=e.allocate(`parameters`,`uint32`,4),o=e.allocate(`sieved`,`uint32`,r);e.graph.add(new I({grid:b(t),labels:i.view,patchCapacity:91,mode:n.mode===1?`merge`:`remove`,connectivity:n.connectivity===1?8:4,parameters:a.view,output:{labels:o.view}}));let s=e.upload(`colored`,`uint32`,T(t.labels,t.colors)),c=e.allocate(`hover`,`uint32`,r);return{outputs:{labels:i,sieved:o,colored:s,hover:c},update:e=>{a.write(M({minimumPixels:e.minimumPixels})),c.write(E(t.labels,t.focus.label))}}},async afterRun({inputs:e,outputs:t,upload:n}){let r=S(await t.sieved.buffer.readAsync());return{display:n(`display`,`uint32`,T(r,e.colors))}},interaction:{...R,resolve:(e,t)=>{let n=R.resolve(e,t),r={...t.values,...n.values},{inputs:i,outputs:a,world:o}=t,{index:s}=w(i,o,e),c=i.labels[s];i.focus.label=c,a.hover.write(E(i.labels,c));let l=i.sizes[c],u=0;for(let e=1;e<i.sizes.length;e++)i.sizes[e]>0&&i.sizes[e]<r.minimumPixels&&u++;return{...n,readout:`minimumPixels ${r.minimumPixels} | ${u} patches sieved | this one ${l} px`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>[x(e,t.colored,{context:!0,ramp:`categorical`,categories:O(r)}),...n?[x(e,n.display,{ramp:`categorical`,categories:O(r)})]:[],x(e,t.hover,{ramp:`categorical`,categories:k})]});export{z as default};
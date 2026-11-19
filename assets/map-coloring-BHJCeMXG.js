import{r as e}from"./kernel-C3tfQOn4.js";import{o as t,t as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,v as i,x as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{a as s,c,f as l,i as u,m as d,s as f}from"./raster-relaxation-Di9wjtxp.js";import{t as p}from"./hash-uint32-wL4xFYeV.js";import{a as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as h}from"./geographic-distribution-validation-BjMQQJgk.js";import{r as g}from"./geometry-storage-validation-T89ZhmCj.js";import{i as _,s as v,t as y}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{i as b}from"./chunked-weights-kernels-BYYFtXmd.js";import{t as x}from"./define-run-B-brkCe4.js";import{m as S,u as C}from"./inputs-CJKKIRKK.js";import{a as w}from"./pick-Bo3t3hjv.js";import{Bo as T}from"./entry-preview-D70TdSbK.js";var E=`GPUMapColoring`,D=`
  if (otherDegree != rowDegree) {
    return otherDegree > rowDegree;
  }`,O=4294967295;function k(e={}){let t=e.seed??0;if(!Number.isInteger(t)||t<0||t>4294967295)throw Error(`GPUMapColoring seed must be a uint32`);return Uint32Array.of(t)}var A=class{id;props;maximumIterations;priority;constructor(e){let t=e.id??`map-coloring`;if(this.id=t,this.props=e,this.maximumIterations=e.maximumIterations??64,d(t,`maximumIterations`,this.maximumIterations),this.priority=e.priority??`largest-degree-first`,this.priority!==`largest-degree-first`&&this.priority!==`random`)throw Error(`${t} priority must be 'largest-degree-first' or 'random'`);let n=v(t,e.weights),{output:i}=e;if(r(e.parameters,[`uint32`],`${t} parameters`),e.parameters.length<1)throw Error(`${t} parameters must hold 1 uint32`);if(r(i.colors,[`uint32`],`${t} output.colors`),i.colors.length!==n)throw Error(`${t} output.colors length must equal the weights row count`);for(let[e,n]of[[`colorCount`,i.colorCount],[`conflictCount`,i.conflictCount],[`status.converged`,i.status?.converged],[`status.iterationCount`,i.status?.iterationCount]])if(n&&(r(n,[`uint32`],`${t} output.${e}`),n.length<1))throw Error(`${t} output.${e} must hold one uint32`);o(t,j(i),[..._(e.weights),e.parameters]),h(t,{},{colors:i.colors,colorCount:i.colorCount,conflictCount:i.conflictCount,converged:i.status?.converged,iterationCount:i.status?.iterationCount},[{reads:{},writes:{colors:i.colors,colorCount:i.colorCount}},{reads:{},writes:{converged:i.status?.converged,iterationCount:i.status?.iterationCount}}]),g(t,[i.colors],[i.conflictCount])}getCommandNodes(r){let{id:o,props:d,maximumIterations:h,priority:g}=this,{weights:v,parameters:x,output:S}=d,{colors:C}=S,w=y(v);m(r,o,{parameters:x,...Object.fromEntries(j(S).map((e,t)=>[`output.${t}`,e])),...Object.fromEntries(w.flatMap((e,t)=>[[`offsets.${t}`,e.weights.offsets],[`neighbors.${t}`,e.weights.neighbors]]))});for(let[e,t]of[[`color iteration status scratch`,16],[`color indirect dispatch scratch`,12],[`color count scratch`,4],...g===`largest-degree-first`&&!w[0].isWhole?[[`color degree scratch`,C.length*4]]:[]])i(r.device,o,e,t,`reduce row count`);a(o,r,[..._(v),x,...j(S)]);let T=C.length,k=c(r,`${o}-rounds`,E,T,o),A=S.colorCount??e(r,`${o}-color-count`,`uint32`,1),M=g===`largest-degree-first`,N=`const UNCOLORED: u32 = ${O}u;

${p}
fn priorityHash(value: u32, seed: u32) -> u32 {
  return hashUint32(value ^ seed);
}

// True when row 'other' is claimed before row 'index' (a strict total order: no two rows tie).
fn outranks(other: u32, otherDegree: u32, index: u32, rowDegree: u32, seed: u32) -> bool {${M?D:``}
  let otherHash = priorityHash(other, seed);
  let rowHash = priorityHash(index, seed);
  return otherHash > rowHash || (otherHash == rowHash && other < index);
}`,P=[n(r,{id:`${o}-colors-clear`,operation:E,view:C,type:`u32`,value:`${O}u`}),n(r,{id:`${o}-color-count-clear`,operation:E,view:A,type:`u32`,value:`0u`,componentCount:1}),f(r,{id:`${o}-rounds-reset`,operation:E,state:k})],F=M&&!w[0].isWhole?e(r,`${o}-degrees`,`uint32`,T):void 0;F&&P.push(...w.map(e=>t(r,{id:`${o}-degrees${e.idSuffix}`,operation:E,variant:`degrees`,bindings:[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`degrees`,view:F,type:`u32`,access:`read_write`}],invocationCount:e.rowCount,declarations:e.constantsWGSL,body:`degrees[degreesOffset + WEIGHTS_ROW_START + index] = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];`})));for(let e=0;e<h;e++){let n=`${o}-round-${e}`,i=l(k,n);P.push(...w.map(e=>{let a=M?F?`degrees[degreesOffset + other]`:`offsets[offsetsOffset + other + 1u] - offsets[offsetsOffset + other]`:`0u`;return t(r,{id:`${n}-select${e.idSuffix}`,operation:E,variant:`select-apply`,bindings:[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.weights.neighbors,type:`u32`,access:`read`},{name:`parameters`,view:x,type:`u32`,access:`read`},{name:`colors`,view:C,type:`atomic<u32>`,access:`read_write`},{name:`status`,view:k.status,type:`atomic<u32>`,access:`read_write`},{name:`colorCount`,view:A,type:`atomic<u32>`,access:`read_write`},...F?[{name:`degrees`,view:F,type:`u32`,access:`read`}]:[]],invocationCount:e.rowCount,declarations:`${N}${e.constantsWGSL?`\n${e.constantsWGSL}`:``}`,body:b(e,`if (atomicLoad(&colors[colorsOffset + index]) != UNCOLORED) {
    return;
  }
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  let seed = parameters[parametersOffset];
  let rowDegree = ${M?`end - begin`:`0u`};
  // One neighbor pass finds both a blocking (uncolored, higher-priority) neighbor and the used
  // colors of the first 32-color window.
  var used = 0u;
  var blocked = false;
  for (var slot = begin; slot < end; slot++) {
    let other = neighbors[neighborsOffset + slot];
    if (other == index) {
      continue;
    }
    let color = atomicLoad(&colors[colorsOffset + other]);
    if (color == UNCOLORED) {
      let otherDegree = ${a};
      if (outranks(other, otherDegree, index, rowDegree, seed)) {
        blocked = true;
        break;
      }
    } else if (color < 32u) {
      used = used | (1u << color);
    }
  }
  if (blocked) {
    atomicStore(&status[statusOffset], 1u);
    return;
  }
  var chosen = UNCOLORED;
  if (used != 0xffffffffu) {
    chosen = firstTrailingBit(~used);
  } else {
    // Rare: 32 or more colors used around this row; scan the following windows.
    var base = 32u;
    for (var window = 1u; window <= (end - begin) / 32u; window++) {
      var windowUsed = 0u;
      for (var slot = begin; slot < end; slot++) {
        let other = neighbors[neighborsOffset + slot];
        let color = atomicLoad(&colors[colorsOffset + other]);
        if (other != index && color != UNCOLORED && color >= base && color < base + 32u) {
          windowUsed = windowUsed | (1u << (color - base));
        }
      }
      if (windowUsed != 0xffffffffu) {
        chosen = base + firstTrailingBit(~windowUsed);
        break;
      }
      base += 32u;
    }
  }
  if (chosen != UNCOLORED) {
    atomicStore(&colors[colorsOffset + index], chosen);
    atomicMax(&colorCount[colorCountOffset], chosen + 1u);
  }`),condition:i.condition,extraResources:i.extraResources})}),s(r,{id:`${n}-gate`,operation:E,state:k,maxIterations:h}))}return(S.status?.converged||S.status?.iterationCount)&&P.push(u(r,{id:`${o}-status`,operation:E,state:k,converged:S.status?.converged,iterationCount:S.status?.iterationCount})),S.conflictCount&&P.push(n(r,{id:`${o}-conflicts-clear`,operation:E,view:S.conflictCount,type:`u32`,value:`0u`,componentCount:1}),...w.map(e=>t(r,{id:`${o}-conflicts${e.idSuffix}`,operation:E,variant:`conflicts`,bindings:[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.weights.neighbors,type:`u32`,access:`read`},{name:`colors`,view:C,type:`u32`,access:`read`},{name:`conflictCount`,view:S.conflictCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:e.rowCount,...e.constantsWGSL?{declarations:e.constantsWGSL}:{},body:b(e,`let color = colors[colorsOffset + index];
  var conflicts = 0u;
  if (color != ${O}u) {
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let other = neighbors[neighborsOffset + slot];
      if (other > index && colors[colorsOffset + other] == color) {
        conflicts++;
      }
    }
  }
  if (conflicts > 0u) {
    atomicAdd(&conflictCount[conflictCountOffset], conflicts);
  }`)}))),P}};function j(e){return[e.colors,e.colorCount,e.conflictCount,e.status?.converged,e.status?.iterationCount]}var M=x({parameters:T,caption:`Tracts colored so that no two neighbors share a color.`,createInputs:({random:e,world:t})=>C(e,t,9),build(e,t,n){let r=S(e,t.contiguity),i=e.allocate(`seed`,`uint32`,1),a=e.allocate(`colors`,`uint32`,t.count),o=e.allocate(`color-count`,`uint32`,1);return e.graph.add(new A({weights:r,parameters:i.view,priority:n.priority===0?`largest-degree-first`:`random`,output:{colors:a.view,colorCount:o.view}})),{outputs:{colors:a,vertices:e.upload(`vertices`,`float32x2`,t.vertices),indices:e.upload(`indices`,`uint32`,t.triangleIndices),owners:e.upload(`owners`,`uint32`,t.triangleOwners),outline:e.upload(`outline`,`float32x4`,t.outline)},update:e=>i.write(k({seed:e.seed}))}},rebuildOn:[`priority`],interaction:{kind:`probe`,hint:`hover a tract`,resolve(e,{inputs:t,outputs:n,peek:r}){let i=w(t.vertices,t.triangleIndices,t.triangleOwners,e.x,e.y,t.triangleCount);if(i<0)return{};let a=e=>Array.from(t.vertices.subarray(t.ringOffsets[e]*2,t.ringOffsets[e+1]*2)),o=[{shape:`polyline`,points:a(i),closed:!0}],s=r(n.colors),c=0,l=0;for(let e=t.contiguity.offsets[i];e<t.contiguity.offsets[i+1];e++){let n=t.contiguity.neighbors[e];c++,s&&s[n]===s[i]&&l++,o.push({shape:`polyline`,points:a(n),closed:!0,dashed:!0})}return{focus:i,glyphs:o,readout:`tract ${i}${s?` · color ${s[i]}`:``} · ${c} neighbors, ${l} share it`}}},draw:({inputs:e,outputs:t})=>[{layer:`polygons`,positions:t.vertices,indices:t.indices,owners:t.owners,values:t.colors,triangleCount:e.triangleCount,alpha:.9},{layer:`lines`,segments:t.outline,color:[.05,.07,.1,.9],radius:1.4,count:e.outlineCount}]});export{M as default};
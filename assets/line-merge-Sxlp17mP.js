import{s as e}from"./hydrography-N8loF-Ds.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{t as r}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{o as i,t as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{c}from"./wgsl-common-DfyXiYB0.js";import{v as l}from"./spatial-context-DwZEd4EZ.js";import{r as u,t as d}from"./list-ranking-SHiVJGmq.js";import{a as f}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as p,t as ee}from"./geometry-storage-validation-T89ZhmCj.js";import{t as m}from"./define-run-B-brkCe4.js";import{f as h,g,l as _,r as v}from"./lines-kx5qHdti.js";import{Ko as y}from"./entry-preview-D70TdSbK.js";var b=`GPULineMerge`,x=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`])}),S=class{id;props;spatialContext;lineCount;constructor(e){this.id=e.id??`line-merge`,this.spatialContext=l(this.id,e.spatialContext,x),this.props=e;let{id:t}=this,{output:n}=e;o(e.positions,[`float32x2`],`${t} positions`),o(n.positions,[`float32x2`],`${t} output.positions`);for(let[r,i]of[[`lineOffsets`,e.lineOffsets],[`output.chainOffsets`,n.chainOffsets],[`output.status.count`,n.status.count],[`output.status.requiredCount`,n.status.requiredCount],[`output.status.overflow`,n.status.overflow],[`output.status.vertexCount`,n.status.vertexCount],[`output.status.requiredVertexCount`,n.status.requiredVertexCount],[`output.sourceIds`,n.sourceIds],[`output.lineChains`,n.lineChains],[`output.lineOrders`,n.lineOrders],[`output.lineReversed`,n.lineReversed]])i&&o(i,[`uint32`],`${t} ${r}`);if(e.lineOffsets.length<2)throw Error(`${t} lineOffsets needs at least one line`);if(this.lineCount=e.lineOffsets.length-1,n.chainOffsets.length<this.lineCount+1)throw Error(`${t} output.chainOffsets must contain lineCount + 1 rows`);if(n.positions.length<e.positions.length)throw Error(`${t} output.positions must hold at least the input vertex count`);if(n.status.count.length!==1)throw Error(`${t} output.status.count must contain exactly one row`);for(let[e,r]of[[`requiredCount`,n.status.requiredCount],[`overflow`,n.status.overflow],[`vertexCount`,n.status.vertexCount],[`requiredVertexCount`,n.status.requiredVertexCount]])if(r&&r.length!==1)throw Error(`${t} output.status.${e} must contain exactly one row`);for(let[e,r]of[[`sourceIds`,n.sourceIds],[`lineChains`,n.lineChains],[`lineOrders`,n.lineOrders],[`lineReversed`,n.lineReversed]])if(r&&r.length!==this.lineCount)throw Error(`${t} output.${e} length must equal the line count`);let r=w(e);s(t,r,C(e));let i=r.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(i).size!==i.length)throw Error(`${t} outputs must use separate buffers`)}getCommandNodes(e){let{id:o,props:s,lineCount:l}=this,{output:m}=s;f(e,o,Object.fromEntries([...C(s),...w(s)].map((e,t)=>[String(t),e]))),p(e,o,{...ee(l*2),lineColumnsAndScan:l*4});let h=l*2,g=s.directed===!0,_=(n,r,i)=>t(e,`${o}-${n}`,r,Math.max(i,1)),v=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),y=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),x=(t,n,r,a,s=``,c=``)=>i(e,{id:`${o}-${t}${c}`,operation:b,variant:t,bindings:n,invocationCount:r,declarations:`const LINE_COUNT: u32 = ${l}u;
const NONE: u32 = 0xffffffffu;
const INVALID_KEY: u32 = 0xffffffffu;
const DIRECTED: bool = ${g};
${s}`,body:a}),S=_(`key-y`,`uint32`,h),T=_(`key-x`,`uint32`,h),E=_(`endpoint-ids`,`uint32`,h),D=_(`partner`,`uint32`,h),O=_(`leader-exit`,`uint32`,l),k=_(`leader-vertices`,`uint32`,l),A=_(`flags`,`uint32`,l),j=_(`flag-scan`,`uint32`,l),M=_(`vertex-scan`,`uint32`,l),N=[[`requiredCount`,m.status.requiredCount,`requiredCount[requiredCountOffset] = chainCount;`],[`overflow`,m.status.overflow,`overflow[overflowOffset] = 0u;`],[`vertexCount`,m.status.vertexCount,`vertexCount[vertexCountOffset] = totalVertices;`],[`requiredVertexCount`,m.status.requiredVertexCount,`requiredVertexCount[requiredVertexCountOffset] = totalVertices;`]],P=N.flatMap(([e,t])=>t?[y(e,t)]:[]),F=N.flatMap(([,e,t])=>e?[t]:[]).join(`
  `),te=N.map(([,e])=>+!!e).join(``),I=[],L=r(e,{id:`${o}-sort`,operation:b,indices:E,indicesAreIdentity:!0,keys:[{view:T},{view:S}]}),ne=L.sortedIndices,re=L.sortedPrimaryKeys;I.push(x(`keys`,[v(`positions`,s.positions,`f32`),v(`lineOffsets`,s.lineOffsets),y(`keyY`,S),y(`keyX`,T),y(`endpointIds`,E)],h,`let line = index / 2u;
  let first = lineOffsets[lineOffsetsOffset + line];
  let last = lineOffsets[lineOffsetsOffset + line + 1u];
  endpointIds[endpointIdsOffset + index] = index;
  var x = 0.0;
  var y = 0.0;
  var valid = last >= first + 2u;
  if (valid) {
    let vertex = select(first, last - 1u, (index & 1u) == 1u);
    x = positions[positionsOffset + vertex * 2u];
    y = positions[positionsOffset + vertex * 2u + 1u];
    valid = isFiniteBits(x) && isFiniteBits(y);
  }
  if (valid) {
    x = select(x, 0.0, x == 0.0);
    y = select(y, 0.0, y == 0.0);
    keyX[keyXOffset + index] = bitcast<u32>(x);
    keyY[keyYOffset + index] = bitcast<u32>(y);
  } else {
    keyX[keyXOffset + index] = INVALID_KEY;
    keyY[keyYOffset + index] = INVALID_KEY;
  }`,c),...L.nodes,x(`partners`,[v(`sortedX`,re),v(`order`,ne),v(`keyY`,S),y(`partner`,D)],h,`let end = order[orderOffset + index];
  var result = NONE;
  if (sortedX[sortedXOffset + index] != INVALID_KEY) {
    let isLeftSame = index > 0u && isSame(index - 1u, index);
    let isRightSame = index + 1u < ENDPOINT_COUNT && isSame(index, index + 1u);
    if (isLeftSame && !isRightSame && !(index > 1u && isSame(index - 2u, index - 1u))) {
      result = order[orderOffset + index - 1u];
    } else if (isRightSame && !isLeftSame &&
        !(index + 2u < ENDPOINT_COUNT && isSame(index + 1u, index + 2u))) {
      result = order[orderOffset + index + 1u];
    }
    if (result != NONE && (result >> 1u) == (end >> 1u)) {
      result = NONE;
    }
    // Directed joins pair a line end (odd id) with a line start (even id).
    if (DIRECTED && result != NONE && ((result ^ end) & 1u) == 0u) {
      result = NONE;
    }
  }
  partner[partnerOffset + end] = result;`,`const ENDPOINT_COUNT: u32 = ${h}u;
fn isSame(a: u32, b: u32) -> bool {
  return sortedX[sortedXOffset + a] == sortedX[sortedXOffset + b] &&
    keyY[keyYOffset + order[orderOffset + a]] == keyY[keyYOffset + order[orderOffset + b]];
}`));let R=u(l),z=!!m.lineOrders,B=`
fn vertexCount(line: u32) -> u32 {
  return lineOffsets[lineOffsetsOffset + line + 1u] - lineOffsets[lineOffsetsOffset + line];
}`,V=e=>[_(`${e}-a`,`uint32`,h),_(`${e}-b`,`uint32`,h)],H=V(`min-predecessor`),U=V(`min-key`),W=V(`rank-predecessor`),G=V(`rank-head`),K=V(`rank-sum`),q=V(`rank-count`),J=_(`leader-bits`,`uint32`,h),Y=e=>`(${e} >> 1u) * 2u + select(1u, 0u, (${e} & 1u) == 1u)`;I.push(x(`min-init`,[v(`partner`,D),y(`predecessor`,H[0]),y(`minKey`,U[0])],h,`predecessor[predecessorOffset + index] = partner[partnerOffset + (index ^ 1u)];
  minKey[minKeyOffset + index] = ${Y(`index`)};`));let ie=d(e,{id:`${o}-min-round`,operation:b,variant:`min-round`,length:h,rounds:R,pointers:H,values:[{name:`key`,views:U,rule:`minimum`}]});I.push(...ie.nodes);let X=R%2;I.push(x(`rank-init`,[v(`partner`,D),v(`lineOffsets`,s.lineOffsets),v(`cyclePredecessor`,H[X]),v(`cycleKey`,U[X]),y(`predecessor`,W[0]),y(`head`,G[0]),y(`sum`,K[0]),y(`leaderBits`,J)],h,`let isCycle = cyclePredecessor[cyclePredecessorOffset + index] != NONE;
  let minimumKey = cycleKey[cycleKeyOffset + index];
  let isCanonicalCycle = isCycle && (minimumKey & 1u) == 0u;
  let vertices = vertexCount(index >> 1u);
  let isActive = vertices >= 2u && (!isCycle || isCanonicalCycle);
  let isLeader = isActive && isCycle && minimumKey == ${Y(`index`)};
  var previous = NONE;
  if (isActive && !isLeader) {
    previous = partner[partnerOffset + (index ^ 1u)];
  }
  predecessor[predecessorOffset + index] = previous;
  head[headOffset + index] = index;
  sum[sumOffset + index] = select(0u, vertices - 1u, isActive);
  leaderBits[leaderBitsOffset + index] = select(0u, select(1u, 3u, isLeader), isActive);`,B)),z&&I.push(a(e,{id:`${o}-rank-count-init`,operation:b,view:q[0],type:`u32`,value:`1u`}));let ae=d(e,{id:`${o}-rank-round`,operation:b,variant:z?`rank-round-counts`:`rank-round`,length:h,rounds:R,pointers:W,values:[{name:`head`,views:G,rule:`replace`},{name:`sum`,views:K,rule:`add`},...z?[{name:`count`,views:q,rule:`add`}]:[]]});I.push(...ae.nodes);let Z=R%2,Q=G[Z],$=K[Z];if(I.push(a(e,{id:`${o}-clear-leaders`,operation:b,view:k,type:`u32`,value:`0u`}),x(`tails`,[v(`partner`,D),v(`leaderBits`,J),v(`head`,Q),v(`sum`,$),y(`leaderExit`,O),y(`leaderVertices`,k)],h,`if (leaderBits[leaderBitsOffset + index] == 0u) {
    return;
  }
  let next = partner[partnerOffset + index];
  if (next != NONE && (leaderBits[leaderBitsOffset + (next ^ 1u)] & 2u) == 0u) {
    return;
  }
  let first = head[headOffset + index];
  var isCanonical = (leaderBits[leaderBitsOffset + first] & 2u) != 0u;
  if (!isCanonical) {
    isCanonical = select((first ^ 1u) < index, (first & 1u) == 1u, DIRECTED);
  }
  if (isCanonical) {
    leaderVertices[leaderVerticesOffset + (first >> 1u)] = sum[sumOffset + index] + 1u;
    leaderExit[leaderExitOffset + (first >> 1u)] = first;
  }`),x(`flags`,[v(`leaderVertices`,k),y(`flags`,A)],l,`flags[flagsOffset + index] = select(0u, 1u, leaderVertices[leaderVerticesOffset + index] > 0u);`),...new n({id:`${o}-chain-scan`,input:A,output:j,mode:`inclusive`}).getCommandNodes(e),...new n({id:`${o}-vertex-scan`,input:k,output:M,mode:`inclusive`}).getCommandNodes(e),x(`offsets`,[v(`leaderVertices`,k),v(`flagScan`,j),v(`vertexScan`,M),y(`chainOffsets`,m.chainOffsets),y(`count`,m.status.count)],l+1,`let chainCount = flagScan[flagScanOffset + LINE_COUNT - 1u];
  let totalVertices = vertexScan[vertexScanOffset + LINE_COUNT - 1u];
  if (index == 0u) {
    count[countOffset] = chainCount;
  }
  if (index >= chainCount) {
    chainOffsets[chainOffsetsOffset + index] = totalVertices;
  }
  if (index < LINE_COUNT && leaderVertices[leaderVerticesOffset + index] > 0u) {
    let chain = flagScan[flagScanOffset + index] - 1u;
    chainOffsets[chainOffsetsOffset + chain] =
      vertexScan[vertexScanOffset + index] - leaderVertices[leaderVerticesOffset + index];
  }`),x(`positions`,[v(`positions`,s.positions,`f32`),v(`lineOffsets`,s.lineOffsets),v(`head`,Q),v(`sum`,$),v(`leaderExit`,O),v(`leaderVertices`,k),v(`vertexScan`,M),y(`outPositions`,m.positions,`f32`)],h,`let first = head[headOffset + index];
  let headLine = first >> 1u;
  if (leaderVertices[leaderVerticesOffset + headLine] == 0u ||
      leaderExit[leaderExitOffset + headLine] != first) {
    return;
  }
  let line = index >> 1u;
  let firstVertex = lineOffsets[lineOffsetsOffset + line];
  let count = vertexCount(line);
  let isForward = (index & 1u) == 1u;
  let base = vertexScan[vertexScanOffset + headLine] - leaderVertices[leaderVerticesOffset + headLine];
  let cursor = base + sum[sumOffset + index] - (count - 1u);
  for (var k = select(1u, 0u, index == first); k < count; k++) {
    let vertex = select(firstVertex + count - 1u - k, firstVertex + k, isForward);
    let slot = cursor + k;
    outPositions[outPositionsOffset + slot * 2u] = positions[positionsOffset + vertex * 2u];
    outPositions[outPositionsOffset + slot * 2u + 1u] = positions[positionsOffset + vertex * 2u + 1u];
  }`,B)),m.lineChains||m.lineOrders||m.lineReversed){let t=[v(`head`,Q),v(`leaderExit`,O),v(`leaderVertices`,k),v(`flagScan`,j)],n=`let first = head[headOffset + index];
  let headLine = first >> 1u;
  if (leaderVertices[leaderVerticesOffset + headLine] == 0u ||
      leaderExit[leaderExitOffset + headLine] != first) {
    return;
  }
  let line = index >> 1u;`;m.lineChains&&(t.push(y(`lineChains`,m.lineChains)),n+=`
  lineChains[lineChainsOffset + line] = flagScan[flagScanOffset + headLine] - 1u;`,I.push(a(e,{id:`${o}-clear-line-chains`,operation:b,view:m.lineChains,type:`u32`,value:`0xffffffffu`}))),m.lineOrders&&(t.push(v(`lineCountPrefix`,q[Z]),y(`lineOrders`,m.lineOrders)),n+=`
  lineOrders[lineOrdersOffset + line] = lineCountPrefix[lineCountPrefixOffset + index] - 1u;`),m.lineReversed&&(t.push(y(`lineReversed`,m.lineReversed)),n+=`
  lineReversed[lineReversedOffset + line] = select(1u, 0u, (index & 1u) == 1u);`),I.push(x(`line-info`,t,h,n))}return P.length>0&&I.push(x(`status-${te}`,[v(`flagScan`,j),v(`vertexScan`,M),...P],1,`let chainCount = flagScan[flagScanOffset + LINE_COUNT - 1u];
  let totalVertices = vertexScan[vertexScanOffset + LINE_COUNT - 1u];
  ${F}`)),m.sourceIds&&I.push(x(`source-ids`,[v(`leaderVertices`,k),v(`flagScan`,j),y(`sourceIds`,m.sourceIds)],l,`let chainCount = flagScan[flagScanOffset + LINE_COUNT - 1u];
  if (index >= chainCount) {
    sourceIds[sourceIdsOffset + index] = NONE;
  }
  if (leaderVertices[leaderVerticesOffset + index] > 0u) {
    sourceIds[sourceIdsOffset + flagScan[flagScanOffset + index] - 1u] = index;
  }`)),I}};function C(e){return[e.positions,e.lineOffsets]}function w(e){let{output:t}=e;return[t.chainOffsets,t.positions,t.status.count,t.status.requiredCount,t.status.overflow,t.status.vertexCount,t.status.requiredVertexCount,t.sourceIds,t.lineChains,t.lineOrders,t.lineReversed]}var T=4,E=4,D=5;function O(e){let t=Math.sin(e*78.233+1.7)*43758.5453;return t-Math.floor(t)}var k=m({parameters:y,caption:`Fragments (muted) merged into continuous chains, one color per chain.`,createInputs:({world:t})=>{let n=[],r=g(0,T,t);for(let i=0;i<T;i++){let a=e(r[i],17,t,i,T);for(let e=0;e<E;e++){let t=e*4;n.push(a.slice(t,t+D))}}let i=n.map((e,t)=>t).sort((e,t)=>O(e)-O(t)).map(e=>n[e]),a=Uint32Array.from(i.map((e,t)=>t*D).concat(i.length*D));return{fragmentCount:i.length,lineOffsets:a,getPositions:e=>Float32Array.from(i.flatMap((t,n)=>(O(n+11)<e?[...t].reverse():t).flat()))}},build(e,t,n){let r=t.getPositions(n.reversed),i=e.upload(`positions`,`float32x2`,r),a=e.upload(`line-offsets`,`uint32`,t.lineOffsets),o=e.allocate(`chain-offsets`,`uint32`,t.fragmentCount+1),s=e.allocate(`merged-positions`,`float32x2`,r.length/2),c=e.allocate(`chain-count`,`uint32`,1);e.graph.add(new S({positions:i.view,lineOffsets:a.view,directed:n.directed===1,output:{chainOffsets:o.view,positions:s.view,status:{count:c.view}}}));let l=v(e,`chains`,s,o),u=e.allocate(`original`,`float32x4`,r.length/2-t.fragmentCount),d=e.allocate(`endpoints`,`float32x2`,t.fragmentCount*2);return{outputs:{original:u,endpoints:d,merged:s,chainOffsets:o,count:c,...l},update:e=>{let n=t.getPositions(e.reversed);i.write(n),u.write(_(n,t.lineOffsets,!1)),d.write(Float32Array.from({length:t.fragmentCount*4},(e,t)=>{let r=Math.floor(t/4)*D+(t%4<2?0:4);return n[r*2+t%2]}))}}},rebuildOn:[`directed`],interaction:{kind:`probe`,hint:`hover a chain`,resolve(e,{outputs:t,peek:n}){let r=n(t.merged),i=n(t.chainOffsets),a=n(t.count)?.[0];if(!r||!i||a===void 0||a===0)return{};let o={positions:r,lineOffsets:i.subarray(0,a+1)},s=h(o,e.x,e.y);if(!s)return{};let c=i[s.line],l=i[s.line+1],u=Math.round((l-c-1)/4),d=[{shape:`polyline`,points:Array.from(r.subarray(c*2,l*2))},{shape:`ring`,at:[r[c*2],r[c*2+1]],radius:.02},{shape:`ring`,at:[r[(l-1)*2],r[(l-1)*2+1]],radius:.02}];return{focus:s.line,glyphs:d,readout:`chain ${s.line} · ${u} fragments · ${a} chains`}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.original,radius:4.5},{layer:`dots`,context:!0,positions:e.endpoints,radius:6},{layer:`segments`,nodes:e.points,edges:e.edges,values:e.lineIds,radius:2},{layer:`dots`,positions:e.points,values:e.lineIds,radius:3}]});export{k as default};
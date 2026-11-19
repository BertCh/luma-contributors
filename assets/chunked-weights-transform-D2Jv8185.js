import{i as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{l as n}from"./gpu-command-graph-BdzdmV6u.js";import{r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-reduction-tNgOVQ_C.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,x as s}from"./gpu-contributor-utils-BIwLQON5.js";import{l as c}from"./wgsl-common-DfyXiYB0.js";import{i as l,o as u}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{t as d}from"./gpu-chunked-spatial-weights-transpose-xWSecIf6.js";import{t as f}from"./gpu-spatial-weights-transform-lEdsoyYI.js";import{t as p}from"./define-run-B-brkCe4.js";import{f as m,r as h}from"./inputs-CJKKIRKK.js";import{n as g,t as _}from"./pick-Bo3t3hjv.js";import{Xn as v}from"./entry-preview-D70TdSbK.js";import{_ as y,b,d as x,h as S,i as C,n as w,p as T,u as E,v as D,x as O}from"./weights-BRH93xop.js";var k=`GPUChunkedSpatialWeightsTransform`,A=class{id;props;outputWeights;chunkTransforms;constructor(t){let r=t.id??`chunked-spatial-weights-transform`;this.id=r,this.props=t;let i=u(r,t.weights),{weights:a,output:s}=t;if(s&&s.length!==a.chunks.length)throw Error(`${r} output must hold one entry per chunk`);for(let e of s??[])e&&o(e,[`float32`],`${r} output`);if(e(r,{weights:new n({id:`${r}-writable-chunks`,name:`weights`,format:`float32`,length:a.chunks.reduce((e,t)=>e+t.weights.weights.length,0),valueLength:a.chunks.reduce((e,t)=>e+t.weights.weights.length,0),stride:1,byteStride:4,rowByteLength:4,data:a.chunks.map((e,t)=>s?.[t]??e.weights.weights)})}),this.chunkTransforms=a.chunks.map((e,n)=>new f({...t,id:`${r}-chunk${n}`,weights:e.weights,output:s?.[n],operation:t.operation===`double`||t.operation===`variance`||t.operation===`symmetrize`?`binary`:t.operation})),t.operation===`symmetrize`&&this.chunkTransforms.some(e=>e.isInPlace()))throw Error(`${r} 'symmetrize' needs a separate output per chunk`);let c=a.information,l=t.operation===`binary`||t.operation===`double`||t.operation===`symmetrize`||t.operation===`kernel`&&t.bandwidth!==`adaptive`,d={rowCount:i,transform:t.operation===`symmetrize`?`symmetric`:t.operation,symmetric:c?.symmetric===!0&&l?!0:`unknown`,hasSelfWeights:c?.hasSelfWeights,provenance:{...c?.provenance,source:k,...t.operation===`kernel`&&typeof t.bandwidth==`number`?{radius:t.bandwidth}:{}}};this.outputWeights={...a,information:d,chunks:a.chunks.map((e,t)=>({...e,weights:{...e.weights,weights:s?.[t]??e.weights.weights,information:{...d,rowCount:e.rowCount}}}))}}getCommandNodes(e){t(e,this.id,this.props,{"row scratch":(this.props.weights.rowCount+1)*4,"transpose scratch":this.props.operation===`symmetrize`?this.props.weights.chunks.reduce((e,t)=>e+t.weights.neighbors.length,0)*4:0});let{id:n,props:o}=this,{weights:u,operation:d}=o;if(s(n,e,[...l(u),...o.output??[]]),d===`row`||d===`binary`||d===`kernel`)return this.chunkTransforms.flatMap(t=>t.getCommandNodes(e));let f=u.rowCount,p=e=>this.chunkTransforms[e].isInPlace(),m=e=>{let t=u.chunks[e],n=p(e);return{source:n?`values`:`source`,bindings:[{name:`offsets`,view:t.weights.offsets,type:`u32`,access:`read`},...n?[{name:`values`,view:t.weights.weights,type:`f32`,access:`read_write`}]:[{name:`source`,view:t.weights.weights,type:`f32`,access:`read`},{name:`values`,view:o.output[e],type:`f32`,access:`read_write`}]]}};if(d===`symmetrize`)return this.getSymmetrizeNodes(e);let h=r(e,`${n}-partials`,`float32`,f),g=r(e,`${n}-total`,`float32`,1),_=d===`variance`,v=[];u.chunks.forEach((t,r)=>{let{source:i,bindings:o}=m(r),s=e=>`${i}[${i}Offset + ${e}]`;v.push(a(e,{id:`${n}-partials-chunk${r}`,operation:k,variant:`${d}-partials`,bindings:[...o,{name:`partials`,view:h,type:`f32`,access:`read_write`}],invocationCount:t.rowCount,declarations:`${c}\nconst WEIGHTS_ROW_START: u32 = ${t.rowStart}u;`,body:`let row = index + WEIGHTS_ROW_START;
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  var squares = 0.0;
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let value = ${s(`slot`)};
    sum += value;
    squares += value * value;
  }
  ${_?`let norm = sqrt(squares);
  let valid = norm > 0.0 && isFiniteFloat(norm) && isFiniteFloat(sum);
  partials[partialsOffset + row] = select(0.0, sum / norm, valid);`:`partials[partialsOffset + row] = select(0.0, sum, isFiniteFloat(sum));`}`}))}),v.push(...new i({id:`${n}-reduce`,input:h,output:g,operation:`sum`}).getCommandNodes(e));let y=_?`let totalValue = total[totalOffset];
  let scale = select(0.0, f32(ROWS) / totalValue, totalValue > 0.0 && isFiniteFloat(totalValue));`:`let totalValue = total[totalOffset];
  let scale = select(0.0, ${o.doubleSum===`rows`?`f32(ROWS)`:`1.0`} / totalValue, totalValue > 0.0 && isFiniteFloat(totalValue));`;return u.chunks.forEach((t,r)=>{let{source:i,bindings:o}=m(r),s=e=>`${i}[${i}Offset + ${e}]`;v.push(a(e,{id:`${n}-apply-chunk${r}`,operation:k,variant:`${d}-apply`,bindings:[...o,{name:`total`,view:g,type:`f32`,access:`read`}],invocationCount:t.rowCount,declarations:`${c}\nconst ROWS: u32 = ${f}u;`,body:`let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${y}
  ${_?`var squares = 0.0;
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let value = ${s(`slot`)};
    sum += value;
    squares += value * value;
  }
  let norm = sqrt(squares);
  let valid = norm > 0.0 && isFiniteFloat(norm) && isFiniteFloat(sum);
  let rowScale = select(0.0, scale / norm, valid);`:`let rowScale = scale;`}
  for (var slot = begin; slot < end; slot++) {
    let result = ${s(`slot`)} * rowScale;
    values[valuesOffset + slot] = select(0.0, result, isFiniteFloat(result) && result >= 0.0);
  }`}))}),v}getSymmetrizeNodes(e){let{id:t,props:n}=this,{weights:i}=n,o=i.rowCount,s=i.chunks.reduce((e,t)=>e+t.weights.neighbors.length,0),c={offsets:r(e,`${t}-reverse-offsets`,`uint32`,o+1),neighbors:r(e,`${t}-reverse-neighbors`,`uint32`,s),weights:r(e,`${t}-reverse-weights`,`float32`,s)},l=[...new d({id:`${t}-reverse`,weights:i,output:c}).getCommandNodes(e)];return i.chunks.forEach((r,i)=>{l.push(a(e,{id:`${t}-symmetrize-chunk${i}`,operation:k,variant:`symmetrize`,bindings:[{name:`offsets`,view:r.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.weights.neighbors,type:`u32`,access:`read`},{name:`source`,view:r.weights.weights,type:`f32`,access:`read`},{name:`values`,view:n.output[i],type:`f32`,access:`read_write`},{name:`reverseOffsets`,view:c.offsets,type:`u32`,access:`read`},{name:`reverseNeighbors`,view:c.neighbors,type:`u32`,access:`read`},{name:`reverseWeights`,view:c.weights,type:`f32`,access:`read`}],invocationCount:r.weights.neighbors.length,declarations:`const ROWS: u32 = ${o}u;
const CHUNK_ROWS: u32 = ${r.rowCount}u;
const WEIGHTS_ROW_START: u32 = ${r.rowStart}u;`,body:`if (index >= offsets[offsetsOffset + CHUNK_ROWS]) {
    return;
  }
  var localRow = 0u;
  var high = CHUNK_ROWS;
  while (localRow + 1u < high) {
    let middle = (localRow + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      localRow = middle;
    } else {
      high = middle;
    }
  }
  let row = localRow + WEIGHTS_ROW_START;
  let neighbor = neighbors[neighborsOffset + index];
  var reverse = 0.0;
  if (neighbor < ROWS) {
    // w_ji is entry j of row i of the transpose.
    var low = reverseOffsets[reverseOffsetsOffset + row];
    var upper = reverseOffsets[reverseOffsetsOffset + row + 1u];
    while (low < upper) {
      let middle = (low + upper) / 2u;
      let candidate = reverseNeighbors[reverseNeighborsOffset + middle];
      if (candidate == neighbor) {
        reverse = reverseWeights[reverseWeightsOffset + middle];
        break;
      }
      if (candidate < neighbor) {
        low = middle + 1u;
      } else {
        upper = middle;
      }
    }
  }
  values[valuesOffset + index] = 0.5 * (source[sourceOffset + index] + reverse);`}))}),l}},j=.1,M=[`row`,`binary`,`kernel`,`double`,`variance`],N=p({parameters:v,caption:`Row sums of the transformed weights, computed chunk by chunk.`,createInputs:({random:e,world:t})=>{let n=D(m(e,t,260,.1),t),r=y(n.positions,n.values);return{...r,index:g(r.positions),focus:{row:-1}}},build(e,t,n){let r=t.values.length,i=x(r,n.chunkCount),a=e.upload(`positions`,`float32x2`,t.positions),o=O(e,`band`,h(t.positions,j),i),s=o.chunks.map((t,n)=>e.allocate(`transformed-${n}`,`float32`,t.capacity));return e.graph.add(new A({operation:M[n.operation],weights:o.input,output:s.map(e=>e.view),kernel:`triangular`,bandwidth:n.bandwidth})),{outputs:{chunkStarts:o.chunks.map(e=>e.rowStart),boundaries:b(e,t.positions,o.chunks.map(e=>e.rowStart)),chunks:o.chunks.map((n,r)=>({positions:e.upload(`chunk-positions-${r}`,`float32x2`,t.positions.slice(n.rowStart*2,(n.rowStart+n.rowCount)*2)),rowSums:C(e,`transformed-${r}`,n,s[r]),links:w(e,`band-${r}`,n,a,{rowStart:n.rowStart})}))}}},rebuildOn:[`operation`,`bandwidth`,`chunkCount`],interaction:{kind:`probe`,hint:`hover a point`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=_(t.index,e.x,e.y,t.focus.row);if(t.focus.row=a,a<0)return{};let o=[t.positions[a*2],t.positions[a*2+1]],s=n.chunks.flatMap(e=>S(i(e.links),o)),c=E(n.chunkStarts,a),l=n.chunks[c-1],u=i(l.rowSums),d=u?u[a-n.chunkStarts[c-1]]:void 0;return{focus:a,glyphs:[{shape:`ring`,at:o,radius:r.bandwidth,dashed:!0},...T(o,s)],readout:`row ${a} (chunk ${c}) · ${s.length} links · sum ${d?.toFixed(2)??`-`}`}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.boundaries,count:e.chunkStarts.length-1},...e.chunks.map(e=>({layer:`lines`,segments:e.links,color:[.6,.65,.85,.3],radius:1.1})),...e.chunks.map(e=>({layer:`dots`,positions:e.positions,values:e.rowSums,radius:3.6,range:[0,6]}))]});export{N as default};
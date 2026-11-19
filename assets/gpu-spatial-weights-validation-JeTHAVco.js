import{t as e}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-reduction-tNgOVQ_C.js";import{t as r}from"./spatial-statistics-row-dispatch-DNaskhxv.js";import{i}from"./graph-chunk-dispatch-zBt8k2-o.js";import{E as a,x as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{l as c}from"./wgsl-common-DfyXiYB0.js";import{t as l}from"./spatial-weights-CqWysstN.js";import{a as u,i as d,s as f,t as p}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";var m=`GPUSpatialWeightsValidation`,h=`GPUChunkedSpatialWeightsValidation`,g={offsets:1,neighborRange:2,order:4,self:8,weight:16},_=class{id;props;outputWeights;constructor(e){let t=e.id??(u(e.weights)?`chunked-spatial-weights-validation`:`spatial-weights-validation`);this.id=t,this.props=e;let n=f(t,e.weights);if(a(e.output.status.invalidCount,[`uint32`],`${t} invalidCount`),e.output.status.invalidCount.length<1)throw Error(`${t} invalidCount must hold one uint32`);if(e.output.flags&&(a(e.output.flags,[`uint32`],`${t} flags`),e.output.flags.length!==n))throw Error(`${t} flags length must equal the weights row count`);i(t,{flags:e.output.flags,invalidCount:e.output.status.invalidCount}),s(t,[e.output.status.invalidCount,e.output.flags],d(e.weights));let r=e.weights.information??{rowCount:n,transform:`unknown`,symmetric:`unknown`};this.outputWeights=u(e.weights)?{...e.weights,information:r,status:{...e.weights.status,invalidCount:e.output.status.invalidCount}}:l(e.weights,{information:r,status:{invalidCount:e.output.status.invalidCount}})}getCommandNodes(i){e(i,this.id,this.props,{"row scratch":p(this.props.weights).reduce((e,t)=>e+t.rowCount,0)*4});let{id:a,props:s}=this,{weights:l}=s;o(a,i,[...d(l),s.output.status.invalidCount,s.output.flags]);let f=p(l),_=f.reduce((e,t)=>e+t.rowCount,0),v=t(i,`${a}-violated`,`uint32`,_),y=s.output.flags??t(i,`${a}-flags`,`uint32`,_),b=g,x=u(l)?h:m,S=f.map(e=>r(i,{rowBase:e.rowStart,id:`${a}-rows${e.isWhole?``:e.idSuffix}`,operation:x,variant:`rows`,bindings:[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:e.weights.weights,type:`f32`,access:`read`},{name:`flags`,view:y,type:`u32`,access:`read_write`},{name:`violated`,view:v,type:`u32`,access:`read_write`}],invocationCount:e.rowCount,declarations:`${c}
const ROWS: u32 = ${_}u;
`,body:`let row = index; let localRow = index - WEIGHTS_ROW_START;
  let begin = offsets[offsetsOffset + localRow];
  let end = offsets[offsetsOffset + localRow + 1u];
  var rowFlags = 0u;
  if ((localRow == 0u && begin != 0u) || begin > end || (localRow + 2u == arrayLength(&offsets) - offsetsOffset && end > CAPACITY)) {
    rowFlags |= ${b.offsets}u;
  }
  var previous = 0u;
  for (var slot = begin; slot < min(end, CAPACITY); slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = weights[weightsOffset + slot];
    if (neighbor >= ROWS) {
      rowFlags |= ${b.neighborRange}u;
    }
    if (neighbor == row) {
      rowFlags |= ${b.self}u;
    }
    if (slot > begin && neighbor <= previous) {
      rowFlags |= ${b.order}u;
    }
    if (!isFiniteFloat(weight) || weight < 0.0) {
      rowFlags |= ${b.weight}u;
    }
    previous = neighbor;
  }
  flags[flagsOffset + row] = rowFlags;
  violated[violatedOffset + row] = select(0u, 1u, rowFlags != 0u);`}));return S.push(...new n({id:`${a}-count`,input:v,output:s.output.status.invalidCount,operation:`sum`}).getCommandNodes(i)),S}};export{_ as t};
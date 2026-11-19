import{d as e,f as t,p as n,r,u as i}from"./gpu-command-graph-BdzdmV6u.js";import{a,c as o,o as s,s as c,t as l,u}from"./kernel-C3tfQOn4.js";import{n as d,t as f}from"./gpu-dispatch-utils-CbRzFAgT.js";import{n as p,r as m,t as h}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";var g=256,_=Uint32Array.BYTES_PER_ELEMENT,v=class{id;source;indices;output;wordsPerRow;constructor(t){this.id=t.id??`gpu-gather`,this.source=t.source,this.indices=t.indices,this.output=t.output;for(let e of m(this.indices))u(e,`${this.id} indices`);for(let e of[this.source,this.output])for(let t of m(e))b(t,this.id,e===this.output);if(this.source.format!==this.output.format)throw Error(`${this.id} source and output must use the same format`);if(this.output.length<this.indices.length)throw Error(`${this.id} output must contain at least indices.length rows`);let n=new Set([...m(this.source).map(e=>e.buffer),...m(this.indices).map(e=>e.buffer)]);if(m(this.output).some(e=>n.has(e.buffer)))throw Error(`${this.id} output must use a separate buffer`);this.wordsPerRow=e(this.source.format).byteLength/_}getCommandNodes(e){return y(e,this)}};function y(e,t,n=0,o=`GPUGather`){let u=[];for(let n of[t.source,t.indices,t.output])for(let r of m(n))if(r.buffer.graph!==e)throw Error(`${t.id} views must belong to the target graph`);if(t.indices.length===0)return u;let d=a(e,t.output,t.indices.length),v=p(e,[t.indices,d]),y=i(m(t.source)).filter(e=>e.length),b=y.length?y:[void 0];for(let[i,[a,d]]of v.entries()){let p=f(o,a.length,g,e.device.limits.maxComputeWorkgroupsPerDimension);for(let[f,m]of b.entries()){let y=m?.data;for(let t of[d,...y?[y,a]:[]])if(c(t).size>e.device.limits.maxStorageBufferBindingSize)throw Error(`Active chunks must fit a storage binding`);let S=v.length===1&&b.length===1?t.id:`${t.id}-span-${i}-source-${f}`,C=x({source:y,sourceOffset:m?.offset??0,indices:a,output:d,wordsPerRow:t.wordsPerRow,initialize:f===0,invalidValue:n,dispatchLayout:p});u.push(h(r({id:S,workload:{operation:o,commandCount:1,maximumWorkgroupCount:p.x*p.y*p.z,maximumInvocationCount:p.x*p.y*p.z*g,readByteLength:y?a.length*(_+y.rowByteLength):0,writeByteLength:a.length*d.rowByteLength},resources:[...y?[{buffer:y,usage:`storage-read`},{buffer:a,usage:`storage-read`}]:[],{buffer:d,usage:`storage-write`}],compile:({device:e})=>{let t=new l(e,{id:S,source:C,shaderLayout:{bindings:[{name:`outputWords`,type:`storage`,group:0,location:0},...y?[{name:`sourceWords`,type:`read-only-storage`,group:0,location:1},{name:`indices`,type:`read-only-storage`,group:0,location:2}]:[]]}});return{encode:({computePass:e,getBuffer:n})=>{let r={outputWords:s(d,n)};y&&(r.sourceWords=s(y,n),r.indices=s(a,n)),t.dispatch(e,{bindings:r,x:p.x,y:p.y,z:p.z})},destroy:()=>t.destroy()}}}),[p.x,p.y,p.z]))}}return u}function b(r,i,a){if(n(r.format)||t(r.format))throw Error(`${i} must use a fixed-width GPU data format`);let o=e(r.format);if((a?r.byteStride!==o.byteLength:r.byteStride<o.byteLength||r.byteStride%_!==0)||r.rowByteLength!==o.byteLength||r.byteOffset%_!==0||r.rowByteLength%_!==0)throw Error(`${i} must be packed, uint32-aligned GPU data with 32-bit-word rows`)}function x(e){let{source:t,indices:n,output:r,wordsPerRow:i,initialize:a,invalidValue:s,dispatchLayout:c}=e;return`const INDEX_COUNT: u32 = ${n.length}u;
const WORDS_PER_ROW: u32 = ${i}u;
const OUTPUT_OFFSET: u32 = ${o(r)}u;
@group(0) @binding(0) var<storage, read_write> outputWords: array<u32>;
${t?`
const SOURCE_LENGTH: u32 = ${t.length}u;
const SOURCE_ROW_OFFSET: u32 = ${e.sourceOffset}u;
const SOURCE_OFFSET: u32 = ${o(t)}u;
const INDEX_OFFSET: u32 = ${o(n)}u;
@group(0) @binding(1) var<storage, read> sourceWords: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
`:``}
@compute @workgroup_size(${g})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${d(c,g)}
  if (index >= INDEX_COUNT) { return; }
  let outputBase = OUTPUT_OFFSET + index * WORDS_PER_ROW;
${t?`
  let sourceIndex = indices[INDEX_OFFSET + index];
  if (sourceIndex >= SOURCE_ROW_OFFSET && sourceIndex - SOURCE_ROW_OFFSET < SOURCE_LENGTH) {
    let sourceBase = SOURCE_OFFSET + (sourceIndex - SOURCE_ROW_OFFSET) * ${t.byteStride/_}u;
    for (var word = 0u; word < WORDS_PER_ROW; word++) {
      outputWords[outputBase + word] = sourceWords[sourceBase + word];
    }
    return;
  }
`:``}
${a?`
  for (var word = 0u; word < WORDS_PER_ROW; word++) {
    outputWords[outputBase + word] = ${s}u;
  }
`:``}
}`}export{y as n,v as t};
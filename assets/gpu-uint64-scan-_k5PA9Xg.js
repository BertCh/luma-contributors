import{l as e,r as t}from"./gpu-command-graph-BdzdmV6u.js";import{a as n,c as r,i,n as a,o,r as s,t as c,u as l}from"./kernel-C3tfQOn4.js";import{n as u,t as d}from"./gpu-dispatch-utils-CbRzFAgT.js";import{t as f}from"./gpu-scan-B4NNTfF9.js";import{n as p,r as m}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";var h=256,g=class{id;props;constructor(e){this.id=e.id??`gpu-scan-uint64`,this.props=Object.freeze({...e,id:this.id}),v(this.props)}getCommandNodes(t){let r=[],i=this.props;for(let e of[i.inputLow,i.inputHigh,i.outputLow,i.outputHigh])if(m(e).some(e=>e.buffer.graph!==t))throw Error(`${this.id} views must belong to the target graph`);if(i.inputLow.length===0)return r;let o=n(t,i.outputLow,i.inputLow.length),c=n(t,i.outputHigh,i.inputLow.length);r.push(...new f({id:`${this.id}-low`,input:i.inputLow,output:o,mode:`inclusive`}).getCommandNodes(t));let l=i.inputHigh instanceof e?a(t,`${this.id}-adjusted-high`,i.inputHigh):s(t,`${this.id}-adjusted-high`,`uint32`,i.inputHigh.length),u=p(t,[i.inputHigh,o,l]),d;for(let[e,[n,i,a]]of u.entries())r.push(..._(t,`${this.id}-carry-${e}`,n,i,a,d)),d=t.createDataView(i.buffer,{format:`uint32`,length:1,byteOffset:i.byteOffset+(i.length-1)*4});return r.push(...new f({id:`${this.id}-high`,input:l,output:c,mode:`inclusive`}).getCommandNodes(t)),r}};function _(e,n,i,a,s,l){let f=[],p=i.length,m=d(`GPUScanUint64`,p,h,e.device.limits.maxComputeWorkgroupsPerDimension),g=`const LENGTH: u32 = ${p}u;
const INPUT_HIGH_OFFSET: u32 = ${r(i)}u;
const PREFIX_LOW_OFFSET: u32 = ${r(a)}u;
const OUTPUT_OFFSET: u32 = ${r(s)}u;
@group(0) @binding(0) var<storage, read> inputHigh: array<u32>;
@group(0) @binding(1) var<storage, read> prefixLow: array<u32>;
@group(0) @binding(2) var<storage, read_write> adjustedHigh: array<u32>;
${l?`@group(0) @binding(3) var<storage, read> previousLow: array<u32>;`:``}
@compute @workgroup_size(${h})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${u(m,h)}
  if (index >= LENGTH) {
    return;
  }
  var preceding = ${l?`previousLow[${r(l)}u]`:`0u`};
  if (index > 0u) {
    preceding = prefixLow[PREFIX_LOW_OFFSET + index - 1u];
  }
  let carry = select(0u, 1u, prefixLow[PREFIX_LOW_OFFSET + index] < preceding);
  adjustedHigh[OUTPUT_OFFSET + index] = inputHigh[INPUT_HIGH_OFFSET + index] + carry;
}`;return f.push(t({id:n,workload:{operation:`GPUScanUint64`,commandCount:1,maximumWorkgroupCount:Math.ceil(p/h),maximumInvocationCount:Math.ceil(p/h)*h,readByteLength:(p*2+ +!!l)*Uint32Array.BYTES_PER_ELEMENT,writeByteLength:p*Uint32Array.BYTES_PER_ELEMENT},resources:[{buffer:i,usage:`storage-read`},{buffer:a,usage:`storage-read`},{buffer:s,usage:`storage-write`},...l?[{buffer:l,usage:`storage-read`}]:[]],compile:({device:e})=>{let t=new c(e,{id:n,source:g,shaderLayout:{bindings:[{name:`inputHigh`,type:`read-only-storage`,group:0,location:0},{name:`prefixLow`,type:`read-only-storage`,group:0,location:1},{name:`adjustedHigh`,type:`storage`,group:0,location:2},...l?[{name:`previousLow`,type:`read-only-storage`,group:0,location:3}]:[]]}});return{encode:({computePass:e,getBuffer:n})=>{let r={inputHigh:o(i,n),prefixLow:o(a,n),adjustedHigh:o(s,n)};l&&(r.previousLow=o(l,n)),t.dispatch(e,{bindings:r,x:m.x,y:m.y,z:m.z})},destroy:()=>t.destroy()}}})),f}function v(e){for(let[t,n]of Object.entries({inputLow:e.inputLow,inputHigh:e.inputHigh,outputLow:e.outputLow,outputHigh:e.outputHigh}))for(let r of m(n))l(r,`${e.id} ${t}`);if(e.inputHigh.length!==e.inputLow.length||e.outputLow.length<e.inputLow.length||e.outputHigh.length<e.inputLow.length)throw Error(`${e.id} inputs must match and outputs must cover every input row`);let t=m(e.inputLow),n=m(e.inputHigh),r=m(e.outputLow),a=m(e.outputHigh);if(r.some(e=>[...t,...n,...a].some(t=>t.buffer===e.buffer))||a.some(e=>t.some(t=>t.buffer===e.buffer)))throw Error(`${e.id} split-word outputs must use safe separate buffers`);for(let t of[r,a])for(let[n,r]of t.entries())if(t.slice(0,n).some(e=>i(e,r)))throw Error(`${e.id} output chunks must not overlap`)}export{g as t};
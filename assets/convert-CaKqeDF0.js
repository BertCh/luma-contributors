import{c as e,o as t,t as n}from"./kernel-C3tfQOn4.js";import{r}from"./gpu-command-graph-BdzdmV6u.js";import{n as i,t as a}from"./gpu-dispatch-utils-CbRzFAgT.js";var o=64,s={1:`float32`,2:`float32x2`,3:`float32x3`,4:`float32x4`},c=class{id;source;target;layout;constructor(e,t,n,r){this.id=e,this.source=t,this.target=n,this.layout=r}getCommandNodes(s){let{id:c,source:l,target:u}=this,{outputComponents:d=1,componentStart:f=0}=this.layout,p=u.length*d,m=l.rowByteLength/4,h=a(c,p,o,s.device.limits.maxComputeWorkgroupsPerDimension),g=l.format.startsWith(`sint`)?`i32`:l.format.startsWith(`uint`)?`u32`:`f32`,{stride:_=m,offset:v=0,flipColumns:y}=this.layout,b=y?Math.floor(u.length/y):0,x=y?`(${b-1}u - row / ${y}u) * ${y}u + row % ${y}u`:`row`,S=`
const inputOffset: u32 = ${e(l)}u;
const outputOffset: u32 = ${e(u)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<${g}>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;
@compute @workgroup_size(${o})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${i(h,o)}
  if (index >= ${p}u) { return; }
  let row = index / ${d}u;
  let component = index % ${d}u;
  outputValues[outputOffset + index] = f32(inputValues[inputOffset + ${v}u + (${x}) * ${_}u + ${f}u + component]);
}`;return[r({id:c,resources:[{buffer:l,usage:`storage-read`},{buffer:u,usage:`storage-read-write`}],workload:{operation:`IntegerToFloat`,variant:l.format,commandCount:1,maximumWorkgroupCount:h.x*h.y*h.z,maximumInvocationCount:h.x*h.y*h.z*o,readByteLength:u.length*4,writeByteLength:u.length*4},compile:({device:e})=>{let r=new n(e,{id:c,source:S,shaderLayout:{bindings:[{name:`inputValues`,type:`storage`,group:0,location:0},{name:`outputValues`,type:`storage`,group:0,location:1}]}});return{encode:({computePass:e,getBuffer:n})=>{let i={inputValues:t(l,n),outputValues:t(u,n)};r.dispatch(e,{bindings:i,...h})},destroy:()=>r.destroy()}}})]}};function l(e,t,n){return u(e,t,n,n.length)}function u(e,t,n,r,i={}){let a=e.allocate(t,s[i.outputComponents??1],r);return e.graph.add(new c(`${t}-copy`,n.view,a.view,i)),a}export{l as n,u as t};
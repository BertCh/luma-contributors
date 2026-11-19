import{c as e,o as t,r as n,t as r}from"./kernel-C3tfQOn4.js";import{f as i,h as a,i as o,n as s,o as c,r as l,s as u,u as d}from"./raster-utils-CztuVGLY.js";import{i as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{t as m}from"./construction-snapshot-DdkX9Pc0.js";import{a as h}from"./raster-band-utils-BcTxc8_Y.js";var g=class{id;width;height;left;right;operation;output;outputValidity;epsilon;clamp;constructor(e){if(this.id=e.id??`gpu-raster-band-math`,this.width=e.width,this.height=e.height,this.left=e.left,this.right=e.right,this.operation=e.operation,this.output=e.output,this.outputValidity=e.outputValidity,this.epsilon=e.epsilon??0,this.clamp=e.clamp,!Number.isSafeInteger(this.width)||this.width<=0||!Number.isSafeInteger(this.height)||this.height<=0)throw Error(`${this.id} dimensions must be positive integers`);let t=this.width*this.height;if(!Number.isSafeInteger(t)||t>4294967295)throw Error(`${this.id} pixel count must fit in uint32`);if(!_(this.operation))throw Error(`${this.id} operation is not supported`);if(!Number.isFinite(this.epsilon)||this.epsilon<0)throw Error(`${this.id} epsilon must be finite and non-negative`);if(o(this.epsilon),this.clamp){if(this.clamp.length!==2||!Number.isFinite(this.clamp[0])||!Number.isFinite(this.clamp[1])||this.clamp[0]>this.clamp[1])throw Error(`${this.id} clamp must contain an ordered finite range`);o(this.clamp[0]),o(this.clamp[1])}if(this.left.storage.kind!==`buffer`||this.right.storage.kind!==`buffer`)throw Error(`${this.id} requires buffer-backed input bands`);let n=d(this.left,this,`${this.id} left`);if(d(this.right,this,`${this.id} right`)!==n||(i(this.output,`float32`,t,`${this.id} output`),a(this.outputValidity,t,`${this.id} output validity`),this.output.buffer.graph!==n||this.outputValidity.buffer.graph!==n))throw Error(`${this.id} resources must belong to the same graph`);let r=[this.left.storage.values.buffer,this.right.storage.values.buffer,...this.left.validity?[this.left.validity.buffer]:[],...this.right.validity?[this.right.validity.buffer]:[]];if(this.output.buffer===this.outputValidity.buffer||r.includes(this.output.buffer)||r.includes(this.outputValidity.buffer))throw Error(`${this.id} inputs and outputs must use separate buffers`);for(let e of[this.left,this.right])o(e.scale??1),o(e.offset??0)}addToGraph(e){let n=[this.left.storage.values,this.right.storage.values,...this.left.validity?[this.left.validity]:[],...this.right.validity?[this.right.validity]:[]];for(let t of[...n,this.output,this.outputValidity]){if(t.buffer.graph!==e)throw Error(`${this.id} resources must belong to the target graph`);s(e.device,t,`${this.id} ${t.buffer.id}`)}let[i,a]=l(e.device,this.width,this.height,this.id),o=[{buffer:this.left.storage.values,usage:`storage-read`},{buffer:this.right.storage.values,usage:`storage-read`},{buffer:this.output,usage:`storage-write`},{buffer:this.outputValidity,usage:`storage-write`}];this.left.validity&&o.push({buffer:this.left.validity,usage:`storage-read`}),this.right.validity&&o.push({buffer:this.right.validity,usage:`storage-read`}),e.addComputePass({id:this.id,resources:o,compile:({device:e})=>{let n=[{name:`leftValues`,type:`read-only-storage`,group:0,location:0},{name:`rightValues`,type:`read-only-storage`,group:0,location:1},{name:`outputValues`,type:`storage`,group:0,location:2},{name:`outputValidity`,type:`storage`,group:0,location:3}];this.left.validity&&n.push({name:`leftValidity`,type:`read-only-storage`,group:0,location:4}),this.right.validity&&n.push({name:`rightValidity`,type:`read-only-storage`,group:0,location:this.left.validity?5:4});let o=new r(e,{id:this.id,source:this.getShaderSource(),shaderLayout:{bindings:n}});return{encode:({computePass:e,getBuffer:n})=>{let r={leftValues:t(this.left.storage.values,n),rightValues:t(this.right.storage.values,n),outputValues:t(this.output,n),outputValidity:t(this.outputValidity,n)};this.left.validity&&(r.leftValidity=t(this.left.validity,n)),this.right.validity&&(r.rightValidity=t(this.right.validity,n));let s=r;o.dispatch(e,{bindings:s,x:i,y:a})},destroy:()=>o.destroy()}}})}getShaderSource(){let t=this.left.validity?`@group(0) @binding(4) var<storage, read> leftValidity: array<u32>;\nconst LEFT_VALIDITY_OFFSET: u32 = ${e(this.left.validity)}u;`:``,n=this.right.validity?`@group(0) @binding(${this.left.validity?5:4}) var<storage, read> rightValidity: array<u32>;\nconst RIGHT_VALIDITY_OFFSET: u32 = ${e(this.right.validity)}u;`:``,r=[`isFiniteValue(leftSample)`,`isFiniteValue(rightSample)`,...v(this.left,`left`),...v(this.right,`right`)],i=this.operation===`divide`?`\n  let denominator = rightSample;\n  if (!isFiniteValue(denominator) || abs(denominator) <= ${o(this.epsilon)}) {\n    validSample = false;\n  }`:this.operation===`normalized-difference`?`
  let leftDominant = abs(leftSample) >= abs(rightSample);
  let dominantSample = select(rightSample, leftSample, leftDominant);
  let secondarySample = select(leftSample, rightSample, leftDominant);
  let safeDominantSample = select(1.0, dominantSample, dominantSample != 0.0);
  let sampleRatio = secondarySample / safeDominantSample;
  let scaledDenominator = 1.0 + sampleRatio;
  let denominator = leftSample + rightSample;
  if (!isFiniteValue(denominator) || abs(denominator) <= ${o(this.epsilon)}) {
    validSample = false;
  }`:``,a=y(this.operation),s=this.clamp?`clamp(result, ${o(this.clamp[0])}, ${o(this.clamp[1])})`:`result`;return`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const LEFT_OFFSET: u32 = ${e(this.left.storage.values)}u;
const RIGHT_OFFSET: u32 = ${e(this.right.storage.values)}u;
const OUTPUT_OFFSET: u32 = ${e(this.output)}u;
const OUTPUT_VALIDITY_OFFSET: u32 = ${e(this.outputValidity)}u;
@group(0) @binding(0) var<storage, read> leftValues: array<${u(this.left.format)}>;
@group(0) @binding(1) var<storage, read> rightValues: array<${u(this.right.format)}>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputValidity: array<u32>;
${t}
${n}

fn isFiniteValue(value: f32) -> bool {
  return value == value && abs(value) <= bitcast<f32>(0x7f7fffffu);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let leftRawSample = leftValues[LEFT_OFFSET + pixelIndex];
  let rightRawSample = rightValues[RIGHT_OFFSET + pixelIndex];
  let leftSample = f32(leftRawSample) * ${o(this.left.scale??1)} + ${o(this.left.offset??0)};
  let rightSample = f32(rightRawSample) * ${o(this.right.scale??1)} + ${o(this.right.offset??0)};
  var validSample = ${r.join(` && `)};${i}
  var outputSample = bitcast<f32>(0x7fc00000u | (pixelIndex & 0u));
  if (validSample) {
    let result = ${a};
    if (isFiniteValue(result)) {
      outputSample = ${s};
    } else {
      validSample = false;
    }
  }
  outputValues[OUTPUT_OFFSET + pixelIndex] = outputSample;
  outputValidity[OUTPUT_VALIDITY_OFFSET + pixelIndex] = select(0u, 1u, validSample);
}`}};function _(e){return e===`add`||e===`subtract`||e===`multiply`||e===`divide`||e===`normalized-difference`}function v(e,t){let n=[];if(e.validity){let e=t===`left`?`LEFT_VALIDITY_OFFSET`:`RIGHT_VALIDITY_OFFSET`;n.push(`${t}Validity[${e} + pixelIndex] != 0u`)}return e.format===`float32`&&n.push(`isFiniteValue(${t}RawSample)`),e.noDataValue!==void 0&&!Number.isNaN(e.noDataValue)&&n.push(`${t}RawSample != ${c(e.noDataValue,e.format)}`),n}function y(e){switch(e){case`add`:return`leftSample + rightSample`;case`subtract`:return`leftSample - rightSample`;case`multiply`:return`leftSample * rightSample`;case`divide`:return`leftSample / denominator`;case`normalized-difference`:return`select((sampleRatio - 1.0) / scaledDenominator, (1.0 - sampleRatio) / scaledDenominator, leftDominant)`}}function b(e,t){let n=m(e,[`id`,`width`,`height`,`output`,`outputValidity`,`epsilon`,`clamp`,...t]),r=t.filter(e=>![`method`,`soilAdjustment`].includes(String(e))).map(e=>[e,Object.freeze(h(n[e]))]);return Object.freeze({...n,...Object.fromEntries(r),...n.clamp?{clamp:Object.freeze([...n.clamp])}:{}})}var x=class{id;props;commandNodes=new WeakMap;getCommandNodes(e){let t=this.commandNodes.get(e);if(t)return t;let n=Object.freeze(f(e,()=>this.addIndexNodes(e)));return this.commandNodes.set(e,n),n}constructor(e,t,n){this.id=e.id??t,this.props=e;for(let t of n)new g({...e,id:this.id,left:t,right:t,operation:`add`}),p(this.id,{output:e.output,outputValidity:e.outputValidity},{values:t.storage.values,validity:t.validity});p(this.id,{output:e.output},{outputValidity:e.outputValidity})}addStage(e,t,r,i,a){let o=`${this.id}-${t}`,s=this.props.width*this.props.height,c=n(e,`${o}-values`,`float32`,s),l=n(e,`${o}-validity`,`uint32`,s);return new g({id:o,width:this.props.width,height:this.props.height,left:r,right:i,operation:a,output:c,outputValidity:l}).addToGraph(e),{id:o,format:`float32`,storage:{kind:`buffer`,values:c},validity:l}}addOutput(e,t,n,r){new g({...this.props,id:this.id,left:t,right:n,operation:r}).addToGraph(e)}};export{b as n,x as t};
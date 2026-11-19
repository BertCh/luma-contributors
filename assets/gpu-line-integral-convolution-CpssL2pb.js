import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{d as t,r as n,u as r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-raster-buffer-to-texture-BDZQU4A5.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{C as o,i as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{t as u}from"./vector-field-sampling-KzXxZj8z.js";import{n as d,t as f}from"./philox-random-DfVxWNPn.js";import{u as p}from"./raster-band-utils-BcTxc8_Y.js";var m=f.lineIntegralConvolutionNoise.id;function h(e,t=new Float32Array(12)){if(t.length<12)throw Error(`Line integral convolution parameter target must hold 12 elements`);for(let t of[`fieldExtent`,`outputExtent`])o(`GPULineIntegralConvolution`,`${t} cellWidth`,e[t][2]),o(`GPULineIntegralConvolution`,`${t} cellHeight`,e[t][3]);return t.set([...e.fieldExtent,...e.outputExtent,e.stepLength??.5,e.minimumSpeed??0,e.phase??0,e.period??0]),t}function g(e,t=new Uint32Array(4)){if(t.length<4)throw Error(`Line integral convolution uint32 parameter target must hold 4 elements`);let n=e.seed??0;if(!Number.isInteger(n)||n<0||n>4294967295)throw Error(`Line integral convolution seed must be a uint32 integer`);return t.set([n,0,0,0]),t}var _=`GPULineIntegralConvolution`,v=256,y=class{id;stepCount;props;constructor(e){let n=e.id??`line-integral-convolution`;for(let[t,r]of[[`fieldWidth`,e.fieldWidth],[`fieldHeight`,e.fieldHeight],[`grid width`,e.grid.width],[`grid height`,e.grid.height]])if(!Number.isInteger(r)||r<1||r>65535)throw Error(`${n} ${t} must be an integer in [1, 65535]`);let i=e.stepCount??20;if(!Number.isInteger(i)||i<0||i>v)throw Error(`${n} stepCount must be an integer in [0, ${v}]`);if(t(e.velocities,[`float32x2`],`${n} velocities`),e.velocities.length!==e.fieldWidth*e.fieldHeight)throw Error(`${n} velocities must have fieldWidth * fieldHeight rows`);if(t(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<12)throw Error(`${n} parameters must contain 12 float32 rows`);if(r(e.uint32Parameters,`${n} uint32Parameters`),e.uint32Parameters.length<4)throw Error(`${n} uint32Parameters must contain 4 uint32 rows`);let a=p(n,e.grid),{output:o}=e;if(t(o.values,[`float32`],`${n} output.values`),o.values.length!==a)throw Error(`${n} output.values must have grid width * height rows`);if(o.speeds&&(t(o.speeds,[`float32`],`${n} output.speeds`),o.speeds.length!==a))throw Error(`${n} output.speeds must have grid width * height rows`);if(o.texture&&(o.texture.format!==`r32float`||o.texture.width!==e.grid.width||o.texture.height!==e.grid.height))throw Error(`${n} output.texture must be r32float with width x height`);l(n,[o.values,o.speeds],[e.velocities,e.parameters,e.uint32Parameters]),this.id=n,this.stepCount=i,this.props=e}getCommandNodes(t){let r=this.props,o=r.grid.width*r.grid.height;e(t,this.id,[r.velocities,r.parameters,r.uint32Parameters,r.output.values,r.output.speeds],{"noise plane":o*4});let{id:l,props:f,stepCount:p}=this,{output:h}=f,{width:g,height:v}=f.grid;if(c(l,t,[f.velocities,f.parameters,f.uint32Parameters,h.values,h.speeds]),h.texture&&h.texture.texture.graph!==t)throw Error(`${l} views must belong to the target graph`);let y=g*v,b=n(t,`${l}-noise`,`float32`,y),x=[a(t,{id:`${l}-noise`,operation:_,variant:`noise`,bindings:[{name:`words`,view:f.uint32Parameters,type:`u32`,access:`read`},{name:`noise`,view:b,type:`f32`,access:`read_write`}],invocationCount:y,declarations:`${d}
const WIDTH: u32 = ${g}u;`,body:`let seed = words[wordsOffset];
  let counter = vec4<u32>(index % WIDTH, index / WIDTH, 0u, 0u);
  noise[noiseOffset + index] = philoxUnitFloat(
    philox4x32(counter, vec2<u32>(seed, ${m}u)).x
  );`})],S=[{name:`velocities`,view:f.velocities,type:`f32`,access:`read`},{name:`parameters`,view:f.parameters,type:`f32`,access:`read`},{name:`noise`,view:b,type:`f32`,access:`read`},{name:`values`,view:h.values,type:`f32`,access:`read_write`}];if(h.speeds&&S.push({name:`speeds`,view:h.speeds,type:`f32`,access:`read_write`}),x.push(a(t,{id:`${l}-convolve`,operation:_,variant:`convolve`,bindings:S,invocationCount:y,declarations:`${u(`velocities`,f.fieldWidth,f.fieldHeight)}
const WIDTH: u32 = ${g}u;
const HEIGHT: u32 = ${v}u;
const STEP_COUNT: u32 = ${p}u;
const PI: f32 = 3.14159265358979;
fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }
fn getWeight(signedStep: f32, phase: f32, period: f32) -> f32 {
  let window = 0.5 * (1.0 + cos(PI * signedStep / f32(STEP_COUNT + 1u)));
  var ripple = 1.0;
  if (period > 0.0) {
    ripple = 0.5 * (1.0 + cos(2.0 * PI * (signedStep / period - phase)));
  }
  return window * ripple;
}
// Unit flow direction times sign, or (0, 0, 0) where the trace must stop.
fn getDirection(position: vec2<f32>, fieldExtent: vec4<f32>, directionSign: f32, minimumSpeed: f32) -> vec3<f32> {
  let sample = sampleField(position, fieldExtent);
  if (sample.z == 0.0) {
    return vec3<f32>(0.0);
  }
  let speed = length(sample.xy);
  if (!(speed > 0.0) || speed < minimumSpeed) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(directionSign * sample.xy / speed, 1.0);
}
// World displacement of one step along the unit world direction: one positive scalar times the
// direction, chosen so the step is stepLength output pixels long. Scaling each axis by its own
// cell size instead would bend the trace away from the field on non-square cells.
fn getStepVector(direction: vec2<f32>, cell: vec2<f32>, stepLength: f32) -> vec2<f32> {
  return direction * (stepLength / length(direction / cell));
}`,body:`
  let fieldExtent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let origin = vec2<f32>(readParameter(4u), readParameter(5u));
  let cell = vec2<f32>(readParameter(6u), readParameter(7u));
  let stepLength = readParameter(8u);
  let minimumSpeed = readParameter(9u);
  let phase = readParameter(10u);
  let period = readParameter(11u);
  let column = index % WIDTH;
  let row = index / WIDTH;
  let centre = origin + (vec2<f32>(f32(column), f32(row)) + vec2<f32>(0.5)) * cell;
  let centreSample = sampleField(centre, fieldExtent);
  if (centreSample.z == 0.0) {
    var nanBits = 0x7fc00000u;
    values[valuesOffset + index] = bitcast<f32>(nanBits);
    ${h.speeds?`speeds[speedsOffset + index] = bitcast<f32>(nanBits);`:``}
    return;
  }
  ${h.speeds?`speeds[speedsOffset + index] = length(centreSample.xy);`:``}
  var weightSum = getWeight(0.0, phase, period);
  var sum = weightSum * noise[noiseOffset + index];
  for (var direction = 0u; direction < 2u; direction = direction + 1u) {
    let directionSign = select(1.0, -1.0, direction == 1u);
    var position = centre;
    for (var stepIndex = 1u; stepIndex <= STEP_COUNT; stepIndex = stepIndex + 1u) {
      let first = getDirection(position, fieldExtent, directionSign, minimumSpeed);
      if (first.z == 0.0) {
        break;
      }
      let middle = getDirection(position + 0.5 * getStepVector(first.xy, cell, stepLength), fieldExtent, directionSign, minimumSpeed);
      if (middle.z == 0.0) {
        break;
      }
      position = position + getStepVector(middle.xy, cell, stepLength);
      let pixel = floor((position - origin) / cell);
      if (pixel.x < 0.0 || pixel.y < 0.0 || pixel.x >= f32(WIDTH) || pixel.y >= f32(HEIGHT)) {
        break;
      }
      let weight = getWeight(directionSign * f32(stepIndex), phase, period);
      sum = sum + weight * noise[noiseOffset + u32(pixel.y) * WIDTH + u32(pixel.x)];
      weightSum = weightSum + weight;
    }
  }
  values[valuesOffset + index] = select(noise[noiseOffset + index], sum / weightSum, weightSum > 0.0);`})),h.texture){let e=h.texture;x.push(...s(t,()=>new i({id:`${l}-texture`,input:{id:`${l}-texture-band`,format:`float32`,storage:{kind:`buffer`,values:h.values}},output:e}).addToGraph(t)))}return x}};export{h as n,g as r,y as t};
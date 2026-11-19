import{r as e}from"./kernel-C3tfQOn4.js";import{c as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{l as r}from"./wgsl-common-DfyXiYB0.js";import{t as i}from"./bounds-reduction-6MyKJx-7.js";function a(e){return Number.isFinite(e)?t(Math.fround(e)):`bitcast<f32>(0x7f800000u | (index & 0u))`}function o(t,o){let{id:s,operation:c,positions:l,radius:u,geographic:d,times:f,timeRadius:p}=o,m=l.length,h=f?15:12,g=e(t,`${s}-parameters`,`float32`,h),_=[],v=d?void 0:e(t,`${s}-point-bounds`,`float32`,4);v&&_.push(...i(t,{id:`${s}-point-bounds`,operation:c,variant:`neighbor-parameters-point-bounds`,bindings:[{name:`positions`,view:l,type:`f32`,access:`read`}],declarations:`${r}
fn readItem(row: u32) -> vec4f {
  let x = positions[positionsOffset + row * 2u];
  let y = positions[positionsOffset + row * 2u + 1u];
  return vec4f(x, y, select(0.0, 1.0, isFiniteFloat(x) && isFiniteFloat(y)), 0.0);
}`,itemCount:m,output:v}));let y=f?e(t,`${s}-time-bounds`,`float32`,4):void 0;f&&y&&_.push(...i(t,{id:`${s}-time-bounds`,operation:c,variant:`neighbor-parameters-time-bounds`,bindings:[{name:`times`,view:f,type:`f32`,access:`read`}],declarations:`${r}
fn readItem(row: u32) -> vec4f {
  let time = times[timesOffset + row];
  return vec4f(time, 0.0, select(0.0, 1.0, isFiniteFloat(time)), 0.0);
}`,itemCount:m,output:y}));let b=v?`for (var word = 0u; word < 4u; word++) {
    parameters[parametersOffset + word] = bounds[boundsOffset + word];
  }`:`parameters[parametersOffset] = -180.0;
  parameters[parametersOffset + 1u] = -90.0;
  parameters[parametersOffset + 2u] = 180.0;
  parameters[parametersOffset + 3u] = 90.0;`;_.push(n(t,{id:`${s}-parameters`,operation:c,variant:`neighbor-parameters`,bindings:[...v?[{name:`bounds`,view:v,type:`f32`,access:`read`}]:[],...y?[{name:`timeBounds`,view:y,type:`f32`,access:`read`}]:[],{name:`parameters`,view:g,type:`f32`,access:`read_write`}],invocationCount:1,body:`${b}
  parameters[parametersOffset + 4u] = ${a(u)};
  parameters[parametersOffset + 5u] = 0.0;
  parameters[parametersOffset + 6u] = 1.0;
  parameters[parametersOffset + 7u] = 0.0;
  parameters[parametersOffset + 8u] = 1.0;
  parameters[parametersOffset + 9u] = 0.0;
  parameters[parametersOffset + 10u] = 0.0;
  parameters[parametersOffset + 11u] = 0.0;${y?`
  parameters[parametersOffset + 12u] = ${a(p??NaN)};
  parameters[parametersOffset + 13u] = timeBounds[timeBoundsOffset];
  parameters[parametersOffset + 14u] = timeBounds[timeBoundsOffset + 2u];`:``}`}));let x=Math.max(1,Math.ceil(Math.sqrt(m/2)));return{nodes:_,parameters:g,gridSize:[x,x]}}export{o as t};
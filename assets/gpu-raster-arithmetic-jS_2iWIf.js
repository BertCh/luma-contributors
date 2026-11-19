import{s as e}from"./gpu-contributor-utils-BIwLQON5.js";import{n as t}from"./raster-storage-preflight-CIxFc-0g.js";import{a as n,d as r,f as i,h as a,i as o,m as s,o as c,p as l,r as u,s as d,t as f,u as p}from"./raster-algebra-input-5ZxymrFJ.js";import{o as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as h}from"./raster-band-utils-BcTxc8_Y.js";var g=`GPURasterArithmetic`,_=class{id;props;constructor(e){this.id=e.id??`raster-arithmetic`,this.props=e;let{id:t}=this,{cellCount:n,output:o}=e;if(i(t,`cellCount`,n,0,4294967295),s(t,e.noDataValue),!e.a)throw Error(`${t} needs a`);if(c(t,`a`,e.a,e.grid,n),e.b&&c(t,`b`,e.b,e.grid,n),a(t,`mask`,e.mask,`uint32`,n),a(t,`parameters`,e.parameters,`float32`,8),!o?.values)throw Error(`${t} needs output.values`);a(t,`output.values`,o.values,`float32`,n),r(t,{"output.values":o.values},{...f(e.a,`a`),...f(e.b,`b`),mask:e.mask,parameters:e.parameters})}getCommandNodes(r){let i=this.props;d(r,this.id,[i.a,i.b],i.grid),t(r,this.id,[i.mask,i.parameters,...e(i.output)]);let{id:a,props:s}=this;l(a,r,[...o(s.a),...o(s.b),s.mask,s.parameters,s.output.values]);for(let e of[s.a,s.b])e&&n(e)&&h(a,r,e,[]);let c=u(r,`${a}-a`,s.a,s.grid),f=s.b?u(r,`${a}-b`,s.b,s.grid):void 0,_=[{name:`aValues`,view:c.values,type:`f32`,access:`read`},...s.b?[{name:`bValues`,view:f.values,type:`f32`,access:`read`}]:[],...s.mask?[{name:`mask`,view:s.mask,type:`u32`,access:`read`}]:[],{name:`params`,view:s.parameters,type:`f32`,access:`read`},{name:`valuesOut`,view:s.output.values,type:`f32`,access:`read_write`}];return[...c.nodes,...f?.nodes??[],m(r,{id:`${a}-evaluate`,operation:g,variant:`evaluate`,bindings:_,invocationCount:s.cellCount,declarations:`${p(s.noDataValue)}
fn getPower(base: f32, exponent: f32) -> f32 {
  if (base > 0.0) {
    return pow(base, exponent);
  }
  if (base == 0.0 && exponent > 0.0) {
    return 0.0;
  }
  return getNaN();
}`,body:`let operation = u32(params[paramsOffset]);
  let aRaw = aValues[aValuesOffset + index];
  let bRaw = ${s.b?`bValues[bValuesOffset + index]`:`params[paramsOffset + 5u]`};
  let isUnary = operation >= 16u;
  // The sentinel marks raster cells only; a constant second operand that equals it is a value.
  let bMissing = ${s.b?`${n(s.b)?`isNaNValue`:`isNoDataValue`}(bRaw)`:`isNaNValue(bRaw)`};
  let isDefined = !${n(s.a)?`isNaNValue`:`isNoDataValue`}(aRaw) && (isUnary || !bMissing)${s.mask?` && mask[maskOffset + index] != 0u`:``};
  let a = aRaw * params[paramsOffset + 1u] + params[paramsOffset + 2u];
  let b = bRaw * params[paramsOffset + 3u] + params[paramsOffset + 4u];
  let nan = getNaN();
  var result = nan;
  switch operation {
    case 0u: { result = a + b; }
    case 1u: { result = a - b; }
    case 2u: { result = a * b; }
    case 3u: { result = select(a / b, nan, b == 0.0); }
    case 4u: { result = min(a, b); }
    case 5u: { result = max(a, b); }
    case 6u: { result = getPower(a, b); }
    case 7u: { result = abs(a - b); }
    case 8u: {
      let total = a + b;
      result = select((a - b) / total, nan, total == 0.0);
    }
    case 16u: { result = abs(a); }
    case 17u: { result = select(sqrt(a), nan, a < 0.0); }
    case 18u: { result = select(log(a), nan, a <= 0.0); }
    case 19u: { result = exp(a); }
    case 20u: { result = floor(a); }
    case 21u: { result = ceil(a); }
    case 22u: { result = round(a); }
    default: { result = nan; }
  }
  if (isDefined && !isNaNValue(result)) {
    result = min(max(result, params[paramsOffset + 6u]), params[paramsOffset + 7u]);
  } else {
    result = nan;
  }
  valuesOut[valuesOutOffset + index] = result;`})]}};export{_ as t};
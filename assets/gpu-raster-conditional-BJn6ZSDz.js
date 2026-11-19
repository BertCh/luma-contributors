import{s as e}from"./gpu-contributor-utils-BIwLQON5.js";import{n as t}from"./raster-storage-preflight-CIxFc-0g.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as r,d as i,f as a,h as o,i as s,m as c,o as l,p as u,r as d,s as f,t as p,u as m}from"./raster-algebra-input-5ZxymrFJ.js";var h=`GPURasterConditional`,g=class{id;props;constructor(e){this.id=e.id??`raster-conditional`,this.props=e;let{id:t}=this,{cellCount:n,output:r}=e;if(a(t,`cellCount`,n,0,4294967295),c(t,e.noDataValue),!!e.mask==!!e.conditionValues)throw Error(`${t} needs exactly one of mask and conditionValues`);if(o(t,`mask`,e.mask,`uint32`,n),e.conditionValues&&l(t,`conditionValues`,e.conditionValues,e.grid,n),e.a&&l(t,`a`,e.a,e.grid,n),e.b&&l(t,`b`,e.b,e.grid,n),o(t,`parameters`,e.parameters,`float32`,8),!r?.values)throw Error(`${t} needs output.values`);o(t,`output.values`,r.values,`float32`,n),o(t,`output.conditionResult`,r.conditionResult,`uint32`,n),i(t,{"output.values":r.values,"output.conditionResult":r.conditionResult},{...p(e.conditionValues,`conditionValues`),...p(e.a,`a`),...p(e.b,`b`),mask:e.mask,parameters:e.parameters})}getCommandNodes(i){let a=this.props;f(i,this.id,[a.conditionValues,a.a,a.b],a.grid),t(i,this.id,[a.mask,a.parameters,...e(a.output)]);let{id:o,props:c}=this,{output:l}=c;u(o,i,[c.mask,...s(c.conditionValues),...s(c.a),...s(c.b),c.parameters,l.values,l.conditionResult]);let p=c.conditionValues?d(i,`${o}-condition`,c.conditionValues,c.grid):void 0,g=c.a?d(i,`${o}-a`,c.a,c.grid):void 0,_=c.b?d(i,`${o}-b`,c.b,c.grid):void 0,v=[{name:`params`,view:c.parameters,type:`f32`,access:`read`}];return c.mask&&v.push({name:`conditionMask`,view:c.mask,type:`u32`,access:`read`}),c.conditionValues&&v.push({name:`conditionValues`,view:p.values,type:`f32`,access:`read`}),c.a&&v.push({name:`aValues`,view:g.values,type:`f32`,access:`read`}),c.b&&v.push({name:`bValues`,view:_.values,type:`f32`,access:`read`}),v.push({name:`valuesOut`,view:l.values,type:`f32`,access:`read_write`}),l.conditionResult&&v.push({name:`maskOut`,view:l.conditionResult,type:`u32`,access:`read_write`}),[...p?.nodes??[],...g?.nodes??[],..._?.nodes??[],n(i,{id:`${o}-select`,operation:h,variant:`select`,bindings:v,invocationCount:c.cellCount,declarations:m(c.noDataValue),body:`var conditionDefined = true;
  var condition = false;
  ${c.mask?`condition = conditionMask[conditionMaskOffset + index] != 0u;`:`let conditionValue = conditionValues[conditionValuesOffset + index];
  conditionDefined = !${c.conditionValues&&r(c.conditionValues)?`isNaNValue`:`isNoDataValue`}(conditionValue);
  let comparison = u32(params[paramsOffset]);
  let threshold = params[paramsOffset + 1u];
  let upperThreshold = params[paramsOffset + 2u];
  switch comparison {
    case 0u: { condition = conditionValue < threshold; }
    case 1u: { condition = conditionValue <= threshold; }
    case 2u: { condition = conditionValue > threshold; }
    case 3u: { condition = conditionValue >= threshold; }
    case 4u: { condition = conditionValue == threshold; }
    case 5u: { condition = conditionValue != threshold; }
    default: { condition = conditionValue >= threshold && conditionValue <= upperThreshold; }
  }`}
  // The legacy sentinel marks bare raster cells only; each band resolves its own sentinel.
  // A per-frame constant that equals the sentinel is a value
  // (ArcGIS Con, numpy.where); a NaN constant still gives NaN through the select below.
  let aValue = ${c.a?`aValues[aValuesOffset + index]`:`params[paramsOffset + 3u]`};
  let bValue = ${c.b?`bValues[bValuesOffset + index]`:`params[paramsOffset + 4u]`};
  let aMissing = ${c.a&&!r(c.a)?`isNoDataValue(aValue)`:`isNaNValue(aValue)`};
  let bMissing = ${c.b&&!r(c.b)?`isNoDataValue(bValue)`:`isNaNValue(bValue)`};
  let chosen = select(bValue, aValue, condition);
  let isDefined = conditionDefined && !select(bMissing, aMissing, condition);
  valuesOut[valuesOutOffset + index] = select(getNaN(), chosen, isDefined);
  ${l.conditionResult?`maskOut[maskOutOffset + index] = select(0u, 1u, conditionDefined && condition);`:``}`})]}};export{g as t};
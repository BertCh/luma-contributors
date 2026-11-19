import{a as e,i as t,n,r,t as i}from"./raster-time-series-chunks-b-CPab_8L5.js";import{T as a,f as o,g as s}from"./gpu-contributor-utils-BIwLQON5.js";import{t as c}from"./raster-kernel-utils-DcIN7G5I.js";import{a as l,i as u,o as d,r as f,t as p}from"./raster-irregular-date-axis-B315UxmE.js";import{t as m}from"./define-run-B-brkCe4.js";import{a as h,c as g}from"./pointer-overlay-Qh8qflcA.js";import{_,a as v,b as y,c as b,d as x,g as S,m as C,o as w,p as T,r as E,t as D,u as O,x as k,y as A}from"./charts-Br0w6lsw.js";import{Hr as j}from"./entry-preview-D70TdSbK.js";var M=4096;function N(e,t=new Float32Array(3)){return s(`getGPURasterTemporalResampleParameterValues`,`gridStep`,e.gridStep,0),t[0]=e.gridStart,t[1]=e.gridStep,t[2]=e.maximumGap??1e30,t}var P=class{id;props;chunkPlan;estimatedCommandNodeCount;ticksPerYear;constructor(t){this.props=t;let r=t.id??`raster-temporal-resample`;this.id=r,o(r,`cellCount`,t.cellCount),o(r,`sliceCount`,t.sliceCount,2,256),o(r,`gridCount`,t.gridCount,1,M),o(r,`cellCount times gridCount`,t.cellCount*t.gridCount);let i=t.cellCount*t.sliceCount;o(r,`cellCount times sliceCount`,i),a(r,t.values.length>=i&&(!t.mask||t.mask.length>=i),`values and mask must hold cellCount times sliceCount entries`),a(r,t.output.values.length>=t.cellCount*t.gridCount&&t.output.status.length>=t.cellCount*t.gridCount,`output.values and output.status must hold cellCount times gridCount entries`),a(r,t.parameters.length>=3,`parameters must hold 3 entries`),this.ticksPerYear=l(r,t.dateAxis,t.sliceCount),this.chunkPlan=n(this.id,t.cellCount,{values:{view:t.values,entriesPerCell:t.sliceCount},mask:{view:t.mask,entriesPerCell:t.sliceCount},"output.values":{view:t.output.values,entriesPerCell:t.gridCount},"output.status":{view:t.output.status,entriesPerCell:t.gridCount}}),this.estimatedCommandNodeCount=this.chunkPlan.estimatedCommandNodeCount;let s=this.getInputs(),c=this.getOutputs();e(this.id,s,c,this.chunkPlan,[{inputs:s,outputs:c}])}getInputs(){let{values:e,mask:t,parameters:n}=this.props;return[...this.props.dateAxis?[this.props.dateAxis.acquisitionTimes]:[],e,n,...t?[t]:[]]}getOutputs(){return[this.props.output.values,this.props.output.status]}getCommandNodes(e){let{props:n,id:a}=this,{sliceCount:o,gridCount:s,values:l,mask:p,parameters:m,dateAxis:h,output:g}=n;return t(a,e,this.getInputs(),this.getOutputs()),d(a,e,h),i(e,this.chunkPlan,{id:a,operation:`GPURasterTemporalResample`,invocationCount:n.cellCount,workgroupSize:16,bindings:[r(`values`,l),r(`parameters`,m),...p?[r(`mask`,p)]:[],f(h),r(`resampled`,g.values,!0),r(`resampleStatus`,g.status,!0)],declarations:`${c}
${u(this.ticksPerYear,o)}
fn isObserved(cell: u32, slice: u32) -> bool {
  let source = cell * ${o}u + slice;
  return finite(values[valuesOffset + source]) ${p?`&& mask[maskOffset + source] != 0u`:``};
}
fn advanceObserved(cell: u32, start: u32) -> u32 {
  var slice = start;
  while (slice < ${o}u && !isObserved(cell, slice)) { slice++; }
  return slice;
}`,body:`let gridStart = parameters[parametersOffset];
let gridStep = parameters[parametersOffset + 1u];
let maximumGap = parameters[parametersOffset + 2u];
var left = ${o}u;
var right = advanceObserved(index, 0u);
for (var point = 0u; point < ${s}u; point++) {
  let gridYears = gridStart + f32(point) * gridStep;
  while (right < ${o}u && rasterSliceYears(right) <= gridYears) {
    left = right;
    right = advanceObserved(index, right + 1u);
  }
  var value = missing();
  var code = 1u;
  if (left < ${o}u && right < ${o}u) {
    let leftYears = rasterSliceYears(left);
    let span = rasterYearsBetween(right, left);
    if (span > maximumGap) {
      code = 2u;
    } else {
      let fraction = (gridYears - leftYears) / span;
      value = mix(values[valuesOffset + index * ${o}u + left], values[valuesOffset + index * ${o}u + right], fraction);
      code = 0u;
    }
  } else if (left < ${o}u && rasterSliceYears(left) == gridYears) {
    value = values[valuesOffset + index * ${o}u + left];
    code = 0u;
  }
  resampled[resampledOffset + index * ${s}u + point] = value;
  resampleStatus[resampleStatusOffset + index * ${s}u + point] = code;
}`})}},[F]=E(1),I=[-.2,1];function L(e,t,n){return Array.from({length:n},(r,i)=>{let a=0,o=0;for(let r=0;r<t;r++){let t=e[r*n+i];Number.isFinite(t)&&(a+=t,o++)}return o>0?a/o:NaN})}function R(e){let t=_(e);return{...t,means:L(t.observed,t.grid.cellCount,t.sliceCount)}}function z(e,t){let{grid:n,sliceCount:r}=t,i=e.upload(`series`,`float32`,t.observed),a=e.allocate(`parameters`,`float32`,3),o={values:e.allocate(`resampled`,`float32`,n.cellCount),status:e.allocate(`status`,`uint32`,n.cellCount)};e.graph.add(new P({cellCount:n.cellCount,sliceCount:r,gridCount:1,values:i.view,parameters:a.view,dateAxis:k(e,t.times),output:{values:o.values.view,status:o.status.view}}));let s=D(e,`scene-means`,r);return S(s,Array.from(t.years),t.means),{outputs:{...o,means:s,preFire:e.upload(`pre-fire`,`float32`,A(t,6))},update:e=>a.write(N({gridStart:e.gridStart,gridStep:.05,maximumGap:e.maximumGap}))}}var B=m({parameters:j,caption:`Measured Dixie NDVI, 16 irregular Sentinel-2 scenes, resampled linearly between scenes onto a one-point grid.`,createInputs:({aspect:e})=>R(e),build:(e,t)=>z(e,t),interaction:h(j,{kind:`time`,hint:`scrub gridStart`,x:`gridStart`,y:`maximumGap`,decorate:(e,t,{inputs:n,world:r})=>{let i=e.gridStart,a=[n.years[0],n.years[n.sliceCount-1]],o=Array.from(n.years).findIndex(e=>e>i),s=y(BigInt(Math.round(x(n.times)+i*p))),c=o>0,l=c?n.years[o]-n.years[o-1]:0;return{glyphs:[w(r,F),...C(r,F,a,x(n.times)),T(r,F,a,i)],readout:c?l>e.maximumGap?`gridStart ${s} · gap ${g(l)} years over maximumGap`:`gridStart ${s} · between ${y(n.times[o-1])} and ${y(n.times[o])}`:`gridStart ${s} · outside the scenes`}}}),draw:({inputs:e,outputs:t,theme:n})=>{let r=[e.years[0],e.years[e.sliceCount-1]];return[O(t.preFire,e.grid,e.grid.mapLeft,{context:!0,ramp:`sequential`,range:[0,1]}),O(t.values,e.grid,e.grid.mapLeft,{ramp:`sequential`,range:[-.5,.9]}),v(t.means,F,I,r,b(n),5)]}});export{B as default};
import{o as e}from"./raster-band-utils-BcTxc8_Y.js";import{s as t}from"./gpu-contributor-utils-BIwLQON5.js";import{n}from"./raster-storage-preflight-CIxFc-0g.js";import{o as r,t as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{a,c as o,d as s,f as c,h as l,i as u,m as d,o as f,p,r as m,s as h,t as g,u as _}from"./raster-algebra-input-5ZxymrFJ.js";import{a as v}from"./pointer-overlay-Qh8qflcA.js";import{t as y}from"./define-run-B-brkCe4.js";import{i as b}from"./inputs-CJKKIRKK.js";import{Pi as x}from"./entry-preview-D70TdSbK.js";import{t as S}from"./bands-CVDIl0aE.js";function C(t,n=new Float32Array(4)){if(e(`getGPURasterReclassifyParameterValues`,n,4),!Number.isSafeInteger(t.breakCount)||t.breakCount<0)throw Error(`getGPURasterReclassifyParameterValues breakCount must be a non-negative integer`);return n[0]=t.breakCount,n[1]=+(t.closed===`right`),n[2]=0,n[3]=0,n}var w=`GPURasterReclassify`,T=256,E=class{id;props;rowCount;maximumBreakCount;constructor(e){this.id=e.id??`raster-reclassify`,this.props=e;let{id:t}=this,{output:n}=e,r=f(t,`values`,e.values,e.grid);if(this.rowCount=e.rowCount??r,c(t,`rowCount`,this.rowCount,0,r),d(t,e.noDataValue),l(t,`mask`,e.mask,`uint32`,this.rowCount),l(t,`breaks`,e.breaks,`float32`,1),this.maximumBreakCount=e.breaks.length,l(t,`classValues`,e.classValues,`float32`,this.maximumBreakCount+1),l(t,`parameters`,e.parameters,`float32`,4),!n.classes&&!n.reclassified&&!n.classCounts)throw Error(`${t} needs at least one output`);if(n.reclassified&&!e.classValues)throw Error(`${t} output.reclassified requires classValues`);l(t,`output.classes`,n.classes,`uint32`,this.rowCount),l(t,`output.reclassified`,n.reclassified,`float32`,this.rowCount),l(t,`output.classCounts`,n.classCounts,`uint32`,this.maximumBreakCount+1),s(t,{"output.classes":n.classes,"output.reclassified":n.reclassified,"output.classCounts":n.classCounts},{...g(e.values,`values`),mask:e.mask,breaks:e.breaks,classValues:e.classValues,parameters:e.parameters})}getCommandNodes(e){let s=this.props;h(e,this.id,[s.values],s.grid),n(e,this.id,[s.mask,s.breaks,s.classValues,s.parameters,...t(s.output)]);let{id:c,props:l,rowCount:d,maximumBreakCount:f}=this,{output:g}=l;p(c,e,[...u(l.values),l.mask,l.breaks,l.classValues,l.parameters,g.classes,g.reclassified,g.classCounts]);let v=m(e,`${c}-input`,l.values,l.grid),y=[...v.nodes];g.classCounts&&y.push(i(e,{id:`${c}-clear-counts`,operation:w,view:g.classCounts,type:`u32`,value:`0u`,componentCount:f+1}));let b=!!g.classCounts&&(f+1)*4<=e.device.limits.maxComputeWorkgroupStorageSize,x=[{name:`source`,view:v.values,type:`f32`,access:`read`},{name:`breaks`,view:l.breaks,type:`f32`,access:`read`},{name:`params`,view:l.parameters,type:`f32`,access:`read`}];return l.mask&&x.push({name:`validity`,view:l.mask,type:`u32`,access:`read`}),l.classValues&&x.push({name:`classValues`,view:l.classValues,type:`f32`,access:`read`}),g.classes&&x.push({name:`classesOut`,view:g.classes,type:`u32`,access:`read_write`}),g.reclassified&&x.push({name:`reclassifiedOut`,view:g.reclassified,type:`f32`,access:`read_write`}),g.classCounts&&x.push({name:`countsOut`,view:g.classCounts,type:`atomic<u32>`,access:`read_write`}),y.push(r(e,{id:`${c}-classify`,operation:w,variant:`classify`,bindings:x,invocationCount:d,guardIndex:!b,workgroupSize:b?T:void 0,declarations:`const MAXIMUM_BREAK_COUNT: u32 = ${f}u;
${b?`var<workgroup> localCounts: array<atomic<u32>, ${f+1}>;`:``}
const NO_DATA_CLASS: u32 = 0xffffffffu;
${_(a(l.values)?void 0:l.noDataValue)}
${o(`countBreaksBelow`,`breaks`)}`,body:`${b?`for (var slot = localInvocationIndex; slot <= MAXIMUM_BREAK_COUNT; slot += ${T}u) {
    atomicStore(&localCounts[slot], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {`:``}
  let value = source[sourceOffset + index];
  var isValid = !isNoDataValue(value);
  ${l.mask?`isValid = isValid && validity[validityOffset + index] != 0u;`:``}
  let breakCountValue = params[paramsOffset];
  let breakCount = min(u32(clamp(select(0.0, breakCountValue, isFiniteValue(breakCountValue)), 0.0, 4294967040.0)), MAXIMUM_BREAK_COUNT);
  let closedRight = params[paramsOffset + 1u] != 0.0;
  var classIndex = NO_DATA_CLASS;
  if (isValid) {
    classIndex = countBreaksBelow(0u, breakCount, value, closedRight);
  }
  ${g.classes?`classesOut[classesOutOffset + index] = classIndex;`:``}
  ${g.reclassified?`reclassifiedOut[reclassifiedOutOffset + index] = select(getNaN(), classValues[classValuesOffset + min(classIndex, MAXIMUM_BREAK_COUNT)], isValid);`:``}
  ${g.classCounts?b?`if (isValid) {
    atomicAdd(&localCounts[classIndex], 1u);
  }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot <= MAXIMUM_BREAK_COUNT; slot += ${T}u) {
    let partial = atomicLoad(&localCounts[slot]);
    if (partial != 0u) {
      atomicAdd(&countsOut[countsOutOffset + slot], partial);
    }
  }`:`if (isValid) {
    atomicAdd(&countsOut[countsOutOffset + classIndex], 1u);
  }`:``}`})),y}},D=7,O=v(x,{kind:`level`,hint:`move the first break and count`,x:`firstBreak`,y:`breakCount`}),k=y({parameters:x,caption:`A continuous raster in; one class per cell out.`,createInputs:({world:e})=>b(e),build(e,t){let n=t.values.length,r=e.upload(`values`,`float32`,t.values),i=e.allocate(`breaks`,`float32`,D),a=e.allocate(`parameters`,`float32`,4),o=e.allocate(`classes`,`uint32`,n);return e.graph.add(new E({values:r.view,breaks:i.view,parameters:a.view,output:{classes:o.view}})),{outputs:{source:r,classes:o},update:e=>{let t=(.95-e.firstBreak)/Math.max(1,e.breakCount-1);i.write(Float32Array.from({length:D},(n,r)=>e.firstBreak+r*t)),a.write(C({breakCount:e.breakCount}))}}},interaction:{...O,resolve:(e,t)=>{let n=O.resolve(e,t),r={...t.values,...n.values};return{...n,glyphs:[{shape:`crosshair`,at:[e.x,e.y]}],readout:`breaks[0] ${r.firstBreak.toFixed(2)} | ${r.breakCount+1} classes`}}},draw:({inputs:e,outputs:t,values:n})=>[S(e,t.source,{context:!0}),S(e,t.classes,{ramp:`sequential`,range:[0,n.breakCount],bands:n.breakCount+1})]});export{k as default};
import{i as e,t}from"./graph-chunk-groups-BB8ACpQO.js";import{u as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-dispatch-utils-CbRzFAgT.js";import{E as i,T as a,c as o,p as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{a as l,i as u,o as d,r as f}from"./graph-chunk-dispatch-zBt8k2-o.js";var p=`GPUClassAssignment`,m=256,h=4294967295,g=class{id;props;classifiedValues;estimatedCommandNodeCount;chunkPlan;constructor(e){let t=e.id??`class-assignment`;this.id=t,this.props=e,this.classifiedValues={classes:this.props.output,mask:this.props.mask,classCount:e.classCount};let r=e.intervalClosure??`left`;a(t,r===`left`||r===`right`,`intervalClosure must be 'left' or 'right'`);for(let n of o(e.values))i(n,[`float32`],`${t} values`);i(e.breaks,[`float32`],`${t} breaks`),n(e.classCount,`${t} classCount`);for(let r of o(e.output))n(r,`${t} output`);if(e.classCount.length<1)throw Error(`${t} classCount must hold one row`);if(e.breaks.length<2)throw Error(`${t} breaks must hold at least two rows`);if(e.output.length!==e.values.length)throw Error(`${t} output length must equal values length`);if(e.mask){for(let r of o(e.mask))n(r,`${t} mask`);if(e.mask.length!==e.values.length)throw Error(`${t} mask length must equal values length`)}c(t,[e.output],[e.values,e.breaks,e.classCount,e.mask]),u(t,{output:e.output}),this.chunkPlan=f(t,{values:e.values,mask:e.mask,output:e.output}),this.estimatedCommandNodeCount=this.chunkPlan.nonemptyChunkCount,s(t,this.estimatedCommandNodeCount)}getCommandNodes(n){let{id:i,props:a,chunkPlan:o}=this,{breaks:s,classCount:c,mask:u}=a;d(n,i,o),l(n,i,{breaks:s,classCount:c}),e(n,i,4+Number(!!u),m);for(let e of o.chunks)e.length>0&&r(i,e.length,m,n.device.limits.maxComputeWorkgroupsPerDimension);let f=a.intervalClosure===`right`,g=f?`breaks[breaksOffset + edge] < value`:`breaks[breaksOffset + edge] <= value`;return[...o.chunks.filter(e=>e.length>0).map(e=>t(n,{id:`${i}-assign-chunk-${e.chunkIndex}`,operation:p,...f?{variant:`right-closed`}:{},bindings:[{name:`values`,view:e.views.values,type:`f32`,access:`read`},{name:`breaks`,view:s,type:`f32`,access:`read`},{name:`classCount`,view:c,type:`u32`,access:`read`},...u?[{name:`mask`,view:e.views.mask,type:`u32`,access:`read`}]:[],{name:`output`,view:e.views.output,type:`u32`,access:`read_write`}],invocationCount:e.length,workgroupSize:m,body:`if (index >= getChunkParameter(1u)) { return; }
  let valuesBase = arrayLength(&values) - getChunkParameter(1u);
  let outputBase = arrayLength(&output) - getChunkParameter(1u);
  ${u?`let maskBase = arrayLength(&mask) - getChunkParameter(1u);`:``}
  let value = values[valuesBase + index];
  let count = min(classCount[classCountOffset], arrayLength(&breaks) - breaksOffset - 1u);
  var assigned = ${h}u;
  // WGSL gives no NaN guarantees to comparisons, so test the exponent and mantissa bits.
  let isNaN = (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
  if (count > 0u && !isNaN && ${u?`mask[maskBase + index] != 0u`:`true`}) {
    var below = 0u;
    for (var edge = 1u; edge < count; edge++) {
      if (${g}) {
        below++;
      }
    }
    assigned = below;
  }
  output[outputBase + index] = assigned;`}))]}};export{h as n,g as t};
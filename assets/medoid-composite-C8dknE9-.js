import{a as e,i as t,n,r,t as i}from"./raster-time-series-chunks-b-CPab_8L5.js";import{T as a,f as o}from"./gpu-contributor-utils-BIwLQON5.js";import{t as s}from"./raster-kernel-utils-DcIN7G5I.js";import{t as c}from"./define-run-B-brkCe4.js";import{c as l}from"./pointer-overlay-Qh8qflcA.js";import{a as u,i as d,n as f,r as p,t as m}from"./panel-lHgDcAzn.js";import{pi as h}from"./entry-preview-D70TdSbK.js";import{a as g,i as _,r as v}from"./gestures-C1AoqRjN.js";import{a as y,i as b,n as x,o as S,r as C,t as w}from"./hazy-scenes-CtxODDyG.js";var T=class{id;props;chunkPlan;estimatedCommandNodeCount;constructor(t){this.props=t;let r=t.id??`raster-medoid-composite`;this.id=r,o(r,`cellCount`,t.cellCount),o(r,`sliceCount`,t.sliceCount,1,64),o(r,`bandCount`,t.bandCount,1,16);let i=t.cellCount*t.sliceCount;o(r,`cellCount times sliceCount`,i),a(r,t.values.length>=i*t.bandCount&&(!t.mask||t.mask.length>=i),`values must hold cellCount times sliceCount times bandCount entries and mask cellCount times sliceCount`),a(r,t.output.composite.length>=t.cellCount*t.bandCount&&t.output.sliceIndex.length>=t.cellCount&&t.output.status.length>=t.cellCount,`output.composite must hold cellCount times bandCount entries; sliceIndex and status one per cell`),this.chunkPlan=n(this.id,t.cellCount,{values:{view:t.values,entriesPerCell:t.sliceCount*t.bandCount},mask:{view:t.mask,entriesPerCell:t.sliceCount},"output.composite":{view:t.output.composite,entriesPerCell:t.bandCount},"output.sliceIndex":{view:t.output.sliceIndex,entriesPerCell:1},"output.status":{view:t.output.status,entriesPerCell:1}}),this.estimatedCommandNodeCount=this.chunkPlan.estimatedCommandNodeCount,e(this.id,this.getInputs(),this.getOutputs(),this.chunkPlan)}getInputs(){return[this.props.values,...this.props.mask?[this.props.mask]:[]]}getOutputs(){let{composite:e,sliceIndex:t,status:n}=this.props.output;return[e,t,n]}getCommandNodes(e){let{props:n,id:a}=this,{sliceCount:o,bandCount:c,values:l,mask:u,output:d}=n;return t(a,e,this.getInputs(),this.getOutputs()),i(e,this.chunkPlan,{id:a,operation:`GPURasterMedoidComposite`,invocationCount:n.cellCount,workgroupSize:64,bindings:[r(`values`,l),...u?[r(`mask`,u)]:[],r(`composite`,d.composite,!0),r(`sliceIndex`,d.sliceIndex,!0),r(`status`,d.status,!0)],declarations:`${s}
fn isObservationValid(cell: u32, slice: u32) -> bool {
  let source = cell * ${o}u + slice;
  ${u?`if (mask[maskOffset + source] == 0u) { return false; }`:``}
  for (var band = 0u; band < ${c}u; band++) {
    if (!finite(values[valuesOffset + source * ${c}u + band])) { return false; }
  }
  return true;
}`,body:`var validMask = array<u32, 2>(0u, 0u);
for (var slice = 0u; slice < ${o}u; slice++) {
  if (isObservationValid(index, slice)) { validMask[slice / 32u] |= 1u << (slice % 32u); }
}
var bestSlice = 0xffffffffu;
var bestSum = 0.0;
for (var candidate = 0u; candidate < ${o}u; candidate++) {
  if (((validMask[candidate / 32u] >> (candidate % 32u)) & 1u) == 0u) { continue; }
  var sum = 0.0;
  for (var other = 0u; other < ${o}u; other++) {
    if (other == candidate || ((validMask[other / 32u] >> (other % 32u)) & 1u) == 0u) { continue; }
    var squared = 0.0;
    for (var band = 0u; band < ${c}u; band++) {
      let difference = values[valuesOffset + (index * ${o}u + candidate) * ${c}u + band]
        - values[valuesOffset + (index * ${o}u + other) * ${c}u + band];
      squared += difference * difference;
    }
    sum += sqrt(squared);
  }
  if (bestSlice == 0xffffffffu || sum < bestSum) { bestSlice = candidate; bestSum = sum; }
}
sliceIndex[sliceIndexOffset + index] = bestSlice;
status[statusOffset + index] = select(0u, 1u, bestSlice == 0xffffffffu);
for (var band = 0u; band < ${c}u; band++) {
  var component = missing();
  if (bestSlice != 0xffffffffu) { component = values[valuesOffset + (index * ${o}u + bestSlice) * ${c}u + band]; }
  composite[compositeOffset + index * ${c}u + band] = component;
}`})}},E=4294967295,D=[.03,.06,.4,.44],O=[0,.55],k=c({parameters:h,rebuildOn:[`sliceCount`],caption:`Eight synthetic scenes from one July image in; one real observation per cell out.`,createInputs:({random:e,aspect:t,world:n})=>({panel:m(n,[p(D)]),scenes:C(e,t,20),composed:null}),build(e,t,n){let{cellCount:r}=t.scenes,i=Math.round(n.sliceCount),a=e.allocate(`values`,`float32`,r*i*w),o=e.allocate(`validity`,`uint32`,r*i),s=e.allocate(`composite`,`float32`,r*w),c=e.allocate(`slice-index`,`uint32`,r),l=e.allocate(`status`,`uint32`,r);e.graph.add(new T({cellCount:r,sliceCount:i,bandCount:w,values:a.view,mask:o.view,output:{composite:s.view,sliceIndex:c.view,status:l.view}}));let d=i*w,f=Float32Array.from({length:d},(e,t)=>t%w);return{outputs:{panel:u(e,t.panel),composite:s,sliceIndex:c,bandX:e.upload(`band-x`,`float32`,f),validY:e.allocate(`valid-y`,`float32`,d),flaggedY:e.allocate(`flagged-y`,`float32`,d),meanLine:e.allocate(`mean-line`,`float32`,w),medoidLine:e.allocate(`medoid-line`,`float32`,w)},update:e=>{let n=x(t.scenes,i,e.haze);t.composed=n,a.write(n.values),o.write(n.validity)}}},async afterRun({inputs:e,outputs:t,upload:n}){let r=await t.composite.buffer.readAsync(),i=new Float32Array(r.buffer,r.byteOffset,e.scenes.cellCount*w);return{nearInfrared:n(`near-infrared`,`float32`,Float32Array.from({length:e.scenes.cellCount},(e,t)=>i[t*w+3]))}},interaction:{kind:`probe`,hint:`probe a cell`,rest:({inputs:e,world:t,toUnit:n})=>{let{column:r,row:i}=b(e.scenes,8);return n(g(e.scenes,t,r,i))},resolve:(e,{inputs:t,outputs:n,values:r,world:i,peek:a})=>{let{scenes:o,composed:s}=t,c={columns:o.columns,rows:o.rows},{column:u,row:d,index:f}=v(c,i,e);if(!s)return{};let p=Math.round(r.sliceCount),m=y(s,f,p);n.validY.write(m.accepted),n.flaggedY.write(m.all),n.meanLine.write(S(s,f,p));let h=a(n.composite),g=a(n.sliceIndex);h&&n.medoidLine.write(h.slice(f*w,(f+1)*w));let b=0;for(let e=0;e<p;e++)b+=s.validity[f*p+e];let x=g?g[f]:E;return{spotlight:null,glyphs:[_(c,i,u,d),{shape:`box`,minimum:[i.width*D[0],i.height*D[1]],maximum:[i.width*D[2],i.height*D[3]]}],readout:x===E?`no valid observation (${l(b,0)} of ${p})`:`medoid: scene ${x} | ${l(b,0)} of ${p} valid`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>[{layer:`raster`,values:n.nearInfrared,columns:e.scenes.columns,rows:e.scenes.rows,ramp:`sequential`,range:[0,.3]},...f(t.panel,r),{layer:`chart`,kind:`scatter`,values:t.flaggedY,xValues:t.bandX,xRange:[-.5,w-.5],range:O,rectangle:D,radius:3.5,color:[1,.72,.2,.95]},{layer:`chart`,kind:`scatter`,values:t.validY,xValues:t.bandX,xRange:[-.5,w-.5],range:O,rectangle:D,radius:3.5,color:d(r.ink,.8)},{layer:`chart`,kind:`line`,values:t.meanLine,range:O,rectangle:D,color:d(r.muted,1)},{layer:`chart`,kind:`line`,values:t.medoidLine,range:O,rectangle:D,color:d(r.categories[0],1)}]});export{k as default};
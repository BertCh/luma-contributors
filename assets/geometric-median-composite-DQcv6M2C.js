import{d as e}from"./kernel-C3tfQOn4.js";import{a as t,i as n,n as r,t as i}from"./raster-time-series-chunks-FYvfnBw_.js";import{T as a,c as o,f as s,g as c,p as l}from"./gpu-contributor-utils-BIwLQON5.js";import{a as u,t as d}from"./raster-kernel-utils-DcIN7G5I.js";import{t as f}from"./define-run-B-brkCe4.js";import{c as p}from"./pointer-overlay-Qh8qflcA.js";import{a as m,i as h,n as g,r as _,t as v}from"./panel-lHgDcAzn.js";import{mi as y}from"./entry-preview-D70TdSbK.js";import{a as b,i as x,r as S}from"./gestures-C1AoqRjN.js";import{a as C,i as w,n as T,o as E,r as D,t as O}from"./hazy-scenes-CtxODDyG.js";function k(e){let{id:t=`getGPURasterGeometricMedianCompositeOutputLayout`,cellCount:n,bandCount:r}=e;s(t,`cellCount`,n),s(t,`bandCount`,r);let i=n*r;return s(t,`cellCount times bandCount`,i),{composite:{format:`float32`,length:i},status:{format:`uint32`,length:n},...e.lastStep?{lastStep:{format:`float32`,length:n}}:{}}}function A(e={},t=new Float32Array(1)){let n=e.distanceEpsilon??1e-5;return c(`getGPURasterGeometricMedianCompositeParameterValues`,`distanceEpsilon`,n,Number.MIN_VALUE,1/0),t[0]=n,t}var j=class{props;id;estimatedCommandNodeCount;chunkProps;constructor(n){this.props=n;let i=n.id??`raster-geometric-median-composite`;this.id=i,s(i,`cellCount`,n.cellCount),s(i,`sliceCount`,n.sliceCount,1,256),s(i,`bandCount`,n.bandCount,1,16),s(i,`iterationCount`,n.iterationCount,1,64);let c=n.cellCount*n.sliceCount;s(i,`cellCount times sliceCount`,c),a(i,n.values.length>=c*n.bandCount&&(!n.mask||n.mask.length>=c),`values must hold cellCount times sliceCount times bandCount entries and mask cellCount times sliceCount`);let u=k({id:i,cellCount:n.cellCount,bandCount:n.bandCount,lastStep:n.output.lastStep!==void 0});for(let[t,r]of Object.entries(u)){let s=n.output[t];a(i,s.length>=r.length,`output.${t} must hold at least ${r.length} entries`);for(let n of o(s))e(n,[r.format],`${i} output.${t}`)}a(i,n.parameters.length>=1,`parameters must hold 1 entries`),this.chunkProps=r(i,n,{values:n.sliceCount*n.bandCount,mask:n.sliceCount,"output.composite":n.bandCount,"output.status":1,"output.lastStep":1}),this.estimatedCommandNodeCount=this.chunkProps.length*1,l(i,this.estimatedCommandNodeCount),t(i,this.getInputs(),this.getOutputs())}getInputs(){let{values:e,mask:t,parameters:n}=this.props;return[e,n,...t?[t]:[]]}getOutputs(){let{composite:e,status:t,lastStep:n}=this.props.output;return[e,t,...n?[n]:[]]}getCommandNodes(e){return n(e,this.id,[...this.getInputs(),...this.getOutputs()]),this.chunkProps.flatMap((t,n)=>this.getChunkCommandNodes(e,{...t,id:`${this.id}-chunk-${n}`}))}getChunkCommandNodes(e,t){let n=t.id,{sliceCount:r,bandCount:a,iterationCount:o,values:s,mask:c,parameters:l,output:f}=t;return[i(e,{id:n,operation:`GPURasterGeometricMedianComposite`,invocationCount:t.cellCount,workgroupSize:64,bindings:[u(`values`,s),u(`parameters`,l),...c?[u(`mask`,c)]:[],u(`composite`,f.composite,!0),u(`status`,f.status,!0),...f.lastStep?[u(`lastStep`,f.lastStep,!0)]:[]],declarations:`${d}
fn isObservationValid(cell: u32, slice: u32) -> bool {
  let source = cell * ${r}u + slice;
  ${c?`if (mask[maskOffset + source] == 0u) { return false; }`:``}
  for (var band = 0u; band < ${a}u; band++) {
    if (!finite(values[valuesOffset + source * ${a}u + band])) { return false; }
  }
  return true;
}`,body:`let distanceEpsilon = parameters[parametersOffset];
var estimate: array<f32, ${a}>;
var numerator: array<f32, ${a}>;
var validCount = 0u;
for (var slice = 0u; slice < ${r}u; slice++) {
  if (!isObservationValid(index, slice)) { continue; }
  validCount++;
  for (var band = 0u; band < ${a}u; band++) {
    estimate[band] += values[valuesOffset + (index * ${r}u + slice) * ${a}u + band];
  }
}
var step = 0.0;
if (validCount > 0u) {
  for (var band = 0u; band < ${a}u; band++) { estimate[band] /= f32(validCount); }
  for (var iteration = 0u; iteration < ${o}u; iteration++) {
    var weightSum = 0.0;
    for (var band = 0u; band < ${a}u; band++) { numerator[band] = 0.0; }
    for (var slice = 0u; slice < ${r}u; slice++) {
      if (!isObservationValid(index, slice)) { continue; }
      let base = (index * ${r}u + slice) * ${a}u;
      var squared = 0.0;
      for (var band = 0u; band < ${a}u; band++) {
        let difference = values[valuesOffset + base + band] - estimate[band];
        squared += difference * difference;
      }
      let weight = 1.0 / max(sqrt(squared), distanceEpsilon);
      weightSum += weight;
      for (var band = 0u; band < ${a}u; band++) { numerator[band] += weight * values[valuesOffset + base + band]; }
    }
    var squaredStep = 0.0;
    for (var band = 0u; band < ${a}u; band++) {
      let next = numerator[band] / weightSum;
      squaredStep += (next - estimate[band]) * (next - estimate[band]);
      estimate[band] = next;
    }
    step = sqrt(squaredStep);
  }
}
status[statusOffset + index] = select(0u, 1u, validCount == 0u);
${f.lastStep?`lastStep[lastStepOffset + index] = select(step, missing(), validCount == 0u);`:``}
for (var band = 0u; band < ${a}u; band++) {
  composite[compositeOffset + index * ${a}u + band] = select(estimate[band], missing(), validCount == 0u);
}`})]}},M=[1,2,4,8,16,32],N=[1e-5,.001,.01,.05,.2,.5],P=2,F=3,I=[.03,.06,.4,.5],L=[0,.35],R=[.05,.35],z=f({parameters:y,rebuildOn:[`iterationCount`],buildCache:6,caption:`Eight synthetic scenes from one July image in; the geometric median per cell out.`,createInputs:({random:e,aspect:t,world:n})=>({panel:v(n,[_(I)]),scenes:D(e,t,20),composed:null}),build(e,t,n){let{cellCount:r}=t.scenes,i=e.allocate(`values`,`float32`,r*8*O),a=e.allocate(`validity`,`uint32`,r*8),o=e.allocate(`parameters`,`float32`,1),s=e.allocate(`composite`,`float32`,r*O),c=e.allocate(`status`,`uint32`,r),l=e.allocate(`last-step`,`float32`,r);e.graph.add(new j({cellCount:r,sliceCount:8,bandCount:O,iterationCount:M[Math.round(n.iterationCount)],values:i.view,mask:a.view,parameters:o.view,output:{composite:s.view,status:c.view,lastStep:l.view}}));let u=t=>e.allocate(t,`float32`,8);return{outputs:{panel:m(e,t.panel),composite:s,lastStep:l,allRed:u(`all-red`),allNearInfrared:u(`all-near-infrared`),acceptedRed:u(`accepted-red`),acceptedNearInfrared:u(`accepted-near-infrared`),meanRed:e.allocate(`mean-red`,`float32`,1),meanNearInfrared:e.allocate(`mean-near-infrared`,`float32`,1),medianRed:e.allocate(`median-red`,`float32`,1),medianNearInfrared:e.allocate(`median-near-infrared`,`float32`,1)},update:e=>{let n=T(t.scenes,8,e.haze);t.composed=n,i.write(n.values),a.write(n.validity),o.write(A({distanceEpsilon:N[Math.round(e.distanceEpsilon)]}))}}},async afterRun({inputs:e,outputs:t,upload:n}){let r=await t.composite.buffer.readAsync(),{cellCount:i}=e.scenes,a=new Float32Array(r.buffer,r.byteOffset,i*O);return{nearInfrared:n(`near-infrared`,`float32`,Float32Array.from({length:i},(e,t)=>a[t*O+F]))}},interaction:{kind:`probe`,hint:`probe a cell`,rest:({inputs:e,world:t,toUnit:n})=>{let{column:r,row:i}=w(e.scenes,8);return n(b(e.scenes,t,r,i))},resolve:(e,{inputs:t,outputs:n,world:r,peek:i})=>{let{scenes:a,composed:o}=t,s={columns:a.columns,rows:a.rows},{column:c,row:l,index:u}=S(s,r,e);if(!o)return{};let d=C(o,u,8),f=(e,t)=>Float32Array.from({length:8},(n,r)=>e[r*O+t]);n.allRed.write(f(d.all,P)),n.allNearInfrared.write(f(d.all,F)),n.acceptedRed.write(f(d.accepted,P)),n.acceptedNearInfrared.write(f(d.accepted,F));let m=E(o,u,8);n.meanRed.write(Float32Array.of(m[P])),n.meanNearInfrared.write(Float32Array.of(m[F]));let h=i(n.composite),g=i(n.lastStep);h&&(n.medianRed.write(Float32Array.of(h[u*O+P])),n.medianNearInfrared.write(Float32Array.of(h[u*O+F])));let _=h?Math.hypot(h[u*O+P]-m[P],h[u*O+F]-m[F]):NaN;return{spotlight:null,glyphs:[x(s,r,c,l),{shape:`box`,minimum:[r.width*I[0],r.height*I[1]],maximum:[r.width*I[2],r.height*I[3]]}],readout:`median moved (red + NIR) ${p(_,3)} from the mean | last step ${g?p(g[u],4):`-`}`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>{let i={layer:`chart`,kind:`scatter`,xRange:L,range:R,rectangle:I};return[{layer:`raster`,values:n.nearInfrared,columns:e.scenes.columns,rows:e.scenes.rows,ramp:`sequential`,range:[0,.3]},...g(t.panel,r),{...i,values:t.allNearInfrared,xValues:t.allRed,radius:4.5,color:[1,.72,.2,.95]},{...i,values:t.acceptedNearInfrared,xValues:t.acceptedRed,radius:4.5,color:h(r.ink,.8)},{...i,values:t.meanNearInfrared,xValues:t.meanRed,radius:8,color:h(r.muted,1)},{...i,values:t.medianNearInfrared,xValues:t.medianRed,radius:8,color:h(r.categories[0],1)}]}});export{z as default};
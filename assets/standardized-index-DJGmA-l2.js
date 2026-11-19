import{a as e,i as t,n,r,t as i}from"./raster-time-series-chunks-b-CPab_8L5.js";import{T as a,f as o}from"./gpu-contributor-utils-BIwLQON5.js";import{t as s}from"./scaled-quotient-wgsl-UDyhQag6.js";import{t as c}from"./raster-kernel-utils-DcIN7G5I.js";import{d as l,r as u}from"./wgsl-common-DfyXiYB0.js";import{t as d}from"./incomplete-gamma-wgsl-BipKCma-.js";import{n as f,t as p}from"./gpu-raster-standardized-index-calibration-DySGuRcZ.js";import{t as m}from"./define-run-B-brkCe4.js";import{a as h,c as g}from"./pointer-overlay-Qh8qflcA.js";import{ui as _}from"./entry-preview-D70TdSbK.js";import{a as v,n as y,o as b,r as x,t as S}from"./climate-stack-CSQImhLZ.js";import{a as C,i as w,n as T,t as E}from"./panel-lHgDcAzn.js";var D=`
fn getNormalQuantileValue(probability: f32) -> f32 {
  if (probability <= 0.0) { return -1.0 / 0.000000000000001; }
  if (probability >= 1.0) { return 1.0 / 0.000000000000001; }
  let centered = probability - 0.5;
  if (abs(centered) <= 0.425) {
    let radius = 0.180625 - centered * centered;
    return centered * (((((((radius * 2509.0809287301226727 + 33430.575583588128105) * radius + 67265.770927008700853) * radius + 45921.953931549871457) * radius + 13731.693765509461125) * radius + 1971.5909503065514427) * radius + 133.14166789178437745) * radius + 3.387132872796366608)
      / (((((((radius * 5226.495278852854561 + 28729.085735721942674) * radius + 39307.89580009271061) * radius + 21213.794301586595867) * radius + 5394.1960214247511077) * radius + 687.1870074920579083) * radius + 42.313330701600911252) * radius + 1.0);
  }
  let tail = select(1.0 - probability, probability, centered < 0.0);
  var radius = sqrt(-log(tail));
  var value: f32;
  if (radius <= 5.0) {
    radius -= 1.6;
    value = (((((((radius * 7.7454501427834140764e-4 + 0.0227238449892691845833) * radius + 0.24178072517745061177) * radius + 1.27045825245236838258) * radius + 3.64784832476320460504) * radius + 5.7694972214606914055) * radius + 4.6303378461565452959) * radius + 1.42343711074968357734)
      / (((((((radius * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * radius + 0.0151986665636164571966) * radius + 0.14810397642748007459) * radius + 0.68976733498510000455) * radius + 1.6763848301838038494) * radius + 2.05319162663775882187) * radius + 1.0);
  } else {
    radius -= 5.0;
    value = (((((((radius * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * radius + 0.0012426609473880784386) * radius + 0.026532189526576123093) * radius + 0.29656057182850489123) * radius + 1.7848265399172913358) * radius + 5.4637849111641143699) * radius + 6.6579046435011037772)
      / (((((((radius * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * radius + 1.8463183175100546818e-5) * radius + 7.868691311456132591e-4) * radius + 0.0148753612908506148525) * radius + 0.13692988092273580531) * radius + 0.59983220655588793769) * radius + 1.0);
  }
  return select(value, -value, centered < 0.0);
}
`,O=256,k=`${u}
${l}
${d}
fn getRegularizedGammaTails(a: f32, x: f32) -> vec2<f32> {
  if (!(x > 0.0)) { return vec2<f32>(0.0, 1.0); }
  let logPrefactor = -x + a * log(x) - getLogGammaValue(a);
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    for (var iteration = 1u; iteration <= ${O}u; iteration++) {
      term = term * x / (a + f32(iteration));
      sum = sum + term;
      if (abs(term) < abs(sum) * 1e-7) { break; }
    }
    let lower = clamp(sum * exp(logPrefactor), 0.0, 1.0);
    return vec2<f32>(lower, 1.0 - lower);
  }
  let tiny = 1e-30;
  var b = x + 1.0 - a;
  var c = 1.0 / tiny;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1u; iteration <= ${O}u; iteration++) {
    let stepIndex = f32(iteration);
    let an = -stepIndex * (stepIndex - a);
    b += 2.0;
    d = an * d + b;
    if (abs(d) < tiny) { d = tiny; }
    c = b + an / c;
    if (abs(c) < tiny) { c = tiny; }
    d = 1.0 / d;
    let delta = d * c;
    h *= delta;
    if (abs(delta - 1.0) < 2e-7) { break; }
  }
  let upper = clamp(exp(logPrefactor) * h, 0.0, 1.0);
  return vec2<f32>(1.0 - upper, upper);
}
`,A=2**24,j=8.21;function M(e,t,n=new Float32Array(2)){return n[0]=e.activeSliceCount??t,n[1]=e.location??0,n}var N=class{id;props;chunkPlan;estimatedCommandNodeCount;slotCount;constructor(t){this.props=t;let r=t.id??`raster-standardized-index`;this.id=r;let i=t.slotCount??1;this.slotCount=i,o(r,`cellCount`,t.cellCount),o(r,`chunkSliceCount`,t.chunkSliceCount,1,A),o(r,`slotCount`,i,1,366),a(r,t.distribution===`gamma`||t.distribution===`fisk`,`distribution must be 'gamma' or 'fisk'`);let s=t.cellCount*t.chunkSliceCount;o(r,`cellCount times chunkSliceCount`,s),a(r,t.values.length>=s&&(!t.mask||t.mask.length>=s)&&t.output.index.length>=s&&t.output.status.length>=s,`values, mask, output.index and output.status must hold cellCount times chunkSliceCount entries`),a(r,t.fit.length>=t.cellCount*i*3,`fit must hold three entries per slot`),a(r,i===1||!!t.slotIndices&&t.slotIndices.length>=t.chunkSliceCount,`slotIndices must hold one entry per chunk slice when slotCount exceeds 1`),a(r,t.parameters.length>=2,`parameters must hold 2 entries`),this.chunkPlan=n(this.id,t.cellCount,{values:{view:t.values,entriesPerCell:t.chunkSliceCount},mask:{view:t.mask,entriesPerCell:t.chunkSliceCount},fit:{view:t.fit,entriesPerCell:(t.slotCount??1)*3},"output.index":{view:t.output.index,entriesPerCell:t.chunkSliceCount},"output.status":{view:t.output.status,entriesPerCell:t.chunkSliceCount}}),this.estimatedCommandNodeCount=this.chunkPlan.estimatedCommandNodeCount,e(this.id,this.getInputs(),this.getOutputs(),this.chunkPlan)}getInputs(){let{values:e,mask:t,slotIndices:n,fit:r,parameters:i}=this.props;return[e,r,i,...t?[t]:[],...(this.props.slotCount??1)>1&&n?[n]:[]]}getOutputs(){return[this.props.output.index,this.props.output.status]}getCommandNodes(e){let{props:n,id:a,slotCount:o}=this,{chunkSliceCount:l,values:u,mask:d,slotIndices:f,fit:p,parameters:m,output:h}=n,g=n.zeroInflated??!1;t(a,e,this.getInputs(),this.getOutputs());let _=n.distribution===`gamma`?`getRegularizedGammaTails(fitShape, getScaledQuotient(positive, fitScale, 1.0, 0))`:`getFiskTails(fitShape, getScaledQuotient(positive, fitScale, 1.0, 0))`;return i(e,this.chunkPlan,{id:a,operation:`GPURasterStandardizedIndex`,invocationCount:n.cellCount*l,workgroupSize:64,bindings:[r(`values`,u),r(`fit`,p),r(`parameters`,m),...d?[r(`mask`,d)]:[],...o>1&&f?[r(`slotIndices`,f)]:[],r(`indexOut`,h.index,!0),r(`statusOut`,h.status,!0)],declarations:`${c}
${s}
${n.distribution===`gamma`?k:``}
${D}
fn getFiskTails(shape: f32, ratio: f32) -> vec2<f32> {
  if (!(ratio > 0.0)) { return vec2<f32>(0.0, 1.0); }
  let logistic = shape * log(ratio);
  return vec2<f32>(1.0 / (1.0 + exp(-logistic)), 1.0 / (1.0 + exp(logistic)));
}`,body:`let cell = index / ${l}u;
let slice = index % ${l}u;
let source = cell * ${l}u + slice;
var result = missing();
var code = 1u;
let activeSliceCount = u32(clamp(parameters[parametersOffset], 0.0, ${l}.0));
let value = values[valuesOffset + source];
if (slice < activeSliceCount && finite(value) ${d?`&& mask[maskOffset + source] != 0u`:``}) {
  let slot = ${o>1?`slotIndices[slotIndicesOffset + slice]`:`0u`};
  code = 2u;
  if (slot < ${o}u) {
    let entry = cell * ${o}u + slot;
    let fitShape = fit[fitOffset + 3u * entry];
    let fitScale = fit[fitOffset + 3u * entry + 1u];
    let zeroProbability = fit[fitOffset + 3u * entry + 2u];
    if (finite(fitShape) && finite(fitScale) && fitShape > 0.0 && fitScale > 0.0) {
      code = 0u;
      var lowerTail = zeroProbability;
      var upperTail = 1.0 - zeroProbability;
      ${g?`if (value != 0.0) {
        let positive = value - parameters[parametersOffset + 1u];
        let tails = ${_};
        lowerTail = zeroProbability + (1.0 - zeroProbability) * tails.x;
        upperTail = (1.0 - zeroProbability) * tails.y;
      }`:`let positive = value - parameters[parametersOffset + 1u];
      let tails = ${_};
      lowerTail = tails.x;
      upperTail = tails.y;`}
      result = select(getNormalQuantileValue(lowerTail), -getNormalQuantileValue(upperTail), upperTail < lowerTail);
      result = clamp(result, -${j}, ${j});
    }
  }
}
indexOut[indexOutOffset + index] = result;
statusOut[statusOutOffset + index] = code;`})}},P=150,F=300,I=60,L=12,R=[`gamma`,`fisk`],z=[.03,.03,.97,.22],B=[.06,.06,.94,.2],V=[-3,3],H=m({parameters:_,rebuildOn:[`distribution`],buildCache:2,caption:`Synthetic 30-year rainfall, Helene-interpolated wetness, in; standard deviations out.`,createInputs:({random:e,world:t})=>({...x(e,t),panel:E(t,[z])}),build(e,t,n){let{cellCount:r}=t,i=r*L,a=R[Math.round(n.distribution)],o=e.allocate(`state-counts`,`uint32`,i*3),s=e.allocate(`state-moments`,`uint32`,i*9),c=e.allocate(`fit`,`float32`,i*3);for(let n of[0,1])e.graph.add(new p({id:`calibration-${n}`,cellCount:r,chunkSliceCount:P,slotCount:L,distribution:a,values:e.upload(`values-${n}`,`float32`,v(t,n*P,P)).view,slotIndices:e.upload(`slots-${n}`,`uint32`,b(n*P,P)).view,parameters:e.upload(`calibration-parameters-${n}`,`float32`,f({reset:n===0,maximumValueMagnitude:S},P)).view,state:{counts:o.view,moments:s.view},fit:c.view}));let l=e.allocate(`index`,`float32`,r*I),u=e.allocate(`status`,`uint32`,r*I);return e.graph.add(new N({cellCount:r,chunkSliceCount:I,slotCount:L,distribution:a,values:e.upload(`evaluated`,`float32`,v(t,F,I)).view,slotIndices:e.upload(`evaluated-slots`,`uint32`,b(F,I)).view,fit:c.view,parameters:e.upload(`index-parameters`,`float32`,M({},I)).view,output:{index:l.view,status:u.view}})),{outputs:{index:l,panel:C(e,t.panel)},update:()=>void 0}},async afterRun({inputs:e,outputs:t,values:n,upload:r}){let i=await t.index.buffer.readAsync(),a=new Float32Array(i.buffer,i.byteOffset,e.cellCount*I),o=Math.round(n.month);return{shown:r(`shown`,`float32`,Float32Array.from({length:e.cellCount},(e,t)=>a[t*I+o])),strip:r(`strip`,`float32`,U(a,e.cellCount))}},interaction:h(_,{kind:`time`,hint:`scrub the month`,x:`month`,decorate:(e,t,{inputs:n,outputs:r,peek:i,world:a})=>{let o=Math.round(e.month),s=i(r.index),c=``;if(s){let e=U(s,n.cellCount),t=0;for(let e=0;e<n.cellCount;e++)s[e*I+o]<-1.5&&t++;c=` · area mean ${g(e[o],2)} · ${g(100*t/n.cellCount,0)}% below -1.5`}let l=a.width*(B[0]+(o+.5)/I*(B[2]-B[0]));return{glyphs:[{shape:`line`,from:[l,a.height*B[1]],to:[l,a.height*B[3]],dashed:!1}],readout:`${y[(F+o)%12]} of year ${Math.floor((F+o)/12)+1}${c}`}}}),draw:({inputs:e,outputs:t,extra:n,theme:r})=>[{layer:`raster`,values:n.shown,columns:e.columns,rows:e.rows,ramp:`diverging`,range:[-3,3]},...T(t.panel,r),{layer:`chart`,kind:`bars`,values:n.strip,range:V,rectangle:B,color:w(r.categories[0])}]});function U(e,t){let n=new Float32Array(I);for(let r=0;r<I;r++){let i=0,a=0;for(let n=0;n<t;n++){let t=e[n*I+r];Number.isFinite(t)&&(i+=t,a++)}n[r]=a?i/a:0}return n}export{H as default};
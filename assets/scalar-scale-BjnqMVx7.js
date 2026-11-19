import{t as e}from"./classification-kernel-node-C62eCEbJ.js";import{E as t,T as n,c as r,f as i,p as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{u as s}from"./wgsl-common-DfyXiYB0.js";import{i as c,n as l,r as u,t as d}from"./column-classification-rows-D2hiB8L9.js";import{t as f}from"./define-run-B-brkCe4.js";import{a as p}from"./pointer-overlay-Qh8qflcA.js";import{t as m}from"./lightning-C0gvbE9a.js";import{hc as h}from"./entry-preview-D70TdSbK.js";var g=`GPUScalarScale`,_={linear:0,sqrt:1,log:2,identity:3,point:4};function v(e,t=new Float32Array(4)){n(g,t.length>=4,`parameter target must hold at least 4 values`),n(g,Object.hasOwn(_,e.scale),`scale must be linear, sqrt, log, identity or point`);let r=e.domainCount;n(g,Number.isInteger(r)&&r>=0&&r<=256&&(e.scale===`point`||r===2),`domainCount must be an integer from 0 to 256, and exactly 2 unless scale is point`);let i=e.nullValue??0;return n(g,Number.isFinite(Math.fround(i)),`nullValue must be finite as float32`),n(g,!e.invert||e.scale===`identity`,`invert is accepted only for the identity scale`),t[0]=_[e.scale],t[1]=r,t[2]=i,t[3]=Number(e.invert??!1),t}var y=`
${s}
fn scalarTransform(value: f32, scale: u32) -> f32 {
  if (scale == 1u) {
    return select(sqrt(abs(value)), -sqrt(abs(value)), value < 0.0);
  }
  if (scale == 2u) {
    // For negative log domains d3 uses -log(-x); the sign cancels in normalization.
    return log(abs(value));
  }
  return value;
}
fn scalarInterpolate(first: f32, second: f32, position: f32) -> f32 {
  if (position == 0.0) { return first; }
  if (position == 1.0) { return second; }
  // Same convex interpolation convention as d3; unlike first+(second-first)*t this
  // remains finite for opposite large endpoints when the result is in range.
  return first * (1.0 - position) + second * position;
}
`,b=class{id;props;estimatedCommandNodeCount;mask;constructor(e){this.id=e.id??`scalar-scale`,this.props=e,this.mask=e.mask??e.validity;let s=this.id;i(s,`maximumDomainCount`,e.maximumDomainCount,2,256),c(s,`values`,e.values,[`float32`,`uint32`]),t(e.domain,[`float32`],`${s} domain`),t(e.range,[`float32`],`${s} range`),t(e.parameters,[`float32`],`${s} parameters`),l(s,`output.values`,e.output.values,`float32`,e.values),n(s,e.domain.length>=e.maximumDomainCount,`domain must hold at least maximumDomainCount entries`),n(s,e.range.length>=2,`range must hold at least 2 entries`),n(s,e.parameters.length>=4,`parameters must hold at least 4 values`),n(s,e.output.values.length>=e.values.length,`output.values must hold at least one value per source row`),this.mask&&(c(s,`mask`,this.mask,[`uint32`],e.values),n(s,this.mask.length===e.values.length,`mask must hold exactly one entry per source row`)),e.output.validity&&(l(s,`output.validity`,e.output.validity,`uint32`,e.values),n(s,e.output.validity.length>=e.values.length,`output.validity must hold at least one value per source row`));let d=[e.output.values,e.output.validity];u(s,d),o(s,d,[e.values,this.mask,e.domain,e.range,e.parameters]),this.estimatedCommandNodeCount=r(e.values).filter(e=>e.length>0).length,a(s,this.estimatedCommandNodeCount)}getCommandNodes(t){let{props:n,id:i}=this;d(i,t,{values:n.values,mask:this.mask,domain:n.domain,range:n.range,parameters:n.parameters,"output.values":n.output.values,"output.validity":n.output.validity});let a=[],o=r(n.values);for(let n=0;n<o.length;n++){let s=o[n];if(s.length===0)continue;let c={...this.props,values:s,output:{values:r(this.props.output.values)[n],validity:this.props.output.validity?r(this.props.output.validity)[n]:void 0}},l=this.mask?r(this.mask)[n]:void 0,u=o.length===1?i:`${i}-chunk-${n}`,d=(e,t,n)=>({name:e,view:t,type:n,access:`read`});a.push(e(t,{id:`${u}-map`,operation:`GPUScalarScale`,variant:`finite-float32`,invocationCount:c.values.length,bindings:[d(`values`,c.values,c.values.format===`uint32`?`u32`:`f32`),...l?[d(`sourceMask`,l,`u32`)]:[],d(`domain`,c.domain,`f32`),d(`range`,c.range,`f32`),d(`parameters`,c.parameters,`f32`),{name:`outputValues`,view:c.output.values,type:`f32`,access:`read_write`},...c.output.validity?[{name:`outputValidity`,view:c.output.validity,type:`u32`,access:`read_write`}]:[]],declarations:`const MAXIMUM_DOMAIN_COUNT: u32 = ${c.maximumDomainCount}u;\n${y}`,body:`let nullValue = parameters[parametersOffset + 2u];
  outputValues[outputValuesOffset + index] = select(0.0, nullValue, isFiniteValue(nullValue));
  ${c.output.validity?`outputValidity[outputValidityOffset + index] = 0u;`:``}
  ${l?`if (sourceMask[sourceMaskOffset + index] == 0u) { return; }`:``}
  let scaleParameter = parameters[parametersOffset];
  let domainCountParameter = parameters[parametersOffset + 1u];
  if (!isFiniteValue(scaleParameter) || !isFiniteValue(domainCountParameter) || !isFiniteValue(nullValue)) { return; }
  if (scaleParameter < 0.0 || scaleParameter > 4.0 || floor(scaleParameter) != scaleParameter || domainCountParameter < 0.0 || domainCountParameter > f32(MAXIMUM_DOMAIN_COUNT) || floor(domainCountParameter) != domainCountParameter) { return; }
  let invertParameter = parameters[parametersOffset + 3u];
  if (invertParameter != 0.0 && invertParameter != 1.0) { return; }
  if (invertParameter == 1.0 && scaleParameter != 3.0) { return; }
  let scale = u32(scaleParameter);
  let domainCount = u32(domainCountParameter);
  ${c.values.format===`uint32`?`if (scale != 4u || values[valuesOffset + index] > 16777216u) { return; }
  let value = f32(values[valuesOffset + index]);`:`let value = values[valuesOffset + index];`}
  if (!isFiniteValue(value)) { return; }
  var mapped: f32;
  if (scale == 4u) {
    if (domainCount == 0u) { return; }
    let firstRange = range[rangeOffset];
    let secondRange = range[rangeOffset + 1u];
    if (!isFiniteValue(firstRange) || !isFiniteValue(secondRange)) { return; }
    var categoryIndex = MAXIMUM_DOMAIN_COUNT;
    for (var candidate = 0u; candidate < domainCount; candidate++) {
      let category = domain[domainOffset + candidate];
      if (!isFiniteValue(category)) { return; }
      if (category == value && categoryIndex == MAXIMUM_DOMAIN_COUNT) { categoryIndex = candidate; }
    }
    if (categoryIndex == MAXIMUM_DOMAIN_COUNT) { return; }
    let position = select(0.5, f32(categoryIndex) / f32(max(domainCount - 1u, 1u)), domainCount > 1u);
    mapped = scalarInterpolate(firstRange, secondRange, position);
  } else {
    if (domainCount != 2u) { return; }
    let firstDomain = domain[domainOffset];
    let secondDomain = domain[domainOffset + 1u];
    if (!isFiniteValue(firstDomain) || !isFiniteValue(secondDomain)) { return; }
    if (scale == 2u && !((firstDomain > 0.0 && secondDomain > 0.0) || (firstDomain < 0.0 && secondDomain < 0.0))) { return; }
    if (scale == 3u && invertParameter == 0.0) {
      mapped = select(value, firstDomain, firstDomain == secondDomain);
    } else {
      let firstRange = select(range[rangeOffset], secondDomain, scale == 3u);
      let secondRange = select(range[rangeOffset + 1u], firstDomain, scale == 3u);
      if (!isFiniteValue(firstRange) || !isFiniteValue(secondRange)) { return; }
      if (firstDomain == secondDomain) {
        mapped = select(scalarInterpolate(firstRange, secondRange, 0.5), firstDomain, scale == 3u);
      } else {
        if (scale == 2u && ((value <= 0.0 && firstDomain > 0.0) || (value >= 0.0 && firstDomain < 0.0))) { return; }
        let first = scalarTransform(firstDomain, scale);
        let second = scalarTransform(secondDomain, scale);
        let transformed = scalarTransform(value, scale);
        if (!isFiniteValue(first) || !isFiniteValue(second) || !isFiniteValue(transformed) || first == second) { return; }
        // Normalize large magnitudes before dividing: hardware may flush the reciprocal of
        // a near-maximum f32 denominator to zero even when the quotient is ordinary finite.
        let magnitude = max(max(abs(first), abs(second)), abs(transformed));
        let normalization = select(1.0, 1e-20, magnitude > 1e19);
        let scaledFirst = first * normalization;
        let scaledSecond = second * normalization;
        if (scaledFirst == scaledSecond) { return; }
        let position = (transformed * normalization - scaledFirst) / (scaledSecond - scaledFirst);
        if (!isFiniteValue(position)) { return; }
        if (scale == 3u) {
          // Keep both interpolation operands normalized too: GPU algebraic optimization may
          // form their difference, which overflows for opposite near-maximum f32 endpoints.
          mapped = scalarInterpolate(second * normalization, first * normalization, position) / normalization;
        } else {
          mapped = scalarInterpolate(firstRange, secondRange, position);
        }
      }
    }
  }
  if (!isFiniteValue(mapped)) { return; }
  outputValues[outputValuesOffset + index] = mapped;
  ${c.output.validity?`outputValidity[outputValidityOffset + index] = 1u;`:``}`}))}return a}},x=1200,S=[`linear`,`sqrt`,`log`],C=[.08,.1,.94,.9],w=80,T=30,E=f({parameters:h,caption:`Measured flash energy in units of 10 fJ (x) and the scaled width (y).`,createInputs:({world:e})=>Float32Array.from(m(e,x).energy,e=>e/10),build(e,t){let n=e.upload(`values`,`float32`,t),r=e.allocate(`domain`,`float32`,2),i=e.allocate(`range`,`float32`,2),a=e.allocate(`parameters`,`float32`,4),o=e.allocate(`widths`,`float32`,x);return e.graph.add(new b({values:n.view,domain:r.view,range:i.view,parameters:a.view,maximumDomainCount:2,output:{values:o.view}})),{outputs:{input:n,widths:o},update:e=>{r.write(Float32Array.of(1,e.domainMaximum)),i.write(Float32Array.of(1,e.rangeMaximum)),a.write(v({scale:S[e.scale],domainCount:2}))}}},rebuildOn:[],interaction:p(h,{kind:`pad`,hint:`move the clamp knee`,x:`domainMaximum`,y:`rangeMaximum`,decorate:(e,t,{world:n})=>({glyphs:[{shape:`crosshair`,at:[(C[0]+e.domainMaximum/w*(C[2]-C[0]))*n.width,(C[1]+e.rangeMaximum/T*(C[3]-C[1]))*n.height],size:10}],readout:`${S[e.scale]} · domain 1-${e.domainMaximum} · range 1-${e.rangeMaximum}`})}),draw:({outputs:e})=>[{layer:`chart`,kind:`scatter`,context:!0,values:e.input,xValues:e.input,xRange:[0,w],radius:2,range:[0,T],rectangle:C},{layer:`chart`,kind:`scatter`,values:e.widths,xValues:e.input,xRange:[0,w],range:[0,T],rectangle:C,radius:7,color:[.2,.7,1,1]}]});export{E as default};
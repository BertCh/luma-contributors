import{a as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t}from"./construction-snapshot-DdkX9Pc0.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,T as i,f as a}from"./gpu-contributor-utils-BIwLQON5.js";import{a as o,s,t as c}from"./raster-kernel-utils-DcIN7G5I.js";import{b as l,o as u,r as d,v as f,x as p,y as m}from"./embedding-DKQ2dOol.js";import{t as h}from"./define-run-B-brkCe4.js";import{c as g}from"./pointer-overlay-Qh8qflcA.js";import{t as _}from"./kernels-iXfy7fsH.js";import{$r as v}from"./entry-preview-D70TdSbK.js";var y=class{id;width;height;horizontalScale;verticalScale;props;constructor(e){e=Object.freeze({...t(e,[`id`,`sourceWidth`,`sourceHeight`,`dimensions`,`scale`,`vectors`,`quantized`,`aggregation`,`mask`,`output`]),scale:typeof e.scale==`number`?e.scale:Object.freeze([...e.scale]),aggregation:e.aggregation&&t(e.aggregation,[`vectorSums`,`validCount`]),output:t(e.output,[`vectors`,`vectorSums`,`validity`,`validCount`,`meanLength`])}),this.props=e;let n=e.id??`raster-embedding-pyramid-level`;this.id=n;let{sourceWidth:o,sourceHeight:c,dimensions:u,output:d}=e;a(n,`sourceWidth`,o),a(n,`sourceHeight`,c);let[f,h]=typeof e.scale==`number`?[e.scale,e.scale]:e.scale;a(n,`horizontal scale`,f,1,16),a(n,`vertical scale`,h,1,16),this.horizontalScale=f,this.verticalScale=h,this.width=Math.ceil(o/f),this.height=Math.ceil(c/h);let g=o*c,_=this.width*this.height;l(n,g,u),i(n,Number(e.vectors!==void 0)+Number(e.quantized!==void 0)+Number(e.aggregation!==void 0)===1,`needs exactly one of vectors, quantized, and aggregation`),e.vectors?p(n,`vectors`,e.vectors,g,u,e.mask):e.aggregation?(p(n,`aggregation.vectorSums`,e.aggregation.vectorSums,g,u,e.mask),r(e.aggregation.validCount,[`uint32`],`${n} aggregation.validCount`),i(n,e.aggregation.validCount.length===g,`aggregation.validCount must hold sourceWidth times sourceHeight entries`)):(m(n,`quantized`,e.quantized,g,u),i(n,!e.mask||e.mask.length===g,`mask must hold sourceWidth times sourceHeight entries`)),p(n,`output.vectors`,d.vectors,_,u),d.vectorSums&&p(n,`output.vectorSums`,d.vectorSums,_,u),i(n,(!d.validity||d.validity.length===_)&&(!d.validCount||d.validCount.length===_)&&(!d.meanLength||d.meanLength.length===_),`overview outputs must hold ceil(sourceWidth / scale) times ceil(sourceHeight / scale) entries`);let v=this.getInputs(),y=this.getOutputs();s(n,v,y,[{inputs:v,outputs:y}])}getInputs(){let{vectors:e,quantized:t,aggregation:n,mask:r}=this.props;return[...e?[e]:[],...t?[t]:[],...n?[n.vectorSums,n.validCount]:[],...r?[r]:[]]}getOutputs(){let{vectors:e,vectorSums:t,validity:n,validCount:r,meanLength:i}=this.props.output;return[e,...t?[t]:[],...n?[n]:[],...r?[r]:[],...i?[i]:[]]}getCommandNodes(t){let{id:r,props:i,width:a,height:s,horizontalScale:l,verticalScale:u}=this,{sourceWidth:d,sourceHeight:p,dimensions:m,output:h}=i;e(t,r,{vectors:i.vectors,quantized:i.quantized,"aggregation.vectorSums":i.aggregation?.vectorSums,"aggregation.validCount":i.aggregation?.validCount,mask:i.mask,"output.vectors":h.vectors,"output.vectorSums":h.vectorSums,"output.validity":h.validity,"output.validCount":h.validCount,"output.meanLength":h.meanLength});let g=i.vectors??i.aggregation?.vectorSums??i.quantized,_=i.vectors?`vectors`:i.aggregation?`sourceSums`:`quantized`,v=i.vectors||i.aggregation?`let value = ${_}[${_}Offset + pixel * ${m}u + band];
        if (!finite(value)) { pixelValid = false; }`:`let byte = readEmbeddingByte(pixel * ${m}u + band);
        if (byte == -128) { pixelValid = false; }
        let value = getEmbeddingValue(byte);`;return[n(t,{id:r,operation:`GPURasterEmbeddingPyramidLevel`,variant:`unit-vector-mean`,invocationCount:a*s,workgroupSize:64,bindings:[o(_,g),...i.aggregation?[o(`sourceValidCount`,i.aggregation.validCount)]:[],...i.mask?[o(`mask`,i.mask)]:[],o(`overview`,h.vectors,!0),...h.vectorSums?[o(`vectorSums`,h.vectorSums,!0)]:[],...h.validity?[o(`overviewValidity`,h.validity,!0)]:[],...h.validCount?[o(`validCount`,h.validCount,!0)]:[],...h.meanLength?[o(`meanLength`,h.meanLength,!0)]:[]],declarations:`${c}${i.quantized?f(`quantized`):``}`,body:`let outColumn = index % ${a}u;
let outRow = index / ${a}u;
var total: array<f32, ${m}>;
var validPixels = 0u;
for (var row = outRow * ${u}u; row < min((outRow + 1u) * ${u}u, ${p}u); row++) {
  for (var column = outColumn * ${l}u; column < min((outColumn + 1u) * ${l}u, ${d}u); column++) {
    let pixel = row * ${d}u + column;
    let pixelCount = ${i.aggregation?`sourceValidCount[sourceValidCountOffset + pixel]`:`1u`};
    var pixelValid = ${i.mask?`mask[maskOffset + pixel] != 0u`:`true`};
    var decoded: array<f32, ${m}>;
    for (var band = 0u; band < ${m}u; band++) {
      ${v}
      decoded[band] = value;
    }
    if (pixelValid && pixelCount > 0u) {
      for (var band = 0u; band < ${m}u; band++) { total[band] += decoded[band]; }
      validPixels += pixelCount;
    }
  }
}
var largestComponent = 0.0;
for (var band = 0u; band < ${m}u; band++) {
  ${h.vectorSums?`vectorSums[vectorSumsOffset + index * ${m}u + band] = total[band];`:``}
  largestComponent = max(largestComponent, abs(total[band]));
}
let componentScale = select(1.0, largestComponent, largestComponent > 0.0);
var squaredLength = 0.0;
for (var band = 0u; band < ${m}u; band++) {
  let scaled = total[band] / componentScale;
  squaredLength += scaled * scaled;
}
let scaledLength = sqrt(squaredLength);
let meanLengthValue = largestComponent * (scaledLength / f32(max(validPixels, 1u)));
let isValid = validPixels > 0u && largestComponent > 0.0 && finite(scaledLength);
for (var band = 0u; band < ${m}u; band++) {
  overview[overviewOffset + index * ${m}u + band] = select(missing(), (total[band] / componentScale) / max(scaledLength, 1.0), isValid);
}
${h.validity?`overviewValidity[overviewValidityOffset + index] = select(0u, 1u, isValid);`:``}
${h.validCount?`validCount[validCountOffset + index] = validPixels;`:``}
${h.meanLength?`meanLength[meanLengthOffset + index] = select(missing(), meanLengthValue, validPixels > 0u);`:``}`})]}},b=[1,2,4,8],x=h({parameters:v,rebuildOn:[`scale`],buildCache:5,caption:`Overview pixels average the valid vectors under them, then renormalize; dark marks a mixed footprint. 10% of source pixels are set invalid by default. Six Sentinel-2 bands stand in for a learned embedding.`,createInputs:({aspect:e,random:t})=>{let n=d(e,48,8);return{...n,order:Float32Array.from({length:n.pixelCount},()=>t())}},build(e,t,n){let{scale:r,width:i,height:a}=S(t,n),o=i*a,s=e.allocate(`validity`,`uint32`,t.pixelCount),c=e.allocate(`overview-vectors`,`float32`,o*6),l=e.allocate(`valid-count`,`uint32`,o),u=e.allocate(`mean-length`,`float32`,o);e.graph.add(new y({sourceWidth:t.columns,sourceHeight:t.rows,dimensions:6,scale:r,vectors:e.upload(`vectors`,`float32`,t.vectors).view,mask:s.view,output:{vectors:c.view,validCount:l.view,meanLength:u.view}}));let d=e.upload(`component`,`uint32`,Uint32Array.of(3)),f=e.allocate(`band`,`float32`,o);return _(e,`band-extract`,c,6,d,f),{outputs:{band:f,meanLength:u,validCount:l},update:e=>s.write(Uint32Array.from(t.order,t=>t<e.noData?0:1))}},interaction:{kind:`probe`,hint:`probe a footprint`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let{width:o,height:s,scale:c}=S(t,r),{index:l,column:d,row:f}=u({columns:o,rows:s},i,e.x,e.y),p=i.width/o,m=i.height/s,h=a(n.validCount)?.[l],_=a(n.meanLength)?.[l];return{spotlight:null,glyphs:[{shape:`box`,minimum:[d*p,i.height-(f+1)*m],maximum:[(d+1)*p,i.height-f*m]}],readout:`validCount ${h??`-`} of ${c*c} | meanLength ${_===void 0?`-`:g(_,3)}`}}},draw:({inputs:e,outputs:t,values:n})=>{let{width:r,height:i}=S(e,n);return[n.shown===0?{layer:`raster`,values:t.band,columns:r,rows:i,ramp:`diverging`,range:[-1,1]}:{layer:`raster`,values:t.meanLength,columns:r,rows:i,ramp:`sequential`,range:[.3,1]}]}});function S(e,t){let n=b[Math.round(t.scale)];return{scale:n,width:Math.ceil(e.columns/n),height:Math.ceil(e.rows/n)}}export{x as default};
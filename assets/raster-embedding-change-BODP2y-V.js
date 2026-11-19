import{r as e}from"./kernel-C3tfQOn4.js";import{s as t}from"./wgsl-common-DfyXiYB0.js";import{E as n,T as r,g as i,v as a}from"./gpu-contributor-utils-BIwLQON5.js";import{t as o}from"./scaled-quotient-wgsl-UDyhQag6.js";import{a as s,t as c}from"./raster-kernel-utils-DcIN7G5I.js";import{a as l,b as u,c as d,l as f,o as p,p as m,r as h,s as g,u as _,x as v}from"./embedding-DKQ2dOol.js";import{a as y,i as b,n as x,r as S,t as C}from"./raster-embedding-chunks-w-sv9jw0.js";import{t as w}from"./define-run-B-brkCe4.js";import{si as T}from"./entry-preview-D70TdSbK.js";function E(e,t=new Float32Array(1)){return i(`getGPURasterEmbeddingChangeParameterValues`,`similarityThreshold`,e.similarityThreshold,-1,1),t[0]=e.similarityThreshold,t}var D=class{props;id;chunks;estimatedCommandNodeCount;constructor(e){this.props=e;let t=e.id??`raster-embedding-change`;this.id=t;let{pixelCount:i,dimensions:a,output:o}=e;u(t,i,a,m,!1),v(t,`before`,e.before,i,a,e.beforeMask),v(t,`after`,e.after,i,a,e.afterMask),r(t,!!(o.similarity||o.angle||o.magnitude||o.changed),`output needs at least one view`),r(t,this.getOutputs().every(e=>e.length===i),`outputs must hold pixelCount entries`),r(t,!o.changed||e.parameters!==void 0&&e.parameters.length>=1,`output.changed needs parameters`),e.parameters&&n(e.parameters,[`float32`],`${t} parameters`),y(t,this.getInputs(),this.getOutputs(),[{inputs:[e.before,e.after,e.beforeMask,e.afterMask].filter(e=>e!==void 0),outputs:[o.similarity,o.angle,o.magnitude].filter(e=>e!==void 0)},...o.changed?[{inputs:[e.parameters],outputs:[o.similarity,o.changed].filter(e=>e!==void 0)}]:[]]),this.chunks=S(t,i,{before:{view:e.before,format:`float32`,components:a},after:{view:e.after,format:`float32`,components:a},beforeMask:{view:e.beforeMask,format:`uint32`,components:1},afterMask:{view:e.afterMask,format:`uint32`,components:1},"output.similarity":{view:e.output.similarity,format:`float32`,components:1},"output.angle":{view:e.output.angle,format:`float32`,components:1},"output.magnitude":{view:e.output.magnitude,format:`float32`,components:1},"output.changed":{view:e.output.changed,format:`uint32`,components:1}},`before`,o.changed?2:1),this.estimatedCommandNodeCount=1+this.chunks.filter(e=>e.pixelCount>0).length*(o.changed?2:1)}getInputs(){let{before:e,after:t,beforeMask:n,afterMask:r,parameters:i}=this.props;return[e,t,...n?[n]:[],...r?[r]:[],...i&&this.props.output.changed?[i]:[]]}getOutputs(){let{similarity:e,angle:t,magnitude:n,changed:r}=this.props.output;return[e,t,n,r].filter(e=>e!==void 0)}getCommandNodes(e){if(b(e,this.id,this.getInputs(),this.getOutputs(),this.estimatedCommandNodeCount-1,this.props,2+Number(!!this.props.beforeMask)+Number(!!this.props.afterMask)+Number(!!(this.props.output.similarity||this.props.output.changed))+Number(!!this.props.output.angle)+Number(!!this.props.output.magnitude),this.chunks),!this.props.output.similarity&&this.props.output.changed)for(let t of this.chunks)a(e.device,this.id,`similarity scratch`,Math.max(t.pixelCount,1)*4,`reduce the embedding chunk pixelCount`);let t=this.chunks.flatMap((t,n)=>t.pixelCount===0?[]:this.getChunkKernels(e,x(this.props,t),n));return C(e,this.id,t)}getChunkKernels(n,r,i){let a=`${this.id}-chunk-${i}`,{pixelCount:l,dimensions:u,output:d}=r,f=d.similarity??(d.changed?e(n,`${a}-similarity`,`float32`,l):void 0),p=[{id:a,operation:`GPURasterEmbeddingChange`,invocationCount:l,pixelCount:l,bindings:[s(`before`,r.before),s(`after`,r.after),...r.beforeMask?[s(`beforeMask`,r.beforeMask)]:[],...r.afterMask?[s(`afterMask`,r.afterMask)]:[],...f?[s(`similarity`,f,!0)]:[],...d.angle?[s(`angle`,d.angle,!0)]:[],...d.magnitude?[s(`magnitude`,d.magnitude,!0)]:[]],declarations:`${c}\n${t}\n${o}`,body:`var dotProduct = 0.0;
var beforeLength = 0.0;
var afterLength = 0.0;
var differenceLength = 0.0;
var largestBefore = 0.0;
var largestAfter = 0.0;
var largestDifference = 0.0;
var isValid = ${[r.beforeMask?`beforeMask[beforeMaskOffset + index] != 0u`:``,r.afterMask?`afterMask[afterMaskOffset + index] != 0u`:``].filter(Boolean).join(` && `)||`true`};
for (var band = 0u; band < ${u}u; band++) {
  let left = before[beforeOffset + index * ${u}u + band];
  let right = after[afterOffset + index * ${u}u + band];
  if (!finite(left) || !finite(right)) { isValid = false; }
  largestBefore = max(largestBefore, abs(left));
  largestAfter = max(largestAfter, abs(right));
  largestDifference = max(largestDifference, abs(right - left));
}
let beforeDivisor = select(1.0, largestBefore, largestBefore > 0.0);
let afterDivisor = select(1.0, largestAfter, largestAfter > 0.0);
let differenceDivisor = select(1.0, largestDifference, largestDifference > 0.0 && finite(largestDifference));
for (var band = 0u; band < ${u}u; band++) {
  let beforeValue = before[beforeOffset + index * ${u}u + band];
  let afterValue = after[afterOffset + index * ${u}u + band];
  let left = getScaledQuotient(beforeValue, beforeDivisor, 1.0, 0);
  let right = getScaledQuotient(afterValue, afterDivisor, 1.0, 0);
  dotProduct += left * right;
  beforeLength += left * left;
  afterLength += right * right;
  let difference = getScaledQuotient(afterValue - beforeValue, differenceDivisor, 1.0, 0);
  differenceLength += difference * difference;
}
let denominator = sqrt(beforeLength * afterLength);
let cosine = select(select(0.0, 1.0, beforeLength == 0.0 && afterLength == 0.0), clamp(dotProduct / denominator, -1.0, 1.0), denominator > 0.0);
${f?`similarity[similarityOffset + index] = select(missing(), cosine, isValid);`:``}
${d.angle?`// acos of a float32 cosine cannot resolve angles below about 5e-4 rad (the cosine is within one
// float32 step of 1). The angle is taken from the chord and the sum of the two unit vectors,
// 2 atan2(|u - v|, |u + v|), which keeps its relative accuracy at every angle.
var angleValue = acos(cosine);
if (beforeLength > 0.0 && afterLength > 0.0) {
  var chordSquared = 0.0;
  var sumSquared = 0.0;
  for (var band = 0u; band < ${u}u; band++) {
    let left = getScaledQuotient(before[beforeOffset + index * ${u}u + band], beforeDivisor, sqrt(beforeLength), 0);
    let right = getScaledQuotient(after[afterOffset + index * ${u}u + band], afterDivisor, sqrt(afterLength), 0);
    chordSquared += (left - right) * (left - right);
    sumSquared += (left + right) * (left + right);
  }
  angleValue = 2.0 * guardedAtan2(sqrt(chordSquared), sqrt(sumSquared));
}
angle[angleOffset + index] = select(missing(), angleValue, isValid);`:``}
${d.magnitude?`magnitude[magnitudeOffset + index] = select(missing(), sqrt(differenceLength) * largestDifference, isValid);`:``}`}];return d.changed&&f&&p.push({id:`${a}-threshold`,operation:`GPURasterEmbeddingChange`,variant:`threshold`,invocationCount:l,pixelCount:l,bindings:[s(`similarity`,f,!0),s(`parameters`,r.parameters),s(`changed`,d.changed,!0)],body:`changed[changedOffset + index] = select(0u, 1u, similarity[similarityOffset + index] < parameters[parametersOffset]);`}),p}},O=.28,k=[.4,.45],A=[.43,.47],j=.995,M=[[0,0,0,0],[1,1,1,.45]],N=w({parameters:T,caption:`Point at a pixel: its similarity between years becomes the threshold, and everything that changed as much is marked. The later year is modelled: a clearing scaled by each pixel's vegetation. Six Sentinel-2 bands stand in for a learned embedding.`,createInputs:({aspect:e,random:t})=>{let n=h(e),r={width:n.columns/n.rows,height:1},i=0,a=1/0,o=-1/0;for(let e=0;e<n.pixelCount;e++){let t=n.vegetationIndex[e];t<n.vegetationIndex[i]&&(i=e),a=Math.min(a,t),o=Math.max(o,t)}let s=[k[0]*r.width,1-k[1]],c=Float32Array.from({length:n.pixelCount},(e,t)=>{let[i,c]=d(n,r,t),l=Math.hypot(i-s[0],c-s[1])/O;return Math.min(1,Math.max(0,(1-l)*4))*(n.vegetationIndex[t]-a)/(o-a||1)}),l=Float32Array.from({length:n.vectors.length},()=>t()*2-1),u=Math.floor(A[1]*n.rows)*n.columns+Math.floor(A[0]*n.columns);return{...n,barest:i,noise:l,exposure:c,center:s,levelPixel:u,threshold:.7}},build(e,t){let n=e.upload(`before`,`float32`,t.vectors),r=e.allocate(`after`,`float32`,t.vectors.length),i=e.allocate(`parameters`,`float32`,1),a=e.allocate(`similarity`,`float32`,t.pixelCount),o=e.allocate(`magnitude`,`float32`,t.pixelCount),s=e.allocate(`changed`,`uint32`,t.pixelCount);return e.graph.add(new D({pixelCount:t.pixelCount,dimensions:6,before:n.view,after:r.view,parameters:i.view,output:{similarity:a.view,magnitude:o.view,changed:s.view}})),{outputs:{similarity:a,magnitude:o,changed:s,nearInfrared:e.upload(`near-infrared`,`float32`,t.nearInfrared)},update:e=>{let n=Float32Array.from(t.vectors);for(let r=0;r<t.pixelCount;r++){let i=e.severity*t.exposure[r];for(let a=0;a<6;a++){let o=r*6+a;n[o]=(1-i)*n[o]+i*t.vectors[t.barest*6+a]+e.noise*.5*t.noise[o]}_(n,r)}r.write(n);let a=l(f(t.vectors,t.levelPixel),f(n,t.levelPixel));t.threshold=Math.min(j,Math.ceil((a+5e-4)*1e3)/1e3),i.write(E({similarityThreshold:t.threshold}))}}},interaction:{kind:`level`,hint:`set the threshold`,rest:A,resolve(e,{inputs:t,outputs:n,world:r,peek:i}){let{index:a}=p(t,r,e.x,e.y),o=a!==t.levelPixel;t.levelPixel=a;let s=i(n.changed),c=0;if(s)for(let e of s)c+=e;return{rerun:o,spotlight:null,glyphs:[g(t,r,a),{shape:`ring`,at:t.center,radius:O,dashed:!0}],readout:`similarityThreshold ${t.threshold.toFixed(3)} · ${(100*c/t.pixelCount).toFixed(1)}% changed`}}},draw:({inputs:e,outputs:t})=>[{layer:`raster`,context:!0,values:t.nearInfrared,columns:e.columns,rows:e.rows,ramp:`grey`,smoothing:!0},{layer:`raster`,values:t.magnitude,columns:e.columns,rows:e.rows,ramp:`sequential`,range:[0,2],fade:!0},{layer:`raster`,values:t.changed,columns:e.columns,rows:e.rows,categories:M}]});export{N as default};
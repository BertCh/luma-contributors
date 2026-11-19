import{r as e}from"./kernel-C3tfQOn4.js";import{E as t,T as n,f as r,p as i,v as a}from"./gpu-contributor-utils-BIwLQON5.js";import{t as o}from"./scaled-quotient-wgsl-UDyhQag6.js";import{a as s,t as c}from"./raster-kernel-utils-DcIN7G5I.js";import{b as l,c as u,g as d,i as f,l as p,m,n as h,o as g,p as _,r as v,s as y,x as b}from"./embedding-DKQ2dOol.js";import{a as x,i as S,n as C,r as w,t as T}from"./raster-embedding-chunks-w-sv9jw0.js";import{t as E}from"./define-run-B-brkCe4.js";import{c as D}from"./pointer-overlay-Qh8qflcA.js";import{ri as O}from"./entry-preview-D70TdSbK.js";var k=4096,A=m,j=class{props;id;chunks;estimatedCommandNodeCount;constructor(e){this.props=e;let a=e.id??`raster-embedding-knn-classifier`;this.id=a,n(a,e.metric===void 0||[`euclidean`,`cosine`].includes(e.metric),`metric must be one of euclidean, cosine`);let{pixelCount:o,dimensions:s,output:c}=e;l(a,o,s,_,!1),n(a,e.searchStructure===void 0||[`tiled`,`brute-force`].includes(e.searchStructure),`searchStructure must be tiled or brute-force`),n(a,e.trainingCount*s<=m,`trainingCount times dimensions must stay below 2^32`),r(a,`trainingCount`,e.trainingCount,1,e.searchStructure===`brute-force`?k:A),r(a,`classCount`,e.classCount,1,32),r(a,`neighborCount`,e.neighborCount,1,Math.min(16,e.trainingCount)),b(a,`vectors`,e.vectors,o,s,e.mask),n(a,e.trainingVectors.length===e.trainingCount*s&&e.trainingLabels.length===e.trainingCount,`trainingVectors must hold trainingCount times dimensions entries and trainingLabels trainingCount`),n(a,c.labels.length===o&&(!c.confidence||c.confidence.length===o)&&(!c.probabilities||c.probabilities.length===o*e.classCount),`outputs must hold pixelCount entries (probabilities pixelCount times classCount)`),e.trainingVectors&&t(e.trainingVectors,[`float32`],`${a} trainingVectors`),e.trainingLabels&&t(e.trainingLabels,[`uint32`],`${a} trainingLabels`),x(a,this.getInputs(),this.getOutputs(),e.searchStructure===`brute-force`?void 0:[{inputs:[e.trainingLabels],outputs:this.getOutputs()}]),this.chunks=w(a,o,{vectors:{view:e.vectors,format:`float32`,components:s},mask:{view:e.mask,format:`uint32`,components:1},"output.labels":{view:e.output.labels,format:`uint32`,components:1},"output.confidence":{view:e.output.confidence,format:`float32`,components:1},"output.probabilities":{view:e.output.probabilities,format:`float32`,components:e.classCount}},`vectors`,1);let u=e.searchStructure===`brute-force`?1:Math.ceil(e.trainingCount/256)+1;this.estimatedCommandNodeCount=1+this.chunks.filter(e=>e.pixelCount>0).length*u,i(a,this.estimatedCommandNodeCount)}getInputs(){let{vectors:e,mask:t,trainingVectors:n,trainingLabels:r}=this.props;return[e,n,r,...t?[t]:[]]}getOutputs(){let{labels:e,confidence:t,probabilities:n}=this.props.output;return[e,...t?[t]:[],...n?[n]:[]]}getCommandNodes(e){if(S(e,this.id,this.getInputs(),this.getOutputs(),this.estimatedCommandNodeCount-1,this.props,this.props.searchStructure===`brute-force`?4+Number(!!this.props.mask)+Number(!!this.props.output.confidence)+Number(!!this.props.output.probabilities):Math.max(4+Number(!!this.props.mask),3+Number(!!this.props.output.confidence)+Number(!!this.props.output.probabilities)),this.chunks),this.props.searchStructure!==`brute-force`)for(let t of this.chunks)a(e.device,this.id,`nearest-neighbor scratch`,Math.max(t.pixelCount*this.props.neighborCount,1)*4,`reduce the embedding chunk pixelCount or neighborCount`);let t=this.chunks.flatMap((t,n)=>t.pixelCount===0?[]:this.getChunkKernels(e,C(this.props,t),n));return T(e,this.id,t)}getChunkKernels(e,t,n){let r=`${this.id}-chunk-${n}`,{pixelCount:i,dimensions:a,trainingCount:l,classCount:u,neighborCount:f,output:p}=t,m=t.metric===`cosine`;return t.searchStructure===`brute-force`?[{id:r,operation:`GPURasterEmbeddingKNearestClassifier`,variant:m?`cosine`:`euclidean`,invocationCount:i,pixelCount:i,bindings:[s(`vectors`,t.vectors),s(`trainingVectors`,t.trainingVectors),s(`trainingLabels`,t.trainingLabels),...t.mask?[s(`mask`,t.mask)]:[],s(`labels`,p.labels,!0),...p.confidence?[s(`confidence`,p.confidence,!0)]:[],...p.probabilities?[s(`probabilities`,p.probabilities,!0)]:[]],declarations:`${c}\n${o}`,body:`var squaredLength = 0.0;
var pixelScale = 0.0;
var isValid = ${t.mask?`mask[maskOffset + index] != 0u`:`true`};
for (var band = 0u; band < ${a}u; band++) {
  let value = vectors[vectorsOffset + index * ${a}u + band];
  if (!finite(value)) { isValid = false; }
  ${m?`pixelScale = max(pixelScale, abs(value));`:`squaredLength += value * value;`}
}
${m?`for (var band = 0u; band < ${a}u; band++) {
  let value = getScaledQuotient(vectors[vectorsOffset + index * ${a}u + band], select(1.0, pixelScale, pixelScale > 0.0), 1.0, 0);
  squaredLength += value * value;
}`:``}
if (!isValid) {
  labels[labelsOffset + index] = ${d}u;
  ${p.confidence?`confidence[confidenceOffset + index] = missing();`:``}
  ${p.probabilities?`for (var classIndex = 0u; classIndex < ${u}u; classIndex++) { probabilities[probabilitiesOffset + index * ${u}u + classIndex] = missing(); }`:``}
  return;
}
var nearestDistance: array<f32, ${f}>;
var nearestSample: array<u32, ${f}>;
var filledNeighborCount = 0u;
for (var sampleIndex = 0u; sampleIndex < ${l}u; sampleIndex++) {
  var dotProduct = 0.0;
  var sampleLength = 0.0;
  var isSampleValid = true;
  ${m?`var sampleScale = 0.0;
  for (var band = 0u; band < ${a}u; band++) {
    let component = trainingVectors[trainingVectorsOffset + sampleIndex * ${a}u + band];
    isSampleValid = isSampleValid && finite(component);
    sampleScale = max(sampleScale, abs(component));
  }`:``}
  for (var band = 0u; band < ${a}u; band++) {
    let pixelComponent = ${m?`getScaledQuotient(vectors[vectorsOffset + index * ${a}u + band], select(1.0, pixelScale, pixelScale > 0.0), 1.0, 0)`:`vectors[vectorsOffset + index * ${a}u + band]`};
    let component = ${m?`getScaledQuotient(trainingVectors[trainingVectorsOffset + sampleIndex * ${a}u + band], select(1.0, sampleScale, sampleScale > 0.0), 1.0, 0)`:`trainingVectors[trainingVectorsOffset + sampleIndex * ${a}u + band]`};
    if (!finite(component)) { isSampleValid = false; }
    ${m?`dotProduct += pixelComponent * component;
    sampleLength += component * component;`:`dotProduct += (pixelComponent - component) * (pixelComponent - component);`}
  }
  if (!isSampleValid) { continue; }
  ${m?`let denominator = sqrt(squaredLength * sampleLength);
  // Zero vectors as in the published GPUSimilaritySearch: two zero vectors have similarity 1,
  // exactly one zero vector gives 0.
  let neighborDistance = 1.0 - select(select(0.0, 1.0, squaredLength == 0.0 && sampleLength == 0.0), dotProduct / denominator, denominator > 0.0);`:`let neighborDistance = dotProduct;`}
  if (!finite(neighborDistance)) { continue; }
  // Strict comparison keeps the lower sample index on equal distances.
  if (filledNeighborCount < ${f}u || neighborDistance < nearestDistance[${f-1}u]) {
    var slot = min(filledNeighborCount, ${f-1}u);
    while (slot > 0u && neighborDistance < nearestDistance[slot - 1u]) {
      nearestDistance[slot] = nearestDistance[slot - 1u];
      nearestSample[slot] = nearestSample[slot - 1u];
      slot--;
    }
    nearestDistance[slot] = neighborDistance;
    nearestSample[slot] = sampleIndex;
    filledNeighborCount = min(filledNeighborCount + 1u, ${f}u);
  }
}
if (filledNeighborCount == 0u) {
  labels[labelsOffset + index] = ${d}u;
  ${p.confidence?`confidence[confidenceOffset + index] = missing();`:``}
  ${p.probabilities?`for (var classIndex = 0u; classIndex < ${u}u; classIndex++) { probabilities[probabilitiesOffset + index * ${u}u + classIndex] = missing(); }`:``}
  return;
}
var votes: array<u32, ${u}>;
for (var slot = 0u; slot < filledNeighborCount; slot++) {
  let classIndex = min(trainingLabels[trainingLabelsOffset + nearestSample[slot]], ${u-1}u);
  votes[classIndex] += 1u;
}
var best = 0u;
for (var classIndex = 1u; classIndex < ${u}u; classIndex++) {
  if (votes[classIndex] > votes[best]) { best = classIndex; }
}
labels[labelsOffset + index] = best;
${p.confidence?`confidence[confidenceOffset + index] = f32(votes[best]) / f32(filledNeighborCount);`:``}
${p.probabilities?`for (var classIndex = 0u; classIndex < ${u}u; classIndex++) { probabilities[probabilitiesOffset + index * ${u}u + classIndex] = f32(votes[classIndex]) / f32(filledNeighborCount); }`:``}`}]:this.getTiledKernels(e,t,r)}getTiledKernels(t,n,r){let{pixelCount:i,dimensions:a,trainingCount:l,classCount:u,neighborCount:f,output:p}=n,m=n.metric===`cosine`,h=e(t,`${r}-nearest-samples`,`uint32`,i*f),g=e(t,`${r}-nearest-distances`,`float32`,i*f),_=[];for(let e=0;e<l;e+=256)_.push({id:`${r}-training-${e/256}`,operation:`GPURasterEmbeddingKNearestClassifier`,variant:`tiled-scan`,invocationCount:i,pixelCount:i,tileBase:e,tileCount:Math.min(256,l-e),bindings:[s(`vectors`,n.vectors),s(`trainingVectors`,n.trainingVectors),...n.mask?[s(`mask`,n.mask)]:[],s(`savedSamples`,h,!0),s(`savedDistances`,g,!0)],declarations:`${c}\n${o}`,body:`var squaredLength = 0.0;
var pixelScale = 0.0;
var isValid = ${n.mask?`mask[maskOffset + index] != 0u`:`true`};
for (var band = 0u; band < ${a}u; band++) {
  let value = vectors[vectorsOffset + index * ${a}u + band];
  if (!finite(value)) { isValid = false; }
  ${m?`pixelScale = max(pixelScale, abs(value));`:`squaredLength += value * value;`}
}
${m?`for (var band = 0u; band < ${a}u; band++) {
  let value = getScaledQuotient(vectors[vectorsOffset + index * ${a}u + band], select(1.0, pixelScale, pixelScale > 0.0), 1.0, 0);
  squaredLength += value * value;
}`:``}
var nearestDistance: array<f32, ${f}>;
var nearestSample: array<u32, ${f}>;
var filledNeighborCount = 0u;
for (var slot = 0u; slot < ${f}u; slot++) {
  nearestSample[slot] = 0xffffffffu;
  if (embeddingDispatch.words[0u][2u] != 0u) {
    nearestSample[slot] = savedSamples[savedSamplesOffset + index * ${f}u + slot];
    nearestDistance[slot] = savedDistances[savedDistancesOffset + index * ${f}u + slot];
    if (nearestSample[slot] != 0xffffffffu) { filledNeighborCount++; }
  }
}
for (var tileRow = 0u; tileRow < embeddingDispatch.words[0u][3u]; tileRow++) {
  if (!isValid) { break; }
  let sampleIndex = embeddingDispatch.words[0u][2u] + tileRow;
  var dotProduct = 0.0;
  var sampleLength = 0.0;
  var isSampleValid = true;
  ${m?`var sampleScale = 0.0;
  for (var band = 0u; band < ${a}u; band++) {
    let component = trainingVectors[trainingVectorsOffset + sampleIndex * ${a}u + band];
    isSampleValid = isSampleValid && finite(component);
    sampleScale = max(sampleScale, abs(component));
  }`:``}
  for (var band = 0u; band < ${a}u; band++) {
    let pixelComponent = ${m?`getScaledQuotient(vectors[vectorsOffset + index * ${a}u + band], select(1.0, pixelScale, pixelScale > 0.0), 1.0, 0)`:`vectors[vectorsOffset + index * ${a}u + band]`};
    let component = ${m?`getScaledQuotient(trainingVectors[trainingVectorsOffset + sampleIndex * ${a}u + band], select(1.0, sampleScale, sampleScale > 0.0), 1.0, 0)`:`trainingVectors[trainingVectorsOffset + sampleIndex * ${a}u + band]`};
    if (!finite(component)) { isSampleValid = false; }
    ${m?`dotProduct += pixelComponent * component;
    sampleLength += component * component;`:`dotProduct += (pixelComponent - component) * (pixelComponent - component);`}
  }
  if (!isSampleValid) { continue; }
  ${m?`let denominator = sqrt(squaredLength * sampleLength);
  // Zero vectors as in the published GPUSimilaritySearch: two zero vectors have similarity 1,
  // exactly one zero vector gives 0.
  let neighborDistance = 1.0 - select(select(0.0, 1.0, squaredLength == 0.0 && sampleLength == 0.0), dotProduct / denominator, denominator > 0.0);`:`let neighborDistance = dotProduct;`}
  if (!finite(neighborDistance)) { continue; }
  // Strict comparison keeps the lower sample index on equal distances.
  if (filledNeighborCount < ${f}u || neighborDistance < nearestDistance[${f-1}u]) {
    var slot = min(filledNeighborCount, ${f-1}u);
    while (slot > 0u && neighborDistance < nearestDistance[slot - 1u]) {
      nearestDistance[slot] = nearestDistance[slot - 1u];
      nearestSample[slot] = nearestSample[slot - 1u];
      slot--;
    }
    nearestDistance[slot] = neighborDistance;
    nearestSample[slot] = sampleIndex;
    filledNeighborCount = min(filledNeighborCount + 1u, ${f}u);
  }
}
for (var slot = 0u; slot < ${f}u; slot++) {
  savedSamples[savedSamplesOffset + index * ${f}u + slot] = nearestSample[slot];
  savedDistances[savedDistancesOffset + index * ${f}u + slot] = nearestDistance[slot];
}
`});return _.push({id:`${r}-vote`,operation:`GPURasterEmbeddingKNearestClassifier`,variant:`tiled-vote`,invocationCount:i,pixelCount:i,bindings:[s(`savedSamples`,h),s(`trainingLabels`,n.trainingLabels),s(`labels`,p.labels,!0),...p.confidence?[s(`confidence`,p.confidence,!0)]:[],...p.probabilities?[s(`probabilities`,p.probabilities,!0)]:[]],declarations:`${c}\n${o}`,body:`var filledNeighborCount = 0u;
for (var slot = 0u; slot < ${f}u; slot++) {
  if (savedSamples[savedSamplesOffset + index * ${f}u + slot] != 0xffffffffu) { filledNeighborCount++; }
}
if (filledNeighborCount == 0u) {
  labels[labelsOffset + index] = ${d}u;
  ${p.confidence?`confidence[confidenceOffset + index] = missing();`:``}
  ${p.probabilities?`for (var classIndex = 0u; classIndex < ${u}u; classIndex++) { probabilities[probabilitiesOffset + index * ${u}u + classIndex] = missing(); }`:``}
  return;
}
var votes: array<u32, ${u}>;
for (var slot = 0u; slot < filledNeighborCount; slot++) {
  let classIndex = min(trainingLabels[trainingLabelsOffset + savedSamples[savedSamplesOffset + index * ${f}u + slot]], ${u-1}u);
  votes[classIndex] += 1u;
}
var best = 0u;
for (var classIndex = 1u; classIndex < ${u}u; classIndex++) {
  if (votes[classIndex] > votes[best]) { best = classIndex; }
}
labels[labelsOffset + index] = best;
${p.confidence?`confidence[confidenceOffset + index] = f32(votes[best]) / f32(filledNeighborCount);`:``}
${p.probabilities?`for (var classIndex = 0u; classIndex < ${u}u; classIndex++) { probabilities[probabilitiesOffset + index * ${u}u + classIndex] = f32(votes[classIndex]) / f32(filledNeighborCount); }`:``}`}),_}},M=[3,8,20],N=E({parameters:O,rebuildOn:[`neighborCount`,`perClass`],caption:`Each pixel votes among its nearest labelled pixels (dots); 24 labelled pixels by default (8 per class), labelled by spectral rules, not field data. Six Sentinel-2 bands stand in for a learned embedding.`,createInputs:({aspect:e,random:t})=>{let n=v(e),r={width:n.columns/n.rows,height:1},i=M.map(e=>{let i=f(n,t,e),a=Float32Array.from(i.pixels.flatMap(e=>u(n,r,e)));return{...i,positions:a}});return{...n,trainingSets:i}},build(e,t,n){let r=t.trainingSets[Math.round(n.perClass)],i=e.allocate(`labels`,`uint32`,t.pixelCount),a=e.allocate(`confidence`,`float32`,t.pixelCount);return e.graph.add(new j({pixelCount:t.pixelCount,dimensions:6,vectors:e.upload(`vectors`,`float32`,t.vectors).view,trainingVectors:e.upload(`training-vectors`,`float32`,r.vectors).view,trainingLabels:e.upload(`training-labels`,`uint32`,r.labels).view,trainingCount:r.count,classCount:h.length,neighborCount:Math.min(r.count,Math.round(n.neighborCount)),metric:`euclidean`,output:{labels:i.view,confidence:a.view}})),{outputs:{labels:i,confidence:a,samplePositions:e.upload(`sample-positions`,`float32x2`,r.positions),sampleLabels:e.upload(`sample-labels`,`uint32`,r.labels)}}},interaction:{kind:`probe`,hint:`probe a pixel`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let{index:o}=g(t,i,e.x,e.y),s=p(t.vectors,o),c=t.trainingSets[Math.round(r.perClass)],l=c.pixels.map((e,t)=>{let n=p(c.vectors,t);return Math.hypot(...s.map((e,t)=>e-n[t]))}),d=Array.from(l.keys()).sort((e,t)=>l[e]-l[t]||e-t).slice(0,Math.round(r.neighborCount)),f=u(t,i,o),m=a(n.labels)?.[o],_=a(n.confidence)?.[o];return{spotlight:null,glyphs:[y(t,i,o),...d.map(e=>({shape:`line`,from:f,to:[c.positions[e*2],c.positions[e*2+1]]}))],readout:`${m===void 0?`-`:h[m]??`none`} | confidence ${_===void 0?`-`:D(_)}`}}},draw:({inputs:e,outputs:t,values:n})=>[n.shown===0?{layer:`raster`,values:t.labels,columns:e.columns,rows:e.rows}:{layer:`raster`,values:t.confidence,columns:e.columns,rows:e.rows,ramp:`sequential`,range:[0,1]},{layer:`dots`,context:!0,positions:t.samplePositions,values:t.sampleLabels,radius:5}]});export{N as default};
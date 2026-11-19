import{E as e,T as t,f as n,g as r}from"./gpu-contributor-utils-BIwLQON5.js";import{t as i}from"./scaled-quotient-wgsl-UDyhQag6.js";import{a,t as o}from"./raster-kernel-utils-DcIN7G5I.js";import{b as s,g as c,p as l,x as u}from"./embedding-DKQ2dOol.js";import{a as d,i as f,n as p,r as m,t as h}from"./raster-embedding-chunks-w-sv9jw0.js";function g(e,t=new Float32Array(1)){return r(`getGPURasterEmbeddingSimilarityParameterValues`,`matchThreshold`,e.matchThreshold),t[0]=e.matchThreshold,t}var _=class{props;id;chunks;estimatedCommandNodeCount;constructor(r){this.props=r;let i=r.id??`raster-embedding-similarity`;this.id=i,t(i,r.metric===void 0||[`cosine`,`dot`].includes(r.metric),`metric must be one of cosine, dot`);let{pixelCount:a,dimensions:o,output:c}=r;s(i,a,o,l,!1),n(i,`queryCount`,r.queryCount,1,64),u(i,`vectors`,r.vectors,a,o,r.mask),t(i,r.queries.length===r.queryCount*o,`queries must hold queryCount times dimensions entries`),t(i,!!(c.similarity||c.maximum||c.bestQuery||c.matches),`output needs at least one view`),t(i,(!c.similarity||c.similarity.length===a*r.queryCount)&&(!c.maximum||c.maximum.length===a)&&(!c.bestQuery||c.bestQuery.length===a)&&(!c.matches||c.matches.length===a),`output.similarity must hold pixelCount times queryCount entries and the other outputs pixelCount`),t(i,!c.matches||r.parameters!==void 0&&r.parameters.length>=1,`output.matches needs parameters`),r.queries&&e(r.queries,[`float32`],`${i} queries`),r.parameters&&e(r.parameters,[`float32`],`${i} parameters`);let f=this.getInputs(),p=this.getOutputs();d(i,f,p,[{inputs:f,outputs:p}]),this.chunks=m(i,a,{vectors:{view:r.vectors,format:`float32`,components:o},mask:{view:r.mask,format:`uint32`,components:1},"output.similarity":{view:r.output.similarity,format:`float32`,components:r.queryCount},"output.maximum":{view:r.output.maximum,format:`float32`,components:1},"output.bestQuery":{view:r.output.bestQuery,format:`uint32`,components:1},"output.matches":{view:r.output.matches,format:`uint32`,components:1}},`vectors`,1),this.estimatedCommandNodeCount=1+this.chunks.filter(e=>e.pixelCount>0).length*1}getInputs(){let{vectors:e,mask:t,queries:n,parameters:r}=this.props;return[e,n,...t?[t]:[],...r&&this.props.output.matches?[r]:[]]}getOutputs(){let{similarity:e,maximum:t,bestQuery:n,matches:r}=this.props.output;return[e,t,n,r].filter(e=>e!==void 0)}getCommandNodes(e){f(e,this.id,this.getInputs(),this.getOutputs(),this.estimatedCommandNodeCount-1,this.props,2+Number(!!this.props.mask)+Number(!!(this.props.parameters&&this.props.output.matches))+Object.values(this.props.output).filter(Boolean).length,this.chunks);let t=this.chunks.flatMap((t,n)=>t.pixelCount===0?[]:this.getChunkKernels(e,p(this.props,t),n));return h(e,this.id,t)}getChunkKernels(e,t,n){let r=`${this.id}-chunk-${n}`,{pixelCount:s,dimensions:l,queryCount:u,output:d}=t,f=(t.metric??`cosine`)===`cosine`,p=`${d.similarity?`for (var query = 0u; query < ${u}u; query++) { similarity[similarityOffset + index * ${u}u + query] = missing(); }`:``}
${d.maximum?`maximum[maximumOffset + index] = missing();`:``}
${d.bestQuery?`bestQuery[bestQueryOffset + index] = ${c}u;`:``}
${d.matches?`matches[matchesOffset + index] = 0u;`:``}`;return[{id:r,operation:`GPURasterEmbeddingSimilarity`,variant:f?`cosine`:`dot`,invocationCount:s,pixelCount:s,bindings:[a(`vectors`,t.vectors),a(`queries`,t.queries),...t.mask?[a(`mask`,t.mask)]:[],...t.parameters&&d.matches?[a(`parameters`,t.parameters)]:[],...d.similarity?[a(`similarity`,d.similarity,!0)]:[],...d.maximum?[a(`maximum`,d.maximum,!0)]:[],...d.bestQuery?[a(`bestQuery`,d.bestQuery,!0)]:[],...d.matches?[a(`matches`,d.matches,!0)]:[]],declarations:`${o}\n${i}`,body:`var squaredLength = 0.0;
var pixelScale = 0.0;
var isValid = ${t.mask?`mask[maskOffset + index] != 0u`:`true`};
for (var band = 0u; band < ${l}u; band++) {
  let value = vectors[vectorsOffset + index * ${l}u + band];
  if (!finite(value)) { isValid = false; }
  ${f?`pixelScale = max(pixelScale, abs(value));`:`squaredLength += value * value;`}
}
${f?`for (var band = 0u; band < ${l}u; band++) {
  let value = getScaledQuotient(vectors[vectorsOffset + index * ${l}u + band], select(1.0, pixelScale, pixelScale > 0.0), 1.0, 0);
  squaredLength += value * value;
}`:``}
if (!isValid) {
  ${p}
  return;
}
var best = -3.0e38;
var bestIndex = 0u;
for (var query = 0u; query < ${u}u; query++) {
  var dotProduct = 0.0;
  var queryLength = 0.0;
  ${f?`var queryScale = 0.0;
  for (var band = 0u; band < ${l}u; band++) {
    queryScale = max(queryScale, abs(queries[queriesOffset + query * ${l}u + band]));
  }`:``}
  for (var band = 0u; band < ${l}u; band++) {
    let component = ${f?`getScaledQuotient(queries[queriesOffset + query * ${l}u + band], select(1.0, queryScale, queryScale > 0.0), 1.0, 0)`:`queries[queriesOffset + query * ${l}u + band]`};
    let pixelComponent = ${f?`getScaledQuotient(vectors[vectorsOffset + index * ${l}u + band], select(1.0, pixelScale, pixelScale > 0.0), 1.0, 0)`:`vectors[vectorsOffset + index * ${l}u + band]`};
    dotProduct += pixelComponent * component;
    queryLength += component * component;
  }
  ${f?`let denominator = sqrt(squaredLength * queryLength);
  let similarityValue = select(select(0.0, 1.0, squaredLength == 0.0 && queryLength == 0.0), dotProduct / denominator, denominator > 0.0);`:`let similarityValue = dotProduct;`}
  ${d.similarity?`similarity[similarityOffset + index * ${u}u + query] = similarityValue;`:``}
  if (similarityValue > best) { best = similarityValue; bestIndex = query; }
}
${d.maximum?`maximum[maximumOffset + index] = best;`:``}
${d.bestQuery?`bestQuery[bestQueryOffset + index] = bestIndex;`:``}
${d.matches?`matches[matchesOffset + index] = select(0u, 1u, best >= parameters[parametersOffset]);`:``}`}]}};export{g as n,_ as t};
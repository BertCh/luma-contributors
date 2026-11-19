import{c as e,d as t,i as n,o as r,r as i,s as a,t as o,u as s}from"./kernel-C3tfQOn4.js";import{t as c}from"./assemble-compute-kernel-C8V2OpkX.js";import{n as l,t as u}from"./gpu-dispatch-utils-CbRzFAgT.js";import{t as d}from"./gpu-hash-index-BmikHOtA.js";import{T as f,f as p,i as m}from"./gpu-contributor-utils-BIwLQON5.js";import{_ as h,b as g,l as _,o as v,r as y,s as ee,x as b}from"./embedding-DKQ2dOol.js";import{t as x}from"./define-run-B-brkCe4.js";import{r as S}from"./kernels-iXfy7fsH.js";import{ti as C}from"./entry-preview-D70TdSbK.js";var w=64,T=4294967295,E=256,D=Uint32Array.BYTES_PER_ELEMENT,O=16,k=class{id;dataset;queries;outputIds;outputScores;resultCounts;candidateCounts;k;metric;filterMask;candidateIds;queryFilterMask;excludeSelf;tileSize;registered=!1;constructor(e){this.id=e.id??`gpu-similarity-search`,this.dataset=e.dataset,this.queries=e.queries,this.outputIds=e.outputIds,this.outputScores=e.outputScores,this.resultCounts=e.resultCounts,this.candidateCounts=e.candidateCounts,this.k=e.k,this.metric=e.metric??`squared-euclidean`,this.filterMask=e.filterMask,this.candidateIds=e.candidateIds,this.queryFilterMask=e.queryFilterMask,this.excludeSelf=e.excludeSelf??!1,this.tileSize=e.tileSize??T,te(this)}addToGraph(e){if(this.registered)throw Error(`${this.id} similarity search has already been added to a command graph`);if(M(this,e),this.queries.rowCount===0){this.registered=!0;return}let t=V(this.dataset),n=F(e,this,t);for(let[t,r]of n.entries())W(e,this,r,t);if(this.dataset.rowCount===0||this.k===0&&!this.candidateCounts){this.registered=!0;return}let r=U(e,this),a=P(e,this);for(let[o,s]of a.entries()){let a=i(e,`${this.id}-eligible-source-ids-${o}`,`uint32`,s.rowCount);G(e,this,s,o,a,r);for(let[r,i]of n.entries()){let n=this.queryFilterMask?z(e,this.queryFilterMask,i,s,t):void 0;i.candidateCounts&&K(e,this,s,i,o,r,a,n,t),this.k>0&&q(e,this,s,i,o,r,a,n,t)}}this.registered=!0}};function te(e){if(!Number.isSafeInteger(e.dataset.dimensions)||e.dataset.dimensions<1||e.dataset.dimensions!==e.queries.dimensions)throw Error(`${e.id} dataset and query embedding dimensions must match`);if(!Number.isSafeInteger(e.k)||e.k<0||e.k>T||!Number.isSafeInteger(e.k*e.queries.rowCount)||e.k*e.queries.rowCount>T)throw Error(`${e.id} result count and query-major output length must fit into uint32`);if(![`squared-euclidean`,`inner-product`,`cosine`].includes(e.metric))throw Error(`${e.id} metric must be squared-euclidean, inner-product, or cosine`);if(!Number.isSafeInteger(e.tileSize)||e.tileSize<1)throw Error(`${e.id} tileSize must be a positive safe integer`);A(e.dataset,`${e.id} dataset`),A(e.queries,`${e.id} queries`),s(e.outputIds,`${e.id} output IDs`),t(e.outputScores,[`float32`],`${e.id} output scores`),s(e.resultCounts,`${e.id} result counts`);let n=e.queries.rowCount*e.k;if(e.outputIds.length<n||e.outputScores.length<n)throw Error(`${e.id} result destinations must contain queryCount * k values`);if(e.resultCounts.length<e.queries.rowCount)throw Error(`${e.id} result counts must contain one value per query`);if(e.candidateCounts&&(s(e.candidateCounts,`${e.id} candidate counts`),e.candidateCounts.length<e.queries.rowCount))throw Error(`${e.id} candidate counts must contain one value per query`);e.candidateIds&&s(e.candidateIds,`${e.id} candidate IDs`);let r=V(e.dataset);if(e.filterMask&&ne(e,r),e.queryFilterMask&&(s(e.queryFilterMask,`${e.id} query filter mask`),!Number.isSafeInteger(r*e.queries.rowCount)||e.queryFilterMask.length<r*e.queries.rowCount))throw Error(`${e.id} query filter mask must contain query-major source flags`);j(e)}function A(e,n){if(!Number.isSafeInteger(e.rowCount)||e.rowCount<0)throw Error(`${n} row count must be a non-negative safe integer`);let r=0;for(let i of e.chunks){if(t(i.values,[`float32`],`${n} flat values`),!Number.isSafeInteger(i.rowCount)||i.rowCount<0||!Number.isSafeInteger(i.rowStride)||i.rowStride<e.dimensions||i.byteOffset!==i.values.byteOffset||!Number.isSafeInteger(i.sourceRowOffset)||i.sourceRowOffset<0||i.sourceRowOffset+i.rowCount>T||i.rowCount>0&&i.values.length<(i.rowCount-1)*i.rowStride+e.dimensions)throw Error(`${n} chunk must preserve aligned, bounded float32 embedding rows`);for(let e of[i.sourceRowIds,i.validity])if(e&&(s(e,`${n} row metadata`),e.length<i.rowCount))throw Error(`${n} row metadata must align with its source chunk`);r+=i.rowCount}if(r!==e.rowCount||r>T)throw Error(`${n} row count must match its chunk rows and fit into uint32`)}function ne(e,t){let n=e.filterMask;if(n){if(H(n)){if(n.data.length!==e.dataset.chunks.length||n.data.some((t,n)=>t.length!==e.dataset.chunks[n].rowCount))throw Error(`${e.id} filter mask must preserve the dataset chunk topology`);for(let t of n.data)s(t,`${e.id} filter mask`)}else if(s(n,`${e.id} filter mask`),n.length<t)throw Error(`${e.id} filter mask must contain source-aligned selection flags`)}}function j(e){let t=[...e.dataset.chunks.flatMap(N),...e.queries.chunks.flatMap(N),...e.filterMask?H(e.filterMask)?e.filterMask.data:[e.filterMask]:[],...e.candidateIds?[e.candidateIds]:[],...e.queryFilterMask?[e.queryFilterMask]:[]],r=[e.outputIds,e.outputScores,e.resultCounts,...e.candidateCounts?[e.candidateCounts]:[]];for(let[i,a]of r.entries())if(t.some(e=>n(e,a))||r.slice(0,i).some(e=>n(e,a)))throw Error(`${e.id} writable outputs must not overlap source or destination data`)}function M(e,t){if([...e.dataset.chunks.flatMap(N),...e.queries.chunks.flatMap(N),e.outputIds,e.outputScores,e.resultCounts,...e.candidateCounts?[e.candidateCounts]:[],...e.candidateIds?[e.candidateIds]:[],...e.queryFilterMask?[e.queryFilterMask]:[],...e.filterMask?H(e.filterMask)?e.filterMask.data:[e.filterMask]:[]].some(e=>e.buffer.graph!==t))throw Error(`${e.id} inputs and outputs must belong to their target command graph`)}function N(e){return[e.values,...e.sourceRowIds?[e.sourceRowIds]:[],...e.validity?[e.validity]:[]]}function P(e,t){let n=e.device.limits.maxStorageBufferBindingSize,r=[];for(let[i,a]of t.dataset.chunks.entries()){let o=0;for(;o<a.rowCount;){let s=Math.min(a.rowCount-o,t.tileSize,B(a,o,t.dataset.dimensions,n),Math.max(1,Math.floor((n-E+1)/D))),c=I(e,a,o,s,t.dataset.dimensions),l={chunk:a,chunkIndex:i,chunkRowOffset:o,sourceRowOffset:a.sourceRowOffset+o,rowCount:s,values:c,...a.sourceRowIds?{sourceRowIds:L(e,a.sourceRowIds,o,s)}:{},...a.validity?{validity:L(e,a.validity,o,s)}:{}},u=R(e,t.filterMask,l);u&&(l.filterMask=u),r.push(l),o+=s}}return r}function F(e,t,n){let r=e.device.limits.maxStorageBufferBindingSize,a=Math.max(1,Math.floor((r-E+1)/(Math.max(t.k,1)*D))),o=t.queryFilterMask?Math.max(1,Math.floor((r-E+1)/(Math.max(n,1)*D))):T,s=Math.min(a,o),c=[],l=0;for(let n of t.queries.chunks){let a=0;for(;a<n.rowCount;){let o=Math.min(n.rowCount-a,s,B(n,a,t.queries.dimensions,r)),u=l+a,d=u*t.k,f={logicalRowOffset:u,rowCount:o,values:I(e,n,a,o,t.queries.dimensions),rowStride:n.rowStride,sourceRowOffset:n.sourceRowOffset+a,rowMetadata:i(e,`${t.id}-query-metadata-${c.length}`,`uint32`,o),outputIds:L(e,t.outputIds,d,o*t.k),outputScores:L(e,t.outputScores,d,o*t.k),resultCounts:L(e,t.resultCounts,u,o),...t.candidateCounts?{candidateCounts:L(e,t.candidateCounts,u,o)}:{},...n.sourceRowIds?{sourceRowIds:L(e,n.sourceRowIds,a,o)}:{},...n.validity?{validity:L(e,n.validity,a,o)}:{}};c.push(f),a+=o}l+=n.rowCount}return c}function I(e,t,n,r,i){return e.createDataView(t.values.buffer,{format:`float32`,length:r===0?0:(r-1)*t.rowStride+i,byteOffset:t.byteOffset+n*t.rowStride*Float32Array.BYTES_PER_ELEMENT})}function L(e,t,n,r){return e.createDataView(t.buffer,{format:t.format,length:r,byteOffset:t.byteOffset+n*D})}function R(e,t,n){if(t)return H(t)?L(e,t.data[n.chunkIndex],n.chunkRowOffset,n.rowCount):L(e,t,n.sourceRowOffset,n.rowCount)}function z(e,t,n,r,i){let o=L(e,t,n.logicalRowOffset*i+r.sourceRowOffset,(n.rowCount-1)*i+r.rowCount);if(a(o).size>e.device.limits.maxStorageBufferBindingSize)throw Error(`GPU embedding query filter tile exceeds maxStorageBufferBindingSize`);return o}function B(e,t,n,r){let i=(e.byteOffset+t*e.rowStride*D)%E,a=n*D,o=r-i;if(o<a)throw Error(`GPU embedding row exceeds maxStorageBufferBindingSize`);return Math.floor((o-a)/(e.rowStride*D))+1}function V(e){return e.chunks.reduce((e,t)=>Math.max(e,t.sourceRowOffset+t.rowCount),0)}function H(e){return Array.isArray(e.data)}function U(e,t){if(!t.candidateIds)return;let n=e.device.limits.maxStorageBufferBindingSize;if(a(t.candidateIds).size>n)throw Error(`${t.id} candidate IDs exceed maxStorageBufferBindingSize`);if(t.candidateIds.length<=O)return;let r=2,o=t.candidateIds.length*2;for(;r<o;)r*=2;if(r*D>n)throw Error(`${t.id} candidate-ID membership exceeds maxStorageBufferBindingSize; use a source-aligned filter mask`);let s=i(e,`${t.id}-candidate-index-keys`,`uint32`,r),c=i(e,`${t.id}-candidate-index-values`,`uint32`,r),l=i(e,`${t.id}-candidate-index-statistics`,`uint32`,6);return e.add(new d({id:`${t.id}-candidate-index`,keys:t.candidateIds,tableKeys:s,tableValues:c,statistics:l})),{keys:s,statistics:l,capacity:r}}function W(t,n,r,i){let a=`${n.id}-initialize-query-tile-${i}`,o={queryValues:r.values,queryMetadata:r.rowMetadata,outputIds:r.outputIds,outputScores:r.outputScores,resultCounts:r.resultCounts,...r.candidateCounts?{candidateCounts:r.candidateCounts}:{},...r.validity?{queryValidity:r.validity}:{},...r.sourceRowIds?{querySourceIds:r.sourceRowIds}:{}};J(t,a,`
${X(n,r.rowCount)}
const QUERY_OFFSET: u32 = ${e(r.values)}u;
const QUERY_STRIDE: u32 = ${r.rowStride}u;
const QUERY_SOURCE_OFFSET: u32 = ${r.sourceRowOffset}u;
const QUERY_METADATA_OFFSET: u32 = ${e(r.rowMetadata)}u;
const OUTPUT_ID_OFFSET: u32 = ${e(r.outputIds)}u;
const OUTPUT_SCORE_OFFSET: u32 = ${e(r.outputScores)}u;
const RESULT_COUNT_OFFSET: u32 = ${e(r.resultCounts)}u;
${r.candidateCounts?`const CANDIDATE_COUNT_OFFSET: u32 = ${e(r.candidateCounts)}u;`:``}
${r.validity?`const QUERY_VALIDITY_OFFSET: u32 = ${e(r.validity)}u;`:``}
${r.sourceRowIds?`const QUERY_SOURCE_ID_OFFSET: u32 = ${e(r.sourceRowIds)}u;`:``}
@group(0) @binding(auto) var<storage, read> queryValues: array<f32>;
@group(0) @binding(auto) var<storage, read_write> queryMetadata: array<u32>;
@group(0) @binding(auto) var<storage, read_write> outputIds: array<u32>;
@group(0) @binding(auto) var<storage, read_write> outputScores: array<u32>;
@group(0) @binding(auto) var<storage, read_write> resultCounts: array<u32>;
${r.candidateCounts?`@group(0) @binding(auto) var<storage, read_write> candidateCounts: array<u32>;`:``}
${r.validity?`@group(0) @binding(auto) var<storage, read> queryValidity: array<u32>;`:``}
${r.sourceRowIds?`@group(0) @binding(auto) var<storage, read> querySourceIds: array<u32>;`:``}

${Z()}

@compute @workgroup_size(${w})
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${Y(a,r.rowCount,t)}
  if (index >= QUERY_COUNT) { return; }
  var valid = ${r.validity?`queryValidity[QUERY_VALIDITY_OFFSET + index] != 0u`:`true`};
  for (var dimension = 0u; dimension < DIMENSIONS; dimension++) {
    if (!isFiniteFloat(queryValues[QUERY_OFFSET + index * QUERY_STRIDE + dimension])) {
      valid = false;
    }
  }
  let sourceId = ${r.sourceRowIds?`querySourceIds[QUERY_SOURCE_ID_OFFSET + index]`:`QUERY_SOURCE_OFFSET + index`};
  queryMetadata[QUERY_METADATA_OFFSET + index] = select(INVALID_ID, sourceId, valid && sourceId != INVALID_ID);
  resultCounts[RESULT_COUNT_OFFSET + index] = 0u;
  ${r.candidateCounts?`candidateCounts[CANDIDATE_COUNT_OFFSET + index] = 0u;`:``}
  for (var resultIndex = 0u; resultIndex < MAX_RESULTS; resultIndex++) {
    let outputIndex = index * MAX_RESULTS + resultIndex;
    outputIds[OUTPUT_ID_OFFSET + outputIndex] = INVALID_ID;
    outputScores[OUTPUT_SCORE_OFFSET + outputIndex] = ${n.metric===`squared-euclidean`?`0x7f800000u`:`0xff800000u`};
  }
}`,o,r.rowCount,new Set([`queryValues`,`queryValidity`,`querySourceIds`]))}function G(t,n,r,i,a,o){let s=`${n.id}-prepare-dataset-tile-${i}`,c={datasetValues:r.values,eligibleSourceIds:a,...r.sourceRowIds?{sourceRowIds:r.sourceRowIds}:{},...r.validity?{sourceValidity:r.validity}:{},...r.filterMask?{selectionMask:r.filterMask}:{},...o?{candidateIndexKeys:o.keys,candidateIndexStatistics:o.statistics,candidateIds:n.candidateIds}:n.candidateIds?{candidateIds:n.candidateIds}:{}};J(t,s,`
${X(n,r.rowCount)}
const ROW_COUNT: u32 = ${r.rowCount}u;
const DATASET_OFFSET: u32 = ${e(r.values)}u;
const DATASET_STRIDE: u32 = ${r.chunk.rowStride}u;
const SOURCE_ROW_OFFSET: u32 = ${r.sourceRowOffset}u;
const ELIGIBLE_OFFSET: u32 = ${e(a)}u;
${r.sourceRowIds?`const SOURCE_ID_OFFSET: u32 = ${e(r.sourceRowIds)}u;`:``}
${r.validity?`const VALIDITY_OFFSET: u32 = ${e(r.validity)}u;`:``}
${r.filterMask?`const SELECTION_OFFSET: u32 = ${e(r.filterMask)}u;`:``}
${o?`const CANDIDATE_INDEX_OFFSET: u32 = ${e(o.keys)}u;\nconst CANDIDATE_INDEX_MASK: u32 = ${o.capacity-1}u;\nconst CANDIDATE_INDEX_CAPACITY: u32 = ${o.capacity}u;\nconst CANDIDATE_STATISTICS_OFFSET: u32 = ${e(o.statistics)}u;\nconst CANDIDATE_ID_OFFSET: u32 = ${e(n.candidateIds)}u;\nconst CANDIDATE_ID_COUNT: u32 = ${n.candidateIds.length}u;`:n.candidateIds?`const CANDIDATE_ID_OFFSET: u32 = ${e(n.candidateIds)}u;\nconst CANDIDATE_ID_COUNT: u32 = ${n.candidateIds.length}u;`:``}
@group(0) @binding(auto) var<storage, read> datasetValues: array<f32>;
@group(0) @binding(auto) var<storage, read_write> eligibleSourceIds: array<u32>;
${r.sourceRowIds?`@group(0) @binding(auto) var<storage, read> sourceRowIds: array<u32>;`:``}
${r.validity?`@group(0) @binding(auto) var<storage, read> sourceValidity: array<u32>;`:``}
${r.filterMask?`@group(0) @binding(auto) var<storage, read> selectionMask: array<u32>;`:``}
${o?`@group(0) @binding(auto) var<storage, read> candidateIndexKeys: array<u32>;
@group(0) @binding(auto) var<storage, read> candidateIndexStatistics: array<u32>;
@group(0) @binding(auto) var<storage, read> candidateIds: array<u32>;`:n.candidateIds?`@group(0) @binding(auto) var<storage, read> candidateIds: array<u32>;`:``}

${Z()}
${o?`fn hashCandidateKey(key: u32) -> u32 {
  var value = key;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}`:``}

@compute @workgroup_size(${w})
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${Y(s,r.rowCount,t)}
  if (index >= ROW_COUNT) { return; }
  let sourceId = ${r.sourceRowIds?`sourceRowIds[SOURCE_ID_OFFSET + index]`:`SOURCE_ROW_OFFSET + index`};
  var eligible = sourceId != INVALID_ID;
  ${r.validity?`eligible = eligible && sourceValidity[VALIDITY_OFFSET + index] != 0u;`:``}
  ${r.filterMask?`eligible = eligible && selectionMask[SELECTION_OFFSET + index] != 0u;`:``}
  for (var dimension = 0u; dimension < DIMENSIONS; dimension++) {
    if (!isFiniteFloat(datasetValues[DATASET_OFFSET + index * DATASET_STRIDE + dimension])) {
      eligible = false;
    }
  }
  ${o?`var allowed = false;
  if (candidateIndexStatistics[CANDIDATE_STATISTICS_OFFSET + 2u] == 0u) {
    let firstSlot = hashCandidateKey(sourceId) & CANDIDATE_INDEX_MASK;
    for (var probe = 0u; probe < CANDIDATE_INDEX_CAPACITY; probe++) {
      let stored = candidateIndexKeys[CANDIDATE_INDEX_OFFSET + ((firstSlot + probe) & CANDIDATE_INDEX_MASK)];
      if (stored == INVALID_ID) { break; }
      if (stored == sourceId) { allowed = true; break; }
    }
  } else {
    for (var candidateIndex = 0u; candidateIndex < CANDIDATE_ID_COUNT; candidateIndex++) {
      allowed = allowed || candidateIds[CANDIDATE_ID_OFFSET + candidateIndex] == sourceId;
    }
  }
  eligible = eligible && allowed;`:n.candidateIds?`var allowed = false;
  for (var candidateIndex = 0u; candidateIndex < CANDIDATE_ID_COUNT; candidateIndex++) {
    allowed = allowed || candidateIds[CANDIDATE_ID_OFFSET + candidateIndex] == sourceId;
  }
  eligible = eligible && allowed;`:``}
  eligibleSourceIds[ELIGIBLE_OFFSET + index] = select(INVALID_ID, sourceId, eligible);
}`,c,r.rowCount,new Set(Object.keys(c).filter(e=>e!==`eligibleSourceIds`)))}function K(t,n,r,i,a,o,s,c,l){if(!i.candidateCounts)return;let u=`${n.id}-count-candidates-${a}-${o}`,d={eligibleSourceIds:s,queryMetadata:i.rowMetadata,candidateCounts:i.candidateCounts,...c?{queryFilterMask:c}:{}};J(t,u,`
${X(n,i.rowCount)}
const ROW_COUNT: u32 = ${r.rowCount}u;
const ELIGIBLE_OFFSET: u32 = ${e(s)}u;
const QUERY_METADATA_OFFSET: u32 = ${e(i.rowMetadata)}u;
const CANDIDATE_COUNT_OFFSET: u32 = ${e(i.candidateCounts)}u;
${c?`const QUERY_FILTER_OFFSET: u32 = ${e(c)}u;\nconst SOURCE_SPAN: u32 = ${l}u;`:``}
@group(0) @binding(auto) var<storage, read> eligibleSourceIds: array<u32>;
@group(0) @binding(auto) var<storage, read> queryMetadata: array<u32>;
@group(0) @binding(auto) var<storage, read_write> candidateCounts: array<u32>;
${c?`@group(0) @binding(auto) var<storage, read> queryFilterMask: array<u32>;`:``}

@compute @workgroup_size(${w})
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${Y(u,i.rowCount,t)}
  if (index >= QUERY_COUNT) { return; }
  let querySourceId = queryMetadata[QUERY_METADATA_OFFSET + index];
  if (querySourceId == INVALID_ID) { return; }
  var count = candidateCounts[CANDIDATE_COUNT_OFFSET + index];
  for (var candidateIndex = 0u; candidateIndex < ROW_COUNT; candidateIndex++) {
    let sourceId = eligibleSourceIds[ELIGIBLE_OFFSET + candidateIndex];
    var accepted = sourceId != INVALID_ID;
    ${n.excludeSelf?`accepted = accepted && sourceId != querySourceId;`:``}
    ${c?`accepted = accepted && queryFilterMask[QUERY_FILTER_OFFSET + index * SOURCE_SPAN + candidateIndex] != 0u;`:``}
    if (accepted) { count++; }
  }
  candidateCounts[CANDIDATE_COUNT_OFFSET + index] = count;
}`,d,i.rowCount,new Set([`eligibleSourceIds`,`queryMetadata`,`queryFilterMask`]))}function q(t,n,r,i,a,o,s,c,l){let u=`${n.id}-search-dataset-${a}-queries-${o}`,d={queryValues:i.values,datasetValues:r.values,eligibleSourceIds:s,queryMetadata:i.rowMetadata,outputIds:i.outputIds,outputScores:i.outputScores,resultCounts:i.resultCounts,...c?{queryFilterMask:c}:{}};J(t,u,`
${X(n,i.rowCount)}
const ROW_COUNT: u32 = ${r.rowCount}u;
const QUERY_OFFSET: u32 = ${e(i.values)}u;
const QUERY_STRIDE: u32 = ${i.rowStride}u;
const DATASET_OFFSET: u32 = ${e(r.values)}u;
const DATASET_STRIDE: u32 = ${r.chunk.rowStride}u;
const ELIGIBLE_OFFSET: u32 = ${e(s)}u;
const QUERY_METADATA_OFFSET: u32 = ${e(i.rowMetadata)}u;
const OUTPUT_ID_OFFSET: u32 = ${e(i.outputIds)}u;
const OUTPUT_SCORE_OFFSET: u32 = ${e(i.outputScores)}u;
const RESULT_COUNT_OFFSET: u32 = ${e(i.resultCounts)}u;
${c?`const QUERY_FILTER_OFFSET: u32 = ${e(c)}u;\nconst SOURCE_SPAN: u32 = ${l}u;`:``}
@group(0) @binding(auto) var<storage, read> queryValues: array<f32>;
@group(0) @binding(auto) var<storage, read> datasetValues: array<f32>;
@group(0) @binding(auto) var<storage, read> eligibleSourceIds: array<u32>;
@group(0) @binding(auto) var<storage, read> queryMetadata: array<u32>;
@group(0) @binding(auto) var<storage, read_write> outputIds: array<u32>;
@group(0) @binding(auto) var<storage, read_write> outputScores: array<f32>;
@group(0) @binding(auto) var<storage, read_write> resultCounts: array<u32>;
${c?`@group(0) @binding(auto) var<storage, read> queryFilterMask: array<u32>;`:``}

${Z()}

struct CandidateScore {
  value: f32,
  valid: bool
}

fn candidateScore(queryIndex: u32, candidateIndex: u32) -> CandidateScore {
  var dotProduct = 0.0;
  var queryNorm = 0.0;
  var candidateNorm = 0.0;
  var squaredDistance = 0.0;
  var queryScale = 0.0;
  var candidateScale = 0.0;
  ${n.metric===`inner-product`?`var hasPositiveInfiniteProduct = false;
  var hasNegativeInfiniteProduct = false;`:``}
  for (var dimension = 0u; dimension < DIMENSIONS; dimension++) {
    let queryValue = queryValues[QUERY_OFFSET + queryIndex * QUERY_STRIDE + dimension];
    let candidateValue = datasetValues[DATASET_OFFSET + candidateIndex * DATASET_STRIDE + dimension];
    ${n.metric===`squared-euclidean`?`let delta = queryValue - candidateValue;
    squaredDistance += delta * delta;`:n.metric===`inner-product`?`let componentProduct = queryValue * candidateValue;
    let componentProductBits = bitcast<u32>(componentProduct);
    if ((componentProductBits & 0x7fffffffu) == 0x7f800000u) {
      hasNegativeInfiniteProduct = hasNegativeInfiniteProduct || (componentProductBits & 0x80000000u) != 0u;
      hasPositiveInfiniteProduct = hasPositiveInfiniteProduct || (componentProductBits & 0x80000000u) == 0u;
    }
    dotProduct += componentProduct;`:`queryScale = max(queryScale, abs(queryValue));
    candidateScale = max(candidateScale, abs(candidateValue));`}
  }
  ${n.metric===`squared-euclidean`?`return CandidateScore(squaredDistance, true);`:n.metric===`inner-product`?`return CandidateScore(dotProduct, !(hasPositiveInfiniteProduct && hasNegativeInfiniteProduct));`:`if (queryScale == 0.0 && candidateScale == 0.0) { return CandidateScore(1.0, true); }
  if (queryScale == 0.0 || candidateScale == 0.0) { return CandidateScore(0.0, true); }
  // Clamp divisors so reciprocal-lowering GPU backends never flush them to zero.
  let minimumDivisor = bitcast<f32>(0x00800000u);
  let maximumDivisor = bitcast<f32>(0x7e800000u);
  let queryDivisor = clamp(queryScale, minimumDivisor, maximumDivisor);
  let candidateDivisor = clamp(candidateScale, minimumDivisor, maximumDivisor);
  for (var dimension = 0u; dimension < DIMENSIONS; dimension++) {
    let normalizedQuery = queryValues[QUERY_OFFSET + queryIndex * QUERY_STRIDE + dimension] / queryDivisor;
    let normalizedCandidate = datasetValues[DATASET_OFFSET + candidateIndex * DATASET_STRIDE + dimension] / candidateDivisor;
    dotProduct += normalizedQuery * normalizedCandidate;
    queryNorm += normalizedQuery * normalizedQuery;
    candidateNorm += normalizedCandidate * normalizedCandidate;
  }
  return CandidateScore(dotProduct / (sqrt(queryNorm) * sqrt(candidateNorm)), true);`}
}

fn isBetter(score: f32, sourceId: u32, previousScore: f32, previousId: u32) -> bool {
  if (previousId == INVALID_ID) { return true; }
  if (score == previousScore) { return sourceId < previousId; }
  return score ${n.metric===`squared-euclidean`?`<`:`>`} previousScore;
}

@compute @workgroup_size(${w})
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${Y(u,i.rowCount,t)}
  if (index >= QUERY_COUNT) { return; }
  let querySourceId = queryMetadata[QUERY_METADATA_OFFSET + index];
  if (querySourceId == INVALID_ID) { return; }
  let queryResultOffset = index * MAX_RESULTS;
  var resultCount = resultCounts[RESULT_COUNT_OFFSET + index];
  for (var candidateIndex = 0u; candidateIndex < ROW_COUNT; candidateIndex++) {
    let sourceId = eligibleSourceIds[ELIGIBLE_OFFSET + candidateIndex];
    if (sourceId == INVALID_ID) { continue; }
    ${n.excludeSelf?`if (sourceId == querySourceId) { continue; }`:``}
    ${c?`if (queryFilterMask[QUERY_FILTER_OFFSET + index * SOURCE_SPAN + candidateIndex] == 0u) { continue; }`:``}
    let candidateScoreResult = candidateScore(index, candidateIndex);
    if (!candidateScoreResult.valid) { continue; }
    let score = candidateScoreResult.value;
    // Finite source values may legitimately overflow a Float32 distance or inner product.
    if ((bitcast<u32>(score) & 0x7fffffffu) > 0x7f800000u) { continue; }

    var insertionIndex = min(resultCount, MAX_RESULTS);
    for (var resultIndex = 0u; resultIndex < resultCount; resultIndex++) {
      let currentId = outputIds[OUTPUT_ID_OFFSET + queryResultOffset + resultIndex];
      let currentScore = outputScores[OUTPUT_SCORE_OFFSET + queryResultOffset + resultIndex];
      if (isBetter(score, sourceId, currentScore, currentId)) {
        insertionIndex = resultIndex;
        break;
      }
    }
    if (insertionIndex >= MAX_RESULTS) { continue; }
    var destinationIndex = min(resultCount, MAX_RESULTS - 1u);
    while (destinationIndex > insertionIndex) {
      let destination = queryResultOffset + destinationIndex;
      let previous = destination - 1u;
      outputIds[OUTPUT_ID_OFFSET + destination] = outputIds[OUTPUT_ID_OFFSET + previous];
      outputScores[OUTPUT_SCORE_OFFSET + destination] = outputScores[OUTPUT_SCORE_OFFSET + previous];
      destinationIndex--;
    }
    outputIds[OUTPUT_ID_OFFSET + queryResultOffset + insertionIndex] = sourceId;
    outputScores[OUTPUT_SCORE_OFFSET + queryResultOffset + insertionIndex] = score;
    resultCount = min(resultCount + 1u, MAX_RESULTS);
  }
  resultCounts[RESULT_COUNT_OFFSET + index] = resultCount;
}`,d,i.rowCount,new Set([`queryValues`,`datasetValues`,`eligibleSourceIds`,`queryMetadata`,`queryFilterMask`]))}function J(e,t,n,i,s,l){let d=Object.entries(i).map(([e,t])=>({buffer:t,usage:l.has(e)?`storage-read`:`storage-read-write`})),f=u(t,s,w,e.device.limits.maxComputeWorkgroupsPerDimension);e.addComputePass({id:t,resources:d,compile:({device:e})=>{let s=new o(e,c(e,{id:t,source:n,shaderLayout:{bindings:Object.keys(i).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}}));return{encode:({computePass:n,getBuffer:o})=>{let c={};for(let[n,s]of Object.entries(i)){if(a(s).size>e.limits.maxStorageBufferBindingSize)throw Error(`${t} storage binding exceeds maxStorageBufferBindingSize`);c[n]=r(s,o)}let l=c;s.dispatch(n,{bindings:l,x:f.x,y:f.y,z:f.z})},destroy:()=>s.destroy()}}})}function Y(e,t,n){let r=u(e,t,w,n.device.limits.maxComputeWorkgroupsPerDimension);return l(r,w)}function X(e,t){return`const INVALID_ID: u32 = 0xffffffffu;
const DIMENSIONS: u32 = ${e.dataset.dimensions}u;
const QUERY_COUNT: u32 = ${t}u;
const MAX_RESULTS: u32 = ${e.k}u;`}function Z(){return`fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}`}var re=class{props;id;search;commandNodes=new WeakMap;constructor(e){this.props=e;let t=e.id??`raster-embedding-nearest-pixels`;this.id=t;let{pixelCount:n,dimensions:r,output:i}=e;g(t,n,r),b(t,`vectors`,e.vectors,n,r,e.mask),p(t,`queryCount`,e.queryCount,1,64),p(t,`resultCount`,e.resultCount),f(t,e.queries.length===e.queryCount*r,`queries must hold queryCount times dimensions entries`),this.search=new k({id:t,dataset:h(e.vectors,n,r,e.mask),queries:h(e.queries,e.queryCount,r),outputIds:i.pixelRows,outputScores:i.scores,resultCounts:i.resultCounts,k:e.resultCount,metric:`cosine`})}getCommandNodes(e){let t=this.commandNodes.get(e);if(t)return t;let n=Object.freeze(m(e,()=>this.search.addToGraph(e)));return this.commandNodes.set(e,n),n}},Q=[8,32,128,512],$=[.3,.4],ie=x({parameters:C,rebuildOn:[`resultCount`],caption:`Pick a pixel: the most similar pixels in the scene, ranked; ties order by pixel index. Six Sentinel-2 bands stand in for a learned embedding.`,createInputs:({aspect:e})=>{let t=y(e),n=Math.floor($[1]*t.rows)*t.columns+Math.floor($[0]*t.columns);return{...t,queryPixel:n}},build(e,t,n){let r=Q[Math.round(n.resultCount)],i=e.upload(`vectors`,`float32`,t.vectors),a=e.allocate(`queries`,`float32`,6),o=e.allocate(`pixel-rows`,`uint32`,r),s=e.allocate(`scores`,`float32`,r),c=e.allocate(`result-counts`,`uint32`,1);e.graph.add(new re({pixelCount:t.pixelCount,dimensions:6,vectors:i.view,queryCount:1,queries:a.view,resultCount:r,output:{pixelRows:o.view,scores:s.view,resultCounts:c.view}}));let l={width:t.columns/t.rows,height:1},u=e.allocate(`positions`,`float32x2`,r);return S(e,`result-positions`,o,t,l,u),{outputs:{positions:u,scores:s,nearInfrared:e.upload(`near-infrared`,`float32`,t.nearInfrared),vegetationIndex:e.upload(`vegetation-index`,`float32`,t.vegetationIndex)},update:()=>a.write(Float32Array.from(_(t.vectors,t.queryPixel)))}},interaction:{kind:`probe`,hint:`pick a pixel`,rest:$,resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let{index:o}=v(t,i,e.x,e.y),s=o!==t.queryPixel;t.queryPixel=o;let c=a(n.scores),l=c?c[c.length-1]:NaN;return{rerun:s,spotlight:null,glyphs:[ee(t,i,o,1)],readout:`resultCount ${Q[Math.round(r.resultCount)]} | lowest score ${Number.isFinite(l)?l.toFixed(3):`-`}`}}},draw:({inputs:e,outputs:t,values:n})=>[{layer:`raster`,context:!0,values:n.context===0?t.nearInfrared:t.vegetationIndex,columns:e.columns,rows:e.rows,ramp:`grey`,smoothing:!0},{layer:`dots`,positions:t.positions,values:t.scores,ramp:`sequential`,range:`auto`,radius:7}]});export{ie as default};
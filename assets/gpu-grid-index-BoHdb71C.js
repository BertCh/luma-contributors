import{d as e,f as t,l as n,p as r}from"./gpu-command-graph-BdzdmV6u.js";import{t as i}from"./gpu-grid-coordinate-DwQPYH7E.js";import{n as a,t as o}from"./gpu-dispatch-utils-CbRzFAgT.js";import{n as s,t as c}from"./gpu-scan-B4NNTfF9.js";import{a as l,c as u,d,i as f,n as p,r as m,u as h}from"./kernel-C3tfQOn4.js";import{n as g,r as _}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{n as v,r as y,t as b}from"./gpu-chunk-utils-DMYEopiS.js";var x=256,S=Uint32Array.BYTES_PER_ELEMENT,C=class{id;source;indices;output;wordsPerRow;constructor(e){this.id=e.id??`gpu-scatter`,this.source=e.source,this.indices=e.indices,this.output=e.output;for(let e of _(this.source))T(e,`${this.id} source`);for(let e of _(this.indices))h(e,`${this.id} indices`);for(let e of _(this.output))T(e,`${this.id} output`);if(this.source.format!==this.output.format)throw Error(`${this.id} source and output must use the same format`);if(this.source.length<this.indices.length)throw Error(`${this.id} source must contain at least indices.length rows`);let t=new Set([..._(this.source).map(e=>e.buffer),..._(this.indices).map(e=>e.buffer)]);if(_(this.output).some(e=>t.has(e.buffer)))throw Error(`${this.id} output must use a separate buffer`);this.wordsPerRow=this.source.rowByteLength/S}getCommandNodes(e){return w(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)}};function w(e,t,n){y(t,[e.source,e.indices],[e.output]);let r=[],i=g(t,[l(t,e.source,e.indices.length),e.indices]);for(let[a,[s,c]]of i.entries()){let l=o(e.id,c.length,x,n),u=0;for(let[n,o]of _(e.output).entries())o.length&&r.push(b(t,{id:i.length===1&&_(e.output).length===1?e.id:`${e.id}-${a}-${n}`,inputs:{sourceWords:s,indices:c},outputs:{outputWords:o},dispatch:l,source:E({source:s,indices:c,output:o,wordsPerRow:e.wordsPerRow,outputStart:u},l)})),u+=o.length}return r}function T(n,i){if(r(n.format)||t(n.format))throw Error(`${i} must use a fixed-width GPU data format`);let a=e(n.format);if(n.byteStride!==a.byteLength||n.rowByteLength!==a.byteLength||n.byteOffset%S!==0||n.rowByteLength%S!==0)throw Error(`${i} must be packed, uint32-aligned GPU data with 32-bit-word rows`)}function E(e,t){return`const INDEX_COUNT: u32 = ${e.indices.length}u;
const OUTPUT_LENGTH: u32 = ${e.output.length}u;
const WORDS_PER_ROW: u32 = ${e.wordsPerRow}u;
const SOURCE_OFFSET: u32 = ${u(e.source)}u;
const INDEX_OFFSET: u32 = ${u(e.indices)}u;
const OUTPUT_OFFSET: u32 = ${u(e.output)}u;
@group(0) @binding(0) var<storage, read> sourceWords: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputWords: array<u32>;
@compute @workgroup_size(${x})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${a(t,x)}
  if (index >= INDEX_COUNT) { return; }
  let destinationIndex = indices[INDEX_OFFSET + index];
  if (destinationIndex < ${e.outputStart}u || destinationIndex - ${e.outputStart}u >= OUTPUT_LENGTH) { return; }

  let sourceBase = SOURCE_OFFSET + index * WORDS_PER_ROW;
  let outputBase = OUTPUT_OFFSET + (destinationIndex - ${e.outputStart}u) * WORDS_PER_ROW;
  for (var word = 0u; word < WORDS_PER_ROW; word++) {
    outputWords[outputBase + word] = sourceWords[sourceBase + word];
  }
}`}var D=256;function O(e,t,n){y(t,[e.positions,...e.sourceIds?[e.sourceIds]:[]],[e.cellOffsets,e.objectIds,e.count,e.overflow]);let r=[],i=v(t,e.cellOffsets,0,e.cellCount),a=p(t,`${e.id}-counts`,i),o=p(t,`${e.id}-cursors`,i),l=_(e.positions),d=l.map((n,r)=>m(t,`${e.id}-ranks-${r}`,`uint32`,n.length)),f=e.sourceIds?void 0:l.map((n,r)=>m(t,`${e.id}-ids-${r}`,`uint32`,n.length)),h=e.firstSourceIndex;for(let[i,a]of l.entries()){if(a.length){let o=d[i],s=f?.[i],c=A(a.length,n);r.push(b(t,{id:`${e.id}-initialize-ranks-${i}`,outputs:{ranks:o,...s?{ids:s}:{}},dispatch:c,source:`
@group(0) @binding(0) var<storage, read_write> ranks: array<u32>;
${s?`@group(0) @binding(1) var<storage, read_write> ids: array<u32>;`:``}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${j(c)}
  if (index >= ${a.length}u) { return; }
  ranks[index] = 0xffffffffu;
  ${s?`ids[index] = ${h}u + index;`:``}
}`}))}h+=a.length}let g=0;for(let[i,s]of a.data.entries()){let a=o.data[i],c=A(s.length,n);r.push(b(t,{id:`${e.id}-initialize-${i}`,outputs:{cellCounts:s,cellCursors:a},dispatch:c,source:`
@group(0) @binding(0) var<storage, read_write> cellCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellCursors: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${j(c)}
  if (index >= ${s.length}u) { return; }
  cellCounts[index] = 0u;
  cellCursors[index] = 0u;
}`}));for(let[a,o]of l.entries()){if(!o.length)continue;let c=A(o.length,n);r.push(b(t,{id:`${e.id}-count-${i}-${a}`,inputs:{positions:o},outputs:{cellCounts:s},dispatch:c,source:k(e,o,c,`@group(0) @binding(1) var<storage, read_write> cellCounts: array<atomic<u32>>;`,`
  if (accepted && cellIndex >= ${g}u && cellIndex - ${g}u < ${s.length}u) {
    atomicAdd(&cellCounts[cellIndex - ${g}u], 1u);
  }`)}))}g+=s.length}r.push(...s(new c({id:`${e.id}-scan`,input:a,output:i}),t,n));let x=a.data[a.data.length-1],S=i.data[i.data.length-1],T=v(t,e.cellOffsets,e.cellCount,1).data[0];r.push(b(t,{id:`${e.id}-finalize`,inputs:{counts:x,offsets:S},outputs:{count:e.count,overflow:e.overflow},dispatch:{x:1,y:1,z:1},workgroupSize:1,source:`
@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> count: array<u32>;
@group(0) @binding(3) var<storage, read_write> overflow: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let total = counts[${x.length-1}u] + offsets[${u(S)+S.length-1}u];
  count[${u(e.count)}u] = total;
  overflow[${u(e.overflow)}u] = select(0u, 1u, total > ${e.objectIds.length}u);
}`})),r.push(b(t,{id:`${e.id}-terminal`,inputs:{count:e.count},outputs:{terminal:T},dispatch:{x:1,y:1,z:1},workgroupSize:1,source:`
@group(0) @binding(0) var<storage, read> count: array<u32>;
@group(0) @binding(1) var<storage, read_write> terminal: array<u32>;
@compute @workgroup_size(1)
fn main() {
  terminal[${u(T)}u] = count[${u(e.count)}u];
}`})),g=0;for(let[a,s]of i.data.entries()){let i=o.data[a];for(let[o,c]of l.entries()){if(!c.length||!e.objectIds.length)continue;let l=A(c.length,n);r.push(b(t,{id:`${e.id}-rank-${a}-${o}`,inputs:{positions:c,cellOffsets:s},outputs:{cellCursors:i,ranks:d[o]},dispatch:l,source:k(e,c,l,`
@group(0) @binding(1) var<storage, read> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellCursors: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> ranks: array<u32>;`,`
  if (accepted && cellIndex >= ${g}u && cellIndex - ${g}u < ${s.length}u) {
    let cell = cellIndex - ${g}u;
    ranks[index] = cellOffsets[${u(s)}u + cell] + atomicAdd(&cellCursors[cell], 1u);
  }`)}))}g+=s.length}let E=0;for(let[i,a]of l.entries())a.length&&e.objectIds.length&&r.push(...w(new C({id:`${e.id}-scatter-${i}`,source:e.sourceIds?v(t,e.sourceIds,E,a.length):f[i],indices:d[i],output:e.objectIds}),t,n)),E+=a.length;return r}function k(e,t,n,r,a){let o=e.gridSize[0],s=e.gridSize[1],c=e.dimension===3?e.gridSize[2]:1,l=e.dimension,d=e.dimension===3?`let z = positions[POSITIONS_OFFSET + index * 3u + 2u];
    let finiteZ = z == z && abs(z) <= 3.402823466e+38;
    let inZ = z >= ${M(e.bounds[2])} && z <= ${M(e.bounds[5])};`:`let finiteZ = true;
    let inZ = true;`,f=e.dimension===3?`let layer = getCoordinate(z, ${M(e.bounds[2])}, ${M(e.bounds[5])}, DEPTH);`:`let layer = 0u;`;return`
const ELEMENT_COUNT: u32 = ${t.length}u;
const POSITIONS_OFFSET: u32 = ${u(t)}u;
const WIDTH: u32 = ${o}u;
const HEIGHT: u32 = ${s}u;
const DEPTH: u32 = ${c}u;
@group(0) @binding(0) var<storage, read> positions: array<f32>;
${r}

${i}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${j(n)}
  var accepted = false;
  var cellIndex = 0u;
  if (index < ELEMENT_COUNT) {
    let x = positions[POSITIONS_OFFSET + index * ${e.dimension}u];
    let y = positions[POSITIONS_OFFSET + index * ${e.dimension}u + 1u];
    ${d}
    let finiteXY = x == x && y == y && abs(x) <= 3.402823466e+38 && abs(y) <= 3.402823466e+38;
    let inX = x >= ${M(e.bounds[0])} && x <= ${M(e.bounds[l])};
    let inY = y >= ${M(e.bounds[1])} && y <= ${M(e.bounds[l+1])};
    if (finiteXY && finiteZ && inX && inY && inZ) {
      accepted = true;
      let column = getCoordinate(x, ${M(e.bounds[0])}, ${M(e.bounds[l])}, WIDTH);
      let row = getCoordinate(y, ${M(e.bounds[1])}, ${M(e.bounds[l+1])}, HEIGHT);
      ${f}
      cellIndex = (layer * HEIGHT + row) * WIDTH + column;
    }
  }
  ${a}
}`}function A(e,t){return o(`GPUGridIndex`,e,D,t)}function j(e){return a(e,D)}function M(e){let t=`${Math.fround(e)}`;return t.includes(`.`)||t.includes(`e`)?t:`${t}.0`}var N=4294967295,P=class{id;positions;sourceIds;firstSourceIndex;gridSize;bounds;cellOffsets;objectIds;count;overflow;dimension;cellCount;updatePolicy=`rebuild`;constructor(e){this.id=e.id??`gpu-grid-index`,this.positions=e.positions,this.sourceIds=e.sourceIds,this.firstSourceIndex=e.firstSourceIndex??0,this.gridSize=e.gridSize,this.bounds=e.bounds,this.cellOffsets=e.cellOffsets,this.objectIds=e.objectIds,this.count=e.count,this.overflow=e.overflow,this.dimension=this.positions.format===`float32x2`?2:3;let t=this.dimension===2?`float32x2`:`float32x3`;for(let e of L(this.positions))d(e,[t],`${this.id} positions`);if(this.gridSize.length!==this.dimension||this.bounds.length!==this.dimension*2)throw Error(`${this.id} positions, gridSize, and bounds must have matching dimensions`);if(this.gridSize.some(e=>!Number.isSafeInteger(e)||e<=0))throw Error(`${this.id} gridSize must contain positive integers`);if(this.cellCount=this.gridSize.reduce((e,t)=>e*t,1),!Number.isSafeInteger(this.cellCount)||this.cellCount>=N)throw Error(`${this.id} cell count must fit in uint32`);if(!this.bounds.every(Number.isFinite)||Array.from({length:this.dimension},(e,t)=>t).some(e=>this.bounds[e]>this.bounds[e+this.dimension]))throw Error(`${this.id} bounds must contain finite ordered minima and maxima`);for(let e of R(this.cellOffsets))h(e,`${this.id} cellOffsets`);for(let e of R(this.objectIds))h(e,`${this.id} objectIds`);if(h(this.count,`${this.id} count`),h(this.overflow,`${this.id} overflow`),this.cellOffsets.length!==this.cellCount+1)throw Error(`${this.id} cellOffsets.length must equal cellCount + 1`);if(this.count.length<1||this.overflow.length<1)throw Error(`${this.id} count and overflow must each contain one uint32 row`);if(this.positions.length>N)throw Error(`${this.id} position count must fit in uint32`);if(!Number.isSafeInteger(this.firstSourceIndex)||this.firstSourceIndex<0||this.firstSourceIndex+this.positions.length>4294967296)throw Error(`${this.id} generated source IDs must fit in uint32`);if(this.sourceIds&&e.firstSourceIndex!==void 0)throw Error(`${this.id} sourceIds and firstSourceIndex are mutually exclusive`);F(this.id,this.positions,this.sourceIds),I(this.id,this.positions,this.sourceIds,this.objectIds)}getCommandNodes(e){let t=[];return t.push(...O(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)),t}};function F(e,t,n){if(n){for(let t of R(n))h(t,`${e} sourceIds`);if(t.length!==n.length)throw Error(`${e} sourceIds.length must equal positions.length`)}}function I(e,t,n,r){for(let n of L(t))if(R(r).some(e=>f(n,e)))throw Error(`${e} positions and objectIds must not overlap`);for(let t of R(n))if(R(r).some(e=>f(t,e)))throw Error(`${e} sourceIds and objectIds must not overlap`)}function L(e){return e instanceof n?e.data:[e]}function R(e){return e?e instanceof n?e.data:[e]:[]}export{P as t};
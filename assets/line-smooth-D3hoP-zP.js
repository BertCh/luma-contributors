import{r as e}from"./geometry-storage-validation-T89ZhmCj.js";import{v as t}from"./spatial-context-DwZEd4EZ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{E as i,f as a,v as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{a as c}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as l}from"./validate-outline-output-passes-DIcqH6Pw.js";import{o as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as d,s as f}from"./line-segmentize-kernels-J-DWl7up.js";import{n as p,r as m,t as h}from"./line-geometry-output-Eyy3fjKL.js";import{t as g}from"./define-run-B-brkCe4.js";import{d as _,s as v}from"./pointer-overlay-Qh8qflcA.js";import{a as y,c as b,f as x,l as S,n as C,r as w}from"./lines-kx5qHdti.js";import{Uo as T}from"./entry-preview-D70TdSbK.js";function E(e,t){return u(e,{id:`${t.id}-metadata-prepare`,operation:`GPULineSmooth`,variant:`metadata-prepare`,invocationCount:t.positions.length,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`rowMeasures`,view:t.rowMeasures,type:`f32`,access:`read`},{name:`destination`,view:t.destination,type:`u32`,access:`read_write`}],declarations:`const LINE_COUNT: u32 = ${t.lineOffsets.length-1}u;
const CLOSED: bool = ${t.closed};
fn point(row: u32) -> vec2f { return vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]); }`,body:`var low = 0u; var high = LINE_COUNT;
while (high - low > 1u) { let middle = low + (high - low) / 2u; if (lineOffsets[lineOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; } }
let first = lineOffsets[lineOffsetsOffset + low];
let last = min(lineOffsets[lineOffsetsOffset + low + 1u], (arrayLength(&positions) - positionsOffset) / 2u);
var total = 0.0;
if (last > first) {
  total = rowMeasures[rowMeasuresOffset + last - 1u];
  if (CLOSED) { total += length(point(first) - point(last - 1u)); }
}
let base = destinationOffset + index * 4u;
destination[base] = index - first;
destination[base + 1u] = 0u;
destination[base + 2u] = bitcast<u32>(rowMeasures[rowMeasuresOffset + index]);
destination[base + 3u] = bitcast<u32>(total);`})}function D(e,t){let{output:n}=t;return u(e,{id:`${t.id}-metadata-publish`,operation:`GPULineSmooth`,variant:`metadata-publish`,invocationCount:n.positions.length,bindings:[{name:`metadata`,view:t.metadata,type:`u32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`outputOffsets`,view:n.lineOffsets,type:`u32`,access:`read`},{name:`smoothStarts`,view:t.smoothStarts,type:`u32`,access:`read`},...n.sourceRows?[{name:`sourceRows`,view:n.sourceRows,type:`u32`,access:`read_write`}]:[],...n.measures?[{name:`measures`,view:n.measures,type:`f32`,access:`read_write`}]:[]],declarations:`const LINE_COUNT: u32 = ${t.lineOffsets.length-1}u;
const CLOSED: bool = ${t.closed};
const SCALE: u32 = ${2**t.iterationCount}u;`,body:`let lineCount = LINE_COUNT;
var sourceRow = 0xffffffffu; var measure = 0.0;
if (index < outputOffsets[outputOffsetsOffset + lineCount]) {
  var low = 0u; var high = lineCount;
  while (high - low > 1u) { let middle = low + (high - low) / 2u; if (outputOffsets[outputOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; } }
  let base = metadataOffset + index * 4u;
  sourceRow = lineOffsets[lineOffsetsOffset + low] + metadata[base];
  measure = bitcast<f32>(metadata[base + 2u]);
  let smoothCount = smoothStarts[smoothStartsOffset + low + 1u] - smoothStarts[smoothStartsOffset + low];
  let first = outputOffsets[outputOffsetsOffset + low];
  let firstMeasure = bitcast<f32>(metadata[metadataOffset + first * 4u + 2u]);
  if (CLOSED && smoothCount > 0u && (measure < firstMeasure || index - first == smoothCount * SCALE)) { measure += bitcast<f32>(metadata[base + 3u]); }
}
${n.sourceRows?`sourceRows[sourceRowsOffset + index] = sourceRow;`:``}
${n.measures?`measures[measuresOffset + index] = measure;`:``}`})}function O(e,t){let n=t.lineOffsets.length-1;return u(e,{id:t.id,operation:t.operation,variant:`smooth-prepare`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`smoothCounts`,view:t.smoothCounts,type:`u32`,access:`read_write`},{name:`plainCounts`,view:t.plainCounts,type:`u32`,access:`read_write`},{name:`smoothFlags`,view:t.smoothFlags,type:`u32`,access:`read_write`}],invocationCount:n+1,declarations:`const LINE_COUNT: u32 = ${n}u;
const ROW_COUNT: u32 = ${t.positions.length}u;
const CLOSED: bool = ${t.closed};

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,body:`var smoothCount = 0u;
  var plainCount = 0u;
  if (index < LINE_COUNT) {
    let start = min(lineOffsets[lineOffsetsOffset + index], ROW_COUNT);
    let end = min(max(lineOffsets[lineOffsetsOffset + index + 1u], start), ROW_COUNT);
    let count = end - start;
    var effectiveCount = count;
    if (CLOSED && count >= 2u && all(getPosition(start) == getPosition(end - 1u))) {
      effectiveCount = count - 1u;
    }
    let smoothable = select(count >= 2u, effectiveCount >= 3u, CLOSED);
    smoothCount = select(0u, effectiveCount, smoothable);
    plainCount = select(count, 0u, smoothable);
  }
  smoothCounts[smoothCountsOffset + index] = smoothCount;
  plainCounts[plainCountsOffset + index] = plainCount;
  smoothFlags[smoothFlagsOffset + index] = select(0u, 1u, smoothCount > 0u);`})}var k=`
fn getLevelOffset(line: u32, scale: u32, withClosing: bool) -> u32 {
  var offset = smoothStarts[smoothStartsOffset + line] * scale + plainStarts[plainStartsOffset + line];
  if (withClosing) {
    offset += getClosingStart(line);
  }
  return offset;
}
`;function A(e,t){let n=t.closed,r=[{name:`source`,view:t.source,type:t.metadata?`u32`:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`smoothStarts`,view:t.smoothStarts,type:`u32`,access:`read`},{name:`plainStarts`,view:t.plainStarts,type:`u32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`destination`,view:t.destination,type:t.metadata?`u32`:`f32`,access:`read_write`},{name:`levelState`,view:t.levelState,type:`u32`,access:`read`}];n&&t.smoothFlagStarts&&r.push({name:`smoothFlagStarts`,view:t.smoothFlagStarts,type:`u32`,access:`read`});let i=t.destination.length;return u(e,{id:t.id,operation:t.operation,variant:t.metadata?`smooth-level-metadata`:`smooth-level`,bindings:r,invocationCount:i,declarations:`const LINE_COUNT: u32 = ${t.lineCount}u;
const CAPACITY: u32 = ${i}u;
const CLOSED: bool = ${t.closed};

fn getClosingStart(line: u32) -> u32 {
  ${n?`return smoothFlagStarts[smoothFlagStartsOffset + line];`:`return 0u;`}
}
${k}

fn getSource(row: u32) -> ${t.metadata?`vec4f`:`vec2f`} {
  ${t.metadata?`let base = sourceOffset + 4u * row; return vec4f(0.0, bitcast<f32>(source[base + 1u]), bitcast<f32>(source[base + 2u]), bitcast<f32>(source[base + 3u]));`:`return vec2f(source[sourceOffset + 2u * row], source[sourceOffset + 2u * row + 1u]);`}
}`,body:`// The level lives in a GPU word, so no pipeline depends on the level (rule 12).
  let currentLevel = levelState[levelStateOffset];
  let IS_FIRST_LEVEL = currentLevel == 0u;
  let WITH_CLOSING = CLOSED && currentLevel + 1u == levelState[levelStateOffset + 1u];
  let SOURCE_LENGTH = (arrayLength(&source) - sourceOffset) / ${t.metadata?4:2}u;
  let sourceScale = 1u << currentLevel;
  let destinationScale = sourceScale << 1u;
  let total = getLevelOffset(LINE_COUNT, destinationScale, WITH_CLOSING);
  if (index >= total) {
    return;
  }
  // Largest line whose destination offset is at most index.
  var low = 0u;
  var high = LINE_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (getLevelOffset(middle, destinationScale, WITH_CLOSING) <= index) {
      low = middle;
    } else {
      high = middle;
    }
  }
  let line = low;
  let local = index - getLevelOffset(line, destinationScale, WITH_CLOSING);
  let smoothCount = smoothStarts[smoothStartsOffset + line + 1u] - smoothStarts[smoothStartsOffset + line];
  var sourceStart = getLevelOffset(line, sourceScale, false);
  if (IS_FIRST_LEVEL) {
    sourceStart = lineOffsets[lineOffsetsOffset + line];
  }
  var vertex: ${t.metadata?`vec4f`:`vec2f`};
  ${t.metadata?`var sourceRow = 0u;`:``}
  if (smoothCount == 0u) {
    // Copied line.
    if (sourceStart + local >= SOURCE_LENGTH) {
      return;
    }
    vertex = getSource(sourceStart + local);
    ${t.metadata?`sourceRow = source[sourceOffset + 4u * (sourceStart + local)];`:``}
  } else {
    let count = smoothCount * sourceScale;
    let ratio = parameters[parametersOffset];
    var edge = 0u;
    var isSecond = false;
    var copyRow = 0xffffffffu;
    if (CLOSED) {
      let wrapped = select(local, 0u, local == 2u * count);
      edge = wrapped / 2u;
      isSecond = (wrapped & 1u) == 1u;
    } else if (local == 0u) {
      copyRow = 0u;
    } else if (local == 2u * count - 1u) {
      copyRow = count - 1u;
    } else {
      edge = (local - 1u) / 2u;
      isSecond = ((local - 1u) & 1u) == 1u;
    }
    if (copyRow != 0xffffffffu) {
      if (sourceStart + copyRow >= SOURCE_LENGTH) {
        return;
      }
      vertex = getSource(sourceStart + copyRow);
      ${t.metadata?`sourceRow = source[sourceOffset + 4u * (sourceStart + copyRow)];`:``}
    } else {
      let nextEdge = select(edge + 1u, 0u, edge + 1u == count);
      if (sourceStart + max(edge, nextEdge) >= SOURCE_LENGTH) {
        return;
      }
      let start = getSource(sourceStart + edge);
      var end = getSource(sourceStart + nextEdge);
      ${t.metadata?`let startRow = source[sourceOffset + 4u * (sourceStart + edge)];
      let endRow = source[sourceOffset + 4u * (sourceStart + nextEdge)];
      var rowDistance = endRow - startRow;
      let wrapped = CLOSED && (endRow < startRow || (endRow == startRow && end.y < start.y));
      if (wrapped) { rowDistance = smoothCount - startRow + endRow; end.z += start.w; }
      end.y += f32(rowDistance);`:``}
      let weight = select(ratio, 1.0 - ratio, isSecond);
      vertex = start * (1.0 - weight) + end * weight;
      ${t.metadata?`vertex.w = start.w;
      let advance = u32(floor(vertex.y));
      sourceRow = startRow + advance;
      vertex.y -= f32(advance);
      if (CLOSED && sourceRow >= smoothCount) { sourceRow -= smoothCount; vertex.z -= vertex.w; }`:``}
    }
  }
  if (index < CAPACITY) {
    ${t.metadata?`destination[destinationOffset + 4u * index] = sourceRow;
    destination[destinationOffset + 4u * index + 1u] = bitcast<u32>(vertex.y);
    destination[destinationOffset + 4u * index + 2u] = bitcast<u32>(vertex.z);
    destination[destinationOffset + 4u * index + 3u] = bitcast<u32>(vertex.w);`:`destination[destinationOffset + 2u * index] = vertex.x;
    destination[destinationOffset + 2u * index + 1u] = vertex.y;`}
  }`})}function j(e,t){let n=[{name:`smoothStarts`,view:t.smoothStarts,type:`u32`,access:`read`},{name:`plainStarts`,view:t.plainStarts,type:`u32`,access:`read`}];return t.closed&&t.smoothFlagStarts&&n.push({name:`smoothFlagStarts`,view:t.smoothFlagStarts,type:`u32`,access:`read`}),n.push({name:`outputLineOffsets`,view:t.outputLineOffsets,type:`u32`,access:`read_write`},{name:`overflowOut`,view:t.overflow,type:`u32`,access:`read_write`}),t.vertexCount&&n.push({name:`vertexCountOut`,view:t.vertexCount,type:`u32`,access:`read_write`}),t.requiredVertexCount&&n.push({name:`totalOut`,view:t.requiredVertexCount,type:`u32`,access:`read_write`}),u(e,{id:t.id,operation:t.operation,variant:`smooth-publish`,bindings:n,invocationCount:t.lineCount+1,declarations:`const LINE_COUNT: u32 = ${t.lineCount}u;
const CAPACITY: u32 = ${t.capacity}u;
const SCALE: u32 = ${2**t.iterationCount}u;

fn getClosingStart(line: u32) -> u32 {
  ${t.closed?`return smoothFlagStarts[smoothFlagStartsOffset + line];`:`return 0u;`}
}
${k}`,body:`let total = getLevelOffset(LINE_COUNT, SCALE, true);
  outputLineOffsets[outputLineOffsetsOffset + index] = min(getLevelOffset(index, SCALE, true), CAPACITY);
  if (index == 0u) {
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
    ${t.vertexCount?`vertexCountOut[vertexCountOutOffset] = min(total, CAPACITY);`:``}
    ${t.requiredVertexCount?`totalOut[totalOutOffset] = total;`:``}
  }`})}function M(e,t){return u(e,{id:t.id,operation:t.operation,variant:t.reset?`smooth-level-reset`:`smooth-level-advance`,bindings:[{name:`levelState`,view:t.levelState,type:`u32`,access:`read_write`}],invocationCount:1,body:t.reset?`levelState[levelStateOffset] = 0u; levelState[levelStateOffset + 1u] = ${t.iterationCount??1}u;`:`levelState[levelStateOffset] = levelState[levelStateOffset] + 1u;`})}var N=`GPULineSmooth`,P=2147483647,F=1073741823,I=4294967295;function L(e={},t=new Float32Array(4)){if(t.length<4)throw Error(`Line smooth target must hold 4 elements`);let n=e.ratio??.25;if(!(n>0&&n<=.5))throw Error(`Line smooth ratio must be in (0, 0.5]`);return t.set([n,0,0,0]),t}var R=class{id;props;spatialContext;closed;constructor(n){this.id=n.id??`line-smooth`,this.props=n,this.closed=n.closed??!1,this.spatialContext=t(this.id,n.spatialContext);let{id:r}=this,{iterationCount:o}=n;a(r,`iterationCount`,o,1,10),i(n.positions,[`float32x2`],`${r} positions`);let c=n.positions.length;if(c<1)throw Error(`${r} needs at least one position`);if(i(n.lineOffsets,[`uint32`],`${r} lineOffsets`),n.lineOffsets.length<2)throw Error(`${r} lineOffsets needs two rows`);let u=n.lineOffsets.length-1;if(a(r,`positions.length * 2^iterationCount + lineCount`,c*2**o+u,1,I),i(n.parameters,[`float32`],`${r} parameters`),n.parameters.length<4)throw Error(`${r} parameters must hold 4 float32 values`);m(r,n.output,u),a(r,`output.positions.length`,n.output.positions.length,1,P),(n.output.sourceRows||n.output.measures)&&a(r,`metadata row count`,Math.max(c,n.output.positions.length),1,F),l(r,{positions:n.output.positions,lineOffsets:n.output.lineOffsets,...n.output.status,sourceRows:n.output.sourceRows,measures:n.output.measures,sourceLineIndices:n.output.sourceLineIndices},[[`lineOffsets`,`overflow`,`vertexCount`,`requiredVertexCount`],[`count`,`requiredCount`],[`lineOffsets`,`sourceRows`,`measures`],[`positions`],[`sourceLineIndices`]]),a(r,`level position count`,n.output.positions.length+Number(this.closed&&n.iterationCount>1),1,P),(n.output.sourceRows||n.output.measures)&&a(r,`level metadata row count`,n.output.positions.length+Number(this.closed&&n.iterationCount>1),1,F),e(r,[n.output.lineOffsets],[n.output.sourceRows,n.output.measures]),s(r,p(n.output),[n.positions,n.lineOffsets,n.parameters])}getCommandNodes(e){let{props:t,id:i,closed:s}=this,{output:l,iterationCount:m}=t;c(e,i,{positions:t.positions,lineOffsets:t.lineOffsets,parameters:t.parameters,...Object.fromEntries(p(l).map((e,t)=>[`output.${t}`,e]))});let g=l.positions.length+Number(s&&m>1),_=!!(l.sourceRows||l.measures);for(let[n,r]of Object.entries({lineColumnsAndScan:t.lineOffsets.length*4,inputCopy:t.positions.length*8,levelPositions:g*8,..._?{inputMetadata:t.positions.length*16,levelMetadata:g*16,sourceMeasures:t.positions.length*4}:{}}))o(e.device,i,n,r,`split lines or reduce output capacity`),a(i,`${n} byteLength`,r,1,e.device.limits.maxBufferSize);let v=t.lineOffsets.length-1,y=l.positions.length,b=t=>n(e,`${i}-${t}`,`uint32`,v+1),x=b(`smooth-counts`),S=b(`plain-counts`),C=b(`smooth-flags`),w=b(`smooth-starts`),T=b(`plain-starts`),k=s?b(`smooth-flag-starts`):void 0,P=[O(e,{id:`${i}-prepare`,operation:N,positions:t.positions,lineOffsets:t.lineOffsets,closed:s,smoothCounts:x,plainCounts:S,smoothFlags:C}),...new r({id:`${i}-smooth-scan`,input:x,output:w}).getCommandNodes(e),...new r({id:`${i}-plain-scan`,input:S,output:T}).getCommandNodes(e)];k&&P.push(...new r({id:`${i}-flag-scan`,input:C,output:k}).getCommandNodes(e));let F=n(e,`${i}-input-copy`,`float32x2`,t.positions.length),I=[0,1].map(t=>n(e,`${i}-level-${t}`,`float32x2`,g)),L=n(e,`${i}-level-state`,`uint32`,2),R=(t,n,r)=>u(e,{id:`${i}-${t}`,operation:N,variant:`copy-positions`,invocationCount:r.length,bindings:[{name:`source`,view:n,type:`f32`,access:`read`},{name:`destination`,view:r,type:`f32`,access:`read_write`}],body:`destination[destinationOffset + index * 2u] = source[sourceOffset + index * 2u]; destination[destinationOffset + index * 2u + 1u] = source[sourceOffset + index * 2u + 1u];`});P.push(R(`copy-input`,t.positions,F));let z=_?[0,1].map(t=>n(e,`${i}-metadata-${t}`,`uint32x4`,g)):[],B=_?n(e,`${i}-input-metadata`,`uint32x4`,t.positions.length):void 0;if(B){let r=n(e,`${i}-source-measures`,`float32`,t.positions.length);P.push(f(e,{id:i,operation:N,positions:t.positions,lineOffsets:t.lineOffsets,coordinateSystem:`planar`,radius:1,rowMeasures:r})),P.push(E(e,{id:i,positions:t.positions,lineOffsets:t.lineOffsets,rowMeasures:r,destination:B,closed:s}))}P.push(M(e,{id:`${i}-level-reset`,operation:N,levelState:L,iterationCount:m,reset:!0}));let V=F;for(let n=0;n<m;n++){let r=n===m-1,a=I[n%2];if(P.push(A(e,{id:`${i}-level-${n}`,operation:N,level:n,levelState:L,isFinal:r,closed:s,lineCount:v,source:V,lineOffsets:t.lineOffsets,smoothStarts:w,plainStarts:T,smoothFlagStarts:k,parameters:t.parameters,destination:a})),B){let a=z[n%2];P.push(A(e,{id:`${i}-metadata-level-${n}`,operation:N,level:n,levelState:L,isFinal:r,closed:s,lineCount:v,source:B,destination:a,metadata:!0,lineOffsets:t.lineOffsets,smoothStarts:w,plainStarts:T,smoothFlagStarts:k,parameters:t.parameters})),B=a}r||P.push(M(e,{id:`${i}-level-${n}-advance`,operation:N,levelState:L,reset:!1})),V=a}return P.push(j(e,{id:`${i}-publish`,operation:N,iterationCount:m,closed:s,lineCount:v,smoothStarts:w,plainStarts:T,smoothFlagStarts:k,capacity:y,outputLineOffsets:l.lineOffsets,overflow:l.status.overflow,vertexCount:l.status.vertexCount,requiredVertexCount:l.status.requiredVertexCount}),h(e,{id:i,operation:N,output:l,lineCapacity:v})),P.push(R(`copy-output`,V,l.positions)),B&&P.push(D(e,{id:i,metadata:B,lineOffsets:t.lineOffsets,smoothStarts:w,output:l,closed:s,iterationCount:m})),l.sourceLineIndices&&P.push(d(e,{id:`${i}-source-line-indexs`,operation:N,sourceLineIndices:l.sourceLineIndices})),P}},z=g({parameters:T,caption:`Chaikin corner cutting: coarse polylines (muted) and their smoothed output.`,createInputs:({random:e,world:t})=>b(e,t,3,7),build(e,t,n){let r=e.upload(`positions`,`float32x2`,t.positions),i=e.upload(`line-offsets`,`uint32`,t.lineOffsets),a=y(e,`smoothed`,4e3,t.lineCount),o=e.allocate(`smooth-parameters`,`float32`,4);e.graph.add(new R({positions:r.view,lineOffsets:i.view,iterationCount:n.iterationCount,parameters:o.view,output:a.output}));let s=w(e,`smoothed`,a.positions,a.lineOffsets);return{outputs:{original:e.upload(`original`,`float32x4`,S(t.positions,t.lineOffsets,!1)),originalPoints:e.upload(`original-points`,`float32x2`,t.positions),...s},update:e=>o.write(L({ratio:e.ratio}))}},rebuildOn:[`iterationCount`],interaction:{kind:`reach`,hint:`slide along an edge`,rest:({inputs:e,toUnit:t})=>{let n=v(T,`ratio`).default,r=e.lineOffsets[0]+2,i=e.positions;return t([i[r*2]+(i[r*2+2]-i[r*2])*n,i[r*2+1]+(i[r*2+3]-i[r*2+1])*n])},resolve(e,{inputs:t,values:n}){let r=x(t,e.x,e.y);if(!r)return{};let i=_(v(T,`ratio`),Math.min(r.fraction,1-r.fraction)),a=t.positions,o=a[r.row*2],s=a[r.row*2+1],c=a[r.row*2+2],l=a[r.row*2+3],u=[o+(c-o)*i,s+(l-s)*i],d=[c+(o-c)*i,l+(s-l)*i];return{values:{ratio:i},glyphs:[{shape:`line`,from:u,to:d,dashed:!0},{shape:`ring`,at:r.fraction<.5?u:d,radius:.02},{shape:`dot`,at:r.fraction<.5?d:u}],readout:`ratio ${i.toFixed(2)} · ${n.iterationCount} iterations`}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.original},{layer:`dots`,context:!0,positions:e.originalPoints,radius:5},{layer:`lines`,segments:e.segments,color:C,radius:3.2}]});export{z as default};
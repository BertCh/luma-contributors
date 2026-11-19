import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-scan-B4NNTfF9.js";import{E as n,v as r,y as i}from"./gpu-contributor-utils-BIwLQON5.js";import{a,m as o,v as s,y as c}from"./spatial-context-DwZEd4EZ.js";import{d as l,s as u}from"./line-segmentize-kernels-J-DWl7up.js";import{r as d}from"./scan-total-C4QddMmi.js";import{o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as p}from"./geodesic-wgsl-I0CYDKcz.js";import{n as m,r as h,t as g}from"./line-geometry-output-Eyy3fjKL.js";import{a as _}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as v}from"./geometry-storage-validation-T89ZhmCj.js";import{t as y}from"./define-run-B-brkCe4.js";import{d as b,s as x}from"./pointer-overlay-Qh8qflcA.js";import{_ as S,a as C,c as w,d as T,f as E,l as D,r as O}from"./lines-kx5qHdti.js";import{Yo as k}from"./entry-preview-D70TdSbK.js";var A=1;function j(e,t){return f(e,{id:t.id,operation:t.operation,variant:`piece-count`,bindings:[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`rowMeasures`,view:t.rowMeasures,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`pieceCounts`,view:t.pieceCounts,type:`u32`,access:`read_write`},...t.requiredPieceCounts?[{name:`requiredPieceCounts`,view:t.requiredPieceCounts,type:`u32`,access:`read_write`}]:[]],invocationCount:t.lineOffsets.length-1,declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const MAXIMUM_PIECES: u32 = ${t.maximumPieces}u;
const IS_CHUNK: bool = ${t.mode===`chunk`};`,body:`let start = min(lineOffsets[lineOffsetsOffset + index], ROW_COUNT);
  let end = min(max(lineOffsets[lineOffsetsOffset + index + 1u], start), ROW_COUNT);
  var count = 0u;
  if (end > start) {
    count = 1u;
    let lineLength = rowMeasures[rowMeasuresOffset + end - 1u];
    let chunkLength = parameters[parametersOffset];
    if (IS_CHUNK && lineLength > 0.0 && chunkLength > 0.0) {
      let requirement = ceil(lineLength / chunkLength);
      // f32(0xffffffffu) rounds up to 2^32. Branch before conversion: converting that
      // value (or infinity from a tiny chunk length) to u32 is outside its range.
      count = 0xffffffffu;
      if (requirement < 4294967296.0) {
        count = max(u32(requirement), 1u);
      }
    }
  }
  ${t.requiredPieceCounts?`requiredPieceCounts[requiredPieceCountsOffset + index] = count;`:``}
  pieceCounts[pieceCountsOffset + index] = min(count, MAXIMUM_PIECES);`})}function M(e,t){let n=t.lineOffsets.length-1;return f(e,{id:t.id,operation:t.operation,variant:`piece-range`,bindings:[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`rowMeasures`,view:t.rowMeasures,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`pieceCounts`,view:t.pieceCounts,type:`u32`,access:`read`},{name:`pieceStarts`,view:t.pieceStarts,type:`u32`,access:`read`},{name:`pieces`,view:t.pieces,type:`u32`,access:`read_write`},{name:`vertexCounts`,view:t.vertexCounts,type:`u32`,access:`read_write`},{name:`pieceTotal`,view:t.pieceTotal,type:`u32`,access:`read_write`}],invocationCount:t.vertexCounts.length,declarations:`const LINE_COUNT: u32 = ${n}u;
const ROW_COUNT: u32 = ${t.rowCount}u;
const STRIDE: u32 = 10u;
const IS_CHUNK: bool = ${t.mode===`chunk`};
const PIECE_COPY: u32 = ${A}u;

fn getMeasure(row: u32) -> f32 {
  return rowMeasures[rowMeasuresOffset + row];
}

// First row in [low, high) whose measure is greater than (strict) or at least (!strict) the value.
fn findRow(low: u32, high: u32, value: f32, strict: bool) -> u32 {
  var first = low;
  var last = high;
  while (first < last) {
    let middle = (first + last) / 2u;
    let measure = getMeasure(middle);
    if ((strict && measure <= value) || (!strict && measure < value)) {
      first = middle + 1u;
    } else {
      last = middle;
    }
  }
  return first;
}`,body:`let total = pieceStarts[pieceStartsOffset + LINE_COUNT - 1u] + pieceCounts[pieceCountsOffset + LINE_COUNT - 1u];
  if (index == 0u) {
    pieceTotal[pieceTotalOffset] = total;
  }
  let base = piecesOffset + index * STRIDE;
  if (index >= total) {
    pieces[base] = 0xffffffffu;
    pieces[base + 6u] = 0u;
    vertexCounts[vertexCountsOffset + index] = 0u;
    return;
  }
  // Largest line whose first piece is at most index (lines without pieces are skipped).
  var low = 0u;
  var high = LINE_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (pieceStarts[pieceStartsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle;
    }
  }
  let line = low;
  let local = index - pieceStarts[pieceStartsOffset + line];
  let pieceCount = pieceCounts[pieceCountsOffset + line];
  let start = min(lineOffsets[lineOffsetsOffset + line], ROW_COUNT);
  let end = min(max(lineOffsets[lineOffsetsOffset + line + 1u], start), ROW_COUNT);
  let lineLength = getMeasure(end - 1u);
  var startMeasure = 0.0;
  var endMeasure = lineLength;
  var flags = 0u;
  var firstInterior = start;
  var interiorEnd = end;
  var vertexCount = end - start;
  if (!(lineLength > 0.0)) {
    flags = PIECE_COPY;
  } else {
    if (IS_CHUNK) {
      let chunkLength = parameters[parametersOffset];
      if (chunkLength > 0.0) {
        startMeasure = f32(local) * chunkLength;
        if (local + 1u < pieceCount) {
          endMeasure = f32(local + 1u) * chunkLength;
        }
      }
    } else {
      startMeasure = clamp(parameters[parametersOffset + 1u], 0.0, lineLength);
      endMeasure = clamp(parameters[parametersOffset + 2u], 0.0, lineLength);
    }
    if (startMeasure > endMeasure) {
      vertexCount = 0u;
      firstInterior = start;
      interiorEnd = start;
    } else {
      firstInterior = findRow(start, end, startMeasure, true);
      interiorEnd = max(findRow(start, end, endMeasure, false), firstInterior);
      vertexCount = 2u + interiorEnd - firstInterior;
    }
  }
  pieces[base] = line;
  pieces[base + 1u] = bitcast<u32>(startMeasure);
  pieces[base + 2u] = bitcast<u32>(endMeasure);
  pieces[base + 3u] = flags;
  pieces[base + 4u] = firstInterior;
  pieces[base + 5u] = interiorEnd;
  pieces[base + 6u] = vertexCount;
  pieces[base + 7u] = start;
  pieces[base + 8u] = end;
  vertexCounts[vertexCountsOffset + index] = vertexCount;`})}function N(e,t){let n=t.coordinateSystem===`spherical`,r=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`rowMeasures`,view:t.rowMeasures,type:`f32`,access:`read`},{name:`pieces`,view:t.pieces,type:`u32`,access:`read`},{name:`vertexStarts`,view:t.vertexStarts,type:`u32`,access:`read`},{name:`outputPositions`,view:t.outputPositions,type:`f32`,access:`read_write`}];return n&&t.rowShifts&&r.push({name:`rowShifts`,view:t.rowShifts,type:`f32`,access:`read`}),t.outputMeasures&&r.push({name:`outputMeasures`,view:t.outputMeasures,type:`f32`,access:`read_write`}),t.outputSourceRows&&r.push({name:`outputSourceRows`,view:t.outputSourceRows,type:`u32`,access:`read_write`}),f(e,{id:t.id,operation:t.operation,variant:`piece-emit`,bindings:r,invocationCount:t.outputPositions.length,workgroupSize:128,guardIndex:!1,declarations:`const CAPACITY: u32 = ${t.outputPositions.length}u;
const STRIDE: u32 = 10u;
const PIECE_COPY: u32 = ${A}u;
${n?p:``}
${l(t.vertexStarts.length)}
fn balancedStart(piece: u32) -> u32 { return vertexStarts[vertexStartsOffset + piece]; }
fn balancedCount(piece: u32) -> u32 { return pieces[piecesOffset + piece * STRIDE + 6u]; }

fn getMeasure(row: u32) -> f32 {
  return rowMeasures[rowMeasuresOffset + row];
}

// Position of an input row, with spherical longitudes unwrapped along the line.
fn getVertex(row: u32) -> vec2<f32> {
  var vertex = vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
  ${n&&t.rowShifts?`vertex.x = vertex.x + rowShifts[rowShiftsOffset + row];`:``}
  return vertex;
}

fn interpolate(segmentRow: u32, measure: f32) -> vec2<f32> {
  let start = getVertex(segmentRow);
  let end = getVertex(segmentRow + 1u);
  let segmentMeasure = getMeasure(segmentRow + 1u) - getMeasure(segmentRow);
  var fraction = 0.0;
  if (segmentMeasure > 0.0) {
    fraction = clamp((measure - getMeasure(segmentRow)) / segmentMeasure, 0.0, 1.0);
  }
  ${n?`return geodesicInterpolate(start, end, geodesicCentralAngle(start, end), fraction);`:`return mix(start, end, fraction);`}
}

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  outputPositions[outputPositionsOffset + 2u * outputRow] = vertex.x;
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${t.outputMeasures?`outputMeasures[outputMeasuresOffset + outputRow] = measure;`:``}
  ${t.outputSourceRows?`outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;`:``}
}`,body:`let piece = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (piece == BALANCED_NONE) {
    return;
  }
  let base = piecesOffset + piece * STRIDE;
  let local = index - vertexStarts[vertexStartsOffset + piece];
  let flags = pieces[base + 3u];
  let firstInterior = pieces[base + 4u];
  let interiorEnd = pieces[base + 5u];
  if ((flags & PIECE_COPY) != 0u) {
    let row = firstInterior + local;
    writeVertex(index, getVertex(row), getMeasure(row), row);
    return;
  }
  let lineStart = pieces[base + 7u];
  let lineEnd = pieces[base + 8u];
  let startMeasure = bitcast<f32>(pieces[base + 1u]);
  let endMeasure = bitcast<f32>(pieces[base + 2u]);
  // A measure lies on the segment ending at the first row past it, clamped to the line's
  // segments (the line has at least two rows because its length is positive).
  let startSegment = clamp(firstInterior, lineStart + 1u, lineEnd - 1u) - 1u;
  let endSegment = clamp(interiorEnd, lineStart + 1u, lineEnd - 1u) - 1u;
  if (local == 0u) {
    writeVertex(index, interpolate(startSegment, startMeasure), startMeasure, startSegment);
  } else if (local <= interiorEnd - firstInterior) {
    let row = firstInterior + local - 1u;
    writeVertex(index, getVertex(row), getMeasure(row), row);
  } else {
    writeVertex(index, interpolate(endSegment, endMeasure), endMeasure, endSegment);
  }`})}function P(e,t){let n=t.vertexStarts.length,r=[{name:`pieceTotal`,view:t.pieceTotal,type:`u32`,access:`read`},{name:`pieces`,view:t.pieces,type:`u32`,access:`read`},{name:`vertexStarts`,view:t.vertexStarts,type:`u32`,access:`read`},{name:`outputLineOffsets`,view:t.outputLineOffsets,type:`u32`,access:`read_write`},{name:`overflowOut`,view:t.overflow,type:`u32`,access:`read_write`}];return t.vertexCount&&r.push({name:`vertexCountOut`,view:t.vertexCount,type:`u32`,access:`read_write`}),t.requiredVertexCount&&r.push({name:`totalOut`,view:t.requiredVertexCount,type:`u32`,access:`read_write`}),f(e,{id:t.id,operation:t.operation,variant:`piece-publish`,bindings:r,invocationCount:n+1,declarations:`const LINE_CAPACITY: u32 = ${n}u;
const CAPACITY: u32 = ${t.capacity}u;
const STRIDE: u32 = 10u;`,body:`let lastBase = piecesOffset + (LINE_CAPACITY - 1u) * STRIDE;
  let total = vertexStarts[vertexStartsOffset + LINE_CAPACITY - 1u] + pieces[lastBase + 6u];
  var start = total;
  if (index < LINE_CAPACITY) {
    start = vertexStarts[vertexStartsOffset + index];
  }
  outputLineOffsets[outputLineOffsetsOffset + index] = min(start, CAPACITY);
  if (index == 0u) {
    let requiredPieces = pieceTotal[pieceTotalOffset];
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY || requiredPieces > LINE_CAPACITY);
    ${t.vertexCount?`vertexCountOut[vertexCountOutOffset] = min(total, CAPACITY);`:``}
    ${t.requiredVertexCount?`totalOut[totalOutOffset] = total;`:``}
  }`})}function F(e,t){return f(e,{id:t.id,operation:t.operation,variant:`piece-source-line-indexs`,bindings:[{name:`pieces`,view:t.pieces,type:`u32`,access:`read`},{name:`sourceLineIndices`,view:t.sourceLineIndices,type:`u32`,access:`read_write`}],invocationCount:t.sourceLineIndices.length,declarations:`const STRIDE: u32 = 10u;`,body:`sourceLineIndices[sourceLineIndicesOffset + index] = pieces[piecesOffset + index * STRIDE];`})}var I=`GPULineChunk`,L=Object.freeze({...a,measure:`distance`});function R(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Line chunk target must hold 4 elements`);let n=e.chunkLength??0,r=e.startMeasure??0,i=Math.min(e.endMeasure??34e37,34e37);if([n,r,i].some(Number.isNaN)||n<0)throw Error(`Line chunk settings must be numbers and chunkLength non-negative`);return t.set([n,Math.max(r,-34e37),i,0]),c(`GPULineChunk`,e.spatialContext,e.chunkLength),t}var z=class{id;props;spatialContext;coordinateSystem;radius;lineCapacity;constructor(e){this.id=e.id??`line-chunk`,this.props=e,this.spatialContext=s(this.id,e.spatialContext,L),this.coordinateSystem=this.spatialContext.coordinateSpace===`longitude-latitude`?`spherical`:`planar`,this.radius=o(this.spatialContext)??1;let{id:t}=this;if(e.mode!==`chunk`&&e.mode!==`substring`)throw Error(`${t} mode must be 'chunk' or 'substring'`);n(e.positions,[`float32x2`],`${t} positions`);let r=e.positions.length;if(r<1)throw Error(`${t} needs at least one position`);if(n(e.lineOffsets,[`uint32`],`${t} lineOffsets`),e.lineOffsets.length<2)throw Error(`${t} lineOffsets needs two rows`);let a=e.lineOffsets.length-1;if(n(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 float32 values`);if(this.lineCapacity=e.mode===`chunk`?Math.max(e.output.lineOffsets.length-1,1):a,h(t,e.output,this.lineCapacity),a*(this.lineCapacity+1)>4294967295||2*this.lineCapacity+r>4294967295)throw Error(`${t} line count times line capacity must fit in 32 bits`);i(t,m(e.output),[e.positions,e.lineOffsets,e.parameters])}getCommandNodes(n){let{props:i,id:a,coordinateSystem:o,lineCapacity:s}=this,{output:c}=i;_(n,a,Object.fromEntries([i.positions,i.lineOffsets,i.parameters,...m(c)].map((e,t)=>[String(t),e]))),v(n,a,{rowMeasures:i.positions.length*4,lineCountsAndScan:(i.lineOffsets.length-1)*4,pieceRecords:s*10*4});let l=i.positions.length,f=i.lineOffsets.length-1;i.mode===`chunk`&&c.status.requiredCount&&r(n.device,a,`required piece counts and offsets`,f*Uint32Array.BYTES_PER_ELEMENT,`reduce the input line count`);let p=e(n,`${a}-row-measures`,`float32`,l),h=o===`spherical`?e(n,`${a}-row-shifts`,`float32`,l):void 0,y=e(n,`${a}-piece-counts`,`uint32`,f),b=i.mode===`chunk`&&c.status.requiredCount?e(n,`${a}-required-piece-counts`,`uint32`,f):void 0,x=e(n,`${a}-piece-starts`,`uint32`,f),S=e(n,`${a}-pieces`,`uint32`,s*10),C=e(n,`${a}-vertex-counts`,`uint32`,s),w=e(n,`${a}-vertex-starts`,`uint32`,s),T=e(n,`${a}-piece-total`,`uint32`,1),E=[u(n,{id:`${a}-prefix`,operation:I,positions:i.positions,lineOffsets:i.lineOffsets,coordinateSystem:o,radius:this.radius,rowMeasures:p,rowShifts:h}),j(n,{id:`${a}-piece-count`,operation:I,mode:i.mode,lineOffsets:i.lineOffsets,rowMeasures:p,parameters:i.parameters,rowCount:l,maximumPieces:s+1,pieceCounts:y,requiredPieceCounts:b}),...new t({id:`${a}-piece-scan`,input:y,output:x}).getCommandNodes(n),M(n,{id:`${a}-piece-range`,operation:I,mode:i.mode,lineOffsets:i.lineOffsets,rowMeasures:p,parameters:i.parameters,pieceCounts:y,pieceStarts:x,rowCount:l,pieces:S,vertexCounts:C,pieceTotal:T}),...new t({id:`${a}-vertex-scan`,input:C,output:w}).getCommandNodes(n),N(n,{id:`${a}-emit`,operation:I,coordinateSystem:o,positions:i.positions,rowMeasures:p,rowShifts:h,pieces:S,vertexStarts:w,outputPositions:c.positions,outputMeasures:c.measures,outputSourceRows:c.sourceRows}),P(n,{id:`${a}-publish`,operation:I,pieceTotal:T,pieces:S,vertexStarts:w,capacity:c.positions.length,outputLineOffsets:c.lineOffsets,overflow:c.status.overflow,vertexCount:c.status.vertexCount,requiredVertexCount:c.status.requiredVertexCount}),g(n,{id:a,operation:I,output:c,lineCapacity:s,lineTotal:T})];if(b&&c.status.requiredCount){let r=e(n,`${a}-required-piece-starts`,`uint32`,f);E.push(...new t({id:`${a}-required-piece-scan`,input:b,output:r}).getCommandNodes(n),...d(n,{id:`${a}-required-pieces`,operation:I,counts:b,offsets:r,rowCount:f,total:c.status.requiredCount}))}return c.sourceLineIndices&&E.push(F(n,{id:`${a}-source-line-indexs`,operation:I,pieces:S,sourceLineIndices:c.sourceLineIndices})),E}},B=y({parameters:k,caption:`Lines cut into equal pieces; each output line has its own color.`,createInputs:({random:e,world:t})=>w(e,t,4,8),build(e,t){let n=e.upload(`positions`,`float32x2`,t.positions),r=e.upload(`line-offsets`,`uint32`,t.lineOffsets),i=C(e,`pieces`,3e3,400),a=e.allocate(`chunk-parameters`,`float32`,4);e.graph.add(new z({positions:n.view,lineOffsets:r.view,mode:`chunk`,parameters:a.view,output:i.output}));let o=O(e,`pieces`,i.positions,i.lineOffsets,{lineCount:i.lineCount});return{outputs:{original:e.upload(`original`,`float32x4`,D(t.positions,t.lineOffsets,!1)),pieceCount:i.lineCount,...o},update:e=>a.write(R({chunkLength:e.chunkLength}))}},interaction:{kind:`reach`,hint:`slide along a line`,rest:({inputs:e,toUnit:t})=>t(S(e,0,x(k,`chunkLength`).default).point),resolve(e,{inputs:t,outputs:n,peek:r}){let i=E(t,e.x,e.y);if(!i)return{};let a=b(x(k,`chunkLength`),i.measure),o=r(n.pieceCount)?.[0];return{values:{chunkLength:a},glyphs:[T(t.positions,i.row,i.point,.035),{shape:`dot`,at:i.point}],readout:`chunkLength ${a.toFixed(2)}${o===void 0?``:` · ${o} pieces`}`}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.original,radius:3},{layer:`segments`,nodes:e.points,edges:e.edges,values:e.lineIds,radius:3.6},{layer:`dots`,positions:e.points,values:e.lineIds,radius:6}]});export{B as default};
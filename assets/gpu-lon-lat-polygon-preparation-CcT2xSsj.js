import{n as e,t}from"./scan-total-C4QddMmi.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{f as i,v as a}from"./spatial-context-DwZEd4EZ.js";import{t as o}from"./status-aOwAAXek.js";import{E as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as f}from"./geographic-distribution-validation-BjMQQJgk.js";import{c as p,l as m,o as h,s as g}from"./line-segmentize-kernels-J-DWl7up.js";import{a as _,c as v,o as ee,s as te}from"./rectangle-clip-kernels-CyaxtIt4.js";import{t as y}from"./lon-lat-ring-wgsl-DpZBIfuE.js";import{t as b}from"./geodesic-wgsl-I0CYDKcz.js";import{o as x}from"./wgsl-kernel-nodes-DagJklH2.js";function S(e,t={},n=new Float32Array(4)){let r=t.maximumSegmentDegrees??(e===`great-circle`?.1:0);if(!(r>=0))throw Error(`getGPULonLatPolygonPreparationParameterValues maximumSegmentDegrees must be >= 0`);if(n.length<4)throw Error(`getGPULonLatPolygonPreparationParameterValues target must hold 4 elements`);return n.set([r,0,0,0]),n}var C=`GPULonLatPolygonPreparation`,w=6;function T(e,t,n=`u32`){return{name:e,view:t,type:n,access:`read`}}function E(e,t,n=`u32`){return{name:e,view:t,type:n,access:`read_write`}}function D(e,t){return x(e,{id:t.id,operation:C,variant:`initialize`,bindings:[E(`diagnostics`,t.diagnostics),E(`rectangle`,t.rectangle,`f32`),E(`stageFlags`,t.stageFlags)],invocationCount:1,body:`${Array.from({length:w},(e,t)=>`stageFlags[stageFlagsOffset + ${t}u] = 0u;`).join(`
`)}
  diagnostics[diagnosticsOffset] = 0u;
  diagnostics[diagnosticsOffset + 1u] = 0u;
  rectangle[rectangleOffset] = -180.0;
  rectangle[rectangleOffset + 1u] = -90.0;
  rectangle[rectangleOffset + 2u] = 180.0;
  rectangle[rectangleOffset + 3u] = 90.0;`})}function O(e,t){return x(e,{id:t.id,operation:C,variant:`classify`,bindings:[T(`positions`,t.positions,`f32`),T(`ringOffsets`,t.ringOffsets),T(`densePositions`,t.densePositions,`f32`),T(`denseOffsets`,t.denseOffsets),T(`parameters`,t.parameters,`f32`),E(`metadata`,t.metadata),E(`counts`,t.counts),E(`diagnostics`,t.diagnostics,`atomic<u32>`)],invocationCount:t.ringOffsets.length-1,declarations:`${y}\n${b}
fn position(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn finitePosition(value: vec2<f32>) -> bool {
  return all((bitcast<vec2<u32>>(value) & vec2<u32>(0x7f800000u)) != vec2<u32>(0x7f800000u));
}`,body:`let begin = ringOffsets[ringOffsetsOffset + index];
  var end = ringOffsets[ringOffsetsOffset + index + 1u];
  var flags = 0u;
  var invalid = false;
  var turn = 0i;
  var winding = 0.0;
  var pole = 0.0;
  var hasSecond = false;
  var hasThird = false;
  let denseBegin = denseOffsets[denseOffsetsOffset + index];
  let denseEnd = denseOffsets[denseOffsetsOffset + index + 1u];
  if (end > begin) {
    let first = position(begin);
    var second = first;
    if (end > begin + 1u && all(first == position(end - 1u))) { end--; }
    var minimum = first.x;
    var maximum = first.x;
    var longitude = first.x;
    for (var row = begin; row < end; row++) {
      let current = position(row);
      let next = position(select(row + 1u, begin, row + 1u >= end));
      if (!finitePosition(current) || abs(current.y) > 90.0) { flags |= 16u; continue; }
      if (any(current != first)) {
        if (!hasSecond) { second = current; hasSecond = true; }
        else if (any(current != second)) { hasThird = true; }
      }
      if (!finitePosition(next)) { continue; }
      if (lonLatIsCrossingEdge(current, next)) { flags |= 1u; }
      if (${t.edgeModel===`linear`?`lonLatIsAmbiguousLinearEdge`:`lonLatIsAmbiguousGreatCircleEdge`}(current, next)) { flags |= 4u; }
      let step = lonLatRingStep(current, next);
      winding += step;
      if (!lonLatIsPoleEdge(current, next)) { turn -= i32(round((next.x - current.x) / 360.0)); }
      longitude = next.x + 360.0 * f32(turn);
      minimum = min(minimum, longitude);
      maximum = max(maximum, longitude);
      let maximumStep = parameters[parametersOffset];
      let extent = max(abs(lonLatWrapLongitudeStep(next.x - current.x)), geodesicCentralAngle(current, next) * GEODESIC_RADIANS_TO_DEGREES);
      if (maximumStep > 0.0 && ceil(extent / maximumStep) > ${t.maximumPieceCount}.0) { flags |= 32u; }
    }
    if (!hasThird) { flags &= 16u; winding = 0.0; }
    if (hasThird && maximum - minimum > 360.0) { flags |= 8u; }
    invalid = (flags & 28u) != 0u ${t.antimeridian===`reject`?`|| (flags & 1u) != 0u`:``};
    if (lonLatIsPoleWinding(winding)) {
      var minimumLatitude = 90.0;
      var maximumLatitude = -90.0;
      for (var row = denseBegin; row < denseEnd; row++) {
        let latitude = densePositions[densePositionsOffset + 2u * row + 1u];
        minimumLatitude = min(minimumLatitude, latitude);
        maximumLatitude = max(maximumLatitude, latitude);
      }
      pole = sign(minimumLatitude + maximumLatitude) * 90.0;
      invalid = invalid || pole == 0.0 ${t.poleRings===`reject`?`|| true`:``};
      if (!invalid) { flags |= 2u; }
    }
  }
  metadata[metadataOffset + 4u * index] = flags;
  metadata[metadataOffset + 4u * index + 1u] = select(0u, 1u, invalid);
  metadata[metadataOffset + 4u * index + 2u] = bitcast<u32>(pole);
  metadata[metadataOffset + 4u * index + 3u] = bitcast<u32>(turn);
  counts[countsOffset + index] = select(0u, denseEnd - denseBegin + select(0u, 3u, (flags & 2u) != 0u), hasThird && !invalid);
  ${t.edgeModel===`great-circle`?`if (parameters[parametersOffset] == 0.0 || (flags & 32u) != 0u) {
    atomicStore(&diagnostics[diagnosticsOffset + 1u], 1u);
  }`:``}`})}function k(e,t){return x(e,{id:t.id,operation:C,variant:`feature-validity`,bindings:[T(`featureOffsets`,t.featureOffsets),T(`polygonOffsets`,t.polygonOffsets),T(`metadata`,t.metadata),E(`counts`,t.counts),E(`diagnostics`,t.diagnostics,`atomic<u32>`)],invocationCount:t.featureOffsets.length-1,body:`let begin = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index]];
  let end = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index + 1u]];
  var invalid = false;
  for (var ring = begin; ring < end; ring++) { invalid = invalid || metadata[metadataOffset + 4u * ring + 1u] != 0u; }
  if (invalid) {
    atomicAdd(&diagnostics[diagnosticsOffset], 1u);
    for (var ring = begin; ring < end; ring++) { counts[countsOffset + ring] = 0u; }
  } else {
    // An empty shell has no filled area; its holes cannot become standalone polygons.
    for (var polygon = featureOffsets[featureOffsetsOffset + index]; polygon < featureOffsets[featureOffsetsOffset + index + 1u]; polygon++) {
      let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
      let lastRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
      if (firstRing < lastRing && counts[countsOffset + firstRing] == 0u) {
        for (var ring = firstRing + 1u; ring < lastRing; ring++) { counts[countsOffset + ring] = 0u; }
      }
    }
  }`})}function ne(e,t){return x(e,{id:t.id,operation:C,variant:`cap`,bindings:[T(`positions`,t.positions,`f32`),T(`turns`,t.turns,`i32`),T(`offsets`,t.offsets),T(`metadata`,t.metadata),T(`starts`,t.starts),T(`counts`,t.counts),E(`outputPositions`,t.outputPositions,`f32`),E(`outputTurns`,t.outputTurns,`i32`)],invocationCount:t.counts.length,declarations:`fn copyVertex(destination: u32, source: u32, latitude: f32, turn: i32) {
  if (destination >= ${t.outputPositions.length}u) { return; }
  outputPositions[outputPositionsOffset + 2u * destination] = positions[positionsOffset + 2u * source];
  outputPositions[outputPositionsOffset + 2u * destination + 1u] = latitude;
  outputTurns[outputTurnsOffset + destination] = turn;
}`,body:`if (counts[countsOffset + index] == 0u) { return; }
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  let start = starts[startsOffset + index];
  for (var row = begin; row < end; row++) {
    copyVertex(start + row - begin, row, positions[positionsOffset + 2u * row + 1u], turns[turnsOffset + row]);
  }
  if ((metadata[metadataOffset + 4u * index] & 2u) != 0u) {
    let firstTurn = turns[turnsOffset + begin];
    let closureTurn = firstTurn + bitcast<i32>(metadata[metadataOffset + 4u * index + 3u]);
    let pole = bitcast<f32>(metadata[metadataOffset + 4u * index + 2u]);
    copyVertex(start + end - begin, begin, positions[positionsOffset + 2u * begin + 1u], closureTurn);
    copyVertex(start + end - begin + 1u, begin, pole, closureTurn);
    copyVertex(start + end - begin + 2u, begin, pole, firstTurn);
  }`})}function A(e,t){return x(e,{id:t.id,operation:C,variant:`ring-offsets`,bindings:[T(`starts`,t.starts),T(`counts`,t.counts),E(`offsets`,t.offsets),...t.stageFlags?[E(`stageFlags`,t.stageFlags)]:[]],invocationCount:t.counts.length+1,body:`var value = 0u;
  if (index < ${t.counts.length}u) { value = starts[startsOffset + index]; }
  else if (index > 0u) { value = starts[startsOffset + index - 1u] + counts[countsOffset + index - 1u]; }
  offsets[offsetsOffset + index] = min(value, ${t.capacity}u);
  ${t.stageFlags?`if (index == ${t.counts.length}u) { stageFlags[stageFlagsOffset + ${t.stage??0}u] = value; }`:``}`})}function re(e,t){return x(e,{id:t.id,operation:C,variant:`normalize`,bindings:[T(`positions`,t.positions,`f32`),E(`turns`,t.turns,`i32`),T(`ringOffsets`,t.ringOffsets),T(`polygonOffsets`,t.polygonOffsets)],invocationCount:t.polygonOffsets.length-1,declarations:`fn longitude(row: u32) -> f32 { return positions[positionsOffset + 2u * row] + 360.0 * f32(turns[turnsOffset + row]); }`,body:`let shell = polygonOffsets[polygonOffsetsOffset + index];
  let ringEnd = polygonOffsets[polygonOffsetsOffset + index + 1u];
  if (shell == ringEnd) { return; }
  let begin = ringOffsets[ringOffsetsOffset + shell];
  let end = ringOffsets[ringOffsetsOffset + shell + 1u];
  if (begin == end) { return; }
  var minimum = longitude(begin);
  var maximum = minimum;
  for (var row = begin + 1u; row < end; row++) { minimum = min(minimum, longitude(row)); maximum = max(maximum, longitude(row)); }
  let shellTurns = i32(floor((minimum + 180.0) / 360.0));
  let centre = (minimum + maximum) * 0.5;
  for (var ring = shell; ring < ringEnd; ring++) {
    let first = ringOffsets[ringOffsetsOffset + ring];
    let last = ringOffsets[ringOffsetsOffset + ring + 1u];
    if (first == last) { continue; }
    var shift = -shellTurns;
    if (ring != shell) { shift += i32(floor((centre - longitude(first)) / 360.0 + 0.5)); }
    for (var row = first; row < last; row++) { turns[turnsOffset + row] += shift; }
  }`})}function ie(e,t){return x(e,{id:t.id,operation:C,variant:`topology`,bindings:[T(`featureOffsets`,t.featureOffsets),T(`polygonOffsets`,t.polygonOffsets),E(`outputFeatureOffsets`,t.outputFeatureOffsets),E(`outputPolygonOffsets`,t.outputPolygonOffsets),E(`ringMap`,t.ringMap)],invocationCount:t.featureOffsets.length,body:`let first = featureOffsets[featureOffsetsOffset + index];
  outputFeatureOffsets[outputFeatureOffsetsOffset + index] = 2u * first;
  if (index == ${t.featureOffsets.length-1}u) {
    outputPolygonOffsets[outputPolygonOffsetsOffset + 2u * first] = 2u * polygonOffsets[polygonOffsetsOffset + first];
    return;
  }
  let last = featureOffsets[featureOffsetsOffset + index + 1u];
  let firstRing = polygonOffsets[polygonOffsetsOffset + first];
  let lastRing = polygonOffsets[polygonOffsetsOffset + last];
  for (var copy = 0u; copy < 2u; copy++) {
    for (var polygon = first; polygon < last; polygon++) {
      let targetPolygon = 2u * first + copy * (last - first) + polygon - first;
      let sourceRing = polygonOffsets[polygonOffsetsOffset + polygon];
      let targetRing = 2u * firstRing + copy * (lastRing - firstRing) + sourceRing - firstRing;
      outputPolygonOffsets[outputPolygonOffsetsOffset + targetPolygon] = targetRing;
      for (var ring = sourceRing; ring < polygonOffsets[polygonOffsetsOffset + polygon + 1u]; ring++) {
        ringMap[ringMapOffset + targetRing + ring - sourceRing] = 2u * ring + copy;
      }
    }
  }`})}var j=`
fn getPosition(row: u32) -> vec2<f32> {
  var longitude = positions[positionsOffset + 2u * row];
  let turn = turns[turnsOffset + row];
  if (turn != 0i) { longitude += 360.0 * f32(turn); }
  return vec2<f32>(longitude, positions[positionsOffset + 2u * row + 1u]);
}
fn ringEnd(begin: u32, last: u32) -> u32 {
  var end = last;
  if (end > begin + 1u && all(getPosition(begin) == getPosition(end - 1u))) { end--; }
  return end;
}
fn keepRow(row: u32, begin: u32) -> bool {
  if (row == begin) { return true; }
  return any(getPosition(row) != getPosition(row - 1u));
}`;function ae(e,t){return x(e,{id:t.id,operation:C,variant:`merge-count-${t.copy}`,bindings:[T(`positions`,t.positions,`f32`),T(`turns`,t.turns,`i32`),T(`offsets`,t.offsets),T(`ringMap`,t.ringMap),E(`counts`,t.counts)],invocationCount:t.counts.length,declarations:j,body:`let mapping = ringMap[ringMapOffset + index];
  if (mapping % 2u != ${t.copy}u) { return; }
  let ring = mapping / 2u;
  let begin = offsets[offsetsOffset + ring];
  let end = ringEnd(begin, offsets[offsetsOffset + ring + 1u]);
  var count = 0u;
  var second = vec2<f32>(0.0);
  var hasSecond = false;
  var hasThird = false;
  for (var row = begin; row < end; row++) {
    let current = getPosition(row);
    if (any(current != getPosition(begin))) {
      if (!hasSecond) { second = current; hasSecond = true; }
      else if (any(current != second)) { hasThird = true; }
    }
    if (keepRow(row, begin)) { count++; }
  }
  counts[countsOffset + index] = select(0u, count, hasThird);`})}function oe(e,t){return x(e,{id:t.id,operation:C,variant:`merge-emit-${t.copy}`,bindings:[T(`positions`,t.positions,`f32`),T(`turns`,t.turns,`i32`),T(`offsets`,t.offsets),T(`ringMap`,t.ringMap),T(`counts`,t.counts),T(`starts`,t.starts),E(`output`,t.output,`f32`)],invocationCount:t.counts.length,declarations:j,body:`let mapping = ringMap[ringMapOffset + index];
  if (mapping % 2u != ${t.copy}u || counts[countsOffset + index] == 0u) { return; }
  let ring = mapping / 2u;
  let begin = offsets[offsetsOffset + ring];
  let end = ringEnd(begin, offsets[offsetsOffset + ring + 1u]);
  var destination = starts[startsOffset + index];
  for (var row = begin; row < end; row++) {
    if (keepRow(row, begin)) {
      if (destination < ${t.output.length}u) {
        let position = getPosition(row);
        output[outputOffset + 2u * destination] = position.x;
        output[outputOffset + 2u * destination + 1u] = position.y;
      }
      destination++;
    }
  }`})}function se(e,n){let r=[T(`counts`,n.counts),T(`starts`,n.starts),E(`stageFlags`,n.stageFlags),T(`metadata`,n.metadata),T(`diagnostics`,n.diagnostics)];for(let e of[`ringFlags`,`invalidCount`,`approximation`])n[e]&&r.push(E(e,n[e]));return x(e,{id:n.id,operation:C,variant:`status`,bindings:r,invocationCount:Math.max(1,n.metadata.length/4),declarations:t,body:`${n.ringFlags?`ringFlags[ringFlagsOffset + index] = metadata[metadataOffset + 4u * index];`:``}
  if (index == 0u) {
    let last = ${n.counts.length-1}u;
    let total = starts[startsOffset + last] + counts[countsOffset + last];
    let denseTotal = stageFlags[stageFlagsOffset];
    var intermediateTotal = 0u;
    for (var stage = 0u; stage < 6u; stage++) {
      intermediateTotal = max(intermediateTotal, stageFlags[stageFlagsOffset + stage]);
    }
    let requiredCount = select(total, getSaturatedScanProduct(getSaturatedScanSum(denseTotal, ${3*n.metadata.length/4}u), 8u), intermediateTotal > ${n.capacity}u);
    stageFlags[stageFlagsOffset] = intermediateTotal;
    stageFlags[stageFlagsOffset + 1u] = 0u;
    stageFlags[stageFlagsOffset + 2u] = 0u;
    stageFlags[stageFlagsOffset + 3u] = total;
    stageFlags[stageFlagsOffset + 4u] = requiredCount;
    ${n.invalidCount?`invalidCount[invalidCountOffset] = diagnostics[diagnosticsOffset];`:``}
    ${n.approximation?`approximation[approximationOffset] = diagnostics[diagnosticsOffset + 1u];`:``}
  }`})}var M=`GPULonLatPolygonPreparation`,N=class{id;props;spatialContext;inputContext;constructor(e){this.id=e.id??`lon-lat-polygon-preparation`,this.props=e;let{id:t}=this;this.inputContext=a(t,e.spatialContext,{coordinateSpaces:[`longitude-latitude`],metrics:[`none`,`great-circle`],edgeModels:[`linear`,`great-circle`]}),this.spatialContext={...this.inputContext,edgeModel:`linear`};let n=e.maximumPieceCount??1024;if(!Number.isInteger(n)||n<1||n>65536)throw Error(`${t} maximumPieceCount must be an integer in [1, 65536]`);if(e.antimeridian!==void 0&&![`split`,`reject`].includes(e.antimeridian))throw Error(`${t} antimeridian must be split or reject`);if(e.poleRings!==void 0&&![`cap`,`reject`].includes(e.poleRings))throw Error(`${t} poleRings must be cap or reject`);if(s(e.positions,[`float32x2`],`${t} positions`),e.positions.length<1)throw Error(`${t} needs at least one position`);for(let n of[`featureOffsets`,`polygonOffsets`,`ringOffsets`]){if(s(e[n],[`uint32`],`${t} ${n}`),e[n].length<2)throw Error(`${t} ${n} needs two rows`);s(e.output[n],[`uint32`],`${t} output.${n}`);let r=n===`featureOffsets`?e[n].length:2*e[n].length-1;if(e.output[n].length!==r)throw Error(`${t} output.${n} must hold ${r} rows`)}if(e.output.ringFlags&&(s(e.output.ringFlags,[`uint32`],`${t} output.ringFlags`),e.output.ringFlags.length!==e.ringOffsets.length-1))throw Error(`${t} output.ringFlags must hold ${e.ringOffsets.length-1} rows`);if(s(e.output.positions,[`float32x2`],`${t} output.positions`),s(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 float32 values`);o(t,e.output.status),u(t,this.getOutputViews(),this.getInputViews());let r=e.output;f(t,{},Object.fromEntries(this.getOutputViews().map((e,t)=>[`output.${t}`,e])),[{reads:{},writes:{featureOffsets:r.featureOffsets,polygonOffsets:r.polygonOffsets}},{reads:{},writes:{ringFlags:r.ringFlags,invalidCount:r.status.invalidCount,approximation:r.status.approximation}},{reads:{},writes:{overflow:r.status.overflow,count:r.status.count,requiredCount:r.status.requiredCount}}])}getCommandNodes(t){let{id:a,props:o}=this,{output:s}=o;d(t,a,Object.fromEntries([...this.getInputViews(),...this.getOutputViews().filter(e=>e!==s.positions||e.length>0)].map((e,t)=>[`binding.${t}`,e])));for(let[e,n]of[[`preparation position streams`,Math.max(s.positions.length*8,4)],[`preparation row scan scratch`,o.positions.length*4],[`preparation ring metadata`,(o.ringOffsets.length-1)*16],[`preparation ring scan scratch`,Math.max(2*(o.ringOffsets.length-1),1)*4],[`preparation stage totals`,24]])c(t.device,a,e,n,`reduce vertex capacity or ring count`);l(a,t,[...this.getInputViews(),...this.getOutputViews()]);let u=o.maximumPieceCount??1024,f=o.ringOffsets.length-1,y=s.positions.length,b=(e,r)=>n(t,`${a}-${e}`,`uint32`,r),x=(e,r)=>n(t,`${a}-${e}`,`float32x2`,r),S=(e,r)=>n(t,`${a}-${e}`,`sint32`,r),C=[],w=(e,n,i)=>{C.push(...new r({id:`${a}-${e}`,input:n,output:i,mode:`exclusive`}).getCommandNodes(t))},T=b(`stage-totals`,6),E=t.createDataView(T.buffer,{format:`uint32`,length:1}),j=b(`diagnostics`,2),N=t.createDataView(j.buffer,{format:`uint32`,length:1,byteOffset:Uint32Array.BYTES_PER_ELEMENT}),P=n(t,`${a}-rectangle`,`float32`,4),F=b(`metadata`,4*f),I=S(`row-turns`,o.positions.length),L=b(`dense-counts`,o.positions.length),R=b(`dense-starts`,o.positions.length),z=x(`dense-positions`,y),B=S(`dense-turns`,y),V=b(`dense-offsets`,f+1),H=i(this.inputContext),U=H===`linear`?`linear-degrees`:`great-circle`,W={operation:M,positions:o.positions,lineOffsets:o.ringOffsets,coordinateSystem:`spherical`,radius:1,closedLines:!0,polygonEdges:!0};C.push(D(t,{id:`${a}-initialize`,diagnostics:j,rectangle:P,stageFlags:T}),g(t,{...W,id:`${a}-prefix`,rowTurns:I}),p(t,{...W,id:`${a}-dense-count`,parameters:o.parameters,maximumPieceCount:u,pieceCountModel:`degree-step`,approximation:H===`great-circle`?N:void 0,interpolation:U,counts:L})),C.push(...e(t,{id:`${a}-dense-scan`,operation:M,counts:L,offsets:R,rowCount:o.positions.length,total:E})),C.push(m(t,{...W,id:`${a}-dense-emit`,counts:L,starts:R,rowTurns:I,interpolation:U,outputPositions:z,outputTurns:B}),h(t,{id:`${a}-dense-offsets-publish`,operation:M,lineOffsets:o.ringOffsets,lineCount:f,rowCount:o.positions.length,counts:L,starts:R,outputLineOffsets:V,capacity:y,overflow:b(`dense-overflow`,1),requiredVertexCount:E}));let G=b(`cap-counts`,f),K=b(`cap-starts`,f),q=x(`cap-positions`,y),J=S(`cap-turns`,y),Y=b(`cap-offsets`,f+1);C.push(O(t,{id:`${a}-classify`,positions:o.positions,ringOffsets:o.ringOffsets,densePositions:z,denseOffsets:V,parameters:o.parameters,metadata:F,counts:G,diagnostics:j,edgeModel:H,maximumPieceCount:u,antimeridian:o.antimeridian??`split`,poleRings:o.poleRings??`cap`}),k(t,{id:`${a}-features`,featureOffsets:o.featureOffsets,polygonOffsets:o.polygonOffsets,metadata:F,counts:G,diagnostics:j})),w(`cap-scan`,G,K),C.push(ne(t,{id:`${a}-cap`,positions:z,turns:B,offsets:V,metadata:F,starts:K,counts:G,outputPositions:q,outputTurns:J}),A(t,{id:`${a}-cap-offsets-publish`,starts:K,counts:G,offsets:Y,capacity:y,stageFlags:T,stage:1}),re(t,{id:`${a}-normalize`,positions:q,turns:J,ringOffsets:Y,polygonOffsets:o.polygonOffsets}));let X=[];for(let e=0;e<2;e++){let n=q,r=J,i=Y,o=[x(`clip-${e}-positions-a`,y),x(`clip-${e}-positions-b`,y)],s=[S(`clip-${e}-turns-a`,y),S(`clip-${e}-turns-b`,y)],c=[b(`clip-${e}-offsets-a`,f+1),b(`clip-${e}-offsets-b`,f+1)];for(let l=0;l<(y===0?0:2);l++){let u=o[l%2],d=s[l%2],f=c[l%2],p=b(`clip-${e}-${l}-counts`,n.length),m=b(`clip-${e}-${l}-starts`,n.length),h={operation:M,boundary:l,positions:n,positionTurns:r,longitudeCopy:l===0?e:0,ringOffsets:i,parameters:P};C.push(ee(t,{...h,id:`${a}-clip-${e}-${l}-count`,counts:p})),w(`clip-${e}-${l}-scan`,p,m),C.push(te(t,{...h,id:`${a}-clip-${e}-${l}-emit`,starts:m,outputPositions:u,outputTurns:d}),v(t,{id:`${a}-clip-${e}-${l}-offsets`,operation:M,capacity:y,inputOffsets:i,starts:m,counts:p,outputOffsets:f,stageFlags:T,stage:2+e*2+l})),n=u,r=d,i=f}X.push({positions:n,turns:r,offsets:i})}let Z=b(`ring-map`,2*f),Q=b(`merge-counts`,2*f),$=b(`merge-starts`,2*f);return C.push(ie(t,{id:`${a}-topology`,featureOffsets:o.featureOffsets,polygonOffsets:o.polygonOffsets,outputFeatureOffsets:s.featureOffsets,outputPolygonOffsets:s.polygonOffsets,ringMap:Z})),X.forEach((e,n)=>C.push(ae(t,{...e,id:`${a}-merge-${n}-count`,ringMap:Z,counts:Q,copy:n}))),w(`merge-scan`,Q,$),y>0&&X.forEach((e,n)=>C.push(oe(t,{...e,id:`${a}-merge-${n}-emit`,ringMap:Z,counts:Q,starts:$,output:s.positions,copy:n}))),C.push(A(t,{id:`${a}-output-offsets`,starts:$,counts:Q,offsets:s.ringOffsets,capacity:s.positions.length}),se(t,{id:`${a}-status`,counts:Q,starts:$,stageFlags:T,capacity:y,metadata:F,diagnostics:j,ringFlags:s.ringFlags,invalidCount:s.status.invalidCount,approximation:s.status.approximation}),_(t,{id:`${a}-publish`,operation:M,capacity:s.positions.length,stageFlags:T,overflow:s.status.overflow,vertexCount:s.status.count,requiredVertexCount:s.status.requiredCount,requiredVertexCountIndex:4})),C}getInputViews(){let{props:e}=this;return[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets,e.parameters]}getOutputViews(){let{output:e}=this.props;return[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets,e.ringFlags,...Object.values(e.status)].filter(e=>!!e)}};export{S as n,N as t};
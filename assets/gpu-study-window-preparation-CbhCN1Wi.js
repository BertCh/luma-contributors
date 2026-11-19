import{Q as e}from"./dist-BusamGlx.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{s as r,v as i}from"./spatial-context-DwZEd4EZ.js";import{i as a,t as o}from"./polygon-rasterization-parameters-B9SJE1ko.js";import{t as ee}from"./gpu-polygon-rasterization-DQD7gtSB.js";import{E as s,d as c,f as l,v as te,x as ne,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as re}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as ie}from"./polygon-edge-slabs-CRnrFvP2.js";import{i as d,r as f,t as p}from"./study-window-wgsl-B3mOUanG.js";import{t as m}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{t as h}from"./construction-snapshot-DdkX9Pc0.js";import{t as ae}from"./exact-geometry-wgsl-y7TUnqOh.js";import{t as g}from"./exact-orientation-wgsl-B_jI-HEc.js";import{c as _,r as v}from"./wgsl-common-DfyXiYB0.js";import{o as y}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as oe,t as b}from"./prepared-geometry-snapshot-BdnpFSR9.js";var x=`GPUStudyWindowPreparation`,S=10,C=1048576,w=4294967294,se=1024,ce=4,T=4,E=256,D=16,O=class{kind=`polygon`;id;spatialContext;vertexCount;props;sampleGrid;encodedBuildCount=0;graph;views;ownedBuffers=[];rebuildGeneration=new Uint32Array([1,0]);encodingRequestPending=!0;constructor(e){this.id=e.id??`study-window`,this.spatialContext=i(this.id,e.spatialContext,r),this.props=e,this.vertexCount=e.positions.length,this.sampleGrid=e.samplingGrid??[256,256];let{id:t,sampleGrid:n}=this;s(e.positions,[`float32x2`],`${t} positions`);for(let n of[`featureOffsets`,`polygonOffsets`,`ringOffsets`])s(e[n],[`uint32`],`${t} ${n}`);if(e.featureOffsets.length!==2)throw Error(`${t} study window needs exactly one feature (featureOffsets of length 2)`);if(this.vertexCount<3||e.ringOffsets.length<2||e.polygonOffsets.length<2)throw Error(`${t} study window needs at least one ring of 3 or more vertices`);if(n.length!==2||!n.every(e=>Number.isInteger(e)&&e>0)||n[0]*n[1]>C)throw Error(`${t} samplingGrid must be two positive integers with a product of at most 1048576`);if(l(t,`crossingCapacity`,e.crossingCapacity??4*n[1]*(e.ringOffsets.length-1),2,w),e.slabEntryCapacity!==void 0&&l(t,`slabEntryCapacity`,e.slabEntryCapacity,1,w),e.output?.summary&&(s(e.output.summary,[`float32`],`${t} output.summary`),e.output.summary.length!==8))throw Error(`${t} output.summary must contain 8 float32 values`);u(t,[e.output?.summary],[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets])}getViews(e){let t=this.views;if(!this.isDeclaredIn(e)||!t)throw Error(`${this.id} study window needs graph.add before its consumers`);return t}isDeclaredIn(e){return this.graph!==void 0&&this.graph===e}invalidate(){b(this.rebuildGeneration),this.encodingRequestPending=!0}destroy(){for(let e of this.ownedBuffers)e.destroy();this.views=void 0,this.ownedBuffers=[],this.graph=void 0,this.rebuildGeneration.set([1,0]),this.encodingRequestPending=!0}createPersistentView(t,n,r,i){let a=t.device.createBuffer({id:`${this.id}-${n}`,byteLength:Math.max(i,1)*(r===`uint32x2`?8:4),usage:e.STORAGE|e.COPY_SRC|e.COPY_DST});return this.ownedBuffers.push(a),c(t,`${this.id}-${n}`,a,r,i)}getCommandNodes(e){let{id:r,vertexCount:i,sampleGrid:s}=this,c=this.props;if(this.graph)throw Error(`${r} is already added to a graph`);let l=c.ringOffsets.length-1,u=s[0]*s[1];ne(r,e,[c.positions,c.featureOffsets,c.polygonOffsets,c.ringOffsets,c.output?.summary]),re(e,r,{positions:c.positions,featureOffsets:c.featureOffsets,polygonOffsets:c.polygonOffsets,ringOffsets:c.ringOffsets,summary:c.output?.summary});let C=Math.floor(i/T)+1,w=Math.min(Math.max(0,Math.floor(e.device.limits.maxStorageBufferBindingSize/8)-i),c.slabEntryCapacity??Math.max(ce*i,se)),O=f({vertexCount:i,slabOffsetCount:C+1,slabEntryCount:i+w,sampleCellCapacity:u}),k=c.crossingCapacity??4*s[1]*l;for(let[t,n]of[[`ring terms`,9*i*4],[`edges`,4*i*4],[`rings`,S*l*4],[`sample scan`,(u+1)*4],[`slab entries`,(i+w)*8],[`slab offsets`,(C+1)*4],[`slab parameters`,24],[`packed window`,O.length*4],[`raster crossings`,k*4],[`raster sort histogram`,Math.ceil(k/E)*D*4]])te(e.device,r,t,n,`reduce vertices, samplingGrid or capacities`);let A=oe(e,{id:r,views:{positions:c.positions,featureOffsets:c.featureOffsets,polygonOffsets:c.polygonOffsets,ringOffsets:c.ringOffsets},generation:this.rebuildGeneration,ownedBuffers:this.ownedBuffers,recordEncoding:e=>{this.encodingRequestPending?this.encodedBuildCount++:this.props.rebuildWhen?.(e)&&(b(this.rebuildGeneration),this.encodedBuildCount++),this.encodingRequestPending=!1}});c={...h(c,[`id`,`spatialContext`,`positions`,`featureOffsets`,`polygonOffsets`,`ringOffsets`,`samplingGrid`,`crossingCapacity`,`slabEntryCapacity`,`rebuildWhen`,`output`]),...A.views},this.graph=e;let j=this.createPersistentView(e,`edges`,`float32`,4*i),M=this.createPersistentView(e,`summary`,`float32`,8),N=t(e,`${r}-terms`,`float32`,9*i),P=t(e,`${r}-rings`,`float32`,S*l),F=this.createPersistentView(e,`sample-cells`,`uint32`,u),I=this.createPersistentView(e,`sample-cell-count`,`uint32`,1),L={name:`positions`,view:c.positions,type:`f32`,access:`read`},R={name:`ringOffsets`,view:c.ringOffsets,type:`u32`,access:`read`},z={name:`rings`,view:P,type:`f32`,access:`read`},B={name:`summary`,view:M,type:`f32`,access:`read`},V=a(`findRing`,`ringOffsets`,l),H=`fn readVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}`,U=[y(e,{id:`${r}-ring-terms`,operation:x,variant:`ring-terms`,bindings:[L,R,{name:`terms`,view:N,type:`f32`,access:`read_write`}],invocationCount:i,declarations:`${V}${H}${g}${v}`,body:`let base = termsOffset + 9u * index;
  for (var slot = 0u; slot < 4u; slot++) { terms[base + slot] = 0.0; }
  terms[base + 4u] = 3.402823466e38;
  terms[base + 5u] = 3.402823466e38;
  terms[base + 6u] = -3.402823466e38;
  terms[base + 7u] = -3.402823466e38;
  terms[base + 8u] = 0.0;
  let ring = findRing(index);
  if (ring == 0xffffffffu) { return; }
  let first = ringOffsets[ringOffsetsOffset + ring];
  let end = ringOffsets[ringOffsetsOffset + ring + 1u];
  let next = select(index + 1u, first, index + 1u == end);
  let startPosition = readVertex(index);
  let endPosition = readVertex(next);
  let origin = readVertex(first);
  // One exact predicate call site; uncertified orientations must not discard a ring.
  terms[base + 8u] = select(0.0, 1.0, orientSign(origin, startPosition, endPosition) != 0);
  let relativeStart = startPosition - origin;
  let relativeEnd = endPosition - origin;
  let forward = relativeStart.x * relativeEnd.y;
  let backward = relativeStart.y * relativeEnd.x;
  var determinant = forward - backward;
  if (!isFiniteBits(startPosition.x) || !isFiniteBits(startPosition.y) ||
      !isFiniteBits(endPosition.x) || !isFiniteBits(endPosition.y)) { determinant = getNaN(); }
  terms[base] = determinant;
  terms[base + 1u] = abs(forward) + abs(backward);
  terms[base + 2u] = distance(startPosition, endPosition);
  terms[base + 3u] = 1.0;
  if (isFiniteBits(startPosition.x) && isFiniteBits(startPosition.y)) {
    terms[base + 4u] = startPosition.x;
    terms[base + 5u] = startPosition.y;
    terms[base + 6u] = startPosition.x;
    terms[base + 7u] = startPosition.y;
  }`})];for(let t=1;t<i;t*=2)U.push(y(e,{id:`${r}-ring-reduce-${t}`,operation:x,variant:`ring-reduce`,bindings:[R,{name:`terms`,view:N,type:`f32`,access:`read_write`}],invocationCount:i,declarations:`${V}\noverride TREE_STRIDE: u32;`,constants:{TREE_STRIDE:t},body:`let ring = findRing(index);
  if (ring == 0xffffffffu) { return; }
  let first = ringOffsets[ringOffsetsOffset + ring];
  let end = ringOffsets[ringOffsetsOffset + ring + 1u];
  if ((index - first) % (2u * TREE_STRIDE) != 0u || index + TREE_STRIDE >= end) { return; }
  let base = termsOffset + 9u * index;
  let partner = termsOffset + 9u * (index + TREE_STRIDE);
  for (var slot = 0u; slot < 4u; slot++) { terms[base + slot] += terms[partner + slot]; }
  for (var slot = 4u; slot < 6u; slot++) { terms[base + slot] = min(terms[base + slot], terms[partner + slot]); }
  for (var slot = 6u; slot < 8u; slot++) { terms[base + slot] = max(terms[base + slot], terms[partner + slot]); }
  terms[base + 8u] += terms[partner + 8u];`}));U.push(y(e,{id:`${r}-ring-orientation`,operation:x,variant:`ring-orientation`,bindings:[R,{name:`polygonOffsets`,view:c.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:c.featureOffsets,type:`u32`,access:`read`},{name:`terms`,view:N,type:`f32`,access:`read`},{...z,access:`read_write`}],invocationCount:l,declarations:`${_}${a(`findPolygon`,`polygonOffsets`,c.polygonOffsets.length-1)}`,body:`let base = ringsOffset + ${S}u * index;
  for (var slot = 0u; slot < ${S}u; slot++) { rings[base + slot] = 0.0; }
  let polygon = findPolygon(index);
  if (polygon == 0xffffffffu || polygon < featureOffsets[featureOffsetsOffset] ||
      polygon >= featureOffsets[featureOffsetsOffset + 1u]) { return; }
  let first = ringOffsets[ringOffsetsOffset + index];
  let end = ringOffsets[ringOffsetsOffset + index + 1u];
  if (end - first < 3u) { return; }
  let source = termsOffset + 9u * first;
  if (terms[source + 8u] == 0.0) { return; }
  rings[base + 9u] = 1.0;
  let doubledArea = terms[source];
  let role = select(-1.0, 1.0, index == polygonOffsets[polygonOffsetsOffset + polygon]);
  rings[base] = select(0.0, abs(doubledArea) * 0.5 * role, end - first >= 3u);
  rings[base + 1u] = terms[source + 2u];
  rings[base + 2u] = terms[source + 3u];
  let errorBound = ${8*2**-23*(1+Math.ceil(Math.log2(i)))} * terms[source + 1u];
  if (!isFiniteBits(doubledArea) || abs(doubledArea) <= errorBound) {
    rings[base + 3u] = 1.0;
  }
  for (var slot = 4u; slot < 8u; slot++) { rings[base + slot] = terms[source + slot]; }
  rings[base + 8u] = sign(doubledArea) * role;`}),y(e,{id:`${r}-orient-edges`,operation:x,variant:`orient-edges`,bindings:[L,R,z,{name:`edges`,view:j,type:`f32`,access:`read_write`}],invocationCount:i,declarations:`${V}${H}${v}`,body:`let base = edgesOffset + 4u * index;
  for (var slot = 0u; slot < 4u; slot++) { edges[base + slot] = getNaN(); }
  let ring = findRing(index);
  if (ring == 0xffffffffu) { return; }
  let ringBase = ringsOffset + ${S}u * ring;
  if (rings[ringBase + 9u] == 0.0) { return; }
  let first = ringOffsets[ringOffsetsOffset + ring];
  let end = ringOffsets[ringOffsetsOffset + ring + 1u];
  let next = select(index + 1u, first, index + 1u == end);
  let reverse = rings[ringBase + 8u] < 0.0;
  let startPosition = readVertex(select(index, next, reverse));
  let endPosition = readVertex(select(next, index, reverse));
  edges[base] = startPosition.x;
  edges[base + 1u] = startPosition.y;
  edges[base + 2u] = endPosition.x;
  edges[base + 3u] = endPosition.y;`}),y(e,{id:`${r}-summarize`,operation:x,variant:`summarize`,bindings:[z,{...B,access:`read_write`}],invocationCount:1,declarations:`${_}${v}${m}`,body:`var totals = vec3f(0.0);
  var roundingErrors = vec3f(0.0);
  var minimum = vec2f(3.402823466e38);
  var maximum = vec2f(-3.402823466e38);
  var flags = 0u;
  for (var ring = 0u; ring < ${l}u; ring++) {
    let base = ringsOffset + ${S}u * ring;
    if (rings[base + 9u] == 0.0) { continue; }
    let values = vec3f(rings[base], rings[base + 1u], rings[base + 2u]);
    let nextTotals = totals + values;
    roundingErrors += vec3f(
      getAdditionRoundingError(totals.x, values.x, nextTotals.x),
      getAdditionRoundingError(totals.y, values.y, nextTotals.y),
      getAdditionRoundingError(totals.z, values.z, nextTotals.z)
    );
    totals = nextTotals;
    flags |= u32(rings[base + 3u]);
    if (values.z > 0.0) {
      minimum = min(minimum, vec2f(rings[base + 4u], rings[base + 5u]));
      maximum = max(maximum, vec2f(rings[base + 6u], rings[base + 7u]));
    }
  }
  totals += roundingErrors;
  if (!isFiniteBits(totals.x) || totals.x <= 0.0) { flags |= 2u; }
  if (any(minimum > maximum)) { minimum = vec2f(getNaN()); maximum = minimum; }
  summary[summaryOffset] = minimum.x;
  summary[summaryOffset + 1u] = minimum.y;
  summary[summaryOffset + 2u] = maximum.x;
  summary[summaryOffset + 3u] = maximum.y;
  summary[summaryOffset + 4u] = totals.x;
  summary[summaryOffset + 5u] = totals.y;
  summary[summaryOffset + 6u] = totals.z;
  summary[summaryOffset + 7u] = f32(flags);`}));let{nodes:le,slabs:W}=ie(e,{id:r,operation:x,owner:`feature`,positions:c.positions,featureOffsets:c.featureOffsets,polygonOffsets:c.polygonOffsets,ringOffsets:c.ringOffsets,entryCapacity:c.slabEntryCapacity});U.push(...le);let G={...W,params:this.createPersistentView(e,`edge-slab-params`,`uint32`,W.params.length),slabOffsets:this.createPersistentView(e,`edge-slab-offsets`,`uint32`,W.slabOffsets.length),slabEntries:this.createPersistentView(e,`edge-slab-entries`,`uint32x2`,W.slabEntries.length)};for(let t of[`params`,`slabOffsets`,`slabEntries`]){let n=W[t].length*(t===`slabEntries`?2:1);U.push(y(e,{id:`${r}-edges-copy-${t}`,operation:x,variant:`copy-words`,bindings:[{name:`source`,view:W[t],type:`u32`,access:`read`},{name:`destination`,view:G[t],type:`u32`,access:`read_write`}],invocationCount:Math.max(n,1),declarations:`const WORD_COUNT: u32 = ${n}u;`,body:`if (index >= WORD_COUNT) { return; }
  destination[destinationOffset + index] = source[sourceOffset + index];`}))}let K=t(e,`${r}-sample-parameters`,`float32`,4),q=t(e,`${r}-sample-zones`,`uint32`,u),J=t(e,`${r}-sample-boundary`,`uint32`,u),Y=t(e,`${r}-sample-overflow`,`uint32`,1);U.push(y(e,{id:`${r}-sample-parameters`,operation:x,variant:`sample-parameters`,bindings:[B,{name:`parameters`,view:K,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`fn readSummary(slot: u32) -> f32 { return summary[summaryOffset + slot]; }
${d(s,`readSummary`)}`,body:`let cell = getStudyWindowSampleCell(0u);
  for (var slot = 0u; slot < 4u; slot++) { parameters[parametersOffset + slot] = cell[slot]; }`}),...new ee({id:`${r}-raster`,spatialContext:this.spatialContext,grid:{width:s[0],height:s[1],affine:[1,0,0,0,1,0],pixelInterpretation:`area`},parameters:K,positions:c.positions,featureOffsets:c.featureOffsets,polygonOffsets:c.polygonOffsets,ringOffsets:c.ringOffsets,crossingCapacity:c.crossingCapacity??4*s[1]*l,output:{zones:q,boundary:J,status:{overflow:Y}}}).getCommandNodes(e)),U.push(y(e,{id:`${r}-sample-overflow-recovery`,operation:x,variant:`sample-overflow-recovery`,bindings:[B,{name:`edges`,view:j,type:`f32`,access:`read`},{name:`overflow`,view:Y,type:`u32`,access:`read`},{name:`zones`,view:q,type:`u32`,access:`read_write`}],invocationCount:u,declarations:`${g}${ae}
fn readSummary(slot: u32) -> f32 { return summary[summaryOffset + slot]; }
${d(s,`readSummary`)}`,body:`if (overflow[overflowOffset] == 0u) { return; }
  let cell = getStudyWindowSampleCell(index);
  let center = cell.xy + 0.5 * cell.zw;
  let probe = ExactProbe(center, center, center, EXACT_PROBE_POINT);
  var winding = 0;
  for (var vertex = 0u; vertex < ${i}u; vertex++) {
    let base = edgesOffset + 4u * vertex;
    let startPosition = vec2f(edges[base], edges[base + 1u]);
    let endPosition = vec2f(edges[base + 2u], edges[base + 3u]);
    if (!isFiniteBits(startPosition.x) || !isFiniteBits(startPosition.y) ||
        !isFiniteBits(endPosition.x) || !isFiniteBits(endPosition.y)) { continue; }
    winding += exactProbeCrossing(probe, startPosition, endPosition).x;
  }
  zones[zonesOffset + index] = select(${o}u, 0u, (winding & 1) != 0);`}));let X=t(e,`${r}-sample-counts`,`uint32`,u+1),Z=t(e,`${r}-sample-offsets`,`uint32`,u+1);U.push(y(e,{id:`${r}-sample-count`,operation:x,variant:`sample-count`,bindings:[{name:`zones`,view:q,type:`u32`,access:`read`},{name:`boundary`,view:J,type:`u32`,access:`read`},{name:`counts`,view:X,type:`u32`,access:`read_write`}],invocationCount:u+1,body:`var count = 0u;
  if (index < ${u}u) {
    count = select(0u, 1u, zones[zonesOffset + index] != ${o}u || boundary[boundaryOffset + index] != 0u);
  }
  counts[countsOffset + index] = count;`}),...new n({id:`${r}-sample-scan`,input:X,output:Z,mode:`exclusive`}).getCommandNodes(e),y(e,{id:`${r}-sample-compact`,operation:x,variant:`sample-compact`,bindings:[{name:`counts`,view:X,type:`u32`,access:`read`},{name:`offsets`,view:Z,type:`u32`,access:`read`},{name:`boundary`,view:J,type:`u32`,access:`read`},{name:`sampleCells`,view:F,type:`u32`,access:`read_write`},{name:`sampleCellCount`,view:I,type:`u32`,access:`read_write`}],invocationCount:u+1,body:`let count = offsets[offsetsOffset + ${u}u];
  if (index == ${u}u) { sampleCellCount[sampleCellCountOffset] = count; return; }
  if (index >= count) { sampleCells[sampleCellsOffset + index] = 0u; }
  if (counts[countsOffset + index] != 0u) {
    sampleCells[sampleCellsOffset + offsets[offsetsOffset + index]] = index |
      select(0u, ${p}u, boundary[boundaryOffset + index] != 0u);
  }`}));let Q=O,$=this.createPersistentView(e,`packed`,`uint32`,Q.length);for(let[t,n,i]of[[`summary`,M,M.length],[`edges`,j,j.length],[`slabParams`,G.params,G.params.length],[`slabOffsets`,G.slabOffsets,G.slabOffsets.length],[`slabEntries`,G.slabEntries,2*G.slabEntries.length],[`sampleCellCount`,I,1],[`sampleCells`,F,u]])U.push(y(e,{id:`${r}-pack-${t}`,operation:x,variant:`pack`,bindings:[{name:`source`,view:n,type:`u32`,access:`read`},{name:`packed`,view:$,type:`u32`,access:`read_write`}],invocationCount:i,declarations:`override DESTINATION_OFFSET: u32;`,constants:{DESTINATION_OFFSET:Q[t]},body:`packed[packedOffset + DESTINATION_OFFSET + index] = source[sourceOffset + index];`}));return c.output?.summary&&U.push(y(e,{id:`${r}-copy-summary`,operation:x,variant:`copy-summary`,bindings:[B,{name:`destination`,view:c.output.summary,type:`f32`,access:`read_write`}],invocationCount:8,body:`destination[destinationOffset + index] = summary[summaryOffset + index];`})),this.views={edges:j,summary:M,slabs:G,sampleCells:F,sampleCellCount:I,sampleGrid:s,packed:$},[...A.nodes,...U,A.commit(U)]}};export{O as t};
import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as i,T as a,g as o,l as s,s as c,v as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{a as f}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as p}from"./geographic-distribution-validation-BjMQQJgk.js";import{i as ee,n as m,t as h}from"./key-pair-grouping-DgpiCtBP.js";import{t as te}from"./bounded-key-pair-sort-BTKR6XBC.js";import{t as ne}from"./gpu-segment-ring-assembly-VExWprVe.js";import{n as re,r as ie,t as ae}from"./coverage-topology-kernels-ltNrCaNR.js";import{l as g}from"./wgsl-common-DfyXiYB0.js";import{v as _}from"./spatial-context-DwZEd4EZ.js";import{r as v,t as y}from"./tract-coverage-BWGWYaF1.js";import{t as b}from"./define-run-B-brkCe4.js";import{r as x}from"./lines-kx5qHdti.js";import{u as S}from"./pick-Bo3t3hjv.js";import{a as C,l as w,s as T}from"./polygons-D0hzhPvj.js";import{as as E}from"./entry-preview-D70TdSbK.js";var D=`GPUCoverageDissolve`,O=`${h}u`,k=`0x80000000u`,oe=1e-6;function A(e){let{boundarySegmentCount:t,uncertainCount:n,...r}=e.status;return{boundarySegmentCount:t,uncertainCount:n,ringOutput:{...e,status:r}}}var j=class{id;props;spatialContext;constructor(e){let n=e.id??`coverage-dissolve`;if(this.id=n,this.props=e,this.spatialContext=_(n,e.spatialContext),i(e.positions,[`float32x2`],`${n} positions`),e.positions.length<1||e.positions.length>=2**30)throw Error(`${n} positions must hold between 1 and 2^30 - 1 vertices`);t(e.ringOffsets,`${n} ringOffsets`),t(e.polygonOffsets,`${n} polygonOffsets`),t(e.featureOffsets,`${n} featureOffsets`);for(let[t,r]of[[`ringOffsets`,e.ringOffsets],[`polygonOffsets`,e.polygonOffsets],[`featureOffsets`,e.featureOffsets]])if(r.length<2)throw Error(`${n} ${t} must hold at least two entries`);if(t(e.labels,`${n} labels`),e.labels.length!==e.featureOffsets.length-1)throw Error(`${n} labels length must equal the feature count`);if(e.snapTolerance!==void 0&&(!Number.isFinite(e.snapTolerance)||e.snapTolerance<0))throw Error(`${n} snapTolerance must be a finite number >= 0`);if(e.vertexTolerance!==void 0&&!(e.vertexTolerance>0))throw Error(`${n} vertexTolerance must be positive`);e.snapTolerance!==void 0&&o(n,`snapTolerance`,e.snapTolerance,0),e.vertexTolerance!==void 0&&(o(n,`vertexTolerance`,e.vertexTolerance,0),a(n,Math.fround(e.vertexTolerance)>0,`vertexTolerance must remain positive after float32 conversion`));let{boundarySegmentCount:r,uncertainCount:s,ringOutput:l}=A(e.output);for(let[e,i]of[[`boundarySegmentCount`,r],[`uncertainCount`,s]])if(i&&(t(i,`${n} output.status.${e}`),i.length<1))throw Error(`${n} output.status.${e} must hold one uint32`);i(l.positions,[`float32x2`],`${n} output.positions`),d(n,[l.ringOffsets,l.positions,l.ringAreas,l.ringIsHole,l.ringShells,l.ringGroups,l.polygons?.positions,l.polygons?.ringOffsets,l.polygons?.polygonOffsets,l.polygons?.featureOffsets,l.polygons?.sourceIds,l.status.count,l.status.overflow,l.status.requiredCount,l.status.openSegmentCount,l.status.touchingSegmentCount,r,s],[e.positions,e.ringOffsets,e.polygonOffsets,e.featureOffsets,e.labels]),p(n,{},Object.fromEntries([...c(e.output),...l.polygons?c(l.polygons):[]].map((e,t)=>[`output.${t}`,e])),[])}getCommandNodes(t){let{id:i,props:a}=this,{positions:o,ringOffsets:d,polygonOffsets:p,featureOffsets:h,labels:_}=a,{boundarySegmentCount:v,uncertainCount:y,ringOutput:b}=A(a.output);u(i,t,[o,d,p,h,_,v,y]),f(t,i,Object.fromEntries([o,d,p,h,_,...c(a.output),...b.polygons?c(b.polygons):[]].map((e,t)=>[`binding.${t}`,e])));let x=b.ringOffsets.length-1;for(let[e,n]of[[`dissolve endpoint scratch`,o.length*16],[`dissolve polygon labels scratch`,(p.length-1)*4],[`dissolve ring orientation scratch`,(d.length-1)*4],[`assembly ring counters scratch`,24],[`assembly frame partials scratch`,Math.min(256,Math.max(1,Math.ceil(o.length/4096)))*16],...b.ringShells||b.polygons?[[`assembly shell records scratch`,x*12],[`assembly shell bounds scratch`,x*16],[`assembly shell hierarchy scratch`,(2*s(x)-1)*8]]:[]])l(t.device,i,e,Math.max(n,4),`reduce vertex or ring capacity`);let S=o.length,C=d.length-1,w=p.length-1,T=h.length-1,E=a.snapTolerance??0,j=(n,r)=>e(t,`${i}-${n}`,`uint32`,r),M=j(`polygon-labels`,w),N=j(`ring-flip`,C),P=j(`vertex-high`,S),F=j(`vertex-low`,S),I=j(`vertex-polygon`,S),L=j(`vertex-next`,S),R=[r(t,{id:`${i}-polygon-labels`,operation:D,variant:`polygon-labels`,bindings:[{name:`featureOffsets`,view:h,type:`u32`,access:`read`},{name:`featureLabels`,view:_,type:`u32`,access:`read`},{name:`polygonLabels`,view:M,type:`u32`,access:`read_write`}],invocationCount:w,declarations:`const FEATURE_COUNT: u32 = ${T}u;`,body:`var feature = 0u;
  var high = FEATURE_COUNT;
  while (feature + 1u < high) {
    let middle = (feature + high) / 2u;
    if (featureOffsets[featureOffsetsOffset + middle] <= index) {
      feature = middle;
    } else {
      high = middle;
    }
  }
  polygonLabels[polygonLabelsOffset + index] = featureLabels[featureLabelsOffset + feature];`}),re(t,{id:`${i}-ring-orientation`,operation:D,positions:o,ringOffsets:d,polygonOffsets:p,ringFlip:N}),r(t,{id:`${i}-vertex-topology`,operation:D,variant:`vertex-topology`,bindings:[{name:`positions`,view:o,type:`f32`,access:`read`},{name:`ringOffsets`,view:d,type:`u32`,access:`read`},{name:`polygonOffsets`,view:p,type:`u32`,access:`read`},{name:`ringFlip`,view:N,type:`u32`,access:`read`},{name:`vertexHigh`,view:P,type:`u32`,access:`read_write`},{name:`vertexLow`,view:F,type:`u32`,access:`read_write`},{name:`vertexPolygon`,view:I,type:`u32`,access:`read_write`},{name:`vertexNext`,view:L,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`const RING_COUNT: u32 = ${C}u;
const POLYGON_COUNT: u32 = ${w}u;

${g}
${ie(E,C)}`,body:`var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  var polygon = 0u;
  high = POLYGON_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= ring) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let inRing = index >= ringBegin && index < ringEnd &&
    ring >= polygonOffsets[polygonOffsetsOffset] && ring < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT] &&
    ring < polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  let valid = inRing && isFiniteFloat(x) && isFiniteFloat(y);
  let snapOrigin = getSnapOrigin();
  vertexHigh[vertexHighOffset + index] = select(${O}, quantise(x, snapOrigin.x), valid);
  vertexLow[vertexLowOffset + index] = select(${O}, quantise(y, snapOrigin.y), valid);
  vertexPolygon[vertexPolygonOffset + index] = select(${O}, polygon, valid);
  let next = select(index + 1u, ringBegin, index + 1u == ringEnd);
  vertexNext[vertexNextOffset + index] = next | select(0u, ${k}, ringFlip[ringFlipOffset + ring] != 0u);`})],z=ee(t,`${i}-vertex`,D,S,P,F),B=m(t,`${i}-vertex`,D,S,P,F,z.sortedItems),V=j(`point-ids`,S),H=e(t,`${i}-canonical-positions`,`float32x2`,S),U=j(`edge-high`,S),W=j(`edge-low`,S);R.push(...z.nodes,...B.nodes,r(t,{id:`${i}-point-ids`,operation:D,variant:`point-ids`,bindings:[{name:`sortedItems`,view:z.sortedItems,type:`u32`,access:`read`},{name:`groupIndex`,view:B.groupIndex,type:`u32`,access:`read`},{name:`groupStarts`,view:B.groupStarts,type:`u32`,access:`read`},{name:`vertexPolygon`,view:I,type:`u32`,access:`read`},{name:`positions`,view:o,type:`f32`,access:`read`},{name:`pointIds`,view:V,type:`u32`,access:`read_write`},{name:`canonicalPositions`,view:H,type:`f32`,access:`read_write`}],invocationCount:S,body:`let vertex = sortedItems[sortedItemsOffset + index];
  let groupId = groupIndex[groupIndexOffset + index] - 1u;
  let valid = vertexPolygon[vertexPolygonOffset + vertex] != ${O};
  let canonical = sortedItems[sortedItemsOffset + groupStarts[groupStartsOffset + groupId]];
  pointIds[pointIdsOffset + vertex] = select(${O}, groupId, valid);
  let source = select(vertex, canonical, valid);
  canonicalPositions[canonicalPositionsOffset + vertex * 2u] = positions[positionsOffset + source * 2u];
  canonicalPositions[canonicalPositionsOffset + vertex * 2u + 1u] = positions[positionsOffset + source * 2u + 1u];`}),r(t,{id:`${i}-edge-keys`,operation:D,variant:`edge-keys`,bindings:[{name:`vertexPolygon`,view:I,type:`u32`,access:`read`},{name:`vertexNext`,view:L,type:`u32`,access:`read`},{name:`pointIds`,view:V,type:`u32`,access:`read`},{name:`edgeHigh`,view:U,type:`u32`,access:`read_write`},{name:`edgeLow`,view:W,type:`u32`,access:`read_write`}],invocationCount:S,body:`let first = pointIds[pointIdsOffset + index];
  let second = pointIds[pointIdsOffset + (vertexNext[vertexNextOffset + index] & 0x7fffffffu)];
  let valid = first != ${O} && second != ${O} && first != second &&
    vertexPolygon[vertexPolygonOffset + index] != ${O};
  edgeHigh[edgeHighOffset + index] = select(${O}, min(first, second), valid);
  edgeLow[edgeLowOffset + index] = select(${O}, max(first, second), valid);`}));let G=te(t,`${i}-edge`,D,S,U,W,{lowKeyLimit:S,highKeyLimit:S}),K=m(t,`${i}-edge`,D,S,U,W,G.sortedItems);R.push(...G.nodes,...K.nodes);let q=j(`keep-flag`,S),J=j(`keep-rank`,S);R.push(r(t,{id:`${i}-edge-keep`,operation:D,variant:`edge-keep`,bindings:[{name:`sortedItems`,view:G.sortedItems,type:`u32`,access:`read`},{name:`edgeHigh`,view:U,type:`u32`,access:`read`},{name:`vertexPolygon`,view:I,type:`u32`,access:`read`},{name:`groupIndex`,view:K.groupIndex,type:`u32`,access:`read`},{name:`groupStarts`,view:K.groupStarts,type:`u32`,access:`read`},{name:`labels`,view:M,type:`u32`,access:`read`},{name:`keepFlag`,view:q,type:`u32`,access:`read_write`}],invocationCount:S,body:`let item = sortedItems[sortedItemsOffset + index];
  let polygon = vertexPolygon[vertexPolygonOffset + item];
  var keep = polygon != ${O} && edgeHigh[edgeHighOffset + item] != ${O};
  if (keep) {
    let label = labels[labelsOffset + polygon];
    let groupId = groupIndex[groupIndexOffset + index] - 1u;
    let end = groupStarts[groupStartsOffset + groupId + 1u];
    for (var other = groupStarts[groupStartsOffset + groupId]; other < end; other++) {
      let otherPolygon = vertexPolygon[vertexPolygonOffset + sortedItems[sortedItemsOffset + other]];
      if (otherPolygon != ${O} && otherPolygon != polygon && labels[labelsOffset + otherPolygon] == label) {
        keep = false;
      }
    }
  }
  keepFlag[keepFlagOffset + item] = select(0u, 1u, keep);`}),...new n({id:`${i}-keep-scan`,input:q,output:J,mode:`exclusive`}).getCommandNodes(t));let Y=e(t,`${i}-segments-full`,`float32x4`,S),X=j(`groups-full`,S),Z=e(t,`${i}-segments`,`float32x4`,S),Q=j(`segment-groups`,S),$=j(`segment-count`,1);R.push(r(t,{id:`${i}-segment-endpoints`,operation:D,variant:`segment-endpoints`,bindings:[{name:`canonicalPositions`,view:H,type:`f32`,access:`read`},{name:`vertexNext`,view:L,type:`u32`,access:`read`},{name:`vertexPolygon`,view:I,type:`u32`,access:`read`},{name:`labels`,view:M,type:`u32`,access:`read`},{name:`segmentsFull`,view:Y,type:`f32`,access:`read_write`},{name:`groupsFull`,view:X,type:`u32`,access:`read_write`}],invocationCount:S,body:`let entry = vertexNext[vertexNextOffset + index];
  let next = entry & 0x7fffffffu;
  let flip = (entry & ${k}) != 0u;
  let startVertex = select(index, next, flip);
  let endVertex = select(next, index, flip);
  segmentsFull[segmentsFullOffset + index * 4u] = canonicalPositions[canonicalPositionsOffset + startVertex * 2u];
  segmentsFull[segmentsFullOffset + index * 4u + 1u] = canonicalPositions[canonicalPositionsOffset + startVertex * 2u + 1u];
  segmentsFull[segmentsFullOffset + index * 4u + 2u] = canonicalPositions[canonicalPositionsOffset + endVertex * 2u];
  segmentsFull[segmentsFullOffset + index * 4u + 3u] = canonicalPositions[canonicalPositionsOffset + endVertex * 2u + 1u];
  let polygon = vertexPolygon[vertexPolygonOffset + index];
  groupsFull[groupsFullOffset + index] = select(0u, labels[labelsOffset + polygon], polygon != ${O});`}),r(t,{id:`${i}-compact`,operation:D,variant:`compact`,bindings:[{name:`segmentsFull`,view:Y,type:`f32`,access:`read`},{name:`groupsFull`,view:X,type:`u32`,access:`read`},{name:`keepFlag`,view:q,type:`u32`,access:`read`},{name:`keepRank`,view:J,type:`u32`,access:`read`},{name:`segments`,view:Z,type:`f32`,access:`read_write`},{name:`segmentGroups`,view:Q,type:`u32`,access:`read_write`},{name:`segmentCount`,view:$,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`const VERTEX_COUNT: u32 = ${S}u;`,body:`let flag = keepFlag[keepFlagOffset + index];
  let rank = keepRank[keepRankOffset + index];
  if (flag != 0u) {
    for (var component = 0u; component < 4u; component++) {
      segments[segmentsOffset + rank * 4u + component] = segmentsFull[segmentsFullOffset + index * 4u + component];
    }
    segmentGroups[segmentGroupsOffset + rank] = groupsFull[groupsFullOffset + index];
  }
  if (index == VERTEX_COUNT - 1u) {
    segmentCount[segmentCountOffset] = rank + flag;
  }`})),v&&R.push(r(t,{id:`${i}-publish-boundary-count`,operation:D,invocationCount:1,bindings:[{name:`countIn`,view:$,type:`u32`,access:`read`},{name:`countOut`,view:v,type:`u32`,access:`read_write`}],body:`countOut[countOutOffset] = countIn[countInOffset];`}));let se=a.vertexTolerance??(E>0?E/4:oe);return R.push(...new ne({id:`${i}-assembly`,endpoints:Z,count:$,groups:Q,vertexTolerance:se,interiorSide:`left`,spatialContext:this.spatialContext,splitTouchingRings:a.splitTouchingRings,output:b}).getCommandNodes(t)),y&&R.push(...ae(t,{id:`${i}-snap-range`,operation:D,positions:o,ringOffsets:d,snapTolerance:E,uncertainCount:y})),R}},M=b({parameters:E,caption:`Cells merged by district label; internal boundaries disappear.`,createInputs:({random:e,world:t})=>y(e,t,5),build(e,t,n){let r=w(e,`coverage`,t),i=e.allocate(`labels`,`uint32`,t.featureCount),a=t.ringOffsets.length,o=t.positions.length/2+a,s=e.allocate(`dissolved-ring-offsets`,`uint32`,a+1),c=e.allocate(`dissolved-positions`,`float32x2`,o),l=e.allocate(`dissolved-areas`,`float32`,a),u=e.allocate(`dissolved-groups`,`uint32`,a),d=e.allocate(`dissolved-count`,`uint32`,1),f=e.allocate(`dissolved-overflow`,`uint32`,1);e.graph.add(new j({...T(r),labels:i.view,snapTolerance:1e-5,output:{ringOffsets:s.view,positions:c.view,ringAreas:l.view,ringGroups:u.view,status:{count:d.view,overflow:f.view}}}));let p=x(e,`dissolved`,c,s,{closed:!0,lineCount:d,lineValues:u});return{outputs:{original:e.upload(`original`,`float32x4`,C(t)),...p},update:e=>{i.write(v(t.contiguity,e.districtSize))}}},interaction:{kind:`probe`,hint:`hover a tract`,resolve(e,{inputs:t,values:n}){let r=-1;for(let n=0;n<t.featureCount&&r<0;n++){let i=t.ringOffsets[n],a=t.ringOffsets[n+1]-i;S(t.positions,e.x,e.y,i,a)&&(r=n)}if(r<0)return{};let i=v(t.contiguity,n.districtSize),a=[],o=0,s=0;for(let e=0;e<t.featureCount;e++)if(i[e]===i[r]){o++,a.push({shape:`polyline`,points:Array.from(t.positions.subarray(t.ringOffsets[e]*2,t.ringOffsets[e+1]*2)),closed:!0,dashed:e!==r});for(let n=t.contiguity.offsets[e];n<t.contiguity.offsets[e+1];n++)i[t.contiguity.neighbors[n]]===i[r]&&s++}return{focus:r,glyphs:a,readout:`district ${i[r]} · ${o} tracts · ${s/2} borders dropped`}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.original,radius:1.5},{layer:`segments`,nodes:e.points,edges:e.edges,values:e.lineIds,radius:4}]});export{M as default};
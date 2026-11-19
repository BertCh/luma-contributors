import{t as e}from"./ports-xmdKgL07.js";import{n as t,r as n}from"./lon-lat-ring-wgsl-DpZBIfuE.js";import{i as r,o as i,t as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as o,s}from"./kernel-C3tfQOn4.js";import{l as c}from"./gpu-command-graph-BdzdmV6u.js";import{n as l}from"./gpu-pairwise-point-in-polygon-DCUVp3i5.js";import{t as u}from"./exact-orientation-wgsl-B_jI-HEc.js";import{_ as d,n as f,v as p}from"./spatial-context-DwZEd4EZ.js";import{i as m,n as h,r as ee,t as g}from"./polygon-edge-slabs-CRnrFvP2.js";import{E as _,c as v,f as y,l as b,m as x,p as S,u as C,v as w,w as T}from"./gpu-contributor-utils-BIwLQON5.js";import{C as E,D,S as te,_ as ne,a as O,d as k,f as A,g as re,i as ie,o as j,s as M,u as N,v as P,w as F,x as ae,y as oe}from"./spatial-join-passes-CNxNpnDa.js";import{c as I,d as L,l as R,o as z,s as B,u as se}from"./spatial-join-geometry-DYRzw652.js";import{Q as V}from"./dist-BusamGlx.js";import{i as ce}from"./graph-chunk-groups-BB8ACpQO.js";var H=`GPUPointInPolygonJoin`,U=4294967295,W=2147483646,le=4,ue=1024,G=1024,K=80,q=8;function J(e){return e.coordinateSpace===`longitude-latitude`?{coordinateSpace:`longitude-latitude`,metric:`none`,...e.edgeModel===void 0?{}:{edgeModel:e.edgeModel},...e.datum===void 0?{}:{datum:e.datum}}:e}var Y=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),edgeModels:f,metrics:Object.freeze([`none`,`native`])}),X=256,Z=class{id;props;featureCount;leafCapacity;candidateCapacity;candidatePages;boundaryRule;spatialContext;spatialSort;querySort;edgeSlabs;estimatedCommandNodeCount;constructor(t){this.id=t.id??`point-in-polygon-join`;let{id:n}=this,r=e(n,`points`,t.points),i={...t,points:{kind:r.kind,positions:r.positions,sourceIds:r.sourceIds}};this.props=i;let{points:a,polygons:o,output:s}=i,{status:l}=s;if(a.kind!==`points`||o.kind!==`polygons`)throw Error(`${n} needs point and polygon geometry`);I(n,`points`,a);for(let e of v(a.positions))_(e,[`float32x2`],`${n} points.positions`);for(let[e,t]of Object.entries({...Q(i),...$(i)}))if(t){y(n,`${e}.length`,t.length,0,U);for(let r of v(t))y(n,`${e} chunk.length`,r.length,0,U)}B(n,`polygons`,o,{allowEmpty:!0}),this.featureCount=o.featureOffsets.length-1;for(let[e,t]of[[`output.pointFeatureIds`,s.pointFeatureIds],[`points.sourceIds`,a.sourceIds]]){for(let r of t?v(t):[])_(r,[`uint32`],`${n} ${e}`);F(n,e,a.positions,t)}for(let[e,t]of[[`polygons.sourceIds`,o.sourceIds],[`output.featureCounts`,s.featureCounts]])if(t&&(_(t,[`uint32`],`${n} ${e}`),t.length!==this.featureCount))throw Error(`${n} ${e} needs one row per feature`);if(R(n,l),s.candidateTruncated){for(let e of v(s.candidateTruncated))_(e,[`uint32`],`${n} output.candidateTruncated`);if(s.candidateTruncated instanceof c&&F(n,`output.candidateTruncated`,a.positions,s.candidateTruncated),s.candidateTruncated.length!==a.positions.length)throw Error(`${n} output.candidateTruncated needs one row per point`)}if(s.matches&&x(n,s.matches),this.candidateCapacity=i.candidateCapacity,y(n,`candidateCapacity`,this.candidateCapacity,1,W),this.leafCapacity=i.leafCapacity??b(Math.max(this.featureCount,1)),T(n,`leafCapacity`,this.leafCapacity),i.spatialContext?.coordinateSpace===`longitude-latitude`&&i.spatialContext.metric!==`none`)throw Error(`${n} needs metric 'none' on longitude-latitude`);if(this.spatialContext=p(n,i.spatialContext,Y),d(n,`polygons`,i.polygonsSpatialContext,this.spatialContext,Y.edgeModels),i.boundaryRule!==void 0&&![`half-open`,`closed`,`open`].includes(i.boundaryRule))throw Error(`${n} boundaryRule must be 'half-open', 'closed' or 'open'`);if(this.boundaryRule=i.boundaryRule??`half-open`,this.spatialSort=i.prepared?.spatialSort??i.spatialSort??A(this.featureCount),this.querySort=i.querySort??k(i.points.positions.length,this.featureCount),this.edgeSlabs=i.edgeSlabs??i.prepared?.edgeSlabs??N(o.positions.length,this.candidateCapacity,this.featureCount),i.prepared&&(this.leafCapacity=i.prepared.leafCapacity,!z(i.prepared.geometry,o)))throw Error(`${n} prepared must index the same polygon views`);y(n,`point offset scratch words`,a.positions.length+5*v(a.positions).length,0,U),this.candidatePages=D(o.positions.buffer.graph.device,n,this.candidateCapacity,i.candidateTileCapacity,this.edgeSlabs?16:8,0,v(a.positions).length);let u=this.querySort&&v(a.positions).some(e=>e.length>=2),f=v(a.positions).filter(e=>e.length>0).length;this.estimatedCommandNodeCount=this.candidatePages.estimatedCommandNodeCount+G+f*(u?K:q),S(n,this.estimatedCommandNodeCount),E(n,Object.values(Q(i,this.edgeSlabs)),Object.values($(i)),[{pointFeatureIds:s.pointFeatureIds,featureCounts:s.featureCounts},{overflow:s.status.overflow,candidateCount:s.status.candidateCount,uncertainCount:s.status.uncertainCount,matchCount:s.matches?.status.count,matchOverflow:s.matches?.status.overflow,matchRequired:s.matches?.status.requiredCount},...this.spatialContext.coordinateSpace===`longitude-latitude`?[{antimeridian:s.status.antimeridianUncertainCount,pole:s.status.poleUncertainCount}]:[]]),a.positions.length>0&&L(n,[o.sourceIds],[s.pointFeatureIds,s.featureCounts]),i.prepared&&L(n,[(i.prepared.isDeclaredIn(o.positions.buffer.graph)?i.prepared.storage:i.prepared.props.storage)?.overflow],[s.status.overflow,s.status.candidateCount,s.status.uncertainCount,s.matches?.status.count,s.matches?.status.overflow,s.matches?.status.requiredCount])}getCommandNodes(e){let{id:d,props:f,featureCount:p,leafCapacity:m,candidateCapacity:h}=this,{polygons:g,output:_}=f,b=f.points.positions,{matches:x,status:S}=_;if(f.prepared&&!f.prepared.isDeclaredIn(e))throw Error(`${d} requires prepared to be added to the graph first`);let T={...Q(f,this.edgeSlabs),...$(f)};se(e,d,T,{});let E=(t,n,r)=>{w(e.device,d,t,n,r),y(d,`${t} byteLength (device.limits.maxBufferSize)`,Math.max(4,n),0,e.device.limits.maxBufferSize)};if(E(`point assignment, probe and match scratch`,(b.length+5*v(b).length)*Uint32Array.BYTES_PER_ELEMENT,`reduce points.positions.length; join scratch requires one packed binding`),E(`state`,20,`use a device supporting at least 20-byte storage bindings`),E(`candidate pairs`,this.candidatePages.capacity*(this.edgeSlabs?4:2)*Uint32Array.BYTES_PER_ELEMENT,`reduce candidateCapacity`),!f.prepared)E(`BVH node bounds and children`,(2*m-1)*8,`reduce leafCapacity`),E(`BVH leaf IDs`,m*4,`reduce leafCapacity`),E(`feature bounds and sorted bounds`,p*8,`reduce polygons feature count`),E(`feature radix histogram`,16*Math.ceil(p/256)*4,`reduce polygons feature count`),E(`feature sort tile bounds`,Math.ceil(p/256)*16,`reduce polygons feature count`);else for(let e of Object.values(f.prepared.storage))E(`prepared BVH storage`,s(e).size,`reduce prepared leafCapacity`);this.querySort&&(E(`point query sort`,b.length*4,`reduce points.positions.length`),E(`point query radix histogram`,16*Math.ceil(b.length/256)*4,`reduce points.positions.length`));let D;if(this.edgeSlabs){let t=f.prepared?.edgeSlabStorage;if(t)for(let e of[t.slabs.params,t.slabs.slabOffsets,t.slabs.slabEntries,t.layout])E(`prepared edge slab storage`,s(e).size,`reduce the prepared polygon index size`);else{let t=g.positions.length,n=g.polygonOffsets.length-1;E(`edge slab parameters`,Math.max(n,1)*6*4,`reduce polygons part count or disable edgeSlabs`),E(`edge slab owner scan`,(n+1)*4,`reduce polygons part count or disable edgeSlabs`),E(`edge slab ring owners`,Math.max(g.ringOffsets.length-1,1)*4,`reduce polygons ring count or disable edgeSlabs`),E(`edge slab vertex owners`,t*4,`reduce polygons vertex count or disable edgeSlabs`),E(`edge slab offsets`,(Math.floor(t/4)+2)*4,`reduce polygons vertex count or disable edgeSlabs`),E(`edge slab base edges`,t*8,`reduce polygons vertex count or disable edgeSlabs`),D=Math.min(C(e.device,8,t),Math.max(le*t,ue)),E(`edge slab entries`,(t+D)*8,`reduce polygons vertex count or disable edgeSlabs`),E(`edge slab layout`,re(t,p)*4,`reduce polygons feature count or disable edgeSlabs`)}}ce(e,d,8);let k=b.length,A=[];if(this.spatialContext.coordinateSpace===`longitude-latitude`&&n(S)){for(let t of[`antimeridianUncertainCount`,`poleUncertainCount`]){let n=S[t];n&&A.push(a(e,{id:`${d}-clear-${t}`,operation:H,view:n,type:`u32`,value:`0u`,componentCount:1}))}A.push(t(e,{id:`${d}-lon-lat-polygon-check-0`,operation:H,positions:g.positions,ringOffsets:g.ringOffsets,status:S}))}let N;if(f.prepared){let{storage:e}=f.prepared;N={...e,internalNodeCount:f.prepared.bvhInternalNodeCount}}else{let t=o(e,`${d}-feature-minima`,`float32x2`,p),n=o(e,`${d}-feature-maxima`,`float32x2`,p);p>0&&A.push(ie(e,{id:`${d}-bounds`,operation:H,featureCount:p,source:g,minima:t,maxima:n}));let r=P(e,d,H,t,n,m,this.spatialSort,void 0,this.props.spatialSortCurve);A.push(...r.nodes),N=r.bvh}let F=o(e,`${d}-state`,`uint32`,5),I=o(e,`${d}-assignment`,`uint32`,k),L=this.edgeSlabs,R=L?4:2,z=o(e,`${d}-candidate-pairs`,`uint32x2`,this.candidatePages.capacity*(R/2)),B=L?void 0:o(e,`${d}-candidate-points`,`float32x2`,this.candidatePages.capacity),U,W;if(L){let t=f.prepared?.edgeSlabStorage;if(t)({slabs:U,layout:W}=t);else{let t=ee(e,{id:`${d}-edges`,operation:H,owner:`polygon`,entryCapacity:D||void 0,positions:g.positions,polygonOffsets:g.polygonOffsets,ringOffsets:g.ringOffsets}),n=ne(e,{id:`${d}-edges`,operation:H,polygons:g});A.push(...t.nodes,...n.nodes),U=t.slabs,W=n.layout}}let G=o(e,`${d}-pair-classifications`,`uint32`,this.candidatePages.capacity);A.push(j(e,{id:`${d}-clear`,operation:H,state:F,assignment:I,pointCount:k,featureCounts:_.featureCounts}));let K=_.candidateTruncated,q=K instanceof c?o(e,`${d}-candidate-truncated`,`uint32`,k):K,J=o(e,`${d}-page-counts`,`uint32`,k),Y=o(e,`${d}-page-offsets`,`uint32`,k+5*v(b).length,V.STORAGE|V.INDIRECT),Z=new Map;for(let t=0;t<this.candidatePages.pageCount;t++){let n=this.candidatePages.pageCount>1?`${this.id}-tile-${t}`:this.id,a=this.candidatePages.capacity,o=t*a,s=Math.min(a,this.candidateCapacity-o);if(A.push(...te(e,{id:`${n}-probe`,operation:H,points:b,bvh:N,featureCount:p,candidateCapacity:this.candidateCapacity,page:{firstSlot:o,length:s,initialize:t===0,counts:J,offsets:Y,orders:Z,metadataRow:k},state:F,candidatePairs:z,candidatePoints:B,packedCandidates:L,pointRowScale:1,pointRowOffset:0,querySort:this.querySort,candidateTruncated:q,candidateOverflow:_.status.candidateOverflow,bvhOverflow:N.overflow})),t===0&&_.candidateTruncated instanceof c&&q){let t=0;for(let[r,i]of _.candidateTruncated.data.entries())i.length>0&&A.push(O(e,{id:`${n}-candidate-truncated-${r}`,operation:H,bindings:[{name:`flags`,view:q,type:`u32`,access:`read`},{name:`outputFlags`,view:i,type:`u32`,access:`read_write`}],invocationCount:i.length,firstRow:t,body:`outputFlags[outputFlagsOffset + index] = flags[flagsOffset + CHUNK_FIRST_ROW + index];`})),t+=i.length}A.push(O(e,{id:`${n}-page-count`,operation:H,invocationCount:1,firstRow:0,candidatePageFirstSlot:o,candidatePageLength:s,bindings:[{name:`state`,view:F,type:`u32`,access:`read_write`}],body:`let retained = min(state[stateOffset], ${this.candidateCapacity}u);
  let first = min(CANDIDATE_PAGE_FIRST_SLOT, retained);
  state[stateOffset + 4u] = min(retained - first, CANDIDATE_PAGE_LENGTH);`}));let d=r(e,{id:`${n}-candidate-dispatch`,operation:H,count:F,countRow:4,maximumItemCount:a});A.push(d.updateNode),U&&W?A.push(this.createSlabClassifyNode(e,{condition:d.condition,id:n,capacity:a,state:F,candidatePairs:z,pairClassifications:G,slabs:U,edgeLayout:W})):A.push(i(e,{id:`${n}-classify`,operation:H,variant:`classify-exact`,bindings:[{name:`state`,view:F,type:`u32`,access:`read`},{name:`candidatePairs`,view:z,type:`u32`,access:`read`},{name:`candidatePoints`,view:B,type:`f32`,access:`read`},{name:`featureOffsets`,view:g.featureOffsets,type:`u32`,access:`read`},{name:`polygonPositions`,view:g.positions,type:`f32`,access:`read`},{name:`polygonOffsets`,view:g.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:g.ringOffsets,type:`u32`,access:`read`},{name:`pairClassifications`,view:G,type:`u32`,access:`read_write`}],invocationCount:a,condition:d.condition,declarations:`const CANDIDATE_CAPACITY: u32 = ${a}u;
  const POLYGON_COUNT: u32 = ${g.polygonOffsets.length-1}u;
  const RING_COUNT: u32 = ${g.ringOffsets.length-1}u;
  const VERTEX_COUNT: u32 = ${g.positions.length}u;
  const OUTSIDE: u32 = ${l.outside}u;
  const INSIDE: u32 = ${l.inside}u;
  const BOUNDARY: u32 = ${l.boundary}u;
  const UNCERTAIN: u32 = ${l.uncertain}u;
  const HALF_OPEN_OWNER_BIT: u32 = ${X}u;
  ${u}
  fn readVertex(vertexIndex: u32) -> vec2f {
    return vec2f(
      polygonPositions[polygonPositionsOffset + vertexIndex * 2u],
      polygonPositions[polygonPositionsOffset + vertexIndex * 2u + 1u]
    );
  }
  // Exact even/odd classification, or UNCERTAIN when some orientation has no sign. A BOUNDARY result
  // also carries HALF_OPEN_OWNER_BIT when the point belongs to the feature under the half-open rule.
  //
  // Half-open rule: classify the point moved by (epsilon^2, epsilon) with epsilon infinitesimal. The
  // crossing test against an edge only changes when the point is on the edge's supporting line, where
  // the sign of the moved point is the sign of dx (the epsilon term), or of -dy for a vertical edge
  // (the epsilon^2 term). The y comparisons are unchanged because a vertex y is either equal to the
  // point's y or at least one ulp away.
  fn classifyExactly(point: vec2f, geometryStart: u32, geometryEnd: u32) -> u32 {
    var geometryInside = false;
    var geometryOwned = false;
    var onBoundary = false;
    for (var polygonIndex = geometryStart; polygonIndex < geometryEnd; polygonIndex++) {
      let polygonStart = polygonOffsets[polygonOffsetsOffset + polygonIndex];
      let polygonEnd = polygonOffsets[polygonOffsetsOffset + polygonIndex + 1u];
      // Malformed offsets cannot be classified; an empty part contains nothing (as in shapely) and
      // must not hide the other parts of the feature.
      if (polygonStart > polygonEnd || polygonEnd > RING_COUNT) { return UNCERTAIN; }
      var polygonInside = false;
      var polygonOwned = false;
      for (var ringIndex = polygonStart; ringIndex < polygonEnd; ringIndex++) {
        let ringStart = ringOffsets[ringOffsetsOffset + ringIndex];
        let ringEnd = ringOffsets[ringOffsetsOffset + ringIndex + 1u];
        if (ringEnd < ringStart || ringEnd > VERTEX_COUNT) { return UNCERTAIN; }
        // A ring with fewer than 3 vertices bounds no area: skip it, as shapely treats empty parts.
        if (ringEnd < ringStart + 3u) { continue; }
        var previous = readVertex(ringEnd - 1u);
        for (var vertexIndex = ringStart; vertexIndex < ringEnd; vertexIndex++) {
          let current = readVertex(vertexIndex);
          let sign = orientSign(previous, current, point);
          if (sign == 2) { return UNCERTAIN; }
          var side = sign;
          if (sign == 0) {
            if (point.x >= min(previous.x, current.x) && point.x <= max(previous.x, current.x) &&
                point.y >= min(previous.y, current.y) && point.y <= max(previous.y, current.y)) {
              onBoundary = true;
            }
            side = select(select(0, -1, current.x < previous.x), 1, current.x > previous.x);
            if (side == 0) {
              side = select(select(0, 1, current.y < previous.y), -1, current.y > previous.y);
            }
          }
          if ((previous.y > point.y) != (current.y > point.y)) {
            if (current.y > previous.y && sign > 0) { polygonInside = !polygonInside; }
            if (current.y < previous.y && sign < 0) { polygonInside = !polygonInside; }
            if ((current.y > previous.y && side > 0) || (current.y < previous.y && side < 0)) {
              polygonOwned = !polygonOwned;
            }
          }
          previous = current;
        }
      }
      geometryInside = geometryInside || polygonInside;
      geometryOwned = geometryOwned || polygonOwned;
    }
    if (onBoundary) { return BOUNDARY | select(0u, HALF_OPEN_OWNER_BIT, geometryOwned); }
    return select(OUTSIDE, INSIDE, geometryInside);
  }`,body:`let activeCount = min(state[stateOffset + 4u], CANDIDATE_CAPACITY);
    if (index >= activeCount) { return; }
    let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let geometryStart = featureOffsets[featureOffsetsOffset + featureRow];
    let geometryEnd = featureOffsets[featureOffsetsOffset + featureRow + 1u];
    var classification = UNCERTAIN;
    if (geometryStart <= geometryEnd && geometryEnd <= POLYGON_COUNT) {
      let point = vec2f(candidatePoints[candidatePointsOffset + index * 2u], candidatePoints[candidatePointsOffset + index * 2u + 1u]);
      classification = classifyExactly(point, geometryStart, geometryEnd);
    }
    pairClassifications[pairClassificationsOffset + index] = classification;`})),A.push(i(e,{id:`${n}-resolve`,operation:H,variant:`resolve`,bindings:[{name:`state`,view:F,type:`atomic<u32>`,access:`read_write`},{name:`candidatePairs`,view:z,type:`u32`,access:`read`},{name:`pairClassifications`,view:G,type:`u32`,access:`read`},{name:`assignment`,view:I,type:`atomic<u32>`,access:`read_write`}],invocationCount:a,condition:d.condition,declarations:`const CANDIDATE_CAPACITY: u32 = ${a}u;
const INSIDE: u32 = ${l.inside}u;
const BOUNDARY: u32 = ${l.boundary}u;
const UNCERTAIN: u32 = ${l.uncertain}u;
const HALF_OPEN_OWNER_BIT: u32 = ${X}u;
const BOUNDARY_CLOSED: bool = ${this.boundaryRule===`closed`};
const BOUNDARY_HALF_OPEN: bool = ${this.boundaryRule===`half-open`};`,body:`let activeCount = min(atomicLoad(&state[stateOffset + 4u]), CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * ${R}u];
  let featureRow = candidatePairs[candidatePairsOffset + index * ${R}u + 1u];
  let word = pairClassifications[pairClassificationsOffset + index];
  let classification = word & ~HALF_OPEN_OWNER_BIT;
  let owned = BOUNDARY_CLOSED || (BOUNDARY_HALF_OPEN && (word & HALF_OPEN_OWNER_BIT) != 0u);
  if (classification == INSIDE || (owned && classification == BOUNDARY)) {
    atomicMin(&assignment[assignmentOffset + pointRow], featureRow);
  } else if (classification == UNCERTAIN) {
    atomicAdd(&state[stateOffset + 1u], 1u);
  }`}))}return A.push(...oe(e,{id:`${d}-assign`,operation:H,assignment:I,pointFeatureIds:_.pointFeatureIds,featureIds:g.sourceIds,featureCounts:_.featureCounts})),x&&A.push(...ae(e,{id:`${d}-collect-matches`,operation:H,points:b,assignment:I,state:F,sourceIds:f.points.sourceIds,matches:x})),A.push(...M(e,{id:`${d}-finalize`,operation:H,state:F,bvhOverflow:N.overflow,candidateCapacity:h,overflow:S.overflow,candidateCount:S.candidateCount,uncertainCount:S.uncertainCount,matches:x})),A}createSlabClassifyNode(e,t){let{id:n,capacity:r}=t,{polygons:a}=this.props,o=h({owner:`polygonIndex`,lowY:`point.y`,highY:`point.y`,getVertex:`readVertex`,pointNames:[`previous`,`current`],body:`let isUsableRing = ((edgeLayout[edgeLayoutOffset + (edgeStart >> 5u)] >> (edgeStart & 31u)) & 1u) != 0u;
        if (!isUsableRing) { continue; }
        let sign = orientSign(previous, current, point);
        if (sign == 2) { return UNCERTAIN; }
        var side = sign;
        if (sign == 0) {
          if (point.x >= min(previous.x, current.x) && point.x <= max(previous.x, current.x) &&
              point.y >= min(previous.y, current.y) && point.y <= max(previous.y, current.y)) {
            onBoundary = true;
          }
          side = select(select(0, -1, current.x < previous.x), 1, current.x > previous.x);
          if (side == 0) {
            side = select(select(0, 1, current.y < previous.y), -1, current.y > previous.y);
          }
        }
        if ((previous.y > point.y) != (current.y > point.y)) {
          if (current.y > previous.y && sign > 0) { polygonInside = !polygonInside; }
          if (current.y < previous.y && sign < 0) { polygonInside = !polygonInside; }
          if ((current.y > previous.y && side > 0) || (current.y < previous.y && side < 0)) {
            polygonOwned = !polygonOwned;
          }
        }`});return i(e,{id:`${n}-classify`,operation:H,variant:`classify-exact-slabs`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read`},{name:`polygonPositions`,view:a.positions,type:`f32`,access:`read`},...g(t.slabs),{name:`edgeLayout`,view:t.edgeLayout,type:`u32`,access:`read`},{name:`pairClassifications`,view:t.pairClassifications,type:`u32`,access:`read_write`}],invocationCount:r,condition:t.condition,declarations:`const CANDIDATE_CAPACITY: u32 = ${r}u;
const OUTSIDE: u32 = ${l.outside}u;
const INSIDE: u32 = ${l.inside}u;
const BOUNDARY: u32 = ${l.boundary}u;
const UNCERTAIN: u32 = ${l.uncertain}u;
const HALF_OPEN_OWNER_BIT: u32 = ${X}u;
const NO_PART: u32 = 0xffffffffu;
const FEATURE_BASE: u32 = ${Math.max(Math.ceil(a.positions.length/32),1)}u;
${u}
${m(t.slabs)}
fn readVertex(vertexIndex: u32) -> vec2f {
  return vec2f(
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u],
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u + 1u]
  );
}
// Same classification as the unslabbed classifyExactly, over the edges of the slab at point.y of
// each part: see the edgeSlabs prop. Parity is per part (each part is a slab owner) and combined
// with OR across parts.
fn classifySlabs(point: vec2f, partStart: u32, partEnd: u32) -> u32 {
  var geometryInside = false;
  var geometryOwned = false;
  var onBoundary = false;
  for (var polygonIndex = partStart; polygonIndex < partEnd; polygonIndex++) {
    var polygonInside = false;
    var polygonOwned = false;
    ${o}
    geometryInside = geometryInside || polygonInside;
    geometryOwned = geometryOwned || polygonOwned;
  }
  if (onBoundary) { return BOUNDARY | select(0u, HALF_OPEN_OWNER_BIT, geometryOwned); }
  return select(OUTSIDE, INSIDE, geometryInside);
}`,body:`let activeCount = min(state[stateOffset + 4u], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let candidate = candidatePairsOffset + index * 4u;
  let featureRow = candidatePairs[candidate + 1u];
  let point = vec2f(bitcast<f32>(candidatePairs[candidate + 2u]), bitcast<f32>(candidatePairs[candidate + 3u]));
  let feature = edgeLayoutOffset + FEATURE_BASE + featureRow * 2u;
  var classification = UNCERTAIN;
  let partStart = edgeLayout[feature];
  // A malformed feature (partStart == NO_PART) is UNCERTAIN, as in the unslabbed walk.
  if (partStart != NO_PART) {
    classification = classifySlabs(point, partStart, edgeLayout[feature + 1u]);
  }
  pairClassifications[pairClassificationsOffset + index] = classification;`})}};function Q(e,t=!0){let{points:n,polygons:r}=e;return{"points.positions":n.positions,"points.sourceIds":e.output.matches&&n.positions.length>0?n.sourceIds:void 0,"polygons.positions":r.positions,"polygons.featureOffsets":r.featureOffsets,"polygons.polygonOffsets":r.polygonOffsets,"polygons.ringOffsets":r.ringOffsets,"polygons.sourceIds":n.positions.length>0?r.sourceIds:void 0,...Object.fromEntries(Object.entries(e.prepared?.isDeclaredIn(r.positions.buffer.graph)?e.prepared.storage:e.prepared?.props.storage??{}).map(([e,t])=>[`prepared.${e}`,t])),...t&&e.prepared?.edgeSlabStorage?{"prepared.slabParams":e.prepared.edgeSlabStorage.slabs.params,"prepared.slabOffsets":e.prepared.edgeSlabStorage.slabs.slabOffsets,"prepared.slabEntries":e.prepared.edgeSlabStorage.slabs.slabEntries,"prepared.edgeLayout":e.prepared.edgeSlabStorage.layout}:{}}}function $(e){let{output:t}=e,{matches:n,status:r}=t;return{"output.pointFeatureIds":t.pointFeatureIds,"output.featureCounts":t.featureCounts,"output.candidateTruncated":t.candidateTruncated,"output.status.overflow":r.overflow,"output.status.candidateOverflow":r.candidateOverflow,"output.status.candidateCount":r.candidateCount,"output.status.uncertainCount":r.uncertainCount,...e.spatialContext?.coordinateSpace===`longitude-latitude`?{"output.status.antimeridianUncertainCount":r.antimeridianUncertainCount,"output.status.poleUncertainCount":r.poleUncertainCount}:{},"output.matches.ids":n?.ids,"output.matches.status.count":n?.status.count,"output.matches.status.overflow":n?.status.overflow,"output.matches.status.requiredCount":n?.status.requiredCount,"output.matches.status.candidateOverflow":n?.status.candidateOverflow}}export{J as n,Z as t};
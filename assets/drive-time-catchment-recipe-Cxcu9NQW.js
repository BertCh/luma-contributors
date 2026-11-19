import{d as e,f as t,i as n,l as r,t as i,u as a}from"./recipe-utils-CICB7qiS.js";import{r as o}from"./kernel-C3tfQOn4.js";import{t as s}from"./gpu-sort-xRm8l5yn.js";import{A as c,T as l}from"./gpu-contributor-utils-BIwLQON5.js";import{t as u}from"./binary-search-wgsl-CMmaMYIt.js";import{a as d}from"./sorted-segment-sums-Dvmm6OdR.js";import{t as ee}from"./gpu-group-statistics-CitFlfwS.js";import{n as te,t as f}from"./gpu-network-snapping-DbjuKXIK.js";import{t as ne}from"./gpu-network-service-areas-DYSymXtS.js";import{t as re}from"./gpu-network-isochrones-jZEBWzNO.js";import{g as ie,v as ae}from"./spatial-context-DwZEd4EZ.js";import{t as oe}from"./gpu-point-in-polygon-join-B0GGIWim.js";import{o as p}from"./wgsl-kernel-nodes-DagJklH2.js";var m=`GPUDriveTimeCatchmentRecipe`,se=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),metrics:Object.freeze([`none`,`native`,`great-circle`]),measure:`distance`}),h=4294967295;function ce(e){let t=[],n=e?.raster,r=n?.output?.status;r?.count&&r.overflow&&t.push({stage:`isochrone-bands`,status:{count:r.count,overflow:r.overflow,requiredCount:r.requiredCount}}),r?.edgeCount&&r.edgeOverflow&&t.push({stage:`isochrone-band-edges`,status:{count:r.edgeCount,overflow:r.edgeOverflow,requiredCount:r.edgeRequiredCount}});let i=n?.rings?.output;i&&t.push({stage:`isochrone-band-rings`,status:{count:i.status.count,overflow:i.status.overflow,requiredCount:i.status.requiredCount}});let a=e?.cellOutline;a?.table&&t.push({stage:`isochrone-cells`,status:{count:a.table.status.count,overflow:a.table.status.overflow,requiredCount:a.table.status.requiredCount}}),a&&t.push({stage:`isochrone-outline`,status:{count:a.output.status.count,overflow:a.output.status.overflow,requiredCount:a.output.status.requiredCount}});let o=a?.rings?.output;return o&&t.push({stage:`isochrone-rings`,status:{count:o.status.count,overflow:o.status.overflow,requiredCount:o.status.requiredCount}}),t}function g(e,t){let{graph:n}=e,{id:r,snappedEdges:i,edgeSources:a,topology:s,nodeCount:c}=t,l=o(n,`${r}-nodes`,`uint32x2`,i.length),u=o(n,`${r}-costs`,`float32x2`,i.length),d=o(n,`${r}-keys`,`uint32`,i.length);return e.addAdapter(()=>n.add(p(n,{id:`${r}-kernel`,operation:`GPURecipeDriveTimeCatchment`,variant:`snapped-roads`,bindings:[{name:`snappedEdges`,view:i,type:`u32`,access:`read`},{name:`edgeSources`,view:a,type:`u32`,access:`read`},{name:`offsets`,view:s.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read`},{name:`weights`,view:s.weights,type:`f32`,access:`read`},{name:`roads`,view:l,type:`u32`,access:`read_write`},{name:`roadCosts`,view:u,type:`f32`,access:`read_write`},{name:`roadKeys`,view:d,type:`u32`,access:`read_write`}],invocationCount:i.length,declarations:`const NODE_COUNT: u32 = ${c}u;
const EDGE_COUNT: u32 = ${s.neighbors.length}u;
const NONE: u32 = 0xffffffffu;
const IMPASSABLE: f32 = 3.0e38;
fn getRoadCost(fromNode: u32, toNode: u32) -> f32 {
  var roadCost = -1.0;
  let end = min(offsets[offsetsOffset + fromNode + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + fromNode]; edge < end; edge++) {
    let weight = weights[weightsOffset + edge];
    if (neighbors[neighborsOffset + edge] == toNode && weight >= 0.0 && weight < IMPASSABLE && (roadCost < 0.0 || weight < roadCost)) {
      roadCost = weight;
    }
  }
  return roadCost;
}
// Smallest edge row from the first node to the second, passable or not, or the edge capacity.
fn getFirstRoadEdge(fromNode: u32, toNode: u32) -> u32 {
  let end = min(offsets[offsetsOffset + fromNode + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + fromNode]; edge < end; edge++) {
    if (neighbors[neighborsOffset + edge] == toNode) {
      return edge;
    }
  }
  return EDGE_COUNT;
}`,body:`let edge = snappedEdges[snappedEdgesOffset + index];
  var sourceNode = NONE;
  var targetNode = NONE;
  if (edge < EDGE_COUNT) {
    sourceNode = edgeSources[edgeSourcesOffset + edge];
    targetNode = neighbors[neighborsOffset + edge];
  }
  var forwardCost = -1.0;
  var reverseCost = -1.0;
  var roadKey = EDGE_COUNT;
  if (sourceNode < NODE_COUNT && targetNode < NODE_COUNT) {
    forwardCost = getRoadCost(sourceNode, targetNode);
    reverseCost = getRoadCost(targetNode, sourceNode);
    roadKey = min(getFirstRoadEdge(sourceNode, targetNode), getFirstRoadEdge(targetNode, sourceNode));
  } else {
    sourceNode = NONE;
    targetNode = NONE;
  }
  roads[roadsOffset + 2u * index] = sourceNode;
  roads[roadsOffset + 2u * index + 1u] = targetNode;
  roadCosts[roadCostsOffset + 2u * index] = forwardCost;
  roadCosts[roadCostsOffset + 2u * index + 1u] = reverseCost;
  roadKeys[roadKeysOffset + index] = roadKey;`}))),{nodes:l,costs:u,keys:d}}function _(e,n){return t(e,n,v)}function v(t,h){let _=h.id??`drive-time-catchment`,v=ae(_,h.spatialContext,se),y=new i(t),{topology:b}=h,x=h.outputs??{},S=h.scratch??{},{nodeCount:C}=c(_,b,{weights:!0,nodePositions:!0}),w=h.facilities.length,T=h.demand.length,E=h.costBreaks.length;n(m,E>=1,`costBreaks must not be empty`),e(t,_,[b.offsets,b.neighbors,b.weights,b.nodePositions,h.facilities,h.demand,h.demandValues,h.maximumSnapDistance,h.costBreaks,h.costLimit,h.isochrones?.costBreaks,h.isochrones?.parameters],[x.assignments,x.nodeCosts,x.status?.serviceAreaConverged,x.demandTimes,x.demandBands,x.bands?.keys,x.bands?.counts,x.bands?.status?.count,x.bands?.status?.overflow,x.bands?.status?.requiredCount,h.demandValues?x.bands?.sumValues:void 0,h.demandValues?x.bands?.means:void 0,h.isochrones?.joinDemand?x.isochroneDemand?.pointFeatureIds:void 0,h.isochrones?.joinDemand?x.isochroneDemand?.status?.overflow:void 0,h.isochrones?.joinDemand?x.isochroneDemand?.status?.uncertainCount:void 0,h.candidateCapacity?x.snapping?.status?.facilitiesOverflow:void 0,h.candidateCapacity?x.snapping?.status?.demandOverflow:void 0,S.startNodes,S.startCosts,S.facilityEdges,S.demandEdges,S.demandSourceCosts,S.demandTargetCosts,h.demandValues?S.bandSumWords:void 0]);let D=b.neighbors.length,O=o(t,`${_}-edge-sources`,`uint32`,Math.max(D,1));for(let e of te(t,{id:`${_}-edge-sources-kernel`,operation:`GPURecipeDriveTimeCatchment`,offsets:b.offsets,edgeSources:O,nodeCount:C}))y.addAdapter(()=>t.add(e));let k=v.coordinateSpace===`longitude-latitude`?{coordinateSpace:`longitude-latitude`,metric:`great-circle`,sphereRadius:ie(v),...v.units?{units:v.units}:{}}:v,A=x.snapping??{},j=h.candidateCapacity===void 0?void 0:{status:{facilitiesOverflow:r(t,`${_}-snap-facilities-overflow`,`uint32`,1,A.status?.facilitiesOverflow),demandOverflow:r(t,`${_}-snap-demand-overflow`,`uint32`,1,A.status?.demandOverflow)}},M=r(t,`${_}-start-nodes`,`uint32`,2*w,S.startNodes),N=r(t,`${_}-start-costs`,`float32`,2*w,S.startCosts),P=r(t,`${_}-facility-edges`,`uint32`,w,S.facilityEdges),F=o(t,`${_}-facility-fractions`,`float32`,w);y.add(new f({id:`${_}-snap-facilities`,spatialContext:k,positions:h.facilities,topology:b,maximumSnapDistance:h.maximumSnapDistance,candidateCapacity:h.candidateCapacity,maximumUnindexedWork:2**53-1,output:{status:{overflow:j?.status.facilitiesOverflow},snappedEdges:P,snapFractions:F}}));let I=g(y,{id:`${_}-facility-roads`,snappedEdges:P,edgeSources:O,topology:b,nodeCount:C});y.addAdapter(()=>t.add(p(t,{id:`${_}-facility-starts-kernel`,operation:`GPURecipeDriveTimeCatchment`,variant:`facility-starts`,bindings:[{name:`roads`,view:I.nodes,type:`u32`,access:`read`},{name:`roadCosts`,view:I.costs,type:`f32`,access:`read`},{name:`fractions`,view:F,type:`f32`,access:`read`},{name:`startNodes`,view:M,type:`u32`,access:`read_write`},{name:`startCosts`,view:N,type:`f32`,access:`read_write`}],invocationCount:w,declarations:`const NONE: u32 = 0xffffffffu;`,body:`let fraction = fractions[fractionsOffset + index];
  let forwardCost = roadCosts[roadCostsOffset + 2u * index];
  let reverseCost = roadCosts[roadCostsOffset + 2u * index + 1u];
  let useSource = reverseCost >= 0.0 || fraction == 0.0;
  let useTarget = forwardCost >= 0.0 || fraction == 1.0;
  startNodes[startNodesOffset + 2u * index] = select(NONE, roads[roadsOffset + 2u * index], useSource);
  startCosts[startCostsOffset + 2u * index] = select(-1.0, fraction * max(reverseCost, 0.0), useSource);
  startNodes[startNodesOffset + 2u * index + 1u] = select(NONE, roads[roadsOffset + 2u * index + 1u], useTarget);
  startCosts[startCostsOffset + 2u * index + 1u] = select(-1.0, (1.0 - fraction) * max(forwardCost, 0.0), useTarget);`})));let L=o(t,`${_}-facility-oriented-fractions`,`uint32`,w),R=o(t,`${_}-sorted-facility-keys`,`uint32`,w),z=o(t,`${_}-sorted-facility-fractions`,`uint32`,w);y.addAdapter(()=>t.add([p(t,{id:`${_}-facility-oriented-fractions-kernel`,operation:`GPURecipeDriveTimeCatchment`,variant:`facility-oriented-fractions`,bindings:[{name:`roads`,view:I.nodes,type:`u32`,access:`read`},{name:`fractions`,view:F,type:`f32`,access:`read`},{name:`oriented`,view:L,type:`u32`,access:`read_write`}],invocationCount:w,body:`let isReversed = roads[roadsOffset + 2u * index] > roads[roadsOffset + 2u * index + 1u];
  let fractionBits = bitcast<u32>(fractions[fractionsOffset + index]) & 0x7fffffffu;
  oriented[orientedOffset + index] = fractionBits | select(0u, 0x80000000u, isReversed);`}),...new s({id:`${_}-facility-road-sort`,keys:I.keys,values:L,outputKeys:R,outputValues:z,keyBits:d(D)}).getCommandNodes(t)]));let B=r(t,`${_}-assignments`,`uint32`,C,x.assignments),V=r(t,`${_}-node-costs`,`float32`,C,x.nodeCosts),H=r(t,`${_}-service-area-converged`,`uint32`,1,x.status?.serviceAreaConverged);y.add(new ne({id:`${_}-service-areas`,topology:b,facilities:M,facilityCosts:N,costLimit:h.costLimit,maximumIterations:h.maximumIterations,output:{assignments:B,costs:V,status:{converged:H}}}));let U=h.isochrones;if(U){n(m,!!(U.raster||U.cellOutline),`isochrones needs raster or cellOutline`),n(m,!U.joinDemand||!!U.cellOutline?.rings?.output.polygons,`isochrones.joinDemand needs cellOutline.rings.output.polygons`);let e=U.raster;if(e){l(m,(e.mode??`min`)===`min`,`isochrones.raster.mode must be min to include direct travel from snapped facilities`),l(m,!e.sourcePaths,`isochrones.raster.sourcePaths is owned by the drive-time facility adapter`);let n=o(t,`${_}-source-path-positions`,`float32x2`,3*w),r=o(t,`${_}-source-path-costs`,`float32`,3*w),i=o(t,`${_}-source-path-assignments`,`uint32`,3*w),a=o(t,`${_}-source-path-offsets`,`uint32`,3*w+1),s=o(t,`${_}-source-path-neighbors`,`uint32`,2*w),c=o(t,`${_}-source-path-weights`,`float32`,2*w);y.addAdapter(()=>t.add(p(t,{id:`${_}-source-path-geometry`,operation:`GPURecipeDriveTimeCatchment`,invocationCount:w,bindings:[{name:`roads`,view:I.nodes,type:`u32`,access:`read`},{name:`fractions`,view:F,type:`f32`,access:`read`},{name:`positions`,view:b.nodePositions,type:`f32`,access:`read`},{name:`startNodes`,view:M,type:`u32`,access:`read`},{name:`pathPositions`,view:n,type:`f32`,access:`read_write`},{name:`pathCosts`,view:r,type:`f32`,access:`read_write`},{name:`pathAssignments`,view:i,type:`u32`,access:`read_write`}],declarations:`const NODE_COUNT: u32 = ${C}u;`,body:`let sourceNode = roads[roadsOffset + 2u * index];
  let targetNode = roads[roadsOffset + 2u * index + 1u];
  let valid = sourceNode < NODE_COUNT && targetNode < NODE_COUNT;
  var startPoint = vec2f(0.0);
  var endPoint = vec2f(0.0);
  if (valid) {
    startPoint = vec2f(positions[positionsOffset + 2u * sourceNode], positions[positionsOffset + 2u * sourceNode + 1u]);
    endPoint = vec2f(positions[positionsOffset + 2u * targetNode], positions[positionsOffset + 2u * targetNode + 1u]);
  }
  let point = mix(startPoint, endPoint, vec2f(fractions[fractionsOffset + index]));
  let offset = pathPositionsOffset + 6u * index;
  pathPositions[offset] = point.x;
  pathPositions[offset + 1u] = point.y;
  pathPositions[offset + 2u] = startPoint.x;
  pathPositions[offset + 3u] = startPoint.y;
  pathPositions[offset + 4u] = endPoint.x;
  pathPositions[offset + 5u] = endPoint.y;
  // A runtime bitcast preserves infinity without an invalid WGSL constant expression.
  var infinityBits = 0x7f800000u;
  let infinity = bitcast<f32>(infinityBits);
  pathCosts[pathCostsOffset + 3u * index] = select(infinity, 0.0, valid);
  pathCosts[pathCostsOffset + 3u * index + 1u] = infinity;
  pathCosts[pathCostsOffset + 3u * index + 2u] = infinity;
  let assignment = select(2u * index + 1u, 2u * index, startNodes[startNodesOffset + 2u * index] != 0xffffffffu);
  pathAssignments[pathAssignmentsOffset + 3u * index] = assignment;
  pathAssignments[pathAssignmentsOffset + 3u * index + 1u] = assignment;
  pathAssignments[pathAssignmentsOffset + 3u * index + 2u] = assignment;`}))),y.addAdapter(()=>t.add(p(t,{id:`${_}-source-path-topology`,operation:`GPURecipeDriveTimeCatchment`,invocationCount:w,bindings:[{name:`roadCosts`,view:I.costs,type:`f32`,access:`read`},{name:`fractions`,view:F,type:`f32`,access:`read`},{name:`offsets`,view:a,type:`u32`,access:`read_write`},{name:`neighbors`,view:s,type:`u32`,access:`read_write`},{name:`weights`,view:c,type:`f32`,access:`read_write`}],declarations:`const FACILITY_COUNT: u32 = ${w}u;`,body:`let fraction = fractions[fractionsOffset + index];
  let forwardCost = roadCosts[roadCostsOffset + 2u * index];
  let reverseCost = roadCosts[roadCostsOffset + 2u * index + 1u];
  offsets[offsetsOffset + 3u * index] = 2u * index;
  offsets[offsetsOffset + 3u * index + 1u] = 2u * index + 2u;
  offsets[offsetsOffset + 3u * index + 2u] = 2u * index + 2u;
  if (index + 1u == FACILITY_COUNT) { offsets[offsetsOffset + 3u * FACILITY_COUNT] = 2u * FACILITY_COUNT; }
  neighbors[neighborsOffset + 2u * index] = 3u * index + 1u;
  neighbors[neighborsOffset + 2u * index + 1u] = 3u * index + 2u;
  weights[weightsOffset + 2u * index] = select(-1.0, fraction * max(reverseCost, 0.0), reverseCost >= 0.0 || fraction == 0.0);
  weights[weightsOffset + 2u * index + 1u] = select(-1.0, (1.0 - fraction) * max(forwardCost, 0.0), forwardCost >= 0.0 || fraction == 1.0);`}))),e={...e,sourcePaths:{topology:{offsets:a,neighbors:s,weights:c,nodePositions:n},costs:r,assignments:i}}}y.add(new re({id:`${_}-isochrones`,topology:b,costs:V,assignments:B,costBreaks:U.costBreaks,parameters:U.parameters,raster:e,cellOutline:U.cellOutline}))}let W,G=U?.cellOutline?.rings?.output.polygons;if(U?.joinDemand&&G){let e=U.joinDemand,n=x.isochroneDemand??{};W={pointFeatureIds:r(t,`${_}-isochrone-demand`,`uint32`,T,n.pointFeatureIds),status:{overflow:r(t,`${_}-isochrone-join-overflow`,`uint32`,1,n.status?.overflow),uncertainCount:r(t,`${_}-isochrone-join-uncertain`,`uint32`,1,n.status?.uncertainCount)}},y.add(new oe({id:`${_}-isochrone-join`,points:{kind:`points`,positions:h.demand},polygons:{kind:`polygons`,positions:G.positions,featureOffsets:G.featureOffsets,polygonOffsets:G.polygonOffsets,ringOffsets:G.ringOffsets},candidateCapacity:e.candidateCapacity,output:{pointFeatureIds:W.pointFeatureIds,status:W.status}}))}let K=r(t,`${_}-demand-edges`,`uint32`,T,S.demandEdges),q=o(t,`${_}-demand-fractions`,`float32`,T);y.add(new f({id:`${_}-snap-demand`,spatialContext:k,positions:h.demand,topology:b,maximumSnapDistance:h.maximumSnapDistance,candidateCapacity:h.candidateCapacity,maximumUnindexedWork:2**53-1,output:{status:{overflow:j?.status.demandOverflow},snappedEdges:K,snapFractions:q,sourceCosts:S.demandSourceCosts,targetCosts:S.demandTargetCosts}}));let J=a(t,`${_}-demand-times`,`float32`,T,x.demandTimes),Y=g(y,{id:`${_}-demand-roads`,snappedEdges:K,edgeSources:O,topology:b,nodeCount:C});y.addAdapter(()=>t.add(p(t,{id:`${_}-demand-times-kernel`,operation:`GPURecipeDriveTimeCatchment`,variant:`demand-time`,bindings:[{name:`roads`,view:Y.nodes,type:`u32`,access:`read`},{name:`roadCosts`,view:Y.costs,type:`f32`,access:`read`},{name:`roadKeys`,view:Y.keys,type:`u32`,access:`read`},{name:`fractions`,view:q,type:`f32`,access:`read`},{name:`nodeCosts`,view:V,type:`f32`,access:`read`},{name:`sortedFacilityKeys`,view:R,type:`u32`,access:`read`},{name:`sortedFacilityFractions`,view:z,type:`u32`,access:`read`},{name:`times`,view:J,type:`f32`,access:`read_write`}],invocationCount:T,declarations:`const NODE_COUNT: u32 = ${C}u;
const FACILITY_COUNT: u32 = ${w}u;
const UNREACHED: f32 = 3.0e38;
${u(`sortedFacilityKeys`)}`,body:`let sourceNode = roads[roadsOffset + 2u * index];
  let targetNode = roads[roadsOffset + 2u * index + 1u];
  var best = UNREACHED;
  if (sourceNode < NODE_COUNT && targetNode < NODE_COUNT) {
    let fraction = fractions[fractionsOffset + index];
    let forwardCost = roadCosts[roadCostsOffset + 2u * index];
    let reverseCost = roadCosts[roadCostsOffset + 2u * index + 1u];
    if (forwardCost >= 0.0 || fraction == 0.0) {
      let viaSource = nodeCosts[nodeCostsOffset + sourceNode] + fraction * max(forwardCost, 0.0);
      if (viaSource < best) {
        best = viaSource;
      }
    }
    if (reverseCost >= 0.0 || fraction == 1.0) {
      let viaTarget = nodeCosts[nodeCostsOffset + targetNode] + (1.0 - fraction) * max(reverseCost, 0.0);
      if (viaTarget < best) {
        best = viaTarget;
      }
    }
    // The facilities on this road are one run of the sorted keys: a bounded neighbourhood of the
    // demand point (the facilities that share its road), not a scan of every facility.
    let roadKey = roadKeys[roadKeysOffset + index];
    let isReversed = sourceNode > targetNode;
    for (
      var position = lowerBoundSortedFacilityKeys(0u, FACILITY_COUNT, roadKey);
      position < FACILITY_COUNT && sortedFacilityKeys[sortedFacilityKeysOffset + position] == roadKey;
      position++
    ) {
      // Fraction of the facility along this demand's edge, then the signed way from it.
      let facilityBits = sortedFacilityFractions[sortedFacilityFractionsOffset + position];
      let facilityFraction = bitcast<f32>(facilityBits & 0x7fffffffu);
      let sameDirection = ((facilityBits & 0x80000000u) != 0u) == isReversed;
      let ahead = fraction - select(1.0 - facilityFraction, facilityFraction, sameDirection);
      if (ahead >= 0.0 && forwardCost >= 0.0 && ahead * forwardCost < best) {
        best = ahead * forwardCost;
      }
      if (ahead <= 0.0 && reverseCost >= 0.0 && -ahead * reverseCost < best) {
        best = -ahead * reverseCost;
      }
    }
  }
  if (best < UNREACHED) {
    times[timesOffset + index] = best;
  } else {
    var infinityBits = 0x7f800000u;
    times[timesOffset + index] = bitcast<f32>(infinityBits);
  }`})));let X=a(t,`${_}-demand-bands`,`uint32`,T,x.demandBands);y.addAdapter(()=>t.add(p(t,{id:`${_}-demand-bands-kernel`,operation:`GPURecipeDriveTimeCatchment`,variant:`band`,bindings:[{name:`times`,view:J,type:`f32`,access:`read`},{name:`bandBreaks`,view:h.costBreaks,type:`f32`,access:`read`},{name:`bands`,view:X,type:`u32`,access:`read_write`}],invocationCount:T,declarations:`const BAND_COUNT: u32 = ${E}u;`,body:`let time = times[timesOffset + index];
  var band = 0xffffffffu;
  if (time < 3.0e38) {
    for (var k = 0u; k < BAND_COUNT; k++) {
      if (time <= bandBreaks[bandBreaksOffset + k]) {
        band = k;
        break;
      }
    }
  }
  bands[bandsOffset + index] = band;`})));let Z=h.bandCapacity??E,Q=x.bands??{},$={keys:r(t,`${_}-band-keys`,`uint32`,Z,Q.keys),counts:r(t,`${_}-band-counts`,`uint32`,Z,Q.counts),status:{count:r(t,`${_}-band-count`,`uint32`,1,Q.status?.count),overflow:r(t,`${_}-band-overflow`,`uint32`,1,Q.status?.overflow),requiredCount:r(t,`${_}-band-required-count`,`uint32`,1,Q.status?.requiredCount)},sumValues:h.demandValues?r(t,`${_}-band-sums`,`float32`,Z,Q.sumValues):void 0,means:h.demandValues?r(t,`${_}-band-means`,`float32`,Z,Q.means):void 0};return y.add(new ee({id:`${_}-band-statistics`,keys:X,columns:h.demandValues?[{values:h.demandValues,statistics:[`sum`,`mean`],output:{sums:r(t,`${_}-band-sum-words`,`uint32x2`,Z,S.bandSumWords),sumValues:$.sumValues,means:$.means}}]:[],output:{keys:$.keys,counts:$.counts,status:$.status}})),{spatialContext:v,contributors:y.contributors,startNodes:M,startCosts:N,assignments:B,nodeCosts:V,demandTimes:J,demandBands:X,bands:$,isochroneRings:U?.cellOutline?.rings?.output,isochroneDemand:W,snapping:j,outputs:{assignments:B,nodeCosts:V,status:{serviceAreaConverged:H},demandTimes:J,demandBands:X,bands:$,...W?{isochroneDemand:W}:{},...j?{snapping:j}:{}},intermediates:{startNodes:M,startCosts:N},status:{serviceAreaConverged:H,stages:[...j?[{stage:`snap-facilities`,status:{overflow:j.status.facilitiesOverflow}},{stage:`snap-demand`,status:{overflow:j.status.demandOverflow}}]:[],{stage:`service-areas`,status:{converged:H}},...ce(U),{stage:`band-statistics`,status:$.status},...W?[{stage:`isochrone-demand`,status:W.status}]:[]]}}}export{_ as n,h as t};
import{a as e,o as t,s as n}from"./network-view-utils-B1U8oNKx.js";import{r}from"./kernel-C3tfQOn4.js";import{a as i,m as a,v as o}from"./spatial-context-DwZEd4EZ.js";import{D as s,E as c,S as l,T as u,f as d,g as f}from"./gpu-contributor-utils-BIwLQON5.js";import{o as p,t as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{o as h}from"./movement-common-nOvANN43.js";import{t as g}from"./define-run-B-brkCe4.js";import{a as _,i as v,r as y,t as b}from"./zone-kernels-D1gWG_xq.js";import{vs as x}from"./entry-preview-D70TdSbK.js";var S=`GPUKernelFlowSmoothing`,C=1024;function w(e,t=new Float32Array(4)){if(t.length<4)throw Error(`${S} parameter target must hold 4 elements`);let n=e.bandwidthRatio??.5;return f(S,`neighborhoodSize`,e.neighborhoodSize),f(S,`bandwidthRatio`,n,Number.MIN_VALUE),t.set([e.neighborhoodSize,n,0,0]),t}var T=class{id;zoneCount;spatialContext;props;constructor(t){let r=t.id??`kernel-flow-smoothing`;this.spatialContext=o(r,t.spatialContext,i),c(t.positions,[`float32x2`],`${r} positions`),this.zoneCount=t.positions.length,d(r,`zoneCount`,this.zoneCount,1),l(r,`zoneCount`,this.zoneCount,C),c(t.locationWeights,[`float32`],`${r} locationWeights`),s(r,`locationWeights`,t.locationWeights,this.zoneCount),c(t.parameters,[`float32`],`${r} parameters`),u(r,t.parameters.length>=4,`parameters must contain 4 float32 rows`);let{flows:a,output:f}=t,p=this.zoneCount*this.zoneCount;a.kind===`matrix`?(c(a.matrix,[`float32`],`${r} flows.matrix`),s(r,`flows.matrix`,a.matrix,p)):(u(r,a.kind===`pairs`,`flows.kind must be 'matrix' or 'pairs'`),c(a.originZoneIds,[`uint32`],`${r} flows.originZoneIds`),c(a.destinationZoneIds,[`uint32`],`${r} flows.destinationZoneIds`),c(a.weights,[`float32`],`${r} flows.weights`),c(a.validPairCount,[`uint32`],`${r} flows.validPairCount`),s(r,`flows.destinationZoneIds`,a.destinationZoneIds,a.originZoneIds.length),s(r,`flows.weights`,a.weights,a.originZoneIds.length),u(r,a.validPairCount.length>=1,`flows.validPairCount must contain one uint32 row`)),c(f.smoothed,[`float32`],`${r} output.smoothed`),s(r,`output.smoothed`,f.smoothed,p),f.radii&&(c(f.radii,[`float32`],`${r} output.radii`),s(r,`output.radii`,f.radii,this.zoneCount)),n(r,t.output,[]),e(r,[f.smoothed,f.radii],[t.positions,t.locationWeights,t.parameters,...a.kind===`matrix`?[a.matrix]:[a.originZoneIds,a.destinationZoneIds,a.weights,a.validPairCount]]),this.id=r,this.props=t}getCommandNodes(e){let{id:n,props:i,zoneCount:o}=this,{flows:s,output:c}=i;t(n,e,[i.positions,i.locationWeights,i.parameters,c.smoothed,c.radii,...s.kind===`matrix`?[s.matrix]:[s.originZoneIds,s.destinationZoneIds,s.weights,s.validPairCount]],{radii:o*4,kernelWeights:o*o*4,rowSmoothed:o*o*4,denseMatrix:s.kind===`pairs`?o*o*4:4});let l=o*o,u=c.radii??r(e,`${n}-radii`,`float32`,o),d=r(e,`${n}-kernel`,`float32`,l),f=r(e,`${n}-rows`,`float32`,l),g=[],_;s.kind===`matrix`?_=s.matrix:(_=r(e,`${n}-dense`,`float32`,l),g.push(m(e,{id:`${n}-dense-reset`,operation:S,view:_,type:`f32`,value:`0.0`}),p(e,{id:`${n}-dense-scatter`,operation:S,variant:`scatter`,invocationCount:s.originZoneIds.length,bindings:[{name:`originZoneIds`,view:s.originZoneIds,type:`u32`,access:`read`},{name:`destinationZoneIds`,view:s.destinationZoneIds,type:`u32`,access:`read`},{name:`weights`,view:s.weights,type:`f32`,access:`read`},{name:`pairCount`,view:s.validPairCount,type:`u32`,access:`read`},{name:`dense`,view:_,type:`f32`,access:`read_write`}],body:`
  let origin = originZoneIds[originZoneIdsOffset + index];
  let destination = destinationZoneIds[destinationZoneIdsOffset + index];
  let weight = weights[weightsOffset + index];
  let isFinite = (bitcast<u32>(weight) & 0x7f800000u) != 0x7f800000u;
  if (index < pairCount[pairCountOffset] && origin < ${o}u && destination < ${o}u) {
    dense[denseOffset + origin * ${o}u + destination] = select(0.0, weight, isFinite);
  }`})));let v=h(a(this.spatialContext));return g.push(p(e,{id:`${n}-radii`,operation:S,variant:`radii`,invocationCount:o,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`locationWeights`,view:i.locationWeights,type:`f32`,access:`read`},{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`radii`,view:u,type:`f32`,access:`read_write`}],declarations:`${v}
const ZONE_COUNT: u32 = ${o}u;`,body:`
  let size = parameters[parametersOffset];
  var best = -1.0;
  var farthest = 0.0;
  for (var candidate = 0u; candidate < ZONE_COUNT; candidate = candidate + 1u) {
    let radius = rowDistance(index, candidate);
    farthest = max(farthest, radius);
    var weight = 0.0;
    for (var zone = 0u; zone < ZONE_COUNT; zone = zone + 1u) {
      if (rowDistance(index, zone) <= radius) {
        weight = weight + locationWeights[locationWeightsOffset + zone];
      }
    }
    if (weight >= size && (best < 0.0 || radius < best)) {
      best = radius;
    }
  }
  radii[radiiOffset + index] = select(farthest, best, best >= 0.0);`}),p(e,{id:`${n}-kernel`,operation:S,variant:`weights`,invocationCount:l,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`radii`,view:u,type:`f32`,access:`read`},{name:`kernelWeights`,view:d,type:`f32`,access:`read_write`}],declarations:`${v}
const ZONE_COUNT: u32 = ${o}u;
const RADIUS_MEMBERSHIP_MARGIN: f32 = 1.000001;`,body:`
  let zone = index / ZONE_COUNT;
  let neighbour = index % ZONE_COUNT;
  let distance = rowDistance(zone, neighbour);
  let radius = radii[radiiOffset + zone];
  let sigma = radius * parameters[parametersOffset + 1u];
  var weight = 0.0;
  // The radius is a distance the radii kernel computed in another pipeline. A compiler may round
  // the same expression differently there (fused multiply-add), so the zone that defines the
  // radius is admitted with a margin of a few rounding steps instead of an exact comparison.
  if (distance <= radius * RADIUS_MEMBERSHIP_MARGIN) {
    weight = select(1.0, exp(-distance * distance / (2.0 * sigma * sigma)), sigma > 0.0);
  }
  kernelWeights[kernelWeightsOffset + index] = weight;`}),p(e,{id:`${n}-rows`,operation:S,variant:`origins`,invocationCount:l,bindings:[{name:`kernelWeights`,view:d,type:`f32`,access:`read`},{name:`matrix`,view:_,type:`f32`,access:`read`},{name:`rowSmoothed`,view:f,type:`f32`,access:`read_write`}],declarations:`const ZONE_COUNT: u32 = ${o}u;`,body:`
  let origin = index / ZONE_COUNT;
  let destination = index % ZONE_COUNT;
  var sum = 0.0;
  var total = 0.0;
  for (var neighbour = 0u; neighbour < ZONE_COUNT; neighbour = neighbour + 1u) {
    let weight = kernelWeights[kernelWeightsOffset + origin * ZONE_COUNT + neighbour];
    total = total + weight;
    if (weight != 0.0) {
      sum = sum + weight * matrix[matrixOffset + neighbour * ZONE_COUNT + destination];
    }
  }
  rowSmoothed[rowSmoothedOffset + index] = sum / total;`}),p(e,{id:`${n}-columns`,operation:S,variant:`destinations`,invocationCount:l,bindings:[{name:`kernelWeights`,view:d,type:`f32`,access:`read`},{name:`rowSmoothed`,view:f,type:`f32`,access:`read`},{name:`smoothed`,view:c.smoothed,type:`f32`,access:`read_write`}],declarations:`const ZONE_COUNT: u32 = ${o}u;`,body:`
  let origin = index / ZONE_COUNT;
  let destination = index % ZONE_COUNT;
  var sum = 0.0;
  var total = 0.0;
  for (var neighbour = 0u; neighbour < ZONE_COUNT; neighbour = neighbour + 1u) {
    let weight = kernelWeights[kernelWeightsOffset + destination * ZONE_COUNT + neighbour];
    total = total + weight;
    if (weight != 0.0) {
      sum = sum + weight * rowSmoothed[rowSmoothedOffset + origin * ZONE_COUNT + neighbour];
    }
  }
  smoothed[smoothedOffset + index] = sum / total;`})),g}};function E({world:e}){let t=y(e);return{flows:t,origin:v(t),restOrigin:v(t)}}var D=g({parameters:x,caption:`Flows out of one Chicago tract after kernel smoothing of the 1,200 largest tract-to-tract flows (LODES, 2021): each is re-estimated from neighboring origins and destinations, and pairs not listed count as zero.`,createInputs:E,build(e,t){let{flows:n}=t,r=e.upload(`positions`,`float32x2`,n.positions),i=e.allocate(`parameters`,`float32`,4),a=e.allocate(`smoothed`,`float32`,n.zoneCount**2),o=e.allocate(`radii`,`float32`,n.zoneCount);e.graph.add(new T({positions:r.view,locationWeights:e.upload(`location-weights`,`float32`,new Float32Array(n.zoneCount).fill(1)).view,flows:{kind:`pairs`,originZoneIds:e.upload(`origin-zones`,`uint32`,n.origins).view,destinationZoneIds:e.upload(`destination-zones`,`uint32`,n.destinations).view,weights:e.upload(`weights`,`float32`,n.jobs).view,validPairCount:e.upload(`pair-count`,`uint32`,Uint32Array.of(n.jobs.length)).view},parameters:i.view,output:{smoothed:a.view,radii:o.view}}));let s=e.allocate(`selection`,`uint32`,1);return{outputs:{row:b(e,{id:`origin`,matrix:a,zoneCount:n.zoneCount,selection:s}),smoothed:a,radii:o,positions:r,lines:e.upload(`flow-lines`,`float32x4`,n.segments)},update:e=>{i.write(w({neighborhoodSize:e.neighborhoodSize,bandwidthRatio:e.bandwidthRatio})),s.write(Uint32Array.of(t.origin))}}},interaction:{kind:`probe`,hint:`hover an origin tract`,rest:({inputs:e,toUnit:t})=>t([e.flows.positions[e.restOrigin*2],e.flows.positions[e.restOrigin*2+1]]),resolve(e,{inputs:t,outputs:n,peek:r}){let{flows:i}=t,a=_(i,e.x,e.y),o=a!==t.origin;t.origin=a;let s=[i.positions[a*2],i.positions[a*2+1]],c=[{shape:`dot`,at:s}],l=i.flowsByOrigin[a][0];for(let e of i.flowsByOrigin[a].slice(0,6)){let t=i.destinations[e];c.push({shape:`arrow`,from:s,to:[i.positions[t*2],i.positions[t*2+1]]})}let u=r(n.radii),d=r(n.smoothed),f=`origin ${a}`;if(u&&l!==void 0){let e=i.destinations[l],t=[i.positions[e*2],i.positions[e*2+1]];c.push({shape:`ring`,at:s,radius:u[a],dashed:!0},{shape:`ring`,at:t,radius:u[e],dashed:!0});let n=0;for(let e=0;e<i.zoneCount;e++)Math.hypot(i.positions[e*2]-s[0],i.positions[e*2+1]-s[1])<=u[a]&&n++;let r=d?d[a*i.zoneCount+e]:NaN;f+=` · top flow raw ${Math.round(i.jobs[l])} → smoothed ${Number.isFinite(r)?r.toFixed(1):`…`} · pools ${n} tracts at the origin`}return{rerun:o,spotlight:null,glyphs:c,readout:f}}},draw:({inputs:e,outputs:t})=>[{layer:`lines`,context:!0,segments:t.lines,radius:.8,color:[.5,.55,.65,.06]},{layer:`dots`,context:!0,positions:t.positions,radius:3.5},{layer:`dots`,positions:t.positions,values:t.row,count:e.flows.zoneCount,logarithmic:!0,range:[0,6],radius:6}]});export{D as default};
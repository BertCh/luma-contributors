import{t as e}from"./scaled-quotient-wgsl-UDyhQag6.js";import{o as t,s as n}from"./network-view-utils-B1U8oNKx.js";import{r}from"./kernel-C3tfQOn4.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,S as o,T as s,f as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{l as u}from"./wgsl-common-DfyXiYB0.js";import{n as d}from"./point-process-utils-BVnRkMJj.js";import{t as f}from"./define-run-B-brkCe4.js";import{l as p,n as m}from"./access-Co69Wfld.js";import{i as h,p as g,t as _,u as v}from"./roads-DzovamHj.js";import{o as y,s as b}from"./pick-Bo3t3hjv.js";import{wa as x}from"./entry-preview-D70TdSbK.js";var S=1024,C=1048576,w=65536,T=16777216,E=65536,D=256,O=256,k=`1.0e-5`,A=[`none`,`ang`],j=[`network-length`,`inverse-intensity`,`inverse-intensity-squared`],M=class{id;props;constructor(e){this.id=e.id??`network-inhomogeneous-k`,this.props=e,n(this.id,e.output,[{outputs:{k:e.output.k,numerator:e.output.numerator,status:e.output.status},inputs:{normalization:e.output.normalization}}]);let t=e.eventNodeIds.length,r=t*t,i=e.maximumPairCount??1048576,{id:u}=this;c(u,`maximumPairCount`,i,1,C),o(u,`event pairs (eventNodeIds rows squared)`,r,i),c(u,`eventNodeIds rows`,t,1,S),c(u,`nodeCount`,e.nodeCount,1,w),o(u,`nodeDistances rows (eventNodeIds rows times nodeCount)`,t*e.nodeCount,T),c(u,`edgeLengths rows`,e.edgeLengths.length,1,E),c(u,`radii rows`,e.radii.length,1,D),s(u,e.correction===void 0||A.includes(e.correction),`correction must be one of ${A.join(`, `)}`),s(u,e.normalization===void 0||j.includes(e.normalization),`normalization must be one of ${j.join(`, `)}`),d(e.eventNodeIds,`uint32`,t),d(e.intensity,`float32`,t),d(e.nodeDistances,`float32`,t*e.nodeCount),d(e.edgeLengths,`float32`,e.edgeLengths.length),a(e.edgeNodes,[`uint32x2`],`network-inhomogeneous-k edgeNodes`),s(u,e.edgeNodes.length===e.edgeLengths.length,`edgeNodes must hold one row per edge length`),d(e.radii,`float32`,e.radii.length),d(e.output.k,`float32`,e.radii.length),d(e.output.numerator,`float32`,e.radii.length),d(e.output.normalization,`float32`,3),d(e.output.status,`uint32`,e.radii.length),l(this.id,Object.values(e.output),N(e))}getCommandNodes(n){let{id:a,props:o}=this,s=o.eventNodeIds.length,c=s*s,l=o.correction??`ang`,d=o.normalization??`network-length`;t(a,n,[...N(o),...Object.values(o.output)]),t(a,n,[],{pairs:c*8,pairStatus:c*4});let f=r(n,`${a}-pairs`,`float32x2`,c),p=r(n,`${a}-pair-status`,`uint32`,c),m=r(n,`${a}-input-status`,`uint32`,1),h=`${u}
${e}
fn validDistance(value: f32) -> bool {
  return value >= 0.0 && (isFiniteFloat(value) || bitcast<u32>(value) == 0x7f800000u);
}
fn nodeDistance(event: u32, node: u32) -> f32 {
  return distances[distancesOffset + event * ${o.nodeCount}u + node];
}
const NO_MERGED_NODE: u32 = 0xffffffffu;
// Two nodes that are both within the tolerance of the shell and joined by an edge that is tight
// within the tolerance (a shortest-path edge, so at most about two tolerances long) hold one
// level-set location between them, not two. The later of the two (larger distance, then larger
// node ID) is merged into the earlier; this returns the merged node of an edge, or NO_MERGED_NODE.
// The earliest node of such a run is never merged, so every run keeps exactly its entries.
fn getShellMergedNode(event: u32, edge: u32, pairDistance: f32, shellTolerance: f32) -> u32 {
  let first = edges[edgesOffset + 2u * edge];
  let second = edges[edgesOffset + 2u * edge + 1u];
  if (first >= ${o.nodeCount}u || second >= ${o.nodeCount}u || first == second) { return NO_MERGED_NODE; }
  let firstDistance = nodeDistance(event, first);
  let secondDistance = nodeDistance(event, second);
  if (!isFiniteFloat(firstDistance) || !isFiniteFloat(secondDistance)) { return NO_MERGED_NODE; }
  if (abs(firstDistance - pairDistance) > shellTolerance || abs(secondDistance - pairDistance) > shellTolerance) { return NO_MERGED_NODE; }
  let length = lengths[lengthsOffset + edge];
  let isTight = abs(firstDistance + length - secondDistance) <= shellTolerance ||
    abs(secondDistance + length - firstDistance) <= shellTolerance;
  if (!isTight) { return NO_MERGED_NODE; }
  let firstIsEarlier = firstDistance < secondDistance || (firstDistance == secondDistance && first < second);
  return select(first, second, firstIsEarlier);
}`;return[i(n,{id:`${a}-normalize`,operation:`GPUNetworkInhomogeneousK`,variant:d,invocationCount:1,bindings:[{name:`eventNodes`,view:o.eventNodeIds,type:`u32`,access:`read`},{name:`intensity`,view:o.intensity,type:`f32`,access:`read`},{name:`edges`,view:o.edgeNodes,type:`u32`,access:`read`},{name:`lengths`,view:o.edgeLengths,type:`f32`,access:`read`},{name:`normalization`,view:o.output.normalization,type:`f32`,access:`read_write`},{name:`status`,view:m,type:`u32`,access:`read_write`}],declarations:`${u}
${e}`,body:`var flags = 0u;
  var length = 0.0;
  var inverseSum = 0.0;
  for (var edge = 0u; edge < ${o.edgeLengths.length}u; edge++) {
    let value = lengths[lengthsOffset + edge];
    let first = edges[edgesOffset + 2u * edge];
    let second = edges[edgesOffset + 2u * edge + 1u];
    if (!isFiniteFloat(value) || value <= 0.0 || first >= ${o.nodeCount}u || second >= ${o.nodeCount}u || first == second) { flags |= 2u; }
    length += value;
  }
  for (var event = 0u; event < ${s}u; event++) {
    let value = intensity[intensityOffset + event];
    if (!isFiniteFloat(value) || value <= 0.0 || eventNodes[eventNodesOffset + event] >= ${o.nodeCount}u) { flags |= 1u; }
    if (isFiniteFloat(value) && value > 0.0) { inverseSum += getScaledQuotient(1.0, value, 1.0, 0); }
  }
  var denominator = 0.0;
  if (isFiniteFloat(length) && length > 0.0 && isFiniteFloat(inverseSum)) {
    denominator = ${d===`network-length`?`length`:d===`inverse-intensity`?`inverseSum`:`inverseSum * getScaledQuotient(inverseSum, length, 1.0, 0)`};
  }
  if (!isFiniteFloat(length) || !isFiniteFloat(inverseSum) || !isFiniteFloat(denominator) || denominator <= 0.0) { flags |= 32u; }
  normalization[normalizationOffset] = length;
  normalization[normalizationOffset + 1u] = inverseSum;
  normalization[normalizationOffset + 2u] = denominator;
  status[statusOffset] = flags;`}),i(n,{id:`${a}-pairs`,operation:`GPUNetworkInhomogeneousK`,variant:l,invocationCount:c,bindings:[{name:`distances`,view:o.nodeDistances,type:`f32`,access:`read`},{name:`eventNodes`,view:o.eventNodeIds,type:`u32`,access:`read`},{name:`intensity`,view:o.intensity,type:`f32`,access:`read`},{name:`edges`,view:o.edgeNodes,type:`u32`,access:`read`},{name:`lengths`,view:o.edgeLengths,type:`f32`,access:`read`},{name:`pairs`,view:f,type:`f32`,access:`read_write`},{name:`status`,view:p,type:`u32`,access:`read_write`}],declarations:h,body:`let source = index / ${s}u;
  let destination = index % ${s}u;
  let sourceNode = eventNodes[eventNodesOffset + source];
  let destinationNode = eventNodes[eventNodesOffset + destination];
  var flags = 0u;
  var pairDistance = bitcast<f32>(0x7f800000u | (index & 0u));
  var weight = 0.0;
  if (sourceNode >= ${o.nodeCount}u || destinationNode >= ${o.nodeCount}u) { flags |= 1u; }
  else {
    pairDistance = nodeDistance(source, destinationNode);
    if (!validDistance(pairDistance) || nodeDistance(source, sourceNode) != 0.0) { flags |= 4u; }
    if (source != destination && isFiniteFloat(pairDistance) && pairDistance >= 0.0) {
      var shellCount = 1u;
      ${l===`ang`?`
      shellCount = 0u;
      // Two level-set locations closer than this along the network are one location. The table
      // entries are rounded sums, so a root that is really an edge end lands a rounding step
      // inside the edge about half the time; an exact comparison would count it twice.
      let shellTolerance = pairDistance * ${k};
      for (var node = 0u; node < ${o.nodeCount}u; node++) {
        let value = nodeDistance(source, node);
        if (!validDistance(value)) { flags |= 4u; }
        if (node == destinationNode || (isFiniteFloat(value) && abs(value - pairDistance) <= shellTolerance)) { shellCount++; }
      }
      for (var edge = 0u; edge < ${o.edgeLengths.length}u; edge++) {
        let first = edges[edgesOffset + 2u * edge];
        let second = edges[edgesOffset + 2u * edge + 1u];
        if (first >= ${o.nodeCount}u || second >= ${o.nodeCount}u) { continue; }
        // Both ends on the shell and the edge tight: the node loop counted two nodes for one
        // location. Take the later one off again, once per node however many edges merge it.
        // Such an edge has no interior root: both candidate roots are within the tolerance of
        // an end.
        let mergedNode = getShellMergedNode(source, edge, pairDistance, shellTolerance);
        if (mergedNode != NO_MERGED_NODE) {
          var isFirstMerge = true;
          for (var earlier = 0u; earlier < edge; earlier++) {
            if (getShellMergedNode(source, earlier, pairDistance, shellTolerance) == mergedNode) {
              isFirstMerge = false;
              break;
            }
          }
          if (isFirstMerge) { shellCount--; }
          continue;
        }
        let length = lengths[lengthsOffset + edge];
        let firstDistance = nodeDistance(source, first);
        let secondDistance = nodeDistance(source, second);
        // Roots of min(firstDistance+s, secondDistance+length-s)=pairDistance, both measured from
        // the first end. A root within the tolerance of an edge end is that node, counted above.
        let fromFirst = pairDistance - firstDistance;
        let fromSecond = length - (pairDistance - secondDistance);
        let interiorEnd = length - shellTolerance;
        let firstInterior = isFiniteFloat(firstDistance) && fromFirst > shellTolerance && fromFirst < interiorEnd;
        let secondInterior = isFiniteFloat(secondDistance) && fromSecond > shellTolerance && fromSecond < interiorEnd;
        if (!isFiniteFloat(firstDistance) || !isFiniteFloat(secondDistance)) {
          // Only one end is reachable in the supplied table: its branch is the whole profile.
          if (firstInterior || secondInterior) { shellCount++; }
          continue;
        }
        // fromSecond - fromFirst is twice the height of the edge profile's peak above pairDistance:
        // positive means two separate roots, about zero one root at the peak, negative none.
        let rootSeparation = fromSecond - fromFirst;
        if (rootSeparation > shellTolerance) {
          shellCount += u32(firstInterior) + u32(secondInterior);
        } else if (rootSeparation >= -shellTolerance && (firstInterior || secondInterior)) {
          shellCount++;
        }
      }
      if (shellCount == 0u) { flags |= 8u; }`:``}
      let sourceIntensity = intensity[intensityOffset + source];
      let destinationIntensity = intensity[intensityOffset + destination];
      if (isFiniteFloat(sourceIntensity) && sourceIntensity > 0.0 &&
          isFiniteFloat(destinationIntensity) && destinationIntensity > 0.0 && shellCount > 0u) {
        weight = getScaledQuotient(1.0, sourceIntensity, destinationIntensity, 0) / f32(shellCount);
      } else { flags |= 32u; }
      if (!isFiniteFloat(weight)) { flags |= 32u; }
    }
  }
  pairs[pairsOffset + 2u * index] = pairDistance;
  pairs[pairsOffset + 2u * index + 1u] = weight;
  status[statusOffset + index] = flags;`}),i(n,{id:`${a}-finish`,operation:`GPUNetworkInhomogeneousK`,variant:`radius-reduction`,workgroupSize:O,invocationCount:o.radii.length*O,guardIndex:!1,bindings:[{name:`pairs`,view:f,type:`f32`,access:`read`},{name:`pairStatus`,view:p,type:`u32`,access:`read`},{name:`inputStatus`,view:m,type:`u32`,access:`read`},{name:`normalization`,view:o.output.normalization,type:`f32`,access:`read`},{name:`radii`,view:o.radii,type:`f32`,access:`read`},{name:`numerator`,view:o.output.numerator,type:`f32`,access:`read_write`},{name:`result`,view:o.output.k,type:`f32`,access:`read_write`},{name:`status`,view:o.output.status,type:`u32`,access:`read_write`}],declarations:`${u}
${e}
var<workgroup> partialSums: array<f32, ${O}>;
var<workgroup> partialFlags: array<u32, ${O}>;`,body:`let radiusIndex = index / ${O}u;
  let radius = radii[radiiOffset + radiusIndex];
  var partialFlag = 0u;
  var partialSum = 0.0;
  for (var pair = localInvocationIndex; pair < ${c}u; pair += ${O}u) {
    partialFlag |= pairStatus[pairStatusOffset + pair];
    if (pairs[pairsOffset + 2u * pair] <= radius) { partialSum += pairs[pairsOffset + 2u * pair + 1u]; }
  }
  partialSums[localInvocationIndex] = partialSum;
  partialFlags[localInvocationIndex] = partialFlag;
  workgroupBarrier();
  for (var stride = ${O/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialFlags[localInvocationIndex] |= partialFlags[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    var flags = inputStatus[inputStatusOffset] | partialFlags[0];
    let sum = partialSums[0];
    if (!isFiniteFloat(radius) || radius < 0.0) { flags |= 16u; }
    if (radiusIndex > 0u && radius <= radii[radiiOffset + radiusIndex - 1u]) { flags |= 16u; }
    let denominator = normalization[normalizationOffset + 2u];
    var value = 0.0;
    if (isFiniteFloat(sum) && isFiniteFloat(denominator) && denominator > 0.0) {
      value = getScaledQuotient(sum, denominator, 1.0, 0);
    }
    if (!isFiniteFloat(value) || !isFiniteFloat(sum)) { flags |= 32u; }
    let invalid = bitcast<f32>(0x7fc00000u | (radiusIndex & 0u));
    result[resultOffset + radiusIndex] = select(invalid, value, flags == 0u);
    numerator[numeratorOffset + radiusIndex] = select(invalid, sum, flags == 0u);
    status[statusOffset + radiusIndex] = flags;
  }`})]}};function N(e){return[e.eventNodeIds,e.intensity,e.nodeDistances,e.edgeNodes,e.edgeLengths,e.radii]}var P=[.56,.14,.97,.88],F=32,I=24,L=f({parameters:x,caption:`Inhomogeneous K(r): pair counts weighted by inverse intensity, per radius.`,createInputs:({random:e,world:t})=>{let n=h(e,t,6),r=new Set;for(;r.size<F;)r.add(n.findNearestNode(e()*t.width,e()));return{network:n,world:t,eventNodes:Uint32Array.from(r),edgeLengths:Float32Array.from({length:n.edgeCount},(e,t)=>Math.hypot(n.nodes[n.edgePairs[t*2]*2]-n.nodes[n.edgePairs[t*2+1]*2],n.nodes[n.edgePairs[t*2]*2+1]-n.nodes[n.edgePairs[t*2+1]*2+1])),inset:_(t)}},build(e,t,n){let{network:r}=t,i=p(e,r),a=e.upload(`event-nodes`,`uint32`,t.eventNodes),o=m(e,i,a,r.nodeCount),s=e.allocate(`intensity`,`float32`,F),c=e.allocate(`radii`,`float32`,I),l=e.allocate(`k`,`float32`,I),u=e.allocate(`numerator`,`float32`,I),d=e.allocate(`normalization`,`float32`,3),f=e.allocate(`status`,`uint32`,I);return e.graph.add(new M({eventNodeIds:a.view,intensity:s.view,nodeDistances:o.view,nodeCount:r.nodeCount,edgeNodes:e.upload(`edge-nodes`,`uint32x2`,r.edgePairs).view,edgeLengths:e.upload(`edge-lengths`,`float32`,t.edgeLengths).view,radii:c.view,correction:n.correction===1?`ang`:`none`,output:{k:l.view,numerator:u.view,normalization:d.view,status:f.view}})),{outputs:{k:l,numerator:u,nodes:e.upload(`display-nodes`,`float32x2`,t.inset.transform(r.nodes)),edges:e.upload(`road-edges`,`uint32x2`,r.edgePairs),eventDots:e.upload(`event-dots`,`float32x2`,t.inset.transform(v(r.nodes,t.eventNodes)))},update:e=>{i.costLimit.write(Float32Array.of(8)),c.write(Float32Array.from({length:I},(t,n)=>(n+1)/I*e.radii)),s.write(Float32Array.from(t.eventNodes,n=>1+e.gradient*r.nodes[n*2]/t.world.width))}}},rebuildOn:[`correction`],interaction:{kind:`level`,hint:`read K at a radius`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=y(P,i,e.x,e.y),s={u:e.unitX<P[0]?e.unitX/P[0]:o.u},c=Math.min(23,Math.floor(s.u*I)),l=a(n.k),u=(c+1)/I*r.radii,d=t.eventNodes[0],f=e=>Math.hypot(t.network.nodes[e*2]-i.width/2,t.network.nodes[e*2+1]-.5);for(let e of t.eventNodes)f(e)<f(d)&&(d=e);return{glyphs:[{shape:`curtain`,x:b(P,i,(c+.5)/I)},{shape:`ring`,at:t.inset.toMap(t.network.nodes[d*2],t.network.nodes[d*2+1]),radius:u*t.inset.scale}],readout:`r ${u.toFixed(2)} · K ${l?l[c].toFixed(2):`-`}`}}},draw:({outputs:e})=>[g(e),{layer:`dots`,positions:e.eventDots,radius:4,color:[1,.78,.25,1]},{layer:`chart`,kind:`bars`,rectangle:P,values:e.numerator,range:[0,20],alpha:.35},{layer:`chart`,kind:`line`,rectangle:P,values:e.k,range:[0,2.5],radius:3}]});export{L as default};
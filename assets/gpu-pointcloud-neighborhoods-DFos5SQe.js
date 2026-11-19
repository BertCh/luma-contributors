import{t as e}from"./construction-snapshot-DdkX9Pc0.js";import{r as t}from"./kernel-C3tfQOn4.js";import{D as n,E as r,T as i,f as a,g as o}from"./gpu-contributor-utils-BIwLQON5.js";import{a as s,i as c,r as l,t as u,u as d}from"./pointcloud-grid-index-Bt2ZJi74.js";import{a as f,i as p,n as m}from"./pointcloud-kernel-nodes-YcR7W84x.js";import{a as h,t as g}from"./graph-chunk-dispatch-zBt8k2-o.js";function _(e,t){o(e,`radius`,t.radius),i(e,t.radius>0&&Math.fround(t.radius**2)>0&&Number.isFinite(Math.fround(t.radius**2)),`radius must be positive with a finite nonzero float32 square`),t.supportBounds&&i(e,t.supportBounds.minimum.every((e,n)=>Number.isFinite(Math.fround(e))&&Math.fround(e)<Math.fround(t.supportBounds.maximum[n]))&&t.supportBounds.maximum.every(e=>Number.isFinite(Math.fround(e))),`supportBounds must hold finite float32 minimum values below the maximum values`)}var v=class{props;id;constructor(t){this.props=t,t=e(t,[`gridParameters`,`parameters`,`id`,`positions`,`queryRows`,`supportBounds`,`radius`,`capacity`,`strategy`,`grid`,`output`]),this.id=t.id??`gpu-pointcloud-neighborhoods`,i(this.id,t.output!==void 0&&t.output!==null,`output must be given`),this.props=t=Object.freeze({...t,output:e(t.output,[`indices`,`squaredDistances`,`counts`,`status`])}),s(this.id,t.gridParameters),i(this.id,!t.gridParameters||!!t.grid,`gridParameters requires grid`),i(this.id,t.strategy===void 0||t.strategy===`serial`||t.strategy===`cooperative`,`strategy must be 'serial' or 'cooperative'`),a(this.id,`capacity`,t.capacity,1,64),_(this.id,t);for(let[e,n,i]of[[`positions`,t.positions,`float32x3`],[`queryRows`,t.queryRows,`uint32`],[`indices`,t.output.indices,`uint32`],[`squaredDistances`,t.output.squaredDistances,`float32`],[`counts`,t.output.counts,`uint32`],[`status`,t.output.status,`uint32`],[`parameters`,t.parameters,`float32`]])n&&r(n,[i],`${this.id} ${e}`);t.parameters&&n(this.id,`parameters`,t.parameters,9);let o={positions:t.positions,queryRows:t.queryRows,parameters:t.parameters,gridParameters:t.gridParameters},l={indices:t.output.indices,squaredDistances:t.output.squaredDistances,counts:t.output.counts,status:t.output.status};p(this.id,o,l,t.parameters||t.gridParameters?[{inputs:t.gridParameters?{}:o,outputs:{indices:t.output.indices,squaredDistances:t.output.squaredDistances,counts:t.output.counts}},...t.output.status?[{inputs:{...o,counts:t.output.counts},outputs:{status:t.output.status}}]:[]]:[{inputs:o,outputs:l}]);let u=t.queryRows?.length??t.positions.length;t.gridParameters&&a(this.id,`runtime-grid packed neighbor words`,u*(2*t.capacity+1),0,g),n(this.id,`indices`,t.output.indices,u*t.capacity),n(this.id,`squaredDistances`,t.output.squaredDistances,t.output.indices.length),n(this.id,`counts`,t.output.counts,u),t.output.status&&n(this.id,`status`,t.output.status,u),t.supportBounds&&i(this.id,!!t.output.status,`supportBounds requires a status view`),t.grid?(c(this.id,`grid`,t.grid),i(this.id,Number.isFinite(Math.fround(t.radius/t.grid.cellSize)),`radius divided by grid.cellSize must be finite in float32`)):i(this.id,u*t.positions.length<=d,`queries x positions is ${u*t.positions.length}, above the ${d} limit of a search without an index; pass grid`)}getCommandNodes(e){let{positions:n,radius:r,capacity:i,grid:a,output:{indices:o,squaredDistances:s,counts:c,status:d}}=this.props,p=this.props.parameters,g=this.props.gridParameters,_=!!(p||g);h(e,this.id,{positions:n,indices:o,squaredDistances:s,counts:c,status:d,parameters:p,gridParameters:g,queryRows:this.props.queryRows}),a&&l(e,this.id,n,a.dimensions,g),g&&f(e,this.id,{"runtime-grid neighborhood scratch":c.length*(2*i+1)*4});let v=p?`readParameters(0u)`:`${Math.fround(r)}`,y=p?`readParameters(1u)`:`${Math.fround(r*r)}`,b=[],x=g?t(e,`${this.id}-packed-neighbors`,`uint32`,c.length*(2*i+1)):void 0,S=this.props.strategy===`cooperative`,C=Math.min(128,2**Math.floor(Math.log2(1024/i))),w={},T=``,E=`all(abs(position) <= vec3f(3.402823e38))`,D=S?`for (var candidateRound = 0u; candidateRound < ${Math.ceil(n.length/C)}u; candidateRound++) {
    let candidate = candidateRound * ${C}u + localInvocationIndex;
    if (candidate >= ${n.length}u) { continue; }`:`for (var candidate = 0u; candidate < ${n.length}u; candidate++) {`,O=`}`;if(a){let t=u(e,`${this.id}-grid`,n,a,!1,`north`,g);b.push(...t.nodes),w={sortedIndices:t.sortedIndices,cellOffsets:t.offsets},T=`let coordinate=${t.coordinate};`,E+=` && (${t.inside})`;let i=p||g?`i32(min(ceil(getScaledQuotient(${v},${g?`readGridParameters(3u)`:Math.fround(a.cellSize)},1.0,0))+1.0,${Math.max(...a.dimensions)}.0))`:Math.min(Math.ceil(r/a.cellSize)+1,Math.max(...a.dimensions));D=`let center=vec3i(floor(coordinate));
let lower=max(center-vec3i(${i}),vec3i(0));
let upper=min(center+vec3i(${i}),vec3i(${a.dimensions.map(e=>e-1).join(`,`)}));
for (var depth=lower.z; depth<=upper.z; depth++) {
 for (var row=lower.y; row<=upper.y; row++) {
  for (var column=lower.x; column<=upper.x; column++) {
   let cell=(u32(depth)*${a.dimensions[1]}u+u32(row))*${a.dimensions[0]}u+u32(column);
   ${S?`let cellBegin = readCellOffsets(cell);
   let cellLength = readCellOffsets(cell + 1u) - cellBegin;
   let candidateRounds = cellLength / ${C}u + u32(cellLength % ${C}u != 0u);
   for (var candidateRound = 0u; candidateRound < candidateRounds; candidateRound++) {
    let relativeSlot = candidateRound * ${C}u + localInvocationIndex;
    if (relativeSlot >= cellLength) { continue; }
    let slotIndex = cellBegin + relativeSlot;`:`for (var slotIndex=readCellOffsets(cell); slotIndex<readCellOffsets(cell+1u); slotIndex++) {`}
    let candidate=readSortedIndices(slotIndex);`,O=`} } } }`}return b.push(...m(e,{id:this.id,length:c.length,...S?{cooperative:{workgroupSize:C,declarations:`var<workgroup> retainedIds: array<u32, ${C*i}>;
var<workgroup> retainedDistances: array<f32, ${C*i}>;
var<workgroup> supportCounts: array<u32, ${C}>;`}}:{},inputs:{positions:n,...p?{parameters:p}:{},...g?{gridParameters:g}:{},...w,...this.props.queryRows?{queryRows:this.props.queryRows}:{}},outputs:x?{packedNeighbors:x}:{indices:o,squaredDistances:s,counts:c,...!_&&d?{status:d}:{}},declarations:x?`
fn writeIndices(row: u32, value: u32) { writePackedNeighbors((row / ${i}u) * ${2*i+1}u + row % ${i}u, value); }
fn writeSquaredDistances(row: u32, value: f32) { writePackedNeighbors((row / ${i}u) * ${2*i+1}u + ${i}u + row % ${i}u, bitcast<u32>(value)); }
fn writeCounts(row: u32, value: u32) { writePackedNeighbors(row * ${2*i+1}u + ${2*i}u, value); }`:void 0,body:`
var nearestIds: array<u32, ${i}>;
var nearestDistances: array<f32, ${i}>;
for (var slot = 0u; slot < ${i}u; slot++) {
  nearestIds[slot] = 0xffffffffu; nearestDistances[slot] = 3.402823e38;
}
let queryRow = ${this.props.queryRows?`readQueryRows(index)`:`index`};
let position = readPositions(min(queryRow,${Math.max(n.length-1,0)}u));
var count = 0u;
${T}
let validQuery = queryRow < ${n.length}u && (${E});
if (validQuery) {
  ${D}
    if (candidate == queryRow) { continue; }
    let difference = readPositions(candidate) - position;
    let distanceSquared = dot(difference, difference);
    if (!(distanceSquared <= ${y})) { continue; }
    count++;
    ${S?`// Count every radius match before rejecting a noncompetitive candidate.
    if (distanceSquared > nearestDistances[${i-1}u] ||
        (distanceSquared == nearestDistances[${i-1}u] && candidate >= nearestIds[${i-1}u])) { continue; }`:``}
    var insertion = ${i}u;
    for (var slot = 0u; slot < ${i}u; slot++) {
      if (distanceSquared < nearestDistances[slot] || (distanceSquared == nearestDistances[slot] && candidate < nearestIds[slot])) {
        insertion = slot; break;
      }
    }
    if (insertion < ${i}u) {
      for (var slot = ${i-1}u; slot > insertion; slot--) {
        nearestIds[slot] = nearestIds[slot - 1u]; nearestDistances[slot] = nearestDistances[slot - 1u];
      }
      nearestIds[insertion] = candidate; nearestDistances[insertion] = distanceSquared;
    }
  ${O}
}
${S?`
for (var slot = 0u; slot < ${i}u; slot++) {
  retainedIds[localInvocationIndex * ${i}u + slot] = nearestIds[slot];
  retainedDistances[localInvocationIndex * ${i}u + slot] = nearestDistances[slot];
}
supportCounts[localInvocationIndex] = count;
workgroupBarrier();
for (var mergeStride = 1u; mergeStride < ${C}u; mergeStride *= 2u) {
  if (localInvocationIndex % (mergeStride * 2u) == 0u) {
    let firstStart = localInvocationIndex * ${i}u;
    let secondStart = (localInvocationIndex + mergeStride) * ${i}u;
    var firstSlot = 0u;
    var secondSlot = 0u;
    // Read complete inputs into lane-private output before writing shared storage in place.
    for (var slot = 0u; slot < ${i}u; slot++) {
      let firstDistance = retainedDistances[firstStart + min(firstSlot, ${i-1}u)];
      let firstId = retainedIds[firstStart + min(firstSlot, ${i-1}u)];
      let secondDistance = retainedDistances[secondStart + min(secondSlot, ${i-1}u)];
      let secondId = retainedIds[secondStart + min(secondSlot, ${i-1}u)];
      let takeFirst = secondSlot == ${i}u ||
        (firstSlot < ${i}u && (firstDistance < secondDistance ||
          (firstDistance == secondDistance && firstId <= secondId)));
      nearestIds[slot] = select(secondId, firstId, takeFirst);
      nearestDistances[slot] = select(secondDistance, firstDistance, takeFirst);
      firstSlot += u32(takeFirst);
      secondSlot += u32(!takeFirst);
    }
    for (var slot = 0u; slot < ${i}u; slot++) {
      retainedIds[firstStart + slot] = nearestIds[slot];
      retainedDistances[firstStart + slot] = nearestDistances[slot];
    }
    supportCounts[localInvocationIndex] += supportCounts[localInvocationIndex + mergeStride];
  }
  workgroupBarrier();
}
if (localInvocationIndex != 0u) { return; }
count = supportCounts[0];
`:``}
writeCounts(index, count);
${!_&&d?`writeStatus(index,select(1u,0u,validQuery)|select(0u,2u,count>${i}u)${this.props.supportBounds?`|select(4u,0u,all(position-vec3f(${Math.fround(r)})>=vec3f(${this.props.supportBounds.minimum.map(Math.fround).join(`,`)}))&&all(position+vec3f(${Math.fround(r)})<=vec3f(${this.props.supportBounds.maximum.map(Math.fround).join(`,`)})))`:``});`:``}
for (var slot = 0u; slot < ${i}u; slot++) {
  writeIndices(index * ${i}u + slot, nearestIds[slot]);
  writeSquaredDistances(index * ${i}u + slot, nearestDistances[slot]);
}`})),x&&b.push(...m(e,{id:`${this.id}-unpack`,length:c.length,inputs:{packedNeighbors:x},outputs:{indices:o,squaredDistances:s,counts:c},body:`let base = index * ${2*i+1}u;
writeCounts(index, readPackedNeighbors(base + ${2*i}u));
for (var slot = 0u; slot < ${i}u; slot++) {
  writeIndices(index * ${i}u + slot, readPackedNeighbors(base + slot));
  writeSquaredDistances(index * ${i}u + slot, bitcast<f32>(readPackedNeighbors(base + ${i}u + slot)));
}`})),_&&d&&b.push(...m(e,{id:`${this.id}-status`,length:c.length,inputs:{positions:n,counts:c,...p?{parameters:p}:{},...g?{gridParameters:g}:{},...this.props.queryRows?{queryRows:this.props.queryRows}:{}},outputs:{status:d},body:`let queryRow = ${this.props.queryRows?`readQueryRows(index)`:`index`};
let position = readPositions(min(queryRow,${Math.max(n.length-1,0)}u));
${T}
let validQuery = queryRow < ${n.length}u && (${E});
${p?`let lower = vec3f(readParameters(2u), readParameters(3u), readParameters(4u));
let upper = vec3f(readParameters(5u), readParameters(6u), readParameters(7u));
let supported = readParameters(8u) == 0.0 || (all(position - vec3f(${v}) >= lower) && all(position + vec3f(${v}) <= upper));`:`let supported = ${this.props.supportBounds?`all(position - vec3f(${v}) >= vec3f(${this.props.supportBounds.minimum.map(Math.fround).join(`,`)})) && all(position + vec3f(${v}) <= vec3f(${this.props.supportBounds.maximum.map(Math.fround).join(`,`)}))`:`true`};`}
writeStatus(index, select(1u, 0u, validQuery) | select(0u, 2u, readCounts(index) > ${i}u) | select(4u, 0u, supported));`})),b}};export{v as t};
import{t as e}from"./common-CshkX-d_.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";var i=4294967295;function a(i,a){let{id:o,operation:s,pointCount:c,triangles:l}=a,u=l.length,d=3*u;if(d===0)return[];if(a.adjacency.length!==d)throw Error(`${o} adjacency length must equal three times the triangle capacity`);let f=t(i,`${o}-candidate-ids`,`uint32`,d),p=t(i,`${o}-minimum-keys`,`uint32`,d),m=t(i,`${o}-maximum-keys`,`uint32`,d),h=e(32-Math.clz32(c),1,32),g=n(i,{id:`${o}-sort`,operation:s,indices:f,keys:[{view:p,keyBits:h},{view:m,keyBits:h}]}),_=a.validity?[{name:`validity`,view:a.validity,type:`u32`,access:`read`}]:[];return[r(i,{id:`${o}-keys`,operation:s,variant:`triangle-adjacency-keys`,bindings:[{name:`triangles`,view:l,type:`u32`,access:`read`},{name:`triangleCount`,view:a.triangleCount,type:`u32`,access:`read`},..._,{name:`candidateIds`,view:f,type:`u32`,access:`read_write`},{name:`minimumKeys`,view:p,type:`u32`,access:`read_write`},{name:`maximumKeys`,view:m,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`const POINT_COUNT: u32 = ${c}u;
const TRIANGLE_CAPACITY: u32 = ${u}u;`,body:`let triangle = index / 3u;
let edge = index % 3u;
candidateIds[candidateIdsOffset + index] = index;
var minimum = POINT_COUNT; var maximum = POINT_COUNT;
var live = triangle < min(triangleCount[triangleCountOffset], TRIANGLE_CAPACITY);
${a.validity?`live = live && validity[validityOffset + triangle] != 0u;`:``}
if (live) {
  let a = triangles[trianglesOffset + triangle * 3u + edge];
  let b = triangles[trianglesOffset + triangle * 3u + (edge + 1u) % 3u];
  if (a < POINT_COUNT && b < POINT_COUNT && a != b) { minimum = min(a, b); maximum = max(a, b); }
}
minimumKeys[minimumKeysOffset + index] = minimum;
maximumKeys[maximumKeysOffset + index] = maximum;`}),...g.nodes,r(i,{id:`${o}-resolve`,operation:s,variant:`triangle-adjacency-resolve`,bindings:[{name:`sortedCandidates`,view:g.sortedIndices,type:`u32`,access:`read`},{name:`sortedMinimumKeys`,view:g.sortedPrimaryKeys,type:`u32`,access:`read`},{name:`maximumKeys`,view:m,type:`u32`,access:`read`},{name:`adjacency`,view:a.adjacency,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`const POINT_COUNT: u32 = ${c}u;
const HALF_EDGE_COUNT: u32 = ${d}u;
fn sortedMaximumKey(row: u32) -> u32 { return maximumKeys[maximumKeysOffset + sortedCandidates[sortedCandidatesOffset + row]]; }
fn sortedKeyEquals(row: u32, minimum: u32, maximum: u32) -> bool {
  return sortedMinimumKeys[sortedMinimumKeysOffset + row] == minimum && sortedMaximumKey(row) == maximum;
}`,body:`let candidate = sortedCandidates[sortedCandidatesOffset + index];
let triangle = candidate / 3u;
let minimum = sortedMinimumKeys[sortedMinimumKeysOffset + index];
var neighbor = 0xffffffffu;
if (minimum < POINT_COUNT) {
  let maximum = sortedMaximumKey(index);
  var low = 0u; var high = index;
  while (low < high) {
    let middle = (low + high) >> 1u;
    let middleMinimum = sortedMinimumKeys[sortedMinimumKeysOffset + middle];
    if (middleMinimum < minimum || (middleMinimum == minimum && sortedMaximumKey(middle) < maximum)) { low = middle + 1u; } else { high = middle; }
  }
  var row = low;
  while (row < HALF_EDGE_COUNT && sortedKeyEquals(row, minimum, maximum)) {
    let otherTriangle = sortedCandidates[sortedCandidatesOffset + row] / 3u;
    if (otherTriangle != triangle) { neighbor = otherTriangle; break; }
    row++;
  }
}
adjacency[adjacencyOffset + candidate] = neighbor;`})]}export{a as n,i as t};
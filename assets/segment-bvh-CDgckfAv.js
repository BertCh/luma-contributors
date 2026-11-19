import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-bvh-CBtZpuN_.js";import{t as n}from"./gpu-sort-xRm8l5yn.js";import{m as r}from"./wgsl-common-DfyXiYB0.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as a}from"./bounds-reduction-6MyKJx-7.js";function o(o,s){let{id:c,operation:l,minima:u,maxima:d,leafCapacity:f,spatialSort:p}=s,m=u.length,h=[],g=2*f-1,_=u,v=d,y;if(p&&m>=2){let t=e(o,`${c}-sort-scene-bounds`,`float32`,4),s=e(o,`${c}-sort-keys`,`uint32`,m),f=e(o,`${c}-sort-rows`,`uint32`,m),p=e(o,`${c}-sort-sorted-keys`,`uint32`,m),g=e(o,`${c}-sort-sorted-rows`,`uint32`,m);_=e(o,`${c}-sort-minima`,`float32x2`,m),v=e(o,`${c}-sort-maxima`,`float32x2`,m),y=g;let b=[{name:`segmentMinima`,view:u,type:`f32`,access:`read`},{name:`segmentMaxima`,view:d,type:`f32`,access:`read`}],x=`const SEGMENT_COUNT: u32 = ${m}u;
fn readCenter(row: u32) -> vec3f {
  // (center.x, center.y, valid). Skipped segments have inverted bounds.
  let minimum = vec2f(segmentMinima[segmentMinimaOffset + row * 2u], segmentMinima[segmentMinimaOffset + row * 2u + 1u]);
  let maximum = vec2f(segmentMaxima[segmentMaximaOffset + row * 2u], segmentMaxima[segmentMaximaOffset + row * 2u + 1u]);
  let valid = minimum.x <= maximum.x && minimum.y <= maximum.y;
  let center = minimum * 0.5 + maximum * 0.5;
  return vec3f(center, select(0.0, 1.0, valid));
}`;h.push(...a(o,{id:`${c}-sort-scene`,operation:l,variant:`sort-scene-bounds`,bindings:b,declarations:`${x}
fn readItem(row: u32) -> vec4f {
  let center = readCenter(row);
  return vec4f(center.xy, center.z, 0.0);
}`,itemCount:m,output:t})),h.push(i(o,{id:`${c}-sort-keys`,operation:l,variant:`sort-keys`,bindings:[...b,{name:`sceneBounds`,view:t,type:`f32`,access:`read`},{name:`sortKeys`,view:s,type:`u32`,access:`read_write`},{name:`sortRows`,view:f,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`${x}
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
${r}
fn quantizeAxis(value: f32, minimum: f32, maximum: f32) -> u32 {
  let extent = maximum - minimum;
  if (!(extent > 0.0) || extent > FLOAT32_MAXIMUM) { return 0u; }
  let normalized = clamp((value - minimum) / extent, 0.0, 1.0);
  return min(u32(normalized * 65535.0 + 0.5), 65535u);
}`,body:`sortRows[sortRowsOffset + index] = index;
  let center = readCenter(index);
  var key = 0xffffffffu;
  if (center.z > 0.5) {
    let quantizedX = quantizeAxis(center.x, sceneBounds[sceneBoundsOffset], sceneBounds[sceneBoundsOffset + 2u]);
    let quantizedY = quantizeAxis(center.y, sceneBounds[sceneBoundsOffset + 1u], sceneBounds[sceneBoundsOffset + 3u]);
    key = spreadBits(quantizedX) | (spreadBits(quantizedY) << 1u);
  }
  sortKeys[sortKeysOffset + index] = key;`})),h.push(...new n({id:`${c}-sort-order`,keys:s,values:f,outputKeys:p,outputValues:g,algorithm:`radix`}).getCommandNodes(o)),h.push(i(o,{id:`${c}-sort-gather`,operation:l,variant:`sort-gather`,bindings:[...b,{name:`sortedRows`,view:g,type:`u32`,access:`read`},{name:`sortedMinima`,view:_,type:`f32`,access:`read_write`},{name:`sortedMaxima`,view:v,type:`f32`,access:`read_write`}],invocationCount:m,declarations:`const SEGMENT_COUNT: u32 = ${m}u;`,body:`let row = min(sortedRows[sortedRowsOffset + index], SEGMENT_COUNT - 1u);
  sortedMinima[sortedMinimaOffset + index * 2u] = segmentMinima[segmentMinimaOffset + row * 2u];
  sortedMinima[sortedMinimaOffset + index * 2u + 1u] = segmentMinima[segmentMinimaOffset + row * 2u + 1u];
  sortedMaxima[sortedMaximaOffset + index * 2u] = segmentMaxima[segmentMaximaOffset + row * 2u];
  sortedMaxima[sortedMaximaOffset + index * 2u + 1u] = segmentMaxima[segmentMaximaOffset + row * 2u + 1u];`}))}let b=new t({id:`${c}-bvh`,minima:_,maxima:v,sourceIds:y,leafCapacity:f,nodeMinima:e(o,`${c}-bvh-node-minima`,`float32x2`,g),nodeMaxima:e(o,`${c}-bvh-node-maxima`,`float32x2`,g),nodeChildren:e(o,`${c}-bvh-node-children`,`uint32x2`,g),leafIds:e(o,`${c}-bvh-leaf-ids`,`uint32`,f),count:e(o,`${c}-bvh-count`,`uint32`,1),overflow:e(o,`${c}-bvh-overflow`,`uint32`,1)});return h.push(...b.getCommandNodes(o)),{bvh:b,nodes:h}}export{o as t};
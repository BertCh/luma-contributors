import{v as e}from"./spatial-context-DwZEd4EZ.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-reduction-tNgOVQ_C.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{t as i}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as s}from"./sorted-segment-sums-Dvmm6OdR.js";import{n as c}from"./sorted-segment-offsets-ChmOtA9F.js";import{E as l,T as u,f as d,v as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{a as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as h}from"./wgsl-common-DfyXiYB0.js";var g=`GPUGroupConvexHull`,_=2**24-1,v=29,y=30,b=8192,x=256,S=8,C=65536,w=256,T=16,E=256,D=2**v,O=2;function k(e,t,n,r,i=n>=b?O:0,a=r<=C){for(let[o,s]of[[`hull row scratch`,n],[`hull offsets`,r+1],[`hull chain scratch`,n+r],[`hull prefilter scratch`,i>0?2*n+2:0],[`hull extreme scratch`,a?16*r:0],[`hull sort scratch`,n<=256?2**Math.ceil(Math.log2(Math.max(n,1))):n]])s!==0&&f(e,t,o,s*4,`reduce positions rows or groupCount`)}var A=class{id;props;spatialContext;groupCount;prefilterLevels;extremePointFilter;constructor(t){this.id=t.id??`group-convex-hull`,this.props=t,this.spatialContext=e(this.id,t.spatialContext);let{id:n}=this,{output:r}=t;l(t.positions,[`float32x2`],`${n} positions`);let i=t.positions.length;if(i<1)throw Error(`${n} needs at least one row`);if(i>_)throw Error(`${n} supports fewer than 2^24 rows`);if(l(t.labels,[`uint32`],`${n} labels`),t.labels.length!==i)throw Error(`${n} labels length must equal positions length`);if(this.groupCount=t.groupCount,d(n,`groupCount`,this.groupCount,1,_),t.noiseLabel!==void 0&&d(n,`noiseLabel`,t.noiseLabel,0),d(n,`maximumVerticesPerGroup`,t.maximumVerticesPerGroup,3),d(n,`vertexCapacity`,t.vertexCapacity),this.prefilterLevels=t.prefilterLevels??(i>=b?O:0),d(n,`prefilterLevels`,this.prefilterLevels,0,4),this.extremePointFilter=t.extremePointFilter??this.groupCount<=C,this.extremePointFilter&&this.groupCount>C)throw Error(`${n} extremePointFilter supports at most ${C} groups`);let a=this.groupCount,o=[[`vertexIndices`,r.vertexIndices,[`uint32`],t.vertexCapacity],[`vertexPositions`,r.vertexPositions,[`float32x2`],t.vertexCapacity],[`offsets`,r.offsets,[`uint32`],a+1],[`counts`,r.counts,[`uint32`],a],[`sizes`,r.sizes,[`uint32`],a],[`status.overflow`,r.status.overflow,[`uint32`],1],[`status.groupOverflow`,r.status.groupOverflow,[`uint32`],1],[`status.vertexOverflow`,r.status.vertexOverflow,[`uint32`],1]];for(let[e,t,r,i]of o)t&&(l(t,r,`${n} output.${e}`),u(n,t.length>=i,`output.${e} must hold at least ${i} rows`));for(let[e,t]of Object.entries(r.status))if(t)for(let[r,i]of o){if(!i||r===`status.${e}`)continue;let a=t.buffer===i.buffer&&t.byteOffset<i.byteOffset+i.length*i.rowByteLength&&i.byteOffset<t.byteOffset+t.length*t.rowByteLength;u(n,!a,`output.status.${e} must not overlap output.${r}`)}p(n,o.map(([,e])=>e),[t.positions,t.labels])}getCommandNodes(e){let{props:l,id:u,groupCount:d}=this,{output:f,positions:p,labels:_,maximumVerticesPerGroup:v,vertexCapacity:b}=l,{status:S,...C}=f;m(e,u,{positions:p,labels:_,...C,...Object.fromEntries(Object.entries(S).map(([e,t])=>[`status.${e}`,t]))}),k(e.device,u,p.length,d,this.prefilterLevels,this.extremePointFilter);let w=p.length,T=(n,r)=>t(e,`${u}-${n}`,`float32`,r),E=(n,r)=>t(e,`${u}-${n}`,`uint32`,r),D=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),O=(e,t,n=`f32`)=>({name:e,view:t,type:n,access:`read_write`}),A=(t,n,r,i,o=``)=>a(e,{id:`${u}-${t}`,operation:g,variant:t,bindings:r,invocationCount:n,declarations:`const GROUP_COUNT: u32 = ${d}u;\n${o}`,body:i}),j=[],N=E(`group-keys`,w),P=E(`valid-flags`,w),F=T(`xs`,w),I=T(`ys`,w),L=E(`row-indices`,w),ee=l.noiseLabel!==void 0&&l.noiseLabel<d?`valid = valid && label != ${l.noiseLabel}u;`:``;j.push(A(`keys`,w,[D(`positions`,p,`f32`),D(`labels`,_,`u32`),O(`groupKeys`,N,`u32`),O(`validFlags`,P,`u32`),O(`xs`,F),O(`ys`,I),O(`rowIndices`,L,`u32`)],`let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let label = labels[labelsOffset + index];
  var valid = (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u && (bitcast<u32>(y) & 0x7f800000u) != 0x7f800000u;
  valid = valid && label < GROUP_COUNT;
  ${ee}
  groupKeys[groupKeysOffset + index] = select(GROUP_COUNT, label, valid);
  validFlags[validFlagsOffset + index] = select(0u, 1u, valid);
  xs[xsOffset + index] = select(0.0, x, valid);
  ys[ysOffset + index] = select(0.0, y, valid);
  rowIndices[rowIndicesOffset + index] = index;`));let te=T(`extent-x`,2),R=T(`extent-y`,2),z=T(`lattice`,3);j.push(...new n({id:`${u}-extent-x`,input:F,mask:P,output:te,operation:`extent`}).getCommandNodes(e),...new n({id:`${u}-extent-y`,input:I,mask:P,output:R,operation:`extent`}).getCommandNodes(e),A(`lattice`,1,[D(`extentX`,te,`f32`),D(`extentY`,R,`f32`),O(`lattice`,z)],`let width = extentX[extentXOffset + 1u] - extentX[extentXOffset];
  let height = extentY[extentYOffset + 1u] - extentY[extentYOffset];
  let extent = max(width, height);
  var exponent = i32((bitcast<u32>(extent) >> 23u) & 255u) - 127;
  exponent = max(exponent, -99);
  let scale = bitcast<f32>(u32(127 + 28 - exponent) << 23u);
  lattice[latticeOffset] = extentX[extentXOffset];
  lattice[latticeOffset + 1u] = extentY[extentYOffset];
  lattice[latticeOffset + 2u] = select(scale, 1.0, !(extent > 0.0));`));let B=E(`lattice-x`,w),V=E(`lattice-y`,w);j.push(A(`quantize`,w,[D(`xs`,F,`f32`),D(`ys`,I,`f32`),D(`validFlags`,P,`u32`),D(`lattice`,z,`f32`),O(`latticeX`,B,`u32`),O(`latticeY`,V,`u32`)],`let isValid = validFlags[validFlagsOffset + index] != 0u;
  let scale = lattice[latticeOffset + 2u];
  // Nearest lattice point from the exact fraction: d * scale + 0.5 is a tie in f32 for odd
  // values from 2^23, which would move a point that is already on the lattice by one step.
  let scaled = vec2f(
    xs[xsOffset + index] - lattice[latticeOffset],
    ys[ysOffset + index] - lattice[latticeOffset + 1u]
  ) * scale;
  let lower = floor(scaled);
  let nearest = lower + step(vec2f(0.5), scaled - lower);
  latticeX[latticeXOffset + index] = select(0u, u32(nearest.x), isValid);
  latticeY[latticeYOffset + index] = select(0u, u32(nearest.y), isValid);`));let H=N;this.extremePointFilter&&(H=E(`filtered-group-keys`,w),j.push(...this.getExtremePointFilterNodes(e,{rows:w,groupKeys:N,latticeX:B,latticeY:V,filteredGroupKeys:H})));let U=i(e,{id:`${u}-sort`,operation:g,indices:L,indicesAreIdentity:!0,keys:[{view:H,keyBits:s(d)},{view:B,keyBits:y},{view:V,keyBits:y}]});j.push(...U.nodes);let ne=U.sortedPrimaryKeys,W=U.sortedIndices,G=E(`segment-offsets`,d+1);j.push(...c(e,{id:u,operation:g,groupCount:d,sortedGroupKeys:ne,segmentOffsets:G}));let K=E(`sorted-lattice-x`,w),q=E(`sorted-lattice-y`,w);j.push(A(`gather-lattice`,w,[D(`permutation`,W,`u32`),D(`latticeX`,B,`u32`),D(`latticeY`,V,`u32`),O(`sortedLatticeX`,K,`u32`),O(`sortedLatticeY`,q,`u32`)],`let row = permutation[permutationOffset + index];
  sortedLatticeX[sortedLatticeXOffset + index] = latticeX[latticeXOffset + row];
  sortedLatticeY[sortedLatticeYOffset + index] = latticeY[latticeYOffset + row];`));let J=G,Y=K,X=q,Z=W;for(let t=0;t<this.prefilterLevels;t++){let n=`prefilter${t}`,i=Math.ceil(w/x),a=E(`${n}-survivor-flags`,w),o=E(`${n}-survivor-offsets`,w),s=E(`${n}-scratch`,2*w+2),c=E(`${n}-compact-offsets`,d+1),l=t===0?[D(`groupKeys`,ne,`u32`)]:[D(`groupOffsets`,J,`u32`)],f=t===0?`${w}u`:`groupOffsets[groupOffsetsOffset + GROUP_COUNT]`,p=t===0?`let key = groupKeys[groupKeysOffset + runBegin];
    // The excluded-row run is last in sorted order: skip it whole instead of walking it.
    var runEnd = chunkEnd;
    if (key < GROUP_COUNT) {
      runEnd = runBegin + 1u;
      while (runEnd < chunkEnd && groupKeys[groupKeysOffset + runEnd] == key) {
        runEnd++;
      }
    }`:`let key = findGroup(runBegin);
    let runEnd = min(chunkEnd, groupOffsets[groupOffsetsOffset + key + 1u]);`;j.push(A(n,i,[...l,D(`latticeX`,Y,`u32`),D(`latticeY`,X,`u32`),O(`flags`,a,`u32`),O(`scratch`,s,`u32`)],`let chunkStart = index * ${x}u;
  let slotEnd = min(chunkStart + ${x}u, ${w}u);
  for (var row = chunkStart; row < slotEnd; row++) {
    flags[flagsOffset + row] = 0u;
  }
  let chunkEnd = min(slotEnd, ${f});
  var runBegin = chunkStart;
  while (runBegin < chunkEnd) {
    ${p}
    if (key < GROUP_COUNT) {
      let begin = runBegin;
      let base = 2u * begin;
      var lastRep = runEnd - 1u;
      while (!isRepresentative(lastRep, begin)) {
        lastRep = lastRep - 1u;
      }
      if (lastRep == begin) {
        flags[flagsOffset + begin] = 1u;
      } else {
        var count = 0u;
        for (var row = begin; row <= lastRep; row++) {
          if (!isRepresentative(row, begin)) {
            continue;
          }
          while (count >= 2u && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], row) <= 0) {
            count = count - 1u;
          }
          scratch[scratchOffset + base + count] = row;
          count = count + 1u;
        }
        let lowerCount = count + 1u;
        var back = lastRep;
        while (back > begin) {
          back = back - 1u;
          if (!isRepresentative(back, begin)) {
            continue;
          }
          while (count >= lowerCount && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], back) <= 0) {
            count = count - 1u;
          }
          scratch[scratchOffset + base + count] = back;
          count = count + 1u;
        }
        for (var vertex = 0u; vertex + 1u < count; vertex++) {
          flags[flagsOffset + scratch[scratchOffset + base + vertex]] = 1u;
        }
      }
    }
    runBegin = runEnd;
  }`,t===0?M:`${M}
// Largest group whose offset is at or before the row (skips empty groups).
fn findGroup(row: u32) -> u32 {
  var low = 0u;
  var high = GROUP_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) >> 1u;
    if (groupOffsets[groupOffsetsOffset + middle] <= row) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  return low;
}`),...new r({id:`${u}-${n}-scan`,input:a,output:o,mode:`exclusive`}).getCommandNodes(e),A(`${n}-offsets`,d+1,[D(`segmentOffsets`,J,`u32`),D(`flags`,a,`u32`),D(`survivorOffsets`,o,`u32`),O(`compactOffsets`,c,`u32`)],`let begin = segmentOffsets[segmentOffsetsOffset + index];
  let total = survivorOffsets[survivorOffsetsOffset + ${w-1}u] + flags[flagsOffset + ${w-1}u];
  compactOffsets[compactOffsetsOffset + index] = select(survivorOffsets[survivorOffsetsOffset + min(begin, ${w-1}u)], total, begin >= ${w}u);`));let m=E(`${n}-lattice-x`,w),h=E(`${n}-lattice-y`,w),g=E(`${n}-permutation`,w);j.push(A(`${n}-compact`,w,[D(`flags`,a,`u32`),D(`survivorOffsets`,o,`u32`),D(`permutation`,Z,`u32`),D(`latticeX`,Y,`u32`),D(`latticeY`,X,`u32`),O(`compactLatticeX`,m,`u32`),O(`compactLatticeY`,h,`u32`),O(`compactPermutation`,g,`u32`)],`if (flags[flagsOffset + index] != 0u) {
    let slot = survivorOffsets[survivorOffsetsOffset + index];
    compactLatticeX[compactLatticeXOffset + slot] = latticeX[latticeXOffset + index];
    compactLatticeY[compactLatticeYOffset + slot] = latticeY[latticeYOffset + index];
    compactPermutation[compactPermutationOffset + slot] = permutation[permutationOffset + index];
  }`)),Y=m,X=h,Z=g,J=c}let re=E(`chain-scratch`,w+d),ie=f.sizes??E(`true-sizes`,d);j.push(A(`chain`,d,[D(`segmentOffsets`,J,`u32`),D(`latticeX`,Y,`u32`),D(`latticeY`,X,`u32`),O(`scratch`,re,`u32`),O(`sizes`,ie,`u32`)],`let begin = segmentOffsets[segmentOffsetsOffset + index];
  let end = segmentOffsets[segmentOffsetsOffset + index + 1u];
  let base = begin + index;
  if (end == begin) {
    sizes[sizesOffset + index] = 0u;
    return;
  }
  var lastRep = end - 1u;
  while (!isRepresentative(lastRep, begin)) {
    lastRep = lastRep - 1u;
  }
  if (lastRep == begin) {
    scratch[scratchOffset + base] = begin;
    sizes[sizesOffset + index] = 1u;
    return;
  }
  var count = 0u;
  for (var row = begin; row <= lastRep; row++) {
    if (!isRepresentative(row, begin)) {
      continue;
    }
    while (count >= 2u && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], row) <= 0) {
      count = count - 1u;
    }
    scratch[scratchOffset + base + count] = row;
    count = count + 1u;
  }
  let lowerCount = count + 1u;
  var row = lastRep;
  while (row > begin) {
    row = row - 1u;
    if (!isRepresentative(row, begin)) {
      continue;
    }
    while (count >= lowerCount && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], row) <= 0) {
      count = count - 1u;
    }
    scratch[scratchOffset + base + count] = row;
    count = count + 1u;
  }
  sizes[sizesOffset + index] = count - 1u;`,M));let Q=E(`eligible-sizes`,d),$=E(`overflow-flags`,1);return j.push(o(e,{id:`${u}-zero-overflow`,operation:g,view:$,type:`u32`,value:`0u`,componentCount:1}),A(`eligible`,d,[D(`sizes`,ie,`u32`),O(`eligible`,Q,`u32`),O(`overflow`,$,`atomic<u32>`)],`let size = sizes[sizesOffset + index];
  let isEligible = size <= ${v}u;
  eligible[eligibleOffset + index] = select(0u, size, isEligible);
  if (!isEligible) {
    atomicOr(&overflow[overflowOffset], 1u);
  }`),...new r({id:`${u}-offset-scan`,input:Q,output:f.offsets,mode:`exclusive`}).getCommandNodes(e),A(`offset-total`,1,[D(`eligible`,Q,`u32`),O(`offsets`,f.offsets,`u32`)],`offsets[offsetsOffset + ${d}u] =
    offsets[offsetsOffset + ${d-1}u] + eligible[eligibleOffset + ${d-1}u];`),o(e,{id:`${u}-fill-indices`,operation:g,view:f.vertexIndices,type:`u32`,value:`0xffffffffu`,componentCount:b}),A(`emit`,d,[D(`segmentOffsets`,J,`u32`),D(`offsets`,f.offsets,`u32`),D(`eligible`,Q,`u32`),D(`scratch`,re,`u32`),D(`permutation`,Z,`u32`),O(`vertexIndices`,f.vertexIndices,`u32`),O(`counts`,f.counts,`u32`),O(`overflow`,$,`atomic<u32>`)],`let size = eligible[eligibleOffset + index];
  let offset = offsets[offsetsOffset + index];
  let base = segmentOffsets[segmentOffsetsOffset + index] + index;
  var emitted = 0u;
  if (size > 0u) {
    if (offset + size > ${b}u) {
      atomicOr(&overflow[overflowOffset], 2u);
    } else {
      emitted = size;
      for (var vertex = 0u; vertex < size; vertex++) {
        vertexIndices[vertexIndicesOffset + offset + vertex] = permutation[permutationOffset + scratch[scratchOffset + base + vertex]];
      }
    }
  }
  counts[countsOffset + index] = emitted;`)),j.push(A(`publish-status`,1,[D(`flags`,$,`u32`),O(`overflow`,f.status.overflow,`u32`),...f.status.groupOverflow?[O(`groupOverflow`,f.status.groupOverflow,`u32`)]:[],...f.status.vertexOverflow?[O(`vertexOverflow`,f.status.vertexOverflow,`u32`)]:[]],`let failures = flags[flagsOffset];
  overflow[overflowOffset] = select(0u, 1u, failures != 0u);
  ${f.status.groupOverflow?`groupOverflow[groupOverflowOffset] = select(0u, 1u, (failures & 1u) != 0u);`:``}
  ${f.status.vertexOverflow?`vertexOverflow[vertexOverflowOffset] = select(0u, 1u, (failures & 2u) != 0u);`:``}`)),f.vertexPositions&&j.push(A(`vertex-positions`,b,[D(`vertexIndices`,f.vertexIndices,`u32`),D(`positions`,p,`f32`),O(`vertexPositions`,f.vertexPositions)],`let row = vertexIndices[vertexIndicesOffset + index];
  let isEmpty = row == 0xffffffffu;
  let nan = getQuietNaN(index);
  vertexPositions[vertexPositionsOffset + index * 2u] = select(positions[positionsOffset + row * 2u], nan, isEmpty);
  vertexPositions[vertexPositionsOffset + index * 2u + 1u] = select(positions[positionsOffset + row * 2u + 1u], nan, isEmpty);`,h)),j}getExtremePointFilterNodes(e,n){let{id:r,groupCount:i}=this,{rows:s,groupKeys:c,latticeX:l,latticeY:u,filteredGroupKeys:d}=n,f=i*S,p=t(e,`${r}-extreme-values`,`uint32`,f),m=t(e,`${r}-extreme-ties`,`uint32`,f),h=t(e,`${r}-octagon`,`uint32`,2*f),_=(e,t,n=`u32`,r=`read`)=>({name:e,view:t,type:n,access:r}),v=i<=w,y=`const GROUP_COUNT: u32 = ${i}u;
const LATTICE_BIAS: u32 = ${D}u;`,b=(t,n,i,o)=>{let d=`let group = groupKeys[groupKeysOffset + row];
    if (group < GROUP_COUNT) {
      let x = latticeX[latticeXOffset + row];
      let y = latticeY[latticeYOffset + row];
      let sum = x + y;
      let difference = x + LATTICE_BIAS - y;
      let base = group * ${S}u;
      ${o((e,t)=>v?`atomicMin(&localTable[${e}], ${t});`:`atomicMin(&table[tableOffset + ${e}], ${t});`)}
    }`,p=[_(`groupKeys`,c),_(`latticeX`,l),_(`latticeY`,u),...i,_(`table`,n,`atomic<u32>`,`read_write`)];if(!v)return a(e,{id:`${r}-${t}`,operation:g,variant:t,bindings:p,invocationCount:s,declarations:y,body:`let row = index;
  ${d}`});let m=Math.ceil(s/T),h=Math.ceil(m/E)*E;return a(e,{id:`${r}-${t}`,operation:g,variant:t,bindings:p,invocationCount:m,workgroupSize:E,guardIndex:!1,declarations:`${y}
const TABLE_LENGTH: u32 = ${f}u;
var<workgroup> localTable: array<atomic<u32>, ${f}>;`,body:`for (var slot = localInvocationIndex; slot < TABLE_LENGTH; slot += ${E}u) {
    atomicStore(&localTable[slot], 0xffffffffu);
  }
  workgroupBarrier();
  for (var rowStep = 0u; rowStep < ${T}u; rowStep++) {
    let row = index + rowStep * ${h}u;
    if (row < ${s}u) {
    ${d}
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < TABLE_LENGTH; slot += ${E}u) {
    let value = atomicLoad(&localTable[slot]);
    if (value != 0xffffffffu) {
      atomicMin(&table[tableOffset + slot], value);
    }
  }`})},x=(t,n)=>o(e,{id:`${r}-fill-${t}`,operation:g,view:n,type:`u32`,value:`0xffffffffu`,componentCount:f}),C=`let minX = values[valuesOffset + base];
      let maxX = ~values[valuesOffset + base + 1u];
      let minY = values[valuesOffset + base + 2u];
      let maxY = ~values[valuesOffset + base + 3u];
      let minSum = values[valuesOffset + base + 4u];
      let maxSum = ~values[valuesOffset + base + 5u];
      let minDifference = values[valuesOffset + base + 6u];
      let maxDifference = ~values[valuesOffset + base + 7u];`;return[x(`extreme-values`,p),b(`extreme-values`,p,[],e=>`${e(`base`,`x`)}
      ${e(`base + 1u`,`~x`)}
      ${e(`base + 2u`,`y`)}
      ${e(`base + 3u`,`~y`)}
      ${e(`base + 4u`,`sum`)}
      ${e(`base + 5u`,`~sum`)}
      ${e(`base + 6u`,`difference`)}
      ${e(`base + 7u`,`~difference`)}`),x(`extreme-ties`,m),b(`extreme-ties`,m,[_(`values`,p)],e=>`${C}
      if (x == minX) { ${e(`base`,`y`)} }
      if (x == maxX) { ${e(`base + 1u`,`y`)} }
      if (y == minY) { ${e(`base + 2u`,`x`)} }
      if (y == maxY) { ${e(`base + 3u`,`x`)} }
      if (sum == minSum) { ${e(`base + 4u`,`x`)} }
      if (sum == maxSum) { ${e(`base + 5u`,`x`)} }
      if (difference == minDifference) { ${e(`base + 6u`,`x`)} }
      if (difference == maxDifference) { ${e(`base + 7u`,`x`)} }`),a(e,{id:`${r}-extreme-points`,operation:g,variant:`extreme-points`,bindings:[_(`values`,p),_(`ties`,m),_(`octagon`,h,`u32`,`read_write`)],invocationCount:i,declarations:`const LATTICE_BIAS: u32 = ${D}u;`,body:`let base = index * ${S}u;
  let outputBase = index * 16u;
  ${C.replace(/^ {6}/gm,`  `)}
  // Empty groups decode to arbitrary words; no row ever reads them, and u32 arithmetic wraps.
  var points = array<vec2<u32>, 8>(
    vec2<u32>(ties[tiesOffset + base + 2u], minY),
    vec2<u32>(ties[tiesOffset + base + 7u], ties[tiesOffset + base + 7u] + LATTICE_BIAS - maxDifference),
    vec2<u32>(maxX, ties[tiesOffset + base + 1u]),
    vec2<u32>(ties[tiesOffset + base + 5u], maxSum - ties[tiesOffset + base + 5u]),
    vec2<u32>(ties[tiesOffset + base + 3u], maxY),
    vec2<u32>(ties[tiesOffset + base + 6u], ties[tiesOffset + base + 6u] + LATTICE_BIAS - minDifference),
    vec2<u32>(minX, ties[tiesOffset + base]),
    vec2<u32>(ties[tiesOffset + base + 4u], minSum - ties[tiesOffset + base + 4u])
  );
  for (var vertex = 0u; vertex < 8u; vertex++) {
    octagon[octagonOffset + outputBase + 2u * vertex] = points[vertex].x;
    octagon[octagonOffset + outputBase + 2u * vertex + 1u] = points[vertex].y;
  }`}),a(e,{id:`${r}-filter-rows`,operation:g,variant:`filter-rows`,bindings:[_(`groupKeys`,c),_(`latticeX`,l),_(`latticeY`,u),_(`octagon`,h),_(`filteredKeys`,d,`u32`,`read_write`)],invocationCount:s,declarations:`${y}
${j}`,body:`let group = groupKeys[groupKeysOffset + index];
  var key = group;
  if (group < GROUP_COUNT) {
    let px = i32(latticeX[latticeXOffset + index]);
    let py = i32(latticeY[latticeYOffset + index]);
    var edgeCount = 0u;
    var isInterior = true;
    for (var edge = 0u; edge < 8u; edge++) {
      let startIndex = octagonOffset + group * 16u + edge * 2u;
      let endIndex = octagonOffset + group * 16u + ((edge + 1u) & 7u) * 2u;
      let ax = i32(octagon[startIndex]);
      let ay = i32(octagon[startIndex + 1u]);
      let bx = i32(octagon[endIndex]);
      let by = i32(octagon[endIndex + 1u]);
      if (ax == bx && ay == by) {
        continue;
      }
      edgeCount++;
      if (orientationOfPoints(ax, ay, bx, by, px, py) <= 0) {
        isInterior = false;
        break;
      }
    }
    if (isInterior && edgeCount > 0u) {
      key = GROUP_COUNT;
    }
  }
  filteredKeys[filteredKeysOffset + index] = key;`})]}},j=`
// Exact 32 x 32 -> 64 bit unsigned product as (high, low); operands are below 2^30.
fn multiplyUnsigned(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xffffu;
  let a1 = a >> 16u;
  let b0 = b & 0xffffu;
  let b1 = b >> 16u;
  let middle = a0 * b1 + a1 * b0;
  let low = a0 * b0 + (middle << 16u);
  let carry = select(0u, 1u, low < a0 * b0);
  return vec2<u32>(a1 * b1 + (middle >> 16u) + carry, low);
}

fn getSign(value: i32) -> i32 { return select(select(0, 1, value > 0), -1, value < 0); }

// Sign of the exact value left - right where each side is (sign, 64-bit magnitude).
fn compareProducts(leftSign: i32, left: vec2<u32>, rightSign: i32, right: vec2<u32>) -> i32 {
  if (leftSign != rightSign) {
    return select(-1, 1, leftSign > rightSign);
  }
  if (leftSign == 0) {
    return 0;
  }
  var magnitude = 0;
  if (left.x != right.x) {
    magnitude = select(-1, 1, left.x > right.x);
  } else if (left.y != right.y) {
    magnitude = select(-1, 1, left.y > right.y);
  }
  return magnitude * leftSign;
}

// Exact orientation of points a, b, c: 1 counter-clockwise, -1 clockwise, 0 collinear.
fn orientationOfPoints(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> i32 {
  let abx = bx - ax;
  let aby = by - ay;
  let acx = cx - ax;
  let acy = cy - ay;
  let leftSign = getSign(abx) * getSign(acy);
  let rightSign = getSign(aby) * getSign(acx);
  let left = multiplyUnsigned(u32(abs(abx)), u32(abs(acy)));
  let right = multiplyUnsigned(u32(abs(aby)), u32(abs(acx)));
  return compareProducts(leftSign, left, rightSign, right);
}
`,M=`${j}
fn isRepresentative(row: u32, begin: u32) -> bool {
  return row == begin || latticeX[latticeXOffset + row] != latticeX[latticeXOffset + row - 1u] ||
    latticeY[latticeYOffset + row] != latticeY[latticeYOffset + row - 1u];
}

// Exact orientation of sorted rows a, b, c: 1 counter-clockwise, -1 clockwise, 0 collinear.
fn orientation(a: u32, b: u32, c: u32) -> i32 {
  return orientationOfPoints(
    i32(latticeX[latticeXOffset + a]), i32(latticeY[latticeYOffset + a]),
    i32(latticeX[latticeXOffset + b]), i32(latticeY[latticeYOffset + b]),
    i32(latticeX[latticeXOffset + c]), i32(latticeY[latticeYOffset + c])
  );
}
`;export{k as n,A as t};
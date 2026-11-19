import{c as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as n}from"./spatial-regression-solve-Vf-sQz9O.js";import{t as r}from"./geodesic-wgsl-I0CYDKcz.js";import{r as i,s as a}from"./kernel-C3tfQOn4.js";import{t as o}from"./gpu-grid-index-BoHdb71C.js";import{t as s}from"./gpu-reduction-tNgOVQ_C.js";import{Q as c}from"./dist-BusamGlx.js";import{O as l,T as u,f as d,g as f,v as p}from"./gpu-contributor-utils-BIwLQON5.js";import{i as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as h}from"./grid-cell-row-order-DUF05y4e.js";import{c as g,u as _}from"./wgsl-common-DfyXiYB0.js";import{y as v}from"./spatial-context-DwZEd4EZ.js";var y=34028234663852886e22,b=2**-149;function x(e,t,n){f(e,t,n,b,y)}function S(e,t){u(e,!t.approximation||!t.status?.approximation||t.approximation===t.status.approximation,`output.approximation compatibility alias must match output.status.approximation`)}function C(e){let{status:t,...n}=e;return{...n,approximation:t?.approximation??n.approximation}}function w(e,t,n=[{writes:t}]){for(let t of n){m(e,Object.fromEntries(t.writes.map((e,t)=>[`write[${t}]`,e])));for(let n of t.reads??[]){if(!n)continue;let r=n.buffer.defaultBuffer?l(n.buffer.defaultBuffer):n.buffer;for(let n of t.writes){if(!n)continue;let t=n.buffer.defaultBuffer?l(n.buffer.defaultBuffer):n.buffer;u(e,r!==t,`outputs must not bind one physical buffer read-only and writable in the same dispatch`)}}}let r=t.flatMap((e,t)=>e&&e.length>0?[{index:t,buffer:e.buffer.defaultBuffer?l(e.buffer.defaultBuffer):e.buffer,start:e.byteOffset,end:e.byteOffset+(e.length-1)*e.byteStride+e.rowByteLength}]:[]);for(let t=0;t<r.length;t++)for(let n=t+1;n<r.length;n++){let i=r[t],a=r[n];u(e,i.buffer!==a.buffer||i.start>=a.end||a.start>=i.end,`output[${i.index}] and output[${a.index}] logical writable byte ranges overlap`)}}function T(e,t,n,r){for(let[r,i]of n.entries())i&&(u(t,(i.buffer.usage&c.STORAGE)!==0,`view[${r}] buffer usage must include Buffer.STORAGE`),p(e,t,`view[${r}] aligned binding`,a(i).size,`reduce the packed view length`),u(t,i.buffer.byteLength<=e.limits.maxBufferSize,`view[${r}] backing buffer needs ${i.buffer.byteLength} bytes (maxBufferSize ${e.limits.maxBufferSize})`));for(let[n,i]of r)p(e,t,n,i,`reduce the row count or capacities`),u(t,i<=e.limits.maxBufferSize,`${n} scratch needs ${i} bytes (maxBufferSize ${e.limits.maxBufferSize})`)}function E(e,t){return t?[[`grid positions and normalized positions`,e*8],[`grid tile bounds`,Math.ceil(Math.ceil(e/256)/64)*64*16],[`grid bounds`,16],[`grid cell counts, offsets and scan scratch`,(t[0]*t[1]+1)*4],[`grid object IDs, cell keys, row IDs and sort ping-pong`,e*4],[`grid sorted IDs and extent`,(e+4)*4],[`grid radix histograms, offsets and scan scratch`,Math.ceil(e/256)*16*4]]:[]}var D=`
fn getBinIndex(value: f32, lower: f32, upper: f32, binCount: u32) -> u32 {
  if (!(upper > lower)) { return 0u; }
  let fraction = clamp((value - lower) / (upper - lower), 0.0, 1.0);
  return min(u32(fraction * f32(binCount)), binCount - 1u);
}
`,O=.05,k=.38197,A=1e-6,j=4e-6,M=65536,N=16777216,P=1048576,F=1024,I=1024,L=1.00001,R=1e-30,z={gaussian:0,bisquare:1},B={fixed:0,adaptive:1},V={OK:0,SINGULAR:1,EXCLUDED:2},H={LOWER:0,LOWER_INTERIOR:1,UPPER:2,UPPER_INTERIOR:3,LOWER_INTERIOR_ROW:4,UPPER_INTERIOR_ROW:5,ITERATION_COUNT:6,ACTIVE:7,OPTIMUM_ROW:8,EVALUATION_COUNT:9,CONVERGED:10,RESERVED:11};function U(e=32){return 4+e}function W(e,t=32,n=new Float32Array(U(t))){let r=U(t);if(n.length<r)throw Error(`GWR parameter target must hold ${r} elements`);let{bandwidths:i}=e,a=i.length;if(!Number.isInteger(a)||a<1||a>t)throw Error(`GWR bandwidths must hold 1 to ${t} values`);let o=z[e.kernel??`bisquare`],s=B[e.bandwidthMode??`fixed`];if(o===void 0||s===void 0)throw Error(`GWR kernel or bandwidth mode is not recognized`);let c=e.significanceLevel===void 0?void 0:Math.fround(e.significanceLevel);if(c!==void 0&&!(c>0&&c<1))throw Error(`GWR significanceLevel must be in (0, 1) after float32 packing`);let l=new Float32Array(a);for(let t=0;t<l.length;t++){if(l[t]=i[t],!Number.isFinite(l[t]))throw Error(`GWR bandwidths must be finite after float32 packing`);s===B.fixed&&v(`GPUGeographicallyWeightedRegression`,e.spatialContext,l[t])}return n.fill(0,0,r),n[0]=o,n[1]=s,n[2]=l.length,n[3]=c??0,n.set(l,4),n}function G(e){if(e<1024)return;let t=Math.min(Math.max(1,Math.ceil(Math.sqrt(e/8))),I);return[t,t]}var K=256,q=4294967294;function J(e,n){let{id:r,operation:a,positions:c,rowCount:l,gridSize:f}=n;d(r,`rowCount`,l,0),u(r,c.length===l,`positions.length must equal rowCount`),d(r,`gridSize[0]`,f[0],1,q),d(r,`gridSize[1]`,f[1],1,q);let p=Math.ceil(l/K),m=f[0]*f[1];d(r,`gridSize cell count`,m,1,q),T(e.device,r,[c],E(l,f));let v=(t,n,a)=>i(e,`${r}-${t}`,n,Math.max(a,1)),y=Math.ceil(p/64)*64,b=v(`tile-bounds`,`float32`,y*4),x=v(`grid-bounds`,`float32`,4),S=v(`unit-positions`,`float32x2`,l),C=v(`cell-offsets`,`uint32`,m+1),w=v(`object-ids`,`uint32`,l),O=v(`sorted-ids`,`uint32`,l+4),k=v(`index-count`,`uint32`,1),A=v(`index-overflow`,`uint32`,1),j=[`minimumX`,`minimumY`,`maximumX`,`maximumY`].map(e=>v(e,`float32`,1)),M=j.flatMap((t,n)=>new s({id:`${r}-extent-component-${n}`,operation:n<2?`min`:`max`,input:e.createDataView(b.buffer,{format:`float32`,length:p,byteOffset:n*y*4}),output:t}).getCommandNodes(e)),N=`const ROW_COUNT: u32 = ${l}u;
const TILE_ROWS: u32 = ${K}u;
const TILE_COUNT: u32 = ${p}u;
const SENTINEL: f32 = 3.0e38;
${g}`,P=h(e,{id:`${r}-cell-row-order`,operation:a,cellOffsets:C,objectIds:w,indexCount:k,rowCount:l}),F=P.sortedRows;return{nodes:[t(e,{id:`${r}-tile-bounds`,operation:a,variant:`tile-bounds`,bindings:[{name:`positions`,view:c,type:`f32`,access:`read`},{name:`tileBounds`,view:b,type:`f32`,access:`read_write`}],invocationCount:p,declarations:N,body:`let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var bounds = vec4f(SENTINEL, SENTINEL, -SENTINEL, -SENTINEL);
  for (var row = firstRow; row < endRow; row++) {
    let x = positions[positionsOffset + 2u * row];
    let y = positions[positionsOffset + 2u * row + 1u];
    if (isFiniteBits(x) && isFiniteBits(y)) {
      bounds = vec4f(min(bounds.x, x), min(bounds.y, y), max(bounds.z, x), max(bounds.w, y));
    }
  }
  for (var component = 0u; component < 4u; component++) {
    tileBounds[tileBoundsOffset + component * ${y}u + index] = bounds[component];
  }`}),...M,t(e,{id:`${r}-grid-bounds`,operation:a,variant:`grid-bounds`,bindings:[...j.map((e,t)=>({name:`component${t}`,view:e,type:`f32`,access:`read`})),{name:`gridBounds`,view:x,type:`f32`,access:`read_write`}],invocationCount:1,declarations:N,body:`let bounds = vec4f(component0[component0Offset], component1[component1Offset], component2[component2Offset], component3[component3Offset]);
  let isEmpty = bounds.x > bounds.z || bounds.y > bounds.w;
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  for (var component = 0u; component < 4u; component++) {
    gridBounds[gridBoundsOffset + component] = select(bounds[component], invalid, isEmpty);
  }`}),t(e,{id:`${r}-unit-positions`,operation:a,variant:`unit-positions`,bindings:[{name:`positions`,view:c,type:`f32`,access:`read`},{name:`gridBounds`,view:x,type:`f32`,access:`read`},{name:`unitPositions`,view:S,type:`f32`,access:`read_write`}],invocationCount:l,declarations:`${_}${D}`,body:`let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let minimum = vec2f(gridBounds[gridBoundsOffset], gridBounds[gridBoundsOffset + 1u]);
  let maximum = vec2f(gridBounds[gridBoundsOffset + 2u], gridBounds[gridBoundsOffset + 3u]);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  let extent = maximum - minimum;
  let included = isFiniteValue(position.x) && isFiniteValue(position.y) && isFiniteValue(minimum.x) && isFiniteValue(minimum.y);
  let unit = select(vec2f(0.0), (position - minimum) / extent, extent > vec2f(0.0));
  unitPositions[unitPositionsOffset + 2u * index] = select(invalid, unit.x, included);
  unitPositions[unitPositionsOffset + 2u * index + 1u] = select(invalid, unit.y, included);`}),...new o({id:`${r}-grid-index`,positions:S,gridSize:[f[0],f[1]],bounds:[0,0,1,1],cellOffsets:C,objectIds:w,count:k,overflow:A}).getCommandNodes(e),...P.nodes,t(e,{id:`${r}-publish-sorted-ids`,operation:a,variant:`publish-sorted-ids`,bindings:[{name:`sortedRows`,view:F,type:`u32`,access:`read`},{name:`gridBounds`,view:x,type:`f32`,access:`read`},{name:`sortedIds`,view:O,type:`u32`,access:`read_write`}],invocationCount:Math.max(l,4),declarations:`const ROW_COUNT: u32 = ${l}u;`,body:`if (index < ROW_COUNT) {
    sortedIds[sortedIdsOffset + index] = sortedRows[sortedRowsOffset + index];
  }
  if (index < 4u) {
    sortedIds[sortedIdsOffset + ROW_COUNT + index] = bitcast<u32>(gridBounds[gridBoundsOffset + index]);
  }`})],cellOffsets:C,sortedIds:O}}function Y(e){return`
const INDEX_WIDTH: u32 = ${e[0]}u;
const INDEX_HEIGHT: u32 = ${e[1]}u;

${_}${D}
// Inclusive cell range (columnLow, columnHigh, rowLow, rowHigh) covering distance < bandwidth.
// The reach is padded for f32 rounding of the box edges; rows in the padding get weight 0 or
// a weight below 1e-8 and the exact distance test in the weight rejects the rest.
fn getGridRange(origin: vec2f, bandwidth: f32) -> vec4u {
  let boundsBase = sortedIdsOffset + ROW_COUNT;
  let minimum = vec2f(bitcast<f32>(sortedIds[boundsBase]), bitcast<f32>(sortedIds[boundsBase + 1u]));
  let maximum = vec2f(bitcast<f32>(sortedIds[boundsBase + 2u]), bitcast<f32>(sortedIds[boundsBase + 3u]));
  let reach = vec2f(bandwidth * 1.0001) + abs(origin) * 2.4e-7;
  let low = clamp(origin - reach, minimum, maximum);
  let high = clamp(origin + reach, minimum, maximum);
  return vec4u(
    getBinIndex(low.x, minimum.x, maximum.x, INDEX_WIDTH),
    getBinIndex(high.x, minimum.x, maximum.x, INDEX_WIDTH),
    getBinIndex(low.y, minimum.y, maximum.y, INDEX_HEIGHT),
    getBinIndex(high.y, minimum.y, maximum.y, INDEX_HEIGHT)
  );
}
`}function X(e=`selectNeighborDistances`,t=!1){return`
fn ${e}(i: u32, maximumK: u32, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) {
  for (var slot = 0u; slot < MAXIMUM_NEIGHBORS; slot++) {
    (*list)[slot] = SENTINEL;
  }
  let origin = getPosition(i);
  let boundsBase = sortedIdsOffset + ROW_COUNT;
  let minimum = vec2f(bitcast<f32>(sortedIds[boundsBase]), bitcast<f32>(sortedIds[boundsBase + 1u]));
  let maximum = vec2f(bitcast<f32>(sortedIds[boundsBase + 2u]), bitcast<f32>(sortedIds[boundsBase + 3u]));
  let cellExtent = (maximum - minimum) / vec2f(f32(INDEX_WIDTH), f32(INDEX_HEIGHT));
  let cellSlack = min(cellExtent.x, cellExtent.y);
  let lastColumn = i32(INDEX_WIDTH) - 1;
  let lastRow = i32(INDEX_HEIGHT) - 1;
  let centerColumn = i32(getBinIndex(clamp(origin.x, minimum.x, maximum.x), minimum.x, maximum.x, INDEX_WIDTH));
  let centerRow = i32(getBinIndex(clamp(origin.y, minimum.y, maximum.y), minimum.y, maximum.y, INDEX_HEIGHT));
  let lastRing = max(max(centerColumn, lastColumn - centerColumn), max(centerRow, lastRow - centerRow));
  for (var ring = 0; ring <= lastRing; ring++) {
    let ringColumnLow = max(centerColumn - ring, 0);
    let ringColumnHigh = min(centerColumn + ring, lastColumn);
    for (var cellRow = max(centerRow - ring, 0); cellRow <= min(centerRow + ring, lastRow); cellRow++) {
      let isEdgeRow = cellRow == centerRow - ring || cellRow == centerRow + ring;
      for (var side = 0; side < 2; side++) {
        var firstColumn = ringColumnLow;
        var endColumn = ringColumnHigh;
        if (isEdgeRow) {
          if (side == 1) { continue; }
        } else {
          let column = select(centerColumn + ring, centerColumn - ring, side == 0);
          if (column < 0 || column > lastColumn) { continue; }
          firstColumn = column;
          endColumn = column;
        }
        let rowBase = u32(cellRow) * INDEX_WIDTH;
        let start = cellOffsets[cellOffsetsOffset + rowBase + u32(firstColumn)];
        let end = cellOffsets[cellOffsetsOffset + rowBase + u32(endColumn) + 1u];
        for (var slot = start; slot < end; slot++) {
          let row = sortedIds[sortedIdsOffset + slot];
          let delta = getPosition(row) - origin;
          let squared = dot(delta, delta);
          if (${t?`squared > 0.0 && `:``}squared < (*list)[maximumK - 1u]) {
            var position = maximumK - 1u;
            loop {
              if (position == 0u || (*list)[position - 1u] <= squared) {
                break;
              }
              (*list)[position] = (*list)[position - 1u];
              position = position - 1u;
            }
            (*list)[position] = squared;
          }
        }
      }
    }
    if ((*list)[maximumK - 1u] < SENTINEL) {
      var gap = 3.0e38;
      if (centerColumn - ring > 0) {
        gap = min(gap, origin.x - (minimum.x + f32(centerColumn - ring) * cellExtent.x));
      }
      if (centerColumn + ring < lastColumn) {
        gap = min(gap, minimum.x + f32(centerColumn + ring + 1) * cellExtent.x - origin.x);
      }
      if (centerRow - ring > 0) {
        gap = min(gap, origin.y - (minimum.y + f32(centerRow - ring) * cellExtent.y));
      }
      if (centerRow + ring < lastRow) {
        gap = min(gap, minimum.y + f32(centerRow + ring + 1) * cellExtent.y - origin.y);
      }
      gap = gap - cellSlack;
      if (gap > 0.0 && (*list)[maximumK - 1u] < gap * gap) {
        break;
      }
    }
  }
}
`}function Z(t=null){return`${t===null?``:`// Gaussian weights vanish beyond this many bandwidths (compile-time gaussianCutoff).
const GAUSSIAN_CUTOFF: f32 = ${e(t)};
`}
fn getLocalKernelWeight(distance: f32, bandwidth: f32, isBisquare: bool) -> f32 {
  let ratio = distance / bandwidth;
  if (isBisquare) {
    if (distance >= bandwidth) {
      return 0.0;
    }
    let t = 1.0 - ratio * ratio;
    return t * t;
  }${t===null?``:`
  if (distance > GAUSSIAN_CUTOFF * bandwidth) {
    return 0.0;
  }`}
  return exp(-0.5 * ratio * ratio);
}
`}function Q(t){let r=e(n**2);return`
fn factorEquilibrated_${t}(a: ptr<function, array<f32, ${t*t}>>, scale: ptr<function, array<f32, ${t}>>) -> bool {
  let factorization = factorDenseCholesky_${t}(a, ${t}u, true, ${r});
  *scale = factorization.scale;
  return factorization.status == 0u;
}

fn solveEquilibrated_${t}(a: ptr<function, array<f32, ${t*t}>>, scale: ptr<function, array<f32, ${t}>>, vector: ptr<function, array<f32, ${t}>>) {
  var factorization = DenseCholesky_${t}(*scale, ${t}u, 0u, 1.0);
  solveDenseCholesky_${t}(a, &factorization, vector);
}

fn inverseDiagonalEquilibrated_${t}(a: ptr<function, array<f32, ${t*t}>>, scale: ptr<function, array<f32, ${t}>>, column: u32) -> f32 {
  return choleskyInverseDiagonal_${t}(a, column) * (*scale)[column] * (*scale)[column];
}
`}function $(t){return t===null?`fn getDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let delta = pointB - pointA;
  return sqrt(dot(delta, delta));
}

fn getSquaredDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let delta = pointB - pointA;
  return dot(delta, delta);
}`:`${r}
// Great-circle distance in the units of the sphere radius; positions are (longitude, latitude).
fn getDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  return geodesicHaversineDistance(pointA, pointB, ${e(t)});
}

fn getSquaredDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let distance = getDistance(pointA, pointB);
  return distance * distance;
}`}function ee(e,t,n=`rowValid[rowValidOffset + row] != 0u`){return t?`let useGrid = isGridBandwidth(bandwidth);
  let range = select(vec4u(0u, 0u, 0u, 0u), getGridRange(origin, getSupportRadius(bandwidth)), useGrid);
  let gridRowCount = select(1u, range.w - range.z + 1u, useGrid);
  for (var gridRowStep = 0u; gridRowStep < gridRowCount; gridRowStep++) {
    var firstSlot = 0u;
    var endSlot = ROW_COUNT;
    if (useGrid) {
      let cellRow = (range.z + gridRowStep) * INDEX_WIDTH;
      firstSlot = cellOffsets[cellOffsetsOffset + cellRow + range.x];
      endSlot = cellOffsets[cellOffsetsOffset + cellRow + range.y + 1u];
    }
    for (var slot = firstSlot; slot < endSlot; slot++) {
      var row = slot;
      if (useGrid) {
        row = sortedIds[sortedIdsOffset + slot];
      }
      if (!(${n})) {
        continue;
      }
      ${e}
    }
  }`:`for (var row = 0u; row < ROW_COUNT; row++) {
    if (!(${n})) {
      continue;
    }
    ${e}
  }`}function te(e,t){return`
${Y(e)}

// Bisquare has bounded support (fixed or adaptive), so the grid is exact for it; the Gaussian
// kernel is bounded only with a compile-time cutoff.
fn isGridBandwidth(bandwidth: f32) -> bool {
  return ${t===null?`getIsBisquare()`:`true`} && bandwidth > 0.0 && isFiniteBits(bandwidth);
}

// Distance beyond which the weight is exactly 0.
fn getSupportRadius(bandwidth: f32) -> f32 {
  ${t===null?`return bandwidth;`:`return select(bandwidth * GAUSSIAN_CUTOFF, bandwidth, getIsBisquare());`}
}

// Included rows in grid-cell order: invocation \`slot\` fits the slot-th indexed row.
fn getFocalCount() -> u32 {
  return min(cellOffsets[cellOffsetsOffset + INDEX_WIDTH * INDEX_HEIGHT], ROW_COUNT);
}

fn getFocalRow(slot: u32) -> u32 {
  return sortedIds[sortedIdsOffset + slot];
}
`}export{w as A,U as C,E as D,C as E,S as O,V as S,D as T,N as _,Z as a,R as b,Y as c,O as d,k as f,P as g,I as h,te as i,x as j,T as k,X as l,A as m,Q as n,J as o,j as p,$ as r,G as s,ee as t,L as u,M as v,W as w,H as x,F as y};
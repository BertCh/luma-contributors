import{a as e,i as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n,t as r}from"./storage-backed-top-k-Dm9YLNx9.js";import{r as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./gpu-grid-coordinate-DwQPYH7E.js";import{t as o}from"./gpu-grid-index-BoHdb71C.js";import{c as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as c,S as l,f as u,g as d,o as f,p,v as m,x as h,y as g}from"./gpu-contributor-utils-BIwLQON5.js";import{t as _}from"./grid-cell-row-order-DUF05y4e.js";import{u as v}from"./raster-band-utils-BcTxc8_Y.js";import{d as y,r as b,u as x}from"./wgsl-common-DfyXiYB0.js";import{n as S,t as C}from"./separation-wgsl-UgBSv4wS.js";function w(e){return`
    for (var indexRow = rowLow; indexRow <= rowHigh; indexRow++) {
      // Cells of one index row are contiguous in cellOffsets, so the row span is one slot range.
      let start = cellOffsets[cellOffsetsOffset + indexRow * INDEX_WIDTH + columnLow];
      let end = cellOffsets[cellOffsetsOffset + indexRow * INDEX_WIDTH + columnHigh + 1u];
      for (var slot = start; slot < end; slot++) {
        let sample = sortedIds[sortedIdsOffset + slot];
        ${e}
      }
    }`}function T(e){return`
    let cellExtent = (DOMAIN_MAX - DOMAIN_MIN) / vec2f(f32(INDEX_WIDTH), f32(INDEX_HEIGHT));
    let cellSlack = min(cellExtent.x, cellExtent.y);
    let centerPoint = clamp(center, DOMAIN_MIN, DOMAIN_MAX);
    let lowColumn = i32(columnLow);
    let highColumn = i32(columnHigh);
    let lowRow = i32(rowLow);
    let highRow = i32(rowHigh);
    let centerColumn = clamp(i32(getCoordinate(centerPoint.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH)), lowColumn, highColumn);
    let centerRow = clamp(i32(getCoordinate(centerPoint.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT)), lowRow, highRow);
    let lastRing = max(max(centerColumn - lowColumn, highColumn - centerColumn), max(centerRow - lowRow, highRow - centerRow));
    for (var ring = 0; ring <= lastRing; ring++) {
      let ringColumnLow = max(centerColumn - ring, lowColumn);
      let ringColumnHigh = min(centerColumn + ring, highColumn);
      for (var cellRow = max(centerRow - ring, lowRow); cellRow <= min(centerRow + ring, highRow); cellRow++) {
        let isEdgeRow = cellRow == centerRow - ring || cellRow == centerRow + ring;
        for (var side = 0; side < 2; side++) {
          var firstColumn = ringColumnLow;
          var lastColumn = ringColumnHigh;
          if (isEdgeRow) {
            // Top and bottom rows of the ring are one contiguous slot range.
            if (side == 1) { continue; }
          } else {
            // Interior rows of the ring touch only its left and right cells.
            let column = select(centerColumn + ring, centerColumn - ring, side == 0);
            if (column < lowColumn || column > highColumn) { continue; }
            firstColumn = column;
            lastColumn = column;
          }
          let rowBase = u32(cellRow) * INDEX_WIDTH;
          let start = cellOffsets[cellOffsetsOffset + rowBase + u32(firstColumn)];
          let end = cellOffsets[cellOffsetsOffset + rowBase + u32(lastColumn) + 1u];
          for (var slot = start; slot < end; slot++) {
            let sample = sortedIds[sortedIdsOffset + slot];
            ${e}
          }
        }
      }
      if (nearCount >= neighborLimit) {
        // Distance from the center to the nearest border of the visited block that still has
        // unvisited cells behind it.
        var gap = 3.0e38;
        if (centerColumn - ring > lowColumn) {
          gap = min(gap, center.x - (DOMAIN_MIN.x + f32(centerColumn - ring) * cellExtent.x));
        }
        if (centerColumn + ring < highColumn) {
          gap = min(gap, DOMAIN_MIN.x + f32(centerColumn + ring + 1) * cellExtent.x - center.x);
        }
        if (centerRow - ring > lowRow) {
          gap = min(gap, center.y - (DOMAIN_MIN.y + f32(centerRow - ring) * cellExtent.y));
        }
        if (centerRow + ring < highRow) {
          gap = min(gap, DOMAIN_MIN.y + f32(centerRow + ring + 1) * cellExtent.y - center.y);
        }
        gap = gap - cellSlack;
        if (gap > 0.0 && nearDistances[neighborLimit - 1u] < gap * gap) {
          break;
        }
      }
    }`}var E=`GPUInverseDistanceWeighting`,D=64,O=16777216,k=4294967295,A=34028234663852886e22,j=1401298464324817e-60,M=1024,N=class{id;props;cellCount;spatialContext;separation;maximumNeighborCount;estimatedCommandNodeCount;constructor(e){this.id=e.id??`inverse-distance-weighting`,this.props=e;let{id:n}=this;if(this.separation=S(n,e.spatialContext),this.spatialContext=this.separation.context,this.separation.geographic&&d(n,`spatialContext.sphereRadius`,this.separation.sphereRadius,j,A),this.cellCount=v(n,e.grid),e.grid.affine[1]!==0||e.grid.affine[3]!==0)throw Error(`${n} needs an axis-aligned grid`);for(let[t,r]of[[`indexGridSize[0]`,e.indexGridSize[0]],[`indexGridSize[1]`,e.indexGridSize[1]]])u(n,t,r,1,k);e.indexBounds.forEach((e,t)=>d(n,`indexBounds[${t}]`,e,-34028234663852886e22,A));let[r,i,a,o]=e.indexBounds;if(e.indexBounds.length!==4||!e.indexBounds.every(Number.isFinite)||r>a||i>o)throw Error(`${n} indexBounds must be finite and ordered`);this.maximumNeighborCount=e.maximumNeighborCount??0,u(n,`maximumNeighborCount`,this.maximumNeighborCount,0,O),l(n,`grid cells`,this.cellCount,k),e.invocationTileSize!==void 0&&u(n,`invocationTileSize`,e.invocationTileSize,1,k);let s=this.maximumNeighborCount>D?2+this.maximumNeighborCount*2:2,f=Math.max(1,Math.floor((e.output.values.buffer.graph.device.limits.maxStorageBufferBindingSize/4-1)/s));this.estimatedCommandNodeCount=M+3*Math.ceil(this.cellCount/Math.min(e.invocationTileSize??this.cellCount,f))+1,p(n,this.estimatedCommandNodeCount),c(e.positions,[`float32x2`],`${n} positions`),c(e.values,[`float32`],`${n} values`);let m=e.values.length;if(m<1)throw Error(`${n} needs at least one sample`);if(e.positions.length!==m)throw Error(`${n} positions length must equal values length`);if(e.mask&&(c(e.mask,[`uint32`],`${n} mask`),e.mask.length!==m))throw Error(`${n} mask length must equal values length`);if(c(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<9)throw Error(`${n} parameters must hold 9 float32 values`);if(c(e.output.values,[`float32`],`${n} output.values`),e.output.values.length<this.cellCount)throw Error(`${n} output.values must hold grid.width * grid.height rows`);if(e.output.counts&&(c(e.output.counts,[`uint32`],`${n} output.counts`),e.output.counts.length<this.cellCount))throw Error(`${n} output.counts must hold grid.width * grid.height rows`);t(n,e.output),g(n,[e.output.values,e.output.counts],[e.positions,e.values,e.mask,e.parameters])}getCommandNodes(t){let{id:n,props:a,cellCount:c,maximumNeighborCount:u}=this,{output:d,grid:p,indexGridSize:g,indexBounds:v}=a,{width:y,height:b}=p;h(n,t,[a.positions,a.values,a.mask,a.parameters,d.values,d.counts]),l(n,`indexGridSize cells`,g[0]*g[1],f(t.device));let x=u>D,S=x||(a.invocationTileSize??c)<c,w=x?2+u*2:2,T=Math.floor((t.device.limits.maxStorageBufferBindingSize/4-1)/w),O=Math.min(a.invocationTileSize??c,c,S?Math.max(1,T):c),k=O*w+1;e(t,n,{positions:a.positions,values:a.values,mask:a.mask,parameters:a.parameters,outputValues:d.values,outputCounts:d.counts});for(let[e,r]of[[`cellOffsets`,(g[0]*g[1]+1)*4],[`sample sorting scratch`,a.values.length*8],[`top-k scratch`,S?k*4:0]])m(t.device,n,e,r,`reduce grid cells, sample count or maximumNeighborCount`);let A=a.values.length,j=g[0]*g[1],M=i(t,`${n}-cell-offsets`,`uint32`,j+1),N=i(t,`${n}-object-ids`,`uint32`,A),L=i(t,`${n}-index-count`,`uint32`,1),R=i(t,`${n}-index-overflow`,`uint32`,1),z=[...new o({id:`${n}-index`,positions:a.positions,gridSize:g,bounds:v,cellOffsets:M,objectIds:N,count:L,overflow:R}).getCommandNodes(t)],B=_(t,{id:`${n}-cell-row-order`,operation:E,cellOffsets:M,objectIds:N,indexCount:L,rowCount:A});z.push(...B.nodes);let V=B.sortedRows,H=S?i(t,`${n}-top-k`,`uint32`,k):void 0,U=[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`sampleValues`,view:a.values,type:`f32`,access:`read`},...a.mask?[{name:`sampleMask`,view:a.mask,type:`u32`,access:`read`}]:[],{name:`params`,view:a.parameters,type:`f32`,access:`read`},{name:`cellOffsets`,view:M,type:`u32`,access:`read`},{name:`sortedIds`,view:V,type:`u32`,access:`read`},...H?[{name:`topk`,view:H,type:`u32`,access:`read_write`}]:[{name:`valuesOut`,view:d.values,type:`f32`,access:`read_write`}],...!H&&d.counts?[{name:`countsOut`,view:d.counts,type:`u32`,access:`read_write`}]:[]],W=u>0,G=`const WIDTH: u32 = ${y}u;
const HEIGHT: u32 = ${b}u;
const INDEX_WIDTH: u32 = ${g[0]}u;
const INDEX_HEIGHT: u32 = ${g[1]}u;
const MAX_NEIGHBORS: u32 = ${u}u;
const DOMAIN_MIN: vec2f = vec2f(${s(v[0])}, ${s(v[1])});
const DOMAIN_MAX: vec2f = vec2f(${s(v[2])}, ${s(v[3])});
${P}
${C(this.separation)}
${F}
const CELL_COUNT: u32 = ${c}u;
const SCRATCH_STRIDE: u32 = ${w}u;
const CURSOR_WORD: u32 = ${O*w}u;`,K=I(!!a.mask,!!d.counts,W,this.separation.geographic,x,!!H);H&&z.push(r(t,{id:`${n}-reset-cursor`,operation:E,variant:`reset-cursor`,bindings:[{name:`topk`,view:H,type:`u32`,access:`read_write`}],invocationCount:1,declarations:G,body:`topk[topkOffset + CURSOR_WORD] = 0u;`}));for(let e=0;e<Math.ceil(c/O);e++)z.push(r(t,{id:`${n}-gather-${e}`,operation:E,variant:W?`gather-nearest`:`gather-radius`,bindings:U,invocationCount:Math.min(O,c-e*O),declarations:G,body:H?`{ let index = index + topk[topkOffset + CURSOR_WORD];
  if (index >= CELL_COUNT) { return; }
  ${K}
}`:K})),H&&z.push(r(t,{id:`${n}-publish-${e}`,operation:E,variant:`publish-storage`,bindings:[{name:`topk`,view:H,type:`u32`,access:`read`},{name:`valuesOut`,view:d.values,type:`f32`,access:`read_write`},...d.counts?[{name:`countsOut`,view:d.counts,type:`u32`,access:`read_write`}]:[]],invocationCount:Math.min(O,c-e*O),declarations:G,body:`let globalRow = topk[topkOffset + CURSOR_WORD] + index;
  if (globalRow >= CELL_COUNT) { return; }
  let base = topkOffset + index * SCRATCH_STRIDE;
  valuesOut[valuesOutOffset + globalRow] = bitcast<f32>(topk[base]);
  ${d.counts?`countsOut[countsOutOffset + globalRow] = topk[base + 1u];`:``}`})),H&&e+1<Math.ceil(c/O)&&z.push(r(t,{id:`${n}-advance-${e}`,operation:E,variant:`advance-cursor`,bindings:[{name:`topk`,view:H,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const CURSOR_WORD: u32 = ${O*w}u;
const TILE_SIZE: u32 = ${O}u;`,body:`topk[topkOffset + CURSOR_WORD] += TILE_SIZE;`}));return z}},P=`
${x}
${y}// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${b}
${a}`,F=`
struct WeightAccumulator {
  maximumLogWeight: f32,
  weightSum: f32,
  weightedSum: f32,
  isEmpty: bool,
}

fn addWeightedSample(accumulator: ptr<function, WeightAccumulator>, power: f32, radius: f32, useTaper: bool, distanceSquared: f32, value: f32) {
  if (distanceSquared == 0.0) {
    return;
  }
  var logWeight = -0.5 * power * log2(distanceSquared);
  if (useTaper) {
    // Franke-Nielson: w = ((R - d) / (R d))^p, zero at and beyond R.
    let distance = sqrt(distanceSquared);
    if (!(distance < radius)) {
      return;
    }
    logWeight = power * (log2(radius - distance) - log2(radius * distance));
  }
  if ((*accumulator).isEmpty) {
    (*accumulator).maximumLogWeight = logWeight;
    (*accumulator).weightSum = 1.0;
    (*accumulator).weightedSum = value;
    (*accumulator).isEmpty = false;
  } else if (logWeight > (*accumulator).maximumLogWeight) {
    let scale = exp2((*accumulator).maximumLogWeight - logWeight);
    (*accumulator).weightSum = (*accumulator).weightSum * scale + 1.0;
    (*accumulator).weightedSum = (*accumulator).weightedSum * scale + value;
    (*accumulator).maximumLogWeight = logWeight;
  } else {
    let weight = exp2(logWeight - (*accumulator).maximumLogWeight);
    (*accumulator).weightSum += weight;
    (*accumulator).weightedSum += weight * value;
  }
}`;function I(e,t,r,i,a,o){let s=n(`topk`,`topkOffset + scratchIndex * SCRATCH_STRIDE + 2u`),c=e=>a?s.readRow(e):`nearRows[${e}]`,l=e=>a?s.readDistance(e):`nearDistances[${e}]`,u=(e,t)=>a?s.writeRow(e,t):`nearRows[${e}] = ${t};`,d=(e,t)=>a?s.writeDistance(e,t):`nearDistances[${e}] = ${t};`,f=r?`${a?``:`var nearDistances: array<f32, MAX_NEIGHBORS>;
  var nearRows: array<u32, MAX_NEIGHBORS>;`}
  var nearCount = 0u;`:``,p=r?`if (neighborLimit == 0u) {
          count += 1u;
          addWeightedSample(&accumulator, power, radius, useTaper, getSquaredMeters(distanceSquared), value);
        } else {
          let last = neighborLimit - 1u;
          let isFull = nearCount >= neighborLimit;
          if (!isFull || distanceSquared < ${l(`last`)} ||
              (distanceSquared == ${l(`last`)} && sample < ${c(`last`)})) {
            var position = select(nearCount, last, isFull);
            loop {
              if (position == 0u) { break; }
              let previousDistance = ${l(`position - 1u`)};
              let previousRow = ${c(`position - 1u`)};
              if (previousDistance < distanceSquared ||
                  (previousDistance == distanceSquared && previousRow < sample)) {
                break;
              }
              ${d(`position`,`previousDistance`)}
              ${u(`position`,`previousRow`)}
              position -= 1u;
            }
            ${d(`position`,`distanceSquared`)}
            ${u(`position`,`sample`)}
            nearCount = min(nearCount + 1u, neighborLimit);
          }
        }`:`count += 1u;
        addWeightedSample(&accumulator, power, radius, useTaper, getSquaredMeters(distanceSquared), value);`,m=r?`if (neighborLimit > 0u) {
    count = nearCount;
    for (var near = 0u; near < nearCount; near++) {
      let row = ${c(`near`)};
      addWeightedSample(&accumulator, power, radius, useTaper, getSquaredMeters(${l(`near`)}), sampleValues[sampleValuesOffset + row]);
    }
  }`:``,h=`let value = sampleValues[sampleValuesOffset + sample];
        if (isNanBits(value)) {
          continue;
        }
        ${e?`if (sampleMask[sampleMaskOffset + sample] == 0u) {
          continue;
        }`:``}
        let position = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]);
        let distanceSquared = getSeparationKey(position - origin, localCenter);
        if (!(distanceSquared <= radiusSquared)) {
          continue;
        }
        if (distanceSquared == 0.0 && sample < exactRow) {
          exactRow = sample;
          exactValue = value;
        }
        ${p}`;return`${o?`let scratchIndex = index - topk[topkOffset + CURSOR_WORD];`:``}
  let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let radius = params[paramsOffset + 4u];
  let power = params[paramsOffset + 5u];
  let neighborLimitValue = params[paramsOffset + 6u];
  let minimumNeighborValue = params[paramsOffset + 7u];
  let neighborLimit = u32(clamp(select(0.0, neighborLimitValue, isFiniteValue(neighborLimitValue)), 0.0, f32(MAX_NEIGHBORS)));
  let minimumNeighbors = max(u32(clamp(select(1.0, minimumNeighborValue, isFiniteValue(minimumNeighborValue)), 0.0, 4294967040.0)), 1u);
  // Use the same positive offsets for either row order. Computing southward offsets from the
  // opposite corner can round a coincident sample away from zero, averaging duplicate hits and
  // applying the minimum-neighbor gate instead of the exact-hit rule.
  let gridMinimum = min(gridStart, gridEnd);
  let cellSize = abs(gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let sampleColumn = select(column, WIDTH - 1u - column, gridEnd.x < gridStart.x);
  let sampleRow = select(rasterRow, HEIGHT - 1u - rasterRow, gridEnd.y < gridStart.y);
  let cellOffset = (vec2f(f32(sampleColumn), f32(sampleRow)) + vec2f(0.5)) * cellSize;
  let center = gridMinimum + cellOffset;
  // Great-circle distances take absolute lon/lat, so the origin is zero there.
  let origin = ${i?`vec2f(0.0)`:`gridMinimum`};
  let localCenter = ${i?`center`:`cellOffset`};
  let useTaper = params[paramsOffset + 8u] > 0.5;
  let radiusSquared = getRadiusKey(radius);
  let queryBox = getQueryBox(center, radius);
  let queryMin = queryBox.minimum;
  let queryMax = queryBox.maximum;
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) && radius >= 0.0 &&
    isFiniteValue(power) && power >= 0.0 && (!useTaper || isFiniteValue(radius)) &&
    all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX);
  var accumulator = WeightAccumulator(0.0, 0.0, 0.0, true);
  var count = 0u;
  var exactRow = 0xffffffffu;
  var exactValue = 0.0;
  ${f}
  if (valid) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    // Widen by one index cell per side to absorb f32 rounding in the binning pass; the exact
    // distance test rejects the extra candidates.
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    // With a per-frame k, the k nearest are order independent (total order (d^2, row)), so an
    // expanding-ring walk may stop early. Radius-only sums depend on the visiting order, so they
    // keep the fixed sorted row-major sweep.
    ${r&&!i?`if (neighborLimit > 0u) {
      ${T(h).replaceAll(`nearDistances[neighborLimit - 1u]`,l(`neighborLimit - 1u`))}
    } else {
      ${w(h)}
    }`:w(h)}
  }
  ${m}
  var result = getNaN();
  if (exactRow != 0xffffffffu) {
    result = exactValue;
  } else if (count >= minimumNeighbors && !accumulator.isEmpty) {
    result = accumulator.weightedSum / accumulator.weightSum;
  }
  ${o?`let resultBase = topkOffset + scratchIndex * SCRATCH_STRIDE;
  topk[resultBase] = bitcast<u32>(result);
  topk[resultBase + 1u] = count;`:`valuesOut[valuesOutOffset + index] = result;
  ${t?`countsOut[countsOutOffset + index] = count;`:``}`}`}export{T as i,P as n,w as r,N as t};
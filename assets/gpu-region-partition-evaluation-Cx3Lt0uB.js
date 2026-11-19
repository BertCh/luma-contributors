import{t as e}from"./spatial-output-validation-nw5lOmVL.js";import{a as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{E as n,S as r,T as i,f as a,s as o,u as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{r as d}from"./kernel-C3tfQOn4.js";import{t as f}from"./gpu-sort-xRm8l5yn.js";import{n as p}from"./load-balanced-for-each-GB0Xy_f0.js";import{o as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as h,s as g,t as _}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{l as v,r as y}from"./wgsl-common-DfyXiYB0.js";var b=`GPURegionPartitionEvaluation`,x=256;function S(e){return s(e,8)}function C(e){return s(e,4,1)}function w(e){return s(e,4)}var T={regionCount:0,withinSumOfSquares:1,betweenSumOfSquares:2,totalSumOfSquares:3,minimumSize:4,maximumSize:5,crossLinkFraction:6,ignoredRows:7,pseudoF:8,length:9},E=class{id;props;constructor(t){let r=t.id??`region-partition-evaluation`;this.id=r,this.props=t;let s=t.columnCount??1,{output:c}=t;a(r,`columnCount`,s,1,32),n(t.values,[`float32`],`${r} values`),n(t.labels,[`uint32`],`${r} labels`);let l=t.labels.length;if(l<1||t.values.length!==l*s)throw Error(`${r} values length must equal labels.length * columnCount`);let d=t.labelCapacity??l;if(a(r,`labelCapacity`,d),t.weights&&g(r,t.weights)!==l)throw Error(`${r} weights row count must equal labels.length`);if(n(c.summary,[`float32`],`${r} output.summary`),c.summary.length<T.length)throw Error(`${r} output.summary must hold ${T.length} words`);if(c.regionSizes&&(n(c.regionSizes,[`uint32`],`${r} output.regionSizes`),c.regionSizes.length<d))throw Error(`${r} output.regionSizes must hold labelCapacity entries`);if(c.regionWithinSumOfSquares&&(n(c.regionWithinSumOfSquares,[`float32`],`${r} output.regionWithinSumOfSquares`),c.regionWithinSumOfSquares.length<d))throw Error(`${r} output.regionWithinSumOfSquares must hold labelCapacity entries`);if(c.regionMeans&&(n(c.regionMeans,[`float32`],`${r} output.regionMeans`),c.regionMeans.length<d*s))throw Error(`${r} output.regionMeans must hold labelCapacity * columnCount entries`);c.invalidRowCount&&(n(c.invalidRowCount,[`uint32`],`${r} output.invalidRowCount`),i(r,c.invalidRowCount.length>=1,`output.invalidRowCount must hold one row`)),e(this.id,Object.fromEntries(o(t.output).map((e,t)=>[`output[${t}]`,e])),[[c.regionSizes,c.regionMeans],[c.summary,c.invalidRowCount]]),u(r,[c.summary,c.regionSizes,c.regionWithinSumOfSquares,c.regionMeans,c.invalidRowCount],[t.values,t.labels,...t.weights?h(t.weights):[]])}getCommandNodes(e){let{id:n,props:i}=this,{values:s,labels:u,weights:g,output:T}=i;l(n,e,[s,u,T.summary,T.regionSizes,T.regionWithinSumOfSquares,T.regionMeans,T.invalidRowCount,...g?h(g):[]]),t(e,n,{values:s,labels:u,...Object.fromEntries(o(T).map((e,t)=>[`output[${t}]`,e])),...Object.fromEntries((g?h(g):[]).map((e,t)=>[`weights[${t}]`,e]))});let E=u.length,D=i.columnCount??1,O=i.labelCapacity??E;r(n,`labels rows`,E,S(e.device)),a(n,`labelCapacity`,O,1,C(e.device)),r(n,`labelCapacity x columnCount`,O*D,w(e.device));for(let[t,r]of[[`region table scratch`,O*D*4],[`label offsets`,(O+1)*4],[`row scratch`,E*4],[`link counts`,g?E*8:0],[`row tile counts`,Math.ceil(E/x)*20],[`row tile sums`,Math.ceil(E/x)*D*4],[`label tile counts`,Math.ceil(O/x)*12]])c(e.device,n,t,r,`reduce rows, labelCapacity or columnCount`);let k=T.regionSizes??d(e,`${n}-sizes`,`uint32`,O),A=T.regionWithinSumOfSquares??d(e,`${n}-within`,`float32`,O),j=g?d(e,`${n}-link-counts`,`uint32`,2*E):void 0,M=d(e,`${n}-label-offsets`,`uint32`,O+1),N=d(e,`${n}-region-column-sums`,`float32`,O*D),P=d(e,`${n}-region-column-deviations`,`float32`,O*D),F=`const ROWS: u32 = ${E}u;
const COLUMNS: u32 = ${D}u;
const LABELS: u32 = ${O}u;`,I=d(e,`${n}-sort-keys`,`uint32`,E),L=d(e,`${n}-row-ids`,`uint32`,E),R=d(e,`${n}-sorted-keys`,`uint32`,E),z=d(e,`${n}-sorted-rows`,`uint32`,E),B=d(e,`${n}-used-labels`,`uint32`,E),V=Math.max(1,Math.ceil(Math.log2(O+1))),H=[m(e,{id:`${n}-sort-keys`,operation:b,variant:`sort-keys`,bindings:[{name:`labels`,view:u,type:`u32`,access:`read`},{name:`values`,view:s,type:`f32`,access:`read`},{name:`sortKeys`,view:I,type:`u32`,access:`read_write`},{name:`usedLabels`,view:B,type:`u32`,access:`read_write`},{name:`rowIds`,view:L,type:`u32`,access:`read_write`}],invocationCount:E,declarations:`${F}
${v}`,body:`let label = labels[labelsOffset + index];
  var isUsed = label < LABELS;
  for (var column = 0u; column < COLUMNS; column++) {
    isUsed = isUsed && isFiniteFloat(values[valuesOffset + index * COLUMNS + column]);
  }
  sortKeys[sortKeysOffset + index] = select(LABELS, label, isUsed);
  usedLabels[usedLabelsOffset + index] = select(0xffffffffu, label, isUsed);
  rowIds[rowIdsOffset + index] = index;`}),...new f({id:`${n}-sort`,keys:I,values:L,outputKeys:R,outputValues:z,keyBits:V}).getCommandNodes(e),m(e,{id:`${n}-label-offsets`,operation:b,variant:`label-offsets`,bindings:[{name:`sortedKeys`,view:R,type:`u32`,access:`read`},{name:`labelOffsets`,view:M,type:`u32`,access:`read_write`},{name:`sizes`,view:k,type:`u32`,access:`read_write`}],invocationCount:O,declarations:`${F}
fn getLowerBound(key: u32) -> u32 {
  var low = 0u;
  var high = ROWS;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,body:`let first = getLowerBound(index);
  let end = getLowerBound(index + 1u);
  labelOffsets[labelOffsetsOffset + index] = first;
  sizes[sizesOffset + index] = end - first;
  if (index + 1u == LABELS) {
    labelOffsets[labelOffsetsOffset + LABELS] = end;
  }`}),...p(e,{id:`${n}-region-sums`,operation:b,offsets:M,itemCapacity:E,laneCount:D,accumulatorType:`f32`,output:N,bindings:[{name:`values`,view:s,type:`f32`,access:`read`},{name:`sortedRows`,view:z,type:`u32`,access:`read`}],declarations:`${F}
fn getLoadBalancedContribution(label: u32, item: u32, column: u32) -> f32 {
  return values[valuesOffset + sortedRows[sortedRowsOffset + item] * COLUMNS + column];
}`}),...p(e,{id:`${n}-region-deviations`,operation:b,offsets:M,itemCapacity:E,laneCount:D,accumulatorType:`f32`,output:P,bindings:[{name:`values`,view:s,type:`f32`,access:`read`},{name:`sortedRows`,view:z,type:`u32`,access:`read`},{name:`regionSums`,view:N,type:`f32`,access:`read`},{name:`sizes`,view:k,type:`u32`,access:`read`}],declarations:`${F}
fn getLoadBalancedContribution(label: u32, item: u32, column: u32) -> f32 {
  // A visited label has at least one row, so the count is positive.
  let mean = regionSums[regionSumsOffset + label * COLUMNS + column] / f32(sizes[sizesOffset + label]);
  let delta = values[valuesOffset + sortedRows[sortedRowsOffset + item] * COLUMNS + column] - mean;
  return delta * delta;
}`}),...T.regionMeans?[m(e,{id:`${n}-region-means`,operation:b,variant:`region-means`,bindings:[{name:`regionSums`,view:N,type:`f32`,access:`read`},{name:`sizes`,view:k,type:`u32`,access:`read`},{name:`regionMeans`,view:T.regionMeans,type:`f32`,access:`read_write`}],invocationCount:O*D,declarations:`${F}
${y}`,body:`let size = sizes[sizesOffset + index / COLUMNS];
  regionMeans[regionMeansOffset + index] = select(getNaN(), regionSums[regionSumsOffset + index] / f32(size), size > 0u);`})]:[],m(e,{id:`${n}-region-within`,operation:b,variant:`region-within`,bindings:[{name:`regionDeviations`,view:P,type:`f32`,access:`read`},{name:`within`,view:A,type:`f32`,access:`read_write`}],invocationCount:O,declarations:F,body:`var ssd = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    ssd += regionDeviations[regionDeviationsOffset + index * COLUMNS + column];
  }
  within[withinOffset + index] = ssd;`})];if(g&&j)for(let t of _(g)){let r=t.isWhole?j:d(e,`${n}-link-counts${t.idSuffix}`,`uint32`,2*t.rowCount);H.push(...p(e,{id:`${n}-links${t.idSuffix}`,operation:b,offsets:t.weights.offsets,itemCapacity:t.weights.neighbors.length,accumulatorType:`vec2<u32>`,output:r,bindings:[{name:`neighbors`,view:t.weights.neighbors,type:`u32`,access:`read`},{name:`labels`,view:B,type:`u32`,access:`read`}],declarations:`${F}${t.constantsWGSL?`\n${t.constantsWGSL}`:``}
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> vec2<u32> {
  let own = labels[labelsOffset + ${t.isWhole?`row`:`WEIGHTS_ROW_START + row`}];
  let other = labels[labelsOffset + neighbors[neighborsOffset + item]];
  if (own >= LABELS || other >= LABELS) {
    return vec2<u32>();
  }
  return vec2<u32>(1u, select(0u, 1u, other != own));
}`})),t.isWhole||H.push(m(e,{id:`${n}-link-counts-copy${t.idSuffix}`,operation:b,variant:`link-counts-copy`,bindings:[{name:`chunkLinks`,view:r,type:`u32`,access:`read`},{name:`links`,view:j,type:`u32`,access:`read_write`}],invocationCount:2*t.rowCount,declarations:t.constantsWGSL,body:`links[linksOffset + 2u * WEIGHTS_ROW_START + index] = chunkLinks[chunkLinksOffset + index];`}))}let U=Math.ceil(E/x),W=Math.ceil(O/x),G=d(e,`${n}-row-counts`,`uint32`,U*5),K=d(e,`${n}-row-sums`,`float32`,U*D),q=d(e,`${n}-label-counts`,`uint32`,W*3),J=d(e,`${n}-label-within`,`float32`,W),Y=d(e,`${n}-label-between`,`float32`,W),X=d(e,`${n}-means`,`float32`,D),Z=d(e,`${n}-deviations`,`float32`,U*D),Q=`${F}
const SUMMARY_TILE: u32 = ${x}u;
const ROW_TILES: u32 = ${U}u;
const LABEL_TILES: u32 = ${W}u;`;return H.push(m(e,{id:`${n}-row-tiles`,operation:b,variant:`row-tiles`,bindings:[{name:`values`,view:s,type:`f32`,access:`read`},{name:`labels`,view:B,type:`u32`,access:`read`},{name:`sourceLabels`,view:u,type:`u32`,access:`read`},...j?[{name:`links`,view:j,type:`u32`,access:`read`}]:[],{name:`rowCounts`,view:G,type:`u32`,access:`read_write`},{name:`rowSums`,view:K,type:`f32`,access:`read_write`}],invocationCount:U,declarations:Q,body:`var labelled = 0u;
  var ignored = 0u;
  var invalid = 0u;
  var cross = 0u;
  var total = 0u;
  var sums: array<f32, ${D}>;
  let firstRow = index * SUMMARY_TILE;
  for (var row = firstRow; row < min(firstRow + SUMMARY_TILE, ROWS); row++) {
    if (labels[labelsOffset + row] < LABELS) {
      labelled += 1u;
      ${j?`total += links[linksOffset + 2u * row]; cross += links[linksOffset + 2u * row + 1u];`:``}
      for (var column = 0u; column < COLUMNS; column++) {
        sums[column] += values[valuesOffset + row * COLUMNS + column];
      }
    } else {
      ignored += 1u;
      // Labelled in the input, so it was left out for a non-finite value.
      invalid += select(0u, 1u, sourceLabels[sourceLabelsOffset + row] < LABELS);
    }
  }
  rowCounts[rowCountsOffset + 5u * index] = labelled;
  rowCounts[rowCountsOffset + 5u * index + 1u] = ignored;
  rowCounts[rowCountsOffset + 5u * index + 2u] = cross;
  rowCounts[rowCountsOffset + 5u * index + 3u] = total;
  rowCounts[rowCountsOffset + 5u * index + 4u] = invalid;
  for (var column = 0u; column < COLUMNS; column++) {
    rowSums[rowSumsOffset + index * COLUMNS + column] = sums[column];
  }`}),m(e,{id:`${n}-means`,operation:b,variant:`means`,bindings:[{name:`rowCounts`,view:G,type:`u32`,access:`read`},{name:`rowSums`,view:K,type:`f32`,access:`read`},{name:`means`,view:X,type:`f32`,access:`read_write`}],invocationCount:D,declarations:Q,body:`var labelled = 0u;
  var sum = 0.0;
  for (var tile = 0u; tile < ROW_TILES; tile++) {
    labelled += rowCounts[rowCountsOffset + 5u * tile];
    sum += rowSums[rowSumsOffset + tile * COLUMNS + index];
  }
  means[meansOffset + index] = sum / max(f32(labelled), 1.0);`}),m(e,{id:`${n}-label-tiles`,operation:b,variant:`label-tiles`,bindings:[{name:`sizes`,view:k,type:`u32`,access:`read`},{name:`within`,view:A,type:`f32`,access:`read`},{name:`regionSums`,view:N,type:`f32`,access:`read`},{name:`means`,view:X,type:`f32`,access:`read`},{name:`labelCounts`,view:q,type:`u32`,access:`read_write`},{name:`labelWithin`,view:J,type:`f32`,access:`read_write`},{name:`labelBetween`,view:Y,type:`f32`,access:`read_write`}],invocationCount:W,declarations:Q,body:`var regionCount = 0u;
  var withinSsd = 0.0;
  var betweenSsd = 0.0;
  var minimumSize = 0xffffffffu;
  var maximumSize = 0u;
  let firstLabel = index * SUMMARY_TILE;
  for (var label = firstLabel; label < min(firstLabel + SUMMARY_TILE, LABELS); label++) {
    let size = sizes[sizesOffset + label];
    if (size > 0u) {
      regionCount += 1u;
      withinSsd += within[withinOffset + label];
      // Between-region sum of squares in its direct form, n_k |mean_k - mean|^2. As
      // total - within it is a difference of two large sums and is lost when the partition
      // explains little.
      let count = f32(size);
      var squaredDistance = 0.0;
      for (var column = 0u; column < COLUMNS; column++) {
        let delta = regionSums[regionSumsOffset + label * COLUMNS + column] / count - means[meansOffset + column];
        squaredDistance += delta * delta;
      }
      betweenSsd += count * squaredDistance;
      minimumSize = min(minimumSize, size);
      maximumSize = max(maximumSize, size);
    }
  }
  labelCounts[labelCountsOffset + 3u * index] = regionCount;
  labelCounts[labelCountsOffset + 3u * index + 1u] = minimumSize;
  labelCounts[labelCountsOffset + 3u * index + 2u] = maximumSize;
  labelWithin[labelWithinOffset + index] = withinSsd;
  labelBetween[labelBetweenOffset + index] = betweenSsd;`}),m(e,{id:`${n}-deviation-tiles`,operation:b,variant:`deviation-tiles`,bindings:[{name:`values`,view:s,type:`f32`,access:`read`},{name:`labels`,view:B,type:`u32`,access:`read`},{name:`means`,view:X,type:`f32`,access:`read`},{name:`deviations`,view:Z,type:`f32`,access:`read_write`}],invocationCount:U,declarations:Q,body:`var squares: array<f32, ${D}>;
  let firstRow = index * SUMMARY_TILE;
  for (var row = firstRow; row < min(firstRow + SUMMARY_TILE, ROWS); row++) {
    if (labels[labelsOffset + row] < LABELS) {
      for (var column = 0u; column < COLUMNS; column++) {
        let delta = values[valuesOffset + row * COLUMNS + column] - means[meansOffset + column];
        squares[column] += delta * delta;
      }
    }
  }
  for (var column = 0u; column < COLUMNS; column++) {
    deviations[deviationsOffset + index * COLUMNS + column] = squares[column];
  }`}),m(e,{id:`${n}-summary`,operation:b,variant:`summary`,bindings:[{name:`rowCounts`,view:G,type:`u32`,access:`read`},{name:`labelCounts`,view:q,type:`u32`,access:`read`},{name:`labelWithin`,view:J,type:`f32`,access:`read`},{name:`labelBetween`,view:Y,type:`f32`,access:`read`},{name:`deviations`,view:Z,type:`f32`,access:`read`},{name:`summary`,view:T.summary,type:`f32`,access:`read_write`},...T.invalidRowCount?[{name:`invalidRowCount`,view:T.invalidRowCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`${Q}
${y}`,body:`var regionCount = 0u;
  var withinSsd = 0.0;
  var between = 0.0;
  var minimumSize = 0xffffffffu;
  var maximumSize = 0u;
  for (var tile = 0u; tile < LABEL_TILES; tile++) {
    regionCount += labelCounts[labelCountsOffset + 3u * tile];
    withinSsd += labelWithin[labelWithinOffset + tile];
    between += labelBetween[labelBetweenOffset + tile];
    minimumSize = min(minimumSize, labelCounts[labelCountsOffset + 3u * tile + 1u]);
    maximumSize = max(maximumSize, labelCounts[labelCountsOffset + 3u * tile + 2u]);
  }
  var ignored = 0u;
  var invalid = 0u;
  var cross = 0u;
  var total = 0u;
  for (var tile = 0u; tile < ROW_TILES; tile++) {
    ignored += rowCounts[rowCountsOffset + 5u * tile + 1u];
    cross += rowCounts[rowCountsOffset + 5u * tile + 2u];
    total += rowCounts[rowCountsOffset + 5u * tile + 3u];
    invalid += rowCounts[rowCountsOffset + 5u * tile + 4u];
  }
  ${T.invalidRowCount?`invalidRowCount[invalidRowCountOffset] = invalid;`:``}
  var totalSsd = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    for (var tile = 0u; tile < ROW_TILES; tile++) {
      totalSsd += deviations[deviationsOffset + tile * COLUMNS + column];
    }
  }
  summary[summaryOffset + 0u] = f32(regionCount);
  summary[summaryOffset + 1u] = withinSsd;
  summary[summaryOffset + 2u] = between;
  summary[summaryOffset + 3u] = totalSsd;
  summary[summaryOffset + 4u] = f32(select(minimumSize, 0u, regionCount == 0u));
  summary[summaryOffset + 5u] = f32(maximumSize);
  summary[summaryOffset + 6u] = select(0.0, f32(cross) / f32(total), total > 0u);
  summary[summaryOffset + 7u] = f32(ignored);
  // Calinski-Harabasz: undefined below two regions or without more rows than regions.
  let labelledRows = f32(ROWS - ignored);
  let regions = f32(regionCount);
  var pseudoF = getNaN();
  if (regionCount >= 2u && labelledRows > regions) {
    pseudoF = select(between * (labelledRows - regions) / (withinSsd * (regions - 1.0)), 1.0, withinSsd == 0.0);
  }
  summary[summaryOffset + 8u] = pseudoF;`})),H}};export{T as n,S as r,E as t};
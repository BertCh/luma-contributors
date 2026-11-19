import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-sort-xRm8l5yn.js";import{l as n,o as r}from"./gpu-command-graph-BdzdmV6u.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,S as o,c as s,f as c,v as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{t as d}from"./symmetric-eigen-wgsl-W1CmtqPF.js";import{c as f,d as p,i as m,r as h,t as g,u as _}from"./wgsl-common-DfyXiYB0.js";import{a as v,i as y}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as b}from"./define-run-B-brkCe4.js";import{c as x,d as S,o as C,s as w}from"./pointer-overlay-Qh8qflcA.js";import{u as T}from"./inputs-CJKKIRKK.js";import{C as E}from"./helpers-DPAd8TjN.js";import{Hc as D}from"./entry-preview-D70TdSbK.js";var O=1e-7,k=1e-5,A=1e-4,j={"min-max":0,"z-score":1,rank:2},M={"weighted-sum":0,"weighted-geometric-mean":1,"principal-component":2};function N(e,t=new Float32Array(36)){if(t.length<36)throw Error(`Composite score parameter target must hold 36 elements`);let{weights:n,directions:r}=e;if(n.length>16||r&&r.length>16)throw Error(`Composite score supports at most 16 weights and directions`);let i=e.epsilon??1e-6;if(!(i>0)||!Number.isFinite(i))throw Error(`Composite score epsilon must be finite and positive`);let a=j[e.scaler??`min-max`],o=M[e.aggregation??`weighted-sum`];if(a===void 0||o===void 0)throw Error(`Composite score scaler or aggregation is not recognized`);t.fill(0,0,36),t[0]=a,t[1]=o,t[2]=i;for(let e=0;e<16;e++){let i=e<n.length?n[e]:0;if(!Number.isFinite(i))throw Error(`Composite score weights must be finite`);if(i<0&&o===M[`weighted-geometric-mean`])throw Error(`Composite score geometric mean weights must be non-negative`);t[4+e]=i;let a=r&&e<r.length?r[e]:1;t[20+e]=a<0?-1:1}return t}var P=4294967295,F=2147483648,I=`GPUCompositeScore`,L=256,R=256,z=class{id;props;rowCount;constructor(e){this.id=e.id??`composite-score`,this.props=e;let t=this.id,{indicatorCount:i,output:l}=e;if(c(t,`indicatorCount`,i,1,16),a(e.indicators,[`float32`],`${t} indicators`),e.indicators.length%i!==0)throw Error(`${t} indicators length must be a multiple of indicatorCount`);if(this.rowCount=e.indicators.length/i,o(t,`indicators length`,e.indicators.length,P),e.enableRank&&o(t,`values rows with enableRank`,this.rowCount,F),this.rowCount<1)throw Error(`${t} needs at least one row`);if(e.mask&&(a(e.mask,[`uint32`],`${t} mask`),e.mask.length!==this.rowCount))throw Error(`${t} mask length must equal the row count`);if(a(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<36)throw Error(`${t} parameters must hold 36 float32 values`);let d=[[`score`,this.rowCount],[`scaled`,this.rowCount*i],[`columnStatistics`,i*4],[`loadings`,i],[`principalComponentSummary`,3]];for(let[e,n]of d){let r=l[e];if(!r){if(e===`score`)throw Error(`${t} output.score is required`)}else if(a(r,[`float32`],`${t} output.${e}`),r.length<n)throw Error(`${t} output.${e} must hold at least ${n} values`)}if((l.loadings||l.principalComponentSummary)&&!e.enablePrincipalComponent)throw Error(`${t} principal-component outputs need enablePrincipalComponent`);{let e=l,i=[[l.loadings,l.principalComponentSummary],[l.columnStatistics,l.loadings,l.score,l.scaled]],a=Object.values(e).flatMap(e=>e?s(e):[]).map(e=>new r(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset,byteStride:4,rowByteLength:4})),o=a.reduce((e,t)=>e+t.length,0);y(t,{outputs:new n({id:`${t}-writable-ranges`,name:`outputs`,format:`uint32`,length:o,valueLength:o,stride:1,byteStride:4,rowByteLength:4,data:a})});for(let e of i)y(t,Object.fromEntries(e.map((e,t)=>[String(t),e])))}u(t,d.map(([e])=>l[e]),[e.indicators,e.mask,e.parameters])}getCommandNodes(n){let{props:r,id:a,rowCount:s}=this,{output:c,indicatorCount:u,parameters:f,indicators:p}=r;v(n,a,{indicators:p,mask:r.mask,parameters:f,...c});let m=!!r.enableRank,h=!!r.enablePrincipalComponent,g=Math.ceil(s/L),y=h?u*u:u,b=h?u*(u+1)/2:u;for(let[e,t]of Object.entries({rowValid:s,tileSums:g*u,tileCounts:g,tileMoments:g*b,moments:y,columnStatistics:u*4,loadings:h?u:0,principalComponentSummary:h?3:0,ranks:m?s*u:0,sortRows:m?s:0,scalars:1}))t>0&&(l(n.device,a,`${e} scratch`,t*4,`reduce rows or indicatorCount`),o(a,`${e} scratch byteLength`,t*4,n.device.limits.maxBufferSize));let x=(t,r,i)=>e(n,`${a}-${t}`,r,Math.max(i,1)),S=x(`row-valid`,`uint32`,s),C=x(`valid-count`,`uint32`,1),w=x(`tile-sums`,`float32`,g*u),T=x(`tile-min-keys`,`uint32`,g*u),E=x(`tile-max-keys`,`uint32`,g*u),D=x(`tile-counts`,`uint32`,g),j=h?u*(u+1)/2:u,M=x(`tile-moments`,`float32`,g*j),N=x(`moments`,`float32`,y),P=c.columnStatistics??x(`column-statistics`,`float32`,u*4),F=h?c.loadings??x(`loadings`,`float32`,u):void 0,z=m?x(`ranks`,`float32`,s*u):void 0,W=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),G=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`}),K=`const COLUMN_COUNT: u32 = ${u}u;
const ROW_COUNT: u32 = ${s}u;
const TILE_ROWS: u32 = ${L}u;
const TILE_COUNT: u32 = ${g}u;
const MOMENT_COUNT: u32 = ${y}u;
const PAIR_COUNT: u32 = ${j}u;
const FULL_MOMENTS: bool = ${h};
const MAXIMUM_COLUMNS: u32 = 16u;
${B}`,q=[];if(q.push(i(n,{id:`${a}-validate`,operation:I,variant:`validate`,bindings:[W(`indicators`,p,`f32`),...r.mask?[W(`rowMask`,r.mask,`u32`)]:[],G(`rowValid`,S,`u32`)],invocationCount:s,declarations:K,body:`var isValid = true;
  ${r.mask?`isValid = rowMask[rowMaskOffset + index] != 0u;`:``}
  let base = indicatorsOffset + index * COLUMN_COUNT;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    if (!isFiniteBits(indicators[base + column])) {
      isValid = false;
    }
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, isValid);`})),q.push(i(n,{id:`${a}-tile-sums`,operation:I,variant:`tile-sums`,bindings:[W(`indicators`,p,`f32`),W(`rowValid`,S,`u32`),G(`tileSums`,w,`f32`),G(`tileMinKeys`,T,`u32`),G(`tileMaxKeys`,E,`u32`),G(`tileCounts`,D,`u32`)],invocationCount:g*u,declarations:K,body:`let tile = index / COLUMN_COUNT;
  let column = index % COLUMN_COUNT;
  let firstRow = tile * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var sum = 0.0;
  var count = 0u;
  var minimumKey = 0xffffffffu;
  var maximumKey = 0u;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] != 0u) {
      let value = indicators[indicatorsOffset + row * COLUMN_COUNT + column];
      sum = sum + value;
      count = count + 1u;
      let key = getOrderedKey(value);
      minimumKey = min(minimumKey, key);
      maximumKey = max(maximumKey, key);
    }
  }
  tileSums[tileSumsOffset + index] = sum;
  tileMinKeys[tileMinKeysOffset + index] = minimumKey;
  tileMaxKeys[tileMaxKeysOffset + index] = maximumKey;
  if (column == 0u) {
    tileCounts[tileCountsOffset + tile] = count;
  }`})),q.push(i(n,{id:`${a}-means`,operation:I,variant:`means`,bindings:[W(`tileSums`,w,`f32`),W(`tileMinKeys`,T,`u32`),W(`tileMaxKeys`,E,`u32`),W(`tileCounts`,D,`u32`),G(`validCount`,C,`u32`),G(`statistics`,P,`f32`)],invocationCount:u*R,workgroupSize:R,guardIndex:!1,declarations:`${K}
var<workgroup> laneCounts: array<u32, ${R}>;
var<workgroup> laneSums: array<f32, ${R}>;
var<workgroup> laneMinimumKeys: array<u32, ${R}>;
var<workgroup> laneMaximumKeys: array<u32, ${R}>;`,body:`let column = workgroupIndex;
  let lane = localInvocationIndex;
  var count = 0u;
  var sum = 0.0;
  var minimumKey = 0xffffffffu;
  var maximumKey = 0u;
  for (var tile = lane; tile < TILE_COUNT; tile += ${R}u) {
    count = count + tileCounts[tileCountsOffset + tile];
    sum = sum + tileSums[tileSumsOffset + tile * COLUMN_COUNT + column];
    minimumKey = min(minimumKey, tileMinKeys[tileMinKeysOffset + tile * COLUMN_COUNT + column]);
    maximumKey = max(maximumKey, tileMaxKeys[tileMaxKeysOffset + tile * COLUMN_COUNT + column]);
  }
  laneCounts[lane] = count;
  laneSums[lane] = sum;
  laneMinimumKeys[lane] = minimumKey;
  laneMaximumKeys[lane] = maximumKey;
  workgroupBarrier();
  for (var stride = ${R/2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneCounts[lane] = laneCounts[lane] + laneCounts[lane + stride];
      laneSums[lane] = laneSums[lane] + laneSums[lane + stride];
      laneMinimumKeys[lane] = min(laneMinimumKeys[lane], laneMinimumKeys[lane + stride]);
      laneMaximumKeys[lane] = max(laneMaximumKeys[lane], laneMaximumKeys[lane + stride]);
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let totalCount = laneCounts[0];
    if (column == 0u) {
      validCount[validCountOffset] = totalCount;
    }
    let nan = getNaN();
    let hasRows = totalCount != 0u;
    let base = statisticsOffset + column * 4u;
    statistics[base] = select(nan, decodeOrderedKey(laneMinimumKeys[0]), hasRows);
    statistics[base + 1u] = select(nan, decodeOrderedKey(laneMaximumKeys[0]), hasRows);
    statistics[base + 2u] = select(nan, laneSums[0] / f32(max(totalCount, 1u)), hasRows);
  }`})),q.push(i(n,{id:`${a}-tile-moments`,operation:I,variant:h?`tile-covariance`:`tile-variance`,bindings:[W(`indicators`,p,`f32`),W(`rowValid`,S,`u32`),W(`statistics`,P,`f32`),G(`tileMoments`,M,`f32`)],invocationCount:g*j,declarations:K,body:`let tile = index / PAIR_COUNT;
  let pair = index % PAIR_COUNT;
  var left = pair;
  var right = pair;
  if (FULL_MOMENTS) {
    // Unrank the upper-triangle pair index into (left <= right).
    left = 0u;
    var remaining = pair;
    while (remaining >= COLUMN_COUNT - left) {
      remaining = remaining - (COLUMN_COUNT - left);
      left = left + 1u;
    }
    right = left + remaining;
  }
  let leftMean = statistics[statisticsOffset + left * 4u + 2u];
  let rightMean = statistics[statisticsOffset + right * 4u + 2u];
  let firstRow = tile * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var sum = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] != 0u) {
      let base = indicatorsOffset + row * COLUMN_COUNT;
      sum = sum + (indicators[base + left] - leftMean) * (indicators[base + right] - rightMean);
    }
  }
  tileMoments[tileMomentsOffset + index] = sum;`})),q.push(i(n,{id:`${a}-moments`,operation:I,variant:`moments`,bindings:[W(`tileMoments`,M,`f32`),W(`validCount`,C,`u32`),G(`moments`,N,`f32`),G(`statistics`,P,`f32`)],invocationCount:y*R,workgroupSize:R,guardIndex:!1,declarations:`${K}
var<workgroup> laneSums: array<f32, ${R}>;`,body:`let momentIndex = workgroupIndex;
  let lane = localInvocationIndex;
  var pair = momentIndex;
  if (FULL_MOMENTS) {
    // Mirror the lower triangle onto the upper one; the products commute, so this is exact.
    let first = momentIndex / COLUMN_COUNT;
    let second = momentIndex % COLUMN_COUNT;
    let low = min(first, second);
    let high = max(first, second);
    pair = low * COLUMN_COUNT - (low * (low - 1u)) / 2u + (high - low);
  }
  var sum = 0.0;
  for (var tile = lane; tile < TILE_COUNT; tile += ${R}u) {
    sum = sum + tileMoments[tileMomentsOffset + tile * PAIR_COUNT + pair];
  }
  laneSums[lane] = sum;
  workgroupBarrier();
  for (var stride = ${R/2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneSums[lane] = laneSums[lane] + laneSums[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let count = validCount[validCountOffset];
    let moment = select(getNaN(), laneSums[0] / f32(max(count, 1u)), count != 0u);
    moments[momentsOffset + momentIndex] = moment;
    var column = momentIndex;
    var isDiagonal = true;
    if (FULL_MOMENTS) {
      column = momentIndex / COLUMN_COUNT;
      isDiagonal = column == momentIndex % COLUMN_COUNT;
    }
    if (isDiagonal) {
      statistics[statisticsOffset + column * 4u + 3u] = select(sqrt(max(moment, 0.0)), moment, isNanBits(moment));
    }
  }`})),h&&F){let e=c.principalComponentSummary??x(`principal-component-summary`,`float32`,3);q.push(i(n,{id:`${a}-principal-component`,operation:I,variant:`principal-component`,bindings:[W(`moments`,N,`f32`),W(`params`,f,`f32`),G(`loadings`,F,`f32`),G(`summary`,e,`f32`)],invocationCount:1,declarations:`${K}
${_}
${d(16)}
const INVERSE_SQRT_TWO: f32 = 0.70710678118654752;
const JACOBI_MAXIMUM_SWEEPS: u32 = 24u;
const JACOBI_TOLERANCE: f32 = ${O};
const POLISH_STEPS: u32 = 4u;
const ZERO_SUM_TOLERANCE: f32 = ${k};
const LOADING_TIE_TOLERANCE: f32 = ${A};`,body:V}))}if(m&&z){let e=x(`sort-keys`,`uint32`,s),r=x(`sort-rows`,`uint32`,s),o=x(`sorted-keys`,`uint32`,s),c=x(`sorted-rows`,`uint32`,s),l=x(`rank-column`,`uint32`,1);q.push(i(n,{id:`${a}-rank-reset`,operation:I,variant:`rank-reset`,bindings:[G(`rankColumnState`,l,`u32`)],invocationCount:1,body:`rankColumnState[rankColumnStateOffset] = 0u;`}));for(let d=0;d<u;d++)q.push(i(n,{id:`${a}-rank-keys-${d}`,operation:I,variant:`rank-keys`,bindings:[W(`indicators`,p,`f32`),W(`rowValid`,S,`u32`),G(`sortKeys`,e,`u32`),G(`sortRows`,r,`u32`),W(`rankColumnState`,l,`u32`)],invocationCount:s,declarations:K,body:`let rankColumn = rankColumnState[rankColumnStateOffset];
  let isValid = rowValid[rowValidOffset + index] != 0u;
  let value = indicators[indicatorsOffset + index * COLUMN_COUNT + rankColumn];
  // Finite keys are below 0xff800000, so excluded rows sort after every included row; -0 ranks as +0.
  sortKeys[sortKeysOffset + index] = select(0xffffffffu, getOrderedKey(select(value, 0.0, value == 0.0)), isValid);
  sortRows[sortRowsOffset + index] = index;`})),q.push(...new t({id:`${a}-rank-sort-${d}`,keys:e,values:r,outputKeys:o,outputValues:c,keyBits:32}).getCommandNodes(n)),q.push(i(n,{id:`${a}-rank-scatter-${d}`,operation:I,variant:`rank-scatter`,bindings:[W(`sortedKeys`,o,`u32`),W(`sortedRows`,c,`u32`),W(`validCount`,C,`u32`),G(`ranks`,z,`f32`),W(`rankColumnState`,l,`u32`)],invocationCount:s,declarations:K,body:H})),q.push(i(n,{id:`${a}-rank-advance-${d}`,operation:I,variant:`rank-advance`,bindings:[G(`rankColumnState`,l,`u32`)],invocationCount:1,declarations:`const COLUMN_COUNT: u32 = ${u}u;`,body:`rankColumnState[rankColumnStateOffset] = (rankColumnState[rankColumnStateOffset] + 1u) % COLUMN_COUNT;`}))}let J=!!c.scaled;return q.push(i(n,{id:`${a}-score`,operation:I,variant:`score`,bindings:[W(`indicators`,p,`f32`),W(`rowValid`,S,`u32`),G(`statistics`,P,`f32`),W(`params`,f,`f32`),...z?[W(`ranks`,z,`f32`)]:[],...F?[G(`loadings`,F,`f32`)]:[],...c.scaled?[G(`scaledOut`,c.scaled,`f32`)]:[],G(`score`,c.score,`f32`)],invocationCount:s,declarations:`${K}
const HAS_RANK: bool = ${m};
const HAS_PRINCIPAL_COMPONENT: bool = ${h};`,body:U({hasRanks:m,hasLoadings:!!F,hasScaled:J})})),q}},B=`
${h}

${p}

${f}

${m}

${g}
`,V=`let directionBase = paramsOffset + 4u + MAXIMUM_COLUMNS;
  var deviations: array<f32, 16>;
  var nonConstant = 0.0;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    let variance = moments[momentsOffset + column * COLUMN_COUNT + column];
    let isUsable = variance > 0.0;
    deviations[column] = select(0.0, sqrt(max(variance, 0.0)), isUsable);
    nonConstant = nonConstant + select(0.0, 1.0, isUsable);
  }
  var correlation: array<f32, 256>;
  for (var row = 0u; row < COLUMN_COUNT; row++) {
    for (var column = 0u; column < COLUMN_COUNT; column++) {
      let scale = deviations[row] * deviations[column];
      let sign = select(-1.0, 1.0, params[directionBase + row] >= 0.0) *
        select(-1.0, 1.0, params[directionBase + column] >= 0.0);
      var value = 0.0;
      if (scale > 0.0) {
        value = sign * moments[momentsOffset + row * COLUMN_COUNT + column] / scale;
      }
      correlation[row * COLUMN_COUNT + column] = value;
    }
  }
  // The first eigenvector. A power iteration is not used: it cannot separate two leading
  // eigenvalues that are close (a correlation of -0.01 between two indicators leaves 64 steps
  // from any start between the two eigenvectors), and its start can be an exact eigenvector of
  // the wrong eigenvalue.
  var vector: array<f32, 16>;
  var product: array<f32, 16>;
  if (COLUMN_COUNT == 2u && nonConstant == 2.0) {
    // Two indicators: [[1, r], [r, 1]] has the eigenpairs (1 + r, (1, 1) / sqrt(2)) and
    // (1 - r, (1, -1) / sqrt(2)) for every r. With r exactly 0 the two tie and (1, 1) is taken.
    vector[0] = INVERSE_SQRT_TWO;
    vector[1] = select(-INVERSE_SQRT_TWO, INVERSE_SQRT_TWO, correlation[1] >= 0.0);
  } else if (nonConstant > 0.0) {
    // Cyclic Jacobi rotations (the shared symmetric eigen decomposition), which resolve close
    // and repeated eigenvalues. It works on a copy with its own stride and destroys it.
    var working: array<f32, 256>;
    var eigenvectors: array<f32, 256>;
    for (var row = 0u; row < COLUMN_COUNT; row++) {
      for (var column = 0u; column < COLUMN_COUNT; column++) {
        working[row * MAXIMUM_COLUMNS + column] = correlation[row * COLUMN_COUNT + column];
      }
    }
    let eigen = getSymmetricEigen_16(
      &working,
      &eigenvectors,
      COLUMN_COUNT,
      JACOBI_MAXIMUM_SWEEPS,
      JACOBI_TOLERANCE
    );
    // Eigenvalues ascend: the last column is the first principal component. A non-finite
    // correlation (no included row) leaves no decomposition and gives NaN loadings.
    for (var column = 0u; column < COLUMN_COUNT; column++) {
      vector[column] = select(
        eigenvectors[column * MAXIMUM_COLUMNS + COLUMN_COUNT - 1u],
        getNaN(),
        eigen.status == 1u
      );
    }
    // A few power steps from the Jacobi vector remove the float32 rounding the rotations leave
    // in the other eigen-directions; they start at the answer, so they need no convergence.
    for (var step = 0u; step < POLISH_STEPS; step++) {
      var norm = 0.0;
      for (var row = 0u; row < COLUMN_COUNT; row++) {
        var sum = 0.0;
        for (var column = 0u; column < COLUMN_COUNT; column++) {
          sum = sum + correlation[row * COLUMN_COUNT + column] * vector[column];
        }
        product[row] = sum;
        norm = norm + sum * sum;
      }
      norm = sqrt(norm);
      if (!(norm > 0.0)) {
        break;
      }
      for (var row = 0u; row < COLUMN_COUNT; row++) {
        vector[row] = product[row] / norm;
      }
    }
  }
  // Sign: the sum of the loadings is positive. When the sum is zero to rounding (always the case
  // for two indicators with a negative correlation), the first loading that ties the largest
  // magnitude is positive, so that rounding does not decide the sign of every score.
  var total = 0.0;
  var absoluteTotal = 0.0;
  var largestMagnitude = 0.0;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    total = total + vector[column];
    absoluteTotal = absoluteTotal + abs(vector[column]);
    largestMagnitude = max(largestMagnitude, abs(vector[column]));
  }
  var leading = 0u;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    if (abs(vector[column]) >= largestMagnitude * (1.0 - LOADING_TIE_TOLERANCE)) {
      leading = column;
      break;
    }
  }
  let isZeroSum = abs(total) <= ZERO_SUM_TOLERANCE * absoluteTotal;
  let orientation = select(
    -1.0,
    1.0,
    (!isZeroSum && total > 0.0) || (isZeroSum && vector[leading] >= 0.0)
  );
  var eigenvalue = 0.0;
  for (var row = 0u; row < COLUMN_COUNT; row++) {
    vector[row] = orientation * vector[row];
  }
  for (var row = 0u; row < COLUMN_COUNT; row++) {
    var sum = 0.0;
    for (var column = 0u; column < COLUMN_COUNT; column++) {
      sum = sum + correlation[row * COLUMN_COUNT + column] * vector[column];
    }
    product[row] = sum;
    eigenvalue = eigenvalue + vector[row] * sum;
  }
  var residual = 0.0;
  for (var row = 0u; row < COLUMN_COUNT; row++) {
    let difference = product[row] - eigenvalue * vector[row];
    residual = residual + difference * difference;
    loadings[loadingsOffset + row] = vector[row];
  }
  let nan = getNaN();
  let hasVariance = nonConstant > 0.0;
  summary[summaryOffset] = select(nan, eigenvalue, hasVariance);
  summary[summaryOffset + 1u] = select(nan, eigenvalue / max(nonConstant, 1.0), hasVariance);
  summary[summaryOffset + 2u] = select(nan, sqrt(residual), hasVariance);`,H=`let rankColumn = rankColumnState[rankColumnStateOffset];
  let count = validCount[validCountOffset];
  if (index >= count) {
    return;
  }
  let key = sortedKeys[sortedKeysOffset + index];
  var low = 0u;
  var high = index;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let first = low;
  low = index;
  high = count;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let last = low - 1u;
  let averageRank = 0.5 * (f32(first) + f32(last));
  let percentile = select(0.5, averageRank / f32(max(count, 2u) - 1u), count > 1u);
  let row = sortedRows[sortedRowsOffset + index];
  ranks[ranksOffset + row * COLUMN_COUNT + rankColumn] = percentile;`;function U(e){let{hasRanks:t,hasLoadings:n,hasScaled:r}=e;return`let nan = getNaN();
  let isValid = rowValid[rowValidOffset + index] != 0u;
  let scaler = u32(params[paramsOffset]);
  let aggregation = u32(params[paramsOffset + 1u]);
  let epsilon = params[paramsOffset + 2u];
  let weightBase = paramsOffset + 4u;
  let directionBase = paramsOffset + 4u + MAXIMUM_COLUMNS;
  let base = indicatorsOffset + index * COLUMN_COUNT;
  let isPrincipal = aggregation == 2u;
  var isSupported = isValid && aggregation <= 2u && scaler <= 2u;
  if (scaler == 2u && !isPrincipal && !HAS_RANK) {
    isSupported = false;
  }
  if (isPrincipal && !HAS_PRINCIPAL_COMPONENT) {
    isSupported = false;
  }
  var weightedSum = 0.0;
  var weightTotal = 0.0;
  var logSum = 0.0;
  var positiveWeightTotal = 0.0;
  var projection = 0.0;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    let value = indicators[base + column];
    let statisticBase = statisticsOffset + column * 4u;
    let isReversed = params[directionBase + column] < 0.0;
    var scaled = 0.0;
    if (isPrincipal || scaler == 1u) {
      let deviation = statistics[statisticBase + 3u];
      scaled = select(0.0, (value - statistics[statisticBase + 2u]) / deviation, deviation > 0.0);
      scaled = select(scaled, -scaled, isReversed);
    } else {
      if (scaler == 0u) {
        let minimum = statistics[statisticBase];
        let range = statistics[statisticBase + 1u] - minimum;
        scaled = select(0.0, (value - minimum) / range, range > 0.0);
      } else {
        ${t?`scaled = ranks[ranksOffset + index * COLUMN_COUNT + column];`:``}
      }
      scaled = select(scaled, 1.0 - scaled, isReversed);
    }
    ${r?`scaledOut[scaledOutOffset + index * COLUMN_COUNT + column] = select(nan, scaled, isSupported);`:``}
    let weight = params[weightBase + column];
    weightedSum = weightedSum + weight * scaled;
    weightTotal = weightTotal + abs(weight);
    let positiveWeight = max(weight, 0.0);
    if (positiveWeight > 0.0) {
      let shifted = scaled + epsilon;
      logSum = logSum + positiveWeight * select(nan, log(shifted), shifted > 0.0);
      positiveWeightTotal = positiveWeightTotal + positiveWeight;
    }
    ${n?`projection = projection + loadings[loadingsOffset + column] * scaled;`:``}
  }
  var result = nan;
  if (isSupported) {
    if (aggregation == 0u) {
      result = select(nan, weightedSum / weightTotal, weightTotal > 0.0);
    } else if (aggregation == 1u) {
      result = select(nan, exp(logSum / positiveWeightTotal), positiveWeightTotal > 0.0);
    } else {
      result = projection;
    }
  }
  score[scoreOffset + index] = result;`}var W=[`min-max`,`z-score`,`rank`],G=[`poverty`,`unemployment`,`age 65+`,`population`],K=[`weighted-sum`,`weighted-geometric-mean`,`principal-component`];function q(e,t){return W[e]===`z-score`||K[t]===`principal-component`?{ramp:`diverging`,range:[-2,2]}:{ramp:`sequential`,range:[0,.7]}}function J({random:e,world:t}){let n=T(e,t,14,.1),r=new Float32Array(n.count*4);for(let e=0;e<n.count;e++)r[e*4]=n.povertyShare[e],r[e*4+1]=n.unemploymentShare[e],r[e*4+2]=n.age65[e]/n.population[e],r[e*4+3]=n.population[e];return{tracts:n,indicators:r,world:t}}var Y=b({parameters:D,caption:`Composite of poverty, unemployment, older population share and population size; the dots show the weighted indicator.`,createInputs:J,build(e,t){let{tracts:n}=t,r=e.upload(`indicators`,`float32`,t.indicators),i=e.allocate(`parameters`,`float32`,36),a=e.allocate(`score`,`float32`,n.count);e.graph.add(new z({indicators:r.view,indicatorCount:4,enableRank:!0,enablePrincipalComponent:!0,parameters:i.view,output:{score:a.view}}));let o=e.allocate(`indicator`,`float32`,n.count);return{outputs:{...E(e,n),score:a,centers:e.upload(`centers`,`float32x2`,n.centers),indicator:o},update:e=>{let r=[1,1,1,1];r[e.indicator]=e.weight,i.write(N({scaler:W[e.scaler],aggregation:K[e.aggregation],weights:r,directions:[1,1,1,1]}));let a=Float32Array.from({length:n.count},(n,r)=>t.indicators[r*4+e.indicator]),s=a.reduce((e,t)=>Math.min(e,t),1/0),c=a.reduce((e,t)=>Math.max(e,t),-1/0);o.write(a.map(e=>(e-s)/(c-s||1)))}}},rebuildOn:[],interaction:{kind:`aim`,hint:`aim at an indicator`,rest:({world:e})=>{let t=e.height*.45,n=w(D,`weight`);return[.5+t*((n.default-n.min)/(n.max-n.min))/e.width,.5]},resolve(e,{world:t,values:n}){let r=t.height*.45,i=[t.width/2,t.height/2],a=e.x-i[0],o=e.y-i[1],s=(90-Math.atan2(o,a)*180/Math.PI+360)%360,c=w(D,`indicator`),l=w(D,`weight`),u={weight:S(l,l.min+C(Math.hypot(a,o)/r,0,1)*(l.max-l.min))};Math.hypot(a,o)>r*.06&&(u.indicator=S(c,Math.floor((s+45)%360/90)));let d={...n,...u},f=r*.3;return{values:u,glyphs:[{shape:`line`,from:[i[0],i[1]+f],to:[i[0],i[1]+r*.9],dashed:!0},{shape:`line`,from:[i[0]+f,i[1]],to:[i[0]+r*.9,i[1]],dashed:!0},{shape:`line`,from:[i[0],i[1]-f],to:[i[0],i[1]-r*.9],dashed:!0},{shape:`line`,from:[i[0]-f,i[1]],to:[i[0]-r*.9,i[1]],dashed:!0},{shape:`arrow`,from:i,to:[e.x,e.y]},{shape:`dot`,at:i}],readout:`weights[${G[d.indicator]}] ${x(d.weight)} · others 1`}}},draw:({inputs:e,outputs:t,values:n})=>[{layer:`polygons`,positions:t.vertices,indices:t.indices,owners:t.owners,values:t.score,triangleCount:e.tracts.triangleCount,...q(n.scaler,n.aggregation)},{layer:`lines`,context:!0,segments:t.outline,count:e.tracts.outlineCount},{layer:`dots`,context:!0,positions:t.centers,values:t.indicator,range:[0,1],radius:3}]});export{Y as default};
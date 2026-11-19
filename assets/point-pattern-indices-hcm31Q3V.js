import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as r}from"./wgsl-common-DfyXiYB0.js";import{E as i,x as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{a as s,c,i as l,l as u,n as d,o as f,s as p,t as m,u as h}from"./pair-statistics-grid-L8bKgCyF.js";import{t as g}from"./pair-statistics-parameters-BSUD2FEq.js";import{v as _}from"./spatial-context-DwZEd4EZ.js";import{t as v}from"./study-window-BK-B07m5.js";import{t as y}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as b}from"./exact-geometry-wgsl-y7TUnqOh.js";import{a as x,n as S,o as C}from"./study-window-wgsl-B3mOUanG.js";import{t as w}from"./define-run-B-brkCe4.js";import{c as T,n as E,t as D}from"./pick-Bo3t3hjv.js";import{t as O}from"./convert-CaKqeDF0.js";import{c as k,p as A,t as j,v as M}from"./helpers-DPAd8TjN.js";import{Ac as N}from"./entry-preview-D70TdSbK.js";var P={none:0,donnelly:1,guard:2},F={"two-sided":0,greater:1,lesser:2},I=65536;function L(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPUPointPatternIndices target must hold 8 elements`);if(g(`GPUPointPatternIndices`,e.bounds,e.maximumDistance),e.bounds===`study-window`){if(e.clarkEvansCorrection===void 0)throw Error(`GPUPointPatternIndices needs an explicit clarkEvansCorrection ('guard' or 'none') with a study window`);if(e.clarkEvansCorrection===`donnelly`)throw Error(`GPUPointPatternIndices clarkEvansCorrection 'donnelly' is defined for rectangular windows only; use 'guard' or 'none' with a study window`)}let n=e.clarkEvansCorrection??`donnelly`;if(!Object.prototype.hasOwnProperty.call(P,n))throw Error(`GPUPointPatternIndices clarkEvansCorrection must be 'none', 'donnelly' or 'guard'`);if(n===`guard`&&(!Number.isFinite(e.guardDistance)||!(e.guardDistance>0)))throw Error(`GPUPointPatternIndices clarkEvansCorrection 'guard' needs a positive guardDistance`);let r=e.alternative??`two-sided`;if(!Object.prototype.hasOwnProperty.call(F,r))throw Error(`GPUPointPatternIndices alternative must be 'two-sided', 'greater' or 'lesser'`);return t.set([...e.bounds===`study-window`?[NaN,NaN,NaN,NaN]:e.bounds,e.maximumDistance,P[n],F[r],e.guardDistance??0]),t}var R=`GPUPointPatternIndices`,z=`
fn getNormalPValue(z: f32, alternative: u32) -> f32 {
  let root = 0.70710678118655;
  if (alternative == 1u) {
    return 0.5 * erfcValue(z * root);
  }
  if (alternative == 2u) {
    return 0.5 * erfcValue(-z * root);
  }
  return erfcValue(abs(z) * root);
}
`,B=`// Stirling's log-gamma expansion (DLMF 5.11.1), centered at x = a to avoid subtracting
// terms of order a * log(a). The log1p(delta) - delta series also avoids cancellation.
fn getGammaPrefix(shape: f32, value: f32) -> f32 {
  if (shape <= 50.0) {
    return exp(shape * log(value) - value - getLogGamma(shape));
  }
  let delta = (value - shape) / shape;
  var centeredLog: f32;
  if (abs(delta) <= 0.25) {
    var power = -delta * delta;
    centeredLog = 0.5 * power;
    for (var order = 3; order <= 20; order++) {
      power *= -delta;
      centeredLog += power / f32(order);
    }
  } else {
    centeredLog = log(value / shape) - delta;
  }
  let inverse = 1.0 / shape;
  let correction = inverse * (0.08333333333333 - inverse * inverse * 0.00277777777778);
  return exp(shape * centeredLog + 0.5 * log(shape) - 0.91893853320467 - correction);
}

`;function V(e){let t=e>101;return`
fn getLogGamma(value: f32) -> f32 {
  var x = value;
  var shift = 0.0;
  loop {
    if (x >= 8.0) { break; }
    shift += log(x);
    x += 1.0;
  }
  let inverse = 1.0 / x;
  let inverseSquared = inverse * inverse;
  return (x - 0.5) * log(x) - x + 0.91893853320467 +
    inverse * (0.08333333333333 - inverseSquared * (0.00277777777778 - inverseSquared * 0.00079365079365)) - shift;
}

${t?B:``}// Returns (lower tail P(a, x), upper tail Q(a, x)) of the regularized incomplete gamma function.
fn getGammaTails(a: f32, x: f32) -> vec2<f32> {
  if (x <= 0.0) {
    return vec2<f32>(0.0, 1.0);
  }
  let prefix = ${t?`getGammaPrefix(a, x)`:`exp(a * log(x) - x - getLogGamma(a))`};
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    var denominator = a;
    for (var iteration = 0; iteration < ${t?2048:500}; iteration++) {
      denominator += 1.0;
      term *= x / denominator;
      sum += term;
      if (abs(term) < abs(sum) * 1.0e-7) { break; }
    }
    let lower = clamp(sum * prefix, 0.0, 1.0);
    return vec2<f32>(lower, 1.0 - lower);
  }
  var b = x + 1.0 - a;
  var c = 1.0e30;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1; iteration < 500; iteration++) {
    let an = -f32(iteration) * (f32(iteration) - a);
    b += 2.0;
    d = an * d + b;
    if (abs(d) < 1.0e-30) { d = 1.0e-30; }
    c = b + an / c;
    if (abs(c) < 1.0e-30) { c = 1.0e-30; }
    d = 1.0 / d;
    let delta = d * c;
    h *= delta;
    if (abs(delta - 1.0) < 1.0e-7) { break; }
  }
  let upper = clamp(prefix * h, 0.0, 1.0);
  return vec2<f32>(1.0 - upper, upper);
}

fn getChiSquarePValue(x: f32, degreesOfFreedom: f32, alternative: u32) -> f32 {
${t?`  let tails = getGammaTails(0.5 * degreesOfFreedom, 0.5 * x);
  let lower = tails.x;
  let upper = tails.y;`:`  var lower: f32;
  var upper: f32;
  if (degreesOfFreedom <= 100.0) {
    let tails = getGammaTails(0.5 * degreesOfFreedom, 0.5 * x);
    lower = tails.x;
    upper = tails.y;
  } else {
    let scale = 2.0 / (9.0 * degreesOfFreedom);
    let z = (pow(x / degreesOfFreedom, 1.0 / 3.0) - (1.0 - scale)) / sqrt(scale);
    lower = getNormalPValue(z, 2u);
    upper = getNormalPValue(z, 1u);
  }`}
  if (alternative == 1u) { return upper; }
  if (alternative == 2u) { return lower; }
  return min(1.0, 2.0 * min(lower, upper));
}
`}var H=class{id;props;spatialContext;studyWindow;constructor(e){this.id=e.id??`point-pattern-indices`,this.props=e;let n=this.id;this.spatialContext=_(n,e.spatialContext),this.studyWindow=v(n,e.studyWindow,this.spatialContext);let{output:r}=e,a=u(n,{...e,parameterLength:8}),s=r.nearestNeighborDistances||r.nearestNeighborIds||r.clarkEvans,c=r.quadratCounts||r.quadratStatistics||r.quadratAreas;if(!s&&!c)throw Error(`${n} output needs at least one view`);if(r.nearestNeighborDistances&&(i(r.nearestNeighborDistances,[`float32`],`${n} output.nearestNeighborDistances`),r.nearestNeighborDistances.length!==a))throw Error(`${n} output.nearestNeighborDistances length must equal positions length`);if(r.nearestNeighborIds&&(t(r.nearestNeighborIds,`${n} output.nearestNeighborIds`),r.nearestNeighborIds.length!==a))throw Error(`${n} output.nearestNeighborIds length must equal positions length`);if(r.clarkEvans&&(i(r.clarkEvans,[`float32`],`${n} output.clarkEvans`),r.clarkEvans.length<7))throw Error(`${n} output.clarkEvans must hold 7 rows`);if(c){let a=e.quadratGrid;if(!a||!Number.isInteger(a[0])||!Number.isInteger(a[1])||a[0]<1||a[1]<1||a[0]*a[1]>65536)throw Error(`${n} quadratGrid must be two positive integers with columns * rows <= ${I}`);let o=e.quadratSamples??16;if(!Number.isInteger(o)||o<1||o**2*a[0]*a[1]>16777216)throw Error(`${n} quadratSamples squared times the quadrat count must be at most 16777216`);if(r.quadratAreas&&(i(r.quadratAreas,[`float32`],`${n} output.quadratAreas`),r.quadratAreas.length!==a[0]*a[1]))throw Error(`${n} output.quadratAreas must hold columns * rows rows`);if(r.quadratCounts&&(t(r.quadratCounts,`${n} output.quadratCounts`),r.quadratCounts.length!==a[0]*a[1]))throw Error(`${n} output.quadratCounts must hold columns * rows rows`);if(r.quadratStatistics&&(i(r.quadratStatistics,[`float32`],`${n} output.quadratStatistics`),r.quadratStatistics.length<7))throw Error(`${n} output.quadratStatistics must hold 7 rows`)}else if(e.quadratGrid)throw Error(`${n} quadratGrid needs output.quadratCounts, output.quadratAreas or output.quadratStatistics`);o(n,[r.nearestNeighborDistances,r.nearestNeighborIds,r.clarkEvans,r.quadratCounts,r.quadratAreas,r.quadratStatistics],[e.positions,e.parameters,e.mask])}getCommandNodes(t){let{id:i,props:o}=this,{positions:u,parameters:g,gridSize:_,output:v}=o;h(i,t.device,u.length),a(i,t,[u,g,o.mask,v.nearestNeighborDistances,v.nearestNeighborIds,v.clarkEvans,v.quadratCounts,v.quadratAreas,v.quadratStatistics]);let w=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(t);w&&a(i,t,[w.edges,w.summary,w.slabs.params,w.slabs.slabOffsets,w.slabs.slabEntries,w.sampleCells,w.sampleCellCount,w.packed]);let T=w?c(t,{id:i,operation:R,parameters:g,parameterLength:g.length,window:w}):void 0,E=T?.parameters??g,D=w?`${y}
${b}
${x(C(w))}`:``,O=u.length,k=_[0]*_[1],A=l(t,{id:i,operation:R,positions:u,parameters:E,gridSize:_,mask:o.mask,studyWindow:w,legacyIndex:!0}),j=[...T?.nodes??[],...A.nodes],M=s(_);if(v.nearestNeighborDistances||v.nearestNeighborIds||v.clarkEvans){let a=v.nearestNeighborDistances??e(t,`${i}-nearest-distances`,`float32`,O),o=v.nearestNeighborIds??e(t,`${i}-nearest-ids`,`uint32`,O),s=f(t,{id:i,operation:R,positions:u,sortedRows:A.sortedRows,cellOffsets:A.cellOffsets,gridSize:_});if(j.push(...s.nodes,n(t,{id:`${i}-nearest`,operation:R,variant:`nearest-neighbor`,bindings:[{name:`parameters`,view:E,type:`f32`,access:`read`},{name:`sortedRows`,view:A.sortedRows,type:`u32`,access:`read`},{name:`sortedPoints`,view:s.sortedPoints,type:`u32`,access:`read`},{name:`cellOffsets`,view:A.cellOffsets,type:`u32`,access:`read`},{name:`distances`,view:a,type:`f32`,access:`read_write`},{name:`ids`,view:o,type:`u32`,access:`read_write`}],invocationCount:O,declarations:M,body:`let lattice = readLattice();
  let focus = sortedRows[sortedRowsOffset + index];
  var nearestDistance = getQuietNaN(index);
  var nearestId = 0xffffffffu;
  if (index < cellOffsets[cellOffsetsOffset + CELL_COUNT] && lattice.valid) {
    let x = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u]);
    let y = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u + 1u]);
    let queryColumn = i32(getCellColumn(lattice, x));
    let queryRow = i32(getCellRow(lattice, y));
    let lastColumn = i32(lattice.columns) - 1;
    let lastRow = i32(lattice.rows) - 1;
    var bestSquared = 3.0e38;
    var bestId = 0xffffffffu;
    let ringCount = max(lattice.columns, lattice.rows);
    // Absolute slack covering f32 rounding of the cell assignment (as in the kNN search).
    let slack = (lattice.cellWidth + lattice.cellHeight) * 0.0009765625 +
      (abs(lattice.minimumX) + abs(lattice.maximumX) + abs(lattice.minimumY) + abs(lattice.maximumY)) * 0.00000095367431640625;
    for (var ringIndex = 0u; ringIndex < ringCount; ringIndex++) {
      let ring = i32(ringIndex);
      let firstColumn = max(queryColumn - ring, 0);
      let endColumn = min(queryColumn + ring, lastColumn);
      let firstRow = max(queryRow - ring, 0);
      let endRow = min(queryRow + ring, lastRow);
      for (var cellRow = firstRow; cellRow <= endRow; cellRow++) {
        let rowBase = u32(cellRow) * lattice.columns;
        // The top and bottom rows of the ring are full (one contiguous slot range, as cells are
        // row major); the others only hold its two end cells.
        let isFullRow = abs(cellRow - queryRow) == ring;
        let step = select(max(2 * ring, 1), 1, isFullRow);
        var columnRangeEnd = endColumn;
        if (!isFullRow) {
          columnRangeEnd = queryColumn + ring;
        }
        var cellColumn = select(queryColumn - ring, firstColumn, isFullRow);
        loop {
          if (cellColumn > columnRangeEnd) { break; }
          if (cellColumn >= 0 && cellColumn <= lastColumn) {
            var begin = cellOffsets[cellOffsetsOffset + rowBase + u32(cellColumn)];
            var end = cellOffsets[cellOffsetsOffset + rowBase + u32(cellColumn) + 1u];
            if (isFullRow) {
              end = cellOffsets[cellOffsetsOffset + rowBase + u32(endColumn) + 1u];
            }
            for (var slot = begin; slot < end; slot++) {
              let neighbor = sortedPoints[sortedPointsOffset + slot * 3u + 2u];
              if (neighbor == focus) {
                continue;
              }
              let deltaX = bitcast<f32>(sortedPoints[sortedPointsOffset + slot * 3u]) - x;
              let deltaY = bitcast<f32>(sortedPoints[sortedPointsOffset + slot * 3u + 1u]) - y;
              let distanceSquared = deltaX * deltaX + deltaY * deltaY;
              if (distanceSquared < bestSquared || (distanceSquared == bestSquared && neighbor < bestId)) {
                bestSquared = distanceSquared;
                bestId = neighbor;
              }
            }
          }
          if (isFullRow) { break; }
          cellColumn += step;
        }
      }
      // Clearance of the visited box on each side; a side at the lattice edge has nothing beyond
      // it. Every unvisited point is at least that far away, so stop once the best is strictly
      // closer (ties at the best distance therefore always resolve to the lowest ID).
      var clearance = 3.0e38;
      var open = false;
      if (queryColumn - ring > 0) {
        clearance = min(clearance, x - (lattice.minimumX + f32(queryColumn - ring) * lattice.cellWidth));
        open = true;
      }
      if (queryColumn + ring < lastColumn) {
        clearance = min(clearance, lattice.minimumX + f32(queryColumn + ring + 1) * lattice.cellWidth - x);
        open = true;
      }
      if (queryRow - ring > 0) {
        clearance = min(clearance, y - (lattice.minimumY + f32(queryRow - ring) * lattice.cellHeight));
        open = true;
      }
      if (queryRow + ring < lastRow) {
        clearance = min(clearance, lattice.minimumY + f32(queryRow + ring + 1) * lattice.cellHeight - y);
        open = true;
      }
      if (!open) {
        break;
      }
      let safeClearance = max(clearance - slack, 0.0);
      if (bestId != 0xffffffffu && bestSquared < safeClearance * safeClearance) {
        break;
      }
    }
    if (bestId != 0xffffffffu) {
      nearestDistance = sqrt(bestSquared);
      nearestId = bestId;
    }
  }
  distances[distancesOffset + focus] = nearestDistance;
  ids[idsOffset + focus] = nearestId;`})),v.clarkEvans){let o=e(t,`${i}-nearest-terms`,`float32`,O),s=e(t,`${i}-nearest-total`,`float32`,1);j.push(n(t,{id:`${i}-nearest-terms`,operation:R,variant:`nearest-terms`,bindings:[{name:`distances`,view:a,type:`f32`,access:`read`},{name:`terms`,view:o,type:`f32`,access:`read_write`}],invocationCount:O,declarations:m,body:`let distance = distances[distancesOffset + index];
  terms[termsOffset + index] = select(0.0, distance, isFiniteValue(distance));`}),...p(t,{id:`${i}-nearest-sum`,operation:R,input:o,output:s}),n(t,{id:`${i}-clark-evans`,operation:R,variant:`clark-evans`,bindings:[...w?[S(w)]:[],{name:`total`,view:s,type:`f32`,access:`read`},{name:`cellOffsets`,view:A.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:E,type:`f32`,access:`read`},{name:`clarkEvans`,view:v.clarkEvans,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${m}${D}
${r}
${z}
const CELL_COUNT: u32 = ${k}u;`,body:`let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let width = parameters[parametersOffset + 2u] - parameters[parametersOffset];
  let height = parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u];
  let area = ${w?`studyWindowSummaryValue(4u)`:`width * height`};
  let correction = u32(parameters[parametersOffset + 5u] + 0.5);
  let alternative = u32(parameters[parametersOffset + 6u] + 0.5);
  let nan = getQuietNaN(index);
  var observed = nan;
  var expected = nan;
  var ratio = nan;
  var standardError = nan;
  var zScore = nan;
  var pValue = nan;
  if (n >= 2.0 && isFiniteValue(area) && area > 0.0) {
    observed = total[totalOffset] / n;
    expected = 0.5 * sqrt(area / n);
    standardError = 0.26136 * sqrt(area) / n;
    if (correction == 1u) {
      // Donnelly (1978) edge correction for a rectangle of perimeter P. sqrt(A / n^5) is
      // evaluated as sqrt(A) / (n^2 sqrt(n)) to stay inside the f32 range for large n.
      let perimeter = 2.0 * (width + height);
      expected += (0.0514 + 0.041 / sqrt(n)) * perimeter / n;
      standardError = sqrt(0.0703 * area / (n * n) + 0.037 * perimeter * sqrt(area) / (n * n * sqrt(n)));
    }
    ${w?`if (correction == 1u) { expected = nan; standardError = nan; }`:``}${w?`
    `:``}ratio = observed / expected;
    zScore = (observed - expected) / standardError;
    pValue = getNormalPValue(zScore, alternative);
  }
  clarkEvans[clarkEvansOffset] = n;
  clarkEvans[clarkEvansOffset + 1u] = observed;
  clarkEvans[clarkEvansOffset + 2u] = expected;
  clarkEvans[clarkEvansOffset + 3u] = ratio;
  clarkEvans[clarkEvansOffset + 4u] = standardError;
  clarkEvans[clarkEvansOffset + 5u] = zScore;
  clarkEvans[clarkEvansOffset + 6u] = pValue;`}));let c=e(t,`${i}-guard-terms`,`float32`,O),l=e(t,`${i}-guard-focus`,`float32`,O),d=e(t,`${i}-guard-total`,`float32`,1),f=e(t,`${i}-guard-count`,`float32`,1);j.push(n(t,{id:`${i}-guard-terms`,operation:R,variant:`guard-terms`,bindings:[{name:`positions`,view:u,type:`f32`,access:`read`},{name:`parameters`,view:E,type:`f32`,access:`read`},{name:`distances`,view:a,type:`f32`,access:`read`},{name:`guardTerms`,view:c,type:`f32`,access:`read_write`},{name:`guardFocus`,view:l,type:`f32`,access:`read_write`},...w?[S(w)]:[]],invocationCount:O,declarations:`${m}${D}`,body:`let distance = distances[distancesOffset + index];
  let guardDistance = parameters[parametersOffset + 7u];
  var focus = false;
  if (parameters[parametersOffset + 5u] == 2.0 && isFiniteValue(distance) &&
      isFiniteValue(guardDistance) && guardDistance > 0.0) {
    let point = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
    let boundaryDistance = ${w?`getStudyWindowBoundaryDistance(point, guardDistance)`:`min(min(point.x - parameters[parametersOffset], parameters[parametersOffset + 2u] - point.x),
      min(point.y - parameters[parametersOffset + 1u], parameters[parametersOffset + 3u] - point.y))`};
    focus = boundaryDistance >= guardDistance;
  }
  guardTerms[guardTermsOffset + index] = select(0.0, distance, focus);
  guardFocus[guardFocusOffset + index] = select(0.0, 1.0, focus);`}),...p(t,{id:`${i}-guard-sum`,operation:R,input:c,output:d}),...p(t,{id:`${i}-guard-count-sum`,operation:R,input:l,output:f}),n(t,{id:`${i}-guard-clark-evans`,operation:R,variant:`guard-clark-evans`,bindings:[{name:`guardTotal`,view:d,type:`f32`,access:`read`},{name:`guardCount`,view:f,type:`f32`,access:`read`},{name:`parameters`,view:E,type:`f32`,access:`read`},{name:`clarkEvans`,view:v.clarkEvans,type:`f32`,access:`read_write`},...w?[S(w)]:[]],invocationCount:1,declarations:`${m}${D}
${r}
${z}`,body:`if (parameters[parametersOffset + 5u] != 2.0) { return; }
  let count = clarkEvans[clarkEvansOffset];
  let focusCount = guardCount[guardCountOffset];
  let area = ${w?`studyWindowSummaryValue(4u)`:`(parameters[parametersOffset + 2u] - parameters[parametersOffset]) * (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u])`};
  let nan = getQuietNaN(index);
  for (var row = 1u; row < 7u; row++) { clarkEvans[clarkEvansOffset + row] = nan; }
  if (count >= 2.0 && focusCount > 0.0 && isFiniteValue(area) && area > 0.0) {
    let observed = guardTotal[guardTotalOffset] / focusCount;
    let expected = 0.5 * sqrt(area / count);
    let standardError = 0.26136 * sqrt(area / (count * focusCount));
    let zScore = (observed - expected) / standardError;
    clarkEvans[clarkEvansOffset + 1u] = observed;
    clarkEvans[clarkEvansOffset + 2u] = expected;
    clarkEvans[clarkEvansOffset + 3u] = observed / expected;
    clarkEvans[clarkEvansOffset + 4u] = standardError;
    clarkEvans[clarkEvansOffset + 5u] = zScore;
    clarkEvans[clarkEvansOffset + 6u] = getNormalPValue(zScore, u32(parameters[parametersOffset + 6u] + 0.5));
  }`}))}}if(v.quadratCounts||v.quadratStatistics||v.quadratAreas){let[a,s]=o.quadratGrid,c=a*s,l=v.quadratCounts??e(t,`${i}-quadrat-counts`,`uint32`,c);j.push(d(t,`${i}-quadrat-clear`,R,l),n(t,{id:`${i}-quadrat-counts`,operation:R,variant:`quadrat-counts`,bindings:[{name:`positions`,view:u,type:`f32`,access:`read`},{name:`parameters`,view:E,type:`f32`,access:`read`},{name:`cellKeys`,view:A.cellKeys,type:`u32`,access:`read`},{name:`quadratCounts`,view:l,type:`atomic<u32>`,access:`read_write`}],invocationCount:O,declarations:`${M}
const QUADRAT_COLUMNS: u32 = ${a}u;
const QUADRAT_ROWS: u32 = ${s}u;`,body:`if (cellKeys[cellKeysOffset + index] < CELL_COUNT) {
    let x = positions[positionsOffset + index * 2u];
    let y = positions[positionsOffset + index * 2u + 1u];
    let width = readParameter(2u) - readParameter(0u);
    let height = readParameter(3u) - readParameter(1u);
    var column = 0u;
    var row = 0u;
    if (width > 0.0) {
      column = min(u32(floor((x - readParameter(0u)) / width * f32(QUADRAT_COLUMNS))), QUADRAT_COLUMNS - 1u);
    }
    if (height > 0.0) {
      row = min(u32(floor((y - readParameter(1u)) / height * f32(QUADRAT_ROWS))), QUADRAT_ROWS - 1u);
    }
    atomicAdd(&quadratCounts[quadratCountsOffset + row * QUADRAT_COLUMNS + column], 1u);
  }`}));let f=w?e(t,`${i}-quadrat-area-samples`,`uint32`,c):void 0,p=w?e(t,`${i}-quadrat-area-totals`,`uint32`,3):void 0;if(w&&f&&p){let e=o.quadratSamples??16;j.push(n(t,{id:`${i}-quadrat-area-samples`,operation:R,variant:`quadrat-area-samples`,bindings:[S(w),{name:`areaSamples`,view:f,type:`u32`,access:`read_write`}],invocationCount:c,declarations:D,body:`let minimum = vec2f(studyWindowSummaryValue(0u), studyWindowSummaryValue(1u));
  let size = (vec2f(studyWindowSummaryValue(2u), studyWindowSummaryValue(3u)) - minimum) / vec2f(${a}.0, ${s}.0);
  let cell = vec2f(f32(index % ${a}u), f32(index / ${a}u));
  var count = 0u;
  for (var sample = 0u; sample < ${e**2}u; sample++) {
    let fraction = (vec2f(f32(sample % ${e}u), f32(sample / ${e}u)) + vec2f(0.5)) / ${e}.0;
    let point = minimum + (cell + fraction) * size;
    if (isInsideStudyWindow(point)) { count++; }
  }
  areaSamples[areaSamplesOffset + index] = count;`}),n(t,{id:`${i}-quadrat-area-totals`,operation:R,variant:`quadrat-area-totals`,bindings:[{name:`areaSamples`,view:f,type:`u32`,access:`read`},{name:`areaTotals`,view:p,type:`u32`,access:`read_write`}],invocationCount:1,body:`var total = 0u;
  var count = 0u;
  var equal = true;
  for (var quadrat = 0u; quadrat < ${c}u; quadrat++) {
    let samples = areaSamples[areaSamplesOffset + quadrat];
    total += samples;
    count += select(0u, 1u, samples > 0u);
    equal = equal && samples == areaSamples[areaSamplesOffset];
  }
  areaTotals[areaTotalsOffset] = total;
  areaTotals[areaTotalsOffset + 1u] = count;
  areaTotals[areaTotalsOffset + 2u] = select(0u, 1u, equal);`}))}v.quadratAreas&&j.push(n(t,{id:`${i}-quadrat-areas`,operation:R,variant:`quadrat-areas`,bindings:[{name:`parameters`,view:E,type:`f32`,access:`read`},{name:`quadratAreas`,view:v.quadratAreas,type:`f32`,access:`read_write`},...w&&f&&p?[S(w),{name:`areaSamples`,view:f,type:`u32`,access:`read`},{name:`areaTotals`,view:p,type:`u32`,access:`read`}]:[]],invocationCount:c,declarations:`${m}${D}`,body:w?`let total = areaTotals[areaTotalsOffset];
  var area = getQuietNaN(index);
  if (total > 0u && isStudyWindowUsable()) {
    area = f32(areaSamples[areaSamplesOffset + index]) * studyWindowSummaryValue(4u) / f32(total);
  }
  quadratAreas[quadratAreasOffset + index] = area;`:`quadratAreas[quadratAreasOffset + index] =
    (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]) / ${c}.0;`})),v.quadratStatistics&&j.push(n(t,{id:`${i}-quadrat-statistics`,operation:R,variant:`quadrat-statistics`,bindings:[{name:`quadratCounts`,view:l,type:`u32`,access:`read`},...f&&p?[{name:`areaSamples`,view:f,type:`u32`,access:`read`},{name:`areaTotals`,view:p,type:`u32`,access:`read`}]:[],{name:`parameters`,view:E,type:`f32`,access:`read`},{name:`cellOffsets`,view:A.cellOffsets,type:`u32`,access:`read`},{name:`quadratStatistics`,view:v.quadratStatistics,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${m}
${r}
${z}
${V(c)}
const CELL_COUNT: u32 = ${k}u;
const QUADRAT_COUNT: u32 = ${c}u;`,body:w?`let nan = getQuietNaN(index);
  let m = f32(areaTotals[areaTotalsOffset + 1u]);
  let total = f32(areaTotals[areaTotalsOffset]);
  let equalAreas = areaTotals[areaTotalsOffset + 2u] != 0u;
  let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let mean = select(nan, n / m, m > 0.0);
  var squares = 0.0;
  var chiSquare = 0.0;
  for (var quadrat = 0u; quadrat < QUADRAT_COUNT; quadrat++) {
    let samples = areaSamples[areaSamplesOffset + quadrat];
    if (samples > 0u) {
      let expected = select(n * f32(samples) / total, mean, equalAreas);
      let centered = f32(quadratCounts[quadratCountsOffset + quadrat]) - expected;
      squares += centered * centered;
      chiSquare += centered * centered / expected;
    }
  }
  let variance = select(nan, squares / (m - 1.0), m > 1.0);
  var ratio = select(nan, chiSquare / (m - 1.0), m > 1.0 && n > 0.0);
  if (equalAreas) {
    // Preserve the legacy operations exactly when every quadrat has the same positive area.
    ratio = select(nan, variance / mean, m > 1.0 && mean > 0.0);
    chiSquare = (m - 1.0) * ratio;
  }
  if (!(m > 1.0 && n > 0.0)) { chiSquare = nan; }
  quadratStatistics[quadratStatisticsOffset] = m;
  quadratStatistics[quadratStatisticsOffset + 1u] = mean;
  quadratStatistics[quadratStatisticsOffset + 2u] = variance;
  quadratStatistics[quadratStatisticsOffset + 3u] = ratio;
  quadratStatistics[quadratStatisticsOffset + 4u] = chiSquare;
  quadratStatistics[quadratStatisticsOffset + 5u] = m - 1.0;
  quadratStatistics[quadratStatisticsOffset + 6u] = select(nan,
    getChiSquarePValue(chiSquare, m - 1.0, u32(parameters[parametersOffset + 6u] + 0.5)),
    isFiniteValue(chiSquare));`:`let nan = getQuietNaN(index);
  let m = f32(QUADRAT_COUNT);
  let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let mean = n / m;
  // One serial fixed-order loop: bitwise reproducible.
  var squares = 0.0;
  for (var quadrat = 0u; quadrat < QUADRAT_COUNT; quadrat++) {
    let centered = f32(quadratCounts[quadratCountsOffset + quadrat]) - mean;
    squares += centered * centered;
  }
  let hasVariance = QUADRAT_COUNT > 1u;
  let variance = select(nan, squares / (m - 1.0), hasVariance);
  let ratio = select(nan, variance / mean, hasVariance && mean > 0.0);
  quadratStatistics[quadratStatisticsOffset] = m;
  quadratStatistics[quadratStatisticsOffset + 1u] = mean;
  quadratStatistics[quadratStatisticsOffset + 2u] = variance;
  quadratStatistics[quadratStatisticsOffset + 3u] = ratio;
  let chiSquare = select(nan, (m - 1.0) * ratio, isFiniteValue(ratio));
  quadratStatistics[quadratStatisticsOffset + 4u] = chiSquare;
  quadratStatistics[quadratStatisticsOffset + 5u] = m - 1.0;
  quadratStatistics[quadratStatisticsOffset + 6u] = select(
    nan,
    getChiSquarePValue(chiSquare, m - 1.0, u32(parameters[parametersOffset + 6u] + 0.5)),
    isFiniteValue(chiSquare));`}))}return j}},U=700;function W({random:e,world:t}){return{source:k(e,U),world:t,pattern:{key:``,positions:new Float32Array(2),index:E(new Float32Array(2)),nearest:new Int32Array,clarkEvans:0},focus:{row:-1}}}function G(e,t,n){let r=`${t}:${n}`;if(e.pattern.key===r)return e.pattern;let i=M(e.source,e.world.width,1,t),a=j(i,A(e.world.width,1,n)),o=Array.from({length:U},(e,t)=>t).filter(e=>a[e]),s=new Int32Array(U).fill(-1),c=0;for(let e of o){let t=1/0;for(let n of o){if(n===e)continue;let r=Math.hypot(i[e*2]-i[n*2],i[e*2+1]-i[n*2+1]);r<t&&(t=r,s[e]=n)}c+=t}let l=e.world.width*(n/100),u=.5/Math.sqrt(o.length/l);return e.pattern={key:r,positions:i,index:E(i),nearest:s,clarkEvans:o.length>1?c/o.length/u:0},e.pattern}var K=w({parameters:N,caption:`Quadrat counts (grid) and nearest-neighbor distance (dots, dark is close).`,createInputs:W,build(e,t,n){let r=n.quadratGrid,{world:i}=t,a=e.allocate(`positions`,`float32x2`,U),o=e.allocate(`mask`,`uint32`,U),s=e.allocate(`parameters`,`float32`,8),c=e.allocate(`nearest`,`float32`,U),l=e.allocate(`quadrat-counts`,`uint32`,r*r);return e.graph.add(new H({positions:a.view,mask:o.view,parameters:s.view,gridSize:[32,32],quadratGrid:[r,r],output:{nearestNeighborDistances:c.view,quadratCounts:l.view}})),{outputs:{positions:a,nearest:c,cells:O(e,`cells`,l,r*r,{flipColumns:r})},update:e=>{a.write(M(t.source,i.width,1,e.clustering)),o.write(j(M(t.source,i.width,1,e.clustering),A(i.width,1,e.mask))),s.write(L({bounds:[0,0,i.width,1],maximumDistance:.05}))}}},rebuildOn:[`quadratGrid`],interaction:{kind:`probe`,hint:`hover a point or a quadrat`,rest:[.4,.5],resolve(e,{inputs:t,outputs:n,values:r,pixelSize:i,peek:a}){let o=G(t,r.clustering,r.mask),s=r.quadratGrid,c=D(o.index,e.x,e.y,t.focus.row),l=c>=0&&o.nearest[c]>=0&&o.index.distanceTo(c,e.x,e.y)<i*9;t.focus.row=l?c:-1;let u=`R ${o.clarkEvans.toFixed(2)}`,d=[];if(l){let e=o.nearest[c],t=[o.positions[c*2],o.positions[c*2+1]],n=[o.positions[e*2],o.positions[e*2+1]];return d.push({shape:`line`,from:t,to:n},{shape:`dot`,at:n},{shape:`ring`,at:t,radius:i*7}),{glyphs:d,readout:`nn ${Math.hypot(n[0]-t[0],n[1]-t[1]).toFixed(3)} · ${u}`}}let f=T(s,s,[0,0,t.world.width,1],e.x,e.y),p=f.column/s*t.world.width,m=(f.column+1)/s*t.world.width;d.push({shape:`box`,minimum:[p,1-(f.row+1)/s],maximum:[m,1-f.row/s]});let h=a(n.cells)?.[f.index],g=U*(r.mask/100)/(s*s);return{glyphs:d,readout:`quadrat ${h??`…`} points (expected ${g.toFixed(0)}) · ${u}`}}},draw:({outputs:e,values:t})=>[{layer:`raster`,values:e.cells,columns:t.quadratGrid,rows:t.quadratGrid,range:[0,U/(t.quadratGrid*t.quadratGrid)*3.5],alpha:.8},{layer:`dots`,positions:e.positions,values:e.nearest,ramp:`sequential`,range:[0,.04],radius:2.8}]});export{K as default};
import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-compaction-B6IzcF0P.js";import{t as n}from"./gpu-flag-offsets-uwVnNtbl.js";import{t as r}from"./gpu-group-aggregation-CEYb6GJw.js";import{n as i,o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as s,S as c,m as l,o as u,v as d,x as ee,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{a as p}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as m}from"./spatial-output-validation-nw5lOmVL.js";import{i as h,n as g}from"./key-pair-grouping-DgpiCtBP.js";import{l as _}from"./wgsl-common-DfyXiYB0.js";import{t as v}from"./geodesic-wgsl-I0CYDKcz.js";import{n as te,t as y}from"./point-neighbor-grid-index-D0wS1y1I.js";import{o as b}from"./sorted-segment-sums-Dvmm6OdR.js";import{n as x}from"./temporal-context-xddVhAJD.js";import{a as S,g as C,v as w}from"./spatial-context-DwZEd4EZ.js";var T=`GPUSpatialClustering`;function E(e,t,n,r=!1){let i=n!==void 0,a=i?String(Math.fround(n)):`1`,o=i?`return geodesicHaversineTerm(vec2<f32>(ax, ay), vec2<f32>(bx, by)) <= lattice.epsilonSquared;`:`let deltaX = bx - ax;
  let deltaY = by - ay;
  return deltaX * deltaX + deltaY * deltaY <= lattice.epsilonSquared;`;return`
${i?v:``}
const GEOGRAPHIC: bool = ${i};
const SPHERE_RADIUS: f32 = ${/[.e]/.test(a)?a:`${a}.0`};
const DEGREES_PER_RADIAN: f32 = 57.2957795130823;
const RADIANS_PER_DEGREE: f32 = 0.0174532925199433;
const COLUMNS: u32 = ${e}u;
const ROWS: u32 = ${t}u;
const NOISE: u32 = 0xffffffffu;
const CELL_MARGIN: f32 = 1.0009765625;

struct Lattice {
  valid: bool,
  minimumX: f32,
  minimumY: f32,
  maximumX: f32,
  maximumY: f32,
  cellWidth: f32,
  cellHeight: f32,
  columns: u32,
  rows: u32,
  epsilonSquared: f32,
  minimumPoints: u32${r?`,
  timeEpsilon: f32`:``}
}

${_}

fn readLattice() -> Lattice {
  var lattice: Lattice;
  lattice.cellWidth = 1.0;
  lattice.cellHeight = 1.0;
  lattice.columns = 1u;
  lattice.rows = 1u;
  lattice.epsilonSquared = 0.0;
  lattice.minimumPoints = 1u;${r?`
  lattice.timeEpsilon = 0.0;
  let timeEpsilon = parameters[parametersOffset + 6u];`:``}
  let minimumX = parameters[parametersOffset];
  let minimumY = parameters[parametersOffset + 1u];
  let maximumX = parameters[parametersOffset + 2u];
  let maximumY = parameters[parametersOffset + 3u];
  let epsilon = parameters[parametersOffset + 4u];
  let minimumPointsValue = parameters[parametersOffset + 5u];
  let width = maximumX - minimumX;
  let height = maximumY - minimumY;
  // Planar: neighbors satisfy dx^2 + dy^2 <= epsilon^2, so a cell must be epsilon wide. Geographic:
  // the haversine term of neighbors is at most sin^2(epsilon / (2 R)).
  var epsilonSquared = epsilon * epsilon;
  var cellStepX = epsilon * CELL_MARGIN;
  var cellStepY = cellStepX;
  var inRange = true;
  if (GEOGRAPHIC) {
    let halfAngle = 0.5 * epsilon / SPHERE_RADIUS;
    let sinHalf = sin(halfAngle);
    epsilonSquared = sinHalf * sinHalf;
    cellStepY = 2.0 * halfAngle * DEGREES_PER_RADIAN * CELL_MARGIN;
    let cosLatitude = cos(max(abs(minimumY), abs(maximumY)) * RADIANS_PER_DEGREE);
    let ratio = sinHalf / max(cosLatitude, 1e-6);
    cellStepX = select(2.0 * asin(ratio) * DEGREES_PER_RADIAN * CELL_MARGIN, 360.0, ratio >= 0.99);
    inRange = minimumX >= -180.0 && maximumX <= 180.0 && minimumY >= -90.0 && maximumY <= 90.0 &&
      halfAngle < 1.5707963;
  }
  lattice.valid = inRange &&
    isFiniteFloat(minimumX) && isFiniteFloat(minimumY) && isFiniteFloat(maximumX) &&
    isFiniteFloat(maximumY) && isFiniteFloat(epsilon) && isFiniteFloat(minimumPointsValue) &&
    isFiniteFloat(width) && isFiniteFloat(height) && isFiniteFloat(epsilonSquared) &&
    isFiniteFloat(cellStepX) && isFiniteFloat(cellStepY) && epsilon > 0.0 && width >= 0.0 &&
    height >= 0.0${r?` &&
    isFiniteFloat(timeEpsilon) && timeEpsilon > 0.0`:``};
  if (!lattice.valid) {
    return lattice;
  }
  lattice.minimumX = minimumX;
  lattice.minimumY = minimumY;
  lattice.maximumX = maximumX;
  lattice.maximumY = maximumY;
  lattice.cellWidth = max(width / f32(COLUMNS), cellStepX);
  lattice.cellHeight = max(height / f32(ROWS), cellStepY);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.epsilonSquared = epsilonSquared;
  lattice.minimumPoints = u32(max(minimumPointsValue, 0.0));${r?`
  lattice.timeEpsilon = timeEpsilon;`:``}
  return lattice;
}
${r?`
// Time filter of ST-DBSCAN: two points are related only when their relative times differ by at
// most the temporal epsilon. A non-finite time compares false, so such a point has no neighbor,
// not even itself, and ends as noise.
fn isWithinTime(lattice: Lattice, ta: f32, tb: f32) -> bool {
  return abs(tb - ta) <= lattice.timeEpsilon;
}
`:``}
fn isWithinEpsilon(lattice: Lattice, ax: f32, ay: f32, bx: f32, by: f32) -> bool {
  ${o}
}

fn isPointValid(lattice: Lattice, x: f32, y: f32) -> bool {
  return lattice.valid && isFiniteFloat(x) && isFiniteFloat(y) &&
    x >= lattice.minimumX && x <= lattice.maximumX &&
    y >= lattice.minimumY && y <= lattice.maximumY;
}

fn getCellColumn(lattice: Lattice, x: f32) -> u32 {
  return min(u32(floor((x - lattice.minimumX) / lattice.cellWidth)), lattice.columns - 1u);
}

fn getCellRow(lattice: Lattice, y: f32) -> u32 {
  return min(u32(floor((y - lattice.minimumY) / lattice.cellHeight)), lattice.rows - 1u);
}
`}function D(e,t=`true`,n=!1){let r=`let cell = cellRow * COLUMNS + cellColumn;
      let start = cellOffsets[cellOffsetsOffset + cell];
      let end = cellOffsets[cellOffsetsOffset + cell + 1u];
      for (var slot = start; slot < end && ${t}; slot++) {
        if (isWithinEpsilon(lattice, x, y, cellPositions[cellPositionsOffset + slot * 2u],
            cellPositions[cellPositionsOffset + slot * 2u + 1u])${n?` &&
            isWithinTime(lattice, time, cellTimes[cellTimesOffset + slot])`:``}) {
          let neighbor = sortedRows[sortedRowsOffset + slot];
          ${e}
        }
      }`;return`
  let column = getCellColumn(lattice, x);
  let row = getCellRow(lattice, y);
  let firstColumn = max(column, 1u) - 1u;
  let lastColumn = min(column + 1u, lattice.columns - 1u);
  let firstRow = max(row, 1u) - 1u;
  let lastRow = min(row + 1u, lattice.rows - 1u);
  for (var cellRow = firstRow; cellRow <= lastRow && ${t}; cellRow++) {
    for (var cellColumn = firstColumn; cellColumn <= lastColumn && ${t}; cellColumn++) {
      ${r}
    }
  }
  if (GEOGRAPHIC) {
    // Longitude is periodic: a neighbor across +-180 lies within one cell width of the point's
    // image at x - 360 or x + 360. Each image that reaches the bounds adds the columns its
    // interval covers, less the columns already scanned, so no point is visited twice. The slack
    // covers the rounding of x +- 360; a wider interval only adds candidates, which the haversine
    // test (periodic in longitude by itself) still decides.
    let imageSlack = 360.0 * 0x1p-20;
    var otherFirst = 1u;
    var otherLast = 0u;
    for (var wrapSide = 0u; wrapSide < 2u && ${t}; wrapSide++) {
      let image = x + select(-360.0, 360.0, wrapSide == 1u);
      let imageLow = image - lattice.cellWidth - imageSlack;
      let imageHigh = image + lattice.cellWidth + imageSlack;
      if (imageHigh < lattice.minimumX || imageLow > lattice.maximumX) {
        continue;
      }
      let wrappedFirst = getCellColumn(lattice, max(imageLow, lattice.minimumX));
      let wrappedLast = getCellColumn(lattice, min(imageHigh, lattice.maximumX));
      for (var cellRow = firstRow; cellRow <= lastRow && ${t}; cellRow++) {
        for (var cellColumn = wrappedFirst; cellColumn <= wrappedLast && ${t}; cellColumn++) {
          if ((cellColumn >= firstColumn && cellColumn <= lastColumn) ||
              (cellColumn >= otherFirst && cellColumn <= otherLast)) {
            continue;
          }
          ${r}
        }
      }
      otherFirst = wrappedFirst;
      otherLast = wrappedLast;
    }
  }`}var O=`if (index >= gridCount[gridCountOffset]) {
    return;
  }
  let point = sortedRows[sortedRowsOffset + index];
  let x = cellPositions[cellPositionsOffset + index * 2u];
  let y = cellPositions[cellPositionsOffset + index * 2u + 1u];`;function k(e){return e?`${O}
  let time = cellTimes[cellTimesOffset + index];`:O}function A(e){let t=e.cellTimes?[{name:`cellTimes`,view:e.cellTimes,type:`f32`,access:`read`}]:[];return[{name:`cellPositions`,view:e.cellPositions,type:`f32`,access:`read`},{name:`parameters`,view:e.parameters,type:`f32`,access:`read`},{name:`sortedRows`,view:e.sortedRows,type:`u32`,access:`read`},{name:`cellOffsets`,view:e.cellOffsets,type:`u32`,access:`read`},{name:`gridCount`,view:e.gridCount,type:`u32`,access:`read`},...t]}function ne(e,t){return a(e,{id:t.id,operation:T,variant:`grid-positions`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`gridPositions`,view:t.gridPositions,type:`f32`,access:`read_write`},{name:`rowIds`,view:t.rowIds,type:`u32`,access:`read_write`}],invocationCount:t.positions.length,declarations:`${E(t.gridSize[0],t.gridSize[1],t.geographicRadius)}
${y}`,body:`let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let valid = isPointValid(lattice, x, y);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  let unit = getGridUnitPosition(lattice, x, y);
  gridPositions[gridPositionsOffset + index * 2u] = select(invalid, unit.x, valid);
  gridPositions[gridPositionsOffset + index * 2u + 1u] = select(invalid, unit.y, valid);
  rowIds[rowIdsOffset + index] = index;`})}function j(e,t){let n=t.variant===`cell-positions`,r=t.variant===`cell-core-flags`?`u32`:`f32`,i=n?`cellValues[cellValuesOffset + index * 2u] = values[valuesOffset + row * 2u];
  cellValues[cellValuesOffset + index * 2u + 1u] = values[valuesOffset + row * 2u + 1u];`:`cellValues[cellValuesOffset + index] = values[valuesOffset + row];`;return a(e,{id:t.id,operation:T,variant:t.variant,bindings:[{name:`values`,view:t.values,type:r,access:`read`},{name:`sortedRows`,view:t.sortedRows,type:`u32`,access:`read`},{name:`gridCount`,view:t.gridCount,type:`u32`,access:`read`},{name:`cellValues`,view:t.cellValues,type:r,access:`read_write`}],invocationCount:t.sortedRows.length,body:`if (index >= gridCount[gridCountOffset]) {
    return;
  }
  let row = sortedRows[sortedRowsOffset + index];
  ${i}`})}function re(e,t){let{boxCounts:n}=t,r=!!t.cellTimes;return a(e,{id:t.id,operation:T,variant:`${n?`core-dense-box`:`core`}${r?`-time`:``}`,bindings:[...A(t),{name:`coreFlags`,view:t.coreFlags,type:`u32`,access:`read_write`},...n?[{name:`boxCounts`,view:n,type:`u32`,access:`read`}]:[]],invocationCount:t.cellPositions.length,declarations:E(t.gridSize[0],t.gridSize[1],t.geographicRadius,r),body:`${k(r)}
  let lattice = readLattice();
  var isCore = 0u;
  if (isPointValid(lattice, x, y)) {
    ${n?`if (boxCounts[boxCountsOffset + point] >= lattice.minimumPoints) {
      isCore = 1u;
    } else {`:`{`}
      var neighborCount = 0u;
      ${D(`neighborCount++;`,`neighborCount < lattice.minimumPoints`,r)}
      isCore = select(0u, 1u, neighborCount >= lattice.minimumPoints);
    }
  }
  coreFlags[coreFlagsOffset + point] = isCore;`})}function M(e,t){return a(e,{id:t.id,operation:T,variant:`box-keys`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`boxHigh`,view:t.boxHigh,type:`u32`,access:`read_write`},{name:`boxLow`,view:t.boxLow,type:`u32`,access:`read_write`}],invocationCount:t.positions.length,declarations:`${E(t.gridSize[0],t.gridSize[1],t.geographicRadius)}
const BOX_SIDE_FACTOR: f32 = 0.7;
const MAXIMUM_BOX_INDEX: f32 = 65536.0;`,body:`let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var high = 0xffffffffu;
  var low = 0xffffffffu;
  if (isPointValid(lattice, x, y)) {
    let side = parameters[parametersOffset + 4u] * BOX_SIDE_FACTOR;
    let boxX = floor((x - lattice.minimumX) / side);
    let boxY = floor((y - lattice.minimumY) / side);
    if (boxX >= 0.0 && boxX < MAXIMUM_BOX_INDEX && boxY >= 0.0 && boxY < MAXIMUM_BOX_INDEX) {
      high = u32(boxX);
      low = u32(boxY);
    }
  }
  boxHigh[boxHighOffset + index] = high;
  boxLow[boxLowOffset + index] = low;`})}function N(e,t){return a(e,{id:t.id,operation:T,variant:`box-counts`,bindings:[{name:`sortedItems`,view:t.sortedItems,type:`u32`,access:`read`},{name:`groupIndex`,view:t.groupIndex,type:`u32`,access:`read`},{name:`groupStarts`,view:t.groupStarts,type:`u32`,access:`read`},{name:`boxHigh`,view:t.boxHigh,type:`u32`,access:`read`},{name:`boxCounts`,view:t.boxCounts,type:`u32`,access:`read_write`}],invocationCount:t.sortedItems.length,body:`let item = sortedItems[sortedItemsOffset + index];
  let group = groupIndex[groupIndexOffset + index] - 1u;
  let count = groupStarts[groupStartsOffset + group + 1u] - groupStarts[groupStartsOffset + group];
  boxCounts[boxCountsOffset + item] = select(0u, count, boxHigh[boxHighOffset + item] != 0xffffffffu);`})}function P(e,t){return a(e,{id:t.id,operation:T,variant:`parents-init`,bindings:[{name:`parents`,view:t.parents,type:`u32`,access:`read_write`}],invocationCount:t.parents.length,body:`parents[parentsOffset + index] = index;`})}function F(e,t){let n=!!t.cellTimes;return a(e,{id:t.id,operation:T,variant:n?`union-time`:`union`,bindings:[...A(t),{name:`cellCoreFlags`,view:t.cellCoreFlags,type:`u32`,access:`read`},{name:`parents`,view:t.parents,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.cellPositions.length,declarations:`${E(t.gridSize[0],t.gridSize[1],t.geographicRadius,n)}
// Path halving: a visited node is a non-root forever, so any ancestor is a valid parent for it.
fn findRoot(start: u32) -> u32 {
  var current = start;
  loop {
    let parent = atomicLoad(&parents[parentsOffset + current]);
    if (parent == current) {
      return current;
    }
    let grandparent = atomicLoad(&parents[parentsOffset + parent]);
    if (grandparent != parent) {
      atomicStore(&parents[parentsOffset + current], grandparent);
    }
    current = parent;
  }
}

// Hooks the larger root under the smaller one; retries with fresh roots when another invocation
// hooked the larger root first (or the weak compare-exchange failed spuriously).
fn unite(first: u32, second: u32) {
  var a = findRoot(first);
  var b = findRoot(second);
  loop {
    if (a == b) {
      return;
    }
    let low = min(a, b);
    let high = max(a, b);
    let result = atomicCompareExchangeWeak(&parents[parentsOffset + high], high, low);
    if (result.exchanged) {
      return;
    }
    a = findRoot(low);
    b = findRoot(high);
  }
}`,body:`${k(n)}
  if (cellCoreFlags[cellCoreFlagsOffset + index] == 0u) {
    return;
  }
  let lattice = readLattice();
  ${D(`if (neighbor < point && cellCoreFlags[cellCoreFlagsOffset + slot] != 0u) {
            unite(point, neighbor);
          }`,`true`,n)}`})}function I(e,t){return a(e,{id:t.id,operation:T,variant:`resolve`,bindings:[{name:`coreFlags`,view:t.coreFlags,type:`u32`,access:`read`},{name:`parents`,view:t.parents,type:`u32`,access:`read`},{name:`rootLabels`,view:t.rootLabels,type:`u32`,access:`read_write`}],invocationCount:t.rootLabels.length,body:`var root = 0xffffffffu;
  if (coreFlags[coreFlagsOffset + index] != 0u) {
    root = index;
    loop {
      let parent = parents[parentsOffset + root];
      if (parent == root) {
        break;
      }
      root = parent;
    }
  }
  rootLabels[rootLabelsOffset + index] = root;`})}function L(e,t){let n=!!t.cellTimes;return a(e,{id:t.id,operation:T,variant:n?`border-time`:`border`,bindings:[...A(t),{name:`cellCoreFlags`,view:t.cellCoreFlags,type:`u32`,access:`read`},{name:`rootLabels`,view:t.rootLabels,type:`u32`,access:`read_write`}],invocationCount:t.cellPositions.length,declarations:E(t.gridSize[0],t.gridSize[1],t.geographicRadius,n),body:`${k(n)}
  if (cellCoreFlags[cellCoreFlagsOffset + index] != 0u) {
    return;
  }
  let lattice = readLattice();
  if (!isPointValid(lattice, x, y)) {
    return;
  }
  var bestRoot = NOISE;
  ${D(`if (cellCoreFlags[cellCoreFlagsOffset + slot] != 0u) {
            bestRoot = min(bestRoot, rootLabels[rootLabelsOffset + neighbor]);
          }`,`true`,n)}
  rootLabels[rootLabelsOffset + point] = bestRoot;`})}function R(e,t){return a(e,{id:t.id,operation:T,variant:`root-flags`,bindings:[{name:`coreFlags`,view:t.coreFlags,type:`u32`,access:`read`},{name:`rootLabels`,view:t.rootLabels,type:`u32`,access:`read`},{name:`rootFlags`,view:t.rootFlags,type:`u32`,access:`read_write`}],invocationCount:t.rootLabels.length,body:`let isRoot = coreFlags[coreFlagsOffset + index] != 0u &&
    rootLabels[rootLabelsOffset + index] == index;
  rootFlags[rootFlagsOffset + index] = select(0u, 1u, isRoot);`})}function z(e,t){let n=[{name:`rootLabels`,view:t.rootLabels,type:`u32`,access:`read`},{name:`clusterOffsets`,view:t.clusterOffsets,type:`u32`,access:`read`},{name:`labels`,view:t.labels,type:`u32`,access:`read_write`}],r=t.memberXs&&t.memberYs;return t.memberXs&&t.memberYs&&n.push({name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`memberXs`,view:t.memberXs,type:`f32`,access:`read_write`},{name:`memberYs`,view:t.memberYs,type:`f32`,access:`read_write`}),a(e,{id:t.id,operation:T,variant:`labels`,bindings:n,invocationCount:t.rootLabels.length,body:`let root = rootLabels[rootLabelsOffset + index];
  labels[labelsOffset + index] = select(clusterOffsets[clusterOffsetsOffset + min(root, INVOCATION_COUNT - 1u)], 0xffffffffu, root == 0xffffffffu);
  ${r?`memberXs[memberXsOffset + index] = positions[positionsOffset + index * 2u];
  memberYs[memberYsOffset + index] = positions[positionsOffset + index * 2u + 1u];`:``}`})}function B(e,t){return a(e,{id:t.id,operation:T,variant:`centroids`,bindings:[{name:`sizes`,view:t.sizes,type:`u32`,access:`read`},{name:`sumsX`,view:t.sumsX,type:`f32`,access:`read`},{name:`sumsY`,view:t.sumsY,type:`f32`,access:`read`},{name:`centroids`,view:t.centroids,type:`f32`,access:`read_write`}],invocationCount:t.sizes.length,body:`let size = sizes[sizesOffset + index];
  let divisor = f32(max(size, 1u));
  centroids[centroidsOffset + index * 2u] = select(0.0, sumsX[sumsXOffset + index] / divisor, size > 0u);
  centroids[centroidsOffset + index * 2u + 1u] = select(0.0, sumsY[sumsYOffset + index] / divisor, size > 0u);`})}var V=Object.freeze({...S,measure:`distance`}),H=`GPUSpatialClustering`,U=Object.freeze({encodings:Object.freeze([`float32-relative`])}),W=2**24-1,G=class{id;props;spatialContext;temporalContext;geographicRadius;constructor(e){this.id=e.id??`spatial-clustering`,this.props=e;let t=this.id;if(this.spatialContext=w(t,e.spatialContext,V),this.spatialContext.coordinateSpace===`longitude-latitude`&&this.spatialContext.metric!==`great-circle`)throw Error(`${t} longitude-latitude positions need the great-circle metric`);if(this.geographicRadius=this.spatialContext.coordinateSpace===`longitude-latitude`?C(this.spatialContext):void 0,this.geographicRadius!==void 0&&e.denseBoxShortcut)throw Error(`${t} denseBoxShortcut is planar only`);let n=e.positions.length;if(e.times){if(!e.temporalContext)throw Error(`${t} times need a temporalContext`);if(this.temporalContext=x(t,e.temporalContext,U),s(e.times,[`float32`],`${t} times`),e.times.length!==n)throw Error(`${t} times length must equal positions length`);if(e.denseBoxShortcut)throw Error(`${t} denseBoxShortcut cannot be combined with times`)}else if(e.temporalContext)throw Error(`${t} temporalContext needs times`);if(e.sumOrder!==void 0&&e.sumOrder!==`atomic`&&e.sumOrder!==`sorted`)throw Error(`${t} sumOrder must be atomic or sorted`);if(s(e.positions,[`float32x2`],`${t} positions`),s(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<8)throw Error(`${t} parameters must hold 8 float32 values`);let[r,i]=e.gridSize;if(!Number.isInteger(r)||!Number.isInteger(i)||r<1||i<1)throw Error(`${t} gridSize must be two positive integers`);for(let[r,i]of[[`sourceIds`,e.sourceIds],[`output.labels`,e.output.labels],[`output.rootRows`,e.output.rootRows],[`output.coreFlags`,e.output.coreFlags]])if(i&&(s(i,[`uint32`],`${t} ${r}`),i.length!==n))throw Error(`${t} ${r} length must equal positions length`);if(e.output.clusterCount&&(s(e.output.clusterCount,[`uint32`],`${t} output.clusterCount`),e.output.clusterCount.length<1))throw Error(`${t} output.clusterCount must contain one uint32 row`);if(e.output.clusters&&(l(t,e.output.clusters),e.output.clusters.ids.length<1))throw Error(`${t} output.clusters.ids must hold at least one row`);if(e.output.drawInstanceCount){if(s(e.output.drawInstanceCount,[`uint32`],`${t} output.drawInstanceCount`),e.output.drawInstanceCount.length<1)throw Error(`${t} output.drawInstanceCount must contain one uint32 row`);if(!e.output.clusters)throw Error(`${t} output.drawInstanceCount requires clusters`)}if((e.output.clusterSizes||e.output.clusterCentroids)&&!e.output.clusters)throw Error(`${t} output.clusterSizes and clusterCentroids require clusters`);let a=e.output.clusters?.ids.length??0;if(e.output.clusterSizes&&(s(e.output.clusterSizes,[`uint32`],`${t} output.clusterSizes`),e.output.clusterSizes.length!==a))throw Error(`${t} output.clusterSizes length must equal clusters.ids length`);if(e.output.clusterCentroids){if(s(e.output.clusterCentroids,[`float32x2`],`${t} output.clusterCentroids`),e.output.clusterCentroids.length!==a)throw Error(`${t} output.clusterCentroids length must equal clusters.ids length`);e.sumOrder!==`atomic`&&c(t,`clusters.ids rows with sorted centroid sums`,a,W)}m(t,{labels:e.output.labels,rootRows:e.output.rootRows,coreFlags:e.output.coreFlags,clusterCount:e.output.clusterCount,clusterIds:n>0?e.output.clusters?.ids:void 0,count:e.output.clusters?.status.count,overflow:e.output.clusters?.status.overflow,requiredCount:e.output.clusters?.status.requiredCount,clusterSizes:e.output.clusterSizes,clusterCentroids:e.output.clusterCentroids,drawInstanceCount:e.output.clusters?e.output.drawInstanceCount:void 0},[...n>0?[{read:[e.output.coreFlags],write:[e.output.rootRows]},{read:[e.output.rootRows],write:[e.output.labels]},...e.output.clusters?[{read:[e.output.labels],write:[e.output.clusterSizes]},{read:[e.output.clusterSizes],write:[e.output.clusterCentroids]}]:[]]:[],...e.output.clusters?[{read:[e.output.clusterCount],write:[n>0?e.output.clusters.ids:void 0,e.output.clusters.status.count,e.output.clusters.status.overflow,e.output.clusters.status.requiredCount,e.output.drawInstanceCount]}]:[]]),f(t,[e.output.labels,e.output.rootRows,e.output.coreFlags,e.output.clusterCount,e.output.clusters?.ids,e.output.clusters?.status.count,e.output.clusters?.status.overflow,e.output.clusters?.status.requiredCount,e.output.clusterSizes,e.output.clusterCentroids,e.output.drawInstanceCount],n>0?[e.positions,e.parameters,e.sourceIds,e.times]:[])}_getBoxCountNodes(t,n){let{id:r,props:i}=this,a=i.positions.length,o=e(t,`${r}-box-high`,`uint32`,a),s=e(t,`${r}-box-low`,`uint32`,a),c=h(t,`${r}-box`,H,a,o,s),l=g(t,`${r}-box`,H,a,o,s,c.sortedItems);return[M(t,{id:`${r}-box-keys`,positions:i.positions,parameters:i.parameters,gridSize:i.gridSize,boxHigh:o,boxLow:s}),...c.nodes,...l.nodes,N(t,{id:`${r}-box-counts`,sortedItems:c.sortedItems,groupIndex:l.groupIndex,groupStarts:l.groupStarts,boxHigh:o,boxCounts:n})]}getCommandNodes(a){let{props:s,id:l}=this,{positions:f,parameters:m,gridSize:h}=s,{clusters:g}=s.output,_=this.geographicRadius;ee(l,a,[f,m,s.times,s.sourceIds,s.output.labels,s.output.rootRows,s.output.coreFlags,s.output.clusterCount,g?.ids,g?.status.count,g?.status.overflow,g?.status.requiredCount,s.output.clusterSizes,s.output.clusterCentroids,s.output.drawInstanceCount]);let v=h[0]*h[1];c(l,`gridSize cells`,v,u(a.device));let y=f.length,x=g?.ids.length??0;p(a,l,{...y>0?{positions:f,parameters:m,times:s.times,sourceIds:g?s.sourceIds:void 0,labels:s.output.labels,rootRows:s.output.rootRows,coreFlags:s.output.coreFlags}:{},clusterCount:s.output.clusterCount,...g?{clusterIds:y>0?g.ids:void 0,count:g.status.count,overflow:g.status.overflow,requiredCount:g.status.requiredCount,clusterSizes:s.output.clusterSizes,clusterCentroids:s.output.clusterCentroids,drawInstanceCount:s.output.drawInstanceCount}:{}});for(let[e,t]of[[`point scratch`,y*8],[`grid offsets`,y>0?(v+1)*4:0],[`centroid sums`,y>0&&s.output.clusterCentroids?x*4:0]])d(a.device,l,e,t,`reduce positions rows, gridSize or output capacity`);let S=!!s.output.clusterCentroids,C=[],w=s.output.clusterCount??e(a,`${l}-cluster-count`,`uint32`,1);if(y===0)return C.push(o(a,{id:`${l}-cluster-count`,operation:H,view:w,type:`u32`,value:`0u`,componentCount:1})),s.output.clusterSizes&&C.push(o(a,{id:`${l}-cluster-sizes`,operation:H,view:s.output.clusterSizes,type:`u32`,value:`0u`})),s.output.clusterCentroids&&C.push(o(a,{id:`${l}-cluster-centroids`,operation:H,view:s.output.clusterCentroids,type:`f32`,value:`0.0`,componentCount:x*2})),g&&C.push(i(a,{id:`${l}-publish`,operation:H,requiredCount:w,output:g,extraCounts:s.output.drawInstanceCount?[s.output.drawInstanceCount]:[]})),C;let T=e(a,`${l}-grid-positions`,`float32x2`,y),E=e(a,`${l}-row-ids`,`uint32`,y),D=te(a,{id:l,positions:T,gridSize:[h[0],h[1]],rowCount:y}),{cellOffsets:O,sortedRows:k,gridCount:A}=D,M=e(a,`${l}-cell-positions`,`float32x2`,y),N=e(a,`${l}-cell-core-flags`,`uint32`,y),V=s.times?e(a,`${l}-cell-times`,`float32`,y):void 0,U=s.output.coreFlags??e(a,`${l}-core-flags`,`uint32`,y),W=e(a,`${l}-parents`,`uint32`,y),G=s.output.rootRows??e(a,`${l}-root-labels`,`uint32`,y),K=e(a,`${l}-root-flags`,`uint32`,y),q=e(a,`${l}-cluster-offsets`,`uint32`,y),J=s.denseBoxShortcut?e(a,`${l}-box-counts`,`uint32`,y):void 0,Y={cellPositions:M,parameters:m,sortedRows:k,gridCount:A,cellOffsets:O,cellTimes:V};C.push(ne(a,{id:`${l}-grid-positions`,positions:f,parameters:m,gridSize:h,geographicRadius:_,gridPositions:T,rowIds:E}),...D.nodes,j(a,{id:`${l}-cell-positions`,variant:`cell-positions`,values:f,sortedRows:k,gridCount:A,cellValues:M}),...s.times&&V?[j(a,{id:`${l}-cell-times`,variant:`cell-times`,values:s.times,sortedRows:k,gridCount:A,cellValues:V})]:[],...J?this._getBoxCountNodes(a,J):[],o(a,{id:`${l}-core-flags-clear`,operation:`GPUSpatialClustering`,view:U,type:`u32`,value:`0u`}),re(a,{id:`${l}-core`,...Y,gridSize:h,geographicRadius:_,coreFlags:U,boxCounts:J}),j(a,{id:`${l}-cell-core-flags`,variant:`cell-core-flags`,values:U,sortedRows:k,gridCount:A,cellValues:N}),P(a,{id:`${l}-parents-init`,parents:W}),F(a,{id:`${l}-union`,...Y,gridSize:h,geographicRadius:_,cellCoreFlags:N,parents:W}),I(a,{id:`${l}-resolve`,coreFlags:U,parents:W,rootLabels:G}),L(a,{id:`${l}-border`,...Y,gridSize:h,geographicRadius:_,cellCoreFlags:N,rootLabels:G}),R(a,{id:`${l}-root-flags`,coreFlags:U,rootLabels:G,rootFlags:K}),...new n({id:`${l}-cluster-offsets`,flags:K,offsets:q,count:w}).getCommandNodes(a));let X=S?e(a,`${l}-member-xs`,`float32`,y):void 0,Z=S?e(a,`${l}-member-ys`,`float32`,y):void 0;if(C.push(z(a,{id:`${l}-labels`,positions:f,rootLabels:G,clusterOffsets:q,labels:s.output.labels,memberXs:X,memberYs:Z})),!g)return C;let Q=e(a,`${l}-root-ids`,`uint32`,y),ie=e(a,`${l}-compact-count`,`uint32`,1),ae=s.sourceIds??E;C.push(...new t({id:`${l}-cluster-roots`,input:ae,flags:K,output:Q,count:ie}).getCommandNodes(a));let $=s.output.clusterSizes??(S?e(a,`${l}-cluster-size-values`,`uint32`,x):void 0);if($&&C.push(...new r({id:`${l}-cluster-sizes`,keys:s.output.labels,output:$}).getCommandNodes(a)),s.output.clusterCentroids&&$&&X&&Z){let t=e(a,`${l}-cluster-sum-values-x`,`float32`,x),n=e(a,`${l}-cluster-sum-values-y`,`float32`,x);s.sumOrder===`atomic`?C.push(...new r({id:`${l}-cluster-sums-x`,keys:s.output.labels,values:X,output:t,operation:`sum`}).getCommandNodes(a),...new r({id:`${l}-cluster-sums-y`,keys:s.output.labels,values:Z,output:n,operation:`sum`}).getCommandNodes(a)):C.push(...b(a,{id:`${l}-cluster-sums`,operation:H,segmentCount:x,segmentKeys:s.output.labels,reductions:[{name:`x`,contributions:X,output:t},{name:`y`,contributions:Z,output:n}]})),C.push(B(a,{id:`${l}-cluster-centroids`,sizes:$,sumsX:t,sumsY:n,centroids:s.output.clusterCentroids}))}return C.push(i(a,{id:`${l}-publish`,operation:H,requiredCount:w,compactIds:Q,output:g,extraCounts:s.output.drawInstanceCount?[s.output.drawInstanceCount]:[]})),C}};export{G as t};
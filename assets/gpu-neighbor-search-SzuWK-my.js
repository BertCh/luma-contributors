import{n as e}from"./scan-total-C4QddMmi.js";import{i as t,r as n}from"./weights-construction-snapshot-CikLt8kl.js";import{t as r}from"./spatial-statistics-writable-dispatches-BC35SiUi.js";import{m as i,v as a}from"./spatial-context-DwZEd4EZ.js";import{r as o,s}from"./kernel-C3tfQOn4.js";import{t as c}from"./gpu-scan-B4NNTfF9.js";import{t as l}from"./common-CshkX-d_.js";import{c as u,o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as f,S as p,T as m,f as h,l as ee,o as te,v as ne,x as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{t as v}from"./geodesic-wgsl-I0CYDKcz.js";import{l as y}from"./wgsl-common-DfyXiYB0.js";import{t as b}from"./nearest-nodes-CMiGBIQD.js";import{n as x,t as re}from"./point-neighbor-grid-index-D0wS1y1I.js";import{l as S,t as C}from"./spatial-weights-CqWysstN.js";import{n as ie}from"./gpu-spatial-weights-transform-lEdsoyYI.js";import{n as ae}from"./temporal-context-xddVhAJD.js";function w(e,t,n){let r=t.invocationCount,i=l(r,1,n??r);if(i>=r)return[d(e,t)];let a=[];for(let n=0;n<r;n+=i)a.push(d(e,{...t,id:`${t.id}-tile-${n}`,invocationCount:Math.min(i,r-n),constants:{...t.constants,TILE_ROW_BASE:n},declarations:`override TILE_ROW_BASE: u32;\n${t.declarations??``}`,body:`{ let index = index + TILE_ROW_BASE;\n${t.body}\n}`}));return a}var T=y,E=`
// Reduce the stored f32 angle exactly modulo a turn. Floating quotient/product reduction can
// erase the remainder for large angles (the stored f32 near 1e20 is -88 degrees, not zero).
// For fractional angles, use integer significand units. For integer angles, modular powers of
// two need at most seven iterations even at the largest finite f32 exponent.
fn getNeighborSearchCanonicalLongitude(longitude: f32) -> f32 {
  if (abs(longitude) <= 180.0) { return longitude; }
  let bits = bitcast<u32>(longitude);
  let exponent = (bits >> 23u) & 255u;
  let significand = (bits & 0x7fffffu) | 0x800000u;
  var wrapped: f32;
  if (exponent < 150u) {
    let fractionalBits = 150u - exponent;
    let turnUnits = 360u << fractionalBits;
    let remainder = significand % turnUnits;
    let centered = select(i32(remainder), i32(remainder) - i32(turnUnits), remainder > turnUnits / 2u);
    wrapped = f32(centered) * bitcast<f32>((127u - fractionalBits) << 23u);
  } else {
    var remainingExponent = exponent - 150u;
    var power = 2u;
    var remainder = significand % 360u;
    while (remainingExponent > 0u) {
      if ((remainingExponent & 1u) != 0u) { remainder = (remainder * power) % 360u; }
      power = (power * power) % 360u;
      remainingExponent >>= 1u;
    }
    wrapped = f32(select(i32(remainder), i32(remainder) - 360, remainder > 180u));
  }
  return select(wrapped, -wrapped, (bits & 0x80000000u) != 0u);
}
`;function D(e){if(!e)return{preamble:``,open:``,close:``,condition:``,rowBase:e=>`${e} * COLUMNS`};let t=` && abs(bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 3u]) - queryTime) <= lattice.timeRadius`;return e.bucketCount<=1?{preamble:``,open:``,close:``,condition:t,rowBase:e=>`${e} * COLUMNS`}:{preamble:`let timeSlack = getTimeBucketSlack(lattice);
  let firstTimeBucket = getTimeBucket(lattice, queryTime - lattice.timeRadius - timeSlack);
  let lastTimeBucket = getTimeBucket(lattice, queryTime + lattice.timeRadius + timeSlack);`,open:`for (var timeBucket = firstTimeBucket; timeBucket <= lastTimeBucket; timeBucket++) {`,close:`}`,condition:t,rowBase:e=>`(timeBucket * ROWS + ${e}) * COLUMNS`}}function oe(e,t,n,r){let i=n!==void 0,a=r?.bucketCount??1,o=i?e===`radius`?`let radiusDegrees = min(degrees(radius / SPHERE_RADIUS) * CELL_MARGIN + 0.0001, 360.0);
  lattice.valid = lattice.valid && isFiniteFloat(radius) && isFiniteFloat(radiusDegrees) && radius > 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.radiusDegrees = min(radiusDegrees, 180.0);
  lattice.cellWidth = max(width / f32(COLUMNS), radiusDegrees);
  lattice.cellHeight = max(height / f32(ROWS), radiusDegrees);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.bounded = true;`:`lattice.cellWidth = select(1.0, width / f32(COLUMNS), width > 0.0);
  lattice.cellHeight = select(1.0, height / f32(ROWS), height > 0.0);
  lattice.columns = COLUMNS;
  lattice.rows = ROWS;
  lattice.bounded = isFiniteFloat(radius) && radius > 0.0;`:e===`radius`?`let cellRadius = radius * CELL_MARGIN;
  lattice.valid = lattice.valid && isFiniteFloat(radius) && isFiniteFloat(radiusSquared) &&
    isFiniteFloat(cellRadius) && radius > 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.cellWidth = max(width / f32(COLUMNS), cellRadius);
  lattice.cellHeight = max(height / f32(ROWS), cellRadius);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.bounded = true;`:`lattice.cellWidth = select(1.0, width / f32(COLUMNS), width > 0.0);
  lattice.cellHeight = select(1.0, height / f32(ROWS), height > 0.0);
  lattice.columns = COLUMNS;
  lattice.rows = ROWS;
  lattice.bounded = isFiniteFloat(radius) && isFiniteFloat(radiusSquared) && radius > 0.0;`;return`
const COLUMNS: u32 = ${t[0]}u;
const ROWS: u32 = ${t[1]}u;
const CELL_COUNT: u32 = ${t[0]*t[1]}u;
const CELL_MARGIN: f32 = 1.0009765625;
const TARGET_WORDS: u32 = ${r?4:3}u;
${r?`const TIME_BUCKETS: u32 = ${a}u;`:``}
${i?`const SPHERE_RADIUS: f32 = ${Number(n).toPrecision(17)};`:``}

struct Lattice {
  valid: bool,
  bounded: bool,
  minimumX: f32,
  minimumY: f32,
  maximumX: f32,
  maximumY: f32,
  cellWidth: f32,
  cellHeight: f32,
  columns: u32,
  rows: u32,
  radius: f32,
  radiusSquared: f32,
  radiusDegrees: f32,
  periodic: bool${r?`,
  timeRadius: f32,
  minimumTime: f32,
  maximumTime: f32,
  timeBucketWidth: f32`:``}
}

fn readParameter(slot: u32) -> f32 {
  return parameters[parametersOffset + slot];
}

fn readLattice() -> Lattice {
  var lattice: Lattice;
  lattice.cellWidth = 1.0;
  lattice.cellHeight = 1.0;
  lattice.columns = 1u;
  lattice.rows = 1u;
  lattice.bounded = false;
  lattice.radiusDegrees = 0.0;
  lattice.periodic = false;
  let minimumX = readParameter(0u);
  let minimumY = readParameter(1u);
  let maximumX = readParameter(2u);
  let maximumY = readParameter(3u);
  let radius = readParameter(4u);
  let width = maximumX - minimumX;
  let height = maximumY - minimumY;
  let radiusSquared = radius * radius;
  lattice.minimumX = minimumX;
  lattice.minimumY = minimumY;
  lattice.maximumX = maximumX;
  lattice.maximumY = maximumY;
  lattice.radius = radius;
  lattice.radiusSquared = radiusSquared;
  lattice.valid =
    isFiniteFloat(minimumX) && isFiniteFloat(minimumY) && isFiniteFloat(maximumX) &&
    isFiniteFloat(maximumY) && isFiniteFloat(width) && isFiniteFloat(height) &&
    width >= 0.0 && height >= 0.0;
  ${r?`let timeRadius = readParameter(12u);
  let minimumTime = readParameter(13u);
  let maximumTime = readParameter(14u);
  let timeSpan = maximumTime - minimumTime;
  lattice.timeRadius = timeRadius;
  lattice.minimumTime = minimumTime;
  lattice.maximumTime = maximumTime;
  lattice.timeBucketWidth = select(1.0, timeSpan / f32(TIME_BUCKETS), timeSpan > 0.0);
  lattice.valid = lattice.valid && isFiniteFloat(timeRadius) && timeRadius > 0.0 &&
    isFiniteFloat(minimumTime) && isFiniteFloat(maximumTime) && isFiniteFloat(timeSpan) &&
    timeSpan >= 0.0;`:``}
  ${i?`lattice.valid = lattice.valid && width <= 360.001;
  lattice.periodic = lattice.valid && abs(width - 360.0) <= 0.001;`:``}
  if (!lattice.valid) {
    return lattice;
  }
  ${o}
  return lattice;
}

// Finite and inside the bounds, independent of whether the radius or bounds form a usable lattice.
fn isPointInBounds(lattice: Lattice, x: f32, y: f32) -> bool {
  return isFiniteFloat(x) && isFiniteFloat(y) &&
    x >= lattice.minimumX && x <= lattice.maximumX &&
    y >= lattice.minimumY && y <= lattice.maximumY${i?` && abs(y) <= 90.0`:``};
}

fn isPointValid(lattice: Lattice, x: f32, y: f32) -> bool {
  return lattice.valid && isPointInBounds(lattice, x, y);
}

// Cross-join queries are independent of the target bounds. Great-circle queries also need a
// latitude on the sphere; their longitude is wrapped near the lattice for lookup and distance.
fn isCrossJoinQueryInBounds(lattice: Lattice, x: f32, y: f32) -> bool {
  return isFiniteFloat(x) && isFiniteFloat(y)${i?` && abs(y) <= 90.0`:``};
}

fn isCrossJoinQueryValid(lattice: Lattice, x: f32, y: f32) -> bool {
  return lattice.valid && isCrossJoinQueryInBounds(lattice, x, y);
}
${r?`
// Finite and inside the per-frame time bounds, independent of whether the lattice is usable.
fn isTimeInBounds(lattice: Lattice, time: f32) -> bool {
  return isFiniteFloat(time) && time >= lattice.minimumTime && time <= lattice.maximumTime;
}

// A cross-join query time only needs to be finite: the bucket range clamps to the buckets that
// hold targets, and the window test itself is exact.
fn isCrossJoinQueryTimeInBounds(time: f32) -> bool {
  return isFiniteFloat(time);
}

// A rounding margin for bucket lookups: a target whose bucket rounds one way and a query whose
// range rounds the other must still meet.
fn getTimeBucketSlack(lattice: Lattice) -> f32 {
  return lattice.timeBucketWidth * 0.0009765625 +
    (abs(lattice.minimumTime) + abs(lattice.maximumTime)) * 0.00000095367431640625;
}

fn getTimeBucket(lattice: Lattice, time: f32) -> u32 {
  let bucket = floor((time - lattice.minimumTime) / lattice.timeBucketWidth);
  return u32(clamp(bucket, 0.0, f32(TIME_BUCKETS - 1u)));
}
`:``}

// Both clamp a position outside the lattice to the nearest edge cell.
fn getCellColumn(lattice: Lattice, x: f32) -> u32 {
  return u32(clamp(floor((x - lattice.minimumX) / lattice.cellWidth), 0.0, f32(lattice.columns - 1u)));
}

fn getCellRow(lattice: Lattice, y: f32) -> u32 {
  return u32(clamp(floor((y - lattice.minimumY) / lattice.cellHeight), 0.0, f32(lattice.rows - 1u)));
}

// Distance along one axis from position (in cell queryCell) to the near edge of cell; zero
// for the query cell itself. A lower bound of the distance to any point of that cell, up to the
// rounding slack the callers subtract.
fn getAxisGap(position: f32, minimum: f32, cellSize: f32, queryCell: i32, cell: i32) -> f32 {
  if (cell < queryCell) {
    return position - (minimum + f32(cell + 1) * cellSize);
  }
  if (cell > queryCell) {
    return (minimum + f32(cell) * cellSize) - position;
  }
  return 0.0;
}

${T}
${i?v+O:``}
`}function se(e,t){let n=D(t);return`
  ${n.preamble}
  let column = getCellColumn(lattice, x);
  let row = getCellRow(lattice, y);
  let firstColumn = max(column, 1u) - 1u;
  let lastColumn = min(column + 1u, lattice.columns - 1u);
  let firstRow = max(row, 1u) - 1u;
  let lastRow = min(row + 1u, lattice.rows - 1u);
  if (lattice.cellWidth > lattice.radius * 2.0 || lattice.cellHeight > lattice.radius * 2.0) {
    // Coarse grids can put many irrelevant targets in the neighboring cells. Test each cell's
    // distance lower bound before touching its target range.
    let slack = (lattice.cellWidth + lattice.cellHeight) * 0.0009765625 +
      (abs(lattice.minimumX) + abs(lattice.maximumX) + abs(lattice.minimumY) + abs(lattice.maximumY)) * 0.00000095367431640625;
    ${n.open}
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowGap = max(getAxisGap(y, lattice.minimumY, lattice.cellHeight, i32(row), i32(cellRow)) - slack, 0.0);
      let rowGapSquared = rowGap * rowGap;
      let rowBase = ${n.rowBase(`cellRow`)};
      for (var cellColumn = firstColumn; cellColumn <= lastColumn; cellColumn++) {
        let columnGap = max(getAxisGap(x, lattice.minimumX, lattice.cellWidth, i32(column), i32(cellColumn)) - slack, 0.0);
        if (columnGap * columnGap + rowGapSquared <= lattice.radiusSquared) {
          let cell = rowBase + cellColumn;
          let cellBegin = cellOffsets[cellOffsetsOffset + cell];
          let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
          for (var cellSlot = cellBegin; cellSlot < cellEnd; cellSlot++) {
            let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 2u];
            let deltaX = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS]) - x;
            let deltaY = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 1u]) - y;
            let distanceSquared = deltaX * deltaX + deltaY * deltaY;
            if (distanceSquared <= lattice.radiusSquared${n.condition}) {
              ${e}
            }
          }
        }
      }
    }
    ${n.close}
  } else {
    // Fine grids already have radius-sized cells. Preserve contiguous reads across each row.
    ${n.open}
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowBase = ${n.rowBase(`cellRow`)};
      let cellBegin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let cellEnd = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      for (var cellSlot = cellBegin; cellSlot < cellEnd; cellSlot++) {
        let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 2u];
        let deltaX = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS]) - x;
        let deltaY = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 1u]) - y;
        let distanceSquared = deltaX * deltaX + deltaY * deltaY;
        if (distanceSquared <= lattice.radiusSquared${n.condition}) {
          ${e}
        }
      }
    }
    ${n.close}
  }`}function ce(e,t){let n=D(t),r=(t,r)=>`for (var cellSlot = ${t}; cellSlot < ${r}; cellSlot++) {
          let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 2u];
          if (${e}) {
            let deltaX = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS]) - x;
            let deltaY = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 1u]) - y;
            let distanceSquared = deltaX * deltaX + deltaY * deltaY;
            if ((!lattice.bounded || distanceSquared <= lattice.radiusSquared)${n.condition}) {
              insertNeighbor(&bestDistances, &bestIds, &found, distanceSquared, neighbor);
            }
          }
        }`;return`
  ${n.preamble}
  let queryColumn = i32(getCellColumn(lattice, x));
  let queryRow = i32(getCellRow(lattice, y));
  let lastColumn = i32(lattice.columns) - 1;
  let lastRow = i32(lattice.rows) - 1;
  let maximumRing = max(lattice.columns, lattice.rows);
  let slack = (lattice.cellWidth + lattice.cellHeight) * 0.0009765625 +
    (abs(lattice.minimumX) + abs(lattice.maximumX) + abs(lattice.minimumY) + abs(lattice.maximumY)) * 0.00000095367431640625;
  for (var ringIndex = 0u; ringIndex < maximumRing; ringIndex++) {
    let ring = i32(ringIndex);
    let firstColumn = max(queryColumn - ring, 0);
    let endColumn = min(queryColumn + ring, lastColumn);
    let firstRow = max(queryRow - ring, 0);
    let endRow = min(queryRow + ring, lastRow);
    ${n.open}
    for (var cellRow = firstRow; cellRow <= endRow; cellRow++) {
      let rowBase = ${n.rowBase(`u32(cellRow)`)};
      // Cell-level pruning once K candidates are held: a cell whose near edges are farther than
      // the k-th best cannot contribute (strict test, so k-th distance ties are still visited).
      let rowGap = max(getAxisGap(y, lattice.minimumY, lattice.cellHeight, queryRow, cellRow) - slack, 0.0);
      let rowGapSquared = rowGap * rowGap;
      if (found == K && bestDistances[K - 1u] < rowGapSquared) {
        continue;
      }
      if (abs(cellRow - queryRow) == ring) {
        // A full row of the ring: only the columns within reach of the k-th best are visited.
        var rangeFirst = firstColumn;
        var rangeEnd = endColumn;
        // Not narrowed while the k-th squared distance has overflowed to infinity (queries beyond
        // about 1e19): the reach would be inf - inf = NaN and skip tied lower IDs in other columns.
        if (found == K && bestDistances[K - 1u] < 3.0e38) {
          let reach = sqrt(max(bestDistances[K - 1u] - rowGapSquared, 0.0)) + slack;
          let lowColumn = floor((x - reach - lattice.minimumX) / lattice.cellWidth);
          let highColumn = floor((x + reach - lattice.minimumX) / lattice.cellWidth);
          rangeFirst = max(firstColumn, i32(clamp(lowColumn, -1.0, f32(lastColumn))));
          rangeEnd = min(endColumn, i32(clamp(highColumn, -1.0, f32(lastColumn))));
        }
        if (rangeFirst <= rangeEnd) {
          let cellBegin = cellOffsets[cellOffsetsOffset + rowBase + u32(rangeFirst)];
          let cellEnd = cellOffsets[cellOffsetsOffset + rowBase + u32(rangeEnd) + 1u];
          ${r(`cellBegin`,`cellEnd`)}
        }
      } else {
        if (queryColumn - ring >= 0) {
          let columnGap = max(getAxisGap(x, lattice.minimumX, lattice.cellWidth, queryColumn, queryColumn - ring) - slack, 0.0);
          if (!(found == K && bestDistances[K - 1u] < columnGap * columnGap + rowGapSquared)) {
            let cell = rowBase + u32(queryColumn - ring);
            let cellBegin = cellOffsets[cellOffsetsOffset + cell];
            let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
            ${r(`cellBegin`,`cellEnd`)}
          }
        }
        if (queryColumn + ring <= lastColumn) {
          let columnGap = max(getAxisGap(x, lattice.minimumX, lattice.cellWidth, queryColumn, queryColumn + ring) - slack, 0.0);
          if (!(found == K && bestDistances[K - 1u] < columnGap * columnGap + rowGapSquared)) {
            let cell = rowBase + u32(queryColumn + ring);
            let cellBegin = cellOffsets[cellOffsetsOffset + cell];
            let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
            ${r(`cellBegin`,`cellEnd`)}
          }
        }
      }
    }
    ${n.close}
    // Clearance of the visited box on each side; a side at the lattice edge has nothing beyond it.
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
    let safeSquared = safeClearance * safeClearance;
    if (found == K && bestDistances[K - 1u] < safeSquared) {
      break;
    }
    if (lattice.bounded && lattice.radiusSquared < safeSquared) {
      break;
    }
  }`}function le(e){return`
const K: u32 = ${e}u;

fn isNeighborBefore(leftDistance: f32, leftId: u32, rightDistance: f32, rightId: u32) -> bool {
  return leftDistance < rightDistance || (leftDistance == rightDistance && leftId < rightId);
}

fn insertNeighbor(
  bestDistances: ptr<function, array<f32, ${e}>>,
  bestIds: ptr<function, array<u32, ${e}>>,
  found: ptr<function, u32>,
  distanceSquared: f32,
  neighbor: u32
) {
  if (*found == K &&
      !isNeighborBefore(distanceSquared, neighbor, (*bestDistances)[K - 1u], (*bestIds)[K - 1u])) {
    return;
  }
  var slot = min(*found, K - 1u);
  loop {
    if (slot == 0u) {
      break;
    }
    let previous = slot - 1u;
    if (isNeighborBefore((*bestDistances)[previous], (*bestIds)[previous], distanceSquared, neighbor)) {
      break;
    }
    (*bestDistances)[slot] = (*bestDistances)[previous];
    (*bestIds)[slot] = (*bestIds)[previous];
    slot = previous;
  }
  (*bestDistances)[slot] = distanceSquared;
  (*bestIds)[slot] = neighbor;
  *found = min(*found + 1u, K);
}
`}var ue=`
fn getNeighborWeight(distance: f32, bandwidth: f32) -> f32 {
  let weightKind = u32(readParameter(5u));
  var weight = 1.0;
  if (weightKind == 1u) {
    weight = pow(max(distance, readParameter(7u)), -readParameter(6u));
  } else if (weightKind == 2u) {
    let z = select(0.0, distance / bandwidth, bandwidth > 0.0);
    // 1 - z from the difference of the distances: for the adaptive k-th neighbor the bandwidth is
    // d_k * 1.0000001, and 1 - d / bandwidth rounds to 0 in f32 for some rows, which would drop the
    // slot (zero weights are not neighbors). The difference stays positive and accurate.
    let oneMinusZ = select(1.0, (bandwidth - distance) / bandwidth, bandwidth > 0.0);
    let kernel = u32(readParameter(8u));
    if (kernel == 0u) {
      weight = exp(-0.5 * z * z) * 0.3989422804014327;
    } else if (kernel == 1u) {
      weight = max(oneMinusZ, 0.0);
    } else if (kernel == 2u) {
      weight = 0.75 * max(oneMinusZ * (1.0 + z), 0.0);
    } else if (kernel == 3u) {
      let base = max(oneMinusZ * (1.0 + z), 0.0);
      weight = 0.9375 * base * base;
    } else {
      weight = 0.5;
    }
  }
  return select(0.0, weight, isFiniteFloat(weight) && weight >= 0.0);
}
`;function de(e,t){let n=D(t);return`
  ${n.preamble}
  let longitude = getGeodesicQueryLongitude(lattice, x);
  let reach = lattice.radiusDegrees;
  let firstRow = getCellRow(lattice, max(y - reach, lattice.minimumY));
  let lastRow = getCellRow(lattice, min(y + reach, lattice.maximumY));
  var reachLongitude = 180.0;
  if (abs(y) + reach < 90.0) {
    let ratio = min(sin(radians(reach)) / geodesicCosLatitude(y), 1.0);
    reachLongitude = min(degrees(asin(ratio)) * CELL_MARGIN + 0.0001, 180.0);
  }
  let queryPoint = vec2<f32>(getNeighborSearchCanonicalLongitude(x), y);
  ${n.open}
  for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
    let rowBase = ${n.rowBase(`cellRow`)};
    var nextColumn = 0u;
    for (var shiftIndex = 0u; shiftIndex < 3u; shiftIndex++) {
      var lowLongitude = lattice.minimumX;
      var highLongitude = lattice.maximumX;
      if (reachLongitude < 180.0) {
        let columnOffset = 360.0 * (f32(shiftIndex) - 1.0);
        lowLongitude = max(longitude - reachLongitude + columnOffset, lattice.minimumX);
        highLongitude = min(longitude + reachLongitude + columnOffset, lattice.maximumX);
      } else if (shiftIndex != 1u) {
        continue;
      }
      if (lowLongitude > highLongitude) {
        continue;
      }
      let firstColumn = max(getCellColumn(lattice, lowLongitude), nextColumn);
      let lastColumn = getCellColumn(lattice, highLongitude);
      if (firstColumn > lastColumn) {
        continue;
      }
      nextColumn = lastColumn + 1u;
      let cellBegin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let cellEnd = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      for (var cellSlot = cellBegin; cellSlot < cellEnd; cellSlot++) {
        let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 2u];
        let targetPoint = vec2<f32>(
          getNeighborSearchCanonicalLongitude(bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS])),
          bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 1u]));
        let distance = geodesicHaversineDistance(queryPoint, targetPoint, SPHERE_RADIUS);
        if (distance <= lattice.radius${n.condition}) {
          ${e}
        }
      }
    }
  }
  ${n.close}`}function fe(e,t){let n=D(t);return`
  ${n.preamble}
  let longitude = getGeodesicQueryLongitude(lattice, x);
  let queryColumn = i32(getCellColumn(lattice, longitude));
  let queryRow = i32(getCellRow(lattice, y));
  let columnCount = i32(lattice.columns);
  let lastColumn = columnCount - 1;
  let lastRow = i32(lattice.rows) - 1;
  let periodic = lattice.periodic;
  let lowReach = select(queryColumn, columnCount / 2, periodic);
  let highReach = select(lastColumn - queryColumn, columnCount - 1 - columnCount / 2, periodic);
  let maximumRing = max(lattice.columns, lattice.rows);
  let slack = (lattice.cellWidth + lattice.cellHeight) * 0.0009765625 +
    (abs(lattice.minimumX) + abs(lattice.maximumX) + abs(lattice.minimumY) + abs(lattice.maximumY)) * 0.00000095367431640625;
  let queryPoint = vec2<f32>(getNeighborSearchCanonicalLongitude(x), y);
  let queryCosine = geodesicCosLatitude(y);
  for (var ringIndex = 0u; ringIndex < maximumRing; ringIndex++) {
    let ring = i32(ringIndex);
    let lowExtent = min(ring, lowReach);
    let highExtent = min(ring, highReach);
    let firstRow = max(queryRow - ring, 0);
    let endRow = min(queryRow + ring, lastRow);
    ${n.open}
    for (var cellRow = firstRow; cellRow <= endRow; cellRow++) {
      let rowBase = ${n.rowBase(`u32(cellRow)`)};
      let fullRow = abs(cellRow - queryRow) == ring;
      // Full rows of the ring visit every column of the box; other rows only its two sides
      // (a non-full row has ring > 0), each only while it still holds a column not seen before.
      let columnStep = select(2 * ring, 1, fullRow);
      let startOffset = select(-ring, -lowExtent, fullRow);
      let stopOffset = select(ring, highExtent, fullRow);
      for (var columnOffset = startOffset; columnOffset <= stopOffset; columnOffset += columnStep) {
        if (!fullRow && ((columnOffset < 0 && ring > lowReach) || (columnOffset > 0 && ring > highReach))) {
          continue;
        }
        var column = queryColumn + columnOffset;
        if (periodic) {
          column = ((column % columnCount) + columnCount) % columnCount;
        }
        let cell = rowBase + u32(column);
        let cellBegin = cellOffsets[cellOffsetsOffset + cell];
        let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
        for (var cellSlot = cellBegin; cellSlot < cellEnd; cellSlot++) {
          let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 2u];
          if (${e}) {
            let targetPoint = vec2<f32>(
              getNeighborSearchCanonicalLongitude(bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS])),
              bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * TARGET_WORDS + 1u]));
            let distance = geodesicHaversineDistance(queryPoint, targetPoint, SPHERE_RADIUS);
            if ((!lattice.bounded || distance <= lattice.radius)${n.condition}) {
              insertNeighbor(&bestDistances, &bestIds, &found, distance, neighbor);
            }
          }
        }
      }
    }
    ${n.close}
    // The visited box: rows [firstRow, endRow], virtual columns [queryColumn - lowExtent, queryColumn + highExtent].
    let hasSouth = firstRow > 0;
    let hasNorth = endRow < lastRow;
    let arcFull = lowExtent + highExtent + 1 >= columnCount;
    let hasWest = select(queryColumn - lowExtent > 0, !arcFull, periodic);
    let hasEast = select(queryColumn + highExtent < lastColumn, !arcFull, periodic);
    if (!(hasSouth || hasNorth || hasWest || hasEast)) {
      break;
    }
    var hLatitude = 1.0;
    if (hasSouth || hasNorth) {
      var clearance = 360.0;
      if (hasSouth) {
        clearance = min(clearance, y - (lattice.minimumY + f32(firstRow) * lattice.cellHeight));
      }
      if (hasNorth) {
        clearance = min(clearance, lattice.minimumY + f32(endRow + 1) * lattice.cellHeight - y);
      }
      let halfSine = sin(0.5 * radians(clamp(clearance - slack, 0.0, 180.0)));
      hLatitude = halfSine * halfSine;
    }
    var hLongitude = 1.0;
    if (hasWest || hasEast) {
      let westEdge = lattice.minimumX + f32(queryColumn - lowExtent) * lattice.cellWidth;
      let eastEdge = lattice.minimumX + f32(queryColumn + highExtent + 1) * lattice.cellWidth;
      var longitudeBound = 360.0;
      if (hasEast) {
        let farEdge = select(lattice.maximumX, westEdge + 360.0, periodic);
        longitudeBound = min(longitudeBound, getWrappedLongitudeBound(eastEdge - longitude - slack, farEdge - longitude + slack));
      }
      if (hasWest) {
        let farEdge = select(lattice.minimumX, eastEdge - 360.0, periodic);
        longitudeBound = min(longitudeBound, getWrappedLongitudeBound(longitude - westEdge - slack, longitude - farEdge + slack));
      }
      let southLatitude = lattice.minimumY + f32(firstRow) * lattice.cellHeight;
      let northLatitude = min(lattice.minimumY + f32(endRow + 1) * lattice.cellHeight, lattice.maximumY);
      let maximumLatitude = clamp(max(abs(southLatitude), abs(northLatitude)) + slack, 0.0, 90.0);
      let halfSine = sin(0.5 * radians(clamp(longitudeBound, 0.0, 180.0)));
      hLongitude = queryCosine * geodesicCosLatitude(maximumLatitude) * halfSine * halfSine;
    }
    let boundTerm = clamp(min(hLatitude, hLongitude), 0.0, 1.0);
    let safeDistance = 2.0 * SPHERE_RADIUS * asin(sqrt(boundTerm)) * 0.9998779296875;
    if (found == K && bestDistances[K - 1u] < safeDistance) {
      break;
    }
    if (lattice.bounded && lattice.radius < safeDistance) {
      break;
    }
  }`}var O=`
${E}

// Preserve the established lookup for queries inside the lattice. Outside it, use the
// equivalent longitude nearest its midpoint, so three shifted cap intervals cover every cell.
fn getGeodesicQueryLongitude(lattice: Lattice, longitude: f32) -> f32 {
  if (longitude >= lattice.minimumX && longitude <= lattice.maximumX) {
    return longitude;
  }
  let midpoint = lattice.minimumX + (lattice.maximumX - lattice.minimumX) * 0.5;
  let wrappedLongitude = getNeighborSearchCanonicalLongitude(longitude);
  // Add only whole turns: midpoint + (wrapped - midpoint) would erase tiny offsets near zero.
  // Distance evaluation uses the canonical value directly, before this lattice-relative shift.
  return wrappedLongitude + 360.0 * round((midpoint - wrappedLongitude) / 360.0);
}

fn getWrappedLongitudeBound(lowValue: f32, highValue: f32) -> f32 {
  let lowEnd = min(lowValue, highValue);
  let highEnd = max(lowValue, highValue);
  if (ceil(lowEnd / 360.0) * 360.0 <= highEnd) {
    return 0.0;
  }
  return min(abs(lowEnd - 360.0 * round(lowEnd / 360.0)), abs(highEnd - 360.0 * round(highEnd / 360.0)));
}
`;function pe(e){return`
const BVH_NODE_WORDS: u32 = 5u;
const BVH_TARGET_ROWS: u32 = ${e.targetRows}u;
const BVH_INTERNAL_NODE_COUNT: u32 = ${e.internalNodeCount}u;
const BVH_STACK_SIZE: u32 = ${e.levelCount+2}u;
const BVH_BOUND_SLACK: f32 = 1.000001;

fn getBVHBoxBound(node: u32, x: f32, y: f32) -> f32 {
  let base = nodeDataOffset + node * BVH_NODE_WORDS;
  let minimumX = bitcast<f32>(nodeData[base]);
  let minimumY = bitcast<f32>(nodeData[base + 1u]);
  let maximumX = bitcast<f32>(nodeData[base + 2u]);
  let maximumY = bitcast<f32>(nodeData[base + 3u]);
  if (minimumX > maximumX || minimumY > maximumY) {
    return -1.0;
  }
  let gapX = max(max(minimumX - x, x - maximumX), 0.0);
  let gapY = max(max(minimumY - y, y - maximumY), 0.0);
  return gapX * gapX + gapY * gapY;
}
`}function me(e){return e?`let queryIndex = index;`:`let queryIndex = nodeData[nodeDataOffset + (BVH_INTERNAL_NODE_COUNT + index) * BVH_NODE_WORDS + 4u];
  if (queryIndex >= BVH_TARGET_ROWS) {
    return;
  }`}function he(e){return`
  var stackNode: array<u32, BVH_STACK_SIZE>;
  var stackBound: array<f32, BVH_STACK_SIZE>;
  var top = 0u;
  let rootBound = getBVHBoxBound(0u, x, y);
  if (rootBound >= 0.0) {
    stackNode[0] = 0u;
    stackBound[0] = rootBound;
    top = 1u;
  }
  loop {
    if (top == 0u) {
      break;
    }
    top = top - 1u;
    let node = stackNode[top];
    var limited = lattice.bounded;
    var limit = lattice.radiusSquared * BVH_BOUND_SLACK;
    if (found == K) {
      limit = select(bestDistances[K - 1u] * BVH_BOUND_SLACK,
        min(limit, bestDistances[K - 1u] * BVH_BOUND_SLACK), limited);
      limited = true;
    }
    if (limited && stackBound[top] > limit) {
      continue;
    }
    if (node < BVH_INTERNAL_NODE_COUNT) {
      let first = node * 2u + 1u;
      let second = first + 1u;
      let firstBound = getBVHBoxBound(first, x, y);
      let secondBound = getBVHBoxBound(second, x, y);
      let firstUsable = firstBound >= 0.0 && !(limited && firstBound > limit);
      let secondUsable = secondBound >= 0.0 && !(limited && secondBound > limit);
      // Push the farther child first so the nearer one is popped first.
      if (firstUsable && secondUsable) {
        if (firstBound <= secondBound) {
          stackNode[top] = second;
          stackBound[top] = secondBound;
          stackNode[top + 1u] = first;
          stackBound[top + 1u] = firstBound;
        } else {
          stackNode[top] = first;
          stackBound[top] = firstBound;
          stackNode[top + 1u] = second;
          stackBound[top + 1u] = secondBound;
        }
        top = top + 2u;
      } else if (firstUsable) {
        stackNode[top] = first;
        stackBound[top] = firstBound;
        top = top + 1u;
      } else if (secondUsable) {
        stackNode[top] = second;
        stackBound[top] = secondBound;
        top = top + 1u;
      }
      continue;
    }
    let base = nodeDataOffset + node * BVH_NODE_WORDS;
    let neighbor = nodeData[base + 4u];
    if (neighbor >= BVH_TARGET_ROWS || !(${e})) {
      continue;
    }
    let deltaX = bitcast<f32>(nodeData[base]) - x;
    let deltaY = bitcast<f32>(nodeData[base + 1u]) - y;
    let distanceSquared = deltaX * deltaX + deltaY * deltaY;
    if (!lattice.bounded || distanceSquared <= lattice.radiusSquared) {
      insertNeighbor(&bestDistances, &bestIds, &found, distanceSquared, neighbor);
    }
  }`}function ge(e){return`
  var stackNode: array<u32, BVH_STACK_SIZE>;
  var top = 0u;
  let limit = lattice.radiusSquared * BVH_BOUND_SLACK;
  let rootBound = getBVHBoxBound(0u, x, y);
  if (rootBound >= 0.0 && rootBound <= limit) {
    stackNode[0] = 0u;
    top = 1u;
  }
  loop {
    if (top == 0u) {
      break;
    }
    top = top - 1u;
    let node = stackNode[top];
    if (node < BVH_INTERNAL_NODE_COUNT) {
      let first = node * 2u + 1u;
      let second = first + 1u;
      let firstBound = getBVHBoxBound(first, x, y);
      let secondBound = getBVHBoxBound(second, x, y);
      if (firstBound >= 0.0 && firstBound <= limit) {
        stackNode[top] = first;
        top = top + 1u;
      }
      if (secondBound >= 0.0 && secondBound <= limit) {
        stackNode[top] = second;
        top = top + 1u;
      }
      continue;
    }
    let base = nodeDataOffset + node * BVH_NODE_WORDS;
    let neighbor = nodeData[base + 4u];
    if (neighbor >= BVH_TARGET_ROWS) {
      continue;
    }
    let deltaX = bitcast<f32>(nodeData[base]) - x;
    let deltaY = bitcast<f32>(nodeData[base + 1u]) - y;
    let distanceSquared = deltaX * deltaX + deltaY * deltaY;
    if (distanceSquared <= lattice.radiusSquared) {
      ${e}
    }
  }`}var k=`GPUNeighborSearch`,_e=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),metricsByCoordinateSpace:Object.freeze({planar:Object.freeze([`none`,`native`]),"longitude-latitude":Object.freeze([`great-circle`])}),measure:`distance`}),A=65536,j=Object.freeze({encodings:Object.freeze([`float32-relative`])}),ve=class{id;props;spatialContext;outputWeights;searchStructure;temporalContext;constructor(e){let i=n(e,[`id`,`mode`,`searchStructure`,`spatialContext`,`temporalContext`,`positions`,`queryPositions`,`mask`,`queryMask`,`times`,`queryTimes`,`gridSize`,`k`,`parameters`,`rowTileSize`,`timeBucketCount`,`rowStandardize`,`weightTransform`,`output`]),o=n(i.output,[`offsets`,`neighbors`,`weights`,`distances`,`status`,`neighborCounts`]),s=i.temporalContext?n(i.temporalContext,[`unit`,`epoch`,`encoding`,`timeZone`,`axis`]):void 0,c=s?.axis?n(s.axis,[`kind`,`step`]):void 0;e=Object.freeze({...i,output:Object.freeze({...o,status:n(o.status,[`overflow`,`candidateOverflow`,`requiredCount`,`invalidCount`])}),...i.gridSize?{gridSize:Object.freeze([...i.gridSize])}:{},...s&&c?{temporalContext:Object.freeze({...s,axis:Object.freeze({...c,...`step`in c&&c.step&&typeof c.step==`object`?{step:n(c.step,[`calendarUnit`,`interval`])}:{}})})}:{},...i.spatialContext?{spatialContext:t({spatialContext:i.spatialContext}).spatialContext}:{}});let l=e.id??`neighbor-search`;if(this.id=l,this.props=e,this.spatialContext=a(l,e.spatialContext,_e),e.mode!==`knn`&&e.mode!==`radius`)throw Error(`${l} mode must be 'knn' or 'radius'`);let u=e.searchStructure??`auto`;if(![`auto`,`grid`,`bvh`].includes(u))throw Error(`${l} searchStructure must be 'auto', 'grid' or 'bvh'`);if(u===`bvh`&&this.spatialContext.coordinateSpace===`longitude-latitude`)throw Error(`${l} searchStructure 'bvh' supports planar positions only`);this.searchStructure=this.spatialContext.coordinateSpace===`longitude-latitude`?`grid`:u,f(e.positions,[`float32x2`],`${l} positions`);let d=e.positions.length;h(l,`positions length`,d,1,2**31-1),e.queryPositions&&(f(e.queryPositions,[`float32x2`],`${l} queryPositions`),h(l,`queryPositions length`,e.queryPositions.length,1,2**31-1));let p=this.getQueryRows();e.times?(m(l,e.temporalContext!==void 0,`times need a temporalContext`),this.temporalContext=ae(l,e.temporalContext,j),m(l,this.searchStructure!==`bvh`,`times need searchStructure 'grid' or 'auto'`),f(e.times,[`float32`],`${l} times`),m(l,e.times.length===d,`times length must equal positions length`),e.queryPositions?(m(l,e.queryTimes!==void 0,`queryTimes are required with times in a cross join`),f(e.queryTimes,[`float32`],`${l} queryTimes`),m(l,e.queryTimes.length===p,`queryTimes length must equal queryPositions length`)):m(l,e.queryTimes===void 0,`queryTimes need queryPositions`),h(l,`timeBucketCount`,e.timeBucketCount??1,1,A)):m(l,e.temporalContext===void 0&&e.queryTimes===void 0&&e.timeBucketCount===void 0,`temporalContext, queryTimes and timeBucketCount need times`),f(e.parameters,[`float32`],`${l} parameters`);let ee=e.times?15:12;if(e.parameters.length<ee)throw Error(`${l} parameters must hold ${ee} float32 values`);let[te,ne]=e.gridSize;if(h(l,`gridSize columns`,te,1,2**53-1),h(l,`gridSize rows`,ne,1,2**53-1),e.mode===`knn`&&h(l,`k`,e.k??NaN,1,256),e.mask&&(f(e.mask,[`uint32`],`${l} mask`),e.mask.length!==d))throw Error(`${l} mask length must equal positions length`);if(e.queryMask&&(f(e.queryMask,[`uint32`],`${l} queryMask`),e.queryMask.length!==p))throw Error(`${l} queryMask length must equal the query row count`);let{output:g}=e,v=S(l,g);if(r(l,{offsets:g.offsets,neighbors:g.neighbors,weights:g.weights,distances:g.distances,overflow:g.status.overflow,candidateOverflow:g.status.candidateOverflow,requiredCount:g.status.requiredCount,invalidCount:g.status.invalidCount,neighborCounts:g.neighborCounts},[]),v!==p)throw Error(`${l} output.offsets length must equal the query row count + 1`);let{requiredCount:y,invalidCount:b}=g.status;for(let[e,t]of[[`output.status.overflow`,g.status.overflow],[`output.status.candidateOverflow`,g.status.candidateOverflow],[`output.status.requiredCount`,y],[`output.status.invalidCount`,b]])if(t&&(f(t,[`uint32`],`${l} ${e}`),t.length<1))throw Error(`${l} ${e} must hold one uint32`);if(g.neighborCounts&&(f(g.neighborCounts,[`uint32`],`${l} output.neighborCounts`),g.neighborCounts.length!==p))throw Error(`${l} output.neighborCounts length must equal the query row count`);if(e.weightTransform!==void 0&&![`binary`,`raw`,`kernel`,`row`].includes(e.weightTransform))throw Error(`${l} weightTransform must be 'binary', 'raw', 'kernel' or 'row'`);if(e.rowStandardize!==void 0&&e.weightTransform!==void 0&&e.rowStandardize!==(e.weightTransform===`row`))throw Error(`${l} rowStandardize ${e.rowStandardize} conflicts with weightTransform '${e.weightTransform}'`);let x=e.rowStandardize?`row`:e.weightTransform??`unknown`,re=this.spatialContext.coordinateSpace===`planar`&&e.mode===`radius`&&!e.queryPositions&&!e.queryMask;this.outputWeights=C(g,{spatialContext:this.spatialContext,information:{rowCount:p,transform:x,symmetric:re&&x!==`unknown`&&x!==`row`?!0:`unknown`,hasSelfWeights:!1,provenance:{source:k,criterion:e.mode===`knn`?`knn`:`distance-band`,k:e.mode===`knn`?e.k:void 0}}}),_(l,[b,g.offsets,g.neighbors,g.weights,g.distances,g.status.overflow,g.status.candidateOverflow,y,g.neighborCounts],[e.positions,e.queryPositions,e.mask,e.queryMask,e.parameters,e.times,e.queryTimes])}getQueryRows(){return(this.props.queryPositions??this.props.positions).length}getCommandNodes(t){let{id:n,props:r}=this,{positions:a,queryPositions:l,parameters:f,gridSize:m,output:h,mode:_}=r;g(n,t,[a,l,r.mask,r.queryMask,f,r.times,r.queryTimes,h.offsets,h.neighbors,h.weights,h.distances,h.status.overflow,h.status.candidateOverflow,h.status.requiredCount,h.status.invalidCount,h.neighborCounts]);let{times:v}=r,y=v?r.timeBucketCount??1:1,S=v?{bucketCount:y}:void 0,C=y>1,ae=m[0]*m[1]*y;p(n,`gridSize cells`,ae,te(t.device));let T=a.length,E=this.getQueryRows(),D=h.neighbors.length;h.status.candidateOverflow&&ne(t.device,n,`output.status.candidateOverflow`,s(h.status.candidateOverflow).size,`use a smaller packed status view`);let O=r.k??1;_===`knn`&&ne(t.device,n,`k-nearest lists`,E*O*4,`lower k or the query row count`);let{spatialContext:_e}=this,A=_e.coordinateSpace===`longitude-latitude`,j=`${oe(_,m,i(_e),S)}
${re}
${S?`fn getGridUnitTime(lattice: Lattice, time: f32) -> f32 {
  // The grid drops a coordinate above 1, and float32 rounding can put the latest time there.
  return min(1.0, (time - lattice.minimumTime) / (lattice.timeBucketWidth * f32(TIME_BUCKETS)));
}`:``}`,ve=A?de:se,M=!!l,ye=l?[{name:`queryPositions`,view:l,type:`f32`,access:`read`}]:[],N=v?4:3,be=r.queryTimes??v,P=e=>{let t=e===`times`?v:be;return t?[{name:e,view:t,type:`f32`,access:`read`}]:[]},F=(e,t)=>`${e}[${e}Offset + ${t}]`,I=M?`queryTimes`:`times`,xe=P(I),Se=M?`isCrossJoinQueryInBounds`:`isPointInBounds`,Ce=M?`isCrossJoinQueryValid`:`isPointValid`,we=e=>M?`isCrossJoinQueryTimeInBounds(${F(I,e)})`:`isTimeInBounds(lattice, ${F(I,e)})`,Te=e=>v?` && ${we(e)}`:``,L=M?`queryPositions`:`positions`,R=e=>`let x = ${L}[${L}Offset + ${e} * 2u];
  let y = ${L}[${L}Offset + ${e} * 2u + 1u];`,z=M?`true`:`neighbor != queryIndex`,B=r.queryMask??(M?void 0:r.mask),V=!M&&!r.queryMask,H=this.searchStructure===`bvh`,U=!H,Ee=e=>U?e:1,W=C?3:2,De=C?`float32x3`:`float32x2`,Oe=o(t,`${n}-grid-positions`,De,T),ke=C?[m[0],m[1],y]:[m[0],m[1]],G=U?x(t,{id:n,positions:Oe,gridSize:ke,rowCount:T}):void 0,Ae=G?.cellOffsets??o(t,`${n}-cell-offsets`,`uint32`,1),je=G?.sortedRows??o(t,`${n}-sorted-rows`,`uint32`,1),Me=G?.gridCount??o(t,`${n}-grid-count`,`uint32`,1),Ne=o(t,`${n}-sorted-targets`,`uint32`,Ee(T*N)),K=h.neighborCounts??o(t,`${n}-counts`,`uint32`,E),Pe=o(t,`${n}-starts`,`uint32`,E),q=h.distances??o(t,`${n}-distances`,`float32`,D),Fe=je,Ie=V?Ne:o(t,`${n}-sorted-queries`,`uint32`,Ee(E*N)),Le=Me,J=[...h.status.candidateOverflow?[d(t,{id:`${n}-candidate-status`,operation:k,invocationCount:1,bindings:[{name:`candidateOverflow`,view:h.status.candidateOverflow,type:`u32`,access:`read_write`}],body:`candidateOverflow[candidateOverflowOffset] = 0u;`})]:[],d(t,{id:`${n}-grid-positions`,operation:k,variant:`grid-positions`,bindings:[{name:`positions`,view:a,type:`f32`,access:`read`},{name:`parameters`,view:f,type:`f32`,access:`read`},...r.mask?[{name:`mask`,view:r.mask,type:`u32`,access:`read`}]:[],...P(`times`),{name:`gridPositions`,view:Oe,type:`f32`,access:`read_write`}],invocationCount:T,declarations:j,body:`let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${r.mask?`mask[maskOffset + index] != 0u`:`true`} && isPointValid(lattice, x, y)${v?` && isTimeInBounds(lattice, ${F(`times`,`index`)})`:``};
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${U?`let unit = getGridUnitPosition(lattice, x, y);`:`let unit = vec2f(x, y);`}
  gridPositions[gridPositionsOffset + index * ${W}u] = select(invalid, unit.x, included);
  gridPositions[gridPositionsOffset + index * ${W}u + 1u] = select(invalid, unit.y, included);${C?`
  gridPositions[gridPositionsOffset + index * 3u + 2u] = select(invalid, getGridUnitTime(lattice, ${F(`times`,`index`)}), included);`:``}`}),...G?.nodes??[],d(t,{id:`${n}-query-valid`,operation:k,variant:`query-valid`,bindings:[...M?ye:[{name:`positions`,view:a,type:`f32`,access:`read`}],{name:`parameters`,view:f,type:`f32`,access:`read`},...B?[{name:`queryMask`,view:B,type:`u32`,access:`read`}]:[],...xe,{name:`counts`,view:K,type:`u32`,access:`read_write`}],invocationCount:E,declarations:j,body:`let lattice = readLattice();
  ${R(`index`)}
  let included = ${B?`queryMask[queryMaskOffset + index] != 0u`:`true`};
  counts[countsOffset + index] = select(0u, 1u, included && ${Ce}(lattice, x, y)${Te(`index`)});`})];if(U&&!V){let e=o(t,`${n}-query-grid-positions`,De,E),r=x(t,{id:`${n}-query`,positions:e,gridSize:ke,rowCount:E});J.push(d(t,{id:`${n}-query-grid-positions`,operation:k,variant:`query-grid-positions`,bindings:[...M?ye:[{name:`positions`,view:a,type:`f32`,access:`read`}],{name:`parameters`,view:f,type:`f32`,access:`read`},...B?[{name:`queryMask`,view:B,type:`u32`,access:`read`}]:[],...xe,{name:`queryGridPositions`,view:e,type:`f32`,access:`read_write`}],invocationCount:E,declarations:j,body:`let lattice = readLattice();
  ${R(`index`)}
  let included = ${B?`queryMask[queryMaskOffset + index] != 0u`:`true`} && ${Ce}(lattice, x, y)${Te(`index`)};
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  // Only the cell order of the queries is used, so a query outside the lattice sorts with the
  // nearest edge cell; the grid index would otherwise skip it.
  let unit = clamp(getGridUnitPosition(lattice, x, y), vec2f(0.0), vec2f(1.0));
  queryGridPositions[queryGridPositionsOffset + index * ${W}u] = select(invalid, unit.x, included);
  queryGridPositions[queryGridPositionsOffset + index * ${W}u + 1u] = select(invalid, unit.y, included);${C?`
  queryGridPositions[queryGridPositionsOffset + index * 3u + 2u] = select(invalid, max(0.0, getGridUnitTime(lattice, ${F(I,`index`)})), included);`:``}`}),...r.nodes),Fe=r.sortedRows,Le=r.gridCount}let Y=V?`sortedTargets`:`sortedQueries`,Re=V?`gridCount`:`queryCount`,ze=[{name:`sortedTargets`,view:Ne,type:`u32`,access:`read`},...V?[]:[{name:`sortedQueries`,view:Ie,type:`u32`,access:`read`}],{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`cellOffsets`,view:Ae,type:`u32`,access:`read`},{name:Re,view:Le,type:`u32`,access:`read`}],Be=`if (index >= ${Re}[${Re}Offset]) {
    return;
  }
  let queryIndex = ${Y}[${Y}Offset + index * TARGET_WORDS + 2u];
  let x = bitcast<f32>(${Y}[${Y}Offset + index * TARGET_WORDS]);
  let y = bitcast<f32>(${Y}[${Y}Offset + index * TARGET_WORDS + 1u]);${v?`
  let queryTime = bitcast<f32>(${Y}[${Y}Offset + index * TARGET_WORDS + 3u]);`:``}`;U&&J.push(d(t,{id:`${n}-sorted-targets`,operation:k,variant:`sorted-targets`,bindings:[{name:`positions`,view:a,type:`f32`,access:`read`},...P(`times`),{name:`sortedRows`,view:je,type:`u32`,access:`read`},{name:`gridCount`,view:Me,type:`u32`,access:`read`},{name:`sortedTargets`,view:Ne,type:`u32`,access:`read_write`}],invocationCount:T,body:`if (index >= gridCount[gridCountOffset]) {
    return;
  }
  let row = sortedRows[sortedRowsOffset + index];
  sortedTargets[sortedTargetsOffset + index * ${N}u] = bitcast<u32>(positions[positionsOffset + row * 2u]);
  sortedTargets[sortedTargetsOffset + index * ${N}u + 1u] = bitcast<u32>(positions[positionsOffset + row * 2u + 1u]);
  sortedTargets[sortedTargetsOffset + index * ${N}u + 2u] = row;${v?`
  sortedTargets[sortedTargetsOffset + index * ${N}u + 3u] = bitcast<u32>(${F(`times`,`row`)});`:``}`})),U&&!V&&J.push(d(t,{id:`${n}-sorted-queries`,operation:k,variant:`sorted-queries`,bindings:[{name:`queryPositions`,view:l??a,type:`f32`,access:`read`},...P(`queryTimes`),{name:`queryOrder`,view:Fe,type:`u32`,access:`read`},{name:`queryCount`,view:Le,type:`u32`,access:`read`},{name:`sortedQueries`,view:Ie,type:`u32`,access:`read_write`}],invocationCount:E,body:`if (index >= queryCount[queryCountOffset]) {
    return;
  }
  let row = queryOrder[queryOrderOffset + index];
  sortedQueries[sortedQueriesOffset + index * ${N}u] = bitcast<u32>(queryPositions[queryPositionsOffset + row * 2u]);
  sortedQueries[sortedQueriesOffset + index * ${N}u + 1u] = bitcast<u32>(queryPositions[queryPositionsOffset + row * 2u + 1u]);
  sortedQueries[sortedQueriesOffset + index * ${N}u + 2u] = row;${v?`
  sortedQueries[sortedQueriesOffset + index * ${N}u + 3u] = bitcast<u32>(${F(`queryTimes`,`row`)});`:``}`}));let X,Z=``;if(H){let{bvh:e,nodes:r}=b(t,{id:`${n}-bvh`,operation:k,features:{kind:`points`,positions:Oe},featureCount:T,leafCapacity:Math.max(2,ee(T)),spatialSort:!0});J.push(...r),X=o(t,`${n}-bvh-node-data`,`uint32`,e.nodeCount*5),Z=pe({targetRows:T,internalNodeCount:e.internalNodeCount,levelCount:e.levelCount}),J.push(d(t,{id:`${n}-bvh-node-data`,operation:k,variant:`bvh-node-data`,bindings:[{name:`nodeMinima`,view:e.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:e.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:e.leafIds,type:`u32`,access:`read`},{name:`nodeData`,view:X,type:`u32`,access:`read_write`}],invocationCount:e.nodeCount,declarations:`const INTERNAL_NODE_COUNT: u32 = ${e.internalNodeCount}u;
const NO_ROW: u32 = 0xffffffffu;`,body:`let base = nodeDataOffset + index * 5u;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(NO_ROW, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`}))}let Ve=X?[{name:`nodeData`,view:X,type:`u32`,access:`read`},{name:L,view:l??a,type:`f32`,access:`read`},{name:`parameters`,view:f,type:`f32`,access:`read`}]:[],He=me(M),Q,$;if(_===`knn`){Q=o(t,`${n}-knn-ids`,`uint32`,E*O),$=o(t,`${n}-knn-distances`,`float32`,E*O);let e=`// Re-sort the selected neighbors by ID (insertion sort over at most K entries).
  for (var slot = 1u; slot < found; slot++) {
    let neighborId = bestIds[slot];
    let distanceSquared = bestDistances[slot];
    var destination = slot;
    while (destination > 0u && bestIds[destination - 1u] > neighborId) {
      bestIds[destination] = bestIds[destination - 1u];
      bestDistances[destination] = bestDistances[destination - 1u];
      destination--;
    }
    bestIds[destination] = neighborId;
    bestDistances[destination] = distanceSquared;
  }
  for (var slot = 0u; slot < found; slot++) {
    knnIds[knnIdsOffset + queryIndex * K + slot] = bestIds[slot];
    knnDistances[knnDistancesOffset + queryIndex * K + slot] = ${A?`bestDistances[slot]`:`sqrt(bestDistances[slot])`};
  }
  counts[countsOffset + queryIndex] = found;`,i=[{name:`counts`,view:K,type:`u32`,access:`read_write`},{name:`knnIds`,view:Q,type:`u32`,access:`read_write`},{name:`knnDistances`,view:$,type:`f32`,access:`read_write`}];U&&J.push(...w(t,{id:`${n}-knn`,operation:k,variant:`knn`,bindings:[...ze,...i],invocationCount:E,declarations:`${j}
${le(O)}`,body:`${Be}
  var found = 0u;
  var bestDistances: array<f32, ${O}>;
  var bestIds: array<u32, ${O}>;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${A?fe(z,S):ce(z,S)}
  }
  ${e}`},r.rowTileSize)),H&&J.push(...w(t,{id:`${n}-bvh-knn`,operation:k,variant:`bvh-knn`,bindings:[...Ve,...i],invocationCount:E,declarations:`${j}
${le(O)}
${Z}`,body:`${He}
  var found = 0u;
  var bestDistances: array<f32, ${O}>;
  var bestIds: array<u32, ${O}>;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${R(`queryIndex`)}
    ${he(z)}
  }
  ${e}`},r.rowTileSize))}else U&&J.push(...w(t,{id:`${n}-count`,operation:k,variant:`radius-count`,bindings:[...ze,{name:`counts`,view:K,type:`u32`,access:`read_write`}],invocationCount:E,declarations:j,body:`${Be}
  var count = 0u;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${ve(`if (${z}) {
          count++;
        }`,S)}
  }
  counts[countsOffset + queryIndex] = count;`},r.rowTileSize)),H&&J.push(...w(t,{id:`${n}-bvh-count`,operation:k,variant:`bvh-radius-count`,bindings:[...Ve,{name:`counts`,view:K,type:`u32`,access:`read_write`}],invocationCount:E,declarations:`${j}
${Z}`,body:`${He}
  var count = 0u;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${R(`queryIndex`)}
    ${ge(`if (${z}) {
        count++;
      }`)}
  }
  counts[countsOffset + queryIndex] = count;`},r.rowTileSize));let Ue=o(t,`${n}-scan-total`,`uint32`,1);if(J.push(...e(t,{id:`${n}-scan`,operation:k,counts:K,offsets:Pe,rowCount:E,total:Ue}),d(t,{id:`${n}-offsets`,operation:k,variant:`offsets`,bindings:[{name:`scanTotal`,view:Ue,type:`u32`,access:`read`},{name:`starts`,view:Pe,type:`u32`,access:`read`},{name:`offsets`,view:h.offsets,type:`u32`,access:`read_write`},{name:`overflow`,view:h.status.overflow,type:`u32`,access:`read_write`},...h.status.requiredCount?[{name:`totalNeighbors`,view:h.status.requiredCount,type:`u32`,access:`read_write`}]:[]],invocationCount:E+1,declarations:`const QUERY_ROWS: u32 = ${E}u;
const CAPACITY: u32 = ${D}u;`,body:`if (index < QUERY_ROWS) {
    offsets[offsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = scanTotal[scanTotalOffset];
    offsets[offsetsOffset + QUERY_ROWS] = min(total, CAPACITY);
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${h.status.requiredCount?`totalNeighbors[totalNeighborsOffset] = total;`:``}
  }`})),Q&&$)J.push(...w(t,{id:`${n}-emit`,operation:k,variant:`knn-emit`,bindings:[{name:`knnIds`,view:Q,type:`u32`,access:`read`},{name:`knnDistances`,view:$,type:`f32`,access:`read`},{name:`offsets`,view:h.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:h.neighbors,type:`u32`,access:`read_write`},{name:`distances`,view:q,type:`f32`,access:`read_write`}],invocationCount:E,declarations:`const K: u32 = ${O}u;`,body:`let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var slot = begin; slot < end; slot++) {
    let source = index * K + slot - begin;
    neighbors[neighborsOffset + slot] = knnIds[knnIdsOffset + source];
    distances[distancesOffset + slot] = knnDistances[knnDistancesOffset + source];
  }`},r.rowTileSize));else{let e=[{name:`offsets`,view:h.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:h.neighbors,type:`u32`,access:`read_write`},{name:`distances`,view:q,type:`f32`,access:`read_write`}];H&&J.push(...w(t,{id:`${n}-bvh-emit`,operation:k,variant:`bvh-radius-emit`,bindings:[...Ve,...e],invocationCount:E,declarations:`${j}
${Z}`,body:`${He}
  let begin = offsets[offsetsOffset + queryIndex];
  let end = offsets[offsetsOffset + queryIndex + 1u];
  if (end > begin) {
    let lattice = readLattice();
    ${R(`queryIndex`)}
    var next = begin;
    ${ge(`if (${z} && next < end) {
        neighbors[neighborsOffset + next] = neighbor;
        distances[distancesOffset + next] = sqrt(distanceSquared);
        next++;
      }`)}
  }`},r.rowTileSize)),U&&J.push(...w(t,{id:`${n}-emit`,operation:k,variant:`radius-emit`,bindings:[...ze,...e],invocationCount:E,declarations:j,body:`${Be}
  let begin = offsets[offsetsOffset + queryIndex];
  let end = offsets[offsetsOffset + queryIndex + 1u];
  if (end > begin) {
    let lattice = readLattice();
    var next = begin;
    ${ve(`if (${z} && next < end) {
          neighbors[neighborsOffset + next] = neighbor;
          distances[distancesOffset + next] = ${A?`distance`:`sqrt(distanceSquared)`};
          next++;
        }`,S)}
  }`},r.rowTileSize)),J.push(...w(t,{id:`${n}-sort-rows`,operation:k,variant:`radius-sort-rows`,bindings:[{name:`offsets`,view:h.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:h.neighbors,type:`u32`,access:`read_write`},{name:`distances`,view:q,type:`f32`,access:`read_write`}],invocationCount:E,body:`let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var slot = begin + 1u; slot < end; slot++) {
    let neighborId = neighbors[neighborsOffset + slot];
    let distance = distances[distancesOffset + slot];
    var destination = slot;
    while (destination > begin && neighbors[neighborsOffset + destination - 1u] > neighborId) {
      neighbors[neighborsOffset + destination] = neighbors[neighborsOffset + destination - 1u];
      distances[distancesOffset + destination] = distances[distancesOffset + destination - 1u];
      destination--;
    }
    neighbors[neighborsOffset + destination] = neighborId;
    distances[distancesOffset + destination] = distance;
  }`},r.rowTileSize))}if(h.status.invalidCount){let e=T+(M?E:0),i=o(t,`${n}-invalid-flags`,`uint32`,e),s=o(t,`${n}-invalid-starts`,`uint32`,e),l=(e,t)=>`let x = ${e}[${e}Offset + ${t} * 2u];
    let y = ${e}[${e}Offset + ${t} * 2u + 1u];`;J.push(d(t,{id:`${n}-invalid-flags`,operation:k,variant:`invalid-flags`,bindings:[{name:`positions`,view:a,type:`f32`,access:`read`},...ye,{name:`parameters`,view:f,type:`f32`,access:`read`},...P(`times`),...M?P(`queryTimes`):[],...r.mask?[{name:`mask`,view:r.mask,type:`u32`,access:`read`}]:[],...M&&r.queryMask?[{name:`queryMask`,view:r.queryMask,type:`u32`,access:`read`}]:[],{name:`invalidFlags`,view:i,type:`u32`,access:`read_write`}],invocationCount:e,declarations:`${j}
const TARGET_ROWS: u32 = ${T}u;`,body:`let lattice = readLattice();
  var included = true;
  var inside = false;
  if (index < TARGET_ROWS) {
    ${l(`positions`,`index`)}
    included = ${r.mask?`mask[maskOffset + index] != 0u`:`true`};
    inside = isPointInBounds(lattice, x, y)${v?` && isTimeInBounds(lattice, ${F(`times`,`index`)})`:``};
  } else {
    ${M?`let queryRow = index - TARGET_ROWS;
    ${l(`queryPositions`,`queryRow`)}
    included = ${M&&r.queryMask?`queryMask[queryMaskOffset + queryRow] != 0u`:`true`};
    inside = ${Se}(lattice, x, y)${v?` && ${we(`queryRow`)}`:``};`:``}
  }
  invalidFlags[invalidFlagsOffset + index] = select(0u, 1u, included && !inside);`}),...new c({id:`${n}-invalid-scan`,input:i,output:s,mode:`exclusive`}).getCommandNodes(t),d(t,{id:`${n}-invalid-count`,operation:k,variant:`invalid-count`,bindings:[{name:`invalidFlags`,view:i,type:`u32`,access:`read`},{name:`invalidStarts`,view:s,type:`u32`,access:`read`},{name:`invalidCount`,view:h.status.invalidCount,type:`u32`,access:`read_write`}],invocationCount:1,body:`invalidCount[invalidCountOffset] =
    invalidStarts[invalidStartsOffset + ${e-1}u] + invalidFlags[invalidFlagsOffset + ${e-1}u];`}))}let We=_===`radius`?`let bandwidth = readParameter(4u);`:`var bandwidth = 0.0;
  for (var slot = begin; slot < end; slot++) {
    bandwidth = max(bandwidth, distances[distancesOffset + slot]);
  }
  bandwidth = bandwidth * ${u(ie)};`;return J.push(d(t,{id:`${n}-weights`,operation:k,variant:`weights`,bindings:[{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`offsets`,view:h.offsets,type:`u32`,access:`read`},{name:`distances`,view:q,type:`f32`,access:`read`},{name:`weights`,view:h.weights,type:`f32`,access:`read_write`}],invocationCount:E,declarations:`${j}
${ue}`,body:`let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${We}
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let weight = getNeighborWeight(distances[distancesOffset + slot], bandwidth);
    weights[weightsOffset + slot] = weight;
    sum += weight;
  }
  if (${r.rowStandardize===void 0?`readParameter(9u) != 0.0`:String(r.rowStandardize)} && sum > 0.0 && isFiniteFloat(sum)) {
    for (var slot = begin; slot < end; slot++) {
      weights[weightsOffset + slot] = weights[weightsOffset + slot] / sum;
    }
  }`})),J}};export{oe as i,le as n,ce as r,ve as t};
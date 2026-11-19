import{S as e,f as t}from"./gpu-contributor-utils-BIwLQON5.js";import{r as n}from"./kernel-C3tfQOn4.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as i,n as a,o}from"./pair-statistics-grid-L8bKgCyF.js";var s=2048,c=`4294967040.0`,l=`9223372036854775808.0`;function u(e,n,r){if(t(e,`slotCount`,n,1),t(e,`channelCount`,r,1),n*r>2048)throw Error(`${e} needs at most ${s} histogram accumulators (bins x channels)`)}function d(t,s){let{id:d,operation:f,positions:p,parameters:m,gridSize:h,sortedRows:g,cellOffsets:_}=s,{slotCount:v,channelCount:y}=s;u(d,v,y);let b=s.extraBindings??[];e(d,`extraBindings length`,b.length,3);let x=v*y,S=n(t,`${d}-accumulators`,`uint32`,x*2+1),C=p.length,w=s.pairOrder===`ordered`,T=o(t,{id:d,operation:f,positions:p,sortedRows:g,cellOffsets:_,gridSize:h}),E=(e,t)=>`for (var candidateSlot = ${e}; candidateSlot < ${t}; candidateSlot++) {
        let neighbor = sortedPoints[sortedPointsOffset + candidateSlot * 3u + 2u];
        if (${w?`neighbor != focus`:`true`}) {
          let neighborX = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u]);
          let neighborY = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u + 1u]);
          ${w?`let deltaX = neighborX - x;
          let deltaY = neighborY - y;`:`// Oriented from the lower to the higher row ID, whichever row is the focus.
          let focusIsLower = focus < neighbor;
          let deltaX = select(x - neighborX, neighborX - x, focusIsLower);
          let deltaY = select(y - neighborY, neighborY - y, focusIsLower);`}
          let distanceSquared = deltaX * deltaX + deltaY * deltaY;
          if (distanceSquared <= lattice.radiusSquared) {
            let pairDistance = sqrt(distanceSquared);
            ${s.pairAction}
          }
        }
      }`;return{nodes:[...T.nodes,a(t,`${d}-clear`,f,S),r(t,{id:`${d}-pairs`,operation:f,variant:`pair-histogram-${s.pairOrder}`,bindings:[{name:`parameters`,view:m,type:`f32`,access:`read`},{name:`sortedPoints`,view:T.sortedPoints,type:`u32`,access:`read`},{name:`cellOffsets`,view:_,type:`u32`,access:`read`},...b,{name:`accumulators`,view:S,type:`atomic<u32>`,access:`read_write`}],invocationCount:C,workgroupSize:64,guardIndex:!1,declarations:`${i(h)}
const SLOT_COUNT: u32 = ${v}u;
const CHANNEL_COUNT: u32 = ${y}u;
const ACCUMULATOR_COUNT: u32 = ${x}u;
var<workgroup> localLow: array<atomic<u32>, ${x}>;
var<workgroup> localHigh: array<atomic<u32>, ${x}>;

// Adds the 64-bit amount (highAmount * 2^32 + amount) to an accumulator. highAmount is at most
// 2^31. A carry out of bit 63 of the workgroup partial sets the global overflow flag.
fn accumulateWide(slot: u32, channel: u32, amount: u32, highAmount: u32) {
  if ((amount == 0u && highAmount == 0u) || slot >= SLOT_COUNT) {
    return;
  }
  let word = slot * CHANNEL_COUNT + channel;
  var highAddend = highAmount;
  if (amount != 0u) {
    let previous = atomicAdd(&localLow[word], amount);
    if (previous > 0xffffffffu - amount) {
      highAddend += 1u;
    }
  }
  if (highAddend != 0u) {
    let previousHigh = atomicAdd(&localHigh[word], highAddend);
    if (previousHigh > 0xffffffffu - highAddend) {
      atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
    }
  }
}

fn accumulate(slot: u32, channel: u32, amount: u32) {
  accumulateWide(slot, channel, amount, 0u);
}

// Rounds a non-negative scaled term to a u32 amount; negative or NaN terms give 0.
fn quantizePairAmount(scaledValue: f32) -> u32 {
  return select(0u, u32(min(scaledValue + 0.5, ${c})), scaledValue > 0.0);
}

// Accumulates a non-negative scaled term without letting its magnitude set the rounding unit:
// a term up to 2^32 - 256 units is rounded to the nearest unit into smallChannel; a larger
// term (rare: an outlier or sentinel pair) is rounded to a multiple of 2^12 units (relative
// error below 2^-20, plus the f32 ulp) and added as an exact 64-bit amount to largeChannel,
// which readers scale back by 2^12. A term beyond 2^75 units is clamped and
// raises the overflow flag, as does any accumulator sum that carries out of 64 bits.
fn accumulateScaled(slot: u32, smallChannel: u32, largeChannel: u32, scaledValue: f32) {
  if (!(scaledValue > 0.0)) {
    return;
  }
  if (scaledValue <= ${c}) {
    accumulateWide(slot, smallChannel, u32(scaledValue + 0.5), 0u);
    return;
  }
  var coarse = scaledValue * ${1/4096};
  if (!(coarse <= ${l})) {
    atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
    coarse = ${l};
  }
  // Exact in f32: scaling by powers of two, and coarse - high * 2^32 has no more bits than coarse.
  let high = floor(coarse * 2.3283064365386963e-10);
  let low = coarse - high * 4294967296.0;
  accumulateWide(slot, largeChannel, u32(min(low + 0.5, ${c})), u32(high));
}

${s.declarations??``}`,body:`for (var word = localInvocationIndex; word < ACCUMULATOR_COUNT; word += 64u) {
    atomicStore(&localLow[word], 0u);
    atomicStore(&localHigh[word], 0u);
  }
  workgroupBarrier();
  let lattice = readLattice();
  let includedCount = cellOffsets[cellOffsetsOffset + CELL_COUNT];
  if (index < INVOCATION_COUNT && index < includedCount && lattice.valid) {
    let focus = sortedPoints[sortedPointsOffset + index * 3u + 2u];
    let x = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u]);
    let y = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u + 1u]);
    ${s.focusPrologue??``}
    let column = getCellColumn(lattice, x);
    let row = getCellRow(lattice, y);
    let firstColumn = max(column, 1u) - 1u;
    let lastColumn = min(column + 1u, lattice.columns - 1u);
    ${w?`let firstRow = max(row, 1u) - 1u;
    let lastRow = min(row + 1u, lattice.rows - 1u);
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowBase = cellRow * lattice.columns;
      let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      ${E(`begin`,`end`)}
    }`:`// Forward half stencil: the cells after the focus cell in row-major cell order. The rest
    // of the focus row up to the right neighbor cell is one slot range, the row below another.
    {
      let rowEnd = cellOffsets[cellOffsetsOffset + row * lattice.columns + lastColumn + 1u];
      ${E(`index + 1u`,`rowEnd`)}
    }
    if (row + 1u < lattice.rows) {
      let rowBase = (row + 1u) * lattice.columns;
      let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      ${E(`begin`,`end`)}
    }`}
    ${s.focusEpilogue??``}
  }
  workgroupBarrier();
  for (var word = localInvocationIndex; word < ACCUMULATOR_COUNT; word += 64u) {
    let low = atomicLoad(&localLow[word]);
    let high = atomicLoad(&localHigh[word]);
    if (low != 0u || high != 0u) {
      let previous = atomicAdd(&accumulators[accumulatorsOffset + word * 2u], low);
      let carry = select(0u, 1u, previous > 0xffffffffu - low);
      let highAddend = high + carry;
      if (highAddend < high) {
        atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
      }
      if (highAddend != 0u) {
        let previousHigh = atomicAdd(&accumulators[accumulatorsOffset + word * 2u + 1u], highAddend);
        if (previousHigh > 0xffffffffu - highAddend) {
          atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
        }
      }
    }
  }
`})],accumulators:S,sortedPoints:T.sortedPoints}}function f(e,t,n=``,r){return`
fn readPairAccumulator${n}(slot: u32, channel: u32) -> f32 {
  let word = (slot * ${t}u + channel) * 2u;
  return f32(${e}[${e}Offset + word + 1u]) * 4294967296.0 +
    f32(${e}[${e}Offset + word]);
}

fn readPairCount${n}(slot: u32, channel: u32) -> u32 {
  let word = (slot * ${t}u + channel) * 2u;
  return select(
    ${e}[${e}Offset + word],
    0xffffffffu,
    ${e}[${e}Offset + word + 1u] != 0u
  );
}
${r===void 0?``:`
fn readPairOverflow${n}() -> bool {
  return ${e}[${e}Offset + ${r*t*2}u] != 0u;
}
`}`}export{u as i,d as n,f as r,s as t};
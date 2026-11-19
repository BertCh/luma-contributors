import{r as e}from"./kernel-C3tfQOn4.js";import{T as t}from"./gpu-contributor-utils-BIwLQON5.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";var r=1024;function i(e,t){return Math.max(1,Math.ceil((e+t)/r))}function a(e,t){let r=l(t);return n(e,{id:t.id,operation:t.operation,variant:t.variant??`load-balanced-for-each`,bindings:[c(t),...t.bindings],workgroupSize:256,invocationCount:r.workgroupCount*256,guardIndex:!1,declarations:`${u()}
${t.declarations}`,constants:r.constants,body:`${d({consume:`visitLoadBalancedItem(loadBalancedRow, loadBalancedSlot, loadBalancedLane);`,emit:``})}`})}function o(r,i){let a=l(i),{accumulatorType:o}=i,f=s(o),p=o.includes(`f32`)?`f32`:`u32`,m=i.offsets.length-1,h=m*a.laneCount*f;t(i.id,i.output.length>=h,`output must hold at least ${h} elements (rows x lanes x components), got ${i.output.length}`);let g=1+f,_=e(r,`${i.id}-tile-carries`,`uint32`,a.workgroupCount*g),v=(e,t)=>f===1?e:`${e}[${t}]`,y=e=>p===`f32`?`bitcast<u32>(${e})`:e,b=e=>p===`f32`?`bitcast<f32>(${e})`:e,x=(e,t,n)=>Array.from({length:f},(r,i)=>`loadBalancedOutput[loadBalancedOutputOffset + (${e} * LOAD_BALANCED_LANE_COUNT + loadBalancedLane) * ${f}u + ${i}u] ${n} ${v(t,i)};`).join(`
      `),S=e=>f===1?b(`loadBalancedCarries[loadBalancedCarriesOffset + ${e} * ${g}u + 1u]`):`${o}(${Array.from({length:f},(t,n)=>b(`loadBalancedCarries[loadBalancedCarriesOffset + ${e} * ${g}u + ${n+1}u]`)).join(`, `)})`,C={name:`loadBalancedOutput`,view:i.output,type:p,access:`read_write`},w={name:`loadBalancedCarries`,view:_,type:`u32`,access:`read_write`};return[n(r,{id:i.id,operation:i.operation,variant:i.variant??`load-balanced-segment-sum`,bindings:[c(i),C,w,...i.bindings],workgroupSize:256,invocationCount:a.workgroupCount*256,guardIndex:!1,declarations:`${u()}
var<workgroup> loadBalancedCarryRows: array<u32, 256>;
var<workgroup> loadBalancedCarryValues: array<${o}, 256>;
${i.declarations}`,constants:a.constants,body:`var loadBalancedRunning = ${o}();
  var loadBalancedFirstValue = ${o}();
  var loadBalancedHasEmitted = false;
  ${d({consume:`loadBalancedRunning = loadBalancedRunning +
          getLoadBalancedContribution(loadBalancedRow, loadBalancedSlot, loadBalancedLane);`,emit:`if (loadBalancedHasEmitted) {
            ${x(`loadBalancedRow`,`loadBalancedRunning`,`=`)}
          } else {
            loadBalancedFirstValue = loadBalancedRunning;
            loadBalancedHasEmitted = true;
          }
          loadBalancedRunning = ${o}();`})}
  // Inclusive segmented scan of the thread runs, keyed by row (rows never decrease by thread).
  loadBalancedCarryRows[localInvocationIndex] = loadBalancedRow;
  loadBalancedCarryValues[localInvocationIndex] = loadBalancedRunning;
  workgroupBarrier();
  for (var stride = 1u; stride < 256u; stride = stride * 2u) {
    var previousRow = 0xffffffffu;
    var previousValue = ${o}();
    if (localInvocationIndex >= stride) {
      previousRow = loadBalancedCarryRows[localInvocationIndex - stride];
      previousValue = loadBalancedCarryValues[localInvocationIndex - stride];
    }
    workgroupBarrier();
    if (previousRow == loadBalancedCarryRows[localInvocationIndex]) {
      loadBalancedCarryValues[localInvocationIndex] =
        previousValue + loadBalancedCarryValues[localInvocationIndex];
    }
    workgroupBarrier();
  }
  if (loadBalancedIsActive) {
    if (loadBalancedHasEmitted) {
      // Thread t starts in the row where thread t - 1 stopped; its scan holds the earlier runs.
      var carryIn = ${o}();
      if (localInvocationIndex > 0u &&
          loadBalancedCarryRows[localInvocationIndex - 1u] == loadBalancedStartRow) {
        carryIn = loadBalancedCarryValues[localInvocationIndex - 1u];
      }
      let rowValue = carryIn + loadBalancedFirstValue;
      ${x(`loadBalancedStartRow`,`rowValue`,`=`)}
    }
    if (localInvocationIndex == 255u) {
      let carryRow = loadBalancedCarryRows[localInvocationIndex];
      let carryValue = loadBalancedCarryValues[localInvocationIndex];
      let record = loadBalancedCarriesOffset + workgroupIndex * ${g}u;
      loadBalancedCarries[record] = select(LOAD_BALANCED_ROW_COUNT, carryRow, loadBalancedTileLength > 0u);
      ${Array.from({length:f},(e,t)=>`loadBalancedCarries[record + ${t+1}u] = ${y(v(`carryValue`,t))};`).join(`
      `)}
    }
  }`}),n(r,{id:`${i.id}-fixup`,operation:i.operation,variant:`load-balanced-segment-sum-fixup`,bindings:[w,C],invocationCount:a.workgroupCount,workgroupSize:64,declarations:`override LOAD_BALANCED_ROW_COUNT: u32;
override LOAD_BALANCED_LANE_COUNT: u32;
override LOAD_BALANCED_TILES_PER_LANE: u32;`,constants:{LOAD_BALANCED_ROW_COUNT:m,LOAD_BALANCED_LANE_COUNT:a.laneCount,LOAD_BALANCED_TILES_PER_LANE:a.tileCount},body:`let loadBalancedLane = index / LOAD_BALANCED_TILES_PER_LANE;
  let tile = index % LOAD_BALANCED_TILES_PER_LANE;
  let row = loadBalancedCarries[loadBalancedCarriesOffset + index * ${g}u];
  if (row >= LOAD_BALANCED_ROW_COUNT) {
    return;
  }
  // Only the first tile of a run of tiles carrying the same row adds the run, in tile order.
  if (tile > 0u && loadBalancedCarries[loadBalancedCarriesOffset + (index - 1u) * ${g}u] == row) {
    return;
  }
  var sum = ${S(`index`)};
  let laneEnd = (loadBalancedLane + 1u) * LOAD_BALANCED_TILES_PER_LANE;
  var next = index + 1u;
  loop {
    if (next >= laneEnd ||
        loadBalancedCarries[loadBalancedCarriesOffset + next * ${g}u] != row) {
      break;
    }
    sum = sum + ${S(`next`)};
    next = next + 1u;
  }
  ${x(`row`,`sum`,`+=`)}`})]}function s(e){return e.startsWith(`vec4`)?4:e.startsWith(`vec2`)?2:1}function c(e){return{name:`loadBalancedOffsets`,view:e.offsets,type:`u32`,access:`read`}}function l(e){let t=e.offsets.length-1,n=e.laneCount??1;if(t<0||!Number.isSafeInteger(e.itemCapacity)||e.itemCapacity<0||!Number.isSafeInteger(n)||n<1||n*(t+e.itemCapacity)>4294967295)throw Error(`${e.operation}: invalid load-balanced topology for ${e.id}`);let r=i(t,e.itemCapacity),a=n*r;if(a*256>4294967295)throw Error(`${e.operation}: load-balanced dispatch of ${e.id} is too large`);return{laneCount:n,tileCount:r,workgroupCount:a,constants:{LOAD_BALANCED_ROW_COUNT:t,LOAD_BALANCED_ITEM_CAPACITY:e.itemCapacity,LOAD_BALANCED_LANE_COUNT:n,LOAD_BALANCED_TILES_PER_LANE:r}}}function u(){return`override LOAD_BALANCED_ROW_COUNT: u32;
override LOAD_BALANCED_ITEM_CAPACITY: u32;
override LOAD_BALANCED_LANE_COUNT: u32;
override LOAD_BALANCED_TILES_PER_LANE: u32;
var<workgroup> loadBalancedTileRowEnds: array<u32, 1025>;
var<workgroup> loadBalancedTileBounds: array<u32, 2>;
/** End of row \`row\` relative to \`itemBase\`, clamped to the items \`[itemBase, itemLimit)\`. */
fn getLoadBalancedRowEnd(row: u32, itemBase: u32, itemLimit: u32) -> u32 {
  return clamp(loadBalancedOffsets[loadBalancedOffsetsOffset + row + 1u], itemBase, itemLimit) - itemBase;
}
/** Rows consumed at merge step \`diagonal\` (Merrill-Garland merge-path search). */
fn searchLoadBalancedPath(diagonal: u32, itemBase: u32, itemLimit: u32) -> u32 {
  let itemCount = itemLimit - itemBase;
  var low = select(0u, diagonal - itemCount, diagonal > itemCount);
  var high = min(diagonal, LOAD_BALANCED_ROW_COUNT);
  loop {
    if (low >= high) { break; }
    let pivot = (low + high) / 2u;
    if (getLoadBalancedRowEnd(pivot, itemBase, itemLimit) <= diagonal - pivot - 1u) {
      low = pivot + 1u;
    } else {
      high = pivot;
    }
  }
  return low;
}`}function d(e){return`let loadBalancedIsActive =
    workgroupIndex < LOAD_BALANCED_LANE_COUNT * LOAD_BALANCED_TILES_PER_LANE;
  let loadBalancedLane = workgroupIndex / LOAD_BALANCED_TILES_PER_LANE;
  let loadBalancedTile = workgroupIndex % LOAD_BALANCED_TILES_PER_LANE;
  let loadBalancedItemBase = min(
    loadBalancedOffsets[loadBalancedOffsetsOffset],
    LOAD_BALANCED_ITEM_CAPACITY
  );
  let loadBalancedItemLimit = clamp(
    loadBalancedOffsets[loadBalancedOffsetsOffset + LOAD_BALANCED_ROW_COUNT],
    loadBalancedItemBase,
    LOAD_BALANCED_ITEM_CAPACITY
  );
  let loadBalancedMergeCount =
    LOAD_BALANCED_ROW_COUNT + loadBalancedItemLimit - loadBalancedItemBase;
  let loadBalancedTileBegin = select(
    loadBalancedMergeCount,
    min(loadBalancedTile * ${r}u, loadBalancedMergeCount),
    loadBalancedIsActive
  );
  let loadBalancedTileEnd = min(loadBalancedTileBegin + ${r}u, loadBalancedMergeCount);
  let loadBalancedTileLength = loadBalancedTileEnd - loadBalancedTileBegin;
  if (localInvocationIndex < 2u) {
    loadBalancedTileBounds[localInvocationIndex] = searchLoadBalancedPath(
      select(loadBalancedTileBegin, loadBalancedTileEnd, localInvocationIndex == 1u),
      loadBalancedItemBase,
      loadBalancedItemLimit
    );
  }
  workgroupBarrier();
  let loadBalancedTileRowBegin = loadBalancedTileBounds[0];
  let loadBalancedTileRowCount = loadBalancedTileBounds[1] - loadBalancedTileRowBegin;
  // Item coordinates are relative to loadBalancedItemBase; rows at or past ROW_COUNT never end.
  let loadBalancedTileItemBegin = loadBalancedTileBegin - loadBalancedTileRowBegin;
  let loadBalancedTileItemCount =
    loadBalancedTileEnd - loadBalancedTileBounds[1] - loadBalancedTileItemBegin;
  for (var entry = localInvocationIndex; entry <= loadBalancedTileRowCount; entry += 256u) {
    let entryRow = loadBalancedTileRowBegin + entry;
    var rowEnd = 0xffffffffu;
    if (entryRow < LOAD_BALANCED_ROW_COUNT) {
      rowEnd = getLoadBalancedRowEnd(entryRow, loadBalancedItemBase, loadBalancedItemLimit);
    }
    loadBalancedTileRowEnds[entry] = rowEnd;
  }
  workgroupBarrier();
  let loadBalancedDiagonal = min(
    localInvocationIndex * 4u,
    loadBalancedTileLength
  );
  var loadBalancedLow = select(
    0u,
    loadBalancedDiagonal - loadBalancedTileItemCount,
    loadBalancedDiagonal > loadBalancedTileItemCount
  );
  var loadBalancedHigh = min(loadBalancedDiagonal, loadBalancedTileRowCount);
  loop {
    if (loadBalancedLow >= loadBalancedHigh) { break; }
    let pivot = (loadBalancedLow + loadBalancedHigh) / 2u;
    if (loadBalancedTileRowEnds[pivot] <= loadBalancedTileItemBegin + loadBalancedDiagonal - pivot - 1u) {
      loadBalancedLow = pivot + 1u;
    } else {
      loadBalancedHigh = pivot;
    }
  }
  var loadBalancedRow = loadBalancedTileRowBegin + loadBalancedLow;
  let loadBalancedStartRow = loadBalancedRow;
  var loadBalancedItem = loadBalancedTileItemBegin + loadBalancedDiagonal - loadBalancedLow;
  let loadBalancedStepCount = min(
    4u,
    loadBalancedTileLength - loadBalancedDiagonal
  );
  for (var walkStep = 0u; walkStep < 4u; walkStep = walkStep + 1u) {
    if (walkStep < loadBalancedStepCount) {
      if (loadBalancedItem < loadBalancedTileRowEnds[loadBalancedRow - loadBalancedTileRowBegin]) {
        let loadBalancedSlot = loadBalancedItemBase + loadBalancedItem;
        ${e.consume}
        loadBalancedItem = loadBalancedItem + 1u;
      } else {
        ${e.emit}
        loadBalancedRow = loadBalancedRow + 1u;
      }
    }
  }`}export{o as n,i as r,a as t};
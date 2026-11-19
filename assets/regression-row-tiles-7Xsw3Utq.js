import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";import{t}from"./pairwise-sum-wgsl-iErcC5Or.js";var n=4096,r=64,i=2**24;function a(e){let t=Math.max(r,Math.ceil(e/n));return{tileRowCount:t,tileCount:Math.max(1,Math.ceil(e/t))}}function o(e,t){return[[e.tileCount,64],[t]]}function s(n,r){let{layout:i,sumCount:a}=r,o=a+(r.maximumCount??0),s=`const TERM_COUNT: u32 = ${o}u;
const SUM_COUNT: u32 = ${a}u;`;return[e(n,{id:`${r.id}-tiles`,operation:r.operation,variant:`${r.variant}-tiles`,workgroupSize:64,invocationCount:i.tileCount,constants:{ROW_COUNT:r.rowCount,TILE_ROWS:i.tileRowCount},...r.tileGate??{},bindings:[...r.bindings,{name:`tilePartial`,view:r.partial,type:`f32`,access:`read_write`}],declarations:`override ROW_COUNT: u32;
override TILE_ROWS: u32;
${s}
${r.declarations}`,body:`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var terms: array<f32, ${o}>;
  for (var row = firstRow; row < lastRow; row++) {
    accumulateRow(row, &terms);
  }
  for (var term = 0u; term < TERM_COUNT; term++) {
    tilePartial[tilePartialOffset + index * TERM_COUNT + term] = terms[term];
  }`}),e(n,{id:`${r.id}-merge`,operation:r.operation,variant:`${r.variant}-merge`,invocationCount:o,constants:{TILE_COUNT:i.tileCount},...r.mergeGate??{},bindings:[{name:`tilePartial`,view:r.partial,type:`f32`,access:`read`},{name:`tileTotal`,view:r.output,type:`f32`,access:`read_write`}],declarations:`override TILE_COUNT: u32;
${s}
const OUTPUT_SLOT: u32 = ${r.outputSlot??0}u;`,body:`if (index < SUM_COUNT) {
    ${t({count:`TILE_COUNT`,term:`tilePartial[tilePartialOffset + sumItem * TERM_COUNT + index]`})}
    tileTotal[tileTotalOffset + OUTPUT_SLOT + index] = sum;
    return;
  }
  var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, tilePartial[tilePartialOffset + tile * TERM_COUNT + index]);
  }
  tileTotal[tileTotalOffset + OUTPUT_SLOT + index] = largest;`})]}export{s as i,a as n,o as r,i as t};
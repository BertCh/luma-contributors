// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Use one workgroup to amortize dispatch overhead on moderate fragment tables. */
export const MAXIMUM_OFFSET_WORKGROUP_FRAGMENT_CAPACITY = 65536;
export const OFFSET_ASSEMBLY_WORKGROUP_SIZE = 64;
export const MAXIMUM_OFFSET_WORKGROUP_ROUNDS_PER_DISPATCH = 32;

/** Each lane visits at most ceil(capacity / workgroupSize) rows in a round. */
export function getOffsetAssemblyWorkgroupWGSL(capacity: number) {
  const declarations = `
const FRAGMENT_CAPACITY:u32=${capacity}u;
const LANES:u32=${OFFSET_ASSEMBLY_WORKGROUP_SIZE}u;
const ROUNDS:u32=${MAXIMUM_OFFSET_WORKGROUP_ROUNDS_PER_DISPATCH}u;
var<workgroup> liveRows:u32;
`;
  return {
    declarations,
    cancelDeclarations: `
var<workgroup> firstCandidate:u32;
var<workgroup> oppositeRow:atomic<u32>;
`,
    cancelBody: `
if(index==0u) {
 liveRows=min(live[liveOffset],FRAGMENT_CAPACITY);
 firstCandidate=control[controlOffset];
}
let count=workgroupUniformLoad(&liveRows);
let begin=workgroupUniformLoad(&firstCandidate);
for(var round=0u;round<ROUNDS;round++) {
 let candidate=begin+round;
 if(candidate>=count) {break;}
 if(index==0u) {atomicStore(&oppositeRow,NONE);}
 workgroupBarrier();
 let eligible=state(candidate)==1u;
 if(eligible) {
  for(var row=index;row<FRAGMENT_CAPACITY && row<count;row+=LANES) {
   if(isOpposite(candidate,row)) {atomicMin(&oppositeRow,row);}
  }
 }
 // Finish all reads of the candidate and the reduction before any lane changes its state.
 storageBarrier();
 workgroupBarrier();
 let opposite=atomicLoad(&oppositeRow);
 if(eligible) {
  for(var row=index;row<FRAGMENT_CAPACITY && row<count;row+=LANES) {
   if(shouldCancel(candidate,row,opposite)) {metadata[metadataOffset+original(row)*4u+2u]=0u;}
  }
 }
 storageBarrier();
 workgroupBarrier();
}
if(index==0u) {control[controlOffset]=min(begin+ROUNDS,count);}
`,
    traverseDeclarations: `
var<workgroup> currentRow:u32;
var<workgroup> committedRow:u32;
var<workgroup> firstSeed:atomic<u32>;
var<workgroup> firstActive:atomic<u32>;
var<workgroup> firstNext:atomic<u32>;
`,
    traverseBody: `
if(index==0u) {liveRows=min(control[controlOffset+11u],FRAGMENT_CAPACITY);}
let count=workgroupUniformLoad(&liveRows);
for(var round=0u;round<ROUNDS;round++) {
 if(index==0u) {
  currentRow=control[controlOffset];
  atomicStore(&firstSeed,NONE);atomicStore(&firstActive,NONE);atomicStore(&firstNext,NONE);
 }
 let current=workgroupUniformLoad(&currentRow);
 for(var row=index;row<FRAGMENT_CAPACITY && row<count;row+=LANES) {
  if(state(row)!=1u) {continue;}
  atomicMin(&firstActive,row);
  if(incoming[incomingOffset+row]==0u) {atomicMin(&firstSeed,row);}
  if(current<count && follows(current,row)) {atomicMin(&firstNext,row);}
 }
 storageBarrier();
 workgroupBarrier();
 if(index==0u) {
  committedRow=commitSelection(current,atomicLoad(&firstSeed),atomicLoad(&firstActive),atomicLoad(&firstNext),count);
 }
 let used=workgroupUniformLoad(&committedRow);
 if(used==NONE) {break;}
 storageBarrier();
 for(var row=index;row<FRAGMENT_CAPACITY && row<count;row+=LANES) {
  if(state(row)==1u && follows(used,row)) {
   incoming[incomingOffset+row]=incoming[incomingOffset+row]-1u;
  }
 }
 storageBarrier();
}
`
  };
}

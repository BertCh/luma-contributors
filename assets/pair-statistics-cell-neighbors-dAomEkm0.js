import{r as e}from"./kernel-C3tfQOn4.js";import{S as t,f as n}from"./gpu-contributor-utils-BIwLQON5.js";import{t as r}from"./graph-chunk-groups-BB8ACpQO.js";var i=256;function a(t,n){let{id:a,operation:o,segmentCount:s}=n,c=n.input.format,l=c===`float32`?`f32`:`u32`,u=[],d=n.input,f=n.rowCount;for(let p=0;;p++){let m=Math.ceil(f/i),h=m===1?n.output:e(t,`${a}-level-${p}`,c,s*m*(n.detectOverflow?2:1));if(u.push(r(t,{id:`${a}-level-${p}`,operation:o,variant:`segment-total-${l}`,invocationCount:s*m,parameterValues:[m,f,s,Number(p>0)],bindings:[{name:`entries`,view:d,type:l,access:`read`},{name:`totals`,view:h,type:l,access:`read_write`}],declarations:`fn readTotalParameter(slot: u32) -> u32 { return getChunkParameter(36u + slot); }`,body:`let groupCount = readTotalParameter(0u); let entryCount = readTotalParameter(1u);
  let segmentCount = readTotalParameter(2u);
  ${n.detectOverflow?`let segment = index / groupCount; let group = index % groupCount;
  let begin = segment * entryCount + group * ${i}u;
  let end = segment * entryCount + min((group + 1u) * ${i}u, entryCount);
  var total = ${l}(0);
  var overflow = 0u;
  for (var entry = begin; entry < end; entry++) {
    let value = entries[entriesOffset + entry];
    let exceeds = value > 0xffffffffu - total;
    total = select(total + value, 0xffffffffu, exceeds);
    overflow |= u32(exceeds);
    if (readTotalParameter(3u) != 0u) { overflow |= entries[entriesOffset + segmentCount * entryCount + entry]; }
  }
  totals[totalsOffset + index] = total;
  totals[totalsOffset + (segmentCount * groupCount) + index] = overflow;`:`let segment = index / groupCount; let group = index % groupCount;
  let begin = segment * entryCount + group * ${i}u;
  let end = segment * entryCount + min((group + 1u) * ${i}u, entryCount);
  var total = ${l}(0);
  for (var entry = begin; entry < end; entry++) { total += entries[entriesOffset + entry]; }
  totals[totalsOffset + index] = total;`} `})),m===1)return u;d=h,f=m}}var o=2**24;function s(e,r){n(e,`gridSize[0]`,r[0],1,2**24-1),n(e,`gridSize[1]`,r[1],1,2**24-1),t(e,`gridSize[0] x gridSize[1]`,r[0]*r[1],2**24-1)}function c(e,t){return`{
  let focusColumn = i32(${e} % lattice.columns);
  let focusRow = i32(${e} / lattice.columns);
  let lastCellRow = min(focusRow + 1, i32(lattice.rows) - 1);
  let lastCellColumn = min(focusColumn + 1, i32(lattice.columns) - 1);
  for (var cellRow = max(focusRow - 1, 0); cellRow <= lastCellRow; cellRow++) {
    for (var cellColumn = max(focusColumn - 1, 0); cellColumn <= lastCellColumn; cellColumn++) {
      let cell = u32(cellRow) * lattice.columns + u32(cellColumn);
      let slotEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
      for (var slot = cellOffsets[cellOffsetsOffset + cell]; slot < slotEnd; slot++) {
        let neighbor = sortedRows[sortedRowsOffset + slot];
        ${t}
      }
    }
  }
}`}export{s as i,c as n,a as r,o as t};
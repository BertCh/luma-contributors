import{a as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t}from"./spatial-output-validation-nw5lOmVL.js";import{r as n}from"./kernel-C3tfQOn4.js";import{d as r}from"./wgsl-common-DfyXiYB0.js";import{a as i,o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,S as s,f as c,p as l,u,v as d,x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{t as m}from"./gpu-sort-xRm8l5yn.js";import{t as h}from"./load-balanced-for-each-GB0Xy_f0.js";import{t as g}from"./sorted-segment-offsets-ChmOtA9F.js";import{t as _}from"./define-run-B-brkCe4.js";import{i as v,l as y,n as b,s as x}from"./access-Co69Wfld.js";import{r as S}from"./display-Bu20g6o5.js";import{d as C,s as w}from"./pointer-overlay-Qh8qflcA.js";import{l as T,p as E,v as D}from"./roads-DzovamHj.js";import{Ia as O}from"./entry-preview-D70TdSbK.js";function k(e){return[{name:`rowOffsets`,view:e.offsets,type:`u32`,access:`read`},{name:`rowFacilities`,view:e.facilities,type:`u32`,access:`read`},{name:`rowCosts`,view:e.costs,type:`f32`,access:`read`}]}function A(e){return[{name:`columnOffsets`,view:e.columnOffsets,type:`u32`,access:`read`},{name:`columnEntries`,view:e.columnEntries,type:`u32`,access:`read`}]}var j=`
fn getSparseRowCost(row: u32, facility: u32) -> f32 {
  var low = rowOffsets[rowOffsetsOffset + row];
  let end = rowOffsets[rowOffsetsOffset + row + 1u];
  var high = end;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (rowFacilities[rowFacilitiesOffset + middle] < facility) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && rowFacilities[rowFacilitiesOffset + low] == facility) {
    return rowCosts[rowCostsOffset + low];
  }
  // A runtime value: a constant-expression bitcast to infinity does not compile.
  return bitcast<f32>(0x7f800000u | (facility & 0u));
}`;function M(e,t){let{sparse:n,status:r}=t,i=n.facilities.length,o=n.offsets.length-1;return[a(e,{id:`${t.id}-validate-offsets`,operation:t.operation,variant:`sparse-validate-offsets`,invocationCount:Math.max(i,o+1),bindings:[{name:`rowOffsets`,view:n.offsets,type:`u32`,access:`read`},{name:`rowCosts`,view:n.costs,type:`f32`,access:`read`},{name:`status`,view:r,type:`atomic<u32>`,access:`read_write`}],declarations:t.declarations,body:`if (index <= DEMAND_COUNT) {
    let offset = rowOffsets[rowOffsetsOffset + index];
    var bad = false;
    if (index == 0u) { bad = bad || offset != 0u; }
    if (index == DEMAND_COUNT) {
      bad = bad || offset != SPARSE_ENTRY_COUNT;
    } else {
      bad = bad || offset > rowOffsets[rowOffsetsOffset + index + 1u];
    }
    if (bad) {
      atomicAdd(&status[statusOffset + 3u], 1u);
    }
  }
  if (index < SPARSE_ENTRY_COUNT && isInvalid(rowCosts[rowCostsOffset + index])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }`}),h(e,{id:`${t.id}-validate-rows`,operation:t.operation,variant:`sparse-validate-rows`,offsets:n.offsets,itemCapacity:i,bindings:[{name:`rowFacilities`,view:n.facilities,type:`u32`,access:`read`},{name:`status`,view:r,type:`atomic<u32>`,access:`read_write`}],declarations:`${t.declarations}
fn visitLoadBalancedItem(row: u32, item: u32, lane: u32) {
  let facility = rowFacilities[rowFacilitiesOffset + item];
  var bad = facility >= FACILITY_COUNT;
  if (item > loadBalancedOffsets[loadBalancedOffsetsOffset + row] &&
      rowFacilities[rowFacilitiesOffset + item - 1u] >= facility) {
    bad = true;
  }
  if (bad) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }
}`})]}function N(e,t){let{id:r,sparse:i,facilityCount:o}=t,s=i.facilities.length,c=(t,i)=>n(e,`${r}-transpose-${t}`,`uint32`,i),l=c(`entry-rows`,s),u=c(`entry-ids`,s),d=c(`entry-keys`,s),f=c(`sorted-keys`,s),p=c(`sorted-entry-ids`,s),_=c(`column-offsets`,o+1),v=c(`column-entries`,2*s);return{nodes:[h(e,{id:`${r}-transpose-rows`,operation:t.operation,variant:`sparse-transpose-rows`,offsets:i.offsets,itemCapacity:s,bindings:[{name:`rowFacilities`,view:i.facilities,type:`u32`,access:`read`},{name:`entryRows`,view:l,type:`u32`,access:`read_write`},{name:`entryIds`,view:u,type:`u32`,access:`read_write`},{name:`entryKeys`,view:d,type:`u32`,access:`read_write`}],declarations:`${t.declarations}
fn visitLoadBalancedItem(row: u32, item: u32, lane: u32) {
  entryRows[entryRowsOffset + item] = row;
  entryIds[entryIdsOffset + item] = item;
  // Invalid facilities (reported by validation) get the excluded key, which sorts last.
  entryKeys[entryKeysOffset + item] = min(rowFacilities[rowFacilitiesOffset + item], FACILITY_COUNT);
}`}),...new m({id:`${r}-transpose-sort`,keys:d,values:u,outputKeys:f,outputValues:p,keyBits:Math.max(1,o.toString(2).length)}).getCommandNodes(e),g(e,{id:`${r}-transpose-offsets`,operation:t.operation,variant:`sparse-transpose-offsets`,segmentCount:o,sortedKeys:f,segmentOffsets:_}),a(e,{id:`${r}-transpose-gather`,operation:t.operation,variant:`sparse-transpose-gather`,invocationCount:s,bindings:[{name:`rowCosts`,view:i.costs,type:`f32`,access:`read`},{name:`entryRows`,view:l,type:`u32`,access:`read`},{name:`sortedEntryIds`,view:p,type:`u32`,access:`read`},{name:`columnEntries`,view:v,type:`u32`,access:`read_write`}],body:`let entry = sortedEntryIds[sortedEntryIdsOffset + index];
  columnEntries[columnEntriesOffset + 2u * index] = entryRows[entryRowsOffset + entry];
  columnEntries[columnEntriesOffset + 2u * index + 1u] = bitcast<u32>(rowCosts[rowCostsOffset + entry]);`})],transpose:{columnOffsets:_,columnEntries:v}}}var P=`GPULocationAllocation`,F=256,I={selectedFacilityCount:0,uncoveredDemandCount:1,complete:2,invalidCount:3,swapIterationCount:4,converged:5,iterationLimitReached:6,length:7},L=1e3,R=64,z=60,B=1e-6;function V(e){return u(e,8)}function H(e){return u(e,4)}function U(e){return u(e,12)}function W(e){return u(e,8)}function G(e){return u(e,8)}function K(e,t){return`${e?`if (index < DEMAND_COUNT && isInvalid(demandWeights[demandWeightsOffset + index])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }`:``}
  ${t?`if (index == 0u && isInvalid(coverageDistance[coverageDistanceOffset])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }`:``}`}var q=class{id;props;demandCount;maximumSwapIterations;estimatedCommandNodeCount;constructor(e){this.id=e.id??`location-allocation`,this.props=e;let{id:n}=this,{output:r}=e;if(e.costs&&e.sparseCosts)throw Error(`${n} costs and sparseCosts are mutually exclusive`);if(!e.costs&&!e.sparseCosts)throw Error(`${n} requires costs or sparseCosts`);if(c(n,`facilityCount`,e.facilityCount),e.costs){if(o(e.costs,[`float32`],`${n} costs`),e.costs.length<e.facilityCount||e.costs.length%e.facilityCount!==0)throw Error(`${n} costs length must be a positive multiple of facilityCount`);this.demandCount=e.costs.length/e.facilityCount}else{let t=e.sparseCosts;if(o(t.offsets,[`uint32`],`${n} sparseCosts.offsets`),o(t.facilities,[`uint32`],`${n} sparseCosts.facilities`),o(t.costs,[`float32`],`${n} sparseCosts.costs`),t.offsets.length<2)throw Error(`${n} sparseCosts.offsets must hold demand rows + 1 offsets`);if(t.facilities.length<1||t.facilities.length!==t.costs.length)throw Error(`${n} sparseCosts.facilities and costs must be equally long and not empty`);this.demandCount=t.offsets.length-1}c(n,`maximumFacilityCount`,e.maximumFacilityCount,1,e.facilityCount);let i=e.maximumSwapIterations;if(i!==void 0&&c(n,`maximumSwapIterations`,i,0,L),e.operation===`set-covering`){if(i)throw Error(`${n} set-covering has no swap phase; omit maximumSwapIterations`);this.maximumSwapIterations=0}else this.maximumSwapIterations=i??100;if(e.demandWeights&&(o(e.demandWeights,[`float32`],`${n} demandWeights`),e.demandWeights.length!==this.demandCount))throw Error(`${n} demandWeights length must equal the demand row count`);if(e.operation!==`p-median`){if(!e.coverageDistance)throw Error(`${n} coverageDistance is required for ${e.operation}`);if(o(e.coverageDistance,[`float32`],`${n} coverageDistance`),e.coverageDistance.length<1)throw Error(`${n} coverageDistance must hold one value`)}else e.coverageDistance&&o(e.coverageDistance,[`float32`],`${n} coverageDistance`);if(o(r.selectedFacilities,[`uint32`],`${n} output.selectedFacilities`),o(r.assignments,[`uint32`],`${n} output.assignments`),o(r.status,[`uint32`],`${n} output.status`),o(r.objective,[`float32`],`${n} output.objective`),r.selectedFacilities.length<e.facilityCount)throw Error(`${n} output.selectedFacilities must hold one flag per facility`);if(r.assignments.length<this.demandCount)throw Error(`${n} output.assignments must hold one row per demand`);if(r.objective.length<1)throw Error(`${n} output.objective must hold one value`);if(r.status.length<I.length)throw Error(`${n} output.status must hold ${I.length} words`);if(r.assignedCosts&&(o(r.assignedCosts,[`float32`],`${n} output.assignedCosts`),r.assignedCosts.length<this.demandCount))throw Error(`${n} output.assignedCosts must hold one row per demand`);t(n,r,[[r.selectedFacilities,r.assignments,r.objective,r.status],{read:[r.selectedFacilities],write:[r.assignments,r.assignedCosts]}]),p(n,[r.selectedFacilities,r.assignments,r.assignedCosts,r.objective,r.status],[...this.getCostViews(),e.demandWeights,e.coverageDistance]);let a=this.maximumSwapIterations>0,s=e.sparseCosts?2+Number(e.demandWeights!==void 0||e.operation!==`p-median`)+3+R:1;this.estimatedCommandNodeCount=2+Number(!!e.sparseCosts)+1+s+Number(e.operation===`p-median`||a&&e.demandWeights!==void 0)+3*e.maximumFacilityCount+(a?1+6*this.maximumSwapIterations:0),l(n,this.estimatedCommandNodeCount)}getCostViews(){let{costs:e,sparseCosts:t}=this.props;return e?[e]:[t.offsets,t.facilities,t.costs]}getSwapNodes(e,t){let{id:r,props:o,demandCount:s}=this,{output:c,facilityCount:l}=o,u=o.sparseCosts,{declarations:d,swapState:f,weightBinding:p,coverageBinding:m,demandWeightExpression:h}=t,g=o.maximumFacilityCount,_=g*l,v=o.operation===`maximum-coverage`,y=[{name:`scoreScale`,view:t.scoreScale,type:`u32`,access:`read`}],b=v?`let costScoreExponent = 0;
  let reachScoreExponent = getAllocationWeightExponent(scoreScale[scoreScaleOffset + 2u]);
  let costScoreLimit = getUnreachableCost();`:`// High bit requests the full input scale when a positive-weight row is unreachable.
  var requiredScale = 0u;
  for (var demand = localInvocationIndex; demand < DEMAND_COUNT; demand += SWAP_WORKGROUP_SIZE) {
    let weight = bitcast<u32>(${h});
    let nearest = demandData[demandDataOffset + demand * 3u];
    if (weight != 0u && weight < 0x7f800000u) {
      var rowScale = 0u;
      if ((nearest & 0x7fffffffu) >= 0x7f800000u) {
        rowScale = 0x80000000u | scoreScale[scoreScaleOffset + 1u];
      } else if ((nearest & 0x7fffffffu) != 0u) {
        // Extra demand-count headroom bounds a whole sparse loss/extra sum of capped costs.
        rowScale = u32(max(0, getAllocationValueExponent(nearest) + getAllocationValueExponent(weight) + ${2*Math.ceil(Math.log2(s))} - 120));
      }
      requiredScale = max(requiredScale, rowScale);
    }
  }
  sharedScale[localInvocationIndex] = requiredScale;
  workgroupBarrier();
  for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (localInvocationIndex < stride) {
      sharedScale[localInvocationIndex] = max(sharedScale[localInvocationIndex], sharedScale[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  let costScoreExponent = i32(sharedScale[0] & 0x7fffffffu);
  let reachScoreExponent = getAllocationWeightExponent(scoreScale[scoreScaleOffset + 2u]);
  var sumObjective = 0.0;
  for (var demand = localInvocationIndex; demand < DEMAND_COUNT; demand += SWAP_WORKGROUP_SIZE) {
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    if (isReachable(nearest)) {
      sumObjective += getAllocationScoreCost(nearest, ${h}, costScoreExponent);
    }
  }
  sharedB[localInvocationIndex] = sumObjective;
  workgroupBarrier();
  for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (localInvocationIndex < stride) { sharedB[localInvocationIndex] += sharedB[localInvocationIndex + stride]; }
    workgroupBarrier();
  }
  let currentObjective = sharedB[0];
  var costScoreLimit = getUnreachableCost();
  if ((sharedScale[0] & 0x80000000u) == 0u) { costScoreLimit = 2.0 * currentObjective; }
  // All lanes must finish reading the objective before the next reduction reuses sharedB.
  workgroupBarrier();`,x=(t,i,a)=>n(e,`${r}-swap-${t}`,i,a),S=x(`demand-data`,`uint32`,s*3),C=x(`selected-list`,`uint32`,g),w=x(`slot-of-facility`,`uint32`,l),T=x(`totals`,`float32`,3*g+5),E=x(`pair-scores`,`float32`,2*_),D=v?`coverageDistance[coverageDistanceOffset]`:`0.0`,O=u?g+1:1,M=i(e,{id:`${r}-swap-dispatch`,operation:P,shapes:[[1],[s],[O*F,F],[l],[F,F]]}),N=M.binding,I=`if (cost < nearestCost) {
      secondCost = nearestCost;
      nearestCost = cost;
      nearestSlot = slot;
    } else if (cost < secondCost) {
      secondCost = cost;
    }
    if (isReachable(cost) && cost <= threshold) {
      coveringCount++;
      coveringSlot = slot;
    }`,L=`for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
    let facility = selectedList[selectedListOffset + slot];
    if (facility == NONE) { break; }
    let cost = ${u?`getSparseRowCost(demand, facility)`:`costs[costsOffset + demand * FACILITY_COUNT + facility]`};
    ${I}
  }`,R=u?`let rowStart = rowOffsets[rowOffsetsOffset + demand];
  let rowEnd = rowOffsets[rowOffsetsOffset + demand + 1u];
  if (rowEnd - rowStart <= SWAP_SLOTS) {
    for (var entry = rowStart; entry < rowEnd; entry++) {
      let slot = slotOfFacility[slotOfFacilityOffset + rowFacilities[rowFacilitiesOffset + entry]];
      if (slot == NONE) { continue; }
      let cost = rowCosts[rowCostsOffset + entry];
      ${I}
    }
  } else {
    ${L}
  }`:L,z=[...u?[...k(u),{name:`slotOfFacility`,view:w,type:`u32`,access:`read`}]:[{name:`costs`,view:o.costs,type:`f32`,access:`read`}],{name:`selectedList`,view:C,type:`u32`,access:`read`},{name:`demandData`,view:S,type:`u32`,access:`read_write`},...m],B=e=>u?`let columnEnd = columnOffsets[columnOffsetsOffset + candidate + 1u];
  for (var entry = columnOffsets[columnOffsetsOffset + candidate]; entry < columnEnd; entry++) {
    let demand = columnEntries[columnEntriesOffset + 2u * entry];
    let c = bitcast<f32>(columnEntries[columnEntriesOffset + 2u * entry + 1u]);
    ${e}
  }`:`for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
    let c = costs[costsOffset + demand * FACILITY_COUNT + candidate];
    ${e}
  }`,V=[...u?A(t.transpose):[{name:`costs`,view:o.costs,type:`f32`,access:`read`}],{name:`slotOfFacility`,view:w,type:`u32`,access:`read`},{name:`demandData`,view:S,type:`u32`,access:`read`},{name:`swapTotals`,view:T,type:`f32`,access:`read`},...u?[]:[{name:`selectedList`,view:C,type:`u32`,access:`read`}],{name:`pairScores`,view:E,type:`f32`,access:`read_write`},...p,...m],H=B(v?`let weight = getAllocationScoreCost(1.0, ${h}, reachScoreExponent);
    if (isReachable(c) && c <= threshold) {
      let count = demandData[demandDataOffset + demand * 3u];
      let sole = demandData[demandDataOffset + demand * 3u + 2u];
      if (count == 0u) {
        gainCover += weight;
      } else if (sole != NONE) {
        pairScores[pairScoresOffset + 2u * (sole * FACILITY_COUNT + candidate)] += weight;
      }
    }`:`let weight = ${h};
    let scoreWeight = getAllocationScoreCost(1.0, weight, reachScoreExponent);
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
    let slot = demandData[demandDataOffset + demand * 3u + 2u];
    if (c < nearest) {
      // gain(i): the candidate beats the nearest selected facility (or reaches an unreached row).
      if (isReachable(nearest)) {
        gainCost += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit) - getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit);
      } else {
        gainReach += scoreWeight;
        gainCost -= getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit);
      }
    }
    if (slot != NONE && c < second) {
      // extra(i, r): the row loses its nearest facility r but the candidate beats the second.
      var extraReach = 0.0;
      var extraCost = 0.0;
      if (isReachable(second)) {
        extraCost = getAllocationSwapCost(second, weight, costScoreExponent, costScoreLimit) - max(getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit), getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit));
      } else {
        extraReach = scoreWeight;
        // min(nearest, c) - c - nearest = -max(nearest, c), without an overflowing intermediate.
        extraCost = -max(getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit), getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit));
      }
      let pair = pairScoresOffset + 2u * (slot * FACILITY_COUNT + candidate);
      pairScores[pair] += extraReach;
      pairScores[pair + 1u] += extraCost;
    }`),U=B(v?`let weight = getAllocationScoreCost(1.0, ${h}, reachScoreExponent);
    let count = demandData[demandDataOffset + demand * 3u];
    let sole = demandData[demandDataOffset + demand * 3u + 2u];
    if (count == 0u) {
      if (isReachable(c) && c <= threshold) { gainCover += weight; }
    } else if (sole != NONE && (!isReachable(c) || c > threshold)) {
      // The only covering facility leaves and the candidate does not cover the row.
      pairScores[pairScoresOffset + 2u * (sole * FACILITY_COUNT + candidate)] -= weight;
    }`:`let weight = ${h};
    let scoreWeight = getAllocationScoreCost(1.0, weight, reachScoreExponent);
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
    let slot = demandData[demandDataOffset + demand * 3u + 2u];
    if (c < nearest) {
      if (isReachable(nearest)) {
        gainCost += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit) - getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit);
      } else {
        gainReach += scoreWeight;
        gainCost -= getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit);
      }
    } else if (slot != NONE) {
      // The row loses its nearest facility and falls back to the cheaper of the second and the
      // candidate (c >= nearest here, so the candidate does not beat the nearest).
      let kept = min(second, c);
      if (kept > nearest) {
        let pair = pairScoresOffset + 2u * (slot * FACILITY_COUNT + candidate);
        if (isReachable(kept)) {
          pairScores[pair + 1u] += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit) - getAllocationSwapCost(kept, weight, costScoreExponent, costScoreLimit);
        } else {
          pairScores[pair] -= scoreWeight;
          pairScores[pair + 1u] += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit);
        }
      }
    }`),W=u?H:U,G=u?`swapTotals[swapTotalsOffset + 3u * slot]`:`0.0`,K=u?`swapTotals[swapTotalsOffset + 3u * slot + 2u] < 0.5`:`selectedList[selectedListOffset + slot] == NONE`,q=v?`let improvement = gainCover + ${G} + pairScores[pairScoresOffset + pair];
    if (improvement > SWAP_TOLERANCE * scale) {
      pairScores[pairScoresOffset + pair] = improvement;
      pairScores[pairScoresOffset + pair + 1u] = 0.0;
    } else {
      pairScores[pairScoresOffset + pair] = -1.0;
      pairScores[pairScoresOffset + pair + 1u] = -1.0;
    }`:`let reachGain = gainReach + ${G} + pairScores[pairScoresOffset + pair];
    let improvement = gainCost + ${u?`swapTotals[swapTotalsOffset + 3u * slot + 1u]`:`0.0`} + pairScores[pairScoresOffset + pair + 1u];
    var primary = -1.0;
    var secondary = -1.0;
    if (reachGain > SWAP_TOLERANCE * totalWeight) {
      primary = reachGain;
      secondary = improvement;
    } else if (abs(reachGain) <= SWAP_TOLERANCE * totalWeight && improvement > SWAP_TOLERANCE * scale) {
      primary = 0.0;
      secondary = improvement;
    }
    pairScores[pairScoresOffset + pair] = primary;
    pairScores[pairScoresOffset + pair + 1u] = secondary;`,J=v?`let weight = getAllocationScoreCost(1.0, ${h}, reachScoreExponent);
    if (removalTarget == SWAP_SLOTS) {
      sumA += weight;
    } else if (demandData[demandDataOffset + demand * 3u + 2u] == removalTarget) {
      sumA -= weight;
    }`:`let weight = ${h};
    let scoreWeight = getAllocationScoreCost(1.0, weight, reachScoreExponent);
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
    if (removalTarget == SWAP_SLOTS) {
      sumA += scoreWeight;
    } else if (demandData[demandDataOffset + demand * 3u + 2u] == removalTarget) {
      // loss(r): the row falls back to its second-nearest facility, or becomes unreachable.
      if (isReachable(second)) {
        sumB += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit) - getAllocationSwapCost(second, weight, costScoreExponent, costScoreLimit);
      } else {
        sumA -= scoreWeight;
        sumB += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit);
      }
    }`,Y=[a(e,{id:`${r}-swap-init`,operation:P,variant:`swap-init`,invocationCount:1,bindings:[{name:`swapState`,view:f,type:`u32`,access:`read_write`},N],declarations:d,body:`swapState[swapStateOffset] = 0u;
  swapState[swapStateOffset + 1u] = 0u;
  swapState[swapStateOffset + 2u] = NONE;
  ${M.armWGSL}`})];for(let t=0;t<this.maximumSwapIterations;t++){let n=`${r}-swap-${t}`;Y.push(a(e,{id:`${n}-list`,operation:P,variant:`swap-list`,invocationCount:1,...M.getGate(0),bindings:[{name:`selected`,view:c.selectedFacilities,type:`u32`,access:`read`},{name:`status`,view:c.status,type:`u32`,access:`read`},{name:`swapState`,view:f,type:`u32`,access:`read`},{name:`selectedList`,view:C,type:`u32`,access:`read_write`},{name:`slotOfFacility`,view:w,type:`u32`,access:`read_write`}],declarations:d,body:`for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
  selectedList[selectedListOffset + slot] = NONE;
}
let isActive = swapState[swapStateOffset] == 0u && status[statusOffset + 3u] == 0u;
var slot = 0u;
for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
  var facilitySlot = NONE;
  if (isActive && selected[selectedOffset + facility] != 0u && slot < SWAP_SLOTS) {
    selectedList[selectedListOffset + slot] = facility;
    facilitySlot = slot;
    slot++;
  }
  slotOfFacility[slotOfFacilityOffset + facility] = facilitySlot;
}`}),a(e,{id:`${n}-prepare`,operation:P,variant:`${o.operation}-swap-prepare${u?`-sparse`:``}`,invocationCount:s,...M.getGate(1),bindings:z,declarations:u?`${d}\n${j}`:d,body:`let demand = index;
  var nearestCost = getUnreachableCost();
  var secondCost = getUnreachableCost();
  var nearestSlot = NONE;
  var coveringCount = 0u;
  var coveringSlot = NONE;
  let threshold = ${D};
  ${R}
  demandData[demandDataOffset + demand * 3u] = ${v?`coveringCount`:`bitcast<u32>(nearestCost)`};
  demandData[demandDataOffset + demand * 3u + 1u] = ${v?`0u`:`bitcast<u32>(secondCost)`};
  demandData[demandDataOffset + demand * 3u + 2u] = ${v?`select(NONE, coveringSlot, coveringCount == 1u)`:`nearestSlot`};`}),a(e,{id:`${n}-${u?`removal`:`totals`}`,operation:P,variant:`${o.operation}-swap-${u?`removal`:`totals`}`,invocationCount:O*F,workgroupSize:F,guardIndex:!1,...M.getGate(2),bindings:[{name:`demandData`,view:S,type:`u32`,access:`read`},{name:`selectedList`,view:C,type:`u32`,access:`read`},{name:`swapTotals`,view:T,type:`f32`,access:`read_write`},...y,...p],declarations:`${d}
const SWAP_WORKGROUP_SIZE: u32 = ${F}u;
var<workgroup> sharedA: array<f32, ${F}>;
var<workgroup> sharedB: array<f32, ${F}>;
${v?``:`var<workgroup> sharedScale: array<u32, ${F}>;`}`,body:`let removalTarget = ${u?`workgroupIndex`:`SWAP_SLOTS`};
  ${b}
  var sumA = 0.0;
  var sumB = ${v?`0.0`:`select(0.0, currentObjective, removalTarget == SWAP_SLOTS && localInvocationIndex == 0u)`};
  for (var demand = localInvocationIndex; demand < DEMAND_COUNT; demand += SWAP_WORKGROUP_SIZE) {
    ${J}
  }
  sharedA[localInvocationIndex] = sumA;
  sharedB[localInvocationIndex] = sumB;
  workgroupBarrier();
  for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedA[localInvocationIndex] += sharedA[localInvocationIndex + stride];
      sharedB[localInvocationIndex] += sharedB[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    if (removalTarget == SWAP_SLOTS) {
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS] = ${v?`sharedA[0]`:`sharedB[0]`};
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 1u] = sharedA[0];
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 2u] = f32(costScoreExponent);
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 3u] = f32(reachScoreExponent);
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 4u] = costScoreLimit;
    } else {
      swapTotals[swapTotalsOffset + 3u * removalTarget] = sharedA[0];
      swapTotals[swapTotalsOffset + 3u * removalTarget + 1u] = sharedB[0];
      swapTotals[swapTotalsOffset + 3u * removalTarget + 2u] =
        select(0.0, 1.0, selectedList[selectedListOffset + removalTarget] != NONE);
    }
  }`}),a(e,{id:`${n}-score`,operation:P,variant:`${o.operation}-swap-score${u?`-sparse`:``}`,invocationCount:l,...M.getGate(3),bindings:V,declarations:d,body:`let candidate = index;
  let costScoreExponent = i32(swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 2u]);
  let reachScoreExponent = i32(swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 3u]);
  let costScoreLimit = swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 4u];
  let isCandidate = slotOfFacility[slotOfFacilityOffset + candidate] == NONE;
  let threshold = ${D};
  for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
    let pair = pairScoresOffset + 2u * (slot * FACILITY_COUNT + candidate);
    pairScores[pair] = select(-1.0, 0.0, isCandidate);
    pairScores[pair + 1u] = select(-1.0, 0.0, isCandidate);
  }
  if (!isCandidate) {
    return;
  }
  var gainReach = 0.0;
  var gainCost = 0.0;
  var gainCover = 0.0;
  ${W}
  let scale = swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS];
  let totalWeight = swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 1u];
  for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
    let pair = 2u * (slot * FACILITY_COUNT + candidate);
    if (${K}) {
      pairScores[pairScoresOffset + pair] = -1.0;
      pairScores[pairScoresOffset + pair + 1u] = -1.0;
      continue;
    }
    ${q}
  }`}),a(e,{id:`${n}-choose`,operation:P,variant:`swap-choose`,invocationCount:F,workgroupSize:F,guardIndex:!1,...M.getGate(4),bindings:[{name:`pairScores`,view:E,type:`f32`,access:`read`},{name:`swapState`,view:f,type:`u32`,access:`read_write`}],declarations:`${d}
const PAIR_COUNT: u32 = ${_}u;
const SWAP_WORKGROUP_SIZE: u32 = ${F}u;
var<workgroup> sharedScore: array<f32, ${F}>;
var<workgroup> sharedSecondary: array<f32, ${F}>;
var<workgroup> sharedPair: array<u32, ${F}>;
fn isBetterPair(
  score: f32, secondary: f32, pair: u32, bestScore: f32, bestSecondary: f32, bestPair: u32
) -> bool {
  if (score != bestScore) { return score > bestScore; }
  if (secondary != bestSecondary) { return secondary > bestSecondary; }
  return pair < bestPair;
}`,body:`var bestPair = NONE;
var bestScore = -1.0;
var bestSecondary = -1.0;
if (swapState[swapStateOffset] == 0u) {
  for (var pair = localInvocationIndex; pair < PAIR_COUNT; pair += SWAP_WORKGROUP_SIZE) {
    let score = pairScores[pairScoresOffset + 2u * pair];
    let secondary = pairScores[pairScoresOffset + 2u * pair + 1u];
    if (isBetterPair(score, secondary, pair, bestScore, bestSecondary, bestPair)) {
      bestPair = pair;
      bestScore = score;
      bestSecondary = secondary;
    }
  }
}
sharedScore[localInvocationIndex] = bestScore;
sharedSecondary[localInvocationIndex] = bestSecondary;
sharedPair[localInvocationIndex] = bestPair;
workgroupBarrier();
for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
  if (localInvocationIndex < stride &&
      isBetterPair(
        sharedScore[localInvocationIndex + stride],
        sharedSecondary[localInvocationIndex + stride],
        sharedPair[localInvocationIndex + stride],
        sharedScore[localInvocationIndex],
        sharedSecondary[localInvocationIndex],
        sharedPair[localInvocationIndex])) {
    sharedScore[localInvocationIndex] = sharedScore[localInvocationIndex + stride];
    sharedSecondary[localInvocationIndex] = sharedSecondary[localInvocationIndex + stride];
    sharedPair[localInvocationIndex] = sharedPair[localInvocationIndex + stride];
  }
  workgroupBarrier();
}
if (localInvocationIndex == 0u) {
  // A qualifying pair has a non-negative primary score; -1 marks "no swap".
  swapState[swapStateOffset + 2u] = select(NONE, sharedPair[0], sharedScore[0] >= 0.0);
}`}),a(e,{id:`${n}-apply`,operation:P,variant:`swap-apply`,invocationCount:1,bindings:[{name:`selected`,view:c.selectedFacilities,type:`u32`,access:`read_write`},{name:`selectedList`,view:C,type:`u32`,access:`read`},{name:`swapState`,view:f,type:`u32`,access:`read_write`},N],declarations:d,body:`if (swapState[swapStateOffset] != 0u) {
  return;
}
let pair = swapState[swapStateOffset + 2u];
if (pair == NONE) {
  swapState[swapStateOffset] = 1u;
  ${M.disarmWGSL}
  return;
}
selected[selectedOffset + selectedList[selectedListOffset + pair / FACILITY_COUNT]] = 0u;
selected[selectedOffset + pair % FACILITY_COUNT] = 1u;
swapState[swapStateOffset + 1u] = swapState[swapStateOffset + 1u] + 1u;`}))}return Y}getSparseSetupNodes(e,t){let{id:n,props:r,demandCount:i}=this,o=r.sparseCosts,{declarations:s,weightBinding:c,coverageBinding:l,needsCoverage:u}=t,d=[...M(e,{id:n,operation:P,sparse:o,declarations:s,status:r.output.status})];(r.demandWeights||u)&&d.push(a(e,{id:`${n}-validate-inputs`,operation:P,variant:`validate-sparse-inputs`,invocationCount:Math.max(i,1),bindings:[...c,...l,{name:`status`,view:r.output.status,type:`atomic<u32>`,access:`read_write`}],declarations:s,body:K(r.demandWeights!==void 0,u)}));let{nodes:f,transpose:p}=N(e,{id:n,operation:P,sparse:o,facilityCount:r.facilityCount,declarations:s});return d.push(...f),{nodes:d,transpose:p}}getCommandNodes(t){let{id:o,props:l,demandCount:u}=this,{output:p,facilityCount:m}=l;if(f(o,t,[...this.getCostViews(),l.demandWeights,l.coverageDistance,p.selectedFacilities,p.assignments,p.assignedCosts,p.objective,p.status]),e(t,o,{...Object.fromEntries(this.getCostViews().map((e,t)=>[`costs[${t}]`,e])),demandWeights:l.demandWeights,coverageDistance:l.coverageDistance,...p}),c(o,`facilityCount`,l.facilityCount,1,V(t.device)),l.costs?s(o,`costs rows`,l.costs.length,H(t.device)):s(o,`sparseCosts entries`,l.sparseCosts.facilities.length,W(t.device)),s(o,`demand rows`,u,U(t.device)),this.maximumSwapIterations>0){let e=G(t.device);if(l.maximumFacilityCount*l.facilityCount>e)throw Error(`${o} swap phase needs maximumFacilityCount * facilityCount <= ${e}; reduce them or set maximumSwapIterations: 0`)}for(let[e,n]of[[`demand scratch`,u*(this.maximumSwapIterations>0?12:4)],[`facility scores`,m*8],[`swap totals`,this.maximumSwapIterations>0?(3*l.maximumFacilityCount+5)*4:0],[`swap pair scores`,this.maximumSwapIterations>0?8*l.maximumFacilityCount*m:0],[`sparse transpose entries`,l.sparseCosts?l.sparseCosts.facilities.length*8:0],[`sparse transpose offsets`,l.sparseCosts?(m+1)*4:0],[`swap dispatch`,this.maximumSwapIterations>0?z:0]])d(t.device,o,e,n,`reduce demand or facility counts`);let h=n(t,`${o}-best-costs`,`float32`,u),g=n(t,`${o}-facility-scores`,`float32`,2*m),_=n(t,`${o}-chosen-facility`,`uint32`,3),v=n(t,`${o}-demand-objectives`,`float32`,u),y=n(t,`${o}-uncovered`,`uint32`,u),b=l.operation===`p-median`?0:l.operation===`maximum-coverage`?1:2,x=b!==0,S=this.maximumSwapIterations,C=l.sparseCosts,w=C?k(C):[{name:`costs`,view:l.costs,type:`f32`,access:`read`}],T=`const DEMAND_COUNT: u32 = ${u}u;
const FACILITY_COUNT: u32 = ${m}u;${C?`\nconst SPARSE_ENTRY_COUNT: u32 = ${C.facilities.length}u;`:``}
const MAXIMUM_FACILITIES: u32 = ${l.maximumFacilityCount}u;
const MODE: u32 = ${b}u;
const NONE: u32 = 0xffffffffu;
${r}
fn isInvalid(value: f32) -> bool { return isNanBits(value) || value < 0.0; }
// Non-finite constant-expression bitcasts are invalid WGSL; a function-local var is runtime.
fn getUnreachableCost() -> f32 {
  var bits = 0x7f800000u;
  return bitcast<f32>(bits);
}
// Inspect the representation explicitly: every finite nonnegative cost is reachable.
fn isReachable(cost: f32) -> bool { return (bitcast<u32>(cost) & 0x7fffffffu) < 0x7f800000u; }
// Leave ordinary scores in caller units. At overflow risk, all candidates share the same
// binary exponent, with headroom for the complete demand sum and finite weighted products.
fn getAllocationValueExponent(bits: u32) -> i32 {
  let exponent = i32((bits >> 23u) & 255u);
  if (exponent != 0) { return exponent - 127; }
  return 31 - i32(countLeadingZeros(bits & 0x7fffffu)) - 149;
}
fn getAllocationWeightExponent(weightMaximum: u32) -> i32 {
  return max(0, getAllocationValueExponent(weightMaximum) + ${Math.ceil(Math.log2(u))} - 120);
}
fn getAllocationProductExponent(cost: u32, weight: u32) -> u32 {
  // Exclude zero products; independent maxima can belong to different, noncontributing rows.
  if (cost == 0u || weight == 0u || cost >= 0x7f800000u || weight >= 0x7f800000u) { return 0u; }
  return u32(max(0, getAllocationValueExponent(cost) + getAllocationValueExponent(weight) + ${Math.ceil(Math.log2(u))} - 120));
}
fn getAllocationScoreCost(cost: f32, weight: f32, exponent: i32) -> f32 {
  if (!isReachable(weight)) { return cost * weight; }
  if (exponent == 0) { return cost * weight; }
  if (cost == 0.0 || weight == 0.0) { return 0.0; }
  let costParts = frexp(cost);
  let weightParts = frexp(weight);
  return ldexp(costParts.fract * weightParts.fract, costParts.exp + weightParts.exp - exponent);
}
// A cost above twice the current objective cannot improve a fully reachable solution. Clip
// weighted costs before differencing, so a huge sparse loss and its extra term cannot erase
// the useful gain. Check the exponent before ldexp, avoiding an overflowing intermediate.
fn getAllocationSwapCost(cost: f32, weight: f32, exponent: i32, limit: f32) -> f32 {
  if (!isReachable(limit)) { return getAllocationScoreCost(cost, weight, exponent); }
  if (limit == 0.0 || cost == 0.0 || weight == 0.0) { return 0.0; }
  let costParts = frexp(cost);
  let weightParts = frexp(weight);
  let productExponent = costParts.exp + weightParts.exp - exponent;
  if (productExponent > getAllocationValueExponent(bitcast<u32>(limit)) + 2) { return limit; }
  return min(ldexp(costParts.fract * weightParts.fract, productExponent), limit);
}`,E=`${T}
const MAXIMUM_SWAPS: u32 = ${S}u;
const SWAP_SLOTS: u32 = ${l.maximumFacilityCount}u;
const SWAP_TOLERANCE: f32 = ${B};`,D=S>0?n(t,`${o}-swap-state`,`uint32`,3):void 0,O=l.demandWeights?[{name:`demandWeights`,view:l.demandWeights,type:`f32`,access:`read`}]:[],M=x?[{name:`coverageDistance`,view:l.coverageDistance,type:`f32`,access:`read`}]:[],N=l.demandWeights?`demandWeights[demandWeightsOffset + demand]`:`1.0`,L=x?i(t,{id:`${o}-greedy-dispatch`,operation:P,shapes:[[m],[u]]}):null,R=L?[L.binding]:[],q=C?this.getSparseSetupNodes(t,{declarations:T,weightBinding:O,coverageBinding:M,needsCoverage:x}):void 0,J=[a(t,{id:`${o}-initialize`,operation:P,variant:`initialize`,invocationCount:Math.max(m,u,I.length),bindings:[{name:`selected`,view:p.selectedFacilities,type:`u32`,access:`read_write`},{name:`assignments`,view:p.assignments,type:`u32`,access:`read_write`},{name:`bestCosts`,view:h,type:`f32`,access:`read_write`},{name:`objective`,view:p.objective,type:`f32`,access:`read_write`},{name:`status`,view:p.status,type:`u32`,access:`read_write`},{name:`chosen`,view:_,type:`u32`,access:`read_write`},...R],declarations:T,body:`if (index < FACILITY_COUNT) {
    selected[selectedOffset + index] = 0u;
  }
  if (index < DEMAND_COUNT) {
    bestCosts[bestCostsOffset + index] = getUnreachableCost();
    assignments[assignmentsOffset + index] = NONE;
  }
  if (index < ${I.length}u) {
    status[statusOffset + index] = select(0u, DEMAND_COUNT, index == 1u);
  }
  if (index == 0u) {
    objective[objectiveOffset] = 0.0;
    // Any value but NONE: the first choose step is not stopped.
    chosen[chosenOffset] = 0u;
    chosen[chosenOffset + 1u] = 0u;
    chosen[chosenOffset + 2u] = ${l.demandWeights?`0u`:`0x3f800000u`};
    ${L?L.armWGSL:``}
  }`}),...q?q.nodes:[a(t,{id:`${o}-validate`,operation:P,variant:`validate`,invocationCount:Math.max(l.costs.length,u,1),bindings:[{name:`costs`,view:l.costs,type:`f32`,access:`read`},...O,...M,{name:`status`,view:p.status,type:`atomic<u32>`,access:`read_write`}],declarations:T,body:`if (index < DEMAND_COUNT * FACILITY_COUNT && isInvalid(costs[costsOffset + index])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }
  ${K(l.demandWeights!==void 0,x)}`})]];if(b===0||S>0&&l.demandWeights!==void 0){let e=C?C.costs.length:l.costs.length;J.push(a(t,{id:`${o}-score-scale`,operation:P,variant:`score-scale`,invocationCount:b===0?Math.max(e,u):u,bindings:[...b===0?C?[{name:`columnEntries`,view:q.transpose.columnEntries,type:`u32`,access:`read`}]:[{name:`costs`,view:l.costs,type:`u32`,access:`read`}]:[],...O,{name:`chosen`,view:_,type:`atomic<u32>`,access:`read_write`}],declarations:T,body:`${b===0?`if (index < ${e}u) {
    let demand = ${C?`columnEntries[columnEntriesOffset + 2u * index]`:`index / FACILITY_COUNT`};
    let bits = ${C?`columnEntries[columnEntriesOffset + 2u * index + 1u]`:`costs[costsOffset + index]`};
    if (demand < DEMAND_COUNT) {
      let weight = bitcast<u32>(${N});
      atomicMax(&chosen[chosenOffset + 1u], getAllocationProductExponent(bits, weight));
    }
  }`:``}
  ${l.demandWeights?`if (index < DEMAND_COUNT) {
    let bits = bitcast<u32>(demandWeights[demandWeightsOffset + index]);
    if (bits < 0x7f800000u) { atomicMax(&chosen[chosenOffset + 2u], bits); }
  }`:``}`}))}for(let e=0;e<l.maximumFacilityCount;e++)J.push(a(t,{id:`${o}-score-${e}`,operation:P,variant:`${l.operation}-score`,invocationCount:m,...L?L.getGate(0):{},bindings:[...q?A(q.transpose):[{name:`costs`,view:l.costs,type:`f32`,access:`read`}],{name:`selected`,view:p.selectedFacilities,type:`u32`,access:`read`},{name:`bestCosts`,view:h,type:`f32`,access:`read`},{name:`scores`,view:g,type:`f32`,access:`read_write`},{name:`status`,view:p.status,type:`u32`,access:`read`},...b===0?[{name:`chosen`,view:_,type:`u32`,access:`read`}]:[],...O,...M],declarations:T,body:`let facility = index;
  let unavailableScore = select(-1.0, getUnreachableCost(), MODE == 0u);
  if (status[statusOffset + 3u] != 0u || selected[selectedOffset + facility] != 0u) {
    scores[scoresOffset + 2u * facility] = unavailableScore;
    scores[scoresOffset + 2u * facility + 1u] = unavailableScore;
    return;
  }
  let threshold = ${x?`coverageDistance[coverageDistanceOffset]`:`0.0`};
  let costScoreExponent = ${b===0?`i32(chosen[chosenOffset + 1u])`:`0`};
  let reachScoreExponent = ${b===0?`getAllocationWeightExponent(chosen[chosenOffset + 2u])`:`0`};
  // p-median scores are lexicographic: (demand weight left unreachable, weighted cost of the
  // reachable rows). Coverage modes use only the first value: maximum coverage the demand weight
  // newly covered, set covering the number of rows newly covered (a row of weight 0 must still
  // be covered), counted in u32 so the count is exact.
  var score = 0.0;
  var reachableCost = 0.0;
  var newlyCoveredRowCount = 0u;
  ${q?`// Sparse: the facility only changes the rows it lists, so the score is the negated gain
  // (reachable weight won, cost saved), which orders candidates like the dense totals.
  let columnEnd = columnOffsets[columnOffsetsOffset + facility + 1u];
  for (var entry = columnOffsets[columnOffsetsOffset + facility]; entry < columnEnd; entry++) {
    let demand = columnEntries[columnEntriesOffset + 2u * entry];
    let candidate = bitcast<f32>(columnEntries[columnEntriesOffset + 2u * entry + 1u]);
    let current = bestCosts[bestCostsOffset + demand];
    if (MODE == 0u) {
      if (candidate < current) {
        if (isReachable(current)) {
          reachableCost += getAllocationScoreCost(current - candidate, ${N}, costScoreExponent);
        } else {
          score += getAllocationScoreCost(1.0, ${N}, reachScoreExponent);
          reachableCost -= getAllocationScoreCost(candidate, ${N}, costScoreExponent);
        }
      }
    } else if ((!isReachable(current) || current > threshold) && isReachable(candidate) && candidate <= threshold) {
      score += ${N};
      newlyCoveredRowCount += 1u;
    }
  }
  if (MODE == 0u) {
    score = -score;
    reachableCost = -reachableCost;
  }`:`for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
    let current = bestCosts[bestCostsOffset + demand];
    let candidate = costs[costsOffset + demand * FACILITY_COUNT + facility];
    if (MODE == 0u) {
      let next = min(current, candidate);
      if (isReachable(next)) {
        reachableCost += getAllocationScoreCost(next, ${N}, costScoreExponent);
      } else {
        score += getAllocationScoreCost(1.0, ${N}, reachScoreExponent);
      }
    } else if ((!isReachable(current) || current > threshold) && isReachable(candidate) && candidate <= threshold) {
      score += ${N};
      newlyCoveredRowCount += 1u;
    }
  }`}
  if (MODE == 2u) {
    score = f32(newlyCoveredRowCount);
  }
  scores[scoresOffset + 2u * facility] = score;
  scores[scoresOffset + 2u * facility + 1u] = reachableCost;`}),a(t,{id:`${o}-choose-${e}`,operation:P,variant:`${l.operation}-choose`,invocationCount:F,workgroupSize:F,guardIndex:!1,bindings:[{name:`selected`,view:p.selectedFacilities,type:`u32`,access:`read_write`},{name:`scores`,view:g,type:`f32`,access:`read`},{name:`chosen`,view:_,type:`u32`,access:`read_write`},{name:`status`,view:p.status,type:`u32`,access:`read_write`},...R],declarations:`${T}
const CHOOSE_WORKGROUP_SIZE: u32 = ${F}u;
var<workgroup> sharedScore: array<f32, ${F}>;
var<workgroup> sharedSecondary: array<f32, ${F}>;
var<workgroup> sharedFacility: array<u32, ${F}>;
// p-median: lowest (unreachable weight, reachable cost); coverage: highest score.
fn isBetterCandidate(
  score: f32, secondary: f32, facility: u32,
  bestScore: f32, bestSecondary: f32, bestFacility: u32
) -> bool {
  if (MODE == 0u) {
    if (score != bestScore) { return score < bestScore; }
    if (secondary != bestSecondary) { return secondary < bestSecondary; }
    return facility < bestFacility;
  }
  return score > bestScore || (score == bestScore && facility < bestFacility);
}`,body:`var bestFacility = NONE;
  var bestScore = select(-1.0, getUnreachableCost(), MODE == 0u);
  var bestSecondary = select(-1.0, getUnreachableCost(), MODE == 0u);
  let stopped = MODE != 0u && chosen[chosenOffset] == NONE;
  if (status[statusOffset + 3u] == 0u && !stopped) {
    for (var facility = localInvocationIndex; facility < FACILITY_COUNT; facility += CHOOSE_WORKGROUP_SIZE) {
      if (selected[selectedOffset + facility] != 0u) { continue; }
      let score = scores[scoresOffset + 2u * facility];
      let secondary = scores[scoresOffset + 2u * facility + 1u];
      if (isBetterCandidate(score, secondary, facility, bestScore, bestSecondary, bestFacility)) {
        bestFacility = facility;
        bestScore = score;
        bestSecondary = secondary;
      }
    }
  }
  sharedScore[localInvocationIndex] = bestScore;
  sharedSecondary[localInvocationIndex] = bestSecondary;
  sharedFacility[localInvocationIndex] = bestFacility;
  workgroupBarrier();
  for (var stride = CHOOSE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride &&
        isBetterCandidate(
          sharedScore[localInvocationIndex + stride],
          sharedSecondary[localInvocationIndex + stride],
          sharedFacility[localInvocationIndex + stride],
          sharedScore[localInvocationIndex],
          sharedSecondary[localInvocationIndex],
          sharedFacility[localInvocationIndex])) {
      sharedScore[localInvocationIndex] = sharedScore[localInvocationIndex + stride];
      sharedSecondary[localInvocationIndex] = sharedSecondary[localInvocationIndex + stride];
      sharedFacility[localInvocationIndex] = sharedFacility[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    bestFacility = sharedFacility[0];
    if (bestFacility != NONE && (MODE == 0u || sharedScore[0] > 0.0)) {
      selected[selectedOffset + bestFacility] = 1u;
      status[statusOffset] += 1u;
    } else {
      bestFacility = NONE;${L?`
      ${L.disarmWGSL}`:``}
    }
    chosen[chosenOffset] = bestFacility;
  }`}),a(t,{id:`${o}-update-${e}`,operation:P,variant:`${l.operation}-update`,invocationCount:u,...L?L.getGate(1):{},bindings:[...w,{name:`bestCosts`,view:h,type:`f32`,access:`read_write`},{name:`chosen`,view:_,type:`u32`,access:`read`}],declarations:C?`${T}\n${j}`:T,body:`let facility = chosen[chosenOffset];
  if (facility != NONE) {
    let candidate = ${C?`getSparseRowCost(index, facility)`:`costs[costsOffset + index * FACILITY_COUNT + facility]`};
    bestCosts[bestCostsOffset + index] = min(bestCosts[bestCostsOffset + index], candidate);
  }`}));S>0&&J.push(...this.getSwapNodes(t,{declarations:E,swapState:D,scoreScale:_,weightBinding:O,coverageBinding:M,demandWeightExpression:N,transpose:q?.transpose}));let Y=p.assignedCosts?`assignedCosts[assignedCostsOffset + demand] = select(bitcast<f32>(0x7f800000u | (demand & 0u)), assignedCost, assigned != NONE);`:``,X=`let threshold = ${x?`coverageDistance[coverageDistanceOffset]`:`0.0`};
  let isUncovered = assigned == NONE || (MODE != 0u && assignedCost > threshold);
  uncovered[uncoveredOffset + demand] = select(0u, 1u, isUncovered);
  let weight = ${N};
  // p-median sums the reachable rows only; unreachable rows are counted in uncoveredDemandCount.
  demandObjectives[demandObjectivesOffset + demand] = select(
    select(0.0, weight, !isUncovered),
    select(weight * assignedCost, 0.0, isUncovered),
    MODE == 0u
  );`,Z=p.assignedCosts?[{name:`assignedCosts`,view:p.assignedCosts,type:`f32`,access:`read_write`}]:[],Q=[{name:`demandObjectives`,view:v,type:`f32`,access:`read_write`},{name:`uncovered`,view:y,type:`u32`,access:`read_write`},...O,...M,...Z];if(C){let e=n(t,`${o}-assigned-cost-scratch`,`float32`,u);J.push(a(t,{id:`${o}-assign`,operation:P,variant:`${l.operation}-assign-sparse`,invocationCount:u,bindings:[...w,{name:`selected`,view:p.selectedFacilities,type:`u32`,access:`read`},{name:`assignments`,view:p.assignments,type:`u32`,access:`read_write`},{name:`assignedCostScratch`,view:e,type:`f32`,access:`read_write`}],declarations:T,body:`let demand = index;
  var assigned = NONE;
  var assignedCost = getUnreachableCost();
  let rowEnd = min(rowOffsets[rowOffsetsOffset + demand + 1u], SPARSE_ENTRY_COUNT);
  for (var entry = rowOffsets[rowOffsetsOffset + demand]; entry < rowEnd; entry++) {
    let facility = rowFacilities[rowFacilitiesOffset + entry];
    if (facility >= FACILITY_COUNT || selected[selectedOffset + facility] == 0u) { continue; }
    let cost = rowCosts[rowCostsOffset + entry];
    if (isReachable(cost) && (cost < assignedCost || (cost == assignedCost && facility < assigned))) {
      assignedCost = cost;
      assigned = facility;
    }
  }
  assignments[assignmentsOffset + demand] = assigned;
  assignedCostScratch[assignedCostScratchOffset + demand] = assignedCost;`}),a(t,{id:`${o}-objective`,operation:P,variant:`${l.operation}-objective`,invocationCount:u,bindings:[{name:`assignments`,view:p.assignments,type:`u32`,access:`read`},{name:`assignedCostScratch`,view:e,type:`f32`,access:`read`},...Q],declarations:T,body:`let demand = index;
  let assigned = assignments[assignmentsOffset + demand];
  let assignedCost = assignedCostScratch[assignedCostScratchOffset + demand];
  ${Y}
  ${X}`}))}else J.push(a(t,{id:`${o}-assign`,operation:P,variant:`${l.operation}-assign`,invocationCount:u,bindings:[...w,{name:`selected`,view:p.selectedFacilities,type:`u32`,access:`read`},{name:`assignments`,view:p.assignments,type:`u32`,access:`read_write`},...Q],declarations:T,body:`let demand = index;
  var assigned = NONE;
  var assignedCost = getUnreachableCost();
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
    if (selected[selectedOffset + facility] == 0u) { continue; }
    let cost = costs[costsOffset + demand * FACILITY_COUNT + facility];
    if (isReachable(cost) && (cost < assignedCost || (cost == assignedCost && facility < assigned))) {
      assignedCost = cost;
      assigned = facility;
    }
  }
  assignments[assignmentsOffset + demand] = assigned;
  ${Y}
  ${X}`}));return J.push(a(t,{id:`${o}-finalize`,operation:P,variant:`${l.operation}-finalize`,invocationCount:F,workgroupSize:F,guardIndex:!1,bindings:[{name:`demandObjectives`,view:v,type:`f32`,access:`read`},{name:`uncovered`,view:y,type:`u32`,access:`read`},{name:`objective`,view:p.objective,type:`f32`,access:`read_write`},{name:`status`,view:p.status,type:`u32`,access:`read_write`},...S>0?[{name:`swapState`,view:D,type:`u32`,access:`read`}]:[]],declarations:`${E}
const FINALIZE_WORKGROUP_SIZE: u32 = ${F}u;
var<workgroup> sharedObjective: array<f32, ${F}>;
var<workgroup> sharedUncovered: array<u32, ${F}>;`,body:`var uncoveredCount = 0u;
  var objectiveValue = 0.0;
  for (var demand = localInvocationIndex; demand < DEMAND_COUNT; demand += FINALIZE_WORKGROUP_SIZE) {
    uncoveredCount += uncovered[uncoveredOffset + demand];
    objectiveValue += demandObjectives[demandObjectivesOffset + demand];
  }
  sharedObjective[localInvocationIndex] = objectiveValue;
  sharedUncovered[localInvocationIndex] = uncoveredCount;
  workgroupBarrier();
  for (var stride = FINALIZE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedObjective[localInvocationIndex] += sharedObjective[localInvocationIndex + stride];
      sharedUncovered[localInvocationIndex] += sharedUncovered[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && status[statusOffset + 3u] == 0u) {
    let selectedCount = status[statusOffset];
    let totalUncovered = sharedUncovered[0];
    objective[objectiveOffset] = select(sharedObjective[0], f32(selectedCount), MODE == 2u);
    status[statusOffset + 1u] = totalUncovered;
    status[statusOffset + 2u] = select(
      0u,
      1u,
      select(selectedCount == MAXIMUM_FACILITIES, totalUncovered == 0u, MODE == 2u)
    );${S>0?`
    status[statusOffset + 4u] = swapState[swapStateOffset + 1u];
    status[statusOffset + 5u] = swapState[swapStateOffset];
    status[statusOffset + 6u] = select(
      0u,
      1u,
      swapState[swapStateOffset] == 0u && swapState[swapStateOffset + 1u] >= MAXIMUM_SWAPS
    );`:``}
  }`})),J}},J=w(O,`coverageDistance`),Y=_({parameters:O,caption:`Selected facilities (bright) and the demand locations assigned to each.`,createInputs:({random:e,world:t})=>v(e,t,8,120),build(e,t,n){let{network:r}=t,i=t.facilityNodes.length,a=y(e,r),o=b(e,a,t.demandNodes,r.nodeCount,t.facilityNodes),s=e.allocate(`coverage-distance`,`float32`,1),c=e.allocate(`selected`,`uint32`,i),l=e.allocate(`assignments`,`uint32`,t.demandNodes.length),u=e.allocate(`objective`,`float32`,1),d=e.allocate(`status`,`uint32`,7),f=n.operation===1;return e.graph.add(new q({operation:f?`maximum-coverage`:`p-median`,costs:o.view,facilityCount:i,maximumFacilityCount:Math.round(n.maximumFacilityCount),demandWeights:e.upload(`demand-weights`,`float32`,t.population).view,coverageDistance:f?s.view:void 0,output:{selectedFacilities:c.view,assignments:l.view,objective:u.view,status:d.view}})),{outputs:{...D(e,r),assignments:l,selected:c,objective:u,selectedIds:S(e,{id:`selected`,flags:c,count:i}),demands:e.upload(`demands`,`float32x2`,t.demandPositions),candidates:e.upload(`candidates`,`float32x2`,t.facilityPositions)},update:e=>{a.costLimit.write(Float32Array.of(4)),s.write(Float32Array.of(e.coverageDistance))}}},rebuildOn:[`operation`,`maximumFacilityCount`],interaction:{kind:`reach`,hint:`reach for coverage`,rest:({inputs:e,world:t,toUnit:n})=>n(T(e.facilityPositions,J.default,t)),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=0,o=1/0;for(let n=0;n<t.facilityNodes.length;n++){let r=Math.hypot(t.facilityPositions[n*2]-e.x,t.facilityPositions[n*2+1]-e.y);r<o&&(o=r,a=n)}let s=C(J,o),c=i(n.selected),l=i(n.objective),u=t.population.reduce((e,t)=>e+t,0),d=[{shape:`line`,from:[t.facilityPositions[a*2],t.facilityPositions[a*2+1]],to:[e.x,e.y],dashed:!0}];if(c)for(let e=0;e<c.length;e++)c[e]===1&&d.push({shape:`ring`,at:[t.facilityPositions[e*2],t.facilityPositions[e*2+1]],radius:s,dashed:!0});let f=l&&r.operation===1?` · covered ${Math.round(100*l[0]/u)}%`:``;return{values:{coverageDistance:s},glyphs:d,readout:`coverageDistance ${s.toFixed(2)}${f}`}}},draw:({outputs:e,theme:t})=>[E(e),{layer:`dots`,positions:e.demands,values:e.assignments,radius:6.5},...x(e.candidates,t,{ids:e.selectedIds,radius:11})]});export{Y as default};
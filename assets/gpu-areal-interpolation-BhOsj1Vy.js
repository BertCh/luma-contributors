import{n as e,t}from"./interpolation-view-validation-pYo9tqh3.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-gather-BJIYd708.js";import{o as i,t as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,S as s,f as c,u as l,v as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as f,o as p}from"./sorted-segment-sums-Dvmm6OdR.js";import{r as m}from"./wgsl-common-DfyXiYB0.js";import{a as h,d as ee,v as g}from"./spatial-context-DwZEd4EZ.js";import{i as _,t as v}from"./spherical-cell-area-3w3VAb_m.js";import{u as y}from"./raster-band-utils-BcTxc8_Y.js";import{t as b}from"./zone-anchor-nodes-Dhba8wh9.js";import{n as x,t as S}from"./key-pair-grouping-DgpiCtBP.js";import{t as te}from"./bounded-key-pair-sort-BTKR6XBC.js";import{c as C}from"./spatial-weights-CqWysstN.js";var w=`GPUArealInterpolation`;function T(e){return l(e,4,1)}var E=2**24-1,D=Object.freeze({...h,measure:`area`}),O=`${S}u`,k=8,A=4294967294,j=class{id;props;spatialContext;extensiveDenominator;intensiveDenominator;constructor(n){let r=n.id??`areal-interpolation`;this.id=r,this.props=n,this.spatialContext=g(r,n.spatialContext,D);let i=this.spatialContext.coordinateSpace===`longitude-latitude`,a=y(r,n.grid);if(i)_(r,n.parameters);else if(n.parameters)throw Error(`${r} parameters need a longitude-latitude spatialContext`);let{sourceZoneCount:l,targetZoneCount:u,output:d}=n;if(c(r,`sourceZoneCount`,l,1,4294967294),c(r,`targetZoneCount`,u,1,4294967294),o(n.sourceZones,[`uint32`],`${r} sourceZones`),o(n.targetZones,[`uint32`],`${r} targetZones`),n.sourceZones.length!==a||n.targetZones.length!==a)throw Error(`${r} sourceZones and targetZones must contain one row per grid cell`);if(n.cellWeights&&(o(n.cellWeights,[`float32`],`${r} cellWeights`),n.cellWeights.length!==a))throw Error(`${r} cellWeights length must equal the grid cell count`);let f=typeof n.denominator==`object`?n.denominator:void 0,p=typeof n.denominator==`string`?n.denominator:void 0;this.extensiveDenominator=f?.extensive??p??`zone`,this.intensiveDenominator=f?.intensive??p??`overlap`;for(let e of[this.extensiveDenominator,this.intensiveDenominator])if(e!==`zone`&&e!==`overlap`)throw Error(`${r} denominator must be 'zone' or 'overlap'`);if(n.mode!==void 0&&n.mode!==`extensive`&&n.mode!==`intensive`)throw Error(`${r} mode must be 'extensive' or 'intensive'`);if(n.sourceAnchorCells&&(o(n.sourceAnchorCells,[`uint32`],`${r} sourceAnchorCells`),n.sourceAnchorCells.length!==l))throw Error(`${r} sourceAnchorCells must have sourceZoneCount rows`);for(let[e,t]of[[`output.sourceAreas`,d.sourceAreas],[`output.sourceTransferredFractions`,d.sourceTransferredFractions]])if(t&&(o(t,[`float32`],`${r} ${e}`),t.length!==l))throw Error(`${r} ${e} must have sourceZoneCount rows`);if(C(r,d.weights,`output.weights`)!==u)throw Error(`${r} output.weights.offsets length must equal targetZoneCount + 1`);let m=d.weights.neighbors.length;(n.cellWeights||n.unweightedFastPath===!1||this.spatialContext.coordinateSpace===`longitude-latitude`)&&s(r,`pair slots (output.weights.neighbors rows, cells)`,Math.min(m,n.sourceZones.length),E);for(let[e,t]of[[`output.alternateWeights`,d.alternateWeights],[`output.areas`,d.areas]])if(t&&(o(t,[`float32`],`${r} ${e}`),t.length!==m))throw Error(`${r} ${e} length must equal output.weights.neighbors length`);for(let[e,t]of[[`output.status.overflow`,d.status.overflow],[`output.status.requiredPairCount`,d.status.requiredPairCount],[`output.status.invalidCount`,d.status.invalidCount]])if(t&&(o(t,[`uint32`],`${r} ${e}`),t.length<1))throw Error(`${r} ${e} must hold one uint32`);if(n.categories){let{sourceCategories:e,categoryCount:t}=n.categories,{categoryShares:i}=d;if(o(e,[`uint32`],`${r} categories.sourceCategories`),c(r,`categories.categoryCount`,t),s(r,`targetZoneCount * categories.categoryCount`,u*t,A),e.length!==l)throw Error(`${r} categories.sourceCategories must have sourceZoneCount rows`);if(!i)throw Error(`${r} categories needs output.categoryShares`);if(o(i,[`float32`],`${r} output.categoryShares`),i.length!==u*t)throw Error(`${r} output.categoryShares must have targetZoneCount * categoryCount rows`)}else if(d.categoryShares)throw Error(`${r} output.categoryShares needs categories`);let h=!n.cellWeights&&n.unweightedFastPath!==!1&&this.spatialContext.coordinateSpace!==`longitude-latitude`;t(r,this.getOutputBindings(),[{write:{sourceAreas:d.sourceAreas,fractions:d.sourceTransferredFractions,invalidCount:d.status.invalidCount}},{write:{neighbors:d.weights.neighbors,areas:h?d.areas:void 0}},{write:{offsets:d.weights.offsets,overflow:d.status.overflow,requiredPairCount:d.status.requiredPairCount}},{read:{neighbors:d.weights.neighbors,areas:d.areas},write:{weights:d.weights.weights,alternateWeights:d.alternateWeights}},{write:{categoryShares:d.categoryShares}}]),e(r,this.getOutputViews(),this.getInputViews())}getInputViews(){let{props:e}=this;return[e.sourceZones,e.targetZones,e.cellWeights,e.parameters,e.sourceAnchorCells,e.categories?.sourceCategories]}getOutputViews(){return Object.values(this.getOutputBindings())}getOutputBindings(){let{output:e}=this.props;return{"output.weights.offsets":e.weights.offsets,"output.weights.neighbors":e.weights.neighbors,"output.weights.weights":e.weights.weights,"output.alternateWeights":e.alternateWeights,"output.areas":e.areas,"output.status.overflow":e.status.overflow,"output.status.requiredPairCount":e.status.requiredPairCount,"output.status.invalidCount":e.status.invalidCount,"output.sourceAreas":e.sourceAreas,"output.sourceTransferredFractions":e.sourceTransferredFractions,"output.categoryShares":e.categoryShares}}getCommandNodes(e){let{id:t,props:o}=this;d(e,t,{sourceZones:o.sourceZones,targetZones:o.targetZones,cellWeights:o.cellWeights,parameters:o.parameters,sourceAnchorCells:o.sourceAnchorCells,sourceCategories:o.categories?.sourceCategories,...this.getOutputBindings()}),s(t,`sourceZones rows`,o.sourceZones.length,T(e.device));let{targetZones:c,sourceZoneCount:l,targetZoneCount:h,output:g}=o,{weights:_}=g,y=o.sourceZones.length,S=_.neighbors.length,C=Math.min(S,y),E=o.mode??`extensive`,D=this.spatialContext.coordinateSpace===`longitude-latitude`,A=this.extensiveDenominator===`overlap`,j=this.intensiveDenominator===`overlap`,M=A||!!g.sourceTransferredFractions,N=j,P=M||N,F=D||!!o.cellWeights||o.unweightedFastPath===!1;for(let[n,r]of Object.entries({cellScratch:y,groupStarts:y+1,pairScratch:S,sourceTotals:l,targetTotals:h,sourceSegmentOffsets:F?l+1:0,targetSegmentOffsets:F?h+1:0,categorySegmentOffsets:o.categories?h*o.categories.categoryCount+1:0}))u(e.device,t,n,r*Uint32Array.BYTES_PER_ELEMENT,`reduce the grid, pair capacity or zone counts`);let I=[],L=o.sourceZones;if(o.sourceAnchorCells){let n=b(e,{id:t,operation:w,zones:o.sourceZones,zoneCount:l,anchorCells:o.sourceAnchorCells});L=n.zones,I.push(...n.nodes)}let R=o.cellWeights;if(D){let r=n(e,`${t}-area-weights`,`float32`,y);I.push(v(e,{id:`${t}-area-weights`,operation:w,cellCount:y,width:o.grid.width,height:o.grid.height,radius:ee(this.spatialContext),extent:o.parameters,cellWeights:R,output:r})),R=r}let z=n(e,`${t}-key-high`,`uint32`,y),B=n(e,`${t}-key-low`,`uint32`,y),V=!R&&o.unweightedFastPath!==!1,H=V?void 0:n(e,`${t}-cell-mask`,`float32`,y),U=M?n(e,`${t}-source-key`,`uint32`,y):void 0,W=N?n(e,`${t}-target-key`,`uint32`,y):void 0;I.push(i(e,{id:`${t}-cell-keys`,operation:w,variant:`cell-keys-${M?`s`:``}${N?`t`:``}${P?`overlap`:`zone`}${R?`-masked`:``}${V?`-fast`:``}`,bindings:[{name:`sourceZones`,view:L,type:`u32`,access:`read`},{name:`targetZones`,view:c,type:`u32`,access:`read`},...R?[{name:`cellWeights`,view:R,type:`f32`,access:`read`}]:[],{name:`keyHigh`,view:z,type:`u32`,access:`read_write`},{name:`keyLow`,view:B,type:`u32`,access:`read_write`},...H?[{name:`cellMask`,view:H,type:`f32`,access:`read_write`}]:[],...U?[{name:`sourceKey`,view:U,type:`u32`,access:`read_write`}]:[],...W?[{name:`targetKey`,view:W,type:`u32`,access:`read_write`}]:[]],invocationCount:y,declarations:`const SOURCE_COUNT: u32 = ${l}u;
const TARGET_COUNT: u32 = ${h}u;`,body:`let source = sourceZones[sourceZonesOffset + index];
  let targetRow = targetZones[targetZonesOffset + index];
  ${R?`var weight = cellWeights[cellWeightsOffset + index];
  // Non-finite (NaN or infinity) and non-positive weights carry no area.
  if ((bitcast<u32>(weight) & 0x7f800000u) == 0x7f800000u || !(weight > 0.0)) {
    weight = 0.0;
  }`:`let weight = 1.0;`}
  let valid = source < SOURCE_COUNT && targetRow < TARGET_COUNT && weight > 0.0;
  keyHigh[keyHighOffset + index] = select(${O}, targetRow, valid);
  keyLow[keyLowOffset + index] = select(${O}, source, valid);
  ${H?`cellMask[cellMaskOffset + index] = weight;`:``}
  ${U?`sourceKey[sourceKeyOffset + index] = select(${O}, source, valid);`:``}
  ${W?`targetKey[targetKeyOffset + index] = select(${O}, targetRow, valid);`:``}`}));let G=te(e,`${t}-cell`,w,y,z,B,{lowKeyLimit:l,highKeyLimit:h}),K=x(e,`${t}-cell`,w,y,z,B,G.sortedItems);I.push(...G.nodes,...K.nodes);let q=g.areas??n(e,`${t}-areas`,`float32`,S);if(H){let i=n(e,`${t}-sorted-mask`,`float32`,y);I.push(...new r({id:`${t}-gather-mask`,source:H,indices:G.sortedItems,output:i}).getCommandNodes(e),f(e,{id:`${t}-pair-areas`,operation:w,segmentCount:C,input:i,segmentOffsets:K.groupStarts,output:q}))}let J=(r,o,s)=>{let c=n(e,`${t}-${r}-counts`,`uint32`,s),l=n(e,`${t}-${r}-totals`,`float32`,s);return I.push(a(e,{id:`${t}-${r}-counts-clear`,operation:w,view:c,type:`u32`,value:`0u`}),i(e,{id:`${t}-${r}-counts`,operation:w,variant:`zone-counts`,bindings:[{name:`keys`,view:o,type:`u32`,access:`read`},{name:`counts`,view:c,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(y/k),declarations:`const ZONE_COUNT: u32 = ${s}u;
const CELL_COUNT: u32 = ${y}u;
const CHUNK: u32 = ${k}u;`,body:`let end = min((index + 1u) * CHUNK, CELL_COUNT);
  var runKey = 0xffffffffu;
  var runLength = 0u;
  for (var cell = index * CHUNK; cell < end; cell++) {
    let key = keys[keysOffset + cell];
    if (key != runKey) {
      if (runKey < ZONE_COUNT) {
        atomicAdd(&counts[countsOffset + runKey], runLength);
      }
      runKey = key;
      runLength = 0u;
    }
    runLength++;
  }
  if (runKey < ZONE_COUNT) {
    atomicAdd(&counts[countsOffset + runKey], runLength);
  }`})),H?I.push(...p(e,{id:`${t}-${r}`,operation:w,segmentCount:s,segmentKeys:o,sumContributions:H,sums:l})):I.push(i(e,{id:`${t}-${r}-totals`,operation:w,variant:`zone-totals-from-counts`,bindings:[{name:`counts`,view:c,type:`u32`,access:`read`},{name:`totals`,view:l,type:`f32`,access:`read_write`}],invocationCount:s,body:`totals[totalsOffset + index] = f32(counts[countsOffset + index]);`})),l},Y=J(`source`,A?U:L,l),ne=J(`target`,j?W:c,h),{sourceAreas:X,sourceTransferredFractions:Z}=g,{invalidCount:Q}=g.status;if(X||Z||Q){let n=A?J(`source-zone`,L,l):Y,r=Z?A?Y:J(`source-overlap`,U,l):void 0;Q&&I.push(a(e,{id:`${t}-invalid-count-clear`,operation:w,view:Q,type:`u32`,value:`0u`,componentCount:1}));let o={zoneAreas:`zoneAreas`,denominatorAreas:`denominatorAreas`,overlapAreas:`overlapAreas`},s=[[`zoneAreas`,n],[`denominatorAreas`,Y],[`overlapAreas`,r]],c=new Map;for(let[e,t]of s)t&&(o[e]=c.get(t)??e,c.set(t,o[e]));I.push(i(e,{id:`${t}-source-diagnostics`,operation:w,variant:`source-diagnostics`,bindings:[...[...c].map(([e,t])=>({name:t,view:e,type:`f32`,access:`read`})),...X?[{name:`sourceAreas`,view:X,type:`f32`,access:`read_write`}]:[],...Z?[{name:`fractions`,view:Z,type:`u32`,access:`read_write`}]:[],...Q?[{name:`invalidCount`,view:Q,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:l,declarations:m,body:`let area = ${o.zoneAreas}[${o.zoneAreas}Offset + index];
  let hasArea = area > 0.0;
  ${X?`sourceAreas[sourceAreasOffset + index] = select(0.0, area, hasArea);`:``}
  ${Z?`let denominator = ${o.denominatorAreas}[${o.denominatorAreas}Offset + index];
  var fraction = bitcast<u32>(getNaN());
  if (hasArea) {
    fraction = bitcast<u32>(select(0.0, ${o.overlapAreas}[${o.overlapAreas}Offset + index] / denominator, denominator > 0.0));
  }
  fractions[fractionsOffset + index] = fraction;`:``}
  ${Q?`if (!hasArea) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }`:``}`}))}let $=n(e,`${t}-pair-target`,`uint32`,S);if(I.push(i(e,{id:`${t}-pair-keys`,operation:w,variant:H?`pair-keys`:`pair-keys-run-lengths`,bindings:[{name:`sortedItems`,view:G.sortedItems,type:`u32`,access:`read`},{name:`groupStarts`,view:K.groupStarts,type:`u32`,access:`read`},{name:`groupIndex`,view:K.groupIndex,type:`u32`,access:`read`},{name:`keyHigh`,view:z,type:`u32`,access:`read`},{name:`keyLow`,view:B,type:`u32`,access:`read`},{name:`neighbors`,view:_.neighbors,type:`u32`,access:`read_write`},{name:`pairTarget`,view:$,type:`u32`,access:`read_write`},...H?[]:[{name:`areas`,view:q,type:`f32`,access:`read_write`}]],invocationCount:C,declarations:`const CELL_COUNT: u32 = ${y}u;`,body:`let groupCount = groupIndex[groupIndexOffset + CELL_COUNT - 1u];
  var neighbor = ${O};
  var targetRow = ${O};
  if (index < groupCount) {
    let item = sortedItems[sortedItemsOffset + groupStarts[groupStartsOffset + index]];
    neighbor = keyLow[keyLowOffset + item];
    targetRow = keyHigh[keyHighOffset + item];
  }
  neighbors[neighborsOffset + index] = neighbor;
  pairTarget[pairTargetOffset + index] = targetRow;${H?``:`
  // Unit cell weights: the pair area is the run length.
  var area = 0.0;
  if (index < groupCount) {
    area = f32(groupStarts[groupStartsOffset + index + 1u] - groupStarts[groupStartsOffset + index]);
  }
  areas[areasOffset + index] = area;`}`}),i(e,{id:`${t}-offsets`,operation:w,variant:`offsets`,bindings:[{name:`sortedItems`,view:G.sortedItems,type:`u32`,access:`read`},{name:`groupStarts`,view:K.groupStarts,type:`u32`,access:`read`},{name:`groupIndex`,view:K.groupIndex,type:`u32`,access:`read`},{name:`keyHigh`,view:z,type:`u32`,access:`read`},{name:`offsets`,view:_.offsets,type:`u32`,access:`read_write`},{name:`overflow`,view:g.status.overflow,type:`u32`,access:`read_write`},...g.status.requiredPairCount?[{name:`totalPairs`,view:g.status.requiredPairCount,type:`u32`,access:`read_write`}]:[]],invocationCount:h+1,declarations:`const CELL_COUNT: u32 = ${y}u;
const ROWS: u32 = ${h}u;
const CAPACITY: u32 = ${S}u;`,body:`// Lower bound: the first pair run whose targetRow is >= index (invalid runs hold 0xffffffff).
  let groupCount = groupIndex[groupIndexOffset + CELL_COUNT - 1u];
  var low = 0u;
  var high = groupCount;
  while (low < high) {
    let middle = (low + high) / 2u;
    let item = sortedItems[sortedItemsOffset + groupStarts[groupStartsOffset + middle]];
    if (keyHigh[keyHighOffset + item] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  offsets[offsetsOffset + index] = min(low, CAPACITY);
  if (index == ROWS) {
    overflow[overflowOffset] = select(0u, 1u, low > CAPACITY);
    ${g.status.requiredPairCount?`totalPairs[totalPairsOffset] = low;`:``}
  }`}),i(e,{id:`${t}-weights`,operation:w,variant:`weights-${E}${g.alternateWeights?`-both`:``}`,bindings:[{name:`areas`,view:q,type:`f32`,access:`read`},{name:`neighbors`,view:_.neighbors,type:`u32`,access:`read`},{name:`pairTarget`,view:$,type:`u32`,access:`read`},{name:`sourceTotals`,view:Y,type:`f32`,access:`read`},{name:`targetTotals`,view:ne,type:`f32`,access:`read`},{name:`weights`,view:_.weights,type:`f32`,access:`read_write`},...g.alternateWeights?[{name:`alternateWeights`,view:g.alternateWeights,type:`f32`,access:`read_write`}]:[]],invocationCount:C,declarations:`const SOURCE_COUNT: u32 = ${l}u;
const TARGET_COUNT: u32 = ${h}u;`,body:`let source = neighbors[neighborsOffset + index];
  let targetRow = pairTarget[pairTargetOffset + index];
  var extensive = 0.0;
  var intensive = 0.0;
  if (source < SOURCE_COUNT && targetRow < TARGET_COUNT) {
    let area = areas[areasOffset + index];
    let sourceTotal = sourceTotals[sourceTotalsOffset + source];
    let targetTotal = targetTotals[targetTotalsOffset + targetRow];
    extensive = select(0.0, area / sourceTotal, sourceTotal > 0.0);
    intensive = select(0.0, area / targetTotal, targetTotal > 0.0);
  }
  weights[weightsOffset + index] = ${E===`extensive`?`extensive`:`intensive`};
  ${g.alternateWeights?`alternateWeights[alternateWeightsOffset + index] = ${E===`extensive`?`intensive`:`extensive`};`:``}`})),o.categories&&g.categoryShares){let{sourceCategories:r,categoryCount:a}=o.categories,s=h*a,c=n(e,`${t}-category-keys`,`uint32`,C),u=n(e,`${t}-category-areas`,`float32`,C),d=n(e,`${t}-covered-areas`,`float32`,h);I.push(i(e,{id:`${t}-category-keys`,operation:w,variant:`category-keys`,bindings:[{name:`pairTarget`,view:$,type:`u32`,access:`read`},{name:`neighbors`,view:_.neighbors,type:`u32`,access:`read`},{name:`areas`,view:q,type:`f32`,access:`read`},{name:`sourceCategories`,view:r,type:`u32`,access:`read`},{name:`categoryKeys`,view:c,type:`u32`,access:`read_write`},{name:`categoryAreas`,view:u,type:`f32`,access:`read_write`}],invocationCount:C,declarations:`const SOURCE_COUNT: u32 = ${l}u;
const TARGET_COUNT: u32 = ${h}u;
const CATEGORY_COUNT: u32 = ${a}u;
const CATEGORY_SHARE_COUNT: u32 = ${s}u;`,body:`let source = neighbors[neighborsOffset + index];
  let targetRow = pairTarget[pairTargetOffset + index];
  var key = CATEGORY_SHARE_COUNT;
  var area = 0.0;
  if (source < SOURCE_COUNT && targetRow < TARGET_COUNT) {
    let category = sourceCategories[sourceCategoriesOffset + source];
    if (category < CATEGORY_COUNT) {
      key = targetRow * CATEGORY_COUNT + category;
      area = areas[areasOffset + index];
    }
  }
  categoryKeys[categoryKeysOffset + index] = key;
  categoryAreas[categoryAreasOffset + index] = area;`}),f(e,{id:`${t}-category-coverage`,operation:w,segmentCount:h,input:q,segmentOffsets:_.offsets,output:d}),...p(e,{id:`${t}-category-totals`,operation:w,segmentCount:s,segmentKeys:c,sumContributions:u,sums:g.categoryShares}),i(e,{id:`${t}-category-shares`,operation:w,variant:`category-shares`,bindings:[{name:`coveredAreas`,view:d,type:`f32`,access:`read`},{name:`output`,view:g.categoryShares,type:`f32`,access:`read_write`}],invocationCount:s,declarations:`const CATEGORY_COUNT: u32 = ${a}u;`,body:`let covered = coveredAreas[coveredAreasOffset + index / CATEGORY_COUNT];
  output[outputOffset + index] = select(0.0, output[outputOffset + index] / covered, covered > 0.0);`}))}return I}};export{j as t};
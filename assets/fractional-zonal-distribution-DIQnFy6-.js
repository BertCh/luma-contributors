import{r as e,s as t}from"./kernel-C3tfQOn4.js";import{E as n,T as r,f as i,g as a,s as o}from"./gpu-contributor-utils-BIwLQON5.js";import{o as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{u as c}from"./wgsl-common-DfyXiYB0.js";import{t as l}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{t as u}from"./sorted-segment-offsets-ChmOtA9F.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{d as f,h as p,n as m,r as h,v as g}from"./raster-band-utils-BcTxc8_Y.js";import{u as _}from"./raster-grid-VB0AM_GY.js";import{n as v,t as y}from"./raster-cell-area-scale-DIJz5iDQ.js";import{i as b}from"./raster-expansion-utils-AZOLseCG.js";import{n as x,r as S}from"./raster-fractional-tile-kernel-D2BjD3g1.js";import{t as C}from"./gpu-raster-grid-C_izEAl0.js";import{a as w}from"./pointer-overlay-Qh8qflcA.js";import{t as T}from"./define-run-B-brkCe4.js";import{Xi as E}from"./entry-preview-D70TdSbK.js";import{t as D}from"./bands-CVDIl0aE.js";import{a as O,i as k,l as A,o as j,s as M,t as N,u as P}from"./zones-Bbw4kokc.js";var F=6,I=65536,L=65536,R=`GPURasterFractionalZonalDistribution`,z=7,B=11,V=256,H=`${c}
fn missingValue(seed: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (seed & 0u)); }`,U=class{id;props;constructor(e){this.id=e.id??`raster-fractional-zonal-distribution`,this.props=e,a(this.id,`input.scale`,e.input.scale??1),a(this.id,`input.offset`,e.input.offset??0),new g({id:`${this.id} input`,metadata:e.grid,bands:[e.input]});for(let t of[...W(e),...o(e.output)])n(t,[t.format],this.id);S(this.id,e.output,W(e),[[e.output.quantiles,e.output.quantileValidity,e.probabilities],[e.output.means,e.output.populationVariances,e.output.sampleVariances,e.output.effectiveSampleSizes],[e.output.weightSums,e.output.squaredWeightSums,e.output.coveredArea,e.output.validArea],[e.output.validity,e.output.status]]);for(let[t,r]of Object.entries(e.output))n(r,t===`status`||t===`validity`||t===`quantileValidity`?[`uint32`]:[`float32`],`${this.id} output.${t}`);for(let[t,r]of Object.entries(e.coverage))t!==`status`&&n(r,t===`fractions`?[`float32`]:[`uint32`],`${this.id} coverage.${t}`);for(let[t,r]of Object.entries(e.coverage.status))r&&n(r,[`uint32`],`${this.id} coverage.status.${t}`);n(e.probabilities,[`float32`],`${this.id} probabilities`);for(let[t,n]of Object.entries({observationWeights:e.observationWeights,cellAreas:e.cellAreas}))r(this.id,!n||n.format===`float32`&&n.length===e.grid.width*e.grid.height,`${t} must be a float32 view with one entry per grid cell`);r(this.id,e.quantileMethod===void 0||[`exactextract`,`inverse-cdf`].includes(e.quantileMethod),`quantileMethod must be 'exactextract' or 'inverse-cdf'`);let t=e.coverage.zoneIds.length;i(this.id,`zoneCount`,e.zoneCount,0,L),i(this.id,`coverage.zoneIds length`,t,1,I),i(this.id,`probabilities length`,e.probabilities.length,1,32),r(this.id,e.coverage.cellIds.length===t&&e.coverage.fractions.length===t,`coverage.cellIds and coverage.fractions must match the length of coverage.zoneIds`),r(this.id,e.coverage.zoneValidity.length===e.zoneCount,`coverage.zoneValidity must hold zoneCount entries`);for(let[t,n]of Object.entries(e.coverage.status))n&&r(this.id,n.length>=1,`coverage.status.${t} must hold at least one entry`);for(let[t,n]of Object.entries(e.output)){let i=t===`quantiles`||t===`quantileValidity`?e.zoneCount*e.probabilities.length:e.zoneCount;r(this.id,n.length===i,`output.${t} must hold ${i} entries`)}e.parameters&&p(this.id,e.parameters,7),_(this.id,e.cellSizeMode),f(this.id,e.grid,e.cellSizeMode),r(this.id,!e.cellAreas||(e.cellSizeMode??`uniform`)===`uniform`,`cellSizeMode must be uniform when cellAreas is given (cellAreas already declares the areas)`);let s=e.grid.affine[0]*e.grid.affine[4]-e.grid.affine[1]*e.grid.affine[3];r(this.id,Number.isFinite(Math.fround(Math.abs(s)*t))&&Math.fround(Math.abs(s))>0,`grid.affine must have a finite nonzero determinant whose total area fits float32`)}getCommandNodes(n){let{id:r,props:i}=this,{coverage:a,output:c,zoneCount:f,probabilities:p}=i,m=a.zoneIds.length,g=i.grid.width*i.grid.height;d(n,r,Object.fromEntries([...W(i),...o(c)].map((e,t)=>[`view.${t}`,e])));let _=i.parameters?`abs(georeference[georeferenceOffset + ${F}u])`:String(Math.abs(i.grid.affine[0]*i.grid.affine[4]-i.grid.affine[1]*i.grid.affine[3])),S=i.cellSizeMode??`uniform`,C=S===`uniform`?``:v(i.grid,S,i.parameters?`georeference`:void 0),w=S===`uniform`?`f32(${_})`:`f32(${_}) * select(1.0, getRowAreaScale(cell / ${i.grid.width}u), cell < ${g}u)`;b(r,n,[...W(i),...o(c)]);for(let e of[...W(i),...o(c)])x(n.device,r,e.buffer.id,t(e).size,`reduce the packed input or output size`,e.buffer.byteLength);for(let[e,t]of[[`canonical raster band`,g*4],[`records`,m*z*4],[`tuple columns`,m*4],[`sort scratch`,(m<=V?2**Math.ceil(Math.log2(m)):m)*4],[`segment offsets`,(f+1)*4],[`summary`,f*B*4],[`control`,12]])x(n.device,r,e,t,`reduce tuple capacity, zoneCount or grid dimensions`);let T=h(n,r,i.input,i.grid,!0),E=e(n,`${r}-records`,`uint32`,m*z),D=e(n,`${r}-control`,`uint32`,3),O=e(n,`${r}-centered-values`,`float32`,m),k=e(n,`${r}-indices`,`uint32`,m),A=e(n,`${r}-zone-keys`,`uint32`,m),j=e(n,`${r}-value-keys`,`uint32`,m),M=e(n,`${r}-cell-keys`,`uint32`,m),N=e(n,`${r}-offsets`,`uint32`,f+1),P=e(n,`${r}-summary`,`float32`,f*B),I=l(n,{id:`${r}-sort`,operation:R,indices:k,keys:[{view:A},{view:j},{view:M}]}),L=[...T.nodes,s(n,{id:`${r}-reset`,operation:R,invocationCount:1,bindings:[{name:`count`,view:a.status.count,type:`u32`,access:`read`},{name:`required`,view:a.status.requiredCount,type:`u32`,access:`read`},{name:`overflow`,view:a.status.overflow,type:`u32`,access:`read`},...a.status.candidateOverflow?[{name:`candidateOverflow`,view:a.status.candidateOverflow,type:`u32`,access:`read`}]:[],...S!==`uniform`&&i.parameters?[{name:`georeference`,view:i.parameters,type:`f32`,access:`read`}]:[],{name:`control`,view:D,type:`u32`,access:`read_write`}],body:`let countValue = count[countOffset];
  control[controlOffset] = min(countValue, ${m}u);
  control[controlOffset + 1u] = u32(countValue <= ${m}u && required[requiredOffset] == countValue && overflow[overflowOffset] == 0u ${a.status.candidateOverflow?`&&candidateOverflow[candidateOverflowOffset]==0u`:``} && (${y(i.grid,S,i.parameters?`georeference`:void 0)}));
  control[controlOffset + 2u] = 0u;`}),s(n,{id:`${r}-gather`,operation:R,invocationCount:m,bindings:[{name:`values`,view:T.band.storage.values,type:`f32`,access:`read`},{name:`validity`,view:T.band.validity,type:`u32`,access:`read`},{name:`zoneIds`,view:a.zoneIds,type:`u32`,access:`read`},{name:`cellIds`,view:a.cellIds,type:`u32`,access:`read`},{name:`fractions`,view:a.fractions,type:`f32`,access:`read`},{name:`control`,view:D,type:`atomic<u32>`,access:`read_write`},{name:`records`,view:E,type:`u32`,access:`read_write`},...i.parameters?[{name:`georeference`,view:i.parameters,type:`f32`,access:`read`}]:[]],declarations:`${H}${C}`,body:`let base = recordsOffset + index * ${z}u;
  var zone = ${f}u; var cell = 0xffffffffu; var value = missingValue(index); var fraction = 0.0; var flags = 0u;
  if (index < atomicLoad(&control[controlOffset])) {
    zone = zoneIds[zoneIdsOffset + index]; cell = cellIds[cellIdsOffset + index]; fraction = fractions[fractionsOffset + index];
    if (zone >= ${f}u) { atomicStore(&control[controlOffset + 2u], 1u); zone = ${f}u; }
    if (cell < ${g}u && validity[validityOffset + cell] != 0u) { value = values[valuesOffset + cell]; }
    if (cell >= ${g}u || !isFiniteValue(fraction) || fraction < 0.0 || fraction > 1.0) { flags = 2u; }
    else if (isFiniteValue(value) && fraction > 0.0) { flags = 1u; }
  }
  records[base] = zone; records[base + 1u] = cell;
  records[base + 2u] = select(bitcast<u32>(value), 0u, value == 0.0);
  records[base + 3u] = bitcast<u32>(fraction); records[base + 4u] = flags;
  records[base + 5u] = bitcast<u32>(fraction); records[base + 6u] = bitcast<u32>(${w});`}),...i.observationWeights||i.cellAreas?[s(n,{id:`${r}-observation-support`,operation:R,invocationCount:m,bindings:[{name:`records`,view:E,type:`u32`,access:`read_write`},...i.observationWeights?[{name:`observationWeights`,view:i.observationWeights,type:`f32`,access:`read`}]:[],...i.cellAreas?[{name:`cellAreas`,view:i.cellAreas,type:`f32`,access:`read`}]:[]],declarations:H,body:`let base = recordsOffset + index * ${z}u; let cell = records[base + 1u];
if (cell >= ${g}u) { return; }
let observationWeight = ${i.observationWeights?`observationWeights[observationWeightsOffset + cell]`:`1.0`};
let area: f32 = ${i.cellAreas?`cellAreas[cellAreasOffset + cell]`:i.parameters||S!==`uniform`?`bitcast<f32>(records[base + 6u])`:_};
let weight = bitcast<f32>(records[base + 3u]) * observationWeight;
if (!isFiniteValue(observationWeight) || observationWeight < 0.0 || !isFiniteValue(weight) || !isFiniteValue(area) || area <= 0.0) { records[base + 4u] = 2u; }
records[base + 5u] = bitcast<u32>(weight); records[base + 6u] = bitcast<u32>(area);`})]:[],s(n,{id:`${r}-keys`,operation:R,invocationCount:m,bindings:[{name:`records`,view:E,type:`u32`,access:`read`},{name:`indices`,view:k,type:`u32`,access:`read_write`},{name:`zoneKeys`,view:A,type:`u32`,access:`read_write`},{name:`valueKeys`,view:j,type:`u32`,access:`read_write`},{name:`cellKeys`,view:M,type:`u32`,access:`read_write`}],body:`let base = recordsOffset + index * ${z}u; let bits = records[base + 2u];
  indices[indicesOffset + index] = index; zoneKeys[zoneKeysOffset + index] = records[base];
  valueKeys[valueKeysOffset + index] = select(0xffffffffu, select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u), (bits & 0x7f800000u) != 0x7f800000u);
  cellKeys[cellKeysOffset + index] = records[base + 1u];`}),...I.nodes,u(n,{id:`${r}-offsets`,operation:R,segmentCount:f,sortedKeys:I.sortedPrimaryKeys,segmentOffsets:N}),s(n,{id:`${r}-center`,operation:R,invocationCount:f,bindings:[{name:`records`,view:E,type:`u32`,access:`read`},{name:`indices`,view:I.sortedIndices,type:`u32`,access:`read`},{name:`offsets`,view:N,type:`u32`,access:`read`},{name:`centeredValues`,view:O,type:`f32`,access:`read_write`}],body:`let start = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
  var anchor = 0.0; var hasAnchor = false;
  for (var cursor = start; cursor < end; cursor++) {
    let originalIndex = indices[indicesOffset + cursor];
    let base = recordsOffset + originalIndex * ${z}u;
    var centered = 0.0;
    if (records[base + 4u] == 1u && bitcast<f32>(records[base + 5u]) > 0.0) {
      let value = bitcast<f32>(records[base + 2u]);
      if (!hasAnchor) { anchor = value; hasAnchor = true; }
      centered = value - anchor;
    }
    centeredValues[centeredValuesOffset + originalIndex] = centered;
  }`}),s(n,{id:`${r}-moments`,operation:R,invocationCount:f,bindings:[{name:`records`,view:E,type:`u32`,access:`read`},{name:`indices`,view:I.sortedIndices,type:`u32`,access:`read`},{name:`offsets`,view:N,type:`u32`,access:`read`},{name:`control`,view:D,type:`u32`,access:`read`},{name:`zoneValidity`,view:a.zoneValidity,type:`u32`,access:`read`},{name:`centeredValues`,view:O,type:`f32`,access:`read`},{name:`summary`,view:P,type:`f32`,access:`read_write`}],declarations:H,body:`let start = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
  var status = select(1u, 0u, control[controlOffset + 1u] != 0u && zoneValidity[zoneValidityOffset + index] != 0u);
  if (control[controlOffset + 2u] != 0u) { status |= 2u; }
  var covered = 0.0; var validArea = 0.0; var weight = 0.0; var squaredWeight = 0.0; var centeredSum = 0.0; var centeredPartials: array<f32, 256>; var supportPartials: array<vec4<f32>, 256>; var anchor = 0.0;
  var validCellCount = 0u; var uniqueCount = 0u; var previousValue = 0.0; var previousCell = 0xffffffffu;
  for (var cursor = start; cursor < end; cursor++) {
    let base = recordsOffset + indices[indicesOffset + cursor] * ${z}u;
    let cell = records[base + 1u]; let flags = records[base + 4u];
    if (flags == 2u || (cursor > start && cell == previousCell)) { status |= 2u; }
    previousCell = cell;
    if (flags == 2u) { continue; }
    let fraction = bitcast<f32>(records[base + 3u]); let area = bitcast<f32>(records[base + 6u]); supportPartials[(cursor - start) % 256u].x += fraction * area;
    if (flags != 1u) { continue; }
    supportPartials[(cursor - start) % 256u].y += fraction * area;
    let observationWeight = bitcast<f32>(records[base + 5u]);
    if (observationWeight <= 0.0) { continue; }
    let value = bitcast<f32>(records[base + 2u]);
    if (uniqueCount == 0u) { anchor = value; }
    if (uniqueCount == 0u || value != previousValue) { uniqueCount++; }
    previousValue = value; validCellCount++; supportPartials[(cursor - start) % 256u].z += observationWeight; supportPartials[(cursor - start) % 256u].w += observationWeight * observationWeight;
    centeredPartials[(cursor - start) % 256u] += observationWeight * centeredValues[centeredValuesOffset + indices[indicesOffset + cursor]];
  }
  for (var stride = 128u; stride > 0u; stride /= 2u) { for (var partial = 0u; partial < stride; partial++) { centeredPartials[partial] += centeredPartials[partial + stride]; supportPartials[partial] += supportPartials[partial + stride]; } }
  centeredSum = centeredPartials[0]; covered = supportPartials[0].x; validArea = supportPartials[0].y; weight = supportPartials[0].z; squaredWeight = supportPartials[0].w;
  var centeredMean = 0.0; var mean = missingValue(index); var populationVariance = missingValue(index); var sampleVariance = missingValue(index); var effectiveSupport = 0.0;
  if (weight > 0.0) { centeredMean = centeredSum / weight; mean = anchor + centeredMean; }
  var secondMoment = 0.0; var momentPartials: array<f32, 256>;
  for (var cursor = start; cursor < end; cursor++) {
    let base = recordsOffset + indices[indicesOffset + cursor] * ${z}u;
    if (records[base + 4u] != 1u || bitcast<f32>(records[base + 5u]) <= 0.0) { continue; }
    let difference = centeredValues[centeredValuesOffset + indices[indicesOffset + cursor]] - centeredMean;
    momentPartials[(cursor - start) % 256u] += bitcast<f32>(records[base + 5u]) * difference * difference;
  }
  for (var stride = 128u; stride > 0u; stride /= 2u) { for (var partial = 0u; partial < stride; partial++) { momentPartials[partial] += momentPartials[partial + stride]; } }
  secondMoment = momentPartials[0];
  var denominator = 0.0;
  if (weight > 0.0) {
    populationVariance = secondMoment / weight; effectiveSupport = weight * weight / squaredWeight;
    denominator = weight - squaredWeight / weight;
    if (validCellCount >= 2u && denominator > 0.0) { sampleVariance = secondMoment / denominator; }
    if (!isFiniteValue(mean) || !isFiniteValue(populationVariance) || !isFiniteValue(effectiveSupport) || (validCellCount >= 2u && denominator > 0.0 && !isFiniteValue(sampleVariance))) { status |= 16u; }
  } else { status |= 4u; }
  if (validCellCount < 2u || denominator <= 0.0) { status |= 8u; }
  if (!isFiniteValue(covered) || !isFiniteValue(validArea)) { status |= 16u; }
  let complete = (status & 3u) == 0u;
  let valid = (status & 23u) == 0u;
  let base = summaryOffset + index * ${B}u;
  summary[base] = select(missingValue(index), mean, valid);
  summary[base + 1u] = select(missingValue(index), populationVariance, valid);
  summary[base + 2u] = select(missingValue(index), sampleVariance, valid);
  summary[base + 3u] = select(missingValue(index), effectiveSupport, complete && (status & 16u) == 0u);
  summary[base + 4u] = select(missingValue(index), weight, complete);
  summary[base + 5u] = select(missingValue(index), squaredWeight, complete);
  summary[base + 6u] = select(missingValue(index), covered, complete);
  summary[base + 7u] = select(missingValue(index), validArea, complete);
  summary[base + 8u] = f32(valid); summary[base + 9u] = f32(status); summary[base + 10u] = f32(uniqueCount);`}),s(n,{id:`${r}-quantiles`,operation:R,invocationCount:f*p.length,bindings:[{name:`records`,view:E,type:`u32`,access:`read`},{name:`indices`,view:I.sortedIndices,type:`u32`,access:`read`},{name:`offsets`,view:N,type:`u32`,access:`read`},{name:`summary`,view:P,type:`f32`,access:`read`},{name:`probabilities`,view:p,type:`f32`,access:`read`},{name:`quantiles`,view:c.quantiles,type:`f32`,access:`read_write`},{name:`validity`,view:c.quantileValidity,type:`u32`,access:`read_write`}],declarations:H,body:`let zone = index / ${p.length}u; let probability = probabilities[probabilitiesOffset + index % ${p.length}u];
  let base = summaryOffset + zone * ${B}u;
  let valid = summary[base + 8u] != 0.0 && isFiniteValue(probability) && probability >= 0.0 && probability <= 1.0;
  var result = missingValue(index);
  if (valid) {
    let distinctCount = summary[base + 10u]; let totalWeight = summary[base + 4u];
    let quantilePosition = probability * (distinctCount - 1.0) * totalWeight;
    var cursor = offsets[offsetsOffset + zone]; let end = offsets[offsetsOffset + zone + 1u];
    var cumulativeWeight = 0.0; var distinctIndex = 0u; var previousPosition = 0.0; var previousValue = 0.0;
    while (cursor < end) {
      var record = recordsOffset + indices[indicesOffset + cursor] * ${z}u;
      if (records[record + 4u] != 1u || bitcast<f32>(records[record + 5u]) <= 0.0) { cursor++; continue; }
      let value = bitcast<f32>(records[record + 2u]); var groupWeight = 0.0;
      while (cursor < end) {
        record = recordsOffset + indices[indicesOffset + cursor] * ${z}u;
        if (bitcast<f32>(records[record + 2u]) != value) { break; }
        if (records[record + 4u] == 1u) { groupWeight += bitcast<f32>(records[record + 5u]); }
        cursor++;
      }
      ${i.quantileMethod===`inverse-cdf`?`if (cumulativeWeight + groupWeight >= probability * totalWeight) { result = value; break; }`:``}
      let position = f32(distinctIndex) * groupWeight + (distinctCount - 1.0) * cumulativeWeight;
      if (${i.quantileMethod===`inverse-cdf`?`false`:`true`} && distinctIndex > 0u && quantilePosition <= position) {
        result = previousValue + (value - previousValue) * ((quantilePosition - previousPosition) / (position - previousPosition)); break;
      }
      result = value; previousValue = value; previousPosition = position;
      cumulativeWeight += groupWeight; distinctIndex++;
    }
  }
  quantiles[quantilesOffset + index] = result; validity[validityOffset + index] = u32(valid && isFiniteValue(result));`})],U=[c.means,c.populationVariances,c.sampleVariances,c.effectiveSampleSizes,c.weightSums,c.squaredWeightSums,c.coveredArea,c.validArea];for(let e=0;e<U.length;e+=4){let t=U.slice(e,e+4);L.push(s(n,{id:`${r}-publish-${e}`,operation:R,invocationCount:f,bindings:[{name:`summary`,view:P,type:`f32`,access:`read`},...t.map((e,t)=>({name:`column${t}`,view:e,type:`f32`,access:`read_write`}))],body:t.map((t,n)=>`column${n}[column${n}Offset + index] = summary[summaryOffset + index * ${B}u + ${e+n}u];`).join(`
`)}))}return L.push(s(n,{id:`${r}-publish-status`,operation:R,invocationCount:f,bindings:[{name:`summary`,view:P,type:`f32`,access:`read`},{name:`validity`,view:c.validity,type:`u32`,access:`read_write`},{name:`status`,view:c.status,type:`u32`,access:`read_write`}],body:`validity[validityOffset + index] = u32(summary[summaryOffset + index * ${B}u + 8u]);
  status[statusOffset + index] = u32(summary[summaryOffset + index * ${B}u + 9u]);`})),L}};function W(e){return[...m(e.input),...o(e.coverage),e.probabilities,e.observationWeights,e.cellAreas,e.parameters].filter(e=>!!e)}var G=w(E,{kind:`level`,hint:`move the probability`,x:`probability`}),K=T({parameters:E,rebuildOn:[`quantileMethod`,`gridWidth`],caption:`Polygons over a raster: the quantile of the values each polygon covers.`,createInputs:({random:e,world:t})=>k(e,t),build(e,t,n){let{features:r}=t,i=N(e,t,n.gridWidth),a=e.upload(`values`,`float32`,A(t,i.columns,i.rows)),o=r.featureCount,s=e.allocate(`probabilities`,`float32`,1),c={means:e.allocate(`means`,`float32`,o),populationVariances:e.allocate(`population-variances`,`float32`,o),sampleVariances:e.allocate(`sample-variances`,`float32`,o),effectiveSampleSizes:e.allocate(`effective-sample-sizes`,`float32`,o),weightSums:e.allocate(`weight-sums`,`float32`,o),squaredWeightSums:e.allocate(`squared-weight-sums`,`float32`,o),coveredArea:e.allocate(`covered-area`,`float32`,o),validArea:e.allocate(`valid-area`,`float32`,o),quantiles:e.allocate(`quantiles`,`float32`,o),quantileValidity:e.allocate(`quantile-validity`,`uint32`,o),validity:e.allocate(`validity`,`uint32`,o),status:e.allocate(`status`,`uint32`,o)};return e.graph.add(new U({grid:i.grid,input:C(a.view,i.grid,{id:`values`}),zoneCount:o,coverage:i.output,probabilities:s.view,quantileMethod:n.quantileMethod===1?`inverse-cdf`:`exactextract`,output:Object.fromEntries(Object.entries(c).map(([e,t])=>[e,t.view]))})),{outputs:{quantiles:c.quantiles,sampleVariances:c.sampleVariances,cellValues:a,columns:i.columns,rows:i.rows,mesh:P(e,r)},update:e=>s.write(Float32Array.of(e.probability))}},interaction:{...G,resolve:(e,t)=>{let n=G.resolve(e,t),r={...t.values,...n.values},{inputs:i,outputs:a,world:o,peek:s}=t,c=j(i.features,e.x,e.y),l=s(a.quantiles),u=o.width*.05,d=o.width*.95,f=o.height*.04,p=u+(d-u)*r.probability;return{...n,glyphs:[{shape:`line`,from:[u,f],to:[d,f]},{shape:`dot`,at:[p,f]},...c>=0?[{shape:`polyline`,closed:!0,points:M(i.features,c)}]:[]],readout:`probability ${r.probability.toFixed(2)} | tract quantile ${c>=0&&l?l[c].toFixed(2):`-`}`}}},draw:({inputs:e,outputs:t,theme:n})=>[D(t,t.cellValues,{context:!0}),O(e.features,t.mesh,t.quantiles,{ramp:`sequential`,range:[.1,.9]}),{layer:`lines`,context:!0,segments:t.mesh.outline,color:[...n.ink,.8]}]});export{K as default};
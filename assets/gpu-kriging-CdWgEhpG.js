import{n as e,t}from"./interpolation-view-validation-pYo9tqh3.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-grid-index-BoHdb71C.js";import{c as i,o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as s}from"./wgsl-common-DfyXiYB0.js";import{n as c}from"./dense-solve-wgsl-0xzjbd8V.js";import{t as l}from"./graph-chunk-groups-BB8ACpQO.js";import{a as u}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as d}from"./status-aOwAAXek.js";import{E as f,S as p,f as m,g as h,o as g,p as _,v}from"./gpu-contributor-utils-BIwLQON5.js";import{u as y}from"./raster-band-utils-BcTxc8_Y.js";import{i as b,n as x,r as S}from"./gpu-inverse-distance-weighting-KIb4SzJG.js";import{n as C,t as w}from"./separation-wgsl-UgBSv4wS.js";import{t as T}from"./kriging-parameters-DGogQ0zZ.js";import{r as E}from"./variogram-model-l9c6-v8f.js";function D(e,t,n){let r=Math.min(t.invocationCount,n??t.invocationCount),i=[];for(let n=0;n<t.invocationCount;n+=r)i.push(l(e,{id:r>=t.invocationCount?t.id:`${t.id}-tile-${n}`,pipelineId:t.id,operation:t.operation,variant:t.variant,bindings:t.bindings,workgroupSize:t.workgroupSize??64,invocationCount:Math.min(r,t.invocationCount-n),rowBase:n,totalLength:t.invocationCount,declarations:t.declarations,body:`let interpolationRow = getChunkParameter(0u) + index;\n${t.body.replace(/\bindex\b/g,`interpolationRow`)}`}));return i}var O=`GPUKriging`,k=16,A=17,j=2048,M=1e-6,N=1e-4,P=0,F=1,I=2,L=class{id;props;spatialContext;separation;cellCount;maximumNeighborCount;estimatedCommandNodeCount;constructor(n){this.id=n.id??`kriging`,this.props=n;let{id:r}=this;if(this.separation=C(r,n.spatialContext),this.spatialContext=this.separation.context,this.cellCount=y(r,n.grid),n.grid.affine[1]!==0||n.grid.affine[3]!==0)throw Error(`${r} needs an axis-aligned grid`);for(let[e,t]of[[`indexGridSize[0]`,n.indexGridSize[0]],[`indexGridSize[1]`,n.indexGridSize[1]]])m(r,e,t,1,2**53-1);let[i,a,o,s]=n.indexBounds;if(n.indexBounds.length!==4||!n.indexBounds.every(Number.isFinite)||i>o||a>s)throw Error(`${r} indexBounds must be finite and ordered`);for(let[e,t]of n.indexBounds.entries())h(r,`indexBounds[${e}]`,t);n.invocationTileSize!==void 0&&m(r,`invocationTileSize`,n.invocationTileSize),t(r,{values:n.output.values,variance:n.output.variance,crossValidationPrediction:n.output.crossValidationPrediction,crossValidationVariance:n.output.crossValidationVariance,...n.output.status},[{write:{values:n.output.values,variance:n.output.variance}},{write:{prediction:n.output.crossValidationPrediction,variance:n.output.crossValidationVariance}},{write:{invalidCount:n.output.status?.invalidCount,uncertainCount:n.output.status?.uncertainCount}}]),this.maximumNeighborCount=n.maximumNeighborCount??k,m(r,`maximumNeighborCount`,this.maximumNeighborCount,1,k);let{output:c}=n;f(n.positions,[`float32x2`],`${r} positions`),f(n.values,[`float32`],`${r} values`);let l=n.values.length;if(l<1)throw Error(`${r} needs at least one sample`);if(n.positions.length!==l)throw Error(`${r} positions length must equal values length`);if(n.mask&&(f(n.mask,[`uint32`],`${r} mask`),n.mask.length!==l))throw Error(`${r} mask length must equal values length`);if(f(n.parameters,[`float32`],`${r} parameters`),n.parameters.length<12)throw Error(`${r} parameters must hold 12 float32 values`);if(f(c.values,[`float32`],`${r} output.values`),c.values.length<this.cellCount)throw Error(`${r} output.values must hold grid.width * grid.height rows`);if(c.variance&&(f(c.variance,[`float32`],`${r} output.variance`),c.variance.length<this.cellCount))throw Error(`${r} output.variance must hold grid.width * grid.height rows`);if(c.crossValidationVariance&&!c.crossValidationPrediction)throw Error(`${r} output.crossValidationVariance needs output.crossValidationPrediction`);if(c.crossValidationPrediction){for(let[e,t]of[[`crossValidationPrediction`,c.crossValidationPrediction],[`crossValidationVariance`,c.crossValidationVariance]])if(t&&(f(t,[`float32`],`${r} output.${e}`),t.length<l))throw Error(`${r} output.${e} must hold one row per sample`)}d(r,{invalidCount:c.status?.invalidCount,uncertainCount:c.status?.uncertainCount}),e(r,[c.values,c.variance,c.crossValidationPrediction,c.crossValidationVariance,c.status?.invalidCount,c.status?.uncertainCount],[n.positions,n.values,n.mask,n.parameters]),this.estimatedCommandNodeCount=j+Math.ceil(this.cellCount/(n.invocationTileSize??this.cellCount))+(c.crossValidationPrediction?Math.ceil(l/(n.invocationTileSize??l)):0),_(r,this.estimatedCommandNodeCount)}getCommandNodes(e){let{id:t,props:c,cellCount:l,maximumNeighborCount:d}=this,{output:f,grid:m,indexGridSize:h,indexBounds:_}=c,{width:y,height:b}=m;u(e,t,{positions:c.positions,values:c.values,mask:c.mask,parameters:c.parameters,outputValues:f.values,variance:f.variance,crossValidationPrediction:f.crossValidationPrediction,crossValidationVariance:f.crossValidationVariance,...f.status}),p(t,`indexGridSize cells`,h[0]*h[1],g(e.device));let S=c.values.length,C=h[0]*h[1];for(let[n,r]of Object.entries({cellOffsets:C+1,objectIds:S,indexStatus:1,maskedValues:c.mask?S:0,cellStatus:f.status?.invalidCount||f.status?.uncertainCount?l:0}))v(e.device,t,n,r*4,`reduce indexGridSize, sample count or grid size`);let w=n(e,`${t}-cell-offsets`,`uint32`,C+1),E=n(e,`${t}-object-ids`,`uint32`,S),k=n(e,`${t}-index-count`,`uint32`,1),j=n(e,`${t}-index-overflow`,`uint32`,1),L=[...new r({id:`${t}-index`,positions:c.positions,gridSize:h,bounds:_,cellOffsets:w,objectIds:E,count:k,overflow:j}).getCommandNodes(e)],z=c.values;c.mask&&(z=n(e,`${t}-masked-values`,`float32`,S),L.push(a(e,{id:`${t}-apply-mask`,operation:O,variant:`apply-mask`,bindings:[{name:`rawValues`,view:c.values,type:`f32`,access:`read`},{name:`sampleMask`,view:c.mask,type:`u32`,access:`read`},{name:`maskedValues`,view:z,type:`f32`,access:`read_write`}],invocationCount:S,declarations:s,body:`maskedValues[maskedValuesOffset + index] = select(getNaN(), rawValues[rawValuesOffset + index], sampleMask[sampleMaskOffset + index] != 0u);`})));let B=[{name:`positions`,view:c.positions,type:`f32`,access:`read`},{name:`sampleValues`,view:z,type:`f32`,access:`read`},{name:`params`,view:c.parameters,type:`f32`,access:`read`},{name:`cellOffsets`,view:w,type:`u32`,access:`read`},{name:`sortedIds`,view:E,type:`u32`,access:`read`}],V=`const INDEX_WIDTH: u32 = ${h[0]}u;
const INDEX_HEIGHT: u32 = ${h[1]}u;
const MAX_NEIGHBORS: u32 = ${d}u;
const SYSTEM_STRIDE: u32 = ${A}u;
const DOMAIN_MIN: vec2f = vec2f(${i(_[0])}, ${i(_[1])});
const DOMAIN_MAX: vec2f = vec2f(${i(_[2])}, ${i(_[3])});
const RELATIVE_PIVOT_TOLERANCE: f32 = ${i(M)};
const PIVOT_RATIO_TOLERANCE: f32 = ${i(T)};
const NEGATIVE_VARIANCE_TOLERANCE: f32 = ${i(N)};
const STATUS_OK: u32 = ${P}u;
const STATUS_INVALID: u32 = ${F}u;
const STATUS_UNCERTAIN: u32 = ${I}u;
const ORIGIN: vec2f = ${this.separation.geographic?`vec2f(0.0)`:`DOMAIN_MIN`};
${x}
${R(this.separation)}`,H=f.status?.invalidCount||f.status?.uncertainCount?n(e,`${t}-cell-status`,`uint32`,l):void 0;if(L.push(...D(e,{id:`${t}-solve`,operation:O,variant:`local-ordinary`,bindings:[...B,{name:`valuesOut`,view:f.values,type:`f32`,access:`read_write`},...f.variance?[{name:`varianceOut`,view:f.variance,type:`f32`,access:`read_write`}]:[],...H?[{name:`cellStatus`,view:H,type:`u32`,access:`read_write`}]:[]],invocationCount:l,workgroupSize:64,declarations:`const WIDTH: u32 = ${y}u;
const HEIGHT: u32 = ${b}u;
${V}`,body:`let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  // Planar offsets from the grid start corner stay small whatever the absolute coordinates are;
  // great-circle distances take absolute lon/lat, so the origin is zero.
  let origin = ${this.separation.geographic?`vec2f(0.0)`:`gridStart`};
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let estimate = estimateKriging(center, ${this.separation.geographic?`center`:`cellOffset`}, origin, 0xffffffffu);
  valuesOut[valuesOutOffset + index] = estimate.value;
  ${f.variance?`varianceOut[varianceOutOffset + index] = estimate.variance;`:``}
  ${H?`cellStatus[cellStatusOffset + index] = estimate.status;`:``}`},c.invocationTileSize)),f.crossValidationPrediction){let{crossValidationPrediction:n,crossValidationVariance:r}=f;L.push(...D(e,{id:`${t}-cross-validation`,operation:O,variant:`leave-one-out`,bindings:[...B,{name:`predictionOut`,view:n,type:`f32`,access:`read_write`},...r?[{name:`predictionVarianceOut`,view:r,type:`f32`,access:`read_write`}]:[]],invocationCount:S,workgroupSize:64,declarations:V,body:`let value = sampleValues[sampleValuesOffset + index];
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  var prediction = getNaN();
  var predictionVariance = getNaN();
  if (!isNanBits(value) && isFiniteValue(position.x) && isFiniteValue(position.y)) {
    // Kriged from every OTHER sample: the row itself is excluded from the neighbor search.
    let estimate = estimateKriging(position, position - ORIGIN, ORIGIN, index);
    prediction = estimate.value;
    predictionVariance = estimate.variance;
  }
  predictionOut[predictionOutOffset + index] = prediction;
  ${r?`predictionVarianceOut[predictionVarianceOutOffset + index] = predictionVariance;`:``}`},c.invocationTileSize))}if(H){for(let[n,r]of[[`invalid`,f.status?.invalidCount],[`uncertain`,f.status?.uncertainCount]])r&&L.push(o(e,{id:`${t}-clear-${n}`,operation:O,view:r,type:`u32`,value:`0u`,componentCount:r.length}));L.push(a(e,{id:`${t}-count-status`,operation:O,variant:`count-status`,bindings:[{name:`cellStatus`,view:H,type:`u32`,access:`read`},...f.status?.invalidCount?[{name:`invalidCountOut`,view:f.status?.invalidCount,type:`atomic<u32>`,access:`read_write`}]:[],...f.status?.uncertainCount?[{name:`uncertainCountOut`,view:f.status?.uncertainCount,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:l,declarations:`const STATUS_INVALID: u32 = ${F}u;
const STATUS_UNCERTAIN: u32 = ${I}u;`,body:`let status = cellStatus[cellStatusOffset + index];
  ${f.status?.invalidCount?`if (status == STATUS_INVALID) { atomicAdd(&invalidCountOut[invalidCountOutOffset], 1u); }`:``}
  ${f.status?.uncertainCount?`if (status == STATUS_UNCERTAIN) { atomicAdd(&uncertainCountOut[uncertainCountOutOffset], 1u); }`:``}`}))}return L}};function R(e){let t=e.geographic?S:b;return`
${w(e)}
struct KrigingEstimate {
  value: f32,
  variance: f32,
  status: u32,
}

${E}
// Variogram normalized by its total sill; gamma(0) = 0. The shape is the shared float32
// definition, which keeps its digits at lags that are a small fraction of the range.
fn evaluateVariogram(distance: f32, modelCode: u32, nugget: f32, sill: f32, range: f32) -> f32 {
  if (distance <= 0.0) { return 0.0; }
  let shape = getVariogramModelShape(modelCode, distance / range);
  return (nugget + sill * shape) / (nugget + sill);
}

// The bordered system is solved by the shared LU with partial pivoting of analysis-core
// (factorDenseLU_${A}), without equilibration. It is rejected when a pivot is
// negligible against the largest entry (RELATIVE_PIVOT_TOLERANCE) or the pivot ratio (a condition
// estimate) is below PIVOT_RATIO_TOLERANCE.
${c(A)}
// Ordinary kriging estimate at one query point. center is the absolute position (used only to
// find index cells); localCenter and origin give the same point as an offset from origin, and
// every distance is a difference of offsets. excludeRow removes one sample (leave-one-out).
fn estimateKriging(center: vec2f, localCenter: vec2f, origin: vec2f, excludeRow: u32) -> KrigingEstimate {
  let radius = params[paramsOffset + 4u];
  let neighborLimitValue = params[paramsOffset + 5u];
  let modelCode = u32(clamp(params[paramsOffset + 6u], 0.0, 4.0));
  let nugget = params[paramsOffset + 7u];
  let sill = params[paramsOffset + 8u];
  let range = params[paramsOffset + 9u];
  let minimumNeighborValue = params[paramsOffset + 10u];
  let neighborLimit = u32(clamp(select(0.0, neighborLimitValue, isFiniteValue(neighborLimitValue)), 0.0, f32(MAX_NEIGHBORS)));
  let minimumNeighbors = max(u32(clamp(select(1.0, minimumNeighborValue, isFiniteValue(minimumNeighborValue)), 0.0, 4294967040.0)), 1u);
  let radiusKey = getRadiusKey(radius);
  let queryBox = getQueryBox(center, radius);
  let queryMin = queryBox.minimum;
  let queryMax = queryBox.maximum;
  let variogramValid = nugget >= 0.0 && sill >= 0.0 && nugget + sill > 0.0 && range > 0.0 && isFiniteValue(range);
  let valid = variogramValid && neighborLimit > 0u && isFiniteValue(center.x) && isFiniteValue(center.y) &&
    radius >= 0.0 && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX);
  var estimate = KrigingEstimate(getNaN(), getNaN(), STATUS_INVALID);
  var nearDistances: array<f32, MAX_NEIGHBORS>;
  var nearRows: array<u32, MAX_NEIGHBORS>;
  var nearCount = 0u;
  if (valid) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    ${t(`if (sample == excludeRow) {
          continue;
        }
        let value = sampleValues[sampleValuesOffset + sample];
        if (isNanBits(value)) {
          continue;
        }
        let position = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]);
        let distanceSquared = getSeparationKey(position - origin, localCenter);
        if (!(distanceSquared <= radiusKey)) {
          continue;
        }
        let last = neighborLimit - 1u;
        let isFull = nearCount >= neighborLimit;
        if (!isFull || distanceSquared < nearDistances[last] ||
            (distanceSquared == nearDistances[last] && sample < nearRows[last])) {
          var slotPosition = select(nearCount, last, isFull);
          loop {
            if (slotPosition == 0u) { break; }
            let previousDistance = nearDistances[slotPosition - 1u];
            let previousRow = nearRows[slotPosition - 1u];
            if (previousDistance < distanceSquared ||
                (previousDistance == distanceSquared && previousRow < sample)) {
              break;
            }
            nearDistances[slotPosition] = previousDistance;
            nearRows[slotPosition] = previousRow;
            slotPosition -= 1u;
          }
          nearDistances[slotPosition] = distanceSquared;
          nearRows[slotPosition] = sample;
          nearCount = min(nearCount + 1u, neighborLimit);
        }`)}
  }
  if (nearCount == 0u) {
    return estimate;
  }

  // Coalesce samples at identical positions into one kriging point carrying their mean value.
  // Identical positions have identical distances, and the list order is deterministic, so the
  // sums are too.
  var uniqueCount = 0u;
  var uniquePositions: array<vec2f, MAX_NEIGHBORS>;
  var uniqueDistances: array<f32, MAX_NEIGHBORS>;
  var uniqueValueSums: array<f32, MAX_NEIGHBORS>;
  var uniqueMultiplicities: array<f32, MAX_NEIGHBORS>;
  for (var i = 0u; i < nearCount; i++) {
    let row = nearRows[i];
    let localPosition = vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]) - origin;
    let value = sampleValues[sampleValuesOffset + row];
    var matched = false;
    for (var j = 0u; j < uniqueCount; j++) {
      if (all(uniquePositions[j] == localPosition)) {
        uniqueValueSums[j] += value;
        uniqueMultiplicities[j] += 1.0;
        matched = true;
        break;
      }
    }
    if (!matched) {
      uniquePositions[uniqueCount] = localPosition;
      uniqueDistances[uniqueCount] = nearDistances[i];
      uniqueValueSums[uniqueCount] = value;
      uniqueMultiplicities[uniqueCount] = 1.0;
      uniqueCount += 1u;
    }
  }

  // Exact hit: the nearest distinct position coincides with the query.
  if (isExactHit(uniqueDistances[0], uniquePositions[0], localCenter)) {
    return KrigingEstimate(uniqueValueSums[0] / uniqueMultiplicities[0], 0.0, STATUS_OK);
  }
  if (uniqueCount < minimumNeighbors) {
    return estimate;
  }

  let n = uniqueCount + 1u;
  var system: array<f32, 289>;
  // The right-hand side, then the weights and the Lagrange multiplier.
  var solution: array<f32, ${A}>;
  var targetGamma: array<f32, MAX_NEIGHBORS>;
  for (var i = 0u; i < uniqueCount; i++) {
    system[i * SYSTEM_STRIDE + i] = 0.0;
    // The variogram matrix is symmetric: one evaluation fills both triangles.
    for (var j = i + 1u; j < uniqueCount; j++) {
      let gamma = evaluateVariogram(getPairMeters(uniquePositions[i], uniquePositions[j]), modelCode, nugget, sill, range);
      system[i * SYSTEM_STRIDE + j] = gamma;
      system[j * SYSTEM_STRIDE + i] = gamma;
    }
    system[i * SYSTEM_STRIDE + uniqueCount] = 1.0;
    system[uniqueCount * SYSTEM_STRIDE + i] = 1.0;
    targetGamma[i] = evaluateVariogram(getSeparationMeters(uniqueDistances[i]), modelCode, nugget, sill, range);
    solution[i] = targetGamma[i];
  }
  system[uniqueCount * SYSTEM_STRIDE + uniqueCount] = 0.0;
  solution[uniqueCount] = 1.0;
  var factorization = factorDenseLU_${A}(&system, n, 0u, RELATIVE_PIVOT_TOLERANCE, PIVOT_RATIO_TOLERANCE);
  if (factorization.status != 0u) {
    estimate.status = STATUS_UNCERTAIN;
    return estimate;
  }
  solveDenseLU_${A}(&system, &factorization, &solution);
  let baseValue = uniqueValueSums[0] / uniqueMultiplicities[0];
  var prediction = 0.0;
  var predictionVariance = solution[uniqueCount];
  for (var i = 0u; i < uniqueCount; i++) {
    let weight = solution[i];
    prediction += weight * (uniqueValueSums[i] / uniqueMultiplicities[i] - baseValue);
    predictionVariance += weight * targetGamma[i];
  }
  // Variance is in units of the total sill and nonnegative for a valid model. Only f32 rounding
  // is clamped; a materially negative value means the weights are unreliable.
  if (!(predictionVariance >= -NEGATIVE_VARIANCE_TOLERANCE) || !isFiniteValue(prediction)) {
    estimate.status = STATUS_UNCERTAIN;
    return estimate;
  }
  return KrigingEstimate(baseValue + prediction, max(predictionVariance, 0.0) * (nugget + sill), STATUS_OK);
}`}export{L as t};
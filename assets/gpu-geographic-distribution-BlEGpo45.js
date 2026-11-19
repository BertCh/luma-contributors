import{n as e,t}from"./spatial-analysis-packed-kernel-CV3ZCYPQ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{t as i}from"./gpu-sort-xRm8l5yn.js";import{a,c as o,t as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as c,n as l,r as u}from"./sorted-segment-sums-Dvmm6OdR.js";import{E as d,S as f,f as p,p as m}from"./gpu-contributor-utils-BIwLQON5.js";import{a as h,u as g}from"./wgsl-common-DfyXiYB0.js";import{s as _,v}from"./spatial-context-DwZEd4EZ.js";import{n as y,t as b}from"./grouped-sum-reduction-Bfd_PmGa.js";import{a as x}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as S,t as C}from"./geographic-distribution-validation-BjMQQJgk.js";import{t as w}from"./spatial-analysis-quotient-CbSG4rNM.js";var T=`
const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;
const TWO_PI: f32 = 6.28318530717959;
// Bit test: NaN and infinity have an all-ones exponent, and a float comparison may be folded away.
${g}
${h}
`,E=1e-12,D=2**24-1;function O(e={},t=new Float32Array(8)){if(t.length<8)throw Error(`Geographic distribution target must hold 8 elements`);let[n,r]=e.origin??[0,0],i=e.standardDeviations??1,a=e.ellipseConvention??`root-two`,o=e.medianTolerance??.001;for(let e of[n,r,i,o])if(!Number.isFinite(e))throw Error(`Geographic distribution origin, standardDeviations and medianTolerance must be finite`);if(i<=0)throw Error(`Geographic distribution standardDeviations must be positive`);if(o<0)throw Error(`Geographic distribution medianTolerance must not be negative`);if(a!==`root-two`&&a!==`standard`)throw Error(`Geographic distribution ellipseConvention must be root-two or standard`);return t.set([n,r,i,a===`root-two`?Math.SQRT2:1,+!!e.orientationOnly,o,0,0]),t}var k=`GPUGeographicDistribution`,A=Object.freeze({..._,measure:`distance`}),j=256,M=256,N=256,P=2147483647,F=256,I=16,L=24,R=64;function z(e,t,n){return`let coincidentWeight = ${n};
  if (coincidentWeight > 0.0) {
    let pullLength = sqrt(${e} * ${e} + ${t} * ${t});
    var gamma = 1.0;
    if (pullLength > 0.0) {
      gamma = min(1.0, getSpatialAnalysisQuotient(coincidentWeight,pullLength));
    }
    stepX = stepX * (1.0 - gamma);
    stepY = stepY * (1.0 - gamma);
  }`}var B=64,V=class{id;props;groupCount;maximumMedianIterations;estimatedCommandNodeCount;polygonVertexCount;reduction;spatialContext;constructor(e){this.id=e.id??`geographic-distribution`,this.props=e;let{id:t}=this;this.spatialContext=v(t,e.spatialContext,A);let{output:n}=e;d(e.positions,[`float32x2`],`${t} positions`);let r=e.positions.length;if(r<1)throw Error(`${t} needs at least one row`);if(r>D)throw Error(`${t} supports fewer than 2^24 rows`);e.weights&&d(e.weights,[`float32`],`${t} weights`),e.groupIds&&d(e.groupIds,[`uint32`],`${t} groupIds`),e.mask&&d(e.mask,[`uint32`],`${t} mask`),e.lineEnds&&d(e.lineEnds,[`float32x2`],`${t} lineEnds`);for(let[n,i]of[[`weights`,e.weights],[`groupIds`,e.groupIds],[`mask`,e.mask],[`lineEnds`,e.lineEnds]])if(i&&i.length!==r)throw Error(`${t} ${n} length must equal positions length`);if(d(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<8)throw Error(`${t} parameters must hold 8 float32 values`);if(this.groupCount=e.groupCount??1,p(t,`groupCount`,this.groupCount,1,D),!e.groupIds&&this.groupCount!==1)throw Error(`${t} groupCount must be 1 without groupIds`);this.maximumMedianIterations=e.maximumMedianIterations??L,p(t,`maximumMedianIterations`,this.maximumMedianIterations,1,N);let i=e.reduction??`auto`;if(i!==`auto`&&i!==`chunked`&&i!==`sorted`)throw Error(`${t} reduction must be auto, chunked, or sorted`);if(i===`chunked`&&this.groupCount>256)throw Error(`${t} chunked reduction supports at most 256 groups`);this.reduction=i===`auto`?this.groupCount<=B?`chunked`:`sorted`:i,this.polygonVertexCount=e.polygonVertexCount??R,p(t,`polygonVertexCount`,this.polygonVertexCount,3);let a=this.groupCount,o=a*this.polygonVertexCount;(n.ellipseVertices||n.circleVertices)&&f(t,`groupCount * polygonVertexCount`,o,P),this.estimatedCommandNodeCount=F+(n.medianCenters||n.medianConverged?this.maximumMedianIterations*I:0),m(t,this.estimatedCommandNodeCount);let s=[[`counts`,n.counts,[`uint32`],a],[`weightSums`,n.weightSums,[`float32`],a],[`meanCenters`,n.meanCenters,[`float32x2`],a],[`medianCenters`,n.medianCenters,[`float32x2`],a],[`medianConverged`,n.medianConverged,[`uint32`],a],[`standardDistances`,n.standardDistances,[`float32`],a],[`ellipses`,n.ellipses,[`float32`],a*3],[`directionalMeans`,n.directionalMeans,[`float32`],a*3],[`ellipseVertices`,n.ellipseVertices,[`float32x2`],o],[`circleVertices`,n.circleVertices,[`float32x2`],o]],c=0;for(let[e,n,r,i]of s)if(n&&(c++,d(n,r,`${t} output.${e}`),n.length<i))throw Error(`${t} output.${e} must hold at least ${i} rows`);if(c===0)throw Error(`${t} needs at least one output`);if(n.directionalMeans&&!e.lineEnds)throw Error(`${t} output.directionalMeans requires lineEnds`);let l={positions:e.positions,weights:e.weights,groupIds:e.groupIds,mask:e.mask,lineEnds:n.directionalMeans?e.lineEnds:void 0,parameters:e.parameters},u={parameters:e.parameters};S(t,l,n,[{reads:u,writes:{counts:n.counts,weightSums:n.weightSums,meanCenters:n.meanCenters}},{reads:u,writes:{standardDistances:n.standardDistances,ellipses:n.ellipses}},{reads:u,writes:{medianCenters:n.medianCenters,medianConverged:n.medianConverged}},{reads:u,writes:{directionalMeans:n.directionalMeans}},{reads:{...u,ellipses:n.ellipses},writes:{ellipseVertices:n.ellipseVertices}},{reads:{...u,standardDistances:n.standardDistances},writes:{circleVertices:n.circleVertices}}])}getCommandNodes(d){let{props:f,id:p,groupCount:m,polygonVertexCount:h}=this,{output:g,positions:_,weights:v,groupIds:S,mask:D,lineEnds:O,parameters:A}=f;e(d,p,g),x(d,p,{positions:_,weights:v,groupIds:S,mask:D,lineEnds:g.directionalMeans?O:void 0,parameters:A,...g});let N=(e,t)=>C(d.device,p,e,t);if(this.reduction===`sorted`&&N(`row scratch`,_.length*4),N(`group mean scratch`,m*8),N(`group offsets scratch`,(m+1)*4),(g.ellipses||g.ellipseVertices)&&N(`ellipse scratch`,m*12),(g.medianCenters||g.medianConverged)&&(N(`median records scratch`,(4*m+1)*4),N(`median dispatch scratch`,36)),this.reduction===`chunked`){let e=Math.min(1024,Math.max(1,Math.ceil(_.length/256)));N(`grouped partial sums scratch`,(g.directionalMeans||g.medianCenters||g.medianConverged?4:3)*m*e*4)}let P=_.length,F=(e,t)=>n(d,`${p}-${e}`,`float32`,t),I=(e,t)=>n(d,`${p}-${e}`,`uint32`,t),L=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),R=(e,t,n=`f32`)=>({name:e,view:t,type:n,access:`read_write`}),B=(e,n,r,i,a=``,o,s=e)=>t(d,{id:`${p}-${e}`,operation:k,variant:s,bindings:r,invocationCount:n,declarations:`${T}
${w}\n${a}`,body:i,...o}),V=`const GROUP_COUNT: u32 = ${m}u;`,H=[],U=this.reduction===`chunked`,W=I(`group-counts`,m),G=F(`weight-sums`,m),K=n(d,`${p}-mean`,`float32x2`,m),q=g.standardDistances||g.ellipses||g.ellipseVertices||g.circleVertices,J=g.standardDistances??(q?F(`standard-distances`,m):void 0),Y=g.ellipses??(g.ellipseVertices?F(`ellipses`,m*3):void 0),X=[L(`positions`,_,`f32`),L(`parameters`,A,`f32`),...v?[L(`weights`,v,`f32`)]:[],...S?[L(`groupIds`,S,`u32`)]:[],...D?[L(`mask`,D,`u32`)]:[]],ee=`${T}
${w}
struct RowData {
  group: u32,
  weight: f32,
  x: f32,
  y: f32
}
// An excluded row has group NO_GROUP (and weight 0); x and y are relative to the local origin.
fn loadRow(row: u32) -> RowData {
  var data = RowData(0xffffffffu, 0.0, 0.0, 0.0);
  let x = positions[positionsOffset + row * 2u];
  let y = positions[positionsOffset + row * 2u + 1u];
  var valid = isFiniteValue(x) && isFiniteValue(y);
  var weight = 1.0;
  ${v?`weight = weights[weightsOffset + row]; valid = valid && isFiniteValue(weight) && weight > 0.0;`:``}
  var group = 0u;
  ${S?`group = groupIds[groupIdsOffset + row]; valid = valid && group < GROUP_COUNT;`:``}
  ${D?`valid = valid && mask[maskOffset + row] != 0u;`:``}
  if (valid) {
    data = RowData(group, weight, x - parameters[parametersOffset], y - parameters[parametersOffset + 1u]);
  }
  return data;
}`,Z=(e,t)=>y(d,{id:`${p}-${e}`,operation:k,rows:P,groupCount:m,bindings:[...X,...t.bindings??[]],declarations:`${ee}\n${t.declarations??``}`,rowSnippet:t.rowSnippet,outputs:t.outputs,countOutput:t.countOutput,finish:t.finish,foldSuffix:t.foldSuffix,gates:t.gates}),Q=F(`sum-x`,m),$=F(`sum-y`,m),te=(e,t,n)=>{let r=[L(`weightSums`,G,`f32`),L(`sumXX`,e,`f32`),L(`sumYY`,t,`f32`),L(`sumXY`,n,`f32`),L(`parameters`,A,`f32`)];return J&&r.push(R(`standardDistances`,J)),Y&&r.push(R(`ellipses`,Y)),B(`spread`,m,r,`let total = weightSums[weightSumsOffset + index];
  let isEmpty = !(total > 0.0);
  let varianceX = getSpatialAnalysisQuotient(sumXX[sumXXOffset + index],total);
  let varianceY = getSpatialAnalysisQuotient(sumYY[sumYYOffset + index],total);
  let covariance = getSpatialAnalysisQuotient(sumXY[sumXYOffset + index],total);
  let multiplier = parameters[parametersOffset + 2u];
  let scale = multiplier * parameters[parametersOffset + 3u];
  ${J?`standardDistances[standardDistancesOffset + index] = select(multiplier * sqrt(max(varianceX + varianceY, 0.0)), getQuietNaN(index), isEmpty);`:``}
  ${Y?`let halfSum = 0.5 * (varianceX + varianceY);
  let halfDifference = 0.5 * (varianceX - varianceY);
  let radius = sqrt(halfDifference * halfDifference + covariance * covariance);
  var majorAngle = 0.0;
  if (radius > 0.0) {
    majorAngle = 0.5 * atan2(covariance, halfDifference);
  }
  var angle = majorAngle + HALF_PI;
  if (angle > HALF_PI) {
    angle = angle - PI;
  }
  let nan = getQuietNaN(index);
  ellipses[ellipsesOffset + index * 3u] = select(angle, nan, isEmpty);
  ellipses[ellipsesOffset + index * 3u + 1u] = select(scale * sqrt(max(halfSum - radius, 0.0)), nan, isEmpty);
  ellipses[ellipsesOffset + index * 3u + 2u] = select(scale * sqrt(max(halfSum + radius, 0.0)), nan, isEmpty);`:``}`)},ne=(e,t,n,r)=>B(`median-start`,m,[L(`mean`,K,`f32`),R(`median`,e),R(`lastMoves`,t),R(`dispatch`,n.view,`u32`),...r?[R(`medianActive`,r,`u32`)]:[]],`median[medianOffset + index * 4u] = 0.0;
  median[medianOffset + index * 4u + 1u] = 0.0;
  median[medianOffset + index * 4u + 2u] = mean[meanOffset + index * 2u];
  median[medianOffset + index * 4u + 3u] = mean[meanOffset + index * 2u + 1u];
  lastMoves[lastMovesOffset + index] = 0.0;
  if (index == 0u) {
    median[medianOffset + 4u * ${m}u] = 0.0;
    ${r?`medianActive[medianActiveOffset] = 0u;`:``}
    ${n.armWGSL}
  }`),re=`fn isMedianMoving(medianX: f32, stepLength: f32, tolerance: f32) -> bool {
  return isFiniteValue(medianX) && !(stepLength <= tolerance);
}`,ie=(e,t)=>{let n=[L(`median`,e,`f32`),L(`lastMoves`,t,`f32`),L(`parameters`,A,`f32`)];return g.medianCenters&&n.push(R(`medianCenters`,g.medianCenters)),g.medianConverged&&n.push(R(`medianConverged`,g.medianConverged,`u32`)),B(`median-publish`,m,n,`${g.medianCenters?`medianCenters[medianCentersOffset + index * 2u] =
    (median[medianOffset + index * 4u + 2u] + median[medianOffset + index * 4u]) + parameters[parametersOffset];
  medianCenters[medianCentersOffset + index * 2u + 1u] =
    (median[medianOffset + index * 4u + 3u] + median[medianOffset + index * 4u + 1u]) + parameters[parametersOffset + 1u];`:``}
  ${g.medianConverged?`let isValid = isFiniteValue(median[medianOffset + index * 4u + 2u]) && isFiniteValue(median[medianOffset + index * 4u]);
  medianConverged[medianConvergedOffset + index] = select(0u, 1u, isValid && lastMoves[lastMovesOffset + index] <= parameters[parametersOffset + 5u]);`:``}`)},ae=(e,t,n,r)=>B(`direction-publish`,m,[L(`sumCosine`,e,`f32`),L(`sumSine`,t,`f32`),L(`sumLength`,n,`f32`),L(`sumWeight`,r,`f32`),L(`parameters`,A,`f32`),R(`directionalMeans`,g.directionalMeans)],`let total = sumWeight[sumWeightOffset + index];
  let isEmpty = !(total > 0.0);
  let cosine = sumCosine[sumCosineOffset + index];
  let sine = sumSine[sumSineOffset + index];
  let resultant = min(length(vec2<f32>(getSpatialAnalysisQuotient(cosine,total),getSpatialAnalysisQuotient(sine,total))), 1.0);
  let hasAngle = cosine != 0.0 || sine != 0.0;
  var angle = atan2(sine, cosine);
  if (parameters[parametersOffset + 4u] > 0.5) {
    angle = 0.5 * angle;
  }
  let nan = getQuietNaN(index);
  directionalMeans[directionalMeansOffset + index * 3u] = select(angle, nan, isEmpty || !hasAngle);
  directionalMeans[directionalMeansOffset + index * 3u + 1u] = select(1.0 - resultant, nan, isEmpty);
  directionalMeans[directionalMeansOffset + index * 3u + 2u] = select(getSpatialAnalysisQuotient(sumLength[sumLengthOffset + index],total), nan, isEmpty);`),oe=()=>B(`mean`,m,[L(`weightSums`,G,`f32`),L(`sumX`,Q,`f32`),L(`sumY`,$,`f32`),R(`mean`,K)],`let total = weightSums[weightSumsOffset + index];
  let isEmpty = !(total > 0.0);
  mean[meanOffset + index * 2u] = select(getSpatialAnalysisQuotient(sumX[sumXOffset + index],total), getQuietNaN(index), isEmpty);
  mean[meanOffset + index * 2u + 1u] = select(getSpatialAnalysisQuotient(sumY[sumYOffset + index],total), getQuietNaN(index), isEmpty);`);if(U){if(H.push(...Z(`mean-sums`,{rowSnippet:`let data = loadRow(row);
      group = data.group;
      value0 = data.weight;
      value1 = data.weight * data.x;
      value2 = data.weight * data.y;`,outputs:[G,Q,$],countOutput:W}),oe()),q){let e=F(`sum-xx`,m),t=F(`sum-yy`,m),n=F(`sum-xy`,m);H.push(...Z(`moments`,{bindings:[L(`mean`,K,`f32`)],rowSnippet:`let data = loadRow(row);
      group = data.group;
      if (group != NO_GROUP) {
        let dx = data.x - mean[meanOffset + group * 2u];
        let dy = data.y - mean[meanOffset + group * 2u + 1u];
        value0 = data.weight * dx * dx;
        value1 = data.weight * dy * dy;
        value2 = data.weight * dx * dy;
      }`,outputs:[e,t,n]}),te(e,t,n))}if(g.medianCenters||g.medianConverged){let e=n(d,`${p}-median`,`float32`,4*m+1),r=F(`median-moves`,m),i=F(`median-sum-weight`,m),s=F(`median-sum-x`,m),c=F(`median-sum-y`,m),l=F(`median-sum-coincident`,m),u=o(E),f=b(P,m),h=a(d,{id:`${p}-median-dispatch`,operation:k,shapes:[f.partial,f.fold]});H.push(ne(e,r,h));for(let n=0;n<this.maximumMedianIterations;n++)H.push(...Z(`median-${n}`,{gates:{partial:h.getGate(0),fold:h.getGate(1)},bindings:[L(`median`,e,`f32`)],rowSnippet:`let data = loadRow(row);
      group = data.group;
      if (group != NO_GROUP) {
        let opaqueZero = bitcast<u32>(median[medianOffset + 4u * GROUP_COUNT]);
        let relative = bitcast<vec2f>(bitcast<vec2u>(vec2f(data.x, data.y) -
          vec2f(median[medianOffset + group * 4u + 2u], median[medianOffset + group * 4u + 3u])) ^ vec2u(opaqueZero));
        let dx = relative.x - median[medianOffset + group * 4u];
        let dy = relative.y - median[medianOffset + group * 4u + 1u];
        let distanceSquared = dx * dx + dy * dy;
        if (distanceSquared > ${u}) {
          let inverseWeight = data.weight / sqrt(distanceSquared);
          value0 = inverseWeight;
          value1 = inverseWeight * dx;
          value2 = inverseWeight * dy;
        } else {
          value3 = data.weight;
        }
      }`,outputs:[i,s,c,l],foldSuffix:`update`,finish:{declarations:`${T}
${w}`,bindings:[R(`median`,e),R(`lastMoves`,r)],body:`var stepX = 0.0;
    var stepY = 0.0;
    if (sum0 > 0.0) {
      stepX = getSpatialAnalysisQuotient(sum1,sum0);
      stepY = getSpatialAnalysisQuotient(sum2,sum0);
    }
    ${z(`sum1`,`sum2`,`sum3`)}
    median[medianOffset + group * 4u] = median[medianOffset + group * 4u] + stepX;
    median[medianOffset + group * 4u + 1u] = median[medianOffset + group * 4u + 1u] + stepY;
    lastMoves[lastMovesOffset + group] = sqrt(stepX * stepX + stepY * stepY);`}}),t(d,{id:`${p}-median-${n}-check`,operation:k,variant:`median-check`,bindings:[L(`median`,e,`f32`),L(`lastMoves`,r,`f32`),L(`parameters`,A,`f32`),R(`dispatch`,h.view,`u32`)],invocationCount:j,workgroupSize:j,guardIndex:!1,declarations:`${T}
${w}
${re}
${V}
var<workgroup> moving: atomic<u32>;`,body:`if (localInvocationIndex == 0u) {
    atomicStore(&moving, 0u);
  }
  workgroupBarrier();
  let group = localInvocationIndex;
  if (group < GROUP_COUNT &&
      isMedianMoving(median[medianOffset + group * 4u + 2u] + median[medianOffset + group * 4u], lastMoves[lastMovesOffset + group], parameters[parametersOffset + 5u])) {
    atomicOr(&moving, 1u);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u && atomicLoad(&moving) == 0u) {
    ${h.disarmWGSL}
  }`}));H.push(ie(e,r))}if(g.directionalMeans&&O){let e=F(`direction-sum-cosine`,m),t=F(`direction-sum-sine`,m),n=F(`direction-sum-length`,m),r=F(`direction-sum-weight`,m);H.push(...Z(`direction`,{bindings:[L(`lineEnds`,O,`f32`)],declarations:`
fn getLineWeight(weight: f32, dx: f32, dy: f32) -> f32 {
  let lengthSquared = dx * dx + dy * dy;
  let isValid = weight > 0.0 && isFiniteValue(dx) && isFiniteValue(dy) && isFiniteValue(lengthSquared) && lengthSquared > 0.0;
  return select(0.0, weight, isValid);
}`,rowSnippet:`let data = loadRow(row);
      group = data.group;
      if (group != NO_GROUP) {
        let dx = lineEnds[lineEndsOffset + row * 2u] - positions[positionsOffset + row * 2u];
        let dy = lineEnds[lineEndsOffset + row * 2u + 1u] - positions[positionsOffset + row * 2u + 1u];
        let lineWeight = getLineWeight(data.weight, dx, dy);
        if (lineWeight > 0.0) {
          let lengthSquared = dx * dx + dy * dy;
          var cosine = 0.0;
          var sine = 0.0;
          if (parameters[parametersOffset + 4u] > 0.5) {
            cosine = getSpatialAnalysisQuotient(dx * dx - dy * dy,lengthSquared);
            sine = getSpatialAnalysisQuotient(2.0 * dx * dy,lengthSquared);
          } else {
            let lineLength = sqrt(lengthSquared);
            cosine = dx / lineLength;
            sine = dy / lineLength;
          }
          value0 = lineWeight * cosine;
          value1 = lineWeight * sine;
          value2 = lineWeight * sqrt(lengthSquared);
          value3 = lineWeight;
        }
      }`,outputs:[e,t,n,r]}),ae(e,t,n,r))}}else{let e=I(`sort-keys`,P),f=I(`sort-indices`,P),h=F(`row-weights`,P),y=[L(`positions`,_,`f32`)];v&&y.push(L(`weights`,v,`f32`)),S&&y.push(L(`groupIds`,S,`u32`)),D&&y.push(L(`mask`,D,`u32`)),y.push(R(`sortKeys`,e,`u32`),R(`sortIndices`,f,`u32`),R(`rowWeights`,h)),H.push(B(`prepare`,P,y,`let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var valid = isFiniteValue(x) && isFiniteValue(y);
  var weight = 1.0;
  ${v?`weight = weights[weightsOffset + index]; valid = valid && isFiniteValue(weight) && weight > 0.0;`:``}
  var group = 0u;
  ${S?`group = groupIds[groupIdsOffset + index]; valid = valid && group < GROUP_COUNT;`:``}
  ${D?`valid = valid && mask[maskOffset + index] != 0u;`:``}
  sortKeys[sortKeysOffset + index] = select(GROUP_COUNT, group, valid);
  sortIndices[sortIndicesOffset + index] = index;
  rowWeights[rowWeightsOffset + index] = select(0.0, weight, valid);`,V),s(d,{id:`${p}-zero-counts`,operation:k,view:W,type:`u32`,value:`0u`}),B(`count`,P,[L(`sortKeys`,e,`u32`),R(`groupCounts`,W,`atomic<u32>`)],`let group = sortKeys[sortKeysOffset + index];
  if (group < GROUP_COUNT) {
    atomicAdd(&groupCounts[groupCountsOffset + group], 1u);
  }`,V));let b=I(`sorted-keys`,P),x=I(`sorted-indices`,P);H.push(...new i({id:`${p}-sort`,keys:e,values:f,outputKeys:b,outputValues:x,keyBits:c(m)}).getCommandNodes(d));let C=I(`segment-offsets`,m+1);H.push(...new r({id:`${p}-segment-scan`,input:W,output:C,mode:`exclusive`}).getCommandNodes(d),B(`segment-total`,1,[L(`counts`,W,`u32`),R(`segmentOffsets`,C,`u32`)],`segmentOffsets[segmentOffsetsOffset + ${m}u] =
    segmentOffsets[segmentOffsetsOffset + ${m-1}u] + counts[countsOffset + ${m-1}u];`));let j=(e,t,n,r)=>l(d,{id:`${p}-sum-${e}`,operation:k,segmentCount:m,input:t,segmentOffsets:C,output:n,...r}),N=F(`sorted-x`,P),U=F(`sorted-y`,P),J=F(`sorted-weights`,P),Y=F(`weighted-x`,P),X=F(`weighted-y`,P);if(H.push(B(`gather`,P,[L(`sortedIndices`,x,`u32`),L(`positions`,_,`f32`),L(`rowWeights`,h,`f32`),L(`parameters`,A,`f32`),R(`sortedX`,N),R(`sortedY`,U),R(`sortedWeights`,J)],`let row = sortedIndices[sortedIndicesOffset + index];
  let weight = rowWeights[rowWeightsOffset + row];
  let isIncluded = weight > 0.0;
  sortedX[sortedXOffset + index] = select(0.0, positions[positionsOffset + row * 2u] - parameters[parametersOffset], isIncluded);
  sortedY[sortedYOffset + index] = select(0.0, positions[positionsOffset + row * 2u + 1u] - parameters[parametersOffset + 1u], isIncluded);
  sortedWeights[sortedWeightsOffset + index] = weight;`),B(`weighted-coordinates`,P,[L(`sortedX`,N,`f32`),L(`sortedY`,U,`f32`),L(`sortedWeights`,J,`f32`),R(`weightedX`,Y),R(`weightedY`,X)],`let weight = sortedWeights[sortedWeightsOffset + index];
  weightedX[weightedXOffset + index] = weight * sortedX[sortedXOffset + index];
  weightedY[weightedYOffset + index] = weight * sortedY[sortedYOffset + index];`),j(`weight`,J,G),j(`x`,Y,Q),j(`y`,X,$)),H.push(oe()),q){let e=F(`contribution-xx`,P),t=F(`contribution-yy`,P),n=F(`contribution-xy`,P),r=F(`sum-xx`,m),i=F(`sum-yy`,m),a=F(`sum-xy`,m);H.push(B(`moments-diagonal`,P,[L(`sortedKeys`,b,`u32`),L(`sortedX`,N,`f32`),L(`sortedY`,U,`f32`),L(`sortedWeights`,J,`f32`),L(`mean`,K,`f32`),R(`contributionXX`,e),R(`contributionYY`,t)],`let group = sortedKeys[sortedKeysOffset + index];
  var xx = 0.0;
  var yy = 0.0;
  if (group < GROUP_COUNT) {
    let weight = sortedWeights[sortedWeightsOffset + index];
    let dx = sortedX[sortedXOffset + index] - mean[meanOffset + group * 2u];
    let dy = sortedY[sortedYOffset + index] - mean[meanOffset + group * 2u + 1u];
    xx = weight * dx * dx;
    yy = weight * dy * dy;
  }
  contributionXX[contributionXXOffset + index] = xx;
  contributionYY[contributionYYOffset + index] = yy;`,V),B(`moments-cross`,P,[L(`sortedKeys`,b,`u32`),L(`sortedX`,N,`f32`),L(`sortedY`,U,`f32`),L(`sortedWeights`,J,`f32`),L(`mean`,K,`f32`),R(`contributionXY`,n)],`let group = sortedKeys[sortedKeysOffset + index];
  var xy = 0.0;
  if (group < GROUP_COUNT) {
    let weight = sortedWeights[sortedWeightsOffset + index];
    let dx = sortedX[sortedXOffset + index] - mean[meanOffset + group * 2u];
    let dy = sortedY[sortedYOffset + index] - mean[meanOffset + group * 2u + 1u];
    xy = weight * dx * dy;
  }
  contributionXY[contributionXYOffset + index] = xy;`,V),j(`xx`,e,r),j(`yy`,t,i),j(`xy`,n,a)),H.push(te(r,i,a))}if(g.medianCenters||g.medianConverged){let e=n(d,`${p}-median`,`float32`,4*m+1),r=F(`median-moves`,m),i=F(`median-contribution-weight`,P),s=F(`median-contribution-x`,P),c=F(`median-contribution-y`,P),l=F(`median-sum-weight`,m),f=F(`median-sum-x`,m),h=F(`median-sum-y`,m),g=F(`median-contribution-coincident`,P),_=F(`median-sum-coincident`,m),v=o(E),y=a(d,{id:`${p}-median-dispatch`,operation:k,shapes:[[P],[m,M],u(m)]}),x=I(`median-medianActive`,1);H.push(ne(e,r,y,x));for(let n=0;n<this.maximumMedianIterations;n++){let a=`median-${n}`;H.push(B(`${a}-x`,P,[L(`sortedKeys`,b,`u32`),L(`sortedX`,N,`f32`),L(`sortedY`,U,`f32`),L(`sortedWeights`,J,`f32`),L(`median`,e,`f32`),R(`contributionWeight`,i),R(`contributionX`,s),R(`contributionCoincident`,g)],`let group = sortedKeys[sortedKeysOffset + index];
  var inverseWeight = 0.0;
  var deltaX = 0.0;
  var coincidentWeight = 0.0;
  if (group < GROUP_COUNT) {
    let opaqueZero = bitcast<u32>(median[medianOffset + 4u * GROUP_COUNT]);
    let relative = bitcast<vec2f>(bitcast<vec2u>(vec2f(sortedX[sortedXOffset + index], sortedY[sortedYOffset + index]) -
      vec2f(median[medianOffset + group * 4u + 2u], median[medianOffset + group * 4u + 3u])) ^ vec2u(opaqueZero));
    let dx = relative.x - median[medianOffset + group * 4u];
    let dy = relative.y - median[medianOffset + group * 4u + 1u];
    let distanceSquared = dx * dx + dy * dy;
    if (distanceSquared > ${v}) {
      inverseWeight = sortedWeights[sortedWeightsOffset + index] / sqrt(distanceSquared);
      deltaX = inverseWeight * dx;
    } else {
      coincidentWeight = sortedWeights[sortedWeightsOffset + index];
    }
  }
  contributionWeight[contributionWeightOffset + index] = inverseWeight;
  contributionX[contributionXOffset + index] = deltaX;
  contributionCoincident[contributionCoincidentOffset + index] = coincidentWeight;`,V,y.getGate(0),`median-x`),B(`${a}-y`,P,[L(`sortedKeys`,b,`u32`),L(`sortedY`,U,`f32`),L(`contributionWeight`,i,`f32`),L(`median`,e,`f32`),R(`contributionY`,c)],`let group = sortedKeys[sortedKeysOffset + index];
  var deltaY = 0.0;
  if (group < GROUP_COUNT) {
    deltaY = contributionWeight[contributionWeightOffset + index] * (bitcast<f32>(bitcast<u32>(sortedY[sortedYOffset + index] - median[medianOffset + group * 4u + 3u]) ^
      bitcast<u32>(median[medianOffset + 4u * GROUP_COUNT])) - median[medianOffset + group * 4u + 1u]);
  }
  contributionY[contributionYOffset + index] = deltaY;`,V,y.getGate(0),`median-y`),j(`${a}-weight`,i,l,y.getGate(2)),j(`${a}-x`,s,f,y.getGate(2)),j(`${a}-y`,c,h,y.getGate(2)),j(`${a}-coincident`,g,_,y.getGate(2)),t(d,{id:`${p}-${a}-update`,operation:k,variant:`median-update`,bindings:[L(`sumWeight`,l,`f32`),L(`sumX`,f,`f32`),L(`sumY`,h,`f32`),L(`sumCoincident`,_,`f32`),R(`median`,e),R(`lastMoves`,r),L(`parameters`,A,`f32`),R(`medianActive`,x,`atomic<u32>`)],invocationCount:m,workgroupSize:M,guardIndex:!1,...y.getGate(1),declarations:`${T}
${w}
${re}
var<workgroup> moving: atomic<u32>;`,body:`if (localInvocationIndex == 0u) {
    atomicStore(&moving, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
  let total = sumWeight[sumWeightOffset + index];
  var stepX = 0.0;
  var stepY = 0.0;
  if (total > 0.0) {
    stepX = getSpatialAnalysisQuotient(sumX[sumXOffset + index],total);
    stepY = getSpatialAnalysisQuotient(sumY[sumYOffset + index],total);
  }
  ${z(`sumX[sumXOffset + index]`,`sumY[sumYOffset + index]`,`sumCoincident[sumCoincidentOffset + index]`)}
  let nextX = median[medianOffset + index * 4u] + stepX;
  median[medianOffset + index * 4u] = nextX;
  median[medianOffset + index * 4u + 1u] = median[medianOffset + index * 4u + 1u] + stepY;
  let stepLength = sqrt(stepX * stepX + stepY * stepY);
  lastMoves[lastMovesOffset + index] = stepLength;
  if (isMedianMoving(median[medianOffset + index * 4u + 2u] + nextX, stepLength, parameters[parametersOffset + 5u])) {
    atomicOr(&moving, 1u);
  }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u && atomicLoad(&moving) != 0u) {
    atomicOr(&medianActive[medianActiveOffset], 1u);
  }`}),t(d,{id:`${p}-${a}-check`,operation:k,variant:`median-check`,bindings:[R(`medianActive`,x,`u32`),R(`dispatch`,y.view,`u32`)],invocationCount:1,body:`if (medianActive[medianActiveOffset] == 0u) {
    ${y.disarmWGSL}
  }
  medianActive[medianActiveOffset] = 0u;`}))}H.push(ie(e,r))}if(g.directionalMeans&&O){let e=F(`direction-contribution-cosine`,P),t=F(`direction-contribution-sine`,P),n=F(`direction-contribution-length`,P),r=F(`direction-contribution-weight`,P),i=F(`direction-sum-cosine`,m),a=F(`direction-sum-sine`,m),o=F(`direction-sum-length`,m),s=F(`direction-sum-weight`,m),c=`
fn getLineWeight(weight: f32, dx: f32, dy: f32) -> f32 {
  let lengthSquared = dx * dx + dy * dy;
  let isValid = weight > 0.0 && isFiniteValue(dx) && isFiniteValue(dy) && isFiniteValue(lengthSquared) && lengthSquared > 0.0;
  return select(0.0, weight, isValid);
}`,l=e=>[L(`sortedIndices`,x,`u32`),L(`positions`,_,`f32`),L(`lineEnds`,O,`f32`),L(`rowWeights`,h,`f32`),...e],u=`let row = sortedIndices[sortedIndicesOffset + index];
  let dx = lineEnds[lineEndsOffset + row * 2u] - positions[positionsOffset + row * 2u];
  let dy = lineEnds[lineEndsOffset + row * 2u + 1u] - positions[positionsOffset + row * 2u + 1u];
  let weight = getLineWeight(rowWeights[rowWeightsOffset + row], dx, dy);`;H.push(B(`direction-angles`,P,l([L(`parameters`,A,`f32`),R(`contributionCosine`,e),R(`contributionSine`,t)]),`${u}
  var cosine = 0.0;
  var sine = 0.0;
  if (weight > 0.0) {
    let lengthSquared = dx * dx + dy * dy;
    if (parameters[parametersOffset + 4u] > 0.5) {
      cosine = getSpatialAnalysisQuotient(dx * dx - dy * dy,lengthSquared);
      sine = getSpatialAnalysisQuotient(2.0 * dx * dy,lengthSquared);
    } else {
      let lineLength = sqrt(lengthSquared);
      cosine = dx / lineLength;
      sine = dy / lineLength;
    }
  }
  contributionCosine[contributionCosineOffset + index] = weight * cosine;
  contributionSine[contributionSineOffset + index] = weight * sine;`,c),B(`direction-lengths`,P,l([R(`contributionLength`,n),R(`contributionWeight`,r)]),`${u}
  contributionLength[contributionLengthOffset + index] = weight * sqrt(dx * dx + dy * dy) * select(0.0, 1.0, weight > 0.0);
  contributionWeight[contributionWeightOffset + index] = weight;`,c),j(`direction-cosine`,e,i),j(`direction-sine`,t,a),j(`direction-length`,n,o),j(`direction-weight`,r,s),ae(i,a,o,s))}}if(g.counts||g.weightSums||g.meanCenters){let e=[L(`groupCounts`,W,`u32`),L(`weightSums`,G,`f32`),L(`mean`,K,`f32`),L(`parameters`,A,`f32`)];g.counts&&e.push(R(`countsOut`,g.counts,`u32`)),g.weightSums&&e.push(R(`weightSumsOut`,g.weightSums)),g.meanCenters&&e.push(R(`meanCentersOut`,g.meanCenters)),H.push(B(`publish`,m,e,`${g.counts?`countsOut[countsOutOffset + index] = groupCounts[groupCountsOffset + index];`:``}
  ${g.weightSums?`weightSumsOut[weightSumsOutOffset + index] = weightSums[weightSumsOffset + index];`:``}
  ${g.meanCenters?`meanCentersOut[meanCentersOutOffset + index * 2u] = mean[meanOffset + index * 2u] + parameters[parametersOffset];
  meanCentersOut[meanCentersOutOffset + index * 2u + 1u] = mean[meanOffset + index * 2u + 1u] + parameters[parametersOffset + 1u];`:``}`))}let se=`const VERTEX_COUNT: u32 = ${h}u;`;return g.ellipseVertices&&Y&&H.push(B(`ellipse-vertices`,m*h,[L(`ellipses`,Y,`f32`),L(`mean`,K,`f32`),L(`parameters`,A,`f32`),R(`vertices`,g.ellipseVertices)],`let group = index / VERTEX_COUNT;
  let t = TWO_PI * f32(index % VERTEX_COUNT) / f32(VERTEX_COUNT);
  let angle = ellipses[ellipsesOffset + group * 3u];
  let localX = ellipses[ellipsesOffset + group * 3u + 1u] * cos(t);
  let localY = ellipses[ellipsesOffset + group * 3u + 2u] * sin(t);
  vertices[verticesOffset + index * 2u] = mean[meanOffset + group * 2u] + parameters[parametersOffset] + localX * cos(angle) - localY * sin(angle);
  vertices[verticesOffset + index * 2u + 1u] = mean[meanOffset + group * 2u + 1u] + parameters[parametersOffset + 1u] + localX * sin(angle) + localY * cos(angle);`,se)),g.circleVertices&&J&&H.push(B(`circle-vertices`,m*h,[L(`standardDistances`,J,`f32`),L(`mean`,K,`f32`),L(`parameters`,A,`f32`),R(`vertices`,g.circleVertices)],`let group = index / VERTEX_COUNT;
  let t = TWO_PI * f32(index % VERTEX_COUNT) / f32(VERTEX_COUNT);
  let radius = standardDistances[standardDistancesOffset + group];
  vertices[verticesOffset + index * 2u] = mean[meanOffset + group * 2u] + parameters[parametersOffset] + radius * cos(t);
  vertices[verticesOffset + index * 2u + 1u] = mean[meanOffset + group * 2u + 1u] + parameters[parametersOffset + 1u] + radius * sin(t);`,se)),H.flat()}};export{O as n,V as t};
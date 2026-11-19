import{d as e,r as t,u as n}from"./kernel-C3tfQOn4.js";import{c as r,o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,T as o,f as s,g as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{r as d,t as f}from"./gpu-multiple-testing-adjustment-Bghjpx33.js";import{t as p}from"./spatial-statistics-row-dispatch-DNaskhxv.js";import{i as m,s as h}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as g,c as _,f as v,i as y,l as b,n as x,o as S,s as C,t as w}from"./spatial-autocorrelation-kernels-CCrGBZuG.js";import{i as T,t as E}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{o as D,s as O}from"./spatial-weights-CqWysstN.js";import{n as k,t as A}from"./spatial-autocorrelation-status-aaIIJ64M.js";var j=2**-126;function M(e,t){c(e,`selfWeight`,t,0),o(e,t===0||Math.fround(t)>=j,`selfWeight must be 0 (Gi) or round to a finite normal float32 >= ${j} (Gi*)`)}function N(e={},t=new Float32Array(1)){let n=`getGPUHotSpotAnalysisParameterValues`;return s(n,`target.length`,t.length,1),M(n,e.selfWeight??1),t[0]=e.selfWeight??1,t}var P=8,F=`GPUHotSpotAnalysis`,I=class{id;props;multipleTesting;islands;minimumNeighbors;selfWeightValues;constructor(t){this.id=t.id??`hot-spot-analysis`,this.props=t;let r=this.id;this.islands=O(r,t.islands),this.minimumNeighbors=g(r,t.minimumNeighbors),this.multipleTesting=d(r,t.multipleTesting);let i=C({...t,id:r,globalCounts:t.output.globalCounts,globalStatistics:t.output.globalStatistics}),o=t.selfWeight??1;typeof o==`number`?(M(r,o),this.selfWeightValues=N({selfWeight:o})):(a(o,[`float32`],`${r} selfWeight`),s(r,`selfWeight.length`,o.length,1)),e(t.output.zScores,[`float32`],`${r} zScores`),t.output.bins&&e(t.output.bins,[`sint32`],`${r} bins`),t.output.pValues&&e(t.output.pValues,[`float32`],`${r} pValues`),t.output.adjustedPValues&&e(t.output.adjustedPValues,[`float32`],`${r} adjustedPValues`),t.output.neighborCounts&&n(t.output.neighborCounts,`${r} neighborCounts`);for(let[e,n]of[[`zScores`,t.output.zScores],[`bins`,t.output.bins],[`pValues`,t.output.pValues],[`adjustedPValues`,t.output.adjustedPValues],[`neighborCounts`,t.output.neighborCounts]])if(n&&n.length!==i)throw Error(`${r} ${e} length must equal the weights row count`);k(r,t.output.status,[t.output.zScores,t.output.pValues,t.output.adjustedPValues,t.output.neighborCounts,t.output.globalStatistics,t.output.globalCounts,t.output.bins]),u(r,[t.output.zScores,t.output.bins,t.output.pValues,t.output.adjustedPValues,t.output.neighborCounts,t.output.globalStatistics,t.output.globalCounts,...Object.values(t.output.status??{})],[...T(t.weights),t.weights.status?.overflow,t.weights.status?.candidateOverflow,t.values,t.parameters,typeof t.selfWeight==`object`?t.selfWeight:void 0,t.mask]),m(r,{zScores:t.output.zScores,bins:t.output.bins,pValues:t.output.pValues,adjustedPValues:t.output.adjustedPValues,neighborCounts:t.output.neighborCounts,globalCounts:t.output.globalCounts,globalStatistics:t.output.globalStatistics,...Object.fromEntries(Object.entries(t.output.status??{}).map(([e,t])=>[`status.${e}`,t])),status:void 0})}getInferenceMetadata(e={}){return v(e),{alternative:`two-sided`,permutationCount:0,multipleTesting:this.multipleTesting,testedRowsDefinition:`included rows with at least ${this.minimumNeighbors} active neighbor${this.minimumNeighbors===1?``:`s`}${this.islands===`exclude`?` (islands excluded)`:``} and a defined z-score`}}getCommandNodes(e){let{id:n,props:a}=this,{weights:o,parameters:s}=a,{zScores:c}=a.output,u=E(o);l(n,e,[...T(o),o.status?.overflow,o.status?.candidateOverflow,a.values,s,typeof a.selfWeight==`object`?a.selfWeight:void 0,a.mask,c,a.output.bins,a.output.pValues,a.output.adjustedPValues,a.output.neighborCounts,a.output.globalStatistics,a.output.globalCounts,...Object.values(a.output.status??{})]),S(e,{...a,id:n},[a.output.zScores,a.output.pValues,a.output.adjustedPValues,a.output.neighborCounts,a.output.globalStatistics,a.output.globalCounts,a.output.bins,...Object.values(a.output.status??{}),typeof a.selfWeight==`object`?a.selfWeight:void 0]);let d=A(e,n,F,a),m=h(e,`${n}-self-weight-parameters`,this.selfWeightValues??a.selfWeight),g=a.values.length,v=x(e,{...a,globalCounts:a.output.globalCounts,globalStatistics:a.output.globalStatistics,id:n,islands:this.islands,operation:F}),C=[...d,...v.nodes,...m.nodes],O=w,k=[{name:`selfWeightParameters`,view:m.parameters,type:`f32`,access:`read`},{name:`statistics`,view:v.statistics,type:`f32`,access:`read`},{name:`counts`,view:v.counts,type:`u32`,access:`read`},{name:`zScores`,view:c,type:`f32`,access:`read_write`}];a.output.neighborCounts&&k.push({name:`neighborCounts`,view:a.output.neighborCounts,type:`u32`,access:`read_write`});let j=this.multipleTesting!==`none`,M=!!(a.output.bins||a.output.pValues||a.output.adjustedPValues),N=j&&!!(a.output.bins||a.output.adjustedPValues),I=Number(!!a.output.bins)+Number(!!a.output.pValues)+Number(!!a.output.adjustedPValues),L=M&&!N&&k.length+3+I<=P,R=N?a.output.pValues??t(e,`${n}-p-values`,`float32`,g):void 0,z=N?a.output.adjustedPValues??t(e,`${n}-adjusted-p-values`,`float32`,g):void 0,[B,V,H]=_.map(r),[U,W,G]=b.map(r),K=[];a.output.bins&&K.push({name:`bins`,view:a.output.bins,type:`i32`,access:`read_write`}),a.output.pValues&&K.push({name:`pValues`,view:a.output.pValues,type:`f32`,access:`read_write`}),a.output.adjustedPValues&&K.push({name:`adjustedPValues`,view:a.output.adjustedPValues,type:`f32`,access:`read_write`});let q=`let finite = isFiniteFloat(zScore);
  var level = 0;
  if (finite) {
    let absoluteZ = abs(zScore);
    level = select(select(select(0, 1, absoluteZ >= ${B}), 2, absoluteZ >= ${V}), 3, absoluteZ >= ${H});
  }
  ${a.output.bins?`bins[binsOffset + index] = select(level, -level, zScore < 0.0);`:``}
  ${a.output.pValues||a.output.adjustedPValues?`let pValue = select(getQuietNaN(index), getTwoSidedPValue(zScore), finite);
  ${a.output.pValues?`pValues[pValuesOffset + index] = pValue;`:``}
  ${a.output.adjustedPValues?`adjustedPValues[adjustedPValuesOffset + index] = pValue;`:``}`:``}`;return L&&k.push(...K),C.push(...u.map(t=>p(e,{rowBase:t.rowStart,id:`${n}-neighbors${t.idSuffix}`,operation:F,variant:`gi${L?`-classify`:``}${N?`-p-value`:``}`,bindings:[{name:`offsets`,view:t.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t.weights.weights,type:`f32`,access:`read`},...k],invocationCount:t.rowCount,declarations:`${O}
const MOMENTS: u32 = ${g}u;
const ROWS: u32 = ${g}u;
const MINIMUM_NEIGHBORS: u32 = ${this.minimumNeighbors}u;
const STANDARDIZE: bool = ${D(o)};`,body:`let centered = statistics[statisticsOffset + index];
  var zScore = getQuietNaN(index);
  var neighborCount = 0u;
  if (isFiniteFloat(centered)) {
    var weightedSum = 0.0;
    var weightSum = 0.0;
    var squareSum = 0.0;
    ${y(`weightedSum += weight * statistics[statisticsOffset + neighbor];
      weightSum += weight;
      squareSum += weight * weight;
      neighborCount++;`,`WEIGHTS_ROW_START`)}
    if (STANDARDIZE && weightSum > 0.0) {
      // Row-standardized weights are divided by what remains of their row sum.
      weightedSum = weightedSum / weightSum;
      squareSum = squareSum / (weightSum * weightSum);
      weightSum = 1.0;
    }
    let exactCount = counts[countsOffset];
    let count = f32(exactCount);
    let variance = statistics[statisticsOffset + MOMENTS + 2u];
    if (neighborCount >= MINIMUM_NEIGHBORS) {
      let selfWeight = selfWeightParameters[selfWeightParametersOffset];
      if (selfWeight == 0.0) {
      // Ord-Getis Gi: standardize with the n - 1 other observations. The centered values of the
      // population sum to zero, so the others sum to -centered.
      let othersCount = max(exactCount, 1u) - 1u;
      let others = f32(othersCount);
      let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
      let otherMean = divideBySpatialCount(-centered, othersCount);
      let otherVariance = divideBySpatialCount(sumOfSquares - centered * centered, othersCount) - otherMean * otherMean;
      let spread = divideBySpatialCount(multiplyBySpatialCount(squareSum, othersCount) - weightSum * weightSum, max(othersCount, 1u) - 1u);
      if (others >= 2.0 && otherVariance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
        zScore = (weightedSum - weightSum * otherMean) / (sqrt(otherVariance) * sqrt(spread));
      }
      } else {
      weightedSum += selfWeight * centered;
      weightSum += selfWeight;
      squareSum += selfWeight * selfWeight;
      let spread = divideBySpatialCount(multiplyBySpatialCount(squareSum, exactCount) - weightSum * weightSum, max(exactCount, 1u) - 1u);
      if (count >= 2.0 && variance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
        zScore = weightedSum / (sqrt(variance) * sqrt(spread));
      }
      }
    }
  }
  zScores[zScoresOffset + index] = zScore;
  ${a.output.neighborCounts?`neighborCounts[neighborCountsOffset + index] = neighborCount;`:``}
  
  ${L?q:``}`}))),M&&!N&&!L&&C.push(i(e,{id:`${n}-classify`,operation:F,variant:`classify`,bindings:[{name:`zScores`,view:c,type:`f32`,access:`read`},...K],invocationCount:g,declarations:O,body:`let zScore = zScores[zScoresOffset + index];
  ${q}`})),R&&C.push(i(e,{id:`${n}-analytic-p-values`,operation:F,invocationCount:g,bindings:[{name:`zScores`,view:c,type:`f32`,access:`read`},{name:`pValues`,view:R,type:`f32`,access:`read_write`}],declarations:w,body:`let score = zScores[zScoresOffset + index];
  pValues[pValuesOffset + index] = select(getQuietNaN(index), getTwoSidedPValue(score), isFiniteFloat(score));`})),N&&C.push(...new f({id:`${n}-multiple-testing`,pValues:R,method:this.multipleTesting,output:{adjustedPValues:z}}).getCommandNodes(e)),!N||!a.output.bins||C.push(i(e,{id:`${n}-classify`,operation:F,variant:`classify-adjusted`,bindings:[{name:`zScores`,view:c,type:`f32`,access:`read`},{name:`adjustedPValues`,view:z,type:`f32`,access:`read`},{name:`bins`,view:a.output.bins,type:`i32`,access:`read_write`}],invocationCount:g,declarations:w,body:`let zScore = zScores[zScoresOffset + index];
  let adjusted = adjustedPValues[adjustedPValuesOffset + index];
  var level = 0;
  if (isFiniteFloat(zScore) && isFiniteFloat(adjusted)) {
    level = select(select(select(0, 1, adjusted <= ${U}), 2, adjusted <= ${W}), 3, adjusted <= ${G});
  }
  bins[binsOffset + index] = select(level, -level, zScore < 0.0);`})),C}};export{I as t};
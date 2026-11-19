import{d as e,r as t,u as n}from"./kernel-C3tfQOn4.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{x as i,y as a}from"./gpu-contributor-utils-BIwLQON5.js";import{r as o,t as s}from"./gpu-multiple-testing-adjustment-Bghjpx33.js";import{t as c}from"./spatial-statistics-row-dispatch-DNaskhxv.js";import{i as l}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as u,f as d,i as f,n as p,o as m,r as h,s as g,t as _,u as v}from"./spatial-autocorrelation-kernels-CCrGBZuG.js";import{i as y,t as b}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{o as x,s as S}from"./spatial-weights-CqWysstN.js";import{n as C,t as w}from"./spatial-autocorrelation-status-aaIIJ64M.js";var T=`GPULocalMoran`,E=class{id;props;multipleTesting;islands;minimumNeighbors;constructor(t){this.id=t.id??`local-moran`,this.props=t;let r=this.id;this.islands=S(r,t.islands),this.minimumNeighbors=u(r,t.minimumNeighbors),this.multipleTesting=o(r,t.multipleTesting);let i=g({...t,id:r,globalCounts:t.output.globalCounts,globalStatistics:t.output.globalStatistics});for(let[n,i]of[[`zScores`,t.output.zScores],[`localI`,t.output.localI],[`spatialLag`,t.output.spatialLag],[`pValues`,t.output.pValues],[`adjustedPValues`,t.output.adjustedPValues]])i&&e(i,[`float32`],`${r} ${n}`);for(let[e,i]of[[`quadrants`,t.output.quadrants],[`neighborCounts`,t.output.neighborCounts]])i&&n(i,`${r} ${e}`);for(let[e,n]of[[`zScores`,t.output.zScores],[`localI`,t.output.localI],[`spatialLag`,t.output.spatialLag],[`quadrants`,t.output.quadrants],[`pValues`,t.output.pValues],[`adjustedPValues`,t.output.adjustedPValues],[`neighborCounts`,t.output.neighborCounts]])if(n&&n.length!==i)throw Error(`${r} ${e} length must equal the weights row count`);if(t.quadrantGating===`none`&&this.multipleTesting!==`none`)throw Error(`${r} quadrantGating 'none' cannot be combined with multipleTesting`);C(r,t.output.status,[t.output.zScores,t.output.pValues,t.output.adjustedPValues,t.output.neighborCounts,t.output.globalStatistics,t.output.globalCounts,t.output.localI,t.output.spatialLag,t.output.quadrants]),a(r,[t.output.zScores,t.output.localI,t.output.spatialLag,t.output.quadrants,t.output.pValues,t.output.adjustedPValues,t.output.neighborCounts,t.output.globalStatistics,t.output.globalCounts,...Object.values(t.output.status??{})],[...y(t.weights),t.weights.status?.overflow,t.weights.status?.candidateOverflow,t.values,t.parameters,t.mask]),l(r,{zScores:t.output.zScores,localI:t.output.localI,spatialLag:t.output.spatialLag,quadrants:t.output.quadrants,pValues:t.output.pValues,adjustedPValues:t.output.adjustedPValues,neighborCounts:t.output.neighborCounts,globalCounts:t.output.globalCounts,globalStatistics:t.output.globalStatistics,...Object.fromEntries(Object.entries(t.output.status??{}).map(([e,t])=>[`status.${e}`,t])),status:void 0})}getInferenceMetadata(e={}){d(e);let t=e.significanceLevel??.05;return{alternative:`two-sided`,permutationCount:0,multipleTesting:this.multipleTesting,significanceLevel:t,testedRowsDefinition:`included rows with at least ${this.minimumNeighbors} active neighbor${this.minimumNeighbors===1?``:`s`}${this.islands===`exclude`?` (islands excluded)`:``}, a nonzero centered value and a defined z-score`}}getCommandNodes(e){let{id:n,props:a}=this,{weights:o,parameters:l}=a,{zScores:u}=a.output,d=b(o);i(n,e,[...y(o),o.status?.overflow,o.status?.candidateOverflow,a.values,l,a.mask,u,a.output.localI,a.output.spatialLag,a.output.quadrants,a.output.pValues,a.output.adjustedPValues,a.output.neighborCounts,a.output.globalStatistics,a.output.globalCounts,...Object.values(a.output.status??{})]),m(e,{...a,id:n},[a.output.zScores,a.output.pValues,a.output.adjustedPValues,a.output.neighborCounts,a.output.globalStatistics,a.output.globalCounts,a.output.localI,a.output.spatialLag,a.output.quadrants,...Object.values(a.output.status??{})]);let g=w(e,n,T,a),S=a.values.length,C=p(e,{...a,globalCounts:a.output.globalCounts,globalStatistics:a.output.globalStatistics,id:n,islands:this.islands,operation:T}),E=[...g,...C.nodes],D=`const ROWS: u32 = ${S}u;
const MOMENTS: u32 = ${S}u;
const MINIMUM_NEIGHBORS: u32 = ${this.minimumNeighbors}u;
const STANDARDIZE: bool = ${x(o)};`,O=`${h()}\n${D}`,k=`${_}\n${D}`,{HIGH_HIGH:A,LOW_HIGH:j,LOW_LOW:M,HIGH_LOW:N,NOT_SIGNIFICANT:P}=v,F=this.multipleTesting!==`none`,I=!!(a.output.quadrants||a.output.pValues||a.output.adjustedPValues),L=!!a.output.quadrants&&a.quadrantGating!==`none`,R=F&&!!(a.output.adjustedPValues||a.output.quadrants)&&I,z=R?a.output.pValues??t(e,`${n}-p-values`,`float32`,S):void 0,B=R?a.output.adjustedPValues??t(e,`${n}-adjusted-p-values`,`float32`,S):void 0,V=e=>I&&!e&&(!R||!!a.output.quadrants),H=e.device.limits.maxStorageBuffersPerShaderStage,U=(e,t)=>6+(a.output.spatialLag||a.output.localI&&!e||V(t)?1:0)+ +!!a.output.neighborCounts+(a.output.localI&&e?1:0)+(t?Number(L)+ +!!a.output.quadrants+ +!!a.output.pValues+ +!!a.output.adjustedPValues:0),W=I&&!R,G=!!a.output.localI,K=W;U(G,K)>H&&(G=!1),U(G,K)>H&&(K=!1,G=!!a.output.localI&&U(!0,!1)<=H);let q=a.output.spatialLag||a.output.localI&&!G||V(K)?a.output.spatialLag??t(e,`${n}-spatial-lag`,`float32`,S):void 0,J=(e,t,n)=>`
  var quadrant = ${P}u;
  if (${n}) {
    if (${e} > 0.0 && ${t} > 0.0) {
      quadrant = ${A}u;
    } else if (${e} < 0.0 && ${t} > 0.0) {
      quadrant = ${j}u;
    } else if (${e} < 0.0 && ${t} < 0.0) {
      quadrant = ${M}u;
    } else if (${e} > 0.0 && ${t} < 0.0) {
      quadrant = ${N}u;
    }
  }
  quadrants[quadrantsOffset + index] = quadrant;`,Y=(e,t,n)=>`let pValue = select(getQuietNaN(index), getTwoSidedPValue(${e}), isFiniteFloat(${e}));
  ${a.output.quadrants?J(t,n,a.quadrantGating===`none`?`isFiniteFloat(${t})`:`isFiniteFloat(${e}) && pValue <= readParameter(0u)`):``}
  ${a.output.pValues?`pValues[pValuesOffset + index] = pValue;`:``}
  ${a.output.adjustedPValues?`adjustedPValues[adjustedPValuesOffset + index] = pValue;`:``}`,X=[{name:`statistics`,view:C.statistics,type:`f32`,access:`read`},{name:`counts`,view:C.counts,type:`u32`,access:`read`},{name:`zScores`,view:u,type:`f32`,access:`read_write`}];if(q&&X.push({name:`spatialLag`,view:q,type:`f32`,access:`read_write`}),a.output.neighborCounts&&X.push({name:`neighborCounts`,view:a.output.neighborCounts,type:`u32`,access:`read_write`}),a.output.localI&&G&&X.push({name:`localI`,view:a.output.localI,type:`f32`,access:`read_write`}),K&&(L&&X.push({name:`parameters`,view:l,type:`f32`,access:`read`}),a.output.quadrants&&X.push({name:`quadrants`,view:a.output.quadrants,type:`u32`,access:`read_write`}),a.output.pValues&&X.push({name:`pValues`,view:a.output.pValues,type:`f32`,access:`read_write`}),a.output.adjustedPValues&&X.push({name:`adjustedPValues`,view:a.output.adjustedPValues,type:`f32`,access:`read_write`})),E.push(...d.map(t=>c(e,{rowBase:t.rowStart,id:`${n}-neighbors${t.idSuffix}`,operation:T,variant:`local-moran${a.output.localI&&G?`-local-i`:``}${K?`-classify`:``}${R?`-p-value`:``}`,bindings:[{name:`offsets`,view:t.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t.weights.weights,type:`f32`,access:`read`},...X],invocationCount:t.rowCount,declarations:K&&L?O:k,body:`let centered = statistics[statisticsOffset + index];
  var zScore = getQuietNaN(index);
  var lag = getQuietNaN(index);
  var neighborCount = 0u;
  if (isFiniteFloat(centered)) {
    var neighborSum = 0.0;
    var weightSum = 0.0;
    var squareSum = 0.0;
    ${f(`neighborSum += weight * statistics[statisticsOffset + neighbor];
      weightSum += weight;
      squareSum += weight * weight;
      neighborCount++;`,`WEIGHTS_ROW_START`)}
    if (STANDARDIZE && weightSum > 0.0) {
      // Row-standardized weights are divided by what remains of their row sum.
      neighborSum = neighborSum / weightSum;
      squareSum = squareSum / (weightSum * weightSum);
      weightSum = 1.0;
    }
    lag = neighborSum;
    let othersCount = max(counts[countsOffset], 1u) - 1u;
    let others = f32(othersCount);
    let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
    let otherMean = divideBySpatialCount(-centered, othersCount);
    let otherVariance = divideBySpatialCount(sumOfSquares - centered * centered, othersCount) - otherMean * otherMean;
    let lagVariance = otherVariance * divideBySpatialCount(multiplyBySpatialCount(squareSum, othersCount) - weightSum * weightSum, max(othersCount, 1u) - 1u);
    if (others >= 2.0 && centered != 0.0 && neighborCount >= MINIMUM_NEIGHBORS &&
        lagVariance > 0.0 && isFiniteFloat(lagVariance)) {
      let lagZ = (neighborSum - weightSum * otherMean) / sqrt(lagVariance);
      zScore = select(lagZ, -lagZ, centered < 0.0);
    }
  }
  zScores[zScoresOffset + index] = zScore;
  ${q?`spatialLag[spatialLagOffset + index] = lag;`:``}
  ${a.output.neighborCounts?`neighborCounts[neighborCountsOffset + index] = neighborCount;`:``}
  
  ${a.output.localI&&G?`{
    let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
    // Constant values (M2 = 0) leave I undefined: NaN, not the 0 of a kept island.
    let scale = select(getQuietNaN(index), divideSpatialCountByValue(max(counts[countsOffset], 1u) - 1u, sumOfSquares), sumOfSquares > 0.0);
    // Excluded rows keep their NaN centered value; kept islands have a zero lag and so I = 0.
    localI[localIOffset + index] = select(select(centered * lag * scale, 0.0, lag == 0.0 && isFiniteFloat(scale)), getQuietNaN(index), !isFiniteFloat(centered));
  }`:``}
  ${K?Y(`zScore`,`centered`,`lag`):``}`}))),a.output.localI&&!G&&E.push(r(e,{id:`${n}-local-i`,operation:T,variant:`local-i`,bindings:[{name:`statistics`,view:C.statistics,type:`f32`,access:`read`},{name:`counts`,view:C.counts,type:`u32`,access:`read`},{name:`spatialLag`,view:q,type:`f32`,access:`read`},{name:`localI`,view:a.output.localI,type:`f32`,access:`read_write`}],invocationCount:S,declarations:k,body:`let centered = statistics[statisticsOffset + index];
  let lag = spatialLag[spatialLagOffset + index];
  let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
  // Constant values (M2 = 0) leave I undefined: NaN, not the 0 of a kept island.
  let scale = select(getQuietNaN(index), divideSpatialCountByValue(max(counts[countsOffset], 1u) - 1u, sumOfSquares), sumOfSquares > 0.0);
  // Excluded rows keep their NaN centered value; kept islands have a zero lag and so I = 0.
  localI[localIOffset + index] = select(select(centered * lag * scale, 0.0, lag == 0.0 && isFiniteFloat(scale)), getQuietNaN(index), !isFiniteFloat(centered));`})),z&&E.push(r(e,{id:`${n}-analytic-p-values`,operation:T,invocationCount:S,bindings:[{name:`zScores`,view:u,type:`f32`,access:`read`},{name:`pValues`,view:z,type:`f32`,access:`read_write`}],declarations:_,body:`let score = zScores[zScoresOffset + index];
  pValues[pValuesOffset + index] = select(getQuietNaN(index), getTwoSidedPValue(score), isFiniteFloat(score));`})),R&&E.push(...new s({id:`${n}-multiple-testing`,pValues:z,method:this.multipleTesting,output:{adjustedPValues:B}}).getCommandNodes(e)),!V(K))return E;let Z=[...L?[{name:`parameters`,view:l,type:`f32`,access:`read`}]:[],{name:`statistics`,view:C.statistics,type:`f32`,access:`read`},{name:`zScores`,view:u,type:`f32`,access:`read`},{name:`spatialLag`,view:q,type:`f32`,access:`read`}];return R&&Z.push({name:`adjustedPValues`,view:B,type:`f32`,access:`read`}),a.output.quadrants&&Z.push({name:`quadrants`,view:a.output.quadrants,type:`u32`,access:`read_write`}),!R&&a.output.pValues&&Z.push({name:`pValues`,view:a.output.pValues,type:`f32`,access:`read_write`}),!R&&a.output.adjustedPValues&&Z.push({name:`adjustedPValues`,view:a.output.adjustedPValues,type:`f32`,access:`read_write`}),E.push(r(e,{id:`${n}-classify`,operation:T,variant:R?`classify-adjusted`:`classify`,bindings:Z,invocationCount:S,declarations:L?h():_,body:`let zScore = zScores[zScoresOffset + index];
  let centered = statistics[statisticsOffset + index];
  let lag = spatialLag[spatialLagOffset + index];
  ${R?`let adjusted = adjustedPValues[adjustedPValuesOffset + index];
  ${J(`centered`,`lag`,`isFiniteFloat(zScore) && isFiniteFloat(adjusted) && adjusted <= readParameter(0u)`)}`:Y(`zScore`,`centered`,`lag`)}`})),E}};export{E as t};
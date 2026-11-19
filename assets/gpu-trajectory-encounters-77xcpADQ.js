import{n as e,t}from"./trajectory-counts-wgsl-CVOZcx6G.js";import{a as n,i as r,n as i,r as a}from"./trajectory-binding-validation-C89oJIAZ.js";import{n as o}from"./common-CshkX-d_.js";import{a as s,m as c,v as l,y as u}from"./spatial-context-DwZEd4EZ.js";import{r as d}from"./kernel-C3tfQOn4.js";import{t as f}from"./gpu-compaction-B6IzcF0P.js";import{t as p}from"./gpu-grid-index-BoHdb71C.js";import{t as m}from"./gpu-sort-xRm8l5yn.js";import{c as h,n as g,o as _}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as v,S as y,f as b,g as x,h as S,m as C,o as ee,v as te,x as w,y as T}from"./gpu-contributor-utils-BIwLQON5.js";import{t as E}from"./gpu-grid-coordinate-DwQPYH7E.js";import{t as D}from"./geodesic-wgsl-I0CYDKcz.js";import{u as O}from"./wgsl-common-DfyXiYB0.js";import{r as k}from"./spatial-join-passes-CNxNpnDa.js";import{i as A}from"./trajectory-common-DFp8lX7h.js";var j=`GPUTrajectoryEncounters`;function M(e){let[t,n,r,i]=e.bounds.map(h);return`const TRACK_COUNT: u32 = ${e.trackCount}u;
const BUCKET_COUNT: u32 = ${e.bucketCount}u;
const HIT_CAPACITY: u32 = ${e.hitCapacity}u;
const PAIR_CAPACITY: u32 = ${e.pairCapacity}u;
const COLUMNS: u32 = ${e.columns}u;
const ROWS: u32 = ${e.rows}u;
const MINIMUM_X: f32 = ${t};
const MINIMUM_Y: f32 = ${n};
const MAXIMUM_X: f32 = ${r};
const MAXIMUM_Y: f32 = ${i};
const CELL_SIZE: f32 = ${h(e.geodesic?e.geodesic.maximumDistance:e.cellSize)};`}function ne(e,t){let n=[{name:`samples`,view:t.samples,type:`f32`,access:`read`}];return t.trackValid&&n.push({name:`trackValid`,view:t.trackValid,type:`u32`,access:`read`}),n.push({name:`positions`,view:t.positions,type:`f32`,access:`read_write`}),_(e,{id:t.id,operation:j,variant:`positions`,bindings:n,invocationCount:t.shape.trackCount*t.shape.bucketCount,declarations:`${t.shape.geodesic?.wrapColumns?O:``}
${M(t.shape)}`,body:`let track = index / BUCKET_COUNT;
  let bucket = index % BUCKET_COUNT;
  var x = samples[samplesOffset + index * 2u];
  var y = samples[samplesOffset + index * 2u + 1u];
  ${t.trackValid?`if (trackValid[trackValidOffset + track] == 0u) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    x = bitcast<f32>(0x7fc00000u | (index & 0u));
    y = x;
  }`:``}
  ${t.shape.geodesic?.wrapColumns?`// Unwrapped longitudes (for example from a segment across the antimeridian) fold into [-180, 180].
  if (isFiniteValue(x)) {
    x = x - 360.0 * round(x / 360.0);
  }`:``}
  positions[positionsOffset + index * 3u] = x;
  positions[positionsOffset + index * 3u + 1u] = y;
  positions[positionsOffset + index * 3u + 2u] = f32(bucket) + 0.5;`})}function N(t,n){return _(t,{id:n.id,operation:j,variant:`scan`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`cellOffsets`,view:n.cellOffsets,type:`u32`,access:`read`},{name:`objectIds`,view:n.objectIds,type:`u32`,access:`read`},{name:`distance`,view:n.distance,type:`f32`,access:`read`},{name:`state`,view:n.state,type:`atomic<u32>`,access:`read_write`},{name:`hitPairs`,view:n.hitPairs,type:`u32`,access:`read_write`},{name:`hitBuckets`,view:n.hitBuckets,type:`u32`,access:`read_write`},{name:`hitDistances`,view:n.hitDistances,type:`f32`,access:`read_write`}],invocationCount:n.shape.trackCount*n.shape.bucketCount,declarations:`${e}
${k}
${n.shape.geodesic?D:``}
${M(n.shape)}
// Use the builder's public rule, including its extreme-domain overflow guard.
${E}`,body:`let track = index / BUCKET_COUNT;
  let bucket = index % BUCKET_COUNT;
  let x = positions[positionsOffset + index * 3u];
  let y = positions[positionsOffset + index * 3u + 1u];
  if (!isFiniteValue(x) || !isFiniteValue(y)) { return; }
  if (x < MINIMUM_X || x > MAXIMUM_X || y < MINIMUM_Y || y > MAXIMUM_Y) { return; }
  let searchDistance = distance[distanceOffset];
  if (!(searchDistance >= 0.0)) { return; }
  let limit = min(searchDistance, CELL_SIZE);
  let column = getCoordinate(x, MINIMUM_X, MAXIMUM_X, COLUMNS);
  let row = getCoordinate(y, MINIMUM_Y, MAXIMUM_Y, ROWS);
  // Half-shell traversal: the own cell plus the four forward neighbours (east, and the three
  // cells of the next row at columns -1, 0, +1). Every unordered pair of adjacent cells is visited
  // from exactly one side, so each pair is tested once instead of twice. Inside the own cell the
  // higher track is the partner. A track has one sample per bucket, so another cell never holds
  // the same track.
  for (var shell = 0u; shell < 5u; shell++) {
    var neighborColumn = column;
    var neighborRow = row;
    ${n.shape.geodesic?.wrapColumns?`// The columns span all longitudes, so the east and west neighbours wrap across the
    // antimeridian (the lattice has at least three columns, so no cell is visited twice).
    if (shell == 1u) { neighborColumn = (column + 1u) % COLUMNS; }
    else if (shell == 2u) { neighborColumn = (column + COLUMNS - 1u) % COLUMNS; neighborRow = row + 1u; }
    else if (shell == 3u) { neighborRow = row + 1u; }
    else if (shell == 4u) { neighborColumn = (column + 1u) % COLUMNS; neighborRow = row + 1u; }`:`if (shell == 1u) { neighborColumn = column + 1u; }
    else if (shell == 2u) { neighborColumn = column - 1u; neighborRow = row + 1u; }
    else if (shell == 3u) { neighborRow = row + 1u; }
    else if (shell == 4u) { neighborColumn = column + 1u; neighborRow = row + 1u; }
    // Unsigned wrap of column - 1u at column 0 gives a value above COLUMNS, rejected here.`}
    if (neighborColumn >= COLUMNS || neighborRow >= ROWS) { continue; }
    let cell = (bucket * ROWS + neighborRow) * COLUMNS + neighborColumn;
    let cellBegin = cellOffsets[cellOffsetsOffset + cell];
    let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
    for (var entry = cellBegin; entry < cellEnd; entry++) {
      let other = objectIds[objectIdsOffset + entry];
      let otherTrack = other / BUCKET_COUNT;
      if (shell == 0u && otherTrack <= track) { continue; }
      ${n.shape.geodesic?`let separation = geodesicHaversineDistance(
        vec2<f32>(x, y),
        vec2<f32>(positions[positionsOffset + other * 3u], positions[positionsOffset + other * 3u + 1u]),
        ${h(n.shape.geodesic.radius)}
      );`:`let offset = vec2f(
        positions[positionsOffset + other * 3u] - x,
        positions[positionsOffset + other * 3u + 1u] - y
      );
      let separation = length(offset);`}
      if (separation <= limit) {
        let slot = reserveTrajectoryCount();
        if (slot < HIT_CAPACITY) {
          hitPairs[hitPairsOffset + slot * 2u] = min(track, otherTrack);
          hitPairs[hitPairsOffset + slot * 2u + 1u] = max(track, otherTrack);
          hitBuckets[hitBucketsOffset + slot] = bucket;
          hitDistances[hitDistancesOffset + slot] = separation;
        }
      }
    }
  }`})}function re(e,t){return _(e,{id:t.id,operation:j,variant:`clear`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read_write`}],invocationCount:1,body:`state[stateOffset] = 0u;`})}function P(e,t){let n=[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`hitPairs`,view:t.hitPairs,type:`u32`,access:`read`}],r;if(t.variant===`bucket`)n.push({name:`hitBuckets`,view:t.hitBuckets,type:`u32`,access:`read`},{name:`identity`,view:t.identity,type:`u32`,access:`read_write`}),r=`identity[identityOffset + index] = index;
  keys[keysOffset + index] = select(0u, hitBuckets[hitBucketsOffset + index], index < activeCount);`;else{n.push({name:`order`,view:t.order,type:`u32`,access:`read`});let e=+(t.variant===`partner`);r=`let slot = order[orderOffset + index];
  keys[keysOffset + index] = select(${t.variant===`partner`?`0u`:`TRACK_COUNT`}, hitPairs[hitPairsOffset + slot * 2u + ${e}u], slot < activeCount);`}return n.push({name:`keys`,view:t.keys,type:`u32`,access:`read_write`}),_(e,{id:t.id,operation:j,variant:`sort-key-${t.variant}`,bindings:n,invocationCount:t.shape.hitCapacity,declarations:M(t.shape),body:`let activeCount = min(state[stateOffset], HIT_CAPACITY);
  ${r}`})}function ie(e,t){let{bits:n}=t;return _(e,{id:t.id,operation:j,variant:`sort-key-composite`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`hitPairs`,view:t.hitPairs,type:`u32`,access:`read`},{name:`hitBuckets`,view:t.hitBuckets,type:`u32`,access:`read`},{name:`identity`,view:t.identity,type:`u32`,access:`read_write`},{name:`keys`,view:t.keys,type:`u32`,access:`read_write`}],invocationCount:t.shape.hitCapacity,declarations:`${M(t.shape)}
const PARTNER_SHIFT: u32 = ${n.bucket}u;
const TRACK_SHIFT: u32 = ${n.bucket+n.partner}u;`,body:`let activeCount = min(state[stateOffset], HIT_CAPACITY);
  identity[identityOffset + index] = index;
  var key = TRACK_COUNT << TRACK_SHIFT;
  if (index < activeCount) {
    key = (hitPairs[hitPairsOffset + index * 2u] << TRACK_SHIFT)
      | (hitPairs[hitPairsOffset + index * 2u + 1u] << PARTNER_SHIFT)
      | hitBuckets[hitBucketsOffset + index];
  }
  keys[keysOffset + index] = key;`})}function ae(e,t){if(t.composite){let{sortedKeys:n,sortedTracks:r,bits:i}=t.composite;return _(e,{id:t.id,operation:j,variant:`sorted-hits-composite`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`order`,view:t.order,type:`u32`,access:`read`},{name:`sortedKeys`,view:n,type:`u32`,access:`read`},{name:`hitDistances`,view:t.hitDistances,type:`f32`,access:`read`},{name:`sortedTracks`,view:r,type:`u32`,access:`read_write`},{name:`sortedPartners`,view:t.sortedPartners,type:`u32`,access:`read_write`},{name:`sortedBuckets`,view:t.sortedBuckets,type:`u32`,access:`read_write`},{name:`sortedDistances`,view:t.sortedDistances,type:`f32`,access:`read_write`}],invocationCount:t.shape.hitCapacity,declarations:`${M(t.shape)}
const BUCKET_MASK: u32 = ${2**i.bucket-1>>>0}u;
const PARTNER_MASK: u32 = ${2**i.partner-1>>>0}u;
const PARTNER_SHIFT: u32 = ${i.bucket}u;
const TRACK_SHIFT: u32 = ${i.bucket+i.partner}u;`,body:`if (index >= min(state[stateOffset], HIT_CAPACITY)) {
    sortedTracks[sortedTracksOffset + index] = 0xffffffffu;
    sortedPartners[sortedPartnersOffset + index] = 0xffffffffu;
    sortedBuckets[sortedBucketsOffset + index] = 0xffffffffu;
    sortedDistances[sortedDistancesOffset + index] = 0.0;
    return;
  }
  let key = sortedKeys[sortedKeysOffset + index];
  sortedTracks[sortedTracksOffset + index] = key >> TRACK_SHIFT;
  sortedPartners[sortedPartnersOffset + index] = (key >> PARTNER_SHIFT) & PARTNER_MASK;
  sortedBuckets[sortedBucketsOffset + index] = key & BUCKET_MASK;
  sortedDistances[sortedDistancesOffset + index] = hitDistances[hitDistancesOffset + order[orderOffset + index]];`})}let n=t.hitPairs,r=t.hitBuckets;return _(e,{id:t.id,operation:j,variant:`sorted-hits`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`order`,view:t.order,type:`u32`,access:`read`},{name:`hitPairs`,view:n,type:`u32`,access:`read`},{name:`hitBuckets`,view:r,type:`u32`,access:`read`},{name:`hitDistances`,view:t.hitDistances,type:`f32`,access:`read`},{name:`sortedPartners`,view:t.sortedPartners,type:`u32`,access:`read_write`},{name:`sortedBuckets`,view:t.sortedBuckets,type:`u32`,access:`read_write`},{name:`sortedDistances`,view:t.sortedDistances,type:`f32`,access:`read_write`}],invocationCount:t.shape.hitCapacity,declarations:M(t.shape),body:`if (index >= min(state[stateOffset], HIT_CAPACITY)) {
    sortedPartners[sortedPartnersOffset + index] = 0xffffffffu;
    sortedBuckets[sortedBucketsOffset + index] = 0xffffffffu;
    sortedDistances[sortedDistancesOffset + index] = 0.0;
    return;
  }
  let slot = order[orderOffset + index];
  sortedPartners[sortedPartnersOffset + index] = hitPairs[hitPairsOffset + slot * 2u + 1u];
  sortedBuckets[sortedBucketsOffset + index] = hitBuckets[hitBucketsOffset + slot];
  sortedDistances[sortedDistancesOffset + index] = hitDistances[hitDistancesOffset + slot];`})}function F(e,t){return _(e,{id:t.id,operation:j,variant:`run-flags`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`sortedTracks`,view:t.sortedTracks,type:`u32`,access:`read`},{name:`sortedPartners`,view:t.sortedPartners,type:`u32`,access:`read`},{name:`flags`,view:t.flags,type:`u32`,access:`read_write`},{name:`rowIds`,view:t.rowIds,type:`u32`,access:`read_write`}],invocationCount:t.shape.hitCapacity,declarations:M(t.shape),body:`rowIds[rowIdsOffset + index] = index;
  var isStart = index < min(state[stateOffset], HIT_CAPACITY);
  if (isStart && index > 0u) {
    isStart = sortedTracks[sortedTracksOffset + index] != sortedTracks[sortedTracksOffset + index - 1u]
      || sortedPartners[sortedPartnersOffset + index] != sortedPartners[sortedPartnersOffset + index - 1u];
  }
  flags[flagsOffset + index] = select(0u, 1u, isStart);`})}function I(e,t){let n=[],r=M(t.shape),i=`let isLive = index < min(runCount[runCountOffset], PAIR_CAPACITY);
  let start = select(0u, runStarts[runStartsOffset + index], isLive);`;if(n.push(_(e,{id:`${t.id}-identity`,operation:j,variant:`reduce-identity`,bindings:[{name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`},{name:`runCount`,view:t.runCount,type:`u32`,access:`read`},{name:`sortedTracks`,view:t.sortedTracks,type:`u32`,access:`read`},{name:`sortedPartners`,view:t.sortedPartners,type:`u32`,access:`read`},{name:`ids`,view:t.ids,type:`u32`,access:`read_write`},...t.partners?[{name:`partners`,view:t.partners,type:`u32`,access:`read_write`}]:[]],invocationCount:t.shape.pairCapacity,declarations:r,body:`${i}
  ids[idsOffset + index] = select(0xffffffffu, sortedTracks[sortedTracksOffset + start], isLive);
  ${t.partners?`partners[partnersOffset + index] = select(0xffffffffu, sortedPartners[sortedPartnersOffset + start], isLive);`:``}`})),t.firstBuckets||t.minimumDistances||t.bucketCounts){let a=[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`},{name:`runCount`,view:t.runCount,type:`u32`,access:`read`},{name:`sortedBuckets`,view:t.sortedBuckets,type:`u32`,access:`read`},{name:`sortedDistances`,view:t.sortedDistances,type:`f32`,access:`read`}];for(let[e,n]of[[`firstBuckets`,t.firstBuckets],[`minimumDistances`,t.minimumDistances],[`bucketCounts`,t.bucketCounts]])n&&a.push({name:e,view:n,type:e===`minimumDistances`?`f32`:`u32`,access:`read_write`});n.push(_(e,{id:`${t.id}-statistics`,operation:j,variant:`reduce-statistics`,bindings:a,invocationCount:t.shape.pairCapacity,declarations:r,body:`${i}
  let hitTotal = min(state[stateOffset], HIT_CAPACITY);
  var end = start;
  if (isLive) {
    end = hitTotal;
    // Run boundaries belong to the complete hit scratch, even when output pairs are truncated.
    if (index + 1u < runCount[runCountOffset]) { end = runStarts[runStartsOffset + index + 1u]; }
  }
  // Hits of one pair are in bucket order, so the run start is the first encounter.
  var minimumDistance = sortedDistances[sortedDistancesOffset + start];
  for (var hit = start + 1u; hit < end; hit++) {
    minimumDistance = min(minimumDistance, sortedDistances[sortedDistancesOffset + hit]);
  }
  ${t.firstBuckets?`firstBuckets[firstBucketsOffset + index] = select(0xffffffffu, sortedBuckets[sortedBucketsOffset + start], isLive);`:``}
  ${t.minimumDistances?`minimumDistances[minimumDistancesOffset + index] = select(0.0, minimumDistance, isLive);`:``}
  ${t.bucketCounts?`bucketCounts[bucketCountsOffset + index] = select(0u, end - start, isLive);`:``}`}))}return t.firstTimes&&t.bucketTimes&&t.firstBuckets&&n.push(_(e,{id:`${t.id}-first-times`,operation:j,variant:`first-times`,bindings:[{name:`firstBuckets`,view:t.firstBuckets,type:`u32`,access:`read`},{name:`bucketTimes`,view:t.bucketTimes,type:`f32`,access:`read`},{name:`firstTimes`,view:t.firstTimes,type:`f32`,access:`read_write`}],invocationCount:t.shape.pairCapacity,declarations:r,body:`let bucket = firstBuckets[firstBucketsOffset + index];
  firstTimes[firstTimesOffset + index] = select(0.0, bucketTimes[bucketTimesOffset + min(bucket, BUCKET_COUNT - 1u)], bucket < BUCKET_COUNT);`})),n}function L(e,t){return _(e,{id:t.id,operation:j,variant:`overflow`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`flag`,view:t.flag,type:`u32`,access:`read_write`}],invocationCount:1,declarations:M(t.shape),body:`flag[flagOffset] = select(0u, 1u, state[stateOffset] > HIT_CAPACITY);`})}function R(e,t){return _(e,{id:t.id,operation:j,variant:t.isWordMode?`clock-bucket-times-words`:`clock-bucket-times`,bindings:[{name:`clockParameters`,view:t.clock,type:t.isWordMode?`u32`:`f32`,access:`read`},{name:`bucketTimes`,view:t.bucketTimes,type:`f32`,access:`read_write`}],invocationCount:t.bucketCount,body:`let step = ${t.isWordMode?`bitcast<f32>(clockParameters[clockParametersOffset + 3u])`:`clockParameters[clockParametersOffset + 1u]`};
  bucketTimes[bucketTimesOffset + index] = f32(index) * step;`})}var z=Object.freeze({...s,measure:`distance`}),B=`GPUTrajectoryEncounters`,V=class{id;props;columns;rows;spatialContext;maximumDistance;geodesic;constructor(e){this.id=e.id??`trajectory-encounters`;let r=e;this.props=r;let{id:i}=this;this.spatialContext=l(i,r.spatialContext,z);let s=c(this.spatialContext);b(i,`trackCount`,r.trackCount),b(i,`bucketCount`,r.bucketCount),b(i,`hitCapacity`,r.hitCapacity,1,t);let d=r.trackCount*r.bucketCount;if(d>=2147483647)throw Error(`${i} trackCount * bucketCount must be below 2^31`);if(v(r.samples,[`float32x2`],`${i} samples`),r.samples.length!==d)throw Error(`${i} samples length must equal trackCount * bucketCount`);if(r.mask&&(v(r.mask,[`uint32`],`${i} mask`),r.mask.length!==r.trackCount))throw Error(`${i} mask length must equal trackCount`);if(v(r.distance,[`float32`],`${i} distance`),r.distance.length<1)throw Error(`${i} distance must contain one float32 row`);if(!(Number.isFinite(Math.fround(r.cellSize))&&Math.fround(r.cellSize)>0))throw Error(`${i} cellSize must be a positive finite number`);let[f,p,m,h]=r.bounds;if(S(i,r.bounds,`bounds`),r.bounds.forEach((e,t)=>x(i,`bounds[${t}]`,e)),!(Math.fround(f)<Math.fround(m))||!(Math.fround(p)<Math.fround(h)))throw Error(`${i} bounds collapse in float32`);if(s===void 0)this.maximumDistance=r.cellSize;else{let e=Math.max(Math.abs(p),Math.abs(h));if(f<-180||m>180||e>=90)throw Error(`${i} great-circle bounds need longitudes within [-180, 180] and latitudes within (-90, 90)`);let t=f===-180&&m===180;if(r.cellSize>=(t?120:90))throw Error(`${i} great-circle cellSize must be below ${t?120:90} degrees`);let n=o(r.cellSize),a=o(e),c=Math.asin(Math.sin(n)*Math.cos(a));this.maximumDistance=s*c*(1-1e-6),this.geodesic={radius:s,maximumDistance:this.maximumDistance,wrapColumns:t}}u(`GPUTrajectoryEncounters`,this.spatialContext,this.maximumDistance),this.columns=Math.max(1,Math.floor((Math.fround(m)-Math.fround(f))/r.cellSize)),this.rows=Math.max(1,Math.floor((Math.fround(h)-Math.fround(p))/r.cellSize));let{pairs:g,status:_}=r.output;if(C(i,g),v(_.candidateOverflow,[`uint32`],`${i} status.candidateOverflow`),_.candidateOverflow.length<1)throw Error(`${i} status.candidateOverflow must hold one row`);let y=g.ids.length;for(let[e,t,n]of[[`partners`,g.partners,`uint32`],[`firstBuckets`,g.firstBuckets,`uint32`],[`minimumDistances`,g.minimumDistances,`float32`],[`bucketCounts`,g.bucketCounts,`uint32`],[`firstTimes`,g.firstTimes,`float32`]])if(t&&(v(t,[n],`${i} pairs.${e}`),t.length!==y))throw Error(`${i} pairs.${e} length must equal the capacity`);if(g.firstTimes&&!(r.bucketTimes&&g.firstBuckets))throw Error(`${i} pairs.firstTimes requires bucketTimes and pairs.firstBuckets`);if(r.bucketTimes&&(v(r.bucketTimes,[`float32`],`${i} bucketTimes`),r.bucketTimes.length!==r.bucketCount))throw Error(`${i} bucketTimes length must equal bucketCount`);a(this.id,this.props.output),n(i,[{ids:g.ids,partners:g.partners},{firstBuckets:g.firstBuckets,minimumDistances:g.minimumDistances,bucketCounts:g.bucketCounts},{...g.status}]),T(i,[g.firstTimes],[g.firstBuckets]),T(i,[g.ids,g.status.count,g.status.overflow,g.status.candidateOverflow,g.status.requiredCount,_.candidateOverflow,g.partners,g.firstBuckets,g.minimumDistances,g.bucketCounts,g.firstTimes],[r.samples,r.mask,r.distance,r.bucketTimes])}getCommandNodes(e){i(e,this.id,this.props);let{id:t,props:n,columns:a,rows:o}=this,{pairs:s,status:c}=n.output;w(t,e,[n.samples,n.mask,n.distance,n.bucketTimes,s.ids,s.status.count,s.status.overflow,s.status.candidateOverflow,s.status.requiredCount,c.candidateOverflow,s.partners,s.firstBuckets,s.minimumDistances,s.bucketCounts,s.firstTimes]),y(t,`columns x rows x bucketCount`,a*o*n.bucketCount,ee(e.device));let l=n.bounds.map(Math.fround),u={trackCount:n.trackCount,bucketCount:n.bucketCount,hitCapacity:n.hitCapacity,pairCapacity:s.ids.length,columns:a,rows:o,bounds:l,cellSize:n.cellSize,geodesic:this.geodesic},h=n.trackCount*n.bucketCount,v=n.hitCapacity;r(e,t,{"sample positions":h*12,"hit pairs":v*8,"cell offsets":(a*o*n.bucketCount+1)*4,"sort histograms":16*Math.ceil(Math.max(h,v)/256)*4});let b=(n,r)=>d(e,`${t}-${n}`,`uint32`,Math.max(r,1)),x=(n,r)=>d(e,`${t}-${n}`,`float32`,Math.max(r,1)),S=[],C=d(e,`${t}-positions`,`float32x3`,h);te(e.device,t,`cell-offsets`,(a*o*n.bucketCount+1)*4,`raise cellSize or lower bucketCount`);let T=b(`cell-offsets`,a*o*n.bucketCount+1),E=b(`object-ids`,h),D=b(`grid-count`,1),O=b(`grid-overflow`,1),k=b(`state`,1),j=d(e,`${t}-hit-pairs`,`uint32x2`,v),M=b(`hit-buckets`,v),R=x(`hit-distances`,v);S.push(ne(e,{id:`${t}-positions`,shape:u,samples:n.samples,trackValid:n.mask,positions:C}),...new p({id:`${t}-grid`,positions:C,gridSize:[a,o,n.bucketCount],bounds:[l[0],l[1],0,l[2],l[3],n.bucketCount],cellOffsets:T,objectIds:E,count:D,overflow:O}).getCommandNodes(e),re(e,{id:`${t}-clear`,state:k}),N(e,{id:`${t}-scan`,shape:u,positions:C,cellOffsets:T,objectIds:E,distance:n.distance,state:k,hitPairs:j,hitBuckets:M,hitDistances:R}));let z={track:A(n.trackCount),partner:A(n.trackCount-1),bucket:A(n.bucketCount-1)},V=z.track+z.partner+z.bucket<=32,H=b(`identity`,v),U=b(`sorted-tracks`,v),W=b(`order`,v),G;if(V){let n=b(`composite-keys`,v);G=b(`composite-sorted-keys`,v),S.push(ie(e,{id:`${t}-composite-keys`,shape:u,bits:z,state:k,hitPairs:j,hitBuckets:M,keys:n,identity:H}),...new m({id:`${t}-composite-sort`,keys:n,values:H,outputKeys:G,outputValues:W,keyBits:z.track+z.partner+z.bucket}).getCommandNodes(e))}else{let n=b(`bucket-keys`,v),r=b(`bucket-sorted-keys`,v),i=b(`bucket-order`,v),a=b(`partner-keys`,v),o=b(`partner-sorted-keys`,v),s=b(`partner-order`,v),c=b(`track-keys`,v);S.push(P(e,{id:`${t}-bucket-keys`,shape:u,variant:`bucket`,state:k,hitPairs:j,hitBuckets:M,keys:n,identity:H}),...new m({id:`${t}-bucket-sort`,keys:n,values:H,outputKeys:r,outputValues:i,keyBits:z.bucket}).getCommandNodes(e),P(e,{id:`${t}-partner-keys`,shape:u,variant:`partner`,state:k,order:i,hitPairs:j,keys:a}),...new m({id:`${t}-partner-sort`,keys:a,values:i,outputKeys:o,outputValues:s,keyBits:z.partner}).getCommandNodes(e),P(e,{id:`${t}-track-keys`,shape:u,variant:`track`,state:k,order:s,hitPairs:j,keys:c}),...new m({id:`${t}-track-sort`,keys:c,values:s,outputKeys:U,outputValues:W,keyBits:z.track}).getCommandNodes(e))}let K=b(`sorted-partners`,v),q=b(`sorted-buckets`,v),J=x(`sorted-distances`,v),Y=b(`run-flags`,v),X=b(`row-ids`,v),Z=b(`run-starts`,v),Q=b(`run-count`,1),$=b(`overflow-flag`,1);return S.push(ae(e,{id:`${t}-sorted-hits`,shape:u,state:k,order:W,hitPairs:V?void 0:j,hitBuckets:V?void 0:M,hitDistances:R,sortedPartners:K,sortedBuckets:q,sortedDistances:J,composite:G?{sortedKeys:G,sortedTracks:U,bits:z}:void 0}),F(e,{id:`${t}-run-flags`,shape:u,state:k,sortedTracks:U,sortedPartners:K,flags:Y,rowIds:X}),...new f({id:`${t}-runs`,input:X,flags:Y,output:Z,count:Q}).getCommandNodes(e),...I(e,{id:`${t}-reduce`,shape:u,state:k,runStarts:Z,runCount:Q,sortedTracks:U,sortedPartners:K,sortedBuckets:q,sortedDistances:J,ids:s.ids,partners:s.partners,firstBuckets:s.firstBuckets,minimumDistances:s.minimumDistances,bucketCounts:s.bucketCounts,bucketTimes:n.bucketTimes,firstTimes:s.firstTimes}),L(e,{id:`${t}-overflow`,shape:u,state:k,flag:$}),_(e,{id:`${t}-candidate-overflow`,operation:B,variant:`candidate-overflow`,bindings:[{name:`hitOverflow`,view:$,type:`u32`,access:`read`},{name:`gridOverflow`,view:O,type:`u32`,access:`read`},{name:`candidateOverflow`,view:c.candidateOverflow,type:`u32`,access:`read_write`}],invocationCount:1,body:`candidateOverflow[candidateOverflowOffset] =
  select(0u, 1u, hitOverflow[hitOverflowOffset] != 0u || gridOverflow[gridOverflowOffset] != 0u);`}),g(e,{id:`${t}-publish`,operation:B,requiredCount:Q,output:s,candidateOverflowSources:s.status.candidateOverflow?[c.candidateOverflow]:[]})),S}};export{R as n,V as t};
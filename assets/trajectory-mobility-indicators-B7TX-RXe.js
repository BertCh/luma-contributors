import{n as e,t}from"./gpu-trajectory-locations-kz1jcoC4.js";import{c as n,d as r,f as i,l as a,n as o,r as s,s as c,t as l,u}from"./movement-DZVB3aIP.js";import{t as d}from"./construction-snapshot-DdkX9Pc0.js";import{r as f}from"./wgsl-common-DfyXiYB0.js";import{a as p,i as m,n as h,r as g}from"./trajectory-binding-validation-C89oJIAZ.js";import{r as _}from"./kernel-C3tfQOn4.js";import{a as v,m as y,v as b}from"./spatial-context-DwZEd4EZ.js";import{D as x,E as S,S as C,T as w,f as T,v as E,x as D,y as O}from"./gpu-contributor-utils-BIwLQON5.js";import{c as k,o as A,t as j}from"./wgsl-kernel-nodes-DagJklH2.js";import{d as M,f as N,u as P}from"./trajectory-metrics-kernels-Dj0TkZQS.js";import{t as F}from"./geodesic-wgsl-I0CYDKcz.js";import{a as I,t as L}from"./movement-common-nOvANN43.js";import{t as R}from"./define-run-B-brkCe4.js";import{t as z}from"./units-Boef4ppn.js";import{m as B}from"./entry-preview-D70TdSbK.js";var V=`GPUMobilityIndicators`,H=Object.freeze({...v,measure:`distance`}),U=16384,W=class{id;props;spatialContext;maximumEntropySequenceRows;constructor(e){e=d(e,[`id`,`spatialContext`,`trackCount`,`timestamps`,`staypoints`,`triplegs`,`maximumEntropySequenceRows`,`output`]);let t=d(e.staypoints,[`tracks`,`centroids`,`startRows`,`durations`,`locationIds`,`status`]),n=e.triplegs&&d(e.triplegs,[`tracks`,`startRows`,`endRows`,`status`,`staypointFinishRows`]),r=d(e.output,[`radiusOfGyration`,`jumpLengths`,`waitingTimes`,`randomEntropy`,`uncorrelatedEntropy`,`realEntropy`,`homeLocations`,`workLocations`,`trackingQuality`,`status`]);e={...e,staypoints:{...t,status:d(t.status,[`count`,`overflow`,`candidateOverflow`])},triplegs:n&&{...n,status:d(n.status,[`count`,`overflow`,`candidateOverflow`])},output:{...r,status:r.status&&d(r.status,[`candidateOverflow`])}},this.id=e.id??`mobility-indicators`,this.props=e;let i=this.id;this.spatialContext=b(i,e.spatialContext,H);let{staypoints:a,triplegs:o,output:s,timestamps:c}=e;for(let[e,t]of Object.entries(a.status))t&&(S(t,[`uint32`],`${i} staypoints.status.${e}`),w(i,t.length>=1,`staypoints.status.${e} must contain at least one row`));for(let[e,t]of Object.entries(o?.status??{}))t&&(S(t,[`uint32`],`${i} triplegs.status.${e}`),w(i,t.length>=1,`triplegs.status.${e} must contain at least one row`));T(i,`trackCount`,e.trackCount),S(c,[`float32`,`uint32x2`],`${i} timestamps`),S(a.tracks,[`uint32`],`${i} staypoints.tracks`),S(a.centroids,[`float32x2`],`${i} staypoints.centroids`),S(a.startRows,[`uint32`],`${i} staypoints.startRows`),S(a.durations,[`float32`],`${i} staypoints.durations`),S(a.locationIds,[`uint32`],`${i} staypoints.locationIds`);let l=a.tracks.length;for(let e of[a.centroids,a.startRows,a.durations,a.locationIds])x(i,`staypoint column`,e,l);for(let[t,n,r]of[[`radiusOfGyration`,s.radiusOfGyration,`float32`],[`randomEntropy`,s.randomEntropy,`float32`],[`uncorrelatedEntropy`,s.uncorrelatedEntropy,`float32`],[`realEntropy`,s.realEntropy,`float32`],[`homeLocations`,s.homeLocations,`uint32`],[`workLocations`,s.workLocations,`uint32`],[`trackingQuality`,s.trackingQuality,`float32`]])n&&(S(n,[r],`${i} output.${t}`),x(i,`output.${t}`,n,e.trackCount));for(let[e,t]of[[`jumpLengths`,s.jumpLengths],[`waitingTimes`,s.waitingTimes]])t&&(S(t,[`float32`],`${i} output.${e}`),x(i,`output.${e}`,t,l));w(i,!!(s.radiusOfGyration||s.jumpLengths||s.waitingTimes||s.randomEntropy||s.uncorrelatedEntropy||s.realEntropy||s.homeLocations||s.workLocations||s.trackingQuality),`requires at least one output`),w(i,!(s.trackingQuality&&!o),`trackingQuality requires triplegs`),s.status?.candidateOverflow&&(S(s.status.candidateOverflow,[`uint32`],`${i} output.status.candidateOverflow`),x(i,`output.status.candidateOverflow`,s.status.candidateOverflow,1)),this.maximumEntropySequenceRows=e.maximumEntropySequenceRows??1024,C(i,`maximumEntropySequenceRows`,this.maximumEntropySequenceRows,U),g(this.id,this.props.output),p(this.id,[{randomEntropy:s.randomEntropy,uncorrelatedEntropy:s.uncorrelatedEntropy,homeLocations:s.homeLocations,workLocations:s.workLocations},{realEntropy:s.realEntropy,candidateOverflow:s.status?.candidateOverflow},{jumpLengths:s.jumpLengths,waitingTimes:s.waitingTimes}]),O(i,this.getOutputViews(),this.getInputViews())}getInputViews(){let{staypoints:e,triplegs:t,timestamps:n}=this.props;return[n,e.tracks,e.centroids,e.startRows,e.durations,e.locationIds,e.status.count,e.status.overflow,e.status.candidateOverflow,t?.tracks,t?.startRows,t?.endRows,t?.status.count,t?.status.overflow,t?.status.candidateOverflow,t?.staypointFinishRows]}getOutputViews(){let{output:e}=this.props;return[e.radiusOfGyration,e.jumpLengths,e.waitingTimes,e.randomEntropy,e.uncorrelatedEntropy,e.realEntropy,e.homeLocations,e.workLocations,e.trackingQuality,e.status?.candidateOverflow]}getCommandNodes(e){let{id:t,props:n}=this,{staypoints:r,triplegs:i,output:a,timestamps:o,trackCount:s}=n;D(t,e,[...this.getInputViews(),...this.getOutputViews()]);let c=r.tracks.length;E(e.device,t,`centroids`,c*8,`use fewer staypoints per graph`),h(e,this.id,this.props),m(e,t,{"track offsets":(s+1)*4});let l=y(this.spatialContext),u=M({timestamps:o}),d=_(e,`${t}-track-offsets`,`uint32`,s+1),p=[I(e,{id:`${t}-offsets`,operation:V,tracks:r.tracks,count:r.status.count,offsets:d,capacity:c,trackCount:s})],g=`${l===void 0?``:F}
fn stayPoint(row: u32) -> vec2<f32> {
  return vec2<f32>(centroids[centroidsOffset + 2u * row], centroids[centroidsOffset + 2u * row + 1u]);
}
fn pointDistance(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return ${l===void 0?`length(a - b)`:`geodesicHaversineDistance(a, b, ${k(l)})`};
}`;if(a.radiusOfGyration&&p.push(A(e,{id:`${t}-radius-of-gyration`,operation:V,invocationCount:s,bindings:[{name:`centroids`,view:r.centroids,type:`f32`,access:`read`},{name:`trackOffsets`,view:d,type:`u32`,access:`read`},{name:`radiusOfGyration`,view:a.radiusOfGyration,type:`f32`,access:`read_write`}],declarations:g,body:`let begin = trackOffsets[trackOffsetsOffset + index];
  let end = trackOffsets[trackOffsetsOffset + index + 1u];
  var result = 0.0;
  if (end > begin) {
    let first = stayPoint(begin);
    var sum = vec2<f32>(0.0, 0.0);
    for (var row = begin; row < end; row++) { sum += stayPoint(row) - first; }
    let center = first + sum / f32(end - begin);
    var squares = 0.0;
    for (var row = begin; row < end; row++) {
      let separation = pointDistance(stayPoint(row), center);
      squares += separation * separation;
    }
    result = sqrt(squares / f32(end - begin));
  }
  radiusOfGyration[radiusOfGyrationOffset + index] = result;`})),(a.jumpLengths||a.waitingTimes)&&p.push(A(e,{id:`${t}-jumps`,operation:V,variant:u,invocationCount:c,bindings:[{name:`centroids`,view:r.centroids,type:`f32`,access:`read`},{name:`tracks`,view:r.tracks,type:`u32`,access:`read`},{name:`startRows`,view:r.startRows,type:`u32`,access:`read`},{name:`stayCount`,view:r.status.count,type:`u32`,access:`read`},...P({timestamps:o}),...a.jumpLengths?[{name:`jumpLengths`,view:a.jumpLengths,type:`f32`,access:`read_write`}]:[],...a.waitingTimes?[{name:`waitingTimes`,view:a.waitingTimes,type:`f32`,access:`read_write`}]:[]],declarations:`${g}\n${N(u)}`,body:`let isActive = index < min(stayCount[stayCountOffset], ${c}u);
  let hasPrevious = isActive && index > 0u && tracks[tracksOffset + index] == tracks[tracksOffset + index - 1u];
  ${a.jumpLengths?`jumpLengths[jumpLengthsOffset + index] = select(0.0, pointDistance(stayPoint(index - 1u), stayPoint(index)), hasPrevious);`:``}
  ${a.waitingTimes?`waitingTimes[waitingTimesOffset + index] = select(0.0, timeDifferenceRows(startRows[startRowsOffset + index], startRows[startRowsOffset + max(index, 1u) - 1u]), hasPrevious);`:``}`})),(a.randomEntropy||a.uncorrelatedEntropy||a.homeLocations||a.workLocations)&&p.push(A(e,{id:`${t}-locations`,operation:V,invocationCount:s,bindings:[{name:`locationIds`,view:r.locationIds,type:`u32`,access:`read`},{name:`durations`,view:r.durations,type:`f32`,access:`read`},{name:`trackOffsets`,view:d,type:`u32`,access:`read`},...a.randomEntropy?[{name:`randomEntropy`,view:a.randomEntropy,type:`f32`,access:`read_write`}]:[],...a.uncorrelatedEntropy?[{name:`uncorrelatedEntropy`,view:a.uncorrelatedEntropy,type:`f32`,access:`read_write`}]:[],...a.homeLocations?[{name:`homeLocations`,view:a.homeLocations,type:`u32`,access:`read_write`}]:[],...a.workLocations?[{name:`workLocations`,view:a.workLocations,type:`u32`,access:`read_write`}]:[]],body:`let begin = trackOffsets[trackOffsetsOffset + index];
  let end = trackOffsets[trackOffsetsOffset + index + 1u];
  let total = f32(end - begin);
  var distinctCount = 0u;
  var entropy = 0.0;
  var bestDuration = -1.0;
  var bestLocation = ${L};
  var secondDuration = -1.0;
  var secondLocation = ${L};
  for (var row = begin; row < end; row++) {
    let locationId = locationIds[locationIdsOffset + row];
    var isFirst = true;
    for (var earlier = begin; earlier < row; earlier++) {
      if (locationIds[locationIdsOffset + earlier] == locationId) { isFirst = false; break; }
    }
    if (!isFirst) { continue; }
    distinctCount++;
    var count = 0u;
    var summed = 0.0;
    for (var other = row; other < end; other++) {
      if (locationIds[locationIdsOffset + other] == locationId) {
        count++;
        summed += durations[durationsOffset + other];
      }
    }
    let share = f32(count) / total;
    entropy -= share * log2(share);
    if (summed > bestDuration || (summed == bestDuration && locationId < bestLocation)) {
      secondDuration = bestDuration;
      secondLocation = bestLocation;
      bestDuration = summed;
      bestLocation = locationId;
    } else if (summed > secondDuration || (summed == secondDuration && locationId < secondLocation)) {
      secondDuration = summed;
      secondLocation = locationId;
    }
  }
  ${a.randomEntropy?`randomEntropy[randomEntropyOffset + index] = select(0.0, log2(f32(distinctCount)), distinctCount > 0u);`:``}
  ${a.uncorrelatedEntropy?`uncorrelatedEntropy[uncorrelatedEntropyOffset + index] = entropy;`:``}
  ${a.homeLocations?`homeLocations[homeLocationsOffset + index] = bestLocation;`:``}
  ${a.workLocations?`workLocations[workLocationsOffset + index] = secondLocation;`:``}`})),a.realEntropy&&(a.status?.candidateOverflow&&p.push(j(e,{id:`${t}-entropy-overflow-reset`,operation:V,view:a.status.candidateOverflow,type:`u32`,value:`0u`,componentCount:1})),p.push(A(e,{id:`${t}-real-entropy`,operation:V,invocationCount:s,bindings:[{name:`locationIds`,view:r.locationIds,type:`u32`,access:`read`},{name:`trackOffsets`,view:d,type:`u32`,access:`read`},{name:`realEntropy`,view:a.realEntropy,type:`f32`,access:`read_write`},...a.status?.candidateOverflow?[{name:`candidateOverflow`,view:a.status.candidateOverflow,type:`atomic<u32>`,access:`read_write`}]:[]],body:`let begin = trackOffsets[trackOffsetsOffset + index];
  let end = trackOffsets[trackOffsetsOffset + index + 1u];
  let sequenceLength = end - begin;
  var result = 0.0;
  if (sequenceLength > ${this.maximumEntropySequenceRows}u) {
    ${a.status?.candidateOverflow?`atomicStore(&candidateOverflow[candidateOverflowOffset], 1u);`:``}
  } else if (sequenceLength >= 2u) {
    // Scikit-mobility true_entropy: first and last elements contribute 1 and 2.
    var sumLambda = 3.0;
    for (var position = 1u; position + 1u < sequenceLength; position++) {
      var longestMatch = 0u;
      for (var candidate = 0u; candidate < position; candidate++) {
        var matchRows = 0u;
        while (candidate + matchRows < position && position + matchRows < sequenceLength &&
          locationIds[locationIdsOffset + begin + candidate + matchRows] == locationIds[locationIdsOffset + begin + position + matchRows]) {
          matchRows++;
        }
        longestMatch = max(longestMatch, matchRows);
      }
      // scikit-mobility: the first unseen length, plus one when the match reaches the sequence end.
      var symbolLength = longestMatch + 1u;
      if (longestMatch + 1u >= sequenceLength - position) { symbolLength = sequenceLength - position + 1u; }
      sumLambda += f32(symbolLength);
    }
    let n = f32(sequenceLength);
    result = n * log2(n) / sumLambda;
  }
  realEntropy[realEntropyOffset + index] = result;`}))),a.trackingQuality&&i){let n=[i.status.overflow,i.status.candidateOverflow].flatMap((e,t)=>e?[{name:`triplegLoss${t}`,view:e,type:`u32`,access:`read`}]:[]),c=n.map(e=>`${e.name}[${e.name}Offset] != 0u`).join(` || `)||`false`,l=i.tracks.length,m=_(e,`${t}-tripleg-offsets`,`uint32`,s+1);p.push(I(e,{id:`${t}-tripleg-offset-table`,operation:V,tracks:i.tracks,count:i.status.count,offsets:m,capacity:l,trackCount:s}),A(e,{id:`${t}-tracking-quality`,operation:V,variant:u,invocationCount:s,bindings:[...P({timestamps:o}),{name:`stayStarts`,view:r.startRows,type:`u32`,access:`read`},{name:`stayFinishes`,view:i.staypointFinishRows,type:`u32`,access:`read`},{name:`stayOffsets`,view:d,type:`u32`,access:`read`},{name:`triplegStarts`,view:i.startRows,type:`u32`,access:`read`},{name:`triplegEnds`,view:i.endRows,type:`u32`,access:`read`},{name:`triplegOffsets`,view:m,type:`u32`,access:`read`},{name:`trackingQuality`,view:a.trackingQuality,type:`f32`,access:`read_write`}],declarations:N(u),body:`var tracked = 0.0;
  var firstStart = ${L};
  var lastFinish = ${L};
  for (var row = stayOffsets[stayOffsetsOffset + index]; row < stayOffsets[stayOffsetsOffset + index + 1u]; row++) {
    let start = stayStarts[stayStartsOffset + row];
    let finish = stayFinishes[stayFinishesOffset + row];
    if (timeDifferenceRows(finish, start) <= 0.0) { continue; }
    tracked += timeDifferenceRows(finish, start);
    if (firstStart == ${L} || timeDifferenceRows(start, firstStart) < 0.0) { firstStart = start; }
    if (lastFinish == ${L} || timeDifferenceRows(finish, lastFinish) > 0.0) { lastFinish = finish; }
  }
  for (var row = triplegOffsets[triplegOffsetsOffset + index]; row < triplegOffsets[triplegOffsetsOffset + index + 1u]; row++) {
    let start = triplegStarts[triplegStartsOffset + row];
    let finish = triplegEnds[triplegEndsOffset + row] - 1u;
    if (timeDifferenceRows(finish, start) <= 0.0) { continue; }
    tracked += timeDifferenceRows(finish, start);
    if (firstStart == ${L} || timeDifferenceRows(start, firstStart) < 0.0) { firstStart = start; }
    if (lastFinish == ${L} || timeDifferenceRows(finish, lastFinish) > 0.0) { lastFinish = finish; }
  }
  var quality = 0.0;
  if (firstStart != ${L} && lastFinish != ${L}) {
    let extent = timeDifferenceRows(lastFinish, firstStart);
    if (extent > 0.0) { quality = tracked / extent; }
  }
  trackingQuality[trackingQualityOffset + index] = quality;`})),n.length&&p.push(A(e,{id:`${t}-suppress-incomplete-quality`,operation:V,invocationCount:s,bindings:[...n,{name:`trackingQuality`,view:a.trackingQuality,type:`f32`,access:`read_write`}],declarations:f,body:`if (${c}) { trackingQuality[trackingQualityOffset + index] = getNaN(); }`}))}let v=[r.status.overflow,r.status.candidateOverflow].flatMap((e,t)=>e?[{name:`stayLoss${t}`,view:e,type:`u32`,access:`read`}]:[]),b=v.map(e=>`${e.name}[${e.name}Offset] != 0u`).join(` || `)||`false`;for(let n of[`radiusOfGyration`,`jumpLengths`,`waitingTimes`,`randomEntropy`,`uncorrelatedEntropy`,`realEntropy`,`homeLocations`,`workLocations`,`trackingQuality`]){let r=a[n];if(!r||!v.length)continue;let i=n===`homeLocations`||n===`workLocations`;p.push(A(e,{id:`${t}-suppress-incomplete-stays-${n}`,operation:V,invocationCount:r.length,bindings:[...v,{name:`values`,view:r,type:i?`u32`:`f32`,access:`read_write`}],declarations:i?``:f,body:`if (${b}) { values[valuesOffset + index] = ${i?`0xffffffffu`:`getNaN()`}; }`}))}return a.status?.candidateOverflow&&(a.realEntropy||p.push(j(e,{id:`${t}-candidate-overflow-reset`,operation:V,view:a.status.candidateOverflow,type:`u32`,value:`0u`,componentCount:1})),v.length&&p.push(A(e,{id:`${t}-staypoint-loss-status`,operation:V,invocationCount:1,bindings:[...v,{name:`candidateOverflow`,view:a.status.candidateOverflow,type:`u32`,access:`read_write`}],body:`if (${b}) { candidateOverflow[candidateOverflowOffset] = 1u; }`}))),p}},G=12,K=.62,q=.38,J=[.05,.04,.95,.19],Y=[.05,.21,.95,.33],X=R({parameters:B,caption:`One day of twelve Bay vessel legs, scaled to leave room for the bars. Indicators made for people over weeks are here read on vessels. Bars: radius of gyration (blue) and real entropy (amber). Distances are approximate: the frame stretches x and y differently.`,createInputs:({world:e})=>{let t=s(e,G),n=(t,n)=>[(t-e.width/2)*K+e.width/2,q+n*K],r=Float32Array.from(t.positions);for(let e=0;e<t.rowCount;e++)r.set(n(t.positions[e*2],t.positions[e*2+1]),e*2);let i=Float32Array.from(t.segments);for(let e=0;e<t.segmentCount;e++)i.set(n(t.segments[e*4],t.segments[e*4+1]),e*4),i.set(n(t.segments[e*4+2],t.segments[e*4+3]),e*4+2);return{...t,world:e,displayPositions:r,displaySegments:i,toDisplay:n,track:3}},build(n,a){let s=u(n,a),c=o(n,a,s),d=l(n,a,s,c),f=n.allocate(`location-parameters`,`float32`,4),p=n.allocate(`location-ids`,`uint32`,512),m=n.allocate(`location-count`,`uint32`,1);n.graph.add(new t({centroids:c.centroids.view,tracks:c.tracks.view,count:c.count.view,parameters:f.view,output:{locationIds:p.view,status:{count:m.view}}}));let h=a.trackCount,g=n.allocate(`radius-of-gyration`,`float32`,h),_=n.allocate(`real-entropy`,`float32`,h),v=n.allocate(`random-entropy`,`float32`,h),y=n.allocate(`home-locations`,`uint32`,h),b=n.allocate(`tracking-quality`,`float32`,h),x=n.allocate(`jump-lengths`,`float32`,512),S=n.allocate(`waiting-times`,`float32`,512);return n.graph.add(new W({trackCount:h,timestamps:s.timestamps.view,staypoints:{tracks:c.tracks.view,centroids:c.centroids.view,startRows:c.startRows.view,durations:c.durations.view,locationIds:p.view,status:{count:c.count.view,overflow:c.overflow.view,candidateOverflow:c.candidateOverflow.view}},triplegs:{tracks:d.triplegTracks.view,startRows:d.triplegStartRows.view,endRows:d.triplegEndRows.view,status:{count:d.triplegCount.view,overflow:d.triplegOverflow.view,candidateOverflow:d.triplegCandidateOverflow.view},staypointFinishRows:c.finishRows.view},output:{radiusOfGyration:g.view,jumpLengths:x.view,waitingTimes:S.view,randomEntropy:v.view,realEntropy:_.view,homeLocations:y.view,trackingQuality:b.view}})),{outputs:{segments:n.upload(`display-segments`,`float32x4`,a.displaySegments),positions:n.upload(`display-positions`,`float32x2`,a.displayPositions),centroids:c.centroids,tracks:c.tracks,staypointCount:c.count,staypointOverflow:c.overflow,staypointCandidateOverflow:c.candidateOverflow,locationIds:p,radiusOfGyration:g,realEntropy:_,randomEntropy:v,homeLocations:y,trackingQuality:b,selected:n.allocate(`selected-bar`,`float32`,h),selectedStays:n.allocate(`selected-stays`,`float32x2`,512)},update:t=>{r(c,t),d.parameters.write(i({maximumGap:0,activityMinimumDuration:40})),f.write(e({epsilon:t.epsilon}))}}},async afterRun({inputs:e,outputs:t,upload:r}){let[i,o,s,c]=await Promise.all([a(t.staypointCount),n(t.centroids),a(t.staypointOverflow),a(t.staypointCandidateOverflow)]),l=!!(s[0]||c[0]),u=l?0:Math.min(i[0],512),d=new Float32Array(Math.max(1,u)*2);for(let t=0;t<u;t++)d.set(e.toDisplay(o[t*2],o[t*2+1]),t*2);return{stays:u,incomplete:l,positions:r(`display-stays`,`float32x2`,d)}},interaction:{kind:`probe`,hint:`probe a track`,rest:({toUnit:e,inputs:t})=>e([t.displayPositions[240],t.displayPositions[241]]),resolve(e,{inputs:t,outputs:n,peek:r}){let i=0,a=1/0;for(let n=0;n<t.rowCount;n++){let r=Math.hypot(t.displayPositions[n*2]-e.x,t.displayPositions[n*2+1]-e.y);r<a&&(i=n,a=r)}let o=c(t.trackOffsets,i),s=r(n.radiusOfGyration),l=r(n.realEntropy),u=r(n.trackingQuality),d=r(n.centroids),f=r(n.tracks),p=r(n.staypointCount);if(!s||!l||!u||!d||!f||!p)return{};let m=r(n.staypointOverflow),h=r(n.staypointCandidateOverflow);if(!m||!h)return{readout:`Mobility indicators unavailable: status not read`};if(m[0]||h[0])return n.selected.write(new Float32Array(t.trackCount)),{glyphs:[],readout:`Mobility indicators unavailable: incomplete staypoint table`};let g=0,_=0,v=0,y=[];for(let e=0;e<Math.min(p[0],512);e++){if(f[e]!==o)continue;let n=t.toDisplay(d[e*2],d[e*2+1]);y.push({shape:`dot`,at:n}),g+=d[e*2],_+=d[e*2+1],v++}let b=new Float32Array(t.trackCount);b[o]=s[o],n.selected.write(b);let x=v?t.toDisplay(g/v,_/v):null;return{glyphs:[...y,...x?[{shape:`ring`,at:x,radius:s[o]*K,dashed:!0},{shape:`crosshair`,at:x}]:[]],readout:`track ${o} · gyration ${z(s[o],t.metersPerUnit)} · ${v} stays · entropy ${l[o].toFixed(1)} bits · ${Number.isFinite(u[o])?`tracked ${Math.round(u[o]*100)}%`:`tracking quality unavailable: incomplete tripleg table`}`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>n?.incomplete?[{layer:`lines`,context:!0,segments:t.segments,count:e.segmentCount},{layer:`dots`,context:!0,positions:t.positions,radius:2.5}]:[{layer:`lines`,context:!0,segments:t.segments,count:e.segmentCount},{layer:`dots`,context:!0,positions:t.positions,radius:2.5},...n?[{layer:`dots`,positions:n.positions,count:n.stays,radius:7,color:[1,.7,.2,.9]}]:[],{layer:`chart`,kind:`bars`,values:t.realEntropy,range:[0,4],rectangle:Y,color:[...r.categories[1],1]},{layer:`chart`,kind:`bars`,values:t.radiusOfGyration,range:[0,.3],rectangle:J,color:[...r.categories[0],1]},{layer:`chart`,kind:`bars`,values:t.selected,range:[0,.3],rectangle:J,color:[1,.7,.2,1]}]});export{X as default};
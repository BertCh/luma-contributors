import{s as e}from"./shape-segments-DMcyF4V-.js";var t=1e6;function n(n,i){let a=n.allocate(`bucket-links`,`float32x4`,i.pairCapacity);return e(n,{id:`bucket-links-kernel`,invocationCount:i.pairCapacity,bindings:[{name:`ids`,buffer:i.ids,access:`read`},{name:`partners`,buffer:i.partners,access:`read`},{name:`firstBuckets`,buffer:i.firstBuckets,access:`read`},{name:`count`,buffer:i.count,access:`read`},{name:`samples`,buffer:i.samples,access:`read`},{name:`links`,buffer:a,access:`read_write`}],body:`var lowerX = ${t.toFixed(1)};
  var lowerY = ${t.toFixed(1)};
  var partnerX = ${t.toFixed(1)};
  var partnerY = ${t.toFixed(1)};
  if (index < count[countOffset]) {
    let bucket = firstBuckets[firstBucketsOffset + index];
    let lowerRow = ids[idsOffset + index] * ${i.bucketCount}u + bucket;
    let partnerRow = partners[partnersOffset + index] * ${i.bucketCount}u + bucket;
    lowerX = samples[samplesOffset + lowerRow * 2u];
    lowerY = samples[samplesOffset + lowerRow * 2u + 1u];
    partnerX = samples[samplesOffset + partnerRow * 2u];
    partnerY = samples[samplesOffset + partnerRow * 2u + 1u];
  }
  links[linksOffset + index * 4u] = lowerX;
  links[linksOffset + index * 4u + 1u] = lowerY;
  links[linksOffset + index * 4u + 2u] = partnerX;
  links[linksOffset + index * 4u + 3u] = partnerY;`}),{links:a,ends:r(n,a,i.pairCapacity)}}function r(t,n,r){let i=t.allocate(`link-ends`,`float32x2`,r*2);return e(t,{id:`${n.id}-ends-kernel`,invocationCount:r,bindings:[{name:`links`,buffer:n,access:`read`},{name:`ends`,buffer:i,access:`read_write`}],body:`ends[endsOffset + index * 4u] = links[linksOffset + index * 4u];
  ends[endsOffset + index * 4u + 1u] = links[linksOffset + index * 4u + 1u];
  ends[endsOffset + index * 4u + 2u] = links[linksOffset + index * 4u + 2u];
  ends[endsOffset + index * 4u + 3u] = links[linksOffset + index * 4u + 3u];`}),i}function i(e,n){return`var ${e}X = ${t.toFixed(1)};
    var ${e}Y = ${t.toFixed(1)};
    let ${e}Start = trackOffsets[trackOffsetsOffset + ${n}];
    let ${e}End = trackOffsets[trackOffsetsOffset + ${n} + 1u];
    for (var row = ${e}Start; row + 1u < ${e}End; row = row + 1u) {
      let before = timestamps[timestampsOffset + row];
      let after = timestamps[timestampsOffset + row + 1u];
      if (time >= before && time <= after) {
        let fraction = select(0.0, (time - before) / (after - before), after > before);
        ${e}X = mix(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 2u], fraction);
        ${e}Y = mix(positions[positionsOffset + row * 2u + 1u], positions[positionsOffset + row * 2u + 3u], fraction);
        break;
      }
    }`}function a(n,a){let o=n.allocate(`approach-links`,`float32x4`,a.pairCapacity);return e(n,{id:`approach-links-kernel`,invocationCount:a.pairCapacity,bindings:[{name:`ids`,buffer:a.ids,access:`read`},{name:`partners`,buffer:a.partners,access:`read`},{name:`closestTimes`,buffer:a.closestTimes,access:`read`},{name:`count`,buffer:a.count,access:`read`},{name:`positions`,buffer:a.positions,access:`read`},{name:`timestamps`,buffer:a.timestamps,access:`read`},{name:`trackOffsets`,buffer:a.trackOffsets,access:`read`},{name:`links`,buffer:o,access:`read_write`}],body:`var lowerX = ${t.toFixed(1)};
  var lowerY = ${t.toFixed(1)};
  var partnerX = ${t.toFixed(1)};
  var partnerY = ${t.toFixed(1)};
  if (index < count[countOffset]) {
    let lowerTrack = ids[idsOffset + index];
    let partnerTrack = partners[partnersOffset + index];
    let time = timestamps[timestampsOffset + trackOffsets[trackOffsetsOffset + lowerTrack]] + closestTimes[closestTimesOffset + index];
    ${i(`lowerPosition`,`lowerTrack`)}
    ${i(`partnerPosition`,`partnerTrack`)}
    lowerX = lowerPositionX;
    lowerY = lowerPositionY;
    partnerX = partnerPositionX;
    partnerY = partnerPositionY;
  }
  links[linksOffset + index * 4u] = lowerX;
  links[linksOffset + index * 4u + 1u] = lowerY;
  links[linksOffset + index * 4u + 2u] = partnerX;
  links[linksOffset + index * 4u + 3u] = partnerY;`}),{links:o,ends:r(n,o,a.pairCapacity)}}export{a as n,n as t};
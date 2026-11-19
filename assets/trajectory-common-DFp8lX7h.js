import{n as e}from"./temporal-context-xddVhAJD.js";import{t}from"./time-words-BFl8KCxq.js";var n=`3.402823466e+38`;function r(e){return`select(${e}, ${n}, ${e} == 0.0)`}function i(t,n,r,i){if(!r)return;if(!n)return e(t,r,{axisKinds:[`irregular`]});let a=n.format===`uint32x2`?`int64-words`:i?`double-single`:`float32-relative`;return e(t,r,{encodings:[a],axisKinds:[`irregular`]})}function a(e,t){let n=!!(t.times||t.offsets),r=!!(t.timestamps||t.trackOffsets),i=t.times??t.timestamps,a=t.offsets??t.trackOffsets;if(n===r||!i||!a)throw Error(`${e} takes either timestamps and trackOffsets, or times and offsets`);return{times:i,offsets:a}}function o(e){return Math.max(1,32-Math.clz32(Math.max(e,1)))}function s(e){return e?`${t}
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  let wordsA = vec2<u32>(timestamps[timestampsOffset + 2u * a], timestamps[timestampsOffset + 2u * a + 1u]);
  let wordsB = vec2<u32>(timestamps[timestampsOffset + 2u * b], timestamps[timestampsOffset + 2u * b + 1u]);
  return timeWordsDifferenceToF32(wordsA, wordsB);
}`:`
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}`}function c(e,t,n){if(e.length<t)throw Error(`${n} target must hold ${t} elements`)}export{s as a,o as i,r as n,i as o,a as r,c as t};
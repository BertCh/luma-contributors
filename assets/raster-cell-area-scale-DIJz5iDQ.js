import{r as e,s as t}from"./raster-grid-VB0AM_GY.js";import{c as n}from"./wgsl-kernel-nodes-DagJklH2.js";var r=1,i=5;function a(e,t,n){return t===`uniform`||(n?n[3]===0&&n[4]===0:e.affine[1]===0&&e.affine[3]===0)}function o(e,t,n){return t===`uniform`?`true`:n?`${n}[${n}Offset+3u]==0.0&&${n}[${n}Offset+4u]==0.0`:String(a(e,t))}function s(t,a){let o=t.pixelInterpretation===`point`?-.5:0,s=`let rowFraction = (f32(row) + 0.5) / ${n(t.height)};`;if(a===void 0){let{northEdge:r,southEdge:i}=e(t);return`${s}
  let edge = mix(${n(r)}, ${n(i)}, rowFraction);`}let c=`${a}Offset`;return`${s}
  let originY = ${a}[${c} + ${r}u];
  let rowStepY = ${a}[${c} + ${i}u];
  let northEdge = originY + rowStepY * ${n(o)};
  let southEdge = originY + rowStepY * ${n(t.height+o)};
  let edge = mix(northEdge, southEdge, rowFraction);`}function c(e,n,r){return`
fn getRowAreaScale(row: u32) -> f32 {
  ${s(e,r)}
  ${t(n)}
  return groundScale.x * groundScale.y;
}`}export{c as n,s as r,o as t};
import{p as e}from"./wgsl-kernel-nodes-DagJklH2.js";function t(t,n=`u32`,r=t.charAt(0).toUpperCase()+t.slice(1)){if(e(`getBinarySearchWGSL`,`bindingName`,t),e(`getBinarySearchWGSL`,`binding offset`,`${t}Offset`),!/^[\p{XID_Continue}]+$/u.test(r))throw Error(`getBinarySearchWGSL bindingName and functionSuffix must be WGSL identifiers, got ${t}, ${r}`);let i=r,a=e=>n===`u64`?`vec2<u32>(${t}[${t}Offset + 2u * ${e} + 1u], ${t}[${t}Offset + 2u * ${e}])`:`${t}[${t}Offset + ${e}]`,o=n===`u64`?`vec2<u32>`:n,s=``,c,l;n===`f32`?(s=`fn getBinarySearchOrderedKey${i}(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`,c=(e,t)=>`getBinarySearchOrderedKey${i}(${e}) < getBinarySearchOrderedKey${i}(${t})`,l=(e,t)=>`getBinarySearchOrderedKey${i}(${e}) <= getBinarySearchOrderedKey${i}(${t})`):n===`u64`?(c=(e,t)=>`(${e}.x < ${t}.x || (${e}.x == ${t}.x && ${e}.y < ${t}.y))`,l=(e,t)=>`(${e}.x < ${t}.x || (${e}.x == ${t}.x && ${e}.y <= ${t}.y))`):(c=(e,t)=>`${e} < ${t}`,l=(e,t)=>`${e} <= ${t}`);let u=(e,t)=>`
fn ${e}${i}(first: u32, last: u32, key: ${o}) -> u32 {
  var low = first;
  var high = max(first, last);
  while (low < high) {
    let middle = low + (high - low) / 2u;
    let value = ${a(`middle`)};
    if (${t(`value`)}) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}
`;return`
${s}${u(`lowerBound`,e=>c(e,`key`))}${u(`upperBound`,e=>l(e,`key`))}`}export{t};
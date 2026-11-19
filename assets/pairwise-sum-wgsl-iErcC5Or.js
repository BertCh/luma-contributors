function e(e){let t=e.name??`sum`;return`var ${t}Levels: array<f32, 32>;
  var ${t}Depth = 0u;
  for (var ${t}Item = 0u; ${t}Item < ${e.count}; ${t}Item++) {
    ${e.setup??``}
    var ${t}Block = ${e.term};
    // One merge per trailing one bit of the number of terms already taken.
    var ${t}Carry = ${t}Item;
    for (var level = 0u; level < 32u; level++) {
      if ((${t}Carry & 1u) == 0u) {
        break;
      }
      ${t}Depth--;
      ${t}Block = ${t}Levels[${t}Depth] + ${t}Block;
      ${t}Carry = ${t}Carry >> 1u;
    }
    ${t}Levels[${t}Depth] = ${t}Block;
    ${t}Depth++;
  }
  var ${t} = 0.0;
  for (var level = 0u; level < 32u; level++) {
    if (${t}Depth == 0u) {
      break;
    }
    ${t}Depth--;
    ${t} = ${t}Levels[${t}Depth] + ${t};
  }`}export{e as t};
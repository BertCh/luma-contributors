function e(e,t){return`fn ${e}(row: u32) -> vec2f { return vec2f(${t}[${t}Offset + row * 2u], ${t}[${t}Offset + row * 2u + 1u]); }`}function t(e,t){return`fn ${e}(row: u32) -> vec3f { let base = ${t}Offset + row * 3u; return vec3f(${t}[base], ${t}[base + 1u], ${t}[base + 2u]); }`}var n=`
fn isSpatialFinite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
`,r=`
fn getSpatialNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }
`;export{e as i,n,t as r,r as t};
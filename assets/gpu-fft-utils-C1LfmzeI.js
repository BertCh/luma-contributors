var e=2048,t=`
const GPU_FFT_PI: f32 = 3.14159265358979323846;

fn reverseLowBits(value: u32, bitCount: u32) -> u32 {
  var source = value;
  var reversed = 0u;
  for (var bitIndex = 0u; bitIndex < bitCount; bitIndex++) {
    reversed = (reversed << 1u) | (source & 1u);
    source = source >> 1u;
  }
  return reversed;
}

fn multiplyComplex(left: vec2f, right: vec2f) -> vec2f {
  return vec2f(
    left.x * right.x - left.y * right.y,
    left.x * right.y + left.y * right.x
  );
}
`;function n(e){let t=Math.log2(e);return[{kind:`bit-reversal`,stage:t},...Array.from({length:t},(e,t)=>({kind:`butterfly`,stage:t+1}))]}function r(t,n,r){if(!Number.isInteger(r))return`${t} ${n} must be an integer.`;if(r<2||r>2048)return`${t} ${n} must be from 2 through ${e}.`;if(r&r-1)return`${t} ${n} must be a power of two.`}export{n as i,e as n,r,t};
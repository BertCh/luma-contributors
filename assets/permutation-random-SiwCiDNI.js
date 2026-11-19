import{n as e}from"./philox-random-DfVxWNPn.js";var t=`
${e}

fn getPhiloxMultiplyHigh(a: u32, b: u32) -> u32 {
  return philoxMulHiLo(a, b).x;
}

fn getPhilox4x32(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
  return philox4x32(counter, key);
}

struct PhiloxStream {
  key: vec2<u32>,
  words: vec3<u32>,
  block: u32,
  lane: u32,
  buffer: vec4<u32>
}

fn createPhiloxStream(key: vec2<u32>, a: u32, b: u32, c: u32) -> PhiloxStream {
  var stream: PhiloxStream;
  stream.key = key;
  stream.words = vec3<u32>(a, b, c);
  stream.block = 0u;
  stream.lane = 4u;
  stream.buffer = vec4<u32>(0u);
  return stream;
}

fn nextPhiloxUint32(stream: ptr<function, PhiloxStream>) -> u32 {
  if ((*stream).lane == 4u) {
    (*stream).buffer = getPhilox4x32(
      vec4<u32>((*stream).block, (*stream).words.x, (*stream).words.y, (*stream).words.z),
      (*stream).key
    );
    (*stream).block = (*stream).block + 1u;
    (*stream).lane = 0u;
  }
  let word = (*stream).buffer[(*stream).lane];
  (*stream).lane = (*stream).lane + 1u;
  return word;
}

fn nextPhiloxBelow(stream: ptr<function, PhiloxStream>, bound: u32) -> u32 {
  var value = nextPhiloxUint32(stream);
  var low = value * bound;
  if (low < bound) {
    let threshold = (0u - bound) % bound;
    while (low < threshold) {
      value = nextPhiloxUint32(stream);
      low = value * bound;
    }
  }
  return getPhiloxMultiplyHigh(value, bound);
}
`;export{t};
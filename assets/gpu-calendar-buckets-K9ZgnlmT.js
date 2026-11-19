import{i as e,t}from"./graph-chunk-groups-BB8ACpQO.js";import{t as n}from"./gpu-dispatch-utils-CbRzFAgT.js";import{l as r,o as i}from"./gpu-command-graph-BdzdmV6u.js";import{t as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as o}from"./temporal-context-xddVhAJD.js";import{E as s,c,p as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d,i as f,o as p,r as m}from"./graph-chunk-dispatch-zBt8k2-o.js";var h=1440,g=1e9;function _(e,t=0,n=new Int32Array(2)){if(n.length<2)throw Error(`Calendar bucket parameter target must hold 2 elements`);if(!Number.isInteger(e)||Math.abs(e)>1440)throw Error(`Calendar bucket UTC offset must be an integer in [-${h}, ${h}] minutes`);if(!Number.isInteger(t)||t<0||t>6)throw Error(`Calendar bucket first day of week must be an integer in [0, 6]`);return n[0]=e,n[1]=t,n}var v=`GPUCalendarBuckets`,y=8,b=256,x=Object.freeze({encodings:Object.freeze([`int64-words`]),unixEpoch:!0}),S=[{name:`year`,field:`year`,signed:!0},{name:`month`,field:`month`},{name:`dayOfMonth`,field:`day`},{name:`hour`,field:`hour`},{name:`minute`,field:`minute`},{name:`weekday`,field:`weekday`},{name:`dayOfYear`,field:`dayOfYear`},{name:`isoWeek`,field:`isoWeek`},{name:`isoWeekYear`,field:`isoWeekYear`,signed:!0},{name:`quarter`,field:`quarter`}],C=class{id;props;rowCount;unit;estimatedCommandNodeCount;chunkPlan;constructor(e){this.id=e.id??`calendar-buckets`,this.props=e;let t=this.id,{output:n}=e;this.unit=w(t,e);for(let n of c(e.timestamps))s(n,[`uint32x2`],`${t} timestamps`);if(this.rowCount=e.timestamps.length,this.rowCount<1)throw Error(`${t} needs at least one row`);if(e.mask){for(let n of c(e.mask))s(n,[`uint32`],`${t} mask`);if(e.mask.length!==this.rowCount)throw Error(`${t} mask length must equal timestamps length`)}if(e.utcOffsets){for(let n of c(e.utcOffsets))s(n,[`sint32`],`${t} utcOffsets`);if(e.utcOffsets.length!==this.rowCount)throw Error(`${t} utcOffsets length must equal timestamps length`)}if(e.timeZone){if(e.utcOffsets)throw Error(`${t} timeZone and utcOffsets are mutually exclusive`);if(s(e.timeZone.transitions,[`uint32x2`],`${t} timeZone.transitions`),s(e.timeZone.offsets,[`sint32`],`${t} timeZone.offsets`),e.timeZone.transitions.length<1)throw Error(`${t} timeZone needs at least one transition`);if(e.timeZone.offsets.length!==e.timeZone.transitions.length+1)throw Error(`${t} timeZone.offsets length must be transitions length + 1`)}if(s(e.parameters,[`sint32`],`${t} parameters`),e.parameters.length<2)throw Error(`${t} parameters must hold 2 sint32 values`);let a=0;for(let e of S){let r=n[e.name];if(r){a++;for(let n of c(r))s(n,[e.signed?`sint32`:`uint32`],`${t} output.${e.name}`);if(r.length<this.rowCount)throw Error(`${t} output.${e.name} must hold one row per timestamp`)}}if(n.hourWeekdayCounts&&(a++,s(n.hourWeekdayCounts,[`uint32`],`${t} output.hourWeekdayCounts`),n.hourWeekdayCounts.length<168))throw Error(`${t} output.hourWeekdayCounts must hold 168 elements`);if(a===0)throw Error(`${t} needs at least one output`);u(t,this.getOutputViews(),this.getInputViews());let o=Object.fromEntries(S.map(({name:e})=>{let t=n[e];return[e,t instanceof i&&t.length>this.rowCount?new i(t.buffer,{...t,length:this.rowCount}):t]}));this.chunkPlan=m(t,{timestamps:e.timestamps,mask:e.mask,utcOffsets:e.utcOffsets,...o});let d=2+Number(!!e.mask)+Number(!!e.utcOffsets)+(e.timeZone?2:0),p=n.hourWeekdayCounts,h=p?new i(p.buffer,{...p,length:168}):void 0,g=[...S.filter(e=>n[e.name]).map(e=>o[e.name]),...p?[p]:[]],_=[];for(let e=0;e<g.length;e+=y-d)_.push(g.slice(e,e+y-d));{let e={...o,hourWeekdayCounts:h},n=_,a=Object.values(e).flatMap(e=>e?c(e):[]).map(e=>new i(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset,byteStride:4,rowByteLength:4})),s=a.reduce((e,t)=>e+t.length,0);f(t,{outputs:new r({id:`${t}-writable-ranges`,name:`outputs`,format:`uint32`,length:s,valueLength:s,stride:1,byteStride:4,rowByteLength:4,data:a})});for(let e of n)f(t,Object.fromEntries(e.map((e,t)=>[String(t),e])))}this.estimatedCommandNodeCount=Number(!!n.hourWeekdayCounts)+this.chunkPlan.nonemptyChunkCount*Math.ceil(a/(y-d)),l(t,this.estimatedCommandNodeCount)}getCommandNodes(r){let{props:i,id:o,unit:s,chunkPlan:c}=this,{output:l}=i;p(r,o,c),d(r,o,{parameters:i.parameters,transitions:i.timeZone?.transitions,zoneOffsets:i.timeZone?.offsets,hourWeekdayCounts:l.hourWeekdayCounts}),e(r,o,Math.min(y,2+Number(!!i.mask)+Number(!!i.utcOffsets)+(i.timeZone?2:0)+Object.values(l).filter(Boolean).length),b);for(let e of c.chunks)e.length>0&&n(o,e.length,b,r.device.limits.maxComputeWorkgroupsPerDimension);let u=[],f=[];for(let e of S)l[e.name]&&f.push({name:e.name,type:e.signed?`i32`:`u32`,write:`${e.name}Out[${e.name}OutBase + index] = fields.${e.field};`});l.hourWeekdayCounts&&(u.push(a(r,{id:`${o}-fill-matrix`,operation:v,view:l.hourWeekdayCounts,type:`u32`,value:`0u`,componentCount:168})),f.push({name:`hourWeekdayCounts`,type:`atomic<u32>`,write:`if (fields.valid) {
    atomicAdd(&hourWeekdayCountsOut[hourWeekdayCountsOutOffset + fields.weekday * 24u + fields.hour], 1u);
  }`}));let m=(e,t,n)=>({name:e,view:t,type:n,access:`read`});for(let e of c.chunks){if(e.length===0)continue;let n=[m(`timestamps`,e.views.timestamps,`u32`),m(`params`,i.parameters,`i32`)];i.mask&&n.push(m(`rowMask`,e.views.mask,`u32`)),i.utcOffsets&&n.push(m(`rowOffsets`,e.views.utcOffsets,`i32`)),i.timeZone&&n.push(m(`zoneTransitions`,i.timeZone.transitions,`u32`),m(`zoneOffsets`,i.timeZone.offsets,`i32`));let a=y-n.length;for(let c=0,d=0;c<f.length;c+=a,d++){let p=f.slice(c,c+a);u.push(t(r,{id:`${o}-decode-${d}-chunk-${e.chunkIndex}`,operation:v,variant:`decode`,bindings:[...n,...p.map(t=>({name:`${t.name}Out`,view:t.name===`hourWeekdayCounts`?l.hourWeekdayCounts:e.views[t.name],type:t.type,access:`read_write`}))],invocationCount:e.length,workgroupSize:b,declarations:`${T}${O[s]}${i.timeZone?k:``}`,body:`if (index >= getChunkParameter(1u)) { return; }
  let timestampsBase = arrayLength(&timestamps) - 2u * getChunkParameter(1u);
  ${i.mask?`let rowMaskBase = arrayLength(&rowMask) - getChunkParameter(1u);`:``}
  ${i.utcOffsets?`let rowOffsetsBase = arrayLength(&rowOffsets) - getChunkParameter(1u);`:``}
  ${p.filter(e=>e.name!==`hourWeekdayCounts`).map(e=>`let ${e.name}OutBase = arrayLength(&${e.name}Out) - getChunkParameter(1u);`).join(`
  `)}
  let time = toEpochMilliseconds(vec2<u32>(timestamps[timestampsBase + 2u * index], timestamps[timestampsBase + 2u * index + 1u]));
  let offsetMilliseconds = ${i.timeZone?`getZoneOffsetMilliseconds(time)`:`minutesToMilliseconds(${i.utcOffsets?`rowOffsets[rowOffsetsBase + index]`:`params[paramsOffset]`})`};
  let firstDay = u32(((params[paramsOffset + 1u] % 7) + 7) % 7);
  var fields = decodeCalendar(time, offsetMilliseconds, firstDay);
  ${i.mask?`if (rowMask[rowMaskBase + index] == 0u) {
    fields = invalidCalendar();
  }`:``}
  ${p.map(e=>e.write).join(`
  `)}`}))}}return u}getInputViews(){let{props:e}=this;return[e.timestamps,e.mask,e.utcOffsets,e.timeZone?.transitions,e.timeZone?.offsets,e.parameters].filter(e=>!!e)}getOutputViews(){return Object.values(this.props.output).filter(e=>!!e)}};function w(e,{unit:t,temporalContext:n}){if(!n)return t??`millisecond`;let r=o(e,n,x).unit;if(t!==void 0&&t!==r)throw Error(`${e} unit ${t} contradicts temporalContext unit ${r}`);return r}var T=`
const MILLISECONDS_PER_DAY: u32 = 86400000u;
const MAXIMUM_DAYS: u32 = ${g}u;

struct CalendarFields {
  valid: bool,
  year: i32,
  month: u32,
  day: u32,
  hour: u32,
  minute: u32,
  weekday: u32,
  dayOfYear: u32,
  isoWeek: u32,
  isoWeekYear: i32,
  quarter: u32
}

fn invalidCalendar() -> CalendarFields {
  let invalidYear = bitcast<i32>(0x80000000u);
  return CalendarFields(false, invalidYear, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
    0xffffffffu, 0xffffffffu, 0xffffffffu, invalidYear, 0xffffffffu);
}

fn isLeapYear(year: i32) -> bool {
  return (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
}

// Hinnant civil_from_days: days since 1970-01-01 to (year, month, day).
fn civilFromDays(days: i32) -> vec3<i32> {
  let z = days + 719468;
  let era = select((z - 146096) / 146097, z / 146097, z >= 0);
  let dayOfEra = z - era * 146097;
  let yearOfEra = (dayOfEra - dayOfEra / 1460 + dayOfEra / 36524 - dayOfEra / 146096) / 365;
  let dayOfMarchYear = dayOfEra - (365 * yearOfEra + yearOfEra / 4 - yearOfEra / 100);
  let monthPrime = (5 * dayOfMarchYear + 2) / 153;
  let day = dayOfMarchYear - (153 * monthPrime + 2) / 5 + 1;
  let month = select(monthPrime - 9, monthPrime + 3, monthPrime < 10);
  let year = yearOfEra + era * 400 + select(0, 1, month <= 2);
  return vec3<i32>(year, month, day);
}

fn getDayOfYear(date: vec3<i32>) -> i32 {
  let adjustment = select(select(-2, -1, isLeapYear(date.x)), 0, date.y <= 2);
  return (367 * date.y - 362) / 12 + date.z + adjustment;
}

struct DivideResult {
  quotientLow: u32,
  quotientHigh: u32,
  remainder: u32
}

// Exact unsigned 64-bit division by a divisor below 2^28, one nibble at a time so that
// remainder * 16 never overflows 32 bits.
fn divideUnsigned64(value: vec2<u32>, divisor: u32) -> DivideResult {
  var remainder = value.y % divisor;
  var quotientLow = 0u;
  for (var nibble = 7i; nibble >= 0i; nibble = nibble - 1i) {
    let shift = u32(nibble) * 4u;
    let current = (remainder << 4u) | ((value.x >> shift) & 0xfu);
    quotientLow = quotientLow | ((current / divisor) << shift);
    remainder = current % divisor;
  }
  return DivideResult(quotientLow, value.y / divisor, remainder);
}

// Minutes to milliseconds; an offset beyond +-1440 minutes maps to a value the decode rejects.
fn minutesToMilliseconds(offsetMinutes: i32) -> i32 {
  return select(offsetMinutes * 60000, 0x7fffffff, offsetMinutes < -1440 || offsetMinutes > 1440);
}

fn decodeCalendar(time: vec2<u32>, offsetMilliseconds: i32, firstDay: u32) -> CalendarFields {
  let invalid = invalidCalendar();
  if (offsetMilliseconds < -86400000 || offsetMilliseconds > 86400000) {
    return invalid;
  }
  // Signed 64-bit local time = time + offset, with overflow detection.
  let offsetLow = bitcast<u32>(offsetMilliseconds);
  let offsetHigh = select(0u, 0xffffffffu, offsetMilliseconds < 0);
  let low = time.x + offsetLow;
  let high = time.y + offsetHigh + select(0u, 1u, low < time.x);
  let timeSign = time.y >> 31u;
  let sumSign = high >> 31u;
  if (timeSign == (offsetHigh >> 31u) && sumSign != timeSign) {
    return invalid;
  }
  let isNegative = sumSign != 0u;
  var magnitude = vec2<u32>(low, high);
  if (isNegative) {
    let negatedLow = ~low + 1u;
    magnitude = vec2<u32>(negatedLow, ~high + select(0u, 1u, negatedLow == 0u));
  }
  let division = divideUnsigned64(magnitude, MILLISECONDS_PER_DAY);
  if (division.quotientHigh != 0u || division.quotientLow > MAXIMUM_DAYS) {
    return invalid;
  }
  var days = i32(division.quotientLow);
  var millisecondsOfDay = division.remainder;
  if (isNegative) {
    // floor(-(q * D + r) / D) = -(q + 1) with D - r left over when r is not zero.
    if (millisecondsOfDay != 0u) {
      days = days + 1;
      millisecondsOfDay = MILLISECONDS_PER_DAY - millisecondsOfDay;
    }
    days = -days;
  }
  let date = civilFromDays(days);
  let isoWeekday = u32(((days + 3) % 7 + 7) % 7);
  let thursday = days - i32(isoWeekday) + 3;
  let thursdayDate = civilFromDays(thursday);
  var fields = invalid;
  fields.valid = true;
  fields.year = date.x;
  fields.month = u32(date.y);
  fields.day = u32(date.z);
  fields.hour = millisecondsOfDay / 3600000u;
  fields.minute = (millisecondsOfDay % 3600000u) / 60000u;
  fields.weekday = (isoWeekday + 7u - firstDay) % 7u;
  fields.dayOfYear = u32(getDayOfYear(date));
  fields.isoWeek = u32((getDayOfYear(thursdayDate) - 1) / 7 + 1);
  fields.isoWeekYear = thursdayDate.x;
  fields.quarter = u32((date.y - 1) / 3 + 1);
  return fields;
}
`,E=`
fn negateInt64(value: vec2<u32>) -> vec2<u32> {
  let low = ~value.x + 1u;
  return vec2<u32>(low, ~value.y + select(0u, 1u, low == 0u));
}
`,D=e=>`${E}
fn toEpochMilliseconds(time: vec2<u32>) -> vec2<u32> {
  let isNegative = (time.y >> 31u) != 0u;
  let magnitude = select(time, negateInt64(time), isNegative);
  let division = divideUnsigned64(magnitude, ${e}u);
  var quotient = vec2<u32>(division.quotientLow, division.quotientHigh);
  if (isNegative && division.remainder != 0u) {
    quotient = vec2<u32>(quotient.x + 1u, quotient.y + select(0u, 1u, quotient.x == 0xffffffffu));
  }
  return select(quotient, negateInt64(quotient), isNegative);
}
`,O={millisecond:`
fn toEpochMilliseconds(time: vec2<u32>) -> vec2<u32> {
  return time;
}
`,microsecond:D(`1000`),nanosecond:D(`1000000`),second:`${E}
fn toEpochMilliseconds(time: vec2<u32>) -> vec2<u32> {
  let isNegative = (time.y >> 31u) != 0u;
  let magnitude = select(time, negateInt64(time), isNegative);
  if (magnitude.y >= 0x100000u) {
    return select(vec2<u32>(0xffffffffu, 0x7fffffffu), vec2<u32>(0u, 0x80000000u), isNegative);
  }
  let lowPart = (magnitude.x & 0xffffu) * 1000u;
  let highPart = (magnitude.x >> 16u) * 1000u;
  let shifted = highPart << 16u;
  let low = shifted + lowPart;
  let high = magnitude.y * 1000u + (highPart >> 16u) + select(0u, 1u, low < shifted);
  let product = vec2<u32>(low, high);
  return select(product, negateInt64(product), isNegative);
}
`},k=`
// Signed 64-bit a <= b on (low, high) words.
fn lessOrEqualInt64(a: vec2<u32>, b: vec2<u32>) -> bool {
  let aHigh = bitcast<i32>(a.y);
  let bHigh = bitcast<i32>(b.y);
  if (aHigh != bHigh) {
    return aHigh < bHigh;
  }
  return a.x <= b.x;
}

// Upper-bound binary search: the number of transitions at or before the instant, which indexes the
// offset in force.
fn getZoneOffsetMilliseconds(time: vec2<u32>) -> i32 {
  var low = 0u;
  var high = (arrayLength(&zoneTransitions) - zoneTransitionsOffset) / 2u;
  loop {
    if (low >= high) {
      break;
    }
    let middle = low + (high - low) / 2u;
    let transition = vec2<u32>(
      zoneTransitions[zoneTransitionsOffset + 2u * middle],
      zoneTransitions[zoneTransitionsOffset + 2u * middle + 1u]);
    if (lessOrEqualInt64(transition, time)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let seconds = zoneOffsets[zoneOffsetsOffset + low];
  // A table offset beyond +-86400 s maps to a value the decode rejects.
  return select(seconds * 1000, 0x7fffffff, seconds < -86400 || seconds > 86400);
}
`;export{_ as n,C as t};
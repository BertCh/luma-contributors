import{i as e,t}from"./graph-chunk-groups-BB8ACpQO.js";import{t as n}from"./gpu-dispatch-utils-CbRzFAgT.js";import{o as r}from"./gpu-command-graph-BdzdmV6u.js";import{n as i}from"./temporal-context-xddVhAJD.js";import{t as a}from"./status-aOwAAXek.js";import{D as o,E as s,T as c,c as l,f as u,p as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{a as p,i as m,o as h,r as g}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as _}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as v}from"./temporal-reduction-word-division-DIM5475u.js";import{t as y}from"./time-words-BFl8KCxq.js";import{t as b}from"./uint64-wgsl-DCuW6-R6.js";function x(e,t,n){let r=t<=2?e-1:e,i=Math.trunc((r>=0?r:r-399)/400),a=r-i*400,o=(t+9)%12,s=Math.trunc((153*o+2)/5)+n-1,c=a*365+Math.trunc(a/4)-Math.trunc(a/100)+s;return i*146097+c-719468}function S(e,t){let{year:n,month:r=1,day:i=1,hour:a=0}=t;switch(e){case`year`:return n;case`quarter`:return n*4+Math.trunc((r-1)/3);case`month`:return n*12+r-1;case`week`:case`day`:return x(n,r,i);case`hour`:return x(n,r,i)*24+a}}function C(e){return(e.interval??1)*(e.calendarUnit===`week`?7:1)}function w(e,t){u(e,`periods.step steps per period (MAXIMUM_CALENDAR_STEPS_PER_PERIOD)`,C(t),1,2147483647)}function T(e,t,n){u(e,`periods.origin step index (MINIMUM_CALENDAR_STEP_INDEX..MAXIMUM_CALENDAR_STEP_INDEX)`,S(t,n),-2147483648,2147483647)}var E=`GPUPeriodIndex`,D=256,O=Object.freeze({encodings:Object.freeze([`int64-words`]),unixEpoch:!0}),k=[`year`,`quarter`,`month`,`week`,`day`,`hour`],A=(1n<<63n)-1n,j=-(1n<<63n),M=4294967295,N=4294967294,P=4294967293,F=class{id;props;rowCount;temporalContext;estimatedCommandNodeCount;chunkPlan;constructor(e){this.id=e.id??`period-index`,this.props=e;let t=this.id,{periods:n,output:p}=e;if(this.temporalContext=i(t,e.temporalContext,O),u(t,`periods.count`,n.count,1,P),this.rowCount=n.kind===`calendar`?this.validateCalendar(n):this.validateFixed(n),this.rowCount<1)throw Error(`${t} needs at least one row`);if(e.mask){for(let n of l(e.mask))s(n,[`uint32`],`${t} mask`);o(t,`mask`,e.mask,this.rowCount)}for(let e of l(p.periods))s(e,[`uint32`],`${t} output.periods`);c(t,p.periods.length>=this.rowCount,`output.periods must hold one row per input row`),a(t,{invalidCount:p.status?.invalidCount}),f(t,this.getOutputViews(),this.getInputViews());let h=p.periods instanceof r&&p.periods.length>this.rowCount?new r(p.periods.buffer,{...p.periods,length:this.rowCount}):p.periods;m(t,{periods:h,invalidCount:p.status?.invalidCount});let _=n.kind===`calendar`?n.step.calendarUnit:void 0;this.chunkPlan=g(t,{...n.kind===`fixed`?{timestamps:e.timestamps}:{calendarYear:e.columns.year,calendarMonth:_===`year`?void 0:e.columns.month,calendarDay:[`week`,`day`,`hour`].includes(_)?e.columns.dayOfMonth:void 0,calendarHour:_===`hour`?e.columns.hour:void 0},rowMask:e.mask,periodsOut:h}),this.estimatedCommandNodeCount=Number(!!p.status?.invalidCount)+this.chunkPlan.nonemptyChunkCount,d(t,this.estimatedCommandNodeCount)}getCommandNodes(r){let{props:i,id:a,chunkPlan:o}=this,{periods:s,output:c}=i;h(r,a,o),p(r,a,{invalidCount:c.status?.invalidCount}),e(r,a,1+Number(!!i.mask)+Number(!!c.status?.invalidCount)+(s.kind===`fixed`?1:1+Number(s.step.calendarUnit!==`year`)+Number([`week`,`day`,`hour`].includes(s.step.calendarUnit))+Number(s.step.calendarUnit===`hour`)),D);for(let e of o.chunks)e.length>0&&n(a,e.length,D,r.device.limits.maxComputeWorkgroupsPerDimension);let l=[],u=c.status?.invalidCount;u&&l.push(_(r,{id:`${a}-clear-invalid-count`,operation:E,view:u,type:`u32`,value:`0u`}));let d=(e,t,n)=>({name:e,view:t,type:n,access:`read`});for(let e of o.chunks){if(e.length===0)continue;let n=[],o,c;if(s.kind===`calendar`){let{calendarUnit:t}=s.step,r=t!==`year`,a=[`week`,`day`,`hour`].includes(t),l=t===`hour`;n.push(d(`calendarYear`,e.views.calendarYear,`i32`)),r&&n.push(d(`calendarMonth`,e.views.calendarMonth,`u32`)),a&&n.push(d(`calendarDay`,e.views.calendarDay,`u32`)),l&&n.push(d(`calendarHour`,e.views.calendarHour,`u32`));let u={year:`year`,quarter:`year * 4 + (month - 1) / 3`,month:`year * 12 + month - 1`,week:`daysFromCivil(year, month, day)`,day:`daysFromCivil(year, month, day)`,hour:``}[t],f=l?`addUint64(multiplyUint64ByUint32(calendarStepWords(daysFromCivil(year, month, day)), 24u), vec2<u32>(0u, u32(hour)))`:`calendarStepWords(${u})`,p=S(t,s.origin),[m,h]=I(BigInt(p));o=`${L}\n${b}
const ORIGIN_STEP_WORDS = vec2<u32>(${h}u, ${m}u);`,c=`let year = calendarYear[calendarYearBase + index];
  ${r?`let month = i32(calendarMonth[calendarMonthBase + index]);`:``}
  ${a?`let day = i32(calendarDay[calendarDayBase + index]);`:``}
  ${l?`let hour = i32(calendarHour[calendarHourBase + index]);`:``}
  let isTimeValid = year != bitcast<i32>(0x80000000u);
  var period = ${M}u;
  if (isTimeValid${i.mask?` && rowMask[rowMaskBase + index] != 0u`:``}) {
    let stepWords = ${f};
    period = ${N}u;
    if (compareInt64(stepWords, ORIGIN_STEP_WORDS) >= 0i) {
      let stepOffset = subtractUint64(stepWords, ORIGIN_STEP_WORDS);
      let division = divideUint64ByUint32(stepOffset, ${C(s.step)}u);
      let candidate = division.quotient;
      if (candidate.x == 0u && candidate.y < ${s.count}u) {
        period = candidate.y;
      }
    }
  }`}else{let[t,r]=I(s.origin),[a,l]=I(BigInt(s.width));n.push(d(`timestamps`,e.views.timestamps,`u32`)),o=`${y}\n${v}
const ORIGIN_WORDS = vec2<u32>(${t}u, ${r}u);
const WIDTH_WORDS = vec2<u32>(${a}u, ${l}u);`,c=`let time = vec2<u32>(timestamps[timestampsBase + 2u * index], timestamps[timestampsBase + 2u * index + 1u]);
  var period = ${M}u;
  if (${i.mask?`rowMask[rowMaskBase + index] != 0u`:`true`}) {
    period = ${N}u;
    if (timeWordsCompare(time, ORIGIN_WORDS) >= 0i) {
      // time >= origin, so the unsigned 64-bit difference is exact even beyond 2^63.
      let division = divideTimeDifference(timeWordsSubtract(time, ORIGIN_WORDS), WIDTH_WORDS);
      if (!division.overflow && division.quotient < ${s.count}u) {
        period = division.quotient;
      }
    }
  }`}i.mask&&n.push(d(`rowMask`,e.views.rowMask,`u32`)),n.push({name:`periodsOut`,view:e.views.periodsOut,type:`u32`,access:`read_write`}),u&&n.push({name:`invalidCountOut`,view:u,type:`atomic<u32>`,access:`read_write`}),l.push(t(r,{id:`${a}-periods-chunk-${e.chunkIndex}`,operation:E,variant:s.kind,bindings:n,invocationCount:e.length,workgroupSize:D,declarations:o,body:`if (index >= getChunkParameter(1u)) { return; }
  ${n.filter(e=>![`invalidCountOut`,`chunkParameters`].includes(e.name)).map(e=>`let ${e.name}Base = arrayLength(&${e.name}) - ${e.name===`timestamps`?`2u * `:``}getChunkParameter(1u);`).join(`
  `)}
  ${c}
  periodsOut[periodsOutBase + index] = period;${u?`
  if (period == ${N}u) {
    atomicAdd(&invalidCountOut[invalidCountOutOffset], 1u);
  }`:``}`}))}return l}validateCalendar(e){let{id:t}=this,{columns:n}=this.props,{calendarUnit:r,interval:i=1}=e.step;c(t,k.includes(r)&&Number.isInteger(i)&&i>=1,`periods.step needs a calendar unit and a positive integer interval`),w(t,e.step),c(t,n!==void 0,`calendar periods need columns`);let a={year:!0,month:r!==`year`,dayOfMonth:[`week`,`day`,`hour`].includes(r),hour:r===`hour`};for(let e of l(n.year))s(e,[`sint32`],`${t} columns.year`);let u=n.year.length;for(let e of[`month`,`dayOfMonth`,`hour`]){let i=n[e];if(a[e]){c(t,i!==void 0,`columns.${e} is required for a ${r} step`);for(let n of l(i))s(n,[`uint32`],`${t} columns.${e}`);o(t,`columns.${e}`,i,u)}}let{origin:d}=e;for(let[e,n,r,i]of[[`year`,d.year,-1e6,1e6],[`month`,d.month??1,1,12],[`day`,d.day??1,1,31],[`hour`,d.hour??0,0,23]])c(t,Number.isInteger(n)&&n>=r&&n<=i,`periods.origin.${e} must be an integer in ${r}..${i}`);return T(t,r,d),u}validateFixed(e){let{id:t}=this,{timestamps:n}=this.props,{width:r,origin:i}=e;c(t,n!==void 0,`fixed periods need timestamps`);for(let e of l(n))s(e,[`uint32x2`],`${t} timestamps`);return c(t,(typeof r==`bigint`||Number.isSafeInteger(r))&&BigInt(r)>=1n&&BigInt(r)<=A,`periods.width must be a positive integer below 2^63`),c(t,typeof i==`bigint`&&i>=j&&i<=A,`periods.origin must be a signed Int64 bigint`),n.length}getInputViews(){let{props:e}=this,{columns:t}=e;return[e.timestamps,e.mask,t?.year,t?.month,t?.dayOfMonth,t?.hour].filter(e=>!!e)}getOutputViews(){let{output:e}=this.props;return[e.periods,e.status?.invalidCount].filter(e=>!!e)}};function I(e){let t=BigInt.asUintN(64,e);return[Number(t&4294967295n),Number(t>>32n)]}var L=`fn daysFromCivil(year: i32, month: i32, day: i32) -> i32 {
  let adjustedYear = select(year, year - 1, month <= 2);
  let era = select(adjustedYear, adjustedYear - 399, adjustedYear < 0) / 400;
  let yearOfEra = adjustedYear - era * 400;
  let shiftedMonth = (month + 9) % 12;
  let dayOfYear = (153 * shiftedMonth + 2) / 5 + day - 1;
  let dayOfEra = yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear;
  return era * 146097 + dayOfEra - 719468;
}

// Shared uint64 arithmetic uses (high, low); sign-extend before multiplying hours or subtracting.
fn calendarStepWords(step: i32) -> vec2<u32> {
  return vec2<u32>(select(0u, 0xffffffffu, step < 0), bitcast<u32>(step));
}`;export{w as i,N as n,T as r,F as t};
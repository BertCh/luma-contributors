import{r as e}from"./geospatial-utils-DOOj6ORH.js";function t(e){let t=new Uint32Array(e.length*8),n=new Float32Array(t.buffer);return e.forEach((e,r)=>{let i=r*8;n.set(e.bounds,i),t[i+4]=e.escape,t[i+5]=e.firstPatch,t[i+6]=e.patchEnd}),t}var n=10,r=.01,i=8,a=7,o=4096,s=32,c=16,l=11,u=1e-13;function d(e){let{projection:t,bounds:n,tolerance:s=r,precision:c=`local-f32`,maxDepth:l=i,degree:u=3,sampleCount:d=a,maxPatches:f=o}=e;_({projection:t,bounds:n,tolerance:s,precision:c,maxDepth:l,degree:u,sampleCount:d,maxPatches:f});let p=v(t),m=p(H(n)),h=[];return y({project:p,destinationOrigin:m,precision:c,degree:u,tolerance:s,maxDepth:l,maxPatches:f,sampleCount:d,patches:h},n,0),{precision:c,bounds:[...n],destinationOrigin:m,patches:h,degree:u,tolerance:s,maxError:h.reduce((e,t)=>Math.max(e,t.maxError),0),float32MaxError:h.reduce((e,t)=>Math.max(e,t.float32MaxError),0),doubleSingleMaxError:h.reduce((e,t)=>Math.max(e,t.doubleSingleMaxError),0)}}function f(e){let r=e.patches.length*64,i=t(e.routingIndex??[]),a=e.strictDomains?e.patches.length*20:0,o=new Uint32Array(p(e));o.set(i,r+12+a);let s=new DataView(o.buffer);for(let t of e.patches){let i=t.id*64;if(!Number.isSafeInteger(t.id)||t.id<0||t.id>=e.patches.length)throw Error(`projection patch IDs must be contiguous zero-based indices`);s.setFloat64((i+0)*4,t.sourceOrigin[0],!0),s.setFloat64((i+2)*4,t.sourceOrigin[1],!0),s.setFloat32((i+4)*4,t.sourceScale[0],!0),s.setFloat32((i+5)*4,t.sourceScale[1],!0),s.setFloat32((i+6)*4,t.destinationOrigin[0]-e.destinationOrigin[0],!0),s.setFloat32((i+7)*4,t.destinationOrigin[1]-e.destinationOrigin[1],!0),s.setFloat32((i+8)*4,(t.bounds[0]-t.sourceOrigin[0])/t.sourceScale[0],!0),s.setFloat32((i+9)*4,(t.bounds[1]-t.sourceOrigin[1])/t.sourceScale[1],!0),s.setFloat32((i+10)*4,t.sourceOrigin[0]-Math.fround(t.sourceOrigin[0]),!0),s.setFloat32((i+11)*4,t.sourceOrigin[1]-Math.fround(t.sourceOrigin[1]),!0),o[i+12]=t.degree,s.setFloat32((i+13)*4,t.sourceOrigin[0],!0),s.setFloat32((i+14)*4,t.sourceOrigin[1],!0),o[i+15]=0,s.setFloat64((i+16)*4,t.destinationOrigin[0],!0),s.setFloat64((i+18)*4,t.destinationOrigin[1],!0);for(let e=0;e<n;e++){let n=t.doubleSingleCoefficientsX[e]??0,r=t.doubleSingleCoefficientsY[e]??0,a=Math.fround(n),o=Math.fround(r);s.setFloat32((i+20+e)*4,a,!0),s.setFloat32((i+30+e)*4,o,!0),s.setFloat32((i+40+e)*4,n-a,!0),s.setFloat32((i+50+e)*4,r-o,!0)}if(s.setFloat32((i+60)*4,t.sourceScale[0]-Math.fround(t.sourceScale[0]),!0),s.setFloat32((i+61)*4,t.sourceScale[1]-Math.fround(t.sourceScale[1]),!0),e.strictDomains){let e=r+12+t.id*20;t.bounds.forEach((t,n)=>{let r=Math.fround(t),i=Math.fround(t-r);(n<2&&r+i<t||n>=2&&r+i>t)&&(i=g(i,n<2?1:-1)),s.setFloat64((e+n*2)*4,t,!0),s.setFloat32((e+8+n*2)*4,r,!0),s.setFloat32((e+9+n*2)*4,i,!0),s.setFloat32((e+16+n)*4,n<2?m(t):h(t),!0)})}}for(let t=0;t<e.bounds.length;t++)s.setFloat64((r+t*2)*Uint32Array.BYTES_PER_ELEMENT,e.bounds[t],!0);let c=[m(e.bounds[0]),m(e.bounds[1]),h(e.bounds[2]),h(e.bounds[3])];for(let e=0;e<c.length;e++)s.setFloat32((r+8+e)*Uint32Array.BYTES_PER_ELEMENT,c[e],!0);return o}function p(e){return e.patches.length*(64+(e.strictDomains?20:0))+12+(e.routingIndex?.length??0)*8}function m(e){let t=Math.fround(e);return t>=e?t:g(t,1)}function h(e){let t=Math.fround(e);return t<=e?t:g(t,-1)}function g(e,t){let n=new Float32Array(1),r=new Uint32Array(n.buffer);return n[0]=e,e===0?r[0]=t===1?1:2147483649:e>0==(t===1)?r[0]++:r[0]--,n[0]}function _(e){let{projection:t,bounds:n,tolerance:r,precision:i,maxDepth:a,degree:o,sampleCount:l,maxPatches:u}=e;if(!(typeof t==`function`||typeof t==`object`&&t&&typeof t.project==`function`))throw Error(`projection must be a function or expose a project method`);if(!Array.isArray(n)||n.length!==4||!n.every(Number.isFinite)||n[0]>=n[2]||n[1]>=n[3])throw Error(`projection bounds must contain finite, increasing source coordinates`);if(!Number.isFinite(r)||r<=0)throw Error(`projection tolerance must be a positive finite number`);if(i!==`local-f32`&&i!==`double-single`)throw Error(`projection precision must be local-f32 or double-single`);if(!Number.isSafeInteger(a)||a<0||a>c)throw Error(`projection maxDepth must be an integer between 0 and 16`);if(o!==1&&o!==2&&o!==3)throw Error(`projection polynomial degree must be 1, 2, or 3`);if(!Number.isSafeInteger(l)||l<o+1||l>s)throw Error(`projection sampleCount must be between degree + 1 and 32`);if(!Number.isSafeInteger(u)||u<1)throw Error(`projection maxPatches must be a positive safe integer`)}function v(e){return t=>{let n=[t[0],t[1]],r=typeof e==`function`?e(n):e.project(n);if(!Array.isArray(r)||r.length<2||!Number.isFinite(r[0])||!Number.isFinite(r[1]))throw Error(`projection provider must return two finite destination coordinates`);return[r[0],r[1]]}}function y(e,t,n){let r=H(t),i=[(t[2]-t[0])/2,(t[3]-t[1])/2];if(i.some(e=>e<=0||!Number.isFinite(Math.fround(e))||Math.fround(e)===0))throw Error(`projection source bounds must have finite, nonzero float32 local scales`);let a=e.project(r),o=b(e,t,r,i,a);if(o.maxError<=e.tolerance){if(e.patches.length>=e.maxPatches)throw Error(`projection approximation exceeds the configured maximum patch count`);e.patches.push({id:e.patches.length,bounds:[...t],sourceOrigin:r,sourceScale:i,destinationOrigin:a,coefficientsX:o.coefficientsX,coefficientsY:o.coefficientsY,doubleSingleCoefficientsX:o.doubleSingleCoefficientsX,doubleSingleCoefficientsY:o.doubleSingleCoefficientsY,degree:e.degree,maxError:o.maxError,float32MaxError:o.float32MaxError,doubleSingleMaxError:o.doubleSingleMaxError});return}if(n>=e.maxDepth)throw Error(`projection approximation exceeds its tolerance at the maximum subdivision depth`);let[s,c]=r;if(s<=t[0]||s>=t[2]||c<=t[1]||c>=t[3])throw Error(`projection source bounds cannot be subdivided at binary64 precision`);y(e,[t[0],t[1],s,c],n+1),y(e,[s,t[1],t[2],c],n+1),y(e,[t[0],c,s,t[3]],n+1),y(e,[s,c,t[2],t[3]],n+1)}function b(e,t,n,r,i){let a=U(e.degree),o=new Float64Array(a*a),s=new Float64Array(a),c=new Float64Array(a),u=new Float64Array(a);for(let t=0;t<e.sampleCount;t++){let l=W(t,e.sampleCount);for(let t=0;t<e.sampleCount;t++){let d=W(t,e.sampleCount),f=[n[0]+d*r[0],n[1]+l*r[1]];k(u,e.precision===`double-single`?D(f,n,r):E(f,n,r),e.degree);let p=e.project(f),m=p[0]-i[0],h=p[1]-i[1];for(let e=0;e<a;e++){let t=u[e];s[e]+=t*m,c[e]+=t*h;for(let n=0;n<a;n++)o[e*a+n]+=t*u[n]}}}let d=C(o,s,a),f=C(o,c,a),p=Float32Array.from(d),m=Float32Array.from(f),h=0,g=0;for(let a=0;a<l;a++){let o=-1+2*a/10;for(let s=0;s<l;s++){let c=-1+2*s/10,l=[s===0?t[0]:s===10?t[2]:n[0]+c*r[0],a===0?t[1]:a===10?t[3]:n[1]+o*r[1]],u=e.project(l),_=E(l,n,r);h=Math.max(h,x(e,i,p,m,_,u)),g=Math.max(g,S(i,d,f,l,n,r,e.degree,u,`binary64`));let v=[Math.fround(l[0]),Math.fround(l[1])];if(v.every(Number.isFinite)&&V(t,v)){let t=v[0]===l[0]&&v[1]===l[1]?u:e.project(v),a=O(v,n,r);h=Math.max(h,x(e,i,p,m,a,t)),g=Math.max(g,S(i,d,f,v,n,r,e.degree,t,`float32`))}}}return{coefficientsX:p,coefficientsY:m,doubleSingleCoefficientsX:d,doubleSingleCoefficientsY:f,maxError:e.precision===`double-single`?g:h,float32MaxError:h,doubleSingleMaxError:g}}function x(e,t,n,r,i,a){let o=A(n,i,e.degree),s=A(r,i,e.degree),c=Math.fround(Math.fround(t[0]-e.destinationOrigin[0])+o),l=Math.fround(Math.fround(t[1]-e.destinationOrigin[1])+s);return Math.hypot(c-(a[0]-e.destinationOrigin[0]),l-(a[1]-e.destinationOrigin[1]))}function S(e,t,n,r,i,a,o,s,c){let l=c===`float32`?T(r,i,a):w(r,i,a),u=j(t,l,o),d=j(n,l,o),f=I(M(e[0]),u),p=I(M(e[1]),d);return Math.hypot(f[0]+f[1]-s[0],p[0]+p[1]-s[1])}function C(e,t,n){let r=e.slice(),i=t.slice();for(let e=0;e<n;e++){let t=e,a=Math.abs(r[e*n+e]);for(let i=e+1;i<n;i++){let o=Math.abs(r[i*n+e]);o>a&&(a=o,t=i)}if(!Number.isFinite(a)||a<=u)throw Error(`projection samples cannot support a stable polynomial approximation`);if(t!==e){for(let i=e;i<n;i++){let a=e*n+i,o=t*n+i;[r[a],r[o]]=[r[o],r[a]]}[i[e],i[t]]=[i[t],i[e]]}let o=r[e*n+e];for(let t=e+1;t<n;t++){let a=r[t*n+e]/o;r[t*n+e]=0;for(let i=e+1;i<n;i++)r[t*n+i]-=a*r[e*n+i];i[t]-=a*i[e]}}for(let e=n-1;e>=0;e--){let t=i[e];for(let a=e+1;a<n;a++)t-=r[e*n+a]*i[a];i[e]=t/r[e*n+e]}return i}function w(e,t,n){let r=M(e[0]-t[0]),i=M(e[1]-t[1]);return[z(r,M(n[0])),z(i,M(n[1]))]}function T(e,t,n){let r=L([Math.fround(e[0]),0],M(t[0])),i=L([Math.fround(e[1]),0],M(t[1]));return[z(r,M(n[0])),z(i,M(n[1]))]}function E(e,t,n){return[Math.fround(Math.fround(e[0]-t[0])/Math.fround(n[0])),Math.fround(Math.fround(e[1]-t[1])/Math.fround(n[1]))]}function D(e,t,n){return[(e[0]-t[0])/n[0],(e[1]-t[1])/n[1]]}function O(e,t,n){let r=Math.fround(t[0]),i=Math.fround(t[1]),a=Math.fround(t[0]-r),o=Math.fround(t[1]-i),s=Math.fround(Math.fround(e[0]-r)-a),c=Math.fround(Math.fround(e[1]-i)-o);return[Math.fround(s/Math.fround(n[0])),Math.fround(c/Math.fround(n[1]))]}function k(e,t,n){let[r,i]=t,a=0;for(let t=0;t<=n;t++)for(let n=0;n<=t;n++){let o=t-n;e[a++]=r**o*i**n}}function A(e,t,n){let[r,i]=t,a=U(n),o=t=>t<a?e[t]:0,s=B(r,B(r,o(6),o(3)),o(1)),c=B(r,o(7),o(4)),l=B(r,o(8),o(5));return B(i,B(i,B(i,o(9),l),B(r,c,o(2))),B(r,s,o(0)))}function j(e,t,n){let[r,i]=t,a=U(n),o=t=>M(t<a?e[t]:0),s=(e,t,n)=>I(n,R(e,t)),c=s(r,s(r,o(6),o(3)),o(1)),l=s(r,o(7),o(4)),u=s(r,o(8),o(5));return s(i,s(i,s(i,o(9),u),s(r,l,o(2))),s(r,c,o(0)))}function M(e){let t=Math.fround(e);return[t,Math.fround(e-t)]}function N(e,t){let n=Math.fround(e+t);return[n,Math.fround(t-Math.fround(n-e))]}function P(e,t){let n=Math.fround(e+t),r=Math.fround(n-e),i=Math.fround(e-Math.fround(n-r)),a=Math.fround(t-r);return[n,Math.fround(i+a)]}function F(e,t){let n=Math.fround(e*t);return[n,Math.fround(e*t-n)]}function I(e,t){let n=P(e[0],t[0]),r=P(e[1],t[1]);return n=[n[0],Math.fround(n[1]+r[0])],n=N(n[0],n[1]),n=[n[0],Math.fround(n[1]+r[1])],N(n[0],n[1])}function L(e,t){return I(e,[-t[0],-t[1]])}function R(e,t){let n=F(e[0],t[0]);return n=[n[0],Math.fround(n[1]+Math.fround(e[0]*t[1]))],n=N(n[0],n[1]),n=[n[0],Math.fround(n[1]+Math.fround(e[1]*t[0]))],N(n[0],n[1])}function z(e,t){let n=Math.fround(1/t[0]),r=R(e,[n,0]),i=L(e,R(t,r))[0];return I(r,F(n,i))}function B(e,t,n){return Math.fround(Math.fround(e*t)+n)}function V(e,t){return t[0]>=e[0]&&t[0]<=e[2]&&t[1]>=e[1]&&t[1]<=e[3]}function H(e){return[e[0]+(e[2]-e[0])/2,e[1]+(e[3]-e[1])/2]}function U(e){return(e+1)*(e+2)/2}function W(e,t){let n=Math.cos((2*e+1)*Math.PI/(2*t));return Math.abs(n)<2**-52?0:n}function G(e){let{doubleSingle:t,patchIdOffset:n,validityOffset:r}=e;return`
${e.inputDeclaration}
${K(e)}
@group(0) @binding(auto) var<storage, read_write> outputPositions: array<${t?`vec4f`:`vec2f`}>;
${n===void 0?``:`@group(0) @binding(auto) var<storage, read> projectionPatchIds: array<u32>;`}
${r===void 0?``:`@group(0) @binding(auto) var<storage, read_write> projectionValidity: array<u32>;`}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_id) localId: vec3u) {
  ${e.invocationIndexSource}
  if (index >= ${e.elementCount}u) { return; }
  ${n===void 0?``:`if (projectionPatchIds[${n}u + index] >= PATCH_COUNT) {
    outputPositions[${e.outputOffset}u + index] = ${t?`vec4f(0.0)`:`vec2f(0.0)`};
    ${r===void 0?``:`projectionValidity[${r}u + index] = 0u;`}
    return;
  }`}
  let result = project(${e.readPosition}, ${n===void 0?`INVALID_PATCH`:`projectionPatchIds[${n}u + index]`});
  outputPositions[${e.outputOffset}u + index] = result.position;
  ${r===void 0?``:`projectionValidity[${r}u + index] = result.valid;`}
}`}function K(t){let{precise:n,doubleSingle:r,patchCount:i,planOffset:a,doubleSingleInput:o,inputBoundsOffset:s}=t,c=n?`RawPoint`:o?`vec4f`:`vec2f`,l=t.routingNodeCount??0,u=a+i*64+12,d=u+(t.strictDomains?i*20:0),f=n?`vec2f(sub_fp64u32_to_f32(position.x, vec2u(0u)), sub_fp64u32_to_f32(position.y, vec2u(0u)))`:o?`vec2f(position.x + position.y, position.z + position.w)`:`position`,p=t.strictDomains?`
  let domainOffset = ${u}u + patchIndex * 20u;
  for (var boundIndex = 0u; boundIndex < 4u; boundIndex += 1u) {
    ${n?`let offset = domainOffset + boundIndex * 2u;
    let bound = vec2u(projectionPlans[offset + 1u], projectionPlans[offset]);
    let coordinate = select(position.x, position.y, (boundIndex & 1u) != 0u);
    let comparison = compareFiniteBinary64(coordinate, bound);`:o?`let offset = domainOffset + 8u + boundIndex * 2u;
    let bound = vec2f(bitcast<f32>(projectionPlans[offset]), bitcast<f32>(projectionPlans[offset + 1u]));
    let coordinate = select(position.xy, position.zw, (boundIndex & 1u) != 0u);
    let comparison = compare_fp64(coordinate, bound);`:`let bound = bitcast<f32>(projectionPlans[domainOffset + 16u + boundIndex]);
    let coordinate = position[boundIndex & 1u];
    let comparison = select(select(0, 1, coordinate > bound), -1, coordinate < bound);`}
    if ((boundIndex < 2u && comparison < 0) || (boundIndex >= 2u && comparison > 0)) { return false; }
  }`:``,m=o?`let offset = projectionSourceOffsetFP64(position, patchIndex);
  return vec2f(offset.x.x + offset.x.y, offset.y.x + offset.y.y);`:n?`let originX = vec2u(projectionPlanWord(patchIndex, 1u), projectionPlanWord(patchIndex, 0u));
  let originY = vec2u(projectionPlanWord(patchIndex, 3u), projectionPlanWord(patchIndex, 2u));
  return vec2f(
    sub_fp64u32_to_f32(position.x, originX),
    sub_fp64u32_to_f32(position.y, originY)
  );`:`let sourceOriginHigh = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 13u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 14u))
  );
  let sourceOriginLow = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 10u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 11u))
  );
  // Route the rounded high-limb subtraction through an opaque runtime-zero integer operation.
  // Metal otherwise reassociates these subtractions and silently drops the low origin limb.
  let highOffsetBits = bitcast<vec2u>(position - sourceOriginHigh) ^
    vec2u(projectionPlanWord(patchIndex, 15u));
  let highOffset = bitcast<vec2f>(highOffsetBits);
  return highOffset - sourceOriginLow;`,h=o?`is_finite_fp64(position.xy) && is_finite_fp64(position.zw)`:n?`rawPointIsFinite(position)`:`all((bitcast<vec2u>(position) & vec2u(0x7f800000u)) != vec2u(0x7f800000u))`,g=o?`return compare_fp64(position.xy, projectionInputBound(0u)) >= 0 &&
    compare_fp64(position.zw, projectionInputBound(1u)) >= 0 &&
    compare_fp64(position.xy, projectionInputBound(2u)) <= 0 &&
    compare_fp64(position.zw, projectionInputBound(3u)) <= 0;`:n?`let minimumX = projectionPlanRawBound(0u);
  let minimumY = projectionPlanRawBound(1u);
  let maximumX = projectionPlanRawBound(2u);
  let maximumY = projectionPlanRawBound(3u);
  return compareFiniteBinary64(position.x, minimumX) >= 0 &&
    compareFiniteBinary64(position.y, minimumY) >= 0 &&
    compareFiniteBinary64(position.x, maximumX) <= 0 &&
    compareFiniteBinary64(position.y, maximumY) <= 0;`:`let minimum = vec2f(
    bitcast<f32>(projectionPlans[PLAN_BOUNDS_OFFSET + 8u]),
    bitcast<f32>(projectionPlans[PLAN_BOUNDS_OFFSET + 9u])
  );
  let maximum = vec2f(
    bitcast<f32>(projectionPlans[PLAN_BOUNDS_OFFSET + 10u]),
    bitcast<f32>(projectionPlans[PLAN_BOUNDS_OFFSET + 11u])
  );
  return all(position >= minimum) && all(position <= maximum);`,_=n?`fn projectionPlanRawBound(boundIndex: u32) -> vec2u {
  let wordOffset = PLAN_BOUNDS_OFFSET + boundIndex * 2u;
  return vec2u(
    projectionPlans[wordOffset + 1u],
    projectionPlans[wordOffset]
  );
}

fn compareFiniteBinary64(firstBits: vec2u, secondBits: vec2u) -> i32 {
  let first = fp64_decode_bits(firstBits);
  let second = fp64_decode_bits(secondBits);
  if (first.isZero && second.isZero) {
    return 0;
  }
  if (first.sign != second.sign) {
    return select(1, -1, first.sign == 1u);
  }
  let magnitudeComparison = fp64_finite_magnitude_compare(first, second);
  return select(magnitudeComparison, -magnitudeComparison, first.sign == 1u);
}`:``,v=r?`struct ProjectionPointFP64 { x: vec2f, y: vec2f }

${n?``:`fn projectionSourceOffsetFP64(
  position: ${c},
  patchIndex: u32
) -> ProjectionPointFP64 {
  ${o?`return ProjectionPointFP64(
    sub_fp64(position.xy, vec2f(
      bitcast<f32>(projectionPlanWord(patchIndex, 13u)),
      bitcast<f32>(projectionPlanWord(patchIndex, 10u))
    )),
    sub_fp64(position.zw, vec2f(
      bitcast<f32>(projectionPlanWord(patchIndex, 14u)),
      bitcast<f32>(projectionPlanWord(patchIndex, 11u))
    ))
  );`:`let sourceOriginX = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 13u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 10u))
  );
  let sourceOriginY = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 14u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 11u))
  );
  return ProjectionPointFP64(
    sub_fp64(vec2f(position.x, 0.0), sourceOriginX),
    sub_fp64(vec2f(position.y, 0.0), sourceOriginY)
  );`}
}
`}
fn projectionCoefficientFP64(
  patchIndex: u32,
  highOffset: u32,
  lowOffset: u32,
  coefficientIndex: u32
) -> vec2f {
  return vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, highOffset + coefficientIndex)),
    bitcast<f32>(projectionPlanWord(patchIndex, lowOffset + coefficientIndex))
  );
}

fn projectionMultiplyAddFP64(multiplier: vec2f, multiplicand: vec2f, addend: vec2f) -> vec2f {
  return sum_fp64(addend, mul_fp64(multiplier, multiplicand));
}

fn evaluateProjectionPolynomialFP64(
  patchIndex: u32,
  highOffset: u32,
  lowOffset: u32,
  normalized: ProjectionPointFP64
) -> vec2f {
  let coefficient0 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 0u);
  let coefficient1 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 1u);
  let coefficient2 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 2u);
  let coefficient3 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 3u);
  let coefficient4 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 4u);
  let coefficient5 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 5u);
  let coefficient6 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 6u);
  let coefficient7 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 7u);
  let coefficient8 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 8u);
  let coefficient9 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 9u);

  let xQuadratic = projectionMultiplyAddFP64(normalized.x, coefficient6, coefficient3);
  let xLinear = projectionMultiplyAddFP64(normalized.x, xQuadratic, coefficient1);
  let mixedLinear = projectionMultiplyAddFP64(normalized.x, coefficient7, coefficient4);
  let yQuadraticX = projectionMultiplyAddFP64(normalized.x, coefficient8, coefficient5);
  let yQuadratic = projectionMultiplyAddFP64(normalized.y, coefficient9, yQuadraticX);
  let yLinearX = projectionMultiplyAddFP64(normalized.x, mixedLinear, coefficient2);
  let yLinear = projectionMultiplyAddFP64(normalized.y, yQuadratic, yLinearX);
  let xContribution = projectionMultiplyAddFP64(normalized.x, xLinear, coefficient0);
  return projectionMultiplyAddFP64(normalized.y, yLinear, xContribution);
}${n?``:`

fn projectionDestinationOriginFP64(patchIndex: u32, wordOffset: u32) -> vec2f {
  let value = vec2u(
    projectionPlanWord(patchIndex, wordOffset + 1u),
    projectionPlanWord(patchIndex, wordOffset)
  );
  return sub_fp64u32_to_fp64(value, vec2u(0u));
}`}`:``,y=r?`${n?`// Source offsets and destination origins share one raw binary64 conversion call site.
  var rawValues = array<vec2u, 4>(
    position.x,
    position.y,
    vec2u(projectionPlanWord(patchIndex, 17u), projectionPlanWord(patchIndex, 16u)),
    vec2u(projectionPlanWord(patchIndex, 19u), projectionPlanWord(patchIndex, 18u))
  );
  var rawOrigins = array<vec2u, 4>(
    vec2u(projectionPlanWord(patchIndex, 1u), projectionPlanWord(patchIndex, 0u)),
    vec2u(projectionPlanWord(patchIndex, 3u), projectionPlanWord(patchIndex, 2u)),
    vec2u(0u),
    vec2u(0u)
  );
  var converted: array<vec2f, 4>;
  for (var index = 0u; index < 4u; index += 1u) {
    converted[index] = sub_fp64u32_to_fp64(rawValues[index], rawOrigins[index]);
  }`:`let sourceOffset = projectionSourceOffsetFP64(position, patchIndex);
  var converted: array<vec2f, 4>;
  converted[0] = sourceOffset.x;
  converted[1] = sourceOffset.y;
  for (var axis = 0u; axis < 2u; axis += 1u) {
    converted[2u + axis] = projectionDestinationOriginFP64(patchIndex, 16u + 2u * axis);
  }`}
  var sourceScales = array<vec2f, 2>(
    vec2f(
      bitcast<f32>(projectionPlanWord(patchIndex, 4u)),
      bitcast<f32>(projectionPlanWord(patchIndex, 60u))
    ),
    vec2f(
      bitcast<f32>(projectionPlanWord(patchIndex, 5u)),
      bitcast<f32>(projectionPlanWord(patchIndex, 61u))
    )
  );
  var normalizedAxes: array<vec2f, 2>;
  for (var axis = 0u; axis < 2u; axis += 1u) {
    normalizedAxes[axis] = div_fp64(converted[axis], sourceScales[axis]);
  }
  let normalizedFP64 = ProjectionPointFP64(normalizedAxes[0], normalizedAxes[1]);
  var projectedAxes: array<vec2f, 2>;
  for (var axis = 0u; axis < 2u; axis += 1u) {
    projectedAxes[axis] = sum_fp64(
      converted[2u + axis],
      evaluateProjectionPolynomialFP64(patchIndex, 20u + 10u * axis, 40u + 10u * axis, normalizedFP64)
    );
  }
  let projected = vec4f(
    projectedAxes[0].x,
    projectedAxes[0].y,
    projectedAxes[1].x,
    projectedAxes[1].y
  );`:`let normalized = normalizeProjectionPosition(position, patchIndex);
  let destinationOffset = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 6u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 7u))
  );
  let projected = destinationOffset + vec2f(
    evaluateProjectionPolynomial(patchIndex, 20u, normalized),
    evaluateProjectionPolynomial(patchIndex, 30u, normalized)
  );`;return`
${n?e:``}
const PATCH_COUNT: u32 = ${i}u;
const PATCH_WORD_LENGTH: u32 = 64u;
const PLAN_OFFSET: u32 = ${a}u;
const PLAN_BOUNDS_OFFSET: u32 = PLAN_OFFSET + PATCH_COUNT * PATCH_WORD_LENGTH;
const INVALID_PATCH: u32 = 0xffffffffu;

@group(0) @binding(auto) var<storage, read> projectionPlans: array<u32>;
struct ProjectionResult { position: ${r?`vec4f`:`vec2f`}, valid: u32 }
${o?`fn projectionInputBound(boundIndex: u32) -> vec2f {
  let offset = ${s}u + boundIndex * 2u;
  return vec2f(bitcast<f32>(projectionPlans[offset]), bitcast<f32>(projectionPlans[offset + 1u]));
}`:``}

fn projectionPlanWord(patchIndex: u32, wordIndex: u32) -> u32 {
  return projectionPlans[PLAN_OFFSET + patchIndex * PATCH_WORD_LENGTH + wordIndex];
}

${_}

fn projectionPlanContains(position: ${c}) -> bool {
  ${g}
}

fn projectionSourceOffset(position: ${c}, patchIndex: u32) -> vec2f {
  ${m}
}

fn normalizeProjectionPosition(position: ${c}, patchIndex: u32) -> vec2f {
  let sourceScale = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 4u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 5u))
  );
  return projectionSourceOffset(position, patchIndex) / sourceScale;
}

fn projectionPatchContains(position: ${c}, patchIndex: u32) -> bool {
  ${p}
  let normalized = normalizeProjectionPosition(position, patchIndex);
  let minimum = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 8u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 9u))
  );
  // Source scales are half-extents, so the opposite boundary is exactly two normalized units
  // away. Two float32 ULPs keep legitimate binary64 endpoints from falling through a seam.
  let maximum = minimum + vec2f(2.0);
  let boundaryTolerance = vec2f(2.384185791015625e-7);
  return all(normalized >= minimum - boundaryTolerance) &&
    all(normalized <= maximum + boundaryTolerance);
}

// Returns the first patch containing the position: the assigned patch when one is given,
// otherwise the first match in search order. Every candidate goes through the one
// projectionPatchContains call site below, because D3D12 inlines each call site into DXIL.
fn selectProjectionPatch(position: ${c}, assignedPatch: u32) -> u32 {
  let searching = assignedPatch == INVALID_PATCH;
  ${l?`var routingPosition = vec2f(0.0);
  if (searching) { routingPosition = ${f}; }
  var nodeIndex = 0u;
  // Bound traversal even for malformed caller-owned storage. A valid traversal visits each
  // node at most once; corrupt escape links must not cause an unbounded GPU loop. An assigned
  // patch uses the first visit as a one-patch range.
  for (var visitIndex = 0u; visitIndex <= ${l}u; visitIndex += 1u) {
    var firstPatch = 0u;
    var patchEnd = 0u;
    if (!searching) {
      if (visitIndex > 0u) { break; }
      firstPatch = assignedPatch;
      patchEnd = select(assignedPatch, assignedPatch + 1u, assignedPatch < PATCH_COUNT);
    } else {
      if (visitIndex >= ${l}u || nodeIndex >= ${l}u) { break; }
      let offset = ${d}u + nodeIndex * 8u;
      let minimum = vec2f(bitcast<f32>(projectionPlans[offset]), bitcast<f32>(projectionPlans[offset + 1u]));
      let maximum = vec2f(bitcast<f32>(projectionPlans[offset + 2u]), bitcast<f32>(projectionPlans[offset + 3u]));
      if (any(routingPosition < minimum) || any(routingPosition > maximum)) {
        nodeIndex = projectionPlans[offset + 4u];
        continue;
      }
      firstPatch = projectionPlans[offset + 5u];
      let storedPatchEnd = projectionPlans[offset + 6u];
      // Reject malformed borrowed storage before scanning. Internal nodes have empty ranges;
      // valid leaves never exceed the builder's fixed capacity.
      if (firstPatch >= PATCH_COUNT || storedPatchEnd < firstPatch ||
          storedPatchEnd > PATCH_COUNT || storedPatchEnd - firstPatch > 4u) {
        return INVALID_PATCH;
      }
      patchEnd = min(storedPatchEnd, PATCH_COUNT);
      nodeIndex += 1u;
    }
    for (var patchIndex = firstPatch; patchIndex < patchEnd; patchIndex += 1u) {
      if (projectionPatchContains(position, patchIndex)) { return patchIndex; }
    }
  }`:`let firstCandidate = select(assignedPatch, 0u, searching);
  let candidateCount = select(1u, PATCH_COUNT, searching);
  for (var candidateOffset = 0u; candidateOffset < candidateCount; candidateOffset += 1u) {
    let candidate = firstCandidate + candidateOffset;
    if (candidate < PATCH_COUNT && projectionPatchContains(position, candidate)) {
      return candidate;
    }
  }`}
  return INVALID_PATCH;
}

fn findProjectionPatch(position: ${c}) -> u32 {
  return selectProjectionPatch(position, INVALID_PATCH);
}

fn evaluateProjectionPolynomial(
  patchIndex: u32,
  coefficientOffset: u32,
  normalized: vec2f
) -> f32 {
  // Lower-degree records are zero-padded, so one branch-free Horner expression handles every
  // supported degree without dynamically indexed temporary arrays or nested shader loops.
  let coefficient0 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 0u));
  let coefficient1 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 1u));
  let coefficient2 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 2u));
  let coefficient3 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 3u));
  let coefficient4 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 4u));
  let coefficient5 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 5u));
  let coefficient6 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 6u));
  let coefficient7 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 7u));
  let coefficient8 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 8u));
  let coefficient9 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 9u));

  let xQuadratic = coefficient3 + normalized.x * coefficient6;
  let xLinear = coefficient1 + normalized.x * xQuadratic;
  let mixedLinear = coefficient4 + normalized.x * coefficient7;
  let yQuadraticX = coefficient5 + normalized.x * coefficient8;
  let yQuadratic = yQuadraticX + normalized.y * coefficient9;
  let yLinearX = coefficient2 + normalized.x * mixedLinear;
  let yLinear = yLinearX + normalized.y * yQuadratic;
  let xContribution = coefficient0 + normalized.x * xLinear;
  return xContribution + normalized.y * yLinear;
}

${v}

fn project(position: ${c}, assignedPatch: u32) -> ProjectionResult {
  let invalid = ProjectionResult(${r?`vec4f(0.0)`:`vec2f(0.0)`}, 0u);
  if (!(${h}) || !projectionPlanContains(position)) { return invalid; }
  let patchIndex = selectProjectionPatch(position, assignedPatch);
  if (patchIndex == INVALID_PATCH) {
    return invalid;
  }
  ${y}
  if (!all((bitcast<${r?`vec4u`:`vec2u`}>(projected) &
    ${r?`vec4u`:`vec2u`}(0x7f800000u)) != ${r?`vec4u`:`vec2u`}(0x7f800000u))) {
    return invalid;
  }
  return ProjectionResult(projected, 1u);
}`}export{f as a,p as i,G as n,d as r,K as t};
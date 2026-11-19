import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as n,x as r}from"./gpu-contributor-utils-BIwLQON5.js";import{h as i,l as a,o}from"./raster-band-utils-BcTxc8_Y.js";import{n as s}from"./solar-position-BynngLH_.js";import{n as c,u as l}from"./terrain-illumination-utils-CoEB4UZd.js";import{t as u}from"./define-run-B-brkCe4.js";import{d,s as f}from"./pointer-overlay-Qh8qflcA.js";import{t as p}from"./coastline-shKuV2bs.js";import{r as m}from"./hydrography-N8loF-Ds.js";import{M as h}from"./entry-preview-D70TdSbK.js";function g(e,t=new Float32Array(4)){o(`getGPUSolarPositionParameterValues`,t,4);let{dayNumber:n,dayFraction:r}=s(e.timestamp);return t.set([n,r,e.daylightAltitudeDegrees??-.833,0]),t}var _=class{id;props;constructor(e){this.id=e.id??`solar-position`,this.props=e;let{id:t}=this,{output:r}=e;n(e.positions,[`float32x2`],`${t} positions`);let o=e.positions.length;if(!r.azimuth&&!r.altitude&&!r.daylight)throw Error(`${t} requires at least one output`);l(t,`azimuth`,r.azimuth,`float32`,o),l(t,`altitude`,r.altitude,`float32`,o),l(t,`daylight`,r.daylight,`uint32`,o),i(t,e.parameters,4),a(t,[r.azimuth,r.altitude,r.daylight],[e.positions,e.parameters])}getCommandNodes(n){e(n,this.id,this.props);let{id:i,props:a}=this,{output:o}=a;r(i,n,[a.positions,a.parameters,o.azimuth,o.altitude,o.daylight]);let s=[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`settings`,view:a.parameters,type:`f32`,access:`read`}];o.azimuth&&s.push({name:`azimuthValues`,view:o.azimuth,type:`f32`,access:`read_write`}),o.altitude&&s.push({name:`altitudeValues`,view:o.altitude,type:`f32`,access:`read_write`}),o.daylight&&s.push({name:`daylightValues`,view:o.daylight,type:`u32`,access:`read_write`});let l=a.refraction??!0;return[t(n,{id:`${i}-sun`,operation:`GPUSolarPosition`,variant:l?`refracted`:`geometric`,bindings:s,invocationCount:a.positions.length,declarations:`${c}
fn wrapDegrees(value: f32, divisor: f32) -> f32 { return value - divisor * floor(value / divisor); }
fn getRefractionDegrees(elevation: f32) -> f32 {
  if (elevation > 85.0) { return 0.0; }
  let tangent = tan(elevation * DEGREES_TO_RADIANS);
  var arcseconds = 0.0;
  if (elevation > 5.0) {
    arcseconds = 58.1 / tangent - 0.07 / (tangent * tangent * tangent) +
      0.000086 / (tangent * tangent * tangent * tangent * tangent);
  } else if (elevation > -0.575) {
    arcseconds = 1735.0 + elevation * (-518.2 + elevation * (103.4 + elevation * (-12.79 + elevation * 0.711)));
  } else {
    arcseconds = -20.772 / tangent;
  }
  return arcseconds / 3600.0;
}`,body:`let longitude = positions[positionsOffset + index * 2u];
  let latitude = positions[positionsOffset + index * 2u + 1u];
  let dayNumber = settings[settingsOffset];
  let dayFraction = settings[settingsOffset + 1u];
  let isValid = isFiniteValue(longitude) && isFiniteValue(latitude) && abs(latitude) <= 90.0;
  // Angles that grow by ~1 degree per day are reduced from the whole-day part first so float32
  // preserves the fractional-day contribution. The rate is 36000.76983 degrees per Julian century.
  let julianCentury = (dayNumber + dayFraction) / 36525.0;
  let meanLongitude = wrapDegrees(
    wrapDegrees(280.46646 + 0.98564736 * dayNumber, 360.0) + 0.98564736 * dayFraction +
      0.0003032 * julianCentury * julianCentury,
    360.0
  );
  let meanAnomalyDegrees = wrapDegrees(
    wrapDegrees(357.52911 + 0.98560028 * dayNumber, 360.0) + 0.98560028 * dayFraction -
      0.0001537 * julianCentury * julianCentury,
    360.0
  );
  let anomaly = meanAnomalyDegrees * DEGREES_TO_RADIANS;
  let eccentricity = 0.016708634 - julianCentury * (0.000042037 + 0.0000001267 * julianCentury);
  let equationOfCenter = sin(anomaly) * (1.914602 - julianCentury * (0.004817 + 0.000014 * julianCentury)) +
    sin(2.0 * anomaly) * (0.019993 - 0.000101 * julianCentury) + sin(3.0 * anomaly) * 0.000289;
  let omega = wrapDegrees(125.04 - 1934.136 * julianCentury, 360.0) * DEGREES_TO_RADIANS;
  let apparentLongitude =
    (meanLongitude + equationOfCenter - 0.00569 - 0.00478 * sin(omega)) * DEGREES_TO_RADIANS;
  let meanObliquity = 23.0 + (26.0 + (21.448 - julianCentury * (46.815 + julianCentury *
    (0.00059 - julianCentury * 0.001813))) / 60.0) / 60.0;
  let obliquity = (meanObliquity + 0.00256 * cos(omega)) * DEGREES_TO_RADIANS;
  let declination = asin(sin(obliquity) * sin(apparentLongitude));
  let y = tan(obliquity * 0.5) * tan(obliquity * 0.5);
  let longitudeRadians = meanLongitude * DEGREES_TO_RADIANS;
  let equationOfTime = 4.0 * RADIANS_TO_DEGREES * (y * sin(2.0 * longitudeRadians) -
    2.0 * eccentricity * sin(anomaly) +
    4.0 * eccentricity * y * sin(anomaly) * cos(2.0 * longitudeRadians) -
    0.5 * y * y * sin(4.0 * longitudeRadians) -
    1.25 * eccentricity * eccentricity * sin(2.0 * anomaly));
  let minutesUTC = fract(dayFraction + 0.5) * 1440.0;
  let trueSolarTime = wrapDegrees(minutesUTC + equationOfTime + 4.0 * longitude, 1440.0);
  let hourAngleDegrees = trueSolarTime / 4.0 - 180.0;
  let hourAngle = hourAngleDegrees * DEGREES_TO_RADIANS;
  let latitudeRadians = latitude * DEGREES_TO_RADIANS;
  let cosZenith = clamp(sin(latitudeRadians) * sin(declination) +
    cos(latitudeRadians) * cos(declination) * cos(hourAngle), -1.0, 1.0);
  let geometricAltitude = 90.0 - acos(cosZenith) * RADIANS_TO_DEGREES;
  let altitude = geometricAltitude${l?` + getRefractionDegrees(geometricAltitude)`:``};
  let azimuth = wrapDegrees(atan2(sin(hourAngle),
    cos(hourAngle) * sin(latitudeRadians) - tan(declination) * cos(latitudeRadians)) *
    RADIANS_TO_DEGREES + 180.0, 360.0);
  let invalidValue = getQuietNaN(index);
  ${o.azimuth?`azimuthValues[azimuthValuesOffset + index] = select(invalidValue, azimuth, isValid);`:``}
  ${o.altitude?`altitudeValues[altitudeValuesOffset + index] = select(invalidValue, altitude, isValid);`:``}
  ${o.daylight?`daylightValues[daylightValuesOffset + index] = select(0u, 1u, isValid && geometricAltitude > settings[settingsOffset + 2u]);`:``}`})]}},v=72,y=36,b=[`Jan`,`Feb`,`Mar`,`Apr`,`May`,`Jun`,`Jul`,`Aug`,`Sep`,`Oct`,`Nov`,`Dec`],x=[-20.1,-10.8,.1,11.6,20,23.4,20.5,12.3,.4,-10.4,-19.8,-23.4],S=[12,1,2,3,4,5,6];function C(e,t){return[(((12-t)*15+180)%360+360)%360-180,x[e-1]]}var w=u({parameters:h,caption:`Solar altitude across the world and at eighteen Natural Earth cities on the 21st of each month.`,createInputs:({world:e})=>{let t=new Float32Array(5184);for(let e=0;e<y;e++)for(let n=0;n<v;n++){let r=e*v+n;t[r*2]=-180+360*(n+.5)/v,t[r*2+1]=85-170*(e+.5)/y}let n=Object.values(m),r=Float32Array.from(n.flatMap(([t,n])=>[(t+180)*e.width/360,(n+85)*e.height/170])),i=p([e.width/360,e.height/170],[e.width/2,e.height/2]);return{places:t,cities:Float32Array.from(n.flat()),drawn:r,coastline:i}},build(e,t){let n=e.upload(`places`,`float32x2`,t.places),r=e.upload(`coastline`,`float32x4`,t.coastline),i=e.upload(`drawn`,`float32x2`,t.drawn),a=e.allocate(`parameters`,`float32`,4),o=e.allocate(`altitude`,`float32`,2592),s=e.allocate(`azimuth`,`float32`,2592),c=e.allocate(`daylight`,`uint32`,2592);e.graph.add(new _({positions:n.view,refraction:!0,parameters:a.view,output:{altitude:o.view,azimuth:s.view,daylight:c.view}}));let l=e.allocate(`city-altitude`,`float32`,t.cities.length/2);return e.graph.add(new _({id:`city-solar-position`,positions:e.upload(`cities`,`float32x2`,t.cities).view,refraction:!0,parameters:a.view,output:{altitude:l.view}})),{outputs:{drawn:i,altitude:o,cityAltitude:l,coastline:r},update:e=>a.write(g({timestamp:new Date(Date.UTC(2026,e.month-1,21,Math.floor(e.hour),e.hour%1*60))}))}},interaction:{kind:`drag`,hint:`place the sun overhead`,rest:({world:e})=>{let[t,n]=C(f(h,`month`).default,f(h,`hour`).default);return[(t+180)/360,1-(n+85)/170].map(e=>e)},resolve(e,{world:t}){let n=f(h,`hour`),r=f(h,`month`),i=e.x/t.width*360-180,a=e.y/t.height*170-85,o=d(n,((12-i/15)%24+24)%24),s=S[0];for(let e of S)Math.abs(x[e-1]-a)<Math.abs(x[s-1]-a)&&(s=e);s=d(r,s);let[c,l]=C(s,o),u=[(c+180)*t.width/360,(l+85)*t.height/170],p=Math.floor(o),m=o%1==0?`00`:`30`;return{values:{hour:o,month:s},glyphs:[{shape:`ring`,at:u,radius:t.height*.06},{shape:`dot`,at:u}],readout:`21 ${b[s-1]} ${String(p).padStart(2,`0`)}:${m} UTC · overhead ${Math.abs(Math.round(l))}°${l>=0?`N`:`S`}`}}},draw:({outputs:e,theme:t})=>[{layer:`raster`,values:e.altitude,columns:v,rows:y,ramp:`diverging`,range:[-90,90],smoothing:!0},{layer:`lines`,segments:e.coastline,radius:.8,color:[...t.ink,.65]},{layer:`dots`,positions:e.drawn,radius:5,color:[...t.ink,.85]},{layer:`dots`,positions:e.drawn,values:e.cityAltitude,ramp:`diverging`,radius:3.5,range:[-90,90]}]});export{w as default};
var e=`
const POINT_DENSITY_SQRT3: f32 = 1.7320508075688772;

// Odd-r offset [column, row] of the pointy-top hexagon containing (x, y).
fn getPointDensityHexagonCell(x: f32, y: f32, originX: f32, originY: f32, radius: f32) -> vec2<i32> {
  let localX = (x - originX) / radius;
  let localY = (y - originY) / radius;
  let q = (POINT_DENSITY_SQRT3 / 3.0) * localX - localY / 3.0;
  let r = (2.0 / 3.0) * localY;
  let s = -q - r;
  var roundedQ = round(q);
  var roundedR = round(r);
  let roundedS = round(s);
  let deltaQ = abs(roundedQ - q);
  let deltaR = abs(roundedR - r);
  let deltaS = abs(roundedS - s);
  if (deltaQ > deltaR && deltaQ > deltaS) {
    roundedQ = -roundedR - roundedS;
  } else if (deltaR > deltaS) {
    roundedR = -roundedQ - roundedS;
  }
  let axialQ = i32(roundedQ);
  let axialR = i32(roundedR);
  return vec2<i32>(axialQ + (axialR - (axialR & 1)) / 2, axialR);
}

// Center of odd-r hexagon (column, row).
fn getPointDensityHexagonCenter(column: i32, row: i32, originX: f32, originY: f32, radius: f32) -> vec2<f32> {
  return vec2<f32>(
    originX + radius * POINT_DENSITY_SQRT3 * (f32(column) + 0.5 * f32(row & 1)),
    originY + 1.5 * radius * f32(row)
  );
}
`;export{e as t};
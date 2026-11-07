// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// Direct CGAL 6.0.1 oracle; compiled only in the isolated oracle environment.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/create_straight_skeleton_from_polygon_with_holes_2.h>
#include <CGAL/create_offset_polygons_2.h>
#include <iomanip>
#include <iostream>
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Polygon = CGAL::Polygon_2<Kernel>;
int main() {
  unsigned ringCount; double distance;
  std::cin >> ringCount >> distance;
  CGAL::Polygon_with_holes_2<Kernel> geometry;
  for (unsigned ring = 0; ring < ringCount; ring++) {
    unsigned count; std::cin >> count; Polygon polygon;
    for (unsigned vertex = 0; vertex < count; vertex++) {
      double x, y; std::cin >> x >> y; polygon.push_back(Kernel::Point_2(x, y));
    }
    if (ring == 0) geometry.outer_boundary() = polygon; else geometry.add_hole(polygon);
  }
  auto skeleton = CGAL::create_interior_straight_skeleton_2(geometry);
  if (!skeleton) return 2;
  std::cout << std::setprecision(17) << "{\"arcs\":[";
  bool first = true;
  for (auto edge = skeleton->halfedges_begin(); edge != skeleton->halfedges_end(); ++edge) {
    if (!edge->is_bisector() || edge->id() > edge->opposite()->id()) continue;
    if (!first) std::cout << ','; first = false;
    auto start = edge->opposite()->vertex(); auto end = edge->vertex();
    std::cout << '[' << start->point().x() << ',' << start->point().y() << ',' << end->point().x() << ',' << end->point().y() << ',' << start->time() << ',' << end->time() << ']';
  }
  std::cout << "],\"offsets\":["; first = true;
  auto offsets = CGAL::create_offset_polygons_2<Polygon>(distance, *skeleton);
  for (const auto& polygon : offsets) {
    if (!first) std::cout << ','; first = false; std::cout << '[';
    for (unsigned vertex = 0; vertex < polygon->size(); vertex++) {
      if (vertex) std::cout << ',';
      std::cout << '[' << (*polygon)[vertex].x() << ',' << (*polygon)[vertex].y() << ']';
    }
    std::cout << ']';
  }
  std::cout << "]}\n";
}

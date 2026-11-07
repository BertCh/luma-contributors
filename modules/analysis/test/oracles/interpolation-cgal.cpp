// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// Offline reference generator only; CGAL remains an external GPL/commercial dependency.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
int main() {
  using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
  using Point = Kernel::Point_2;
  CGAL::Delaunay_triangulation_2<Kernel> triangulation;
  std::map<Point, double, Kernel::Less_xy_2> values;
  std::map<Point, unsigned, Kernel::Less_xy_2> indices;
  unsigned sampleCount, queryCount; std::cin >> sampleCount >> queryCount;
  for (unsigned index = 0; index < sampleCount; ++index) {
    double horizontal, vertical, value; std::cin >> horizontal >> vertical >> value;
    Point point(horizontal, vertical); triangulation.insert(point); values[point] = value; indices[point] = index;
  }
  std::cout << std::setprecision(17) << "[";
  for (unsigned query = 0; query < queryCount; ++query) {
    double horizontal, vertical; std::cin >> horizontal >> vertical; Point point(horizontal, vertical);
    std::vector<std::pair<Point, double>> coordinates;
    auto result = CGAL::natural_neighbor_coordinates_2(triangulation, point, std::back_inserter(coordinates));
    if (query) std::cout << ",";
    if (!result.third) { std::cout << "null"; continue; }
    double prediction = CGAL::linear_interpolation(coordinates.begin(), coordinates.end(), result.second, CGAL::Data_access<decltype(values)>(values));
    std::cout << "{\"prediction\":" << prediction << ",\"weights\":[";
    bool first = true;
    for (const auto &coordinate : coordinates) {
      if (!first) std::cout << ","; first = false;
      std::cout << "[" << indices[coordinate.first] << "," << coordinate.second / result.second << "]";
    }
    std::cout << "]}";
  }
  std::cout << "]\n";
}

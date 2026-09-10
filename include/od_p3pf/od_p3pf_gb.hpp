#pragma once

#include "od_p3pf/od_p3pf.hpp"

namespace od_p3pf {

// Same polynomial problem as solve(), evaluated by the 16x22
// Groebner/action-matrix template generated with the public automatic solver
// generator of Kukelova, Bujnak, and Pajdla (ECCV 2008).  This implementation
// is an experimental baseline, not the recommended production solver.
std::vector<Solution> solve_gb(const std::array<Vec3, 3>& world,
                               const std::array<Vec2, 3>& pixels,
                               double known_z);

}  // namespace od_p3pf

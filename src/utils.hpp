//
// Created by xetql on 7/28/20.
//

#ifndef YALBB_EXAMPLE_UTILS_HPP
#define YALBB_EXAMPLE_UTILS_HPP

#include "spatial_elements.hpp"

template<int N, class T>
std::array<T, N> midpoint(const std::array<T, N>& a, const std::array<T, N>& b){
    std::array<T, N> c;
    for(int i = 0; i < N; ++i) c.at(i) = (a.at(i) + b.at(i)) / 2.0;
    return c;
}

template<int N>
elements::Element<N> midpoint(const elements::Element<N>& a, const elements::Element<N>& b) {
    return elements::Element<N>(midpoint(a.position, b.position), midpoint(a.velocity, b.velocity), 0, 0);
}
#endif //YALBB_EXAMPLE_UTILS_HPP

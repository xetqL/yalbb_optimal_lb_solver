//
// Created by xetql on 7/28/20.
//

#ifndef YALBB_EXAMPLE_UTILS_HPP
#define YALBB_EXAMPLE_UTILS_HPP

#include "spatial_elements.hpp"

template<typename T>
constexpr auto convert(T&& t) {
    if constexpr (std::is_same<std::remove_cv_t<std::remove_reference_t<T>>, std::string>::value) {
        return std::forward<T>(t).c_str();
    } else {
        return std::forward<T>(t);
    }
}

/**
 * printf like formatting for C++ with std::string
 * Original source: https://stackoverflow.com/a/26221725/11722
 */
template<typename ... Args>
std::string stringFormatInternal(const std::string& format, Args&& ... args)
{
    size_t size = snprintf(nullptr, 0, format.c_str(), std::forward<Args>(args) ...) + 1;
    if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1);
}

template<typename ... Args>
std::string fmt(std::string fmt, Args&& ... args) {
    return stringFormatInternal(fmt, convert(std::forward<Args>(args))...);
}

template<int N, class T>
std::array<T, N> midpoint(const std::array<T, N>& a, const std::array<T, N>& b){
    std::array<T, N> c;
    for(int i = 0; i < N; ++i) c.at(i) = (a.at(i) + b.at(i)) / 2.0;
    return c;
}

template<int N>
elements::Element<N> midpoint(const elements::Element<N>& a, const elements::Element<N>& b) {
    return elements::Element<N>(midpoint<N>(a.position, b.position), midpoint<N>(a.velocity, b.velocity), 0, 0);
}
#endif //YALBB_EXAMPLE_UTILS_HPP

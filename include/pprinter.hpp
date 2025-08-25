// pretty printer

#pragma once

#include <sstream>

#include "spd.hpp"


template <typename T>
std::string to_string(spd<T> A) {
    std::stringstream stream; 

    const int n = A.size();

	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < n; ++j) {
			stream << A[i, j] << "  ";
		}
		stream << "\n";
	}

    return stream.str();
}

template <typename T>
std::string to_string_sparsity_pattern(spd<T> A) {
    std::stringstream stream; 

    const int n = A.size();

	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < n; ++j) {
            if (A[i, j] != 0)
                stream << "*  ";
            else
                stream << "   ";
		}
		stream << "\n";
	}

    return stream.str();
}

std::string to_string(elimination_tree parent) {
    std::stringstream stream;

    stream << "[";

    for (int v : parent) {
        stream << v << ", ";
    }

    stream << "]\n";

    return stream.str();
}


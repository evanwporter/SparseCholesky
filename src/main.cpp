#include <iostream>
#include <vector>

#include "spd.hpp"
#include "pprinter.hpp"

int main() {

	std::vector<std::vector<int>> pattern = {
		{0}, // row 0
		{1}, // row 1
		{0, 2}, // row 2
		{3}, // row 3
		{0, 2, 4}, // row 4
		{0, 1, 3, 5}, // row 5
		{0, 2, 5, 6} // row 6
	};

    spd<double> A = build_spd_from_pattern<double>(pattern);

	// Print matrix in dense format using operator[]
	std::cout << "Matrix A:\n";
	std::cout << to_string<double>(A) << std::endl;

    auto parent = etree(A);
    std::cout << "\nElimination tree:\n" << to_string(parent) << std::endl;

    return 0;
}

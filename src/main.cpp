#include <iostream>
#include <vector>

#include "spd.hpp"

int main() {
	std::vector<int> ti = {0, 1, 1, 2, 2}; // Row indices
	std::vector<int> tj = {0, 0, 1, 1, 2}; // Col indices
	std::vector<double> tx = {4.0, 1.0, 3.0, 2.0, 5.0}; // Values

	int n = 3; // Matrix size

	// Convert triplet to SPD format
	spd<double> A = triplet_to_spd(ti, tj, tx, n);

	// Print matrix in dense format using operator[]
	std::cout << "Matrix A (dense form):\n";
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < n; ++j) {
			std::cout << A[i, j] << " ";
		}
		std::cout << "\n";
	}

	return 0;
}

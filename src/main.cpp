#include <iostream>
#include <vector>

#include "chol.hpp"
#include "pprinter.hpp"

int main() {

    {
        std::vector<std::vector<int>> pattern = {
            { 0 }, // row 0
            { 1 }, // row 1
            { 0, 2 }, // row 2
            { 3 }, // row 3
            { 0, 2, 4 }, // row 4
            { 0, 1, 3, 5 }, // row 5
            { 0, 2, 5, 6 } // row 6
        };

        csc_matrix<double, sym::upper> A = build_csc_matrix_from_pattern<double>(pattern);

        int n = static_cast<int>(A.size());

        // Print matrix in dense format using operator[]
        std::cout << "Matrix A:\n";
        std::cout << to_string<double>(A) << std::endl;

        auto parent = etree(A);
        std::cout << "\nElimination tree:\n"
                  << to_string(parent) << std::endl;

        auto post = post_order(parent);
        std::cout << "\nPost Order tree:\n"
                  << to_string(parent) << std::endl;

        std::cout << "Column Count:\n"
                  << to_string(col_count(A, parent, post)) << std::endl;

        std::cout << to_string(random_sparse<double>(10)) << std::endl;

        std::vector<int> w(n, -1);
        std::vector<int> s(n);
        std::vector<double> x(n);

        int top = ereach(A, 1, parent, s, w, x, n);

        // The pattern of column 1 of L is in s[top..n-1]
        std::cout << "Reach of Column 1:\n";
        for (int idx = top; idx < n; ++idx) {
            std::cout << s[idx] << " ";
        }
        std::cout << std::endl
                  << std::endl;
    }

    {
        std::vector<int> ti = { 0, 0, 0, 1, 1, 2 };
        std::vector<int> tj = { 0, 1, 2, 1, 2, 2 };
        std::vector<double> tx = { 4.0, 1.0, 1.0, 3.0, 0.0, 2.0 };

        const int n = 3;

        auto A = triplet_to_csc_matrix(ti, tj, tx, n);

        std::cout << to_string(A) << std::endl;

        auto parent = etree(A);

        auto S = schol(A);
        auto L_err = chol(A, S); // numeric

        if (!L_err)
            std::cout
                << L_err.error() << std::endl;
        else
            std::cout << "L factor:\n"
                      << to_string(L_err.value()) << "\n";
    }

    {
        std::vector<int> ti = {
            0, 1, 2, 1, 3, 2, 3, 3, 4, 4
        }; // row indices

        std::vector<int> tj = {
            0, 0, 0, 1, 1, 2, 2, 3, 3, 4
        }; // col indices

        std::vector<double> tx = {
            5, 1, 1, 4, 1, 4, 1, 5, 1, 3
        }; // values

        const int n = 5;

        auto A = triplet_to_csc_matrix(ti, tj, tx, n);

        std::cout << to_string(A) << std::endl;

        auto parent = etree(A);

        auto S = schol(A);
        auto L_err = chol(A, S); // numeric

        if (!L_err)
            std::cout
                << L_err.error() << std::endl;
        else
            std::cout << "L factor:\n"
                      << to_string(L_err.value()) << "\n";
    }

    return 0;
}

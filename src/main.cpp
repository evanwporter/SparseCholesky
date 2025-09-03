#include <iostream>
#include <vector>

#include "chol.hpp"
#include "mtx_reader.hpp"
#include "pprinter.hpp"

int main() {

    // {
    //     std::vector<std::vector<int>> pattern = {
    //         { 0 }, // row 0
    //         { 1 }, // row 1
    //         { 0, 2 }, // row 2
    //         { 3 }, // row 3
    //         { 0, 2, 4 }, // row 4
    //         { 0, 1, 3, 5 }, // row 5
    //         { 0, 2, 5, 6 } // row 6
    //     };

    //     csc_matrix<double, sym::upper> A = build_csc_matrix_from_pattern<double>(pattern);

    //     const auto n = A.size();

    //     // Print matrix in dense format using operator[]
    //     std::cout << "Matrix A:\n";
    //     std::cout << A << std::endl;

    //     auto parent = etree(A);
    //     std::cout << "\nElimination tree:\n"
    //               << to_string(parent) << std::endl;

    //     auto post = post_order(parent);
    //     std::cout << "\nPost Order tree:\n"
    //               << to_string(parent) << std::endl;

    //     std::cout << "Column Count:\n"
    //               << to_string(col_count(A, parent, post)) << std::endl;

    //     std::cout << random_sparse<double>(10) << std::endl;

    //     std::vector<int> w(n, -1);
    //     std::vector<int> s(n);
    //     std::vector<double> x(n);

    //     const auto top = ereach(A, 5, parent, s, w, x, n);

    //     // The pattern of column 1 of L is in s[top..n-1]
    //     std::cout << "Reach of Column 5:\n";

    //     for (auto t = top; t < n; t++) {
    //         std::cout << s[t] << " ";
    //     }

    //     auto Schol = schol(A);

    //     std::cout << Schol << std::endl;

    //     std::cout << to_string(col_count(A, parent, post)) << std::endl;

    //     std::cout
    //         << std::endl
    //         << std::endl;
    // }

    // {
    //     std::vector<int> ti = { 0, 0, 0, 1, 1, 2 };
    //     std::vector<int> tj = { 0, 1, 2, 1, 2, 2 };
    //     std::vector<double> tx = { 4.0, 1.0, 1.0, 3.0, 0.0, 2.0 };

    //     const int n = 3;

    //     auto A = triplet_to_csc_matrix(ti, tj, tx, n);

    //     std::cout << A << std::endl;

    //     auto parent = etree(A);

    //     auto L_err = chol(A); // numeric

    //     if (!L_err)
    //         std::cout
    //             << L_err.error() << std::endl;
    //     else
    //         std::cout << "L factor:\n"
    //                   << L_err.value() << "\n";
    // }

    // {
    //     std::vector<int> ti = {
    //         0, 1, 2, 1, 3, 2, 3, 3, 4, 4
    //     }; // row indices

    //     std::vector<int> tj = {
    //         0, 0, 0, 1, 1, 2, 2, 3, 3, 4
    //     }; // col indices

    //     std::vector<double> tx = {
    //         5, 1, 1, 4, 1, 4, 1, 5, 1, 3
    //     }; // values

    //     const int n = 5;

    //     auto A = triplet_to_csc_matrix(ti, tj, tx, n);

    //     std::cout << A << std::endl;

    //     auto parent = etree(A);

    //     auto L_err = chol(A); // numeric

    //     if (!L_err)
    //         std::cout
    //             << L_err.error() << std::endl;
    //     else
    //         std::cout << "L factor:\n"
    //                   << L_err.value() << "\n";
    // }

    // {
    //     int n = 2;
    //     int lda = n;
    //     int info;
    //     char uplo = 'L';

    //     double A[4] = {
    //         4.0, 12.0, // column 1
    //         12.0,
    //         37.0 // column 2
    //     };

    //     dpotrf_(&uplo, &n, A, &lda, &info);

    //     if (info == 0) {
    //         std::cout << "Cholesky factor L (stored in lower triangle of A):\n";
    //         for (int i = 0; i < n; i++) {
    //             for (int j = 0; j < n; j++) {
    //                 if (j <= i)
    //                     std::cout << std::setw(8) << A[i + j * lda] << " ";
    //                 else
    //                     std::cout << std::setw(8) << 0.0 << " ";
    //             }
    //             std::cout << "\n";
    //         }
    //         std::cout << "\n";
    //     }
    // }

    // {
    //     // Build a tiny 4x4 SPD sparse matrix
    //     // A = [ 4  1  0  0
    //     //       1  3  1  0
    //     //       0  1  3  1
    //     //       0  0  1  2 ]
    //     std::vector<int> ti = { 0, 0, 1, 1, 2, 2, 3 }; // row indices
    //     std::vector<int> tj = { 0, 1, 1, 2, 2, 3, 3 }; // col indices
    //     std::vector<double> tx = { 4, 1, 7, 1, 7, 1, 6 }; // values
    //     int n = 4;

    //     auto A = triplet_to_csc_matrix(ti, tj, tx, n);

    //     std::cout << A << "\n";

    //     // Symbolic analysis
    //     auto S = schol(A);

    //     // Allocate L with symbolic size
    //     csc_matrix<double, sym::none> L(S);

    //     // Supernode partition
    //     std::vector<std::size_t> supernodes;
    //     auto sn_id = compute_supernodes(S, supernodes);

    //     std::cout << "Supernodes: ";
    //     for (auto s : supernodes)
    //         std::cout << s << " ";
    //     std::cout << "\n\n";

    //     // Pick first supernode [0..end)
    //     for (int s = 0; s + 1 < (int)supernodes.size(); ++s) {
    //         const auto start = supernodes[s];
    //         const auto end = supernodes[s + 1];

    //         // Compute rows spanned by this supernode
    //         const auto rows = supernode_rows(A, S.parent, start, end);

    //         // std::cout << "Supernode Rows: ";
    //         // for (int r : rows)
    //         //     std::cout << r << " ";
    //         // std::cout << "\n";

    //         // Extract dense panel
    //         panel<double> P = extract_panel(A.transpose(), start, end, rows);

    //         std::cout << "The panel for supernode " << s << " is:\n"
    //                   << P << std::endl;

    //         std::cout << "Preallocated L:\n"
    //                   << S << std::endl;

    //         // Factorize this supernode
    //         auto result = factorize_sn(start, end, P, S, L);
    //         if (!result) {
    //             std::cerr << "Factorization failed: " << result.error() << "\n";
    //             return 1;
    //         }
    //         auto upd = result.value();

    //         std::cout << "Matrix L now:\n"
    //                   << L << std::endl;

    //         // Print results
    //         std::cout
    //             << "Supernode " << s << " [" << start << "," << end << "):\n";
    //         std::cout << "  Rows: ";
    //         for (int r : rows)
    //             std::cout << r << " ";
    //         std::cout << "\n";

    //         std::cout << "  Update block size = " << upd.rows.size() << "x" << upd.rows.size() << "\n";
    //         const auto mb = upd.rows.size();
    //         for (int r = 0; r < mb; ++r) {
    //             for (int c = 0; c < mb; ++c) {
    //                 double val;
    //                 if (c <= r) {
    //                     // take value from lower triangle
    //                     val = upd.C[r + c * upd.ld];
    //                 } else {
    //                     // mirror upper triangle
    //                     val = upd.C[c + r * upd.ld];
    //                 }
    //                 std::cout << std::setw(10) << std::fixed << std::setprecision(4) << val << " ";
    //             }
    //             std::cout << "\n";
    //         }
    //         std::cout << std::endl;

    //         apply_update(A, upd);

    //         std::cout << "After the update for Supernode " << s << ", A now looks like:" << std::endl
    //                   << A << std::endl;
    //     }

    //     std::cout << "Matrix L now:\n"
    //               << L << std::endl;
    // }

    // {
    //     // Build a tiny 4x4 SPD sparse matrix
    //     // A = [ 4  1  0  0
    //     //       1  3  1  0
    //     //       0  1  3  1
    //     //       0  0  1  2 ]
    //     std::vector<int> ti = { 0, 0, 1, 1, 2, 2, 3 }; // row indices
    //     std::vector<int> tj = { 0, 1, 1, 2, 2, 3, 3 }; // col indices
    //     std::vector<double> tx = { 4, 1, 7, 1, 7, 1, 6 }; // values
    //     int n = 4;

    //     auto A = triplet_to_csc_matrix(ti, tj, tx, n);

    //     auto L_err = chol_sn(A);

    //     if (L_err.has_value())
    //         std::cout << "Cholesky L:\n"
    //                   << L_err.value() << std::endl;

    //     else
    //         std::cout << "Factorization failed" << L_err.error() << std::endl;
    // }

    // {
    //     std::vector<std::vector<int>> pattern = {
    //         { 0 }, // row 0
    //         { 0, 1 }, // row 1
    //         { 2 }, // row 2
    //         { 2, 3 }, // row 3
    //         { 4 }, // row 4
    //         { 4, 5 }, // row 5
    //         { 3, 4, 6 }, // row 6
    //         { 0, 1, 2, 3, 5, 7 }, // row 7
    //         { 0, 1, 5, 6, 8 } // row 8
    //     };

    //     csc_matrix<double, sym::upper> A = build_csc_matrix_from_pattern<double>(pattern);

    //     std::cout << A << std::endl;

    //     A.transpose();

    //     std::cout << to_string(col_count(A, etree(A), post_order(etree(A)))) << std::endl;

    //     const auto S = schol(A);

    //     std::cout << "schol the symbolic cholesky:\n"
    //               << S << std::endl;

    //     const auto n = A.size();

    //     std::vector<int> w(n, -1);
    //     std::vector<int> s(n);
    //     std::vector<double> x(n);

    //     const auto top = ereach(A, 2, etree(A), s, w, x, n);

    //     // The pattern of column 1 of L is in s[top..n-1]
    //     std::cout << "Reach of Column 2:\n";

    //     for (auto idx = top; idx < n; ++idx) {
    //         std::cout << s[idx] << " ";
    //     }

    //     std::cout << "\n\n";

    //     std::vector<std::size_t> supernodes;

    //     auto sn_id = compute_supernodes(S, supernodes);

    //     std::cout << "supernode id " << to_string(sn_id) << std::endl
    //               << "supernode " << to_string(supernodes) << std::endl;

    //     std::cout << "atree: " << to_string(atree(S, sn_id, supernodes)) << std::endl;

    //     auto B = random_sparse<double>(S).transpose();

    //     std::cout << "generated random matrix\n"
    //               << B << std::endl;

    //     std::cout << "schol the symbolic cholesky:\n"
    //               << S << std::endl;

    //     const auto L_err = chol_sn(B);

    //     if (L_err.has_value()) {
    //         std::cout << "chol_sn:\n"
    //                   << L_err.value() << std::endl;
    //     }

    //     else {
    //         std::cout << L_err.error() << std::endl;
    //     }
    // }

    {
        auto A = load_matrix_market_to_csc<double>("data/bcsstk01/bcsstk01.mtx");

        // std::cout << "Loaded Matrix\n"
        //           << A << std::endl;

        auto L_err = chol_sn(A);
    }

    return 0;
}

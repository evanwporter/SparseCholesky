#include <gtest/gtest.h>

#include "chol.hpp"

extern "C" {
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
}

// Pattern from page 26 of my thesis
TEST(CholeskyTest, EliminationTree) {
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

    auto parent = etree(A);

    const std::vector<int> expected = { 2, 5, 4, 5, 5, 6, -1 };

    ASSERT_EQ(parent.size(), A.size());
    EXPECT_EQ(parent, expected);
}

TEST(CholeskyTest, ColumnReach) {
    std::vector<std::vector<int>> pattern = {
        { 0 }, // row 0
        { 1 }, // row 1
        { 0, 2 }, // row 2
        { 3 }, // row 3
        { 0, 2, 4 }, // row 4
        { 0, 1, 3, 5 }, // row 5
        { 0, 2, 5, 6 } // row 6
    };

    const std::vector<int> expected = { 3, 1, 0, 2, 4, 5, 6 };

    csc_matrix<double, sym::upper> A = build_csc_matrix_from_pattern<double>(pattern);

    const auto n = A.size();

    std::vector<int> w(n, -1);
    std::vector<int> s(n);
    std::vector<double> x(n);

    auto parent = etree(A);

    auto _ = ereach(A, 5, parent, s, w, x, n);

    EXPECT_EQ(s, expected);

    _ = ereach(A, 5, parent, s, w, n);

    EXPECT_EQ(s, expected);
}

TEST(CholeskyTest, SimplicialCholesky) {
    // clang-format off
    double expected[9] = {
        4.0, 1.0, 1.0, // col 0
        1.0, 3.0, 0.0, // col 1
        1.0, 0.0, 2.0  // col 2
    };
    // clang-format on

    int n = 3;
    int lda = n;
    int info;
    char uplo = 'L';

    dpotrf_(&uplo, &n, expected, &lda, &info);

    ASSERT_TRUE(info == 0);

    std::vector<int> ti = { 0, 0, 0, 1, 1, 2 };
    std::vector<int> tj = { 0, 1, 2, 1, 2, 2 };
    std::vector<double> tx = { 4.0, 1.0, 1.0, 3.0, 0.0, 2.0 };

    auto A = triplet_to_csc_matrix(ti, tj, tx, n);

    auto L = chol(A);

    ASSERT_TRUE(L.has_value());

    auto result = csc_to_dense(*L);

    // we only compare the lower triangle since the upper triangle of `expected`
    // is filled with old values from `A`
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) { // only lower triangle
            EXPECT_NEAR(result[i + j * n], expected[i + j * n], 1e-9)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST(CholeskyTest, SupernodalCholesky) {
    // clang-format off
    double expected[9] = {
        4.0, 1.0, 1.0, // col 0
        1.0, 3.0, 0.0, // col 1
        1.0, 0.0, 2.0  // col 2
    };
    // clang-format on

    int n = 3;
    int lda = n;
    int info;
    char uplo = 'L';

    dpotrf_(&uplo, &n, expected, &lda, &info);

    ASSERT_TRUE(info == 0);

    std::vector<int> ti = { 0, 0, 0, 1, 1, 2 };
    std::vector<int> tj = { 0, 1, 2, 1, 2, 2 };
    std::vector<double> tx = { 4.0, 1.0, 1.0, 3.0, 0.0, 2.0 };

    auto A = triplet_to_csc_matrix(ti, tj, tx, n);

    auto L = chol_sn(A);

    ASSERT_TRUE(L.has_value());

    auto result = csc_to_dense(*L);

    // we only compare the lower triangle since the upper triangle of `expected`
    // is filled with old values from `A`
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) { // only lower triangle
            EXPECT_NEAR(result[i + j * n], expected[i + j * n], 1e-9)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}
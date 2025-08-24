#include <gtest/gtest.h>

#include "spd.hpp"

TEST(SpdMatrixTest, Construction) {
	spd<double> A(4, 6);
	EXPECT_EQ(A.size(), 4);
	EXPECT_EQ(A.capacity(), 6);
	EXPECT_EQ(A.p().size(), 5); // n + 1
	EXPECT_EQ(A.j().size(), 6);
	EXPECT_EQ(A.x().size(), 6);
}

TEST(SpdMatrixTest, EmptyAccess) {
	spd<double> A(3, 3);
	EXPECT_DOUBLE_EQ(A[0, 0], 0.0);
	EXPECT_DOUBLE_EQ(A[1, 2], 0.0);
}

TEST(SpdMatrixTest, TripletConversionBasic) {
	std::vector<int> ti = {0, 1, 2};
	std::vector<int> tj = {0, 1, 2};
	std::vector<double> tx = {10.0, 20.0, 30.0};

	spd<double> A = triplet_to_spd(ti, tj, tx, 3);

	EXPECT_DOUBLE_EQ(A[0, 0], 10.0);
	EXPECT_DOUBLE_EQ(A[1, 1], 20.0);
	EXPECT_DOUBLE_EQ(A[2, 2], 30.0);
	EXPECT_DOUBLE_EQ(A[0, 2], 0.0); // not stored, not in triplet
}

TEST(SpdMatrixTest, SymmetricStorageEnforcement) {
	std::vector<int> ti = {0, 2}; // implies upper triangle
	std::vector<int> tj = {2, 0};
	std::vector<double> tx = {5.0, 7.0};

	spd<double> A = triplet_to_spd(ti, tj, tx, 3);

	// Entries should be merged into (2,0)
	EXPECT_DOUBLE_EQ(A[2, 0], 12.0);
	EXPECT_DOUBLE_EQ(A[0, 2], 12.0); // symmetric access
}

TEST(SpdMatrixTest, DuplicateEntryMerge) {
	std::vector<int> ti = {1, 1, 1};
	std::vector<int> tj = {0, 0, 0};
	std::vector<double> tx = {1.0, 2.0, 3.0};

	spd<double> A = triplet_to_spd(ti, tj, tx, 3);
	EXPECT_DOUBLE_EQ(A[1, 0], 6.0);
}

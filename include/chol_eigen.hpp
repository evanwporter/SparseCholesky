/**
 * @brief Factorize a supernode using Eigen3
 * @param start supernode starting column
 * @param end supernode ending column `[start, end)`
 * @param P assembled dense panel (col-major)
 * @param S symbolic output (cp)
 * @param L global L to update in-place
 */
template <typename T>
std::expected<UpdateBlock, std::string> factorize_sn_eigen(std::size_t start, std::size_t end, panel<T>& P, const SChol& S, csc_matrix<T, sym::none>& L) {

    /// super node width (in columns)
    const auto w = end - start;

    /// #rows in panel
    const auto m = P.nrows();

    /// #below diagonal rows
    const auto mb = m - w;

    const auto& rows = P.get_rows();

    if (w <= 0)
        return std::unexpected("Empty supernode.");

    // Map dense panel into Eigen
    Eigen::Map<Eigen::MatrixXd> Pmat(P.data_ptr(), m, w);

    // L_{diag}
    // We find `L_{diag}` by performing the Cholesky on `A_{diag}`
    auto L_diag = Pmat.topRows(w);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(L_diag);
    if (llt.info() != Eigen::Success)
        return std::unexpected("Cholesky failed: block not positive definite.");

    // L_{rect}
    /// We find `L_{rect}` using a triangular solve on the equation
    /// `L_{rect} = A_{rect} L_{diag}^{-T}`
    if (mb > 0) {
        auto L_rect = Pmat.bottomRows(mb);
        // clang-format off
        L_rect = L_rect * L_diag.transpose()
                                .triangularView<Eigen::Lower>()
                                .solve(Eigen::MatrixXd::Identity(w, w));
        // clang-format on
    }

    // Scatter both diagonal & rectangular pieces back into L's CSC columns
    for (auto j = start; j < end; ++j) {
        const auto cj = j - start;

        /// Address of column `j` in panel
        const auto* Pcol = P.data_ptr() + cj * m;

        scatter_panel_column_into_L(L, j, rows, Pcol);
    }

    // Build the trailing update:
    // A_{off} := A_{off} - L_{rect} * L_{rect}^T (lower, dense)
    UpdateBlock upd;
    upd.ld = std::max(0, static_cast<int>(mb));
    if (mb > 0) {
        upd.rows.assign(rows.begin() + w, rows.end());
        upd.C.resize(mb * mb);

        auto L_rect = Pmat.bottomRows(mb);
        Eigen::MatrixXd C = -L_rect * L_rect.transpose();

        Eigen::Map<Eigen::MatrixXd>(upd.C.data(), mb, mb) = C.selfadjointView<Eigen::Lower>();
    }

    return upd;
}
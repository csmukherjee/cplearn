from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate

ArrayLike = Union[np.ndarray, Sequence[int]]
Label = Union[int, str]

from scipy.stats import rankdata, norm
import time


@dataclass
class DEOptions:
    pseudocount: float = 1.0
    min_cells_per_group: int = 3
    min_detect_pct: float = 0.05  # drop genes with both groups having pct_expr <= this
    use_exact: bool = False      # exact MWU for very small samples
    alternative: str = "two-sided"  # 'two-sided', 'less', or 'greater'
    gene_filter_min_nonzero: int = 0  # filter genes by #nonzero across all cells


class DEAnalyzer:
    """
    Wilcoxon rank-sum (Mann-Whitney U) DE for layered single-cell datasets.

    Parameters
    ----------
    X : (n_cells, n_genes) array-like
        Expression/counts matrix. Dense np.ndarray expected. If sparse, convert to CSR before init and
        use X.toarray() when extracting slices for now (or adapt below for sparse-friendly ops).
    labels : (n_cells,) array-like of ints/str
        Cluster or biological labels per cell. Label -1 will be ignored by default in comparisons.
    layers : mapping layer_id -> 1D array of cell indices
        Disjoint ring/layer assignment. The iteration order of layers in cumulative analyses follows
        `layer_order` if provided to functions; otherwise it's sorted by key.
    features : list[str] | None
        Feature names; if None, will default to [g0, g1, ...].
    options : DEOptions
        Tunable knobs for filtering, test behavior, etc.
    """

    def __init__(
        self,
        corespect_obj,
        genes: Optional[Sequence[str]] = None,
        options: Optional[DEOptions] = None,
    ) -> None:

        if corespect_obj.count_mat is None:
            print("Selecting corespect.X as the count matrix")
            cell_x_feat=corespect_obj.X
        else:
            cell_x_feat=corespect_obj.count_mat

        labels=corespect_obj.labels_
        layers={i + 1: np.array(l, dtype=int) for i, l in enumerate(corespect_obj.layers_)}


        cell_x_feat = np.asarray(cell_x_feat)
        if cell_x_feat.ndim != 2:
            raise ValueError("count matrix must be a 2D array (n_cells, n_features)")
        self.X = cell_x_feat
        self.n_cells, self.n_genes = cell_x_feat.shape

        self.labels = np.asarray(labels)
        if self.labels.shape[0] != self.n_cells:

            raise ValueError("labels length must match number of cells:",self.labels.shape[0], "vs. ", self.n_cells)

        self.layers = {k: np.asarray(v, dtype=int) for k, v in layers.items()}
        self._validate_layers_disjoint()

        if genes is None:
            self.genes = self.genes = np.array([f"g{i}" for i in range(self.n_genes)], dtype=object) #type: ignore
        else:
            if len(genes) != self.n_genes:
                raise ValueError("genes length must match number of columns in X")
            self.genes = np.array(genes, dtype=object) #type: ignore

        self.options = options or DEOptions()

        # Optional global gene prefilter
        if self.options.gene_filter_min_nonzero > 0:
            nnz = np.count_nonzero(self.X, axis=0)
            self._gene_keep_mask = nnz >= self.options.gene_filter_min_nonzero
        else:
            self._gene_keep_mask = np.ones(self.n_genes, dtype=bool)

        self.X_masked = np.ascontiguousarray(self.X[:, self._gene_keep_mask])
        self.genes_masked = self.genes[self._gene_keep_mask]

    # ---------- Validation helpers ----------
    def _validate_layers_disjoint(self) -> None:
        seen = set()
        for k, idx in self.layers.items():
            if idx.size == 0:
                continue
            if np.any(idx < 0) or np.any(idx >= self.n_cells):
                raise ValueError(f"Layer {k} has out-of-range indices")
            overlap = seen.intersection(idx.tolist())
            if overlap:
                raise ValueError(f"Layers are not disjoint; overlap detected in layer {k}")
            seen.update(idx.tolist())

    @staticmethod
    def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
        _, p_adj, _, _ = multipletests(pvals, method='fdr_bh')
        return p_adj










    # ---------- Core test on two index sets ----------
    def _test_two_groups(
        self,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
        meta: Dict[str, Union[str, int, float]]
    ) -> pd.DataFrame:
        opts = self.options

        print("Starting DE test between two groups:", meta, flush=True)
        t01=time.time()

        # Early exit on tiny groups
        if idx_a.size < opts.min_cells_per_group or idx_b.size < opts.min_cells_per_group:
            return pd.DataFrame(columns=[
                "gene", "statistic", "pvalue", "p_adj", "log2fc",
                "mean_a", "mean_b", "pct_expr_a", "pct_expr_b","z_score", *meta.keys()
            ])

        t02=time.time()

        # Apply gene mask
        X_a = self.X_masked.take(idx_a, axis=0)
        X_b = self.X_masked.take(idx_b, axis=0)
        genes_kept = self.genes_masked

        n_g = genes_kept.size
        stats = np.empty(n_g, dtype=float)
        pvals = np.empty(n_g, dtype=float)

        t03=time.time()
        # Means and detect pct for interpretability
        mean_a = X_a.mean(axis=0)
        mean_b = X_b.mean(axis=0)

        t04=time.time()


        pct_a = (X_a > 0).mean(axis=0)
        pct_b = (X_b > 0).mean(axis=0)
        t05=time.time()

        print(f"Time for data slicing: {t02 - t01:.3f} seconds", flush=True)
        print(f"Time for group extraction: {t03 - t02:.3f} seconds", flush=True)
        print(f"Time for means computation: {t04 - t03:.3f} seconds", flush=True)
        print(f"Time for pct computation: {t05 - t04:.3f} seconds", flush=True)



        t0=time.time()
        print("Pre-MWU filtering completed in:", t0 - t01, "seconds", flush=True)


        # Optional drop by detect pct
        if opts.min_detect_pct > 0.0:
            keep2 = np.logical_or(pct_a > opts.min_detect_pct, pct_b > opts.min_detect_pct)
            if not np.all(keep2):
                X_a = X_a[:, keep2]
                X_b = X_b[:, keep2]
                genes_kept = genes_kept[keep2]
                mean_a = mean_a[keep2]
                mean_b = mean_b[keep2]
                pct_a = pct_a[keep2]
                pct_b = pct_b[keep2]
                n_g = genes_kept.size
                stats = np.empty(n_g, dtype=float)
                pvals = np.empty(n_g, dtype=float)


        t1=time.time()

        # Run MWU (Wilcoxon rank-sum) per gene
        for g in range(n_g):
            # SciPy 1.7+: mannwhitneyu implements Wilcoxon rank-sum
            try:
                res = mannwhitneyu(
                    X_a[:, g], X_b[:, g],
                    alternative=opts.alternative,
                    method='exact' if opts.use_exact else 'asymptotic'
                )
                stats[g] = res.statistic
                pvals[g] = res.pvalue
            except Exception:
                stats[g] = np.nan
                pvals[g] = 1.0

        t2=time.time()

        # Concatenate groups along axis=0  → shape (n_a+n_b, n_g)
        X_all = np.vstack([X_a, X_b])
        n1, n2 = X_a.shape[0], X_b.shape[0]
        n = n1 + n2

        # Compute ranks for each gene column in one go
        # rankdata ranks along the first axis (across samples) independently for each gene
        ranks = np.apply_along_axis(rankdata, 0, X_all)

        # Sum of ranks for group A (first n1 rows)
        R1 = ranks[:n1, :].sum(axis=0)

        # Mann–Whitney U statistic per gene
        U = R1 - n1 * (n1 + 1) / 2.0
        stats = U

        # Normal approximation (no ties correction here; fine for large n)
        mean_U = n1 * n2 / 2.0
        std_U = np.sqrt(n1 * n2 * (n + 1) / 12.0)

        # z-score (two-sided)
        z = (U - mean_U) / std_U
        pvals = 2 * norm.sf(np.abs(z))

        t3=time.time()

        print(f"Time for pre-MWU filtering: {t1-t0:.3f} seconds", flush=True)
        print(f"Time for MWU loop: {t2-t1:.3f} seconds", flush=True)
        print(f"Time for rank-based U computation: {t3-t2:.3f} seconds", flush=True)

        # Rank-biserial effect size (for two-sided it's signed around direction of mean difference)
        n1 = X_a.shape[0]
        n2 = X_b.shape[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            # Convert U to rank-biserial r = 1 - 2U/(n1*n2)
            r_effect = 1.0 - (2.0 * stats) / (n1 * n2)

        # Log2 fold-change with pseudocount
        pc = opts.pseudocount
        log2fc = np.log2((mean_a + pc) / (mean_b + pc))

        # ---- Compute z-score (standardized mean difference) ----
        n_a, n_b = len(idx_a), len(idx_b)
        mean_diff = mean_a - mean_b

        # Standard deviations per gene (axis=0 = across cells)
        std_a = X_a.std(axis=0, ddof=1)
        std_b = X_b.std(axis=0, ddof=1)

        # Pooled standard error
        se = np.sqrt((std_a ** 2) / n_a + (std_b ** 2) / n_b)

        # Avoid divide-by-zero
        z_score = np.divide(mean_diff, se, out=np.zeros_like(mean_diff), where=se > 0)

        t4=time.time()

        df = pd.DataFrame({
            "gene": genes_kept,
            "statistic": stats,
            "pvalue": pvals,
            "log2fc": log2fc,
            "effect_r": r_effect,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "pct_expr_a": pct_a,
            "pct_expr_b": pct_b,
            "z_score": z_score
        })



        df["p_adj"] = self._bh_fdr(df["pvalue"].values) # type: ignore

        # Attach metadata columns (e.g., layer, comparison labels)
        for k, v in meta.items():
            df[k] = v

        # Canonical ordering: by p_adj then by |log2fc|
        #df = df.sort_values(["p_adj", "log2fc"], ascending=[True, False], ignore_index=True)

        t5=time.time()

        df=self._sort_with_pvalue_buckets(df)

        t6=time.time()

        print(f"Time for assembling DataFrame: {t5-t4:.3f} seconds", flush=True)
        print(f"Time for sorting results: {t6-t5:.3f} seconds", flush=True)



        return df

    # ---------- Public modes ----------
    def run_default(
        self,
        set1: Iterable[Label],
        set2: Optional[Iterable[Label]] = None,
        ignore_label: Label = -1,
        name1: Optional[str] = None,
        name2: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Global label comparison: cells with labels in set1 vs set2.
        If set2 is None, compare set1 vs all other labels (excluding ignore_label).
        """

        t11 = time.time()

        set1 = set(set1)
        if set2 is None:
            others = set(np.unique(self.labels)) - set1 - {ignore_label}
            set2 = others
        else:
            set2 = set(set2)

        idx_a = np.where(np.isin(self.labels, list(set1)))[0]
        idx_b = np.where(np.isin(self.labels, list(set2)))[0]

        meta = {
            "mode": "default",
            "group_a": name1 if name1 is not None else f"{sorted(list(set1))}",
            "group_b": name2 if name2 is not None else f"{sorted(list(set2))}",
        }

        t12= time.time()

        print(f"Time for preparing indices: {t12 - t11:.3f} seconds", flush=True)

        return self._test_two_groups(idx_a, idx_b, meta)

    def run_layerwise(
        self,
        set1: Iterable[Label],
        set2: Optional[Iterable[Label]] = None,
        layer_order: Optional[Sequence[Label]] = None,
        compute_per_layer=True,
        compute_cumulative=True,
        ignore_label: Label = -1,
        name1: Optional[str] = None,
        name2: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Layer-wise differential expression. For each layer L_i, compare set1 vs set2 restricted to that layer.
        Optionally also produce cumulative results using union of layers up to i (L_1 ∪ ... ∪ L_i).
        """
        set1 = set(set1)
        if set2 is None:
            others = set(np.unique(self.labels)) - set1 - {ignore_label}
            set2 = others
        else:
            set2 = set(set2)

        order = list(layer_order) if layer_order is not None else sorted(self.layers.keys(), key=lambda x: str(x))

        out_frames: List[pd.DataFrame] = []

        if compute_per_layer:
            for i, lid in enumerate(order, start=1):
                layer_idx = self.layers[lid]
                mask_a = np.isin(self.labels[layer_idx], list(set1))
                mask_b = np.isin(self.labels[layer_idx], list(set2))
                idx_a = layer_idx[mask_a]
                idx_b = layer_idx[mask_b]
                meta = {
                    "mode": "layerwise",
                    "layer": lid,
                    "i": i,
                    "cumulative": False,
                    "group_a": name1 if name1 is not None else f"{sorted(list(set1))}",
                    "group_b": name2 if name2 is not None else f"{sorted(list(set2))}",
                }
                df = self._test_two_groups(idx_a, idx_b, meta)
                out_frames.append(df)

        if compute_cumulative:
            running = []  # list of indices
            for i, lid in enumerate(order, start=1):
                running.extend(self.layers[lid])
                upto_idx = np.array(running)
                mask_a = np.isin(self.labels[upto_idx], list(set1))
                mask_b = np.isin(self.labels[upto_idx], list(set2))
                idx_a = upto_idx[mask_a]
                idx_b = upto_idx[mask_b]
                meta = {
                    "mode": "layerwise",
                    "layer": lid,
                    "i": i,
                    "cumulative": True,
                    "group_a": name1 if name1 is not None else f"{sorted(list(set1))}",
                    "group_b": name2 if name2 is not None else f"{sorted(list(set2))}",
                }
                df = self._test_two_groups(idx_a, idx_b, meta)
                out_frames.append(df)

        if not out_frames:
            return pd.DataFrame(columns=[
                "gene", "statistic", "pvalue", "p_adj", "log2fc",
                "mean_a", "mean_b", "pct_expr_a", "pct_expr_b",
                "mode", "layer", "i", "cumulative", "group_a", "group_b","z_score"
            ])
        return pd.concat(out_frames, axis=0, ignore_index=True)

    def run_core_vs_periphery(
        self,
        cluster: Label,
        layer_order: Optional[Sequence[Label]] = None,
        core_layer: Optional[Label] = None,
        name_core: str = "core",
        name_outer_prefix: str = "L",
    ) -> pd.DataFrame:
        """
        For a given cluster V, compute DEs between core (default: first layer in order or `core_layer`) and
        each outer layer intersected with that cluster.
        Returns a tidy DataFrame with metadata columns identifying the outer layer.
        """
        order = list(layer_order) if layer_order is not None else sorted(self.layers.keys(), key=lambda x: str(x))
        core_id = core_layer if core_layer is not None else order[0]

        # Indices of the cluster across all cells
        clu_mask = self.labels == cluster
        core_idx = self.layers[core_id]
        core_idx = core_idx[np.isin(self.labels[core_idx], [cluster])]

        out_frames: List[pd.DataFrame] = []
        for lid in order:
            if lid == core_id:
                continue
            outer_idx = self.layers[lid]
            outer_idx = outer_idx[np.isin(self.labels[outer_idx], [cluster])]

            meta = {
                "mode": "core_vs_periphery",
                "cluster": cluster,
                "core_layer": core_id,
                "outer_layer": lid,
                "group_a": name_core,
                "group_b": f"{name_outer_prefix}{lid}",
            }
            df = self._test_two_groups(core_idx, outer_idx, meta)
            out_frames.append(df)

        if not out_frames:
            return pd.DataFrame(columns=[
                "gene", "statistic", "pvalue", "p_adj", "log2fc",
                "mean_a", "mean_b", "pct_expr_a", "pct_expr_b",
                "mode", "cluster", "core_layer", "outer_layer", "group_a", "group_b","z_score"
            ])
        return pd.concat(out_frames, axis=0, ignore_index=True)

    # ---------- Utilities ----------
    #-------- P-adj valued buckets and then sort by |Logf2change| --------#
    @staticmethod
    def _sort_with_pvalue_buckets(df: pd.DataFrame) -> pd.DataFrame:

        # Define bins and labels
        bins = [0, 0.001, 0.01, 0.05, np.inf]
        labels = [1, 2, 3, 4]  # smaller = more significant

        df["abs_log2fc"] = df["log2fc"].abs()


        df = df.copy()
        df["p_bucket"] = pd.cut(df["p_adj"], bins=bins, labels=labels, include_lowest=True)
        df["p_bucket"] = df["p_bucket"].astype(int)

        # Sort by bucket, then by absolute fold change
        return df.sort_values(["p_bucket", "abs_log2fc"], ascending=[True, False])

    @staticmethod
    def observe_de(
            df: pd.DataFrame,
            n_rows: int = 10,
            fmt: str = "github",
            genes: list[str] = None
    ) -> None:
        """
        Display a concise tabulated summary of differential expression results.

        Parameters
        ----------
        df : pd.DataFrame
            DE results DataFrame (from run_default, run_layerwise, etc.)
        n_rows : int
            Number of top rows to display.
        fmt : str
            Tabulate formatting style (e.g. "github", "psql", "fancy_grid").
        genes : list[str], optional
            If provided, restrict output to these genes.
        """
        if df.empty:
            print("(no results)")
            return

        # Optional gene filtering
        if genes is not None:
            df = df[df["gene"].isin(genes)]
            if df.empty:
                print(f"(no results for selected genes: {genes})")
                return

        # Select relevant columns
        cols = [
            "gene", "log2fc", "z_score", "mean_a", "mean_b",
            "pct_expr_a", "pct_expr_b", "pvalue"
        ]
        cols = [c for c in cols if c in df.columns]

        df_show = df[cols].copy()
        df_show = df_show.round({
            "log2fc": 3, "z_score": 3,
            "mean_a": 3, "mean_b": 3,
            "pct_expr_a": 3, "pct_expr_b": 3,
            "pvalue": 3
        })

        print(tabulate(df_show.head(n_rows), headers="keys", tablefmt=fmt, showindex=False))


    @staticmethod
    def observe_layerwise(
            df: pd.DataFrame,
            n_rows: int = 20,
            metrics: list[str] = None,
            cumulative: bool = False,
            genes: list[str] = None
    ) -> None:
        """
        Show how DE statistics for each gene evolve across layers,
        ordered by highest |log2fc| across all layers.
        """
        if df.empty:
            print("(no results to display)")
            return

        if metrics is None:
            metrics = [
                "log2fc", "z_score", "mean_a", "mean_b",
                "pct_expr_a", "pct_expr_b", "pvalue"
            ]

        # Filter cumulative / non-cumulative
        if "cumulative" in df.columns:
            df = df[df["cumulative"] == cumulative].copy()
        if df.empty:
            print(f"(no {'cumulative' if cumulative else 'per-layer'} results)")
            return

        # Filter by gene names BEFORE ranking
        if genes is not None:
            df = df[df["gene"].isin(genes)]
            if df.empty:
                print(f"(no results for selected genes: {genes})")
                return

        # Rank genes by their maximum absolute log2FC across layers
        df["abs_fc"] = df["log2fc"].abs()
        top_genes = (
            df.groupby("gene")["abs_fc"]
            .max()
            .sort_values(ascending=False)
            .head(n_rows)
            .index.tolist()
        )

        # Display each gene’s stats (in order of descending |log2fc|)
        for g in top_genes:
            sub = df[df["gene"] == g].copy()
            sub["abs_fc"] = sub["log2fc"].abs()
            sub = sub.sort_values("layer", ascending=True)  # sort by |log2fc|
            print(f"\n {g}")
            cols = ["layer"] + [m for m in metrics if m in sub.columns]
            disp = sub[cols].round(4)
            print(tabulate(disp, headers="keys", tablefmt="github", showindex=False))

    @staticmethod
    def observe_core_vs_periphery(
            df: pd.DataFrame,
            n: int = 20,  # top-N per outer_layer before union
            metrics: list[str] = None,
            genes: list[str] = None
    ) -> None:
        """
        Gene-centric view across outer layers (core vs periphery).
        - Restrict to `genes` if provided (before selection).
        - Take top-N genes by |log2fc| PER outer_layer, union them.
        - Order genes by total change Σ|Δ log2fc| across outer layers.
        """
        if df.empty:
            print("(no results to display)")
            return

        if metrics is None:
            metrics = [
                "log2fc", "z_score", "mean_a", "mean_b",
                "pct_expr_a", "pct_expr_b", "pvalue"
            ]

        # 1) Optional hard filter first (restrict candidate pool)
        if genes is not None:
            df = df[df["gene"].isin(genes)]
            if df.empty:
                print(f"(no results for selected genes: {genes})")
                return

        # 2) Top-N per outer_layer by |log2fc|
        cand = (
            df.assign(abs_fc=df["log2fc"].abs())
            .sort_values(["outer_layer", "abs_fc"], ascending=[True, False])
            .groupby("outer_layer", group_keys=False)
            .head(n)
        )

        # 3) Union of candidate genes (this was missing before)
        union_genes = set(cand["gene"].unique())
        if not union_genes:
            print("(no genes selected after filtering)")
            return

        # Work only on the union set
        dfu = df[df["gene"].isin(union_genes)].copy()

        # 4) Rank genes by total change Σ|Δ log2fc| across outer layers
        total_change = {}
        for g, sub in dfu.groupby("gene"):
            sub = sub.sort_values("outer_layer")
            if sub.shape[0] > 1:
                diffs = np.diff(sub["log2fc"].values)
                total_change[g] = float(np.sum(np.abs(diffs)))
            else:
                total_change[g] = 0.0

        ranked_genes = sorted(total_change, key=total_change.get, reverse=True)
        if not ranked_genes:
            print("(no genes selected for display)")
            return

        # 5) Print one block per gene (rows = outer layers)
        for g in ranked_genes:
            sub = dfu[dfu["gene"] == g].sort_values("outer_layer")
            print(f"\n🧬 {g}   (Σ|Δlog2fc| = {total_change[g]:.3f})")
            cols = ["outer_layer"] + [m for m in metrics if m in sub.columns]
            disp = sub[cols].round(4)
            print(tabulate(disp, headers="keys", tablefmt="github", showindex=False))



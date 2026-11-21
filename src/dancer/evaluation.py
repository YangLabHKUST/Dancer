"""Reference-selected markers, MECR/PMP/ASW, count UMAPs and a static report."""
import base64
import html
from collections import Counter
from itertools import combinations
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .utils import save_json, log


def normalize(a, total):
    a = a.copy()
    sc.pp.normalize_total(a, target_sum=total)
    sc.pp.log1p(a)
    return a


def select_markers(reference, cfg, dest):
    p = cfg["evaluation"]["markers"]
    positivity = cfg["evaluation"]["positivity_threshold"]
    labels = reference.obs.cell_type.astype(str).to_numpy()
    types = sorted(np.unique(labels))
    ref = normalize(reference, p["normalization_total"])
    ref.obs["cell_type"] = pd.Categorical(labels, categories=types)
    sc.tl.rank_genes_groups(ref, "cell_type", method="wilcoxon", use_raw=False, tie_correct=False)
    binary = sparse.csr_matrix(reference.X >= positivity)
    total_positive = np.asarray(binary.sum(0)).ravel()
    tables, pmp, mecr = [], {}, {}
    for t in types:
        in_type = labels==t
        inside = np.asarray(binary[in_type].sum(0)).ravel()
        fractions = pd.DataFrame({"gene":reference.var_names, "fraction_in":inside/in_type.sum(),
            "fraction_out":(total_positive-inside)/(~in_type).sum()}).set_index("gene")
        table = sc.get.rank_genes_groups_df(ref, group=t).rename(columns={"names":"gene"})
        table = table.join(fractions, on="gene")
        table["cell_type"] = t
        table["selected"] = (table.pvals_adj<=p["adjusted_p"]) & (table.logfoldchanges>=p["logfoldchange"]) & (table.fraction_in>=p["min_in_fraction"]) & (table.fraction_out<=p["max_out_fraction"])
        pmp[t] = table.loc[table.selected, "gene"].tolist()
        tables.append(table)
    top = {t:gs[:p["mecr_top_n"]] for t,gs in pmp.items()}
    frequency = Counter(g for genes in top.values() for g in genes)
    mecr = {t:[g for g in gs if frequency[g]==1] for t,gs in top.items()}
    owners = {g:t for t,gs in mecr.items() for g in gs}
    pairs = [(g,h) for g,h in combinations(owners, 2) if owners[g]!=owners[h]]
    dest.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(dest/"reference_differential_expression.csv", index=False)
    pd.DataFrame([dict(cell_type=t,gene=g,rank=i+1) for t,gs in pmp.items() for i,g in enumerate(gs)], columns=["cell_type","gene","rank"]).to_csv(dest/"pmp_markers.csv", index=False)
    pd.DataFrame([dict(cell_type=t,gene=g,rank=i+1) for t,gs in mecr.items() for i,g in enumerate(gs)], columns=["cell_type","gene","rank"]).to_csv(dest/"mecr_markers.csv", index=False)
    pd.DataFrame([dict(cell_type_1=owners[g],gene_1=g,cell_type_2=owners[h],gene_2=h) for g,h in pairs], columns=["cell_type_1","gene_1","cell_type_2","gene_2"]).to_csv(dest/"mecr_pairs.csv", index=False)
    coverage = pd.DataFrame([dict(cell_type=t, PMP_markers=len(pmp[t]), MECR_markers=len(mecr[t])) for t in types])
    coverage.to_csv(dest/"coverage.csv", index=False)
    save_json(dest/"markers.json", dict(pmp=pmp, mecr=mecr, pairs=pairs, settings=p, positivity_threshold=positivity))
    return pmp, mecr, pairs, coverage


def contamination_metrics(a, pmp, pairs, positivity):
    x = sparse.csr_matrix(a.X, dtype=np.float64)
    genes = a.var_names.get_indexer(list({g for pair in pairs for g in pair}))
    pair_genes = a.var_names[genes].tolist()
    pos = (x[:, genes]>=positivity).astype(np.int64)
    co = (pos.T@pos).toarray()
    totals = np.asarray(pos.sum(0)).ravel()
    lookup = {g:i for i,g in enumerate(pair_genes)}
    values = []
    for g,h in pairs:
        i,j = lookup[g], lookup[h]
        union = totals[i]+totals[j]-co[i,j]
        if union:
            values.append(co[i,j]/union)
    libraries = np.asarray(x.sum(1)).ravel()
    labels = a.obs.cell_type.astype(str).to_numpy()
    purity = []
    for t in sorted(np.unique(labels)):
        rows = labels==t
        gi = a.var_names.get_indexer(pmp.get(t, []))
        sums = np.asarray(x[rows][:, gi].sum(1)).ravel()
        fractions = np.divide(sums, libraries[rows], out=np.zeros_like(sums), where=libraries[rows]>0)
        purity.append(dict(cell_type=t, cells=int(rows.sum()), PMP=float(fractions.mean())))
    return dict(MECR=float(np.mean(values)) if values else float("nan"), PMP=float(np.mean([v["PMP"] for v in purity]))), purity


def embedding(a, settings, seed, umap=True):
    local = normalize(a, settings["normalization_total"])
    n = min(settings["n_pcs"], local.n_vars-1, local.n_obs-1)
    if n<2:
        raise ValueError("PCA requires at least three cells and genes")
    sc.pp.pca(local, n_comps=n, random_state=seed, svd_solver="arpack")
    if umap:
        sc.pp.neighbors(local, n_neighbors=min(settings["n_neighbors"],local.n_obs-1), use_rep="X_pca", random_state=seed)
        sc.tl.umap(local, min_dist=settings["min_dist"], random_state=seed)
    return local


def save_figure(fig, dest, name):
    fig.savefig(dest/(name+".pdf"), bbox_inches="tight")
    fig.savefig(dest/(name+".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def dotplot(reference, raw, corrected, pmp, cfg, folder):
    types = sorted(pmp)
    display = list(dict.fromkeys(g for t in types for g in pmp[t][:cfg["evaluation"]["markers"]["display_per_type"]]))
    if not display:
        return None
    values, fractions = [], []
    for a in (reference, raw, corrected):
        normalized = normalize(a, 10000)
        gi = a.var_names.get_indexer(display)
        labels = a.obs.cell_type.astype(str).to_numpy()
        values.append(np.stack([np.asarray(normalized.X[labels==t][:,gi].mean(0)).ravel() for t in types]))
        fractions.append(np.stack([np.asarray((a.X[labels==t][:,gi]>=cfg["evaluation"]["positivity_threshold"]).mean(0)).ravel() for t in types]))
    scale = np.maximum(np.stack(values).max(axis=(0,1)), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(max(12,len(display)*.5), max(4,len(types)*.3)), sharey=True, layout="constrained")
    xx, yy = np.meshgrid(np.arange(len(display)), np.arange(len(types)))
    for ax, title, mean, fraction in zip(axes, ["Reference", "Raw", "Dancer"], values, fractions):
        points = ax.scatter(xx.ravel(), yy.ravel(), s=90*fraction.ravel(), c=(mean/scale).ravel(), vmin=0,vmax=1,cmap="viridis", linewidths=.2,edgecolors="#555555")
        ax.set(title=title,xticks=np.arange(len(display)),yticks=np.arange(len(types)),yticklabels=types)
        ax.set_xticklabels(display,rotation=90,fontsize=7)
        ax.set_ylim(len(types)-.4,-.6)
        ax.set_xlim(-.6,len(display)-.4)
        ax.tick_params(axis="y",labelsize=8)
    fig.colorbar(points,ax=axes,label="Relative mean log1p(CP10k)",shrink=.7)
    handles=[axes[-1].scatter([],[],s=90*f,color="gray",label=f"{f:.0%}") for f in [.25,.5,1.]]
    axes[-1].legend(handles=handles,title="Positive-cell fraction",loc="upper center",bbox_to_anchor=(.5,-.3),ncol=3,frameon=False,fontsize=8)
    save_figure(fig,folder,"marker_dotplot")
    return "marker_dotplot"


def report(root, metrics, coverage, descriptions, figure_names):
    sections=[]
    for title, name in figure_names:
        png=root/"figures"/(name+".png")
        if png.exists():
            encoded=base64.b64encode(png.read_bytes()).decode()
            sections.append(f'<h2>{html.escape(title)}</h2><img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}">')
    text = """<!doctype html><html><head><meta charset="utf-8"><title>Dancer evaluation</title>
<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;padding:0 24px;color:#222}img{max-width:100%;height:auto}table{border-collapse:collapse;margin:18px 0}th,td{padding:8px 15px;border-bottom:1px solid #ddd;text-align:left}h2{margin-top:36px}li{margin:8px 0}</style></head><body><h1>Dancer: breast cancer evaluation</h1>"""
    text += metrics.to_html(index=False,float_format=lambda x:f"{x:.5f}",na_rep="Not calculated")
    text += "<ul>"+"".join(f"<li>{html.escape(s)}</li>" for s in descriptions)+"</ul>"
    text += "".join(sections)+"<h2>Reference-selected marker coverage</h2>"+coverage.to_html(index=False)
    text += "<p>Separate PDF/PNG figures are in figures/. Marker lists, metric tables and embedding coordinates are in evaluation/.</p></body></html>"
    (root/"report.html").write_text(text)


def evaluate(cfg):
    root=Path(cfg["paths"]["results"])
    tables=root/"evaluation"; figures=root/"figures"
    for p in [tables,figures,tables/"embeddings"]:
        p.mkdir(parents=True,exist_ok=True)
    ref=ad.read_h5ad(cfg["paths"]["reference"])
    raw=ad.read_h5ad(cfg["paths"]["spatial"])
    corrected=ad.read_h5ad(root/"outputs/corrected.h5ad")
    if not raw.obs_names.equals(corrected.obs_names) or not raw.var_names.equals(corrected.var_names):
        raise ValueError("Raw and Dancer cells/genes must be aligned")
    if not raw.var_names.equals(ref.var_names):
        raise ValueError("Reference genes must match the measured panel")
    if not np.array_equal(raw.obs.cell_type.astype(str),corrected.obs.cell_type.astype(str)):
        raise ValueError("Raw and Dancer annotations must agree")
    for a in [ref,raw,corrected]:
        a.X=sparse.csr_matrix(a.X,dtype=np.float32)
    settings=cfg["evaluation"]; u=settings["umap"]; seed=cfg["seed"]
    sc.settings.n_jobs=cfg["threads"]
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none"})
    with threadpool_limits(limits=cfg["threads"]):
        log("evaluate_markers")
        pmp, mecr, pairs, coverage=select_markers(ref,cfg,tables/"markers")
        rows=[]
        asw_keep=(np.asarray(raw.X.sum(1)).ravel()>=settings["asw"]["min_counts"]) & (np.asarray(corrected.X.sum(1)).ravel()>=settings["asw"]["min_counts"])
        pd.DataFrame({"cell_id":raw.obs_names[asw_keep]}).to_csv(tables/"asw_cells.csv",index=False)
        embeddings={}
        for method,a in [("Raw",raw),("Dancer",corrected)]:
            log("evaluate_metrics",method=method)
            score,purity=contamination_metrics(a,pmp,pairs,settings["positivity_threshold"])
            pd.DataFrame(purity).to_csv(tables/(method.lower()+"_pmp_by_type.csv"),index=False)
            score.update(Method=method,ASW=float("nan"),ASW_cells=int(asw_keep.sum()))
            if settings["asw"]["enabled"]:
                from scib_metrics import silhouette_label
                local=embedding(a[asw_keep].copy(),u,seed,umap=False)
                log("evaluate_asw_full",method=method,cells=local.n_obs)
                score["ASW"]=float(silhouette_label(local.obsm["X_pca"],local.obs.cell_type.astype(str).to_numpy(),rescale=True))
            rows.append(score)
            pd.DataFrame(rows).to_csv(tables/"metrics.csv",index=False)
            keep=np.asarray(a.X.sum(1)).ravel()>=u["min_counts"]
            log("evaluate_umap",method=method,cells=int(keep.sum()))
            local=embedding(a[keep].copy(),u,seed)
            embeddings[method]=local
            np.savez_compressed(tables/"embeddings"/(method.lower()+"_umap.npz"),X_umap=local.obsm["X_umap"],cell_ids=local.obs_names.to_numpy(dtype=str),labels=local.obs.cell_type.to_numpy(dtype=str))
        metrics=pd.DataFrame(rows)[["Method","MECR","PMP","ASW","ASW_cells"]]
        metrics.to_csv(tables/"metrics.csv",index=False)
        fig,axes=plt.subplots(1,3,figsize=(9,3),layout="constrained")
        for ax,name,title in zip(axes,["MECR","PMP","ASW"],["MECR (lower is better)","PMP (higher is better)","ASW (higher is better)"]):
            ax.bar(metrics.Method,metrics[name],color=["#929aa4","#c23f75"],width=.65)
            ax.set_title(title,fontsize=10);ax.spines[["top","right"]].set_visible(False)
        save_figure(fig,figures,"metrics")
        types=sorted(pmp); colors={t:plt.get_cmap("tab20")(i%20) for i,t in enumerate(types)}
        fig,axes=plt.subplots(1,2,figsize=(11,6))
        for ax,(name,a) in zip(axes,embeddings.items()):
            xy=a.obsm["X_umap"]; labels=a.obs.cell_type.astype(str).to_numpy()
            for t in types:
                k=labels==t; ax.scatter(xy[k,0],xy[k,1],s=1,color=colors[t],linewidths=0)
            ax.set(title=name,xlabel="UMAP 1",ylabel="UMAP 2",xticks=[],yticks=[])
            ax.spines[["top","right"]].set_visible(False)
        handles=[Line2D([],[],marker="o",linestyle="",color=colors[t],label=t,markersize=5) for t in types]
        fig.legend(handles=handles,loc="lower center",ncol=min(6,len(types)),frameon=False,fontsize=7)
        fig.subplots_adjust(bottom=.22,wspace=.16)
        save_figure(fig,figures,"umap_raw_dancer")
        dot=dotplot(ref,raw,corrected,pmp,cfg,figures)
    descriptions=["MECR and PMP use all input spatial cells; markers are selected once from the reference and fixed across methods.",
        f"Marker thresholds: adjusted P <= {settings['markers']['adjusted_p']}, log fold change >= {settings['markers']['logfoldchange']}, target detection >= {settings['markers']['min_in_fraction']:.0%}, other-cell detection <= {settings['markers']['max_out_fraction']:.0%}.",
        f"ASW uses the intersection of Raw and Dancer cells with total counts >= {settings['asw']['min_counts']} ({int(asw_keep.sum()):,} cells); rescaled to 0–1, without subsampling.",
        f"UMAPs are independently computed on each method's cells with counts >= {u['min_counts']}; CP10k, log1p, PCA, neighbor graph and UMAP; seed {seed}.",
        "The comparison set here is Raw and Dancer. An ASW cohort that also includes other methods can contain different cells."]
    if not settings["asw"]["enabled"]:
        descriptions.append("ASW was disabled in this configuration.")
    if not pairs:
        descriptions.append("No mutually exclusive marker pairs passed these thresholds; MECR is not defined.")
    charts=[("Metric comparison","metrics"),("Raw and Dancer UMAP","umap_raw_dancer")]
    if dot:
        charts.append(("Reference, Raw and Dancer markers",dot))
    report(root,metrics,coverage,descriptions,charts)
    save_json(tables/"evaluation_settings.json",dict(config=settings,seed=seed,asw_cells=int(asw_keep.sum())))
    print(metrics.to_string(index=False),flush=True)
    log("evaluation_complete",report=str(root/"report.html"),figures=str(figures),tables=str(tables))

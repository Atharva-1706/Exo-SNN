"""Real-TESS fine-tuning for ExoSNN v9.

The catalog labels come from the NASA Exoplanet Archive TOI table:
CP/KP -> planet (1), FP/FA -> false positive/alarm (0). Targets with any
CP/KP disposition are excluded from the negative pool. Splitting is done by
TIC before any augmentation, so the same host never appears in train and val.

Default holdout excludes the three benchmark TICs used during development:
172518755, 224245334 and 331484419.
"""
import argparse, copy, json, os, time
from urllib.parse import urlencode
import requests
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from root.checkpoint_io import load_checkpoint
from root.data.ingest import download_tess_lightcurve, list_available_sectors
from root.preprocessing.clean import clean_and_flatten
from root.detection.bls_detector import run_bls
from root.detection.dual_stream import extract_dual_views
from root.classification.tri_branch_ensemble import TriBranchTESSNet

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT, "weights")
DEFAULT_OUT = os.path.join(WEIGHTS_DIR, "tri_branch_tess_net.pt")
DEFAULT_PRETRAINED = os.path.join(WEIGHTS_DIR, "tri_branch_tess_net_pretrained.pt")
ARCHIVE_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
DEFAULT_HOLDOUT = {172518755, 224245334, 331484419}


def archive_query(sql):
    r = requests.get(ARCHIVE_TAP, params={"query": sql, "format": "json"}, timeout=60)
    r.raise_for_status()
    payload = r.json()
    # TAP JSON normally returns records as dictionaries. Normalize them to
    # the positional column order used by build_catalog below.
    if isinstance(payload, dict):
        rows = payload.get("data", payload.get("rows"))
        if rows is None:
            raise RuntimeError(f"Unexpected NASA Archive JSON response keys: {list(payload)[:10]}")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise RuntimeError(f"Unexpected NASA Archive response type: {type(payload).__name__}")

    columns = ["toi", "tid", "tfopwg_disp", "pl_orbper", "pl_tranmid",
               "pl_trandurh", "pl_trandep", "st_teff", "st_logg", "st_rad"]
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append([row.get(c) for c in columns])
        elif isinstance(row, (list, tuple)):
            normalized.append(list(row))
        else:
            continue
    return normalized


def fetch_toi_catalog():
    sql = (
        "select toi,tid,tfopwg_disp,pl_orbper,pl_tranmid,"
        "pl_trandurh,pl_trandep,st_teff,st_logg,st_rad "
        "from toi where tid is not null and tfopwg_disp is not null"
    )
    rows = archive_query(sql)
    if not rows:
        raise RuntimeError("NASA Exoplanet Archive returned no TOI rows.")
    return rows


def build_catalog(max_per_class, holdout):
    rows = fetch_toi_catalog()
    by_tic = {}
    positive_tics = set()
    for r in rows:
        try: tic = int(float(r[1]))
        except Exception: continue
        disp = str(r[2]).upper().strip()
        by_tic.setdefault(tic, []).append(r)
        if disp in {"CP", "KP"}: positive_tics.add(tic)
    positives, negatives = [], []
    for tic, rs in by_tic.items():
        if tic in holdout: continue
        pos = [r for r in rs if str(r[2]).upper().strip() in {"CP", "KP"}]
        neg = [r for r in rs if str(r[2]).upper().strip() in {"FP", "FA"}]
        if pos:
            # Prefer a row with a finite orbital period.
            r = next((x for x in pos if _finite(x[3]) and float(x[3]) > 0), pos[0])
            positives.append(_row_meta(tic, r, 1))
        elif neg:
            r = next((x for x in neg if _finite(x[3]) and float(x[3]) > 0), neg[0])
            negatives.append(_row_meta(tic, r, 0))
    rng = np.random.default_rng(42)
    rng.shuffle(positives); rng.shuffle(negatives)
    positives = positives[:max_per_class]; negatives = negatives[:max_per_class]
    return positives, negatives


def _finite(x):
    try: return np.isfinite(float(x))
    except Exception: return False


def _row_meta(tic, r, label):
    return {
        "tic": tic,
        "toi": str(r[0]),
        "label": label,
        "period": float(r[3]) if _finite(r[3]) else None,
        "epoch_bjd": float(r[4]) if _finite(r[4]) else None,
        "duration_h": float(r[5]) if _finite(r[5]) else None,
        "depth_ppm": float(r[6]) if _finite(r[6]) else None,
        "teff": float(r[7]) if _finite(r[7]) else None,
        "logg": float(r[8]) if _finite(r[8]) else None,
        "rstar": float(r[9]) if _finite(r[9]) else None,
    }


def choose_training_sector(meta):
    """Choose a sector containing a catalog-predicted transit when possible."""
    tic = meta["tic"]
    period = meta.get("period")
    epoch_bjd = meta.get("epoch_bjd")
    sectors = list_available_sectors(tic)

    if not sectors:
        raise ValueError(f"No TESS sectors available for TIC {tic}")

    # Positive TOIs need a sector that actually contains a predicted transit.
    # For negatives, use catalog timing when available; otherwise keep the
    # first available sector because there is no confirmed transit epoch.
    if meta["label"] == 1 and (not _finite(period) or float(period) <= 0 or not _finite(epoch_bjd)):
        raise ValueError(f"TIC {tic} has no usable catalog period/epoch")

    if _finite(period) and float(period) > 0 and _finite(epoch_bjd):
        period = float(period)
        epoch_btjd = float(epoch_bjd) - 2457000.0
        candidates = []

        for item in sectors:
            sector = item.get("sector")
            t_min = item.get("t_min")
            t_max = item.get("t_max")
            if sector is None or t_min is None or t_max is None:
                continue
            t_min, t_max = float(t_min), float(t_max)
            if t_min > 2400000:
                t_min -= 2457000.0
                t_max -= 2457000.0

            n0 = int(np.floor((t_min - epoch_btjd) / period)) - 1
            n1 = int(np.ceil((t_max - epoch_btjd) / period)) + 1
            predicted = [epoch_btjd + n * period for n in range(n0, n1 + 1)]
            inside = [t for t in predicted if t_min <= t <= t_max]
            if inside:
                candidates.append({
                    "sector": int(sector),
                    "n_transits": len(inside),
                    "first_transit": min(inside),
                })

        if candidates:
            candidates.sort(key=lambda x: (x["n_transits"], -abs(x["first_transit"] - epoch_btjd)), reverse=True)
            selected = candidates[0]["sector"]
            print(f"   Selected sector {selected} for TIC {tic}: {candidates[0]['n_transits']} predicted transit(s)")
            return selected

        if meta["label"] == 1:
            raise ValueError(
                f"No available TESS sector contains a predicted transit "
                f"(P={period:.6f} d, epoch={epoch_bjd:.5f} BJD)"
            )

    selected = next((x.get("sector") for x in sectors if x.get("sector") is not None), None)
    if selected is None:
        raise ValueError(f"No usable sector metadata for TIC {tic}")
    print(f"   Selected sector {selected} for TIC {tic} (no usable transit epoch)")
    return int(selected)


def prepare_target(meta, cache_dir, verbose=True):
    tic = meta["tic"]
    os.makedirs(cache_dir, exist_ok=True)

    try:
        sector = choose_training_sector(meta)
        path = os.path.join(cache_dir, f"TIC_{tic}_S{sector}.npz")
        if os.path.exists(path):
            d = np.load(path)
            cached_meta = dict(meta)
            cached_meta["sector"] = sector
            return d["global_view"].astype(np.float32), d["local_view"].astype(np.float32), cached_meta

        t, f, ferr = download_tess_lightcurve(tic, sector=sector)
        tc, fc, ferr_clean = clean_and_flatten(t, f, ferr)
        b = run_bls(tc, fc, ferr_clean)

        # A positive training example must recover the catalog period or a
        # common BLS harmonic. This prevents unrelated sectors/artifacts from
        # being silently labeled as planet morphology.
        if meta["label"] == 1 and _finite(meta.get("period")):
            catalog_p = float(meta["period"])
            ratio = float(b["period"]) / catalog_p
            mismatch = min(
                abs(ratio - 1.0), abs(ratio - 0.5), abs(ratio - 2.0),
                abs(ratio - (1.0 / 3.0)), abs(ratio - 3.0)
            )
            if mismatch > 0.03:
                raise ValueError(
                    f"BLS period {b['period']:.5f} d does not match "
                    f"catalog period {catalog_p:.5f} d"
                )

        g, l = extract_dual_views(tc, fc, b["period"], b["t0"], b["duration"])
        meta = dict(meta)
        meta.update({
            "sector": sector,
            "bls_period": float(b["period"]),
            "bls_snr": float(b["snr"]),
        })
        np.savez_compressed(
            path,
            global_view=g.astype(np.float32),
            local_view=l.astype(np.float32),
        )
        return g.astype(np.float32), l.astype(np.float32), meta
    except Exception as e:
        if verbose: print(f"   SKIP TIC {tic}: {type(e).__name__}: {e}")
        return None


def load_cache(cache_dir, metas):
    G, L, Y, META = [], [], [], []

    for m in metas:
        tic = m["tic"]
        sector = m.get("sector")

        if sector is not None:
            path = os.path.join(
                cache_dir,
                f"TIC_{tic}_S{sector}.npz"
            )
        else:
            path = os.path.join(
                cache_dir,
                f"TIC_{tic}.npz"
            )

        if not os.path.exists(path):
            print(f"   CACHE MISS: TIC {tic} sector {sector}")
            continue

        try:
            d = np.load(path)

            G.append(
                d["global_view"].astype(np.float32)
            )
            L.append(
                d["local_view"].astype(np.float32)
            )
            Y.append(
                int(m["label"])
            )
            META.append(m)

        except Exception as e:
            print(
                f"   CACHE ERROR: TIC {tic}: "
                f"{type(e).__name__}: {e}"
            )

    if not Y:
        raise RuntimeError(
            "No usable real-TESS samples were cached."
        )

    return (
        np.stack(G),
        np.stack(L),
        np.asarray(Y, dtype=np.int64),
        META
    )


def split_by_tic(metas, val_frac, seed):
    rng=np.random.default_rng(seed)
    train=[]; val=[]
    for cls in (0,1):
        items=[m for m in metas if m['label']==cls]
        rng.shuffle(items); n=max(1,int(round(len(items)*val_frac)))
        val.extend(items[:n]); train.extend(items[n:])
    rng.shuffle(train); rng.shuffle(val)
    return train,val


def augment(g,l,rng):
    g=g.copy(); l=l.copy()
    # Small amplitude/noise perturbations preserve transit morphology after z-score normalization.
    if rng.random()<0.8: l += rng.normal(0,0.035,size=l.shape).astype(np.float32)
    if rng.random()<0.6: g += rng.normal(0,0.020,size=g.shape).astype(np.float32)
    if rng.random()<0.25:
        k=int(rng.integers(1,4)); idx=rng.choice(len(l),size=k,replace=False); l[idx]=0.0
    return g,l


def tensors(G,L,augment_n=0,seed=42):
    rng=np.random.default_rng(seed); gs=[G]; ls=[L]
    for _ in range(augment_n):
        ag=[]; al=[]
        for g,l in zip(G,L):
            x,y=augment(g,l,rng); ag.append(x); al.append(y)
        gs.append(np.asarray(ag,np.float32)); ls.append(np.asarray(al,np.float32))
    return np.concatenate(gs),np.concatenate(ls)


def logits(model,G,L,device):
    with torch.no_grad():
        x=torch.from_numpy(G).to(device).unsqueeze(1); y=torch.from_numpy(L).to(device).unsqueeze(1)
        return model(x,y).cpu()


def metrics(model,G,L,y,device,temperature=1.0):
    z=logits(model,G,L,device)/float(temperature); p=torch.softmax(z,1)[:,1].numpy(); pred=(p>=0.5).astype(int)
    return {"accuracy":float(accuracy_score(y,pred)),"f1":float(f1_score(y,pred,zero_division=0)),
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float('nan'),
            "average_precision":float(average_precision_score(y,p)) if len(np.unique(y))>1 else float('nan'),
            "mean_planet_score":float(p[y==1].mean()) if np.any(y==1) else float('nan'),
            "mean_fp_score":float(p[y==0].mean()) if np.any(y==0) else float('nan')},p


def fit_temperature(model,G,L,y,device):
    z=logits(model,G,L,device); yt=torch.from_numpy(y).to(device)
    log_t=torch.nn.Parameter(torch.zeros(1,device=device))
    opt=torch.optim.LBFGS([log_t],lr=0.1,max_iter=50)
    loss_fn=nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss=loss_fn(z.to(device)/torch.exp(log_t),yt); loss.backward(); return loss
    opt.step(closure)
    return float(torch.exp(log_t).detach().cpu().item())


def train(model,G,L,y,Gv,Lv,yv,epochs,backbone_lr,head_lr,batch,device,patience=7):
    """Fine-tune synthetic-pretrained morphology weights on real TESS."""
    model.to(device)
    backbone = list(model.global_conv.parameters()) + list(model.local_conv.parameters()) + list(model.snn_branch.parameters())
    head = list(model.classifier.parameters())
    opt=torch.optim.AdamW([
        {"params": backbone, "lr": backbone_lr},
        {"params": head, "lr": head_lr},
    ], weight_decay=2e-4)
    loss_fn=nn.CrossEntropyLoss(label_smoothing=0.02)
    rng=np.random.default_rng(123)
    best_auc=-np.inf; best_state=None; best_epoch=0; stale=0
    gv=torch.from_numpy(Gv).to(device).unsqueeze(1)
    lv=torch.from_numpy(Lv).to(device).unsqueeze(1)
    yvt=torch.from_numpy(yv).to(device)
    for ep in range(epochs):
        model.train(); order=rng.permutation(len(y)); total=0
        for s in range(0,len(y),batch):
            ix=order[s:s+batch]
            gb,lb=augment(G[ix],L[ix],rng)
            gx=torch.from_numpy(gb).to(device).unsqueeze(1); lx=torch.from_numpy(lb).to(device).unsqueeze(1)
            yy=torch.from_numpy(y[ix]).to(device)
            opt.zero_grad(set_to_none=True); out=model(gx,lx); loss=loss_fn(out,yy); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step(); total+=loss.item()*len(ix)
        model.eval()
        with torch.no_grad():
            val_logits=model(gv,lv); val_loss=loss_fn(val_logits,yvt).item()
            val_probs=torch.softmax(val_logits,dim=1)[:,1].detach().cpu().numpy()
        val_auc=float(roc_auc_score(yv,val_probs)) if len(np.unique(yv))>1 else float("nan")
        val_f1=float(f1_score(yv,(val_probs>=0.5).astype(int),zero_division=0))
        improved=np.isfinite(val_auc) and val_auc > best_auc + 1e-9
        if improved:
            best_auc=val_auc; best_state=copy.deepcopy(model.state_dict()); best_epoch=ep+1; stale=0
        else: stale += 1
        print(f"   epoch {ep+1:02d}/{epochs} loss={total/len(y):.4f} val_loss={val_loss:.4f} val_auc={val_auc:.3f} val_f1={val_f1:.3f}")
        if stale >= patience and ep >= 7:
            print(f"   early stopping after epoch {ep+1}; best validation AUC={best_auc:.3f} at epoch {best_epoch}")
            break
    if best_state is not None: model.load_state_dict(best_state)
    return model,best_auc,best_epoch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-per-class',type=int,default=40)
    ap.add_argument('--val-frac',type=float,default=0.2)
    ap.add_argument('--epochs',type=int,default=25)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--lr',type=float,default=2e-5, help='Backbone fine-tuning learning rate')
    ap.add_argument('--head-lr',type=float,default=5e-5, help='Classifier-head learning rate')
    ap.add_argument('--augment',type=int,default=2)
    ap.add_argument('--pretrained',default=DEFAULT_PRETRAINED, help='Synthetic-pretrained checkpoint from train_all.py')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--cache-dir',default=os.path.join(ROOT,'real_data_cache'))
    ap.add_argument('--weights-out',default=DEFAULT_OUT)
    ap.add_argument('--holdout',default=','.join(map(str,sorted(DEFAULT_HOLDOUT))))
    args=ap.parse_args(); holdout={int(x) for x in args.holdout.split(',') if x.strip()}
    print('=== v9 REAL-TESS DATASET ===')
    pos,neg=build_catalog(args.max_per_class,holdout)
    metas=pos+neg
    print(f'Catalog candidates: {len(pos)} planet TICs, {len(neg)} false-positive/alarm TICs')
    # Download sequentially to avoid hammering MAST and to make failures reproducible.
    good=[]
    for i,m in enumerate(metas,1):
        print(f'[{i}/{len(metas)}] TIC {m["tic"]} label={m["label"]}')
        prepared = prepare_target(m, args.cache_dir)

        if prepared is not None:
            _, _, prepared_meta = prepared
            good.append(prepared_meta)
    train_meta,val_meta=split_by_tic(good,args.val_frac,args.seed)
    print(f'Real target split: train={len(train_meta)}, val={len(val_meta)}')
    Gtr,Ltr,ytr,_=load_cache(args.cache_dir,train_meta); Gv,Lv,yv,_=load_cache(args.cache_dir,val_meta)
    Gaug,Laug=tensors(Gtr,Ltr,args.augment,args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'; print('Device:',device)
    model=TriBranchTESSNet()
    if not os.path.exists(args.pretrained):
        raise FileNotFoundError(
            f'Synthetic pretrained checkpoint not found: {args.pretrained}\n'
            'Run `python -m root.train_all --n-samples 1200 --epochs 30` first.'
        )
    print(f'Loading synthetic pretrained weights: {args.pretrained}')
    pre=load_checkpoint(args.pretrained)
    state=pre.get('state_dict',pre) if isinstance(pre,dict) else pre
    missing,unexpected=model.load_state_dict(state,strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Pretrained checkpoint is incompatible. Missing={missing}, unexpected={unexpected}')
    print('Pretrained morphology weights loaded successfully.')
    model,best_auc,best_epoch=train(model,Gaug,Laug,np.tile(ytr,args.augment+1),Gv,Lv,yv,args.epochs,args.lr,args.head_lr,args.batch_size,device)
    before,tv=metrics(model,Gv,Lv,yv,device,1.0)
    if len(yv) >= 10 and np.sum(yv == 0) >= 4 and np.sum(yv == 1) >= 4:
        temp=fit_temperature(model,Gv,Lv,yv,device)
        after,_=metrics(model,Gv,Lv,yv,device,temp)
    else:
        print('Validation set too small for temperature calibration; using temperature=1.0')
        temp=1.0
        after=before
    print('Validation before calibration:',before); print(f'Temperature={temp:.4f}'); print('Validation after calibration:',after)
    os.makedirs(os.path.dirname(args.weights_out),exist_ok=True)
    checkpoint={'format_version':6,'model_contract':'v9_real_tess_morphology_only_finetuned_from_synthetic','view_normalization':'per_sample_zscore','normalization_layer':'GroupNorm','snn_temporal_sequence':61,'temperature':temp,'pretrained_checkpoint':os.path.abspath(args.pretrained),'best_validation_auc':float(best_auc),'best_validation_epoch':int(best_epoch),'real_tess_validation':after,'real_tess_validation_uncalibrated':before,'train_tics':[m['tic'] for m in train_meta],'validation_tics':[m['tic'] for m in val_meta],'excluded_holdout_tics':sorted(holdout),'catalog_source':'NASA Exoplanet Archive TOI table; CP/KP=planet, FP/FA=false-positive/alarm','state_dict':model.cpu().state_dict()}
    torch.save(checkpoint,args.weights_out)
    with open(args.weights_out+'.json','w') as f: json.dump({k:v for k,v in checkpoint.items() if k!='state_dict'},f,indent=2)
    print('Saved:',args.weights_out)

if __name__=='__main__': main()

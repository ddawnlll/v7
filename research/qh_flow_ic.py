#!/usr/bin/env python3
"""qh_flow_ic.py — single-variable IC: quarter-hour aggressor flow vs forward returns.
Reads parquet→numpy. No model, no sklearn, no synthetic data. Output: table + JSON + self-eval."""
import argparse, json, sys, math
from pathlib import Path
import numpy as np
def _norm(c): return c.strip().lower().replace(" ","_").replace("-","_")
def _pick(dfc, aliases):
    m = {_norm(c): c for c in dfc}
    for a in aliases:
        if _norm(a) in m: return m[_norm(a)]
    raise KeyError(f"missing {aliases[0]}; have {list(dfc)}")
def _load_trades(path):
    import pandas as pd
    df = pd.read_parquet(path); c = df.columns
    ts  = df[_pick(c, ["ts_ms","timestamp_ms","timestamp","ts","time","trade_time"])].to_numpy(np.int64)
    px  = df[_pick(c, ["price","px","trade_price"])].to_numpy(np.float64)
    sz  = df[_pick(c, ["size","qty","quantity","amount","base_volume"])].to_numpy(np.float64)
    raw = df[_pick(c, ["side","aggressor_side","taker_side","trade_side","is_buyer_maker"])]
    if hasattr(raw.iloc[0],"__bool__") and not isinstance(raw.iloc[0],str):
        ss = np.where(raw.to_numpy(bool), -1.0, 1.0)
    else:
        s = raw.astype(str).str.strip().str.lower().to_numpy()
        ss = np.where(np.isin(s, ["buy","b","buyer","bid","long","1","+1"]), 1.0, -1.0)
    o = np.argsort(ts); return ts[o], px[o], sz[o], ss.astype(np.float64)[o]
def _load_book(path):
    import pandas as pd
    df = pd.read_parquet(path); c = df.columns
    ts = df[_pick(c, ["ts_ms","timestamp_ms","timestamp","ts","time"])].to_numpy(np.int64)
    td = np.zeros(len(df), np.float64); m = {_norm(k): k for k in c}
    for side in ("bid","ask"):
        for lv in range(1, 6):
            for al in [f"{side}_sz{lv}",f"{side}_size{lv}",f"{side}{lv}_size",f"{side[0]}{lv}_sz"]:
                if _norm(al) in m: td += df[m[_norm(al)]].to_numpy(np.float64); break
    o = np.argsort(ts); return ts[o], td[o]
def _load_snap(snap_dir):
    import pandas as pd
    p = Path(snap_dir); pq = p/"trade_bars_5m.parquet" if p.is_dir() else p
    if not pq.exists(): raise FileNotFoundError(str(pq))
    df = pd.read_parquet(pq)
    ts = df["open_ts"].to_numpy(np.int64)
    mid = (df["high"].to_numpy(np.float64) + df["low"].to_numpy(np.float64)) / 2.0
    o = np.argsort(ts); return ts[o], mid[o]

def _ic_table(fz, rd, re, hlabs, ml):
    rows = []
    for j, lab in enumerate(hlabs):
        fj = np.isfinite(fz); rdc = rd[:,j]; rec = re[:,j]
        md = fj & np.isfinite(rdc); me = fj & np.isfinite(rec)
        # Spearman inline
        def sp(x,y):
            m=np.isfinite(x)&np.isfinite(y); n=m.sum()
            if n<5: return np.nan,n
            rx=np.argsort(np.argsort(x[m])).astype(np.float64)
            ry=np.argsort(np.argsort(y[m])).astype(np.float64)
            xc=rx-np.mean(rx); yc=ry-np.mean(ry)
            d=np.sqrt(np.sum(xc*xc)*np.sum(yc*yc))
            return (np.sum(xc*yc)/d,n) if d>1e-15 else (np.nan,n)
        r_d, nd = sp(fz[md], rdc[md])
        r_e, ne = sp(fz[me], rec[me])
        p = np.nan
        if md.sum() >= 10 and not np.isnan(r_d):
            xm, ym = fz[md], rdc[md]
            rx = np.argsort(np.argsort(xm)).astype(np.float64)
            ry = np.argsort(np.argsort(ym)).astype(np.float64)
            rx = (rx - np.mean(rx)) / np.std(rx)
            ry = (ry - np.mean(ry)) / np.std(ry)
            infl = rx*ry - r_d*0.5*(rx*rx + ry*ry)
            n_ = len(infl); ml2 = min(ml or int(np.floor(4*(n_/100.0)**(2.0/9.0))), n_-2)
            lrv = np.mean(infl**2)
            for k in range(1, ml2+1):
                w = 1.0 - k/(ml2+1); lrv += 2.0*w*np.mean(infl[k:]*infl[:-k])
            se = math.sqrt(lrv/n_)
            if se > 1e-15: p = 2.0*(1.0 - 0.5*(1.0 + math.erf(abs(r_d)/se/math.sqrt(2.0))))
        ratio = r_e/r_d if (not np.isnan(r_d) and abs(r_d) > 1e-10) else np.nan
        rows.append({"horizon":lab,
                     "IC_decision":round(float(r_d),6) if not np.isnan(r_d) else None,
                     "IC_entry":round(float(r_e),6) if not np.isnan(r_e) else None,
                     "n":int(nd),
                     "p_value":round(float(p),6) if not np.isnan(p) else None,
                     "IC_entry/IC_decision":round(float(ratio),4) if not np.isnan(ratio) else None})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades",required=True); ap.add_argument("--book",required=True)
    ap.add_argument("--snapshot",required=True); a = ap.parse_args()
    tp = Path(a.trades.split("=",1)[1]); bp = Path(a.book.split("=",1)[1])
    sp = Path(a.snapshot.split("=",1)[1])
    print("Loading...", file=sys.stderr)
    t_ts, t_px, t_sz, t_side = _load_trades(tp)
    b_ts, b_depth = _load_book(bp)
    bar_ts, bar_mid = _load_snap(sp)
    QMS=900_000; WMS=10_000; EMS=300_000; PPD=96; RD=30
    HLAB=["5m","15m","1h","4h","8h"]
    HMS=[5*60_000,15*60_000,60*60_000,4*60*60_000,8*60*60_000]
    t0=max(t_ts[0],bar_ts[0]); t1=min(t_ts[-1],bar_ts[-1])
    fq=((t0//QMS)+1)*QMS; lq=(t1//QMS)*QMS
    qs=np.arange(fq,lq+1,QMS,dtype=np.int64); N=len(qs)
    print(f"Quarters: {N}", file=sys.stderr)
    flow=np.full(N,np.nan,np.float64); depth=np.full(N,np.nan,np.float64)
    for i in range(N):
        a0=np.searchsorted(t_ts,qs[i],"left"); b0=np.searchsorted(t_ts,qs[i]+WMS,"left")
        if b0>a0: flow[i]=float(np.sum(t_px[a0:b0]*t_sz[a0:b0]*t_side[a0:b0]))
        jj=int(np.searchsorted(b_ts,qs[i],"right"))-1
        if 0<=jj<len(b_ts): depth[i]=float(b_depth[jj])
    W=RD*PPD; fz=np.full(N,np.nan,np.float64); dz=np.full(N,np.nan,np.float64)
    for s,o in ((flow,fz),(depth,dz)):
        for i in range(N):
            if i<2 or np.isnan(s[i]): continue
            h=s[max(0,i-W):i]; h=h[np.isfinite(h)]
            if len(h)<2: continue
            mu,sd=np.mean(h),np.std(h)
            if sd>1e-15: o[i]=(s[i]-mu)/sd
    rd=np.full((N,len(HMS)),np.nan,np.float64); re=np.full((N,len(HMS)),np.nan,np.float64)
    def _mid(bt,bm,t):
        ii=int(np.searchsorted(bt,t,side="right"))-1
        return bm[ii] if 0<=ii<len(bt) else np.nan
    for i in range(N):
        dt=qs[i]+WMS; et=qs[i]+EMS
        md=_mid(bar_ts,bar_mid,dt); me=_mid(bar_ts,bar_mid,et)
        if np.isnan(md) or md<=0 or np.isnan(me) or me<=0: continue
        for j,hms in enumerate(HMS):
            mdf=_mid(bar_ts,bar_mid,dt+hms); mef=_mid(bar_ts,bar_mid,et+hms)
            if not np.isnan(mdf) and mdf>0: rd[i,j]=mdf/md-1.0
            if not np.isnan(mef) and mef>0: re[i,j]=mef/me-1.0
    ml=max(HMS)//300_000
    tab=_ic_table(fz,rd,re,HLAB,ml)
    med=np.nanmedian(dz); hi=dz>=med; lo=~hi
    reg={"high_depth":_ic_table(fz[hi],rd[hi],re[hi],HLAB,ml),
         "low_depth":_ic_table(fz[lo],rd[lo],re[lo],HLAB,ml)}
    # output
    hdr=f"{'horizon':>6}  {'IC_dec':>10}  {'IC_ent':>10}  {'n':>6}  {'p_val':>10}  {'IC_e/IC_d':>10}"
    print(hdr); print("-"*len(hdr))
    for r in tab:
        def f(v): return "       nan" if v is None else (f"{v:10.6f}" if isinstance(v,float) else f"{v:>10}")
        print(f"{r['horizon']:>6}  {f(r['IC_decision'])}  {f(r['IC_entry'])}  {f(r['n'])}  {f(r['p_value'])}  {f(r['IC_entry/IC_decision'])}")
    for label,rows in [("high_depth",reg["high_depth"]),("low_depth",reg["low_depth"])]:
        print(f"\n--- {label} ---")
        for r in rows: print(f"  {r['horizon']}: IC_d={r['IC_decision']} IC_e={r['IC_entry']} n={r['n']} p={r['p_value']}")
    print(json.dumps({"ic_table":tab,"regime":reg},indent=2,default=str))
    # success criteria
    ic5=tab[0]["IC_entry"]; ic8=tab[-1]["IC_entry"]; rat=tab[0]["IC_entry/IC_decision"]
    print("\n=== SUCCESS CRITERIA ===")
    def v(c,msg): return f"[{'PASS' if c else 'FAIL'}] {msg}"
    print(v(ic5 is not None and abs(ic5)<0.01,f"IC_entry(5m)={ic5} < 0.01"))
    print(v(ic8 is not None and abs(ic8)<0.01,f"IC_entry(8h)={ic8} < 0.01"))
    print(v(rat is not None and rat<0.3,f"IC_entry/IC_decision={rat} < 0.3"))
if __name__=="__main__": main()

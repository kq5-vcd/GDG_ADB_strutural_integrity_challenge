"""Step 1: Fast feature extraction (optimized)"""
import os, csv, time, json
import numpy as np
from collections import defaultdict

BASE = "/sessions/bold-magical-dirac/mnt/GDG-Hackaton/Structural.integrity.Seonsor.Board.data/Sensor Board Update Initial Test"
OUT = "/sessions/bold-magical-dirac/mnt/GDG-Hackaton/classifier_results"
os.makedirs(OUT, exist_ok=True)

FS = 27000; N = 8192
freqs = np.fft.rfftfreq(N, 1/FS)
mask_full = (freqs >= 50) & (freqs <= 4000)
mask_low = (freqs >= 50) & (freqs < 200)
mask_mid = (freqs >= 200) & (freqs < 1000)
mask_high = (freqs >= 1000) & (freqs < 4000)
freqs_band = freqs[mask_full]

# Load metadata
with open(f"{BASE}/Test.csv") as f:
    meta_rows = list(csv.DictReader(f))
for r in meta_rows:
    r["Bolt Status"] = r["Bolt Status"].replace("Mix- 45 Deg", "Mix-45 Deg")

test_meta = {}
for r in meta_rows:
    sf1, ef1 = r["Start Freq 1"], r["End Freq 1"]
    sf2, ef2 = r.get("Start Freq 2",""), r.get("End Freq 2","")
    pat = f"{sf1}-{ef1}Hz"
    if sf2: pat += f"+{sf2}-{ef2}Hz"
    test_meta[(r["Run"], r["Test"])] = pat

def extract_features(data):
    feats = []
    fft_mags = []
    for ax in range(3):
        s = data[:, ax]; sc = s - s.mean()
        fft_m = np.abs(np.fft.rfft(sc))
        fft_mags.append(fft_m)
        rms = np.sqrt(np.mean(sc**2))
        fb = fft_m[mask_full]
        
        # Time domain
        feats.extend([
            rms,
            np.std(s),
            s.mean(),
            np.mean(sc**4)/(np.mean(sc**2)**2+1e-10),  # kurtosis
            np.mean(sc**3)/(np.std(sc)**3+1e-10),  # skewness
            np.max(np.abs(sc))/(rms+1e-10),  # crest
            np.max(s)-np.min(s),  # p2p
            np.sum(np.abs(np.diff(np.sign(sc))))/(2*N),  # zcr
        ])
        
        # Frequency domain
        centroid = np.sum(freqs_band*fb)/(np.sum(fb)+1e-10)
        feats.extend([
            centroid,
            np.sqrt(np.sum((freqs_band-centroid)**2*fb)/(np.sum(fb)+1e-10)),  # spread
            freqs_band[np.argmax(fb)] if len(fb)>0 else 0,  # peak freq
        ])
        
        # Rolloff
        cs = np.cumsum(fb**2); tot = cs[-1] if len(cs)>0 else 1e-10
        ri = np.searchsorted(cs, 0.95*tot)
        feats.append(freqs_band[min(ri, len(freqs_band)-1)] if len(freqs_band)>0 else 0)
        
        # Flatness
        lm = np.mean(np.log(fb+1e-10))
        feats.append(np.exp(lm)/(np.mean(fb)+1e-10))
        
        # Band energies
        el = np.sum(fft_m[mask_low]**2)
        em = np.sum(fft_m[mask_mid]**2)
        eh = np.sum(fft_m[mask_high]**2)
        et = el+em+eh+1e-10
        feats.extend([el/et, em/et, eh/et, et, (el+1e-10)/(eh+1e-10), (em+1e-10)/(eh+1e-10)])
        
        # Half-power bandwidth
        pk = np.max(fb) if len(fb)>0 else 0
        hp = pk/np.sqrt(2)
        above = fb >= hp
        if np.any(above):
            idx_a = np.where(above)[0]
            bw = freqs_band[idx_a[-1]]-freqs_band[idx_a[0]]
        else:
            bw = 0
        pf = freqs_band[np.argmax(fb)] if len(fb)>0 else 0
        feats.extend([bw, pf/(bw+1e-10)])  # bw, Q factor
    
    # Cross-axis features
    for a1, a2 in [(0,1),(0,2),(1,2)]:
        s1 = data[:,a1]-data[:,a1].mean()
        s2 = data[:,a2]-data[:,a2].mean()
        feats.append(np.corrcoef(s1,s2)[0,1])  # correlation
        feats.append(np.sqrt(np.mean(s1**2))/(np.sqrt(np.mean(s2**2))+1e-10))  # rms ratio
        feats.append(np.corrcoef(fft_mags[a1][mask_full], fft_mags[a2][mask_full])[0,1])  # spectral corr
    
    # Magnitude
    mag = np.sqrt(np.sum(data**2, axis=1))
    mc = mag-mag.mean()
    feats.extend([np.sqrt(np.mean(mc**2)), np.mean(mc**4)/(np.mean(mc**2)**2+1e-10)])
    
    return np.array(feats, dtype=np.float64)

# Feature names (must match extract_features order)
fn = []
for ax in ["X","Y","Z"]:
    fn += [f"{ax}_rms",f"{ax}_std",f"{ax}_mean",f"{ax}_kurtosis",f"{ax}_skewness",
           f"{ax}_crest",f"{ax}_p2p",f"{ax}_zcr",
           f"{ax}_centroid",f"{ax}_spread",f"{ax}_peak_freq",f"{ax}_rolloff",f"{ax}_flatness",
           f"{ax}_e_low",f"{ax}_e_mid",f"{ax}_e_high",f"{ax}_e_total",f"{ax}_elh_ratio",f"{ax}_emh_ratio",
           f"{ax}_hp_bw",f"{ax}_q_factor"]
for a1,a2 in [("X","Y"),("X","Z"),("Y","Z")]:
    fn += [f"corr_{a1}{a2}", f"rms_ratio_{a1}{a2}", f"spec_corr_{a1}{a2}"]
fn += ["mag_rms", "mag_kurtosis"]

print(f"Features per sample: {len(fn)}")

# Process all files
t0 = time.time()
X_all = []; y_all = []; meta_all = []
label_map = {"A":0,"B":1,"C":2}
processed = 0; skipped = 0

for run in ["A","B","C"]:
    rp = f"{BASE}/{run}"
    for td in sorted(os.listdir(rp)):
        tp = os.path.join(rp, td)
        if not os.path.isdir(tp): continue
        key = (run, td)
        if key not in test_meta: skipped += 1; continue
        pat = test_meta[key]
        
        for board in os.listdir(tp):
            bp = os.path.join(tp, board)
            if not os.path.isdir(bp): continue
            files = [f for f in os.listdir(bp) if f.endswith(".csv") and not f.endswith("_i.csv")]
            if not files: continue
            fp = os.path.join(bp, sorted(files)[0])
            try:
                data = np.genfromtxt(fp, delimiter=",", skip_header=1, usecols=(1,2,3))
                if data.shape[0] != N: skipped += 1; continue
                if np.any(np.isnan(data)): skipped += 1; continue
                f_vec = extract_features(data)
                if np.any(~np.isfinite(f_vec)): f_vec[~np.isfinite(f_vec)] = 0
                X_all.append(f_vec)
                y_all.append(label_map[run])
                meta_all.append({"run":run,"test":td,"board":board,"pattern":pat})
                processed += 1
            except:
                skipped += 1
        
        if processed % 500 == 0 and processed > 0:
            print(f"  {processed} processed, {time.time()-t0:.1f}s elapsed")

X = np.array(X_all)
y = np.array(y_all)

print(f"\nDone: {processed} samples, {skipped} skipped in {time.time()-t0:.1f}s")
print(f"X shape: {X.shape}, classes: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 2={np.sum(y==2)}")

np.save(f"{OUT}/X_features.npy", X)
np.save(f"{OUT}/y_labels.npy", y)
with open(f"{OUT}/meta_info.json","w") as f: json.dump(meta_all, f)
with open(f"{OUT}/feature_names.json","w") as f: json.dump(fn, f)
print("Saved to", OUT)

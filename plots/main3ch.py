import sys
sys.path.append(".")
import os
import json
import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime, timezone, timedelta

from utils.normalizer import Normalizer
from utils.interp import (
    load_data_meta_from_npz,
    nearest_grid_indices,
    sample_field_nearest,
    interpolate_linear,
    interpolate_cubic_hermite,
)
from utils.utils import iter_filelist

# ---------------- helpers (unchanged) ----------------

def _to_utc_epoch_seconds(dt) -> int:
    ts = pd.to_datetime(dt, utc=True)
    return int(ts.timestamp())

def get_obs_df_for_date(obs: pd.DataFrame, date) -> pd.DataFrame:
    if "validdate" not in obs.columns:
        raise KeyError("DataFrame must contain a 'validdate' column.")
    vd = obs["validdate"].to_numpy()
    is_ms = np.nanmax(vd) > 1e12
    vd_sec = (vd // 1000).astype(np.int64) if is_ms else vd.astype(np.int64)
    target_epoch = _to_utc_epoch_seconds(date)
    mask = (vd_sec == target_epoch)
    out = obs.loc[mask].copy()
    out["validdate_utc"] = pd.to_datetime(vd_sec[mask], unit="s", utc=True)
    return out.reset_index(drop=True)

def attach_sample_column(valid_stations: pd.DataFrame, samples, colname: str) -> pd.DataFrame:
    df = valid_stations.reset_index(drop=True).copy()
    vals = np.asarray(samples)
    if len(vals) != len(df):
        raise ValueError(f"Length mismatch: samples={len(vals)} vs valid_stations={len(df)}")
    df[colname] = vals
    return df

def get_info(stations_file, latlon_file):
    _df = pd.read_csv(stations_file, parse_dates=["validdate"])
    _stations = _df.loc[:, ["SID", "lat", "lon"]].drop_duplicates(subset=["SID"], keep="first")
    _stations = _stations.sort_values("SID").reset_index(drop=True)

    latlon = np.load(latlon_file)
    lat2d = latlon[0]
    lon2d = latlon[1]

    # nearest grid selection (2000 m cap) -> mask invalid then recompute on filtered stations
    interp_info = nearest_grid_indices(lat2d, lon2d, _stations["lat"], _stations["lon"], max_dist_m=2000)
    _mask = np.asarray(interp_info["valid"], dtype=bool)
    valid_stations = _stations.loc[_mask].drop_duplicates(subset="SID").reset_index(drop=True)
    interp_info = nearest_grid_indices(lat2d, lon2d, valid_stations["lat"], valid_stations["lon"], max_dist_m=2000)
    return interp_info, valid_stations, lat2d, lon2d

# ---------------- config ----------------

normalizer = Normalizer(
    stats_npz="/home/users/u101329/p200177_t1_hp/u101329/npy_interp/sequences-stats-2023-2024.npz",
    mode="symrange",
    average_key="global_mean",
    norm_const=0.95,
    channel_indices=[1, 2, 3],
)

print("start")
latlon_file = "/home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep/latlon.npy"
obs_file = "/home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep/2024-m10-m12-obs.csv"
stations_file = "/home/users/u101329/p200177_t2/u101329/diffusion-interp/_saved/stations.csv"

# Load unique stations & grid info
interp_info, valid_stations, lat2d, lon2d = get_info(stations_file, latlon_file)
obs_all = pd.read_csv(obs_file)

# channel indices (GT & prediction tensors)
guch, gvch, gtch = 1, 2, 3   # GT: U10, V10, T2m channels in your stored GT
puch, pvch, ptch = 0, 1, 2   # Pred: U10, V10, T2m channels in prediction
# ---- expected hours from windows: 'A-B' => range(A, B) (B excluded)
def parse_window(win: str) -> range:
    a, b = win.split("-")
    a, b = int(a), int(b)
    if b < a:
        raise ValueError(f"Invalid window '{win}': end < start")
    return range(a, b+1)  # end excluded
for member in range(0, 1):
    merged_big = None
    print(f"processing member {member}")
    samples_file = f"/home/users/u101329/p200177_t2/u101329/diffusion-interp/_saved/new_attempt/m{member}_npz_files.txt"

    for p in iter_filelist(samples_file):
        # ----- load sample -----
        npz_path = os.path.join("/home/users/u101329/p200177_t1_hp/u101329/results/samples_utv_2024_2", p)
        pred, meta, gt = load_data_meta_from_npz(npz_path)  # pred: (T,C,H,W), gt: (T+2,C,H,W) or similar

        window = meta["window"]
        leads = parse_window(window)  # validate window format
        member = meta["member"]

   
        # T internal frames between endpoints
        T_internal = int(pred.shape[0])
        internal_range = range(T_internal)

        # ---- denormalize pred by frame to (H,W) values ----
        pred = np.array([normalizer.denormalize(pred[i, ...]) for i in internal_range])  # (T, C, H, W)

        # ---- split GT & Pred into U,V,S,T stacks ----
        gt_u_3d = gt[:, guch, :, :]
        gt_v_3d = gt[:, gvch, :, :]
        gt_s_3d = np.sqrt(gt_u_3d * gt_u_3d + gt_v_3d * gt_v_3d)  # (T+2?, H, W)
        gt_t_3d = gt[:, gtch, :, :]

        pred_u_3d = pred[:, puch, :, :]
        pred_v_3d = pred[:, pvch, :, :]
        pred_s_3d = np.sqrt(pred_u_3d * pred_u_3d + pred_v_3d * pred_v_3d)  # (T, H, W)
        pred_t_3d = pred[:, ptch, :, :]

        # ---- sample endpoints (GT) at stations ----
        gt_s_start_1d = sample_field_nearest(gt_s_3d[0], interp_info)      # (Nstations,)
        gt_s_end_1d   = sample_field_nearest(gt_s_3d[-1], interp_info)
        gt_t_start_1d = sample_field_nearest(gt_t_3d[0], interp_info)
        gt_t_end_1d   = sample_field_nearest(gt_t_3d[-1], interp_info)

        # ---- build interpolated internals between endpoints (station space) ----
        # linear / cubic Hermite for S10m
        gt_s_interpL_internal_2d = interpolate_linear(gt_s_start_1d, gt_s_end_1d).squeeze(1)      # (T, N)
        gt_s_interpC_internal_2d = interpolate_cubic_hermite(gt_s_start_1d, gt_s_end_1d).squeeze(1)
        # linear / cubic Hermite for T2m
        gt_t_interpL_internal_2d = interpolate_linear(gt_t_start_1d, gt_t_end_1d).squeeze(1)      # (T, N)
        gt_t_interpC_internal_2d = interpolate_cubic_hermite(gt_t_start_1d, gt_t_end_1d).squeeze(1)

        # ---- sample predicted internals at stations ----
        pred_s_2d = np.array([sample_field_nearest(pred_s_3d[i], interp_info) for i in internal_range])  # (T, N)
        pred_t_2d = np.array([sample_field_nearest(pred_t_3d[i], interp_info) for i in internal_range])  # (T, N)

        # ---- AUGMENT with GT endpoints (T+2, N) ----
        s_L_aug = np.vstack([gt_s_start_1d[None, :], gt_s_interpL_internal_2d, gt_s_end_1d[None, :]])
        s_C_aug = np.vstack([gt_s_start_1d[None, :], gt_s_interpC_internal_2d, gt_s_end_1d[None, :]])
        s_P_aug = np.vstack([gt_s_start_1d[None, :], pred_s_2d,               gt_s_end_1d[None, :]])

        t_L_aug = np.vstack([gt_t_start_1d[None, :], gt_t_interpL_internal_2d, gt_t_end_1d[None, :]])
        t_C_aug = np.vstack([gt_t_start_1d[None, :], gt_t_interpC_internal_2d, gt_t_end_1d[None, :]])
        t_P_aug = np.vstack([gt_t_start_1d[None, :], pred_t_2d,               gt_t_end_1d[None, :]])

        # ---- timeline of length T+2 (do NOT assume strictly 1h; compute from endpoints) ----
        start_valid_time = datetime.fromisoformat(meta["start_valid_time"].replace("Z", "+00:00"))
        end_valid_time   = datetime.fromisoformat(meta["end_valid_time"].replace("Z", "+00:00"))
        the_date = datetime.fromisoformat(meta["date"].replace("Z", "+00:00"))
        if T_internal <= 0:
            continue
        
        delta = timedelta(hours=1)
        times_utc = [start_valid_time + i * delta for i in range(T_internal + 2)]
        hours = [ts.hour for ts in times_utc]
        dates_str = [ts.strftime("%Y-%m-%d") for ts in times_utc]
        is_endpoint_flags = [True] + [False] * T_internal + [True]
        # ---- iterate all frames including endpoints ----
        for k, ts in enumerate(times_utc):
            # assemble per-time-frame station DataFrames
            lt2m = attach_sample_column(valid_stations, t_L_aug[k], "lt2m")  # GT-linear incl endpoints
            ct2m = attach_sample_column(valid_stations, t_C_aug[k], "ct2m")  # GT-cubic  incl endpoints
            pt2m = attach_sample_column(valid_stations, t_P_aug[k], "pt2m")  # Pred internals + GT endpoints

            ls10m = attach_sample_column(valid_stations, s_L_aug[k], "ls10m")
            cs10m = attach_sample_column(valid_stations, s_C_aug[k], "cs10m")
            ps10m = attach_sample_column(valid_stations, s_P_aug[k], "ps10m")

            dfs = [lt2m, ct2m, pt2m, ls10m, cs10m, ps10m]
            _merged = reduce(
                lambda left, right: pd.merge(left, right, on=["SID", "lat", "lon"], how="inner"),
                dfs,
            )

            n = len(_merged)
            _merged["window"] = [window] * n
            _merged["member"] = [member] * n
            _merged["validdate_utc"] = [ts] * n
            _merged["date"] = [dates_str[k]] * n
            _merged["hour"] = [hours[k]] * n
            _merged["lead"] = [leads[k]] * n  # expected lead hours from window
            _merged["is_endpoint"] = [is_endpoint_flags[k]] * n
            _merged["start_date"] = [the_date] * n


            # observations at this exact time
            obs_at_time = get_obs_df_for_date(obs_all, ts)

            merged = pd.merge(
                _merged,
                obs_at_time,
                on=["SID", "lat", "lon"],
                how="inner",
                validate="one_to_one",
            )

            if merged_big is None:
                merged_big = merged
            else:
                merged_big = pd.concat([merged_big, merged], ignore_index=True)

    # write once per member
    out_csv = f"/home/users/u101329/p200177_t2/u101329/diffusion-interp/_saved/new_attempt/all_in_one_m{member}-v3.csv"
    merged_big.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

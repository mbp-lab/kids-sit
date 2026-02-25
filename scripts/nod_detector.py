import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt


class NodDetector:
    def __init__(
        self, 
        smooth_win=5,
        smooth_poly=2,
        T_min=0.2,
        T_max=1.5,
        dom_ratio=2.0,
        k_std_velocity=1.0,
        k_std_still_pitch=4.0,
        
    ):
        self.smooth_win = smooth_win if smooth_win % 2 == 1 else smooth_win + 1
        self.smooth_poly = smooth_poly
        self.T_min = T_min
        self.T_max = T_max
        self.dom_ratio = dom_ratio
        self.k_std_velocity = k_std_velocity
        self.k_std_still_pitch = k_std_still_pitch
        

    # -------------------------
    # Post-processing
    # -------------------------
    def post_process(self, events, fps):
        if not events:
            return []

        events.sort()
        gap = int(self.T_min * fps)

        merged = [events[0]]
        for s, e in events[1:]:
            ps, pe = merged[-1]
            if s <= pe + gap:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))

        return merged

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _mad_sigma(x):
        x = x[np.isfinite(x)]
        if len(x) < 10:
            print("Warning: not enough valid data points to compute MAD.")
            return np.nan
        m = np.median(x)
        return 1.4826 * np.median(np.abs(x - m))
    

    # -------------------------
    # Main detection method
    # -------------------------

    def detect_nod(self, pitch, yaw, fps, radians):
        pitch = np.asarray(pitch, float)
        yaw   = np.asarray(yaw, float)

        trace = {"fps": fps, "radians": radians}

        # ---- step 0: input ----
        trace["pitch_in"] = pitch.copy()
        trace["yaw_in"] = yaw.copy()

        # ---- step 1: radians->deg (+ your sign flip) ----
        if radians:
            pitch = np.degrees(pitch) * -1 #here: change sign only for OpenFace (PyAFAR already in head coordinates)
            yaw   = np.degrees(yaw) * -1
        trace["pitch_deg"] = pitch.copy()
        trace["yaw_deg"] = yaw.copy()

        # ---- step 2: smoothing ----
        p = savgol_filter(pitch, self.smooth_win, self.smooth_poly)
        y = savgol_filter(yaw, self.smooth_win, self.smooth_poly)
        trace["pitch_smooth"] = p
        trace["yaw_smooth"] = y

        # ---- step 3: velocity ----
        v = np.gradient(p) * fps
        v_abs = np.abs(v)
        trace["v"] = v
        trace["v_abs"] = v_abs

        # ---- step 4: MAD sigma for velocity + still mask ----
        sigma_v = self._mad_sigma(v_abs)
        med_v = np.median(v_abs[np.isfinite(v_abs)])
        v_still_thr = med_v + self.k_std_velocity * sigma_v
        still = v_abs < v_still_thr

        trace["sigma_v"] = sigma_v
        trace["med_v"] = med_v
        trace["v_still_thr"] = v_still_thr
        trace["still_mask"] = still

        # ---- step 5: MAD sigma for pitch in still + A_min + prom ----
        sigma_p = self._mad_sigma(p[still])
        A_min = self.k_std_still_pitch * sigma_p

        trace["sigma_p_still"] = sigma_p
        trace["A_min"] = A_min

        # ---- step 6: troughs only from pitch ----
        min_dist = max(1, int(0.15 * fps))
        troughs, _ = find_peaks(-p, distance=min_dist)

        trace["min_dist_frames"] = min_dist
        trace["troughs"] = troughs

        # half-window in frames (T_max/2)
        W = max(1, int((self.T_max / 2.0) * fps))
        trace["half_window_frames"] = W

        # ---- step 7: candidates by global max in left/right windows ----
        triplets = []
        raw_events = []

        for b in troughs:
            b = int(b)

            # define windows
            L0 = max(0, b - W)
            L1 = b
            R0 = b
            R1 = min(len(p) - 1, b + W)

            # need at least 2 samples each side
            if L1 <= L0 or R1 <= R0:
                triplets.append({"a": None, "b": b, "c": None,
                                "passed": False, "fail_reason": "window_too_small"})
                continue

            # global maxima on each side
            a = int(L0 + np.argmax(p[L0:L1+1]))
            c = int(R0 + np.argmax(p[R0:R1+1]))

            rec = {
                "a": a, "b": b, "c": c,
                "L0": int(L0), "L1": int(L1), "R0": int(R0), "R1": int(R1)
            }

            # duration gate (between peaks)
            dur = (c - a) / fps
            rec["dur_s"] = float(dur)
            dur_ok = (self.T_min <= dur <= self.T_max)
            rec["dur_ok"] = bool(dur_ok)

            # amplitude gate relative to trough
            down_mag = float(p[a] - p[b])
            up_mag   = float(p[c] - p[b])
            rec["down_mag"] = down_mag
            rec["up_mag"] = up_mag

            amp_ok = (down_mag >= A_min and up_mag >= A_min)
            rec["amp_ok"] = bool(amp_ok)

            # yaw dominance gate over [a,c]
            amp_pp = float(np.max(p[a:c+1]) - np.min(p[a:c+1]))
            yaw_amp = float(np.max(y[a:c+1]) - np.min(y[a:c+1]))
            rec["amp_pp"] = amp_pp
            rec["yaw_amp"] = yaw_amp

            yaw_ok = not (yaw_amp > 0 and amp_pp < self.dom_ratio * (yaw_amp + 1e-6))
            rec["yaw_ok"] = bool(yaw_ok)

            passed = dur_ok and amp_ok and yaw_ok
            rec["passed"] = bool(passed)

            if not dur_ok:
                rec["fail_reason"] = "duration"
            elif not amp_ok:
                rec["fail_reason"] = "amplitude"
            elif not yaw_ok:
                rec["fail_reason"] = "yaw_dominant"
            else:
                rec["fail_reason"] = ""

            triplets.append(rec)
            if passed:
                raw_events.append((a, c))

        trace["triplets"] = triplets
        trace["events_raw"] = raw_events

        # ---- step 8: merge ----
        trace["events_merged"] = self.post_process(raw_events, fps)

        return trace["events_merged"], trace


    def plot_trace(self, trace, title="Nod detection trace", show_triplet_labels=True):
        fps = trace["fps"]
        p_in = trace["pitch_in"]
        p_deg = trace["pitch_deg"]
        p = trace["pitch_smooth"]

        y_in = trace["yaw_in"]
        y_deg = trace["yaw_deg"]
        y = trace["yaw_smooth"]

        v_abs = trace["v_abs"]

        t = np.arange(len(p)) / fps

        still = trace["still_mask"]
        med_v = trace["med_v"]
        v_thr = trace["v_still_thr"]

        A_min = trace["A_min"]
        troughs = trace["troughs"]

        triplets = trace["triplets"]
        events_raw = trace["events_raw"]
        events_merged = trace["events_merged"]

        # ---------- figure ----------
        fig, axes = plt.subplots(8, 1, figsize=(18, 18), sharex=True)
        fig.suptitle(title)

        # 1) Raw pitch/yaw input
        axes[0].plot(t, p_in, label="pitch input")
        axes[0].plot(t, y_in, label="yaw input", alpha=0.7)
        axes[0].set_ylabel("raw")
        axes[0].legend(loc="upper right")
        axes[0].grid(True)

        # 2) Degrees-converted pitch/yaw
        axes[1].plot(t, p_deg, label="pitch deg")
        axes[1].plot(t, y_deg, label="yaw deg", alpha=0.7)
        axes[1].set_ylabel("deg")
        axes[1].legend(loc="upper right")
        axes[1].grid(True)

        # 3) Smoothed pitch/yaw
        axes[2].plot(t, p, label="pitch smooth")
        axes[2].plot(t, y, label="yaw smooth", alpha=0.7)
        axes[2].set_ylabel("smooth")
        axes[2].legend(loc="upper right")
        axes[2].grid(True)

        # 4) |velocity| + median + threshold + still shading
        axes[3].plot(t, v_abs, label="|v|")
        axes[3].axhline(med_v, linestyle="--", label=f"median |v| = {med_v:.3f}")
        axes[3].axhline(v_thr, linestyle="--", label=f"v_still_thr = {v_thr:.3f}")
        axes[3].fill_between(t, 0, np.nanmax(v_abs) if np.isfinite(v_abs).any() else 1.0,
                            where=still, alpha=0.15, step="pre", label="still")
        axes[3].set_ylabel("|deg/s|")
        axes[3].legend(loc="upper right")
        axes[3].grid(True)

        # 5) Smoothed pitch with troughs 
        axes[4].plot(t, p, label="pitch smooth")
        troughs = trace["troughs"]
        if len(troughs):
            axes[4].scatter(t[troughs], p[troughs], marker="x", s=35,color ='red', label="troughs")
        axes[4].set_ylabel("pitch")
        axes[4].legend(loc="upper right")
        axes[4].grid(True)

        # 6) Triplet gates visualization on pitch (color-coded by first failure reason)
        axes[5].plot(t, p, label="pitch smooth")

        reason_to_marker = {
            "shape": ("x", 35),
            "duration": ("o", 25),
            "amplitude": ("s", 25),
            "yaw_dominant": ("D", 25),
            "": ("*", 60),  # passed
        }

        for rec in triplets:
            a, b, c = rec["a"], rec["b"], rec["c"]
            passed = rec.get("passed", False)
            if a is None or b is None or c is None:
                continue

            # figure out reason key
            # expected: rec["fail_reason"] is one of the keys; if not present, infer from passed
            reason = rec.get("fail_reason", "" if passed else "shape")
            marker, size = reason_to_marker.get(reason, ("x", 35))

            # connecting polyline: solid if passed, dashed if failed
            axes[5].plot(
                [t[a], t[b], t[c]],
                [p[a], p[b], p[c]],
                linestyle="-" if passed else "--",
                alpha=0.7
            )

            # show a and c as before
            axes[5].scatter(t[a], p[a], marker="<", s=40)
            axes[5].scatter(t[c], p[c], marker=">", s=40)

            # show decision point b using reason_to_marker
            axes[5].scatter(t[b], p[b], marker=marker, s=size)
            # optional: label (only if you want)
            if show_triplet_labels:
                axes[5].text(t[b], p[b], reason, fontsize=8, va="bottom", ha="left", alpha=0.8)

        axes[5].set_ylabel("pitch")
        axes[5].grid(True)


        # 7) Amplitude gate shown explicitly per passed triplet (down_mag/up_mag vs A_min)
        # We plot bars at b: [p[b] -> p[a]] and [p[b] -> p[c]] for passed triplets
        axes[6].plot(t, p, alpha=0.6, label="pitch smooth")
        axes[6].text(0.01, 0.95, f"A_min = {A_min:.3f} deg (computed from still frames)",
                    transform=axes[4].transAxes, va="top")
        for rec in triplets:
            
            a, b, c = rec["a"], rec["b"], rec["c"]
            if a is None or c is None:
                continue
            # shade left/right search windows lightly
            axes[6].vlines(t[b], p[b], p[a], linewidth=2)
            axes[6].vlines(t[b], p[b], p[c], linewidth=2)
        axes[6].set_ylabel("pitch")
        axes[6].grid(True)

        # 8) Events: raw spans vs merged spans
        axes[7].plot(t, p, label="pitch smooth")
        for (s, e) in events_raw:
            axes[7].axvspan(t[s], t[e], alpha=0.15, label="raw event" if (s, e) == events_raw[0] else None)
        for (s, e) in events_merged:
            axes[7].axvspan(t[s], t[e], alpha=0.35, label="merged event" if (s, e) == events_merged[0] else None)
        gt = np.asarray(trace.get("nod_agreed", None))
        if gt is None:
            # If not provided in trace, try to use df_sub passed externally by setting trace['nod_agreed'] before calling plot_trace
            pass
        else:
            gt = (gt.astype(float) > 0).astype(int)

            # find contiguous segments of 1s
            idx = np.where(gt == 1)[0]
            if idx.size > 0:
                breaks = np.where(np.diff(idx) > 1)[0]
                starts = np.r_[idx[0], idx[breaks + 1]]
                ends = np.r_[idx[breaks], idx[-1]]

                for k, (s, e) in enumerate(zip(starts, ends)):
                    axes[7].axvspan(
                        t[s], t[e],
                        alpha=0.25,
                        hatch="//",
                        label="nod_agreed (GT)" if k == 0 else None
                    )
        axes[7].set_ylabel("pitch")
        axes[7].set_xlabel("time (s)")
        axes[7].legend(loc="upper right")
        axes[7].grid(True)

        plt.tight_layout()
        plt.show()

    
if __name__ == "__main__":

    df = pd.read_csv("...") # path to your dataframe with columns: participant_id, clip, frame, timestamp, pitch_openface, yaw_openface, pitch_pyafar, yaw_pyafar
    df = df.sort_values(['participant_id', 'clip', 'frame'], kind='mergesort')

    df['nod_openface'] = 0
    df['nod_pyafar'] = 0

    for (participant_id, clip), df_sub in df.groupby(['participant_id', 'clip'], sort=False):

        fps = (df_sub['frame'].iloc[-1] - df_sub['frame'].iloc[0]) / (
             df_sub['timestamp'].iloc[-1] - df_sub['timestamp'].iloc[0])

        # --- OPENFACE --- 
        detector = NodDetector(k_std_velocity=1.0,k_std_still_pitch=4.0) # found using grid search
        events_openface, _ = detector.detect_nod(
            pitch=df_sub['pitch_openface'].values,
            yaw=df_sub['yaw_openface'].values,
            fps=fps,
            radians=True
        )
        for start, end in events_openface:
            idx = df_sub.index[start:end+1]   
            df.loc[idx, 'nod_openface'] = 1

        # --- PYAFAR ---
        detector = NodDetector(k_std_velocity=1.0,k_std_still_pitch=4.0) # found using grid search
        events_pyafar, _ = detector.detect_nod(
            pitch=df_sub['pitch_pyafar'].values,
            yaw=df_sub['yaw_pyafar'].values,
            fps=fps,
            radians=False
        )
        for start, end in events_pyafar:
            idx = df_sub.index[start:end+1]
            df.loc[idx, 'nod_pyafar'] = 1

    # -------------------------
    # Debugging plots for one participant and one clip --> run in jupyter notebook
    # -------------------------
    # detector = NodDetector(k_std_velocity=1.0,k_std_still_pitch=4.0)
    # p, c = _, _
    # for (participant_id, clip), df_sub in df.groupby(['participant_id', 'clip'], sort=False):
    #     fps = (df_sub['frame'].iloc[-1] - df_sub['frame'].iloc[0]) / (
    #             df_sub['timestamp'].iloc[-1] - df_sub['timestamp'].iloc[0])

    #     # pick participant_id and clip to debug 
    #     if participant_id == p and clip == c:
    #         events, trace = detector.detect_nod(
    #             pitch=df_sub['pitch_openface'].values,
    #             yaw=df_sub['yaw_openface'].values,
    #             fps=fps,
    #             radians=True
    #         )
    #         trace["nod_agreed"] = df_sub["nod_agreed"].values # add GT to trace for plotting
    #         detector.plot_trace(trace, title=f"TRACE OpenFace pid={participant_id}, clip={clip}", show_triplet_labels=False)
        

    #df.to_csv("...", index=False)
    print("done!")

"""``pyaeroflow plot probes``: probe time series and spectra from ``<name>.probes.csv``.

One panel per quantity (p, u, v, T) with all probes overlaid, over the
physical time (unsteady) or the iteration (steady). ``spectrum`` names
columns (``probe1_v``, ``v`` for every probe, or ``all``) whose amplitude
spectrum is written: mean removed, Hann window, from ``x_from`` on. The
dominant frequency (parabolic peak fit) is printed with the Strouhal number
St = f L_ref / U_ref, L_ref and U_ref from ``<name>.resolved.toml`` next to
the file or given explicitly.
"""

import re
import tomllib
from pathlib import Path

import numpy as np

from pyaeroflow.plotting.common import pyplot, save
from pyaeroflow.plotting.readers import read_table

UNITS = {"p": "p [Pa]", "u": "u [m/s]", "v": "v [m/s]", "t": "T [K]"}


def _probe_columns(columns):
    """{quantity: [(probe number, column name), ...]}."""
    out = {}
    for name in columns:
        m = re.fullmatch(r"probe(\d+)_(\w+)", name)
        if m:
            out.setdefault(m.group(2), []).append((int(m.group(1)), name))
    return out


def reference_scales(path):
    """(L_ref, U_ref) of ``<name>.resolved.toml`` next to ``path``, or None."""
    path = Path(path)
    resolved = path.with_name(path.name.split(".")[0] + ".resolved.toml")
    if not resolved.exists():
        return None
    with open(resolved, "rb") as f:
        ref = tomllib.load(f).get("reference", {})
    if ref.get("length") and ref.get("velocity"):
        return float(ref["length"]), float(ref["velocity"])
    return None


def dominant_frequency(t, y):
    """(f, freqs, amplitude) of a uniformly sampled signal (Hann window)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if t.size < 8:
        raise ValueError("a spectrum needs at least 8 samples")
    dt = np.diff(t)
    if np.ptp(dt) > 1e-6 * dt.mean():
        raise ValueError("a spectrum needs a constant time step")
    w = np.hanning(t.size)
    amp = 2.0 * np.abs(np.fft.rfft((y - y.mean()) * w)) / w.sum()
    freqs = np.fft.rfftfreq(t.size, dt.mean())
    k = int(np.argmax(amp[1:])) + 1
    f = freqs[k]
    if k == 1:
        print(
            "note: the spectrum peaks at the lowest frequency; the window holds "
            "less than about two periods (or the signal only drifts)"
        )
    elif k < amp.size - 1:  # parabolic interpolation of the peak, within a bin
        a, b, c = amp[k - 1], amp[k], amp[k + 1]
        denom = a - 2.0 * b + c
        if denom != 0:
            shift = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
            f += shift * (freqs[1] - freqs[0])
    return f, freqs, amp


def plot_probes(
    path,
    prefix=None,
    quantities=None,
    spectrum=(),
    x_from=None,
    length=None,
    velocity=None,
    title=None,
):
    """Plot probe time series (and spectra); returns the written files."""
    path = Path(path)
    c = read_table(path)[0].columns
    unsteady = "time" in c
    x = c["time"] if unsteady else c["iteration"]
    groups = _probe_columns(c)
    quantities = [q.lower() for q in (quantities or groups)]
    quantities = [q for q in quantities if q in groups]
    if not quantities:
        raise ValueError(f"{path}: no probe columns")
    plt = pyplot()
    stem = path.name.split(".")[0]
    prefix = str(prefix) if prefix else str(path.with_name(stem))
    fig, axes = plt.subplots(
        len(quantities), 1, sharex=True, figsize=(13, 2.6 * len(quantities) + 1),
        gridspec_kw={"hspace": 0.08},
    )  # fmt: skip
    axes = np.atleast_1d(axes)
    for a, q in zip(axes, quantities, strict=True):
        for k, (number, name) in enumerate(groups[q]):
            a.plot(x, c[name], lw=1, color=f"C{k}", label=f"probe {number}")
        a.set_ylabel(UNITS.get(q, q))
        a.legend(fontsize=8, loc="upper right", ncol=min(4, len(groups[q])))
    if x_from is not None:
        for a in axes:
            a.axvline(x_from, color="k", ls=":", lw=0.8)
    axes[-1].set_xlabel("time [s]" if unsteady else "iteration")
    written = [save(fig, f"{prefix}_probes.png", f"Probes: {title or path.name}")]

    if not spectrum:
        return written
    names = []
    for spec in spectrum:
        spec = spec.lower()
        if spec == "all":
            names += [n for q in quantities for _, n in groups[q]]
        elif spec in groups:
            names += [n for _, n in groups[spec]]
        elif spec in c:
            names.append(spec)
        else:
            raise ValueError(f"spectrum: no column '{spec}' in {path.name}")
    if not unsteady:
        raise ValueError("a spectrum needs an unsteady probe file (time column)")
    sel = x >= (x_from if x_from is not None else x.min())
    scales = (length, velocity) if length and velocity else reference_scales(path)
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, name in enumerate(dict.fromkeys(names)):
        f, freqs, amp = dominant_frequency(x[sel], c[name][sel])
        st = f"  St = {f * scales[0] / scales[1]:.4f}" if scales else ""
        print(f"{name}: dominant frequency {f:.5g} Hz{st}")
        ax.semilogy(freqs[1:], np.maximum(amp[1:], 1e-300), color=f"C{k}", lw=1,
                    label=f"{name}: f = {f:.4g} Hz{st}")  # fmt: skip
        ax.axvline(f, color=f"C{k}", ls=":", lw=0.8)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("amplitude")
    ax.legend(fontsize=8)
    t0, t1 = x[sel].min(), x[sel].max()
    written.append(
        save(fig, f"{prefix}_spectrum.png",
             f"Spectrum ({t0:.4g} to {t1:.4g} s, Hann): {title or path.name}")
    )  # fmt: skip
    return written

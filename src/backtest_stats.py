"""
backtest_stats.py — Statistiques de backtest (López de Prado ch. 14) et risque de
stratégie (ch. 15).

ROLE CONCRET : ces mesures complètent le Sharpe / PSR / DSR déjà calculés. Elles
répondent à des questions que le seul Sharpe masque :
  - la performance repose-t-elle sur quelques coups de chance ?  -> HHI (ch. 14.5.1)
  - combien de temps la stratégie reste-t-elle en perte ?         -> Time under Water (ch. 14.5.2)
  - quelle est la qualité des paris ?                             -> hit ratio, gain/perte moyens
  - quelle est la probabilité que la stratégie ÉCHOUE à tenir     -> P[p < p_θ*] (ch. 15)
    sa cible de Sharpe, vu ses paris asymétriques ?

A quelle phase : section performance/risque du rapport, pour crédibiliser les
résultats au-delà du Sharpe et distinguer risque de PORTEFEUILLE et risque de STRATÉGIE.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# ch. 14.5.1 — Concentration des rendements (Herfindahl-Hirschman)
# ---------------------------------------------------------------------------
def hhi(returns):
    """Concentration HHI d'une série de rendements de paris. 0 = rendements uniformes,
    1 = un seul pari porte tout (fat tail). On la calcule sur les gains, sur les pertes
    et sur le temps entre paris."""
    r = pd.Series(returns).dropna()
    if r.shape[0] <= 2 or r.sum() == 0:
        return np.nan
    w = r / r.sum()
    h = (w ** 2).sum()
    n = r.shape[0]
    return (h - 1.0 / n) / (1.0 - 1.0 / n)


def concentration_report(returns, freq="ME"):
    """HHI sur (gains, pertes, temps entre paris). freq = regroupement temporel."""
    r = pd.Series(returns).dropna()
    return {
        "hhi_gains": hhi(r[r >= 0]),
        "hhi_pertes": hhi(r[r < 0]),
        "hhi_temps": hhi(r.groupby(pd.Grouper(freq=freq)).count()) if isinstance(r.index, pd.DatetimeIndex) else np.nan,
    }


# ---------------------------------------------------------------------------
# ch. 14.5.2 — Drawdown et Time under Water
# ---------------------------------------------------------------------------
def drawdown_tuw(returns):
    """Série des drawdowns (DD) et des temps sous l'eau (TuW, en années) à partir
    d'une série de rendements. TuW = durée entre un plus-haut et le moment où le
    cumul le dépasse à nouveau. Retourne (dd_series, tuw_series)."""
    eq = (1 + pd.Series(returns).dropna()).cumprod()
    df0 = eq.to_frame("pnl")
    df0["hwm"] = eq.expanding().max()
    df1 = df0.groupby("hwm").min().reset_index()
    df1.columns = ["hwm", "min"]
    df1.index = df0["hwm"].drop_duplicates(keep="first").index
    df1 = df1[df1["hwm"] > df1["min"]]
    dd = 1 - df1["min"] / df1["hwm"]
    if len(df1) > 1:
        days = (df1.index[1:] - df1.index[:-1]).days
        tuw = pd.Series(days / 365.25, index=df1.index[:-1])
    else:
        tuw = pd.Series(dtype=float)
    return dd, tuw


# ---------------------------------------------------------------------------
# ch. 14.4 — hit ratio et gains/pertes moyens
# ---------------------------------------------------------------------------
def hit_stats(bet_returns):
    """hit ratio (part de paris gagnants), gain moyen des gagnants, perte moyenne
    des perdants, sur une série de P&L de PARIS (round-trips)."""
    b = pd.Series(bet_returns).dropna()
    return {
        "hit_ratio": (b > 0).mean(),
        "gain_moyen": b[b > 0].mean(),
        "perte_moyenne": b[b <= 0].mean(),
        "n_paris": len(b),
    }


# ---------------------------------------------------------------------------
# ch. 15 — Risque de stratégie : probabilité d'échec P[p < p_θ*]
# ---------------------------------------------------------------------------
def implied_precision(sl, pt, freq, target_sr):
    """binHR (ch. 15.3) : précision p minimale requise pour atteindre target_sr,
    étant donné stop-loss (sl<0), profit-taking (pt>0) et fréquence de paris/an."""
    a = (freq + target_sr ** 2) * (pt - sl) ** 2
    b = (2 * freq * sl - target_sr ** 2 * (pt - sl)) * (pt - sl)
    c = freq * sl ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return np.nan
    return (-b + disc ** 0.5) / (2 * a)


def prob_failure(bet_returns, freq, target_sr):
    """P[p < p_θ*] (ch. 15.4) : probabilité que la stratégie n'atteigne pas son
    Sharpe cible. On estime π+ / π- des paris, la précision p_θ* requise, puis la
    probabilité que la vraie précision p tombe sous p_θ*, via l'approximation normale
    p ~ N[p_obs, p_obs(1-p_obs)/N] (N = nombre de paris ; forme corrigée du snippet 15.5,
    dont l'écart-type dépend du nombre de paris observés).

    C'est le RISQUE DE STRATÉGIE (va-t-elle tenir dans le temps ?), distinct du
    risque de portefeuille (volatilité du P&L). On rejette usuellement si > 0.05."""
    b = np.asarray(pd.Series(bet_returns).dropna())
    pos, neg = b[b > 0].mean(), b[b <= 0].mean()
    p_obs = (b > 0).mean()
    n = len(b)
    p_star = implied_precision(neg, pos, freq, target_sr)
    scale = np.sqrt(p_obs * (1 - p_obs) / n)
    prob = float(norm.cdf(p_star, loc=p_obs, scale=scale))   # P[p < p_θ*]
    return {"pi_plus": pos, "pi_minus": neg, "freq": freq, "n_paris": n,
            "p_observe": p_obs, "p_requis": p_star, "prob_echec": prob}

"""
screening.py — Screening de paires + Sharpe deflate (rigueur Lopez de Prado).

ROLE CONCRET : ce module factorise la logique de screening qui etait dupliquee
dans les notebooks 06b et 08b, et il ajoute la brique manquante de la Partie 3 de
Lopez de Prado : le "record every backtest" (ch. 11/14) et le Deflated Sharpe
Ratio (ch. 14) calcules sur le VRAI nombre d'essais du screening, pas sur une
poignee de configurations.

A QUOI CA SERT, ETAPE PAR ETAPE :
  1. generate_candidates       -> reconstitue l'univers d'essais (paires intra-secteur
                                  filtrees par correlation). C'est le N de la multiplicite.
  2. vec_pair_backtest         -> rejoue UNE paire avec la regle z-score/couts. Version
                                  vectorisee du moteur evenementiel (validee a 0.02 de
                                  Sharpe pres sur ADI/MPWR : 0.589 vs 0.610 moteur).
  3. record_all_trials         -> rejoue TOUTES les candidates et enregistre le theta
                                  (Sharpe par pari, ch. 15) de chacune. C'est la preuve
                                  "j'ai enregistre chaque backtest".
  4. deflated_sharpe           -> deflate le Sharpe du gagnant sur la distribution des N
                                  essais (SR* = maximum attendu sous H0, ch. 14).
  5. calibrate_ou_thresholds   -> grille triple-barriere (entree x sortie x stop-loss) sur
                                  chemins O-U synthetiques (ch. 13). Sert a la Phase paires
                                  a choisir les seuils SANS simulation historique.

POURQUOI le theta (Sharpe PAR PARI) et pas le Sharpe quotidien : une paire est a
plat la plupart du temps. Le Sharpe quotidien annualise par sqrt(252) d'une serie
gonflee de zeros explose pour les paires qui tradent peu (artefact). Le theta du
ch. 15, theta = SR_pari * sqrt(n_paris/an), est robuste a cela et unifie ch. 14 et 15.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from scipy.stats import norm
from scipy.integrate import quad

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# 1. Univers d'essais
# ---------------------------------------------------------------------------
def generate_candidates(adj, sectors, corr_min=0.40):
    """Paires intra-secteur dont la correlation des rendements log >= corr_min.
    Retourne la liste [(a, b), ...]. len(...) est le nombre d'essais conduits."""
    log_ret = np.log(adj).diff().dropna()
    cands = []
    for sec in sectors.unique():
        names = [n for n in sectors[sectors == sec].index if n in log_ret.columns]
        for a, b in combinations(sorted(names), 2):
            if log_ret[a].corr(log_ret[b]) >= corr_min:
                cands.append((a, b))
    return cands


# ---------------------------------------------------------------------------
# 2. Backtest vectorise d'une paire (miroir du moteur evenementiel)
# ---------------------------------------------------------------------------
def _ols_beta(y, x):
    return sm.OLS(y, sm.add_constant(x)).fit().params.iloc[1]


def vec_pair_backtest(adj, A, B, win_z=43, entry=2.5, exit=0.5,
                      stop_loss=np.inf, cost=3e-4, beta=None):
    """Rejoue la paire (A, B) avec la meme regle que PairsTradingStrategy :
    spread = A - beta*B (beta OLS plein echantillon), z-score glissant win_z,
    entree a |z|>entry, sortie a |z|<exit, plus un stop-loss OPTIONNEL a |z|>stop_loss.
    Dollar-neutral equipondere (0.5/0.5). Couts = cost par unite de rotation.

    Retourne (daily_returns, bet_pnls) :
      daily_returns : Serie des rendements quotidiens de la strategie
      bet_pnls      : np.array du P&L de chaque pari (round-trip), pour le ch. 15
    """
    y, x = adj[A], adj[B]
    if beta is None:
        beta = _ols_beta(y, x)          # OLS par défaut ; passer beta=... pour Johansen
    spread = y - beta * x
    m = spread.rolling(win_z).mean()
    s = spread.rolling(win_z).std()
    z = ((spread - m) / s).values

    pos = np.zeros(len(z))
    state = 0
    for i in range(len(z)):
        if np.isnan(z[i]):
            pos[i] = 0
            continue
        if state == 0:
            if z[i] > entry:
                state = -1
            elif z[i] < -entry:
                state = 1
        else:
            if abs(z[i]) > stop_loss or abs(z[i]) < exit:
                state = 0
        pos[i] = state
    pos = pd.Series(pos, index=adj.index)

    rA, rB = adj[A].pct_change(), adj[B].pct_change()
    turn = pos.diff().abs().fillna(0)
    daily = (pos.shift(1) * (0.5 * rA - 0.5 * rB) - turn * cost).fillna(0)

    # extraction des paris (du jour d'entree 0->+-1 au jour de sortie +-1->0)
    p = pos.values
    d = daily.values
    bets, cur, inpos = [], 0.0, False
    for i in range(len(p)):
        if not inpos and p[i] != 0 and (i == 0 or p[i - 1] == 0):
            inpos, cur = True, 0.0
        if inpos:
            cur += d[i]
            if p[i] == 0:
                bets.append(cur)
                inpos, cur = False, 0.0
    if inpos:
        bets.append(cur)
    return daily, np.array(bets)


def bet_theta(bet_pnls, years):
    """Sharpe annualise PAR PARI (ch. 15) : theta = mean/std(paris) * sqrt(n_paris/an).
    Retourne (theta, n_bets, n_per_year, sr_bet). np.nan si trop peu de paris."""
    n = len(bet_pnls)
    if n < 5 or bet_pnls.std(ddof=1) == 0:
        return np.nan, n, np.nan, np.nan
    n_per_year = n / years
    sr_bet = bet_pnls.mean() / bet_pnls.std(ddof=1)
    return sr_bet * np.sqrt(n_per_year), n, n_per_year, sr_bet


# ---------------------------------------------------------------------------
# 3. "Record every backtest" : theta de CHAQUE candidate
# ---------------------------------------------------------------------------
def record_all_trials(adj, sectors, corr_min=0.40, **bt_kwargs):
    """Rejoue toutes les candidates et enregistre leur theta. C'est l'artefact
    exige par la 3e loi de Lopez de Prado (ch. 14). Retourne un DataFrame trie."""
    years = len(adj) / TRADING_DAYS
    rows = []
    for a, b in generate_candidates(adj, sectors, corr_min):
        _, bets = vec_pair_backtest(adj, a, b, **bt_kwargs)
        theta, nb, npy, _ = bet_theta(bets, years)
        rows.append({"paire": f"{a}/{b}", "a": a, "b": b,
                     "theta": theta, "n_bets": nb})
    return pd.DataFrame(rows).sort_values("theta", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Deflated Sharpe Ratio sur le vrai nombre d'essais
# ---------------------------------------------------------------------------
def expected_max_sharpe(var_sr, n_trials):
    """SR* = maximum attendu du Sharpe sous H0 (vrai Sharpe nul), ch. 14.
    Croit avec le nombre d'essais N et avec la variance des Sharpe d'essais."""
    return np.sqrt(var_sr) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
        + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )


def deflated_sharpe(trials_theta, winner_bets, years):
    """DSR du gagnant, deflate sur la distribution des N essais.
    trials_theta : Serie/array des theta de tous les essais (record_all_trials["theta"]).
    winner_bets  : np.array des paris du gagnant.
    Retourne dict {N, sr_star, theta_winner, dsr, rank}."""
    t = pd.Series(trials_theta).dropna()
    N = len(t)
    var_sr = t.var(ddof=1)
    sr_star = expected_max_sharpe(var_sr, N)                 # niveau annualise
    theta_w, n, n_per_year, sr_bet = bet_theta(winner_bets, years)

    # PSR au niveau PARI : on ramene SR* a la frequence des paris du gagnant
    sr_star_bet = sr_star / np.sqrt(n_per_year)
    b = pd.Series(winner_bets)
    sk, ku = b.skew(), b.kurtosis() + 3.0
    zstat = ((sr_bet - sr_star_bet) * np.sqrt(n - 1)
             / np.sqrt(1 - sk * sr_bet + ((ku - 1) / 4) * sr_bet ** 2))
    dsr = float(norm.cdf(zstat))
    rank = int((t > theta_w).sum()) + 1
    return {"N": N, "sr_star": float(sr_star), "theta_winner": float(theta_w),
            "dsr": dsr, "rank": rank}


# ---------------------------------------------------------------------------
# 5. Calibration triple-barriere sur donnees synthetiques O-U (ch. 13)
# ---------------------------------------------------------------------------
def fit_ou(adj, A, B):
    """Ajuste un O-U discret (AR1) sur le spread : retourne mu, phi, sigma_e, sigma_eq,
    demi-vie. Estimation plein echantillon = conforme au ch. 13 (pas de look-ahead)."""
    beta = _ols_beta(adj[A], adj[B])
    spread = adj[A] - beta * adj[B]
    lag = spread.shift(1).dropna()
    dl = spread.diff().dropna()
    lag = lag.loc[dl.index]
    reg = sm.OLS(dl, sm.add_constant(lag)).fit()
    a0, b = reg.params.iloc[0], reg.params.iloc[1]
    theta = -b
    phi = 1 + b
    sig_e = reg.resid.std()
    sig_eq = sig_e / np.sqrt(1 - phi ** 2)
    return {"mu": a0 / theta, "phi": phi, "sigma_e": sig_e,
            "sigma_eq": sig_eq, "half_life": np.log(2) / theta}


def _simulate_ou_z(ou, n, seed):
    rng = np.random.default_rng(seed)
    v = np.empty(n)
    v[0] = ou["mu"]
    for t in range(1, n):
        v[t] = ou["mu"] + ou["phi"] * (v[t - 1] - ou["mu"]) + ou["sigma_e"] * rng.standard_normal()
    return (v - ou["mu"]) / ou["sigma_eq"]


def calibrate_ou_thresholds(adj, A, B, entries=(2.0, 2.5, 3.0),
                            exits=(0.0, 0.5, 1.0),
                            stop_losses=(2.75, 3.5, 5.0, np.inf),
                            n_paths=400, n_len=2000):
    """Grille triple-barriere (entree x sortie x stop-loss) evaluee sur chemins O-U
    synthetiques. Retourne un DataFrame de Sharpe moyens. C'est la parade du ch. 13
    au sur-ajustement : on calibre sur le processus, pas sur l'unique chemin historique."""
    ou = fit_ou(adj, A, B)
    paths = [_simulate_ou_z(ou, n_len, s) for s in range(n_paths)]

    def one(zp, entry, exit_, sl):
        pos, pnl = 0, []
        for i in range(1, len(zp)):
            r = pos * (zp[i - 1] - zp[i]) if pos != 0 else 0.0
            if pos == 0:
                if zp[i] > entry:
                    pos = -1
                elif zp[i] < -entry:
                    pos = 1
            else:
                if abs(zp[i]) > sl or abs(zp[i]) < exit_:
                    pos = 0
            pnl.append(r)
        pnl = np.array(pnl)
        return pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0

    rows = []
    for e in entries:
        for xt in exits:
            for sl in stop_losses:
                sr = np.mean([one(p, e, xt, sl) for p in paths])
                rows.append({"entry": e, "exit": xt,
                             "stop_loss": ("inf" if np.isinf(sl) else sl),
                             "sharpe": sr})
    return pd.DataFrame(rows), ou


# ---------------------------------------------------------------------------
# 5bis. Seuils decouples (Option B) : ENTREE via Bertram, SORTIE via Lopez de Prado
# ---------------------------------------------------------------------------
# Pourquoi deux couches, et pas la grille 3D unique de calibrate_ou_thresholds ?
# Parce que les deux decisions ne repondent PAS a la meme question :
#   - QUAND OUVRIR une position ? -> Bertram [2009]. Il exprime la duree d'un trade
#     comme un temps de premier passage de l'O-U, en deduit la frequence de trading
#     E[1/T], puis le TAUX DE PROFIT mu_P = (m - a - c) * E[1/T], et maximise sur les
#     niveaux (a, m). On ne garde de Bertram que l'ENTREE a* (Option B).
#   - COMMENT SORTIR une position DEJA ouverte ? -> Lopez de Prado, AFML ch. 13. Il
#     prend l'entree comme donnee et calibre le couple (profit-taking, stop-loss) sur
#     le P&L du trade en cours, par simulation de l'O-U (parade au sur-ajustement).
# LdP renvoie lui-meme a Bertram pour la question d'entree : on suit donc la division
# que les deux auteurs tracent. La couche 1 est ANALYTIQUE (EDO de premier passage),
# la couche 2 est par SIMULATION -- c'est voulu et fidele a chaque source.

_S2PI = np.sqrt(2.0 * np.pi)


def _ou_alpha(ou):
    """Vitesse de reversion continue de l'O-U (par jour), alpha = 1 - phi.
    E[T] sortira en JOURS, coherent avec la demi-vie (= ln2/alpha)."""
    return 1.0 - ou["phi"]


def _mfpt_up(x, m, alpha):
    """Temps moyen de premier passage O-U (std stationnaire=1) pour MONTER de x a m,
    barriere basse naturelle en -inf. Solution de l'EDO de Bertram (eq. 18/32) :
        u'' - z u' = -1/alpha,  u(m)=0  ->  T_up = (1/a) INT_x^m e^{y^2/2} sqrt(2pi) Phi(y) dy."""
    return quad(lambda y: np.exp(y * y / 2) * _S2PI * norm.cdf(y), x, m)[0] / alpha


def _mfpt_down(x, a, alpha):
    """Symetrique : temps moyen pour DESCENDRE de x a a, barriere haute naturelle en +inf."""
    return quad(lambda y: np.exp(y * y / 2) * _S2PI * (1 - norm.cdf(y)), a, x)[0] / alpha


def bertram_entry(ou, c=0.0, grid=np.arange(0.25, 3.01, 0.25)):
    """COUCHE 1 (Bertram 2009) -- niveau d'ENTREE optimal du z-spread.

    A QUOI CA SERT : remplace le choix ad hoc / grille-Sharpe de l'entree par le
    niveau qui MAXIMISE LE TAUX DE PROFIT de l'O-U, obtenu analytiquement via les
    temps de premier passage (aucune simulation, aucun sur-ajustement a un chemin).

    On maximise mu_P = (m - a - c) / (E[T_montee] + E[T_descente]) sur l'aller-retour
    symetrique a<0<m (en unites d'ecart-type stationnaire du spread). `c` est le cout
    de transaction aller-retour exprime dans CES memes unites (c=0 = repere sans
    friction). Consequence attendue et pedagogique : c=0 donne une entree serree ;
    plus `c` augmente, plus l'entree optimale s'ecarte de la moyenne (Table 1 de
    Bertram) -- c'est exactement l'effet des couts que le projet observe empiriquement.

    Retour : dict {entry, exit, mu_P}. En Option B on n'utilise que `entry` ;
    `exit` est fourni pour reference (il sera REMPLACE par la couche LdP)."""
    alpha = _ou_alpha(ou)
    best = {"mu_P": -np.inf, "entry": None, "exit": None}
    for a in -grid:
        for m in grid:
            r = (m - a) - c
            if r <= 0:
                continue
            ett = _mfpt_up(a, m, alpha) + _mfpt_down(m, a, alpha)
            mu = r / ett
            if mu > best["mu_P"]:
                best = {"mu_P": mu, "entry": abs(a), "exit": m}
    return best


def ldp_exit_corridor(ou, entry_z, grid=np.arange(0.25, 3.01, 0.25),
                      n_iter=40000, max_hp=200, seed=0):
    """COUCHE 2 (Lopez de Prado, AFML ch. 13) -- corridor de SORTIE (profit-taking,
    stop-loss) d'une position DEJA ouverte a l'entree Bertram `entry_z`.

    A QUOI CA SERT : la position existe (ouverte a z0 = -entry_z, on parie sur le
    retour vers 0). On cherche le couple (pt, sl) qui maximise le Sharpe du P&L du
    trade, en simulant l'O-U discret estime (snippets 13.1/13.2 de LdP reparametres).
    C'est la question d'EXECUTION -- distincte de l'entree, d'ou la couche separee.

    Barriere verticale `max_hp` = duree de detention maximale (triple-barriere ch. 3).
    Notre spread a un equilibre nul : on retombe donc sur le cas "Forecast=0" de LdP
    (Fig. 13.1), ou l'optimum est petit profit-taking / grand stop-loss.

    Retour : dict {profit_taking, stop_loss, sharpe} du meilleur noeud, + la grille
    complete sous 'surface' (pour tracer la heat-map du rapport)."""
    phi = ou["phi"]
    sig = np.sqrt(1 - phi ** 2)                 # sigma en unites z (std stationnaire=1)
    rng = np.random.default_rng(seed)
    z0 = -entry_z
    P = np.empty((n_iter, max_hp + 1))
    P[:, 0] = z0
    for t in range(1, max_hp + 1):              # O-U discret, esperance 0 (equilibre nul)
        P[:, t] = phi * P[:, t - 1] + sig * rng.standard_normal(n_iter)
    pnl = P - z0                                # P&L par unite : gain quand z remonte vers 0
    idx_all = np.arange(n_iter)
    rows = []
    for pt in grid:                             # profit-taking (barriere favorable)
        for sl in grid:                         # stop-loss (barriere adverse)
            hit = (pnl >= pt) | (pnl <= -sl)    # premiere barriere touchee (vectorise)
            first = np.where(hit.any(1), hit.argmax(1), max_hp)
            out = pnl[idx_all, first]
            sd = out.std()
            rows.append((pt, sl, out.mean() / sd if sd > 0 else 0.0))
    surf = pd.DataFrame(rows, columns=["profit_taking", "stop_loss", "sharpe"])
    top = surf.loc[surf.sharpe.idxmax()]
    return {"profit_taking": float(top.profit_taking),
            "stop_loss": float(top.stop_loss),
            "sharpe": float(top.sharpe), "surface": surf}


# ---------------------------------------------------------------------------
# 6. Screening de cointegration (factorise l'ancien 08b : EG + BH + SCORE)
# ---------------------------------------------------------------------------
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests


def eg_pvalue(adj, a, b):
    """p-value Engle-Granger, min des deux directions (corrige l'asymetrie OLS)."""
    return min(coint(adj[a], adj[b])[1], coint(adj[b], adj[a])[1])


def _half_life(spread):
    lag = spread.shift(1).dropna()
    dl = spread.diff().dropna()
    lag = lag.loc[dl.index]
    b = sm.OLS(dl, sm.add_constant(lag)).fit().params.iloc[1]
    return np.log(2) / (-b) if b < 0 else np.nan


def _subwindow_frac(adj, a, b):
    y, x = adj[a], adj[b]
    yrs = sorted(set(y.index.year))
    h = t = 0
    for yr in range(yrs[0], yrs[-1], 2):
        m = (y.index.year >= yr) & (y.index.year < yr + 2)
        if m.sum() < 200:
            continue
        t += 1
        if coint(y[m], x[m])[1] < 0.05:
            h += 1
    return h / t if t else np.nan


def _beta_cv(adj, a, b, win=252):
    y, x = adj[a], adj[b]
    bser = (y.rolling(win).cov(x) / x.rolling(win).var()).dropna()
    return bser.std() / abs(bser.mean()) if bser.mean() != 0 else np.nan


def _hl_score(hl):
    if np.isnan(hl) or hl <= 0:
        return 0.0
    if 10 <= hl <= 60:
        return 1.0
    if 5 <= hl < 10 or 60 < hl <= 120:
        return 0.5
    return 0.2


def screen_cointegration(adj, sectors, corr_min=0.40, alpha=0.05):
    """Pipeline complet : candidates -> EG p-value -> Benjamini-Hochberg ->
    metriques (demi-vie, robustesse sous-fenetres, stabilite beta) -> SCORE.
    Retourne (ranked_survivors_df, n_candidates). Reproduit exactement l'ancien 08b."""
    cand = pd.DataFrame(
        [{"paire": f"{a}/{b}", "a": a, "b": b,
          "secteur": sectors[a], "corr": np.log(adj).diff()[a].corr(np.log(adj).diff()[b])}
         for a, b in generate_candidates(adj, sectors, corr_min)]
    )
    cand["p_value"] = [eg_pvalue(adj, r.a, r.b) for r in cand.itertuples()]
    reject, p_bh, _, _ = multipletests(cand["p_value"], alpha=alpha, method="fdr_bh")
    cand["p_bh"], cand["signif_bh"] = p_bh, reject

    sig = cand[cand["signif_bh"]].copy()
    for i, r in sig.iterrows():
        sig.loc[i, "half_life"] = _half_life(adj[r.a] - _ols_beta(adj[r.a], adj[r.b]) * adj[r.b])
        sig.loc[i, "subwin_frac"] = _subwindow_frac(adj, r.a, r.b)
        sig.loc[i, "beta_cv"] = _beta_cv(adj, r.a, r.b)
    sig["SCORE"] = (0.30 * (1 - sig["p_bh"]) + 0.30 * sig["subwin_frac"]
                    + 0.20 * sig["half_life"].apply(_hl_score) + 0.20 * (1 / (1 + sig["beta_cv"])))
    ranked = sig.sort_values("SCORE", ascending=False, ignore_index=True)
    return ranked, len(cand)


# ---------------------------------------------------------------------------
# 7. Diagnostics de cointegration / mean-reversion (Johansen, Hurst, VR)
# ---------------------------------------------------------------------------
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_beta(adj, a, b, det_order=0, k_ar_diff=1):
    """Hedge ratio via Johansen (traite les deux titres symetriquement, contrairement
    a l'OLS qui privilegie une variable dependante). beta tel que spread = a - beta*b.
    Retourne (beta, trace_stat, crit_95)."""
    jr = coint_johansen(adj[[a, b]].values, det_order, k_ar_diff)
    v = jr.evec[:, 0]
    beta = -v[1] / v[0]                       # normalisation : coeff de 'a' = 1
    return beta, jr.lr1[0], jr.cvt[0, 1]      # trace r=0 vs valeur critique 95%


def hurst_exponent(series, max_lag=100):
    """Exposant de Hurst par la methode des increments : std des ecarts a l'horizon tau,
    pente de log(std) vs log(tau). H<0.5 = mean-reverting, 0.5 = marche aleatoire, >0.5 = tendanciel."""
    s = np.asarray(series, dtype=float)
    lags = range(2, max_lag)
    tau = [np.std(s[l:] - s[:-l]) for l in lags]
    return np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]


def variance_ratio(series, q=20):
    """Ratio de variance de Lo-MacKinlay sur les increments : VR(q) = Var(q pas)/(q*Var(1 pas)).
    VR<1 = mean-reverting, ~1 = marche aleatoire, >1 = tendanciel."""
    s = np.asarray(series, dtype=float)
    d = np.diff(s)
    var1 = np.var(d, ddof=1)
    dq = s[q:] - s[:-q]
    varq = np.var(dq, ddof=1)
    return varq / (q * var1)
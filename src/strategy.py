"""
strategy.py — La Strategy : la regle de decision.

ROLE CONCRET : ecoute les MarketEvent et, quand la regle est declenchee, emet
un SignalEvent. Elle ne sait RIEN du portefeuille, du capital ou de l'execution.
Cette ignorance est voulue : elle rend la strategie interchangeable. Au Projet 5,
un signal d'alpha issu du machine learning se branchera ici a l'identique, sans
toucher au reste du moteur.
"""
from abc import ABC, abstractmethod
import numpy as np
from event import SignalEvent


class Strategy(ABC):
    @abstractmethod
    def calculate_signals(self, event):
        ...


class MovingAverageCrossStrategy(Strategy):
    """
    Croisement de moyennes mobiles, logique long/flat (identique a la Phase 1).
    - courte > longue et pas en position -> signal LONG
    - courte < longue et en position     -> signal EXIT
    """
    def __init__(self, data, events, symbol_list, short=50, long=200):
        self.data = data
        self.events = events
        self.symbol_list = symbol_list
        self.short = short
        self.long = long
        # etat par symbole : 'OUT' (a l'ecart) ou 'LONG' (investi)
        self.state = {s: "OUT" for s in symbol_list}

    def calculate_signals(self, event):
        if event.type != "MARKET":
            return
        for s in self.symbol_list:
            bars = self.data.get_latest_bars(s, N=self.long)
            if len(bars) < self.long:
                continue  # pas assez d'historique pour la MA longue
            closes = np.array([b.close for b in bars])
            ma_s = closes[-self.short:].mean()
            ma_l = closes.mean()
            dt = bars[-1].datetime

            if ma_s > ma_l and self.state[s] == "OUT":
                self.events.put(SignalEvent(s, dt, "LONG"))
                self.state[s] = "LONG"
            elif ma_s < ma_l and self.state[s] == "LONG":
                self.events.put(SignalEvent(s, dt, "EXIT"))
                self.state[s] = "OUT"


class PairsTradingStrategy(Strategy):
    """
    Pairs trading (retour a la moyenne, long-short) sur deux symboles [Y, X].
    Le spread est defini par Y - beta*X, avec un ratio de couverture GLISSANT
    (regression sur win_beta jours). Le z-score glissant (win_z jours) declenche :
      - z > entry  et FLAT -> SHORT le spread : SHORT Y, LONG X
      - z < -entry et FLAT -> LONG  le spread : LONG Y,  SHORT X
      - |z| < exit et en position -> EXIT les deux jambes
    Execution dollar-neutral a poids egaux (leg_weight par jambe) : le beta sert au
    SIGNAL (construction du spread), le dimensionnement reste equipondere, un choix
    standard et honnete qu'on assume dans le rapport.
    """
    def __init__(self, data, events, y_sym, x_sym,
                 win_beta=252, win_z=43, entry=2.0, exit=0.5, leg_weight=0.5,
                 fixed_beta=None):
        self.data = data
        self.events = events
        self.y = y_sym
        self.x = x_sym
        self.win_beta = win_beta
        self.win_z = win_z
        self.entry = entry
        self.exit = exit
        self.leg_weight = leg_weight
        self.fixed_beta = fixed_beta   # si fourni : hedge ratio statique (paire cointegree)
        self.state = "FLAT"          # FLAT | LONG_SPREAD | SHORT_SPREAD

    def _zscore(self):
        L = self.win_beta
        by = self.data.get_latest_bars(self.y, N=L)
        bx = self.data.get_latest_bars(self.x, N=L)
        need = self.win_z if self.fixed_beta is not None else L
        if len(by) < need or len(bx) < need:
            return None
        Y = np.array([b.close for b in by])
        X = np.array([b.close for b in bx])
        if self.fixed_beta is not None:
            beta = self.fixed_beta                       # cointegration : beta stable
        else:
            beta = np.cov(Y, X)[0, 1] / np.var(X)        # sinon : beta glissant (adaptatif)
        spread = Y - beta * X
        w = spread[-self.win_z:]
        sd = w.std()
        if sd == 0:
            return None
        return (spread[-1] - w.mean()) / sd

    def calculate_signals(self, event):
        if event.type != "MARKET":
            return
        z = self._zscore()
        if z is None:
            return
        dt = self.data.get_latest_bar_datetime(self.y)

        if self.state == "FLAT":
            if z > self.entry:       # spread trop haut -> parier sur la baisse
                self.events.put(SignalEvent(self.y, dt, "SHORT", self.leg_weight))
                self.events.put(SignalEvent(self.x, dt, "LONG",  self.leg_weight))
                self.state = "SHORT_SPREAD"
            elif z < -self.entry:    # spread trop bas -> parier sur la hausse
                self.events.put(SignalEvent(self.y, dt, "LONG",  self.leg_weight))
                self.events.put(SignalEvent(self.x, dt, "SHORT", self.leg_weight))
                self.state = "LONG_SPREAD"
        else:
            if abs(z) < self.exit:   # retour vers la moyenne -> on debouche
                self.events.put(SignalEvent(self.y, dt, "EXIT"))
                self.events.put(SignalEvent(self.x, dt, "EXIT"))
                self.state = "FLAT"


class MomentumStrategy(Strategy):
    """
    Momentum cross-sectional, rebalance mensuel, dans le moteur evenementiel.
    A chaque changement de mois : classe les titres par performance sur lookback
    mois, detient le top_k a poids egaux. Emet EXIT pour les sortants et LONG
    (strength = 1/top_k) pour les entrants.
    """
    def __init__(self, data, events, symbol_list, lookback_months=3, top_k=3):
        self.data = data
        self.events = events
        self.symbol_list = symbol_list
        self.lookback = lookback_months
        self.top_k = top_k
        self.held = set()
        self.cur_month = None
        self.approx_days = 21 * lookback_months + 5  # barres necessaires

    def calculate_signals(self, event):
        if event.type != "MARKET":
            return
        dt = self.data.get_latest_bar_datetime(self.symbol_list[0])
        ym = (dt.year, dt.month)
        if ym == self.cur_month:
            return                       # on ne rebalance qu'au changement de mois
        self.cur_month = ym

        window = 21 * self.lookback
        perf = {}
        for s in self.symbol_list:
            bars = self.data.get_latest_bars(s, N=window + 1)
            if len(bars) < window + 1:
                return
            perf[s] = bars[-1].close / bars[-window].close - 1.0

        ranked = sorted(perf, key=perf.get, reverse=True)
        new_top = set(ranked[:self.top_k])

        for s in self.held - new_top:    # sortants
            self.events.put(SignalEvent(s, dt, "EXIT"))
        for s in new_top:                # entrants / maintenus
            self.events.put(SignalEvent(s, dt, "LONG", 1.0 / self.top_k))
        self.held = new_top

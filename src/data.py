"""
data.py — Le DataHandler : la source de donnees du moteur.

ROLE CONCRET : rejoue l'historique barre par barre, comme si les prix arrivaient
en direct. A chaque appel, il "revele" un jour de plus et emet un MarketEvent.
C'est ce mecanisme qui garantit qu'a l'instant t, la strategie ne connait QUE
le passe (jusqu'a t) : le look-ahead bias devient structurellement impossible.

En Phase 2 il migre depuis les fonctions du notebook 00. Il consomme le fichier
fige data/adj_close.csv produit en Phase 0.
"""
from abc import ABC, abstractmethod
from collections import namedtuple
import pandas as pd
from event import MarketEvent

Bar = namedtuple("Bar", ["datetime", "close"])


class DataHandler(ABC):
    """Interface abstraite. N'importe quelle source (CSV, base, live) doit la respecter."""

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        ...

    @abstractmethod
    def get_latest_bar_value(self, symbol):
        ...

    @abstractmethod
    def update_bars(self):
        ...


class HistoricCSVDataHandler(DataHandler):
    """
    Rejoue un DataFrame de cours ajustes (colonnes = symboles, index = dates).
    On travaille sur le cours ajuste uniquement (suffisant pour ce projet).
    """
    def __init__(self, events, prices, symbol_list):
        self.events = events
        self.symbol_list = symbol_list
        self.continue_backtest = True

        # Historique complet en memoire, aligne sur un index commun
        self._prices = prices[symbol_list].dropna()
        self._dates = list(self._prices.index)
        self._i = -1  # pointeur sur la barre courante (-1 = rien encore revele)

        # Barres deja revelees (ce que la strategie a le droit de "voir")
        self.latest_bars = {s: [] for s in symbol_list}

    def update_bars(self):
        """Revele la barre suivante et emet un MarketEvent. Termine le backtest a la fin."""
        self._i += 1
        if self._i >= len(self._dates):
            self.continue_backtest = False
            return
        dt = self._dates[self._i]
        for s in self.symbol_list:
            self.latest_bars[s].append(Bar(dt, float(self._prices.iloc[self._i][s])))
        self.events.put(MarketEvent())

    def get_latest_bars(self, symbol, N=1):
        """Renvoie les N dernieres barres revelees (jamais le futur)."""
        return self.latest_bars[symbol][-N:]

    def get_latest_bar_value(self, symbol):
        """Dernier cours connu (sert de prix d'execution)."""
        return self.latest_bars[symbol][-1].close

    def get_latest_bar_datetime(self, symbol):
        return self.latest_bars[symbol][-1].datetime

"""
portfolio.py — Le Portfolio : le gestionnaire de risque et de capital.

ROLE CONCRET : c'est le cerveau comptable du moteur. Il fait trois choses.
1. Sur SignalEvent : traduit un signal en OrderEvent concret (dimensionnement).
2. Sur FillEvent   : met a jour positions et tresorerie apres execution.
3. Sur MarketEvent : valorise le portefeuille au prix du jour (mark-to-market)
   et enregistre la courbe d'equite.

Dimensionnement retenu ici : cible en pourcentage du capital (target percent).
Un signal LONG vise TARGET_PCT du capital total investi dans le titre ; un EXIT
vise 0. C'est comparable a la logique "tout ou rien" du backtest vectoriel, ce
qui permettra de VALIDER le moteur en retrouvant des resultats proches.
"""
import pandas as pd
import numpy as np
from event import OrderEvent


class Portfolio:
    def __init__(self, data, events, symbol_list, start_date,
                 initial_capital=100_000.0, target_pct=0.95):
        self.data = data
        self.events = events
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.target_pct = target_pct  # marge de 5% pour eviter le sur-investissement

        self.positions = {s: 0 for s in symbol_list}   # quantite detenue
        self.cash = initial_capital
        self.history = []  # liste de dicts {datetime, total, cash, ...}

    # --- 3. valorisation a chaque barre ---
    def update_timeindex(self, event):
        dt = self.data.get_latest_bar_datetime(self.symbol_list[0])
        market_value = 0.0
        for s in self.symbol_list:
            price = self.data.get_latest_bar_value(s)
            market_value += self.positions[s] * price
        total = self.cash + market_value
        self.history.append({"datetime": dt, "cash": self.cash,
                             "market_value": market_value, "total": total,
                             **{f"pos_{s}": self.positions[s] for s in self.symbol_list}})

    # --- 1. signal -> ordre (dimensionnement) ---
    def update_signal(self, event):
        if event.type != "SIGNAL":
            return
        order = self._generate_order(event)
        if order is not None:
            self.events.put(order)

    def _generate_order(self, signal):
        """
        Regle unifiee par POIDS CIBLE (target weight), qui gere long, short et paires.
          LONG  -> poids cible = +target_pct * strength
          SHORT -> poids cible = -target_pct * strength
          EXIT  -> poids cible = 0
        On calcule la quantite entiere cible, puis l'ecart avec la position actuelle,
        et on emet un ordre pour combler cet ecart (BUY si positif, SELL si negatif).

        Retrocompatibilite : une strategie long/flat qui emet LONG (strength=1) puis
        EXIT reproduit exactement l'ancien comportement -> la validation croisee tient.
        """
        s = signal.symbol
        price = self.data.get_latest_bar_value(s)
        cur_qty = self.positions[s]
        total_equity = self.cash + sum(
            self.positions[k] * self.data.get_latest_bar_value(k) for k in self.symbol_list)

        if signal.signal_type == "LONG":
            target_w = self.target_pct * signal.strength
        elif signal.signal_type == "SHORT":
            target_w = -self.target_pct * signal.strength
        elif signal.signal_type == "EXIT":
            target_w = 0.0
        else:
            return None

        target_qty = int(target_w * total_equity / price)
        delta = target_qty - cur_qty
        if delta != 0:
            return OrderEvent(s, "MKT", abs(delta), direction=(1 if delta > 0 else -1))
        return None

    # --- 2. fill -> mise a jour positions et cash ---
    def update_fill(self, event):
        if event.type != "FILL":
            return
        s = event.symbol
        signed_qty = event.direction * event.quantity
        self.positions[s] += signed_qty
        # cash : -notionnel a l'achat, +notionnel a la vente ; commission toujours retranchee
        self.cash -= event.direction * event.fill_cost
        self.cash -= event.commission

    # --- restitution ---
    def equity_curve(self):
        df = pd.DataFrame(self.history).set_index("datetime")
        df["returns"] = df["total"].pct_change()
        df["equity"] = df["total"] / self.initial_capital
        return df

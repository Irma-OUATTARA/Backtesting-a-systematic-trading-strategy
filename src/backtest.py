"""
backtest.py — Le moteur : la double boucle evenementielle.

ROLE CONCRET : c'est le chef d'orchestre. Il fait avancer le temps (boucle
externe = "heartbeat") et, a chaque battement, vide la file d'evenements
(boucle interne) en aiguillant chaque evenement vers le bon composant.

    Boucle externe  : tant qu'il reste des barres -> update_bars()
    Boucle interne  : tant que la file n'est pas vide -> traiter l'evenement

Ordre de traitement d'une barre :
    MARKET -> la strategie calcule ses signaux, puis on valorise le portefeuille
    SIGNAL -> le portefeuille dimensionne et emet un ordre
    ORDER  -> l'execution simule le remplissage
    FILL   -> le portefeuille met a jour positions et cash

Point cle anti look-ahead : la valorisation (update_timeindex) d'une barre est
faite AVANT que le fill de cette meme barre ne modifie les positions. La nouvelle
position ne pese donc sur le P&L qu'a partir de la barre suivante. C'est
exactement l'equivalent structurel du shift(1) du backtest vectoriel.
"""
import queue


def run_backtest(data, strategy, portfolio, execution):
    events = data.events
    while data.continue_backtest:
        data.update_bars()               # boucle externe : une barre de plus
        while True:                      # boucle interne : vider la file
            try:
                event = events.get(False)
            except queue.Empty:
                break
            if event.type == "MARKET":
                strategy.calculate_signals(event)
                portfolio.update_timeindex(event)
            elif event.type == "SIGNAL":
                portfolio.update_signal(event)
            elif event.type == "ORDER":
                execution.execute_order(event)
            elif event.type == "FILL":
                portfolio.update_fill(event)
    return portfolio.equity_curve()

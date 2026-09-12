"""
event.py — Les 4 types d'evenements qui circulent dans le moteur.

ROLE CONCRET : chaque objet est un "message" depose dans la file d'evenements.
Le moteur les traite un par un. C'est le systeme nerveux du backtester : rien
ne se passe sans qu'un evenement soit cree puis consomme.

Le cycle de vie d'une decision de trading :
    MarketEvent  -> nouvelle barre de prix disponible (emis par le DataHandler)
    SignalEvent  -> la strategie a detecte une opportunite (emis par la Strategy)
    OrderEvent   -> le portefeuille decide d'un ordre (emis par le Portfolio)
    FillEvent    -> l'ordre est execute sur le marche (emis par l'ExecutionHandler)
"""


class Event:
    """Classe de base. Sert d'interface commune aux 4 evenements."""
    pass


class MarketEvent(Event):
    """Signale l'arrivee d'une nouvelle barre de marche (nouveau jour de cotation)."""
    def __init__(self):
        self.type = "MARKET"


class SignalEvent(Event):
    """
    Signal genere par une Strategy a partir des donnees de marche.
    signal_type : 'LONG' (entrer/rester acheteur) ou 'EXIT' (sortir).
    strength    : intensite du signal (utile plus tard pour le dimensionnement).
    """
    def __init__(self, symbol, datetime, signal_type, strength=1.0):
        self.type = "SIGNAL"
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength


class OrderEvent(Event):
    """
    Ordre a transmettre au systeme d'execution.
    direction : +1 pour acheter (BUY), -1 pour vendre (SELL).
    quantity  : nombre d'actions (toujours positif).
    """
    def __init__(self, symbol, order_type, quantity, direction):
        self.type = "ORDER"
        self.symbol = symbol
        self.order_type = order_type   # 'MKT' = ordre au marche
        self.quantity = int(abs(quantity))
        self.direction = direction

    def __repr__(self):
        d = "BUY" if self.direction == 1 else "SELL"
        return f"ORDER {d} {self.quantity} {self.symbol} @ {self.order_type}"


class FillEvent(Event):
    """
    Confirmation d'execution d'un ordre.
    fill_cost  : montant notionnel de la transaction (quantite * prix), positif.
    commission : cout de transaction (0 pour l'instant, ajoute en Phase 3).
    """
    def __init__(self, timeindex, symbol, quantity, direction, fill_cost, commission=0.0):
        self.type = "FILL"
        self.timeindex = timeindex
        self.symbol = symbol
        self.quantity = int(abs(quantity))
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission

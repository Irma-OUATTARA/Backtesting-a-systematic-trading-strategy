"""
execution.py — L'ExecutionHandler : le simulateur de marche.

ROLE CONCRET : recoit les OrderEvent et les transforme en FillEvent, comme le
ferait un courtier.

Phase 2 : execution parfaite, sans cout (SimulatedExecutionHandler avec couts=0).
Phase 3 : on modelise les frictions reelles via deux parametres exprimes en
points de base (bps ; 1 bps = 0,01 %) :
  - commission_bps : commission proportionnelle au notionnel (frais de courtage) ;
  - slippage_bps   : ecart entre le cours affiche et le prix reellement obtenu
                     (on achete un peu plus cher, on vend un peu moins cher).
Un plancher de commission (min_commission) modelise le cout fixe minimal par ordre.

Grace au decouplage du moteur, TOUT le modele de cout vit ici : aucun autre
module n'est modifie. C'est le retour sur investissement de l'architecture.
"""
from abc import ABC, abstractmethod
from event import FillEvent


class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, event):
        ...


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulateur avec couts parametrables.
    Par defaut tous les couts valent 0 -> comportement identique a la Phase 2
    (retrocompatibilite : la validation croisee reste valable).
    """
    def __init__(self, data, events,
                 commission_bps=0.0, slippage_bps=0.0, min_commission=0.0):
        self.data = data
        self.events = events
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.min_commission = min_commission

    def execute_order(self, event):
        if event.type != "ORDER":
            return
        mid = self.data.get_latest_bar_value(event.symbol)

        # 1. Slippage : le prix d'execution est defavorable.
        #    Achat (direction=+1) -> on paie plus cher ; vente (-1) -> on recoit moins.
        fill_price = mid * (1.0 + event.direction * self.slippage_bps / 1e4)
        fill_cost = event.quantity * fill_price          # notionnel reellement echange

        # 2. Commission : proportionnelle au notionnel, avec un plancher.
        commission = max(self.min_commission,
                         self.commission_bps / 1e4 * event.quantity * mid)

        dt = self.data.get_latest_bar_datetime(event.symbol)
        self.events.put(FillEvent(dt, event.symbol, event.quantity,
                                  event.direction, fill_cost, commission))

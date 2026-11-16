"""(8e) The same, but a hunter never attacks another hunter: is it hunters who kill the young?"""
import sys
sys.argv = ['x']
sys.path.insert(0, 'scripts')
import numpy as np
from eco_hunter_check import Scripted
from ailearns.games.ecosystem import World, WorldConfig
from ailearns.games.ecosystem_life import LifeConfig
class S(Scripted):
    def think(self, i, x):
        out = super().think(i, x)
        h = self.diet[i] >= 0.99
        tgt = self._ahead
        kin = np.zeros(len(i), bool)
        ok = tgt >= 0
        kin[ok] = self.diet[i[tgt[ok]]] >= 0.99
        out[h & kin, 4] = -9.0           # never attack another hunter
        return out
    tally = {}
    def _deaths(self, i, killed=None):
        k = np.zeros(len(i), bool) if killed is None else killed
        h = self.diet[i] >= 0.99
        young = self.mass[i] < 0.5 * self.adult[i]
        for name, m in (("killed", k), ("starved", ~k & (self.reserve[i] < 0)), ("old", ~k & (self.reserve[i] >= 0) & (self.age[i] > self.lifespan(self.adult[i])))):
            for age, y in (("young", young), ("grown", ~young)):
                self.tally[(name, age)] = self.tally.get((name, age), 0) + int((m & h & y).sum())
        super()._deaths(i, killed)
w = World.make(1, WorldConfig()); life = S(w, 0, 1, LifeConfig()); life.add(1000, diet=(0.0, 0.1))
for _ in range(240 * 60): life.step()
life.cfg = LifeConfig(sigma_diet=0.0); idx = life.add(50, diet=(1.0, 1.0)); life.alone[idx] = 1.0
b0 = life.books.births
for _ in range(180 * 60): life.step()
hu = life.alive & (life.diet >= 0.99)
print("hunter deaths by cause:", {k: v for k, v in S.tally.items() if v}, "alive:", hu.sum())
print("all births in world:", life.books.births - b0, " hunter tissue kg (med adult):", np.median(life.adult[idx]))

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import random
SEED=42
Počet_iterací=10000
Počet_iterací2=10000
#Hlavní informace
random.seed(SEED)
duch=90 #90 duchů
stab_uspech=1/2
kill_bez_s=1/2
kill_s_s=1/2+1/2*1/2 #+ přehození 1čky
risk_s=(1/6*2/6)+1/6*1/2 
risk_bez_s=1/6*1/2
bod_duch=15
duch_ork=15/9
ork=150 #150 orků
enemy=9
nahoda=[]
# Vytvořte seznamy, do kterých budete ukládat hodnoty pro každou simulaci
value_ork = []
value_duch = []
value_ork_bez=[]
value_duch_bez=[]

# Se stabem
for i in range(Počet_iterací):
    # Inicializujte hodnoty proměnných ork a duch na jejich počáteční hodnoty
    ork = 150
    duch = 90
    # Provádějte simulaci, dokud nejsou orkové nebo duchové vyčerpáni
    while (ork and duch) <= 150 and (ork and duch) > 0:
        nahoda=random.randint(1,2)
        if nahoda==1: #vyhrály duchové
            ork=ork-(duch)*kill_s_s
        else: #vyhraly orkove
            duch=duch-(ork)*risk_s
        if (ork or duch) <= 0:
            break
    # Uložte hodnoty proměnných ork a duch do seznamů pro další zpracování
    value_ork.append(ork*9)
    value_duch.append(duch*15)

# Bez stabu    
for i in range(Počet_iterací2):
    # Inicializujte hodnoty proměnných ork a duch na jejich počáteční hodnoty
    ork2 = 150
    duch2 = 90
    # Provádějte simulaci, dokud nejsou orkové nebo duchové vyčerpáni
    while (ork2 and duch2) <= 150 and (ork2 and duch2) > 0:
        nahoda=random.randint(1,2)
        if nahoda==1: #vyhrály duchové
            ork2=ork2-(duch2)*kill_bez_s
        else: #vyhraly orkove
            duch2=duch2-(ork2)*risk_bez_s
        if (ork2 or duch2) <= 0:
            break
    # Uložte hodnoty proměnných ork a duch do seznamů pro další zpracování
    value_ork_bez.append(ork2*9)
    value_duch_bez.append(duch2*15)
plt.hist(value_ork_bez)
plt.hist(value_duch_bez)
plt.hist(value_ork)
plt.hist(value_duch)
print(kill_bez_s) #killrate bez stabu
print(kill_s_s) #killrate se stabem
print(risk_bez_s) #smrtrate bez stabu
print(risk) #smrtrate se stabem
print(duch_ork) #poměr value orka k duchovi

# Výpočet procenta výher orků a duchů
pocet_vyher_orku = 0
pocet_vyher_duchu = 0
for i in range(Počet_iterací):
    # Pokud je počet orků větší než 0, znamená to, že orkové vyhráli
    if value_ork[i] > 0:
        pocet_vyher_orku += 1
    # Pokud je počet duchů větší než 0, znamená to, že duchové vyhráli
    if value_duch[i] > 0:
        pocet_vyher_duchu += 1

# Výpočet procenta výher orků a duchů bez stabu
pocet_vyher_orku_bez = 0
pocet_vyher_duchu_bez = 0
for i in range(Počet_iterací2):
    # Pokud je počet orků větší než 0, znamená to, že orkové vyhráli
    if value_ork_bez[i] > 0:
        pocet_vyher_orku_bez += 1
    # Pokud je počet duchů větší než 0, znamená to, že duchové vyhráli
    if value_duch_bez[i] > 0:
        pocet_vyher_duchu_bez += 1

# Výpočet procenta výher orků a duchů
procento_vyher_orku = pocet_vyher_orku / Počet_iterací * 100
procento_vyher_duchu = pocet_vyher_duchu / Počet_iterací * 100

# Výpočet procenta výher orků a duchů bez stabu
procento_vyher_orku_bez = pocet_vyher_orku_bez / Počet_iterací2 * 100
procento_vyher_duchu_bez = pocet_vyher_duchu_bez / Počet_iterací2 * 100

print("Bez stabu vyhráli orkové tolikrát:", procento_vyher_orku_bez)
print("Bez stabu vyhráli duchove tolikrát:", procento_vyher_duchu_bez)
print("Se stabem vyhráli duchu tolikrát:", procento_vyher_duchu)
print("Se stabem vyhráli orkové tolikrát:", procento_vyher_orku)


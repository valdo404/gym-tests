# POC: Optimisation Prix & Contenus Airbnb avec Contextual Bandits

## 🎯 Objectif

Optimiser dynamiquement les **prix** et **variantes de contenu** des listings Airbnb en utilisant un algorithme de **contextual bandit** (MABWiser LinUCB).

## 🚀 Résultats

**+3.1% d'amélioration du revenu** vs stratégie fixe (prix de base, contenu neutre)

| Méthode | Revenu Total | Taux Conversion |
|---------|--------------|-----------------|
| Bandit Contextuel (LinUCB) | **25,830 €** | 18.56% |
| Baseline (fixe) | 25,050 € | 18.56% |

## 🏗️ Architecture

### Framework: MABWiser (Fidelity Investments)
- **Algorithme**: LinUCB (Linear Upper Confidence Bound)
- **Type**: Contextual Bandit
- **Apprentissage**: Online learning (incrémental)

### Actions Testées
1. **Prix**: 3 niveaux
   - -10% (135€)
   - Base (150€)
   - +10% (165€)

2. **Contenu**: 3 variantes
   - `cozy_local` - Style chaleureux et local
   - `luxury_premium` - Style premium et luxe
   - `practical_clean` - Style pratique et fonctionnel

### Contexte (Features)
- **Saison** (one-hot: hiver, printemps, été, automne)
- **Weekend** (booléen)
- **Jours avant réservation** (1-30)
- **Type de logement** (apartment, house, studio)

## 📊 Fonctionnement

Le bandit apprend à:
- 📉 **Baisser les prix** en basse saison
- 📈 **Augmenter les prix** en haute saison/weekends
- 📝 **Choisir le contenu optimal** selon le contexte
  - "Cozy" marche mieux en hiver
  - "Luxury" marche mieux en été et weekends
  - "Practical" est neutre et stable

## 🔧 Utilisation

```bash
# Installer les dépendances
poetry install

# Lancer le POC
poetry run python airbnb_bandit_poc.py
```

### Résultats
- Console: métriques de performance
- `airbnb_bandit_results.png`: visualisations
  - Revenu cumulé
  - Taux de conversion
  - Distribution des prix
  - Distribution des contenus

## 🎓 Concepts Clés

### Contextual Bandit
Problème d'apprentissage par renforcement simplifié où :
- **Agent** choisit une **action** (prix + contenu)
- Dans un **contexte** (saison, weekend, etc.)
- Reçoit une **récompense** (revenu si réservation)
- Apprend à **maximiser le revenu total**

### LinUCB (Linear UCB)
- Modèle linéaire pour estimer les récompenses
- **Exploration** via upper confidence bound
- **Exploitation** des actions prometteuses
- Équilibre automatique exploration/exploitation

### Online Learning
- Le modèle **s'améliore en continu**
- Chaque interaction = nouvelle donnée d'entraînement
- Pas besoin de réentraînement batch
- Adaptation rapide aux changements

## 🔄 Améliorations Possibles

### 1. Plus d'actions
```python
# Prix plus granulaire
price_actions = [-0.20, -0.10, 0.0, 0.10, 0.20, 0.30]

# Plus de variantes de contenu
content_variants = [
    "cozy_local",
    "luxury_premium",
    "practical_clean",
    "family_friendly",
    "business_modern"
]
```

### 2. Contexte enrichi
```python
# Ajouter:
- Distance au centre-ville
- Note du listing
- Nombre de reviews
- Prix des concurrents
- Événements locaux
- Météo prévue
```

### 3. Algorithmes alternatifs

**Thompson Sampling** (MABWiser)
```python
from mabwiser.mab import LearningPolicy

mab = MAB(
    arms=actions,
    learning_policy=LearningPolicy.LinTS()  # Linear Thompson Sampling
)
```

**Neural Contextual Bandit**
```python
# Pour contextes complexes (images, textes)
from mabwiser.mab import LearningPolicy

mab = MAB(
    arms=actions,
    learning_policy=LearningPolicy.UCB1()  # Peut être combiné avec DNN
)
```

### 4. Production

**A/B Testing avec bandits**
```python
# Allocation dynamique du trafic
- 90% trafic → bandit (exploitation)
- 10% trafic → exploration random
```

**Logging pour audit**
```python
# Logger chaque décision
{
    "timestamp": "2024-01-01 10:30",
    "listing_id": "123",
    "context": {...},
    "action": "prix +10%, luxury",
    "reward": 165.0,
    "booked": true
}
```

**Monitoring**
```python
# Métriques à suivre:
- Revenu par jour
- Taux de conversion
- Distribution des actions
- Regret cumulé
- Exploration rate
```

### 5. Optimisation Multi-objectifs

```python
# Maximiser revenu ET satisfaction client
reward = 0.7 * revenue + 0.3 * customer_satisfaction_score

# Contraintes
if price_too_high:
    reward *= 0.5  # Pénalité
```

## 📚 Ressources

### MABWiser
- [Documentation](https://fidelity.github.io/mabwiser/)
- [GitHub](https://github.com/fidelity/mabwiser)
- [Paper](https://arxiv.org/abs/2011.11524)

### Contextual Bandits
- [Introduction](https://www.findingtheta.com/blog/ultimate-guide-to-contextual-bandits-from-theory-to-python-implementation)
- [A Contextual Bandit Bake-off](https://arxiv.org/abs/1802.04064)

### Autres frameworks
- [Vowpal Wabbit](https://vowpalwabbit.org/) - C++, ultra rapide
- [TF-Agents](https://www.tensorflow.org/agents) - TensorFlow
- [contextualbandits](https://github.com/david-cortes/contextualbandits) - Python

## 🎯 Cas d'usage réels

### E-commerce
- Prix dynamiques
- Recommandations produits
- Layouts de page

### Marketing
- Allocation budget publicitaire
- Personnalisation emails
- A/B testing dynamique

### FinTech
- Portfolio allocation
- Fraud detection
- Loan pricing

### Media
- Recommandation contenu
- Placement publicité
- Paywall optimization

## 🤝 Contribution

Ce POC est une base pour expérimenter. Améliorations bienvenues :
- [ ] Tester Thompson Sampling vs LinUCB
- [ ] Ajouter des contraintes business
- [ ] Intégrer des données réelles
- [ ] Comparer avec Vowpal Wabbit
- [ ] Déployer en production avec monitoring

---

**Créé le 26 décembre 2024 à 21h43** 🌙

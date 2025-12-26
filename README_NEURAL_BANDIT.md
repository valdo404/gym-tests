# Neural Contextual Bandits vs LinUCB - Comparaison

## 🧠 Objectif

Comparer **LinUCB** (linéaire) vs **Neural Bandit** (deep learning) pour l'optimisation de prix et contenus Airbnb.

## 📊 Résultats

### Performance (moyenne sur 3 runs)

| Algorithme | Revenu moyen | Écart-type | Taux conversion |
|------------|--------------|------------|-----------------|
| **LinUCB** | **30,035 €** | ±3,298 | 23.67% |
| Neural Bandit | 28,115 € | ±761 | 22.19% |

**Verdict: LinUCB est 6.4% meilleur** ✅

### Observations

**LinUCB (Linear UCB):**
- ✅ **Meilleure performance** (+6.4%)
- ⚠️ Variance plus élevée (±3,298)
- ⚡ Très rapide (pas d'entraînement neural)
- 🎯 Simple et efficace

**Neural Bandit (MLP Ensemble):**
- ⚖️ Performance légèrement inférieure
- ✅ **Variance plus faible** (±761) - plus stable
- 🐌 Plus lent (entraînement MLP)
- 🔧 Plus complexe à tuner

## 💡 Conclusion

**Pour ce problème Airbnb: LinUCB gagne**

Le problème d'optimisation prix/contenu Airbnb est **essentiellement linéaire**. Les features contextuelles (saison, weekend, type de logement) ont des interactions linéaires avec le revenu.

### Pourquoi LinUCB est meilleur ici?

1. **Problème linéaire**: Les relations contexte → récompense sont linéaires
2. **Peu de données**: 900 interactions (90 jours × 10/jour) favorise modèles simples
3. **Features simples**: Saison, weekend, type de logement (9 features)
4. **Pas d'interactions complexes**: Pas de patterns non-linéaires subtils

### Quand utiliser Neural Bandits?

Les Neural Bandits sont meilleurs quand :

#### 1. **Interactions non-linéaires complexes**
```python
# Exemple: prix optimal dépend d'interactions complexes
if (saison == été AND weekend AND proximité_plage < 500m AND note > 4.5):
    prix_optimal = très_élevé
elif (saison == hiver AND ski_nearby AND groupe > 6):
    prix_optimal = élevé
# ... patterns complexes
```

#### 2. **Features riches et haute dimension**
- **Images**: Photos du listing (CNN)
- **Texte**: Descriptions (embeddings, transformers)
- **Séries temporelles**: Historique de réservations
- **Graphes**: Réseau social de reviews

#### 3. **Datasets massifs**
- Millions d'interactions
- Les réseaux de neurones excellent avec beaucoup de données
- LinUCB peut plafonner

#### 4. **Patterns évolutifs**
- Tendances qui changent dans le temps
- Neural bandits s'adaptent mieux aux shifts

## 🏗️ Architecture Neural Bandit

### Implémentation

```python
class NeuralBandit:
    """
    Neural Bandit avec ensemble de MLPRegressors.

    Stratégie:
    - Un ensemble de 5 réseaux par action
    - Bootstrap pour diversité
    - UCB via incertitude de l'ensemble
    """

    def __init__(self):
        # 9 actions × 5 modèles = 45 MLPs
        for action in actions:
            ensemble = [
                MLPRegressor(
                    hidden_layers=(64, 32),
                    activation='relu'
                )
                for _ in range(5)
            ]
            self.models.append(ensemble)

    def predict_with_uncertainty(self, context):
        # Prédire avec chaque modèle de l'ensemble
        predictions = [model.predict(context) for model in ensemble]

        mean = np.mean(predictions)  # Estimation
        std = np.std(predictions)    # Incertitude

        return mean, std

    def select_action(self, context):
        # Upper Confidence Bound
        ucb_scores = mean + exploration_coef * std
        return argmax(ucb_scores)
```

### Hyperparamètres testés

```python
{
    "hidden_layers": (64, 32),
    "n_ensembles": 5,
    "exploration_coef": 1.5,
    "update_frequency": 20,  # tous les 20 interactions
    "warmup_size": 50,
    "learning_rate": 0.001
}
```

## 📈 Visualisations

Voir `airbnb_neural_vs_linucb.png` pour:
- Revenu cumulé comparé
- Taux de conversion
- Distribution des stratégies de prix
- Barplot performance moyenne

## 🔧 Usage

```bash
# Lancer la comparaison
poetry run python airbnb_neural_bandit_poc.py
```

### Résultats
- Console: métriques détaillées
- `airbnb_neural_vs_linucb.png`: visualisations comparatives

## 🎓 Leçons apprises

### 1. Simple > Complexe (souvent)
Pour ce POC, LinUCB (linear) bat Neural Bandit. La complexité n'est pas toujours meilleure.

### 2. Diagnostiquer avant d'optimiser
Toujours tester un modèle simple (LinUCB) avant d'essayer du deep learning.

### 3. Variance vs Biais
- LinUCB: haute variance, meilleure performance moyenne
- Neural: basse variance, plus stable mais moins performant

### 4. Contexte matters
Le choix LinUCB vs Neural dépend fortement de:
- Nature du problème (linéaire vs non-linéaire)
- Quantité de données
- Richesse des features
- Contraintes de latence

## 🚀 Améliorations possibles

### Pour améliorer le Neural Bandit

1. **Plus de données**
```python
# Augmenter les interactions
n_days = 365  # 1 an au lieu de 90 jours
interactions_per_day = 50  # Au lieu de 10
```

2. **Features plus riches**
```python
context = [
    season_onehot,
    is_weekend,
    days_before,
    listing_type,
    # Nouveautés:
    photo_embeddings,  # CNN sur photos
    description_embeddings,  # BERT sur texte
    competitor_prices,  # Prix concurrents
    local_events,  # Événements dans la ville
    weather_forecast  # Météo prévue
]
```

3. **Architecture plus sophistiquée**
```python
# Utiliser des réseaux plus profonds
hidden_layers = (128, 64, 32)

# Ou des architectures spécialisées
- Attention mechanisms
- Residual connections
- Batch normalization
```

4. **Tuning d'hyperparamètres**
```python
# Optimisation bayésienne avec Optuna
study = optuna.create_study()
study.optimize(objective_function, n_trials=100)
```

## 📚 Références

### Papiers académiques
- [Neural Contextual Bandits with UCB](https://arxiv.org/abs/1911.04462)
- [Deep Bayesian Bandits Showdown](https://arxiv.org/abs/1802.09127)
- [A Contextual Bandit Bake-off](https://arxiv.org/abs/1802.04064)

### Frameworks
- [MABWiser](https://github.com/fidelity/mabwiser) - LinUCB (utilisé)
- [scikit-learn](https://scikit-learn.org/) - MLPRegressor (utilisé)
- [PyTorch Deep Bayesian Bandits](https://github.com/andrewk1/pytorch-deep-bayesian-bandits)
- [TF-Agents](https://www.tensorflow.org/agents)

## 🎯 Cas d'usage réels

### Quand LinUCB est optimal
- **E-commerce**: Prix dynamiques simples
- **Ads**: Allocation de budget publicitaire
- **Content**: Recommandation d'articles
- **Finance**: Portfolio allocation simple

### Quand Neural Bandits brillent
- **Recommendation**: Netflix, YouTube (features complexes)
- **Ads**: Google Ads (embeddings, images, texte)
- **Robotics**: Contrôle adaptatif avec vision
- **Healthcare**: Traitement personnalisé (données riches)

## 🔍 Debugging tips

### Neural Bandit n'apprend pas?

1. **Vérifier la convergence**
```python
# Augmenter max_iter
MLPRegressor(max_iter=500)  # Au lieu de 100
```

2. **Normalisation**
```python
# Toujours normaliser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

3. **Learning rate**
```python
# Essayer différents learning rates
learning_rate_init = [0.001, 0.01, 0.0001]
```

4. **Exploration**
```python
# Augmenter l'exploration
exploration_coef = 2.5  # Au lieu de 1.5
```

## ⚖️ Table de décision

| Critère | LinUCB | Neural Bandit |
|---------|--------|---------------|
| Problème linéaire | ✅ **Excellent** | ⚠️ Overkill |
| Problème non-linéaire | ❌ Limité | ✅ **Excellent** |
| < 10K interactions | ✅ **Excellent** | ❌ Pas assez de données |
| > 100K interactions | ✅ Bon | ✅ **Excellent** |
| Features simples (<20) | ✅ **Excellent** | ⚠️ Overkill |
| Features riches (>100) | ❌ Limité | ✅ **Excellent** |
| Latence critique | ✅ **Très rapide** | ⚠️ Plus lent |
| Interprétabilité | ✅ **Transparent** | ❌ Boîte noire |
| Stabilité | ⚠️ Variance élevée | ✅ **Stable** |

---

**Créé le 26 décembre 2024 à 21h43** 🌙

**TL;DR**: Pour l'optimisation Airbnb, LinUCB gagne. Neural Bandits brillent avec des features complexes et beaucoup de données. Always start simple!

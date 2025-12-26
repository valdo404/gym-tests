"""
POC: Neural Contextual Bandits pour Airbnb

Compare LinUCB (linéaire) vs Neural Bandit (deep learning avec scikit-learn)
sur le même problème d'optimisation prix/contenu.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Importer le simulateur du POC précédent
import sys
sys.path.insert(0, '/home/user/gym-tests')
from airbnb_bandit_poc import AirbnbListingSimulator


class NeuralBandit:
    """
    Neural Bandit utilisant MLPRegressor de scikit-learn.

    Stratégie:
    - Un réseau de neurones par action
    - Estimation de l'incertitude via bootstrap (multiple fits)
    - UCB: mean + exploration_coef * std
    """

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        hidden_layers: Tuple[int] = (64, 32),
        learning_rate: float = 0.001,
        exploration_coef: float = 2.0,
        n_ensembles: int = 5
    ):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.exploration_coef = exploration_coef
        self.n_ensembles = n_ensembles

        # Créer un ensemble de modèles par action (pour l'incertitude)
        self.models = []
        for _ in range(n_actions):
            ensemble = []
            for _ in range(n_ensembles):
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation='relu',
                    solver='adam',
                    learning_rate_init=learning_rate,
                    max_iter=100,
                    warm_start=True,  # Important pour l'apprentissage incrémental
                    random_state=np.random.randint(0, 10000)
                )
                ensemble.append(model)
            self.models.append(ensemble)

        # Scaler pour normaliser les features
        self.scalers = [StandardScaler() for _ in range(n_actions)]

        # Tracking
        self.is_fitted = [False] * n_actions
        self.training_counts = [0] * n_actions

    def predict_with_uncertainty(
        self,
        context: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédit la récompense et l'incertitude pour chaque action.

        L'incertitude est estimée via un ensemble de modèles:
        - Mean = moyenne des prédictions de l'ensemble
        - Std = écart-type des prédictions de l'ensemble
        """
        context = context.reshape(1, -1)

        mean_rewards = np.zeros(self.n_actions)
        std_rewards = np.zeros(self.n_actions)

        for action_idx in range(self.n_actions):
            if not self.is_fitted[action_idx]:
                # Pas encore entraîné, retourne des valeurs optimistes
                mean_rewards[action_idx] = 150.0  # Prix moyen optimiste
                std_rewards[action_idx] = 50.0    # Incertitude élevée
            else:
                # Normaliser le contexte
                context_scaled = self.scalers[action_idx].transform(context)

                # Prédire avec chaque modèle de l'ensemble
                predictions = []
                for model in self.models[action_idx]:
                    pred = model.predict(context_scaled)[0]
                    predictions.append(pred)

                mean_rewards[action_idx] = np.mean(predictions)
                std_rewards[action_idx] = np.std(predictions)

        return mean_rewards, std_rewards

    def select_action(self, context: np.ndarray) -> int:
        """
        Sélectionne l'action selon UCB:
        action = argmax(mean_reward + exploration_coef * std_reward)
        """
        mean_rewards, std_rewards = self.predict_with_uncertainty(context)

        # UCB score
        ucb_scores = mean_rewards + self.exploration_coef * std_rewards

        return int(np.argmax(ucb_scores))

    def update(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray
    ):
        """
        Met à jour les modèles avec les nouvelles observations.

        Pour chaque action, entraîne son ensemble de modèles sur les contextes
        où cette action a été choisie.
        """
        for action_idx in range(self.n_actions):
            # Trouver les indices où cette action a été choisie
            mask = actions == action_idx

            if mask.sum() < 5:  # Besoin d'au moins 5 observations
                continue

            action_contexts = contexts[mask]
            action_rewards = rewards[mask]

            # Normaliser les contextes
            if not self.is_fitted[action_idx]:
                self.scalers[action_idx].fit(action_contexts)
                self.is_fitted[action_idx] = True

            action_contexts_scaled = self.scalers[action_idx].transform(action_contexts)

            # Entraîner chaque modèle de l'ensemble avec bootstrap
            for model in self.models[action_idx]:
                # Bootstrap: échantillonner avec remplacement
                n_samples = len(action_contexts_scaled)
                bootstrap_indices = np.random.choice(
                    n_samples, size=n_samples, replace=True
                )

                X_boot = action_contexts_scaled[bootstrap_indices]
                y_boot = action_rewards[bootstrap_indices]

                model.fit(X_boot, y_boot)

            self.training_counts[action_idx] += 1


class AirbnbNeuralBanditOptimizer:
    """
    Optimiseur utilisant un Neural Bandit.
    """

    def __init__(self, simulator: AirbnbListingSimulator, exploration_coef: float = 2.0):
        self.simulator = simulator

        # Créer les actions combinées
        self.actions = []
        for price_mod in simulator.price_actions:
            for content in simulator.content_variants:
                self.actions.append((price_mod, content))

        # Dimension du contexte
        sample_context = simulator.get_context(datetime(2024, 1, 1))
        context_dim = len(sample_context)

        # Neural Bandit
        self.neural_bandit = NeuralBandit(
            n_actions=len(self.actions),
            context_dim=context_dim,
            hidden_layers=(64, 32),
            learning_rate=0.001,
            exploration_coef=exploration_coef,
            n_ensembles=5
        )

        self.history = []

    def train_and_evaluate(
        self,
        start_date: datetime,
        n_days: int,
        interactions_per_day: int = 10,
        warmup_size: int = 50,
        update_frequency: int = 20  # Update tous les N interactions
    ) -> pd.DataFrame:
        """
        Entraînement en ligne du neural bandit.
        """
        all_contexts = []
        all_actions = []
        all_rewards = []

        current_date = start_date
        interaction_count = 0

        for day in range(n_days):
            for interaction in range(interactions_per_day):
                context = self.simulator.get_context(current_date)

                # Choisir l'action
                if interaction_count < warmup_size:
                    # Phase de warm-up: exploration aléatoire
                    action_idx = np.random.randint(0, len(self.actions))
                else:
                    # Phase d'exploitation: utiliser le neural bandit
                    action_idx = self.neural_bandit.select_action(context)

                price_mod, content = self.actions[action_idx]

                # Simuler le résultat
                reward, was_booked = self.simulator.get_reward(
                    context, price_mod, content
                )

                # Enregistrer
                all_contexts.append(context)
                all_actions.append(action_idx)
                all_rewards.append(reward)

                self.history.append({
                    'day': day,
                    'date': current_date,
                    'context_features': context,
                    'action_idx': action_idx,
                    'price_modifier': price_mod,
                    'content_variant': content,
                    'reward': reward,
                    'was_booked': was_booked,
                    'actual_price': self.simulator.base_price * (1 + price_mod)
                })

                interaction_count += 1

                # Update du modèle tous les N interactions
                if interaction_count >= warmup_size and \
                   interaction_count % update_frequency == 0:
                    self.neural_bandit.update(
                        np.array(all_contexts),
                        np.array(all_actions),
                        np.array(all_rewards)
                    )

            current_date += timedelta(days=1)

        return pd.DataFrame(self.history)


def compare_linucb_vs_neural(
    simulator: AirbnbListingSimulator,
    start_date: datetime,
    n_days: int,
    interactions_per_day: int = 10,
    n_runs: int = 3
) -> Dict:
    """
    Compare LinUCB vs Neural Bandit sur plusieurs runs.
    """
    print("🔬 Comparaison: LinUCB vs Neural Bandit")
    print("=" * 70)
    print(f"📅 {n_days} jours, {interactions_per_day} interactions/jour")
    print(f"🔄 {n_runs} runs pour moyenner les résultats\n")

    linucb_results = []
    neural_results = []

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        # Reset random seed pour reproductibilité
        np.random.seed(42 + run)

        # Test LinUCB
        print("⚙️  LinUCB...")
        from airbnb_bandit_poc import AirbnbBanditOptimizer

        linucb_optimizer = AirbnbBanditOptimizer(simulator)
        df_linucb = linucb_optimizer.train_and_evaluate(
            start_date, n_days, interactions_per_day
        )
        linucb_revenue = df_linucb['reward'].sum()
        linucb_conversion = df_linucb['was_booked'].mean()
        linucb_results.append({
            'revenue': linucb_revenue,
            'conversion': linucb_conversion,
            'df': df_linucb
        })
        print(f"   Revenue: {linucb_revenue:,.0f} €, Conversion: {linucb_conversion:.2%}")

        # Reset random seed
        np.random.seed(42 + run)

        # Test Neural Bandit
        print("⚙️  Neural Bandit...")
        neural_optimizer = AirbnbNeuralBanditOptimizer(simulator, exploration_coef=1.5)
        df_neural = neural_optimizer.train_and_evaluate(
            start_date, n_days, interactions_per_day
        )
        neural_revenue = df_neural['reward'].sum()
        neural_conversion = df_neural['was_booked'].mean()
        neural_results.append({
            'revenue': neural_revenue,
            'conversion': neural_conversion,
            'df': df_neural
        })
        print(f"   Revenue: {neural_revenue:,.0f} €, Conversion: {neural_conversion:.2%}")

    # Moyennes
    linucb_avg_revenue = np.mean([r['revenue'] for r in linucb_results])
    linucb_std_revenue = np.std([r['revenue'] for r in linucb_results])
    linucb_avg_conversion = np.mean([r['conversion'] for r in linucb_results])

    neural_avg_revenue = np.mean([r['revenue'] for r in neural_results])
    neural_std_revenue = np.std([r['revenue'] for r in neural_results])
    neural_avg_conversion = np.mean([r['conversion'] for r in neural_results])

    # Résultats
    print("\n" + "=" * 70)
    print("📊 RÉSULTATS (moyenne sur {} runs)".format(n_runs))
    print("=" * 70)

    print(f"\n🔵 LinUCB (Linear):")
    print(f"   Revenu moyen: {linucb_avg_revenue:,.0f} € (±{linucb_std_revenue:.0f})")
    print(f"   Taux conversion: {linucb_avg_conversion:.2%}")

    print(f"\n🟣 Neural Bandit (MLP Ensemble):")
    print(f"   Revenu moyen: {neural_avg_revenue:,.0f} € (±{neural_std_revenue:.0f})")
    print(f"   Taux conversion: {neural_avg_conversion:.2%}")

    improvement = (neural_avg_revenue - linucb_avg_revenue) / linucb_avg_revenue * 100
    print(f"\n{'🚀' if improvement > 0 else '📉'} Neural vs LinUCB: {improvement:+.1f}%")

    if improvement > 1:
        print("   ✅ Neural Bandit est MEILLEUR")
    elif improvement < -1:
        print("   ✅ LinUCB est MEILLEUR")
    else:
        print("   ⚖️  Performance SIMILAIRE")

    print("\n💡 Analyse:")
    if improvement > 1:
        print("   Le neural bandit capture mieux les non-linéarités")
    elif improvement < -1:
        print("   Le problème est probablement linéaire, LinUCB suffit")
    else:
        print("   Les deux approches sont équivalentes pour ce problème")

    print("\n" + "=" * 70)

    return {
        'linucb': {
            'avg_revenue': linucb_avg_revenue,
            'std_revenue': linucb_std_revenue,
            'avg_conversion': linucb_avg_conversion,
            'results': linucb_results
        },
        'neural': {
            'avg_revenue': neural_avg_revenue,
            'std_revenue': neural_std_revenue,
            'avg_conversion': neural_avg_conversion,
            'results': neural_results
        },
        'improvement_pct': improvement
    }


def plot_comparison(comparison_results: Dict):
    """Visualise la comparaison LinUCB vs Neural."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    linucb_results = comparison_results['linucb']['results']
    neural_results = comparison_results['neural']['results']

    # On prend le premier run pour les graphiques détaillés
    df_linucb = linucb_results[0]['df']
    df_neural = neural_results[0]['df']

    # 1. Revenu cumulé comparé
    ax = axes[0, 0]

    cumrev_linucb = df_linucb.groupby('day')['reward'].sum().cumsum()
    cumrev_neural = df_neural.groupby('day')['reward'].sum().cumsum()

    ax.plot(cumrev_linucb.index, cumrev_linucb.values,
            label='LinUCB (Linear)', linewidth=2.5, color='#3498db')
    ax.plot(cumrev_neural.index, cumrev_neural.values,
            label='Neural Bandit (MLP)', linewidth=2.5, color='#9b59b6', linestyle='--')

    ax.set_xlabel('Jour', fontsize=11)
    ax.set_ylabel('Revenu cumulé (€)', fontsize=11)
    ax.set_title('Revenu cumulé: LinUCB vs Neural Bandit', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Taux de conversion comparé
    ax = axes[0, 1]

    # Moving average pour lisser
    window = 5
    conv_linucb = df_linucb.groupby('day')['was_booked'].mean().rolling(window).mean()
    conv_neural = df_neural.groupby('day')['was_booked'].mean().rolling(window).mean()

    ax.plot(conv_linucb.index, conv_linucb.values,
            label='LinUCB', linewidth=2.5, color='#3498db')
    ax.plot(conv_neural.index, conv_neural.values,
            label='Neural Bandit', linewidth=2.5, color='#9b59b6', linestyle='--')

    ax.set_xlabel('Jour', fontsize=11)
    ax.set_ylabel('Taux de conversion (MA-5)', fontsize=11)
    ax.set_title('Taux de conversion (moyenne mobile 5 jours)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, None])

    # 3. Barplot comparaison revenue moyen
    ax = axes[1, 0]

    revenues = [
        comparison_results['linucb']['avg_revenue'],
        comparison_results['neural']['avg_revenue']
    ]
    stds = [
        comparison_results['linucb']['std_revenue'],
        comparison_results['neural']['std_revenue']
    ]

    bars = ax.bar(['LinUCB\n(Linear)', 'Neural Bandit\n(MLP Ensemble)'], revenues,
                   color=['#3498db', '#9b59b6'], yerr=stds, capsize=10, alpha=0.8)

    ax.set_ylabel('Revenu moyen (€)', fontsize=11)
    ax.set_title('Revenu moyen sur 3 runs (±écart-type)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Ajouter les valeurs sur les barres
    for i, (bar, rev) in enumerate(zip(bars, revenues)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rev:,.0f} €',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 4. Comparaison distribution des prix
    ax = axes[1, 1]

    price_dist_linucb = df_linucb['price_modifier'].value_counts().sort_index()
    price_dist_neural = df_neural['price_modifier'].value_counts().sort_index()

    x = np.arange(len(price_dist_linucb))
    width = 0.35

    ax.bar(x - width/2, price_dist_linucb.values, width,
           label='LinUCB', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, price_dist_neural.values, width,
           label='Neural', color='#9b59b6', alpha=0.8)

    ax.set_xlabel('Modificateur de prix', fontsize=11)
    ax.set_ylabel('Fréquence', fontsize=11)
    ax.set_title('Distribution des stratégies de prix', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:+.0%}" for p in price_dist_linucb.index])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('airbnb_neural_vs_linucb.png', dpi=150, bbox_inches='tight')
    print("\n📊 Graphiques sauvegardés: 'airbnb_neural_vs_linucb.png'")

    return fig


if __name__ == "__main__":
    print("🧠 POC: Neural Contextual Bandits pour Airbnb")
    print("=" * 70)
    print()

    # Configuration
    np.random.seed(42)

    # Même simulateur que LinUCB
    simulator = AirbnbListingSimulator(
        base_price=150,
        listing_type="apartment"
    )

    start_date = datetime(2024, 1, 1)
    n_days = 90
    interactions_per_day = 10
    n_runs = 3

    print(f"📅 Période: {n_days} jours")
    print(f"🎯 {interactions_per_day} interactions/jour")
    print(f"🔄 {n_runs} runs pour robustesse")
    print(f"💰 Prix de base: {simulator.base_price}€/nuit")
    print(f"🎲 Actions: {len(simulator.price_actions)} prix × {len(simulator.content_variants)} contenus = 9 total")
    print()

    # Comparaison
    results = compare_linucb_vs_neural(
        simulator,
        start_date,
        n_days,
        interactions_per_day,
        n_runs
    )

    # Visualisation
    print("\n📈 Génération des visualisations...")
    plot_comparison(results)

    print("\n✅ POC Neural Bandit terminé!")
    print("\nℹ️  Neural Bandit config:")
    print("   - Architecture: MLP [64, 32] hidden units")
    print("   - Ensemble: 5 modèles (bootstrap)")
    print("   - UCB: mean + 1.5 * std")
    print("   - Update: tous les 20 interactions")

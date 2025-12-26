"""
POC: Optimisation de prix et contenus Airbnb avec Contextual Bandits (MABWiser)

Ce POC simule l'optimisation dynamique de:
- Prix des listings (3 niveaux: -10%, base, +10%)
- Variantes de contenu (3 styles de titres/descriptions)

Basé sur des contextes: saison, jour de semaine, type de logement, capacité, etc.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy


class AirbnbListingSimulator:
    """
    Simule un listing Airbnb avec différentes stratégies de prix et de contenu.

    Le taux de conversion dépend de:
    - Le prix (trop cher = moins de bookings)
    - Le contenu (certains styles attirent plus selon le contexte)
    - Le contexte (saison, jour, type de logement)
    """

    def __init__(self, base_price: float = 100, listing_type: str = "apartment"):
        self.base_price = base_price
        self.listing_type = listing_type
        self.price_actions = [-0.10, 0.0, 0.10]  # -10%, base, +10%
        self.content_variants = [
            "cozy_local",      # Style chaleureux et local
            "luxury_premium",  # Style premium et luxe
            "practical_clean"  # Style pratique et fonctionnel
        ]

    def get_context(self, date: datetime) -> np.ndarray:
        """
        Extrait les features contextuelles pour une date donnée.

        Features:
        - Saison (one-hot encoded: hiver, printemps, été, automne)
        - Weekend (0 ou 1)
        - Jours avant réservation (simulé)
        - Type de logement (one-hot encoded)
        """
        # Saison
        month = date.month
        season = np.zeros(4)
        if month in [12, 1, 2]:
            season[0] = 1  # Hiver
        elif month in [3, 4, 5]:
            season[1] = 1  # Printemps
        elif month in [6, 7, 8]:
            season[2] = 1  # Été
        else:
            season[3] = 1  # Automne

        # Weekend
        is_weekend = 1 if date.weekday() >= 5 else 0

        # Jours avant réservation (simulé entre 1 et 30)
        days_before = np.random.randint(1, 31) / 30.0

        # Type de logement
        listing_types = ["apartment", "house", "studio"]
        listing_type_encoded = np.zeros(3)
        if self.listing_type in listing_types:
            listing_type_encoded[listing_types.index(self.listing_type)] = 1

        context = np.concatenate([
            season,
            [is_weekend, days_before],
            listing_type_encoded
        ])

        return context

    def simulate_booking_probability(
        self,
        context: np.ndarray,
        price_modifier: float,
        content_variant: str
    ) -> float:
        """
        Simule la probabilité de réservation en fonction du contexte et des actions.

        Logique:
        - Prix bas en haute saison = bon
        - Prix élevé en basse saison = mauvais
        - Contenu "cozy" marche mieux en hiver
        - Contenu "luxury" marche mieux en été et weekends
        - Contenu "practical" est neutre
        """
        base_prob = 0.15  # Probabilité de base

        # Contexte saisonnier
        is_winter = context[0] == 1
        is_summer = context[2] == 1
        is_weekend = context[4] == 1

        # Impact du prix
        price_impact = -price_modifier * 0.8  # Prix élevé = moins de bookings

        if is_summer or is_weekend:
            # Haute saison: tolérance au prix plus élevée
            price_impact *= 0.5

        # Impact du contenu selon le contexte
        content_impact = 0.0

        if content_variant == "cozy_local":
            if is_winter:
                content_impact = 0.08
            else:
                content_impact = 0.03

        elif content_variant == "luxury_premium":
            if is_summer or is_weekend:
                content_impact = 0.10
            else:
                content_impact = 0.02

        elif content_variant == "practical_clean":
            content_impact = 0.04  # Performance stable

        # Probabilité finale
        prob = base_prob + price_impact + content_impact

        # Ajout de bruit pour simuler la variabilité
        prob += np.random.normal(0, 0.02)

        return np.clip(prob, 0.0, 1.0)

    def get_reward(
        self,
        context: np.ndarray,
        price_modifier: float,
        content_variant: str
    ) -> Tuple[float, bool]:
        """
        Simule une interaction: retourne (revenu, was_booked).

        Le revenu est le prix si la réservation est faite, 0 sinon.
        """
        booking_prob = self.simulate_booking_probability(
            context, price_modifier, content_variant
        )

        was_booked = np.random.random() < booking_prob
        actual_price = self.base_price * (1 + price_modifier)

        reward = actual_price if was_booked else 0.0

        return reward, was_booked


class AirbnbBanditOptimizer:
    """
    Optimiseur de prix et contenu utilisant MABWiser avec LinUCB.
    """

    def __init__(self, simulator: AirbnbListingSimulator):
        self.simulator = simulator

        # Créer les actions combinées: (price_modifier, content_variant)
        self.actions = []
        for price_mod in simulator.price_actions:
            for content in simulator.content_variants:
                self.actions.append((price_mod, content))

        # MABWiser avec LinUCB (contextual bandit linéaire)
        self.mab = MAB(
            arms=list(range(len(self.actions))),
            learning_policy=LearningPolicy.LinUCB(alpha=1.0),
        )

        self.history = []

    def get_action_name(self, action_idx: int) -> str:
        """Retourne une description lisible de l'action."""
        price_mod, content = self.actions[action_idx]
        price_str = f"{price_mod:+.0%}"
        return f"Prix {price_str}, {content}"

    def train_and_evaluate(
        self,
        start_date: datetime,
        n_days: int,
        interactions_per_day: int = 10
    ) -> pd.DataFrame:
        """
        Simule l'apprentissage en ligne sur plusieurs jours.

        Returns:
            DataFrame avec l'historique des interactions
        """
        contexts = []
        decisions = []
        rewards = []

        current_date = start_date
        is_fitted = False
        warmup_size = 10  # Nombre d'interactions pour le warm-up initial

        for day in range(n_days):
            for interaction in range(interactions_per_day):
                # Obtenir le contexte
                context = self.simulator.get_context(current_date)

                # Le bandit choisit une action
                if is_fitted:
                    # Après le warm-up, utilise le modèle
                    action_idx = self.mab.predict(contexts=[context])
                else:
                    # Exploration initiale aléatoire pour le warm-up
                    action_idx = np.random.randint(0, len(self.actions))

                price_mod, content = self.actions[action_idx]

                # Simuler le résultat
                reward, was_booked = self.simulator.get_reward(
                    context, price_mod, content
                )

                # Enregistrer pour l'apprentissage
                contexts.append(context)
                decisions.append(action_idx)
                rewards.append(reward)

                # Enregistrer l'historique
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

                # Warm-up: fit après avoir collecté assez de données
                if not is_fitted and len(contexts) >= warmup_size:
                    self.mab.fit(
                        decisions=decisions,
                        rewards=rewards,
                        contexts=contexts
                    )
                    is_fitted = True

            # Après le warm-up, apprendre des interactions de la journée
            if is_fitted:
                self.mab.fit(
                    decisions=decisions,
                    rewards=rewards,
                    contexts=contexts
                )

            current_date += timedelta(days=1)

        return pd.DataFrame(self.history)

    def plot_results(self, df: pd.DataFrame):
        """Visualise les résultats de l'optimisation."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Revenu cumulé
        ax = axes[0, 0]
        cumulative_reward = df.groupby('day')['reward'].sum().cumsum()
        ax.plot(cumulative_reward.index, cumulative_reward.values, linewidth=2)
        ax.set_xlabel('Jour')
        ax.set_ylabel('Revenu cumulé (€)')
        ax.set_title('Revenu cumulé au fil du temps')
        ax.grid(True, alpha=0.3)

        # 2. Taux de conversion par jour
        ax = axes[0, 1]
        daily_conversion = df.groupby('day')['was_booked'].mean()
        ax.plot(daily_conversion.index, daily_conversion.values, linewidth=2, color='green')
        ax.set_xlabel('Jour')
        ax.set_ylabel('Taux de conversion')
        ax.set_title('Taux de conversion par jour')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 3. Distribution des prix choisis
        ax = axes[1, 0]
        price_distribution = df['price_modifier'].value_counts().sort_index()
        ax.bar(
            [f"{x:+.0%}" for x in price_distribution.index],
            price_distribution.values,
            color='skyblue'
        )
        ax.set_xlabel('Modificateur de prix')
        ax.set_ylabel('Nombre de fois choisi')
        ax.set_title('Distribution des stratégies de prix')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Distribution des contenus choisis
        ax = axes[1, 1]
        content_distribution = df['content_variant'].value_counts()
        ax.bar(
            content_distribution.index,
            content_distribution.values,
            color='coral'
        )
        ax.set_xlabel('Variante de contenu')
        ax.set_ylabel('Nombre de fois choisie')
        ax.set_title('Distribution des variantes de contenu')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)

        plt.tight_layout()
        plt.savefig('airbnb_bandit_results.png', dpi=150, bbox_inches='tight')
        print("📊 Graphiques sauvegardés dans 'airbnb_bandit_results.png'")

        return fig


def compare_with_baseline(
    simulator: AirbnbListingSimulator,
    start_date: datetime,
    n_days: int,
    interactions_per_day: int = 10
) -> Dict[str, float]:
    """
    Compare le bandit contextuel avec une stratégie baseline (prix fixe, contenu fixe).
    """
    print("🎯 Comparaison: Bandit Contextuel vs Baseline\n")

    # Stratégie 1: Bandit contextuel
    print("⚙️  Training bandit contextuel...")
    bandit = AirbnbBanditOptimizer(simulator)
    df_bandit = bandit.train_and_evaluate(start_date, n_days, interactions_per_day)
    bandit_revenue = df_bandit['reward'].sum()
    bandit_conversion = df_bandit['was_booked'].mean()

    # Stratégie 2: Baseline - prix de base, contenu "practical_clean"
    print("⚙️  Testing baseline (prix base, contenu neutre)...")
    baseline_revenues = []
    baseline_conversions = []

    current_date = start_date
    for day in range(n_days):
        for _ in range(interactions_per_day):
            context = simulator.get_context(current_date)
            reward, booked = simulator.get_reward(context, 0.0, "practical_clean")
            baseline_revenues.append(reward)
            baseline_conversions.append(booked)
        current_date += timedelta(days=1)

    baseline_revenue = sum(baseline_revenues)
    baseline_conversion = np.mean(baseline_conversions)

    # Résultats
    print("\n" + "="*60)
    print("📊 RÉSULTATS")
    print("="*60)

    print(f"\n🤖 Bandit Contextuel (MABWiser LinUCB):")
    print(f"   Revenu total: {bandit_revenue:,.2f} €")
    print(f"   Taux conversion: {bandit_conversion:.2%}")

    print(f"\n📌 Baseline (prix fixe, contenu neutre):")
    print(f"   Revenu total: {baseline_revenue:,.2f} €")
    print(f"   Taux conversion: {baseline_conversion:.2%}")

    improvement = (bandit_revenue - baseline_revenue) / baseline_revenue * 100
    print(f"\n🚀 Amélioration du revenu: {improvement:+.1f}%")

    print("\n" + "="*60)

    return {
        'bandit_revenue': bandit_revenue,
        'bandit_conversion': bandit_conversion,
        'baseline_revenue': baseline_revenue,
        'baseline_conversion': baseline_conversion,
        'improvement_pct': improvement,
        'df_bandit': df_bandit,
        'bandit_optimizer': bandit
    }


if __name__ == "__main__":
    print("🏠 POC: Airbnb Pricing & Content Optimization avec Contextual Bandits")
    print("=" * 70)
    print()

    # Configuration
    np.random.seed(42)

    # Créer un listing Airbnb simulé
    simulator = AirbnbListingSimulator(
        base_price=150,  # 150€ par nuit
        listing_type="apartment"
    )

    # Période de simulation: 90 jours, ~10 interactions/jour
    start_date = datetime(2024, 1, 1)
    n_days = 90
    interactions_per_day = 10

    print(f"📅 Période: {n_days} jours ({start_date.strftime('%Y-%m-%d')})")
    print(f"🎯 {interactions_per_day} interactions/jour = {n_days * interactions_per_day} total")
    print(f"💰 Prix de base: {simulator.base_price}€/nuit")
    print(f"🏷️  Prix testés: {[f'{simulator.base_price * (1+m):.0f}€' for m in simulator.price_actions]}")
    print(f"📝 Contenus testés: {simulator.content_variants}")
    print()

    # Lancer la comparaison
    results = compare_with_baseline(
        simulator,
        start_date,
        n_days,
        interactions_per_day
    )

    # Visualiser
    print("\n📈 Génération des visualisations...")
    results['bandit_optimizer'].plot_results(results['df_bandit'])

    print("\n✅ POC terminé!")
    print("\nℹ️  Le bandit apprend à:")
    print("   - Baisser les prix en basse saison")
    print("   - Augmenter les prix en haute saison/weekends")
    print("   - Choisir le contenu optimal selon le contexte")

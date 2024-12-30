import tensorflow as tf
import numpy as np
import optuna
from sklearn.model_selection import KFold
from datetime import datetime

# Generate synthetic dataset
np.random.seed(42)
tf.random.set_seed(42)
W_true, b_true = 0.5, 1.4
X = np.linspace(0, 100, num=100)
y = np.random.normal(loc=W_true * X + b_true, scale=2.0, size=len(X))
now = datetime.now()
logdir = "logs/regression"
writer = tf.summary.create_file_writer(logdir)

def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 0.05, 0.075)
    epochs = trial.suggest_int("epochs", 350, 450, step=10)
    k_folds = trial.suggest_int("k_folds", 3, 7)  # Tune number of folds

    # Cross-validation setup
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_losses = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        # Split into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Define model parameters
        weight = tf.Variable(tf.random.normal([1]), name="weight", dtype=tf.float32)
        bias = tf.Variable(tf.random.normal([1]), name="bias", dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # Training loop with early pruning
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                y_pred = weight * X_train + bias
                train_loss = tf.reduce_mean((y_pred - y_train) ** 2)
            gradients = tape.gradient(train_loss, [weight, bias])
            optimizer.apply_gradients(zip(gradients, [weight, bias]))

            # Evaluate on validation set
            y_val_pred = weight * X_val + bias
            val_loss = tf.reduce_mean((y_val_pred - y_val) ** 2).numpy()

            # Report to Optuna for pruning
            #if epoch % 10 == 0:  # Reduce frequency of reporting (optional)
            #    trial.report(val_loss, step=epoch)
            #   if trial.should_prune():
            #       raise optuna.TrialPruned()

        fold_losses.append(val_loss)

    # Return the average validation loss across folds
    return np.mean(fold_losses)


# Create an Optuna study with k_folds optimization
pruner = optuna.pruners.HyperbandPruner()
study = optuna.create_study(direction="minimize", pruner=pruner, sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=5, show_progress_bar=True, n_jobs=1)

# Print best hyperparameters
print("Best hyperparameters:")
print(study.best_params)

# Train the final model with best hyperparameters
best_learning_rate = study.best_params["learning_rate"]
best_epochs = study.best_params["epochs"]
best_k_folds = study.best_params["k_folds"]

kf = KFold(n_splits=best_k_folds, shuffle=True, random_state=42)
final_losses = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    weight = tf.Variable(tf.random.normal([1]), name="weight", dtype=tf.float32)
    bias = tf.Variable(tf.random.normal([1]), name="bias", dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=best_learning_rate)

    for epoch in range(best_epochs):
        with tf.GradientTape() as tape:
            y_pred = weight * X_train + bias
            train_loss = tf.reduce_mean((y_pred - y_train) ** 2)
        gradients = tape.gradient(train_loss, [weight, bias])
        optimizer.apply_gradients(zip(gradients, [weight, bias]))

        with writer.as_default():
            # Include the fold index in the step parameter
            step = epoch + (fold * best_epochs)
            tf.summary.scalar('MSE Loss', train_loss, step=step)
            tf.summary.histogram('Model Weight', weight, step=step)
            tf.summary.histogram('Model Bias', bias, step=step)

    # Evaluate final model on validation set
    y_val_pred = weight * X_val + bias
    val_loss = tf.reduce_mean((y_val_pred - y_val) ** 2).numpy()
    final_losses.append(val_loss)

# Print final results
print(f"Final Validation Loss (avg across folds): {np.mean(final_losses)}")
print(f"Final Weight: {weight.numpy()[0]}, Final Bias: {bias.numpy()[0]}")
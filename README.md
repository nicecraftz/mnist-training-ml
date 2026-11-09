# MNIST Training ML

Progetto di studio sul Machine Learning: una rete neurale scritta da zero con NumPy, senza framework come PyTorch o TensorFlow, che riconosce le cifre scritte a mano del dataset [MNIST](https://en.wikipedia.org/wiki/MNIST_database).

Lo scopo è capire cosa succede dentro una rete neurale, non ottenere il miglior risultato possibile.

## Cosa è stato scritto a mano e cosa con l'AI

- **Scritto a mano**: tutta la parte matematica della rete in [network.py](network.py): inizializzazione dei pesi, forward pass, attivazioni, softmax, backpropagation e aggiornamento dei parametri con la discesa del gradiente.
- **Scritto con l'AI**: solo il boilerplate, cioè il caricamento del dataset in [dataset.py](dataset.py) e il ciclo di training in [main.py](main.py).

## La rete

Una rete fully connected con questa struttura:

```
Input (784) → Dense (16) + ReLU → Dense (16) + ReLU → Dense (10) + Softmax
```

- **Input**: immagine 28×28 appiattita in un vettore di 784 pixel, normalizzati in [0, 1]
- **Inizializzazione**: He initialization (varianza dei pesi = 2 / numero di input del layer)
- **Loss**: cross-entropy (il gradiente dell'output è `A3 - Y`)
- **Ottimizzazione**: mini-batch gradient descent (batch da 100, learning rate 0.1, 50 epoche)

## Struttura

| File | Contenuto |
|------|-----------|
| [network.py](network.py) | La rete: forward, backward e aggiornamento dei pesi |
| [dataset.py](dataset.py) | Lettura dei file IDX di MNIST e codifica one-hot delle etichette |
| [main.py](main.py) | Ciclo di training e calcolo dell'accuratezza su train e test |
| `dataset/mnist/` | I file del dataset MNIST |

## Come eseguirlo

Il progetto usa [uv](https://docs.astral.sh/uv/) e richiede Python 3.12 o superiore.

```bash
uv sync
uv run main.py
```

A ogni epoca viene stampata l'accuratezza su training set e test set.

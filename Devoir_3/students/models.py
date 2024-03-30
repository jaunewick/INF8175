import numpy as np
import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset

class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        # Calculer le produit scalaire entre les poids et les données
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        # Retourner 1 si le score est supérieur ou égal à 0, sinon -1
        score = nn.as_scalar(self.run(x))
        return 1 if score >= 0 else -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        is_converged = False
        # Entraîner le modèle jusqu'à ce qu'il converge
        while not is_converged:
            is_converged = True
            for x, y in dataset.iterate_once(1):
                y_scalar = nn.as_scalar(y)
                # Si la prédiction est incorrecte, mettre à jour les poids
                if self.get_prediction(x) != y_scalar:
                    self.w.update(x, y_scalar)
                    is_converged = False

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        self.layers = []

        # Hyperparamètres du modèle :
        # Dimensions des couches cachées
        self.layer_sizes = [400, 400]
        # Taux d'apprentissage
        self.learning_rate = 0.08
        # Nombre de couches cachées
        self.num_hidden_layers = 2

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        y_pred = x
        for weight, bias, weight_out, bias_out in self.layers:
            # Appliquer une transformation linéaire
            linear = nn.Linear(y_pred, weight)
            # Ajouter le biais
            biased = nn.AddBias(linear, bias)
            # Appliquer la fonction d'activation ReLU
            activated = nn.ReLU(biased)
            # Appliquer une transformation linéaire
            linear_out = nn.Linear(activated, weight_out)
            # Ajouter le biais
            y_pred = nn.AddBias(linear_out, bias_out)
        return y_pred

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        # Seuil de perte
        threshold = 0.02
        # Taille du batch pour l'entraînement
        self.batch_size = int(0.10 * len(dataset.x))

        # Ajuster la taille du batch pour qu'il soit un multiple de la taille du dataset
        while len(dataset.x) % self.batch_size :
            self.batch_size += 1
        
        # Initialiser les paramètres du modèle
        self.layers = [
            [
                nn.Parameter(1, self.layer_sizes[i]),
                nn.Parameter(1, self.layer_sizes[i]),
                nn.Parameter(self.layer_sizes[i], 1),
                nn.Parameter(1, 1)
            ]
            for i in range(self.num_hidden_layers)
        ]
        
        # Entraîner le modèle
        while True:
            loss_values = []
            for x_batch, y_batch in dataset.iterate_once(self.batch_size) :
                loss = self.get_loss(x_batch, y_batch)
                loss_values.append(nn.as_scalar(loss))
                
                # Récupérer tous les paramètres du modèle
                params = [param for layer in self.layers for param in layer]
                
                # Calculer les gradients de la loss par rapport à chaque paramètre
                gradients = nn.gradients(loss, params)
                
                # Mettre à jour les paramètres du modèle
                for param, gradient in zip(params, gradients) :
                    param.update(gradient, -self.learning_rate)
            
            # Perte inférieure ou égale à 0.02
            if np.mean(loss_values) <= threshold :
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        # Initialiser les paramètres du modèle
        self.layers = [
            [
                nn.Parameter(784, 256),
                nn.Parameter(1, 256),
                nn.Parameter(256, 784),
                nn.Parameter(1,784)
            ],
            [
                nn.Parameter(784, 128),
                nn.Parameter(1, 128),
                nn.Parameter(128, 10),
                nn.Parameter(1,10)
            ]
        ]
        # Taux d'apprentissage
        self.learning_rate = 0.14

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

        y_pred = x
        for weight, bias, weight_out, bias_out in self.layers:
            # Appliquer une transformation linéaire
            linear = nn.Linear(y_pred, weight)
            # Ajouter le biais
            biased = nn.AddBias(linear, bias)
            # Appliquer la fonction d'activation ReLU
            activated = nn.ReLU(biased)
            # Appliquer une transformation linéaire
            linear_out = nn.Linear(activated, weight_out)
            # Ajouter le biais
            y_pred = nn.AddBias(linear_out, bias_out)
        return y_pred


    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        # Seuil de précision
        threshold = 0.97
        # Taille du batch pour l'entraînement
        self.batch_size = int(0.001 * len(dataset.x))

        # Ajuster la taille du batch pour qu'il soit un multiple de la taille du dataset
        while len(dataset.x) % self.batch_size :
            self.batch_size += 1

        # Entraîner le modèle
        while True:
            loss_values = []
            for x_batch, y_batch in dataset.iterate_once(self.batch_size) :
                loss = self.get_loss(x_batch, y_batch)
                loss_values.append(nn.as_scalar(loss))
                
                # Récupérer tous les paramètres du modèle
                params = [param for layer in self.layers for param in layer]
                
                # Calculer les gradients de la loss par rapport à chaque paramètre
                gradients = nn.gradients(loss, params)
                
                # Mettre à jour les paramètres du modèle
                for param, gradient in zip(params, gradients) :
                    param.update(gradient, -self.learning_rate)
            
            # Précision d'au moins 97% sur l'ensemble de test
            if dataset.get_validation_accuracy() > threshold :
                break
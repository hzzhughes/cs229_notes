import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n=x.shape
        phi=np.mean(y)
        mu0=np.mean(x[y==0],axis=0)
        mu1=np.mean(x[y==1],axis=0)
        mu_yi=(1-y).reshape(m,1)@mu0.reshape(1,n)+y.reshape(m,1)@mu1.reshape(1,n)
        Sigma=(x-mu_yi).T@(x-mu_yi)/m
        theta=np.linalg.inv(Sigma)@(mu1-mu0)
        theta_0=(mu0@np.linalg.inv(Sigma)@mu0-mu1@np.linalg.inv(Sigma)@mu1)/2+np.log(phi/(1-phi))
        self.theta=np.append([theta_0],theta)
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE

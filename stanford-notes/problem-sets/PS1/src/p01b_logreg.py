import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf=LogisticRegression()
    clf.fit(x_train, y_train)
    print(f'theta={clf.theta}')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def g(z):
            return 1/(1+np.exp(-z))
        m,n=x.shape
        theta=np.zeros(n)
        delta_theta=np.ones(n)
        while np.linalg.norm(delta_theta,1)>=1e-5:
            H=sum(g(x_i@theta)*(1-g(x_i@theta))*x_i.reshape(n,1)@x_i.reshape(1,n) for x_i in x)/m
            grad=-(y-g(x@theta))@x/m
            delta_theta=-np.linalg.inv(H)@grad
            theta+=delta_theta
        self.theta=theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        def g(z):
            return 1/(1+np.exp(-z))
        return g(x@self.theta)
        # *** END CODE HERE ***

main('stanford-notes/problem-sets/PS1/data/ds1_train.csv','','')
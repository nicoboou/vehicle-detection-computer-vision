# Standard library imports
from typing import List
import pickle
import time

# Third party imports
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


class VehicleClassifier:
    """
    Class to build a vehicle classifier based on machine learning algorithm.

    Algorithm:
    ----------

    """

    def __init__(
        self,
        vehicle_features: List[np.ndarray],
        non_vehicle_features: List[np.ndarray],
    ) -> None:
        """
        Description:
        ------------
        Initialize the vehicle classifier.

        Parameters:
        -----------
        vehicle_features (list): list of vehicle features
        non_vehicle_features (list): list of non-vehicle features

        Returns:
        --------
        None

        """
        self.vehicles_features = vehicle_features
        self.non_vehicles_features = non_vehicle_features

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.clf = None

    def build_train_test_sets(self, test_size: float = 0.2, random_state: int = 42):

        """
        Description:
        ------------
        Build the train and test sets.

        Parameters:
        -----------
        test_size (float): percentage of the test set.
        random_state (int): random state

        Returns:
        --------
        X_train (np.ndarray): train set
        X_test (np.ndarray): test set
        y_train (np.ndarray): train labels
        y_test (np.ndarray): test labels
        scaler (sklearn.preprocessing.StandardScaler): scaler used to scale the features
        """
        X = np.vstack((self.vehicles_features, self.non_vehicles_features)).astype(
            float
        )

        # Fit scaler
        self.scaler = StandardScaler().fit(X)

        # Apply scaler
        X_scaled = self.scaler.transform(X)

        # Define the labels vector
        y = np.hstack(
            (
                np.ones(len(self.vehicles_features)),
                np.zeros(len(self.non_vehicles_features)),
            )
        )

        # Split up data into randomized training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return self.X_train, self.X_test, self.y_train, self.y_test, self.scaler

    def save_datasets(
        self,
        path: str,
        file_name: str,
    ):
        """
        Save train and test data for easy access
        """

        data_pickle_file = path + file_name + ".pkl"
        print(f"Saving data to pickle file {data_pickle_file}...")

        try:
            with open(data_pickle_file, "wb") as pfile:
                pickle.dump(
                    {
                        "X_train": self.X_train,
                        "X_test": self.X_test,
                        "y_train": self.y_train,
                        "y_test": self.y_test,
                        "scaler": self.scaler,
                    },
                    pfile,
                    pickle.HIGHEST_PROTOCOL,
                )

        except Exception as exc:
            print(f"Unable to save data to {data_pickle_file} : {exc}")
            raise

        print("Data cached in pickle file.")

        return None

    def train(self, clf: str = "LinearSVC", path_to_save: str = None):
        """
        Description:
        ------------
        Train the classifier among the possible ones:
        - Linear SVC
        - Logistic Regression
        - Random Forest Classifier

        Parameters:
        -----------
        clf (sklearn): classifier

        Returns:
        --------
        clf (sklearn): trained classifier

        """
        print("=" * 30 + "TRAINING" + "=" * 30)
        print(f"Training {clf}...")
        print("Feature vector length:", len(self.X_train[0]))

        if clf == "LinearSVC":
            # Use a linear SVC
            # Note: I select the algorithm to either solve the dual or primal optimization problem
            # => prefer dual=False when n_samples > n_features
            self.clf = LinearSVC(dual=False)

            # Check the training time for the SVC
            start = time.time()

            # Fit the model
            self.clf.fit(self.X_train, self.y_train)

            print(f"{round(time.time() - start, 2)} seconds to train {clf}...")
            print("=" * 30 + "METRICS" + "=" * 30)
            # Check the score of the SVC
            print(
                f"Training Accuracy of {clf} = ",
                round(self.clf.score(self.X_train, self.y_train), 4),
            )
            print(
                f"Test Accuracy of {clf} = ",
                round(self.clf.score(self.X_test, self.y_test), 4),
            )

        elif clf == "RandomForestClassifier":
            # Use a random forest
            self.clf = RandomForestClassifier()

            # Check the training time for the Random Forest
            start = time.time()

            # Fit the model
            self.clf.fit(self.X_train, self.y_train)

            print(f"{round(time.time() - start, 2)} seconds to train {clf}...")
            print("=" * 30 + "METRICS" + "=" * 30)
            # Check the score of the Random Forest
            print(
                f"Training Accuracy of {clf} = ",
                self.clf.score(self.X_train, self.y_train),
            )
            print(
                f"Test Accuracy of {clf} = {round(self.clf.score(self.X_test, self.y_test), 4)}"
            )

        elif clf == "SGDClassifier":
            # Use a SGDClassifier
            self.clf = SGDClassifier(loss="hinge")

            # Check the training time for the SGDClassifier
            start = time.time()

            # Fit the model
            self.clf.fit(self.X_train, self.y_train)

            print(f"{round(time.time() - start, 2)} seconds to train {clf}...")
            print("=" * 30 + "METRICS" + "=" * 30)
            # Check the score of the SGDClassifier
            print(
                f"Training Accuracy of {clf} = ",
                self.clf.score(self.X_train, self.y_train),
            )
            print(
                f"Test Accuracy of {clf} = {round(self.clf.score(self.X_test, self.y_test), 4)}"
            )

        else:
            print(f"Classifier {clf} not implemented yet.")
            return None

        print("=" * 30 + "Saving model..." + "=" * 30)
        # Save the model
        model_pickle_file = path_to_save + clf + ".pkl"

        with open(model_pickle_file, "wb") as pfile:
            pickle.dump(
                {"model": self.clf, "scaler": self.scaler},
                pfile,
                pickle.HIGHEST_PROTOCOL,
            )

        print(f"Model saved in {model_pickle_file}")

        return self.clf

import numpy as np
from abc import ABC, abstractmethod
from src.DataLoader import DataConfig
from src.base_classes import BaseUtilityModel


class EventReconstructorBase(BaseUtilityModel, ABC):
    def __init__(
        self,
        config: DataConfig,
        assignment_name,
        full_reco_name,
        neutrino_name=None,
        perform_regression=True,
        use_nu_flows=True,
    ):
        BaseUtilityModel.__init__(
            self,
            config=config,
            assignment_name=assignment_name,
            full_reco_name=full_reco_name,
            neutrino_name=neutrino_name,
        )
        self.max_jets = config.max_jets
        self.NUM_LEPTONS = config.NUM_LEPTONS
        if perform_regression and not config.has_neutrino_truth:
            print(
                "WARNING: perform_regression is set to True, but config.has_neutrino_truth is False. Setting perform_regression to False."
            )
            perform_regression = False
        if use_nu_flows and not config.has_nu_flows_neutrino_regression:
            print(
                "WARNING: use_nu_flows is set to True, but config.use_nu_flows is False. Setting use_nu_flows to False."
            )
            use_nu_flows = False
        if perform_regression and use_nu_flows:
            print(
                "WARNING: perform_regression is set to True, but use_nu_flows, is also True. Setting use_nu_flows False to make us of neutrino regression implementation."
            )

        self.perform_regression = perform_regression
        self.use_nu_flows = use_nu_flows

    def predict_indices(self, data_dict):
        pass

    def reconstruct_neutrinos(self, data_dict: dict[str : np.ndarray]):
        if self.perform_regression:
            raise NotImplementedError(
                "This method should be implemented in subclasses that perform regression."
            )
        if self.use_nu_flows:
            if "nu_flows_neutrino_regression" in data_dict:
                return data_dict["nu_flows_neutrino_regression"]
            print(
                "WARNING: use_nu_flows is True but 'nu_flows_neutrino_regression' not found in data_dict. Falling back to 'neutrino_truth'."
            )
        if "regression" in data_dict:
            return data_dict["regression"]
        print(f"data_dict keys: {list(data_dict.keys())}")
        raise ValueError(
            "No regression targets found in data_dict for neutrino reconstruction."
        )

    def evaluate_accuracy(self, data_dict, true_labels, per_event=False):
        """
        Evaluates the model's performance on the provided data and true indices.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
            true_labels (np.ndarray): The true labels (one-hot) to compare against the model's predictions.
        Returns:
            float | np.ndarray: The accuracy of the model's predictions. If per_event is True,
            returns an array of booleans indicating correctness per event; otherwise, returns overall accuracy.
        """
        predictions = self.predict_indices(data_dict)
        return self.compute_accuracy(predictions, true_labels, per_event)

    def evaluate_regression(self, data_dict, true_values):
        """
        Evaluates the regression performance of the model on the provided data and true values.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
            true_values (np.ndarray): The true regression target values to compare against the model's predictions.
        Returns:
            float: The mean squared error of the model's regression predictions.
        """
        predicted_values = self.reconstruct_neutrinos(data_dict)
        return self.compute_regression_mse(predicted_values, true_values)

    def compute_accuracy(self, pred_values, true_values, per_event=False):
        predicted_indices = np.argmax(pred_values, axis=-2)
        true_indices = np.argmax(true_values, axis=-2)
        if per_event:
            correct_predictions = np.all(predicted_indices == true_indices, axis=-1)
            return correct_predictions
        else:
            correct_predictions = np.all(predicted_indices == true_indices, axis=-1)
            accuracy = np.mean(correct_predictions)
            return accuracy

    def compute_regression_mse(self, pred_values, true_values):
        relative_errors = (pred_values - true_values) / np.where(
            true_values != 0, true_values, 1
        )
        mse = np.mean(np.square(relative_errors))
        return mse

    def evaluate(self, data_dict):
        results = {}
        if "assignment_truth" in data_dict:
            accuracy = self.evaluate_accuracy(
                data_dict, data_dict["assignment_truth"], per_event=False
            )
            results["accuracy"] = accuracy
        if self.perform_regression and "regression" in data_dict:
            mse = self.evaluate_regression(data_dict, data_dict["regression"])
            results["regression_mse"] = mse
        return results

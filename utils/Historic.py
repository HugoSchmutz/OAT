import numpy as np
import pandas as pd

class Historic:
    """A class to save running data from the experiments"""
    def __init__(self, name, T_stream, budget, nb_repet):
        self.T_stream = T_stream
        self.name = name
        self.budget = budget
        self.nb_repet = nb_repet
        
        self.scores = []
        self.selection = []
        self.permutation = []
        self.probabilities = []
        self.time = []
        self.nb_selected = []
        self.run_id = []

    def update(self, scores, selection, permutation, probabilities, run_id):
        self.scores.append(scores)
        self.selection.append(selection)
        self.permutation.append(permutation)
        self.probabilities.append(probabilities)
        self.time.append(np.arange(1,self.T_stream+1))
        self.nb_selected.append(selection.cumsum())
        self.run_id.append(np.ones(self.T_stream)*run_id)
    
    def _get_unique_filename(self, direction):
        filename = f"{direction}{self.name}_{self.nb_repet}_{self.T_stream}_{self.budget}.csv"
        return filename

    def save_to_csv(self, direction="results/"):
        filename = self._get_unique_filename(direction)
        df = self.to_dataframe()
        df.to_csv(filename)
        print(f"Data saved to: {filename}")
        
    def to_dataframe(self):
        df = pd.DataFrame({
            "time": np.concatenate(self.time),
            "scores": np.concatenate(self.scores),
            "selection": np.concatenate(self.selection),
            "nb_selected": np.concatenate(self.nb_selected),
            "element": np.concatenate(self.permutation),
            "probabilities": np.concatenate(self.probabilities),
            "series": self.name,
            "run_id": np.concatenate(self.run_id),
        })

        return df

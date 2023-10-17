from ray.tune.logger import LoggerCallback
from ray.tune import Callback
from typing import TYPE_CHECKING
from collections import defaultdict
import pprint
import json
import os

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial  

class LocalFileCallback(LoggerCallback):
    def __init__(self, dir):
        self.dir = dir
        try:
            os.mkdir(dir)
        except:
            print("folder exists")

    def log_trial_start(self, trial: "Trial"):
        """Handle logging when a trial starts.

        Args:
            trial: Trial object.
        """
        try:
            os.mkdir(f"{self.dir}/{trial.trial_id}")
        except:
            print("folder exists")

    def on_trial_result(self, iteration, trials, trial, result, **info):
        with open(f"{self.dir}/{trial.trial_id}/result.json", "a") as outfile:
            json.dump(result, outfile, indent=4,  
                        separators=(',',': '))

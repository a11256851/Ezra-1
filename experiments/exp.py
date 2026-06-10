import os
import pickle
import shutil
from pathlib import Path
from typing import List

from src.agent import (
    LangChainAgentManager,
    Parser,
    Plain_text_chunker,
    RetrievalManager,
    School_rules_chunker,
)


class Experiment:
    def __init__(self):
        self.experiments_dir = Path(__file__).parent
        self.data_dir = self.experiments_dir / "data"
        self.test_set = self.data_dir / "test1.json"  # default
        self.results_dir = self.experiments_dir / "results"

    def showInfo(self):
        pass

    def eval(self):
        pass

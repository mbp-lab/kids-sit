from abc import ABC, abstractmethod
from typing import List


class FeaturesMixin(ABC):
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        raise NotImplementedError

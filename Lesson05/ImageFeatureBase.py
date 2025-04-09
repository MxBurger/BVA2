from abc import ABC, abstractmethod

class ImageFeatureBase(ABC):

    def __init__(self):
        self.description = ""

    @abstractmethod
    def CalcFeatureVal(self, imgRegion, FG_val):
        pass
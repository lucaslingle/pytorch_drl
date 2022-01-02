import abc


class Preprocessing(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def preprocessed(self, x):
        pass

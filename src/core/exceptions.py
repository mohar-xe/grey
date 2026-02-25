class GreyBaseError(Exception):
    pass

class IngestionError(GreyBaseError):
    pass

class RetrievalError(GreyBaseError):
    pass

class GraphError(GreyBaseError):
    pass

class EmbeddingError(GreyBaseError):
    pass

class ProcessingError(GreyBaseError):
    pass
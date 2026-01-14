from transformers import AutoConfig, AutoModel
from .configuration_tellm import TeLLMSolverConfig 
from .modeling_tellm import TeLLMSolverModel


AutoConfig.register("tellm", TeLLMSolverConfig)
AutoModel.register(TeLLMSolverConfig, TeLLMSolverModel)
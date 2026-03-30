from transformers import AutoConfig, AutoModelForCausalLM

from .configuration import Fast_dLLM_QwenConfig
from .modeling import Fast_dLLM_QwenForCausalLM

AutoConfig.register("Fast_dLLM_Qwen", Fast_dLLM_QwenConfig)
AutoModelForCausalLM.register(Fast_dLLM_QwenConfig, Fast_dLLM_QwenForCausalLM)

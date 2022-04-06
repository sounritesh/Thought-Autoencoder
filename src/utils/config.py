# from transformers import AutoModel, AutoTokenizer
import torch

# BASE_MODEL_PATH = "vinai/bertweet-base"
# BASE_MODEL = AutoModel.from_pretrained(BASE_MODEL_PATH)
# BASE_TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

INPUT_SIZE = 768*2 + 163 + 12
OUTPUT_SIZE = 2

DEVICE = torch.device("cuda")
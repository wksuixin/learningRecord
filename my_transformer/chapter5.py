from transformers import Qwen2Model
from transformers import AutoModel

save_path = "/home/wk/code/model/Qwen2.5-0.5B_copy"
model = Qwen2Model.from_pretrained("/home/wk/code/model/Qwen2.5-0.5B")
print(type(model))

model.save_pretrained(save_path)
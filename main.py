from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')

model.eval()

sentence = 'בשנת 1948 השלים אפרים קישון את [MASK] בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'

output = model(tokenizer.encode(sentence, return_tensors='pt'))
# the [MASK] is the 7th token (including [CLS])
top_2 = torch.topk(output.logits[0, 7, :], 2)[1]
print('\n'.join(tokenizer.convert_ids_to_tokens(top_2))) # should print מחקרו / התמחותו

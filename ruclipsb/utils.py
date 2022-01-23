import torch

def tokenize(tokenizer, texts, max_len=77):
    input_ids = []
    attention_masks = []
    for sent in texts:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            truncation=True,
                            add_special_tokens = True,
                            max_length = max_len,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt', 
                      )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

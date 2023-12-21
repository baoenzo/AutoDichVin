import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
device_en2vi = torch.device("cuda")
model_en2vi.to(device_en2vi)


def translate_en2vi(en_texts: str) -> str:
    input_ids = tokenizer_en2vi(en_texts, padding=True, return_tensors="pt").to(device_en2vi)
    output_ids = model_en2vi.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    return vi_texts
def translate_document():
    with open('text.txt' ,"r",encoding="utf8") as f:
        lines = f.readlines()
    list_fi = translate_en2vi(lines)
    with open('fi.txt', "w+",encoding="utf8") as wf:
        for text in list_fi:
            print(text)
            wf.write(text + "\r")

translate_document()
print("xong")
          
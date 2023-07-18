import pypdfium2 as pdfium
from paddlenlp import Taskflow
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def donut_inf(file, usr_inp):
    image = Image.open(file)

    # prepare decoder inputs
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    question = usr_inp
    prompt = task_prompt.replace("{user_input}", question)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    return processor.token2json(sequence)


def ernie_inf(file, usr_inp):
    doc_prompt = Taskflow("document_intelligence", lang="en", topn=2)
    res1 = doc_prompt({"doc": file, "prompt": [usr_inp]})
    res2 = res1[0].get('result')
    res3 = [d['value'] for d in res2]
    return res3


def pdf_to_im(filepath):
    pdf = pdfium.PdfDocument(filepath)
    n_pages = len(pdf)
    page_indices = [i for i in range(n_pages)]  # all pages
    renderer = pdf.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=300 / 72,  # 300dpi resolution
    )
    for i, image in zip(page_indices, renderer):
        image.save(os.path.dirname(filepath) + '/' + 'out_%0*d.png' % (n_pages, i))

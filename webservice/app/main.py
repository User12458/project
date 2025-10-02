import re
from typing import List, Dict, Tuple
import torch
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import re
import torch
from typing import List, Union, Dict, Any, Tuple

id2label = {
    0: 'O',
    1: 'B-BRAND',
    2: 'I-BRAND',
    3: 'B-PERCENT',
    4: 'I-PERCENT',
    5: 'B-TYPE',
    6: 'I-TYPE',
    7: 'B-VOLUME',
    8: 'I-VOLUME'
}
label2id = {v: k for k, v in id2label.items()}

# подгружаем все модели и токенизаторы
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_deberta = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
tokenizer_rosberta = AutoTokenizer.from_pretrained("ai-forever/ru-en-RoSBERTa", add_prefix_space=True)

model_deberta_42 = AutoModelForTokenClassification.from_pretrained(
    "microsoft/mdeberta-v3-base",
    num_labels=9,
    id2label=id2label,
    label2id=label2id,
).cuda()
model_deberta_42.load_state_dict(torch.load('app/deberta_with_bio.pt'))
model_deberta_42.eval()

model_deberta_56 = AutoModelForTokenClassification.from_pretrained(
    "microsoft/mdeberta-v3-base",
    num_labels=9,
    id2label=id2label,
    label2id=label2id,
).cuda()
model_deberta_56.load_state_dict(torch.load('app/deberta_without_bio.pt'))
model_deberta_56.eval()

model_rosberta = AutoModelForTokenClassification.from_pretrained(
    "ai-forever/ru-en-RoSBERTa",
    num_labels=9,
    id2label=id2label,
    label2id=label2id,
).cuda()
model_rosberta.load_state_dict(torch.load('app/rosberta_without_bio.pth'))
model_rosberta.eval()

# постпроцессим
# смотрим, чтобы не было случаем (B-ENT, B-ENT) или (I-ENT, B-ENT) 
def post_proc(val_predicts):
    val_predicts_new = []
    for tripls in val_predicts:
        row,used = [],[]
        for i,j,k in tripls:
            if k == "O":
                row.append({"start_index": i, "end_index": j, "entity": k})
                continue
            tp = k.split("-")[1]
            if tp in used:
                row.append({"start_index": i, "end_index": j, "entity": k.replace("B-", "I-")})
            else:
                row.append({"start_index": i, "end_index": j, "entity": k.replace("I-", "B-")})
            used.append(tp)
        val_predicts_new.append(row)
    return val_predicts_new

# функция инференса
def ner_word_tuples(
    texts,
    model,
    tokenizer,
    batch_size: int = 16,
    i_to_b_k: int = 0,
):

    single_input = isinstance(texts, str)
    texts_list = [texts] if single_input else list(texts)

    spans_per_text = []
    words_per_text = []
    for t in texts_list:
        spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", t)]
        spans_per_text.append(spans)
        words_per_text.append([w for w, _, _ in spans])

    if not any(words_per_text):
        empty = [{
            'words': [],
            'word_spans': [],
            'labels': [],
            'word_tuples': [],
            'entities': [],
        } for _ in texts_list]
        return empty[0] if single_input else empty

    models = model if isinstance(model, (list, tuple)) else [model]
    tokenizers = tokenizer if isinstance(tokenizer, (list, tuple)) else [tokenizer]

    label_set = set()
    per_model_labels = []
    for m in models:
        id2label = m.config.id2label
        if isinstance(id2label, dict):
            local_labels = [id2label[i] for i in range(len(id2label))]
        else:
            local_labels = list(id2label)
        per_model_labels.append(local_labels)
        label_set.update(local_labels)

    to_add = set()
    for lab in label_set:
        if isinstance(lab, str) and lab.startswith("I-"):
            b_lab = "B-" + lab[2:]
            if b_lab not in label_set:
                to_add.add(b_lab)
    label_set.update(to_add)

    if "O" in label_set:
        global_labels = ["O"] + sorted([l for l in label_set if l != "O"])
    else:
        global_labels = sorted(label_set)
    label2idx = {lab: i for i, lab in enumerate(global_labels)}
    i_to_b_k = max(0, min(int(i_to_b_k), len(models)))
    local2global = []
    for mi, local_labels in enumerate(per_model_labels):
        mapped = []
        for lab in local_labels:
            if mi < i_to_b_k and isinstance(lab, str) and lab.startswith("I-"):
                b_lab = "B-" + lab[2:]
                mapped_idx = label2idx[b_lab]
            else:
                mapped_idx = label2idx[lab]
            mapped.append(mapped_idx)
        local2global.append(torch.tensor(mapped, dtype=torch.long))

    L = len(global_labels)
    O_idx = label2idx.get("O", 0)
    o_vec = torch.zeros(L, dtype=torch.float32)
    o_vec[O_idx] = 1.0

    labels_per_text = [None] * len(texts_list)

    # Батчевый прогон
    for b in range(0, len(texts_list), batch_size):
        batch_words = words_per_text[b:b + batch_size]

        per_model_word_probs = []
        for mi, (m, tok) in enumerate(zip(models, tokenizers)):
            enc = tok(
                batch_words,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            word_ids_list = [enc.word_ids(batch_index=i) for i in range(len(batch_words))]

            m_device = next(m.parameters()).device
            enc_gpu = {k: v.to(m_device) for k, v in enc.items()}

            with torch.inference_mode():
                logits = m(**enc_gpu).logits  # [bsz, seq, local_L]
                probs = torch.nn.functional.softmax(logits, dim=-1)

            # Переносим локальные вероятности в глобальное пространство (с учётом I->B)
            map_idx = local2global[mi].to(probs.device)
            gprobs = torch.zeros(probs.size(0), probs.size(1), L, device=probs.device, dtype=probs.dtype)
            gprobs.scatter_add_(dim=-1, index=map_idx.view(1, 1, -1).expand_as(probs), src=probs)
            gprobs = gprobs.float().cpu()

            # Собираем по словам: берём первый субтокен слова
            model_word_probs = []
            for si, wids in enumerate(word_ids_list):
                arr = [None] * len(batch_words[si])
                gp = gprobs[si]
                for ti, wid in enumerate(wids):
                    if wid is None:
                        continue
                    if 0 <= wid < len(arr) and arr[wid] is None:
                        arr[wid] = gp[ti]
                for wi in range(len(arr)):
                    if arr[wi] is None:
                        arr[wi] = o_vec
                model_word_probs.append(arr)
            per_model_word_probs.append(model_word_probs)

        # Усреднение по моделям и argmax
        for si in range(len(batch_words)):
            seq_len = len(batch_words[si])
            seq_labels = []
            for wi in range(seq_len):
                vecs = [per_model_word_probs[mj][si][wi] for mj in range(len(models))]
                mean_vec = torch.stack(vecs, dim=0).mean(dim=0)
                seq_labels.append(global_labels[int(mean_vec.argmax().item())])
            labels_per_text[b + si] = seq_labels

    results = []
    for ti, (text, spans, labels) in enumerate(zip(texts_list, spans_per_text, labels_per_text)):
        words = [w for w, _, _ in spans]
        word_spans = [(s, e) for _, s, e in spans]
        word_tuples = [(s, e, lab) for (w, (s, e), lab) in zip(words, word_spans, labels)]
        results.append(word_tuples)

    return results[0] if single_input else results

# Функция для того, чтобы ловить несклько запросов и прогонять сразу батч
async def batch_worker():
    while True:
        batch = []
        futures = []
        try:
            req, fut = await request_queue.get()
            batch.append(req)
            futures.append(fut)

            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < 0.005:
                try:
                    req2, fut2 = request_queue.get_nowait()
                    batch.append(req2)
                    futures.append(fut2)
                except asyncio.QueueEmpty:
                    break

            texts = [r.input for r in batch]
            preds = post_proc(
                ner_word_tuples(
                    texts, 
                    [model_deberta_42, model_deberta_56, model_rosberta], 
                    [tokenizer_deberta, tokenizer_deberta, tokenizer_rosberta], 
                    batch_size=128, 
                    i_to_b_k=2
                )
            )
            for fut, p in zip(futures, preds):
                fut.set_result(p)

        except Exception as e:
            for fut in futures:
                fut.set_exception(e)

request_queue = asyncio.Queue()

class InputData(BaseModel):
    input: str

app = FastAPI()
asyncio.create_task(batch_worker())

# ручка
@app.post("/api/predict")
async def get_predictions(data: InputData):
    fut = asyncio.get_event_loop().create_future()
    await request_queue.put((data, fut))
    result = await fut
    return result
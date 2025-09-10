from datasets import load_dataset, Dataset
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType,prepare_model_for_kbit_training
import torch
from transformers import get_scheduler, TrainingArguments, default_data_collator
from datasets import Dataset
from transformers import BitsAndBytesConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
# 載入模型（Unsloth 4bit Mistral）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)

model_name = "mistralai/Mistral-7B-v0.1"

# ✅ 正確順序：先下載 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ✅ 只呼叫一次 from_pretrained
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# ✅ 加入 LoRA 之前，要先用這個做準備（會處理 gradient checkpoint / input embedding 問題）
model = prepare_model_for_kbit_training(model)

# ✅ LoRA 設定
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj","k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ✅ 加入 PEFT 模組
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 資料路徑（請確認存在）
dataset1 = load_dataset("json", data_files=r"C:\Users\USER\toysub\dataset\english_data\toysub_data.json", split="train")
dataset2 = load_dataset("json", data_files=r"C:\Users\USER\toysub\dataset\english_data\translated_web_data.json", split="train")
dataset3 = load_dataset("json", data_files=r"C:\Users\USER\toysub\dataset\english_data\reject_dataset.json", split="train")
dataset4 = load_dataset("json", data_files=r"C:\Users\USER\toysub\dataset\english_data\translated_toys_data.json", split="train")
# 合併與清洗
df1 = pd.DataFrame(dataset1)[["question", "input", "output","TOPIC"]]
df2 = pd.DataFrame(dataset2)[["question", "input", "output","TOPIC"]]
df3 = pd.DataFrame(dataset3)           
df3[["input", "TOPIC"]] = ""                     
df3 = df3[["question", "input", "output","TOPIC"]]
df4 = pd.DataFrame(dataset4)[["question", "input", "output","TOPIC"]]

merged_df = pd.concat([df1, df2,df3,df4], ignore_index=True)

def preprocess(examples):
    def safe_text(val):
        return val if isinstance(val, str) and val.strip() else "[空資料]"

    # 1. 拼接 prompt + output
    prompts = []
    outputs = []
    for q, i, o ,t in zip(examples["question"], examples["input"], examples["output"],examples["TOPIC"]):
        q = safe_text(q)
        i = safe_text(i)
        o = safe_text(o)
        t= safe_text(t)
        # 模板（你可以自行改格式）
        system_prompt = """<|im_start|>system
        You are an expert toy consultant at TOYSUB. Answer all questions in fluent, professional English, regardless of input language.
        Your job is to provide precise, informative, and age-appropriate toy suggestions.
        Use the context below to craft detailed responses.
        Avoid generic phrases, emotional appeals, or closing remarks.
        Only focus on the toy’s features, recommended age, benefits, and how it helps development.
        When possible, answer in bullet points or structured explanations.
        <|im_end|>"""

        if i == "[空資料]" and t == "[空資料]":
            user_prompt = f"<|im_start|>user\nQuestion: {q}<|im_end|>"
        else:
            user_prompt = f"<|im_start|>user\nQuestion: {q}\nContext: {i}<|im_end|>"
        
        assistant_prefix = "<|im_start|>assistant\n"
        prompt_template = f"{system_prompt}\n{user_prompt}\n{assistant_prefix}"

        prompts.append(prompt_template)
        outputs.append(o)

    # 最終 tokenize
    model_inputs = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            outputs,
            truncation=True,
            max_length=512,
            padding="max_length"
        )["input_ids"]

    # 將 prompt 部分的 label 設為 -100
    for i in range(len(labels)):
        prompt_len = len(tokenizer(prompts[i], truncation=True, max_length=512)["input_ids"])
        labels[i] = [-100] * prompt_len + labels[i]
        labels[i] = labels[i][:512]  # truncate labels to match input_ids length

    model_inputs["labels"] = labels
    return model_inputs

dataset = Dataset.from_pandas(merged_df[["question","input", "output","TOPIC"]])
train_dataset = dataset.map(preprocess, batched=True)

def custom_collator(features):
    import torch

    batch = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        values = [f[key] for f in features]
        if any(v is None for v in values):
            print(f"❌ Found None in {key}: {values}")
            raise ValueError(f"{key} 有 None")

        batch[key] = torch.tensor(values, dtype=torch.long)

    return batch

# 訓練設定
training_args = TrainingArguments(
    output_dir="./mistral_lora",
    learning_rate=5e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    weight_decay=0.01, 
    num_train_epochs=15,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    max_grad_norm=1.0,
    report_to="none"
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=custom_collator
)

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Scheduler：線性遞減學習率，並 warmup 前一個 epoch
num_training_steps = len(train_dataloader) * training_args.num_train_epochs
num_warmup_steps = len(train_dataloader)  # 1 epoch warmup

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.train()
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
for epoch in tqdm(range(training_args.num_train_epochs), desc="training"):
    total_loss = 0
    step_count = 0

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if torch.isnan(loss) or loss.item() == 0:
            print("⚠️ Loss is NaN or 0, skip this batch.")
            continue

        loss.to(torch.float32).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        step_count += 1

    avg_loss = total_loss / step_count
    print(f"✅ Epoch {epoch+1} / {training_args.num_train_epochs}, Avg Loss: {avg_loss:.4f}")

# 儲存 HuggingFace 格式模型
model.save_pretrained("./final_623_model")
tokenizer.save_pretrained("./final_623_model")

# 測試生成
messages = [
    {"role": "user", "content": "What toys are recommended for 1 year olds?"}
]
prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant:"

# Tokenization
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

streamer = TextStreamer(tokenizer, skip_prompt=True)

_ = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    streamer=streamer
)

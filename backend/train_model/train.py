# Último Checkpoint: 3% - 153

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import torch
import os
import evaluate

os.environ["HF_TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_TF"] = "1"

model_name = "google/mt5-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = [[l if l != -100 else tokenizer.pad_token_id for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return result

# =========================
# Definir ruta absoluta para guardar el modelo
# =========================
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "aulivoxmodel"))
os.makedirs(output_dir, exist_ok=True)

# =========================
# Cargar dataset
# =========================

dataset = load_dataset("json", data_files={
    "train": "train_model/dataset_train.jsonl"
})
dataset = dataset["train"].train_test_split(test_size=0.3)
# =========================
# Tokenización
# =========================

def preprocess(examples):
    inputs = examples["input"]
    targets = examples["target"]

    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=200,
        truncation=True,
        padding="max_length"
    )

    # Reemplazar padding por -100 para que no afecte la loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# =========================
# Data Collator
# =========================

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model
)


# =========================
# Configuración de entrenamiento
# =========================

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=10,
    logging_steps=10,
    save_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=5,
    eval_steps=50,
    fp16=torch.cuda.is_available()  # 🔥 solo si hay GPU
)

trainer = Trainer(
    model=model,
    args=training_args,    
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Reanudar desde el último checkpoint si existe
import glob
checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), reverse=True)
if checkpoints:
    last_checkpoint = checkpoints[0]
    print(f"Reanudando entrenamiento desde: {last_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    train_result = trainer.train()


# Carpeta para logs
logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training_logs"))
os.makedirs(logs_dir, exist_ok=True)


# Fecha y hora para identificar cada entrenamiento
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Calcular número de intento
import glob
existing_logs = glob.glob(os.path.join(logs_dir, "train_metrics_I-*.json"))
attempt_number = len(existing_logs) + 1
identifier = f"I-{attempt_number}"

# Guardar métricas de entrenamiento
metrics = train_result.metrics
metrics_path = os.path.join(logs_dir, f"train_metrics_{identifier}_{timestamp}_epoch{int(metrics.get('epoch', 0))}.json")
with open(metrics_path, "w", encoding="utf8") as f:
    import json
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"\n=== Métricas de entrenamiento guardadas en {metrics_path} ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Guardar métricas de evaluación
eval_metrics = trainer.evaluate()
eval_metrics_path = os.path.join(logs_dir, f"eval_metrics_{identifier}_{timestamp}_epoch{int(eval_metrics.get('epoch', 0))}.json")
with open(eval_metrics_path, "w", encoding="utf8") as f:
    json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
print(f"\n=== Métricas de evaluación (ROUGE) guardadas en {eval_metrics_path} ===")
for k, v in eval_metrics.items():
    print(f"{k}: {v}")

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
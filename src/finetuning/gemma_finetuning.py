from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import TrainingArguments
from trl import SFTTrainer
import os
from trl import SFTTrainer
import transformers
from peft import get_peft_model


model_id = "google/gemma-2b-it"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=nf4_config)

tokenizer = AutoTokenizer.from_pretrained(model_id, max_length = 512, truncation = True, padding = True) #,  add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset("json", data_files="/home/asdw/amazon-llm/code/CKM/dataset/medquad_instruct_train_12k.json", split="train")


peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=['k_proj', 'q_proj', 'gate_proj', 'o_proj', 'v_proj', 'down_proj', 'up_proj'],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)


args = TrainingArguments(
    output_dir="/media/asdw/130a808e-ec26-42a1-93ab-42b857d97bd4/capstone_files/code/gemma-2b-qa-med-16k-max-len-512-r64-a16",          # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    auto_find_batch_size=True,
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    # gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="paged_adamw_8bit",             
    # logging_steps=50,                       # log every 50 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler can change it to cosine
    
)

max_seq_length = 1024 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    dataset_text_field="text",
    # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

model.config.use_cache = False 
torch.cuda.empty_cache()
trainer.train()

# save model
save_directory = "/media/asdw/130a808e-ec26-42a1-93ab-42b857d97bd4/capstone_files/gemma-2b-qa-med-16k-max-len-512-r64-a16"
trainer.save_model(save_directory)
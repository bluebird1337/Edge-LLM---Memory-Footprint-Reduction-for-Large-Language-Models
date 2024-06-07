from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import TrainingArguments
from trl import SFTTrainer
import os
from trl import SFTTrainer


# Load jsonl data from disk
dataset = load_dataset("json", data_files="/home/asdw/amazon-llm/code/CKM/dataset/medquad_instruct_train_12k.json", split="train")
model_id = "meta-llama/Llama-2-7b-chat-hf" 

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=nf4_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id, max_length = 512, truncation = True, padding = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules=['o_proj','k_proj','q_proj','gate_proj','down_proj','v_proj', 'up_proj'],
        task_type="CAUSAL_LM",
        
)

args = TrainingArguments(
    output_dir="/media/asdw/130a808e-ec26-42a1-93ab-42b857d97bd4/capstone_files/code/llama2-7b-qa-med-16k-max-len-512-r64-a16",          # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    auto_find_batch_size=True,
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_bnb_8bit",             
    # logging_steps=50,                     # log every 50 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,
    weight_decay=0.001,                     # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",
    # group_by_length = True,                 # use constant learning rate scheduler
)


max_seq_length = None # max sequence length for model and packing of the dataset

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

# start training, the model will be automatically saved to the hub and the output directory
os.environ["WANDB_DISABLED"] = "true"
trainer.train()

# save model
save_directory = "/media/asdw/130a808e-ec26-42a1-93ab-42b857d97bd4/capstone_files/llama2-7b-qa-med-16k-max-len-512-r64-a16"
trainer.save_model(save_directory)


# CFG_FILE="projects/openelm/peft_configs/openelm_lora_3B.yaml"
# WTS_FILE="https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/peft/openelm_lora_3B.pt"
# TOKENIZER_FILE="/media/asdw/130a808e-ec26-42a1-93ab-42b857d97bd4/capstone_files/llama2_folder/tokenizer.model"
# # NOTE: The dataset can currently be obtained from https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json.
# DATASET_FILE="/home/asdw/amazon-llm/code/CKM/dataset/2023-04-12_oasst_prompts.messages.json"
# corenet-train --common.config-file $CFG_FILE \
#     --model.language-modeling.pretrained $WTS_FILE \
#     --text-tokenizer.sentence-piece.model-path $TOKENIZER_FILE \
#     --dataset.language-modeling.commonsense-170k.path $DATASET_FILE`ii`
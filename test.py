#%%
from huggingface_hub import snapshot_download

# 下载整个数据集到指定目录
local_dir = "/home/bowen/workspace/opencompass/data/BioProBench"
snapshot_download(
    repo_id="BioProBench/BioProBench",
    repo_type="dataset",
    revision="main",            # 固定版本（也可用具体 commit）
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("Downloaded to:", local_dir)

# ...existing code...
#%%
from opencompass.datasets.bioprobench import BioProBenchDataset
ds = BioProBenchDataset.load()
# %%
from datasets import load_dataset
ds = load_dataset("bowenxian/BioProBench", name="PQA", split="test")
# %%
from opencompass.models import HuggingFaceCausalLM

models = [
	dict(
		type=HuggingFaceCausalLM,
		abbr='qwen3-1.7b-hf',
		path='Qwen/Qwen3-1.7B',
		tokenizer_path='Qwen/Qwen3-1.7B',
		tokenizer_kwargs=dict(
			padding_side='left',
			truncation_side='left',
			trust_remote_code=True,
			use_fast=False,
		),
		# If you want the chat-tuned variant, change both paths to
		# 'Qwen/Qwen3-1.7B-Instruct'.
		max_out_len=512,
		max_seq_len=32768,
		batch_size=8,
		model_kwargs=dict(device_map='auto', trust_remote_code=True),
		generation_kwargs=dict(temperature=0.2, top_p=0.95, do_sample=False),
		run_cfg=dict(num_gpus=1, num_procs=1),
	)
]
# LLM_Demo

This repository contains implementations for [Qwen2-1.5b](https://github.com/QwenLM/Qwen2.5) on [GSM8K](https://github.com/openai/grade-school-math) (LLM) and [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4) on [MME](https://huggingface.co/datasets/lmms-lab/MME) (VLM). Follow the instructions below to set up and run each model separately.

## Table of Contents

- [General Setup](#general-setup)
- [Qwen](#qwen)
- [MiniGPT](#minigpt)

## General Setup

**Clone the Repository**:
   ```sh
   git clone https://github.com/GreatDanPeng/LLM_Demo.git
   cd LLM_Demo
   ```
**Create a Conda Environment**:
   ```sh
   conda create -n MLLM --file=MLLM_environment.yml
   conda activate MLLM
   ## if bugs
   pip3 install requirements.txt
   ```
## Qwen

**Run the Inference Script (Estimated Running Time: 7hrs on a 8GB NVIDIA RTX 3060Ti GPU)**:
   ```sh
   python Qwen2-1.5b_inference.py 
   ```

**Evaluation**:
   ```sh
   python QwenEval.py 
   ```

## MiniGPT

Following [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4), set pretrained weights and check points with Vicuna0-7B.

**Run the Inference Script**:
   ```sh
   python MiniGPT4_inference.py --cfg-path configs/minigpt4_eval.yaml --gpu-id 0
   ```


## Notes

- Ensure that you have **>16G** GPU resources available for MiniGPT4 inference on **7B** pretrained weights.
- Adjust the paths in the configuration files of MiniGPT4 to match your directory structure.
- For any issues or questions, please open an issue in the repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
import argparse
import json
import os
from tqdm import tqdm
from typing import List, Dict, Any
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVLInferenceModel:
    """Qwen2.5-VL模型推理类"""
    
    def __init__(self, model_path, device="cuda"):
        """
        初始化模型
        
        Args:
            model_path: 模型路径或名称
            device: 设备 ("cuda" 或 "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # 设置模型加载参数
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        # 加载模型
        print(f"Loading model from {model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # 如果设备是CPU，手动移动模型到CPU
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        
        # 加载处理器
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
        )
        
        print(f"Model loaded successfully on {self.device}")
    
    def infer(self, prompt: str, image_path: str, max_new_tokens: int = 128) -> str:
        """
        单次推理函数
        
        Args:
            prompt: 文本提示
            image_path: 图像路径
            max_new_tokens: 生成的最大token数
            
        Returns:
            模型生成的文本响应
        """
        try:
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # 准备消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # 准备推理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # 移动到正确的设备
            inputs = inputs.to(self.device)
            
            # 推理生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens
                )
            
            # 解码输出
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else ""
            
        except Exception as e:
            print(f"Error during inference for image {image_path}: {e}")
            return ""


class MVEIDatasetProcessor:
    """MVEI数据集处理器"""
    
    def __init__(self, data_dir):
        """
        初始化数据集处理器
        
        Args:
            data_dir: 图像基础路径
        """
        self.data_dir = data_dir
        
    def load_metadata(self, json_path: str) -> List[Dict]:
        """加载元数据JSON文件"""
        input_path = os.path.join(self.data_dir, json_path)
        print(f"Loading metadata from {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} items")
        return metadata
    
    def construct_prompt(self, statement: str) -> str:
        """构造推理提示"""
        prompt = f"<image><statement> {statement} Based on the provided image and emotional statement, please determine whether the statement aligns with the content of the image. If it does, respond with Correct. If it does not, respond with Incorrect. Do not output the reason."
        return prompt
    
    def get_full_image_path(self, image_id: str) -> str:
        """获取完整的图像路径"""
        return os.path.join(self.data_dir, "images", image_id)
    
    def process_dataset(self, metadata: List[Dict], model: QwenVLInferenceModel) -> List[Dict]:
        """
        处理数据集并添加预测结果
        
        Args:
            metadata: 元数据列表
            model: 模型实例
            
        Returns:
            处理后的元数据
        """
        processed_data = []
        
        for idx in tqdm(range(0, len(metadata))):
            item = metadata[idx].copy()  # 创建副本以避免修改原始数据
            image_id = item["image_id"]
            image_path = self.get_full_image_path(image_id)
            
            print(f"Processing item {idx + 1}/{len(metadata)}: {image_id}")
            
            # 处理每个statement
            for statement_item in item["statement_list"]:
                statement = statement_item["statement"]
                prompt = self.construct_prompt(statement)
                
                # 进行推理
                response = model.infer(prompt, image_path)
                
                # 添加预测结果
                statement_item["predict"] = response.strip()
                
                # 打印进度
                # print(f"  Statement: {statement[:50]}...")
                # print(f"  Response: {response}")
            
            processed_data.append(item)
            
            # 可选：每处理10个item保存一次中间结果
            if (idx + 1) % 10 == 0:
                print(f"Completed {idx + 1} items")
        
        return processed_data
    
    def save_results(self, processed_data: List[Dict], output_json: str):
        """保存处理结果到JSON文件"""
        output_path = os.path.join(self.data_dir, "predictions", output_json)
        print(f"Saving results to {output_path}...")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved successfully. Total items: {len(processed_data)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MVEI数据集示例推理程序")
    parser.add_argument("--MVEI_path", type=str, default="/mnt/bn/wdq-base1/data/VLMs/datasets/MVEI",
                        help="输入MVEI数据集的路径(可从huggingface获取)")
    parser.add_argument("--input_json", type=str, default="MVEI_metadata.json",
                       help="输入JSON文件路径")
    parser.add_argument("--output_json", type=str, default="MVEI_predict.json",
                       help="输出JSON文件路径")
    parser.add_argument("--model_path", type=str, default="/mnt/bn/wdq-base1/data/models/Qwen2.5-VL-7B-Instruct",
                       help="模型路径或名称")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="推理设备")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="生成的最大token数")
    
    args = parser.parse_args()

    # 初始化处理器
    processor = MVEIDatasetProcessor(args.MVEI_path)
    
    # 加载元数据
    metadata = processor.load_metadata(args.input_json)
    
    # 初始化模型，以qwen2.5-vl-7b-instruct为例
    model = QwenVLInferenceModel(
        model_path=args.model_path,
        device=args.device,
    )
    
    # 处理数据集
    processed_data = processor.process_dataset(
        metadata=metadata,
        model=model,
    )
    
    # 保存结果
    processor.save_results(processed_data, args.output_json)


if __name__ == "__main__":
    main()
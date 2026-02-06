import argparse
import json
from collections import defaultdict
from typing import List, Dict, Any

def parse_predict(predict_str: str) -> str:
    """解析预测字符串，返回'correct', 'incorrect'或'abstain'"""
    predict_lower = predict_str.lower()
    
    if 'incorrect' in predict_lower:
        return 'incorrect'
    elif 'correct' in predict_lower:
        return 'correct'
    else:
        return 'abstain'

def calculate_accuracy(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """计算准确率和相关统计"""
    
    # 按类别统计
    class_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'predict_correct': 0,
        'abstain': 0
    })
    
    # 总体统计
    overall_stats = {
        'total': 0,
        'correct': 0,
        'predict_correct': 0,
        'abstain': 0
    }
    
    for item in data:
        for statement in item['statement_list']:
            label = statement['label'].lower()
            predict = statement.get('predict', '')
            
            # 解析预测
            parsed_predict = parse_predict(predict)
            
            # 获取类别
            stmt_class = statement['class']
            
            # 更新类别统计
            class_stats[stmt_class]['total'] += 1
            overall_stats['total'] += 1
            
            # 统计预测为correct的情况
            if parsed_predict == 'correct':
                class_stats[stmt_class]['predict_correct'] += 1
                overall_stats['predict_correct'] += 1
            
            # 统计弃权情况
            if parsed_predict == 'abstain':
                class_stats[stmt_class]['abstain'] += 1
                overall_stats['abstain'] += 1
                continue
            
            # 检查预测是否正确
            if parsed_predict == label:
                class_stats[stmt_class]['correct'] += 1
                overall_stats['correct'] += 1
    
    # 计算最终结果
    results = {}
    
    # 计算整体统计
    if overall_stats['total'] > 0:
        results['overall'] = {
            'accuracy': overall_stats['correct'] / overall_stats['total'],
            'positive_ratio': overall_stats['predict_correct'] / overall_stats['total'],
            'give_up_ratio': overall_stats['abstain'] / overall_stats['total'],
            'total': overall_stats['total']
        }
    
    # 计算各类别统计
    for class_name, stats in class_stats.items():
        if stats['total'] > 0:
            results[class_name] = {
                'accuracy': stats['correct'] / stats['total'],
                'positive_ratio': stats['predict_correct'] / stats['total'],
                'give_up_ratio': stats['abstain'] / stats['total'],
                'total': stats['total']
            }
    
    return results

def get_class_abbreviation(class_name: str) -> str:
    """获取类别的缩写"""
    abbreviations = {
        'sentiment polarity': 'SP',
        'emotion interpretation': 'EI', 
        'scene context': 'SC',
        'perception subjectivity': 'PS'
    }
    return abbreviations.get(class_name, class_name)

def save_results_to_txt(results: Dict[str, Dict[str, float]], output_path: str):
    """保存结果到txt文件"""
    
    # 定义要输出的类别顺序
    target_classes = ['sentiment polarity', 'emotion interpretation', 
                     'scene context', 'perception subjectivity']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write(f"{'Class':<5} {'N':>6} {'Accuracy':>10} {'Positive Ratio':>15} {'Give up Ratio':>15}\n")
        f.write("-" * 61 + "\n")
        
        # 写入各类别结果
        for class_name in target_classes:
            if class_name in results:
                stats = results[class_name]
                class_abbr = get_class_abbreviation(class_name)
                
                f.write(f"{class_abbr:<5} "
                       f"{stats['total']:>6} "
                       f"{stats['accuracy']:>10.4f} "
                       f"{stats['positive_ratio']:>15.4f} "
                       f"{stats['give_up_ratio']:>15.4f}\n")
        
        # 写入整体统计
        f.write("-" * 61 + "\n")
        if 'overall' in results:
            stats = results['overall']
            f.write(f"{'Overall':<5} "
                   f"{stats['total']:>6} "
                   f"{stats['accuracy']:>10.4f} "
                   f"{stats['positive_ratio']:>15.4f} "
                   f"{stats['give_up_ratio']:>15.4f}\n")
        
        # 添加空行并写入详细说明
        f.write("\n" + "=" * 61 + "\n")
        f.write("统计说明:\n")
        f.write("1. Class: 类别缩写 (SP: sentiment polarity, EI: emotion interpretation,\n")
        f.write("           SC: scene context, PS: perception subjectivity)\n")
        f.write("2. N: 样本数量\n")
        f.write("3. Accuracy: 预测准确率 (预测与label一致的比例)\n")
        f.write("4. Positive Ratio: 模型预测为'correct'的比例\n")
        f.write("5. Give up Ratio: 模型弃权的比例 (既不是correct也不是incorrect)\n")
        f.write("\n预测解析规则:\n")
        f.write("1. 如果'incorrect'出现在predict中(忽略大小写): 预测为incorrect\n")
        f.write("2. 否则如果'correct'出现在predict中(忽略大小写): 预测为correct\n")
        f.write("3. 否则: 预测为弃权(abstain)\n")

def main():
    parser = argparse.ArgumentParser(description='统计模型预测准确率')
    parser.add_argument('--input_json', type=str, default="/mnt/bn/wdq-base1/data/VLMs/datasets/MVEI/MVEI_predict.json",
                       help='输入JSON文件路径')
    parser.add_argument('--output_txt', type=str, default="/mnt/bn/wdq-base1/data/VLMs/datasets/MVEI/MVEI_metrics.txt",
                       help='输出TXT文件路径')
    
    args = parser.parse_args()
    
    # 读取JSON文件
    print(f"读取文件: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"成功读取 {len(data)} 个数据项")
    
    # 计算统计结果
    results = calculate_accuracy(data)
    
    # 保存结果到txt文件
    save_results_to_txt(results, args.output_txt)
    
    print(f"结果已保存到: {args.output_txt}")
    
    # 在控制台也打印一份结果
    print("\n统计结果:")
    print(f"{'Class':<5} {'N':>6} {'Accuracy':>10} {'Positive Ratio':>15} {'Give up Ratio':>15}")
    print("-" * 61)
    
    target_classes = ['sentiment polarity', 'emotion interpretation', 
                     'scene context', 'perception subjectivity']
    
    for class_name in target_classes:
        if class_name in results:
            stats = results[class_name]
            class_abbr = get_class_abbreviation(class_name)
            print(f"{class_abbr:<5} "
                  f"{stats['total']:>6} "
                  f"{stats['accuracy']:>10.4f} "
                  f"{stats['positive_ratio']:>15.4f} "
                  f"{stats['give_up_ratio']:>15.4f}")
    
    print("-" * 61)
    if 'overall' in results:
        stats = results['overall']
        print(f"{'Overall':<5} "
              f"{stats['total']:>6} "
              f"{stats['accuracy']:>10.4f} "
              f"{stats['positive_ratio']:>15.4f} "
              f"{stats['give_up_ratio']:>15.4f}")

if __name__ == '__main__':
    main()
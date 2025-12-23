import os
import json
import shutil
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataClassifier:
    """数据分类与划分工具"""

    def __init__(self, img_dir, label_dir, output_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.output_dir = output_dir

        # 创建输出目录结构
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'val')    # 新增：验证集目录
        self.test_dir = os.path.join(output_dir, 'test')

        # 为每个类别创建子目录
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            for class_name in ['OK', 'NG']:
                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    def classify_and_split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """分类数据并划分为训练集、验证集和测试集"""
        # 获取所有图像文件
        img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.bmp')]

        # 过滤出存在对应标签的图像
        valid_img_files = []
        for img_file in img_files:
            # 正确生成标签文件名: .bmp.json
            label_file = img_file.replace('.bmp', '.bmp.json')
            label_path = os.path.join(self.label_dir, label_file)

            if os.path.exists(label_path):
                valid_img_files.append(img_file)
            else:
                print(f"警告: 找不到标签文件 {label_file}，跳过图像 {img_file}")

        if not valid_img_files:
            raise ValueError("没有找到有效的图像文件（含对应标签）")

        # 分类数据
        ok_images = []
        ng_images = []

        print("开始分类数据...")
        for img_file in tqdm(valid_img_files):
            # 正确生成标签文件名: .bmp.json
            label_file = img_file.replace('.bmp', '.bmp.json')
            label_path = os.path.join(self.label_dir, label_file)

            # 解析标签
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)

                # 根据"ng"字段分类
                is_ng = label_data.get('ng', False)
                if is_ng:
                    ng_images.append(img_file)
                else:
                    ok_images.append(img_file)
            except Exception as e:
                print(f"警告: 解析标签文件 {label_file} 出错: {e}，跳过图像 {img_file}")

        print(f"分类完成: OK = {len(ok_images)}, NG = {len(ng_images)}")

        if len(ok_images) + len(ng_images) == 0:
            raise ValueError("分类后没有有效样本，请检查标签文件内容")

        # 划分数据集
        print("开始划分数据集...")

        # 两次划分策略：先划分测试集，再从剩余数据中划分验证集
        # 验证集占比是相对于原始数据的比例
        train_val_size = 1.0 - test_size
        adjusted_val_size = val_size / train_val_size  # 相对于train+val的比例

        # 对OK类进行划分
        ok_train_val, ok_test = train_test_split(
            ok_images, test_size=test_size, random_state=random_state
        )
        ok_train, ok_val = train_test_split(
            ok_train_val, test_size=adjusted_val_size, random_state=random_state
        )

        # 对NG类进行划分
        ng_train_val, ng_test = train_test_split(
            ng_images, test_size=test_size, random_state=random_state
        )
        ng_train, ng_val = train_test_split(
            ng_train_val, test_size=adjusted_val_size, random_state=random_state
        )

        # 打印数据集统计信息
        print("\n数据集划分结果:")
        print(f"训练集: OK = {len(ok_train)}, NG = {len(ng_train)}, 总计 = {len(ok_train) + len(ng_train)}")
        print(f"验证集: OK = {len(ok_val)}, NG = {len(ng_val)}, 总计 = {len(ok_val) + len(ng_val)}")
        print(f"测试集: OK = {len(ok_test)}, NG = {len(ng_test)}, 总计 = {len(ok_test) + len(ng_test)}")

        # 保存数据集到文件
        print("开始复制文件到目标目录...")
        self._copy_files_to_dir(ok_train, 'OK', self.train_dir)
        self._copy_files_to_dir(ng_train, 'NG', self.train_dir)
        self._copy_files_to_dir(ok_val, 'OK', self.val_dir)    # 新增：复制验证集文件
        self._copy_files_to_dir(ng_val, 'NG', self.val_dir)    # 新增：复制验证集文件
        self._copy_files_to_dir(ok_test, 'OK', self.test_dir)
        self._copy_files_to_dir(ng_test, 'NG', self.test_dir)

        print(f"数据处理完成，已保存到 {self.output_dir}")

    def _copy_files_to_dir(self, files, class_name, target_dir):
        """将文件复制到目标目录"""
        for file in tqdm(files, desc=f"复制 {class_name} 文件到 {os.path.basename(target_dir)}"):
            src_img = os.path.join(self.img_dir, file)
            # 正确生成标签文件名: .bmp.json
            label_file = file.replace('.bmp', '.bmp.json')
            src_label = os.path.join(self.label_dir, label_file)

            dst_img = os.path.join(target_dir, class_name, file)
            dst_label = os.path.join(target_dir, class_name, label_file)

            # 复制图像和标签文件
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_label, dst_label)

    def get_dataset_statistics(self):
        """获取数据集统计信息"""
        stats = {}

        # 更新：包含验证集统计
        for split_name, split_dir in [('train', self.train_dir), ('val', self.val_dir), ('test', self.test_dir)]:
            split_stats = {}
            for class_name in ['OK', 'NG']:
                class_dir = os.path.join(split_dir, class_name)
                img_count = len([f for f in os.listdir(class_dir) if f.endswith('.bmp')])
                split_stats[class_name] = img_count

            total = sum(split_stats.values())
            split_stats['total'] = total
            stats[split_name] = split_stats

        return stats


# 示例：使用数据分类器
if __name__ == "__main__":
    # 配置路径
    IMG_DIR = "包装盒/原图"
    LABEL_DIR = "包装盒/标签图"
    OUTPUT_DIR = "包装盒_resnet"

    try:
        # 创建数据分类器
        classifier = DataClassifier(
            img_dir=IMG_DIR,
            label_dir=LABEL_DIR,
            output_dir=OUTPUT_DIR
        )

        # 分类并划分数据（默认测试集20%，验证集20%，训练集60%）
        classifier.classify_and_split_data(
            test_size=0.2,    # 测试集比例
            val_size=0.2,    # 验证集比例
            random_state=42  # 随机种子，保证结果可复现
        )

        # 获取并打印数据集统计信息
        stats = classifier.get_dataset_statistics()
        print("\n数据集统计信息:")
        for split_name, split_stats in stats.items():
            total = split_stats['total']
            ok_count = split_stats['OK']
            ng_count = split_stats['NG']
            print(f"{split_name}: 总数={total}, OK={ok_count}, NG={ng_count}, OK比例={ok_count / total:.2%}")

    except Exception as e:
        print(f"处理过程中出错: {e}")
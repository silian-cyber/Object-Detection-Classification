import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from torchvision import transforms
import time

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class PackagingInspector:
    """包装OK/NG检测系统"""

    def __init__(self, root):
        self.root = root
        self.root.title("包装OK/NG智能检测系统")
        self.root.geometry("1200x750")
        self.root.resizable(True, True)

        # 初始化设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 先创建日志组件
        self.create_log_widget()

        # 加载模型
        self.model = self.load_model()

        # 图像预处理
        self.transform = self.get_transform()

        # 类别配置
        self.class_names = ["NG", "OK"]  # 0=NG, 1=OK

        # 界面状态
        self.image_path = ""
        self.current_image = None
        self.result_images = {}  # 存储生成的图表图像

        # 创建其他界面组件
        self.create_other_widgets()

        self.log(f"系统初始化完成，使用设备: {self.device}")

    def create_log_widget(self):
        """创建日志组件"""
        log_frame = tk.Frame(self.root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        tk.Label(log_frame, text="操作日志:", font=("SimHei", 12, "bold"), fg="#333").pack(anchor=tk.W)
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, font=("SimHei", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

    def load_model(self):
        """加载训练好的模型"""
        model_path = "resnet18_model_optimized.pth"
        if not os.path.exists(model_path):
            messagebox.showerror("错误", f"未找到模型文件: {model_path}")
            self.root.quit()

        try:
            self.log("正在加载模型...")

            # 加载预训练ResNet18
            model = torch.hub.load('pytorch/vision:v0.13.0', 'resnet18', weights=None)
            self.log("基础模型加载成功")

            # 确保模型全连接层结构与训练时一致
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),  # 假设训练时使用了该Dropout层
                nn.Linear(num_ftrs, 2)
            )
            self.log("模型全连接层结构初始化完成")

            # 加载模型权重
            state_dict = torch.load(model_path, map_location=self.device)
            self.log("权重加载成功")

            # 加载权重到模型
            model.load_state_dict(state_dict)
            self.log("权重与模型匹配成功")

            model = model.to(self.device)
            model.eval()
            self.log("模型已设置为评估模式")

            return model

        except Exception as e:
            self.log(f"模型加载异常: {str(e)}")
            messagebox.showerror("模型加载错误", f"加载失败详情:\n{str(e)}")
            self.root.quit()

    def get_transform(self):
        """图像预处理转换"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def create_other_widgets(self):
        """创建除日志外的其他界面组件"""
        # 顶部控制栏
        control_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        control_frame.pack(fill=tk.X, padx=20)

        tk.Button(control_frame, text="选择图片", command=self.select_image,
                  font=("SimHei", 12), bg="#4CAF50", fg="white",
                  padx=15, pady=5, relief=tk.RAISED, bd=2).pack(side=tk.LEFT)

        tk.Button(control_frame, text="开始检测", command=self.predict_image,
                  font=("SimHei", 12), bg="#2196F3", fg="white",
                  padx=15, pady=5, relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="清除", command=self.clear_all,
                  font=("SimHei", 12), bg="#f44336", fg="white",
                  padx=15, pady=5, relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=5)

        # 模型状态显示
        self.model_status = tk.Label(control_frame, text="模型已加载",
                                     font=("SimHei", 10), fg="#333", bg="#f0f0f0")
        self.model_status.pack(side=tk.RIGHT, padx=10)

        # 图像显示区
        display_frame = tk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 原始图像
        self.original_frame = tk.LabelFrame(display_frame, text="原始图像",
                                            font=("SimHei", 12, "bold"), padx=10, pady=10)
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.original_label = tk.Label(self.original_frame, text="未选择图片",
                                       font=("SimHei", 12), fg="#666",
                                       width=40, height=20,
                                       relief=tk.GROOVE, bd=2)
        self.original_label.pack(fill=tk.BOTH, expand=True)

        # 结果图像
        self.result_frame = tk.LabelFrame(display_frame, text="检测结果",
                                          font=("SimHei", 12, "bold"), padx=10, pady=10)
        self.result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        self.result_label = tk.Label(self.result_frame, text="检测结果将显示在这里",
                                     font=("SimHei", 12), fg="#666",
                                     width=40, height=20,
                                     relief=tk.GROOVE, bd=2)
        self.result_label.pack(fill=tk.BOTH, expand=True)

        # 结果信息区
        info_frame = tk.Frame(self.root, pady=10)
        info_frame.pack(fill=tk.X, padx=20)

        tk.Label(info_frame, text="检测结果:", font=("SimHei", 14, "bold"), fg="#333").pack(side=tk.LEFT, padx=5)
        self.result_var = tk.StringVar(value="未进行检测")
        result_label = tk.Label(info_frame, textvariable=self.result_var,
                                font=("SimHei", 14), fg="#007ACC",
                                bg="#f9f9f9", padx=10, pady=5,
                                relief=tk.RAISED, bd=2)
        result_label.pack(side=tk.LEFT, padx=10)

        # 概率信息
        prob_frame = tk.Frame(self.root, pady=5)
        prob_frame.pack(fill=tk.X, padx=20)

        self.ng_prob = tk.StringVar(value="NG概率: --")
        self.ok_prob = tk.StringVar(value="OK概率: --")

        tk.Label(prob_frame, textvariable=self.ng_prob, font=("SimHei", 12),
                 bg="#f9f9f9", padx=10, pady=5,
                 relief=tk.SUNKEN, bd=1).pack(side=tk.LEFT, padx=10)

        tk.Label(prob_frame, textvariable=self.ok_prob, font=("SimHei", 12),
                 bg="#f9f9f9", padx=10, pady=5,
                 relief=tk.SUNKEN, bd=1).pack(side=tk.LEFT, padx=10)

        # 底部状态栏
        status_bar = tk.Label(self.root, text="就绪",
                              font=("SimHei", 10), fg="#666", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def select_image(self):
        """选择图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择包装图片",
            filetypes=[("图像文件", "*.bmp *.jpg *.jpeg *.png *.gif")]
        )

        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.log(f"已选择图片: {os.path.basename(file_path)}")
            self.result_var.set("未进行检测")
            self.ng_prob.set("NG概率: --")
            self.ok_prob.set("OK概率: --")

    def display_image(self, path):
        """显示图像"""
        try:
            # 读取图像并调整大小
            img = Image.open(path)
            img = img.convert("RGB")
            max_size = (400, 400)
            img.thumbnail(max_size, Image.LANCZOS)

            # 显示原始图像
            photo = ImageTk.PhotoImage(img)
            self.original_label.config(image=photo)
            self.original_label.image = photo
            self.original_label.config(text="")

        except Exception as e:
            self.log(f"图像显示错误: {str(e)}")
            messagebox.showerror("错误", f"无法显示图像: {str(e)}")

    def predict_image(self):
        """执行图像预测"""
        if not self.image_path:
            messagebox.showwarning("警告", "请先选择图片")
            return

        self.log("开始检测...")
        self.model_status.config(text="检测中...", fg="blue")
        self.root.update()  # 更新界面显示

        try:
            # 预处理图像
            start_time = time.time()
            img = Image.open(self.image_path).convert("RGB")
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # 模型预测
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, 1)

                # 解析结果
                pred_class = self.class_names[pred_idx.item()]
                ng_prob = probs[0, 0].item()  # NG概率
                ok_prob = probs[0, 1].item()  # OK概率

                # 设置结果颜色
                result_color = "red" if pred_class == "NG" else "green"
                bg_color = "#fff8e1" if pred_class == "NG" else "#e8f5e9"

                # 更新结果显示
                self.result_var.set(f"检测结果: {pred_class}, 置信度: {confidence.item():.2f}")
                self.result_label_widget = self.result_var

                self.result_label.config(fg=result_color, bg=bg_color,
                                         font=("SimHei", 14, "bold"))
                self.ng_prob.set(f"NG概率: {ng_prob:.4f}")
                self.ok_prob.set(f"OK概率: {ok_prob:.4f}")

                # 生成概率图
                self.show_probability_plot(ok_prob, ng_prob)

                # 计算耗时
                elapsed_time = time.time() - start_time
                self.log(f"检测完成: {pred_class}, 置信度: {confidence.item():.2f}, 耗时: {elapsed_time:.2f}秒")
                self.model_status.config(text="检测完成", fg="#333")

        except Exception as e:
            self.log(f"检测错误: {str(e)}")
            messagebox.showerror("检测错误", f"检测过程出错: {str(e)}")
            self.model_status.config(text="检测失败", fg="red")

    def show_probability_plot(self, ok_prob, ng_prob):
        """显示概率条形图"""
        try:
            plt.figure(figsize=(6, 4))
            classes = ['NG', 'OK']  # 类别顺序
            probs = [ng_prob, ok_prob]  # 对应概率

            # 颜色配置：NG红色，OK绿色，最大值高亮
            colors = ['#ff6b6b', '#4cd137']
            max_idx = probs.index(max(probs))
            colors[max_idx] = '#ff9f1c'  # 最大值用橙色突出

            bars = plt.bar(classes, probs, color=colors, alpha=0.8, zorder=3)

            # 添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                         f'{probs[i]:.4f}', ha='center', va='bottom')

            plt.title('类别概率分布', fontsize=14)
            plt.ylim(0, 1.05)
            plt.grid(axis='y', linestyle='--', zorder=0)

            # 优化图表样式
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # 保存图表
            timestamp = int(time.time())
            plot_dir = "检测结果"
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir,
                                     f"prob_{os.path.basename(self.image_path).split('.')[0]}_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 显示概率图
            img = Image.open(plot_path)
            img.thumbnail((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.result_label.config(image=photo)
            self.result_label.image = photo
            self.result_label.config(text="")

            self.log(f"检测结果图已保存至: {plot_path}")

        except Exception as e:
            self.log(f"图表生成错误: {str(e)}")
            self.result_label.config(text="概率图生成失败")

    def clear_all(self):
        """清除所有内容"""
        self.original_label.config(text="未选择图片")
        self.original_label.config(image="")
        self.result_label.config(text="检测结果将显示在这里")
        self.result_label.config(image="")
        self.result_var.set("未进行检测")
        self.result_var.tk.config(fg="#007ACC", bg="SystemButtonFace",
                                  font=("SimHei", 14))
        self.ng_prob.set("NG概率: --")
        self.ok_prob.set("OK概率: --")
        self.image_path = ""
        self.log("已清除所有检测内容")

    def log(self, message):
        """记录日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    # 检查模型文件
    if not os.path.exists("resnet18_model_optimized.pth"):
        messagebox.showerror("错误", "未找到模型文件，请确保resnet18_model_optimized.pth存在")
        exit(1)

    root = tk.Tk()
    app = PackagingInspector(root)
    root.mainloop()
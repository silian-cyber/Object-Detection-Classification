import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('Agg')  # 设置非GUI后端
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms, models
import time
import io
import base64

# 确保中文显示正常（后续通过手动加载字体强化）
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

app = Flask(__name__)

# 配置上传文件夹和允许的文件类型
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = '检测结果'
ALLOWED_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'png', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 初始化设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 加载模型
def load_model():
    """加载训练好的模型"""
    model_path = "resnet18_model_optimized.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    try:
        print("正在加载模型...")

        # 创建ResNet-18模型结构 (不下载预训练权重)
        model = models.resnet18(pretrained=False)

        # 确保模型全连接层结构与训练时一致
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),  # 假设训练时使用了该Dropout层
            nn.Linear(num_ftrs, 2)
        )
        print("模型结构初始化完成")

        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        print("权重加载成功")

        # 加载权重到模型
        model.load_state_dict(state_dict)
        print("权重与模型匹配成功")

        model = model.to(device)
        model.eval()
        print("模型已设置为评估模式")

        return model

    except Exception as e:
        print(f"模型加载异常: {str(e)}")
        raise


# 图像预处理
def get_transform():
    """图像预处理转换"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# 检查文件扩展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 【新增】手动加载中文字体（需确保项目中有SimHei.ttf文件）
def load_chinese_font():
    font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'SimHei.ttf')  # 字体路径：项目根目录/fonts/SimHei.ttf
    if not os.path.exists(font_path):
        raise FileNotFoundError("未找到SimHei.ttf字体文件，请检查路径或添加字体文件")
    return FontProperties(fname=font_path)


# 预测函数
def predict_image(image_path, model):
    """执行图像预测"""
    try:
        # 预处理图像
        start_time = time.time()
        img = Image.open(image_path).convert("RGB")
        transform = get_transform()
        input_tensor = transform(img).unsqueeze(0).to(device)

        # 模型预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

            # 解析结果
            class_names = ["NG", "OK"]
            pred_class = class_names[pred_idx.item()]
            ng_prob = probs[0, 0].item()  # NG概率
            ok_prob = probs[0, 1].item()  # OK概率

            # 计算耗时
            elapsed_time = time.time() - start_time

            # 生成概率图
            prob_plot = generate_probability_plot(ok_prob, ng_prob)

            return {
                'pred_class': pred_class,
                'confidence': confidence.item(),
                'ng_prob': ng_prob,
                'ok_prob': ok_prob,
                'elapsed_time': elapsed_time,
                'prob_plot': prob_plot
            }

    except Exception as e:
        print(f"检测错误: {str(e)}")
        raise


# 生成概率图（修改后：支持中文显示）
def generate_probability_plot(ok_prob, ng_prob):
    """生成概率条形图并返回Base64编码的图像"""
    try:
        # 加载自定义中文字体
        chinese_font = load_chinese_font()

        with plt.style.context('seaborn'):
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

            # 标题和坐标轴使用中文字体
            plt.title('类别概率分布', fontproperties=chinese_font, fontsize=14)
            plt.xticks(fontproperties=chinese_font, fontsize=12)  # X轴类别（NG/OK为英文，可选增强）
            plt.yticks(fontsize=12)

            plt.ylim(0, 1.05)
            plt.grid(axis='y', linestyle='--', zorder=0)

            # 优化图表样式
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)

            # 将图表保存到内存中
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plt.close()  # 确保关闭图表以释放资源

            # 转换为Base64编码
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return image_base64

    except Exception as e:
        print(f"图表生成错误: {str(e)}")
        return None


# 路由定义
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': '未找到文件'}), 400

    file = request.files['file']

    # 如果用户没有选择文件，浏览器可能会提交一个空文件
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    # 检查文件类型
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # 预测图像
            result = predict_image(file_path, model)

            # 返回结果
            return jsonify({
                'success': True,
                'filename': filename,
                'result': result
            })
        except Exception as e:
            return jsonify({'error': f'处理图像时出错: {str(e)}'}), 500
    else:
        return jsonify({'error': '不支持的文件类型'}), 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# 错误处理
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': '文件大小超过限制 (16MB)'}), 413


# 加载模型
try:
    model = load_model()
    print("模型加载成功，系统就绪")
except Exception as e:
    print(f"系统初始化失败: {str(e)}")
    model = None

if __name__ == '__main__':
    # 检查模型文件
    if model is None:
        print("未找到有效模型，应用无法启动")
    else:
        # 生产环境应关闭调试模式
        app.run(debug=False)
        # 如需保留调试模式，但禁用重新加载功能，可使用：
        # app.run(debug=True, use_reloader=False)
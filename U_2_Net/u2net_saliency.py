import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from PIL import Image
import glob

# 假设这些模块在您的项目中可用
from U_2_Net.data_loader import RescaleT, ToTensorLab
from U_2_Net.model import U2NET, U2NETP

class U2NetSaliencyDetector:
    def __init__(self, model_path, model_type='u2net', device=None):
        """
        初始化U2Net显著性检测器
        
        参数:
            model_path (str): 预训练模型权重文件的路径
            model_type (str): 模型类型，'u2net' 或 'u2netp'
            device (str): 计算设备，None表示自动选择 (cuda如果可用，否则cpu)
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # 加载模型
        if model_type == 'u2net':
            self.net = U2NET(3, 1)
            print("加载U2NET---173.6 MB")
        elif model_type == 'u2netp':
            self.net = U2NETP(3, 1)
            print("加载U2NEP---4.7 MB")
        else:
            raise ValueError("model_type必须是'u2net'或'u2netp'")
            
        # 加载权重
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_path))
        else:
            self.net.load_state_dict(torch.load(model_path, map_location='cpu'))
            
        self.net.to(self.device)
        self.net.eval()
        
        # 定义图像变换
        self.transform = transforms.Compose([
            RescaleT(320),
            ToTensorLab(flag=0)
        ])
    
    def normalize_prediction(self, prediction):
        """归一化预测结果到[0, 1]范围"""
        ma = torch.max(prediction)
        mi = torch.min(prediction)
        return (prediction - mi) / (ma - mi)
    
    def predict_saliency(self, image_path, output_dir=None):
        """
        对单张图像进行显著性检测
        
        参数:
            image_path (str): 输入图像路径
            output_dir (str): 输出目录，如果为None则不保存结果
            
        返回:
            numpy数组: 显著性图，值范围[0, 1]
        """
        # 创建数据加载项
        from U_2_Net.data_loader import SalObjDataset
        dataset = SalObjDataset(
            img_name_list=[image_path],
            lbl_name_list=[],
            transform=self.transform
        )
        
        # 获取图像
        data = dataset[0]
        inputs_test = data['image'].unsqueeze(0).type(torch.FloatTensor)
        
        # 移动到设备
        inputs_test = Variable(inputs_test.to(self.device))
        
        # 推理
        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7 = self.net(inputs_test)
        
        # 归一化预测
        pred = d1[:, 0, :, :]
        pred = self.normalize_prediction(pred)
        
        # 转换为numpy数组
        predict_np = pred.squeeze().cpu().numpy()
        
        # 如果需要保存结果
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 读取原始图像以获取尺寸
            original_image = Image.open(image_path).convert('RGB')
            original_size = original_image.size
            
            # 调整显著性图大小
            saliency_img = Image.fromarray((predict_np * 255).astype(np.uint8))
            saliency_img = saliency_img.resize(original_size, Image.BILINEAR)
            
            # 保存结果
            img_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(img_name)[0]
            output_path = os.path.join(output_dir, f"{name_without_ext}_saliency.png")
            saliency_img.save(output_path)
            print(f"显著性图已保存至: {output_path}")
        
        return predict_np
    
    def predict_saliency_from_image(self, image, output_path=None):
        """
        从PIL图像对象进行显著性检测
        
        参数:
            image (PIL.Image): 输入图像
            output_path (str): 输出路径，如果为None则不保存结果
            
        返回:
            numpy数组: 显著性图，值范围[0, 1]
        """
        # 临时保存图像
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # 使用predict_saliency方法
        result = self.predict_saliency(temp_path, 
                                      output_dir=os.path.dirname(output_path) if output_path else None)
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # 如果需要重命名输出文件
        if output_path and os.path.exists(os.path.join(os.path.dirname(output_path), 
                                                     f"{os.path.splitext(os.path.basename(temp_path))[0]}_saliency.png")):
            os.rename(
                os.path.join(os.path.dirname(output_path), 
                            f"{os.path.splitext(os.path.basename(temp_path))[0]}_saliency.png"),
                output_path
            )
            
        return result
    
    def process_folder(self, input_dir, output_dir):
        """
        处理整个文件夹中的图像
        
        参数:
            input_dir (str): 输入图像文件夹路径
            output_dir (str): 输出文件夹路径
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 获取所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for extension in image_extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, extension)))
            
        print(f"找到 {len(image_paths)} 张图像")
        
        # 处理每张图像
        for i, image_path in enumerate(image_paths):
            print(f"处理图像 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            self.predict_saliency(image_path, output_dir)


# 使用示例
if __name__ == "__main__":
    # 初始化检测器
    model_path = "/home/mxxie/SemAID/U-2-Net/model/u2net.pth"  # 替换为您的模型路径
    detector = U2NetSaliencyDetector(model_path, model_type='u2net')
    
    # 处理单张图像
    saliency_map = detector.predict_saliency("test_image.jpg", "output_directory")
    
    # 处理整个文件夹
    # detector.process_folder("input_images", "output_directory")
    
    # 从PIL图像对象处理
    # from PIL import Image
    # img = Image.open("test_image.jpg")
    # saliency_map = detector.predict_saliency_from_image(img, "output.png")
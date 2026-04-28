import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# 1. 网页基本设置
st.set_page_config(page_title="榫卯结构智能识别", page_icon="🪑", layout="wide")


# 2. 缓存模型（非常关键：加上这个，模型只加载一次，网页不会卡顿）
@st.cache_resource
def load_model():
    # 组长注意：如果你的 best.pt 挪地方了，请修改这里的路径
    return YOLO(r'C:\Users\wmt\runs\detect\train3\weights\best.pt')


model = load_model()

# 3. 侧边栏设置 (给评委操作的控制台)
st.sidebar.title("⚙️ AI 控制台")
st.sidebar.markdown("---")
# 添加一个滑动条，让评委可以实时调整阈值
conf_threshold = st.sidebar.slider(
    "置信度阈值 (Confidence)",
    min_value=0.01, max_value=1.0, value=0.15, step=0.01,
    help="调低此值，AI会更激进地识别；调高此值，AI会更保守。"
)

st.sidebar.markdown("""
### 识别类别 (9类)
* 霸王棖 (Bawang_Cheng)
* 斗拱 (Dougong)
* 燕尾榫 (Dovetail_Joint)
* 扇面榫 (Fan_Shaped)
* 插肩榫 (Inserted_Shoulder)
* 格角榫 (Mitred_Joint)
* 馒头榫 (Mortise_Tenon)
* 抱肩榫 (Shoulder_Hugging)
* 企口榫 (Tongue_Groove)
""")

# 4. 网页主界面
st.title("🪑 中国传统榫卯结构 AI 识别系统")
st.markdown("#### 研发团队：wmt 组长及团队 | 基于 YOLO11 核心驱动")
st.markdown("---")

# 5. 图片上传组件
uploaded_file = st.file_uploader("请在此上传需要检测的榫卯图片 (支持 jpg, png, jpeg)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 将上传的文件转为图像
    image = Image.open(uploaded_file)

    # 将界面分成左右两列
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 原始输入图片")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🤖 AI 识别结果")
        # 增加一个加载动画，显得更专业
        with st.spinner('AI 正在高速推理中...'):
            # 将 PIL 图像转换为 OpenCV 可用的 NumPy 数组
            img_array = np.array(image)

            # YOLO 预测
            results = model.predict(source=img_array, conf=conf_threshold, save=False)

            # 获取画好框的图片 (BGR格式)
            res_plotted = results[0].plot()

            # OpenCV 默认是 BGR，Streamlit 显示需要 RGB，做一次转换
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            # 在右侧显示结果
            st.image(res_rgb, use_container_width=True)

        # 提取识别到了什么，显示在下方
        detected_classes = results[0].boxes.cls.tolist()
        names = model.names
        if detected_classes:
            st.success("✅ 检测完成！")
            for cls_id in set(detected_classes):
                st.write(f"- 发现了: **{names[int(cls_id)]}**")
        else:
            st.warning("⚠️ 在当前阈值下，未能识别出明确的榫卯结构，请尝试在左侧调低【置信度阈值】。")
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# 1. 网页基本设置
st.set_page_config(page_title="榫卯结构智能识别", page_icon="🪑", layout="wide")

# 2. 缓存模型加载
@st.cache_resource
def load_model():
    # 自动识别环境：如果是在本地跑，请确保文件在同级目录
    # 如果是在云端，GitHub 仓库根目录必须有 best.pt
    model_path = "best.pt" 
    return YOLO(model_path)

# 尝试加载模型
try:
    model = load_model()
    model_names = model.names
except Exception as e:
    st.error(f"模型加载失败，请检查仓库中是否存在 best.pt 文件。错误详情: {e}")
    st.stop()

# 3. 侧边栏设置
st.sidebar.title("⚙️ AI 控制台")
st.sidebar.markdown("---")
conf_threshold = st.sidebar.slider(
    "置信度阈值 (Confidence)",
    min_value=0.01, max_value=1.0, value=0.15, step=0.01,
    help="调低此值，AI会更激进地识别；调高此值，AI会更保守。"
)

# 中文映射增强展示
CN_NAMES = {
    "Bawang_Cheng": "霸王棖",
    "Dougong": "斗拱",
    "Dovetail_Joint": "燕尾榫",
    "Fan_Shaped": "扇面榫",
    "Inserted_Shoulder": "插肩榫",
    "Mitred_Joint": "格角榫",
    "Mortise_Tenon": "馒头榫",
    "Shoulder_Hugging": "抱肩榫",
    "Tongue_Groove": "企口榫"
}

st.sidebar.markdown("### 当前支持类别")
for eng, chi in CN_NAMES.items():
    st.sidebar.text(f"• {chi} ({eng})")

# 4. 网页主界面
st.title("🪑 中国传统榫卯结构 AI 识别系统")
st.markdown("#### 研发团队：wmt 组长及团队 | 实时云端版")
st.markdown("---")

# 5. 图片上传
uploaded_file = st.file_uploader("上传榫卯图片 (jpg, png, jpeg)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 原始输入图片")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🤖 AI 识别结果")
        with st.spinner('AI 正在高速推理中...'):
            # 将 PIL 图像直接传入模型
            results = model.predict(source=image, conf=conf_threshold, save=False)

            # 获取绘制后的结果数组 (注意：YOLO 绘图默认是 BGR 格式)
            res_plotted = results[0].plot()

            # 颜色空间转换：从 BGR 转为 RGB 供 Streamlit 显示
            # 我们不直接调用 cv2.cvtColor 以防环境崩溃，改用 numpy 翻转
            res_rgb = res_plotted[:, :, ::-1]

            st.image(res_rgb, use_container_width=True)

        # 结果明细展示
        detected_indices = results[0].boxes.cls.tolist()
        if detected_indices:
            st.success(f"✅ 检测完成！共发现 {len(detected_indices)} 个目标")
            
            # 使用 Set 去重显示类别
            unique_classes = set(detected_indices)
            for cls_id in unique_classes:
                eng_name = model_names[int(cls_id)]
                chi_name = CN_NAMES.get(eng_name, "未知类别")
                st.info(f"🔍 检测到：**{chi_name}** ({eng_name})")
        else:
            st.warning("⚠️ 未能识别出明确的榫卯结构。建议：调低左侧【置信度阈值】或更换拍摄角度再试。")

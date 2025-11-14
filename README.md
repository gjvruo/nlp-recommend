# 🛒 基于NLP的用户评论情感分析与商品推荐系统

这是一个基于自然语言处理（NLP）技术实现的智能商品推荐系统。系统能够分析用户评论的情感倾向，并结合多种策略（如热度、口碑）为用户提供更精准、更高质量的商品推荐。

该项目是我的本科毕业设计，旨在探索情感分析在电子商务推荐领域的应用。

## ✨ 项目亮点 (Features)

- 情感驱动: 核心优势在于将用户评论的情感量化为“口碑”，优化传统推荐结果。
- 多策略推荐: 内置多种推荐策略（基线热度、情感过滤、加权综合分、口碑优先），可灵活切换。
- 交互式Web界面: 使用 Streamlit 构建了简单直观的Web用户界面，方便上传数据并实时查看推荐结果。
- 模块化设计: 代码结构清晰，将数据处理、模型、推荐逻辑和UI界面解耦，易于维护和扩展。

## 🚀 如何运行 (Quick Start)

1.  克隆仓库
    git clone https://github.com/你的用户名/intelligent-recommendation-system.git
    cd intelligent-recommendation-system
    
3.  创建虚拟环境并安装依赖
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt

4.  首次运行：训练模型
    在首次运行前，需要先生成情感分析模型文件。
  
    python recommender_system.py
 
    运行成功后，你会在 `model/` 文件夹下看到 `.pkl` 文件。

5.  启动Web应用

    streamlit run app.py
    
    应用将在 `http://localhost:8501` 启动。

## 🛠️ 技术栈 (Technology Stack)

- 核心算法: Python, Pandas, Scikit-learn (TF-IDF, SVM)
- 中文分词: Jieba
- Web框架: Streamlit
- 项目管理: Git, GitHub


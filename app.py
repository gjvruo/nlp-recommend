
# app.py (修改后)
import streamlit as st
import pandas as pd
# <<< 核心修复：同时导入 run_pipeline 和 jieba_tokenizer >>>
from recommender_system import run_pipeline, jieba_tokenizer

# --- 页面配置 ---
st.set_page_config(
    page_title="智能商品推荐系统",
    page_icon="🛒",
    layout="wide"
)

# --- 主标题 ---
st.title("🛒 基于NLP的用户评论情感分析与商品推荐系统")
st.write("上传您的天猫评论数据，选择一个推荐策略，系统将为您生成Top-10商品推荐列表。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("控制面板")
    # 1. 文件上传组件
    uploaded_file = st.file_uploader(
        "请上传您的评论数据 (CSV格式)",
        type=['csv']
    )

    # 2. 策略选择下拉菜单
    strategy = st.selectbox(
        "请选择推荐策略",
        (
            '基线策略 (仅热度)',
            '情感过滤+热度',
            '加权综合分',
            '口碑优先'
        ),
        index=3  # 默认选择'口碑优先'
    )

    # 3. 执行按钮
    run_button = st.button("🚀 开始分析与推荐")

# --- 主内容区 ---
# app.py

# 只有当文件被上传并且按钮被点击后，才执行核心逻辑
if uploaded_file is not None and run_button:
    # 读取上传的CSV文件为DataFrame
    try:
        df_reviews = pd.read_csv(uploaded_file)
        st.info(f"✅ 文件上传成功，包含 {len(df_reviews)} 条评论。")
        st.info(f"您选择的策略是： **{strategy}**")


        # 使用一个加载动画，提升用户体验
        with st.spinner('系统正在进行情感分析和推荐计算，请稍候...'):
            # 调用你的核心处理流程！
            # 【注意】你需要确保run_pipeline函数能够正确处理df_reviews
            # 并且你的 build_profiles 函数能够工作
            # 为了演示，我们先假设一个可以工作的 build_profiles
            # 在实际使用时，请替换成你自己的完整 pipeline

            # 临时的 build_profiles 模拟
            def build_profiles_mock(df):
                # 假设 df 有 'ProductID' 和 'Sentiment' (1=positive, -1=negative)
                # 这里你需要替换成你论文中真实的聚合逻辑
                profiles = df.groupby('ProductID')['Sentiment'].agg(['count', 'sum']).reset_index()
                profiles.rename(columns={'count': 'ReviewCount', 'sum': 'SentimentScore'}, inplace=True)
                profiles['PositiveCount'] = df[df['Sentiment'] == 1].groupby('ProductID').size().reindex(
                    profiles['ProductID']).fillna(0)
                profiles['NegativeCount'] = df[df['Sentiment'] == -1].groupby('ProductID').size().reindex(
                    profiles['ProductID']).fillna(0)
                profiles['PositiveRatio'] = profiles['PositiveCount'] / profiles['ReviewCount']
                profiles['NegativeRatio'] = profiles['NegativeCount'] / profiles['ReviewCount']
                return profiles



            recommendations = run_pipeline(df_reviews, strategy)

            # 我们用模拟函数来演示
            # 先模拟一个Sentiment列
            import numpy as np

            #df_reviews['Sentiment'] = np.random.choice([1, -1, 0], size=len(df_reviews), p=[0.7, 0.2, 0.1])
            #profiles_df = build_profiles_mock(df_reviews)
            #recommendations = get_recommendations(profiles_df, strategy)

        # app.py (修改后，更健壮)
        # ...
        recommendations = run_pipeline(df_reviews, strategy)
        st.success("🎉 推荐结果生成完毕！")

        # <<< 核心修复：在展示结果前，先检查一下是否有结果 >>>
        if recommendations is not None and not recommendations.empty:
            st.subheader("🏆 Top-10 商品推荐列表")
            st.dataframe(recommendations)

            st.subheader("📊 推荐结果可视化")

            # 将商品ID转换为字符串，以便绘图
            # 确保 'ProductID' 列存在
            if 'ProductID' in recommendations.columns:
                recommendations['ProductID'] = recommendations['ProductID'].astype(str)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**好评率 (Positive Ratio)**")
                    st.bar_chart(recommendations.set_index('ProductID')['positive_ratio'])

                with col2:
                    st.write("**总评论数 (Review Count)**")
                    st.bar_chart(recommendations.set_index('ProductID')['review_count'])
            else:
                st.warning("推荐结果中未找到 'ProductID' 列，无法进行可视化。")

        else:
            # 如果没有推荐结果，就显示友好提示
            st.warning("🤔 根据您选择的策略和上传的数据，未能筛选出合适的推荐商品。")
    except Exception as e:
        st.error(f"处理文件时发生错误: {e}")
        st.warning("请确保上传的CSV文件包含 'ProductID' 列。")

elif run_button:
    st.warning("请先上传一个评论数据文件！")
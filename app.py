# app.py
# app.py

# ==============================================================================
# 核心库导入
# ==============================================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import jieba
import os


from recommender_system import run_pipeline, load_stopwords, clean_comment_text, jieba_tokenizer
# ==============================================================================
# 辅助函数 (用于UI)
# ==============================================================================


# 使用 @st.cache_data 缓存函数结果，避免重复计算，提升性能
@st.cache_data
def create_sentiment_pie_chart(df):
    # 确保列名正确
    if 'predicted_sentiment' not in df.columns:
        st.error("无法创建饼图：DataFrame中缺少 'predicted_sentiment' 列。")
        return plt.figure()

    sentiment_counts = df['predicted_sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90,
           colors=['#66b3ff', '#ff9999', '#99ff99'])
    ax.axis('equal')
    return fig



@st.cache_data
def create_word_cloud(df):
    # 将所有评论文本合并成一个大字符串
    text = " ".join(review for review in df.cleaned_text)
    stopwords = load_stopwords("data/stopwords.txt")  # 加载停用词

    FONT_PATH = "data/ShanHaiJiGuSongKe-JianFan-2.ttf"

    # 创建词云对象，注意需要指定中文字体路径
    try:
        # 严格检查字体文件是否存在
        if not os.path.exists(FONT_PATH):
            raise FileNotFoundError(f"指定的词云字体文件 '{FONT_PATH}' 不存在。请替换为您自己的字体路径。")

        wordcloud = WordCloud(
            font_path=FONT_PATH,
            width=800,
            height=400,
            background_color='white',
            stopwords=stopwords

        ).generate(" ".join(jieba.cut(text)))

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except FileNotFoundError as e:
        st.warning(str(e))
        return None
    except Exception as e:
        st.error(f"生成词云时发生其他错误: {e}")
        return None



@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')


# ==============================================================================
# Streamlit 主界面
# ==============================================================================


# --- 页面配置 ---
st.set_page_config(
    page_title="智能商品推荐系统",
    page_icon="🛒",
    layout="wide"
)

# --- 主标题 ---
st.title("🛒 基于NLP的用户评论情感分析与商品推荐系统")
st.write("上传您的天猫评论数据，选择一个推荐策略，系统将为您生成商品推荐列表。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 控制面板")

    uploaded_file = st.file_uploader(
        "请上传您的评论数据 (CSV格式)",
        type=['csv']
    )

    strategy = st.selectbox(
        "请选择推荐策略",
        ('口碑优先', '加权综合分', '情感过滤+热度', '基线策略 (仅热度)')
    )

    top_k = st.slider("选择推荐结果的数量 (Top K)", min_value=5, max_value=20, value=10, step=1)

    run_button = st.button("🚀 开始分析与推荐")


# --- 主内容区 ---
if uploaded_file is not None:
    # 只要上传了文件，就先读取并缓存
    df_reviews = pd.read_csv(uploaded_file)

    st.info(f"✅ 文件上传成功，包含 {len(df_reviews)} 条评论。")

    if run_button:
        st.info(f"您选择的策略是： **{strategy}**，推荐数量为：**Top {top_k}**")

        with st.spinner('系统正在进行情感分析和推荐计算，请稍候...'):
            try:
                # <<< 修复: 接收 run_pipeline 返回的两个值 >>>
                recommendations, df_with_predictions = run_pipeline(df_reviews.copy(), strategy, top_k=top_k)

                st.success("🎉 推荐结果生成完毕！")

                # --- 结果展示区 ---
                st.subheader("🏆 商品推荐列表")

                if recommendations is not None and not recommendations.empty:
                    st.dataframe(recommendations)


                    csv_to_download = convert_df_to_csv(recommendations)
                    st.download_button(
                        label="📥 下载推荐结果为 CSV",
                        data=csv_to_download,
                        file_name=f"recommendations_{strategy}_top{top_k}.csv",
                        mime="text/csv",
                    )

                    # --- 可视化区 ---
                    st.subheader("📊 数据洞察与可视化")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**情感分布饼图**")
                        # 传递带有真实预测结果的DataFrame
                        pie_fig = create_sentiment_pie_chart(df_with_predictions)
                        st.pyplot(pie_fig)

                    with col2:
                        st.write("**评论高频词云**")
                        # 传递带有真实预测结果的DataFrame
                        wc_fig = create_word_cloud(df_with_predictions)
                        if wc_fig:
                            st.pyplot(wc_fig)

                else:
                    st.warning("🤔 根据您选择的策略和上传的数据，未能筛选出合适的推荐商品。")

            except FileNotFoundError as e:
                st.error(f"处理文件时发生错误: {e}")
                st.info(
                    "提示：如果您是首次运行，请确保您已经运行过一次 `recommender_system.py` 中的 `train_and_save_model()` 函数来生成模型文件。")
            except Exception as e:
                st.error(f"处理文件时发生错误: {e}")

elif run_button:
    st.warning("⚠️ 请先上传一个评论数据文件！")

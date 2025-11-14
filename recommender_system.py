# recommender_system.py

# -*- coding: utf-8 -*-

# 1. 核心库导入
import os
import re
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib  # 用于加载/保存模型


# ==============================================================================
# 模块一：文本预处理
# ==============================================================================

def clean_comment_text(text):
    """清洗单条评论文本。"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_stopwords(filename="data/stopwords.txt"):
    """从文件加载停用词列表。"""
    stopwords = set()
    if not os.path.exists(filename):
        print(f"警告：停用词文件 '{filename}' 不存在。")
        return stopwords
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
    except:
        with open(filename, 'r', encoding='gbk') as f:
            for line in f:
                stopwords.add(line.strip())
    return stopwords


def jieba_tokenizer(text):
    """用于TfidfVectorizer的Jieba分词器。"""
    return jieba.lcut(text)


# ==============================================================================
# 模块二：模型训练与加载 ( <<< 变化点 1: 新增一个函数用于训练和保存模型)
# ==============================================================================
# 我们只在需要时训练一次模型，然后保存它。App运行时直接加载，速度会快很多。

def train_and_save_model(data_file="data/smalldata_ws.csv", model_path="model/svm_model.pkl",
                         vectorizer_path="model/tfidf_vectorizer.pkl"):
    """训练情感分析模型并将其保存到磁盘。"""
    print("--- 开始训练并保存模型 ---")
    df = pd.read_csv(data_file, nrows=5000)
    df.rename(columns={'feedback': 'comment_text'}, inplace=True)
    df['cleaned_text'] = df['comment_text'].apply(clean_comment_text)

    # 简单的标签（实际项目中这里可以用SnowNLP或人工标注）
    df['sentiment_label'] = df['cleaned_text'].apply(lambda x: 'positive' if len(x) > 10 else 'negative')

    stopwords_set = load_stopwords()
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer, max_features=5000, stop_words=list(stopwords_set))
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment_label']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train, y_train)

    joblib.dump(svm_classifier, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"模型已保存到: {model_path}")
    print(f"Vectorizer已保存到: {vectorizer_path}")
    print("--- 模型训练完成 ---")


# ==============================================================================
# 模块三：核心推荐流程 ( <<< 变化点 2: 这就是app.py需要的函数！)
# ==============================================================================

# recommender_system.py

# ... (文件顶部的其他import和函数保持不变) ...

def run_pipeline(df_reviews, strategy, model_path="model/svm_model.pkl", vectorizer_path="model/tfidf_vectorizer.pkl"):
    """
    接收DataFrame和策略，返回推荐结果。
    这是整个系统的核心可调用函数。
    """
    print("\n--- [核心流程] 开始运行推荐管道 ---")

    # --- 步骤 1: 检查模型文件是否存在 ---
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("模型文件或Vectorizer文件未找到！请先运行 train_and_save_model() 进行训练。")

    # --- 步骤 2: 加载预训练的模型和Vectorizer ---
    print("[流程 1/5] 加载预训练模型...")
    svm_classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # =========================================================================
    # |           ↓↓↓ 核心修复在这里 ↓↓↓                                    |
    # =========================================================================

    # --- 步骤 3: 预处理上传的数据 ---
    print("[流程 2/5] 预处理上传的数据...")

    # 我们从截图中已经知道列名是 'item_id' 和 'feedback'
    # 我们将它们重命名为我们程序内部统一使用的 'ProductID' 和 'comment_text'
    df_reviews.rename(columns={
        'item_id': 'ProductID',
        'feedback': 'comment_text'
    }, inplace=True)

    # 增加一个严格的检查，确保重命名成功
    if 'ProductID' not in df_reviews.columns or 'comment_text' not in df_reviews.columns:
        raise ValueError("列重命名失败！请检查上传的CSV文件是否包含 'item_id' 和 'feedback' 列。")

    df_reviews['cleaned_text'] = df_reviews['comment_text'].apply(clean_comment_text)

    # =========================================================================
    # |           ↑↑↑ 核心修复完毕 ↑↑↑                                      |
    # =========================================================================

    # --- 步骤 4: 应用模型进行情感预测 ---
    print("[流程 3/5] 应用模型进行情感预测...")
    features = vectorizer.transform(df_reviews['cleaned_text'])
    df_reviews['predicted_sentiment'] = svm_classifier.predict(features)

    # --- 步骤 5: 构建商品级情感画像 ---
    print("[流程 4/5] 构建商品情感画像...")
    # 现在的df_reviews里一定有'ProductID'列了，所以这步不会再报错
    sentiment_counts = df_reviews.groupby(['ProductID', 'predicted_sentiment']).size().unstack(fill_value=0)
    review_counts = df_reviews.groupby('ProductID').size()

    product_profile_df = pd.DataFrame(review_counts, columns=['review_count'])
    product_profile_df = product_profile_df.join(sentiment_counts).fillna(0)

    for sentiment_col in ['positive', 'negative', 'neutral']:
        if sentiment_col not in product_profile_df.columns:
            product_profile_df[sentiment_col] = 0

    product_profile_df['positive_ratio'] = product_profile_df['positive'] / product_profile_df['review_count']
    product_profile_df['negative_ratio'] = product_profile_df['negative'] / product_profile_df['review_count']

    # --- 步骤 6: 根据策略执行推荐 ---
    # recommender_system.py -> run_pipeline 函数内

    # ... (前面的代码，直到“构建商品情感画像”结束) ...

    print(f"[流程 5/5] 执行推荐策略: {strategy}")

    # =========================================================================
    # |           ↓↓↓ 终极稳定版修复在这里 ↓↓↓                            |
    # =========================================================================

    # 先定义一个空的DataFrame，以防万一
    recommendations = pd.DataFrame()

    if strategy == '基线策略 (仅热度)':
        # 先排序，再把索引变回列
        recommendations = product_profile_df.sort_values(by='review_count', ascending=False).reset_index()

    elif strategy == '情感过滤+热度':
        filtered = product_profile_df[
            (product_profile_df['positive_ratio'] > 0.7) & (product_profile_df['negative_ratio'] < 0.1)]
        # 对过滤后的结果排序，再把索引变回列
        recommendations = filtered.sort_values(by='review_count', ascending=False).reset_index()

    elif strategy == '加权综合分':
        p_df_norm = product_profile_df.copy()
        if len(p_df_norm) > 1 and (p_df_norm['review_count'].max() != p_df_norm['review_count'].min()):
            p_df_norm['norm_review_count'] = (p_df_norm['review_count'] - p_df_norm['review_count'].min()) / (
                        p_df_norm['review_count'].max() - p_df_norm['review_count'].min())
        else:
            p_df_norm['norm_review_count'] = 1.0

        p_df_norm['score'] = 0.4 * p_df_norm['norm_review_count'] + 0.4 * p_df_norm['positive_ratio'] - 0.2 * p_df_norm[
            'negative_ratio']
        # 排序后，再把索引变回列
        recommendations = p_df_norm.sort_values(by='score', ascending=False).reset_index()

    elif strategy == '口碑优先':
        # 这是最关键的策略，我们确保它正确
        # 先排序，再把索引变回列
        recommendations = product_profile_df.sort_values(by=['positive_ratio', 'review_count'],
                                                         ascending=[False, False]).reset_index()

    # 检查一下我们是否真的得到了结果
    if recommendations.empty:
        print("警告：在执行排序后，推荐列表为空！")
        return pd.DataFrame()  # 明确返回一个空的DataFrame

    print("推荐列表生成成功，返回前10条。")
    return recommendations.head(10)

    # =========================================================================
    # |           ↑↑↑ 终极稳定版修复完毕 ↑↑↑                                |
    # =========================================================================



# ==============================================================================
# ( <<< 变化点 3: 保留这个部分，用于独立测试或首次训练模型)
# ==============================================================================
if __name__ == "__main__":
    # 首次运行时，取消下面这行代码的注释来训练并保存模型
    train_and_save_model()

    print("\n--- [独立测试] 测试 run_pipeline 函数 ---")
    try:
        test_df = pd.read_csv("data/smalldata_ws.csv", nrows=1000)
        test_df.rename(columns={'item_id': 'ProductID'}, inplace=True)

        # 测试口碑优先策略
        test_recs = run_pipeline(test_df, strategy='口碑优先')

        print("\n--- 测试成功！---")
        print("口碑优先推荐结果预览:")
        print(test_recs)
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        print("提示：请先运行 train_and_save_model() 来生成模型文件。")
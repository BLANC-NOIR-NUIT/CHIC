import os
import gradio as gr
import json
from openai import OpenAI
import base64
import tempfile
import shutil
from databricks.vector_search.client import VectorSearchClient

# テンポラリディレクトリの作成
TEMP_DIR = './temp'
os.makedirs(TEMP_DIR, exist_ok=True)

# OpenAI クライアントの初期化（APIキーは後で設定）
client = OpenAI(
    api_key="OPENAI_API_KEY" ,  
    base_url="OPENAI_ENDPOINTS" 
)

# 環境変数をセット
os.environ['DATABRICKS'] = 'DATABRICKS_ENVIRON'
os.environ['DATABRICKS'] = 'DATABRICKS_ENVIRON'

def encode_image(image_path):
    """画像をBase64エンコードする関数"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_similar_coordinations(coordination_text):
    """ベクトル検索で類似のコーディネーションを見つける関数"""
    try:
        # Vector Search Clientの初期化
        vsc = VectorSearchClient(disable_notice=True)
        
        # Vector Searchインデックスに接続
        vs_index = vsc.get_index(
            endpoint_name="vs_endpoint",
            index_name="dev.haruna_osaki.fashion_documentation_vs_index"
        )
        
        # 類似性検索の実行
        results = vs_index.similarity_search(
            query_text=coordination_text,
            columns=["ID", "Detail", "Category", "Color", "Pass"],
            num_results=3,  # 上位3件の結果を取得
            filters={}  # 必要に応じてフィルタを追加
        )
        
        # 結果のフィルタリングと処理
        returned_docs = []
        docs = results.get('result', {}).get('data_array', [])
        
        for doc in docs:
            # 関連性スコアでフィルタリング（0.5以上）
            if doc[-1] > 0.5:
                returned_docs.append({
                    "id": doc[0], 
                    "detail": doc[1], 
                    "category": doc[2], 
                    "color": doc[3], 
                    "image_path": doc[4]
                })
        
        return returned_docs
    
    except Exception as e:
        print(f"Vector search error: {e}")
        return []

def diagnose_body_type(image):
    """骨格診断を行う関数"""
    if image is None:
        return "画像が提供されていません"
    
    base64_image = encode_image(image)
    
    query = """体型の特徴を以下の3つのタイプから1つ選んで診断してください：
    1. ストレートタイプ
    2. ナチュラルタイプ
    3. ウェーブタイプ

    診断結果は次の形式で返してください：
    タイプ: [診断されたタイプ名]
    特徴: [3〜4行の簡潔な特徴説明]
    おすすめのスタイル: [そのタイプに最適な服装や着こなしのヒント]"""
    
    response = client.chat.completions.create(
        model="aoai-gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは専門的な骨格診断ができるファッションコンサルタントです。"},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=300,
        temperature=0.2
    )
    
    return response.choices[0].message.content

def diagnose_personal_color(image):
    """パーソナルカラー診断を行う関数"""
    if image is None:
        return "画像が提供されていません"
    
    base64_image = encode_image(image)
    
    query = """肌の色や特徴から以下の4つのシーズンタイプから1つ選んでパーソナルカラーを診断してください：
    1. スプリング
    2. サマー
    3. オータム
    4. ウィンター

    診断結果は次の形式で返してください：
    タイプ: [診断されたシーズンタイプ名]
    特徴: [肌、髪、目の色の特徴]
    似合う色: [そのタイプに最適な色の例3〜4色]
    避けるべき色: [そのタイプに合わない色の例3〜4色]"""
    
    response = client.chat.completions.create(
        model="aoai-gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは専門的なパーソナルカラー診断ができるスタイリストです。"},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=300,
        temperature=0.2
    )
    
    return response.choices[0].message.content

def generate_coordination(body_type_result, color_type_result):
    """コーディネート提案を生成する関数"""
    query = f"""以下の診断結果に基づいて、最適なコーディネートを提案してください：
    
    {body_type_result}
    
    {color_type_result}

    提案は以下の形式で返してください：
    トップス: [具体的な服のアイテム]
    ボトムス: [具体的な服のアイテム]
    アウター/羽織: [必要に応じて]
    小物: [アクセサリーやバッグなど]
    """
    
    response = client.chat.completions.create(
        model="aoai-gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは最先端のファッションスタイリストです。"},
            {"role": "user", "content": query}
        ],
        max_tokens=500,
        temperature=0.3
    )
    
    return response.choices[0].message.content

def main_app():
    """メインのGradioアプリケーション"""
    with gr.Blocks() as demo:
        gr.Markdown("# パーソナルスタイリストAI")
        
        # 画像アップロード
        upload_input = gr.Image(type="filepath", label="あなたの写真をアップロード")
        
        with gr.Row():
            combined_diagnosis_btn = gr.Button("診断開始")
        
        # 診断結果出力
        body_diagnosis_output = gr.Textbox(label="骨格診断結果")
        color_diagnosis_output = gr.Textbox(label="パーソナルカラー診断結果")
        
        # コーディネート提案ボタン
        coordination_btn = gr.Button("コーディネート提案")
        coordination_output = gr.Textbox(label="コーディネート提案")

        # 類似コーディネーション表示用のギャラリー
        similar_coordinations_gallery = gr.Gallery(
            label="おすすめのコーディネーション", 
            columns=3, 
            rows=1, 
            object_fit="contain", 
            height="auto"
        )
        
        # 診断ボタンのイベントハンドラ
        def combined_diagnosis(image):
            """骨格診断とパーソナルカラー診断を同時に行う関数"""
            body_result = diagnose_body_type(image)
            color_result = diagnose_personal_color(image)
            return body_result, color_result
        
        # イベントハンドラの定義
        combined_diagnosis_btn.click(
            fn=combined_diagnosis, 
            inputs=upload_input, 
            outputs=[body_diagnosis_output, color_diagnosis_output]
        )
        
        # コーディネート提案と類似コーディネーション検索のイベントハンドラ
        def generate_and_find_coordinations(body_type_result, color_type_result):
            # コーディネート提案を生成
            coordination_text = generate_coordination(body_type_result, color_type_result)
            
            # 類似のコーディネーションを検索
            similar_coordinations = find_similar_coordinations(coordination_text)
            
            # 画像パスのリストを作成
            similar_images = []
            for doc in similar_coordinations:
                try:
                    # テンポラリディレクトリに画像をコピー
                    temp_image_path = os.path.join(TEMP_DIR, doc['image_path'])
                    os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                    
                    # Databricksから画像をダウンロード（この部分は実際の実装に依存）
                    # 例: 
                    # download_image_from_databricks(doc['image_path'], temp_image_path)
                    
                    similar_images.append(temp_image_path)
                except Exception as e:
                    print(f"画像の処理中にエラーが発生しました: {e}")
            
            return coordination_text, similar_images
        
        # コーディネート提案と類似画像検索のイベント
        coordination_btn.click(
            fn=generate_and_find_coordinations,
            inputs=[body_diagnosis_output, color_diagnosis_output],
            outputs=[coordination_output, similar_coordinations_gallery]
        )
    
    return demo

# アプリケーションの起動
if __name__ == "__main__":
    demo = main_app()
    demo.launch()
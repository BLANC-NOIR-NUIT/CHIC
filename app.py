import os
import gradio as gr
import json
from openai import OpenAI
import base64
import tempfile
import shutil
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import requests
from PIL import Image
import io
import time
from pathlib import Path

# テンポラリディレクトリの作成
TEMP_DIR = './temp'
os.makedirs(TEMP_DIR, exist_ok=True)

# OpenAI クライアントの初期化（APIキーは後で設定）
client = OpenAI(
    api_key="OPENAI_API_KEY" ,  
    base_url="OPENAI_ENDPOINTS" 
)

# グローバル変数の定義
DATABRICKS_HOST = '＊＊＊＊＊＊＊＊＊'
DATABRICKS_TOKEN = '＊＊＊＊＊＊＊＊＊'

# 環境変数から他の認証関連の変数を削除
if 'DATABRICKS_CLIENT_ID' in os.environ:
    del os.environ['DATABRICKS_CLIENT_ID']
if 'DATABRICKS_CLIENT_SECRET' in os.environ:
    del os.environ['DATABRICKS_CLIENT_SECRET']

# 環境変数をセット
os.environ['DATABRICKS'] = 'DATABRICKS_ENVIRON'
os.environ['DATABRICKS'] = 'DATABRICKS_ENVIRON'

def get_image_from_volumes(image_path):
    """Databricks Volumesから画像を取得する関数"""
    try:
        # WorkspaceClientの初期化
        w = WorkspaceClient()
        
        # Volumesパスを処理
        dbfs_path = image_path.replace('/Volumes/', '/dbfs/Volumes/')
        
        # 画像データの読み込み
        with open(dbfs_path, 'rb') as f:
            image_data = f.read()
        
        # 画像データをPIL Imageオブジェクトとして返す
        return Image.open(io.BytesIO(image_data))
            
    except Exception as e:
        print(f"画像の取得中にエラーが発生しました: {e}")
        return None

def encode_image(image_path):
    """画像をBase64エンコードする関数"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

import time

def find_similar_coordinations(coordination_text, max_retries=5, wait_time=10):
    """ベクトル検索で類似のコーディネーションを見つける関数（リトライ対応）"""
    for attempt in range(max_retries):
        try:
            vsc = VectorSearchClient(disable_notice=True)
            vs_index = vsc.get_index(
                endpoint_name="vs_endpoint",
                index_name="dev.haruna_osaki.fashion_documentation_vs_index"
            )

            results = vs_index.similarity_search(
                query_text=coordination_text,
                columns=["ID", "Detail", "Category", "Color", "Pass"],
                num_results=1,
                filters={}
            )

            returned_docs = []
            docs = results.get('result', {}).get('data_array', [])

            for doc in docs:
                if doc[-1] > 0.5:
                    image_path = f"/Volumes/dev/haruna_osaki/images/{doc[4]}"
                    returned_docs.append({
                        "id": doc[0],
                        "detail": doc[1],
                        "category": doc[2],
                        "color": doc[3],
                        "image_path": image_path
                    })

            return returned_docs

        except requests.exceptions.RequestException as e:
            print(f"リクエストエラー: {e}")
        except Exception as e:
            if "CANCELLED" in str(e) and attempt < max_retries - 1:
                print(f"モデルがスケールアップ中… {wait_time}秒待機 (試行 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Vector search error: {e}")
                return []

def generate_and_find_coordinations(body_type_result, color_type_result):
    """コーディネート提案と類似コーディネーション検索を行う関数"""
    try:
        # コーディネート提案を生成
        coordination_text = generate_coordination(body_type_result, color_type_result)
        print(f"Generated coordination text: {coordination_text}")  # デバッグ用
        
        # 類似のコーディネーションを検索
        similar_coordinations = find_similar_coordinations(coordination_text)
        print(f"Found similar coordinations: {similar_coordinations}")  # デバッグ用
        
        # 画像パスのリストを作成
        similar_images = []
        for doc in similar_coordinations:
            if 'image_path' in doc and doc['image_path']:
                similar_images.append(doc['image_path'])
        
        print(f"Final image paths: {similar_images}")  # デバッグ用
        return coordination_text, similar_images
    
    except Exception as e:
        print(f"コーディネーション生成中にエラーが発生しました: {e}")
        return "エラーが発生しました", []
    


def diagnose_body_type(image):
    """骨格診断を行う関数"""
    if image is None:
        return "画像が提供されていません"
    
    base64_image = encode_image(image)
    
    query = """あなたはファッションコンサルタントであり、骨格診断を正確に行う必要があります。以下のステップに従って診断してください：

1. 画像を解析して、以下の3つのタイプのいずれに該当するか確認してください。
    - ストレートタイプ: 上半身に厚みがあり、直線的なライン。
    - ナチュラルタイプ: 骨の存在感があり、フレームがしっかりしている。
    - ウェーブタイプ: ソフトな体型で、曲線的なライン。

2. 判断は以下の基準に基づいてください（例: 骨の形状、脂肪分布、体のライン）。
    - 特徴を詳細に記述し、タイプを選定してください。

3. 一貫性を保つために、これらの判断基準に厳密に従ってください。

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
    
    query = """あなたは専門的なパーソナルカラー診断を行う必要があります。以下の手順に従い診断を実施してください：

1. 画像を解析し、以下の4つのタイプのいずれかを選定してください：
    - スプリング: 明るい暖色、黄味のある肌。
    - サマー: 柔らかい寒色、青味のある肌。
    - オータム: 深い暖色、黄味がかった肌。
    - ウィンター: 鮮やかな寒色、青白い肌。

2. 判断基準として、肌の色、髪、目の色の特徴を観察してください。
3. 必ず基準に基づいて診断し、ランダム性を排除してください。

結果は次の形式で返してください：
タイプ: [診断されたシーズンタイプ名]
特徴: [肌、髪、目の色の特徴]
似合う色: [そのタイプに最適な色の例3〜4色]
避けるべき色: [そのタイプに合わない色の例3〜4色]
"""
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
    with gr.Blocks(css="""
        /* メインコンテンツのスペーシング */
        .main-content {
            margin-bottom: 20px;  /* 余分なスペースを減らす */
        }

        /* ツールチップコンテナ */
        .terms-container {
            margin-top: 20px;  /* 上部のスペースを減らす */
            margin-bottom: 20px;  /* 下部のスペースを減らす */
            position: relative;
        }

        /* カスタムツールチップ */
        .custom-tooltip {
            display: inline-block;
            border-bottom: 1px dotted #000;
            position: relative;
            cursor: pointer;
            margin: 0 4px;
        }

        .custom-tooltip .tooltip-text {
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 12px;
            width: 250px;
            
            /* ポジショニング調整 */
            position: fixed;  /* fixedに変更してページ全体に対して配置 */
            z-index: 10000;  /* より高いz-indexを設定 */
            transform: translateX(-50%);
            
            /* ツールチップの位置を動的に計算 */
            left: 50%;
            bottom: auto;  /* 自動位置調整 */
            
            /* 表示制御 */
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.3s, visibility 0.3s;
            
            /* テキスト設定 */
            font-size: 14px;
            line-height: 1.5;
            white-space: normal;
            
            /* 背景をより目立たせる */
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        /* ホバー時の表示 */
        .custom-tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
            /* ホバー時にツールチップを要素の上に配置 */
            transform: translate(-50%, -100%);
            margin-bottom: 15px;  /* 矢印のためのスペース */
        }

        /* 矢印 */
        .custom-tooltip .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
    """) as demo:
        # マークダウンをdivで囲む
        gr.HTML("""
        <div class="main-content">
            <h1>CHIC</h1>
            <p>トレンドと科学で導く、あなたのパーソナルスタイル</p>
            
            <h3>使い方</h3>
            <ol>
                <li>全身写真をアップロードしてください</li>
                <li>「診断開始」を押すと、あなたの骨格タイプとベストカラーを分析します</li>
                <li>「コーディネート提案」で、あなたに似合うスタイルをご提案します</li>
            </ol>
        </div>
        """)
        
        # 専門用語の説明
        gr.HTML("""
        <div class="terms-container">
            <span class="custom-tooltip">
                骨格診断
                <span class="tooltip-text">体の骨格や筋肉のつき方から、あなたに似合うファッションスタイルを見つける診断方法です</span>
            </span>
            ・
            <span class="custom-tooltip">
                パーソナルカラー診断
                <span class="tooltip-text">肌の色や髪の色から、あなたに最も似合う色を見つける診断方法です</span>
            </span>
            について詳しく知りたい方は各用語にカーソルを合わせてください。
        </div>
        """)
        
        # 画像アップロード
        with gr.Column():
            gr.Markdown("""
            📸 **写真のポイント**
            - 全身が写っている写真を使用してください
            - なるべく体のラインが分かりやすい服装で
            - 自然光の下で撮影すると、より正確な診断が可能です
            """)
        upload_input = gr.Image(
            type="filepath", 
            label="あなたの写真をアップロード",
            height=700,  
            width=1000,   
            )
        
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
            #value = similar_images,
            label="おすすめのコーディネーション", 
            show_label=True,
            elem_id="coordination-gallery",
            columns=2,
            object_fit="contain",
            height="auto"
        )
        
        # 診断ボタンのイベントハンドラ
        def combined_diagnosis(image):
            """骨格診断とパーソナルカラー診断を同時に行う関数"""
            body_result = diagnose_body_type(image)
            color_result = diagnose_personal_color(image)
            return body_result, color_result
        
        def process_coordinations(body_result, color_result):
            coordination_text, similar_coordinations = generate_and_find_coordinations(body_result, color_result)
            
            # 画像パスのリストを作成
            image_paths = []
            for doc in similar_coordinations:
                if 'image_path' in doc:
                    image_paths.append(doc['image_path'])
            
            return coordination_text, image_paths
            
        
        # イベントハンドラの定義
        combined_diagnosis_btn.click(
            fn=combined_diagnosis, 
            inputs=upload_input, 
            outputs=[body_diagnosis_output, color_diagnosis_output]
        )
        
        # コーディネート提案と類似画像検索のイベント
        coordination_btn.click(
            fn=process_coordinations,
            inputs=[body_diagnosis_output, color_diagnosis_output],
            outputs=[coordination_output, similar_coordinations_gallery]
        )
    
    return demo

# アプリケーションの起動
if __name__ == "__main__":
    demo = main_app()
    demo.launch()
    demo.launch()
